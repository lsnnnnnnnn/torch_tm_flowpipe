"""Support-specialized tape execution and the sparse advance() path (M10).

The dense executors (ode_compiler.exec_point/exec_valid) and stage pipeline
(flowpipe.advance) run every polynomial op over the full monomial basis; this
module runs the SAME algorithms over union supports (support.py):

  * `specialize()` walks a compiled tape ONCE per (tape, truncation order,
    per-variable input supports) doing pure CPU support algebra — for every
    instruction it precomputes the output support and the sparse pair/union
    tables the executors need. Specializations are cached on the SparseEngine
    (value-keyed: supports are canonical), so steady-state advances do no CPU
    table work at all (measured supports stabilize across steps).
  * `exec_point_s` / `exec_valid_s` interpret the specialized tape on
    support-aligned tensors; the elementary series run the GENERIC bodies of
    elementary.py (bitwise-pinned to the dense implementations by
    tests/unit/test_sparse.py) through sparse contexts driven by the
    specialization's per-iteration chains.
  * `compose_s` is the monomial-image composition building ONLY the images
    the substituted polynomial actually references (plus DAG ancestors) on
    per-level union supports.
  * `advance_sparse` mirrors flowpipe.advance stage-for-stage (same Flow*
    line citations apply) on a SparseState; the refinement loop and the
    symbolic-remainder machinery are shared with the dense path unchanged
    (both are representation-free).

Soundness/parity: every sparse op touches a SUPERSET of the nonzero slots of
its dense counterpart, so tier-P results agree bitwise on CPU and to ulps on
CUDA, and tier-I results are equal-or-tighter (support.py module docstring —
the zero-removal argument and its backend caveat).
Support growth is re-grounded by measurement: per Picard sweep and per advance
the ACTUAL nonzero support is extracted (one small host sync each; the data
shrinkage from cutoffs is what structural analysis cannot see).

Shapes: B batch, n state dim, S* support sizes; coefficient tensors
[B, S] / [B, S, 2] per tape slot, [B, n, S] / [B, n, S, 2] per TM vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import elementary as elem
from . import interval as iv
from . import injective_index as ii
from . import polynomial as poly
from . import support as sp
from .composition import CompositionSchedule
from .config import Settings
from .glue import glue_cache
from .graphing import graph_cache
from .monomials import MonomialTables
from .ode_compiler import CompiledODE, exec_replay
from .polynomial import StepTables
from .rounding import dot_error_bound, next_up
from .symbolic_remainder import SymbolicRemainder, preconditioning_scale, propagate

# ---------------------------------------------------------------------------
# Engine-attached caches (extend SparseEngine without touching its module)
# ---------------------------------------------------------------------------


def _eng_cache(eng: sp.SparseEngine, name: str) -> dict:
    """Lazily attach a named cache dict to a SparseEngine instance."""
    c = getattr(eng, name, None)
    if c is None:
        c = {}
        setattr(eng, name, c)
    return c


# Code objects may be collected while engine caches outlive them. A raw
# id(code) can then designate another program; never recycle its cache serial.
import itertools
import weakref

_code_serials: dict = {}
_code_serial_counter = itertools.count()


def _code_serial(code: CompiledODE) -> int:
    """Stable serial for the lifetime of this exact compiled program."""
    key = id(code)
    entry = _code_serials.get(key)
    if entry is not None and entry[0]() is code:
        return entry[1]
    serial = next(_code_serial_counter)
    _code_serials[key] = (weakref.ref(code, lambda ref: _code_serials.pop(key, None)), serial)
    return serial


# ---------------------------------------------------------------------------
# Specialized instructions
# ---------------------------------------------------------------------------


@dataclass
class SInstr:
    """One support-specialized tape instruction.

    Populated fields depend on op (unused stay None):
        sup_out: support of the dst slot AFTER the op.
        gather: [S_var] device idx — var leaf's positions inside the x layout.
        keep_len: local length of the degree <= k prefix (var/mul truncation).
        sup_in: input support (var: the var's measured support).
        pair: PairBind for a generic mul (a x b).
        sup_a / sup_b: operand supports for a generic mul.
        emb: (sup_u, emb_a idx, emb_b idx) union bind for add/sub.
        chain: per-iteration [(PairBind, keep_len, sup_in_a)] for pow/series
            multiply sequences (execution order).
        acc: per-iteration union binds for the sin/cos accumulator.
        pair2: div's final multiply bind (rec_result x numerator).
        sup_f: the series' fixed-argument support.
    """

    op: str
    dst: int
    a: int = -1
    b: int = -1
    var: int = -1
    const: float = 0.0
    crem_lo: float = 0.0
    crem_hi: float = 0.0
    power: int = 0
    cache_base: int = -1
    strict_slot: int = -1
    sup_out: sp.Support | None = None
    gather: torch.Tensor | None = None
    keep_len: int = 0
    sup_in: sp.Support | None = None
    pair: sp.PairBind | None = None
    sup_a: sp.Support | None = None
    sup_b: sp.Support | None = None
    emb: tuple | None = None
    chain: list = field(default_factory=list)
    acc: list = field(default_factory=list)
    pair2: tuple | None = None
    sup_f: sp.Support | None = None


@dataclass
class SpecTape:
    """A tape specialized for (k, per-var supports): instruction list plus the
    output-union embedding (per-component embed idx into sup_f_union)."""

    k: int
    instrs: list
    out_slots: list
    out_sups: list
    sup_out_union: sp.Support
    out_embeds: list  # per component: device idx into sup_out_union


def _sup0(eng: sp.SparseEngine) -> sp.Support:
    """The {constant} support of the engine's basis family."""
    return sp.make_support(eng.tables.n, eng.tables.k, False, (0,))


def _kept(pb: sp.PairBind, k: int) -> tuple[sp.Support, int]:
    """Degree <= k prefix of a pair product's output support."""
    keep = pb.sup_out.keep_len(k)
    return sp.make_support(pb.sup_out.n, pb.sup_out.k, pb.sup_out.spatial,
                           pb.sup_out.ids[:keep]), keep


def specialize(
    code: CompiledODE,
    var_sups: tuple,
    k: int,
    x_sup: sp.Support,
    eng: sp.SparseEngine,
    *, validated_full_leaves: bool = False, validated_direct_leaves: bool = False, validated_linear_leaves: bool = False,
) -> SpecTape:
    """Support-specialize a compiled tape (cached per engine).

    var_sups: per-variable measured supports of the tape input x (each a
    subset of x_sup, the union layout x is stored on). k: the truncation /
    series order this execution runs at (Picard sweep k_i or order-1).
    Pure CPU set algebra + table builds; every product/union bind is cached
    at the support level too, so respecialization after a support change only
    pays for the parts that actually changed.
    """
    cache = _eng_cache(eng, "_spec")
    if validated_full_leaves and (
        k != eng.tables.k - 1
        or any(ins.op not in {"var", "const", "neg", "add", "sub", "mul", "pow"} for ins in code.instrs)
    ):
        raise ValueError("deferred validation supports polynomial RHS at order-1 only")
    # A validated plan must never alias the final point Picard plan at this k.
    if (validated_direct_leaves or validated_linear_leaves) and k != eng.tables.k - 1:
        raise ValueError("direct leaf validation requires order-1")
    key = (_code_serial(code), k, x_sup, var_sups, validated_full_leaves, validated_direct_leaves, validated_linear_leaves)
    hit = cache.get(key)
    if hit is not None:
        return hit

    # Only output VAR slots without internal consumers may retain full leaves.
    # All nonlinear uses retain the original truncated plan and strict tails.
    consumed = {v for ins in code.instrs for v in (ins.a, ins.b) if v >= 0}
    direct_outputs = set(code.out_slots) - consumed if validated_direct_leaves else set()
    if validated_linear_leaves:
        # Retain a leaf only when every consumer path to an output is linear.
        # Inspect the parent's operation: sin/mul/etc. always block the path.
        consumers = {}
        for ins in code.instrs:
            for slot in (ins.a, ins.b):
                if slot >= 0:
                    consumers.setdefault(slot, []).append(ins)
        roots = set(code.out_slots)
        for ins in reversed(code.instrs):
            parents = consumers.get(ins.dst, ())
            if (ins.dst in roots or parents) and all(
                p.op in {"neg", "add", "sub"} and p.dst in direct_outputs
                for p in parents
            ):
                direct_outputs.add(ins.dst)

    sup_zero = _sup0(eng)
    sups: dict[int, sp.Support] = {}
    out: list[SInstr] = []

    def series_chain(sup_f: sp.Support, iters: int, seed: sp.Support) -> tuple:
        """(chain, final_sup): res_{j+1} = kept_k(res_j * F) recurrence."""
        chain = []
        cur = seed
        for _ in range(iters):
            pb = eng.pair(cur, sup_f)
            kept_sup, keep = _kept(pb, k)
            chain.append((pb, keep, cur))
            cur = kept_sup
        return chain, cur

    for ins in code.instrs:
        si = SInstr(
            op=ins.op, dst=ins.dst, a=ins.a, b=ins.b, var=ins.var,
            const=ins.const, crem_lo=ins.crem_lo, crem_hi=ins.crem_hi,
            power=ins.power, cache_base=ins.cache_base,
            strict_slot=ins.strict_slot,
        )
        if ins.op == "var":
            sv = var_sups[ins.var]
            # Retain the full order-degree input only in validated evaluation.
            # Products still truncate to k; discarded terms remain in strict
            # tails. A linear RHS can reach degree order here and order+1 on
            # integration; _g_validpost retains its full polynomial difference.
            keep = sv.keep_len(eng.tables.k if validated_full_leaves or ins.dst in direct_outputs else k)
            si.sup_in = sv
            si.keep_len = keep
            si.gather = eng.embed_idx(sv, x_sup)
            si.sup_out = sp.make_support(sv.n, sv.k, False, sv.ids[:keep])
        elif ins.op == "const":
            si.sup_out = sup_zero
        elif ins.op == "neg":
            si.sup_out = sups[ins.a]
        elif ins.op in ("add", "sub"):
            su, ea, eb = eng.union(sups[ins.a], sups[ins.b])
            si.emb = (su, ea, eb)
            si.sup_out = su
        elif ins.op == "mul":
            pb = eng.pair(sups[ins.a], sups[ins.b])
            si.pair = pb
            si.sup_a, si.sup_b = sups[ins.a], sups[ins.b]
            si.sup_out, si.keep_len = _kept(pb, k)
        elif ins.op == "pow":
            # Mirror the square-and-multiply structure: entries tagged by
            # which register they update ('r' result, 't' temp), in the exact
            # execution (= cache cursor) order.
            res_s = tmp_s = sups[ins.a]
            chain = []
            i = ins.power - 1
            while i > 0:
                if i & 1:
                    pb = eng.pair(res_s, tmp_s)
                    ks, keep = _kept(pb, k)
                    chain.append(("r", pb, keep, res_s, tmp_s))
                    res_s = ks
                i >>= 1
                if i > 0:
                    pb = eng.pair(tmp_s, tmp_s)
                    ks, keep = _kept(pb, k)
                    chain.append(("t", pb, keep, tmp_s))
                    tmp_s = ks
            si.chain = chain
            si.sup_out = res_s if ins.power > 0 else sup_zero
        elif ins.op in ("exp", "rec", "sin", "cos", "log", "sqrt", "div"):
            sup_f = sups[ins.a] if ins.op != "div" else sups[ins.b]
            si.sup_f = sup_f
            if ins.op in ("exp", "rec", "div"):
                si.chain, res_s = series_chain(sup_f, k, sup_zero)
                if ins.op == "div":
                    pb2 = eng.pair(sups[ins.a], res_s)
                    ks, keep = _kept(pb2, k)
                    si.pair2 = (pb2, keep, sups[ins.a], res_s)
                    si.sup_a = sups[ins.a]
                    res_s = ks
                si.sup_out = res_s
            elif ins.op in ("sin", "cos"):
                # pow chain from {0}; the accumulator unions res with pow.
                si.chain, _ = series_chain(sup_f, k, sup_zero)
                res_s = sup_zero
                for pb, keep, _sup_in in si.chain:
                    pow_s, _ = _kept(pb, k)
                    su, er, ep = eng.union(res_s, pow_s)
                    si.acc.append((su, er, ep))
                    res_s = su
                si.sup_out = res_s
            else:  # log / sqrt: seeded at F, k-1 (log) / k-1 (sqrt) multiplies
                iters = max(k - 1, 0)
                si.chain, res_s = series_chain(sup_f, iters, sup_f)
                si.sup_out = res_s
        else:  # pragma: no cover  # compile_ode emits only known ops
            raise AssertionError(f"bad op {ins.op}")
        sups[ins.dst] = si.sup_out
        # POW writes a second register slot; its support is tracked in-chain.
        out.append(si)

    out_sups = [sups[s] for s in code.out_slots]
    u = out_sups[0]
    for s2 in out_sups[1:]:
        u, _, _ = eng.union(u, s2)
    spec = SpecTape(
        k=k,
        instrs=out,
        out_slots=list(code.out_slots),
        out_sups=out_sups,
        sup_out_union=u,
        out_embeds=[eng.embed_idx(s2, u) for s2 in out_sups],
    )
    cache[key] = spec
    return spec


# ---------------------------------------------------------------------------
# Sparse series contexts (consumed by elementary.*_g generic bodies)
# ---------------------------------------------------------------------------


def tm_mul_valid_s(
    a_c: torch.Tensor,
    a_r: torch.Tensor,
    b_c: torch.Tensor,
    b_r: torch.Tensor,
    pb: sp.PairBind,
    keep_len: int,
    sup_a: sp.Support,
    eng: sp.SparseEngine,
    cutoff_threshold: float,
    cache: torch.Tensor,
    cache_at: int,
    range_b: torch.Tensor,
    preserve_intervals: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse mul_insert_ctrunc_normal (mirrors elementary.tm_mul_valid).

    a_c [B, Sa, 2], b_c [B, Sb, 2] -> (kept [B, keep_len, 2], rem [B, 2]);
    cache triple (rangeP1, rangeP2, trunc+round) written at cache_at.
    """
    prod = sp.mul_iv_s(a_c, b_c, pb)  # [B, S2, 2]
    range_p1 = sp.range_normal_iv_s(a_c, eng, sup_a)  # [B, 2]

    rem = iv.mul(a_r, b_r)
    rem = iv.add(rem, iv.mul(range_b, a_r))
    rem = iv.add(rem, iv.mul(range_p1, b_r))

    factors = eng.factor(pb.sup_out)  # [S2, 2]
    kept = prod[:, :keep_len]
    if keep_len < pb.sup_out.size:
        dropped = prod[:, keep_len:]  # [B, S2-K, 2]
        tail = iv.sum(iv.mul(dropped, factors[keep_len:]), dim=-1)  # [B, 2]
    else:
        tail = torch.zeros_like(rem)
    kept, round_rng = _cutoff_interval_rows(kept, factors[:keep_len], cutoff_threshold, preserve_intervals=preserve_intervals)
    trunc_round = iv.add(tail, round_rng)
    rem = iv.add(rem, trunc_round)

    cache[:, cache_at] = range_p1
    cache[:, cache_at + 1] = range_b
    cache[:, cache_at + 2] = trunc_round
    return kept, rem


def _cutoff_interval_rows(
    coeffs_iv: torch.Tensor, factors: torch.Tensor, cutoff_threshold: float,
    max_width: float = 1e-12,
    preserve_intervals: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cutoff_normal_interval with explicit factor rows (support-agnostic)."""
    if preserve_intervals:
        # Wide coefficient uncertainty carries time dependence in the weighted
        # validator. Keep it; discarded coefficients retain their complete range.
        small = ((coeffs_iv[..., 0] >= -cutoff_threshold)
                 & (coeffs_iv[..., 1] <= cutoff_threshold))
        dropped = torch.where(small.unsqueeze(-1), coeffs_iv, torch.zeros_like(coeffs_iv))
        kept = torch.where(small.unsqueeze(-1), torch.zeros_like(coeffs_iv), coeffs_iv)
        return kept, iv.sum(iv.mul(dropped, factors), dim=-1)
    from .rounding import next_down

    lo_c, hi_c = coeffs_iv[..., 0], coeffs_iv[..., 1]
    w = next_up(hi_c - lo_c)
    mid_rn = (lo_c + hi_c) * 0.5
    mid_iv = torch.stack((next_down(mid_rn), next_up(mid_rn)), dim=-1)
    wide = w >= max_width
    small = (~wide) & (mid_iv[..., 0] >= -cutoff_threshold) & (
        mid_iv[..., 1] <= cutoff_threshold
    )
    radius = iv.sub(coeffs_iv, mid_iv)
    zeros2 = torch.zeros_like(coeffs_iv)
    to_rem = torch.where(wide.unsqueeze(-1), radius, zeros2)
    to_rem = torch.where(small.unsqueeze(-1), coeffs_iv, to_rem)
    prod = iv.mul(to_rem, factors)
    dropped_range = iv.sum(prod, dim=-1)
    kept = torch.where(wide.unsqueeze(-1), mid_iv, coeffs_iv)
    kept = torch.where(small.unsqueeze(-1), zeros2, kept)
    return kept, dropped_range


class SparseValidCtx:
    """Valid-tier series context over a specialization chain (elementary.py
    protocol). Consumes one chain entry per mul_valid call."""

    def __init__(self, eng, chain, acc, tm_f, f_rem, sup_f, cutoff, preserve_intervals=False):
        self.eng = eng
        self.chain = chain
        self.acc_binds = acc
        self.tm_f = tm_f  # [B, Sf, 2]
        self.f_rem = f_rem  # [B, 2]
        self.sup_f = sup_f
        self.cutoff = cutoff
        self.preserve_intervals = preserve_intervals
        # tm_f may be None at construction (exec_valid_s builds the ctx shell
        # first; the make_ctx closure binds F and computes the range).
        self._range_f = None if tm_f is None else sp.range_normal_iv_s(tm_f, eng, sup_f)
        self._i = 0
        self._j = 0

    def range_f(self):
        return self._range_f

    def mul_valid(self, a_c, a_r, cache, at):
        pb, keep, sup_a = self.chain[self._i]
        self._i += 1
        return tm_mul_valid_s(
            a_c, a_r, self.tm_f, self.f_rem, pb, keep, sup_a, self.eng,
            self.cutoff, cache, at, self._range_f,
            preserve_intervals=self.preserve_intervals,
        )

    def acc_add(self, res_c, term_c):
        su, er, ep = self.acc_binds[self._j]
        self._j += 1
        rv = torch.zeros(res_c.shape[0], su.size, 2, dtype=res_c.dtype, device=res_c.device)
        tv = torch.zeros_like(rv)
        rv.index_copy_(-2, er, res_c)
        tv.index_copy_(-2, ep, term_c)
        return iv.add(rv, tv)

    def fresh_one(self):
        out = torch.zeros(self.tm_f.shape[0], 1, 2, dtype=self.tm_f.dtype,
                          device=self.tm_f.device)
        out[:, 0] = 1.0
        return out

    def fresh_zero(self):
        return torch.zeros(self.tm_f.shape[0], 1, 2, dtype=self.tm_f.dtype,
                           device=self.tm_f.device)

    def seed_f(self):
        return self.tm_f.clone()


class SparsePointCtx:
    """Point-tier series context over a specialization chain."""

    def __init__(self, eng, chain, acc, f, cutoff):
        self.eng = eng
        self.chain = chain
        self.acc_binds = acc
        self.f = f  # [B, Sf]
        self.cutoff = cutoff
        self._i = 0
        self._j = 0

    def mul(self, a):
        pb, keep, _sup_a = self.chain[self._i]
        self._i += 1
        prod = sp.mul_point_s(a, self.f, pb)  # [B, S2]
        return poly.cutoff_drop(prod[..., :keep], self.cutoff)

    def mul_nc(self, a):
        pb, keep, _sup_a = self.chain[self._i]
        self._i += 1
        return sp.mul_point_s(a, self.f, pb)[..., :keep]

    def acc_add(self, res, term):
        su, er, ep = self.acc_binds[self._j]
        self._j += 1
        rv = torch.zeros(res.shape[0], su.size, dtype=res.dtype, device=res.device)
        tv = torch.zeros_like(rv)
        rv.index_copy_(-1, er, res)
        tv.index_copy_(-1, ep, term)
        return rv + tv

    def fresh_one(self):
        out = torch.zeros(self.f.shape[0], 1, dtype=self.f.dtype, device=self.f.device)
        out[:, 0] = 1.0
        return out

    def fresh_zero(self):
        return torch.zeros(self.f.shape[0], 1, dtype=self.f.dtype, device=self.f.device)

    def seed_f(self):
        return self.f.clone()


# ---------------------------------------------------------------------------
# Sparse executors
# ---------------------------------------------------------------------------


def exec_point_s(
    spec: SpecTape, x: torch.Tensor, eng: sp.SparseEngine, cutoff_threshold: float
) -> tuple[torch.Tensor, sp.Support]:
    """Sparse evaluate_no_remainder: x [B, n, Sx] -> (f [B, n, Sfu], sup).

    Mirrors ode_compiler.exec_point instruction-for-instruction; every op's
    layout comes from the specialization.
    """
    B = x.shape[0]
    dt, dev = x.dtype, x.device
    slots: dict[int, torch.Tensor] = {}

    _point_series = {
        "sin": lambda xx, si, mk: elem._trig_series_point_g(xx, spec.k, mk, cutoff_threshold, False),
        "cos": lambda xx, si, mk: elem._trig_series_point_g(xx, spec.k, mk, cutoff_threshold, True),
        "exp": lambda xx, si, mk: elem.exp_series_point_g(xx, spec.k, mk),
        "log": lambda xx, si, mk: elem.log_series_point_g(xx, spec.k, mk),
        "sqrt": lambda xx, si, mk: elem.sqrt_series_point_g(xx, spec.k, mk),
    }

    for si in spec.instrs:
        if si.op == "var":
            slots[si.dst] = x[:, si.var, si.gather][..., : si.keep_len]
        elif si.op == "const":
            t = torch.zeros(B, 1, dtype=dt, device=dev)
            t[:, 0] = si.const
            slots[si.dst] = t
        elif si.op == "neg":
            slots[si.dst] = -slots[si.a]
        elif si.op in ("add", "sub"):
            su, ea, eb = si.emb
            av = torch.zeros(B, su.size, dtype=dt, device=dev)
            bv = torch.zeros_like(av)
            av.index_copy_(-1, ea, slots[si.a])
            bv.index_copy_(-1, eb, slots[si.b])
            slots[si.dst] = av + bv if si.op == "add" else av - bv
        elif si.op == "mul":
            prod = sp.mul_point_s(slots[si.a], slots[si.b], si.pair)
            slots[si.dst] = poly.cutoff_drop(prod[..., : si.keep_len], cutoff_threshold)
        elif si.op == "div":
            ctx = SparsePointCtx(eng, si.chain, si.acc, None, cutoff_threshold)

            def mk(F, _ctx=ctx):
                _ctx.f = F
                return _ctx

            rec_b = elem.rec_series_point_g(slots[si.b], spec.k, mk)
            pb2, keep2, _sa, _sr = si.pair2
            prod = sp.mul_point_s(slots[si.a], rec_b, pb2)
            slots[si.dst] = poly.cutoff_drop(prod[..., :keep2], cutoff_threshold)
        elif si.op in _point_series:
            ctx = SparsePointCtx(eng, si.chain, si.acc, None, cutoff_threshold)

            def mk(F, _ctx=ctx):
                _ctx.f = F
                return _ctx

            slots[si.dst] = _point_series[si.op](slots[si.a], si, mk)
        elif si.op == "pow":
            if si.power == 0:
                t = torch.zeros(B, 1, dtype=dt, device=dev)
                t[:, 0] = 1.0
                slots[si.dst] = t
            else:
                result = slots[si.a].clone()
                temp = slots[si.a].clone()
                for tag, pb, keep, *_ in si.chain:
                    if tag == "r":
                        prod = sp.mul_point_s(result, temp, pb)
                        result = poly.cutoff_drop(prod[..., :keep], cutoff_threshold)
                    else:
                        prod = sp.mul_point_s(temp, temp, pb)
                        temp = poly.cutoff_drop(prod[..., :keep], cutoff_threshold)
                slots[si.dst] = result
        else:  # pragma: no cover  # unreachable: specialize emits only known ops
            raise AssertionError(f"bad op {si.op}")

    out = torch.zeros(B, len(spec.out_slots), spec.sup_out_union.size, dtype=dt, device=dev)
    for i, (slot, emb) in enumerate(zip(spec.out_slots, spec.out_embeds)):
        out[:, i].index_copy_(-1, emb, slots[slot])
    return out, spec.sup_out_union


def exec_valid_s(
    spec: SpecTape,
    code: CompiledODE,
    x: torch.Tensor,
    x_rem: torch.Tensor,
    eng: sp.SparseEngine,
    cutoff_threshold: float,
    tabs,
    bad_out: torch.Tensor | None = None,
    *, interval_coefficients: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse validated evaluation with range caching (exec_valid mirror).

    x [B, n, Sx] point coefficients; x_rem [B, n, 2]. Returns
    (coeffs_iv [B, n, Sfu, 2], rem [B, n, 2], cache [B, n_cache, 2],
    strict_tails [B, n_strict, 2]) — cache layout IDENTICAL to the dense
    executor (exec_replay consumes it unchanged).
    """
    if interval_coefficients and (x.ndim != 4 or x.shape[-1] != 2):
        raise ValueError("interval coefficient input requires [B,n,S,2]")
    B = x.shape[0]
    dt, dev = x.dtype, x.device
    coeffs: dict[int, torch.Tensor] = {}
    rems: dict[int, torch.Tensor] = {}
    cache = torch.zeros(B, code.n_cache, 2, dtype=dt, device=dev)
    strict_tails = torch.zeros(B, max(code.n_strict, 1), 2, dtype=dt, device=dev)

    def mkctx(si):
        ctx = SparseValidCtx(eng, si.chain, si.acc, None, None, si.sup_f, cutoff_threshold, preserve_intervals=interval_coefficients)

        def mk(F, Fr, _ctx=ctx):
            _ctx.tm_f = F
            _ctx.f_rem = Fr
            _ctx._range_f = sp.range_normal_iv_s(F, _ctx.eng, _ctx.sup_f)
            return _ctx

        return mk

    for si in spec.instrs:
        if si.op == "var":
            v = x[:, si.var, si.gather]  # [B, S_var]
            v_iv = v if interval_coefficients else iv.from_point(v)
            kept = v_iv[:, : si.keep_len]
            if si.keep_len < si.sup_in.size:
                dropped = v_iv[:, si.keep_len :]
                tail = iv.sum(iv.mul(dropped, eng.factor(si.sup_in)[si.keep_len :]), dim=-1)
            else:
                tail = torch.zeros(B, 2, dtype=dt, device=dev)
            coeffs[si.dst] = kept
            rems[si.dst] = iv.add(x_rem[:, si.var], tail)
            strict_tails[:, si.strict_slot] = tail
        elif si.op == "const":
            t = torch.zeros(B, 1, 2, dtype=dt, device=dev)
            t[:, 0, 0] = si.const
            t[:, 0, 1] = si.const
            coeffs[si.dst] = t
            r = torch.zeros(B, 2, dtype=dt, device=dev)
            r[:, 0] = si.crem_lo
            r[:, 1] = si.crem_hi
            rems[si.dst] = r
            cache[:, si.cache_base, 0] = si.crem_lo
            cache[:, si.cache_base, 1] = si.crem_hi
        elif si.op == "neg":
            coeffs[si.dst] = iv.neg(coeffs[si.a])
            rems[si.dst] = iv.neg(rems[si.a])
        elif si.op in ("add", "sub"):
            su, ea, eb = si.emb
            av = torch.zeros(B, su.size, 2, dtype=dt, device=dev)
            bv = torch.zeros_like(av)
            av.index_copy_(-2, ea, coeffs[si.a])
            bv.index_copy_(-2, eb, coeffs[si.b])
            if si.op == "add":
                coeffs[si.dst] = iv.add(av, bv)
                rems[si.dst] = iv.add(rems[si.a], rems[si.b])
            else:
                coeffs[si.dst] = iv.sub(av, bv)
                rems[si.dst] = iv.sub(rems[si.a], rems[si.b])
        elif si.op == "mul":
            range_b = sp.range_normal_iv_s(coeffs[si.b], eng, si.sup_b)
            kept, rem = tm_mul_valid_s(
                coeffs[si.a], rems[si.a], coeffs[si.b], rems[si.b],
                si.pair, si.keep_len, si.sup_a, eng, cutoff_threshold,
                cache, si.cache_base, range_b,
                preserve_intervals=interval_coefficients,
            )
            coeffs[si.dst] = kept
            rems[si.dst] = rem
        elif si.op == "div":
            rec_c, rec_r, bad = elem.rec_series_valid_g(
                coeffs[si.b], rems[si.b], spec.k, cache, si.cache_base, tabs, mkctx(si)
            )
            bad_out |= bad
            pb2, keep2, sup_a2, sup_rec = si.pair2
            range_b = sp.range_normal_iv_s(rec_c, eng, sup_rec)
            kept, rem = tm_mul_valid_s(
                coeffs[si.a], rems[si.a], rec_c, rec_r, pb2, keep2, sup_a2,
                eng, cutoff_threshold, cache, si.cache_base + 3 * spec.k + 3, range_b,
                preserve_intervals=interval_coefficients,
            )
            coeffs[si.dst] = kept
            rems[si.dst] = rem
        elif si.op in ("exp", "sin", "cos"):
            fn = {
                "exp": elem.exp_series_valid_g,
                "sin": lambda c, r, k, ca, b2, tb, mk: elem._trig_series_valid_g(
                    c, r, k, ca, b2, tb, mk, False),
                "cos": lambda c, r, k, ca, b2, tb, mk: elem._trig_series_valid_g(
                    c, r, k, ca, b2, tb, mk, True),
            }[si.op]
            out_c, out_r = fn(
                coeffs[si.a], rems[si.a], spec.k, cache, si.cache_base, tabs, mkctx(si)
            )
            coeffs[si.dst] = out_c
            rems[si.dst] = out_r
        elif si.op in ("log", "sqrt"):
            fn = elem.log_series_valid_g if si.op == "log" else elem.sqrt_series_valid_g
            out_c, out_r, bad = fn(
                coeffs[si.a], rems[si.a], spec.k, cache, si.cache_base, tabs, mkctx(si)
            )
            bad_out |= bad
            coeffs[si.dst] = out_c
            rems[si.dst] = out_r
        elif si.op == "pow":
            if si.power == 0:
                t = torch.zeros(B, 1, 2, dtype=dt, device=dev)
                t[:, 0, 0] = 1.0
                t[:, 0, 1] = 1.0
                coeffs[si.dst] = t
                rems[si.dst] = torch.zeros(B, 2, dtype=dt, device=dev)
            else:
                res_c = coeffs[si.a].clone()
                res_r = rems[si.a].clone()
                tmp_c = res_c.clone()
                tmp_r = res_r.clone()
                cursor = si.cache_base
                for entry in si.chain:
                    if entry[0] == "r":
                        _tag, pb, keep, sup_res, sup_tmp = entry
                        range_b = sp.range_normal_iv_s(tmp_c, eng, sup_tmp)
                        res_c, res_r = tm_mul_valid_s(
                            res_c, res_r, tmp_c, tmp_r, pb, keep, sup_res,
                            eng, cutoff_threshold, cache, cursor, range_b,
                            preserve_intervals=interval_coefficients,
                        )
                    else:
                        _tag, pb, keep, sup_tmp = entry
                        range_b = sp.range_normal_iv_s(tmp_c, eng, sup_tmp)
                        tmp_c, tmp_r = tm_mul_valid_s(
                            tmp_c, tmp_r, tmp_c, tmp_r, pb, keep, sup_tmp,
                            eng, cutoff_threshold, cache, cursor, range_b,
                            preserve_intervals=interval_coefficients,
                        )
                    cursor += 3
                coeffs[si.dst] = res_c
                rems[si.dst] = res_r
        else:  # pragma: no cover  # unreachable: specialize emits only known ops
            raise AssertionError(f"bad op {si.op}")

    out_c = torch.zeros(B, len(spec.out_slots), spec.sup_out_union.size, 2, dtype=dt, device=dev)
    out_r = torch.zeros(B, len(spec.out_slots), 2, dtype=dt, device=dev)
    for i, (slot, emb) in enumerate(zip(spec.out_slots, spec.out_embeds)):
        out_c[:, i].index_copy_(-2, emb, coeffs[slot])
        out_r[:, i] = rems[slot]
    return out_c, out_r, cache, strict_tails


# ---------------------------------------------------------------------------
# Sparse composition (mirrors composition.compose on needed images only)
# ---------------------------------------------------------------------------


@dataclass
class ComposeSpec:
    """Specialized composition plan for (sup_a spatial, sup_g spatial, order).

    members: spatial ids of every monomial whose image the contraction needs
        (degree >= 1 entries of sup_a), ascending.
    a_gather [M] dev: local positions of members inside sup_a's layout.
    levels: per degree d >= 2: (rows [L] dev — row index of each member's
        image in the level tensor being built, parent_src (level_idx, rows or
        var ids), lastvars [L] dev, parent_emb idx, PairBind, keep_len,
        sup_lvl) — everything exec-time needs to build the level in one
        batched multiply on the level's union layout.
    sup_m: the union layout every needed image is embedded into for the
        contraction; m_embeds: per group embedding indices.
    """

    members: tuple
    a_gather: torch.Tensor
    var_members: torch.Tensor  # positions (0..M-1) of degree-1 members
    var_ids: torch.Tensor  # which state var each degree-1 member is (g row)
    levels: list
    sup_m: sp.Support
    n_members: int


def specialize_compose(
    sup_a: sp.Support, sup_g: sp.Support, order: int, sched: CompositionSchedule,
    eng: sp.SparseEngine,
) -> ComposeSpec:
    """Build (and cache) the composition plan.

    Support algebra: image_sup(var) = sup_g; image_sup(m) =
    kept_order(pair(image_sup(parent(m)), sup_g)) — computed on per-level
    UNIONS (a superset per member, sound and batchable).
    """
    cache = _eng_cache(eng, "_cspec")
    key = (sup_a, sup_g, order)
    hit = cache.get(key)
    if hit is not None:
        return hit

    parent = sched.parent.cpu().numpy()
    lastvar = sched.lastvar.cpu().numpy()
    # Needed = degree >= 1 monomials of sup_a + DAG ancestors (parents chain).
    a_ids = [i for i, d in zip(sup_a.ids, sup_a.degs) if d >= 1]
    need = set()
    stack = list(a_ids)
    while stack:
        m = stack.pop()
        if m in need:
            continue
        need.add(m)
        p = int(parent[m])
        if p >= 0:
            stack.append(p)
    members = tuple(sorted(need))
    m_pos = {m: i for i, m in enumerate(members)}

    # Degrees within the spatial basis: reuse Support metadata via make_support.
    deg_of = dict(zip(sup_a.ids, sup_a.degs))
    sup_all = sp.make_support(sup_a.n, sup_a.k, True, members)
    deg_of.update(dict(zip(sup_all.ids, sup_all.degs)))

    var_members, var_ids = [], []
    by_deg: dict[int, list] = {}
    for m in members:
        d = deg_of[m]
        if d == 1:
            var_members.append(m_pos[m])
            var_ids.append(int(np.nonzero(sched.var_image.cpu().numpy() == m)[0][0]))
        else:
            # members hold ONLY degree >= 1 monomials by construction (the
            # ancestor closure starts from degree >= 1 ids and parents stop
            # at degree 1), so this arm is exactly "degree >= 2".
            by_deg.setdefault(d, []).append(m)

    img_sup: dict[int, sp.Support] = {}
    for vm in sched.var_image.cpu().numpy():
        img_sup[int(vm)] = sup_g

    levels = []
    dev = eng.device
    for d in sorted(by_deg):
        mems = by_deg[d]
        # Union of parents' image supports = the level input layout.
        psups = [img_sup[int(parent[m])] for m in mems]
        sup_in = psups[0]
        for s2 in psups[1:]:
            sup_in, _, _ = eng.union(sup_in, s2)
        pb = eng.pair(sup_in, sup_g)
        keep = pb.sup_out.keep_len(order)
        sup_lvl = sp.make_support(pb.sup_out.n, pb.sup_out.k, True, pb.sup_out.ids[:keep])
        for m in mems:
            img_sup[m] = sup_lvl  # per-member superset = level layout
        # Parent source: (kind, idx) — 'var' parents read g rows, 'img'
        # parents read a previous level's tensor rows.
        pk, pidx, pemb = [], [], []
        for m in mems:
            p = int(parent[m])
            if deg_of.get(p, sup_all.degs[sup_all.ids.index(p)] if p in sup_all.ids else 1) == 1:
                pk.append(0)
                pidx.append(int(np.nonzero(sched.var_image.cpu().numpy() == p)[0][0]))
            else:
                pk.append(1)
                pidx.append(m_pos[p])
        levels.append({
            "rows": torch.tensor([m_pos[m] for m in mems], dtype=torch.int64, device=dev),
            "pkind": tuple(pk),
            "pidx": tuple(pidx),
            "lastvars": torch.tensor([int(lastvar[m]) for m in mems], dtype=torch.int64, device=dev),
            "pb": pb,
            "keep": keep,
            "sup_in": sup_in,
            "sup_lvl": sup_lvl,
            "psups": psups,
            "mems": mems,
        })

    # The contraction layout: union of every member's image layout.
    sup_m = sup_g
    for lv in levels:
        sup_m, _, _ = eng.union(sup_m, lv["sup_lvl"])

    # a_gather: contraction weight position per member. DAG ANCESTORS pulled
    # in by the parent-chain closure need not be in sup_a at all (cutoff can
    # kill a monomial while keeping its multiples — REPORT finding of the
    # sparse test suite); absent members read an appended ZERO column
    # (sentinel index len(sup_a.ids)) so they contribute exactly nothing.
    a_pos = {m: i for i, m in enumerate(sup_a.ids)}
    spec = ComposeSpec(
        members=members,
        a_gather=torch.tensor(
            [a_pos.get(m, len(sup_a.ids)) for m in members],
            dtype=torch.int64, device=dev,
        ),
        var_members=torch.tensor(var_members, dtype=torch.int64, device=dev),
        var_ids=torch.tensor(var_ids, dtype=torch.int64, device=dev),
        levels=levels,
        sup_m=sup_m,
        n_members=len(members),
    )
    cache[key] = spec
    return spec


def _compose_s_eager(
    a_coeffs: torch.Tensor,
    a_rem: torch.Tensor,
    g_coeffs: torch.Tensor,
    g_rem: torch.Tensor,
    sup_a: sp.Support,
    sup_g: sp.Support,
    order: int,
    cutoff_threshold: float,
    strict: bool,
    sched: CompositionSchedule,
    eng: sp.SparseEngine,
) -> tuple[torch.Tensor, torch.Tensor, sp.Support]:
    """Sparse new_tmv = a o g (mirrors composition.compose).

    a_coeffs [B, n, Sa] spatial, constant slot zero; g_coeffs [B, n, Sg]
    spatial; rems [B, n, 2]. Returns (coeffs [B, n, Sm], rem [B, n, 2], sup).
    """
    B, n, _ = a_coeffs.shape
    dt, dev = a_coeffs.dtype, a_coeffs.device
    cs = specialize_compose(sup_a, sup_g, order, sched, eng)
    M = cs.n_members

    if M == 0:
        # Purely-constant a (slot 0 removed => nothing to substitute).
        zero = torch.zeros(B, n, sup_g.size, dtype=dt, device=dev)
        return zero, iv.add(a_rem, torch.zeros_like(a_rem)), sup_g

    g_range = sp.range_spatial_s(g_coeffs, eng, sup_g)  # [B, n, 2]

    # Image storage on the contraction layout.
    m_coeffs = torch.zeros(B, M, cs.sup_m.size, dtype=dt, device=dev)
    m_rem = torch.zeros(B, M, 2, dtype=dt, device=dev)

    # Degree-1 members: image of r_i is g_i itself.
    if cs.var_members.numel():
        emb_g = eng.embed_idx(sup_g, cs.sup_m)
        rows = torch.zeros(B, cs.var_members.numel(), cs.sup_m.size, dtype=dt, device=dev)
        rows.index_copy_(-1, emb_g, g_coeffs[:, cs.var_ids])
        m_coeffs[:, cs.var_members] = rows
        m_rem[:, cs.var_members] = g_rem[:, cs.var_ids]

    for lv in cs.levels:
        # Assemble parent tensors on the level input union layout.
        L = lv["rows"].numel()
        pa = torch.zeros(B, L, lv["sup_in"].size, dtype=dt, device=dev)
        p_rem = torch.zeros(B, L, 2, dtype=dt, device=dev)
        pa_range = torch.zeros(B, L, 2, dtype=dt, device=dev)
        for j, (m, psup) in enumerate(zip(lv["mems"], lv["psups"])):
            # Static plan metadata stays on the host: reading CUDA scalars here
            # would synchronize the stream for every parent monomial.
            kind = lv["pkind"][j]
            idx = lv["pidx"][j]
            if kind == 0:  # var parent: g row
                src = g_coeffs[:, idx]
                pa[:, j].index_copy_(-1, eng.embed_idx(sup_g, lv["sup_in"]), src)
                p_rem[:, j] = g_rem[:, idx]
                pa_range[:, j] = g_range[:, idx]
            else:  # image parent: previous level row (on sup_m layout)
                src = m_coeffs[:, idx]
                # psup is the parent's level layout, embedded within sup_m.
                gather = eng.embed_idx(psup, cs.sup_m)
                sub = src[:, gather]  # [B, S_p]
                pa[:, j].index_copy_(-1, eng.embed_idx(psup, lv["sup_in"]), sub)
                p_rem[:, j] = m_rem[:, idx]
                pa_range[:, j] = sp.range_spatial_s(sub, eng, psup)
        gv = g_coeffs[:, lv["lastvars"]]  # [B, L, Sg]
        gv_rem = g_rem[:, lv["lastvars"]]
        if strict:
            prod, coefficient_roundoff = sp.mul_point_s_with_roundoff(
                pa, gv, lv["pb"], eng
            )
        else:
            prod = sp.mul_point_s(pa, gv, lv["pb"])  # [B, L, S2]
            coefficient_roundoff = None
        rem = iv.mul(p_rem, gv_rem)
        rem = iv.add(rem, iv.mul(g_range[:, lv["lastvars"]], p_rem))
        rem = iv.add(rem, iv.mul(pa_range, gv_rem))
        kept = prod[..., : lv["keep"]]
        if lv["keep"] < lv["pb"].sup_out.size:
            tail = iv.dot_point_iv(
                prod[..., lv["keep"] :], eng.cat(lv["pb"].sup_out)[lv["keep"] :], dim=-1
            )
        else:
            tail = torch.zeros(B, L, 2, dtype=dt, device=dev)
        kept, round_rng = sp.cutoff_spatial_s(kept, eng, lv["sup_lvl"], cutoff_threshold)
        rem = iv.add(rem, iv.add(tail, round_rng))
        if coefficient_roundoff is not None:
            rem = iv.add(rem, coefficient_roundoff)
        out_rows = torch.zeros(B, L, cs.sup_m.size, dtype=dt, device=dev)
        out_rows.index_copy_(-1, eng.embed_idx(lv["sup_lvl"], cs.sup_m), kept)
        m_coeffs[:, lv["rows"]] = out_rows
        m_rem[:, lv["rows"]] = rem

    # Contraction: tier-P GEMM over the needed members + interval rem dot.
    # (zero-pad column serves the absent-ancestor sentinel in a_gather.)
    a_pad = torch.cat(
        [a_coeffs, torch.zeros(B, n, 1, dtype=dt, device=dev)], dim=-1
    )  # [B, n, Sa+1]
    a_sub = a_pad[..., cs.a_gather]  # [B, n, M]
    out_coeffs = torch.einsum("bim,bmt->bit", a_sub, m_coeffs)  # [B, n, Sm]
    w_rem = iv.dot_point_iv(a_sub, m_rem.unsqueeze(1), dim=-1)  # [B, n, 2]
    out_rem = iv.add(w_rem, a_rem)

    if strict:
        abs_dot = torch.einsum("bim,bmt->bit", a_sub.abs(), m_coeffs.abs())  # noqa: E501 — a_sub already zero for absent members
        gemm_err = dot_error_bound(abs_dot, M)  # [B, n, Sm]
        err_rng = iv.dot_point_iv(gemm_err, torch.abs(eng.cat(cs.sup_m)), dim=-1)
        sym = err_rng[..., 1]
        out_rem = iv.add(out_rem, torch.stack((-sym, sym), dim=-1))

    return out_coeffs, out_rem, cs.sup_m


def compose_s(
    a_coeffs, a_rem, g_coeffs, g_rem, sup_a, sup_g,
    order, cutoff_threshold, strict, sched, eng,
):
    """Replay the same strict/eager composition kernels for a fixed support.

    Only plan metadata is closed over; all changing coefficient/remainder
    tensors are explicit graph inputs. GraphCache owns its private pool and
    the engine owns the plan tables for the lifetime of every replay.
    """
    kernel = _compose_s_eager
    if strict and COMPOSITION_MODE == "horner":
        from .sparse_horner import compose_horner
        kernel = compose_horner


    def compute(a, ar, g, gr):
        return kernel(
            a, ar, g, gr, sup_a, sup_g, order, cutoff_threshold, strict, sched, eng
        )
    if not a_coeffs.is_cuda:
        return compute(a_coeffs, a_rem, g_coeffs, g_rem)
    key = ("compose", COMPOSITION_MODE, id(getattr(eng, "horner_edge_kernel", None)),
           iv.USE_KERNELS, sup_a, sup_g, order, cutoff_threshold, strict,
           a_coeffs.shape[0], a_coeffs.dtype)
    return graph_cache(eng, str(a_coeffs.device)).run(
        key, compute, [a_coeffs, a_rem, g_coeffs, g_rem]
    )


# ---------------------------------------------------------------------------
# Sparse flowpipe state + advance
# ---------------------------------------------------------------------------

# Diagnostics: refinement iterations executed per graphed refine call
# (appended by refine_loop_graphed; consumed by bench/profile tooling).
REFINE_ITERS: list = []

# Dispatch switch for the one-launch interpreter kernels (tape_kernels).
# Default on; tests flip it off to exercise/capture the eager executors the
# kernels are differentially pinned against. FLOWSTAR_NO_TAPE_KERNEL=1 turns
# them off at import (diagnostic bisection: tapes vs graphs vs eager).
import os as _os

USE_TAPE_KERNELS = _os.environ.get("FLOWSTAR_NO_TAPE_KERNEL", "0") != "1"
COMPOSITION_MODE = _os.environ.get("FLOWSTAR_COMPOSITION", "monomial")
SUPPORT_POLICY = _os.environ.get("FLOWSTAR_SUPPORT_POLICY", "measured")
VALIDATION_POLICY = _os.environ.get("FLOWSTAR_VALIDATION_POLICY", "truncated")
INJECTIVE_GLUE = _os.environ.get("FLOWSTAR_INJECTIVE_GLUE", "0") == "1"
if VALIDATION_POLICY not in ("truncated", "defer_polynomial", "defer_direct_leaves", "defer_linear_leaves", "solution_order", "solution_plus_one"):
    raise ValueError(f"unknown validation policy: {VALIDATION_POLICY}")
if SUPPORT_POLICY not in ("measured", "structural"):
    raise ValueError(f"unknown support policy: {SUPPORT_POLICY}")

if COMPOSITION_MODE not in ("monomial", "horner"):
    raise ValueError(f"unknown sparse composition mode: {COMPOSITION_MODE}")

ACTIVE = 0
FAILED_CONTRACTION = 1
DONE = 2
FAILED_DIV = 3
INITIAL_SIMP = 1e-4


@dataclass
class SparseState:
    """Sparse flowpipe batch: support-aligned twins of FlowpipeBatch.

    pre [B, n, Sp] on pre_sup (FULL-basis ids); tmv [B, n, St] on tmv_sup
    (SPATIAL ids); rems [B, n, 2]; status [B] i8 (same codes as flowpipe).
    """

    pre: torch.Tensor
    pre_sup: sp.Support
    pre_rem: torch.Tensor
    tmv: torch.Tensor
    tmv_sup: sp.Support
    tmv_rem: torch.Tensor
    status: torch.Tensor


def initial_sparse_state(
    boxes: torch.Tensor, eng: sp.SparseEngine, sched: CompositionSchedule
) -> SparseState:
    """Box initial sets -> canonical sparse flowpipes (initial_flowpipe twin).

    Initial supports are exact: pre on {1} u {r_i} (full ids), tmv identity on
    {1} u {r_i} (spatial ids).
    """
    B, n, _ = boxes.shape
    dt, dev = boxes.dtype, str(boxes.device)
    tables = eng.tables
    mid = (boxes[..., 0] + boxes[..., 1]) * 0.5  # [B, n]
    rad = next_up(torch.maximum(boxes[..., 1] - mid, mid - boxes[..., 0]))
    rad = torch.where(boxes[..., 1] == boxes[..., 0], torch.zeros_like(rad), rad)

    var_sp = tuple(int(v) for v in sched.var_image.cpu().numpy())  # spatial ids
    var_full = tuple(int(v) for v in tables.spatial_index.cpu().numpy()[list(var_sp)])
    sup_pre = sp.make_support(tables.n, tables.k, False, var_full)
    sup_tmv = sp.make_support(tables.n, tables.k, True, var_sp)

    pre = torch.zeros(B, n, sup_pre.size, dtype=dt, device=dev)
    pre[..., 0] = mid
    pos_full = [sup_pre.ids.index(v) for v in var_full]  # local slots of r_i
    pre[:, torch.arange(n, device=dev), torch.tensor(pos_full, device=dev)] = rad

    tmv = torch.zeros(B, n, sup_tmv.size, dtype=dt, device=dev)
    pos_sp = [sup_tmv.ids.index(v) for v in var_sp]
    tmv[:, torch.arange(n, device=dev), torch.tensor(pos_sp, device=dev)] = 1.0

    zero2 = torch.zeros(B, n, 2, dtype=dt, device=dev)
    return SparseState(
        pre=pre, pre_sup=sup_pre, pre_rem=zero2.clone(),
        tmv=tmv, tmv_sup=sup_tmv, tmv_rem=zero2.clone(),
        status=torch.zeros(B, dtype=torch.int8, device=dev),
    )


def _accum_sups(eng: sp.SparseEngine, key, sups: tuple) -> tuple:
    """Monotone per-slot support accumulation (spec/graph stability).

    Measured supports oscillate ulp-level across steps (coefficients crossing
    the cutoff), which would flip the specialization key — and every new key
    costs a fresh CUDA-graph capture (~1s). Union-accumulating per slot makes
    the keys stabilize after the first few steps; a SUPERSET support is exact
    (extra slots hold zeros — the zero-removal argument in reverse).
    """
    store = _eng_cache(eng, "_acc_sups")
    prev = store.get(key)
    if prev is None:
        store[key] = sups
        return sups
    merged = []
    changed = False
    for a, b in zip(prev, sups):
        if b.ids == a.ids or set(b.ids) <= set(a.ids):
            merged.append(a)
        else:
            u, _, _ = sp.union_maps(a, b)
            merged.append(u)
            changed = True
    merged = tuple(merged)
    if changed:
        store[key] = merged
    return store[key]


def _extract_row_sups(
    x: torch.Tensor, sup_x: sp.Support, eng: sp.SparseEngine
) -> tuple:
    """Per-row measured supports of x [B, n, S]: ONE host sync for all rows.

    Returns a tuple of n Supports (each a subset of sup_x, plus id 0)."""
    mask = (x != 0).any(dim=0).cpu().numpy()  # [n, S] SYNC
    ids = np.asarray(sup_x.ids, dtype=np.int64)
    n = mask.shape[0]
    return tuple(
        sp.make_support(sup_x.n, sup_x.k, sup_x.spatial,
                        tuple(int(v) for v in ids[mask[i]]))
        for i in range(n)
    )


def _refine_dispatch(
    code: CompiledODE,
    accepted: torch.Tensor,
    ok: torch.Tensor,
    bad: torch.Tensor,
    cache: torch.Tensor,
    tails: torch.Tensor,
    strict: bool,
    int_diff: torch.Tensor,
    step: StepTables,
    settings: Settings,
    eng: sp.SparseEngine,
    cache_state=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refinement dispatcher: ONE-LAUNCH tape-interpreter kernel when the
    tape serializes (tape_kernels; bitwise-equal to the eager loop by its
    test contract, 200-450x faster), else the chunked graph, else eager."""
    from . import tape_kernels as tk
    from .flowpipe import refine_loop

    if settings.refinement_callback is not None:
        return refine_loop(
            code, accepted, ok, bad, cache, tails, strict, int_diff, step,
            settings, cache_state=cache_state,
        )

    if USE_TAPE_KERNELS and accepted.is_cuda and tk.available():
        tapes = _eng_cache(eng, "_rtape")
        tape = tapes.get(_code_serial(code))
        if tape is None and _code_serial(code) not in _eng_cache(eng, "_rtape_no"):
            try:
                tape = tk.serialize_replay_tape(code, str(accepted.device))
                tapes[_code_serial(code)] = tape
            except ValueError:  # capacity guards (n_slots/n) -> graph path
                _eng_cache(eng, "_rtape_no")[_code_serial(code)] = True
        if tape is not None:
            dts = _eng_cache(eng, "_dt_pair")
            pair = dts.get(id(step))
            if pair is None:
                pair = (float(step.pow_iv[1, 0]), float(step.pow_iv[1, 1]))
                dts[id(step)] = pair  # one-time host sync per run
            cur, bad_out, _iters = tk.refine_tape_run(
                tape, accepted, ok, bad, cache, tails, strict, int_diff,
                pair, settings.stop_ratio, settings.max_refinement_steps,
            )
            return cur, bad_out
    return refine_loop_graphed(code, accepted, ok, bad, cache, tails, strict,
                               int_diff, step, settings, eng)


def refine_loop_graphed(
    code: CompiledODE,
    accepted: torch.Tensor,
    ok: torch.Tensor,
    bad: torch.Tensor,
    cache: torch.Tensor,
    tails: torch.Tensor,
    strict: bool,
    int_diff: torch.Tensor,
    step: StepTables,
    settings: Settings,
    eng: sp.SparseEngine,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA-graph refinement: ONE captured iteration body replayed in chunks.

    Bitwise-equal to flowpipe.refine_loop: the body is the same op sequence;
    host stop-checks run at doubling boundaries (1, 2, 4, 8...) instead of
    every iteration, and iterations past the stop point are exact no-ops
    (every update mask is False once `refining` clears — flowpipe.refine_loop
    docstring argument); the MAX_REFINEMENT_STEPS cap is enforced exactly by
    clipping the final chunk. Falls back to the eager loop off-CUDA.
    """
    from .flowpipe import refine_loop

    gc = graph_cache(eng, str(accepted.device))
    if not gc.enabled:
        return refine_loop(code, accepted, ok, bad, cache, tails, strict,
                           int_diff, step, settings)

    B, n, _ = accepted.shape
    dev = accepted.device
    # Closure constants MUST outlive the captured graph (graphing.py keepalive
    # contract): key them on the engine so hit-path replays read live memory.
    key = (_code_serial(code), "refine", strict, B, n, cache.shape[1])
    consts = _eng_cache(eng, "_refine_consts")
    ka = consts.get(key)
    if ka is None:
        ka = (
            step.dt(2).expand(B, n, 2).contiguous(),  # [B, n, 2]
            torch.arange(n, device=dev).unsqueeze(0),  # [1, n]
            torch.full((B,), n, device=dev, dtype=torch.long),  # [B]
        )
        consts[key] = ka
    dt_iv, dim_idx, full_n = ka
    stop_ratio = settings.stop_ratio

    def body(cur_s, refining_s, bad_s, cache_s, tails_s, int_diff_s):
        bad_replay = torch.zeros_like(bad_s)
        new_rem = exec_replay(code, cur_s, cache_s, tails_s, strict, bad_out=bad_replay)
        bad_replay |= ~torch.isfinite(new_rem).all(dim=(1, 2))
        live = refining_s & ~bad_replay  # [B]
        new_rem = torch.where(
            bad_replay.view(B, 1, 1), torch.zeros_like(new_rem), new_rem
        )
        new_rem = iv.mul(new_rem, dt_iv)
        new_rem = iv.add(new_rem, int_diff_s)
        sub_ok = iv.contains(cur_s, new_rem)  # [B, n]
        fail_any = ~sub_ok
        first_fail = torch.where(fail_any.any(dim=-1),
                                 fail_any.float().argmax(dim=-1), full_n)
        update = (dim_idx < first_fail.unsqueeze(-1)) & live.unsqueeze(-1)
        ratio = iv.width(new_rem) / iv.width(cur_s).clamp(min=5e-324)
        improved = (ratio <= stop_ratio) & update
        cur_s.copy_(torch.where(update.unsqueeze(-1), new_rem, cur_s))
        bad_s |= bad_replay & refining_s
        refining_s.copy_(live & (~fail_any.any(dim=-1)) & improved.any(dim=-1))
        return cur_s, refining_s, bad_s

    ent = gc._segs.get(key)
    if ent is None:
        ent = gc._capture(key, body, [accepted, ok, bad, cache, tails, int_diff],
                          keepalive=ka)
    statics, graph, _out, _ka = ent
    for st_buf, x in zip(statics, (accepted, ok, bad, cache, tails, int_diff)):
        st_buf.copy_(x)
    cur_s, refining_s, bad_s = statics[0], statics[1], statics[2]

    done, chunk = 0, 1
    max_steps = settings.max_refinement_steps
    while done < max_steps:
        if not bool(refining_s.any()):  # host check (SYNC) at chunk boundary
            break
        todo = min(chunk, max_steps - done)
        for _ in range(todo):
            graph.replay()
        done += todo
        chunk = min(chunk * 2, 8)
    REFINE_ITERS.append(done)  # diagnostics ring (bench/profile readers)
    if len(REFINE_ITERS) > 4096:
        del REFINE_ITERS[:2048]
    return cur_s.clone(), bad_s.clone()


def _graphed_point(spec, x, eng, cutoff_eps):
    """exec_point_s dispatcher: one-launch interpreter kernel (tape_kernels,
    bitwise-faithful by its test contract) when the spec serializes, else a
    CUDA-graph capture, else eager."""
    from . import tape_kernels as tk

    if USE_TAPE_KERNELS and x.is_cuda and tk.valid_available():
        tapes = _eng_cache(eng, "_ptape")
        key = id(spec)
        if key not in tapes:
            try:
                tapes[key] = tk.serialize_point_tape(spec, eng, cutoff_eps, str(x.device))
            except ValueError:  # capacity guard -> graph fallback
                tapes[key] = None
        pt = tapes[key]
        if pt is not None:
            return tk.exec_point_tape(pt, x), spec.sup_out_union
    gc = graph_cache(eng, str(x.device))
    out = gc.run(("pt", id(spec)), lambda xs: exec_point_s(spec, xs, eng, cutoff_eps)[0], [x])
    return out, spec.sup_out_union


def _graphed_valid(spec, code, x, x_rem, bad, eng, cutoff_eps, tabs):
    """exec_valid_s dispatcher: interpreter kernel -> graph -> eager. The
    mutated bad mask rides in the outputs on every path."""
    from . import tape_kernels as tk

    if USE_TAPE_KERNELS and x.is_cuda and tk.valid_available():
        tapes = _eng_cache(eng, "_vtape")
        key = id(spec)
        if key not in tapes:
            try:
                tapes[key] = tk.serialize_valid_tape(spec, code, eng, cutoff_eps, str(x.device))
            except ValueError:  # capacity guard -> graph fallback
                tapes[key] = None
        vt = tapes[key]
        if vt is not None:
            bd = bad.clone()
            c, r, ca, tl = tk.exec_valid_tape(vt, x, x_rem, bd)
            return c, r, ca, tl, bd
    gc = graph_cache(eng, str(x.device))

    def fn(xs, xr, bd):
        c, r, ca, tl = exec_valid_s(spec, code, xs, xr, eng, cutoff_eps, tabs, bad_out=bd)
        return c, r, ca, tl, bd

    return gc.run(("vd", id(spec)), fn, [x, x_rem, bad])


# ---------------------------------------------------------------------------
# Compiled glue regions (M11): pure-tensor segments of advance_sparse fused by
# torch.inductor (glue.py). Each takes ONLY tensors + static ints; support/
# engine lookups happen in the caller. DETERMINISM: index_copy_/index_add_
# targets in here are injective (glue.py module docstring rule).
# Optional maps are certified from CPU Support values before capture and live
# in the engine cache. They are static graph inputs, never coefficient-derived.
# The explicit policy bit is in each affected plan/graph key; compile mode
# keeps its original torch operations and receives maps=None.
# ---------------------------------------------------------------------------


def _g_precond(
    new_tmv, new_tmv_rem, cat_f, c0, pos_full, arange_n, x0_size: int,
    strict: bool = False,
):
    """Stage (e) minus the SR branch: precondition + new_x0 assembly.

    new_tmv [B, n, St], new_tmv_rem [B, n, 2], cat_f [St, 2], c0 [B, n],
    pos_full [n], arange_n [n] -> (tmv_scaled, rem_scaled, S, point_dim,
    scale, new_x0 [B, n, x0_size])."""
    rng = iv.add(iv.dot_point_iv(new_tmv, cat_f, dim=-1), new_tmv_rem)
    S, point_dim, scale = preconditioning_scale(rng, strict=strict)
    tmv_scaled = new_tmv * scale.unsqueeze(-1)
    rem_scaled = iv.mul_point(new_tmv_rem, scale)
    B, n = S.shape
    new_x0 = torch.zeros(B, n, x0_size, dtype=new_tmv.dtype, device=new_tmv.device)
    new_x0[..., 0] = c0
    new_x0[:, arange_n, pos_full] = S
    return tmv_scaled, rem_scaled, S, point_dim, scale, new_x0



def _center_preconditioning(pre_tmv, pre_rem, center, fresh_j, factors, use_sr):
    """Translate the normalization origin; retain all constant-rounding error."""
    bounds = iv.add(iv.dot_point_iv(pre_tmv, factors, dim=-1), pre_rem)
    shift = bounds[..., 0] * 0.5 + bounds[..., 1] * 0.5
    moved_center = center + shift
    moved = pre_tmv.clone()
    moved[..., 0] = pre_tmv[..., 0] - shift
    old_constant = iv.add(iv.from_point(center), iv.from_point(pre_tmv[..., 0]))
    new_constant = iv.add(iv.from_point(moved_center), iv.from_point(moved[..., 0]))
    error = iv.sub(old_constant, new_constant)
    unchanged = (moved_center == center) & (moved[..., 0] == pre_tmv[..., 0])
    error = torch.where(unchanged.unsqueeze(-1), torch.zeros_like(error), error)
    return moved, iv.add(pre_rem, error), moved_center, iv.add(fresh_j, error) if use_sr else fresh_j


def _g_sweep(f, int_scale, int_tgt, new_x0, emb_x0, emb_int, out_size: int, x_size: int,
             maps=None):
    """Stage (f) per-sweep glue: t-integration + union-aligned add.

    f [B, n, Sf]; injective integration scatter then injective union embeds
    (integrate_t_s + add_aligned inlined so inductor fuses the chain)."""
    B, n = f.shape[0], f.shape[1]
    src = f * int_scale
    int_f = torch.zeros(B, n, out_size, dtype=f.dtype, device=f.device)
    if maps is None:
        int_f.index_add_(-1, int_tgt, src)
    else:
        ii.update_(int_f, -1, src, maps[0], add=True)
    av = torch.zeros(B, n, x_size, dtype=f.dtype, device=f.device)
    bv = torch.zeros(B, n, x_size, dtype=f.dtype, device=f.device)
    if maps is None:
        av.index_copy_(-1, emb_x0, new_x0)
        bv.index_copy_(-1, emb_int, int_f)
    else:
        ii.update_(av, -1, new_x0, maps[1])
        ii.update_(bv, -1, int_f, maps[2])
    return av + bv


def _g_validpost(f_iv, f_rem, new_x0, x, x_rem, int_scale_iv, int_tgt,
                 e_int, e_x0, e_t, e_xx, factor_d, dt_iv,
                 int_size: int, t_size: int, d_size: int, maps=None):
    """Stages (h)-(j) glue: interval t-integration, tmv_tmp/diff assembly,
    range of the difference, contraction inputs. Returns (total, ok_dims)."""
    B, n = f_iv.shape[0], f_iv.shape[1]
    src = iv.mul(f_iv, int_scale_iv)
    int_f_iv = torch.zeros(B, n, int_size, 2, dtype=f_iv.dtype, device=f_iv.device)
    if maps is None:
        int_f_iv.index_add_(-2, int_tgt, src)
    else:
        ii.update_(int_f_iv, -2, src, maps[0], add=True)
    tmv_tmp = torch.zeros(B, n, t_size, 2, dtype=f_iv.dtype, device=f_iv.device)
    if maps is None:
        tmv_tmp.index_copy_(-2, e_int, int_f_iv)
    else:
        ii.update_(tmv_tmp, -2, int_f_iv, maps[1])
    x0_iv = torch.zeros(B, n, t_size, 2, dtype=f_iv.dtype, device=f_iv.device)
    if maps is None:
        x0_iv.index_copy_(-2, e_x0, iv.from_point(new_x0))
    else:
        ii.update_(x0_iv, -2, iv.from_point(new_x0), maps[2])
    tmv_tmp = iv.add(tmv_tmp, x0_iv)
    tmv_tmp_rem = iv.mul(f_rem, dt_iv)
    lhs = torch.zeros(B, n, d_size, 2, dtype=f_iv.dtype, device=f_iv.device)
    if maps is None:
        lhs.index_copy_(-2, e_t, tmv_tmp)
    else:
        ii.update_(lhs, -2, tmv_tmp, maps[3])
    rhs = torch.zeros(B, n, d_size, 2, dtype=f_iv.dtype, device=f_iv.device)
    if maps is None:
        rhs.index_copy_(-2, e_xx, iv.from_point(x))
    else:
        ii.update_(rhs, -2, iv.from_point(x), maps[4])
    diff_iv = iv.sub(lhs, rhs)
    prod = iv.mul(diff_iv, factor_d)
    int_diff = iv.sum(prod, dim=-1)
    total = iv.add(tmv_tmp_rem, int_diff)
    ok_dims = iv.contains(x_rem, total)
    return total, ok_dims, int_diff


def _g_emit(x, pre_old_src, new_tmv, tmv_old_src, cur, pre_rem_old,
            new_tmv_rem, tmv_rem_old, status, ok, bad,
            e_new, e_old, t_new, t_old, up_size: int, ut_size: int,
            active_code: int, div_code: int, contract_code: int, maps=None):
    """Stage (l): checked frozen-lane merge (all injective embeds)."""
    from .flowpipe import _lane_nonfinite
    bad = bad | _lane_nonfinite(cur)
    B, n = x.shape[0], x.shape[1]
    was_active = status == active_code
    ok = ok & ~bad & was_active
    sel = ok.view(B, 1, 1)
    pre_new = torch.zeros(B, n, up_size, dtype=x.dtype, device=x.device)
    if maps is None:
        pre_new.index_copy_(-1, e_new, x)
    else:
        ii.update_(pre_new, -1, x, maps[0])
    pre_old = torch.zeros_like(pre_new)
    if maps is None:
        pre_old.index_copy_(-1, e_old, pre_old_src)
    else:
        ii.update_(pre_old, -1, pre_old_src, maps[1])
    tmv_new = torch.zeros(B, n, ut_size, dtype=x.dtype, device=x.device)
    if maps is None:
        tmv_new.index_copy_(-1, t_new, new_tmv)
    else:
        ii.update_(tmv_new, -1, new_tmv, maps[2])
    tmv_old = torch.zeros_like(tmv_new)
    if maps is None:
        tmv_old.index_copy_(-1, t_old, tmv_old_src)
    else:
        ii.update_(tmv_old, -1, tmv_old_src, maps[3])
    fail_code = torch.where(
        bad,
        torch.full_like(status, div_code),
        torch.full_like(status, contract_code),
    )
    return (
        torch.where(sel, pre_new, pre_old),
        torch.where(sel, cur, pre_rem_old),
        torch.where(sel, tmv_new, tmv_old),
        torch.where(sel, new_tmv_rem, tmv_rem_old),
        torch.where(ok | ~was_active, status, fail_code),
        ok,
    )


def _structural_picard(new_x0, sup_x0n, var_sups0, code, eng, order, cutoff_eps):
    """Use exact structural support supersets, independent of coefficient values.

    For each row the next support is initial support UNION the integrated
    tape output support. This retains all possible nonzero terms; a cutoff
    may zero a term, but no zero/nonzero host test may exclude another term.
    The optional route may retain more zeros than the measured policy.
    """
    from . import tape_kernels as tk
    from .glue import GLUE_MODE
    plans = _eng_cache(eng, "_structural_picard")
    injective = INJECTIVE_GLUE and ii.ENABLED and new_x0.is_cuda and GLUE_MODE != "compile"
    key = (_code_serial(code), order, cutoff_eps, injective)
    plan = plans.get(key)
    if plan is None:
        sup_x, var_sups, sweeps = sup_x0n, var_sups0, []
        serializable = new_x0.is_cuda and USE_TAPE_KERNELS and tk.valid_available()
        for i in range(1, order + 1):
            spec = specialize(code, var_sups, i - 1 if i > 1 else 1, sup_x, eng)
            sup_f = spec.sup_out_union
            sup_int, int_tgt, int_scale, _ = eng.integ(sup_f)
            sup_next, e_x0n, e_int = eng.union(sup_x0n, sup_int)
            var_sups = tuple(eng.union(initial, eng.integ(output)[0])[0]
                             for initial, output in zip(var_sups0, spec.out_sups))
            point_tape = None
            if serializable:
                tapes = _eng_cache(eng, "_ptape")
                if id(spec) not in tapes:
                    try:
                        tapes[id(spec)] = tk.serialize_point_tape(spec, eng, cutoff_eps, str(new_x0.device))
                    except ValueError:
                        tapes[id(spec)] = None
                point_tape = tapes[id(spec)]
                serializable = point_tape is not None
            maps = (eng.integ_plan(sup_f), eng.embed_plan(sup_x0n, sup_next),
                    eng.embed_plan(sup_int, sup_next)) if injective else None
            sweeps.append((spec, point_tape, int_scale, int_tgt, e_x0n, e_int,
                           sup_int.size, sup_next.size, maps))
            sup_x = sup_next
        plan = (sweeps, sup_x, var_sups, serializable)
        plans[key] = plan
    sweeps, sup_x, var_sups, serializable = plan
    def compute(initial):
        x = initial
        for spec, point_tape, int_scale, int_tgt, e_x0n, e_int, int_size, out_size, maps in sweeps:
            if point_tape is not None:
                field = tk.exec_point_tape(point_tape, x)
            else:
                field = _graphed_point(spec, x, eng, cutoff_eps)[0]
            x = _g_sweep(field, int_scale, int_tgt, initial, e_x0n, e_int, int_size, out_size, maps)
        return x
    if serializable and GLUE_MODE == "graph":
        x = graph_cache(eng, str(new_x0.device)).run(
            ("structural_picard", key, new_x0.shape[0]), compute, [new_x0],
        )
    else:
        x = compute(new_x0)
    return x, sup_x, var_sups


def _sr_linear_prepare(x0f, scalars, scalars_iv, pos_t, present, strict):
    """Pure SR inputs; callers own all persistent state and graph metadata."""
    B, n, _ = x0f.shape
    dt, dev = x0f.dtype, x0f.device
    x0p = torch.cat([x0f, torch.zeros(B, n, 1, dtype=dt, device=dev)], dim=-1)
    lin_a = x0p[..., pos_t]
    x0_other = x0f.clone()
    x0_other.index_fill_(-1, present, 0.0)
    phi_i = lin_a * scalars.unsqueeze(1)  # [B, n, n]
    phi_i_iv = None
    if strict:
        phi_i_iv = iv.mul(
            iv.from_point(lin_a),
            (
                scalars_iv.unsqueeze(1)
                if scalars_iv is not None
                else iv.from_point(scalars.unsqueeze(1))
            ),
        )
    return lin_a, x0_other, phi_i, phi_i_iv


def validation_code_for_order(code, order, eng):
    """Re-layout every order-dependent cache slot without changing instructions.

    Point Picard execution keeps its original code and degree. Only validated
    RHS evaluation and its matching refinement use this separate code object.
    """
    from dataclasses import replace
    cache = _eng_cache(eng, "_validation_codes")
    key = (_code_serial(code), order)
    if key not in cache:
        instructions, cursor = [], 0
        for ins in code.instrs:
            length = elem.cache_len(ins.op, order) if ins.op in {
                "sin", "cos", "exp", "rec", "div", "log", "sqrt"
            } else ins.cache_len
            instructions.append(replace(ins, cache_base=cursor if length else ins.cache_base,
                                        cache_len=length))
            cursor += length
        cache[key] = replace(code, instrs=instructions, order=order, n_cache=cursor)
    return cache[key]


def validation_engine_for_order(eng, order):
    """Separate validation tables; never change the solution engine or support.

    Graded-lex monomial IDs have a common prefix across basis orders. Retain
    all coefficients, and check that prefix before rehoming support metadata.
    Only fixed steps are qualified for the opt-in higher validation policy.
    """
    if order <= eng.tables.k:
        return eng
    if eng.step.lanes:
        raise ValueError("higher validation order currently requires fixed steps")
    cache = _eng_cache(eng, "_validation_engines")
    key = (order, eng.step.delta)
    if key not in cache:
        from .monomials import build_tables
        tables = build_tables(eng.tables.n, order)
        original = build_tables(eng.tables.n, eng.tables.k)
        if not torch.equal(tables.exponents[:original.T2], original.exponents):
            raise ValueError("validation monomial basis is not a common prefix")
        tables = tables.to(eng.device)
        cache[key] = sp.SparseEngine(tables, poly.build_step_tables(tables, eng.step.delta), eng.device)
    return cache[key]


def _retry_self_map(remainder, total, dimensions, difference, bad, cache, tails,
                    evaluate, max_attempts):
    """Propose smaller boxes; accept only a freshly evaluated full self-map.

    Intersecting a failed image is not itself a containment argument. Each
    proposed box contains zero, stays inside the configured box, and is
    independently validated. Successful and invalid lanes are never retried.
    Own graph outputs before the next replay can overwrite their storage.
    """
    ok = dimensions.all(-1) & ~bad
    eligible = ((remainder[..., 0] <= 0) & (remainder[..., 1] >= 0)).all(-1)
    eligible &= torch.isfinite(remainder).flatten(1).all(-1)
    active = ~ok & ~bad & eligible
    used = 0
    if max_attempts <= 0 or not bool(active.any()):
        return remainder, total, dimensions, difference, bad, cache, tails, used
    remainder, total, dimensions, difference, bad, cache, tails = (
        value.clone() for value in (remainder, total, dimensions, difference, bad, cache, tails)
    )
    def merge(mask, new, old):
        return torch.where(mask.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)
    for _ in range(max_attempts):
        lo = torch.maximum(remainder[..., 0], torch.minimum(total[..., 0], torch.zeros_like(total[..., 0])))
        hi = torch.minimum(remainder[..., 1], torch.maximum(total[..., 1], torch.zeros_like(total[..., 1])))
        proposal = merge(active, torch.stack((lo, hi), -1), remainder)
        new_total, new_dims, new_difference, new_bad, new_cache, new_tails = evaluate(proposal)
        # The evaluator must carry all domain failures; independently reject
        # any nonfinite returned value before allowing acceptance.
        for value in (new_total, new_difference, new_cache, new_tails):
            new_bad = new_bad | ~torch.isfinite(value).flatten(1).all(-1)
        remainder = proposal
        total, dimensions, difference, bad, cache, tails = (
            merge(active, new, old) for new, old in zip(
                (new_total, new_dims, new_difference, new_bad, new_cache, new_tails),
                (total, dimensions, difference, bad, cache, tails))
        )
        used += 1
        ok = dimensions.all(-1) & ~bad
        active = active & ~ok & ~bad
        if not bool(active.any()):
            break
    return remainder, total, dimensions, difference, bad, cache, tails, used


def advance_sparse(
    st: SparseState,
    code: CompiledODE,
    eng: sp.SparseEngine,
    sched: CompositionSchedule,
    settings: Settings,
    rem_est: torch.Tensor,
    sr: SymbolicRemainder | None = None,
) -> tuple[SparseState, torch.Tensor]:
    """One sparse integration step — flowpipe.advance stage-for-stage (same
    Continuous.cpp citations), on support-aligned tensors.

    Support lifecycle per step: pre/tmv supports are MEASURED (extracted) at
    the end of the previous advance; within the step every op's support is
    structural from the specialization; after each Picard sweep the iterate's
    actual per-row supports are re-measured (one small sync each) so the
    data shrinkage from cutoffs feeds the next specialization.
    """
    B, n, _ = st.pre.shape
    from .flowpipe import _lane_nonfinite
    dt, dev = st.pre.dtype, st.pre.device
    strict = settings.mode == "strict"
    center_mode = _os.environ.get("FLOWSTAR_CENTER_NORMALIZATION", "0")
    if center_mode not in {"0", "1"} or (center_mode == "1" and not strict):
        raise ValueError("centered normalization requires strict mode and a flag of 0 or 1")
    center_normalization = center_mode == "1"
    import os
    selfmap_retries = int(os.environ.get("FLOWSTAR_SELF_MAP_RETRIES", "0"))
    if not 0 <= selfmap_retries <= 8 or (selfmap_retries and not strict):
        raise ValueError("self-map retries require strict mode and a count in [0, 8]")
    weighted_mode = os.environ.get("FLOWSTAR_WEIGHTED_VALIDATION", "0")
    if weighted_mode not in {"0", "1"} or (weighted_mode == "1" and not strict):
        raise ValueError("weighted validation requires strict mode and a flag of 0 or 1")
    recenter_mode = os.environ.get("FLOWSTAR_RECENTER_VALIDATION", "0")
    if recenter_mode not in {"0", "1"} or (recenter_mode == "1" and not strict):
        raise ValueError("recentered validation requires strict mode and a flag of 0 or 1")
    early_weighted = os.environ.get("FLOWSTAR_EARLY_WEIGHTED", "0")
    if early_weighted not in {"0", "1"} or (early_weighted == "1" and not strict):
        raise ValueError("early weighted refinement requires strict mode and a flag of 0 or 1")
    if early_weighted == "1" and sr is not None and sr.qlen + 1 > sr.phi_buf.shape[0]:
        # Reclaim optional graph pools before SR's transient old+new buffers.
        # Public weighted outputs own copies; every graph can be recaptured.
        owners = [eng, *getattr(eng, "_validation_engines", {}).values()]
        for owner in owners:
            cache = getattr(owner, "_weighted_graphs", None)
            if cache is not None and cache._segs:
                torch.cuda.synchronize(dev)
                cache._segs.clear()
        if st.pre.is_cuda:
            torch.cuda.empty_cache()
    solution_order = VALIDATION_POLICY in {"solution_order", "solution_plus_one"}
    if solution_order and any(ins.op in {"log", "sqrt"} for ins in code.instrs):
        raise ValueError("solution_order currently excludes log/sqrt graph paths")
    deferred = VALIDATION_POLICY == "defer_polynomial"
    direct_leaves = VALIDATION_POLICY == "defer_direct_leaves"
    linear_leaves = VALIDATION_POLICY == "defer_linear_leaves"
    if (direct_leaves or linear_leaves or solution_order) and not strict:
        raise ValueError(f"{VALIDATION_POLICY} requires strict mode")
    if deferred and (not strict or code.has_elem):
        raise ValueError("defer_polynomial requires strict mode and a polynomial RHS")
    order = settings.order
    cutoff_eps = settings.cutoff
    tables, step = eng.tables, eng.step

    # (a) Endpoint substitution and constant split are a pure tensor region.
    # Strict cached reduction counts have already been validated; no GPU
    # scalar is read on graph replay. The changing pre/pre_rem are inputs.
    from .glue import GLUE_MODE
    injective = INJECTIVE_GLUE and ii.ENABLED and st.pre.is_cuda and GLUE_MODE != "compile"
    sup0f = eng.evalt(st.pre_sup, st.pre.dim())[0]
    def endpoint(pre, pre_rem):
        if strict:
            value, _support, error = sp.evaluate_time_end_s_with_roundoff(pre, eng, st.pre_sup)
            remainder = iv.add(pre_rem, error)
        else:
            value, _support = sp.evaluate_time_end_s(pre, eng, st.pre_sup)
            remainder = pre_rem
        center = value[..., 0].clone()
        value = value.clone()
        value[..., 0] = 0.0
        return value, remainder, center
    if GLUE_MODE == "graph" and st.pre.is_cuda:
        x0f, x0_rem, c0 = graph_cache(eng, str(dev)).run(
            ("endpoint", st.pre_sup, B, strict), endpoint, [st.pre, st.pre_rem]
        )
    else:
        x0f, x0_rem, c0 = endpoint(st.pre, st.pre_rem)
    sup0sp = sp.full_to_spatial_ids(tables.n, tables.k, sup0f)

    if sr is None:
        new_tmv, new_tmv_rem, sup_nt = compose_s(
            x0f, x0_rem, st.tmv, st.tmv_rem, sup0sp, st.tmv_sup,
            order, cutoff_eps, strict, sched, eng,
        )
    else:
        # (c-SR) Continuous.cpp:2151-2292 — linear part peeled off.
        sr_maps = _eng_cache(eng, "_sr_linear_maps")
        positions = sr_maps.get(sup0sp)
        if positions is None:
            var_sp = sched.var_image.cpu().numpy()
            ids0 = np.asarray(sup0sp.ids, dtype=np.int64)
            pos = np.searchsorted(ids0, var_sp)
            pos = np.where(
                (pos < len(ids0)) & (ids0[np.clip(pos, 0, len(ids0) - 1)] == var_sp),
                pos, len(ids0),
            )
            positions = (torch.from_numpy(pos).to(dev),
                         torch.from_numpy(pos[pos < len(ids0)]).to(dev))
            sr_maps[sup0sp] = positions
        pos_t, present = positions
        def prepare_linear(value, scales, *scales_iv):
            return _sr_linear_prepare(
                value, scales, scales_iv[0] if scales_iv else None,
                pos_t, present, strict,
            )
        operands = [x0f, sr.scalars]
        if sr.scalars_iv is not None:
            operands.append(sr.scalars_iv)
        if GLUE_MODE == "graph" and x0f.is_cuda:
            linear = graph_cache(eng, str(dev)).run(
                ("sr_linear_prepare", sup0sp, B, strict,
                 sr.scalars_iv is not None, dt),
                prepare_linear, operands, keepalive=positions,
            )
        else:
            linear = prepare_linear(*operands)
        lin_a, x0_other, phi_i, phi_i_iv = linear
        j_lin = propagate(sr, phi_i, strict=strict, phi_i_iv=phi_i_iv)  # [B, n, 2]

        if sr.queue_len > 0:
            comp_c, comp_rem, sup_c = compose_s(
                x0_other, x0_rem, st.tmv, st.tmv_rem, sup0sp, st.tmv_sup,
                order, cutoff_eps, strict, sched, eng,
            )
            sup_nt = eng.union(sup_c, st.tmv_sup)[0]
            def combine_linear(linear, old_tmv, comp, comp_remainder, history):
                lin_part = torch.einsum("bij,bjt->bit", linear, old_tmv)
                value, _support = sp.add_aligned(
                    comp, sup_c, lin_part, st.tmv_sup, eng, interval=False
                )
                fresh = comp_remainder
                if strict:
                    abs_dot = torch.einsum("bij,bjt->bit", linear.abs(), old_tmv.abs())
                    lin_radius = dot_error_bound(abs_dot, n)
                    lin_error_src = torch.stack((-lin_radius, lin_radius), dim=-1)
                    lin_error = sp.embed_s(lin_error_src, eng, st.tmv_sup, sup_nt, interval=True)
                    exact_add, add_sup = sp.add_aligned(
                        iv.from_point(comp), sup_c, iv.from_point(lin_part), st.tmv_sup,
                        eng, interval=True,
                    )
                    if add_sup is not sup_nt and add_sup.ids != sup_nt.ids:
                        raise RuntimeError("strict sparse addition support mismatch")
                    add_error = iv.sub(exact_add, iv.from_point(value))
                    coeff_error = iv.add(lin_error, add_error)
                    fresh = iv.add(fresh, sp.range_spatial_iv_s(coeff_error, eng, sup_nt))
                return value, fresh, iv.add(fresh, history)
            linear_inputs = [lin_a, st.tmv, comp_c, comp_rem, j_lin]
            if GLUE_MODE == "graph" and st.pre.is_cuda:
                new_tmv, j_new, new_tmv_rem = graph_cache(eng, str(dev)).run(
                    ("combine_linear", st.tmv_sup, sup_c, B, strict),
                    combine_linear, linear_inputs,
                )
            else:
                new_tmv, j_new, new_tmv_rem = combine_linear(*linear_inputs)
        else:
            new_tmv, new_tmv_rem, sup_nt = compose_s(
                x0f, x0_rem, st.tmv, st.tmv_rem, sup0sp, st.tmv_sup,
                order, cutoff_eps, strict, sched, eng,
            )
            j_new = new_tmv_rem
        if not strict:
            sr.append_j(j_new)

    # (e) Diagonal preconditioning + new_x0 assembly — compiled glue region
    #     (_g_precond; glue.py rationale). The SR INITIAL_SIMP branch stays
    #     eager (it mutates sr.scalars and is SR-path-only).
    gl = glue_cache(eng)
    aux = _eng_cache(eng, "_precond_aux")
    ax = aux.get(("x0n",))
    if ax is None:
        var_sp_l = tuple(int(v) for v in sched.var_image.cpu().numpy())
        var_full_l = tuple(int(v) for v in tables.spatial_index.cpu().numpy()[list(var_sp_l)])
        sup_x0n = sp.make_support(tables.n, tables.k, False, var_full_l)
        ax = (
            sup_x0n,
            torch.tensor([sup_x0n.ids.index(v) for v in var_full_l], device=dev),
            torch.arange(n, device=dev),
            tuple(sp.make_support(tables.n, tables.k, False, (0, var_full_l[i]))
                  for i in range(n)),
        )
        aux[("x0n",)] = ax
    sup_x0n, pos_full, arange_n, var_sups0 = ax
    # The normalization and all fresh-error charges form one pure tensor
    # region. Its graph reads explicit inputs; it never captures an SR object.
    from .glue import GLUE_MODE
    use_sr = sr is not None
    capture_normalize = GLUE_MODE == "graph" and new_tmv.is_cuda

    if center_normalization and (not sup_nt.ids or sup_nt.ids[0] != 0):
        raise ValueError("centered normalization requires a constant slot")

    def normalize(pre_tmv, pre_rem, center, fresh_j):
        if center_normalization:
            pre_tmv, pre_rem, center, fresh_j = _center_preconditioning(
                pre_tmv, pre_rem, center, fresh_j, eng.cat(sup_nt), use_sr
            )
        precond_args = (pre_tmv, pre_rem, eng.cat(sup_nt), center,
                        pos_full, arange_n, sup_x0n.size, strict)
        if capture_normalize:
            tmv, rem, radius, point, inverse, x0 = _g_precond(*precond_args)
        else:
            tmv, rem, radius, point, inverse, x0 = gl.run(
                ("precond", sup_nt, B, strict), _g_precond, *precond_args
            )
        inverse_iv = None
        if strict:
            safe_radius = torch.where(point, torch.ones_like(radius), radius)
            inverse_iv, _bad_inverse = iv.rec(iv.from_point(safe_radius))
            inverse_iv = torch.where(
                point.unsqueeze(-1), iv.from_point(torch.ones_like(radius)), inverse_iv
            )
            exact_scaled = iv.mul(iv.from_point(pre_tmv), inverse_iv.unsqueeze(-2))
            coefficient_error = iv.sub(exact_scaled, iv.from_point(tmv))
            rem = iv.mul(pre_rem, inverse_iv)
            scale_error = sp.range_spatial_iv_s(coefficient_error, eng, sup_nt)
            rem = iv.add(rem, scale_error)
            if use_sr:
                fresh_j = iv.add(fresh_j, iv.mul_point(scale_error, radius))
        scalars = scalars_iv = None
        if use_sr:
            scalars = torch.where(point, torch.zeros_like(inverse), inverse)
            if strict:
                scalars_iv = torch.where(
                    point.unsqueeze(-1), iv.from_point(torch.zeros_like(radius)), inverse_iv
                )
            tmv, dropped = sp.cutoff_spatial_s(
                tmv, eng, sup_nt, settings.cutoff if strict else INITIAL_SIMP
            )
            rem = iv.add(rem, dropped)
            if strict:
                fresh_j = iv.add(fresh_j, iv.mul_point(dropped, radius))
        return tmv, rem, x0, scalars, scalars_iv, fresh_j

    fresh = j_new if use_sr else torch.zeros_like(new_tmv_rem)
    operands = [new_tmv, new_tmv_rem, c0, fresh]
    if capture_normalize:
        normalized = graph_cache(eng, str(dev)).run(
            ("normalize", sup_nt, B, strict, use_sr, settings.cutoff, center_normalization),
            normalize, operands,
        )
    else:
        normalized = normalize(*operands)
    new_tmv, new_tmv_rem, new_x0, scalars, scalars_iv, j_new = normalized
    if use_sr:
        # Graph outputs are reusable workspace; persistent SR scalars must
        # survive the next normalization, including a failed speculative step.
        sr.scalars = scalars.clone() if capture_normalize else scalars
        if strict:
            sr.scalars_iv = scalars_iv.clone() if capture_normalize else scalars_iv
            sr.append_j(j_new)

    # (f) Polynomial Picard sweeps: optional structural supersets remove
    # coefficient-dependent host synchronization and allow one complete graph.
    if SUPPORT_POLICY == "structural":
        x, sup_x, var_sups = _structural_picard(
            new_x0, sup_x0n, var_sups0, code, eng, order, cutoff_eps
        )
    else:
        x, sup_x = new_x0, sup_x0n
        var_sups = var_sups0
        for i in range(1, order + 1):
            k_i = i - 1 if i > 1 else 1
            spec = specialize(code, var_sups, k_i, sup_x, eng)
            f, sup_f = _graphed_point(spec, x, eng, cutoff_eps)  # [B, n, Sf]
            sup_int, int_tgt, int_scale, _siv = eng.integ(sup_f)
            sup_x_new, e_x0n, e_int = eng.union(sup_x0n, sup_int)
            maps = (eng.integ_plan(sup_f), eng.embed_plan(sup_x0n, sup_x_new),
                    eng.embed_plan(sup_int, sup_x_new)) if injective else None
            x = gl.run(
                ("sweep", sup_f, sup_x0n, B, injective), _g_sweep,
                f, int_scale, int_tgt, new_x0, e_x0n, e_int,
                sup_int.size, sup_x_new.size, maps,
            )
            sup_x = sup_x_new
            measured = _extract_row_sups(x, sup_x, eng)  # SYNC (one per sweep)
            var_sups = _accum_sups(eng, ("vs", _code_serial(code), i), measured)
            # The accumulated union may exceed x's current layout; re-lay x onto
            # the union of both so every var_sup stays a subset of sup_x.
            sup_need = var_sups[0]
            for vsup in var_sups[1:]:
                sup_need, _, _ = eng.union(sup_need, vsup)
            if sup_need is not sup_x and not set(sup_need.ids) <= set(sup_x.ids):
                sup_x2, ex, _ = eng.union(sup_x, sup_need)
                x2 = torch.zeros(B, n, sup_x2.size, dtype=dt, device=dev)
                x2.index_copy_(-1, ex, x)
                x, sup_x = x2, sup_x2


    # (g)-(k) validated pass + refinement.
    validation_order = order + 1 if VALIDATION_POLICY == "solution_plus_one" else (order if solution_order else order - 1)
    v_eng = validation_engine_for_order(eng, validation_order)
    def validation_support(sup):
        return sup if v_eng is eng else sp.make_support(sup.n, v_eng.tables.k, sup.spatial, sup.ids)
    v_sup_x, v_sup_x0n = validation_support(sup_x), validation_support(sup_x0n)
    v_var_sups = tuple(validation_support(sup) for sup in var_sups)
    validation_code = validation_code_for_order(code, validation_order, v_eng) if solution_order else code
    x_rem = rem_est.clone()  # [B, n, 2]
    bad = torch.zeros(B, dtype=torch.bool, device=dev)
    tabs = elem.elem_tables(validation_code.order, str(dev)) if code.has_elem else None
    if code.has_elem and code.needs_bad:
        pass  # bad is always supplied below
    spec_v = specialize(validation_code, v_var_sups, validation_order, v_sup_x, v_eng,
                        validated_full_leaves=deferred, validated_direct_leaves=direct_leaves,
                        validated_linear_leaves=linear_leaves)
    f_iv, f_rem, cache, tails, bad = _graphed_valid(
        spec_v, validation_code, x, x_rem, bad, v_eng, cutoff_eps, tabs
    )
    # (h)-(j) interval integration + difference bound + contraction inputs
    # — one compiled glue region (_g_validpost).
    sup_fv = spec_v.sup_out_union
    sup_ivo, int_tgt_v, _isc, int_scale_iv = v_eng.integ(sup_fv)
    sup_t, e_int, e_x0 = v_eng.union(sup_ivo, v_sup_x0n)
    sup_d, e_t, e_xx = v_eng.union(sup_t, v_sup_x)
    dts = _eng_cache(v_eng, "_dtiv")
    dt_iv = dts.get((B, n))
    if dt_iv is None:
        dt_iv = step.dt(2).expand(B, n, 2).contiguous()
        dts[(B, n)] = dt_iv
    post_maps = (v_eng.integ_plan(sup_fv), v_eng.embed_plan(sup_ivo, sup_t),
                 v_eng.embed_plan(v_sup_x0n, sup_t), v_eng.embed_plan(sup_t, sup_d),
                 v_eng.embed_plan(v_sup_x, sup_d)) if injective else None
    def validation_post(x_value, initial, remainder, field, field_rem, ranges, strict_tails, bad_input):
        bad_result = bad_input | _lane_nonfinite(
            x_value, initial, remainder, field, field_rem, ranges, strict_tails
        )
        post_args = (field, field_rem, initial, x_value, remainder, int_scale_iv, int_tgt_v,
                     e_int, e_x0, e_t, e_xx, v_eng.factor(sup_d), dt_iv,
                     sup_ivo.size, sup_t.size, sup_d.size, post_maps)
        if GLUE_MODE == "graph" and x_value.is_cuda:
            total_value, dimensions, difference = _g_validpost(*post_args)
        else:
            total_value, dimensions, difference = gl.run(
                ("validpost", sup_fv, v_sup_x0n, v_sup_x, B, injective), _g_validpost, *post_args
            )
        bad_result = bad_result | _lane_nonfinite(total_value, difference)
        return total_value, dimensions, difference, bad_result
    validation_inputs = [x, new_x0, x_rem, f_iv, f_rem, cache, tails, bad]
    if GLUE_MODE == "graph" and x.is_cuda:
        total, ok_dims, int_diff, bad = graph_cache(v_eng, str(dev)).run(
            ("validpost_checked", sup_fv, v_sup_x0n, v_sup_x, B,
             tuple(cache.shape), tuple(tails.shape), injective), validation_post, validation_inputs,
        )
    else:
        total, ok_dims, int_diff, bad = validation_post(*validation_inputs)
    initial_ok = ok_dims.all(dim=-1) & ~bad
    if selfmap_retries:
        def reevaluate(remainder):
            fresh_bad = torch.zeros(B, dtype=torch.bool, device=dev)
            field, field_rem, ranges, strict_tails, fresh_bad = _graphed_valid(
                spec_v, validation_code, x, remainder, fresh_bad, v_eng, cutoff_eps, tabs
            )
            fresh_total, fresh_dims, fresh_diff, fresh_bad = validation_post(
                x, new_x0, remainder, field, field_rem, ranges, strict_tails, fresh_bad
            )
            return fresh_total, fresh_dims, fresh_diff, fresh_bad, ranges, strict_tails
        x_rem, total, ok_dims, int_diff, bad, cache, tails, used = _retry_self_map(
            x_rem, total, ok_dims, int_diff, bad, cache, tails, reevaluate, selfmap_retries
        )
        if used:
            records = getattr(eng, "selfmap_diagnostics", None)
            if records is None:
                records = eng.selfmap_diagnostics = []
            records.append(dict(attempts=used, initially_failed=int((~initial_ok).sum()),
                                recovered=int((~initial_ok & ok_dims.all(-1) & ~bad).sum())))
    ok = ok_dims.all(dim=-1) & ~bad
    cache_state = None
    if settings.refinement_callback is not None:
        from .flowpipe import (
            RefinementCacheState,
            _refinement_cache_id,
            _trace_refinement,
        )

        cache_state = RefinementCacheState(
            cache_id=_refinement_cache_id(cache, "range-cache"),
            tails_id=_refinement_cache_id(tails, "strict-tails"),
            generation=0,
            owner_remainder=total.clone(),
            source_remainder=x_rem.clone(),
        )
        for lane in range(B):
            attempted_step = (
                step.delta
                if step.lanes == 0
                else float(step.deltas[lane].item())
            )
            _trace_refinement(
                settings,
                {
                    "event": "initial_self_map",
                    "lane": lane,
                    "initial_self_map_ok": bool(ok[lane].item()),
                    "input_remainder_vector": x_rem[lane].detach().cpu().tolist(),
                    "proposal_interval": total[lane].detach().cpu().tolist(),
                    "attempted_step": attempted_step,
                    "subset_by_component": ok_dims[lane].detach().cpu().tolist(),
                    "subset_margin_by_component": {
                        "lower": (
                            total[lane, :, 0] - x_rem[lane, :, 0]
                        ).detach().cpu().tolist(),
                        "upper": (
                            x_rem[lane, :, 1] - total[lane, :, 1]
                        ).detach().cpu().tolist(),
                    },
                    "bad_nonfinite_mask": bool(bad[lane].item()),
                    "input_generation": 0,
                    "proposal_generation": 0,
                    "cache_id": cache_state.cache_id,
                    "tails_id": cache_state.tails_id,
                },
            )
    cur, bad = _refine_dispatch(validation_code, total, ok, bad, cache, tails, strict,
                                int_diff, step, settings, v_eng, cache_state)

    if early_weighted == "1" and bool((ok & ~bad & (st.status == ACTIVE)).any()):
        from .weighted_validation import refine_accepted
        trace = (lambda event: _trace_refinement(settings, event)) if settings.refinement_callback is not None else None
        cur, diagnostics = refine_accepted(
            validation_code, x, v_sup_x, new_x0, v_sup_x0n, v_var_sups,
            rem_est, cur, ok & ~bad & (st.status == ACTIVE), v_eng, cutoff_eps, trace=trace,
        )
        records = getattr(eng, "early_weighted_diagnostics", None)
        if records is None:
            records = eng.early_weighted_diagnostics = []
        records.append(diagnostics)

    if weighted_mode == "1" and bool((~ok & ~bad & (st.status == ACTIVE)).any()):
        from .weighted_validation import validate_failed
        trace = (lambda event: _trace_refinement(settings, event)) if settings.refinement_callback is not None else None
        recovered, weighted_rem, diagnostics = validate_failed(
            validation_code, x, v_sup_x, new_x0, v_sup_x0n, v_var_sups,
            rem_est, ~ok & ~bad & (st.status == ACTIVE), v_eng, cutoff_eps, trace=trace,
        )
        # The weighted proof has no compatible constant-remainder replay cache.
        # Ordinary accepted lanes have already been refined above and stay intact.
        cur = torch.where(recovered[:, None, None], weighted_rem, cur)
        ok = ok | recovered
        records = getattr(eng, "weighted_diagnostics", None)
        if records is None:
            records = eng.weighted_diagnostics = []
        records.append(diagnostics)

    if recenter_mode == "1" and bool((~ok & ~bad & (st.status == ACTIVE)).any()):
        from .weighted_validation import validate_recentered
        trace = (lambda event: _trace_refinement(settings, event)) if settings.refinement_callback is not None else None
        recovered, recentered_x, recentered_rem, diagnostics = validate_recentered(
            validation_code, x, v_sup_x, new_x0, v_sup_x0n, v_var_sups,
            rem_est, ~ok & ~bad & (st.status == ACTIVE), v_eng, cutoff_eps, trace=trace,
        )
        # Both parts of this new proof must be emitted together. A recovered
        # remainder paired with the old x would have no containment guarantee.
        x = recentered_x
        cur = torch.where(recovered[:, None, None], recentered_rem, cur)
        ok = ok | recovered
        records = getattr(eng, "recenter_diagnostics", None)
        if records is None:
            records = eng.recenter_diagnostics = []
        records.append(diagnostics)

    # (l) Emit — compiled glue region (_g_emit; frozen-lane union merge).
    sup_up, e_new, e_old = eng.union(sup_x, st.pre_sup)
    sup_ut, t_new, t_old = eng.union(sup_nt, st.tmv_sup)
    emit_maps = (eng.embed_plan(sup_x, sup_up), eng.embed_plan(st.pre_sup, sup_up),
                 eng.embed_plan(sup_nt, sup_ut), eng.embed_plan(st.tmv_sup, sup_ut)) if injective else None
    pre, pre_rem, tmv, tmv_rem, status, ok = gl.run(
        ("emit", sup_x, st.pre_sup, sup_nt, st.tmv_sup, B, injective), _g_emit,
        x, st.pre, new_tmv, st.tmv, cur, st.pre_rem, new_tmv_rem, st.tmv_rem,
        st.status, ok, bad, e_new, e_old, t_new, t_old,
        sup_up.size, sup_ut.size, ACTIVE, FAILED_DIV, FAILED_CONTRACTION, emit_maps,
    )
    if GLUE_MODE == "graph" and pre.is_cuda:
        # The public advance result owns its storage. A later replay of the
        # same emit graph must not change old states or acceptance flags.
        pre, pre_rem, tmv, tmv_rem, status, ok = (
            value.clone() for value in (pre, pre_rem, tmv, tmv_rem, status, ok)
        )
    out = SparseState(
        pre=pre, pre_sup=sup_up, pre_rem=pre_rem,
        tmv=tmv, tmv_sup=sup_ut, tmv_rem=tmv_rem, status=status,
    )
    return out, ok


def prune_state(st: SparseState, eng: sp.SparseEngine) -> SparseState:
    """Compact pre/tmv, measuring supports only when their layouts can grow.

    Called once per advance (driver loop) so the emit-stage unions cannot
    ratchet the layouts upward across steps. Supports are accumulated
    monotonically (see _accum_sups) so downstream spec keys stay stable.
    """
    accumulated = getattr(eng, "_acc_sups", {}).get(("state",), (None, None))
    # A measured support is a subset of its layout (which includes id 0).
    # Union with an identical accumulated support cannot change that support.
    sup_p, sup_t = _accum_sups(eng, ("state",), (
        st.pre_sup if accumulated[0] == st.pre_sup else
        sp.extract_support(st.pre, eng.tables, False, base=st.pre_sup),
        st.tmv_sup if accumulated[1] == st.tmv_sup else
        sp.extract_support(st.tmv, eng.tables, True, base=st.tmv_sup),
    ))
    gp = eng.embed_idx(sup_p, st.pre_sup)
    gt = eng.embed_idx(sup_t, st.tmv_sup)
    return SparseState(
        pre=st.pre[..., gp].contiguous(), pre_sup=sup_p, pre_rem=st.pre_rem,
        tmv=st.tmv[..., gt].contiguous(), tmv_sup=sup_t, tmv_rem=st.tmv_rem,
        status=st.status,
    )
