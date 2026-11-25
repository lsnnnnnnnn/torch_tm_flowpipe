"""Union-support sparse representation: supports, cached index tables, ops (M10).

WHY (profiled, comparison/REPORT.md + PROGRESS M10): the dense representation
multiplies full basis vectors — T = 171..1716 slots at CROWN-Reach shapes —
while the measured union support (nonzero coefficient slots across the whole
batch) is 28..50. Dense pair products therefore move 100-1000x more memory
than the data contains. This module carries coefficients only on an explicit
support: a sorted subset of the global monomial basis.

SOUNDNESS/PARITY ARGUMENT (load-bearing, tested by the differential suite):
a support is always a SUPERSET of the nonzero slots, so the terms a sparse op
never touches are exact zeros in the dense computation. Removing exact-zero
terms from an RN sum leaves every partial sum bitwise identical (x + 0.0 == x
for every finite x; the only effect is the sign of a zero result, which no
downstream consumer distinguishes — rec() guards zero denominators
explicitly). Hence:
  * tier-P results are BITWISE equal to the dense path (modulo +/-0.0)
    wherever the backend reduces segments sequentially — exact on CPU and
    asserted there by the differential suite; on CUDA, segment_reduce/GEMM
    blocking is shape-dependent, so the cross-representation contract is
    ulp-level agreement (each path remains independently sound: the Rump
    bounds cover ANY reduction order);
  * tier-I endpoint sums follow the same argument and the Rump reduction
    bound shrinks (fewer terms => smaller m) — remainders get EQUAL OR
    TIGHTER on CPU, ulp-equivalent on CUDA. With the fused kernels
    (cuda_kernels.py) the interval results are strictly tighter still
    (true directed rounding, no error-model slack).

Representation invariants:
  * `Support.ids` is a strictly ascending tuple of GLOBAL monomial indices
    into the degree-major graded basis of `MonomialTables` — either the full
    (t, x1..xn) product basis (spatial=False, ids < T2) or the spatial t-free
    basis (spatial=True, ids < Ts2). Ascending global order means ascending
    graded order, so truncation to degree <= d is the contiguous local prefix
    [: prefix_by_deg[d]] — the same prefix property the dense layout has.
  * id 0 (the constant monomial) is ALWAYS in the support (slot 0). This
    keeps constant splitting/injection O(1) and makes empty supports
    impossible (every op has a well-formed operand).
  * A coefficient tensor "on" a support has its last data dim == len(ids),
    aligned entry-for-entry: coeffs[..., s] is the coefficient of monomial
    ids[s].

All index tables are built once on CPU with numpy (integer combinatorics
only) and cached by value keys; device copies are cached per device. The pair
enumeration is a-major in ascending global ids, stable-sorted by output slot —
the SAME in-segment accumulation order as the dense tables restricted to the
support, which is what makes the bitwise claim above hold per partial sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from . import cuda_kernels as ck
from . import injective_index as ii
from . import interval as iv
from .monomials import MonomialTables, build_tables
from .polynomial import StepTables, _segment_sum_with
from .rounding import dot_error_bound, next_down, next_up

# ---------------------------------------------------------------------------
# Support objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Support:
    """A sorted monomial-id subset of one basis family of (n, k) tables.

    Attributes:
        n, k: the MonomialTables family this indexes into.
        spatial: True -> ids index the spatial (t-free) degree<=2k basis
            (0..Ts2-1); False -> the full product basis (0..T2-1).
        ids [S]: strictly ascending global indices; ids[0] == 0 always.
        degs [S]: total degree per id (spatial ids: spatial degree).
        prefix_by_deg [2k+2]: prefix_by_deg[d] = #ids with degree <= d-1 —
            the local slice [: prefix_by_deg[d+1]] is exactly "degree <= d"
            (mirrors MonomialTables.prefix_len semantics).
    """

    n: int
    k: int
    spatial: bool
    ids: tuple
    degs: tuple
    prefix_by_deg: tuple

    @property
    def size(self) -> int:
        return len(self.ids)

    def keep_len(self, order: int) -> int:
        """Local length of the degree <= order prefix (contiguous by the
        ascending-graded-id invariant)."""
        return self.prefix_by_deg[min(order + 1, len(self.prefix_by_deg) - 1)]


# Value-stable serial per canonical Support: the segment-sum memo in
# polynomial._segment_sum_with keys by a string tag, and id()-based tags could
# collide after cache eviction recycles addresses (the historical seglen-cache
# bug class). Serials are handed out once per distinct Support value.
_sup_serial: dict = {}


def _serial(sup: "Support") -> int:
    r = _sup_serial.get(sup)
    if r is None:
        r = len(_sup_serial)
        _sup_serial[sup] = r
    return r


@lru_cache(maxsize=4096)
def make_support(n: int, k: int, spatial: bool, ids: tuple) -> Support:
    """Canonical (cached) Support for an id tuple; injects id 0, sorts, and
    precomputes degree metadata. ids entries must be valid for the basis."""
    tables = build_tables(n, k)  # CPU instance (lru-cached at build_tables)
    if spatial:
        # Spatial degrees: total degree of the spatial-2k basis entries.
        sp_full = tables.spatial_index.numpy()
        # spatial-2k index -> full index: rebuild the map (cheap, cached here).
        tdeg = tables.t_deg.numpy()
        full_sp2 = np.nonzero(tdeg == 0)[0]  # [Ts2]
        deg_all = tables.total_deg.numpy()[full_sp2]  # [Ts2] spatial degrees
        _ = sp_full
    else:
        deg_all = tables.total_deg.numpy()  # [T2]
    id_arr = np.unique(np.concatenate(([0], np.asarray(ids, dtype=np.int64))))
    if id_arr[-1] >= deg_all.shape[0]:
        raise ValueError("support id out of basis range")
    degs = deg_all[id_arr]  # [S]
    prefix = np.zeros(2 * k + 2, dtype=np.int64)
    prefix[1:] = np.cumsum(np.bincount(degs, minlength=2 * k + 1))
    return Support(
        n=n, k=k, spatial=spatial,
        ids=tuple(int(v) for v in id_arr),
        degs=tuple(int(v) for v in degs),
        prefix_by_deg=tuple(int(v) for v in prefix),
    )


def full_support(tables: MonomialTables, order: int, spatial: bool) -> Support:
    """The complete degree <= order basis as a Support (dense fallback /
    initial supports)."""
    n, k = tables.n, tables.k
    m = tables.sp_prefix_len_cpu[order + 1] if spatial else tables.prefix_len_cpu[order + 1]
    return make_support(n, k, spatial, tuple(range(m)))


def extract_support(
    coeffs: torch.Tensor, tables: MonomialTables, spatial: bool, base: Support | None = None
) -> Support:
    """Measure the union support of a DENSE-prefix or support-aligned tensor.

    coeffs [..., M] (dense prefix of the basis) when base is None, else
    [..., S] aligned to `base` — returns the Support of slots where ANY
    element over the leading dims is nonzero. ONE host-device sync (the
    boolean mask transfer) — callers batch extractions where possible.
    """
    mask = (coeffs != 0).reshape(-1, coeffs.shape[-1]).any(dim=0)  # [M] bool
    idx = np.flatnonzero(mask.cpu().numpy())  # One fixed-size transfer; CPU indices.
    if base is not None:
        base_ids = np.asarray(base.ids, dtype=np.int64)
        ids = base_ids[idx]
    else:
        ids = idx
    return make_support(tables.n, tables.k, spatial, tuple(int(v) for v in ids))


# ---------------------------------------------------------------------------
# CPU index-table builders (all cached by value keys)
# ---------------------------------------------------------------------------


def _exps_for(sup: Support) -> np.ndarray:
    """Exponent rows [S, n+1] (full) or [S, n] (spatial) of a support's ids."""
    tables = build_tables(sup.n, sup.k)
    ids = np.asarray(sup.ids, dtype=np.int64)
    if sup.spatial:
        full_sp2 = np.nonzero(tables.t_deg.numpy() == 0)[0]  # [Ts2] full idx
        return tables.exponents.numpy()[full_sp2[ids]][:, 1:]  # [S, n]
    return tables.exponents.numpy()[ids]  # [S, n+1]


def _radix_keys(exps: np.ndarray, radix: int) -> np.ndarray:
    """Mixed-radix monomial keys (linear in exponents, as in monomials.py)."""
    weights = radix ** np.arange(exps.shape[1], dtype=np.int64)
    return exps @ weights


@lru_cache(maxsize=4096)
def pair_tables(sup_a: Support, sup_b: Support) -> tuple:
    """Sparse product tables for a * b over supports (degree <= k inputs).

    Returns (sup_out, pair_ia [P] np, pair_jb [P] np, seg_len [S_out] np):
    every ordered pair (local ia in a, local jb in b); the output monomial is
    ids_a[ia] (+) ids_b[jb] (exponent sum), always degree <= 2k. Pairs are
    enumerated a-major (ascending ids) and stable-sorted by output slot —
    the dense pair order restricted to the support (bitwise-parity contract,
    module docstring). sup_out contains EXACTLY the reachable products (plus
    id 0 by the Support invariant).
    """
    if (sup_a.n, sup_a.k, sup_a.spatial) != (sup_b.n, sup_b.k, sup_b.spatial):
        raise ValueError("pair_tables: mismatched support families")
    n, k = sup_a.n, sup_a.k
    tables = build_tables(n, k)
    if sup_a.degs and max(sup_a.degs) > k or sup_b.degs and max(sup_b.degs) > k:
        raise ValueError("pair_tables: inputs must be degree <= k supports")

    exps_a = _exps_for(sup_a)  # [Sa, v]
    exps_b = _exps_for(sup_b)  # [Sb, v]
    radix = 2 * k + 1
    ka = _radix_keys(exps_a, radix)  # [Sa]
    kb = _radix_keys(exps_b, radix)  # [Sb]
    pair_keys = (ka[:, None] + kb[None, :]).reshape(-1)  # [Sa*Sb] a-major

    # Global basis keys for the product family.
    if sup_a.spatial:
        full_sp2 = np.nonzero(tables.t_deg.numpy() == 0)[0]
        gexps = tables.exponents.numpy()[full_sp2][:, 1:]  # [Ts2, n]
    else:
        gexps = tables.exponents.numpy()  # [T2, n+1]
    gkeys = _radix_keys(gexps, radix)
    gorder = np.argsort(gkeys)
    pos = np.searchsorted(gkeys[gorder], pair_keys)
    out_global = gorder[np.clip(pos, 0, gkeys.shape[0] - 1)]  # [P]
    assert (gkeys[out_global] == pair_keys).all(), "sparse product lookup failed"

    out_ids = np.unique(out_global)  # ascending global ids
    sup_out = make_support(n, k, sup_a.spatial, tuple(int(v) for v in out_ids))
    # Local output slot per pair (sup_out.ids includes id 0 by invariant).
    out_ids_full = np.asarray(sup_out.ids, dtype=np.int64)
    out_local = np.searchsorted(out_ids_full, out_global)  # [P]

    perm = np.argsort(out_local, kind="stable")  # keeps a-major order in segs
    sa = exps_a.shape[0]
    pair_ia = perm // len(sup_b.ids)
    pair_jb = perm % len(sup_b.ids)
    seg_len = np.bincount(out_local[perm], minlength=len(sup_out.ids))
    _ = sa
    return sup_out, pair_ia, pair_jb, seg_len


@lru_cache(maxsize=4096)
def union_maps(sup_a: Support, sup_b: Support) -> tuple:
    """Union support + local embedding index arrays.

    Returns (sup_u, emb_a [Sa] np, emb_b [Sb] np): position of each of a's /
    b's slots inside the union's local layout.
    """
    if (sup_a.n, sup_a.k, sup_a.spatial) != (sup_b.n, sup_b.k, sup_b.spatial):
        raise ValueError("union_maps: mismatched support families")
    ids_a = np.asarray(sup_a.ids, dtype=np.int64)
    ids_b = np.asarray(sup_b.ids, dtype=np.int64)
    ids_u = np.union1d(ids_a, ids_b)
    sup_u = make_support(sup_a.n, sup_a.k, sup_a.spatial, tuple(int(v) for v in ids_u))
    ids_uf = np.asarray(sup_u.ids, dtype=np.int64)
    return sup_u, np.searchsorted(ids_uf, ids_a), np.searchsorted(ids_uf, ids_b)


@lru_cache(maxsize=4096)
def subset_map(sup_small: Support, sup_big: Support) -> np.ndarray:
    """Local positions of sup_small's ids inside sup_big (must be a superset)."""
    ids_s = np.asarray(sup_small.ids, dtype=np.int64)
    ids_b = np.asarray(sup_big.ids, dtype=np.int64)
    pos = np.searchsorted(ids_b, ids_s)
    if not (ids_b[np.clip(pos, 0, len(ids_b) - 1)] == ids_s).all():
        raise ValueError("subset_map: sup_small is not a subset of sup_big")
    return pos


@lru_cache(maxsize=4096)
def integrate_maps(sup: Support) -> tuple:
    """t-integration maps for a FULL-basis support of degree <= 2k-1.

    Returns (sup_out, tgt_local [S] np): monomial ids[s] integrates to
    sup_out.ids[tgt_local[s]] (injective t-bump; scale = int_scale[ids[s]]).
    """
    if sup.spatial:
        raise ValueError("integrate_maps: integration needs the full basis")
    tables = build_tables(sup.n, sup.k)
    ids = np.asarray(sup.ids, dtype=np.int64)
    int_map = tables.int_map.numpy()[ids]  # [S] global targets
    if (int_map < 0).any():
        raise ValueError("integrate_maps: input support exceeds degree 2k-1")
    out_ids = np.unique(int_map)
    # Include id 0 per Support invariant; targets never map TO 0 (t-degree
    # >= 1), so slot 0 of the result is structurally zero — harmless.
    sup_out = make_support(sup.n, sup.k, False, tuple(int(v) for v in out_ids))
    ids_of = np.asarray(sup_out.ids, dtype=np.int64)
    return sup_out, np.searchsorted(ids_of, int_map)


@lru_cache(maxsize=4096)
def eval_time_maps(sup: Support) -> tuple:
    """t := delta substitution maps for a FULL-basis support.

    Returns (sup_out_spatialfull, tgt_local [S] np, t_deg [S] np): monomial
    ids[s] lands on the t-free FULL-basis monomial sup_out.ids[tgt_local[s]]
    with point weight delta^t_deg[s]. Different t-powers of one spatial
    monomial COLLIDE (accumulated by the caller's index_add_). The output
    support stays in FULL-basis ids (t-exponent 0), matching how tmvPre is
    stored after an end-of-step substitution.
    """
    if sup.spatial:
        raise ValueError("eval_time_maps: input must be full-basis")
    tables = build_tables(sup.n, sup.k)
    ids = np.asarray(sup.ids, dtype=np.int64)
    exps = tables.exponents.numpy()[ids]  # [S, n+1]
    tdeg = exps[:, 0].copy()  # [S]
    strip = exps.copy()
    strip[:, 0] = 0
    radix = 2 * sup.k + 1
    gkeys = _radix_keys(tables.exponents.numpy(), radix)
    gorder = np.argsort(gkeys)
    skeys = _radix_keys(strip, radix)
    pos = np.searchsorted(gkeys[gorder], skeys)
    tgt_global = gorder[np.clip(pos, 0, gkeys.shape[0] - 1)]
    assert (gkeys[tgt_global] == skeys).all()
    out_ids = np.unique(tgt_global)
    sup_out = make_support(sup.n, sup.k, False, tuple(int(v) for v in out_ids))
    ids_of = np.asarray(sup_out.ids, dtype=np.int64)
    return sup_out, np.searchsorted(ids_of, tgt_global), tdeg


@lru_cache(maxsize=4096)
def spatial_to_full_ids(n: int, k: int, sup_sp: Support) -> Support:
    """Reinterpret a SPATIAL support as the same monomials' FULL-basis ids
    (t-exponent 0) — the embedding tmv coefficients use when meeting tmvPre."""
    tables = build_tables(n, k)
    full_sp2 = np.nonzero(tables.t_deg.numpy() == 0)[0]  # [Ts2]
    ids = full_sp2[np.asarray(sup_sp.ids, dtype=np.int64)]
    return make_support(n, k, False, tuple(int(v) for v in ids))


@lru_cache(maxsize=4096)
def full_to_spatial_ids(n: int, k: int, sup_full: Support) -> Support:
    """Reinterpret a FULL-basis t-free support as spatial-basis ids (inverse
    of spatial_to_full_ids; every id must have t-exponent 0)."""
    tables = build_tables(n, k)
    tdeg = tables.t_deg.numpy()
    ids = np.asarray(sup_full.ids, dtype=np.int64)
    if tdeg[ids].any():
        raise ValueError("full_to_spatial_ids: support has t-dependent monomials")
    full_sp2 = np.nonzero(tdeg == 0)[0]  # [Ts2] ascending full ids
    pos = np.searchsorted(full_sp2, ids)
    assert (full_sp2[pos] == ids).all()
    return make_support(n, k, True, tuple(int(v) for v in pos))


# ---------------------------------------------------------------------------
# Device bindings (per SparseEngine: device copies of the CPU tables)
# ---------------------------------------------------------------------------


@dataclass
class PairBind:
    """Device binding of pair_tables output. Shapes: pair_ia/pair_jb [P],
    seg_len [S_out]; sup_out the output Support."""

    sup_out: Support
    pair_ia: torch.Tensor
    pair_jb: torch.Tensor
    seg_len: torch.Tensor
    seg_offsets: torch.Tensor  # [S_out+1] i64 (CUDA-kernel consumption)
    tag: str  # segment-cache key for _segment_sum_with


class SparseEngine:
    """Per-run owner of device-bound sparse tables and range factors.

    Holds (tables, step, device) plus caches keyed by Support VALUES — safe
    because supports are canonical (make_support is value-cached). One
    engine per reach/driver run; steady-state calls are pure dict hits.
    """

    def __init__(self, tables: MonomialTables, step: StepTables, device: str):
        self.tables = tables
        self.step = step
        self.device = device
        # Owned before graph capture; cutoff only reads this constant.
        self._cutoff_zero = torch.zeros((), dtype=torch.float64, device=device)
        self._pair: dict = {}
        self._emb: dict = {}
        self._sub: dict = {}
        self._integ: dict = {}
        self._evalt: dict = {}
        self._evalt_iv: dict = {}
        self._evalt_error: dict = {}
        self._factor: dict = {}
        self._cat: dict = {}
        self._endf: dict = {}
        self._ids: dict = {}
        self._injective: dict = {}

    # -- index bindings ----------------------------------------------------

    def ids_t(self, sup: Support) -> torch.Tensor:
        """Device i64 tensor of a support's global ids [S]."""
        r = self._ids.get(sup)
        if r is None:
            r = torch.tensor(sup.ids, dtype=torch.int64, device=self.device)
            self._ids[sup] = r
        return r

    def pair(self, sup_a: Support, sup_b: Support) -> PairBind:
        key = (sup_a, sup_b)
        r = self._pair.get(key)
        if r is None:
            sup_out, ia, jb, seg = pair_tables(sup_a, sup_b)
            off = np.zeros(len(seg) + 1, dtype=np.int64)
            off[1:] = np.cumsum(seg)
            r = PairBind(
                sup_out=sup_out,
                pair_ia=torch.from_numpy(ia).to(self.device),
                pair_jb=torch.from_numpy(jb).to(self.device),
                seg_len=torch.from_numpy(seg).to(self.device),
                seg_offsets=torch.from_numpy(off).to(self.device),
                tag=f"sp[{_serial(sup_a)}x{_serial(sup_b)}]",
            )
            self._pair[key] = r
        return r

    def embed_idx(self, sup_small: Support, sup_big: Support) -> torch.Tensor:
        """Device positions [S_small] of sup_small inside sup_big."""
        key = (sup_small, sup_big)
        r = self._emb.get(key)
        if r is None:
            r = torch.from_numpy(subset_map(sup_small, sup_big)).to(self.device)
            self._emb[key] = r
        return r

    def embed_plan(self, sup_small: Support, sup_big: Support) -> ii.InjectiveMap:
        """Certified set embedding; the host support IDs are unique and ordered."""
        key = ('embed', sup_small, sup_big)
        plan = self._injective.get(key)
        if plan is None:
            plan = ii.bind(subset_map(sup_small, sup_big), sup_big.size, self.device)
            self._injective[key] = plan
        return plan

    def integ_plan(self, sup: Support) -> ii.InjectiveMap:
        """Incrementing the time exponent is injective; certify its local map."""
        key = ('integ', sup)
        plan = self._injective.get(key)
        if plan is None:
            sup_out, targets = integrate_maps(sup)
            plan = ii.bind(targets, sup_out.size, self.device)
            self._injective[key] = plan
        return plan

    def union(self, sup_a: Support, sup_b: Support) -> tuple:
        """(sup_u, emb_a [Sa] dev, emb_b [Sb] dev)."""
        key = (sup_a, sup_b)
        r = self._sub.get(key)
        if r is None:
            sup_u, ea, eb = union_maps(sup_a, sup_b)
            r = (
                sup_u,
                torch.from_numpy(ea).to(self.device),
                torch.from_numpy(eb).to(self.device),
            )
            self._sub[key] = r
        return r

    def integ(self, sup: Support) -> tuple:
        """(sup_out, tgt_local [S] dev, scale [S] dev, scale_iv [S, 2] dev)."""
        r = self._integ.get(sup)
        if r is None:
            sup_out, tgt = integrate_maps(sup)
            ids = self.ids_t(sup)
            r = (
                sup_out,
                torch.from_numpy(tgt).to(self.device),
                self.tables.int_scale[ids],
                self.tables.int_scale_iv[ids],
            )
            self._integ[sup] = r
        return r

    def evalt(self, sup: Support, ref_rank: int = 3) -> tuple:
        """(sup_out, tgt_local [S] dev, end_factor [S] dev): t := delta maps
        with the point weights delta^t_deg gathered from the step tables."""
        key = (sup, ref_rank)
        r = self._evalt.get(key)
        if r is None:
            sup_out, tgt, tdeg = eval_time_maps(sup)
            endf = self.step.end_gather(torch.from_numpy(tdeg).to(self.device), ref_rank)
            r = (sup_out, torch.from_numpy(tgt).to(self.device), endf)
            self._evalt[key] = r
        return r

    def evalt_interval(self, sup: Support, ref_rank: int = 3) -> torch.Tensor:
        """Cached strict endpoint powers, with fixed/per-lane broadcast shape."""
        key = (sup, ref_rank)
        r = self._evalt_iv.get(key)
        if r is None:
            _, _, tdeg = eval_time_maps(sup)
            r = self.step.end_iv_gather(torch.from_numpy(tdeg).to(self.device), ref_rank)
            self._evalt_iv[key] = r
        return r

    def evalt_error(self, sup: Support) -> tuple:
        """Cached validated factors for the static endpoint reduction map."""
        r = self._evalt_error.get(sup)
        if r is None:
            from .rounding import prepare_dot_error_bound
            sup_out, tgt, _ = eval_time_maps(sup)
            counts = torch.from_numpy(np.bincount(tgt, minlength=sup_out.size)).to(self.device)
            r = prepare_dot_error_bound(counts)
            self._evalt_error[sup] = r
        return r

    # -- range factors -----------------------------------------------------

    def factor(self, sup: Support) -> torch.Tensor:
        """Closed-form intEvalNormal weights for a FULL support: [S, 2] =
        step.factor_iv[ids] (cat x [0, delta]^t_deg)."""
        r = self._factor.get(sup)
        if r is None:
            r = self.step.factor_iv[self.ids_t(sup)].contiguous()
            self._factor[sup] = r
        return r

    def cat(self, sup: Support) -> torch.Tensor:
        """Spatial range weights: [S, 2] = sp_cat_iv[ids] (pure categories)."""
        if not sup.spatial:
            raise ValueError("cat: spatial-basis support required")
        r = self._cat.get(sup)
        if r is None:
            r = self.tables.sp_cat_iv[self.ids_t(sup)].contiguous()
            self._cat[sup] = r
        return r

    def range_idx(self, sup: Support) -> tuple:
        """(idx [S], seg_offsets [2]) for the single-segment range dot."""
        r = self._endf.get(sup)
        if r is None:
            r = (
                torch.arange(sup.size, dtype=torch.int64, device=self.device),
                torch.tensor([0, sup.size], dtype=torch.int64, device=self.device),
            )
            self._endf[sup] = r
        return r


# ---------------------------------------------------------------------------
# Sparse polynomial ops (mirror polynomial.py semantics on support tensors)
# ---------------------------------------------------------------------------


# Fused-kernel switch: the CUDA kernels (cuda_kernels.py) compute the same
# segment products with TRUE directed rounding — sound (classical directed-
# rounding argument, no error-model slack) and tighter than the torch path.
# Flipped off via FLOWSTAR_NO_CUDA_KERNEL=1 or this module flag (tests use it
# to pin the torch path for cross-representation bitwise comparisons).
USE_KERNELS = True


def _kern(t: torch.Tensor) -> bool:
    return USE_KERNELS and t.is_cuda and ck.available()


def mul_point_s(a: torch.Tensor, b: torch.Tensor, pb: PairBind) -> torch.Tensor:
    """Tier-P sparse product: a [..., Sa], b [..., Sb] -> [..., S_out].

    Torch path: same RN gather + deterministic segment sum as the dense
    mul_point — the surviving pairs' in-segment order equals the dense order
    (pair_tables), so partial sums match dense bitwise wherever the backend
    reduces sequentially (CPU). Kernel path: fused fma accumulation in the
    same pair order (tier-P has no per-op error contract; the validated pass
    absorbs the difference — GOTCHAS #15 mechanism).
    """
    if _kern(a):
        lead = a.shape[:-1]
        r = int(torch.tensor(lead).prod()) if lead else 1
        out = ck.seg_mul_pt(
            a.reshape(r, a.shape[-1]), b.reshape(r, b.shape[-1]),
            pb.pair_ia, pb.pair_jb, pb.seg_offsets,
        )
        return out.reshape(*lead, pb.sup_out.size)
    prods = a[..., pb.pair_ia] * b[..., pb.pair_jb]  # [..., P] RN
    return _segment_sum_with(prods, pb.seg_len, pb.sup_out.size, pb.tag)


def mul_iv_s(a_iv: torch.Tensor, b_iv: torch.Tensor, pb: PairBind) -> torch.Tensor:
    """Tier-I sparse product: [..., Sa, 2] x [..., Sb, 2] -> [..., S_out, 2].

    Kernel path: one fused pass with hardware directed rounding (sound and
    TIGHTER — no Rump inflation). Torch path: endpoint sums in RN + per-slot
    Rump bound over the segment lengths — the dense mul_iv derivation.
    """
    if _kern(a_iv):
        lead = a_iv.shape[:-2]
        r = int(torch.tensor(lead).prod()) if lead else 1
        out = ck.seg_mul_iv(
            a_iv.reshape(r, a_iv.shape[-2], 2), b_iv.reshape(r, b_iv.shape[-2], 2),
            pb.pair_ia, pb.pair_jb, pb.seg_offsets,
        )
        return out.reshape(*lead, pb.sup_out.size, 2)
    pa = a_iv[..., pb.pair_ia, :]  # [..., P, 2]
    pbv = b_iv[..., pb.pair_jb, :]  # [..., P, 2]
    p = iv.mul(pa, pbv)  # [..., P, 2]
    lo_hat = _segment_sum_with(p[..., 0], pb.seg_len, pb.sup_out.size, pb.tag)
    hi_hat = _segment_sum_with(p[..., 1], pb.seg_len, pb.sup_out.size, pb.tag)
    mag_hat = _segment_sum_with(iv.mag(p), pb.seg_len, pb.sup_out.size, pb.tag)
    err = dot_error_bound(mag_hat, pb.seg_len)
    return torch.stack((next_down(lo_hat - err), next_up(hi_hat + err)), dim=-1)


def mul_point_s_with_roundoff(
    a: torch.Tensor, b: torch.Tensor, pb: PairBind, eng: SparseEngine
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse tier-P product plus a strict range of coefficient roundoff.

    The point result follows the normal production path unchanged.  A second,
    rigorous interval-coefficient product encloses the exact real polynomial
    induced by the binary64 inputs.  Their difference is ranged over the
    output support and is suitable for one insertion into the TM remainder.
    """
    point = mul_point_s(a, b, pb)
    exact_coeff = mul_iv_s(iv.from_point(a), iv.from_point(b), pb)
    coefficient_error = iv.sub(exact_coeff, iv.from_point(point))
    return point, range_spatial_iv_s(coefficient_error, eng, pb.sup_out)


def range_normal_s(coeffs: torch.Tensor, eng: SparseEngine, sup: Support) -> torch.Tensor:
    """intEvalNormal of a point polynomial on a FULL support: [..., S] -> [..., 2]."""
    if _kern(coeffs):
        idx, off = eng.range_idx(sup)
        lead = coeffs.shape[:-1]
        r = int(torch.tensor(lead).prod()) if lead else 1
        out = ck.seg_dot_pt_iv(coeffs.reshape(r, sup.size), eng.factor(sup), off, idx)
        return out.reshape(*lead, 2)
    return iv.dot_point_iv(coeffs, eng.factor(sup), dim=-1)


def range_normal_iv_s(
    coeffs_iv: torch.Tensor, eng: SparseEngine, sup: Support
) -> torch.Tensor:
    """intEvalNormal of an interval polynomial on a FULL support:
    [..., S, 2] -> [..., 2]."""
    prod = iv.mul(coeffs_iv, eng.factor(sup))  # [..., S, 2]
    return iv.sum(prod, dim=-1)


def range_spatial_s(coeffs: torch.Tensor, eng: SparseEngine, sup: Support) -> torch.Tensor:
    """Range of a t-free point polynomial on a SPATIAL support over [-1,1]^n."""
    if _kern(coeffs):
        idx, off = eng.range_idx(sup)
        lead = coeffs.shape[:-1]
        r = int(torch.tensor(lead).prod()) if lead else 1
        out = ck.seg_dot_pt_iv(coeffs.reshape(r, sup.size), eng.cat(sup), off, idx)
        return out.reshape(*lead, 2)
    return iv.dot_point_iv(coeffs, eng.cat(sup), dim=-1)


def range_spatial_iv_s(
    coeffs_iv: torch.Tensor, eng: SparseEngine, sup: Support
) -> torch.Tensor:
    """Range a sparse t-free interval polynomial over ``[-1,1]^n``."""
    return iv.sum(iv.mul(coeffs_iv, eng.cat(sup)), dim=-1)


def ctrunc_iv_s(
    coeffs_iv: torch.Tensor, eng: SparseEngine, sup: Support, order: int
) -> tuple[torch.Tensor, torch.Tensor, Support]:
    """Conservative truncation on a FULL support (ctrunc_normal_iv).

    [..., S, 2] -> (kept [..., K, 2], tail [..., 2], sup_kept). The kept part
    is the contiguous degree <= order prefix (ascending-graded invariant).
    """
    keep = sup.keep_len(order)
    if keep >= sup.size:
        zero = torch.zeros(*coeffs_iv.shape[:-2], 2, dtype=coeffs_iv.dtype,
                           device=coeffs_iv.device)
        return coeffs_iv, zero, sup
    dropped = coeffs_iv[..., keep:, :]  # [..., S-K, 2]
    prod = iv.mul(dropped, eng.factor(sup)[keep:])  # [..., S-K, 2]
    tail = iv.sum(prod, dim=-1)
    sup_kept = make_support(sup.n, sup.k, sup.spatial, sup.ids[:keep])
    return coeffs_iv[..., :keep, :], tail, sup_kept


def ctrunc_spatial_s(
    coeffs: torch.Tensor, eng: SparseEngine, sup: Support, order: int
) -> tuple[torch.Tensor, torch.Tensor, Support]:
    """Spatial point ctrunc: [..., S] -> (kept [..., K], tail [..., 2], sup_kept)."""
    keep = sup.keep_len(order)
    if keep >= sup.size:
        zero = torch.zeros(*coeffs.shape[:-1], 2, dtype=coeffs.dtype, device=coeffs.device)
        return coeffs, zero, sup
    tail = iv.dot_point_iv(coeffs[..., keep:], eng.cat(sup)[keep:], dim=-1)
    sup_kept = make_support(sup.n, sup.k, sup.spatial, sup.ids[:keep])
    return coeffs[..., :keep], tail, sup_kept


def cutoff_normal_s(
    coeffs: torch.Tensor, eng: SparseEngine, sup: Support, cutoff_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Point cutoff on a FULL support (cutoff_normal): zero |c| <= eps slots,
    range the dropped part. [..., S] -> (kept [..., S], dropped_rng [..., 2])."""
    small = coeffs.abs() <= cutoff_threshold
    zeros = eng._cutoff_zero
    if zeros.dtype != coeffs.dtype or zeros.device != coeffs.device:
        zeros = torch.zeros_like(coeffs)
    dropped = torch.where(small, coeffs, zeros)
    rng = iv.dot_point_iv(dropped, eng.factor(sup), dim=-1)
    kept = torch.where(small, zeros, coeffs)
    return kept, rng


def cutoff_spatial_s(
    coeffs: torch.Tensor, eng: SparseEngine, sup: Support, cutoff_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatial point cutoff (cutoff_normal_spatial semantics)."""
    small = coeffs.abs() <= cutoff_threshold
    zeros = eng._cutoff_zero
    if zeros.dtype != coeffs.dtype or zeros.device != coeffs.device:
        zeros = torch.zeros_like(coeffs)
    dropped = torch.where(small, coeffs, zeros)
    rng = iv.dot_point_iv(dropped, eng.cat(sup), dim=-1)
    kept = torch.where(small, zeros, coeffs)
    return kept, rng


def cutoff_interval_s(
    coeffs_iv: torch.Tensor, eng: SparseEngine, sup: Support,
    cutoff_threshold: float, max_width: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval-coefficient cutoff on a FULL support — the exact
    cutoff_normal_interval logic (width split / midpoint drop) with the
    range dot restricted to the support. [..., S, 2] -> (kept, dropped_rng).
    """
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
    prod = iv.mul(to_rem, eng.factor(sup))
    dropped_range = iv.sum(prod, dim=-1)
    kept = torch.where(wide.unsqueeze(-1), mid_iv, coeffs_iv)
    kept = torch.where(small.unsqueeze(-1), zeros2, kept)
    return kept, dropped_range


def integrate_t_s(
    coeffs: torch.Tensor, eng: SparseEngine, sup: Support
) -> tuple[torch.Tensor, Support]:
    """Sparse tier-P t-integration: [..., S] -> ([..., S_out], sup_out).

    Injective local scatter (int targets are unique per source), RN scale —
    integrate_t semantics on the support.
    """
    sup_out, tgt, scale, _ = eng.integ(sup)
    src = coeffs * scale  # [..., S] RN
    out = torch.zeros(*coeffs.shape[:-1], sup_out.size, dtype=coeffs.dtype,
                      device=coeffs.device)
    if ii.ENABLED:
        ii.update_(out, -1, src, eng.integ_plan(sup), add=True)
    else:
        out.index_add_(-1, tgt, src)
    return out, sup_out


def integrate_t_iv_s(
    coeffs_iv: torch.Tensor, eng: SparseEngine, sup: Support
) -> tuple[torch.Tensor, Support]:
    """Sparse tier-I t-integration: [..., S, 2] -> ([..., S_out, 2], sup_out)."""
    sup_out, tgt, _, scale_iv = eng.integ(sup)
    src = iv.mul(coeffs_iv, scale_iv)  # [..., S, 2]
    out = torch.zeros(*coeffs_iv.shape[:-2], sup_out.size, 2,
                      dtype=coeffs_iv.dtype, device=coeffs_iv.device)
    if ii.ENABLED:
        ii.update_(out, -2, src, eng.integ_plan(sup), add=True)
    else:
        out.index_add_(-2, tgt, src)
    return out, sup_out


def evaluate_time_end_s(
    coeffs: torch.Tensor, eng: SparseEngine, sup: Support
) -> tuple[torch.Tensor, Support]:
    """Sparse t := delta substitution: [..., S] -> ([..., S_out], sup_out).

    Colliding t-powers accumulate via deterministic index_add_ (same kernel
    contract as evaluate_time_end). Output ids are t-free FULL-basis ids.
    """
    sup_out, tgt, endf = eng.evalt(sup, coeffs.dim())
    src = coeffs * endf  # [..., S] RN
    out = torch.zeros(*coeffs.shape[:-1], sup_out.size, dtype=coeffs.dtype,
                      device=coeffs.device)
    out.index_add_(-1, tgt, src)
    return out, sup_out


def evaluate_time_end_s_with_roundoff(
    coeffs: torch.Tensor, eng: SparseEngine, sup: Support
) -> tuple[torch.Tensor, Support, torch.Tensor]:
    """Sparse endpoint evaluation plus strict coefficient-error range."""
    sup_out, tgt, endf = eng.evalt(sup, coeffs.dim())
    src = coeffs * endf
    src_iv = iv.mul(iv.from_point(coeffs), eng.evalt_interval(sup, coeffs.dim()))
    # Independent rows share one static map. Preserve source-column order
    # and the +0 seed of each original deterministic index_add reduction.
    sources = torch.stack((src, src_iv[..., 0], src_iv[..., 1], iv.mag(src_iv)), dim=0)
    sums = torch.zeros(*sources.shape[:-1], sup_out.size,
                       dtype=coeffs.dtype, device=coeffs.device)
    sums.index_add_(-1, tgt, sources)
    point, lo_hat, hi_hat, mag_hat = sums.unbind(0)
    from .rounding import apply_dot_error_bound
    error = apply_dot_error_bound(mag_hat, eng.evalt_error(sup))
    exact_coeff = torch.stack((next_down(lo_hat - error), next_up(hi_hat + error)), dim=-1)
    coefficient_error = iv.sub(exact_coeff, iv.from_point(point))
    # Endpoint substitution retains FULL-basis IDs. Spatial category IDs are
    # different from degree two onward; conversion preserves coefficient order.
    spatial_out = full_to_spatial_ids(sup_out.n, sup_out.k, sup_out)
    return point, sup_out, range_spatial_iv_s(coefficient_error, eng, spatial_out)


def embed_s(
    coeffs: torch.Tensor, eng: SparseEngine, sup_from: Support, sup_to: Support,
    interval: bool = False,
) -> torch.Tensor:
    """Re-embed [..., S_from] (point) or [..., S_from, 2] (interval=True)
    onto a SUPERSET support -> [..., S_to(, 2)]. Pure placement, no
    arithmetic — embedding zeros is exact.
    """
    if sup_from is sup_to:
        return coeffs
    idx = eng.embed_idx(sup_from, sup_to)
    if interval:
        out = torch.zeros(*coeffs.shape[:-2], sup_to.size, 2,
                          dtype=coeffs.dtype, device=coeffs.device)
        if ii.ENABLED:
            ii.update_(out, -2, coeffs, eng.embed_plan(sup_from, sup_to))
        else:
            out.index_copy_(-2, idx, coeffs)
    else:
        out = torch.zeros(*coeffs.shape[:-1], sup_to.size,
                          dtype=coeffs.dtype, device=coeffs.device)
        if ii.ENABLED:
            ii.update_(out, -1, coeffs, eng.embed_plan(sup_from, sup_to))
        else:
            out.index_copy_(-1, idx, coeffs)
    return out


def add_aligned(
    a: torch.Tensor, sup_a: Support, b: torch.Tensor, sup_b: Support,
    eng: SparseEngine, interval: bool,
) -> tuple[torch.Tensor, Support]:
    """a + b over possibly different supports -> (sum, union support).

    Tier-P when interval=False (plain RN add); tier-I (iv.add outward) when
    True. Embedding first is exact (zeros), so semantics match the dense op.
    """
    sup_u, _, _ = eng.union(sup_a, sup_b)
    av = embed_s(a, eng, sup_a, sup_u, interval=interval).contiguous()
    bv = embed_s(b, eng, sup_b, sup_u, interval=interval).contiguous()
    return (iv.add(av, bv) if interval else av + bv), sup_u
