"""Tier-I rigorous interval arithmetic on float64 tensors (plan D2).

Representation: an interval tensor has a trailing dimension of size 2 —
`iv[..., 0] = lo`, `iv[..., 1] = hi` — abbreviated `[..., 2]` in shape comments.
Invariant: lo <= hi (checked by `assert_valid` at pipeline boundaries, not per op).

Soundness model
---------------
Each elementwise op computes in round-to-nearest and steps the result outward
with `rounding.next_up/next_down` (sound because RN returns the nearest float:
the exact value lies within one ulp of it). Reductions use RN in ANY order plus
Rump-style a-priori error bounds (`rounding.dot_error_bound`;
`sum_error_bound` is the pedagogical base case of the derivation, with no
production caller), which makes FMA and nondeterministic reduction orders
harmless.

Flow* correspondence (ALGORITHM.md "Arithmetic primitives" table):
  * mul: Flow*'s 9-case sign dispatch (Interval.cpp:1448) equals the 4-product
    min/max form used here — the dispatch is an optimization, not a semantics
    change — so results agree up to our <=1 ulp outward slack.
  * div: reciprocal-then-multiply, matching Flow* (Interval.cpp:1586), which is
    deliberately WIDER than a direct quotient. Parity requires this shape.
  * rec of a zero-containing interval: Flow* silently returns [-1e5, 1e5]
    (unsound; GOTCHAS #2). We instead return [-inf, +inf] AND report the lanes
    so the caller can mark them FAILED_DIV.
  * pow: successive interval multiplication (square-free ascending chain), the
    same structure as Flow*'s step_exp_table construction (intProd *= intStep).
"""

from __future__ import annotations

import torch

from . import cuda_kernels as _ck
from .rounding import dot_error_bound, next_down, next_up

# ---------------------------------------------------------------------------
# Fused-kernel dispatch (M10 perf)
# ---------------------------------------------------------------------------
# On CUDA f64 inputs the ops below route to the fused kernels in
# cuda_kernels.py, which use the hardware's TRUE directed-rounding intrinsics
# (__dmul_rd/__dmul_ru, __dadd_rd/__dadd_ru): one kernel launch instead of the
# 4-10 launch nextafter chains, sound by the classical MPFR argument with ZERO
# emulation slack — always at least as tight as the torch path below (which
# pays <= 1 ulp per elementwise op plus Rump reduction inflation). Reductions
# accumulate sequentially in fixed index order, so determinism and the
# lane==B=1 batch-independence gate hold by construction. The torch bodies
# remain the CPU path and the test oracle; flip USE_KERNELS = False to force
# them everywhere (parity/debug).

USE_KERNELS = True


def _kern(*tensors: torch.Tensor) -> bool:
    """True iff every input is CUDA f64 and the fused kernels are built."""
    if not USE_KERNELS:
        return False
    for t in tensors:
        # Empty operands use the torch path; fused elementwise launches require N > 0.
        if not (t.is_cuda and t.dtype is torch.float64) or t.numel() == 0:
            return False
    return _ck.available()

# ---------------------------------------------------------------------------
# Construction / accessors
# ---------------------------------------------------------------------------
# make/lo/hi are convenience accessors for tests/notebooks; kernels index the
# endpoints directly ([..., 0] / [..., 1]).


def make(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Stack endpoint tensors [...] , [...] -> interval tensor [..., 2]."""
    return torch.stack((lo, hi), dim=-1)


def from_point(x: torch.Tensor) -> torch.Tensor:
    """Degenerate interval [x, x]: [...] -> [..., 2]. No rounding: exact."""
    return torch.stack((x, x), dim=-1)


def lo(iv: torch.Tensor) -> torch.Tensor:
    """Lower endpoints: [..., 2] -> [...]."""
    return iv[..., 0]


def hi(iv: torch.Tensor) -> torch.Tensor:
    """Upper endpoints: [..., 2] -> [...]."""
    return iv[..., 1]


def assert_valid(iv: torch.Tensor, what: str = "interval") -> None:
    """Finite-certificate boundary check: ordered and entirely finite.

    Not called inside kernels (cost + graph breaks); pipeline stages call it on
    entry/exit under a config flag, and tests call it liberally.
    """
    if torch.isnan(iv).any():
        raise FloatingPointError(f"{what}: NaN endpoint")
    if torch.isinf(iv).any():
        raise FloatingPointError(f"{what}: infinite endpoint is not a finite certificate")
    if (iv[..., 0] > iv[..., 1]).any():
        raise FloatingPointError(f"{what}: lo > hi")


# ---------------------------------------------------------------------------
# Elementwise ring ops — each is RN + one outward nextafter step (<= 1 ulp slack)
# ---------------------------------------------------------------------------


def neg(a: torch.Tensor) -> torch.Tensor:
    """-[lo, hi] = [-hi, -lo]. Negation is EXACT in IEEE-754: no rounding step."""
    if _kern(a):
        return _ck.iv_neg(a)
    return torch.stack((-a[..., 1], -a[..., 0]), dim=-1)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[..., 2] + [..., 2] -> [..., 2]. Mirrors Interval::operator+= (RNDD/RNDU).

    Kernel path: true directed adds (exactly MPFR RNDD/RNDU, tighter than the
    nextafter emulation below)."""
    if _kern(a, b):
        if a.shape != b.shape:
            a, b = torch.broadcast_tensors(a, b)
        return _ck.iv_add(a, b)
    return torch.stack(
        (next_down(a[..., 0] + b[..., 0]), next_up(a[..., 1] + b[..., 1])), dim=-1
    )


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[..., 2] - [..., 2] -> [..., 2]: [alo - bhi, ahi - blo], outward-rounded.

    NOTE (Flow* parity): interval subtraction WIDENS — a - a is not [0, 0].
    Kernel path: true directed rounding (tighter than the emulation below).
    """
    if _kern(a, b):
        if a.shape != b.shape:
            a, b = torch.broadcast_tensors(a, b)
        return _ck.iv_sub(a, b)
    return torch.stack(
        (next_down(a[..., 0] - b[..., 1]), next_up(a[..., 1] - b[..., 0])), dim=-1
    )


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[..., 2] * [..., 2] -> [..., 2] via all four endpoint products.

    candidates [..., 4] = {al*bl, al*bh, ah*bl, ah*bh}; result = [min, max]
    outward-rounded. Identical result to Flow*'s 9-case sign dispatch (the
    dispatch merely avoids computing candidates it can prove non-extremal).

    CAUTION: 0 * inf = NaN would poison min/max silently, so this function must
    never see infinite endpoints; `div` therefore filters zero-crossing
    denominators BEFORE multiplying (callers get a failure mask instead).

    Kernel path: the four candidates per endpoint under TRUE directed rounding
    (one launch instead of nine; sound with zero slack).
    """
    if _kern(a, b):
        if a.shape != b.shape:
            a, b = torch.broadcast_tensors(a, b)
        return _ck.iv_mul(a, b)
    al, ah = a[..., 0], a[..., 1]
    bl, bh = b[..., 0], b[..., 1]
    # candidates: [..., 4]
    cand = torch.stack((al * bl, al * bh, ah * bl, ah * bh), dim=-1)
    return torch.stack(
        (next_down(cand.amin(dim=-1)), next_up(cand.amax(dim=-1))), dim=-1
    )


def mul_point(a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Interval [..., 2] times POINT tensor [...] -> [..., 2].

    Only two candidate products since p is a single value per element.
    Kernel path: directed candidate products, one launch.
    """
    if _kern(a, p):
        common = torch.broadcast_shapes(a.shape[:-1], p.shape)
        return _ck.iv_mul_point(
            a.broadcast_to(*common, 2).contiguous(), p.broadcast_to(common).contiguous()
        )
    cl, ch = a[..., 0] * p, a[..., 1] * p
    return torch.stack(
        (next_down(torch.minimum(cl, ch)), next_up(torch.maximum(cl, ch))), dim=-1
    )


# ---------------------------------------------------------------------------
# Division — Flow* semantics: reciprocal then multiply
# ---------------------------------------------------------------------------


def rec(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval reciprocal. Returns (result [..., 2], bad_mask [...] bool).

    Where 0 in [lo, hi] (bad_mask True) the result is [-inf, +inf] — sound but
    useless — and callers MUST fail those lanes (we never replicate Flow*'s
    silent [-1e5, 1e5] fallback, GOTCHAS #2). Elsewhere: [1/hi down, 1/lo up]
    (anti-monotone, valid for both all-positive and all-negative intervals).
    """
    al, ah = a[..., 0], a[..., 1]
    bad = (al <= 0) & (ah >= 0)  # [...] bool: zero-containing (incl. touching 0)

    lo_r = next_down(1.0 / ah)
    hi_r = next_up(1.0 / al)
    # Overflow guard: a denominator endpoint with magnitude < ~5.56e-309 gives
    # 1/x = inf even though the interval excludes zero; downstream `mul` would
    # then produce 0 * inf = NaN silently. Such lanes are failed loudly instead
    # (found by the M1 property fuzzer; div is only sound for denominators
    # bounded away from zero anyway — GOTCHAS #2 policy).
    bad = bad | torch.isinf(lo_r) | torch.isinf(hi_r)
    out = torch.stack((lo_r, hi_r), dim=-1)
    out = torch.where(
        bad.unsqueeze(-1),
        torch.stack((torch.full_like(al, -torch.inf), torch.full_like(ah, torch.inf)), dim=-1),
        out,
    )
    return out, bad


def div(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """a / b as rec(b) * a — deliberately Flow*-shaped (two roundings, wider
    than a direct quotient; Interval.cpp:1586-1594). Returns (result, bad_mask).

    The multiply runs on a zero-filled substitute wherever bad_mask is set (to
    keep inf out of `mul`, see its docstring); those lanes' outputs are then
    forced to [-inf, +inf] and must be failed by the caller anyway.
    """
    r, bad = rec(b)
    r_safe = torch.where(bad.unsqueeze(-1), torch.zeros_like(r), r)
    out = mul(a, r_safe)
    out = torch.where(
        bad.unsqueeze(-1),
        torch.stack(
            (torch.full_like(out[..., 0], -torch.inf), torch.full_like(out[..., 1], torch.inf)),
            dim=-1,
        ),
        out,
    )
    return out, bad


# ---------------------------------------------------------------------------
# Integer powers
# ---------------------------------------------------------------------------


def powers(a: torch.Tensor, kmax: int) -> torch.Tensor:
    """Table of a^0 .. a^kmax by successive interval multiplication.

    Input [..., 2]; output [kmax+1, ..., 2] with powers[0] = [1, 1] exactly.
    This is the same ascending-product structure Flow* uses to build
    step_exp_table (construct_step_exp_table: intProd *= intStep), so widths
    match Flow*'s up to our <=1 ulp/step slack. Used for [0, delta]^j tables.
    """
    if kmax < 0:
        raise ValueError(f"kmax must be >= 0, got {kmax}")
    one = torch.ones_like(a)  # [..., 2] = [1, 1] exactly
    out = [one]
    acc = one
    for _ in range(kmax):
        acc = mul(acc, a)
        out.append(acc)
    return torch.stack(out, dim=0)


def pow_int(a: torch.Tensor, n: int) -> torch.Tensor:
    """a^n with the tight even/odd endpoint rule (Flow* Interval::pow semantics).

    odd n : monotone — [lo^n down, hi^n up]
    even n: |a|^n — if 0 in a: [0, max(|lo|, |hi|)^n up]; else
            [min(|lo|, |hi|)^n down, max(|lo|, |hi|)^n up]

    Endpoint powers are computed by a chain of RN multiplications with one
    outward step per multiply (torch.pow has no accuracy guarantee, so we do
    NOT use it in rigorous code). n >= 1.
    """
    if n < 1:
        raise ValueError(f"pow_int needs n >= 1, got {n}")

    def chain_up(x: torch.Tensor) -> torch.Tensor:
        # Upper bound of x^n for x >= 0: round every partial product up.
        acc = x
        for _ in range(n - 1):
            acc = next_up(acc * x)
        return acc

    def chain_down(x: torch.Tensor) -> torch.Tensor:
        # Lower bound of x^n for x >= 0: round every partial product down
        # (partials stay >= 0, so rounding down keeps them valid lower bounds).
        acc = x
        for _ in range(n - 1):
            acc = torch.clamp(next_down(acc * x), min=0.0)
        return acc

    al, ah = a[..., 0], a[..., 1]
    if n % 2 == 1:
        # Odd: x^n is monotone over the whole real line. Signed endpoint powers:
        # for negative endpoints the DOWN-rounded magnitude chain gives the
        # wrong direction, so compute via |x|^n with sign restored, choosing the
        # rounding direction per endpoint sign.
        lo_out = torch.where(al >= 0, chain_down(al), -chain_up(-al))
        hi_out = torch.where(ah >= 0, chain_up(ah), -chain_down(-ah))
        return torch.stack((lo_out, hi_out), dim=-1)

    # Even: work on magnitudes.
    mag_min = torch.minimum(al.abs(), ah.abs())  # [...]
    mag_max = torch.maximum(al.abs(), ah.abs())  # [...]
    crosses = (al <= 0) & (ah >= 0)  # [...] bool
    lo_out = torch.where(crosses, torch.zeros_like(mag_min), chain_down(mag_min))
    hi_out = chain_up(mag_max)
    return torch.stack((lo_out, hi_out), dim=-1)


# ---------------------------------------------------------------------------
# Reductions — RN any-order + Rump inflation (FMA/reordering licensed)
# ---------------------------------------------------------------------------


def sum(iv: torch.Tensor, dim: int) -> torch.Tensor:  # noqa: A001 - mirrors torch.sum
    """Sum of intervals over one dim of the ENDPOINT-STRIPPED shape.

    Convention (holds for every reduction in this module): `dim` indexes the
    shape WITHOUT the trailing endpoint axis — i.e. the shape of iv[..., 0].
    So for iv [..., m, 2], `dim=-1` reduces over m. This avoids the perpetual
    off-by-one against callers who think in "logical" polynomial shapes.

    Endpoint sums run in RN (any reduction order), then each endpoint is pushed
    outward by `dot_error_bound` on the RN sum of max(|lo|, |hi|) — one shared
    magnitude reduction bounds BOTH endpoints' accumulated rounding error.
    (dot_error_bound rather than sum_error_bound: its extra product-error and
    underflow slack also covers callers that fused a multiply into the terms.)

    Kernel path (reduction over the LAST stripped dim only): sequential
    directed accumulation in index order — deterministic, batch-independent,
    and tighter than RN + Rump. Other dims fall through to the torch path.
    """
    if _kern(iv):
        nd = iv.dim() - 1  # endpoint-stripped rank
        d = dim if dim >= 0 else dim + nd
        if nd >= 1 and d == nd - 1:
            x = iv.contiguous()
            m = x.shape[-2]
            return _ck.iv_sum(x.view(-1, m, 2)).view(*x.shape[:-2], 2)
    los = iv[..., 0]  # [...] endpoint-stripped view
    m = los.shape[dim]
    lo_hat = los.sum(dim=dim)  # [...] RN, any order
    hi_hat = iv[..., 1].sum(dim=dim)  # [...]
    mag_hat = torch.maximum(iv[..., 0].abs(), iv[..., 1].abs()).sum(dim=dim)  # [...]
    err = dot_error_bound(mag_hat, m)  # [...]
    return torch.stack((next_down(lo_hat - err), next_up(hi_hat + err)), dim=-1)


def dot_point_iv(p: torch.Tensor, iv: torch.Tensor, dim: int) -> torch.Tensor:
    """Dot of a POINT tensor with an interval tensor over `dim`.

    p broadcastable to iv's endpoint-stripped shape; `dim` indexes that
    endpoint-stripped/broadcast shape (see `sum` for the convention) — for
    p [..., m] with iv [..., m, 2], `dim=-1` reduces over m; result [..., 2].

    Candidate endpoint products are {p*lo, p*hi} (p is a single value per
    slot), reduced in RN and inflated by the a-priori dot bound. This is the
    workhorse behind closed-form intEvalNormal range evaluation.

    Kernel path (reduction over the LAST stripped dim only): sequential
    directed accumulation; a [m, 2] weight shared across rows is passed
    without materializing its broadcast (the intEvalNormal common case).
    """
    if _kern(p, iv):
        stripped = torch.broadcast_shapes(p.shape, iv.shape[:-1])
        nd = len(stripped)
        d = dim if dim >= 0 else dim + nd
        if nd >= 1 and d == nd - 1:
            m = stripped[-1]
            p_b = p.broadcast_to(stripped).contiguous().view(-1, m)
            if iv.dim() == 2 and iv.shape[0] == m:
                out = _ck.iv_dot_point_iv(p_b, iv.contiguous())
            else:
                w_b = iv.broadcast_to(*stripped, 2).contiguous().view(-1, m, 2)
                out = _ck.iv_dot_point_iv(p_b, w_b)
            return out.view(*stripped[:-1], 2)
    cl = p * iv[..., 0]  # [...] candidate 1 (broadcast shape)
    ch = p * iv[..., 1]  # [...] candidate 2
    m = cl.shape[dim]
    lo_hat = torch.minimum(cl, ch).sum(dim=dim)  # [...] RN
    hi_hat = torch.maximum(cl, ch).sum(dim=dim)  # [...]
    mag_hat = (p.abs() * torch.maximum(iv[..., 0].abs(), iv[..., 1].abs())).sum(dim=dim)
    err = dot_error_bound(mag_hat, m)  # [...]
    return torch.stack((next_down(lo_hat - err), next_up(hi_hat + err)), dim=-1)


# ---------------------------------------------------------------------------
# Predicates / measures
# ---------------------------------------------------------------------------


def contains(outer: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
    """inner subseteq outer, elementwise: [..., 2], [..., 2] -> [...] bool.

    Matches Flow* Interval::subseteq (Interval.cpp:1243): closed-interval
    containment with plain float comparisons.
    """
    finite = torch.isfinite(outer).all(dim=-1) & torch.isfinite(inner).all(dim=-1)
    return finite & (outer[..., 0] <= inner[..., 0]) & (inner[..., 1] <= outer[..., 1])


def width(iv: torch.Tensor) -> torch.Tensor:
    """Upper bound of hi - lo: [..., 2] -> [...]. Flow* width() rounds up (RNDU)."""
    return next_up(iv[..., 1] - iv[..., 0])


def mag(iv: torch.Tensor) -> torch.Tensor:
    """max(|lo|, |hi|): [..., 2] -> [...]. Exact (abs and max round nothing)."""
    return torch.maximum(iv[..., 0].abs(), iv[..., 1].abs())
