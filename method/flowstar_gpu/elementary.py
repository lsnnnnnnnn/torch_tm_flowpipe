"""Elementary-function Taylor-model operations (M5): exp/rec/sin/cos/log/sqrt.

Flow* evaluates a non-polynomial RHS node three ways (same three-tape design as
ode_compiler.py); this module provides the per-function kernels the tapes
dispatch to:

  * `*_series_point`  — Polynomial<Real>::*_taylor (Polynomial.h:1892-2223):
        tier-P point-coefficient series for the polynomial-only Picard pass.
  * `*_series_valid`  — TaylorModel<Interval>::*_taylor with `ranges` caching
        (TaylorModel.h:1266-1895): interval coefficients, remainder algebra,
        cache pushes in Flow*'s exact order.
  * `*_series_replay` — the `*_taylor_only_remainder` twins (expression.h:37-
        384): remainder-only recomputation consuming the cache in write order.
  * `*_taylor_remainder` — the Lagrange tail formulas (settings.h:322-452).
  * `elem_tables` — rigorous enclosures of Flow*'s Global_Setting tables
        1/i! (factorial_rec), i!! (double_factorial) plus 1/i (settings.h:23-67).

Cache layouts (slots per op at series order k; counted from the Flow* sources,
non-constant path — the tape compiler sizes cache_base/cache_len from these):

    exp  (TaylorModel.h:1266): 1 [exp(c)] + 3k [k mul triples] + 1 [FPolyRange]
                             = 3k + 2
    rec  (TaylorModel.h:1343): 2 [c, 1/c] + 3k + 1                = 3k + 3
    sin  (TaylorModel.h:1426): 1 [c] + 4k [triple + tmp each] + 1 = 4k + 2
    cos  (TaylorModel.h:1577): same as sin                        = 4k + 2
    log  (TaylorModel.h:1730): 1 [c] + 3(k-1) + 1                 = 3k - 1
    sqrt (TaylorModel.h:1809): 1 [c] + 3(k-1) + 1 [2*FPolyRange]  = 3k - 1
    div  (expression.h OPT_DIV:1554): rec layout + 3 [final mul]  = 3k + 6

GOTCHAS #19 (Flow* soundness BUG, found and verified here by direct execution
of Polynomial<Real>::sin_taylor): Flow*'s sin/cos series use the coefficient
`deriv_i(c) / i` where the Taylor series requires `deriv_i(c) / i!`
(Polynomial.h:2020/2102 `tmp = sinc / i`, same in TaylorModel.h:1496 etc.) —
sin(x) at order 5 comes out as x - x^3/3 + x^5/5 (the arctan series!) while the
Lagrange remainder is computed for the TRUE series, so Flow*'s sin/cos
enclosures at order >= 3 do NOT contain sin/cos. We do NOT replicate: our
coefficient is `deriv_i(c) * enclosure(1/i!)` (point tier: `deriv_i(c)/i!` in
RN), keeping Flow*'s exact loop structure and cache layout (the per-iteration
`tmp` slot holds the corrected value). exp/rec/log/sqrt were verified correct
the same way.

GOTCHAS #20 (design deviation): Flow*'s series take a sentinel shortcut when
the argument TM is a bare constant (`ranges` gets 1-2 slots + an invalid
Interval(1,-1) marker, expression.h:41-57), making cache length DATA-dependent
— impossible in a static batched tape. Resolution: ode_compiler constant-folds
function-of-constant subtrees at COMPILE time (mpmath correctly-rounded), so
the runtime series only ever see the non-constant layout. A runtime TM whose
polynomial happens to be constant still takes the full series — sound (the
Horner loop degenerates to the constant result + Lagrange tail), merely wider
than Flow*'s special case.

Domain violations (rec/div of a zero-containing interval, log of a nonpositive
one, sqrt of a negative one) never take Flow*'s silent [-1e5, 1e5] fallback /
exit(1) (GOTCHAS #2): the lane's `bad` flag is returned (callers freeze it as
FAILED_DIV) and the lane's outputs are sanitized to finite garbage (zeros) so
no NaN/inf can poison batched kernels.

Shapes: coefficient tensors are [B, T, 2] interval prefixes over the working
monomial basis (constant term at slot 0), remainders [B, 2], the shared cache
[B, n_cache, 2], lane flags [B] bool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import torch

from . import interval as iv
from . import polynomial as poly
from .monomials import MonomialTables
from .polynomial import StepTables
from .rounding import next_down, next_up
from .transcendental import (
    cos_iv,
    exp_iv,
    log_iv,
    sigmoid_iv,
    sin_iv,
    sqrt_iv,
    tanh_iv,
)

# ---------------------------------------------------------------------------
# Rigorous constant tables (Flow* Global_Setting, settings.h:23-67)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElemTables:
    """Interval enclosures of the Flow* Global_Setting series constants.

    Flow* builds factorial_rec (1/i!) and double_factorial (i!!) as Real (RN
    cascade products, settings.h:23-67) and uses them as POINT factors in the
    Lagrange formulas. For soundness we instead store the TIGHTEST float64
    interval bracket of the TRUE rational value (exact when representable) —
    within 1 ulp of Flow*'s cascade values.

    Attributes (N = 2*max_order + 4; index i holds the value for i):
        frec_iv [N+1, 2]: enclosure of 1/i!.
        dfact_iv [N+1, 2]: enclosure of i!! (0!! = 1!! = 1).
        int_rec_iv [N+1, 2]: enclosure of 1/i, i >= 1 (row 0 is the unused
            placeholder [1, 1]) — the TaylorModel<Interval> /= i factor, which
            Flow* computes as MPFR rec([i, i]) (TaylorModel.h:592 -> Interval
            division = rec-then-mul, Interval.cpp:1586).
        device: torch device string of the tensors.
    """

    max_order: int
    frec_iv: torch.Tensor
    dfact_iv: torch.Tensor
    int_rec_iv: torch.Tensor
    device: str


def _dbracket(value: Fraction) -> tuple[float, float]:
    """Tightest float64 bracket [lo, hi] of an exact rational; lo == hi iff
    the value is representable. Fraction.__float__ is correctly rounded, so
    one nextafter step in the deficient direction is exact bracketing."""
    q = float(value)
    fq = Fraction(q)
    if fq == value:
        return q, q
    if fq < value:
        return q, math.nextafter(q, math.inf)
    return math.nextafter(q, -math.inf), q


@lru_cache(maxsize=32)
def elem_tables(max_order: int, device: str = "cpu") -> ElemTables:
    """Build (and cache per (order, device)) the series-constant tables.

    Sized for Lagrange tails at order max_order+1: frec index max_order+1 and
    dfact index 2*(max_order+1)-3 both fit in N = 2*max_order + 4. Pure exact
    rational arithmetic + correctly-rounded bracketing — deterministic, so the
    process-wide cache is safe; caching per device keeps repeated exec_* calls
    allocation-free.
    """
    if max_order < 1:
        raise ValueError(f"elem_tables needs max_order >= 1, got {max_order}")
    n = 2 * max_order + 4
    frec, dfact, int_rec = [], [], []
    fact = Fraction(1)
    df_val = [Fraction(1), Fraction(1)]  # i!!: (i-2)!! memo by parity
    for i in range(n + 1):
        if i > 0:
            fact *= i
        frec.append(_dbracket(1 / fact))
        if i >= 2:
            df_val[i % 2] *= i
        dfact.append(_dbracket(df_val[i % 2]))
        int_rec.append(_dbracket(Fraction(1, i)) if i >= 1 else (1.0, 1.0))
    as_t = lambda rows: torch.tensor(rows, dtype=torch.float64, device=device)  # noqa: E731
    return ElemTables(
        max_order=max_order,
        frec_iv=as_t(frec),
        dfact_iv=as_t(dfact),
        int_rec_iv=as_t(int_rec),
        device=device,
    )


# ---------------------------------------------------------------------------
# Small interval helpers mirroring Flow*'s scalar Interval operators
# ---------------------------------------------------------------------------


def _div_int(a: torch.Tensor, i: int) -> torch.Tensor:
    """Interval divided by a nonzero int: Interval::operator/=(double)
    (Interval.cpp:1616) — DIRECT directed endpoint division (1 rounding), NOT
    the rec-then-mul used for interval/interval. [..., 2] -> [..., 2]."""
    if i > 0:
        lo, hi = a[..., 0] / i, a[..., 1] / i
    else:
        lo, hi = a[..., 1] / i, a[..., 0] / i
    return torch.stack((next_down(lo), next_up(hi)), dim=-1)


def _mul_pos_int(a: torch.Tensor, j: int) -> torch.Tensor:
    """Interval times a positive int: Interval::mul_assign(double c > 0)
    (Interval.cpp:1862) — directed endpoint products. [..., 2] -> [..., 2]."""
    return torch.stack((next_down(a[..., 0] * j), next_up(a[..., 1] * j)), dim=-1)


def _double(a: torch.Tensor) -> torch.Tensor:
    """Interval times 2 — EXACT in binary FP (mantissa unchanged; overflow
    saturates to +/-inf which stays a sound outer bound), matching Flow*'s
    exact mpfr `*= 2` in sqrt_taylor. [..., 2] -> [..., 2]."""
    return a * 2.0


def _sanitize(a: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
    """Replace bad lanes' intervals with [0, 0] so no inf/NaN flows onward.

    a [..., 2], bad [...] bool broadcastable to a[..., 0]. Bad lanes are dead
    (caller flags FAILED_DIV); zeroing keeps every downstream torch op finite.
    """
    return torch.where(bad.unsqueeze(-1), torch.zeros_like(a), a)


def _checked_rec(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """iv.rec with NaN-input defense and sanitized (finite) bad lanes."""
    r, bad = iv.rec(a)
    bad = bad | torch.isnan(a).any(dim=-1)
    return _sanitize(r, bad), bad


def _checked_div(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """iv.div (rec-then-mul, Flow* shape) with sanitized bad lanes."""
    r, bad = iv.div(a, b)
    bad = bad | torch.isnan(a).any(dim=-1) | torch.isnan(b).any(dim=-1)
    return _sanitize(r, bad), bad


def _checked_log(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """log_iv with NaN-input defense and sanitized bad lanes."""
    r, bad = log_iv(a)
    bad = bad | torch.isnan(a).any(dim=-1)
    return _sanitize(r, bad), bad


def _checked_sqrt(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """sqrt_iv with NaN-input defense and sanitized bad lanes."""
    r, bad = sqrt_iv(a)
    bad = bad | torch.isnan(a).any(dim=-1)
    return _sanitize(r, bad), bad


def _one_iv(like: torch.Tensor) -> torch.Tensor:
    """The interval constant [1, 1] shaped like `like` ([..., 2])."""
    return torch.ones_like(like)


# ---------------------------------------------------------------------------
# Lagrange remainder formulas (settings.h:322-452, toolbox variants — GOTCHAS #14)
# ---------------------------------------------------------------------------


def exp_taylor_remainder(tm_range: torch.Tensor, order: int, tabs: ElemTables) -> torch.Tensor:
    """settings.h:322: factorial_rec[order] * tmRange^order * exp(tmRange).

    tm_range [..., 2] -> [..., 2]. Multiplication order mirrors the C++
    left-to-right chain (frec * pow) * J.
    """
    prod = iv.pow_int(tm_range, order)
    j = exp_iv(tm_range)
    return iv.mul(iv.mul(tabs.frec_iv[order], prod), j)


def rec_taylor_remainder(
    c: torch.Tensor, tm_range: torch.Tensor, order: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """settings.h:332: (tmRange/(c+tmRange))^order / (c+tmRange), negated for
    odd order. Both divisions are rec-then-mul (Interval::operator/); zero-
    containing denominators flag bad. Returns (rem [..., 2], bad [...])."""
    dom = iv.add(c, tm_range)
    q, bad1 = _checked_div(tm_range, dom)
    q = iv.pow_int(q, order)
    q, bad2 = _checked_div(q, dom)
    if order % 2 == 1:
        q = iv.neg(q)
    return q, bad1 | bad2


def _trig_taylor_remainder(
    c: torch.Tensor, tm_range: torch.Tensor, order: int, tabs: ElemTables, cos_cycle: bool
) -> torch.Tensor:
    """Shared sin/cos Lagrange tail (settings.h:356/386): factorial_rec[order]
    * tmRange^order * D(tmRange + c) where D cycles through the order-th
    derivative of sin (cos_cycle=False) or cos (True) by order mod 4."""
    prod = iv.pow_int(tm_range, order)
    j = iv.add(tm_range, c)
    k4 = (order + 1) % 4 if cos_cycle else order % 4
    # sin cycle by residue: 0 -> sin, 1 -> cos, 2 -> -sin, 3 -> -cos; the cos
    # formula is the same cycle shifted by one (cos = sin', settings.h:393).
    if k4 == 0:
        j = sin_iv(j)
    elif k4 == 1:
        j = cos_iv(j)
    elif k4 == 2:
        j = iv.neg(sin_iv(j))
    else:
        j = iv.neg(cos_iv(j))
    return iv.mul(iv.mul(tabs.frec_iv[order], prod), j)


def sin_taylor_remainder(
    c: torch.Tensor, tm_range: torch.Tensor, order: int, tabs: ElemTables
) -> torch.Tensor:
    """settings.h:356. c, tm_range [..., 2] -> [..., 2]."""
    return _trig_taylor_remainder(c, tm_range, order, tabs, cos_cycle=False)


def cos_taylor_remainder(
    c: torch.Tensor, tm_range: torch.Tensor, order: int, tabs: ElemTables
) -> torch.Tensor:
    """settings.h:386. c, tm_range [..., 2] -> [..., 2]."""
    return _trig_taylor_remainder(c, tm_range, order, tabs, cos_cycle=True)


def log_taylor_remainder(
    tm_range: torch.Tensor, order: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """settings.h:416: (tmRange/(1+tmRange))^order / order, negated when
    order+1 is odd. The final /= order is Flow*'s DIRECT scalar division.
    Returns (rem [..., 2], bad [...])."""
    r, bad = _checked_rec(iv.add(tm_range, _one_iv(tm_range)))
    r = iv.mul(r, tm_range)
    r = iv.pow_int(r, order)
    r = _div_int(r, order)
    if (order + 1) % 2 == 1:
        r = iv.neg(r)
    return r, bad


def sqrt_taylor_remainder(
    tm_range: torch.Tensor, order: int, tabs: ElemTables
) -> tuple[torch.Tensor, torch.Tensor]:
    """settings.h:434: (tmRange / (2*sqrt(1+tmRange)... precisely
    (sqrt(1/(1+tmRange)) * tmRange / 2)^order * (2*order-3)!! / order!,
    negated for even order. Returns (rem [..., 2], bad [...])."""
    r, bad1 = _checked_rec(iv.add(tm_range, _one_iv(tm_range)))
    r, bad2 = _checked_sqrt(r)
    r = iv.mul(r, tm_range)
    r = _div_int(r, 2)
    r = iv.pow_int(r, order)
    factor = iv.mul(tabs.dfact_iv[2 * order - 3], tabs.frec_iv[order])
    r = iv.mul(r, factor)
    if order % 2 == 0:
        r = iv.neg(r)
    return r, bad1 | bad2


# ---------------------------------------------------------------------------
# The shared validated TM multiply (TaylorModel<Interval>::mul_insert_ctrunc_
# normal, TaylorModel.h:829) — also used by ode_compiler.exec_valid
# ---------------------------------------------------------------------------


def tm_mul_valid(
    a_coeffs: torch.Tensor,
    a_rem: torch.Tensor,
    b_coeffs: torch.Tensor,
    b_rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    cache: torch.Tensor,
    cache_at: int,
    range_p2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TaylorModel<Interval>::mul_insert_ctrunc_normal, batched.

    a/b_coeffs [B, T, 2], a/b_rem [B, 2] -> (kept [B, K_k, 2], rem [B, 2]);
    writes cache[:, cache_at .. cache_at+2] = (rangeP1, rangeP2, trunc+round)
    — Flow*'s exact push order. Remainder cross terms in Flow*'s add order:
    I1*I2 + P2*I1 + P1*I2 (TaylorModel.h:851-857), then + trunc+round.

    `range_p2`: the series loops precompute the fixed factor's polyRangeNormal
    ONCE (Flow* passes tmFPolyRange into every multiply); passing it here skips
    the recomputation and caches the identical value. When omitted it is
    computed fresh (the generic MUL/POW tape behavior, expression.h:1544).
    """
    if range_p2 is None:
        range_p2 = poly.range_normal_iv(b_coeffs, tables, step)  # [B, 2] (intPoly2)
    prod = poly.mul_iv(a_coeffs, b_coeffs, tables)  # [B, T2, 2]
    range_p1 = poly.range_normal_iv(a_coeffs, tables, step)  # [B, 2] (intPoly1)

    rem = iv.mul(a_rem, b_rem)
    rem = iv.add(rem, iv.mul(range_p2, a_rem))
    rem = iv.add(rem, iv.mul(range_p1, b_rem))

    kept, tail = poly.ctrunc_normal_iv(prod, tables, step, k)  # [B, K_k, 2], [B, 2]
    kept, round_rng = poly.cutoff_normal_interval(kept, tables, step, cutoff_threshold)
    trunc_round = iv.add(tail, round_rng)  # [B, 2]
    rem = iv.add(rem, trunc_round)

    cache[:, cache_at] = range_p1
    cache[:, cache_at + 1] = range_p2
    cache[:, cache_at + 2] = trunc_round
    return kept, rem


def _pad(kept: torch.Tensor, t_width: int) -> torch.Tensor:
    """Zero-pad a kept prefix [B, K, 2] back to the full slot width [B, T, 2]."""
    out = torch.zeros(kept.shape[0], t_width, 2, dtype=kept.dtype, device=kept.device)
    out[:, : kept.shape[1]] = kept
    return out


def _split_const(coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """tmF.constant(c); tmF.rmConstant() — split [B, T, 2] into the constant
    interval c [B, 2] and the centered polynomial tmF [B, T, 2]."""
    c = coeffs[:, 0].clone()
    tm_f = coeffs.clone()
    tm_f[:, 0] = 0.0
    return c, tm_f


# ---------------------------------------------------------------------------
# Validated series (TaylorModel.h:1266-1895), non-constant path
# ---------------------------------------------------------------------------
# Common cast: the argument TM is (coeffs [B, T, 2], rem [B, 2]); k is BOTH the
# series order and the ctrunc order (Flow* passes one `order` for both);
# `cache`/`base` locate this op's slots; `tabs` the constant tables.


def exp_series_valid(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    cache: torch.Tensor,
    base: int,
    tabs: ElemTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TaylorModel<Interval>::exp_taylor (TaylorModel.h:1266).

    Horner form of 1 + F + F^2/2! + ... + F^k/k!, then * exp(c), then the
    Lagrange tail exp_taylor_remainder(range(F)+rem, k+1) scaled by exp(c).
    Cache: exp(c); per iteration (rangeP1, rangeF, trunc); rangeF. Never bad.
    """
    c, tm_f = _split_const(coeffs)
    exp_c = exp_iv(c)  # [B, 2] const_part.exp_assign()
    cache[:, base] = exp_c
    cursor = base + 1

    t_width = coeffs.shape[1]
    res_c = torch.zeros_like(coeffs)
    res_c[:, 0] = 1.0  # polyOne
    res_r = torch.zeros_like(rem)
    f_range = poly.range_normal_iv(tm_f, tables, step)  # [B, 2] tmFPolyRange (once)

    for i in range(k, 0, -1):
        # result /= i: TM<Interval> /= Interval(i) = rec([i,i]) then interval
        # multiply on every coefficient AND the remainder (TaylorModel.h:592).
        ri = tabs.int_rec_iv[i]  # [2] enclosure of 1/i
        res_c = iv.mul(res_c, ri)
        res_r = iv.mul(res_r, ri)
        kept, res_r = tm_mul_valid(
            res_c, res_r, tm_f, rem, k, tables, step, cutoff_threshold, cache, cursor, f_range
        )
        cursor += 3
        res_c = _pad(kept, t_width)
        res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))  # expansion += 1

    res_c = iv.mul(res_c, exp_c.unsqueeze(1))  # result *= exp(c)
    res_r = iv.mul(res_r, exp_c)
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, rem)
    lag = exp_taylor_remainder(tm_range, k + 1, tabs)
    res_r = iv.add(res_r, iv.mul(exp_c, lag))  # remainder += const_part * rem
    return res_c, res_r


def rec_series_valid(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    cache: torch.Tensor,
    base: int,
    tabs: ElemTables,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TaylorModel<Interval>::rec_taylor (TaylorModel.h:1343).

    Horner form of 1 - F/c + (F/c)^2 - ... over tmF_c = F * (1/c), then * 1/c,
    plus rec_taylor_remainder(c, (range+rem(F/c)) * c, k+1) * (1/c).
    Cache: c; 1/c; per iteration triple; rangeF_c. Bad where 0 in c.
    """
    c, tm_f = _split_const(coeffs)
    rec_c, bad = _checked_rec(c)  # [B, 2] const_part.rec_assign()
    cache[:, base] = c  # c_f
    cache[:, base + 1] = rec_c
    cursor = base + 2

    tmf_c = iv.mul(tm_f, rec_c.unsqueeze(1))  # tmF_c = tmF * (1/c)
    tmf_c_rem = iv.mul(rem, rec_c)
    t_width = coeffs.shape[1]
    res_c = torch.zeros_like(coeffs)
    res_c[:, 0] = 1.0
    res_r = torch.zeros_like(rem)
    f_range = poly.range_normal_iv(tmf_c, tables, step)  # tmF_cPolyRange (once)

    for _ in range(k, 0, -1):
        # result *= -1: interval mul by [-1,-1] is exact negation.
        res_c = iv.neg(res_c)
        res_r = iv.neg(res_r)
        kept, res_r = tm_mul_valid(
            res_c, res_r, tmf_c, tmf_c_rem, k, tables, step, cutoff_threshold,
            cache, cursor, f_range,
        )
        cursor += 3
        res_c = _pad(kept, t_width)
        res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))

    res_c = iv.mul(res_c, rec_c.unsqueeze(1))
    res_r = iv.mul(res_r, rec_c)
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, tmf_c_rem)
    lag, bad2 = rec_taylor_remainder(c, iv.mul(tm_range, c), k + 1)
    res_r = iv.add(res_r, iv.mul(lag, rec_c))  # remainder += rem * const_part
    return res_c, res_r, bad | bad2


def _trig_series_valid(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    cache: torch.Tensor,
    base: int,
    tabs: ElemTables,
    cos_cycle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared sin/cos series (TaylorModel.h:1426/1577).

    result = D0(c) + sum_i F^i * D_i(c) * (1/i!) with D_i the i-th derivative
    (i mod 4 cycle); tmPowerTmF accumulates F^i through mul_insert_ctrunc.
    Cache: c; per iteration (rangeP1, rangeF, trunc, tmp) where tmp is the
    CORRECTED coefficient D_i(c) * enclosure(1/i!) — Flow* stores its buggy
    D_i(c)/i there (GOTCHAS #19, module docstring). Never bad.
    """
    c, tm_f = _split_const(coeffs)
    cache[:, base] = c  # const_part (pushed BEFORE the derivative constants)
    cursor = base + 1

    sinc, cosc = sin_iv(c), cos_iv(c)  # [B, 2] each
    msinc, mcosc = iv.neg(sinc), iv.neg(cosc)
    # Derivative-by-residue tables: sin -> (sin, cos, -sin, -cos)[i mod 4];
    # cos runs the same wheel shifted by one (TaylorModel.h:1630 `int k = 1;
    # ... ++k; k %= 4` vs sin's `k = i % 4`).
    if cos_cycle:
        wheel = {1: msinc, 2: mcosc, 3: sinc, 0: cosc}
        d0 = cosc
    else:
        wheel = {1: cosc, 2: msinc, 3: mcosc, 0: sinc}
        d0 = sinc

    t_width = coeffs.shape[1]
    res_c = torch.zeros_like(coeffs)
    res_c[:, 0] = d0  # result = TM(sin(c) | cos(c)) — interval constant coeff
    res_r = torch.zeros_like(rem)
    pow_c = torch.zeros_like(coeffs)
    pow_c[:, 0] = 1.0  # tmPowerTmF = TM(1)
    pow_r = torch.zeros_like(rem)
    f_range = poly.range_normal_iv(tm_f, tables, step)  # tmFPolyRange (once)

    for i in range(1, k + 1):
        kept, pow_r = tm_mul_valid(
            pow_c, pow_r, tm_f, rem, k, tables, step, cutoff_threshold, cache, cursor, f_range
        )
        cursor += 3
        pow_c = _pad(kept, t_width)
        # GOTCHAS #19 FIX: Flow* computes tmp = D_i(c) / i (direct interval /
        # int); the true Taylor coefficient needs / i!. Same slot, fixed value.
        tmp = iv.mul(wheel[i % 4], tabs.frec_iv[i])  # [B, 2]
        cache[:, cursor] = tmp
        cursor += 1
        res_c = iv.add(res_c, iv.mul(pow_c, tmp.unsqueeze(1)))  # result += pow * tmp
        res_r = iv.add(res_r, iv.mul(pow_r, tmp))

    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, rem)
    if cos_cycle:
        lag = cos_taylor_remainder(c, tm_range, k + 1, tabs)
    else:
        lag = sin_taylor_remainder(c, tm_range, k + 1, tabs)
    res_r = iv.add(res_r, lag)
    return res_c, res_r


def sin_series_valid(
    coeffs, rem, k, tables, step, cutoff_threshold, cache, base, tabs
) -> tuple[torch.Tensor, torch.Tensor]:
    """TaylorModel<Interval>::sin_taylor (TaylorModel.h:1426); see _trig_series_valid."""
    return _trig_series_valid(
        coeffs, rem, k, tables, step, cutoff_threshold, cache, base, tabs, cos_cycle=False
    )


def cos_series_valid(
    coeffs, rem, k, tables, step, cutoff_threshold, cache, base, tabs
) -> tuple[torch.Tensor, torch.Tensor]:
    """TaylorModel<Interval>::cos_taylor (TaylorModel.h:1577); see _trig_series_valid."""
    return _trig_series_valid(
        coeffs, rem, k, tables, step, cutoff_threshold, cache, base, tabs, cos_cycle=True
    )


def log_series_valid(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    cache: torch.Tensor,
    base: int,
    tabs: ElemTables,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TaylorModel<Interval>::log_taylor (TaylorModel.h:1730).

    Horner form of log(c) + F/c - (F/c)^2/2 + ... via result = F_c/k;
    loop i = k-1..1: result -= 1/i (RN point constant!); *= -1; *= F_c.
    Cache: c; k-1 iteration triples; rangeF_c. Bad where c <= 0 (log) or
    0 in c (the /c). NOTE: no final constant multiplier — log adds log(c).
    """
    c, tm_f = _split_const(coeffs)
    cache[:, base] = c  # C, pushed before everything (line 1741)
    cursor = base + 1

    log_c, bad = _checked_log(c)  # const_part.log_assign()
    rec_c, bad2 = _checked_rec(c)  # tmF / C = tmF * rec(C) per coefficient
    bad = bad | bad2
    tmf_c = iv.mul(tm_f, rec_c.unsqueeze(1))
    tmf_c_rem = iv.mul(rem, rec_c)
    # result = tmF_c / order: TM /= Interval(k) = rec([k,k])-then-mul.
    rk = tabs.int_rec_iv[k]
    res_c = iv.mul(tmf_c, rk)
    res_r = iv.mul(tmf_c_rem, rk)
    t_width = coeffs.shape[1]
    f_range = poly.range_normal_iv(tmf_c, tables, step)  # tmF_cPolyRange (once)

    for i in range(k - 1, 0, -1):
        # result -= TM(1/i): Flow* builds the constant as Real (RN double) and
        # subtracts it from the constant coefficient only (TaylorModel.h:1781).
        inv_i = 1.0 / i  # RN, bit-identical to Flow*'s Real division
        res_c[:, 0] = iv.sub(res_c[:, 0], torch.full_like(res_c[:, 0], inv_i))
        res_c = iv.neg(res_c)  # result *= -1
        res_r = iv.neg(res_r)
        kept, res_r = tm_mul_valid(
            res_c, res_r, tmf_c, tmf_c_rem, k, tables, step, cutoff_threshold,
            cache, cursor, f_range,
        )
        cursor += 3
        res_c = _pad(kept, t_width)

    res_c[:, 0] = iv.add(res_c[:, 0], log_c)  # result += TM(log(c))
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, tmf_c_rem)
    lag, bad3 = log_taylor_remainder(tm_range, k + 1)
    res_r = iv.add(res_r, lag)  # remainder += rem (no scaling)
    return res_c, res_r, bad | bad3


def sqrt_series_valid(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    cache: torch.Tensor,
    base: int,
    tabs: ElemTables,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TaylorModel<Interval>::sqrt_taylor (TaylorModel.h:1809).

    sqrt(c) * Horner over F_2c = F/(2c) with coefficients j/(-i) (double-
    factorial ratios), result seeded with F_2c itself. Cache: c; k-1 iteration
    triples; 2*rangeF_2c (the DOUBLED range — unique to sqrt). Bad where c < 0
    (sqrt) or 0 in 2c (the division).
    """
    c, tm_f = _split_const(coeffs)
    cache[:, base] = c  # C, pushed before everything (line 1820)
    cursor = base + 1

    sqrt_c, bad = _checked_sqrt(c)  # const_part.sqrt_assign()
    two_c = _double(c)  # tmp = C; tmp *= 2 (exact)
    rec_2c, bad2 = _checked_rec(two_c)
    bad = bad | bad2
    tmf_2c = iv.mul(tm_f, rec_2c.unsqueeze(1))  # tmF_2c = tmF / (2c)
    tmf_2c_rem = iv.mul(rem, rec_2c)
    res_c = tmf_2c.clone()  # result = tmF_2c
    res_r = tmf_2c_rem.clone()
    t_width = coeffs.shape[1]
    f_range = poly.range_normal_iv(tmf_2c, tables, step)  # tmF_2cPolyRange (once)

    i, j = k, 2 * k - 3
    while i >= 2:
        # tmp = Interval(j) / (-i): direct directed scalar division of the
        # exact-int interval [j, j] — a lane-independent scalar interval.
        q = j / -i  # Python RN division
        if Fraction(q) == Fraction(j, -i):
            tmp_pair = (q, q)
        else:
            tmp_pair = (math.nextafter(q, -math.inf), math.nextafter(q, math.inf))
        tmp = torch.tensor(tmp_pair, dtype=coeffs.dtype, device=coeffs.device)  # [2]
        res_c = iv.mul(res_c, tmp)  # result *= tmp (interval mul)
        res_r = iv.mul(res_r, tmp)
        res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))  # expansion += 1
        kept, res_r = tm_mul_valid(
            res_c, res_r, tmf_2c, tmf_2c_rem, k, tables, step, cutoff_threshold,
            cache, cursor, f_range,
        )
        cursor += 3
        res_c = _pad(kept, t_width)
        i, j = i - 1, j - 2

    res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))  # expansion += 1
    res_c = iv.mul(res_c, sqrt_c.unsqueeze(1))  # result *= sqrt(c)
    res_r = iv.mul(res_r, sqrt_c)
    f2 = _double(f_range)  # tmF_cRange = 2 * tmF_2cPolyRange (exact)
    cache[:, cursor] = f2
    tm_range = iv.add(f2, _double(tmf_2c_rem))
    lag, bad3 = sqrt_taylor_remainder(tm_range, k + 1, tabs)
    res_r = iv.add(res_r, iv.mul(lag, sqrt_c))  # remainder += rem * const_part
    return res_c, res_r, bad | bad3


# ---------------------------------------------------------------------------
# Replay twins (expression.h:37-384) — remainder-only, cache consumed in order
# ---------------------------------------------------------------------------
# Common cast: `rem` [B, 2] is the (refined) candidate remainder of the
# ARGUMENT TM; returns the op's result remainder [B, 2]. The non-constant
# layout is always taken (GOTCHAS #20: sentinels folded out at compile time).


def exp_series_replay(
    rem: torch.Tensor, cache: torch.Tensor, base: int, k: int, tabs: ElemTables
) -> torch.Tensor:
    """exp_taylor_only_remainder (expression.h:37).

    NOTE the Flow* asymmetry mirrored here: the replay's `result /= i` is a
    DIRECT scalar interval division (Interval::operator/=(double)) while the
    valid pass divides the TM via rec([i,i])-then-mul — so replay==valid holds
    only to reassociation/rounding ulps, not bitwise.
    """
    const_part = cache[:, base]  # exp(c)
    cursor = base + 1
    result = torch.zeros_like(rem)
    for i in range(k, 0, -1):
        result = _div_int(result, i)
        t = iv.mul(cache[:, cursor], rem)  # P1 x I2
        t = iv.add(t, iv.mul(cache[:, cursor + 1], result))  # P2 x I1
        t = iv.add(t, iv.mul(rem, result))  # I2 x I1
        t = iv.add(t, cache[:, cursor + 2])  # truncation
        cursor += 3
        result = t
    result = iv.mul(result, const_part)
    tm_range = iv.add(cache[:, cursor], rem)
    lag = exp_taylor_remainder(tm_range, k + 1, tabs)
    return iv.add(result, iv.mul(const_part, lag))


def rec_series_replay(
    rem: torch.Tensor, cache: torch.Tensor, base: int, k: int, tabs: ElemTables
) -> tuple[torch.Tensor, torch.Tensor]:
    """rec_taylor_only_remainder (expression.h:88). Returns (rem, bad)."""
    c_f = cache[:, base]
    const_part = cache[:, base + 1]  # 1/c
    cursor = base + 2
    r2 = iv.mul(rem, const_part)  # tmF_c_remainder = remainder * const_part
    result = torch.zeros_like(rem)
    for _ in range(k, 0, -1):
        result = iv.neg(result)  # result *= -1
        t = iv.mul(cache[:, cursor], r2)  # P1 x I2
        t = iv.add(t, iv.mul(cache[:, cursor + 1], result))  # P2 x I1
        t = iv.add(t, iv.mul(r2, result))  # I2 x I1
        t = iv.add(t, cache[:, cursor + 2])  # truncation
        cursor += 3
        result = t
    result = iv.mul(result, const_part)
    tm_range = iv.add(cache[:, cursor], r2)
    lag, bad = rec_taylor_remainder(c_f, iv.mul(tm_range, c_f), k + 1)
    return iv.add(result, iv.mul(lag, const_part)), bad


def _trig_series_replay(
    rem: torch.Tensor, cache: torch.Tensor, base: int, k: int, tabs: ElemTables, cos_cycle: bool
) -> torch.Tensor:
    """sin/cos_taylor_only_remainder (expression.h:145/206), shared body."""
    const_part = cache[:, base]  # c
    cursor = base + 1
    pow_r = torch.zeros_like(rem)  # tmPowerTmF_remainder
    result = torch.zeros_like(rem)
    for _ in range(1, k + 1):
        t = iv.mul(cache[:, cursor], rem)  # P1 x I2
        t = iv.add(t, iv.mul(cache[:, cursor + 1], pow_r))  # P2 x I1
        t = iv.add(t, iv.mul(rem, pow_r))  # I2 x I1
        t = iv.add(t, cache[:, cursor + 2])  # truncation
        cursor += 3
        pow_r = t
        result = iv.add(result, iv.mul(pow_r, cache[:, cursor]))  # * tmp slot
        cursor += 1
    tm_range = iv.add(cache[:, cursor], rem)
    if cos_cycle:
        lag = cos_taylor_remainder(const_part, tm_range, k + 1, tabs)
    else:
        lag = sin_taylor_remainder(const_part, tm_range, k + 1, tabs)
    return iv.add(result, lag)


def sin_series_replay(rem, cache, base, k, tabs) -> torch.Tensor:
    """sin_taylor_only_remainder (expression.h:145)."""
    return _trig_series_replay(rem, cache, base, k, tabs, cos_cycle=False)


def cos_series_replay(rem, cache, base, k, tabs) -> torch.Tensor:
    """cos_taylor_only_remainder (expression.h:206)."""
    return _trig_series_replay(rem, cache, base, k, tabs, cos_cycle=True)


def log_series_replay(
    rem: torch.Tensor, cache: torch.Tensor, base: int, k: int, tabs: ElemTables
) -> tuple[torch.Tensor, torch.Tensor]:
    """log_taylor_only_remainder (expression.h:263). Returns (rem, bad).

    Asymmetries mirrored from Flow*: the twin divides the remainder by C via
    interval division (rec-then-mul) but by `order` via DIRECT scalar division
    (div_assign), where the valid pass used rec([k,k])-then-mul.
    """
    c = cache[:, base]  # C
    cursor = base + 1
    r2, bad = _checked_div(rem, c)  # tmF_c_remainder = remainder / C
    result = _div_int(r2, k)  # result.div_assign((double)order)
    for _ in range(k, 1, -1):
        result = iv.neg(result)  # result.inv_assign()
        t = iv.mul(cache[:, cursor], r2)  # P1 x I2
        t = iv.add(t, iv.mul(cache[:, cursor + 1], result))  # P2 x I1
        t = iv.add(t, iv.mul(r2, result))  # I2 x I1
        t = iv.add(t, cache[:, cursor + 2])  # truncation
        cursor += 3
        result = t
    tm_range = iv.add(cache[:, cursor], r2)
    lag, bad2 = log_taylor_remainder(tm_range, k + 1)
    return iv.add(result, lag), bad | bad2


def sqrt_series_replay(
    rem: torch.Tensor, cache: torch.Tensor, base: int, k: int, tabs: ElemTables
) -> tuple[torch.Tensor, torch.Tensor]:
    """sqrt_taylor_only_remainder (expression.h:322). Returns (rem, bad).

    Mirrored asymmetries: the twin computes (rem / C) / 2 (interval division
    then direct halving) where the valid pass divided by the interval 2C; the
    loop scales by /(-i) then *j directly where the valid pass built the
    scalar interval j/(-i) first.
    """
    c = cache[:, base]  # C
    cursor = base + 1
    r_c, bad = _checked_div(rem, c)
    r2 = _div_int(r_c, 2)  # tmF_2c_remainder = (remainder / C) / 2
    result = r2.clone()
    i, j = k, 2 * k - 3
    while i >= 2:
        result = _div_int(result, -i)  # result /= -i
        result = _mul_pos_int(result, j)  # result *= j
        t = iv.mul(cache[:, cursor], r2)  # P1 x I2
        t = iv.add(t, iv.mul(cache[:, cursor + 1], result))  # P2 x I1
        t = iv.add(t, iv.mul(r2, result))  # I2 x I1
        t = iv.add(t, cache[:, cursor + 2])  # truncation
        cursor += 3
        result = t
        i, j = i - 1, j - 2
    const_part, bad2 = _checked_sqrt(c)  # const_part = sqrt(C)
    result = iv.mul(result, const_part)
    tm_range = iv.add(cache[:, cursor], _double(r2))  # cached 2*range + 2*rem
    lag, bad3 = sqrt_taylor_remainder(tm_range, k + 1, tabs)
    return iv.add(result, iv.mul(lag, const_part)), bad | bad2 | bad3


# ---------------------------------------------------------------------------
# Point-coefficient series (Polynomial<Real>::*_taylor, Polynomial.h:1892-2223)
# — tier P (plain RN, no remainders), used by ode_compiler.exec_point
# ---------------------------------------------------------------------------


def mul_no_remainder(
    a: torch.Tensor, b: torch.Tensor, k: int, tables: MonomialTables, cutoff_threshold: float
) -> torch.Tensor:
    """Point product + nctrunc(k) + drop-cutoff — the `result *= F; nctrunc;
    cutoff` step of every point series (and exec_point's MUL). [B, T] x
    [B, T] -> [B, T] (zero beyond the degree-k prefix)."""
    keep = tables.prefix_len_cpu[k + 1]
    prod = poly.mul_point(a, b, tables)  # [B, T2]
    kept = poly.cutoff_drop(prod[..., :keep], cutoff_threshold)
    out = torch.zeros_like(a)
    out[..., :keep] = kept
    return out


def _split_const_point(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Point-tier constant split: [B, T] -> (c [B], F [B, T] with F[0] = 0)."""
    c = x[..., 0].clone()
    f = x.clone()
    f[..., 0] = 0.0
    return c, f


def exp_series_point(
    x: torch.Tensor, k: int, tables: MonomialTables, cutoff_threshold: float
) -> torch.Tensor:
    """Polynomial<Real>::exp_taylor (Polynomial.h:1892). [B, T] -> [B, T]."""
    c, f = _split_const_point(x)
    result = torch.zeros_like(x)
    result[..., 0] = 1.0
    for i in range(k, 0, -1):
        result = result / i  # RN division, exactly Flow*'s Real /= i
        result = mul_no_remainder(result, f, k, tables, cutoff_threshold)
        result[..., 0] += 1.0
    return result * torch.exp(c).unsqueeze(-1)  # RN, like Real::exp_assign


def rec_series_point(
    x: torch.Tensor, k: int, tables: MonomialTables, cutoff_threshold: float
) -> torch.Tensor:
    """Polynomial<Real>::rec_taylor (Polynomial.h:1933). [B, T] -> [B, T].

    Tier P has no failure channel: a zero constant produces inf/NaN
    coefficients (as Flow*'s Real would) and the lane is flagged FAILED_DIV by
    the validated pass.
    """
    c, f = _split_const_point(x)
    rec_c = 1.0 / c  # [B] RN (Real::rec_assign)
    f_c = f * rec_c.unsqueeze(-1)  # F_c = F * (1/c), per-coeff RN
    result = torch.zeros_like(x)
    result[..., 0] = 1.0
    for _ in range(k, 0, -1):
        result = -result
        result = mul_no_remainder(result, f_c, k, tables, cutoff_threshold)
        result[..., 0] += 1.0
    return result * rec_c.unsqueeze(-1)


def _trig_series_point(
    x: torch.Tensor, k: int, tables: MonomialTables, cutoff_threshold: float, cos_cycle: bool
) -> torch.Tensor:
    """Polynomial<Real>::sin/cos_taylor (Polynomial.h:1975/2057), with the
    GOTCHAS #19 factorial FIX: coefficient D_i(c)/i! (RN division by the
    correctly-rounded double of i!) instead of Flow*'s buggy D_i(c)/i.

    Structure mirrored exactly otherwise: polyPowerF is nctrunc'd but NOT
    cutoff per iteration; one final cutoff on the result (Polynomial.h:2053).
    """
    c, f = _split_const_point(x)
    keep = tables.prefix_len_cpu[k + 1]
    sinc, cosc = torch.sin(c), torch.cos(c)  # [B] RN
    if cos_cycle:
        wheel = {1: -sinc, 2: -cosc, 3: sinc, 0: cosc}
        d0 = cosc
    else:
        wheel = {1: cosc, 2: -sinc, 3: -cosc, 0: sinc}
        d0 = sinc
    result = torch.zeros_like(x)
    result[..., 0] = d0
    pow_f = torch.zeros_like(x)
    pow_f[..., 0] = 1.0
    for i in range(1, k + 1):
        prod = poly.mul_point(pow_f, f, tables)  # polyPowerF *= F
        pow_f = torch.zeros_like(x)
        pow_f[..., :keep] = prod[..., :keep]  # nctrunc(order), no cutoff
        tmp = wheel[i % 4] / float(math.factorial(i))  # [B] the FIXED coefficient
        result = result + pow_f * tmp.unsqueeze(-1)
    return poly.cutoff_drop(result, cutoff_threshold)


def sin_series_point(x, k, tables, cutoff_threshold) -> torch.Tensor:
    """Polynomial<Real>::sin_taylor (Polynomial.h:1975) + GOTCHAS #19 fix."""
    return _trig_series_point(x, k, tables, cutoff_threshold, cos_cycle=False)


def cos_series_point(x, k, tables, cutoff_threshold) -> torch.Tensor:
    """Polynomial<Real>::cos_taylor (Polynomial.h:2057) + GOTCHAS #19 fix."""
    return _trig_series_point(x, k, tables, cutoff_threshold, cos_cycle=True)


def log_series_point(
    x: torch.Tensor, k: int, tables: MonomialTables, cutoff_threshold: float
) -> torch.Tensor:
    """Polynomial<Real>::log_taylor (Polynomial.h:2139). [B, T] -> [B, T]."""
    c, f = _split_const_point(x)
    f_c = f / c.unsqueeze(-1)  # F / C: per-coeff RN division (Real semantics)
    result = f_c / k  # result = F_c / order
    for i in range(k - 1, 0, -1):
        result[..., 0] -= 1.0 / i  # result -= TM(Real 1/i)
        result = -result
        result = mul_no_remainder(result, f_c, k, tables, cutoff_threshold)
    result[..., 0] += torch.log(c)
    return result


def sqrt_series_point(
    x: torch.Tensor, k: int, tables: MonomialTables, cutoff_threshold: float
) -> torch.Tensor:
    """Polynomial<Real>::sqrt_taylor (Polynomial.h:2183). [B, T] -> [B, T]."""
    c, f = _split_const_point(x)
    f_2c = f / (2.0 * c).unsqueeze(-1)  # F / (2C), RN
    result = f_2c.clone()
    i, j = k, 2 * k - 3
    while i >= 2:
        result = result * (j / -i)  # tmp = Real(j) / (-i), RN scalar
        result[..., 0] += 1.0
        result = mul_no_remainder(result, f_2c, k, tables, cutoff_threshold)
        i, j = i - 1, j - 2
    result[..., 0] += 1.0
    return result * torch.sqrt(c).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Static cache layout (used by ode_compiler.compile_ode)
# ---------------------------------------------------------------------------


def cache_len(op: str, order: int) -> int:
    """Cache slots op consumes at series order `order` (module docstring table)."""
    if op == "exp":
        return 3 * order + 2
    if op in ("sin", "cos"):
        return 4 * order + 2
    if op in ("log", "sqrt"):
        return 3 * order - 1
    if op == "div":  # rec layout (3*order + 3) + the final multiply's triple
        return 3 * order + 6
    raise ValueError(f"no elementary cache layout for op {op!r}")


# ---------------------------------------------------------------------------
# Generic ctx-parameterized series bodies (M10 union-support sparsity)
# ---------------------------------------------------------------------------
# The union-support sparse layout (support.py / sparse_exec.py) needs the
# SAME series loops on support-aligned tensors. Rather than duplicate the
# subtlest soundness-bearing code in the project, the loops below are the
# dense bodies above transcribed with every layout-dependent step routed
# through a context object; tests/unit/test_sparse.py asserts the generic
# body + DenseSeriesCtx reproduces each dense function BITWISE, which pins
# the transcription. Layout-independent steps (constant-slot updates at
# index 0 — the constant monomial is slot 0 in BOTH layouts by the Support
# id-0 invariant — and elementwise interval scalings) stay inline.
#
# Valid-tier context protocol (duck-typed; B batch, S the context's layout):
#   mul_valid(a_c [B, S, 2], a_r [B, 2], cache, at) -> (a_c' [B, S', 2], a_r')
#       one mul_insert_ctrunc_normal of a against the FIXED series argument F
#       (kept product re-laid-out per context; cache triple written at `at`).
#   range_f() -> [B, 2]  rangeNormal of F's polynomial (computed once).
#   acc_add(res_c, term_c) -> res_c'  layout-aligning interval add (sin/cos).
#   fresh_one() / fresh_zero() -> constant-1 / all-zero polynomial seeds.
#   seed_f() -> a fresh copy of F itself (log/sqrt seeds).
# Point-tier protocol: mul(a) [nctrunc+cutoff], mul_nc(a) [nctrunc only],
#   acc_add, fresh_one, fresh_zero, seed_f.


class DenseSeriesCtx:
    """Dense [B, T, 2] valid-tier context — bit-for-bit the historical path."""

    def __init__(self, tm_f, f_rem, k, tables, step, cutoff_threshold):
        self.tm_f = tm_f  # [B, T, 2] FIXED (centered, pre-scaled) argument
        self.f_rem = f_rem  # [B, 2] its remainder twin
        self.k = k
        self.tables = tables
        self.step = step
        self.cutoff = cutoff_threshold
        self._range_f = poly.range_normal_iv(tm_f, tables, step)  # [B, 2] once

    def range_f(self):
        return self._range_f

    def mul_valid(self, a_c, a_r, cache, at):
        kept, rem = tm_mul_valid(
            a_c, a_r, self.tm_f, self.f_rem, self.k, self.tables, self.step,
            self.cutoff, cache, at, self._range_f,
        )
        return _pad(kept, self.tm_f.shape[1]), rem

    def acc_add(self, res_c, term_c):
        return iv.add(res_c, term_c)

    def fresh_one(self):
        out = torch.zeros_like(self.tm_f)
        out[:, 0] = 1.0
        return out

    def fresh_zero(self):
        return torch.zeros_like(self.tm_f)

    def seed_f(self):
        return self.tm_f.clone()


class DensePointCtx:
    """Dense [B, T] point-tier context — the historical exec_point plumbing."""

    def __init__(self, f, k, tables, cutoff_threshold):
        self.f = f  # [B, T] FIXED argument polynomial
        self.k = k
        self.tables = tables
        self.cutoff = cutoff_threshold
        self.keep = tables.prefix_len_cpu[k + 1]

    def mul(self, a):
        return mul_no_remainder(a, self.f, self.k, self.tables, self.cutoff)

    def mul_nc(self, a):
        # nctrunc WITHOUT cutoff (sin/cos power chain, Polynomial.h:2033).
        prod = poly.mul_point(a, self.f, self.tables)  # [B, T2]
        out = torch.zeros_like(a)
        out[..., : self.keep] = prod[..., : self.keep]
        return out

    def acc_add(self, res, term):
        return res + term

    def fresh_one(self):
        out = torch.zeros_like(self.f)
        out[..., 0] = 1.0
        return out

    def fresh_zero(self):
        return torch.zeros_like(self.f)

    def seed_f(self):
        return self.f.clone()


def exp_series_valid_g(coeffs, rem, k, cache, base, tabs, make_ctx):
    """Generic exp_taylor body; make_ctx(F, F_rem) supplies the layout."""
    c, tm_f = _split_const(coeffs)
    exp_c = exp_iv(c)
    cache[:, base] = exp_c
    cursor = base + 1
    ctx = make_ctx(tm_f, rem)
    res_c = ctx.fresh_one()
    res_r = torch.zeros_like(rem)
    f_range = ctx.range_f()
    for i in range(k, 0, -1):
        ri = tabs.int_rec_iv[i]
        res_c = iv.mul(res_c, ri)
        res_r = iv.mul(res_r, ri)
        res_c, res_r = ctx.mul_valid(res_c, res_r, cache, cursor)
        cursor += 3
        res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))
    res_c = iv.mul(res_c, exp_c.unsqueeze(1))
    res_r = iv.mul(res_r, exp_c)
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, rem)
    lag = exp_taylor_remainder(tm_range, k + 1, tabs)
    res_r = iv.add(res_r, iv.mul(exp_c, lag))
    return res_c, res_r


def rec_series_valid_g(coeffs, rem, k, cache, base, tabs, make_ctx):
    """Generic rec_taylor body. Returns (res_c, res_r, bad)."""
    c, tm_f = _split_const(coeffs)
    rec_c, bad = _checked_rec(c)
    cache[:, base] = c
    cache[:, base + 1] = rec_c
    cursor = base + 2
    tmf_c = iv.mul(tm_f, rec_c.unsqueeze(1))
    tmf_c_rem = iv.mul(rem, rec_c)
    ctx = make_ctx(tmf_c, tmf_c_rem)
    res_c = ctx.fresh_one()
    res_r = torch.zeros_like(rem)
    f_range = ctx.range_f()
    for _ in range(k, 0, -1):
        res_c = iv.neg(res_c)
        res_r = iv.neg(res_r)
        res_c, res_r = ctx.mul_valid(res_c, res_r, cache, cursor)
        cursor += 3
        res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))
    res_c = iv.mul(res_c, rec_c.unsqueeze(1))
    res_r = iv.mul(res_r, rec_c)
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, tmf_c_rem)
    lag, bad2 = rec_taylor_remainder(c, iv.mul(tm_range, c), k + 1)
    res_r = iv.add(res_r, iv.mul(lag, rec_c))
    return res_c, res_r, bad | bad2


def _trig_series_valid_g(coeffs, rem, k, cache, base, tabs, make_ctx, cos_cycle):
    """Generic sin/cos_taylor body (GOTCHAS #19 corrected coefficients)."""
    c, tm_f = _split_const(coeffs)
    cache[:, base] = c
    cursor = base + 1
    sinc, cosc = sin_iv(c), cos_iv(c)
    msinc, mcosc = iv.neg(sinc), iv.neg(cosc)
    if cos_cycle:
        wheel = {1: msinc, 2: mcosc, 3: sinc, 0: cosc}
        d0 = cosc
    else:
        wheel = {1: cosc, 2: msinc, 3: mcosc, 0: sinc}
        d0 = sinc
    ctx = make_ctx(tm_f, rem)
    res_c = ctx.fresh_zero()
    res_c[:, 0] = d0
    res_r = torch.zeros_like(rem)
    pow_c = ctx.fresh_one()
    pow_r = torch.zeros_like(rem)
    f_range = ctx.range_f()
    for i in range(1, k + 1):
        pow_c, pow_r = ctx.mul_valid(pow_c, pow_r, cache, cursor)
        cursor += 3
        tmp = iv.mul(wheel[i % 4], tabs.frec_iv[i])
        cache[:, cursor] = tmp
        cursor += 1
        res_c = ctx.acc_add(res_c, iv.mul(pow_c, tmp.unsqueeze(1)))
        res_r = iv.add(res_r, iv.mul(pow_r, tmp))
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, rem)
    if cos_cycle:
        lag = cos_taylor_remainder(c, tm_range, k + 1, tabs)
    else:
        lag = sin_taylor_remainder(c, tm_range, k + 1, tabs)
    res_r = iv.add(res_r, lag)
    return res_c, res_r


def log_series_valid_g(coeffs, rem, k, cache, base, tabs, make_ctx):
    """Generic log_taylor body. Returns (res_c, res_r, bad)."""
    c, tm_f = _split_const(coeffs)
    cache[:, base] = c
    cursor = base + 1
    log_c, bad = _checked_log(c)
    rec_c, bad2 = _checked_rec(c)
    bad = bad | bad2
    tmf_c = iv.mul(tm_f, rec_c.unsqueeze(1))
    tmf_c_rem = iv.mul(rem, rec_c)
    ctx = make_ctx(tmf_c, tmf_c_rem)
    rk = tabs.int_rec_iv[k]
    res_c = iv.mul(ctx.seed_f(), rk)
    res_r = iv.mul(tmf_c_rem, rk)
    f_range = ctx.range_f()
    for i in range(k - 1, 0, -1):
        inv_i = 1.0 / i  # RN, bit-identical to Flow*'s Real division
        res_c[:, 0] = iv.sub(res_c[:, 0], torch.full_like(res_c[:, 0], inv_i))
        res_c = iv.neg(res_c)
        res_r = iv.neg(res_r)
        res_c, res_r = ctx.mul_valid(res_c, res_r, cache, cursor)
        cursor += 3
    res_c[:, 0] = iv.add(res_c[:, 0], log_c)
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, tmf_c_rem)
    lag, bad3 = log_taylor_remainder(tm_range, k + 1)
    res_r = iv.add(res_r, lag)
    return res_c, res_r, bad | bad3


def sqrt_series_valid_g(coeffs, rem, k, cache, base, tabs, make_ctx):
    """Generic sqrt_taylor body. Returns (res_c, res_r, bad)."""
    c, tm_f = _split_const(coeffs)
    cache[:, base] = c
    cursor = base + 1
    sqrt_c, bad = _checked_sqrt(c)
    two_c = _double(c)
    rec_2c, bad2 = _checked_rec(two_c)
    bad = bad | bad2
    tmf_2c = iv.mul(tm_f, rec_2c.unsqueeze(1))
    tmf_2c_rem = iv.mul(rem, rec_2c)
    ctx = make_ctx(tmf_2c, tmf_2c_rem)
    res_c = ctx.seed_f()
    res_r = tmf_2c_rem.clone()
    f_range = ctx.range_f()
    i, j = k, 2 * k - 3
    while i >= 2:
        q = j / -i
        if Fraction(q) == Fraction(j, -i):
            tmp_pair = (q, q)
        else:
            tmp_pair = (math.nextafter(q, -math.inf), math.nextafter(q, math.inf))
        tmp = torch.tensor(tmp_pair, dtype=coeffs.dtype, device=coeffs.device)
        res_c = iv.mul(res_c, tmp)
        res_r = iv.mul(res_r, tmp)
        res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))
        res_c, res_r = ctx.mul_valid(res_c, res_r, cache, cursor)
        cursor += 3
        i, j = i - 1, j - 2
    res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))
    res_c = iv.mul(res_c, sqrt_c.unsqueeze(1))
    res_r = iv.mul(res_r, sqrt_c)
    f2 = _double(f_range)
    cache[:, cursor] = f2
    tm_range = iv.add(f2, _double(tmf_2c_rem))
    lag, bad3 = sqrt_taylor_remainder(tm_range, k + 1, tabs)
    res_r = iv.add(res_r, iv.mul(lag, sqrt_c))
    return res_c, res_r, bad | bad3


def exp_series_point_g(x, k, make_ctx):
    """Generic point exp_taylor body; make_ctx(F) supplies the layout."""
    c, f = _split_const_point(x)
    ctx = make_ctx(f)
    result = ctx.fresh_one()
    for i in range(k, 0, -1):
        result = result / i  # RN, Flow* Real /= i
        result = ctx.mul(result)
        result[..., 0] += 1.0
    return result * torch.exp(c).unsqueeze(-1)


def rec_series_point_g(x, k, make_ctx):
    """Generic point rec_taylor body."""
    c, f = _split_const_point(x)
    rec_c = 1.0 / c  # [B] RN (Real::rec_assign)
    f_c = f * rec_c.unsqueeze(-1)
    ctx = make_ctx(f_c)
    result = ctx.fresh_one()
    for _ in range(k, 0, -1):
        result = -result
        result = ctx.mul(result)
        result[..., 0] += 1.0
    return result * rec_c.unsqueeze(-1)


def _trig_series_point_g(x, k, make_ctx, cutoff_threshold, cos_cycle):
    """Generic point sin/cos_taylor body (+ GOTCHAS #19 fix)."""
    c, f = _split_const_point(x)
    sinc, cosc = torch.sin(c), torch.cos(c)
    if cos_cycle:
        wheel = {1: -sinc, 2: -cosc, 3: sinc, 0: cosc}
        d0 = cosc
    else:
        wheel = {1: cosc, 2: -sinc, 3: -cosc, 0: sinc}
        d0 = sinc
    ctx = make_ctx(f)
    result = ctx.fresh_zero()
    result[..., 0] = d0
    pow_f = ctx.fresh_one()
    for i in range(1, k + 1):
        pow_f = ctx.mul_nc(pow_f)
        tmp = wheel[i % 4] / float(math.factorial(i))
        result = ctx.acc_add(result, pow_f * tmp.unsqueeze(-1))
    return poly.cutoff_drop(result, cutoff_threshold)


def log_series_point_g(x, k, make_ctx):
    """Generic point log_taylor body."""
    c, f = _split_const_point(x)
    f_c = f / c.unsqueeze(-1)
    ctx = make_ctx(f_c)
    result = ctx.seed_f() / k
    for i in range(k - 1, 0, -1):
        result[..., 0] -= 1.0 / i
        result = -result
        result = ctx.mul(result)
    result[..., 0] += torch.log(c)
    return result


def sqrt_series_point_g(x, k, make_ctx):
    """Generic point sqrt_taylor body."""
    c, f = _split_const_point(x)
    f_2c = f / (2.0 * c).unsqueeze(-1)
    ctx = make_ctx(f_2c)
    result = ctx.seed_f()
    i, j = k, 2 * k - 3
    while i >= 2:
        result = result * (j / -i)
        result[..., 0] += 1.0
        result = ctx.mul(result)
        i, j = i - 1, j - 2
    result[..., 0] += 1.0
    return result * torch.sqrt(c).unsqueeze(-1)


# ---------------------------------------------------------------------------
# tanh / sigmoid series (poly-CROWN P3 — TM-through-NN activation composition)
#
# Unlike sin/cos/exp, the Taylor coefficients of tanh/sigmoid at a midpoint c
# have no closed form; they follow from the defining ODEs
#     tanh'    = 1 − tanh²        =>  (n+1)·a_{n+1} = [n=0] − Σ_{i+j=n} a_i a_j
#     sigmoid' = sigmoid·(1−sigmoid) => (n+1)·b_{n+1} = b_n − Σ_{i+j=n} b_i b_j
# evaluated here in INTERVAL arithmetic from the transcendental enclosure of
# f(c) — every coefficient is a sound enclosure of the true derivative/n!.
# The Lagrange tail comes from the derivative-polynomial identity
#     f^{(n)}(x) = P_n(f(x)),  P_1 the ODE RHS,  P_{n+1} = P_n'·P_1
# with EXACT integer coefficients (host-cached; |coeff| < 2^53 asserted), so
#     |rem| ⊆ P_{k+1}(f(range))·range^{k+1}/(k+1)!
# interval-evaluated over f(range) ⊆ [-1,1] (tanh) / [0,1] (sigmoid) — finite
# and TIGHT even for saturated neurons where an exp-composite overflows
# (Docking's ±100-scale pre-activations; the reason these ops exist).
# Prototype validation: 400 random (c, h, k) mpmath trials, zero containment
# failures, worst true-error/tail = 0.999 (PROGRESS 2026-08-05).
# ---------------------------------------------------------------------------

_DERIV_POLY_CACHE: dict = {}


def _derivative_polys(kind: str, nmax: int) -> list:
    """[None, P_1, ..., P_nmax] integer coefficient lists (ascending powers)."""
    key = (kind, nmax)
    r = _DERIV_POLY_CACHE.get(key)
    if r is not None:
        return r
    mul = [1, 0, -1] if kind == "tanh" else [0, 1, -1]
    p = list(mul)
    out = [None, p]
    for _ in range(2, nmax + 1):
        dp = [p[i] * i for i in range(1, len(p))]
        nxt = [0] * (len(dp) + len(mul) - 1)
        for i, a in enumerate(dp):
            for j, b in enumerate(mul):
                nxt[i + j] += a * b
        p = nxt
        out.append(p)
    biggest = max(abs(cf) for q in out[1:] for cf in q)
    assert biggest < 2**53, (
        f"derivative-poly coefficient {biggest} not exactly representable in "
        f"float64 — cap the series order (n <= 16 is safe)")
    _DERIV_POLY_CACHE[key] = out
    return out


def _poly_int_iv_eval(coeffs_int: list, y: torch.Tensor) -> torch.Tensor:
    """Interval evaluation of an integer-coefficient polynomial: [B, 2] y ->
    [B, 2]. Term-wise pow_int + exact-coefficient scale (each cf is exactly a
    float64), outward-rounded sums."""
    total = torch.zeros_like(y)
    for i, cf in enumerate(coeffs_int):
        if cf == 0:
            continue
        if i == 0:
            term = torch.zeros_like(y)
            term[..., 0] = float(cf)
            term[..., 1] = float(cf)
        else:
            term = iv.mul_point(
                iv.pow_int(y, i), torch.full_like(y[..., 0], float(cf)))
        total = iv.add(total, term)
    return total


def _sigmoidal_series_coeffs(
    c: torch.Tensor, k: int, kind: str, tabs: ElemTables
) -> list:
    """Interval Taylor coefficients a_0..a_k of tanh/sigmoid at c [B, 2]."""
    a = [tanh_iv(c) if kind == "tanh" else sigmoid_iv(c)]
    one = torch.zeros_like(c)
    one[..., 0] = 1.0
    one[..., 1] = 1.0
    for n in range(k):
        conv = iv.mul(a[0], a[n])
        for j in range(1, n + 1):
            conv = iv.add(conv, iv.mul(a[j], a[n - j]))
        if kind == "tanh":
            num = iv.neg(conv)
            if n == 0:
                num = iv.add(num, one)
        else:
            num = iv.sub(a[n], conv)
        a.append(iv.mul(num, tabs.int_rec_iv[n + 1]))
    return a


def _sigmoidal_series_valid(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    cache: torch.Tensor,
    base: int,
    tabs: ElemTables,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared tanh/sigmoid series body (the _trig_series_valid pattern with
    recurrence coefficients instead of the sin/cos wheel).

    result = a_0(c) + Σ_i F^i · a_i(c), Lagrange tail from P_{k+1} over the
    transcendental enclosure of f(c + range(F) + rem). Cache: c; per
    iteration (rangeP1, rangeF, trunc, a_i); rangeF — 4k + 2 slots. Never
    bad: both functions are entire and their enclosures are codomain-clamped.
    """
    c, tm_f = _split_const(coeffs)
    cache[:, base] = c
    cursor = base + 1

    a = _sigmoidal_series_coeffs(c, k, kind, tabs)
    t_width = coeffs.shape[1]
    res_c = torch.zeros_like(coeffs)
    res_c[:, 0] = a[0]
    res_r = torch.zeros_like(rem)
    pow_c = torch.zeros_like(coeffs)
    pow_c[:, 0] = 1.0
    pow_r = torch.zeros_like(rem)
    f_range = poly.range_normal_iv(tm_f, tables, step)  # [B, 2] (once)

    for i in range(1, k + 1):
        kept, pow_r = tm_mul_valid(
            pow_c, pow_r, tm_f, rem, k, tables, step, cutoff_threshold,
            cache, cursor, f_range,
        )
        cursor += 3
        pow_c = _pad(kept, t_width)
        cache[:, cursor] = a[i]
        cursor += 1
        res_c = iv.add(res_c, iv.mul(pow_c, a[i].unsqueeze(1)))
        res_r = iv.add(res_r, iv.mul(pow_r, a[i]))

    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, rem)
    y_rng = (tanh_iv if kind == "tanh" else sigmoid_iv)(iv.add(c, tm_range))
    p_top = _derivative_polys(kind, k + 1)[k + 1]
    m_iv = _poly_int_iv_eval(p_top, y_rng)  # ⊇ f^{(k+1)} over the argument range
    lag = iv.mul(iv.mul(m_iv, iv.pow_int(tm_range, k + 1)), tabs.frec_iv[k + 1])
    res_r = iv.add(res_r, lag)
    return res_c, res_r


def tanh_series_valid(coeffs, rem, k, tables, step, cutoff_threshold,
                      cache, base, tabs):
    """tanh(TM) with validated recurrence coefficients + P_{k+1} Lagrange
    tail. Saturation-safe (module section comment). [B, T, 2] iv coeffs."""
    return _sigmoidal_series_valid(coeffs, rem, k, tables, step,
                                   cutoff_threshold, cache, base, tabs, "tanh")


def sigmoid_series_valid(coeffs, rem, k, tables, step, cutoff_threshold,
                         cache, base, tabs):
    """sigmoid(TM); see tanh_series_valid."""
    return _sigmoidal_series_valid(coeffs, rem, k, tables, step,
                                   cutoff_threshold, cache, base, tabs,
                                   "sigmoid")


def _sigmoidal_series_valid_g(coeffs, rem, k, cache, base, tabs, make_ctx, kind):
    """Generic tanh/sigmoid series body (poly-CROWN S2-1).

    Transcription of `_sigmoidal_series_valid` (above) onto the ctx protocol,
    exactly as `_trig_series_valid_g` transcribes the trig body: the pow
    chain and res accumulation route through the context (the res support is
    the growing union of pow supports — the sin/cos pattern); the interval
    recurrence coefficients (`_sigmoidal_series_coeffs`), the constant-slot
    updates (Support id-0 invariant), and the derivative-polynomial Lagrange
    tail are layout-free and stay inline. Cache layout identical: c, per
    iteration (mul triple, a_i), f_range — 4k + 2 slots.
    """
    c, tm_f = _split_const(coeffs)
    cache[:, base] = c
    cursor = base + 1
    a = _sigmoidal_series_coeffs(c, k, kind, tabs)
    ctx = make_ctx(tm_f, rem)
    res_c = ctx.fresh_zero()
    res_c[:, 0] = a[0]
    res_r = torch.zeros_like(rem)
    pow_c = ctx.fresh_one()
    pow_r = torch.zeros_like(rem)
    f_range = ctx.range_f()
    for i in range(1, k + 1):
        pow_c, pow_r = ctx.mul_valid(pow_c, pow_r, cache, cursor)
        cursor += 3
        cache[:, cursor] = a[i]
        cursor += 1
        res_c = ctx.acc_add(res_c, iv.mul(pow_c, a[i].unsqueeze(1)))
        res_r = iv.add(res_r, iv.mul(pow_r, a[i]))
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, rem)
    y_rng = (tanh_iv if kind == "tanh" else sigmoid_iv)(iv.add(c, tm_range))
    p_top = _derivative_polys(kind, k + 1)[k + 1]
    m_iv = _poly_int_iv_eval(p_top, y_rng)
    lag = iv.mul(iv.mul(m_iv, iv.pow_int(tm_range, k + 1)), tabs.frec_iv[k + 1])
    res_r = iv.add(res_r, lag)
    return res_c, res_r


def tanh_series_valid_g(coeffs, rem, k, cache, base, tabs, make_ctx):
    """Generic-ctx tanh series; see _sigmoidal_series_valid_g."""
    return _sigmoidal_series_valid_g(coeffs, rem, k, cache, base, tabs,
                                     make_ctx, "tanh")


def sigmoid_series_valid_g(coeffs, rem, k, cache, base, tabs, make_ctx):
    """Generic-ctx sigmoid series; see _sigmoidal_series_valid_g."""
    return _sigmoidal_series_valid_g(coeffs, rem, k, cache, base, tabs,
                                     make_ctx, "sigmoid")
