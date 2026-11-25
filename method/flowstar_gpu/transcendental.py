"""Tier-I rigorous interval transcendentals on float64 tensors (plan D2, M5).

Same representation as `interval`: trailing dim of size 2, `iv[..., 0] = lo`,
`iv[..., 1] = hi`, written `[..., 2]` in shape comments. Every function returns
a SOUND enclosure: the exact mathematical image of the input interval is
contained in the result.

Soundness model — why slack constants instead of nextafter-once
---------------------------------------------------------------
`interval` gets away with ONE outward nextafter step per op because IEEE-754
requires +,-,*,/,sqrt to be correctly rounded (RN lands on the float nearest
the exact result, so the exact result lies within one ulp). Transcendentals
have NO such guarantee: torch.exp/log/sin/cos may be several ulps off, and the
error differs between the CPU vector-math path and the CUDA math library. We
therefore step each endpoint outward by an empirically calibrated number of
nextafter steps (ULP_SLACK_*), where "calibrated" means: measured against
mpmath at 60 dps over >= 10^6 adversarial samples on BOTH devices, constant =
measured max + 2 safety steps, and the measurement is re-asserted by
tests/properties/test_transcendental_props.py so a torch upgrade that degrades
accuracy fails the suite loudly instead of silently losing soundness.

Flow* correspondence (Interval.cpp): exp ~2048, log ~2478, sqrt ~1784 are the
same monotone endpoint maps, but with MPFR's true directed rounding where we
use RN + slack. sin ~2058 / cos ~2268 use the identical quadrant decomposition
(floor of endpoint / (pi/2), full period => [-1, 1]); Flow*'s 16-case switch
on (k_lo mod 4, k_hi mod 4) is equivalent to the uniform rule implemented
here: min/max over the endpoint values plus -1/+1 for every crossed critical
point (see `_sin_cos`).
"""

from __future__ import annotations

import torch

from .interval import mag
from .rounding import next_down, next_up

# ---------------------------------------------------------------------------
# Calibrated outward-slack constants (nextafter steps per endpoint).
#
# Measured max required steps (mpmath 60 dps oracle, 10^6 samples per function:
# log-uniform magnitudes, denormal args/results, near-k*pi/2 points; torch
# 2.13.0+cu126, 2026-07-26):        cpu   cuda (Tesla V100)
#     torch.exp                      1     1
#     torch.log                      1     1
#     torch.sin                      1     2
#     torch.cos                      1     2
# Constant = measured max over devices + 2 safety steps. The calibration test
# re-measures and FAILS if measured + 2 ever exceeds these values.
# ---------------------------------------------------------------------------
ULP_SLACK_EXP = 3
ULP_SLACK_LOG = 3
ULP_SLACK_TRIG = 4
# tanh (poly-CROWN P3: TM-through-NN activation series): measured max 1 ulp
# on cpu AND cuda (torch.tanh vs mpmath 40 dps, 20k log-uniform samples over
# |x| in [1e-4, 1e3], 2026-08-05; saturation is safe by construction — one
# up-step from the -1.0 flush jumps 2^-53 relative, far past the true
# 2·e^{-2|x|} gap) + 1 for RN-vs-directed + 2 safety steps. sigmoid has NO
# slack constant: torch.sigmoid flushes to exact 0.0 below x ~ -710 while the
# true value is a denormal (measured >= 100000 ulps off), so sigmoid_iv is
# computed through exp_iv + exact interval arithmetic instead.
ULP_SLACK_SIGMOIDAL = 4

# ---------------------------------------------------------------------------
# Directed float64 brackets of pi (Flow* include.h:118-119 keeps 50-digit
# decimal brackets str_pi_lo/str_pi_up for the same purpose). Hardcoded hex —
# no mpmath at import time. Derivation (mpmath, mp.dps = 60):
#     mp.pi                = 3.14159265358979323846264338327950288419716939937510582...
#     float(mp.pi)         = 0x1.921fb54442d18p+1  (RN; verified < pi)
#     nextafter(., +inf)   = 0x1.921fb54442d19p+1  (verified > pi)
# Halving is exact in binary FP, so _HALF_PI_LO < pi/2 < _HALF_PI_HI strictly
# (pi is irrational, so no float equals pi/2 and the bracket never collapses).
# ---------------------------------------------------------------------------
PI_LO = float.fromhex("0x1.921fb54442d18p+1")  # largest float64 <= pi
PI_HI = float.fromhex("0x1.921fb54442d19p+1")  # smallest float64 >= pi
_HALF_PI_LO = PI_LO / 2.0  # exact: largest float64 <= pi/2
_HALF_PI_HI = PI_HI / 2.0  # exact: smallest float64 >= pi/2

# Precision cliff for sin/cos quadrant analysis. At |x| = 1e12 the quadrant
# index is ~6.4e11: the directed-pi bracket ambiguity is ~1e-4 quadrants and
# ulp(x) is ~1e-4 rad, so quadrant analysis is still meaningful with orders of
# magnitude to spare. By |x| ~ 1e15 the bracket ambiguity approaches a whole
# quadrant (and beyond ~1e16, ulp(x) exceeds pi/2 itself), making the indices
# meaningless. Past the threshold we return the trivially sound envelope
# [-1, 1] instead. 1e12 (not 1e15) keeps a 3-decade safety margin and keeps
# the int64 quadrant arithmetic far away from any representability edge.
TRIG_ENVELOPE = 1e12


def _steps_down(x: torch.Tensor, n: int) -> torch.Tensor:
    """n nextafter steps toward -inf, elementwise. Shape preserved.

    WHY a loop: n is a tiny compile-time constant (<= ULP_SLACK_*), so a chain
    of nextafter kernels is cheaper and simpler than ordinal-space integer
    arithmetic; -inf/+inf fixed points of nextafter keep saturation sound.
    """
    for _ in range(n):
        x = next_down(x)
    return x


def _steps_up(x: torch.Tensor, n: int) -> torch.Tensor:
    """n nextafter steps toward +inf, elementwise. Shape preserved."""
    for _ in range(n):
        x = next_up(x)
    return x


def exp_iv(a: torch.Tensor) -> torch.Tensor:
    """Interval exp: [..., 2] -> [..., 2].

    WHY: exp is strictly increasing, so the exact image of [lo, hi] is
    [exp(lo), exp(hi)]; each endpoint is RN-computed and stepped outward by
    ULP_SLACK_EXP (soundness model in the module docstring). Mirrors Flow*
    Interval::exp (Interval.cpp:2048, MPFR RNDD/RNDU).

    Overflow (lo endpoint): if RN(exp(lo)) = +inf then exp(lo) > MAX_F64, and
    the first down-step maps +inf to MAX_F64 — a finite, sound lower bound
    (any slack >= 1 guarantees this; the calibration metric counts this case).
    The hi endpoint may saturate at +inf, which is sound for an upper bound.
    Underflow (lo endpoint): stepping down can cross 0 into negative values;
    exp > 0 always, so clamping the lower endpoint at 0 is sound and tighter.
    """
    lo_r = torch.clamp(_steps_down(torch.exp(a[..., 0]), ULP_SLACK_EXP), min=0.0)
    hi_r = _steps_up(torch.exp(a[..., 1]), ULP_SLACK_EXP)
    return torch.stack((lo_r, hi_r), dim=-1)


def log_iv(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval log: [..., 2] -> (result [..., 2], bad_mask [...] bool).

    WHY: log is strictly increasing on (0, inf), so the exact image is
    [log(lo), log(hi)] with ULP_SLACK_LOG outward steps. Lanes with lo <= 0
    (domain violation) get bad_mask True and the sound-but-useless sentinel
    [-inf, +inf], mirroring the rec/div convention (interval.py) — callers
    MUST fail those lanes; we never replicate Flow*'s exit(1)
    (Interval.cpp:2478). The log input is substituted with 1.0 on bad lanes so
    torch.log never sees a non-positive value (its NaN/-inf would be
    discarded by the where anyway, but NaN must never be computed on a lane
    that could be reduced later). No overflow concerns: |log x| <= 745 for
    every positive finite float64.
    """
    al, ah = a[..., 0], a[..., 1]
    bad = al <= 0  # [...] bool: domain violation (hi >= lo, so lo decides)
    one = torch.ones_like(al)
    lo_r = _steps_down(torch.log(torch.where(bad, one, al)), ULP_SLACK_LOG)
    hi_r = _steps_up(torch.log(torch.where(bad, one, ah)), ULP_SLACK_LOG)
    out = torch.stack((lo_r, hi_r), dim=-1)
    out = torch.where(
        bad.unsqueeze(-1),
        torch.stack((torch.full_like(al, -torch.inf), torch.full_like(ah, torch.inf)), dim=-1),
        out,
    )
    return out, bad


def sqrt_iv(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval sqrt: [..., 2] -> (result [..., 2], bad_mask [...] bool).

    WHY: sqrt is strictly increasing on [0, inf); exact image [sqrt(lo),
    sqrt(hi)]. Lanes with lo < 0 are domain violations: bad_mask True and
    [-inf, +inf] (rec/div convention; Flow* exits instead, Interval.cpp:1784).
    Note lo = 0 is FINE (sqrt(0) = 0), unlike log — hence < 0, not <= 0.

    IEEE-754 requires sqrt to be correctly rounded, so ONE outward nextafter
    step per endpoint is already rigorous (identical to the +,-,*,/ treatment
    in interval.py) — no empirical slack constant needed. We still take that
    step even though a correctly-rounded sqrt of a representable square would
    not need it, both for the rigorous RN->directed conversion and as defense
    against a nonconforming backend vector-sqrt path. The lower endpoint is
    clamped at 0 (image of sqrt is >= 0; stepping down from sqrt(0) = 0 would
    otherwise yield -5e-324). Bad lanes substitute 0 into torch.sqrt so no
    NaN is ever computed (same rationale as log_iv).
    """
    al, ah = a[..., 0], a[..., 1]
    bad = al < 0  # [...] bool: any negative point in the interval
    zero = torch.zeros_like(al)
    lo_r = torch.clamp(next_down(torch.sqrt(torch.where(bad, zero, al))), min=0.0)
    hi_r = next_up(torch.sqrt(torch.where(bad, zero, ah)))
    out = torch.stack((lo_r, hi_r), dim=-1)
    out = torch.where(
        bad.unsqueeze(-1),
        torch.stack((torch.full_like(al, -torch.inf), torch.full_like(ah, torch.inf)), dim=-1),
        out,
    )
    return out, bad


def _sin_cos(a: torch.Tensor, fn, r_max: int, r_min: int) -> torch.Tensor:
    """Shared sin/cos quadrant machinery: [..., 2] -> [..., 2].

    `fn` is torch.sin or torch.cos; `r_max`/`r_min` are the residues mod 4 of
    the quadrant boundaries j (x = j*pi/2) where fn attains +1 / -1:
        sin: +1 at j = 1 (mod 4)   [x = pi/2 + 2*pi*m],  -1 at j = 3 (mod 4)
        cos: +1 at j = 0 (mod 4)   [x = 2*pi*m],         -1 at j = 2 (mod 4)

    Soundness argument
    ------------------
    sin/cos are piecewise monotone with critical points EXACTLY at the +/-1
    boundaries above, so the exact image of [lo, hi] is
        hull{ fn(lo), fn(hi) }  U  { +/-1 for each critical point crossed }.
    We compute a superset:
      1. Endpoint values via RN fn + ULP_SLACK_TRIG outward steps (calibrated
         enclosure of the true fn at each endpoint double).
      2. Conservative quadrant indices. The true index of endpoint x is
         floor(x / (pi/2)); pi/2 is not representable, so we divide by BOTH
         directed brackets _HALF_PI_LO/_HALF_PI_HI (exact halves of the pi
         brackets) — the true quotient lies between the two exact quotients,
         and IEEE division is correctly rounded, so one nextafter step outward
         from the min/max RN quotient brackets it rigorously:
             x/(pi/2) in [next_down(min(q_hi, q_lo)), next_up(max(q_hi, q_lo))]
         Taking floor of the LOW bracket for k_lo and of the HIGH bracket for
         k_hi treats every ambiguity (bracket disagreement OR division
         rounding) as widening: k_lo <= true k(lo), k_hi >= true k(hi), so the
         crossed-boundary set {k_lo+1, ..., k_hi} is a SUPERSET of the truly
         crossed boundaries and no extremum can be missed.
      3. A critical point with residue r is crossed iff some j = r (mod 4)
         lies in (k_lo, k_hi]; the count of such j is
         floor((k_hi - r)/4) - floor((k_lo - r)/4).
    If k_hi - k_lo >= 4 a full period is (possibly) covered: all four residues
    occur in (k_lo, k_hi], both extrema fire, and the result is [-1, 1] — the
    same early-out Flow* takes for iPeriod >= 4, falling out of the uniform
    rule with no special case. Endpoint slack may push past +/-1; the exact
    image never exceeds [-1, 1], so clamping is sound and tighter.

    Precision cliff: lanes with mag(a) > TRIG_ENVELOPE skip the analysis and
    return [-1, 1] (see the constant's comment). Their endpoints are
    substituted with 0 BEFORE the division so the int64 quadrant cast below
    never sees a non-representable quotient (e.g. 1e300 / (pi/2)) and
    torch.sin/cos of an infinite endpoint (= NaN) is never computed — infinite
    inputs ride the envelope soundly.
    """
    al, ah = a[..., 0], a[..., 1]
    env = mag(a) > TRIG_ENVELOPE  # [...] bool: precision-cliff lanes
    zero = torch.zeros_like(al)
    al_s = torch.where(env, zero, al)  # [...] safe endpoints
    ah_s = torch.where(env, zero, ah)

    # Conservative quadrant indices (soundness argument, step 2). Quotient
    # magnitude <= 1e12/(pi/2) ~ 6.4e11: floor is exact in f64 and the int64
    # cast is exact.
    q_lo = next_down(torch.minimum(al_s / _HALF_PI_HI, al_s / _HALF_PI_LO))  # [...]
    q_hi = next_up(torch.maximum(ah_s / _HALF_PI_HI, ah_s / _HALF_PI_LO))  # [...]
    k_lo = torch.floor(q_lo).to(torch.int64)  # [...] <= true quadrant of lo
    k_hi = torch.floor(q_hi).to(torch.int64)  # [...] >= true quadrant of hi

    # Endpoint enclosures (step 1); min/max commute with the monotone steps.
    v_lo = fn(al_s)  # [...] RN
    v_hi = fn(ah_s)  # [...] RN
    lo_r = torch.clamp(_steps_down(torch.minimum(v_lo, v_hi), ULP_SLACK_TRIG), min=-1.0)
    hi_r = torch.clamp(_steps_up(torch.maximum(v_lo, v_hi), ULP_SLACK_TRIG), max=1.0)

    # Crossed-extremum masks (step 3).
    def crossed(r: int) -> torch.Tensor:
        hi_c = torch.div(k_hi - r, 4, rounding_mode="floor")
        lo_c = torch.div(k_lo - r, 4, rounding_mode="floor")
        return hi_c > lo_c  # [...] bool

    lo_r = torch.where(crossed(r_min), torch.full_like(lo_r, -1.0), lo_r)
    hi_r = torch.where(crossed(r_max), torch.full_like(hi_r, 1.0), hi_r)

    out = torch.stack((lo_r, hi_r), dim=-1)
    envelope = torch.stack((torch.full_like(al, -1.0), torch.full_like(ah, 1.0)), dim=-1)
    return torch.where(env.unsqueeze(-1), envelope, out)


def sin_iv(a: torch.Tensor) -> torch.Tensor:
    """Interval sin: [..., 2] -> [..., 2], always within [-1, 1].

    WHY: quadrant analysis mirroring Flow* Interval::sin (Interval.cpp:2058);
    algorithm and soundness argument in `_sin_cos`. sin attains +1 at
    x = pi/2 + 2*pi*m (boundary index j = 1 mod 4) and -1 at x = 3*pi/2 +
    2*pi*m (j = 3 mod 4).
    """
    return _sin_cos(a, torch.sin, r_max=1, r_min=3)


def cos_iv(a: torch.Tensor) -> torch.Tensor:
    """Interval cos: [..., 2] -> [..., 2], always within [-1, 1].

    WHY: quadrant analysis mirroring Flow* Interval::cos (Interval.cpp:2268);
    algorithm and soundness argument in `_sin_cos`. cos attains +1 at
    x = 2*pi*m (boundary index j = 0 mod 4) and -1 at x = pi + 2*pi*m
    (j = 2 mod 4).
    """
    return _sin_cos(a, torch.cos, r_max=0, r_min=2)


def tanh_iv(a: torch.Tensor) -> torch.Tensor:
    """Interval tanh: [..., 2] -> [..., 2], always within [-1, 1].

    WHY (poly-CROWN P3): the TM-through-NN activation series needs enclosures
    of tanh at the pre-activation midpoint and range. tanh is strictly
    increasing, so the exact image of [lo, hi] is [tanh(lo), tanh(hi)]; RN
    endpoints stepped outward by ULP_SLACK_SIGMOIDAL, then clamped into the
    codomain [-1, 1] (sound and tighter, the exp>0 clamp's analogue — and the
    saturation regime |x| >~ 20 lands exactly on ±1.0 this way, which is what
    keeps deep-saturated controller neurons finite where an exp-based
    composite overflows).
    """
    lo = torch.clamp(_steps_down(torch.tanh(a[..., 0]), ULP_SLACK_SIGMOIDAL),
                     min=-1.0)
    hi = torch.clamp(_steps_up(torch.tanh(a[..., 1]), ULP_SLACK_SIGMOIDAL),
                     max=1.0)
    return torch.stack((lo, hi), dim=-1)


def sigmoid_iv(a: torch.Tensor) -> torch.Tensor:
    """Interval logistic sigmoid: [..., 2] -> [..., 2], always within [0, 1].

    NOT a stepped torch.sigmoid: that flushes to exact 0.0 below x ~ -710
    while the true value is a positive denormal (measured >= 1e5 ulps of
    unsoundness for the upper endpoint), and no fixed ulp slack recovers a
    flush-to-zero. Instead the composite 1/(1 + exp(-x)) is evaluated in the
    already-calibrated interval machinery: exp_iv's overflow saturation makes
    the denominator's +inf a SOUND upper bound whose reciprocal is a sound
    0.0 lower endpoint, and every other step is an exact-directed interval
    op. Codomain-clamped into [0, 1] (rec's outward step can cross 1 for
    arguments near +-0).
    """
    from . import interval as iv

    e = exp_iv(torch.stack((-a[..., 1], -a[..., 0]), dim=-1))  # exp(-x)
    denom = iv.add(e, torch.ones_like(e))  # [1 + exp(-x)] >= 1: rec never bad
    r, _bad = iv.rec(denom)
    return torch.stack(
        (torch.clamp(r[..., 0], min=0.0), torch.clamp(r[..., 1], max=1.0)),
        dim=-1,
    )
