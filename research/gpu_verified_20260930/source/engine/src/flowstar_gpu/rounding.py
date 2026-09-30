"""Directed-rounding emulation on round-to-nearest (RN) float64 tensor hardware.

Flow* gets soundness from MPFR's per-operation directed rounding (RNDD for lower
bounds, RNDU for upper bounds). PyTorch exposes no rounding-mode control, so this
module provides the two mechanisms that replace it (plan D2, ALGORITHM.md):

1. **nextafter outward stepping** for single elementwise operations. IEEE-754 RN
   returns the float CLOSEST to the exact result, so the exact result always lies
   in [pred(rn), succ(rn)] — one `torch.nextafter` step outward yields a sound
   directed bound at most 1 ulp looser than true directed rounding. This is exact
   in semantics (not an estimate) and holds through underflow because fp64
   denormals are not flushed on our V100s (verified, GOTCHAS E3).

2. **Rump-style a-priori error bounds** for REDUCTIONS (sums, dots, matmuls).
   Stepping nextafter between every partial addition would force a serial
   evaluation order; instead we let the hardware reduce in ANY order (FMA, tree,
   atomics — all fine) and add an a-priori bound on the worst-case accumulated
   rounding error. See `sum_error_bound` for the derivation.

Everything here is dtype-f64-only by design: soundness constants are wired to
IEEE-754 binary64 and the code asserts against silent dtype drift.

Documented-and-deferred alternative: error-free transformations (twoSum /
twoProd compensated arithmetic) could reproduce MPFR's directed rounding
bit-exactly for +, -, * at ~3-5x the elementwise cost. Rejected because
bit-parity with Flow* is not a goal — the parity gates are functional
(range/width), see ALGORITHM.md.
"""

from __future__ import annotations

import math

import torch

# IEEE-754 binary64 constants.
# U: unit roundoff (half ulp of 1.0): RN(x op y) = (x op y)(1 + d) with |d| <= U.
U = 2.0**-53
# ETA: smallest positive subnormal. Product roundoff can be BELOW the relative
# model when the result underflows; each product contributes at most ETA/2 of
# absolute underflow error (Higham, Accuracy and Stability, sec 2.2 extended
# model: fl(x op y) = (x op y)(1+d) + e with |e| <= ETA/2, d*e = 0).
ETA = 5e-324


def _assert_f64(x: torch.Tensor) -> None:
    """Soundness guard: every rigorous op in this project is float64-only."""
    if x.dtype is not torch.float64:
        raise TypeError(f"rigorous arithmetic requires float64, got {x.dtype}")


def next_up(x: torch.Tensor) -> torch.Tensor:
    """Smallest float64 strictly greater than x, elementwise. Shape preserved.

    Used to turn an RN result into a sound UPPER bound: for any single RN
    operation, exact <= next_up(rn_result) always holds (RN picks the nearest
    float, so the exact value cannot exceed the next float up).
    +inf maps to +inf (nextafter(inf, inf) = inf), which stays sound.
    """
    _assert_f64(x)
    return torch.nextafter(x, torch.full_like(x, math.inf))


def next_down(x: torch.Tensor) -> torch.Tensor:
    """Largest float64 strictly smaller than x, elementwise. Shape preserved.

    Sound LOWER bound counterpart of `next_up`.
    """
    _assert_f64(x)
    return torch.nextafter(x, torch.full_like(x, -math.inf))


def sum_error_bound(abs_sum_rn: torch.Tensor, m: int) -> torch.Tensor:
    """A-priori bound on |RN-reduction - exact| for a sum of m float64 terms.

    No production caller: `dot_error_bound` supersedes it in-pipeline (its
    extra product-error and underflow slack covers pure sums too); kept as the
    pedagogical base case the dot bound's derivation builds on.

    Input: abs_sum_rn [..., any shape] — the RN-computed sum of |terms| over the
    reduced dimension(s); m — the reduction length. Output: same shape, an upper
    bound on the absolute rounding error of ANY evaluation order of the sum of
    the (signed) terms, including FMA-fused and nondeterministically reordered
    reductions.

    Derivation (standard Wilkinson/Higham forward error analysis):
      * Any summation order of m terms performs m-1 RN additions; the classic
        bound is |err| <= (m-1) * U * sum|x_i| (additions cannot underflow into
        the extended-model e-term: the error of an fp addition is itself a
        representable float, so no ETA term is needed for pure sums).
      * We only have the RN-computed S_hat of sum|x_i|, which may UNDERestimate
        the true sum|x_i| by at most the same mechanism, i.e.
        sum|x_i| <= S_hat * (1 + (m-1)U) / (1 - ...) — we absorb this with the
        crude but safe factor (1 + 2mU), valid whenever m*U < 0.25 (i.e.
        m < 2^51, astronomically beyond any tensor length here; asserted).
      * Callers reducing over PRODUCTS (dots/matmuls) must add the per-product
        underflow slack themselves: + m * ETA (see `dot_error_bound`).

    The final multiply/add chain computing the bound is itself RN, so we take
    one `next_up` at the end to make the bound an upper bound of itself.
    """
    _assert_f64(abs_sum_rn)
    if m < 1:
        raise ValueError(f"reduction length must be >= 1, got {m}")
    if not m * U < 0.25:
        raise ValueError(f"reduction too long for the error model: m={m}")
    if m == 1:
        # A single term is copied, not added: no rounding error at all.
        return torch.zeros_like(abs_sum_rn)

    factor = (m - 1) * U * (1.0 + 2.0 * m * U)
    return next_up(abs_sum_rn * factor)


def dot_error_bound(abs_dot_rn: torch.Tensor, m: int | torch.Tensor) -> torch.Tensor:
    """A-priori bound on |RN dot/matmul - exact| for inner dimension m.

    Input: abs_dot_rn [...] — RN-computed |a| . |b| (elementwise-abs dot) over
    the inner dimension; m — inner dimension length, either a Python int or an
    integer tensor broadcastable to abs_dot_rn's shape (per-slot segment
    lengths in the polynomial product use the tensor form). Output: same shape.

    A dot product is m RN multiplications (each with relative error U and
    absolute underflow slack ETA/2) followed by an m-term summation, evaluated
    in unknown order and possibly FMA-fused (FMA only ever REDUCES the error,
    since it skips one rounding). Bound:

        |err| <= m*U*(|a|.|b|) [products] + (m-1)*U*(|a|.|b|)(1+..) [summation]
                 + m*ETA/2 [underflow]
              <= (2m-1)*U*(1+2mU) * abs_dot_rn_inflated + m*ETA

    We fold both relative parts into one factor on the RN-computed abs-dot
    (inflated by the same (1+2mU) sloppiness as in `sum_error_bound`) and add
    the underflow slack outward. m = 0 (empty segment) yields exactly 0.
    """
    _assert_f64(abs_dot_rn)
    if isinstance(m, int):
        if m < 0:
            raise ValueError(f"reduction length must be >= 0, got {m}")
        if not m * U < 0.25:
            raise ValueError(f"reduction too long for the error model: m={m}")
        if m == 0:
            return torch.zeros_like(abs_dot_rn)
        factor = (2.0 * m - 1.0) * U * (1.0 + 2.0 * m * U)
        return next_up(next_up(abs_dot_rn * factor) + m * ETA)

    return apply_dot_error_bound(abs_dot_rn, prepare_dot_error_bound(m))


def prepare_dot_error_bound(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate static reduction counts once and cache their f64 error factors.

    The returned tensors belong to the caller's immutable plan. Dynamic or
    user-supplied counts still pass through dot_error_bound on every call.
    """
    if (m < 0).any():
        raise ValueError("reduction lengths must be >= 0")
    if not (m.amax().item() * U < 0.25):
        raise ValueError(f"reduction too long for the error model: m={m.amax().item()}")
    m_f = m.to(torch.float64)
    factor = (2.0 * m_f - 1.0).clamp(min=0.0) * U * (1.0 + 2.0 * m_f * U)
    return factor, m_f * ETA, m_f > 0


def apply_dot_error_bound(abs_dot_rn: torch.Tensor, parameters: tuple) -> torch.Tensor:
    """Apply a validated static reduction plan without reading GPU scalars."""
    _assert_f64(abs_dot_rn)
    factor, underflow, nonempty = parameters
    bound = next_up(next_up(abs_dot_rn * factor) + underflow)
    return torch.where(nonempty, bound, torch.zeros_like(bound))
