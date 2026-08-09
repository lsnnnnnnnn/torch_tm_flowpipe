"""Sound PyTorch-native algorithm-aligned TORA-Q3 plant lane."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .batched_dense_tm import (
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRemainderLedger,
    _down,
    _interval_add,
    _interval_div_positive_integer,
    _interval_mul,
    _point_sin_cos_enclosure,
    _sound_add_tm,
    _sound_mul_tm,
    _sound_scale_tm_interval,
    _subset_margin,
    _up,
    dense_polynomial_picard,
    require_dense_condition,
)
from .tora_q3 import (
    ToraQ3Step,
    _endpoint_bounds,
    _zero_exact_held_remainder,
)


def _interval_cosine(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    maximum_abs_argument: float = 8.0,
    series_terms: int = 32,
    point_enclosure_backend: str = "eager",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outward cosine enclosure for bounded binary64 intervals."""
    if lower.dtype != torch.float64 or upper.dtype != torch.float64:
        raise TypeError("formal interval cosine requires float64")
    if lower.shape != upper.shape:
        raise ValueError("interval cosine shape mismatch")
    require_dense_condition(lower <= upper, "invalid cosine interval")
    require_dense_condition(
        torch.maximum(torch.abs(lower), torch.abs(upper))
        <= abs(float(maximum_abs_argument)),
        "interval cosine argument exceeds the proved domain",
    )
    _sin_lo, _sin_hi, lower_cos_lo, lower_cos_hi = (
        _point_sin_cos_enclosure(
            lower,
            series_terms=series_terms,
            maximum_abs_center=maximum_abs_argument,
            backend=point_enclosure_backend,
        )
    )
    _sin_lo, _sin_hi, upper_cos_lo, upper_cos_hi = (
        _point_sin_cos_enclosure(
            upper,
            series_terms=series_terms,
            maximum_abs_center=maximum_abs_argument,
            backend=point_enclosure_backend,
        )
    )
    result_lo = torch.minimum(lower_cos_lo, upper_cos_lo)
    result_hi = torch.maximum(lower_cos_hi, upper_cos_hi)
    pi_lo = math.nextafter(math.pi, -math.inf)
    pi_hi = math.nextafter(math.pi, math.inf)
    critical_limit = math.ceil(abs(float(maximum_abs_argument)) / pi_lo) + 1
    for multiple in range(-critical_limit, critical_limit + 1):
        if multiple >= 0:
            critical_lo = float(multiple) * pi_lo
            critical_hi = float(multiple) * pi_hi
        else:
            critical_lo = float(multiple) * pi_hi
            critical_hi = float(multiple) * pi_lo
        possible = (lower <= critical_hi) & (upper >= critical_lo)
        if multiple % 2:
            result_lo = torch.where(
                possible,
                torch.nextafter(
                    -torch.ones_like(result_lo),
                    torch.full_like(result_lo, -torch.inf),
                ),
                result_lo,
            )
        else:
            result_hi = torch.where(
                possible,
                torch.nextafter(
                    torch.ones_like(result_hi),
                    torch.full_like(result_hi, torch.inf),
                ),
                result_hi,
            )
    return result_lo, result_hi


def aligned_sin_tm(
    model: BatchedTaylorModel,
    *,
    maximum_delta_radius: float = 4.0,
    maximum_abs_argument: float = 8.0,
    series_terms: int = 32,
    point_enclosure_backend: str = "eager",
) -> BatchedTaylorModel:
    """Centered quadratic sine with separated, outward remainder routing.

    For ``x = c + p + r``, the retained polynomial is the Q3 projection of
    ``sin(c) + cos(c)p - sin(c)p²/2``.  The interval remainder independently
    encloses point-coefficient uncertainty, Q3 square overflow, input-remainder
    propagation, and the third-derivative Taylor residual over the complete
    line segment from ``c`` to ``x``.
    """
    if model.poly.coeffs.dtype != torch.float64:
        raise TypeError("formal aligned sine requires float64")
    if not model.is_finite():
        raise ValueError("aligned sine input must be finite")
    require_dense_condition(
        model.rem_lo <= model.rem_hi,
        "aligned sine input remainder is invalid",
    )
    basis = model.poly.basis
    constant = model.poly.coeffs[..., basis.constant_index]
    sin_lo, sin_hi, cos_lo, cos_hi = _point_sin_cos_enclosure(
        constant,
        series_terms=series_terms,
        maximum_abs_center=maximum_abs_argument,
        backend=point_enclosure_backend,
    )
    deviation_coefficients = model.poly.coeffs.clone()
    deviation_coefficients[..., basis.constant_index] = 0.0
    zeros = torch.zeros_like(model.rem_lo)
    deviation = BatchedTaylorModel(
        BatchedPolynomial(deviation_coefficients, basis),
        zeros,
        zeros.clone(),
        model.domain_lo,
        model.domain_hi,
        DenseRemainderLedger.empty(),
        model.range_policy,
        model.range_trace,
    )
    polynomial_lo, polynomial_hi = deviation.range_bound(
        context="aligned_sine_polynomial_deviation"
    )
    delta_lo, delta_hi = _interval_add(
        polynomial_lo,
        polynomial_hi,
        model.rem_lo,
        model.rem_hi,
    )
    delta_radius = torch.maximum(torch.abs(delta_lo), torch.abs(delta_hi))
    require_dense_condition(
        delta_radius <= abs(float(maximum_delta_radius)),
        "aligned sine composition radius exceeds maximum_delta_radius",
    )

    sine_mid = sin_lo + 0.5 * (sin_hi - sin_lo)
    constant_polynomial = BatchedPolynomial.constants(sine_mid, basis)
    stored_sine = constant_polynomial.coeffs[..., basis.constant_index]
    constant_error_lo = _down(sin_lo - stored_sine)
    constant_error_hi = _up(sin_hi - stored_sine)
    approximation = BatchedTaylorModel(
        constant_polynomial,
        constant_error_lo,
        constant_error_hi,
        model.domain_lo,
        model.domain_hi,
        DenseRemainderLedger.empty().add(
            "composition_overflow", constant_error_lo, constant_error_hi
        ),
        model.range_policy,
        model.range_trace,
    )
    approximation = _sound_add_tm(
        approximation,
        _sound_scale_tm_interval(deviation, cos_lo, cos_hi),
    )
    square = _sound_mul_tm(deviation, deviation)
    second_coefficient_lo = _down(-0.5 * sin_hi)
    second_coefficient_hi = _up(-0.5 * sin_lo)
    approximation = _sound_add_tm(
        approximation,
        _sound_scale_tm_interval(
            square, second_coefficient_lo, second_coefficient_hi
        ),
    )

    propagated_lo, propagated_hi = _interval_mul(
        cos_lo, cos_hi, model.rem_lo, model.rem_hi
    )
    twice_poly_rem_lo, twice_poly_rem_hi = _interval_mul(
        polynomial_lo,
        polynomial_hi,
        model.rem_lo,
        model.rem_hi,
    )
    twice_poly_rem_lo = _down(2.0 * twice_poly_rem_lo)
    twice_poly_rem_hi = _up(2.0 * twice_poly_rem_hi)
    remainder_square_lo, remainder_square_hi = _interval_mul(
        model.rem_lo, model.rem_hi, model.rem_lo, model.rem_hi
    )
    square_correction_lo, square_correction_hi = _interval_add(
        twice_poly_rem_lo,
        twice_poly_rem_hi,
        remainder_square_lo,
        remainder_square_hi,
    )
    second_lo, second_hi = _interval_mul(
        second_coefficient_lo,
        second_coefficient_hi,
        square_correction_lo,
        square_correction_hi,
    )

    segment_delta_lo = torch.minimum(torch.zeros_like(delta_lo), delta_lo)
    segment_delta_hi = torch.maximum(torch.zeros_like(delta_hi), delta_hi)
    argument_lo, argument_hi = _interval_add(
        constant,
        constant,
        segment_delta_lo,
        segment_delta_hi,
    )
    cosine_lo, cosine_hi = _interval_cosine(
        argument_lo,
        argument_hi,
        maximum_abs_argument=maximum_abs_argument,
        series_terms=series_terms,
        point_enclosure_backend=point_enclosure_backend,
    )
    third_derivative_lo = -cosine_hi
    third_derivative_hi = -cosine_lo
    delta_square_lo, delta_square_hi = _interval_mul(
        delta_lo, delta_hi, delta_lo, delta_hi
    )
    delta_cube_lo, delta_cube_hi = _interval_mul(
        delta_square_lo, delta_square_hi, delta_lo, delta_hi
    )
    third_lo, third_hi = _interval_mul(
        third_derivative_lo,
        third_derivative_hi,
        delta_cube_lo,
        delta_cube_hi,
    )
    third_lo, third_hi = _interval_div_positive_integer(
        third_lo, third_hi, 6
    )
    correction_lo, correction_hi = _interval_add(
        propagated_lo,
        propagated_hi,
        second_lo,
        second_hi,
    )
    correction_lo, correction_hi = _interval_add(
        correction_lo, correction_hi, third_lo, third_hi
    )
    result_lo, result_hi = _interval_add(
        approximation.rem_lo,
        approximation.rem_hi,
        correction_lo,
        correction_hi,
    )
    return BatchedTaylorModel(
        approximation.poly,
        result_lo,
        result_hi,
        model.domain_lo,
        model.domain_hi,
        approximation.ledger.add(
            "composition_overflow", correction_lo, correction_hi
        ),
        model.range_policy,
        model.range_trace,
    )


def algorithm_aligned_tora_rhs(
    state: BatchedTaylorModel,
    *,
    point_enclosure_backend: str = "eager",
) -> BatchedTaylorModel:
    if state.poly.out_dim != 5:
        raise ValueError("TORA RHS requires [x1,x2,x3,x4,u1]")
    x1 = state.component(0)
    x2 = state.component(1)
    x3 = state.component(2)
    x4 = state.component(3)
    control = state.component(4)
    sine = aligned_sin_tm(
        x3, point_enclosure_backend=point_enclosure_backend
    )
    held = BatchedTaylorModel.constants_like(0.0, control)
    return BatchedTaylorModel.concat(
        (x2, -x1 + sine.scale(0.1), x4, control - 10.0, held)
    )


def algorithm_aligned_q3_step(
    base: BatchedTaylorModel,
    *,
    h: float = 0.1,
    polynomial_picard_rounds: int = 2,
    remainder_rounds: int = 10,
    seed: float = 0.01,
    capture_trace: bool = True,
    point_enclosure_backend: str = "eager",
) -> ToraQ3Step:
    """Complete-Q3 K2 plus aligned-sine ten-round remainder Picard."""
    if base.poly.basis.dim != 6 or base.poly.basis.order != 3:
        raise ValueError("algorithm-aligned TORA step requires complete Q3")
    if polynomial_picard_rounds != 2 or remainder_rounds != 10:
        raise ValueError("algorithm-aligned lane freezes K2 and ten rounds")
    require_dense_condition(
        torch.isclose(
            base.domain_lo[:, 0], torch.zeros_like(base.domain_lo[:, 0])
        ),
        "local time lower bound must be zero",
    )
    require_dense_condition(
        torch.isclose(
            base.domain_hi[:, 0],
            torch.full_like(base.domain_hi[:, 0], float(h)),
        ),
        "local time upper bound must equal h",
    )
    rhs = lambda value: algorithm_aligned_tora_rhs(
        value, point_enclosure_backend=point_enclosure_backend
    )
    candidate, polynomial_trace = dense_polynomial_picard(
        rhs,
        base.without_remainder(),
        tau_index=0,
        order=3,
        iterations=polynomial_picard_rounds,
        cutoff_threshold=None,
        capture_trace=capture_trace,
    )
    seed_vector = torch.tensor(
        [seed, seed, seed, seed, 0.0],
        dtype=base.poly.coeffs.dtype,
        device=base.poly.coeffs.device,
    ).view(1, 5).expand(base.poly.batch, -1)
    current = candidate.with_remainder(-seed_vector, seed_vector)
    base_polynomial = base.without_remainder()
    initial_image = _zero_exact_held_remainder(
        base_polynomial.add(rhs(current).integrate(0))
    )
    initial_margin = _subset_margin(
        -seed_vector,
        seed_vector,
        initial_image.rem_lo,
        initial_image.rem_hi,
    )
    initial_shrink = initial_margin >= 0.0
    difference = BatchedPolynomial(
        initial_image.poly.coeffs - current.poly.coeffs,
        current.poly.basis,
    )
    roundoff_lo, roundoff_hi = difference.range_bound(
        base.domain_lo,
        base.domain_hi,
        policy=base.range_policy,
        context="tora_algorithm_aligned_picard_roundoff",
        trace=base.range_trace,
    )
    roundoff_lo[:, 4] = 0.0
    roundoff_hi[:, 4] = 0.0
    rows: list[Mapping[str, Any]] = []
    all_rounds_shrink = torch.ones_like(initial_shrink)
    for round_index in range(1, remainder_rounds + 1):
        image = _zero_exact_held_remainder(
            base_polynomial.add(rhs(current).integrate(0))
        )
        candidate_lo, candidate_hi = _interval_add(
            image.rem_lo,
            image.rem_hi,
            roundoff_lo,
            roundoff_hi,
        )
        candidate_lo[:, 4] = 0.0
        candidate_hi[:, 4] = 0.0
        margin = _subset_margin(
            current.rem_lo,
            current.rem_hi,
            candidate_lo,
            candidate_hi,
        )
        shrink = margin >= 0.0
        accepted_lo = torch.where(shrink, candidate_lo, current.rem_lo)
        accepted_hi = torch.where(shrink, candidate_hi, current.rem_hi)
        all_rounds_shrink &= shrink
        if capture_trace:
            rows.append(
                {
                    "round": round_index,
                    "candidate_lower": candidate_lo.detach().cpu().tolist(),
                    "candidate_upper": candidate_hi.detach().cpu().tolist(),
                    "accepted_lower": accepted_lo.detach().cpu().tolist(),
                    "accepted_upper": accepted_hi.detach().cpu().tolist(),
                    "shrink_mask": shrink.detach().cpu().tolist(),
                    "subset_margin": margin.detach().cpu().tolist(),
                }
            )
        current = BatchedTaylorModel(
            image.poly,
            accepted_lo,
            accepted_hi,
            base.domain_lo,
            base.domain_hi,
            DenseRemainderLedger(
                {
                    "picard_residual": (
                        accepted_lo.clone(),
                        accepted_hi.clone(),
                    )
                }
            ),
            base.range_policy,
            base.range_trace,
        )
    tube_lower, tube_upper = current.range_bound(
        context="tora_algorithm_aligned_full_step_tube"
    )
    endpoint_lower, endpoint_upper = _endpoint_bounds(current, h=h)
    finite = (
        torch.isfinite(tube_lower).all(dim=1)
        & torch.isfinite(tube_upper).all(dim=1)
        & torch.isfinite(endpoint_lower).all(dim=1)
        & torch.isfinite(endpoint_upper).all(dim=1)
    )
    property_ok = (
        torch.maximum(torch.abs(tube_lower[:, :4]), torch.abs(tube_upper[:, :4]))
        <= 2.0
    ).all(dim=1)
    initial_subset_ok = initial_shrink.all(dim=1)
    all_remainder_rounds_ok = all_rounds_shrink.all(dim=1)
    accepted = (
        finite & property_ok & initial_subset_ok & all_remainder_rounds_ok
    )
    all_accepted = bool(torch.all(accepted))
    return ToraQ3Step(
        segment_tm=current,
        endpoint_tm=current.endpoint(0, float(h)) if all_accepted else None,
        tube_lower=tube_lower,
        tube_upper=tube_upper,
        endpoint_lower=endpoint_lower,
        endpoint_upper=endpoint_upper,
        finite_ok_by_leaf=finite,
        initial_subset_ok_by_leaf=initial_subset_ok,
        all_remainder_rounds_ok_by_leaf=all_remainder_rounds_ok,
        local_property_ok_by_leaf=property_ok,
        composed_property_ok_by_leaf=property_ok,
        accepted_by_leaf=accepted,
        initial_shrink_mask=initial_shrink,
        initial_margin=initial_margin,
        round_trace=tuple(rows),
        polynomial_trace=polynomial_trace,
        status="validated" if all_accepted else "failed",
        message=(
            ""
            if all_accepted
            else "algorithm-aligned local validation or property check failed"
        ),
    )


__all__ = [
    "algorithm_aligned_q3_step",
    "algorithm_aligned_tora_rhs",
    "aligned_sin_tm",
]
