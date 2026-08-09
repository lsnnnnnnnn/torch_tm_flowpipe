"""Fixed-shape pure-Tensor TORA-Q3 local-step kernels.

The functions below deliberately operate only on tensors and immutable Q3
route metadata.  Public Taylor-model objects are unpacked and reconstructed by
the wrapper outside the compiled boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import torch

from .batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRemainderLedger,
    _point_sin_cos_enclosure_kernel,
    require_dense_condition,
)
from .tora_q3 import (
    ToraQ3AffineBoundary,
    ToraQ3AffineCarry,
    ToraQ3Step,
    _endpoint_bounds,
    compose_tora_q3_tm,
)


TensorTuple = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ToraQ3KernelMetadata:
    exponents: torch.Tensor
    multiply_left: torch.Tensor
    multiply_right: torch.Tensor
    multiply_out: torch.Tensor
    integrate_kept_in: torch.Tensor
    integrate_kept_out: torch.Tensor
    integrate_kept_factor: torch.Tensor
    integrate_overflow_in: torch.Tensor
    integrate_overflow_exponents: torch.Tensor
    integrate_overflow_factor: torch.Tensor
    endpoint_out: torch.Tensor
    endpoint_exponents: torch.Tensor


_METADATA_CACHE: dict[tuple[str, torch.dtype], ToraQ3KernelMetadata] = {}
_COMPILED_CACHE: dict[tuple[Any, ...], Callable[..., TensorTuple]] = {}
_COMPILED_DISABLED: dict[tuple[Any, ...], str] = {}
_COMPILED_VERIFIED: dict[tuple[Any, ...], dict[str, Any]] = {}
_SEGMENTED_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]
] = {}
_SEGMENTED_DISABLED: dict[tuple[Any, ...], str] = {}
_SEGMENTED_VERIFIED: dict[tuple[Any, ...], dict[str, Any]] = {}


def tora_q3_kernel_metadata(
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> ToraQ3KernelMetadata:
    device_t = torch.device(device)
    key = (str(device_t), dtype)
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    basis = BatchedMonomialBasis.build(6, 3, str(device_t))
    (
        multiply_left,
        multiply_right,
        multiply_out,
        _dropped_left,
        _dropped_right,
        _dropped_merge,
        _dropped_exponents,
    ) = basis.multiplication_plan_for_degree(None)
    (
        integrate_kept_in,
        integrate_kept_out,
        integrate_kept_factor,
        integrate_overflow_in,
        integrate_overflow_exponents,
        integrate_overflow_factor,
    ) = basis.integration_plan(0, dtype=dtype)
    endpoint_basis = BatchedMonomialBasis.build(5, 3, str(device_t))
    endpoint_indices = []
    for exponent in basis.exponents.detach().cpu().tolist():
        endpoint_indices.append(endpoint_basis.term_index(tuple(exponent[1:])))
    metadata = ToraQ3KernelMetadata(
        exponents=basis.exponents,
        multiply_left=multiply_left,
        multiply_right=multiply_right,
        multiply_out=multiply_out,
        integrate_kept_in=integrate_kept_in,
        integrate_kept_out=integrate_kept_out,
        integrate_kept_factor=integrate_kept_factor,
        integrate_overflow_in=integrate_overflow_in,
        integrate_overflow_exponents=integrate_overflow_exponents,
        integrate_overflow_factor=integrate_overflow_factor,
        endpoint_out=torch.tensor(
            endpoint_indices, dtype=torch.long, device=device_t
        ),
        endpoint_exponents=endpoint_basis.exponents,
    )
    _METADATA_CACHE[key] = metadata
    return metadata


def _down(value: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(value, torch.full_like(value, -torch.inf))


def _up(value: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(value, torch.full_like(value, torch.inf))


def _expand_ulps(
    lower: torch.Tensor, upper: torch.Tensor, steps: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    result_lo = lower
    result_hi = upper
    for _step in range(steps):
        result_lo = _down(result_lo)
        result_hi = _up(result_hi)
    return result_lo, result_hi


def _interval_add(
    left_lo: torch.Tensor,
    left_hi: torch.Tensor,
    right_lo: torch.Tensor,
    right_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _down(left_lo + right_lo), _up(left_hi + right_hi)


def _interval_mul(
    left_lo: torch.Tensor,
    left_hi: torch.Tensor,
    right_lo: torch.Tensor,
    right_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    p1 = left_lo * right_lo
    p2 = left_lo * right_hi
    p3 = left_hi * right_lo
    p4 = left_hi * right_hi
    return _down(torch.minimum(torch.minimum(p1, p2), torch.minimum(p3, p4))), _up(
        torch.maximum(torch.maximum(p1, p2), torch.maximum(p3, p4))
    )


def _interval_square(
    lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_square = _up(torch.abs(lower) * torch.abs(lower))
    upper_square = _up(torch.abs(upper) * torch.abs(upper))
    maximum = torch.maximum(lower_square, upper_square)
    minimum = torch.minimum(lower_square, upper_square)
    crosses_zero = (lower <= 0.0) & (upper >= 0.0)
    return torch.where(crosses_zero, torch.zeros_like(minimum), _down(minimum)), maximum


def _interval_div_positive(
    lower: torch.Tensor, upper: torch.Tensor, divisor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.full_like(lower, float(divisor))
    return _down(lower / value), _up(upper / value)


def _monomial_bounds(
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    exponents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    exponent = exponents.unsqueeze(0)
    lower_power = torch.pow(domain_lo.unsqueeze(1), exponent)
    upper_power = torch.pow(domain_hi.unsqueeze(1), exponent)
    odd = (exponent.remainder(2) == 1)
    zero = exponent == 0
    crosses = (domain_lo.unsqueeze(1) <= 0.0) & (
        domain_hi.unsqueeze(1) >= 0.0
    )
    even_lower = torch.where(
        crosses,
        torch.zeros_like(lower_power),
        torch.minimum(lower_power, upper_power),
    )
    variable_lo = torch.where(odd, lower_power, even_lower)
    variable_hi = torch.where(
        odd, upper_power, torch.maximum(lower_power, upper_power)
    )
    variable_lo = torch.where(zero, torch.ones_like(variable_lo), variable_lo)
    variable_hi = torch.where(zero, torch.ones_like(variable_hi), variable_hi)
    result_lo = torch.ones_like(variable_lo[..., 0])
    result_hi = torch.ones_like(variable_hi[..., 0])
    for variable in range(exponents.shape[1]):
        result_lo, result_hi = _interval_mul(
            result_lo,
            result_hi,
            variable_lo[..., variable],
            variable_hi[..., variable],
        )
    return result_lo, result_hi


def fused_natural_range_kernel(
    coefficients: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    exponents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """F1: outward natural range for a fixed exponent table."""
    monomial_lo, monomial_hi = _monomial_bounds(
        domain_lo, domain_hi, exponents
    )
    left = coefficients * monomial_lo.unsqueeze(1)
    right = coefficients * monomial_hi.unsqueeze(1)
    term_lo = _down(torch.minimum(left, right))
    term_hi = _up(torch.maximum(left, right))
    result_lo = torch.sum(term_lo, dim=-1)
    result_hi = torch.sum(term_hi, dim=-1)
    epsilon = torch.finfo(coefficients.dtype).eps
    # Reserve twice the baseline reduction budget: one half covers the
    # reduction itself and the other covers vectorized monomial/product
    # ordering before that reduction.
    operations = float(4 * coefficients.shape[-1] + 1)
    gamma = (operations * epsilon) / (1.0 - operations * epsilon)
    lower_magnitude = torch.sum(torch.abs(term_lo), dim=-1)
    upper_magnitude = torch.sum(torch.abs(term_hi), dim=-1)
    tiny = torch.finfo(coefficients.dtype).tiny
    lower_error = _up(
        _up(lower_magnitude) * ((1.0 + gamma) * gamma) + tiny
    )
    upper_error = _up(
        _up(upper_magnitude) * ((1.0 + gamma) * gamma) + tiny
    )
    return _down(result_lo - lower_error), _up(result_hi + upper_error)


def _roundoff_radius(
    coefficients: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    exponents: torch.Tensor,
    *,
    operation_budget: int = 4096,
) -> torch.Tensor:
    monomial_lo, monomial_hi = _monomial_bounds(
        domain_lo, domain_hi, exponents
    )
    magnitude = torch.sum(
        torch.abs(coefficients)
        * torch.maximum(torch.abs(monomial_lo), torch.abs(monomial_hi)).unsqueeze(1),
        dim=-1,
    )
    epsilon = torch.finfo(coefficients.dtype).eps
    gamma = (float(operation_budget) * epsilon) / (
        1.0 - float(operation_budget) * epsilon
    )
    return _up(_up(magnitude) * (2.0 * gamma))


def _projected_product(
    left: torch.Tensor,
    right: torch.Tensor,
    multiply_left: torch.Tensor,
    multiply_right: torch.Tensor,
    multiply_out: torch.Tensor,
) -> torch.Tensor:
    products = left.index_select(-1, multiply_left) * right.index_select(
        -1, multiply_right
    )
    output = torch.zeros_like(left)
    target = multiply_out.view(1, 1, -1).expand_as(products)
    return output.scatter_add(-1, target, products)


def _interval_cosine(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    series_terms: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _sl, _sh, lower_cos_lo, lower_cos_hi = _point_sin_cos_enclosure_kernel(
        lower, series_terms=series_terms
    )
    _sl, _sh, upper_cos_lo, upper_cos_hi = _point_sin_cos_enclosure_kernel(
        upper, series_terms=series_terms
    )
    result_lo = torch.minimum(lower_cos_lo, upper_cos_lo)
    result_hi = torch.maximum(lower_cos_hi, upper_cos_hi)
    pi_lo = math.nextafter(math.pi, -math.inf)
    pi_hi = math.nextafter(math.pi, math.inf)
    for multiple in range(-3, 4):
        if multiple >= 0:
            critical_lo = float(multiple) * pi_lo
            critical_hi = float(multiple) * pi_hi
        else:
            critical_lo = float(multiple) * pi_hi
            critical_hi = float(multiple) * pi_lo
        possible = (lower <= critical_hi) & (upper >= critical_lo)
        if multiple % 2:
            result_lo = torch.where(possible, _down(-torch.ones_like(result_lo)), result_lo)
        else:
            result_hi = torch.where(possible, _up(torch.ones_like(result_hi)), result_hi)
    return result_lo, result_hi


def _aligned_sine_kernel(
    coefficients: torch.Tensor,
    remainder_lo: torch.Tensor,
    remainder_hi: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    exponents: torch.Tensor,
    multiply_left: torch.Tensor,
    multiply_right: torch.Tensor,
    multiply_out: torch.Tensor,
    *,
    series_terms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    constant = coefficients[..., 0]
    deviation = coefficients.clone()
    deviation[..., 0] = 0.0
    polynomial_lo, polynomial_hi = fused_natural_range_kernel(
        deviation, domain_lo, domain_hi, exponents
    )
    delta_lo, delta_hi = _interval_add(
        polynomial_lo, polynomial_hi, remainder_lo, remainder_hi
    )
    sin_lo, sin_hi, cos_lo, cos_hi = _point_sin_cos_enclosure_kernel(
        constant, series_terms=series_terms
    )
    sine_mid = sin_lo + 0.5 * (sin_hi - sin_lo)
    cosine_mid = cos_lo + 0.5 * (cos_hi - cos_lo)
    second_lo, second_hi = _interval_div_positive(-sin_hi, -sin_lo, 2.0)
    second_mid = second_lo + 0.5 * (second_hi - second_lo)
    square = _projected_product(
        deviation,
        deviation,
        multiply_left,
        multiply_right,
        multiply_out,
    )
    output = cosine_mid.unsqueeze(-1) * deviation
    output[..., 0] = sine_mid
    output = output + second_mid.unsqueeze(-1) * square

    result_lo = _down(sin_lo - sine_mid)
    result_hi = _up(sin_hi - sine_mid)
    cosine_error_lo = _down(cos_lo - cosine_mid)
    cosine_error_hi = _up(cos_hi - cosine_mid)
    error_lo, error_hi = _interval_mul(
        cosine_error_lo,
        cosine_error_hi,
        polynomial_lo,
        polynomial_hi,
    )
    result_lo, result_hi = _interval_add(result_lo, result_hi, error_lo, error_hi)
    propagated_lo, propagated_hi = _interval_mul(
        cos_lo, cos_hi, remainder_lo, remainder_hi
    )
    result_lo, result_hi = _interval_add(
        result_lo, result_hi, propagated_lo, propagated_hi
    )

    delta_square_lo, delta_square_hi = _interval_square(delta_lo, delta_hi)
    square_range_lo, square_range_hi = fused_natural_range_kernel(
        square, domain_lo, domain_hi, exponents
    )
    square_difference_lo, square_difference_hi = _interval_add(
        delta_square_lo,
        delta_square_hi,
        -square_range_hi,
        -square_range_lo,
    )
    second_route_lo, second_route_hi = _interval_mul(
        second_lo,
        second_hi,
        square_difference_lo,
        square_difference_hi,
    )
    result_lo, result_hi = _interval_add(
        result_lo, result_hi, second_route_lo, second_route_hi
    )
    second_error_lo = _down(second_lo - second_mid)
    second_error_hi = _up(second_hi - second_mid)
    second_uncertainty_lo, second_uncertainty_hi = _interval_mul(
        second_error_lo,
        second_error_hi,
        square_range_lo,
        square_range_hi,
    )
    result_lo, result_hi = _interval_add(
        result_lo,
        result_hi,
        second_uncertainty_lo,
        second_uncertainty_hi,
    )

    segment_delta_lo = torch.minimum(torch.zeros_like(delta_lo), delta_lo)
    segment_delta_hi = torch.maximum(torch.zeros_like(delta_hi), delta_hi)
    argument_lo, argument_hi = _interval_add(
        constant, constant, segment_delta_lo, segment_delta_hi
    )
    cosine_lo, cosine_hi = _interval_cosine(
        argument_lo, argument_hi, series_terms=series_terms
    )
    delta_squared_lo, delta_squared_hi = _interval_mul(
        delta_lo, delta_hi, delta_lo, delta_hi
    )
    delta_cube_lo, delta_cube_hi = _interval_mul(
        delta_squared_lo, delta_squared_hi, delta_lo, delta_hi
    )
    tail_lo, tail_hi = _interval_mul(
        -cosine_hi, -cosine_lo, delta_cube_lo, delta_cube_hi
    )
    tail_lo, tail_hi = _interval_div_positive(tail_lo, tail_hi, 6.0)
    result_lo, result_hi = _interval_add(result_lo, result_hi, tail_lo, tail_hi)
    route_radius = _roundoff_radius(output, domain_lo, domain_hi, exponents)
    result_lo, result_hi = _interval_add(
        result_lo, result_hi, -route_radius, route_radius
    )
    return output, result_lo, result_hi


def _rhs_kernel(
    coefficients: torch.Tensor,
    remainder_lo: torch.Tensor,
    remainder_hi: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
    *,
    series_terms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sine_coeff, sine_lo, sine_hi = _aligned_sine_kernel(
        coefficients[:, 2:3],
        remainder_lo[:, 2:3],
        remainder_hi[:, 2:3],
        domain_lo,
        domain_hi,
        metadata.exponents,
        metadata.multiply_left,
        metadata.multiply_right,
        metadata.multiply_out,
        series_terms=series_terms,
    )
    state1_coeff = -coefficients[:, 0:1] + 0.1 * sine_coeff
    state3_coeff = coefficients[:, 4:5].clone()
    state3_coeff[..., 0] = state3_coeff[..., 0] - 10.0
    zeros_coeff = torch.zeros_like(coefficients[:, 0:1])
    output_coeff = torch.cat(
        (
            coefficients[:, 1:2],
            state1_coeff,
            coefficients[:, 3:4],
            state3_coeff,
            zeros_coeff,
        ),
        dim=1,
    )
    state1_lo, state1_hi = _interval_add(
        -remainder_hi[:, 0:1],
        -remainder_lo[:, 0:1],
        _down(0.1 * sine_lo),
        _up(0.1 * sine_hi),
    )
    zeros = torch.zeros_like(remainder_lo[:, 0:1])
    output_lo = torch.cat(
        (
            remainder_lo[:, 1:2],
            state1_lo,
            remainder_lo[:, 3:4],
            remainder_lo[:, 4:5],
            zeros,
        ),
        dim=1,
    )
    output_hi = torch.cat(
        (
            remainder_hi[:, 1:2],
            state1_hi,
            remainder_hi[:, 3:4],
            remainder_hi[:, 4:5],
            zeros,
        ),
        dim=1,
    )
    route_radius = _roundoff_radius(
        output_coeff, domain_lo, domain_hi, metadata.exponents
    )
    route_radius[:, 4] = 0.0
    output_lo, output_hi = _interval_add(
        output_lo, output_hi, -route_radius, route_radius
    )
    return output_coeff, output_lo, output_hi


def _integrate_polynomial(
    coefficients: torch.Tensor, metadata: ToraQ3KernelMetadata
) -> torch.Tensor:
    kept = coefficients.index_select(-1, metadata.integrate_kept_in)
    kept = kept * metadata.integrate_kept_factor.view(1, 1, -1)
    output = torch.zeros_like(coefficients)
    target = metadata.integrate_kept_out.view(1, 1, -1).expand_as(kept)
    return output.scatter_add(-1, target, kept)


def _integrate_tm(
    coefficients: torch.Tensor,
    remainder_lo: torch.Tensor,
    remainder_hi: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = _integrate_polynomial(coefficients, metadata)
    overflow_coeff = coefficients.index_select(
        -1, metadata.integrate_overflow_in
    ) * metadata.integrate_overflow_factor.view(1, 1, -1)
    overflow_lo, overflow_hi = fused_natural_range_kernel(
        overflow_coeff,
        domain_lo,
        domain_hi,
        metadata.integrate_overflow_exponents,
    )
    tau_lo = domain_lo[:, 0:1]
    tau_hi = domain_hi[:, 0:1]
    routed_lo, routed_hi = _interval_mul(
        tau_lo, tau_hi, remainder_lo, remainder_hi
    )
    result_lo, result_hi = _interval_add(
        routed_lo, routed_hi, overflow_lo, overflow_hi
    )
    route_radius = _roundoff_radius(
        output, domain_lo, domain_hi, metadata.exponents
    )
    result_lo, result_hi = _interval_add(
        result_lo, result_hi, -route_radius, route_radius
    )
    return output, result_lo, result_hi


def fused_polynomial_picard_kernel(
    base_coefficients: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
    *,
    series_terms: int = 32,
) -> torch.Tensor:
    """F2: RHS plus the frozen K2 polynomial Picard phase."""
    current = base_coefficients
    zeros = torch.zeros_like(base_coefficients[..., 0])
    for _round in range(2):
        rhs_coeff, _rhs_lo, _rhs_hi = _rhs_kernel(
            current,
            zeros,
            zeros,
            domain_lo,
            domain_hi,
            metadata,
            series_terms=series_terms,
        )
        integrated = _integrate_polynomial(rhs_coeff, metadata)
        current = base_coefficients + integrated
    return current


def _image_kernel(
    base_coefficients: torch.Tensor,
    current_coefficients: torch.Tensor,
    current_lo: torch.Tensor,
    current_hi: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
    *,
    series_terms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rhs_coeff, rhs_lo, rhs_hi = _rhs_kernel(
        current_coefficients,
        current_lo,
        current_hi,
        domain_lo,
        domain_hi,
        metadata,
        series_terms=series_terms,
    )
    integrated_coeff, integrated_lo, integrated_hi = _integrate_tm(
        rhs_coeff,
        rhs_lo,
        rhs_hi,
        domain_lo,
        domain_hi,
        metadata,
    )
    output_coeff = base_coefficients + integrated_coeff
    route_radius = _roundoff_radius(
        torch.abs(base_coefficients) + torch.abs(integrated_coeff),
        domain_lo,
        domain_hi,
        metadata.exponents,
    )
    output_lo, output_hi = _interval_add(
        integrated_lo, integrated_hi, -route_radius, route_radius
    )
    output_lo[:, 4] = 0.0
    output_hi[:, 4] = 0.0
    return output_coeff, output_lo, output_hi


def fused_image_kernel(
    base_coefficients: torch.Tensor,
    current_coefficients: torch.Tensor,
    current_lo: torch.Tensor,
    current_hi: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
    *,
    series_terms: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One complete fixed-shape RHS and integration image."""
    return _image_kernel(
        base_coefficients,
        current_coefficients,
        current_lo,
        current_hi,
        domain_lo,
        domain_hi,
        metadata,
        series_terms=series_terms,
    )


def fused_remainder_initialize_kernel(
    base_coefficients: torch.Tensor,
    candidate_coefficients: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
    *,
    series_terms: int = 32,
) -> TensorTuple:
    seed = torch.tensor(
        [0.01, 0.01, 0.01, 0.01, 0.0],
        dtype=base_coefficients.dtype,
        device=base_coefficients.device,
    ).view(1, 5)
    current_hi = seed.expand(base_coefficients.shape[0], -1).clone()
    current_lo = -current_hi
    image_coeff, image_lo, image_hi = fused_image_kernel(
        base_coefficients,
        candidate_coefficients,
        current_lo,
        current_hi,
        domain_lo,
        domain_hi,
        metadata,
        series_terms=series_terms,
    )
    initial_margin = torch.minimum(image_lo - current_lo, current_hi - image_hi)
    initial_subset = initial_margin >= 0.0
    difference = image_coeff - candidate_coefficients
    roundoff_lo, roundoff_hi = fused_natural_range_kernel(
        difference, domain_lo, domain_hi, metadata.exponents
    )
    roundoff_lo[:, 4] = 0.0
    roundoff_hi[:, 4] = 0.0
    return (
        candidate_coefficients,
        current_lo,
        current_hi,
        initial_subset,
        initial_margin,
        roundoff_lo,
        roundoff_hi,
    )


def fused_remainder_round_kernel(
    base_coefficients: torch.Tensor,
    current_coefficients: torch.Tensor,
    current_lo: torch.Tensor,
    current_hi: torch.Tensor,
    roundoff_lo: torch.Tensor,
    roundoff_hi: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
    *,
    series_terms: int = 32,
) -> TensorTuple:
    image_coeff, image_lo, image_hi = fused_image_kernel(
        base_coefficients,
        current_coefficients,
        current_lo,
        current_hi,
        domain_lo,
        domain_hi,
        metadata,
        series_terms=series_terms,
    )
    candidate_lo, candidate_hi = _interval_add(
        image_lo, image_hi, roundoff_lo, roundoff_hi
    )
    candidate_lo[:, 4] = 0.0
    candidate_hi[:, 4] = 0.0
    margin = torch.minimum(candidate_lo - current_lo, current_hi - candidate_hi)
    shrink = margin >= 0.0
    accepted_lo = torch.where(shrink, candidate_lo, current_lo)
    accepted_hi = torch.where(shrink, candidate_hi, current_hi)
    return image_coeff, accepted_lo, accepted_hi, shrink, margin


def fused_remainder_picard_kernel(
    base_coefficients: torch.Tensor,
    candidate_coefficients: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
    *,
    series_terms: int = 32,
) -> TensorTuple:
    """F3: initial image and ten componentwise remainder rounds."""
    (
        current_coeff,
        current_lo,
        current_hi,
        initial_subset,
        initial_margin,
        roundoff_lo,
        roundoff_hi,
    ) = fused_remainder_initialize_kernel(
        base_coefficients,
        candidate_coefficients,
        domain_lo,
        domain_hi,
        metadata,
        series_terms=series_terms,
    )
    all_rounds = torch.ones_like(initial_subset)
    for _round in range(10):
        (
            current_coeff,
            current_lo,
            current_hi,
            shrink,
            margin,
        ) = fused_remainder_round_kernel(
            base_coefficients,
            current_coeff,
            current_lo,
            current_hi,
            roundoff_lo,
            roundoff_hi,
            domain_lo,
            domain_hi,
            metadata,
            series_terms=series_terms,
        )
        all_rounds = all_rounds & shrink
    return (
        current_coeff,
        current_lo,
        current_hi,
        initial_subset,
        all_rounds,
        initial_margin,
    )


def _endpoint_coefficients(
    coefficients: torch.Tensor,
    h: float,
    metadata: ToraQ3KernelMetadata,
) -> torch.Tensor:
    time_powers = metadata.exponents[:, 0]
    factors = torch.full(
        (coefficients.shape[0],),
        float(h),
        dtype=coefficients.dtype,
        device=coefficients.device,
    ).unsqueeze(1).pow(time_powers.unsqueeze(0))
    scaled = coefficients * factors.unsqueeze(1)
    output = torch.zeros(
        (
            coefficients.shape[0],
            coefficients.shape[1],
            metadata.endpoint_exponents.shape[0],
        ),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    target = metadata.endpoint_out.view(1, 1, -1).expand_as(scaled)
    return output.scatter_add(-1, target, scaled)


def fused_finalize_kernel(
    coefficients: torch.Tensor,
    remainder_lo: torch.Tensor,
    remainder_hi: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    initial_subset: torch.Tensor,
    all_rounds: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
    *,
    h: float = 0.1,
) -> TensorTuple:
    """F4: tube, exact-time endpoint aggregation, and leaf predicates."""
    polynomial_lo, polynomial_hi = fused_natural_range_kernel(
        coefficients, domain_lo, domain_hi, metadata.exponents
    )
    tube_lo, tube_hi = _interval_add(
        polynomial_lo, polynomial_hi, remainder_lo, remainder_hi
    )
    tube_radius = _roundoff_radius(
        coefficients, domain_lo, domain_hi, metadata.exponents
    )
    tube_lo, tube_hi = _interval_add(
        tube_lo, tube_hi, -tube_radius, tube_radius
    )
    endpoint_coeff = _endpoint_coefficients(coefficients, h, metadata)
    endpoint_domain_lo = domain_lo[:, 1:]
    endpoint_domain_hi = domain_hi[:, 1:]
    endpoint_poly_lo, endpoint_poly_hi = fused_natural_range_kernel(
        endpoint_coeff,
        endpoint_domain_lo,
        endpoint_domain_hi,
        metadata.endpoint_exponents,
    )
    endpoint_lo, endpoint_hi = _interval_add(
        endpoint_poly_lo, endpoint_poly_hi, remainder_lo, remainder_hi
    )
    endpoint_radius = _roundoff_radius(
        endpoint_coeff,
        endpoint_domain_lo,
        endpoint_domain_hi,
        metadata.endpoint_exponents,
    )
    endpoint_lo, endpoint_hi = _interval_add(
        endpoint_lo, endpoint_hi, -endpoint_radius, endpoint_radius
    )
    finite = (
        torch.isfinite(coefficients).all(dim=(1, 2))
        & torch.isfinite(remainder_lo).all(dim=1)
        & torch.isfinite(remainder_hi).all(dim=1)
        & torch.isfinite(tube_lo).all(dim=1)
        & torch.isfinite(tube_hi).all(dim=1)
        & torch.isfinite(endpoint_lo).all(dim=1)
        & torch.isfinite(endpoint_hi).all(dim=1)
        & (remainder_lo <= remainder_hi).all(dim=1)
        & (domain_lo <= domain_hi).all(dim=1)
    )
    property_ok = (
        torch.maximum(torch.abs(tube_lo[:, :4]), torch.abs(tube_hi[:, :4]))
        <= 2.0
    ).all(dim=1)
    initial_ok = initial_subset.all(dim=1)
    rounds_ok = all_rounds.all(dim=1)
    accepted = finite & initial_ok & rounds_ok & property_ok
    return (
        endpoint_lo,
        endpoint_hi,
        tube_lo,
        tube_hi,
        finite,
        initial_ok,
        rounds_ok,
        property_ok,
        accepted,
    )


def fused_full_step_kernel(
    coefficients: torch.Tensor,
    remainder_lo: torch.Tensor,
    remainder_hi: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
    metadata: ToraQ3KernelMetadata,
    *,
    h: float = 0.1,
    series_terms: int = 32,
) -> TensorTuple:
    """F5: complete fixed-shape local TORA-Q3 step."""
    del remainder_lo, remainder_hi
    candidate = fused_polynomial_picard_kernel(
        coefficients,
        domain_lo,
        domain_hi,
        metadata,
        series_terms=series_terms,
    )
    (
        result_coeff,
        result_lo,
        result_hi,
        initial_subset,
        all_rounds,
        initial_margin,
    ) = fused_remainder_picard_kernel(
        coefficients,
        candidate,
        domain_lo,
        domain_hi,
        metadata,
        series_terms=series_terms,
    )
    final = fused_finalize_kernel(
        result_coeff,
        result_lo,
        result_hi,
        domain_lo,
        domain_hi,
        initial_subset,
        all_rounds,
        metadata,
        h=h,
    )
    return (
        result_coeff,
        result_lo,
        result_hi,
        *final,
        initial_margin,
    )


def _signature(base: BatchedTaylorModel, h: float, series_terms: int) -> tuple[Any, ...]:
    tensor_layouts = tuple(
        (
            tuple(value.shape),
            tuple(value.stride()),
            int(value.storage_offset()),
            bool(value.requires_grad),
        )
        for value in (
            base.poly.coeffs,
            base.rem_lo,
            base.rem_hi,
            base.domain_lo,
            base.domain_hi,
        )
    )
    return (
        tensor_layouts,
        str(base.poly.coeffs.device),
        base.poly.coeffs.dtype,
        bool(torch.is_grad_enabled()),
        float(h),
        int(series_terms),
    )


def _kernel_callable(
    metadata: ToraQ3KernelMetadata,
    *,
    h: float,
    series_terms: int,
) -> Callable[..., TensorTuple]:
    def fixed(
        coefficients: torch.Tensor,
        remainder_lo: torch.Tensor,
        remainder_hi: torch.Tensor,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
    ) -> TensorTuple:
        return fused_full_step_kernel(
            coefficients,
            remainder_lo,
            remainder_hi,
            domain_lo,
            domain_hi,
            metadata,
            h=h,
            series_terms=series_terms,
        )

    return fixed


def _compiled_contains_eager(compiled: TensorTuple, eager: TensorTuple) -> bool:
    if len(compiled) != len(eager):
        return False
    if not torch.equal(compiled[0], eager[0]):
        return False
    interval_pairs = ((1, 2), (3, 4), (5, 6))
    for lower_index, upper_index in interval_pairs:
        if not bool(torch.all(compiled[lower_index] <= eager[lower_index])):
            return False
        if not bool(torch.all(compiled[upper_index] >= eager[upper_index])):
            return False
    for predicate_index in range(7, 12):
        if not bool(torch.all((~compiled[predicate_index]) | eager[predicate_index])):
            return False
    return bool(torch.all(compiled[12] <= eager[12]))


def run_fused_full_step(
    base: BatchedTaylorModel,
    *,
    backend: str = "eager",
    h: float = 0.1,
    series_terms: int = 32,
) -> tuple[TensorTuple, str]:
    if backend not in {"eager", "compiled"}:
        raise ValueError("fused backend must be eager or compiled")
    metadata = tora_q3_kernel_metadata(base.poly.coeffs.device)
    eager_callable = _kernel_callable(
        metadata, h=h, series_terms=series_terms
    )
    arguments = (
        base.poly.coeffs,
        base.rem_lo,
        base.rem_hi,
        base.domain_lo,
        base.domain_hi,
    )
    if backend == "eager" or base.poly.coeffs.device.type != "cuda":
        return eager_callable(*arguments), "eager"
    signature = _signature(base, h, series_terms)
    if signature in _COMPILED_DISABLED:
        return eager_callable(*arguments), "eager_fallback"
    compiled_callable = _COMPILED_CACHE.get(signature)
    if compiled_callable is None:
        try:
            compiled_callable = torch.compile(
                eager_callable, fullgraph=True, dynamic=False
            )
            eager = eager_callable(*arguments)
            compiled = compiled_callable(*arguments)
            bitwise = all(
                torch.equal(left, right)
                for left, right in zip(compiled, eager, strict=True)
            )
            outward = bitwise or _compiled_contains_eager(compiled, eager)
            if not outward:
                _COMPILED_DISABLED[signature] = (
                    "compiled output neither bitwise-matches nor outwardly contains eager"
                )
                return eager, "eager_fallback"
            _COMPILED_CACHE[signature] = compiled_callable
            _COMPILED_VERIFIED[signature] = {
                "bitwise": bitwise,
                "outward_contains_eager": outward,
            }
            return compiled, "compiled_verified"
        except Exception as exception:
            _COMPILED_DISABLED[signature] = f"{type(exception).__name__}: {exception}"
            return eager_callable(*arguments), "eager_fallback"
    return compiled_callable(*arguments), "compiled_verified"


def _segmented_execute(
    arguments: tuple[torch.Tensor, ...],
    callables: tuple[
        Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]
    ],
) -> TensorTuple:
    coefficients, _remainder_lo, _remainder_hi, domain_lo, domain_hi = arguments
    polynomial, initialize, remainder_round, finalize = callables
    candidate = polynomial(coefficients, domain_lo, domain_hi)
    (
        current_coeff,
        current_lo,
        current_hi,
        initial_subset,
        initial_margin,
        roundoff_lo,
        roundoff_hi,
    ) = initialize(coefficients, candidate, domain_lo, domain_hi)
    all_rounds = torch.ones_like(initial_subset)
    for _round_index in range(10):
        (
            current_coeff,
            current_lo,
            current_hi,
            shrink,
            margin,
        ) = remainder_round(
            coefficients,
            current_coeff,
            current_lo,
            current_hi,
            roundoff_lo,
            roundoff_hi,
            domain_lo,
            domain_hi,
        )
        all_rounds = all_rounds & shrink
    final = finalize(
        current_coeff,
        current_lo,
        current_hi,
        domain_lo,
        domain_hi,
        initial_subset,
        all_rounds,
    )
    return (
        current_coeff,
        current_lo,
        current_hi,
        *final,
        initial_margin,
    )


def _segmented_callables(
    metadata: ToraQ3KernelMetadata,
    *,
    h: float,
    series_terms: int,
    compiled_padding: bool = False,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    def polynomial(
        coefficients: torch.Tensor,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
    ) -> torch.Tensor:
        return fused_polynomial_picard_kernel(
            coefficients,
            domain_lo,
            domain_hi,
            metadata,
            series_terms=series_terms,
        )

    def initialize(
        coefficients: torch.Tensor,
        candidate: torch.Tensor,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
    ) -> TensorTuple:
        result = fused_remainder_initialize_kernel(
            coefficients,
            candidate,
            domain_lo,
            domain_hi,
            metadata,
            series_terms=series_terms,
        )
        if not compiled_padding:
            return result
        (
            current_coeff,
            current_lo,
            current_hi,
            _initial_subset,
            initial_margin,
            roundoff_lo,
            roundoff_hi,
        ) = result
        padded_margin = initial_margin
        for _step in range(16):
            padded_margin = _down(padded_margin)
        padded_margin = torch.where(
            initial_margin == 0.0, initial_margin, padded_margin
        )
        roundoff_lo, roundoff_hi = _expand_ulps(roundoff_lo, roundoff_hi)
        roundoff_lo[:, 4] = 0.0
        roundoff_hi[:, 4] = 0.0
        return (
            current_coeff,
            current_lo,
            current_hi,
            padded_margin >= 0.0,
            padded_margin,
            roundoff_lo,
            roundoff_hi,
        )

    def remainder_round(
        coefficients: torch.Tensor,
        current_coeff: torch.Tensor,
        current_lo: torch.Tensor,
        current_hi: torch.Tensor,
        roundoff_lo: torch.Tensor,
        roundoff_hi: torch.Tensor,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
    ) -> TensorTuple:
        result = fused_remainder_round_kernel(
            coefficients,
            current_coeff,
            current_lo,
            current_hi,
            roundoff_lo,
            roundoff_hi,
            domain_lo,
            domain_hi,
            metadata,
            series_terms=series_terms,
        )
        if not compiled_padding:
            return result
        image_coeff, accepted_lo, accepted_hi, original_shrink, _margin = result
        padded_lo, padded_hi = _expand_ulps(accepted_lo, accepted_hi)
        padded_lo[:, 4] = 0.0
        padded_hi[:, 4] = 0.0
        padded_margin = torch.minimum(
            padded_lo - current_lo, current_hi - padded_hi
        )
        shrink = original_shrink & (padded_margin >= 0.0)
        return (
            image_coeff,
            torch.where(shrink, padded_lo, current_lo),
            torch.where(shrink, padded_hi, current_hi),
            shrink,
            padded_margin,
        )

    def finalize(
        coefficients: torch.Tensor,
        remainder_lo: torch.Tensor,
        remainder_hi: torch.Tensor,
        domain_lo: torch.Tensor,
        domain_hi: torch.Tensor,
        initial_subset: torch.Tensor,
        all_rounds: torch.Tensor,
    ) -> TensorTuple:
        result = fused_finalize_kernel(
            coefficients,
            remainder_lo,
            remainder_hi,
            domain_lo,
            domain_hi,
            initial_subset,
            all_rounds,
            metadata,
            h=h,
        )
        if not compiled_padding:
            return result
        (
            endpoint_lo,
            endpoint_hi,
            tube_lo,
            tube_hi,
            finite,
            initial_ok,
            rounds_ok,
            _property_ok,
            _accepted,
        ) = result
        endpoint_lo, endpoint_hi = _expand_ulps(endpoint_lo, endpoint_hi)
        tube_lo, tube_hi = _expand_ulps(tube_lo, tube_hi)
        property_ok = (
            torch.maximum(torch.abs(tube_lo[:, :4]), torch.abs(tube_hi[:, :4]))
            <= 2.0
        ).all(dim=1)
        accepted = finite & initial_ok & rounds_ok & property_ok
        return (
            endpoint_lo,
            endpoint_hi,
            tube_lo,
            tube_hi,
            finite,
            initial_ok,
            rounds_ok,
            property_ok,
            accepted,
        )

    return polynomial, initialize, remainder_round, finalize


def run_segmented_fused_step(
    base: BatchedTaylorModel,
    *,
    backend: str = "compiled",
    h: float = 0.1,
    series_terms: int = 32,
) -> tuple[TensorTuple, str]:
    """Run F2/F3-image/F4 fullgraphs with no per-round host decision."""
    if backend not in {"eager", "compiled"}:
        raise ValueError("segmented fused backend must be eager or compiled")
    metadata = tora_q3_kernel_metadata(base.poly.coeffs.device)
    eager_callables = _segmented_callables(
        metadata, h=h, series_terms=series_terms
    )
    arguments = (
        base.poly.coeffs,
        base.rem_lo,
        base.rem_hi,
        base.domain_lo,
        base.domain_hi,
    )
    if backend == "eager" or base.poly.coeffs.device.type != "cuda":
        return _segmented_execute(arguments, eager_callables), "segmented_eager"
    signature = ("segmented", *_signature(base, h, series_terms))
    if signature in _SEGMENTED_DISABLED:
        return _segmented_execute(arguments, eager_callables), "segmented_eager_fallback"
    compiled_callables = _SEGMENTED_CACHE.get(signature)
    if compiled_callables is None:
        try:
            compiled_source_callables = _segmented_callables(
                metadata,
                h=h,
                series_terms=series_terms,
                compiled_padding=True,
            )
            compiled_callables = tuple(
                torch.compile(function, fullgraph=True, dynamic=False)
                for function in compiled_source_callables
            )
            eager = _segmented_execute(arguments, eager_callables)
            compiled = _segmented_execute(arguments, compiled_callables)
            bitwise = all(
                torch.equal(left, right)
                for left, right in zip(compiled, eager, strict=True)
            )
            outward = bitwise or _compiled_contains_eager(compiled, eager)
            if not outward:
                _SEGMENTED_DISABLED[signature] = (
                    "segmented output neither bitwise-matches nor outwardly contains eager"
                )
                return eager, "segmented_eager_fallback"
            _SEGMENTED_CACHE[signature] = compiled_callables
            _SEGMENTED_VERIFIED[signature] = {
                "bitwise": bitwise,
                "fullgraph_stage_count": 4,
                "logical_stage_invocations": 13,
                "maximum_initial_margin_difference": float(
                    torch.max(torch.abs(compiled[12] - eager[12])).item()
                ),
                "outward_contains_eager": outward,
                "per_round_host_decisions": 0,
            }
            return compiled, "segmented_compiled_verified"
        except Exception as exception:
            _SEGMENTED_DISABLED[signature] = f"{type(exception).__name__}: {exception}"
            return _segmented_execute(arguments, eager_callables), "segmented_eager_fallback"
    return _segmented_execute(arguments, compiled_callables), "segmented_compiled_verified"


def fused_kernel_status() -> dict[str, Any]:
    return {
        "cached_metadata_signatures": len(_METADATA_CACHE),
        "compiled_disabled_signatures": {
            repr(key): reason for key, reason in _COMPILED_DISABLED.items()
        },
        "compiled_verified_signatures": {
            repr(key): value for key, value in _COMPILED_VERIFIED.items()
        },
        "segmented_disabled_signatures": {
            repr(key): reason for key, reason in _SEGMENTED_DISABLED.items()
        },
        "segmented_verified_signatures": {
            repr(key): value for key, value in _SEGMENTED_VERIFIED.items()
        },
    }


def fused_algorithm_aligned_q3_step(
    base: BatchedTaylorModel,
    *,
    backend: str = "eager",
    h: float = 0.1,
    series_terms: int = 32,
    batched_fail_closed: bool = False,
) -> ToraQ3Step:
    if base.poly.coeffs.dtype != torch.float64:
        raise TypeError("formal fused TORA-Q3 lane requires float64")
    if base.poly.coeffs.shape[1:] != (5, 84):
        raise ValueError("fused TORA-Q3 lane requires [B,5,84] coefficients")
    if base.domain_lo.shape != (base.poly.batch, 6):
        raise ValueError("fused TORA-Q3 lane requires [B,6] domain tensors")
    if backend == "segmented_compiled":
        outputs, _selected_backend = run_segmented_fused_step(
            base, backend="compiled", h=h, series_terms=series_terms
        )
    else:
        outputs, _selected_backend = run_fused_full_step(
            base, backend=backend, h=h, series_terms=series_terms
        )
    (
        coefficients,
        remainder_lo,
        remainder_hi,
        endpoint_lo,
        endpoint_hi,
        tube_lo,
        tube_hi,
        finite,
        initial_ok,
        rounds_ok,
        property_ok,
        accepted,
        initial_margin,
    ) = outputs
    segment = BatchedTaylorModel(
        BatchedPolynomial(coefficients, base.poly.basis),
        remainder_lo,
        remainder_hi,
        base.domain_lo,
        base.domain_hi,
        DenseRemainderLedger(
            {"picard_residual": (remainder_lo.clone(), remainder_hi.clone())}
        ),
        base.range_policy,
        base.range_trace,
    )
    if batched_fail_closed:
        require_dense_condition(
            accepted,
            "fused local validation or property check failed",
            RuntimeError,
        )
        all_accepted = True
    else:
        all_accepted = bool(torch.all(accepted))
    return ToraQ3Step(
        segment_tm=segment,
        endpoint_tm=segment.endpoint(0, float(h)) if all_accepted else None,
        tube_lower=tube_lo,
        tube_upper=tube_hi,
        endpoint_lower=endpoint_lo,
        endpoint_upper=endpoint_hi,
        finite_ok_by_leaf=finite,
        initial_subset_ok_by_leaf=initial_ok,
        all_remainder_rounds_ok_by_leaf=rounds_ok,
        local_property_ok_by_leaf=property_ok,
        composed_property_ok_by_leaf=property_ok,
        accepted_by_leaf=accepted,
        initial_shrink_mask=initial_margin >= 0.0,
        initial_margin=initial_margin,
        round_trace=(),
        polynomial_trace=(),
        status="validated" if all_accepted else "failed",
        message="" if all_accepted else "fused local validation or property check failed",
    )


def fused_tora_q3_boundary_from_model(
    model: BatchedTaylorModel,
) -> ToraQ3AffineBoundary:
    """Extract a fixed Q3 affine boundary with a batched fail-closed check."""
    if model.poly.basis.dim != 6 or model.poly.out_dim != 5:
        raise ValueError("TORA boundary model contract mismatch")
    basis = model.poly.basis
    center = model.poly.coeffs[..., basis.constant_index]
    linear = torch.zeros(
        (model.poly.batch, 5, 5),
        dtype=model.poly.coeffs.dtype,
        device=model.poly.coeffs.device,
    )
    allowed = torch.zeros(
        basis.num_terms, dtype=torch.bool, device=basis.device
    )
    allowed[basis.constant_index] = True
    for variable in range(5):
        exponent = [0] * 6
        exponent[variable + 1] = 1
        index = basis.term_index(exponent)
        linear[:, :, variable] = model.poly.coeffs[..., index]
        allowed[index] = True
    require_dense_condition(
        ~torch.any(model.poly.coeffs[..., ~allowed] != 0),
        "boundary model contains non-affine or local-time terms",
    )
    return ToraQ3AffineBoundary(
        center,
        linear,
        model.rem_lo,
        model.rem_hi,
    )


def compose_fused_tora_q3_step(
    local_step: ToraQ3Step,
    parameterization: ToraQ3AffineCarry,
    *,
    h: float = 0.1,
) -> ToraQ3Step:
    """Compose a fused local step with one deferred fail-closed decision."""
    composed = compose_tora_q3_tm(local_step.segment_tm, parameterization)
    tube_lower, tube_upper = composed.range_bound(
        context="tora_fused_composed_step_tube"
    )
    endpoint_lower, endpoint_upper = _endpoint_bounds(composed, h=h)
    finite = (
        torch.isfinite(tube_lower).all(dim=1)
        & torch.isfinite(tube_upper).all(dim=1)
        & torch.isfinite(endpoint_lower).all(dim=1)
        & torch.isfinite(endpoint_upper).all(dim=1)
    )
    property_ok = (
        torch.maximum(
            torch.abs(tube_lower[:, :4]), torch.abs(tube_upper[:, :4])
        )
        <= 2.0
    ).all(dim=1)
    finite_ok = local_step.finite_ok_by_leaf & finite
    accepted = (
        finite_ok
        & local_step.initial_subset_ok_by_leaf
        & local_step.all_remainder_rounds_ok_by_leaf
        & local_step.local_property_ok_by_leaf
        & property_ok
    )
    require_dense_condition(
        accepted,
        "composed fused TORA-Q3 validation or property check failed",
        RuntimeError,
    )
    return ToraQ3Step(
        segment_tm=composed,
        endpoint_tm=composed.endpoint(0, float(h)),
        tube_lower=tube_lower,
        tube_upper=tube_upper,
        endpoint_lower=endpoint_lower,
        endpoint_upper=endpoint_upper,
        finite_ok_by_leaf=finite_ok,
        initial_subset_ok_by_leaf=local_step.initial_subset_ok_by_leaf,
        all_remainder_rounds_ok_by_leaf=(
            local_step.all_remainder_rounds_ok_by_leaf
        ),
        local_property_ok_by_leaf=local_step.local_property_ok_by_leaf,
        composed_property_ok_by_leaf=property_ok,
        accepted_by_leaf=accepted,
        initial_shrink_mask=local_step.initial_shrink_mask,
        initial_margin=local_step.initial_margin,
        round_trace=local_step.round_trace,
        polynomial_trace=local_step.polynomial_trace,
        status="validated",
        message="",
    )


__all__ = [
    "ToraQ3KernelMetadata",
    "compose_fused_tora_q3_step",
    "fused_algorithm_aligned_q3_step",
    "fused_finalize_kernel",
    "fused_full_step_kernel",
    "fused_image_kernel",
    "fused_kernel_status",
    "fused_natural_range_kernel",
    "fused_polynomial_picard_kernel",
    "fused_remainder_initialize_kernel",
    "fused_remainder_picard_kernel",
    "fused_remainder_round_kernel",
    "fused_tora_q3_boundary_from_model",
    "run_fused_full_step",
    "run_segmented_fused_step",
    "tora_q3_kernel_metadata",
]
