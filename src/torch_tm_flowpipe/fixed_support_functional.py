"""Tensor-only functional core for fixed-support Taylor-model reachability.

The public object model remains useful for inspection.  This module exposes the
same DiffReach-style restricted-support arithmetic as a fixed tensor state so a
logical step or a bounded chunk can be compiled without constructing Python
Taylor-model, interval, ledger, or symbolic-state dataclasses in the hot loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .fixed_support import FixedSupportKernelPlan


TensorState = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class FixedSupportFunctionalState:
    model_coeffs: torch.Tensor
    model_rem_lo: torch.Tensor
    model_rem_hi: torch.Tensor
    parameter_coeffs: torch.Tensor
    parameter_rem_lo: torch.Tensor
    parameter_rem_hi: torch.Tensor
    phi_buffer: torch.Tensor
    j_lo: torch.Tensor
    j_hi: torch.Tensor
    queue_count: torch.Tensor
    inverse_scale: torch.Tensor
    active_mask: torch.Tensor
    first_failure_index: torch.Tensor
    step_index: torch.Tensor
    last_endpoint_lo: torch.Tensor
    last_endpoint_hi: torch.Tensor
    last_tube_lo: torch.Tensor
    last_tube_hi: torch.Tensor
    tube_hull_lo: torch.Tensor
    tube_hull_hi: torch.Tensor
    has_valid_tube: torch.Tensor
    last_initial_mask: torch.Tensor
    last_round_masks: torch.Tensor
    all_initial_masks: torch.Tensor
    all_round_masks: torch.Tensor
    normalization_scale: torch.Tensor

    def tensors(self) -> TensorState:
        return tuple(getattr(self, field) for field in self.__dataclass_fields__)

    @classmethod
    def from_tensors(cls, values: TensorState) -> "FixedSupportFunctionalState":
        if len(values) != len(cls.__dataclass_fields__):
            raise ValueError("functional state tensor count mismatch")
        return cls(*values)


@dataclass(frozen=True)
class FixedSupportFunctionalResult:
    final_state: FixedSupportFunctionalState
    endpoint_lo: torch.Tensor | None
    endpoint_hi: torch.Tensor | None
    tube_lo: torch.Tensor | None
    tube_hi: torch.Tensor | None
    initial_inclusion_masks: torch.Tensor | None
    round_inclusion_masks: torch.Tensor | None
    validated_steps_per_batch: torch.Tensor
    completed_mask: torch.Tensor
    first_failure_step: int | None
    validated_steps: int
    requested_steps: int
    host_synchronizations: int
    device_transfers: int
    trace_mode: bool


def _interval_add(
    left_lo: torch.Tensor,
    left_hi: torch.Tensor,
    right_lo: torch.Tensor,
    right_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return left_lo + right_lo, left_hi + right_hi


def _interval_sub(
    left_lo: torch.Tensor,
    left_hi: torch.Tensor,
    right_lo: torch.Tensor,
    right_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return left_lo - right_hi, left_hi - right_lo


def _interval_scale(
    lo: torch.Tensor, hi: torch.Tensor, factor: torch.Tensor | float
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled_lo = lo * factor
    scaled_hi = hi * factor
    return torch.minimum(scaled_lo, scaled_hi), torch.maximum(scaled_lo, scaled_hi)


def _interval_mul(
    left_lo: torch.Tensor,
    left_hi: torch.Tensor,
    right_lo: torch.Tensor,
    right_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    products = torch.stack(
        (
            left_lo * right_lo,
            left_lo * right_hi,
            left_hi * right_lo,
            left_hi * right_hi,
        ),
        dim=0,
    )
    return products.amin(dim=0), products.amax(dim=0)


def _interval_affine(
    lo: torch.Tensor, hi: torch.Tensor, matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = matrix >= 0
    positive_matrix = matrix * positive
    negative_matrix = matrix * (~positive)
    lo_col = lo.unsqueeze(-1)
    hi_col = hi.unsqueeze(-1)
    out_lo = torch.sum(positive_matrix @ lo_col + negative_matrix @ hi_col, dim=-1)
    out_hi = torch.sum(positive_matrix @ hi_col + negative_matrix @ lo_col, dim=-1)
    return out_lo, out_hi


def _linear_form_interval(
    coefficients: torch.Tensor,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_products = coefficients * box_lo[:, None, :]
    upper_products = coefficients * box_hi[:, None, :]
    return (
        torch.minimum(lower_products, upper_products).sum(dim=-1),
        torch.maximum(lower_products, upper_products).sum(dim=-1),
    )


def _groups(
    coefficients: torch.Tensor, plan: FixedSupportKernelPlan
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        coefficients[..., plan.constant_slot],
        coefficients.index_select(-1, plan.linear_slots),
        coefficients.index_select(-1, plan.time_cross_slots),
    )


def _polynomial_mul_trunc(
    left: torch.Tensor,
    right: torch.Tensor,
    plan: FixedSupportKernelPlan,
) -> torch.Tensor:
    result = torch.zeros_like(left)
    for stage in range(plan.multiply_left.shape[0]):
        left_stage = left.index_select(-1, plan.multiply_left[stage])
        right_stage = right.index_select(-1, plan.multiply_right[stage])
        contribution = left_stage * right_stage * plan.multiply_sign[stage]
        result = result + torch.where(
            plan.multiply_valid[stage], contribution, torch.zeros_like(contribution)
        )
    return result


def _polynomial_range(
    coefficients: torch.Tensor,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    constant, linear, cross = _groups(coefficients, plan)
    time_lo = box_lo[:, plan.local_time_index]
    time_hi = box_hi[:, plan.local_time_index]
    time_interval_lo = time_lo[:, None].expand(-1, coefficients.shape[1])
    time_interval_hi = time_hi[:, None].expand(-1, coefficients.shape[1])
    linear_spatial = linear.index_select(-1, plan.spatial_indices)
    cross_spatial = cross.index_select(-1, plan.spatial_indices)
    spatial_lo = box_lo.index_select(-1, plan.spatial_indices)
    spatial_hi = box_hi.index_select(-1, plan.spatial_indices)
    linear_x_lo, linear_x_hi = _linear_form_interval(
        linear_spatial, spatial_lo, spatial_hi
    )
    cross_x_lo, cross_x_hi = _linear_form_interval(
        cross_spatial, spatial_lo, spatial_hi
    )
    linear_time = linear[..., plan.local_time_index]
    cross_time = cross[..., plan.local_time_index]
    inner_time_lo, inner_time_hi = _interval_mul(
        cross_time,
        cross_time,
        time_interval_lo,
        time_interval_hi,
    )
    linear_cross_lo, linear_cross_hi = _interval_add(
        linear_time,
        linear_time,
        cross_x_lo,
        cross_x_hi,
    )
    inner_lo, inner_hi = _interval_add(
        inner_time_lo, inner_time_hi, linear_cross_lo, linear_cross_hi
    )
    time_product_lo, time_product_hi = _interval_mul(
        time_interval_lo,
        time_interval_hi,
        inner_lo,
        inner_hi,
    )
    return _interval_add(
        time_product_lo,
        time_product_hi,
        constant + linear_x_lo,
        constant + linear_x_hi,
    )


def _polynomial_mul_ctrunc(
    left: torch.Tensor,
    right: torch.Tensor,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kept = _polynomial_mul_trunc(left, right, plan)
    _, left_linear, left_cross = _groups(left, plan)
    _, right_linear, right_cross = _groups(right, plan)
    spatial_lo = box_lo.index_select(-1, plan.spatial_indices)
    spatial_hi = box_hi.index_select(-1, plan.spatial_indices)

    left_x = left_linear.index_select(-1, plan.spatial_indices)
    right_x = right_linear.index_select(-1, plan.spatial_indices)
    square_lo = torch.minimum(spatial_lo * spatial_lo, spatial_hi * spatial_hi)
    square_hi = torch.maximum(spatial_lo * spatial_lo, spatial_hi * spatial_hi)
    square_lo = torch.where(
        (spatial_lo <= 0) & (spatial_hi >= 0), torch.zeros_like(square_lo), square_lo
    )
    diagonal_coefficient = left_x * right_x
    diagonal_lo = torch.minimum(
        diagonal_coefficient * square_lo[:, None, :],
        diagonal_coefficient * square_hi[:, None, :],
    ).sum(dim=-1)
    diagonal_hi = torch.maximum(
        diagonal_coefficient * square_lo[:, None, :],
        diagonal_coefficient * square_hi[:, None, :],
    ).sum(dim=-1)

    lo_i = spatial_lo[:, :, None]
    hi_i = spatial_hi[:, :, None]
    lo_j = spatial_lo[:, None, :]
    hi_j = spatial_hi[:, None, :]
    corners = torch.stack((lo_i * lo_j, lo_i * hi_j, hi_i * lo_j, hi_i * hi_j), dim=0)
    products_lo = corners.amin(dim=0) * plan.spatial_off_diagonal_mask
    products_hi = corners.amax(dim=0) * plan.spatial_off_diagonal_mask
    pair_coefficient = left_x[:, :, :, None] * right_x[:, :, None, :]
    off_lo = torch.minimum(
        pair_coefficient * products_lo[:, None, :, :],
        pair_coefficient * products_hi[:, None, :, :],
    ).sum(dim=(-2, -1))
    off_hi = torch.maximum(
        pair_coefficient * products_lo[:, None, :, :],
        pair_coefficient * products_hi[:, None, :, :],
    ).sum(dim=(-2, -1))
    pure_spatial_lo = diagonal_lo + off_lo
    pure_spatial_hi = diagonal_hi + off_hi

    left_linear_lo, left_linear_hi = _linear_form_interval(left_linear, box_lo, box_hi)
    right_linear_lo, right_linear_hi = _linear_form_interval(right_linear, box_lo, box_hi)
    left_cross_lo, left_cross_hi = _linear_form_interval(left_cross, box_lo, box_hi)
    right_cross_lo, right_cross_hi = _linear_form_interval(right_cross, box_lo, box_hi)
    time_lo = box_lo[:, plan.local_time_index : plan.local_time_index + 1].expand(
        -1, left.shape[1]
    )
    time_hi = box_hi[:, plan.local_time_index : plan.local_time_index + 1].expand(
        -1, left.shape[1]
    )
    linear_cross_lo, linear_cross_hi = _interval_mul(
        left_linear_lo, left_linear_hi, right_cross_lo, right_cross_hi
    )
    time_cubic_left_lo, time_cubic_left_hi = _interval_mul(
        time_lo, time_hi, linear_cross_lo, linear_cross_hi
    )
    cross_linear_lo, cross_linear_hi = _interval_mul(
        right_linear_lo, right_linear_hi, left_cross_lo, left_cross_hi
    )
    time_cubic_right_lo, time_cubic_right_hi = _interval_mul(
        time_lo, time_hi, cross_linear_lo, cross_linear_hi
    )
    time_cubic_lo, time_cubic_hi = _interval_add(
        time_cubic_left_lo,
        time_cubic_left_hi,
        time_cubic_right_lo,
        time_cubic_right_hi,
    )
    time_square_lo, time_square_hi = _interval_mul(time_lo, time_hi, time_lo, time_hi)
    cross_product_lo, cross_product_hi = _interval_mul(
        left_cross_lo, left_cross_hi, right_cross_lo, right_cross_hi
    )
    time_quartic_lo, time_quartic_hi = _interval_mul(
        time_square_lo,
        time_square_hi,
        cross_product_lo,
        cross_product_hi,
    )
    overflow_lo = pure_spatial_lo + time_cubic_lo + time_quartic_lo
    overflow_hi = pure_spatial_hi + time_cubic_hi + time_quartic_hi
    return kept, overflow_lo, overflow_hi


def _tm_mul(
    left_coeffs: torch.Tensor,
    left_rem_lo: torch.Tensor,
    left_rem_hi: torch.Tensor,
    right_coeffs: torch.Tensor,
    right_rem_lo: torch.Tensor,
    right_rem_hi: torch.Tensor,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kept, overflow_lo, overflow_hi = _polynomial_mul_ctrunc(
        left_coeffs, right_coeffs, box_lo, box_hi, plan
    )
    left_range_lo, left_range_hi = _polynomial_range(left_coeffs, box_lo, box_hi, plan)
    right_range_lo, right_range_hi = _polynomial_range(right_coeffs, box_lo, box_hi, plan)
    left_right_lo, left_right_hi = _interval_mul(
        left_range_lo, left_range_hi, right_rem_lo, right_rem_hi
    )
    right_left_lo, right_left_hi = _interval_mul(
        right_range_lo, right_range_hi, left_rem_lo, left_rem_hi
    )
    rem_rem_lo, rem_rem_hi = _interval_mul(
        left_rem_lo, left_rem_hi, right_rem_lo, right_rem_hi
    )
    remainder_lo = left_right_lo + right_left_lo + rem_rem_lo + overflow_lo
    remainder_hi = left_right_hi + right_left_hi + rem_rem_hi + overflow_hi
    return kept, remainder_lo, remainder_hi


def _polynomial_integrate(
    coefficients: torch.Tensor, plan: FixedSupportKernelPlan
) -> torch.Tensor:
    result = torch.zeros_like(coefficients)
    for route_index, (input_slot, output_slot) in enumerate(
        zip(plan.integration_input_indices, plan.integration_output_indices)
    ):
        contribution = coefficients[..., input_slot] * plan.integration_factor[route_index]
        result[..., output_slot] = result[..., output_slot] + contribution
    return result


def _tm_integrate(
    coefficients: torch.Tensor,
    rem_lo: torch.Tensor,
    rem_hi: torch.Tensor,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kept = _polynomial_integrate(coefficients, plan)
    _, _, cross = _groups(coefficients, plan)
    time_lo = box_lo[:, plan.local_time_index]
    time_hi = box_hi[:, plan.local_time_index]
    time_cube_lo = (time_lo**3)[:, None]
    time_cube_hi = (time_hi**3)[:, None]
    time_square_lo = (time_lo**2)[:, None, None]
    time_square_hi = (time_hi**2)[:, None, None]
    time_square_coefficient = cross[..., plan.local_time_index]
    cubic_lo = torch.minimum(
        time_square_coefficient * time_cube_lo,
        time_square_coefficient * time_cube_hi,
    ) / 3.0
    cubic_hi = torch.maximum(
        time_square_coefficient * time_cube_lo,
        time_square_coefficient * time_cube_hi,
    ) / 3.0

    spatial_lo = box_lo.index_select(-1, plan.spatial_indices)[:, None, :]
    spatial_hi = box_hi.index_select(-1, plan.spatial_indices)[:, None, :]
    corners = torch.stack(
        (
            time_square_lo * spatial_lo,
            time_square_lo * spatial_hi,
            time_square_hi * spatial_lo,
            time_square_hi * spatial_hi,
        ),
        dim=0,
    )
    time_square_x_lo = corners.amin(dim=0)
    time_square_x_hi = corners.amax(dim=0)
    spatial_cross = cross.index_select(-1, plan.spatial_indices)
    cross_lo = torch.minimum(
        spatial_cross * time_square_x_lo, spatial_cross * time_square_x_hi
    ).sum(dim=-1) * 0.5
    cross_hi = torch.maximum(
        spatial_cross * time_square_x_lo, spatial_cross * time_square_x_hi
    ).sum(dim=-1) * 0.5
    overflow_lo = cubic_lo + cross_lo
    overflow_hi = cubic_hi + cross_hi
    time_magnitude = torch.maximum(
        torch.abs(box_lo[:, plan.local_time_index : plan.local_time_index + 1]),
        torch.abs(box_hi[:, plan.local_time_index : plan.local_time_index + 1]),
    )
    integrated_rem_lo, integrated_rem_hi = _interval_scale(rem_lo, rem_hi, time_magnitude)
    return kept, overflow_lo + integrated_rem_lo, overflow_hi + integrated_rem_hi


def _vdp_polynomial_rhs(
    coefficients: torch.Tensor, plan: FixedSupportKernelPlan
) -> torch.Tensor:
    x = coefficients[:, 0:1, :]
    y = coefficients[:, 1:2, :]
    one = torch.zeros_like(x)
    one[..., plan.constant_slot] = 1.0
    x_square = _polynomial_mul_trunc(x, x, plan)
    second = _polynomial_mul_trunc(one - x_square, y, plan) - x
    return torch.cat((y, second), dim=1)


def _vdp_tm_rhs(
    coefficients: torch.Tensor,
    rem_lo: torch.Tensor,
    rem_hi: torch.Tensor,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_coeffs = coefficients[:, 0:1, :]
    y_coeffs = coefficients[:, 1:2, :]
    x_rem_lo = rem_lo[:, 0:1]
    x_rem_hi = rem_hi[:, 0:1]
    y_rem_lo = rem_lo[:, 1:2]
    y_rem_hi = rem_hi[:, 1:2]
    one = torch.zeros_like(x_coeffs)
    one[..., plan.constant_slot] = 1.0
    zero = torch.zeros_like(x_rem_lo)
    square_coeffs, square_lo, square_hi = _tm_mul(
        x_coeffs,
        x_rem_lo,
        x_rem_hi,
        x_coeffs,
        x_rem_lo,
        x_rem_hi,
        box_lo,
        box_hi,
        plan,
    )
    left_coeffs = one - square_coeffs
    left_lo, left_hi = _interval_sub(zero, zero, square_lo, square_hi)
    product_coeffs, product_lo, product_hi = _tm_mul(
        left_coeffs,
        left_lo,
        left_hi,
        y_coeffs,
        y_rem_lo,
        y_rem_hi,
        box_lo,
        box_hi,
        plan,
    )
    second_coeffs = product_coeffs - x_coeffs
    second_lo, second_hi = _interval_sub(
        product_lo, product_hi, x_rem_lo, x_rem_hi
    )
    return (
        torch.cat((y_coeffs, second_coeffs), dim=1),
        torch.cat((y_rem_lo, second_lo), dim=1),
        torch.cat((y_rem_hi, second_hi), dim=1),
    )


def _evaluate_time(
    coefficients: torch.Tensor,
    time_value: float,
    plan: FixedSupportKernelPlan,
) -> torch.Tensor:
    result = torch.zeros_like(coefficients)
    for input_slot in range(plan.num_slots):
        output_slot = plan.time_evaluate_output_indices[input_slot]
        power = plan.time_evaluate_power_integers[input_slot]
        result[..., output_slot] = (
            result[..., output_slot] + coefficients[..., input_slot] * (time_value**power)
        )
    return result


def _build_linear_coefficients(
    center: torch.Tensor,
    scale: torch.Tensor,
    plan: FixedSupportKernelPlan,
) -> torch.Tensor:
    coefficients = torch.zeros(
        (center.shape[0], center.shape[1], plan.num_slots),
        dtype=center.dtype,
        device=center.device,
    )
    coefficients[..., plan.constant_slot] = center
    coefficients[:, plan.state_indices, plan.spatial_linear_slots] = scale
    return coefficients


def _polynomial_picard(
    base: torch.Tensor,
    plan: FixedSupportKernelPlan,
    iterations: int,
) -> torch.Tensor:
    current = base
    for _ in range(iterations):
        current = base + _polynomial_integrate(_vdp_polynomial_rhs(current, plan), plan)
    return current


def _dr_picard(
    new_x0_coeffs: torch.Tensor,
    polynomial: torch.Tensor,
    initial_remainder: torch.Tensor,
    box_lo: torch.Tensor,
    box_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
    rounds: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    seed_lo = -initial_remainder
    seed_hi = initial_remainder
    initial_rhs_coeffs, initial_rhs_lo, initial_rhs_hi = _vdp_tm_rhs(
        polynomial, seed_lo, seed_hi, box_lo, box_hi, plan
    )
    initial_int_coeffs, initial_int_lo, initial_int_hi = _tm_integrate(
        initial_rhs_coeffs,
        initial_rhs_lo,
        initial_rhs_hi,
        box_lo,
        box_hi,
        plan,
    )
    initial_next_coeffs = new_x0_coeffs + initial_int_coeffs
    initial_mask = (initial_int_lo >= seed_lo) & (initial_int_hi <= seed_hi)
    roundoff_lo, roundoff_hi = _polynomial_range(
        initial_next_coeffs - polynomial, box_lo, box_hi, plan
    )

    current_coeffs = polynomial
    current_lo = seed_lo
    current_hi = seed_hi
    masks: list[torch.Tensor] = []
    for _ in range(rounds):
        rhs_coeffs, rhs_lo, rhs_hi = _vdp_tm_rhs(
            current_coeffs, current_lo, current_hi, box_lo, box_hi, plan
        )
        int_coeffs, int_lo, int_hi = _tm_integrate(
            rhs_coeffs, rhs_lo, rhs_hi, box_lo, box_hi, plan
        )
        candidate_coeffs = new_x0_coeffs + int_coeffs
        next_lo = int_lo + roundoff_lo
        next_hi = int_hi + roundoff_hi
        mask = (next_lo >= current_lo) & (next_hi <= current_hi)
        current_lo = torch.where(mask, next_lo, current_lo)
        current_hi = torch.where(mask, next_hi, current_hi)
        current_coeffs = candidate_coeffs
        masks.append(mask)
    return current_coeffs, current_lo, current_hi, initial_mask, torch.stack(masks, dim=0)


def _symbolic_step(
    parameter_coeffs: torch.Tensor,
    parameter_rem_lo: torch.Tensor,
    parameter_rem_hi: torch.Tensor,
    endpoint_coeffs: torch.Tensor,
    endpoint_rem_lo: torch.Tensor,
    endpoint_rem_hi: torch.Tensor,
    phi_buffer: torch.Tensor,
    j_lo: torch.Tensor,
    j_hi: torch.Tensor,
    queue_count: torch.Tensor,
    inverse_scale: torch.Tensor,
    slot_indices: torch.Tensor,
    eval_lo: torch.Tensor,
    eval_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
    epsilon: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    parameter_constant, parameter_linear_all, _ = _groups(parameter_coeffs, plan)
    del parameter_constant
    endpoint_constant, endpoint_linear_all, _ = _groups(endpoint_coeffs, plan)
    del endpoint_constant
    parameter_linear = parameter_linear_all.index_select(-1, plan.spatial_indices)
    endpoint_linear = endpoint_linear_all.index_select(-1, plan.spatial_indices)
    composed_linear = torch.einsum("bij,bjk->bik", endpoint_linear, parameter_linear)
    phi_new = endpoint_linear * inverse_scale[:, None, :]

    active = slot_indices[None, :] < queue_count[:, None]
    phi_updated = torch.einsum("bij,bmjk->bmik", phi_new, phi_buffer)
    # Explicit slicing avoids a Triton 3.x code-generation failure for the
    # negative wrapped load emitted by torch.roll on CUDA float64.
    phi_roll = torch.cat((phi_updated[:, 1:], phi_updated[:, :1]), dim=1)
    last_active = slot_indices[None, :] == (queue_count[:, None] - 1)
    phi_contributions = torch.where(
        last_active[:, :, None, None], phi_new[:, None, :, :], phi_roll
    )
    past_all_lo, past_all_hi = _interval_affine(j_lo, j_hi, phi_contributions)
    active_float = active.to(dtype=past_all_lo.dtype)[:, :, None]
    past_lo = (past_all_lo * active_float).sum(dim=1)
    past_hi = (past_all_hi * active_float).sum(dim=1)
    seed_lo, seed_hi = _interval_affine(
        parameter_rem_lo, parameter_rem_hi, endpoint_linear
    )
    candidate_j_lo = endpoint_rem_lo + seed_lo
    candidate_j_hi = endpoint_rem_hi + seed_hi
    empty = queue_count == 0
    current_j_lo = torch.where(empty[:, None], candidate_j_lo, endpoint_rem_lo)
    current_j_hi = torch.where(empty[:, None], candidate_j_hi, endpoint_rem_hi)
    next_rem_lo = past_lo + current_j_lo
    next_rem_hi = past_hi + current_j_hi

    next_coeffs = torch.zeros_like(parameter_coeffs)
    next_coeffs[:, plan.state_indices[:, None], plan.spatial_linear_slots[None, :]] = composed_linear
    range_lo, range_hi = _polynomial_range(next_coeffs, eval_lo, eval_hi, plan)
    range_lo = range_lo + next_rem_lo
    range_hi = range_hi + next_rem_hi
    scale = torch.maximum(torch.abs(range_hi), torch.abs(range_lo))
    next_inverse_scale = 1.0 / (scale + epsilon)
    next_coeffs = next_coeffs * next_inverse_scale[..., None]
    next_rem_lo, next_rem_hi = _interval_scale(
        next_rem_lo, next_rem_hi, next_inverse_scale
    )

    phi_buffer_next = torch.where(
        active[:, :, None, None], phi_updated, phi_buffer
    )
    insertion = slot_indices[None, :] == queue_count[:, None]
    phi_buffer_next = torch.where(
        insertion[:, :, None, None], phi_new[:, None, :, :], phi_buffer_next
    )
    j_lo_next = torch.where(insertion[:, :, None], current_j_lo[:, None, :], j_lo)
    j_hi_next = torch.where(insertion[:, :, None], current_j_hi[:, None, :], j_hi)
    queue_next = torch.minimum(
        queue_count + 1,
        torch.full_like(queue_count, slot_indices.shape[0]),
    )
    just_full = queue_next == slot_indices.shape[0]
    phi_buffer_next = torch.where(
        just_full[:, None, None, None], torch.zeros_like(phi_buffer_next), phi_buffer_next
    )
    j_lo_next = torch.where(just_full[:, None, None], torch.zeros_like(j_lo_next), j_lo_next)
    j_hi_next = torch.where(just_full[:, None, None], torch.zeros_like(j_hi_next), j_hi_next)
    queue_next = torch.where(just_full, torch.zeros_like(queue_next), queue_next)
    return (
        scale,
        next_coeffs,
        next_rem_lo,
        next_rem_hi,
        phi_buffer_next,
        j_lo_next,
        j_hi_next,
        queue_next,
        next_inverse_scale,
    )


def _compose_affine(
    coefficients: torch.Tensor,
    rem_lo: torch.Tensor,
    rem_hi: torch.Tensor,
    parameter_coeffs: torch.Tensor,
    parameter_rem_lo: torch.Tensor,
    parameter_rem_hi: torch.Tensor,
    step_size: float,
    plan: FixedSupportKernelPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    self_constant, self_linear, self_cross = _groups(coefficients, plan)
    other_constant, other_linear, _ = _groups(parameter_coeffs, plan)
    self_linear_spatial = self_linear.index_select(-1, plan.spatial_indices)
    self_cross_spatial = self_cross.index_select(-1, plan.spatial_indices)
    linear_new = torch.einsum(
        "bdv,bvw->bdw", self_linear_spatial, other_linear
    )
    linear_new[..., plan.local_time_index] = (
        linear_new[..., plan.local_time_index]
        + self_linear[..., plan.local_time_index]
    )
    constant_new = self_constant + (
        self_linear_spatial * other_constant[:, None, :]
    ).sum(dim=-1)
    cross_new = torch.einsum(
        "bdv,bvw->bdw", self_cross_spatial, other_linear
    )
    cross_new[..., plan.local_time_index] = (
        cross_new[..., plan.local_time_index]
        + self_cross[..., plan.local_time_index]
    )
    linear_new[..., plan.local_time_index] = (
        linear_new[..., plan.local_time_index]
        + (self_cross_spatial * other_constant[:, None, :]).sum(dim=-1)
    )
    output = torch.zeros_like(coefficients)
    output[..., plan.constant_slot] = constant_new
    output[..., plan.linear_slots] = linear_new
    output[..., plan.time_cross_slots] = cross_new
    linear_rem_lo, linear_rem_hi = _interval_affine(
        parameter_rem_lo,
        parameter_rem_hi,
        self_linear.index_select(-1, plan.spatial_indices),
    )
    cross_rem_lo, cross_rem_hi = _interval_affine(
        parameter_rem_lo,
        parameter_rem_hi,
        self_cross.index_select(-1, plan.spatial_indices),
    )
    cross_rem_lo, cross_rem_hi = _interval_scale(
        cross_rem_lo, cross_rem_hi, step_size
    )
    output_lo = rem_lo + linear_rem_lo + cross_rem_lo
    output_hi = rem_hi + linear_rem_hi + cross_rem_hi
    return output, output_lo, output_hi


def _select_batch(
    mask: torch.Tensor, new: torch.Tensor, old: torch.Tensor
) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < new.ndim:
        expanded = expanded.unsqueeze(-1)
    return torch.where(expanded, new, old)


def make_fixed_support_vdp_functional_step(
    plan: FixedSupportKernelPlan,
    *,
    step_lo: torch.Tensor,
    step_hi: torch.Tensor,
    eval_lo: torch.Tensor,
    eval_hi: torch.Tensor,
    initial_remainder: torch.Tensor,
    slot_indices: torch.Tensor,
    step_size: float,
    polynomial_picard_iterations: int = 2,
    remainder_rounds: int = 10,
    normalization_epsilon: float = 1e-12,
) -> Callable[[TensorState], TensorState]:
    """Create one tensor-only VDP logical step with immutable plan/constants."""

    if plan.overflow_policy != "diffreach_restricted_quadratic_grouped":
        raise NotImplementedError("optimized functional step requires grouped overflow")
    if plan.range_policy != "diffreach_restricted_quadratic_horner":
        raise NotImplementedError("optimized functional step requires restricted Horner range")

    def step(state: TensorState) -> TensorState:
        (
            model_coeffs,
            model_rem_lo,
            model_rem_hi,
            parameter_coeffs,
            parameter_rem_lo,
            parameter_rem_hi,
            phi_buffer,
            j_lo,
            j_hi,
            queue_count,
            inverse_scale,
            active_mask,
            first_failure_index,
            step_index,
            last_endpoint_lo,
            last_endpoint_hi,
            last_tube_lo,
            last_tube_hi,
            tube_hull_lo,
            tube_hull_hi,
            has_valid_tube,
            last_initial_mask,
            last_round_masks,
            all_initial_masks,
            all_round_masks,
            normalization_scale,
        ) = state
        endpoint_previous_coeffs = _evaluate_time(model_coeffs, step_size, plan)
        (
            scale,
            normalized_parameter_coeffs,
            normalized_parameter_rem_lo,
            normalized_parameter_rem_hi,
            phi_next,
            j_lo_next,
            j_hi_next,
            queue_count_next,
            inverse_scale_next,
        ) = _symbolic_step(
            parameter_coeffs,
            parameter_rem_lo,
            parameter_rem_hi,
            endpoint_previous_coeffs,
            model_rem_lo,
            model_rem_hi,
            phi_buffer,
            j_lo,
            j_hi,
            queue_count,
            inverse_scale,
            slot_indices,
            eval_lo,
            eval_hi,
            plan,
            normalization_epsilon,
        )
        center = endpoint_previous_coeffs[..., plan.constant_slot]
        new_x0_coeffs = _build_linear_coefficients(center, scale, plan)
        polynomial = _polynomial_picard(
            new_x0_coeffs, plan, polynomial_picard_iterations
        )
        (
            next_model_coeffs,
            next_model_rem_lo,
            next_model_rem_hi,
            initial_mask,
            round_masks,
        ) = _dr_picard(
            new_x0_coeffs,
            polynomial,
            initial_remainder,
            step_lo,
            step_hi,
            plan,
            remainder_rounds,
        )
        composed_coeffs, composed_rem_lo, composed_rem_hi = _compose_affine(
            next_model_coeffs,
            next_model_rem_lo,
            next_model_rem_hi,
            normalized_parameter_coeffs,
            normalized_parameter_rem_lo,
            normalized_parameter_rem_hi,
            step_size,
            plan,
        )
        endpoint_box_lo = step_lo.clone()
        endpoint_box_lo[:, plan.local_time_index] = step_size
        endpoint_poly_lo, endpoint_poly_hi = _polynomial_range(
            composed_coeffs, endpoint_box_lo, step_hi, plan
        )
        endpoint_lo = endpoint_poly_lo + composed_rem_lo
        endpoint_hi = endpoint_poly_hi + composed_rem_hi
        tube_poly_lo, tube_poly_hi = _polynomial_range(
            composed_coeffs, step_lo, step_hi, plan
        )
        tube_lo = tube_poly_lo + composed_rem_lo
        tube_hi = tube_poly_hi + composed_rem_hi

        batch_passed = initial_mask.all(dim=1)
        accepted = active_mask & batch_passed
        failed_now = active_mask & (~batch_passed)
        failure_value = step_index.expand_as(first_failure_index)
        first_failure_next = torch.where(
            failed_now & (first_failure_index < 0),
            failure_value,
            first_failure_index,
        )
        active_next = accepted
        next_model_coeffs = _select_batch(accepted, next_model_coeffs, model_coeffs)
        next_model_rem_lo = _select_batch(accepted, next_model_rem_lo, model_rem_lo)
        next_model_rem_hi = _select_batch(accepted, next_model_rem_hi, model_rem_hi)
        normalized_parameter_coeffs = _select_batch(
            accepted, normalized_parameter_coeffs, parameter_coeffs
        )
        normalized_parameter_rem_lo = _select_batch(
            accepted, normalized_parameter_rem_lo, parameter_rem_lo
        )
        normalized_parameter_rem_hi = _select_batch(
            accepted, normalized_parameter_rem_hi, parameter_rem_hi
        )
        phi_next = _select_batch(accepted, phi_next, phi_buffer)
        j_lo_next = _select_batch(accepted, j_lo_next, j_lo)
        j_hi_next = _select_batch(accepted, j_hi_next, j_hi)
        queue_count_next = _select_batch(accepted, queue_count_next, queue_count)
        inverse_scale_next = _select_batch(accepted, inverse_scale_next, inverse_scale)
        endpoint_lo_next = _select_batch(accepted, endpoint_lo, last_endpoint_lo)
        endpoint_hi_next = _select_batch(accepted, endpoint_hi, last_endpoint_hi)
        tube_lo_next = _select_batch(accepted, tube_lo, last_tube_lo)
        tube_hi_next = _select_batch(accepted, tube_hi, last_tube_hi)
        hull_lo_candidate = torch.where(
            has_valid_tube[:, None], torch.minimum(tube_hull_lo, tube_lo), tube_lo
        )
        hull_hi_candidate = torch.where(
            has_valid_tube[:, None], torch.maximum(tube_hull_hi, tube_hi), tube_hi
        )
        tube_hull_lo_next = _select_batch(accepted, hull_lo_candidate, tube_hull_lo)
        tube_hull_hi_next = _select_batch(accepted, hull_hi_candidate, tube_hull_hi)
        has_valid_tube_next = has_valid_tube | accepted
        last_initial_mask_next = _select_batch(
            active_mask, initial_mask, last_initial_mask
        )
        round_masks_batch_first = round_masks.permute(1, 0, 2)
        last_round_batch_first = last_round_masks.permute(1, 0, 2)
        last_round_next = _select_batch(
            active_mask, round_masks_batch_first, last_round_batch_first
        ).permute(1, 0, 2)
        all_initial_next = torch.where(
            active_mask[:, None], all_initial_masks & initial_mask, all_initial_masks
        )
        active_round = active_mask[None, :, None]
        all_round_next = torch.where(
            active_round, all_round_masks & round_masks, all_round_masks
        )
        normalization_scale_next = _select_batch(
            accepted, scale, normalization_scale
        )
        return (
            next_model_coeffs,
            next_model_rem_lo,
            next_model_rem_hi,
            normalized_parameter_coeffs,
            normalized_parameter_rem_lo,
            normalized_parameter_rem_hi,
            phi_next,
            j_lo_next,
            j_hi_next,
            queue_count_next,
            inverse_scale_next,
            active_next,
            first_failure_next,
            step_index + 1,
            endpoint_lo_next,
            endpoint_hi_next,
            tube_lo_next,
            tube_hi_next,
            tube_hull_lo_next,
            tube_hull_hi_next,
            has_valid_tube_next,
            last_initial_mask_next,
            last_round_next,
            all_initial_next,
            all_round_next,
            normalization_scale_next,
        )

    return step


def initialize_fixed_support_functional_state(
    initial_lo: torch.Tensor,
    initial_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
    *,
    queue_capacity: int,
    remainder_rounds: int = 10,
) -> tuple[FixedSupportFunctionalState, torch.Tensor]:
    if initial_lo.shape != initial_hi.shape or initial_lo.ndim != 2:
        raise ValueError("initial bounds must have equal [batch,state] shape")
    if initial_lo.shape[1] != plan.support_dim - 1:
        raise ValueError("initial state dimension does not match kernel plan")
    if queue_capacity <= 0:
        raise ValueError("queue capacity must be positive")
    center = 0.5 * (initial_lo + initial_hi)
    scale = 0.5 * (initial_hi - initial_lo)
    model_coeffs = _build_linear_coefficients(center, scale, plan)
    parameter_coeffs = _build_linear_coefficients(
        torch.zeros_like(center), torch.ones_like(scale), plan
    )
    zero_rem = torch.zeros_like(center)
    phi = torch.zeros(
        (initial_lo.shape[0], queue_capacity, center.shape[1], center.shape[1]),
        dtype=initial_lo.dtype,
        device=initial_lo.device,
    )
    j_lo = torch.zeros(
        (initial_lo.shape[0], queue_capacity, center.shape[1]),
        dtype=initial_lo.dtype,
        device=initial_lo.device,
    )
    j_hi = torch.zeros_like(j_lo)
    queue_count = torch.zeros(
        initial_lo.shape[0], dtype=torch.long, device=initial_lo.device
    )
    slot_indices = torch.arange(
        queue_capacity, dtype=torch.long, device=initial_lo.device
    )
    state = FixedSupportFunctionalState(
        model_coeffs=model_coeffs,
        model_rem_lo=zero_rem,
        model_rem_hi=zero_rem,
        parameter_coeffs=parameter_coeffs,
        parameter_rem_lo=zero_rem,
        parameter_rem_hi=zero_rem,
        phi_buffer=phi,
        j_lo=j_lo,
        j_hi=j_hi,
        queue_count=queue_count,
        inverse_scale=torch.ones_like(center),
        active_mask=torch.ones(
            initial_lo.shape[0], dtype=torch.bool, device=initial_lo.device
        ),
        first_failure_index=torch.full(
            (initial_lo.shape[0],), -1, dtype=torch.long, device=initial_lo.device
        ),
        step_index=torch.zeros((), dtype=torch.long, device=initial_lo.device),
        last_endpoint_lo=initial_lo,
        last_endpoint_hi=initial_hi,
        last_tube_lo=initial_lo,
        last_tube_hi=initial_hi,
        tube_hull_lo=torch.full_like(initial_lo, torch.inf),
        tube_hull_hi=torch.full_like(initial_hi, -torch.inf),
        has_valid_tube=torch.zeros(
            initial_lo.shape[0], dtype=torch.bool, device=initial_lo.device
        ),
        last_initial_mask=torch.ones_like(initial_lo, dtype=torch.bool),
        last_round_masks=torch.ones(
            (remainder_rounds, *initial_lo.shape),
            dtype=torch.bool,
            device=initial_lo.device,
        ),
        all_initial_masks=torch.ones_like(initial_lo, dtype=torch.bool),
        all_round_masks=torch.ones(
            (remainder_rounds, *initial_lo.shape),
            dtype=torch.bool,
            device=initial_lo.device,
        ),
        normalization_scale=scale,
    )
    return state, slot_indices


def prepare_fixed_support_vdp_functional_step(
    initial_lo: torch.Tensor,
    initial_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
    *,
    step_size: float,
    steps: int,
    initial_remainder_radius: float = 0.01,
    polynomial_picard_iterations: int = 2,
    remainder_rounds: int = 10,
    symbolic_window_size: int = 1000,
    normalization_epsilon: float = 1e-12,
) -> tuple[FixedSupportFunctionalState, Callable[[TensorState], TensorState]]:
    """Prepare tensor state and a matching logical-step callable."""

    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    queue_capacity = min(int(symbolic_window_size), max(1, steps))
    state, slot_indices = initialize_fixed_support_functional_state(
        initial_lo,
        initial_hi,
        plan,
        queue_capacity=queue_capacity,
        remainder_rounds=remainder_rounds,
    )
    batch, state_dim = initial_lo.shape
    zeros = torch.zeros((batch, 1), dtype=initial_lo.dtype, device=initial_lo.device)
    step_time = torch.full(
        (batch, 1), float(step_size), dtype=initial_lo.dtype, device=initial_lo.device
    )
    ones = torch.ones((batch, state_dim), dtype=initial_lo.dtype, device=initial_lo.device)
    step_lo = torch.cat((zeros, -ones), dim=1)
    step_hi = torch.cat((step_time, ones), dim=1)
    eval_lo = torch.cat((zeros, -ones), dim=1)
    eval_hi = torch.cat((zeros, ones), dim=1)
    initial_remainder = torch.full(
        (batch, state_dim),
        float(initial_remainder_radius),
        dtype=initial_lo.dtype,
        device=initial_lo.device,
    )
    step = make_fixed_support_vdp_functional_step(
        plan,
        step_lo=step_lo,
        step_hi=step_hi,
        eval_lo=eval_lo,
        eval_hi=eval_hi,
        initial_remainder=initial_remainder,
        slot_indices=slot_indices,
        step_size=step_size,
        polynomial_picard_iterations=polynomial_picard_iterations,
        remainder_rounds=remainder_rounds,
        normalization_epsilon=normalization_epsilon,
    )
    return state, step


def make_fixed_support_functional_chunk(
    step_callable: Callable[[TensorState], TensorState],
    *,
    chunk_size: int,
) -> Callable[[TensorState], TensorState]:
    """Create a bounded static chunk suitable for ``torch.compile``."""

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    def chunk(state: TensorState) -> TensorState:
        for _ in range(chunk_size):
            state = step_callable(state)
        return state

    return chunk


def fixed_support_functional_verify(
    initial_lo: torch.Tensor,
    initial_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
    *,
    step_size: float,
    steps: int,
    initial_remainder_radius: float = 0.01,
    polynomial_picard_iterations: int = 2,
    remainder_rounds: int = 10,
    symbolic_window_size: int = 1000,
    normalization_epsilon: float = 1e-12,
    trace: bool = False,
    step_callable: Callable[[TensorState], TensorState] | None = None,
) -> FixedSupportFunctionalResult:
    steps = int(steps)
    if step_callable is None:
        state_object, step_callable = prepare_fixed_support_vdp_functional_step(
            initial_lo,
            initial_hi,
            plan,
            step_size=step_size,
            steps=steps,
            initial_remainder_radius=initial_remainder_radius,
            polynomial_picard_iterations=polynomial_picard_iterations,
            remainder_rounds=remainder_rounds,
            symbolic_window_size=symbolic_window_size,
            normalization_epsilon=normalization_epsilon,
        )
    else:
        state_object, _ = prepare_fixed_support_vdp_functional_step(
            initial_lo,
            initial_hi,
            plan,
            step_size=step_size,
            steps=steps,
            initial_remainder_radius=initial_remainder_radius,
            polynomial_picard_iterations=polynomial_picard_iterations,
            remainder_rounds=remainder_rounds,
            symbolic_window_size=symbolic_window_size,
            normalization_epsilon=normalization_epsilon,
        )
    state = state_object.tensors()
    endpoint_los = [initial_lo] if trace else None
    endpoint_his = [initial_hi] if trace else None
    tube_los: list[torch.Tensor] | None = [] if trace else None
    tube_his: list[torch.Tensor] | None = [] if trace else None
    initial_masks: list[torch.Tensor] | None = [] if trace else None
    round_masks: list[torch.Tensor] | None = [] if trace else None
    for _ in range(steps):
        state = step_callable(state)
        if trace:
            current = FixedSupportFunctionalState.from_tensors(state)
            assert endpoint_los is not None and endpoint_his is not None
            assert tube_los is not None and tube_his is not None
            assert initial_masks is not None and round_masks is not None
            endpoint_los.append(current.last_endpoint_lo)
            endpoint_his.append(current.last_endpoint_hi)
            tube_los.append(current.last_tube_lo)
            tube_his.append(current.last_tube_hi)
            initial_masks.append(current.last_initial_mask)
            round_masks.append(current.last_round_masks)
    final_state = FixedSupportFunctionalState.from_tensors(state)
    completed_mask = final_state.first_failure_index < 0
    validated_steps_per_batch = torch.where(
        completed_mask,
        final_state.step_index.expand_as(final_state.first_failure_index),
        final_state.first_failure_index,
    )
    failure_cpu = final_state.first_failure_index.detach().cpu()
    failures = [int(value) for value in failure_cpu.tolist() if int(value) >= 0]
    first_failure_step = min(failures) if failures else None
    validated_steps = steps if first_failure_step is None else first_failure_step
    return FixedSupportFunctionalResult(
        final_state=final_state,
        endpoint_lo=torch.stack(endpoint_los, dim=1) if endpoint_los is not None else None,
        endpoint_hi=torch.stack(endpoint_his, dim=1) if endpoint_his is not None else None,
        tube_lo=torch.stack(tube_los, dim=1) if tube_los is not None else None,
        tube_hi=torch.stack(tube_his, dim=1) if tube_his is not None else None,
        initial_inclusion_masks=(
            torch.stack(initial_masks, dim=0) if initial_masks is not None else None
        ),
        round_inclusion_masks=(
            torch.stack(round_masks, dim=0) if round_masks is not None else None
        ),
        validated_steps_per_batch=validated_steps_per_batch,
        completed_mask=completed_mask,
        first_failure_step=first_failure_step,
        validated_steps=validated_steps,
        requested_steps=steps,
        host_synchronizations=1,
        device_transfers=0,
        trace_mode=trace,
    )


__all__ = [
    "FixedSupportFunctionalResult",
    "FixedSupportFunctionalState",
    "TensorState",
    "fixed_support_functional_verify",
    "initialize_fixed_support_functional_state",
    "make_fixed_support_functional_chunk",
    "make_fixed_support_vdp_functional_step",
    "prepare_fixed_support_vdp_functional_step",
]
