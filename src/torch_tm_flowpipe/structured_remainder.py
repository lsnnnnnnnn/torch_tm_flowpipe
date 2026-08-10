"""Bounded structured-remainder candidate S1.

The primitive is state-dimension and system agnostic.  It consumes an additive
typed source ledger, propagates old columns through a safeguarded interval
linear map, materializes nonlinear residuals, and evicts the oldest column
without losing its set contribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .fixed_support_outward import (
    OutwardIntervalTensor,
    outward_matmul,
    outward_sum,
)


STRUCTURED_REMAINDER_CAPACITY = 16
STRUCTURED_REMAINDER_CANDIDATE = "normalized_insertion_structured_remainder_k16"
ELIGIBLE_STRUCTURED_SOURCES = (
    "polynomial_truncation",
    "integration_overflow",
)
STRUCTURED_SOURCE_IDS = {
    "polynomial_truncation": 1,
    "integration_overflow": 2,
}


@dataclass(frozen=True)
class StructuredRemainderState:
    ordinary_rem_lo: torch.Tensor
    ordinary_rem_hi: torch.Tensor
    j_lo: torch.Tensor
    j_hi: torch.Tensor
    phi_lo: torch.Tensor
    phi_hi: torch.Tensor
    active: torch.Tensor
    age: torch.Tensor
    source_id: torch.Tensor
    inverse_scale: torch.Tensor

    @property
    def batch(self) -> int:
        return self.ordinary_rem_lo.shape[0]

    @property
    def state_dim(self) -> int:
        return self.ordinary_rem_lo.shape[1]

    @property
    def capacity(self) -> int:
        return self.j_lo.shape[1]


@dataclass(frozen=True)
class StructuredRemainderBoundaryResult:
    state: StructuredRemainderState
    materialized_lo: torch.Tensor
    materialized_hi: torch.Tensor
    propagated_symbolic_lo: torch.Tensor
    propagated_symbolic_hi: torch.Tensor
    new_symbolic_lo: torch.Tensor
    new_symbolic_hi: torch.Tensor
    nonlinear_residual_lo: torch.Tensor
    nonlinear_residual_hi: torch.Tensor
    evicted_materialized_lo: torch.Tensor
    evicted_materialized_hi: torch.Tensor
    decomposition_padding_lo: torch.Tensor
    decomposition_padding_hi: torch.Tensor
    pre_split_lo: torch.Tensor
    pre_split_hi: torch.Tensor
    accepted: torch.Tensor
    conservation_mask: torch.Tensor
    source_decomposition_mask: torch.Tensor
    failure_reason: str
    evicted_source_id: torch.Tensor
    evicted_age: torch.Tensor


def initialize_structured_remainder_state(
    batch: int,
    state_dim: int,
    *,
    capacity: int = STRUCTURED_REMAINDER_CAPACITY,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> StructuredRemainderState:
    if batch <= 0 or state_dim <= 0 or capacity <= 0:
        raise ValueError("batch, state_dim, and capacity must be positive")
    if dtype != torch.float64:
        raise ValueError("S1 is qualified only for float64")
    device_t = torch.device(device)
    ordinary = torch.zeros((batch, state_dim), dtype=dtype, device=device_t)
    j = torch.zeros((batch, capacity, state_dim), dtype=dtype, device=device_t)
    phi = torch.zeros(
        (batch, capacity, state_dim, state_dim), dtype=dtype, device=device_t
    )
    return StructuredRemainderState(
        ordinary_rem_lo=ordinary,
        ordinary_rem_hi=ordinary,
        j_lo=j,
        j_hi=j,
        phi_lo=phi,
        phi_hi=phi,
        active=torch.zeros((batch, capacity), dtype=torch.bool, device=device_t),
        age=torch.full((batch, capacity), -1, dtype=torch.long, device=device_t),
        source_id=torch.zeros((batch, capacity), dtype=torch.long, device=device_t),
        inverse_scale=torch.ones((batch, state_dim), dtype=dtype, device=device_t),
    )


def _zero(state: StructuredRemainderState) -> OutwardIntervalTensor:
    return OutwardIntervalTensor.zeros_like(state.ordinary_rem_lo)


def _masked_columns(
    interval: OutwardIntervalTensor, active: torch.Tensor
) -> OutwardIntervalTensor:
    mask = active
    while mask.ndim < interval.lo.ndim:
        mask = mask.unsqueeze(-1)
    zero = torch.zeros_like(interval.lo)
    return OutwardIntervalTensor(
        torch.where(mask, interval.lo, zero),
        torch.where(mask, interval.hi, zero),
    )


def structured_column_contributions(
    state: StructuredRemainderState,
) -> OutwardIntervalTensor:
    phi = OutwardIntervalTensor(state.phi_lo, state.phi_hi)
    column = OutwardIntervalTensor(state.j_lo[..., None], state.j_hi[..., None])
    contribution = outward_matmul(phi, column)
    return _masked_columns(
        OutwardIntervalTensor(contribution.lo[..., 0], contribution.hi[..., 0]),
        state.active,
    )


def materialize_structured_remainder(
    state: StructuredRemainderState,
) -> OutwardIntervalTensor:
    columns = structured_column_contributions(state)
    symbolic = columns.sum(dim=1)
    ordinary = OutwardIntervalTensor(
        state.ordinary_rem_lo, state.ordinary_rem_hi
    )
    return ordinary.add(symbolic)


def _ledger_total(
    ledger: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    like: torch.Tensor,
) -> OutwardIntervalTensor:
    total = OutwardIntervalTensor.zeros_like(like)
    for name in sorted(ledger):
        lo, hi = ledger[name]
        total = total.add(OutwardIntervalTensor(lo, hi))
    return total


def _affine_map_interval(
    matrix: OutwardIntervalTensor,
    value: OutwardIntervalTensor,
) -> OutwardIntervalTensor:
    return OutwardIntervalTensor(
        outward_matmul(
            matrix,
            OutwardIntervalTensor(value.lo[..., None], value.hi[..., None]),
        ).lo[..., 0],
        outward_matmul(
            matrix,
            OutwardIntervalTensor(value.lo[..., None], value.hi[..., None]),
        ).hi[..., 0],
    )


def _center_source(
    source: OutwardIntervalTensor,
) -> tuple[OutwardIntervalTensor, OutwardIntervalTensor]:
    center_value = 0.5 * (source.lo + source.hi)
    center = OutwardIntervalTensor.point(center_value)
    left_radius = torch.nextafter(
        center_value - source.lo, torch.full_like(source.lo, torch.inf)
    )
    right_radius = torch.nextafter(
        source.hi - center_value, torch.full_like(source.hi, torch.inf)
    )
    radius = torch.maximum(left_radius, right_radius)
    symmetric = OutwardIntervalTensor(-radius, radius)
    return center, symmetric


def _failure_result(
    state: StructuredRemainderState,
    reason: str,
) -> StructuredRemainderBoundaryResult:
    materialized = materialize_structured_remainder(state)
    zero = torch.zeros_like(state.ordinary_rem_lo)
    false = torch.zeros(state.batch, dtype=torch.bool, device=zero.device)
    minus_one = torch.full(
        (state.batch,), -1, dtype=torch.long, device=zero.device
    )
    return StructuredRemainderBoundaryResult(
        state=state,
        materialized_lo=materialized.lo,
        materialized_hi=materialized.hi,
        propagated_symbolic_lo=zero,
        propagated_symbolic_hi=zero,
        new_symbolic_lo=zero,
        new_symbolic_hi=zero,
        nonlinear_residual_lo=zero,
        nonlinear_residual_hi=zero,
        evicted_materialized_lo=zero,
        evicted_materialized_hi=zero,
        decomposition_padding_lo=zero,
        decomposition_padding_hi=zero,
        pre_split_lo=materialized.lo,
        pre_split_hi=materialized.hi,
        accepted=false,
        conservation_mask=false,
        source_decomposition_mask=false,
        failure_reason=reason,
        evicted_source_id=minus_one,
        evicted_age=minus_one,
    )


def structured_remainder_boundary_update(
    state: StructuredRemainderState,
    *,
    typed_sources: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    validated_remainder_lo: torch.Tensor,
    validated_remainder_hi: torch.Tensor,
    linear_map_lo: torch.Tensor,
    linear_map_hi: torch.Tensor,
    nonlinear_residual_lo: torch.Tensor | None,
    nonlinear_residual_hi: torch.Tensor | None,
    normalization_scale: torch.Tensor,
    boundary_index: int,
    map_is_affine: bool = False,
) -> StructuredRemainderBoundaryResult:
    """Apply one S1 boundary with frozen K16 oldest-first eviction."""

    expected_vector = (state.batch, state.state_dim)
    expected_matrix = (state.batch, state.state_dim, state.state_dim)
    if (
        validated_remainder_lo.shape != expected_vector
        or validated_remainder_hi.shape != expected_vector
        or linear_map_lo.shape != expected_matrix
        or linear_map_hi.shape != expected_matrix
        or normalization_scale.shape != expected_vector
    ):
        return _failure_result(state, "dimension_mismatch")
    if state.capacity != STRUCTURED_REMAINDER_CAPACITY and state.capacity != 1:
        return _failure_result(state, "capacity_not_frozen_k16_or_unit_test_k1")
    if any(name not in ELIGIBLE_STRUCTURED_SOURCES and name == "" for name in typed_sources):
        return _failure_result(state, "invalid_source_name")
    if not map_is_affine and (
        nonlinear_residual_lo is None or nonlinear_residual_hi is None
    ):
        return _failure_result(state, "missing_structured_nonlinear_residual")
    if nonlinear_residual_lo is None or nonlinear_residual_hi is None:
        nonlinear_residual_lo = torch.zeros_like(validated_remainder_lo)
        nonlinear_residual_hi = torch.zeros_like(validated_remainder_hi)
    if (
        nonlinear_residual_lo.shape != expected_vector
        or nonlinear_residual_hi.shape != expected_vector
    ):
        return _failure_result(state, "nonlinear_residual_dimension_mismatch")
    tensors = (
        validated_remainder_lo,
        validated_remainder_hi,
        linear_map_lo,
        linear_map_hi,
        nonlinear_residual_lo,
        nonlinear_residual_hi,
        normalization_scale,
    )
    if any(tensor.dtype != torch.float64 for tensor in tensors):
        return _failure_result(state, "dtype_mismatch")
    if any(not bool(torch.all(torch.isfinite(tensor))) for tensor in tensors):
        return _failure_result(state, "nonfinite_input")
    if (
        not bool(torch.all(validated_remainder_lo <= validated_remainder_hi))
        or not bool(torch.all(linear_map_lo <= linear_map_hi))
        or not bool(torch.all(nonlinear_residual_lo <= nonlinear_residual_hi))
        or not bool(torch.all(normalization_scale > 0))
    ):
        return _failure_result(state, "invalid_interval_or_scale")
    for name, (lo, hi) in typed_sources.items():
        if lo.shape != expected_vector or hi.shape != expected_vector:
            return _failure_result(state, f"source_dimension_mismatch:{name}")
        if not bool(torch.all(torch.isfinite(lo)) and torch.all(torch.isfinite(hi))):
            return _failure_result(state, f"nonfinite_source:{name}")
        if not bool(torch.all(lo <= hi)):
            return _failure_result(state, f"invalid_source_interval:{name}")

    linear_map = OutwardIntervalTensor(linear_map_lo, linear_map_hi)
    old_materialized = materialize_structured_remainder(state)
    propagated_pre_split = _affine_map_interval(linear_map, old_materialized)
    ledger_total = _ledger_total(typed_sources, validated_remainder_lo)
    nonlinear = OutwardIntervalTensor(
        nonlinear_residual_lo, nonlinear_residual_hi
    )
    pre_split = outward_sum((propagated_pre_split, ledger_total, nonlinear))
    validated = OutwardIntervalTensor(
        validated_remainder_lo, validated_remainder_hi
    )
    source_decomposition_mask = (
        (pre_split.lo <= validated.lo) & (pre_split.hi >= validated.hi)
    ).all(dim=1)

    ordinary_old = _affine_map_interval(
        linear_map,
        OutwardIntervalTensor(state.ordinary_rem_lo, state.ordinary_rem_hi),
    )
    ordinary = ordinary_old.add(nonlinear)
    for name in sorted(typed_sources):
        if name in ELIGIBLE_STRUCTURED_SOURCES:
            continue
        ordinary = ordinary.add(OutwardIntervalTensor(*typed_sources[name]))

    propagated_phi = outward_matmul(
        OutwardIntervalTensor(
            linear_map.lo[:, None, :, :], linear_map.hi[:, None, :, :]
        ),
        OutwardIntervalTensor(state.phi_lo, state.phi_hi),
    )
    next_state = StructuredRemainderState(
        ordinary_rem_lo=ordinary.lo,
        ordinary_rem_hi=ordinary.hi,
        j_lo=state.j_lo,
        j_hi=state.j_hi,
        phi_lo=propagated_phi.lo,
        phi_hi=propagated_phi.hi,
        active=state.active,
        age=state.age,
        source_id=state.source_id,
        inverse_scale=1.0 / normalization_scale,
    )
    propagated_columns = structured_column_contributions(next_state).sum(dim=1)
    new_symbolic = _zero(state)
    evicted_total = _zero(state)
    evicted_source_id = torch.full(
        (state.batch,), -1, dtype=torch.long, device=state.active.device
    )
    evicted_age = torch.full_like(evicted_source_id, -1)
    identity = torch.eye(
        state.state_dim,
        dtype=state.ordinary_rem_lo.dtype,
        device=state.ordinary_rem_lo.device,
    ).expand(state.batch, -1, -1)

    for name in ELIGIBLE_STRUCTURED_SOURCES:
        if name not in typed_sources:
            continue
        center, symmetric = _center_source(
            OutwardIntervalTensor(*typed_sources[name])
        )
        ordinary = OutwardIntervalTensor(
            next_state.ordinary_rem_lo, next_state.ordinary_rem_hi
        ).add(center)
        next_state = StructuredRemainderState(
            ordinary.lo,
            ordinary.hi,
            next_state.j_lo,
            next_state.j_hi,
            next_state.phi_lo,
            next_state.phi_hi,
            next_state.active,
            next_state.age,
            next_state.source_id,
            next_state.inverse_scale,
        )
        full = next_state.active.all(dim=1)
        first_inactive = torch.argmax((~next_state.active).to(torch.int64), dim=1)
        largest_age = torch.iinfo(torch.long).max
        oldest = torch.argmin(
            torch.where(
                next_state.active,
                next_state.age,
                torch.full_like(next_state.age, largest_age),
            ),
            dim=1,
        )
        slot = torch.where(full, oldest, first_inactive)
        one_hot = torch.nn.functional.one_hot(
            slot, num_classes=state.capacity
        ).to(torch.bool)
        columns = structured_column_contributions(next_state)
        selected = OutwardIntervalTensor(
            (columns.lo * one_hot[..., None]).sum(dim=1),
            (columns.hi * one_hot[..., None]).sum(dim=1),
        )
        evicted = OutwardIntervalTensor(
            torch.where(full[:, None], selected.lo, torch.zeros_like(selected.lo)),
            torch.where(full[:, None], selected.hi, torch.zeros_like(selected.hi)),
        )
        ordinary_after_eviction = OutwardIntervalTensor(
            next_state.ordinary_rem_lo, next_state.ordinary_rem_hi
        ).add(evicted)
        evicted_total = evicted_total.add(evicted)
        selected_source = torch.sum(
            next_state.source_id * one_hot.to(next_state.source_id.dtype), dim=1
        )
        selected_age = torch.sum(
            next_state.age * one_hot.to(next_state.age.dtype), dim=1
        )
        evicted_source_id = torch.where(full, selected_source, evicted_source_id)
        evicted_age = torch.where(full, selected_age, evicted_age)
        mask_vector = one_hot[..., None]
        mask_matrix = one_hot[..., None, None]
        next_state = StructuredRemainderState(
            ordinary_after_eviction.lo,
            ordinary_after_eviction.hi,
            torch.where(mask_vector, symmetric.lo[:, None, :], next_state.j_lo),
            torch.where(mask_vector, symmetric.hi[:, None, :], next_state.j_hi),
            torch.where(mask_matrix, identity[:, None, :, :], next_state.phi_lo),
            torch.where(mask_matrix, identity[:, None, :, :], next_state.phi_hi),
            next_state.active | one_hot,
            torch.where(
                one_hot,
                torch.full_like(next_state.age, int(boundary_index)),
                next_state.age,
            ),
            torch.where(
                one_hot,
                torch.full_like(next_state.source_id, STRUCTURED_SOURCE_IDS[name]),
                next_state.source_id,
            ),
            next_state.inverse_scale,
        )
        new_symbolic = new_symbolic.add(symmetric)

    materialized = materialize_structured_remainder(next_state)
    padding_lo = torch.minimum(
        pre_split.lo - materialized.lo, torch.zeros_like(materialized.lo)
    )
    padding_hi = torch.maximum(
        pre_split.hi - materialized.hi, torch.zeros_like(materialized.hi)
    )
    padding = OutwardIntervalTensor(
        torch.nextafter(padding_lo, torch.full_like(padding_lo, -torch.inf)),
        torch.nextafter(padding_hi, torch.full_like(padding_hi, torch.inf)),
    ).sanitized()
    padded_ordinary = OutwardIntervalTensor(
        next_state.ordinary_rem_lo, next_state.ordinary_rem_hi
    ).add(padding)
    next_state = StructuredRemainderState(
        padded_ordinary.lo,
        padded_ordinary.hi,
        next_state.j_lo,
        next_state.j_hi,
        next_state.phi_lo,
        next_state.phi_hi,
        next_state.active,
        next_state.age,
        next_state.source_id,
        next_state.inverse_scale,
    )
    materialized = materialize_structured_remainder(next_state)
    conservation_mask = (
        (materialized.lo <= pre_split.lo) & (materialized.hi >= pre_split.hi)
    ).all(dim=1)
    finite = (
        torch.isfinite(materialized.lo)
        & torch.isfinite(materialized.hi)
        & (materialized.lo <= materialized.hi)
    ).all(dim=1)
    accepted = source_decomposition_mask & conservation_mask & finite
    failure_reason = "" if bool(torch.all(accepted)) else "conservation_or_source_decomposition_failed"
    return StructuredRemainderBoundaryResult(
        state=next_state,
        materialized_lo=materialized.lo,
        materialized_hi=materialized.hi,
        propagated_symbolic_lo=propagated_columns.lo,
        propagated_symbolic_hi=propagated_columns.hi,
        new_symbolic_lo=new_symbolic.lo,
        new_symbolic_hi=new_symbolic.hi,
        nonlinear_residual_lo=nonlinear.lo,
        nonlinear_residual_hi=nonlinear.hi,
        evicted_materialized_lo=evicted_total.lo,
        evicted_materialized_hi=evicted_total.hi,
        decomposition_padding_lo=padding.lo,
        decomposition_padding_hi=padding.hi,
        pre_split_lo=pre_split.lo,
        pre_split_hi=pre_split.hi,
        accepted=accepted,
        conservation_mask=conservation_mask,
        source_decomposition_mask=source_decomposition_mask,
        failure_reason=failure_reason,
        evicted_source_id=evicted_source_id,
        evicted_age=evicted_age,
    )


def structured_quadratic_nonlinear_residual(
    quadratic_coefficients: torch.Tensor,
    perturbation_lo: torch.Tensor,
    perturbation_hi: torch.Tensor,
) -> OutwardIntervalTensor:
    """Bound ``sum_ij Q[o,i,j] z_i z_j`` by safeguarded intervals."""

    if quadratic_coefficients.ndim != 4:
        raise ValueError("quadratic coefficients must have shape [B,O,S,S]")
    if perturbation_lo.shape != perturbation_hi.shape:
        raise ValueError("perturbation endpoints must agree")
    z = OutwardIntervalTensor(perturbation_lo, perturbation_hi)
    terms: list[OutwardIntervalTensor] = []
    for left in range(perturbation_lo.shape[1]):
        for right in range(perturbation_lo.shape[1]):
            product = OutwardIntervalTensor(
                z.lo[:, left], z.hi[:, left]
            ).mul(OutwardIntervalTensor(z.lo[:, right], z.hi[:, right]))
            coefficient = OutwardIntervalTensor.point(
                quadratic_coefficients[:, :, left, right]
            )
            terms.append(
                coefficient.mul(
                    OutwardIntervalTensor(product.lo[:, None], product.hi[:, None])
                )
            )
    return outward_sum(terms)


__all__ = [
    "ELIGIBLE_STRUCTURED_SOURCES",
    "STRUCTURED_REMAINDER_CANDIDATE",
    "STRUCTURED_REMAINDER_CAPACITY",
    "STRUCTURED_SOURCE_IDS",
    "StructuredRemainderBoundaryResult",
    "StructuredRemainderState",
    "initialize_structured_remainder_state",
    "materialize_structured_remainder",
    "structured_column_contributions",
    "structured_quadratic_nonlinear_residual",
    "structured_remainder_boundary_update",
]
