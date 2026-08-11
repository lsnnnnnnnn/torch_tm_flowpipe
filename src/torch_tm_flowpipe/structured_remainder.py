"""Bounded structured-remainder candidate S1.

The primitive is state-dimension and system agnostic.  It consumes an additive
typed source ledger, propagates old columns through a safeguarded interval
linear map, materializes nonlinear residuals, and evicts the oldest column
without losing its set contribution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from typing import Mapping

import torch

from .batched_dense_tm import (
    REMAINDER_LEDGER_CATEGORIES,
    VALIDATED_REMAINDER_SOURCE_SCHEMA,
    VALIDATED_REMAINDER_SOURCE_SCHEMA_VERSION,
)
from .fixed_support_outward import (
    OutwardIntervalTensor,
    outward_matmul,
    outward_sum,
)


STRUCTURED_REMAINDER_CAPACITY = 16
STRUCTURED_REMAINDER_CANDIDATE = "normalized_insertion_structured_remainder_k16"
STRUCTURED_TOTAL_DELTA_CANDIDATE = "normalized_insertion_structured_total_delta_k16"
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
    source_boundary_index: torch.Tensor | None = None
    source_occurrence_index: torch.Tensor | None = None
    accepted_boundary_index: int = 0
    event_count: torch.Tensor | None = None

    def __post_init__(self) -> None:
        batch, state_dim = self.ordinary_rem_lo.shape
        capacity = self.j_lo.shape[1]
        if self.ordinary_rem_hi.shape != (batch, state_dim):
            raise ValueError("ordinary structured remainder shape mismatch")
        if self.j_lo.shape != (batch, capacity, state_dim) or self.j_hi.shape != self.j_lo.shape:
            raise ValueError("structured J tensor shape mismatch")
        if self.phi_lo.shape != (batch, capacity, state_dim, state_dim) or self.phi_hi.shape != self.phi_lo.shape:
            raise ValueError("structured Phi tensor shape mismatch")
        if self.active.shape != (batch, capacity):
            raise ValueError("structured active mask shape mismatch")
        for value in (self.age, self.source_id):
            if value.shape != self.active.shape or value.dtype != torch.long:
                raise ValueError("structured integer slot metadata shape/dtype mismatch")
        if self.inverse_scale.shape != (batch, state_dim):
            raise ValueError("structured inverse scale shape mismatch")
        minus_one = torch.full_like(self.age, -1)
        if self.source_boundary_index is None:
            object.__setattr__(self, "source_boundary_index", minus_one.clone())
        if self.source_occurrence_index is None:
            object.__setattr__(self, "source_occurrence_index", minus_one.clone())
        assert self.source_boundary_index is not None
        assert self.source_occurrence_index is not None
        if (
            self.source_boundary_index.shape != self.active.shape
            or self.source_occurrence_index.shape != self.active.shape
            or self.source_boundary_index.dtype != torch.long
            or self.source_occurrence_index.dtype != torch.long
        ):
            raise ValueError("structured unique source identity metadata mismatch")
        if int(self.accepted_boundary_index) < 0:
            raise ValueError("accepted structured boundary index must be nonnegative")
        if self.event_count is None:
            object.__setattr__(
                self,
                "event_count",
                torch.zeros(batch, dtype=torch.long, device=self.active.device),
            )
        assert self.event_count is not None
        if self.event_count.shape != (batch,) or self.event_count.dtype != torch.long:
            raise ValueError("structured event count must be a long tensor with shape [batch]")

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
    source_events: tuple["StructuredSourceEvent", ...] = ()


@dataclass(frozen=True)
class StructuredSourceEvent:
    reason: str
    active_mask: torch.Tensor
    accepted_boundary_index: int
    slot: torch.Tensor
    source_category_id: torch.Tensor
    source_category: tuple[str, ...]
    source_boundary_index: torch.Tensor
    source_occurrence_index: torch.Tensor
    age: torch.Tensor
    pre_propagation_lo: torch.Tensor
    pre_propagation_hi: torch.Tensor
    post_propagation_lo: torch.Tensor
    post_propagation_hi: torch.Tensor
    materialized_lo: torch.Tensor
    materialized_hi: torch.Tensor

    def __post_init__(self) -> None:
        if self.reason not in {"insertion", "capacity_eviction"}:
            raise ValueError("unknown structured source event reason")
        if len(self.source_category) != int(self.active_mask.shape[0]):
            raise ValueError("structured source event category batch mismatch")


@dataclass(frozen=True)
class CompletePolynomialStructuredImage:
    """Outward image of a structured perturbation through a complete polynomial."""

    affine_map_lo: torch.Tensor
    affine_map_hi: torch.Tensor
    affine_image_lo: torch.Tensor
    affine_image_hi: torch.Tensor
    nonlinear_residual_lo: torch.Tensor
    nonlinear_residual_hi: torch.Tensor
    total_difference_lo: torch.Tensor
    total_difference_hi: torch.Tensor
    reconstruction_lo: torch.Tensor
    reconstruction_hi: torch.Tensor
    decomposition_padding_lo: torch.Tensor
    decomposition_padding_hi: torch.Tensor
    containment_mask: torch.Tensor
    domain_scope: str
    proof_diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class CompletePolynomialContractComparison:
    """Shadow comparison of the current and total-delta image contracts.

    ``current_reconstruction`` is the implementation's post-hoc expression
    ``A_current R_o + A_current Z + N_current``.  ``total_delta_reconstruction``
    is ``A_total (R_o + Z) + N_total`` for a base of ``range(Q)``.  The latter
    includes ordinary, structured, and mixed nonlinear routes in one complete
    polynomial difference.  This type is diagnostic only; constructing it
    does not authorize a production state transition.
    """

    current_image: CompletePolynomialStructuredImage
    total_delta_image: CompletePolynomialStructuredImage
    current_affine_ordinary_lo: torch.Tensor
    current_affine_ordinary_hi: torch.Tensor
    current_affine_structured_lo: torch.Tensor
    current_affine_structured_hi: torch.Tensor
    current_reconstruction_lo: torch.Tensor
    current_reconstruction_hi: torch.Tensor
    total_delta_reconstruction_lo: torch.Tensor
    total_delta_reconstruction_hi: torch.Tensor
    current_contains_total_delta_mask: torch.Tensor


def _interval_image(
    matrix_lo: torch.Tensor,
    matrix_hi: torch.Tensor,
    vector_lo: torch.Tensor,
    vector_hi: torch.Tensor,
) -> OutwardIntervalTensor:
    product = outward_matmul(
        OutwardIntervalTensor(matrix_lo, matrix_hi),
        OutwardIntervalTensor(vector_lo[..., None], vector_hi[..., None]),
    )
    return OutwardIntervalTensor(product.lo[..., 0], product.hi[..., 0])


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
        source_boundary_index=torch.full(
            (batch, capacity), -1, dtype=torch.long, device=device_t
        ),
        source_occurrence_index=torch.full(
            (batch, capacity), -1, dtype=torch.long, device=device_t
        ),
        accepted_boundary_index=0,
        event_count=torch.zeros(batch, dtype=torch.long, device=device_t),
    )


def physical_interval_to_normal(
    physical_lo: torch.Tensor,
    physical_hi: torch.Tensor,
    *,
    forward_scale: torch.Tensor,
    inverse_scale: torch.Tensor,
) -> OutwardIntervalTensor:
    """Map a physical interval vector to normalized coordinates, fail closed at zero scale."""
    if not (
        physical_lo.shape
        == physical_hi.shape
        == forward_scale.shape
        == inverse_scale.shape
    ):
        raise ValueError("physical/normal coordinate tensor shapes disagree")
    tensors = (physical_lo, physical_hi, forward_scale, inverse_scale)
    if any(value.dtype != torch.float64 for value in tensors):
        raise ValueError("physical/normal coordinate maps require float64")
    if any(not bool(torch.all(torch.isfinite(value))) for value in tensors):
        raise FloatingPointError("physical/normal coordinate map received nonfinite input")
    if not bool(torch.all(physical_lo <= physical_hi) and torch.all(forward_scale >= 0)):
        raise ValueError("physical interval or forward scale is invalid")
    zero_scale = forward_scale == 0
    nonzero_source = (physical_lo != 0) | (physical_hi != 0)
    if bool(torch.any(zero_scale & nonzero_source)):
        raise ValueError("nonzero physical source cannot enter an exactly zero-scale coordinate")
    expected_inverse = torch.where(
        zero_scale,
        torch.ones_like(forward_scale),
        1.0 / forward_scale,
    )
    if not bool(torch.equal(inverse_scale, expected_inverse)):
        raise ValueError("forward/inverse normalization scales are inconsistent")
    normalized = OutwardIntervalTensor(physical_lo, physical_hi).mul(
        OutwardIntervalTensor.point(inverse_scale)
    )
    zero = torch.zeros_like(normalized.lo)
    exact_zero = (physical_lo == 0) & (physical_hi == 0)
    return OutwardIntervalTensor(
        torch.where(zero_scale | exact_zero, zero, normalized.lo),
        torch.where(zero_scale | exact_zero, zero, normalized.hi),
    )


def normal_interval_to_physical(
    normal_lo: torch.Tensor,
    normal_hi: torch.Tensor,
    *,
    forward_scale: torch.Tensor,
) -> OutwardIntervalTensor:
    """Map a normalized interval vector to physical coordinates outwardly."""
    if not (normal_lo.shape == normal_hi.shape == forward_scale.shape):
        raise ValueError("normal/physical coordinate tensor shapes disagree")
    if any(value.dtype != torch.float64 for value in (normal_lo, normal_hi, forward_scale)):
        raise ValueError("normal/physical coordinate maps require float64")
    if not bool(
        torch.all(torch.isfinite(normal_lo))
        and torch.all(torch.isfinite(normal_hi))
        and torch.all(torch.isfinite(forward_scale))
        and torch.all(normal_lo <= normal_hi)
        and torch.all(forward_scale >= 0)
    ):
        raise ValueError("normal interval or forward scale is invalid")
    physical = OutwardIntervalTensor(normal_lo, normal_hi).mul(
        OutwardIntervalTensor.point(forward_scale)
    )
    exact_zero = ((normal_lo == 0) & (normal_hi == 0)) | (forward_scale == 0)
    zero = torch.zeros_like(physical.lo)
    return OutwardIntervalTensor(
        torch.where(exact_zero, zero, physical.lo),
        torch.where(exact_zero, zero, physical.hi),
    )


def physical_source_to_new_normal_phi(
    source_lo: torch.Tensor,
    source_hi: torch.Tensor,
    *,
    new_forward_scale: torch.Tensor,
    new_inverse_scale: torch.Tensor,
) -> OutwardIntervalTensor:
    """Return the point diagonal physical-source to new-normal coordinate map."""
    physical_interval_to_normal(
        source_lo,
        source_hi,
        forward_scale=new_forward_scale,
        inverse_scale=new_inverse_scale,
    )
    diagonal = torch.diag_embed(new_inverse_scale)
    zero_scale = new_forward_scale == 0
    diagonal = torch.where(zero_scale[..., None], torch.zeros_like(diagonal), diagonal)
    return OutwardIntervalTensor.point(diagonal)


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


def split_structured_source_center(
    source_lo: torch.Tensor,
    source_hi: torch.Tensor,
) -> tuple[OutwardIntervalTensor, OutwardIntervalTensor]:
    """Split one source into its point center and outward symmetric radius."""
    if source_lo.shape != source_hi.shape or source_lo.dtype != torch.float64:
        raise ValueError("structured source center split requires matching float64 tensors")
    if not bool(
        torch.all(torch.isfinite(source_lo))
        and torch.all(torch.isfinite(source_hi))
        and torch.all(source_lo <= source_hi)
    ):
        raise ValueError("structured source center split received an invalid interval")
    return _center_source(OutwardIntervalTensor(source_lo, source_hi))


def _event_source_categories(
    source_ids: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[str, ...]:
    by_id = {value: name for name, value in STRUCTURED_SOURCE_IDS.items()}
    return tuple(
        by_id.get(int(source_id), "") if bool(active) else ""
        for source_id, active in zip(source_ids.detach().cpu(), active_mask.detach().cpu())
    )


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
    A_old_normal_to_new_normal_lo: torch.Tensor,
    A_old_normal_to_new_normal_hi: torch.Tensor,
    nonlinear_residual_lo: torch.Tensor | None,
    nonlinear_residual_hi: torch.Tensor | None,
    new_forward_scale: torch.Tensor,
    boundary_index: int,
    map_is_affine: bool = False,
    source_schema: str = VALIDATED_REMAINDER_SOURCE_SCHEMA,
    source_schema_version: int = VALIDATED_REMAINDER_SOURCE_SCHEMA_VERSION,
) -> StructuredRemainderBoundaryResult:
    """Apply one S1 boundary with frozen K16 oldest-first eviction."""

    expected_vector = (state.batch, state.state_dim)
    expected_matrix = (state.batch, state.state_dim, state.state_dim)
    if (
        validated_remainder_lo.shape != expected_vector
        or validated_remainder_hi.shape != expected_vector
        or A_old_normal_to_new_normal_lo.shape != expected_matrix
        or A_old_normal_to_new_normal_hi.shape != expected_matrix
        or new_forward_scale.shape != expected_vector
    ):
        return _failure_result(state, "dimension_mismatch")
    if state.capacity != STRUCTURED_REMAINDER_CAPACITY and state.capacity != 1:
        return _failure_result(state, "capacity_not_frozen_k16_or_unit_test_k1")
    if int(boundary_index) != int(state.accepted_boundary_index):
        return _failure_result(state, "accepted_boundary_index_mismatch")
    assert state.source_boundary_index is not None
    assert state.source_occurrence_index is not None
    duplicate_identity = torch.zeros(
        state.batch, dtype=torch.bool, device=state.active.device
    )
    for left in range(state.capacity):
        for right in range(left + 1, state.capacity):
            duplicate_identity |= (
                state.active[:, left]
                & state.active[:, right]
                & (state.source_id[:, left] == state.source_id[:, right])
                & (
                    state.source_boundary_index[:, left]
                    == state.source_boundary_index[:, right]
                )
                & (
                    state.source_occurrence_index[:, left]
                    == state.source_occurrence_index[:, right]
                )
            )
    if bool(torch.any(duplicate_identity)):
        return _failure_result(state, "duplicate_unique_source_identity")
    if source_schema != VALIDATED_REMAINDER_SOURCE_SCHEMA:
        return _failure_result(state, "source_schema_mismatch")
    if int(source_schema_version) != VALIDATED_REMAINDER_SOURCE_SCHEMA_VERSION:
        return _failure_result(state, "source_schema_version_mismatch")
    unknown = set(typed_sources) - set(REMAINDER_LEDGER_CATEGORIES)
    if unknown:
        return _failure_result(state, f"unknown_source_category:{sorted(unknown)[0]}")
    missing = set(REMAINDER_LEDGER_CATEGORIES) - set(typed_sources)
    if missing:
        return _failure_result(state, f"missing_source_category:{sorted(missing)[0]}")
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
        A_old_normal_to_new_normal_lo,
        A_old_normal_to_new_normal_hi,
        nonlinear_residual_lo,
        nonlinear_residual_hi,
        new_forward_scale,
    )
    if any(tensor.dtype != torch.float64 for tensor in tensors):
        return _failure_result(state, "dtype_mismatch")
    if any(not bool(torch.all(torch.isfinite(tensor))) for tensor in tensors):
        return _failure_result(state, "nonfinite_input")
    if (
        not bool(torch.all(validated_remainder_lo <= validated_remainder_hi))
        or not bool(torch.all(A_old_normal_to_new_normal_lo <= A_old_normal_to_new_normal_hi))
        or not bool(torch.all(nonlinear_residual_lo <= nonlinear_residual_hi))
        or not bool(torch.all(new_forward_scale >= 0))
    ):
        return _failure_result(state, "invalid_interval_or_scale")
    for name in REMAINDER_LEDGER_CATEGORIES:
        lo, hi = typed_sources[name]
        if lo.shape != expected_vector or hi.shape != expected_vector:
            return _failure_result(state, f"source_dimension_mismatch:{name}")
        if not bool(torch.all(torch.isfinite(lo)) and torch.all(torch.isfinite(hi))):
            return _failure_result(state, f"nonfinite_source:{name}")
        if not bool(torch.all(lo <= hi)):
            return _failure_result(state, f"invalid_source_interval:{name}")

    inverse_scale = torch.where(
        new_forward_scale == 0,
        torch.ones_like(new_forward_scale),
        1.0 / new_forward_scale,
    )
    linear_map = OutwardIntervalTensor(
        A_old_normal_to_new_normal_lo,
        A_old_normal_to_new_normal_hi,
    )
    pre_propagation_columns = structured_column_contributions(state)
    source_events: list[StructuredSourceEvent] = []
    event_count_increment = torch.zeros(
        state.batch, dtype=torch.long, device=state.active.device
    )
    old_materialized = materialize_structured_remainder(state)
    propagated_pre_split = _affine_map_interval(linear_map, old_materialized)
    normalized_sources: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    try:
        for name in REMAINDER_LEDGER_CATEGORIES:
            normalized = physical_interval_to_normal(
                *typed_sources[name],
                forward_scale=new_forward_scale,
                inverse_scale=inverse_scale,
            )
            normalized_sources[name] = (normalized.lo, normalized.hi)
        validated_normal = physical_interval_to_normal(
            validated_remainder_lo,
            validated_remainder_hi,
            forward_scale=new_forward_scale,
            inverse_scale=inverse_scale,
        )
    except (FloatingPointError, ValueError) as exc:
        return _failure_result(state, f"coordinate_map_failed:{exc}")
    ledger_total = _ledger_total(normalized_sources, validated_remainder_lo)
    nonlinear = OutwardIntervalTensor(
        nonlinear_residual_lo, nonlinear_residual_hi
    )
    pre_split = outward_sum((propagated_pre_split, ledger_total, nonlinear))
    validated = validated_normal
    source_decomposition_mask = (
        (pre_split.lo <= validated.lo) & (pre_split.hi >= validated.hi)
    ).all(dim=1)

    ordinary_old = _affine_map_interval(
        linear_map,
        OutwardIntervalTensor(state.ordinary_rem_lo, state.ordinary_rem_hi),
    )
    ordinary = ordinary_old.add(nonlinear)
    for name in REMAINDER_LEDGER_CATEGORIES:
        if name in ELIGIBLE_STRUCTURED_SOURCES:
            continue
        ordinary = ordinary.add(OutwardIntervalTensor(*normalized_sources[name]))

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
        inverse_scale=inverse_scale,
        source_boundary_index=state.source_boundary_index,
        source_occurrence_index=state.source_occurrence_index,
        accepted_boundary_index=state.accepted_boundary_index,
        event_count=state.event_count,
    )
    propagated_columns = structured_column_contributions(next_state).sum(dim=1)
    new_symbolic = _zero(state)
    evicted_total = _zero(state)
    evicted_source_id = torch.full(
        (state.batch,), -1, dtype=torch.long, device=state.active.device
    )
    evicted_age = torch.full_like(evicted_source_id, -1)
    for name in ELIGIBLE_STRUCTURED_SOURCES:
        source_present = (
            (typed_sources[name][0] != 0) | (typed_sources[name][1] != 0)
        ).any(dim=1)
        if not bool(torch.any(source_present)):
            continue
        center, symmetric = _center_source(
            OutwardIntervalTensor(*typed_sources[name])
        )
        center_normal = physical_interval_to_normal(
            center.lo,
            center.hi,
            forward_scale=new_forward_scale,
            inverse_scale=inverse_scale,
        )
        source_normal = physical_interval_to_normal(
            symmetric.lo,
            symmetric.hi,
            forward_scale=new_forward_scale,
            inverse_scale=inverse_scale,
        )
        new_source_phi = physical_source_to_new_normal_phi(
            symmetric.lo,
            symmetric.hi,
            new_forward_scale=new_forward_scale,
            new_inverse_scale=inverse_scale,
        )
        ordinary = OutwardIntervalTensor(
            next_state.ordinary_rem_lo, next_state.ordinary_rem_hi
        ).add(center_normal)
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
            next_state.source_boundary_index,
            next_state.source_occurrence_index,
            next_state.accepted_boundary_index,
            next_state.event_count,
        )
        full = next_state.active.all(dim=1) & source_present
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
        ).to(torch.bool) & source_present[:, None]
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
        selected_boundary = torch.sum(
            next_state.source_boundary_index
            * one_hot.to(next_state.source_boundary_index.dtype),
            dim=1,
        )
        selected_occurrence = torch.sum(
            next_state.source_occurrence_index
            * one_hot.to(next_state.source_occurrence_index.dtype),
            dim=1,
        )
        pre_selected = OutwardIntervalTensor(
            (pre_propagation_columns.lo * one_hot[..., None]).sum(dim=1),
            (pre_propagation_columns.hi * one_hot[..., None]).sum(dim=1),
        )
        source_events.append(
            StructuredSourceEvent(
                reason="capacity_eviction",
                active_mask=full,
                accepted_boundary_index=int(boundary_index),
                slot=slot,
                source_category_id=selected_source,
                source_category=_event_source_categories(selected_source, full),
                source_boundary_index=selected_boundary,
                source_occurrence_index=selected_occurrence,
                age=selected_age,
                pre_propagation_lo=pre_selected.lo,
                pre_propagation_hi=pre_selected.hi,
                post_propagation_lo=selected.lo,
                post_propagation_hi=selected.hi,
                materialized_lo=evicted.lo,
                materialized_hi=evicted.hi,
            )
        )
        event_count_increment += full.to(torch.long)
        mask_vector = one_hot[..., None]
        mask_matrix = one_hot[..., None, None]
        next_state = StructuredRemainderState(
            ordinary_after_eviction.lo,
            ordinary_after_eviction.hi,
            torch.where(mask_vector, symmetric.lo[:, None, :], next_state.j_lo),
            torch.where(mask_vector, symmetric.hi[:, None, :], next_state.j_hi),
            torch.where(
                mask_matrix,
                new_source_phi.lo[:, None, :, :],
                next_state.phi_lo,
            ),
            torch.where(
                mask_matrix,
                new_source_phi.hi[:, None, :, :],
                next_state.phi_hi,
            ),
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
            torch.where(
                one_hot,
                torch.full_like(next_state.source_boundary_index, int(boundary_index)),
                next_state.source_boundary_index,
            ),
            torch.where(
                one_hot,
                torch.zeros_like(next_state.source_occurrence_index),
                next_state.source_occurrence_index,
            ),
            next_state.accepted_boundary_index,
            next_state.event_count,
        )
        new_symbolic = new_symbolic.add(source_normal)
        source_events.append(
            StructuredSourceEvent(
                reason="insertion",
                active_mask=source_present,
                accepted_boundary_index=int(boundary_index),
                slot=slot,
                source_category_id=torch.full_like(
                    slot, STRUCTURED_SOURCE_IDS[name]
                ),
                source_category=tuple(
                    name if bool(present) else ""
                    for present in source_present.detach().cpu()
                ),
                source_boundary_index=torch.full_like(slot, int(boundary_index)),
                source_occurrence_index=torch.zeros_like(slot),
                age=torch.full_like(slot, int(boundary_index)),
                pre_propagation_lo=symmetric.lo,
                pre_propagation_hi=symmetric.hi,
                post_propagation_lo=source_normal.lo,
                post_propagation_hi=source_normal.hi,
                materialized_lo=torch.zeros_like(source_normal.lo),
                materialized_hi=torch.zeros_like(source_normal.hi),
            )
        )
        event_count_increment += source_present.to(torch.long)

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
        next_state.source_boundary_index,
        next_state.source_occurrence_index,
        next_state.accepted_boundary_index,
        next_state.event_count,
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
    committed_state = (
        replace(
            next_state,
            accepted_boundary_index=int(state.accepted_boundary_index) + 1,
            event_count=state.event_count + event_count_increment,
        )
        if bool(torch.all(accepted))
        else state
    )
    return StructuredRemainderBoundaryResult(
        state=committed_state,
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
        source_events=tuple(source_events),
    )


def _outward_power(
    value: OutwardIntervalTensor,
    exponent: int,
) -> OutwardIntervalTensor:
    if int(exponent) < 0:
        raise ValueError("interval powers require a nonnegative exponent")
    result = OutwardIntervalTensor.point(torch.ones_like(value.lo))
    for _ in range(int(exponent)):
        result = result.mul(value)
    return result


def _outward_sum_or_zero(
    values: list[OutwardIntervalTensor],
    like: torch.Tensor,
) -> OutwardIntervalTensor:
    return outward_sum(values) if values else OutwardIntervalTensor.zeros_like(like)


def complete_polynomial_structured_image(
    polynomial: object,
    base_domain: tuple[torch.Tensor, torch.Tensor],
    structured_box: tuple[torch.Tensor, torch.Tensor],
    coordinate_map: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    tau_domain: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    tau_index: int | None = None,
) -> CompletePolynomialStructuredImage:
    """Enclose ``P(q + C z) - P(q)`` for every retained degree-0..4 monomial.

    ``q`` and ``z`` are independent interval boxes. ``C`` maps structured
    coordinates into polynomial input coordinates. The degree-one-in-``z``
    terms define an interval affine map; every higher structured degree and
    any decomposition alignment padding is returned as nonlinear residual.
    The qualified implementation is eager CPU binary64 only.
    """

    if isinstance(polynomial, tuple) and len(polynomial) == 2:
        coefficients, exponents = polynomial
    else:
        coefficients = getattr(polynomial, "coeffs", None)
        basis = getattr(polynomial, "basis", None)
        exponents = getattr(basis, "exponents", None)
    if not isinstance(coefficients, torch.Tensor) or not isinstance(exponents, torch.Tensor):
        raise TypeError("polynomial must expose coefficients [B,O,M] and exponents [M,V]")
    if coefficients.ndim != 3 or exponents.ndim != 2:
        raise ValueError("polynomial coefficients/exponents have invalid rank")
    if coefficients.shape[-1] != exponents.shape[0]:
        raise ValueError("polynomial coefficient and exponent counts disagree")
    if coefficients.dtype != torch.float64 or coefficients.device.type != "cpu":
        raise ValueError("complete-O4 structured image is qualified only for CPU float64")
    if exponents.shape[1] <= 0 or bool(torch.any(exponents < 0)):
        raise ValueError("polynomial exponents must be nonnegative and nonempty")
    active_terms = torch.any(coefficients != 0, dim=(0, 1))
    maximum_degree = (
        int(exponents[active_terms].sum(dim=1).max().item())
        if bool(torch.any(active_terms))
        else 0
    )
    if maximum_degree > 4:
        raise ValueError("complete-O4 structured image rejects retained degree above four")

    base_lo, base_hi = base_domain
    structured_lo, structured_hi = structured_box
    if isinstance(coordinate_map, tuple):
        map_lo, map_hi = coordinate_map
    else:
        map_lo = coordinate_map
        map_hi = coordinate_map
    batch, output_dim, _ = coefficients.shape
    variable_dim = exponents.shape[1]
    if base_lo.shape != (batch, variable_dim) or base_hi.shape != base_lo.shape:
        raise ValueError("base domain must have shape [batch, polynomial variables]")
    if structured_lo.ndim != 2 or structured_hi.shape != structured_lo.shape:
        raise ValueError("structured box must have shape [batch, structured dimension]")
    structured_dim = structured_lo.shape[1]
    if structured_lo.shape[0] != batch:
        raise ValueError("structured and polynomial batch dimensions disagree")
    if map_lo.shape != (batch, variable_dim, structured_dim) or map_hi.shape != map_lo.shape:
        raise ValueError("coordinate map must have shape [batch, polynomial variables, structured dimension]")
    tensors = (
        coefficients,
        base_lo,
        base_hi,
        structured_lo,
        structured_hi,
        map_lo,
        map_hi,
    )
    if any(value.dtype != torch.float64 or value.device.type != "cpu" for value in tensors):
        raise ValueError("all complete-O4 structured image tensors must be CPU float64")
    if any(not bool(torch.all(torch.isfinite(value))) for value in tensors):
        raise FloatingPointError("complete-O4 structured image received a nonfinite tensor")
    if not bool(
        torch.all(base_lo <= base_hi)
        and torch.all(structured_lo <= structured_hi)
        and torch.all(map_lo <= map_hi)
    ):
        raise ValueError("complete-O4 structured image received an inverted interval")

    effective_base_lo = base_lo.clone()
    effective_base_hi = base_hi.clone()
    domain_scope = "unspecified"
    if tau_domain is not None:
        if tau_index is None or not 0 <= int(tau_index) < variable_dim:
            raise ValueError("tau_domain requires a valid tau_index")
        tau_lo, tau_hi = tau_domain
        if tau_lo.shape == (batch,):
            tau_lo = tau_lo[:, None]
            tau_hi = tau_hi[:, None]
        if tau_lo.shape != (batch, 1) or tau_hi.shape != tau_lo.shape:
            raise ValueError("tau domain must have shape [batch] or [batch,1]")
        if tau_lo.dtype != torch.float64 or tau_hi.dtype != torch.float64:
            raise ValueError("tau domain must use float64")
        if not bool(
            torch.all(torch.isfinite(tau_lo))
            and torch.all(torch.isfinite(tau_hi))
            and torch.all(tau_lo <= tau_hi)
        ):
            raise ValueError("tau domain is invalid")
        if not bool(
            torch.all(map_lo[:, int(tau_index), :] == 0)
            and torch.all(map_hi[:, int(tau_index), :] == 0)
        ):
            raise ValueError("structured coordinate map must not perturb local time")
        effective_base_lo[:, int(tau_index)] = tau_lo[:, 0]
        effective_base_hi[:, int(tau_index)] = tau_hi[:, 0]
        domain_scope = (
            "endpoint_tau_point"
            if bool(torch.equal(tau_lo, tau_hi))
            else "tube_tau_interval"
        )

    base = OutwardIntervalTensor(effective_base_lo, effective_base_hi)
    structured = OutwardIntervalTensor(structured_lo, structured_hi)
    coordinate = OutwardIntervalTensor(map_lo, map_hi)
    delta_matrix = outward_matmul(
        coordinate,
        OutwardIntervalTensor(structured.lo[..., None], structured.hi[..., None]),
    )
    delta = OutwardIntervalTensor(delta_matrix.lo[..., 0], delta_matrix.hi[..., 0])

    total_terms: list[OutwardIntervalTensor] = []
    nonlinear_terms: list[OutwardIntervalTensor] = []
    affine_terms: list[OutwardIntervalTensor] = []
    route_count = 0
    nonlinear_route_count = 0
    exponent_rows = exponents.detach().cpu().to(torch.long).tolist()
    for term_index, exponent_row in enumerate(exponent_rows):
        if not bool(active_terms[term_index]):
            continue
        coefficient = OutwardIntervalTensor.point(coefficients[..., term_index])
        for route in product(*(range(int(power) + 1) for power in exponent_row)):
            structured_degree = sum(route)
            if structured_degree == 0:
                continue
            route_count += 1
            term = coefficient
            for variable_index, (power, selected) in enumerate(zip(exponent_row, route)):
                combinatorial = math.comb(int(power), int(selected))
                base_factor = _outward_power(
                    OutwardIntervalTensor(
                        base.lo[:, variable_index, None],
                        base.hi[:, variable_index, None],
                    ),
                    int(power) - int(selected),
                )
                delta_factor = _outward_power(
                    OutwardIntervalTensor(
                        delta.lo[:, variable_index, None],
                        delta.hi[:, variable_index, None],
                    ),
                    int(selected),
                )
                term = term.mul(base_factor.mul(delta_factor).scale(float(combinatorial)))
            total_terms.append(term)
            if structured_degree >= 2:
                nonlinear_route_count += 1
                nonlinear_terms.append(term)

        for variable_index, power in enumerate(exponent_row):
            if int(power) == 0:
                continue
            derivative = coefficient.scale(float(power))
            for base_index, base_power in enumerate(exponent_row):
                derivative_power = int(base_power) - (1 if base_index == variable_index else 0)
                derivative = derivative.mul(
                    _outward_power(
                        OutwardIntervalTensor(
                            base.lo[:, base_index, None],
                            base.hi[:, base_index, None],
                        ),
                        derivative_power,
                    )
                )
            affine_terms.append(
                OutwardIntervalTensor(
                    derivative.lo[..., None], derivative.hi[..., None]
                ).mul(
                    OutwardIntervalTensor(
                        coordinate.lo[:, variable_index, None, :],
                        coordinate.hi[:, variable_index, None, :],
                    )
                )
            )

    output_zero = torch.zeros((batch, output_dim), dtype=torch.float64)
    affine_zero = torch.zeros(
        (batch, output_dim, structured_dim), dtype=torch.float64
    )
    affine_map = _outward_sum_or_zero(affine_terms, affine_zero)
    affine_product = outward_matmul(
        affine_map,
        OutwardIntervalTensor(structured.lo[..., None], structured.hi[..., None]),
    )
    affine_image = OutwardIntervalTensor(
        affine_product.lo[..., 0], affine_product.hi[..., 0]
    )
    total_difference = _outward_sum_or_zero(total_terms, output_zero)
    nonlinear = _outward_sum_or_zero(nonlinear_terms, output_zero)

    if maximum_degree <= 1:
        # Preserve the exact affine invariant and use the grouped affine image
        # as the canonical difference enclosure.
        total_difference = affine_image
        nonlinear = OutwardIntervalTensor.zeros_like(output_zero)
        padding = OutwardIntervalTensor.zeros_like(output_zero)
        reconstruction = affine_image
    else:
        reconstruction_before_padding = affine_image.add(nonlinear)
        padding_lo = torch.minimum(
            total_difference.lo - reconstruction_before_padding.lo,
            torch.zeros_like(output_zero),
        )
        padding_hi = torch.maximum(
            total_difference.hi - reconstruction_before_padding.hi,
            torch.zeros_like(output_zero),
        )
        padding = OutwardIntervalTensor(
            torch.nextafter(padding_lo, torch.full_like(padding_lo, -torch.inf)),
            torch.nextafter(padding_hi, torch.full_like(padding_hi, torch.inf)),
        ).sanitized()
        nonlinear = nonlinear.add(padding)
        reconstruction = affine_image.add(nonlinear)
    zero_structured = ((structured_lo == 0) & (structured_hi == 0)).all(dim=1)
    if bool(torch.any(zero_structured)):
        mask = zero_structured[:, None]
        exact_zero = torch.zeros_like(output_zero)
        affine_image = OutwardIntervalTensor(
            torch.where(mask, exact_zero, affine_image.lo),
            torch.where(mask, exact_zero, affine_image.hi),
        )
        nonlinear = OutwardIntervalTensor(
            torch.where(mask, exact_zero, nonlinear.lo),
            torch.where(mask, exact_zero, nonlinear.hi),
        )
        total_difference = OutwardIntervalTensor(
            torch.where(mask, exact_zero, total_difference.lo),
            torch.where(mask, exact_zero, total_difference.hi),
        )
        reconstruction = OutwardIntervalTensor(
            torch.where(mask, exact_zero, reconstruction.lo),
            torch.where(mask, exact_zero, reconstruction.hi),
        )
        padding = OutwardIntervalTensor(
            torch.where(mask, exact_zero, padding.lo),
            torch.where(mask, exact_zero, padding.hi),
        )
    containment = (
        (reconstruction.lo <= total_difference.lo)
        & (reconstruction.hi >= total_difference.hi)
        & reconstruction.finite()
        & total_difference.finite()
    ).all(dim=1)
    if not bool(torch.all(containment)):
        raise FloatingPointError("complete-O4 affine/nonlinear reconstruction lost containment")
    return CompletePolynomialStructuredImage(
        affine_map.lo,
        affine_map.hi,
        affine_image.lo,
        affine_image.hi,
        nonlinear.lo,
        nonlinear.hi,
        total_difference.lo,
        total_difference.hi,
        reconstruction.lo,
        reconstruction.hi,
        padding.lo,
        padding.hi,
        containment,
        domain_scope,
        {
            "maximum_retained_degree": maximum_degree,
            "source_term_count": int(coefficients.shape[-1]),
            "binomial_difference_route_count": route_count,
            "nonlinear_route_count": nonlinear_route_count,
            "outward_semantics": "CPU float64 nextafter after each interval operation",
            "affine_choice": "degree-one-in-structured-variables coefficient interval over base domain",
        },
    )


def compare_complete_polynomial_contracts(
    polynomial: object,
    *,
    polynomial_base_domain: tuple[torch.Tensor, torch.Tensor],
    current_base_domain: tuple[torch.Tensor, torch.Tensor],
    ordinary_box: tuple[torch.Tensor, torch.Tensor],
    structured_box: tuple[torch.Tensor, torch.Tensor],
    coordinate_map: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> CompletePolynomialContractComparison:
    """Build the two Phase-2 image contracts from the same tensor prestate.

    ``current_base_domain`` must be the actual box supplied by the current
    implementation (normally ``range(Q + R_o)``); it is deliberately not
    reconstructed here so this shadow cannot perturb production arithmetic.
    ``polynomial_base_domain`` is ``range(Q)``.  Ordinary and structured boxes
    share the old normalized coordinate system.
    """

    ordinary = OutwardIntervalTensor(*ordinary_box)
    structured = OutwardIntervalTensor(*structured_box)
    if ordinary.lo.shape != structured.lo.shape:
        raise ValueError("ordinary and structured perturbation boxes must agree")
    total_delta = ordinary.add(structured)
    current_image = complete_polynomial_structured_image(
        polynomial,
        current_base_domain,
        structured_box,
        coordinate_map,
    )
    total_delta_image = complete_polynomial_structured_image(
        polynomial,
        polynomial_base_domain,
        (total_delta.lo, total_delta.hi),
        coordinate_map,
    )
    current_affine_ordinary = _interval_image(
        current_image.affine_map_lo,
        current_image.affine_map_hi,
        ordinary.lo,
        ordinary.hi,
    )
    current_affine_structured = _interval_image(
        current_image.affine_map_lo,
        current_image.affine_map_hi,
        structured.lo,
        structured.hi,
    )
    current_reconstruction = current_affine_ordinary.add(
        current_affine_structured
    ).add(
        OutwardIntervalTensor(
            current_image.nonlinear_residual_lo,
            current_image.nonlinear_residual_hi,
        )
    )
    total_reconstruction = OutwardIntervalTensor(
        total_delta_image.reconstruction_lo,
        total_delta_image.reconstruction_hi,
    )
    contains_total = (
        (current_reconstruction.lo <= total_delta_image.total_difference_lo)
        & (current_reconstruction.hi >= total_delta_image.total_difference_hi)
    ).all(dim=1)
    return CompletePolynomialContractComparison(
        current_image=current_image,
        total_delta_image=total_delta_image,
        current_affine_ordinary_lo=current_affine_ordinary.lo,
        current_affine_ordinary_hi=current_affine_ordinary.hi,
        current_affine_structured_lo=current_affine_structured.lo,
        current_affine_structured_hi=current_affine_structured.hi,
        current_reconstruction_lo=current_reconstruction.lo,
        current_reconstruction_hi=current_reconstruction.hi,
        total_delta_reconstruction_lo=total_reconstruction.lo,
        total_delta_reconstruction_hi=total_reconstruction.hi,
        current_contains_total_delta_mask=contains_total,
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
    "CompletePolynomialContractComparison",
    "CompletePolynomialStructuredImage",
    "ELIGIBLE_STRUCTURED_SOURCES",
    "STRUCTURED_REMAINDER_CANDIDATE",
    "STRUCTURED_REMAINDER_CAPACITY",
    "STRUCTURED_TOTAL_DELTA_CANDIDATE",
    "STRUCTURED_SOURCE_IDS",
    "StructuredRemainderBoundaryResult",
    "StructuredRemainderState",
    "StructuredSourceEvent",
    "complete_polynomial_structured_image",
    "compare_complete_polynomial_contracts",
    "initialize_structured_remainder_state",
    "materialize_structured_remainder",
    "normal_interval_to_physical",
    "physical_interval_to_normal",
    "physical_source_to_new_normal_phi",
    "split_structured_source_center",
    "structured_column_contributions",
    "structured_quadratic_nonlinear_residual",
    "structured_remainder_boundary_update",
]
