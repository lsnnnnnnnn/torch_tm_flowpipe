"""Experimental symbolic remainder helpers.

This module is intentionally diagnostic-only.  Noise symbols are represented as
ordinary polynomial variables with domain [-1, 1], so the existing TaylorModel
arithmetic can carry recent residual sources symbolically without changing the
default TaylorModel implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import torch

from .interval import Interval
from .polynomial import Polynomial
from .taylor_model import TaylorModel
from .tm_vector import TMVector


def symbolic_noise_domain() -> Interval:
    return Interval(-1.0, 1.0)


@dataclass(frozen=True)
class SymbolicNoiseSymbol:
    """Metadata for one bounded noise variable eps_i in [-1, 1]."""

    symbol_id: int
    var_index: int
    state_dim: int
    source: str = "picard_residual"


@dataclass(frozen=True)
class SymbolicRemainderState:
    """Queue metadata for diagnostic symbolic remainders."""

    symbols: tuple[SymbolicNoiseSymbol, ...] = ()
    next_symbol_id: int = 0
    max_symbolic_remainders: int = 0

    @staticmethod
    def empty(max_symbolic_remainders: int = 0) -> "SymbolicRemainderState":
        return SymbolicRemainderState((), 0, int(max_symbolic_remainders))

    def active_var_indices(self) -> tuple[int, ...]:
        return tuple(symbol.var_index for symbol in self.symbols)

    def with_queue_size(self, max_symbolic_remainders: int) -> "SymbolicRemainderState":
        return replace(self, max_symbolic_remainders=int(max_symbolic_remainders))


IntervalColumn = tuple[Interval, ...]
RealMatrix = tuple[tuple[float, ...], ...]
IntervalMatrix = tuple[tuple[Interval, ...], ...]
ACCEPTED_BOUNDARY_SR_OWNER_SCHEMA = "accepted_boundary_sr_v1"
C3_CROSS_STEP_SR_OWNER_SCHEMA = "c3_cross_step_sr_v1"


@dataclass(frozen=True)
class FlowstarSymbolicRemainderQueue:
    """Clean-room skeleton of Flow*'s J/Phi_L/scalars remainder queue.

    This is separate from ``SymbolicRemainderState`` above. Flow* does not add
    ordinary polynomial noise variables for each residual; it keeps a queue of
    interval remainder columns and propagates older columns through the linear
    part of each accepted reset map.
    """

    J: tuple[IntervalColumn, ...]
    Phi_L: tuple[RealMatrix, ...]
    scalars: tuple[float, ...]
    max_size: int
    # Certified mirrors and accepted-boundary ownership metadata.  The legacy
    # diagnostic queue leaves these empty; generic SR and its frozen C3 wrapper
    # validate every field before consuming the state.
    Phi_L_iv: tuple[IntervalMatrix, ...] = ()
    scalars_iv: tuple[Interval, ...] = ()
    generation: int = 0
    accepted_boundary_index: int = 0
    owner_generations: tuple[int, ...] = ()
    owner_boundary_indices: tuple[int, ...] = ()
    reset_count: int = 0
    owner_schema: str = "legacy_diagnostic"

    @staticmethod
    def empty(dim: int, max_size: int = 100) -> "FlowstarSymbolicRemainderQueue":
        return FlowstarSymbolicRemainderQueue((), (), tuple(1.0 for _ in range(int(dim))), int(max_size))

    @staticmethod
    def empty_c3(
        dim: int,
        max_size: int = 100,
        *,
        accepted_boundary_index: int = 0,
        generation: int | None = None,
        reset_count: int = 0,
        reference: Interval | None = None,
    ) -> "FlowstarSymbolicRemainderQueue":
        """Create an explicitly owned C3 queue at an accepted boundary."""

        return FlowstarSymbolicRemainderQueue.empty_accepted_boundary_sr(
            dim,
            max_size,
            accepted_boundary_index=accepted_boundary_index,
            generation=generation,
            reset_count=reset_count,
            reference=reference,
            owner_schema=C3_CROSS_STEP_SR_OWNER_SCHEMA,
        )

    @staticmethod
    def empty_accepted_boundary_sr(
        dim: int,
        max_size: int = 100,
        *,
        accepted_boundary_index: int = 0,
        generation: int | None = None,
        reset_count: int = 0,
        reference: Interval | None = None,
        owner_schema: str = ACCEPTED_BOUNDARY_SR_OWNER_SCHEMA,
    ) -> "FlowstarSymbolicRemainderQueue":
        """Create a plant-independent queue owned by one accepted boundary."""

        if int(max_size) < 1:
            raise ValueError("accepted-boundary SR queue max_size must be positive")
        if int(dim) < 1:
            raise ValueError("accepted-boundary SR queue dimension must be positive")
        if owner_schema not in {
            ACCEPTED_BOUNDARY_SR_OWNER_SCHEMA,
            C3_CROSS_STEP_SR_OWNER_SCHEMA,
        }:
            raise ValueError("accepted-boundary SR queue owner schema is unsupported")
        if generation is None:
            generation = int(accepted_boundary_index)
        if reference is None:
            reference = Interval.zero()
        ones = tuple(1.0 for _ in range(int(dim)))
        ones_iv = tuple(Interval.point(torch.ones_like(reference.lo)) for _ in range(int(dim)))
        return FlowstarSymbolicRemainderQueue(
            (),
            (),
            ones,
            int(max_size),
            Phi_L_iv=(),
            scalars_iv=ones_iv,
            generation=int(generation),
            accepted_boundary_index=int(accepted_boundary_index),
            owner_generations=(),
            owner_boundary_indices=(),
            reset_count=int(reset_count),
            owner_schema=owner_schema,
        )

    @property
    def dim(self) -> int:
        return len(self.scalars)

    def reset(self, dim: int | None = None) -> "FlowstarSymbolicRemainderQueue":
        return FlowstarSymbolicRemainderQueue.empty(self.dim if dim is None else int(dim), self.max_size)


def _zero_interval_like(domain: Sequence[Interval]) -> Interval:
    if domain:
        return Interval.zero(dtype=domain[0].dtype, device=domain[0].device)
    return Interval.zero()


def _zero_interval_like_interval(iv: Interval) -> Interval:
    return Interval.zero(dtype=iv.dtype, device=iv.device)


def _linear_coefficients(tm: TMVector) -> RealMatrix:
    dim = len(tm)
    rows: list[list[float]] = []
    for model in tm:
        row = [0.0 for _ in range(dim)]
        for exp, coeff in model.polynomial.terms.items():
            if sum(exp) != 1:
                continue
            for var_index in range(min(dim, len(exp))):
                if exp[var_index] == 1 and all(power == 0 for j, power in enumerate(exp) if j != var_index):
                    row[var_index] = float(coeff.detach().cpu())
                    break
        rows.append(row)
    return tuple(tuple(row) for row in rows)


def _right_scale_matrix(matrix: RealMatrix, scalars: Sequence[float]) -> RealMatrix:
    return tuple(tuple(value * float(scalars[j]) for j, value in enumerate(row)) for row in matrix)


def _identity_matrix(dim: int) -> RealMatrix:
    return tuple(tuple(1.0 if i == j else 0.0 for j in range(int(dim))) for i in range(int(dim)))


def _matmul_real(a: RealMatrix, b: RealMatrix) -> RealMatrix:
    if not a:
        return ()
    cols = len(b[0]) if b else 0
    out: list[tuple[float, ...]] = []
    for row in a:
        out_row = []
        for col in range(cols):
            out_row.append(sum(float(row[k]) * float(b[k][col]) for k in range(len(b))))
        out.append(tuple(out_row))
    return tuple(out)


def _matmul_interval_col(matrix: RealMatrix, column: IntervalColumn, reference: Interval) -> IntervalColumn:
    out: list[Interval] = []
    for row in matrix:
        acc = _zero_interval_like_interval(reference)
        for scalar, iv in zip(row, column):
            if scalar:
                acc = acc + iv * float(scalar)
        out.append(acc)
    return tuple(out)


def _identity_interval_matrix(dim: int, reference: Interval) -> IntervalMatrix:
    return tuple(
        tuple(
            Interval.point(
                torch.as_tensor(
                    1.0 if i == j else 0.0,
                    dtype=reference.dtype,
                    device=reference.device,
                )
            )
            for j in range(int(dim))
        )
        for i in range(int(dim))
    )


def _matmul_interval_matrix_scalar(
    a: IntervalMatrix,
    b: IntervalMatrix,
    reference: Interval,
) -> IntervalMatrix:
    """Original scalar-object schedule retained as a bitwise test oracle."""

    dim = len(a)
    out: list[tuple[Interval, ...]] = []
    for i in range(dim):
        row: list[Interval] = []
        for j in range(dim):
            acc = _zero_interval_like_interval(reference)
            for k in range(dim):
                acc = acc + a[i][k] * b[k][j]
            row.append(acc)
        out.append(tuple(row))
    return tuple(out)


def _matmul_interval_matrix_col_scalar(
    matrix: IntervalMatrix,
    column: IntervalColumn,
    reference: Interval,
) -> IntervalColumn:
    """Original scalar-object schedule retained as a bitwise test oracle."""

    out: list[Interval] = []
    for row in matrix:
        acc = _zero_interval_like_interval(reference)
        for value, interval in zip(row, column):
            acc = acc + value * interval
        out.append(acc)
    return tuple(out)


def _pack_interval_matrices(
    matrices: Sequence[IntervalMatrix],
    *,
    dim: int,
    reference: Interval,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not matrices:
        empty = torch.empty(
            (0, int(dim), int(dim)),
            dtype=reference.dtype,
            device=reference.device,
        )
        return empty, empty.clone()
    lo = torch.stack(
        [torch.stack([torch.stack([value.lo for value in row]) for row in matrix]) for matrix in matrices]
    )
    hi = torch.stack(
        [torch.stack([torch.stack([value.hi for value in row]) for row in matrix]) for matrix in matrices]
    )
    return lo, hi


def _pack_interval_columns(
    columns: Sequence[IntervalColumn],
    *,
    dim: int,
    reference: Interval,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not columns:
        empty = torch.empty(
            (0, int(dim)),
            dtype=reference.dtype,
            device=reference.device,
        )
        return empty, empty.clone()
    return (
        torch.stack([torch.stack([value.lo for value in column]) for column in columns]),
        torch.stack([torch.stack([value.hi for value in column]) for column in columns]),
    )


def _tensor_interval_mul(
    a_lo: torch.Tensor,
    a_hi: torch.Tensor,
    b_lo: torch.Tensor,
    b_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.stack(
        (a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi),
        dim=0,
    )
    return (
        torch.nextafter(torch.amin(candidates, dim=0), torch.full_like(a_lo, -torch.inf)),
        torch.nextafter(torch.amax(candidates, dim=0), torch.full_like(a_hi, torch.inf)),
    )


def _tensor_interval_add(
    a_lo: torch.Tensor,
    a_hi: torch.Tensor,
    b_lo: torch.Tensor,
    b_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.nextafter(a_lo + b_lo, torch.full_like(a_lo, -torch.inf)),
        torch.nextafter(a_hi + b_hi, torch.full_like(a_hi, torch.inf)),
    )


def _tensorized_interval_matrix_update_and_image(
    left: IntervalMatrix,
    matrices: Sequence[IntervalMatrix],
    columns: Sequence[IntervalColumn],
    reference: Interval,
) -> tuple[tuple[IntervalMatrix, ...], IntervalColumn]:
    """Run the unchanged owner schedule in packed CPU float64 tensors.

    Matrix products are parallel only across independent owners.  The inner-k
    and owner accumulation loops retain the scalar reference order and apply
    one outward ``nextafter`` after every multiplication and addition.
    """

    dim = len(left)
    owner_count = len(matrices)
    if owner_count != len(columns):
        raise ValueError("accepted-boundary SR tensor pack owner count mismatch")
    if owner_count == 0:
        return (), tuple(_zero_interval_like_interval(reference) for _ in range(dim))
    left_lo, left_hi = _pack_interval_matrices(
        (left,), dim=dim, reference=reference
    )
    matrix_lo, matrix_hi = _pack_interval_matrices(
        matrices, dim=dim, reference=reference
    )
    updated_lo = torch.zeros_like(matrix_lo)
    updated_hi = torch.zeros_like(matrix_hi)
    for k in range(dim):
        a_lo = left_lo[0, :, k].view(1, dim, 1)
        a_hi = left_hi[0, :, k].view(1, dim, 1)
        b_lo = matrix_lo[:, k, :].view(owner_count, 1, dim)
        b_hi = matrix_hi[:, k, :].view(owner_count, 1, dim)
        product_lo, product_hi = _tensor_interval_mul(a_lo, a_hi, b_lo, b_hi)
        updated_lo, updated_hi = _tensor_interval_add(
            updated_lo,
            updated_hi,
            product_lo,
            product_hi,
        )

    column_lo, column_hi = _pack_interval_columns(
        columns, dim=dim, reference=reference
    )
    image_lo = torch.zeros((owner_count, dim), dtype=reference.dtype, device=reference.device)
    image_hi = torch.zeros_like(image_lo)
    for k in range(dim):
        product_lo, product_hi = _tensor_interval_mul(
            updated_lo[:, :, k],
            updated_hi[:, :, k],
            column_lo[:, k].view(owner_count, 1),
            column_hi[:, k].view(owner_count, 1),
        )
        image_lo, image_hi = _tensor_interval_add(
            image_lo,
            image_hi,
            product_lo,
            product_hi,
        )

    propagated_lo = torch.zeros((dim,), dtype=reference.dtype, device=reference.device)
    propagated_hi = torch.zeros_like(propagated_lo)
    for owner in range(owner_count):
        propagated_lo, propagated_hi = _tensor_interval_add(
            propagated_lo,
            propagated_hi,
            image_lo[owner],
            image_hi[owner],
        )

    updated = tuple(
        tuple(
            tuple(
                Interval(updated_lo[owner, row, col], updated_hi[owner, row, col])
                for col in range(dim)
            )
            for row in range(dim)
        )
        for owner in range(owner_count)
    )
    propagated = tuple(
        Interval(propagated_lo[row], propagated_hi[row])
        for row in range(dim)
    )
    return updated, propagated


def _real_matrix_as_intervals(matrix: RealMatrix, reference: Interval) -> IntervalMatrix:
    return tuple(
        tuple(
            Interval.point(
                torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
            )
            for value in row
        )
        for row in matrix
    )


def _right_scale_interval_matrix(
    matrix: RealMatrix,
    scalars: Sequence[Interval],
    reference: Interval,
) -> IntervalMatrix:
    point = _real_matrix_as_intervals(matrix, reference)
    return tuple(
        tuple(value * scalars[j] for j, value in enumerate(row))
        for row in point
    )


def _add_interval_columns(a: IntervalColumn, b: IntervalColumn) -> IntervalColumn:
    return tuple(x + y for x, y in zip(a, b))


def _column_width_sum(column: IntervalColumn) -> float:
    return sum(_interval_width(iv) for iv in column)


def _column_widths(column: IntervalColumn) -> list[float]:
    return [_interval_width(iv) for iv in column]


def _matrix_norm(matrix: RealMatrix) -> float:
    return sum(abs(v) for row in matrix for v in row)


def _matrix_entries(matrix: RealMatrix) -> str:
    return ";".join(",".join(f"{float(v):.17g}" for v in row) for row in matrix)


def _inverse_scales(scales: Sequence[float], dim: int) -> tuple[float, ...]:
    values: list[float] = []
    for i in range(int(dim)):
        scale = float(scales[i]) if i < len(scales) else 0.0
        values.append(1.0 if scale == 0.0 else 1.0 / scale)
    return tuple(values)


def _range_widths(tmv: TMVector) -> list[float]:
    return [_interval_width(iv) for iv in tmv.range_box()]


def _nonlinear_remainder_widths(tmv: TMVector) -> list[float]:
    widths: list[float] = []
    for model in tmv:
        nonlinear_terms = {
            exp: coeff
            for exp, coeff in model.polynomial.terms.items()
            if sum(exp) >= 2
        }
        nonlinear_poly = Polynomial(nonlinear_terms, model.n_vars)
        nonlinear_range = nonlinear_poly.evaluate_interval(model.domain) if nonlinear_terms else _zero_interval_like(model.domain)
        widths.append(_interval_width(nonlinear_range + model.remainder))
    return widths


def _set_component_width_stats(stats: dict[str, Any], prefix: str, widths: Sequence[float]) -> None:
    stats[f"{prefix}_width_x"] = widths[0] if len(widths) > 0 else ""
    stats[f"{prefix}_width_y"] = widths[1] if len(widths) > 1 else ""
    stats[f"{prefix}_width_sum"] = sum(float(w) for w in widths)


def _propagate_queue_v2(
    state: FlowstarSymbolicRemainderQueue,
    phi_l_i: RealMatrix,
    reference: Interval,
) -> tuple[tuple[RealMatrix, ...], IntervalColumn]:
    identity = _identity_matrix(state.dim)
    old_phi = tuple(state.Phi_L)
    if len(old_phi) < len(state.J):
        old_phi = old_phi + tuple(identity for _ in range(len(state.J) - len(old_phi)))
    updated_phi = tuple(_matmul_real(phi_l_i, phi) for phi in old_phi[: len(state.J)])
    propagated = tuple(_zero_interval_like_interval(reference) for _ in range(state.dim))
    for phi, column in zip(updated_phi, state.J):
        propagated = _add_interval_columns(propagated, _matmul_interval_col(phi, column, reference))
    return updated_phi, propagated


def _accepted_boundary_sr_interval_finite(value: Interval) -> bool:
    return (
        value.device.type == "cpu"
        and value.dtype == torch.float64
        and value.is_finite()
        and bool(torch.all(torch.isfinite(value.width())))
    )


def _validated_interval_bounds(
    values: Sequence[Interval],
    *,
    message: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate scalar CPU binary64 intervals with one packed tensor pass."""

    if not values:
        empty = torch.empty((0,), dtype=torch.float64)
        return empty, empty.clone()
    if any(
        value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.lo.numel() != 1
        or value.hi.numel() != 1
        for value in values
    ):
        raise FloatingPointError(message)
    lo = torch.stack([value.lo.reshape(()) for value in values])
    hi = torch.stack([value.hi.reshape(()) for value in values])
    width = torch.nextafter(hi - lo, torch.full_like(hi, torch.inf))
    if not bool(
        torch.all(torch.isfinite(lo))
        and torch.all(torch.isfinite(hi))
        and torch.all(torch.isfinite(width))
        and torch.all(lo <= hi)
    ):
        raise FloatingPointError(message)
    return lo, hi


def validate_accepted_boundary_sr_queue(
    state: FlowstarSymbolicRemainderQueue,
    *,
    expected_boundary_index: int | None = None,
) -> None:
    """Fail closed on stale, partial, nonfinite, or multiply-owned queue state."""

    if state.owner_schema not in {
        ACCEPTED_BOUNDARY_SR_OWNER_SCHEMA,
        C3_CROSS_STEP_SR_OWNER_SCHEMA,
    }:
        raise ValueError("accepted-boundary SR queue owner schema mismatch")
    if state.dim < 1 or state.max_size < 1:
        raise ValueError("accepted-boundary SR queue has invalid dimension/capacity")
    lengths = {
        len(state.J),
        len(state.Phi_L),
        len(state.Phi_L_iv),
        len(state.owner_generations),
        len(state.owner_boundary_indices),
    }
    if len(lengths) != 1:
        raise ValueError("accepted-boundary SR queue partial update: queue payload lengths disagree")
    if len(state.J) >= state.max_size:
        raise ValueError("accepted-boundary SR queue reached capacity without reset")
    if len(state.scalars) != state.dim or len(state.scalars_iv) != state.dim:
        raise ValueError("accepted-boundary SR queue scalar dimension mismatch")
    if state.generation != state.accepted_boundary_index:
        raise ValueError("accepted-boundary SR queue stale generation")
    if expected_boundary_index is not None and state.accepted_boundary_index != int(expected_boundary_index):
        raise ValueError("accepted-boundary SR queue stale accepted-boundary owner")
    if tuple(sorted(state.owner_generations)) != state.owner_generations:
        raise ValueError("accepted-boundary SR queue owner generations are not monotone")
    if len(set(state.owner_generations)) != len(state.owner_generations):
        raise ValueError("accepted-boundary SR queue duplicate generation owner")
    if tuple(sorted(state.owner_boundary_indices)) != state.owner_boundary_indices:
        raise ValueError("accepted-boundary SR queue owner boundaries are not monotone")
    if state.owner_generations != state.owner_boundary_indices:
        raise ValueError("accepted-boundary SR queue generation/boundary ownership mismatch")
    if state.owner_generations and state.owner_generations[-1] > state.generation:
        raise ValueError("accepted-boundary SR queue owner is newer than its state generation")
    scalar_points = torch.as_tensor(
        tuple(float(value) for value in state.scalars),
        dtype=torch.float64,
    )
    if not bool(torch.all(torch.isfinite(scalar_points))):
        raise FloatingPointError("accepted-boundary SR queue has nonfinite point scalar")
    scalar_lo, scalar_hi = _validated_interval_bounds(
        state.scalars_iv,
        message="accepted-boundary SR queue scalar enclosure is invalid",
    )
    if not bool(torch.all((scalar_points >= scalar_lo) & (scalar_points <= scalar_hi))):
        raise FloatingPointError("accepted-boundary SR queue scalar enclosure is invalid")

    point_phi_values: list[float] = []
    interval_phi_values: list[Interval] = []
    for point_matrix, interval_matrix in zip(state.Phi_L, state.Phi_L_iv):
        if len(point_matrix) != state.dim or len(interval_matrix) != state.dim:
            raise ValueError("accepted-boundary SR queue Phi row dimension mismatch")
        for point_row, interval_row in zip(point_matrix, interval_matrix):
            if len(point_row) != state.dim or len(interval_row) != state.dim:
                raise ValueError("accepted-boundary SR queue Phi column dimension mismatch")
            point_phi_values.extend(float(point) for point in point_row)
            interval_phi_values.extend(interval_row)
    if point_phi_values:
        point_phi = torch.as_tensor(point_phi_values, dtype=torch.float64)
        if not bool(torch.all(torch.isfinite(point_phi))):
            raise FloatingPointError("accepted-boundary SR queue has nonfinite point Phi")
        phi_lo, phi_hi = _validated_interval_bounds(
            interval_phi_values,
            message="accepted-boundary SR queue Phi enclosure is invalid",
        )
        if not bool(torch.all((point_phi >= phi_lo) & (point_phi <= phi_hi))):
            raise FloatingPointError("accepted-boundary SR queue Phi enclosure is invalid")

    j_values: list[Interval] = []
    for column in state.J:
        if len(column) != state.dim:
            raise FloatingPointError("accepted-boundary SR queue J column is invalid")
        j_values.extend(column)
    _validated_interval_bounds(
        j_values,
        message="accepted-boundary SR queue J column is invalid",
    )


def accepted_boundary_sr_queue_propagate(
    state: FlowstarSymbolicRemainderQueue,
    linear: RealMatrix,
    *,
    expected_boundary_index: int,
    reference: Interval,
    _validated: bool = False,
) -> tuple[tuple[RealMatrix, ...], tuple[IntervalMatrix, ...], IntervalColumn, dict[str, Any]]:
    """Outward-propagate existing owners without mutating accepted state."""

    if not _validated:
        validate_accepted_boundary_sr_queue(
            state,
            expected_boundary_index=expected_boundary_index,
        )
    if len(linear) != state.dim or any(len(row) != state.dim for row in linear):
        raise ValueError("accepted-boundary SR queue linear map dimension mismatch")
    if any(not torch.isfinite(torch.as_tensor(float(value), dtype=torch.float64)) for row in linear for value in row):
        raise FloatingPointError("accepted-boundary SR queue received nonfinite linear map")

    phi_i = _right_scale_matrix(linear, state.scalars)
    phi_i_iv = _right_scale_interval_matrix(linear, state.scalars_iv, reference)
    updated_point = tuple(_matmul_real(phi_i, value) for value in state.Phi_L)
    updated_interval, propagated = _tensorized_interval_matrix_update_and_image(
        phi_i_iv,
        state.Phi_L_iv,
        state.J,
        reference,
    )
    stats = {
        "queue_size_before": len(state.J),
        "generation_before": state.generation,
        "accepted_boundary_before": state.accepted_boundary_index,
        "current_linear_map_entries": _matrix_entries(linear),
        "current_phi_l_map_entries": _matrix_entries(phi_i),
        "propagated_symbolic_width_sum": _column_width_sum(propagated),
        "propagated_symbolic_width_x": _column_widths(propagated)[0],
        "propagated_symbolic_width_y": _column_widths(propagated)[1] if state.dim > 1 else "",
    }
    return updated_point, updated_interval, propagated, stats


def accepted_boundary_sr_queue_commit(
    state: FlowstarSymbolicRemainderQueue,
    updated_phi: tuple[RealMatrix, ...],
    updated_phi_iv: tuple[IntervalMatrix, ...],
    current_j: IntervalColumn,
    *,
    scales: Sequence[float],
    accepted_boundary_index: int,
    reference: Interval,
) -> tuple[FlowstarSymbolicRemainderQueue, dict[str, Any]]:
    """Atomically commit one accepted owner, then reset exactly at capacity."""

    validate_accepted_boundary_sr_queue(
        state,
        expected_boundary_index=int(accepted_boundary_index) - 1,
    )
    if len(updated_phi) != len(state.J) or len(updated_phi_iv) != len(state.J):
        raise ValueError("accepted-boundary SR queue propagation/commit partial update")
    if len(current_j) != state.dim or not all(
        _accepted_boundary_sr_interval_finite(value) for value in current_j
    ):
        raise FloatingPointError("accepted-boundary SR queue refuses nonfinite current owner")
    if len(scales) != state.dim:
        raise ValueError("accepted-boundary SR queue scale dimension mismatch")
    scale_points = tuple(float(value) for value in scales)
    if any(
        value < 0.0
        or not torch.isfinite(torch.as_tensor(value, dtype=torch.float64))
        for value in scale_points
    ):
        raise FloatingPointError("accepted-boundary SR queue refuses invalid normalization scale")
    inverse_point = tuple(0.0 if value == 0.0 else 1.0 / value for value in scale_points)
    inverse_iv = tuple(
        Interval.zero(dtype=reference.dtype, device=reference.device)
        if value == 0.0
        else Interval.point(
            torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        ).reciprocal()
        for value in scale_points
    )
    for point, enclosure in zip(inverse_point, inverse_iv):
        if not enclosure.contains(point):
            raise AssertionError("accepted-boundary SR inverse scale lost its point representative")

    identity = _identity_matrix(state.dim)
    identity_iv = _identity_interval_matrix(state.dim, reference)
    next_generation = state.generation + 1
    if next_generation != int(accepted_boundary_index):
        raise ValueError("accepted-boundary SR queue generation did not advance exactly once")
    pending_j = state.J + (tuple(current_j),)
    pending_phi = tuple(updated_phi) + (identity,)
    pending_phi_iv = tuple(updated_phi_iv) + (identity_iv,)
    pending_owner_generations = state.owner_generations + (next_generation,)
    pending_owner_boundaries = state.owner_boundary_indices + (int(accepted_boundary_index),)
    queue_reset = len(pending_j) >= state.max_size
    if queue_reset:
        next_state = FlowstarSymbolicRemainderQueue.empty_accepted_boundary_sr(
            state.dim,
            state.max_size,
            accepted_boundary_index=int(accepted_boundary_index),
            generation=next_generation,
            reset_count=state.reset_count + 1,
            reference=reference,
            owner_schema=state.owner_schema,
        )
    else:
        next_state = FlowstarSymbolicRemainderQueue(
            pending_j,
            pending_phi,
            inverse_point,
            state.max_size,
            Phi_L_iv=pending_phi_iv,
            scalars_iv=inverse_iv,
            generation=next_generation,
            accepted_boundary_index=int(accepted_boundary_index),
            owner_generations=pending_owner_generations,
            owner_boundary_indices=pending_owner_boundaries,
            reset_count=state.reset_count,
            owner_schema=state.owner_schema,
        )
        validate_accepted_boundary_sr_queue(
            next_state,
            expected_boundary_index=accepted_boundary_index,
        )
    return next_state, {
        "queue_size_after": len(next_state.J),
        "queue_size_before_reset": len(pending_j),
        "queue_reset": queue_reset,
        "generation_after": next_state.generation,
        "accepted_boundary_after": next_state.accepted_boundary_index,
        "reset_count": next_state.reset_count,
        "current_owner_generation": next_generation,
        "current_owner_boundary": int(accepted_boundary_index),
        "current_owner_width_sum": _column_width_sum(tuple(current_j)),
    }


def accepted_boundary_sr_queue_sha256(state: FlowstarSymbolicRemainderQueue) -> str:
    """Canonical binary64/interval fingerprint for ledgers and checkpoints."""

    validate_accepted_boundary_sr_queue(state)
    payload = {
        "max_size": state.max_size,
        "generation": state.generation,
        "accepted_boundary_index": state.accepted_boundary_index,
        "reset_count": state.reset_count,
        "owners": list(zip(state.owner_generations, state.owner_boundary_indices)),
        "scalars": [float(value).hex() for value in state.scalars],
        "scalars_iv": [
            [float(value.lo.detach().cpu()).hex(), float(value.hi.detach().cpu()).hex()]
            for value in state.scalars_iv
        ],
        "J": [
            [
                [float(value.lo.detach().cpu()).hex(), float(value.hi.detach().cpu()).hex()]
                for value in column
            ]
            for column in state.J
        ],
        "Phi_L": [[[float(value).hex() for value in row] for row in matrix] for matrix in state.Phi_L],
        "Phi_L_iv": [
            [
                [
                    [float(value.lo.detach().cpu()).hex(), float(value.hi.detach().cpu()).hex()]
                    for value in row
                ]
                for row in matrix
            ]
            for matrix in state.Phi_L_iv
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def validate_c3_symbolic_queue(
    state: FlowstarSymbolicRemainderQueue,
    *,
    expected_boundary_index: int | None = None,
) -> None:
    """Frozen C3 wrapper over the generic accepted-boundary queue validator."""

    if state.owner_schema != C3_CROSS_STEP_SR_OWNER_SCHEMA:
        raise ValueError("C3 symbolic queue owner schema mismatch")
    validate_accepted_boundary_sr_queue(
        state,
        expected_boundary_index=expected_boundary_index,
    )


def c3_symbolic_queue_propagate(
    state: FlowstarSymbolicRemainderQueue,
    linear: RealMatrix,
    *,
    expected_boundary_index: int,
    reference: Interval,
) -> tuple[tuple[RealMatrix, ...], tuple[IntervalMatrix, ...], IntervalColumn, dict[str, Any]]:
    """Frozen C3 wrapper over generic outward owner propagation."""

    if state.owner_schema != C3_CROSS_STEP_SR_OWNER_SCHEMA:
        raise ValueError("C3 symbolic queue owner schema mismatch")
    return accepted_boundary_sr_queue_propagate(
        state,
        linear,
        expected_boundary_index=expected_boundary_index,
        reference=reference,
    )


def c3_symbolic_queue_commit(
    state: FlowstarSymbolicRemainderQueue,
    updated_phi: tuple[RealMatrix, ...],
    updated_phi_iv: tuple[IntervalMatrix, ...],
    current_j: IntervalColumn,
    *,
    scales: Sequence[float],
    accepted_boundary_index: int,
    reference: Interval,
) -> tuple[FlowstarSymbolicRemainderQueue, dict[str, Any]]:
    """Frozen C3 wrapper over the generic accepted-only atomic commit."""

    if state.owner_schema != C3_CROSS_STEP_SR_OWNER_SCHEMA:
        raise ValueError("C3 symbolic queue owner schema mismatch")
    return accepted_boundary_sr_queue_commit(
        state,
        updated_phi,
        updated_phi_iv,
        current_j,
        scales=scales,
        accepted_boundary_index=accepted_boundary_index,
        reference=reference,
    )


def c3_symbolic_queue_sha256(state: FlowstarSymbolicRemainderQueue) -> str:
    """Frozen C3 wrapper over the generic queue fingerprint."""

    if state.owner_schema != C3_CROSS_STEP_SR_OWNER_SCHEMA:
        raise ValueError("C3 symbolic queue owner schema mismatch")
    return accepted_boundary_sr_queue_sha256(state)


def _updated_phi_and_propagated_remainder(
    state: FlowstarSymbolicRemainderQueue,
    phi_l_i: RealMatrix,
    reference: Interval,
) -> tuple[tuple[RealMatrix, ...], IntervalColumn]:
    updated_phi = list(state.Phi_L)
    for i in range(1, len(updated_phi)):
        updated_phi[i] = _matmul_real(phi_l_i, updated_phi[i])
    updated_phi.append(phi_l_i)

    propagated = tuple(_zero_interval_like_interval(reference) for _ in range(state.dim))
    for i in range(1, len(updated_phi)):
        if i - 1 >= len(state.J):
            break
        propagated = _add_interval_columns(propagated, _matmul_interval_col(updated_phi[i], state.J[i - 1], reference))
    return tuple(updated_phi), propagated


def flowstar_symbolic_remainder_queue_reset(
    tm: TMVector,
    state: FlowstarSymbolicRemainderQueue | None,
    *,
    max_size: int = 100,
) -> tuple[TMVector, FlowstarSymbolicRemainderQueue, dict[str, Any]]:
    """Propagate endpoint remainders through a Flow*-style linear queue.

    The helper is intentionally conservative: it preserves the endpoint
    polynomial dependency and replaces the ordinary remainder by the current
    endpoint remainder plus the queued linear propagation of older remainders.
    """

    dim = len(tm)
    if dim == 0:
        empty = FlowstarSymbolicRemainderQueue.empty(0, max_size)
        return tm, empty, {"active_queue_size": 0, "queue_reset": False}
    if state is None or state.dim != dim or int(state.max_size) != int(max_size):
        state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)

    reference = tm[0].remainder
    linear = _linear_coefficients(tm)
    phi_l_i = _right_scale_matrix(linear, state.scalars)
    updated_phi, propagated = _updated_phi_and_propagated_remainder(state, phi_l_i, reference)
    current_j = tuple(model.remainder for model in tm)
    total_remainders = _add_interval_columns(current_j, propagated)
    reset_tm = TMVector(model.with_remainder(rem) for model, rem in zip(tm, total_remainders))

    widths = reset_tm.range_box()
    scalars: list[float] = []
    for box in widths:
        mag = max(abs(float(box.lo.detach().cpu())), abs(float(box.hi.detach().cpu())))
        scalars.append(0.0 if mag == 0 else 1.0 / mag)

    new_j = state.J + (current_j,)
    queue_reset = bool(int(max_size) > 0 and len(new_j) >= int(max_size))
    if queue_reset:
        new_state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)
    else:
        new_state = FlowstarSymbolicRemainderQueue(tuple(new_j), updated_phi, tuple(scalars), int(max_size))

    stats = {
        "queue_size_before": len(state.J),
        "queue_size_after": len(new_state.J),
        "queue_reset": queue_reset,
        "current_remainder_width_sum": _column_width_sum(current_j),
        "propagated_remainder_width_sum": _column_width_sum(propagated),
        "total_remainder_width_sum": _column_width_sum(total_remainders),
        "linear_map_abs_sum": sum(abs(v) for row in linear for v in row),
    }
    return reset_tm, new_state, stats


def flowstar_normalized_insertion_symbolic_queue_reset(
    inserted_endpoint: TMVector,
    reset_tm: TMVector,
    state: FlowstarSymbolicRemainderQueue | None,
    *,
    scales: Sequence[float],
    max_size: int = 100,
    materialize_propagated_on_reset: bool = True,
) -> tuple[TMVector, FlowstarSymbolicRemainderQueue, dict[str, Any]]:
    """Conservative symbolic queue update after normalized insertion.

    ``inserted_endpoint`` is the accepted nonconstant endpoint after Flow*-style
    normal insertion. ``reset_tm`` is the fresh normalized local initial set for
    the next step. The helper propagates older queued interval columns through
    the linear part of ``inserted_endpoint``. Existing symqueue mode materializes
    that propagated width on ``reset_tm``; split mode returns a clean ordinary
    reset and exposes the propagated width for output/range materialization.
    """

    dim = len(reset_tm)
    if dim == 0:
        empty = FlowstarSymbolicRemainderQueue.empty(0, max_size)
        return reset_tm, empty, {"queue_size": 0, "queue_reset": False}
    if state is None or state.dim != dim or int(state.max_size) != int(max_size):
        state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)

    reference = reset_tm[0].remainder
    linear = _linear_coefficients(inserted_endpoint)
    phi_l_i = _right_scale_matrix(linear, state.scalars)
    updated_phi, propagated = _updated_phi_and_propagated_remainder(state, phi_l_i, reference)
    current_j = tuple(model.remainder for model in inserted_endpoint)

    if materialize_propagated_on_reset:
        reset_with_queue = TMVector(
            model.with_remainder(model.remainder + propagated_i)
            for model, propagated_i in zip(reset_tm, propagated)
        )
        output_symbolic_remainders = tuple(_zero_interval_like_interval(reference) for _ in range(dim))
    else:
        reset_with_queue = reset_tm
        output_symbolic_remainders = propagated

    new_j = state.J + (current_j,)
    queue_reset = bool(int(max_size) > 0 and len(new_j) >= int(max_size))
    if queue_reset:
        new_state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)
    else:
        scalar_tuple = tuple(float(s) for s in scales[:dim])
        if len(scalar_tuple) < dim:
            scalar_tuple = scalar_tuple + tuple(1.0 for _ in range(dim - len(scalar_tuple)))
        new_state = FlowstarSymbolicRemainderQueue(tuple(new_j), updated_phi, scalar_tuple, int(max_size))

    propagated_widths = [_interval_width(iv) for iv in propagated]
    new_widths = [_interval_width(iv) for iv in current_j]
    ordinary_remainder_widths = [_interval_width(model.remainder) for model in reset_tm]
    materialized_widths = (
        [_interval_width(model.remainder) for model in reset_with_queue]
        if materialize_propagated_on_reset
        else propagated_widths
    )
    ordinary_only_range_width = sum(_interval_width(iv) for iv in reset_tm.range_box())
    total_with_symbolic_tm = TMVector(
        model.with_remainder(model.remainder + rem)
        for model, rem in zip(reset_with_queue, output_symbolic_remainders)
    )
    total_range_width_with_symbolic = sum(_interval_width(iv) for iv in total_with_symbolic_tm.range_box())
    linear_norm = sum(abs(v) for row in linear for v in row)
    stats = {
        "queue_size_before": len(state.J),
        "queue_size_after": len(new_state.J),
        "queue_size": len(new_state.J),
        "queue_reset": queue_reset,
        "semantic_split": not materialize_propagated_on_reset,
        "propagated_symbolic_width_x": propagated_widths[0] if len(propagated_widths) > 0 else "",
        "propagated_symbolic_width_y": propagated_widths[1] if len(propagated_widths) > 1 else "",
        "propagated_symbolic_width_sum": sum(propagated_widths),
        "new_symbolic_width_x": new_widths[0] if len(new_widths) > 0 else "",
        "new_symbolic_width_y": new_widths[1] if len(new_widths) > 1 else "",
        "new_symbolic_width_sum": sum(new_widths),
        "materialized_width_x": materialized_widths[0] if len(materialized_widths) > 0 else "",
        "materialized_width_y": materialized_widths[1] if len(materialized_widths) > 1 else "",
        "materialized_width_sum": sum(materialized_widths),
        "materialized_for_output_width": sum(materialized_widths),
        "ordinary_only_range_width": ordinary_only_range_width,
        "symbolic_contribution_width": sum(propagated_widths),
        "total_range_width_with_symbolic": total_range_width_with_symbolic,
        "target_checked_width": sum(ordinary_remainder_widths),
        "linear_map_norm": linear_norm,
        "linear_map_abs_sum": linear_norm,
        "scalars": ";".join(f"{float(s):.17g}" for s in scales[:dim]),
        "_symbolic_output_remainders": output_symbolic_remainders,
        "approximation": (
            "limited_normalized_insertion_symqueue_split; current insertion uncertainty "
            "is queued for future propagation and propagated old queue width is "
            "materialized for output/range only"
            if not materialize_propagated_on_reset
            else
            "limited_normalized_insertion_symqueue; current insertion uncertainty "
            "is queued for future propagation and propagated old queue width is "
            "materialized on the next normalized reset"
        ),
    }
    return reset_with_queue, new_state, stats



def flowstar_normalized_insertion_linear_queue_v2_reset(
    inserted_endpoint: TMVector,
    reset_tm: TMVector,
    state: FlowstarSymbolicRemainderQueue | None,
    *,
    scales: Sequence[float],
    max_size: int = 100,
    target_remainder_radius: float | None = None,
) -> tuple[TMVector, FlowstarSymbolicRemainderQueue, dict[str, Any]]:
    """Flow*-style linear symbolic queue v2 for normalized insertion.

    This remains a clean-room approximation. The ordinary normalized reset is
    left target-clean; propagated old symbolic queue width is returned as an
    output/range-only contribution, and the current insertion remainder is
    queued for future linear propagation.
    """

    dim = len(reset_tm)
    if dim == 0:
        empty = FlowstarSymbolicRemainderQueue.empty(0, max_size)
        return reset_tm, empty, {"queue_size": 0, "queue_reset": False, "symbolic_queue_mode": "flowstar_linear_v2"}
    if state is None or state.dim != dim or int(state.max_size) != int(max_size):
        state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)

    reference = reset_tm[0].remainder
    current_linear = _linear_coefficients(inserted_endpoint)
    phi_l_i = _right_scale_matrix(current_linear, state.scalars)
    updated_phi, propagated = _propagate_queue_v2(state, phi_l_i, reference)
    current_j = tuple(model.remainder for model in inserted_endpoint)
    inverse_scale_tuple = _inverse_scales(scales, dim)

    new_j = state.J + (current_j,)
    new_phi = updated_phi + (_identity_matrix(dim),)
    queue_reset = bool(int(max_size) > 0 and len(new_j) >= int(max_size))
    if queue_reset:
        new_state = FlowstarSymbolicRemainderQueue.empty(dim, max_size)
    else:
        new_state = FlowstarSymbolicRemainderQueue(tuple(new_j), tuple(new_phi), inverse_scale_tuple, int(max_size))

    propagated_widths = _column_widths(propagated)
    new_widths = _column_widths(current_j)
    target_widths = _column_widths(tuple(model.remainder for model in reset_tm))
    right_map_widths = _range_widths(inserted_endpoint)
    reset_widths = _range_widths(reset_tm)
    nonlinear_widths = _nonlinear_remainder_widths(inserted_endpoint)
    total_with_symbolic_tm = TMVector(
        model.with_remainder(model.remainder + rem)
        for model, rem in zip(reset_tm, propagated)
    )
    ordinary_only_range_width = sum(reset_widths)
    total_range_width_with_symbolic = sum(_range_widths(total_with_symbolic_tm))
    target_radius = float(target_remainder_radius) if target_remainder_radius is not None else None
    target_exceeds = (
        any(width > target_radius + 1e-15 for width in target_widths)
        if target_radius is not None
        else False
    )
    output_includes_symbolic = (
        total_range_width_with_symbolic + 1e-15 >= ordinary_only_range_width
        and total_range_width_with_symbolic + 1e-15 >= sum(propagated_widths)
    )

    stats: dict[str, Any] = {
        "queue_size_before": len(state.J),
        "queue_size_after": len(new_state.J),
        "queue_size": len(new_state.J),
        "queue_reset": queue_reset,
        "semantic_split": True,
        "symbolic_queue_mode": "flowstar_linear_v2",
        "j_count": len(new_state.J),
        "phi_l_count": len(new_state.Phi_L),
        "current_linear_map_entries": _matrix_entries(current_linear),
        "current_linear_map_norm": _matrix_norm(current_linear),
        "current_phi_l_map_entries": _matrix_entries(phi_l_i),
        "current_phi_l_map_norm": _matrix_norm(phi_l_i),
        "linear_map_norm": _matrix_norm(current_linear),
        "linear_map_abs_sum": _matrix_norm(current_linear),
        "scalars": ";".join(f"{float(s):.17g}" for s in inverse_scale_tuple),
        "scalar_x": inverse_scale_tuple[0] if len(inverse_scale_tuple) > 0 else "",
        "scalar_y": inverse_scale_tuple[1] if len(inverse_scale_tuple) > 1 else "",
        "ordinary_only_range_width": ordinary_only_range_width,
        "symbolic_contribution_width": sum(propagated_widths),
        "total_range_width_with_symbolic": total_range_width_with_symbolic,
        "target_checked_width": sum(target_widths),
        "target_check_exceeds_target": target_exceeds,
        "output_range_includes_symbolic_contributions": output_includes_symbolic,
        "materialized_for_output_width": sum(propagated_widths),
        "conservative": output_includes_symbolic,
        "_symbolic_output_remainders": propagated,
        "approximation": (
            "flowstar_linear_v2; old interval columns are propagated through "
            "degree-1 normalized-insertion maps with inverse reset scalars, "
            "while nonlinear polynomial terms and ordinary local remainders stay outside the queue"
        ),
    }
    _set_component_width_stats(stats, "propagated_symbolic", propagated_widths)
    _set_component_width_stats(stats, "new_symbolic", new_widths)
    _set_component_width_stats(stats, "materialized", propagated_widths)
    _set_component_width_stats(stats, "ordinary_step_remainder", new_widths)
    _set_component_width_stats(stats, "current_nonlinear_remainder", nonlinear_widths)
    _set_component_width_stats(stats, "reset_box", reset_widths)
    _set_component_width_stats(stats, "right_map_range", right_map_widths)
    _set_component_width_stats(stats, "target_check", target_widths)
    _set_component_width_stats(stats, "output_only_symbolic", propagated_widths)
    return reset_tm, new_state, stats


def _interval_width(iv: Interval) -> float:
    return float(iv.width().detach().cpu())


def _zero_remainder_model(model: TaylorModel) -> TaylorModel:
    return TaylorModel(model.polynomial, _zero_interval_like(model.domain), list(model.domain), order=model.order)


def _with_polynomial(model: TaylorModel, polynomial: Polynomial) -> TaylorModel:
    return TaylorModel(polynomial, model.remainder, list(model.domain), order=model.order)


def split_polynomial_by_variables(
    polynomial: Polynomial,
    variable_indices: Iterable[int],
) -> tuple[Polynomial, Polynomial]:
    """Split a polynomial into terms independent of and dependent on variables."""

    indices = set(int(i) for i in variable_indices)
    independent: dict[tuple[int, ...], Any] = {}
    dependent: dict[tuple[int, ...], Any] = {}
    for exp, coeff in polynomial.terms.items():
        target = dependent if any(exp[i] for i in indices if i < len(exp)) else independent
        target[exp] = coeff
    return Polynomial(independent, polynomial.n_vars), Polynomial(dependent, polynomial.n_vars)


def symbolic_remainder_widths(tm: TMVector, state: SymbolicRemainderState) -> list[float]:
    indices = state.active_var_indices()
    if not indices:
        return [0.0 for _ in tm]
    widths: list[float] = []
    for model in tm:
        _plain, symbolic = split_polynomial_by_variables(model.polynomial, indices)
        widths.append(_interval_width(symbolic.evaluate_interval(model.domain)))
    return widths


def ordinary_remainder_widths(tm: TMVector) -> list[float]:
    return [_interval_width(model.remainder) for model in tm]


def _drop_one_variable(
    tm: TMVector,
    state: SymbolicRemainderState,
    var_index: int,
    *,
    remove_symbol_id: int | None = None,
) -> tuple[TMVector, SymbolicRemainderState, list[float]]:
    new_models: list[TaylorModel] = []
    materialized_widths: list[float] = []
    for model in tm:
        kept: dict[tuple[int, ...], Any] = {}
        dropped: dict[tuple[int, ...], Any] = {}
        for exp, coeff in model.polynomial.terms.items():
            if exp[var_index]:
                dropped[exp] = coeff
            else:
                kept[exp] = coeff

        dropped_poly = Polynomial(dropped, model.n_vars)
        contribution = dropped_poly.evaluate_interval(model.domain) if dropped else _zero_interval_like(model.domain)
        kept_poly = Polynomial(kept, model.n_vars).drop_variable(var_index, require_zero_exponent=True)
        new_domain = [dom for i, dom in enumerate(model.domain) if i != var_index]
        new_models.append(TaylorModel(kept_poly, model.remainder + contribution, new_domain, order=model.order))
        materialized_widths.append(_interval_width(contribution))

    new_symbols: list[SymbolicNoiseSymbol] = []
    for symbol in state.symbols:
        if remove_symbol_id is not None and symbol.symbol_id == remove_symbol_id:
            continue
        if symbol.var_index == var_index:
            continue
        if symbol.var_index > var_index:
            new_symbols.append(replace(symbol, var_index=symbol.var_index - 1))
        else:
            new_symbols.append(symbol)

    return (
        TMVector(new_models),
        replace(state, symbols=tuple(new_symbols)),
        materialized_widths,
    )


def materialize_oldest_symbols(
    tm: TMVector,
    state: SymbolicRemainderState,
    max_symbolic_remainders: int,
) -> tuple[TMVector, SymbolicRemainderState, dict[str, Any]]:
    """Materialize oldest queue entries until at most max symbols remain."""

    max_count = max(0, int(max_symbolic_remainders))
    state = state.with_queue_size(max_count)
    materialized_symbol_ids: list[int] = []
    width_sums = [0.0 for _ in tm]
    current_tm = tm
    current_state = state
    while len(current_state.symbols) > max_count:
        oldest = current_state.symbols[0]
        current_tm, current_state, widths = _drop_one_variable(
            current_tm,
            current_state,
            oldest.var_index,
            remove_symbol_id=oldest.symbol_id,
        )
        materialized_symbol_ids.append(oldest.symbol_id)
        width_sums = [a + b for a, b in zip(width_sums, widths)]

    return (
        current_tm,
        current_state,
        {
            "materialized_symbol_ids": tuple(materialized_symbol_ids),
            "materialized_remainder_widths": tuple(width_sums),
            "materialized_remainder_width_sum": sum(width_sums),
        },
    )


def materialize_non_symbolic_variables(
    tm: TMVector,
    state: SymbolicRemainderState,
) -> tuple[TMVector, SymbolicRemainderState, dict[str, Any]]:
    """Materialize all variables that are not tracked noise symbols."""

    noise_indices = set(state.active_var_indices())
    current_tm = tm
    current_state = state
    width_sums = [0.0 for _ in tm]
    dropped_indices: list[int] = []
    for var_index in sorted((i for i in range(tm.n_vars) if i not in noise_indices), reverse=True):
        current_tm, current_state, widths = _drop_one_variable(current_tm, current_state, var_index)
        dropped_indices.append(var_index)
        width_sums = [a + b for a, b in zip(width_sums, widths)]
    return (
        current_tm,
        current_state,
        {
            "materialized_variable_indices": tuple(dropped_indices),
            "materialized_remainder_widths": tuple(width_sums),
            "materialized_remainder_width_sum": sum(width_sums),
        },
    )


def materialize_all_symbols(tm: TMVector, state: SymbolicRemainderState) -> TMVector:
    materialized, _state, _stats = materialize_oldest_symbols(tm, state, 0)
    return materialized


def introduce_symbolic_remainders(
    tm: TMVector,
    state: SymbolicRemainderState | None,
    *,
    max_symbolic_remainders: int,
    source: str = "picard_residual",
) -> tuple[TMVector, SymbolicRemainderState, dict[str, Any]]:
    """Move each component's interval remainder into a fresh noise symbol."""

    if state is None:
        state = SymbolicRemainderState.empty(max_symbolic_remainders)
    else:
        state = state.with_queue_size(max_symbolic_remainders)

    remainders = [model.remainder for model in tm]
    models = [_zero_remainder_model(model) for model in tm]
    current_symbols = list(state.symbols)
    next_symbol_id = state.next_symbol_id
    introduced: list[int] = []

    for state_dim, remainder in enumerate(remainders):
        eps_index = models[0].n_vars if models else 0
        models = [model.extend_domain(symbolic_noise_domain()) for model in models]
        n_vars = models[0].n_vars if models else 0
        center = remainder.mid()
        radius = remainder.radius()
        residual_poly = (
            Polynomial.constant(center, n_vars)
            + Polynomial.variable(eps_index, n_vars, dtype=radius.dtype, device=radius.device) * radius
        )
        models[state_dim] = _with_polynomial(models[state_dim], models[state_dim].polynomial + residual_poly)
        current_symbols.append(SymbolicNoiseSymbol(next_symbol_id, eps_index, state_dim, source))
        introduced.append(next_symbol_id)
        next_symbol_id += 1

    current_tm = TMVector(models)
    current_state = SymbolicRemainderState(tuple(current_symbols), next_symbol_id, int(max_symbolic_remainders))
    current_tm, current_state, materialized_stats = materialize_oldest_symbols(
        current_tm,
        current_state,
        max_symbolic_remainders,
    )
    symbolic_width = symbolic_remainder_widths(current_tm, current_state)
    ordinary_width = ordinary_remainder_widths(current_tm)
    stats = {
        **materialized_stats,
        "introduced_symbol_ids": tuple(int(i) for i in introduced),
        "introduced_symbols": len(introduced),
        "active_noise_symbols": len(current_state.symbols),
        "symbolic_remainder_widths": tuple(symbolic_width),
        "symbolic_remainder_width_sum": sum(symbolic_width),
        "ordinary_remainder_widths": tuple(ordinary_width),
        "ordinary_remainder_width_sum": sum(ordinary_width),
    }
    return current_tm, current_state, stats


def _merge_states(a: SymbolicRemainderState, b: SymbolicRemainderState) -> SymbolicRemainderState:
    by_id = {symbol.symbol_id: symbol for symbol in a.symbols}
    for symbol in b.symbols:
        by_id.setdefault(symbol.symbol_id, symbol)
    symbols = tuple(sorted(by_id.values(), key=lambda symbol: symbol.symbol_id))
    return SymbolicRemainderState(
        symbols,
        max(a.next_symbol_id, b.next_symbol_id),
        max(a.max_symbolic_remainders, b.max_symbolic_remainders),
    )


@dataclass(frozen=True)
class SymbolicTaylorModel:
    """Small wrapper around a TaylorModel that carries noise metadata."""

    base: TaylorModel
    state: SymbolicRemainderState = SymbolicRemainderState()

    @property
    def symbolic_remainder_terms(self) -> tuple[SymbolicNoiseSymbol, ...]:
        return self.state.symbols

    @property
    def noise_domains(self) -> Mapping[int, Interval]:
        return {symbol.var_index: self.base.domain[symbol.var_index] for symbol in self.state.symbols}

    @property
    def max_symbolic_remainders(self) -> int:
        return self.state.max_symbolic_remainders

    def _coerce(self, other: Any) -> tuple[TaylorModel, SymbolicRemainderState]:
        if isinstance(other, SymbolicTaylorModel):
            return other.base, _merge_states(self.state, other.state)
        return self.base._coerce(other), self.state

    def __add__(self, other: Any) -> "SymbolicTaylorModel":
        other_base, state = self._coerce(other)
        return SymbolicTaylorModel(self.base + other_base, state)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "SymbolicTaylorModel":
        other_base, state = self._coerce(other)
        return SymbolicTaylorModel(self.base - other_base, state)

    def __rsub__(self, other: Any) -> "SymbolicTaylorModel":
        other_base, state = self._coerce(other)
        return SymbolicTaylorModel(other_base - self.base, state)

    def __neg__(self) -> "SymbolicTaylorModel":
        return SymbolicTaylorModel(-self.base, self.state)

    def __mul__(self, other: Any) -> "SymbolicTaylorModel":
        other_base, state = self._coerce(other)
        return SymbolicTaylorModel(self.base * other_base, state)

    __rmul__ = __mul__

    def range_box(self) -> Interval:
        return self.base.range_box()

    def materialize(self) -> TaylorModel:
        return materialize_all_symbols(TMVector([self.base]), self.state)[0]
