"""Plant-independent accepted-boundary symbolic-remainder operator.

The operator knows nothing about an ODE or a scheduler.  It consumes only an
accepted endpoint Taylor map, its linear/nonlinear split, the previous
normalized right map, normalization scales, explicit owner metadata, and the
queue state.  Callers retain responsibility for deciding that a step is
accepted; this module provides an immutable prepare phase and one atomic commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .interval import Interval
from .polynomial import Polynomial
from .symbolic_remainder import (
    ACCEPTED_BOUNDARY_SR_OWNER_SCHEMA,
    FlowstarSymbolicRemainderQueue,
    IntervalColumn,
    IntervalMatrix,
    RealMatrix,
    accepted_boundary_sr_queue_commit,
    accepted_boundary_sr_queue_propagate,
    validate_accepted_boundary_sr_queue,
)
from .taylor_model import TaylorModel
from .tm_vector import TMVector


CompositionOperator = Callable[
    [TMVector, TMVector, int, float | None, Sequence[Interval], dict[str, Any]],
    TaylorModel | TMVector,
]


@dataclass(frozen=True)
class AcceptedBoundarySRPrepared:
    """Immutable shadow transition produced before the accepted-only commit."""

    inserted: TMVector
    current_owner: IntervalColumn
    propagated_history: IntervalColumn
    updated_phi: tuple[RealMatrix, ...]
    updated_phi_iv: tuple[IntervalMatrix, ...]
    queue_before: FlowstarSymbolicRemainderQueue
    accepted_boundary_index: int
    composition_branch: str
    stats: Mapping[str, Any]


@dataclass(frozen=True)
class AcceptedBoundarySRCommitted:
    """Normalized right map and queue state after one atomic accepted commit."""

    normalized_right_map: TMVector
    queue_after: FlowstarSymbolicRemainderQueue
    current_owner: IntervalColumn
    unscaled_roundoff_cutoff_owner: IntervalColumn
    stats: Mapping[str, Any]


def _interval_width(value: Interval) -> float:
    return float(value.width().detach().cpu())


def _validate_cpu_float64_finite(tmv: TMVector, label: str) -> None:
    if not tmv:
        raise ValueError(f"accepted-boundary SR {label} must be nonempty")
    for component, model in enumerate(tmv):
        if model.remainder.device.type != "cpu" or model.remainder.dtype != torch.float64:
            raise ValueError(
                f"accepted-boundary SR {label}[{component}] must use CPU float64 intervals"
            )
        if not model.remainder.is_finite():
            raise FloatingPointError(
                f"accepted-boundary SR {label}[{component}] has nonfinite remainder"
            )
        for variable, interval in enumerate(model.domain):
            if interval.device.type != "cpu" or interval.dtype != torch.float64:
                raise ValueError(
                    f"accepted-boundary SR {label}[{component}] domain[{variable}] "
                    "must use CPU float64 intervals"
                )
            if not interval.is_finite():
                raise FloatingPointError(
                    f"accepted-boundary SR {label}[{component}] has nonfinite domain"
                )
        for exponent, coefficient in model.polynomial.terms.items():
            if coefficient.device.type != "cpu" or coefficient.dtype != torch.float64:
                raise ValueError(
                    f"accepted-boundary SR {label}[{component}] coefficient {exponent} "
                    "must use CPU float64"
                )
            if not bool(torch.all(torch.isfinite(coefficient))):
                raise FloatingPointError(
                    f"accepted-boundary SR {label}[{component}] has nonfinite coefficient"
                )


def _interval_polynomial_range(
    coefficient_intervals: Mapping[tuple[int, ...], Interval],
    domain: Sequence[Interval],
    *,
    reference: Interval,
) -> Interval:
    total = Interval.zero(dtype=reference.dtype, device=reference.device)
    for exponent in sorted(coefficient_intervals):
        term = coefficient_intervals[exponent]
        for variable, power in zip(domain, exponent):
            if power:
                term = term * variable.pow_int(int(power))
        total = total + term
    return total


def split_endpoint_taylor_map(outer: TMVector) -> tuple[RealMatrix, TMVector]:
    """Split ``A*r`` from a constant-free accepted endpoint Taylor map."""

    dim = len(outer)
    if dim < 1:
        raise ValueError("accepted-boundary SR endpoint map must be nonempty")
    matrix: list[tuple[float, ...]] = []
    nonlinear_models: list[TaylorModel] = []
    for model in outer:
        if model.n_vars != dim:
            raise ValueError("accepted-boundary SR endpoint map must be square")
        row = [0.0 for _ in range(dim)]
        nonlinear_terms: dict[tuple[int, ...], torch.Tensor] = {}
        for exponent, coefficient in model.polynomial.terms.items():
            if sum(exponent) == 0 and bool(torch.any(coefficient != 0.0)):
                raise ValueError(
                    "accepted-boundary SR endpoint map must be constant-free"
                )
            if sum(exponent) == 1:
                variable = next(
                    (index for index, power in enumerate(exponent) if power == 1),
                    None,
                )
                if variable is not None and variable < dim:
                    row[variable] = float(coefficient.detach().cpu())
                    continue
            nonlinear_terms[exponent] = coefficient
        matrix.append(tuple(row))
        nonlinear_models.append(
            TaylorModel(
                Polynomial(nonlinear_terms, model.n_vars),
                model.remainder,
                list(model.domain),
                order=model.order,
                truncation_range_split=model.truncation_range_split,
            )
        )
    return tuple(matrix), TMVector(nonlinear_models)


def _linear_polynomial_image(
    linear: RealMatrix,
    inner: TMVector,
) -> TMVector:
    """Apply ``A`` to polynomials and outward-charge coefficient roundoff."""

    if any(model.polynomial.device.type != "cpu" for model in inner):
        raise ValueError(
            "accepted-boundary SR is CPU-authoritative and rejects non-CPU polynomial state"
        )
    dim = len(inner)
    if len(linear) != dim or any(len(row) != dim for row in linear):
        raise ValueError("accepted-boundary SR linear split dimension mismatch")
    reference = inner[0].remainder
    models: list[TaylorModel] = []
    for output_index in range(dim):
        point_terms: dict[tuple[int, ...], torch.Tensor] = {}
        exact_terms: dict[tuple[int, ...], Interval] = {}
        for input_index in range(dim):
            scalar = float(linear[output_index][input_index])
            scalar_tensor = torch.as_tensor(
                scalar,
                dtype=reference.dtype,
                device=reference.device,
            )
            for exponent, coefficient in inner[input_index].polynomial.terms.items():
                product = scalar_tensor * coefficient
                point_terms[exponent] = point_terms.get(
                    exponent,
                    torch.zeros_like(product),
                ) + product
                exact_product = Interval.point(scalar_tensor) * Interval.point(coefficient)
                exact_terms[exponent] = exact_terms.get(
                    exponent,
                    Interval.zero(dtype=reference.dtype, device=reference.device),
                ) + exact_product
        polynomial = Polynomial(point_terms, inner[0].n_vars)
        coefficient_errors: dict[tuple[int, ...], Interval] = {}
        for exponent in set(exact_terms) | set(polynomial.terms):
            point = polynomial.terms.get(
                exponent,
                torch.zeros((), dtype=reference.dtype, device=reference.device),
            )
            exact = exact_terms.get(
                exponent,
                Interval.zero(dtype=reference.dtype, device=reference.device),
            )
            coefficient_errors[exponent] = exact - Interval.point(point)
        error_range = _interval_polynomial_range(
            coefficient_errors,
            inner.domain,
            reference=reference,
        )
        models.append(
            TaylorModel(
                polynomial,
                error_range,
                list(inner.domain),
                order=max(model.order or 0 for model in inner),
            )
        )
    return TMVector(models)


def _add_polynomial_images(
    nonlinear: TMVector,
    linear: TMVector,
    propagated: Sequence[Interval],
) -> tuple[TMVector, IntervalColumn]:
    """Combine polynomial paths; history is not assigned to the current owner."""

    reference = nonlinear[0].remainder
    models: list[TaylorModel] = []
    current_owner: list[Interval] = []
    for nonlinear_model, linear_model, history in zip(nonlinear, linear, propagated):
        polynomial = nonlinear_model.polynomial + linear_model.polynomial
        addition_errors: dict[tuple[int, ...], Interval] = {}
        for exponent in set(nonlinear_model.polynomial.terms) | set(
            linear_model.polynomial.terms
        ):
            zero = torch.zeros((), dtype=reference.dtype, device=reference.device)
            left = nonlinear_model.polynomial.terms.get(exponent, zero)
            right = linear_model.polynomial.terms.get(exponent, zero)
            point = polynomial.terms.get(exponent, zero)
            addition_errors[exponent] = (
                Interval.point(left) + Interval.point(right) - Interval.point(point)
            )
        addition_error = _interval_polynomial_range(
            addition_errors,
            nonlinear_model.domain,
            reference=reference,
        )
        owner = nonlinear_model.remainder + linear_model.remainder + addition_error
        current_owner.append(owner)
        models.append(
            TaylorModel(
                polynomial,
                owner + history,
                list(nonlinear_model.domain),
                order=nonlinear_model.order,
                truncation_range_split=nonlinear_model.truncation_range_split,
            )
        )
    return TMVector(models), tuple(current_owner)


def _scale_and_cutoff_right_map(
    inserted: TMVector,
    scales: Sequence[float],
    cutoff_threshold: float | None,
) -> tuple[TMVector, IntervalColumn]:
    """Normalize the right map and expose newly created unscaled owners."""

    if len(scales) != len(inserted):
        raise ValueError("accepted-boundary SR normalization dimension mismatch")
    reference = inserted[0].remainder
    models: list[TaylorModel] = []
    unscaled_sources: list[Interval] = []
    for model, scale in zip(inserted, scales):
        inv_scale = 1.0 if float(scale) == 0.0 else 1.0 / float(scale)
        inv_tensor = torch.as_tensor(
            inv_scale,
            dtype=reference.dtype,
            device=reference.device,
        )
        point_terms = {
            exponent: coefficient * inv_tensor
            for exponent, coefficient in model.polynomial.terms.items()
        }
        polynomial = Polynomial(point_terms, model.n_vars)
        coefficient_errors: dict[tuple[int, ...], Interval] = {}
        for exponent, coefficient in model.polynomial.terms.items():
            exact = Interval.point(coefficient) * Interval.point(inv_tensor)
            coefficient_errors[exponent] = exact - Interval.point(
                polynomial.terms[exponent]
            )
        coefficient_error = _interval_polynomial_range(
            coefficient_errors,
            model.domain,
            reference=reference,
        )
        kept, cutoff_range = polynomial.cutoff(cutoff_threshold, model.domain)
        scaled_remainder = model.remainder * Interval.point(inv_tensor)
        added_scaled_source = coefficient_error + cutoff_range
        models.append(
            TaylorModel(
                kept,
                scaled_remainder + added_scaled_source,
                list(model.domain),
                order=model.order,
                truncation_range_split=model.truncation_range_split,
            )
        )
        scale_tensor = torch.as_tensor(
            float(scale),
            dtype=reference.dtype,
            device=reference.device,
        )
        unscaled_sources.append(added_scaled_source * Interval.point(scale_tensor))
    return TMVector(models), tuple(unscaled_sources)


def prepare_accepted_boundary_sr(
    endpoint_map_without_constants: TMVector,
    right_map: TMVector,
    *,
    domain: Sequence[Interval],
    order: int,
    cutoff_threshold: float | None,
    queue_state: FlowstarSymbolicRemainderQueue | None,
    queue_capacity: int,
    previous_accepted_boundary_index: int,
    compose: CompositionOperator,
    diagnostics: dict[str, Any],
    owner_schema: str = ACCEPTED_BOUNDARY_SR_OWNER_SCHEMA,
) -> AcceptedBoundarySRPrepared:
    """Prepare one plant-independent boundary transition without committing it."""

    if int(order) < 1:
        raise ValueError("accepted-boundary SR order must be positive")
    if int(queue_capacity) < 1:
        raise ValueError("accepted-boundary SR queue capacity must be positive")
    if cutoff_threshold is not None and (
        not torch.isfinite(torch.as_tensor(float(cutoff_threshold), dtype=torch.float64))
        or float(cutoff_threshold) < 0.0
    ):
        raise ValueError("accepted-boundary SR cutoff must be finite and nonnegative")
    if len(endpoint_map_without_constants) != len(right_map):
        raise ValueError("accepted-boundary SR endpoint/right-map dimension mismatch")
    if len(domain) != right_map.n_vars:
        raise ValueError("accepted-boundary SR composition domain dimension mismatch")
    _validate_cpu_float64_finite(
        endpoint_map_without_constants,
        "endpoint map",
    )
    _validate_cpu_float64_finite(right_map, "right map")
    reference = endpoint_map_without_constants[0].remainder
    queue = queue_state
    if queue is None:
        queue = FlowstarSymbolicRemainderQueue.empty_accepted_boundary_sr(
            len(endpoint_map_without_constants),
            queue_capacity,
            accepted_boundary_index=previous_accepted_boundary_index,
            generation=previous_accepted_boundary_index,
            reference=reference,
            owner_schema=owner_schema,
        )
    if queue.owner_schema != owner_schema:
        raise ValueError("accepted-boundary SR wrapper/queue owner schema mismatch")
    if queue.dim != len(endpoint_map_without_constants):
        raise ValueError("accepted-boundary SR queue dimension mismatch")
    if queue.max_size != int(queue_capacity):
        raise ValueError("accepted-boundary SR queue capacity mismatch")
    validate_accepted_boundary_sr_queue(
        queue,
        expected_boundary_index=previous_accepted_boundary_index,
    )
    linear, nonlinear_outer = split_endpoint_taylor_map(
        endpoint_map_without_constants
    )
    updated_phi, updated_phi_iv, propagated, queue_stats = (
        accepted_boundary_sr_queue_propagate(
            queue,
            linear,
            expected_boundary_index=previous_accepted_boundary_index,
            reference=reference,
            _validated=True,
        )
    )
    if len(queue.J) == 0:
        inserted = compose(
            endpoint_map_without_constants,
            right_map,
            int(order),
            cutoff_threshold,
            domain,
            diagnostics,
        )
        if not isinstance(inserted, TMVector):
            raise TypeError("accepted-boundary SR composition did not return a TMVector")
        current_owner = tuple(model.remainder for model in inserted)
        branch = "full_reanchor"
    else:
        nonlinear_inserted = compose(
            nonlinear_outer,
            right_map,
            int(order),
            cutoff_threshold,
            domain,
            diagnostics,
        )
        if not isinstance(nonlinear_inserted, TMVector):
            raise TypeError("accepted-boundary SR nonlinear composition did not return a TMVector")
        linear_inserted = _linear_polynomial_image(linear, right_map)
        inserted, current_owner = _add_polynomial_images(
            nonlinear_inserted,
            linear_inserted,
            propagated,
        )
        branch = "nonlinear_plus_linear_queue"
    return AcceptedBoundarySRPrepared(
        inserted=inserted,
        current_owner=current_owner,
        propagated_history=propagated,
        updated_phi=updated_phi,
        updated_phi_iv=updated_phi_iv,
        queue_before=queue,
        accepted_boundary_index=previous_accepted_boundary_index + 1,
        composition_branch=branch,
        stats=dict(queue_stats),
    )


def commit_accepted_boundary_sr(
    prepared: AcceptedBoundarySRPrepared,
    *,
    normalization_scales: Sequence[float],
    cutoff_threshold: float | None,
) -> AcceptedBoundarySRCommitted:
    """Normalize and atomically commit a prepared transition exactly once."""

    normalized, unscaled_sources = _scale_and_cutoff_right_map(
        prepared.inserted,
        normalization_scales,
        cutoff_threshold,
    )
    current_owner = tuple(
        owner + source
        for owner, source in zip(prepared.current_owner, unscaled_sources)
    )
    queue_after, commit_stats = accepted_boundary_sr_queue_commit(
        prepared.queue_before,
        prepared.updated_phi,
        prepared.updated_phi_iv,
        current_owner,
        scales=normalization_scales,
        accepted_boundary_index=prepared.accepted_boundary_index,
        reference=prepared.inserted[0].remainder,
    )
    stats = {
        **dict(prepared.stats),
        **commit_stats,
        "composition_branch": prepared.composition_branch,
        "unscaled_roundoff_cutoff_owner_width_sum": sum(
            _interval_width(value) for value in unscaled_sources
        ),
        "current_owner_width_sum": sum(
            _interval_width(value) for value in current_owner
        ),
        "total_interval_image_width_sum": sum(
            _interval_width(value) for value in prepared.propagated_history
        ),
    }
    return AcceptedBoundarySRCommitted(
        normalized_right_map=normalized,
        queue_after=queue_after,
        current_owner=current_owner,
        unscaled_roundoff_cutoff_owner=unscaled_sources,
        stats=stats,
    )
