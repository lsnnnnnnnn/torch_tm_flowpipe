"""Native dense complete-Q3 TORA plant primitives.

This module contains no Xiangru imports.  It implements the frozen five-state
held-control plant on the repository's generic dense Taylor-model tensors; the
six polynomial variables are local time followed by four state parameters and
one control parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import torch

from .batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRemainderLedger,
    DenseRangePolicy,
    _down,
    _interval_add,
    _interval_mul,
    _polynomial_error_radius,
    _subset_margin,
    _sound_mul_tm,
    _up,
    dense_polynomial_picard,
    sin_tm,
)


TORA_STATE_ORDER = ("x1", "x2", "x3", "x4", "u1")
TORA_BASIS_VARIABLES = (
    "local_time",
    "x1_parameter",
    "x2_parameter",
    "x3_parameter",
    "x4_parameter",
    "u1_parameter",
)
TORA_INITIAL_SET = (
    (0.6, 0.7),
    (-0.7, -0.6),
    (-0.4, -0.3),
    (0.5, 0.6),
)
TORA_B48_SPLITS = (8, 6, 1, 1)


@dataclass(frozen=True)
class ToraQ3Step:
    segment_tm: BatchedTaylorModel
    endpoint_tm: BatchedTaylorModel | None
    tube_lower: torch.Tensor
    tube_upper: torch.Tensor
    endpoint_lower: torch.Tensor
    endpoint_upper: torch.Tensor
    accepted_by_leaf: torch.Tensor
    initial_shrink_mask: torch.Tensor
    initial_margin: torch.Tensor
    round_trace: tuple[Mapping[str, Any], ...]
    polynomial_trace: tuple[Mapping[str, Any], ...]
    status: str
    message: str

    @property
    def accepted(self) -> bool:
        return self.status == "validated"


@dataclass(frozen=True)
class ToraQ3AffineBoundary:
    center: torch.Tensor
    linear: torch.Tensor
    remainder_lower: torch.Tensor
    remainder_upper: torch.Tensor


@dataclass(frozen=True)
class ToraQ3AffineCarry:
    linear: torch.Tensor
    remainder_lower: torch.Tensor
    remainder_upper: torch.Tensor


def identity_tora_q3_carry(
    batch: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
) -> ToraQ3AffineCarry:
    identity = torch.eye(5, dtype=dtype, device=device).expand(batch, -1, -1).clone()
    zeros = torch.zeros((batch, 5), dtype=dtype, device=device)
    return ToraQ3AffineCarry(identity, zeros, zeros.clone())


def _affine_interval_product(
    matrix: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = torch.clamp_min(matrix, 0.0)
    negative = torch.clamp_max(matrix, 0.0)
    lo = torch.matmul(positive, lower.unsqueeze(-1)).squeeze(-1)
    lo = lo + torch.matmul(negative, upper.unsqueeze(-1)).squeeze(-1)
    hi = torch.matmul(positive, upper.unsqueeze(-1)).squeeze(-1)
    hi = hi + torch.matmul(negative, lower.unsqueeze(-1)).squeeze(-1)
    pad = torch.as_tensor(abs(float(epsilon)), dtype=lo.dtype, device=lo.device)
    return _down(lo - pad), _up(hi + pad)


def tora_q3_boundary_from_model(model: BatchedTaylorModel) -> ToraQ3AffineBoundary:
    if model.poly.basis.dim != 6 or model.poly.out_dim != 5:
        raise ValueError("TORA boundary model contract mismatch")
    basis = model.poly.basis
    center = model.poly.coeffs[..., basis.constant_index]
    linear = torch.zeros(
        (model.poly.batch, 5, 5),
        dtype=model.poly.coeffs.dtype,
        device=model.poly.coeffs.device,
    )
    for variable in range(5):
        exponent = [0] * 6
        exponent[variable + 1] = 1
        linear[:, :, variable] = model.poly.coeffs[..., basis.term_index(exponent)]
    allowed = torch.zeros(basis.num_terms, dtype=torch.bool, device=basis.device)
    allowed[basis.constant_index] = True
    for variable in range(5):
        exponent = [0] * 6
        exponent[variable + 1] = 1
        allowed[basis.term_index(exponent)] = True
    if bool(torch.any(model.poly.coeffs[..., ~allowed] != 0)):
        raise ValueError("boundary model contains non-affine or local-time terms")
    return ToraQ3AffineBoundary(
        center,
        linear,
        model.rem_lo,
        model.rem_hi,
    )


def tora_q3_boundary_box(
    boundary: ToraQ3AffineBoundary,
) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude = torch.sum(torch.abs(boundary.linear), dim=2)
    lower = _down(boundary.center - magnitude)
    upper = _up(boundary.center + magnitude)
    return _interval_add(
        lower,
        upper,
        boundary.remainder_lower,
        boundary.remainder_upper,
    )


def compose_tora_q3_boundary(
    boundary: ToraQ3AffineBoundary,
    carry: ToraQ3AffineCarry,
    *,
    epsilon: float = 1e-12,
) -> ToraQ3AffineBoundary:
    """Materialize a boundary in the current five normalized generators."""
    if boundary.linear.shape != carry.linear.shape:
        raise ValueError("boundary/carry batch or generator mismatch")
    transformed = torch.matmul(boundary.linear, carry.linear)
    carried_lo, carried_hi = _affine_interval_product(
        boundary.linear,
        carry.remainder_lower,
        carry.remainder_upper,
        epsilon=epsilon,
    )
    remainder_lo, remainder_hi = _interval_add(
        boundary.remainder_lower,
        boundary.remainder_upper,
        carried_lo,
        carried_hi,
    )
    return ToraQ3AffineBoundary(
        boundary.center,
        transformed,
        remainder_lo,
        remainder_hi,
    )


def compose_tora_q3_tm(
    value: BatchedTaylorModel,
    parameterization: ToraQ3AffineCarry,
) -> BatchedTaylorModel:
    """Substitute the five affine normalized-state maps into a local Q3 TM.

    Local time is retained as the first polynomial variable.  The other five
    variables are replaced by ``parameterization``.  All degree-two and
    degree-three monomials are formed with the sound dense TM product, and
    both the final contraction and its floating-point reduction receive an
    explicit outward roundoff envelope.
    """
    if value.poly.basis.dim != 6 or value.poly.basis.order != 3:
        raise ValueError("TORA composition requires a six-variable complete-Q3 TM")
    batch = value.poly.batch
    if parameterization.linear.shape != (batch, 5, 5):
        raise ValueError("TORA parameterization must have shape [batch,5,5]")
    if parameterization.remainder_lower.shape != (batch, 5):
        raise ValueError("TORA parameterization remainder must have shape [batch,5]")
    if parameterization.remainder_upper.shape != (batch, 5):
        raise ValueError("TORA parameterization remainder must have shape [batch,5]")
    if not bool(
        torch.all(
            parameterization.remainder_lower
            <= parameterization.remainder_upper
        )
    ):
        raise ValueError("TORA parameterization remainder is invalid")

    basis = value.poly.basis
    dtype = value.poly.coeffs.dtype
    device = value.poly.coeffs.device
    slots = basis.num_terms
    monomial_coefficients = torch.zeros(
        (batch, slots, slots), dtype=dtype, device=device
    )
    monomial_lower = torch.zeros((batch, slots), dtype=dtype, device=device)
    monomial_upper = torch.zeros_like(monomial_lower)
    monomial_coefficients[
        :, basis.constant_index, basis.constant_index
    ] = 1.0

    variable_slots: list[int] = []
    for variable in range(6):
        exponent = [0] * 6
        exponent[variable] = 1
        variable_slots.append(basis.term_index(exponent))
    monomial_coefficients[:, variable_slots[0], variable_slots[0]] = 1.0
    for local_variable in range(5):
        target = variable_slots[local_variable + 1]
        monomial_coefficients[
            :, target, variable_slots[1:]
        ] = parameterization.linear[:, local_variable, :]
        monomial_lower[:, target] = parameterization.remainder_lower[
            :, local_variable
        ]
        monomial_upper[:, target] = parameterization.remainder_upper[
            :, local_variable
        ]

    exponent_rows: list[tuple[int, ...] | None] = [None] * slots
    for exponent, slot in basis.exponent_to_index.items():
        exponent_rows[slot] = exponent
    if any(exponent is None for exponent in exponent_rows):
        raise RuntimeError("incomplete dense basis exponent map")
    complete_exponents = [
        exponent for exponent in exponent_rows if exponent is not None
    ]
    exponent_to_index = basis.exponent_to_index
    for degree in (2, 3):
        targets = [
            index for index, exponent in enumerate(complete_exponents)
            if sum(exponent) == degree
        ]
        parents: list[int] = []
        variables: list[int] = []
        for exponent in (complete_exponents[index] for index in targets):
            variable = next(i for i, power in enumerate(exponent) if power)
            parent = list(exponent)
            parent[variable] -= 1
            parents.append(exponent_to_index[tuple(parent)])
            variables.append(variable_slots[variable])
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
        parent_tensor = torch.as_tensor(parents, dtype=torch.long, device=device)
        variable_tensor = torch.as_tensor(variables, dtype=torch.long, device=device)

        def selected(indices: torch.Tensor) -> BatchedTaylorModel:
            return BatchedTaylorModel(
                BatchedPolynomial(
                    monomial_coefficients.index_select(1, indices), basis
                ),
                monomial_lower.index_select(1, indices),
                monomial_upper.index_select(1, indices),
                value.domain_lo,
                value.domain_hi,
                DenseRemainderLedger.empty().add(
                    "composition_overflow",
                    monomial_lower.index_select(1, indices),
                    monomial_upper.index_select(1, indices),
                ),
                value.range_policy,
                value.range_trace,
            )

        products = _sound_mul_tm(
            selected(parent_tensor), selected(variable_tensor)
        )
        monomial_coefficients[:, target_tensor, :] = products.poly.coeffs
        monomial_lower[:, target_tensor] = products.rem_lo
        monomial_upper[:, target_tensor] = products.rem_hi

    source = value.poly.coeffs
    coefficients = torch.bmm(source, monomial_coefficients)
    epsilon = torch.finfo(dtype).eps
    operations = 2 * slots + 1
    gamma = (operations * epsilon) / (1.0 - operations * epsilon)
    absolute_contraction = torch.bmm(
        torch.abs(source), torch.abs(monomial_coefficients)
    )
    coefficient_error = _up(
        absolute_contraction * (2.0 * gamma) + torch.finfo(dtype).tiny
    )
    coefficient_roundoff = _polynomial_error_radius(
        coefficient_error, basis, value.domain_lo, value.domain_hi
    )

    scaled_lower, scaled_upper = _interval_mul(
        source,
        source,
        monomial_lower[:, None, :],
        monomial_upper[:, None, :],
    )
    remainder_lower = scaled_lower.sum(dim=2)
    remainder_upper = scaled_upper.sum(dim=2)
    reduction_magnitude = torch.maximum(
        torch.abs(scaled_lower), torch.abs(scaled_upper)
    ).sum(dim=2)
    reduction_pad = _up(
        reduction_magnitude * (2.0 * gamma) + torch.finfo(dtype).tiny
    )
    remainder_lower = _down(remainder_lower - reduction_pad)
    remainder_upper = _up(remainder_upper + reduction_pad)
    composition_lower = _down(remainder_lower - coefficient_roundoff)
    composition_upper = _up(remainder_upper + coefficient_roundoff)
    remainder_lower, remainder_upper = _interval_add(
        value.rem_lo,
        value.rem_hi,
        composition_lower,
        composition_upper,
    )
    ledger = value.ledger.add(
        "composition_overflow", composition_lower, composition_upper
    )
    return BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        remainder_lower,
        remainder_upper,
        value.domain_lo,
        value.domain_hi,
        ledger,
        value.range_policy,
        value.range_trace,
    )


def compose_tora_q3_step(
    local_step: ToraQ3Step,
    parameterization: ToraQ3AffineCarry,
    *,
    h: float = 0.1,
) -> ToraQ3Step:
    """Return the physical-coordinate enclosure for a validated local step."""
    composed = compose_tora_q3_tm(local_step.segment_tm, parameterization)
    tube_lower, tube_upper = composed.range_bound(
        context="tora_composed_step_tube"
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
    accepted = local_step.accepted_by_leaf & finite & property_ok
    all_accepted = bool(torch.all(accepted))
    return ToraQ3Step(
        segment_tm=composed,
        endpoint_tm=(
            composed.endpoint(0, float(h)) if all_accepted else None
        ),
        tube_lower=tube_lower,
        tube_upper=tube_upper,
        endpoint_lower=endpoint_lower,
        endpoint_upper=endpoint_upper,
        accepted_by_leaf=accepted,
        initial_shrink_mask=local_step.initial_shrink_mask,
        initial_margin=local_step.initial_margin,
        round_trace=local_step.round_trace,
        polynomial_trace=local_step.polynomial_trace,
        status="validated" if all_accepted else "failed",
        message=(
            ""
            if all_accepted
            else "local validation or composed physical-coordinate property check failed"
        ),
    )


def install_interval_control_on_boundary(
    boundary: ToraQ3AffineBoundary,
    control_lower: torch.Tensor,
    control_upper: torch.Tensor,
) -> ToraQ3AffineBoundary:
    lower = torch.as_tensor(
        control_lower, dtype=boundary.center.dtype, device=boundary.center.device
    ).reshape(-1)
    upper = torch.as_tensor(
        control_upper, dtype=boundary.center.dtype, device=boundary.center.device
    ).reshape(-1)
    if lower.shape != (boundary.center.shape[0],) or upper.shape != lower.shape:
        raise ValueError("control boundary batch mismatch")
    center = boundary.center.clone()
    linear = boundary.linear.clone()
    rem_lo = boundary.remainder_lower.clone()
    rem_hi = boundary.remainder_upper.clone()
    center[:, 4] = lower + 0.5 * (upper - lower)
    linear[:, 4, :] = 0.0
    linear[:, 4, 4] = _up(
        torch.maximum(center[:, 4] - lower, upper - center[:, 4])
    )
    rem_lo[:, 4] = 0.0
    rem_hi[:, 4] = 0.0
    return ToraQ3AffineBoundary(center, linear, rem_lo, rem_hi)


def normalize_tora_q3_boundary(
    boundary: ToraQ3AffineBoundary,
    carry: ToraQ3AffineCarry,
    *,
    h: float = 0.1,
    epsilon: float = 1e-12,
) -> tuple[BatchedTaylorModel, ToraQ3AffineCarry]:
    """Normalize one affine boundary while retaining cross-state generators."""
    transformed = torch.matmul(boundary.linear, carry.linear)
    carried_lo, carried_hi = _affine_interval_product(
        boundary.linear,
        carry.remainder_lower,
        carry.remainder_upper,
        epsilon=epsilon,
    )
    remainder_lo, remainder_hi = _interval_add(
        boundary.remainder_lower,
        boundary.remainder_upper,
        carried_lo,
        carried_hi,
    )
    linear_radius = torch.sum(torch.abs(transformed), dim=2)
    lower = _down(-linear_radius + remainder_lo)
    upper = _up(linear_radius + remainder_hi)
    scale = _up(torch.maximum(torch.abs(lower), torch.abs(upper)))
    scale = torch.maximum(
        scale,
        torch.full_like(scale, abs(float(epsilon))),
    )
    normalized_linear = transformed / scale.unsqueeze(-1)
    normalized_lo = _down(remainder_lo / scale - abs(float(epsilon)))
    normalized_hi = _up(remainder_hi / scale + abs(float(epsilon)))
    next_carry = ToraQ3AffineCarry(
        normalized_linear,
        normalized_lo,
        normalized_hi,
    )

    basis = BatchedMonomialBasis.build(6, 3, str(boundary.center.device))
    coefficients = torch.zeros(
        (boundary.center.shape[0], 5, basis.num_terms),
        dtype=boundary.center.dtype,
        device=boundary.center.device,
    )
    coefficients[..., basis.constant_index] = boundary.center
    for state in range(5):
        exponent = [0] * 6
        exponent[state + 1] = 1
        coefficients[:, state, basis.term_index(exponent)] = scale[:, state]
    zeros = torch.zeros_like(boundary.center)
    domain_lo = torch.full(
        (boundary.center.shape[0], 6),
        -1.0,
        dtype=boundary.center.dtype,
        device=boundary.center.device,
    )
    domain_hi = torch.ones_like(domain_lo)
    domain_lo[:, 0] = 0.0
    domain_hi[:, 0] = float(h)
    local = BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        zeros,
        zeros.clone(),
        domain_lo,
        domain_hi,
        DenseRemainderLedger.empty(),
        DenseRangePolicy(method="natural"),
    )
    return local, next_carry


def project_tora_q3_endpoint_to_affine(
    model: BatchedTaylorModel,
    *,
    h: float = 0.1,
    epsilon: float = 1e-12,
) -> ToraQ3AffineBoundary:
    """Project the exact-time Q3 endpoint to constant/linear spatial terms."""
    if model.poly.basis.dim != 6 or model.poly.out_dim != 5:
        raise ValueError("endpoint projection requires a TORA-Q3 segment model")
    endpoint = model.endpoint(0, float(h))
    basis = endpoint.poly.basis
    exponents = basis.exponents
    spatial_degree = exponents.sum(dim=1)
    constant_mask = spatial_degree == 0
    center = endpoint.poly.coeffs[..., basis.constant_index]
    linear = torch.zeros(
        (model.poly.batch, 5, 5),
        dtype=model.poly.coeffs.dtype,
        device=model.poly.coeffs.device,
    )
    retained_mask = constant_mask.clone()
    for variable in range(5):
        target = torch.zeros(5, dtype=torch.long, device=exponents.device)
        target[variable] = 1
        mask = torch.all(exponents == target, dim=1)
        linear[:, :, variable] = endpoint.poly.coeffs[..., mask].squeeze(-1)
        retained_mask |= mask

    overflow_coefficients = torch.where(
        retained_mask.view(1, 1, -1),
        torch.zeros_like(endpoint.poly.coeffs),
        endpoint.poly.coeffs,
    )
    overflow = BatchedPolynomial(overflow_coefficients, basis)
    overflow_lo, overflow_hi = overflow.range_bound(
        endpoint.domain_lo,
        endpoint.domain_hi,
        policy=endpoint.range_policy,
        context="tora_endpoint_projection_overflow",
        trace=endpoint.range_trace,
    )
    remainder_lo, remainder_hi = _interval_add(
        endpoint.rem_lo,
        endpoint.rem_hi,
        overflow_lo,
        overflow_hi,
    )
    pad = torch.full_like(remainder_lo, abs(float(epsilon)))
    remainder_lo = _down(remainder_lo - pad)
    remainder_hi = _up(remainder_hi + pad)
    return ToraQ3AffineBoundary(center, linear, remainder_lo, remainder_hi)


def tora_b48_boxes(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the frozen Xiangru Cartesian leaf order for B48."""
    device_t = torch.device(device)
    endpoints = [
        torch.linspace(lo, hi, count + 1, dtype=dtype, device=device_t)
        for (lo, hi), count in zip(TORA_INITIAL_SET, TORA_B48_SPLITS, strict=True)
    ]
    cells = torch.tensor(
        list(product(*(range(count) for count in TORA_B48_SPLITS))),
        dtype=torch.long,
        device=device_t,
    )
    lower = torch.stack(
        [endpoints[state].index_select(0, cells[:, state]) for state in range(4)],
        dim=1,
    )
    upper = torch.stack(
        [endpoints[state].index_select(0, cells[:, state] + 1) for state in range(4)],
        dim=1,
    )
    return lower, upper


def build_tora_q3_initial_model(
    control_lower: torch.Tensor,
    control_upper: torch.Tensor,
    *,
    h: float = 0.1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    range_policy: DenseRangePolicy | None = None,
    range_trace: list[dict[str, Any]] | None = None,
) -> BatchedTaylorModel:
    if dtype != torch.float64:
        raise TypeError("formal TORA-Q3 lane requires float64")
    if h <= 0:
        raise ValueError("h must be positive")
    device_t = torch.device(device)
    state_lower, state_upper = tora_b48_boxes(device=device_t, dtype=dtype)
    return build_tora_q3_box_model(
        state_lower,
        state_upper,
        control_lower,
        control_upper,
        h=h,
        device=device_t,
        dtype=dtype,
        range_policy=range_policy,
        range_trace=range_trace,
    )


def build_tora_q3_box_model(
    state_lower: torch.Tensor,
    state_upper: torch.Tensor,
    control_lower: torch.Tensor,
    control_upper: torch.Tensor,
    *,
    h: float = 0.1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    range_policy: DenseRangePolicy | None = None,
    range_trace: list[dict[str, Any]] | None = None,
) -> BatchedTaylorModel:
    """Build an outward affine B48 boundary from physical state/control boxes."""
    if dtype != torch.float64:
        raise TypeError("formal TORA-Q3 lane requires float64")
    if h <= 0:
        raise ValueError("h must be positive")
    device_t = torch.device(device)
    state_lo = torch.as_tensor(state_lower, dtype=dtype, device=device_t)
    state_hi = torch.as_tensor(state_upper, dtype=dtype, device=device_t)
    if state_lo.ndim != 2 or state_lo.shape[1] != 4 or state_hi.shape != state_lo.shape:
        raise ValueError("TORA state boxes must have shape [batch,4]")
    if not bool(torch.all(state_lo <= state_hi)):
        raise ValueError("state lower bounds exceed upper bounds")
    control_lo = torch.as_tensor(control_lower, dtype=dtype, device=device_t).reshape(-1)
    control_hi = torch.as_tensor(control_upper, dtype=dtype, device=device_t).reshape(-1)
    batch = int(state_lo.shape[0])
    if control_lo.shape != (batch,) or control_hi.shape != (batch,):
        raise ValueError("TORA boxes require one control interval per leaf")
    if not bool(torch.all(control_lo <= control_hi)):
        raise ValueError("control lower bounds exceed upper bounds")
    lower = torch.cat((state_lo, control_lo[:, None]), dim=1)
    upper = torch.cat((state_hi, control_hi[:, None]), dim=1)
    center = lower + 0.5 * (upper - lower)
    radius = _up(torch.maximum(center - lower, upper - center))
    basis = BatchedMonomialBasis.build(6, 3, str(device_t))
    coefficients = torch.zeros(
        (batch, 5, basis.num_terms), dtype=dtype, device=device_t
    )
    coefficients[..., basis.constant_index] = center
    for state in range(5):
        exponent = [0] * 6
        exponent[state + 1] = 1
        coefficients[:, state, basis.term_index(exponent)] = radius[:, state]
    zeros = torch.zeros((batch, 5), dtype=dtype, device=device_t)
    domain_lo = torch.full((batch, 6), -1.0, dtype=dtype, device=device_t)
    domain_hi = torch.ones_like(domain_lo)
    domain_lo[:, 0] = 0.0
    domain_hi[:, 0] = float(h)
    return BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        zeros,
        zeros.clone(),
        domain_lo,
        domain_hi,
        DenseRemainderLedger.empty(),
        range_policy or DenseRangePolicy(method="natural"),
        range_trace,
    )


def replace_tora_held_control(
    model: BatchedTaylorModel,
    control_lower: torch.Tensor,
    control_upper: torch.Tensor,
) -> BatchedTaylorModel:
    """Install a frozen independent u1 interval without changing state rows."""
    if model.poly.basis.dim != 6 or model.poly.out_dim != 5:
        raise ValueError("held-control reset requires a six-variable TORA model")
    lower = torch.as_tensor(
        control_lower, dtype=model.poly.coeffs.dtype, device=model.poly.coeffs.device
    ).reshape(-1)
    upper = torch.as_tensor(
        control_upper, dtype=model.poly.coeffs.dtype, device=model.poly.coeffs.device
    ).reshape(-1)
    if lower.shape != (model.poly.batch,) or upper.shape != lower.shape:
        raise ValueError("control reset batch mismatch")
    if not bool(torch.all(lower <= upper)):
        raise ValueError("control lower bounds exceed upper bounds")
    coefficients = model.poly.coeffs.clone()
    coefficients[:, 4, :] = 0.0
    coefficients[:, 4, model.poly.basis.constant_index] = lower + 0.5 * (upper - lower)
    exponent = [0] * 6
    exponent[5] = 1
    coefficients[:, 4, model.poly.basis.term_index(exponent)] = 0.5 * (upper - lower)
    rem_lo = model.rem_lo.clone()
    rem_hi = model.rem_hi.clone()
    rem_lo[:, 4] = 0.0
    rem_hi[:, 4] = 0.0
    ledger_entries = {}
    for category, (entry_lo, entry_hi) in model.ledger.entries.items():
        updated_lo = entry_lo.clone()
        updated_hi = entry_hi.clone()
        updated_lo[:, 4] = 0.0
        updated_hi[:, 4] = 0.0
        ledger_entries[category] = (updated_lo, updated_hi)
    return BatchedTaylorModel(
        BatchedPolynomial(coefficients, model.poly.basis),
        rem_lo,
        rem_hi,
        model.domain_lo,
        model.domain_hi,
        DenseRemainderLedger(ledger_entries),
        model.range_policy,
        model.range_trace,
    )


def tora_q3_rhs(
    state: BatchedTaylorModel,
    *,
    sine_order: int = 2,
) -> BatchedTaylorModel:
    if state.poly.out_dim != 5:
        raise ValueError("TORA RHS requires [x1,x2,x3,x4,u1]")
    x1 = state.component(0)
    x2 = state.component(1)
    x3 = state.component(2)
    x4 = state.component(3)
    control = state.component(4)
    state2 = -x1 + sin_tm(x3, order=sine_order).scale(0.1)
    state4 = control - 10.0
    held = BatchedTaylorModel.constants_like(0.0, control)
    return BatchedTaylorModel.concat((x2, state2, x4, state4, held))


def _endpoint_bounds(
    model: BatchedTaylorModel,
    *,
    h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.endpoint(0, float(h)).range_bound(context="tora_endpoint")


def _zero_exact_held_remainder(model: BatchedTaylorModel) -> BatchedTaylorModel:
    """Preserve the exact invariant u1'=0 in remainder bookkeeping."""
    rem_lo = model.rem_lo.clone()
    rem_hi = model.rem_hi.clone()
    rem_lo[:, 4] = 0.0
    rem_hi[:, 4] = 0.0
    entries = {}
    for category, (entry_lo, entry_hi) in model.ledger.entries.items():
        lo = entry_lo.clone()
        hi = entry_hi.clone()
        lo[:, 4] = 0.0
        hi[:, 4] = 0.0
        entries[category] = (lo, hi)
    return BatchedTaylorModel(
        model.poly,
        rem_lo,
        rem_hi,
        model.domain_lo,
        model.domain_hi,
        DenseRemainderLedger(entries),
        model.range_policy,
        model.range_trace,
    )


def dense_tora_q3_dr_step(
    base: BatchedTaylorModel,
    *,
    h: float = 0.1,
    sine_order: int = 2,
    polynomial_picard_rounds: int = 2,
    remainder_rounds: int = 10,
    seed: float = 0.01,
    capture_trace: bool = True,
) -> ToraQ3Step:
    """K2 plus componentwise ten-round remainder Picard on dense tensors."""
    if base.poly.basis.dim != 6 or base.poly.basis.order != 3:
        raise ValueError("TORA-Q3 step requires the six-variable complete-Q3 basis")
    if polynomial_picard_rounds <= 0 or remainder_rounds < 0:
        raise ValueError("Picard round counts are invalid")
    if not torch.allclose(base.domain_lo[:, 0], torch.zeros_like(base.domain_lo[:, 0])):
        raise ValueError("local time lower bound must be zero")
    if not torch.allclose(base.domain_hi[:, 0], torch.full_like(base.domain_hi[:, 0], float(h))):
        raise ValueError("local time upper bound must equal h")

    rhs = lambda value: tora_q3_rhs(value, sine_order=sine_order)
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
    seeded = candidate.with_remainder(-seed_vector, seed_vector)
    initial_image = _zero_exact_held_remainder(
        base.without_remainder().add(rhs(seeded).integrate(0))
    )
    initial_margin = _subset_margin(
        -seed_vector,
        seed_vector,
        initial_image.rem_lo,
        initial_image.rem_hi,
    )
    initial_shrink = initial_margin >= 0.0

    difference = BatchedPolynomial(
        initial_image.poly.coeffs - seeded.poly.coeffs,
        seeded.poly.basis,
    )
    roundoff_lo, roundoff_hi = difference.range_bound(
        base.domain_lo,
        base.domain_hi,
        policy=base.range_policy,
        context="tora_picard_roundoff",
        trace=base.range_trace,
    )
    roundoff_lo[:, 4] = 0.0
    roundoff_hi[:, 4] = 0.0
    current = seeded
    rows: list[Mapping[str, Any]] = []
    all_rounds_shrink = torch.ones_like(initial_shrink)
    for round_index in range(1, remainder_rounds + 1):
        image = _zero_exact_held_remainder(
            base.without_remainder().add(rhs(current).integrate(0))
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
            DenseRemainderLedger.empty().add(
                "picard_residual", accepted_lo, accepted_hi
            ),
            base.range_policy,
            base.range_trace,
        )

    tube_lower, tube_upper = current.range_bound(context="tora_full_step_tube")
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
    certificate = initial_shrink.all(dim=1) & all_rounds_shrink.all(dim=1)
    accepted = finite & property_ok & certificate
    endpoint_tm = current.endpoint(0, float(h)) if bool(torch.all(accepted)) else None
    return ToraQ3Step(
        segment_tm=current,
        endpoint_tm=endpoint_tm,
        tube_lower=tube_lower,
        tube_upper=tube_upper,
        endpoint_lower=endpoint_lower,
        endpoint_upper=endpoint_upper,
        accepted_by_leaf=accepted,
        initial_shrink_mask=initial_shrink,
        initial_margin=initial_margin,
        round_trace=tuple(rows),
        polynomial_trace=polynomial_trace,
        status="validated" if bool(torch.all(accepted)) else "failed",
        message=(
            ""
            if bool(torch.all(accepted))
            else "one or more leaves failed finiteness, property, or DR-style subset validation"
        ),
    )


def lift_tora_q3_endpoint(
    endpoint: BatchedTaylorModel,
    *,
    h: float = 0.1,
) -> BatchedTaylorModel:
    """Prepend a fresh local-time variable while retaining all spatial terms."""
    if endpoint.poly.basis.dim != 5 or endpoint.poly.basis.order != 3:
        raise ValueError("endpoint must use a five-parameter complete-Q3 basis")
    basis = BatchedMonomialBasis.build(
        6, 3, str(endpoint.poly.coeffs.device)
    )
    coefficients = torch.zeros(
        (endpoint.poly.batch, endpoint.poly.out_dim, basis.num_terms),
        dtype=endpoint.poly.coeffs.dtype,
        device=endpoint.poly.coeffs.device,
    )
    targets = torch.tensor(
        [
            basis.term_index((0, *tuple(int(value) for value in exponent)))
            for exponent in endpoint.poly.basis.exponents.detach().cpu().tolist()
        ],
        dtype=torch.long,
        device=endpoint.poly.coeffs.device,
    )
    coefficients.index_copy_(2, targets, endpoint.poly.coeffs)
    domain_lo = torch.cat(
        (
            torch.zeros((endpoint.poly.batch, 1), dtype=endpoint.domain_lo.dtype, device=endpoint.domain_lo.device),
            endpoint.domain_lo,
        ),
        dim=1,
    )
    domain_hi = torch.cat(
        (
            torch.full((endpoint.poly.batch, 1), float(h), dtype=endpoint.domain_hi.dtype, device=endpoint.domain_hi.device),
            endpoint.domain_hi,
        ),
        dim=1,
    )
    return BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        endpoint.rem_lo,
        endpoint.rem_hi,
        domain_lo,
        domain_hi,
        endpoint.ledger,
        endpoint.range_policy,
        endpoint.range_trace,
    )


__all__ = [
    "TORA_BASIS_VARIABLES",
    "TORA_B48_SPLITS",
    "TORA_INITIAL_SET",
    "TORA_STATE_ORDER",
    "ToraQ3AffineBoundary",
    "ToraQ3AffineCarry",
    "ToraQ3Step",
    "build_tora_q3_initial_model",
    "build_tora_q3_box_model",
    "compose_tora_q3_boundary",
    "compose_tora_q3_step",
    "compose_tora_q3_tm",
    "dense_tora_q3_dr_step",
    "identity_tora_q3_carry",
    "install_interval_control_on_boundary",
    "lift_tora_q3_endpoint",
    "replace_tora_held_control",
    "normalize_tora_q3_boundary",
    "project_tora_q3_endpoint_to_affine",
    "tora_b48_boxes",
    "tora_q3_boundary_box",
    "tora_q3_boundary_from_model",
    "tora_q3_rhs",
]
