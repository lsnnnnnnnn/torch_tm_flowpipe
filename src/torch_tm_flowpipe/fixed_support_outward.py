"""Safeguarded outward CPU reference for fixed-support Taylor models.

The performance core deliberately uses ordinary round-to-nearest tensors.  This
module is a separate, slower reference: every retained coefficient is an
interval and every arithmetic operation/reduction is expanded with
``nextafter``.  Its claim is limited to IEEE-754 binary64 CPU execution under
the declared PyTorch backend assumptions; it is not a proof about CUDA or the
ordinary compiled lane.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .fixed_support import FixedSupportKernelPlan


def _negative_infinity(value: torch.Tensor) -> torch.Tensor:
    return torch.full_like(value, -torch.inf)


def _positive_infinity(value: torch.Tensor) -> torch.Tensor:
    return torch.full_like(value, torch.inf)


def _down(value: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(value, _negative_infinity(value))


def _up(value: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(value, _positive_infinity(value))


def _sanitize(lo: torch.Tensor, hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    invalid = torch.isnan(lo) | torch.isnan(hi) | (lo > hi)
    return (
        torch.where(invalid, _negative_infinity(lo), lo),
        torch.where(invalid, _positive_infinity(hi), hi),
    )


@dataclass(frozen=True)
class OutwardIntervalTensor:
    """A tensor-valued closed interval with safeguarded binary64 operations."""

    lo: torch.Tensor
    hi: torch.Tensor

    def __post_init__(self) -> None:
        if self.lo.shape != self.hi.shape:
            raise ValueError("outward interval endpoints must have the same shape")
        if self.lo.dtype != self.hi.dtype or self.lo.device != self.hi.device:
            raise ValueError("outward interval endpoints must share dtype/device")
        if self.lo.dtype != torch.float64:
            raise ValueError("the outward reference is qualified only for float64")

    @classmethod
    def point(cls, value: torch.Tensor) -> "OutwardIntervalTensor":
        return cls(value, value)

    @classmethod
    def zeros_like(cls, value: torch.Tensor) -> "OutwardIntervalTensor":
        zero = torch.zeros_like(value)
        return cls(zero, zero)

    def sanitized(self) -> "OutwardIntervalTensor":
        lo, hi = _sanitize(self.lo, self.hi)
        return OutwardIntervalTensor(lo, hi)

    def add(self, other: "OutwardIntervalTensor") -> "OutwardIntervalTensor":
        return OutwardIntervalTensor(
            _down(self.lo + other.lo), _up(self.hi + other.hi)
        ).sanitized()

    def sub(self, other: "OutwardIntervalTensor") -> "OutwardIntervalTensor":
        return OutwardIntervalTensor(
            _down(self.lo - other.hi), _up(self.hi - other.lo)
        ).sanitized()

    def neg(self) -> "OutwardIntervalTensor":
        return OutwardIntervalTensor(-self.hi, -self.lo)

    def mul(self, other: "OutwardIntervalTensor") -> "OutwardIntervalTensor":
        products = torch.stack(
            (
                self.lo * other.lo,
                self.lo * other.hi,
                self.hi * other.lo,
                self.hi * other.hi,
            ),
            dim=0,
        )
        return OutwardIntervalTensor(
            _down(products.amin(dim=0)), _up(products.amax(dim=0))
        ).sanitized()

    def scale(self, factor: torch.Tensor | float) -> "OutwardIntervalTensor":
        factor_tensor = torch.as_tensor(
            factor, dtype=self.lo.dtype, device=self.lo.device
        )
        return self.mul(OutwardIntervalTensor.point(factor_tensor))

    def reciprocal(self) -> "OutwardIntervalTensor":
        crosses_zero = (self.lo <= 0) & (self.hi >= 0)
        reciprocal_lo = _down(1.0 / self.hi)
        reciprocal_hi = _up(1.0 / self.lo)
        lo = torch.minimum(reciprocal_lo, reciprocal_hi)
        hi = torch.maximum(reciprocal_lo, reciprocal_hi)
        return OutwardIntervalTensor(
            torch.where(crosses_zero, _negative_infinity(lo), lo),
            torch.where(crosses_zero, _positive_infinity(hi), hi),
        ).sanitized()

    def div(self, other: "OutwardIntervalTensor") -> "OutwardIntervalTensor":
        return self.mul(other.reciprocal())

    def sum(self, dim: int | Sequence[int]) -> "OutwardIntervalTensor":
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        normalized = tuple(index % self.lo.ndim for index in dims)
        result = self
        for reduction_dim in sorted(normalized, reverse=True):
            moved_lo = result.lo.movedim(reduction_dim, -1)
            moved_hi = result.hi.movedim(reduction_dim, -1)
            total = OutwardIntervalTensor.zeros_like(moved_lo[..., 0])
            for index in range(moved_lo.shape[-1]):
                total = total.add(
                    OutwardIntervalTensor(moved_lo[..., index], moved_hi[..., index])
                )
            result = total
        return result

    def abs(self) -> "OutwardIntervalTensor":
        crosses_zero = (self.lo <= 0) & (self.hi >= 0)
        lo = torch.where(
            crosses_zero,
            torch.zeros_like(self.lo),
            torch.minimum(torch.abs(self.lo), torch.abs(self.hi)),
        )
        hi = torch.maximum(torch.abs(self.lo), torch.abs(self.hi))
        return OutwardIntervalTensor(lo, _up(hi)).sanitized()

    def index_select(self, dim: int, index: torch.Tensor) -> "OutwardIntervalTensor":
        return OutwardIntervalTensor(
            self.lo.index_select(dim, index), self.hi.index_select(dim, index)
        )

    def finite(self) -> torch.Tensor:
        return torch.isfinite(self.lo) & torch.isfinite(self.hi) & (self.lo <= self.hi)


def outward_sum(values: Iterable[OutwardIntervalTensor]) -> OutwardIntervalTensor:
    iterator = iter(values)
    try:
        result = next(iterator)
    except StopIteration as error:
        raise ValueError("outward_sum requires at least one value") from error
    for value in iterator:
        result = result.add(value)
    return result


def outward_where(
    mask: torch.Tensor,
    new: OutwardIntervalTensor,
    old: OutwardIntervalTensor,
) -> OutwardIntervalTensor:
    expanded = mask
    while expanded.ndim < new.lo.ndim:
        expanded = expanded.unsqueeze(-1)
    return OutwardIntervalTensor(
        torch.where(expanded, new.lo, old.lo),
        torch.where(expanded, new.hi, old.hi),
    )


def outward_matmul(
    left: OutwardIntervalTensor,
    right: OutwardIntervalTensor,
) -> OutwardIntervalTensor:
    if left.lo.shape[-1] != right.lo.shape[-2]:
        raise ValueError("outward matmul inner dimensions do not agree")
    products: list[OutwardIntervalTensor] = []
    for index in range(left.lo.shape[-1]):
        products.append(
            OutwardIntervalTensor(
                left.lo[..., :, index, None], left.hi[..., :, index, None]
            ).mul(
                OutwardIntervalTensor(
                    right.lo[..., index, None, :], right.hi[..., index, None, :]
                )
            )
        )
    return outward_sum(products)


@dataclass(frozen=True)
class OutwardFixedSupportPolynomial:
    coefficients: OutwardIntervalTensor
    plan: FixedSupportKernelPlan

    @classmethod
    def point(
        cls, coefficients: torch.Tensor, plan: FixedSupportKernelPlan
    ) -> "OutwardFixedSupportPolynomial":
        if coefficients.shape[-1] != plan.num_slots:
            raise ValueError("coefficient support does not match plan")
        return cls(OutwardIntervalTensor.point(coefficients), plan)

    @classmethod
    def zeros_like(
        cls, coefficients: torch.Tensor, plan: FixedSupportKernelPlan
    ) -> "OutwardFixedSupportPolynomial":
        return cls(OutwardIntervalTensor.zeros_like(coefficients), plan)

    def add(self, other: "OutwardFixedSupportPolynomial") -> "OutwardFixedSupportPolynomial":
        return OutwardFixedSupportPolynomial(
            self.coefficients.add(other.coefficients), self.plan
        )

    def sub(self, other: "OutwardFixedSupportPolynomial") -> "OutwardFixedSupportPolynomial":
        return OutwardFixedSupportPolynomial(
            self.coefficients.sub(other.coefficients), self.plan
        )

    def range(self, box: OutwardIntervalTensor) -> OutwardIntervalTensor:
        terms: list[OutwardIntervalTensor] = []
        for slot, exponent in enumerate(self.plan.exponent_tuples):
            monomial = OutwardIntervalTensor.point(
                torch.ones(
                    box.lo.shape[0], dtype=box.lo.dtype, device=box.lo.device
                )
            )
            for variable_index, power in enumerate(exponent):
                variable = OutwardIntervalTensor(
                    box.lo[:, variable_index], box.hi[:, variable_index]
                )
                for _ in range(power):
                    monomial = monomial.mul(variable)
            coefficient = OutwardIntervalTensor(
                self.coefficients.lo[..., slot], self.coefficients.hi[..., slot]
            )
            terms.append(
                coefficient.mul(
                    OutwardIntervalTensor(
                        monomial.lo[:, None], monomial.hi[:, None]
                    )
                )
            )
        return outward_sum(terms)

    def multiply_project(
        self,
        other: "OutwardFixedSupportPolynomial",
        box: OutwardIntervalTensor,
    ) -> tuple["OutwardFixedSupportPolynomial", OutwardIntervalTensor]:
        zeros = torch.zeros_like(self.coefficients.lo)
        retained = OutwardIntervalTensor.point(zeros)
        retained_by_exponent: dict[tuple[int, ...], OutwardIntervalTensor] = {}
        for left_slot, right_slot, output_slot, sign in self.plan.multiply_route_indices:
            product = OutwardIntervalTensor(
                self.coefficients.lo[..., left_slot],
                self.coefficients.hi[..., left_slot],
            ).mul(
                OutwardIntervalTensor(
                    other.coefficients.lo[..., right_slot],
                    other.coefficients.hi[..., right_slot],
                )
            ).scale(float(sign))
            output = OutwardIntervalTensor(
                retained.lo[..., output_slot], retained.hi[..., output_slot]
            ).add(product)
            retained_lo = retained.lo.clone()
            retained_hi = retained.hi.clone()
            retained_lo[..., output_slot] = output.lo
            retained_hi[..., output_slot] = output.hi
            retained = OutwardIntervalTensor(retained_lo, retained_hi)
            exponent = self.plan.exponent_tuples[output_slot]
            retained_by_exponent[exponent] = (
                retained_by_exponent[exponent].add(product)
                if exponent in retained_by_exponent
                else product
            )

        full_by_exponent: dict[tuple[int, ...], OutwardIntervalTensor] = {}
        for left_slot, left_exponent in enumerate(self.plan.exponent_tuples):
            for right_slot, right_exponent in enumerate(self.plan.exponent_tuples):
                exponent = tuple(
                    left_power + right_power
                    for left_power, right_power in zip(left_exponent, right_exponent)
                )
                product = OutwardIntervalTensor(
                    self.coefficients.lo[..., left_slot],
                    self.coefficients.hi[..., left_slot],
                ).mul(
                    OutwardIntervalTensor(
                        other.coefficients.lo[..., right_slot],
                        other.coefficients.hi[..., right_slot],
                    )
                )
                full_by_exponent[exponent] = (
                    full_by_exponent[exponent].add(product)
                    if exponent in full_by_exponent
                    else product
                )
        residual_terms: list[OutwardIntervalTensor] = []
        for exponent in sorted(set(full_by_exponent) | set(retained_by_exponent)):
            coefficient = full_by_exponent.get(exponent)
            retained_coefficient = retained_by_exponent.get(exponent)
            if coefficient is None:
                assert retained_coefficient is not None
                coefficient = OutwardIntervalTensor.zeros_like(retained_coefficient.lo)
            if retained_coefficient is not None:
                coefficient = coefficient.sub(retained_coefficient)
            monomial = OutwardIntervalTensor.point(
                torch.ones(box.lo.shape[0], dtype=box.lo.dtype, device=box.lo.device)
            )
            for variable_index, power in enumerate(exponent):
                variable = OutwardIntervalTensor(
                    box.lo[:, variable_index], box.hi[:, variable_index]
                )
                for _ in range(power):
                    monomial = monomial.mul(variable)
            residual_terms.append(
                coefficient.mul(
                    OutwardIntervalTensor(monomial.lo[:, None], monomial.hi[:, None])
                )
            )
        overflow = outward_sum(residual_terms)
        return OutwardFixedSupportPolynomial(retained, self.plan), overflow

    def integrate_project(
        self, box: OutwardIntervalTensor
    ) -> tuple["OutwardFixedSupportPolynomial", OutwardIntervalTensor]:
        zeros = torch.zeros_like(self.coefficients.lo)
        retained = OutwardIntervalTensor.point(zeros)
        exponent_to_slot = {
            exponent: slot for slot, exponent in enumerate(self.plan.exponent_tuples)
        }
        overflow_terms: list[OutwardIntervalTensor] = []
        for input_slot, exponent in enumerate(self.plan.exponent_tuples):
            integrated_exponent = list(exponent)
            factor = 1.0 / (integrated_exponent[self.plan.local_time_index] + 1)
            integrated_exponent[self.plan.local_time_index] += 1
            integrated_exponent_tuple = tuple(integrated_exponent)
            coefficient = OutwardIntervalTensor(
                self.coefficients.lo[..., input_slot],
                self.coefficients.hi[..., input_slot],
            ).scale(factor)
            output_slot = exponent_to_slot.get(integrated_exponent_tuple)
            if output_slot is not None:
                output = OutwardIntervalTensor(
                    retained.lo[..., output_slot], retained.hi[..., output_slot]
                ).add(coefficient)
                retained_lo = retained.lo.clone()
                retained_hi = retained.hi.clone()
                retained_lo[..., output_slot] = output.lo
                retained_hi[..., output_slot] = output.hi
                retained = OutwardIntervalTensor(retained_lo, retained_hi)
                continue
            monomial = OutwardIntervalTensor.point(
                torch.ones(box.lo.shape[0], dtype=box.lo.dtype, device=box.lo.device)
            )
            for variable_index, power in enumerate(integrated_exponent_tuple):
                variable = OutwardIntervalTensor(
                    box.lo[:, variable_index], box.hi[:, variable_index]
                )
                for _ in range(power):
                    monomial = monomial.mul(variable)
            overflow_terms.append(
                coefficient.mul(
                    OutwardIntervalTensor(monomial.lo[:, None], monomial.hi[:, None])
                )
            )
        overflow = (
            outward_sum(overflow_terms)
            if overflow_terms
            else OutwardIntervalTensor.zeros_like(self.coefficients.lo[..., 0])
        )
        return OutwardFixedSupportPolynomial(retained, self.plan), overflow


@dataclass(frozen=True)
class OutwardFixedSupportTaylorModel:
    polynomial: OutwardFixedSupportPolynomial
    remainder: OutwardIntervalTensor

    def add(self, other: "OutwardFixedSupportTaylorModel") -> "OutwardFixedSupportTaylorModel":
        return OutwardFixedSupportTaylorModel(
            self.polynomial.add(other.polynomial), self.remainder.add(other.remainder)
        )

    def sub(self, other: "OutwardFixedSupportTaylorModel") -> "OutwardFixedSupportTaylorModel":
        return OutwardFixedSupportTaylorModel(
            self.polynomial.sub(other.polynomial), self.remainder.sub(other.remainder)
        )

    def multiply(
        self,
        other: "OutwardFixedSupportTaylorModel",
        box: OutwardIntervalTensor,
    ) -> "OutwardFixedSupportTaylorModel":
        polynomial, overflow = self.polynomial.multiply_project(other.polynomial, box)
        left_range = self.polynomial.range(box)
        right_range = other.polynomial.range(box)
        remainder = outward_sum(
            (
                left_range.mul(other.remainder),
                right_range.mul(self.remainder),
                self.remainder.mul(other.remainder),
                overflow,
            )
        )
        return OutwardFixedSupportTaylorModel(polynomial, remainder)

    def integrate(self, box: OutwardIntervalTensor) -> "OutwardFixedSupportTaylorModel":
        polynomial, overflow = self.polynomial.integrate_project(box)
        time = OutwardIntervalTensor(
            box.lo[:, self.polynomial.plan.local_time_index, None],
            box.hi[:, self.polynomial.plan.local_time_index, None],
        )
        return OutwardFixedSupportTaylorModel(
            polynomial, overflow.add(self.remainder.mul(time))
        )

    def range(self, box: OutwardIntervalTensor) -> OutwardIntervalTensor:
        return self.polynomial.range(box).add(self.remainder)


@dataclass(frozen=True)
class OutwardFixedSupportStep:
    model: OutwardFixedSupportTaylorModel
    endpoint: OutwardIntervalTensor
    tube: OutwardIntervalTensor
    initial_inclusion_mask: torch.Tensor
    round_inclusion_masks: torch.Tensor
    accepted_mask: torch.Tensor
    finite_mask: torch.Tensor


@dataclass(frozen=True)
class OutwardFixedSupportReachResult:
    endpoint_lo: torch.Tensor
    endpoint_hi: torch.Tensor
    tube_hull_lo: torch.Tensor
    tube_hull_hi: torch.Tensor
    active_mask: torch.Tensor
    first_failure_index: torch.Tensor
    validated_steps: int
    requested_steps: int
    host_synchronizations: int
    device_transfers: int


def _linear_model_from_box(
    initial_box: OutwardIntervalTensor,
    plan: FixedSupportKernelPlan,
) -> OutwardFixedSupportTaylorModel:
    center = 0.5 * (initial_box.lo + initial_box.hi)
    left_radius = _up(center - initial_box.lo)
    right_radius = _up(initial_box.hi - center)
    radius = torch.maximum(left_radius, right_radius)
    coefficients = torch.zeros(
        (initial_box.lo.shape[0], initial_box.lo.shape[1], plan.num_slots),
        dtype=initial_box.lo.dtype,
        device=initial_box.lo.device,
    )
    coefficients[..., plan.constant_slot] = center
    coefficients[:, plan.state_indices, plan.spatial_linear_slots] = radius
    return OutwardFixedSupportTaylorModel(
        OutwardFixedSupportPolynomial.point(coefficients, plan),
        OutwardIntervalTensor.zeros_like(center),
    )


def _vdp_polynomial_rhs_outward(
    polynomial: OutwardFixedSupportPolynomial,
    box: OutwardIntervalTensor,
) -> tuple[OutwardFixedSupportPolynomial, OutwardIntervalTensor]:
    plan = polynomial.plan
    x = OutwardFixedSupportPolynomial(
        OutwardIntervalTensor(
            polynomial.coefficients.lo[:, 0:1], polynomial.coefficients.hi[:, 0:1]
        ),
        plan,
    )
    y = OutwardFixedSupportPolynomial(
        OutwardIntervalTensor(
            polynomial.coefficients.lo[:, 1:2], polynomial.coefficients.hi[:, 1:2]
        ),
        plan,
    )
    x_square, x_square_overflow = x.multiply_project(x, box)
    one_coefficients = torch.zeros_like(x.coefficients.lo)
    one_coefficients[..., plan.constant_slot] = 1.0
    one = OutwardFixedSupportPolynomial.point(one_coefficients, plan)
    factor = one.sub(x_square)
    product, product_overflow = factor.multiply_project(y, box)
    second = product.sub(x)
    coefficients = OutwardIntervalTensor(
        torch.cat((y.coefficients.lo, second.coefficients.lo), dim=1),
        torch.cat((y.coefficients.hi, second.coefficients.hi), dim=1),
    )
    zero = OutwardIntervalTensor.zeros_like(product_overflow.lo)
    overflow = OutwardIntervalTensor(
        torch.cat((zero.lo, product_overflow.add(x_square_overflow).lo), dim=1),
        torch.cat((zero.hi, product_overflow.add(x_square_overflow).hi), dim=1),
    )
    return OutwardFixedSupportPolynomial(coefficients, plan), overflow


def _vdp_tm_rhs_outward(
    model: OutwardFixedSupportTaylorModel,
    box: OutwardIntervalTensor,
) -> OutwardFixedSupportTaylorModel:
    plan = model.polynomial.plan
    x = OutwardFixedSupportTaylorModel(
        OutwardFixedSupportPolynomial(
            OutwardIntervalTensor(
                model.polynomial.coefficients.lo[:, 0:1],
                model.polynomial.coefficients.hi[:, 0:1],
            ),
            plan,
        ),
        OutwardIntervalTensor(model.remainder.lo[:, 0:1], model.remainder.hi[:, 0:1]),
    )
    y = OutwardFixedSupportTaylorModel(
        OutwardFixedSupportPolynomial(
            OutwardIntervalTensor(
                model.polynomial.coefficients.lo[:, 1:2],
                model.polynomial.coefficients.hi[:, 1:2],
            ),
            plan,
        ),
        OutwardIntervalTensor(model.remainder.lo[:, 1:2], model.remainder.hi[:, 1:2]),
    )
    one_coefficients = torch.zeros_like(x.polynomial.coefficients.lo)
    one_coefficients[..., plan.constant_slot] = 1.0
    one = OutwardFixedSupportTaylorModel(
        OutwardFixedSupportPolynomial.point(one_coefficients, plan),
        OutwardIntervalTensor.zeros_like(x.remainder.lo),
    )
    second = one.sub(x.multiply(x, box)).multiply(y, box).sub(x)
    return OutwardFixedSupportTaylorModel(
        OutwardFixedSupportPolynomial(
            OutwardIntervalTensor(
                torch.cat(
                    (y.polynomial.coefficients.lo, second.polynomial.coefficients.lo),
                    dim=1,
                ),
                torch.cat(
                    (y.polynomial.coefficients.hi, second.polynomial.coefficients.hi),
                    dim=1,
                ),
            ),
            plan,
        ),
        OutwardIntervalTensor(
            torch.cat((y.remainder.lo, second.remainder.lo), dim=1),
            torch.cat((y.remainder.hi, second.remainder.hi), dim=1),
        ),
    )


def fixed_support_outward_vdp_step(
    initial_box: OutwardIntervalTensor,
    plan: FixedSupportKernelPlan,
    *,
    step_size: float = 0.01,
    initial_remainder_radius: float = 0.01,
    polynomial_picard_iterations: int = 2,
    remainder_rounds: int = 10,
) -> OutwardFixedSupportStep:
    if initial_box.lo.device.type != "cpu":
        raise ValueError("the qualified outward reference is CPU-only")
    batch, state_dim = initial_box.lo.shape
    zero_time = torch.zeros((batch, 1), dtype=torch.float64)
    step_time = torch.full((batch, 1), float(step_size), dtype=torch.float64)
    ones = torch.ones((batch, state_dim), dtype=torch.float64)
    domain = OutwardIntervalTensor(
        torch.cat((zero_time, -ones), dim=1),
        torch.cat((step_time, ones), dim=1),
    )
    base = _linear_model_from_box(initial_box, plan)
    polynomial = base.polynomial
    for _ in range(polynomial_picard_iterations):
        rhs, rhs_overflow = _vdp_polynomial_rhs_outward(polynomial, domain)
        integrated, integration_overflow = rhs.integrate_project(domain)
        polynomial = base.polynomial.add(integrated)
        del rhs_overflow, integration_overflow

    radius = torch.full(
        (batch, state_dim), float(initial_remainder_radius), dtype=torch.float64
    )
    seed = OutwardIntervalTensor(-radius, radius)
    initial_rhs = _vdp_tm_rhs_outward(
        OutwardFixedSupportTaylorModel(polynomial, seed), domain
    )
    initial_integrated = initial_rhs.integrate(domain)
    initial_candidate = OutwardFixedSupportTaylorModel(
        base.polynomial.add(initial_integrated.polynomial), initial_integrated.remainder
    )
    initial_mask = (initial_integrated.remainder.lo >= seed.lo) & (
        initial_integrated.remainder.hi <= seed.hi
    )
    roundoff = initial_candidate.polynomial.sub(polynomial).range(domain)
    current = OutwardFixedSupportTaylorModel(polynomial, seed)
    masks: list[torch.Tensor] = []
    for _ in range(remainder_rounds):
        rhs = _vdp_tm_rhs_outward(current, domain)
        integrated = rhs.integrate(domain)
        candidate = OutwardFixedSupportTaylorModel(
            base.polynomial.add(integrated.polynomial),
            integrated.remainder.add(roundoff),
        )
        mask = (candidate.remainder.lo >= current.remainder.lo) & (
            candidate.remainder.hi <= current.remainder.hi
        )
        current = OutwardFixedSupportTaylorModel(
            candidate.polynomial,
            OutwardIntervalTensor(
                torch.where(mask, candidate.remainder.lo, current.remainder.lo),
                torch.where(mask, candidate.remainder.hi, current.remainder.hi),
            ),
        )
        masks.append(mask)
    endpoint_domain = OutwardIntervalTensor(
        torch.cat((step_time, -ones), dim=1),
        torch.cat((step_time, ones), dim=1),
    )
    endpoint = current.range(endpoint_domain)
    tube = current.range(domain)
    finite = (
        current.polynomial.coefficients.finite().flatten(1).all(dim=1)
        & current.remainder.finite().all(dim=1)
        & endpoint.finite().all(dim=1)
        & tube.finite().all(dim=1)
    )
    accepted = initial_mask.all(dim=1) & finite
    return OutwardFixedSupportStep(
        model=current,
        endpoint=endpoint,
        tube=tube,
        initial_inclusion_mask=initial_mask,
        round_inclusion_masks=torch.stack(masks, dim=0),
        accepted_mask=accepted,
        finite_mask=finite,
    )


def fixed_support_outward_vdp_verify(
    initial_lo: torch.Tensor,
    initial_hi: torch.Tensor,
    plan: FixedSupportKernelPlan,
    *,
    steps: int,
    step_size: float = 0.01,
) -> OutwardFixedSupportReachResult:
    """Run the independent box-recentered outward reference fail closed."""

    if initial_lo.device.type != "cpu" or initial_lo.dtype != torch.float64:
        raise ValueError("the qualified outward reference requires CPU float64")
    box = OutwardIntervalTensor(initial_lo, initial_hi)
    active = torch.ones(initial_lo.shape[0], dtype=torch.bool)
    failures = torch.full((initial_lo.shape[0],), -1, dtype=torch.long)
    hull_lo = torch.full_like(initial_lo, torch.inf)
    hull_hi = torch.full_like(initial_hi, -torch.inf)
    for step_index in range(int(steps)):
        result = fixed_support_outward_vdp_step(box, plan, step_size=step_size)
        accepted = active & result.accepted_mask
        failed = active & (~result.accepted_mask)
        failures = torch.where(failed & (failures < 0), step_index, failures)
        box = OutwardIntervalTensor(
            torch.where(accepted[:, None], result.endpoint.lo, box.lo),
            torch.where(accepted[:, None], result.endpoint.hi, box.hi),
        )
        hull_lo = torch.where(
            accepted[:, None], torch.minimum(hull_lo, result.tube.lo), hull_lo
        )
        hull_hi = torch.where(
            accepted[:, None], torch.maximum(hull_hi, result.tube.hi), hull_hi
        )
        active = accepted
    failure_values = [int(value) for value in failures.tolist() if int(value) >= 0]
    validated = int(steps) if not failure_values else min(failure_values)
    return OutwardFixedSupportReachResult(
        endpoint_lo=box.lo,
        endpoint_hi=box.hi,
        tube_hull_lo=hull_lo,
        tube_hull_hi=hull_hi,
        active_mask=active,
        first_failure_index=failures,
        validated_steps=validated,
        requested_steps=int(steps),
        host_synchronizations=1,
        device_transfers=0,
    )


__all__ = [
    "OutwardFixedSupportPolynomial",
    "OutwardFixedSupportReachResult",
    "OutwardFixedSupportStep",
    "OutwardFixedSupportTaylorModel",
    "OutwardIntervalTensor",
    "fixed_support_outward_vdp_step",
    "fixed_support_outward_vdp_verify",
    "outward_matmul",
    "outward_sum",
    "outward_where",
]
