from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest
import torch

from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    FixedSupportReachability,
    diffreach_vdp_polynomial_rhs,
    diffreach_vdp_tm_rhs,
    fixed_support_kernel_plan,
)
from torch_tm_flowpipe.fixed_support_outward import (
    OutwardFixedSupportPolynomial,
    OutwardIntervalTensor,
    fixed_support_outward_vdp_step,
    fixed_support_outward_vdp_verify,
    outward_matmul,
)


def _q(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _contains(interval: OutwardIntervalTensor, exact: Fraction, index=()) -> bool:
    return _q(interval.lo[index].item()) <= exact <= _q(interval.hi[index].item())


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((1.0, 1.0), (2.0, 2.0)),
        ((-3.0, -1.0), (2.0, 5.0)),
        ((-2.0, 4.0), (-7.0, 3.0)),
        ((torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item(),) * 2, (2.0, 2.0)),
    ],
)
def test_outward_interval_mul_contains_independent_fraction_corners(left, right):
    a = OutwardIntervalTensor(
        torch.tensor(left[0], dtype=torch.float64),
        torch.tensor(left[1], dtype=torch.float64),
    )
    b = OutwardIntervalTensor(
        torch.tensor(right[0], dtype=torch.float64),
        torch.tensor(right[1], dtype=torch.float64),
    )
    result = a.mul(b)
    exact = [_q(x) * _q(y) for x, y in product(left, right)]
    assert _q(result.lo.item()) <= min(exact)
    assert _q(result.hi.item()) >= max(exact)


def test_outward_sequential_matmul_contains_fraction_cancellation_oracle():
    left_values = [[1.0e16, 1.0, -1.0e16], [1.0 / 3.0, -2.0, 7.0]]
    right_values = [[1.0, 2.0], [1.0, -3.0], [1.0, 4.0]]
    left = OutwardIntervalTensor.point(torch.tensor([left_values], dtype=torch.float64))
    right = OutwardIntervalTensor.point(torch.tensor([right_values], dtype=torch.float64))
    result = outward_matmul(left, right)
    for row in range(2):
        for column in range(2):
            exact = sum(
                _q(left_values[row][inner]) * _q(right_values[inner][column])
                for inner in range(3)
            )
            assert _contains(result, exact, (0, row, column))


def _plan():
    descriptor = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    return descriptor, fixed_support_kernel_plan(
        descriptor, device="cpu", dtype=torch.float64
    )


def test_projected_multiply_duplicate_routes_and_overflow_contain_fraction_oracle():
    _, plan = _plan()
    left_values = [1.0 / 3.0, -2.0, 0.5, 3.0, -0.25, 0.75, -1.5]
    right_values = [-0.2, 4.0, -3.0, 0.125, 2.0, -0.5, 0.6]
    left_tensor = torch.tensor([[left_values]], dtype=torch.float64)
    right_tensor = torch.tensor([[right_values]], dtype=torch.float64)
    box = OutwardIntervalTensor(
        torch.tensor([[0.0, -0.75, -0.2]], dtype=torch.float64),
        torch.tensor([[0.01, 0.5, 0.9]], dtype=torch.float64),
    )
    retained, overflow = OutwardFixedSupportPolynomial.point(
        left_tensor, plan
    ).multiply_project(OutwardFixedSupportPolynomial.point(right_tensor, plan), box)
    for output_slot in range(plan.num_slots):
        exact = Fraction(0)
        for left_slot, right_slot, route_output, sign in plan.multiply_route_indices:
            if route_output == output_slot:
                exact += (
                    _q(left_values[left_slot])
                    * _q(right_values[right_slot])
                    * sign
                )
        assert _contains(retained.coefficients, exact, (0, 0, output_slot))

    exponent_to_slot = {
        exponent: slot for slot, exponent in enumerate(plan.exponent_tuples)
    }
    for point_values in product((0.0, 0.01), (-0.75, 0.5), (-0.2, 0.9)):
        full = Fraction(0)
        kept = Fraction(0)
        for left_slot, left_exp in enumerate(plan.exponent_tuples):
            for right_slot, right_exp in enumerate(plan.exponent_tuples):
                exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
                monomial = Fraction(1)
                for value, power in zip(point_values, exponent):
                    monomial *= _q(value) ** power
                full += _q(left_values[left_slot]) * _q(right_values[right_slot]) * monomial
        for slot, exponent in enumerate(plan.exponent_tuples):
            monomial = Fraction(1)
            for value, power in zip(point_values, exponent):
                monomial *= _q(value) ** power
            exact_retained = Fraction(0)
            for left_slot, right_slot, route_output, sign in plan.multiply_route_indices:
                if route_output == slot:
                    exact_retained += _q(left_values[left_slot]) * _q(right_values[right_slot]) * sign
            kept += exact_retained * monomial
        residual = full - kept
        assert _q(overflow.lo[0, 0].item()) <= residual <= _q(overflow.hi[0, 0].item())
    assert exponent_to_slot[(2, 0, 0)] == 4


def test_projected_integration_contains_fraction_factors_and_discarded_terms():
    _, plan = _plan()
    values = [1.0 / 3.0, -2.0, 0.5, 3.0, -0.25, 0.75, -1.5]
    polynomial = OutwardFixedSupportPolynomial.point(
        torch.tensor([[values]], dtype=torch.float64), plan
    )
    box = OutwardIntervalTensor(
        torch.tensor([[0.0, -1.0, -1.0]], dtype=torch.float64),
        torch.tensor([[0.01, 1.0, 1.0]], dtype=torch.float64),
    )
    integrated, overflow = polynomial.integrate_project(box)
    for input_slot, output_slot in zip(
        plan.integration_input_indices, plan.integration_output_indices
    ):
        exponent = plan.exponent_tuples[input_slot]
        exact = _q(values[input_slot]) / (
            exponent[plan.local_time_index] + 1
        )
        assert _contains(integrated.coefficients, exact, (0, 0, output_slot))
    assert torch.all(overflow.lo <= overflow.hi)


def test_outward_vdp_step_contains_ordinary_eager_endpoint_and_tube():
    descriptor, plan = _plan()
    lo = torch.tensor([[1.1, 2.35]], dtype=torch.float64)
    hi = torch.tensor([[1.4, 2.45]], dtype=torch.float64)
    ordinary = FixedSupportReachability(
        support=descriptor,
        state_dim=2,
        polynomial_rhs=diffreach_vdp_polynomial_rhs,
        tm_rhs=diffreach_vdp_tm_rhs,
        step_size=0.01,
        initial_remainder=0.01,
        polynomial_picard_iterations=2,
        remainder_rounds=10,
        symbolic_window_size=1000,
    ).verify(lo, hi, steps=1)
    outward = fixed_support_outward_vdp_step(OutwardIntervalTensor(lo, hi), plan)
    assert outward.accepted_mask.tolist() == [True]
    assert torch.all(outward.endpoint.lo <= ordinary.endpoint_lo[:, 1])
    assert torch.all(outward.endpoint.hi >= ordinary.endpoint_hi[:, 1])
    assert torch.all(outward.tube.lo <= ordinary.tube_lo[:, 0])
    assert torch.all(outward.tube.hi >= ordinary.tube_hi[:, 0])


@pytest.mark.slow
def test_outward_vdp_reference_completes_tenth_step_without_nonfinite_output():
    _, plan = _plan()
    lo = torch.tensor([[1.1, 2.35]], dtype=torch.float64)
    hi = torch.tensor([[1.4, 2.45]], dtype=torch.float64)
    result = fixed_support_outward_vdp_verify(lo, hi, plan, steps=10)
    assert result.validated_steps == 10
    assert result.active_mask.tolist() == [True]
    assert torch.isfinite(result.endpoint_lo).all()
    assert torch.isfinite(result.endpoint_hi).all()


def test_outward_reference_fails_closed_on_large_nonfinite_workload():
    _, plan = _plan()
    lo = torch.tensor([[1.0e308, 1.0e308]], dtype=torch.float64)
    hi = torch.tensor([[1.1e308, 1.1e308]], dtype=torch.float64)
    step = fixed_support_outward_vdp_step(OutwardIntervalTensor(lo, hi), plan)
    assert not step.accepted_mask.item()
    assert not step.finite_mask.item()
