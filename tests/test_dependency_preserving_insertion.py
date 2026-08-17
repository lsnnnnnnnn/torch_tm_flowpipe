from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest
import torch

from torch_tm_flowpipe import (
    FlowstarNormalFlowpipeState,
    Interval,
    Polynomial,
    TaylorModel,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    insert_ctrunc_normal_dependency_preserving,
    insert_ctrunc_normal_horner_diagnostic,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode


RationalInterval = tuple[Fraction, Fraction]
RationalPolynomial = dict[tuple[int, ...], Fraction]


def _q(value: float | torch.Tensor) -> Fraction:
    return Fraction.from_float(float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value))


def _poly_add(left: RationalPolynomial, right: RationalPolynomial) -> RationalPolynomial:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def _poly_mul(left: RationalPolynomial, right: RationalPolynomial) -> RationalPolynomial:
    result: RationalPolynomial = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
            result[exponent] = result.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def _poly_pow(value: RationalPolynomial, power: int) -> RationalPolynomial:
    variables = len(next(iter(value)))
    result = {(0,) * variables: Fraction(1)}
    for _ in range(power):
        result = _poly_mul(result, value)
    return result


def _poly_range_bernstein(
    value: RationalPolynomial,
    domain: list[RationalInterval],
) -> RationalInterval:
    """Exact multivariate power-to-Bernstein enclosure on one box."""

    variables = len(domain)
    unit_power: RationalPolynomial = {}
    for exponent, coefficient in value.items():
        expanded = {(0,) * variables: coefficient}
        for variable, power in enumerate(exponent):
            factor: RationalPolynomial = {}
            lower, upper = domain[variable]
            width = upper - lower
            for unit_power_value in range(power + 1):
                term_exponent = [0] * variables
                term_exponent[variable] = unit_power_value
                factor[tuple(term_exponent)] = (
                    Fraction(math.comb(power, unit_power_value))
                    * lower ** (power - unit_power_value)
                    * width ** unit_power_value
                )
            expanded = _poly_mul(expanded, factor)
        unit_power = _poly_add(unit_power, expanded)
    degrees = [
        max((exponent[variable] for exponent in unit_power), default=0)
        for variable in range(variables)
    ]
    coefficients: list[Fraction] = []
    for beta in itertools.product(*(range(degree + 1) for degree in degrees)):
        bernstein = Fraction(0)
        for alpha, coefficient in unit_power.items():
            if any(a > b for a, b in zip(alpha, beta)):
                continue
            weight = Fraction(1)
            for a, b, degree in zip(alpha, beta, degrees):
                weight *= Fraction(math.comb(b, a), math.comb(degree, a))
            bernstein += coefficient * weight
        coefficients.append(bernstein)
    return min(coefficients), max(coefficients)


def _fraction_residual_range(
    outer: TaylorModel,
    inner: TMVector,
    result: TaylorModel,
) -> RationalInterval:
    base_variables = inner.n_vars
    augmented_variables = base_variables + len(inner)
    augmented_inner: list[RationalPolynomial] = []
    for remainder_index, model in enumerate(inner):
        terms = {
            tuple(exponent) + (0,) * len(inner): _q(coefficient)
            for exponent, coefficient in model.polynomial.terms.items()
        }
        remainder_exponent = [0] * augmented_variables
        remainder_exponent[base_variables + remainder_index] = 1
        terms[tuple(remainder_exponent)] = Fraction(1)
        augmented_inner.append(terms)

    exact: RationalPolynomial = {}
    for outer_exponent, outer_coefficient in outer.polynomial.terms.items():
        term = {(0,) * augmented_variables: _q(outer_coefficient)}
        for model, power in zip(augmented_inner, outer_exponent):
            term = _poly_mul(term, _poly_pow(model, int(power)))
        exact = _poly_add(exact, term)
    represented = {
        tuple(exponent) + (0,) * len(inner): -_q(coefficient)
        for exponent, coefficient in result.polynomial.terms.items()
    }
    residual = _poly_add(exact, represented)
    domain = [
        (_q(interval.lo), _q(interval.hi))
        for interval in inner.domain
    ] + [
        (_q(model.remainder.lo), _q(model.remainder.hi))
        for model in inner
    ]
    lower, upper = _poly_range_bernstein(residual, domain)
    lower += _q(outer.remainder.lo)
    upper += _q(outer.remainder.hi)
    return lower, upper


def _assert_fraction_oracle_contained(
    outer: TaylorModel,
    inner: TMVector,
    result: TaylorModel,
) -> None:
    lower, upper = _fraction_residual_range(outer, inner, result)
    assert _q(result.remainder.lo) <= lower
    assert _q(result.remainder.hi) >= upper


@pytest.mark.parametrize("seed", range(8))
def test_dependency_preserving_random_exact_micro_oracle(seed: int):
    generator = torch.Generator().manual_seed(seed)
    domain = [Interval(-0.75, 0.5), Interval(-0.25, 0.875)]
    terms: dict[tuple[int, int], float] = {}
    for exponent in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1)):
        numerator = int(torch.randint(-8, 9, (), generator=generator))
        if numerator:
            terms[exponent] = numerator / 16.0
    outer = TaylorModel(
        Polynomial(terms, 2),
        Interval(-2.0**-24, 3.0 * 2.0**-24),
        domain,
        order=5,
    )
    x = TaylorModel.variable(0, domain, order=5)
    y = TaylorModel.variable(1, domain, order=5)
    inner = TMVector(
        [
            0.125 + 0.5 * x - 0.25 * y + 0.125 * x * y + Interval(-2.0**-20, 3.0 * 2.0**-20),
            -0.25 + 0.125 * x + 0.5 * y - 0.0625 * x * x + Interval(-3.0 * 2.0**-21, 2.0**-21),
        ]
    )

    actual = insert_ctrunc_normal_dependency_preserving(
        outer, inner, order=4, cutoff_threshold=2.0**-18, domain=domain
    )
    assert isinstance(actual, TaylorModel)
    diagnostic = insert_ctrunc_normal_horner_diagnostic(
        outer, inner, order=4, cutoff_threshold=2.0**-18, domain=domain
    )
    expected = diagnostic.horner_result
    assert isinstance(expected, TaylorModel)
    assert actual.polynomial.terms.keys() == expected.polynomial.terms.keys()
    for exponent in actual.polynomial.terms:
        assert torch.equal(actual.polynomial.terms[exponent], expected.polynomial.terms[exponent])
    assert torch.equal(actual.remainder.lo, expected.remainder.lo)
    assert torch.equal(actual.remainder.hi, expected.remainder.hi)
    _assert_fraction_oracle_contained(outer, inner, actual)


def test_dependency_preserving_zero_remainder_cutoff_boundary_and_order_overflow():
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    x = TaylorModel.variable(0, domain, order=8)
    y = TaylorModel.variable(1, domain, order=8)
    inner = TMVector([x + 0.25 * y + 0.125 * x * y, y - 0.125 * x * x])
    threshold = 2.0**-20
    outer = x.pow_int(3) + 0.5 * x * y + threshold * y + Interval(-2.0**-25, 2.0**-24)
    diagnostics: dict[str, object] = {}

    actual = insert_ctrunc_normal_dependency_preserving(
        outer,
        inner,
        order=2,
        cutoff_threshold=threshold,
        domain=domain,
        diagnostics=diagnostics,
    )
    assert isinstance(actual, TaylorModel)
    assert diagnostics["insertion_canonical_variable_order"] == [0, 1]
    assert diagnostics["insertion_truncation_width"] > 0.0
    assert diagnostics["insertion_cutoff_width"] > 0.0
    assert diagnostics["insertion_inner_remainder_times_poly_width"] <= 2.0**-1060
    assert diagnostics["insertion_remainder_times_remainder_width"] <= 2.0**-1060
    _assert_fraction_oracle_contained(outer, inner, actual)


def _candidate_step(current, normal_state):
    return flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        current,
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion_dependency_preserving",
        flowstar_normal_state=normal_state,
    )


def test_dependency_preserving_checkpoint_resume_is_bitwise(tmp_path):
    initial_state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("1.1", "1.4"), ("2.35", "2.45")], 4
    )
    first = _candidate_step(initial_state.normalized_initial_tm(4), initial_state)
    assert first.status == "validated"
    assert first.reset_tm is not None and first.flowstar_normal_state is not None
    uninterrupted = _candidate_step(first.reset_tm, first.flowstar_normal_state)
    assert uninterrupted.status == "validated"

    checkpoint_dir = tmp_path / "checkpoint"
    save_terminal_checkpoint(
        checkpoint_dir,
        current=first.reset_tm,
        normal_state=first.flowstar_normal_state,
        scheduler={"current_time": 0.01, "h_next": 0.01},
        contract={"order": 4, "cutoff": 1e-10, "h": 0.01},
        provenance={"test": True},
    )
    loaded = load_terminal_checkpoint(checkpoint_dir, expected_order=4, expected_dtype="float64")
    resumed = _candidate_step(loaded.current, loaded.normal_state)
    assert resumed.status == "validated"
    assert uninterrupted.reset_tm is not None and resumed.reset_tm is not None
    for left, right in zip(uninterrupted.reset_tm, resumed.reset_tm):
        assert left.polynomial.terms.keys() == right.polynomial.terms.keys()
        for exponent in left.polynomial.terms:
            assert torch.equal(left.polynomial.terms[exponent], right.polynomial.terms[exponent])
        assert torch.equal(left.remainder.lo, right.remainder.lo)
        assert torch.equal(left.remainder.hi, right.remainder.hi)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dependency_preserving_cpu_cuda_consistency_only():
    def make(device: str):
        domain = [
            Interval(torch.tensor(-1.0, dtype=torch.float64, device=device), torch.tensor(1.0, dtype=torch.float64, device=device)),
            Interval(torch.tensor(-0.5, dtype=torch.float64, device=device), torch.tensor(0.75, dtype=torch.float64, device=device)),
        ]
        x = TaylorModel.variable(0, domain, order=5)
        y = TaylorModel.variable(1, domain, order=5)
        outer = x.pow_int(3) - 0.25 * x * y + y.pow_int(2) + Interval(
            torch.tensor(-1e-8, dtype=torch.float64, device=device),
            torch.tensor(2e-8, dtype=torch.float64, device=device),
        )
        inner = TMVector(
            [
                x + 0.2 * y + Interval(torch.tensor(-1e-6, dtype=torch.float64, device=device), torch.tensor(2e-6, dtype=torch.float64, device=device)),
                y - 0.1 * x + Interval(torch.tensor(-2e-6, dtype=torch.float64, device=device), torch.tensor(1e-6, dtype=torch.float64, device=device)),
            ]
        )
        return outer, inner, domain

    cpu_outer, cpu_inner, cpu_domain = make("cpu")
    cuda_outer, cuda_inner, cuda_domain = make("cuda")
    cpu = insert_ctrunc_normal_dependency_preserving(cpu_outer, cpu_inner, 4, 1e-10, cpu_domain)
    cuda = insert_ctrunc_normal_dependency_preserving(cuda_outer, cuda_inner, 4, 1e-10, cuda_domain)
    assert isinstance(cpu, TaylorModel) and isinstance(cuda, TaylorModel)
    assert cpu.polynomial.terms.keys() == cuda.polynomial.terms.keys()
    for exponent in cpu.polynomial.terms:
        assert torch.equal(cpu.polynomial.terms[exponent], cuda.polynomial.terms[exponent].cpu())
    assert torch.equal(cpu.remainder.lo, cuda.remainder.lo.cpu())
    assert torch.equal(cpu.remainder.hi, cuda.remainder.hi.cpu())
