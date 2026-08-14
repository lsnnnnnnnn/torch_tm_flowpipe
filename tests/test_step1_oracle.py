from __future__ import annotations

from fractions import Fraction

import pytest

from torch_tm_flowpipe.step1_oracle import (
    H,
    RationalInterval,
    RationalPolynomial,
    complete_support,
    exact_fixture_polynomials,
    exact_picard_iterations,
    exact_step1_polynomials,
    exact_step1_remainder_oracle,
    exact_vdp_time_series,
    formal_true_solution_enclosure,
)


@pytest.mark.unit
def test_complete_o4_support_has_35_monomials() -> None:
    support = complete_support(3, 4)
    assert len(support) == 35
    assert support[0] == (0, 0, 0)
    assert all(sum(item) <= 4 for item in support)


@pytest.mark.unit
def test_exact_flowstar_staged_and_torch_complete_picard_agree() -> None:
    flowstar = exact_picard_iterations(construction="flowstar_staged")
    torch = exact_picard_iterations(construction="torch_complete")
    assert (flowstar[-1].image_x, flowstar[-1].image_y) == (
        torch[-1].image_x,
        torch[-1].image_y,
    )
    x, y = exact_step1_polynomials()
    assert len(x.terms) == 13
    assert len(y.terms) == 18
    assert x.degree == y.degree == 4


@pytest.mark.unit
def test_hand_checked_affine_through_quartic_fixtures() -> None:
    fixtures = exact_fixture_polynomials()
    domain = (RationalInterval(-1, 1), RationalInterval(-1, 1))
    assert fixtures["affine"].natural_range(domain) == RationalInterval(Fraction(-3, 2), Fraction(7, 2))
    assert fixtures["quadratic"].terms[(1, 1)] == -1
    assert fixtures["cubic"].terms[(2, 1)] == -4
    assert fixtures["quartic"].degree == 4


@pytest.mark.unit
def test_exact_integral_and_truncation_are_hand_checkable() -> None:
    p = RationalPolynomial(2, {(0, 0): 2, (1, 0): 3, (3, 1): 5})
    retained, discarded = p.truncate(2)
    assert retained.terms == {(0, 0): 2, (1, 0): 3}
    assert discarded.terms == {(3, 1): 5}
    integral, overflow = retained.integrate(0, max_total_degree=2)
    assert integral.terms == {(1, 0): 2, (2, 0): Fraction(3, 2)}
    assert not overflow.terms


@pytest.mark.unit
def test_exact_remainder_oracle_closes_target_self_map() -> None:
    oracle = exact_step1_remainder_oracle()
    assert oracle.refinement
    assert all(row.subset_x and row.subset_y for row in oracle.refinement)
    assert oracle.final_remainder_x.subseteq(RationalInterval.symmetric(Fraction(1, 10_000)))
    assert oracle.final_remainder_y.subseteq(RationalInterval.symmetric(Fraction(1, 10_000)))
    assert oracle.endpoint_polynomial_x.lo <= oracle.endpoint_polynomial_x.hi
    assert oracle.segment_final_y.lo <= oracle.endpoint_final_y.lo
    assert H == Fraction(1, 100)


@pytest.mark.unit
def test_formal_corner_enclosure_has_strict_picard_and_cauchy_bounds() -> None:
    enclosure = formal_true_solution_enclosure(series_degree=100)
    assert enclosure.self_map_x_bound < enclosure.state_ball_radius
    assert enclosure.self_map_y_bound < enclosure.state_ball_radius
    assert enclosure.contraction_bound < 1
    assert enclosure.endpoint_x.lo > Fraction(1123, 1000)
    assert enclosure.endpoint_x.hi < Fraction(1425, 1000)
    assert enclosure.endpoint_y.lo > Fraction(2312, 1000)
    assert enclosure.endpoint_y.hi < Fraction(2434, 1000)
    assert enclosure.segment_x.lo == Fraction(11, 10)
    assert enclosure.segment_y.hi == Fraction(49, 20)


@pytest.mark.unit
def test_exact_time_series_starts_with_the_frozen_ode() -> None:
    x, y = exact_vdp_time_series(Fraction(11, 10), Fraction(47, 20), degree=3)
    assert x[0] == Fraction(11, 10)
    assert y[0] == Fraction(47, 20)
    assert x[1] == y[0]
    assert y[1] == y[0] - x[0] - x[0] * x[0] * y[0]
