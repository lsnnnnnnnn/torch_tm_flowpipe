from __future__ import annotations

from fractions import Fraction

from defect_diagnostic import (
    polynomial_defect,
    polynomial_from_terms,
)


def test_exact_rational_riccati_equilibrium_has_zero_defect():
    polynomial = polynomial_from_terms(
        [{"exponents": [0, 0], "coefficient": 0}], exact=True
    )
    rhs = [{"terms": [{"coefficient": 1, "powers": [2]}]}]
    assert polynomial_defect([polynomial], rhs, 0, exact=True) == [{}]


def test_exact_rational_coupled_constant_defect():
    half = {(0, 0, 0): Fraction(1, 2)}
    quarter = {(0, 0, 0): Fraction(1, 4)}
    rhs = [
        {"terms": [{"coefficient": 1, "powers": [1, 1]}]},
        {
            "terms": [
                {"coefficient": 1, "powers": [2, 0]},
                {"coefficient": -1, "powers": [0, 1]},
            ]
        },
    ]
    defect = polynomial_defect(
        [half, quarter], rhs, 0, exact=True
    )
    assert defect[0] == {(0, 0, 0): Fraction(-1, 8)}
    assert defect[1] == {}


def test_exact_rational_constant_velocity_polynomial_is_exact():
    polynomial = {
        (0, 0): Fraction(3, 7),
        (1, 0): Fraction(1, 1),
    }
    rhs = [{"terms": [{"coefficient": 1, "powers": [0]}]}]
    assert polynomial_defect(
        [polynomial], rhs, 0, exact=True
    ) == [{}]
