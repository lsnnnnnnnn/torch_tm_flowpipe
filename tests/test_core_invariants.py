from __future__ import annotations

import pytest

from torch_tm_flowpipe import Interval, Polynomial


pytestmark = [pytest.mark.unit, pytest.mark.property]


@pytest.mark.parametrize(
    ("inner", "outer"),
    [
        (Interval(-0.5, 0.5), Interval(-1.0, 1.0)),
        (Interval(0.0, 0.25), Interval(-0.1, 0.5)),
        (Interval(-2.0, -1.0), Interval(-3.0, 0.0)),
    ],
)
def test_interval_widening_preserves_enclosure(
    inner: Interval, outer: Interval
) -> None:
    other = Interval(-0.75, 1.25)
    assert (outer + other).contains_interval(inner + other)
    assert (outer * other).contains_interval(inner * other)


def test_polynomial_zero_and_identity_are_stable() -> None:
    x = Polynomial.variable(0, 2)
    zero = Polynomial.zero(2)
    assert (x + zero).terms.keys() == x.terms.keys()
    assert (x * 1.0).terms.keys() == x.terms.keys()
    assert not (x * 0.0).terms
