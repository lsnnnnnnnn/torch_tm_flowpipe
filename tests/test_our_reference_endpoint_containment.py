"""Required mathematical assertion; keep visible failures until separately repaired."""
from fractions import Fraction

import pytest

from experiments.our_solver_performance.reference_endpoint import ENTRIES, endpoint_witness


@pytest.mark.parametrize("entry", ENTRIES)
def test_binary64_endpoint_full_model_contains_exact_value(entry):
    _, endpoint, exact = endpoint_witness(entry)
    assert exact == Fraction(3, 144115188075855872)
    # Test the substituted polynomial PLUS its entire ordinary remainder.
    # A point-coefficient comparison or decimal 1/100 would miss this defect.
    for component, bounds in enumerate(endpoint.range_box()):
        lo, hi = Fraction(float(bounds.lo)), Fraction(float(bounds.hi))
        assert lo <= exact <= hi, (
            f"{entry} component {component}: full range [{lo}, {hi}] excludes {exact}"
        )
