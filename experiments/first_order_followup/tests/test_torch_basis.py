from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT.parents[1]
for path in (EXPERIMENT, REPO_ROOT / "src", EXPERIMENT.parent / "first_order_three_way"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import evaluate_rhs, load_spec
from torch_basis import (
    affine_reset,
    finite_basis_step_from_tm,
    normalized_initial_tm,
    retained_dictionary,
)
from torch_tm_flowpipe import TMVector


def _rhs(system):
    def rhs(state: TMVector, control=None):
        del control
        return TMVector(evaluate_rhs(list(state), system))

    return rhs


@pytest.mark.parametrize("basis", ["B1", "B_DR", "B2"])
def test_one_step_support_matches_finite_dictionary(basis):
    spec = load_spec(EXPERIMENT / "benchmark_spec.yaml")
    system = spec["systems"]["harmonic"]
    initial = normalized_initial_tm(system["initial_box"])
    segment, records = finite_basis_step_from_tm(
        _rhs(system), initial, 0.01, basis, picard_iterations=2
    )
    assert segment.status == "validated"
    allowed = set(
        retained_dictionary(
            basis,
            segment.tm.n_vars,
            tau_index=segment.tau_index,
        )
    )
    support = {
        exponent for model in segment.tm for exponent in model.polynomial.terms
    }
    assert support <= allowed
    if basis == "B1":
        assert all(sum(exponent) <= 1 for exponent in support)
    elif basis == "B_DR":
        assert any(
            exponent[segment.tau_index] == 1 and sum(exponent) == 2
            for exponent in support
        )
    else:
        assert all(sum(exponent) <= 2 for exponent in support)
    assert all(
        segment.final_tm.active_variables() <= set(range(initial.n_vars))
        for _ in [0]
    )
    assert all(len(record.exponent) == segment.tm.n_vars for record in records)


def test_bdr_keeps_time_state_but_discards_state_state():
    dictionary = set(retained_dictionary("B_DR", 3, tau_index=2))
    assert (1, 0, 1) in dictionary
    assert (0, 1, 1) in dictionary
    assert (0, 0, 2) in dictionary
    assert (2, 0, 0) not in dictionary
    assert (1, 1, 0) not in dictionary


def test_affine_box_and_qr_resets_contain_input():
    spec = load_spec(EXPERIMENT / "benchmark_spec.yaml")
    initial = normalized_initial_tm(spec["systems"]["harmonic"]["initial_box"])
    segment, _ = finite_basis_step_from_tm(
        _rhs(spec["systems"]["harmonic"]), initial, 0.01, "B_DR"
    )
    for method in ("box", "qr"):
        reset, stats = affine_reset(segment.final_tm, method=method)
        for reset_interval, input_interval in zip(
            reset.range_box(), segment.final_tm.range_box()
        ):
            assert reset_interval.contains_interval(input_interval, tol=1e-12)
        assert stats["method"] == method
        assert math.isfinite(stats["generator_condition_number"])
