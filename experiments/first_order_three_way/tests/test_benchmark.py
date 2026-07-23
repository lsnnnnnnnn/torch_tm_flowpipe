from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import pytest
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (EXPERIMENT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RAW_FIELDS, evaluate_rhs, exact_endpoint, exact_steps, flowstar_expression, load_spec
from run_flowstar import render_cpp
from run_torch import _ode
from torch_tm_flowpipe import Interval, TMVector, flowpipe_multi_step


@pytest.fixture(scope="module")
def spec():
    return load_spec(EXPERIMENT / "benchmark_spec.yaml")


def _eval_flowstar(expressions, names, point):
    namespace = dict(zip(names, point))
    return [
        eval(expression.replace("^", "**"), {"__builtins__": {}}, namespace)
        for expression in expressions
    ]


def test_all_adapter_equations_match_at_random_points(spec):
    rng = random.Random(20260723)
    for system in spec["systems"].values():
        names = system["state_names"]
        expressions = [flowstar_expression(poly, names) for poly in system["rhs"]]
        torch_rhs = _ode(system)
        for _ in range(20):
            point = [rng.uniform(-2.0, 2.0) for _ in names]
            expected = [float(value) for value in evaluate_rhs(point, system)]
            flowstar = _eval_flowstar(expressions, names, point)
            # Constants exercise the exact same generic RHS without confusing
            # physical state values with TMVector.identity's normalized
            # generator coordinates.
            tm_state = TMVector.constants(point, [], order=1)
            torch_values = [
                float(model.evaluate_point([]))
                for model in torch_rhs(tm_state)
            ]
            assert flowstar == pytest.approx(expected, abs=1e-14)
            assert torch_values == pytest.approx(expected, abs=1e-14)


def test_initial_boxes_and_state_order_are_canonical(spec):
    assert spec["systems"]["riccati"]["state_names"] == ["x"]
    assert spec["systems"]["riccati"]["initial_box"] == [[0.0, 0.1]]
    assert spec["systems"]["harmonic"]["state_names"] == ["x1", "x2"]
    assert spec["systems"]["harmonic"]["initial_box"] == [[-0.1, 0.1], [-0.1, 0.1]]
    assert spec["systems"]["van_der_pol"]["state_names"] == ["x1", "x2"]
    assert spec["systems"]["van_der_pol"]["initial_box"] == [[1.1, 1.4], [2.35, 2.45]]


def test_every_requested_grid_has_exact_integer_step_count(spec):
    for system in spec["systems"].values():
        for h in system["step_sizes"]:
            for horizon in system["horizons"]:
                steps = exact_steps(h, horizon)
                assert math.isclose(steps * h, horizon, rel_tol=0.0, abs_tol=1e-12)


def test_torch_main_protocol_retains_only_total_degree_one(spec):
    system = spec["systems"]["riccati"]
    result = flowpipe_multi_step(
        _ode(system),
        [Interval(*system["initial_box"][0])],
        h=0.01,
        steps=3,
        order=1,
        mode="dependency_preserving",
        symbolic_remainder=False,
    )
    assert result.status == "validated"
    for segment in result.segments:
        assert segment.order == 1
        assert all(model.polynomial.degree() <= 1 for model in segment.tm)
        assert all(model.polynomial.degree() <= 1 for model in segment.final_tm)


def test_flowstar_primary_source_requests_fixed_order_one_and_no_adaptation(spec):
    source = render_cpp(
        spec["systems"]["riccati"],
        h=0.01,
        horizon=0.1,
        order=1,
        remainder_estimation=spec["flowstar"]["remainder_estimation"],
        cutoff=spec["flowstar"]["cutoff"],
    )
    assert "setFixedStepsize(0.01, 1)" in source
    assert "setAdaptiveStepsize" not in source
    assert "setFixedStepsize(0.01, 1," not in source
    assert "FLOWSTAR_UNSUPPORTED_ORDER" in source


def _results_dir() -> Path | None:
    value = os.environ.get("BENCHMARK_RESULTS_DIR")
    return Path(value) if value else None


def test_raw_schema_complete_when_results_are_supplied():
    directory = _results_dir()
    if directory is None:
        pytest.skip("BENCHMARK_RESULTS_DIR not set")
    with (directory / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == RAW_FIELDS
        rows = list(reader)
    assert rows
    assert all(row["status"] for row in rows)
    assert all(row["requested_order_label"] for row in rows)
    assert all(row["retained_basis"] for row in rows)


def test_exact_riccati_and_harmonic_checks_pass_or_downgrade_the_run():
    directory = _results_dir()
    if directory is None:
        pytest.skip("BENCHMARK_RESULTS_DIR not set")
    checks = json.loads((directory / "correctness_checks.json").read_text())
    with (directory / "raw_results.csv").open(newline="", encoding="utf-8") as handle:
        status_by_run = {
            row["run_id"]: row["status"]
            for row in csv.DictReader(handle)
        }
    exact_checks = [
        check for check in checks["checks"]
        if check["check"] == "exact_endpoint_containment" and check["checked"]
    ]
    assert exact_checks
    for check in exact_checks:
        if check["violations"]:
            assert status_by_run[check["run_id"]] == "sample_violation"
        else:
            assert status_by_run[check["run_id"]] == "certified_ok"


def test_sampled_trajectory_tube_checks_pass_but_are_not_called_a_proof():
    directory = _results_dir()
    if directory is None:
        pytest.skip("BENCHMARK_RESULTS_DIR not set")
    checks = json.loads((directory / "correctness_checks.json").read_text())
    sample_checks = [
        check for check in checks["checks"]
        if check["check"] == "sampled_trajectory_tube_containment" and check["checked"]
    ]
    assert sample_checks
    assert all(check["violations"] == 0 for check in sample_checks)
    assert checks["sample_checks_are_formal_proof"] is False


def test_closed_form_reference_values():
    riccati = exact_endpoint("riccati", 1.0, [[0.0, 0.1]])
    assert riccati is not None
    assert riccati[0] == pytest.approx((0.0, 1.0 / 9.0))
    harmonic = exact_endpoint("harmonic", 0.0, [[-0.1, 0.1], [-0.1, 0.1]])
    assert harmonic is not None
    assert harmonic[0] == pytest.approx((-0.1, 0.1))
    assert harmonic[1] == pytest.approx((-0.1, 0.1))
