from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from common import evaluate_polynomial_point, validate_record


@pytest.mark.parametrize("name", ["torch", "diffreach", "flowstar"])
def test_external_export_if_supplied(name: str, request: pytest.FixtureRequest) -> None:
    path = request.config.getoption(f"--{name}-segment")
    if not path:
        pytest.skip(f"--{name}-segment was not supplied")
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    checks = validate_record(record)
    assert checks["passed"], checks
    samples = record["native_metadata"].get("native_point_samples", [])
    for sample in samples:
        if sample.get("kind") not in (None, "tube"):
            continue
        state_index = int(sample.get("state", 0))
        point = sample["point"]
        actual = evaluate_polynomial_point(
            record["states"][state_index]["polynomial_terms"], point
        )
        if "polynomial_values" in sample:
            expected = sample["polynomial_values"][state_index]
            assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-12)
        elif "total_interval" in sample:
            remainder = record["states"][state_index][
                "independent_interval_remainder"
            ]
            expected = sample["total_interval"]
            assert actual + remainder[0] >= expected[0] - 5e-12
            assert actual + remainder[1] <= expected[1] + 5e-12


def test_flowstar_endpoint_and_adaptive_artifact_gates_if_supplied(
    request: pytest.FixtureRequest,
) -> None:
    supplied = request.config.getoption("--flowstar-segment")
    if not supplied:
        pytest.skip("--flowstar-segment was not supplied")
    path = Path(supplied)
    record = json.loads(path.read_text(encoding="utf-8"))
    endpoint_paths = record["native_metadata"]["endpoint_path_audit"]
    assert len(endpoint_paths) == record["state_dimension"]
    for state, audit in enumerate(endpoint_paths):
        endpoint = record["raw_endpoint_box"][state]
        assert math.isclose(
            endpoint[0],
            audit["native_lower"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            endpoint[1],
            audit["native_upper"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        collapsed = record["enclosures"]["endpoint_collapsed"]["box"][state]
        repaired = record["enclosures"]["repaired_hull"]["box"][state]
        assert collapsed == [audit["collapsed_lower"], audit["collapsed_upper"]]
        assert repaired == [audit["repaired_lower"], audit["repaired_upper"]]

    results = path.parents[1]
    correctness_path = results / "flowstar_correctness_summary.json"
    if not correctness_path.is_file():
        # A standalone current-run exporter check is complete at this point.
        # Historical adaptive-run assertions are exercised only when that
        # separate artifact is explicitly colocated with the supplied export.
        return
    correctness = json.loads(
        correctness_path.read_text(encoding="utf-8")
    )
    parity = correctness["original_parity"]
    assert parity["passed"]
    assert parity["original_reached_horizon_10"]
    assert parity["schedule_agreement"]
    assert parity["original_segments"] == 290
    adaptive = json.loads(
        (
            results
            / "flowstar_adaptive_trajectory_audit"
            / "flowstar_adaptive_trajectory_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert adaptive["passed"]
    assert adaptive["first_failure"]["segment_index"] == 3
    assert adaptive["authoritative_repaired_trajectory_failures"] == 0
    assert not adaptive["excluded_from_authoritative"]
