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
