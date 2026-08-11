import math

import pytest
import torch

from torch_tm_flowpipe.s1_boundary_attribution import (
    BOUNDARY_STAGE_NAMES,
    S1BoundaryAttributionRecord,
    S1BoundaryStage,
    compare_binary64_scalar,
    compare_interval,
    ulp_distance,
)


def test_scalar_comparison_preserves_exact_hex_ulp_and_absolute_difference():
    left = 1.0
    right = math.nextafter(left, math.inf)
    comparison = compare_binary64_scalar(left, right)
    assert comparison["exact_binary64_equal"] is False
    assert comparison["left_hex"] != comparison["right_hex"]
    assert comparison["ulp_distance"] == ulp_distance(left, right) == 1
    assert comparison["absolute_difference"] == right - left
    assert compare_binary64_scalar(-0.0, 0.0)["exact_binary64_equal"] is False


def test_interval_comparison_reports_endpoints_ratio_and_containment_direction():
    comparison = compare_interval(-2.0, 3.0, -1.0, 2.0)
    assert comparison["left_endpoint_difference"] == 1.0
    assert comparison["right_endpoint_difference"] == -1.0
    assert comparison["width_ratio_right_over_left"] == 3.0 / 5.0
    assert comparison["componentwise_containment_direction"] == "left_contains_right"
    assert comparison["left_endpoint"]["ulp_distance"] > 0
    with pytest.raises(ValueError, match="inverted"):
        compare_interval(1.0, 0.0, 0.0, 1.0)


def test_boundary_record_requires_every_named_stage_with_declared_units():
    zero = torch.zeros((1, 2), dtype=torch.float64)
    stages = tuple(
        S1BoundaryStage(name, f"stage {name}", "old normalized", zero, zero)
        for name in BOUNDARY_STAGE_NAMES
    )
    record = S1BoundaryAttributionRecord(3, "C_current", stages, {})
    serialized = record.as_dict()
    assert [row["stage"] for row in serialized["stages"]] == list(BOUNDARY_STAGE_NAMES)
    assert all(row["units"] for row in serialized["stages"])
    with pytest.raises(ValueError, match="A0..B16"):
        S1BoundaryAttributionRecord(3, "C_current", stages[:-1], {})
