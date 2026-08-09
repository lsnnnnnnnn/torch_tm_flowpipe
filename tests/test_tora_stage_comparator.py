from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tora_q3_stage_parity_fused_20260809"


@pytest.mark.regression
@pytest.mark.protocol
def test_stage_first_divergence_table_preserves_all_stage_contract_fields() -> None:
    with (OUTPUT / "stage_parity/stage_first_divergence.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["stage"] for row in rows] == [f"A{index}" for index in range(13)]
    assert rows[0]["classification"] == "numerically_negligible"
    assert rows[1]["classification"] == "numerically_negligible"
    assert rows[2]["classification"] == "expected_outward_roundoff"
    assert rows[3]["classification"] == "dominant_candidate"
    assert float(rows[2]["width_difference_maximum_absolute"]) < 1e-13
    assert float(rows[3]["remainder_contribution_difference"]) > 1e-2


@pytest.mark.regression
@pytest.mark.protocol
def test_r1_r2_csv_splits_center_radius_width_and_remainder() -> None:
    with (OUTPUT / "stage_parity/r1_r2_center_radius_remainder.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {(row["replay"], row["segment"]) for row in rows} == {
        ("R1", "10"),
        ("R2", "40"),
    }
    assert {row["quantity"] for row in rows} == {
        "interval_remainder",
        "endpoint",
        "tube",
    }
    for row in rows:
        width = float(row["width_difference_maximum_absolute"])
        radius_bound = float(row["radius_difference_maximum_absolute_upper_bound"])
        assert radius_bound == pytest.approx(width / 2.0)
        assert float(row["center_difference_maximum_absolute"]) >= 0.0


@pytest.mark.regression
def test_material_root_cause_is_a3_not_point_roundoff_or_coordinate_map() -> None:
    result = json.loads(
        (OUTPUT / "stage_parity/root_cause.json").read_text(encoding="utf-8")
    )
    assert result["first_differences"]["first_material"] == {
        "stage": "A3",
        "segment": 1,
        "leaf": 0,
        "reason": "same-input sine remainder width difference exceeds 1e-3 while point and retained-polynomial errors remain roundoff-scale",
    }
    assert result["dominant_candidate"]["point_enclosure_maximum_error"] < 5e-15
    assert result["segment_40_remainder_attribution"][
        "dominant_accumulated_ledger_category"
    ] == "composition_overflow"
