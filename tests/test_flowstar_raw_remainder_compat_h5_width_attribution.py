from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_h5_width_attribution.py"
spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_h5_width_attribution", SCRIPT)
assert spec is not None and spec.loader is not None
attr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attr)

H5_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5"
DIVERGENCE_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5_divergence"


def test_fixture_detects_component_that_first_crosses_threshold():
    row = {
        "raw_ctrunc_residual_growth_vs_previous_event": 1.05,
        "post_cutoff_residual_growth_vs_previous_event": 1.01,
        "right_map_range_growth_vs_previous_event": 1.30,
        "reset_width_growth_vs_previous_event": 1.10,
        "full_step_tube_growth_vs_previous_event": 1.20,
        "polynomial_range_growth_vs_previous_event": 1.02,
    }

    assert attr.leading_component(row) == "right_map_range"


def test_fixture_distinguishes_raw_residual_vs_right_map_growth():
    raw_driven = {
        "raw_ctrunc_residual_growth_vs_previous_event": 2.0,
        "right_map_range_growth_vs_previous_event": 1.2,
    }
    right_map_driven = {
        "raw_ctrunc_residual_growth_vs_previous_event": 1.1,
        "right_map_range_growth_vs_previous_event": 1.8,
    }

    assert attr.raw_or_right_map_dominates(raw_driven) == "raw_ctrunc_residual"
    assert attr.raw_or_right_map_dominates(right_map_driven) == "right_map_range"


def test_missing_flowstar_component_fields_are_unknown_not_zero():
    event = {"event_name": "fixture", "threshold": "", "flowstar_index": 0}
    segment = {
        "segment_index": 0,
        "t_hi": "0.1",
        "h": "0.1",
        "status": "validated",
        "width_x": "1.0",
        "width_y": "2.0",
        "width_sum": "3.0",
    }

    row = attr._flowstar_event_row(event, segment)

    assert row["raw_ctrunc_residual_y_hi"] == ""
    assert row["right_map_range_width_sum"] == ""
    assert row["reset_width_sum"] == ""
    assert row["raw_ctrunc_residual_y_hi"] != 0
    assert "unknown, not zero" in row["notes"]


def test_h5_width_attribution_has_no_h10_outputs():
    assert "h10" not in str(attr.DEFAULT_OUT_DIR)
    assert not (ROOT / "outputs" / "flowstar_raw_remainder_compat_h10_width_attribution").exists()
    assert "refusing to write h10 outputs" in SCRIPT.read_text(encoding="utf-8")


def test_csv_physical_row_count_guard_uses_csv_reader():
    rows = attr.formatting_rows(H5_DIR, DIVERGENCE_DIR)

    assert rows
    csv_rows = [row for row in rows if row["path"].endswith(".csv")]
    assert csv_rows
    for row in csv_rows:
        assert row["physical_line_count"] == row["csv_reader_row_count"]
        assert row["physical_line_count"] > 1
        assert row["status"] == "ok"

    assert attr.csv_row_count(H5_DIR / "h5_summary.csv") == 6
