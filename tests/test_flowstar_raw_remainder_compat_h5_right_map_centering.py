from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_h5_right_map_centering.py"
spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_h5_right_map_centering", SCRIPT)
assert spec is not None and spec.loader is not None
centering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(centering)


def _one_step(center_mode: str | None = None):
    kwargs = {}
    if center_mode is not None:
        kwargs["right_map_center_mode"] = center_mode
    return flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion",
        **kwargs,
    )


def test_default_right_map_center_mode_is_constant_and_matches_explicit_constant():
    default = _one_step()
    explicit = _one_step("constant")

    assert default.status == "validated"
    assert explicit.status == "validated"
    assert default.flowstar_normal_stats is not None
    assert explicit.flowstar_normal_stats is not None
    assert default.flowstar_normal_stats["right_map_center_mode"] == "constant"
    assert explicit.flowstar_normal_stats["right_map_center_mode"] == "constant"
    assert default.flowstar_normal_stats["center_x"] == explicit.flowstar_normal_stats["center_x"]
    assert default.flowstar_normal_stats["center_y"] == explicit.flowstar_normal_stats["center_y"]
    assert default.flowstar_normal_stats["centered_reset_width_sum"] == explicit.flowstar_normal_stats["centered_reset_width_sum"]


def test_range_midpoint_reconstruction_uses_polynomial_and_remainder_diffs():
    seg = _one_step("range_midpoint")

    assert seg.status == "validated"
    assert seg.flowstar_normal_stats is not None
    stats = seg.flowstar_normal_stats
    assert stats["right_map_center_mode"] == "range_midpoint"
    assert stats["reconstruction_polynomial_max_abs_diff"] <= 1e-15
    assert stats["reconstruction_remainder_lo_diff"] <= 1e-15
    assert stats["reconstruction_remainder_hi_diff"] <= 1e-15
    assert stats["inserted_range_midpoint_shift_abs_sum"] > 0.0


def test_centered_inserted_range_is_translated_without_width_growth_and_scale_reduces():
    seg = _one_step("range_midpoint")
    assert seg.flowstar_normal_stats is not None
    stats = seg.flowstar_normal_stats

    for dim in ("x", "y"):
        assert abs(stats[f"inserted_range_width_{dim}"] - stats[f"centered_inserted_range_width_{dim}"]) <= 1e-14
        assert stats[f"centered_scale_{dim}"] <= stats[f"baseline_scale_{dim}"] + 1e-14
    assert stats["centered_reset_width_sum"] <= stats["baseline_reset_width_sum"] + 1e-14


def test_reset_reconstruction_contains_validated_endpoint_box():
    seg = _one_step("range_midpoint")
    assert seg.flowstar_normal_state is not None
    reconstructed = seg.flowstar_normal_state.endpoint_tm().range_box()
    endpoint = seg.final_tm.range_box()

    for reconstructed_dim, endpoint_dim in zip(reconstructed, endpoint):
        assert reconstructed_dim.contains_interval(endpoint_dim, tol=1e-8)


def test_frozen_replay_preserves_prescribed_h_sequence_on_tiny_fixture():
    summary, rows, _samples = centering.run_centering_h5(
        mode="fixture_frozen",
        run_kind="fixture frozen",
        right_map_center_mode="constant",
        horizon=0.004,
        wall_cap_s=60.0,
        prescribed_h=[0.002, 0.002],
    )

    assert summary["reached_horizon"] is True
    assert len(rows) == 2
    assert all(row["status"] == "validated" for row in rows)
    assert all(abs(float(row["h_delta"])) <= 1e-15 for row in rows)


def test_validation_failure_rows_are_reported_not_forced_accepted():
    frozen_rows = centering.make_frozen_schedule_rows(
        [{"segment_index": 0, "status": "validated", "h": "0.1", "h_delta": "0", "width_sum": "1.0"}],
        [{"segment_index": 0, "status": "failed", "h": "0.1", "h_delta": "0", "width_sum": "1.5"}],
        [0.1],
    )

    assert frozen_rows[0]["range_midpoint_status"] == "failed"
    assert frozen_rows[0]["range_midpoint_validation_failure_recorded"] is True


def test_missing_flowstar_component_fields_are_unknown_not_zero():
    rows = centering._make_flowstar_segment_rows(
        [
            {
                "segment_index": 0,
                "status": "validated",
                "t_lo": 0.0,
                "t_hi": 0.1,
                "h": 0.1,
                "width_x": 1.0,
                "width_y": 2.0,
                "width_sum": 3.0,
            }
        ]
    )

    row = rows[0]
    assert row["missing_flowstar_component_fields"] == centering.UNKNOWN_FLOWSTAR_COMPONENT
    assert row["old_right_map_range_width_sum"] == centering.UNKNOWN_FLOWSTAR_COMPONENT
    assert row["old_right_map_range_width_sum"] != 0


def test_csv_physical_line_count_matches_csv_reader_rows(tmp_path):
    path = tmp_path / "ledger.csv"
    centering._write_csv(path, ["a", "b"], [{"a": "1", "b": "two, with comma"}, {"a": "3", "b": "4"}])

    with path.open(newline="", encoding="utf-8") as handle:
        parsed = sum(1 for _ in csv.reader(handle))
    assert centering.physical_line_count(path) == parsed
    assert centering.csv_row_count(path) == parsed


def test_experiment_is_h5_only_and_does_not_name_h10_outputs():
    text = SCRIPT.read_text(encoding="utf-8")

    assert centering.HORIZON_LIMIT <= 5.0
    assert "flowstar_raw_remainder_compat_h5_right_map_centering" in str(centering.DEFAULT_OUT_DIR)
    assert "flowstar_raw_remainder_compat_h10_right_map_centering" not in text
    assert "refusing to run h10" in text
