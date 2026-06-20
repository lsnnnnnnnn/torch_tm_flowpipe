from __future__ import annotations

import csv
import importlib.util
import inspect
from pathlib import Path

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_h10_right_map_centering.py"
spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_h10_right_map_centering", SCRIPT)
assert spec is not None and spec.loader is not None
h10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h10)


def _summary(mode: str, **updates):
    row = {
        "mode": mode,
        "source": "torch",
        "reached_h10": True,
        "reached_t": 10.0,
        "accepted_steps": 10,
        "raw_residual_target_violations": 0,
        "sample_sanity_violations": 0,
        "minimum_target_margin": 1e-6,
        "max_reconstruction_polynomial_abs_diff": 0.0,
        "max_reconstruction_remainder_endpoint_diff": 0.0,
        "flowstar_final_width_ratio": 2.0,
        "flowstar_tube_width_ratio": 1.01,
    }
    row.update(updates)
    return row


def test_h10_script_default_horizon_is_exactly_10():
    assert h10.DEFAULT_HORIZON == 10.0
    assert "--horizon" in SCRIPT.read_text(encoding="utf-8")


def test_h10_script_does_not_change_default_right_map_center_mode():
    signature = inspect.signature(flowpipe_step_flowstar_style_adaptive)
    assert signature.parameters["right_map_center_mode"].default == "constant"
    assert h10.CONSTANT_ADAPTIVE == "constant_adaptive_h10"


def test_cross_schedule_replay_flags_modified_prescribed_h():
    rows = h10.make_cross_schedule_rows(
        [{"accepted_rejected": "accepted", "h_accepted": 0.1, "final_segment_width_sum": 1.0, "actual_centered_reset_width_sum": 1.0}],
        [{"accepted_rejected": "accepted", "h_accepted": 0.1, "h_delta": 0.0, "prescribed_h": 0.1, "final_segment_width_sum": 0.9, "actual_centered_reset_width_sum": 0.8}],
        replay_kind="fixture",
        source_mode="source",
        replay_mode="replay",
    )
    assert rows[0]["h_sequence_modified"] is False

    changed = h10.make_cross_schedule_rows(
        [{"accepted_rejected": "accepted", "h_accepted": 0.1}],
        [{"accepted_rejected": "accepted", "h_accepted": 0.09, "h_delta": -0.01, "prescribed_h": 0.1}],
        replay_kind="fixture",
        source_mode="source",
        replay_mode="replay",
    )
    assert changed[0]["h_sequence_modified"] is True


def test_sequence_exhaustion_stops_replay_without_forcing_acceptance():
    summary, rows, _attempts = h10.run_centering_h10(
        mode="fixture_replay",
        run_kind="fixture",
        right_map_center_mode="constant",
        horizon=0.004,
        wall_cap_s=60.0,
        prescribed_h=[0.002],
    )

    assert summary["status"] == "failed"
    assert summary["first_failure_reason"] == "prescribed h sequence ended before horizon"
    assert summary["accepted_steps"] == 1
    assert len(rows) == 1


def test_immediate_same_state_and_cumulative_fields_are_separate():
    assert "immediate_reset_reduction_relative" in h10.SEGMENT_FIELDS
    assert "cumulative_reset_reduction_relative" in h10.SEGMENT_FIELDS
    assert "immediate_same_state_saving" in h10.CHECKPOINT_FIELDS
    assert "cumulative_downstream_saving" in h10.CHECKPOINT_FIELDS


def test_immediate_metric_uses_same_inserted_tm_shadow_diagnostic():
    seg = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=0.002,
        h_min=0.002,
        h_max=0.002,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        reset_mode="normalized_insertion",
        right_map_center_mode="constant",
    )
    assert seg.flowstar_normal_stats is not None
    stats = seg.flowstar_normal_stats
    assert stats["right_map_center_mode"] == "constant"
    assert stats["immediate_saving_source"] == "same_inserted_tm_shadow_diagnostic"
    assert stats["actual_centered_scale_sum"] == stats["constant_scale_sum"]
    assert stats["hypothetical_centered_scale_sum"] <= stats["constant_scale_sum"]
    assert stats["immediate_reset_reduction_relative_sum"] > 0.0


def test_decision_checks_frozen_and_adaptive_sample_violations():
    summaries = {
        h10.CONSTANT_ADAPTIVE: _summary(h10.CONSTANT_ADAPTIVE),
        h10.RANGE_ADAPTIVE: _summary(h10.RANGE_ADAPTIVE),
        h10.RANGE_ON_CONSTANT: _summary(h10.RANGE_ON_CONSTANT, sample_sanity_violations=1),
        h10.CONSTANT_ON_RANGE: _summary(h10.CONSTANT_ON_RANGE),
    }
    decision, _metrics, reasons = h10.decide(
        summaries,
        {"width_worsening_count": 0, "final_common_width_improvement": 0.1},
        [{"replay_kind": "range_midpoint_on_constant_schedule", "replay_status": "accepted", "width_reduction_relative": 0.1}],
    )

    assert decision == "reject_due_to_soundness_or_reconstruction_failure"
    assert any("sample sanity violation" in reason for reason in reasons)


def test_non_positive_margin_cannot_get_success_decision():
    summaries = {
        h10.CONSTANT_ADAPTIVE: _summary(h10.CONSTANT_ADAPTIVE),
        h10.RANGE_ADAPTIVE: _summary(h10.RANGE_ADAPTIVE, minimum_target_margin=0.0),
        h10.RANGE_ON_CONSTANT: _summary(h10.RANGE_ON_CONSTANT),
        h10.CONSTANT_ON_RANGE: _summary(h10.CONSTANT_ON_RANGE),
    }
    decision, _metrics, reasons = h10.decide(
        summaries,
        {"width_worsening_count": 0, "final_common_width_improvement": 0.1},
        [{"replay_kind": "range_midpoint_on_constant_schedule", "replay_status": "accepted", "width_reduction_relative": 0.1}],
    )

    assert decision == "reject_due_to_soundness_or_reconstruction_failure"
    assert any("non-positive" in reason for reason in reasons)


def test_missing_flowstar_component_fields_are_unknown_not_zero():
    rows = h10._flowstar_rows(
        [{"segment_index": 0, "t_lo": 0.0, "t_hi": 0.1, "h": 0.1, "width_sum": 3.0}]
    )
    row = rows[0]
    assert row["missing_flowstar_component_fields"] == h10.UNKNOWN_FLOWSTAR_COMPONENT
    assert row["inserted_range_width_sum"] == h10.UNKNOWN_FLOWSTAR_COMPONENT
    assert row["inserted_range_width_sum"] != 0


def test_csv_physical_line_count_matches_csv_reader_rows(tmp_path):
    path = tmp_path / "rows.csv"
    h10._write_csv(path, ["a", "b"], [{"a": "1", "b": "two, with comma"}, {"a": "3", "b": "4"}])

    with path.open(newline="", encoding="utf-8") as handle:
        parsed = sum(1 for _ in csv.reader(handle))
    assert h10.physical_line_count(path) == parsed
    assert h10.csv_row_count(path) == parsed
