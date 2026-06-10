from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H5_SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_h5.py"
ATTEMPT_SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_attempt_sequence.py"

h5_spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_h5", H5_SCRIPT)
assert h5_spec is not None and h5_spec.loader is not None
h5 = importlib.util.module_from_spec(h5_spec)
h5_spec.loader.exec_module(h5)

attempt_spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_attempt_sequence", ATTEMPT_SCRIPT)
assert attempt_spec is not None and attempt_spec.loader is not None
attempt = importlib.util.module_from_spec(attempt_spec)
attempt_spec.loader.exec_module(attempt)

EXPECTED_OUTPUTS = [
    "h5_summary.csv",
    "h5_segments.csv",
    "h5_width_vs_flowstar.csv",
    "h5_schedule_compare.csv",
    "h5_sample_containment.csv",
    "h5_report.md",
]

CHECKED_IN_OUTPUTS = [ROOT / "outputs" / "flowstar_raw_remainder_compat_h5" / name for name in EXPECTED_OUTPUTS]


def _summary_row(mode: str, source: str = "torch", completed: bool = True) -> dict[str, object]:
    return {
        "source": source,
        "mode": mode,
        "status": "completed" if completed else "failed",
        "reached_t": 5.0 if completed else 2.0,
        "completed_h5": completed,
        "accepted_steps": 2,
        "rejected_attempts": 1,
        "min_h_used": 0.0125,
        "h_below_flowstar_min_count": 0,
        "runtime_s": 0.1,
        "final_width_x": 0.2,
        "final_width_y": 0.3,
        "final_width_sum": 0.5,
        "flowstar_reference_final_width_sum": 0.4,
        "last_segment_width_ratio_vs_flowstar": 1.25 if source == "torch" else 1.0,
        "tube_width_ratio_vs_flowstar": 1.1 if source == "torch" else 1.0,
        "schedule_distance_vs_flowstar": 0.0,
        "schedule_prefix_match_count": 2,
        "sample_containment_status": "passed" if source == "torch" else "not_applicable",
        "sample_violations": 0,
        "default_behavior_changed": False,
        "recommendation": "h10_candidate_after_review" if mode == "raw_remainder_compat_flowstar_step_policy" else "reference_only",
        "notes": "fixture",
    }


def _segment_row(mode: str, source: str = "torch") -> dict[str, object]:
    return {
        "source": source,
        "mode": mode,
        "segment_index": 0,
        "status": "validated",
        "t_lo": 0.0,
        "t_hi": 0.0125,
        "h": 0.0125,
        "x_lo": 1.0,
        "x_hi": 1.2,
        "y_lo": 2.0,
        "y_hi": 2.3,
        "width_x": 0.2,
        "width_y": 0.3,
        "width_sum": 0.5,
        "box_semantics": "torch_segment_tm_range" if source == "torch" else "flowstar_gnuplot_segment_box",
        "step_rejections": 0,
        "next_h": 0.01375,
        "message": "fixture",
    }


def test_h5_output_writer_fixture_writes_all_expected_files(tmp_path):
    summary_rows = [
        _summary_row("generated_flowstar_h5_reference", "flowstar"),
        _summary_row(h5.BEST_NORMALIZED_H5_MODE, "torch_existing_artifact"),
        _summary_row("current_no_queue_default_policy"),
        _summary_row("raw_remainder_compat_default_policy"),
        _summary_row("raw_remainder_compat_flowstar_step_policy"),
    ]
    segment_rows = [_segment_row("generated_flowstar_h5_reference", "flowstar"), _segment_row("raw_remainder_compat_flowstar_step_policy")]
    width_rows = [
        {
            "source": "torch",
            "mode": "raw_remainder_compat_flowstar_step_policy",
            "comparison_enabled": True,
            "comparison_semantics": "torch segment TM boxes vs Flowstar GNUPLOT segment boxes",
            "flowstar_reference_final_width_sum": 0.4,
            "last_segment_width_sum": 0.5,
            "last_segment_width_ratio_vs_flowstar": 1.25,
            "flowstar_reference_tube_width_sum": 1.0,
            "tube_width_sum": 1.1,
            "tube_width_ratio_vs_flowstar": 1.1,
            "disabled_reason": "",
        }
    ]
    schedule_rows = [
        {
            "source": "torch",
            "mode": "raw_remainder_compat_flowstar_step_policy",
            "schedule_distance_vs_flowstar": 0.0,
            "schedule_prefix_match_count": 2,
            "schedule_prefix_matches_flowstar": True,
            "accepted_h_sequence": "0.0125;0.01375",
            "flowstar_h_sequence": "0.0125;0.01375",
        }
    ]
    sample_rows = [
        {
            "source": "torch",
            "mode": "raw_remainder_compat_flowstar_step_policy",
            "sample_count": 21,
            "sample_containment_status": "passed",
            "sample_violations": 0,
            "max_violation": 0,
            "checked_segments": 2,
            "notes": "fixture",
        }
    ]

    h5.write_outputs(tmp_path, summary_rows, segment_rows, width_rows, schedule_rows, sample_rows, 5.0)

    for name in EXPECTED_OUTPUTS:
        assert (tmp_path / name).exists()
    with (tmp_path / "h5_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["mode"] == "raw_remainder_compat_flowstar_step_policy"


def test_h5_report_refuses_parity_claim_and_includes_no_h10_guard(tmp_path):
    summary_rows = [
        _summary_row("generated_flowstar_h5_reference", "flowstar"),
        _summary_row("raw_remainder_compat_default_policy"),
        _summary_row("raw_remainder_compat_flowstar_step_policy"),
    ]
    width_rows = [
        {
            "mode": "raw_remainder_compat_flowstar_step_policy",
            "comparison_enabled": True,
            "comparison_semantics": "torch segment TM boxes vs Flowstar GNUPLOT segment boxes",
            "disabled_reason": "",
        }
    ]
    report = tmp_path / "h5_report.md"
    h5.write_report(report, summary_rows, width_rows, 5.0)
    text = report.read_text(encoding="utf-8")

    assert "does not run h10" in text
    assert "does not claim Flowstar parity" in text
    assert "Flowstar parity achieved" not in text
    assert "## Width Semantics" in text


def test_width_comparison_disables_endpoint_vs_segment_ratios():
    assert h5.segment_ratio_vs_flowstar(0.5, 0.4, "endpoint_box", "flowstar_gnuplot_segment_box") == ""
    assert h5.tube_ratio_vs_flowstar(1.0, 0.5, "endpoint_tube", "flowstar_gnuplot_segment_tube") == ""
    assert h5.segment_ratio_vs_flowstar(0.5, 0.4, "segment_box", "flowstar_gnuplot_segment_box") == 1.25


def test_default_mode_unchanged_and_compat_stays_opt_in():
    current_diag, current_seg = attempt.run_torch_attempt("current_no_queue", 0.025)
    compat_diag, compat_seg = attempt.run_torch_attempt("flowstar_raw_remainder_compat", 0.025)

    assert current_seg.status == "validated"
    assert compat_seg.status == "failed"
    assert current_diag["flowstar_raw_remainder_compat_enabled"] is False
    assert compat_diag["flowstar_raw_remainder_compat_enabled"] is True


def test_h5_script_has_no_h10_default_or_output_path():
    assert h5.DEFAULT_OUT_DIR.name == "flowstar_raw_remainder_compat_h5"
    assert "h10" not in str(h5.DEFAULT_OUT_DIR)
    assert h5.H_MAX == 0.1
    assert h5.H_MIN == 0.002


def test_checked_in_h5_outputs_have_physical_lines():
    for path in CHECKED_IN_OUTPUTS:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".csv":
            assert len(text.splitlines()) > 1, path
        else:
            assert len(text.splitlines()) > 10, path
            assert "\\n" not in text
