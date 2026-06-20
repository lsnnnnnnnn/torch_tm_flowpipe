from __future__ import annotations

import csv
import importlib.util
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flowstar_raw_remainder_compat_h10_right_map_centering_handoff.py"
spec = importlib.util.spec_from_file_location("flowstar_raw_remainder_compat_h10_right_map_centering_handoff", SCRIPT)
assert spec is not None and spec.loader is not None
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def test_read_constant_schedule_uses_only_accepted_constant_rows(tmp_path):
    h10_dir = tmp_path
    path = h10_dir / "h10_right_map_centering_segments.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "accepted_rejected", "h_accepted"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({"mode": handoff.h10.CONSTANT_ADAPTIVE, "accepted_rejected": "accepted", "h_accepted": "0.1"})
        writer.writerow({"mode": handoff.h10.CONSTANT_ADAPTIVE, "accepted_rejected": "rejected", "h_accepted": ""})
        writer.writerow({"mode": handoff.h10.RANGE_ADAPTIVE, "accepted_rejected": "accepted", "h_accepted": "0.2"})

    assert handoff.read_constant_schedule(h10_dir) == [0.1]


def test_standard_next_h_after_frozen_replay_uses_flowstar_growth_rule():
    rows = [{"accepted_rejected": "accepted", "h_accepted": 0.2}]

    assert handoff._standard_next_h_after_frozen_replay(rows) == min(0.2 * handoff.FLOWSTAR_COMPAT_STEP_GROW, handoff.H_MAX)


def test_handoff_run_replays_exact_h_and_passes_same_state_to_continuation(monkeypatch, tmp_path):
    state = object()
    normal_state = object()
    samples = [[1.0, 2.0]]
    calls = []

    monkeypatch.setattr(handoff, "read_constant_schedule", lambda _h10_dir: [0.1, 0.2])
    monkeypatch.setattr(
        handoff,
        "_flowstar_rows",
        lambda: (
            {"accepted_steps": 1, "final_width_sum": 1.0, "_tube_width_sum": 1.0},
            [{"accepted_rejected": "accepted", "t_lo": 0.0, "t_hi": 10.0, "final_segment_width_sum": 1.0, "tube_prefix_width_sum": 1.0}],
            [0.1, 0.2],
        ),
    )
    monkeypatch.setattr(handoff, "_summary_row", lambda _h10_dir, _mode: {"reached_t": 0.31})
    monkeypatch.setattr(
        handoff,
        "_segments_for_mode",
        lambda _h10_dir, _mode: [
            {"accepted_rejected": "accepted", "t_lo": 0.0, "t_hi": 0.31, "final_segment_width_sum": 2.0}
        ],
    )

    def fake_run_centering_h10(**kwargs):
        calls.append(kwargs)
        if kwargs.get("prescribed_h") is not None:
            return (
                {
                    "reached_t": 0.3,
                    "reached_source_schedule_end": True,
                    "source_schedule_end_time": 0.3,
                    "rejected_attempts": 0,
                    "accepted_raw_target_violations": 0,
                    "accepted_nonfinite_enclosures": 0,
                    "sample_sanity_violations": 0,
                    "max_reconstruction_polynomial_abs_diff": 0.0,
                    "max_reconstruction_remainder_endpoint_diff": 0.0,
                    "terminal_raw_target_rejection": False,
                    "_final_current": state,
                    "_final_normal_state": normal_state,
                    "_final_samples": samples,
                },
                [
                    {"accepted_rejected": "accepted", "segment_index": 0, "t_lo": 0.0, "t_hi": 0.1, "h_accepted": 0.1, "h_delta": 0.0, "final_segment_width_sum": 1.8},
                    {"accepted_rejected": "accepted", "segment_index": 1, "t_lo": 0.1, "t_hi": 0.3, "h_accepted": 0.2, "h_delta": 0.0, "final_segment_width_sum": 1.6},
                ],
                [],
            )
        assert kwargs["initial_current"] is state
        assert kwargs["initial_normal_state"] is normal_state
        assert kwargs["initial_samples"] is samples
        assert kwargs["initial_t"] == 0.3
        assert kwargs["initial_h_next"] == min(0.2 * handoff.FLOWSTAR_COMPAT_STEP_GROW, handoff.H_MAX)
        return (
            {
                "reached_t": 0.35,
                "reached_h10": False,
                "rejected_attempts": 1,
                "accepted_raw_target_violations": 0,
                "accepted_nonfinite_enclosures": 0,
                "sample_sanity_violations": 0,
                "max_reconstruction_polynomial_abs_diff": 0.0,
                "max_reconstruction_remainder_endpoint_diff": 0.0,
                "terminal_raw_target_rejection": True,
            },
            [
                {"accepted_rejected": "accepted", "segment_index": 2, "t_lo": 0.3, "t_hi": 0.35, "h_accepted": 0.05, "target_margin_min": 1e-6, "final_segment_width_sum": 1.5},
                {"accepted_rejected": "rejected", "segment_index": 3, "t_lo": 0.35, "t_hi": 0.36, "rejection_reason": "target subset failure"},
            ],
            [{"accepted_rejected": "rejected", "segment_index": 3, "raw_residual_target_violation": True}],
        )

    monkeypatch.setattr(handoff.h10, "run_centering_h10", fake_run_centering_h10)

    summary, _segments, _attempts, decision = handoff.run(
        Namespace(horizon=10.0, h10_dir=tmp_path, out_dir=tmp_path / "out", wall_cap_s=60.0)
    )

    assert len(calls) == 2
    assert calls[0]["prescribed_h"] == [0.1, 0.2]
    assert summary["replay_h_modified_count"] == 0
    assert summary["state_identity_preserved_for_handoff"] is True
    assert decision != "reject_due_to_accepted_soundness_failure"


def test_handoff_decision_rejects_only_accepted_soundness_failure():
    safe_terminal = {
        "continuation_accepted_steps": 0,
        "accepted_raw_target_violations": 0,
        "accepted_nonfinite_enclosures": 0,
        "accepted_sample_sanity_violations": 0,
        "max_reconstruction_polynomial_abs_diff": 0.0,
        "max_reconstruction_remainder_endpoint_diff": 0.0,
        "terminal_raw_target_rejection": True,
        "extension_vs_pure_range_adaptive": 0.0,
        "continuation_reached_h10": False,
    }
    accepted_bad = {**safe_terminal, "continuation_accepted_steps": 1, "accepted_raw_target_violations": 1, "continuation_minimum_accepted_margin": 1e-6}

    assert handoff.decide_handoff(safe_terminal) == "centering_effect_does_not_survive_continuation"
    assert handoff.decide_handoff(accepted_bad) == "reject_due_to_accepted_soundness_failure"


def test_handoff_csv_physical_line_count_matches_csv_reader_rows(tmp_path):
    path = tmp_path / "rows.csv"
    handoff._write_csv(path, ["a", "b"], [{"a": "1", "b": "two, with comma"}])

    with path.open(newline="", encoding="utf-8") as handle:
        parsed = sum(1 for _ in csv.reader(handle))
    assert handoff.h10.physical_line_count(path) == parsed
    assert handoff.h10.csv_row_count(path) == parsed
