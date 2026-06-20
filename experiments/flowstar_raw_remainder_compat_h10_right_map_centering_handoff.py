#!/usr/bin/env python3
"""Frozen range-midpoint schedule handoff into standard adaptive continuation."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from torch_tm_flowpipe.flowpipe import FLOWSTAR_COMPAT_STEP_GROW
from flowstar_raw_remainder_compat_experiment import _format, _read_rows
from flowstar_raw_remainder_compat_short_horizon import H_MAX, schedule_distance
import flowstar_raw_remainder_compat_h10_right_map_centering as h10

DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h10_right_map_centering_handoff"
DEFAULT_H10_DIR = h10.DEFAULT_OUT_DIR
HANDOFF_REPLAY = "range_midpoint_frozen_constant_schedule_handoff_replay"
HANDOFF_CONTINUATION = "range_midpoint_handoff_adaptive_continuation"
DECISIONS = {
    "centering_blocked_by_adaptive_schedule",
    "centering_helpful_but_still_insufficient",
    "centering_effect_does_not_survive_continuation",
    "reject_due_to_accepted_soundness_failure",
}

SUMMARY_FIELDS = [
    "source", "decision", "horizon", "schedule_source_mode", "source_schedule_steps",
    "frozen_replayed_steps", "replay_rejected_attempts", "replay_h_modified_count",
    "replay_reached_source_schedule_end", "source_schedule_end_time", "replay_reached_t",
    "replay_expected_source_time", "replay_reached_expected_source_time",
    "state_identity_preserved_for_handoff", "handoff_first_h_try",
    "continuation_reached_t", "continuation_reached_h10", "continuation_accepted_steps",
    "continuation_rejected_attempts", "total_accepted_steps", "total_rejected_attempts",
    "pure_range_adaptive_reached_t", "extension_vs_pure_range_adaptive",
    "accepted_raw_target_violations", "accepted_nonfinite_enclosures",
    "accepted_sample_sanity_violations", "max_reconstruction_polynomial_abs_diff",
    "max_reconstruction_remainder_endpoint_diff", "terminal_raw_target_rejection",
    "continuation_minimum_accepted_margin", "continuation_minimum_margin_step",
    "continuation_minimum_margin_time", "continuation_minimum_margin_h",
    "first_handoff_margin_lt_2e_6_step", "first_handoff_margin_lt_2e_6_time",
    "first_handoff_margin_lt_2e_6_h", "first_handoff_margin_lt_2e_6_margin",
    "width_improvement_at_handoff_vs_pure_range",
    "final_common_width_improvement_vs_pure_range",
    "width_advantage_lost_ge_5pct_in_first_20_continuation_steps",
    "flowstar_time_aligned_final_width_ratio", "flowstar_time_aligned_tube_ratio",
    "schedule_distance_vs_flowstar_h10", "notes",
]

SEGMENT_FIELDS = [
    "source", "mode", "phase", "segment_index", "t_lo", "t_hi", "h_attempted",
    "h_accepted", "prescribed_h", "h_delta", "accepted_rejected", "rejection_reason",
    "rejection_count", "target_margin_min", "polynomial_range_width_sum",
    "raw_ctrunc_residual_width_sum", "final_segment_width_sum", "tube_prefix_width_sum",
    "immediate_same_state_saving", "flowstar_time_aligned_segment_width",
    "flowstar_time_aligned_width_ratio", "notes",
]

ATTEMPT_FIELDS = [
    "source", "mode", "phase", "segment_index", "attempt_index", "t_before",
    "h_attempted", "prescribed_h", "validation_status", "accepted_rejected",
    "rejection_reason", "target_margin_min", "raw_residual_target_violation",
    "raw_ctrunc_residual_width_sum", "polynomial_range_width_sum", "subset_result",
    "finite_residual",
]

FORMAT_FIELDS = ["path", "physical_line_count", "csv_reader_row_count", "status"]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in fieldnames})


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def formatting_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("h10_right_map_centering_handoff_*.csv")):
        if path.name == "h10_right_map_centering_handoff_formatting.csv":
            continue
        physical = h10.physical_line_count(path)
        parsed = h10.csv_row_count(path)
        rows.append({
            "path": _display_path(path),
            "physical_line_count": physical,
            "csv_reader_row_count": parsed,
            "status": "ok" if physical == parsed else "mismatch",
        })
    for path in sorted(out_dir.glob("h10_right_map_centering_handoff_*.md")) + sorted(out_dir.glob("h10_right_map_centering_handoff_*.txt")):
        rows.append({
            "path": _display_path(path),
            "physical_line_count": h10.physical_line_count(path),
            "csv_reader_row_count": "",
            "status": "ok",
        })
    return rows


def read_constant_schedule(h10_dir: Path) -> list[float]:
    path = h10_dir / "h10_right_map_centering_segments.csv"
    rows = _read_rows(path)
    schedule: list[float] = []
    for row in rows:
        if row.get("mode") != h10.CONSTANT_ADAPTIVE or row.get("accepted_rejected") != "accepted":
            continue
        value = h10._float(row.get("h_accepted"))
        if value is not None:
            schedule.append(value)
    return schedule


def _summary_row(h10_dir: Path, mode: str) -> Mapping[str, str]:
    for row in _read_rows(h10_dir / "h10_right_map_centering_summary.csv"):
        if row.get("mode") == mode:
            return row
    return {}


def _segments_for_mode(h10_dir: Path, mode: str) -> list[dict[str, str]]:
    return [
        row for row in _read_rows(h10_dir / "h10_right_map_centering_segments.csv")
        if row.get("mode") == mode
    ]


def _accepted(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("accepted_rejected") == "accepted"]


def _min_margin_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    accepted = [
        row for row in rows
        if row.get("accepted_rejected") == "accepted" and h10._float(row.get("target_margin_min")) is not None
    ]
    return min(accepted, key=lambda row: float(row["target_margin_min"])) if accepted else {}


def _first_margin_below(rows: Sequence[Mapping[str, Any]], threshold: float) -> Mapping[str, Any]:
    accepted = [
        row for row in rows
        if row.get("accepted_rejected") == "accepted"
        and h10._float(row.get("target_margin_min")) is not None
        and float(row["target_margin_min"]) < threshold
    ]
    return min(accepted, key=lambda row: float(row["t_hi"])) if accepted else {}


def _max_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [h10._float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else ""


def _flowstar_rows() -> tuple[Mapping[str, Any], list[dict[str, Any]], list[float]]:
    flow_ref, flow_segments, flow_h = h10.load_flowstar_reference(h10.DEFAULT_FLOWSTAR_SEGMENTS.resolve(), h10.DEFAULT_HORIZON)
    flow_rows = h10._flowstar_rows(flow_segments)
    for row in flow_rows:
        idx = int(float(row.get("segment_index") or 0))
        row["tube_prefix_width_sum"] = h10._tube_from_segments(flow_rows[: idx + 1]).get("tube_width_sum", "")
    return flow_ref, flow_rows, flow_h


def _time_aligned_width_ratio(row: Mapping[str, Any], flow_rows: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    t = h10._float(row.get("t_hi"))
    flow = h10._segment_containing(flow_rows, t) if t is not None else {}
    if not flow:
        return "", ""
    return flow.get("final_segment_width_sum", ""), h10._ratio(row.get("final_segment_width_sum"), flow.get("final_segment_width_sum"))


def segment_rows(rows: Sequence[Mapping[str, Any]], *, phase: str, flow_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        flow_width, flow_ratio = _time_aligned_width_ratio(row, flow_rows)
        out.append({
            "source": "torch",
            "mode": row.get("mode", ""),
            "phase": phase,
            "segment_index": row.get("segment_index", ""),
            "t_lo": row.get("t_lo", ""),
            "t_hi": row.get("t_hi", ""),
            "h_attempted": row.get("h_attempted", ""),
            "h_accepted": row.get("h_accepted", ""),
            "prescribed_h": row.get("prescribed_h", ""),
            "h_delta": row.get("h_delta", ""),
            "accepted_rejected": row.get("accepted_rejected", ""),
            "rejection_reason": row.get("rejection_reason", ""),
            "rejection_count": row.get("step_rejections", ""),
            "target_margin_min": row.get("target_margin_min", ""),
            "polynomial_range_width_sum": row.get("polynomial_range_width_sum", ""),
            "raw_ctrunc_residual_width_sum": row.get("raw_ctrunc_residual_width_sum", ""),
            "final_segment_width_sum": row.get("final_segment_width_sum", ""),
            "tube_prefix_width_sum": row.get("tube_prefix_width_sum", ""),
            "immediate_same_state_saving": row.get("immediate_reset_reduction_relative", ""),
            "flowstar_time_aligned_segment_width": flow_width,
            "flowstar_time_aligned_width_ratio": flow_ratio,
            "notes": "frozen replay uses h_min=h_max=prescribed_h; continuation uses standard adaptive policy",
        })
    return out


def attempt_rows(rows: Sequence[Mapping[str, Any]], *, phase: str) -> list[dict[str, Any]]:
    return [{**row, "phase": phase} for row in rows]


def _standard_next_h_after_frozen_replay(replay_rows: Sequence[Mapping[str, Any]]) -> float:
    accepted = _accepted(replay_rows)
    if not accepted:
        return H_MAX
    last_h = h10._float(accepted[-1].get("h_accepted"))
    return min(float(last_h or H_MAX) * FLOWSTAR_COMPAT_STEP_GROW, H_MAX)


def _width_improvement_at_time(candidate_rows: Sequence[Mapping[str, Any]], reference_rows: Sequence[Mapping[str, Any]], t: float) -> float | str:
    candidate = h10._segment_containing(candidate_rows, t)
    reference = h10._segment_containing(reference_rows, t)
    if not candidate or not reference:
        return ""
    _abs, rel = h10._reduction(reference.get("final_segment_width_sum"), candidate.get("final_segment_width_sum"))
    return rel


def _first_20_continuation_advantage_lost(
    continuation_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    initial_advantage: float | str,
) -> bool:
    initial = h10._float(initial_advantage)
    if initial is None:
        return False
    compared: list[float] = []
    for row in _accepted(continuation_rows)[:20]:
        t = h10._float(row.get("t_hi"))
        if t is None:
            continue
        rel = _width_improvement_at_time([*continuation_rows], reference_rows, t)
        rel_f = h10._float(rel)
        if rel_f is not None:
            compared.append(rel_f)
    return bool(compared and initial - min(compared) >= 0.05)


def decide_handoff(summary: Mapping[str, Any]) -> str:
    if (
        int(float(summary.get("accepted_raw_target_violations") or 0)) > 0
        or int(float(summary.get("accepted_nonfinite_enclosures") or 0)) > 0
        or int(float(summary.get("accepted_sample_sanity_violations") or 0)) > 0
        or (h10._float(summary.get("max_reconstruction_polynomial_abs_diff")) or 0.0) > 1e-12
        or (h10._float(summary.get("max_reconstruction_remainder_endpoint_diff")) or 0.0) > 1e-15
    ):
        return "reject_due_to_accepted_soundness_failure"
    margin = h10._float(summary.get("continuation_minimum_accepted_margin"))
    if int(float(summary.get("continuation_accepted_steps") or 0)) > 0 and (margin is None or margin <= 0.0):
        return "reject_due_to_accepted_soundness_failure"
    extension = h10._float(summary.get("extension_vs_pure_range_adaptive")) or 0.0
    reached_h10 = h10._bool(summary.get("continuation_reached_h10"))
    common_improvement = h10._float(summary.get("final_common_width_improvement_vs_pure_range")) or 0.0
    lost_advantage = h10._bool(summary.get("width_advantage_lost_ge_5pct_in_first_20_continuation_steps"))
    if reached_h10 or extension >= 0.5:
        return "centering_blocked_by_adaptive_schedule"
    if extension >= 0.1 and common_improvement >= 0.05:
        return "centering_helpful_but_still_insufficient"
    if extension < 0.1 or lost_advantage:
        return "centering_effect_does_not_survive_continuation"
    return "centering_effect_does_not_survive_continuation"


def _soundness_totals(*summaries: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "accepted_raw_target_violations": sum(int(float(row.get("accepted_raw_target_violations") or 0)) for row in summaries),
        "accepted_nonfinite_enclosures": sum(int(float(row.get("accepted_nonfinite_enclosures") or 0)) for row in summaries),
        "accepted_sample_sanity_violations": sum(int(float(row.get("sample_sanity_violations") or 0)) for row in summaries),
        "max_reconstruction_polynomial_abs_diff": max(h10._float(row.get("max_reconstruction_polynomial_abs_diff")) or 0.0 for row in summaries),
        "max_reconstruction_remainder_endpoint_diff": max(h10._float(row.get("max_reconstruction_remainder_endpoint_diff")) or 0.0 for row in summaries),
        "terminal_raw_target_rejection": any(h10._bool(row.get("terminal_raw_target_rejection")) for row in summaries),
    }


def write_report(path: Path, summary: Mapping[str, Any], formatting: Sequence[Mapping[str, Any]]) -> None:
    recommendation = (
        "Next task should implement an opt-in margin/width-aware growth policy; this task did not implement it."
        if summary.get("decision") == "centering_blocked_by_adaptive_schedule"
        else "Next task should stop tuning centering and inspect the full-step polynomial-range operation ledger around t=6.2..failure."
    )
    lines = [
        "# h10 Range-Midpoint Handoff Continuation Audit",
        "",
        "The frozen replay uses the constant-adaptive accepted h sequence exactly, then continues from the replayed state with the standard Flowstar-compatible adaptive policy.",
        "",
        "## Decision",
        "",
        f"- Decision: `{_format(summary.get('decision'))}`.",
        f"- Replay h modified count: `{_format(summary.get('replay_h_modified_count'))}`.",
        f"- Handoff first h_try: `{_format(summary.get('handoff_first_h_try'))}`.",
        f"- Continuation reached t: `{_format(summary.get('continuation_reached_t'))}`.",
        f"- Extension vs pure range adaptive: `{_format(summary.get('extension_vs_pure_range_adaptive'))}`.",
        f"- Minimum accepted continuation margin: `{_format(summary.get('continuation_minimum_accepted_margin'))}`.",
        f"- Accepted raw-target violations: `{_format(summary.get('accepted_raw_target_violations'))}`.",
        f"- Terminal raw-target rejection: `{_format(summary.get('terminal_raw_target_rejection'))}`.",
        f"- Width improvement at handoff vs pure range: `{_format(summary.get('width_improvement_at_handoff_vs_pure_range'))}`.",
        f"- Final common width improvement vs pure range: `{_format(summary.get('final_common_width_improvement_vs_pure_range'))}`.",
        f"- Recommendation: {recommendation}",
        "",
        "## Formatting",
        "",
        "| path | physical lines | csv.reader rows | status |",
        "| --- | --- | --- | --- |",
    ]
    for row in formatting:
        lines.append(f"| {_format(row.get('path'))} | {_format(row.get('physical_line_count'))} | {_format(row.get('csv_reader_row_count'))} | {_format(row.get('status'))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    horizon = float(args.horizon)
    if abs(horizon - h10.DEFAULT_HORIZON) > 1e-12:
        raise ValueError("handoff audit horizon must be exactly 10")
    h10_dir = args.h10_dir.resolve()
    schedule = read_constant_schedule(h10_dir)
    if not schedule:
        raise FileNotFoundError(f"missing constant adaptive schedule in {h10_dir}")
    flow_ref, flow_rows, flow_h = _flowstar_rows()
    pure_range_summary = _summary_row(h10_dir, h10.RANGE_ADAPTIVE)
    pure_range_rows = _segments_for_mode(h10_dir, h10.RANGE_ADAPTIVE)
    replay_expected_source_time = sum(schedule)

    replay_summary, replay_rows, replay_attempts = h10.run_centering_h10(
        mode=HANDOFF_REPLAY,
        run_kind="range_midpoint frozen replay for handoff",
        right_map_center_mode="range_midpoint",
        horizon=horizon,
        wall_cap_s=float(args.wall_cap_s),
        prescribed_h=schedule,
    )
    replay_h_modified = sum(
        1 for row in _accepted(replay_rows)
        if (h10._float(row.get("h_delta")) is not None and abs(float(row["h_delta"])) > 1e-10)
    )
    handoff_current = replay_summary.get("_final_current")
    handoff_normal_state = replay_summary.get("_final_normal_state")
    handoff_samples = replay_summary.get("_final_samples")
    handoff_t = h10._float(replay_summary.get("reached_t")) or 0.0
    if handoff_current is None or not h10._bool(replay_summary.get("reached_source_schedule_end")):
        raise RuntimeError("frozen replay did not complete the source schedule; no handoff state was used")
    first_h_try = min(_standard_next_h_after_frozen_replay(replay_rows), H_MAX, horizon - handoff_t)
    state_identity_preserved = handoff_current is replay_summary.get("_final_current")

    continuation_summary, continuation_rows, continuation_attempts = h10.run_centering_h10(
        mode=HANDOFF_CONTINUATION,
        run_kind="range_midpoint adaptive continuation after frozen schedule",
        right_map_center_mode="range_midpoint",
        horizon=horizon,
        wall_cap_s=float(args.wall_cap_s),
        initial_current=handoff_current,
        initial_normal_state=handoff_normal_state,
        initial_samples=handoff_samples,
        initial_t=handoff_t,
        initial_h_next=first_h_try,
        start_segment_index=len(_accepted(replay_rows)),
    )

    all_rows = [*replay_rows, *continuation_rows]
    continuation_accepted = _accepted(continuation_rows)
    min_margin = _min_margin_row(continuation_rows)
    first_low_margin = _first_margin_below(continuation_rows, 2e-6)
    handoff_advantage = _width_improvement_at_time(replay_rows, pure_range_rows, handoff_t)
    continuation_end = h10._float(continuation_summary.get("reached_t")) or handoff_t
    pure_range_t = h10._float(pure_range_summary.get("reached_t")) or 0.0
    common_t = min(continuation_end, pure_range_t)
    final_common_improvement = _width_improvement_at_time(all_rows, pure_range_rows, common_t)
    advantage_lost = _first_20_continuation_advantage_lost(continuation_rows, pure_range_rows, handoff_advantage)
    last_row = _accepted(all_rows)[-1] if _accepted(all_rows) else {}
    flow_segment = h10._segment_containing(flow_rows, h10._float(last_row.get("t_hi")) or 0.0)
    soundness = _soundness_totals(replay_summary, continuation_summary)
    terminal_target_rejection = bool(
        soundness["terminal_raw_target_rejection"]
        or h10._bool(continuation_summary.get("terminal_raw_target_rejection"))
    )

    summary: dict[str, Any] = {
        "source": "torch",
        "horizon": horizon,
        "schedule_source_mode": h10.CONSTANT_ADAPTIVE,
        "source_schedule_steps": len(schedule),
        "frozen_replayed_steps": len(_accepted(replay_rows)),
        "replay_rejected_attempts": replay_summary.get("rejected_attempts", ""),
        "replay_h_modified_count": replay_h_modified,
        "replay_reached_source_schedule_end": replay_summary.get("reached_source_schedule_end", ""),
        "source_schedule_end_time": replay_summary.get("source_schedule_end_time", ""),
        "replay_reached_t": replay_summary.get("reached_t", ""),
        "replay_expected_source_time": replay_expected_source_time,
        "replay_reached_expected_source_time": abs((h10._float(replay_summary.get("reached_t")) or 0.0) - replay_expected_source_time) <= 1e-10,
        "state_identity_preserved_for_handoff": state_identity_preserved,
        "handoff_first_h_try": first_h_try,
        "continuation_reached_t": continuation_summary.get("reached_t", ""),
        "continuation_reached_h10": continuation_summary.get("reached_h10", ""),
        "continuation_accepted_steps": len(continuation_accepted),
        "continuation_rejected_attempts": continuation_summary.get("rejected_attempts", ""),
        "total_accepted_steps": len(_accepted(all_rows)),
        "total_rejected_attempts": int(float(replay_summary.get("rejected_attempts") or 0)) + int(float(continuation_summary.get("rejected_attempts") or 0)),
        "pure_range_adaptive_reached_t": pure_range_t,
        "extension_vs_pure_range_adaptive": continuation_end - pure_range_t,
        **soundness,
        "terminal_raw_target_rejection": terminal_target_rejection,
        "continuation_minimum_accepted_margin": min_margin.get("target_margin_min", ""),
        "continuation_minimum_margin_step": min_margin.get("segment_index", ""),
        "continuation_minimum_margin_time": min_margin.get("t_hi", ""),
        "continuation_minimum_margin_h": min_margin.get("h_accepted", ""),
        "first_handoff_margin_lt_2e_6_step": first_low_margin.get("segment_index", ""),
        "first_handoff_margin_lt_2e_6_time": first_low_margin.get("t_hi", ""),
        "first_handoff_margin_lt_2e_6_h": first_low_margin.get("h_accepted", ""),
        "first_handoff_margin_lt_2e_6_margin": first_low_margin.get("target_margin_min", ""),
        "width_improvement_at_handoff_vs_pure_range": handoff_advantage,
        "final_common_width_improvement_vs_pure_range": final_common_improvement,
        "width_advantage_lost_ge_5pct_in_first_20_continuation_steps": advantage_lost,
        "flowstar_time_aligned_final_width_ratio": h10._ratio(last_row.get("final_segment_width_sum"), flow_segment.get("final_segment_width_sum")) if flow_segment else "",
        "flowstar_time_aligned_tube_ratio": h10._ratio(h10._tube_from_segments(_accepted(all_rows)).get("tube_width_sum", ""), flow_segment.get("tube_prefix_width_sum")) if flow_segment else "",
        "schedule_distance_vs_flowstar_h10": schedule_distance(list(flow_h), [float(row["h_accepted"]) for row in _accepted(all_rows) if h10._float(row.get("h_accepted")) is not None]),
        "notes": "terminal validation rejection is safe failure-to-progress, not an accepted unsound enclosure",
    }
    summary["decision"] = decide_handoff(summary)

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    segment_out = [
        *segment_rows(replay_rows, phase="frozen_replay", flow_rows=flow_rows),
        *segment_rows(continuation_rows, phase="adaptive_continuation", flow_rows=flow_rows),
    ]
    attempt_out = [
        *attempt_rows(replay_attempts, phase="frozen_replay"),
        *attempt_rows(continuation_attempts, phase="adaptive_continuation"),
    ]
    _write_csv(out / "h10_right_map_centering_handoff_summary.csv", SUMMARY_FIELDS, [summary])
    _write_csv(out / "h10_right_map_centering_handoff_segments.csv", SEGMENT_FIELDS, segment_out)
    _write_csv(out / "h10_right_map_centering_handoff_attempts.csv", ATTEMPT_FIELDS, attempt_out)
    (out / "h10_right_map_centering_handoff_decision.txt").write_text(summary["decision"] + "\n", encoding="utf-8")
    fmt = formatting_rows(out)
    _write_csv(out / "h10_right_map_centering_handoff_formatting.csv", FORMAT_FIELDS, fmt)
    write_report(out / "h10_right_map_centering_handoff_report.md", summary, fmt)
    fmt = formatting_rows(out)
    _write_csv(out / "h10_right_map_centering_handoff_formatting.csv", FORMAT_FIELDS, fmt)
    write_report(out / "h10_right_map_centering_handoff_report.md", summary, fmt)
    return summary, segment_out, attempt_out, summary["decision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, default=h10.DEFAULT_HORIZON)
    parser.add_argument("--h10-dir", type=Path, default=DEFAULT_H10_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--wall-cap-s", type=float, default=7200.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, segments, attempts, decision = run(args)
    print(f"wrote {args.out_dir.resolve() / 'h10_right_map_centering_handoff_summary.csv'} (1 rows)")
    print(f"wrote {args.out_dir.resolve() / 'h10_right_map_centering_handoff_segments.csv'} ({len(segments)} rows)")
    print(f"wrote {args.out_dir.resolve() / 'h10_right_map_centering_handoff_attempts.csv'} ({len(attempts)} rows)")
    print(f"decision {decision}")
    print(f"continuation reached t {summary.get('continuation_reached_t')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
