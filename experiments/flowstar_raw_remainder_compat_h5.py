#!/usr/bin/env python3
"""Horizon-5 audit for opt-in Flowstar raw-remainder compatibility."""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from torch_tm_flowpipe import FlowstarNormalFlowpipeState, Interval, TMVector, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.safety import intervals_are_finite
from flowstar_raw_remainder_compat_experiment import (  # noqa: E402
    ORDER,
    TARGET_RADIUS,
    _float,
    _format,
    _read_rows,
    van_der_pol_flowstar_expression_ode,
)
from flowstar_raw_remainder_compat_short_horizon import (  # noqa: E402
    H_MAX,
    H_MIN,
    _advance_sample,
    _interval_violation,
    schedule_distance,
)

DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5"
DEFAULT_FLOWSTAR_SEGMENTS = ROOT / "outputs" / "flowstar_benchmark_parity" / "generated_flowstar" / "generated_flowstar_segments.csv"
DEFAULT_FLOWSTAR_SUMMARY = ROOT / "outputs" / "flowstar_benchmark_parity" / "generated_flowstar" / "generated_flowstar_summary.csv"
DEFAULT_NORMALIZED_H5_SUMMARY = ROOT / "outputs" / "flowstar_normalized_insertion_rescue" / "normalized_insertion_summary.csv"
DEFAULT_NORMALIZED_H5_SEGMENTS = ROOT / "outputs" / "flowstar_normalized_insertion_rescue" / "rescue_segments.csv"
BEST_NORMALIZED_H5_MODE = "flowstar_style_o6_candidate8_output6_cutoff_insert"
SAMPLE_RANDOM_COUNT = 16

SUMMARY_FIELDS = [
    "source",
    "mode",
    "status",
    "reached_t",
    "completed_h5",
    "accepted_steps",
    "rejected_attempts",
    "min_h_used",
    "h_below_flowstar_min_count",
    "runtime_s",
    "final_width_x",
    "final_width_y",
    "final_width_sum",
    "flowstar_reference_final_width_sum",
    "last_segment_width_ratio_vs_flowstar",
    "tube_width_ratio_vs_flowstar",
    "schedule_distance_vs_flowstar",
    "schedule_prefix_match_count",
    "sample_containment_status",
    "sample_violations",
    "default_behavior_changed",
    "recommendation",
    "notes",
]

SEGMENT_FIELDS = [
    "source",
    "mode",
    "segment_index",
    "status",
    "t_lo",
    "t_hi",
    "h",
    "x_lo",
    "x_hi",
    "y_lo",
    "y_hi",
    "width_x",
    "width_y",
    "width_sum",
    "box_semantics",
    "step_rejections",
    "next_h",
    "message",
]

WIDTH_FIELDS = [
    "source",
    "mode",
    "comparison_enabled",
    "comparison_semantics",
    "flowstar_reference_final_width_sum",
    "last_segment_width_sum",
    "last_segment_width_ratio_vs_flowstar",
    "flowstar_reference_tube_width_sum",
    "tube_width_sum",
    "tube_width_ratio_vs_flowstar",
    "disabled_reason",
]

SCHEDULE_FIELDS = [
    "source",
    "mode",
    "schedule_distance_vs_flowstar",
    "schedule_prefix_match_count",
    "schedule_prefix_matches_flowstar",
    "accepted_h_sequence",
    "flowstar_h_sequence",
]

SAMPLE_FIELDS = [
    "source",
    "mode",
    "sample_count",
    "sample_containment_status",
    "sample_violations",
    "max_violation",
    "checked_segments",
    "notes",
]

MODE_SPECS: dict[str, dict[str, Any]] = {
    "current_no_queue_default_policy": {
        "validation_mode": "target_remainder_flowstar_ctrunc",
        "step_policy_mode": "",
        "grow_factor": 1.5,
    },
    "raw_remainder_compat_default_policy": {
        "validation_mode": "flowstar_raw_remainder_compat",
        "step_policy_mode": "",
        "grow_factor": 1.5,
    },
    "raw_remainder_compat_flowstar_step_policy": {
        "validation_mode": "flowstar_raw_remainder_compat",
        "step_policy_mode": "flowstar_compat",
        "grow_factor": 1.5,
    },
}


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in fieldnames})


def _interval_bounds(box: Sequence[Interval]) -> tuple[float, float, float, float, float, float, float]:
    x, y = box[:2]
    x_lo = float(x.lo.detach().cpu())
    x_hi = float(x.hi.detach().cpu())
    y_lo = float(y.lo.detach().cpu())
    y_hi = float(y.hi.detach().cpu())
    width_x = x_hi - x_lo
    width_y = y_hi - y_lo
    return x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_x + width_y


def _num(value: Any) -> float | None:
    return _float(value)


def _ratio(num: Any, den: Any) -> float | str:
    n = _num(num)
    d = _num(den)
    if n is None or d is None or d <= 0.0:
        return ""
    return n / d


def segment_ratio_vs_flowstar(candidate_width: Any, flowstar_width: Any, candidate_semantics: str, flowstar_semantics: str) -> float | str:
    if candidate_semantics != "segment_box" or flowstar_semantics != "flowstar_gnuplot_segment_box":
        return ""
    return _ratio(candidate_width, flowstar_width)


def tube_ratio_vs_flowstar(candidate_width: Any, flowstar_width: Any, candidate_semantics: str, flowstar_semantics: str) -> float | str:
    if candidate_semantics != "segment_tube" or flowstar_semantics != "flowstar_gnuplot_segment_tube":
        return ""
    return _ratio(candidate_width, flowstar_width)


def _prefix_match_count(reference: Sequence[float], candidate: Sequence[float], tol: float = 1e-12) -> int:
    count = 0
    for expected, actual in zip(reference, candidate):
        if abs(expected - actual) > tol:
            break
        count += 1
    return count


def _tube_from_segments(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finite_rows = [row for row in rows if all(_num(row.get(field)) is not None for field in ("x_lo", "x_hi", "y_lo", "y_hi"))]
    if not finite_rows:
        return {"tube_width_x": "", "tube_width_y": "", "tube_width_sum": ""}
    x_lo = min(float(row["x_lo"]) for row in finite_rows)
    x_hi = max(float(row["x_hi"]) for row in finite_rows)
    y_lo = min(float(row["y_lo"]) for row in finite_rows)
    y_hi = max(float(row["y_hi"]) for row in finite_rows)
    return {"tube_width_x": x_hi - x_lo, "tube_width_y": y_hi - y_lo, "tube_width_sum": (x_hi - x_lo) + (y_hi - y_lo)}


def load_flowstar_reference(path: Path, horizon: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    rows = _read_rows(path)
    selected: list[dict[str, Any]] = []
    h_values: list[float] = []
    for row in rows:
        t_lo = _num(row.get("t_lo"))
        t_hi = _num(row.get("t_hi"))
        if t_lo is None or t_hi is None:
            continue
        if t_lo >= horizon + 1e-12:
            continue
        clipped_t_hi = min(t_hi, horizon)
        clipped_h = clipped_t_hi - t_lo
        if clipped_h <= 0.0:
            continue
        h_values.append(clipped_h)
        selected.append(
            {
                "source": "flowstar",
                "mode": "generated_flowstar_h5_reference",
                "segment_index": row.get("segment_index", len(selected)),
                "status": "validated",
                "t_lo": t_lo,
                "t_hi": t_hi,
                "h": t_hi - t_lo,
                "x_lo": row.get("x_lo", ""),
                "x_hi": row.get("x_hi", ""),
                "y_lo": row.get("y_lo", ""),
                "y_hi": row.get("y_hi", ""),
                "width_x": row.get("width_x", ""),
                "width_y": row.get("width_y", ""),
                "width_sum": row.get("width_sum", ""),
                "box_semantics": "flowstar_gnuplot_segment_box",
                "step_rejections": "",
                "next_h": "",
                "message": "existing generated Flowstar GNUPLOT segment box; not an endpoint box; overlapping final segment is included when restricting the h10 artifact to h5",
            }
        )
    if not selected:
        return {}, [], []
    last = selected[-1]
    tube = _tube_from_segments(selected)
    summary = {
        "source": "flowstar",
        "mode": "generated_flowstar_h5_reference",
        "status": "completed",
        "reached_t": min(horizon, _num(last.get("t_hi")) or 0.0),
        "completed_h5": bool((_num(last.get("t_hi")) or 0.0) >= horizon - 1e-9),
        "accepted_steps": len(selected),
        "rejected_attempts": "",
        "min_h_used": min(h_values) if h_values else "",
        "h_below_flowstar_min_count": sum(1 for h in h_values if h < H_MIN - 1e-12),
        "runtime_s": "",
        "final_width_x": last.get("width_x", ""),
        "final_width_y": last.get("width_y", ""),
        "final_width_sum": last.get("width_sum", ""),
        "flowstar_reference_final_width_sum": last.get("width_sum", ""),
        "last_segment_width_ratio_vs_flowstar": 1.0,
        "tube_width_ratio_vs_flowstar": 1.0,
        "schedule_distance_vs_flowstar": 0.0,
        "schedule_prefix_match_count": len(h_values),
        "sample_containment_status": "not_applicable",
        "sample_violations": "",
        "default_behavior_changed": False,
        "recommendation": "reference_only",
        "notes": "Existing generated Flowstar h10 segment artifact restricted to T=5; the segment overlapping T=5 is included, and GNUPLOT boxes are segment boxes, not endpoints.",
        "_accepted_h": h_values,
        "_last_segment_width_sum": last.get("width_sum", ""),
        "_tube_width_sum": tube.get("tube_width_sum", ""),
        "_last_segment_width_x": last.get("width_x", ""),
        "_last_segment_width_y": last.get("width_y", ""),
    }
    return summary, selected, h_values


def load_existing_normalized_h5(summary_path: Path, segments_path: Path, horizon: float, flowstar_h: Sequence[float], flow_ref: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[float]]:
    if not summary_path.exists() or not segments_path.exists():
        return None, [], []
    summary_rows = [row for row in _read_rows(summary_path) if row.get("run_id") == BEST_NORMALIZED_H5_MODE]
    if not summary_rows:
        return None, [], []
    source_row = summary_rows[0]
    segment_rows = [row for row in _read_rows(segments_path) if row.get("run_id") == BEST_NORMALIZED_H5_MODE and row.get("status") == "validated"]
    out_segments: list[dict[str, Any]] = []
    h_values: list[float] = []
    for row in segment_rows:
        t_lo = _num(row.get("t_lo"))
        t_hi = _num(row.get("t_hi"))
        h = _num(row.get("h"))
        if t_lo is None or t_hi is None or h is None or t_lo >= horizon - 1e-12:
            continue
        h_values.append(h)
        out_segments.append(
            {
                "source": "torch_existing_artifact",
                "mode": BEST_NORMALIZED_H5_MODE,
                "segment_index": row.get("segment_index", len(out_segments)),
                "status": row.get("status", "validated"),
                "t_lo": t_lo,
                "t_hi": t_hi,
                "h": h,
                "x_lo": row.get("x_lo", ""),
                "x_hi": row.get("x_hi", ""),
                "y_lo": row.get("y_lo", ""),
                "y_hi": row.get("y_hi", ""),
                "width_x": row.get("width_x", ""),
                "width_y": row.get("width_y", ""),
                "width_sum": row.get("width_sum", ""),
                "box_semantics": "legacy_endpoint_only_width_artifact",
                "step_rejections": row.get("step_rejections", ""),
                "next_h": row.get("next_h", ""),
                "message": "existing normalized-insertion h5 artifact; Flowstar ratios disabled because width semantics are endpoint-only/legacy",
            }
        )
    dist = schedule_distance(list(flowstar_h), h_values) if flowstar_h and h_values else ""
    prefix = _prefix_match_count(list(flowstar_h), h_values) if flowstar_h and h_values else ""
    reached_t = _num(source_row.get("last_validated_t")) or 0.0
    status = source_row.get("status", "")
    summary = {
        "source": "torch_existing_artifact",
        "mode": BEST_NORMALIZED_H5_MODE,
        "status": status,
        "reached_t": reached_t,
        "completed_h5": reached_t >= horizon - 1e-9,
        "accepted_steps": source_row.get("num_accepted_steps") or source_row.get("validated_segments", ""),
        "rejected_attempts": source_row.get("num_step_rejections", ""),
        "min_h_used": source_row.get("min_h_used", ""),
        "h_below_flowstar_min_count": source_row.get("h_below_flowstar_min_count", ""),
        "runtime_s": source_row.get("runtime_s", ""),
        "final_width_x": "",
        "final_width_y": "",
        "final_width_sum": source_row.get("final_width_sum", ""),
        "flowstar_reference_final_width_sum": flow_ref.get("flowstar_reference_final_width_sum", ""),
        "last_segment_width_ratio_vs_flowstar": "",
        "tube_width_ratio_vs_flowstar": "",
        "schedule_distance_vs_flowstar": dist,
        "schedule_prefix_match_count": prefix,
        "sample_containment_status": "not_run_existing_artifact",
        "sample_violations": "",
        "default_behavior_changed": False,
        "recommendation": "existing_h5_baseline_reference",
        "notes": "Reached h5 in existing normalized-insertion artifact; Flowstar width ratios intentionally disabled for endpoint-only legacy width semantics.",
        "_accepted_h": h_values,
        "_last_segment_width_sum": "",
        "_tube_width_sum": "",
    }
    return summary, out_segments, h_values


def make_samples(count: int = SAMPLE_RANDOM_COUNT) -> list[tuple[float, float]]:
    samples = [
        (1.1, 2.35),
        (1.1, 2.45),
        (1.4, 2.35),
        (1.4, 2.45),
        (1.25, 2.4),
    ]
    rng = random.Random(20260610)
    for _ in range(count):
        samples.append((rng.uniform(1.1, 1.4), rng.uniform(2.35, 2.45)))
    return samples


def _diag_status(row: Mapping[str, Any]) -> str:
    raw = str(row.get("validation_status", "")).strip().lower()
    if raw in {"failed", "failure", "rejected"}:
        return "rejected"
    if raw in {"validated", "accepted", "success", "passed"}:
        return "accepted"
    return raw or "unknown"


def run_torch_h5_mode(mode: str, horizon: float, wall_cap_s: float) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if mode not in MODE_SPECS:
        raise ValueError(f"unsupported mode: {mode}")
    spec = MODE_SPECS[mode]
    current: TMVector | list[Interval] = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state: FlowstarNormalFlowpipeState | None = None
    samples = make_samples()
    h_next = H_MAX
    t = 0.0
    start = time.perf_counter()
    accepted_h: list[float] = []
    segment_rows: list[dict[str, Any]] = []
    rejected_attempts = 0
    sample_violations = 0
    sample_max_violation = 0.0
    status = "completed"
    message = "validated to requested h5 horizon"
    last_segment_width = {"x": "", "y": "", "sum": ""}

    while t < horizon - 1e-12:
        elapsed = time.perf_counter() - start
        if elapsed >= wall_cap_s:
            status = "timeout"
            message = f"wall-time cap reached before segment {len(accepted_h)}"
            break
        remaining = horizon - t
        h_try = min(h_next, H_MAX, remaining)
        local_h_min = min(H_MIN, h_try)
        diagnostics: list[dict[str, Any]] = []
        seg = flowpipe_step_flowstar_style_adaptive(
            van_der_pol_flowstar_expression_ode,
            current,
            h=h_try,
            h_min=local_h_min,
            h_max=H_MAX,
            order=ORDER,
            target_remainder_radius=TARGET_RADIUS,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_mode=str(spec["validation_mode"]),
            reset_mode="normalized_insertion",
            grow_factor=float(spec["grow_factor"]),
            step_policy_mode=str(spec["step_policy_mode"]),
            flowstar_normal_state=normal_state,
            diagnostics=diagnostics,
            diagnostics_context={"mode": mode, "segment_index": len(accepted_h), "t_before": t},
        )
        rejected_attempts += sum(1 for row in diagnostics if _diag_status(row) == "rejected")
        try:
            segment_box = seg.tm.range_box()
            final_box = seg.final_tm.range_box()
        except Exception as exc:
            status = "failed"
            message = f"range evaluation failed: {exc}"
            break
        finite = intervals_are_finite(segment_box) and intervals_are_finite(final_box)
        row_status = "validated" if seg.status == "validated" and finite and seg.reset_tm is not None else "failed"
        x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = _interval_bounds(segment_box)
        t_hi = t + float(seg.h)
        segment_rows.append(
            {
                "source": "torch",
                "mode": mode,
                "segment_index": len(segment_rows),
                "status": row_status,
                "t_lo": t,
                "t_hi": t_hi,
                "h": float(seg.h),
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "width_x": width_x,
                "width_y": width_y,
                "width_sum": width_sum,
                "box_semantics": "torch_segment_tm_range",
                "step_rejections": getattr(seg, "step_rejections", ""),
                "next_h": seg.next_h if seg.next_h is not None else "",
                "message": seg.message,
            }
        )
        if row_status != "validated":
            status = "failed"
            message = seg.message or "validation failed"
            break

        accepted_h.append(float(seg.h))
        samples = [_advance_sample(point, float(seg.h)) for point in samples]
        for point in samples:
            vx = _interval_violation(point[0], final_box[0])
            vy = _interval_violation(point[1], final_box[1])
            if vx > 0.0:
                sample_violations += 1
                sample_max_violation = max(sample_max_violation, vx)
            if vy > 0.0:
                sample_violations += 1
                sample_max_violation = max(sample_max_violation, vy)
        last_segment_width = {"x": width_x, "y": width_y, "sum": width_sum}
        t = t_hi
        current = seg.reset_tm
        normal_state = seg.flowstar_normal_state
        h_next = float(seg.next_h if seg.next_h is not None else min(float(seg.h) * float(spec["grow_factor"]), H_MAX))

    runtime_s = time.perf_counter() - start
    validated = [row for row in segment_rows if row.get("status") == "validated"]
    h_vals = [float(row["h"]) for row in validated]
    non_final_below_min = sum(
        1
        for row in validated
        if float(row["h"]) < H_MIN - 1e-12 and float(row["t_hi"]) < horizon - 1e-9
    )
    if status == "completed" and t < horizon - 1e-12:
        status = "stopped"
        message = "stopped before h5"
    sample_status = "passed" if sample_violations == 0 and validated else "failed"
    summary = {
        "source": "torch",
        "mode": mode,
        "status": status,
        "reached_t": t,
        "completed_h5": bool(status == "completed" and t >= horizon - 1e-9),
        "accepted_steps": len(validated),
        "rejected_attempts": rejected_attempts,
        "min_h_used": min(h_vals) if h_vals else "",
        "h_below_flowstar_min_count": non_final_below_min,
        "runtime_s": runtime_s,
        "final_width_x": last_segment_width["x"],
        "final_width_y": last_segment_width["y"],
        "final_width_sum": last_segment_width["sum"],
        "flowstar_reference_final_width_sum": "",
        "last_segment_width_ratio_vs_flowstar": "",
        "tube_width_ratio_vs_flowstar": "",
        "schedule_distance_vs_flowstar": "",
        "schedule_prefix_match_count": "",
        "sample_containment_status": sample_status,
        "sample_violations": sample_violations,
        "default_behavior_changed": False,
        "recommendation": "",
        "notes": message,
        "_accepted_h": accepted_h,
        "_last_segment_width_sum": last_segment_width["sum"],
        "_tube_width_sum": _tube_from_segments(validated).get("tube_width_sum", ""),
        "_sample_count": len(samples),
        "_sample_max_violation": sample_max_violation,
    }
    sample_row = {
        "source": "torch",
        "mode": mode,
        "sample_count": len(samples),
        "sample_containment_status": sample_status,
        "sample_violations": sample_violations,
        "max_violation": sample_max_violation,
        "checked_segments": len(validated),
        "notes": "corners, center, and deterministic random samples checked against final-time endpoint boxes after each accepted segment",
    }
    return summary, segment_rows, sample_row


def finalize_rows(summary_rows: list[dict[str, Any]], flowstar_h: Sequence[float], flow_ref: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    flow_last = flow_ref.get("_last_segment_width_sum") or flow_ref.get("final_width_sum", "")
    flow_tube = flow_ref.get("_tube_width_sum", "")
    width_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        accepted_h = list(row.get("_accepted_h", []))
        if flowstar_h and accepted_h:
            row["schedule_distance_vs_flowstar"] = schedule_distance(list(flowstar_h), accepted_h)
            row["schedule_prefix_match_count"] = _prefix_match_count(list(flowstar_h), accepted_h)
        row["flowstar_reference_final_width_sum"] = flow_ref.get("final_width_sum", "")
        source = str(row.get("source", ""))
        mode = str(row.get("mode", ""))
        if source == "flowstar":
            last_ratio = 1.0
            tube_ratio = 1.0
            enabled = True
            disabled = ""
            comp_sem = "reference segment/tube"
        elif source == "torch" and row.get("_last_segment_width_sum") not in (None, ""):
            last_ratio = segment_ratio_vs_flowstar(row.get("_last_segment_width_sum"), flow_last, "segment_box", "flowstar_gnuplot_segment_box")
            tube_ratio = tube_ratio_vs_flowstar(row.get("_tube_width_sum"), flow_tube, "segment_tube", "flowstar_gnuplot_segment_tube")
            enabled = last_ratio != "" and tube_ratio != ""
            disabled = "" if enabled else "missing Flowstar segment/tube width reference"
            comp_sem = "torch segment TM boxes vs Flowstar GNUPLOT segment boxes"
        else:
            last_ratio = ""
            tube_ratio = ""
            enabled = False
            disabled = "endpoint-only or legacy width artifact; Flowstar GNUPLOT segment ratio disabled"
            comp_sem = "disabled endpoint-vs-segment comparison"
        row["last_segment_width_ratio_vs_flowstar"] = last_ratio
        row["tube_width_ratio_vs_flowstar"] = tube_ratio
        width_rows.append(
            {
                "source": source,
                "mode": mode,
                "comparison_enabled": enabled,
                "comparison_semantics": comp_sem,
                "flowstar_reference_final_width_sum": flow_ref.get("final_width_sum", ""),
                "last_segment_width_sum": row.get("_last_segment_width_sum", row.get("final_width_sum", "")),
                "last_segment_width_ratio_vs_flowstar": last_ratio,
                "flowstar_reference_tube_width_sum": flow_tube,
                "tube_width_sum": row.get("_tube_width_sum", ""),
                "tube_width_ratio_vs_flowstar": tube_ratio,
                "disabled_reason": disabled,
            }
        )
        schedule_rows.append(
            {
                "source": source,
                "mode": mode,
                "schedule_distance_vs_flowstar": row.get("schedule_distance_vs_flowstar", ""),
                "schedule_prefix_match_count": row.get("schedule_prefix_match_count", ""),
                "schedule_prefix_matches_flowstar": bool(row.get("schedule_prefix_match_count") == len(flowstar_h)) if flowstar_h and accepted_h else "",
                "accepted_h_sequence": ";".join(_format(h) for h in accepted_h),
                "flowstar_h_sequence": ";".join(_format(h) for h in flowstar_h),
            }
        )
    return summary_rows, width_rows, schedule_rows


def add_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mode = {str(row.get("mode")): row for row in rows}
    flow_policy = by_mode.get("raw_remainder_compat_flowstar_step_policy", {})
    compat_default = by_mode.get("raw_remainder_compat_default_policy", {})
    best = by_mode.get(BEST_NORMALIZED_H5_MODE, {})
    h5_completed = bool(flow_policy.get("completed_h5"))
    sample_ok = flow_policy.get("sample_containment_status") == "passed"
    no_submin = int(float(flow_policy.get("h_below_flowstar_min_count") or 0)) == 0
    width_ratio = _num(flow_policy.get("last_segment_width_ratio_vs_flowstar"))
    tube_ratio = _num(flow_policy.get("tube_width_ratio_vs_flowstar"))
    compat_dist = _num(compat_default.get("schedule_distance_vs_flowstar"))
    flow_dist = _num(flow_policy.get("schedule_distance_vs_flowstar"))
    schedule_improved = flow_dist is not None and compat_dist is not None and flow_dist < compat_dist
    width_close = width_ratio is not None and tube_ratio is not None and width_ratio <= 2.0 and tube_ratio <= 2.0
    for row in rows:
        if row.get("source") == "flowstar":
            row["recommendation"] = "reference_only"
        elif row.get("mode") == BEST_NORMALIZED_H5_MODE:
            row["recommendation"] = "existing_normalized_insertion_h5_baseline"
        elif row.get("mode") == "raw_remainder_compat_flowstar_step_policy":
            if h5_completed and sample_ok and no_submin and schedule_improved and width_close:
                row["recommendation"] = "h10_candidate_after_review"
            elif h5_completed and sample_ok and no_submin and schedule_improved:
                row["recommendation"] = "review_width_before_h10"
            elif h5_completed:
                row["recommendation"] = "h5_completed_but_not_h10_ready"
            else:
                row["recommendation"] = "do_not_run_h10"
        elif str(row.get("mode", "")).startswith("raw_remainder_compat"):
            row["recommendation"] = "compare_against_flowstar_step_policy"
        else:
            row["recommendation"] = "baseline_only"
    if best and flow_policy:
        best_width = _num(best.get("final_width_sum"))
        flow_width = _num(flow_policy.get("final_width_sum"))
        if best_width is not None and flow_width is not None:
            flow_policy["notes"] = str(flow_policy.get("notes", "")) + f"; previous_normalized_h5_final_width_sum={_format(best_width)}; raw_compat_flowstar_step_last_segment_width_sum={_format(flow_width)}"
    return rows


def write_report(path: Path, summary_rows: Sequence[Mapping[str, Any]], width_rows: Sequence[Mapping[str, Any]], horizon: float) -> None:
    by_mode = {str(row.get("mode")): row for row in summary_rows}
    flow_policy = by_mode.get("raw_remainder_compat_flowstar_step_policy", {})
    compat_default = by_mode.get("raw_remainder_compat_default_policy", {})
    best = by_mode.get(BEST_NORMALIZED_H5_MODE, {})
    h5_completed = bool(flow_policy.get("completed_h5"))
    sample_ok = flow_policy.get("sample_containment_status") == "passed"
    h_below = flow_policy.get("h_below_flowstar_min_count", "")
    width_ratio = flow_policy.get("last_segment_width_ratio_vs_flowstar", "")
    tube_ratio = flow_policy.get("tube_width_ratio_vs_flowstar", "")
    compat_dist = _num(compat_default.get("schedule_distance_vs_flowstar"))
    flow_dist = _num(flow_policy.get("schedule_distance_vs_flowstar"))
    schedule_improved = flow_dist is not None and compat_dist is not None and flow_dist < compat_dist
    h10_next = flow_policy.get("recommendation") == "h10_candidate_after_review"
    best_width = _num(best.get("final_width_sum"))
    flow_width = _num(flow_policy.get("final_width_sum"))
    if best_width is None or flow_width is None:
        baseline_cmp = "unknown"
    elif flow_width < best_width:
        baseline_cmp = "tighter last-segment width than previous normalized-insertion h5 endpoint artifact"
    elif flow_width > best_width:
        baseline_cmp = "wider last-segment width than previous normalized-insertion h5 endpoint artifact"
    else:
        baseline_cmp = "same width as previous normalized-insertion h5 endpoint artifact"
    lines = [
        "# Flowstar Raw Remainder Compat h5 Audit",
        "",
        "This h5-only audit does not run h10, does not add NNCS/GPU work, does not add symbolic queue variants, does not change default solver behavior, and does not claim Flowstar parity.",
        "",
        "## Scope",
        "",
        f"- Requested horizon: `{_format(horizon)}`.",
        "- Compat remains opt-in through `validation_mode=\"flowstar_raw_remainder_compat\"` and `step_policy_mode=\"flowstar_compat\"`.",
        "- Flowstar GNUPLOT rectangles are treated as segment boxes, not endpoint boxes.",
        "- Endpoint-only legacy artifacts are not used for Flowstar width ratios.",
        "",
        "## Answers",
        "",
        f"- Did raw compat + Flowstar step policy reach h5? `{'yes' if h5_completed else 'no'}`; reached_t `{_format(flow_policy.get('reached_t'))}`.",
        f"- Did it remain sample-contained? `{'yes' if sample_ok else 'no'}`; violations `{_format(flow_policy.get('sample_violations'))}`.",
        f"- Did it use any non-final h below 0.002? `{'yes' if str(h_below) not in {'', '0', '0.0'} else 'no'}`; count `{_format(h_below)}`.",
        f"- Is it width-close to Flowstar h5 segment/tube boxes? last-segment ratio `{_format(width_ratio)}`, tube ratio `{_format(tube_ratio)}`.",
        f"- Compared with previous normalized-insertion h5, it is `{baseline_cmp}`; previous runtime_s `{_format(best.get('runtime_s'))}`, current runtime_s `{_format(flow_policy.get('runtime_s'))}`.",
        f"- Did Flowstar step policy improve schedule match over raw compat default? `{'yes' if schedule_improved else 'no'}`; default `{_format(compat_dist)}`, Flowstar step `{_format(flow_dist)}`.",
        f"- Does h5 justify h10 next? `{'yes' if h10_next else 'no'}`; recommendation `{_format(flow_policy.get('recommendation'))}`.",
        f"- Did compat become too conservative? `{'no' if h5_completed else 'yes'}`.",
        "",
        "## Summary",
        "",
        "| mode | status | reached_t | completed_h5 | accepted | rejected | min_h | below_0.002 | final_width_sum | last_ratio | tube_ratio | schedule_distance | samples | recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                _format(value)
                for value in (
                    row.get("mode"),
                    row.get("status"),
                    row.get("reached_t"),
                    row.get("completed_h5"),
                    row.get("accepted_steps"),
                    row.get("rejected_attempts"),
                    row.get("min_h_used"),
                    row.get("h_below_flowstar_min_count"),
                    row.get("final_width_sum"),
                    row.get("last_segment_width_ratio_vs_flowstar"),
                    row.get("tube_width_ratio_vs_flowstar"),
                    row.get("schedule_distance_vs_flowstar"),
                    row.get("sample_containment_status"),
                    row.get("recommendation"),
                )
            )
            + " |"
        )
    lines.extend([
        "",
        "## Width Semantics",
        "",
        "| mode | enabled | semantics | disabled_reason |",
        "| --- | --- | --- | --- |",
    ])
    for row in width_rows:
        lines.append(
            "| "
            + " | ".join(_format(row.get(field)) for field in ("mode", "comparison_enabled", "comparison_semantics", "disabled_reason"))
            + " |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `outputs/flowstar_raw_remainder_compat_h5/h5_summary.csv`",
        "- `outputs/flowstar_raw_remainder_compat_h5/h5_segments.csv`",
        "- `outputs/flowstar_raw_remainder_compat_h5/h5_width_vs_flowstar.csv`",
        "- `outputs/flowstar_raw_remainder_compat_h5/h5_schedule_compare.csv`",
        "- `outputs/flowstar_raw_remainder_compat_h5/h5_sample_containment.csv`",
        "- `outputs/flowstar_raw_remainder_compat_h5/h5_report.md`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out_dir: Path, summary_rows: Sequence[Mapping[str, Any]], segment_rows: Sequence[Mapping[str, Any]], width_rows: Sequence[Mapping[str, Any]], schedule_rows: Sequence[Mapping[str, Any]], sample_rows: Sequence[Mapping[str, Any]], horizon: float) -> None:
    _write_csv(out_dir / "h5_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "h5_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "h5_width_vs_flowstar.csv", WIDTH_FIELDS, width_rows)
    _write_csv(out_dir / "h5_schedule_compare.csv", SCHEDULE_FIELDS, schedule_rows)
    _write_csv(out_dir / "h5_sample_containment.csv", SAMPLE_FIELDS, sample_rows)
    write_report(out_dir / "h5_report.md", summary_rows, width_rows, horizon)


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    horizon = float(args.horizon)
    if horizon > 5.0 + 1e-12:
        raise ValueError("h5 audit is capped at T=5.0; do not run h10 from this script")
    flow_ref, flow_segments, flow_h = load_flowstar_reference(args.flowstar_segments.resolve(), horizon)
    if not flow_ref:
        raise FileNotFoundError(f"missing usable Flowstar segment reference: {args.flowstar_segments}")
    summary_rows: list[dict[str, Any]] = [flow_ref]
    segment_rows: list[dict[str, Any]] = list(flow_segments)
    sample_rows: list[dict[str, Any]] = [
        {
            "source": "flowstar",
            "mode": "generated_flowstar_h5_reference",
            "sample_count": "",
            "sample_containment_status": "not_applicable",
            "sample_violations": "",
            "max_violation": "",
            "checked_segments": len(flow_segments),
            "notes": "reference segment boxes only; sample containment checked on PyTorch modes",
        }
    ]
    normalized, normalized_segments, _normalized_h = load_existing_normalized_h5(
        args.normalized_h5_summary.resolve(),
        args.normalized_h5_segments.resolve(),
        horizon,
        flow_h,
        flow_ref,
    )
    if normalized is not None:
        summary_rows.append(normalized)
        segment_rows.extend(normalized_segments)
        sample_rows.append(
            {
                "source": "torch_existing_artifact",
                "mode": BEST_NORMALIZED_H5_MODE,
                "sample_count": "",
                "sample_containment_status": "not_run_existing_artifact",
                "sample_violations": "",
                "max_violation": "",
                "checked_segments": len(normalized_segments),
                "notes": "existing artifact comparison only",
            }
        )
    modes = list(MODE_SPECS)
    if args.skip_current_default:
        modes.remove("current_no_queue_default_policy")
    for mode in modes:
        summary, segments, samples = run_torch_h5_mode(mode, horizon, float(args.wall_cap_s))
        summary_rows.append(summary)
        segment_rows.extend(segments)
        sample_rows.append(samples)
    summary_rows, width_rows, schedule_rows = finalize_rows(summary_rows, flow_h, flow_ref)
    summary_rows = add_recommendations(summary_rows)
    for row in summary_rows:
        row.pop("_accepted_h", None)
        row.pop("_last_segment_width_sum", None)
        row.pop("_tube_width_sum", None)
        row.pop("_last_segment_width_x", None)
        row.pop("_last_segment_width_y", None)
        row.pop("_sample_count", None)
        row.pop("_sample_max_violation", None)
    write_outputs(args.out_dir.resolve(), summary_rows, segment_rows, width_rows, schedule_rows, sample_rows, horizon)
    return summary_rows, segment_rows, width_rows, schedule_rows, sample_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--flowstar-segments", type=Path, default=DEFAULT_FLOWSTAR_SEGMENTS)
    parser.add_argument("--flowstar-summary", type=Path, default=DEFAULT_FLOWSTAR_SUMMARY)
    parser.add_argument("--normalized-h5-summary", type=Path, default=DEFAULT_NORMALIZED_H5_SUMMARY)
    parser.add_argument("--normalized-h5-segments", type=Path, default=DEFAULT_NORMALIZED_H5_SEGMENTS)
    parser.add_argument("--wall-cap-s", type=float, default=3600.0)
    parser.add_argument("--skip-current-default", action="store_true", help="Skip the cheapness-gated current default comparison.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_rows, segment_rows, _width_rows, _schedule_rows, _sample_rows = run(args)
    print(f"wrote {args.out_dir.resolve() / 'h5_summary.csv'} ({len(summary_rows)} rows)")
    print(f"wrote {args.out_dir.resolve() / 'h5_segments.csv'} ({len(segment_rows)} rows)")
    print(f"wrote {args.out_dir.resolve() / 'h5_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
