#!/usr/bin/env python3
"""h10 audit for opt-in range-midpoint-centered right-map insertion."""
from __future__ import annotations

import argparse
import csv
import math
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
from flowstar_raw_remainder_compat_experiment import ORDER, TARGET_RADIUS, _format, van_der_pol_flowstar_expression_ode
from flowstar_raw_remainder_compat_h5 import DEFAULT_FLOWSTAR_SEGMENTS, _diag_status, _interval_bounds, _tube_from_segments, load_flowstar_reference, make_samples
from flowstar_raw_remainder_compat_h5_right_map_centering import (
    _bool,
    _float,
    _latest_validation_row,
    _raw_target_violation,
    _ratio,
    _reduction,
    _target_margin,
    _validated_rows,
    csv_row_count,
    physical_line_count,
)
from flowstar_raw_remainder_compat_short_horizon import H_MAX, H_MIN, _advance_sample, _interval_violation, schedule_distance

DEFAULT_HORIZON = 10.0
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h10_right_map_centering"
FLOWSTAR_MODE = "generated_flowstar_h10_reference"
CONSTANT_ADAPTIVE = "constant_adaptive_h10"
RANGE_ADAPTIVE = "range_midpoint_adaptive_h10"
RANGE_ON_CONSTANT = "range_midpoint_on_constant_schedule"
CONSTANT_ON_RANGE = "constant_on_range_midpoint_schedule"
UNKNOWN_FLOWSTAR_COMPONENT = "unknown_missing_h10_reference_component_fields"
DECISIONS = {
    "h10_reached_with_controlled_width",
    "h10_reached_but_width_gap_large",
    "h10_not_reached_but_materially_improved",
    "h10_not_reached_no_material_improvement",
    "reject_due_to_soundness_or_reconstruction_failure",
}

SUMMARY_FIELDS = [
    "source", "run_kind", "mode", "right_map_center_mode", "status", "reached_t", "reached_h10",
    "accepted_steps", "rejected_attempts", "first_failure_step", "first_failure_time", "first_failure_reason",
    "shared_schedule_end_time", "min_h_used", "runtime_s", "final_segment_width_sum", "tube_width_sum",
    "flowstar_final_width_sum", "flowstar_tube_width_sum", "flowstar_final_width_ratio", "flowstar_tube_width_ratio",
    "schedule_distance_vs_flowstar", "sample_sanity_violations", "sample_sanity_status",
    "minimum_target_margin", "minimum_target_margin_step", "minimum_target_margin_time", "minimum_target_margin_h",
    "raw_residual_target_violations", "max_reconstruction_polynomial_abs_diff", "max_reconstruction_remainder_endpoint_diff",
    "max_immediate_same_state_saving", "max_cumulative_downstream_saving", "common_time_width_worsening_count",
    "cross_schedule_centering_improvement", "decision", "notes",
]

SEGMENT_FIELDS = [
    "source", "mode", "run_kind", "right_map_center_mode", "segment_index", "t_lo", "t_hi",
    "h_attempted", "h_accepted", "prescribed_h", "h_delta", "accepted_rejected", "rejection_reason",
    "validation_attempts", "x_lo", "x_hi", "y_lo", "y_hi", "width_x", "width_y", "width_sum",
    "target_margin_min", "raw_residual_target_violation",
    "raw_ctrunc_residual_width_sum", "polynomial_range_width_sum", "full_step_tube_width_sum",
    "inserted_range_width_sum", "constant_scale_sum", "hypothetical_centered_scale_sum",
    "actual_centered_scale_sum", "constant_reset_width_sum", "hypothetical_centered_reset_width_sum",
    "actual_centered_reset_width_sum", "immediate_reset_reduction_relative",
    "cumulative_reset_reduction_relative", "final_segment_width_sum", "tube_prefix_width_sum",
    "reconstruction_polynomial_max_abs_diff", "reconstruction_remainder_endpoint_diff",
    "step_rejections", "had_prior_rejection", "missing_flowstar_component_fields", "message",
]

ATTEMPT_FIELDS = [
    "source", "mode", "run_kind", "segment_index", "attempt_index", "t_before", "h_attempted",
    "prescribed_h", "validation_status", "accepted_rejected", "rejection_reason", "target_margin_min",
    "raw_residual_target_violation", "raw_ctrunc_residual_width_sum", "polynomial_range_width_sum",
    "flowstar_raw_remainder_compat_check_remainder_width_sum", "subset_result", "finite_residual",
]

CROSS_FIELDS = [
    "replay_kind", "schedule_source_mode", "replay_mode", "step_index", "prescribed_h",
    "source_status", "replay_status", "source_h", "replay_h", "replay_h_delta", "h_sequence_modified",
    "source_t_hi", "replay_t_hi", "source_width_sum", "replay_width_sum", "width_reduction_relative",
    "source_reset_width_sum", "replay_reset_width_sum", "cumulative_reset_reduction_relative",
    "source_immediate_same_state_saving", "replay_immediate_same_state_saving",
    "replay_validation_failure_recorded", "notes",
]

CHECKPOINT_FIELDS = [
    "event_name", "checkpoint_t", "threshold", "source", "mode", "run_kind", "segment_index", "status",
    "t_lo", "t_hi", "h", "final_segment_width_sum", "tube_prefix_width_sum", "flowstar_segment_width_sum",
    "flowstar_final_width_ratio", "flowstar_tube_width_ratio", "target_margin_min",
    "immediate_same_state_saving", "cumulative_downstream_saving", "missing_flowstar_component_fields", "notes",
]

MARGIN_FIELDS = [
    "rank", "mode", "run_kind", "segment_index", "t_lo", "t_hi", "h", "target_margin_min",
    "final_segment_width_sum", "tube_prefix_width_sum", "had_prior_rejection", "accepted_rejected",
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
    for path in sorted(out_dir.glob("h10_right_map_centering_*.csv")):
        if path.name == "h10_right_map_centering_formatting.csv":
            continue
        physical = physical_line_count(path)
        parsed = csv_row_count(path)
        rows.append({"path": _display_path(path), "physical_line_count": physical, "csv_reader_row_count": parsed, "status": "ok" if physical == parsed else "mismatch"})
    for path in sorted(out_dir.glob("h10_right_map_centering_*.md")) + sorted(out_dir.glob("h10_right_map_centering_*.txt")):
        rows.append({"path": _display_path(path), "physical_line_count": physical_line_count(path), "csv_reader_row_count": "", "status": "ok"})
    return rows


def _max_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    vals = [_float(row.get(field)) for row in rows]
    finite = [v for v in vals if v is not None]
    return max(finite) if finite else ""


def _accepted_h10(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("accepted_rejected") != "accepted":
            continue
        h = _float(row.get("h_accepted"))
        if h is not None:
            values.append(h)
    return values


def _min_margin_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    candidates = [row for row in rows if row.get("accepted_rejected") == "accepted" and _float(row.get("target_margin_min")) is not None]
    if not candidates:
        return {}
    return min(candidates, key=lambda row: float(row["target_margin_min"]))


def _reconstruction_endpoint_diff(row: Mapping[str, Any]) -> float | str:
    lo = _float(row.get("reconstruction_remainder_lo_diff"))
    hi = _float(row.get("reconstruction_remainder_hi_diff"))
    if lo is None and hi is None:
        return ""
    return max(lo or 0.0, hi or 0.0)


def _flowstar_rows(flow_segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in flow_segments:
        out = {field: "" for field in SEGMENT_FIELDS}
        out.update({
            "source": "flowstar",
            "mode": FLOWSTAR_MODE,
            "run_kind": "reference",
            "right_map_center_mode": "not_applicable",
            "segment_index": row.get("segment_index", len(rows)),
            "t_lo": row.get("t_lo", ""),
            "t_hi": row.get("t_hi", ""),
            "h_accepted": row.get("h", ""),
            "x_lo": row.get("x_lo", ""),
            "x_hi": row.get("x_hi", ""),
            "y_lo": row.get("y_lo", ""),
            "y_hi": row.get("y_hi", ""),
            "width_x": row.get("width_x", ""),
            "width_y": row.get("width_y", ""),
            "width_sum": row.get("width_sum", ""),
            "accepted_rejected": "accepted",
            "final_segment_width_sum": row.get("width_sum", ""),
            "full_step_tube_width_sum": row.get("width_sum", ""),
            "missing_flowstar_component_fields": UNKNOWN_FLOWSTAR_COMPONENT,
            "message": "Flowstar h10 reference exposes segment boxes; internal right-map component fields are unknown, not zero.",
        })
        for field in SEGMENT_FIELDS:
            if field.endswith("_width_sum") and field not in {"final_segment_width_sum", "full_step_tube_width_sum"}:
                out[field] = UNKNOWN_FLOWSTAR_COMPONENT
        rows.append(out)
    return rows


def run_centering_h10(
    *,
    mode: str,
    run_kind: str,
    right_map_center_mode: str,
    horizon: float,
    wall_cap_s: float,
    prescribed_h: Sequence[float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    current: TMVector | list[Interval] = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state: FlowstarNormalFlowpipeState | None = None
    samples = make_samples()
    h_next = H_MAX
    t = 0.0
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    rejected_attempts = 0
    sample_violations = 0
    sample_max_violation = 0.0
    status = "completed"
    message = "validated to requested h10 horizon"
    first_failure_step: int | str = ""
    first_failure_time: float | str = ""
    first_failure_reason = ""
    prescribed = list(prescribed_h or [])
    prescribed_index = 0

    while t < horizon - 1e-12:
        if time.perf_counter() - start >= wall_cap_s:
            status = "timeout"
            message = f"wall-time cap reached before segment {len(rows)}"
            first_failure_step = len(rows)
            first_failure_time = t
            first_failure_reason = message
            break
        remaining = horizon - t
        if prescribed_h is None:
            h_try = min(h_next, H_MAX, remaining)
            prescribed_value: float | str = ""
        else:
            if prescribed_index >= len(prescribed):
                status = "failed"
                message = "prescribed h sequence ended before horizon"
                first_failure_step = len(rows)
                first_failure_time = t
                first_failure_reason = message
                break
            h_try = float(prescribed[prescribed_index])
            prescribed_value = h_try
            prescribed_index += 1
            if h_try > remaining and h_try - remaining <= 1e-10:
                h_try = remaining
            elif h_try > remaining + 1e-10:
                status = "failed"
                message = "prescribed h exceeds remaining horizon"
                first_failure_step = len(rows)
                first_failure_time = t
                first_failure_reason = message
                break
        local_h_min = h_try if prescribed_h is not None else min(H_MIN, h_try)
        local_h_max = h_try if prescribed_h is not None else H_MAX
        diagnostics: list[dict[str, Any]] = []
        seg = flowpipe_step_flowstar_style_adaptive(
            van_der_pol_flowstar_expression_ode,
            current,
            h=h_try,
            h_min=local_h_min,
            h_max=local_h_max,
            order=ORDER,
            target_remainder_radius=TARGET_RADIUS,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_mode="flowstar_raw_remainder_compat",
            reset_mode="normalized_insertion",
            grow_factor=1.5,
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=normal_state,
            right_map_center_mode=right_map_center_mode,
            diagnostics=diagnostics,
            diagnostics_context={"mode": mode, "run_kind": run_kind, "segment_index": len(rows), "t_before": t},
        )
        rejected_attempts += sum(1 for row in diagnostics if _diag_status(row) == "rejected")
        for diag in diagnostics:
            margin = _target_margin(diag)
            attempts.append({
                "source": "torch",
                "mode": mode,
                "run_kind": run_kind,
                "segment_index": len(rows),
                "attempt_index": diag.get("attempt_index", ""),
                "t_before": t,
                "h_attempted": diag.get("h", diag.get("h_try", h_try)),
                "prescribed_h": prescribed_value,
                "validation_status": diag.get("validation_status", ""),
                "accepted_rejected": "accepted" if str(diag.get("validation_status", "")).lower() == "validated" else "rejected",
                "rejection_reason": diag.get("rejection_reason", diag.get("validation_message", "")),
                "target_margin_min": margin,
                "raw_residual_target_violation": _raw_target_violation(diag),
                "raw_ctrunc_residual_width_sum": diag.get("raw_ctrunc_residual_width_sum", ""),
                "polynomial_range_width_sum": diag.get("polynomial_range_width_sum", ""),
                "flowstar_raw_remainder_compat_check_remainder_width_sum": diag.get("flowstar_raw_remainder_compat_check_remainder_width_sum", ""),
                "subset_result": diag.get("subset_result", ""),
                "finite_residual": diag.get("finite_residual", ""),
            })
        validation = _latest_validation_row(diagnostics)
        stats = dict(getattr(seg, "flowstar_normal_stats", None) or {})
        try:
            segment_box = seg.tm.range_box()
            final_box = seg.final_tm.range_box()
            x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = _interval_bounds(segment_box)
            finite = intervals_are_finite(segment_box) and intervals_are_finite(final_box)
        except Exception as exc:
            final_box = []
            x_lo = x_hi = y_lo = y_hi = ""
            width_x = width_y = width_sum = ""
            finite = False
            message = f"range evaluation failed: {exc}"
        accepted = seg.status == "validated" and finite and seg.reset_tm is not None
        row_status = "accepted" if accepted else "rejected"
        t_hi = t + float(seg.h)
        target_margin = _target_margin(validation)
        h_delta = "" if prescribed_value == "" else float(seg.h) - float(prescribed_value)
        row = {
            "source": "torch",
            "mode": mode,
            "run_kind": run_kind,
            "right_map_center_mode": right_map_center_mode,
            "segment_index": len(rows),
            "t_lo": t,
            "t_hi": t_hi,
            "h_attempted": h_try,
            "h_accepted": float(seg.h) if accepted else "",
            "x_lo": x_lo,
            "x_hi": x_hi,
            "y_lo": y_lo,
            "y_hi": y_hi,
            "width_x": width_x,
            "width_y": width_y,
            "width_sum": width_sum,
            "prescribed_h": prescribed_value,
            "h_delta": h_delta,
            "accepted_rejected": row_status,
            "rejection_reason": "" if accepted else (seg.message or validation.get("validation_message", "") or message),
            "validation_attempts": seg.validation_attempts,
            "target_margin_min": target_margin,
            "raw_residual_target_violation": _raw_target_violation(validation),
            "raw_ctrunc_residual_width_sum": validation.get("raw_ctrunc_residual_width_sum", ""),
            "polynomial_range_width_sum": validation.get("polynomial_range_width_sum", ""),
            "full_step_tube_width_sum": width_sum,
            "inserted_range_width_sum": stats.get("inserted_range_width_sum", ""),
            "constant_scale_sum": stats.get("constant_scale_sum", stats.get("baseline_scale_sum", "")),
            "hypothetical_centered_scale_sum": stats.get("hypothetical_centered_scale_sum", ""),
            "actual_centered_scale_sum": stats.get("actual_centered_scale_sum", stats.get("centered_scale_sum", "")),
            "constant_reset_width_sum": stats.get("baseline_reset_width_sum", ""),
            "hypothetical_centered_reset_width_sum": stats.get("hypothetical_centered_reset_width_sum", ""),
            "actual_centered_reset_width_sum": stats.get("centered_reset_width_sum", ""),
            "immediate_reset_reduction_relative": stats.get("immediate_reset_reduction_relative_sum", ""),
            "cumulative_reset_reduction_relative": "",
            "final_segment_width_sum": width_sum,
            "tube_prefix_width_sum": "",
            "reconstruction_polynomial_max_abs_diff": stats.get("reconstruction_polynomial_max_abs_diff", ""),
            "reconstruction_remainder_endpoint_diff": max(_float(stats.get("reconstruction_remainder_lo_diff")) or 0.0, _float(stats.get("reconstruction_remainder_hi_diff")) or 0.0) if stats else "",
            "step_rejections": getattr(seg, "step_rejections", ""),
            "had_prior_rejection": (getattr(seg, "step_rejections", 0) or 0) > 0,
            "missing_flowstar_component_fields": "",
            "message": seg.message or message,
        }
        rows.append(row)
        if not accepted:
            status = "failed"
            message = row["rejection_reason"] or "validation failed"
            first_failure_step = len(rows) - 1
            first_failure_time = t
            first_failure_reason = message
            break
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
        t = t_hi
        current = seg.reset_tm
        normal_state = seg.flowstar_normal_state
        h_next = float(seg.next_h if seg.next_h is not None else min(float(seg.h) * 1.5, H_MAX))
        tube = _tube_from_segments([r for r in rows if r.get("accepted_rejected") == "accepted"])
        rows[-1]["tube_prefix_width_sum"] = tube.get("tube_width_sum", "")

    accepted_rows = [row for row in rows if row.get("accepted_rejected") == "accepted"]
    h_vals = [float(row["h_accepted"]) for row in accepted_rows if _float(row.get("h_accepted")) is not None]
    last = accepted_rows[-1] if accepted_rows else {}
    tube = _tube_from_segments(accepted_rows)
    min_row = _min_margin_row(accepted_rows)
    if status == "completed" and t < horizon - 1e-12:
        status = "stopped"
        message = "stopped before h10"
    summary = {
        "source": "torch",
        "run_kind": run_kind,
        "mode": mode,
        "right_map_center_mode": right_map_center_mode,
        "status": status,
        "reached_t": t,
        "reached_h10": bool(status == "completed" and t >= horizon - 1e-9),
        "accepted_steps": len(accepted_rows),
        "rejected_attempts": rejected_attempts,
        "first_failure_step": first_failure_step,
        "first_failure_time": first_failure_time,
        "first_failure_reason": first_failure_reason,
        "shared_schedule_end_time": t if prescribed_h is not None else "",
        "min_h_used": min(h_vals) if h_vals else "",
        "runtime_s": time.perf_counter() - start,
        "final_segment_width_sum": last.get("final_segment_width_sum", ""),
        "tube_width_sum": tube.get("tube_width_sum", ""),
        "sample_sanity_violations": sample_violations,
        "sample_sanity_status": "passed" if sample_violations == 0 and accepted_rows else "failed",
        "minimum_target_margin": min_row.get("target_margin_min", ""),
        "minimum_target_margin_step": min_row.get("segment_index", ""),
        "minimum_target_margin_time": min_row.get("t_hi", ""),
        "minimum_target_margin_h": min_row.get("h_accepted", ""),
        "raw_residual_target_violations": sum(1 for row in rows if _bool(row.get("raw_residual_target_violation"))),
        "max_reconstruction_polynomial_abs_diff": _max_field(accepted_rows, "reconstruction_polynomial_max_abs_diff"),
        "max_reconstruction_remainder_endpoint_diff": _max_field(accepted_rows, "reconstruction_remainder_endpoint_diff"),
        "max_immediate_same_state_saving": _max_field(accepted_rows, "immediate_reset_reduction_relative"),
        "_accepted_h": h_vals,
        "_sample_max_violation": sample_max_violation,
        "notes": message + "; sample containment is a sanity check only, not a proof",
    }
    return summary, rows, attempts


def make_cross_schedule_rows(source_rows: Sequence[Mapping[str, Any]], replay_rows: Sequence[Mapping[str, Any]], *, replay_kind: str, source_mode: str, replay_mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_len = max(len(source_rows), len(replay_rows))
    for i in range(max_len):
        src = source_rows[i] if i < len(source_rows) else {}
        rep = replay_rows[i] if i < len(replay_rows) else {}
        _abs, width_rel = _reduction(src.get("final_segment_width_sum"), rep.get("final_segment_width_sum"))
        _rabs, reset_rel = _reduction(src.get("actual_centered_reset_width_sum"), rep.get("actual_centered_reset_width_sum"))
        h_delta = _float(rep.get("h_delta"))
        rows.append({
            "replay_kind": replay_kind,
            "schedule_source_mode": source_mode,
            "replay_mode": replay_mode,
            "step_index": i,
            "prescribed_h": rep.get("prescribed_h", ""),
            "source_status": src.get("accepted_rejected", "missing"),
            "replay_status": rep.get("accepted_rejected", "missing"),
            "source_h": src.get("h_accepted", ""),
            "replay_h": rep.get("h_accepted", ""),
            "replay_h_delta": rep.get("h_delta", ""),
            "h_sequence_modified": h_delta is not None and abs(h_delta) > 1e-10,
            "source_t_hi": src.get("t_hi", ""),
            "replay_t_hi": rep.get("t_hi", ""),
            "source_width_sum": src.get("final_segment_width_sum", ""),
            "replay_width_sum": rep.get("final_segment_width_sum", ""),
            "width_reduction_relative": width_rel,
            "source_reset_width_sum": src.get("actual_centered_reset_width_sum", ""),
            "replay_reset_width_sum": rep.get("actual_centered_reset_width_sum", ""),
            "cumulative_reset_reduction_relative": reset_rel,
            "source_immediate_same_state_saving": src.get("immediate_reset_reduction_relative", ""),
            "replay_immediate_same_state_saving": rep.get("immediate_reset_reduction_relative", ""),
            "replay_validation_failure_recorded": bool(rep and rep.get("accepted_rejected") != "accepted"),
            "notes": "cross-schedule replay with h_min=h_max=prescribed_h; no forced acceptance",
        })
    return rows


def _segment_containing(rows: Sequence[Mapping[str, Any]], t: float) -> Mapping[str, Any]:
    accepted = [row for row in rows if row.get("accepted_rejected") == "accepted"]
    for row in accepted:
        lo = _float(row.get("t_lo"))
        hi = _float(row.get("t_hi"))
        if lo is not None and hi is not None and lo - 1e-12 <= t <= hi + 1e-12:
            return row
    return accepted[-1] if accepted and t >= (_float(accepted[-1].get("t_hi")) or 0.0) - 1e-12 else {}


def first_h_divergence(const_rows: Sequence[Mapping[str, Any]], range_rows: Sequence[Mapping[str, Any]]) -> float | str:
    for c, r in zip(const_rows, range_rows):
        ch = _float(c.get("h_accepted"))
        rh = _float(r.get("h_accepted"))
        if ch is None or rh is None:
            continue
        if abs(ch - rh) > 1e-10:
            return min(_float(c.get("t_lo")) or 0.0, _float(r.get("t_lo")) or 0.0)
    return ""


def first_margin_below(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]], threshold: float) -> float | str:
    rows: list[Mapping[str, Any]] = []
    for rows_for_mode in rows_by_mode.values():
        rows.extend(rows_for_mode)
    candidates = [row for row in rows if row.get("accepted_rejected") == "accepted" and (_float(row.get("target_margin_min")) is not None) and float(row["target_margin_min"]) < threshold]
    if not candidates:
        return ""
    return min(candidates, key=lambda row: float(row.get("t_hi") or 0.0)).get("t_hi", "")


def first_ratio_above(torch_rows: Sequence[Mapping[str, Any]], flow_rows: Sequence[Mapping[str, Any]], threshold: float) -> float | str:
    for row in torch_rows:
        if row.get("accepted_rejected") != "accepted":
            continue
        t = _float(row.get("t_hi"))
        if t is None:
            continue
        flow = _segment_containing(flow_rows, t)
        ratio = _ratio(row.get("final_segment_width_sum"), flow.get("final_segment_width_sum"))
        if isinstance(ratio, float) and ratio > threshold:
            return t
    return ""


def make_checkpoints(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]], flow_rows: Sequence[Mapping[str, Any]], horizon: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {"event_name": "t_5", "checkpoint_t": 5.0, "threshold": ""},
        {"event_name": "first_h_sequence_divergence", "checkpoint_t": first_h_divergence(rows_by_mode[CONSTANT_ADAPTIVE], rows_by_mode[RANGE_ADAPTIVE]), "threshold": ""},
        {"event_name": "first_target_margin_lt_1e_6", "checkpoint_t": first_margin_below(rows_by_mode, 1e-6), "threshold": 1e-6},
        {"event_name": "first_target_margin_lt_1e_8", "checkpoint_t": first_margin_below(rows_by_mode, 1e-8), "threshold": 1e-8},
        {"event_name": "historical_window_t_6_473", "checkpoint_t": 6.473, "threshold": ""},
        {"event_name": "historical_window_t_7_496", "checkpoint_t": 7.496, "threshold": ""},
        {"event_name": "first_flowstar_ratio_gt_2", "checkpoint_t": first_ratio_above(rows_by_mode[RANGE_ADAPTIVE], flow_rows, 2.0), "threshold": 2.0},
        {"event_name": "first_flowstar_ratio_gt_5", "checkpoint_t": first_ratio_above(rows_by_mode[RANGE_ADAPTIVE], flow_rows, 5.0), "threshold": 5.0},
        {"event_name": "first_flowstar_ratio_gt_10", "checkpoint_t": first_ratio_above(rows_by_mode[RANGE_ADAPTIVE], flow_rows, 10.0), "threshold": 10.0},
    ]
    range_accepted = [row for row in rows_by_mode[RANGE_ADAPTIVE] if row.get("accepted_rejected") == "accepted"]
    if range_accepted:
        events.append({"event_name": "last_validated_segment", "checkpoint_t": range_accepted[-1].get("t_hi", ""), "threshold": ""})
    if range_accepted and (_float(range_accepted[-1].get("t_hi")) or 0.0) >= horizon - 1e-9:
        events.append({"event_name": "t_10", "checkpoint_t": horizon, "threshold": ""})
    out: list[dict[str, Any]] = []
    all_modes = {FLOWSTAR_MODE: flow_rows, **rows_by_mode}
    for event in events:
        t = _float(event.get("checkpoint_t"))
        if t is None:
            continue
        flow = _segment_containing(flow_rows, t)
        flow_tube = flow.get("tube_prefix_width_sum", "")
        for mode, rows in all_modes.items():
            segment = _segment_containing(rows, t)
            if not segment:
                out.append({"event_name": event["event_name"], "checkpoint_t": t, "threshold": event.get("threshold", ""), "mode": mode, "status": "missing", "notes": "no containing segment; no interpolation used"})
                continue
            ratio = _ratio(segment.get("final_segment_width_sum"), flow.get("final_segment_width_sum")) if mode != FLOWSTAR_MODE else 1.0
            tube_ratio = _ratio(segment.get("tube_prefix_width_sum"), flow_tube) if mode != FLOWSTAR_MODE else 1.0
            out.append({
                "event_name": event["event_name"],
                "checkpoint_t": t,
                "threshold": event.get("threshold", ""),
                "source": segment.get("source", ""),
                "mode": mode,
                "run_kind": segment.get("run_kind", ""),
                "segment_index": segment.get("segment_index", ""),
                "status": segment.get("accepted_rejected", ""),
                "t_lo": segment.get("t_lo", ""),
                "t_hi": segment.get("t_hi", ""),
                "h": segment.get("h_accepted", ""),
                "final_segment_width_sum": segment.get("final_segment_width_sum", ""),
                "tube_prefix_width_sum": segment.get("tube_prefix_width_sum", ""),
                "flowstar_segment_width_sum": flow.get("final_segment_width_sum", ""),
                "flowstar_final_width_ratio": ratio,
                "flowstar_tube_width_ratio": tube_ratio,
                "target_margin_min": segment.get("target_margin_min", ""),
                "immediate_same_state_saving": segment.get("immediate_reset_reduction_relative", UNKNOWN_FLOWSTAR_COMPONENT if mode == FLOWSTAR_MODE else ""),
                "cumulative_downstream_saving": segment.get("cumulative_reset_reduction_relative", UNKNOWN_FLOWSTAR_COMPONENT if mode == FLOWSTAR_MODE else ""),
                "missing_flowstar_component_fields": UNKNOWN_FLOWSTAR_COMPONENT if mode == FLOWSTAR_MODE else "",
                "notes": "diagnostic checkpoint containing physical time; no interpolation",
            })
    return out


def margin_watch(rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]], limit: int = 10) -> list[dict[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for mode_rows in rows_by_mode.values():
        rows.extend(row for row in mode_rows if row.get("accepted_rejected") == "accepted" and _float(row.get("target_margin_min")) is not None)
    rows = sorted(rows, key=lambda row: float(row["target_margin_min"]))[:limit]
    return [{
        "rank": i + 1,
        "mode": row.get("mode", ""),
        "run_kind": row.get("run_kind", ""),
        "segment_index": row.get("segment_index", ""),
        "t_lo": row.get("t_lo", ""),
        "t_hi": row.get("t_hi", ""),
        "h": row.get("h_accepted", ""),
        "target_margin_min": row.get("target_margin_min", ""),
        "final_segment_width_sum": row.get("final_segment_width_sum", ""),
        "tube_prefix_width_sum": row.get("tube_prefix_width_sum", ""),
        "had_prior_rejection": row.get("had_prior_rejection", ""),
        "accepted_rejected": row.get("accepted_rejected", ""),
    } for i, row in enumerate(rows)]


def common_time_stats(const_rows: Sequence[Mapping[str, Any]], range_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    common: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    max_t = min(
        _float([row for row in const_rows if row.get("accepted_rejected") == "accepted"][-1].get("t_hi")) if [row for row in const_rows if row.get("accepted_rejected") == "accepted"] else 0.0,
        _float([row for row in range_rows if row.get("accepted_rejected") == "accepted"][-1].get("t_hi")) if [row for row in range_rows if row.get("accepted_rejected") == "accepted"] else 0.0,
    )
    for c in const_rows:
        t = _float(c.get("t_hi"))
        if c.get("accepted_rejected") != "accepted" or t is None or t > (max_t or 0.0) + 1e-12:
            continue
        r = _segment_containing(range_rows, t)
        if r:
            common.append((c, r))
    worsening = 0
    best_improvement = ""
    final_improvement = ""
    for c, r in common:
        _abs, rel = _reduction(c.get("final_segment_width_sum"), r.get("final_segment_width_sum"))
        if _float(rel) is not None:
            if float(rel) < -1e-12:
                worsening += 1
            best_improvement = max(_float(best_improvement) or -math.inf, float(rel))
            final_improvement = rel
    return {"common_time_count": len(common), "width_worsening_count": worsening, "best_width_improvement": best_improvement, "final_common_width_improvement": final_improvement}


def decide(summary_by_mode: Mapping[str, Mapping[str, Any]], common_stats: Mapping[str, Any], cross_rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any], list[str]]:
    torch_summaries = [row for mode, row in summary_by_mode.items() if mode != FLOWSTAR_MODE]
    range_summary = summary_by_mode[RANGE_ADAPTIVE]
    constant_summary = summary_by_mode[CONSTANT_ADAPTIVE]
    range_on_constant = [row for row in cross_rows if row.get("replay_kind") == "range_midpoint_on_constant_schedule" and row.get("replay_status") == "accepted"]
    cross_improvement = _float(range_on_constant[-1].get("width_reduction_relative")) if range_on_constant else None
    base_metrics = {
        "cross_schedule_centering_improvement": cross_improvement if cross_improvement is not None else "",
        "common_time_width_worsening_count": common_stats.get("width_worsening_count", ""),
    }
    soundness_reasons: list[str] = []
    for row in torch_summaries:
        if int(float(row.get("raw_residual_target_violations") or 0)) > 0:
            soundness_reasons.append(f"{row.get('mode')} raw target violation")
        if int(float(row.get("sample_sanity_violations") or 0)) > 0:
            soundness_reasons.append(f"{row.get('mode')} sample sanity violation")
        margin = _float(row.get("minimum_target_margin"))
        if margin is None or margin <= 0.0:
            soundness_reasons.append(f"{row.get('mode')} non-positive/unknown target margin")
        if (_float(row.get("max_reconstruction_polynomial_abs_diff")) or 0.0) > 1e-12:
            soundness_reasons.append(f"{row.get('mode')} reconstruction polynomial diff")
        if (_float(row.get("max_reconstruction_remainder_endpoint_diff")) or 0.0) > 1e-15:
            soundness_reasons.append(f"{row.get('mode')} reconstruction remainder diff")
    if soundness_reasons:
        return "reject_due_to_soundness_or_reconstruction_failure", base_metrics, soundness_reasons

    final_ratio = _float(range_summary.get("flowstar_final_width_ratio"))
    tube_ratio = _float(range_summary.get("flowstar_tube_width_ratio"))
    range_reached = _bool(range_summary.get("reached_h10"))
    schedule_independent = cross_improvement is not None and cross_improvement >= 0.05
    no_worsening = int(common_stats.get("width_worsening_count") or 0) == 0
    if range_reached:
        controlled = (
            final_ratio is not None and final_ratio <= 5.0
            and tube_ratio is not None and tube_ratio <= 1.10
            and no_worsening
            and schedule_independent
        )
        return ("h10_reached_with_controlled_width" if controlled else "h10_reached_but_width_gap_large"), base_metrics, []

    range_t = _float(range_summary.get("reached_t")) or 0.0
    const_t = _float(constant_summary.get("reached_t")) or 0.0
    range_steps = int(float(range_summary.get("accepted_steps") or 0))
    const_steps = int(float(constant_summary.get("accepted_steps") or 0))
    width_improved = (_float(common_stats.get("final_common_width_improvement")) or 0.0) >= 0.05
    materially = (range_t >= const_t + 0.5 or (range_t > const_t and range_steps >= const_steps + 10)) and width_improved
    return ("h10_not_reached_but_materially_improved" if materially else "h10_not_reached_no_material_improvement"), base_metrics, []


def _flowstar_summary(flow_ref: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": "flowstar", "run_kind": "reference", "mode": FLOWSTAR_MODE, "right_map_center_mode": "not_applicable",
        "status": "completed", "reached_t": DEFAULT_HORIZON, "reached_h10": True,
        "accepted_steps": flow_ref.get("accepted_steps", ""), "final_segment_width_sum": flow_ref.get("final_width_sum", ""),
        "tube_width_sum": flow_ref.get("_tube_width_sum", ""), "flowstar_final_width_sum": flow_ref.get("final_width_sum", ""),
        "flowstar_tube_width_sum": flow_ref.get("_tube_width_sum", ""), "flowstar_final_width_ratio": 1.0,
        "flowstar_tube_width_ratio": 1.0, "decision": "reference_only",
        "notes": "Flowstar h10 segment-box reference; internal component fields unknown, not zero.",
    }


def finalize_summaries(summary_rows: list[dict[str, Any]], flow_ref: Mapping[str, Any], flow_h: Sequence[float], decision: str, decision_metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    flow_final = flow_ref.get("final_width_sum", "")
    flow_tube = flow_ref.get("_tube_width_sum", "")
    for row in summary_rows:
        row["flowstar_final_width_sum"] = flow_final
        row["flowstar_tube_width_sum"] = flow_tube
        if row.get("source") == "torch":
            row["flowstar_final_width_ratio"] = _ratio(row.get("final_segment_width_sum"), flow_final)
            row["flowstar_tube_width_ratio"] = _ratio(row.get("tube_width_sum"), flow_tube)
            accepted = row.get("_accepted_h", [])
            row["schedule_distance_vs_flowstar"] = schedule_distance(list(flow_h), list(accepted)) if accepted else ""
            row["decision"] = decision if row.get("mode") == RANGE_ADAPTIVE else ""
            row["common_time_width_worsening_count"] = decision_metrics.get("common_time_width_worsening_count", "")
            row["cross_schedule_centering_improvement"] = decision_metrics.get("cross_schedule_centering_improvement", "")
            row.pop("_accepted_h", None)
            row.pop("_sample_max_violation", None)
    return summary_rows


def write_report(path: Path, summary_rows: Sequence[Mapping[str, Any]], margin_rows: Sequence[Mapping[str, Any]], decision: str, reasons: Sequence[str], formatting: Sequence[Mapping[str, Any]]) -> None:
    by_mode = {row.get("mode"): row for row in summary_rows}
    lines = [
        "# h10 Right-Map Range-Midpoint Centering Audit",
        "",
        "This h10 audit keeps `right_map_center_mode=\"constant\"` as the default. h10 was run only by this opt-in experiment.",
        "",
        "## Decision",
        "",
        f"- Decision: `{decision}`.",
        f"- Reasons: `{'; '.join(reasons) if reasons else 'criteria evaluated from h10 artifacts'}`.",
        f"- Minimum target margin: `{_format(by_mode.get(RANGE_ADAPTIVE, {}).get('minimum_target_margin'))}` at step `{_format(by_mode.get(RANGE_ADAPTIVE, {}).get('minimum_target_margin_step'))}`, t `{_format(by_mode.get(RANGE_ADAPTIVE, {}).get('minimum_target_margin_time'))}`, h `{_format(by_mode.get(RANGE_ADAPTIVE, {}).get('minimum_target_margin_h'))}`.",
        f"- Immediate same-state saving max: `{_format(by_mode.get(RANGE_ADAPTIVE, {}).get('max_immediate_same_state_saving'))}`.",
        f"- Cumulative downstream saving max: `{_format(by_mode.get(RANGE_ADAPTIVE, {}).get('max_cumulative_downstream_saving'))}`.",
        f"- Common-time width worsening count: `{_format(by_mode.get(RANGE_ADAPTIVE, {}).get('common_time_width_worsening_count'))}`.",
        f"- Cross-schedule centering improvement: `{_format(by_mode.get(RANGE_ADAPTIVE, {}).get('cross_schedule_centering_improvement'))}`.",
        "",
        "## Run Summary",
        "",
        "| mode | reached_t | reached_h10 | accepted | rejected | final width | tube width | Flowstar final ratio | Flowstar tube ratio | samples | min margin |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(_format(row.get(field)) for field in ("mode", "reached_t", "reached_h10", "accepted_steps", "rejected_attempts", "final_segment_width_sum", "tube_width_sum", "flowstar_final_width_ratio", "flowstar_tube_width_ratio", "sample_sanity_violations", "minimum_target_margin")) + " |")
    lines.extend(["", "## Margin Watch", "", "| rank | mode | step | t_hi | h | margin | width | had rejection |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in margin_rows:
        lines.append("| " + " | ".join(_format(row.get(field)) for field in ("rank", "mode", "segment_index", "t_hi", "h", "target_margin_min", "final_segment_width_sum", "had_prior_rejection")) + " |")
    lines.extend(["", "## Formatting", "", "| path | physical lines | csv.reader rows | status |", "| --- | --- | --- | --- |"])
    for row in formatting:
        lines.append(f"| {_format(row.get('path'))} | {_format(row.get('physical_line_count'))} | {_format(row.get('csv_reader_row_count'))} | {_format(row.get('status'))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    horizon = float(args.horizon)
    if abs(horizon - DEFAULT_HORIZON) > 1e-12:
        raise ValueError("h10 audit horizon must be exactly 10")
    flow_ref, flow_segments, flow_h = load_flowstar_reference(args.flowstar_segments.resolve(), horizon)
    if not flow_ref:
        raise FileNotFoundError(f"missing Flowstar reference: {args.flowstar_segments}")
    flow_rows = _flowstar_rows(flow_segments)
    for row in flow_rows:
        idx = int(float(row.get("segment_index") or 0))
        row["tube_prefix_width_sum"] = _tube_from_segments(flow_rows[: idx + 1]).get("tube_width_sum", "")

    const_summary, const_rows, const_attempts = run_centering_h10(mode=CONSTANT_ADAPTIVE, run_kind="constant adaptive", right_map_center_mode="constant", horizon=horizon, wall_cap_s=float(args.wall_cap_s))
    range_summary, range_rows, range_attempts = run_centering_h10(mode=RANGE_ADAPTIVE, run_kind="range_midpoint adaptive", right_map_center_mode="range_midpoint", horizon=horizon, wall_cap_s=float(args.wall_cap_s))
    range_on_const_summary, range_on_const_rows, range_on_const_attempts = run_centering_h10(mode=RANGE_ON_CONSTANT, run_kind="range_midpoint on constant schedule", right_map_center_mode="range_midpoint", horizon=horizon, wall_cap_s=float(args.wall_cap_s), prescribed_h=_accepted_h10(const_rows))
    const_on_range_summary, const_on_range_rows, const_on_range_attempts = run_centering_h10(mode=CONSTANT_ON_RANGE, run_kind="constant on range_midpoint schedule", right_map_center_mode="constant", horizon=horizon, wall_cap_s=float(args.wall_cap_s), prescribed_h=_accepted_h10(range_rows))

    cross_rows = make_cross_schedule_rows(const_rows, range_on_const_rows, replay_kind="range_midpoint_on_constant_schedule", source_mode=CONSTANT_ADAPTIVE, replay_mode=RANGE_ON_CONSTANT)
    cross_rows.extend(make_cross_schedule_rows(range_rows, const_on_range_rows, replay_kind="constant_on_range_midpoint_schedule", source_mode=RANGE_ADAPTIVE, replay_mode=CONSTANT_ON_RANGE))
    for cross in cross_rows:
        idx = int(cross.get("step_index") or -1)
        if cross.get("replay_mode") == RANGE_ON_CONSTANT and 0 <= idx < len(range_on_const_rows):
            range_on_const_rows[idx]["cumulative_reset_reduction_relative"] = cross.get("cumulative_reset_reduction_relative", "")
        if cross.get("replay_mode") == CONSTANT_ON_RANGE and 0 <= idx < len(const_on_range_rows):
            const_on_range_rows[idx]["cumulative_reset_reduction_relative"] = cross.get("cumulative_reset_reduction_relative", "")

    rows_by_mode = {
        CONSTANT_ADAPTIVE: const_rows,
        RANGE_ADAPTIVE: range_rows,
        RANGE_ON_CONSTANT: range_on_const_rows,
        CONSTANT_ON_RANGE: const_on_range_rows,
    }
    checkpoints = make_checkpoints(rows_by_mode, flow_rows, horizon)
    margins = margin_watch(rows_by_mode)
    common_stats = common_time_stats(const_rows, range_rows)
    summaries = [_flowstar_summary(flow_ref), const_summary, range_summary, range_on_const_summary, const_on_range_summary]
    summary_by_mode = {row["mode"]: row for row in summaries}
    decision, decision_metrics, reasons = decide(summary_by_mode, common_stats, cross_rows)
    for row in summaries:
        if row.get("source") == "torch":
            row["max_cumulative_downstream_saving"] = _max_field([r for r in cross_rows if r.get("replay_mode") == RANGE_ON_CONSTANT], "cumulative_reset_reduction_relative")
    summaries = finalize_summaries(summaries, flow_ref, flow_h, decision, decision_metrics)
    all_segments = flow_rows + const_rows + range_rows + range_on_const_rows + const_on_range_rows
    all_attempts = const_attempts + range_attempts + range_on_const_attempts + const_on_range_attempts

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "h10_right_map_centering_summary.csv", SUMMARY_FIELDS, summaries)
    _write_csv(out / "h10_right_map_centering_segments.csv", SEGMENT_FIELDS, all_segments)
    _write_csv(out / "h10_right_map_centering_attempts.csv", ATTEMPT_FIELDS, all_attempts)
    _write_csv(out / "h10_right_map_centering_cross_schedule.csv", CROSS_FIELDS, cross_rows)
    _write_csv(out / "h10_right_map_centering_checkpoints.csv", CHECKPOINT_FIELDS, checkpoints)
    _write_csv(out / "h10_right_map_centering_margin_watch.csv", MARGIN_FIELDS, margins)
    (out / "h10_right_map_centering_decision.txt").write_text(decision + "\n", encoding="utf-8")
    fmt = formatting_rows(out)
    _write_csv(out / "h10_right_map_centering_formatting.csv", FORMAT_FIELDS, fmt)
    write_report(out / "h10_right_map_centering_report.md", summaries, margins, decision, reasons, fmt)
    return summaries, all_segments, all_attempts, checkpoints, decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--flowstar-segments", type=Path, default=DEFAULT_FLOWSTAR_SEGMENTS)
    parser.add_argument("--wall-cap-s", type=float, default=7200.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries, segments, attempts, checkpoints, decision = run(args)
    print(f"wrote {args.out_dir.resolve() / 'h10_right_map_centering_summary.csv'} ({len(summaries)} rows)")
    print(f"wrote {args.out_dir.resolve() / 'h10_right_map_centering_segments.csv'} ({len(segments)} rows)")
    print(f"wrote {args.out_dir.resolve() / 'h10_right_map_centering_attempts.csv'} ({len(attempts)} rows)")
    print(f"wrote {args.out_dir.resolve() / 'h10_right_map_centering_checkpoints.csv'} ({len(checkpoints)} rows)")
    print(f"decision {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
