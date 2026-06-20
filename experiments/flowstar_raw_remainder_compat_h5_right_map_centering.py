#!/usr/bin/env python3
"""h5-only audit for opt-in range-midpoint-centered right-map insertion.

This script does not run h10, does not change default solver behavior, and does
not treat sample containment as a soundness proof.  It compares the existing
raw-remainder-compatible h5 path against an opt-in right-map center mode that
moves the inserted range midpoint from the Taylor-model polynomial constant
into the normal-form center.
"""
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
from flowstar_raw_remainder_compat_experiment import (  # noqa: E402
    ORDER,
    TARGET_RADIUS,
    _format,
    _read_rows,
    van_der_pol_flowstar_expression_ode,
)
from flowstar_raw_remainder_compat_h5 import (  # noqa: E402
    DEFAULT_FLOWSTAR_SEGMENTS,
    _diag_status,
    _interval_bounds,
    _prefix_match_count,
    _tube_from_segments,
    load_flowstar_reference,
    make_samples,
)
from flowstar_raw_remainder_compat_short_horizon import (  # noqa: E402
    H_MAX,
    H_MIN,
    _advance_sample,
    _interval_violation,
    schedule_distance,
)

HORIZON_LIMIT = 5.0
DEFAULT_H5_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5"
DEFAULT_DIVERGENCE_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5_divergence"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5_right_map_centering"
FLOWSTAR_MODE = "generated_flowstar_h5_reference"
BASELINE_EXISTING_MODE = "raw_remainder_compat_flowstar_step_policy"
BASELINE_ADAPTIVE_MODE = "baseline_adaptive_constant"
BASELINE_FROZEN_MODE = "baseline_frozen_constant"
RANGE_FROZEN_MODE = "range_midpoint_frozen"
RANGE_ADAPTIVE_MODE = "range_midpoint_adaptive"
UNKNOWN_FLOWSTAR_COMPONENT = "unknown_missing_h5_reference_component_fields"

SUMMARY_FIELDS = [
    "source",
    "run_kind",
    "mode",
    "right_map_center_mode",
    "status",
    "reached_t",
    "reached_horizon",
    "accepted_steps",
    "rejected_attempts",
    "first_failure_step",
    "first_failure_time",
    "first_failure_reason",
    "min_h_used",
    "runtime_s",
    "final_width_x",
    "final_width_y",
    "final_width_sum",
    "tube_width_sum",
    "flowstar_reference_final_width_sum",
    "flowstar_last_width_ratio",
    "flowstar_reference_tube_width_sum",
    "flowstar_tube_width_ratio",
    "schedule_distance_vs_flowstar",
    "schedule_prefix_match_count",
    "sample_containment_sanity_status",
    "sample_containment_sanity_violations",
    "minimum_target_margin",
    "raw_residual_target_violations",
    "max_reconstruction_polynomial_abs_diff",
    "max_reconstruction_remainder_endpoint_diff",
    "baseline_reproduction_match",
    "baseline_reproduction_notes",
    "final_width_improvement_vs_baseline_frozen",
    "tube_width_change_vs_baseline_frozen",
    "reset_width_reduction_at_1_5",
    "reset_width_reduction_at_2_0",
    "decision",
    "notes",
]

SEGMENT_FIELDS = [
    "source",
    "run_kind",
    "mode",
    "right_map_center_mode",
    "segment_index",
    "status",
    "t_lo",
    "t_hi",
    "h",
    "prescribed_h",
    "h_delta",
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
    "validation_status",
    "validation_message",
    "target_margin_min",
    "raw_residual_target_violation",
    "raw_ctrunc_residual_width_sum",
    "flowstar_raw_remainder_compat_check_remainder_width_sum",
    "poly_diff_range_width_sum",
    "polynomial_range_width_sum",
    "candidate_segment_width_sum",
    "old_right_map_range_width_sum",
    "normal_right_map_range_width_sum",
    "inserted_range_lo_x",
    "inserted_range_hi_x",
    "inserted_range_width_x",
    "inserted_range_lo_y",
    "inserted_range_hi_y",
    "inserted_range_width_y",
    "inserted_range_width_sum",
    "centered_inserted_range_lo_x",
    "centered_inserted_range_hi_x",
    "centered_inserted_range_width_x",
    "centered_inserted_range_lo_y",
    "centered_inserted_range_hi_y",
    "centered_inserted_range_width_y",
    "centered_inserted_range_width_sum",
    "inserted_range_midpoint_shift_x",
    "inserted_range_midpoint_shift_y",
    "inserted_range_midpoint_shift_abs_sum",
    "inserted_range_asymmetry_x",
    "inserted_range_asymmetry_y",
    "inserted_range_asymmetry_sum",
    "baseline_scale_x",
    "baseline_scale_y",
    "baseline_scale_sum",
    "centered_scale_x",
    "centered_scale_y",
    "centered_scale_sum",
    "baseline_reset_width_x",
    "baseline_reset_width_y",
    "baseline_reset_width_sum",
    "centered_reset_width_x",
    "centered_reset_width_y",
    "centered_reset_width_sum",
    "scale_reduction_absolute_x",
    "scale_reduction_absolute_y",
    "scale_reduction_absolute_sum",
    "scale_reduction_relative_x",
    "scale_reduction_relative_y",
    "scale_reduction_relative_sum",
    "reconstruction_polynomial_max_abs_diff_x",
    "reconstruction_polynomial_max_abs_diff_y",
    "reconstruction_polynomial_max_abs_diff",
    "reconstruction_remainder_lo_diff_x",
    "reconstruction_remainder_hi_diff_x",
    "reconstruction_remainder_lo_diff_y",
    "reconstruction_remainder_hi_diff_y",
    "reconstruction_remainder_lo_diff",
    "reconstruction_remainder_hi_diff",
    "scale_x",
    "scale_y",
    "center_x",
    "center_y",
    "missing_flowstar_component_fields",
    "message",
]

FROZEN_FIELDS = [
    "step_index",
    "prescribed_h",
    "constant_status",
    "range_midpoint_status",
    "constant_h",
    "range_midpoint_h",
    "constant_h_delta",
    "range_midpoint_h_delta",
    "h_sequence_modified",
    "constant_t_hi",
    "range_midpoint_t_hi",
    "constant_width_sum",
    "range_midpoint_width_sum",
    "width_reduction_absolute",
    "width_reduction_relative",
    "constant_reset_width_sum",
    "range_midpoint_reset_width_sum",
    "reset_width_reduction_absolute",
    "reset_width_reduction_relative",
    "constant_polynomial_range_width_sum",
    "range_midpoint_polynomial_range_width_sum",
    "constant_right_map_range_width_sum",
    "range_midpoint_right_map_range_width_sum",
    "constant_target_margin",
    "range_midpoint_target_margin",
    "range_midpoint_validation_failure_recorded",
    "notes",
]

CROSSING_FIELDS = [
    "comparison_kind",
    "event_name",
    "threshold",
    "checkpoint_t",
    "source",
    "run_kind",
    "mode",
    "right_map_center_mode",
    "segment_index",
    "status",
    "t_lo",
    "t_hi",
    "h",
    "width_sum",
    "tube_prefix_width_sum",
    "baseline_reset_width_sum",
    "centered_reset_width_sum",
    "scale_reduction_relative_sum",
    "polynomial_range_width_sum",
    "right_map_range_width_sum",
    "target_margin_min",
    "missing_flowstar_component_fields",
    "notes",
]

LINE_COUNT_FIELDS = ["path", "physical_line_count", "csv_reader_row_count", "status"]


def _float(value: Any) -> float | None:
    if value in (None, "", "unknown"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "validated", "passed"}


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in fieldnames})


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle))


def physical_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def output_formatting_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*.csv")):
        if path.name == "h5_right_map_centering_formatting.csv":
            continue
        physical = physical_line_count(path)
        parsed = csv_row_count(path)
        rows.append(
            {
                "path": _display_path(path),
                "physical_line_count": physical,
                "csv_reader_row_count": parsed,
                "status": "ok" if physical == parsed else "mismatch",
            }
        )
    for path in sorted(out_dir.glob("*.md")) + sorted(out_dir.glob("*.txt")):
        rows.append(
            {
                "path": _display_path(path),
                "physical_line_count": physical_line_count(path),
                "csv_reader_row_count": "",
                "status": "ok",
            }
        )
    return rows


def _first_present(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return ""


def _latest_validation_row(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in diagnostics if row.get("validation_status") not in (None, "")]
    return rows[-1] if rows else {}


def _target_margin(row: Mapping[str, Any]) -> float | str:
    margins: list[float] = []
    for dim in ("x", "y"):
        target_lo = _float(row.get(f"target_remainder_before_ctrunc_lo_{dim}"))
        target_hi = _float(row.get(f"target_remainder_before_ctrunc_hi_{dim}"))
        check_lo = _float(
            _first_present(
                row,
                f"flowstar_raw_remainder_compat_check_remainder_lo_{dim}",
                f"raw_remainder_after_poly_diff_lo_{dim}",
                f"tmp_remainder_lo_{dim}",
                f"residual_lo_{dim}",
            )
        )
        check_hi = _float(
            _first_present(
                row,
                f"flowstar_raw_remainder_compat_check_remainder_hi_{dim}",
                f"raw_remainder_after_poly_diff_hi_{dim}",
                f"tmp_remainder_hi_{dim}",
                f"residual_hi_{dim}",
            )
        )
        if None in (target_lo, target_hi, check_lo, check_hi):
            continue
        assert target_lo is not None and target_hi is not None and check_lo is not None and check_hi is not None
        margins.append(min(target_hi - check_hi, check_lo - target_lo))
    return min(margins) if margins else ""


def _raw_target_violation(row: Mapping[str, Any]) -> bool:
    status = str(row.get("validation_status", "")).lower()
    subset = str(row.get("subset_flowstar_raw_remainder_compat", row.get("subset_result", ""))).lower()
    margin = _target_margin(row)
    if status == "failed" and "target" in str(row.get("validation_message", "")).lower():
        return True
    if subset == "false":
        return True
    return _float(margin) is not None and float(margin) < -1e-15


def _max_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else ""


def _min_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    values = [_float(row.get(field)) for row in rows]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else ""


def _ratio(num: Any, den: Any) -> float | str:
    n = _float(num)
    d = _float(den)
    if n is None or d is None or abs(d) <= 0.0:
        return ""
    return n / d


def _reduction(before: Any, after: Any) -> tuple[float | str, float | str]:
    b = _float(before)
    a = _float(after)
    if b is None or a is None:
        return "", ""
    absolute = b - a
    relative = absolute / b if abs(b) > 0.0 else ""
    return absolute, relative


def _validated_rows(rows: Sequence[Mapping[str, Any]], mode: str | None = None) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows if row.get("status") == "validated"]
    if mode is not None:
        out = [row for row in out if row.get("mode") == mode]
    return sorted(out, key=lambda row: int(float(row.get("segment_index") or 0)))


def _accepted_h(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    vals: list[float] = []
    for row in _validated_rows(rows):
        h = _float(row.get("h"))
        if h is not None:
            vals.append(h)
    return vals


def _make_flowstar_segment_rows(flow_segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in flow_segments:
        out = {field: "" for field in SEGMENT_FIELDS}
        out.update(
            {
                "source": "flowstar",
                "run_kind": "reference",
                "mode": FLOWSTAR_MODE,
                "right_map_center_mode": "not_applicable",
                "segment_index": row.get("segment_index", len(rows)),
                "status": row.get("status", "validated"),
                "t_lo": row.get("t_lo", ""),
                "t_hi": row.get("t_hi", ""),
                "h": row.get("h", ""),
                "x_lo": row.get("x_lo", ""),
                "x_hi": row.get("x_hi", ""),
                "y_lo": row.get("y_lo", ""),
                "y_hi": row.get("y_hi", ""),
                "width_x": row.get("width_x", ""),
                "width_y": row.get("width_y", ""),
                "width_sum": row.get("width_sum", ""),
                "box_semantics": "flowstar_gnuplot_segment_box",
                "missing_flowstar_component_fields": UNKNOWN_FLOWSTAR_COMPONENT,
                "message": "Flowstar h5 reference exposes segment boxes, not PyTorch right-map component diagnostics; missing component fields are unknown, not zero.",
            }
        )
        for field in SEGMENT_FIELDS:
            if field.endswith("_width_sum") and field not in {"width_sum"}:
                out[field] = UNKNOWN_FLOWSTAR_COMPONENT
        rows.append(out)
    return rows


def _copy_stat_fields(stats: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in SEGMENT_FIELDS:
        if field in stats:
            out[field] = stats.get(field, "")
        elif field in validation:
            out[field] = validation.get(field, "")
    return out


def run_centering_h5(
    *,
    mode: str,
    run_kind: str,
    right_map_center_mode: str,
    horizon: float,
    wall_cap_s: float,
    prescribed_h: Sequence[float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if horizon > HORIZON_LIMIT + 1e-12:
        raise ValueError("h5 right-map-centering audit is capped at T=5.0; refusing to run h10")
    current: TMVector | list[Interval] = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state: FlowstarNormalFlowpipeState | None = None
    samples = make_samples()
    h_next = H_MAX
    t = 0.0
    start = time.perf_counter()
    segment_rows: list[dict[str, Any]] = []
    rejected_attempts = 0
    sample_violations = 0
    sample_max_violation = 0.0
    status = "completed"
    message = "validated to requested h5 horizon"
    first_failure_step: int | str = ""
    first_failure_time: float | str = ""
    first_failure_reason = ""
    prescribed = list(prescribed_h or [])
    prescribed_index = 0

    while t < horizon - 1e-12:
        if time.perf_counter() - start >= wall_cap_s:
            status = "timeout"
            message = f"wall-time cap reached before segment {len(segment_rows)}"
            break
        remaining = horizon - t
        if prescribed_h is None:
            h_try = min(h_next, H_MAX, remaining)
            prescribed_value: float | str = ""
        else:
            if prescribed_index >= len(prescribed):
                status = "failed"
                message = "prescribed h sequence ended before horizon"
                first_failure_step = len(segment_rows)
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
                first_failure_step = len(segment_rows)
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
            diagnostics_context={"mode": mode, "segment_index": len(segment_rows), "t_before": t, "run_kind": run_kind},
        )
        rejected_attempts += sum(1 for row in diagnostics if _diag_status(row) == "rejected")
        validation = _latest_validation_row(diagnostics)
        stats = dict(getattr(seg, "flowstar_normal_stats", None) or {})
        try:
            segment_box = seg.tm.range_box()
            final_box = seg.final_tm.range_box()
            x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = _interval_bounds(segment_box)
            finite = intervals_are_finite(segment_box) and intervals_are_finite(final_box)
        except Exception as exc:
            x_lo = x_hi = y_lo = y_hi = width_x = width_y = width_sum = ""
            final_box = []
            finite = False
            message = f"range evaluation failed: {exc}"
        row_status = "validated" if seg.status == "validated" and finite and seg.reset_tm is not None else "failed"
        t_hi = t + float(seg.h)
        target_margin = _target_margin(validation)
        raw_violation = _raw_target_violation(validation)
        h_delta = ""
        if prescribed_value != "":
            h_delta = float(seg.h) - float(prescribed_value)
        row = {
            **_copy_stat_fields(stats, validation),
            "source": "torch",
            "run_kind": run_kind,
            "mode": mode,
            "right_map_center_mode": right_map_center_mode,
            "segment_index": len(segment_rows),
            "status": row_status,
            "t_lo": t,
            "t_hi": t_hi,
            "h": float(seg.h),
            "prescribed_h": prescribed_value,
            "h_delta": h_delta,
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
            "validation_status": validation.get("validation_status", ""),
            "validation_message": validation.get("validation_message", ""),
            "target_margin_min": target_margin,
            "raw_residual_target_violation": raw_violation,
            "message": seg.message or message,
        }
        segment_rows.append(row)
        if row_status != "validated":
            status = "failed"
            message = seg.message or str(validation.get("validation_message", "")) or "validation failed"
            first_failure_step = len(segment_rows) - 1
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

    if status == "completed" and t < horizon - 1e-12:
        status = "stopped"
        message = "stopped before h5"
    validated = _validated_rows(segment_rows)
    tube = _tube_from_segments(validated)
    last = validated[-1] if validated else {}
    h_vals = _accepted_h(segment_rows)
    sample_status = "passed" if sample_violations == 0 and validated else "failed"
    summary = {
        "source": "torch",
        "run_kind": run_kind,
        "mode": mode,
        "right_map_center_mode": right_map_center_mode,
        "status": status,
        "reached_t": t,
        "reached_horizon": bool(status == "completed" and t >= horizon - 1e-9),
        "accepted_steps": len(validated),
        "rejected_attempts": rejected_attempts,
        "first_failure_step": first_failure_step,
        "first_failure_time": first_failure_time,
        "first_failure_reason": first_failure_reason,
        "min_h_used": min(h_vals) if h_vals else "",
        "runtime_s": time.perf_counter() - start,
        "final_width_x": last.get("width_x", ""),
        "final_width_y": last.get("width_y", ""),
        "final_width_sum": last.get("width_sum", ""),
        "tube_width_sum": tube.get("tube_width_sum", ""),
        "sample_containment_sanity_status": sample_status,
        "sample_containment_sanity_violations": sample_violations,
        "minimum_target_margin": _min_field(validated, "target_margin_min"),
        "raw_residual_target_violations": sum(1 for row in segment_rows if _bool(row.get("raw_residual_target_violation"))),
        "max_reconstruction_polynomial_abs_diff": _max_field(validated, "reconstruction_polynomial_max_abs_diff"),
        "max_reconstruction_remainder_endpoint_diff": max(
            _float(_max_field(validated, "reconstruction_remainder_lo_diff")) or 0.0,
            _float(_max_field(validated, "reconstruction_remainder_hi_diff")) or 0.0,
        ) if validated else "",
        "notes": message + "; sample containment is a sanity check only, not a soundness proof",
        "_accepted_h": h_vals,
        "_sample_max_violation": sample_max_violation,
    }
    sample_row = {
        "source": "torch",
        "run_kind": run_kind,
        "mode": mode,
        "right_map_center_mode": right_map_center_mode,
        "sample_count": len(samples),
        "sample_containment_sanity_status": sample_status,
        "sample_containment_sanity_violations": sample_violations,
        "sample_max_violation": sample_max_violation,
        "notes": "corners, center, and deterministic random samples checked against final-time endpoint boxes after each accepted segment; sanity check only",
    }
    return summary, segment_rows, sample_row


def baseline_reproduction_check(summary: Mapping[str, Any], segments: Sequence[Mapping[str, Any]], h5_dir: Path) -> tuple[bool, str]:
    summary_path = h5_dir / "h5_summary.csv"
    segments_path = h5_dir / "h5_segments.csv"
    if not summary_path.exists() or not segments_path.exists():
        return False, "existing h5 artifacts missing"
    existing_rows = [row for row in _read_rows(summary_path) if row.get("mode") == BASELINE_EXISTING_MODE]
    existing_segments = _validated_rows(_read_rows(segments_path), BASELINE_EXISTING_MODE)
    if not existing_rows or not existing_segments:
        return False, "existing raw_remainder_compat_flowstar_step_policy rows missing"
    existing = existing_rows[0]
    checks = []
    checks.append(str(existing.get("status")) == str(summary.get("status")))
    checks.append(int(float(existing.get("accepted_steps") or -1)) == int(float(summary.get("accepted_steps") or -2)))
    checks.append(abs((_float(existing.get("final_width_sum")) or math.nan) - (_float(summary.get("final_width_sum")) or math.nan)) <= 1e-9)
    existing_h = [_float(row.get("h")) for row in existing_segments]
    new_h = [_float(row.get("h")) for row in _validated_rows(segments)]
    h_match = len(existing_h) == len(new_h) and all(
        a is not None and b is not None and abs(float(a) - float(b)) <= 1e-10
        for a, b in zip(existing_h, new_h)
    )
    checks.append(h_match)
    notes = (
        f"existing_status={existing.get('status')}; reproduced_status={summary.get('status')}; "
        f"existing_steps={existing.get('accepted_steps')}; reproduced_steps={summary.get('accepted_steps')}; "
        f"existing_final_width_sum={existing.get('final_width_sum')}; reproduced_final_width_sum={summary.get('final_width_sum')}; "
        f"h_sequence_match={h_match}"
    )
    return all(checks), notes


def _flowstar_summary_row(flow_ref: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": "flowstar",
        "run_kind": "reference",
        "mode": FLOWSTAR_MODE,
        "right_map_center_mode": "not_applicable",
        "status": flow_ref.get("status", "completed"),
        "reached_t": flow_ref.get("reached_t", HORIZON_LIMIT),
        "reached_horizon": flow_ref.get("completed_h5", True),
        "accepted_steps": flow_ref.get("accepted_steps", ""),
        "final_width_x": flow_ref.get("final_width_x", ""),
        "final_width_y": flow_ref.get("final_width_y", ""),
        "final_width_sum": flow_ref.get("final_width_sum", ""),
        "tube_width_sum": flow_ref.get("_tube_width_sum", ""),
        "flowstar_reference_final_width_sum": flow_ref.get("final_width_sum", ""),
        "flowstar_last_width_ratio": 1.0,
        "flowstar_reference_tube_width_sum": flow_ref.get("_tube_width_sum", ""),
        "flowstar_tube_width_ratio": 1.0,
        "schedule_distance_vs_flowstar": 0.0,
        "schedule_prefix_match_count": flow_ref.get("accepted_steps", ""),
        "sample_containment_sanity_status": "not_applicable",
        "baseline_reproduction_match": "reference",
        "decision": "reference_only",
        "notes": "Flowstar GNUPLOT segment-box reference restricted to h5; component diagnostics unavailable and reported as unknown, not zero.",
    }


def finalize_summary_rows(
    summary_rows: list[dict[str, Any]],
    flow_ref: Mapping[str, Any],
    flow_h: Sequence[float],
    baseline_frozen: Mapping[str, Any],
    range_frozen: Mapping[str, Any],
    decision: str,
    decision_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    flow_last = flow_ref.get("final_width_sum", "")
    flow_tube = flow_ref.get("_tube_width_sum", "")
    baseline_final = baseline_frozen.get("final_width_sum", "")
    baseline_tube = baseline_frozen.get("tube_width_sum", "")
    for row in summary_rows:
        accepted = list(row.get("_accepted_h", []))
        if row.get("source") == "torch" and accepted:
            row["schedule_distance_vs_flowstar"] = schedule_distance(list(flow_h), accepted) if flow_h else ""
            row["schedule_prefix_match_count"] = _prefix_match_count(list(flow_h), accepted) if flow_h else ""
        row["flowstar_reference_final_width_sum"] = flow_last
        row["flowstar_last_width_ratio"] = _ratio(row.get("final_width_sum"), flow_last) if row.get("source") == "torch" else row.get("flowstar_last_width_ratio", "")
        row["flowstar_reference_tube_width_sum"] = flow_tube
        row["flowstar_tube_width_ratio"] = _ratio(row.get("tube_width_sum"), flow_tube) if row.get("source") == "torch" else row.get("flowstar_tube_width_ratio", "")
        if row.get("mode") in {RANGE_FROZEN_MODE, RANGE_ADAPTIVE_MODE}:
            row["decision"] = decision
        elif row.get("mode") == BASELINE_FROZEN_MODE:
            row["decision"] = "baseline_comparator"
        elif row.get("mode") == BASELINE_ADAPTIVE_MODE:
            row["decision"] = "baseline_reproduction"
        if row.get("mode") == RANGE_FROZEN_MODE:
            row["final_width_improvement_vs_baseline_frozen"] = decision_metrics.get("final_width_improvement", "")
            row["tube_width_change_vs_baseline_frozen"] = decision_metrics.get("tube_width_change", "")
            row["reset_width_reduction_at_1_5"] = decision_metrics.get("reset_reduction_at_1_5", "")
            row["reset_width_reduction_at_2_0"] = decision_metrics.get("reset_reduction_at_2_0", "")
        elif row.get("source") == "torch" and baseline_final not in (None, ""):
            absolute, relative = _reduction(baseline_final, row.get("final_width_sum"))
            row["final_width_improvement_vs_baseline_frozen"] = relative
            tube_change = _ratio(row.get("tube_width_sum"), baseline_tube)
            row["tube_width_change_vs_baseline_frozen"] = (tube_change - 1.0) if isinstance(tube_change, float) else ""
        row.pop("_accepted_h", None)
        row.pop("_sample_max_violation", None)
    return summary_rows


def make_frozen_schedule_rows(constant_rows: Sequence[Mapping[str, Any]], range_rows: Sequence[Mapping[str, Any]], prescribed_h: Sequence[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_len = max(len(constant_rows), len(range_rows), len(prescribed_h))
    for i in range(max_len):
        const = constant_rows[i] if i < len(constant_rows) else {}
        rng = range_rows[i] if i < len(range_rows) else {}
        prescribed = prescribed_h[i] if i < len(prescribed_h) else ""
        width_abs, width_rel = _reduction(const.get("width_sum"), rng.get("width_sum"))
        reset_abs, reset_rel = _reduction(const.get("centered_reset_width_sum"), rng.get("centered_reset_width_sum"))
        const_delta = _float(const.get("h_delta"))
        range_delta = _float(rng.get("h_delta"))
        modified = bool(
            (const_delta is not None and abs(const_delta) > 1e-10)
            or (range_delta is not None and abs(range_delta) > 1e-10)
        )
        rows.append(
            {
                "step_index": i,
                "prescribed_h": prescribed,
                "constant_status": const.get("status", "missing"),
                "range_midpoint_status": rng.get("status", "missing"),
                "constant_h": const.get("h", ""),
                "range_midpoint_h": rng.get("h", ""),
                "constant_h_delta": const.get("h_delta", ""),
                "range_midpoint_h_delta": rng.get("h_delta", ""),
                "h_sequence_modified": modified,
                "constant_t_hi": const.get("t_hi", ""),
                "range_midpoint_t_hi": rng.get("t_hi", ""),
                "constant_width_sum": const.get("width_sum", ""),
                "range_midpoint_width_sum": rng.get("width_sum", ""),
                "width_reduction_absolute": width_abs,
                "width_reduction_relative": width_rel,
                "constant_reset_width_sum": const.get("centered_reset_width_sum", ""),
                "range_midpoint_reset_width_sum": rng.get("centered_reset_width_sum", ""),
                "reset_width_reduction_absolute": reset_abs,
                "reset_width_reduction_relative": reset_rel,
                "constant_polynomial_range_width_sum": const.get("polynomial_range_width_sum", ""),
                "range_midpoint_polynomial_range_width_sum": rng.get("polynomial_range_width_sum", ""),
                "constant_right_map_range_width_sum": const.get("inserted_range_width_sum", const.get("old_right_map_range_width_sum", "")),
                "range_midpoint_right_map_range_width_sum": rng.get("inserted_range_width_sum", rng.get("old_right_map_range_width_sum", "")),
                "constant_target_margin": const.get("target_margin_min", ""),
                "range_midpoint_target_margin": rng.get("target_margin_min", ""),
                "range_midpoint_validation_failure_recorded": bool(rng and rng.get("status") != "validated"),
                "notes": "prescribed h replay; h_min=h_max=prescribed_h, so adaptive shrink cannot silently change the accepted h",
            }
        )
    return rows


def load_checkpoints(divergence_dir: Path, horizon: float) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    schedule_path = divergence_dir / "h5_schedule_divergence.csv"
    width_path = divergence_dir / "h5_width_growth.csv"
    if schedule_path.exists():
        for row in _read_rows(schedule_path):
            if str(row.get("diverged", "")).lower() == "true":
                checkpoints.append(
                    {
                        "event_name": "first_schedule_divergence",
                        "threshold": "",
                        "checkpoint_t": _float(row.get("flowstar_t_before")) or _float(row.get("compat_t_before")) or 0.07631375,
                    }
                )
                break
    if not checkpoints:
        checkpoints.append({"event_name": "first_schedule_divergence", "threshold": "", "checkpoint_t": 0.07631375})
    width_rows = _read_rows(width_path) if width_path.exists() else []
    for threshold in (1.1, 1.5, 2.0):
        crossing_index: int | None = None
        for i, row in enumerate(width_rows):
            ratio = _float(row.get("compat_over_flowstar_ratio"))
            if ratio is not None and ratio > threshold:
                crossing_index = i
                break
        if crossing_index is None:
            continue
        pre = width_rows[max(0, crossing_index - 1)]
        at = width_rows[crossing_index]
        checkpoints.append({"event_name": f"pre_width_ratio_gt_{str(threshold).replace('.', '_')}", "threshold": threshold, "checkpoint_t": _float(pre.get("t"))})
        checkpoints.append({"event_name": f"at_width_ratio_gt_{str(threshold).replace('.', '_')}", "threshold": threshold, "checkpoint_t": _float(at.get("t"))})
    checkpoints.append({"event_name": "final_h5_window", "threshold": "", "checkpoint_t": horizon})
    return [row for row in checkpoints if _float(row.get("checkpoint_t")) is not None]


def _segment_containing(rows: Sequence[Mapping[str, Any]], checkpoint_t: float) -> dict[str, Any]:
    validated = _validated_rows(rows)
    for row in validated:
        lo = _float(row.get("t_lo"))
        hi = _float(row.get("t_hi"))
        if lo is None or hi is None:
            continue
        if lo - 1e-12 <= checkpoint_t <= hi + 1e-12:
            return row
    return validated[-1] if validated and checkpoint_t >= (_float(validated[-1].get("t_hi")) or 0.0) - 1e-12 else {}


def _tube_prefix_width(rows: Sequence[Mapping[str, Any]], segment_index: int) -> float | str:
    prefix = [row for row in _validated_rows(rows) if int(float(row.get("segment_index") or 0)) <= segment_index]
    return _tube_from_segments(prefix).get("tube_width_sum", "") if prefix else ""


def make_crossing_rows(checkpoints: Sequence[Mapping[str, Any]], by_mode_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_t = float(_float(checkpoint.get("checkpoint_t")) or 0.0)
        for mode, mode_rows in by_mode_rows.items():
            segment = _segment_containing(mode_rows, checkpoint_t)
            if not segment:
                rows.append(
                    {
                        "comparison_kind": "physical_time_aligned",
                        "event_name": checkpoint.get("event_name", ""),
                        "threshold": checkpoint.get("threshold", ""),
                        "checkpoint_t": checkpoint_t,
                        "mode": mode,
                        "status": "missing",
                        "notes": "no containing segment; no interpolation used",
                    }
                )
                continue
            index = int(float(segment.get("segment_index") or 0))
            flowstar_missing = UNKNOWN_FLOWSTAR_COMPONENT if segment.get("source") == "flowstar" else ""
            rows.append(
                {
                    "comparison_kind": "physical_time_aligned",
                    "event_name": checkpoint.get("event_name", ""),
                    "threshold": checkpoint.get("threshold", ""),
                    "checkpoint_t": checkpoint_t,
                    "source": segment.get("source", ""),
                    "run_kind": segment.get("run_kind", ""),
                    "mode": mode,
                    "right_map_center_mode": segment.get("right_map_center_mode", ""),
                    "segment_index": index,
                    "status": segment.get("status", ""),
                    "t_lo": segment.get("t_lo", ""),
                    "t_hi": segment.get("t_hi", ""),
                    "h": segment.get("h", ""),
                    "width_sum": segment.get("width_sum", ""),
                    "tube_prefix_width_sum": _tube_prefix_width(mode_rows, index),
                    "baseline_reset_width_sum": segment.get("baseline_reset_width_sum", flowstar_missing),
                    "centered_reset_width_sum": segment.get("centered_reset_width_sum", flowstar_missing),
                    "scale_reduction_relative_sum": segment.get("scale_reduction_relative_sum", flowstar_missing),
                    "polynomial_range_width_sum": segment.get("polynomial_range_width_sum", flowstar_missing),
                    "right_map_range_width_sum": segment.get("inserted_range_width_sum", segment.get("old_right_map_range_width_sum", flowstar_missing)),
                    "target_margin_min": segment.get("target_margin_min", flowstar_missing),
                    "missing_flowstar_component_fields": flowstar_missing,
                    "notes": "contains checkpoint physical time; no box interpolation is performed",
                }
            )
    return rows


def _event_mode_row(crossing_rows: Sequence[Mapping[str, Any]], event_name: str, mode: str) -> Mapping[str, Any]:
    for row in crossing_rows:
        if row.get("event_name") == event_name and row.get("mode") == mode:
            return row
    return {}


def decide(
    *,
    baseline_frozen: Mapping[str, Any],
    range_frozen: Mapping[str, Any],
    range_adaptive: Mapping[str, Any],
    crossing_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any], list[str]]:
    final_abs, final_improvement = _reduction(baseline_frozen.get("final_width_sum"), range_frozen.get("final_width_sum"))
    tube_ratio = _ratio(range_frozen.get("tube_width_sum"), baseline_frozen.get("tube_width_sum"))
    tube_change = (tube_ratio - 1.0) if isinstance(tube_ratio, float) else ""

    def reset_reduction(event: str) -> float | str:
        const = _event_mode_row(crossing_rows, event, BASELINE_FROZEN_MODE)
        rng = _event_mode_row(crossing_rows, event, RANGE_FROZEN_MODE)
        _abs, rel = _reduction(const.get("centered_reset_width_sum"), rng.get("centered_reset_width_sum"))
        return rel

    reset_1_5 = reset_reduction("at_width_ratio_gt_1_5")
    reset_2_0 = reset_reduction("at_width_ratio_gt_2_0")
    frozen_complete = _bool(range_frozen.get("reached_horizon"))
    adaptive_complete = _bool(range_adaptive.get("reached_horizon"))
    min_margin = min(
        value
        for value in [
            _float(range_frozen.get("minimum_target_margin")),
            _float(range_adaptive.get("minimum_target_margin")),
        ]
        if value is not None
    ) if any(_float(row.get("minimum_target_margin")) is not None for row in (range_frozen, range_adaptive)) else None
    raw_violations = int(float(range_frozen.get("raw_residual_target_violations") or 0)) + int(float(range_adaptive.get("raw_residual_target_violations") or 0))
    sample_violations = int(float(range_adaptive.get("sample_containment_sanity_violations") or 0))
    max_poly = max(_float(range_frozen.get("max_reconstruction_polynomial_abs_diff")) or 0.0, _float(range_adaptive.get("max_reconstruction_polynomial_abs_diff")) or 0.0)
    max_rem = max(_float(range_frozen.get("max_reconstruction_remainder_endpoint_diff")) or 0.0, _float(range_adaptive.get("max_reconstruction_remainder_endpoint_diff")) or 0.0)
    reconstruction_pass = max_poly <= 1e-12 and max_rem <= 1e-15

    reasons: list[str] = []
    if not frozen_complete:
        reasons.append("frozen range_midpoint replay did not reach h5")
    if not reconstruction_pass:
        reasons.append("affine reconstruction diagnostics are nonzero")
    if min_margin is None or min_margin <= 0.0:
        reasons.append("minimum target margin is non-positive or unknown")
    if raw_violations:
        reasons.append("raw residual target violation recorded")
    if not adaptive_complete:
        reasons.append("adaptive range_midpoint did not reach h5")
    if sample_violations:
        reasons.append("sample-containment sanity violation recorded")
    if _float(final_improvement) is None or float(final_improvement) <= 0.0:
        reasons.append("downstream frozen final width has no measurable improvement")

    promote = (
        frozen_complete
        and adaptive_complete
        and reconstruction_pass
        and min_margin is not None
        and min_margin > 0.0
        and raw_violations == 0
        and sample_violations == 0
        and _float(reset_1_5) is not None
        and float(reset_1_5) >= 0.10
        and _float(reset_2_0) is not None
        and float(reset_2_0) >= 0.10
        and _float(final_improvement) is not None
        and float(final_improvement) >= 0.05
        and _float(tube_change) is not None
        and float(tube_change) <= 0.005
    )
    if promote:
        decision = "promote_range_midpoint_to_h10_candidate"
    elif (
        frozen_complete
        and adaptive_complete
        and reconstruction_pass
        and min_margin is not None
        and min_margin > 0.0
        and raw_violations == 0
        and _float(reset_1_5) is not None
        and _float(reset_2_0) is not None
        and (float(reset_1_5) > 0.0 or float(reset_2_0) > 0.0)
        and (_float(final_improvement) is None or float(final_improvement) < 0.05)
    ):
        decision = "keep_range_midpoint_as_diagnostic_only"
    else:
        decision = "reject_range_midpoint_centering"

    metrics = {
        "final_width_improvement": final_improvement,
        "final_width_improvement_absolute": final_abs,
        "tube_width_change": tube_change,
        "reset_reduction_at_1_5": reset_1_5,
        "reset_reduction_at_2_0": reset_2_0,
        "minimum_target_margin": min_margin if min_margin is not None else "",
        "raw_residual_target_violations": raw_violations,
        "sample_containment_sanity_violations": sample_violations,
        "reconstruction_pass": reconstruction_pass,
        "max_reconstruction_polynomial_abs_diff": max_poly,
        "max_reconstruction_remainder_endpoint_diff": max_rem,
    }
    return decision, metrics, reasons


def write_report(
    path: Path,
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    crossing_rows: Sequence[Mapping[str, Any]],
    decision: str,
    decision_metrics: Mapping[str, Any],
    decision_reasons: Sequence[str],
    formatting_rows: Sequence[Mapping[str, Any]],
) -> None:
    by_mode = {str(row.get("mode")): row for row in summary_rows}
    baseline = by_mode.get(BASELINE_FROZEN_MODE, {})
    adaptive_baseline = by_mode.get(BASELINE_ADAPTIVE_MODE, {})
    frozen = by_mode.get(RANGE_FROZEN_MODE, {})
    adaptive = by_mode.get(RANGE_ADAPTIVE_MODE, {})

    def event_pair(event: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        return _event_mode_row(crossing_rows, event, BASELINE_FROZEN_MODE), _event_mode_row(crossing_rows, event, RANGE_FROZEN_MODE)

    first_const, first_range = event_pair("at_width_ratio_gt_1_1")
    poly_const, poly_range = event_pair("at_width_ratio_gt_1_5")
    right_const, right_range = event_pair("at_width_ratio_gt_2_0")

    lines = [
        "# Flowstar Raw Remainder Compat h5 Right-Map Centering Audit",
        "",
        "This is an h5-only opt-in mechanism audit. No h10 was run. The default solver behavior remains `right_map_center_mode=\"constant\"`, and sample containment is reported only as a sanity check.",
        "",
        "## Decision",
        "",
        f"- Decision: `{decision}`.",
        f"- Minimum target margin across range-midpoint h5 runs: `{_format(decision_metrics.get('minimum_target_margin'))}`.",
        f"- Frozen final segment width improvement: `{_format(decision_metrics.get('final_width_improvement'))}`.",
        f"- Frozen tube width change: `{_format(decision_metrics.get('tube_width_change'))}`.",
        f"- Reset width reduction at >1.5: `{_format(decision_metrics.get('reset_reduction_at_1_5'))}`.",
        f"- Reset width reduction at >2.0: `{_format(decision_metrics.get('reset_reduction_at_2_0'))}`.",
        f"- Reconstruction pass: `{_format(decision_metrics.get('reconstruction_pass'))}`; max polynomial diff `{_format(decision_metrics.get('max_reconstruction_polynomial_abs_diff'))}`, max remainder endpoint diff `{_format(decision_metrics.get('max_reconstruction_remainder_endpoint_diff'))}`.",
        f"- Raw residual target violations: `{_format(decision_metrics.get('raw_residual_target_violations'))}`.",
        f"- Sample-containment sanity violations: `{_format(decision_metrics.get('sample_containment_sanity_violations'))}`.",
        "",
        "## Reasons",
        "",
    ]
    lines.extend([f"- {reason}." for reason in decision_reasons] or ["- Promote criteria and rejection criteria evaluated from frozen/adaptive h5 rows."])
    lines.extend(
        [
            "",
            "## Key Comparisons",
            "",
            f"- Baseline reproduction from existing h5 artifact: `{_format(adaptive_baseline.get('baseline_reproduction_match'))}`; {_format(adaptive_baseline.get('baseline_reproduction_notes'))}.",
            f"- Frozen schedule complete: `{_format(frozen.get('reached_horizon'))}`; first failure step `{_format(frozen.get('first_failure_step'))}`, reason `{_format(frozen.get('first_failure_reason'))}`.",
            f"- Adaptive range_midpoint h5 complete: `{_format(adaptive.get('reached_horizon'))}`; accepted steps `{_format(adaptive.get('accepted_steps'))}`, rejected attempts `{_format(adaptive.get('rejected_attempts'))}`.",
            f"- Baseline frozen final width sum `{_format(baseline.get('final_width_sum'))}` vs range_midpoint frozen `{_format(frozen.get('final_width_sum'))}`.",
            f"- Baseline frozen tube width sum `{_format(baseline.get('tube_width_sum'))}` vs range_midpoint frozen `{_format(frozen.get('tube_width_sum'))}`.",
            f"- Flowstar last-width ratio: baseline frozen `{_format(baseline.get('flowstar_last_width_ratio'))}`, range_midpoint frozen `{_format(frozen.get('flowstar_last_width_ratio'))}`, range_midpoint adaptive `{_format(adaptive.get('flowstar_last_width_ratio'))}`.",
            f"- Flowstar tube-width ratio: baseline frozen `{_format(baseline.get('flowstar_tube_width_ratio'))}`, range_midpoint frozen `{_format(frozen.get('flowstar_tube_width_ratio'))}`, range_midpoint adaptive `{_format(adaptive.get('flowstar_tube_width_ratio'))}`.",
            "",
            "## Required Window Answers",
            "",
            f"1. Does range centering affect the first >1.1 crossing? Baseline frozen reset width `{_format(first_const.get('centered_reset_width_sum'))}` vs range_midpoint frozen `{_format(first_range.get('centered_reset_width_sum'))}` at checkpoint `{_format(first_range.get('checkpoint_t'))}`.",
            f"2. Does it lower polynomial-range-dominated >1.5 accumulation? Polynomial range width baseline `{_format(poly_const.get('polynomial_range_width_sum'))}` vs range_midpoint `{_format(poly_range.get('polynomial_range_width_sum'))}`; reset reduction `{_format(decision_metrics.get('reset_reduction_at_1_5'))}`.",
            f"3. Does it lower right-map-dominated >2.0 accumulation? Right-map range width baseline `{_format(right_const.get('right_map_range_width_sum'))}` vs range_midpoint `{_format(right_range.get('right_map_range_width_sum'))}`; reset reduction `{_format(decision_metrics.get('reset_reduction_at_2_0'))}`.",
            f"4. Improvement propagation: final frozen width improvement `{_format(decision_metrics.get('final_width_improvement'))}` and tube change `{_format(decision_metrics.get('tube_width_change'))}` distinguish reset-scale-only improvement from downstream width propagation.",
            f"5. Frozen-schedule improvement exists: `{_format((_float(decision_metrics.get('final_width_improvement')) or 0.0) > 0.0)}`.",
            f"6. Adaptive schedule-only explanation: schedule effects are separated because frozen replay uses identical prescribed h values; adaptive range has its own row and is not used alone for the decision.",
            "",
            "## Summary",
            "",
            "| mode | status | reached_h5 | accepted | rejected | final_width_sum | tube_width_sum | Flowstar last ratio | Flowstar tube ratio | min target margin | raw target violations | samples | decision |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                _format(row.get(field))
                for field in (
                    "mode",
                    "status",
                    "reached_horizon",
                    "accepted_steps",
                    "rejected_attempts",
                    "final_width_sum",
                    "tube_width_sum",
                    "flowstar_last_width_ratio",
                    "flowstar_tube_width_ratio",
                    "minimum_target_margin",
                    "raw_residual_target_violations",
                    "sample_containment_sanity_status",
                    "decision",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Time Alignment Semantics",
            "",
            "- Physical-time-aligned rows use the segment that contains the checkpoint time; no box-bound interpolation is performed.",
            "- Frozen-schedule rows are strict step-paired comparisons using the baseline accepted h sequence with `h_min=h_max=prescribed_h`.",
            "- Flowstar component fields remain `unknown_missing_h5_reference_component_fields`, not zero.",
            "",
            "## Formatting Checks",
            "",
            "| path | physical lines | csv.reader rows | status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in formatting_rows:
        lines.append(
            f"| {_format(row.get('path'))} | {_format(row.get('physical_line_count'))} | {_format(row.get('csv_reader_row_count'))} | {_format(row.get('status'))} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_summary.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_segments.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_frozen_schedule.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_crossings.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_decision.txt`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    out_dir: Path,
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
    crossing_rows: Sequence[Mapping[str, Any]],
    decision: str,
    decision_metrics: Mapping[str, Any],
    decision_reasons: Sequence[str],
) -> None:
    _write_csv(out_dir / "h5_right_map_centering_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "h5_right_map_centering_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "h5_right_map_centering_frozen_schedule.csv", FROZEN_FIELDS, frozen_rows)
    _write_csv(out_dir / "h5_right_map_centering_crossings.csv", CROSSING_FIELDS, crossing_rows)
    (out_dir / "h5_right_map_centering_decision.txt").write_text(decision + "\n", encoding="utf-8")
    formatting = output_formatting_rows(out_dir)
    _write_csv(out_dir / "h5_right_map_centering_formatting.csv", LINE_COUNT_FIELDS, formatting)
    write_report(
        out_dir / "h5_right_map_centering_report.md",
        summary_rows=summary_rows,
        crossing_rows=crossing_rows,
        decision=decision,
        decision_metrics=decision_metrics,
        decision_reasons=decision_reasons,
        formatting_rows=formatting,
    )


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    horizon = float(args.horizon)
    if horizon > HORIZON_LIMIT + 1e-12:
        raise ValueError("h5 right-map-centering audit is capped at T=5.0; do not run h10 from this script")
    flow_ref, flow_segments, flow_h = load_flowstar_reference(args.flowstar_segments.resolve(), horizon)
    if not flow_ref:
        raise FileNotFoundError(f"missing usable Flowstar segment reference: {args.flowstar_segments}")
    summary_rows: list[dict[str, Any]] = [_flowstar_summary_row(flow_ref)]
    segment_rows: list[dict[str, Any]] = _make_flowstar_segment_rows(flow_segments)

    baseline_summary, baseline_segments, _baseline_samples = run_centering_h5(
        mode=BASELINE_ADAPTIVE_MODE,
        run_kind="baseline adaptive",
        right_map_center_mode="constant",
        horizon=horizon,
        wall_cap_s=float(args.wall_cap_s),
    )
    baseline_match, baseline_notes = baseline_reproduction_check(baseline_summary, baseline_segments, args.h5_dir.resolve())
    baseline_summary["baseline_reproduction_match"] = baseline_match
    baseline_summary["baseline_reproduction_notes"] = baseline_notes
    summary_rows.append(baseline_summary)
    segment_rows.extend(baseline_segments)

    baseline_h = _accepted_h(baseline_segments)
    constant_frozen_summary, constant_frozen_segments, _constant_frozen_samples = run_centering_h5(
        mode=BASELINE_FROZEN_MODE,
        run_kind="baseline frozen",
        right_map_center_mode="constant",
        horizon=horizon,
        wall_cap_s=float(args.wall_cap_s),
        prescribed_h=baseline_h,
    )
    range_frozen_summary, range_frozen_segments, _range_frozen_samples = run_centering_h5(
        mode=RANGE_FROZEN_MODE,
        run_kind="range_midpoint frozen",
        right_map_center_mode="range_midpoint",
        horizon=horizon,
        wall_cap_s=float(args.wall_cap_s),
        prescribed_h=baseline_h,
    )
    range_adaptive_summary, range_adaptive_segments, _range_adaptive_samples = run_centering_h5(
        mode=RANGE_ADAPTIVE_MODE,
        run_kind="range_midpoint adaptive",
        right_map_center_mode="range_midpoint",
        horizon=horizon,
        wall_cap_s=float(args.wall_cap_s),
    )
    summary_rows.extend([constant_frozen_summary, range_frozen_summary, range_adaptive_summary])
    segment_rows.extend(constant_frozen_segments)
    segment_rows.extend(range_frozen_segments)
    segment_rows.extend(range_adaptive_segments)

    frozen_rows = make_frozen_schedule_rows(constant_frozen_segments, range_frozen_segments, baseline_h)
    checkpoints = load_checkpoints(args.divergence_dir.resolve(), horizon)
    by_mode_rows = {
        FLOWSTAR_MODE: [row for row in segment_rows if row.get("mode") == FLOWSTAR_MODE],
        BASELINE_ADAPTIVE_MODE: baseline_segments,
        BASELINE_FROZEN_MODE: constant_frozen_segments,
        RANGE_FROZEN_MODE: range_frozen_segments,
        RANGE_ADAPTIVE_MODE: range_adaptive_segments,
    }
    crossing_rows = make_crossing_rows(checkpoints, by_mode_rows)
    decision, decision_metrics, decision_reasons = decide(
        baseline_frozen=constant_frozen_summary,
        range_frozen=range_frozen_summary,
        range_adaptive=range_adaptive_summary,
        crossing_rows=crossing_rows,
    )
    summary_rows = finalize_summary_rows(
        summary_rows,
        flow_ref,
        flow_h,
        constant_frozen_summary,
        range_frozen_summary,
        decision,
        decision_metrics,
    )
    write_outputs(
        args.out_dir.resolve(),
        summary_rows=summary_rows,
        segment_rows=segment_rows,
        frozen_rows=frozen_rows,
        crossing_rows=crossing_rows,
        decision=decision,
        decision_metrics=decision_metrics,
        decision_reasons=decision_reasons,
    )
    return summary_rows, segment_rows, frozen_rows, crossing_rows, decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--h5-dir", type=Path, default=DEFAULT_H5_DIR)
    parser.add_argument("--divergence-dir", type=Path, default=DEFAULT_DIVERGENCE_DIR)
    parser.add_argument("--flowstar-segments", type=Path, default=DEFAULT_FLOWSTAR_SEGMENTS)
    parser.add_argument("--wall-cap-s", type=float, default=3600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_rows, segment_rows, frozen_rows, crossing_rows, decision = run(args)
    out = args.out_dir.resolve()
    print(f"wrote {out / 'h5_right_map_centering_summary.csv'} ({len(summary_rows)} rows)")
    print(f"wrote {out / 'h5_right_map_centering_segments.csv'} ({len(segment_rows)} rows)")
    print(f"wrote {out / 'h5_right_map_centering_frozen_schedule.csv'} ({len(frozen_rows)} rows)")
    print(f"wrote {out / 'h5_right_map_centering_crossings.csv'} ({len(crossing_rows)} rows)")
    print(f"decision {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
