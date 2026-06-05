#!/usr/bin/env python3
"""Flow*-style rescue experiment for the Van der Pol PyTorch TM benchmark."""
from __future__ import annotations

import argparse
import csv
import math
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe import (  # noqa: E402
    Interval,
    TMVector,
    flowpipe_step,
    flowpipe_step_flowstar_style_adaptive,
    flowpipe_step_from_tm,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode  # noqa: E402
from torch_tm_flowpipe.safety import intervals_are_finite  # noqa: E402

OLD_BEST_T = 0.7661635
FLOWSTAR_MIN_STEP = 0.002
ORIGINAL_FLOWSTAR_SEGMENTS = (
    REPO_ROOT / "outputs" / "flowstar_benchmark_parity" / "original_flowstar" / "original_flowstar_segments.csv"
)

SUMMARY_FIELDS = [
    "run_id",
    "mode",
    "order",
    "candidate_order",
    "output_order",
    "truncation_range_split",
    "adaptive_order_fallback",
    "fallback_from_order",
    "refinement_pass",
    "residual_subset_current",
    "validation_mode",
    "cutoff_threshold",
    "target_remainder_radius",
    "center_correction_width_factor",
    "center_correction_attempts",
    "center_corrections_applied",
    "center_corrected_dimensions",
    "max_center_correction_abs",
    "max_residual_radius_after_correction",
    "selective_high_degree_terms_top_k",
    "max_selective_retained_terms_count",
    "max_selective_dropped_remainder_width_sum",
    "status",
    "runtime_s",
    "validated_segments",
    "last_validated_t",
    "last_attempted_t",
    "min_h_used",
    "min_regular_h_used",
    "min_final_alignment_h",
    "h_below_flowstar_min_count",
    "max_h_used",
    "num_step_rejections",
    "num_accepted_steps",
    "num_rejected_steps",
    "num_order8_steps",
    "num_order8_attempts",
    "failure_reason",
    "final_width_sum",
    "max_width_sum",
    "max_residual_width_sum",
    "max_remainder_width_sum",
    "notes",
]

SEGMENT_FIELDS = [
    "run_id",
    "mode",
    "order",
    "candidate_order",
    "output_order",
    "truncation_range_split",
    "validation_mode",
    "cutoff_threshold",
    "target_remainder_radius",
    "center_correction_width_factor",
    "selective_high_degree_terms_top_k",
    "selective_retained_terms_count",
    "selective_dropped_terms_count",
    "selective_nonretained_terms_count",
    "selective_dropped_remainder_width_sum",
    "selective_total_dropped_width_sum",
    "segment_index",
    "status",
    "validation_attempts",
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
    "step_rejections",
    "next_h",
    "message",
]

VALIDATION_ATTEMPT_FIELDS = [
    "run_id",
    "mode",
    "order",
    "candidate_order",
    "output_order",
    "truncation_range_split",
    "adaptive_order_fallback",
    "fallback_from_order",
    "refinement_pass",
    "residual_subset_current",
    "validation_mode",
    "cutoff_threshold",
    "target_remainder_radius",
    "center_correction_width_factor",
    "selective_high_degree_terms_top_k",
    "segment_index",
    "adaptive_attempt_index",
    "t_lo",
    "t_hi",
    "attempt_index",
    "h",
    "h_try",
    "h_min",
    "h_max",
    "candidate_segment_width_x",
    "candidate_segment_width_y",
    "candidate_segment_width_sum",
    "candidate_final_width_x",
    "candidate_final_width_y",
    "candidate_final_width_sum",
    "residual_width_x",
    "residual_width_y",
    "residual_width_sum",
    "residual_lo_x",
    "residual_hi_x",
    "residual_lo_y",
    "residual_hi_y",
    "remainder_width_x",
    "remainder_width_y",
    "remainder_width_sum",
    "target_remainder_width",
    "target_remainder_width_sum",
    "center_correction_applied",
    "correction_value_x",
    "correction_value_y",
    "residual_before_lo_x",
    "residual_before_hi_x",
    "residual_before_lo_y",
    "residual_before_hi_y",
    "residual_after_lo_x",
    "residual_after_hi_x",
    "residual_after_lo_y",
    "residual_after_hi_y",
    "residual_before_center_x",
    "residual_before_center_y",
    "residual_after_center_x",
    "residual_after_center_y",
    "residual_before_radius_x",
    "residual_before_radius_y",
    "residual_after_radius_x",
    "residual_after_radius_y",
    "subset_after_correction",
    "subset_result",
    "rejection_reason",
    "polynomial_range_width_x",
    "polynomial_range_width_y",
    "polynomial_range_width_sum",
    "total_range_width_x",
    "total_range_width_y",
    "total_range_width_sum",
    "finite_residual",
    "validation_status",
    "validation_message",
]

COMPARISON_FIELDS = [
    "run_id",
    "py_status",
    "py_segments",
    "py_runtime_s",
    "py_last_validated_t",
    "py_last_width_sum",
    "py_tube_width_sum",
    "flowstar_segments_over_same_horizon",
    "flowstar_last_width_sum_near_T",
    "flowstar_tube_width_sum_over_same_horizon",
    "last_width_ratio",
    "tube_width_ratio",
    "max_time_overlap_width_ratio",
    "median_time_overlap_width_ratio",
]


NEXT_FIELDS = [
    "variant_group",
    "run_id",
    "validation_mode",
    "target_remainder_radius",
    "cutoff_threshold",
    "status",
    "last_validated_t",
    "runtime_s",
    "num_accepted_steps",
    "num_rejected_steps",
    "num_order8_steps",
    "candidate_order",
    "output_order",
    "truncation_range_split",
    "center_corrections_applied",
    "selective_high_degree_terms_top_k",
    "max_selective_retained_terms_count",
    "min_regular_h_used",
    "h_below_flowstar_min_count",
    "final_width_sum",
    "last_width_ratio",
    "tube_width_ratio",
    "notes",
]


RETAINED_TERM_FIELDS = [
    "run_id",
    "segment_index",
    "t_lo",
    "t_hi",
    "status",
    "selective_high_degree_terms_top_k",
    "state_index",
    "state_dimension",
    "term_rank",
    "retained",
    "monomial",
    "coefficient",
    "total_degree",
    "abs_interval_contribution",
    "term_interval_lo",
    "term_interval_hi",
    "term_interval_width",
]

NEXT3_FIELDS = [
    *NEXT_FIELDS,
    "center_corrected_dimensions",
    "max_center_correction_abs",
    "max_selective_dropped_remainder_width_sum",
]


class StepTimeout(RuntimeError):
    pass


def _initial_box() -> list[Interval]:
    return [Interval(1.1, 1.4), Interval(2.35, 2.45)]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in fields})


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.17g}"
        return str(value)
    return value


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _max_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    vals = [_finite_float(row.get(field)) for row in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else ""


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _max_abs_fields(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> float | str:
    vals: list[float] = []
    for row in rows:
        for field in fields:
            value = _finite_float(row.get(field))
            if value is not None:
                vals.append(abs(value))
    return max(vals) if vals else ""


def _interval_tuple(iv: Interval) -> tuple[float, float]:
    return iv.to_tuple()


def _segment_bounds(box: Sequence[Interval]) -> tuple[float, float, float, float, float, float, float]:
    x_lo, x_hi = _interval_tuple(box[0])
    y_lo, y_hi = _interval_tuple(box[1])
    width_x = x_hi - x_lo
    width_y = y_hi - y_lo
    return x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_x + width_y


def _call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
    if timeout_s <= 0:
        raise StepTimeout("wall-time cap reached before validation call")
    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(_signum: int, _frame: Any) -> None:
        raise StepTimeout("wall-time cap reached during validation call")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(float(timeout_s), 1e-6))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _finish_attempt_rows(rows: list[dict[str, Any]], start: int, *, run_id: str, t_lo: float) -> None:
    for row in rows[start:]:
        row.setdefault("run_id", run_id)
        row["t_lo"] = t_lo
        h = _finite_float(row.get("h")) or 0.0
        row["t_hi"] = t_lo + h


def _segment_row(
    *,
    spec: Mapping[str, Any],
    segment_index: int,
    status: str,
    seg: Any,
    t_lo: float,
    t_hi: float,
    box: Sequence[Interval],
) -> dict[str, Any]:
    x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = _segment_bounds(box)
    selective_stats = dict(getattr(seg, "selective_term_stats", None) or {})
    row = {
        "run_id": spec["run_id"],
        "mode": spec["mode"],
        "order": getattr(seg, "order", spec["order"]),
        "candidate_order": spec.get("candidate_order", spec["order"]),
        "output_order": spec.get("order", ""),
        "truncation_range_split": spec.get("truncation_range_split", ""),
        "validation_mode": spec.get("validation_mode", "growth"),
        "cutoff_threshold": "" if spec.get("cutoff_threshold") is None else spec.get("cutoff_threshold"),
        "target_remainder_radius": spec.get("target_remainder_radius", ""),
        "center_correction_width_factor": spec.get("center_correction_width_factor", ""),
        "selective_high_degree_terms_top_k": spec.get("selective_high_degree_terms_top_k", ""),
        "segment_index": segment_index,
        "status": status,
        "validation_attempts": getattr(seg, "validation_attempts", ""),
        "t_lo": t_lo,
        "t_hi": t_hi,
        "h": getattr(seg, "h", t_hi - t_lo),
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "width_x": width_x,
        "width_y": width_y,
        "width_sum": width_sum,
        "step_rejections": getattr(seg, "step_rejections", 0),
        "next_h": "" if getattr(seg, "next_h", None) is None else getattr(seg, "next_h"),
        "message": getattr(seg, "message", ""),
    }
    row.update(selective_stats)
    details = getattr(seg, "selective_term_details", None)
    if details:
        row["_selective_term_details"] = [dict(item) for item in details]
    return row


def _summarize_run(
    spec: Mapping[str, Any],
    *,
    max_horizon: float,
    status: str,
    runtime_s: float,
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    last_attempted_t: float,
    failure_reason: str,
    notes: str,
) -> dict[str, Any]:
    validated = [row for row in segment_rows if row.get("status") == "validated"]
    h_vals = [_finite_float(row.get("h")) for row in validated]
    h_vals = [h for h in h_vals if h is not None]
    flowstar_min_step = float(spec.get("h_min", FLOWSTAR_MIN_STEP))
    adaptive = spec.get("kind") == "adaptive" or spec.get("mode") == "flowstar_style"
    regular_h_vals: list[float] = []
    final_alignment_h_vals: list[float] = []
    h_below_flowstar_min_count = 0
    for row in validated:
        h = _finite_float(row.get("h"))
        t_hi = _finite_float(row.get("t_hi"))
        if h is None:
            continue
        is_final_alignment = (
            adaptive
            and t_hi is not None
            and abs(t_hi - float(max_horizon)) <= 1e-9
            and h < flowstar_min_step - 1e-12
        )
        if is_final_alignment:
            final_alignment_h_vals.append(h)
        else:
            regular_h_vals.append(h)
            if adaptive and h < flowstar_min_step - 1e-12:
                h_below_flowstar_min_count += 1
    num_rejected_steps = sum(int(row.get("step_rejections") or 0) for row in segment_rows)
    center_rows = [row for row in attempt_rows if _truthy(row.get("center_correction_applied"))]
    center_correction_attempts = len(center_rows)
    center_corrected_dimensions = sum(
        1
        for row in center_rows
        for field in ("correction_value_x", "correction_value_y")
        if abs(_finite_float(row.get(field)) or 0.0) > 0.0
    )
    max_center_correction_abs = _max_abs_fields(center_rows, ["correction_value_x", "correction_value_y"])
    max_after_radius = _max_field(center_rows, "residual_after_radius_x")
    max_after_radius_y = _max_field(center_rows, "residual_after_radius_y")
    if max_after_radius == "":
        max_after_radius = max_after_radius_y
    elif max_after_radius_y != "":
        max_after_radius = max(float(max_after_radius), float(max_after_radius_y))
    selective_retained = _max_field(segment_rows, "selective_retained_terms_count")
    selective_drop_width = _max_field(segment_rows, "selective_dropped_remainder_width_sum")
    return {
        "run_id": spec["run_id"],
        "mode": spec["mode"],
        "order": spec["order"],
        "candidate_order": spec.get("candidate_order", spec["order"]),
        "output_order": spec.get("order", ""),
        "truncation_range_split": spec.get("truncation_range_split", ""),
        "validation_mode": spec.get("validation_mode", "growth"),
        "cutoff_threshold": "" if spec.get("cutoff_threshold") is None else spec.get("cutoff_threshold"),
        "target_remainder_radius": spec.get("target_remainder_radius", ""),
        "center_correction_width_factor": spec.get("center_correction_width_factor", ""),
        "center_correction_attempts": center_correction_attempts,
        "center_corrections_applied": center_correction_attempts,
        "center_corrected_dimensions": center_corrected_dimensions,
        "max_center_correction_abs": max_center_correction_abs,
        "max_residual_radius_after_correction": max_after_radius,
        "selective_high_degree_terms_top_k": spec.get("selective_high_degree_terms_top_k", ""),
        "max_selective_retained_terms_count": selective_retained,
        "max_selective_dropped_remainder_width_sum": selective_drop_width,
        "status": status,
        "runtime_s": runtime_s,
        "validated_segments": len(validated),
        "last_validated_t": float(validated[-1]["t_hi"]) if validated else 0.0,
        "last_attempted_t": last_attempted_t,
        "min_h_used": min(h_vals) if h_vals else "",
        "min_regular_h_used": min(regular_h_vals) if regular_h_vals else "",
        "min_final_alignment_h": min(final_alignment_h_vals) if final_alignment_h_vals else "",
        "h_below_flowstar_min_count": h_below_flowstar_min_count,
        "max_h_used": max(h_vals) if h_vals else "",
        "num_step_rejections": num_rejected_steps,
        "num_accepted_steps": len(validated),
        "num_rejected_steps": num_rejected_steps,
        "num_order8_steps": sum(1 for row in validated if int(row.get("order") or 0) == 8),
        "num_order8_attempts": sum(1 for row in attempt_rows if int(row.get("order") or 0) == 8),
        "failure_reason": failure_reason,
        "final_width_sum": validated[-1]["width_sum"] if validated else "",
        "max_width_sum": _max_field(validated, "width_sum"),
        "max_residual_width_sum": _max_field(attempt_rows, "residual_width_sum"),
        "max_remainder_width_sum": _max_field(attempt_rows, "remainder_width_sum"),
        "notes": notes,
    }


def _run_fixed(spec: Mapping[str, Any], *, max_horizon: float, wall_cap_s: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    current_box = _initial_box()
    current_tm = TMVector.identity(current_box, order=int(spec["order"]))
    t = 0.0
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    status = "max_horizon_reached"
    failure_reason = ""
    last_attempted_t = 0.0
    start = time.perf_counter()
    segment_index = 0

    while t < max_horizon - 1e-15:
        elapsed = time.perf_counter() - start
        if elapsed >= wall_cap_s:
            status = "timeout"
            failure_reason = f"wall-time cap reached before segment {segment_index}"
            break
        h = min(float(spec["h"]), max_horizon - t)
        context = {
            "run_id": spec["run_id"],
            "mode": spec["mode"],
            "order": spec["order"],
            "candidate_order": spec.get("candidate_order", spec["order"]),
            "output_order": spec.get("order", ""),
            "truncation_range_split": spec.get("truncation_range_split", ""),
            "validation_mode": "growth",
            "cutoff_threshold": "",
            "target_remainder_radius": "",
            "segment_index": segment_index,
        }
        attempt_start = len(attempt_rows)
        try:
            if spec["mode"] == "range_only":
                seg = _call_with_timeout(
                    lambda: flowpipe_step(
                        van_der_pol_ode,
                        current_box,
                        h,
                        int(spec["order"]),
                        diagnostics=attempt_rows,
                        diagnostics_context=context,
                    ),
                    wall_cap_s - elapsed,
                )
            else:
                seg = _call_with_timeout(
                    lambda: flowpipe_step_from_tm(
                        van_der_pol_ode,
                        current_tm,
                        h,
                        int(spec["order"]),
                        diagnostics=attempt_rows,
                        diagnostics_context=context,
                    ),
                    wall_cap_s - elapsed,
                )
        except StepTimeout as exc:
            status = "timeout"
            failure_reason = str(exc)
            break
        _finish_attempt_rows(attempt_rows, attempt_start, run_id=spec["run_id"], t_lo=t)
        last_attempted_t = t + h
        final_box = seg.final_tm.range_box()
        finite = intervals_are_finite(final_box)
        row_status = "validated" if seg.status == "validated" and finite else "failed"
        segment_rows.append(_segment_row(spec=spec, segment_index=segment_index, status=row_status, seg=seg, t_lo=t, t_hi=t + h, box=final_box))
        if row_status != "validated":
            status = "failed"
            failure_reason = seg.message or "validation failed"
            break
        if spec["mode"] == "range_only":
            current_box = [iv.inflate(1e-9) for iv in final_box]
        else:
            current_tm = seg.final_tm
        t += h
        segment_index += 1

    runtime_s = time.perf_counter() - start
    notes = "validated to requested horizon" if status == "max_horizon_reached" else failure_reason
    return (
        _summarize_run(
            spec,
            max_horizon=max_horizon,
            status=status,
            runtime_s=runtime_s,
            segment_rows=segment_rows,
            attempt_rows=attempt_rows,
            last_attempted_t=last_attempted_t,
            failure_reason=failure_reason,
            notes=notes,
        ),
        segment_rows,
        attempt_rows,
    )


def _run_adaptive(spec: Mapping[str, Any], *, max_horizon: float, wall_cap_s: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    current: Any = _initial_box()
    t = 0.0
    h_request = float(spec.get("h_max", 0.1))
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    status = "max_horizon_reached"
    failure_reason = ""
    last_attempted_t = 0.0
    start = time.perf_counter()
    segment_index = 0

    while t < max_horizon - 1e-15:
        elapsed = time.perf_counter() - start
        if elapsed >= wall_cap_s:
            status = "timeout"
            failure_reason = f"wall-time cap reached before segment {segment_index}"
            break
        h = min(h_request, float(spec.get("h_max", 0.1)), max_horizon - t)
        local_h_min = min(float(spec.get("h_min", 0.002)), h)
        context = {
            "run_id": spec["run_id"],
            "mode": spec["mode"],
            "order": spec["order"],
            "candidate_order": spec.get("candidate_order", spec["order"]),
            "output_order": spec.get("order", ""),
            "truncation_range_split": spec.get("truncation_range_split", ""),
            "validation_mode": spec["validation_mode"],
            "cutoff_threshold": "" if spec.get("cutoff_threshold") is None else spec.get("cutoff_threshold"),
            "target_remainder_radius": spec.get("target_remainder_radius", ""),
            "center_correction_width_factor": spec.get("center_correction_width_factor", ""),
            "selective_high_degree_terms_top_k": spec.get("selective_high_degree_terms_top_k", ""),
            "segment_index": segment_index,
        }
        attempt_start = len(attempt_rows)
        try:
            seg = _call_with_timeout(
                lambda: flowpipe_step_flowstar_style_adaptive(
                    van_der_pol_ode,
                    current,
                    h=h,
                    order=int(spec["order"]),
                    h_min=local_h_min,
                    h_max=float(spec.get("h_max", 0.1)),
                    target_remainder_radius=float(spec.get("target_remainder_radius", 1e-4)),
                    center_correction_width_factor=float(spec.get("center_correction_width_factor") or 1.05),
                    cutoff_threshold=spec.get("cutoff_threshold"),
                    max_validation_attempts=int(spec.get("max_validation_attempts", 2)),
                    validation_mode=str(spec.get("validation_mode", "target_remainder")),
                    adaptive_order_fallback=spec.get("adaptive_order_fallback"),
                    adaptive_order_threshold_factor=float(spec.get("adaptive_order_threshold_factor", 1.25)),
                    candidate_order=spec.get("candidate_order"),
                    truncation_range_split=spec.get("truncation_range_split"),
                    selective_high_degree_terms_top_k=spec.get("selective_high_degree_terms_top_k"),
                    diagnostics=attempt_rows,
                    diagnostics_context=context,
                ),
                wall_cap_s - elapsed,
            )
        except StepTimeout as exc:
            status = "timeout"
            failure_reason = str(exc)
            break
        _finish_attempt_rows(attempt_rows, attempt_start, run_id=spec["run_id"], t_lo=t)
        last_attempted_t = t + float(seg.h)
        final_box = seg.final_tm.range_box()
        finite = intervals_are_finite(final_box)
        row_status = "validated" if seg.status == "validated" and finite else "failed"
        segment_rows.append(_segment_row(spec=spec, segment_index=segment_index, status=row_status, seg=seg, t_lo=t, t_hi=t + float(seg.h), box=final_box))
        if row_status != "validated":
            status = "failed"
            failure_reason = seg.message or "validation failed"
            break
        current = seg.reset_tm if seg.reset_tm is not None else seg.final_tm
        h_request = float(seg.next_h) if seg.next_h is not None else min(float(seg.h) * 1.5, float(spec.get("h_max", 0.1)))
        t += float(seg.h)
        segment_index += 1

    runtime_s = time.perf_counter() - start
    notes = "validated to requested horizon" if status == "max_horizon_reached" else failure_reason
    return (
        _summarize_run(
            spec,
            max_horizon=max_horizon,
            status=status,
            runtime_s=runtime_s,
            segment_rows=segment_rows,
            attempt_rows=attempt_rows,
            last_attempted_t=last_attempted_t,
            failure_reason=failure_reason,
            notes=notes,
        ),
        segment_rows,
        attempt_rows,
    )


def _configs() -> list[dict[str, Any]]:
    def flowstar_spec(
        run_id: str,
        *,
        order: int = 6,
        target_remainder_radius: float = 1e-4,
        cutoff_threshold: float | None = None,
        validation_mode: str = "target_remainder",
        max_validation_attempts: int = 2,
        adaptive_order_fallback: int | None = None,
        candidate_order: int | None = None,
        truncation_range_split: int | None = None,
        center_correction_width_factor: float = 1.05,
        selective_high_degree_terms_top_k: int | None = None,
    ) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "run_id": run_id,
            "mode": "flowstar_style",
            "order": order,
            "validation_mode": validation_mode,
            "target_remainder_radius": target_remainder_radius,
            "center_correction_width_factor": center_correction_width_factor if validation_mode == "target_remainder_centered" else "",
            "cutoff_threshold": cutoff_threshold,
            "h_min": 0.002,
            "h_max": 0.1,
            "max_validation_attempts": max_validation_attempts,
            "kind": "adaptive",
        }
        if candidate_order is not None:
            spec["candidate_order"] = int(candidate_order)
        if truncation_range_split is not None:
            spec["truncation_range_split"] = int(truncation_range_split)
        if selective_high_degree_terms_top_k is not None:
            spec["selective_high_degree_terms_top_k"] = int(selective_high_degree_terms_top_k)
        if adaptive_order_fallback is not None:
            spec["adaptive_order_fallback"] = adaptive_order_fallback
            spec["adaptive_order_threshold_factor"] = 1.25
        return spec

    return [
        {
            "run_id": "baseline_range_only_o6_s4",
            "mode": "range_only",
            "order": 6,
            "validation_mode": "growth",
            "h": 0.025,
            "kind": "fixed",
        },
        {
            "run_id": "baseline_dependency_preserving_o4_s1",
            "mode": "dependency_preserving",
            "order": 4,
            "validation_mode": "growth",
            "h": 0.1,
            "kind": "fixed",
        },
        flowstar_spec("flowstar_style_o4_target", order=4),
        flowstar_spec("flowstar_style_o6_target", order=6),
        flowstar_spec("flowstar_style_o4_target_cutoff", order=4, cutoff_threshold=1e-10),
        flowstar_spec("flowstar_style_o6_target_cutoff", order=6, cutoff_threshold=1e-10),
        flowstar_spec("flowstar_style_o6_target_adaptive_order_8", order=6, adaptive_order_fallback=8),
        flowstar_spec(
            "flowstar_style_o6_target_cutoff_adaptive_order_8",
            order=6,
            cutoff_threshold=1e-10,
            adaptive_order_fallback=8,
        ),
        flowstar_spec("flowstar_style_o6_target_r2e-4", order=6, target_remainder_radius=2e-4),
        flowstar_spec("flowstar_style_o6_target_r5e-4", order=6, target_remainder_radius=5e-4),
        flowstar_spec(
            "flowstar_style_o6_target_refined",
            order=6,
            validation_mode="target_remainder_refined",
            max_validation_attempts=8,
        ),
        flowstar_spec(
            "flowstar_style_o6_target_refined_cutoff",
            order=6,
            validation_mode="target_remainder_refined",
            cutoff_threshold=1e-10,
            max_validation_attempts=8,
        ),
        flowstar_spec("flowstar_style_o6_candidate8_output6", order=6, candidate_order=8),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
        ),
        flowstar_spec("flowstar_style_o6_target_truncsplit2", order=6, truncation_range_split=2),
        flowstar_spec("flowstar_style_o6_target_truncsplit4", order=6, truncation_range_split=4),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_truncsplit2",
            order=6,
            candidate_order=8,
            truncation_range_split=2,
        ),
        flowstar_spec(
            "flowstar_style_o6_target_centered",
            order=6,
            validation_mode="target_remainder_centered",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_centered",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            validation_mode="target_remainder_centered",
        ),
        flowstar_spec("flowstar_style_o6_candidate8_output6_keep1", order=6, candidate_order=8, selective_high_degree_terms_top_k=1),
        flowstar_spec("flowstar_style_o6_candidate8_output6_keep2", order=6, candidate_order=8, selective_high_degree_terms_top_k=2),
        flowstar_spec("flowstar_style_o6_candidate8_output6_keep4", order=6, candidate_order=8, selective_high_degree_terms_top_k=4),
        flowstar_spec("flowstar_style_o6_candidate8_output6_keep8", order=6, candidate_order=8, selective_high_degree_terms_top_k=8),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_keep1_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
            selective_high_degree_terms_top_k=1,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_keep2_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
            selective_high_degree_terms_top_k=2,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_keep4_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
            selective_high_degree_terms_top_k=4,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_keep8_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
            selective_high_degree_terms_top_k=8,
        ),
    ]


def _normalize_config_ids(config_ids: Sequence[str] | None) -> list[str]:
    if config_ids is None:
        return []
    normalized: list[str] = []
    for raw in config_ids:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                normalized.append(part)
    return normalized


def _select_configs(config_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    configs = _configs()
    selected_ids = _normalize_config_ids(config_ids)
    if not selected_ids:
        return configs
    by_id = {str(spec["run_id"]): spec for spec in configs}
    missing = [run_id for run_id in selected_ids if run_id not in by_id]
    if missing:
        raise ValueError(f"unknown config(s): {', '.join(missing)}")
    return [by_id[run_id] for run_id in selected_ids]


def _best(rows: Sequence[Mapping[str, Any]], *, mode: str | None = None, not_mode: str | None = None) -> Mapping[str, Any] | None:
    selected = []
    for row in rows:
        if mode is not None and row.get("mode") != mode:
            continue
        if not_mode is not None and row.get("mode") == not_mode:
            continue
        selected.append(row)
    if not selected:
        return None
    return max(selected, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def write_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    best_old = _best(summary_rows, not_mode="flowstar_style")
    best_rescue = _best(summary_rows, mode="flowstar_style")
    best_rescue_t = _finite_float(best_rescue.get("last_validated_t")) if best_rescue else 0.0
    best_old_t = _finite_float(best_old.get("last_validated_t")) if best_old else 0.0
    o6_target = next((r for r in summary_rows if r.get("run_id") == "flowstar_style_o6_target"), None)
    o6_cutoff = next((r for r in summary_rows if r.get("run_id") == "flowstar_style_o6_target_cutoff"), None)
    cutoff_rows = [r for r in summary_rows if r.get("mode") == "flowstar_style" and str(r.get("cutoff_threshold", "")) not in {"", "None"}]
    no_cutoff_rows = [r for r in summary_rows if r.get("mode") == "flowstar_style" and str(r.get("cutoff_threshold", "")) in {"", "None"}]
    best_cutoff = _best(cutoff_rows)
    best_no_cutoff = _best(no_cutoff_rows)
    cutoff_msg = "inconclusive"
    if best_cutoff and best_no_cutoff:
        ct = _finite_float(best_cutoff.get("last_validated_t")) or 0.0
        nt = _finite_float(best_no_cutoff.get("last_validated_t")) or 0.0
        if ct > nt:
            cutoff_msg = "helped"
        elif ct < nt:
            cutoff_msg = "hurt"
        else:
            cutoff_msg = "tied"
    recenter_msg = "yes" if best_rescue_t > best_old_t else "no"
    target_rows = [r for r in summary_rows if str(r.get("validation_mode", "")).startswith("target_remainder")]
    max_target_rem = _max_field(target_rows, "max_remainder_width_sum")
    target_rem_float = _finite_float(max_target_rem)
    target_bounded = target_rem_float is not None and target_rem_float <= 0.0004 + 1e-15
    reached_requested = best_rescue_t >= float(max_horizon) - 1e-9
    comparison_rows = list(comparison_rows or [])
    best_comparison = max(
        comparison_rows,
        key=lambda r: (_finite_float(r.get("py_last_validated_t")) or 0.0, -(_finite_float(r.get("tube_width_ratio")) or math.inf)),
        default=None,
    )
    width_msg = "not compared"
    tightness_msg = "needs Flow* comparison"
    if best_comparison:
        last_ratio = _finite_float(best_comparison.get("last_width_ratio"))
        tube_ratio = _finite_float(best_comparison.get("tube_width_ratio"))
        width_msg = (
            f"last width ratio=`{best_comparison.get('last_width_ratio', '')}`, "
            f"tube width ratio=`{best_comparison.get('tube_width_ratio', '')}`"
        )
        tightness_msg = "yes" if last_ratio is not None and tube_ratio is not None and last_ratio <= 1.0 and tube_ratio <= 1.0 else "needs more work"
    reachability_tightness = "both" if reached_requested and tightness_msg == "yes" else ("reachability only" if reached_requested else "neither yet")

    lines = [
        "# Flowstar-Style Rescue Report",
        "",
        f"Requested max horizon: `{float(max_horizon):.17g}`.",
        f"Best old baseline in this run: `{best_old['run_id'] if best_old else ''}` at t=`{best_old_t:.17g}`.",
        f"Best flowstar_style run: `{best_rescue['run_id'] if best_rescue else ''}` at t=`{best_rescue_t:.17g}`.",
        "",
        f"Did flowstar_style beat the old best t~={OLD_BEST_T}? {'yes' if best_rescue_t > OLD_BEST_T else 'no'}.",
        f"Did flowstar_style_o6_target reach the requested horizon? {_yes_no(bool(o6_target and (_finite_float(o6_target.get('last_validated_t')) or 0.0) >= float(max_horizon) - 1e-9))}.",
        f"Did cutoff help? {cutoff_msg}.",
        f"Did target remainder stay bounded at width sum 0.0004? {_yes_no(target_bounded)}; max target-mode remainder width sum was `{max_target_rem}`.",
        f"Did recenter/rescale help compared to range_only and dependency_preserving? {recenter_msg}; best flowstar_style t=`{best_rescue_t:.17g}` vs best baseline t=`{best_old_t:.17g}`.",
        f"Best rescue candidate: `{best_rescue['run_id'] if best_rescue else ''}`.",
        f"Accepted/rejected steps for best rescue: `{best_rescue.get('num_accepted_steps', '') if best_rescue else ''}` accepted, `{best_rescue.get('num_rejected_steps', '') if best_rescue else ''}` rejected.",
        f"min_regular_h_used for best rescue: `{best_rescue.get('min_regular_h_used', '') if best_rescue else ''}`.",
        f"Did any non-final step go below Flow* min step 0.002? {_yes_no(bool(best_rescue and int(best_rescue.get('h_below_flowstar_min_count') or 0) > 0))}.",
        f"How do widths compare to original Flow* over the same horizon? {width_msg}.",
        f"Is this a reachability success, a tightness success, or both? {reachability_tightness}.",
        f"Failure mode for the best rescue candidate: `{best_rescue.get('failure_reason', '') if best_rescue else ''}`.",
        "Do not treat this as Flow* parity unless horizon 10 is reached and boxes are compared separately.",
        "",
        "## Summary Rows",
        "",
        "| run_id | status | last_validated_t | accepted | rejected | min_h_used | min_regular_h_used | non_final_h_below_0.002 | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['run_id']} | {row['status']} | {row['last_validated_t']} | {row.get('num_accepted_steps', '')} | "
            f"{row.get('num_rejected_steps', '')} | {row['min_h_used']} | {row.get('min_regular_h_used', '')} | "
            f"{row.get('h_below_flowstar_min_count', '')} | {row['failure_reason']} |"
        )
    if o6_target or o6_cutoff:
        lines.extend(
            [
                "",
                "## Selected Configs",
                "",
                "| run_id | status | last_validated_t | runtime_s | min_regular_h_used | non_final_h_below_0.002 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in [r for r in [o6_target, o6_cutoff] if r is not None]:
            lines.append(
                f"| {row['run_id']} | {row['status']} | {row['last_validated_t']} | {row['runtime_s']} | "
                f"{row.get('min_regular_h_used', '')} | {row.get('h_below_flowstar_min_count', '')} |"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rescue_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]], attempt_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except Exception:
        return

    colors = {
        "baseline_range_only_o6_s4": "#1f77b4",
        "baseline_dependency_preserving_o4_s1": "#ff7f0e",
        "flowstar_style_o4_target": "#2ca02c",
        "flowstar_style_o6_target": "#d62728",
        "flowstar_style_o4_target_cutoff": "#9467bd",
        "flowstar_style_o6_target_cutoff": "#8c564b",
        "flowstar_style_o6_target_adaptive_order_8": "#e377c2",
        "flowstar_style_o6_target_cutoff_adaptive_order_8": "#7f7f7f",
        "flowstar_style_o6_target_r2e-4": "#17becf",
        "flowstar_style_o6_target_r5e-4": "#bcbd22",
        "flowstar_style_o6_target_refined": "#1f77b4",
        "flowstar_style_o6_target_refined_cutoff": "#ff7f0e",
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in segment_rows:
        if row.get("status") == "validated":
            grouped.setdefault(str(row["run_id"]), []).append(row)

    def _plot_time(var: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for run_id, rows in grouped.items():
            rows = sorted(rows, key=lambda r: float(r["t_hi"]))
            t = [float(r["t_hi"]) for r in rows]
            lo = [float(r[f"{var}_lo"]) for r in rows]
            hi = [float(r[f"{var}_hi"]) for r in rows]
            color = colors.get(run_id)
            ax.plot(t, [(a + b) / 2 for a, b in zip(lo, hi)], label=run_id, color=color)
            ax.fill_between(t, lo, hi, alpha=0.16, color=color)
        ax.set_xlabel("t")
        ax.set_ylabel(var)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    _plot_time("x", out_dir / "rescue_t_x.png")
    _plot_time("y", out_dir / "rescue_t_y.png")

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    for run_id, rows in grouped.items():
        color = colors.get(run_id)
        for row in rows:
            x_lo = float(row["x_lo"])
            x_hi = float(row["x_hi"])
            y_lo = float(row["y_lo"])
            y_hi = float(row["y_hi"])
            rect = patches.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, fill=False, alpha=0.22, edgecolor=color)
            ax.add_patch(rect)
        if rows:
            ax.plot(
                [(float(r["x_lo"]) + float(r["x_hi"])) / 2 for r in rows],
                [(float(r["y_lo"]) + float(r["y_hi"])) / 2 for r in rows],
                label=run_id,
                color=color,
            )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "rescue_phase_xy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id, rows in grouped.items():
        rows = [r for r in rows if str(r.get("mode")) == "flowstar_style"]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: float(r["t_hi"]))
        ax.plot(
            [float(r["t_hi"]) for r in rows],
            [float(r["h"]) for r in rows],
            marker="o",
            markersize=2.4,
            linewidth=1.0,
            label=run_id,
            color=colors.get(run_id),
        )
    ax.axhline(FLOWSTAR_MIN_STEP, color="#111111", linewidth=0.9, linestyle="--", label="Flow* min step 0.002")
    ax.set_xlabel("t")
    ax.set_ylabel("accepted h")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "step_size_trace.png", dpi=160)
    plt.close(fig)

    residual_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in attempt_rows:
        if row.get("mode") == "flowstar_style":
            residual_groups.setdefault(str(row["run_id"]), []).append(row)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    target_lines: list[float] = []
    for run_id, rows in residual_groups.items():
        pts: list[tuple[float, float]] = []
        for row in rows:
            t_hi = _finite_float(row.get("t_hi"))
            residual = _finite_float(row.get("residual_width_sum"))
            if t_hi is not None and residual is not None and residual > 0:
                pts.append((t_hi, residual))
            target = _finite_float(row.get("target_remainder_width_sum"))
            if target is not None and target > 0:
                target_lines.append(target)
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=0.8, label=run_id, color=colors.get(run_id))
    if target_lines:
        ax.axhline(max(target_lines), color="#111111", linewidth=0.9, linestyle="--", label="target remainder width sum")
    ax.set_xlabel("t")
    ax.set_ylabel("residual width sum")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_vs_t.png", dpi=160)
    plt.close(fig)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rows_for_run(rows: Sequence[Mapping[str, Any]], run_id: str) -> list[Mapping[str, Any]]:
    return sorted(
        [row for row in rows if row.get("run_id") == run_id and row.get("status") == "validated"],
        key=lambda r: _finite_float(r.get("t_hi")) or 0.0,
    )


def _overlap_rows(rows: Sequence[Mapping[str, Any]], t_lo: float, t_hi: float) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for row in rows:
        row_lo = _finite_float(row.get("t_lo"))
        row_hi = _finite_float(row.get("t_hi"))
        if row_lo is None or row_hi is None:
            continue
        if row_hi > t_lo + 1e-15 and row_lo < t_hi - 1e-15:
            out.append(row)
    return out


def _nearest_time_row(rows: Sequence[Mapping[str, Any]], t: float) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    row = min(rows, key=lambda r: abs(((_finite_float(r.get("t_lo")) or 0.0) + (_finite_float(r.get("t_hi")) or 0.0)) * 0.5 - t))
    return [row]


def _tube_width_sum(rows: Sequence[Mapping[str, Any]]) -> float | str:
    if not rows:
        return ""
    xs_lo = [_finite_float(row.get("x_lo")) for row in rows]
    xs_hi = [_finite_float(row.get("x_hi")) for row in rows]
    ys_lo = [_finite_float(row.get("y_lo")) for row in rows]
    ys_hi = [_finite_float(row.get("y_hi")) for row in rows]
    vals = [v for v in [*xs_lo, *xs_hi, *ys_lo, *ys_hi] if v is not None]
    if len(vals) != len(rows) * 4:
        return ""
    return (max(v for v in xs_hi if v is not None) - min(v for v in xs_lo if v is not None)) + (
        max(v for v in ys_hi if v is not None) - min(v for v in ys_lo if v is not None)
    )


def _safe_ratio(num: Any, den: Any) -> float | str:
    n = _finite_float(num)
    d = _finite_float(den)
    if n is None or d is None or d <= 0:
        return ""
    return n / d


def _time_overlap_ratio_rows(
    run_id: str, py_rows: Sequence[Mapping[str, Any]], flow_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for py in py_rows:
        t_lo = _finite_float(py.get("t_lo"))
        t_hi = _finite_float(py.get("t_hi"))
        if t_lo is None or t_hi is None:
            continue
        flow_overlap = _overlap_rows(flow_rows, t_lo, t_hi)
        flow_width = _tube_width_sum(flow_overlap)
        ratio = _safe_ratio(py.get("width_sum"), flow_width)
        if ratio == "":
            continue
        rows.append(
            {
                "run_id": run_id,
                "t": 0.5 * (t_lo + t_hi),
                "py_width_sum": py.get("width_sum", ""),
                "flowstar_overlap_width_sum": flow_width,
                "width_ratio": ratio,
            }
        )
    return rows


def _comparison_row(
    summary: Mapping[str, Any],
    py_rows: Sequence[Mapping[str, Any]],
    flow_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_id = str(summary["run_id"])
    if not py_rows:
        return (
            {
                "run_id": run_id,
                "py_status": summary.get("status", ""),
                "py_segments": 0,
                "py_runtime_s": summary.get("runtime_s", ""),
                "py_last_validated_t": summary.get("last_validated_t", ""),
            },
            [],
        )
    t = _finite_float(summary.get("last_validated_t")) or (_finite_float(py_rows[-1].get("t_hi")) or 0.0)
    same_horizon_flow = _overlap_rows(flow_rows, 0.0, t)
    py_last = py_rows[-1]
    py_last_t_lo = _finite_float(py_last.get("t_lo")) or 0.0
    py_last_t_hi = _finite_float(py_last.get("t_hi")) or t
    flow_last = _overlap_rows(flow_rows, py_last_t_lo, py_last_t_hi) or _nearest_time_row(flow_rows, t)
    py_tube = _tube_width_sum(py_rows)
    flow_tube = _tube_width_sum(same_horizon_flow)
    flow_last_width = _tube_width_sum(flow_last)
    ratio_rows = _time_overlap_ratio_rows(run_id, py_rows, flow_rows)
    ratios = [_finite_float(row.get("width_ratio")) for row in ratio_rows]
    ratios = [r for r in ratios if r is not None]
    return (
        {
            "run_id": run_id,
            "py_status": summary.get("status", ""),
            "py_segments": len(py_rows),
            "py_runtime_s": summary.get("runtime_s", ""),
            "py_last_validated_t": summary.get("last_validated_t", ""),
            "py_last_width_sum": py_last.get("width_sum", ""),
            "py_tube_width_sum": py_tube,
            "flowstar_segments_over_same_horizon": len(same_horizon_flow),
            "flowstar_last_width_sum_near_T": flow_last_width,
            "flowstar_tube_width_sum_over_same_horizon": flow_tube,
            "last_width_ratio": _safe_ratio(py_last.get("width_sum"), flow_last_width),
            "tube_width_ratio": _safe_ratio(py_tube, flow_tube),
            "max_time_overlap_width_ratio": max(ratios) if ratios else "",
            "median_time_overlap_width_ratio": statistics.median(ratios) if ratios else "",
        },
        ratio_rows,
    )


def _add_tx_boxes(ax: Any, rows: Sequence[Mapping[str, Any]], var: str, *, color: str, label: str, alpha: float) -> None:
    import matplotlib.patches as patches

    for i, row in enumerate(rows):
        t_lo = _finite_float(row.get("t_lo"))
        t_hi = _finite_float(row.get("t_hi"))
        v_lo = _finite_float(row.get(f"{var}_lo"))
        width = _finite_float(row.get(f"width_{var}"))
        if t_lo is None or t_hi is None or v_lo is None or width is None:
            continue
        ax.add_patch(
            patches.Rectangle(
                (t_lo, v_lo),
                t_hi - t_lo,
                width,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0.7,
                label=label if i == 0 else None,
            )
        )


def _add_phase_boxes(ax: Any, rows: Sequence[Mapping[str, Any]], *, color: str, label: str, alpha: float) -> None:
    import matplotlib.patches as patches

    for i, row in enumerate(rows):
        x_lo = _finite_float(row.get("x_lo"))
        y_lo = _finite_float(row.get("y_lo"))
        width_x = _finite_float(row.get("width_x"))
        width_y = _finite_float(row.get("width_y"))
        if x_lo is None or y_lo is None or width_x is None or width_y is None:
            continue
        ax.add_patch(
            patches.Rectangle(
                (x_lo, y_lo),
                width_x,
                width_y,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0.7,
                label=label if i == 0 else None,
            )
        )


def _set_limits_from_rows(ax: Any, rows: Sequence[Mapping[str, Any]], keys: tuple[str, str], axis: str) -> None:
    vals: list[float] = []
    for row in rows:
        lo = _finite_float(row.get(keys[0]))
        hi = _finite_float(row.get(keys[1]))
        if lo is not None and hi is not None:
            vals.extend([lo, hi])
    if not vals:
        return
    pad = max((max(vals) - min(vals)) * 0.05, 1e-6)
    if axis == "x":
        ax.set_xlim(min(vals) - pad, max(vals) + pad)
    else:
        ax.set_ylim(min(vals) - pad, max(vals) + pad)


def make_flowstar_comparison_plots(
    out_dir: Path,
    py_rows: Sequence[Mapping[str, Any]],
    flow_rows: Sequence[Mapping[str, Any]],
    ratio_rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not py_rows:
        return
    t = _finite_float(py_rows[-1].get("t_hi")) or 0.0
    flow_same = _overlap_rows(flow_rows, 0.0, t)
    for var in ("x", "y"):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        _add_tx_boxes(ax, flow_same, var, color="#2ca02c", label="Original Flow*", alpha=0.16)
        _add_tx_boxes(ax, py_rows, var, color="#1f77b4", label="PyTorch rescue", alpha=0.12)
        ax.set_xlabel("t")
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7)
        ax.set_xlim(0.0, max(t, 1e-9))
        _set_limits_from_rows(ax, [*flow_same, *py_rows], (f"{var}_lo", f"{var}_hi"), "y")
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_rescue_vs_original_flowstar_t_{var}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    _add_phase_boxes(ax, flow_same, color="#2ca02c", label="Original Flow*", alpha=0.14)
    _add_phase_boxes(ax, py_rows, color="#1f77b4", label="PyTorch rescue", alpha=0.10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    _set_limits_from_rows(ax, [*flow_same, *py_rows], ("x_lo", "x_hi"), "x")
    _set_limits_from_rows(ax, [*flow_same, *py_rows], ("y_lo", "y_hi"), "y")
    fig.tight_layout()
    fig.savefig(out_dir / "overlay_rescue_vs_original_flowstar_phase_xy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id, rows in ratio_rows_by_run.items():
        pts = [
            (_finite_float(row.get("t")), _finite_float(row.get("width_ratio")))
            for row in rows
            if _finite_float(row.get("t")) is not None and _finite_float(row.get("width_ratio")) is not None
        ]
        if not pts:
            continue
        pts.sort(key=lambda p: p[0] or 0.0)
        ax.plot([p[0] for p in pts if p[0] is not None], [p[1] for p in pts if p[1] is not None], linewidth=1.0, label=run_id)
    ax.axhline(1.0, color="#111111", linewidth=0.9, linestyle="--", label="Flow* width")
    ax.set_xlabel("t")
    ax.set_ylabel("PyTorch width / Flow* overlap hull width")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "width_ratio_vs_t.png", dpi=160)
    plt.close(fig)


def write_flowstar_comparison_report(
    out_dir: Path,
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = max(comparison_rows, key=lambda r: _finite_float(r.get("py_last_validated_t")) or 0.0, default=None)
    reached = bool(best and (_finite_float(best.get("py_last_validated_t")) or 0.0) >= float(max_horizon) - 1e-9)
    width_comparable = "needs more work"
    if best:
        last_ratio = _finite_float(best.get("last_width_ratio"))
        tube_ratio = _finite_float(best.get("tube_width_ratio"))
        if last_ratio is not None and tube_ratio is not None and last_ratio <= 1.0 and tube_ratio <= 1.0:
            width_comparable = "yes"
        elif last_ratio is not None and tube_ratio is not None:
            width_comparable = "no"
    lines = [
        "# Rescue Vs Original Flow* Comparison",
        "",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        "Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.",
        "This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.",
        "",
        f"Best rescue config: `{best.get('run_id', '') if best else ''}`.",
        f"Reached requested horizon? {_yes_no(reached)}.",
        f"Width comparable to Flow*? {width_comparable}.",
        "",
        "## Metrics",
        "",
        "| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['run_id']} | {row['py_status']} | {row['py_last_validated_t']} | {row['py_segments']} | "
            f"{row['last_width_ratio']} | {row['tube_width_ratio']} | {row['max_time_overlap_width_ratio']} | "
            f"{row['median_time_overlap_width_ratio']} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rescue_vs_flowstar_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rescue_vs_flowstar_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> list[dict[str, Any]]:
    if not ORIGINAL_FLOWSTAR_SEGMENTS.exists():
        return []
    flow_rows = _read_csv_rows(ORIGINAL_FLOWSTAR_SEGMENTS)
    comparison_rows: list[dict[str, Any]] = []
    ratio_rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for summary in summary_rows:
        if summary.get("mode") != "flowstar_style":
            continue
        py_rows = _rows_for_run(segment_rows, str(summary["run_id"]))
        if not py_rows:
            continue
        comparison, ratio_rows = _comparison_row(summary, py_rows, flow_rows)
        comparison_rows.append(comparison)
        ratio_rows_by_run[str(summary["run_id"])] = ratio_rows
    if not comparison_rows:
        return []
    _write_csv(out_dir / "rescue_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
    write_flowstar_comparison_report(out_dir, comparison_rows, max_horizon=max_horizon)
    best = max(comparison_rows, key=lambda r: _finite_float(r.get("py_last_validated_t")) or 0.0)
    best_py_rows = _rows_for_run(segment_rows, str(best["run_id"]))
    make_flowstar_comparison_plots(out_dir, best_py_rows, flow_rows, ratio_rows_by_run)
    return comparison_rows


def _flowstar_style_reached_requested_horizon(summary_rows: Sequence[Mapping[str, Any]], max_horizon: float) -> bool:
    for row in summary_rows:
        if row.get("mode") != "flowstar_style":
            continue
        if (_finite_float(row.get("last_validated_t")) or 0.0) >= float(max_horizon) - 1e-9:
            return True
    return False


def run_experiment(
    out_dir: Path,
    *,
    max_horizon: float,
    wall_cap_s: float,
    config_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for spec in _select_configs(config_ids):
        if spec["kind"] == "fixed":
            summary, segments, attempts = _run_fixed(spec, max_horizon=max_horizon, wall_cap_s=wall_cap_s)
        else:
            summary, segments, attempts = _run_adaptive(spec, max_horizon=max_horizon, wall_cap_s=wall_cap_s)
        summary_rows.append(summary)
        segment_rows.extend(segments)
        attempt_rows.extend(attempts)
        _write_outputs(out_dir, summary_rows, segment_rows, attempt_rows, max_horizon=max_horizon)
    _write_outputs(out_dir, summary_rows, segment_rows, attempt_rows, max_horizon=max_horizon)
    make_plots(out_dir, segment_rows, attempt_rows)
    comparison_rows: list[dict[str, Any]] = []
    if max_horizon >= 5.0:
        comparison_rows = write_rescue_vs_flowstar_outputs(
            out_dir,
            summary_rows,
            segment_rows,
            max_horizon=max_horizon,
        )
        if comparison_rows:
            write_report(out_dir, summary_rows, segment_rows, max_horizon=max_horizon, comparison_rows=comparison_rows)
    write_specialized_outputs(
        out_dir,
        summary_rows,
        segment_rows,
        attempt_rows,
        max_horizon=max_horizon,
        comparison_rows=comparison_rows,
    )
    write_rescue_next_outputs(trigger_out_dir=out_dir)
    write_rescue_next2_outputs(trigger_out_dir=out_dir)
    write_rescue_next3_outputs(trigger_out_dir=out_dir)
    return summary_rows, segment_rows, attempt_rows



def _summary_with_h5_baseline(summary_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = list(summary_rows)
    if any(row.get("run_id") == "flowstar_style_o6_target" for row in rows):
        return rows
    baseline_path = REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_summary.csv"
    if baseline_path.exists():
        for row in _read_csv_rows(baseline_path):
            if row.get("run_id") == "flowstar_style_o6_target":
                rows.insert(0, row)
                break
    return rows


def _comparison_by_run(comparison_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("run_id", "")): row for row in comparison_rows}


def _write_adaptive_order_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best(summary_rows)
    baseline_rows = _summary_with_h5_baseline([])
    baseline = next((row for row in baseline_rows if row.get("run_id") == "flowstar_style_o6_target"), None)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    baseline_t = _finite_float(baseline.get("last_validated_t")) if baseline else 2.1095541733932355
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    comp = _comparison_by_run(comparison_rows).get(str(best.get("run_id", ""))) if best else None
    order8_count = sum(int(row.get("num_order8_steps") or 0) for row in summary_rows)
    cutoff_rows = [row for row in summary_rows if str(row.get("cutoff_threshold", "")) not in {"", "None"}]
    no_cutoff_rows = [row for row in summary_rows if str(row.get("cutoff_threshold", "")) in {"", "None"}]
    cutoff_help = "inconclusive"
    if cutoff_rows and no_cutoff_rows:
        ct = max(_finite_float(r.get("last_validated_t")) or 0.0 for r in cutoff_rows)
        nt = max(_finite_float(r.get("last_validated_t")) or 0.0 for r in no_cutoff_rows)
        cutoff_help = "yes" if ct > nt else ("no" if ct < nt else "tied")
    lines = [
        "# Adaptive Order Rescue Report",
        "",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best adaptive-order variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did adaptive order fallback beat t~=2.10955? {_yes_no(bool(best_t is not None and baseline_t is not None and best_t > baseline_t))}.",
        f"Did it reach horizon 5? {_yes_no(reached)}.",
        f"Across all configs, accepted order-8 steps in this artifact: `{order8_count}`; best-run order-8 steps=`{best.get('num_order8_steps', '') if best else ''}`.",
        "If both cutoff and no-cutoff adaptive configs are present, the aggregate count is the total across those configs, not a single-run step count.",
        f"Runtime impact: best runtime_s=`{best.get('runtime_s', '') if best else ''}` vs h5 baseline runtime_s=`{baseline.get('runtime_s', '') if baseline else ''}`.",
        f"Width vs Flow* ratio: last=`{comp.get('last_width_ratio', '') if comp else ''}`, tube=`{comp.get('tube_width_ratio', '') if comp else ''}`.",
        f"Did cutoff help? {cutoff_help}.",
        "",
        "## Rows",
        "",
        "| run_id | status | last_validated_t | order8_steps | runtime_s | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('num_order8_steps', '')} | {row.get('runtime_s', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "adaptive_order_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_remainder_sensitivity_report(out_dir: Path, rows: Sequence[Mapping[str, Any]], *, max_horizon: float) -> None:
    ordered = sorted(rows, key=lambda r: _finite_float(r.get("target_remainder_radius")) or 0.0)
    base = ordered[0] if ordered else None
    reached = [row for row in ordered if (_finite_float(row.get("last_validated_t")) or 0.0) >= float(max_horizon) - 1e-9]
    base_width = _finite_float(base.get("final_width_sum")) if base else None
    lines = [
        "# Target Remainder Sensitivity Report",
        "",
        "This is diagnostic only; larger target remainders are relaxed parameters, not Flow* parity.",
        f"Does loosening target remainder reach horizon 5? {_yes_no(bool(reached))}.",
        f"Is 2e-4 enough? {_yes_no(any(row.get('run_id') == 'flowstar_style_o6_target_r2e-4' for row in reached))}.",
        f"Is 5e-4 enough? {_yes_no(any(row.get('run_id') == 'flowstar_style_o6_target_r5e-4' for row in reached))}.",
        "",
        "## Rows",
        "",
        "| radius | run_id | status | last_validated_t | final_width_sum | width_vs_1e-4 | rejected_steps |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        width = _finite_float(row.get("final_width_sum"))
        width_ratio = width / base_width if width is not None and base_width and base_width > 0 else ""
        lines.append(
            f"| {row.get('target_remainder_radius', '')} | {row.get('run_id', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('final_width_sum', '')} | {width_ratio} | {row.get('num_rejected_steps', '')} |"
        )
    lines.extend(
        [
            "",
            "Relaxed target remainders can reduce rejections only if the validated horizon improves without unacceptable width growth.",
            "Do not recommend relaxed remainders as parity unless the report explicitly labels the parameter change.",
        ]
    )
    (out_dir / "remainder_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_refined_report(out_dir: Path, summary_rows: Sequence[Mapping[str, Any]], *, max_horizon: float) -> None:
    best = _best(summary_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    lines = [
        "# Refined Target Validation Report",
        "",
        f"Best refined variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did refined validation beat t~=2.10955? {_yes_no(bool(best_t is not None and best_t > 2.1095541733932355))}.",
        f"Did it reach horizon 5? {_yes_no(reached)}.",
        f"Runtime impact: best runtime_s=`{best.get('runtime_s', '') if best else ''}`.",
        "Residual-over-target ratios are recorded in `rescue_validation_attempts.csv` via the target and residual width fields.",
        "",
        "## Rows",
        "",
        "| run_id | status | last_validated_t | runtime_s | failure_reason |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('runtime_s', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "refined_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def _best_comparison_for_run(
    comparison_rows: Sequence[Mapping[str, Any]],
    run_id: str,
) -> Mapping[str, Any]:
    return _comparison_by_run(comparison_rows).get(str(run_id), {})


def _adaptive_order_baselines() -> tuple[Mapping[str, str] | None, Mapping[str, str]]:
    rows = _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "adaptive_order_summary.csv")
    comps = _comparison_by_run(
        _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "rescue_vs_flowstar_comparison.csv")
    )
    best = max(rows, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0, default=None)
    return best, comps.get(str(best.get("run_id", ""))) if best else {}


def _write_candidate_order_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best(summary_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    best_comp = _best_comparison_for_run(comparison_rows, str(best.get("run_id", ""))) if best else {}
    adaptive, adaptive_comp = _adaptive_order_baselines()
    adaptive_t = _finite_float(adaptive.get("last_validated_t")) if adaptive else 2.2771582567640953
    adaptive_runtime = adaptive.get("runtime_s", "") if adaptive else ""
    best_runtime = best.get("runtime_s", "") if best else ""
    best_tube = _finite_float(best_comp.get("tube_width_ratio"))
    adaptive_tube = _finite_float(adaptive_comp.get("tube_width_ratio"))
    if best_tube is None or adaptive_tube is None:
        width_msg = "not compared"
    elif best_tube < adaptive_tube:
        width_msg = "improved"
    elif best_tube > adaptive_tube:
        width_msg = "worsened"
    else:
        width_msg = "tied"
    best_residual = _finite_float(best.get("max_residual_width_sum")) if best else None
    adaptive_residual = _finite_float(adaptive.get("max_residual_width_sum")) if adaptive else None
    if best_residual is None or adaptive_residual is None:
        residual_msg = "inconclusive"
    elif best_residual < adaptive_residual:
        residual_msg = "yes by max residual width sum"
    elif best_residual > adaptive_residual:
        residual_msg = "no; max residual width sum increased"
    else:
        residual_msg = "tied by max residual width sum"
    lines = [
        "# Candidate Order Diagnostic Report",
        "",
        "Candidate-order mode validates with a higher Picard polynomial order and truncates the accepted/output Taylor model back to output order with the dropped contribution added to interval uncertainty.",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best candidate-order variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did candidate_order=8/output_order=6 beat t~=2.277? {_yes_no(bool(best_t is not None and adaptive_t is not None and best_t > adaptive_t))}.",
        f"Did it reach horizon 5? {_yes_no(reached)}.",
        f"Width ratio vs adaptive full-order-8 fallback: {width_msg}; candidate last=`{best_comp.get('last_width_ratio', '')}`, tube=`{best_comp.get('tube_width_ratio', '')}`, adaptive tube=`{adaptive_comp.get('tube_width_ratio', '')}`.",
        f"Runtime impact: best runtime_s=`{best_runtime}` vs adaptive fallback runtime_s=`{adaptive_runtime}`.",
        f"Does it reduce truncation containment miss? {residual_msg}; candidate max_residual_width_sum=`{best.get('max_residual_width_sum', '') if best else ''}`, adaptive=`{adaptive.get('max_residual_width_sum', '') if adaptive else ''}`.",
        "",
        "## Rows",
        "",
        "| run_id | status | candidate_order | output_order | last_validated_t | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    comp_by_run = _comparison_by_run(comparison_rows)
    for row in summary_rows:
        comp = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('candidate_order', '')} | "
            f"{row.get('output_order', '')} | {row.get('last_validated_t', '')} | {row.get('runtime_s', '')} | "
            f"{comp.get('last_width_ratio', '')} | {comp.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "candidate_order_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_truncation_range_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best(summary_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    comp_by_run = _comparison_by_run(comparison_rows)
    best_comp = comp_by_run.get(str(best.get("run_id", ""))) if best else {}
    cutoff_split_rows = [row for row in summary_rows if "cutoff" in str(row.get("run_id", "")) and "truncsplit" in str(row.get("run_id", ""))]
    cutoff_msg = "not evaluated by the requested truncation-range config set" if not cutoff_split_rows else "see rows"
    lines = [
        "# Truncation Range Diagnostic Report",
        "",
        "Dropped/truncated polynomial terms are still bounded conservatively; this diagnostic only changes how their interval range is evaluated.",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best truncation-range variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Does tighter dropped-term range bounding beat t~=2.277? {_yes_no(bool(best_t is not None and best_t > 2.2771582567640953))}.",
        f"Does it reach horizon 5? {_yes_no(reached)}.",
        f"Runtime cost for best variant: runtime_s=`{best.get('runtime_s', '') if best else ''}`.",
        f"Width ratio vs Flow*: last=`{best_comp.get('last_width_ratio', '') if best_comp else ''}`, tube=`{best_comp.get('tube_width_ratio', '') if best_comp else ''}`.",
        f"Did cutoff help when combined with truncsplit? {cutoff_msg}.",
        "",
        "## Rows",
        "",
        "| run_id | status | split | candidate_order | output_order | last_validated_t | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        comp = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('truncation_range_split', '')} | "
            f"{row.get('candidate_order', '')} | {row.get('output_order', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('runtime_s', '')} | {comp.get('last_width_ratio', '')} | {comp.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "truncation_range_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def _retained_term_rows(segment_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segment_rows:
        details = segment.get("_selective_term_details") or []
        for detail in details:
            row = {
                "run_id": segment.get("run_id", ""),
                "segment_index": segment.get("segment_index", ""),
                "t_lo": segment.get("t_lo", ""),
                "t_hi": segment.get("t_hi", ""),
                "status": segment.get("status", ""),
                "selective_high_degree_terms_top_k": segment.get("selective_high_degree_terms_top_k", ""),
            }
            row.update(dict(detail))
            rows.append(row)
    rows.sort(
        key=lambda row: (
            _finite_float(row.get("t_lo")) or 0.0,
            _finite_float(row.get("segment_index")) or 0.0,
            0 if _truthy(row.get("retained")) else 1,
            _finite_float(row.get("term_rank")) or 0.0,
        )
    )
    if not rows:
        return []
    near_t = max(_finite_float(row.get("t_lo")) or 0.0 for row in rows)
    near_rows = [row for row in rows if (_finite_float(row.get("t_lo")) or 0.0) >= near_t - 0.25]
    return near_rows or rows


def _write_residual_centering_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best(summary_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    comp = _best_comparison_for_run(comparison_rows, str(best.get("run_id", ""))) if best else {}
    corrections = sum(int(row.get("center_corrections_applied") or 0) for row in summary_rows)
    corrected_dims = sum(int(row.get("center_corrected_dimensions") or 0) for row in summary_rows)
    max_corr = _max_field(summary_rows, "max_center_correction_abs")
    target_radii = {str(row.get("target_remainder_radius", "")) for row in summary_rows}
    target_stayed = target_radii <= {"0.0001", "0.000100000000000000", "1e-04", "1e-4", "0.00010000000000000000"}
    below_min = any(int(row.get("h_below_flowstar_min_count") or 0) > 0 for row in summary_rows)
    after_subset = sum(1 for row in attempt_rows if _truthy(row.get("center_correction_applied")) and _truthy(row.get("subset_after_correction")))
    lines = [
        "# Residual Centering Diagnostic Report",
        "",
        "This opt-in mode keeps the symmetric target remainder and accepts only after recomputing the Picard residual from the corrected candidate.",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best centered variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did centered validation beat t~=2.400737? {_yes_no(bool(best_t is not None and best_t > 2.400737667399793))}.",
        f"Did it reach horizon 5? {_yes_no(reached)}.",
        f"Center-correction attempts: `{corrections}` attempts, `{corrected_dims}` corrected dimensions; subset-after-correction rows=`{after_subset}`.",
        f"Did corrections stay small? max_abs_correction=`{max_corr}`.",
        f"Width ratio vs Flow*: last=`{comp.get('last_width_ratio', '')}`, tube=`{comp.get('tube_width_ratio', '')}`.",
        f"Did target remainder remain at 1e-4? {_yes_no(target_stayed)}.",
        f"Any non-final h below 0.002? {_yes_no(below_min)}.",
        "",
        "## Rows",
        "",
        "| run_id | status | last_validated_t | corrections | corrected_dims | max_abs_correction | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    comp_by_run = _comparison_by_run(comparison_rows)
    for row in summary_rows:
        comp_row = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('center_corrections_applied', '')} | {row.get('center_corrected_dimensions', '')} | "
            f"{row.get('max_center_correction_abs', '')} | {comp_row.get('last_width_ratio', '')} | "
            f"{comp_row.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "residual_centering_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_selective_terms_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    raw_best = _best(summary_rows)
    raw_best_t = _finite_float(raw_best.get("last_validated_t")) if raw_best else 0.0
    tied_best_rows = [
        row
        for row in summary_rows
        if raw_best_t is not None
        and (row_t := _finite_float(row.get("last_validated_t"))) is not None
        and abs(row_t - raw_best_t) <= 1e-12
    ]

    def _drop_width_key(row: Mapping[str, Any]) -> tuple[bool, float]:
        width = _finite_float(row.get("max_selective_dropped_remainder_width_sum"))
        return (width is None, width if width is not None else math.inf)

    best = min(tied_best_rows, key=_drop_width_key, default=raw_best)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    comp = _best_comparison_for_run(comparison_rows, str(best.get("run_id", ""))) if best else {}
    best_k = best.get("selective_high_degree_terms_top_k", "") if best else ""
    if len(tied_best_rows) > 1:
        best_k_summary = f"all tested K values tied on validated time; K=`{best_k}` minimized dropped remainder width"
    else:
        best_k_summary = f"`{best_k}`"
    adaptive, adaptive_comp = _adaptive_order_baselines()
    adaptive_t = _finite_float(adaptive.get("last_validated_t")) if adaptive else 2.2771582567640953
    residual_centers = _max_abs_fields(attempt_rows, ["residual_before_center_x", "residual_before_center_y", "residual_after_center_x", "residual_after_center_y"])
    lines = [
        "# Selective High-Degree Term Diagnostic Report",
        "",
        "This is diagnostic-only: sparse over-order terms are retained beyond output_order=6, so this is not fixed-order Flow* parity.",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best selective variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did selective retention beat t~=2.400737? {_yes_no(bool(best_t is not None and best_t > 2.400737667399793))}.",
        f"Did any variant reach horizon 5? {_yes_no(reached)}.",
        f"Which K worked best? {best_k_summary}.",
        f"Did keeping a few terms reduce residual shift? max recorded residual center magnitude=`{residual_centers}` (compare by row in attempts CSV).",
        f"Runtime impact: best runtime_s=`{best.get('runtime_s', '') if best else ''}`.",
        f"Width ratio vs Flow*: last=`{comp.get('last_width_ratio', '')}`, tube=`{comp.get('tube_width_ratio', '')}`.",
        f"Did this outperform full adaptive order fallback? {_yes_no(bool(best_t is not None and adaptive_t is not None and best_t > adaptive_t))}; adaptive tube=`{adaptive_comp.get('tube_width_ratio', '')}`.",
        "",
        "## Rows",
        "",
        "| run_id | K | status | last_validated_t | retained_terms | dropped_remainder_width | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    comp_by_run = _comparison_by_run(comparison_rows)
    for row in summary_rows:
        comp_row = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('selective_high_degree_terms_top_k', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('max_selective_retained_terms_count', '')} | "
            f"{row.get('max_selective_dropped_remainder_width_sum', '')} | {row.get('runtime_s', '')} | "
            f"{comp_row.get('last_width_ratio', '')} | {comp_row.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "selective_terms_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_specialized_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    name = out_dir.name
    if name == "flowstar_style_rescue_adaptive_order":
        _write_csv(out_dir / "adaptive_order_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "adaptive_order_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_csv(out_dir / "adaptive_order_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
        _write_adaptive_order_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_rescue_remainder_sensitivity":
        rows = _summary_with_h5_baseline(summary_rows)
        _write_csv(out_dir / "remainder_sensitivity_summary.csv", SUMMARY_FIELDS, rows)
        _write_remainder_sensitivity_report(out_dir, rows, max_horizon=max_horizon)
    elif name == "flowstar_style_rescue_refined":
        _write_csv(out_dir / "refined_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_refined_report(out_dir, summary_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_candidate_order":
        _write_csv(out_dir / "candidate_order_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "candidate_order_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_candidate_order_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_truncation_range":
        _write_csv(out_dir / "truncation_range_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_truncation_range_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_residual_centering":
        _write_csv(out_dir / "residual_centering_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "residual_centering_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_csv(out_dir / "residual_centering_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
        _write_residual_centering_report(out_dir, summary_rows, attempt_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_selective_terms":
        _write_csv(out_dir / "selective_terms_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "selective_terms_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_selective_terms_report(out_dir, summary_rows, attempt_rows, comparison_rows, max_horizon=max_horizon)
        _write_csv(out_dir / "retained_terms_near_failure.csv", RETAINED_TERM_FIELDS, _retained_term_rows(segment_rows))


def _read_optional_csv(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(path) if path.exists() else []


def _variant_group(run_id: str) -> str:
    if "keep" in run_id and "centered" in run_id:
        return "selective_terms_centered"
    if "keep" in run_id:
        return "selective_high_degree_terms"
    if "centered" in run_id:
        return "residual_centering"
    if "residual_shift" in run_id:
        return "residual_shift_diagnostic"
    if "candidate8_output6" in run_id and "truncsplit" in run_id:
        return "candidate_order_truncation_split"
    if "candidate8_output6" in run_id:
        return "candidate_order_output_order"
    if "truncsplit" in run_id:
        return "truncation_range_split"
    if "adaptive_order_8" in run_id:
        return "adaptive_order_fallback"
    if "r2e-4" in run_id or "r5e-4" in run_id:
        return "relaxed_target_remainder"
    if "refined" in run_id:
        return "refined_target_validation"
    return "h5_current_best"


def write_rescue_next_outputs(*, trigger_out_dir: Path | None = None) -> None:
    if trigger_out_dir is not None:
        try:
            outputs_root = (REPO_ROOT / "outputs").resolve()
            if not trigger_out_dir.resolve().is_relative_to(outputs_root):
                return
        except Exception:
            return
    candidates: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Mapping[str, Any]] = {}
    sources = [
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "adaptive_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_remainder_sensitivity" / "remainder_sensitivity_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_remainder_sensitivity" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_refined" / "refined_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_refined" / "rescue_vs_flowstar_comparison.csv"),
    ]
    for summary_path, comparison_path in sources:
        for row in _read_optional_csv(summary_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                candidates[run_id] = dict(row)
        for row in _read_optional_csv(comparison_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                comparisons[run_id] = row
    if not candidates:
        return
    rows: list[dict[str, Any]] = []
    for run_id, row in sorted(candidates.items(), key=lambda item: (_variant_group(item[0]), item[0])):
        comp = comparisons.get(run_id, {})
        rows.append(
            {
                "variant_group": _variant_group(run_id),
                "run_id": run_id,
                "validation_mode": row.get("validation_mode", ""),
                "target_remainder_radius": row.get("target_remainder_radius", ""),
                "cutoff_threshold": row.get("cutoff_threshold", ""),
                "status": row.get("status", ""),
                "last_validated_t": row.get("last_validated_t", ""),
                "runtime_s": row.get("runtime_s", ""),
                "num_accepted_steps": row.get("num_accepted_steps", ""),
                "num_rejected_steps": row.get("num_rejected_steps", ""),
                "num_order8_steps": row.get("num_order8_steps", ""),
                "candidate_order": row.get("candidate_order", row.get("order", "")),
                "output_order": row.get("output_order", row.get("order", "")),
                "truncation_range_split": row.get("truncation_range_split", ""),
                "min_regular_h_used": row.get("min_regular_h_used", ""),
                "h_below_flowstar_min_count": row.get("h_below_flowstar_min_count", ""),
                "final_width_sum": row.get("final_width_sum", ""),
                "last_width_ratio": comp.get("last_width_ratio", ""),
                "tube_width_ratio": comp.get("tube_width_ratio", ""),
                "notes": row.get("notes", ""),
            }
        )
    out_dir = REPO_ROOT / "outputs" / "flowstar_style_rescue_next"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rescue_next_summary.csv", NEXT_FIELDS, rows)
    parity_rows = [
        row for row in rows
        if str(row.get("target_remainder_radius", "")) in {"0.0001", "1e-04", "1e-4"}
        and int(row.get("h_below_flowstar_min_count") or 0) == 0
    ]
    best = max(parity_rows or rows, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0)
    relaxed_reached = any(
        _variant_group(str(row.get("run_id", ""))) == "relaxed_target_remainder"
        and (_finite_float(row.get("last_validated_t")) or 0.0) >= 5.0 - 1e-9
        for row in rows
    )
    if (_finite_float(best.get("last_validated_t")) or 0.0) >= 5.0 - 1e-9:
        recommendation = "continue with the best parity-preserving variant and tighten polynomial range bounding."
    elif relaxed_reached:
        recommendation = "use relaxed target remainder only as a diagnostic; prioritize tighter polynomial range bounding or a symbolic remainder queue for parity."
    elif _variant_group(str(best.get("run_id", ""))) == "refined_target_validation":
        recommendation = "continue refined target validation, then add tighter polynomial range bounding."
    else:
        recommendation = "prioritize tighter polynomial range bounding, then a real Flow*-style symbolic remainder queue."
    lines = [
        "# Rescue Variant Comparison",
        "",
        f"Best variant by current decision criteria: `{best.get('run_id', '')}` at t=`{best.get('last_validated_t', '')}`.",
        f"Reached horizon 5? {_yes_no((_finite_float(best.get('last_validated_t')) or 0.0) >= 5.0 - 1e-9)}.",
        f"Width ratio vs Flow*: last=`{best.get('last_width_ratio', '')}`, tube=`{best.get('tube_width_ratio', '')}`.",
        f"Next recommendation: {recommendation}",
        "",
        "Decision criteria: highest last_validated_t, target remainder close to Flow* parameter, runtime, width ratio vs Flow*, and no non-final h below 0.002 except diagnostic runs.",
        "",
        "## Rows",
        "",
        "| group | run_id | status | last_validated_t | radius | last_width_ratio | tube_width_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant_group', '')} | {row.get('run_id', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('target_remainder_radius', '')} | "
            f"{row.get('last_width_ratio', '')} | {row.get('tube_width_ratio', '')} |"
        )
    (out_dir / "rescue_next_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_rescue_next2_outputs(*, trigger_out_dir: Path | None = None) -> None:
    if trigger_out_dir is not None:
        try:
            outputs_root = (REPO_ROOT / "outputs").resolve()
            if not trigger_out_dir.resolve().is_relative_to(outputs_root):
                return
        except Exception:
            return
    candidates: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Mapping[str, Any]] = {}
    sources = [
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "adaptive_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_candidate_order" / "candidate_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_candidate_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_truncation_range" / "truncation_range_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_truncation_range" / "rescue_vs_flowstar_comparison.csv"),
    ]
    for summary_path, comparison_path in sources:
        for row in _read_optional_csv(summary_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                candidates[run_id] = dict(row)
        for row in _read_optional_csv(comparison_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                comparisons[run_id] = row

    residual_rows = _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_style_residual_shift" / "residual_shift.csv")
    if residual_rows:
        row = residual_rows[-1]
        run_id = "residual_shift_diagnostic_y"
        candidates[run_id] = {
            "run_id": run_id,
            "validation_mode": "diagnostic_only",
            "target_remainder_radius": row.get("target_radius", "0.0001"),
            "cutoff_threshold": "",
            "status": "diagnostic_only",
            "last_validated_t": row.get("t_start", ""),
            "runtime_s": "",
            "num_accepted_steps": "",
            "num_rejected_steps": "",
            "num_order8_steps": "",
            "candidate_order": "",
            "output_order": "",
            "truncation_range_split": "",
            "min_regular_h_used": "",
            "h_below_flowstar_min_count": "",
            "final_width_sum": "",
            "notes": "diagnostic only; not an accepted run",
        }
    if not candidates:
        return

    rows: list[dict[str, Any]] = []
    for run_id, row in sorted(candidates.items(), key=lambda item: (_variant_group(item[0]), item[0])):
        comp = comparisons.get(run_id, {})
        rows.append(
            {
                "variant_group": _variant_group(run_id),
                "run_id": run_id,
                "validation_mode": row.get("validation_mode", ""),
                "target_remainder_radius": row.get("target_remainder_radius", ""),
                "cutoff_threshold": row.get("cutoff_threshold", ""),
                "status": row.get("status", ""),
                "last_validated_t": row.get("last_validated_t", ""),
                "runtime_s": row.get("runtime_s", ""),
                "num_accepted_steps": row.get("num_accepted_steps", ""),
                "num_rejected_steps": row.get("num_rejected_steps", ""),
                "num_order8_steps": row.get("num_order8_steps", ""),
                "candidate_order": row.get("candidate_order", row.get("order", "")),
                "output_order": row.get("output_order", row.get("order", "")),
                "truncation_range_split": row.get("truncation_range_split", ""),
                "min_regular_h_used": row.get("min_regular_h_used", ""),
                "h_below_flowstar_min_count": row.get("h_below_flowstar_min_count", ""),
                "final_width_sum": row.get("final_width_sum", ""),
                "last_width_ratio": comp.get("last_width_ratio", ""),
                "tube_width_ratio": comp.get("tube_width_ratio", ""),
                "notes": row.get("notes", ""),
            }
        )
    eligible = [
        row for row in rows
        if row.get("variant_group") != "residual_shift_diagnostic"
        and str(row.get("target_remainder_radius", "")) in {"0.0001", "0.000100000000000000", "1e-04", "1e-4", "0.00010000000000000000"}
        and int(row.get("h_below_flowstar_min_count") or 0) == 0
    ]
    best = max(eligible or [row for row in rows if row.get("variant_group") != "residual_shift_diagnostic"] or rows, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0)
    reached = (_finite_float(best.get("last_validated_t")) or 0.0) >= 5.0 - 1e-9
    trunc_rows = [row for row in rows if row.get("variant_group") in {"truncation_range_split", "candidate_order_truncation_split"}]
    trunc_best_t = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in trunc_rows), default=0.0)
    if reached:
        recommendation = "run h10 next with the best horizon-5 variant, while keeping width and Flow* comparison checks enabled."
    elif trunc_best_t > 2.2771582567640953:
        recommendation = "continue tighter polynomial range bounding because it improved the validated horizon before moving to a symbolic remainder queue."
    else:
        recommendation = "move to a real Flow*-style symbolic remainder queue; the tested tighter range variants did not clear the bottleneck."

    out_dir = REPO_ROOT / "outputs" / "flowstar_style_rescue_next2"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rescue_next2_summary.csv", NEXT_FIELDS, rows)
    lines = [
        "# Rescue Variant Comparison Next2",
        "",
        f"Best variant by decision criteria: `{best.get('run_id', '')}` at t=`{best.get('last_validated_t', '')}`.",
        f"Reached horizon 5 with target_remainder_radius=1e-4? {_yes_no(reached)}.",
        f"Width ratio vs Flow*: last=`{best.get('last_width_ratio', '')}`, tube=`{best.get('tube_width_ratio', '')}`.",
        f"Next recommendation: {recommendation}",
        "",
        "Residual-shift rows are diagnostic only and are not treated as accepted reachability runs.",
        "Decision criteria: reaches horizon 5, no non-final h below 0.002, width ratio not worse than adaptive fallback, runtime, and no fake parity claims.",
        "",
        "## Rows",
        "",
        "| group | run_id | status | last_validated_t | candidate_order | output_order | split | last_width_ratio | tube_width_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant_group', '')} | {row.get('run_id', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('candidate_order', '')} | {row.get('output_order', '')} | "
            f"{row.get('truncation_range_split', '')} | {row.get('last_width_ratio', '')} | {row.get('tube_width_ratio', '')} |"
        )
    (out_dir / "rescue_next2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_rescue_next3_outputs(*, trigger_out_dir: Path | None = None) -> None:
    if trigger_out_dir is not None:
        try:
            outputs_root = (REPO_ROOT / "outputs").resolve()
            if not trigger_out_dir.resolve().is_relative_to(outputs_root):
                return
        except Exception:
            return
    candidates: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Mapping[str, Any]] = {}
    sources = [
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "adaptive_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_candidate_order" / "candidate_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_candidate_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_truncation_range" / "truncation_range_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_truncation_range" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_residual_centering" / "residual_centering_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_residual_centering" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_selective_terms" / "selective_terms_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_selective_terms" / "rescue_vs_flowstar_comparison.csv"),
    ]
    for summary_path, comparison_path in sources:
        for row in _read_optional_csv(summary_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                candidates[run_id] = dict(row)
        for row in _read_optional_csv(comparison_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                comparisons[run_id] = row
    if not candidates:
        return

    rows: list[dict[str, Any]] = []
    for run_id, row in sorted(candidates.items(), key=lambda item: (_variant_group(item[0]), item[0])):
        comp = comparisons.get(run_id, {})
        rows.append(
            {
                "variant_group": _variant_group(run_id),
                "run_id": run_id,
                "validation_mode": row.get("validation_mode", ""),
                "target_remainder_radius": row.get("target_remainder_radius", ""),
                "cutoff_threshold": row.get("cutoff_threshold", ""),
                "status": row.get("status", ""),
                "last_validated_t": row.get("last_validated_t", ""),
                "runtime_s": row.get("runtime_s", ""),
                "num_accepted_steps": row.get("num_accepted_steps", ""),
                "num_rejected_steps": row.get("num_rejected_steps", ""),
                "num_order8_steps": row.get("num_order8_steps", ""),
                "candidate_order": row.get("candidate_order", row.get("order", "")),
                "output_order": row.get("output_order", row.get("order", "")),
                "truncation_range_split": row.get("truncation_range_split", ""),
                "center_corrections_applied": row.get("center_corrections_applied", ""),
                "center_corrected_dimensions": row.get("center_corrected_dimensions", ""),
                "max_center_correction_abs": row.get("max_center_correction_abs", ""),
                "selective_high_degree_terms_top_k": row.get("selective_high_degree_terms_top_k", ""),
                "max_selective_retained_terms_count": row.get("max_selective_retained_terms_count", ""),
                "max_selective_dropped_remainder_width_sum": row.get("max_selective_dropped_remainder_width_sum", ""),
                "min_regular_h_used": row.get("min_regular_h_used", ""),
                "h_below_flowstar_min_count": row.get("h_below_flowstar_min_count", ""),
                "final_width_sum": row.get("final_width_sum", ""),
                "last_width_ratio": comp.get("last_width_ratio", ""),
                "tube_width_ratio": comp.get("tube_width_ratio", ""),
                "notes": row.get("notes", ""),
            }
        )
    eligible = [
        row for row in rows
        if str(row.get("target_remainder_radius", "")) in {"0.0001", "0.000100000000000000", "1e-04", "1e-4", "0.00010000000000000000"}
        and int(row.get("h_below_flowstar_min_count") or 0) == 0
    ]
    best = max(eligible or rows, key=lambda r: (_finite_float(r.get("last_validated_t")) or 0.0, -(_finite_float(r.get("tube_width_ratio")) or math.inf)))
    reached = (_finite_float(best.get("last_validated_t")) or 0.0) >= 5.0 - 1e-9
    candidate_baseline = next((row for row in rows if row.get("run_id") == "flowstar_style_o6_candidate8_output6"), {})
    candidate_tube = _finite_float(candidate_baseline.get("tube_width_ratio"))
    best_tube = _finite_float(best.get("tube_width_ratio"))
    width_ok = best_tube is None or candidate_tube is None or best_tube <= candidate_tube or reached
    if reached:
        recommendation = "run h10 only after reviewing the horizon-5 width and Flow* comparison artifacts."
    elif _variant_group(str(best.get("run_id", ""))) in {"residual_centering", "selective_terms_centered"}:
        recommendation = "continue residual-centering refinement or selective sparse over-order terms before h10."
    elif _variant_group(str(best.get("run_id", ""))) == "selective_high_degree_terms":
        recommendation = "continue selective sparse over-order terms, then compare against a real Flow*-style symbolic remainder queue."
    else:
        recommendation = "choose between residual-centering refinement, selective sparse over-order terms, or a real Flow*-style symbolic remainder queue."

    out_dir = REPO_ROOT / "outputs" / "flowstar_style_rescue_next3"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rescue_next3_summary.csv", NEXT3_FIELDS, rows)
    lines = [
        "# Rescue Variant Comparison Next3",
        "",
        f"Best variant by decision criteria: `{best.get('run_id', '')}` at t=`{best.get('last_validated_t', '')}`.",
        f"Reached horizon 5 with target_remainder_radius=1e-4? {_yes_no(reached)}.",
        f"Width ratio vs Flow*: last=`{best.get('last_width_ratio', '')}`, tube=`{best.get('tube_width_ratio', '')}`.",
        f"Width criterion vs candidate_order baseline acceptable? {_yes_no(width_ok)}.",
        f"Target remainder stayed at 1e-4? {_yes_no(str(best.get('target_remainder_radius', '')) in {'0.0001', '0.000100000000000000', '1e-04', '1e-4', '0.00010000000000000000'})}.",
        f"Next recommendation: {recommendation}",
        "",
        "This comparison is diagnostic-only and does not claim Flow* parity.",
        "Decision criteria: reaches horizon 5, no non-final h below 0.002, target remainder 1e-4, width ratio not worse than candidate_order baseline unless horizon improves substantially, and acceptable runtime.",
        "",
        "## Rows",
        "",
        "| group | run_id | status | last_validated_t | candidate_order | output_order | K | corrections | last_width_ratio | tube_width_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant_group', '')} | {row.get('run_id', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('candidate_order', '')} | {row.get('output_order', '')} | "
            f"{row.get('selective_high_degree_terms_top_k', '')} | {row.get('center_corrections_applied', '')} | "
            f"{row.get('last_width_ratio', '')} | {row.get('tube_width_ratio', '')} |"
        )
    (out_dir / "rescue_next3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    _write_csv(out_dir / "rescue_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "rescue_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "rescue_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    write_report(out_dir, summary_rows, segment_rows, max_horizon=max_horizon)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_style_rescue"))
    parser.add_argument("--max-horizon", type=float, default=1.0)
    parser.add_argument("--wall-cap-s", type=float, default=300.0)
    parser.add_argument("--configs", nargs="*", default=None, help="Run only selected config run_id values.")
    args = parser.parse_args(argv)
    run_experiment(
        args.out_dir,
        max_horizon=float(args.max_horizon),
        wall_cap_s=float(args.wall_cap_s),
        config_ids=args.configs,
    )
    print(f"wrote rescue outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
