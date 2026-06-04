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
    "validation_mode",
    "cutoff_threshold",
    "target_remainder_radius",
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
    "validation_mode",
    "cutoff_threshold",
    "target_remainder_radius",
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
    "validation_mode",
    "cutoff_threshold",
    "target_remainder_radius",
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


class StepTimeout(RuntimeError):
    pass


def _initial_box() -> list[Interval]:
    return [Interval(1.1, 1.4), Interval(2.35, 2.45)]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
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
    return {
        "run_id": spec["run_id"],
        "mode": spec["mode"],
        "order": spec["order"],
        "validation_mode": spec.get("validation_mode", "growth"),
        "cutoff_threshold": "" if spec.get("cutoff_threshold") is None else spec.get("cutoff_threshold"),
        "target_remainder_radius": spec.get("target_remainder_radius", ""),
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
    return {
        "run_id": spec["run_id"],
        "mode": spec["mode"],
        "order": spec["order"],
        "validation_mode": spec.get("validation_mode", "growth"),
        "cutoff_threshold": "" if spec.get("cutoff_threshold") is None else spec.get("cutoff_threshold"),
        "target_remainder_radius": spec.get("target_remainder_radius", ""),
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
            "validation_mode": spec["validation_mode"],
            "cutoff_threshold": "" if spec.get("cutoff_threshold") is None else spec.get("cutoff_threshold"),
            "target_remainder_radius": spec.get("target_remainder_radius", ""),
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
                    cutoff_threshold=spec.get("cutoff_threshold"),
                    max_validation_attempts=int(spec.get("max_validation_attempts", 2)),
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
        {
            "run_id": "flowstar_style_o4_target",
            "mode": "flowstar_style",
            "order": 4,
            "validation_mode": "target_remainder",
            "target_remainder_radius": 1e-4,
            "cutoff_threshold": None,
            "h_min": 0.002,
            "h_max": 0.1,
            "max_validation_attempts": 2,
            "kind": "adaptive",
        },
        {
            "run_id": "flowstar_style_o6_target",
            "mode": "flowstar_style",
            "order": 6,
            "validation_mode": "target_remainder",
            "target_remainder_radius": 1e-4,
            "cutoff_threshold": None,
            "h_min": 0.002,
            "h_max": 0.1,
            "max_validation_attempts": 2,
            "kind": "adaptive",
        },
        {
            "run_id": "flowstar_style_o4_target_cutoff",
            "mode": "flowstar_style",
            "order": 4,
            "validation_mode": "target_remainder",
            "target_remainder_radius": 1e-4,
            "cutoff_threshold": 1e-10,
            "h_min": 0.002,
            "h_max": 0.1,
            "max_validation_attempts": 2,
            "kind": "adaptive",
        },
        {
            "run_id": "flowstar_style_o6_target_cutoff",
            "mode": "flowstar_style",
            "order": 6,
            "validation_mode": "target_remainder",
            "target_remainder_radius": 1e-4,
            "cutoff_threshold": 1e-10,
            "h_min": 0.002,
            "h_max": 0.1,
            "max_validation_attempts": 2,
            "kind": "adaptive",
        },
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
    target_rows = [r for r in summary_rows if r.get("validation_mode") == "target_remainder"]
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
    return summary_rows, segment_rows, attempt_rows


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
