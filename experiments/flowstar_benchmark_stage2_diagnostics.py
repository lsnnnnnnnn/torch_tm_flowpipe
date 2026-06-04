#!/usr/bin/env python3
"""Stage-2 diagnostic localization for PyTorch Taylor-model Van der Pol failures."""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import torch

torch.set_num_threads(1)

from torch_tm_flowpipe import Interval, TMVector, flowpipe_step_from_tm, taylor_model_mul_breakdown
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

from flowstar_benchmark_failure_diagnostics import (  # noqa: E402
    StepTimeout,
    _box_is_finite,
    _call_with_timeout,
    _finite_float,
    _max_field,
    _read_csv,
    _reference_substeps,
    _segment_bounds,
    load_reference_inputs,
)
from flowstar_benchmark_parity import _fmt, _initial_box, _write_csv  # noqa: E402

BREAKDOWN_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
    "dependency_window",
    "adaptive_setting",
    "segment_index",
    "segment_t_lo",
    "segment_t_hi",
    "attempt_index",
    "expression",
    "kept_poly_range_width",
    "dropped_trunc_width",
    "p_self_times_other_remainder_width",
    "p_other_times_self_remainder_width",
    "remainder_times_remainder_width",
    "total_remainder_width",
    "output_total_range_width",
    "finite",
    "rhs_y_poly_range_width",
    "rhs_y_remainder_width",
    "rhs_y_total_range_width",
    "notes",
]
DEPENDENCY_WINDOW_SUMMARY_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
    "dependency_window",
    "status",
    "runtime_s",
    "validated_segments",
    "last_validated_t",
    "last_attempted_t",
    "failed_segment_index",
    "failure_reason",
    "final_validated_width_sum",
    "max_validated_width_sum",
    "max_residual_width_sum",
    "max_remainder_width_sum",
    "max_polynomial_range_width_sum",
    "notes",
]
DEPENDENCY_WINDOW_SEGMENT_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
    "dependency_window",
    "segment_index",
    "reference_segment_index",
    "substep_index",
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
    "final_width_x",
    "final_width_y",
    "final_width_sum",
    "reset_after_segment",
    "message",
]
ADAPTIVE_SUMMARY_FIELDS = [
    "run_id",
    "mode",
    "order",
    "dependency_window",
    "max_bisection_depth",
    "status",
    "runtime_s",
    "last_validated_t",
    "last_attempted_t",
    "accepted_microsteps",
    "min_h_used",
    "max_bisection_depth_hit_count",
    "failed_segment_index",
    "failure_reason",
    "notes",
]
ADAPTIVE_SEGMENT_FIELDS = [
    "run_id",
    "mode",
    "order",
    "dependency_window",
    "max_bisection_depth",
    "segment_index",
    "reference_segment_index",
    "substep_index",
    "adaptive_depth",
    "status",
    "validation_attempts",
    "t_lo",
    "t_hi",
    "h",
    "accepted_microstep_index",
    "x_lo",
    "x_hi",
    "y_lo",
    "y_hi",
    "width_x",
    "width_y",
    "width_sum",
    "final_width_x",
    "final_width_y",
    "final_width_sum",
    "message",
]
SENSITIVITY_FIELDS = [
    "run_id",
    "base_case",
    "mode",
    "order",
    "substep_factor",
    "dependency_window",
    "max_validation_attempts",
    "growth_factor",
    "status",
    "runtime_s",
    "validated_segments",
    "last_validated_t",
    "last_attempted_t",
    "failed_segment_index",
    "failure_reason",
    "final_validated_width_sum",
    "max_validated_width_sum",
    "max_residual_width_sum",
    "max_remainder_width_sum",
    "max_polynomial_range_width_sum",
    "notes",
]
TRACE_FIELDS = [
    "run_id",
    "segment_index",
    "t_lo",
    "t_hi",
    "status",
    "final_width_x",
    "final_width_y",
    "final_width_sum",
    "polynomial_range_width_x",
    "polynomial_range_width_y",
    "polynomial_range_width_sum",
    "remainder_width_x",
    "remainder_width_y",
    "remainder_width_sum",
    "residual_width_x",
    "residual_width_y",
    "residual_width_sum",
    "validation_attempts",
    "failure_reason",
]
FLOWSTAR_RATIO_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
    "torch_segment_index",
    "torch_t_hi",
    "torch_width_sum",
    "flowstar_nearest_segment_index",
    "flowstar_t_hi",
    "flowstar_width_sum",
    "torch_over_flowstar_width_ratio",
]
REPORT_RUN_IDS = [
    "range_only_o4_s1",
    "range_only_o6_s4",
    "dependency_preserving_o4_s1",
    "dependency_preserving_o6_s1",
]
BASELINE_BEST_FIXED_T = 0.7661635


def _width_value(iv: Interval) -> float:
    return float(iv.width().detach().cpu())


def _window_label(window: int | float | str) -> str:
    if isinstance(window, str):
        return window
    if math.isinf(float(window)):
        return "inf"
    return str(int(window))


def _window_number(label: str) -> float:
    return math.inf if label == "inf" else float(label)


def _window_sort_key(label: Any) -> float:
    return 1e9 if str(label) == "inf" else float(label)


def _window_mode(label: str) -> str:
    if label == "1":
        return "dependency_window_1"
    if label == "inf":
        return "dependency_window_inf"
    return f"dependency_window_{label}"


def _safe_segment_bounds(box: Sequence[Interval] | None) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if box is None:
        return "", "", "", "", "", "", ""
    try:
        return _segment_bounds(box)
    except Exception:
        return "", "", "", "", "", "", ""


def _summary_from_rows(
    run_id: str,
    mode: str,
    order: int,
    substep_factor: int,
    dependency_window: str,
    status: str,
    runtime_s: float,
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    last_attempted_t: float | str,
    failed_segment_index: int | str,
    failure_reason: str,
    notes: str,
) -> dict[str, Any]:
    validated = [row for row in segment_rows if row.get("status") == "validated"]
    last_validated_t = float(validated[-1]["t_hi"]) if validated else 0.0
    final_width = validated[-1].get("final_width_sum", validated[-1].get("width_sum", "")) if validated else ""
    return {
        "run_id": run_id,
        "mode": mode,
        "order": int(order),
        "substep_factor": int(substep_factor),
        "dependency_window": dependency_window,
        "status": status,
        "runtime_s": runtime_s,
        "validated_segments": len(validated),
        "last_validated_t": last_validated_t,
        "last_attempted_t": last_attempted_t,
        "failed_segment_index": failed_segment_index,
        "failure_reason": failure_reason,
        "final_validated_width_sum": final_width,
        "max_validated_width_sum": _max_field(validated, "width_sum"),
        "max_residual_width_sum": _max_field(attempt_rows, "residual_width_sum"),
        "max_remainder_width_sum": _max_field(attempt_rows, "remainder_width_sum"),
        "max_polynomial_range_width_sum": _max_field(attempt_rows, "polynomial_range_width_sum"),
        "notes": notes,
    }


def _base_breakdown_context(context: Mapping[str, Any], expression: str) -> dict[str, Any]:
    return {
        "run_id": context.get("run_id", ""),
        "mode": context.get("mode", ""),
        "order": context.get("order", ""),
        "substep_factor": context.get("substep_factor", ""),
        "dependency_window": context.get("dependency_window", ""),
        "adaptive_setting": context.get("adaptive_setting", ""),
        "segment_index": context.get("segment_index", ""),
        "segment_t_lo": context.get("t_lo", ""),
        "segment_t_hi": context.get("t_hi", ""),
        "attempt_index": context.get("attempt_index", ""),
        "expression": expression,
        "notes": "",
    }


def _blank_breakdown_widths() -> dict[str, Any]:
    return {
        "kept_poly_range_width": "",
        "dropped_trunc_width": "",
        "p_self_times_other_remainder_width": "",
        "p_other_times_self_remainder_width": "",
        "remainder_times_remainder_width": "",
        "total_remainder_width": "",
        "output_total_range_width": "",
        "finite": "",
        "rhs_y_poly_range_width": "",
        "rhs_y_remainder_width": "",
        "rhs_y_total_range_width": "",
    }


def _mul_breakdown_row(context: Mapping[str, Any], expression: str, breakdown: Mapping[str, Any]) -> dict[str, Any]:
    row = _base_breakdown_context(context, expression)
    row.update(_blank_breakdown_widths())
    for field in [
        "kept_poly_range_width",
        "dropped_trunc_width",
        "p_self_times_other_remainder_width",
        "p_other_times_self_remainder_width",
        "remainder_times_remainder_width",
        "total_remainder_width",
        "output_total_range_width",
        "finite",
    ]:
        row[field] = breakdown.get(field, "")
    return row


def vdp_rhs_breakdown_rows(candidate: TMVector, order: int, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        x = candidate[0]
        y = candidate[1]
        x_sq_breakdown = taylor_model_mul_breakdown(x, x, order)
        x_sq = x * x
        rows.append(_mul_breakdown_row(context, "x_sq", x_sq_breakdown))

        x_sq_y_breakdown = taylor_model_mul_breakdown(x_sq, y, order)
        x_sq_y = x_sq * y
        rows.append(_mul_breakdown_row(context, "x_sq_y", x_sq_y_breakdown))

        rhs_y = y - x - x_sq_y
        poly_range = rhs_y.polynomial.evaluate_interval(rhs_y.domain)
        remainder = rhs_y.remainder
        total_range = rhs_y.range_box()
        finite = (
            poly_range.is_finite()
            and remainder.is_finite()
            and total_range.is_finite()
            and math.isfinite(_width_value(poly_range))
            and math.isfinite(_width_value(remainder))
            and math.isfinite(_width_value(total_range))
        )
        row = _base_breakdown_context(context, "rhs_y")
        row.update(_blank_breakdown_widths())
        row.update(
            {
                "kept_poly_range_width": _width_value(poly_range),
                "total_remainder_width": _width_value(remainder),
                "output_total_range_width": _width_value(total_range),
                "finite": finite,
                "rhs_y_poly_range_width": _width_value(poly_range),
                "rhs_y_remainder_width": _width_value(remainder),
                "rhs_y_total_range_width": _width_value(total_range),
                "notes": "rhs aggregation for y - x - x*x*y",
            }
        )
        rows.append(row)
    except Exception as exc:
        row = _base_breakdown_context(context, "vdp_rhs")
        row.update(_blank_breakdown_widths())
        row["finite"] = False
        row["notes"] = f"breakdown exception: {exc}"
        rows.append(row)
    return rows


def _make_breakdown_callback(rows: list[dict[str, Any]]):
    def _callback(candidate: TMVector, order: int, _attempt: int, context: Mapping[str, Any]) -> None:
        rows.extend(vdp_rhs_breakdown_rows(candidate, order, context))

    return _callback


def _segment_row_from_step(
    *,
    run_id: str,
    mode: str,
    order: int,
    substep_factor: int,
    dependency_window: str,
    segment_index: int,
    reference_segment_index: int,
    substep_index: int,
    t_lo: float,
    t_hi: float,
    seg: Any,
    reset_after_segment: bool = False,
    status_override: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Interval] | None, str]:
    box = None
    final_box = None
    message = getattr(seg, "message", "")
    try:
        box = seg.tm.range_box()
        final_box = seg.final_tm.range_box()
    except Exception as exc:
        message = message or f"range extraction exception: {exc}"
    finite_box = box is not None and final_box is not None and _box_is_finite(box) and _box_is_finite(final_box)
    row_status = "validated" if getattr(seg, "status", "") == "validated" and finite_box else "failed"
    if getattr(seg, "status", "") == "validated" and not finite_box:
        message = message or "non-finite interval in segment or final Taylor model"
    if status_override is not None:
        row_status = status_override
    x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = _safe_segment_bounds(box)
    _fx_lo, _fx_hi, _fy_lo, _fy_hi, final_width_x, final_width_y, final_width_sum = _safe_segment_bounds(final_box)
    row: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "order": int(order),
        "substep_factor": int(substep_factor),
        "dependency_window": dependency_window,
        "segment_index": segment_index,
        "reference_segment_index": reference_segment_index,
        "substep_index": substep_index,
        "status": row_status,
        "validation_attempts": getattr(seg, "validation_attempts", ""),
        "t_lo": t_lo,
        "t_hi": t_hi,
        "h": t_hi - t_lo,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "width_x": width_x,
        "width_y": width_y,
        "width_sum": width_sum,
        "final_width_x": final_width_x,
        "final_width_y": final_width_y,
        "final_width_sum": final_width_sum,
        "reset_after_segment": reset_after_segment,
        "message": message,
    }
    if extra:
        row.update(extra)
    return row, final_box, row_status


def run_dependency_window_diagnostic(
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
    *,
    order: int,
    substep_factor: int,
    dependency_window: int | float | str,
    max_wall_s_per_run: float,
    max_horizon: float,
    max_validation_attempts: int = 20,
    growth_factor: float = 1.25,
    run_id: str | None = None,
    mode: str | None = None,
    adaptive_setting: str = "",
    collect_breakdowns: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    window_label = _window_label(dependency_window)
    window_number = _window_number(window_label)
    mode = mode or _window_mode(window_label)
    run_id = run_id or f"{mode}_o{order}_s{substep_factor}"
    steps = _reference_substeps(reference_segments, substep_factor, max_horizon)
    current_box = _initial_box(params)
    current_tm = TMVector.identity(current_box, order=order)
    accepted_since_reset = 0
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    callback = _make_breakdown_callback(breakdown_rows) if collect_breakdowns else None
    status = "completed"
    failure_reason = ""
    failed_segment_index: int | str = ""
    last_attempted_t: float | str = ""
    notes = "dependency reset/window diagnostic on the original Flow* segment grid"
    start = time.perf_counter()

    for segment_index, (ref_index, substep_index, t_lo, t_hi) in enumerate(steps):
        elapsed = time.perf_counter() - start
        if elapsed >= max_wall_s_per_run:
            status = "timeout"
            failure_reason = f"wall-time cap reached before attempting segment {segment_index}"
            notes = failure_reason
            break
        h = t_hi - t_lo
        last_attempted_t = t_hi
        context = {
            "run_id": run_id,
            "mode": mode,
            "order": order,
            "substep_factor": substep_factor,
            "dependency_window": window_label,
            "adaptive_setting": adaptive_setting,
            "segment_index": segment_index,
            "reference_segment_index": ref_index,
            "substep_index": substep_index,
            "t_lo": t_lo,
            "t_hi": t_hi,
        }
        remaining_s = max_wall_s_per_run - (time.perf_counter() - start)
        try:
            seg = _call_with_timeout(
                lambda: flowpipe_step_from_tm(
                    van_der_pol_ode,
                    current_tm,
                    h,
                    order,
                    max_validation_attempts=max_validation_attempts,
                    growth_factor=growth_factor,
                    diagnostics=attempt_rows,
                    diagnostics_context=context,
                    rhs_breakdown_callback=callback,
                ),
                remaining_s,
            )
        except StepTimeout as exc:
            status = "timeout"
            failed_segment_index = segment_index
            failure_reason = str(exc)
            notes = f"wall-time cap reached while attempting segment {segment_index}"
            break

        accepted_since_reset_next = accepted_since_reset + 1
        reset_after_segment = math.isfinite(window_number) and accepted_since_reset_next >= window_number
        row, final_box, row_status = _segment_row_from_step(
            run_id=run_id,
            mode=mode,
            order=order,
            substep_factor=substep_factor,
            dependency_window=window_label,
            segment_index=segment_index,
            reference_segment_index=ref_index,
            substep_index=substep_index,
            t_lo=t_lo,
            t_hi=t_hi,
            seg=seg,
            reset_after_segment=reset_after_segment and getattr(seg, "status", "") == "validated",
        )
        segment_rows.append(row)
        if row_status != "validated":
            failed_segment_index = segment_index
            failure_reason = row.get("message") or "validation failed"
            status = "timeout" if "wall-time cap" in str(failure_reason) else "failed"
            notes = f"stopped on first failed dependency-window substep {segment_index}"
            break

        if reset_after_segment:
            current_box = [iv.inflate(1e-9) for iv in (final_box or seg.final_tm.range_box())]
            current_tm = TMVector.identity(current_box, order=order)
            accepted_since_reset = 0
        else:
            current_tm = seg.final_tm
            accepted_since_reset = accepted_since_reset_next

        if time.perf_counter() - start >= max_wall_s_per_run and t_hi < max_horizon - 1e-15:
            status = "timeout"
            failure_reason = f"wall-time cap reached after validating segment {segment_index}"
            notes = failure_reason
            break

    runtime_s = time.perf_counter() - start
    if status == "completed":
        if segment_rows and float(segment_rows[-1]["t_hi"]) >= max_horizon - 1e-12:
            status = "max_horizon_reached"
            notes = "validated to the requested stage-2 horizon"
        elif not steps:
            status = "max_horizon_reached"
            notes = "no positive diagnostic substeps were available"
        else:
            status = "max_horizon_reached"
            notes = "reference grid ended before requested diagnostic horizon"
    if last_attempted_t == "":
        last_attempted_t = float(segment_rows[-1]["t_hi"]) if segment_rows else 0.0

    summary = _summary_from_rows(
        run_id,
        mode,
        order,
        substep_factor,
        window_label,
        status,
        runtime_s,
        segment_rows,
        attempt_rows,
        last_attempted_t,
        failed_segment_index,
        failure_reason,
        notes,
    )
    return summary, segment_rows, attempt_rows, breakdown_rows


def _run_window_spec(spec: Mapping[str, Any]):
    return run_dependency_window_diagnostic(**spec)


def run_dependency_window_sweep(
    out_dir: Path,
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    max_wall_s_per_run: float,
    max_horizon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [
        {
            "params": dict(params),
            "reference_segments": list(reference_segments),
            "order": order,
            "substep_factor": substep_factor,
            "dependency_window": window,
            "max_wall_s_per_run": max_wall_s_per_run,
            "max_horizon": max_horizon,
        }
        for order in (4, 6)
        for substep_factor in (1, 2, 4)
        for window in (1, 2, 4, 8, "inf")
    ]
    run_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    workers = max(1, min(int(workers), len(specs) or 1))
    if workers == 1:
        iterator = (_run_window_spec(spec) for spec in specs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(_run_window_spec, spec) for spec in specs]
        iterator = (future.result() for future in as_completed(futures))
    try:
        for summary, segments, _attempts, breakdowns in iterator:
            run_rows.append(summary)
            segment_rows.extend(segments)
            breakdown_rows.extend(breakdowns)
            _write_dependency_window_outputs(out_dir, run_rows, segment_rows)
    finally:
        if workers != 1:
            pool.shutdown(wait=True)
    _write_dependency_window_outputs(out_dir, run_rows, segment_rows)
    return run_rows, segment_rows, breakdown_rows


def _write_dependency_window_outputs(out_dir: Path, run_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> None:
    run_rows.sort(key=lambda r: (int(r.get("order", 0)), int(r.get("substep_factor", 0)), _window_sort_key(r.get("dependency_window", "inf"))))
    segment_rows.sort(key=lambda r: (str(r.get("run_id", "")), int(r.get("segment_index", 0))))
    _write_csv(out_dir / "dependency_window_summary.csv", DEPENDENCY_WINDOW_SUMMARY_FIELDS, run_rows)
    _write_csv(out_dir / "dependency_window_segments.csv", DEPENDENCY_WINDOW_SEGMENT_FIELDS, segment_rows)


def run_adaptive_bisection_diagnostic(
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    order: int,
    max_bisection_depth: int,
    max_wall_s_per_run: float,
    max_horizon: float,
    min_h: float,
    collect_breakdowns: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    window_label = {"range_only": "1", "dependency_window_2": "2", "dependency_window_4": "4"}[mode]
    window_number = _window_number(window_label)
    run_id = f"{mode}_o{order}_b{max_bisection_depth}"
    steps = _reference_substeps(reference_segments, 1, max_horizon)
    current_tm = TMVector.identity(_initial_box(params), order=order)
    accepted_since_reset = 0
    accepted_microsteps = 0
    min_h_used = math.inf
    max_depth_hits = 0
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    callback = _make_breakdown_callback(breakdown_rows) if collect_breakdowns else None
    status = "completed"
    failure_reason = ""
    failed_segment_index: int | str = ""
    last_attempted_t: float | str = ""
    start = time.perf_counter()

    def attempt_step(ref_index: int, substep_index: int, t_lo: float, t_hi: float, depth: int):
        nonlocal current_tm, accepted_since_reset, accepted_microsteps, min_h_used
        nonlocal max_depth_hits, status, failure_reason, failed_segment_index, last_attempted_t
        segment_index = len(segment_rows)
        h = t_hi - t_lo
        last_attempted_t = t_hi
        if time.perf_counter() - start >= max_wall_s_per_run:
            status = "timeout"
            failure_reason = f"wall-time cap reached before adaptive interval [{t_lo}, {t_hi}]"
            failed_segment_index = segment_index
            return False
        context = {
            "run_id": run_id,
            "mode": mode,
            "order": order,
            "substep_factor": 1,
            "dependency_window": window_label,
            "adaptive_setting": f"max_bisection_depth_{max_bisection_depth}",
            "segment_index": segment_index,
            "reference_segment_index": ref_index,
            "substep_index": substep_index,
            "adaptive_depth": depth,
            "t_lo": t_lo,
            "t_hi": t_hi,
        }
        remaining_s = max_wall_s_per_run - (time.perf_counter() - start)
        try:
            seg = _call_with_timeout(
                lambda: flowpipe_step_from_tm(
                    van_der_pol_ode,
                    current_tm,
                    h,
                    order,
                    diagnostics=attempt_rows,
                    diagnostics_context=context,
                    rhs_breakdown_callback=callback,
                ),
                remaining_s,
            )
        except StepTimeout as exc:
            status = "timeout"
            failure_reason = str(exc)
            failed_segment_index = segment_index
            return False

        accepted_since_reset_next = accepted_since_reset + 1
        reset_after_segment = math.isfinite(window_number) and accepted_since_reset_next >= window_number
        row, final_box, row_status = _segment_row_from_step(
            run_id=run_id,
            mode=mode,
            order=order,
            substep_factor=1,
            dependency_window=window_label,
            segment_index=segment_index,
            reference_segment_index=ref_index,
            substep_index=substep_index,
            t_lo=t_lo,
            t_hi=t_hi,
            seg=seg,
            reset_after_segment=reset_after_segment and getattr(seg, "status", "") == "validated",
            extra={
                "max_bisection_depth": max_bisection_depth,
                "adaptive_depth": depth,
                "accepted_microstep_index": "",
            },
        )
        if row_status == "validated":
            accepted_microsteps += 1
            min_h_used = min(min_h_used, h)
            row["accepted_microstep_index"] = accepted_microsteps
            segment_rows.append(row)
            if reset_after_segment:
                current_tm = TMVector.identity([iv.inflate(1e-9) for iv in (final_box or seg.final_tm.range_box())], order=order)
                accepted_since_reset = 0
            else:
                current_tm = seg.final_tm
                accepted_since_reset = accepted_since_reset_next
            return True

        can_bisect = depth < max_bisection_depth and h * 0.5 >= min_h and time.perf_counter() - start < max_wall_s_per_run
        if can_bisect:
            row["status"] = "rejected_bisected"
            segment_rows.append(row)
            mid = (t_lo + t_hi) / 2.0
            return attempt_step(ref_index, substep_index, t_lo, mid, depth + 1) and attempt_step(ref_index, substep_index, mid, t_hi, depth + 1)

        if depth >= max_bisection_depth:
            max_depth_hits += 1
            failure_reason = row.get("message") or "max bisection depth reached after validation failure"
        elif h * 0.5 < min_h:
            failure_reason = f"minimum h {min_h:g} reached after validation failure"
        else:
            failure_reason = row.get("message") or "adaptive validation failed"
        status = "failed"
        failed_segment_index = segment_index
        segment_rows.append(row)
        return False

    for ref_index, substep_index, t_lo, t_hi in steps:
        if not attempt_step(ref_index, substep_index, t_lo, t_hi, 0):
            break

    runtime_s = time.perf_counter() - start
    last_validated_t = max((float(row["t_hi"]) for row in segment_rows if row.get("status") == "validated"), default=0.0)
    if status == "completed":
        if last_validated_t >= max_horizon - 1e-12:
            status = "max_horizon_reached"
            failure_reason = ""
            notes = "adaptive bisection validated to the requested stage-2 horizon"
        else:
            status = "max_horizon_reached"
            notes = "reference grid ended before requested diagnostic horizon"
    else:
        notes = failure_reason
    if last_attempted_t == "":
        last_attempted_t = last_validated_t
    summary = {
        "run_id": run_id,
        "mode": mode,
        "order": order,
        "dependency_window": window_label,
        "max_bisection_depth": max_bisection_depth,
        "status": status,
        "runtime_s": runtime_s,
        "last_validated_t": last_validated_t,
        "last_attempted_t": last_attempted_t,
        "accepted_microsteps": accepted_microsteps,
        "min_h_used": "" if math.isinf(min_h_used) else min_h_used,
        "max_bisection_depth_hit_count": max_depth_hits,
        "failed_segment_index": failed_segment_index,
        "failure_reason": failure_reason,
        "notes": notes,
    }
    return summary, segment_rows, breakdown_rows


def _run_adaptive_spec(spec: Mapping[str, Any]):
    return run_adaptive_bisection_diagnostic(**spec)


def run_adaptive_bisection_sweep(
    out_dir: Path,
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    max_wall_s_per_run: float,
    max_horizon: float,
    min_h: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [
        {
            "params": dict(params),
            "reference_segments": list(reference_segments),
            "mode": mode,
            "order": order,
            "max_bisection_depth": depth,
            "max_wall_s_per_run": max_wall_s_per_run,
            "max_horizon": max_horizon,
            "min_h": min_h,
        }
        for mode in ("range_only", "dependency_window_2", "dependency_window_4")
        for order in (4, 6)
        for depth in (4, 8)
    ]
    run_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    workers = max(1, min(int(workers), len(specs) or 1))
    if workers == 1:
        iterator = (_run_adaptive_spec(spec) for spec in specs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(_run_adaptive_spec, spec) for spec in specs]
        iterator = (future.result() for future in as_completed(futures))
    try:
        for summary, segments, breakdowns in iterator:
            run_rows.append(summary)
            segment_rows.extend(segments)
            breakdown_rows.extend(breakdowns)
            _write_adaptive_outputs(out_dir, run_rows, segment_rows)
    finally:
        if workers != 1:
            pool.shutdown(wait=True)
    _write_adaptive_outputs(out_dir, run_rows, segment_rows)
    return run_rows, segment_rows, breakdown_rows


def _write_adaptive_outputs(out_dir: Path, run_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> None:
    run_rows.sort(key=lambda r: (str(r.get("mode", "")), int(r.get("order", 0)), int(r.get("max_bisection_depth", 0))))
    segment_rows.sort(key=lambda r: (str(r.get("run_id", "")), int(r.get("segment_index", 0))))
    _write_csv(out_dir / "adaptive_bisection_summary.csv", ADAPTIVE_SUMMARY_FIELDS, run_rows)
    _write_csv(out_dir / "adaptive_bisection_segments.csv", ADAPTIVE_SEGMENT_FIELDS, segment_rows)


def run_validation_sensitivity(
    out_dir: Path,
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    max_wall_s_per_run: float,
    max_horizon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_cases = [
        {"base_case": "range_only_o8_s1", "mode": "range_only", "order": 8, "substep_factor": 1, "dependency_window": 1},
        {"base_case": "dependency_preserving_o6_s1", "mode": "dependency_preserving", "order": 6, "substep_factor": 1, "dependency_window": "inf"},
    ]
    specs = []
    for base in base_cases:
        for max_attempts in (20, 50, 100):
            for growth_factor in (1.25, 1.5, 2.0):
                gf_label = str(growth_factor).replace(".", "p")
                run_id = f"{base['base_case']}_a{max_attempts}_g{gf_label}"
                specs.append(
                    {
                        "params": dict(params),
                        "reference_segments": list(reference_segments),
                        "order": base["order"],
                        "substep_factor": base["substep_factor"],
                        "dependency_window": base["dependency_window"],
                        "max_wall_s_per_run": max_wall_s_per_run,
                        "max_horizon": max_horizon,
                        "max_validation_attempts": max_attempts,
                        "growth_factor": growth_factor,
                        "run_id": run_id,
                        "mode": base["mode"],
                        "adaptive_setting": f"max_attempts_{max_attempts}_growth_{growth_factor}",
                    }
                )
    rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    workers = max(1, min(int(workers), len(specs) or 1))
    if workers == 1:
        iterator = (_run_window_spec(spec) for spec in specs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(_run_window_spec, spec) for spec in specs]
        iterator = (future.result() for future in as_completed(futures))
    try:
        for summary, _segments, _attempts, breakdowns in iterator:
            run_id = str(summary["run_id"])
            base_case, sensitivity_suffix = run_id.rsplit("_a", 1)
            max_attempts_text, growth_text = sensitivity_suffix.split("_g", 1)
            max_attempts = int(max_attempts_text)
            growth_factor = float(growth_text.replace("p", "."))
            rows.append(
                {
                    **summary,
                    "base_case": base_case,
                    "max_validation_attempts": max_attempts,
                    "growth_factor": growth_factor,
                }
            )
            breakdown_rows.extend(breakdowns)
            _write_sensitivity_outputs(out_dir, rows)
    finally:
        if workers != 1:
            pool.shutdown(wait=True)
    _write_sensitivity_outputs(out_dir, rows)
    return rows, breakdown_rows


def _write_sensitivity_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda r: (str(r.get("base_case", "")), int(r.get("max_validation_attempts", 0)), float(r.get("growth_factor", 0.0))))
    _write_csv(out_dir / "validation_parameter_sensitivity.csv", SENSITIVITY_FIELDS, rows)


def _latest_attempt_by_segment(attempt_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in attempt_rows:
        key = (str(row.get("run_id", "")), str(row.get("segment_index", "")))
        prev = out.get(key)
        if prev is None or int(row.get("attempt_index") or 0) >= int(prev.get("attempt_index") or 0):
            out[key] = row
    return out


def write_pre_failure_trace(out_dir: Path, diagnostics_dir: Path, *, n: int = 10) -> list[dict[str, Any]]:
    segment_rows = _read_csv(diagnostics_dir / "diagnostic_segments.csv")
    attempt_rows = _read_csv(diagnostics_dir / "diagnostic_validation_attempts.csv")
    summary_rows = {row["run_id"]: row for row in _read_csv(diagnostics_dir / "diagnostic_runs_summary.csv")}
    latest_attempt = _latest_attempt_by_segment(attempt_rows)
    by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        by_run[str(row["run_id"])].append(row)
    trace_rows: list[dict[str, Any]] = []
    for run_id, rows in sorted(by_run.items()):
        rows.sort(key=lambda r: int(r.get("segment_index") or 0))
        validated = [row for row in rows if row.get("status") == "validated"]
        failed = [row for row in rows if row.get("status") != "validated"]
        selected = validated[-n:] + (failed[:1] if failed else [])
        for row in selected:
            segment_index = str(row.get("segment_index", ""))
            attempt = latest_attempt.get((run_id, segment_index), {})
            status = row.get("status", "")
            failure_reason = ""
            if status != "validated":
                failure_reason = row.get("message") or attempt.get("validation_message") or summary_rows.get(run_id, {}).get("failure_reason", "")
            trace_rows.append(
                {
                    "run_id": run_id,
                    "segment_index": segment_index,
                    "t_lo": row.get("t_lo", ""),
                    "t_hi": row.get("t_hi", ""),
                    "status": status,
                    "final_width_x": attempt.get("candidate_final_width_x") or row.get("width_x", ""),
                    "final_width_y": attempt.get("candidate_final_width_y") or row.get("width_y", ""),
                    "final_width_sum": attempt.get("candidate_final_width_sum") or row.get("width_sum", ""),
                    "polynomial_range_width_x": attempt.get("polynomial_range_width_x", ""),
                    "polynomial_range_width_y": attempt.get("polynomial_range_width_y", ""),
                    "polynomial_range_width_sum": attempt.get("polynomial_range_width_sum", ""),
                    "remainder_width_x": attempt.get("remainder_width_x", ""),
                    "remainder_width_y": attempt.get("remainder_width_y", ""),
                    "remainder_width_sum": attempt.get("remainder_width_sum", ""),
                    "residual_width_x": attempt.get("residual_width_x", ""),
                    "residual_width_y": attempt.get("residual_width_y", ""),
                    "residual_width_sum": attempt.get("residual_width_sum", ""),
                    "validation_attempts": row.get("validation_attempts") or attempt.get("attempt_index", ""),
                    "failure_reason": failure_reason,
                }
            )
    _write_csv(out_dir / "pre_failure_trace.csv", TRACE_FIELDS, trace_rows)
    write_pre_failure_markdown(out_dir, trace_rows)
    return trace_rows


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ["segment_index", "t_lo", "t_hi", "status", "final_width_sum", "polynomial_range_width_sum", "remainder_width_sum", "residual_width_sum", "validation_attempts", "failure_reason"]
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(_fmt(row.get(field, ""))) for field in fields) + " |")
    return "\n".join(lines)


def write_pre_failure_markdown(out_dir: Path, trace_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Stage-2 Pre-Failure Trace",
        "",
        "This is diagnostic-only. Each table lists the last 10 validated rows plus the failed attempted segment when present.",
    ]
    for run_id in REPORT_RUN_IDS:
        rows = [row for row in trace_rows if row.get("run_id") == run_id]
        lines.extend(["", f"## `{run_id}`", "", _markdown_table(rows) if rows else "No trace rows available."])
    (out_dir / "pre_failure_trace.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_flowstar_width_ratios(out_dir: Path, diagnostics_dir: Path, parity_dir: Path) -> list[dict[str, Any]]:
    torch_rows = _read_csv(diagnostics_dir / "diagnostic_segments.csv")
    flowstar_rows = _read_csv(parity_dir / "original_flowstar" / "original_flowstar_segments.csv")
    ratios: list[dict[str, Any]] = []
    for row in torch_rows:
        torch_t_hi = _finite_float(row.get("t_hi"))
        torch_width = _finite_float(row.get("width_sum"))
        if torch_t_hi is None or torch_width is None:
            continue
        containing = [
            ref
            for ref in flowstar_rows
            if _finite_float(ref.get("t_lo")) is not None
            and _finite_float(ref.get("t_hi")) is not None
            and float(ref["t_lo"]) - 1e-14 <= torch_t_hi <= float(ref["t_hi"]) + 1e-14
        ]
        if containing:
            nearest = min(containing, key=lambda ref: abs(float(ref["t_hi"]) - torch_t_hi))
        else:
            nearest = min(flowstar_rows, key=lambda ref: abs(float(ref["t_hi"]) - torch_t_hi))
        flow_width = _finite_float(nearest.get("width_sum"))
        ratio = torch_width / flow_width if flow_width and flow_width > 0 else ""
        ratios.append(
            {
                "run_id": row.get("run_id", ""),
                "mode": row.get("mode", ""),
                "order": row.get("order", ""),
                "substep_factor": row.get("substep_factor", ""),
                "torch_segment_index": row.get("segment_index", ""),
                "torch_t_hi": torch_t_hi,
                "torch_width_sum": torch_width,
                "flowstar_nearest_segment_index": nearest.get("segment_index", ""),
                "flowstar_t_hi": nearest.get("t_hi", ""),
                "flowstar_width_sum": flow_width if flow_width is not None else "",
                "torch_over_flowstar_width_ratio": ratio,
            }
        )
    _write_csv(out_dir / "torch_vs_flowstar_width_trace.csv", FLOWSTAR_RATIO_FIELDS, ratios)
    return ratios


def _read_stage2_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _max_numeric(rows: Sequence[Mapping[str, Any]], field: str, *, expression: str | None = None) -> float:
    vals = []
    for row in rows:
        if expression is not None and row.get("expression") != expression:
            continue
        value = _finite_float(row.get(field))
        if value is not None:
            vals.append(value)
    return max(vals) if vals else 0.0


def _dominant_vdp_term(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, float]]:
    x_sq_rem = max(
        _max_numeric(rows, "p_self_times_other_remainder_width", expression="x_sq"),
        _max_numeric(rows, "p_other_times_self_remainder_width", expression="x_sq"),
        _max_numeric(rows, "remainder_times_remainder_width", expression="x_sq"),
    )
    poly_rem = max(
        _max_numeric(rows, "p_self_times_other_remainder_width", expression="x_sq_y"),
        _max_numeric(rows, "p_other_times_self_remainder_width", expression="x_sq_y"),
    )
    metrics = {
        "x*x truncation": _max_numeric(rows, "dropped_trunc_width", expression="x_sq"),
        "x*x remainder multiplication": x_sq_rem,
        "(x*x)*y truncation": _max_numeric(rows, "dropped_trunc_width", expression="x_sq_y"),
        "polynomial_range * remainder": poly_rem,
        "remainder * remainder": max(
            _max_numeric(rows, "remainder_times_remainder_width", expression="x_sq"),
            _max_numeric(rows, "remainder_times_remainder_width", expression="x_sq_y"),
        ),
        "RHS aggregation": _max_numeric(rows, "rhs_y_remainder_width", expression="rhs_y"),
        "interval polynomial range evaluation": max(
            _max_numeric(rows, "kept_poly_range_width", expression="x_sq"),
            _max_numeric(rows, "kept_poly_range_width", expression="x_sq_y"),
            _max_numeric(rows, "rhs_y_poly_range_width", expression="rhs_y"),
        ),
    }
    dominant = max(metrics, key=metrics.get) if metrics else "interval polynomial range evaluation"
    return dominant, metrics


def _best_by_window(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        label = str(row.get("dependency_window", ""))
        value = _finite_float(row.get("last_validated_t"))
        if value is None:
            continue
        best[label] = max(best.get(label, 0.0), value)
    return best


def _format_best_map(best: Mapping[str, float]) -> str:
    labels = ["1", "2", "4", "8", "inf"]
    return ", ".join(f"K={label}: {_fmt(best.get(label, 0.0))}" for label in labels if label in best)


def _adaptive_answer(rows: Sequence[Mapping[str, Any]]) -> tuple[str, Mapping[str, Any] | None]:
    best_row = None
    best_t = -math.inf
    for row in rows:
        value = _finite_float(row.get("last_validated_t"))
        if value is not None and value > best_t:
            best_t = value
            best_row = row
    if best_row is None:
        return "No adaptive-bisection rows were available.", None
    if best_t > BASELINE_BEST_FIXED_T + 1e-9:
        return (
            f"Yes. Best run `{best_row['run_id']}` reached t={_fmt(best_t)} with "
            f"{best_row.get('accepted_microsteps', '')} accepted microsteps and min h={best_row.get('min_h_used', '')}.",
            best_row,
        )
    return (
        f"No. Best run `{best_row['run_id']}` reached t={_fmt(best_t)}, so width/remainder blowup still dominates before beating t={BASELINE_BEST_FIXED_T}.",
        best_row,
    )


def _sensitivity_answer(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "No validation-parameter sensitivity rows were available."
    parts = []
    for base_case in sorted({str(row.get("base_case", "")) for row in rows}):
        group = [row for row in rows if row.get("base_case") == base_case]
        default = next((row for row in group if row.get("max_validation_attempts") == "20" and row.get("growth_factor") == "1.25"), None)
        best = max(group, key=lambda row: _finite_float(row.get("last_validated_t")) or -math.inf)
        best_t = _finite_float(best.get("last_validated_t")) or 0.0
        default_t = _finite_float(default.get("last_validated_t")) if default else None
        if default_t is None:
            parts.append(f"`{base_case}` best reached t={_fmt(best_t)}.")
        else:
            delta = best_t - default_t
            parts.append(f"`{base_case}` best reached t={_fmt(best_t)} (delta {_fmt(delta)} over default).")
    return " ".join(parts)


def write_stage2_report(out_dir: Path) -> None:
    breakdown_rows = _read_stage2_csv(out_dir / "vdp_rhs_breakdown.csv")
    window_rows = _read_stage2_csv(out_dir / "dependency_window_summary.csv")
    adaptive_rows = _read_stage2_csv(out_dir / "adaptive_bisection_summary.csv")
    sensitivity_rows = _read_stage2_csv(out_dir / "validation_parameter_sensitivity.csv")

    dominant, metrics = _dominant_vdp_term(breakdown_rows)
    best_windows = _best_by_window(window_rows)
    adaptive_text, adaptive_best = _adaptive_answer(adaptive_rows)
    sensitivity_text = _sensitivity_answer(sensitivity_rows)
    timeout_count = sum(1 for row in window_rows if row.get("status") == "timeout")
    long_dependency_hurts = best_windows.get("inf", 0.0) < max(best_windows.get("1", 0.0), best_windows.get("2", 0.0), best_windows.get("4", 0.0), best_windows.get("8", 0.0)) - 1e-12

    if adaptive_best is not None and (_finite_float(adaptive_best.get("last_validated_t")) or 0.0) > BASELINE_BEST_FIXED_T + 1e-9:
        next_target = "adaptive substep fallback"
    elif long_dependency_hurts and best_windows.get("1", 0.0) < max(best_windows.values() or [0.0]) - 1e-12:
        next_target = "dependency reset/windowing"
    elif dominant == "interval polynomial range evaluation":
        next_target = "improved polynomial range bounding"
    elif dominant in {"polynomial_range * remainder", "remainder * remainder", "RHS aggregation"}:
        next_target = "symbolic remainder handling"
    else:
        next_target = "improved polynomial range bounding"

    metric_lines = "\n".join(f"- {name}: max {_fmt(value)}" for name, value in metrics.items())
    window_text = _format_best_map(best_windows) or "no dependency-window rows available"
    dependency_answer = "Yes" if long_dependency_hurts else "No clear monotone penalty in this sweep"
    timeout_note = f" {timeout_count} dependency-window rows hit the wall-time cap before failure; those rows are still useful as capped diagnostics, not success claims." if timeout_count else ""

    text = f"""# Flow* Benchmark PyTorch TM Stage-2 Diagnostics

This report is diagnosis only. It is not a new reachability algorithm and not a Flow* parity claim.

## A. Dominant Van der Pol RHS Blowup Term

Dominant term: **{dominant}**.

{metric_lines}

## B. Dependency Reset Window

{dependency_answer}. Best last-validated times by reset window: {window_text}.{timeout_note}

## C. Adaptive Bisection

{adaptive_text}

## D. Validation Parameter Tuning

{sensitivity_text}

## E. Next Minimal Implementation Target

Pick exactly one: **{next_target}**.
"""
    (out_dir / "diagnostic_stage2_report.md").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-2 PyTorch TM failure-localization diagnostics.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "flowstar_benchmark_diagnostics_stage2"))
    parser.add_argument("--parity-dir", default=str(REPO_ROOT / "outputs" / "flowstar_benchmark_parity"))
    parser.add_argument("--existing-diagnostics-dir", default=str(REPO_ROOT / "outputs" / "flowstar_benchmark_diagnostics"))
    parser.add_argument("--max-horizon", type=float, default=1.0)
    parser.add_argument("--dependency-wall-s", type=float, default=120.0)
    parser.add_argument("--adaptive-wall-s", type=float, default=180.0)
    parser.add_argument("--sensitivity-wall-s", type=float, default=180.0)
    parser.add_argument("--min-h", type=float, default=1e-6)
    parser.add_argument("--trace-n", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--skip-heavy", action="store_true", help="Only regenerate trace and Flow* ratio artifacts.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parity_dir = Path(args.parity_dir)
    diagnostics_dir = Path(args.existing_diagnostics_dir)
    params, reference_segments = load_reference_inputs(parity_dir)

    trace_rows = write_pre_failure_trace(out_dir, diagnostics_dir, n=args.trace_n)
    ratio_rows = write_flowstar_width_ratios(out_dir, diagnostics_dir, parity_dir)
    all_breakdowns: list[dict[str, Any]] = []

    if not args.skip_heavy:
        window_runs, window_segments, window_breakdowns = run_dependency_window_sweep(
            out_dir,
            params,
            reference_segments,
            workers=args.workers,
            max_wall_s_per_run=args.dependency_wall_s,
            max_horizon=args.max_horizon,
        )
        all_breakdowns.extend(window_breakdowns)
        _write_csv(out_dir / "vdp_rhs_breakdown.csv", BREAKDOWN_FIELDS, all_breakdowns)

        adaptive_runs, adaptive_segments, adaptive_breakdowns = run_adaptive_bisection_sweep(
            out_dir,
            params,
            reference_segments,
            workers=args.workers,
            max_wall_s_per_run=args.adaptive_wall_s,
            max_horizon=args.max_horizon,
            min_h=args.min_h,
        )
        all_breakdowns.extend(adaptive_breakdowns)
        _write_csv(out_dir / "vdp_rhs_breakdown.csv", BREAKDOWN_FIELDS, all_breakdowns)

        sensitivity_rows, sensitivity_breakdowns = run_validation_sensitivity(
            out_dir,
            params,
            reference_segments,
            workers=args.workers,
            max_wall_s_per_run=args.sensitivity_wall_s,
            max_horizon=args.max_horizon,
        )
        all_breakdowns.extend(sensitivity_breakdowns)
        _write_csv(out_dir / "vdp_rhs_breakdown.csv", BREAKDOWN_FIELDS, all_breakdowns)
    elif not (out_dir / "vdp_rhs_breakdown.csv").exists():
        _write_csv(out_dir / "vdp_rhs_breakdown.csv", BREAKDOWN_FIELDS, all_breakdowns)

    write_stage2_report(out_dir)
    print(f"wrote {out_dir}")
    print(f"pre_failure_rows={len(trace_rows)} flowstar_ratio_rows={len(ratio_rows)} breakdown_rows={len(all_breakdowns)}")


if __name__ == "__main__":
    main()
