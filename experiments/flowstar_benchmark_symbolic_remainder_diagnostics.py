#!/usr/bin/env python3
"""Stage-3 symbolic-remainder diagnostics for Van der Pol PyTorch TMs."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import torch

torch.set_num_threads(1)

from torch_tm_flowpipe import (  # noqa: E402
    Interval,
    SymbolicRemainderState,
    TMVector,
    flowpipe_step,
    flowpipe_step_from_tm,
    materialize_non_symbolic_variables,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode  # noqa: E402
from torch_tm_flowpipe.polynomial import Polynomial  # noqa: E402
from torch_tm_flowpipe.symbolic_remainder import (  # noqa: E402
    ordinary_remainder_widths,
    split_polynomial_by_variables,
    symbolic_remainder_widths,
)
from torch_tm_flowpipe.taylor_model import TaylorModel  # noqa: E402

from flowstar_benchmark_failure_diagnostics import (  # noqa: E402
    StepTimeout,
    _box_is_finite,
    _call_with_timeout,
    _finite_float,
    _max_field,
    _reference_substeps,
    _segment_bounds,
    load_reference_inputs,
)
from flowstar_benchmark_parity import _fmt, _initial_box, _write_csv  # noqa: E402

BASELINE_BEST_FIXED_T = 0.7661635

SUMMARY_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
    "dependency_window",
    "symbolic_remainder",
    "queue_size",
    "status",
    "runtime_s",
    "validated_segments",
    "last_validated_t",
    "last_attempted_t",
    "failed_segment_index",
    "failure_reason",
    "final_validated_width_sum",
    "max_width_sum",
    "max_ordinary_remainder_width_sum",
    "max_symbolic_remainder_width_sum",
    "max_materialized_remainder_width_sum",
    "max_active_noise_symbols",
    "notes",
]
SEGMENT_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
    "dependency_window",
    "symbolic_remainder",
    "queue_size",
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
    "ordinary_remainder_width_sum",
    "symbolic_remainder_width_sum",
    "materialized_remainder_width_sum",
    "active_noise_symbols",
    "reset_after_segment",
    "message",
]
BREAKDOWN_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
    "dependency_window",
    "symbolic_remainder",
    "queue_size",
    "segment_index",
    "segment_t_lo",
    "segment_t_hi",
    "attempt_index",
    "expression",
    "ordinary_interval_remainder_width",
    "symbolic_remainder_width",
    "materialized_remainder_width",
    "symbolic_poly_times_noise_width",
    "noise_times_noise_width",
    "truncation_width",
    "total_width",
    "active_noise_symbols",
    "finite",
    "notes",
]
PLOT_NAMES = [
    "symbolic_vs_baseline_last_validated_t.png",
    "symbolic_remainder_width_breakdown.png",
    "active_noise_symbols_vs_t.png",
    "ordinary_vs_symbolic_remainder_width.png",
]


def _width_value(iv: Interval) -> float:
    return float(iv.width().detach().cpu())


def _safe_width_value(iv: Interval) -> float | str:
    try:
        value = _width_value(iv)
    except Exception:
        return ""
    return value if math.isfinite(value) else ""


def _safe_segment_bounds(box: Sequence[Interval] | None) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if box is None:
        return "", "", "", "", "", "", ""
    try:
        return _segment_bounds(box)
    except Exception:
        return "", "", "", "", "", "", ""


def _sum_widths(widths: Iterable[Any]) -> float | str:
    values = []
    for width in widths:
        value = _finite_float(width)
        if value is not None:
            values.append(value)
    return sum(values) if values else ""


def _finite_values(*values: Any) -> bool:
    for value in values:
        if value == "":
            continue
        f = _finite_float(value)
        if f is None:
            return False
    return True


def _poly_interval_width(poly: Polynomial, domain: Sequence[Interval]) -> float | str:
    return _safe_width_value(poly.evaluate_interval(domain))


def _mul_polys_width(a: Polynomial, b: Polynomial, domain: Sequence[Interval]) -> float | str:
    if not a.terms or not b.terms:
        return 0.0
    return _poly_interval_width(a * b, domain)


def _ordinary_remainder_interaction(a: TaylorModel, b: TaylorModel) -> Interval:
    a_poly_range = a.polynomial.evaluate_interval(a.domain)
    b_poly_range = b.polynomial.evaluate_interval(b.domain)
    return (a_poly_range * b.remainder) + (b_poly_range * a.remainder) + (a.remainder * b.remainder)


def _base_breakdown_row(context: Mapping[str, Any], expression: str) -> dict[str, Any]:
    return {
        "run_id": context.get("run_id", ""),
        "mode": context.get("mode", ""),
        "order": context.get("order", ""),
        "substep_factor": context.get("substep_factor", ""),
        "dependency_window": context.get("dependency_window", ""),
        "symbolic_remainder": context.get("symbolic_remainder", False),
        "queue_size": context.get("queue_size", ""),
        "segment_index": context.get("segment_index", ""),
        "segment_t_lo": context.get("t_lo", ""),
        "segment_t_hi": context.get("t_hi", ""),
        "attempt_index": context.get("attempt_index", ""),
        "expression": expression,
        "ordinary_interval_remainder_width": "",
        "symbolic_remainder_width": "",
        "materialized_remainder_width": context.get("materialized_remainder_width_sum", ""),
        "symbolic_poly_times_noise_width": "",
        "noise_times_noise_width": "",
        "truncation_width": "",
        "total_width": "",
        "active_noise_symbols": context.get("active_noise_symbols", 0),
        "finite": "",
        "notes": "",
    }


def _symbolic_mul_breakdown_row(
    context: Mapping[str, Any],
    expression: str,
    a: TaylorModel,
    b: TaylorModel,
    order: int,
) -> dict[str, Any]:
    row = _base_breakdown_row(context, expression)
    try:
        noise_indices = tuple(int(i) for i in context.get("noise_indices", ()))
        _kept, dropped = a.polynomial.mul_truncate(b.polynomial, int(order))
        ordinary_width = _safe_width_value(_ordinary_remainder_interaction(a, b))
        trunc_width = _poly_interval_width(dropped, a.domain)
        total_width = _safe_width_value((a * b).range_box())

        if noise_indices:
            a_plain, a_noise = split_polynomial_by_variables(a.polynomial, noise_indices)
            b_plain, b_noise = split_polynomial_by_variables(b.polynomial, noise_indices)
            _plain_product, symbolic_product = split_polynomial_by_variables(a.polynomial * b.polynomial, noise_indices)
            symbolic_width = _poly_interval_width(symbolic_product, a.domain)
            poly_times_noise_width = _mul_polys_width(a_plain, b_noise, a.domain)
            other_poly_times_noise_width = _mul_polys_width(b_plain, a_noise, a.domain)
            if _finite_float(poly_times_noise_width) is not None and _finite_float(other_poly_times_noise_width) is not None:
                symbolic_poly_times_noise_width: float | str = float(poly_times_noise_width) + float(other_poly_times_noise_width)
            else:
                symbolic_poly_times_noise_width = ""
            noise_times_noise_width = _mul_polys_width(a_noise, b_noise, a.domain)
        else:
            symbolic_width = 0.0
            symbolic_poly_times_noise_width = 0.0
            noise_times_noise_width = 0.0

        row.update(
            {
                "ordinary_interval_remainder_width": ordinary_width,
                "symbolic_remainder_width": symbolic_width,
                "symbolic_poly_times_noise_width": symbolic_poly_times_noise_width,
                "noise_times_noise_width": noise_times_noise_width,
                "truncation_width": trunc_width,
                "total_width": total_width,
                "finite": _finite_values(
                    ordinary_width,
                    symbolic_width,
                    symbolic_poly_times_noise_width,
                    noise_times_noise_width,
                    trunc_width,
                    total_width,
                ),
            }
        )
    except Exception as exc:
        row["finite"] = False
        row["notes"] = f"breakdown exception: {exc}"
    return row


def vdp_symbolic_breakdown_rows(candidate: TMVector, order: int, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        x = candidate[0]
        y = candidate[1]
        rows.append(_symbolic_mul_breakdown_row(context, "x_sq", x, x, order))
        x_sq = x * x
        rows.append(_symbolic_mul_breakdown_row(context, "x_sq_y", x_sq, y, order))
    except Exception as exc:
        row = _base_breakdown_row(context, "vdp_rhs")
        row["finite"] = False
        row["notes"] = f"breakdown exception: {exc}"
        rows.append(row)
    return rows


def _make_breakdown_callback(rows: list[dict[str, Any]]):
    def _callback(candidate: TMVector, order: int, _attempt: int, context: Mapping[str, Any]) -> None:
        rows.extend(vdp_symbolic_breakdown_rows(candidate, order, context))

    return _callback


def _segment_row_from_step(
    *,
    run_id: str,
    mode: str,
    order: int,
    substep_factor: int,
    dependency_window: str,
    symbolic_remainder: bool,
    queue_size: int | str,
    segment_index: int,
    reference_segment_index: int,
    substep_index: int,
    t_lo: float,
    t_hi: float,
    seg: Any,
    state: SymbolicRemainderState | None,
    materialized_width_sum: float | str = "",
    reset_after_segment: bool = False,
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
    x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = _safe_segment_bounds(box)
    _fx_lo, _fx_hi, _fy_lo, _fy_hi, final_width_x, final_width_y, final_width_sum = _safe_segment_bounds(final_box)
    active_symbols = len(state.symbols) if state is not None else 0
    if state is not None:
        symbolic_width_sum = _sum_widths(symbolic_remainder_widths(seg.final_tm, state))
    else:
        symbolic_width_sum = 0.0
    ordinary_width_sum = _sum_widths(ordinary_remainder_widths(seg.final_tm))
    row = {
        "run_id": run_id,
        "mode": mode,
        "order": int(order),
        "substep_factor": int(substep_factor),
        "dependency_window": dependency_window,
        "symbolic_remainder": bool(symbolic_remainder),
        "queue_size": queue_size,
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
        "ordinary_remainder_width_sum": ordinary_width_sum,
        "symbolic_remainder_width_sum": symbolic_width_sum,
        "materialized_remainder_width_sum": materialized_width_sum,
        "active_noise_symbols": active_symbols,
        "reset_after_segment": bool(reset_after_segment),
        "message": message,
    }
    return row, final_box, row_status


def _summary_from_segments(
    *,
    run_id: str,
    mode: str,
    order: int,
    substep_factor: int,
    dependency_window: str,
    symbolic_remainder: bool,
    queue_size: int | str,
    status: str,
    runtime_s: float,
    segment_rows: Sequence[Mapping[str, Any]],
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
        "symbolic_remainder": bool(symbolic_remainder),
        "queue_size": queue_size,
        "status": status,
        "runtime_s": runtime_s,
        "validated_segments": len(validated),
        "last_validated_t": last_validated_t,
        "last_attempted_t": last_attempted_t,
        "failed_segment_index": failed_segment_index,
        "failure_reason": failure_reason,
        "final_validated_width_sum": final_width,
        "max_width_sum": _max_field(validated, "final_width_sum"),
        "max_ordinary_remainder_width_sum": _max_field(segment_rows, "ordinary_remainder_width_sum"),
        "max_symbolic_remainder_width_sum": _max_field(segment_rows, "symbolic_remainder_width_sum"),
        "max_materialized_remainder_width_sum": _max_field(segment_rows, "materialized_remainder_width_sum"),
        "max_active_noise_symbols": _max_field(segment_rows, "active_noise_symbols"),
        "notes": notes,
    }


def _window_for_mode(mode: str) -> tuple[str, int]:
    if mode == "range_only":
        return "1", 1
    if mode == "dependency_window_2":
        return "2", 2
    raise ValueError(f"unknown mode {mode}")


def _run_id(mode: str, order: int, substep_factor: int, symbolic_remainder: bool, queue_size: int | str) -> str:
    if symbolic_remainder:
        return f"{mode}_symbolic_o{order}_s{substep_factor}_q{queue_size}"
    return f"{mode}_o{order}_s{substep_factor}_baseline"


def run_symbolic_diagnostic(
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    order: int,
    substep_factor: int,
    symbolic_remainder: bool,
    queue_size: int | str,
    max_wall_s_per_run: float,
    max_horizon: float,
    collect_breakdowns: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    dependency_window, window_count = _window_for_mode(mode)
    run_id = _run_id(mode, order, substep_factor, symbolic_remainder, queue_size)
    steps = _reference_substeps(reference_segments, substep_factor, max_horizon)
    current_box = _initial_box(params)
    current_tm = TMVector.identity(current_box, order=order)
    state = SymbolicRemainderState.empty(int(queue_size) if symbolic_remainder else 0)
    accepted_since_reset = 0
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    callback = _make_breakdown_callback(breakdown_rows) if collect_breakdowns else None
    status = "completed"
    failure_reason = ""
    failed_segment_index: int | str = ""
    last_attempted_t: float | str = ""
    notes = "symbolic remainder diagnostic on the original Flow* segment grid"
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
        materialized_width_sum: float | str = 0.0
        context = {
            "run_id": run_id,
            "mode": mode,
            "order": order,
            "substep_factor": substep_factor,
            "dependency_window": dependency_window,
            "symbolic_remainder": bool(symbolic_remainder),
            "queue_size": queue_size,
            "segment_index": segment_index,
            "reference_segment_index": ref_index,
            "substep_index": substep_index,
            "t_lo": t_lo,
            "t_hi": t_hi,
            "noise_indices": state.active_var_indices() if symbolic_remainder else (),
            "active_noise_symbols": len(state.symbols) if symbolic_remainder else 0,
            "materialized_remainder_width_sum": 0.0,
        }
        remaining_s = max_wall_s_per_run - (time.perf_counter() - start)
        try:
            if symbolic_remainder:
                seg = _call_with_timeout(
                    lambda: flowpipe_step_from_tm(
                        van_der_pol_ode,
                        current_tm,
                        h,
                        order,
                        diagnostics=attempt_rows,
                        diagnostics_context=context,
                        rhs_breakdown_callback=callback,
                        symbolic_remainder=True,
                        max_symbolic_remainders=int(queue_size),
                        symbolic_remainder_state=state,
                    ),
                    remaining_s,
                )
                state = seg.symbolic_remainder_state or state
                stats = dict(seg.symbolic_remainder_stats or {})
                materialized_width_sum = stats.get("materialized_remainder_width_sum", 0.0)
            else:
                seg = _call_with_timeout(
                    lambda: flowpipe_step(
                        van_der_pol_ode,
                        current_box,
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
            failed_segment_index = segment_index
            failure_reason = str(exc)
            notes = f"wall-time cap reached while attempting segment {segment_index}"
            break

        accepted_since_reset_next = accepted_since_reset + 1
        reset_due = symbolic_remainder and accepted_since_reset_next >= window_count
        row, final_box, row_status = _segment_row_from_step(
            run_id=run_id,
            mode=mode,
            order=order,
            substep_factor=substep_factor,
            dependency_window=dependency_window,
            symbolic_remainder=symbolic_remainder,
            queue_size=queue_size if symbolic_remainder else "",
            segment_index=segment_index,
            reference_segment_index=ref_index,
            substep_index=substep_index,
            t_lo=t_lo,
            t_hi=t_hi,
            seg=seg,
            state=state if symbolic_remainder else None,
            materialized_width_sum=materialized_width_sum,
            reset_after_segment=reset_due and getattr(seg, "status", "") == "validated",
        )
        segment_rows.append(row)
        if row_status != "validated":
            failed_segment_index = segment_index
            failure_reason = row.get("message") or "validation failed"
            status = "timeout" if "wall-time cap" in str(failure_reason) else "failed"
            notes = f"stopped on first failed symbolic diagnostic substep {segment_index}"
            break

        if symbolic_remainder:
            if reset_due:
                current_tm, state, reset_stats = materialize_non_symbolic_variables(seg.final_tm, state)
                reset_width = reset_stats.get("materialized_remainder_width_sum", 0.0)
                existing = _finite_float(row.get("materialized_remainder_width_sum")) or 0.0
                row["materialized_remainder_width_sum"] = existing + (reset_width if isinstance(reset_width, float) else 0.0)
                row["ordinary_remainder_width_sum"] = _sum_widths(ordinary_remainder_widths(current_tm))
                row["symbolic_remainder_width_sum"] = _sum_widths(symbolic_remainder_widths(current_tm, state))
                row["active_noise_symbols"] = len(state.symbols)
                accepted_since_reset = 0
            else:
                current_tm = seg.final_tm
                accepted_since_reset = accepted_since_reset_next
        else:
            current_box = [iv.inflate(1e-9) for iv in (final_box or seg.final_tm.range_box())]

        if time.perf_counter() - start >= max_wall_s_per_run and t_hi < max_horizon - 1e-15:
            status = "timeout"
            failure_reason = f"wall-time cap reached after validating segment {segment_index}"
            notes = failure_reason
            break

    runtime_s = time.perf_counter() - start
    if status == "completed":
        if segment_rows and float(segment_rows[-1]["t_hi"]) >= max_horizon - 1e-12:
            status = "max_horizon_reached"
            notes = "validated to the requested stage-3 horizon"
        elif not steps:
            status = "max_horizon_reached"
            notes = "no positive diagnostic substeps were available"
        else:
            status = "max_horizon_reached"
            notes = "reference grid ended before requested diagnostic horizon"
    if last_attempted_t == "":
        last_attempted_t = float(segment_rows[-1]["t_hi"]) if segment_rows else 0.0

    summary = _summary_from_segments(
        run_id=run_id,
        mode=mode,
        order=order,
        substep_factor=substep_factor,
        dependency_window=dependency_window,
        symbolic_remainder=symbolic_remainder,
        queue_size=queue_size if symbolic_remainder else "",
        status=status,
        runtime_s=runtime_s,
        segment_rows=segment_rows,
        last_attempted_t=last_attempted_t,
        failed_segment_index=failed_segment_index,
        failure_reason=failure_reason,
        notes=notes,
    )
    return summary, segment_rows, breakdown_rows


def _run_spec(spec: Mapping[str, Any]):
    return run_symbolic_diagnostic(**spec)


def _sort_outputs(
    summary_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    breakdown_rows: list[dict[str, Any]],
) -> None:
    summary_rows.sort(
        key=lambda r: (
            str(r.get("mode", "")),
            int(r.get("order", 0)),
            int(r.get("substep_factor", 0)),
            int(r.get("queue_size") or 0),
            str(r.get("run_id", "")),
        )
    )
    segment_rows.sort(key=lambda r: (str(r.get("run_id", "")), int(r.get("segment_index", 0))))
    breakdown_rows.sort(
        key=lambda r: (
            str(r.get("run_id", "")),
            int(r.get("segment_index", 0)) if str(r.get("segment_index", "")).strip() else -1,
            int(r.get("attempt_index", 0)) if str(r.get("attempt_index", "")).strip() else -1,
            str(r.get("expression", "")),
        )
    )


def write_outputs(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    breakdown_rows: list[dict[str, Any]],
) -> None:
    _sort_outputs(summary_rows, segment_rows, breakdown_rows)
    _write_csv(out_dir / "symbolic_remainder_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "symbolic_remainder_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "symbolic_remainder_breakdown.csv", BREAKDOWN_FIELDS, breakdown_rows)
    write_symbolic_report(out_dir, summary_rows, segment_rows, breakdown_rows)


def run_stage3_diagnostics(
    out_dir: Path,
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
    *,
    max_wall_s_per_run: float,
    max_horizon: float,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = [
        {
            "params": dict(params),
            "reference_segments": list(reference_segments),
            "mode": "range_only",
            "order": 6,
            "substep_factor": 4,
            "symbolic_remainder": False,
            "queue_size": "",
            "max_wall_s_per_run": max_wall_s_per_run,
            "max_horizon": max_horizon,
        }
    ]
    for mode in ("range_only", "dependency_window_2"):
        for order in (4, 6):
            for queue_size in (4, 8, 16):
                specs.append(
                    {
                        "params": dict(params),
                        "reference_segments": list(reference_segments),
                        "mode": mode,
                        "order": order,
                        "substep_factor": 4,
                        "symbolic_remainder": True,
                        "queue_size": queue_size,
                        "max_wall_s_per_run": max_wall_s_per_run,
                        "max_horizon": max_horizon,
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    requested_workers = max(1, min(int(workers), len(specs) or 1))
    if requested_workers == 1:
        iterator = (_run_spec(spec) for spec in specs)
    else:
        pool = ProcessPoolExecutor(max_workers=requested_workers)
        futures = [pool.submit(_run_spec, spec) for spec in specs]
        iterator = (future.result() for future in as_completed(futures))
    try:
        for summary, segments, breakdowns in iterator:
            summary_rows.append(summary)
            segment_rows.extend(segments)
            breakdown_rows.extend(breakdowns)
            write_outputs(out_dir, summary_rows, segment_rows, breakdown_rows)
    finally:
        if requested_workers != 1:
            pool.shutdown(wait=True)
    write_outputs(out_dir, summary_rows, segment_rows, breakdown_rows)
    make_plots(out_dir, summary_rows, segment_rows, breakdown_rows)
    write_symbolic_report(out_dir, summary_rows, segment_rows, breakdown_rows)
    return summary_rows, segment_rows, breakdown_rows


def _best_row(rows: Sequence[Mapping[str, Any]], *, symbolic_only: bool = False, mode: str | None = None) -> Mapping[str, Any] | None:
    best = None
    best_t = -math.inf
    for row in rows:
        if symbolic_only and str(row.get("symbolic_remainder", "")).lower() not in {"true", "1"} and row.get("symbolic_remainder") is not True:
            continue
        if mode is not None and row.get("mode") != mode:
            continue
        value = _finite_float(row.get("last_validated_t"))
        if value is not None and value > best_t:
            best_t = value
            best = row
    return best


def _max_breakdown(rows: Sequence[Mapping[str, Any]], run_id: str, field: str, *, expression: str = "x_sq_y") -> float:
    vals = []
    for row in rows:
        if row.get("run_id") != run_id or row.get("expression") != expression:
            continue
        value = _finite_float(row.get(field))
        if value is not None:
            vals.append(value)
    return max(vals) if vals else 0.0


def _best_queue_text(rows: Sequence[Mapping[str, Any]]) -> tuple[str, float]:
    by_queue: dict[str, float] = {}
    for row in rows:
        if row.get("symbolic_remainder") is not True and str(row.get("symbolic_remainder", "")).lower() != "true":
            continue
        queue = str(row.get("queue_size", ""))
        value = _finite_float(row.get("last_validated_t"))
        if value is not None:
            by_queue[queue] = max(by_queue.get(queue, 0.0), value)
    if not by_queue:
        return "no symbolic queue rows", 0.0
    best_queue = max(by_queue, key=by_queue.get)
    return best_queue, by_queue[best_queue]


def _recommendation(summary_rows: Sequence[Mapping[str, Any]], breakdown_rows: Sequence[Mapping[str, Any]], best_symbolic: Mapping[str, Any] | None) -> str:
    if best_symbolic is None:
        return "d) stop because symbolic remainder did not help"
    best_t = _finite_float(best_symbolic.get("last_validated_t")) or 0.0
    best_run = str(best_symbolic.get("run_id", ""))
    noise_noise = _max_breakdown(breakdown_rows, best_run, "noise_times_noise_width")
    materialized = _finite_float(best_symbolic.get("max_materialized_remainder_width_sum")) or 0.0
    symbolic_width = _finite_float(best_symbolic.get("max_symbolic_remainder_width_sum")) or 0.0
    ordinary_width = _finite_float(best_symbolic.get("max_ordinary_remainder_width_sum")) or 0.0
    if best_t <= BASELINE_BEST_FIXED_T + 1e-9:
        return "d) stop because symbolic remainder did not help"
    if noise_noise > max(symbolic_width, ordinary_width, 1.0) * 0.25:
        return "a) improve symbolic remainder multiplication"
    if materialized > max(ordinary_width, 1e-300):
        return "b) queue/materialization policy"
    return "c) polynomial range bounding"


def write_symbolic_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    breakdown_rows: Sequence[Mapping[str, Any]],
) -> None:
    baseline = next((row for row in summary_rows if row.get("symbolic_remainder") is False or str(row.get("symbolic_remainder", "")).lower() == "false"), None)
    best_symbolic = _best_row(summary_rows, symbolic_only=True)
    best_range = _best_row(summary_rows, symbolic_only=True, mode="range_only")
    best_window2 = _best_row(summary_rows, symbolic_only=True, mode="dependency_window_2")
    best_queue, best_queue_t = _best_queue_text(summary_rows)

    best_t = _finite_float(best_symbolic.get("last_validated_t")) if best_symbolic else None
    beat_fixed = best_t is not None and best_t > BASELINE_BEST_FIXED_T + 1e-9
    baseline_t = _finite_float(baseline.get("last_validated_t")) if baseline else None
    baseline_run = str(baseline.get("run_id", "")) if baseline else ""
    best_run = str(best_symbolic.get("run_id", "")) if best_symbolic else ""

    baseline_ordinary = _max_breakdown(breakdown_rows, baseline_run, "ordinary_interval_remainder_width") if baseline else 0.0
    best_ordinary = _max_breakdown(breakdown_rows, best_run, "ordinary_interval_remainder_width") if best_symbolic else 0.0
    reduced_blowup = best_symbolic is not None and best_ordinary < baseline_ordinary
    materialized = _finite_float(best_symbolic.get("max_materialized_remainder_width_sum")) if best_symbolic else None
    materialization_text = "No materialization signal was recorded."
    if materialized is not None and materialized > 0:
        if best_symbolic and str(best_symbolic.get("status", "")) in {"failed", "timeout"}:
            materialization_text = f"Possibly. The best run materialized width up to {_fmt(materialized)} before stopping at t={_fmt(best_t)}."
        else:
            materialization_text = f"Materialization occurred (max width {_fmt(materialized)}), but this run did not fail before the requested horizon."

    range_t = _finite_float(best_range.get("last_validated_t")) if best_range else None
    window2_t = _finite_float(best_window2.get("last_validated_t")) if best_window2 else None
    if range_t is None or window2_t is None:
        dependency_text = "Insufficient rows to compare range_only and dependency_window_2."
    elif window2_t > range_t + 1e-9:
        dependency_text = f"Yes. Best dependency_window_2 symbolic reached t={_fmt(window2_t)} vs range_only at t={_fmt(range_t)}."
    elif range_t > window2_t + 1e-9:
        dependency_text = f"No. Best range_only symbolic reached t={_fmt(range_t)} vs dependency_window_2 at t={_fmt(window2_t)}."
    else:
        dependency_text = f"They tied at t={_fmt(range_t)}."

    total_runtime = sum((_finite_float(row.get("runtime_s")) or 0.0) for row in summary_rows)
    timeout_count = sum(1 for row in summary_rows if row.get("status") == "timeout")
    runtime_text = f"Total recorded runtime was {_fmt(total_runtime)} s across {len(summary_rows)} runs; {timeout_count} runs hit the per-run wall cap."
    recommendation = _recommendation(summary_rows, breakdown_rows, best_symbolic)

    table = [
        "| run | status | last_validated_t | queue | max active noise | max ordinary rem | max symbolic rem | max materialized rem |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        table.append(
            f"| `{row.get('run_id', '')}` | `{row.get('status', '')}` | {_fmt(row.get('last_validated_t', ''))} | "
            f"{_fmt(row.get('queue_size', ''))} | {_fmt(row.get('max_active_noise_symbols', ''))} | "
            f"{_fmt(row.get('max_ordinary_remainder_width_sum', ''))} | {_fmt(row.get('max_symbolic_remainder_width_sum', ''))} | "
            f"{_fmt(row.get('max_materialized_remainder_width_sum', ''))} |"
        )

    text = f"""# Stage-3 Symbolic Remainder Diagnostics

This report is diagnostic-only. It does not claim Flow* parity or a new full reachability algorithm.

## Run Summary

{chr(10).join(table)}

## Questions

- Did symbolic remainder handling beat the current best fixed diagnostic run, `range_only_o6_s4` at t ~= {BASELINE_BEST_FIXED_T}? {'Yes' if beat_fixed else 'No'}. Best symbolic run `{best_run}` reached t={_fmt(best_t) if best_t is not None else ''}. The local baseline row `{baseline_run}` reached t={_fmt(baseline_t) if baseline_t is not None else ''}.
- Did it reduce polynomial_range * remainder blowup? {'Yes' if reduced_blowup else 'No'}. Max `x_sq_y` ordinary interval-remainder interaction was {_fmt(best_ordinary)} for the best symbolic run vs {_fmt(baseline_ordinary)} for baseline.
- Which queue size was best? Queue {best_queue} with best last_validated_t={_fmt(best_queue_t)}.
- Did materialization of old symbols cause a later blowup? {materialization_text}
- Did dependency_window_2 + symbolic remainder help more than pure range_only? {dependency_text}
- Was runtime acceptable? {runtime_text}
- Next target: **{recommendation}**.
"""
    (out_dir / "symbolic_remainder_report.md").write_text(text, encoding="utf-8", newline="\n")


def make_plots(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    breakdown_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths: list[Path] = []
    labels = [str(row.get("run_id", "")) for row in summary_rows]
    values = [_finite_float(row.get("last_validated_t")) or 0.0 for row in summary_rows]
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.bar(range(len(labels)), values)
    ax.axhline(BASELINE_BEST_FIXED_T, color="black", linewidth=1.0, linestyle="--", label="range_only_o6_s4 t ~= 0.7661635")
    ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("last validated t")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / PLOT_NAMES[0]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    by_run_breakdown: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in breakdown_rows:
        by_run_breakdown[str(row.get("run_id", ""))].append(row)
    metric_fields = [
        "ordinary_interval_remainder_width",
        "symbolic_remainder_width",
        "materialized_remainder_width",
        "symbolic_poly_times_noise_width",
        "noise_times_noise_width",
        "truncation_width",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    x = range(len(by_run_breakdown))
    bottoms = [0.0 for _ in by_run_breakdown]
    run_ids = list(by_run_breakdown)
    for field in metric_fields:
        vals = []
        for run_id in run_ids:
            vals.append(max((_finite_float(row.get(field)) or 0.0) for row in by_run_breakdown[run_id]))
        ax.bar(x, vals, bottom=bottoms, label=field)
        bottoms = [a + b for a, b in zip(bottoms, vals)]
    ax.set_xticks(list(x), run_ids, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("max width contribution")
    if any(value > 0.0 for value in bottoms):
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=6, ncols=2)
    fig.tight_layout()
    path = out_dir / PLOT_NAMES[1]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    by_run_segments: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        by_run_segments[str(row.get("run_id", ""))].append(row)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for run_id, rows in by_run_segments.items():
        pts = [
            (_finite_float(row.get("t_hi")), _finite_float(row.get("active_noise_symbols")))
            for row in rows
            if row.get("symbolic_remainder") is True or str(row.get("symbolic_remainder", "")).lower() == "true"
        ]
        pts = [(xv, yv) for xv, yv in pts if xv is not None and yv is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=1.0, label=run_id)
    ax.set_xlabel("t")
    ax.set_ylabel("active noise symbols")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5, ncols=2)
    fig.tight_layout()
    path = out_dir / PLOT_NAMES[2]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for run_id, rows in by_run_segments.items():
        pts = [
            (
                _finite_float(row.get("t_hi")),
                _finite_float(row.get("ordinary_remainder_width_sum")),
                _finite_float(row.get("symbolic_remainder_width_sum")),
            )
            for row in rows
            if row.get("symbolic_remainder") is True or str(row.get("symbolic_remainder", "")).lower() == "true"
        ]
        pts = [(xv, ov, sv) for xv, ov, sv in pts if xv is not None and ov is not None and sv is not None]
        if pts:
            ax.plot([p[0] for p in pts], [max(p[1], 1e-320) for p in pts], linewidth=0.9, linestyle="--", label=f"{run_id} ordinary")
            ax.plot([p[0] for p in pts], [max(p[2], 1e-320) for p in pts], linewidth=1.0, label=f"{run_id} symbolic")
    ax.set_xlabel("t")
    ax.set_ylabel("remainder width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=4, ncols=2)
    fig.tight_layout()
    path = out_dir / PLOT_NAMES[3]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-3 symbolic remainder diagnostics on the Flow* Van der Pol grid.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "flowstar_benchmark_diagnostics_stage3"))
    parser.add_argument("--parity-dir", default=str(REPO_ROOT / "outputs" / "flowstar_benchmark_parity"))
    parser.add_argument("--max-horizon", type=float, default=1.0)
    parser.add_argument("--wall-cap-s", type=float, default=240.0)
    parser.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    args = parser.parse_args()

    params, reference_segments = load_reference_inputs(Path(args.parity_dir))
    summary_rows, segment_rows, breakdown_rows = run_stage3_diagnostics(
        Path(args.out_dir),
        params,
        reference_segments,
        max_wall_s_per_run=float(args.wall_cap_s),
        max_horizon=float(args.max_horizon),
        workers=int(args.workers),
    )
    print(f"wrote {args.out_dir}")
    print(f"runs={len(summary_rows)} segments={len(segment_rows)} breakdown_rows={len(breakdown_rows)} workers={args.workers}")


if __name__ == "__main__":
    main()
