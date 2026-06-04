#!/usr/bin/env python3
"""Diagnose PyTorch Taylor-model failures on the original Flow* Van der Pol grid."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
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

from torch_tm_flowpipe import Interval, TMVector, flowpipe_step, flowpipe_step_from_tm
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

from flowstar_benchmark_parity import _fmt, _initial_box, _interval_tuple, _width, _write_csv

RUN_SUMMARY_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
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
SEGMENT_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
    "segment_index",
    "reference_segment_index",
    "substep_index",
    "status",
    "validation_attempts",
    "t_lo",
    "t_hi",
    "x_lo",
    "x_hi",
    "y_lo",
    "y_hi",
    "width_x",
    "width_y",
    "width_sum",
    "box_source",
    "message",
]
VALIDATION_ATTEMPT_FIELDS = [
    "run_id",
    "mode",
    "order",
    "substep_factor",
    "segment_index",
    "reference_segment_index",
    "substep_index",
    "t_lo",
    "t_hi",
    "attempt_index",
    "h",
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
PLOT_NAMES = [
    "width_growth_by_mode_order_substep.png",
    "residual_growth_by_mode_order_substep.png",
    "remainder_vs_polynomial_width.png",
    "last_validated_t_by_run.png",
]
MODE_ORDER = {"range_only": 0, "dependency_preserving": 1}


class StepTimeout(RuntimeError):
    pass


def _call_with_timeout(fn: Any, timeout_s: float) -> Any:
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_reference_inputs(parity_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    params_path = parity_dir / "original_flowstar_params.json"
    segments_path = parity_dir / "original_flowstar" / "original_flowstar_segments.csv"
    if not params_path.exists():
        raise FileNotFoundError(f"missing parity params: {params_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"missing original Flow* segment grid: {segments_path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    segments = _read_csv(segments_path)
    if not segments:
        raise ValueError(f"empty original Flow* segment grid: {segments_path}")
    return params, segments


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _max_field(rows: Iterable[Mapping[str, Any]], field: str) -> float | str:
    vals = [_finite_float(row.get(field)) for row in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else ""


def _reference_substeps(
    reference_segments: Sequence[Mapping[str, Any]],
    substep_factor: int,
    max_horizon: float,
) -> list[tuple[int, int, float, float]]:
    out: list[tuple[int, int, float, float]] = []
    if substep_factor <= 0:
        raise ValueError("substep_factor must be positive")
    for ref_index, ref in enumerate(reference_segments):
        t_lo = float(ref["t_lo"])
        t_hi = float(ref["t_hi"])
        if t_lo >= max_horizon - 1e-15:
            break
        t_hi = min(t_hi, max_horizon)
        if t_hi <= t_lo:
            continue
        h = (t_hi - t_lo) / substep_factor
        for substep_index in range(substep_factor):
            lo = t_lo + substep_index * h
            hi = t_lo + (substep_index + 1) * h
            if hi > max_horizon:
                hi = max_horizon
            if hi > lo:
                out.append((ref_index, substep_index, lo, hi))
    return out


def _box_is_finite(box: Sequence[Interval]) -> bool:
    return all(iv.is_finite() for iv in box)


def _segment_bounds(box: Sequence[Interval]) -> tuple[float, float, float, float, float, float, float]:
    x_lo, x_hi = _interval_tuple(box[0])
    y_lo, y_hi = _interval_tuple(box[1])
    width_x = _width(x_lo, x_hi)
    width_y = _width(y_lo, y_hi)
    return x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_x + width_y


def _summarize_run(
    run_id: str,
    mode: str,
    order: int,
    substep_factor: int,
    status: str,
    runtime_s: float,
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    last_attempted_t: float | str,
    failed_segment_index: int | str,
    failure_reason: str,
    notes: str,
) -> dict[str, Any]:
    validated_rows = [row for row in segment_rows if row.get("status") == "validated"]
    last_validated_t = float(validated_rows[-1]["t_hi"]) if validated_rows else 0.0
    final_width = validated_rows[-1]["width_sum"] if validated_rows else ""
    return {
        "run_id": run_id,
        "mode": mode,
        "order": order,
        "substep_factor": substep_factor,
        "status": status,
        "runtime_s": runtime_s,
        "validated_segments": len(validated_rows),
        "last_validated_t": last_validated_t,
        "last_attempted_t": last_attempted_t,
        "failed_segment_index": failed_segment_index,
        "failure_reason": failure_reason,
        "final_validated_width_sum": final_width,
        "max_validated_width_sum": _max_field(validated_rows, "width_sum"),
        "max_residual_width_sum": _max_field(attempt_rows, "residual_width_sum"),
        "max_remainder_width_sum": _max_field(attempt_rows, "remainder_width_sum"),
        "max_polynomial_range_width_sum": _max_field(attempt_rows, "polynomial_range_width_sum"),
        "notes": notes,
    }


def _run_diagnostic_spec(spec: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    return run_torch_diagnostic(
        spec["params"],
        spec["reference_segments"],
        mode=spec["mode"],
        order=spec["order"],
        substep_factor=spec["substep_factor"],
        max_wall_s_per_run=spec["max_wall_s_per_run"],
        max_horizon=spec["max_horizon"],
    )


def _sort_outputs(
    run_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
) -> None:
    run_rows.sort(key=lambda r: (MODE_ORDER.get(str(r.get("mode")), 99), int(r.get("order", 0)), int(r.get("substep_factor", 0))))
    segment_rows.sort(key=lambda r: (str(r.get("run_id", "")), int(r.get("segment_index", 0))))
    attempt_rows.sort(
        key=lambda r: (
            str(r.get("run_id", "")),
            int(r.get("segment_index", 0)) if str(r.get("segment_index", "")).strip() else -1,
            int(r.get("attempt_index", 0)) if str(r.get("attempt_index", "")).strip() else -1,
        )
    )


def _write_required_outputs(
    out_dir: Path,
    run_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
) -> None:
    _sort_outputs(run_rows, segment_rows, attempt_rows)
    _write_csv(out_dir / "diagnostic_runs_summary.csv", RUN_SUMMARY_FIELDS, run_rows)
    _write_csv(out_dir / "diagnostic_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "diagnostic_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    write_diagnostic_report(out_dir, run_rows, attempt_rows)


def run_torch_diagnostic(
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    order: int,
    substep_factor: int,
    max_wall_s_per_run: float,
    max_horizon: float,
    progress_queue: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if mode not in {"range_only", "dependency_preserving"}:
        raise ValueError("mode must be range_only or dependency_preserving")
    run_id = f"{mode}_o{order}_s{substep_factor}"
    steps = _reference_substeps(reference_segments, substep_factor, max_horizon)
    current_box = _initial_box(params)
    current_tm = TMVector.identity(current_box, order=order)
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    status = "completed"
    notes = "diagnostic run on the original Flow* segment grid"
    failure_reason = ""
    failed_segment_index: int | str = ""
    last_attempted_t: float | str = ""
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
            "segment_index": segment_index,
            "reference_segment_index": ref_index,
            "substep_index": substep_index,
            "substep_factor": substep_factor,
            "t_lo": t_lo,
            "t_hi": t_hi,
        }
        if h <= 0:
            status = "failed"
            failed_segment_index = segment_index
            failure_reason = f"non-positive diagnostic substep at segment {segment_index}"
            notes = failure_reason
            break
        attempt_start = len(attempt_rows)
        remaining_s = max_wall_s_per_run - (time.perf_counter() - start)
        try:
            if mode == "range_only":
                seg = _call_with_timeout(
                    lambda: flowpipe_step(
                        van_der_pol_ode,
                        current_box,
                        h,
                        order,
                        diagnostics=attempt_rows,
                        diagnostics_context=context,
                    ),
                    remaining_s,
                )
            else:
                seg = _call_with_timeout(
                    lambda: flowpipe_step_from_tm(
                        van_der_pol_ode,
                        current_tm,
                        h,
                        order,
                        diagnostics=attempt_rows,
                        diagnostics_context=context,
                    ),
                    remaining_s,
                )
        except StepTimeout as exc:
            status = "timeout"
            failed_segment_index = segment_index
            failure_reason = str(exc)
            notes = f"wall-time cap reached while attempting segment {segment_index}"
            break

        box = seg.tm.range_box()
        final_box = seg.final_tm.range_box()
        x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = _segment_bounds(box)
        finite_box = _box_is_finite(box) and _box_is_finite(final_box)
        row_status = "validated" if seg.status == "validated" and finite_box else "failed"
        message = seg.message
        if seg.status == "validated" and not finite_box:
            message = "non-finite interval in segment or final Taylor model"
        segment_rows.append(
            {
                "run_id": run_id,
                "mode": mode,
                "order": order,
                "substep_factor": substep_factor,
                "segment_index": segment_index,
                "reference_segment_index": ref_index,
                "substep_index": substep_index,
                "status": row_status,
                "validation_attempts": seg.validation_attempts,
                "t_lo": t_lo,
                "t_hi": t_hi,
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "width_x": width_x,
                "width_y": width_y,
                "width_sum": width_sum,
                "box_source": f"torch_tm_{mode}_diagnostic_substep_on_flowstar_grid",
                "message": message,
            }
        )
        if progress_queue is not None:
            progress_queue.put({"type": "segment", "segment": segment_rows[-1], "attempts": attempt_rows[attempt_start:]})
        if row_status != "validated":
            failed_segment_index = segment_index
            failure_reason = message or "validation failed"
            if "wall-time cap" in failure_reason:
                status = "timeout"
                notes = f"wall-time cap reached while attempting segment {segment_index}"
            else:
                status = "failed"
                notes = f"stopped on first failed diagnostic substep {segment_index}"
            break
        if mode == "range_only":
            current_box = [iv.inflate(1e-9) for iv in final_box]
        else:
            current_tm = seg.final_tm
        if time.perf_counter() - start >= max_wall_s_per_run and t_hi < max_horizon - 1e-15:
            status = "timeout"
            failure_reason = f"wall-time cap reached after validating segment {segment_index}"
            notes = failure_reason
            break

    runtime_s = time.perf_counter() - start
    if status == "completed":
        if not steps:
            status = "max_horizon_reached"
            notes = "no positive diagnostic substeps were available"
        elif segment_rows and float(segment_rows[-1]["t_hi"]) + 1e-12 < max_horizon:
            status = "max_horizon_reached"
            notes = "reference grid ended before requested diagnostic horizon"
        else:
            notes = "validated to the requested diagnostic horizon"
    if last_attempted_t == "" and segment_rows:
        last_attempted_t = float(segment_rows[-1]["t_hi"])
    if last_attempted_t == "":
        last_attempted_t = 0.0

    summary = _summarize_run(
        run_id,
        mode,
        order,
        substep_factor,
        status,
        runtime_s,
        segment_rows,
        attempt_rows,
        last_attempted_t,
        failed_segment_index,
        failure_reason,
        notes,
    )
    return summary, segment_rows, attempt_rows


def run_diagnostics(
    out_dir: Path,
    params: Mapping[str, Any],
    reference_segments: Sequence[Mapping[str, Any]],
    *,
    orders: Sequence[int],
    substep_factors: Sequence[int],
    max_wall_s_per_run: float,
    max_horizon: float,
    modes: Sequence[str] = ("range_only", "dependency_preserving"),
    workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    requested_horizon = min(float(params.get("time_horizon", max_horizon)), float(max_horizon))
    specs = [
        {
            "params": dict(params),
            "reference_segments": list(reference_segments),
            "mode": mode,
            "order": int(order),
            "substep_factor": int(substep_factor),
            "max_wall_s_per_run": float(max_wall_s_per_run),
            "max_horizon": requested_horizon,
        }
        for mode in modes
        for order in orders
        for substep_factor in substep_factors
    ]
    workers = max(1, min(int(workers), len(specs) or 1))
    if workers == 1:
        for spec in specs:
            summary, segments, attempts = _run_diagnostic_spec(spec)
            run_rows.append(summary)
            segment_rows.extend(segments)
            attempt_rows.extend(attempts)
            _write_required_outputs(out_dir, run_rows, segment_rows, attempt_rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_diagnostic_spec, spec) for spec in specs]
            for future in as_completed(futures):
                summary, segments, attempts = future.result()
                run_rows.append(summary)
                segment_rows.extend(segments)
                attempt_rows.extend(attempts)
                _write_required_outputs(out_dir, run_rows, segment_rows, attempt_rows)
    _write_required_outputs(out_dir, run_rows, segment_rows, attempt_rows)
    make_plots(out_dir, run_rows, segment_rows, attempt_rows)
    return run_rows, segment_rows, attempt_rows


def _run_label(row: Mapping[str, Any]) -> str:
    return f"{row['mode']} o{row['order']} s{row['substep_factor']}"


def _group_max(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str], value_field: str) -> dict[tuple[Any, ...], float]:
    out: dict[tuple[Any, ...], float] = {}
    for row in rows:
        value = _finite_float(row.get(value_field))
        if value is None:
            continue
        key = tuple(row.get(field) for field in group_fields)
        out[key] = max(out.get(key, -math.inf), value)
    return out


def _improvement_answer(rows: Sequence[Mapping[str, Any]], variable: str) -> str:
    if variable == "order":
        grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["mode"], row["substep_factor"])].append(row)
        improvements = []
        for key, group in grouped.items():
            by_order = {int(row["order"]): float(row["last_validated_t"]) for row in group}
            if 4 in by_order:
                best_order = max(by_order, key=lambda k: by_order[k])
                if by_order[best_order] > by_order[4] + 1e-12:
                    improvements.append(f"{key[0]} substep {key[1]}: order {best_order} reached {by_order[best_order]:.6g} vs order 4 at {by_order[4]:.6g}")
        return "Yes: " + "; ".join(improvements) if improvements else "No clear delay from higher order in these diagnostic runs."
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["order"])].append(row)
    improvements = []
    for key, group in grouped.items():
        by_factor = {int(row["substep_factor"]): float(row["last_validated_t"]) for row in group}
        if 1 in by_factor:
            best_factor = max(by_factor, key=lambda k: by_factor[k])
            if by_factor[best_factor] > by_factor[1] + 1e-12:
                improvements.append(f"{key[0]} order {key[1]}: substep {best_factor} reached {by_factor[best_factor]:.6g} vs factor 1 at {by_factor[1]:.6g}")
    return "Yes: " + "; ".join(improvements) if improvements else "No clear delay from smaller substeps in these diagnostic runs."


def _dominant_mechanism(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, float]]:
    finite_false = [row for row in rows if str(row.get("finite_residual", "")).lower() == "false"]
    metrics = {
        "polynomial range width": _max_field(rows, "polynomial_range_width_sum"),
        "interval remainder width": _max_field(rows, "remainder_width_sum"),
        "residual width": _max_field(rows, "residual_width_sum"),
    }
    numeric = {name: value for name, value in metrics.items() if isinstance(value, float)}
    if finite_false:
        return "non-finite arithmetic", numeric
    if not numeric:
        return "insufficient finite diagnostic data", numeric
    return max(numeric, key=numeric.get), numeric


def _default_mode_comparison(rows: Sequence[Mapping[str, Any]]) -> str:
    default = {row["mode"]: row for row in rows if int(row["order"]) == 4 and int(row["substep_factor"]) == 1}
    if {"range_only", "dependency_preserving"} - set(default):
        return "The default order-4, factor-1 comparison was not available."
    r = float(default["range_only"]["last_validated_t"])
    d = float(default["dependency_preserving"]["last_validated_t"])
    if d < r - 1e-12:
        return f"Yes. Dependency-preserving validated to {d:.6g}, while range-only validated to {r:.6g}."
    if d > r + 1e-12:
        return f"No. Dependency-preserving validated farther ({d:.6g}) than range-only ({r:.6g})."
    return f"They tied at last validated time {d:.6g}."


def _bottleneck_sentence(dominant: str, order_answer: str, substep_answer: str) -> str:
    order_helped = order_answer.startswith("Yes")
    substep_helped = substep_answer.startswith("Yes")
    if dominant == "non-finite arithmetic":
        return "The immediate bottleneck is non-finite arithmetic in validation, after earlier width growth."
    if dominant == "polynomial range width":
        return "The evidence points most strongly to interval polynomial range evaluation looseness in PyTorch's current representation."
    if dominant == "interval remainder width" or dominant == "residual width":
        if order_helped:
            return "The evidence points toward fixed-order truncation and remainder/residual growth as the main bottleneck."
        if substep_helped:
            return "The evidence points toward step size being too large for PyTorch's looser representation."
        return "The evidence points toward fixed-order remainder/residual growth within the current validation representation."
    return "The evidence is insufficient to isolate one bottleneck."


def write_diagnostic_report(out_dir: Path, run_rows: Sequence[Mapping[str, Any]], attempt_rows: Sequence[Mapping[str, Any]]) -> None:
    order_answer = _improvement_answer(run_rows, "order")
    substep_answer = _improvement_answer(run_rows, "substep_factor")
    mode_answer = _default_mode_comparison(run_rows)
    dominant, metric_values = _dominant_mechanism(attempt_rows)
    metric_text = ", ".join(f"{name} max {_fmt(value)}" for name, value in metric_values.items()) or "no finite maxima"
    bottleneck = _bottleneck_sentence(dominant, order_answer, substep_answer)
    table = [
        "| run | status | validated segments | last validated t | last attempted t | failure reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in run_rows:
        table.append(
            f"| `{row['run_id']}` | `{row['status']}` | {_fmt(row['validated_segments'])} | "
            f"{_fmt(row['last_validated_t'])} | {_fmt(row['last_attempted_t'])} | `{row['failure_reason']}` |"
        )
    text = f"""# Flow* Benchmark PyTorch TM Failure Diagnostics

This is a diagnostic report for the PyTorch Taylor-model failure on the original Flow* Van der Pol benchmark grid. It is diagnosis only, not a new reachability algorithm.

## Run Summary

{chr(10).join(table)}

## Questions

- Did higher order delay failure? {order_answer}
- Did smaller substeps delay failure? {substep_answer}
- Did dependency-preserving still fail earlier than range-only? {mode_answer}
- Is the blowup dominated by polynomial range width, interval remainder width, residual width, or non-finite arithmetic? Dominant signal: {dominant}; {metric_text}.
- What bottleneck does this suggest? {bottleneck}

The substep-factor runs split each original Flow* segment for PyTorch diagnostics only. They are not Flow* parity claims. No successful replacement algorithm is claimed here.
"""
    (out_dir / "diagnostic_report.md").write_text(text, encoding="utf-8", newline="\n")


def make_plots(
    out_dir: Path,
    run_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths: list[Path] = []
    by_run_segments: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        by_run_segments[str(row["run_id"])].append(row)
    by_run_attempts: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in attempt_rows:
        by_run_attempts[str(row["run_id"])].append(row)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for run_id, rows in by_run_segments.items():
        xs = [_finite_float(row.get("t_hi")) for row in rows if row.get("status") == "validated"]
        ys = [_finite_float(row.get("width_sum")) for row in rows if row.get("status") == "validated"]
        pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=1.0, label=run_id)
    ax.set_xlabel("t")
    ax.set_ylabel("validated segment width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5, ncols=2)
    fig.tight_layout()
    path = out_dir / PLOT_NAMES[0]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for run_id, rows in by_run_attempts.items():
        xs = [_finite_float(row.get("t_hi")) for row in rows]
        ys = [_finite_float(row.get("residual_width_sum")) for row in rows]
        pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and y > 0]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=1.0, label=run_id)
    ax.set_xlabel("t")
    ax.set_ylabel("residual width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5, ncols=2)
    fig.tight_layout()
    path = out_dir / PLOT_NAMES[1]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    xs = [_finite_float(row.get("polynomial_range_width_sum")) for row in attempt_rows]
    ys = [_finite_float(row.get("remainder_width_sum")) for row in attempt_rows]
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and x > 0 and y > 0]
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=12, alpha=0.55)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("polynomial range width sum")
    ax.set_ylabel("remainder width sum")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out_dir / PLOT_NAMES[2]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    labels = [str(row["run_id"]) for row in run_rows]
    values = [float(row["last_validated_t"]) for row in run_rows]
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("last validated t")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / PLOT_NAMES[3]
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose PyTorch TM failures on the original Flow* Van der Pol benchmark grid.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "flowstar_benchmark_diagnostics"))
    parser.add_argument("--orders", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--substep-factors", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--max-wall-s-per-run", type=float, default=120.0)
    parser.add_argument("--max-horizon", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=min(6, max(1, os.cpu_count() or 1)))
    parser.add_argument("--parity-dir", default=str(REPO_ROOT / "outputs" / "flowstar_benchmark_parity"))
    args = parser.parse_args()

    params, reference_segments = load_reference_inputs(Path(args.parity_dir))
    run_rows, segment_rows, attempt_rows = run_diagnostics(
        Path(args.out_dir),
        params,
        reference_segments,
        orders=args.orders,
        substep_factors=args.substep_factors,
        max_wall_s_per_run=args.max_wall_s_per_run,
        max_horizon=args.max_horizon,
        workers=args.workers,
    )
    print(f"wrote {args.out_dir}")
    print(f"runs={len(run_rows)} segments={len(segment_rows)} validation_attempts={len(attempt_rows)} workers={args.workers}")


if __name__ == "__main__":
    main()
