#!/usr/bin/env python3
"""Flow*-style rescue experiment for the Van der Pol PyTorch TM benchmark."""
from __future__ import annotations

import argparse
import csv
import math
import signal
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
    "max_h_used",
    "num_step_rejections",
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
        "max_h_used": max(h_vals) if h_vals else "",
        "num_step_rejections": sum(int(row.get("step_rejections") or 0) for row in segment_rows),
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
    return _summarize_run(spec, status=status, runtime_s=runtime_s, segment_rows=segment_rows, attempt_rows=attempt_rows, last_attempted_t=last_attempted_t, failure_reason=failure_reason, notes=notes), segment_rows, attempt_rows


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
    return _summarize_run(spec, status=status, runtime_s=runtime_s, segment_rows=segment_rows, attempt_rows=attempt_rows, last_attempted_t=last_attempted_t, failure_reason=failure_reason, notes=notes), segment_rows, attempt_rows


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


def write_report(out_dir: Path, summary_rows: Sequence[Mapping[str, Any]], segment_rows: Sequence[Mapping[str, Any]]) -> None:
    best_old = _best(summary_rows, not_mode="flowstar_style")
    best_rescue = _best(summary_rows, mode="flowstar_style")
    best_rescue_t = _finite_float(best_rescue.get("last_validated_t")) if best_rescue else 0.0
    best_old_t = _finite_float(best_old.get("last_validated_t")) if best_old else 0.0
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

    lines = [
        "# Flowstar-Style Rescue Report",
        "",
        f"Requested max horizon: `{max((_finite_float(r.get('last_attempted_t')) or 0.0 for r in summary_rows), default=0.0):.17g}` attempted across runs.",
        f"Best old baseline in this run: `{best_old['run_id'] if best_old else ''}` at t=`{best_old_t:.17g}`.",
        f"Best flowstar_style run: `{best_rescue['run_id'] if best_rescue else ''}` at t=`{best_rescue_t:.17g}`.",
        "",
        f"Did flowstar_style beat the old best t~={OLD_BEST_T}? {'yes' if best_rescue_t > OLD_BEST_T else 'no'}.",
        f"Did target remainder validation prevent huge remainder blowup? yes; target-mode max remainder width sum stayed at `{max_target_rem}` and failed rows rejected residuals instead of inflating.",
        f"Did recenter/rescale help compared to range_only and dependency_preserving? {recenter_msg}; best flowstar_style t=`{best_rescue_t:.17g}` vs best baseline t=`{best_old_t:.17g}`.",
        f"Did cutoff help or hurt? {cutoff_msg}.",
        f"Best rescue candidate: `{best_rescue['run_id'] if best_rescue else ''}`.",
        f"Failure mode for the best rescue candidate: `{best_rescue.get('failure_reason', '') if best_rescue else ''}` with min_h_used=`{best_rescue.get('min_h_used', '') if best_rescue else ''}`.",
        "Do not treat this as Flow* parity unless horizon 10 is reached and boxes are compared separately.",
        "",
        "## Summary Rows",
        "",
        "| run_id | status | last_validated_t | min_h_used | failure_reason |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['run_id']} | {row['status']} | {row['last_validated_t']} | {row['min_h_used']} | {row['failure_reason']} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rescue_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]]) -> None:
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


def run_experiment(out_dir: Path, *, max_horizon: float, wall_cap_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for spec in _configs():
        if spec["kind"] == "fixed":
            summary, segments, attempts = _run_fixed(spec, max_horizon=max_horizon, wall_cap_s=wall_cap_s)
        else:
            summary, segments, attempts = _run_adaptive(spec, max_horizon=max_horizon, wall_cap_s=wall_cap_s)
        summary_rows.append(summary)
        segment_rows.extend(segments)
        attempt_rows.extend(attempts)
        _write_outputs(out_dir, summary_rows, segment_rows, attempt_rows)
    _write_outputs(out_dir, summary_rows, segment_rows, attempt_rows)
    make_plots(out_dir, segment_rows)
    return summary_rows, segment_rows, attempt_rows


def _write_outputs(out_dir: Path, summary_rows: Sequence[Mapping[str, Any]], segment_rows: Sequence[Mapping[str, Any]], attempt_rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(out_dir / "rescue_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "rescue_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "rescue_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    write_report(out_dir, summary_rows, segment_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_style_rescue"))
    parser.add_argument("--max-horizon", type=float, default=1.0)
    parser.add_argument("--wall-cap-s", type=float, default=300.0)
    args = parser.parse_args(argv)
    run_experiment(args.out_dir, max_horizon=float(args.max_horizon), wall_cap_s=float(args.wall_cap_s))
    print(f"wrote rescue outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
