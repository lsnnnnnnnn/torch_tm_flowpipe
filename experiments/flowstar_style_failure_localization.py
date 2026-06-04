#!/usr/bin/env python3
"""Localize the Flowstar-style Van der Pol target-remainder failure."""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe import (  # noqa: E402
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    taylor_model_mul_breakdown,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode  # noqa: E402
from torch_tm_flowpipe.safety import intervals_are_finite  # noqa: E402

import flowstar_style_rescue_vanderpol as rescue  # noqa: E402

RUN_ID = "flowstar_style_o6_target"
ORDER = 6
TARGET_REMAINDER_RADIUS = 1e-4
H_MIN = 0.002
H_MAX = 0.1

ATTEMPT_FIELDS = [
    "attempt_global_index",
    "t_start",
    "h_try",
    "t_end",
    "order",
    "accepted",
    "rejection_reason",
    "residual_width_x",
    "residual_width_y",
    "residual_width_sum",
    "target_width_x",
    "target_width_y",
    "target_width_sum",
    "residual_over_target_ratio_x",
    "residual_over_target_ratio_y",
    "residual_over_target_ratio_sum",
    "final_width_x",
    "final_width_y",
    "final_width_sum",
    "polynomial_range_width_x",
    "polynomial_range_width_y",
    "remainder_width_x",
    "remainder_width_y",
    "cutoff_uncertainty_width_x",
    "cutoff_uncertainty_width_y",
    "truncation_uncertainty_width_x",
    "truncation_uncertainty_width_y",
    "notes",
]

BREAKDOWN_FIELDS = [
    "t_start",
    "h_try",
    "expression",
    "kept_poly_range_width",
    "dropped_trunc_width",
    "polynomial_range_times_remainder_width",
    "remainder_times_remainder_width",
    "cutoff_uncertainty_width",
    "total_remainder_width",
    "finite",
]

SUMMARY_FIELDS = [
    "run_id",
    "status",
    "last_validated_t",
    "last_attempted_t",
    "num_accepted_steps",
    "num_rejected_attempts",
    "failure_reason",
    "failed_dimension_first",
    "failure_h_try",
    "residual_over_target_ratio_x",
    "residual_over_target_ratio_y",
    "residual_over_target_ratio_sum",
    "dominant_breakdown_component",
    "would_h_below_0.002_likely_help",
    "next_recommendation",
]


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


def _finite(value: Any) -> float | None:
    return rescue._finite_float(value)


def _safe_ratio(num: Any, den: Any) -> float | str:
    n = _finite(num)
    d = _finite(den)
    if n is None or d is None or d <= 0:
        return ""
    return n / d


def _width(value: Any) -> float | str:
    try:
        w = float(value.width().detach().cpu())
    except Exception:
        return ""
    return w if math.isfinite(w) else ""


def _component_sum(row: Mapping[str, Any]) -> float | str:
    a = _finite(row.get("p_self_times_other_remainder_width"))
    b = _finite(row.get("p_other_times_self_remainder_width"))
    if a is None and b is None:
        return ""
    return (a or 0.0) + (b or 0.0)


def _breakdown_row(context: Mapping[str, Any], expression: str, breakdown: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "t_start": context.get("t_start", ""),
        "h_try": context.get("h_try", context.get("h", "")),
        "segment_index": context.get("segment_index", ""),
        "attempt_index": context.get("attempt_index", ""),
        "order": context.get("order", ""),
        "expression": expression,
        "kept_poly_range_width": breakdown.get("kept_poly_range_width", ""),
        "dropped_trunc_width": breakdown.get("dropped_trunc_width", ""),
        "polynomial_range_times_remainder_width": _component_sum(breakdown),
        "remainder_times_remainder_width": breakdown.get("remainder_times_remainder_width", ""),
        "cutoff_uncertainty_width": 0.0,
        "total_remainder_width": breakdown.get("total_remainder_width", ""),
        "finite": breakdown.get("finite", ""),
    }


def _vdp_breakdown(candidate: TMVector, order: int, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        x = candidate[0]
        y = candidate[1]
        x_sq_breakdown = taylor_model_mul_breakdown(x, x, order)
        x_sq = x * x
        rows.append(_breakdown_row(context, "x*x", x_sq_breakdown))
        x_sq_y_breakdown = taylor_model_mul_breakdown(x_sq, y, order)
        x_sq_y = x_sq * y
        rows.append(_breakdown_row(context, "(x*x)*y", x_sq_y_breakdown))
        rhs_y = y - x - x_sq_y
        poly_range = rhs_y.polynomial.evaluate_interval(rhs_y.domain)
        total_range = rhs_y.range_box()
        finite = bool(poly_range.is_finite() and rhs_y.remainder.is_finite() and total_range.is_finite())
        rows.append(
            {
                "t_start": context.get("t_start", ""),
                "h_try": context.get("h_try", context.get("h", "")),
                "segment_index": context.get("segment_index", ""),
                "attempt_index": context.get("attempt_index", ""),
                "order": context.get("order", ""),
                "expression": "y - x - x*x*y",
                "kept_poly_range_width": _width(poly_range),
                "dropped_trunc_width": "",
                "polynomial_range_times_remainder_width": "",
                "remainder_times_remainder_width": "",
                "cutoff_uncertainty_width": 0.0,
                "total_remainder_width": _width(rhs_y.remainder),
                "finite": finite,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "t_start": context.get("t_start", ""),
                "h_try": context.get("h_try", context.get("h", "")),
                "segment_index": context.get("segment_index", ""),
                "attempt_index": context.get("attempt_index", ""),
                "order": context.get("order", ""),
                "expression": "breakdown_exception",
                "finite": False,
                "total_remainder_width": str(exc),
            }
        )
    return rows


def _finish_attempt_rows(rows: list[dict[str, Any]], start: int, *, t_start: float, global_start: int) -> None:
    for offset, row in enumerate(rows[start:]):
        row["attempt_global_index"] = global_start + offset
        row["t_start"] = t_start
        h = _finite(row.get("h_try")) or _finite(row.get("h")) or 0.0
        row["h_try"] = h
        row["t_end"] = t_start + h


def _is_rejected(row: Mapping[str, Any]) -> bool:
    return str(row.get("validation_status", "")).lower() != "validated" or str(row.get("subset_result", "")).lower() == "false"


def _target_widths() -> tuple[float, float, float]:
    width = 2.0 * TARGET_REMAINDER_RADIUS
    return width, width, 2.0 * width


def _truncation_width_for_attempt(breakdown_rows: Sequence[Mapping[str, Any]], attempt: Mapping[str, Any]) -> float | str:
    seg = str(attempt.get("segment_index", ""))
    idx = str(attempt.get("attempt_index", ""))
    vals = [
        _finite(row.get("dropped_trunc_width"))
        for row in breakdown_rows
        if str(row.get("segment_index", "")) == seg and str(row.get("attempt_index", "")) == idx
    ]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else ""


def _attempt_output_row(row: Mapping[str, Any], breakdown_rows: Sequence[Mapping[str, Any]], notes: str) -> dict[str, Any]:
    target_x, target_y, target_sum = _target_widths()
    trunc_y = _truncation_width_for_attempt(breakdown_rows, row)
    return {
        "attempt_global_index": row.get("attempt_global_index", ""),
        "t_start": row.get("t_start", ""),
        "h_try": row.get("h_try", row.get("h", "")),
        "t_end": row.get("t_end", ""),
        "order": row.get("order", ORDER),
        "accepted": not _is_rejected(row),
        "rejection_reason": row.get("rejection_reason") or row.get("validation_message", ""),
        "residual_width_x": row.get("residual_width_x", ""),
        "residual_width_y": row.get("residual_width_y", ""),
        "residual_width_sum": row.get("residual_width_sum", ""),
        "target_width_x": target_x,
        "target_width_y": target_y,
        "target_width_sum": target_sum,
        "residual_over_target_ratio_x": _safe_ratio(row.get("residual_width_x"), target_x),
        "residual_over_target_ratio_y": _safe_ratio(row.get("residual_width_y"), target_y),
        "residual_over_target_ratio_sum": _safe_ratio(row.get("residual_width_sum"), target_sum),
        "final_width_x": row.get("candidate_final_width_x", ""),
        "final_width_y": row.get("candidate_final_width_y", ""),
        "final_width_sum": row.get("candidate_final_width_sum", ""),
        "polynomial_range_width_x": row.get("polynomial_range_width_x", ""),
        "polynomial_range_width_y": row.get("polynomial_range_width_y", ""),
        "remainder_width_x": row.get("remainder_width_x", ""),
        "remainder_width_y": row.get("remainder_width_y", ""),
        "cutoff_uncertainty_width_x": 0.0,
        "cutoff_uncertainty_width_y": 0.0,
        "truncation_uncertainty_width_x": 0.0,
        "truncation_uncertainty_width_y": trunc_y,
        "notes": notes,
    }


def _dominant_component(breakdown_rows: Sequence[Mapping[str, Any]], attempt: Mapping[str, Any]) -> str:
    seg = str(attempt.get("segment_index", ""))
    idx = str(attempt.get("attempt_index", ""))
    totals = {
        "polynomial_range_times_remainder": 0.0,
        "truncation": 0.0,
        "remainder_times_remainder": 0.0,
        "cutoff_uncertainty": 0.0,
    }
    for row in breakdown_rows:
        if str(row.get("segment_index", "")) != seg or str(row.get("attempt_index", "")) != idx:
            continue
        totals["polynomial_range_times_remainder"] = max(
            totals["polynomial_range_times_remainder"], _finite(row.get("polynomial_range_times_remainder_width")) or 0.0
        )
        totals["truncation"] = max(totals["truncation"], _finite(row.get("dropped_trunc_width")) or 0.0)
        totals["remainder_times_remainder"] = max(
            totals["remainder_times_remainder"], _finite(row.get("remainder_times_remainder_width")) or 0.0
        )
        totals["cutoff_uncertainty"] = max(totals["cutoff_uncertainty"], _finite(row.get("cutoff_uncertainty_width")) or 0.0)
    return max(totals.items(), key=lambda kv: kv[1])[0]


def _failed_dimension(row: Mapping[str, Any]) -> str:
    rx = _finite(row.get("residual_over_target_ratio_x")) or 0.0
    ry = _finite(row.get("residual_over_target_ratio_y")) or 0.0
    if rx > 1.0 and ry > 1.0:
        return "x_and_y" if abs(rx - ry) < 1e-12 else ("x" if rx > ry else "y")
    if rx > 1.0:
        return "x"
    if ry > 1.0:
        return "y"
    return "none"


def _failed_dimension_from_bounds(row: Mapping[str, Any]) -> str:
    escaped = []
    for dim in ("x", "y"):
        lo = _finite(row.get(f"residual_lo_{dim}"))
        hi = _finite(row.get(f"residual_hi_{dim}"))
        if lo is not None and hi is not None and (lo < -TARGET_REMAINDER_RADIUS or hi > TARGET_REMAINDER_RADIUS):
            escaped.append(dim)
    if escaped:
        return "_and_".join(escaped)
    return _failed_dimension(row)


def _make_plots(out_dir: Path, attempts: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    pts = sorted(
        [
            (
                _finite(row.get("t_end")),
                _finite(row.get("residual_width_x")),
                _finite(row.get("residual_width_y")),
                _finite(row.get("residual_width_sum")),
            )
            for row in attempts
        ],
        key=lambda p: p[0] or 0.0,
    )
    pts = [p for p in pts if p[0] is not None]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label="residual x", marker="o", markersize=2.5)
        ax.plot([p[0] for p in pts], [p[2] for p in pts], label="residual y", marker="o", markersize=2.5)
        ax.plot([p[0] for p in pts], [p[3] for p in pts], label="residual sum", linewidth=1.2)
    target_x, _target_y, target_sum = _target_widths()
    ax.axhline(target_x, color="#111111", linestyle="--", linewidth=0.8, label="target per dimension")
    ax.axhline(target_sum, color="#666666", linestyle=":", linewidth=0.8, label="target sum")
    ax.set_xlabel("t")
    ax.set_ylabel("width")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_components_near_failure.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    accepted = [(row.get("t_hi"), row.get("h")) for row in segments if row.get("status") == "validated"]
    failed = [(row.get("t_start"), row.get("h_try")) for row in attempts if row.get("accepted") is False]
    if accepted:
        ax.plot([float(t) for t, _h in accepted], [float(h) for _t, h in accepted], marker="o", markersize=2.5, label="accepted h")
    if failed:
        ax.scatter([float(t) for t, _h in failed], [float(h) for _t, h in failed], color="#d62728", s=20, label="rejected h_try")
    ax.axhline(H_MIN, color="#111111", linestyle="--", linewidth=0.8, label="Flow* min step 0.002")
    ax.set_xlabel("t")
    ax.set_ylabel("h")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "step_size_near_failure.png", dpi=160)
    plt.close(fig)


def _write_report(
    out_dir: Path,
    summary: Mapping[str, Any],
    final_failure: Mapping[str, Any] | None,
    dominant: str,
) -> None:
    ratio_sum = final_failure.get("residual_over_target_ratio_sum", "") if final_failure else ""
    h_try = final_failure.get("h_try", "") if final_failure else ""
    slight = (_finite(ratio_sum) or math.inf) < 10.0
    lines = [
        "# Flowstar-Style Failure Localization Report",
        "",
        f"Last validated t: `{summary.get('last_validated_t', '')}`.",
        f"Failure reason: `{summary.get('failure_reason', '')}`.",
        f"Which state dimension fails target containment first? `{summary.get('failed_dimension_first', '')}`.",
        f"Is residual only slightly above target or orders of magnitude above? {'slight containment miss' if slight else 'orders of magnitude or unknown'}; final width-sum ratio=`{ratio_sum}`.",
        "Width ratios are diagnostic only: containment can fail when a residual interval is shifted outside the symmetric target even if its width is below the target width.",
        f"Is the dominant term still polynomial_range * remainder? {_yes_no(dominant == 'polynomial_range_times_remainder')}.",
        f"Is failure triggered by truncation, cutoff uncertainty, interval polynomial range, or RHS aggregation? Dominant recorded component: `{dominant}`.",
        f"What h_try fails at the end? `{h_try}`.",
        f"Would h below 0.002 likely help? `{summary.get('would_h_below_0.002_likely_help', '')}`.",
        f"Should the next fix be adaptive order, remainder-only Picard refinement, tighter range bounding, or real symbolic remainder queue? `{summary.get('next_recommendation', '')}`.",
        "",
        "## Output Files",
        "",
        "- `failure_step_attempts.csv` records the focused accepted/rejected attempts near failure.",
        "- `failure_residual_breakdown.csv` records the Van der Pol RHS multiplication breakdown.",
        "- `residual_components_near_failure.png` and `step_size_near_failure.png` visualize the failure neighborhood.",
    ]
    (out_dir / "failure_localization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def run_localization(out_dir: Path, *, max_horizon: float, wall_cap_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    current: Any = rescue._initial_box()
    t = 0.0
    h_request = H_MAX
    segment_index = 0
    attempt_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    status = "max_horizon_reached"
    failure_reason = ""
    last_attempted_t = 0.0
    start_time = time.perf_counter()

    def callback(candidate: TMVector, order: int, attempt: int, context: Mapping[str, Any]) -> None:
        callback_context = dict(context)
        callback_context["attempt_index"] = attempt
        breakdown_rows.extend(_vdp_breakdown(candidate, order, callback_context))

    while t < max_horizon - 1e-15:
        elapsed = time.perf_counter() - start_time
        if elapsed >= wall_cap_s:
            status = "timeout"
            failure_reason = f"wall-time cap reached before segment {segment_index}"
            break
        h = min(h_request, H_MAX, max_horizon - t)
        local_h_min = min(H_MIN, h)
        context = {
            "run_id": RUN_ID,
            "mode": "flowstar_style",
            "order": ORDER,
            "validation_mode": "target_remainder",
            "target_remainder_radius": TARGET_REMAINDER_RADIUS,
            "cutoff_threshold": "",
            "segment_index": segment_index,
            "t_start": t,
        }
        attempt_start = len(attempt_rows)
        global_start = len(attempt_rows)
        try:
            seg = _call_with_timeout(
                lambda: flowpipe_step_flowstar_style_adaptive(
                    van_der_pol_ode,
                    current,
                    h=h,
                    order=ORDER,
                    h_min=local_h_min,
                    h_max=H_MAX,
                    target_remainder_radius=TARGET_REMAINDER_RADIUS,
                    cutoff_threshold=None,
                    max_validation_attempts=2,
                    diagnostics=attempt_rows,
                    diagnostics_context=context,
                    rhs_breakdown_callback=callback,
                ),
                wall_cap_s - elapsed,
            )
        except StepTimeout as exc:
            status = "timeout"
            failure_reason = str(exc)
            break
        _finish_attempt_rows(attempt_rows, attempt_start, t_start=t, global_start=global_start)
        last_attempted_t = t + float(seg.h)
        final_box = seg.final_tm.range_box()
        finite = intervals_are_finite(final_box)
        row_status = "validated" if seg.status == "validated" and finite else "failed"
        x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = rescue._segment_bounds(final_box)
        segments.append(
            {
                "segment_index": segment_index,
                "status": row_status,
                "t_lo": t,
                "t_hi": t + float(seg.h),
                "h": float(seg.h),
                "width_x": width_x,
                "width_y": width_y,
                "width_sum": width_sum,
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "message": seg.message,
            }
        )
        if row_status != "validated":
            status = "failed"
            failure_reason = seg.message or "validation failed"
            break
        current = seg.reset_tm if seg.reset_tm is not None else seg.final_tm
        h_request = float(seg.next_h) if seg.next_h is not None else min(float(seg.h) * 1.5, H_MAX)
        t += float(seg.h)
        segment_index += 1

    accepted_segments = [row for row in segments if row.get("status") == "validated"]
    last20 = {row["segment_index"] for row in accepted_segments[-20:]}
    focused_raw = []
    for row in attempt_rows:
        t_start = _finite(row.get("t_start")) or 0.0
        in_last20 = row.get("segment_index") in last20
        rejected_after_2 = t_start > 2.0 and _is_rejected(row)
        if in_last20 or rejected_after_2:
            focused_raw.append(row)
    focused = []
    for row in focused_raw:
        note_bits = []
        if row.get("segment_index") in last20:
            note_bits.append("last_20_accepted_window")
        if (_finite(row.get("t_start")) or 0.0) > 2.0 and _is_rejected(row):
            note_bits.append("rejected_after_t_2")
        focused.append(_attempt_output_row(row, breakdown_rows, ";".join(note_bits)))

    kept_keys = {(str(row.get("segment_index", "")), str(row.get("attempt_index", ""))) for row in focused_raw}
    focused_breakdown = [
        {field: row.get(field, "") for field in BREAKDOWN_FIELDS}
        for row in breakdown_rows
        if (str(row.get("segment_index", "")), str(row.get("attempt_index", ""))) in kept_keys
    ]
    failure_pairs = [(raw, out) for raw, out in zip(focused_raw, focused) if out.get("accepted") is False]
    first_failure_raw, first_failure = failure_pairs[0] if failure_pairs else ({}, None)
    final_failure_raw, final_failure = failure_pairs[-1] if failure_pairs else ({}, None)
    dominant = _dominant_component(breakdown_rows, final_failure_raw) if final_failure else "unknown"
    failed_dim = _failed_dimension_from_bounds(first_failure_raw) if first_failure else "none"
    ratio_sum = _finite((final_failure or {}).get("residual_over_target_ratio_sum"))
    h_fail = _finite((final_failure or {}).get("h_try"))
    below_min_help = "unknown"
    if h_fail is not None and h_fail * 0.5 < H_MIN and ratio_sum is not None and ratio_sum < 4.0:
        below_min_help = "likely yes numerically, but only as a diagnostic because it goes below Flow* min step"
    recommendation = "tighter polynomial range bounding or real symbolic remainder queue"
    if dominant == "polynomial_range_times_remainder":
        recommendation = "tighter polynomial range bounding first, then real symbolic remainder queue"
    elif dominant == "truncation":
        recommendation = "adaptive order fallback first, then tighter polynomial range bounding"
    summary = {
        "run_id": RUN_ID,
        "status": status,
        "last_validated_t": accepted_segments[-1]["t_hi"] if accepted_segments else 0.0,
        "last_attempted_t": last_attempted_t,
        "num_accepted_steps": len(accepted_segments),
        "num_rejected_attempts": sum(1 for row in attempt_rows if _is_rejected(row)),
        "failure_reason": failure_reason,
        "failed_dimension_first": failed_dim,
        "failure_h_try": h_fail if h_fail is not None else "",
        "residual_over_target_ratio_x": (final_failure or {}).get("residual_over_target_ratio_x", ""),
        "residual_over_target_ratio_y": (final_failure or {}).get("residual_over_target_ratio_y", ""),
        "residual_over_target_ratio_sum": (final_failure or {}).get("residual_over_target_ratio_sum", ""),
        "dominant_breakdown_component": dominant,
        "would_h_below_0.002_likely_help": below_min_help,
        "next_recommendation": recommendation,
    }
    rescue._write_csv(out_dir / "failure_localization_summary.csv", SUMMARY_FIELDS, [summary])
    rescue._write_csv(out_dir / "failure_step_attempts.csv", ATTEMPT_FIELDS, focused)
    rescue._write_csv(out_dir / "failure_residual_breakdown.csv", BREAKDOWN_FIELDS, focused_breakdown)
    _write_report(out_dir, summary, final_failure, dominant)
    _make_plots(out_dir, focused, segments[-25:])
    return focused, focused_breakdown


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_style_failure_localization"))
    parser.add_argument("--max-horizon", type=float, default=2.2)
    parser.add_argument("--wall-cap-s", type=float, default=600.0)
    args = parser.parse_args(argv)
    run_localization(args.out_dir, max_horizon=float(args.max_horizon), wall_cap_s=float(args.wall_cap_s))
    print(f"wrote failure localization outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
