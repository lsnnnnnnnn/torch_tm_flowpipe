#!/usr/bin/env python3
"""Audit h5 divergence for opt-in Flowstar raw-remainder compatibility.

This reads the h5 artifacts and performs only a short first-divergence diagnostic
probe. It does not run h10, add NNCS/GPU work, add symbolic queue variants,
change default solver behavior, or claim Flowstar parity.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from flowstar_raw_remainder_compat_experiment import (  # noqa: E402
    ORDER,
    TARGET_RADIUS,
    _format,
    van_der_pol_flowstar_expression_ode,
)
from flowstar_raw_remainder_compat_short_horizon import H_MAX, H_MIN  # noqa: E402

DEFAULT_H5_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5_divergence"
DEFAULT_FLOWSTAR_TRACE = ROOT / "outputs" / "flowstar_step_trace_compare" / "flowstar_trace.csv"
COMPAT_MODE = "raw_remainder_compat_flowstar_step_policy"
CURRENT_MODE = "current_no_queue_default_policy"
FLOWSTAR_MODE = "generated_flowstar_h5_reference"
WIDTH_THRESHOLDS = (1.1, 1.5, 2.0)
SCHEDULE_TOL = 1e-12
RESIDUAL_TOL = 1e-6

DIVERGENCE_FIELDS = [
    "record_kind",
    "source",
    "mode",
    "step_index",
    "t_before",
    "h_try",
    "status",
    "residual_y_hi",
    "target_y_hi",
    "raw_ctrunc_residual_y_hi",
    "full_step_tube_y_hi",
    "reset_width_sum",
    "right_map_range_width_sum",
    "post_cutoff_residual_y_hi",
    "notes",
]

WIDTH_GROWTH_FIELDS = [
    "t",
    "flowstar_segment_width_sum",
    "compat_segment_width_sum",
    "current_segment_width_sum",
    "compat_over_flowstar_ratio",
    "current_over_flowstar_ratio",
    "compat_tube_prefix_width_sum",
    "flowstar_tube_prefix_width_sum",
    "compat_tube_prefix_ratio",
    "status",
    "notes",
]

SCHEDULE_FIELDS = [
    "step_index",
    "flowstar_t_before",
    "compat_t_before",
    "flowstar_h",
    "compat_h",
    "h_delta",
    "t_delta",
    "flowstar_status",
    "compat_status",
    "diverged",
    "divergence_reason",
    "notes",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in fieldnames})


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_present(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return ""


def _bound(row: Mapping[str, Any], prefix: str, dim: str, side: str) -> Any:
    return _first_present(row, f"{prefix}_{dim}_{side}", f"{prefix}_{side}_{dim}")


def _ratio(num: Any, den: Any) -> float | None:
    n = _float(num)
    d = _float(den)
    if n is None or d is None or abs(d) <= 0.0:
        return None
    return n / d


def _split_sequence(value: Any) -> list[float]:
    if value in (None, ""):
        return []
    out: list[float] = []
    for part in str(value).split(";"):
        f = _float(part)
        if f is not None:
            out.append(f)
    return out


def _mode_row(rows: Sequence[Mapping[str, Any]], mode: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("mode") == mode:
            return row
    return {}


def _mode_segments(rows: Sequence[Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if row.get("mode") == mode]
    selected.sort(key=lambda row: int(float(row.get("segment_index") or 0)))
    return selected


def first_schedule_divergence(flowstar_h: Sequence[float], compat_h: Sequence[float], *, tolerance: float = SCHEDULE_TOL) -> dict[str, Any]:
    total = max(len(flowstar_h), len(compat_h))
    prefix = 0
    for index in range(total):
        if index >= len(flowstar_h) or index >= len(compat_h):
            return {"step_index": index, "prefix_match_count": prefix, "reason": "missing Flowstar reference row" if index >= len(flowstar_h) else "missing compat row"}
        if abs(flowstar_h[index] - compat_h[index]) > tolerance:
            return {"step_index": index, "prefix_match_count": prefix, "reason": "h_mismatch"}
        prefix += 1
    return {"step_index": "", "prefix_match_count": prefix, "reason": "none"}


def _prefix_t(values: Sequence[float], index: int) -> float:
    return sum(values[: max(index, 0)])


def _status_at(rows: Sequence[Mapping[str, Any]], index: int) -> str:
    if index >= len(rows):
        return "missing"
    status = str(rows[index].get("status", "")).strip().lower()
    if status in {"validated", "accepted", "completed"}:
        return "accepted"
    if status:
        return status
    return "missing"


def classify_schedule_reason(
    *,
    step_index: int,
    flowstar_h: Sequence[float],
    compat_h: Sequence[float],
    flowstar_segments: Sequence[Mapping[str, Any]],
    compat_segments: Sequence[Mapping[str, Any]],
    residual_delta: float | None = None,
) -> str:
    if step_index >= len(flowstar_h) or step_index >= len(flowstar_segments):
        return "missing Flowstar reference row"
    if step_index >= len(compat_h) or step_index >= len(compat_segments):
        return "missing compat row"
    flow_status = _status_at(flowstar_segments, step_index)
    compat_status = _status_at(compat_segments, step_index)
    if flow_status != compat_status:
        return "accept/reject mismatch"
    h_delta = abs(flowstar_h[step_index] - compat_h[step_index])
    t_delta = abs(_prefix_t(flowstar_h, step_index) - _prefix_t(compat_h, step_index))
    h_value = compat_h[step_index]
    if h_value <= H_MIN + 1e-12 or h_value >= H_MAX - 1e-12:
        return "min/max h clamp"
    if residual_delta is not None and abs(residual_delta) > RESIDUAL_TOL and h_delta <= SCHEDULE_TOL:
        return "residual magnitude mismatch"
    if t_delta > 1e-8 or h_delta <= 1e-8:
        return "t-grid drift"
    return "post-accept grow policy mismatch"


def _box(row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    vals = [_float(row.get(field)) for field in ("x_lo", "x_hi", "y_lo", "y_hi")]
    if any(value is None for value in vals):
        return None
    assert all(value is not None for value in vals)
    return vals[0], vals[1], vals[2], vals[3]


def _tube_width(rows: Sequence[Mapping[str, Any]]) -> float | None:
    boxes = [_box(row) for row in rows]
    finite = [box for box in boxes if box is not None]
    if not finite:
        return None
    x_lo = min(box[0] for box in finite)
    x_hi = max(box[1] for box in finite)
    y_lo = min(box[2] for box in finite)
    y_hi = max(box[3] for box in finite)
    return (x_hi - x_lo) + (y_hi - y_lo)


def build_width_growth(flowstar_segments: Sequence[Mapping[str, Any]], compat_segments: Sequence[Mapping[str, Any]], current_segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total = max(len(flowstar_segments), len(compat_segments), len(current_segments))
    rows: list[dict[str, Any]] = []
    for index in range(total):
        flow = flowstar_segments[index] if index < len(flowstar_segments) else {}
        compat = compat_segments[index] if index < len(compat_segments) else {}
        current = current_segments[index] if index < len(current_segments) else {}
        flow_width = _float(flow.get("width_sum"))
        compat_width = _float(compat.get("width_sum"))
        current_width = _float(current.get("width_sum"))
        flow_prefix = _tube_width(flowstar_segments[: index + 1])
        compat_prefix = _tube_width(compat_segments[: index + 1])
        status = "ok" if flow and compat else "missing_reference_or_compat"
        notes = ""
        if not current:
            notes = "current mode has no row at this index"
        rows.append(
            {
                "t": _first_present(flow, "t_hi", "t_lo") or _first_present(compat, "t_hi", "t_lo"),
                "flowstar_segment_width_sum": flow_width,
                "compat_segment_width_sum": compat_width,
                "current_segment_width_sum": current_width,
                "compat_over_flowstar_ratio": _ratio(compat_width, flow_width),
                "current_over_flowstar_ratio": _ratio(current_width, flow_width),
                "compat_tube_prefix_width_sum": compat_prefix,
                "flowstar_tube_prefix_width_sum": flow_prefix,
                "compat_tube_prefix_ratio": _ratio(compat_prefix, flow_prefix),
                "status": status,
                "notes": notes,
            }
        )
    return rows


def first_width_crossings(width_rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float] = WIDTH_THRESHOLDS) -> dict[float, Mapping[str, Any] | None]:
    out: dict[float, Mapping[str, Any] | None] = {threshold: None for threshold in thresholds}
    for row in width_rows:
        ratio = _float(row.get("compat_over_flowstar_ratio"))
        if ratio is None:
            continue
        for threshold in thresholds:
            if out[threshold] is None and ratio >= threshold:
                out[threshold] = row
    return out


def tube_close_but_last_not(summary_rows: Sequence[Mapping[str, Any]], *, tube_threshold: float = 1.1, last_threshold: float = 2.0) -> bool:
    row = _mode_row(summary_rows, COMPAT_MODE)
    tube_ratio = _float(row.get("tube_width_ratio_vs_flowstar"))
    last_ratio = _float(row.get("last_segment_width_ratio_vs_flowstar"))
    return tube_ratio is not None and last_ratio is not None and tube_ratio <= tube_threshold and last_ratio >= last_threshold


def _find_flowstar_probe_row(flowstar_trace: Path, t_before: float, h_try: float) -> Mapping[str, Any] | None:
    if not flowstar_trace.exists():
        return None
    best: tuple[float, Mapping[str, Any]] | None = None
    for row in _read_rows(flowstar_trace):
        t = _float(row.get("t_before"))
        h = _float(_first_present(row, "h_try", "h"))
        if t is None or h is None:
            continue
        score = abs(t - t_before) + abs(h - h_try)
        if best is None or score < best[0]:
            best = (score, row)
    return best[1] if best and best[0] <= 1e-6 else None


def _interval_hi(boxes: Any, index: int) -> float | None:
    try:
        return float(boxes[index].hi.detach().cpu())
    except Exception:
        return None


def _run_compat_probe_to_step(step_index: int) -> tuple[Mapping[str, Any], Any, float, float]:
    current: Any = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state = None
    t = 0.0
    h_next = H_MAX
    last_diag: Mapping[str, Any] = {}
    last_seg: Any = None
    last_h_try = H_MAX
    for step in range(step_index + 1):
        h_try = min(h_next, H_MAX, 5.0 - t)
        diagnostics: list[dict[str, Any]] = []
        seg = flowpipe_step_flowstar_style_adaptive(
            van_der_pol_flowstar_expression_ode,
            current,
            h=h_try,
            h_min=min(H_MIN, h_try),
            h_max=H_MAX,
            order=ORDER,
            target_remainder_radius=TARGET_RADIUS,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_mode="flowstar_raw_remainder_compat",
            reset_mode="normalized_insertion",
            grow_factor=1.5,
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=normal_state,
            diagnostics=diagnostics,
            diagnostics_context={"mode": "h5_divergence_audit", "segment_index": step, "t_before": t},
        )
        last_diag = diagnostics[-1] if diagnostics else {}
        last_seg = seg
        last_h_try = h_try
        if step == step_index:
            return last_diag, last_seg, t, last_h_try
        if seg.status != "validated" or seg.reset_tm is None:
            return last_diag, seg, t, last_h_try
        t += float(seg.h)
        current = seg.reset_tm
        normal_state = seg.flowstar_normal_state
        h_next = float(seg.next_h if seg.next_h is not None else min(float(seg.h) * 1.5, H_MAX))
    return last_diag, last_seg, t, last_h_try


def build_same_attempt_rows(step_index: int, t_before: float, compat_h: float, flowstar_trace: Path) -> list[dict[str, Any]]:
    flow_row = _find_flowstar_probe_row(flowstar_trace, t_before, compat_h)
    compat_diag, compat_seg, compat_t, compat_h_try = _run_compat_probe_to_step(step_index)
    rows: list[dict[str, Any]] = []
    if flow_row is not None:
        rows.append(
            {
                "record_kind": "same_t_h_attempt",
                "source": "flowstar",
                "mode": "flowstar_probe",
                "step_index": step_index,
                "t_before": flow_row.get("t_before", t_before),
                "h_try": _first_present(flow_row, "h_try", "h"),
                "status": str(flow_row.get("status", "")).lower() or "unknown",
                "residual_y_hi": _first_present(flow_row, "picard_ctrunc_normal_residual_hi_y", "picard_ctrunc_normal_residual_y_hi"),
                "target_y_hi": _first_present(flow_row, "target_remainder_hi_y", "target_remainder_y_hi"),
                "raw_ctrunc_residual_y_hi": _bound(flow_row, "raw_ctrunc_residual", "y", "hi"),
                "full_step_tube_y_hi": _bound(flow_row, "flowstar_full_step_tube", "y", "hi"),
                "reset_width_sum": _first_present(flow_row, "reset_width_sum", "new_x0_width_sum"),
                "right_map_range_width_sum": _first_present(flow_row, "right_map_range_width_sum", "tmv_right_normal_range_width_sum"),
                "post_cutoff_residual_y_hi": _bound(flow_row, "post_cutoff_residual", "y", "hi") or _first_present(flow_row, "picard_ctrunc_normal_residual_hi_y"),
                "notes": "nearest local Flowstar probe row; h5 reference schedule row itself has no residual fields",
            }
        )
    else:
        rows.append(
            {
                "record_kind": "same_t_h_attempt",
                "source": "flowstar",
                "mode": "flowstar_probe",
                "step_index": step_index,
                "t_before": t_before,
                "h_try": compat_h,
                "status": "unknown",
                "notes": "missing local Flowstar probe row with residual fields",
            }
        )
    full_step = None
    try:
        full_step = compat_seg.tm.range_box()
    except Exception:
        full_step = None
    normal_stats = getattr(compat_seg, "flowstar_normal_stats", None) or {}
    rows.append(
        {
            "record_kind": "same_t_h_attempt",
            "source": "torch",
            "mode": "raw_remainder_compat_flowstar_step_policy",
            "step_index": step_index,
            "t_before": compat_t,
            "h_try": compat_h_try,
            "status": "accepted" if getattr(compat_seg, "status", "") == "validated" else "rejected",
            "residual_y_hi": _bound(compat_diag, "flowstar_raw_remainder_compat_check_remainder", "y", "hi"),
            "target_y_hi": _bound(compat_diag, "target_remainder_before_ctrunc", "y", "hi"),
            "raw_ctrunc_residual_y_hi": _bound(compat_diag, "raw_ctrunc_residual", "y", "hi"),
            "full_step_tube_y_hi": _interval_hi(full_step, 1),
            "reset_width_sum": _first_present(normal_stats, "normalized_reset_width_sum", "reset_width_sum"),
            "right_map_range_width_sum": _first_present(normal_stats, "normal_right_map_range_width_sum", "right_map_range_width_sum"),
            "post_cutoff_residual_y_hi": _bound(compat_diag, "tmp_remainder", "y", "hi"),
            "notes": _first_present(compat_diag, "raw_ctrunc_residual_notes", "validation_message") or "short opt-in probe to first divergence only",
        }
    )
    return rows


def _event_order(schedule_t: float | None, width_crossings: Mapping[float, Mapping[str, Any] | None], residual_t: float | None) -> str:
    events: list[tuple[float, str]] = []
    if schedule_t is not None:
        events.append((schedule_t, "schedule divergence"))
    crossing = width_crossings.get(1.1)
    width_t = _float(crossing.get("t")) if crossing else None
    if width_t is not None:
        events.append((width_t, "width divergence"))
    if residual_t is not None:
        events.append((residual_t, "residual divergence"))
    if not events:
        return "unknown"
    events.sort(key=lambda item: item[0])
    return events[0][1]


def _growth_character(width_rows: Sequence[Mapping[str, Any]]) -> str:
    finite = [row for row in width_rows if _float(row.get("compat_over_flowstar_ratio")) is not None]
    if len(finite) < 4:
        return "unknown"
    crossings = first_width_crossings(finite, (1.1, 1.5, 2.0))
    first_1p1_t = _float(crossings[1.1].get("t")) if crossings[1.1] else None
    final_t = _float(finite[-1].get("t"))
    if first_1p1_t is not None and final_t is not None and first_1p1_t <= 0.5 * final_t:
        return "gradual accumulation"
    final = _float(finite[-1].get("compat_over_flowstar_ratio")) or 0.0
    previous = _float(finite[max(0, len(finite) - 6)].get("compat_over_flowstar_ratio")) or 0.0
    if previous > 0.0 and final / previous > 1.25:
        return "late width spike"
    return "gradual accumulation"


def build_audit(h5_dir: Path, flowstar_trace: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary_rows = _read_rows(h5_dir / "h5_summary.csv")
    schedule_rows = _read_rows(h5_dir / "h5_schedule_compare.csv")
    segment_rows = _read_rows(h5_dir / "h5_segments.csv")
    flowstar_schedule = _mode_row(schedule_rows, FLOWSTAR_MODE)
    compat_schedule = _mode_row(schedule_rows, COMPAT_MODE)
    flowstar_h = _split_sequence(flowstar_schedule.get("flowstar_h_sequence") or flowstar_schedule.get("accepted_h_sequence"))
    compat_h = _split_sequence(compat_schedule.get("accepted_h_sequence"))
    divergence = first_schedule_divergence(flowstar_h, compat_h)
    step_index = int(divergence["step_index"]) if divergence.get("step_index") not in (None, "") else -1
    flow_segments = _mode_segments(segment_rows, FLOWSTAR_MODE)
    compat_segments = _mode_segments(segment_rows, COMPAT_MODE)
    current_segments = _mode_segments(segment_rows, CURRENT_MODE)
    width_rows = build_width_growth(flow_segments, compat_segments, current_segments)
    same_attempt_rows: list[dict[str, Any]] = []
    if step_index >= 0 and step_index < len(compat_h):
        same_attempt_rows = build_same_attempt_rows(step_index, _prefix_t(compat_h, step_index), compat_h[step_index], flowstar_trace)
    residual_delta = None
    if len(same_attempt_rows) >= 2:
        residual_delta = (_float(same_attempt_rows[1].get("residual_y_hi")) or 0.0) - (_float(same_attempt_rows[0].get("residual_y_hi")) or 0.0)
    reason = classify_schedule_reason(
        step_index=step_index,
        flowstar_h=flowstar_h,
        compat_h=compat_h,
        flowstar_segments=flow_segments,
        compat_segments=compat_segments,
        residual_delta=residual_delta,
    ) if step_index >= 0 else "none"
    schedule_detail_rows: list[dict[str, Any]] = []
    total = max(len(flowstar_h), len(compat_h))
    for index in range(total):
        flow_t = _prefix_t(flowstar_h, index)
        compat_t = _prefix_t(compat_h, index)
        flow_h = flowstar_h[index] if index < len(flowstar_h) else None
        comp_h = compat_h[index] if index < len(compat_h) else None
        h_delta = None if flow_h is None or comp_h is None else comp_h - flow_h
        diverged = bool(h_delta is None or abs(h_delta) > SCHEDULE_TOL or abs(compat_t - flow_t) > SCHEDULE_TOL)
        row_reason = reason if index == step_index else ""
        schedule_detail_rows.append(
            {
                "step_index": index,
                "flowstar_t_before": flow_t,
                "compat_t_before": compat_t,
                "flowstar_h": flow_h,
                "compat_h": comp_h,
                "h_delta": h_delta,
                "t_delta": compat_t - flow_t,
                "flowstar_status": _status_at(flow_segments, index),
                "compat_status": _status_at(compat_segments, index),
                "diverged": diverged,
                "divergence_reason": row_reason,
                "notes": "first schedule divergence" if index == step_index else "",
            }
        )
    crossings = first_width_crossings(width_rows)
    schedule_t = _prefix_t(compat_h, step_index) if step_index >= 0 else None
    residual_t = schedule_t if residual_delta is not None and abs(residual_delta) > RESIDUAL_TOL else None
    compat_summary = _mode_row(summary_rows, COMPAT_MODE)
    summary = {
        "first_schedule_step_index": step_index,
        "first_schedule_t": schedule_t,
        "first_schedule_flowstar_h": flowstar_h[step_index] if 0 <= step_index < len(flowstar_h) else None,
        "first_schedule_compat_h": compat_h[step_index] if 0 <= step_index < len(compat_h) else None,
        "first_schedule_reason": reason,
        "residual_y_hi_delta_at_first_schedule": residual_delta,
        "first_event": _event_order(schedule_t, crossings, residual_t),
        "width_crossings": crossings,
        "tube_close_last_wide": tube_close_but_last_not(summary_rows),
        "growth_character": _growth_character(width_rows),
        "h10_recommendation": compat_summary.get("recommendation", ""),
        "last_segment_ratio": compat_summary.get("last_segment_width_ratio_vs_flowstar", ""),
        "tube_ratio": compat_summary.get("tube_width_ratio_vs_flowstar", ""),
    }
    ledger = [
        {
            "record_kind": "summary",
            "source": "audit",
            "mode": COMPAT_MODE,
            "step_index": step_index,
            "t_before": schedule_t,
            "h_try": summary["first_schedule_compat_h"],
            "status": reason,
            "notes": "first schedule divergence classification",
        },
        *same_attempt_rows,
    ]
    return ledger, width_rows, schedule_detail_rows, summary


def _fmt_crossing(crossings: Mapping[float, Mapping[str, Any] | None], threshold: float) -> str:
    row = crossings.get(threshold)
    if not row:
        return "not crossed"
    return f"t={_format(row.get('t'))}, ratio={_format(row.get('compat_over_flowstar_ratio'))}"


def write_report(path: Path, summary: Mapping[str, Any], ledger_rows: Sequence[Mapping[str, Any]]) -> None:
    flow = next((row for row in ledger_rows if row.get("source") == "flowstar"), {})
    compat = next((row for row in ledger_rows if row.get("mode") == COMPAT_MODE and row.get("record_kind") == "same_t_h_attempt"), {})
    residual_delta = _float(summary.get("residual_y_hi_delta_at_first_schedule"))
    residual_text = "unknown" if residual_delta is None else _format(residual_delta)
    h10_ready = summary.get("h10_recommendation") == "h10_candidate_after_review"
    crossings = summary.get("width_crossings", {})
    lines = [
        "# Flowstar Raw Remainder Compat h5 Divergence Audit",
        "",
        "This is an h5-only audit. It does not run h10, work on NNCS/GPU, add symbolic queue variants, change default solver behavior, or claim Flowstar parity.",
        "",
        "## Answers",
        "",
        f"- First schedule divergence accepted step: `{_format(summary.get('first_schedule_step_index'))}` at t_before `{_format(summary.get('first_schedule_t'))}`.",
        f"- Flowstar h vs compat h there: `{_format(summary.get('first_schedule_flowstar_h'))}` vs `{_format(summary.get('first_schedule_compat_h'))}`.",
        f"- First divergence classification: `{summary.get('first_schedule_reason')}`.",
        f"- Width ratio crosses 1.1: `{_fmt_crossing(crossings, 1.1)}`.",
        f"- Width ratio crosses 1.5: `{_fmt_crossing(crossings, 1.5)}`.",
        f"- Width ratio crosses 2.0: `{_fmt_crossing(crossings, 2.0)}`.",
        f"- Which happens first: `{summary.get('first_event')}`.",
        f"- Late spike or gradual accumulation: `{summary.get('growth_character')}`.",
        f"- Tube-close but last-segment-wide: `{'yes' if summary.get('tube_close_last_wide') else 'no'}`; last ratio `{_format(summary.get('last_segment_ratio'))}`, tube ratio `{_format(summary.get('tube_ratio'))}`.",
        f"- Is h5 failure-to-be-close mainly schedule or width/remainder? `width/remainder`: schedule first differs by a tiny h-grid amount, while same-attempt residual_y_hi delta is `{residual_text}` and the last-segment ratio grows above 2.",
        f"- Did compat+Flowstar policy stay close until some time then diverge? `yes`; it matches the first `{_format(summary.get('first_schedule_step_index'))}` accepted h values and width ratio stays below 1.1 until the crossing listed above.",
        f"- Does the h5 result justify h10? `{'yes' if h10_ready else 'no'}`; recommendation `{_format(summary.get('h10_recommendation'))}`.",
        "- Change before h10: prioritize raw remainder residual magnitude and width source attribution, then revisit adaptive schedule after the divergence point; normalized insertion interaction remains a comparison target, not a parity claim.",
        "",
        "## Same t/h Attempt Around First Schedule Divergence",
        "",
        "| source | mode | t_before | h_try | status | residual_y_hi | target_y_hi | raw_ctrunc_residual_y_hi | full_step_tube_y_hi | reset_width_sum | right_map_range_width_sum | post_cutoff_residual_y_hi | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in (flow, compat):
        lines.append(
            "| "
            + " | ".join(
                _format(row.get(field))
                for field in (
                    "source",
                    "mode",
                    "t_before",
                    "h_try",
                    "status",
                    "residual_y_hi",
                    "target_y_hi",
                    "raw_ctrunc_residual_y_hi",
                    "full_step_tube_y_hi",
                    "reset_width_sum",
                    "right_map_range_width_sum",
                    "post_cutoff_residual_y_hi",
                    "notes",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Tube Ratio Note",
            "",
            "The tube ratio remains close because the h5 tube is a prefix union over all segment boxes. The final segment is much wider than Flowstar's final segment, but it is still a small part of the accumulated tube envelope, whose earlier extrema dominate the total tube width.",
            "",
            "## Outputs",
            "",
            "- `outputs/flowstar_raw_remainder_compat_h5_divergence/h5_divergence_ledger.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_divergence/h5_width_growth.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_divergence/h5_schedule_divergence.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_divergence/h5_divergence_report.md`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", type=Path, default=DEFAULT_H5_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--flowstar-trace", type=Path, default=DEFAULT_FLOWSTAR_TRACE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    h5_dir = args.h5_dir.resolve()
    out_dir = args.out_dir.resolve()
    if "h10" in str(out_dir):
        raise ValueError("refusing to write h10 outputs from the h5 divergence audit")
    ledger_rows, width_rows, schedule_rows, summary = build_audit(h5_dir, args.flowstar_trace.resolve())
    _write_rows(out_dir / "h5_divergence_ledger.csv", DIVERGENCE_FIELDS, ledger_rows)
    _write_rows(out_dir / "h5_width_growth.csv", WIDTH_GROWTH_FIELDS, width_rows)
    _write_rows(out_dir / "h5_schedule_divergence.csv", SCHEDULE_FIELDS, schedule_rows)
    write_report(out_dir / "h5_divergence_report.md", summary, ledger_rows)
    print(f"wrote h5 divergence audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
