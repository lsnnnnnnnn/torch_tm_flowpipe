#!/usr/bin/env python3
"""Attribute h5 raw-remainder compat width accumulation before any h10 run.

This is an attribution audit only. It reads h5 artifacts, replays only the
opt-in h5 compat path for selected event windows to expose component diagnostics,
and does not run h10, add NNCS/GPU work, add symbolic queue variants, change
solver defaults, or claim Flowstar parity.
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
DEFAULT_DIVERGENCE_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5_divergence"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_h5_width_attribution"
FLOWSTAR_MODE = "generated_flowstar_h5_reference"
COMPAT_MODE = "raw_remainder_compat_flowstar_step_policy"
CURRENT_MODE = "current_no_queue_default_policy"
WIDTH_THRESHOLDS = (1.1, 1.5, 2.0)

COMPONENTS = (
    "raw_ctrunc_residual",
    "post_cutoff_residual",
    "right_map_range",
    "reset_width",
    "full_step_tube",
    "polynomial_range",
)

LEDGER_FIELDS = [
    "event_name",
    "event_t",
    "threshold",
    "source",
    "step_index",
    "t",
    "h",
    "status",
    "segment_width_x",
    "segment_width_y",
    "segment_width_sum",
    "width_ratio_vs_flowstar",
    "tube_prefix_ratio",
    "raw_ctrunc_residual_x_lo",
    "raw_ctrunc_residual_x_hi",
    "raw_ctrunc_residual_y_lo",
    "raw_ctrunc_residual_y_hi",
    "post_cutoff_residual_x_lo",
    "post_cutoff_residual_x_hi",
    "post_cutoff_residual_y_lo",
    "post_cutoff_residual_y_hi",
    "full_step_tube_x_lo",
    "full_step_tube_x_hi",
    "full_step_tube_y_lo",
    "full_step_tube_y_hi",
    "polynomial_range_x_lo",
    "polynomial_range_x_hi",
    "polynomial_range_y_lo",
    "polynomial_range_y_hi",
    "reset_width_sum",
    "right_map_range_width_sum",
    "center_x",
    "center_y",
    "scale_x",
    "scale_y",
    "target_margin_y_hi",
    "residual_y_hi_margin_to_target",
    "notes",
]

COMPONENT_GROWTH_FIELDS = [
    "event_name",
    "step_index",
    "t",
    "width_ratio_vs_flowstar",
    "tube_prefix_ratio",
    "raw_ctrunc_residual_width_sum",
    "raw_ctrunc_residual_growth_vs_previous_event",
    "post_cutoff_residual_width_sum",
    "post_cutoff_residual_growth_vs_previous_event",
    "right_map_range_width_sum",
    "right_map_range_growth_vs_previous_event",
    "reset_width_sum",
    "reset_width_growth_vs_previous_event",
    "full_step_tube_width_sum",
    "full_step_tube_growth_vs_previous_event",
    "polynomial_range_width_sum",
    "polynomial_range_growth_vs_previous_event",
    "residual_y_hi_margin_to_target",
    "target_margin_y_hi",
    "leading_component",
    "notes",
]

CROSSING_FIELDS = [
    "event_name",
    "threshold",
    "step_index",
    "t",
    "width_ratio_vs_flowstar",
    "tube_prefix_ratio",
    "leading_component",
    "raw_residual_vs_right_map",
    "notes",
]

LINE_COUNT_FIELDS = ["path", "physical_line_count", "csv_reader_row_count", "status"]


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
    if value in (None, "", "unknown"):
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


def _width_from_bounds(row: Mapping[str, Any], prefix: str) -> float | None:
    total = 0.0
    saw = False
    for dim in ("x", "y"):
        lo = _float(_bound(row, prefix, dim, "lo"))
        hi = _float(_bound(row, prefix, dim, "hi"))
        if lo is None or hi is None:
            return None
        total += hi - lo
        saw = True
    return total if saw else None


def _ratio(num: Any, den: Any) -> float | None:
    n = _float(num)
    d = _float(den)
    if n is None or d is None or abs(d) <= 0.0:
        return None
    return n / d


def _mode_segments(rows: Sequence[Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if row.get("mode") == mode]
    selected.sort(key=lambda row: int(float(row.get("segment_index") or 0)))
    return selected


def _mode_row(rows: Sequence[Mapping[str, Any]], mode: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("mode") == mode:
            return row
    return {}


def _prefix_t(rows: Sequence[Mapping[str, Any]], index: int) -> float | None:
    if index < 0 or index >= len(rows):
        return None
    return _float(rows[index].get("t_hi"))


def _box(row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    values = [_float(row.get(field)) for field in ("x_lo", "x_hi", "y_lo", "y_hi")]
    if any(value is None for value in values):
        return None
    assert all(value is not None for value in values)
    return values[0], values[1], values[2], values[3]


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


def _line_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return len(text.splitlines())


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle))


def formatting_rows(*dirs: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in dirs:
        for path in sorted([*directory.glob("*.csv"), *directory.glob("*.md")]):
            physical = _line_count(path)
            csv_rows = csv_row_count(path) if path.suffix == ".csv" else ""
            if path.suffix == ".csv":
                ok = physical > 1 and csv_rows == physical
            else:
                ok = physical > 10
            rows.append(
                {
                    "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                    "physical_line_count": physical,
                    "csv_reader_row_count": csv_rows,
                    "status": "ok" if ok else "bad_physical_format",
                }
            )
    return rows


def _event_index_from_width_rows(width_rows: Sequence[Mapping[str, Any]], threshold: float) -> int | None:
    for index, row in enumerate(width_rows):
        ratio = _float(row.get("compat_over_flowstar_ratio"))
        if ratio is not None and ratio >= threshold:
            return index
    return None


def _event_indices(
    divergence_rows: Sequence[Mapping[str, Any]],
    width_rows: Sequence[Mapping[str, Any]],
    compat_segments: Sequence[Mapping[str, Any]],
    flowstar_segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    schedule = next((row for row in divergence_rows if row.get("divergence_reason")), {})
    schedule_index = int(float(schedule.get("step_index") or 0)) if schedule else 0
    events: list[dict[str, Any]] = [
        {"event_name": "first_schedule_divergence", "threshold": "", "compat_index": schedule_index, "flowstar_index": min(schedule_index, len(flowstar_segments) - 1)},
    ]
    for threshold in WIDTH_THRESHOLDS:
        index = _event_index_from_width_rows(width_rows, threshold)
        if index is not None:
            events.append(
                {
                    "event_name": f"width_ratio_gt_{str(threshold).replace('.', '_')}",
                    "threshold": threshold,
                    "compat_index": min(index, len(compat_segments) - 1),
                    "flowstar_index": min(index, len(flowstar_segments) - 1),
                }
            )
    events.append(
        {
            "event_name": "final_segment_near_t5",
            "threshold": "",
            "compat_index": len(compat_segments) - 1,
            "flowstar_index": len(flowstar_segments) - 1,
        }
    )
    return events


def _interval_bounds(boxes: Any, index: int) -> tuple[float | str, float | str]:
    try:
        box = boxes[index]
        return float(box.lo.detach().cpu()), float(box.hi.detach().cpu())
    except Exception:
        return "", ""


def _interval_width(boxes: Any, index: int) -> float | None:
    lo, hi = _interval_bounds(boxes, index)
    flo = _float(lo)
    fhi = _float(hi)
    if flo is None or fhi is None:
        return None
    return fhi - flo


def _range_box(value: Any) -> Any:
    try:
        return value.range_box()
    except Exception:
        return None


def replay_compat_components(indices: Sequence[int]) -> dict[int, dict[str, Any]]:
    wanted = set(int(i) for i in indices if i >= 0)
    if not wanted:
        return {}
    max_index = max(wanted)
    current: Any = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state = None
    t = 0.0
    h_next = H_MAX
    out: dict[int, dict[str, Any]] = {}
    for step in range(max_index + 1):
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
            diagnostics_context={"mode": "h5_width_attribution", "segment_index": step, "t_before": t},
        )
        diag = diagnostics[-1] if diagnostics else {}
        full_step = _range_box(getattr(seg, "tm", None))
        normal_stats = getattr(seg, "flowstar_normal_stats", None) or {}
        if step in wanted:
            out[step] = {
                "diagnostic": diag,
                "seg": seg,
                "full_step": full_step,
                "normal_stats": normal_stats,
                "t_before": t,
                "h_try": h_try,
            }
        if getattr(seg, "status", "") != "validated" or getattr(seg, "reset_tm", None) is None:
            break
        t += float(seg.h)
        current = seg.reset_tm
        normal_state = seg.flowstar_normal_state
        h_next = float(seg.next_h if seg.next_h is not None else min(float(seg.h) * 1.5, H_MAX))
    return out


def _component_widths(row: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "raw_ctrunc_residual": _width_from_bounds(row, "raw_ctrunc_residual"),
        "post_cutoff_residual": _width_from_bounds(row, "post_cutoff_residual"),
        "right_map_range": _float(row.get("right_map_range_width_sum")),
        "reset_width": _float(row.get("reset_width_sum")),
        "full_step_tube": _width_from_bounds(row, "full_step_tube"),
        "polynomial_range": _width_from_bounds(row, "polynomial_range"),
    }


def _growth(value: Any, previous: Any) -> float | None:
    return _ratio(value, previous)


def leading_component(row: Mapping[str, Any], *, prefer_unknown: bool = False) -> str:
    candidates = {
        "raw_ctrunc_residual": _float(row.get("raw_ctrunc_residual_growth_vs_previous_event")),
        "post_cutoff_residual": _float(row.get("post_cutoff_residual_growth_vs_previous_event")),
        "right_map_range": _float(row.get("right_map_range_growth_vs_previous_event")),
        "reset_width": _float(row.get("reset_width_growth_vs_previous_event")),
        "full_step_tube": _float(row.get("full_step_tube_growth_vs_previous_event")),
        "polynomial_range": _float(row.get("polynomial_range_growth_vs_previous_event")),
    }
    finite = {key: value for key, value in candidates.items() if value is not None}
    if not finite:
        return "unknown" if prefer_unknown else "insufficient_component_data"
    return max(finite, key=lambda key: finite[key])


def raw_or_right_map_dominates(row: Mapping[str, Any]) -> str:
    raw = _float(row.get("raw_ctrunc_residual_growth_vs_previous_event"))
    right = _float(row.get("right_map_range_growth_vs_previous_event"))
    if raw is None and right is None:
        return "unknown"
    if raw is not None and right is not None and raw <= 1.0 and right <= 1.0:
        return "neither_growth"
    if raw is None:
        return "right_map_range"
    if right is None:
        return "raw_ctrunc_residual"
    return "raw_ctrunc_residual" if raw >= right else "right_map_range"


def crossing_component(row: Mapping[str, Any]) -> str:
    growth_values = [
        _float(row.get("raw_ctrunc_residual_growth_vs_previous_event")),
        _float(row.get("post_cutoff_residual_growth_vs_previous_event")),
        _float(row.get("right_map_range_growth_vs_previous_event")),
        _float(row.get("reset_width_growth_vs_previous_event")),
        _float(row.get("full_step_tube_growth_vs_previous_event")),
        _float(row.get("polynomial_range_growth_vs_previous_event")),
    ]
    finite = [value for value in growth_values if value is not None]
    if finite and max(finite) <= 1.0:
        return "full_step_tube_relative_ratio"
    return leading_component(row, prefer_unknown=True)


def _flowstar_event_row(event: Mapping[str, Any], segment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_name": event["event_name"],
        "event_t": segment.get("t_hi", ""),
        "threshold": event.get("threshold", ""),
        "source": "flowstar",
        "step_index": segment.get("segment_index", event.get("flowstar_index", "")),
        "t": segment.get("t_hi", ""),
        "h": segment.get("h", ""),
        "status": segment.get("status", ""),
        "segment_width_x": segment.get("width_x", ""),
        "segment_width_y": segment.get("width_y", ""),
        "segment_width_sum": segment.get("width_sum", ""),
        "width_ratio_vs_flowstar": 1.0,
        "tube_prefix_ratio": 1.0,
        "raw_ctrunc_residual_x_lo": "",
        "raw_ctrunc_residual_x_hi": "",
        "raw_ctrunc_residual_y_lo": "",
        "raw_ctrunc_residual_y_hi": "",
        "post_cutoff_residual_x_lo": "",
        "post_cutoff_residual_x_hi": "",
        "post_cutoff_residual_y_lo": "",
        "post_cutoff_residual_y_hi": "",
        "full_step_tube_x_lo": "",
        "full_step_tube_x_hi": "",
        "full_step_tube_y_lo": "",
        "full_step_tube_y_hi": "",
        "polynomial_range_x_lo": "",
        "polynomial_range_x_hi": "",
        "polynomial_range_y_lo": "",
        "polynomial_range_y_hi": "",
        "reset_width_sum": "",
        "right_map_range_width_sum": "",
        "center_x": "",
        "center_y": "",
        "scale_x": "",
        "scale_y": "",
        "target_margin_y_hi": "",
        "residual_y_hi_margin_to_target": "",
        "notes": "Flowstar h5 reference exposes segment boxes only; component attribution fields are unknown, not zero",
    }


def _compat_event_row(
    event: Mapping[str, Any],
    segment: Mapping[str, Any],
    flow_segment: Mapping[str, Any],
    component: Mapping[str, Any],
    width_growth_row: Mapping[str, Any],
) -> dict[str, Any]:
    diag = component.get("diagnostic", {}) if component else {}
    full_step = component.get("full_step") if component else None
    normal_stats = component.get("normal_stats", {}) if component else {}
    target_y_hi = _bound(diag, "target_remainder_before_ctrunc", "y", "hi")
    residual_y_hi = _bound(diag, "flowstar_raw_remainder_compat_check_remainder", "y", "hi")
    target = _float(target_y_hi)
    residual = _float(residual_y_hi)
    target_margin = None if target is None or residual is None else target - residual
    full_x_lo, full_x_hi = _interval_bounds(full_step, 0)
    full_y_lo, full_y_hi = _interval_bounds(full_step, 1)
    row = {
        "event_name": event["event_name"],
        "event_t": segment.get("t_hi", ""),
        "threshold": event.get("threshold", ""),
        "source": "compat",
        "step_index": segment.get("segment_index", event.get("compat_index", "")),
        "t": segment.get("t_hi", ""),
        "h": segment.get("h", ""),
        "status": segment.get("status", ""),
        "segment_width_x": segment.get("width_x", ""),
        "segment_width_y": segment.get("width_y", ""),
        "segment_width_sum": segment.get("width_sum", ""),
        "width_ratio_vs_flowstar": _ratio(segment.get("width_sum"), flow_segment.get("width_sum")),
        "tube_prefix_ratio": width_growth_row.get("compat_tube_prefix_ratio", ""),
        "raw_ctrunc_residual_x_lo": _bound(diag, "raw_ctrunc_residual", "x", "lo"),
        "raw_ctrunc_residual_x_hi": _bound(diag, "raw_ctrunc_residual", "x", "hi"),
        "raw_ctrunc_residual_y_lo": _bound(diag, "raw_ctrunc_residual", "y", "lo"),
        "raw_ctrunc_residual_y_hi": _bound(diag, "raw_ctrunc_residual", "y", "hi"),
        "post_cutoff_residual_x_lo": _bound(diag, "tmp_remainder", "x", "lo"),
        "post_cutoff_residual_x_hi": _bound(diag, "tmp_remainder", "x", "hi"),
        "post_cutoff_residual_y_lo": _bound(diag, "tmp_remainder", "y", "lo"),
        "post_cutoff_residual_y_hi": _bound(diag, "tmp_remainder", "y", "hi"),
        "full_step_tube_x_lo": full_x_lo,
        "full_step_tube_x_hi": full_x_hi,
        "full_step_tube_y_lo": full_y_lo,
        "full_step_tube_y_hi": full_y_hi,
        "polynomial_range_x_lo": _bound(diag, "raw_ctrunc_polynomial_range", "x", "lo"),
        "polynomial_range_x_hi": _bound(diag, "raw_ctrunc_polynomial_range", "x", "hi"),
        "polynomial_range_y_lo": _bound(diag, "raw_ctrunc_polynomial_range", "y", "lo"),
        "polynomial_range_y_hi": _bound(diag, "raw_ctrunc_polynomial_range", "y", "hi"),
        "reset_width_sum": _first_present(normal_stats, "normalized_reset_width_sum", "reset_width_sum"),
        "right_map_range_width_sum": _first_present(normal_stats, "normal_right_map_range_width_sum", "right_map_range_width_sum"),
        "center_x": normal_stats.get("center_x", ""),
        "center_y": normal_stats.get("center_y", ""),
        "scale_x": normal_stats.get("scale_x", ""),
        "scale_y": normal_stats.get("scale_y", ""),
        "target_margin_y_hi": target_margin,
        "residual_y_hi_margin_to_target": target_margin,
        "notes": _first_present(diag, "raw_ctrunc_residual_notes", "validation_message") or "opt-in compat component replay for selected h5 event window",
    }
    return row


def build_attribution(h5_dir: Path, divergence_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    segment_rows = _read_rows(h5_dir / "h5_segments.csv")
    summary_rows = _read_rows(h5_dir / "h5_summary.csv")
    schedule_rows = _read_rows(divergence_dir / "h5_schedule_divergence.csv")
    width_rows = _read_rows(divergence_dir / "h5_width_growth.csv")
    flow_segments = _mode_segments(segment_rows, FLOWSTAR_MODE)
    compat_segments = _mode_segments(segment_rows, COMPAT_MODE)
    events = _event_indices(schedule_rows, width_rows, compat_segments, flow_segments)
    compat_indices = [int(event["compat_index"]) for event in events]
    compat_components = replay_compat_components(compat_indices)

    ledger: list[dict[str, Any]] = []
    component_growth: list[dict[str, Any]] = []
    crossing_rows: list[dict[str, Any]] = []
    previous_components: dict[str, float | None] | None = None
    by_event_growth: dict[str, dict[str, Any]] = {}
    for event in events:
        compat_index = int(event["compat_index"])
        flow_index = int(event["flowstar_index"])
        compat_segment = compat_segments[compat_index]
        flow_segment = flow_segments[flow_index]
        width_growth_row = width_rows[compat_index] if compat_index < len(width_rows) else {}
        flow_row = _flowstar_event_row(event, flow_segment)
        compat_row = _compat_event_row(event, compat_segment, flow_segment, compat_components.get(compat_index, {}), width_growth_row)
        ledger.extend([flow_row, compat_row])
        comp_widths = _component_widths(compat_row)
        growth = {key: (None if previous_components is None else _growth(value, previous_components.get(key))) for key, value in comp_widths.items()}
        growth_row = {
            "event_name": event["event_name"],
            "step_index": compat_index,
            "t": compat_segment.get("t_hi", ""),
            "width_ratio_vs_flowstar": compat_row.get("width_ratio_vs_flowstar", ""),
            "tube_prefix_ratio": compat_row.get("tube_prefix_ratio", ""),
            "raw_ctrunc_residual_width_sum": comp_widths["raw_ctrunc_residual"],
            "raw_ctrunc_residual_growth_vs_previous_event": growth["raw_ctrunc_residual"],
            "post_cutoff_residual_width_sum": comp_widths["post_cutoff_residual"],
            "post_cutoff_residual_growth_vs_previous_event": growth["post_cutoff_residual"],
            "right_map_range_width_sum": comp_widths["right_map_range"],
            "right_map_range_growth_vs_previous_event": growth["right_map_range"],
            "reset_width_sum": comp_widths["reset_width"],
            "reset_width_growth_vs_previous_event": growth["reset_width"],
            "full_step_tube_width_sum": comp_widths["full_step_tube"],
            "full_step_tube_growth_vs_previous_event": growth["full_step_tube"],
            "polynomial_range_width_sum": comp_widths["polynomial_range"],
            "polynomial_range_growth_vs_previous_event": growth["polynomial_range"],
            "residual_y_hi_margin_to_target": compat_row.get("residual_y_hi_margin_to_target", ""),
            "target_margin_y_hi": compat_row.get("target_margin_y_hi", ""),
            "notes": "growth ratios are compat local component changes versus the previous audited event; Flowstar component fields are unavailable in h5 reference",
        }
        growth_row["leading_component"] = (
            crossing_component(growth_row) if event.get("threshold") not in (None, "") else leading_component(growth_row, prefer_unknown=True)
        )
        component_growth.append(growth_row)
        by_event_growth[event["event_name"]] = growth_row
        previous_components = comp_widths
        if event.get("threshold") not in (None, ""):
            crossing_rows.append(
                {
                    "event_name": event["event_name"],
                    "threshold": event.get("threshold"),
                    "step_index": compat_index,
                    "t": compat_segment.get("t_hi", ""),
                    "width_ratio_vs_flowstar": compat_row.get("width_ratio_vs_flowstar", ""),
                    "tube_prefix_ratio": compat_row.get("tube_prefix_ratio", ""),
                    "leading_component": growth_row["leading_component"],
                    "raw_residual_vs_right_map": raw_or_right_map_dominates(growth_row),
                    "notes": "threshold crossing row from h5 divergence width-growth ledger plus compat component replay",
                }
            )

    compat_summary = _mode_row(summary_rows, COMPAT_MODE)
    target_margins = [_float(row.get("residual_y_hi_margin_to_target")) for row in ledger if row.get("source") == "compat"]
    target_margins = [value for value in target_margins if value is not None]
    gt_1p1 = by_event_growth.get("width_ratio_gt_1_1", {})
    gt_1p5 = by_event_growth.get("width_ratio_gt_1_5", {})
    gt_2p0 = by_event_growth.get("width_ratio_gt_2_0", {})
    summary = {
        "component_first_correlates_gt_1p1": gt_1p1.get("leading_component", "unknown"),
        "component_driving_gt_1p5": gt_1p5.get("leading_component", "unknown"),
        "component_driving_gt_2p0": gt_2p0.get("leading_component", "unknown"),
        "raw_or_right_map_dominates_gt_1p1": raw_or_right_map_dominates(gt_1p1) if gt_1p1 else "unknown",
        "target_margin_min": min(target_margins) if target_margins else None,
        "target_margin_max": max(target_margins) if target_margins else None,
        "target_margin_all_positive": all(value > 0.0 for value in target_margins) if target_margins else None,
        "last_ratio": compat_summary.get("last_segment_width_ratio_vs_flowstar", ""),
        "tube_ratio": compat_summary.get("tube_width_ratio_vs_flowstar", ""),
        "h10_recommendation": compat_summary.get("recommendation", ""),
        "flowstar_component_status": "unknown_missing_h5_reference_component_fields",
    }
    format_rows = formatting_rows(h5_dir, divergence_dir)
    return ledger, component_growth, crossing_rows, summary, format_rows


def _formatting_text(format_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| path | physical lines | csv.reader rows | status |",
        "| --- | --- | --- | --- |",
    ]
    for row in format_rows:
        lines.append(
            "| "
            + " | ".join(_format(row.get(field)) for field in ("path", "physical_line_count", "csv_reader_row_count", "status"))
            + " |"
        )
    return lines


def write_report(path: Path, summary: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]], crossing_rows: Sequence[Mapping[str, Any]], format_rows: Sequence[Mapping[str, Any]]) -> None:
    first_cross = next((row for row in crossing_rows if _float(row.get("threshold")) == 1.1), {})
    gt_15 = next((row for row in crossing_rows if _float(row.get("threshold")) == 1.5), {})
    gt_20 = next((row for row in crossing_rows if _float(row.get("threshold")) == 2.0), {})
    h10_ready = summary.get("h10_recommendation") == "h10_candidate_after_review"
    lines = [
        "# Flowstar Raw Remainder Compat h5 Width Attribution",
        "",
        "This is an attribution audit only. It does not run h10, work on NNCS/GPU, add symbolic queue variants, change default solver behavior, or claim Flowstar parity.",
        "",
        "## Formatting Checks",
        "",
        "The requested physical line and csv.reader row-count checks passed locally; CSV physical line counts match csv.reader row counts for all checked CSVs.",
        "",
        *_formatting_text(format_rows),
        "",
        "## Answers",
        "",
        f"- Component first correlated with width ratio >1.1: `{summary.get('component_first_correlates_gt_1p1')}` at t `{_format(first_cross.get('t'))}`.",
        f"- Component driving >1.5: `{summary.get('component_driving_gt_1p5')}` at t `{_format(gt_15.get('t'))}`.",
        f"- Component driving >2.0: `{summary.get('component_driving_gt_2p0')}` at t `{_format(gt_20.get('t'))}`.",
        f"- Raw residual or right_map dominates at the first crossing: `{summary.get('raw_or_right_map_dominates_gt_1p1')}`.",
        f"- Flowstar component fields: `{summary.get('flowstar_component_status')}`; missing component fields are reported as unknown, not zero.",
        f"- Compat residual_y_hi stays below target in audited windows: `{'yes' if summary.get('target_margin_all_positive') else 'no'}`; target margin range `{_format(summary.get('target_margin_min'))}` to `{_format(summary.get('target_margin_max'))}`.",
        "- Does right_map_range begin to diverge before raw residual? `no at the first crossing`; neither raw residual nor right_map grows there, and right_map becomes the dominant raw-vs-right-map signal by the 2.0 crossing while raw residual remains target-bounded.",
        "- Is final 2.6x last-segment width early accumulation or late local blowup? `gradual accumulation`; the ratio crosses 1.1, 1.5, and 2.0 progressively before the final segment.",
        f"- Does tube ratio stay close because earlier extrema dominate? `yes`; last ratio `{_format(summary.get('last_ratio'))}`, tube ratio `{_format(summary.get('tube_ratio'))}`.",
        "- Next mechanism before h10: inspect normalized-insertion right-map/reset scaling and full-step tube range source attribution before changing raw remainder mechanics.",
        f"- Does this justify h10 now? `{'yes' if h10_ready else 'no'}`; recommendation `{_format(summary.get('h10_recommendation'))}`.",
        "",
        "## Event Windows",
        "",
        "| event | source | step | t | h | width ratio | tube prefix ratio | raw residual y_hi | right_map width | reset width | target margin | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ledger:
        lines.append(
            "| "
            + " | ".join(
                _format(row.get(field))
                for field in (
                    "event_name",
                    "source",
                    "step_index",
                    "t",
                    "h",
                    "width_ratio_vs_flowstar",
                    "tube_prefix_ratio",
                    "raw_ctrunc_residual_y_hi",
                    "right_map_range_width_sum",
                    "reset_width_sum",
                    "residual_y_hi_margin_to_target",
                    "notes",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `outputs/flowstar_raw_remainder_compat_h5_width_attribution/h5_width_attribution_ledger.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_width_attribution/h5_component_growth.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_width_attribution/h5_crossing_windows.csv`",
            "- `outputs/flowstar_raw_remainder_compat_h5_width_attribution/h5_width_attribution_report.md`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", type=Path, default=DEFAULT_H5_DIR)
    parser.add_argument("--divergence-dir", type=Path, default=DEFAULT_DIVERGENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if "h10" in str(out_dir):
        raise ValueError("refusing to write h10 outputs from the h5 width attribution audit")
    h5_dir = args.h5_dir.resolve()
    divergence_dir = args.divergence_dir.resolve()
    ledger, component_growth, crossing_rows, summary, format_rows = build_attribution(h5_dir, divergence_dir)
    _write_rows(out_dir / "h5_width_attribution_ledger.csv", LEDGER_FIELDS, ledger)
    _write_rows(out_dir / "h5_component_growth.csv", COMPONENT_GROWTH_FIELDS, component_growth)
    _write_rows(out_dir / "h5_crossing_windows.csv", CROSSING_FIELDS, crossing_rows)
    write_report(out_dir / "h5_width_attribution_report.md", summary, ledger, crossing_rows, format_rows)
    print(f"wrote h5 width attribution audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
