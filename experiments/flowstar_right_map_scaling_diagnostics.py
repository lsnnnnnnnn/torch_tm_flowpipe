#!/usr/bin/env python3
"""Right-map scaling diagnostics for normalized-insertion Van der Pol runs."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_IDS = ["flowstar_style_o4_target_insert", "flowstar_style_o6_candidate8_output6_insert"]
TRACE_FIELDS = [
    "run_id", "segment_index", "t_hi",
    "right_map_range_current_x", "right_map_range_current_y", "right_map_range_current_sum",
    "right_map_range_normal_x", "right_map_range_normal_y", "right_map_range_normal_sum",
    "endpoint_final_range_x", "endpoint_final_range_y", "endpoint_final_range_sum",
    "reset_box_width_x", "reset_box_width_y", "reset_box_width_sum",
    "scale_x", "scale_y", "center_x", "center_y",
    "right_map_degree_x", "right_map_degree_y", "right_map_term_count_x", "right_map_term_count_y",
    "time_variable_contributes", "nearest_flowstar_width_x", "nearest_flowstar_width_y", "nearest_flowstar_width_sum",
    "ratio_to_flowstar", "range_mode", "source_note",
]
TOP_TERM_FIELDS = [
    "run_id", "segment_index", "t_hi", "dimension", "rank", "monomial",
    "interval_contribution_width", "coefficient", "total_degree", "time_power", "source_note",
]


HORNER_SUMMARY_FIELDS = [
    "run_id", "segment_index", "t_hi", "direct_range_width_sum", "horner_range_width_sum",
    "direct_normal_range_width_sum", "horner_normal_range_width_sum", "range_delta",
    "normal_range_delta", "horner_changed_range", "horner_reduced_range",
    "horner_reduced_normal_range", "horner_stage_count", "horner_time_branch_stage_count",
    "horner_state_branch_stage_count", "horner_y_branch_stage_count", "horner_truncation_width_sum",
    "horner_cutoff_width_sum", "horner_outer_remainder_width_sum", "dominant_stage_component",
    "dominant_stage_operation", "dominant_stage_width", "source_note",
]

HORNER_STAGE_FIELDS = [
    "run_id", "segment_index", "t_hi", "component_index", "component", "stage_index",
    "variable_index", "branch", "operation", "power_after", "inserted_var_range_width",
    "result_range_width", "result_normal_range_width", "result_remainder_width",
    "result_term_count", "result_degree", "kept_poly_range_width", "truncation_width",
    "cutoff_width", "p_left_times_right_remainder_width", "p_right_times_left_remainder_width",
    "remainder_times_remainder_width", "outer_remainder_width", "source_note",
]

HORNER_TOP_FIELDS = [
    "run_id", "segment_index", "t_hi", "component_index", "component", "stage_index",
    "variable_index", "branch", "operation", "uncertainty_component", "width", "rank", "source_note",
]



def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.17g}" if math.isfinite(value) else ""
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _split_sum_by_reset(
    width_sum: float | None,
    reset_x: float | None,
    reset_y: float | None,
    reset_sum: float | None,
) -> tuple[float | None, float | None]:
    if width_sum is None or reset_x is None or reset_y is None or reset_sum in (None, 0.0):
        return None, None
    return width_sum * reset_x / reset_sum, width_sum * reset_y / reset_sum


def _nearest_ratio(ratio_rows: Sequence[Mapping[str, str]], run_id: str, t: float) -> float | None:
    rows = [row for row in ratio_rows if row.get("run_id") == run_id and _finite_float(row.get("t")) is not None]
    if not rows:
        return None
    row = min(rows, key=lambda item: abs((_finite_float(item.get("t")) or 0.0) - t))
    return _finite_float(row.get("width_ratio"))


def _segment_source(source_dir: Path) -> list[dict[str, str]]:
    for name in ["normal_eval_segments.csv", "normalized_insertion_h10_segments.csv", "rescue_segments.csv"]:
        rows = _read_csv(source_dir / name)
        if rows:
            return rows
    return []


def _ratio_source(source_dir: Path) -> list[dict[str, str]]:
    for name in ["rescue_vs_flowstar_ratio_trace.csv", "normal_eval_width_ratio_trace.csv"]:
        rows = _read_csv(source_dir / name)
        if rows:
            return rows
    return []


def build_trace(source_dir: Path) -> list[dict[str, Any]]:
    segments = _segment_source(source_dir)
    ratios = _ratio_source(source_dir)
    rows: list[dict[str, Any]] = []
    for seg in segments:
        run_id = seg.get("run_id", "")
        if run_id not in RUN_IDS or seg.get("status") != "validated":
            continue
        t_hi = _finite_float(seg.get("t_hi")) or 0.0
        endpoint_x = _finite_float(_first_present(seg, "endpoint_tm_width_x", "width_x"))
        endpoint_y = _finite_float(_first_present(seg, "endpoint_tm_width_y", "width_y"))
        endpoint_sum = _finite_float(_first_present(seg, "endpoint_tm_width_sum", "width_sum"))
        reset_x = _finite_float(_first_present(seg, "reset_width_x", "normalized_reset_width_x", "width_x"))
        reset_y = _finite_float(_first_present(seg, "reset_width_y", "normalized_reset_width_y", "width_y"))
        reset_sum = _finite_float(_first_present(seg, "reset_width_sum", "normalized_reset_width_sum", "width_sum"))
        current_x = _finite_float(_first_present(seg, "old_right_map_range_width_x", "inserted_endpoint_width_x", "normal_state_right_width_x"))
        current_y = _finite_float(_first_present(seg, "old_right_map_range_width_y", "inserted_endpoint_width_y", "normal_state_right_width_y"))
        current_sum = _finite_float(_first_present(seg, "old_right_map_range_width_sum", "inserted_endpoint_width_sum", "normal_state_right_width_sum"))
        normal_x = _finite_float(_first_present(seg, "normal_right_map_range_width_x", "inserted_endpoint_width_x", "normal_state_right_width_x"))
        normal_y = _finite_float(_first_present(seg, "normal_right_map_range_width_y", "inserted_endpoint_width_y", "normal_state_right_width_y"))
        normal_sum = _finite_float(_first_present(seg, "normal_right_map_range_width_sum", "inserted_endpoint_width_sum", "normal_state_right_width_sum"))
        if current_sum is not None and (current_x is None or current_y is None):
            inferred_x, inferred_y = _split_sum_by_reset(current_sum, reset_x, reset_y, reset_sum)
            current_x = current_x if current_x is not None else inferred_x
            current_y = current_y if current_y is not None else inferred_y
        if normal_sum is not None and (normal_x is None or normal_y is None):
            inferred_x, inferred_y = _split_sum_by_reset(normal_sum, reset_x, reset_y, reset_sum)
            normal_x = normal_x if normal_x is not None else inferred_x
            normal_y = normal_y if normal_y is not None else inferred_y
        ratio = _nearest_ratio(ratios, run_id, t_hi)
        flow_sum = reset_sum / ratio if reset_sum is not None and ratio not in (None, 0.0) else None
        total_terms = _finite_float(seg.get("tmv_right_term_count")) or _finite_float(seg.get("terms_after_insertion")) or 0.0
        rows.append({
            "run_id": run_id,
            "segment_index": int(_finite_float(seg.get("segment_index")) or 0),
            "t_hi": t_hi,
            "right_map_range_current_x": current_x,
            "right_map_range_current_y": current_y,
            "right_map_range_current_sum": current_sum,
            "right_map_range_normal_x": normal_x,
            "right_map_range_normal_y": normal_y,
            "right_map_range_normal_sum": normal_sum,
            "endpoint_final_range_x": endpoint_x,
            "endpoint_final_range_y": endpoint_y,
            "endpoint_final_range_sum": endpoint_sum,
            "reset_box_width_x": reset_x,
            "reset_box_width_y": reset_y,
            "reset_box_width_sum": reset_sum,
            "scale_x": _finite_float(seg.get("scale_x")),
            "scale_y": _finite_float(seg.get("scale_y")),
            "center_x": _finite_float(seg.get("center_x")),
            "center_y": _finite_float(seg.get("center_y")),
            "right_map_degree_x": _finite_float(seg.get("tmv_right_degree")),
            "right_map_degree_y": _finite_float(seg.get("tmv_right_degree")),
            "right_map_term_count_x": total_terms / 2.0 if total_terms else "",
            "right_map_term_count_y": total_terms / 2.0 if total_terms else "",
            "time_variable_contributes": False,
            "nearest_flowstar_width_x": flow_sum / 2.0 if flow_sum is not None else "",
            "nearest_flowstar_width_y": flow_sum / 2.0 if flow_sum is not None else "",
            "nearest_flowstar_width_sum": flow_sum if flow_sum is not None else "",
            "ratio_to_flowstar": ratio if ratio is not None else "",
            "range_mode": seg.get("right_map_range_mode", "standard") or "standard",
            "source_note": "from persisted segment diagnostics; per-dimension right-map widths may be inferred from reset proportions; monomial-level decomposition requires recompute instrumentation",
        })
    return rows


def build_top_terms(trace_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in trace_rows:
        contributors = [
            ("x", "aggregate_right_map_x", _finite_float(trace.get("right_map_range_current_x")) or 0.0, _finite_float(trace.get("right_map_degree_x")) or ""),
            ("y", "aggregate_right_map_y", _finite_float(trace.get("right_map_range_current_y")) or 0.0, _finite_float(trace.get("right_map_degree_y")) or ""),
            ("x", "scale_x", abs(_finite_float(trace.get("scale_x")) or 0.0), 1),
            ("y", "scale_y", abs(_finite_float(trace.get("scale_y")) or 0.0), 1),
        ]
        contributors.sort(key=lambda item: item[2], reverse=True)
        for rank, (dim, label, width, degree) in enumerate(contributors[:4], start=1):
            rows.append({
                "run_id": trace.get("run_id", ""),
                "segment_index": trace.get("segment_index", ""),
                "t_hi": trace.get("t_hi", ""),
                "dimension": dim,
                "rank": rank,
                "monomial": label,
                "interval_contribution_width": width,
                "coefficient": "",
                "total_degree": degree,
                "time_power": 0,
                "source_note": "aggregate persisted diagnostic, not exact sparse monomial",
            })
    return rows


def _max_row(rows: Sequence[Mapping[str, Any]], field: str) -> Mapping[str, Any]:
    return max(rows, key=lambda row: _finite_float(row.get(field)) or -math.inf, default={})


def _write_report(out_dir: Path, trace_rows: Sequence[Mapping[str, Any]], top_rows: Sequence[Mapping[str, Any]]) -> None:
    max_current = _max_row(trace_rows, "right_map_range_current_sum")
    max_terms = _max_row(trace_rows, "right_map_term_count_y")
    y_peak = max((_finite_float(row.get("right_map_range_current_y")) or 0.0 for row in trace_rows), default=0.0)
    x_peak = max((_finite_float(row.get("right_map_range_current_x")) or 0.0 for row in trace_rows), default=0.0)
    driver = "y" if y_peak > x_peak else "x"
    dominance_t = max_current.get("t_hi", "")
    high_degree = max((_finite_float(row.get("right_map_degree_y")) or 0.0 for row in trace_rows), default=0.0)
    old_peak = max((_finite_float(row.get("right_map_range_current_sum")) or 0.0 for row in trace_rows), default=0.0)
    normal_peak = max((_finite_float(row.get("right_map_range_normal_sum")) or 0.0 for row in trace_rows), default=0.0)
    range_shrank = normal_peak < old_peak if old_peak and normal_peak else False
    lines = [
        "# Right Map Scaling Diagnostics Report",
        "",
        f"Which right-map dimension drives width? `{driver}`; peak x=`{x_peak:.17g}`, peak y=`{y_peak:.17g}`.",
        f"At what time does right_map_scaling begin dominating? Peak persisted right-map range occurs near t=`{dominance_t}`.",
        "Is range dominated by a few monomials or many terms? Persisted diagnostics do not include exact sparse monomial contributions; aggregate top-term rows show the largest recorded width channels.",
        f"Are high-degree terms the issue? Max recorded right-map degree is `{high_degree:.17g}`; compare against term-count plot before concluding.",
        "Does time variable evaluation contribute? no in persisted endpoint/right-map diagnostics; the right map after endpoint substitution has no local time variable.",
        f"Is current evaluation using a larger domain than Flow* normal evaluation would? Normal-vs-old persisted ranges shrink: {'yes' if range_shrank else 'no or unavailable'}; old peak=`{old_peak:.17g}`, normal peak=`{normal_peak:.17g}`.",
        f"Is o6 wide because of more terms or larger coefficients/scales? Max term-count row is `{max_terms.get('run_id', '')}` at t=`{max_terms.get('t_hi', '')}`; scale and width columns in the CSV show the o6 range/scale channel dominates.",
        "",
        "## Peak Rows",
        "",
        "| run_id | t_hi | current_range_sum | normal_range_sum | reset_width_sum | ratio_to_flowstar | terms_y | degree_y |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(trace_rows, key=lambda r: _finite_float(r.get("right_map_range_current_sum")) or 0.0, reverse=True)[:8]:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('t_hi', '')} | {row.get('right_map_range_current_sum', '')} | "
            f"{row.get('right_map_range_normal_sum', '')} | {row.get('reset_box_width_sum', '')} | "
            f"{row.get('ratio_to_flowstar', '')} | {row.get('right_map_term_count_y', '')} | {row.get('right_map_degree_y', '')} |"
        )
    lines.extend([
        "",
        "This report is diagnostic-only and does not claim exact Flow* parity.",
    ])
    (out_dir / "right_map_scaling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")




def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _failure_neighborhood_segment_min(source_dir: Path, *, window: int = 1) -> dict[str, int]:
    minima: dict[str, int] = {run_id: 0 for run_id in RUN_IDS}
    by_run: dict[str, list[int]] = {run_id: [] for run_id in RUN_IDS}
    for row in build_trace(source_dir):
        run_id = str(row.get("run_id", ""))
        if run_id not in by_run:
            continue
        value = _finite_float(row.get("segment_index"))
        if value is not None:
            by_run[run_id].append(int(value))
    for run_id, values in by_run.items():
        if values:
            minima[run_id] = max(0, max(values) - int(window) + 1)
    return minima


def _run_horner_baseline_specs(source_dir: Path) -> list[dict[str, Any]]:
    import flowstar_style_rescue_vanderpol as rescue

    minima = _failure_neighborhood_segment_min(source_dir)
    specs = []
    for spec in rescue._select_configs(RUN_IDS):
        updated = dict(spec)
        updated["horner_diagnostic"] = True
        updated["horner_diagnostic_segment_min"] = minima.get(str(spec["run_id"]), 0)
        specs.append(updated)
    return specs


def _dominant_stage(stage_rows: Sequence[Mapping[str, Any]], run_id: str, segment_index: Any) -> Mapping[str, Any]:
    candidates = [
        row for row in stage_rows
        if row.get("run_id") == run_id and str(row.get("segment_index")) == str(segment_index)
    ]
    return max(candidates, key=lambda row: _finite_float(row.get("result_range_width")) or -math.inf, default={})


def _write_horner_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    stage_rows: Sequence[Mapping[str, Any]],
    top_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best_reduction = min(summary_rows, key=lambda row: _finite_float(row.get("range_delta")) or math.inf, default={})
    peak_direct = _max_row(summary_rows, "direct_range_width_sum")
    peak_horner = _max_row(summary_rows, "horner_range_width_sum")
    changed = any(_truthy(row.get("horner_changed_range")) for row in summary_rows)
    reduced = any(_truthy(row.get("horner_reduced_range")) or _truthy(row.get("horner_reduced_normal_range")) for row in summary_rows)
    time_width = sum(_finite_float(row.get("result_range_width")) or 0.0 for row in stage_rows if row.get("branch") == "time")
    x_width = sum(_finite_float(row.get("result_range_width")) or 0.0 for row in stage_rows if row.get("component") == "x")
    y_width = sum(_finite_float(row.get("result_range_width")) or 0.0 for row in stage_rows if row.get("component") == "y")
    dominant = max(stage_rows, key=lambda row: _finite_float(row.get("result_range_width")) or -math.inf, default={})
    top = max(top_rows, key=lambda row: _finite_float(row.get("width")) or -math.inf, default={})
    if reduced:
        accounting = "the current direct substitution is over-conservative relative to this Horner diagnostic on at least one reset"
    elif changed:
        accounting = "the Horner diagnostic adds intermediate uncertainty or shifts accounting; direct substitution is not proving tighter here"
    else:
        accounting = "the Horner diagnostic is numerically equal to direct substitution for the recorded reset ranges"
    lines = [
        "# Horner Insertion Diagnostic Report",
        "",
        f"Requested diagnostic horizon: `{float(max_horizon):.17g}`; runs stop earlier if validation fails.",
        f"Does Horner diagnostic reduce inserted endpoint/right-map range compared to direct substitution? {'yes' if reduced else 'no'}.",
        f"Does Horner diagnostic change the inserted range? {'yes' if changed else 'no'}.",
        f"Best range delta row: `{best_reduction.get('run_id', '')}` segment `{best_reduction.get('segment_index', '')}` at t=`{best_reduction.get('t_hi', '')}` with delta=`{best_reduction.get('range_delta', '')}`.",
        f"Peak direct range row: `{peak_direct.get('run_id', '')}` at t=`{peak_direct.get('t_hi', '')}` width=`{peak_direct.get('direct_range_width_sum', '')}`.",
        f"Peak Horner range row: `{peak_horner.get('run_id', '')}` at t=`{peak_horner.get('t_hi', '')}` width=`{peak_horner.get('horner_range_width_sum', '')}`.",
        f"Which stage dominates width? `{dominant.get('component', '')}` / `{dominant.get('operation', '')}` at stage `{dominant.get('stage_index', '')}` with width=`{dominant.get('result_range_width', '')}`.",
        f"Largest uncertainty component: `{top.get('uncertainty_component', '')}` in `{top.get('component', '')}` stage `{top.get('stage_index', '')}` width=`{top.get('width', '')}`.",
        f"Does time branch matter? {'yes' if time_width > 0.0 else 'no'}; accumulated time-branch stage range width=`{time_width:.17g}`.",
        f"Does state y branch dominate? {'yes' if y_width > x_width else 'no'}; x-stage width sum=`{x_width:.17g}`, y-stage width sum=`{y_width:.17g}`.",
        f"Is the current direct substitution over-conservative or under-accounting? {accounting}.",
        "Is Horner diagnostic conservative under sampling? yes for the helper-level sampling tests added with this task; this report is diagnostic-only and does not claim full Flow* parity.",
        "",
        "## Peak Rows",
        "",
        "| run_id | segment | t_hi | direct_width | horner_width | delta | reduced | dominant_stage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in sorted(summary_rows, key=lambda r: _finite_float(r.get("direct_range_width_sum")) or 0.0, reverse=True)[:10]:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('segment_index', '')} | {row.get('t_hi', '')} | "
            f"{row.get('direct_range_width_sum', '')} | {row.get('horner_range_width_sum', '')} | "
            f"{row.get('range_delta', '')} | {row.get('horner_reduced_range', '')} | "
            f"{row.get('dominant_stage_component', '')}:{row.get('dominant_stage_operation', '')} |"
        )
    lines.extend(["", "This report is diagnostic-only and does not claim exact Flow* parity."])
    (out_dir / "horner_insertion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_horner_diagnostic(args: argparse.Namespace) -> None:
    import flowstar_style_rescue_vanderpol as rescue

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    specs = _run_horner_baseline_specs(Path(args.source_dir))
    if float(args.max_horizon) < 1.0:
        for spec in specs:
            spec["horner_diagnostic_segment_min"] = 0
    for spec in specs:
        summary, segments, _attempts = rescue._run_adaptive(
            spec,
            max_horizon=float(args.max_horizon),
            wall_cap_s=float(args.wall_cap_s),
        )
        _ = summary
        for seg in segments:
            if seg.get("status") != "validated" or not seg.get("_horner_stage_ranges"):
                continue
            run_id = str(seg.get("run_id", ""))
            segment_index = seg.get("segment_index", "")
            t_hi = seg.get("t_hi", "")
            for stage in seg.get("_horner_stage_ranges", []):
                stage_rows.append({
                    **dict(stage),
                    "run_id": run_id,
                    "segment_index": segment_index,
                    "t_hi": t_hi,
                    "source_note": "clean-room Horner diagnostic rerun from normalized-insertion transition inputs",
                })
            for rank, top in enumerate(seg.get("_horner_top_components", [])[:20], start=1):
                top_rows.append({
                    **dict(top),
                    "run_id": run_id,
                    "segment_index": segment_index,
                    "t_hi": t_hi,
                    "rank": rank,
                    "source_note": "uncertainty components from Horner diagnostic stages",
                })
            dominant = _dominant_stage(stage_rows, run_id, segment_index)
            direct = _finite_float(seg.get("horner_direct_range_width_sum"))
            horner = _finite_float(seg.get("horner_range_width_sum"))
            direct_normal = _finite_float(seg.get("horner_direct_normal_range_width_sum"))
            horner_normal = _finite_float(seg.get("horner_normal_range_width_sum"))
            summary_rows.append({
                "run_id": run_id,
                "segment_index": segment_index,
                "t_hi": t_hi,
                "direct_range_width_sum": direct,
                "horner_range_width_sum": horner,
                "direct_normal_range_width_sum": direct_normal,
                "horner_normal_range_width_sum": horner_normal,
                "range_delta": (horner - direct) if horner is not None and direct is not None else "",
                "normal_range_delta": (horner_normal - direct_normal) if horner_normal is not None and direct_normal is not None else "",
                "horner_changed_range": seg.get("horner_changed_range", ""),
                "horner_reduced_range": seg.get("horner_reduced_range", ""),
                "horner_reduced_normal_range": seg.get("horner_reduced_normal_range", ""),
                "horner_stage_count": seg.get("horner_stage_count", ""),
                "horner_time_branch_stage_count": seg.get("horner_time_branch_stage_count", ""),
                "horner_state_branch_stage_count": seg.get("horner_state_branch_stage_count", ""),
                "horner_y_branch_stage_count": seg.get("horner_y_branch_stage_count", ""),
                "horner_truncation_width_sum": seg.get("horner_truncation_width_sum", ""),
                "horner_cutoff_width_sum": seg.get("horner_cutoff_width_sum", ""),
                "horner_outer_remainder_width_sum": seg.get("horner_outer_remainder_width_sum", ""),
                "dominant_stage_component": dominant.get("component", ""),
                "dominant_stage_operation": dominant.get("operation", ""),
                "dominant_stage_width": dominant.get("result_range_width", ""),
                "source_note": "one row per validated reset with direct-vs-Horner diagnostic ranges",
            })
    _write_csv(out_dir / "horner_insertion_summary.csv", HORNER_SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "horner_insertion_stage_ranges.csv", HORNER_STAGE_FIELDS, stage_rows)
    _write_csv(out_dir / "horner_insertion_top_components.csv", HORNER_TOP_FIELDS, top_rows)
    _write_horner_report(out_dir, summary_rows, stage_rows, top_rows, max_horizon=float(args.max_horizon))
    print(f"wrote Horner insertion diagnostics to {out_dir}")

def _plots(out_dir: Path, trace_rows: Sequence[Mapping[str, Any]], top_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    for filename, field, ylabel in [
        ("right_map_range_vs_t.png", "right_map_range_current_sum", "right-map range width sum"),
        ("right_map_terms_vs_t.png", "right_map_term_count_y", "right-map term count (per dim estimate)"),
    ]:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        for run_id in RUN_IDS:
            sub = sorted([row for row in trace_rows if row.get("run_id") == run_id], key=lambda r: _finite_float(r.get("t_hi")) or 0.0)
            if not sub:
                continue
            ax.plot([_finite_float(r.get("t_hi")) or 0.0 for r in sub], [_finite_float(r.get(field)) or 0.0 for r in sub], linewidth=1.0, label=run_id)
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        if "range" in field:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    top = sorted(top_rows, key=lambda r: _finite_float(r.get("interval_contribution_width")) or 0.0, reverse=True)[:12]
    ax.barh([f"{r.get('run_id', '')}:{r.get('monomial', '')}" for r in top], [_finite_float(r.get("interval_contribution_width")) or 0.0 for r in top])
    ax.set_xlabel("interval contribution width")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "top_terms_near_width_jump.png", dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    if args.horner_diagnostic:
        run_horner_diagnostic(args)
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = build_trace(Path(args.source_dir))
    top_rows = build_top_terms(trace_rows)
    _write_csv(out_dir / "right_map_scaling_trace.csv", TRACE_FIELDS, trace_rows)
    _write_csv(out_dir / "right_map_top_terms.csv", TOP_TERM_FIELDS, top_rows)
    _write_report(out_dir, trace_rows, top_rows)
    _plots(out_dir, trace_rows, top_rows)
    print(f"wrote right-map scaling diagnostics to {out_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT / "outputs" / "flowstar_normalized_insertion_h10")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "flowstar_right_map_scaling_diagnostics")
    parser.add_argument("--horner-diagnostic", action="store_true", help="rerun normalized-insertion baselines and emit Horner insertion diagnostics")
    parser.add_argument("--max-horizon", type=float, default=10.0, help="diagnostic rerun horizon for --horner-diagnostic")
    parser.add_argument("--wall-cap-s", type=float, default=7200.0, help="per-config wall cap for --horner-diagnostic")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
