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
    (out_dir / "right_map_scaling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
