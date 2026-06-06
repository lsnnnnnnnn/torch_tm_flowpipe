#!/usr/bin/env python3
"""Compare PyTorch Flow*-style reset/segment width growth against original Flow* boxes."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLOWSTAR_SEGMENTS = REPO_ROOT / "outputs" / "flowstar_benchmark_parity" / "original_flowstar" / "original_flowstar_segments.csv"
DEFAULT_PYTORCH_DIR = REPO_ROOT / "outputs" / "flowstar_style_candidate_order"
DEFAULT_SOURCE_RUN = "flowstar_style_o6_candidate8_output6_cutoff"

TRACE_FIELDS = [
    "run_id",
    "segment_index",
    "t_lo",
    "t_hi",
    "h",
    "pytorch_final_width_x",
    "pytorch_final_width_y",
    "pytorch_final_width_sum",
    "pytorch_reset_width_x",
    "pytorch_reset_width_y",
    "pytorch_reset_width_sum",
    "flowstar_width_x",
    "flowstar_width_y",
    "flowstar_width_sum",
    "final_ratio_x",
    "final_ratio_y",
    "final_ratio_sum",
    "reset_ratio_x",
    "reset_ratio_y",
    "reset_ratio_sum",
    "residual_width_sum",
    "target_remainder_width_sum",
    "validation_attempts",
    "step_rejections",
    "candidate_order",
    "output_order",
    "crosses_2x",
    "crosses_5x",
    "crosses_10x",
    "crosses_20x",
    "crosses_50x",
]

RESET_FIELDS = [
    "run_id",
    "segment_index",
    "t_lo",
    "t_hi",
    "pytorch_reset_width_x",
    "pytorch_reset_width_y",
    "pytorch_reset_width_sum",
    "flowstar_width_x",
    "flowstar_width_y",
    "flowstar_width_sum",
    "reset_ratio_x",
    "reset_ratio_y",
    "reset_ratio_sum",
]

SUMMARY_FIELDS = [
    "run_id",
    "segments",
    "last_t_hi",
    "max_final_ratio_sum",
    "max_reset_ratio_sum",
    "first_final_gt_2x_t",
    "first_final_gt_5x_t",
    "first_final_gt_10x_t",
    "first_final_gt_20x_t",
    "first_final_gt_50x_t",
    "dominant_dimension_at_first_10x",
    "max_step_rejections_before_accepted",
    "failure_explainable_by_wide_reset_box",
    "most_likely_missing_mechanism",
]


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.17g}" if math.isfinite(value) else ""
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in fields})


def _float(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _width(row: Mapping[str, Any], dim: str) -> float | None:
    direct = _float(row, f"width_{dim}")
    if direct is not None:
        return direct
    lo = _float(row, f"{dim}_lo")
    hi = _float(row, f"{dim}_hi")
    if lo is None or hi is None:
        return None
    return hi - lo


def _width_sum(row: Mapping[str, Any]) -> float | None:
    direct = _float(row, "width_sum")
    if direct is not None:
        return direct
    x = _width(row, "x")
    y = _width(row, "y")
    if x is None or y is None:
        return None
    return x + y


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b <= 0:
        return None
    return a / b


def _hull_width(rows: Sequence[Mapping[str, Any]], dim: str) -> float | None:
    los = [_float(row, f"{dim}_lo") for row in rows]
    his = [_float(row, f"{dim}_hi") for row in rows]
    los_f = [v for v in los if v is not None]
    his_f = [v for v in his if v is not None]
    if not los_f or not his_f:
        widths = [_width(row, dim) for row in rows]
        widths_f = [v for v in widths if v is not None]
        return max(widths_f) if widths_f else None
    return max(his_f) - min(los_f)


def _flowstar_overlap(flow_rows: Sequence[Mapping[str, Any]], t_lo: float, t_hi: float) -> list[Mapping[str, Any]]:
    overlaps = []
    for row in flow_rows:
        lo = _float(row, "t_lo")
        hi = _float(row, "t_hi")
        if lo is None or hi is None:
            continue
        if hi >= t_lo - 1e-15 and lo <= t_hi + 1e-15:
            overlaps.append(row)
    if overlaps:
        return overlaps
    center = (t_lo + t_hi) / 2.0
    nearest = min(
        flow_rows,
        key=lambda row: abs(((_float(row, "t_lo") or 0.0) + (_float(row, "t_hi") or 0.0)) / 2.0 - center),
        default=None,
    )
    return [nearest] if nearest is not None else []


def _attempt_by_segment(rows: Sequence[Mapping[str, Any]], run_id: str) -> dict[int, Mapping[str, Any]]:
    latest: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("run_id") != run_id:
            continue
        try:
            idx = int(float(row.get("segment_index", "")))
        except (TypeError, ValueError):
            continue
        latest[idx] = row
    return latest


def _reset_by_segment(rows: Sequence[Mapping[str, Any]], run_id: str) -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("run_id") != run_id:
            continue
        try:
            idx = int(float(row.get("segment_index", "")))
        except (TypeError, ValueError):
            continue
        out[idx] = row
    return out


def _build_traces(
    *,
    flow_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    reset_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    attempts = _attempt_by_segment(attempt_rows, run_id)
    resets = _reset_by_segment(reset_rows, run_id)
    rows: list[dict[str, Any]] = []
    reset_trace: list[dict[str, Any]] = []
    for seg in segment_rows:
        if seg.get("run_id") != run_id or seg.get("status") != "validated":
            continue
        t_lo = _float(seg, "t_lo")
        t_hi = _float(seg, "t_hi")
        if t_lo is None or t_hi is None:
            continue
        try:
            segment_index = int(float(seg.get("segment_index", "")))
        except (TypeError, ValueError):
            continue
        flow = _flowstar_overlap(flow_rows, t_lo, t_hi)
        fs_x = _hull_width(flow, "x")
        fs_y = _hull_width(flow, "y")
        fs_sum = (fs_x + fs_y) if fs_x is not None and fs_y is not None else None
        reset = resets.get(segment_index, {})
        attempt = attempts.get(segment_index, {})
        final_x = _width(seg, "x")
        final_y = _width(seg, "y")
        final_sum = _width_sum(seg)
        reset_x = _width(reset, "x") if reset else final_x
        reset_y = _width(reset, "y") if reset else final_y
        reset_sum = _width_sum(reset) if reset else final_sum
        final_ratio_x = _ratio(final_x, fs_x)
        final_ratio_y = _ratio(final_y, fs_y)
        final_ratio_sum = _ratio(final_sum, fs_sum)
        reset_ratio_x = _ratio(reset_x, fs_x)
        reset_ratio_y = _ratio(reset_y, fs_y)
        reset_ratio_sum = _ratio(reset_sum, fs_sum)
        row = {
            "run_id": run_id,
            "segment_index": segment_index,
            "t_lo": t_lo,
            "t_hi": t_hi,
            "h": _float(seg, "h"),
            "pytorch_final_width_x": final_x,
            "pytorch_final_width_y": final_y,
            "pytorch_final_width_sum": final_sum,
            "pytorch_reset_width_x": reset_x,
            "pytorch_reset_width_y": reset_y,
            "pytorch_reset_width_sum": reset_sum,
            "flowstar_width_x": fs_x,
            "flowstar_width_y": fs_y,
            "flowstar_width_sum": fs_sum,
            "final_ratio_x": final_ratio_x,
            "final_ratio_y": final_ratio_y,
            "final_ratio_sum": final_ratio_sum,
            "reset_ratio_x": reset_ratio_x,
            "reset_ratio_y": reset_ratio_y,
            "reset_ratio_sum": reset_ratio_sum,
            "residual_width_sum": _float(attempt, "residual_width_sum"),
            "target_remainder_width_sum": _float(attempt, "target_remainder_width_sum"),
            "validation_attempts": seg.get("validation_attempts", ""),
            "step_rejections": seg.get("step_rejections", ""),
            "candidate_order": seg.get("candidate_order", attempt.get("candidate_order", "")),
            "output_order": seg.get("output_order", attempt.get("output_order", "")),
            "crosses_2x": bool(final_ratio_sum is not None and final_ratio_sum > 2.0),
            "crosses_5x": bool(final_ratio_sum is not None and final_ratio_sum > 5.0),
            "crosses_10x": bool(final_ratio_sum is not None and final_ratio_sum > 10.0),
            "crosses_20x": bool(final_ratio_sum is not None and final_ratio_sum > 20.0),
            "crosses_50x": bool(final_ratio_sum is not None and final_ratio_sum > 50.0),
        }
        rows.append(row)
        reset_trace.append({field: row.get(field, "") for field in RESET_FIELDS})

    def first_t(threshold: float) -> float | str:
        for row in rows:
            ratio = row.get("final_ratio_sum")
            if isinstance(ratio, float) and ratio > threshold:
                return row["t_hi"]
        return ""

    first_10 = next((row for row in rows if isinstance(row.get("final_ratio_sum"), float) and row["final_ratio_sum"] > 10.0), None)
    dominant = ""
    if first_10:
        rx = first_10.get("final_ratio_x")
        ry = first_10.get("final_ratio_y")
        if isinstance(rx, float) and isinstance(ry, float):
            dominant = "x" if rx >= ry else "y"
    max_final = max((row["final_ratio_sum"] for row in rows if isinstance(row.get("final_ratio_sum"), float)), default=0.0)
    max_reset = max((row["reset_ratio_sum"] for row in rows if isinstance(row.get("reset_ratio_sum"), float)), default=0.0)
    last_reset = rows[-1].get("reset_ratio_sum") if rows else None
    summary = {
        "run_id": run_id,
        "segments": len(rows),
        "last_t_hi": rows[-1]["t_hi"] if rows else "",
        "max_final_ratio_sum": max_final,
        "max_reset_ratio_sum": max_reset,
        "first_final_gt_2x_t": first_t(2.0),
        "first_final_gt_5x_t": first_t(5.0),
        "first_final_gt_10x_t": first_t(10.0),
        "first_final_gt_20x_t": first_t(20.0),
        "first_final_gt_50x_t": first_t(50.0),
        "dominant_dimension_at_first_10x": dominant,
        "max_step_rejections_before_accepted": max((_float(row, "step_rejections") or 0.0 for row in rows), default=0.0),
        "failure_explainable_by_wide_reset_box": bool(isinstance(last_reset, float) and last_reset > 10.0),
        "most_likely_missing_mechanism": "Flow*-style symbolic remainder queue plus normalized insertion/composition",
    }
    return rows, reset_trace, summary


def _make_plots(out_dir: Path, trace_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    pts = [(float(row["t_hi"]), float(row["final_ratio_sum"])) for row in trace_rows if row.get("final_ratio_sum") not in {"", None}]
    if pts:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.plot([t for t, _ in pts], [v for _, v in pts], marker="o", markersize=2.2, linewidth=1.0)
        for threshold in [2, 5, 10, 20, 50]:
            ax.axhline(threshold, color="#666666", linewidth=0.7, linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("PyTorch / Flow* final width sum")
        fig.tight_layout()
        fig.savefig(out_dir / "width_ratio_vs_t.png", dpi=160)
        plt.close(fig)
    reset_pts = [(float(row["t_hi"]), float(row["reset_ratio_sum"])) for row in trace_rows if row.get("reset_ratio_sum") not in {"", None}]
    if reset_pts:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.plot([t for t, _ in reset_pts], [v for _, v in reset_pts], marker="o", markersize=2.2, linewidth=1.0)
        for threshold in [2, 5, 10, 20, 50]:
            ax.axhline(threshold, color="#666666", linewidth=0.7, linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("PyTorch / Flow* reset width sum")
        fig.tight_layout()
        fig.savefig(out_dir / "reset_box_ratio_vs_t.png", dpi=160)
        plt.close(fig)


def _write_report(out_dir: Path, summary: Mapping[str, Any], trace_rows: Sequence[Mapping[str, Any]]) -> None:
    responsible = summary.get("dominant_dimension_at_first_10x") or "not determined"
    rejected_rows = [row for row in trace_rows if (_float(row, "step_rejections") or 0.0) > 0]
    rejection_msg = "yes" if rejected_rows else "no accepted segment recorded prior rejections"
    lines = [
        "# Flowstar Width-Growth Diagnostics",
        "",
        f"Source run: `{summary.get('run_id', '')}`.",
        f"Accepted PyTorch segments compared: `{summary.get('segments', '')}` through t=`{summary.get('last_t_hi', '')}`.",
        f"First >2x final width ratio: `{summary.get('first_final_gt_2x_t', '')}`.",
        f"First >5x final width ratio: `{summary.get('first_final_gt_5x_t', '')}`.",
        f"First >10x final width ratio: `{summary.get('first_final_gt_10x_t', '')}`.",
        f"First >20x final width ratio: `{summary.get('first_final_gt_20x_t', '')}`.",
        f"First >50x final width ratio: `{summary.get('first_final_gt_50x_t', '')}`.",
        f"Dominant dimension at first >10x crossing: `{responsible}`.",
        f"Do width jumps occur after step rejections? {rejection_msg}.",
        f"Is the local oracle failure explainable by already-wide reset boxes? {_fmt(summary.get('failure_explainable_by_wide_reset_box'))}.",
        f"Most likely missing Flow* mechanism: {summary.get('most_likely_missing_mechanism', '')}.",
        "",
        "## Threshold Crossings",
        "",
        "| threshold | first t |",
        "| ---: | ---: |",
        f"| 2x | {summary.get('first_final_gt_2x_t', '')} |",
        f"| 5x | {summary.get('first_final_gt_5x_t', '')} |",
        f"| 10x | {summary.get('first_final_gt_10x_t', '')} |",
        f"| 20x | {summary.get('first_final_gt_20x_t', '')} |",
        f"| 50x | {summary.get('first_final_gt_50x_t', '')} |",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "width_growth_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "flowstar_width_growth_diagnostics")
    parser.add_argument("--flowstar-segments", type=Path, default=DEFAULT_FLOWSTAR_SEGMENTS)
    parser.add_argument("--pytorch-dir", type=Path, default=DEFAULT_PYTORCH_DIR)
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    args = parser.parse_args(argv)

    flow_rows = _read_csv(args.flowstar_segments)
    segment_rows = _read_csv(args.pytorch_dir / "rescue_segments.csv")
    reset_rows = _read_csv(args.pytorch_dir / "rescue_reset_boxes.csv")
    attempt_rows = _read_csv(args.pytorch_dir / "rescue_validation_attempts.csv")
    if not flow_rows:
        raise FileNotFoundError(f"no Flow* rows found at {args.flowstar_segments}")
    if not segment_rows:
        raise FileNotFoundError(f"no PyTorch segment rows found under {args.pytorch_dir}")

    trace_rows, reset_trace, summary = _build_traces(
        flow_rows=flow_rows,
        segment_rows=segment_rows,
        reset_rows=reset_rows,
        attempt_rows=attempt_rows,
        run_id=str(args.source_run),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "width_growth_trace.csv", TRACE_FIELDS, trace_rows)
    _write_csv(args.out_dir / "reset_box_vs_flowstar_trace.csv", RESET_FIELDS, reset_trace)
    _write_csv(args.out_dir / "width_growth_summary.csv", SUMMARY_FIELDS, [summary])
    _write_report(args.out_dir, summary, trace_rows)
    _make_plots(args.out_dir, trace_rows)
    print(f"wrote width diagnostics to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
