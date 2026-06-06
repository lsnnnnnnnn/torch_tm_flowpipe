#!/usr/bin/env python3
'''Attribute normalized-insertion width growth by component.'''
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_IDS = ["flowstar_style_o4_target_insert", "flowstar_style_o6_candidate8_output6_insert"]
TRACE_FIELDS = [
    "run_id", "segment_index", "t_hi", "reset_box_width_x", "reset_box_width_y",
    "reset_box_width_sum", "endpoint_tm_width_x", "endpoint_tm_width_y", "endpoint_tm_width_sum",
    "inserted_endpoint_width_x", "inserted_endpoint_width_y", "inserted_endpoint_width_sum",
    "insertion_truncation_width_x", "insertion_truncation_width_y", "insertion_truncation_width_sum",
    "insertion_cutoff_width_x", "insertion_cutoff_width_y", "insertion_cutoff_width_sum",
    "picard_target_residual_width_x", "picard_target_residual_width_y", "picard_target_residual_width_sum",
    "ordinary_remainder_range_width_x", "ordinary_remainder_range_width_y", "ordinary_remainder_range_width_sum",
    "normalized_right_map_range_width_x", "normalized_right_map_range_width_y", "normalized_right_map_range_width_sum",
    "output_range_width_x", "output_range_width_y", "output_range_width_sum", "scale_x", "scale_y",
    "terms_before_insertion_truncation", "terms_after_insertion", "dominant_component",
    "dominant_dimension", "nearest_flowstar_width_ratio",
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


def _safe_sum(*values: Any) -> float | str:
    vals = [_finite_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else ""


def _attempt_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row.get("run_id", "")), int(_finite_float(row.get("segment_index")) or 0)


def _accepted_attempts(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int], Mapping[str, str]]:
    out: dict[tuple[str, int], Mapping[str, str]] = {}
    for row in rows:
        if row.get("validation_status") != "validated":
            continue
        out[_attempt_key(row)] = row
    return out


def _nearest_ratio(ratio_rows: Sequence[Mapping[str, str]], run_id: str, t: float) -> str:
    rows = [row for row in ratio_rows if row.get("run_id") == run_id and _finite_float(row.get("t")) is not None]
    if not rows:
        return ""
    row = min(rows, key=lambda item: abs((_finite_float(item.get("t")) or 0.0) - t))
    return row.get("width_ratio", "")


def _component_value(row: Mapping[str, Any], key: str) -> float:
    if key.endswith("_sum"):
        return _finite_float(row.get(key)) or 0.0
    return _finite_float(row.get(key)) or 0.0


def _dominant_component(row: Mapping[str, Any]) -> str:
    components = {
        "insertion_truncation": _component_value(row, "insertion_truncation_width_sum"),
        "insertion_cutoff": _component_value(row, "insertion_cutoff_width_sum"),
        "picard_residual": _component_value(row, "picard_target_residual_width_sum"),
        "ordinary_remainder": _component_value(row, "ordinary_remainder_range_width_sum"),
        "normalized_right_map": _component_value(row, "normalized_right_map_range_width_sum"),
        "output_range_evaluation": _component_value(row, "output_range_width_sum"),
    }
    return max(components.items(), key=lambda item: item[1])[0]


def _dominant_dimension(row: Mapping[str, Any]) -> str:
    x_vals = [_finite_float(row.get(k)) or 0.0 for k in row if k.endswith("_x")]
    y_vals = [_finite_float(row.get(k)) or 0.0 for k in row if k.endswith("_y")]
    return "x" if max(x_vals or [0.0]) >= max(y_vals or [0.0]) else "y"


def build_trace(source_dir: Path) -> list[dict[str, Any]]:
    segments = _read_csv(source_dir / "normalized_insertion_h10_segments.csv") or _read_csv(source_dir / "rescue_segments.csv")
    attempts = _accepted_attempts(_read_csv(source_dir / "normalized_insertion_h10_validation_attempts.csv") or _read_csv(source_dir / "rescue_validation_attempts.csv"))
    ratios = _read_csv(source_dir / "rescue_vs_flowstar_ratio_trace.csv")
    rows: list[dict[str, Any]] = []
    for seg in segments:
        run_id = seg.get("run_id", "")
        if run_id not in RUN_IDS or seg.get("status") != "validated":
            continue
        idx = int(_finite_float(seg.get("segment_index")) or 0)
        attempt = attempts.get((run_id, idx), {})
        t_hi = _finite_float(seg.get("t_hi")) or 0.0
        trunc_sum = seg.get("insertion_truncation_width_sum") or seg.get("insertion_truncation_width", "")
        cutoff_sum = seg.get("insertion_cutoff_width_sum") or seg.get("insertion_cutoff_width", "")
        output_sum = seg.get("output_remainder_width_sum") or seg.get("output_remainder_width", "")
        right_sum = seg.get("normal_state_right_width_sum") or seg.get("inserted_endpoint_width_sum", "")
        row: dict[str, Any] = {
            "run_id": run_id,
            "segment_index": idx,
            "t_hi": t_hi,
            "reset_box_width_x": seg.get("reset_width_x", ""),
            "reset_box_width_y": seg.get("reset_width_y", ""),
            "reset_box_width_sum": seg.get("reset_width_sum", ""),
            "endpoint_tm_width_x": seg.get("endpoint_tm_width_x") or seg.get("width_x", ""),
            "endpoint_tm_width_y": seg.get("endpoint_tm_width_y") or seg.get("width_y", ""),
            "endpoint_tm_width_sum": seg.get("endpoint_tm_width_sum") or seg.get("width_sum", ""),
            "inserted_endpoint_width_x": seg.get("inserted_endpoint_width_x", ""),
            "inserted_endpoint_width_y": seg.get("inserted_endpoint_width_y", ""),
            "inserted_endpoint_width_sum": seg.get("inserted_endpoint_width_sum", ""),
            "insertion_truncation_width_x": seg.get("insertion_truncation_width_x", ""),
            "insertion_truncation_width_y": seg.get("insertion_truncation_width_y", ""),
            "insertion_truncation_width_sum": trunc_sum,
            "insertion_cutoff_width_x": seg.get("insertion_cutoff_width_x", ""),
            "insertion_cutoff_width_y": seg.get("insertion_cutoff_width_y", ""),
            "insertion_cutoff_width_sum": cutoff_sum,
            "picard_target_residual_width_x": attempt.get("residual_width_x", ""),
            "picard_target_residual_width_y": attempt.get("residual_width_y", ""),
            "picard_target_residual_width_sum": attempt.get("residual_width_sum", ""),
            "ordinary_remainder_range_width_x": attempt.get("remainder_width_x", ""),
            "ordinary_remainder_range_width_y": attempt.get("remainder_width_y", ""),
            "ordinary_remainder_range_width_sum": attempt.get("remainder_width_sum", ""),
            "normalized_right_map_range_width_x": seg.get("normal_state_right_width_x", ""),
            "normalized_right_map_range_width_y": seg.get("normal_state_right_width_y", ""),
            "normalized_right_map_range_width_sum": right_sum,
            "output_range_width_x": seg.get("output_remainder_width_x", ""),
            "output_range_width_y": seg.get("output_remainder_width_y", ""),
            "output_range_width_sum": output_sum,
            "scale_x": seg.get("scale_x", ""),
            "scale_y": seg.get("scale_y", ""),
            "terms_before_insertion_truncation": seg.get("terms_before_insertion_truncation", ""),
            "terms_after_insertion": seg.get("terms_after_insertion", ""),
            "nearest_flowstar_width_ratio": _nearest_ratio(ratios, run_id, t_hi),
        }
        if not row["insertion_truncation_width_sum"]:
            row["insertion_truncation_width_sum"] = _safe_sum(row["insertion_truncation_width_x"], row["insertion_truncation_width_y"])
        if not row["insertion_cutoff_width_sum"]:
            row["insertion_cutoff_width_sum"] = _safe_sum(row["insertion_cutoff_width_x"], row["insertion_cutoff_width_y"])
        row["dominant_component"] = _dominant_component(row)
        row["dominant_dimension"] = _dominant_dimension(row)
        rows.append(row)
    return rows


def _component_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = {
        "insertion_truncation": "insertion_truncation_width_sum",
        "right_map_scaling": "normalized_right_map_range_width_sum",
        "picard_residual": "picard_target_residual_width_sum",
        "output_range_evaluation": "output_range_width_sum",
    }
    return {name: max((_finite_float(row.get(key)) or 0.0 for row in rows), default=0.0) for name, key in keys.items()}


def _write_report(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    totals = _component_totals(rows)
    dominant = max(totals.items(), key=lambda item: item[1])[0] if totals else "unknown"
    o4_rows = [r for r in rows if r.get("run_id") == "flowstar_style_o4_target_insert"]
    o6_rows = [r for r in rows if r.get("run_id") == "flowstar_style_o6_candidate8_output6_insert"]
    o4_t = max((_finite_float(r.get("t_hi")) or 0.0 for r in o4_rows), default=0.0)
    o6_t = max((_finite_float(r.get("t_hi")) or 0.0 for r in o6_rows), default=0.0)
    o4_w = max((_finite_float(r.get("reset_box_width_sum")) or 0.0 for r in o4_rows), default=0.0)
    o6_w = max((_finite_float(r.get("reset_box_width_sum")) or 0.0 for r in o6_rows), default=0.0)
    recommendation = "normal-domain range evaluation" if dominant in {"picard_residual", "output_range_evaluation"} else ("right-map scalar alignment" if dominant == "right_map_scaling" else "Horner insertion accounting")
    lines = [
        "# Flowstar Insertion Width Attribution Report", "",
        f"Which component causes width growth? `{dominant}`; max component widths: {totals}.",
        f"Why does o6 run farther but widen much more? o6 reaches t=`{o6_t}` with max reset width `{o6_w}`, while o4 reaches t=`{o4_t}` with max reset width `{o4_w}`.",
        f"Why does o4 fail earlier but stay tighter? Its lower order/target path keeps reset widths smaller but leaves less residual slack near failure.",
        f"Does normalized right map degree/term count explode? max terms before/after insertion are reported in the CSV; dominant component is `{dominant}`.",
        f"Is the next fix source-map insertion Horner/intermediate range, symbolic queue, or polynomial range bounding? `{recommendation}`.",
        "", "## Component Maxima", "", "| component | max_width |", "| --- | ---: |",
    ]
    for name, value in totals.items():
        lines.append(f"| {name} | {value:.17g} |")
    lines.extend(["", "This report is diagnostic-only and does not claim exact Flow* parity."])
    (out_dir / "insertion_width_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _plots(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    fields = [
        ("insertion_truncation_width_sum", "truncation"),
        ("normalized_right_map_range_width_sum", "right map"),
        ("picard_target_residual_width_sum", "Picard residual"),
        ("output_range_width_sum", "output range"),
    ]
    for filename, stacked in [("insertion_component_stack.png", True), ("o4_vs_o6_width_sources.png", False)]:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for run_id in RUN_IDS:
            sub = sorted([r for r in rows if r.get("run_id") == run_id], key=lambda r: _finite_float(r.get("t_hi")) or 0.0)
            if not sub:
                continue
            ts = [_finite_float(r.get("t_hi")) or 0.0 for r in sub]
            if stacked and run_id == RUN_IDS[0]:
                bottoms = [0.0] * len(sub)
                for field, label in fields:
                    vals = [_finite_float(r.get(field)) or 0.0 for r in sub]
                    ax.fill_between(ts, bottoms, [a + b for a, b in zip(bottoms, vals)], alpha=0.25, label=label)
                    bottoms = [a + b for a, b in zip(bottoms, vals)]
            elif not stacked:
                vals = [_finite_float(r.get("reset_box_width_sum")) or 0.0 for r in sub]
                ax.plot(ts, vals, linewidth=1.0, label=run_id)
        ax.set_xlabel("t")
        ax.set_ylabel("width sum")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_trace(Path(args.source_dir))
    _write_csv(out_dir / "insertion_width_trace.csv", TRACE_FIELDS, rows)
    _write_report(out_dir, rows)
    _plots(out_dir, rows)
    print(f"wrote insertion width attribution to {out_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT / "outputs" / "flowstar_normalized_insertion_h10")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "flowstar_insertion_width_attribution")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
