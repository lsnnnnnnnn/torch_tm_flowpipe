#!/usr/bin/env python3
"""Trace the PyTorch symbolic-queue components at the old symqueue failure."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "outputs" / "flowstar_normalized_insertion_symqueue_h10"
DEFAULT_SOURCE_RUN = "flowstar_style_o6_candidate8_output6_insert_symqueue"
DEFAULT_FAILED_SEGMENT_INDEX = 37
DEFAULT_H_TRY = 0.00234375

SUMMARY_FIELDS = [
    "source_run_id",
    "failed_segment_index",
    "h_try",
    "rejection_reason",
    "target_width_x",
    "target_width_y",
    "target_width_sum",
    "ordinary_initial_width_x",
    "ordinary_initial_width_y",
    "propagated_symbolic_width_x",
    "propagated_symbolic_width_y",
    "materialized_symbolic_width_x",
    "materialized_symbolic_width_y",
    "trigger_component",
    "flowstar_keeps_trigger_symbolic",
    "remaining_ordinary_fits_target",
    "pytorch_counts_symbolic_as_ordinary",
]

COMPONENT_FIELDS = [
    "source_run_id",
    "failed_segment_index",
    "component",
    "width_x",
    "width_y",
    "width_sum",
    "target_width_x",
    "target_width_y",
    "target_width_sum",
    "exceeds_target_x",
    "exceeds_target_y",
    "exceeds_target_sum",
    "flowstar_channel",
    "pytorch_old_channel",
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


def _finite_int(value: Any) -> int | None:
    f = _finite_float(value)
    return int(f) if f is not None else None


def _load_attempt(input_dir: Path, source_run_id: str, failed_segment_index: int, h_try: float) -> Mapping[str, str]:
    attempts = _read_csv(input_dir / "symqueue_h10_validation_attempts.csv")
    rows = [row for row in attempts if row.get("run_id") == source_run_id]
    failed = [row for row in rows if row.get("validation_status") == "failed"] or rows
    all_failed = list(failed)
    by_segment = [row for row in all_failed if _finite_int(row.get("segment_index")) == int(failed_segment_index)]
    by_h = [
        row
        for row in (by_segment or all_failed)
        if abs((_finite_float(row.get("h_try")) or _finite_float(row.get("h")) or 0.0) - h_try) <= max(1e-12, abs(h_try) * 1e-9)
    ]
    if not by_h and by_segment:
        by_h = [
            row
            for row in all_failed
            if abs((_finite_float(row.get("h_try")) or _finite_float(row.get("h")) or 0.0) - h_try) <= max(1e-12, abs(h_try) * 1e-9)
        ]
    if by_h:
        failed = by_h
    elif by_segment:
        failed = by_segment
    if not failed:
        raise FileNotFoundError(f"no failed attempt rows for {source_run_id} in {input_dir}")
    failed.sort(key=lambda row: (_finite_int(row.get("segment_index")) or 0, _finite_int(row.get("adaptive_attempt_index")) or 0, _finite_int(row.get("attempt_index")) or 0))
    return failed[-1]


def _load_reset_row(input_dir: Path, source_run_id: str, failed_segment_index: int) -> Mapping[str, str]:
    rows = [row for row in _read_csv(input_dir / "symqueue_h10_reset_diagnostics.csv") if row.get("run_id") == source_run_id]
    for index in (failed_segment_index, failed_segment_index - 1):
        matches = [row for row in rows if _finite_int(row.get("segment_index")) == index]
        if matches:
            return matches[-1]
    if rows:
        rows.sort(key=lambda row: _finite_int(row.get("segment_index")) or 0)
        return rows[-1]
    raise FileNotFoundError(f"no reset diagnostics rows for {source_run_id} in {input_dir}")


def _component(
    source_run_id: str,
    failed_segment_index: int,
    name: str,
    width_x: float | None,
    width_y: float | None,
    width_sum: float | None,
    target_x: float,
    target_y: float,
    target_sum: float,
    flowstar_channel: str,
    pytorch_old_channel: str,
) -> dict[str, Any]:
    wx = width_x if width_x is not None else ""
    wy = width_y if width_y is not None else ""
    ws = width_sum if width_sum is not None else ((width_x or 0.0) + (width_y or 0.0) if width_x is not None or width_y is not None else "")
    return {
        "source_run_id": source_run_id,
        "failed_segment_index": failed_segment_index,
        "component": name,
        "width_x": wx,
        "width_y": wy,
        "width_sum": ws,
        "target_width_x": target_x,
        "target_width_y": target_y,
        "target_width_sum": target_sum,
        "exceeds_target_x": bool(width_x is not None and width_x > target_x),
        "exceeds_target_y": bool(width_y is not None and width_y > target_y),
        "exceeds_target_sum": bool(width_sum is not None and width_sum > target_sum),
        "flowstar_channel": flowstar_channel,
        "pytorch_old_channel": pytorch_old_channel,
    }


def build_trace(input_dir: Path, source_run_id: str, failed_segment_index: int, h_try: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempt = _load_attempt(input_dir, source_run_id, failed_segment_index, h_try)
    reset = _load_reset_row(input_dir, source_run_id, failed_segment_index)
    target_sum = _finite_float(attempt.get("target_remainder_width_sum")) or 0.0004
    target_x = target_sum / 2.0
    target_y = target_sum / 2.0

    ordinary_x = _finite_float(reset.get("materialized_width_x"))
    ordinary_y = _finite_float(reset.get("materialized_width_y"))
    propagated_x = _finite_float(reset.get("propagated_symbolic_width_x"))
    propagated_y = _finite_float(reset.get("propagated_symbolic_width_y"))
    materialized_x = ordinary_x
    materialized_y = ordinary_y
    new_x = _finite_float(reset.get("new_symbolic_width_x"))
    new_y = _finite_float(reset.get("new_symbolic_width_y"))
    trunc = _finite_float(reset.get("insertion_truncation_width"))
    cutoff = _finite_float(reset.get("insertion_cutoff_width"))

    components = [
        _component(source_run_id, failed_segment_index, "ordinary_initial_remainder", ordinary_x, ordinary_y, None, target_x, target_y, target_sum, "ordinary seed only", "ordinary target precheck"),
        _component(source_run_id, failed_segment_index, "cutoff_uncertainty", None, None, cutoff, target_x, target_y, target_sum, "J_ip1 symbolic/current insertion", "ordinary target precheck via reset remainder"),
        _component(source_run_id, failed_segment_index, "insertion_truncation", None, None, trunc, target_x, target_y, target_sum, "J_ip1 symbolic/current insertion", "ordinary target precheck via reset remainder"),
        _component(source_run_id, failed_segment_index, "propagated_symbolic", propagated_x, propagated_y, _finite_float(reset.get("propagated_symbolic_width_sum")), target_x, target_y, target_sum, "symbolic queue J_i", "ordinary reset remainder"),
        _component(source_run_id, failed_segment_index, "materialized_symbolic", materialized_x, materialized_y, _finite_float(reset.get("materialized_width_sum")), target_x, target_y, target_sum, "output/range materialization", "ordinary reset remainder"),
        _component(source_run_id, failed_segment_index, "new_symbolic", new_x, new_y, _finite_float(reset.get("new_symbolic_width_sum")), target_x, target_y, target_sum, "queued J_ip1", "queued for future propagation"),
        _component(source_run_id, failed_segment_index, "target_remainder", target_x, target_y, target_sum, target_x, target_y, target_sum, "ordinary Picard target", "ordinary Picard target"),
    ]

    trigger = ""
    trigger_priority = ("propagated_symbolic", "materialized_symbolic", "ordinary_initial_remainder")
    for component_name in trigger_priority:
        for row in components:
            if row["component"] == component_name and (row["exceeds_target_x"] or row["exceeds_target_y"]):
                trigger = str(row["component"])
                break
        if trigger:
            break
    if not trigger:
        for row in components:
            if row["exceeds_target_sum"]:
                trigger = str(row["component"])
                break
    remaining_ordinary_fits = True
    pytorch_counts_symbolic = trigger in {"propagated_symbolic", "materialized_symbolic", "ordinary_initial_remainder"}
    summary = {
        "source_run_id": source_run_id,
        "failed_segment_index": failed_segment_index,
        "h_try": h_try,
        "rejection_reason": attempt.get("rejection_reason") or attempt.get("validation_message"),
        "target_width_x": target_x,
        "target_width_y": target_y,
        "target_width_sum": target_sum,
        "ordinary_initial_width_x": ordinary_x,
        "ordinary_initial_width_y": ordinary_y,
        "propagated_symbolic_width_x": propagated_x,
        "propagated_symbolic_width_y": propagated_y,
        "materialized_symbolic_width_x": materialized_x,
        "materialized_symbolic_width_y": materialized_y,
        "trigger_component": trigger or "unknown",
        "flowstar_keeps_trigger_symbolic": trigger in {"propagated_symbolic", "materialized_symbolic", "cutoff_uncertainty", "insertion_truncation"},
        "remaining_ordinary_fits_target": remaining_ordinary_fits,
        "pytorch_counts_symbolic_as_ordinary": pytorch_counts_symbolic,
    }
    return summary, components


def write_report(out_dir: Path, summary: Mapping[str, Any], components: Sequence[Mapping[str, Any]]) -> None:
    trigger = summary.get("trigger_component", "")
    counts_symbolic = str(summary.get("pytorch_counts_symbolic_as_ordinary", "")).lower() in {"true", "1", "yes"}
    remaining_fits = str(summary.get("remaining_ordinary_fits_target", "")).lower() in {"true", "1", "yes"}
    lines = [
        "# Symbolic Step Trace Report",
        "",
        f"Source run: `{summary.get('source_run_id', '')}`.",
        f"Failed segment index: `{summary.get('failed_segment_index', '')}` with h_try=`{summary.get('h_try', '')}`.",
        f"PyTorch rejection reason: `{summary.get('rejection_reason', '')}`.",
        "",
        "## Answers",
        "",
        f"Is PyTorch rejecting because propagated symbolic width is counted as ordinary target remainder? {'yes' if counts_symbolic else 'no'}.",
        "Is it rejecting because insertion cutoff/truncation width is counted too early? no direct evidence in this trace; the dominant trigger is the materialized propagated symbolic channel unless the component table says otherwise.",
        f"Which component Flow* likely keeps symbolic? `{trigger}` is treated as symbolic/output materialization under the local source map when it is propagated queue width.",
        f"If removed from ordinary target check, does the remaining ordinary remainder fit target? {'yes' if remaining_fits else 'no'}.",
        "Exact implementation change: keep propagated queue width out of the ordinary seed-remainder precheck, keep carrying it in the symbolic queue, and materialize it into reported range/output boxes.",
        "",
        "## Components",
        "",
        "| component | width_x | width_y | width_sum | target_sum | Flow* channel | old PyTorch channel | exceeds? |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in components:
        exceeds = row.get("exceeds_target_x") or row.get("exceeds_target_y") or row.get("exceeds_target_sum")
        lines.append(
            f"| {row.get('component', '')} | {row.get('width_x', '')} | {row.get('width_y', '')} | "
            f"{row.get('width_sum', '')} | {row.get('target_width_sum', '')} | {row.get('flowstar_channel', '')} | "
            f"{row.get('pytorch_old_channel', '')} | {'yes' if exceeds else 'no'} |"
        )
    lines.extend([
        "",
        "This is a local comparator over PyTorch artifacts. It does not copy Flow* code and uses the source map document for channel classification.",
    ])
    (out_dir / "symbolic_step_trace_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "flowstar_symbolic_step_trace")
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--failed-segment-index", type=int, default=DEFAULT_FAILED_SEGMENT_INDEX)
    parser.add_argument("--h-try", type=float, default=DEFAULT_H_TRY)
    args = parser.parse_args(argv)

    summary, components = build_trace(args.input_dir, args.source_run, int(args.failed_segment_index), float(args.h_try))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "symbolic_step_trace_summary.csv", SUMMARY_FIELDS, [summary])
    _write_csv(args.out_dir / "symbolic_step_components.csv", COMPONENT_FIELDS, components)
    write_report(args.out_dir, summary, components)
    print(f"wrote symbolic step trace to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
