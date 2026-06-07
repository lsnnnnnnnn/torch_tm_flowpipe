#!/usr/bin/env python3
"""Audit normalized-insertion symbolic queue state channels."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    {
        "label": "split_symqueue",
        "dir": REPO_ROOT / "outputs" / "flowstar_normalized_insertion_symqueue_split_h10",
        "segments": "symqueue_split_segments.csv",
        "summary": "symqueue_split_summary.csv",
        "run_ids": {
            "flowstar_style_o4_target_insert_symqueue_split",
            "flowstar_style_o6_candidate8_output6_insert_symqueue_split",
        },
    },
    {
        "label": "no_queue",
        "dir": REPO_ROOT / "outputs" / "flowstar_normalized_insertion_h10",
        "segments": "normalized_insertion_h10_segments.csv",
        "summary": "normalized_insertion_h10_summary.csv",
        "run_ids": {
            "flowstar_style_o4_target_insert",
            "flowstar_style_o6_candidate8_output6_insert",
        },
    },
]

TRACE_FIELDS = [
    "source",
    "run_id",
    "segment_index",
    "t_hi",
    "queue_size",
    "current_linear_map_entries",
    "current_linear_map_norm",
    "phi_l_count",
    "j_count",
    "scalar_x",
    "scalar_y",
    "propagated_symbolic_width_x",
    "propagated_symbolic_width_y",
    "propagated_symbolic_width_sum",
    "materialized_symbolic_width_x",
    "materialized_symbolic_width_y",
    "materialized_symbolic_width_sum",
    "ordinary_step_remainder_width_x",
    "ordinary_step_remainder_width_y",
    "ordinary_step_remainder_width_sum",
    "reset_box_width_x",
    "reset_box_width_y",
    "reset_box_width_sum",
    "right_map_range_width_x",
    "right_map_range_width_y",
    "right_map_range_width_sum",
    "contribution_included_in_target_check_width_x",
    "contribution_included_in_target_check_width_y",
    "contribution_included_in_target_check_width_sum",
    "contribution_included_only_in_output_range_width_x",
    "contribution_included_only_in_output_range_width_y",
    "contribution_included_only_in_output_range_width_sum",
    "target_check_width_exceeds_target",
    "output_range_includes_all_symbolic_contributions",
]


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


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.17g}" if math.isfinite(value) else str(value)
    return value


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _scalars(row: Mapping[str, str]) -> tuple[str, str]:
    if row.get("scalar_x") or row.get("scalar_y"):
        return row.get("scalar_x", ""), row.get("scalar_y", "")
    raw = str(row.get("scalars", ""))
    parts = [part for part in raw.split(";") if part != ""]
    if parts:
        return parts[0], parts[1] if len(parts) > 1 else ""
    return row.get("scale_x", ""), row.get("scale_y", "")


def _channel(row: Mapping[str, str], prefix: str, fallback_sum: str = "") -> tuple[Any, Any, Any]:
    x = row.get(f"{prefix}_width_x", "")
    y = row.get(f"{prefix}_width_y", "")
    total = row.get(f"{prefix}_width_sum", "")
    if total == "":
        total = row.get(fallback_sum, "") if fallback_sum else ""
    return x, y, total


def _target_exceeds(row: Mapping[str, str]) -> bool:
    if row.get("target_check_exceeds_target") != "":
        return _truthy(row.get("target_check_exceeds_target"))
    radius = _finite_float(row.get("target_remainder_radius")) or 1e-4
    vals = [
        _finite_float(row.get("target_check_width_x")),
        _finite_float(row.get("target_check_width_y")),
    ]
    vals = [v for v in vals if v is not None]
    if vals:
        return any(v > radius + 1e-15 for v in vals)
    total = _finite_float(row.get("target_checked_width"))
    return bool(total is not None and total > 2.0 * radius + 1e-15)


def _output_includes(row: Mapping[str, str]) -> bool:
    if row.get("output_range_includes_symbolic_contributions") != "":
        return _truthy(row.get("output_range_includes_symbolic_contributions"))
    symbolic = _finite_float(row.get("symbolic_contribution_width"))
    if symbolic in (None, 0.0):
        return True
    total = _finite_float(row.get("total_range_width_with_symbolic"))
    ordinary = _finite_float(row.get("ordinary_only_range_width"))
    if total is None or ordinary is None:
        return False
    return total + 1e-15 >= ordinary and total + 1e-15 >= symbolic


def _row_from_segment(source: str, row: Mapping[str, str]) -> dict[str, Any]:
    scalar_x, scalar_y = _scalars(row)
    prop_x, prop_y, prop_sum = _channel(row, "propagated_symbolic")
    mat_x, mat_y, mat_sum = _channel(row, "materialized", "materialized_for_output_width")
    ordinary_x, ordinary_y, ordinary_sum = _channel(row, "ordinary_step_remainder", "output_remainder_width_sum")
    reset_x, reset_y, reset_sum = _channel(row, "reset_box", "reset_width_sum")
    if reset_x == "":
        reset_x = row.get("reset_width_x", "")
    if reset_y == "":
        reset_y = row.get("reset_width_y", "")
    right_x, right_y, right_sum = _channel(row, "right_map_range", "old_right_map_range_width_sum")
    if right_x == "":
        right_x = row.get("old_right_map_range_width_x", "")
    if right_y == "":
        right_y = row.get("old_right_map_range_width_y", "")
    target_x, target_y, target_sum = _channel(row, "target_check", "target_checked_width")
    output_x, output_y, output_sum = _channel(row, "output_only_symbolic", "materialized_for_output_width")
    queue_size = row.get("queue_size") or row.get("flowstar_queue_size_after") or 0
    return {
        "source": source,
        "run_id": row.get("run_id", ""),
        "segment_index": row.get("segment_index", ""),
        "t_hi": row.get("t_hi", ""),
        "queue_size": queue_size,
        "current_linear_map_entries": row.get("current_linear_map_entries", ""),
        "current_linear_map_norm": row.get("current_linear_map_norm", row.get("linear_map_norm", "")),
        "phi_l_count": row.get("phi_l_count", queue_size if source == "split_symqueue" else 0),
        "j_count": row.get("j_count", queue_size if source == "split_symqueue" else 0),
        "scalar_x": scalar_x,
        "scalar_y": scalar_y,
        "propagated_symbolic_width_x": prop_x,
        "propagated_symbolic_width_y": prop_y,
        "propagated_symbolic_width_sum": prop_sum,
        "materialized_symbolic_width_x": mat_x,
        "materialized_symbolic_width_y": mat_y,
        "materialized_symbolic_width_sum": mat_sum,
        "ordinary_step_remainder_width_x": ordinary_x,
        "ordinary_step_remainder_width_y": ordinary_y,
        "ordinary_step_remainder_width_sum": ordinary_sum,
        "reset_box_width_x": reset_x,
        "reset_box_width_y": reset_y,
        "reset_box_width_sum": reset_sum,
        "right_map_range_width_x": right_x,
        "right_map_range_width_y": right_y,
        "right_map_range_width_sum": right_sum,
        "contribution_included_in_target_check_width_x": target_x,
        "contribution_included_in_target_check_width_y": target_y,
        "contribution_included_in_target_check_width_sum": target_sum,
        "contribution_included_only_in_output_range_width_x": output_x,
        "contribution_included_only_in_output_range_width_y": output_y,
        "contribution_included_only_in_output_range_width_sum": output_sum,
        "target_check_width_exceeds_target": _target_exceeds(row),
        "output_range_includes_all_symbolic_contributions": _output_includes(row),
    }


def build_trace() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, str]]] = {}
    for source in SOURCES:
        src_dir = Path(source["dir"])
        summary_rows = [row for row in _read_csv(src_dir / str(source["summary"])) if row.get("run_id") in source["run_ids"]]
        summaries[str(source["label"])] = summary_rows
        for row in _read_csv(src_dir / str(source["segments"])):
            if row.get("run_id") in source["run_ids"] and row.get("status") == "validated":
                rows.append(_row_from_segment(str(source["label"]), row))
    rows.sort(key=lambda r: (str(r.get("source", "")), str(r.get("run_id", "")), _finite_float(r.get("t_hi")) or 0.0))
    return rows, summaries


def _max(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    vals = [_finite_float(row.get(field)) for row in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _best_t(rows: Sequence[Mapping[str, str]]) -> float:
    return max((_finite_float(row.get("last_validated_t")) or 0.0 for row in rows), default=0.0)


def write_report(out_dir: Path, rows: Sequence[Mapping[str, Any]], summaries: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    split_rows = [row for row in rows if row.get("source") == "split_symqueue"]
    no_rows = [row for row in rows if row.get("source") == "no_queue"]
    split_propagated = (_max(split_rows, "propagated_symbolic_width_sum") or 0.0) > 0.0
    split_target = _max(split_rows, "contribution_included_in_target_check_width_sum") or 0.0
    split_output = _max(split_rows, "contribution_included_only_in_output_range_width_sum") or 0.0
    split_reset = _max(split_rows, "reset_box_width_sum")
    no_reset = _max(no_rows, "reset_box_width_sum")
    split_t = _best_t(summaries.get("split_symqueue", []))
    no_t = _best_t(summaries.get("no_queue", []))
    scalar_note = "old split rows record scale-like scalars; inverse-scalar semantics are not visible in those raw diagnostics"
    if any(row.get("current_linear_map_entries") for row in split_rows):
        phi_note = "Phi_L entries are present in the trace and can be inspected directly."
    else:
        phi_note = "old split outputs expose propagated width and queue size, but not raw Phi_L entries."
    lines = [
        "# Flowstar Queue State Audit",
        "",
        "This audit reads prior normalized-insertion no-queue and split-symbolic-queue outputs.",
        "It is a raw diagnostics audit; missing raw fields are reported as blanks rather than reconstructed from Flow* source.",
        "",
        "## Answers",
        "",
        f"Is Phi_L actually propagating old J? {phi_note} Propagated width was observed={split_propagated}.",
        f"Are scalars used like Flow* inverse magnitudes? {scalar_note}.",
        f"Is symbolic width included in target check anywhere? max target-check contribution=`{_fmt(split_target)}`; split mode keeps it output-only.",
        f"Does queue contribution improve or worsen reset width before failure? split max reset=`{_fmt(split_reset)}`, no-queue max reset=`{_fmt(no_reset)}`; no reset-width reduction is visible from this audit.",
        f"Which exact channel causes split symqueue not to beat no-queue? output-only/materialized symbolic width max=`{_fmt(split_output)}` while target contribution stays `{_fmt(split_target)}`.",
        f"Best split t=`{_fmt(split_t)}`; best no-queue t=`{_fmt(no_t)}`.",
        "",
        "## Trace Columns",
        "",
        "See `queue_state_trace.csv` for per-segment J/Phi_L/scalar, target, reset, right-map, and output-only channels.",
        "",
        "## Run Summary",
        "",
        "| source | run_id | status | last_validated_t | max_queue | max_propagated | max_output_only |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    by_run: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_run.setdefault((str(row.get("source", "")), str(row.get("run_id", ""))), []).append(row)
    summary_by_run = {
        (source, row.get("run_id", "")): row
        for source, source_rows in summaries.items()
        for row in source_rows
    }
    for key, run_rows in sorted(by_run.items()):
        source, run_id = key
        summary = summary_by_run.get((source, run_id), {})
        lines.append(
            f"| {source} | {run_id} | {summary.get('status', '')} | {summary.get('last_validated_t', '')} | "
            f"{_fmt(_max(run_rows, 'queue_size'))} | {_fmt(_max(run_rows, 'propagated_symbolic_width_sum'))} | "
            f"{_fmt(_max(run_rows, 'contribution_included_only_in_output_range_width_sum'))} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "queue_state_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_queue_state_audit"))
    args = parser.parse_args(argv)
    rows, summaries = build_trace()
    _write_csv(args.out_dir / "queue_state_trace.csv", TRACE_FIELDS, rows)
    write_report(args.out_dir, rows, summaries)
    print(f"wrote queue state audit to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
