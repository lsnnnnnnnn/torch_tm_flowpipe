#!/usr/bin/env python3
"""Audit normalized-insertion symbolic queue state channels."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_SPECS = [
    {
        "label": "no_queue",
        "dirs": ("flowstar_normalized_insertion_h10",),
        "segments": "normalized_insertion_h10_segments.csv",
        "summary": "normalized_insertion_h10_summary.csv",
        "run_ids": {
            "flowstar_style_o4_target_insert",
            "flowstar_style_o6_candidate8_output6_insert",
        },
    },
    {
        "label": "split_symqueue",
        "dirs": ("flowstar_normalized_insertion_symqueue_split_h10",),
        "segments": "symqueue_split_segments.csv",
        "summary": "symqueue_split_summary.csv",
        "run_ids": {
            "flowstar_style_o4_target_insert_symqueue_split",
            "flowstar_style_o6_candidate8_output6_insert_symqueue_split",
        },
    },
    {
        "label": "v2_symqueue",
        "dirs": (
            "flowstar_normalized_insertion_symqueue_v2_h10",
            "flowstar_symbolic_queue_v2_h10",
        ),
        "segments": "symqueue_v2_segments.csv",
        "summary": "symqueue_v2_summary.csv",
        "run_ids": {
            "flowstar_style_o4_target_insert_symqueue_v2",
            "flowstar_style_o6_candidate8_output6_insert_symqueue_v2",
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

SUMMARY_FIELDS = [
    "source",
    "run_id",
    "status",
    "last_validated_t",
    "reached_h10",
    "rows_loaded",
    "max_queue_size",
    "max_reset_box_width_sum",
    "max_right_map_range_width_sum",
    "max_target_check_width_sum",
    "max_output_only_symbolic_width_sum",
    "max_propagated_symbolic_width_sum",
    "target_check_symbolic_width_max",
    "output_range_includes_all_symbolic_contributions",
    "missing_paths",
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


def _source_dir(repo_root: Path, source: Mapping[str, Any]) -> tuple[Path, list[Path]]:
    candidates = [Path(repo_root) / "outputs" / str(name) for name in source.get("dirs", ())]
    for candidate in candidates:
        if (candidate / str(source["segments"])).exists() or (candidate / str(source["summary"])).exists():
            return candidate, candidates
    return candidates[0], candidates


def _selected_sources(labels: Sequence[str] | None, repo_root: Path) -> list[dict[str, Any]]:
    specs_by_label = {str(spec["label"]): spec for spec in SOURCE_SPECS}
    selected_labels = list(labels) if labels else [str(spec["label"]) for spec in SOURCE_SPECS]
    unknown = [label for label in selected_labels if label not in specs_by_label]
    if unknown:
        raise ValueError(f"unknown source label(s): {', '.join(unknown)}")
    selected: list[dict[str, Any]] = []
    for label in selected_labels:
        spec = dict(specs_by_label[label])
        src_dir, candidates = _source_dir(repo_root, spec)
        spec["dir"] = src_dir
        spec["candidate_dirs"] = candidates
        selected.append(spec)
    return selected


def _source_missing_paths(source: Mapping[str, Any]) -> list[str]:
    src_dir = Path(source["dir"])
    paths = [src_dir / str(source["summary"]), src_dir / str(source["segments"])]
    return [str(path) for path in paths if not path.exists()]


def build_trace(
    *,
    repo_root: Path = REPO_ROOT,
    source_labels: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, str]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for source in _selected_sources(source_labels, repo_root):
        src_dir = Path(source["dir"])
        label = str(source["label"])
        summary_rows = [row for row in _read_csv(src_dir / str(source["summary"])) if row.get("run_id") in source["run_ids"]]
        summaries[label] = summary_rows
        diagnostics[label] = {
            "dir": str(src_dir),
            "missing_paths": _source_missing_paths(source),
        }
        for row in _read_csv(src_dir / str(source["segments"])):
            if row.get("run_id") in source["run_ids"] and row.get("status") == "validated":
                rows.append(_row_from_segment(label, row))
    rows.sort(key=lambda r: (str(r.get("source", "")), str(r.get("run_id", "")), _finite_float(r.get("t_hi")) or 0.0))
    return rows, summaries, diagnostics


def _max(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    vals = [_finite_float(row.get(field)) for row in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _best_t(rows: Sequence[Mapping[str, str]]) -> float:
    return max((_finite_float(row.get("last_validated_t")) or 0.0 for row in rows), default=0.0)


def _rows_by_run(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    by_run: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_run.setdefault((str(row.get("source", "")), str(row.get("run_id", ""))), []).append(row)
    return by_run


def _summary_by_run(summaries: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[tuple[str, str], Mapping[str, str]]:
    return {
        (source, str(row.get("run_id", ""))): row
        for source, source_rows in summaries.items()
        for row in source_rows
    }


def _best_summary_row(rows: Sequence[Mapping[str, str]]) -> Mapping[str, str]:
    return max(rows, key=lambda row: _finite_float(row.get("last_validated_t")) or 0.0, default={})


def _best_run_id(source: str, summaries: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    return str(_best_summary_row(summaries.get(source, [])).get("run_id", ""))


def _rows_for_source_run(rows: Sequence[Mapping[str, Any]], source: str, run_id: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("source") == source and row.get("run_id") == run_id]


def _common_horizon(rows: Sequence[Mapping[str, str]]) -> float:
    vals = [_best_t([row]) for row in rows]
    vals = [value for value in vals if value > 0.0]
    return min(vals) if vals else 0.0


def _rows_to_horizon(rows: Sequence[Mapping[str, Any]], horizon: float) -> list[Mapping[str, Any]]:
    if horizon <= 0.0:
        return list(rows)
    return [row for row in rows if (_finite_float(row.get("t_hi")) or 0.0) <= horizon + 1e-12]


def _first_ratio_crossing(rows: Sequence[Mapping[str, str]], threshold: float = 1.0) -> float | None:
    candidates = []
    for row in rows:
        ratio = _finite_float(row.get("width_ratio"))
        t = _finite_float(row.get("t"))
        if ratio is not None and t is not None and ratio >= threshold:
            candidates.append(t)
    return min(candidates) if candidates else None


def _ratio_rows(diagnostics: Mapping[str, Mapping[str, Any]], source: str) -> list[dict[str, str]]:
    src_dir_raw = diagnostics.get(source, {}).get("dir")
    if not src_dir_raw:
        return []
    return _read_csv(Path(str(src_dir_raw)) / "rescue_vs_flowstar_ratio_trace.csv")


def build_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Sequence[Mapping[str, str]]],
    diagnostics: Mapping[str, Mapping[str, Any]],
    *,
    max_horizon: float = 10.0,
) -> list[dict[str, Any]]:
    by_run = _rows_by_run(rows)
    summary_by_run = _summary_by_run(summaries)
    out_rows: list[dict[str, Any]] = []
    all_keys = sorted(set(by_run) | set(summary_by_run))
    for source, run_id in all_keys:
        run_rows = by_run.get((source, run_id), [])
        summary = summary_by_run.get((source, run_id), {})
        last_t = _finite_float(summary.get("last_validated_t")) or _max(run_rows, "t_hi") or 0.0
        output_flags = [row for row in run_rows if row.get("output_range_includes_all_symbolic_contributions") != ""]
        output_ok = all(_truthy(row.get("output_range_includes_all_symbolic_contributions")) for row in output_flags) if output_flags else ""
        out_rows.append(
            {
                "source": source,
                "run_id": run_id,
                "status": summary.get("status", ""),
                "last_validated_t": last_t,
                "reached_h10": bool(last_t >= max_horizon - 1e-9),
                "rows_loaded": len(run_rows),
                "max_queue_size": _max(run_rows, "queue_size"),
                "max_reset_box_width_sum": _max(run_rows, "reset_box_width_sum"),
                "max_right_map_range_width_sum": _max(run_rows, "right_map_range_width_sum"),
                "max_target_check_width_sum": _max(run_rows, "contribution_included_in_target_check_width_sum"),
                "max_output_only_symbolic_width_sum": _max(run_rows, "contribution_included_only_in_output_range_width_sum"),
                "max_propagated_symbolic_width_sum": _max(run_rows, "propagated_symbolic_width_sum"),
                "target_check_symbolic_width_max": _max(run_rows, "contribution_included_in_target_check_width_sum"),
                "output_range_includes_all_symbolic_contributions": output_ok,
                "missing_paths": ";".join(str(path) for path in diagnostics.get(source, {}).get("missing_paths", [])),
            }
        )
    for source, diag in diagnostics.items():
        if summaries.get(source) or any(row.get("source") == source for row in rows):
            continue
        out_rows.append(
            {
                "source": source,
                "run_id": "",
                "status": "missing",
                "last_validated_t": "",
                "reached_h10": False,
                "rows_loaded": 0,
                "missing_paths": ";".join(str(path) for path in diag.get("missing_paths", [])),
            }
        )
    return out_rows


def write_report(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Sequence[Mapping[str, str]]],
    diagnostics: Mapping[str, Mapping[str, Any]],
    *,
    max_horizon: float = 10.0,
) -> None:
    no_t = _best_t(summaries.get("no_queue", []))
    split_t = _best_t(summaries.get("split_symqueue", []))
    v2_t = _best_t(summaries.get("v2_symqueue", []))
    no_best_run = _best_run_id("no_queue", summaries)
    split_best_run = _best_run_id("split_symqueue", summaries)
    v2_best_run = _best_run_id("v2_symqueue", summaries)
    best_summary_rows = [
        _best_summary_row(summaries.get("no_queue", [])),
        _best_summary_row(summaries.get("split_symqueue", [])),
        _best_summary_row(summaries.get("v2_symqueue", [])),
    ]
    common_horizon = _common_horizon([row for row in best_summary_rows if row])
    no_common = _rows_to_horizon(_rows_for_source_run(rows, "no_queue", no_best_run), common_horizon)
    split_common = _rows_to_horizon(_rows_for_source_run(rows, "split_symqueue", split_best_run), common_horizon)
    v2_common = _rows_to_horizon(_rows_for_source_run(rows, "v2_symqueue", v2_best_run), common_horizon)
    v2_reset = _max(v2_common, "reset_box_width_sum")
    no_reset = _max(no_common, "reset_box_width_sum")
    split_reset = _max(split_common, "reset_box_width_sum")
    v2_right = _max(v2_common, "right_map_range_width_sum")
    no_right = _max(no_common, "right_map_range_width_sum")
    split_right = _max(split_common, "right_map_range_width_sum")
    reset_comparable = v2_reset is not None and no_reset is not None and split_reset is not None
    right_comparable = v2_right is not None and no_right is not None and split_right is not None
    reset_answer = "yes" if reset_comparable and v2_reset < min(no_reset, split_reset) else ("no" if reset_comparable else "not comparable")
    right_answer = "yes" if right_comparable and v2_right < min(no_right, split_right) else ("no" if right_comparable else "not comparable")
    v2_target = _max(v2_common, "contribution_included_in_target_check_width_sum") or 0.0
    v2_output = _max(v2_common, "contribution_included_only_in_output_range_width_sum") or 0.0
    v2_output_ok = all(_truthy(row.get("output_range_includes_all_symbolic_contributions")) for row in v2_common) if v2_common else False
    ratio_note = "Flow* comparison ratio traces are missing for at least one source."
    ratio_crossings = {
        source: _first_ratio_crossing(_ratio_rows(diagnostics, source))
        for source in ("no_queue", "split_symqueue", "v2_symqueue")
    }
    if any(value is not None for value in ratio_crossings.values()):
        ratio_note = ", ".join(
            f"{source} first width_ratio>=1 at {_fmt(value) if value is not None else 'not observed'}"
            for source, value in ratio_crossings.items()
        )
    missing_lines = [
        f"- `{source}` missing `{path}`"
        for source, diag in diagnostics.items()
        for path in diag.get("missing_paths", [])
    ]
    incomplete = bool(missing_lines)
    lines = [
        "# Flowstar Queue State Audit",
        "",
        "This audit reads normalized-insertion no-queue, split-symbolic-queue, and v2 symbolic-queue outputs.",
        "It is a raw diagnostics audit; missing raw fields are reported as blanks rather than reconstructed from Flow* source.",
        f"Audit status: {'incomplete' if incomplete else 'complete three-way input set loaded'}.",
        "",
        "## Answers",
        "",
        f"Did v2 reach h10? {'yes' if v2_t >= max_horizon - 1e-9 else 'no'}; best v2 t=`{_fmt(v2_t)}`.",
        f"Did v2 beat no_queue on last_validated_t? {'yes' if v2_t > no_t + 1e-12 else 'no'}; no_queue best t=`{_fmt(no_t)}`.",
        f"Did v2 beat split_symqueue on last_validated_t? {'yes' if v2_t > split_t + 1e-12 else 'no'}; split best t=`{_fmt(split_t)}`.",
        f"Common horizon for best-run width comparisons: `{_fmt(common_horizon)}`.",
        f"Did v2 reduce max reset width before the common horizon? {reset_answer}; no_queue=`{_fmt(no_reset)}`, split=`{_fmt(split_reset)}`, v2=`{_fmt(v2_reset)}`.",
        f"Did v2 reduce max right-map range width before the common horizon? {right_answer}; no_queue=`{_fmt(no_right)}`, split=`{_fmt(split_right)}`, v2=`{_fmt(v2_right)}`.",
        f"Did v2 move Flow* width-ratio crossings later? {ratio_note}.",
        f"Did v2 keep symbolic width out of target check? {'yes' if v2_target <= 1e-15 else 'no'}; max target contribution=`{_fmt(v2_target)}`.",
        f"Did v2 add symbolic width only to output/range boxes? {'yes' if v2_output > 0.0 and v2_output_ok else 'no'}; max output-only symbolic=`{_fmt(v2_output)}`; output flags all true={v2_output_ok}.",
        "",
        "## Trace Columns",
        "",
        "See `queue_state_trace.csv` for per-segment J/Phi_L/scalar, target, reset, right-map, and output-only channels.",
        "See `queue_state_summary.csv` for source/run-level metrics.",
        "",
        "## Run Summary",
        "",
        "| source | run_id | status | last_validated_t | max_queue | max_reset_common | max_right_common | max_output_only_common |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_run = _rows_by_run(rows)
    summary_by_run = _summary_by_run(summaries)
    for key in sorted(set(by_run) | set(summary_by_run)):
        source, run_id = key
        run_rows = _rows_to_horizon(by_run.get(key, []), common_horizon)
        summary = summary_by_run.get((source, run_id), {})
        lines.append(
            f"| {source} | {run_id} | {summary.get('status', '')} | {summary.get('last_validated_t', '')} | "
            f"{_fmt(_max(run_rows, 'queue_size'))} | {_fmt(_max(run_rows, 'reset_box_width_sum'))} | "
            f"{_fmt(_max(run_rows, 'right_map_range_width_sum'))} | "
            f"{_fmt(_max(run_rows, 'contribution_included_only_in_output_range_width_sum'))} |"
        )
    if missing_lines:
        lines.extend(["", "## Missing Data", "", *missing_lines])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "queue_state_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_queue_state_audit"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source", action="append", choices=[str(spec["label"]) for spec in SOURCE_SPECS], default=None)
    parser.add_argument("--include-missing", action="store_true", help="Keep missing source diagnostics in the summary/report.")
    parser.add_argument("--max-horizon", type=float, default=10.0)
    args = parser.parse_args(argv)
    rows, summaries, diagnostics = build_trace(repo_root=args.repo_root, source_labels=args.source)
    _write_csv(args.out_dir / "queue_state_trace.csv", TRACE_FIELDS, rows)
    summary_rows = build_summary_rows(rows, summaries, diagnostics, max_horizon=float(args.max_horizon))
    _write_csv(args.out_dir / "queue_state_summary.csv", SUMMARY_FIELDS, summary_rows)
    write_report(args.out_dir, rows, summaries, diagnostics, max_horizon=float(args.max_horizon))
    print(f"wrote queue state audit to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
