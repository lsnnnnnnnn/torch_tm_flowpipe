#!/usr/bin/env python3
"""Audit Flow* vs PyTorch box lifecycle alignment at the first mismatch.

This diagnostic reads the stage-labeled traces emitted by
``experiments/flowstar_step_trace_compare.py``. It does not add a solver
mechanism, change default solver behavior, rerun h10, or claim Flow* parity.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "outputs" / "flowstar_step_trace_compare"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_box_lifecycle_alignment_audit"

BOX_PREFIXES = [
    "pre_step_box",
    "endpoint_box_before_center",
    "reset_box_after_center_scale",
]

RESIDUAL_COMPONENTS = [
    "picard_no_remainder_residual",
    "picard_ctrunc_raw_residual",
    "post_cutoff_residual",
]

LEDGER_FIELDS = [
    "source",
    "t_before",
    "h_try",
    "status",
    "pre_step_box_x_lo",
    "pre_step_box_x_hi",
    "pre_step_box_y_lo",
    "pre_step_box_y_hi",
    "endpoint_box_before_center_x_lo",
    "endpoint_box_before_center_x_hi",
    "endpoint_box_before_center_y_lo",
    "endpoint_box_before_center_y_hi",
    "extracted_center_x",
    "extracted_center_y",
    "extracted_scale_x",
    "extracted_scale_y",
    "reset_box_after_center_scale_x_lo",
    "reset_box_after_center_scale_x_hi",
    "reset_box_after_center_scale_y_lo",
    "reset_box_after_center_scale_y_hi",
    "target_remainder_x_lo",
    "target_remainder_x_hi",
    "target_remainder_y_lo",
    "target_remainder_y_hi",
    "picard_no_remainder_residual_x_lo",
    "picard_no_remainder_residual_x_hi",
    "picard_no_remainder_residual_y_lo",
    "picard_no_remainder_residual_y_hi",
    "picard_ctrunc_raw_residual_x_lo",
    "picard_ctrunc_raw_residual_x_hi",
    "picard_ctrunc_raw_residual_y_lo",
    "picard_ctrunc_raw_residual_y_hi",
    "cutoff_polynomial_difference_x_width",
    "cutoff_polynomial_difference_y_width",
    "post_cutoff_residual_x_lo",
    "post_cutoff_residual_x_hi",
    "post_cutoff_residual_y_lo",
    "post_cutoff_residual_y_hi",
    "pre_step_matches_flowstar",
    "pre_step_boxes_equal",
    "endpoint_before_center_comparable",
    "endpoint_before_center_matches_flowstar",
    "endpoint_before_center_boxes_equal",
    "reset_after_center_comparable",
    "reset_after_center_matches_flowstar",
    "reset_after_center_boxes_equal",
    "first_lifecycle_stage_divergence",
    "residual_comparison_stage_valid",
    "picard_residual_comparison",
    "flowstar_missing_residual_components",
    "notes",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in LEDGER_FIELDS})


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.17g}"
    return str(value)


def finite_float(value: Any) -> float | None:
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


def _status(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "missing"
    raw = str(row.get("status", "")).strip().lower()
    if raw:
        return raw
    if str(row.get("accepted", "")).strip().lower() in {"1", "true", "yes", "validated"}:
        return "accepted"
    if str(row.get("rejected", "")).strip().lower() in {"1", "true", "yes", "failed"}:
        return "rejected"
    return "missing"


def _h_try(row: Mapping[str, Any]) -> float | None:
    return finite_float(_first_present(row, "h_try", "h_forced", "h"))


def _find_attempt(rows: Iterable[Mapping[str, Any]], *, t: float, h: float, tolerance: float = 1e-9) -> Mapping[str, Any] | None:
    for row in rows:
        t_before = finite_float(row.get("t_before"))
        h_try = _h_try(row)
        if t_before is None or h_try is None:
            continue
        if abs(t_before - t) <= tolerance and abs(h_try - h) <= tolerance:
            return row
    return None


def _box(row: Mapping[str, Any], prefix: str) -> tuple[float, float, float, float] | None:
    values = (
        finite_float(row.get(f"{prefix}_x_lo")),
        finite_float(row.get(f"{prefix}_x_hi")),
        finite_float(row.get(f"{prefix}_y_lo")),
        finite_float(row.get(f"{prefix}_y_hi")),
    )
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def _box_equal(a: tuple[float, float, float, float], b: tuple[float, float, float, float], *, tol: float) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _compare_to_flowstar(flow: Mapping[str, Any], row: Mapping[str, Any], prefix: str, *, tol: float) -> tuple[str, str]:
    flow_box = _box(flow, prefix)
    row_box = _box(row, prefix)
    if flow_box is None or row_box is None:
        return "unknown", "unknown"
    return "true", "true" if _box_equal(flow_box, row_box, tol=tol) else "false"


def _all_known_true(values: Iterable[str]) -> str:
    vals = list(values)
    if any(value == "unknown" for value in vals):
        return "unknown"
    return "true" if all(value == "true" for value in vals) else "false"


def _component_missing(row: Mapping[str, Any], prefix: str) -> bool:
    return any(row.get(f"{prefix}_{dim}_{side}") in (None, "") for dim in ("x", "y") for side in ("lo", "hi"))


def _flowstar_missing_components(flow: Mapping[str, Any]) -> str:
    missing = [prefix for prefix in RESIDUAL_COMPONENTS if _component_missing(flow, prefix)]
    return ";".join(missing)


def _copy_stage_fields(source: str, row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": source,
        "t_before": row.get("t_before", ""),
        "h_try": _first_present(row, "h_try", "h"),
        "status": _status(row),
    }
    for field in LEDGER_FIELDS:
        if field in out:
            continue
        if field in row:
            out[field] = row.get(field, "")
    return out


def summarize_alignment(rows: list[dict[str, Any]], *, tol: float = 1e-12) -> dict[str, Any]:
    flow = next(row for row in rows if row["source"] == "flowstar")
    candidates = [row for row in rows if row["source"].startswith("torch")]
    pre_matches: list[str] = []
    endpoint_comparable: list[str] = []
    endpoint_matches: list[str] = []
    reset_comparable: list[str] = []
    reset_matches: list[str] = []
    for row in candidates:
        _, pre_match = _compare_to_flowstar(flow, row, "pre_step_box", tol=tol)
        endpoint_is_comparable, endpoint_match = _compare_to_flowstar(flow, row, "endpoint_box_before_center", tol=tol)
        reset_is_comparable, reset_match = _compare_to_flowstar(flow, row, "reset_box_after_center_scale", tol=tol)
        row["pre_step_matches_flowstar"] = pre_match
        row["endpoint_before_center_comparable"] = endpoint_is_comparable
        row["endpoint_before_center_matches_flowstar"] = endpoint_match
        row["reset_after_center_comparable"] = reset_is_comparable
        row["reset_after_center_matches_flowstar"] = reset_match
        pre_matches.append(pre_match)
        endpoint_comparable.append(endpoint_is_comparable)
        endpoint_matches.append(endpoint_match)
        reset_comparable.append(reset_is_comparable)
        reset_matches.append(reset_match)

    flow["pre_step_matches_flowstar"] = "true" if _box(flow, "pre_step_box") is not None else "unknown"
    flow["endpoint_before_center_comparable"] = "true" if _box(flow, "endpoint_box_before_center") is not None else "unknown"
    flow["endpoint_before_center_matches_flowstar"] = flow["endpoint_before_center_comparable"]
    flow["reset_after_center_comparable"] = "true" if _box(flow, "reset_box_after_center_scale") is not None else "unknown"
    flow["reset_after_center_matches_flowstar"] = flow["reset_after_center_comparable"]

    pre_all = _all_known_true(pre_matches)
    endpoint_all_comparable = _all_known_true(endpoint_comparable)
    endpoint_all_match = _all_known_true(endpoint_matches)
    reset_all_comparable = _all_known_true(reset_comparable)
    reset_all_match = _all_known_true(reset_matches)
    first_divergence = "none"
    if pre_all == "false":
        first_divergence = "pre_step_box"
    elif endpoint_all_comparable == "true" and endpoint_all_match == "false":
        first_divergence = "endpoint_box_before_center"
    elif reset_all_comparable == "true" and reset_all_match == "false":
        first_divergence = "reset_box_after_center_scale"
    elif endpoint_all_comparable == "unknown" or reset_all_comparable == "unknown":
        first_divergence = "unknown_missing_stage_fields"

    if pre_all == "false" or endpoint_all_match == "false" or reset_all_match == "false":
        residual_valid = "false"
    elif pre_all == "unknown" or endpoint_all_match == "unknown" or reset_all_match == "unknown":
        residual_valid = "unknown"
    else:
        residual_valid = "true"
    comparison = "same-stage-valid" if residual_valid == "true" else "noncausal/stage-misaligned"
    missing = _flowstar_missing_components(flow)

    summary = {
        "pre_step_boxes_equal": pre_all,
        "endpoint_before_center_comparable": endpoint_all_comparable,
        "endpoint_before_center_boxes_equal": endpoint_all_match,
        "reset_after_center_comparable": reset_all_comparable,
        "reset_after_center_boxes_equal": reset_all_match,
        "first_lifecycle_stage_divergence": first_divergence,
        "residual_comparison_stage_valid": residual_valid,
        "picard_residual_comparison": comparison,
        "flowstar_missing_residual_components": missing,
    }
    for row in rows:
        row["pre_step_boxes_equal"] = pre_all
        row["endpoint_before_center_boxes_equal"] = endpoint_all_match
        row["reset_after_center_boxes_equal"] = reset_all_match
        row["first_lifecycle_stage_divergence"] = first_divergence
        row["residual_comparison_stage_valid"] = residual_valid
        row["picard_residual_comparison"] = comparison
        row["flowstar_missing_residual_components"] = missing
        row["notes"] = _notes_for(row, summary)
    return summary


def _notes_for(row: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    notes: list[str] = []
    if row.get("source") != "flowstar":
        notes.append("generic center/scale fields ignored for same-local-box decisions")
    if summary.get("residual_comparison_stage_valid") != "true":
        notes.append("Picard residual endpoint comparison is noncausal/stage-misaligned")
    if summary.get("flowstar_missing_residual_components"):
        notes.append(f"Flow* missing residual components: {summary.get('flowstar_missing_residual_components')}")
    return "; ".join(notes)


def build_lifecycle_alignment(
    flowstar_rows: list[Mapping[str, Any]],
    noqueue_rows: list[Mapping[str, Any]],
    v2_rows: list[Mapping[str, Any]],
    *,
    t: float,
    h: float,
    tolerance: float = 1e-9,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = [
        ("flowstar", _find_attempt(flowstar_rows, t=t, h=h, tolerance=tolerance)),
        ("torch_noqueue", _find_attempt(noqueue_rows, t=t, h=h, tolerance=tolerance)),
        ("torch_v2", _find_attempt(v2_rows, t=t, h=h, tolerance=tolerance)),
    ]
    missing = [source for source, row in pairs if row is None]
    if missing:
        raise ValueError(f"missing same-t/h trace rows for: {', '.join(missing)}")
    rows = [_copy_stage_fields(source, row) for source, row in pairs if row is not None]
    summary = summarize_alignment(rows)
    return rows, summary


def _fmt_box(row: Mapping[str, Any], prefix: str) -> str:
    box = _box(row, prefix)
    if box is None:
        return "unknown"
    return f"x=[{box[0]:.17g}, {box[1]:.17g}], y=[{box[2]:.17g}, {box[3]:.17g}]"


def _report(out_dir: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any], *, t: float, h: float) -> str:
    lines = [
        "# Flow* Box Lifecycle Alignment Audit",
        "",
        "This audit checks stage-labeled boxes for the first same-t/h Picard residual mismatch. It does not change solver behavior and does not claim Flow* parity.",
        "",
        "## Scope",
        "",
        f"- t_before requested: `{t:.17g}`",
        f"- h_try: `{h:.17g}`",
        "- Input traces: `outputs/flowstar_step_trace_compare/*.csv`",
        "- Output ledger: `outputs/flowstar_box_lifecycle_alignment_audit/box_lifecycle_ledger.csv`",
        "",
        "## Answers",
        "",
        f"- Are Flow* and PyTorch pre_step boxes equal? `{summary['pre_step_boxes_equal']}`.",
        f"- Are endpoint-before-center boxes comparable? `{summary['endpoint_before_center_comparable']}`.",
        f"- Are reset-after-center boxes comparable? `{summary['reset_after_center_comparable']}`.",
        f"- Which lifecycle stage first differs? `{summary['first_lifecycle_stage_divergence']}`.",
        f"- Are residuals computed over the same stage? `{summary['residual_comparison_stage_valid']}`.",
        f"- Picard residual comparison: `{summary['picard_residual_comparison']}`.",
        f"- Flow* residual components still missing: `{summary['flowstar_missing_residual_components'] or 'none'}`.",
        "",
        "## Stage Ledger",
        "",
        "| source | status | pre_step box | endpoint-before-center box | reset-after-center box | residual stage valid |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {source} | {status} | {pre} | {endpoint} | {reset} | {valid} |".format(
                source=row.get("source", ""),
                status=row.get("status", ""),
                pre=_fmt_box(row, "pre_step_box"),
                endpoint=_fmt_box(row, "endpoint_box_before_center"),
                reset=_fmt_box(row, "reset_box_after_center_scale"),
                valid=row.get("residual_comparison_stage_valid", ""),
            )
        )
    if summary.get("residual_comparison_stage_valid") != "true":
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "The residual endpoint mismatch is not yet a valid same-local-box comparison. Stage-labeled boxes must align before the residual source can be attributed to no-remainder, raw-ctrunc, cutoff, target, or tolerance.",
            ]
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "box_lifecycle_report.md").write_text(text, encoding="utf-8")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--t", type=float, default=0.0)
    parser.add_argument("--h", type=float, default=0.025)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_dir = args.trace_dir.resolve()
    out_dir = args.out_dir.resolve()
    flowstar_rows = _read_rows(trace_dir / "flowstar_trace.csv")
    noqueue_rows = _read_rows(trace_dir / "torch_noqueue_trace.csv")
    v2_rows = _read_rows(trace_dir / "torch_v2_trace.csv")
    rows, summary = build_lifecycle_alignment(flowstar_rows, noqueue_rows, v2_rows, t=args.t, h=args.h)
    _write_rows(out_dir / "box_lifecycle_ledger.csv", rows)
    _report(out_dir, rows, summary, t=args.t, h=args.h)
    print(f"wrote box lifecycle alignment audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
