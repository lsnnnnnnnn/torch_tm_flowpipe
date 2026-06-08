#!/usr/bin/env python3
"""Audit endpoint-before-center source objects for the first Van der Pol mismatch.

This is a diagnostic-only report. It reads existing Flow*/PyTorch trace
artifacts and does not change solver behavior, rerun h10, add queue variants, or
claim Flow* parity.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "outputs" / "flowstar_step_trace_compare"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_endpoint_pre_center_source_audit"

LEDGER_FIELDS = [
    "source",
    "row_kind",
    "t_before",
    "h_try",
    "status",
    "endpoint_source_object",
    "domain_semantics",
    "includes_target_remainder",
    "includes_ordinary_remainder",
    "includes_symbolic_output_width",
    "includes_cutoff_poly_diff",
    "x_lo",
    "x_hi",
    "y_lo",
    "y_hi",
    "x_width",
    "y_width",
    "sum_width",
    "range_eval_method",
    "polynomial_degree_order",
    "dropped_terms_width",
    "remainder_width",
    "source_object_matches_flowstar",
    "semantic_comparison_valid",
    "y_hi_delta_vs_flowstar",
    "first_endpoint_divergence",
    "notes",
]

SEMANTIC_FIELDS = [
    "endpoint_source_object",
    "domain_semantics",
    "includes_target_remainder",
    "includes_ordinary_remainder",
    "includes_symbolic_output_width",
    "includes_cutoff_poly_diff",
    "range_eval_method",
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
    accepted = str(row.get("accepted", "")).strip().lower()
    rejected = str(row.get("rejected", "")).strip().lower()
    if accepted in {"true", "1", "yes", "validated"}:
        return "accepted"
    if rejected in {"true", "1", "yes", "failed"}:
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


def _bound(row: Mapping[str, Any], prefix: str, dim: str, side: str) -> Any:
    return _first_present(row, f"{prefix}_{dim}_{side}", f"{prefix}_{side}_{dim}")


def _width_from_bounds(lo: Any, hi: Any) -> float | None:
    f_lo = finite_float(lo)
    f_hi = finite_float(hi)
    if f_lo is None or f_hi is None:
        return None
    return f_hi - f_lo


def _box_width(row: Mapping[str, Any], prefix: str, dim: str) -> Any:
    width = _width_from_bounds(_bound(row, prefix, dim, "lo"), _bound(row, prefix, dim, "hi"))
    if width is not None:
        return width
    return _first_present(row, f"{prefix}_width_{dim}", f"{prefix}_{dim}_width")


def _main_endpoint_row(source: str, row: Mapping[str, Any]) -> dict[str, Any]:
    x_lo = _bound(row, "endpoint_box_before_center", "x", "lo")
    x_hi = _bound(row, "endpoint_box_before_center", "x", "hi")
    y_lo = _bound(row, "endpoint_box_before_center", "y", "lo")
    y_hi = _bound(row, "endpoint_box_before_center", "y", "hi")
    x_width = _width_from_bounds(x_lo, x_hi)
    y_width = _width_from_bounds(y_lo, y_hi)
    sum_width = (x_width + y_width) if x_width is not None and y_width is not None else _first_present(row, "endpoint_pre_center_width_sum")
    notes = [_first_present(row, "endpoint_before_center_notes")]
    if not _first_present(row, "endpoint_before_center_source_object"):
        notes.append("missing endpoint source object label")
    if x_lo == "" or x_hi == "" or y_lo == "" or y_hi == "":
        notes.append("missing endpoint_box_before_center endpoints")
    return {
        "source": source,
        "row_kind": "main_endpoint_box_before_center",
        "t_before": row.get("t_before", ""),
        "h_try": _first_present(row, "h_try", "h"),
        "status": _status(row),
        "endpoint_source_object": _first_present(row, "endpoint_before_center_source_object") or "unknown",
        "domain_semantics": _first_present(row, "endpoint_before_center_domain_semantics") or "unknown",
        "includes_target_remainder": _first_present(row, "endpoint_before_center_includes_target_remainder") or "unknown",
        "includes_ordinary_remainder": _first_present(row, "endpoint_before_center_includes_ordinary_remainder") or "unknown",
        "includes_symbolic_output_width": _first_present(row, "endpoint_before_center_includes_symbolic_output_width") or "unknown",
        "includes_cutoff_poly_diff": _first_present(row, "endpoint_before_center_includes_cutoff_poly_diff") or "unknown",
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "x_width": x_width if x_width is not None else _box_width(row, "endpoint_box_before_center", "x"),
        "y_width": y_width if y_width is not None else _box_width(row, "endpoint_box_before_center", "y"),
        "sum_width": sum_width,
        "range_eval_method": _first_present(row, "endpoint_before_center_range_eval_method") or "unknown",
        "polynomial_degree_order": _first_present(row, "endpoint_before_center_polynomial_order", "order") or "unknown",
        "dropped_terms_width": _first_present(row, "endpoint_before_center_dropped_terms_width_sum", "cutoff_polynomial_difference_width_sum") or "unknown",
        "remainder_width": _first_present(row, "endpoint_before_center_remainder_width_sum", "picard_ctrunc_normal_residual_width_sum") or "unknown",
        "notes": "; ".join(note for note in notes if note),
    }


def _variant_row(source: str, row: Mapping[str, Any], *, kind: str, object_label: str, prefix: str, notes: str) -> dict[str, Any]:
    x_lo = _bound(row, prefix, "x", "lo")
    x_hi = _bound(row, prefix, "x", "hi")
    y_lo = _bound(row, prefix, "y", "lo")
    y_hi = _bound(row, prefix, "y", "hi")
    x_width = _width_from_bounds(x_lo, x_hi)
    y_width = _width_from_bounds(y_lo, y_hi)
    if x_width is None:
        x_width = finite_float(_first_present(row, f"{prefix}_width_x", f"{prefix}_x_width"))
    if y_width is None:
        y_width = finite_float(_first_present(row, f"{prefix}_width_y", f"{prefix}_y_width"))
    sum_width = (x_width + y_width) if x_width is not None and y_width is not None else _first_present(row, f"{prefix}_width_sum")
    return {
        "source": source,
        "row_kind": kind,
        "t_before": row.get("t_before", ""),
        "h_try": _first_present(row, "h_try", "h"),
        "status": _status(row),
        "endpoint_source_object": object_label,
        "domain_semantics": "diagnostic_variant",
        "includes_target_remainder": "unknown",
        "includes_ordinary_remainder": "unknown",
        "includes_symbolic_output_width": "unknown",
        "includes_cutoff_poly_diff": "unknown",
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "x_width": x_width,
        "y_width": y_width,
        "sum_width": sum_width,
        "range_eval_method": "trace-derived width/endpoints only",
        "polynomial_degree_order": "unknown",
        "dropped_terms_width": "unknown",
        "remainder_width": "unknown",
        "notes": notes,
    }


def _target_expanded_variant(source: str, row: Mapping[str, Any]) -> dict[str, Any] | None:
    vals = {
        key: finite_float(_first_present(row, key))
        for key in (
            "endpoint_box_before_center_x_lo",
            "endpoint_box_before_center_x_hi",
            "endpoint_box_before_center_y_lo",
            "endpoint_box_before_center_y_hi",
            "target_remainder_x_lo",
            "target_remainder_x_hi",
            "target_remainder_y_lo",
            "target_remainder_y_hi",
        )
    }
    if any(value is None for value in vals.values()):
        return None
    assert all(value is not None for value in vals.values())
    x_lo = vals["endpoint_box_before_center_x_lo"] + vals["target_remainder_x_lo"]
    x_hi = vals["endpoint_box_before_center_x_hi"] + vals["target_remainder_x_hi"]
    y_lo = vals["endpoint_box_before_center_y_lo"] + vals["target_remainder_y_lo"]
    y_hi = vals["endpoint_box_before_center_y_hi"] + vals["target_remainder_y_hi"]
    return {
        "source": source,
        "row_kind": "diagnostic_endpoint_plus_target_remainder",
        "t_before": row.get("t_before", ""),
        "h_try": _first_present(row, "h_try", "h"),
        "status": _status(row),
        "endpoint_source_object": "endpoint_box_before_center + target_remainder diagnostic",
        "domain_semantics": "diagnostic_variant_not_solver_state",
        "includes_target_remainder": "true",
        "includes_ordinary_remainder": "unknown",
        "includes_symbolic_output_width": "unknown",
        "includes_cutoff_poly_diff": _first_present(row, "endpoint_before_center_includes_cutoff_poly_diff") or "unknown",
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "x_width": x_hi - x_lo,
        "y_width": y_hi - y_lo,
        "sum_width": (x_hi - x_lo) + (y_hi - y_lo),
        "range_eval_method": "diagnostic Minkowski sum of endpoint field and target remainder interval",
        "polynomial_degree_order": "unknown",
        "dropped_terms_width": "unknown",
        "remainder_width": "unknown",
        "notes": "diagnostic-only variant; not used by solver",
    }


def _add_variants(rows: list[dict[str, Any]], source: str, trace_row: Mapping[str, Any]) -> None:
    rows.append(_variant_row(
        source,
        trace_row,
        kind="diagnostic_reset_after_center_scale",
        object_label="reset_box_after_center_scale",
        prefix="reset_box_after_center_scale",
        notes="diagnostic-only reset candidate comparison",
    ))
    if source.startswith("torch"):
        rows.append(_variant_row(
            source,
            trace_row,
            kind="diagnostic_right_map_range_width_only",
            object_label="right_map_range_width_only",
            prefix="right_map_range",
            notes="endpoints are not emitted for this trace channel",
        ))
        rows.append(_variant_row(
            source,
            trace_row,
            kind="diagnostic_output_range_width_only",
            object_label="output_range_width_only",
            prefix="output_range",
            notes="endpoints are not emitted for this trace channel",
        ))
        rows.append(_variant_row(
            source,
            trace_row,
            kind="diagnostic_ordinary_residual_width_only",
            object_label="picard_no_remainder_residual_width_only",
            prefix="picard_no_remainder_residual",
            notes="ordinary residual endpoints may be present, but this is a residual component, not an endpoint range",
        ))
        target_variant = _target_expanded_variant(source, trace_row)
        if target_variant is not None:
            rows.append(target_variant)


def _semantic_match(flow: Mapping[str, Any], row: Mapping[str, Any]) -> bool | None:
    values: list[bool] = []
    for field in SEMANTIC_FIELDS:
        flow_value = flow.get(field, "")
        row_value = row.get(field, "")
        if flow_value in (None, "", "unknown") or row_value in (None, "", "unknown"):
            return None
        values.append(str(flow_value) == str(row_value))
    return all(values)


def summarize_endpoint_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    main = [row for row in rows if row.get("row_kind") == "main_endpoint_box_before_center"]
    flow = next(row for row in main if row.get("source") == "flowstar")
    candidates = [row for row in main if str(row.get("source", "")).startswith("torch")]
    matches: list[bool | None] = []
    y_deltas: list[float | None] = []
    for row in candidates:
        semantic_match = _semantic_match(flow, row)
        matches.append(semantic_match)
        row["source_object_matches_flowstar"] = "unknown" if semantic_match is None else semantic_match
        row["semantic_comparison_valid"] = semantic_match is True
        flow_y_hi = finite_float(flow.get("y_hi"))
        row_y_hi = finite_float(row.get("y_hi"))
        delta = None if flow_y_hi is None or row_y_hi is None else flow_y_hi - row_y_hi
        row["y_hi_delta_vs_flowstar"] = delta
        y_deltas.append(delta)
    flow["source_object_matches_flowstar"] = "true"
    flow["semantic_comparison_valid"] = "reference"
    flow["y_hi_delta_vs_flowstar"] = 0.0

    if any(match is False for match in matches):
        comparison_valid = "false"
        first_divergence = "source_object_semantics"
    elif any(match is None for match in matches):
        comparison_valid = "unknown"
        first_divergence = "unknown_missing_source_fields"
    else:
        comparison_valid = "true"
        finite_deltas = [abs(delta) for delta in y_deltas if delta is not None]
        first_divergence = "y_hi" if finite_deltas and max(finite_deltas) > 0.0 else "none"

    for row in rows:
        row.setdefault("source_object_matches_flowstar", "")
        row.setdefault("semantic_comparison_valid", comparison_valid if row.get("row_kind") == "main_endpoint_box_before_center" else "diagnostic_variant")
        row.setdefault("y_hi_delta_vs_flowstar", "")
        row["first_endpoint_divergence"] = first_divergence
    return {
        "semantic_comparison_valid": comparison_valid,
        "first_endpoint_divergence": first_divergence,
        "flowstar_endpoint_source_object": flow.get("endpoint_source_object", "unknown"),
        "torch_endpoint_source_objects": ";".join(str(row.get("endpoint_source_object", "unknown")) for row in candidates),
        "flowstar_box": _fmt_box(flow),
        "torch_noqueue_box": _fmt_box(next(row for row in candidates if row.get("source") == "torch_noqueue")),
        "torch_v2_box": _fmt_box(next(row for row in candidates if row.get("source") == "torch_v2")),
        "max_y_hi_delta_vs_flowstar": max((abs(delta) for delta in y_deltas if delta is not None), default=None),
    }


def build_endpoint_source_ledger(
    flowstar_rows: list[Mapping[str, Any]],
    noqueue_rows: list[Mapping[str, Any]],
    v2_rows: list[Mapping[str, Any]],
    *,
    t: float,
    h: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = [
        ("flowstar", _find_attempt(flowstar_rows, t=t, h=h)),
        ("torch_noqueue", _find_attempt(noqueue_rows, t=t, h=h)),
        ("torch_v2", _find_attempt(v2_rows, t=t, h=h)),
    ]
    missing = [source for source, row in pairs if row is None]
    if missing:
        raise ValueError(f"missing same-t/h trace rows for: {', '.join(missing)}")
    rows: list[dict[str, Any]] = []
    for source, trace_row in pairs:
        assert trace_row is not None
        rows.append(_main_endpoint_row(source, trace_row))
    for source, trace_row in pairs:
        assert trace_row is not None
        _add_variants(rows, source, trace_row)
    summary = summarize_endpoint_sources(rows)
    return rows, summary


def _fmt_interval(row: Mapping[str, Any], dim: str) -> str:
    return f"[{_format(row.get(f'{dim}_lo'))}, {_format(row.get(f'{dim}_hi'))}]"


def _fmt_box(row: Mapping[str, Any]) -> str:
    return f"x={_fmt_interval(row, 'x')}, y={_fmt_interval(row, 'y')}"


def _report(out_dir: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any], *, t: float, h: float) -> str:
    comparison_valid = summary.get("semantic_comparison_valid", "unknown")
    if comparison_valid == "true":
        causal_text = "The endpoint-before-center rows are a same-source-object comparison."
        component_text = "The first numeric divergence is `y_hi`; inspect dropped_terms_width and remainder_width for the component split."
        under_account_text = "PyTorch under-accounting is possible only if the same-source component widths differ after source labels match."
        compare_instead = "The current endpoint_box_before_center fields may be compared directly."
    else:
        causal_text = "The endpoint-before-center comparison is noncausal because the source object labels or domain semantics do not match."
        component_text = "No term/component attribution is valid yet; the first identified source is semantic stage mismatch, not a polynomial term."
        under_account_text = "This trace does not prove PyTorch under-accounts endpoint-before-center width; it currently labels a different object/stage."
        compare_instead = "Compare Flow* tmvTmp over the full step with a PyTorch validation candidate evaluated over the full tau domain, or compare both systems after tau=h substitution."

    lines = [
        "# Flow* Endpoint-Before-Center Source Audit",
        "",
        "This is diagnostic-only and makes no solver change. It does not rerun h10, add symbolic queue variants, or claim Flow* parity.",
        "",
        "## Scope",
        "",
        f"- t_before requested: `{t:.17g}`",
        f"- h_try: `{h:.17g}`",
        "- Input traces: `outputs/flowstar_step_trace_compare/*.csv`",
        "- Output ledger: `outputs/flowstar_endpoint_pre_center_source_audit/endpoint_pre_center_ledger.csv`",
        "",
        "## Answers",
        "",
        f"- Are Flow* and PyTorch endpoint-before-center fields semantically the same object? `{comparison_valid}`.",
        f"- Causality: {causal_text}",
        f"- What should be compared instead? {compare_instead}",
        f"- If same object, what explains Flow* wider y_hi? {component_text}",
        f"- Does PyTorch currently under-account endpoint-before-center width? {under_account_text}",
        f"- Is Flow* endpoint-before-center perhaps a different stage? `yes`: Flow* source is `{summary.get('flowstar_endpoint_source_object', 'unknown')}`, while PyTorch source object(s) are `{summary.get('torch_endpoint_source_objects', 'unknown')}`.",
        "- Next minimal diagnostic: emit PyTorch validation-candidate full-step endpoint bounds and Flow* tau=h substituted endpoint bounds under the same source labels.",
        "",
        "## Main Endpoint Rows",
        "",
        "| source | status | source object | domain semantics | endpoint box | y_hi delta vs Flow* | semantic valid |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in [r for r in rows if r.get("row_kind") == "main_endpoint_box_before_center"]:
        lines.append(
            "| {source} | {status} | {obj} | {domain} | {box} | {delta} | {valid} |".format(
                source=row.get("source", ""),
                status=row.get("status", ""),
                obj=row.get("endpoint_source_object", ""),
                domain=row.get("domain_semantics", ""),
                box=_fmt_box(row),
                delta=_format(row.get("y_hi_delta_vs_flowstar", "")),
                valid=_format(row.get("semantic_comparison_valid", "")),
            )
        )
    lines.extend(
        [
            "",
            "## Diagnostic Variants",
            "",
            "Variant rows in the CSV are diagnostic-only. Blank endpoint columns mean the trace did not expose those endpoints; they are not treated as zero.",
        ]
    )
    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "endpoint_pre_center_report.md").write_text(text, encoding="utf-8")
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
    rows, summary = build_endpoint_source_ledger(flowstar_rows, noqueue_rows, v2_rows, t=args.t, h=args.h)
    _write_rows(out_dir / "endpoint_pre_center_ledger.csv", rows)
    _report(out_dir, rows, summary, t=args.t, h=args.h)
    print(f"wrote endpoint-before-center source audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
