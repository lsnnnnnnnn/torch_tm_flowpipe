#!/usr/bin/env python3
"""Audit Flow* internal intermediate_ranges at the first Van der Pol divergence.

This diagnostic consumes the same short Flow*/PyTorch step traces used by the
raw ctrunc residual audit. It is passive: it does not change solver behavior,
rerun h10, add queue variants, commit Flow* source, or claim Flow* parity.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "outputs" / "flowstar_step_trace_compare"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_internal_intermediate_ranges_audit"

GAP_MATCH_TOL = 1e-10

FLOWSTAR_SOURCE_REFERENCES = [
    "/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModelVec<DATA_TYPE>::Picard_ctrunc_normal(... Expression ..., intermediate_ranges, Global_Setting)",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModelVec<DATA_TYPE>::Picard_ctrunc_normal_remainder(... Expression ..., intermediate_ranges, Global_Setting)",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModel<DATA_TYPE>::mul_insert_ctrunc_normal(... Interval &tm1, Interval &intTrunc ...)",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::HornerForm<DATA_TYPE>::insert_ctrunc_normal(... intermediate_ranges ...)",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/expression.h::AST_Node<DATA_TYPE>::evaluate(... intermediate_ranges ...)",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/expression.h::AST_Node<DATA_TYPE>::evaluate_remainder(... iterator over intermediate_ranges ...)",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp::advance/result constructions around Picard_ctrunc_normal",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.cpp::not present in this checkout; relevant template implementations are in TaylorModel.h",
    "experiments/flowstar_probe/flowstar_vdp_step_trace_probe.cpp::traced_advance_adaptive_symbolic",
]

LEDGER_FIELDS = [
    "source",
    "t_before",
    "h_try",
    "status",
    "raw_ctrunc_residual_x_lo",
    "raw_ctrunc_residual_x_hi",
    "raw_ctrunc_residual_y_lo",
    "raw_ctrunc_residual_y_hi",
    "expression_evaluate_remainder_x_lo",
    "expression_evaluate_remainder_x_hi",
    "expression_evaluate_remainder_y_lo",
    "expression_evaluate_remainder_y_hi",
    "horner_insert_ctrunc_remainder_x_lo",
    "horner_insert_ctrunc_remainder_x_hi",
    "horner_insert_ctrunc_remainder_y_lo",
    "horner_insert_ctrunc_remainder_y_hi",
    "int_trunc_dropped_terms_x_lo",
    "int_trunc_dropped_terms_x_hi",
    "int_trunc_dropped_terms_y_lo",
    "int_trunc_dropped_terms_y_hi",
    "int_trunc2_dropped_terms_x_lo",
    "int_trunc2_dropped_terms_x_hi",
    "int_trunc2_dropped_terms_y_lo",
    "int_trunc2_dropped_terms_y_hi",
    "mul_ctrunc_normal_remainder_x_lo",
    "mul_ctrunc_normal_remainder_x_hi",
    "mul_ctrunc_normal_remainder_y_lo",
    "mul_ctrunc_normal_remainder_y_hi",
    "accumulated_remainder_before_x0_add_x_lo",
    "accumulated_remainder_before_x0_add_x_hi",
    "accumulated_remainder_before_x0_add_y_lo",
    "accumulated_remainder_before_x0_add_y_hi",
    "accumulated_remainder_after_x0_add_x_lo",
    "accumulated_remainder_after_x0_add_x_hi",
    "accumulated_remainder_after_x0_add_y_lo",
    "accumulated_remainder_after_x0_add_y_hi",
    "intermediate_ranges_entry_count",
    "source_path",
    "raw_y_hi_delta_vs_flowstar",
    "component_matching_raw_y_hi_delta",
    "missing_fields",
    "notes",
]

COMPONENT_SOURCES = {
    "raw_ctrunc_residual": ["raw_ctrunc_residual", "picard_ctrunc_raw_residual"],
    "expression_evaluate_remainder": ["expression_evaluate_remainder", "raw_remainder_integration_remainder"],
    "horner_insert_ctrunc_remainder": ["horner_insert_ctrunc_remainder"],
    "int_trunc_dropped_terms": ["int_trunc_dropped_terms", "raw_remainder_dropped_terms_range"],
    "int_trunc2_dropped_terms": ["int_trunc2_dropped_terms", "raw_remainder_dropped_terms_range"],
    "mul_ctrunc_normal_remainder": ["mul_ctrunc_normal_remainder", "raw_remainder_multiplication_remainder"],
    "accumulated_remainder_before_x0_add": [
        "accumulated_remainder_before_x0_add",
        "raw_remainder_after_cutoff",
        "raw_ctrunc_residual",
    ],
    "accumulated_remainder_after_x0_add": ["accumulated_remainder_after_x0_add", "raw_ctrunc_residual"],
}

EXPLANATORY_COMPONENTS = [
    ("expression_evaluate_remainder", "expression_evaluate_remainder_gap"),
    ("horner_insert_ctrunc_remainder", "horner_insert_ctrunc_remainder_gap"),
    ("int_trunc_dropped_terms", "int_trunc_dropped_terms_gap"),
    ("int_trunc2_dropped_terms", "int_trunc2_dropped_terms_gap"),
    ("mul_ctrunc_normal_remainder", "mul_ctrunc_normal_remainder_gap"),
    ("accumulated_remainder_before_x0_add", "accumulated_remainder_before_x0_add_gap"),
]

CAUSES = {
    "expression_evaluate_remainder_gap": "expression evaluate_remainder / evaluated RHS remainder",
    "horner_insert_ctrunc_remainder_gap": "Horner insert ctrunc remainder",
    "int_trunc_dropped_terms_gap": "first ctrunc_normal dropped terms",
    "int_trunc2_dropped_terms_gap": "second ctrunc_normal dropped terms",
    "mul_ctrunc_normal_remainder_gap": "mul_ctrunc_normal remainder contribution",
    "accumulated_remainder_before_x0_add_gap": "accumulation before x0 add",
    "hidden_internal_intermediate_ranges_gap": "hidden Flow* intermediate_ranges internals",
    "same_raw_y_hi": "no raw y_hi gap",
}

NEXT_OBJECT = (
    "Flow* AST_Node::evaluate NODE_VAR ctrunc_normal side effect: expose the interval added by "
    "result.ctrunc_normal(step_exp_table, order) for each variable node before AST_Node::evaluate_remainder replay. "
    "For non-polynomial ODEs, also expose any Taylor *_taylor_only_remainder iterator consumption."
)


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


def _bounds(row: Mapping[str, Any], prefix: str, dim: str) -> tuple[Any, Any]:
    return (
        _first_present(row, f"{prefix}_{dim}_lo", f"{prefix}_lo_{dim}"),
        _first_present(row, f"{prefix}_{dim}_hi", f"{prefix}_hi_{dim}"),
    )


def _has_any_bounds(row: Mapping[str, Any], prefix: str) -> bool:
    for dim in ("x", "y"):
        lo, hi = _bounds(row, prefix, dim)
        if lo not in (None, "") and hi not in (None, ""):
            return True
    return False


def _put_blank_bounds(out: dict[str, Any], prefix: str) -> None:
    for dim in ("x", "y"):
        out[f"{prefix}_{dim}_lo"] = ""
        out[f"{prefix}_{dim}_hi"] = ""


def _copy_first_bounds(out: dict[str, Any], target_prefix: str, row: Mapping[str, Any], sources: Iterable[str], missing: list[str]) -> str | None:
    for source_prefix in sources:
        if not _has_any_bounds(row, source_prefix):
            continue
        for dim in ("x", "y"):
            lo, hi = _bounds(row, source_prefix, dim)
            out[f"{target_prefix}_{dim}_lo"] = lo
            out[f"{target_prefix}_{dim}_hi"] = hi
        return source_prefix
    _put_blank_bounds(out, target_prefix)
    for dim in ("x", "y"):
        for side in ("lo", "hi"):
            missing.append(f"{out['source']}.{target_prefix}_{dim}_{side}")
    return None


def build_ledger_row(source: str, row: Mapping[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    notes: list[str] = []
    out: dict[str, Any] = {
        "source": source,
        "t_before": row.get("t_before", ""),
        "h_try": _first_present(row, "h_try", "h_forced", "h"),
        "status": _status(row),
        "intermediate_ranges_entry_count": _first_present(row, "flowstar_internal_intermediate_ranges_entry_count"),
        "source_path": _first_present(row, "flowstar_internal_intermediate_ranges_source_path"),
        "raw_y_hi_delta_vs_flowstar": "",
        "component_matching_raw_y_hi_delta": "",
    }
    for target, sources in COMPONENT_SOURCES.items():
        source_prefix = _copy_first_bounds(out, target, row, sources, missing)
        if source_prefix and source_prefix != target:
            notes.append(f"{target} from {source_prefix}")
    if note := _first_present(row, "flowstar_internal_intermediate_ranges_notes"):
        notes.append(str(note))
    if missing:
        notes.append("blank internal columns mean unknown, not zero")
    out["missing_fields"] = "; ".join(sorted(dict.fromkeys(missing)))
    out["notes"] = "; ".join(notes)
    return out


def _row_for(rows: Iterable[Mapping[str, Any]], source: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("source") == source:
            return row
    raise KeyError(source)


def _primary_torch(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    try:
        return _row_for(rows, "torch_noqueue")
    except KeyError:
        for row in rows:
            if str(row.get("source", "")).startswith("torch"):
                return row
    raise KeyError("torch")


def _value(row: Mapping[str, Any], prefix: str, dim: str, side: str) -> float | None:
    return finite_float(row.get(f"{prefix}_{dim}_{side}"))


def _y_hi_delta(row: Mapping[str, Any], flow: Mapping[str, Any], prefix: str) -> float | None:
    value = _value(row, prefix, "y", "hi")
    reference = _value(flow, prefix, "y", "hi")
    if value is None or reference is None:
        return None
    return value - reference


def _all_known_components_match(row: Mapping[str, Any], flow: Mapping[str, Any]) -> bool:
    saw_known = False
    for prefix, _label in EXPLANATORY_COMPONENTS:
        delta = _y_hi_delta(row, flow, prefix)
        if delta is None:
            continue
        saw_known = True
        if abs(delta) > GAP_MATCH_TOL:
            return False
    return saw_known


def _matching_component(row: Mapping[str, Any], flow: Mapping[str, Any], raw_delta: float | None) -> str:
    if row.get("source") == "flowstar":
        return "reference"
    if raw_delta is None:
        return "unknown_missing_raw_ctrunc_residual"
    if abs(raw_delta) <= GAP_MATCH_TOL:
        return "same_raw_y_hi"
    for prefix, label in EXPLANATORY_COMPONENTS:
        delta = _y_hi_delta(row, flow, prefix)
        if delta is not None and abs(delta - raw_delta) <= GAP_MATCH_TOL:
            return label
    if _all_known_components_match(row, flow):
        return "hidden_internal_intermediate_ranges_gap"
    return "unknown_missing_internal_intermediate_ranges_partition"


def build_ledger(
    flowstar_rows: list[Mapping[str, Any]],
    noqueue_rows: list[Mapping[str, Any]],
    v2_rows: list[Mapping[str, Any]],
    *,
    t: float,
    h: float,
) -> list[dict[str, Any]]:
    pairs = [
        ("flowstar", _find_attempt(flowstar_rows, t=t, h=h)),
        ("torch_noqueue", _find_attempt(noqueue_rows, t=t, h=h)),
        ("torch_v2", _find_attempt(v2_rows, t=t, h=h)),
    ]
    missing = [source for source, row in pairs if row is None]
    if missing:
        raise ValueError(f"missing same-t/h trace rows for: {', '.join(missing)}")
    ledger = [build_ledger_row(source, row) for source, row in pairs if row is not None]
    flow = _row_for(ledger, "flowstar")
    for row in ledger:
        delta = _y_hi_delta(row, flow, "raw_ctrunc_residual")
        row["raw_y_hi_delta_vs_flowstar"] = delta if delta is not None else ""
        row["component_matching_raw_y_hi_delta"] = _matching_component(row, flow, delta)
    return ledger


def _component_status(rows: list[Mapping[str, Any]], prefix: str) -> str:
    flow = _row_for(rows, "flowstar")
    saw_known = False
    for row in rows:
        if row.get("source") == "flowstar":
            continue
        delta = _y_hi_delta(row, flow, prefix)
        if delta is None:
            continue
        saw_known = True
        if abs(delta) > GAP_MATCH_TOL:
            return "differs"
    return "same" if saw_known else "unknown"


def summarize(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    flow = _row_for(ledger, "flowstar")
    torch = _primary_torch(ledger)
    component = str(torch.get("component_matching_raw_y_hi_delta") or "unknown")
    missing_flowstar = [field for field in str(flow.get("missing_fields", "")).split("; ") if field]
    still_unknown = component in {"hidden_internal_intermediate_ranges_gap", "unknown_missing_internal_intermediate_ranges_partition"}
    return {
        "first_flowstar_internal_object_explaining_y_hi_delta": component,
        "cause_classification": CAUSES.get(component, "unknown"),
        "raw_y_hi_delta_torch_minus_flowstar": finite_float(torch.get("raw_y_hi_delta_vs_flowstar")),
        "expression_evaluate_remainder_component": _component_status(ledger, "expression_evaluate_remainder"),
        "horner_insert_ctrunc_remainder_component": _component_status(ledger, "horner_insert_ctrunc_remainder"),
        "int_trunc_dropped_terms_component": _component_status(ledger, "int_trunc_dropped_terms"),
        "int_trunc2_dropped_terms_component": _component_status(ledger, "int_trunc2_dropped_terms"),
        "mul_ctrunc_normal_remainder_component": _component_status(ledger, "mul_ctrunc_normal_remainder"),
        "accumulated_remainder_before_x0_add_component": _component_status(ledger, "accumulated_remainder_before_x0_add"),
        "accumulated_remainder_after_x0_add_component": _component_status(ledger, "accumulated_remainder_after_x0_add"),
        "exact_flowstar_internal_object_still_inaccessible": NEXT_OBJECT if still_unknown else "none_for_this_classification",
        "missing_flowstar_fields": "; ".join(missing_flowstar) if missing_flowstar else "none",
    }


def _fmt_interval(row: Mapping[str, Any], prefix: str, dim: str) -> str:
    return f"[{_format(row.get(f'{prefix}_{dim}_lo'))}, {_format(row.get(f'{prefix}_{dim}_hi'))}]"


def _report(out_dir: Path, ledger: list[dict[str, Any]], summary: Mapping[str, Any], *, t: float, h: float) -> str:
    lines = [
        "# Flow* Internal Intermediate Ranges Audit",
        "",
        "This is diagnostic-only. It does not change solver behavior, rerun h10, add queue variants, commit Flow* source, or claim Flow* parity.",
        "",
        "## Scope",
        "",
        f"- t_before requested: `{t:.17g}`",
        f"- h_try: `{h:.17g}`",
        "- Input traces: `outputs/flowstar_step_trace_compare/*.csv`",
        "- Output ledger: `outputs/flowstar_internal_intermediate_ranges_audit/internal_intermediate_ranges_ledger.csv`",
        "",
        "## Flow* Source Inspection",
        "",
    ]
    lines.extend(f"- `{ref}`" for ref in FLOWSTAR_SOURCE_REFERENCES)
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Which Flow* internal object first explains raw y_hi gap: `{summary.get('first_flowstar_internal_object_explaining_y_hi_delta', 'unknown')}`.",
            f"- Raw y_hi delta torch-Flow*: `{_format(summary.get('raw_y_hi_delta_torch_minus_flowstar'))}`.",
            f"- Cause classification: `{summary.get('cause_classification', 'unknown')}`.",
            f"- Dropped terms evidence: intTrunc `{summary.get('int_trunc_dropped_terms_component', 'unknown')}`, intTrunc2 `{summary.get('int_trunc2_dropped_terms_component', 'unknown')}`.",
            f"- Multiplication remainder evidence: `{summary.get('mul_ctrunc_normal_remainder_component', 'unknown')}`.",
            f"- Expression evaluate_remainder evidence: `{summary.get('expression_evaluate_remainder_component', 'unknown')}`.",
            f"- Accumulation before x0 add evidence: `{summary.get('accumulated_remainder_before_x0_add_component', 'unknown')}`.",
            f"- Accumulation after x0 add evidence: `{summary.get('accumulated_remainder_after_x0_add_component', 'unknown')}`.",
            f"- If still unknown, inaccessible Flow* object: {summary.get('exact_flowstar_internal_object_still_inaccessible', 'unknown')}.",
            f"- Missing Flow* fields: {summary.get('missing_flowstar_fields', 'unknown')}.",
            "",
            "## Candidate Rows",
            "",
            "| source | status | raw y | expression y | intTrunc y | intTrunc2 y | mul y | before x0 y | after x0 y | component |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in ledger:
        lines.append(
            "| {source} | {status} | {raw} | {expr} | {trunc} | {trunc2} | {mul} | {before} | {after} | {component} |".format(
                source=row.get("source", ""),
                status=row.get("status", ""),
                raw=_fmt_interval(row, "raw_ctrunc_residual", "y"),
                expr=_fmt_interval(row, "expression_evaluate_remainder", "y"),
                trunc=_fmt_interval(row, "int_trunc_dropped_terms", "y"),
                trunc2=_fmt_interval(row, "int_trunc2_dropped_terms", "y"),
                mul=_fmt_interval(row, "mul_ctrunc_normal_remainder", "y"),
                before=_fmt_interval(row, "accumulated_remainder_before_x0_add", "y"),
                after=_fmt_interval(row, "accumulated_remainder_after_x0_add", "y"),
                component=row.get("component_matching_raw_y_hi_delta", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Blank internal fields mean unknown or not-applicable, not zero.",
            "- The Flow* probe uses the `Expression<Real>` ODE overload; `horner_insert_ctrunc_remainder` is blank unless a Horner call path is exposed.",
            "- PyTorch comparison columns use same-named fields when present and fall back to the existing raw remainder partition diagnostics where noted in the ledger.",
        ]
    )
    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "internal_intermediate_ranges_report.md").write_text(text, encoding="utf-8")
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
    ledger = build_ledger(flowstar_rows, noqueue_rows, v2_rows, t=args.t, h=args.h)
    summary = summarize(ledger)
    _write_rows(out_dir / "internal_intermediate_ranges_ledger.csv", ledger)
    _report(out_dir, ledger, summary, t=args.t, h=args.h)
    print(f"wrote internal intermediate ranges audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
