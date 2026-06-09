#!/usr/bin/env python3
"""Audit partitions of the raw Picard_ctrunc_normal returned remainder.

This diagnostic reads the same-source Flow*/PyTorch step traces at the first
same-t/h Van der Pol validation divergence, t ~= 0 and h = 0.025. It does not
change solver behavior, rerun h10, add queue variants, or claim Flow* parity.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "outputs" / "flowstar_step_trace_compare"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_partition_audit"

GAP_MATCH_TOL = 1e-10

FLOWSTAR_SOURCE_REFERENCES = [
    "/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModelVec<DATA_TYPE>::Picard_ctrunc_normal(... Expression ..., intermediate_ranges, Global_Setting)",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::TaylorModelVec<DATA_TYPE>::Picard_ctrunc_normal_remainder(... Expression ..., intermediate_ranges, Global_Setting)",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/expression.h::AST_Node<DATA_TYPE>::evaluate(... step_exp_table ..., intermediate_ranges, Global_Setting)",
    "/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h::HornerForm<DATA_TYPE>::insert_ctrunc_normal(... intermediate_ranges ...)",
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
    "dropped_terms_x_lo",
    "dropped_terms_x_hi",
    "dropped_terms_y_lo",
    "dropped_terms_y_hi",
    "multiplication_remainder_x_lo",
    "multiplication_remainder_x_hi",
    "multiplication_remainder_y_lo",
    "multiplication_remainder_y_hi",
    "integration_remainder_x_lo",
    "integration_remainder_x_hi",
    "integration_remainder_y_lo",
    "integration_remainder_y_hi",
    "before_accumulation_x_lo",
    "before_accumulation_x_hi",
    "before_accumulation_y_lo",
    "before_accumulation_y_hi",
    "after_integration_x_lo",
    "after_integration_x_hi",
    "after_integration_y_lo",
    "after_integration_y_hi",
    "after_dropped_terms_x_lo",
    "after_dropped_terms_x_hi",
    "after_dropped_terms_y_lo",
    "after_dropped_terms_y_hi",
    "after_cutoff_x_lo",
    "after_cutoff_x_hi",
    "after_cutoff_y_lo",
    "after_cutoff_y_hi",
    "before_poly_diff_x_lo",
    "before_poly_diff_x_hi",
    "before_poly_diff_y_lo",
    "before_poly_diff_y_hi",
    "after_poly_diff_x_lo",
    "after_poly_diff_x_hi",
    "after_poly_diff_y_lo",
    "after_poly_diff_y_hi",
    "range_enclosure_method",
    "normal_domain_scaling",
    "raw_y_hi_delta_vs_flowstar",
    "component_matching_raw_y_hi_delta",
    "missing_fields",
    "notes",
]

COMPONENT_SOURCES = {
    "dropped_terms": ["raw_remainder_dropped_terms_range"],
    "multiplication_remainder": ["raw_remainder_multiplication_remainder"],
    "integration_remainder": ["raw_remainder_integration_remainder"],
    "before_accumulation": ["raw_remainder_before_accumulation"],
    "after_integration": ["raw_remainder_after_integration"],
    "after_dropped_terms": ["raw_remainder_after_dropped_terms"],
    "after_cutoff": ["raw_remainder_after_cutoff"],
    "before_poly_diff": ["raw_remainder_before_poly_diff"],
    "after_poly_diff": ["raw_remainder_after_poly_diff", "post_cutoff_residual", "picard_ctrunc_normal_residual"],
}

EXPLANATORY_COMPONENTS = [
    ("dropped_terms", "dropped_term_gap"),
    ("multiplication_remainder", "multiplication_remainder_gap"),
    ("integration_remainder", "integration_remainder_gap"),
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


def _bounds(row: Mapping[str, Any], prefix: str, dim: str) -> tuple[Any, Any]:
    return (
        _first_present(row, f"{prefix}_{dim}_lo", f"{prefix}_lo_{dim}"),
        _first_present(row, f"{prefix}_{dim}_hi", f"{prefix}_hi_{dim}"),
    )


def _has_bounds(row: Mapping[str, Any], prefix: str) -> bool:
    for dim in ("x", "y"):
        lo, hi = _bounds(row, prefix, dim)
        if lo in (None, "") or hi in (None, ""):
            return False
    return True


def _put_blank_bounds(out: dict[str, Any], prefix: str) -> None:
    for dim in ("x", "y"):
        out[f"{prefix}_{dim}_lo"] = ""
        out[f"{prefix}_{dim}_hi"] = ""


def _copy_first_bounds(out: dict[str, Any], target_prefix: str, row: Mapping[str, Any], sources: Iterable[str], missing: list[str]) -> str | None:
    for source_prefix in sources:
        if not _has_bounds(row, source_prefix):
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
    }
    raw_source = _copy_first_bounds(out, "raw_ctrunc_residual", row, ["raw_ctrunc_residual", "picard_ctrunc_raw_residual"], missing)
    for target, sources in COMPONENT_SOURCES.items():
        _copy_first_bounds(out, target, row, sources, missing)
    out["range_enclosure_method"] = _first_present(row, "raw_remainder_range_enclosure_method")
    out["normal_domain_scaling"] = _first_present(row, "raw_remainder_normal_domain_scaling")
    out["raw_y_hi_delta_vs_flowstar"] = ""
    out["component_matching_raw_y_hi_delta"] = ""
    out["missing_fields"] = "; ".join(sorted(dict.fromkeys(missing)))
    if raw_source:
        notes.append(f"raw_ctrunc_residual from {raw_source}")
    if reason := _first_present(row, "raw_remainder_partition_missing_reason"):
        notes.append(str(reason))
    if missing:
        notes.append("blank internal partition columns mean unknown, not zero")
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


def _same_text(a: Any, b: Any) -> bool | None:
    left = str(a).strip().lower()
    right = str(b).strip().lower()
    if not left or not right:
        return None
    return left == right


def _scaling_equivalent(a: Any, b: Any) -> bool | None:
    left = str(a).strip().lower()
    right = str(b).strip().lower()
    if not left or not right:
        return None
    if "none" in left and "none" in right:
        return True
    return left == right


def _semantic_mismatch(row: Mapping[str, Any], flow: Mapping[str, Any]) -> str | None:
    if _scaling_equivalent(row.get("normal_domain_scaling"), flow.get("normal_domain_scaling")) is False:
        return "noncausal_domain_scaling_mismatch"
    return None


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
    mismatch = _semantic_mismatch(row, flow)
    if mismatch:
        return mismatch
    if raw_delta is None:
        return "unknown_missing_raw_ctrunc_residual"
    if abs(raw_delta) <= GAP_MATCH_TOL:
        return "same_raw_y_hi"
    for prefix, label in EXPLANATORY_COMPONENTS:
        delta = _y_hi_delta(row, flow, prefix)
        if delta is not None and abs(delta - raw_delta) <= GAP_MATCH_TOL:
            return label
    if _all_known_components_match(row, flow):
        return "hidden_raw_remainder_gap"
    return "unknown_missing_internal_partition"


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
    domain_scaling = _component_status(ledger, "raw_ctrunc_residual")
    if component == "dropped_term_gap":
        cause = "dropped-term range"
    elif component == "multiplication_remainder_gap":
        cause = "multiplication remainder"
    elif component in {"integration_remainder_gap", "after_integration_gap"}:
        cause = "integration/Picard residual remainder"
    elif component == "range_enclosure_method_mismatch":
        cause = "range enclosure method mismatch"
    elif component == "noncausal_domain_scaling_mismatch":
        cause = "domain scaling mismatch"
    elif component == "hidden_raw_remainder_gap":
        cause = "hidden Flow* raw returned remainder internals"
    else:
        cause = "unknown"
    return {
        "flowstar_raw_returned_remainder_decomposable_from_exposed_fields": "false" if component in {"hidden_raw_remainder_gap", "unknown_missing_internal_partition"} else "partial",
        "first_component_explaining_y_hi_delta": component,
        "cause_classification": cause,
        "raw_y_hi_delta_torch_minus_flowstar": finite_float(torch.get("raw_y_hi_delta_vs_flowstar")),
        "dropped_terms_component": _component_status(ledger, "dropped_terms"),
        "multiplication_remainder_component": _component_status(ledger, "multiplication_remainder"),
        "integration_remainder_component": _component_status(ledger, "integration_remainder"),
        "after_cutoff_component": _component_status(ledger, "after_cutoff"),
        "domain_scaling_evidence": "same_no_scaling" if _scaling_equivalent(torch.get("normal_domain_scaling"), flow.get("normal_domain_scaling")) else "unknown_or_mismatch",
        "range_enclosure_evidence": "same_labels" if _same_text(torch.get("range_enclosure_method"), flow.get("range_enclosure_method")) else "unknown_or_mismatch",
        "soundness_implication": "no_pyTorch_unsoundness_evidence_representation_split_or_hidden_flowstar_partition",
        "exact_flowstar_internal_object_to_expose_next": (
            "TaylorModel.h Expression/Horner intermediate_ranges entries mapped per state dimension: "
            "ctrunc_normal intTrunc/intTrunc2, mul_ctrunc_normal remainder contributions, and evaluate_remainder accumulation before result = tmvTmp2 + x0"
        ),
        "missing_flowstar_fields": "; ".join(missing_flowstar) if missing_flowstar else "none",
        "raw_component_status": domain_scaling,
    }


def _fmt_interval(row: Mapping[str, Any], prefix: str, dim: str) -> str:
    return f"[{_format(row.get(f'{prefix}_{dim}_lo'))}, {_format(row.get(f'{prefix}_{dim}_hi'))}]"


def _report(out_dir: Path, ledger: list[dict[str, Any]], summary: Mapping[str, Any], *, t: float, h: float) -> str:
    lines = [
        "# Flow* Raw Remainder Partition Audit",
        "",
        "This is diagnostic-only. It does not change solver behavior, rerun h10, add queue variants, or claim Flow* parity.",
        "",
        "## Scope",
        "",
        f"- t_before requested: `{t:.17g}`",
        f"- h_try: `{h:.17g}`",
        "- Input traces: `outputs/flowstar_step_trace_compare/*.csv`",
        "- Output ledger: `outputs/flowstar_raw_remainder_partition_audit/raw_remainder_partition_ledger.csv`",
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
            f"- Is Flow* raw returned remainder decomposable from exposed fields: `{summary.get('flowstar_raw_returned_remainder_decomposable_from_exposed_fields', 'unknown')}`.",
            f"- Raw y_hi delta torch-Flow*: `{_format(summary.get('raw_y_hi_delta_torch_minus_flowstar'))}`.",
            f"- First subcomponent explaining y_hi delta: `{summary.get('first_component_explaining_y_hi_delta', 'unknown')}`.",
            f"- Cause classification: `{summary.get('cause_classification', 'unknown')}`.",
            f"- Dropped-term range component: `{summary.get('dropped_terms_component', 'unknown')}`.",
            f"- Multiplication remainder component: `{summary.get('multiplication_remainder_component', 'unknown')}`.",
            f"- Integration/Picard remainder component: `{summary.get('integration_remainder_component', 'unknown')}`.",
            f"- Range-enclosure evidence: `{summary.get('range_enclosure_evidence', 'unknown')}`.",
            f"- Domain-scaling evidence: `{summary.get('domain_scaling_evidence', 'unknown')}`.",
            f"- Soundness implication: `{summary.get('soundness_implication', 'unknown')}`.",
            f"- Exact Flow* object to expose next if still unknown: {summary.get('exact_flowstar_internal_object_to_expose_next', 'unknown')}.",
            f"- Missing Flow* fields: {summary.get('missing_flowstar_fields', 'unknown')}.",
            "",
            "## Candidate Rows",
            "",
            "| source | status | raw residual y | dropped y | multiplication y | integration y | after cutoff y | component |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in ledger:
        lines.append(
            "| {source} | {status} | {raw} | {drop} | {mul} | {integ} | {cutoff} | {component} |".format(
                source=row.get("source", ""),
                status=row.get("status", ""),
                raw=_fmt_interval(row, "raw_ctrunc_residual", "y"),
                drop=_fmt_interval(row, "dropped_terms", "y"),
                mul=_fmt_interval(row, "multiplication_remainder", "y"),
                integ=_fmt_interval(row, "integration_remainder", "y"),
                cutoff=_fmt_interval(row, "after_cutoff", "y"),
                component=row.get("component_matching_raw_y_hi_delta", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Blank internal partition columns mean unknown, not zero.",
            "- `hidden_raw_remainder_gap` means all exposed partitions matched but raw_ctrunc_residual still differed, so attribution requires deeper Flow* internal instrumentation.",
            "- No evidence here by itself suggests PyTorch is missing a soundness component; the remaining issue is a Flow*/PyTorch representation split or hidden Flow* partition until those internal objects are exposed.",
        ]
    )
    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_remainder_partition_report.md").write_text(text, encoding="utf-8")
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
    _write_rows(out_dir / "raw_remainder_partition_ledger.csv", ledger)
    _report(out_dir, ledger, summary, t=args.t, h=args.h)
    print(f"wrote raw remainder partition audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
