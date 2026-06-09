#!/usr/bin/env python3
"""Audit raw Picard_ctrunc_normal residual construction at first divergence.

This diagnostic reads the same-source Flow*/PyTorch step traces at the first
same-t/h validation candidate divergence, t ~= 0 and h = 0.025. It does not
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
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_ctrunc_residual_audit"

GAP_MATCH_TOL = 1e-10

LEDGER_FIELDS = [
    "source",
    "t_before",
    "h_try",
    "status",
    "raw_ctrunc_residual_x_lo",
    "raw_ctrunc_residual_x_hi",
    "raw_ctrunc_residual_y_lo",
    "raw_ctrunc_residual_y_hi",
    "raw_ctrunc_polynomial_range_x_lo",
    "raw_ctrunc_polynomial_range_x_hi",
    "raw_ctrunc_polynomial_range_y_lo",
    "raw_ctrunc_polynomial_range_y_hi",
    "raw_ctrunc_remainder_x_lo",
    "raw_ctrunc_remainder_x_hi",
    "raw_ctrunc_remainder_y_lo",
    "raw_ctrunc_remainder_y_hi",
    "picard_no_remainder_range_x_lo",
    "picard_no_remainder_range_x_hi",
    "picard_no_remainder_range_y_lo",
    "picard_no_remainder_range_y_hi",
    "picard_no_remainder_remainder_x_lo",
    "picard_no_remainder_remainder_x_hi",
    "picard_no_remainder_remainder_y_lo",
    "picard_no_remainder_remainder_y_hi",
    "target_remainder_x_lo",
    "target_remainder_x_hi",
    "target_remainder_y_lo",
    "target_remainder_y_hi",
    "residual_domain_semantics",
    "includes_target_remainder",
    "includes_ordinary_remainder",
    "includes_cutoff_poly_diff",
    "raw_y_hi_delta_vs_flowstar",
    "component_matching_raw_y_hi_delta",
    "missing_fields",
    "notes",
]

COMPONENT_PREFIXES = [
    "raw_ctrunc_residual",
    "raw_ctrunc_polynomial_range",
    "raw_ctrunc_remainder",
    "picard_no_remainder_range",
    "picard_no_remainder_remainder",
    "target_remainder",
]

FLOWSTAR_AUX_FIELDS = [
    "ordinary_remainder_x_lo",
    "ordinary_remainder_x_hi",
    "ordinary_remainder_y_lo",
    "ordinary_remainder_y_hi",
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


def _copy_first_bounds(
    out: dict[str, Any],
    target_prefix: str,
    row: Mapping[str, Any],
    source_prefixes: Iterable[str],
    missing: list[str],
) -> str | None:
    for source_prefix in source_prefixes:
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


def _raw_semantic_field(row: Mapping[str, Any], field: str) -> str:
    return str(_first_present(row, f"raw_ctrunc_residual_{field}", field)).strip()


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
    _copy_first_bounds(out, "raw_ctrunc_polynomial_range", row, ["raw_ctrunc_polynomial_range", "polynomial_range"], missing)
    _copy_first_bounds(out, "raw_ctrunc_remainder", row, ["raw_ctrunc_remainder", "raw_ctrunc_residual", "picard_ctrunc_raw_residual"], missing)
    _copy_first_bounds(out, "picard_no_remainder_range", row, ["picard_no_remainder_range", "picard_no_remainder_polynomial_range"], missing)
    _copy_first_bounds(out, "picard_no_remainder_remainder", row, ["picard_no_remainder_remainder"], missing)
    _copy_first_bounds(out, "target_remainder", row, ["target_remainder_before_ctrunc", "target_remainder"], missing)
    if source == "flowstar":
        for field in FLOWSTAR_AUX_FIELDS:
            if row.get(field) in (None, ""):
                missing.append(f"{source}.{field}")
    out["residual_domain_semantics"] = _raw_semantic_field(row, "domain_semantics")
    out["includes_target_remainder"] = _raw_semantic_field(row, "includes_target_remainder")
    out["includes_ordinary_remainder"] = _raw_semantic_field(row, "includes_ordinary_remainder")
    out["includes_cutoff_poly_diff"] = _raw_semantic_field(row, "includes_cutoff_poly_diff")
    out["raw_y_hi_delta_vs_flowstar"] = ""
    out["component_matching_raw_y_hi_delta"] = ""
    out["missing_fields"] = "; ".join(sorted(dict.fromkeys(missing)))
    if raw_source:
        notes.append(f"raw_ctrunc_residual from {raw_source}")
    if reason := _first_present(row, "ordinary_remainder_missing_reason"):
        notes.append(str(reason))
    if raw_notes := _first_present(row, "raw_ctrunc_residual_notes"):
        notes.append(str(raw_notes))
    if missing:
        notes.append("blank component endpoint columns mean unknown, not zero")
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


def _semantic_mismatch(row: Mapping[str, Any], flow: Mapping[str, Any]) -> str | None:
    checks = [
        ("residual_domain_semantics", "noncausal_domain_mismatch"),
        ("includes_target_remainder", "noncausal_target_remainder_inclusion_mismatch"),
        ("includes_ordinary_remainder", "noncausal_ordinary_remainder_inclusion_mismatch"),
        ("includes_cutoff_poly_diff", "noncausal_cutoff_poly_diff_inclusion_mismatch"),
    ]
    for field, label in checks:
        same = _same_text(row.get(field), flow.get(field))
        if same is False:
            return label
    return None


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
    for prefix, label in (
        ("raw_ctrunc_polynomial_range", "raw_polynomial_range_gap"),
        ("raw_ctrunc_remainder", "raw_remainder_gap"),
        ("picard_no_remainder_range", "picard_no_remainder_range_gap"),
        ("picard_no_remainder_remainder", "picard_no_remainder_remainder_gap"),
        ("target_remainder", "target_remainder_gap"),
    ):
        delta = _y_hi_delta(row, flow, prefix)
        if delta is not None and abs(delta - raw_delta) <= GAP_MATCH_TOL:
            return label
    return "unknown_unmatched_raw_y_hi_gap"


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
            return "unknown"
        saw_known = True
        if abs(delta) > GAP_MATCH_TOL:
            return "differs"
    return "same" if saw_known else "unknown"


def summarize(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    flow = _row_for(ledger, "flowstar")
    torch = _primary_torch(ledger)
    raw_delta = finite_float(torch.get("raw_y_hi_delta_vs_flowstar"))
    semantic_mismatch = _semantic_mismatch(torch, flow)
    missing_flowstar = [field for field in str(flow.get("missing_fields", "")).split("; ") if field]
    component = str(torch.get("component_matching_raw_y_hi_delta") or "unknown")
    semantically_same = "false" if semantic_mismatch else "true"
    return {
        "raw_objects_semantically_same": semantically_same,
        "semantic_mismatch": semantic_mismatch or "none",
        "primary_raw_y_hi_delta_torch_minus_flowstar": raw_delta,
        "first_component_explaining_raw_y_hi_gap": component,
        "raw_polynomial_range_component": _component_status(ledger, "raw_ctrunc_polynomial_range"),
        "raw_remainder_component": _component_status(ledger, "raw_ctrunc_remainder"),
        "picard_no_remainder_range_component": _component_status(ledger, "picard_no_remainder_range"),
        "picard_no_remainder_remainder_component": _component_status(ledger, "picard_no_remainder_remainder"),
        "target_remainder_component": _component_status(ledger, "target_remainder"),
        "flowstar_puts_extra_component_into_raw_residual": "unknown" if semantic_mismatch else "false_at_exposed_flags",
        "pytorch_puts_component_elsewhere_that_flowstar_puts_in_raw_residual": "unknown_needs_picard_ctrunc_internal_partition",
        "missing_flowstar_fields": "; ".join(missing_flowstar) if missing_flowstar else "none",
        "component_attribution_complete": "false" if component in {"raw_remainder_gap", "unknown_unmatched_raw_y_hi_gap"} else "unknown",
    }


def _fmt_interval(row: Mapping[str, Any], prefix: str, dim: str) -> str:
    return f"[{_format(row.get(f'{prefix}_{dim}_lo'))}, {_format(row.get(f'{prefix}_{dim}_hi'))}]"


def _report(out_dir: Path, ledger: list[dict[str, Any]], summary: Mapping[str, Any], *, t: float, h: float) -> str:
    flow = _row_for(ledger, "flowstar")
    torch = _primary_torch(ledger)
    lines = [
        "# Flow* Raw Ctrunc Residual Audit",
        "",
        "This is diagnostic-only. It does not change solver behavior, rerun h10, add queue variants, or claim Flow* parity.",
        "",
        "## Scope",
        "",
        f"- t_before requested: `{t:.17g}`",
        f"- h_try: `{h:.17g}`",
        "- Input traces: `outputs/flowstar_step_trace_compare/*.csv`",
        "- Output ledger: `outputs/flowstar_raw_ctrunc_residual_audit/raw_ctrunc_residual_ledger.csv`",
        "",
        "## Answers",
        "",
        f"- Are Flow* and PyTorch raw_ctrunc_residual semantically the same object: `{summary.get('raw_objects_semantically_same', 'unknown')}`; semantic mismatch: `{summary.get('semantic_mismatch', 'unknown')}`.",
        f"- Raw y_hi delta torch-Flow*: `{_format(summary.get('primary_raw_y_hi_delta_torch_minus_flowstar'))}`.",
        f"- First exposed component explaining raw y_hi gap: `{summary.get('first_component_explaining_raw_y_hi_gap', 'unknown')}`.",
        f"- Raw polynomial range component: `{summary.get('raw_polynomial_range_component', 'unknown')}`.",
        f"- Raw returned remainder component: `{summary.get('raw_remainder_component', 'unknown')}`.",
        f"- No-remainder Picard range component: `{summary.get('picard_no_remainder_range_component', 'unknown')}`.",
        f"- No-remainder Picard remainder component: `{summary.get('picard_no_remainder_remainder_component', 'unknown')}`.",
        f"- Target remainder component: `{summary.get('target_remainder_component', 'unknown')}`.",
        f"- Does Flow* put a component into raw_ctrunc_residual that PyTorch does not: `{summary.get('flowstar_puts_extra_component_into_raw_residual', 'unknown')}`.",
        f"- Does PyTorch put a component elsewhere that Flow* puts into residual: `{summary.get('pytorch_puts_component_elsewhere_that_flowstar_puts_in_raw_residual', 'unknown')}`.",
        f"- Component attribution complete: `{summary.get('component_attribution_complete', 'unknown')}`.",
        f"- Exact Flow* fields still missing: {summary.get('missing_flowstar_fields', 'unknown')}.",
        "",
        "## Candidate Rows",
        "",
        "| source | status | raw residual y | raw polynomial y | raw remainder y | no-remainder range y | target y | component | domain | flags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ledger:
        flags = (
            f"target={row.get('includes_target_remainder', '')}, "
            f"ordinary={row.get('includes_ordinary_remainder', '')}, "
            f"cutoff={row.get('includes_cutoff_poly_diff', '')}"
        )
        lines.append(
            "| {source} | {status} | {raw} | {poly} | {rem} | {picard} | {target} | {component} | {domain} | {flags} |".format(
                source=row.get("source", ""),
                status=row.get("status", ""),
                raw=_fmt_interval(row, "raw_ctrunc_residual", "y"),
                poly=_fmt_interval(row, "raw_ctrunc_polynomial_range", "y"),
                rem=_fmt_interval(row, "raw_ctrunc_remainder", "y"),
                picard=_fmt_interval(row, "picard_no_remainder_range", "y"),
                target=_fmt_interval(row, "target_remainder", "y"),
                component=row.get("component_matching_raw_y_hi_delta", ""),
                domain=row.get("residual_domain_semantics", "") or "unknown",
                flags=flags,
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Flow* raw_ctrunc_residual y_hi is `{_format(flow.get('raw_ctrunc_residual_y_hi'))}`; PyTorch raw_ctrunc_residual y_hi is `{_format(torch.get('raw_ctrunc_residual_y_hi'))}`.",
            "- `raw_remainder_gap` means the exposed returned remainder carries the raw y_hi gap; it does not by itself close the internal Flow* Picard_ctrunc_normal attribution.",
            "- Blank component endpoint columns mean unknown, not zero.",
        ]
    )
    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_ctrunc_residual_report.md").write_text(text, encoding="utf-8")
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
    _write_rows(out_dir / "raw_ctrunc_residual_ledger.csv", ledger)
    _report(out_dir, ledger, summary, t=args.t, h=args.h)
    print(f"wrote raw ctrunc residual audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
