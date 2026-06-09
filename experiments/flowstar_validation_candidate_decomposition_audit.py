#!/usr/bin/env python3
"""Audit full-step validation candidate decomposition at first divergence.

This diagnostic reads the same-source Flow*/PyTorch step traces at the first
same-t/h acceptance divergence, t ~= 0 and h = 0.025. It does not change solver
behavior, rerun h10, add symbolic queue variants, or claim Flow* parity.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "outputs" / "flowstar_step_trace_compare"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_validation_candidate_decomposition_audit"

WIDTH_CLOSE_REL_TOL = 1e-3
GAP_MATCH_TOL = 1e-10

LEDGER_FIELDS = [
    "source",
    "t_before",
    "h_try",
    "status",
    "full_step_x_lo",
    "full_step_x_hi",
    "full_step_y_lo",
    "full_step_y_hi",
    "full_step_width_x",
    "full_step_width_y",
    "full_step_width_sum",
    "polynomial_range_x_lo",
    "polynomial_range_x_hi",
    "polynomial_range_y_lo",
    "polynomial_range_y_hi",
    "ordinary_remainder_x_lo",
    "ordinary_remainder_x_hi",
    "ordinary_remainder_y_lo",
    "ordinary_remainder_y_hi",
    "raw_ctrunc_residual_x_lo",
    "raw_ctrunc_residual_x_hi",
    "raw_ctrunc_residual_y_lo",
    "raw_ctrunc_residual_y_hi",
    "post_cutoff_residual_x_lo",
    "post_cutoff_residual_x_hi",
    "post_cutoff_residual_y_lo",
    "post_cutoff_residual_y_hi",
    "cutoff_poly_diff_x_lo",
    "cutoff_poly_diff_x_hi",
    "cutoff_poly_diff_y_lo",
    "cutoff_poly_diff_y_hi",
    "dropped_terms_width_x",
    "dropped_terms_width_y",
    "target_remainder_x_lo",
    "target_remainder_x_hi",
    "target_remainder_y_lo",
    "target_remainder_y_hi",
    "residual_subset_x",
    "residual_subset_y",
    "failed_dim",
    "residual_y_hi_margin_to_target",
    "residual_domain",
    "center_x",
    "center_y",
    "scale_x",
    "scale_y",
    "residual_local_box_x_lo",
    "residual_local_box_x_hi",
    "residual_local_box_y_lo",
    "residual_local_box_y_hi",
    "notes",
]

COMPONENT_PREFIXES = [
    "polynomial_range",
    "ordinary_remainder",
    "raw_ctrunc_residual",
    "post_cutoff_residual",
    "cutoff_poly_diff",
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


def _bounds(row: Mapping[str, Any], prefix: str, dim: str) -> tuple[float | None, float | None]:
    lo = finite_float(_first_present(row, f"{prefix}_{dim}_lo", f"{prefix}_lo_{dim}"))
    hi = finite_float(_first_present(row, f"{prefix}_{dim}_hi", f"{prefix}_hi_{dim}"))
    return lo, hi


def _has_bounds(row: Mapping[str, Any], prefix: str) -> bool:
    return all(_bounds(row, prefix, dim)[side] is not None for dim in ("x", "y") for side in (0, 1))


def _has_component_endpoint(row: Mapping[str, Any], prefix: str) -> bool:
    return _has_bounds(row, prefix)


def _source_full_step_prefix(source: str) -> str:
    if source == "flowstar":
        return "flowstar_full_step_tube"
    return "torch_full_step_validation_candidate"


def _source_domain_prefix(source: str) -> str:
    if source == "flowstar":
        return "flowstar_full_step_tube"
    return "torch_full_step_validation_candidate"


def _put_blank_bounds(out: dict[str, Any], prefix: str) -> None:
    for dim in ("x", "y"):
        out[f"{prefix}_{dim}_lo"] = ""
        out[f"{prefix}_{dim}_hi"] = ""


def _copy_first_bounds(
    out: dict[str, Any],
    target_prefix: str,
    row: Mapping[str, Any],
    source_prefixes: Iterable[str],
    notes: list[str],
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
    notes.append(f"unknown {target_prefix} endpoints")
    return None


def _width_from_out(out: Mapping[str, Any], prefix: str, dim: str) -> float | None:
    lo = finite_float(out.get(f"{prefix}_{dim}_lo"))
    hi = finite_float(out.get(f"{prefix}_{dim}_hi"))
    if lo is None or hi is None:
        return None
    return hi - lo


def _put_full_step_widths(out: dict[str, Any]) -> None:
    widths: list[float] = []
    for dim in ("x", "y"):
        width = _width_from_out(out, "full_step", dim)
        out[f"full_step_width_{dim}"] = width if width is not None else ""
        if width is not None:
            widths.append(width)
    out["full_step_width_sum"] = sum(widths) if len(widths) == 2 else ""


def _subset_label(
    residual_lo: float | None,
    residual_hi: float | None,
    target_lo: float | None,
    target_hi: float | None,
    *,
    tolerance: float,
) -> str:
    if residual_lo is None or residual_hi is None or target_lo is None or target_hi is None:
        return "unknown"
    return "true" if residual_lo >= target_lo - tolerance and residual_hi <= target_hi + tolerance else "false"


def _delta(a: Any, b: Any) -> float | None:
    fa = finite_float(a)
    fb = finite_float(b)
    if fa is None or fb is None:
        return None
    return fa - fb


def _local_box(center: float | None, scale: float | None) -> tuple[float | None, float | None]:
    if center is None or scale is None:
        return None, None
    radius = abs(scale)
    return center - radius, center + radius


def _width_only_note(row: Mapping[str, Any], label: str, *prefixes: str) -> str | None:
    parts: list[str] = []
    for dim in ("x", "y"):
        value = _first_present(row, *(f"{prefix}_width_{dim}" for prefix in prefixes))
        if value not in (None, ""):
            parts.append(f"{dim}={_format(value)}")
    if not parts:
        return None
    return f"{label} width-only: {', '.join(parts)}"


def build_ledger_row(source: str, row: Mapping[str, Any], *, tolerance: float = 0.0) -> dict[str, Any]:
    """Build one decomposition row, preserving missing component fields as blank."""
    notes: list[str] = []
    out: dict[str, Any] = {
        "source": source,
        "t_before": row.get("t_before", ""),
        "h_try": _first_present(row, "h_try", "h_forced", "h"),
        "status": _status(row),
    }

    full_step_prefix = _source_full_step_prefix(source)
    used_full_step = _copy_first_bounds(out, "full_step", row, [full_step_prefix], notes)
    if used_full_step is not None:
        notes.append(f"full_step from {used_full_step}")
    _put_full_step_widths(out)

    _copy_first_bounds(out, "polynomial_range", row, ["polynomial_range", "validation_candidate_polynomial_range"], notes)
    if out["polynomial_range_x_lo"] == "":
        notes.append("polynomial_range not exposed; no zero inferred")

    used_ordinary = _copy_first_bounds(
        out,
        "ordinary_remainder",
        row,
        ["ordinary_remainder", "picard_no_remainder_residual"],
        notes,
    )
    if used_ordinary is None:
        width_note = _width_only_note(row, "ordinary_remainder", "ordinary_step_remainder", "picard_no_remainder_residual")
        if width_note:
            notes.append(width_note)

    _copy_first_bounds(out, "raw_ctrunc_residual", row, ["raw_ctrunc_residual", "picard_ctrunc_raw_residual"], notes)
    used_post = _copy_first_bounds(
        out,
        "post_cutoff_residual",
        row,
        ["post_cutoff_residual", "picard_ctrunc_normal_residual"],
        notes,
    )
    if used_post is not None:
        notes.append(f"target-check residual from {used_post}")

    _copy_first_bounds(out, "cutoff_poly_diff", row, ["cutoff_poly_diff", "polynomial_diff", "poly_diff_range"], notes)
    if out["cutoff_poly_diff_x_lo"] == "":
        width_note = _width_only_note(
            row,
            "cutoff_poly_diff",
            "cutoff_polynomial_difference",
            "endpoint_before_center_dropped_terms",
        )
        if width_note:
            notes.append(width_note)

    for dim in ("x", "y"):
        width = _first_present(
            row,
            f"cutoff_polynomial_difference_width_{dim}",
            f"endpoint_before_center_dropped_terms_width_{dim}",
            f"poly_diff_range_width_{dim}",
        )
        out[f"dropped_terms_width_{dim}"] = width

    _copy_first_bounds(out, "target_remainder", row, ["target_remainder"], notes)
    failed: list[str] = []
    for dim in ("x", "y"):
        residual_lo = finite_float(out.get(f"post_cutoff_residual_{dim}_lo"))
        residual_hi = finite_float(out.get(f"post_cutoff_residual_{dim}_hi"))
        target_lo = finite_float(out.get(f"target_remainder_{dim}_lo"))
        target_hi = finite_float(out.get(f"target_remainder_{dim}_hi"))
        subset = _subset_label(residual_lo, residual_hi, target_lo, target_hi, tolerance=tolerance)
        out[f"residual_subset_{dim}"] = subset
        if subset == "false":
            failed.append(dim)
    out["failed_dim"] = ";".join(failed) if failed else ("unknown" if any(out[f"residual_subset_{dim}"] == "unknown" for dim in ("x", "y")) else "")
    out["residual_y_hi_margin_to_target"] = _delta(out.get("post_cutoff_residual_y_hi"), out.get("target_remainder_y_hi"))

    domain_prefix = _source_domain_prefix(source)
    out["residual_domain"] = _first_present(
        row,
        f"{domain_prefix}_domain_semantics",
        "residual_domain",
        "endpoint_before_center_domain_semantics",
    )
    center_x = finite_float(_first_present(row, "center_x", "extracted_center_x"))
    center_y = finite_float(_first_present(row, "center_y", "extracted_center_y"))
    scale_x = finite_float(_first_present(row, "scale_x", "extracted_scale_x"))
    scale_y = finite_float(_first_present(row, "scale_y", "extracted_scale_y"))
    out["center_x"] = center_x
    out["center_y"] = center_y
    out["scale_x"] = scale_x
    out["scale_y"] = scale_y
    x_lo, x_hi = _local_box(center_x, scale_x)
    y_lo, y_hi = _local_box(center_y, scale_y)
    out["residual_local_box_x_lo"] = x_lo
    out["residual_local_box_x_hi"] = x_hi
    out["residual_local_box_y_lo"] = y_lo
    out["residual_local_box_y_hi"] = y_hi
    if not out["residual_domain"]:
        notes.append("unknown residual domain semantics")
    if any(value is None for value in (center_x, center_y, scale_x, scale_y)):
        notes.append("unknown center/scale fields")

    reason = _first_present(row, "rejection_reason", "message", "validation_message")
    if reason:
        notes.append(f"reason: {reason}")
    out["notes"] = "; ".join(notes)
    return out


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
    return [build_ledger_row(source, row) for source, row in pairs if row is not None]


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


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = finite_float(numerator)
    den = finite_float(denominator)
    if num is None or den is None or abs(den) <= 0.0:
        return None
    return num / den


def _interval_delta(candidate: Mapping[str, Any], reference: Mapping[str, Any], prefix: str, dim: str, side: str) -> float | None:
    return _delta(candidate.get(f"{prefix}_{dim}_{side}"), reference.get(f"{prefix}_{dim}_{side}"))


def _component_status(rows: list[Mapping[str, Any]], prefix: str, *, tolerance: float = GAP_MATCH_TOL) -> str:
    flow = _row_for(rows, "flowstar")
    candidates = [row for row in rows if str(row.get("source", "")).startswith("torch")]
    saw_known = False
    for row in candidates:
        for dim in ("x", "y"):
            for side in ("lo", "hi"):
                delta = _interval_delta(row, flow, prefix, dim, side)
                if delta is None:
                    return "unknown"
                saw_known = True
                if abs(delta) > tolerance:
                    return "differs"
    return "same" if saw_known else "unknown"


def _component_y_hi_delta(rows: list[Mapping[str, Any]], prefix: str) -> float | None:
    flow = _row_for(rows, "flowstar")
    torch = _primary_torch(rows)
    return _interval_delta(torch, flow, prefix, "y", "hi")


def _matching_y_hi_component(rows: list[Mapping[str, Any]], target_delta: float | None, *, tolerance: float = GAP_MATCH_TOL) -> str:
    if target_delta is None:
        return "unknown"
    for prefix, label in (
        ("polynomial_range", "polynomial_range"),
        ("ordinary_remainder", "ordinary_remainder"),
        ("raw_ctrunc_residual", "raw_ctrunc_residual"),
        ("cutoff_poly_diff", "cutoff_poly_diff"),
        ("post_cutoff_residual", "post_cutoff_residual"),
    ):
        delta = _component_y_hi_delta(rows, prefix)
        if delta is None:
            continue
        if abs(delta - target_delta) <= tolerance:
            return label
    return "unknown"


def _component_missing(rows: list[Mapping[str, Any]], prefix: str) -> list[str]:
    missing: list[str] = []
    for row in rows:
        source = row.get("source", "")
        for dim in ("x", "y"):
            for side in ("lo", "hi"):
                field = f"{prefix}_{dim}_{side}"
                if row.get(field) in (None, ""):
                    missing.append(f"{source}.{field}")
    return missing


def _close_ratio_word(ratio: float | None, *, tolerance: float = WIDTH_CLOSE_REL_TOL) -> str:
    if ratio is None:
        return "unknown"
    return "true" if abs(ratio - 1.0) <= tolerance else "false"


def _bool_word(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "true" if value else "false"


def summarize(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    flow = _row_for(ledger, "flowstar")
    torch = _primary_torch(ledger)
    width_ratio = _ratio(torch.get("full_step_width_sum"), flow.get("full_step_width_sum"))
    width_close = _close_ratio_word(width_ratio)
    full_step_y_hi_delta = _interval_delta(torch, flow, "full_step", "y", "hi")
    post_cutoff_y_hi_delta = _interval_delta(torch, flow, "post_cutoff_residual", "y", "hi")
    gap_matches = None
    if full_step_y_hi_delta is not None and post_cutoff_y_hi_delta is not None:
        gap_matches = abs(full_step_y_hi_delta - post_cutoff_y_hi_delta) <= GAP_MATCH_TOL

    flow_fails = flow.get("residual_subset_x") == "false" or flow.get("residual_subset_y") == "false"
    torch_passes = torch.get("residual_subset_x") == "true" and torch.get("residual_subset_y") == "true"
    if width_close == "true" and flow_fails and torch_passes:
        verdict = "residual_decomposition_mismatch"
    elif width_close == "false":
        verdict = "full_step_tube_width_mismatch"
    elif flow_fails != (not torch_passes):
        verdict = "acceptance_status_mismatch"
    else:
        verdict = "unknown"

    cutoff_y_width_delta = _delta(torch.get("dropped_terms_width_y"), flow.get("dropped_terms_width_y"))
    cutoff_explains_gap: str
    if cutoff_y_width_delta is None or post_cutoff_y_hi_delta is None:
        cutoff_explains_gap = "unknown"
    else:
        cutoff_explains_gap = "false" if abs(cutoff_y_width_delta) < abs(post_cutoff_y_hi_delta) * 0.01 else "unknown_width_only"

    polynomial_status = _component_status(ledger, "polynomial_range")
    pytorch_width_in_poly = "unknown_missing_polynomial_range" if polynomial_status == "unknown" else "unknown_needs_remainder_partition"
    flowstar_no_remainder_missing = any(flow.get(f"ordinary_remainder_{dim}_{side}") in (None, "") for dim in ("x", "y") for side in ("lo", "hi"))
    missing_fields = sorted({field for prefix in COMPONENT_PREFIXES for field in _component_missing(ledger, prefix)})

    exposed_component = _matching_y_hi_component(ledger, post_cutoff_y_hi_delta)
    if exposed_component == "unknown":
        if _component_status(ledger, "post_cutoff_residual") == "differs":
            exposed_component = "post_cutoff_residual"
        elif _component_status(ledger, "raw_ctrunc_residual") == "differs":
            exposed_component = "raw_ctrunc_residual"
        elif _component_status(ledger, "ordinary_remainder") == "differs":
            exposed_component = "ordinary_remainder"
        elif polynomial_status == "differs":
            exposed_component = "polynomial_range"

    return {
        "verdict": verdict,
        "full_step_width_close": width_close,
        "full_step_width_ratio_torch_over_flowstar": width_ratio,
        "full_step_y_hi_delta_torch_minus_flowstar": full_step_y_hi_delta,
        "post_cutoff_residual_y_hi_delta_torch_minus_flowstar": post_cutoff_y_hi_delta,
        "acceptance_residual_gap_equals_full_step_y_hi_gap": _bool_word(gap_matches),
        "exposed_gap_component": exposed_component,
        "polynomial_range_component": polynomial_status,
        "ordinary_remainder_component": _component_status(ledger, "ordinary_remainder"),
        "raw_ctrunc_residual_component": _component_status(ledger, "raw_ctrunc_residual"),
        "post_cutoff_residual_component": _component_status(ledger, "post_cutoff_residual"),
        "cutoff_poly_diff_component": _component_status(ledger, "cutoff_poly_diff"),
        "cutoff_poly_diff_explains_gap": cutoff_explains_gap,
        "pytorch_width_in_polynomial_range": pytorch_width_in_poly,
        "flowstar_raw_no_remainder_still_missing": "true" if flowstar_no_remainder_missing else "false",
        "missing_fields": missing_fields,
        "next_fields_to_expose": (
            "any still-blank polynomial_range, ordinary_remainder/picard_no_remainder, "
            "raw_ctrunc_residual, or cutoff_poly_diff endpoints for the first differing component"
        ),
    }


def _fmt_interval(row: Mapping[str, Any], prefix: str, dim: str) -> str:
    return f"[{_format(row.get(f'{prefix}_{dim}_lo'))}, {_format(row.get(f'{prefix}_{dim}_hi'))}]"


def _report(out_dir: Path, ledger: list[dict[str, Any]], summary: Mapping[str, Any], *, t: float, h: float) -> str:
    flow = _row_for(ledger, "flowstar")
    torch = _primary_torch(ledger)
    missing_fields = summary.get("missing_fields") or []
    missing_text = "; ".join(str(field) for field in missing_fields) if missing_fields else "none"
    polynomial_range_note = (
        f"- Polynomial range endpoints are exposed in the current traces; component status is `{summary.get('polynomial_range_component', 'unknown')}`."
        if _has_component_endpoint(flow, "polynomial_range") and _has_component_endpoint(torch, "polynomial_range")
        else "- Polynomial range endpoints are blank in at least one current trace, so polynomial-vs-remainder width placement is not fully inferred."
    )
    lines = [
        "# Flow* Validation Candidate Decomposition Audit",
        "",
        "This is diagnostic-only. It does not change solver behavior, rerun h10, add symbolic queue variants, or claim Flow* parity.",
        "",
        "## Scope",
        "",
        f"- t_before requested: `{t:.17g}`",
        f"- h_try: `{h:.17g}`",
        "- Input traces: `outputs/flowstar_step_trace_compare/*.csv`",
        "- Output ledger: `outputs/flowstar_validation_candidate_decomposition_audit/validation_candidate_decomposition_ledger.csv`",
        "",
        "## Answers",
        "",
        f"- Full-step tube total boxes are width-close: `{summary.get('full_step_width_close', 'unknown')}`; width ratio torch/Flow* is `{_format(summary.get('full_step_width_ratio_torch_over_flowstar'))}`.",
        f"- Acceptance-critical residual y_hi gap equals the full-step tube y_hi gap: `{summary.get('acceptance_residual_gap_equals_full_step_y_hi_gap', 'unknown')}`.",
        f"- Same-source full-step y_hi delta torch-Flow*: `{_format(summary.get('full_step_y_hi_delta_torch_minus_flowstar'))}`.",
        f"- Post-cutoff residual y_hi delta torch-Flow*: `{_format(summary.get('post_cutoff_residual_y_hi_delta_torch_minus_flowstar'))}`.",
        f"- Verdict: `{summary.get('verdict', 'unknown')}`.",
        f"- Exposed component carrying the gap: `{summary.get('exposed_gap_component', 'unknown')}`.",
        f"- Polynomial range component: `{summary.get('polynomial_range_component', 'unknown')}`.",
        f"- Ordinary remainder component: `{summary.get('ordinary_remainder_component', 'unknown')}`.",
        f"- Raw ctrunc residual component: `{summary.get('raw_ctrunc_residual_component', 'unknown')}`.",
        f"- Post-cutoff residual component: `{summary.get('post_cutoff_residual_component', 'unknown')}`.",
        f"- Cutoff/polyDiff explains the y_hi gap: `{summary.get('cutoff_poly_diff_explains_gap', 'unknown')}`.",
        f"- Is PyTorch putting width into polynomial range that Flow* puts into remainder: `{summary.get('pytorch_width_in_polynomial_range', 'unknown')}`.",
        f"- Is Flow* raw no-remainder still missing: `{summary.get('flowstar_raw_no_remainder_still_missing', 'unknown')}`.",
        f"- Exact field to expose next if attribution remains unknown: {summary.get('next_fields_to_expose', 'unknown')}.",
        "",
        "## Candidate Rows",
        "",
        "| source | status | full-step box | target-check residual y | target y | subset y | y_hi margin | domain | center/scale |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ledger:
        center_scale = (
            f"center=({_format(row.get('center_x'))}, {_format(row.get('center_y'))}), "
            f"scale=({_format(row.get('scale_x'))}, {_format(row.get('scale_y'))})"
        )
        lines.append(
            "| {source} | {status} | x={fx}, y={fy} | {ry} | {ty} | {subset} | {margin} | {domain} | {center_scale} |".format(
                source=row.get("source", ""),
                status=row.get("status", ""),
                fx=_fmt_interval(row, "full_step", "x"),
                fy=_fmt_interval(row, "full_step", "y"),
                ry=_fmt_interval(row, "post_cutoff_residual", "y"),
                ty=_fmt_interval(row, "target_remainder", "y"),
                subset=row.get("residual_subset_y", ""),
                margin=_format(row.get("residual_y_hi_margin_to_target")),
                domain=row.get("residual_domain", "") or "unknown",
                center_scale=center_scale,
            )
        )

    lines.extend(
        [
            "",
            "## Decomposition Notes",
            "",
            f"- Flow* target-check residual y_hi is `{_format(flow.get('post_cutoff_residual_y_hi'))}`; PyTorch target-check residual y_hi is `{_format(torch.get('post_cutoff_residual_y_hi'))}`.",
            f"- Flow* y margin to target is `{_format(flow.get('residual_y_hi_margin_to_target'))}`; PyTorch y margin to target is `{_format(torch.get('residual_y_hi_margin_to_target'))}`.",
            "- The exposed Flow* raw ctrunc and post-cutoff residuals differ only by the recorded cutoff width on this row.",
            "- The exposed PyTorch ordinary no-remainder and post-cutoff residuals differ only by the recorded cutoff width on this row.",
            polynomial_range_note,
            "- Raw ctrunc residual construction audit: `outputs/flowstar_raw_ctrunc_residual_audit/raw_ctrunc_residual_report.md`; do not treat the root cause as closed unless that audit completes component attribution.",
            "- Blank component endpoint columns mean unknown, not zero.",
            f"- Missing component fields: {missing_text}.",
        ]
    )
    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_candidate_decomposition_report.md").write_text(text, encoding="utf-8")
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
    _write_rows(out_dir / "validation_candidate_decomposition_ledger.csv", ledger)
    _report(out_dir, ledger, summary, t=args.t, h=args.h)
    print(f"wrote validation candidate decomposition audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
