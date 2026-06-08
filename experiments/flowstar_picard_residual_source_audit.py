#!/usr/bin/env python3
"""Audit the source of the first Flow* vs PyTorch Picard residual mismatch.

This diagnostic reads the step traces produced by
``experiments/flowstar_step_trace_compare.py`` and writes a source-component
ledger for the first same-t/h acceptance divergence at t ~= 0, h = 0.025.
It does not add a solver mechanism and does not rerun h10.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "outputs" / "flowstar_step_trace_compare"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_picard_residual_source_audit"
DEFAULT_LIFECYCLE_DIR = ROOT / "outputs" / "flowstar_box_lifecycle_alignment_audit"

LEDGER_FIELDS = [
    "source",
    "t_before",
    "h_try",
    "status",
    "residual_x_lo",
    "residual_x_hi",
    "residual_y_lo",
    "residual_y_hi",
    "target_x_lo",
    "target_x_hi",
    "target_y_lo",
    "target_y_hi",
    "subset_x",
    "subset_y",
    "failed_dim",
    "picard_no_remainder_x_lo",
    "picard_no_remainder_x_hi",
    "picard_no_remainder_y_lo",
    "picard_no_remainder_y_hi",
    "picard_ctrunc_raw_x_lo",
    "picard_ctrunc_raw_x_hi",
    "picard_ctrunc_raw_y_lo",
    "picard_ctrunc_raw_y_hi",
    "polynomial_diff_x_lo",
    "polynomial_diff_x_hi",
    "polynomial_diff_y_lo",
    "polynomial_diff_y_hi",
    "cutoff_uncertainty_x_lo",
    "cutoff_uncertainty_x_hi",
    "cutoff_uncertainty_y_lo",
    "cutoff_uncertainty_y_hi",
    "post_cutoff_residual_x_lo",
    "post_cutoff_residual_x_hi",
    "post_cutoff_residual_y_lo",
    "post_cutoff_residual_y_hi",
    "domain_used",
    "center_x",
    "center_y",
    "scale_x",
    "scale_y",
    "local_box_x_lo",
    "local_box_x_hi",
    "local_box_y_lo",
    "local_box_y_hi",
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


def interval_subset(interval_lo: float, interval_hi: float, target_lo: float, target_hi: float, *, tol: float = 0.0) -> bool:
    """Return true only when the whole interval is contained in the target."""
    return bool(interval_lo >= target_lo - tol and interval_hi <= target_hi + tol)


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


def _bounds(row: Mapping[str, Any], prefix: str, dim: str) -> tuple[float | None, float | None]:
    lo = finite_float(_first_present(row, f"{prefix}_{dim}_lo", f"{prefix}_lo_{dim}"))
    hi = finite_float(_first_present(row, f"{prefix}_{dim}_hi", f"{prefix}_hi_{dim}"))
    return lo, hi


def _has_bounds(row: Mapping[str, Any], prefix: str) -> bool:
    return all(_bounds(row, prefix, dim)[side] is not None for dim in ("x", "y") for side in (0, 1))


def _put_bounds(out: dict[str, Any], target_prefix: str, row: Mapping[str, Any], source_prefix: str, notes: list[str]) -> bool:
    missing: list[str] = []
    for dim in ("x", "y"):
        lo, hi = _bounds(row, source_prefix, dim)
        out[f"{target_prefix}_{dim}_lo"] = lo
        out[f"{target_prefix}_{dim}_hi"] = hi
        if lo is None:
            missing.append(f"{source_prefix}_{dim}_lo")
        if hi is None:
            missing.append(f"{source_prefix}_{dim}_hi")
    if missing:
        notes.append(f"missing {target_prefix} endpoints: {', '.join(missing)}")
        return False
    return True


def _put_blank_component(out: dict[str, Any], prefix: str) -> None:
    for dim in ("x", "y"):
        out[f"{prefix}_{dim}_lo"] = ""
        out[f"{prefix}_{dim}_hi"] = ""


def _width_note(row: Mapping[str, Any], label: str, prefix: str) -> str | None:
    values: list[str] = []
    for dim in ("x", "y"):
        value = _first_present(row, f"{prefix}_width_{dim}")
        if value not in (None, ""):
            values.append(f"{dim}={_format(value)}")
    value_sum = _first_present(row, f"{prefix}_width_sum")
    if value_sum not in (None, ""):
        values.append(f"sum={_format(value_sum)}")
    if not values:
        return None
    return f"{label} width-only: {', '.join(values)}"


def _subset_label(res_lo: float | None, res_hi: float | None, tgt_lo: float | None, tgt_hi: float | None, *, tol: float) -> str:
    if res_lo is None or res_hi is None or tgt_lo is None or tgt_hi is None:
        return "unknown"
    return "true" if interval_subset(res_lo, res_hi, tgt_lo, tgt_hi, tol=tol) else "false"


def _local_box(center: float | None, scale: float | None) -> tuple[float | None, float | None]:
    if center is None or scale is None:
        return None, None
    radius = abs(scale)
    return center - radius, center + radius


def build_ledger_row(source: str, row: Mapping[str, Any], *, tolerance: float = 0.0) -> dict[str, Any]:
    """Build one source row, preserving missing component endpoints as unknown."""
    out: dict[str, Any] = {
        "source": source,
        "t_before": row.get("t_before", ""),
        "h_try": _first_present(row, "h_try", "h"),
        "status": _status(row),
    }
    notes: list[str] = []

    # The acceptance predicate in the trace is the Picard_ctrunc_normal target
    # check. These endpoints are the observed post-cutoff residual endpoints.
    residual_prefix = "post_cutoff_residual" if _has_bounds(row, "post_cutoff_residual") else "picard_ctrunc_normal_residual"
    for dim in ("x", "y"):
        res_lo, res_hi = _bounds(row, residual_prefix, dim)
        tgt_lo, tgt_hi = _bounds(row, "target_remainder", dim)
        out[f"residual_{dim}_lo"] = res_lo
        out[f"residual_{dim}_hi"] = res_hi
        out[f"target_{dim}_lo"] = tgt_lo
        out[f"target_{dim}_hi"] = tgt_hi
        out[f"post_cutoff_residual_{dim}_lo"] = res_lo
        out[f"post_cutoff_residual_{dim}_hi"] = res_hi
        out[f"subset_{dim}"] = _subset_label(res_lo, res_hi, tgt_lo, tgt_hi, tol=tolerance)

    failed: list[str] = [dim for dim in ("x", "y") if out[f"subset_{dim}"] == "false"]
    unknown_subset = any(out[f"subset_{dim}"] == "unknown" for dim in ("x", "y"))
    out["failed_dim"] = ";".join(failed) if failed else ("unknown" if unknown_subset else "")

    if _has_bounds(row, "picard_no_remainder_residual"):
        _put_bounds(out, "picard_no_remainder", row, "picard_no_remainder_residual", notes)
    elif source.startswith("torch") and _has_bounds(row, "residual"):
        _put_bounds(out, "picard_no_remainder", row, "residual", notes)
        notes.append("picard_no_remainder uses PyTorch ordinary residual endpoints from residual_lo/hi")
    else:
        _put_blank_component(out, "picard_no_remainder")
        notes.append("missing picard_no_remainder endpoints")

    if not all(out.get(f"picard_no_remainder_{dim}_{side}") not in (None, "") for dim in ("x", "y") for side in ("lo", "hi")):
        width_note = _width_note(row, "picard_no_remainder_residual", "picard_no_remainder_residual")
        if width_note:
            notes.append(width_note)

    if _has_bounds(row, "picard_ctrunc_raw_residual"):
        _put_bounds(out, "picard_ctrunc_raw", row, "picard_ctrunc_raw_residual", notes)
    elif _has_bounds(row, "picard_ctrunc_raw"):
        _put_bounds(out, "picard_ctrunc_raw", row, "picard_ctrunc_raw", notes)
    else:
        _put_blank_component(out, "picard_ctrunc_raw")
        notes.append("missing picard_ctrunc_raw endpoints")

    # Some historical PyTorch validation CSVs expose poly_diff_range endpoints,
    # but the step trace used here exposes only widths.
    if _has_bounds(row, "poly_diff_range"):
        _put_bounds(out, "polynomial_diff", row, "poly_diff_range", notes)
    else:
        _put_blank_component(out, "polynomial_diff")
        notes.append("missing polynomial_diff endpoints")
        width_note = _width_note(row, "cutoff_polynomial_difference", "cutoff_polynomial_difference")
        if width_note:
            notes.append(width_note)

    if _has_bounds(row, "cutoff_uncertainty"):
        _put_bounds(out, "cutoff_uncertainty", row, "cutoff_uncertainty", notes)
    else:
        _put_blank_component(out, "cutoff_uncertainty")
        notes.append("missing cutoff_uncertainty endpoints")

    center_x = finite_float(row.get("center_x"))
    center_y = finite_float(row.get("center_y"))
    scale_x = finite_float(row.get("scale_x"))
    scale_y = finite_float(row.get("scale_y"))
    out["center_x"] = center_x
    out["center_y"] = center_y
    out["scale_x"] = scale_x
    out["scale_y"] = scale_y
    x_lo, x_hi = _local_box(center_x, scale_x)
    y_lo, y_hi = _local_box(center_y, scale_y)
    out["local_box_x_lo"] = x_lo
    out["local_box_x_hi"] = x_hi
    out["local_box_y_lo"] = y_lo
    out["local_box_y_hi"] = y_hi
    out["domain_used"] = "center_scale_inferred" if all(value is not None for value in (center_x, center_y, scale_x, scale_y)) else "unknown"
    if out["domain_used"] == "unknown":
        notes.append("missing center/scale domain fields")
    else:
        notes.append("generic center/scale local_box is deprecated for same-stage comparison; use lifecycle stage boxes")

    rejection_reason = _first_present(row, "rejection_reason", "message", "validation_message")
    if rejection_reason:
        notes.append(f"reason: {rejection_reason}")
    notes.append("post_cutoff_residual uses picard_ctrunc_normal_residual endpoints")
    out["notes"] = "; ".join(notes)
    return out


def build_ledger(flowstar_rows: list[Mapping[str, Any]], noqueue_rows: list[Mapping[str, Any]], v2_rows: list[Mapping[str, Any]], *, t: float, h: float) -> list[dict[str, Any]]:
    pairs = [
        ("flowstar", _find_attempt(flowstar_rows, t=t, h=h)),
        ("torch_noqueue", _find_attempt(noqueue_rows, t=t, h=h)),
        ("torch_v2", _find_attempt(v2_rows, t=t, h=h)),
    ]
    missing = [source for source, row in pairs if row is None]
    if missing:
        raise ValueError(f"missing same-t/h trace rows for: {', '.join(missing)}")
    return [build_ledger_row(source, row) for source, row in pairs if row is not None]


def _row_for(rows: list[Mapping[str, Any]], source: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("source") == source:
            return row
    raise KeyError(source)


def _num(row: Mapping[str, Any], field: str) -> float | None:
    return finite_float(row.get(field))


def _same_interval(rows: Iterable[Mapping[str, Any]], lo_field: str, hi_field: str) -> bool | None:
    pairs = [(_num(row, lo_field), _num(row, hi_field)) for row in rows]
    if any(lo is None or hi is None for lo, hi in pairs):
        return None
    first = pairs[0]
    return all(pair == first for pair in pairs[1:])


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _abs_delta(a: float | None, b: float | None) -> float | None:
    value = _delta(a, b)
    return None if value is None else abs(value)


def _max_abs(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _fmt_interval(row: Mapping[str, Any], prefix: str, dim: str) -> str:
    return f"[{_format(row.get(f'{prefix}_{dim}_lo'))}, {_format(row.get(f'{prefix}_{dim}_hi'))}]"


def _bool_word(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def _read_lifecycle_summary(lifecycle_dir: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pre_step_boxes_equal": "unknown",
        "endpoint_before_center_comparable": "unknown",
        "endpoint_before_center_boxes_equal": "unknown",
        "reset_after_center_comparable": "unknown",
        "reset_after_center_boxes_equal": "unknown",
        "first_lifecycle_stage_divergence": "unknown_missing_stage_fields",
        "residual_comparison_stage_valid": "unknown",
        "picard_residual_comparison": "noncausal/stage-misaligned",
        "flowstar_missing_residual_components": "",
    }
    if lifecycle_dir is None:
        return summary
    ledger_path = lifecycle_dir / "box_lifecycle_ledger.csv"
    if not ledger_path.exists():
        summary["notes"] = f"missing lifecycle ledger: {ledger_path}"
        return summary
    rows = _read_rows(ledger_path)
    if not rows:
        summary["notes"] = f"empty lifecycle ledger: {ledger_path}"
        return summary
    row = rows[0]
    for key in summary:
        value = row.get(key)
        if value is not None and (value != "" or key == "flowstar_missing_residual_components"):
            summary[key] = value
    return summary


def _component_delta_label(rows: list[dict[str, Any]], prefix: str) -> str:
    same_x = _same_interval(rows, f"{prefix}_x_lo", f"{prefix}_x_hi")
    same_y = _same_interval(rows, f"{prefix}_y_lo", f"{prefix}_y_hi")
    if same_x is None or same_y is None:
        return "unknown"
    return "same" if same_x and same_y else "differs"


def _first_valid_stage_component(ledger: list[dict[str, Any]], target_same_x: bool | None, target_same_y: bool | None) -> str:
    if target_same_x is False or target_same_y is False:
        return "target_remainder"
    for prefix, label in (
        ("picard_no_remainder", "picard_no_remainder"),
        ("picard_ctrunc_raw", "picard_ctrunc_raw"),
        ("polynomial_diff", "polynomial_difference/cutoff"),
        ("post_cutoff_residual", "post_cutoff_residual"),
    ):
        if _component_delta_label(ledger, prefix) == "differs":
            return label
    flow = _row_for(ledger, "flowstar")
    if flow.get("subset_x") == "false" or flow.get("subset_y") == "false":
        return "interval_subset_predicate"
    return "unknown"


def _report(
    out_dir: Path,
    ledger: list[dict[str, Any]],
    *,
    t: float,
    h: float,
    lifecycle_summary: Mapping[str, Any] | None = None,
) -> str:
    flow = _row_for(ledger, "flowstar")
    noq = _row_for(ledger, "torch_noqueue")
    target_same_x = _same_interval(ledger, "target_x_lo", "target_x_hi")
    target_same_y = _same_interval(ledger, "target_y_lo", "target_y_hi")
    target_mismatch = None if target_same_x is None or target_same_y is None else not (target_same_x and target_same_y)
    flow_y_excess = _delta(_num(flow, "residual_y_hi"), _num(flow, "target_y_hi"))
    noqueue_y_gap = _delta(_num(flow, "residual_y_hi"), _num(noq, "residual_y_hi"))
    noqueue_ordinary_to_post_y = _delta(_num(noq, "post_cutoff_residual_y_hi"), _num(noq, "picard_no_remainder_y_hi"))
    lifecycle = dict(lifecycle_summary or _read_lifecycle_summary(DEFAULT_LIFECYCLE_DIR))
    stage_valid = str(lifecycle.get("residual_comparison_stage_valid", "unknown")).strip().lower()
    same_stage_valid = stage_valid == "true"
    first_component = _first_valid_stage_component(ledger, target_same_x, target_same_y) if same_stage_valid else "not-attributed-stage-misaligned"

    lines = [
        "# Flow* Picard Residual Source Audit",
        "",
        "This audit uses the first same-t/h divergence only. It does not add a solver mechanism, does not rerun h10, and does not claim Flow* parity.",
        "",
        "## Scope",
        "",
        f"- t_before requested: `{t:.17g}`",
        f"- h_try: `{h:.17g}`",
        "- Input traces: `outputs/flowstar_step_trace_compare/*.csv`",
        "- Output ledger: `outputs/flowstar_picard_residual_source_audit/picard_residual_source_ledger.csv`",
        "- Lifecycle ledger: `outputs/flowstar_box_lifecycle_alignment_audit/box_lifecycle_ledger.csv`",
        "",
        "## Lifecycle Gate",
        "",
        f"- Pre-step boxes equal: `{lifecycle.get('pre_step_boxes_equal', 'unknown')}`.",
        f"- Endpoint-before-center comparable: `{lifecycle.get('endpoint_before_center_comparable', 'unknown')}`.",
        f"- Reset-after-center boxes equal: `{lifecycle.get('reset_after_center_boxes_equal', 'unknown')}`.",
        f"- First lifecycle stage divergence: `{lifecycle.get('first_lifecycle_stage_divergence', 'unknown')}`.",
        f"- Residual comparison same-stage valid: `{lifecycle.get('residual_comparison_stage_valid', 'unknown')}`.",
        f"- Picard residual comparison: `{lifecycle.get('picard_residual_comparison', 'unknown')}`.",
        f"- Flow* missing residual components: `{lifecycle.get('flowstar_missing_residual_components') or 'none'}`.",
    ]
    if not same_stage_valid:
        lines.extend(
            [
                "",
                "The residual endpoint mismatch is not yet a valid same-local-box comparison.",
            ]
        )

    lines.extend(
        [
            "",
            "## Target-Check Residuals",
            "",
            "| source | status | residual x | residual y | target x | target y | subset x | subset y | failed dim | local box |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in ledger:
        local_box = (
            f"x={_fmt_interval(row, 'local_box', 'x')}, y={_fmt_interval(row, 'local_box', 'y')}"
            if row.get("domain_used") != "unknown"
            else "unknown"
        )
        lines.append(
            "| {source} | {status} | {rx} | {ry} | {tx} | {ty} | {sx} | {sy} | {failed} | {box} |".format(
                source=row.get("source", ""),
                status=row.get("status", ""),
                rx=_fmt_interval(row, "residual", "x"),
                ry=_fmt_interval(row, "residual", "y"),
                tx=_fmt_interval(row, "target", "x"),
                ty=_fmt_interval(row, "target", "y"),
                sx=row.get("subset_x", ""),
                sy=row.get("subset_y", ""),
                failed=row.get("failed_dim", ""),
                box=local_box,
            )
        )

    if same_stage_valid:
        attribution = [
            f"- First component source after lifecycle alignment: `{first_component}`.",
            f"- Picard no-remainder: `{_component_delta_label(ledger, 'picard_no_remainder')}` across exposed endpoints.",
            f"- Picard ctrunc: `{_component_delta_label(ledger, 'picard_ctrunc_raw')}` across exposed raw-ctrunc endpoints.",
            f"- Polynomial difference/cutoff: `{_component_delta_label(ledger, 'polynomial_diff')}` across exposed endpoints; PyTorch ordinary-to-post y upper shift is `{_format(noqueue_ordinary_to_post_y)}`.",
            f"- Target remainder interval mismatch: `{_bool_word(target_mismatch)}`.",
            f"- Interval subset tolerance: Flow* y upper exceeds target by `{_format(flow_y_excess)}`; the subset predicate checks endpoints, not width alone.",
            "- Missing term in PyTorch residual accounting: `unknown` unless an exposed same-stage component remains inconsistent after target and tolerance checks.",
        ]
    else:
        attribution = [
            "- Picard no-remainder: `not attributed`; lifecycle stage alignment is invalid or unknown.",
            f"- Picard ctrunc: `not attributed`; Flow* post-cutoff/Picard_ctrunc_normal y upper is `{_format(flow.get('residual_y_hi'))}`, PyTorch no_queue y upper is `{_format(noq.get('residual_y_hi'))}`, but their local-box stages are not yet proven comparable.",
            f"- Polynomial difference/cutoff: `not attributed`; PyTorch ordinary-to-post y upper shift is `{_format(noqueue_ordinary_to_post_y)}`, but cutoff attribution would be noncausal before same-stage boxes align.",
            f"- Domain/center/scale mismatch: `not evaluated from generic center/scale fields`; use stage-labeled boxes. Lifecycle first divergence is `{lifecycle.get('first_lifecycle_stage_divergence', 'unknown')}`.",
            f"- Target remainder interval mismatch: `{_bool_word(target_mismatch)}`.",
            f"- Interval subset tolerance: `not the observed predicate issue`. Flow* fails endpoint inclusion in y by `{_format(flow_y_excess)}`, but this does not identify the residual source while stage alignment is invalid or unknown.",
            f"- Missing term in PyTorch residual accounting: `unknown`; the Flow* vs PyTorch y-upper gap `{_format(noqueue_y_gap)}` is not a causal residual-accounting comparison until lifecycle boxes align.",
        ]

    lines.extend(
        [
            "",
            "## Attribution Answers",
            "",
            *attribution,
            "",
            "## Missing Fields",
            "",
            "- Blank component endpoint columns mean the source trace did not expose that component endpoint.",
            "- Width-only component evidence is kept in the row notes instead of being converted into fabricated intervals.",
            "- Generic center/scale local_box columns are preserved for continuity but are deprecated for same-stage residual attribution.",
        ]
    )
    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "picard_residual_source_report.md").write_text(text, encoding="utf-8")
    return text

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lifecycle-dir", type=Path, default=DEFAULT_LIFECYCLE_DIR)
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
    _write_rows(out_dir / "picard_residual_source_ledger.csv", ledger)
    lifecycle_summary = _read_lifecycle_summary(args.lifecycle_dir.resolve() if args.lifecycle_dir else None)
    _report(out_dir, ledger, t=args.t, h=args.h, lifecycle_summary=lifecycle_summary)
    print(f"wrote Picard residual source audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
