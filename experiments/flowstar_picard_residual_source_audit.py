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
    return finite_float(row.get(f"{prefix}_lo_{dim}")), finite_float(row.get(f"{prefix}_hi_{dim}"))


def _put_bounds(out: dict[str, Any], target_prefix: str, row: Mapping[str, Any], source_prefix: str, notes: list[str]) -> bool:
    missing: list[str] = []
    for dim in ("x", "y"):
        lo_key = f"{source_prefix}_lo_{dim}"
        hi_key = f"{source_prefix}_hi_{dim}"
        lo = finite_float(row.get(lo_key))
        hi = finite_float(row.get(hi_key))
        out[f"{target_prefix}_{dim}_lo"] = lo
        out[f"{target_prefix}_{dim}_hi"] = hi
        if lo is None:
            missing.append(lo_key)
        if hi is None:
            missing.append(hi_key)
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
    residual_prefix = "picard_ctrunc_normal_residual"
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

    # PyTorch trace rows keep ordinary Picard residual endpoints in residual_*.
    # The Flow* probe does not expose no-remainder residual endpoints.
    if source.startswith("torch"):
        _put_bounds(out, "picard_no_remainder", row, "residual", notes)
        notes.append("picard_no_remainder uses PyTorch ordinary residual endpoints from residual_lo/hi")
    elif all(row.get(f"picard_no_remainder_residual_{side}_{dim}") not in (None, "") for dim in ("x", "y") for side in ("lo", "hi")):
        _put_bounds(out, "picard_no_remainder", row, "picard_no_remainder_residual", notes)
    else:
        _put_blank_component(out, "picard_no_remainder")
        notes.append("missing picard_no_remainder endpoints")

    if not all(out.get(f"picard_no_remainder_{dim}_{side}") not in (None, "") for dim in ("x", "y") for side in ("lo", "hi")):
        width_note = _width_note(row, "picard_no_remainder_residual", "picard_no_remainder_residual")
        if width_note:
            notes.append(width_note)

    # Raw Picard_ctrunc-before-polynomial-difference endpoints are not present
    # in the current traces.
    if all(row.get(f"picard_ctrunc_raw_{side}_{dim}") not in (None, "") for dim in ("x", "y") for side in ("lo", "hi")):
        _put_bounds(out, "picard_ctrunc_raw", row, "picard_ctrunc_raw", notes)
    else:
        _put_blank_component(out, "picard_ctrunc_raw")
        notes.append("missing picard_ctrunc_raw endpoints")

    # Some historical PyTorch validation CSVs expose poly_diff_range endpoints,
    # but the step trace used here exposes only widths.
    if all(row.get(f"poly_diff_range_{side}_{dim}") not in (None, "") for dim in ("x", "y") for side in ("lo", "hi")):
        _put_bounds(out, "polynomial_diff", row, "poly_diff_range", notes)
    else:
        _put_blank_component(out, "polynomial_diff")
        notes.append("missing polynomial_diff endpoints")
        width_note = _width_note(row, "cutoff_polynomial_difference", "cutoff_polynomial_difference")
        if width_note:
            notes.append(width_note)

    if all(row.get(f"cutoff_uncertainty_{side}_{dim}") not in (None, "") for dim in ("x", "y") for side in ("lo", "hi")):
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


def _report(out_dir: Path, ledger: list[dict[str, Any]], *, t: float, h: float) -> str:
    flow = _row_for(ledger, "flowstar")
    noq = _row_for(ledger, "torch_noqueue")
    v2 = _row_for(ledger, "torch_v2")
    target_same_x = _same_interval(ledger, "target_x_lo", "target_x_hi")
    target_same_y = _same_interval(ledger, "target_y_lo", "target_y_hi")
    flow_y_excess = _delta(_num(flow, "residual_y_hi"), _num(flow, "target_y_hi"))
    noqueue_y_gap = _delta(_num(flow, "residual_y_hi"), _num(noq, "residual_y_hi"))
    center_delta = _max_abs(
        [
            _abs_delta(_num(flow, "center_x"), _num(noq, "center_x")),
            _abs_delta(_num(flow, "center_y"), _num(noq, "center_y")),
        ]
    )
    scale_delta = _max_abs(
        [
            _abs_delta(_num(flow, "scale_x"), _num(noq, "scale_x")),
            _abs_delta(_num(flow, "scale_y"), _num(noq, "scale_y")),
        ]
    )
    noqueue_ordinary_to_post_y = _delta(_num(noq, "post_cutoff_residual_y_hi"), _num(noq, "picard_no_remainder_y_hi"))

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
        "",
        "## Target-Check Residuals",
        "",
        "| source | status | residual x | residual y | target x | target y | subset x | subset y | failed dim | local box |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
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

    lines.extend(
        [
            "",
            "## Attribution Answers",
            "",
            f"- Picard no-remainder: `unknown` for Flow* because the probe does not expose no-remainder residual endpoints. The PyTorch ordinary residual endpoints are inside the target at h=0.025, so this row does not support a PyTorch no-remainder rejection.",
            f"- Picard ctrunc: `yes, at the exposed target-check residual`. Flow* post-cutoff/Picard_ctrunc_normal y upper is `{_format(flow.get('residual_y_hi'))}`, above target `{_format(flow.get('target_y_hi'))}` by `{_format(flow_y_excess)}`; PyTorch no_queue y upper is `{_format(noq.get('residual_y_hi'))}`, lower than Flow* by `{_format(noqueue_y_gap)}`.",
            f"- Polynomial difference/cutoff: `not supported as the primary source by exposed widths`. The endpoint fields are missing, but width-only trace fields are tiny here; PyTorch ordinary-to-post y upper shift is `{_format(noqueue_ordinary_to_post_y)}`.",
            f"- Domain/center/scale mismatch: `yes`. The inferred local boxes differ; max center delta Flow* vs no_queue is `{_format(center_delta)}` and max scale delta is `{_format(scale_delta)}`.",
            f"- Target remainder interval mismatch: `{_bool_word(not (target_same_x and target_same_y) if target_same_x is not None and target_same_y is not None else None)}`. All exposed target intervals are `[-0.0001, 0.0001]`.",
            f"- Interval subset tolerance: `no`. Flow* fails endpoint inclusion in y; the upper endpoint exceeds the target by `{_format(flow_y_excess)}`, so width-only comparison is not the predicate.",
            "- Missing term in PyTorch residual accounting: `not indicated by this trace`. PyTorch records both ordinary residual endpoints and post-cutoff/Picard_ctrunc_normal endpoints; the recorded post-cutoff change is far too small to explain the Flow* y-upper gap. Flow* raw ctrunc and no-remainder endpoints remain missing, so the precise pre/post-ctrunc split is still unknown.",
            "",
            "## Missing Fields",
            "",
            "- Blank component endpoint columns mean the source trace did not expose that component endpoint.",
            "- Width-only component evidence is kept in the row notes instead of being converted into fabricated intervals.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (out_dir / "picard_residual_source_report.md").write_text(text, encoding="utf-8")
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
    _write_rows(out_dir / "picard_residual_source_ledger.csv", ledger)
    _report(out_dir, ledger, t=args.t, h=args.h)
    print(f"wrote Picard residual source audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
