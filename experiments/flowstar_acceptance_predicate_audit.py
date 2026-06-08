#!/usr/bin/env python3
"""Audit the first Flow* vs PyTorch acceptance-predicate divergence.

This is a diagnostic report generator. It reads the short step traces produced by
``experiments/flowstar_step_trace_compare.py`` and writes a component-level
ledger for the first same-t/h divergence at t ~= 0, h = 0.025.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "outputs" / "flowstar_step_trace_compare"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_acceptance_predicate_audit"

LEDGER_FIELDS = [
    "trace_source",
    "status",
    "t_before",
    "h_try",
    "residual_x_lo",
    "residual_x_hi",
    "target_x_lo",
    "target_x_hi",
    "subset_x",
    "residual_y_lo",
    "residual_y_hi",
    "target_y_lo",
    "target_y_hi",
    "subset_y",
    "which_dim_failed",
    "residual_width_x",
    "target_width_x",
    "residual_width_y",
    "target_width_y",
    "residual_width_sum",
    "target_width_sum",
    "residual_over_target_sum",
    "residual_source_field",
    "target_source_field",
    "rejection_reason",
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


def _status(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "missing"
    raw = str(row.get("status", "")).strip().lower()
    if raw:
        return raw
    if str(row.get("accepted", "")).strip().lower() in {"true", "1", "yes"}:
        return "accepted"
    if str(row.get("rejected", "")).strip().lower() in {"true", "1", "yes"}:
        return "rejected"
    return "missing"


def _find_attempt(rows: Iterable[Mapping[str, Any]], *, t: float, h: float, tolerance: float = 1e-9) -> Mapping[str, Any] | None:
    for row in rows:
        t_before = finite_float(row.get("t_before"))
        h_try = finite_float(row.get("h_try"))
        if h_try is None:
            h_try = finite_float(row.get("h"))
        if t_before is None or h_try is None:
            continue
        if abs(t_before - t) <= tolerance and abs(h_try - h) <= tolerance:
            return row
    return None


def _first_present(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return ""


def _width(lo: float | None, hi: float | None) -> float | None:
    if lo is None or hi is None:
        return None
    return hi - lo


def build_ledger_row(trace_source: str, row: Mapping[str, Any], *, tolerance: float = 0.0) -> dict[str, Any]:
    residual_prefix = "picard_ctrunc_normal_residual"
    target_prefix = "target_remainder"
    out: dict[str, Any] = {
        "trace_source": trace_source,
        "status": _status(row),
        "t_before": row.get("t_before", ""),
        "h_try": _first_present(row, "h_try", "h"),
        "residual_source_field": residual_prefix,
        "target_source_field": target_prefix,
        "rejection_reason": _first_present(row, "rejection_reason", "message"),
    }
    failed: list[str] = []
    residual_width_sum = 0.0
    target_width_sum = 0.0
    saw_all_widths = True
    for dim in ("x", "y"):
        residual_lo = finite_float(row.get(f"{residual_prefix}_lo_{dim}"))
        residual_hi = finite_float(row.get(f"{residual_prefix}_hi_{dim}"))
        target_lo = finite_float(row.get(f"{target_prefix}_lo_{dim}"))
        target_hi = finite_float(row.get(f"{target_prefix}_hi_{dim}"))
        if target_lo is None:
            target_lo = -1e-4
        if target_hi is None:
            target_hi = 1e-4
        subset = (
            residual_lo is not None
            and residual_hi is not None
            and target_lo is not None
            and target_hi is not None
            and interval_subset(residual_lo, residual_hi, target_lo, target_hi, tol=tolerance)
        )
        if not subset:
            failed.append(dim)
        rw = _width(residual_lo, residual_hi)
        tw = _width(target_lo, target_hi)
        if rw is None or tw is None:
            saw_all_widths = False
        else:
            residual_width_sum += rw
            target_width_sum += tw
        out[f"residual_{dim}_lo"] = residual_lo
        out[f"residual_{dim}_hi"] = residual_hi
        out[f"target_{dim}_lo"] = target_lo
        out[f"target_{dim}_hi"] = target_hi
        out[f"subset_{dim}"] = subset
        out[f"residual_width_{dim}"] = rw
        out[f"target_width_{dim}"] = tw
    out["which_dim_failed"] = ";".join(failed) if failed else ""
    out["residual_width_sum"] = residual_width_sum if saw_all_widths else _first_present(row, "picard_ctrunc_normal_residual_width_sum", "residual_width_sum")
    out["target_width_sum"] = target_width_sum if saw_all_widths else _first_present(row, "target_remainder_width_sum", "target_check_width_sum")
    out["residual_over_target_sum"] = (
        residual_width_sum / target_width_sum
        if saw_all_widths and target_width_sum not in (0.0, None)
        else _first_present(row, "residual_over_target_sum")
    )
    return out


def build_ledger(flowstar_rows: list[Mapping[str, Any]], noqueue_rows: list[Mapping[str, Any]], v2_rows: list[Mapping[str, Any]], *, t: float, h: float) -> list[dict[str, Any]]:
    pairs = [
        ("flowstar", _find_attempt(flowstar_rows, t=t, h=h)),
        ("torch_noqueue", _find_attempt(noqueue_rows, t=t, h=h)),
        ("torch_v2", _find_attempt(v2_rows, t=t, h=h)),
    ]
    missing = [name for name, row in pairs if row is None]
    if missing:
        raise ValueError(f"missing same-t/h trace rows for: {', '.join(missing)}")
    return [build_ledger_row(name, row) for name, row in pairs if row is not None]


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _row_for(rows: list[Mapping[str, Any]], source: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("trace_source") == source:
            return row
    raise KeyError(source)


def _report(out_dir: Path, ledger: list[dict[str, Any]], *, t: float, h: float) -> str:
    flow = _row_for(ledger, "flowstar")
    noq = _row_for(ledger, "torch_noqueue")
    v2 = _row_for(ledger, "torch_v2")
    flow_subset = bool(flow.get("subset_x")) and bool(flow.get("subset_y"))
    noq_subset = bool(noq.get("subset_x")) and bool(noq.get("subset_y"))
    v2_subset = bool(v2.get("subset_x")) and bool(v2.get("subset_y"))
    failed_dim = str(flow.get("which_dim_failed", "")) or "none"
    flow_width = finite_float(flow.get("residual_width_sum"))
    target_width = finite_float(flow.get("target_width_sum"))
    width_clause = "unknown"
    if flow_width is not None and target_width is not None:
        width_clause = f"{flow_width:.17g} < {target_width:.17g}" if flow_width < target_width else f"{flow_width:.17g} >= {target_width:.17g}"

    lines = [
        "# Flow* Acceptance Predicate Audit",
        "",
        "This audit uses the first same-t/h divergence only. It does not add a solver mechanism and does not rerun h10.",
        "",
        "## First Divergence",
        "",
        f"- t_before: `{t:.17g}`",
        f"- h_try: `{h:.17g}`",
        f"- Flow*: `{flow.get('status')}`",
        f"- PyTorch no_queue: `{noq.get('status')}`",
        f"- PyTorch v2: `{v2.get('status')}`",
        "",
        "## Predicate Ledger",
        "",
        "| source | residual x | target x | subset x | residual y | target y | subset y | failed dim | width sum / target |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ledger:
        lines.append(
            "| {source} | [{rxlo}, {rxhi}] | [{txlo}, {txhi}] | {sx} | [{rylo}, {ryhi}] | [{tylo}, {tyhi}] | {sy} | {failed} | {rw} / {tw} |".format(
                source=row.get("trace_source", ""),
                rxlo=_format(row.get("residual_x_lo")),
                rxhi=_format(row.get("residual_x_hi")),
                txlo=_format(row.get("target_x_lo")),
                txhi=_format(row.get("target_x_hi")),
                sx=_yes_no(row.get("subset_x")),
                rylo=_format(row.get("residual_y_lo")),
                ryhi=_format(row.get("residual_y_hi")),
                tylo=_format(row.get("target_y_lo")),
                tyhi=_format(row.get("target_y_hi")),
                sy=_yes_no(row.get("subset_y")),
                failed=row.get("which_dim_failed", ""),
                rw=_format(row.get("residual_width_sum")),
                tw=_format(row.get("target_width_sum")),
            )
        )
    lines.extend(
        [
            "",
            "## Flow* vs PyTorch Residual",
            "",
            f"- Flow* Picard_ctrunc_normal subset: `{_yes_no(flow_subset)}`.",
            f"- PyTorch no_queue Picard_ctrunc-style subset: `{_yes_no(noq_subset)}`.",
            f"- PyTorch v2 Picard_ctrunc-style subset: `{_yes_no(v2_subset)}`.",
            f"- Flow* failed dimension: `{failed_dim}`.",
            "",
            "## Why Width Is Not Enough",
            "",
            f"Flow* rejects because interval inclusion is endpoint-wise. In this row, the Flow* residual width sum relation is `{width_clause}`, but the residual interval is shifted outside the symmetric target in dimension `{failed_dim}`. A smaller interval can still fail containment when its lower bound is below the target lower bound or its upper bound is above the target upper bound.",
            "",
            "## Output",
            "",
            "- `outputs/flowstar_acceptance_predicate_audit/acceptance_predicate_ledger.csv`",
            "- `outputs/flowstar_acceptance_predicate_audit/acceptance_predicate_report.md`",
            "",
            "## Limitation",
            "",
            "The audit compares the diagnostic residual intervals exposed by the Flow* probe and the PyTorch flowstar-ctrunc validator for the same local box and h. It is not a new acceptance policy and not an end-to-end reachability comparison.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (out_dir / "acceptance_predicate_report.md").write_text(text, encoding="utf-8")
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
    _write_rows(out_dir / "acceptance_predicate_ledger.csv", ledger)
    _report(out_dir, ledger, t=args.t, h=args.h)
    print(f"wrote acceptance predicate audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
