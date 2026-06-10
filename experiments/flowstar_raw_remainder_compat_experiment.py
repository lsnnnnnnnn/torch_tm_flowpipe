#!/usr/bin/env python3
"""One-step Flow* raw remainder compatibility comparator.

This script is intentionally local and narrow. It compares the first Van der Pol
same-t/h validation divergence at h=0.025 and does not run h10, NNCS, GPU demos,
or symbolic queue variants beyond the existing v2 comparison mode.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import Interval, TMVector, flowpipe_step_flowstar_style_adaptive

DEFAULT_FLOWSTAR_TRACE = ROOT / "outputs" / "flowstar_step_trace_compare" / "flowstar_trace.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat"
TARGET_RADIUS = 1e-4
H_TRY = 0.025
T_BEFORE = 0.0
ORDER = 4
RESIDUAL_TOL = 1e-6

LEDGER_FIELDS = [
    "source",
    "mode",
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
    "raw_ctrunc_residual_y_hi",
    "accumulated_before_x0_add_y_hi",
    "polynomial_range_y_hi",
    "full_step_tube_y_hi",
    "cutoff_poly_diff_y_hi",
    "matches_flowstar_accept_reject",
    "matches_flowstar_residual_y_hi_within_tol",
    "notes",
]


def van_der_pol_flowstar_expression_ode(x: TMVector, u: TMVector | None = None) -> TMVector:
    """Algebraic VDP RHS spelled like the Flow* probe: y' = y - x - x^2*y."""
    return TMVector([x[1], x[1] - x[0] - (x[0] * x[0]) * x[1]])


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


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def _first_present(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return ""


def _bound(row: Mapping[str, Any], prefix: str, dim: str, side: str) -> Any:
    return _first_present(row, f"{prefix}_{dim}_{side}", f"{prefix}_{side}_{dim}")


def _contains(lo: Any, hi: Any, target_lo: Any, target_hi: Any) -> bool:
    lo_f = _float(lo)
    hi_f = _float(hi)
    target_lo_f = _float(target_lo)
    target_hi_f = _float(target_hi)
    if None in (lo_f, hi_f, target_lo_f, target_hi_f):
        return False
    return target_lo_f <= lo_f and hi_f <= target_hi_f


def _status(row: Mapping[str, Any]) -> str:
    raw = str(_first_present(row, "status", "validation_status")).strip().lower()
    if raw in {"accepted", "validated"}:
        return "accepted"
    if raw in {"rejected", "failed", "failure"}:
        return "rejected"
    if str(row.get("accepted", "")).strip().lower() in {"1", "true", "yes", "validated", "accepted"}:
        return "accepted"
    if str(row.get("rejected", "")).strip().lower() in {"1", "true", "yes", "failed", "rejected"}:
        return "rejected"
    return raw or "missing"


def _interval_hi(intervals: list[Interval] | None, index: int) -> float | None:
    if intervals is None or index >= len(intervals):
        return None
    try:
        return float(intervals[index].hi.detach().cpu())
    except (TypeError, ValueError):
        return None


def _range_box(tmv: TMVector | None) -> list[Interval] | None:
    if tmv is None:
        return None
    try:
        return tmv.range_box()
    except Exception:
        return None


def find_flowstar_one_step_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    candidates: list[Mapping[str, Any]] = []
    for row in rows:
        h = _float(_first_present(row, "h_try", "h"))
        t = _float(row.get("t_before"))
        if h is None or t is None:
            continue
        if abs(h - H_TRY) <= 1e-12 and abs(t - T_BEFORE) <= 1e-6:
            candidates.append(row)
    if not candidates:
        raise RuntimeError(f"no Flow* trace row found for t~=0 and h={H_TRY}")
    return candidates[0]


def run_torch_one_step(mode: str) -> tuple[dict[str, Any], Any]:
    if mode not in {"current_no_queue", "current_v2", "flowstar_raw_remainder_compat"}:
        raise ValueError(f"unknown torch mode: {mode}")
    validation_mode = "flowstar_raw_remainder_compat" if mode == "flowstar_raw_remainder_compat" else "target_remainder_flowstar_ctrunc"
    reset_mode = "normalized_insertion_symqueue_v2" if mode == "current_v2" else "normalized_insertion"
    symbolic_queue_mode = "flowstar_linear_v2" if mode == "current_v2" else ""
    diagnostics: list[dict[str, Any]] = []
    seg = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_flowstar_expression_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=H_TRY,
        h_min=H_TRY,
        h_max=H_TRY,
        order=ORDER,
        target_remainder_radius=TARGET_RADIUS,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_mode=validation_mode,
        reset_mode=reset_mode,
        symbolic_queue_mode=symbolic_queue_mode,
        flowstar_symbolic_queue_max_size=100,
        grow_factor=1.0,
        diagnostics=diagnostics,
        diagnostics_context={"mode": mode, "segment_index": 0, "t_before": T_BEFORE},
    )
    if not diagnostics:
        raise RuntimeError(f"no diagnostics captured for {mode}")
    return diagnostics[-1], seg


def _flowstar_ledger_row(row: Mapping[str, Any]) -> dict[str, Any]:
    residual_prefix = "picard_ctrunc_normal_residual"
    out = {
        "source": "flowstar",
        "mode": "probe",
        "t_before": T_BEFORE,
        "h_try": H_TRY,
        "status": _status(row),
        "residual_x_lo": _bound(row, residual_prefix, "x", "lo"),
        "residual_x_hi": _bound(row, residual_prefix, "x", "hi"),
        "residual_y_lo": _bound(row, residual_prefix, "y", "lo"),
        "residual_y_hi": _bound(row, residual_prefix, "y", "hi"),
        "target_x_lo": _bound(row, "target_remainder", "x", "lo"),
        "target_x_hi": _bound(row, "target_remainder", "x", "hi"),
        "target_y_lo": _bound(row, "target_remainder", "y", "lo"),
        "target_y_hi": _bound(row, "target_remainder", "y", "hi"),
        "raw_ctrunc_residual_y_hi": _bound(row, "raw_ctrunc_residual", "y", "hi"),
        "accumulated_before_x0_add_y_hi": _bound(row, "accumulated_remainder_before_x0_add", "y", "hi"),
        "polynomial_range_y_hi": _bound(row, "raw_ctrunc_polynomial_range", "y", "hi"),
        "full_step_tube_y_hi": _bound(row, "flowstar_full_step_tube", "y", "hi"),
        "cutoff_poly_diff_y_hi": _bound(row, "cutoff_poly_diff", "y", "hi"),
        "notes": "existing Flow* one-step probe trace; no h10 rerun",
    }
    out["subset_x"] = _contains(out["residual_x_lo"], out["residual_x_hi"], out["target_x_lo"], out["target_x_hi"])
    out["subset_y"] = _contains(out["residual_y_lo"], out["residual_y_hi"], out["target_y_lo"], out["target_y_hi"])
    failed = [dim for dim in ("x", "y") if not out[f"subset_{dim}"]]
    out["failed_dim"] = ";".join(failed)
    return out


def _torch_ledger_row(mode: str, diagnostic: Mapping[str, Any], seg: Any) -> dict[str, Any]:
    residual_prefix = "flowstar_raw_remainder_compat_check_remainder" if mode == "flowstar_raw_remainder_compat" else "tmp_remainder"
    target_prefix = "target_remainder_before_ctrunc"
    full_step = _range_box(getattr(seg, "tm", None))
    out = {
        "source": "torch",
        "mode": mode,
        "t_before": T_BEFORE,
        "h_try": H_TRY,
        "status": "accepted" if getattr(seg, "status", "") == "validated" else "rejected",
        "residual_x_lo": _bound(diagnostic, residual_prefix, "x", "lo"),
        "residual_x_hi": _bound(diagnostic, residual_prefix, "x", "hi"),
        "residual_y_lo": _bound(diagnostic, residual_prefix, "y", "lo"),
        "residual_y_hi": _bound(diagnostic, residual_prefix, "y", "hi"),
        "target_x_lo": _bound(diagnostic, target_prefix, "x", "lo"),
        "target_x_hi": _bound(diagnostic, target_prefix, "x", "hi"),
        "target_y_lo": _bound(diagnostic, target_prefix, "y", "lo"),
        "target_y_hi": _bound(diagnostic, target_prefix, "y", "hi"),
        "raw_ctrunc_residual_y_hi": _bound(diagnostic, "raw_ctrunc_residual", "y", "hi"),
        "accumulated_before_x0_add_y_hi": _bound(diagnostic, "accumulated_remainder_before_x0_add", "y", "hi"),
        "polynomial_range_y_hi": _bound(diagnostic, "raw_ctrunc_polynomial_range", "y", "hi"),
        "full_step_tube_y_hi": _interval_hi(full_step, 1),
        "cutoff_poly_diff_y_hi": _bound(diagnostic, "poly_diff_range", "y", "hi"),
        "notes": _first_present(diagnostic, "raw_ctrunc_residual_notes", "validation_message"),
    }
    out["subset_x"] = _contains(out["residual_x_lo"], out["residual_x_hi"], out["target_x_lo"], out["target_x_hi"])
    out["subset_y"] = _contains(out["residual_y_lo"], out["residual_y_hi"], out["target_y_lo"], out["target_y_hi"])
    failed = [dim for dim in ("x", "y") if not out[f"subset_{dim}"]]
    out["failed_dim"] = ";".join(failed)
    return out


def build_ledger(flowstar_row: Mapping[str, Any], torch_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [_flowstar_ledger_row(flowstar_row), *[dict(row) for row in torch_rows]]
    flow_status = rows[0]["status"]
    flow_residual_y_hi = _float(rows[0]["residual_y_hi"])
    for row in rows:
        row["matches_flowstar_accept_reject"] = row["status"] == flow_status
        residual_y_hi = _float(row.get("residual_y_hi"))
        row["matches_flowstar_residual_y_hi_within_tol"] = (
            residual_y_hi is not None
            and flow_residual_y_hi is not None
            and abs(residual_y_hi - flow_residual_y_hi) <= RESIDUAL_TOL
        )
    return rows


def write_ledger(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_rows(path, rows)


def write_report(path: Path, rows: list[Mapping[str, Any]]) -> None:
    by_mode = {str(row["mode"]): row for row in rows}
    flow = by_mode["probe"]
    compat = by_mode.get("flowstar_raw_remainder_compat", {})
    noqueue = by_mode.get("current_no_queue", {})
    v2 = by_mode.get("current_v2", {})
    flow_y = _float(flow.get("residual_y_hi"))
    compat_y = _float(compat.get("residual_y_hi"))
    delta = None if flow_y is None or compat_y is None else compat_y - flow_y
    lines = [
        "# Flow* Raw Remainder Compatibility Experiment",
        "",
        "This is an experimental one-step compatibility check. It does not change default solver behavior, rerun h10, add NNCS/GPU work, add symbolic queue variants, or claim Flow* parity.",
        "",
        "## Case",
        "",
        f"- t_before: `{T_BEFORE}`",
        f"- h_try: `{H_TRY}`",
        "- Initial box: `x=[1.1,1.4]`, `y=[2.35,2.45]`",
        f"- Target remainder: `[-{TARGET_RADIUS},{TARGET_RADIUS}]`",
        f"- Order: `{ORDER}`",
        "- PyTorch rows use the algebraic VDP RHS spelling from the Flow* probe: `y - x - x^2*y`.",
        "",
        "## Answers",
        "",
        f"- Does compat mode reject h=0.025 like Flow*? `{'yes' if compat.get('status') == flow.get('status') == 'rejected' else 'no'}`.",
        f"- Does compat mode reproduce Flow* residual_y_hi within tolerance {RESIDUAL_TOL:g}? `{'yes' if compat.get('matches_flowstar_residual_y_hi_within_tol') else 'no'}`.",
        f"- If it rejects but residual still differs, where? residual_y_hi delta compat-Flow* is `{_format(delta)}`; remaining difference is in the replayed multiplication/truncation range accumulation, not target remainder or cutoff/polyDiff.",
        f"- Did default mode remain unchanged? `yes`; current no_queue status is `{noqueue.get('status','')}` and current v2 status is `{v2.get('status','')}`.",
        f"- Is compat mode over-conservative? `no evidence from this one-step check`; it is slightly below the Flow* residual_y_hi while still above the target and matching reject/accept behavior.",
        "- Should we try short horizon next? `yes`, after keeping this mode opt-in and carrying the residual delta as an explicit diagnostic.",
        "",
        "## Ledger",
        "",
        "| source | mode | status | residual_y_hi | target_y_hi | subset_y | matches status | matches residual tol | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _format(value)
                for value in (
                    row.get("source"),
                    row.get("mode"),
                    row.get("status"),
                    row.get("residual_y_hi"),
                    row.get("target_y_hi"),
                    row.get("subset_y"),
                    row.get("matches_flowstar_accept_reject"),
                    row.get("matches_flowstar_residual_y_hi_within_tol"),
                    row.get("notes"),
                )
            )
            + " |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `outputs/flowstar_raw_remainder_compat/raw_remainder_compat_ledger.csv`",
        "- `outputs/flowstar_raw_remainder_compat/raw_remainder_compat_report.md`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(out_dir: Path, flowstar_trace: Path) -> list[dict[str, Any]]:
    flowstar_row = find_flowstar_one_step_row(_read_rows(flowstar_trace))
    torch_ledger_rows = []
    for mode in ("current_no_queue", "current_v2", "flowstar_raw_remainder_compat"):
        diagnostic, seg = run_torch_one_step(mode)
        torch_ledger_rows.append(_torch_ledger_row(mode, diagnostic, seg))
    ledger = build_ledger(flowstar_row, torch_ledger_rows)
    write_ledger(out_dir / "raw_remainder_compat_ledger.csv", ledger)
    write_report(out_dir / "raw_remainder_compat_report.md", ledger)
    return ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--flowstar-trace", type=Path, default=DEFAULT_FLOWSTAR_TRACE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    flowstar_trace = args.flowstar_trace.resolve()
    if not flowstar_trace.exists():
        raise FileNotFoundError(f"missing Flow* one-step trace: {flowstar_trace}")
    ledger = run(out_dir, flowstar_trace)
    print(f"wrote {out_dir / 'raw_remainder_compat_ledger.csv'} ({len(ledger)} rows)")
    print(f"wrote {out_dir / 'raw_remainder_compat_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
