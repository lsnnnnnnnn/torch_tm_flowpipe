#!/usr/bin/env python3
"""Compare Flow* and opt-in raw-remainder compat on the first adaptive attempts."""
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
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from flowstar_raw_remainder_compat_experiment import (  # noqa: E402
    DEFAULT_FLOWSTAR_TRACE,
    ORDER,
    RESIDUAL_TOL,
    TARGET_RADIUS,
    T_BEFORE,
    _bound,
    _contains,
    _first_present,
    _float,
    _format,
    _read_rows,
    _status,
    van_der_pol_flowstar_expression_ode,
)

DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_attempt_sequence"
H_ATTEMPTS = [0.1, 0.05, 0.025, 0.0125]

LEDGER_FIELDS = [
    "source",
    "mode",
    "attempt_index",
    "t_before",
    "h_try",
    "status",
    "residual_x_lo",
    "residual_x_hi",
    "residual_y_lo",
    "residual_y_hi",
    "target_y_hi",
    "subset_y",
    "failed_dim",
    "matches_flowstar_status",
    "residual_y_hi_delta_vs_flowstar",
    "residual_y_hi_abs_delta_vs_flowstar",
    "overconservative_vs_flowstar",
    "notes",
]


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in LEDGER_FIELDS})


def _find_flowstar_attempts(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    for h_try in H_ATTEMPTS:
        matches = []
        for row in rows:
            h = _float(_first_present(row, "h_try", "h"))
            t = _float(row.get("t_before"))
            if h is None or t is None:
                continue
            if abs(h - h_try) <= 1e-12 and abs(t - T_BEFORE) <= 1e-6:
                matches.append(row)
        if not matches:
            raise RuntimeError(f"no Flow* trace row found for first-attempt h={h_try}")
        found.append(matches[0])
    return found


def _failed_dim(row: Mapping[str, Any]) -> str:
    failed = []
    if not bool(row.get("subset_y")):
        failed.append("y")
    return ";".join(failed)


def _flowstar_row(attempt_index: int, h_try: float, row: Mapping[str, Any]) -> dict[str, Any]:
    residual_prefix = "picard_ctrunc_normal_residual"
    out = {
        "source": "flowstar",
        "mode": "probe",
        "attempt_index": attempt_index,
        "t_before": T_BEFORE,
        "h_try": h_try,
        "status": _status(row),
        "residual_x_lo": _bound(row, residual_prefix, "x", "lo"),
        "residual_x_hi": _bound(row, residual_prefix, "x", "hi"),
        "residual_y_lo": _bound(row, residual_prefix, "y", "lo"),
        "residual_y_hi": _bound(row, residual_prefix, "y", "hi"),
        "target_y_hi": _bound(row, "target_remainder", "y", "hi"),
        "notes": "existing Flow* first-attempt probe trace; no h10 rerun",
    }
    out["subset_y"] = _contains(
        out["residual_y_lo"],
        out["residual_y_hi"],
        _bound(row, "target_remainder", "y", "lo"),
        out["target_y_hi"],
    )
    out["failed_dim"] = _failed_dim(out)
    return out


def run_torch_attempt(mode: str, h_try: float) -> tuple[dict[str, Any], Any]:
    if mode not in {"current_no_queue", "flowstar_raw_remainder_compat"}:
        raise ValueError(f"unsupported mode: {mode}")
    validation_mode = "flowstar_raw_remainder_compat" if mode == "flowstar_raw_remainder_compat" else "target_remainder_flowstar_ctrunc"
    diagnostics: list[dict[str, Any]] = []
    seg = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_flowstar_expression_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=h_try,
        h_min=h_try,
        h_max=h_try,
        order=ORDER,
        target_remainder_radius=TARGET_RADIUS,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_mode=validation_mode,
        reset_mode="normalized_insertion",
        grow_factor=1.0,
        diagnostics=diagnostics,
        diagnostics_context={"mode": mode, "segment_index": 0, "t_before": T_BEFORE},
    )
    if not diagnostics:
        raise RuntimeError(f"no diagnostics captured for {mode} h={h_try}")
    return diagnostics[-1], seg


def _torch_row(attempt_index: int, h_try: float, mode: str, diagnostic: Mapping[str, Any], seg: Any) -> dict[str, Any]:
    residual_prefix = "flowstar_raw_remainder_compat_check_remainder" if mode == "flowstar_raw_remainder_compat" else "tmp_remainder"
    out = {
        "source": "torch",
        "mode": mode,
        "attempt_index": attempt_index,
        "t_before": T_BEFORE,
        "h_try": h_try,
        "status": "accepted" if getattr(seg, "status", "") == "validated" else "rejected",
        "residual_x_lo": _bound(diagnostic, residual_prefix, "x", "lo"),
        "residual_x_hi": _bound(diagnostic, residual_prefix, "x", "hi"),
        "residual_y_lo": _bound(diagnostic, residual_prefix, "y", "lo"),
        "residual_y_hi": _bound(diagnostic, residual_prefix, "y", "hi"),
        "target_y_hi": _bound(diagnostic, "target_remainder_before_ctrunc", "y", "hi"),
        "notes": _first_present(diagnostic, "raw_ctrunc_residual_notes", "validation_message"),
    }
    out["subset_y"] = _contains(
        out["residual_y_lo"],
        out["residual_y_hi"],
        _bound(diagnostic, "target_remainder_before_ctrunc", "y", "lo"),
        out["target_y_hi"],
    )
    out["failed_dim"] = _failed_dim(out)
    return out


def mark_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_attempt: dict[int, dict[str, Any]] = {
        int(row["attempt_index"]): row for row in rows if row.get("source") == "flowstar"
    }
    for row in rows:
        flow = by_attempt.get(int(row["attempt_index"]))
        if flow is None:
            continue
        row["matches_flowstar_status"] = row["status"] == flow["status"]
        row_y = _float(row.get("residual_y_hi"))
        flow_y = _float(flow.get("residual_y_hi"))
        if row_y is not None and flow_y is not None:
            delta = row_y - flow_y
            row["residual_y_hi_delta_vs_flowstar"] = delta
            row["residual_y_hi_abs_delta_vs_flowstar"] = abs(delta)
        else:
            row["residual_y_hi_delta_vs_flowstar"] = ""
            row["residual_y_hi_abs_delta_vs_flowstar"] = ""
        row["overconservative_vs_flowstar"] = detect_overconservative(row, flow)
    return rows


def detect_overconservative(row: Mapping[str, Any], flowstar_row: Mapping[str, Any]) -> bool:
    return str(flowstar_row.get("status")) == "accepted" and str(row.get("status")) == "rejected"


def build_ledger(flowstar_rows: Iterable[Mapping[str, Any]], torch_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (h_try, flow_row) in enumerate(zip(H_ATTEMPTS, flowstar_rows), start=1):
        rows.append(_flowstar_row(index, h_try, flow_row))
    rows.extend(dict(row) for row in torch_rows)
    return mark_comparisons(rows)


def run(out_dir: Path, flowstar_trace: Path) -> list[dict[str, Any]]:
    flowstar_rows = _find_flowstar_attempts(_read_rows(flowstar_trace))
    torch_rows: list[dict[str, Any]] = []
    for index, h_try in enumerate(H_ATTEMPTS, start=1):
        for mode in ("current_no_queue", "flowstar_raw_remainder_compat"):
            diagnostic, seg = run_torch_attempt(mode, h_try)
            torch_rows.append(_torch_row(index, h_try, mode, diagnostic, seg))
    ledger = build_ledger(flowstar_rows, torch_rows)
    write_ledger(out_dir / "attempt_sequence_ledger.csv", ledger)
    write_report(out_dir / "attempt_sequence_report.md", ledger)
    return ledger


def write_ledger(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_rows(path, rows)


def _sequence(rows: Iterable[Mapping[str, Any]], mode: str) -> list[str]:
    return [str(row.get("status", "")) for row in sorted(rows, key=lambda r: int(r.get("attempt_index", 0))) if row.get("mode") == mode]


def _residual_close_all(rows: Iterable[Mapping[str, Any]], mode: str, tol: float = RESIDUAL_TOL) -> bool:
    selected = [row for row in rows if row.get("mode") == mode]
    return bool(selected) and all((_float(row.get("residual_y_hi_abs_delta_vs_flowstar")) or math.inf) <= tol for row in selected)


def write_report(path: Path, rows: list[Mapping[str, Any]]) -> None:
    flow_seq = _sequence(rows, "probe")
    current_seq = _sequence(rows, "current_no_queue")
    compat_seq = _sequence(rows, "flowstar_raw_remainder_compat")
    h0125 = [row for row in rows if row.get("mode") == "flowstar_raw_remainder_compat" and abs(float(row.get("h_try")) - 0.0125) <= 1e-15]
    compat_h0125 = h0125[0] if h0125 else {}
    current_h025 = [row for row in rows if row.get("mode") == "current_no_queue" and abs(float(row.get("h_try")) - 0.025) <= 1e-15]
    current_h025_row = current_h025[0] if current_h025 else {}
    flow_h0125 = [row for row in rows if row.get("mode") == "probe" and abs(float(row.get("h_try")) - 0.0125) <= 1e-15]
    flow_h0125_row = flow_h0125[0] if flow_h0125 else {}
    lines = [
        "# Flow* Raw Remainder Compat Attempt Sequence",
        "",
        "This is a first-adaptive-attempt diagnostic only. It does not run h10, add NNCS/GPU work, add symbolic queue variants, change defaults, or claim Flow* parity.",
        "",
        "## Answers",
        "",
        f"- Does compat reproduce Flow* attempt status sequence? `{'yes' if compat_seq == flow_seq else 'no'}`.",
        f"- Flow* sequence: `{flow_seq}`.",
        f"- Current PyTorch sequence: `{current_seq}`.",
        f"- Compat sequence: `{compat_seq}`.",
        f"- Does compat reject h=0.1, h=0.05, h=0.025 and accept h=0.0125 if Flow* does? `{'yes' if compat_seq == ['rejected', 'rejected', 'rejected', 'accepted'] and flow_seq == compat_seq else 'no'}`.",
        f"- Does current mode diverge at h=0.025 as before? `{'yes' if current_h025_row.get('status') == 'accepted' else 'no'}`.",
        f"- Is compat residual close to Flow* at all attempted h? `{'yes' if _residual_close_all(rows, 'flowstar_raw_remainder_compat') else 'no'}` using tolerance `{RESIDUAL_TOL}`.",
        f"- Is compat over-conservative at h=0.0125 or does it also accept? `{'overconservative' if detect_overconservative(compat_h0125, flow_h0125_row) else compat_h0125.get('status', 'missing')}`.",
        "- Should we proceed to T=0.5 short horizon? `yes`, with the same opt-in mode and explicit residual-delta reporting.",
        "",
        "## Ledger",
        "",
        "| attempt | h | mode | status | residual_y_hi | delta_vs_flowstar | overconservative | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _format(value)
                for value in (
                    row.get("attempt_index"),
                    row.get("h_try"),
                    row.get("mode"),
                    row.get("status"),
                    row.get("residual_y_hi"),
                    row.get("residual_y_hi_delta_vs_flowstar"),
                    row.get("overconservative_vs_flowstar"),
                    row.get("notes"),
                )
            )
            + " |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `outputs/flowstar_raw_remainder_compat_attempt_sequence/attempt_sequence_ledger.csv`",
        "- `outputs/flowstar_raw_remainder_compat_attempt_sequence/attempt_sequence_report.md`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        raise FileNotFoundError(f"missing Flow* trace: {flowstar_trace}")
    ledger = run(out_dir, flowstar_trace)
    print(f"wrote {out_dir / 'attempt_sequence_ledger.csv'} ({len(ledger)} rows)")
    print(f"wrote {out_dir / 'attempt_sequence_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
