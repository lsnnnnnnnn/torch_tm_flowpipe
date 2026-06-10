#!/usr/bin/env python3
"""Audit Flow* raw-remainder compat adaptive step policy on short horizons."""
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

from torch_tm_flowpipe import FlowstarNormalFlowpipeState, Interval, TMVector, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.flowpipe import FLOWSTAR_COMPAT_STEP_GROW, FLOWSTAR_COMPAT_STEP_SHRINK
from flowstar_raw_remainder_compat_experiment import (  # noqa: E402
    DEFAULT_FLOWSTAR_TRACE,
    ORDER,
    TARGET_RADIUS,
    _float,
    _format,
    _read_rows,
    van_der_pol_flowstar_expression_ode,
)
from flowstar_raw_remainder_compat_short_horizon import (  # noqa: E402
    H_MAX,
    H_MIN,
    SAMPLES,
    _advance_sample,
    _interval_violation,
    _prefix_matches,
    _widths,
    flowstar_schedule_summary,
    schedule_distance,
)

DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_step_policy"

SUMMARY_FIELDS = [
    "source",
    "mode",
    "horizon",
    "status",
    "reached_t",
    "accepted_steps",
    "rejected_attempts",
    "accepted_h_sequence",
    "attempt_h_status_sequence",
    "schedule_distance_vs_flowstar",
    "schedule_prefix_matches_flowstar",
    "schedule_prefix_match_count",
    "sample_contained",
    "sample_max_violation",
    "stopped_too_early",
    "final_width_x",
    "final_width_y",
    "final_width_sum",
    "width_ratio_vs_current",
    "width_ratio_vs_compat_default",
    "step_shrink_factor",
    "step_grow_factor",
    "h5_justified",
    "recommendation",
    "notes",
]

MODE_SPECS: dict[str, dict[str, Any]] = {
    "current_no_queue_default_policy": {
        "validation_mode": "target_remainder_flowstar_ctrunc",
        "step_policy_mode": "",
        "grow_factor": 1.5,
    },
    "raw_remainder_compat_default_policy": {
        "validation_mode": "flowstar_raw_remainder_compat",
        "step_policy_mode": "",
        "grow_factor": 1.5,
    },
    "raw_remainder_compat_flowstar_step_policy": {
        "validation_mode": "flowstar_raw_remainder_compat",
        "step_policy_mode": "flowstar_compat",
        "grow_factor": 1.5,
    },
}


def _write_rows(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in fieldnames})


def flowstar_policy_next_after_failure(h: float) -> float:
    return float(h) * FLOWSTAR_COMPAT_STEP_SHRINK


def flowstar_policy_next_after_success(h: float, h_max: float = H_MAX) -> float:
    return min(float(h) * FLOWSTAR_COMPAT_STEP_GROW, float(h_max))


def _diag_status(row: Mapping[str, Any]) -> str:
    raw = str(row.get("validation_status", "")).strip().lower()
    if raw in {"failed", "failure", "rejected"}:
        return "rejected"
    if raw in {"validated", "accepted", "success", "passed"}:
        return "accepted"
    return raw or "unknown"


def _attempt_trace(diagnostics: Iterable[Mapping[str, Any]]) -> list[str]:
    out = []
    for row in diagnostics:
        h_try = _float(row.get("h_try"))
        out.append(f"{_format(h_try)}:{_diag_status(row)}")
    return out


def prefix_match_count(flow_h: list[float], candidate_h: list[float], tol: float = 1e-12) -> int:
    count = 0
    for expected, actual in zip(flow_h, candidate_h):
        if abs(expected - actual) > tol:
            break
        count += 1
    return count


def run_torch_policy_horizon(mode: str, horizon: float) -> dict[str, Any]:
    if mode not in MODE_SPECS:
        raise ValueError(f"unsupported mode: {mode}")
    spec = MODE_SPECS[mode]
    current: TMVector | list[Interval] = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state: FlowstarNormalFlowpipeState | None = None
    h_next = H_MAX
    t = 0.0
    accepted_h: list[float] = []
    rejected_attempts = 0
    attempt_trace: list[str] = []
    sample_points = list(SAMPLES)
    sample_contained = True
    sample_max_violation = 0.0
    last_final_tm: TMVector | None = None
    status = "completed"
    notes: list[str] = []

    while t < horizon - 1e-12:
        remaining = horizon - t
        h_try = min(h_next, H_MAX, remaining)
        local_h_min = min(H_MIN, h_try)
        diagnostics: list[dict[str, Any]] = []
        seg = flowpipe_step_flowstar_style_adaptive(
            van_der_pol_flowstar_expression_ode,
            current,
            h=h_try,
            h_min=local_h_min,
            h_max=H_MAX,
            order=ORDER,
            target_remainder_radius=TARGET_RADIUS,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_mode=str(spec["validation_mode"]),
            reset_mode="normalized_insertion",
            grow_factor=float(spec["grow_factor"]),
            step_policy_mode=str(spec["step_policy_mode"]),
            flowstar_normal_state=normal_state,
            diagnostics=diagnostics,
            diagnostics_context={"mode": mode, "segment_index": len(accepted_h), "t_before": t},
        )
        rejected_attempts += sum(1 for row in diagnostics if _diag_status(row) == "rejected")
        attempt_trace.extend(_attempt_trace(diagnostics))
        if seg.status != "validated" or seg.reset_tm is None:
            status = "stopped"
            notes.append(seg.message or "validation stopped")
            break

        accepted_h.append(float(seg.h))
        sample_points = [_advance_sample(point, float(seg.h)) for point in sample_points]
        boxes = seg.final_tm.range_box()
        for point in sample_points:
            sample_max_violation = max(
                sample_max_violation,
                _interval_violation(point[0], boxes[0]),
                _interval_violation(point[1], boxes[1]),
            )
        if sample_max_violation > 0.0:
            sample_contained = False
        t += float(seg.h)
        current = seg.reset_tm
        normal_state = seg.flowstar_normal_state
        h_next = float(seg.next_h if seg.next_h is not None else min(float(seg.h) * float(spec["grow_factor"]), H_MAX))
        last_final_tm = seg.final_tm

    width_x, width_y, width_sum = _widths(last_final_tm)
    step_grow = FLOWSTAR_COMPAT_STEP_GROW if spec["step_policy_mode"] == "flowstar_compat" else float(spec["grow_factor"])
    return {
        "source": "torch",
        "mode": mode,
        "horizon": horizon,
        "status": status,
        "reached_t": t,
        "accepted_steps": len(accepted_h),
        "rejected_attempts": rejected_attempts,
        "accepted_h_sequence": ";".join(_format(h) for h in accepted_h),
        "attempt_h_status_sequence": ";".join(attempt_trace),
        "schedule_distance_vs_flowstar": "",
        "schedule_prefix_matches_flowstar": "",
        "schedule_prefix_match_count": "",
        "sample_contained": sample_contained,
        "sample_max_violation": sample_max_violation,
        "stopped_too_early": t < horizon - 1e-12,
        "final_width_x": width_x,
        "final_width_y": width_y,
        "final_width_sum": width_sum,
        "width_ratio_vs_current": "",
        "width_ratio_vs_compat_default": "",
        "step_shrink_factor": FLOWSTAR_COMPAT_STEP_SHRINK,
        "step_grow_factor": step_grow,
        "h5_justified": "",
        "recommendation": "",
        "notes": "; ".join(notes) if notes else "endpoint samples from corners and center contained in PyTorch final segment boxes",
        "_accepted_h": accepted_h,
    }


def _finalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flow = next((row for row in rows if row.get("source") == "flowstar"), None)
    current = next((row for row in rows if row.get("mode") == "current_no_queue_default_policy"), None)
    compat_default = next((row for row in rows if row.get("mode") == "raw_remainder_compat_default_policy"), None)
    flow_h = list(flow.get("_accepted_h", [])) if flow else []
    current_width = _float(current.get("final_width_sum")) if current else None
    compat_width = _float(compat_default.get("final_width_sum")) if compat_default else None
    for row in rows:
        accepted_h = list(row.get("_accepted_h", []))
        dist = schedule_distance(flow_h, accepted_h)
        row["schedule_distance_vs_flowstar"] = dist if dist is not None else row.get("schedule_distance_vs_flowstar", "")
        row["schedule_prefix_matches_flowstar"] = _prefix_matches(flow_h, accepted_h) if accepted_h and flow_h else row.get("schedule_prefix_matches_flowstar", "")
        row["schedule_prefix_match_count"] = prefix_match_count(flow_h, accepted_h) if accepted_h and flow_h else row.get("schedule_prefix_match_count", "")
        width = _float(row.get("final_width_sum"))
        if current_width is not None and width is not None and current_width > 0:
            row["width_ratio_vs_current"] = width / current_width
        if compat_width is not None and width is not None and compat_width > 0:
            row["width_ratio_vs_compat_default"] = width / compat_width
        row.pop("_accepted_h", None)
    return rows


def _recommendation(rows: list[dict[str, Any]]) -> tuple[str, bool, str]:
    by_mode = {str(row.get("mode")): row for row in rows}
    compat_default = by_mode.get("raw_remainder_compat_default_policy", {})
    flow_policy = by_mode.get("raw_remainder_compat_flowstar_step_policy", {})
    default_distance = _float(compat_default.get("schedule_distance_vs_flowstar"))
    flow_distance = _float(flow_policy.get("schedule_distance_vs_flowstar"))
    width_ratio = _float(flow_policy.get("width_ratio_vs_compat_default"))
    improves_distance = flow_distance is not None and default_distance is not None and flow_distance < default_distance
    completed = flow_policy.get("status") == "completed" and not bool(flow_policy.get("stopped_too_early"))
    contained = bool(flow_policy.get("sample_contained"))
    material_width_increase = width_ratio is not None and width_ratio > 1.05
    h5_justified = completed and contained and improves_distance and not material_width_increase
    if h5_justified:
        return "flowstar_step_policy_h5_candidate", True, "Flow* step policy improves schedule distance without material width growth"
    return "continue_step_policy_audit_before_h5", False, "h5 remains gated until schedule distance, containment, and width all satisfy the audit gate"


def _analysis_rows(rows: list[dict[str, Any]], horizon: float) -> list[dict[str, Any]]:
    by_mode = {str(row.get("mode")): row for row in rows}
    compat_default = by_mode.get("raw_remainder_compat_default_policy", {})
    flow_policy = by_mode.get("raw_remainder_compat_flowstar_step_policy", {})
    recommendation, h5_justified, recommendation_notes = _recommendation(rows)
    default_distance = _float(compat_default.get("schedule_distance_vs_flowstar"))
    flow_distance = _float(flow_policy.get("schedule_distance_vs_flowstar"))
    distance_delta = None if default_distance is None or flow_distance is None else flow_distance - default_distance
    return [
        *rows,
        {
            "source": "analysis",
            "mode": "flowstar_step_policy_vs_compat_default",
            "horizon": horizon,
            "status": "improved" if distance_delta is not None and distance_delta < 0 else "not_improved",
            "reached_t": "",
            "accepted_steps": "",
            "rejected_attempts": "",
            "accepted_h_sequence": "",
            "attempt_h_status_sequence": "",
            "schedule_distance_vs_flowstar": "",
            "schedule_prefix_matches_flowstar": flow_policy.get("schedule_prefix_matches_flowstar", ""),
            "schedule_prefix_match_count": flow_policy.get("schedule_prefix_match_count", ""),
            "sample_contained": flow_policy.get("sample_contained", ""),
            "sample_max_violation": flow_policy.get("sample_max_violation", ""),
            "stopped_too_early": flow_policy.get("stopped_too_early", ""),
            "final_width_x": "",
            "final_width_y": "",
            "final_width_sum": flow_policy.get("final_width_sum", ""),
            "width_ratio_vs_current": flow_policy.get("width_ratio_vs_current", ""),
            "width_ratio_vs_compat_default": flow_policy.get("width_ratio_vs_compat_default", ""),
            "step_shrink_factor": FLOWSTAR_COMPAT_STEP_SHRINK,
            "step_grow_factor": FLOWSTAR_COMPAT_STEP_GROW,
            "h5_justified": h5_justified,
            "recommendation": recommendation,
            "notes": f"flowstar_policy_distance_minus_compat_default={_format(distance_delta)}",
        },
        {
            "source": "analysis",
            "mode": "h5_gate",
            "horizon": horizon,
            "status": "justified" if h5_justified else "not_justified",
            "reached_t": "",
            "accepted_steps": "",
            "rejected_attempts": "",
            "accepted_h_sequence": "",
            "attempt_h_status_sequence": "",
            "schedule_distance_vs_flowstar": flow_policy.get("schedule_distance_vs_flowstar", ""),
            "schedule_prefix_matches_flowstar": flow_policy.get("schedule_prefix_matches_flowstar", ""),
            "schedule_prefix_match_count": flow_policy.get("schedule_prefix_match_count", ""),
            "sample_contained": flow_policy.get("sample_contained", ""),
            "sample_max_violation": flow_policy.get("sample_max_violation", ""),
            "stopped_too_early": flow_policy.get("stopped_too_early", ""),
            "final_width_x": "",
            "final_width_y": "",
            "final_width_sum": flow_policy.get("final_width_sum", ""),
            "width_ratio_vs_current": flow_policy.get("width_ratio_vs_current", ""),
            "width_ratio_vs_compat_default": flow_policy.get("width_ratio_vs_compat_default", ""),
            "step_shrink_factor": FLOWSTAR_COMPAT_STEP_SHRINK,
            "step_grow_factor": FLOWSTAR_COMPAT_STEP_GROW,
            "h5_justified": h5_justified,
            "recommendation": recommendation,
            "notes": recommendation_notes,
        },
    ]


def run(out_dir: Path, flowstar_trace: Path, horizon: float) -> list[dict[str, Any]]:
    flow = flowstar_schedule_summary(_read_rows(flowstar_trace), horizon)
    rows = [
        flow,
        run_torch_policy_horizon("current_no_queue_default_policy", horizon),
        run_torch_policy_horizon("raw_remainder_compat_default_policy", horizon),
        run_torch_policy_horizon("raw_remainder_compat_flowstar_step_policy", horizon),
    ]
    rows = _analysis_rows(_finalize_rows(rows), horizon)
    write_summary(out_dir / "step_policy_summary.csv", rows)
    write_report(out_dir / "step_policy_report.md", rows, horizon)
    return rows


def write_summary(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_rows(path, SUMMARY_FIELDS, rows)


def write_report(path: Path, rows: list[Mapping[str, Any]], horizon: float) -> None:
    by_mode = {str(row.get("mode")): row for row in rows}
    current = by_mode.get("current_no_queue_default_policy", {})
    compat_default = by_mode.get("raw_remainder_compat_default_policy", {})
    flow_policy = by_mode.get("raw_remainder_compat_flowstar_step_policy", {})
    h5_gate = by_mode.get("h5_gate", {})
    current_dist = _float(current.get("schedule_distance_vs_flowstar"))
    compat_dist = _float(compat_default.get("schedule_distance_vs_flowstar"))
    flow_dist = _float(flow_policy.get("schedule_distance_vs_flowstar"))
    distance_improved = flow_dist is not None and compat_dist is not None and flow_dist < compat_dist
    width_ratio = _float(flow_policy.get("width_ratio_vs_compat_default"))
    width_material = width_ratio is not None and width_ratio > 1.05
    lines = [
        "# Flow* Raw Remainder Compat Step Policy",
        "",
        "This is an opt-in schedule-policy audit only. It does not run h5 or h10, add NNCS/GPU work, add symbolic queue variants, change defaults, or claim Flow* parity.",
        "",
        "## Audited Flow* Policy",
        "",
        f"- Rejected attempt shrink: `{FLOWSTAR_COMPAT_STEP_SHRINK}`.",
        f"- Accepted step grow: `{FLOWSTAR_COMPAT_STEP_GROW}`.",
        f"- Bounds used here: `h_min={H_MIN}`, `h_max={H_MAX}`.",
        "",
        "## Answers",
        "",
        f"- What is Flow* post-accept grow policy? `h_next = min(h * {FLOWSTAR_COMPAT_STEP_GROW}, h_max)`.",
        f"- Does flowstar_step_policy make the accepted h prefix match Flow* for T={horizon}? `{_format(flow_policy.get('schedule_prefix_matches_flowstar'))}`; prefix count `{_format(flow_policy.get('schedule_prefix_match_count'))}`.",
        f"- Does schedule distance improve vs compat default? `{'yes' if distance_improved else 'no'}`; compat default `{_format(compat_dist)}`, flowstar step policy `{_format(flow_dist)}`.",
        f"- Sample-contained? `{'yes' if flow_policy.get('sample_contained') else 'no'}`; max violation `{_format(flow_policy.get('sample_max_violation'))}`.",
        f"- Stop too early? `{'yes' if flow_policy.get('stopped_too_early') else 'no'}`.",
        f"- Width increase material? `{'yes' if width_material else 'no'}`; width ratio vs compat default `{_format(width_ratio)}`.",
        f"- Is h5 now justified? `{_format(h5_gate.get('h5_justified'))}`; recommendation `{_format(h5_gate.get('recommendation'))}`.",
        "",
        "## Summary",
        "",
        "| mode | status | reached_t | accepted_steps | rejected_attempts | schedule_distance | prefix_count | sample_contained | final_width_sum | width_ratio_vs_compat | recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _format(value)
                for value in (
                    row.get("mode"),
                    row.get("status"),
                    row.get("reached_t"),
                    row.get("accepted_steps"),
                    row.get("rejected_attempts"),
                    row.get("schedule_distance_vs_flowstar"),
                    row.get("schedule_prefix_match_count"),
                    row.get("sample_contained"),
                    row.get("final_width_sum"),
                    row.get("width_ratio_vs_compat_default"),
                    row.get("recommendation"),
                )
            )
            + " |"
        )
    lines.extend([
        "",
        "## Schedule Distances",
        "",
        f"- Current default policy: `{_format(current_dist)}`.",
        f"- Raw remainder compat default policy: `{_format(compat_dist)}`.",
        f"- Raw remainder compat Flow* step policy: `{_format(flow_dist)}`.",
        "",
        "## Outputs",
        "",
        "- `outputs/flowstar_raw_remainder_compat_step_policy/step_policy_summary.csv`",
        "- `outputs/flowstar_raw_remainder_compat_step_policy/step_policy_report.md`",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--flowstar-trace", type=Path, default=DEFAULT_FLOWSTAR_TRACE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.horizon > 1.0:
        raise ValueError("step-policy audit is capped at T=1.0 and must not be h5 or h10")
    out_dir = args.out_dir.resolve()
    flowstar_trace = args.flowstar_trace.resolve()
    if not flowstar_trace.exists():
        raise FileNotFoundError(f"missing Flow* trace: {flowstar_trace}")
    rows = run(out_dir, flowstar_trace, float(args.horizon))
    print(f"wrote {out_dir / 'step_policy_summary.csv'} ({len(rows)} rows)")
    print(f"wrote {out_dir / 'step_policy_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
