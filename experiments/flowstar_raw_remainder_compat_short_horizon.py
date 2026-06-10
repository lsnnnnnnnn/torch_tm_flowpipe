#!/usr/bin/env python3
"""Short-horizon Flow* raw-remainder compatibility diagnostic."""
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
from flowstar_raw_remainder_compat_experiment import (  # noqa: E402
    DEFAULT_FLOWSTAR_TRACE,
    ORDER,
    TARGET_RADIUS,
    _first_present,
    _float,
    _format,
    _read_rows,
    _status,
    van_der_pol_flowstar_expression_ode,
)

DEFAULT_OUT_DIR = ROOT / "outputs" / "flowstar_raw_remainder_compat_short_horizon"
H_MAX = 0.1
H_MIN = 0.002
SAMPLES = [
    (1.1, 2.35),
    (1.1, 2.45),
    (1.4, 2.35),
    (1.4, 2.45),
    (1.25, 2.4),
]

SUMMARY_FIELDS = [
    "source",
    "mode",
    "horizon",
    "status",
    "reached_t",
    "accepted_steps",
    "rejected_attempts",
    "accepted_h_sequence",
    "schedule_distance_vs_flowstar",
    "schedule_prefix_matches_flowstar",
    "sample_contained",
    "sample_max_violation",
    "stopped_too_early",
    "final_width_x",
    "final_width_y",
    "final_width_sum",
    "width_ratio_vs_current",
    "notes",
]


def _write_rows(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field, "")) for field in fieldnames})


def _trace_h(row: Mapping[str, Any]) -> float | None:
    return _float(_first_present(row, "h_try", "h"))


def _trace_t_after(row: Mapping[str, Any]) -> float | None:
    t_after = _float(row.get("t_after"))
    if t_after is not None:
        return t_after
    t_before = _float(row.get("t_before"))
    h = _trace_h(row)
    if t_before is None or h is None:
        return None
    return t_before + h


def flowstar_schedule_summary(rows: Iterable[Mapping[str, Any]], horizon: float) -> dict[str, Any]:
    accepted = []
    rejected = 0
    for row in rows:
        t_before = _float(row.get("t_before"))
        if t_before is None or t_before > horizon + 1e-12:
            continue
        status = _status(row)
        if status == "accepted":
            accepted.append(row)
        elif status == "rejected":
            rejected += 1
    accepted_h = [_trace_h(row) for row in accepted if _trace_h(row) is not None]
    reached_candidates = [_trace_t_after(row) for row in accepted]
    reached = max([value for value in reached_candidates if value is not None], default=0.0)
    return {
        "source": "flowstar",
        "mode": "probe_schedule",
        "horizon": horizon,
        "status": "available" if accepted else "missing",
        "reached_t": reached,
        "accepted_steps": len(accepted),
        "rejected_attempts": rejected,
        "accepted_h_sequence": ";".join(_format(h) for h in accepted_h),
        "schedule_distance_vs_flowstar": 0.0 if accepted else "",
        "schedule_prefix_matches_flowstar": True if accepted else "",
        "sample_contained": "",
        "sample_max_violation": "",
        "stopped_too_early": False if reached + 1e-12 >= horizon else True,
        "final_width_x": "",
        "final_width_y": "",
        "final_width_sum": "",
        "width_ratio_vs_current": "",
        "notes": "existing Flow* probe accepted schedule; no h10 rerun",
        "_accepted_h": accepted_h,
    }


def _vdp_rhs(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return y, y - x - x * x * y


def _rk4_step(point: tuple[float, float], dt: float) -> tuple[float, float]:
    x, y = point
    k1x, k1y = _vdp_rhs((x, y))
    k2x, k2y = _vdp_rhs((x + 0.5 * dt * k1x, y + 0.5 * dt * k1y))
    k3x, k3y = _vdp_rhs((x + 0.5 * dt * k2x, y + 0.5 * dt * k2y))
    k4x, k4y = _vdp_rhs((x + dt * k3x, y + dt * k3y))
    return (
        x + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0,
        y + dt * (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0,
    )


def _advance_sample(point: tuple[float, float], h: float) -> tuple[float, float]:
    pieces = max(4, int(math.ceil(abs(h) / 5e-4)))
    dt = h / pieces
    out = point
    for _ in range(pieces):
        out = _rk4_step(out, dt)
    return out


def _interval_violation(value: float, interval: Interval, tol: float = 1e-10) -> float:
    lo = float(interval.lo.detach().cpu())
    hi = float(interval.hi.detach().cpu())
    if value < lo - tol:
        return lo - value
    if value > hi + tol:
        return value - hi
    return 0.0


def _widths(final_tm: TMVector | None) -> tuple[float | None, float | None, float | None]:
    if final_tm is None:
        return None, None, None
    try:
        boxes = final_tm.range_box()
    except Exception:
        return None, None, None
    widths = [float(iv.width().detach().cpu()) for iv in boxes[:2]]
    if len(widths) < 2:
        return None, None, None
    return widths[0], widths[1], widths[0] + widths[1]


def run_torch_horizon(mode: str, horizon: float) -> dict[str, Any]:
    if mode not in {"current_no_queue", "flowstar_raw_remainder_compat"}:
        raise ValueError(f"unsupported mode: {mode}")
    validation_mode = "flowstar_raw_remainder_compat" if mode == "flowstar_raw_remainder_compat" else "target_remainder_flowstar_ctrunc"
    current: TMVector | list[Interval] = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state: FlowstarNormalFlowpipeState | None = None
    h_next = H_MAX
    t = 0.0
    accepted_h: list[float] = []
    rejected_attempts = 0
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
            h_max=max(h_try, local_h_min),
            order=ORDER,
            target_remainder_radius=TARGET_RADIUS,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_mode=validation_mode,
            reset_mode="normalized_insertion",
            grow_factor=1.5,
            flowstar_normal_state=normal_state,
            diagnostics=diagnostics,
            diagnostics_context={"mode": mode, "segment_index": len(accepted_h), "t_before": t},
        )
        rejected_attempts += sum(1 for row in diagnostics if row.get("validation_status") == "failed")
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
        h_next = float(seg.next_h if seg.next_h is not None else min(float(seg.h) * 1.5, H_MAX))
        last_final_tm = seg.final_tm

    width_x, width_y, width_sum = _widths(last_final_tm)
    return {
        "source": "torch",
        "mode": mode,
        "horizon": horizon,
        "status": status,
        "reached_t": t,
        "accepted_steps": len(accepted_h),
        "rejected_attempts": rejected_attempts,
        "accepted_h_sequence": ";".join(_format(h) for h in accepted_h),
        "schedule_distance_vs_flowstar": "",
        "schedule_prefix_matches_flowstar": "",
        "sample_contained": sample_contained,
        "sample_max_violation": sample_max_violation,
        "stopped_too_early": t < horizon - 1e-12,
        "final_width_x": width_x,
        "final_width_y": width_y,
        "final_width_sum": width_sum,
        "width_ratio_vs_current": "",
        "notes": "; ".join(notes) if notes else "endpoint samples from corners and center contained in PyTorch final segment boxes",
        "_accepted_h": accepted_h,
    }


def schedule_distance(flow_h: list[float], candidate_h: list[float]) -> float | None:
    if not flow_h or not candidate_h:
        return None
    total = 0.0
    for a, b in zip(flow_h, candidate_h):
        total += abs(a - b)
    if len(flow_h) > len(candidate_h):
        total += sum(abs(h) for h in flow_h[len(candidate_h):])
    elif len(candidate_h) > len(flow_h):
        total += sum(abs(h) for h in candidate_h[len(flow_h):])
    return total


def _prefix_matches(flow_h: list[float], candidate_h: list[float], tol: float = 1e-12) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(flow_h, candidate_h))


def finalize_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flow = next((row for row in rows if row.get("source") == "flowstar"), None)
    current = next((row for row in rows if row.get("mode") == "current_no_queue"), None)
    flow_h = list(flow.get("_accepted_h", [])) if flow else []
    current_width = _float(current.get("final_width_sum")) if current else None
    for row in rows:
        accepted_h = list(row.get("_accepted_h", []))
        dist = schedule_distance(flow_h, accepted_h)
        row["schedule_distance_vs_flowstar"] = dist if dist is not None else row.get("schedule_distance_vs_flowstar", "")
        row["schedule_prefix_matches_flowstar"] = _prefix_matches(flow_h, accepted_h) if accepted_h and flow_h else row.get("schedule_prefix_matches_flowstar", "")
        width = _float(row.get("final_width_sum"))
        if current_width is not None and width is not None and current_width > 0:
            row["width_ratio_vs_current"] = width / current_width
        row.pop("_accepted_h", None)
    return rows


def add_analysis_rows(rows: list[dict[str, Any]], horizon: float) -> list[dict[str, Any]]:
    by_mode = {str(row.get("mode")): row for row in rows}
    current = by_mode.get("current_no_queue", {})
    compat = by_mode.get("flowstar_raw_remainder_compat", {})
    current_dist = _float(current.get("schedule_distance_vs_flowstar"))
    compat_dist = _float(compat.get("schedule_distance_vs_flowstar"))
    compat_width_ratio = _float(compat.get("width_ratio_vs_current"))
    closer = compat_dist is not None and current_dist is not None and compat_dist < current_dist
    return [
        *rows,
        {
            "source": "analysis",
            "mode": "compat_vs_current",
            "horizon": horizon,
            "status": "closer_to_flowstar" if closer else "not_closer_to_flowstar",
            "reached_t": "",
            "accepted_steps": "",
            "rejected_attempts": "",
            "accepted_h_sequence": "",
            "schedule_distance_vs_flowstar": "",
            "schedule_prefix_matches_flowstar": "",
            "sample_contained": compat.get("sample_contained", ""),
            "sample_max_violation": compat.get("sample_max_violation", ""),
            "stopped_too_early": compat.get("stopped_too_early", ""),
            "final_width_x": "",
            "final_width_y": "",
            "final_width_sum": compat.get("final_width_sum", ""),
            "width_ratio_vs_current": compat_width_ratio if compat_width_ratio is not None else "",
            "notes": f"current_distance={_format(current_dist)}; compat_distance={_format(compat_dist)}",
        },
        {
            "source": "analysis",
            "mode": "step_policy_audit_gate",
            "horizon": horizon,
            "status": "required_before_h5",
            "reached_t": "",
            "accepted_steps": "",
            "rejected_attempts": "",
            "accepted_h_sequence": "",
            "schedule_distance_vs_flowstar": "",
            "schedule_prefix_matches_flowstar": "",
            "sample_contained": compat.get("sample_contained", ""),
            "sample_max_violation": compat.get("sample_max_violation", ""),
            "stopped_too_early": compat.get("stopped_too_early", ""),
            "final_width_x": "",
            "final_width_y": "",
            "final_width_sum": "",
            "width_ratio_vs_current": "",
            "notes": "audit Flow* accept-step growth policy before any h5 run",
        },
    ]


def run(out_dir: Path, flowstar_trace: Path, horizon: float) -> list[dict[str, Any]]:
    flow = flowstar_schedule_summary(_read_rows(flowstar_trace), horizon)
    current = run_torch_horizon("current_no_queue", horizon)
    compat = run_torch_horizon("flowstar_raw_remainder_compat", horizon)
    rows = add_analysis_rows(finalize_summary([flow, current, compat]), horizon)
    write_summary(out_dir / "short_horizon_summary.csv", rows)
    write_report(out_dir / "short_horizon_report.md", rows, horizon)
    return rows


def write_summary(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_rows(path, SUMMARY_FIELDS, rows)


def write_report(path: Path, rows: list[Mapping[str, Any]], horizon: float) -> None:
    by_mode = {str(row.get("mode")): row for row in rows}
    current = by_mode.get("current_no_queue", {})
    compat = by_mode.get("flowstar_raw_remainder_compat", {})
    current_dist = _float(current.get("schedule_distance_vs_flowstar"))
    compat_dist = _float(compat.get("schedule_distance_vs_flowstar"))
    closer = compat_dist is not None and current_dist is not None and compat_dist < current_dist
    compat_completed = compat.get("status") == "completed"
    compat_contained = bool(compat.get("sample_contained"))
    current_width = _float(current.get("final_width_sum"))
    compat_width = _float(compat.get("final_width_sum"))
    width_answer = "unknown" if current_width is None or compat_width is None else ("worsens" if compat_width > current_width else "improves_or_equal")
    next_step = "h5" if compat_completed and compat_contained and closer else "revise compat mechanism"
    lines = [
        "# Flow* Raw Remainder Compat Short Horizon",
        "",
        "This is a short-horizon diagnostic only. It does not run h10, add NNCS/GPU work, add symbolic queue variants, change defaults, or claim Flow* parity.",
        "",
        "## Scope",
        "",
        f"- Horizon: `{horizon}`",
        "- PyTorch ODE spelling: `y - x - x^2*y`",
        "- Sample containment: endpoint checks for four initial corners and the center using RK4 samples.",
        "",
        "## Answers",
        "",
        f"- Does compat follow Flow* accepted h schedule more closely than current? `{'yes' if closer else 'no'}`.",
        f"- Does compat remain sample-contained? `{'yes' if compat_contained else 'no'}`.",
        f"- Does compat become too conservative and stop too early? `{'yes' if compat.get('stopped_too_early') else 'no'}`.",
        f"- Does compat improve or worsen width relative to current? `{width_answer}`; ratio `{_format(compat.get('width_ratio_vs_current'))}`.",
        f"- Should next step be h5, or should compat mechanism be revised? `{next_step}`.",
        "",
        "## Summary",
        "",
        "| mode | status | reached_t | accepted_steps | rejected_attempts | schedule_distance | sample_contained | final_width_sum | width_ratio | notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
                    row.get("sample_contained"),
                    row.get("final_width_sum"),
                    row.get("width_ratio_vs_current"),
                    row.get("notes"),
                )
            )
            + " |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        "- `outputs/flowstar_raw_remainder_compat_short_horizon/short_horizon_summary.csv`",
        "- `outputs/flowstar_raw_remainder_compat_short_horizon/short_horizon_report.md`",
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
        raise ValueError("short-horizon diagnostic is capped at T=1.0 and must not be h10")
    out_dir = args.out_dir.resolve()
    flowstar_trace = args.flowstar_trace.resolve()
    if not flowstar_trace.exists():
        raise FileNotFoundError(f"missing Flow* trace: {flowstar_trace}")
    rows = run(out_dir, flowstar_trace, args.horizon)
    print(f"wrote {out_dir / 'short_horizon_summary.csv'} ({len(rows)} rows)")
    print(f"wrote {out_dir / 'short_horizon_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
