#!/usr/bin/env python3
"""Join fresh G1 runs with the historical Flow* width ledger."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


CHANNEL_FIELDS = {
    "endpoint_x": ("endpoint_x_lo", "endpoint_x_hi", "endpoint_x_width"),
    "endpoint_y": ("endpoint_y_lo", "endpoint_y_hi", "endpoint_y_width"),
    "segment_tube_x": ("segment_x_lo", "segment_x_hi", "segment_x_width"),
    "segment_tube_y": ("segment_y_lo", "segment_y_hi", "segment_y_width"),
}
THRESHOLDS = (1.1, 1.5, 2.0, 5.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summaries(root: Path) -> list[dict[str, Any]]:
    return [load(path) for path in sorted(root.glob("*/summary.json"))]


def request_rows(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for value in values:
        rows.append(
            {
                "reset_mode": value["reset_mode"],
                "requested_horizon": value["requested_horizon"],
                "schedule": value["schedule"]["kind"],
                "status": value["status"],
                "completed_horizon": value["completed_horizon"],
                "accepted_steps": value["accepted_steps"],
                "rejected_steps": value["rejected_step_records"],
                "runtime_s": value["runtime_s"],
                "peak_rss_bytes": value["peak_rss_bytes"],
                "endpoint_x_width": (value.get("raw_endpoint") or {}).get("x_width"),
                "endpoint_y_width": (value.get("raw_endpoint") or {}).get("y_width"),
                "segment_x_width": (value.get("last_segment") or {}).get("x_width"),
                "segment_y_width": (value.get("last_segment") or {}).get("y_width"),
                "fallback_count": value["fallback_count"],
                "message": value["message"],
            }
        )
    return sorted(rows, key=lambda row: (float(row["requested_horizon"]), row["reset_mode"]))


def find_attempt(rows: list[dict[str, str]], segment_index: int) -> dict[str, str]:
    matches = [
        row for row in rows
        if int(row.get("segment_index", -1)) == segment_index
        and row.get("validation_status") == "validated"
    ]
    if not matches:
        raise ValueError(f"no validated attempt for segment {segment_index}")
    return matches[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width-ledger", type=Path, required=True)
    parser.add_argument("--candidate-fixed", type=Path, required=True)
    parser.add_argument("--fixed-smoke-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--extended-root", type=Path, required=True)
    parser.add_argument("--legacy-segments", type=Path, required=True)
    parser.add_argument("--legacy-attempts", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    historical = read_csv(args.width_ledger)
    candidate_segments = read_csv(args.candidate_fixed / "segments.csv")
    if len(candidate_segments) != 632:
        raise ValueError("candidate fixed trace must contain 632 accepted rows")
    curve: list[dict[str, Any]] = []
    previous_candidate_width: dict[str, float] = {}
    previous_candidate_excess: dict[str, float] = {}
    for base in historical:
        step = int(base["step"])
        segment = candidate_segments[step - 1]
        channel = base["channel"]
        lo_field, hi_field, width_field = CHANNEL_FIELDS[channel]
        candidate_lo = float(segment[lo_field])
        candidate_hi = float(segment[hi_field])
        candidate_width = float(segment[width_field])
        flow_width = float(base["flowstar_width"])
        candidate_excess = candidate_width - flow_width
        width_increment = candidate_width - previous_candidate_width.get(channel, 0.0)
        excess_increment = candidate_excess - previous_candidate_excess.get(channel, 0.0)
        previous_candidate_width[channel] = candidate_width
        previous_candidate_excess[channel] = candidate_excess
        curve.append(
            {
                "step": step,
                "time": float(base["time"]),
                "channel": channel,
                "flowstar_lo": float(base["flowstar_lo"]),
                "flowstar_hi": float(base["flowstar_hi"]),
                "flowstar_width": flow_width,
                "legacy_lo": float(base["torch_lo"]),
                "legacy_hi": float(base["torch_hi"]),
                "legacy_width": float(base["torch_width"]),
                "legacy_excess": float(base["absolute_excess"]),
                "legacy_ratio": float(base["relative_ratio"]),
                "legacy_width_increment": float(base["torch_width_increment"]),
                "legacy_excess_increment": float(base["excess_increment"]),
                "candidate_lo": candidate_lo,
                "candidate_hi": candidate_hi,
                "candidate_width": candidate_width,
                "candidate_excess": candidate_excess,
                "candidate_ratio": candidate_width / flow_width,
                "candidate_width_increment": width_increment,
                "candidate_excess_increment": excess_increment,
                "candidate_width_reduction_vs_legacy": float(base["torch_width"]) - candidate_width,
            }
        )
    write_csv(args.output_dir / "fixed_width_curve.csv", curve)

    crossings: list[dict[str, Any]] = []
    for channel in CHANNEL_FIELDS:
        rows = [row for row in curve if row["channel"] == channel]
        for threshold in THRESHOLDS:
            for mode in ("legacy", "candidate"):
                crossing = next((row for row in rows if row[f"{mode}_ratio"] > threshold), None)
                crossings.append(
                    {
                        "channel": channel,
                        "threshold": threshold,
                        "mode": mode,
                        "first_step": crossing["step"] if crossing else "",
                        "first_time": crossing["time"] if crossing else "",
                        "ratio": crossing[f"{mode}_ratio"] if crossing else "",
                    }
                )
    write_csv(args.output_dir / "ratio_crossings.csv", crossings)

    checkpoint_rows = [
        row for row in curve if row["step"] in {1, 2, 50, 100, 200, 300, 632}
    ]
    write_csv(args.output_dir / "checkpoint_widths.csv", checkpoint_rows)

    fixed_values = summaries(args.fixed_smoke_root) + summaries(args.fixed_root)
    fixed_requests = request_rows(fixed_values)
    write_csv(args.output_dir / "fresh_fixed_requests.csv", fixed_requests)
    native_values = summaries(args.native_root) + [
        value for value in summaries(args.extended_root)
        if value["schedule"]["kind"] == "adaptive"
    ]
    native_requests = request_rows(native_values)
    write_csv(args.output_dir / "fresh_native_requests.csv", native_requests)

    legacy_segments = read_csv(args.legacy_segments)
    legacy_attempts = read_csv(args.legacy_attempts)
    candidate_attempts = read_csv(args.candidate_fixed / "attempts.csv")
    legacy_last = legacy_segments[631]
    candidate_last = candidate_segments[631]
    acceleration = {
        "time": 6.32,
        "legacy_prestate_center": json.loads(legacy_last["prestate_center"]),
        "candidate_prestate_center": json.loads(candidate_last["prestate_center"]),
        "legacy_prestate_scale": json.loads(legacy_last["prestate_scale"]),
        "candidate_prestate_scale": json.loads(candidate_last["prestate_scale"]),
        "legacy_raw_picard": {
            "lo": json.loads(find_attempt(legacy_attempts, 631)["picard_image_remainder_lo"]),
            "hi": json.loads(find_attempt(legacy_attempts, 631)["picard_image_remainder_hi"]),
            "subset_margin": json.loads(find_attempt(legacy_attempts, 631)["subset_margin"]),
        },
        "candidate_raw_picard": {
            "lo": json.loads(find_attempt(candidate_attempts, 631)["picard_image_remainder_lo"]),
            "hi": json.loads(find_attempt(candidate_attempts, 631)["picard_image_remainder_hi"]),
            "subset_margin": json.loads(find_attempt(candidate_attempts, 631)["subset_margin"]),
        },
        "candidate_boundary_ordinary_width_mass": float(candidate_last["carry_source_ledger_ordinary_width_mass"]),
        "candidate_boundary_structured_width_mass": float(candidate_last["carry_source_ledger_structured_width_mass"]),
        "candidate_live_source_count": int(candidate_last["carry_source_ledger_live_source_count"]),
        "candidate_collapse_count": int(candidate_last["carry_source_ledger_collapse_count"]),
        "dominant_candidate_boundary_mass": "ordinary_parameterization_and_retired_nonlinear_source_collapse",
    }

    performance = load(args.performance)
    cpu_short = load(args.fixed_smoke_root / "normalized_insertion_bounded_source_ledger_o4_g1" / "summary.json")
    cuda_short = load(args.cuda_root / "summary.json")
    cpu_profile = read_csv(args.fixed_smoke_root / "normalized_insertion_bounded_source_ledger_o4_g1" / "profile.csv")
    summed_steps = sum(float(row["total_wall_s"]) for row in cpu_profile)
    phase_timing = {
        "actual_cpu_b1": performance["actual_b1_phase_timing"],
        "adaptive_outer_loop_and_nontrace_overhead_s_T0p1": (
            cpu_short["runtime_s"] - cpu_short["trace_io_s"] - summed_steps
        ),
        "sum_step_wall_s_T0p1": summed_steps,
        "trace_io_s_T0p1": cpu_short["trace_io_s"],
        "cpu_full_T0p1_runtime_s": cpu_short["runtime_s"],
        "cuda_full_T0p1_runtime_s": cuda_short["runtime_s"],
        "cuda_over_cpu_full_solver_speedup": cpu_short["runtime_s"] / cuda_short["runtime_s"],
        "cuda_device_transfer_count": cuda_short["device_transfer_count"],
        "cuda_full_solver_speedup_claimed": False,
    }
    (args.output_dir / "phase_timing.json").write_text(
        json.dumps(phase_timing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checkpoints: dict[str, Any] = {}
    for step, label in ((100, "T1"), (300, "T3"), (632, "T6p32")):
        rows = [row for row in checkpoint_rows if row["step"] == step]
        checkpoints[label] = {
            row["channel"]: {
                "flowstar_width": row["flowstar_width"],
                "legacy_width": row["legacy_width"],
                "candidate_width": row["candidate_width"],
                "legacy_excess": row["legacy_excess"],
                "candidate_excess": row["candidate_excess"],
                "candidate_width_reduction_vs_legacy": row["candidate_width_reduction_vs_legacy"],
            }
            for row in rows
        }
    threshold_shifted = any(
        left["first_time"] != right["first_time"]
        for left, right in zip(crossings[::2], crossings[1::2])
    )
    candidate_native = [row for row in native_requests if row["reset_mode"].endswith("o4_g1")]
    legacy_native = [row for row in native_requests if row["reset_mode"] == "normalized_insertion"]
    summary = {
        "schema": "bounded_source_ledger_causal_run_summary_v1",
        "conclusion": "T1_T3_WIDTH_CAUSE_CLOSED__EARLY_GAP_IMPROVED__TERMINAL_STILL_OPEN",
        "checkpoints": checkpoints,
        "ratio_crossing_time_shifted_at_0p01_resolution": threshold_shifted,
        "fixed_candidate_completed_T6p32": load(args.candidate_fixed / "summary.json")["completed_requested_horizon"],
        "native_candidate_highest_continuously_validated_time": max(row["completed_horizon"] for row in candidate_native),
        "native_legacy_highest_continuously_validated_time": max(row["completed_horizon"] for row in legacy_native),
        "native_candidate_terminal_time": min(row["completed_horizon"] for row in candidate_native if row["status"] != "completed"),
        "native_legacy_terminal_time": min(row["completed_horizon"] for row in legacy_native if row["status"] != "completed"),
        "acceleration_checkpoint": acceleration,
        "source_policy": {
            "live_source_count": 2,
            "generations": 1,
            "fixed_boundary_variables": 4,
            "collapse_count_at_fixed_T6p32": int(candidate_last["carry_source_ledger_collapse_count"]),
            "fallback_count": 0,
        },
        "performance": phase_timing,
    }
    (args.output_dir / "scientific_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "conclusion": summary["conclusion"],
        "threshold_shifted": threshold_shifted,
        "candidate_terminal": summary["native_candidate_terminal_time"],
        "legacy_terminal": summary["native_legacy_terminal_time"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
