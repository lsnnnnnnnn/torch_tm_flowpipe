#!/usr/bin/env python3
"""Summarize the clean-SHA VDP H2 scientific matrix and fail gates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FLOWSTAR_LEDGER = (
    ROOT
    / "outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z"
    / "04_causal_runs/checkpoint_widths.csv"
)
SCIENTIFIC_SHA = "666c51ecc5575f203518d21f34b5c9948741fb17"
LANES = ("legacy", "h1", "h1_h2")
HORIZONS = (1.0, 3.0, 6.32)
CHANNELS = ("endpoint_x", "endpoint_y", "segment_x", "segment_y")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _widths(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "endpoint_x": float(row["endpoint_x_width"]),
        "endpoint_y": float(row["endpoint_y_width"]),
        "segment_x": float(row["segment_x_width"]),
        "segment_y": float(row["segment_y_width"]),
    }


def _fixed_checkpoints(evidence: Path, lane: str) -> dict[float, dict[str, float]]:
    result: dict[float, dict[str, float]] = {}
    path = evidence / "fixed_T6p32" / lane / "segments.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t_hi = float(row["t_hi"])
            for horizon in HORIZONS:
                if abs(t_hi - horizon) <= 1.0e-12:
                    result[horizon] = _widths(row)
    if set(result) != set(HORIZONS):
        raise ValueError(f"missing fixed checkpoint in {path}")
    return result


def _flowstar_widths() -> dict[float, dict[str, float]]:
    channel_map = {
        "endpoint_x": "endpoint_x",
        "endpoint_y": "endpoint_y",
        "segment_tube_x": "segment_x",
        "segment_tube_y": "segment_y",
    }
    result: dict[float, dict[str, float]] = {horizon: {} for horizon in HORIZONS}
    with FLOWSTAR_LEDGER.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            time_value = float(row["time"])
            if time_value in result and row["channel"] in channel_map:
                result[time_value][channel_map[row["channel"]]] = float(row["flowstar_width"])
    if any(set(widths) != set(CHANNELS) for widths in result.values()):
        raise ValueError("authoritative Flow* ledger is missing a checkpoint channel")
    return result


def _summary_row(path: Path) -> dict[str, Any]:
    row = _load(path / "summary.json")
    if row["commit"] != SCIENTIFIC_SHA or row["worktree_dirty"] is not False:
        raise ValueError(f"non-clean scientific run: {path}")
    return row


def _published_widths(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "endpoint_x": float(summary["raw_endpoint"]["x_width"]),
        "endpoint_y": float(summary["raw_endpoint"]["y_width"]),
        "segment_x": float(summary["last_segment"]["x_width"]),
        "segment_y": float(summary["last_segment"]["y_width"]),
    }


def _compact_run(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "completed_horizon",
        "completed_requested_horizon",
        "accepted_steps",
        "rejected_attempts",
        "rejected_step_records",
        "failure_type",
        "message",
        "runtime_s",
        "peak_rss_bytes",
        "dense_kernel_s",
        "nonkernel_nontransfer_solver_s",
        "device",
        "reset_mode",
        "validation_mode",
        "commit",
        "tracked_diff_sha256",
        "worktree_dirty",
    )
    return {key: summary[key] for key in keys}


def _native_rejection_diagnostic(evidence: Path, lane: str) -> dict[str, Any]:
    path = evidence / "native_T10" / lane / "attempts.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        failed = [
            row
            for row in csv.DictReader(handle)
            if row["validation_status"].lower() == "failed"
        ]
    if not failed:
        raise ValueError(f"native lane has no failed validation attempts: {lane}")
    row = failed[-1]
    image_lo = json.loads(row["picard_image_remainder_lo"])[0]
    image_hi = json.loads(row["picard_image_remainder_hi"])[0]
    target_lo = json.loads(row["candidate_remainder_lo"])[0]
    target_hi = json.loads(row["candidate_remainder_hi"])[0]
    margins = json.loads(row["subset_margin"])[0]
    component = min(range(len(margins)), key=lambda index: margins[index])
    upper_overrun = float(image_hi[component]) - float(target_hi[component])
    lower_overrun = float(target_lo[component]) - float(image_lo[component])
    limiting_side = "upper" if upper_overrun >= lower_overrun else "lower"
    ledger = json.loads(row["validated_remainder_ledger_intervals"])
    widths = {
        category: float(interval["width"][0][component])
        for category, interval in ledger.items()
    }
    largest_category = max(widths, key=widths.__getitem__)
    return {
        "failed_attempt_count": len(failed),
        "last_failed_segment_index": int(row["segment_index"]),
        "last_failed_t_before": float(row["t_before"]),
        "last_failed_h_try": float(row["h_try"]),
        "limiting_component": ("x", "y")[component],
        "limiting_side": limiting_side,
        "subset_margin": float(margins[component]),
        "upper_overrun": upper_overrun,
        "lower_overrun": lower_overrun,
        "image_interval": [float(image_lo[component]), float(image_hi[component])],
        "target_interval": [float(target_lo[component]), float(target_hi[component])],
        "largest_additive_validated_ledger_category": largest_category,
        "largest_additive_validated_ledger_width": widths[largest_category],
        "validated_ledger_widths_for_limiting_component": widths,
        "rejection_reason": row["rejection_reason"],
    }


def summarize(evidence: Path) -> dict[str, Any]:
    evidence = evidence.resolve()
    gate = _load(evidence / "gates/summary.json")
    flow_step1 = _load(evidence / "gates/flowstar_runtime_crosscheck.json")
    fixed = {lane: _fixed_checkpoints(evidence, lane) for lane in LANES}
    flow_fixed = _flowstar_widths()

    checkpoint_rows: list[dict[str, Any]] = []
    fixed_matrix: dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_key = f"T{format(horizon, 'g').replace('.', 'p')}"
        channel_rows: dict[str, Any] = {}
        for channel in CHANNELS:
            flow = flow_fixed[horizon][channel]
            legacy = fixed["legacy"][horizon][channel]
            h1 = fixed["h1"][horizon][channel]
            candidate = fixed["h1_h2"][horizon][channel]
            excess = legacy - flow
            h1_reduction = legacy - h1
            candidate_reduction = legacy - candidate
            incremental_h2 = h1 - candidate
            row = {
                "flowstar_width": flow,
                "legacy_width": legacy,
                "h1_width": h1,
                "h1_h2_width": candidate,
                "legacy_excess": excess,
                "h1_fraction_of_legacy_excess_removed": h1_reduction / excess,
                "h1_h2_fraction_of_legacy_excess_removed": candidate_reduction / excess,
                "incremental_h2_fraction_of_legacy_excess_removed": incremental_h2 / excess,
                "h1_h2_no_wider_than_legacy": candidate <= legacy,
                "h1_h2_no_wider_than_h1": candidate <= h1,
                "h1_h2_meets_10pct": candidate_reduction / excess >= 0.10,
            }
            channel_rows[channel] = row
            checkpoint_rows.append({"horizon": horizon, "channel": channel, **row})
        fixed_matrix[horizon_key] = channel_rows

    step1_torch = {
        lane: _summary_row(evidence / "step1" / lane)
        for lane in LANES
    }
    step1 = {
        "legacy": _published_widths(step1_torch["legacy"]),
        "h1": _published_widths(step1_torch["h1"]),
        "h1_h2": _published_widths(step1_torch["h1_h2"]),
        "flowstar": {
            "endpoint_x": flow_step1["endpoint"][0]["width"],
            "endpoint_y": flow_step1["endpoint"][1]["width"],
            "segment_x": flow_step1["segment"][0]["width"],
            "segment_y": flow_step1["segment"][1]["width"],
        },
        "raw_residual_excess": gate["raw_residual_excess"],
        "torch_runs": {lane: _compact_run(row) for lane, row in step1_torch.items()},
        "flowstar_role": flow_step1["role"],
        "flowstar_source_commit": flow_step1["source_commit"],
    }

    fixed_summaries = {
        lane: _summary_row(evidence / "fixed_T6p32" / lane)
        for lane in LANES
    }
    native_summaries = {
        lane: _summary_row(evidence / "native_T10" / lane)
        for lane in LANES
    }
    native_rejections = {
        lane: _native_rejection_diagnostic(evidence, lane)
        for lane in LANES
    }
    cpu_t01 = {
        lane: _summary_row(evidence / "cpu_T0p1" / lane)
        for lane in LANES
    }
    v100_t01 = {
        lane: _summary_row(evidence / "v100_T0p1" / lane)
        for lane in LANES
    }
    consistency: dict[str, Any] = {}
    for lane in LANES:
        cpu_widths = _published_widths(cpu_t01[lane])
        cuda_widths = _published_widths(v100_t01[lane])
        deltas = {channel: abs(cpu_widths[channel] - cuda_widths[channel]) for channel in CHANNELS}
        consistency[lane] = {
            "scope": "implementation consistency only; V100 is not a directed-rounding soundness lane",
            "width_abs_deltas": deltas,
            "max_width_abs_delta": max(deltas.values()),
            "status_equal": cpu_t01[lane]["status"] == v100_t01[lane]["status"],
            "accepted_steps_equal": cpu_t01[lane]["accepted_steps"] == v100_t01[lane]["accepted_steps"],
            "consistent_at_1e_12": (
                max(deltas.values()) <= 1.0e-12
                and cpu_t01[lane]["status"] == v100_t01[lane]["status"]
                and cpu_t01[lane]["accepted_steps"] == v100_t01[lane]["accepted_steps"]
            ),
        }

    early_gate = all(
        fixed_matrix[horizon][channel]["h1_h2_meets_10pct"]
        for horizon in ("T1", "T3")
        for channel in CHANNELS
    )
    t6_gate = all(
        fixed_matrix["T6p32"][channel]["h1_h2_no_wider_than_h1"]
        for channel in CHANNELS
    )
    native_floor = float(native_summaries["h1_h2"]["completed_horizon"]) >= 6.441433080631058
    runtime_ratios = {
        "fixed_T6p32_h1_h2_over_legacy": (
            float(fixed_summaries["h1_h2"]["runtime_s"])
            / float(fixed_summaries["legacy"]["runtime_s"])
        ),
        "native_T10_request_h1_h2_over_legacy": (
            float(native_summaries["h1_h2"]["runtime_s"])
            / float(native_summaries["legacy"]["runtime_s"])
        ),
    }
    runtime_gate = all(value <= 2.0 for value in runtime_ratios.values())
    gates = {
        "gate_a_exact_operator_ledger": gate["gate_a_pass"],
        "gate_b_same_input_operator": gate["gate_b_pass"],
        "T1_T3_all_four_channels_remove_10pct_legacy_excess": early_gate,
        "T6p32_no_channel_regression_vs_H1": t6_gate,
        "native_at_least_6p441433080631058": native_floor,
        "runtime_at_most_2x_legacy": runtime_gate,
        "reaches_T10_stretch": bool(native_summaries["h1_h2"]["completed_requested_horizon"]),
        "v100_all_lanes_measured": all(row["device"] == "cuda" for row in v100_t01.values()),
        "cpu_v100_consistent_at_1e_12": all(row["consistent_at_1e_12"] for row in consistency.values()),
    }
    production_targets = (
        early_gate,
        t6_gate,
        native_floor,
        runtime_gate,
    )
    matrix = {
        "schema": "vdp_h2_dense_picard_scientific_matrix_v1",
        "scientific_sha": SCIENTIFIC_SHA,
        "step1": step1,
        "fixed": fixed_matrix,
        "fixed_T6p32_runs": {lane: _compact_run(row) for lane, row in fixed_summaries.items()},
        "native_T10_requests": {lane: _compact_run(row) for lane, row in native_summaries.items()},
        "native_rejection_diagnostics": native_rejections,
        "runtime_ratios": runtime_ratios,
        "cpu_T0p1": {lane: _compact_run(row) for lane, row in cpu_t01.items()},
        "v100_T0p1": {lane: _compact_run(row) for lane, row in v100_t01.items()},
        "cpu_v100_consistency_T0p1": consistency,
        "gates": gates,
        "decision": (
            "H2_OPERATOR_ACCEPTED__ALL_PRODUCTION_TARGETS_MET"
            if all(production_targets)
            else "H2_OPERATOR_ACCEPTED__OVERALL_SUCCESS_TARGET_FAILED"
        ),
    }

    matrix_dir = evidence / "matrix"
    if matrix_dir.exists() and any(matrix_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty matrix directory: {matrix_dir}")
    matrix_dir.mkdir(parents=True, exist_ok=True)
    _write_json(matrix_dir / "matrix.json", matrix)
    with (matrix_dir / "checkpoint_widths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    print(json.dumps({"decision": matrix["decision"], "gates": gates}, sort_keys=True))
    return matrix


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    summarize(parse_args().evidence_root)
