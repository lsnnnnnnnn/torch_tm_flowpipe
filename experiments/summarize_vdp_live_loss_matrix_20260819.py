#!/usr/bin/env python3
"""Summarize the clean-SHA VDP live-loss/C1 scientific matrix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_SHA = "dbe03dcdfbf2f36b1d58013373d1d235ace1a48e"
H2_MATRIX = ROOT / (
    "evidence/vdp_h2_dense_picard_first_loss/20260818T091126Z/"
    "02_scientific_matrix/matrix.json"
)
FLOWSTAR_LEDGER = ROOT / (
    "outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z/"
    "04_causal_runs/checkpoint_widths.csv"
)
HORIZONS = (1.0, 3.0, 6.32)
CHANNELS = ("endpoint_x", "endpoint_y", "segment_x", "segment_y")
LANES = ("legacy", "h1", "h1_h2", "candidate")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _summary(path: Path) -> dict[str, Any]:
    row = _load(path / "summary.json")
    if row["commit"] != SCIENTIFIC_SHA:
        raise ValueError(f"scientific SHA mismatch: {path}")
    if row["worktree_dirty"] is not False:
        raise ValueError(f"dirty scientific run: {path}")
    if row["tracked_diff_sha256"] != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
        raise ValueError(f"non-empty tracked diff: {path}")
    return row


def _published_widths(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "endpoint_x": float(summary["raw_endpoint"]["x_width"]),
        "endpoint_y": float(summary["raw_endpoint"]["y_width"]),
        "segment_x": float(summary["last_segment"]["x_width"]),
        "segment_y": float(summary["last_segment"]["y_width"]),
    }


def _compact(summary: Mapping[str, Any]) -> dict[str, Any]:
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
        "endpoint_repair_used",
        "endpoint_tightening_used",
    )
    return {key: summary[key] for key in keys}


def _fixed_checkpoints(path: Path) -> dict[float, dict[str, float]]:
    result: dict[float, dict[str, float]] = {}
    with (path / "segments.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t_hi = float(row["t_hi"])
            for horizon in HORIZONS:
                if abs(t_hi - horizon) <= 1.0e-12:
                    result[horizon] = {
                        "endpoint_x": float(row["endpoint_x_width"]),
                        "endpoint_y": float(row["endpoint_y_width"]),
                        "segment_x": float(row["segment_x_width"]),
                        "segment_y": float(row["segment_y_width"]),
                    }
    if set(result) != set(HORIZONS):
        raise ValueError("candidate fixed run lacks a required checkpoint")
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
        raise ValueError("authoritative Flow* ledger lacks a required checkpoint")
    return result


def _native_failure(path: Path) -> dict[str, Any]:
    with (path / "attempts.csv").open(newline="", encoding="utf-8") as handle:
        failed = [
            row for row in csv.DictReader(handle)
            if row["validation_status"].lower() == "failed"
        ]
    if not failed:
        raise ValueError("native request has no failed attempt")
    row = failed[-1]
    image_lo = json.loads(row["picard_image_remainder_lo"])[0]
    image_hi = json.loads(row["picard_image_remainder_hi"])[0]
    target_lo = json.loads(row["candidate_remainder_lo"])[0]
    target_hi = json.loads(row["candidate_remainder_hi"])[0]
    margins = json.loads(row["subset_margin"])[0]
    component = min(range(len(margins)), key=lambda index: margins[index])
    upper_overrun = float(image_hi[component]) - float(target_hi[component])
    lower_overrun = float(target_lo[component]) - float(image_lo[component])
    ledger = json.loads(row["validated_remainder_ledger_intervals"])
    widths = {
        category: float(interval["width"][0][component])
        for category, interval in ledger.items()
    }
    largest = max(widths, key=widths.__getitem__)
    return {
        "failed_attempt_count": len(failed),
        "last_failed_segment_index": int(row["segment_index"]),
        "last_failed_t_before": float(row["t_before"]),
        "last_failed_h_try": float(row["h_try"]),
        "limiting_component": ("x", "y")[component],
        "limiting_side": "upper" if upper_overrun >= lower_overrun else "lower",
        "subset_margin": float(margins[component]),
        "upper_overrun": upper_overrun,
        "lower_overrun": lower_overrun,
        "image_interval": [float(image_lo[component]), float(image_hi[component])],
        "target_interval": [float(target_lo[component]), float(target_hi[component])],
        "largest_additive_ledger_category": largest,
        "largest_additive_ledger_width": widths[largest],
        "additive_ledger_widths": widths,
        "rejection_reason": row["rejection_reason"],
        "causal_warning": (
            "additive category size is not a same-input causal marginal; the closure owns "
            "the complete joint residual under composition_overflow"
        ),
    }


def summarize(matrix_root: Path) -> dict[str, Any]:
    matrix_root = matrix_root.resolve()
    historical = _load(H2_MATRIX)
    gate = _load(matrix_root / "gates/summary.json")
    production = _load(matrix_root / "gates/production_operator_ledger.json")
    step1_runs = {lane: _summary(matrix_root / "step1" / lane) for lane in LANES}
    step1_widths = {lane: _published_widths(row) for lane, row in step1_runs.items()}
    historical_step1 = historical["step1"]
    defaults_bitwise = all(
        step1_widths[lane] == historical_step1[lane]
        for lane in ("legacy", "h1", "h1_h2")
    )
    if not defaults_bitwise:
        raise ValueError("legacy/H1/H2 step1 widths changed")

    candidate_checkpoints = _fixed_checkpoints(matrix_root / "fixed_T6p32/candidate")
    flowstar = _flowstar_widths()
    fixed: dict[str, Any] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        key = f"T{format(horizon, 'g').replace('.', 'p')}"
        fixed[key] = {}
        for channel in CHANNELS:
            old = historical["fixed"][key][channel]
            legacy = float(old["legacy_width"])
            h1 = float(old["h1_width"])
            h2 = float(old["h1_h2_width"])
            candidate = candidate_checkpoints[horizon][channel]
            flow = flowstar[horizon][channel]
            excess = legacy - flow
            row = {
                "flowstar_width": flow,
                "legacy_width": legacy,
                "h1_width": h1,
                "h1_h2_width": h2,
                "candidate_width": candidate,
                "legacy_excess": excess,
                "candidate_reduction": legacy - candidate,
                "fraction_of_legacy_excess_removed": (legacy - candidate) / excess,
                "candidate_no_wider_than_h1_h2": candidate <= h2,
                "meets_10pct": (legacy - candidate) / excess >= 0.10,
            }
            fixed[key][channel] = row
            checkpoint_rows.append({"horizon": horizon, "channel": channel, **row})

    fixed_runs = {
        lane: _summary(matrix_root / f"fixed_T6p32/{lane}")
        for lane in ("legacy", "candidate")
    }
    native_runs = {
        lane: _summary(matrix_root / f"native_T10/{lane}")
        for lane in ("legacy", "candidate")
    }
    consistency_runs = {
        device: _summary(matrix_root / f"consistency_T0p1/{device}")
        for device in ("cpu", "cuda")
    }
    consistency_widths = {
        device: _published_widths(row) for device, row in consistency_runs.items()
    }
    consistency_deltas = {
        channel: abs(consistency_widths["cpu"][channel] - consistency_widths["cuda"][channel])
        for channel in CHANNELS
    }
    consistency_pass = (
        consistency_runs["cpu"]["status"] == consistency_runs["cuda"]["status"]
        and consistency_runs["cpu"]["accepted_steps"] == consistency_runs["cuda"]["accepted_steps"]
        and consistency_runs["cpu"]["rejected_attempts"] == consistency_runs["cuda"]["rejected_attempts"]
        and max(consistency_deltas.values()) <= 1.0e-12
    )
    runtime_ratios = {
        "fixed_T6p32_candidate_over_legacy": (
            fixed_runs["candidate"]["runtime_s"] / fixed_runs["legacy"]["runtime_s"]
        ),
        "native_T10_request_candidate_over_legacy": (
            native_runs["candidate"]["runtime_s"] / native_runs["legacy"]["runtime_s"]
        ),
    }
    early_gate = all(
        fixed[horizon][channel]["meets_10pct"]
        for horizon in ("T1", "T3")
        for channel in CHANNELS
    )
    t6_gate = all(
        fixed["T6p32"][channel]["candidate_no_wider_than_h1_h2"]
        for channel in CHANNELS
    )
    native_gate = native_runs["candidate"]["completed_horizon"] >= 6.482041958201616
    runtime_gate = all(value <= 2.0 for value in runtime_ratios.values())
    stretch = bool(native_runs["candidate"]["completed_requested_horizon"])
    gates = {
        "gate_a": gate["gate_a_pass"],
        "gate_b": gate["gate_b_pass"],
        "gate_c": gate["gate_c_pass"],
        "legacy_h1_h2_step1_bitwise_unchanged": defaults_bitwise,
        "T1_T3_all_four_channels_remove_10pct_legacy_excess": early_gate,
        "T6p32_no_channel_regression_vs_current_h1_h2": t6_gate,
        "native_at_least_6p482041958201616": native_gate,
        "runtime_at_most_2x_legacy": runtime_gate,
        "v100_candidate_consistent_at_1e_12": consistency_pass,
        "reaches_T10_stretch": stretch,
    }
    causal = production["live_loss"]
    result = {
        "schema": "vdp_live_loss_c1_scientific_matrix_v1",
        "scientific_sha": SCIENTIFIC_SHA,
        "step1": {
            "widths": step1_widths,
            "runs": {lane: _compact(row) for lane, row in step1_runs.items()},
            "legacy_h1_h2_match_historical_bitwise": defaults_bitwise,
        },
        "fixed": fixed,
        "fixed_runs": {lane: _compact(row) for lane, row in fixed_runs.items()},
        "native_T10_requests": {lane: _compact(row) for lane, row in native_runs.items()},
        "native_candidate_terminal_diagnostic": _native_failure(
            matrix_root / "native_T10/candidate"
        ),
        "runtime_ratios": runtime_ratios,
        "v100_consistency": {
            "scope": (
                "implementation consistency only; V100 is not a directed-rounding "
                "soundness or speedup lane"
            ),
            "width_abs_deltas": consistency_deltas,
            "max_width_abs_delta": max(consistency_deltas.values()),
            "consistent_at_1e_12": consistency_pass,
            "runs": {device: _compact(row) for device, row in consistency_runs.items()},
        },
        "concept_separation": {
            "syntactic_first": causal["first_syntactic_strict_surplus"]["stage_id"],
            "first_live": causal["first_live_strict_surplus"]["stage_id"],
            "first_material": causal["first_live_material_surplus"]["stage_id"],
            "largest_causal_marginal_contributor": causal[
                "largest_same_input_marginal_contributor"
            ]["stage_id"],
            "same_input_marginal_scope": "Gate B byte-identical step1 counterfactuals only",
            "largest_additive_ledger_category_scope": (
                "native terminal additive ownership only; not a causal ranking"
            ),
        },
        "gates": gates,
        "decision": (
            "C1_SOUND_AND_PRODUCTION_USEFUL__OVERALL_T1_T3_SUCCESS_FAILED__T10_STRETCH_FAILED"
            if all((gate["gate_a_pass"], gate["gate_b_pass"], gate["gate_c_pass"], t6_gate, native_gate, runtime_gate))
            and not early_gate
            and not stretch
            else "C1_MATRIX_REQUIRES_MANUAL_REVIEW"
        ),
    }
    _write_json(matrix_root / "matrix.json", result)
    with (matrix_root / "checkpoint_widths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    print(json.dumps({"decision": result["decision"], "gates": gates}, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    summarize(parse_args().matrix_root)
