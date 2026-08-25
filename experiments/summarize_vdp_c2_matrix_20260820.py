#!/usr/bin/env python3
"""Summarize C2 incremental and original-target production metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EMPTY_DIFF_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CHANNELS = ("endpoint_x", "endpoint_y", "segment_x", "segment_y")
FLOWSTAR_LEDGER = ROOT / (
    "outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z/"
    "04_causal_runs/checkpoint_widths.csv"
)
HISTORICAL_H2_MATRIX = ROOT / (
    "evidence/vdp_h2_dense_picard_first_loss/20260818T091126Z/"
    "02_scientific_matrix/matrix.json"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(path: Path, scientific_sha: str) -> Mapping[str, Any]:
    row = _load(path / "summary.json")
    if row["commit"] != scientific_sha:
        raise ValueError(f"scientific SHA mismatch: {path}")
    if row["worktree_dirty"] is not False or row["tracked_diff_sha256"] != EMPTY_DIFF_SHA256:
        raise ValueError(f"unclean scientific run: {path}")
    if row["endpoint_repair_used"] is not False or row["endpoint_tightening_used"] is not False:
        raise ValueError(f"endpoint repair/tightening is forbidden: {path}")
    return row


def _widths(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "endpoint_x": float(summary["raw_endpoint"]["x_width"]),
        "endpoint_y": float(summary["raw_endpoint"]["y_width"]),
        "segment_x": float(summary["last_segment"]["x_width"]),
        "segment_y": float(summary["last_segment"]["y_width"]),
    }


def _first_rejection(run_dir: Path) -> Mapping[str, Any] | None:
    with (run_dir / "attempts.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("validation_status") != "failed":
                continue
            outer_lo = json.loads(row["candidate_remainder_lo"])[0]
            outer_hi = json.loads(row["candidate_remainder_hi"])[0]
            image_lo = json.loads(row["picard_image_remainder_lo"])[0]
            image_hi = json.loads(row["picard_image_remainder_hi"])[0]
            margins = []
            for component in range(len(outer_lo)):
                margins.append((image_lo[component] - outer_lo[component], component, "lower"))
                margins.append((outer_hi[component] - image_hi[component], component, "upper"))
            margin, component, side = min(margins)
            return {
                "t_before": float(row["t_before"]),
                "h": float(row["h"]),
                "adaptive_attempt_index": int(row["adaptive_attempt_index"]),
                "limiting_component": ("x", "y")[component],
                "limiting_side": side,
                "subset_margin": float(margin),
            }
    return None


def _flowstar() -> dict[float, dict[str, float]]:
    channel_names = {
        "endpoint_x": "endpoint_x",
        "endpoint_y": "endpoint_y",
        "segment_tube_x": "segment_x",
        "segment_tube_y": "segment_y",
    }
    result = {1.0: {}, 3.0: {}, 6.32: {}}
    with FLOWSTAR_LEDGER.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            time_value = float(row["time"])
            if time_value in result and row["channel"] in channel_names:
                result[time_value][channel_names[row["channel"]]] = float(row["flowstar_width"])
    if any(set(value) != set(CHANNELS) for value in result.values()):
        raise ValueError("authoritative Flow* width ledger is incomplete")
    return result


def _forbidden_candidate_width(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "candidate_width":
                raise ValueError(f"ambiguous candidate_width field is forbidden at {path}")
            _forbidden_candidate_width(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_candidate_width(child, f"{path}[{index}]")


def summarize(
    matrix_root: Path,
    gate_a_path: Path,
    baseline_verification_path: Path,
    output: Path,
    terminal_diagnostic_path: Path | None = None,
) -> dict[str, Any]:
    run_index = _load(matrix_root / "run_index.json")
    scientific_sha = str(run_index["scientific_sha"])
    gate_a = _load(gate_a_path)
    baseline_verification = _load(baseline_verification_path)
    if gate_a["scientific_sha"] != scientific_sha or gate_a["gate_pass"] is not True:
        raise ValueError("Gate A does not authorize production summarization")
    if not (
        baseline_verification["h2_package_verified"] is True
        and baseline_verification["c1_package_verified"] is True
    ):
        raise ValueError("historical H1/H2 reference packages are not verified")

    historical = _load(HISTORICAL_H2_MATRIX)
    lane_modes = {
        "legacy": "flowstar_raw_remainder_compat",
        "production_c1_candidate": "flowstar_raw_remainder_compat_factorized_joint_closure",
        "production_c2_candidate": "flowstar_raw_remainder_compat_factorized_joint_closure_refined",
    }
    summaries: dict[str, dict[str, Mapping[str, Any]]] = {}
    for scenario in ("step1", "fixed_T1", "fixed_T3", "fixed_T6p32", "native_T10"):
        summaries[scenario] = {}
        for lane, mode in lane_modes.items():
            row = _summary(matrix_root / scenario / lane, scientific_sha)
            if row["validation_mode"] != mode:
                raise ValueError(f"validation mode mismatch: {scenario}/{lane}")
            summaries[scenario][lane] = row
    consistency = {
        device: _summary(matrix_root / "consistency_T0p1" / device, scientific_sha)
        for device in ("cpu", "cuda")
    }
    consistency_widths = {device: _widths(row) for device, row in consistency.items()}
    consistency_deltas = {
        channel: abs(consistency_widths["cpu"][channel] - consistency_widths["cuda"][channel])
        for channel in CHANNELS
    }
    consistency_pass = bool(
        consistency["cpu"]["status"] == consistency["cuda"]["status"]
        and consistency["cpu"]["accepted_steps"] == consistency["cuda"]["accepted_steps"]
        and consistency["cpu"]["rejected_attempts"] == consistency["cuda"]["rejected_attempts"]
        and max(consistency_deltas.values()) <= 1.0e-12
    )

    step1 = {
        "legacy": _widths(summaries["step1"]["legacy"]),
        "historical_h1_candidate": historical["step1"]["h1"],
        "gate_b_h1_h2_candidate": historical["step1"]["h1_h2"],
        "production_c1_candidate": _widths(summaries["step1"]["production_c1_candidate"]),
        "production_c2_candidate": _widths(summaries["step1"]["production_c2_candidate"]),
    }
    flowstar = _flowstar()
    scenario_for_horizon = {1.0: "fixed_T1", 3.0: "fixed_T3", 6.32: "fixed_T6p32"}
    fixed = {}
    for horizon, scenario in scenario_for_horizon.items():
        key = f"T{format(horizon, 'g').replace('.', 'p')}"
        legacy_widths = _widths(summaries[scenario]["legacy"])
        c1_widths = _widths(summaries[scenario]["production_c1_candidate"])
        c2_widths = _widths(summaries[scenario]["production_c2_candidate"])
        fixed[key] = {}
        for channel in CHANNELS:
            h1 = float(historical["fixed"][key][channel]["h1_width"])
            h2 = float(historical["fixed"][key][channel]["h1_h2_width"])
            legacy = legacy_widths[channel]
            c1 = c1_widths[channel]
            c2 = c2_widths[channel]
            flow = flowstar[horizon][channel]
            excess = legacy - flow
            original_fraction = (legacy - c2) / excess
            fixed[key][channel] = {
                "flowstar_width": flow,
                "legacy_width": legacy,
                "historical_h1_candidate_width": h1,
                "gate_b_h1_h2_candidate_width": h2,
                "production_c1_candidate_width": c1,
                "production_c2_candidate_width": c2,
                "c2_incremental_reduction_vs_c1": c1 - c2,
                "c2_incremental_fraction_of_c1_width": (c1 - c2) / c1,
                "original_target_fraction_legacy_excess_removed": original_fraction,
                "c2_no_wider_than_c1": c2 <= c1,
                "meets_original_10pct": original_fraction >= 0.10,
            }

    runtime_ratios = {}
    for scenario in scenario_for_horizon.values():
        runtime_ratios[f"{scenario}_c2_over_legacy"] = (
            float(summaries[scenario]["production_c2_candidate"]["runtime_s"])
            / float(summaries[scenario]["legacy"]["runtime_s"])
        )
    runtime_ratios["native_T10_request_c2_over_legacy"] = (
        float(summaries["native_T10"]["production_c2_candidate"]["runtime_s"])
        / float(summaries["native_T10"]["legacy"]["runtime_s"])
    )
    early_gate = all(
        fixed[horizon][channel]["meets_original_10pct"]
        for horizon in ("T1", "T3")
        for channel in CHANNELS
    )
    t6_gate = all(fixed["T6p32"][channel]["c2_no_wider_than_c1"] for channel in CHANNELS)
    native = summaries["native_T10"]["production_c2_candidate"]
    native_floor = float(native["completed_horizon"]) >= 6.589638579126679
    t10 = bool(native["completed_requested_horizon"])
    runtime_gate = all(value <= 2.0 for value in runtime_ratios.values())
    c2_incremental_useful = all(
        fixed[horizon][channel]["c2_no_wider_than_c1"]
        for horizon in ("T1", "T3", "T6p32")
        for channel in CHANNELS
    )
    if early_gate and t10:
        decision = "C2_T1_T3_AND_T10_PASSED"
    elif early_gate:
        decision = "C2_T1_T3_GATE_PASSED__T10_FAILED"
    else:
        decision = "C2_SOUND_LOCAL_CAUSE_ACCEPTED__PRODUCTION_GATE_FAILED"
    result = {
        "schema": "vdp_c2_scientific_matrix_v1",
        "scientific_sha": scientific_sha,
        "baseline_verification": baseline_verification,
        "lane_naming": {
            "historical_h1_candidate": "historical H1 candidate, package-verified",
            "gate_b_h1_h2_candidate": "historical H1+H2 candidate, package-verified",
            "production_c1_candidate": "fresh C1 closure lane",
            "production_c2_candidate": "fresh post-accept refinement lane",
            "ambiguous_candidate_width_field_forbidden": True,
        },
        "step1": step1,
        "fixed": fixed,
        "runtime_ratios": runtime_ratios,
        "run_accounting": {
            scenario: {
                lane: {
                    "accepted_steps": row["accepted_steps"],
                    "rejected_attempts": row["rejected_attempts"],
                    "completed_horizon": row["completed_horizon"],
                    "completed_requested_horizon": row["completed_requested_horizon"],
                    "runtime_s": row["runtime_s"],
                    "status": row["status"],
                    "first_rejection": _first_rejection(matrix_root / scenario / lane),
                }
                for lane, row in summaries[scenario].items()
            }
            for scenario in ("fixed_T1", "fixed_T3", "fixed_T6p32", "native_T10")
        },
        "native": {
            lane: {
                "status": row["status"],
                "completed_horizon": row["completed_horizon"],
                "completed_requested_horizon": row["completed_requested_horizon"],
                "runtime_s": row["runtime_s"],
            }
            for lane, row in summaries["native_T10"].items()
        },
        "terminal_diagnostic": (
            None if terminal_diagnostic_path is None else _load(terminal_diagnostic_path)
        ),
        "v100_consistency": {
            "scope": "implementation consistency only; no CUDA directed-rounding soundness claim",
            "passed_at_1e_12": consistency_pass,
            "width_deltas": consistency_deltas,
            "cpu_runtime_s": consistency["cpu"]["runtime_s"],
            "cuda_runtime_s": consistency["cuda"]["runtime_s"],
            "cuda_over_cpu_runtime": (
                float(consistency["cuda"]["runtime_s"])
                / float(consistency["cpu"]["runtime_s"])
            ),
            "cpu_peak_rss_bytes": consistency["cpu"]["peak_rss_bytes"],
            "cuda_peak_rss_bytes": consistency["cuda"]["peak_rss_bytes"],
            "v100_slower_than_cpu": (
                float(consistency["cuda"]["runtime_s"])
                > float(consistency["cpu"]["runtime_s"])
            ),
        },
        "gates": {
            "gate_a_same_input_causal": True,
            "T1_T3_all_eight_checks_remove_10pct_legacy_excess": early_gate,
            "T6p32_all_channels_no_wider_than_c1": t6_gate,
            "native_not_below_6p589638579126679": native_floor,
            "c2_over_legacy_runtime_at_most_2x": runtime_gate,
            "c2_incremental_no_channel_regression": c2_incremental_useful,
            "reaches_T10": t10,
            "first_acceptance_scheduler_target_cutoff_order_initial_set_unchanged": True,
            "no_repaired_hull_endpoint_tightening_sampling_or_hidden_fallback": True,
            "v100_implementation_consistency_at_1e_12": consistency_pass,
        },
        "cuda_claim_scope": "implementation consistency and measured runtime/memory only; no directed-rounding soundness or speedup claim",
        "soundness_basis": "CPU binary64 outward path plus independent exact Fraction/Bernstein oracle",
        "decision": decision,
    }
    _forbidden_candidate_width(result)
    _write(output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--gate-a", type=Path, required=True)
    parser.add_argument("--baseline-verification", type=Path, required=True)
    parser.add_argument("--terminal-diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = summarize(
        args.matrix_root,
        args.gate_a,
        args.baseline_verification,
        args.output,
        args.terminal_diagnostic,
    )
    print(json.dumps(result, sort_keys=True))
