#!/usr/bin/env python3
"""Freeze hash-addressed TORA-Q3 baseline invariants without private bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(deterministic_json(value)).hexdigest()


def combined_file_sha256(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        data = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def freeze_b48_one_step_rerun(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "tora_q3_one_step_optimization_benchmark_v1":
        raise ValueError("unexpected B48 one-step rerun schema")
    if value.get("status") != "PASS" or value.get("accepted") is not True:
        raise ValueError("B48 one-step rerun did not validate")
    if value.get("batch") != 48 or value.get("remainder_rounds") != 10:
        raise ValueError("B48 one-step rerun does not use the frozen workload")
    return {
        "status": value["status"],
        "accepted": value["accepted"],
        "batch": value["batch"],
        "step_size": value["step_size"],
        "remainder_rounds": value["remainder_rounds"],
        "output_status_sha256": value["output_status_sha256"],
        "source_file_sha256": sha256(path),
    }


def freeze_common_control_rerun(path: Path, trace_sha256: str) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema") != "torch_tora_q3_common_control_summary_v1":
        raise ValueError("unexpected common-control rerun schema")
    if (
        value.get("status") != "VERIFIED"
        or value.get("completed_segments") != 200
        or value.get("certified_horizon") != 20.0
        or value.get("first_failure") is not None
    ):
        raise ValueError("common-control rerun is not a complete T20 replay")
    if value.get("controller_trace_sha256") != trace_sha256:
        raise ValueError("common-control rerun controller trace drift")
    return {
        "status": value["status"],
        "completed_segments": value["completed_segments"],
        "certified_horizon": value["certified_horizon"],
        "first_failure": value["first_failure"],
        "segments_sha256": value["segments_sha256"],
        "peak_cuda_memory_bytes": value["peak_cuda_memory_bytes"],
        "source_file_sha256": sha256(path),
        "timing_scope": "single diagnostic replay; not a formal steady timing",
    }


def freeze_native_rerun(
    path: Path,
    *,
    lane: str,
    polynomial_picard_rounds: int,
    expected_property_segment: int,
    expected_completed_segments: int,
    expected_numerical_segment: int,
    trace_sha256: str,
) -> dict[str, Any]:
    value = load_json(path)
    config = value.get("config")
    first_failure = value.get("first_failure")
    diagnostic_failure = value.get("diagnostic_failure")
    if value.get("schema") != "torch_native_full_closed_loop_tora_q3_summary_v1":
        raise ValueError(f"unexpected {lane} native rerun schema")
    if not isinstance(config, dict):
        raise ValueError(f"missing {lane} native rerun config")
    if (
        config.get("lane") != lane
        or config.get("polynomial_picard_rounds") != polynomial_picard_rounds
        or config.get("remainder_picard_rounds") != 10
        or config.get("property") != "abs(x1..x4) <= 2"
    ):
        raise ValueError(f"{lane} native rerun contract drift")
    if (
        value.get("status") != "FAILED"
        or value.get("completed_segments") != expected_completed_segments
        or not isinstance(first_failure, dict)
        or first_failure.get("segment") != expected_property_segment
        or first_failure.get("reason") != "property"
    ):
        raise ValueError(f"unexpected {lane} formal failure")
    if (
        not isinstance(diagnostic_failure, dict)
        or diagnostic_failure.get("segment") != expected_numerical_segment
        or diagnostic_failure.get("reason") != "numerical_certificate"
    ):
        raise ValueError(f"unexpected {lane} diagnostic failure")
    if value.get("controller_trace_sha256") != trace_sha256:
        raise ValueError(f"{lane} native rerun controller trace drift")
    return {
        "status": value["status"],
        "completed_segments": value["completed_segments"],
        "certified_horizon": value["certified_horizon"],
        "first_failure": first_failure,
        "diagnostic_completed_segments": value["diagnostic_completed_segments"],
        "diagnostic_failure": diagnostic_failure,
        "config": config,
        "config_sha256": value["config_sha256"],
        "segments_sha256": value["segments_sha256"],
        "controller_updates_sha256": value["controller_updates_sha256"],
        "replay_points_sha256": value["replay_points_sha256"],
        "source_sha256": value["source_sha256"],
        "source_file_sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-output-root",
        type=Path,
        default=Path("outputs/tora_q3_native_matched_20260806"),
    )
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--one-step-detail", type=Path, required=True)
    parser.add_argument("--one-step-summary", type=Path, required=True)
    parser.add_argument("--b48-one-step-rerun-summary", type=Path)
    parser.add_argument("--common-control-rerun-summary", type=Path)
    parser.add_argument("--native-k2-rerun-summary", type=Path)
    parser.add_argument("--native-k3-rerun-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.legacy_output_root
    common_path = root / "common_control_replay/gates.json"
    full_path = root / "full_closed_loop/summary.json"
    common = json.loads(common_path.read_text(encoding="utf-8"))
    full = json.loads(full_path.read_text(encoding="utf-8"))
    one_step = json.loads(args.one_step_summary.read_text(encoding="utf-8"))
    detail = json.loads(args.one_step_detail.read_text(encoding="utf-8"))

    common_status = {
        "lane": common["lane"],
        "period_local_observation_restart": common["period_local_observation_restart"],
        "not_independent_closed_loop": common["not_independent_closed_loop"],
        "gates": [
            {
                key: gate[key]
                for key in (
                    "gate",
                    "status",
                    "completed_segments",
                    "certified_horizon",
                    "expected_leaf_count",
                )
                if key in gate
            }
            for gate in common["gates"]
        ],
    }
    if any(gate["status"] != "PASS" for gate in common_status["gates"]):
        raise ValueError("common-control baseline contains a failed gate")
    if common_status["gates"][-1].get("completed_segments") != 200:
        raise ValueError("common-control baseline is not a complete T20 replay")

    endpoint_tube_files = {
        "endpoint_width_over_time.csv": root / "comparison/endpoint_width_over_time.csv",
        "tube_width_over_time.csv": root / "comparison/tube_width_over_time.csv",
        "property_margin_over_time.csv": root / "comparison/property_margin_over_time.csv",
        "failure_horizons.csv": root / "comparison/failure_horizons.csv",
    }
    full_failure = {
        "status": full["status"],
        "completed_segments": full["completed_segments"],
        "certified_horizon": full["certified_horizon"],
        "first_failure": full["first_failure"],
        "closed_loop_gates": full["closed_loop_gates"],
    }
    if full_failure["status"] != "FAILED_AT_T4_4":
        raise ValueError("unexpected full-loop baseline status")

    tensor_fields = (
        "k1_coefficients",
        "k2_coefficients",
        "final_local_coefficients",
        "final_local_remainder_lower",
        "final_local_remainder_upper",
        "physical_coefficients",
        "physical_remainder_lower",
        "physical_remainder_upper",
        "endpoint_lower",
        "endpoint_upper",
        "tube_lower",
        "tube_upper",
        "initial_margin",
        "ledger",
    )
    torch_decimal = detail["torch_decimal"]
    one_step_tensor_payload = {
        name: torch_decimal[name] for name in tensor_fields if name in torch_decimal
    }
    one_step_status = {
        "status": one_step["status"],
        "basis": one_step["basis"],
        "validation": one_step["validation"],
        "one_step_tensor_payload_sha256": payload_sha256(one_step_tensor_payload),
    }
    if one_step_status["status"] != "PASS" or not one_step_status["validation"]["accepted"]:
        raise ValueError("one-step baseline is not accepted")

    output = {
        "schema": "tora_q3_baseline_freeze_v1",
        "common_control": {
            "semantic_status": common_status,
            "accepted_status_sha256": payload_sha256(common_status),
            "source_file_sha256": sha256(common_path),
        },
        "endpoint_tube_aggregate": {
            "combined_sha256": combined_file_sha256(endpoint_tube_files),
            "file_sha256": {
                name: sha256(path) for name, path in sorted(endpoint_tube_files.items())
            },
        },
        "full_loop_t4_4_failure": {
            "semantic_summary": full_failure,
            "semantic_summary_sha256": payload_sha256(full_failure),
            "source_file_sha256": sha256(full_path),
        },
        "one_step": {
            "semantic_status": one_step_status,
            "semantic_status_sha256": payload_sha256(one_step_status),
            "coefficient_remainder_payload_sha256": payload_sha256(one_step_tensor_payload),
            "private_detail_file_sha256": sha256(args.one_step_detail),
            "public_summary_file_sha256": sha256(args.one_step_summary),
        },
        "runtime_input": {
            "controller_trace_sha256": sha256(args.controller_trace),
            "expected_periods": 20,
            "segments_per_period": 10,
            "total_segments": 200,
            "batch": 48,
            "dtype": "float64",
            "step_size": 0.1,
        },
    }
    optional_reruns = (
        args.b48_one_step_rerun_summary,
        args.common_control_rerun_summary,
        args.native_k2_rerun_summary,
        args.native_k3_rerun_summary,
    )
    if any(path is not None for path in optional_reruns):
        if not all(path is not None for path in optional_reruns):
            raise ValueError("all four Phase-0 rerun summaries are required together")
        assert args.b48_one_step_rerun_summary is not None
        assert args.common_control_rerun_summary is not None
        assert args.native_k2_rerun_summary is not None
        assert args.native_k3_rerun_summary is not None
        trace_sha256 = output["runtime_input"]["controller_trace_sha256"]
        output["server_rerun"] = {
            "b48_one_step": freeze_b48_one_step_rerun(
                args.b48_one_step_rerun_summary
            ),
            "common_control_t20": freeze_common_control_rerun(
                args.common_control_rerun_summary, trace_sha256
            ),
            "baseline_native_k2_t5": freeze_native_rerun(
                args.native_k2_rerun_summary,
                lane="baseline_native",
                polynomial_picard_rounds=2,
                expected_property_segment=44,
                expected_completed_segments=43,
                expected_numerical_segment=48,
                trace_sha256=trace_sha256,
            ),
            "prior_k3_picard_t5": freeze_native_rerun(
                args.native_k3_rerun_summary,
                lane="k3_picard",
                polynomial_picard_rounds=3,
                expected_property_segment=45,
                expected_completed_segments=44,
                expected_numerical_segment=48,
                trace_sha256=trace_sha256,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(deterministic_json(output))
    print(
        json.dumps(
            {
                "status": "PASS",
                "common_control_status_sha256": output["common_control"]["accepted_status_sha256"],
                "one_step_coefficient_remainder_sha256": output["one_step"]["coefficient_remainder_payload_sha256"],
                "runtime_input_trace_sha256": output["runtime_input"]["controller_trace_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
