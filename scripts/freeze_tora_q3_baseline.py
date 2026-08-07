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
