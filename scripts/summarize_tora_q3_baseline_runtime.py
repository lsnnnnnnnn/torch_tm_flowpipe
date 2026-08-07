#!/usr/bin/env python3
"""Validate private TORA-Q3 runtime runs and emit sanitized public aggregates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


EXPECTED_TRACE_SHA256 = (
    "89a225add6e2c02ecb3e84b2182b2f7ea872b064dd9e5e534444552485a091d9"
)


def deterministic_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(deterministic_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cold_process_wall(summary: dict[str, Any]) -> float:
    for key in (
        "cold_process_wall_including_warmup_and_repeats_seconds",
        "process_wall_including_warmup_and_repeats_seconds",
    ):
        if key in summary:
            return float(summary[key])
    raise ValueError("runtime summary does not contain cold process wall")


def validate_summary(summary: dict[str, Any], *, label: str) -> None:
    if summary.get("status") != "PASS":
        raise ValueError(f"{label}: status is not PASS")
    if summary.get("controller_trace_sha256") != EXPECTED_TRACE_SHA256:
        raise ValueError(f"{label}: controller trace hash mismatch")
    if summary.get("batch") != 48 or summary.get("dtype") != "float64":
        raise ValueError(f"{label}: expected float64 B48")
    if summary.get("segments_per_repeat") != 200:
        raise ValueError(f"{label}: expected 200 segments")
    if summary.get("measured_repeat_count") != 5:
        raise ValueError(f"{label}: expected five measured repeats")
    repeats = summary.get("repeats", [])
    if len(repeats) != 5:
        raise ValueError(f"{label}: repeat payload is incomplete")
    if any(
        row.get("status") != "VERIFIED"
        or row.get("completed_segments") != 200
        or float(row.get("certified_horizon", -1.0)) != 20.0
        for row in repeats
    ):
        raise ValueError(f"{label}: a repeat is not a complete VERIFIED T20")
    checksums = {float(row["checksum"]) for row in repeats}
    if len(checksums) != 1:
        raise ValueError(f"{label}: repeat output checksums are unstable")
    warmup = summary.get("warmup_excluded", {})
    if (
        warmup.get("status") != "VERIFIED"
        or warmup.get("completed_segments") != 200
        or float(warmup.get("certified_horizon", -1.0)) != 20.0
    ):
        raise ValueError(f"{label}: excluded warmup is not a complete VERIFIED T20")


def _stage_medians(repeats: list[dict[str, Any]]) -> dict[str, float]:
    names = sorted({name for row in repeats for name in row["scopes"]})
    return {
        name: statistics.median(float(row["scopes"].get(name, 0.0)) for row in repeats)
        for name in names
    }


def sanitized_lane(
    summary: dict[str, Any],
    *,
    label: str,
    implementation: str,
    source_commit: str,
    runner_sha256: str,
    private_summary_sha256: str,
) -> dict[str, Any]:
    validate_summary(summary, label=label)
    repeats = summary["repeats"]
    checksums = [float(row["checksum"]) for row in repeats]
    statuses = [str(row["status"]) for row in repeats]
    stability_payload = [
        {
            "repeat": int(row["repeat"]),
            "status": str(row["status"]),
            "checksum": float(row["checksum"]),
            "completed_segments": int(row["completed_segments"]),
        }
        for row in repeats
    ]
    return {
        "label": label,
        "implementation": implementation,
        "source_commit": source_commit,
        "runner_sha256": runner_sha256,
        "private_summary_sha256": private_summary_sha256,
        "environment": summary["environment"],
        "device": summary["device"],
        "dtype": summary["dtype"],
        "batch": summary["batch"],
        "step_size": 0.1,
        "segments_per_repeat": summary["segments_per_repeat"],
        "controller_trace_sha256": summary["controller_trace_sha256"],
        "cold_process_wall_seconds": _cold_process_wall(summary),
        "excluded_full_t20_warmup_seconds": float(summary["warmup_excluded"]["wall_seconds"]),
        "engine_build_seconds": (
            float(summary["engine_build_seconds"])
            if "engine_build_seconds" in summary
            else None
        ),
        "steady_wall_statistics": summary["wall_statistics"],
        "steady_solver_statistics": summary["solver_excluding_serialization_statistics"],
        "steady_stage_median_seconds": _stage_medians(repeats),
        "peak_cpu_resident_memory_bytes": int(summary["peak_cpu_resident_memory_bytes"]),
        "peak_cuda_memory_bytes": max(int(row["peak_cuda_memory_bytes"]) for row in repeats),
        "repeat_wall_seconds": [float(row["wall_seconds"]) for row in repeats],
        "repeat_solver_seconds": [
            float(row["solver_excluding_serialization_seconds"]) for row in repeats
        ],
        "repeat_statuses": statuses,
        "repeat_checksums": checksums,
        "stable_status_and_output": len(set(statuses)) == 1 and len(set(checksums)) == 1,
        "repeat_status_output_sha256": payload_sha256(stability_payload),
        "timing_scope_notes": summary["timing_scope_notes"],
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-py11-summary", type=Path, required=True)
    parser.add_argument("--torch-matched-summary", type=Path, required=True)
    parser.add_argument("--xiangru-matched-summary", type=Path, required=True)
    parser.add_argument("--torch-source-commit", required=True)
    parser.add_argument("--xiangru-source-commit", required=True)
    parser.add_argument("--torch-runner-sha256", required=True)
    parser.add_argument("--xiangru-runner-sha256", required=True)
    parser.add_argument("--controller-model-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "torch_py11": args.torch_py11_summary,
        "torch_matched_crown": args.torch_matched_summary,
        "xiangru_matched_crown": args.xiangru_matched_summary,
    }
    raw = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in paths.items()
    }
    lanes = {
        "torch_py11": sanitized_lane(
            raw["torch_py11"],
            label="torch_py11",
            implementation="torch_tm_flowpipe",
            source_commit=args.torch_source_commit,
            runner_sha256=args.torch_runner_sha256,
            private_summary_sha256=file_sha256(paths["torch_py11"]),
        ),
        "torch_matched_crown": sanitized_lane(
            raw["torch_matched_crown"],
            label="torch_matched_crown",
            implementation="torch_tm_flowpipe",
            source_commit=args.torch_source_commit,
            runner_sha256=args.torch_runner_sha256,
            private_summary_sha256=file_sha256(paths["torch_matched_crown"]),
        ),
        "xiangru_matched_crown": sanitized_lane(
            raw["xiangru_matched_crown"],
            label="xiangru_matched_crown",
            implementation="xiangru_crown_reach",
            source_commit=args.xiangru_source_commit,
            runner_sha256=args.xiangru_runner_sha256,
            private_summary_sha256=file_sha256(paths["xiangru_matched_crown"]),
        ),
    }
    trace_hashes = {lane["controller_trace_sha256"] for lane in lanes.values()}
    if trace_hashes != {EXPECTED_TRACE_SHA256}:
        raise ValueError("runtime lanes did not use the same frozen controller trace")

    torch_py11 = float(lanes["torch_py11"]["steady_wall_statistics"]["median_seconds"])
    torch_matched = float(
        lanes["torch_matched_crown"]["steady_wall_statistics"]["median_seconds"]
    )
    xiangru_matched = float(
        lanes["xiangru_matched_crown"]["steady_wall_statistics"]["median_seconds"]
    )
    comparisons = {
        "same_software_stack_descriptive_ratio": {
            "numerator": "torch_matched_crown",
            "denominator": "xiangru_matched_crown",
            "torch_over_xiangru": torch_matched / xiangru_matched,
            "classification": "descriptive_baseline_runtime_ratio_not_optimized_speedup",
        },
        "torch_environment_effect": {
            "numerator": "torch_py11",
            "denominator": "torch_matched_crown",
            "py11_over_matched_crown": torch_py11 / torch_matched,
            "classification": "environment_effect_only_not_implementation_speedup",
        },
    }
    output = {
        "schema": "tora_q3_baseline_runtime_public_v1",
        "status": "PASS",
        "workload": {
            "lane": "common_control_plant_replay",
            "not_independent_closed_loop": True,
            "controller_trace_sha256": EXPECTED_TRACE_SHA256,
            "controller_model_sha256": args.controller_model_sha256,
            "dtype": "float64",
            "batch": 48,
            "step_size": 0.1,
            "segments": 200,
            "excluded_full_t20_warmups": 1,
            "measured_repeats": 5,
            "host_threads": 1,
        },
        "lanes": lanes,
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline_runtime_summary.json").write_bytes(
        deterministic_json(output)
    )

    repeat_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for label, lane in lanes.items():
        for index, (wall, solver, status, checksum) in enumerate(
            zip(
                lane["repeat_wall_seconds"],
                lane["repeat_solver_seconds"],
                lane["repeat_statuses"],
                lane["repeat_checksums"],
                strict=True,
            ),
            start=1,
        ):
            repeat_rows.append(
                {
                    "lane": label,
                    "repeat": index,
                    "wall_seconds": wall,
                    "solver_excluding_serialization_seconds": solver,
                    "status": status,
                    "checksum": checksum,
                }
            )
        for stage, seconds in lane["steady_stage_median_seconds"].items():
            stage_rows.append(
                {"lane": label, "stage": stage, "median_seconds": seconds}
            )
    write_csv(
        args.output_dir / "baseline_runtime_repeats.csv",
        [
            "lane",
            "repeat",
            "wall_seconds",
            "solver_excluding_serialization_seconds",
            "status",
            "checksum",
        ],
        repeat_rows,
    )
    write_csv(
        args.output_dir / "baseline_runtime_stages.csv",
        ["lane", "stage", "median_seconds"],
        stage_rows,
    )
    comparison_rows = [
        {
            "comparison": "same_software_stack_torch_over_xiangru",
            "numerator": "torch_matched_crown",
            "denominator": "xiangru_matched_crown",
            "ratio": comparisons["same_software_stack_descriptive_ratio"][
                "torch_over_xiangru"
            ],
            "classification": comparisons["same_software_stack_descriptive_ratio"][
                "classification"
            ],
        },
        {
            "comparison": "torch_environment_py11_over_matched_crown",
            "numerator": "torch_py11",
            "denominator": "torch_matched_crown",
            "ratio": comparisons["torch_environment_effect"]["py11_over_matched_crown"],
            "classification": comparisons["torch_environment_effect"]["classification"],
        },
    ]
    write_csv(
        args.output_dir / "baseline_runtime_comparisons.csv",
        ["comparison", "numerator", "denominator", "ratio", "classification"],
        comparison_rows,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "same_stack_torch_over_xiangru": comparisons[
                    "same_software_stack_descriptive_ratio"
                ]["torch_over_xiangru"],
                "py11_over_matched_crown": comparisons["torch_environment_effect"][
                    "py11_over_matched_crown"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
