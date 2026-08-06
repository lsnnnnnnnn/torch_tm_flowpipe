#!/usr/bin/env python3
"""Merge repeated TORA runtime scopes without inventing unavailable fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-repeats", type=Path, required=True)
    parser.add_argument("--xiangru-repeats", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--torch-cold", type=Path, required=True)
    parser.add_argument("--xiangru-cold", type=Path, required=True)
    parser.add_argument("--full-loop", type=Path, required=True)
    parser.add_argument("--torch-peak-cpu-bytes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    torch_repeat = load(args.torch_repeats)
    xiangru_repeat = load(args.xiangru_repeats)
    controller = load(args.controller)
    backend = load(args.backend)
    torch_cold = load(args.torch_cold)
    xiangru_cold = load(args.xiangru_cold)
    full = load(args.full_loop)
    torch_median = torch_repeat["solver_excluding_serialization_statistics"]["median_seconds"]
    xiangru_median = xiangru_repeat["solver_excluding_serialization_statistics"]["median_seconds"]
    gpu_gate = backend["formal_gpu_speed_comparison_gate"]
    result = {
        "schema": "tora_q3_runtime_summary_v1",
        "status": "PASS_REPEATED_MEASUREMENT",
        "hardware": "same Tesla V100-SXM2-16GB; CUDA float64",
        "torch_common_control": {
            "measured_repeat_count": torch_repeat["measured_repeat_count"],
            "solver_excluding_serialization_statistics": torch_repeat[
                "solver_excluding_serialization_statistics"
            ],
            "wall_statistics": torch_repeat["wall_statistics"],
            "cold_evidence_export_wall_seconds": torch_cold[
                "wall_seconds_including_serialization"
            ],
            "peak_gpu_memory_bytes": max(
                row["peak_cuda_memory_bytes"] for row in torch_repeat["repeats"]
            ),
            "peak_cpu_resident_memory_bytes": args.torch_peak_cpu_bytes,
            "scope_note": "frozen control replay; controller bound time is zero",
        },
        "xiangru_common_control": {
            "measured_repeat_count": xiangru_repeat["measured_repeat_count"],
            "solver_excluding_serialization_statistics": xiangru_repeat[
                "solver_excluding_serialization_statistics"
            ],
            "wall_statistics": xiangru_repeat["wall_statistics"],
            "cold_evidence_export_wall_seconds": xiangru_cold[
                "wall_seconds_including_serialization"
            ],
            "peak_gpu_memory_bytes": max(
                row["peak_cuda_memory_bytes"] for row in xiangru_repeat["repeats"]
            ),
            "peak_cpu_resident_memory_bytes": xiangru_repeat[
                "peak_cpu_resident_memory_bytes"
            ],
            "engine_build_seconds": xiangru_repeat["engine_build_seconds"],
            "scope_note": "frozen control replay; controller bound time is zero",
        },
        "controller_runtime": controller,
        "backend_short_repeat_scopes": backend,
        "native_full_closed_loop_runtime": {
            "t20": "N/A",
            "reason": "formal acceptance failed at segment 44 before T5",
            "certified_horizon": full["certified_horizon"],
            "observed_failure_run_timing": full["timing"],
        },
        "comparison": {
            "torch_over_xiangru_solver_time_ratio": torch_median / xiangru_median,
            "numerator": "Torch median solver excluding serialization",
            "denominator": "Xiangru median solver excluding serialization",
            "speedup_claim_authorized": gpu_gate["status"] == "PASS_NO_PROFILED_HOST_SCALAR_SYNCHRONIZATION",
            "speedup_claim_blocker": gpu_gate,
            "interpretation": (
                "The ratio is a measured end-to-end descriptive runtime ratio, "
                "not a pure GPU-kernel or algorithm-identical speedup."
            ),
        },
        "unavailable_fields": [
            "separate Torch polynomial-propagation versus remainder-validation time inside dense_tora_q3_dr_step",
            "native full-closed-loop T20 runtime because the sound run fails at segment 44",
        ],
    }
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for tool, repeated in (("torch", torch_repeat), ("xiangru", xiangru_repeat)):
        for repeat in repeated["repeats"]:
            rows.append({
                "tool": tool,
                "repeat": repeat["repeat"],
                "segments": repeat["completed_segments"],
                "wall_seconds": repeat["wall_seconds"],
                "solver_excluding_serialization_seconds": repeat[
                    "solver_excluding_serialization_seconds"
                ],
                "segment_median_seconds": repeat["segment_median_seconds"],
                "peak_cuda_memory_bytes": repeat["peak_cuda_memory_bytes"],
            })
    with (output / "full_t20_repeat_rows.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "status": result["status"],
        "torch_over_xiangru_ratio": result["comparison"]["torch_over_xiangru_solver_time_ratio"],
        "speedup_claim_authorized": result["comparison"]["speedup_claim_authorized"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
