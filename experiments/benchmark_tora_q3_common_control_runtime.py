#!/usr/bin/env python3
"""Five-repeat steady runtime protocol for native Torch common-control T20."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import time
from typing import Any

import torch

from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    compose_tora_q3_step,
    dense_tora_q3_dr_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
    tora_q3_boundary_from_model,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (
        ordered[upper] - ordered[lower]
    )


def stats(values: list[float]) -> dict[str, float]:
    return {
        "median_seconds": statistics.median(values),
        "iqr_seconds": percentile(values, 0.75) - percentile(values, 0.25),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "mean_seconds": statistics.fmean(values),
    }


def run_t20(rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    scopes = {
        "period_boundary_setup": 0.0,
        "normalization": 0.0,
        "plant_local_step_including_validation": 0.0,
        "affine_composition_and_physical_range": 0.0,
        "endpoint_projection": 0.0,
        "scheduler": 0.0,
        "serialization_io": 0.0,
        "controller_bound_update": 0.0,
    }
    segment_seconds: list[float] = []
    completed = 0
    checksum = torch.zeros((), dtype=torch.float64, device=device)
    synchronize(device)
    wall_started = time.perf_counter()
    for period, row in enumerate(rows, start=1):
        synchronize(device)
        setup_started = time.perf_counter()
        state_lower = torch.as_tensor(
            row["pre_controller_state_box"]["lower"],
            dtype=torch.float64,
            device=device,
        )
        state_upper = torch.as_tensor(
            row["pre_controller_state_box"]["upper"],
            dtype=torch.float64,
            device=device,
        )
        control = row["u1_interval_installed_for_next_ten_segments"]
        control_lower = torch.as_tensor(
            control["lower"], dtype=torch.float64, device=device
        ).reshape(-1)
        control_upper = torch.as_tensor(
            control["upper"], dtype=torch.float64, device=device
        ).reshape(-1)
        model = build_tora_q3_box_model(
            state_lower,
            state_upper,
            control_lower,
            control_upper,
            device=device,
        )
        boundary = tora_q3_boundary_from_model(model)
        carry = identity_tora_q3_carry(48, device=device)
        synchronize(device)
        scopes["period_boundary_setup"] += time.perf_counter() - setup_started
        for local_segment in range(1, 11):
            scheduler_started = time.perf_counter()
            segment = (period - 1) * 10 + local_segment
            scopes["scheduler"] += time.perf_counter() - scheduler_started
            synchronize(device)
            segment_started = time.perf_counter()

            normalize_started = time.perf_counter()
            local_model, carry = normalize_tora_q3_boundary(boundary, carry)
            synchronize(device)
            scopes["normalization"] += time.perf_counter() - normalize_started

            plant_started = time.perf_counter()
            local_step = dense_tora_q3_dr_step(
                local_model, capture_trace=False
            )
            synchronize(device)
            scopes["plant_local_step_including_validation"] += (
                time.perf_counter() - plant_started
            )

            composition_started = time.perf_counter()
            physical_step = compose_tora_q3_step(local_step, carry)
            synchronize(device)
            scopes["affine_composition_and_physical_range"] += (
                time.perf_counter() - composition_started
            )
            if not physical_step.accepted:
                return {
                    "status": "FAILED",
                    "completed_segments": completed,
                    "first_failure_segment": segment,
                    "scopes": scopes,
                }

            projection_started = time.perf_counter()
            boundary = project_tora_q3_endpoint_to_affine(
                local_step.segment_tm
            )
            synchronize(device)
            scopes["endpoint_projection"] += (
                time.perf_counter() - projection_started
            )
            checksum = checksum + physical_step.endpoint_lower.sum()
            checksum = checksum + physical_step.endpoint_upper.sum()
            completed = segment
            segment_seconds.append(time.perf_counter() - segment_started)
    synchronize(device)
    wall = time.perf_counter() - wall_started
    return {
        "status": "VERIFIED",
        "completed_segments": completed,
        "certified_horizon": completed * 0.1,
        "wall_seconds": wall,
        "solver_excluding_serialization_seconds": sum(scopes.values()),
        "scopes": scopes,
        "segment_median_seconds": statistics.median(segment_seconds),
        "segment_min_seconds": min(segment_seconds),
        "segment_max_seconds": max(segment_seconds),
        "checksum": float(checksum.item()),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--expected-controller-trace-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 5:
        raise ValueError("formal full-T20 benchmark requires at least five repeats")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    trace_path = args.controller_trace.resolve()
    trace_hash = sha256(trace_path)
    if trace_hash != args.expected_controller_trace_sha256:
        raise ValueError("controller trace hash mismatch")
    rows = json.loads(trace_path.read_text(encoding="utf-8"))["rows"]
    if len(rows) != 20:
        raise ValueError("runtime protocol requires all twenty periods")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    process_started = time.perf_counter()
    warmup = run_t20(rows, device)
    if warmup["status"] != "VERIFIED":
        raise RuntimeError(f"warm-up failed: {warmup}")
    repeats = []
    for repeat in range(1, args.repeats + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = run_t20(rows, device)
        result["repeat"] = repeat
        repeats.append(result)
        (output / "progress.json").write_text(
            json.dumps(
                {"warmup": warmup, "completed_repeats": repeats},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if result["status"] != "VERIFIED":
            raise RuntimeError(f"measured repeat failed: {result}")

    wall_values = [row["wall_seconds"] for row in repeats]
    solver_values = [
        row["solver_excluding_serialization_seconds"] for row in repeats
    ]
    summary = {
        "schema": "torch_tora_q3_common_control_t20_runtime_v1",
        "status": "PASS",
        "lane": "common_control_plant_replay",
        "device": str(device),
        "dtype": "float64",
        "batch": 48,
        "segments_per_repeat": 200,
        "warmup_excluded": warmup,
        "measured_repeat_count": len(repeats),
        "wall_statistics": stats(wall_values),
        "solver_excluding_serialization_statistics": stats(solver_values),
        "repeats": repeats,
        "cold_process_wall_including_warmup_and_repeats_seconds": (
            time.perf_counter() - process_started
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "torch_num_threads": torch.get_num_threads(),
            "seed": 0,
        },
        "peak_cpu_resident_memory_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "controller_trace_sha256": trace_hash,
        "timing_scope_notes": {
            "controller_bound_update": "zero: frozen Xiangru controller trace is replayed",
            "serialization_io": "zero inside repeats; only timing summaries are written between repeats",
            "compile_graph_warmup": "one complete T20 run excluded from measured statistics",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "measured_repeat_count": len(repeats),
        "wall_statistics": summary["wall_statistics"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
