#!/usr/bin/env python3
"""Repeated native TORA controller-bound timing on the initial B48 boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import time

import torch

from torch_tm_flowpipe.tora_controller import (
    EXPECTED_ORIGINAL_CONTROLLER_SHA256,
    ToraAutoLirpaControllerBounder,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_initial_model,
    tora_q3_boundary_from_model,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    process_started = time.perf_counter()
    if args.repeats < 10:
        raise ValueError("controller benchmark requires ten repeats")
    value = os.environ.get("TORA_CONTROLLER_PATH")
    if not value:
        raise RuntimeError("TORA_CONTROLLER_PATH is required")
    controller = Path(value).resolve()
    if sha256(controller) != EXPECTED_ORIGINAL_CONTROLLER_SHA256:
        raise ValueError("controller SHA256 mismatch")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    zeros = torch.zeros(48, dtype=torch.float64, device=device)
    boundary = tora_q3_boundary_from_model(
        build_tora_q3_initial_model(zeros, zeros, device=device)
    )
    sync(device)
    build_started = time.perf_counter()
    bounder = ToraAutoLirpaControllerBounder(
        controller,
        boundary,
        device=device,
        expected_sha256=EXPECTED_ORIGINAL_CONTROLLER_SHA256,
    )
    sync(device)
    build_seconds = time.perf_counter() - build_started
    warmup = bounder.bound(boundary)
    sync(device)
    outer_samples = []
    reported_bound_samples = []
    reported_composition_samples = []
    for _ in range(args.repeats):
        sync(device)
        started = time.perf_counter()
        result = bounder.bound(boundary)
        sync(device)
        outer_samples.append(time.perf_counter() - started)
        reported_bound_samples.append(result.timing["bound_seconds"])
        reported_composition_samples.append(result.timing["composition_seconds"])
    summary = {
        "schema": "torch_tora_controller_runtime_v1",
        "status": "PASS",
        "device": str(device),
        "dtype": "float64",
        "batch": 48,
        "controller_sha256": EXPECTED_ORIGINAL_CONTROLLER_SHA256,
        "model_build_seconds": build_seconds,
        "warmup_excluded": warmup.timing,
        "measured_repeat_count": args.repeats,
        "outer_synchronized_statistics": stats(outer_samples),
        "autolirpa_bound_statistics": stats(reported_bound_samples),
        "outward_composition_statistics": stats(reported_composition_samples),
        "raw_samples": {
            "outer_synchronized_seconds": outer_samples,
            "autolirpa_bound_seconds": reported_bound_samples,
            "outward_composition_seconds": reported_composition_samples,
        },
        "cold_process_through_benchmark_seconds": (
            time.perf_counter() - process_started
        ),
        "peak_cpu_resident_memory_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "outer_synchronized_statistics": summary["outer_synchronized_statistics"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
