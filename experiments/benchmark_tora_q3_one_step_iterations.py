#!/usr/bin/env python3
"""Formal fixed-input one-step timings for TORA-Q3 optimization iterations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Callable

import torch

from torch_tm_flowpipe.batched_dense_tm import (
    compiled_point_enclosure_status,
    dense_transient_ledger_suppressed,
    monomial_interval_cache_status,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_initial_model,
    compose_tora_q3_step,
    dense_validation_batch,
    dense_tora_q3_dr_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
    tora_q3_boundary_from_model,
)


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
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def statistics_payload(values: list[float]) -> dict[str, float | int]:
    return {
        "repeats": len(values),
        "median_seconds": statistics.median(values),
        "iqr_seconds": percentile(values, 0.75) - percentile(values, 0.25),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "mean_seconds": statistics.fmean(values),
    }


def tensor_sha256(values: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def measure(
    function: Callable[[], Any],
    *,
    device: torch.device,
    repeats: int,
) -> tuple[list[float], Any]:
    samples: list[float] = []
    last: Any = None
    with torch.no_grad():
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter()
            last = function()
            synchronize(device)
            samples.append(time.perf_counter() - started)
    return samples, last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--point-enclosure-backend",
        choices=("eager", "compiled"),
        default="eager",
    )
    args = parser.parse_args()
    if args.repeats < 10:
        raise ValueError("formal one-step benchmark requires at least ten repeats")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    control_lower = torch.full((48,), 9.8, dtype=torch.float64, device=device)
    control_upper = torch.full((48,), 10.2, dtype=torch.float64, device=device)
    base = build_tora_q3_initial_model(control_lower, control_upper, device=device)
    boundary = tora_q3_boundary_from_model(base)
    local, carry = normalize_tora_q3_boundary(
        boundary, identity_tora_q3_carry(48, device=device)
    )

    def dense_step():
        with dense_validation_batch():
            with dense_transient_ledger_suppressed():
                return dense_tora_q3_dr_step(
                    local,
                    capture_trace=False,
                    point_enclosure_backend=args.point_enclosure_backend,
                )

    def logical_step():
        with dense_validation_batch():
            with dense_transient_ledger_suppressed():
                local_step = dense_tora_q3_dr_step(
                    local,
                    capture_trace=False,
                    point_enclosure_backend=args.point_enclosure_backend,
                )
            physical_step = compose_tora_q3_step(local_step, carry)
            projected = project_tora_q3_endpoint_to_affine(local_step.segment_tm)
        return local_step, physical_step, projected

    synchronize(device)
    cold_started = time.perf_counter()
    with torch.no_grad():
        warm_local, warm_physical, warm_projected = logical_step()
    synchronize(device)
    warmup_seconds = time.perf_counter() - cold_started
    if not warm_local.accepted or not warm_physical.accepted:
        raise RuntimeError("excluded full one-step warmup did not validate")

    dense_samples, dense_last = measure(dense_step, device=device, repeats=args.repeats)
    logical_samples, logical_last = measure(logical_step, device=device, repeats=args.repeats)
    local_step, physical_step, projected = logical_last
    if not dense_last.accepted or not local_step.accepted or not physical_step.accepted:
        raise RuntimeError("a measured one-step output did not validate")

    source_paths = (
        Path("src/torch_tm_flowpipe/batched_dense_tm.py"),
        Path("src/torch_tm_flowpipe/tora_q3.py"),
        Path("experiments/benchmark_tora_q3_one_step_iterations.py"),
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *(str(path) for path in source_paths)],
        check=True,
        capture_output=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    output_hash = tensor_sha256(
        (
            local_step.segment_tm.poly.coeffs,
            local_step.segment_tm.rem_lo,
            local_step.segment_tm.rem_hi,
            local_step.endpoint_lower,
            local_step.endpoint_upper,
            local_step.tube_lower,
            local_step.tube_upper,
            physical_step.endpoint_lower,
            physical_step.endpoint_upper,
            projected.center,
            projected.linear,
            projected.remainder_lower,
            projected.remainder_upper,
        )
    )
    summary = {
        "schema": "tora_q3_one_step_optimization_benchmark_v1",
        "status": "PASS",
        "phase_label": args.phase_label,
        "base_commit": commit,
        "instrumented_worktree": True,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "device": str(device),
        "dtype": "float64",
        "batch": 48,
        "step_size": 0.1,
        "remainder_rounds": 10,
        "point_enclosure_backend_requested": args.point_enclosure_backend,
        "point_enclosure_backend_status": compiled_point_enclosure_status(),
        "monomial_interval_cache_status": monomial_interval_cache_status(),
        "excluded_full_one_step_warmup_seconds": warmup_seconds,
        "measured_repeats": args.repeats,
        "dense_validated_step": statistics_payload(dense_samples),
        "logical_step_dense_compose_projection": statistics_payload(logical_samples),
        "dense_samples_seconds": dense_samples,
        "logical_samples_seconds": logical_samples,
        "output_status_sha256": output_hash,
        "accepted": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_num_threads": torch.get_num_threads(),
            "seed": 0,
        },
        "timing_scope_notes": {
            "profiler_enabled": False,
            "cuda_synchronized_at_scope_boundaries": True,
            "serialization_excluded": True,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "phase_label": args.phase_label,
                "dense_median_seconds": summary["dense_validated_step"]["median_seconds"],
                "logical_median_seconds": summary[
                    "logical_step_dense_compose_projection"
                ]["median_seconds"],
                "output_status_sha256": output_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
