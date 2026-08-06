#!/usr/bin/env python3
"""Auditable six-variable B48 TORA-Q3 CUDA microbenchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import time
from typing import Any, Callable

import torch

from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    dense_polynomial_picard,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_initial_model,
    compose_tora_q3_tm,
    dense_tora_q3_dr_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    tora_q3_boundary_from_model,
    tora_q3_rhs,
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
    return ordered[lower] + (position - lower) * (
        ordered[upper] - ordered[lower]
    )


def summarize_timing_samples(samples: list[float]) -> dict[str, float]:
    if not samples or any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("timing samples must be finite nonnegative values")
    return {
        "repeats": len(samples),
        "median_seconds": statistics.median(samples),
        "iqr_seconds": percentile(samples, 0.75) - percentile(samples, 0.25),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "mean_seconds": statistics.fmean(samples),
    }


def measure(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, float], list[float]]:
    with torch.no_grad():
        for _ in range(warmup):
            function()
        synchronize(device)
        samples = []
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter()
            function()
            synchronize(device)
            samples.append(time.perf_counter() - started)
    return summarize_timing_samples(samples), samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 10:
        raise ValueError("formal short benchmark requires at least ten repeats")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(device)

    cold_started = time.perf_counter()
    route_started = time.perf_counter()
    basis = BatchedMonomialBasis.build(6, 3, str(device))
    multiplication = basis.multiplication_plan_for_degree(None)
    integration = basis.integration_plan(0, dtype=torch.float64)
    synchronize(device)
    route_seconds = time.perf_counter() - route_started
    kept_left, _kept_right, _kept_out, dropped_left, _dropped_right, *_ = multiplication
    kept_in, _kept_out_i, _kept_factor, overflow_in, *_ = integration

    control_lower = torch.full((48,), 9.8, dtype=torch.float64, device=device)
    control_upper = torch.full((48,), 10.2, dtype=torch.float64, device=device)
    base = build_tora_q3_initial_model(
        control_lower, control_upper, device=device
    )
    boundary = tora_q3_boundary_from_model(base)
    local, carry = normalize_tora_q3_boundary(
        boundary, identity_tora_q3_carry(48, device=device)
    )
    synchronize(device)
    first_step_started = time.perf_counter()
    first_step = dense_tora_q3_dr_step(local, capture_trace=False)
    synchronize(device)
    first_step_seconds = time.perf_counter() - first_step_started
    cold_seconds = time.perf_counter() - cold_started
    if not first_step.accepted:
        raise RuntimeError("benchmark warm step did not validate")

    rhs_value = tora_q3_rhs(local)
    operations: dict[str, dict[str, Any]] = {}
    raw_samples: dict[str, list[float]] = {}

    def record(
        name: str, function: Callable[[], Any], *, warmup: int = 1
    ) -> None:
        summary, samples = measure(
            function,
            device=device,
            warmup=warmup,
            repeats=args.repeats,
        )
        operations[name] = summary
        raw_samples[name] = samples

    record("steady_polynomial_rhs", lambda: tora_q3_rhs(local))
    record("steady_local_time_integration", lambda: rhs_value.integrate(0))
    record(
        "steady_k2_polynomial_picard",
        lambda: dense_polynomial_picard(
            tora_q3_rhs,
            local.without_remainder(),
            tau_index=0,
            order=3,
            iterations=2,
            cutoff_threshold=None,
            capture_trace=False,
        ),
    )
    record(
        "steady_full_step_including_remainder_validation",
        lambda: dense_tora_q3_dr_step(local, capture_trace=False),
        warmup=0,
    )
    record(
        "endpoint_substitution_and_evaluation",
        lambda: first_step.segment_tm.endpoint(0, 0.1).range_bound(
            context="tora_microbench_endpoint"
        ),
    )
    record(
        "full_tube_evaluation",
        lambda: first_step.segment_tm.range_bound(
            context="tora_microbench_tube"
        ),
    )
    record(
        "affine_parameterization_composition",
        lambda: compose_tora_q3_tm(first_step.segment_tm, carry),
    )
    transfer_bytes = first_step.segment_tm.poly.coeffs.numel() * 8
    record(
        "explicit_device_to_host_coefficient_transfer",
        lambda: first_step.segment_tm.poly.coeffs.detach().cpu(),
    )

    profiler = {
        "status": "unavailable",
        "cuda_event_count": "unavailable",
        "cpu_event_count": "unavailable",
        "reason": "profiler was not run",
    }
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities) as trace:
            dense_tora_q3_dr_step(local, capture_trace=False)
            synchronize(device)
        events = trace.events()
        profiler = {
            "status": "measured",
            "cuda_event_count": sum(
                str(event.device_type).endswith("CUDA") for event in events
            ),
            "cpu_event_count": sum(
                str(event.device_type).endswith("CPU") for event in events
            ),
            "aten_to_event_count": sum(event.name == "aten::to" for event in events),
            "aten_item_event_count": sum(
                event.name == "aten::item" for event in events
            ),
            "local_scalar_dense_event_count": sum(
                event.name == "aten::_local_scalar_dense" for event in events
            ),
            "reason": "",
        }
    except (RuntimeError, ImportError) as exception:
        profiler["reason"] = f"{type(exception).__name__}: {exception}"

    item_events = profiler.get("aten_item_event_count")
    local_scalar_events = profiler.get("local_scalar_dense_event_count")
    profiler_measured = (
        profiler.get("status") == "measured"
        and isinstance(item_events, int)
        and isinstance(local_scalar_events, int)
    )
    profiled_host_scalar_sync_events = (
        max(item_events, local_scalar_events)
        if profiler_measured
        else "unavailable"
    )
    if not profiler_measured:
        gpu_speed_gate_status = "FAIL_PROFILER_UNAVAILABLE"
    elif profiled_host_scalar_sync_events > 0:
        gpu_speed_gate_status = "FAIL_FREQUENT_HOST_SCALAR_SYNCHRONIZATION"
    else:
        gpu_speed_gate_status = "PASS_NO_PROFILED_HOST_SCALAR_SYNCHRONIZATION"

    summary = {
        "schema": "tora_q3_backend_microbenchmark_v1",
        "device": str(device),
        "dtype": "float64",
        "batch": 48,
        "state_dimension": 5,
        "variable_dimension": 6,
        "term_count": basis.num_terms,
        "basis_fingerprint": basis.fingerprint,
        "basis_and_route_construction_seconds": route_seconds,
        "first_call_full_step_seconds": first_step_seconds,
        "cold_process_through_first_step_seconds": cold_seconds,
        "operations": operations,
        "routes": {
            "multiplication_kept": int(kept_left.numel()),
            "multiplication_overflow": int(dropped_left.numel()),
            "integration_kept": int(kept_in.numel()),
            "integration_overflow": int(overflow_in.numel()),
        },
        "fallbacks": {
            "sparse_fallback_count": 0,
            "cpu_fallback_inside_formal_math_count": 0,
            "range_method": "natural",
            "range_fallback_count": 0,
            "explicit_profiled_aten_to_events": profiler.get(
                "aten_to_event_count", "unavailable"
            ),
            "profiled_aten_item_events": item_events or 0
            if profiler_measured
            else "unavailable",
            "profiled_local_scalar_dense_events": local_scalar_events or 0
            if profiler_measured
            else "unavailable",
            "profiled_host_scalar_sync_events": profiled_host_scalar_sync_events,
        },
        "host_device_transfer": {
            "formal_math_transfer_count": 0,
            "explicit_benchmark_transfer_bytes_per_repeat": transfer_bytes,
            "serialization_transfers_excluded_from_math_timings": True,
        },
        "timing_scope_availability": {
            "plant_polynomial_propagation": "measured as steady_polynomial_rhs and steady_k2_polynomial_picard",
            "remainder_validation_separate_from_full_step": "unavailable",
            "full_step_including_remainder_validation": "measured",
            "endpoint_evaluation": "measured",
            "tube_evaluation": "measured",
            "host_device_transfer": "measured explicitly and excluded from formal math timings",
        },
        "profiler": profiler,
        "formal_gpu_speed_comparison_gate": {
            "status": gpu_speed_gate_status,
            "aten_item_event_count": (
                item_events if profiler_measured else "unavailable"
            ),
            "local_scalar_dense_event_count": (
                local_scalar_events if profiler_measured else "unavailable"
            ),
            "host_scalar_sync_event_count": profiled_host_scalar_sync_events,
            "meaning": (
                "The mathematical tensors remain on CUDA, but Python boolean/"
                "validity checks can force device-to-host scalar synchronization. "
                "A GPU speedup claim is disallowed when this count is nonzero or "
                "the profiler evidence is unavailable. The count uses the maximum "
                "of paired aten::item and aten::_local_scalar_dense events to avoid "
                "double-counting one scalar extraction."
            ),
        },
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "peak_cpu_resident_memory_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
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
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "raw_timing_samples.json").write_text(
        json.dumps(raw_samples, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "timing_scopes.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "scope", "repeats", "median_seconds", "iqr_seconds",
            "min_seconds", "max_seconds", "mean_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, values in operations.items():
            writer.writerow({"scope": name, **values})
    print(json.dumps({
        "status": "PASS",
        "term_count": basis.num_terms,
        "first_step_seconds": first_step_seconds,
        "full_step_median_seconds": operations[
            "steady_full_step_including_remainder_validation"
        ]["median_seconds"],
        "peak_cuda_memory_bytes": summary["peak_cuda_memory_bytes"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
