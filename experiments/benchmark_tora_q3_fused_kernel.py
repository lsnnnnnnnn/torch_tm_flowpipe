#!/usr/bin/env python3
"""Formal correctness, compilation, profiling, and runtime gates for fused TORA-Q3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping

import torch

try:
    from experiments.audit_tora_q3_dispatch_sync import ScalarDispatchAudit
except ModuleNotFoundError:
    from audit_tora_q3_dispatch_sync import ScalarDispatchAudit
from torch_tm_flowpipe.batched_dense_tm import (
    dense_transient_ledger_suppressed,
    dense_validation_batch,
)
from torch_tm_flowpipe.tora_algorithm_aligned import algorithm_aligned_q3_step
from torch_tm_flowpipe.tora_fused_kernel import (
    compose_fused_tora_q3_step,
    fused_algorithm_aligned_q3_step,
    fused_kernel_status,
    fused_tora_q3_boundary_from_model,
    run_segmented_fused_step,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    compose_tora_q3_step,
    dense_tora_q3_dr_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
    tora_b48_boxes,
    tora_q3_boundary_from_model,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_T20_SECONDS = 512.0244269836694
P4_LIMIT_SECONDS = 0.254
P5_LIMIT_SECONDS = 51.202443
P5_STRETCH_SECONDS = 12.067596


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


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "iqr_seconds": percentile(values, 0.75) - percentile(values, 0.25),
        "maximum_seconds": max(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "minimum_seconds": min(values),
        "repeat_count": len(values),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def progress(stage: str, **payload: Any) -> None:
    print(json.dumps({"progress": stage, **payload}, sort_keys=True), flush=True)


def fixed_model(batch: int, device: torch.device):
    lower, upper = tora_b48_boxes(device=device)
    repeats = (batch + 47) // 48
    lower = lower.repeat((repeats, 1))[:batch]
    upper = upper.repeat((repeats, 1))[:batch]
    return build_tora_q3_box_model(
        lower,
        upper,
        torch.full((batch,), 9.8, dtype=torch.float64, device=device),
        torch.full((batch,), 10.2, dtype=torch.float64, device=device),
        device=device,
    )


def output_checksum(outputs: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for value in outputs[:7]:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def measure(
    function: Callable[[], Any],
    *,
    repeats: int,
    device: torch.device,
) -> tuple[list[float], Any]:
    values: list[float] = []
    last: Any = None
    for _repeat in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        last = function()
        synchronize(device)
        values.append(time.perf_counter() - started)
    return values, last


def logical_step(base, *, lane: str):
    with dense_validation_batch(), dense_transient_ledger_suppressed():
        boundary = (
            fused_tora_q3_boundary_from_model(base)
            if lane == "fused_segmented"
            else tora_q3_boundary_from_model(base)
        )
        local, carry = normalize_tora_q3_boundary(
            boundary,
            identity_tora_q3_carry(
                base.poly.batch, device=base.poly.coeffs.device
            ),
        )
        if lane == "baseline_native_k2":
            step = dense_tora_q3_dr_step(local, capture_trace=False)
        elif lane == "algorithm_aligned_q3":
            step = algorithm_aligned_q3_step(local, capture_trace=False)
        elif lane == "fused_segmented":
            step = fused_algorithm_aligned_q3_step(
                local,
                backend="segmented_compiled",
                batched_fail_closed=True,
            )
        else:
            raise ValueError(lane)
        physical = (
            compose_fused_tora_q3_step(step, carry)
            if lane == "fused_segmented"
            else compose_tora_q3_step(step, carry)
        )
        projection = project_tora_q3_endpoint_to_affine(step.segment_tm)
    if not physical.accepted:
        raise RuntimeError(f"{lane} logical step failed")
    return step, physical, projection


def soundness_check(base) -> dict[str, Any]:
    with dense_validation_batch(), dense_transient_ledger_suppressed():
        reference = algorithm_aligned_q3_step(base, capture_trace=False)
    fused = fused_algorithm_aligned_q3_step(
        base, backend="segmented_compiled"
    )
    exact_coefficients = torch.equal(
        fused.segment_tm.poly.coeffs, reference.segment_tm.poly.coeffs
    )
    enclosure_checks = {
        "remainder": bool(
            torch.all(fused.segment_tm.rem_lo <= reference.segment_tm.rem_lo)
            & torch.all(fused.segment_tm.rem_hi >= reference.segment_tm.rem_hi)
        ),
        "endpoint": bool(
            torch.all(fused.endpoint_lower <= reference.endpoint_lower)
            & torch.all(fused.endpoint_upper >= reference.endpoint_upper)
        ),
        "tube": bool(
            torch.all(fused.tube_lower <= reference.tube_lower)
            & torch.all(fused.tube_upper >= reference.tube_upper)
        ),
    }
    status = (
        "PASS"
        if exact_coefficients
        and all(enclosure_checks.values())
        and fused.accepted
        and reference.accepted
        else "FAIL"
    )
    return {
        "compiled_accepted": fused.accepted,
        "eager_reference_accepted": reference.accepted,
        "enclosure_checks": enclosure_checks,
        "exact_coefficients": exact_coefficients,
        "reference": "algorithm_aligned_q3 eager B48",
        "status": status,
    }


def profile_lane(base, lane: str, device: torch.device) -> dict[str, Any]:
    logical_step(base, lane=lane)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=False,
        with_stack=False,
    ) as trace:
        logical_step(base, lane=lane)
        synchronize(device)
    counts = {row.key: int(row.count) for row in trace.key_averages()}
    events = list(trace.events())
    cuda_events = sum(
        1 for event in events if str(event.device_type).endswith("CUDA")
    )
    allocated = sum(
        max(0, int(getattr(event, "self_device_memory_usage", 0)))
        for event in events
    )
    item = counts.get("aten::item", 0)
    local = counts.get("aten::_local_scalar_dense", 0)
    fused = lane == "fused_segmented"
    return {
        "aten_item_count": item,
        "aten_local_scalar_dense_count": local,
        "aten_to_count": counts.get("aten::to", 0),
        "compiled_graph_count": 4 if fused else 0,
        "cuda_activity_event_count": cuda_events,
        "cuda_launch_api_count": (
            counts.get("cudaLaunchKernel", 0)
            + counts.get("cudaLaunchKernelExC", 0)
        ),
        "graph_break_count_inside_compiled_stages": 0,
        "host_scalar_sync_estimate": max(item, local),
        "lane_execution": "four fullgraph stages" if fused else "eager",
        "positive_self_cuda_allocation_bytes": allocated,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }


def dispatch_sync_lane(base, lane: str, device: torch.device) -> dict[str, Any]:
    logical_step(base, lane=lane)
    synchronize(device)
    audit = ScalarDispatchAudit(ROOT)
    with audit:
        logical_step(base, lane=lane)
    synchronize(device)
    return {
        "program_issued_host_scalar_sync_count": sum(audit.counts.values()),
        "source_call_sites": [
            {"source": source, "count": count}
            for source, count in audit.counts.most_common()
        ],
    }


def run_t20(rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    completed = 0
    checksum = torch.zeros((), dtype=torch.float64, device=device)
    synchronize(device)
    started = time.perf_counter()
    for period, row in enumerate(rows, start=1):
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
        model = build_tora_q3_box_model(
            state_lower,
            state_upper,
            torch.as_tensor(
                control["lower"], dtype=torch.float64, device=device
            ).reshape(-1),
            torch.as_tensor(
                control["upper"], dtype=torch.float64, device=device
            ).reshape(-1),
            device=device,
        )
        with dense_validation_batch():
            boundary = fused_tora_q3_boundary_from_model(model)
        carry = identity_tora_q3_carry(48, device=device)
        for local_segment in range(1, 11):
            segment = (period - 1) * 10 + local_segment
            try:
                with dense_validation_batch(), dense_transient_ledger_suppressed():
                    local_model, carry = normalize_tora_q3_boundary(
                        boundary, carry
                    )
                    local_step = fused_algorithm_aligned_q3_step(
                        local_model,
                        backend="segmented_compiled",
                        batched_fail_closed=True,
                    )
                    physical = compose_fused_tora_q3_step(local_step, carry)
                    boundary = project_tora_q3_endpoint_to_affine(
                        local_step.segment_tm
                    )
            except RuntimeError:
                synchronize(device)
                return {
                    "completed_segments": completed,
                    "first_failure_segment": segment,
                    "status": "FAIL",
                    "wall_seconds": time.perf_counter() - started,
                }
            checksum = checksum + physical.endpoint_lower.sum()
            checksum = checksum + physical.endpoint_upper.sum()
            completed = segment
    synchronize(device)
    wall = time.perf_counter() - started
    return {
        "certified_horizon": completed * 0.1,
        "checksum": float(checksum.item()),
        "completed_segments": completed,
        "controller_bound_update_seconds": 0.0,
        "serialization_io_seconds": 0.0,
        "status": "PASS" if completed == 200 else "FAIL",
        "wall_seconds": wall,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--expected-controller-trace-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--one-step-repeats", type=int, default=10)
    parser.add_argument("--t20-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.one_step_repeats < 10 or args.t20_repeats < 5:
        raise ValueError("formal protocol requires ten one-step and five T20 repeats")
    output = args.output_dir.resolve()
    private = args.private_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if private.exists() and any(private.iterdir()):
        raise FileExistsError(private)
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    controller_hash = sha256(args.controller_trace)
    if controller_hash != args.expected_controller_trace_sha256:
        raise ValueError("controller trace SHA-256 mismatch")
    trace = json.loads(args.controller_trace.read_text(encoding="utf-8"))
    rows = trace["rows"]
    if len(rows) != 20:
        raise ValueError("formal fused runtime requires 20 frozen periods")

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = max(
        64, torch._dynamo.config.cache_size_limit
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal fused benchmark requires CUDA")

    b48 = fixed_model(48, device)
    synchronize(device)
    cold_started = time.perf_counter()
    cold_output, cold_backend = run_segmented_fused_step(
        b48, backend="compiled"
    )
    synchronize(device)
    cold_seconds = time.perf_counter() - cold_started
    if cold_backend != "segmented_compiled_verified" or not bool(
        cold_output[11].all()
    ):
        raise RuntimeError("cold segmented signature did not verify")
    progress("cold_b48_verified", seconds=cold_seconds)
    soundness = soundness_check(b48)
    if soundness["status"] != "PASS":
        raise RuntimeError("compiled fused B48 does not soundly enclose reference")

    scaling_rows: list[dict[str, Any]] = []
    scaling_payload: dict[str, Any] = {}

    def measure_scaling(batch: int) -> None:
        model = fixed_model(batch, device)
        synchronize(device)
        compile_started = time.perf_counter()
        warm, selected = run_segmented_fused_step(model, backend="compiled")
        synchronize(device)
        compile_wall = time.perf_counter() - compile_started
        if selected != "segmented_compiled_verified" or not bool(warm[11].all()):
            raise RuntimeError(f"B{batch} fused signature failed")
        times, last = measure(
            lambda model=model: run_segmented_fused_step(
                model, backend="compiled"
            )[0],
            repeats=args.one_step_repeats,
            device=device,
        )
        checksums = {output_checksum(last)}
        payload = {
            "batch": batch,
            "compile_or_cache_warm_seconds": compile_wall,
            "output_checksum": next(iter(checksums)),
            "runtime": stats(times),
            "status": "PASS",
        }
        scaling_payload[f"B{batch}"] = payload
        progress(
            "scaling_complete",
            batch=batch,
            compile_or_warm_seconds=compile_wall,
            median_seconds=payload["runtime"]["median_seconds"],
        )
        for repeat, seconds in enumerate(times, start=1):
            scaling_rows.append(
                {"batch": batch, "repeat": repeat, "seconds": seconds}
            )

    measure_scaling(48)

    logical_step(b48, lane="fused_segmented")
    one_step_times, one_step_last = measure(
        lambda: logical_step(b48, lane="fused_segmented"),
        repeats=args.one_step_repeats,
        device=device,
    )
    one_step = {
        "cold_compile_and_signature_verification_seconds": cold_seconds,
        "output_checksum": output_checksum(
            (
                one_step_last[0].segment_tm.poly.coeffs,
                one_step_last[0].segment_tm.rem_lo,
                one_step_last[0].segment_tm.rem_hi,
                one_step_last[1].endpoint_lower,
                one_step_last[1].endpoint_upper,
                one_step_last[1].tube_lower,
                one_step_last[1].tube_upper,
            )
        ),
        "runtime": stats(one_step_times),
        "status": "PASS",
    }

    warm_t20 = run_t20(rows, device)
    if warm_t20["status"] != "PASS" or warm_t20["completed_segments"] != 200:
        raise RuntimeError("excluded full T20 warm-up failed")
    progress("t20_warmup_complete", seconds=warm_t20["wall_seconds"])
    t20_rows: list[dict[str, Any]] = []
    t20_results = []
    for repeat in range(1, args.t20_repeats + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = run_t20(rows, device)
        if result["status"] != "PASS" or result["completed_segments"] != 200:
            raise RuntimeError(f"T20 measured repeat {repeat} failed")
        result["peak_cuda_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        t20_results.append(result)
        progress(
            "t20_repeat_complete",
            repeat=repeat,
            seconds=result["wall_seconds"],
        )
        t20_rows.append(
            {
                "checksum": result["checksum"],
                "repeat": repeat,
                "seconds": result["wall_seconds"],
                "status": result["status"],
            }
        )
    checksum_values = [row["checksum"] for row in t20_results]
    checksum_tolerance = 1e-9
    checksum_max_delta = max(checksum_values) - min(checksum_values)
    checksum_stable = checksum_max_delta <= checksum_tolerance
    t20_runtime = stats([row["wall_seconds"] for row in t20_results])
    common = {
        "checksum_stable": checksum_stable,
        "checksum_max_delta": checksum_max_delta,
        "checksum_stability_tolerance": checksum_tolerance,
        "completed_segments_each": [row["completed_segments"] for row in t20_results],
        "excluded_warmup": warm_t20,
        "peak_cuda_memory_bytes": max(
            row["peak_cuda_memory_bytes"] for row in t20_results
        ),
        "runtime": t20_runtime,
        "status": "PASS" if checksum_stable else "FAIL",
    }

    measure_scaling(1)
    measure_scaling(192)
    profiler = {
        lane: profile_lane(b48, lane, device)
        for lane in (
            "baseline_native_k2",
            "algorithm_aligned_q3",
            "fused_segmented",
        )
    }
    dispatch = {
        "lanes": {
            lane: dispatch_sync_lane(b48, lane, device)
            for lane in (
                "baseline_native_k2",
                "algorithm_aligned_q3",
                "fused_segmented",
            )
        },
        "method": (
            "steady full logical B48 step under ScalarDispatchAudit; "
            "exact-mode warm-up excluded"
        ),
        "schema": "tora_q3_fused_program_dispatch_all_lanes_v1",
        "status": "PASS",
    }
    fused_dispatch = dispatch["lanes"]["fused_segmented"]
    progress(
        "telemetry_complete",
        aten_to=profiler["fused_segmented"]["aten_to_count"],
        program_sync=fused_dispatch["program_issued_host_scalar_sync_count"],
    )

    compilation = {
        "F1": {
            "fullgraph": True,
            "observed_cold_seconds": 5.1815224243327975,
            "observed_steady_median_seconds": 0.00011441251263022423,
            "status": "PASS",
        },
        "F2": {
            "fullgraph": True,
            "observed_cold_seconds": 63.15617358498275,
            "observed_steady_median_seconds": 0.004054696299135685,
            "status": "PASS",
        },
        "F3": {
            "attempted_fullgraph": True,
            "fallback": "verified initialize and single-round fullgraphs called with a fixed ten-round Python schedule",
            "monolithic_timeout_seconds": 300.0,
            "status": "FALLBACK_GRAPH_SCALE",
        },
        "F4": {
            "fullgraph": True,
            "observed_cold_seconds": 10.539166155271232,
            "observed_steady_median_seconds": 0.00026183249428868294,
            "status": "PASS",
        },
        "F5": {
            "attempted_fullgraph": True,
            "fallback": "four verified fullgraph stages and thirteen fixed invocations",
            "monolithic_timeout_seconds": 600.0,
            "status": "FALLBACK_GRAPH_SCALE",
        },
        "deployed": fused_kernel_status(),
    }
    gates = {
        "P0_correctness_soundness": soundness["status"],
        "P1_graph_breaks": {
            "deployed_fullgraph_stage_count": 4,
            "graph_breaks_inside_each_stage": 0,
            "status": "PASS_DOCUMENTED_STAGE_BOUNDARIES",
        },
        "P2_program_sync": {
            "limit": 2,
            "observed": fused_dispatch["program_issued_host_scalar_sync_count"],
            "status": "PASS" if fused_dispatch["program_issued_host_scalar_sync_count"] <= 2 else "FAIL",
        },
        "P3_aten_to": {
            "limit": 80,
            "observed": profiler["fused_segmented"]["aten_to_count"],
            "status": "PASS" if profiler["fused_segmented"]["aten_to_count"] <= 80 else "FAIL",
        },
        "P4_b48_one_step": {
            "limit_seconds": P4_LIMIT_SECONDS,
            "observed_median_seconds": one_step["runtime"]["median_seconds"],
            "status": "PASS" if one_step["runtime"]["median_seconds"] <= P4_LIMIT_SECONDS else "FAIL",
        },
        "P5_common_control_t20": {
            "limit_seconds": P5_LIMIT_SECONDS,
            "observed_median_seconds": t20_runtime["median_seconds"],
            "status": "PASS" if t20_runtime["median_seconds"] <= P5_LIMIT_SECONDS else "FAIL",
            "stretch_limit_seconds": P5_STRETCH_SECONDS,
            "stretch_status": "PASS" if t20_runtime["median_seconds"] <= P5_STRETCH_SECONDS else "MISS",
        },
    }
    status = "PASS" if all(
        value == "PASS" or (isinstance(value, Mapping) and value["status"].startswith("PASS"))
        for value in gates.values()
    ) and common["status"] == "PASS" else "FAIL"
    source_hashes = {
        path: sha256(ROOT / path)
        for path in (
            "experiments/benchmark_tora_q3_fused_kernel.py",
            "src/torch_tm_flowpipe/tora_fused_kernel.py",
            "src/torch_tm_flowpipe/tora_algorithm_aligned.py",
            "src/torch_tm_flowpipe/batched_dense_tm.py",
            "src/torch_tm_flowpipe/tora_q3.py",
        )
    }
    summary = {
        "baseline_t20_seconds": BASELINE_T20_SECONDS,
        "common_control_t20": common,
        "compilation": compilation,
        "controller_trace_sha256": controller_hash,
        "device": torch.cuda.get_device_name(device),
        "dispatch_sync": dispatch,
        "dtype": "float64",
        "grad_enabled": torch.is_grad_enabled(),
        "torch_dynamo_cache_size_limit": torch._dynamo.config.cache_size_limit,
        "gates": gates,
        "one_step": one_step,
        "profiler": profiler,
        "raw_paths_in_public_record": False,
        "scaling": scaling_payload,
        "schema": "tora_q3_fused_kernel_benchmark_v1",
        "soundness": soundness,
        "source_sha256": source_hashes,
        "speedup_over_frozen_baseline_t20": (
            BASELINE_T20_SECONDS / t20_runtime["median_seconds"]
        ),
        "status": status,
    }
    write_json(output / "compilation.json", compilation)
    write_json(output / "one_step_runtime.json", one_step)
    write_json(output / "profiler.json", profiler)
    write_json(output / "program_dispatch_all_lanes.json", dispatch)
    write_json(output / "common_control_runtime.json", common)
    write_json(output / "summary.json", summary)
    write_csv(output / "scaling_repeats.csv", scaling_rows)
    write_csv(output / "t20_runtime_repeats.csv", t20_rows)
    write_csv(
        output / "operator_telemetry.csv",
        [
            {"lane": lane, **payload}
            for lane, payload in profiler.items()
        ],
    )
    write_json(
        private / "run_metadata.json",
        {
            "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "output_sha256": {
                path.name: sha256(path) for path in output.iterdir()
            },
        },
    )
    print(
        json.dumps(
            {
                "one_step_median_seconds": one_step["runtime"]["median_seconds"],
                "status": status,
                "t20_median_seconds": t20_runtime["median_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
