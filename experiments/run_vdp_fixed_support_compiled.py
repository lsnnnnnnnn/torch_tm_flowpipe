#!/usr/bin/env python3
"""Compile and benchmark a bounded functional fixed-support VDP boundary."""
from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch._dynamo.utils import compilation_time_metrics, counters

from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    fixed_support_kernel_plan,
)
from torch_tm_flowpipe.fixed_support_functional import (
    TensorState,
    make_fixed_support_functional_chunk,
    prepare_fixed_support_vdp_functional_step,
)


ROOT = Path(__file__).resolve().parents[1]


def _partition(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    left = math.isqrt(batch)
    while batch % left:
        left -= 1
    split_x, split_y = batch // left, left
    x = torch.linspace(1.1, 1.4, split_x + 1, dtype=torch.float64, device=device)
    y = torch.linspace(2.35, 2.45, split_y + 1, dtype=torch.float64, device=device)
    lo: list[torch.Tensor] = []
    hi: list[torch.Tensor] = []
    for i in range(split_x):
        for j in range(split_y):
            lo.append(torch.stack((x[i], y[j])))
            hi.append(torch.stack((x[i + 1], y[j + 1])))
    return torch.stack(lo), torch.stack(hi)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(callable_: Any, device: torch.device) -> tuple[Any, float]:
    _synchronize(device)
    started = time.perf_counter()
    value = callable_()
    _synchronize(device)
    return value, time.perf_counter() - started


def _differences(expected: TensorState, actual: TensorState) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(expected, actual)):
        if torch.equal(left, right):
            continue
        finite = torch.isfinite(left) & torch.isfinite(right)
        max_abs = 0.0
        if bool(finite.any().detach().cpu()):
            max_abs = float(torch.max(torch.abs(left[finite] - right[finite])).detach().cpu())
        differences.append(
            {
                "state_tensor_index": index,
                "dtype": str(left.dtype),
                "shape": list(left.shape),
                "max_abs_finite_difference": max_abs,
            }
        )
    return differences


def _advance(callable_: Any, state: TensorState, repetitions: int) -> TensorState:
    for _ in range(repetitions):
        state = callable_(state)
    return state


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--boundary", type=int, choices=(1, 10, 100, 1000), default=1)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--later-inputs", type=int, default=5)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % args.boundary:
        raise ValueError("steps must be positive and divisible by boundary")
    if args.warm_runs < 1 or args.later_inputs < 5:
        raise ValueError("at least one warm run and five later inputs are required")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    plan = fixed_support_kernel_plan(support, device=device, dtype=torch.float64)
    initial_lo, initial_hi = _partition(args.batch, device)
    initial_state, eager_step = prepare_fixed_support_vdp_functional_step(
        initial_lo,
        initial_hi,
        plan,
        step_size=0.01,
        steps=args.steps,
    )
    eager_boundary = (
        eager_step
        if args.boundary == 1
        else make_fixed_support_functional_chunk(eager_step, chunk_size=args.boundary)
    )
    compiled_boundary = torch.compile(
        eager_boundary,
        fullgraph=True,
        dynamic=False,
    )

    state = initial_state.tensors()
    first_expected = eager_boundary(state)
    first_actual, compile_execute_s = _elapsed(lambda: compiled_boundary(state), device)
    checks: list[dict[str, Any]] = [
        {
            "input_ordinal": 0,
            "step_index": 0,
            "bit_exact": not _differences(first_expected, first_actual),
            "differences": _differences(first_expected, first_actual),
        }
    ]
    probe_state = first_expected
    for ordinal in range(1, args.later_inputs + 1):
        expected = eager_boundary(probe_state)
        actual = compiled_boundary(probe_state)
        differences = _differences(expected, actual)
        checks.append(
            {
                "input_ordinal": ordinal,
                "step_index": ordinal * args.boundary,
                "bit_exact": not differences,
                "differences": differences,
            }
        )
        probe_state = expected

    boundary_calls = args.steps // args.boundary
    compiled_warm_s: list[float] = []
    compiled_final: TensorState | None = None
    for _ in range(args.warm_runs):
        compiled_final, elapsed = _elapsed(
            lambda: _advance(compiled_boundary, initial_state.tensors(), boundary_calls),
            device,
        )
        compiled_warm_s.append(elapsed)
    eager_final, eager_full_s = _elapsed(
        lambda: _advance(eager_boundary, initial_state.tensors(), boundary_calls),
        device,
    )
    assert compiled_final is not None
    final_differences = _differences(eager_final, compiled_final)
    metrics = {
        key: [float(value) for value in values]
        for key, values in compilation_time_metrics.items()
    }
    dynamo_counters = {
        category: {key: int(value) for key, value in entries.items()}
        for category, entries in counters.items()
    }
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    final_failure = compiled_final[12].detach().cpu()
    completed = bool(torch.all(final_failure < 0))
    floating_final = [value for value in compiled_final if value.is_floating_point()]
    finite_outputs = all(bool(torch.all(torch.isfinite(value)).detach().cpu()) for value in floating_final)
    all_probe_inputs_bit_exact = all(row["bit_exact"] for row in checks)
    full_run_bit_exact = not final_differences
    result = {
        "schema": "torch_tm_flowpipe_fixed_support_compiled_v1",
        "source_sha": source_sha,
        "support_sha256": support.support_sha256,
        "dtype": "torch.float64",
        "device": str(device),
        "batch": args.batch,
        "step_size": 0.01,
        "steps": args.steps,
        "requested_horizon": args.steps * 0.01,
        "compiled_boundary_steps": args.boundary,
        "boundary_calls_per_run": boundary_calls,
        "compile_backend": "inductor",
        "fullgraph_requested": True,
        "dynamic_shapes": False,
        "graph_break_count": int(dynamo_counters.get("graph_break", {}).get("total", 0)),
        "compile_execute_s": compile_execute_s,
        "compilation_metrics_s": metrics,
        "compile_inner_s": sum(metrics.get("_compile.compile_inner", [])),
        "eager_functional_full_s": eager_full_s,
        "compiled_warm_s": compiled_warm_s,
        "compiled_warm_min_s": min(compiled_warm_s),
        "compiled_warm_median_s": sorted(compiled_warm_s)[len(compiled_warm_s) // 2],
        "compiled_warm_max_s": max(compiled_warm_s),
        "first_and_later_input_checks": checks,
        "all_probe_inputs_bit_exact": all_probe_inputs_bit_exact,
        "full_run_bit_exact": full_run_bit_exact,
        "full_run_differences": final_differences,
        "completed": completed,
        "finite_outputs": finite_outputs,
        "first_failure_indices": [int(value) for value in final_failure.tolist()],
        "host_synchronizations_in_solver_core": 0,
        "final_decision_host_synchronizations": 1,
        "solver_device_transfers": 0,
        "benchmark_boundary_synchronizations_per_timed_run": (
            2 if device.type == "cuda" else 0
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torchinductor_cache_isolated": "TORCHINDUCTOR_CACHE_DIR" in __import__("os").environ,
        "dynamo_counters": dynamo_counters,
        "trace_mode": False,
        "numerical_soundness_class": "empirically sampled only",
        "numerical_soundness_scope": "multi-step lane",
        "formal_claim_eligible": False,
        "performance_measurement_eligible": completed and finite_outputs,
        "cross_tool_ranking_eligible": False,
        "compiled_semantics": (
            "ordinary_expression_order_bit_exact"
            if all_probe_inputs_bit_exact and full_run_bit_exact
            else "performance_only_empirical_arithmetic_changed"
        ),
        "implemented_negative_outcome": (
            None
            if all_probe_inputs_bit_exact and full_run_bit_exact
            else "FIXED_SUPPORT_COMPILE_SEMANTICS_CHANGED"
        ),
    }
    _write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if completed and finite_outputs else 2


if __name__ == "__main__":
    raise SystemExit(main())
