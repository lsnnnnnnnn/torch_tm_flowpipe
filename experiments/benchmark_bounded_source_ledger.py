#!/usr/bin/env python3
"""CPU/CUDA performance audit for the bounded source-ledger carry kernels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.source_ledger import affine_lift_interval
from torch_tm_flowpipe import DenseRangePolicy, FlowstarNormalFlowpipeState, Interval, PolynomialODE
from torch_tm_flowpipe.flowpipe import (
    _flowstar_bounded_source_ledger_transition,
    _initialize_bounded_source_normal_state,
    flowpipe_step_from_tm,
)


BATCHES = (1, 8, 64, 256, 512)


def actual_b1_cpu(repeats: int = 7) -> dict[str, Any]:
    """Separate the real B1 dense solve from the accepted-boundary carry."""

    spec = {
        "state_names": ["x", "y"],
        "rhs": [
            {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
            {"terms": [
                {"coefficient": 1.0, "powers": [0, 1]},
                {"coefficient": -1.0, "powers": [2, 1]},
                {"coefficient": -1.0, "powers": [1, 0]},
            ]},
        ],
    }
    ode = PolynomialODE.from_system_spec(spec)
    state = _initialize_bounded_source_normal_state(
        FlowstarNormalFlowpipeState.from_initial_box(
            [Interval(1.1, 1.4), Interval(2.35, 2.45)], 4
        ),
        4,
    )
    reset = state.normalized_initial_tm(4)
    policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
    )
    solve_samples: list[float] = []
    segments = []
    for _ in range(repeats):
        started = time.perf_counter()
        segment = flowpipe_step_from_tm(
            ode,
            reset,
            0.01,
            4,
            max_validation_attempts=2,
            validation_eps=1e-12,
            validation_mode="flowstar_raw_remainder_compat",
            target_remainder_radius=1e-4,
            cutoff_threshold=1e-10,
            tm_backend="dense",
            dense_range_policy=policy,
        )
        solve_samples.append(time.perf_counter() - started)
        if segment.status != "validated":
            raise AssertionError("actual B1 benchmark step rejected")
        segments.append(segment)
    carry_samples: list[float] = []
    for segment in segments:
        started = time.perf_counter()
        _flowstar_bounded_source_ledger_transition(
            segment,
            state,
            4,
            cutoff_threshold=1e-10,
            right_map_range_mode="standard",
            right_map_center_mode="constant",
        )
        carry_samples.append(time.perf_counter() - started)
    solve_median = statistics.median(solve_samples)
    carry_median = statistics.median(carry_samples)
    return {
        "device": "cpu",
        "batch": 1,
        "repeats": repeats,
        "dense_picard_range_validation_median_s": solve_median,
        "accepted_boundary_carry_median_s": carry_median,
        "combined_step_median_s": solve_median + carry_median,
        "carry_fraction_of_combined": carry_median / (solve_median + carry_median),
        "adaptive_outer_loop_included": False,
        "trace_io_included": False,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(device: torch.device, batch: int, repeats: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20260814 + batch)
    lo_cpu = torch.rand((batch, 2), generator=generator, dtype=torch.float64) - 1.0
    hi_cpu = lo_cpu + torch.rand((batch, 2), generator=generator, dtype=torch.float64) * 1e-3
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    transfer_in_started = time.perf_counter()
    lo = lo_cpu.to(device)
    hi = hi_cpu.to(device)
    synchronize(device)
    transfer_in_s = time.perf_counter() - transfer_in_started
    for _ in range(50):
        witness = affine_lift_interval(lo, hi)
    synchronize(device)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        witness = affine_lift_interval(lo, hi)
        synchronize(device)
        samples.append(time.perf_counter() - started)
    transfer_out_started = time.perf_counter()
    represented_lo = witness.represented_lo.cpu()
    represented_hi = witness.represented_hi.cpu()
    synchronize(device)
    transfer_out_s = time.perf_counter() - transfer_out_started
    if not bool(torch.all(represented_lo <= lo_cpu) and torch.all(represented_hi >= hi_cpu)):
        raise AssertionError("benchmark source lift lost containment")
    median = statistics.median(samples)
    return {
        "device": device.type,
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "batch": batch,
        "state_dim": 2,
        "kernel": "complete_validated_ledger_interval_to_affine_source",
        "repeats": repeats,
        "median_kernel_s": median,
        "p10_kernel_s": sorted(samples)[max(0, int(0.1 * len(samples)) - 1)],
        "p90_kernel_s": sorted(samples)[min(len(samples) - 1, int(0.9 * len(samples)))],
        "median_per_state_s": median / batch,
        "states_per_s": batch / median,
        "host_to_device_once_s": transfer_in_s,
        "device_to_host_once_s": transfer_out_s,
        "peak_device_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
        "contains": True,
        "timing_synchronized": True,
        "scope": "carry_lift_kernel_only_not_multistep_solver",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", 0))
    rows = [measure(device, batch, args.repeats) for device in devices for batch in BATCHES]
    actual_b1 = actual_b1_cpu()
    cpu = {row["batch"]: row for row in rows if row["device"] == "cpu"}
    cuda = {row["batch"]: row for row in rows if row["device"] == "cuda"}
    comparisons = []
    for batch in BATCHES:
        comparisons.append(
            {
                "batch": batch,
                "cpu_kernel_s": cpu[batch]["median_kernel_s"],
                "cuda_kernel_s": cuda.get(batch, {}).get("median_kernel_s"),
                "cuda_over_cpu_kernel_speedup": (
                    cpu[batch]["median_kernel_s"] / cuda[batch]["median_kernel_s"]
                    if batch in cuda
                    else None
                ),
                "full_solver_speedup_claimed": False,
            }
        )
    result = {
        "schema": "bounded_source_ledger_performance_v1",
        "authoritative_scientific_lane": "CPU_float64_B1",
        "cuda_semantics": "implementation_consistency_only_not_formal_directed_rounding",
        "rows": rows,
        "actual_b1_phase_timing": actual_b1,
        "comparisons": comparisons,
        "interpretation": (
            "These rows benchmark only the tensor affine-source lift. Production sparse boundary "
            "composition and the dense Picard/range/validator remain separate costs; no complete "
            "multi-step solver speedup follows from batched kernel throughput."
        ),
    }
    (args.output_dir / "performance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "performance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"devices": [str(device) for device in devices], "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
