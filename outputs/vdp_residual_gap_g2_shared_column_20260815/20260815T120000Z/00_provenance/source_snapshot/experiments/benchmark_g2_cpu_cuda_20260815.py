#!/usr/bin/env python3
"""Synchronized B1 CPU/CUDA full-solver timing for the frozen G2 lane."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/run_vdp_dense_backend.py"
G2 = "normalized_insertion_bounded_shared_source_o4_g2"


def run_lane(root: Path, device: str) -> dict[str, Any]:
    output = root / f"full_solver_{device}_B1_T0p1"
    argv = [
        sys.executable,
        str(RUNNER),
        "--output-dir",
        str(output),
        "--tm-backend",
        "dense",
        "--device",
        device,
        "--initialization-contract",
        "exact_decimal_contract",
        "--horizon",
        "0.1",
        "--fixed-step",
        "0.01",
        "--trace-flush-every",
        "0",
        "--wall-cap-s",
        "600",
        "--reset-mode",
        G2,
        "--dense-range-method",
        "adaptive_subdivision",
        "--dense-range-trigger",
        "proactive_depth1_on_named_contexts",
        "--dense-range-max-depth",
        "1",
        "--dense-range-max-leaves",
        "4",
        "--dense-range-split-vars",
        "0,1",
        "--dense-range-contexts",
        "polynomial_truncation",
    ]
    wall_started = time.perf_counter()
    completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    wall_s = time.perf_counter() - wall_started
    (output / "benchmark_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "benchmark_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{device} full solver failed: {completed.stderr[-1000:]}")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    return {
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or platform.machine()
        ),
        "batch": 1,
        "accepted_steps": summary["accepted_steps"],
        "completed_horizon": summary["completed_horizon"],
        "full_solver_runtime_s": summary["runtime_s"],
        "subprocess_wall_s": wall_s,
        "host_to_device_s": summary["host_to_device_s"],
        "dense_picard_range_validator_kernel_s": summary["dense_kernel_s"],
        "device_to_host_s": summary["device_to_host_s"],
        "nonkernel_nontransfer_solver_s": summary["nonkernel_nontransfer_solver_s"],
        "transfer_count": summary["device_transfer_count"],
        "trace_io_s": summary["trace_io_s"],
        "endpoint_raw": summary["raw_endpoint"],
        "last_segment_raw": summary["last_segment"],
        "fallback_count": summary["fallback_count"],
        "timing_synchronized": True,
        "relative_output": str(output.relative_to(root)),
    }


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=5e-13, abs_tol=5e-15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cpu = run_lane(output, "cpu")
    rows = [cpu]
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        rows.append(run_lane(output, "cuda"))
    cuda = rows[-1] if len(rows) == 2 else None
    consistency = None
    if cuda is not None:
        fields = ("x_lo", "x_hi", "x_width", "y_lo", "y_hi", "y_width")
        consistency = {
            "endpoint_fields_close": all(
                close(cpu["endpoint_raw"][field], cuda["endpoint_raw"][field])
                for field in fields
            ),
            "segment_fields_close": all(
                close(cpu["last_segment_raw"][field], cuda["last_segment_raw"][field])
                for field in fields
            ),
            "comparison": "implementation_consistency_only_not_formal_directed_rounding",
        }
    speedup = (
        cpu["full_solver_runtime_s"] / cuda["full_solver_runtime_s"]
        if cuda is not None
        else None
    )
    result = {
        "schema": "g2_cpu_cuda_full_solver_performance_v1",
        "candidate": G2,
        "authoritative_scientific_lane": "CPU_float64_B1",
        "workload": "same_exact_decimal_B1_10_step_fixed_h0p01_T0p1",
        "cuda_available": cuda_available,
        "cuda_semantics": "implementation_consistency_only_not_formal_directed_rounding",
        "rows": rows,
        "implementation_consistency": consistency,
        "cuda_over_cpu_full_solver_speedup": speedup,
        "full_solver_speedup_claimed": bool(speedup is not None and speedup > 1.0),
        "kernel_only_speedup_extrapolated": False,
        "timing_note": (
            "H2D and D2H cover the real sparse/dense boundary conversions; the synchronized kernel "
            "phase covers dense Picard construction, polynomial range extraction, and validation."
        ),
    }
    (output / "performance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "cuda_available": cuda_available,
        "speedup": speedup,
        "speedup_claimed": result["full_solver_speedup_claimed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
