#!/usr/bin/env python3
"""Run non-VDP fixed-support generality fallbacks on real box partitions."""
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

from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    FixedSupportPolynomial,
    FixedSupportReachability,
    FixedSupportTaylorModel,
)


ROOT = Path(__file__).resolve().parents[1]


def harmonic_polynomial(state, box_lo, box_hi):
    del box_lo, box_hi
    x, y = state.component(0), state.component(1)
    return FixedSupportPolynomial.stack((y, x.scale(-1.0)))


def harmonic_tm(state, box_lo, box_hi):
    del box_lo, box_hi
    x, y = state.component(0), state.component(1)
    return FixedSupportTaylorModel.stack((y, x.scale(-1.0)))


def riccati_polynomial(state, box_lo, box_hi):
    del box_lo, box_hi
    x = state.component(0)
    return x.mul_trunc(x)


def riccati_tm(state, box_lo, box_hi):
    x = state.component(0)
    return x.mul(x, box_lo, box_hi)


def _partition(lo: tuple[float, ...], hi: tuple[float, ...], batch: int, device: torch.device):
    if len(lo) == 1:
        edges = torch.linspace(lo[0], hi[0], batch + 1, dtype=torch.float64, device=device)
        return edges[:-1, None], edges[1:, None]
    left = math.isqrt(batch)
    while batch % left:
        left -= 1
    counts = (batch // left, left)
    edges = [
        torch.linspace(lo[index], hi[index], counts[index] + 1, dtype=torch.float64, device=device)
        for index in range(2)
    ]
    lows, highs = [], []
    for i in range(counts[0]):
        for j in range(counts[1]):
            lows.append(torch.stack((edges[0][i], edges[1][j])))
            highs.append(torch.stack((edges[0][i + 1], edges[1][j + 1])))
    return torch.stack(lows), torch.stack(highs)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="cpu,cuda:0")
    parser.add_argument("--batches", default="1,64")
    args = parser.parse_args()
    devices = tuple(item for item in args.devices.split(",") if item)
    batches = tuple(int(item) for item in args.batches.split(",") if item)
    cases = (
        {
            "system": "harmonic_oscillator",
            "dim": 2,
            "lo": (0.9, -0.05),
            "hi": (1.0, 0.05),
            "polynomial_rhs": harmonic_polynomial,
            "tm_rhs": harmonic_tm,
        },
        {
            "system": "scalar_quadratic_riccati",
            "dim": 1,
            "lo": (0.0,),
            "hi": (0.1,),
            "polynomial_rhs": riccati_polynomial,
            "tm_rhs": riccati_tm,
        },
    )
    rows: list[dict[str, Any]] = []
    for device_name in devices:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            rows.append({"device": device_name, "status": "unavailable", "reason": "torch.cuda.is_available is false"})
            continue
        for case in cases:
            support = FixedSupportDescriptor.diffreach_restricted_quadratic(case["dim"])
            solver = FixedSupportReachability(
                support=support,
                state_dim=case["dim"],
                polynomial_rhs=case["polynomial_rhs"],
                tm_rhs=case["tm_rhs"],
                step_size=0.01,
                initial_remainder=0.01,
                polynomial_picard_iterations=2,
                remainder_rounds=10,
                symbolic_window_size=1000,
            )
            for batch in batches:
                lo, hi = _partition(case["lo"], case["hi"], batch, device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                result = solver.verify(lo, hi, steps=100)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime_s = time.perf_counter() - started
                endpoint_lo = result.endpoint_lo[:, -1]
                endpoint_hi = result.endpoint_hi[:, -1]
                exact_contained = False
                exact_bounds: dict[str, Any]
                if case["system"] == "harmonic_oscillator":
                    cosine = math.cos(1.0)
                    sine = math.sin(1.0)
                    corners = [
                        (x * cosine + y * sine, -x * sine + y * cosine)
                        for x in (case["lo"][0], case["hi"][0])
                        for y in (case["lo"][1], case["hi"][1])
                    ]
                    exact_lo = torch.tensor([min(value[index] for value in corners) for index in range(2)], dtype=torch.float64, device=device)
                    exact_hi = torch.tensor([max(value[index] for value in corners) for index in range(2)], dtype=torch.float64, device=device)
                else:
                    exact_lo = torch.tensor([0.0], dtype=torch.float64, device=device)
                    exact_hi = torch.tensor([0.1 / (1.0 - 0.1)], dtype=torch.float64, device=device)
                exact_contained = bool(
                    torch.all(endpoint_lo.amin(dim=0) <= exact_lo)
                    and torch.all(endpoint_hi.amax(dim=0) >= exact_hi)
                )
                exact_bounds = {"lo": exact_lo.detach().cpu().tolist(), "hi": exact_hi.detach().cpu().tolist()}
                rows.append(
                    {
                        "system": case["system"],
                        "device": device_name,
                        "dtype": "torch.float64",
                        "batch": batch,
                        "partition_is_real": batch > 1,
                        "step_size": 0.01,
                        "steps": 100,
                        "requested_horizon": 1.0,
                        "validated_horizon": result.validated_steps * 0.01,
                        "completed": result.completed,
                        "first_failure_step": result.first_failure_step,
                        "first_failure_reason": result.first_failure_reason,
                        "support_name": support.name,
                        "support_sha256": support.support_sha256,
                        "state_order": list(range(case["dim"])),
                        "initial_lo": list(case["lo"]),
                        "initial_hi": list(case["hi"]),
                        "endpoint_available": True,
                        "tube_available": True,
                        "property_available": False,
                        "endpoint_lo": endpoint_lo.detach().cpu().tolist(),
                        "endpoint_hi": endpoint_hi.detach().cpu().tolist(),
                        "exact_endpoint_hull": exact_bounds,
                        "exact_endpoint_hull_contained": exact_contained,
                        "runtime_s": runtime_s,
                        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                        "host_synchronizations": result.host_synchronizations,
                        "solver_device_transfers": result.device_transfers,
                    }
                )
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    usable = [row for row in rows if "system" in row]
    passed = bool(usable) and all(row["completed"] and row["exact_endpoint_hull_contained"] for row in usable)
    artifact = {
        "schema": "second_system_fixed_support_generality_v1",
        "source_sha": source_sha,
        "nav_dr15": {
            "available": False,
            "audit": "pinned DiffReach tree contains no NAV/navigation configuration or source; no private controller asset was used",
            "diffreach_source_revision": "dd628",
        },
        "fallback_contract": "harmonic oscillator and scalar quadratic/Riccati, 100 steps, B1 and real B64 partitions",
        "primitive_is_vdp_agnostic": True,
        "compiled_fallback_available": False,
        "rows": rows,
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "result_label": "GENERALITY_GATE_PASSED" if passed else "GENERALITY_GATE_FAILED",
        "claim_scope": "fixed-support plant-only fallback; endpoint/tube, no navigation property and no full native NAV comparison",
    }
    _write_json(args.output, artifact)
    print(json.dumps(artifact, sort_keys=True, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
