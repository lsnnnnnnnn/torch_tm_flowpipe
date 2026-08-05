#!/usr/bin/env python3
"""Small synchronized CPU/CUDA diagnostic for the subdivision range kernel."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import BatchedMonomialBasis, BatchedPolynomial, DenseRangePolicy


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed(callable_, device: str):
    synchronize(device)
    started = time.perf_counter()
    value = callable_()
    synchronize(device)
    return value, time.perf_counter() - started


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_case(batch: int, depth: int, device: str, warmup: int, repeats: int) -> dict:
    setup_started = time.perf_counter()
    basis = BatchedMonomialBasis.build(3, 4, device=device)
    generator = torch.Generator().manual_seed(20260805 + batch * 100 + depth)
    coeffs = torch.randn((batch, 2, basis.num_terms), generator=generator, dtype=torch.float64).to(device)
    coeffs = coeffs * torch.linspace(0.01, 1.0, basis.num_terms, dtype=torch.float64, device=device)
    polynomial = BatchedPolynomial(coeffs, basis)
    domain_lo = torch.tensor([[-1.0, -0.75, 0.0]], dtype=torch.float64, device=device).repeat(batch, 1)
    domain_hi = torch.tensor([[1.0, 1.25, 0.025]], dtype=torch.float64, device=device).repeat(batch, 1)
    leaves_per_owner = 2 ** (depth + 1)
    policy = DenseRangePolicy(
        method="subdivision",
        max_depth=depth,
        max_leaves=leaves_per_owner,
        split_vars=(0, 1),
    )
    setup_s = time.perf_counter() - setup_started

    def evaluate():
        return polynomial.range_bound(
            domain_lo,
            domain_hi,
            policy=policy,
            context="microbench",
            return_result=True,
        )

    first, first_call_s = timed(evaluate, device)
    warmup_times = [timed(evaluate, device)[1] for _ in range(warmup)]
    steady_results = [timed(evaluate, device) for _ in range(repeats)]
    steady_times = [item[1] for item in steady_results]
    last = steady_results[-1][0]
    return {
        "batch": batch,
        "device": device,
        "dtype": "float64",
        "depth": depth,
        "leaves_per_owner": leaves_per_owner,
        "total_leaves": int(last.cover.lo.shape[0]),
        "execution_mode": "eager",
        "compile_s": 0.0,
        "compile_status": "not_applicable_eager_path",
        "setup_s": setup_s,
        "first_call_s": first_call_s,
        "warmup_repeats": warmup,
        "warmup_total_s": sum(warmup_times),
        "steady_repeats": repeats,
        "steady_min_s": min(steady_times),
        "steady_median_s": statistics.median(steady_times),
        "steady_mean_s": statistics.mean(steady_times),
        "natural_range_median_s": statistics.median(result.timings["natural_range_s"] for result, _ in steady_results),
        "leaf_evaluation_median_s": statistics.median(result.timings["leaf_evaluation_s"] for result, _ in steady_results),
        "hull_median_s": statistics.median(result.timings["hull_s"] for result, _ in steady_results),
        "range_attribution_median_s": statistics.median(result.wall_s for result, _ in steady_results),
        "coverage_valid": bool(first.coverage_report["valid"] and last.coverage_report["valid"]),
        "finite": bool(torch.all(torch.isfinite(last.selected_lo)) and torch.all(torch.isfinite(last.selected_hi))),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rows = [run_case(batch, depth, device, args.warmup, args.repeats) for device in devices for batch in (1, 16, 48) for depth in (1, 3, 5)]
    with (output / "timings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "summary.json",
        {
            "status": "passed" if all(row["coverage_valid"] and row["finite"] for row in rows) else "failed",
            "rows": len(rows),
            "devices": devices,
            "batches": [1, 16, 48],
            "leaves_per_owner": [4, 16, 64],
            "dtype": "float64",
            "cuda_synchronized": "cuda" in devices,
            "warmup_repeats": args.warmup,
            "steady_repeats": args.repeats,
        },
    )
    write_json(
        output / "command.json",
        {
            "argv": sys.argv,
            "branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        },
    )
    print(json.dumps(json.loads((output / "summary.json").read_text()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
