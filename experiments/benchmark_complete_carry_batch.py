#!/usr/bin/env python3
"""Batch scaling for one complete-O4 VDP step plus complete endpoint carry."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRangePolicy,
    dense_picard_validate_step,
    preserve_complete_polynomial_carry,
)


def _partition_factors(batch: int) -> tuple[int, int]:
    root = int(math.sqrt(batch))
    for left in range(root, 0, -1):
        if batch % left == 0:
            return batch // left, left
    raise AssertionError("positive batch has a divisor")


def _partition_initial_box(
    batch: int, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    split_x, split_y = _partition_factors(batch)
    x_edges = torch.linspace(1.1, 1.4, split_x + 1, dtype=dtype, device=device)
    y_edges = torch.linspace(2.35, 2.45, split_y + 1, dtype=dtype, device=device)
    lowers = []
    uppers = []
    for x_index in range(split_x):
        for y_index in range(split_y):
            lowers.append(torch.stack((x_edges[x_index], y_edges[y_index])))
            uppers.append(torch.stack((x_edges[x_index + 1], y_edges[y_index + 1])))
    return torch.stack(lowers), torch.stack(uppers), (split_x, split_y)


def _initial_models(
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    h: float,
    order: int,
) -> BatchedTaylorModel:
    batch = lower.shape[0]
    basis = BatchedMonomialBasis.build(3, order, lower.device)
    coeffs = torch.zeros((batch, 2, basis.num_terms), dtype=lower.dtype, device=lower.device)
    center = (lower + upper) / 2
    radius = (upper - lower) / 2
    coeffs[:, :, basis.constant_index] = center
    coeffs[:, 0, basis.term_index((1, 0, 0))] = radius[:, 0]
    coeffs[:, 1, basis.term_index((0, 1, 0))] = radius[:, 1]
    zeros = torch.zeros((batch, 2), dtype=lower.dtype, device=lower.device)
    domain_lo = torch.empty((batch, 3), dtype=lower.dtype, device=lower.device)
    domain_hi = torch.empty_like(domain_lo)
    domain_lo[:, :2] = -1
    domain_hi[:, :2] = 1
    domain_lo[:, 2] = 0
    domain_hi[:, 2] = h
    return BatchedTaylorModel(
        BatchedPolynomial(coeffs, basis),
        zeros,
        zeros.clone(),
        domain_lo,
        domain_hi,
        range_policy=DenseRangePolicy(
            method="adaptive_subdivision",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
            trigger="proactive_depth1_on_named_contexts",
            named_contexts=("polynomial_truncation",),
            variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
        ),
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_once(base: BatchedTaylorModel, h: float) -> tuple[Any, BatchedTaylorModel | None, float]:
    def rhs(state: Any) -> Any:
        x = state[0]
        y = state[1]
        return y.concat([y, y - x - x * x * y])

    _sync(base.poly.coeffs.device)
    started = time.perf_counter()
    result = dense_picard_validate_step(
        rhs,
        base,
        h=h,
        order=4,
        tau_index=2,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
    )
    carried = (
        preserve_complete_polynomial_carry(result.raw_endpoint)
        if result.accepted and result.raw_endpoint is not None
        else None
    )
    _sync(base.poly.coeffs.device)
    return result, carried, time.perf_counter() - started


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--h", type=float, default=0.002)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    if args.batch <= 0 or args.h <= 0:
        raise ValueError("batch and h must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    lower, upper, splits = _partition_initial_box(
        args.batch, dtype=torch.float64, device=device
    )
    base = _initial_models(lower, upper, h=args.h, order=4)
    cold_result, cold_carry, cold_s = _run_once(base, args.h)
    warm_s: list[float] = []
    final_result = cold_result
    final_carry = cold_carry
    for _ in range(args.warm_runs):
        final_result, final_carry, elapsed = _run_once(base, args.h)
        warm_s.append(elapsed)
    accepted = bool(cold_result.accepted and final_result.accepted)
    bit_exact_carry = bool(
        cold_carry is not None
        and torch.equal(cold_carry.poly.coeffs, cold_result.raw_endpoint.poly.coeffs)
        and torch.equal(cold_carry.rem_lo, cold_result.raw_endpoint.rem_lo)
        and torch.equal(cold_carry.rem_hi, cold_result.raw_endpoint.rem_hi)
    )
    if final_carry is not None:
        endpoint_lo, endpoint_hi = final_carry.range_bound()
        coefficient_sha = _tensor_sha256(final_carry.poly.coeffs)
    else:
        endpoint_lo = endpoint_hi = torch.full(
            (args.batch, 2), torch.nan, dtype=torch.float64, device=device
        )
        coefficient_sha = None
    basis = base.poly.basis
    peak_memory = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    )
    summary = {
        "schema": "torch_tm_complete_carry_batch_v1",
        "source_sha": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "execution_kind": "complete_O4_one_step_plus_complete_endpoint_carry_kernel",
        "scope_limitation": (
            "The adaptive cross-step scheduler is batch-one; this row establishes the generic dense "
            "step-and-carry primitive only and is not a multi-step certificate."
        ),
        "actual_independent_inputs": True,
        "partition_policy": {"batch": args.batch, "splits": list(splits)},
        "h": args.h,
        "status": cold_result.status,
        "accepted_all_batches": accepted,
        "bit_exact_endpoint_carry": bit_exact_carry,
        "dtype_device": {
            "dtype": "float64",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else (platform.processor() or platform.machine()),
        },
        "timing": {
            "boundary": "dense Picard construction, validation, endpoint substitution, and carry clone",
            "cold_first_call_s": cold_s,
            "warm_s": warm_s,
            "warm_min_s": min(warm_s) if warm_s else None,
            "warm_median_s": sorted(warm_s)[len(warm_s) // 2] if warm_s else None,
            "warm_max_s": max(warm_s) if warm_s else None,
            "cuda_synchronized": device.type == "cuda",
        },
        "work": {
            "slots": basis.num_terms,
            "retained_product_routes": int(basis.mul_out_indices.numel()),
            "overflow_product_routes": int(basis.trunc_left_indices.numel()),
            "picard_depth": 4,
            "remainder_rounds": cold_result.validation_attempts,
            "batch_steps": args.batch,
            "carried_coefficient_values": args.batch * 2 * basis.num_terms,
            "intervalized_retained_terms": 0,
            "host_decision_scope": "per-step aggregate inclusion gate",
        },
        "peak_memory_bytes": peak_memory,
        "endpoint_hull": torch.stack(
            (endpoint_lo.amin(dim=0), endpoint_hi.amax(dim=0)), dim=1
        ).detach().cpu().tolist(),
        "coefficient_sha256": coefficient_sha,
        "soundness_classification": "empirically sampled only",
        "soundness_note": (
            "ordinary Torch float64; the exact carry is bit-preserving, but the dense arithmetic is not "
            "universally outward rounded"
        ),
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "SHA256SUMS").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  summary.json\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if accepted and bit_exact_carry else 1


if __name__ == "__main__":
    raise SystemExit(main())
