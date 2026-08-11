#!/usr/bin/env python3
"""Run the canonical Torch fixed-support lane on the official VDP contract."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from torch_tm_flowpipe.fixed_support import (
    DIFFREACH_SOURCE_SHA,
    FixedSupportDescriptor,
    FixedSupportReachResult,
    FixedSupportReachability,
    diffreach_vdp_polynomial_rhs,
    diffreach_vdp_tm_rhs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _partition_factors(batch: int) -> tuple[int, int]:
    root = int(math.sqrt(batch))
    for left in range(root, 0, -1):
        if batch % left == 0:
            return batch // left, left
    raise AssertionError("positive integer always has a divisor")


def _partition_initial_box(
    batch: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    split_x, split_y = _partition_factors(int(batch))
    x_edges = torch.linspace(1.1, 1.4, split_x + 1, dtype=dtype, device=device)
    y_edges = torch.linspace(2.35, 2.45, split_y + 1, dtype=dtype, device=device)
    lowers: list[torch.Tensor] = []
    uppers: list[torch.Tensor] = []
    for x_index in range(split_x):
        for y_index in range(split_y):
            lowers.append(torch.stack((x_edges[x_index], y_edges[y_index])))
            uppers.append(torch.stack((x_edges[x_index + 1], y_edges[y_index + 1])))
    return torch.stack(lowers), torch.stack(uppers), (split_x, split_y)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_once(
    solver: FixedSupportReachability,
    initial_lo: torch.Tensor,
    initial_hi: torch.Tensor,
    steps: int,
) -> tuple[FixedSupportReachResult, float]:
    _synchronize(initial_lo.device)
    started = time.perf_counter()
    result = solver.verify(initial_lo, initial_hi, steps=steps)
    _synchronize(initial_lo.device)
    return result, time.perf_counter() - started


def _aggregate_bounds(result: FixedSupportReachResult) -> dict[str, Any]:
    last_lo = result.endpoint_lo[:, -1, :]
    last_hi = result.endpoint_hi[:, -1, :]
    endpoint = torch.stack((last_lo.amin(dim=0), last_hi.amax(dim=0)), dim=1)
    if result.tube_lo.shape[1]:
        last_tube = torch.stack(
            (result.tube_lo[:, -1, :].amin(dim=0), result.tube_hi[:, -1, :].amax(dim=0)),
            dim=1,
        )
        full_tube = torch.stack(
            (
                result.tube_lo.amin(dim=(0, 1)),
                result.tube_hi.amax(dim=(0, 1)),
            ),
            dim=1,
        )
    else:
        last_tube = torch.full_like(endpoint, torch.nan)
        full_tube = torch.full_like(endpoint, torch.nan)
    return {
        "raw_endpoint": endpoint.detach().cpu().tolist(),
        "last_full_segment_tube": last_tube.detach().cpu().tolist(),
        "full_horizon_tube": full_tube.detach().cpu().tolist(),
        "endpoint_width": (endpoint[:, 1] - endpoint[:, 0]).detach().cpu().tolist(),
        "last_tube_width": (last_tube[:, 1] - last_tube[:, 0]).detach().cpu().tolist(),
        "full_tube_width": (full_tube[:, 1] - full_tube[:, 0]).detach().cpu().tolist(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=float, default=0.1)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warm-runs", type=int, default=1)
    parser.add_argument("--symbolic-window-size", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    process_started = time.perf_counter()
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.horizon <= 0 or args.step_size <= 0:
        raise ValueError("horizon and step size must be positive")
    steps_float = args.horizon / args.step_size
    steps = int(round(steps_float))
    if not math.isclose(steps * args.step_size, args.horizon, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("horizon must be an integer multiple of step size")
    if args.batch <= 0:
        raise ValueError("batch must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    dtype = torch.float64
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    initial_lo, initial_hi, splits = _partition_initial_box(
        args.batch, dtype=dtype, device=device
    )
    solver = FixedSupportReachability(
        support=support,
        state_dim=2,
        polynomial_rhs=diffreach_vdp_polynomial_rhs,
        tm_rhs=diffreach_vdp_tm_rhs,
        step_size=args.step_size,
        initial_remainder=0.01,
        polynomial_picard_iterations=2,
        remainder_rounds=10,
        symbolic_window_size=args.symbolic_window_size,
    )

    command = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": str(device),
    }
    _write_json(args.output_dir / "command.json", command)
    support_manifest = support.manifest()
    support_manifest["support_sha256"] = support.support_sha256
    _write_json(args.output_dir / "support_manifest.json", support_manifest)

    cold_result, cold_s = _run_once(solver, initial_lo, initial_hi, steps)
    warm_times: list[float] = []
    warm_result = cold_result
    for _ in range(int(args.warm_runs)):
        warm_result, elapsed = _run_once(solver, initial_lo, initial_hi, steps)
        warm_times.append(elapsed)

    explicit_transfers = 0

    def to_numpy(value: torch.Tensor) -> np.ndarray:
        nonlocal explicit_transfers
        if value.device.type != "cpu":
            explicit_transfers += 1
        return value.detach().cpu().numpy()

    np.savez_compressed(
        args.output_dir / "bounds.npz",
        times=to_numpy(cold_result.times),
        endpoint_lo=to_numpy(cold_result.endpoint_lo),
        endpoint_hi=to_numpy(cold_result.endpoint_hi),
        tube_lo=to_numpy(cold_result.tube_lo),
        tube_hi=to_numpy(cold_result.tube_hi),
        initial_inclusion_masks=to_numpy(cold_result.initial_inclusion_masks),
        round_inclusion_masks=to_numpy(cold_result.round_inclusion_masks),
    )
    initial_pass = bool(torch.all(cold_result.initial_inclusion_masks).item())
    finite = bool(
        torch.isfinite(cold_result.endpoint_lo).all().item()
        and torch.isfinite(cold_result.endpoint_hi).all().item()
        and torch.isfinite(cold_result.tube_lo).all().item()
        and torch.isfinite(cold_result.tube_hi).all().item()
    )
    later_masks = cold_result.round_inclusion_masks
    later_pass_count = int(later_masks.sum().item())
    later_total_count = int(later_masks.numel())
    bounds = _aggregate_bounds(cold_result)
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_reserved(device))
        device_name = torch.cuda.get_device_name(device)
    else:
        peak_memory = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        device_name = platform.processor() or platform.machine()

    rhs_source = inspect.getsource(diffreach_vdp_polynomial_rhs) + inspect.getsource(diffreach_vdp_tm_rhs)
    summary = {
        "schema": "torch_tm_flowpipe_fixed_support_run_v1",
        "tool": "Torch TM",
        "source_sha": _git_value("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git_value("status", "--porcelain")),
        "execution_kind": "canonical_torch_fixed_support",
        "system": "official_flowstar_vanderpol",
        "rhs_sha256": hashlib.sha256(rhs_source.encode("utf-8")).hexdigest(),
        "initial_set": [[1.1, 1.4], [2.35, 2.45]],
        "state_and_generator_order": ["x", "y", "t", "xi_x", "xi_y"],
        "representation_name": support.name,
        "support_sha256": support.support_sha256,
        "upstream_semantic_source_sha": DIFFREACH_SOURCE_SHA,
        "polynomial_picard_semantics": "two fixed restricted-support Picard iterates",
        "validator_name": "DR-RP",
        "candidate_remainder": {
            "kind": "absolute_symmetric",
            "radius": [0.01, 0.01],
            "radius_hex": [float(0.01).hex(), float(0.01).hex()],
        },
        "remainder_rounds_and_acceptance": {
            "rounds": 10,
            "initial_inclusion_required": True,
            "later_rule": "accept_component_if_subset_else_retain_previous_component",
            "initial_pass": initial_pass,
            "later_pass_count": later_pass_count,
            "later_total_count": later_total_count,
        },
        "step_policy": {
            "kind": "fixed",
            "step_size": args.step_size,
            "step_size_hex": float(args.step_size).hex(),
            "steps": steps,
        },
        "partition_policy": {"batch": args.batch, "splits": list(splits)},
        "carry_reset_policy": "DiffReach-equivalent normalized affine carry",
        "symbolic_remainder_policy": {
            "kind": "J/Phi window with clear-on-cap",
            "capacity": min(args.symbolic_window_size, max(1, steps)),
        },
        "range_policy": support.range_policy,
        "dtype_device": {"dtype": "float64", "device": str(device), "device_name": device_name},
        "requested_horizon": args.horizon,
        "requested_horizon_hex": float(args.horizon).hex(),
        "validated_horizon": cold_result.validated_steps * args.step_size,
        "validated_horizon_hex": float(
            cold_result.validated_steps * args.step_size
        ).hex(),
        "completion_status": "completed" if cold_result.completed else "failed",
        "certificate_status": "all_initial_DR_RP_inclusions_pass" if initial_pass else "failed_initial_DR_RP_inclusion",
        "property_status": "not_applicable_no_property_in_open_loop_contract",
        "first_failure_time_reason": None
        if cold_result.completed
        else {
            "time": cold_result.validated_steps * args.step_size,
            "step": cold_result.first_failure_step,
            "reason": cold_result.first_failure_reason,
        },
        "finite_outputs": finite,
        "endpoint_tube_polynomial_remainder_widths": bounds,
        "accepted_rejected_steps": {
            "accepted": cold_result.validated_steps,
            "rejected_initial_inclusion": 0 if cold_result.completed else 1,
        },
        "algorithmic_work": {
            "batch_steps": args.batch * cold_result.validated_steps,
            "polynomial_picard_evaluations": args.batch * cold_result.validated_steps * 2,
            "remainder_picard_evaluations": args.batch * cold_result.validated_steps * 11,
            "later_component_retain_count": later_total_count - later_pass_count,
        },
        "cold_warm_core_process_runtime": {
            "cold_s": cold_s,
            "warm_s": warm_times,
            "warm_min_s": min(warm_times) if warm_times else None,
            "process_s": time.perf_counter() - process_started,
            "core_definition": "solver.verify including required per-step inclusion host gate",
        },
        "peak_memory_bytes": peak_memory,
        "host_synchronizations": cold_result.host_synchronizations,
        "solver_device_transfers": cold_result.device_transfers,
        "explicit_reporting_device_transfers": explicit_transfers,
        "soundness_classification": "empirically sampled only",
        "soundness_note": "ordinary Torch float64 semantics; independent outward replay required for promotion",
        "undeclared_fallback_or_repair": False,
        "output_semantics": {"endpoint": "tau=h", "tube": "tau in [0,h]", "stored_separately": True},
        "eligible_full_horizon": bool(cold_result.completed and initial_pass and finite),
    }
    _write_json(args.output_dir / "summary.json", summary)
    artifact_paths = [
        args.output_dir / "bounds.npz",
        args.output_dir / "command.json",
        args.output_dir / "summary.json",
        args.output_dir / "support_manifest.json",
    ]
    _write_json(
        args.output_dir / "artifact_manifest.json",
        {
            "schema": "torch_tm_flowpipe_artifact_manifest_v1",
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in artifact_paths
            ],
        },
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["eligible_full_horizon"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
