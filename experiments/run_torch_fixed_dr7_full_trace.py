#!/usr/bin/env python3
"""Run Torch's native fixed-DR7 path with full-horizon read-only tracing."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

from diffreach_torch_full_horizon_common import (
    PARTITION_SHA256,
    SCHEMA,
    SUPPORT_SHA256,
    array_record,
    capture_npz,
    parse_capture_steps,
    partition_arrays,
    records_for_fields,
    write_json,
    write_jsonl_row,
)
from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    FixedSupportReachability,
    FixedSupportSymbolicRemainderState,
    diffreach_vdp_polynomial_rhs,
    diffreach_vdp_tm_rhs,
    fixed_support_build_linear_tm,
    fixed_support_identity_parameterization,
    fixed_support_step_boxes,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _poly_fields(prefix: str, model: Any) -> dict[str, torch.Tensor]:
    polynomial = getattr(model, "polynomial", model)
    coefficients = polynomial.coeffs
    return {
        f"{prefix}_c": coefficients[..., 0],
        f"{prefix}_L": coefficients[..., 1:4],
        f"{prefix}_Lt": coefficients[..., 4:7],
    }


def _step_fields(
    *,
    model: Any,
    parameterization: Any,
    symbolic: Any,
    step: Any,
    endpoint_previous: Any,
    center: torch.Tensor,
    prefix_lo: torch.Tensor,
    prefix_hi: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if step.composed_model is None or len(step.polynomial_picard_trace) != 2:
        raise RuntimeError("native step did not expose the required read-only observer fields")
    fields: dict[str, torch.Tensor] = {}
    fields.update(_poly_fields("pre_model", model))
    fields["pre_model_R_lo"] = model.remainder.lo
    fields["pre_model_R_hi"] = model.remainder.hi
    fields.update(_poly_fields("pre_parameterization", parameterization))
    fields["pre_parameterization_R_lo"] = parameterization.remainder.lo
    fields["pre_parameterization_R_hi"] = parameterization.remainder.hi
    fields["pre_J_lo"] = symbolic.j_buffer.lo
    fields["pre_J_hi"] = symbolic.j_buffer.hi
    fields["pre_Phi"] = symbolic.phi_buffer
    fields["pre_queue_count"] = symbolic.count
    fields["pre_inverse_scale"] = symbolic.inverse_scale
    fields.update(_poly_fields("endpoint_previous", endpoint_previous))
    fields["endpoint_previous_R_lo"] = endpoint_previous.remainder.lo
    fields["endpoint_previous_R_hi"] = endpoint_previous.remainder.hi
    fields["center"] = center
    fields["scale"] = step.normalization_scale
    fields["inverse_scale"] = step.symbolic_state.inverse_scale
    fields.update(_poly_fields("normalized", step.parameterization))
    fields["normalized_R_lo"] = step.parameterization.remainder.lo
    fields["normalized_R_hi"] = step.parameterization.remainder.hi
    fields.update(_poly_fields("poly1", step.polynomial_picard_trace[0]))
    fields.update(_poly_fields("poly2", step.polynomial_picard_trace[1]))
    fields["initial_inclusion_mask"] = step.dr_picard.initial_inclusion_mask
    fields["roundoff_lo"] = step.dr_picard.roundoff_remainder.lo
    fields["roundoff_hi"] = step.dr_picard.roundoff_remainder.hi
    fields["round_masks"] = step.dr_picard.round_inclusion_masks
    fields["round_accepted_lo"] = step.dr_picard.round_remainder_lo
    fields["round_accepted_hi"] = step.dr_picard.round_remainder_hi
    fields.update(_poly_fields("retained", step.model))
    fields["retained_R_lo"] = step.model.remainder.lo
    fields["retained_R_hi"] = step.model.remainder.hi
    fields.update(_poly_fields("composed", step.composed_model))
    fields["composed_R_lo"] = step.composed_model.remainder.lo
    fields["composed_R_hi"] = step.composed_model.remainder.hi
    fields["endpoint_lo"] = step.endpoint.lo
    fields["endpoint_hi"] = step.endpoint.hi
    fields["tube_lo"] = step.full_step_tube.lo
    fields["tube_hi"] = step.full_step_tube.hi
    fields["post_J_lo"] = step.symbolic_state.j_buffer.lo
    fields["post_J_hi"] = step.symbolic_state.j_buffer.hi
    fields["post_Phi"] = step.symbolic_state.phi_buffer
    fields["post_queue_count"] = step.symbolic_state.count
    fields["queue_clear_event"] = (symbolic.count > 0) & (step.symbolic_state.count == 0)
    fields["active_mask"] = step.dr_picard.initial_inclusion_mask
    fields["failure_mask"] = ~step.dr_picard.initial_inclusion_mask
    fields["prefix_tube_hull_lo"] = prefix_lo
    fields["prefix_tube_hull_hi"] = prefix_hi
    return fields


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--symbolic-window", type=int, default=1000)
    parser.add_argument("--capture-steps", default="1,2,62,63,999,1000")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.steps <= 0 or args.step_size != float.fromhex("0x1.47ae147ae147bp-7"):
        raise ValueError("the frozen lane requires positive steps and binary64 h=0.01")
    if args.symbolic_window != 1000:
        raise ValueError("the frozen lane requires symbolic window 1000")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    capture_dir = args.output_dir / "captures"
    capture_dir.mkdir()
    captures = parse_capture_steps(args.capture_steps)
    if any(step > args.steps for step in captures):
        raise ValueError("capture step exceeds requested steps")

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    dtype = torch.float64
    device = torch.device("cpu")
    initial_lo_np, initial_hi_np = partition_arrays()
    initial_lo = torch.from_numpy(initial_lo_np.copy()).to(dtype=dtype, device=device)
    initial_hi = torch.from_numpy(initial_hi_np.copy()).to(dtype=dtype, device=device)
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    if support.support_sha256 != SUPPORT_SHA256:
        raise RuntimeError("R7 support contract hash changed")
    solver = FixedSupportReachability(
        support=support,
        state_dim=2,
        polynomial_rhs=diffreach_vdp_polynomial_rhs,
        tm_rhs=diffreach_vdp_tm_rhs,
        step_size=args.step_size,
        initial_remainder=0.01,
        polynomial_picard_iterations=2,
        remainder_rounds=10,
        symbolic_window_size=args.symbolic_window,
    )
    model = fixed_support_build_linear_tm(
        0.5 * (initial_lo + initial_hi), 0.5 * (initial_hi - initial_lo), support
    )
    parameterization = fixed_support_identity_parameterization(
        64, 2, support, dtype=dtype, device=device
    )
    symbolic = FixedSupportSymbolicRemainderState.initialize(
        64, 2, min(args.symbolic_window, args.steps), dtype=dtype, device=device
    )
    boxes = fixed_support_step_boxes(
        64, 2, args.step_size, dtype=dtype, device=device
    )

    write_json(
        args.output_dir / "command.json",
        {
            "schema": SCHEMA,
            "tool": "torch_fixed_dr7",
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": args.torch_threads,
            "source_sha": _git("rev-parse", "HEAD"),
            "partition_sha256": PARTITION_SHA256,
            "support_sha256": SUPPORT_SHA256,
        },
    )
    trace_path = args.output_dir / "trace.jsonl"
    prefix_lo = torch.full((2,), torch.inf, dtype=dtype)
    prefix_hi = torch.full((2,), -torch.inf, dtype=dtype)
    all_initial = True
    later_pass = 0
    later_total = 0
    endpoint_los: list[np.ndarray] = []
    endpoint_his: list[np.ndarray] = []
    tube_los: list[np.ndarray] = []
    tube_his: list[np.ndarray] = []
    started = time.perf_counter()
    with trace_path.open("w", encoding="utf-8") as trace_handle:
        for zero_index in range(args.steps):
            endpoint_previous = model.evaluate_time(args.step_size)
            center = endpoint_previous.polynomial.coeffs[..., support.constant_slot]
            step = solver.step_once(model, parameterization, symbolic, *boxes)
            prefix_lo = torch.minimum(prefix_lo, step.full_step_tube.lo.amin(dim=0))
            prefix_hi = torch.maximum(prefix_hi, step.full_step_tube.hi.amax(dim=0))
            fields = _step_fields(
                model=model,
                parameterization=parameterization,
                symbolic=symbolic,
                step=step,
                endpoint_previous=endpoint_previous,
                center=center,
                prefix_lo=prefix_lo,
                prefix_hi=prefix_hi,
            )
            step_number = zero_index + 1
            write_jsonl_row(
                trace_handle,
                {
                    "schema": SCHEMA,
                    "tool": "torch_fixed_dr7",
                    "step": step_number,
                    "time": step_number * args.step_size,
                    "time_hex": float(step_number * args.step_size).hex(),
                    "fields": records_for_fields(fields),
                },
            )
            if step_number in captures:
                capture_npz(capture_dir / f"step_{step_number:04d}.npz", fields)
            endpoint_los.append(step.endpoint.lo.detach().cpu().numpy().copy())
            endpoint_his.append(step.endpoint.hi.detach().cpu().numpy().copy())
            tube_los.append(step.full_step_tube.lo.detach().cpu().numpy().copy())
            tube_his.append(step.full_step_tube.hi.detach().cpu().numpy().copy())
            initial_ok = bool(torch.all(step.dr_picard.initial_inclusion_mask).item())
            all_initial = all_initial and initial_ok
            later_pass += int(step.dr_picard.round_inclusion_masks.sum().item())
            later_total += int(step.dr_picard.round_inclusion_masks.numel())
            if not initial_ok:
                raise RuntimeError(f"initial DR-RP inclusion failed at step {step_number}")
            model = step.model
            parameterization = step.parameterization
            symbolic = step.symbolic_state
    runtime_s = time.perf_counter() - started
    bounds_path = args.output_dir / "bounds.npz"
    np.savez_compressed(
        bounds_path,
        endpoint_lo=np.stack(endpoint_los),
        endpoint_hi=np.stack(endpoint_his),
        tube_lo=np.stack(tube_los),
        tube_hi=np.stack(tube_his),
    )

    final_endpoint = step.endpoint
    summary = {
        "schema": SCHEMA,
        "tool": "torch_fixed_dr7",
        "source_sha": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "partition_sha256": PARTITION_SHA256,
        "support_sha256": SUPPORT_SHA256,
        "dtype": "float64",
        "device": "cpu",
        "steps": args.steps,
        "step_size": args.step_size,
        "step_size_hex": args.step_size.hex(),
        "validated_horizon": args.steps * args.step_size,
        "validated_horizon_hex": float(args.steps * args.step_size).hex(),
        "all_initial_masks_true": all_initial,
        "later_mask_pass_count": later_pass,
        "later_mask_total_count": later_total,
        "queue_count_final": int(symbolic.count.item()),
        "final_endpoint_hull_lo": final_endpoint.lo.amin(dim=0).tolist(),
        "final_endpoint_hull_hi": final_endpoint.hi.amax(dim=0).tolist(),
        "final_prefix_tube_hull_lo": prefix_lo.tolist(),
        "final_prefix_tube_hull_hi": prefix_hi.tolist(),
        "trace_sha256": _file_sha256(trace_path),
        "initial_partition_array_records": {
            "lo": array_record(initial_lo_np),
            "hi": array_record(initial_hi_np),
        },
        "runtime_with_hashing_and_capture_s": runtime_s,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "observer_kind": "read_only_fields_returned_by_native_FixedSupportReachability.step_once",
        "observer_inertness_gate": "covered_by_test_step_observer_fields_are_read_only_views_of_native_step",
        "undeclared_fallback_or_repair": False,
        "completion_status": "completed",
    }
    write_json(args.output_dir / "summary.json", summary)
    artifacts = [
        args.output_dir / "command.json",
        trace_path,
        bounds_path,
        args.output_dir / "summary.json",
    ]
    artifacts.extend(sorted(capture_dir.glob("*.npz")))
    write_json(
        args.output_dir / "artifact_manifest.json",
        {
            "schema": SCHEMA,
            "files": [
                {
                    "path": path.relative_to(args.output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
                for path in artifacts
            ],
        },
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
