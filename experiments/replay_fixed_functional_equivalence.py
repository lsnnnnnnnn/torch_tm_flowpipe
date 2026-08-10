#!/usr/bin/env python3
"""Replay the bounded object/functional fixed-support equality matrix."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch

from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    FixedSupportReachability,
    diffreach_vdp_polynomial_rhs,
    diffreach_vdp_tm_rhs,
    fixed_support_kernel_plan,
)
from torch_tm_flowpipe.fixed_support_functional import fixed_support_functional_verify


ROOT = Path(__file__).resolve().parents[1]


def _partition(batch: int, device: torch.device):
    left = math.isqrt(batch)
    while batch % left:
        left -= 1
    split_x, split_y = batch // left, left
    x = torch.linspace(1.1, 1.4, split_x + 1, dtype=torch.float64, device=device)
    y = torch.linspace(2.35, 2.45, split_y + 1, dtype=torch.float64, device=device)
    lo, hi = [], []
    for i in range(split_x):
        for j in range(split_y):
            lo.append(torch.stack((x[i], y[j])))
            hi.append(torch.stack((x[i + 1], y[j + 1])))
    return torch.stack(lo), torch.stack(hi)


def _solver(support):
    return FixedSupportReachability(
        support=support,
        state_dim=2,
        polynomial_rhs=diffreach_vdp_polynomial_rhs,
        tm_rhs=diffreach_vdp_tm_rhs,
        step_size=0.01,
        initial_remainder=0.01,
        polynomial_picard_iterations=2,
        remainder_rounds=10,
        symbolic_window_size=1000,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    signatures = [("cpu", batch, steps) for batch in (1, 8, 64) for steps in (1, 2, 10, 100)]
    if torch.cuda.is_available():
        signatures.extend(("cuda:0", batch, 10) for batch in (1, 8, 64))
    rows: list[dict[str, Any]] = []
    for device_name, batch, steps in signatures:
        device = torch.device(device_name)
        lo, hi = _partition(batch, device)
        object_result = _solver(support).verify(lo, hi, steps=steps)
        functional = fixed_support_functional_verify(
            lo,
            hi,
            fixed_support_kernel_plan(support, device=device, dtype=torch.float64),
            step_size=0.01,
            steps=steps,
            trace=True,
        )
        state = functional.final_state
        symbolic = object_result.final_symbolic_state
        comparisons = {
            "endpoint_lo": torch.equal(object_result.endpoint_lo, functional.endpoint_lo),
            "endpoint_hi": torch.equal(object_result.endpoint_hi, functional.endpoint_hi),
            "tube_lo": torch.equal(object_result.tube_lo, functional.tube_lo),
            "tube_hi": torch.equal(object_result.tube_hi, functional.tube_hi),
            "initial_masks": torch.equal(object_result.initial_inclusion_masks, functional.initial_inclusion_masks),
            "round_masks": torch.equal(object_result.round_inclusion_masks, functional.round_inclusion_masks),
            "model_coefficients": torch.equal(object_result.final_model.polynomial.coeffs, state.model_coeffs),
            "model_remainder_lo": torch.equal(object_result.final_model.remainder.lo, state.model_rem_lo),
            "model_remainder_hi": torch.equal(object_result.final_model.remainder.hi, state.model_rem_hi),
            "parameter_coefficients": torch.equal(object_result.final_parameterization.polynomial.coeffs, state.parameter_coeffs),
            "parameter_remainder_lo": torch.equal(object_result.final_parameterization.remainder.lo, state.parameter_rem_lo),
            "parameter_remainder_hi": torch.equal(object_result.final_parameterization.remainder.hi, state.parameter_rem_hi),
            "phi_buffer": torch.equal(symbolic.phi_buffer, state.phi_buffer),
            "j_lo": torch.equal(symbolic.j_buffer.lo, state.j_lo),
            "j_hi": torch.equal(symbolic.j_buffer.hi, state.j_hi),
            "inverse_scale": torch.equal(symbolic.inverse_scale, state.inverse_scale),
            "queue_count": bool(torch.all(state.queue_count == symbolic.count)),
            "tube_hull_lo": torch.equal(state.tube_hull_lo, functional.tube_lo.amin(dim=1)),
            "tube_hull_hi": torch.equal(state.tube_hull_hi, functional.tube_hi.amax(dim=1)),
            "first_failure": object_result.first_failure_step == functional.first_failure_step,
        }
        rows.append(
            {
                "device": device_name,
                "batch": batch,
                "steps": steps,
                "bit_exact": all(comparisons.values()),
                "comparisons": comparisons,
                "object_completed": object_result.completed,
                "functional_validated_steps": functional.validated_steps,
                "functional_host_synchronizations": functional.host_synchronizations,
                "functional_device_transfers": functional.device_transfers,
            }
        )
    lo, hi = _partition(8, torch.device("cpu"))
    plan = fixed_support_kernel_plan(support, device="cpu", dtype=torch.float64)
    trace = fixed_support_functional_verify(lo, hi, plan, step_size=0.01, steps=10, trace=True)
    summary = fixed_support_functional_verify(lo, hi, plan, step_size=0.01, steps=10, trace=False)
    summary_trace_fields = {
        field: torch.equal(getattr(trace.final_state, field), getattr(summary.final_state, field))
        for field in trace.final_state.__dataclass_fields__
    }
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    artifact = {
        "schema": "fixed_support_functional_equivalence_v1",
        "source_sha": source_sha,
        "support_sha256": support.support_sha256,
        "dtype": "torch.float64",
        "rows": rows,
        "all_object_functional_bit_exact": all(row["bit_exact"] for row in rows),
        "summary_trace_fields": summary_trace_fields,
        "summary_trace_final_state_bit_exact": all(summary_trace_fields.values()),
        "functional_state_tensor_count": len(trace.final_state.tensors()),
        "failure_semantics_gate": "covered by focused unit test with per-batch freeze and first-failure equality",
    }
    _write_json(args.output, artifact)
    print(json.dumps(artifact, sort_keys=True, allow_nan=False))
    return 0 if artifact["all_object_functional_bit_exact"] and artifact["summary_trace_final_state_bit_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
