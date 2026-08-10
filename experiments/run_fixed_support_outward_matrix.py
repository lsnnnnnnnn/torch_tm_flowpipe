#!/usr/bin/env python3
"""Run the bounded CPU outward fixed-support checkpoint matrix once per batch."""
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
    fixed_support_kernel_plan,
)
from torch_tm_flowpipe.fixed_support_outward import (
    OutwardIntervalTensor,
    fixed_support_outward_vdp_step,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (1, 10, 100, 1000)


def _partition(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    left = math.isqrt(batch)
    while batch % left:
        left -= 1
    split_x, split_y = batch // left, left
    x = torch.linspace(1.1, 1.4, split_x + 1, dtype=torch.float64)
    y = torch.linspace(2.35, 2.45, split_y + 1, dtype=torch.float64)
    lo: list[torch.Tensor] = []
    hi: list[torch.Tensor] = []
    for i in range(split_x):
        for j in range(split_y):
            lo.append(torch.stack((x[i], y[j])))
            hi.append(torch.stack((x[i + 1], y[j + 1])))
    return torch.stack(lo), torch.stack(hi)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", default="1,64")
    args = parser.parse_args()
    batches = tuple(int(item) for item in args.batches.split(",") if item)
    if not batches or any(batch <= 0 for batch in batches):
        raise ValueError("batches must be positive")

    descriptor = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    plan = fixed_support_kernel_plan(descriptor, device="cpu", dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    for batch in batches:
        initial_lo, initial_hi = _partition(batch)
        box = OutwardIntervalTensor(initial_lo, initial_hi)
        active = torch.ones(batch, dtype=torch.bool)
        failures = torch.full((batch,), -1, dtype=torch.long)
        started = time.perf_counter()
        stopped_after_failure = False
        checkpoint_index = 0
        for step_index in range(1, CHECKPOINTS[-1] + 1):
            step = fixed_support_outward_vdp_step(box, plan, step_size=0.01)
            accepted = active & step.accepted_mask
            failed = active & ~step.accepted_mask
            failures = torch.where(failed & (failures < 0), step_index - 1, failures)
            box = OutwardIntervalTensor(
                torch.where(accepted[:, None], step.endpoint.lo, box.lo),
                torch.where(accepted[:, None], step.endpoint.hi, box.hi),
            )
            active = accepted
            if step_index == CHECKPOINTS[checkpoint_index]:
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "batch": batch,
                        "steps": step_index,
                        "requested_horizon": step_index * 0.01,
                        "completed": bool(torch.all(active)),
                        "active_count": int(active.sum()),
                        "first_failure_indices": [int(value) for value in failures.tolist()],
                        "endpoint_lo": box.lo.tolist(),
                        "endpoint_hi": box.hi.tolist(),
                        "cumulative_runtime_s": elapsed,
                        "outward_decision": "completed" if bool(torch.all(active)) else "failed_closed",
                        "ordinary_contained_one_step": step_index == 1,
                        "host_synchronizations_in_solver_core": 0,
                        "final_decision_host_synchronizations": 1,
                        "device_transfers": 0,
                    }
                )
                checkpoint_index += 1
                if checkpoint_index == len(CHECKPOINTS):
                    break
            if not bool(torch.any(active)):
                stopped_after_failure = True
                break
        if stopped_after_failure:
            elapsed = time.perf_counter() - started
            completed_steps = step_index
            while checkpoint_index < len(CHECKPOINTS):
                requested = CHECKPOINTS[checkpoint_index]
                rows.append(
                    {
                        "batch": batch,
                        "steps": requested,
                        "requested_horizon": requested * 0.01,
                        "completed": False,
                        "active_count": 0,
                        "first_failure_indices": [int(value) for value in failures.tolist()],
                        "endpoint_lo": box.lo.tolist(),
                        "endpoint_hi": box.hi.tolist(),
                        "cumulative_runtime_s": elapsed,
                        "computation_stopped_after_all_batches_failed_at_step": completed_steps,
                        "outward_decision": "failed_closed",
                        "ordinary_contained_one_step": False,
                        "host_synchronizations_in_solver_core": 0,
                        "final_decision_host_synchronizations": 1,
                        "device_transfers": 0,
                    }
                )
                checkpoint_index += 1

    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    result = {
        "schema": "fixed_support_outward_matrix_v1",
        "source_sha": source_sha,
        "support_sha256": descriptor.support_sha256,
        "dtype": "torch.float64",
        "device": "cpu",
        "step_size": 0.01,
        "checkpoints": list(CHECKPOINTS),
        "rows": rows,
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "numerical_soundness_class": "safeguarded outward under declared IEEE/backend assumptions",
        "numerical_soundness_scope": "primitive / reference multi-step lane",
        "formal_claim_eligible": False,
        "performance_measurement_eligible": True,
        "cross_tool_ranking_eligible": False,
        "implemented_negative_outcome": (
            None if all(row["completed"] for row in rows) else "FIXED_SUPPORT_FORMAL_SOUNDNESS_NOT_CLOSED"
        ),
    }
    _write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
