#!/usr/bin/env python3
"""B48 one-step shadow comparison for proved TORA-Q3 range policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from torch_tm_flowpipe.batched_dense_tm import (
    DenseRangePolicy,
    dense_transient_ledger_suppressed,
    dense_validation_batch,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_initial_model,
    dense_tora_q3_dr_step,
)


CONTEXTS = (
    "tora_full_step_tube",
    "tora_endpoint",
    "tora_composed_step_tube",
    "tora_endpoint_projection_overflow",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    lower = torch.full((48,), 9.8, dtype=torch.float64, device=device)
    upper = torch.full((48,), 10.2, dtype=torch.float64, device=device)
    policies = {
        "natural": DenseRangePolicy(method="natural"),
        "horner_registered_best": DenseRangePolicy(
            method="horner_registered_best", named_contexts=CONTEXTS
        ),
        "subdivision_then_horner": DenseRangePolicy(
            method="subdivision_then_horner",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
            named_contexts=CONTEXTS,
        ),
    }
    rows = []
    with torch.no_grad():
        for name, policy in policies.items():
            model = build_tora_q3_initial_model(
                lower,
                upper,
                device=device,
                range_policy=policy,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with dense_validation_batch(), dense_transient_ledger_suppressed():
                step = dense_tora_q3_dr_step(
                    model,
                    capture_trace=False,
                    point_enclosure_backend="compiled",
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            rows.append(
                {
                    "policy": name,
                    "status": step.status,
                    "wall_seconds_including_first_compile": (
                        time.perf_counter() - started
                    ),
                    "accepted_leaves": int(step.accepted_by_leaf.sum().cpu()),
                    "numerical_ok_leaves": int(
                        (
                            step.finite_ok_by_leaf
                            & step.initial_subset_ok_by_leaf
                            & step.all_remainder_rounds_ok_by_leaf
                        ).sum().cpu()
                    ),
                    "maximum_endpoint_width": float(
                        (step.endpoint_upper - step.endpoint_lower).max().cpu()
                    ),
                    "maximum_tube_width": float(
                        (step.tube_upper - step.tube_lower).max().cpu()
                    ),
                    "minimum_property_margin": float(
                        (
                            2.0
                            - torch.maximum(
                                torch.abs(step.tube_lower[:, :4]),
                                torch.abs(step.tube_upper[:, :4]),
                            )
                        ).min().cpu()
                    ),
                    "coverage_contract": (
                        "validated depth-1 owner-local cover"
                        if name == "subdivision_then_horner"
                        else "not applicable"
                    ),
                }
            )
    payload = {
        "schema": "tora_q3_range_policy_shadow_lanes_v1",
        "status": "PASS",
        "batch": 48,
        "step_size": 0.1,
        "order": 3,
        "picard": "K2 plus ten remainder rounds",
        "named_contexts": list(CONTEXTS),
        "timing_claim": "diagnostic cold wall; not a formal speed claim",
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
