#!/usr/bin/env python3
"""Run the native Torch period-local common-control TORA-Q3 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from torch_tm_flowpipe.batched_dense_tm import BatchedMonomialBasis
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    compose_tora_q3_step,
    dense_tora_q3_dr_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
    tora_q3_boundary_from_model,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--expected-controller-trace-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", default="torch_common_control_plant_replay")
    parser.add_argument("--periods", type=int, default=20)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    trace_path = args.controller_trace.resolve()
    observed_trace_hash = sha256(trace_path)
    if (
        args.expected_controller_trace_sha256
        and observed_trace_hash != args.expected_controller_trace_sha256
    ):
        raise ValueError("controller trace hash mismatch")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    rows = trace["rows"]
    if len(rows) != 20 or not 1 <= args.periods <= 20:
        raise ValueError("replay requires one to twenty rows from the frozen trace")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_default_dtype(torch.float64)
    basis_started = time.perf_counter()
    basis = BatchedMonomialBasis.build(6, 3, str(device))
    synchronize(device)
    basis_seconds = time.perf_counter() - basis_started
    if basis.num_terms != 84:
        raise RuntimeError("complete-Q3 basis does not have 84 slots")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    raw_path = output / "segments.jsonl"
    summaries = []
    first_failure = None
    completed = 0
    plant_total = 0.0
    reset_total = 0.0
    serialization_total = 0.0
    run_started = time.perf_counter()
    with raw_path.open("x", encoding="utf-8") as handle:
        header = {
            "schema": "torch_tora_q3_common_control_header_v1",
            "run_id": args.run_id,
            "lane": "common_control_plant_replay",
            "period_local_observation_restart": True,
            "controller_trace_sha256": observed_trace_hash,
            "basis_variables": [
                "local_time", "x1_parameter", "x2_parameter",
                "x3_parameter", "x4_parameter", "u1_parameter",
            ],
            "basis_exponents": basis.exponents.detach().cpu().tolist(),
            "basis_fingerprint": basis.fingerprint,
            "slot_count": basis.num_terms,
            "device": str(device),
            "dtype": "float64",
            "polynomial_picard_rounds": 2,
            "remainder_rounds": 10,
            "sine_order": 2,
        }
        handle.write(json.dumps(header, separators=(",", ":")) + "\n")
        for period, row in enumerate(rows[: args.periods], start=1):
            state_lo = torch.tensor(
                row["pre_controller_state_box"]["lower"],
                dtype=torch.float64,
                device=device,
            )
            state_hi = torch.tensor(
                row["pre_controller_state_box"]["upper"],
                dtype=torch.float64,
                device=device,
            )
            control = row["u1_interval_installed_for_next_ten_segments"]
            control_lo = torch.tensor(
                control["lower"], dtype=torch.float64, device=device
            ).reshape(-1)
            control_hi = torch.tensor(
                control["upper"], dtype=torch.float64, device=device
            ).reshape(-1)
            model = build_tora_q3_box_model(
                state_lo,
                state_hi,
                control_lo,
                control_hi,
                device=device,
            )
            boundary = tora_q3_boundary_from_model(model)
            carry = identity_tora_q3_carry(48, device=device)
            for local_segment in range(1, 11):
                segment = (period - 1) * 10 + local_segment
                synchronize(device)
                reset_started = time.perf_counter()
                local_model, carry = normalize_tora_q3_boundary(
                    boundary, carry
                )
                synchronize(device)
                reset_seconds = time.perf_counter() - reset_started
                reset_total += reset_seconds
                synchronize(device)
                plant_started = time.perf_counter()
                try:
                    local_step = dense_tora_q3_dr_step(local_model)
                    step = compose_tora_q3_step(local_step, carry)
                    error = None
                except (RuntimeError, ValueError) as exception:
                    step = None
                    error = f"{type(exception).__name__}: {exception}"
                synchronize(device)
                plant_seconds = time.perf_counter() - plant_started
                plant_total += plant_seconds
                if step is None:
                    first_failure = {
                        "segment": segment,
                        "reason": "fail_closed_exception",
                        "message": error,
                    }
                    break
                accepted = step.accepted_by_leaf
                payload: dict[str, Any] = {
                    "schema": "torch_tora_q3_common_control_segment_v1",
                    "run_id": args.run_id,
                    "segment_index": segment,
                    "physical_time": segment * 0.1,
                    "controller_period": period,
                    "local_segment": local_segment,
                    "leaf_id": list(range(48)),
                    "accepted": values(accepted),
                    "endpoint": {
                        "lower": values(step.endpoint_lower),
                        "upper": values(step.endpoint_upper),
                    },
                    "tube": {
                        "lower": values(step.tube_lower),
                        "upper": values(step.tube_upper),
                    },
                    "polynomial_coefficient_vector": values(
                        step.segment_tm.poly.coeffs
                    ),
                    "interval_remainder": {
                        "lower": values(step.segment_tm.rem_lo),
                        "upper": values(step.segment_tm.rem_hi),
                    },
                    "validation": {
                        "initial_shrink_mask": values(
                            step.initial_shrink_mask
                        ),
                        "initial_margin": values(step.initial_margin),
                        "rounds": list(step.round_trace),
                    },
                    "property_margin": {
                        "tube": values(
                            2.0
                            - torch.maximum(
                                torch.abs(step.tube_lower[:, :4]),
                                torch.abs(step.tube_upper[:, :4]),
                            )
                        ),
                        "endpoint": values(
                            2.0
                            - torch.maximum(
                                torch.abs(step.endpoint_lower[:, :4]),
                                torch.abs(step.endpoint_upper[:, :4]),
                            )
                        ),
                    },
                    "plant_seconds": plant_seconds,
                    "reset_seconds": reset_seconds,
                    "basis_fingerprint": basis.fingerprint,
                    "ledger": step.segment_tm.ledger.intervals(),
                }
                canonical = json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                payload["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
                serialization_started = time.perf_counter()
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                serialization_total += time.perf_counter() - serialization_started
                summaries.append(
                    {
                        "segment": segment,
                        "accepted_leaves": int(accepted.sum().item()),
                        "maximum_endpoint_width": float(
                            (step.endpoint_upper - step.endpoint_lower).max().item()
                        ),
                        "maximum_tube_width": float(
                            (step.tube_upper - step.tube_lower).max().item()
                        ),
                        "minimum_property_margin": float(
                            (
                                2.0
                                - torch.maximum(
                                    torch.abs(step.tube_lower[:, :4]),
                                    torch.abs(step.tube_upper[:, :4]),
                                )
                            ).min().item()
                        ),
                        "plant_seconds": plant_seconds,
                        "reset_seconds": reset_seconds,
                    }
                )
                if not step.accepted:
                    first_failure = {
                        "segment": segment,
                        "reason": "acceptance",
                        "failed_leaf_ids": torch.nonzero(~accepted).flatten().cpu().tolist(),
                        "message": step.message,
                    }
                    break
                completed = segment
                boundary = project_tora_q3_endpoint_to_affine(
                    local_step.segment_tm
                )
            if first_failure is not None:
                break
    synchronize(device)
    summary = {
        "schema": "torch_tora_q3_common_control_summary_v1",
        "run_id": args.run_id,
        "lane": "common_control_plant_replay",
        "period_local_observation_restart": True,
        "status": "VERIFIED" if completed == args.periods * 10 else "FAILED",
        "completed_segments": completed,
        "certified_horizon": completed * 0.1,
        "requested_periods": args.periods,
        "first_failure": first_failure,
        "basis_construction_seconds": basis_seconds,
        "plant_seconds": plant_total,
        "reset_seconds": reset_total,
        "serialization_seconds": serialization_total,
        "wall_seconds_including_serialization": time.perf_counter() - run_started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "segments": summaries,
        "controller_trace_sha256": observed_trace_hash,
        "segments_sha256": sha256(raw_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("status", "completed_segments", "certified_horizon", "first_failure", "plant_seconds")}))
    return 0 if summary["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
