#!/usr/bin/env python3
"""Run native Torch TORA-Q3 with native auto_LiRPA controller bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from torch_tm_flowpipe.tora_controller import (
    EXPECTED_ORIGINAL_CONTROLLER_SHA256,
    ToraAutoLirpaControllerBounder,
)
from torch_tm_flowpipe.tora_q3 import (
    compose_tora_q3_boundary,
    compose_tora_q3_step,
    dense_tora_q3_dr_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
    tora_q3_boundary_box,
    tora_q3_boundary_from_model,
    build_tora_q3_initial_model,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def values(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def maximum_abs(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b), initial=0.0))


def onnx_reference_nominal(model_path: Path, states: np.ndarray) -> np.ndarray:
    import onnx
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(onnx.load(model_path))
    value = states.astype(np.float32).reshape(-1, 1, 1, 4)
    return np.asarray(evaluator.run(None, {"input": value})[0]).reshape(-1, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-trace", type=Path)
    parser.add_argument("--expected-controller-trace-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--periods", type=int, default=20)
    parser.add_argument("--run-id", default="torch_native_full_closed_loop_tora_q3")
    args = parser.parse_args()
    if not 1 <= args.periods <= 20:
        raise ValueError("periods must be between one and twenty")
    controller_value = os.environ.get("TORA_CONTROLLER_PATH")
    if not controller_value:
        raise RuntimeError("TORA_CONTROLLER_PATH is required")
    controller_path = Path(controller_value).resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    observed_trace = None
    observed_trace_hash = None
    if args.controller_trace is not None:
        trace_path = args.controller_trace.resolve()
        observed_trace_hash = sha256(trace_path)
        if args.expected_controller_trace_sha256 and observed_trace_hash != args.expected_controller_trace_sha256:
            raise ValueError("controller trace hash mismatch")
        observed_trace = json.loads(trace_path.read_text(encoding="utf-8"))["rows"]

    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    zeros = torch.zeros(48, dtype=torch.float64, device=device)
    initial_model = build_tora_q3_initial_model(zeros, zeros, device=device)
    boundary = tora_q3_boundary_from_model(initial_model)
    carry = identity_tora_q3_carry(48, device=device)
    build_started = time.perf_counter()
    bounder = ToraAutoLirpaControllerBounder(
        controller_path,
        boundary,
        device=device,
        expected_sha256=EXPECTED_ORIGINAL_CONTROLLER_SHA256,
    )
    synchronize(device)
    controller_build_seconds = time.perf_counter() - build_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    nominal_states = boundary.center[:, :4]
    with torch.no_grad():
        native_nominal = bounder.nominal(nominal_states).detach().cpu().numpy()
    reference_nominal = onnx_reference_nominal(
        controller_path, nominal_states.detach().cpu().numpy()
    )
    nominal_maximum_abs_error = maximum_abs(native_nominal, reference_nominal)

    raw_path = output / "segments.jsonl"
    controller_path_out = output / "controller_updates.jsonl"
    segment_summaries = []
    controller_summaries = []
    first_failure = None
    completed = 0
    plant_seconds = 0.0
    controller_bound_seconds = 0.0
    controller_composition_seconds = 0.0
    normalization_seconds = 0.0
    serialization_seconds = 0.0
    run_started = time.perf_counter()
    with raw_path.open("x", encoding="utf-8") as plant_handle, controller_path_out.open("x", encoding="utf-8") as controller_handle:
        for period in range(1, args.periods + 1):
            if period > 1:
                boundary = compose_tora_q3_boundary(boundary, carry)
                carry = identity_tora_q3_carry(48, device=device)
            pre_lower, pre_upper = tora_q3_boundary_box(boundary)
            controller_result = bounder.bound(boundary)
            controller_bound_seconds += controller_result.timing["bound_seconds"]
            controller_composition_seconds += controller_result.timing["composition_seconds"]
            observed = observed_trace[period - 1] if observed_trace is not None else None
            comparison = None
            if observed is not None:
                comparison = {
                    "pre_controller_state_lower_max_abs": maximum_abs(values(pre_lower[:, :4]), observed["pre_controller_state_box"]["lower"]),
                    "pre_controller_state_upper_max_abs": maximum_abs(values(pre_upper[:, :4]), observed["pre_controller_state_box"]["upper"]),
                    "output_before_lower_max_abs": maximum_abs(values(controller_result.output_lower_before_outward), observed["controller_output_interval_before_outward_composition"]["lower"]),
                    "output_before_upper_max_abs": maximum_abs(values(controller_result.output_upper_before_outward), observed["controller_output_interval_before_outward_composition"]["upper"]),
                    "output_after_lower_max_abs": maximum_abs(values(controller_result.output_lower_after_outward), observed["controller_output_interval_after_outward_composition"]["lower"]),
                    "output_after_upper_max_abs": maximum_abs(values(controller_result.output_upper_after_outward), observed["controller_output_interval_after_outward_composition"]["upper"]),
                }
            controller_payload = {
                "schema": "torch_native_tora_q3_controller_update_v1",
                "run_id": args.run_id,
                "controller_period": period,
                "segment_index": (period - 1) * 10 + 1,
                "physical_time": float(period - 1),
                "leaf_id": list(range(48)),
                "pre_controller_state_box": {"lower": values(pre_lower[:, :4]), "upper": values(pre_upper[:, :4])},
                "output_before_outward": {"lower": values(controller_result.output_lower_before_outward), "upper": values(controller_result.output_upper_before_outward)},
                "output_after_outward": {"lower": values(controller_result.output_lower_after_outward), "upper": values(controller_result.output_upper_after_outward)},
                "lower_slope": values(controller_result.lower_slope),
                "upper_slope": values(controller_result.upper_slope),
                "raw_lower_bias": values(controller_result.raw_lower_bias),
                "raw_upper_bias": values(controller_result.raw_upper_bias),
                "maximum_slope_gap": controller_result.maximum_slope_gap,
                "timing": controller_result.timing,
                "comparison_to_xiangru_observation": comparison,
            }
            controller_payload["content_sha256"] = hashlib.sha256(json.dumps(controller_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            controller_handle.write(json.dumps(controller_payload, separators=(",", ":")) + "\n")
            controller_summaries.append({
                "controller_period": period,
                "maximum_slope_gap": controller_result.maximum_slope_gap,
                "minimum_output": float(controller_result.output_lower_after_outward.min()),
                "maximum_output": float(controller_result.output_upper_after_outward.max()),
                "comparison_to_xiangru_observation": comparison,
                **controller_result.timing,
            })
            boundary = controller_result.controlled_boundary
            carry = identity_tora_q3_carry(48, device=device)

            for local_segment in range(1, 11):
                segment = (period - 1) * 10 + local_segment
                synchronize(device)
                normalization_started = time.perf_counter()
                local_model, carry = normalize_tora_q3_boundary(boundary, carry)
                synchronize(device)
                normalization_elapsed = time.perf_counter() - normalization_started
                normalization_seconds += normalization_elapsed
                synchronize(device)
                plant_started = time.perf_counter()
                try:
                    local_step = dense_tora_q3_dr_step(
                        local_model, capture_trace=False
                    )
                    step = compose_tora_q3_step(local_step, carry)
                    error = None
                except (RuntimeError, ValueError) as exception:
                    step = None
                    error = f"{type(exception).__name__}: {exception}"
                synchronize(device)
                plant_elapsed = time.perf_counter() - plant_started
                plant_seconds += plant_elapsed
                if step is None:
                    first_failure = {"segment": segment, "reason": "fail_closed_exception", "message": error}
                    break
                payload = {
                    "schema": "torch_native_full_closed_loop_tora_q3_segment_v1",
                    "run_id": args.run_id,
                    "lane": "native_full_closed_loop",
                    "segment_index": segment,
                    "physical_time": segment * 0.1,
                    "controller_period": period,
                    "local_segment": local_segment,
                    "leaf_id": list(range(48)),
                    "accepted": values(step.accepted_by_leaf),
                    "endpoint": {"lower": values(step.endpoint_lower), "upper": values(step.endpoint_upper)},
                    "tube": {"lower": values(step.tube_lower), "upper": values(step.tube_upper)},
                    "interval_remainder": {"lower": values(step.segment_tm.rem_lo), "upper": values(step.segment_tm.rem_hi)},
                    "property_margin": values(2.0 - torch.maximum(torch.abs(step.tube_lower[:, :4]), torch.abs(step.tube_upper[:, :4]))),
                    "plant_seconds": plant_elapsed,
                    "normalization_seconds": normalization_elapsed,
                }
                serialization_started = time.perf_counter()
                plant_handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                serialization_seconds += time.perf_counter() - serialization_started
                segment_summaries.append({
                    "segment": segment,
                    "accepted_leaves": int(step.accepted_by_leaf.sum().item()),
                    "maximum_endpoint_width": float((step.endpoint_upper - step.endpoint_lower).max().item()),
                    "maximum_tube_width": float((step.tube_upper - step.tube_lower).max().item()),
                    "minimum_property_margin": float((2.0 - torch.maximum(torch.abs(step.tube_lower[:, :4]), torch.abs(step.tube_upper[:, :4]))).min().item()),
                    "plant_seconds": plant_elapsed,
                })
                if not step.accepted:
                    first_failure = {
                        "segment": segment,
                        "reason": "acceptance",
                        "failed_leaf_ids": torch.nonzero(~step.accepted_by_leaf).flatten().cpu().tolist(),
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
        "schema": "torch_native_full_closed_loop_tora_q3_summary_v1",
        "run_id": args.run_id,
        "lane": "native_full_closed_loop",
        "status": "VERIFIED" if completed == args.periods * 10 else "FAILED",
        "requested_periods": args.periods,
        "completed_segments": completed,
        "certified_horizon": completed * 0.1,
        "first_failure": first_failure,
        "controller_sha256": EXPECTED_ORIGINAL_CONTROLLER_SHA256,
        "controller_trace_sha256": observed_trace_hash,
        "controller_build_seconds": controller_build_seconds,
        "controller_bound_seconds": controller_bound_seconds,
        "controller_composition_seconds": controller_composition_seconds,
        "plant_seconds": plant_seconds,
        "normalization_seconds": normalization_seconds,
        "serialization_seconds": serialization_seconds,
        "wall_seconds_including_serialization": time.perf_counter() - run_started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "nominal_gate": {
            "reference": "ONNX ReferenceEvaluator float32 graph",
            "candidate": "NativeToraController float64 flattened graph",
            "maximum_absolute_error": nominal_maximum_abs_error,
        },
        "controller_updates": controller_summaries,
        "segments": segment_summaries,
        "segments_sha256": sha256(raw_path),
        "controller_updates_sha256": sha256(controller_path_out),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "completed_segments", "certified_horizon", "first_failure", "nominal_gate")}))
    return 0 if summary["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
