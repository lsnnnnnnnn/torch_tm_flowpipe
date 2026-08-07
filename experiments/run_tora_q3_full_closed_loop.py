#!/usr/bin/env python3
"""Run native Torch TORA-Q3 with native auto_LiRPA controller bounds."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from torch_tm_flowpipe.batched_dense_tm import (
    DenseRangePolicy,
    dense_transient_ledger_suppressed,
    dense_validation_batch,
)
from torch_tm_flowpipe.tora_controller import (
    EXPECTED_ORIGINAL_CONTROLLER_SHA256,
    ToraAutoLirpaControllerBounder,
)
from torch_tm_flowpipe.tora_q3 import (
    compose_tora_q3_boundary,
    compose_tora_q3_step,
    dense_tora_q3_dr_step,
    identity_tora_q3_carry,
    install_interval_control_on_boundary,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
    ToraQ3AffineBoundary,
    tora_q3_boundary_box,
    tora_q3_boundary_from_model,
    build_tora_q3_initial_model,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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


def width_statistics(lower: torch.Tensor, upper: torch.Tensor) -> dict[str, float]:
    width = (upper - lower).detach().reshape(-1)
    return {
        "median": float(torch.median(width).cpu()),
        "maximum": float(torch.max(width).cpu()),
        "sum": float(torch.sum(width).cpu()),
    }


def affine_boundary_from_box(
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> ToraQ3AffineBoundary:
    center = lower + 0.5 * (upper - lower)
    radius = torch.maximum(center - lower, upper - center)
    linear = torch.diag_embed(radius)
    zeros = torch.zeros_like(center)
    return ToraQ3AffineBoundary(center, linear, zeros, zeros.clone())


def affine_width_decomposition(
    boundary: ToraQ3AffineBoundary,
) -> dict[str, Any]:
    linear_radius = torch.sum(torch.abs(boundary.linear), dim=2)
    remainder_radius = 0.5 * (
        boundary.remainder_upper - boundary.remainder_lower
    )
    lower, upper = tora_q3_boundary_box(boundary)
    return {
        "box_width": width_statistics(lower, upper),
        "linear_radius": width_statistics(
            torch.zeros_like(linear_radius), linear_radius
        ),
        "remainder_radius": width_statistics(
            torch.zeros_like(remainder_radius), remainder_radius
        ),
        "maximum_absolute_center": float(torch.max(torch.abs(boundary.center)).cpu()),
    }


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
    parser.add_argument(
        "--lane",
        choices=(
            "baseline_native",
            "tight_endpoint_box_controller",
            "physical_endpoint_projection",
            "k3_picard",
            "horner_registered_best",
            "subdivision_then_horner",
        ),
        default="baseline_native",
    )
    parser.add_argument(
        "--point-enclosure-backend",
        choices=("eager", "compiled"),
        default="eager",
    )
    parser.add_argument("--optimized-math", action="store_true")
    parser.add_argument("--continue-after-property-failure", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.periods <= 20:
        raise ValueError("periods must be between one and twenty")
    polynomial_picard_rounds = 3 if args.lane == "k3_picard" else 2
    run_config = {
        "batch": 48,
        "continue_after_property_failure": args.continue_after_property_failure,
        "device": args.device,
        "dtype": "float64",
        "lane": args.lane,
        "optimized_math": args.optimized_math,
        "order": 3,
        "periods": args.periods,
        "point_enclosure_backend": args.point_enclosure_backend,
        "polynomial_picard_rounds": polynomial_picard_rounds,
        "property": "abs(x1..x4) <= 2",
        "range_policy": (
            args.lane
            if args.lane in {
                "horner_registered_best",
                "subdivision_then_horner",
            }
            else "natural"
        ),
        "remainder_picard_rounds": 10,
        "step_size": 0.1,
    }
    source_sha256 = {
        "experiments/run_tora_q3_full_closed_loop.py": sha256(
            Path(__file__).resolve()
        ),
        "src/torch_tm_flowpipe/batched_dense_tm.py": sha256(
            REPOSITORY_ROOT / "src/torch_tm_flowpipe/batched_dense_tm.py"
        ),
        "src/torch_tm_flowpipe/tora_q3.py": sha256(
            REPOSITORY_ROOT / "src/torch_tm_flowpipe/tora_q3.py"
        ),
    }
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
    failure_range_contexts = (
        "tora_full_step_tube",
        "tora_endpoint",
        "tora_composed_step_tube",
        "tora_endpoint_projection_overflow",
    )
    range_policy = (
        DenseRangePolicy(
            method=args.lane,
            max_depth=(1 if args.lane == "subdivision_then_horner" else 0),
            max_leaves=4,
            split_vars=(0, 1),
            named_contexts=failure_range_contexts,
        )
        if args.lane in {"horner_registered_best", "subdivision_then_horner"}
        else DenseRangePolicy(method="natural")
    )
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
    diagnostic_failure = None
    formal_completed = 0
    diagnostic_completed = 0
    replay_points: dict[str, Any] = {}
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
            controller_input_boundary = (
                affine_boundary_from_box(pre_lower, pre_upper)
                if args.lane == "tight_endpoint_box_controller"
                else boundary
            )
            controller_result = bounder.bound(controller_input_boundary)
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
                "lane": args.lane,
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
                "controller_input_representation": (
                    "independent_exact_endpoint_box"
                    if args.lane == "tight_endpoint_box_controller"
                    else "correlation_aware_affine_boundary"
                ),
                "controller_input_width": width_statistics(
                    pre_lower[:, :4], pre_upper[:, :4]
                ),
                "controller_output_before_width": width_statistics(
                    controller_result.output_lower_before_outward,
                    controller_result.output_upper_before_outward,
                ),
                "controller_output_after_width": width_statistics(
                    controller_result.output_lower_after_outward,
                    controller_result.output_upper_after_outward,
                ),
            }
            controller_payload["content_sha256"] = hashlib.sha256(json.dumps(controller_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            controller_handle.write(json.dumps(controller_payload, separators=(",", ":")) + "\n")
            replay_name = {2: "R1", 5: "R2"}.get(period)
            if replay_name is not None and replay_name in replay_points:
                replay_points[replay_name]["controller_refresh"] = (
                    controller_payload
                )
            controller_summaries.append({
                "controller_period": period,
                "maximum_slope_gap": controller_result.maximum_slope_gap,
                "minimum_output": float(controller_result.output_lower_after_outward.min()),
                "maximum_output": float(controller_result.output_upper_after_outward.max()),
                "comparison_to_xiangru_observation": comparison,
                **controller_result.timing,
            })
            boundary = (
                install_interval_control_on_boundary(
                    boundary,
                    controller_result.output_lower_after_outward.reshape(-1),
                    controller_result.output_upper_after_outward.reshape(-1),
                )
                if args.lane == "tight_endpoint_box_controller"
                else controller_result.controlled_boundary
            )
            carry = identity_tora_q3_carry(48, device=device)

            for local_segment in range(1, 11):
                segment = (period - 1) * 10 + local_segment
                synchronize(device)
                normalization_started = time.perf_counter()
                local_model, carry = normalize_tora_q3_boundary(
                    boundary, carry, range_policy=range_policy
                )
                synchronize(device)
                normalization_elapsed = time.perf_counter() - normalization_started
                normalization_seconds += normalization_elapsed
                synchronize(device)
                plant_started = time.perf_counter()
                try:
                    validation_scope = (
                        dense_validation_batch()
                        if args.optimized_math
                        else nullcontext()
                    )
                    ledger_scope = (
                        dense_transient_ledger_suppressed()
                        if args.optimized_math
                        else nullcontext()
                    )
                    with validation_scope:
                        with ledger_scope:
                            local_step = dense_tora_q3_dr_step(
                                local_model,
                                capture_trace=False,
                                polynomial_picard_rounds=polynomial_picard_rounds,
                                point_enclosure_backend=(
                                    args.point_enclosure_backend
                                ),
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
                current_local_projection = project_tora_q3_endpoint_to_affine(
                    local_step.segment_tm
                )
                current_physical_projection = compose_tora_q3_boundary(
                    current_local_projection, carry
                )
                physical_endpoint_projection = project_tora_q3_endpoint_to_affine(
                    step.segment_tm
                )
                current_projection_lower, current_projection_upper = (
                    tora_q3_boundary_box(current_physical_projection)
                )
                physical_projection_lower, physical_projection_upper = (
                    tora_q3_boundary_box(physical_endpoint_projection)
                )
                physical_endpoint_tm = step.segment_tm.endpoint(0, 0.1)
                endpoint_poly_lower, endpoint_poly_upper = (
                    physical_endpoint_tm.poly.range_bound(
                        physical_endpoint_tm.domain_lo,
                        physical_endpoint_tm.domain_hi,
                        policy=physical_endpoint_tm.range_policy,
                        context="tora_endpoint_width_attribution",
                    )
                )
                numerical_ok_by_leaf = (
                    step.finite_ok_by_leaf
                    & step.initial_subset_ok_by_leaf
                    & step.all_remainder_rounds_ok_by_leaf
                )
                property_ok_by_leaf = (
                    step.local_property_ok_by_leaf
                    & step.composed_property_ok_by_leaf
                )
                ledger_width = {
                    category: width_statistics(entry_lo, entry_hi)
                    for category, (entry_lo, entry_hi) in (
                        step.segment_tm.ledger.entries.items()
                    )
                }
                payload = {
                    "schema": "torch_native_full_closed_loop_tora_q3_segment_v1",
                    "run_id": args.run_id,
                    "lane": args.lane,
                    "segment_index": segment,
                    "physical_time": segment * 0.1,
                    "controller_period": period,
                    "local_segment": local_segment,
                    "leaf_id": list(range(48)),
                    "accepted": values(step.accepted_by_leaf),
                    "predicates": {
                        "finite_ok_by_leaf": values(step.finite_ok_by_leaf),
                        "initial_subset_ok_by_leaf": values(
                            step.initial_subset_ok_by_leaf
                        ),
                        "all_remainder_rounds_ok_by_leaf": values(
                            step.all_remainder_rounds_ok_by_leaf
                        ),
                        "local_property_ok_by_leaf": values(
                            step.local_property_ok_by_leaf
                        ),
                        "composed_property_ok_by_leaf": values(
                            step.composed_property_ok_by_leaf
                        ),
                        "numerical_ok_by_leaf": values(numerical_ok_by_leaf),
                        "overall_accepted_by_leaf": values(step.accepted_by_leaf),
                    },
                    "endpoint": {"lower": values(step.endpoint_lower), "upper": values(step.endpoint_upper)},
                    "tube": {"lower": values(step.tube_lower), "upper": values(step.tube_upper)},
                    "interval_remainder": {"lower": values(step.segment_tm.rem_lo), "upper": values(step.segment_tm.rem_hi)},
                    "property_margin": values(2.0 - torch.maximum(torch.abs(step.tube_lower[:, :4]), torch.abs(step.tube_upper[:, :4]))),
                    "width_attribution": {
                        "composed_exact_endpoint_direct": width_statistics(
                            step.endpoint_lower, step.endpoint_upper
                        ),
                        "pre_projection_polynomial_range": width_statistics(
                            endpoint_poly_lower,
                            endpoint_poly_upper,
                        ),
                        "pre_projection_interval_remainder": width_statistics(
                            physical_endpoint_tm.rem_lo,
                            physical_endpoint_tm.rem_hi,
                        ),
                        "current_project_local_then_compose": width_statistics(
                            current_projection_lower, current_projection_upper
                        ),
                        "candidate_compose_then_project": width_statistics(
                            physical_projection_lower, physical_projection_upper
                        ),
                        "current_projection_inflation_maximum": float(
                            torch.max(
                                (current_projection_upper - current_projection_lower)
                                - (step.endpoint_upper - step.endpoint_lower)
                            ).cpu()
                        ),
                        "physical_projection_inflation_maximum": float(
                            torch.max(
                                (physical_projection_upper - physical_projection_lower)
                                - (step.endpoint_upper - step.endpoint_lower)
                            ).cpu()
                        ),
                        "current_affine_decomposition": affine_width_decomposition(
                            current_physical_projection
                        ),
                        "physical_affine_decomposition": affine_width_decomposition(
                            physical_endpoint_projection
                        ),
                    },
                    "ledger_widths": ledger_width,
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
                    "finite_ok_leaves": int(step.finite_ok_by_leaf.sum().item()),
                    "initial_subset_ok_leaves": int(
                        step.initial_subset_ok_by_leaf.sum().item()
                    ),
                    "all_remainder_rounds_ok_leaves": int(
                        step.all_remainder_rounds_ok_by_leaf.sum().item()
                    ),
                    "local_property_ok_leaves": int(
                        step.local_property_ok_by_leaf.sum().item()
                    ),
                    "composed_property_ok_leaves": int(
                        step.composed_property_ok_by_leaf.sum().item()
                    ),
                    "numerical_ok_leaves": int(numerical_ok_by_leaf.sum().item()),
                    "plant_seconds": plant_elapsed,
                })
                diagnostic_completed = segment
                if first_failure is None and not step.accepted:
                    first_failure = {
                        "segment": segment,
                        "reason": (
                            "numerical_certificate"
                            if not bool(torch.all(numerical_ok_by_leaf))
                            else "property"
                        ),
                        "failed_leaf_ids": torch.nonzero(~step.accepted_by_leaf).flatten().cpu().tolist(),
                        "message": step.message,
                    }
                if first_failure is None:
                    formal_completed = segment
                numerical_ok = bool(torch.all(numerical_ok_by_leaf))
                if not numerical_ok:
                    diagnostic_failure = {
                        "segment": segment,
                        "reason": "numerical_certificate",
                        "failed_leaf_ids": torch.nonzero(
                            ~numerical_ok_by_leaf
                        ).flatten().cpu().tolist(),
                    }
                    break
                if first_failure is not None and not args.continue_after_property_failure:
                    break
                boundary, carry = (
                    (physical_endpoint_projection, identity_tora_q3_carry(48, device=device))
                    if args.lane == "physical_endpoint_projection"
                    else (current_local_projection, carry)
                )
                if segment in {10, 40}:
                    replay_points[f"R{1 if segment == 10 else 2}"] = payload
            if first_failure is not None:
                if (
                    not args.continue_after_property_failure
                    or first_failure["reason"] != "property"
                ):
                    break
    synchronize(device)
    replay_path = output / "replay_points.json"
    replay_path.write_text(
        json.dumps(replay_points, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    requested_segments = args.periods * 10
    summary = {
        "schema": "torch_native_full_closed_loop_tora_q3_summary_v1",
        "run_id": args.run_id,
        "lane": args.lane,
        "status": (
            "VERIFIED" if formal_completed == requested_segments else "FAILED"
        ),
        "diagnostic_status": (
            "COMPLETED"
            if diagnostic_completed == requested_segments
            else "STOPPED"
        ),
        "continue_after_property_failure": args.continue_after_property_failure,
        "config": run_config,
        "config_sha256": canonical_sha256(run_config),
        "source_sha256": source_sha256,
        "point_enclosure_backend": args.point_enclosure_backend,
        "optimized_math": args.optimized_math,
        "requested_periods": args.periods,
        "completed_segments": formal_completed,
        "diagnostic_completed_segments": diagnostic_completed,
        "certified_horizon": formal_completed * 0.1,
        "diagnostic_horizon": diagnostic_completed * 0.1,
        "first_failure": first_failure,
        "diagnostic_failure": diagnostic_failure,
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
        "replay_points_sha256": sha256(replay_path),
        "replay_points": sorted(replay_points),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "completed_segments", "certified_horizon", "first_failure", "nominal_gate")}))
    return 0 if summary["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
