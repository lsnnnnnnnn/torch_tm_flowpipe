#!/usr/bin/env python3
"""Replay selected Xiangru inputs through native Torch stage observations.

Xiangru tensors are read only as offline observations.  They are never used as
formal step outputs, and the resulting raw tensor tree must be written below a
private evidence root.  The public summary contains hashes and aggregates only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from torch_tm_flowpipe.tora_controller import (
    EXPECTED_ORIGINAL_CONTROLLER_SHA256,
    ToraAutoLirpaControllerBounder,
    _normalized_inputs,
)
from torch_tm_flowpipe.tora_q3 import ToraQ3AffineBoundary
from torch_tm_flowpipe.tora_stage_contract import (
    REPLAY_POINTS,
    SELECTED_SEGMENTS,
    model_and_carry_from_xiangru_record,
    observe_torch_integration_from_xiangru_payload,
    observe_torch_local_step,
    observe_torch_sine_from_xiangru_payload,
    tensor_tree_to_lists,
    validate_xiangru_stage_record,
)


EXPECTED_FUNCTION_HASHES = {
    "generic_fixed_basis.py::tm_sin": (
        "experiments/remainder_ablation/generic_fixed_basis.py",
        "tm_sin",
        "370a19169f350f194e5a3609575df8d19aec63dc76388ca1ffd98407766aae08",
    ),
    "generic_fixed_basis.py::tm_tora_rhs": (
        "experiments/remainder_ablation/generic_fixed_basis.py",
        "tm_tora_rhs",
        "630a2b04e6fcff6a290d73a5abbb554a4b521a25f1079ab6aa1c0d680521829d",
    ),
    "generic_fixed_basis.py::run_tora_remainder_picard": (
        "experiments/remainder_ablation/generic_fixed_basis.py",
        "run_tora_remainder_picard",
        "56a65d39a9c0e66bd64b6e0b650d1ef42de888dc6de2e1d3108fbc5ec1ce7669",
    ),
    "run_s0_tora_static_partition_sweep.py::_run_lane": (
        "experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py",
        "_run_lane",
        "a1facded8dd0ff46d713c89e0ba50dab25e8e9e7573caae3782c184a0d480747",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def function_hash(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        candidate
        for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
        and candidate.name == name
    )
    fragment = "\n".join(
        source.splitlines()[node.lineno - 1 : node.end_lineno]
    ) + "\n"
    return hashlib.sha256(fragment.encode("utf-8")).hexdigest()


def verified_function_hashes(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for label, (relative, function, expected) in EXPECTED_FUNCTION_HASHES.items():
        value = function_hash(root / relative, function)
        if value != expected:
            raise ValueError(f"external function hash drift: {label}")
        observed[label] = value
    return observed


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def interval_aggregate(value: Mapping[str, Any]) -> dict[str, float]:
    lower = torch.as_tensor(value["lower"], dtype=torch.float64)
    upper = torch.as_tensor(value["upper"], dtype=torch.float64)
    center = lower + 0.5 * (upper - lower)
    radius = torch.maximum(center - lower, upper - center)
    return {
        "minimum_lower": float(lower.min()),
        "maximum_upper": float(upper.max()),
        "maximum_absolute_center": float(torch.abs(center).max()),
        "maximum_radius": float(radius.max()),
        "maximum_width": float((upper - lower).max()),
    }


def predicate_counts(stages: Mapping[str, Any]) -> dict[str, int]:
    return {
        name: int(torch.count_nonzero(value).detach().cpu())
        for name, value in stages["predicates"].items()
    }


def coordinate_aggregate(
    record: Mapping[str, Any], basis_exponents: list[list[int]]
) -> dict[str, Any]:
    coefficients = torch.as_tensor(
        record["stage_contract"]["A0_normalized_input"]["polynomial"][
            "coefficients"
        ],
        dtype=torch.float64,
    )
    constant_slot = basis_exponents.index([0, 0, 0, 0, 0, 0])
    linear_slots = []
    for variable in range(1, 6):
        exponent = [0] * 6
        exponent[variable] = 1
        linear_slots.append(basis_exponents.index(exponent))
    center = coefficients[:, :, constant_slot]
    affine = coefficients[:, :, linear_slots]
    diagonal = torch.diagonal(affine, dim1=1, dim2=2)
    off_diagonal = affine.clone()
    indices = torch.arange(5)
    off_diagonal[:, indices, indices] = 0.0
    non_affine = coefficients.clone()
    non_affine[:, :, constant_slot] = 0.0
    non_affine[:, :, linear_slots] = 0.0
    return {
        "affine_center_minimum": float(center.min()),
        "affine_center_maximum": float(center.max()),
        "affine_center_maximum_absolute": float(torch.abs(center).max()),
        "diagonal_scale_minimum": float(diagonal.min()),
        "diagonal_scale_maximum": float(diagonal.max()),
        "off_diagonal_maximum_absolute": float(torch.abs(off_diagonal).max()),
        "non_affine_input_maximum_absolute": float(torch.abs(non_affine).max()),
        "basis_exponents_sha256": canonical_sha256(basis_exponents),
    }


def controller_boundary(
    row: Mapping[str, Any], *, device: torch.device
) -> ToraQ3AffineBoundary:
    polynomial = row["boundary"]["polynomial"]
    full_linear = torch.as_tensor(
        polynomial["L"], dtype=torch.float64, device=device
    )
    full_time_linear = torch.as_tensor(
        polynomial["Lt"], dtype=torch.float64, device=device
    )
    if full_linear.shape != (48, 5, 6):
        raise ValueError("controller boundary L shape is not [48,5,6]")
    if torch.count_nonzero(full_linear[:, :, 0]).item() != 0:
        raise ValueError("controller boundary contains a live local-time slot")
    if torch.count_nonzero(full_time_linear).item() != 0:
        raise ValueError("controller boundary Lt is not zero at a refresh")
    return ToraQ3AffineBoundary(
        torch.as_tensor(polynomial["c"], dtype=torch.float64, device=device),
        full_linear[:, :, 1:],
        torch.as_tensor(
            row["boundary"]["remainder"]["lower"],
            dtype=torch.float64,
            device=device,
        ),
        torch.as_tensor(
            row["boundary"]["remainder"]["upper"],
            dtype=torch.float64,
            device=device,
        ),
    )


def maximum_absolute_error(actual: Any, expected: Any) -> float:
    left = torch.as_tensor(actual, dtype=torch.float64)
    right = torch.as_tensor(expected, dtype=torch.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return float(torch.max(torch.abs(left - right)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xiangru-root", type=Path, required=True)
    parser.add_argument("--xiangru-plant", type=Path, required=True)
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--controller-model", type=Path, required=True)
    parser.add_argument("--expected-xiangru-plant-sha256", required=True)
    parser.add_argument("--expected-controller-trace-sha256", required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.private_output.exists():
        raise FileExistsError(args.private_output)
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    plant_sha256 = sha256(args.xiangru_plant)
    controller_sha256 = sha256(args.controller_trace)
    controller_model_sha256 = sha256(args.controller_model)
    if plant_sha256 != args.expected_xiangru_plant_sha256:
        raise ValueError("Xiangru plant observation hash mismatch")
    if controller_sha256 != args.expected_controller_trace_sha256:
        raise ValueError("controller observation hash mismatch")
    if controller_model_sha256 != EXPECTED_ORIGINAL_CONTROLLER_SHA256:
        raise ValueError("controller model hash mismatch")
    function_hashes = verified_function_hashes(args.xiangru_root.resolve())

    stream = jsonl(args.xiangru_plant)
    header = next(stream)
    if header.get("schema") != "xiangru_tora_q3_plant_trace_header_v1":
        raise ValueError("unexpected Xiangru plant header schema")
    if header.get("basis_slot_count") != 84:
        raise ValueError("Xiangru observation is not complete Q3")
    selected: dict[int, dict[str, Any]] = {}
    observed_segment_count = 0
    for record in stream:
        observed_segment_count += 1
        segment = int(record["segment_index"])
        if segment in SELECTED_SEGMENTS:
            validate_xiangru_stage_record(record)
            selected[segment] = record
    if observed_segment_count != 200:
        raise ValueError("Xiangru observation is not a complete T20 run")
    if tuple(sorted(selected)) != SELECTED_SEGMENTS:
        raise ValueError("one or more selected replay records are missing")

    controller = json.loads(args.controller_trace.read_text(encoding="utf-8"))
    controller_rows = controller.get("rows")
    if not isinstance(controller_rows, list) or len(controller_rows) != 20:
        raise ValueError("controller observation is not a 20-period trace")
    if any(row.get("leaf_id") != list(range(48)) for row in controller_rows):
        raise ValueError("controller observation does not use canonical B48")

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    public_points: dict[str, Any] = {}
    maximum_torch_replay_error = 0.0
    maximum_xiangru_replay_error = 0.0
    all_xiangru_individual_replays_bitwise = True
    with args.private_output.open("x", encoding="utf-8") as handle:
        private_header = {
            "schema": "torch_tora_q3_stage_contract_trace_header_v1",
            "source_xiangru_plant_sha256": plant_sha256,
            "source_controller_trace_sha256": controller_sha256,
            "basis_exponents": header["basis_exponents"],
            "basis_slot_count": 84,
            "selected_segments": list(SELECTED_SEGMENTS),
            "stage_ids": [f"A{index}" for index in range(13)],
            "raw_private_only": True,
        }
        handle.write(json.dumps(private_header, separators=(",", ":")) + "\n")
        for segment in SELECTED_SEGMENTS:
            record = selected[segment]
            model, carry = model_and_carry_from_xiangru_record(
                record, device=device
            )
            observation = observe_torch_local_step(
                model,
                carry,
                segment_index=segment,
                point_enclosure_backend="eager",
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            torch_error = float(
                observation.replay_equivalence["maximum_absolute_error"]
            )
            xiangru_equivalence = record["stage_contract"]["replay_equivalence"]
            xiangru_error = float(xiangru_equivalence["maximum_absolute_error"])
            maximum_torch_replay_error = max(
                maximum_torch_replay_error, torch_error
            )
            maximum_xiangru_replay_error = max(
                maximum_xiangru_replay_error, xiangru_error
            )
            xiangru_stage = record["stage_contract"]
            diagnostic_sources = [
                ("initial", xiangru_stage["A7_initial_remainder_image"]),
                *(
                    (f"round_{row['round']}", row)
                    for row in xiangru_stage["A8_remainder_rounds"]
                ),
            ]
            same_input_diagnostics = []
            for label, source in diagnostic_sources:
                sine = source["rhs"]["sine"]
                integration = source["integration"]
                all_xiangru_individual_replays_bitwise = (
                    all_xiangru_individual_replays_bitwise
                    and bool(sine["replay_equivalence"]["bitwise"])
                    and bool(integration["replay_equivalence"]["bitwise"])
                )
                same_input_diagnostics.append(
                    {
                        "label": label,
                        "sine": observe_torch_sine_from_xiangru_payload(
                            sine["input"],
                            device=device,
                            order=2,
                            point_enclosure_backend="eager",
                        ),
                        "integration": observe_torch_integration_from_xiangru_payload(
                            integration["input"], device=device
                        ),
                    }
                )
            private_row = {
                "schema": "torch_tora_q3_stage_contract_segment_v1",
                "segment_index": segment,
                "physical_time": record["physical_time"],
                "controller_period": record["controller_period"],
                "leaf_id": list(range(48)),
                "source_xiangru_content_sha256": record["content_sha256"],
                "coordinate_contract": {
                    "physical_state_order": record["state_order"],
                    "local_time_variable": 0,
                    "held_control_variable": 5,
                    "basis_exponents": header["basis_exponents"],
                    "basis_slot_permutation": list(range(84)),
                    "input_tensor_source": "exact Xiangru A0 normalized input observation",
                    "affine_carry_source": "exact Xiangru A10 normalized-map observation",
                    "coefficient_coordinate_map": "IDENTICAL_NORMALIZED_INPUT_TENSORS",
                },
                "stages": tensor_tree_to_lists(observation.stages),
                "diagnostic_counterfactual_same_input": tensor_tree_to_lists(
                    same_input_diagnostics
                ),
                "torch_replay_equivalence": observation.replay_equivalence,
                "xiangru_replay_equivalence": xiangru_equivalence,
            }
            private_row["content_sha256"] = canonical_sha256(private_row)
            handle.write(json.dumps(private_row, separators=(",", ":")) + "\n")
            labels = [
                name
                for name, segments in REPLAY_POINTS.items()
                if segment in segments
            ]
            public_points[str(segment)] = {
                "replay_labels": labels,
                "segment_index": segment,
                "physical_time": record["physical_time"],
                "controller_period": record["controller_period"],
                "leaf_subsets": ["leaf_0", "B48"] if segment == 1 else ["B48"],
                "coordinate_map_status": "IDENTICAL_NORMALIZED_INPUT_TENSORS",
                "basis_permutation_bijective": True,
                "coordinate_aggregate": coordinate_aggregate(
                    record, header["basis_exponents"]
                ),
                "torch_reference_status": observation.local_step.status,
                "torch_physical_status": observation.physical_step.status,
                "torch_replay_equivalence": observation.replay_equivalence,
                "xiangru_replay_equivalence": xiangru_equivalence,
                "predicate_true_counts": predicate_counts(observation.stages),
                "private_row_sha256": private_row["content_sha256"],
            }

    selected_controllers = {
        "R1": controller_rows[1],
        "R2": controller_rows[4],
    }
    controller_bounder = ToraAutoLirpaControllerBounder(
        args.controller_model,
        controller_boundary(selected_controllers["R1"], device=device),
        device=device,
        expected_sha256=EXPECTED_ORIGINAL_CONTROLLER_SHA256,
    )
    torch_controller_rows: dict[str, Any] = {}
    controller_errors: dict[str, Any] = {}
    maximum_controller_error = 0.0
    for label, row in selected_controllers.items():
        boundary = controller_boundary(row, device=device)
        native_input_lo, native_input_hi, _weight, _center = _normalized_inputs(
            boundary
        )
        # Xiangru records the four physical controller inputs, whereas the
        # native graph retains five normalized generators plus four explicit
        # remainder variables.  Equality of the reconstructed physical box is
        # the coordinate-contract check at A11.
        positive = torch.clamp_min(boundary.linear[:, :4, :], 0.0)
        negative = torch.clamp_max(boundary.linear[:, :4, :], 0.0)
        generator_lo = -torch.ones(
            (48, 5), dtype=torch.float64, device=device
        )
        generator_hi = torch.ones_like(generator_lo)
        reconstructed_lo = boundary.center[:, :4]
        reconstructed_lo = reconstructed_lo + torch.matmul(
            positive, generator_lo.unsqueeze(-1)
        ).squeeze(-1)
        reconstructed_lo = reconstructed_lo + torch.matmul(
            negative, generator_hi.unsqueeze(-1)
        ).squeeze(-1)
        reconstructed_lo = reconstructed_lo + boundary.remainder_lower[:, :4]
        reconstructed_hi = boundary.center[:, :4]
        reconstructed_hi = reconstructed_hi + torch.matmul(
            positive, generator_hi.unsqueeze(-1)
        ).squeeze(-1)
        reconstructed_hi = reconstructed_hi + torch.matmul(
            negative, generator_lo.unsqueeze(-1)
        ).squeeze(-1)
        reconstructed_hi = reconstructed_hi + boundary.remainder_upper[:, :4]
        result = controller_bounder.bound(boundary)
        errors = {
            "a11_physical_input_lower": maximum_absolute_error(
                reconstructed_lo.cpu(),
                row["controller_input_box_after_normalization"]["lower"],
            ),
            "a11_physical_input_upper": maximum_absolute_error(
                reconstructed_hi.cpu(),
                row["controller_input_box_after_normalization"]["upper"],
            ),
            "a12_before_lower": maximum_absolute_error(
                result.output_lower_before_outward,
                row["controller_output_interval_before_outward_composition"][
                    "lower"
                ],
            ),
            "a12_before_upper": maximum_absolute_error(
                result.output_upper_before_outward,
                row["controller_output_interval_before_outward_composition"][
                    "upper"
                ],
            ),
            "a12_after_lower": maximum_absolute_error(
                result.output_lower_after_outward,
                row["controller_output_interval_after_outward_composition"][
                    "lower"
                ],
            ),
            "a12_after_upper": maximum_absolute_error(
                result.output_upper_after_outward,
                row["controller_output_interval_after_outward_composition"][
                    "upper"
                ],
            ),
        }
        maximum_controller_error = max(maximum_controller_error, *errors.values())
        controller_errors[label] = errors
        torch_controller_rows[label] = {
            "schema": "torch_tora_q3_stage_controller_observation_v1",
            "controller_period": int(row["controller_period"]),
            "native_normalized_input": {
                "lower": native_input_lo,
                "upper": native_input_hi,
            },
            "reconstructed_physical_input": {
                "lower": reconstructed_lo,
                "upper": reconstructed_hi,
            },
            "output_before_outward": {
                "lower": result.output_lower_before_outward,
                "upper": result.output_upper_before_outward,
            },
            "output_after_outward": {
                "lower": result.output_lower_after_outward,
                "upper": result.output_upper_after_outward,
            },
            "maximum_slope_gap": result.maximum_slope_gap,
            "comparison_to_xiangru": errors,
        }
    with args.private_output.open("a", encoding="utf-8") as handle:
        private_controller = {
            "schema": "torch_tora_q3_stage_controller_observations_v1",
            "rows": tensor_tree_to_lists(torch_controller_rows),
        }
        private_controller["content_sha256"] = canonical_sha256(
            private_controller
        )
        handle.write(json.dumps(private_controller, separators=(",", ":")) + "\n")
    controller_contract = {
        label: {
            "controller_period": int(row["controller_period"]),
            "input": interval_aggregate(
                row["controller_input_box_after_normalization"]
            ),
            "output_before_outward": interval_aggregate(
                row["controller_output_interval_before_outward_composition"]
            ),
            "output_after_outward": interval_aggregate(
                row["controller_output_interval_after_outward_composition"]
            ),
            "selected_raw_payload_sha256": canonical_sha256(row),
            "torch_same_input_comparison": controller_errors[label],
        }
        for label, row in selected_controllers.items()
    }
    private_output_sha256 = sha256(args.private_output)
    public = {
        "schema": "tora_q3_stage_observation_contract_summary_v1",
        "status": (
            "PASS"
            if maximum_torch_replay_error <= 5e-15
            and maximum_xiangru_replay_error <= 5e-15
            and maximum_controller_error <= 5e-12
            and all_xiangru_individual_replays_bitwise
            else "FAIL"
        ),
        "observation_only": True,
        "formal_runner_uses_xiangru_outputs": False,
        "selected_replay_points": public_points,
        "selected_controller_contract": controller_contract,
        "coordinate_contract": {
            "physical_state_order": ["x1", "x2", "x3", "x4", "u1"],
            "normalized_generator_order": [
                "x1_parameter",
                "x2_parameter",
                "x3_parameter",
                "x4_parameter",
                "u1_parameter",
            ],
            "local_time_variable": "slot 0, domain [0,0.1]",
            "held_control_variable": "slot 5, derivative exactly zero",
            "basis_slot_permutation": list(range(84)),
            "basis_permutation_bijective": True,
            "remainder_semantics": "additive componentwise interval outside retained Q3 polynomial",
        },
        "stage_coverage": {
            "A0": "basis/domain/normalization",
            "A1": "base polynomial and remainder",
            "A2": "sine point enclosure",
            "A3": "sine retained polynomial/composition overflow/analytic remainder",
            "A4": "K1 polynomial Picard",
            "A5": "K2 polynomial Picard",
            "A6": "polynomial difference and natural range",
            "A7": "initial remainder image and subset margin",
            "A8": "ten remainder images/candidates/accepted boxes/margins",
            "A9": "local and physical endpoint/tube",
            "A10": "affine carry/normalization",
            "A11": "controller input affine box at R1/R2",
            "A12": "controller output before/after outward inflation at R1/R2",
        },
        "instrumentation_replay": {
            "maximum_torch_reference_error": maximum_torch_replay_error,
            "maximum_xiangru_primary_vs_stage_replay_error": maximum_xiangru_replay_error,
            "all_observed_xiangru_sine_and_integration_replays_bitwise": (
                all_xiangru_individual_replays_bitwise
            ),
            "maximum_same_input_controller_error": maximum_controller_error,
        },
        "source_hashes": {
            "xiangru_plant_observation": plant_sha256,
            "controller_observation": controller_sha256,
            "controller_model": controller_model_sha256,
            "private_torch_stage_trace": private_output_sha256,
            "external_functions": function_hashes,
        },
        "raw_arrays_private": True,
        "raw_paths_in_public_record": False,
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": public["status"],
                "selected_segments": list(SELECTED_SEGMENTS),
                "maximum_torch_replay_error": maximum_torch_replay_error,
                "maximum_xiangru_replay_error": maximum_xiangru_replay_error,
                "private_output_sha256": private_output_sha256,
            },
            sort_keys=True,
        )
    )
    return 0 if public["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
