#!/usr/bin/env python3
"""Run the formal TORA-Q3 algorithm-aligned stage and T20 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from torch_tm_flowpipe.batched_dense_tm import (
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRemainderLedger,
    dense_transient_ledger_suppressed,
    dense_validation_batch,
)
from torch_tm_flowpipe.tora_algorithm_aligned import algorithm_aligned_q3_step
from torch_tm_flowpipe.tora_q3 import (
    ToraQ3AffineCarry,
    build_tora_q3_box_model,
    compose_tora_q3_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
    tora_q3_boundary_from_model,
)
from torch_tm_flowpipe.tora_stage_contract import (
    model_and_carry_from_xiangru_record,
)


ROOT = Path(__file__).resolve().parents[1]
GATES = (
    ("G0", 1, True),
    ("G1", 1, False),
    ("G2", 2, False),
    ("G3", 10, False),
    ("G4", 40, False),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def tensor(value: Any, *, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def interval_metrics(
    lower: torch.Tensor,
    upper: torch.Tensor,
    reference_lower: Any,
    reference_upper: Any,
) -> dict[str, Any]:
    ref_lo = tensor(reference_lower, like=lower)
    ref_hi = tensor(reference_upper, like=upper)
    if ref_lo.shape != lower.shape or ref_hi.shape != upper.shape:
        raise ValueError("interval comparison shape mismatch")
    width = upper - lower
    reference_width = ref_hi - ref_lo
    valid_ratio = reference_width > 0.0
    ratios = width[valid_ratio] / reference_width[valid_ratio]
    center_difference = 0.5 * ((lower + upper) - (ref_lo + ref_hi))
    contains = (lower <= ref_lo) & (upper >= ref_hi)
    contained = (ref_lo <= lower) & (ref_hi >= upper)
    return {
        "candidate_contains_reference_coordinates": int(contains.sum().item()),
        "center_difference_maximum_absolute": float(
            torch.abs(center_difference).max().item()
        ),
        "coordinate_count": lower.numel(),
        "lower_difference_maximum_absolute": float(
            torch.abs(lower - ref_lo).max().item()
        ),
        "reference_contains_candidate_coordinates": int(contained.sum().item()),
        "upper_difference_maximum_absolute": float(
            torch.abs(upper - ref_hi).max().item()
        ),
        "width_difference_maximum_absolute": float(
            torch.abs(width - reference_width).max().item()
        ),
        "width_ratio": {
            "count": int(ratios.numel()),
            "maximum": float(ratios.max().item()) if ratios.numel() else 1.0,
            "median": float(ratios.median().item()) if ratios.numel() else 1.0,
            "minimum": float(ratios.min().item()) if ratios.numel() else 1.0,
        },
    }


def coefficient_metrics(
    candidate: torch.Tensor, reference: Any
) -> dict[str, Any]:
    expected = tensor(reference, like=candidate)
    if expected.shape != candidate.shape:
        raise ValueError("coefficient comparison shape mismatch")
    absolute = torch.abs(candidate - expected)
    spacing = torch.abs(
        torch.nextafter(expected, torch.full_like(expected, torch.inf))
        - expected
    )
    nonzero = spacing > 0.0
    ulp = absolute[nonzero] / spacing[nonzero]
    return {
        "coordinate_map": "identity_complete_q3_stage_contract",
        "maximum_absolute_difference": float(absolute.max().item()),
        "maximum_ulp_difference": float(ulp.max().item()) if ulp.numel() else 0.0,
        "shape": list(candidate.shape),
    }


def slice_model(model: BatchedTaylorModel, count: int) -> BatchedTaylorModel:
    return BatchedTaylorModel(
        BatchedPolynomial(model.poly.coeffs[:count], model.poly.basis),
        model.rem_lo[:count],
        model.rem_hi[:count],
        model.domain_lo[:count],
        model.domain_hi[:count],
        DenseRemainderLedger.empty(),
        model.range_policy,
        model.range_trace,
    )


def slice_carry(carry: ToraQ3AffineCarry, count: int) -> ToraQ3AffineCarry:
    return ToraQ3AffineCarry(
        carry.linear[:count],
        carry.remainder_lower[:count],
        carry.remainder_upper[:count],
    )


def select_rows(path: Path, segments: set[int]) -> tuple[dict[str, Any], dict[int, Any]]:
    rows: dict[int, Any] = {}
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        for line in handle:
            record = json.loads(line)
            segment = int(record["segment_index"])
            if segment in segments:
                if "stage_contract" not in record:
                    raise ValueError(f"segment {segment} is missing stage contract")
                rows[segment] = record
            if len(rows) == len(segments):
                break
    if set(rows) != segments:
        raise ValueError("stage plant does not contain every requested gate")
    return header, rows


def sliced_payload(value: Any, count: int) -> Any:
    return value[:count] if isinstance(value, list) else value


def one_step_gate(
    name: str,
    record: Mapping[str, Any],
    *,
    one_leaf: bool,
    device: torch.device,
) -> dict[str, Any]:
    base, carry = model_and_carry_from_xiangru_record(record, device=device)
    count = 1 if one_leaf else base.poly.batch
    if one_leaf:
        base = slice_model(base, count)
        carry = slice_carry(carry, count)
    local_step = algorithm_aligned_q3_step(base, capture_trace=True)
    physical_step = compose_tora_q3_step(local_step, carry)
    xiangru_coefficients = sliced_payload(
        record["picard"]["final_polynomial"]["coefficients"], count
    )
    xiangru_polynomial = BatchedPolynomial(
        tensor(xiangru_coefficients, like=local_step.segment_tm.poly.coeffs),
        local_step.segment_tm.poly.basis,
    )
    xiangru_range = xiangru_polynomial.range_bound(
        base.domain_lo,
        base.domain_hi,
        policy=base.range_policy,
        context=f"algorithm_aligned_{name}_xiangru_polynomial_range",
    )
    candidate_range = local_step.segment_tm.poly.range_bound(
        base.domain_lo,
        base.domain_hi,
        policy=base.range_policy,
        context=f"algorithm_aligned_{name}_polynomial_range",
    )
    xiangru_accepted = torch.as_tensor(
        sliced_payload(record["accepted"], count),
        dtype=torch.bool,
        device=device,
    )
    stage = record["stage_contract"]
    aligned_round_margins = torch.as_tensor(
        [row["subset_margin"] for row in local_step.round_trace],
        dtype=torch.float64,
        device=device,
    )
    xiangru_round_margins = tensor(
        [
            sliced_payload(row["subset_margin"], count)
            for row in stage["A8_remainder_rounds"]
        ],
        like=aligned_round_margins,
    )
    initial_margin = tensor(
        sliced_payload(stage["A7_initial_remainder_image"]["subset_margin"], count),
        like=local_step.initial_margin,
    )
    property_margin = 2.0 - torch.maximum(
        torch.abs(physical_step.tube_lower[:, :4]),
        torch.abs(physical_step.tube_upper[:, :4]),
    )
    expected_property_margin = sliced_payload(
        record["property_margin"]["tube"], count
    )
    all_accepted_equal = torch.equal(
        physical_step.accepted_by_leaf, xiangru_accepted
    )
    return {
        "accepted": {
            "candidate_count": int(physical_step.accepted_by_leaf.sum().item()),
            "equal": all_accepted_equal,
            "reference_count": int(xiangru_accepted.sum().item()),
        },
        "batch": count,
        "coefficients": coefficient_metrics(
            local_step.segment_tm.poly.coeffs, xiangru_coefficients
        ),
        "endpoint": interval_metrics(
            physical_step.endpoint_lower,
            physical_step.endpoint_upper,
            sliced_payload(record["endpoint"]["lower"], count),
            sliced_payload(record["endpoint"]["upper"], count),
        ),
        "gate": name,
        "interval_remainder": interval_metrics(
            local_step.segment_tm.rem_lo,
            local_step.segment_tm.rem_hi,
            sliced_payload(record["picard"]["final_remainder"]["lower"], count),
            sliced_payload(record["picard"]["final_remainder"]["upper"], count),
        ),
        "polynomial_range": interval_metrics(
            candidate_range[0],
            candidate_range[1],
            xiangru_range[0],
            xiangru_range[1],
        ),
        "property_margin_maximum_absolute_difference": float(
            torch.abs(
                property_margin
                - tensor(expected_property_margin, like=property_margin)
            ).max().item()
        ),
        "segment_index": int(record["segment_index"]),
        "status": "PASS" if local_step.accepted and all_accepted_equal else "FAIL",
        "subset_margin": {
            "initial_maximum_absolute_difference": float(
                torch.abs(local_step.initial_margin - initial_margin).max().item()
            ),
            "rounds_maximum_absolute_difference": float(
                torch.abs(aligned_round_margins - xiangru_round_margins).max().item()
            ),
        },
        "tube": interval_metrics(
            physical_step.tube_lower,
            physical_step.tube_upper,
            sliced_payload(record["tube"]["lower"], count),
            sliced_payload(record["tube"]["upper"], count),
        ),
    }


class AggregateIntervals:
    def __init__(self) -> None:
        self.lower_errors: list[float] = []
        self.upper_errors: list[float] = []
        self.center_errors: list[float] = []
        self.width_differences: list[float] = []
        self.ratios: list[float] = []

    def add(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        reference_lower: Any,
        reference_upper: Any,
    ) -> None:
        ref_lo = tensor(reference_lower, like=lower)
        ref_hi = tensor(reference_upper, like=upper)
        width = upper - lower
        reference_width = ref_hi - ref_lo
        self.lower_errors.extend(torch.abs(lower - ref_lo).flatten().cpu().tolist())
        self.upper_errors.extend(torch.abs(upper - ref_hi).flatten().cpu().tolist())
        self.center_errors.extend(
            torch.abs(0.5 * ((lower + upper) - (ref_lo + ref_hi)))
            .flatten()
            .cpu()
            .tolist()
        )
        self.width_differences.extend(
            torch.abs(width - reference_width).flatten().cpu().tolist()
        )
        valid = reference_width > 0.0
        self.ratios.extend((width[valid] / reference_width[valid]).cpu().tolist())

    @staticmethod
    def _summary(values: Iterable[float]) -> dict[str, float | int]:
        ordered = sorted(values)
        if not ordered:
            return {"count": 0, "maximum": 0.0, "median": 0.0, "minimum": 0.0}
        return {
            "count": len(ordered),
            "maximum": ordered[-1],
            "median": ordered[len(ordered) // 2],
            "minimum": ordered[0],
        }

    def payload(self) -> dict[str, Any]:
        return {
            "center_difference_absolute": self._summary(self.center_errors),
            "lower_difference_absolute": self._summary(self.lower_errors),
            "upper_difference_absolute": self._summary(self.upper_errors),
            "width_difference_absolute": self._summary(self.width_differences),
            "width_ratio": self._summary(self.ratios),
        }


def common_control_t20(
    controller_path: Path,
    plant_path: Path,
    *,
    periods: int,
    device: torch.device,
) -> dict[str, Any]:
    trace = json.loads(controller_path.read_text(encoding="utf-8"))
    controller_rows = trace["rows"]
    if len(controller_rows) != 20 or periods != 20:
        raise ValueError("formal algorithm-aligned common-control gate requires T20")
    endpoint = AggregateIntervals()
    tube = AggregateIntervals()
    remainder = AggregateIntervals()
    first_failure: dict[str, Any] | None = None
    completed = 0
    accepted_equal = True
    minimum_property_margin = math.inf
    with plant_path.open(encoding="utf-8") as plant_handle:
        plant_header = json.loads(next(plant_handle))
        with torch.no_grad(), dense_validation_batch(), dense_transient_ledger_suppressed():
            for period, controller in enumerate(controller_rows, start=1):
                state_lo = torch.as_tensor(
                    controller["pre_controller_state_box"]["lower"],
                    dtype=torch.float64,
                    device=device,
                )
                state_hi = torch.as_tensor(
                    controller["pre_controller_state_box"]["upper"],
                    dtype=torch.float64,
                    device=device,
                )
                control = controller["u1_interval_installed_for_next_ten_segments"]
                control_lo = torch.as_tensor(
                    control["lower"], dtype=torch.float64, device=device
                ).reshape(-1)
                control_hi = torch.as_tensor(
                    control["upper"], dtype=torch.float64, device=device
                ).reshape(-1)
                model = build_tora_q3_box_model(
                    state_lo, state_hi, control_lo, control_hi, device=device
                )
                boundary = tora_q3_boundary_from_model(model)
                carry = identity_tora_q3_carry(48, device=device)
                for local_segment in range(1, 11):
                    segment = (period - 1) * 10 + local_segment
                    reference = json.loads(next(plant_handle))
                    if int(reference["segment_index"]) != segment:
                        raise ValueError("common-control plant segment order drift")
                    local_model, carry = normalize_tora_q3_boundary(boundary, carry)
                    local_step = algorithm_aligned_q3_step(
                        local_model, capture_trace=False
                    )
                    physical_step = compose_tora_q3_step(local_step, carry)
                    expected_accepted = torch.as_tensor(
                        reference["accepted"], dtype=torch.bool, device=device
                    )
                    accepted_equal &= torch.equal(
                        physical_step.accepted_by_leaf, expected_accepted
                    )
                    endpoint.add(
                        physical_step.endpoint_lower,
                        physical_step.endpoint_upper,
                        reference["endpoint"]["lower"],
                        reference["endpoint"]["upper"],
                    )
                    tube.add(
                        physical_step.tube_lower,
                        physical_step.tube_upper,
                        reference["tube"]["lower"],
                        reference["tube"]["upper"],
                    )
                    remainder.add(
                        physical_step.segment_tm.rem_lo,
                        physical_step.segment_tm.rem_hi,
                        reference["interval_remainder"]["lower"],
                        reference["interval_remainder"]["upper"],
                    )
                    property_margin = 2.0 - torch.maximum(
                        torch.abs(physical_step.tube_lower[:, :4]),
                        torch.abs(physical_step.tube_upper[:, :4]),
                    )
                    minimum_property_margin = min(
                        minimum_property_margin,
                        float(property_margin.min().item()),
                    )
                    if not physical_step.accepted:
                        first_failure = {
                            "failed_leaf_count": int(
                                (~physical_step.accepted_by_leaf).sum().item()
                            ),
                            "segment_index": segment,
                        }
                        break
                    completed = segment
                    boundary = project_tora_q3_endpoint_to_affine(
                        local_step.segment_tm
                    )
                if first_failure is not None:
                    break
    return {
        "accepted_status_equal_to_reference": accepted_equal,
        "completed_segments": completed,
        "endpoint": endpoint.payload(),
        "first_failure": first_failure,
        "formal_native_result": True,
        "minimum_property_margin": minimum_property_margin,
        "period_local_frozen_xiangru_input_restart": True,
        "plant_header_schema": plant_header.get("schema"),
        "reference_outputs_used_as_native_inputs": False,
        "remainder": remainder.payload(),
        "requested_segments": periods * 10,
        "schema": "torch_tora_q3_algorithm_aligned_common_control_v1",
        "status": "PASS" if completed == periods * 10 and accepted_equal else "FAIL",
        "tube": tube.payload(),
    }


def verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-plant", type=Path, required=True)
    parser.add_argument("--expected-stage-plant-sha256", required=True)
    parser.add_argument("--common-plant", type=Path, required=True)
    parser.add_argument("--expected-common-plant-sha256", required=True)
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--expected-controller-trace-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--periods", type=int, default=20)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "common_plant_sha256": verify_hash(
            args.common_plant,
            args.expected_common_plant_sha256,
            "common plant",
        ),
        "controller_trace_sha256": verify_hash(
            args.controller_trace,
            args.expected_controller_trace_sha256,
            "controller trace",
        ),
        "stage_plant_sha256": verify_hash(
            args.stage_plant,
            args.expected_stage_plant_sha256,
            "stage plant",
        ),
    }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    _header, selected = select_rows(
        args.stage_plant, {segment for _name, segment, _one in GATES}
    )
    gates = [
        one_step_gate(
            name,
            selected[segment],
            one_leaf=one_leaf,
            device=device,
        )
        for name, segment, one_leaf in GATES
    ]
    one_step = {
        "formal_native_result": True,
        "gates": gates,
        "reference_outputs_used_as_native_inputs": False,
        "schema": "torch_tora_q3_algorithm_aligned_one_step_gates_v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in gates) else "FAIL",
    }
    write_json(output / "one_step_gates.json", one_step)
    common = common_control_t20(
        args.controller_trace,
        args.common_plant,
        periods=args.periods,
        device=device,
    )
    write_json(output / "common_control_t20.json", common)
    source_hashes = {
        path: sha256(ROOT / path)
        for path in (
            "experiments/run_tora_q3_algorithm_aligned.py",
            "src/torch_tm_flowpipe/tora_algorithm_aligned.py",
            "src/torch_tm_flowpipe/batched_dense_tm.py",
            "src/torch_tm_flowpipe/tora_q3.py",
        )
    }
    summary = {
        "common_control": common,
        "device": str(device),
        "dtype": "float64",
        "inputs": inputs,
        "lane": "algorithm_aligned_q3",
        "one_step": one_step,
        "raw_paths_in_public_record": False,
        "schema": "torch_tora_q3_algorithm_aligned_summary_v1",
        "source_sha256": source_hashes,
        "status": (
            "PASS"
            if one_step["status"] == "PASS" and common["status"] == "PASS"
            else "FAIL"
        ),
    }
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "common_control_completed_segments": common["completed_segments"],
                "one_step_gates": [row["status"] for row in gates],
                "status": summary["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
