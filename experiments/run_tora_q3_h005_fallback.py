#!/usr/bin/env python3
"""Run the single evidence-selected h=0.05 native TORA-Q3 fallback.

Two validated algorithm-aligned h=0.05 flowpipes are composed into each
0.1-second reporting macro-step.  The controller is still refreshed once per
second, so the changed contract is isolated to the plant step schedule.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_tora_q3_full_closed_loop as frozen_scheduler
from experiments import run_tora_q3_native_hierarchical as hierarchical
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedPolynomial,
    BatchedTaylorModel,
)
from torch_tm_flowpipe.tora_algorithm_aligned import algorithm_aligned_q3_step
from torch_tm_flowpipe.tora_q3 import (
    ToraQ3Step,
    compose_tora_q3_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
)


FORMAL_LANE = "algorithm_aligned_h005_refresh1"
ORIGINAL_COMPOSE = compose_tora_q3_step


@dataclass(frozen=True)
class HalfStepPair:
    first: ToraQ3Step
    second: ToraQ3Step
    aggregate: ToraQ3Step

    def __getattr__(self, name: str) -> Any:
        return getattr(self.aggregate, name)


def with_time_horizon(model: BatchedTaylorModel, h: float) -> BatchedTaylorModel:
    domain_hi = model.domain_hi.clone()
    domain_hi[:, 0] = float(h)
    return BatchedTaylorModel(
        model.poly,
        model.rem_lo,
        model.rem_hi,
        model.domain_lo,
        domain_hi,
        model.ledger,
        model.range_policy,
        model.range_trace,
    )


def rescale_half_step_time(model: BatchedTaylorModel) -> BatchedTaylorModel:
    """Map physical tau in [0,.05] to reporting tau in [0,.1]."""
    degrees = model.poly.basis.exponents[:, 0].to(model.poly.coeffs.dtype)
    factors = torch.pow(
        torch.full_like(degrees, 0.5),
        degrees,
    )
    coefficients = model.poly.coeffs * factors.view(1, 1, -1)
    domain_hi = model.domain_hi.clone()
    domain_hi[:, 0] = 0.1
    return BatchedTaylorModel(
        BatchedPolynomial(coefficients, model.poly.basis),
        model.rem_lo,
        model.rem_hi,
        model.domain_lo,
        domain_hi,
        model.ledger,
        model.range_policy,
        model.range_trace,
    )


def replace_segment(step: ToraQ3Step, segment: BatchedTaylorModel) -> ToraQ3Step:
    return ToraQ3Step(
        segment_tm=segment,
        endpoint_tm=(segment.endpoint(0, 0.1) if step.accepted else None),
        tube_lower=step.tube_lower,
        tube_upper=step.tube_upper,
        endpoint_lower=step.endpoint_lower,
        endpoint_upper=step.endpoint_upper,
        finite_ok_by_leaf=step.finite_ok_by_leaf,
        initial_subset_ok_by_leaf=step.initial_subset_ok_by_leaf,
        all_remainder_rounds_ok_by_leaf=step.all_remainder_rounds_ok_by_leaf,
        local_property_ok_by_leaf=step.local_property_ok_by_leaf,
        composed_property_ok_by_leaf=step.composed_property_ok_by_leaf,
        accepted_by_leaf=step.accepted_by_leaf,
        initial_shrink_mask=step.initial_shrink_mask,
        initial_margin=step.initial_margin,
        round_trace=step.round_trace,
        polynomial_trace=step.polynomial_trace,
        status=step.status,
        message=step.message,
    )


def joined_step(first: ToraQ3Step, second: ToraQ3Step) -> ToraQ3Step:
    tube_lower = torch.minimum(first.tube_lower, second.tube_lower)
    tube_upper = torch.maximum(first.tube_upper, second.tube_upper)
    finite = first.finite_ok_by_leaf & second.finite_ok_by_leaf
    initial = first.initial_subset_ok_by_leaf & second.initial_subset_ok_by_leaf
    rounds = (
        first.all_remainder_rounds_ok_by_leaf
        & second.all_remainder_rounds_ok_by_leaf
    )
    local_property = (
        torch.maximum(torch.abs(tube_lower[:, :4]), torch.abs(tube_upper[:, :4]))
        <= 2.0
    ).all(dim=1)
    composed_property = (
        first.composed_property_ok_by_leaf & second.composed_property_ok_by_leaf
    )
    accepted = finite & initial & rounds & local_property & composed_property
    all_accepted = bool(torch.all(accepted))
    return ToraQ3Step(
        segment_tm=second.segment_tm,
        endpoint_tm=(second.endpoint_tm if all_accepted else None),
        tube_lower=tube_lower,
        tube_upper=tube_upper,
        endpoint_lower=second.endpoint_lower,
        endpoint_upper=second.endpoint_upper,
        finite_ok_by_leaf=finite,
        initial_subset_ok_by_leaf=initial,
        all_remainder_rounds_ok_by_leaf=rounds,
        local_property_ok_by_leaf=local_property,
        composed_property_ok_by_leaf=composed_property,
        accepted_by_leaf=accepted,
        initial_shrink_mask=first.initial_shrink_mask & second.initial_shrink_mask,
        initial_margin=torch.minimum(first.initial_margin, second.initial_margin),
        round_trace=first.round_trace + second.round_trace,
        polynomial_trace=first.polynomial_trace + second.polynomial_trace,
        status="validated" if all_accepted else "failed",
        message="" if all_accepted else "one or both validated h=0.05 substeps failed",
    )


def half_step_adapter(base: BatchedTaylorModel, **kwargs: Any) -> HalfStepPair:
    rounds = int(kwargs.pop("polynomial_picard_rounds", 2))
    backend = str(kwargs.pop("point_enclosure_backend", "eager"))
    capture = bool(kwargs.pop("capture_trace", False))
    if kwargs:
        raise TypeError(f"unsupported h005 scheduler arguments: {sorted(kwargs)}")
    if rounds != 2:
        raise ValueError("h005 fallback freezes polynomial Picard K2")
    first_base = with_time_horizon(base, 0.05)
    first_raw = algorithm_aligned_q3_step(
        first_base,
        h=0.05,
        capture_trace=capture,
        point_enclosure_backend=backend,
    )
    first_boundary = project_tora_q3_endpoint_to_affine(
        first_raw.segment_tm, h=0.05
    )
    internal_carry = identity_tora_q3_carry(
        first_base.poly.batch,
        device=first_base.poly.coeffs.device,
        dtype=first_base.poly.coeffs.dtype,
    )
    second_base, internal_carry = normalize_tora_q3_boundary(
        first_boundary,
        internal_carry,
        h=0.05,
        range_policy=base.range_policy,
    )
    second_raw = algorithm_aligned_q3_step(
        second_base,
        h=0.05,
        capture_trace=capture,
        point_enclosure_backend=backend,
    )
    second_in_first_coordinates = ORIGINAL_COMPOSE(
        second_raw, internal_carry, h=0.05
    )
    first = replace_segment(
        first_raw, rescale_half_step_time(first_raw.segment_tm)
    )
    second = replace_segment(
        second_in_first_coordinates,
        rescale_half_step_time(second_in_first_coordinates.segment_tm),
    )
    return HalfStepPair(first, second, joined_step(first, second))


def compose_half_step_pair(
    local_step: ToraQ3Step | HalfStepPair,
    carry: Any,
    *,
    h: float = 0.1,
    profile_stages: bool = False,
) -> ToraQ3Step:
    if not isinstance(local_step, HalfStepPair):
        return ORIGINAL_COMPOSE(
            local_step, carry, h=h, profile_stages=profile_stages
        )
    if h != 0.1:
        raise ValueError("h005 macro composition requires a 0.1 reporting step")
    first = ORIGINAL_COMPOSE(
        local_step.first,
        carry,
        h=0.1,
        profile_stages=profile_stages,
    )
    second = ORIGINAL_COMPOSE(
        local_step.second,
        carry,
        h=0.1,
        profile_stages=profile_stages,
    )
    return joined_step(first, second)


def invoke(args: argparse.Namespace) -> int:
    argv = [
        "run_tora_q3_full_closed_loop.py",
        "--output-dir",
        str(args.output_dir),
        "--controller-trace",
        str(args.controller_trace),
        "--expected-controller-trace-sha256",
        args.expected_controller_trace_sha256,
        "--device",
        args.device,
        "--periods",
        "20",
        "--run-id",
        args.run_id,
        "--lane",
        "algorithm_aligned_q3",
        "--point-enclosure-backend",
        "compiled",
        "--optimized-math",
        "--continue-after-property-failure",
    ]
    original_argv = sys.argv
    original_add = frozen_scheduler.argparse.ArgumentParser.add_argument
    original_step = frozen_scheduler.dense_tora_q3_dr_step
    original_compose = frozen_scheduler.compose_tora_q3_step

    def accepting_add(self: Any, *names: str, **kwargs: Any) -> Any:
        if "--lane" in names:
            kwargs["choices"] = tuple(kwargs["choices"]) + (
                "algorithm_aligned_q3",
            )
        return original_add(self, *names, **kwargs)

    try:
        sys.argv = argv
        frozen_scheduler.argparse.ArgumentParser.add_argument = accepting_add
        frozen_scheduler.dense_tora_q3_dr_step = half_step_adapter
        frozen_scheduler.compose_tora_q3_step = compose_half_step_pair
        return frozen_scheduler.main()
    finally:
        sys.argv = original_argv
        frozen_scheduler.argparse.ArgumentParser.add_argument = original_add
        frozen_scheduler.dense_tora_q3_dr_step = original_step
        frozen_scheduler.compose_tora_q3_step = original_compose


def rewrite_contract_before_augmentation(args: argparse.Namespace) -> None:
    path = args.output_dir.resolve() / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["scheduler_implementation_config"] = summary["config"]
    summary["scheduler_implementation_config_sha256"] = summary["config_sha256"]
    config = dict(summary["config"])
    config.update(
        {
            "lane": FORMAL_LANE,
            "step_size": 0.05,
            "plant_substeps_per_reporting_step": 2,
            "plant_substeps_per_controller_refresh": 20,
            "reporting_step_size": 0.1,
            "controller_refresh_period": 1.0,
        }
    )
    summary["config"] = config
    summary["config_sha256"] = hierarchical.canonical_sha256(config)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize_source_binding(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    summary_path = output / "summary.json"
    gate_path = output / "hierarchical_gates.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    relative = "experiments/run_tora_q3_h005_fallback.py"
    digest = hierarchical.sha256(Path(__file__).resolve())
    summary["source_sha256"][relative] = digest
    gates["source_sha256"][relative] = digest
    gates["config"] = summary["config"]
    gates["config_sha256"] = summary["config_sha256"]
    hierarchical.write_json(gate_path, gates)
    summary["hierarchical_gates_sha256"] = hierarchical.sha256(gate_path)
    hierarchical.write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--expected-controller-trace-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.formal_lane = FORMAL_LANE
    scheduler_code = invoke(args)
    rewrite_contract_before_augmentation(args)
    hierarchical.augment_run(args, scheduler_code)
    summary = finalize_source_binding(args)
    print(
        json.dumps(
            {
                "formal_lane": FORMAL_LANE,
                "hierarchical_status": summary["hierarchical_status"],
                "certified_horizon": summary["certified_horizon"],
                "first_failure": summary["first_failure"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
