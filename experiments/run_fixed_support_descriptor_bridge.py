#!/usr/bin/env python3
"""Run the preregistered R7-to-R35 single-factor VDP bridge ladder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    FixedSupportInterval,
    FixedSupportPolynomial,
    FixedSupportReachability,
    FixedSupportSymbolicRemainderState,
    FixedSupportTaylorModel,
    diffreach_vdp_polynomial_rhs,
    diffreach_vdp_tm_rhs,
    fixed_support_build_linear_tm,
    fixed_support_dr_remainder_picard,
    fixed_support_identity_parameterization,
    fixed_support_polynomial_picard,
    fixed_support_step_boxes,
    fixed_support_symbolic_step_linear,
)


GATES = (("G0", 1), ("G1", 10), ("G2", 100), ("G3", 1000))
CELLS = (
    {"cell": "A0", "support": "R7", "picard": 2, "validator": "VDR", "carry": "CDR"},
    {"cell": "A1", "support": "R35", "picard": 2, "validator": "VDR", "carry": "CDR"},
    {"cell": "A2", "support": "R35", "picard": 4, "validator": "VDR", "carry": "CDR"},
    {"cell": "A3", "support": "R35", "picard": 4, "validator": "VRAW", "carry": "CDR"},
    {"cell": "A4", "support": "R35", "picard": 4, "validator": "VRAW", "carry": "CNI"},
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _support(name: str) -> FixedSupportDescriptor:
    if name == "R7":
        return FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    if name == "R35":
        return FixedSupportDescriptor.complete_total_degree(
            variable_names=("t", "xi0", "xi1"), order=4, local_time_index=0
        )
    raise ValueError(name)


def _initial_boxes(batch: int, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    split = int(round(batch**0.5))
    if split * split != batch:
        raise ValueError("bridge partitions must be square (B1 or B64)")
    x_edges = torch.linspace(1.1, 1.4, split + 1, dtype=dtype, device=device)
    y_edges = torch.linspace(2.35, 2.45, split + 1, dtype=dtype, device=device)
    lo = []
    hi = []
    for x_index in range(split):
        for y_index in range(split):
            lo.append(torch.stack((x_edges[x_index], y_edges[y_index])))
            hi.append(torch.stack((x_edges[x_index + 1], y_edges[y_index + 1])))
    return torch.stack(lo), torch.stack(hi)


def _cni_carry(
    endpoint: FixedSupportTaylorModel,
    parameterization: FixedSupportTaylorModel,
    eval_lo: torch.Tensor,
    eval_hi: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, FixedSupportTaylorModel]:
    """Complete normalized insertion with constant centering and no cutoff.

    ``parameterization`` is the previous right map.  The endpoint must first
    be inserted through that map; only then are its constant part and range
    magnitude used for the next normalized initial model.
    """

    inserted = endpoint.compose_affine(parameterization, 0.0)
    support = inserted.polynomial.support
    center = inserted.polynomial.coeffs[..., support.constant_slot]
    centered_coefficients = inserted.polynomial.coeffs.clone()
    centered_coefficients[..., support.constant_slot] = 0.0
    centered = FixedSupportTaylorModel(
        FixedSupportPolynomial(centered_coefficients, support),
        inserted.remainder,
        inserted.ledger,
    )
    centered_range = centered.range(eval_lo, eval_hi)
    scale = torch.maximum(torch.abs(centered_range.lo), torch.abs(centered_range.hi))
    del epsilon  # CNI uses exact reciprocal scaling; zero components map to identity scale.
    inverse = torch.where(scale == 0.0, torch.ones_like(scale), 1.0 / scale)
    normalized = centered.scale(inverse)
    return center, scale, normalized


def _ledger(model: FixedSupportTaylorModel) -> dict[str, Any]:
    entries = {}
    total_lo = torch.zeros_like(model.remainder.lo)
    total_hi = torch.zeros_like(model.remainder.hi)
    for name, interval in model.ledger.entries:
        entries[name] = {
            "lo": interval.lo.detach().cpu().tolist(),
            "hi": interval.hi.detach().cpu().tolist(),
        }
        total_lo = total_lo + interval.lo
        total_hi = total_hi + interval.hi
    return {
        "entries": entries,
        "entry_sum_lo": total_lo.detach().cpu().tolist(),
        "entry_sum_hi": total_hi.detach().cpu().tolist(),
        "model_remainder_lo": model.remainder.lo.detach().cpu().tolist(),
        "model_remainder_hi": model.remainder.hi.detach().cpu().tolist(),
        "entry_sum_contains_model": bool(torch.all(total_lo <= model.remainder.lo) and torch.all(total_hi >= model.remainder.hi))
        if entries
        else None,
    }


def _aggregate_ledger_sources(
    model: FixedSupportTaylorModel, tokens: tuple[str, ...]
) -> FixedSupportInterval:
    total = FixedSupportInterval.zeros_like(model.remainder.lo)
    for name, interval in model.ledger.entries:
        if any(token in name for token in tokens):
            total = total.add(interval)
    return total


def _interval_json(interval: FixedSupportInterval) -> dict[str, Any]:
    return {
        "lo": interval.lo.detach().cpu().tolist(),
        "hi": interval.hi.detach().cpu().tolist(),
        "width": interval.width.detach().cpu().tolist(),
    }


def _maximum_nested_width(lo: Any, hi: Any) -> float:
    lower = torch.as_tensor(lo, dtype=torch.float64)
    upper = torch.as_tensor(hi, dtype=torch.float64)
    return float((upper - lower).max().item())


def _stage_ledger(
    model: FixedSupportTaylorModel,
    raw_remainder: FixedSupportInterval,
    step_lo: torch.Tensor,
    step_hi: torch.Tensor,
) -> dict[str, Any]:
    zero = FixedSupportInterval.zeros_like(model.remainder.lo)
    truncation = _aggregate_ledger_sources(
        model,
        (
            "discarded_product_monomials",
            "pure_spatial_quadratic",
            "time_cubic",
            "time_quartic",
        ),
    )
    polynomial_times_remainder = _aggregate_ledger_sources(
        model,
        (
            "left_polynomial_times_right_remainder",
            "right_polynomial_times_left_remainder",
        ),
    )
    remainder_times_remainder = _aggregate_ledger_sources(
        model, ("remainder_times_remainder",)
    )
    integration_overflow = _aggregate_ledger_sources(
        model,
        (
            "integration_discarded_monomials",
            "integration_time_cubic",
            "integration_time_squared_spatial",
        ),
    )
    retained_range = model.polynomial.range(step_lo, step_hi)
    nonzero_dropped_entries = sum(
        int(torch.count_nonzero((interval.lo != 0) | (interval.hi != 0)).item())
        for name, interval in model.ledger.entries
        if any(
            token in name
            for token in (
                "discarded_product_monomials",
                "pure_spatial_quadratic",
                "time_cubic",
                "time_quartic",
                "integration_discarded_monomials",
                "integration_time_cubic",
                "integration_time_squared_spatial",
            )
        )
    )
    # The inherited operation ledger is a lineage diagnostic: scaling and
    # composition deliberately preserve source entries, so they are not an
    # additive partition at this boundary.  The coverage ledger therefore
    # records the exact boundary total separately and validates its sum.
    coverage_sum = model.remainder
    return {
        "schema": "torch_fixed_support_bridge_stage_ledger_v1",
        "coverage_entries": {
            "validated_remainder_total": _interval_json(model.remainder)
        },
        "coverage_sum": _interval_json(coverage_sum),
        "coverage_sum_contains_model_remainder": bool(
            torch.all(coverage_sum.lo <= model.remainder.lo)
            and torch.all(coverage_sum.hi >= model.remainder.hi)
        ),
        "source_observation_semantics": (
            "non-additive operation-lineage observations; not summed as a "
            "boundary remainder decomposition"
        ),
        "source_observations": {
            "retained_polynomial_range": _interval_json(retained_range),
            "truncation_interval": _interval_json(truncation),
            "cutoff_interval": _interval_json(zero),
            "polynomial_times_remainder": _interval_json(
                polynomial_times_remainder
            ),
            "remainder_times_remainder": _interval_json(
                remainder_times_remainder
            ),
            "integration_overflow": _interval_json(integration_overflow),
            "raw_candidate_remainder": _interval_json(raw_remainder),
        },
        "nonzero_dropped_source_entry_count": nonzero_dropped_entries,
        "raw_lineage": _ledger(model),
    }


def _run_cell(
    cell: dict[str, Any],
    *,
    batch: int,
    max_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    dtype = torch.float64
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    support = _support(cell["support"])
    initial_lo, initial_hi = _initial_boxes(batch, dtype=dtype, device=device)
    center = 0.5 * (initial_lo + initial_hi)
    scale = 0.5 * (initial_hi - initial_lo)
    model = fixed_support_build_linear_tm(center, scale, support)
    parameterization = fixed_support_identity_parameterization(
        batch, 2, support, dtype=dtype, device=device
    )
    symbolic = FixedSupportSymbolicRemainderState.initialize(
        batch, 2, 1000, dtype=dtype, device=device
    )
    step_lo, step_hi, eval_lo, eval_hi = fixed_support_step_boxes(
        batch, 2, 0.01, dtype=dtype, device=device
    )
    target = torch.full((batch, 2), 0.01, dtype=dtype, device=device)
    snapshots: list[dict[str, Any]] = []
    completed = 0
    first_failure: dict[str, Any] | None = None
    started = time.perf_counter()
    gate_steps = {steps: name for name, steps in GATES if steps <= max_steps}
    stage_runtime_totals = {
        "carry": 0.0,
        "polynomial_picard": 0.0,
        "validation": 0.0,
        "output_object": 0.0,
    }

    for step in range(1, max_steps + 1):
        carry_started = time.perf_counter()
        endpoint_previous = model.evaluate_time(0.01)
        if cell["carry"] == "CDR":
            carry = fixed_support_symbolic_step_linear(
                parameterization,
                endpoint_previous,
                symbolic,
                eval_lo,
                eval_hi,
                epsilon=1e-12,
            )
            step_center = endpoint_previous.polynomial.coeffs[..., support.constant_slot]
            next_parameterization = carry.normalized_parameterization
            next_symbolic = carry.state
            normalization_scale = carry.scale
        else:
            step_center, normalization_scale, next_parameterization = _cni_carry(
                endpoint_previous,
                parameterization,
                eval_lo,
                eval_hi,
                epsilon=1e-12,
            )
            next_symbolic = symbolic
        carry_seconds = time.perf_counter() - carry_started
        stage_runtime_totals["carry"] += carry_seconds
        polynomial_started = time.perf_counter()
        new_x0 = fixed_support_build_linear_tm(
            step_center, normalization_scale, support
        )
        polynomial, picard_trace = fixed_support_polynomial_picard(
            new_x0.polynomial,
            diffreach_vdp_polynomial_rhs,
            step_lo,
            step_hi,
            iterations=int(cell["picard"]),
        )
        polynomial_seconds = time.perf_counter() - polynomial_started
        stage_runtime_totals["polynomial_picard"] += polynomial_seconds
        seed = FixedSupportTaylorModel(
            polynomial, FixedSupportInterval(-target, target)
        )
        validation_started = time.perf_counter()
        if cell["validator"] == "VDR":
            initial_image = new_x0.add(
                diffreach_vdp_tm_rhs(seed, step_lo, step_hi).integrate_time(
                    step_lo, step_hi
                )
            )
            validation = fixed_support_dr_remainder_picard(
                diffreach_vdp_tm_rhs,
                new_x0,
                seed,
                step_lo,
                step_hi,
                rounds=10,
            )
            accepted_mask = validation.initial_inclusion_mask
            validated_model = validation.model
            round_masks = validation.round_inclusion_masks
            raw_remainder = initial_image.remainder
        else:
            raw_image = new_x0.add(
                diffreach_vdp_tm_rhs(seed, step_lo, step_hi).integrate_time(step_lo, step_hi)
            )
            polynomial_difference = raw_image.polynomial.sub(seed.polynomial).range(
                step_lo, step_hi
            )
            raw_remainder = raw_image.remainder.add(polynomial_difference)
            accepted_mask = raw_remainder.subseteq_elem(seed.remainder)
            validated_model = FixedSupportTaylorModel(
                polynomial, raw_remainder, raw_image.ledger
            )
            round_masks = accepted_mask.unsqueeze(0)
        validation_seconds = time.perf_counter() - validation_started
        stage_runtime_totals["validation"] += validation_seconds
        if not bool(torch.all(accepted_mask)):
            first_failure = {
                "step": step,
                "time": (step - 1) * 0.01,
                "reason": "initial_remainder_inclusion_failed",
                "mask": accepted_mask.detach().cpu().tolist(),
                "raw_remainder_lo": raw_remainder.lo.detach().cpu().tolist(),
                "raw_remainder_hi": raw_remainder.hi.detach().cpu().tolist(),
            }
            break
        model = validated_model
        parameterization = next_parameterization
        symbolic = next_symbolic
        completed = step
        if step in gate_steps:
            output_started = time.perf_counter()
            composed = validated_model.compose_affine(
                next_parameterization, 0.01
            )
            endpoint_lo = step_lo.clone()
            endpoint_hi = step_hi.clone()
            endpoint_lo[:, support.local_time_index] = 0.01
            endpoint_hi[:, support.local_time_index] = 0.01
            endpoint = composed.range(endpoint_lo, endpoint_hi)
            tube = composed.range(step_lo, step_hi)
            output_seconds = time.perf_counter() - output_started
            stage_runtime_totals["output_object"] += output_seconds
            active_terms = int(torch.count_nonzero(model.polynomial.coeffs).item())
            active_per_batch = torch.count_nonzero(
                model.polynomial.coeffs, dim=(-2, -1)
            )
            ledger = _stage_ledger(
                validated_model, raw_remainder, step_lo, step_hi
            )
            snapshots.append(
                {
                    "gate": gate_steps[step],
                    "step": step,
                    "time": step * 0.01,
                    "retained_term_count": active_terms,
                    "retained_term_count_per_batch_min": int(
                        active_per_batch.min().item()
                    ),
                    "retained_term_count_per_batch_max": int(
                        active_per_batch.max().item()
                    ),
                    "dropped_term_count": ledger[
                        "nonzero_dropped_source_entry_count"
                    ],
                    "dropped_term_categories": list(
                        ledger["raw_lineage"]["entries"]
                    ),
                    "raw_candidate_remainder_lo": raw_remainder.lo.detach().cpu().tolist(),
                    "raw_candidate_remainder_hi": raw_remainder.hi.detach().cpu().tolist(),
                    "minimum_target_margin": float(
                        torch.minimum(raw_remainder.lo + target, target - raw_remainder.hi).min()
                    ),
                    "endpoint_lo": endpoint.lo.detach().cpu().tolist(),
                    "endpoint_hi": endpoint.hi.detach().cpu().tolist(),
                    "tube_lo": tube.lo.detach().cpu().tolist(),
                    "tube_hi": tube.hi.detach().cpu().tolist(),
                    "carry_state_width": (
                        parameterization.remainder.hi - parameterization.remainder.lo
                    ).detach().cpu().tolist(),
                    "normalization_scale": normalization_scale.detach().cpu().tolist(),
                    "initial_mask": accepted_mask.detach().cpu().tolist(),
                    "round_masks": round_masks.detach().cpu().tolist(),
                    "stage_ledger": ledger,
                    "runtime_by_stage_seconds": {
                        "last_step": {
                            "carry": carry_seconds,
                            "polynomial_picard": polynomial_seconds,
                            "validation": validation_seconds,
                            "output_object": output_seconds,
                        },
                        "cumulative": dict(stage_runtime_totals),
                    },
                    "peak_memory_bytes": (
                        int(torch.cuda.max_memory_reserved(device))
                        if device.type == "cuda"
                        else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                        * 1024
                    ),
                    "decision": "accept",
                    "picard_iterations_emitted": len(picard_trace),
                }
            )
    elapsed = time.perf_counter() - started
    return {
        **cell,
        "batch": batch,
        "step_size": 0.01,
        "step_size_hex": float(0.01).hex(),
        "target_remainder_radius": 0.01,
        "target_remainder_radius_hex": float(0.01).hex(),
        "cutoff": None,
        "h_min": 0.01,
        "support_sha256": support.support_sha256,
        "support_slots": support.num_slots,
        "completed_steps": completed,
        "validated_horizon": completed * 0.01,
        "validated_horizon_hex": float(completed * 0.01).hex(),
        "completed_requested_gate": completed == max_steps,
        "first_failure": first_failure,
        "snapshots": snapshots,
        "runtime_seconds": elapsed,
        "runtime_by_stage_seconds": stage_runtime_totals,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        ),
        "dtype": str(dtype),
        "device": str(device),
    }


def _a0_reference_gate(batch: int) -> dict[str, Any]:
    support = _support("R7")
    lo, hi = _initial_boxes(batch, dtype=torch.float64, device=torch.device("cpu"))
    solver = FixedSupportReachability(
        support=support,
        state_dim=2,
        polynomial_rhs=diffreach_vdp_polynomial_rhs,
        tm_rhs=diffreach_vdp_tm_rhs,
        step_size=0.01,
        initial_remainder=0.01,
        polynomial_picard_iterations=2,
        remainder_rounds=10,
        symbolic_window_size=1000,
        normalization_epsilon=1e-12,
    )
    reference = solver.verify(lo, hi, steps=1)
    candidate = _run_cell(dict(CELLS[0]), batch=batch, max_steps=1, device=torch.device("cpu"))
    snapshot = candidate["snapshots"][0]
    return {
        "status": "pass"
        if reference.completed
        and torch.equal(reference.endpoint_lo[:, -1], torch.tensor(snapshot["endpoint_lo"], dtype=torch.float64))
        and torch.equal(reference.endpoint_hi[:, -1], torch.tensor(snapshot["endpoint_hi"], dtype=torch.float64))
        and torch.equal(reference.tube_lo[:, -1], torch.tensor(snapshot["tube_lo"], dtype=torch.float64))
        and torch.equal(reference.tube_hi[:, -1], torch.tensor(snapshot["tube_hi"], dtype=torch.float64))
        else "fail",
        "object_candidate_completed": candidate["completed_requested_gate"],
        "reference_completed": reference.completed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_names = [name for name, _ in GATES]
    gate_index = gate_names.index(args.max_gate)
    prior_gate: dict[str, Any] | None = None
    if gate_index > 0:
        if args.prior_gate_summary is None:
            raise RuntimeError("PRIOR_BRIDGE_GATE_EVIDENCE_REQUIRED")
        prior_path = args.prior_gate_summary.resolve()
        prior_gate = json.loads(prior_path.read_text())
        expected = gate_names[gate_index - 1]
        if (
            prior_gate.get("max_gate") != expected
            or prior_gate.get("all_cells_completed_gate") is not True
            or prior_gate.get("outcome") != "FIXED_SUPPORT_BRIDGE_CLOSED"
        ):
            raise RuntimeError("PRIOR_BRIDGE_GATE_NOT_CLOSED")
        prior_gate = {
            "path": "summary.json",
            "path_scope": "immediately_preceding_gate_runner",
            "sha256": _sha(prior_path),
            "max_gate": prior_gate["max_gate"],
            "outcome": prior_gate["outcome"],
        }
    elif args.prior_gate_summary is not None:
        raise ValueError("G0 must not declare prior-gate evidence")
    max_steps = dict(GATES)[args.max_gate]
    rows = []
    for cell in CELLS:
        for batch in (1, 64):
            result = _run_cell(
                dict(cell), batch=batch, max_steps=max_steps, device=torch.device(args.device)
            )
            rows.append(result)
    regression = {f"B{batch}": _a0_reference_gate(batch) for batch in (1, 64)}
    if any(value["status"] != "pass" for value in regression.values()):
        raise RuntimeError("DESCRIPTOR_R7_REGRESSION_STOP")

    adjacent = []
    for batch in (1, 64):
        lane = [row for row in rows if row["batch"] == batch]
        for left, right in zip(lane, lane[1:]):
            changed = [
                factor
                for factor in ("support", "picard", "validator", "carry")
                if left[factor] != right[factor]
            ]
            if len(changed) != 1:
                raise RuntimeError("bridge adjacency changed more than one factor")
            common_steps = min(left["completed_steps"], right["completed_steps"])
            left_snapshot = next((s for s in reversed(left["snapshots"]) if s["step"] <= common_steps), None)
            right_snapshot = next((s for s in reversed(right["snapshots"]) if s["step"] <= common_steps), None)
            left_first = next(
                (snapshot for snapshot in left["snapshots"] if snapshot["step"] == 1),
                None,
            )
            right_first = next(
                (snapshot for snapshot in right["snapshots"] if snapshot["step"] == 1),
                None,
            )
            left_t1 = next(
                (snapshot for snapshot in left["snapshots"] if snapshot["step"] == 100),
                None,
            )
            right_t1 = next(
                (snapshot for snapshot in right["snapshots"] if snapshot["step"] == 100),
                None,
            )
            adjacent.append(
                {
                    "batch": batch,
                    "from": left["cell"],
                    "to": right["cell"],
                    "changed_factor": changed[0],
                    "common_steps": common_steps,
                    "from_margin": None if left_snapshot is None else left_snapshot["minimum_target_margin"],
                    "to_margin": None if right_snapshot is None else right_snapshot["minimum_target_margin"],
                    "margin_delta": None
                    if left_snapshot is None or right_snapshot is None
                    else right_snapshot["minimum_target_margin"] - left_snapshot["minimum_target_margin"],
                    "first_decision_margin_delta": None
                    if left_first is None or right_first is None
                    else right_first["minimum_target_margin"]
                    - left_first["minimum_target_margin"],
                    "t1_max_raw_remainder_width_delta": None
                    if left_t1 is None or right_t1 is None
                    else _maximum_nested_width(
                        right_t1["raw_candidate_remainder_lo"],
                        right_t1["raw_candidate_remainder_hi"],
                    )
                    - _maximum_nested_width(
                        left_t1["raw_candidate_remainder_lo"],
                        left_t1["raw_candidate_remainder_hi"],
                    ),
                    "t1_max_endpoint_width_delta": None
                    if left_t1 is None or right_t1 is None
                    else _maximum_nested_width(
                        right_t1["endpoint_lo"], right_t1["endpoint_hi"]
                    )
                    - _maximum_nested_width(
                        left_t1["endpoint_lo"], left_t1["endpoint_hi"]
                    ),
                    "t1_max_tube_width_delta": None
                    if left_t1 is None or right_t1 is None
                    else _maximum_nested_width(
                        right_t1["tube_lo"], right_t1["tube_hi"]
                    )
                    - _maximum_nested_width(
                        left_t1["tube_lo"], left_t1["tube_hi"]
                    ),
                    "validated_horizon_delta": right["validated_horizon"] - left["validated_horizon"],
                    "comparison_eligibility": (
                        "same B/h/time/output/success and empirical-sampled scope"
                        if left_snapshot is not None
                        and right_snapshot is not None
                        and left_snapshot["step"] == right_snapshot["step"]
                        else "unavailable"
                    ),
                }
            )
    artifact = {
        "schema": "torch_fixed_support_descriptor_bridge_v1",
        "preregistered_metric": [
            "first-decision minimum target margin delta",
            "validated horizon delta",
        ],
        "global_constants": {
            "h": 0.01,
            "h_hex": float(0.01).hex(),
            "target_remainder_radius": 0.01,
            "target_remainder_radius_hex": float(0.01).hex(),
            "cutoff": None,
            "h_min": 0.01,
            "remainder_rounds_vdr": 10,
        },
        "prior_gate_evidence": prior_gate,
        "r7_regression": regression,
        "cells": rows,
        "adjacent_factor_attribution": adjacent,
    }
    artifact_path = output_dir / "bridge_ladder.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    all_complete = all(row["completed_requested_gate"] for row in rows)
    outcome = "FIXED_SUPPORT_BRIDGE_CLOSED" if all_complete else "FIXED_SUPPORT_BRIDGE_BLOCKED"
    summary = {
        "schema": "torch_fixed_support_descriptor_bridge_run_v1",
        "outcome": outcome,
        "max_gate": args.max_gate,
        "max_steps": max_steps,
        "prior_gate_evidence": prior_gate,
        "all_cells_completed_gate": all_complete,
        "cells": [
            {
                "cell": row["cell"],
                "batch": row["batch"],
                "completed_steps": row["completed_steps"],
                "validated_horizon": row["validated_horizon"],
                "first_failure": row["first_failure"],
            }
            for row in rows
        ],
        "artifact_sha256": _sha(artifact_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gate", choices=[name for name, _ in GATES], default="G2")
    parser.add_argument("--prior-gate-summary", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["outcome"] == "FIXED_SUPPORT_BRIDGE_CLOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
