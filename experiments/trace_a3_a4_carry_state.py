#!/usr/bin/env python3
"""Reproduce one frozen A3/A4 cell and trace every carry boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

try:
    from .diffreach_torch_full_horizon_common import (
        array_record, capture_npz, write_json, write_jsonl_row,
    )
    from .run_fixed_support_descriptor_bridge import _initial_boxes, _support
except ImportError:
    from diffreach_torch_full_horizon_common import (
        array_record, capture_npz, write_json, write_jsonl_row,
    )
    from run_fixed_support_descriptor_bridge import _initial_boxes, _support
from torch_tm_flowpipe.fixed_support import (
    FixedSupportInterval,
    FixedSupportPolynomial,
    FixedSupportSymbolicRemainderState,
    FixedSupportTaylorModel,
    diffreach_vdp_polynomial_rhs,
    diffreach_vdp_tm_rhs,
    fixed_support_build_linear_tm,
    fixed_support_identity_parameterization,
    fixed_support_polynomial_picard,
    fixed_support_step_boxes,
    fixed_support_symbolic_step_linear,
)


SCHEMA = "torch_r35_a3_a4_carry_trace_v1"
EXPECTED = {
    ("A3", 1): (1000, None),
    ("A3", 64): (1000, None),
    ("A4", 1): (319, 320),
    ("A4", 64): (333, 334),
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_record(
    value: torch.Tensor,
    *,
    coordinate: str,
    classification: str,
) -> dict[str, Any]:
    record = array_record(value.detach().cpu().numpy())
    return {
        **record,
        "torch_dtype": str(value.dtype),
        "device": str(value.device),
        "finite": bool(torch.all(torch.isfinite(value)).item()) if value.dtype != torch.bool else True,
        "coordinate_semantics": coordinate,
        "classification": classification,
    }


def _model_fields(
    prefix: str,
    model: FixedSupportTaylorModel,
    *,
    coordinate: str,
    classification: str,
) -> dict[str, dict[str, Any]]:
    return {
        f"{prefix}_polynomial": _tensor_record(
            model.polynomial.coeffs, coordinate=coordinate, classification=classification
        ),
        f"{prefix}_remainder_lo": _tensor_record(
            model.remainder.lo, coordinate=coordinate, classification=classification
        ),
        f"{prefix}_remainder_hi": _tensor_record(
            model.remainder.hi, coordinate=coordinate, classification=classification
        ),
    }


def _raw_model(
    polynomial: FixedSupportPolynomial,
    raw_remainder: FixedSupportInterval,
    ledger: Any,
) -> FixedSupportTaylorModel:
    return FixedSupportTaylorModel(polynomial, raw_remainder, ledger)


def _cni_observed(
    endpoint: FixedSupportTaylorModel,
    parameterization: FixedSupportTaylorModel,
    eval_lo: torch.Tensor,
    eval_hi: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    FixedSupportTaylorModel,
    FixedSupportTaylorModel,
    FixedSupportTaylorModel,
]:
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
    inverse = torch.where(scale == 0.0, torch.ones_like(scale), 1.0 / scale)
    normalized = centered.scale(inverse)
    return center, scale, inverse, normalized, inserted, centered


def _capture_prestate(
    path: Path,
    model: FixedSupportTaylorModel,
    parameterization: FixedSupportTaylorModel,
    symbolic: FixedSupportSymbolicRemainderState,
) -> None:
    capture_npz(
        path,
        {
            "model_polynomial": model.polynomial.coeffs.detach().cpu().numpy(),
            "model_remainder_lo": model.remainder.lo.detach().cpu().numpy(),
            "model_remainder_hi": model.remainder.hi.detach().cpu().numpy(),
            "parameterization_polynomial": parameterization.polynomial.coeffs.detach().cpu().numpy(),
            "parameterization_remainder_lo": parameterization.remainder.lo.detach().cpu().numpy(),
            "parameterization_remainder_hi": parameterization.remainder.hi.detach().cpu().numpy(),
            "symbolic_Phi": symbolic.phi_buffer.detach().cpu().numpy(),
            "symbolic_J_lo": symbolic.j_buffer.lo.detach().cpu().numpy(),
            "symbolic_J_hi": symbolic.j_buffer.hi.detach().cpu().numpy(),
            "symbolic_count": symbolic.count.detach().cpu().numpy(),
            "symbolic_inverse_scale": symbolic.inverse_scale.detach().cpu().numpy(),
        },
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=("A3", "A4"), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 64), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--capture-before-steps", default="1,2,101,320")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not args.smoke and args.max_steps != 1000:
        raise ValueError("frozen A3/A4 reproduction requires max_steps=1000")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    capture_dir = args.output_dir / "prestates"
    capture_dir.mkdir()
    capture_steps = {int(value) for value in args.capture_before_steps.split(",") if value}
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    dtype = torch.float64
    device = torch.device("cpu")
    support = _support("R35")
    initial_lo, initial_hi = _initial_boxes(args.batch, dtype=dtype, device=device)
    center = 0.5 * (initial_lo + initial_hi)
    scale = 0.5 * (initial_hi - initial_lo)
    model = fixed_support_build_linear_tm(center, scale, support)
    parameterization = fixed_support_identity_parameterization(
        args.batch, 2, support, dtype=dtype, device=device
    )
    symbolic = FixedSupportSymbolicRemainderState.initialize(
        args.batch, 2, 1000, dtype=dtype, device=device
    )
    step_lo, step_hi, eval_lo, eval_hi = fixed_support_step_boxes(
        args.batch, 2, 0.01, dtype=dtype, device=device
    )
    target = torch.full((args.batch, 2), 0.01, dtype=dtype, device=device)
    trace_path = args.output_dir / "state_trace.jsonl"
    metrics_path = args.output_dir / "metrics.csv"
    completed = 0
    failure: dict[str, Any] | None = None
    started = time.perf_counter()
    with trace_path.open("w", encoding="utf-8") as trace_handle, metrics_path.open(
        "w", encoding="utf-8", newline=""
    ) as metrics_handle:
        metric_fields = (
            "step", "time", "decision", "minimum_target_margin", "model_remainder_width_max",
            "parameterization_remainder_width_max", "scale_max", "inverse_scale_max",
            "endpoint_width_max", "tube_width_max", "composition_ledger_width_max",
        )
        writer = csv.DictWriter(metrics_handle, fieldnames=metric_fields)
        writer.writeheader()
        for step_number in range(1, args.max_steps + 1):
            if args.batch == 1 and step_number in capture_steps:
                _capture_prestate(
                    capture_dir / f"before_step_{step_number:04d}.npz",
                    model,
                    parameterization,
                    symbolic,
                )
            pre_model = model
            pre_parameterization = parameterization
            pre_symbolic = symbolic
            endpoint_previous = model.evaluate_time(0.01)
            inserted: FixedSupportTaylorModel | None = None
            centered_inserted: FixedSupportTaylorModel | None = None
            if args.cell == "A3":
                carry = fixed_support_symbolic_step_linear(
                    parameterization,
                    endpoint_previous,
                    symbolic,
                    eval_lo,
                    eval_hi,
                    epsilon=1e-12,
                )
                step_center = endpoint_previous.polynomial.coeffs[..., support.constant_slot]
                normalization_scale = carry.scale
                inverse_scale = carry.state.inverse_scale
                next_parameterization = carry.normalized_parameterization
                next_symbolic = carry.state
            else:
                (
                    step_center,
                    normalization_scale,
                    inverse_scale,
                    next_parameterization,
                    inserted,
                    centered_inserted,
                ) = _cni_observed(endpoint_previous, parameterization, eval_lo, eval_hi)
                next_symbolic = symbolic
            new_x0 = fixed_support_build_linear_tm(step_center, normalization_scale, support)
            polynomial, picard_trace = fixed_support_polynomial_picard(
                new_x0.polynomial,
                diffreach_vdp_polynomial_rhs,
                step_lo,
                step_hi,
                iterations=4,
            )
            seed = FixedSupportTaylorModel(
                polynomial, FixedSupportInterval(-target, target)
            )
            raw_image = new_x0.add(
                diffreach_vdp_tm_rhs(seed, step_lo, step_hi).integrate_time(step_lo, step_hi)
            )
            polynomial_difference = raw_image.polynomial.sub(seed.polynomial).range(
                step_lo, step_hi
            )
            raw_remainder = raw_image.remainder.add(polynomial_difference)
            accepted_mask = raw_remainder.subseteq_elem(seed.remainder)
            validated_model = _raw_model(polynomial, raw_remainder, raw_image.ledger)
            composed = validated_model.compose_affine(next_parameterization, 0.01)
            endpoint_box_lo = step_lo.clone()
            endpoint_box_lo[:, support.local_time_index] = 0.01
            endpoint = composed.range(endpoint_box_lo, step_hi)
            tube = composed.range(step_lo, step_hi)
            accepted = bool(torch.all(accepted_mask).item())
            fields: dict[str, Any] = {}
            fields.update(
                _model_fields(
                    "model", pre_model, coordinate="previous_local_normalized", classification="normalized"
                )
            )
            fields.update(
                _model_fields(
                    "parameterization",
                    pre_parameterization,
                    coordinate="previous_local_to_physical_right_map",
                    classification="normalized_map",
                )
            )
            fields.update(
                _model_fields(
                    "endpoint_previous",
                    endpoint_previous,
                    coordinate="previous_local_endpoint_tau_h",
                    classification="normalized_endpoint",
                )
            )
            fields["center"] = _tensor_record(
                step_center, coordinate="physical_state", classification="physical"
            )
            fields["scale"] = _tensor_record(
                normalization_scale, coordinate="physical_per_new_normalized", classification="physical_scale"
            )
            fields["inverse_scale"] = _tensor_record(
                inverse_scale, coordinate="new_normalized_per_physical", classification="normalization_map"
            )
            fields.update(
                _model_fields(
                    "normalized_parameterization",
                    next_parameterization,
                    coordinate="new_local_normalized_to_physical_deviation",
                    classification="normalized_map",
                )
            )
            fields.update(
                _model_fields(
                    "raw_candidate",
                    _raw_model(raw_image.polynomial, raw_remainder, raw_image.ledger),
                    coordinate="new_local_normalized",
                    classification="candidate",
                )
            )
            fields.update(
                _model_fields(
                    "validated_model",
                    validated_model,
                    coordinate="new_local_normalized",
                    classification="accepted" if accepted else "rejected_candidate",
                )
            )
            fields["symbolic_J_lo"] = _tensor_record(
                pre_symbolic.j_buffer.lo,
                coordinate="physical_symbolic_remainder",
                classification="symbolic_carry",
            )
            fields["symbolic_J_hi"] = _tensor_record(
                pre_symbolic.j_buffer.hi,
                coordinate="physical_symbolic_remainder",
                classification="symbolic_carry",
            )
            fields["symbolic_Phi"] = _tensor_record(
                pre_symbolic.phi_buffer,
                coordinate="linear_transition",
                classification="symbolic_carry",
            )
            fields["symbolic_queue_count"] = _tensor_record(
                pre_symbolic.count,
                coordinate="queue_index",
                classification="symbolic_carry",
            )
            fields["endpoint_lo"] = _tensor_record(
                endpoint.lo, coordinate="physical_state", classification="physical_endpoint"
            )
            fields["endpoint_hi"] = _tensor_record(
                endpoint.hi, coordinate="physical_state", classification="physical_endpoint"
            )
            fields["tube_lo"] = _tensor_record(
                tube.lo, coordinate="physical_state", classification="physical_segment_tube"
            )
            fields["tube_hi"] = _tensor_record(
                tube.hi, coordinate="physical_state", classification="physical_segment_tube"
            )
            fields["accepted_mask"] = _tensor_record(
                accepted_mask, coordinate="component_decision", classification="decision"
            )
            for picard_index, picard in enumerate(picard_trace, start=1):
                fields[f"polynomial_picard_{picard_index}"] = _tensor_record(
                    picard.coeffs,
                    coordinate="new_local_normalized",
                    classification="polynomial_construction",
                )
            if inserted is not None and centered_inserted is not None:
                fields.update(
                    _model_fields(
                        "inserted_endpoint",
                        inserted,
                        coordinate="physical_over_previous_normalized_generators",
                        classification="physical_inserted",
                    )
                )
                fields.update(
                    _model_fields(
                        "centered_inserted",
                        centered_inserted,
                        coordinate="physical_deviation_over_previous_normalized_generators",
                        classification="physical_centered",
                    )
                )
            write_jsonl_row(
                trace_handle,
                {
                    "schema": SCHEMA,
                    "cell": args.cell,
                    "batch": args.batch,
                    "step": step_number,
                    "time_before": (step_number - 1) * 0.01,
                    "time_before_hex": float((step_number - 1) * 0.01).hex(),
                    "decision": "accept" if accepted else "reject",
                    "inserted_endpoint_status": (
                        "constructed_by_CNI" if inserted is not None else "not_constructed_by_CDR"
                    ),
                    "fields": fields,
                },
            )
            margin = torch.minimum(raw_remainder.lo + target, target - raw_remainder.hi)
            ledger_width = max(
                (
                    float((interval.hi - interval.lo).max().item())
                    for _, interval in composed.ledger.entries
                ),
                default=0.0,
            )
            writer.writerow(
                {
                    "step": step_number,
                    "time": (step_number - 1) * 0.01,
                    "decision": "accept" if accepted else "reject",
                    "minimum_target_margin": float(margin.min().item()),
                    "model_remainder_width_max": float(pre_model.remainder.width.max().item()),
                    "parameterization_remainder_width_max": float(
                        pre_parameterization.remainder.width.max().item()
                    ),
                    "scale_max": float(normalization_scale.max().item()),
                    "inverse_scale_max": float(inverse_scale.max().item()),
                    "endpoint_width_max": float(endpoint.width.max().item()),
                    "tube_width_max": float(tube.width.max().item()),
                    "composition_ledger_width_max": ledger_width,
                }
            )
            metrics_handle.flush()
            if not accepted:
                failure = {
                    "step": step_number,
                    "time": (step_number - 1) * 0.01,
                    "mask": accepted_mask.detach().cpu().tolist(),
                    "raw_remainder_lo": raw_remainder.lo.detach().cpu().tolist(),
                    "raw_remainder_hi": raw_remainder.hi.detach().cpu().tolist(),
                }
                break
            model = validated_model
            parameterization = next_parameterization
            symbolic = next_symbolic
            completed = step_number
    elapsed = time.perf_counter() - started
    expected_completed, expected_failure = EXPECTED[(args.cell, args.batch)]
    reproduced = not args.smoke and completed == expected_completed and (
        (failure is None and expected_failure is None)
        or (failure is not None and failure["step"] == expected_failure)
    )
    summary = {
        "schema": SCHEMA,
        "source_sha": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "cell": args.cell,
        "batch": args.batch,
        "support": "R35",
        "support_sha256": support.support_sha256,
        "picard": 4,
        "validator": "VRAW",
        "carry": "CDR" if args.cell == "A3" else "CNI",
        "h": 0.01,
        "h_hex": float(0.01).hex(),
        "target": 0.01,
        "target_hex": float(0.01).hex(),
        "cutoff": None,
        "completed_steps": completed,
        "validated_horizon": completed * 0.01,
        "validated_horizon_hex": float(completed * 0.01).hex(),
        "first_failure": failure,
        "expected_completed_steps": expected_completed,
        "expected_failure_step": expected_failure,
        "reproduction_status": (
            "smoke_only" if args.smoke else ("reproduced" if reproduced else "BRIDGE_REPRODUCTION_STOP")
        ),
        "trace_sha256": _sha(trace_path),
        "metrics_sha256": _sha(metrics_path),
        "runtime_with_full_trace_s": elapsed,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "no_hidden_fallback": True,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "artifact_manifest.json",
        {
            "schema": SCHEMA,
            "files": [
                {"path": path.relative_to(args.output_dir).as_posix(), "sha256": _sha(path), "bytes": path.stat().st_size}
                for path in [trace_path, metrics_path, *sorted(capture_dir.glob("*.npz")), args.output_dir / "summary.json"]
            ],
        },
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if reproduced or args.smoke else 2


if __name__ == "__main__":
    raise SystemExit(main())
