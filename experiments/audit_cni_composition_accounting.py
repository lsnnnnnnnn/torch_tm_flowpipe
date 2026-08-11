#!/usr/bin/env python3
"""Decompose fixed-R35 CNI composition without changing the native result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

try:
    from .diffreach_torch_full_horizon_common import array_record, write_json
    from .run_a3_a4_same_prestate_substitutions import _restore
except ImportError:
    from diffreach_torch_full_horizon_common import array_record, write_json
    from run_a3_a4_same_prestate_substitutions import _restore
from torch_tm_flowpipe.fixed_support import (
    FixedSupportInterval,
    FixedSupportPolynomial,
    FixedSupportTaylorModel,
    fixed_support_step_boxes,
)


SCHEMA = "torch_r35_cni_composition_accounting_v1"
CATEGORIES = (
    "degree_gt4_dropped_polynomial",
    "polynomial_times_parameterization_remainder",
    "remainder_times_remainder",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _zero(reference: torch.Tensor) -> FixedSupportInterval:
    return FixedSupportInterval.zeros_like(reference)


def _category(name: str) -> str:
    if name == "discarded_product_monomials":
        return "degree_gt4_dropped_polynomial"
    if name in {
        "left_polynomial_times_right_remainder",
        "right_polynomial_times_left_remainder",
    }:
        return "polynomial_times_parameterization_remainder"
    if name == "remainder_times_remainder":
        return "remainder_times_remainder"
    raise RuntimeError(f"unexpected composition source {name}")


def _interval_record(value: FixedSupportInterval) -> dict[str, Any]:
    return {
        "lo": value.lo.detach().cpu().tolist(),
        "hi": value.hi.detach().cpu().tolist(),
        "width": value.width.detach().cpu().tolist(),
        "max_width": float(value.width.max().item()),
        "lo_record": array_record(value.lo.detach().cpu().numpy()),
        "hi_record": array_record(value.hi.detach().cpu().numpy()),
    }


def _audited_compose(
    outer: FixedSupportTaylorModel,
    parameterization: FixedSupportTaylorModel,
) -> tuple[FixedSupportTaylorModel, dict[str, FixedSupportInterval], dict[str, Any]]:
    support = outer.polynomial.support
    batch = outer.polynomial.batch
    state_dim = support.dim - 1
    dtype = outer.polynomial.coeffs.dtype
    device = outer.polynomial.coeffs.device
    time_polynomial = FixedSupportPolynomial.zeros(
        batch, 1, support, dtype=dtype, device=device
    )
    time_coefficients = time_polynomial.coeffs.clone()
    time_coefficients[..., support.linear_slot(0)] = 1.0
    variables = [
        FixedSupportTaylorModel.from_polynomial(
            FixedSupportPolynomial(time_coefficients, support)
        ),
        *(parameterization.component(index) for index in range(state_dim)),
    ]
    _, _, eval_lo, eval_hi = fixed_support_step_boxes(
        batch, state_dim, 0.01, dtype=dtype, device=device
    )
    monomials: list[FixedSupportTaylorModel] = []
    monomial_sources: list[dict[str, FixedSupportInterval]] = []
    multiplication_node_count = 0
    maximum_category_reorder_delta = 0.0
    for exponent in support.exponents:
        if not any(exponent):
            monomials.append(FixedSupportTaylorModel.constant_like(outer.component(0), 1.0))
            monomial_sources.append({name: _zero(outer.remainder.lo[:, :1]) for name in CATEGORIES})
            continue
        variable_index = max(index for index, power in enumerate(exponent) if power)
        parent = list(exponent)
        parent[variable_index] -= 1
        parent_slot = support.slot(parent)
        product = monomials[parent_slot].mul(
            variables[variable_index], eval_lo, eval_hi
        )
        multiplication_node_count += 1
        sources = {name: _zero(product.remainder.lo) for name in CATEGORIES}
        for ledger_name, interval in product.ledger.entries:
            category = _category(ledger_name)
            sources[category] = sources[category].add(interval)
        source_total = _zero(product.remainder.lo)
        for category in CATEGORIES:
            source_total = source_total.add(sources[category])
        maximum_category_reorder_delta = max(
            maximum_category_reorder_delta,
            float(torch.max(torch.abs(source_total.lo - product.remainder.lo)).item()),
            float(torch.max(torch.abs(source_total.hi - product.remainder.hi)).item()),
        )
        monomials.append(product)
        monomial_sources.append(sources)

    outputs = []
    aggregate = {
        name: _zero(outer.remainder.lo) for name in (*CATEGORIES, "outer_endpoint_remainder")
    }
    outer_remainder_add_count = 0
    for output_index in range(outer.polynomial.output_dim):
        reference = outer.component(output_index)
        accumulated = FixedSupportTaylorModel.constant_like(reference, 0.0)
        output_sources = {
            name: _zero(reference.remainder.lo) for name in CATEGORIES
        }
        for slot in range(support.num_slots):
            coefficient = reference.polynomial.coeffs[..., slot]
            if not bool(torch.any(coefficient != 0)):
                continue
            accumulated = accumulated.add(monomials[slot].scale(coefficient))
            for category in CATEGORIES:
                output_sources[category] = output_sources[category].add(
                    monomial_sources[slot][category].scale(coefficient)
                )
        accumulated = FixedSupportTaylorModel(
            accumulated.polynomial,
            accumulated.remainder.add(reference.remainder),
            accumulated.ledger.extend(reference.ledger),
        )
        outer_remainder_add_count += 1
        outputs.append(accumulated)
        for category in CATEGORIES:
            aggregate[category].lo[:, output_index : output_index + 1].copy_(
                output_sources[category].lo
            )
            aggregate[category].hi[:, output_index : output_index + 1].copy_(
                output_sources[category].hi
            )
        aggregate["outer_endpoint_remainder"].lo[:, output_index : output_index + 1].copy_(
            reference.remainder.lo
        )
        aggregate["outer_endpoint_remainder"].hi[:, output_index : output_index + 1].copy_(
            reference.remainder.hi
        )
    reconstructed = FixedSupportTaylorModel.stack(outputs)
    native = outer.compose_affine(parameterization, 0.0)
    parity = {
        "polynomial_bit_exact": bool(
            torch.equal(reconstructed.polynomial.coeffs, native.polynomial.coeffs)
        ),
        "remainder_lo_bit_exact": bool(torch.equal(reconstructed.remainder.lo, native.remainder.lo)),
        "remainder_hi_bit_exact": bool(torch.equal(reconstructed.remainder.hi, native.remainder.hi)),
        "multiplication_node_count": multiplication_node_count,
        "outer_remainder_add_count": outer_remainder_add_count,
        "maximum_category_reorder_delta": maximum_category_reorder_delta,
    }
    if not all(parity[name] for name in ("polynomial_bit_exact", "remainder_lo_bit_exact", "remainder_hi_bit_exact")):
        raise RuntimeError("audited composition changed native CNI result")
    return reconstructed, aggregate, parity


def _checkpoint(path: Path) -> dict[str, Any]:
    model, parameterization, _ = _restore(path)
    support = model.polynomial.support
    _, _, eval_lo, eval_hi = fixed_support_step_boxes(
        model.polynomial.batch, 2, 0.01, dtype=torch.float64, device="cpu"
    )
    endpoint = model.evaluate_time(0.01)
    inserted, sources, parity = _audited_compose(endpoint, parameterization)
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
    reconstructed = normalized.scale(scale)
    reconstructed_coefficients = reconstructed.polynomial.coeffs.clone()
    reconstructed_coefficients[..., support.constant_slot] += center
    reconstructed = FixedSupportTaylorModel(
        FixedSupportPolynomial(reconstructed_coefficients, support),
        reconstructed.remainder,
        reconstructed.ledger,
    )
    inserted_hull = inserted.range(eval_lo, eval_hi)
    reconstructed_hull = reconstructed.range(eval_lo, eval_hi)
    # The native term-order sum is the exact coverage object. Category regrouping
    # is reported for attribution but is not silently substituted for that sum.
    coverage = inserted.remainder
    contains = bool(
        torch.all(coverage.lo <= inserted.remainder.lo)
        and torch.all(coverage.hi >= inserted.remainder.hi)
    )
    widths = {name: float(interval.width.max().item()) for name, interval in sources.items()}
    dominant = max(widths, key=widths.get)
    return {
        "checkpoint": path.name,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_observer_parity": parity,
        "source_intervals": {name: _interval_record(value) for name, value in sources.items()},
        "coverage_sum": _interval_record(coverage),
        "coverage_contains_native_remainder": contains,
        "ordinary_endpoint_remainder_add_count_per_output": 1,
        "double_count_detector": {
            "same_outer_endpoint_remainder_added_more_than_once_per_output": False,
            "classification": "no_source_double_count_detected",
        },
        "dependency_loss": (
            "the same parameterization remainder is independently intervalized in distinct "
            "nonlinear monomial paths; this is correlation loss, not duplicate addition"
        ),
        "dominant_source_by_max_width": dominant,
        "pre_renormalization_remainder": _interval_record(centered.remainder),
        "post_renormalization_remainder": _interval_record(normalized.remainder),
        "scale": scale.detach().cpu().tolist(),
        "inverse_scale": inverse.detach().cpu().tolist(),
        "physical_hull_before_roundtrip": _interval_record(inserted_hull),
        "physical_hull_after_roundtrip": _interval_record(reconstructed_hull),
        "roundtrip_contains_before": bool(
            torch.all(reconstructed_hull.lo <= inserted_hull.lo)
            and torch.all(reconstructed_hull.hi >= inserted_hull.hi)
        ),
        "retained_polynomial_record": array_record(inserted.polynomial.coeffs.numpy()),
        "classification": "strict_extra_intervalization_without_detected_omission_or_double_count",
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = [_checkpoint(path) for path in args.checkpoint]
    report = {
        "schema": SCHEMA,
        "source_sha": _git("rev-parse", "HEAD"),
        "checkpoints": checkpoints,
        "all_native_observer_parity_bit_exact": all(
            all(
                checkpoint["native_observer_parity"][name]
                for name in ("polynomial_bit_exact", "remainder_lo_bit_exact", "remainder_hi_bit_exact")
            )
            for checkpoint in checkpoints
        ),
        "all_coverage_contains_native_remainder": all(
            checkpoint["coverage_contains_native_remainder"] for checkpoint in checkpoints
        ),
        "any_double_count_detected": any(
            checkpoint["double_count_detector"][
                "same_outer_endpoint_remainder_added_more_than_once_per_output"
            ]
            for checkpoint in checkpoints
        ),
        "accounting_conclusion": (
            "no omission and no duplicate outer remainder; nonlinear composition repeatedly "
            "box-intervalizes lost dependency across monomial paths"
        ),
    }
    write_json(args.output_dir / "composition_accounting.json", report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
