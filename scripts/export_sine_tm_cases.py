#!/usr/bin/env python3
"""Export sanitized analytic-sine regression cases for the public audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mpmath
import torch

from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    sin_tm,
)


def affine(center: float, radius: float, remainder: tuple[float, float]) -> BatchedTaylorModel:
    basis = BatchedMonomialBasis.build(2, 3, "cpu")
    coefficients = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    coefficients[..., basis.constant_index] = center
    coefficients[..., basis.term_index((0, 1))] = radius
    return BatchedTaylorModel(
        BatchedPolynomial(coefficients, basis),
        torch.tensor([[remainder[0]]], dtype=torch.float64),
        torch.tensor([[remainder[1]]], dtype=torch.float64),
        torch.tensor([[0.0, -1.0]], dtype=torch.float64),
        torch.tensor([[0.1, 1.0]], dtype=torch.float64),
    )


def case(center: float, radius: float, remainder: tuple[float, float], order: int) -> dict[str, object]:
    model = affine(center, radius, remainder)
    result = sin_tm(model, order=order)
    lower, upper = result.range_bound(context="sine_public_case")
    input_lower = center - radius + remainder[0]
    input_upper = center + radius + remainder[1]
    with mpmath.workdps(100):
        samples = [
            mpmath.sin(mpmath.mpf(input_lower) + (mpmath.mpf(input_upper) - input_lower) * index / 10000)
            for index in range(10001)
        ]
        oracle_lower = float(min(samples))
        oracle_upper = float(max(samples))
    return {
        "input": {"center": center, "affine_radius": radius, "remainder": list(remainder), "physical_interval": [input_lower, input_upper]},
        "order": order,
        "output_enclosure": [lower.item(), upper.item()],
        "high_precision_grid_sanity": [oracle_lower, oracle_upper],
        "contains_grid": lower.item() <= oracle_lower and upper.item() >= oracle_upper,
        "remainder_ledger_categories": sorted(result.ledger.entries),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    generic = [
        case(center, radius, remainder, order)
        for order in range(4)
        for center, radius, remainder in (
            (0.0, 0.0, (0.0, 0.0)),
            (0.0, 0.4, (0.0, 0.0)),
            (-0.35, 0.05, (0.0, 0.0)),
            (0.2, 0.15, (-0.01, 0.02)),
            (1.6, 0.2, (0.0, 0.0)),
        )
    ]
    tora = [
        case(center, radius, (0.0, 0.0), 2)
        for center, radius in ((-0.35, 0.05), (-0.2, 0.2), (0.4, 0.1), (1.5, 0.25))
    ]
    wide_failed = False
    try:
        sin_tm(affine(0.0, 4.01, (0.0, 0.0)), order=3)
    except ValueError:
        wide_failed = True
    unit_payload = {
        "schema": "sine_tm_unit_cases_v1",
        "formal_proof_basis": "outward rational Maclaurin coefficient intervals plus |f^(n)|<=1 Lagrange tail",
        "high_precision_grid_is_only_sanity": True,
        "cases": generic,
        "wide_domain_fail_closed": wide_failed,
    }
    tora_payload = {
        "schema": "sine_tm_tora_domain_cases_v1",
        "cases": tora,
        "all_contain_high_precision_grid": all(row["contains_grid"] for row in tora),
    }
    (output / "unit_cases.json").write_text(json.dumps(unit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "tora_domain_cases.json").write_text(json.dumps(tora_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"generic_cases": len(generic), "tora_cases": len(tora), "wide_domain_fail_closed": wide_failed, "status": "PASS" if all(row["contains_grid"] for row in generic + tora) and wide_failed else "FAIL"}))
    return 0 if all(row["contains_grid"] for row in generic + tora) and wide_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
