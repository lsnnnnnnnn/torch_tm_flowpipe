#!/usr/bin/env python3
"""Minimal clean-room Bernstein range feasibility experiment.

This module does not import BERN-NN-IBF.  It implements the standard
power-to-tensor-product-Bernstein coefficient identity so that the study can
test the one reusable idea without copying the unlicensed upstream source or
pretending that BERN is an ODE reachability tool.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from itertools import product
from math import comb
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from common import write_csv, write_json
from torch_tm_flowpipe import Interval
from torch_tm_flowpipe.polynomial import Polynomial


Exponent = tuple[int, ...]


@dataclass(frozen=True)
class Case:
    name: str
    purpose: str
    terms: Mapping[Exponent, float]
    box: tuple[tuple[float, float], ...]
    exact_range: tuple[float, float]


def _degrees(
    terms: Mapping[Exponent, float], n_vars: int
) -> tuple[int, ...]:
    return tuple(
        max((exponent[index] for exponent in terms), default=0)
        for index in range(n_vars)
    )


def power_to_bernstein_coefficients(
    terms: Mapping[Exponent, float],
    box: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return dense tensor-product Bernstein coefficients on ``box``."""
    n_vars = len(box)
    if any(len(exponent) != n_vars for exponent in terms):
        raise ValueError("exponent and box dimensions differ")
    if any(
        len(bounds) != 2
        or not math.isfinite(float(bounds[0]))
        or not math.isfinite(float(bounds[1]))
        or float(bounds[0]) >= float(bounds[1])
        for bounds in box
    ):
        raise ValueError("box entries must be finite nondegenerate intervals")
    degrees = _degrees(terms, n_vars)
    coefficients = np.zeros(
        tuple(degree + 1 for degree in degrees), dtype=np.float64
    )
    for exponent, coefficient in terms.items():
        rows: list[np.ndarray] = []
        for power, degree, bounds in zip(exponent, degrees, box):
            lower, upper = map(float, bounds)
            row = np.zeros(degree + 1, dtype=np.float64)
            for index in range(degree + 1):
                row[index] = sum(
                    comb(power, unit_power)
                    * lower ** (power - unit_power)
                    * (upper - lower) ** unit_power
                    * comb(index, unit_power)
                    / comb(degree, unit_power)
                    for unit_power in range(min(power, index) + 1)
                )
            rows.append(row)
        contribution = np.asarray(float(coefficient), dtype=np.float64)
        for axis, row in enumerate(rows):
            shape = [1] * n_vars
            shape[axis] = len(row)
            contribution = contribution * row.reshape(shape)
        coefficients += contribution
    return coefficients


def _bernstein_candidate_hull(
    terms: Mapping[Exponent, float],
    box: Sequence[Sequence[float]],
) -> tuple[float, float, np.ndarray]:
    coefficients = power_to_bernstein_coefficients(terms, box)
    lower = float(np.min(coefficients))
    upper = float(np.max(coefficients))
    # This deliberately conservative allowance is a feasibility guard, not a
    # proof of every floating-point operation.  A production backend still
    # needs directed conversion/arithmetic or an MPFR cross-check.
    degrees = _degrees(terms, len(box))
    operation_bound = max(
        1,
        len(terms)
        * (sum(degrees) + len(box) + 2)
        * (max(degrees, default=0) + 2),
    )
    magnitude = max(1.0, float(np.max(np.abs(coefficients))))
    inflation = (
        256.0
        * np.finfo(np.float64).eps
        * float(operation_bound)
        * magnitude
    )
    return (
        float(np.nextafter(lower - inflation, -np.inf)),
        float(np.nextafter(upper + inflation, np.inf)),
        coefficients,
    )


def _sample_range(case: Case, *, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    points = np.column_stack(
        [
            rng.uniform(lower, upper, size=4096)
            for lower, upper in case.box
        ]
    )
    values = np.zeros(points.shape[0], dtype=np.float64)
    for exponent, coefficient in case.terms.items():
        values += float(coefficient) * np.prod(
            points ** np.asarray(exponent, dtype=np.int64), axis=1
        )
    return float(np.min(values)), float(np.max(values))


def _cases() -> list[Case]:
    return [
        Case(
            "difference_squared",
            "cancellation_and_cross_term",
            {(2, 0): 1.0, (1, 1): -2.0, (0, 2): 1.0},
            ((0.0, 1.0), (0.0, 1.0)),
            (0.0, 1.0),
        ),
        Case(
            "difference_fourth",
            "higher_order_cancellation",
            {
                (4, 0): 1.0,
                (3, 1): -4.0,
                (2, 2): 6.0,
                (1, 3): -4.0,
                (0, 4): 1.0,
            },
            ((0.0, 1.0), (0.0, 1.0)),
            (0.0, 1.0),
        ),
        Case(
            "coupled_quadratic_x1_x2",
            "study_cross_term",
            {(1, 1): 1.0},
            ((0.08, 0.12), (0.18, 0.22)),
            (0.08 * 0.18, 0.12 * 0.22),
        ),
        Case(
            "coupled_quadratic_x1_squared_minus_x2",
            "study_rhs_range",
            {(2, 0): 1.0, (0, 1): -1.0},
            ((0.08, 0.12), (0.18, 0.22)),
            (0.08**2 - 0.22, 0.12**2 - 0.18),
        ),
        Case(
            "van_der_pol_x1_squared_x2",
            "study_nonlinear_cross_term",
            {(2, 1): 1.0},
            ((1.10, 1.40), (2.35, 2.45)),
            (1.10**2 * 2.35, 1.40**2 * 2.45),
        ),
    ]


def run(output: Path, *, repetitions: int = 10) -> dict[str, object]:
    torch.set_default_dtype(torch.float64)
    rows: list[dict[str, object]] = []
    for case_index, case in enumerate(_cases()):
        polynomial = Polynomial(dict(case.terms), len(case.box))
        domain = [Interval(lower, upper) for lower, upper in case.box]

        polynomial.evaluate_interval(domain)
        _bernstein_candidate_hull(case.terms, case.box)
        current_times: list[float] = []
        bernstein_times: list[float] = []
        current = None
        bernstein = None
        coefficients = None
        for _ in range(repetitions):
            started = time.perf_counter()
            current = polynomial.evaluate_interval(domain)
            current_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            lower, upper, coefficients = _bernstein_candidate_hull(
                case.terms, case.box
            )
            bernstein = (lower, upper)
            bernstein_times.append(time.perf_counter() - started)
        assert current is not None and bernstein is not None
        assert coefficients is not None
        current_lower, current_upper = current.to_tuple()
        exact_lower, exact_upper = case.exact_range
        sample_lower, sample_upper = _sample_range(
            case, seed=20260729 + case_index
        )
        current_width = current_upper - current_lower
        bernstein_width = bernstein[1] - bernstein[0]
        degrees = _degrees(case.terms, len(case.box))
        dense_bytes = int(coefficients.size * coefficients.itemsize)
        implicit_bytes = int(
            len(case.terms)
            * len(case.box)
            * (max(degrees, default=0) + 1)
            * 8
        )
        rows.append(
            {
                "case": case.name,
                "purpose": case.purpose,
                "dimensions": len(case.box),
                "term_count": len(case.terms),
                "per_variable_degree": json.dumps(degrees),
                "exact_lower": exact_lower,
                "exact_upper": exact_upper,
                "sample_lower": sample_lower,
                "sample_upper": sample_upper,
                "sampling_semantics": "deterministic_sanity_non_proof",
                "current_lower": current_lower,
                "current_upper": current_upper,
                "current_width": current_width,
                "bernstein_lower": bernstein[0],
                "bernstein_upper": bernstein[1],
                "bernstein_width": bernstein_width,
                "bernstein_over_current_width": (
                    bernstein_width / current_width
                    if current_width > 0
                    else math.nan
                ),
                "current_exact_range_contained": (
                    current_lower <= exact_lower
                    and current_upper >= exact_upper
                ),
                "bernstein_exact_range_contained": (
                    bernstein[0] <= exact_lower
                    and bernstein[1] >= exact_upper
                ),
                "dense_coefficient_bytes": dense_bytes,
                "implicit_storage_bytes_estimate": implicit_bytes,
                "storage_semantics": (
                    "float64 payload only; excludes object/allocator overhead"
                ),
                "current_runtime_median_s": statistics.median(current_times),
                "bernstein_runtime_median_s": statistics.median(
                    bernstein_times
                ),
                "runtime_repetitions": repetitions,
                "device": "cpu",
                "dtype": "float64",
                "proof_strength": (
                    "algebraic_Bernstein_hull_with_conservative_"
                    "floating_point_candidate;not_formal_roundoff_proof"
                ),
            }
        )
    write_csv(output / "bern_feasibility.csv", rows)
    tighter = [
        row
        for row in rows
        if float(row["bernstein_width"]) < float(row["current_width"])
    ]
    result: dict[str, object] = {
        "cases": len(rows),
        "all_current_exact_ranges_contained": all(
            bool(row["current_exact_range_contained"]) for row in rows
        ),
        "all_bernstein_exact_ranges_contained": all(
            bool(row["bernstein_exact_range_contained"]) for row in rows
        ),
        "strictly_tighter_cases": len(tighter),
        "strictly_tighter_case_names": [
            str(row["case"]) for row in tighter
        ],
        "external_bern_sha": (
            "ebcf54a0e06597a5388db0387865493c1dc96c07"
        ),
        "external_bern_used_as_runtime_dependency": False,
        "prototype_scope": "polynomial_range_query_only",
        "decision": {
            "polynomial_range_bounding": (
                "promising_after_formal_roundoff_backend_and_sparse_"
                "dimension_guard"
            ),
            "cross_term_dependency_preservation": (
                "algebraically_preserved_inside_each_polynomial;does_not_"
                "solve_multistep_TM_reset_or_remainder_dependency"
            ),
            "nn_controller_bounds": (
                "indirectly_relevant_but_not_exercised_by_plant_only_study"
            ),
            "gpu_batching": (
                "potentially_relevant_only_for_sufficient_batch_workload;"
                "not_measured_without_CUDA"
            ),
            "fourth_comparable_reachability_tool": False,
        },
        "cuda_available": torch.cuda.is_available(),
    }
    write_json(output / "bern_feasibility.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            run(output, repetitions=args.repetitions),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
