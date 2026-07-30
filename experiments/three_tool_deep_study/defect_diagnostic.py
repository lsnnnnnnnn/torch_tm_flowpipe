#!/usr/bin/env python3
"""Common polynomial defect and comparison-radius diagnostic."""
from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from common import (
    evaluate_polynomial_interval,
    interval_add,
    interval_mul,
    interval_pow,
    load_spec,
    write_csv,
    write_json,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
Number = TypeVar("Number", int, float, Fraction)
Exponent = tuple[int, ...]
Polynomial = dict[Exponent, Any]


def polynomial_from_terms(
    terms: Sequence[Mapping[str, Any]], *, exact: bool = False
) -> Polynomial:
    result: Polynomial = {}
    for term in terms:
        exponent = tuple(map(int, term["exponents"]))
        coefficient: Any = (
            Fraction(str(term["coefficient"]))
            if exact
            else float(term["coefficient"])
        )
        result[exponent] = result.get(exponent, 0) + coefficient
    return {key: value for key, value in result.items() if value != 0}


def polynomial_add(
    left: Mapping[Exponent, Any],
    right: Mapping[Exponent, Any],
    *,
    right_scale: Any = 1,
) -> Polynomial:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = (
            result.get(exponent, 0) + right_scale * coefficient
        )
        if result[exponent] == 0:
            del result[exponent]
    return result


def polynomial_mul(
    left: Mapping[Exponent, Any], right: Mapping[Exponent, Any]
) -> Polynomial:
    result: Polynomial = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(
                a + b for a, b in zip(left_exp, right_exp)
            )
            result[exponent] = (
                result.get(exponent, 0)
                + left_coefficient * right_coefficient
            )
    return {key: value for key, value in result.items() if value != 0}


def polynomial_pow(polynomial: Mapping[Exponent, Any], power: int) -> Polynomial:
    if not polynomial:
        return {}
    variables = len(next(iter(polynomial)))
    result: Polynomial = {(0,) * variables: 1}
    for _ in range(int(power)):
        result = polynomial_mul(result, polynomial)
    return result


def polynomial_derivative(
    polynomial: Mapping[Exponent, Any], variable: int
) -> Polynomial:
    result: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        power = exponent[variable]
        if power == 0:
            continue
        derived = list(exponent)
        derived[variable] -= 1
        key = tuple(derived)
        result[key] = result.get(key, 0) + coefficient * power
    return result


def compose_rhs(
    state_polynomials: Sequence[Mapping[Exponent, Any]],
    rhs: Sequence[Mapping[str, Any]],
    *,
    exact: bool = False,
    variables: int | None = None,
) -> list[Polynomial]:
    if not state_polynomials:
        return []
    if variables is None:
        variables = next(
            (
                len(exponent)
                for polynomial in state_polynomials
                for exponent in polynomial
            ),
            1,
        )
    outputs: list[Polynomial] = []
    for expression in rhs:
        value: Polynomial = {}
        for term in expression["terms"]:
            coefficient: Any = (
                Fraction(str(term["coefficient"]))
                if exact
                else float(term["coefficient"])
            )
            product: Polynomial = {(0,) * variables: coefficient}
            for state, power in zip(state_polynomials, term["powers"]):
                product = polynomial_mul(
                    product, polynomial_pow(state, int(power))
                )
            value = polynomial_add(value, product)
        outputs.append(value)
    return outputs


def polynomial_defect(
    state_polynomials: Sequence[Mapping[Exponent, Any]],
    rhs: Sequence[Mapping[str, Any]],
    time_index: int,
    *,
    exact: bool = False,
) -> list[Polynomial]:
    derivative = [
        polynomial_derivative(polynomial, time_index)
        for polynomial in state_polynomials
    ]
    variables = next(
        (
            len(exponent)
            for polynomial in state_polynomials
            for exponent in polynomial
        ),
        time_index + 1,
    )
    vector_field = compose_rhs(
        state_polynomials, rhs, exact=exact, variables=variables
    )
    return [
        polynomial_add(left, right, right_scale=-1)
        for left, right in zip(derivative, vector_field)
    ]


def _terms(polynomial: Mapping[Exponent, Any]) -> list[dict[str, Any]]:
    return [
        {
            "exponents": list(exponent),
            "coefficient": float(coefficient),
        }
        for exponent, coefficient in sorted(polynomial.items())
    ]


def _interval_abs_max(interval: Sequence[float]) -> float:
    return max(abs(float(interval[0])), abs(float(interval[1])))


def jacobian_infinity_bound(
    system: Mapping[str, Any], box: Sequence[Sequence[float]]
) -> float:
    row_bounds: list[float] = []
    dimension = len(system["state_names"])
    for expression in system["rhs"]:
        row_total = 0.0
        for variable in range(dimension):
            derivative_terms: list[dict[str, Any]] = []
            for term in expression["terms"]:
                power = int(term["powers"][variable])
                if power == 0:
                    continue
                powers = list(map(int, term["powers"]))
                powers[variable] -= 1
                derivative_terms.append(
                    {
                        "coefficient": float(term["coefficient"]) * power,
                        "exponents": powers,
                    }
                )
            interval = evaluate_polynomial_interval(
                derivative_terms, box
            )
            row_total = math.nextafter(
                row_total + _interval_abs_max(interval), math.inf
            )
        row_bounds.append(row_total)
    return max(row_bounds, default=0.0)


def comparison_radius(
    defect_norm: float,
    jacobian_bound: float,
    h: float,
    initial_mismatch: float,
) -> float:
    if jacobian_bound == 0.0:
        value = initial_mismatch + defect_norm * h
    else:
        growth = math.exp(jacobian_bound * h)
        value = (
            growth * initial_mismatch
            + defect_norm * (growth - 1.0) / jacobian_bound
        )
    return math.nextafter(value, math.inf)


def diagnose_record(
    record: Mapping[str, Any], system: Mapping[str, Any]
) -> list[dict[str, Any]]:
    time_index = int(record["local_time_index"])
    polynomials = [
        polynomial_from_terms(state["polynomial_terms"])
        for state in record["states"]
    ]
    defects = polynomial_defect(
        polynomials, system["rhs"], time_index
    )
    domains = record["domains"]
    ranges = [
        evaluate_polynomial_interval(_terms(polynomial), domains)
        for polynomial in defects
    ]
    defect_norm = max(map(_interval_abs_max, ranges), default=0.0)
    jacobian = jacobian_infinity_bound(
        system, record["whole_tube_box"]
    )
    native_radii = [
        _interval_abs_max(state["independent_interval_remainder"])
        for state in record["states"]
    ]
    initial_mismatch = max(native_radii, default=0.0)
    radius = comparison_radius(
        defect_norm,
        jacobian,
        float(record["h"]),
        initial_mismatch,
    )
    rows: list[dict[str, Any]] = []
    for state_index, (polynomial, interval, native_radius) in enumerate(
        zip(defects, ranges, native_radii)
    ):
        rows.append(
            {
                "tool": record["tool"],
                "variant": record["variant"],
                "protocol": "one_step_common_defect",
                "system": record["system"],
                "h": record["h"],
                "state_index": state_index,
                "defect_lower": interval[0],
                "defect_upper": interval[1],
                "component_defect_bound": _interval_abs_max(interval),
                "defect_norm_inf": defect_norm,
                "jacobian_comparison_bound_inf": jacobian,
                "initial_mismatch_or_native_remainder_radius": (
                    initial_mismatch
                ),
                "common_certified_radius": radius,
                "native_certified_radius": native_radius,
                "common_certificate_available": True,
                "defect_term_count": len(polynomial),
                "defect_terms": json.dumps(
                    _terms(polynomial), sort_keys=True
                ),
                "native_validation_passed": record[
                    "native_validation_passed"
                ],
                "directed_rounding_or_mpfr": record[
                    "native_metadata"
                ].get("directed_rounding_or_mpfr", ""),
                "floating_point_enclosure_candidate": record[
                    "native_metadata"
                ].get("floating_point_enclosure_candidate", ""),
                "certificate_note": (
                    "Gronwall comparison for exported polynomial core; "
                    "initial mismatch conservatively uses the largest "
                    "exposed local independent-remainder magnitude."
                ),
            }
        )
    return rows


def run_diagnostics(
    spec: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    segment_dir = output / "common_segments"
    paths = sorted(segment_dir.glob("*.json"))
    if not paths:
        raise RuntimeError(
            f"no common segment records found under {segment_dir}"
        )
    unique: dict[tuple[Any, ...], tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (
            record["tool"],
            record["variant"],
            record["system"],
            float(record["h"]),
            record.get("native_metadata", {}).get("requested_order"),
        )
        unique[key] = (path, record)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path, record in unique.values():
        try:
            rows.extend(
                diagnose_record(
                    record, spec["systems"][record["system"]]
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "failure_category": "defect_diagnostic_failure",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    write_csv(output / "defect_summary.csv", rows)
    write_json(output / "defect_failures.json", failures)
    summary = {
        "segment_records": len(unique),
        "duplicate_superseded_records": len(paths) - len(unique),
        "component_rows": len(rows),
        "failures": len(failures),
        "passed": not failures,
    }
    write_json(output / "defect_checks.json", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} defect diagnostics failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", default=str(REPO_ROOT / "benchmarks" / "canonical.yaml")
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    print(
        json.dumps(
            run_diagnostics(spec, output), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
