#!/usr/bin/env python3
"""Implementation-independent exact-rational oracle for the fixed G2 contract.

This executable intentionally uses only the Python standard library.  It does
not import the project polynomial, Taylor-model, Picard, interval, or ledger
implementations.  Project code is treated as a black box through a JSON table.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Iterable


Exponent = tuple[int, ...]
Poly = dict[Exponent, Fraction]


def frac(text: str) -> Fraction:
    return Fraction(text)


def clean(poly: Poly) -> Poly:
    return {exp: value for exp, value in poly.items() if value}


def add(left: Poly, right: Poly) -> Poly:
    result = dict(left)
    for exponent, value in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + value
    return clean(result)


def mul(left: Poly, right: Poly) -> Poly:
    result: Poly = {}
    for left_exp, left_value in left.items():
        for right_exp, right_value in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
            result[exponent] = result.get(exponent, Fraction(0)) + left_value * right_value
    return clean(result)


def power(poly: Poly, exponent: int) -> Poly:
    n_vars = len(next(iter(poly)))
    result: Poly = {(0,) * n_vars: Fraction(1)}
    for _ in range(exponent):
        result = mul(result, poly)
    return result


def parse_fraction_table(rows: Iterable[list[object]]) -> Poly:
    result: Poly = {}
    for exponent, value in rows:
        key = tuple(int(item) for item in exponent)
        result[key] = result.get(key, Fraction(0)) + frac(str(value))
    return clean(result)


def parse_hex_table(rows: Iterable[list[object]]) -> Poly:
    result: Poly = {}
    for exponent, value in rows:
        key = tuple(int(item) for item in exponent)
        number = float.fromhex(str(value))
        exact = Fraction(*number.as_integer_ratio())
        result[key] = result.get(key, Fraction(0)) + exact
    return clean(result)


def exact_natural_interval(poly: Poly) -> tuple[Fraction, Fraction]:
    """Exact natural interval enclosure on [-1,1]^n."""

    lower = Fraction(0)
    upper = Fraction(0)
    for exponent, coefficient in poly.items():
        active = [power_value for power_value in exponent if power_value]
        if any(power_value % 2 for power_value in active):
            magnitude = abs(coefficient)
            term_lo, term_hi = -magnitude, magnitude
        elif active:
            term_lo, term_hi = sorted((Fraction(0), coefficient))
        else:
            term_lo = term_hi = coefficient
        lower += term_lo
        upper += term_hi
    return lower, upper


def canonical_rows(poly: Poly) -> list[list[object]]:
    return [[list(exp), f"{value.numerator}/{value.denominator}"] for exp, value in sorted(poly.items())]


def check(condition: bool, name: str, checks: list[dict[str, object]], detail: str = "") -> None:
    checks.append({"name": name, "passed": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def run(input_path: Path) -> dict[str, object]:
    raw_bytes = input_path.read_bytes()
    payload = json.loads(raw_bytes)
    if payload.get("schema") != "g2_project_blackbox_coefficients_v1":
        raise ValueError("black-box coefficient schema mismatch")
    cases = payload["cases"]
    checks: list[dict[str, object]] = []

    merge_case = cases["canonical_merge"]
    merge_expected = parse_fraction_table(merge_case["input_terms"])
    merge_observed = parse_hex_table(merge_case["observed"])
    check(merge_expected == merge_observed, "canonical_monomial_merge_exact", checks)

    affine = cases["affine_shared_column_x2y"]
    x = parse_fraction_table(affine["x"])
    y = parse_fraction_table(affine["y"])
    x2y_expected = mul(mul(x, x), y)
    x2y_observed = parse_hex_table(affine["observed"])
    check(x2y_expected == x2y_observed, "affine_shared_column_substitution_exact", checks)
    check(
        any(exp[1] == 3 for exp in x2y_expected),
        "x2y_cubic_shared_source_expansion_present",
        checks,
    )

    retirement = cases["oldest_current_retirement"]
    mixed = parse_hex_table(retirement["input"])
    oldest = tuple(int(index) for index in retirement["oldest_indices"])
    current = tuple(int(index) for index in retirement["current_indices"])
    retired_poly = {
        exp: value for exp, value in mixed.items() if any(exp[index] for index in oldest)
    }
    surviving = {
        exp: value for exp, value in mixed.items() if not any(exp[index] for index in oldest)
    }
    observed_surviving = parse_hex_table(retirement["retained_after_collapse"])
    check(surviving == observed_surviving, "oldest_partition_exact", checks)
    check(
        any(any(exp[index] for index in oldest) and any(exp[index] for index in current) for exp in retired_poly),
        "oldest_current_mixed_term_is_retired",
        checks,
    )
    exact_lo, exact_hi = exact_natural_interval(retired_poly)
    observed_lo = Fraction(*float.fromhex(retirement["collapsed_interval_hex"][0]).as_integer_ratio())
    observed_hi = Fraction(*float.fromhex(retirement["collapsed_interval_hex"][1]).as_integer_ratio())
    check(
        observed_lo <= exact_lo and observed_hi >= exact_hi,
        "independent_exact_rational_retirement_containment",
        checks,
        f"project=[{observed_lo},{observed_hi}], exact-natural=[{exact_lo},{exact_hi}]",
    )

    current_poly = {
        exp: value for exp, value in surviving.items() if any(exp[index] for index in current)
    }
    rotated_expected: Poly = {}
    for exponent, value in current_poly.items():
        updated = list(exponent)
        for offset in range(2):
            updated[2 + offset] = exponent[4 + offset]
            updated[4 + offset] = 0
        key = tuple(updated)
        rotated_expected[key] = rotated_expected.get(key, Fraction(0)) + value
    rotated_expected = clean(rotated_expected)
    rotated_observed = parse_hex_table(retirement["rotated_current"])
    check(rotated_expected == rotated_observed, "two_generation_bank_rotation_exact", checks)
    check(
        all(exp[4] == exp[5] == 0 for exp in rotated_observed),
        "fresh_bank_cleared_after_rotation",
        checks,
    )

    truncation = cases["degree4_truncation"]
    base_poly = parse_fraction_table(truncation["base"])
    expanded = power(base_poly, int(truncation["power"]))
    order = int(truncation["order"])
    kept_expected = {exp: value for exp, value in expanded.items() if sum(exp) <= order}
    dropped_expected = {exp: value for exp, value in expanded.items() if sum(exp) > order}
    check(
        kept_expected == parse_hex_table(truncation["kept"]),
        "degree4_retained_owner_exact",
        checks,
    )
    check(
        dropped_expected == parse_hex_table(truncation["dropped"]),
        "degree4_truncation_owner_exact",
        checks,
    )
    dropped_lo, dropped_hi = exact_natural_interval(dropped_expected)
    check(dropped_lo <= dropped_hi, "degree4_exact_rational_containment_enclosure", checks)

    retry = cases["retry_atomicity"]
    check(retry["accepted"] is False, "retry_fixture_is_rejected", checks)
    check(retry["object_identity_preserved"] is True, "rejected_retry_object_identity", checks)
    check(retry["before"] == retry["after"], "rejected_retry_canonical_state_immutable", checks)
    check(int(retry["proposed_generation"]) == 1, "uncommitted_successor_was_one_generation", checks)

    return {
        "schema": "independent_g2_exact_oracle_v1",
        "status": "PASS",
        "candidate": payload["candidate"],
        "implementation_independent": True,
        "imports_project_core": False,
        "arithmetic": "fractions.Fraction exact rational",
        "interval_proof": "exact natural rational enclosure on [-1,1]^n",
        "sampling_used": False,
        "blackbox_input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "checks_passed": len(checks),
        "checks": checks,
        "selected_exact_tables": {
            "x2y": canonical_rows(x2y_expected),
            "rotated_current": canonical_rows(rotated_expected),
            "degree4_dropped": canonical_rows(dropped_expected),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "checks_passed": result["checks_passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
