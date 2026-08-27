#!/usr/bin/env python3
"""Exact-Fraction oracle for one strict Huan symbolic-queue propagation step."""

from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


Interval = tuple[Fraction, Fraction]


def _fraction(value: float) -> Fraction:
    if not isinstance(value, float):
        value = float(value)
    return Fraction.from_float(value)


def _interval(value: list[float]) -> Interval:
    if len(value) != 2:
        raise ValueError("interval must have two endpoints")
    lo, hi = map(_fraction, value)
    if lo > hi:
        raise ValueError("inverted interval")
    return lo, hi


def _add(left: Interval, right: Interval) -> Interval:
    return left[0] + right[0], left[1] + right[1]


def _mul(left: Interval, right: Interval) -> Interval:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def _zero() -> Interval:
    return Fraction(0), Fraction(0)


def _matmul(left: list[list[Interval]], right: list[list[Interval]]) -> list[list[Interval]]:
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    if any(len(row) != inner for row in left):
        raise ValueError("left matrix shape mismatch")
    if any(len(row) != columns for row in right):
        raise ValueError("right matrix shape mismatch")
    return [
        [
            _sum(_mul(left[i][k], right[k][j]) for k in range(inner))
            for j in range(columns)
        ]
        for i in range(rows)
    ]


def _matvec(matrix: list[list[Interval]], vector: list[Interval]) -> list[Interval]:
    return [
        _sum(_mul(coefficient, value) for coefficient, value in zip(row, vector))
        for row in matrix
    ]


def _sum(values: Iterable[Interval]) -> Interval:
    result = _zero()
    for value in values:
        result = _add(result, value)
    return result


def _matrix(value: list[list[list[float]]]) -> list[list[Interval]]:
    return [[_interval(cell) for cell in row] for row in value]


def _vector(value: list[list[float]]) -> list[Interval]:
    return [_interval(cell) for cell in value]


def _contains(observed: list[float], exact: Interval) -> bool:
    outer = _interval(observed)
    return outer[0] <= exact[0] and exact[1] <= outer[1]


def _render(value: Interval) -> dict[str, str]:
    return {
        "lower": f"{value[0].numerator}/{value[0].denominator}",
        "upper": f"{value[1].numerator}/{value[1].denominator}",
    }


def _read_step(path: Path, step: int) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "causal_step" and row.get("step") == step:
                if "detail" not in row:
                    raise ValueError(f"step {step} lacks a detailed snapshot")
                return row
    raise ValueError(f"causal step {step} not found")


def verify(path: Path, step: int) -> dict[str, Any]:
    row = _read_step(path, step)
    detail = row["detail"]
    before = detail["queue_before"]
    after = detail["queue_after"]
    composition = detail["composition"]
    if row["mode"] != "strict":
        raise ValueError("Fraction queue oracle requires the strict lane")
    if before["generation"] != after["generation"]:
        raise ValueError("queue generation changed inside one advance")
    if before["live_count"] != before["phi_live_count"]:
        raise ValueError("queue entered advance with mismatched Phi/J owners")
    if after["live_count"] != before["live_count"] + 1:
        raise ValueError("queue did not append exactly one completed owner")
    if after["phi_live_count"] != after["live_count"]:
        raise ValueError("queue left advance with mismatched Phi/J owners")

    phi_i_raw = composition["scaled_linear_map_interval"]
    if phi_i_raw is None:
        raise ValueError("strict snapshot omitted scaled linear interval map")
    phi_i = _matrix(phi_i_raw[0])
    before_phi = [_matrix(item[0]) for item in before["phi_interval"]]
    after_phi_raw = after["phi_interval"]
    before_j = [_vector(item[0]) for item in before["j"]]
    after_j_raw = after["j"]
    qlen = before["live_count"]

    exact_phi = list(before_phi)
    for queue_index in range(1, qlen):
        exact_phi[queue_index] = _matmul(phi_i, before_phi[queue_index])
    exact_phi.append(phi_i)

    phi_checks = []
    for queue_index in range(1, qlen + 1):
        for i, exact_row in enumerate(exact_phi[queue_index]):
            for j, exact_value in enumerate(exact_row):
                observed = after_phi_raw[queue_index][0][i][j]
                phi_checks.append(
                    {
                        "queue_index": queue_index,
                        "row": i,
                        "column": j,
                        "contains": _contains(observed, exact_value),
                        "exact": _render(exact_value),
                        "observed": observed,
                    }
                )

    exact_linear = [_zero() for _ in range(len(phi_i))]
    for queue_index in range(qlen):
        contribution = _matvec(exact_phi[queue_index + 1], before_j[queue_index])
        exact_linear = [
            _add(total, value) for total, value in zip(exact_linear, contribution)
        ]
    observed_linear = composition["linear_history_interval_image"][0]
    linear_checks = [
        {
            "component": component,
            "contains": _contains(observed_linear[component], exact_value),
            "exact": _render(exact_value),
            "observed": observed_linear[component],
        }
        for component, exact_value in enumerate(exact_linear)
    ]

    prefix_unchanged = all(
        before["j"][index] == after_j_raw[index] for index in range(qlen)
    )
    appended_owner_exact = (
        after_j_raw[qlen]
        == composition["new_queue_column"]
    )
    passed = (
        all(check["contains"] for check in phi_checks)
        and all(check["contains"] for check in linear_checks)
        and prefix_unchanged
        and appended_owner_exact
    )
    return {
        "schema": "torch_tm_flowpipe.vdp_c3_huan_symbolic_queue_fraction/1",
        "input": str(path.resolve()),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "step": step,
        "mode": row["mode"],
        "queue_capacity": before["capacity"],
        "queue_generation": before["generation"],
        "live_count_before": qlen,
        "live_count_after": after["live_count"],
        "phi_outward_checks": phi_checks,
        "linear_history_outward_checks": linear_checks,
        "previous_j_owners_unchanged": prefix_unchanged,
        "new_j_owner_appended_exactly_once": appended_owner_exact,
        "double_count_check": "one Phi[q+1] times one J[q] for every q in the live prefix",
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-ledger", type=Path, required=True)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(args.causal_ledger.resolve(), args.step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": payload["passed"]}, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
