#!/usr/bin/env python3
"""Independent exact-Fraction oracle cases for the outward primitives."""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any

import torch

from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    fixed_support_kernel_plan,
)
from torch_tm_flowpipe.fixed_support_outward import (
    OutwardFixedSupportPolynomial,
    OutwardIntervalTensor,
    outward_matmul,
)


def _q(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _contains(interval: OutwardIntervalTensor, exact: Fraction, index=()) -> bool:
    return _q(interval.lo[index].item()) <= exact <= _q(interval.hi[index].item())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases: list[dict[str, Any]] = []

    def record(name: str, passed: bool, scope: str) -> None:
        cases.append({"case": name, "passed": bool(passed), "scope": scope})

    # Constant, affine, asymmetric, subnormal, and large-finite interval products.
    pairs = {
        "constant": ((1.0, 1.0), (2.0, 2.0)),
        "asymmetric": ((-3.0, 1.25), (-0.5, 7.0)),
        "subnormal": ((torch.nextafter(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)).item(),) * 2, (2.0, 2.0)),
        "large_finite": ((1.0e150, 1.1e150), (-2.0e150, 3.0e150)),
    }
    for name, (left, right) in pairs.items():
        actual = OutwardIntervalTensor(
            torch.tensor(left[0], dtype=torch.float64), torch.tensor(left[1], dtype=torch.float64)
        ).mul(OutwardIntervalTensor(
            torch.tensor(right[0], dtype=torch.float64), torch.tensor(right[1], dtype=torch.float64)
        ))
        exact = [_q(x) * _q(y) for x, y in product(left, right)]
        record(name, _q(actual.lo.item()) <= min(exact) and _q(actual.hi.item()) >= max(exact), "interval multiplication")

    left_values = [[1.0e16, 1.0, -1.0e16], [1.0 / 3.0, -2.0, 7.0]]
    right_values = [[1.0, 2.0], [1.0, -3.0], [1.0, 4.0]]
    matrix = outward_matmul(
        OutwardIntervalTensor.point(torch.tensor([left_values], dtype=torch.float64)),
        OutwardIntervalTensor.point(torch.tensor([right_values], dtype=torch.float64)),
    )
    cancellation_ok = True
    for row in range(2):
        for column in range(2):
            exact = sum(_q(left_values[row][inner]) * _q(right_values[inner][column]) for inner in range(3))
            cancellation_ok &= _contains(matrix, exact, (0, row, column))
    record("cancellation_sequential_sum", cancellation_ok, "matrix reduction")

    descriptor = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    plan = fixed_support_kernel_plan(descriptor, device="cpu", dtype=torch.float64)
    left_coeff = [1.0 / 3.0, -2.0, 0.5, 3.0, -0.25, 0.75, -1.5]
    right_coeff = [-0.2, 4.0, -3.0, 0.125, 2.0, -0.5, 0.6]
    box = OutwardIntervalTensor(
        torch.tensor([[0.0, -0.75, -0.2]], dtype=torch.float64),
        torch.tensor([[0.01, 0.5, 0.9]], dtype=torch.float64),
    )
    retained, overflow = OutwardFixedSupportPolynomial.point(
        torch.tensor([[left_coeff]], dtype=torch.float64), plan
    ).multiply_project(
        OutwardFixedSupportPolynomial.point(torch.tensor([[right_coeff]], dtype=torch.float64), plan), box
    )
    duplicate_ok = True
    for output_slot in range(plan.num_slots):
        exact = Fraction(0)
        for left_slot, right_slot, route_output, sign in plan.multiply_route_indices:
            if route_output == output_slot:
                exact += _q(left_coeff[left_slot]) * _q(right_coeff[right_slot]) * sign
        duplicate_ok &= _contains(retained.coefficients, exact, (0, 0, output_slot))
    record("duplicate_routes", duplicate_ok and bool(torch.all(overflow.lo <= overflow.hi)), "projected polynomial product")

    # Analytic polynomial systems are evaluated from exact corner formulas, not Torch operators.
    rational_boxes = [
        ((Fraction(11, 10), Fraction(7, 5)), (Fraction(47, 20), Fraction(49, 20))),
        ((Fraction(-3, 2), Fraction(-1, 1)), (Fraction(1, 4), Fraction(3, 4))),
        ((Fraction(-1, 5), Fraction(2, 5)), (Fraction(-7, 4), Fraction(-5, 4))),
    ]
    for ordinal, ((xlo, xhi), (ylo, yhi)) in enumerate(rational_boxes):
        x = OutwardIntervalTensor(
            torch.tensor(float(xlo), dtype=torch.float64),
            torch.tensor(float(xhi), dtype=torch.float64),
        )
        y = OutwardIntervalTensor(
            torch.tensor(float(ylo), dtype=torch.float64),
            torch.tensor(float(yhi), dtype=torch.float64),
        )
        one = OutwardIntervalTensor.point(torch.tensor(1.0, dtype=torch.float64))
        vdp_y = one.sub(x.mul(x)).mul(y).sub(x)
        exact = []
        for yv in (ylo, yhi):
            x_candidates = [xlo, xhi]
            if yv:
                critical = -Fraction(1, 2) / yv
                if xlo <= critical <= xhi:
                    x_candidates.append(critical)
            exact.extend((Fraction(1) - xv * xv) * yv - xv for xv in x_candidates)
        record(
            f"vdp_rational_box_{ordinal}",
            _q(vdp_y.lo.item()) <= min(exact) and _q(vdp_y.hi.item()) >= max(exact),
            "VDP polynomial extrema oracle",
        )

    qlo, qhi = Fraction(-3, 4), Fraction(5, 4)
    q = OutwardIntervalTensor(
        torch.tensor(float(qlo), dtype=torch.float64),
        torch.tensor(float(qhi), dtype=torch.float64),
    )
    q2 = q.mul(q)
    exact_q2 = [value * value for value in (qlo, qhi)] + [Fraction(0)]
    record("scalar_quadratic_riccati", _q(q2.lo.item()) <= min(exact_q2) and _q(q2.hi.item()) >= max(exact_q2), "quadratic polynomial")

    hx, hy = Fraction(-2, 3), Fraction(7, 5)
    harmonic = outward_matmul(
        OutwardIntervalTensor.point(torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]], dtype=torch.float64)),
        OutwardIntervalTensor.point(torch.tensor([[[float(hx)], [float(hy)]]], dtype=torch.float64)),
    )
    record("harmonic_oscillator_affine", _contains(harmonic, hy, (0, 0, 0)) and _contains(harmonic, -hx, (0, 1, 0)), "affine matrix map")

    result = {
        "schema": "fixed_support_outward_fraction_oracle_v1",
        "oracle_independence": "expected values are computed with fractions.Fraction and explicit formulas; expected paths do not call Torch operators under test",
        "cases": cases,
        "case_count": len(cases),
        "all_passed": all(row["passed"] for row in cases),
        "numerical_soundness_class": "safeguarded outward under declared IEEE/backend assumptions",
        "numerical_soundness_scope": "primitive",
    }
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
