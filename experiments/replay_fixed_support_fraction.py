#!/usr/bin/env python3
"""Exact-rational one-step replay for the Torch/DiffReach DR7 workload.

The oracle below is deliberately independent of the tensor implementation. It
uses :class:`fractions.Fraction` for exact retained coefficients and exact
interval endpoint arithmetic for the rational VDP fixture.  It reports both
direct containment by ordinary binary64 outputs and the minimum outward ULP
expansion needed when direct containment is absent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch

from torch_tm_flowpipe.fixed_support import (
    FixedSupportDescriptor,
    FixedSupportInterval,
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


Q = Fraction


def q(value: str | int) -> Q:
    return Q(str(value))


@dataclass(frozen=True)
class QInterval:
    lo: Q
    hi: Q

    def add(self, other: "QInterval") -> "QInterval":
        return QInterval(self.lo + other.lo, self.hi + other.hi)

    def sub(self, other: "QInterval") -> "QInterval":
        return QInterval(self.lo - other.hi, self.hi - other.lo)

    def scale(self, value: Q) -> "QInterval":
        products = (self.lo * value, self.hi * value)
        return QInterval(min(products), max(products))

    def mul(self, other: "QInterval") -> "QInterval":
        products = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        return QInterval(min(products), max(products))

    def square(self) -> "QInterval":
        if self.lo <= 0 <= self.hi:
            return QInterval(Q(0), max(self.lo * self.lo, self.hi * self.hi))
        return QInterval(min(self.lo * self.lo, self.hi * self.hi), max(self.lo * self.lo, self.hi * self.hi))


ZERO_IV = QInterval(Q(0), Q(0))


def qsum(intervals: Iterable[QInterval]) -> QInterval:
    result = ZERO_IV
    for interval in intervals:
        result = result.add(interval)
    return result


def linear_form(coefficients: Sequence[Q], box: Sequence[QInterval]) -> QInterval:
    return qsum(QInterval(value, value).mul(interval) for value, interval in zip(coefficients, box))


@dataclass(frozen=True)
class QPoly:
    c: tuple[Q, ...]
    linear: tuple[tuple[Q, ...], ...]
    cross: tuple[tuple[Q, ...], ...]

    @property
    def output_dim(self) -> int:
        return len(self.c)

    def component(self, index: int) -> "QPoly":
        return QPoly((self.c[index],), (self.linear[index],), (self.cross[index],))

    @classmethod
    def stack(cls, values: Sequence["QPoly"]) -> "QPoly":
        return cls(
            tuple(value.c[0] for value in values),
            tuple(value.linear[0] for value in values),
            tuple(value.cross[0] for value in values),
        )

    def add(self, other: "QPoly") -> "QPoly":
        return QPoly(
            tuple(a + b for a, b in zip(self.c, other.c)),
            tuple(tuple(a + b for a, b in zip(left, right)) for left, right in zip(self.linear, other.linear)),
            tuple(tuple(a + b for a, b in zip(left, right)) for left, right in zip(self.cross, other.cross)),
        )

    def sub(self, other: "QPoly") -> "QPoly":
        return QPoly(
            tuple(a - b for a, b in zip(self.c, other.c)),
            tuple(tuple(a - b for a, b in zip(left, right)) for left, right in zip(self.linear, other.linear)),
            tuple(tuple(a - b for a, b in zip(left, right)) for left, right in zip(self.cross, other.cross)),
        )

    def mul_trunc(self, other: "QPoly") -> "QPoly":
        constants: list[Q] = []
        linear_out: list[tuple[Q, ...]] = []
        cross_out: list[tuple[Q, ...]] = []
        for output in range(self.output_dim):
            c1, c2 = self.c[output], other.c[output]
            l1, l2 = self.linear[output], other.linear[output]
            lt1, lt2 = self.cross[output], other.cross[output]
            constants.append(c1 * c2)
            linear_out.append(tuple(c1 * b + c2 * a for a, b in zip(l1, l2)))
            cross_values = [
                c1 * lt2[index]
                + c2 * lt1[index]
                + l1[0] * l2[index]
                + l2[0] * l1[index]
                for index in range(3)
            ]
            cross_values[0] -= l1[0] * l2[0]
            cross_out.append(tuple(cross_values))
        return QPoly(tuple(constants), tuple(linear_out), tuple(cross_out))

    def range(self, box: Sequence[QInterval]) -> tuple[QInterval, ...]:
        time = box[0]
        output: list[QInterval] = []
        for component in range(self.output_dim):
            linear_x = linear_form(self.linear[component][1:], box[1:])
            cross_x = linear_form(self.cross[component][1:], box[1:])
            inner = QInterval(self.cross[component][0], self.cross[component][0]).mul(time).add(
                QInterval(self.linear[component][0], self.linear[component][0]).add(cross_x)
            )
            base = QInterval(self.c[component], self.c[component]).add(linear_x)
            output.append(time.mul(inner).add(base))
        return tuple(output)

    def mul_ctrunc(
        self, other: "QPoly", box: Sequence[QInterval]
    ) -> tuple["QPoly", dict[str, tuple[QInterval, ...]]]:
        kept = self.mul_trunc(other)
        pure: list[QInterval] = []
        cubic: list[QInterval] = []
        quartic: list[QInterval] = []
        time = box[0]
        for component in range(self.output_dim):
            spatial_terms: list[QInterval] = []
            for left_index in range(1, 3):
                for right_index in range(1, 3):
                    if left_index == right_index:
                        monomial = box[left_index].square()
                    else:
                        monomial = box[left_index].mul(box[right_index])
                    spatial_terms.append(
                        monomial.scale(
                            self.linear[component][left_index]
                            * other.linear[component][right_index]
                        )
                    )
            pure.append(qsum(spatial_terms))
            left_linear = linear_form(self.linear[component], box)
            right_linear = linear_form(other.linear[component], box)
            left_cross = linear_form(self.cross[component], box)
            right_cross = linear_form(other.cross[component], box)
            cubic.append(
                time.mul(left_linear.mul(right_cross)).add(
                    time.mul(right_linear.mul(left_cross))
                )
            )
            quartic.append(time.mul(time).mul(left_cross.mul(right_cross)))
        return kept, {
            "pure_spatial_quadratic": tuple(pure),
            "time_cubic": tuple(cubic),
            "time_quartic": tuple(quartic),
        }

    def integrate_trunc(self) -> "QPoly":
        return QPoly(
            tuple(Q(0) for _ in self.c),
            tuple((self.c[index], Q(0), Q(0)) for index in range(self.output_dim)),
            tuple(
                (
                    self.linear[index][0] / 2,
                    self.linear[index][1],
                    self.linear[index][2],
                )
                for index in range(self.output_dim)
            ),
        )

    def integrate_ctrunc(
        self, box: Sequence[QInterval]
    ) -> tuple["QPoly", dict[str, tuple[QInterval, ...]]]:
        kept = self.integrate_trunc()
        time_cube = QInterval(box[0].lo**3, box[0].hi**3)
        time_square = box[0].square()
        cubics: list[QInterval] = []
        squared_spatial: list[QInterval] = []
        for component in range(self.output_dim):
            cubics.append(time_cube.scale(self.cross[component][0] / 3))
            squared_spatial.append(
                qsum(
                    time_square.mul(box[spatial]).scale(self.cross[component][spatial] / 2)
                    for spatial in range(1, 3)
                )
            )
        return kept, {
            "integration_time_cubic": tuple(cubics),
            "integration_time_squared_spatial": tuple(squared_spatial),
        }


@dataclass(frozen=True)
class QTM:
    polynomial: QPoly
    remainder: tuple[QInterval, ...]

    def component(self, index: int) -> "QTM":
        return QTM(self.polynomial.component(index), (self.remainder[index],))

    @classmethod
    def stack(cls, values: Sequence["QTM"]) -> "QTM":
        return cls(QPoly.stack([value.polynomial for value in values]), tuple(value.remainder[0] for value in values))

    def add(self, other: "QTM") -> "QTM":
        return QTM(self.polynomial.add(other.polynomial), tuple(a.add(b) for a, b in zip(self.remainder, other.remainder)))

    def sub(self, other: "QTM") -> "QTM":
        return QTM(self.polynomial.sub(other.polynomial), tuple(a.sub(b) for a, b in zip(self.remainder, other.remainder)))

    def range(self, box: Sequence[QInterval]) -> tuple[QInterval, ...]:
        return tuple(poly.add(rem) for poly, rem in zip(self.polynomial.range(box), self.remainder))

    def mul(self, other: "QTM", box: Sequence[QInterval]) -> tuple["QTM", dict[str, tuple[QInterval, ...]]]:
        kept, ledger = self.polynomial.mul_ctrunc(other.polynomial, box)
        left_range = self.polynomial.range(box)
        right_range = other.polynomial.range(box)
        remainder: list[QInterval] = []
        for component in range(self.polynomial.output_dim):
            polynomial_overflow = qsum(values[component] for values in ledger.values())
            remainder.append(
                left_range[component]
                .mul(other.remainder[component])
                .add(right_range[component].mul(self.remainder[component]))
                .add(self.remainder[component].mul(other.remainder[component]))
                .add(polynomial_overflow)
            )
        return QTM(kept, tuple(remainder)), ledger

    def integrate(self, box: Sequence[QInterval]) -> tuple["QTM", dict[str, tuple[QInterval, ...]]]:
        kept, ledger = self.polynomial.integrate_ctrunc(box)
        time_magnitude = max(abs(box[0].lo), abs(box[0].hi))
        remainder = tuple(
            qsum(values[index] for values in ledger.values()).add(self.remainder[index].scale(time_magnitude))
            for index in range(self.polynomial.output_dim)
        )
        return QTM(kept, remainder), ledger


def qconstant_like(reference: QPoly | QTM, value: Q):
    polynomial = reference.polynomial if isinstance(reference, QTM) else reference
    constant = QPoly(
        tuple(value for _ in polynomial.c),
        tuple((Q(0), Q(0), Q(0)) for _ in polynomial.c),
        tuple((Q(0), Q(0), Q(0)) for _ in polynomial.c),
    )
    return QTM(constant, tuple(ZERO_IV for _ in polynomial.c)) if isinstance(reference, QTM) else constant


def qpoly_rhs(state: QPoly) -> QPoly:
    x, y = state.component(0), state.component(1)
    second = qconstant_like(x, Q(1)).sub(x.mul_trunc(x)).mul_trunc(y).sub(x)
    return QPoly.stack((y, second))


def qtm_rhs(state: QTM, box: Sequence[QInterval]) -> QTM:
    x, y = state.component(0), state.component(1)
    square, _ = x.mul(x, box)
    first_product, _ = qconstant_like(x, Q(1)).sub(square).mul(y, box)
    second = first_product.sub(x)
    return QTM.stack((y, second))


def q_fixture() -> dict[str, object]:
    box = (QInterval(Q(0), q("0.01")), QInterval(Q(-1), Q(1)), QInterval(Q(-1), Q(1)))
    base = QPoly(
        (q("1.25"), q("2.4")),
        ((Q(0), q("0.15"), Q(0)), (Q(0), Q(0), q("0.05"))),
        ((Q(0), Q(0), Q(0)), (Q(0), Q(0), Q(0))),
    )
    poly1 = base.add(qpoly_rhs(base).integrate_trunc())
    poly2 = base.add(qpoly_rhs(poly1).integrate_trunc())
    new_x0 = QTM(base, (ZERO_IV, ZERO_IV))
    seed = QTM(poly2, (QInterval(q("-0.01"), q("0.01")),) * 2)
    initial_integral, integration_ledger = qtm_rhs(seed, box).integrate(box)
    initial = new_x0.add(initial_integral)
    initial_mask = tuple(
        candidate.lo >= prior.lo and candidate.hi <= prior.hi
        for candidate, prior in zip(initial.remainder, seed.remainder)
    )
    difference = initial.polynomial.sub(seed.polynomial).range(box)
    current = seed
    round_masks: list[tuple[bool, ...]] = []
    for _ in range(10):
        integral, _ = qtm_rhs(current, box).integrate(box)
        candidate = new_x0.add(integral)
        next_remainder = tuple(a.add(b) for a, b in zip(candidate.remainder, difference))
        mask = tuple(
            nxt.lo >= prior.lo and nxt.hi <= prior.hi
            for nxt, prior in zip(next_remainder, current.remainder)
        )
        accepted = tuple(nxt if keep else prior for nxt, prior, keep in zip(next_remainder, current.remainder, mask))
        current = QTM(candidate.polynomial, accepted)
        round_masks.append(mask)

    # First-step normalized parameterization.  The exact epsilon is part of
    # the pinned carry contract and all values remain rational.
    epsilon = q("0.000000000001")
    normalized_scale = (q("0.15") / (q("0.15") + epsilon), q("0.05") / (q("0.05") + epsilon))
    composed_linear: list[tuple[Q, ...]] = []
    composed_cross: list[tuple[Q, ...]] = []
    composed_constant: list[Q] = []
    for component in range(2):
        composed_constant.append(current.polynomial.c[component])
        composed_linear.append(
            (
                current.polynomial.linear[component][0],
                current.polynomial.linear[component][1] * normalized_scale[0],
                current.polynomial.linear[component][2] * normalized_scale[1],
            )
        )
        composed_cross.append(
            (
                current.polynomial.cross[component][0],
                current.polynomial.cross[component][1] * normalized_scale[0],
                current.polynomial.cross[component][2] * normalized_scale[1],
            )
        )
    composed = QTM(
        QPoly(tuple(composed_constant), tuple(composed_linear), tuple(composed_cross)),
        current.remainder,
    )
    endpoint_box = (QInterval(q("0.01"), q("0.01")), *box[1:])

    x_seed = seed.component(0)
    exact_square, multiplication_ledger = x_seed.mul(x_seed, box)
    del exact_square
    return {
        "box": box,
        "poly1": poly1,
        "poly2": poly2,
        "initial_mask": initial_mask,
        "round_masks": tuple(round_masks),
        "final": current,
        "tube": composed.range(box),
        "endpoint": composed.range(endpoint_box),
        "multiplication_ledger": multiplication_ledger,
        "integration_ledger": integration_ledger,
    }


def float_fraction(value: float) -> Q:
    return Q.from_float(float(value))


def ulps_to_cover_lower(value: float, exact: Q, limit: int = 4096) -> int | None:
    current = float(value)
    for count in range(limit + 1):
        if float_fraction(current) <= exact:
            return count
        current = math.nextafter(current, -math.inf)
    return None


def ulps_to_cover_upper(value: float, exact: Q, limit: int = 4096) -> int | None:
    current = float(value)
    for count in range(limit + 1):
        if float_fraction(current) >= exact:
            return count
        current = math.nextafter(current, math.inf)
    return None


def compare_point(name: str, observed: torch.Tensor, exact: Sequence[Sequence[Q]]) -> dict[str, object]:
    exact_flat = [value for row in exact for value in row]
    observed_flat = observed.detach().cpu().reshape(-1).tolist()
    ulps = [
        max(ulps_to_cover_lower(value, target) or 0, ulps_to_cover_upper(value, target) or 0)
        for value, target in zip(observed_flat, exact_flat)
    ]
    return {
        "name": name,
        "values": len(ulps),
        "direct_exact": all(value == 0 for value in ulps),
        "max_outward_ulps_needed": max(ulps, default=0),
    }


def compare_intervals(
    name: str,
    observed_lo: torch.Tensor,
    observed_hi: torch.Tensor,
    exact: Sequence[QInterval],
) -> dict[str, object]:
    lo_values = observed_lo.detach().cpu().reshape(-1).tolist()
    hi_values = observed_hi.detach().cpu().reshape(-1).tolist()
    exact_values = list(exact)
    lower_ulps = [ulps_to_cover_lower(value, target.lo) for value, target in zip(lo_values, exact_values)]
    upper_ulps = [ulps_to_cover_upper(value, target.hi) for value, target in zip(hi_values, exact_values)]
    if any(value is None for value in (*lower_ulps, *upper_ulps)):
        max_ulps: int | None = None
    else:
        max_ulps = max([int(value) for value in (*lower_ulps, *upper_ulps)], default=0)
    return {
        "name": name,
        "components": len(exact_values),
        "direct_containment": max_ulps == 0,
        "max_outward_ulps_needed": max_ulps,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = torch.float64
    support = FixedSupportDescriptor.diffreach_restricted_quadratic(2)
    box_lo, box_hi, eval_lo, eval_hi = fixed_support_step_boxes(
        1, 2, 0.01, dtype=dtype, device=device
    )
    center = torch.tensor([[1.25, 2.4]], dtype=dtype, device=device)
    scale = torch.tensor([[0.15, 0.05]], dtype=dtype, device=device)
    base = fixed_support_build_linear_tm(center, scale, support)
    poly2, polynomial_rounds = fixed_support_polynomial_picard(
        base.polynomial, diffreach_vdp_polynomial_rhs, box_lo, box_hi, iterations=2
    )
    seed = FixedSupportTaylorModel(
        poly2,
        FixedSupportInterval(
            torch.full((1, 2), -0.01, dtype=dtype, device=device),
            torch.full((1, 2), 0.01, dtype=dtype, device=device),
        ),
    )
    dr = fixed_support_dr_remainder_picard(
        diffreach_vdp_tm_rhs, base, seed, box_lo, box_hi, rounds=10
    )
    identity = fixed_support_identity_parameterization(
        1, 2, support, dtype=dtype, device=device
    )
    symbolic = fixed_support_symbolic_step_linear(
        identity, base.evaluate_time(0.01),
        FixedSupportSymbolicRemainderState.initialize(1, 2, 1, dtype=dtype, device=device),
        eval_lo,
        eval_hi,
    )
    composed = dr.model.compose_affine(symbolic.normalized_parameterization, 0.01)
    endpoint_lo = box_lo.clone()
    endpoint_lo[:, 0] = 0.01
    observed_tube = composed.range(box_lo, box_hi)
    observed_endpoint = composed.range(endpoint_lo, box_hi)
    square = seed.component(0).mul(seed.component(0), box_lo, box_hi)
    rhs_integrated = diffreach_vdp_tm_rhs(seed, box_lo, box_hi).integrate_time(box_lo, box_hi)

    oracle = q_fixture()
    exact_poly1: QPoly = oracle["poly1"]  # type: ignore[assignment]
    exact_poly2: QPoly = oracle["poly2"]  # type: ignore[assignment]
    exact_final: QTM = oracle["final"]  # type: ignore[assignment]
    checks: list[dict[str, object]] = []
    for name, observed, exact_poly in (
        ("retained_coefficients_poly1", polynomial_rounds[0].coeffs[0], exact_poly1),
        ("retained_coefficients_poly2", polynomial_rounds[1].coeffs[0], exact_poly2),
        ("retained_coefficients_final", dr.model.polynomial.coeffs[0], exact_final.polynomial),
    ):
        exact_slots = [
            [exact_poly.c[index], *exact_poly.linear[index], *exact_poly.cross[index]]
            for index in range(exact_poly.output_dim)
        ]
        checks.append(compare_point(name, observed, exact_slots))

    exact_multiplication: dict[str, tuple[QInterval, ...]] = oracle["multiplication_ledger"]  # type: ignore[assignment]
    for ledger_name, exact_values in exact_multiplication.items():
        observed = square.ledger.as_dict()[ledger_name]
        checks.append(compare_intervals(f"multiplication_overflow.{ledger_name}", observed.lo[0], observed.hi[0], exact_values))
    exact_integration: dict[str, tuple[QInterval, ...]] = oracle["integration_ledger"]  # type: ignore[assignment]
    for ledger_name, exact_values in exact_integration.items():
        observed = rhs_integrated.ledger.as_dict()[ledger_name]
        checks.append(compare_intervals(f"integration_overflow.{ledger_name}", observed.lo[0], observed.hi[0], exact_values))

    checks.append(compare_intervals("final_accepted_remainder", dr.model.remainder.lo[0], dr.model.remainder.hi[0], exact_final.remainder))
    checks.append(compare_intervals("full_step_tube", observed_tube.lo[0], observed_tube.hi[0], oracle["tube"]))  # type: ignore[arg-type]
    checks.append(compare_intervals("endpoint", observed_endpoint.lo[0], observed_endpoint.hi[0], oracle["endpoint"]))  # type: ignore[arg-type]
    initial_mask_equal = dr.initial_inclusion_mask[0].detach().cpu().tolist() == list(oracle["initial_mask"])
    round_masks_equal = dr.round_inclusion_masks[:, 0, :].detach().cpu().tolist() == [list(row) for row in oracle["round_masks"]]
    max_ulps = max(
        (int(check["max_outward_ulps_needed"]) for check in checks if check["max_outward_ulps_needed"] is not None),
        default=0,
    )
    direct = all(
        bool(check.get("direct_containment", check.get("direct_exact", False))) for check in checks
    )
    report = {
        "schema": "torch_tm_flowpipe_fixed_support_fraction_replay_v1",
        "arithmetic": "exact fractions.Fraction rational interval endpoints",
        "device": str(device),
        "support_sha256": support.support_sha256,
        "initial_inclusion_mask_equal": initial_mask_equal,
        "all_round_masks_equal": round_masks_equal,
        "checks": checks,
        "all_directly_contained": direct and initial_mask_equal and round_masks_equal,
        "max_outward_ulps_needed": max_ulps,
        "ordinary_binary64_directly_qualified": direct and initial_mask_equal and round_masks_equal,
        "replay_envelope_qualified": initial_mask_equal and round_masks_equal and max_ulps <= 8,
        "ordinary_lane_classification": "empirically sampled only",
        "replay_envelope_classification": "independently outward replayed for exact benchmark workload",
        "replay_envelope_rule": "expand every observed retained point and interval endpoint outward by the reported bounded ULP count",
        "scope": "one exact rational VDP segment; not universal GPU directed rounding",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(report, sort_keys=True))
    return 0 if report["replay_envelope_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
