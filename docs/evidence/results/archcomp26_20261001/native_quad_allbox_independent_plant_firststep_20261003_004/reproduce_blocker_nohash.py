#!/usr/bin/env python3
"""Preserve the exact first-box Decimal rounding blocker; no solver or digest."""

import json
from decimal import Decimal as D, getcontext
from fractions import Fraction as F
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "tools"))
import check_native_quad_fullset_interval_gate_nohash as interval  # noqa: E402
import check_native_quad_algebraic_gate_nohash as algebra  # noqa: E402

RPC = HERE.parent / "native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json"


def main():
    record = json.loads(RPC.read_text())
    params, coeff = record["params"], record["coefficients"]
    pair = list(zip(params["input_lb"][:12], params["input_ub"][:12]))
    box = [(F.from_float(lo), F.from_float(hi)) for lo, hi in pair]
    row = [algebra.binary32(value) for value in coeff["T"][0][1]]
    low = algebra.binary32(coeff["u_min"][0][1])
    high = algebra.binary32(coeff["u_max"][0][1])
    center, radius = (high + low) / 2, (high - low) / 2
    exact = algebra.linear_range(row, box, center, radius)
    decimal_box = [interval.I(D.from_float(lo), D.from_float(hi)) for lo, hi in pair]
    computed = interval.I(D.from_float(float(center))) + interval.I(
        -D.from_float(float(radius)), D.from_float(float(radius)))
    for coefficient, input_interval in zip(row, decimal_box):
        computed += interval.I(D.from_float(float(coefficient))) * input_interval
    radius_decimal = D.from_float(float(radius))
    example = D.from_float(0.0014866764919133857)
    trig_radius = interval.trig_remainder(D.from_float(0.01), 7, 5040)
    result = {
        "schema": "native-quad-decimal-negation-first-blocker-nohash-v1",
        "saved_rpc_lane": 0,
        "control": "u2",
        "default_decimal_context_precision": getcontext().prec,
        "exact_control_lower": str(exact[0]),
        "computed_decimal_control_lower": str(computed.lo),
        "computed_minus_exact_lower": str(F(computed.lo) - exact[0]),
        "computed_minus_exact_lower_approx": float(F(computed.lo) - exact[0]),
        "lower_hull_omitted": F(computed.lo) > exact[0],
        "radius_exact_decimal": str(radius_decimal),
        "radius_unary_negation": str(-radius_decimal),
        "radius_copy_negate": str(radius_decimal.copy_negate()),
        "unary_negation_exact": F(-radius_decimal) == -F(radius_decimal),
        "example_exact_decimal": str(example),
        "example_unary_negation": str(-example),
        "example_copy_negate": str(example.copy_negate()),
        "example_abs": str(abs(example.copy_negate())),
        "example_copy_abs": str(example.copy_negate().copy_abs()),
        "abs_exact": F(abs(example.copy_negate())) == abs(F(example.copy_negate())),
        "trig_remainder_exact_decimal": str(trig_radius),
        "trig_remainder_unary_negation": str(-trig_radius),
        "trig_remainder_copy_negate": str(trig_radius.copy_negate()),
        "trig_unary_negation_exact": F(-trig_radius) == -F(trig_radius),
        "affected_sites": [
            "tools/check_native_quad_fullset_interval_gate_nohash.py I.__neg__",
            "tools/check_native_quad_fullset_interval_gate_nohash.py sine I(-r,r)",
            "tools/check_native_quad_fullset_interval_gate_nohash.py cosine I(-r,r)",
            "tools/check_native_quad_fullset_interval_gate_nohash.py sine/cosine abs(x.lo), abs(x.hi)",
        ],
        "meaning": "The first-box probe stopped before Picard. This is a directed-Decimal method defect, not a Flow* containment failure or a plant counterexample.",
    }
    (HERE / "BLOCKER.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "lower_hull_omitted", "computed_minus_exact_lower_approx",
        "unary_negation_exact", "trig_unary_negation_exact")}))
    return 0 if result["lower_hull_omitted"] and not result["unary_negation_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
