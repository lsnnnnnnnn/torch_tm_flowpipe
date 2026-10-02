#!/usr/bin/env python3
"""Check exact Decimal sign/abs and outward primitive arithmetic."""

from decimal import Decimal as D, getcontext
from fractions import Fraction as F
import json
from pathlib import Path

import interval_fixed_nohash as gate


HERE = Path(__file__).resolve().parent


def contains(interval, exact):
    return F(interval.lo) <= exact <= F(interval.hi)


def main():
    exact_decimal = D.from_float(-0.0014866764919133857)
    positive = exact_decimal.copy_abs()
    negated = -gate.I(positive)
    check = {
        "default_context_precision": getcontext().prec,
        "copy_abs_exact": F(exact_decimal.copy_abs()) == abs(F(exact_decimal)),
        "context_abs_inexact": F(abs(exact_decimal)) != abs(F(exact_decimal)),
        "copy_negate_exact": F(positive.copy_negate()) == -F(positive),
        "interval_negation_contains_exact": contains(negated, -F(positive)),
    }
    a = gate.I(D.from_float(-0.3), D.from_float(0.2))
    b = gate.I(D.from_float(0.7), D.from_float(1.1))
    for name, interval, exact in (
        ("add", a + b, F(a.lo) + F(b.lo)),
        ("subtract", a - b, F(a.lo) - F(b.hi)),
        ("multiply", a * b, F(a.lo) * F(b.hi)),
        ("divide", a / b, F(a.lo) / F(b.lo)),
    ):
        check[name + "_contains_endpoint"] = contains(interval, exact)
    magnitude = D.from_float(0.01)
    for name, power, factorial in (("sine", 7, 5040), ("cosine", 6, 720)):
        remainder = gate.trig_remainder(magnitude, power, factorial)
        check[name + "_remainder_up"] = F(remainder) >= F(magnitude) ** power / factorial
        check[name + "_negative_remainder_exact"] = (
            F(remainder.copy_negate()) == -F(remainder))
    result = {
        "schema": "native-quad-fixed-decimal-primitive-self-check-nohash-v1",
        "checks": check,
        "passed": all(check.values()),
        "direct_decimal_operations_audited": [
            "I.__neg__: copy_negate exact",
            "sine/cosine interval magnitude: copy_abs exact",
            "sine/cosine Taylor remainder negative radius: copy_negate exact",
            "load_contract/control construction radius: copy_negate exact",
            "Picard proposal epsilon negative radius: copy_negate exact",
            "ODE decimal 0.054-0.104 and 0.104-0.054: short exact terminating differences",
            "H/STEPS=0.005/1000: exact terminating decimal 0.000005",
            "other interval +,* and reciprocal divisions: localcontext directed down/up",
        ],
    }
    (HERE / "METHOD_SELF_CHECK.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": check}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
