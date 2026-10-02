#!/usr/bin/env python3
"""Exact-rational sufficiency check for the saved QUAD RPC's native float32 transport.

Reads the existing RPC once. It neither calls CROWN nor evaluates the network.
"""

import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
RPC = HERE.parent / "native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json"


def exact(value):
    return Fraction.from_float(value)


def native_f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def display(value):
    with localcontext() as context:
        context.prec = 26
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def main():
    rpc = json.loads(RPC.read_text())
    params = rpc["params"]
    coeff = rpc["coefficients"]
    lower = params["input_lb"]
    upper = params["input_ub"]
    assert len(lower) == len(upper) == 1024 * 12
    assert all(len(coeff[name]) == 1024 for name in ("T", "u_min", "u_max"))
    totals = [{"lower_failures": 0, "upper_failures": 0, "either_failures": 0,
               "max_lower_expansion": Fraction(0), "max_upper_expansion": Fraction(0),
               "max_lower_lane": None, "max_upper_lane": None} for _ in range(3)]
    lane0 = []
    for lane in range(1024):
        box = [(exact(lower[lane * 12 + i]), exact(upper[lane * 12 + i]))
               for i in range(12)]
        assert all(lo <= hi for lo, hi in box)
        assert len(coeff["T"][lane]) == len(coeff["u_min"][lane]) == len(coeff["u_max"][lane]) == 3
        for output in range(3):
            row = coeff["T"][lane][output]
            assert len(row) == 12
            raw_lower = exact(coeff["u_min"][lane][output])
            raw_upper = exact(coeff["u_max"][lane][output])
            transported_lower = exact(native_f32(coeff["u_min"][lane][output]))
            transported_upper = exact(native_f32(coeff["u_max"][lane][output]))
            assert raw_lower <= raw_upper and transported_lower <= transported_upper

            # The upper calculation assumes the unsaved uA equals saved lA.
            # The lower calculation uses saved lA and lbias alone.
            delta_min = Fraction(0)
            delta_max = Fraction(0)
            for slope, (lo, hi) in zip(row, box):
                delta = exact(slope) - exact(native_f32(slope))
                delta_min += min(delta * lo, delta * hi)
                delta_max += max(delta * lo, delta * hi)
            lower_slack = raw_lower + delta_min - transported_lower
            upper_slack = transported_upper - raw_upper - delta_max
            lower_expansion = max(Fraction(0), -lower_slack)
            upper_expansion = max(Fraction(0), -upper_slack)
            total = totals[output]
            total["lower_failures"] += lower_expansion > 0
            total["upper_failures"] += upper_expansion > 0
            total["either_failures"] += lower_expansion > 0 or upper_expansion > 0
            if lower_expansion > total["max_lower_expansion"]:
                total["max_lower_expansion"] = lower_expansion
                total["max_lower_lane"] = lane
            if upper_expansion > total["max_upper_expansion"]:
                total["max_upper_expansion"] = upper_expansion
                total["max_upper_lane"] = lane
            if lane == 0:
                lane0.append({
                    "output": output + 1,
                    "raw_lower": display(raw_lower),
                    "raw_upper": display(raw_upper),
                    "transported_lower": display(transported_lower),
                    "transported_upper": display(transported_upper),
                    "lower_slack": display(lower_slack),
                    "upper_slack_assuming_same_slope": display(upper_slack),
                    "minimum_lower_expansion": display(lower_expansion),
                    "minimum_upper_expansion_assuming_same_slope": display(upper_expansion),
                })
    result = {
        "scope": "saved 1024-box first-call RPC only; no CROWN, NN, or plant execution",
        "input": str(RPC.relative_to(HERE.parent)),
        "input_interpretation": "RPC JSON numbers parsed as binary64, then lifted exactly to rational values",
        "transport": "each saved slope and bias is rounded to IEEE binary32 as C++ JsonCpp .asFloat() does",
        "assumption_for_upper": "unsaved uA equals saved lA; same-slope option alone is not a saved equality witness",
        "ideal_certificate_assumption": "CROWN real-arithmetic affine lower/upper inequalities are valid on each RPC input box",
        "lower_formula": "L_raw + min_X((lA - f32(lA)) dot x) - f32(L_raw)",
        "upper_formula_if_same_slope": "f32(U_raw) - U_raw - max_X((lA - f32(lA)) dot x)",
        "passing_rule": "both slacks >= 0; a failed rule only means this direct certificate transfer is inconclusive",
        "boxes": 1024,
        "lane0": lane0,
        "per_output": [{
            "output": j + 1,
            "lower_failures": t["lower_failures"],
            "upper_failures_assuming_same_slope": t["upper_failures"],
            "either_failures_assuming_same_slope": t["either_failures"],
            "max_lower_expansion": display(t["max_lower_expansion"]),
            "max_lower_lane": t["max_lower_lane"],
            "max_upper_expansion_assuming_same_slope": display(t["max_upper_expansion"]),
            "max_upper_lane_assuming_same_slope": t["max_upper_lane"],
        } for j, t in enumerate(totals)],
        "conclusion": "Sufficient direct transport conditions fail for many boxes; no actual network violation is shown.",
    }
    (HERE / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"boxes": result["boxes"], "per_output": result["per_output"]}, indent=2))


if __name__ == "__main__":
    main()
