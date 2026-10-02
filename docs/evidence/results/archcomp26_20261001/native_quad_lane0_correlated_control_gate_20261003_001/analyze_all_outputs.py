#!/usr/bin/env python3
"""Read-only exact analysis of all three saved lane-0 control trace rows."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

import check_lane0 as gate


HERE = Path(__file__).resolve().parent


def exact(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "decimal": gate.shown(value)}


def first_outward_endpoint(start, required, lower):
    value = start
    direction = -math.inf if lower else math.inf
    for steps in range(1000001):
        if (Fraction(value) <= required if lower else Fraction(value) >= required):
            return {"binary64_ulps": steps, "endpoint_hex": value.hex(),
                    "actual_outward_shift": exact(abs(Fraction(value) - Fraction(start)))}
        value = math.nextafter(value, direction)
        if not math.isfinite(value):
            break
    raise RuntimeError("outward endpoint search exceeded finite 1,000,000-ULP cap")


def analyze(trace_path):
    # Reuse the frozen first-refusal checker for its source/RPC/input preconditions.
    gate.preflight()
    first = gate.check_trace(trace_path)
    trace = json.loads(trace_path.read_text())
    old = json.loads(gate.OLD.read_text())
    corrections = json.loads(gate.CORRECTED.read_text())
    domain = [gate.endpoints(pair) for pair in trace["domain"]]
    inputs = [gate.polynomial(item, len(domain)) for item in trace["input_tms"]]
    input_remainders = [gate.endpoints(item["remainder"]) for item in trace["input_tms"]]
    assert len(inputs) == 12 and len(trace["outputs"]) == 3

    rows = []
    for j, record in enumerate(trace["outputs"]):
        assert record["output"] == j + 1 and len(record["slope"]) == 12
        slopes = [gate.from_hex(value) for value in record["slope"]]
        assert slopes == [Fraction(gate.as_f32(value)) for value in old["coefficients"]["T"][0][j]]
        bias = gate.endpoints(record["corrected_bias"])
        assert bias == (Fraction(corrections["u_min"][0][j]),
                        Fraction(corrections["u_max"][0][j]))
        center = gate.from_hex(record["center"])
        radius = gate.from_hex(record["radius"])
        assert center - radius == bias[0] and center + radius == bias[1]
        output = gate.polynomial(record["tm"], len(domain))
        native_remainder = gate.endpoints(record["tm"]["remainder"])

        difference = {}
        for slope, polynomial in zip(slopes, inputs):
            for powers, coefficient in polynomial.items():
                difference[powers] = difference.get(powers, Fraction(0)) + slope * coefficient
        for powers, coefficient in output.items():
            difference[powers] = difference.get(powers, Fraction(0)) - coefficient
        delta_lo, delta_hi = gate.affine_range(difference, domain)
        input_lo = sum(min(a * r[0], a * r[1]) for a, r in zip(slopes, input_remainders))
        input_hi = sum(max(a * r[0], a * r[1]) for a, r in zip(slopes, input_remainders))
        required_lo = bias[0] + delta_lo + input_lo
        required_hi = bias[1] + delta_hi + input_hi
        lower_margin = required_lo - native_remainder[0]
        upper_margin = native_remainder[1] - required_hi
        rows.append({
            "output": j + 1,
            "lower_margin": exact(lower_margin),
            "upper_margin": exact(upper_margin),
            "lower_outward_deficit": exact(max(Fraction(0), -lower_margin)),
            "upper_outward_deficit": exact(max(Fraction(0), -upper_margin)),
            "minimum_binary64_lower_remainder_endpoint": first_outward_endpoint(
                float(native_remainder[0]), required_lo, True),
            "minimum_binary64_upper_remainder_endpoint": first_outward_endpoint(
                float(native_remainder[1]), required_hi, False),
            "current_same_symbol_inclusion": lower_margin >= 0 and upper_margin >= 0,
        })
    assert first["rows"][0]["lower_margin"] == rows[0]["lower_margin"]["decimal"]
    assert first["rows"][0]["upper_margin"] == rows[0]["upper_margin"]["decimal"]
    return {
        "status": "READ_ONLY_ALL_THREE_OUTPUTS_ANALYZED",
        "basis": "already saved lane-0 full-precision native TM trace; no new native execution",
        "condition": "original real-affine CROWN inequalities hold on the saved first RPC box",
        "rows": rows,
        "native_rows_satisfying_same_symbol_gate": sum(row["current_same_symbol_inclusion"] for row in rows),
        "interpretation": "Endpoint shifts refer only to holding the saved native polynomial and input TMs fixed while widening its final remainder; changing source biases would reconstruct the TM and needs a new isolated gate.",
        "limits": "Not an NN counterexample, independent CROWN proof, plant check, later-control result, or full-time property proof.",
        "no_crown_calls": True, "no_ode_calls": True, "no_digest_verification": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=HERE / "LANE0_TM_TRACE.json")
    parser.add_argument("--output", type=Path, default=HERE / "ALL_OUTPUTS_ANALYSIS.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.trace), indent=2) + "\n")
