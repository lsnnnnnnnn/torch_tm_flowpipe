#!/usr/bin/env python3
"""Exact-rational first-box control-construction gate; no model or ODE call."""

import argparse
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import os
from pathlib import Path
import struct

if not __debug__:
    raise RuntimeError("assertions required for this gate")


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = Path(os.environ.get("QUAD_OLD_RPC", BASE / "native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json"))
SOURCE = Path(os.environ.get("QUAD_SOURCE_BOXES", BASE / "native_quad_corrected_control_construct_20261003_001/source_boxes.csv"))
CORRECTED = Path(os.environ.get("QUAD_CORRECTED_BIASES", BASE / "native_quad_crown_transport_correction_20261003_001/CORRECTED_BIASES.json"))
REPLAY = Path(os.environ.get("QUAD_SAME_SLOPE_REPLAY", BASE / "native_quad_crown_same_slope_batch_20261003_001/RAW_COEFFICIENTS.json"))


def rational(number):
    return Fraction(number)


def from_hex(value):
    return rational(float.fromhex(value))


def as_f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def shown(value):
    with localcontext() as context:
        context.prec = 45
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def endpoints(values):
    assert len(values) == 2
    lo, hi = (from_hex(value) for value in values)
    assert lo <= hi
    return lo, hi


def affine_range(poly, domain):
    """Exact extrema; refuse nonlinear or unexpected symbol dimensions."""
    zero = (0,) * len(domain)
    lo = hi = poly.get(zero, Fraction(0))
    for powers, coefficient in poly.items():
        if powers == zero:
            continue
        assert len(powers) == len(domain) and sum(powers) == 1
        symbol = powers.index(1)
        a, b = domain[symbol]
        lo += min(coefficient * a, coefficient * b)
        hi += max(coefficient * a, coefficient * b)
    return lo, hi


def polynomial(record, dimension):
    values = {}
    for term in record["terms"]:
        powers = tuple(term["degrees"] or [0] * dimension)
        assert len(powers) == dimension and all(isinstance(v, int) and v >= 0 for v in powers)
        assert sum(powers) <= 1
        values[powers] = values.get(powers, Fraction(0)) + from_hex(term["coefficient"])
    return values


def preflight():
    old = json.loads(OLD.read_text())
    corrected = json.loads(CORRECTED.read_text())
    replay = json.loads(REPLAY.read_text())
    with SOURCE.open(newline="") as source_file:
        first = list(csv.DictReader(source_file))[:12]
    assert len(first) == 12 and all(int(row["lane"]) == 0 for row in first)
    source = []
    for i, row in enumerate(first):
        assert int(row["coord"]) == i + 1
        source.append((rational(float(row["lo"])), rational(float(row["hi"]))))
    assert len(old["params"]["input_lb"]) == len(old["params"]["input_ub"]) == 12288
    assert len(old["coefficients"]["T"]) == len(corrected["u_min"]) == len(corrected["u_max"]) == 1024
    rpc = [(rational(old["params"]["input_lb"][i]),
            rational(old["params"]["input_ub"][i])) for i in range(12)]
    assert all(a <= x <= y <= b for (a, b), (x, y) in zip(rpc, source))
    rows = []
    for j in range(3):
        raw = old["coefficients"]["T"][0][j]
        assert len(raw) == 12
        assert all(replay["lA"][0][j][i] == replay["uA"][0][j][i] == raw[i]
                   for i in range(12))
        low = old["coefficients"]["u_min"][0][j]
        high = old["coefficients"]["u_max"][0][j]
        assert replay["lbias"][0][j] == low and replay["ubias"][0][j] == high
        b_lo = corrected["u_min"][0][j]
        b_hi = corrected["u_max"][0][j]
        assert as_f32(b_lo) == b_lo and as_f32(b_hi) == b_hi and b_lo <= b_hi
        slope_delta = [rational(raw[i]) - rational(as_f32(raw[i])) for i in range(12)]
        minimum = sum(min(d * a, d * b) for d, (a, b) in zip(slope_delta, rpc))
        maximum = sum(max(d * a, d * b) for d, (a, b) in zip(slope_delta, rpc))
        lower_margin = rational(low) + minimum - rational(b_lo)
        upper_margin = rational(b_hi) - rational(high) - maximum
        assert lower_margin >= 0 and upper_margin >= 0
        rows.append({"output": j + 1, "conditional_transfer_lower_margin": shown(lower_margin),
                     "conditional_transfer_upper_margin": shown(upper_margin)})
    return {"status": "SAVED_LANE0_INPUTS_AND_CONDITIONAL_CORRECTIONS_VALID",
            "source_coordinates": 12, "rpc_coordinates": 12, "outputs": rows,
            "scope": "saved first lane only; assumes original real-affine CROWN inequalities",
            "no_crown_calls": True, "no_ode_calls": True, "no_digest_verification": True}


def check_trace(path):
    old = json.loads(OLD.read_text())
    corrected = json.loads(CORRECTED.read_text())
    trace = json.loads(path.read_text())
    assert trace["lane"] == 0 and trace["mpfr_precision_bits"] == 53
    assert len(trace["source_box"]) == 16 and len(trace["domain"]) == 17
    assert len(trace["input_tms"]) == len(trace["native_rpc_input"]) == len(trace["saved_rpc_input"]) == 12
    assert len(trace["outputs"]) == 3
    domain = [endpoints(pair) for pair in trace["domain"]]
    assert domain[0] == (0, 0) and all(pair == (-1, 1) for pair in domain[1:])
    with SOURCE.open(newline="") as source_file:
        first = list(csv.DictReader(source_file))[:12]
    for i, row in enumerate(first):
        assert endpoints(trace["source_box"][i]) == (rational(float(row["lo"])), rational(float(row["hi"])))
    assert all(endpoints(pair) == (0, 0) for pair in trace["source_box"][12:])

    input_polys = []
    input_remainders = []
    for i, record in enumerate(trace["input_tms"]):
        rpc = (rational(old["params"]["input_lb"][i]),
               rational(old["params"]["input_ub"][i]))
        assert endpoints(trace["native_rpc_input"][i]) == endpoints(trace["saved_rpc_input"][i]) == rpc
        poly = polynomial(record, len(domain))
        # The first physical box is a product only if each input uses its own symbol.
        for powers in poly:
            assert sum(powers) == 0 or (sum(powers) == 1 and powers[i + 1] == 1)
        rem = endpoints(record["remainder"])
        a, b = affine_range(poly, domain)
        assert rpc[0] <= a + rem[0] and b + rem[1] <= rpc[1]
        source = endpoints(trace["source_box"][i])
        assert a + rem[0] <= source[0] and source[1] <= b + rem[1]
        input_polys.append(poly)
        input_remainders.append(rem)

    results = []
    for j, record in enumerate(trace["outputs"]):
        assert record["output"] == j + 1 and len(record["slope"]) == 12
        slope = [from_hex(value) for value in record["slope"]]
        expected_slope = [rational(as_f32(v)) for v in old["coefficients"]["T"][0][j]]
        assert slope == expected_slope
        bias = endpoints(record["corrected_bias"])
        expected_bias = (rational(corrected["u_min"][0][j]),
                         rational(corrected["u_max"][0][j]))
        assert bias == expected_bias
        assert (from_hex(record["center"]) - from_hex(record["radius"]) == bias[0] and
                from_hex(record["center"]) + from_hex(record["radius"]) == bias[1])

        output = polynomial(record["tm"], len(domain))
        remainder = endpoints(record["tm"]["remainder"])
        difference = {}
        for coefficient, poly in zip(slope, input_polys):
            for powers, value in poly.items():
                difference[powers] = difference.get(powers, Fraction(0)) + coefficient * value
        for powers, value in output.items():
            difference[powers] = difference.get(powers, Fraction(0)) - value
        delta_lo, delta_hi = affine_range(difference, domain)
        input_lo = sum(min(a * r[0], a * r[1]) for a, r in zip(slope, input_remainders))
        input_hi = sum(max(a * r[0], a * r[1]) for a, r in zip(slope, input_remainders))
        needed_lo = bias[0] + delta_lo + input_lo
        needed_hi = bias[1] + delta_hi + input_hi
        lower_margin = needed_lo - remainder[0]
        upper_margin = remainder[1] - needed_hi
        results.append({"output": j + 1, "lower_margin": shown(lower_margin),
                        "upper_margin": shown(upper_margin),
                        "same_symbol_inclusion": lower_margin >= 0 and upper_margin >= 0})
        if lower_margin < 0 or upper_margin < 0:
            break  # First refusal only; a negative sufficient margin is inconclusive.
    complete = len(results) == 3 and all(row["same_symbol_inclusion"] for row in results)
    return {"status": "CONDITIONAL_FIRST_BOX_CONTROL_CONSTRUCTION_CLOSED" if complete
                      else "UNDECIDED_FIRST_REFUSAL",
            "rows_checked": len(results), "rows": results,
            "condition": "original real-affine CROWN inequalities hold on saved RPC box",
            "limits": "No independent NN/CROWN soundness, later control, plant step or full-time property",
            "no_crown_calls": True, "no_ode_calls": True, "no_digest_verification": True}


def self_check():
    poly = {(0, 0): Fraction(1), (1, 0): Fraction(2), (0, 1): Fraction(-3)}
    assert affine_range(poly, [(0, 0), (-1, 1)]) == (-2, 4)
    assert as_f32(0.5) == 0.5 and from_hex("0x1p-1") == Fraction(1, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, help="native full-precision lane-0 trace")
    parser.add_argument("--output", type=Path, help="write receipt here")
    args = parser.parse_args()
    self_check()
    audit = preflight()
    if args.trace:
        audit = {"input_audit": audit, "construction": check_trace(args.trace)}
    payload = json.dumps(audit, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    if args.trace and audit["construction"]["status"] != "CONDITIONAL_FIRST_BOX_CONTROL_CONSTRUCTION_CLOSED":
        raise SystemExit(2)
