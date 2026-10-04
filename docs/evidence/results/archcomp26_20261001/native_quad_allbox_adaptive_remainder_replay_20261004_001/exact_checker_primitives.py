#!/usr/bin/env python3
"""Offline, first-refusal conditional remainder plan from saved binary64 TMs.

No archived checker is executed or imported. Fractions derive the proposal;
exact affine vertex evaluation independently checks the proposed inclusion.
"""

import csv
from datetime import datetime, timezone
from fractions import Fraction as Q
from itertools import product
import json
import math
from pathlib import Path
import signal
import struct
import sys
import time


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCES = {
    "trace": "native_quad_allbox_correlated_remainder_gate_20261003_001/ALLBOX_TM_TRACE.jsonl",
    "rpc": "native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json",
    "source": "native_quad_corrected_control_construct_20261003_001/source_boxes.csv",
    "bias": "native_quad_crown_transport_correction_20261003_001/CORRECTED_BIASES.json",
    "replay": "native_quad_crown_same_slope_batch_20261003_001/RAW_COEFFICIENTS.json",
}
CONDITION = "Original real-affine CROWN inequalities hold on each saved RPC box."


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def number(value):
    require(isinstance(value, (int, float)) and math.isfinite(value), "nonfinite numeric input")
    return Q(value)


def hexq(value):
    return number(float.fromhex(value))


def interval(pair):
    require(len(pair) == 2, "interval shape")
    lo, hi = map(hexq, pair)
    require(lo <= hi, "unordered interval")
    return lo, hi


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def poly(record):
    coefficients = [Q(0)] * 18  # constant followed by the 17 domain symbols
    for term in record["terms"]:
        powers = term["degrees"] or [0] * 17
        require(len(powers) == 17 and all(type(v) is int and v >= 0 for v in powers)
                and sum(powers) <= 1, "non-affine or malformed TM")
        index = 0 if sum(powers) == 0 else 1 + powers.index(1)
        coefficients[index] += hexq(term["coefficient"])
    require(coefficients[1] == 0, "unexpected local-time dependence")
    return coefficients


def extrema(coefficients):
    radius = sum(abs(v) for v in coefficients[2:])
    return coefficients[0] - radius, coefficients[0] + radius


def directed(value, toward):
    """Nearest finite binary64 no farther inward than the exact rational."""
    answer = float(value)
    require(math.isfinite(answer), "nonfinite endpoint conversion")
    if (toward < 0 and Q(answer) > value) or (toward > 0 and Q(answer) < value):
        answer = math.nextafter(answer, math.copysign(math.inf, toward))
    return answer


def ordered(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (2**64 - 1 - bits) if bits >> 63 else (2**63 + bits)


def source_check(trace, lane, rpc, boxes):
    require(trace["lane"] == lane and trace["mpfr_precision_bits"] == 53, "lane/precision")
    require(len(trace["source_box"]) == 16 and len(trace["domain"]) == 17, "domain shape")
    require([interval(v) for v in trace["domain"]] == [(Q(0), Q(0))] + [(-1, 1)] * 16,
            "normalized domain")
    require(all(len(trace[k]) == 12 for k in ("input_tms", "native_rpc_input", "saved_rpc_input"))
            and len(trace["outputs"]) == 3, "input/output shape")
    inputs, rests, request = [], [], []
    for i, tm in enumerate(trace["input_tms"]):
        row = boxes[lane * 12 + i]
        require(int(row["lane"]) == lane and int(row["coord"]) == i + 1, "source CSV ordering")
        box = (number(float(row["lo"])), number(float(row["hi"])))
        x = (number(rpc["params"]["input_lb"][lane * 12 + i]),
             number(rpc["params"]["input_ub"][lane * 12 + i]))
        require(interval(trace["source_box"][i]) == box, "source box mismatch")
        require(interval(trace["native_rpc_input"][i]) == interval(trace["saved_rpc_input"][i]) == x,
                "RPC input mismatch")
        p, r = poly(tm), interval(tm["remainder"])
        require(all(v == 0 for k, v in enumerate(p) if k not in (0, i + 2)), "input symbols overlap")
        lo, hi = extrema(p)
        require(x[0] <= lo + r[0] <= box[0] <= box[1] <= hi + r[1] <= x[1],
                "source/TM/RPC inclusion")
        inputs.append(p)
        rests.append(r)
        request.append(x)
    require(all(interval(v) == (0, 0) for v in trace["source_box"][12:]), "extra source coordinates")
    return inputs, rests, request


def propose(record, lane, j, inputs, rests, request, rpc, biases, replay):
    require(record["output"] == j + 1 and len(record["slope"]) == 12, "output ordering/shape")
    raw = rpc["coefficients"]["T"][lane][j]
    require(len(raw) == 12 and replay["lA"][lane][j] == replay["uA"][lane][j] == raw,
            "saved same-slope equality")
    low = number(rpc["coefficients"]["u_min"][lane][j])
    high = number(rpc["coefficients"]["u_max"][lane][j])
    require(number(replay["lbias"][lane][j]) == low and number(replay["ubias"][lane][j]) == high,
            "replayed bias equality")
    slopes = list(map(hexq, record["slope"]))
    require(slopes == [number(f32(v)) for v in raw], "transported slope equality")
    bias = interval(record["corrected_bias"])
    require(bias == (number(biases["u_min"][lane][j]), number(biases["u_max"][lane][j]))
            and all(number(f32(float(v))) == v for v in bias), "corrected bias equality")
    delta = [number(a) - b for a, b in zip(raw, slopes)]
    require(bias[0] <= low + sum(min(d * x, d * y) for d, (x, y) in zip(delta, request))
            and bias[1] >= high + sum(max(d * x, d * y) for d, (x, y) in zip(delta, request)),
            "conditional real-CROWN to transported-envelope inclusion")
    require(hexq(record["center"]) == number((float(bias[1]) + float(bias[0])) / 2)
            and hexq(record["radius"]) == number((float(bias[1]) - float(bias[0])) / 2),
            "native center/radius trace")
    original = interval(record["original_tm_remainder"])
    # Old fixed-pad data identifies the source, but is not the proposal baseline.
    expanded = interval(record["expanded_tm_remainder"])
    require(interval(record["tm"]["remainder"]) == expanded and hexq(record["lower_pad"]) == Q(1, 2**50)
            and expanded[0] <= original[0] - Q(1, 2**50) and expanded[1] == original[1],
            "saved original/expanded remainder relation")
    output = poly(record["tm"])
    difference = [sum(s * p[k] for s, p in zip(slopes, inputs)) - output[k] for k in range(18)]
    dlo, dhi = extrema(difference)
    rlo = sum(min(s * r[0], s * r[1]) for s, r in zip(slopes, rests))
    rhi = sum(max(s * r[0], s * r[1]) for s, r in zip(slopes, rests))
    required_lo, required_hi = bias[0] + dlo + rlo, bias[1] + dhi + rhi
    proposed = (min(float(original[0]), directed(required_lo, -1)),
                max(float(original[1]), directed(required_hi, 1)))
    return slopes, bias, output, original, proposed, (required_lo, required_hi), (rlo, rhi)


def verify_vertices(inputs, slopes, bias, output, original, proposed, required, residual):
    """Independent inclusion: evaluate each affine term at every active vertex.

    Linear functions attain extrema at vertices; the input TM remainder
    intervals are added with exact sign-dependent extrema.
    """
    active = [k for k in range(2, 18) if output[k] or any(p[k] for p in inputs)]
    require(len(active) <= 6, "unexpected active symbol count")
    low = high = None
    count = 0
    for signs in product((-1, 1), repeat=len(active)):
        expected = sum(s * (p[0] + sum(p[k] * v for k, v in zip(active, signs)))
                       for s, p in zip(slopes, inputs))
        observed = output[0] + sum(output[k] * v for k, v in zip(active, signs))
        vertex_lo = bias[0] + expected - observed + residual[0]
        vertex_hi = bias[1] + expected - observed + residual[1]
        require(Q(proposed[0]) <= vertex_lo and vertex_hi <= Q(proposed[1]),
                "conditional affine-vertex inclusion refused")
        low = vertex_lo if low is None else min(low, vertex_lo)
        high = vertex_hi if high is None else max(high, vertex_hi)
        count += 1
    require((low, high) == required, "vertex and coefficient extrema disagree")
    require(Q(proposed[0]) <= original[0] and Q(proposed[1]) >= original[1], "old interval not preserved")
    # Minimal means preserving the original interval AND covering this reference.
    require(Q(math.nextafter(proposed[0], math.inf)) > min(original[0], low)
            and Q(math.nextafter(proposed[1], -math.inf)) < max(original[1], high),
            "proposed endpoint not nearest outward binary64")
    return count


def self_check():
    require(directed(Q(1) + Q(1, 2**54), -1) == 1.0, "lower rounding check")
    require(directed(Q(1) + Q(1, 2**54), 1) == math.nextafter(1.0, math.inf), "upper rounding check")
    p = [Q(0)] * 18
    p[2] = Q(1)
    o = p.copy()
    o[0] = Q(1, 2**54)
    required = (Q(-1) - Q(1, 2**54), Q(1) - Q(1, 2**54))
    candidate = (directed(required[0], -1), 1.0)
    require(verify_vertices([p], [Q(1)], (-1, 1), o, (-1, 1), candidate, required, (0, 0)) == 2,
            "synthetic corrected inclusion")
    try:
        verify_vertices([p], [Q(1)], (-1, 1), o, (-1, 1), (-1.0, 1.0), required, (0, 0))
    except ValueError:
        pass
    else:
        raise ValueError("synthetic under-enclosure was not refused")


def main():
    self_check()
    if sys.argv[1:] == ["--self-check"]:
        print("SELF_CHECK_ROUNDING_VERTEX_INCLUSION_AND_REFUSAL_PASSED")
        return 0
    require(not sys.argv[1:], "no arguments except --self-check")
    for name in ("INPUT.json", "RESULT.json", "PLAN.csv"):
        require(not (HERE / name).exists(), "refusing to overwrite " + name)
    started = time.monotonic()
    receipt = {"scope": "Saved first-control 1024 boxes x 3 outputs; offline plan only",
               "condition": CONDITION, "native_production_gate": "CLOSED",
               "native_plan_applied": False, "no_crown_calls": True, "no_ode_calls": True,
               "no_network_calls": True, "no_digest_verification": True,
               "source_boxes_checked": 0, "output_rows_checked": 0, "exact_vertex_checks": 0}
    inputs = {"started_utc": datetime.now(timezone.utc).isoformat(), "cap_seconds": 300,
              "sources_relative_to_parent": SOURCES, "condition": CONDITION,
              "reference": "transported f32 slopes and conditionally corrected f32 bias interval",
              "baseline": "original_tm_remainder before the archived fixed 2^-50 pad",
              "selection": "nearest outward binary64 endpoints preserving the original remainder",
              "scope_exclusions": "NN/CROWN soundness, native application, ODE, later controls and full horizon"}
    (HERE / "INPUT.json").write_text(json.dumps(inputs, indent=2) + "\n")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("300 second cap")))
    signal.alarm(300)
    lane = output_index = None
    stats = [{"output": j + 1, "lower_changed": 0, "upper_changed": 0,
              "either_changed": 0, "max_lower_expansion": Q(0), "max_upper_expansion": Q(0),
              "max_lower_ulps": 0, "max_upper_ulps": 0} for j in range(3)]
    changed_lanes = 0
    fields = ["lane", "output", "original_lower_hex", "original_upper_hex", "required_lower_rational",
              "required_upper_rational", "proposed_lower_hex", "proposed_upper_hex", "lower_expansion_rational",
              "upper_expansion_rational", "lower_ulps", "upper_ulps", "lower_margin_rational", "upper_margin_rational"]
    try:
        rpc, biases, replay = [json.loads((BASE / SOURCES[k]).read_text()) for k in ("rpc", "bias", "replay")]
        with (BASE / SOURCES["source"]).open(newline="") as stream:
            boxes = list(csv.DictReader(stream))
        require(len(boxes) == len(rpc["params"]["input_lb"]) == len(rpc["params"]["input_ub"]) == 12288,
                "source/RPC batch size")
        require(all(len(a) == 1024 for a in (rpc["coefficients"]["T"], biases["u_min"], biases["u_max"],
                    replay["lA"], replay["uA"], replay["lbias"], replay["ubias"])), "coefficient batch size")
        with (BASE / SOURCES["trace"]).open() as traces, (HERE / "PLAN.csv").open("w", newline="") as ledger:
            writer = csv.DictWriter(ledger, fieldnames=fields)
            writer.writeheader()
            for lane in range(1024):
                line = traces.readline()
                require(bool(line), "missing trace row")
                trace = json.loads(line)
                polys, rests, request = source_check(trace, lane, rpc, boxes)
                changed = False
                for output_index, record in enumerate(trace["outputs"]):
                    values = propose(record, lane, output_index, polys, rests, request, rpc, biases, replay)
                    slopes, bias, out, original, proposed, needed, residual = values
                    vertices = verify_vertices(polys, slopes, bias, out, original, proposed, needed, residual)
                    dlo, dhi = original[0] - Q(proposed[0]), Q(proposed[1]) - original[1]
                    ulps = (ordered(float(original[0])) - ordered(proposed[0]),
                            ordered(proposed[1]) - ordered(float(original[1])))
                    row = dict(zip(fields, [lane, output_index + 1, float(original[0]).hex(), float(original[1]).hex(),
                        str(needed[0]), str(needed[1]), proposed[0].hex(), proposed[1].hex(), str(dlo), str(dhi),
                        *ulps, str(needed[0] - Q(proposed[0])), str(Q(proposed[1]) - needed[1])]))
                    writer.writerow(row)
                    ledger.flush()  # Keep the verified prefix if any later row refuses.
                    receipt["output_rows_checked"] += 1
                    receipt["exact_vertex_checks"] += vertices
                    slot = stats[output_index]
                    slot["lower_changed"] += dlo > 0
                    slot["upper_changed"] += dhi > 0
                    slot["either_changed"] += dlo + dhi > 0
                    for key, value in (("max_lower_expansion", dlo), ("max_upper_expansion", dhi),
                                       ("max_lower_ulps", ulps[0]), ("max_upper_ulps", ulps[1])):
                        slot[key] = max(slot[key], value)
                    changed |= dlo + dhi > 0
                receipt["source_boxes_checked"] += 1
                changed_lanes += changed
            require(not traces.readline(), "extra trace row")
        receipt["status"] = "CONDITIONAL_OFFLINE_REMAINDER_PLAN_VERIFIED"
        receipt["boxes_with_any_endpoint_change"] = changed_lanes
        receipt["per_output"] = [{k: str(v) if isinstance(v, Q) else v for k, v in row.items()} for row in stats]
        code = 0
    except (ValueError, KeyError, TypeError, IndexError, OverflowError, OSError, TimeoutError) as error:
        receipt.update(status="FIRST_REFUSAL_OFFLINE_PLAN", at_lane=lane,
                       at_output=None if output_index is None else output_index + 1,
                       reason=f"{type(error).__name__}: {error}")
        code = 2
    finally:
        signal.alarm(0)
        receipt["elapsed_seconds"] = time.monotonic() - started
        (HERE / "RESULT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
