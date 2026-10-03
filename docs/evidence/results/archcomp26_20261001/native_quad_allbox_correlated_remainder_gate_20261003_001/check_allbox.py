#!/usr/bin/env python3
"""First-refusal exact same-symbol audit of 1,024 saved native control traces."""

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PRIOR = BASE / "native_quad_lane0_correlated_control_gate_20261003_001"
PADDED_LANE0 = BASE / "native_quad_lane0_correlated_remainder_gate_20261003_001/LANE0_TM_TRACE.json"
sys.path.insert(0, str(PRIOR))
import check_lane0 as gate  # noqa: E402; reuse archived exact arithmetic


class FirstRefusal(Exception):
    pass


def require(condition, reason):
    if not condition:
        raise FirstRefusal(reason)


def check_record(trace, lane, saved, corrected, replay, source_rows):
    require(trace["lane"] == lane and trace["mpfr_precision_bits"] == 53,
            "lane or precision differs")
    require(len(trace["source_box"]) == 16 and len(trace["domain"]) == 17,
            "source or normalized domain shape differs")
    require(all(len(trace[key]) == 12 for key in
                ("native_rpc_input", "saved_rpc_input", "input_tms")),
            "input trace shape differs")
    require(len(trace["outputs"]) == 3, "output trace shape differs")
    if lane == 0:
        require(trace == json.loads(PADDED_LANE0.read_text()),
                "new lane-0 trace differs from archived native replay")
    domain = [gate.endpoints(pair) for pair in trace["domain"]]
    require(domain[0] == (0, 0) and all(pair == (-1, 1) for pair in domain[1:]),
            "normalized domain differs")
    inputs = []
    remainders = []
    for i in range(12):
        row = source_rows[lane * 12 + i]
        require(int(row["lane"]) == lane and int(row["coord"]) == i + 1,
                f"source CSV order differs at coordinate {i + 1}")
        source = (Fraction(float(row["lo"])), Fraction(float(row["hi"])))
        require(gate.endpoints(trace["source_box"][i]) == source,
                f"source interval differs at coordinate {i + 1}")
        rpc = (Fraction(saved["params"]["input_lb"][lane * 12 + i]),
               Fraction(saved["params"]["input_ub"][lane * 12 + i]))
        require(gate.endpoints(trace["native_rpc_input"][i]) == rpc and
                gate.endpoints(trace["saved_rpc_input"][i]) == rpc,
                f"native/RPC input differs at coordinate {i + 1}")
        require(rpc[0] <= source[0] <= source[1] <= rpc[1],
                f"RPC input omits source at coordinate {i + 1}")
        record = trace["input_tms"][i]
        poly = gate.polynomial(record, len(domain))
        require(all(sum(powers) == 0 or
                    (sum(powers) == 1 and powers[i + 1] == 1) for powers in poly),
                f"input TM has another symbol at coordinate {i + 1}")
        rem = gate.endpoints(record["remainder"])
        low, high = gate.affine_range(poly, domain)
        require(rpc[0] <= low + rem[0] <= source[0] and
                source[1] <= high + rem[1] <= rpc[1],
                f"input TM/source/RPC inclusion differs at coordinate {i + 1}")
        inputs.append(poly)
        remainders.append(rem)
    require(all(gate.endpoints(item) == (0, 0) for item in trace["source_box"][12:]),
            "time/control source coordinates differ")

    pad = Fraction(1, 2**50)
    rows = []
    for j, record in enumerate(trace["outputs"]):
        require(record["output"] == j + 1 and len(record["slope"]) == 12,
                f"output {j + 1} shape differs")
        raw = saved["coefficients"]["T"][lane][j]
        require(len(raw) == 12 and all(replay["lA"][lane][j][i] ==
                    replay["uA"][lane][j][i] == raw[i] for i in range(12)),
                f"output {j + 1} saved same-slope receipt differs")
        require(replay["lbias"][lane][j] == saved["coefficients"]["u_min"][lane][j]
                and replay["ubias"][lane][j] == saved["coefficients"]["u_max"][lane][j],
                f"output {j + 1} saved bias replay differs")
        slopes = [gate.from_hex(value) for value in record["slope"]]
        require(slopes == [Fraction(gate.as_f32(value)) for value in raw],
                f"output {j + 1} transported slopes differ")
        bias = gate.endpoints(record["corrected_bias"])
        require(bias == (Fraction(corrected["u_min"][lane][j]),
                         Fraction(corrected["u_max"][lane][j])),
                f"output {j + 1} corrected bias differs")
        expected_center = (float(bias[1]) + float(bias[0])) / 2
        expected_radius = (float(bias[1]) - float(bias[0])) / 2
        require(gate.from_hex(record["center"]) == Fraction(expected_center) and
                gate.from_hex(record["radius"]) == Fraction(expected_radius),
                f"output {j + 1} native double center/radius construction differs")
        slope_delta = [Fraction(value) - transported for value, transported in zip(raw, slopes)]
        rpc = [(Fraction(saved["params"]["input_lb"][lane * 12 + i]),
                Fraction(saved["params"]["input_ub"][lane * 12 + i])) for i in range(12)]
        transfer_lo = sum(min(a * x, a * y) for a, (x, y) in zip(slope_delta, rpc))
        transfer_hi = sum(max(a * x, a * y) for a, (x, y) in zip(slope_delta, rpc))
        require(Fraction(saved["coefficients"]["u_min"][lane][j]) + transfer_lo >= bias[0]
                and bias[1] >= Fraction(saved["coefficients"]["u_max"][lane][j]) + transfer_hi,
                f"output {j + 1} conditional float32 transfer differs")

        old_rem = gate.endpoints(record["original_tm_remainder"])
        new_rem = gate.endpoints(record["expanded_tm_remainder"])
        require(new_rem == gate.endpoints(record["tm"]["remainder"]) and
                gate.from_hex(record["lower_pad"]) == pad and
                new_rem[0] <= old_rem[0] - pad and new_rem[1] == old_rem[1],
                f"output {j + 1} native remainder pad differs")
        output = gate.polynomial(record["tm"], len(domain))
        delta = {}
        for slope, poly in zip(slopes, inputs):
            for powers, value in poly.items():
                delta[powers] = delta.get(powers, Fraction(0)) + slope * value
        for powers, value in output.items():
            delta[powers] = delta.get(powers, Fraction(0)) - value
        delta_lo, delta_hi = gate.affine_range(delta, domain)
        input_lo = sum(min(a * r[0], a * r[1]) for a, r in zip(slopes, remainders))
        input_hi = sum(max(a * r[0], a * r[1]) for a, r in zip(slopes, remainders))
        lower_margin = bias[0] + delta_lo + input_lo - new_rem[0]
        upper_margin = new_rem[1] - bias[1] - delta_hi - input_hi
        row = {"lane": lane, "output": j + 1,
               "lower_margin": gate.shown(lower_margin),
               "upper_margin": gate.shown(upper_margin)}
        rows.append(row)
        if lower_margin < 0 or upper_margin < 0:
            raise FirstRefusal("conditional same-symbol inclusion undecided", row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    require(args.self_check or (args.trace and args.output), "trace and output paths required")
    saved = json.loads(gate.OLD.read_text())
    corrected = json.loads(gate.CORRECTED.read_text())
    replay = json.loads(gate.REPLAY.read_text())
    with gate.SOURCE.open(newline="") as source_file:
        source_rows = list(csv.DictReader(source_file))
    require(len(source_rows) == 12288 and len(saved["params"]["input_lb"]) == 12288 and
            len(saved["params"]["input_ub"]) == 12288 and
            len(saved["coefficients"]["T"]) == len(corrected["u_min"]) ==
            len(corrected["u_max"]) == len(replay["lA"]) == len(replay["uA"]) == 1024,
            "saved first-batch cardinality differs")
    if args.self_check:
        record = json.loads(PADDED_LANE0.read_text())
        rows = check_record(record, 0, saved, corrected, replay, source_rows)
        require(len(rows) == 3, "lane-0 fixture did not pass")
        print("SELF_CHECK_ARCHIVED_LANE0_3_OF_3_CONDITIONAL_ROWS")
        return 0

    receipt = {"scope": "first control call, 1,024 source boxes, 3 outputs per box",
               "condition": "original real-affine CROWN inequalities on the saved RPC boxes",
               "no_crown_calls": True, "no_ode_calls": True,
               "no_digest_verification": True,
               "source_boxes_checked": 0, "output_rows_checked": 0}
    try:
        with args.trace.open() as trace_file:
            for lane in range(1024):
                line = trace_file.readline()
                require(line, f"missing native trace at lane {lane}")
                rows = check_record(json.loads(line), lane, saved, corrected, replay, source_rows)
                receipt["source_boxes_checked"] += 1
                receipt["output_rows_checked"] += len(rows)
            require(not trace_file.readline(), "extra native trace after lane 1023")
        receipt["status"] = "CONDITIONAL_ALLBOX_CONTROL_CONSTRUCTION_CLOSED"
        exit_code = 0
    except FirstRefusal as error:
        receipt["status"] = ("UNDECIDED_FIRST_REFUSAL" if len(error.args) > 1
                             else "FIRST_REFUSAL_PRECONDITION")
        receipt["reason"] = error.args[0]
        receipt["at_lane"] = lane
        if len(error.args) > 1:
            receipt["first_refusal"] = error.args[1]
            receipt["output_rows_checked"] += error.args[1]["output"]
        exit_code = 2
    except (AssertionError, KeyError, TypeError, ValueError, IndexError) as error:
        receipt["status"] = "FIRST_REFUSAL_PRECONDITION"
        receipt["reason"] = f"{type(error).__name__}: {error}"
        receipt["at_lane"] = lane
        exit_code = 2
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
