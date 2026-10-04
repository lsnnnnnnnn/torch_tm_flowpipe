#!/usr/bin/env python3
"""First-refusal exact check of newly observed native endpoint replacements."""

import copy
import csv
import json
import os
from pathlib import Path
import signal
import sys
import time

import exact_checker_primitives as exact


HERE = Path(__file__).resolve().parent


def check():
    started = time.monotonic()
    receipt = {"scope": "1024 saved first-control boxes x 3 native control TMs; construction only",
               "condition": exact.CONDITION, "native_production_gate": "CLOSED",
               "no_crown_calls": True, "no_ode_calls": True, "no_nn_inference": True,
               "no_digest_verification": True, "source_boxes_checked": 0,
               "output_rows_checked": 0, "exact_vertex_checks": 0}
    lane = output = None
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("300 second checker cap")))
    signal.alarm(300)
    try:
        exact.require(not (HERE / "RESULT.json").exists() and not (HERE / "CHECK_ROWS.csv").exists(),
                      "refusing existing checker outputs")
        rpc, bias, replay = [json.loads(Path(os.environ[name]).read_text()) for name in
                            ("QUAD_OLD_RPC", "QUAD_CORRECTED_BIASES", "QUAD_SAME_SLOPE_REPLAY")]
        with Path(os.environ["QUAD_SOURCE_BOXES"]).open(newline="") as stream:
            boxes = list(csv.DictReader(stream))
        with (HERE / "PLAN.csv").open(newline="") as stream:
            plan = list(csv.DictReader(stream))
        plan_native = json.loads((HERE / "PLAN_ENDPOINTS.json").read_text())["rows"]
        exact.require(len(boxes) == 12288 and len(plan) == 3072 and len(plan_native) == 1024,
                      "source or plan row counts")
        exact.require(len(rpc["params"]["input_lb"]) == len(rpc["params"]["input_ub"]) == 12288,
                      "RPC cardinality")
        exact.require(all(len(value) == 1024 for value in (rpc["coefficients"]["T"], bias["u_min"],
                      bias["u_max"], replay["lA"], replay["uA"], replay["lbias"], replay["ubias"])),
                      "coefficient cardinality")
        fields = ["lane", "output", "actual_lower_hex", "actual_upper_hex",
                  "lower_margin_rational", "upper_margin_rational", "vertex_checks"]
        with Path(os.environ["QUAD_PRIOR_TRACE"]).open() as old_stream, \
             Path(os.environ["QUAD_TRACE_OUT"]).open() as new_stream, \
             (HERE / "CHECK_ROWS.csv").open("w", newline="") as ledger:
            writer = csv.DictWriter(ledger, fieldnames=fields)
            writer.writeheader()
            for lane in range(1024):
                output = None
                old_line, new_line = old_stream.readline(), new_stream.readline()
                exact.require(bool(old_line) and bool(new_line), "missing trace row")
                old, new = json.loads(old_line), json.loads(new_line)
                exact.require({k: v for k, v in new.items() if k != "outputs"} ==
                              {k: v for k, v in old.items() if k != "outputs"},
                              "native source/input/domain trace changed")
                inputs, rests, request = exact.source_check(new, lane, rpc, boxes)
                for output in range(3):
                    prior, actual = old["outputs"][output], new["outputs"][output]
                    before = copy.deepcopy(actual)
                    before.pop("replacement_tm_remainder")
                    before["tm"]["remainder"] = before["original_tm_remainder"]
                    expected = copy.deepcopy(prior)
                    expected.pop("expanded_tm_remainder")
                    expected.pop("lower_pad")
                    expected["tm"]["remainder"] = expected["original_tm_remainder"]
                    exact.require(before == expected, "native original output polynomial/inputs/metadata changed")
                    slopes, bias_pair, polynomial, original, proposed, needed, residual = exact.propose(
                        prior, lane, output, inputs, rests, request, rpc, bias, replay)
                    row = plan[lane * 3 + output]
                    exact.require(int(row["lane"]) == lane and int(row["output"]) == output + 1,
                                  "plan CSV ordering")
                    pair = (exact.hexq(row["proposed_lower_hex"]), exact.hexq(row["proposed_upper_hex"]))
                    observed = exact.interval(actual["tm"]["remainder"])
                    exact.require(observed == exact.interval(actual["replacement_tm_remainder"]) == pair
                                  == tuple(map(exact.Q, proposed)), "native endpoint differs from exact plan")
                    expected_plan = plan_native[lane]
                    exact.require(expected_plan["lane"] == lane and
                                  expected_plan["outputs"][output]["output"] == output + 1 and
                                  exact.interval(expected_plan["outputs"][output]["remainder"]) == observed,
                                  "native input plan differs from CSV")
                    exact.require(exact.Q(row["required_lower_rational"]) == needed[0] and
                                  exact.Q(row["required_upper_rational"]) == needed[1],
                                  "recorded required rational interval differs")
                    count = exact.verify_vertices(inputs, slopes, bias_pair, polynomial, original,
                                                  tuple(map(float, observed)), needed, residual)
                    writer.writerow(dict(zip(fields, [lane, output + 1, float(observed[0]).hex(),
                        float(observed[1]).hex(), str(needed[0] - observed[0]), str(observed[1] - needed[1]), count])))
                    ledger.flush()
                    receipt["output_rows_checked"] += 1
                    receipt["exact_vertex_checks"] += count
                receipt["source_boxes_checked"] += 1
            exact.require(not old_stream.readline() and not new_stream.readline(), "extra trace row")
        receipt["status"] = "CONDITIONAL_NATIVE_ADAPTIVE_REMAINDER_CONSTRUCTION_VERIFIED"
        code = 0
    except (ValueError, KeyError, TypeError, IndexError, OverflowError, OSError, TimeoutError) as error:
        receipt.update(status="FIRST_REFUSAL_NATIVE_ADAPTIVE_REMAINDER", at_lane=lane,
                       at_output=None if output is None else output + 1, reason=f"{type(error).__name__}: {error}")
        code = 2
    finally:
        signal.alarm(0)
        receipt["elapsed_seconds"] = time.monotonic() - started
    # Refuse replacement even if a concurrent external writer appeared.
    with (HERE / "RESULT.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2))
    return code


if __name__ == "__main__":
    exact.self_check()
    if sys.argv[1:] == ["--self-check"]:
        print("SELF_CHECK_EXACT_ARITHMETIC_AND_SYNTHETIC_REFUSAL_PASSED")
    elif not sys.argv[1:]:
        raise SystemExit(check())
    else:
        raise SystemExit("only --self-check is supported")
