#!/usr/bin/env python3
"""Check one isolated native remainder expansion against the saved lane-0 trace."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "native_quad_lane0_correlated_control_gate_20261003_001"
sys.path.insert(0, str(PRIOR))
import check_lane0  # noqa: E402: reuse the archived exact-rational gate


class FirstRefusal(Exception):
    pass


def require(condition, reason):
    if not condition:
        raise FirstRefusal(reason)


def check_baseline(new):
    old = json.loads((PRIOR / "LANE0_TM_TRACE.json").read_text())
    for field in ("lane", "mpfr_precision_bits", "source_box", "domain",
                  "native_rpc_input", "saved_rpc_input", "input_tms"):
        require(new[field] == old[field], f"saved baseline differs: {field}")
    require(len(new["outputs"]) == len(old["outputs"]) == 3, "output count differs")
    receipt = []
    pad = Fraction(1, 2**50)
    for index, (after, before) in enumerate(zip(new["outputs"], old["outputs"]), 1):
        for field in ("output", "slope", "corrected_bias", "center", "radius"):
            require(after[field] == before[field], f"output {index}: {field} changed")
        require(after["tm"]["terms"] == before["tm"]["terms"],
                f"output {index}: polynomial changed")
        require(after["original_tm_remainder"] == before["tm"]["remainder"],
                f"output {index}: original remainder changed")
        require(after["expanded_tm_remainder"] == after["tm"]["remainder"],
                f"output {index}: final remainder trace disagrees")
        require(Fraction(float.fromhex(after["lower_pad"])) == pad,
                f"output {index}: lower pad differs")
        old_lo, old_hi = check_lane0.endpoints(before["tm"]["remainder"])
        new_lo, new_hi = check_lane0.endpoints(after["tm"]["remainder"])
        require(new_lo <= old_lo - pad and new_hi == old_hi,
                f"output {index}: native remainder was not widened downward as requested")
        receipt.append({"output": index, "original_remainder": before["tm"]["remainder"],
                        "expanded_remainder": after["tm"]["remainder"],
                        "actual_lower_shift": str(old_lo - new_lo)})
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"scope": "lane 0 only; conditional on original real-affine CROWN inequalities",
              "pad": "2^-50 at final native TM remainder lower side on all three outputs",
              "no_crown_calls": True, "no_ode_calls": True,
              "no_digest_verification": True}
    try:
        new = json.loads(args.trace.read_text())
        result["baseline"] = check_baseline(new)
        result["input_audit"] = check_lane0.preflight()
        result["construction"] = check_lane0.check_trace(args.trace)
        result["status"] = result["construction"]["status"]
        exit_code = 0 if result["status"] == "CONDITIONAL_FIRST_BOX_CONTROL_CONSTRUCTION_CLOSED" else 2
    except (FirstRefusal, AssertionError, KeyError, TypeError, ValueError) as error:
        result["status"] = "FIRST_REFUSAL_PRECONDITION"
        result["reason"] = str(error)
        exit_code = 2
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
