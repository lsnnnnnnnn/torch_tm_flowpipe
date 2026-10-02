#!/usr/bin/env python3
"""Read-only explanation of the stopped lane-0/output-1 construction receipt."""

import csv
import json
from fractions import Fraction
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
CORRECTION = HERE.parent / "native_quad_crown_transport_correction_20261003_001/CORRECTED_BIASES.json"


def main():
    result = json.loads((HERE / "RESULT.json").read_text())
    rows = list(csv.DictReader((HERE / "construction.csv").open()))
    correction = json.loads(CORRECTION.read_text())
    assert result["run_exit_code"] == 4 and result["construction_rows"] == 1
    assert len(rows) == 1 and rows[0]["lane"] == "0" and rows[0]["output"] == "1"
    assert "FIRST_REFUSAL_CONTROL_CONSTRUCTION lane=0 output=1" in (HERE / "stderr.log").read_text()
    lower = correction["u_min"][0][0]
    upper = correction["u_max"][0][0]
    assert float(rows[0]["corrected_lower"]) == lower
    assert float(rows[0]["corrected_upper"]) == upper
    for value in (lower, upper):
        assert struct.unpack("<f", struct.pack("<f", value))[0] == value
    center = (upper + lower) / 2
    radius = (upper - lower) / 2
    q = Fraction.from_float
    exact_lower = q(center) - q(radius) == q(lower)
    exact_upper = q(center) + q(radius) == q(upper)
    assert exact_lower and exact_upper
    analysis = {
        "scope": "saved first refusal only; no construction rerun",
        "first_refusal": {"lane": 0, "output": 1, "run_exit_code": 4},
        "corrected_lower_binary64_hex": lower.hex(),
        "corrected_upper_binary64_hex": upper.hex(),
        "computed_center_binary64_hex": center.hex(),
        "computed_radius_binary64_hex": radius.hex(),
        "exact_rational_center_minus_radius_equals_corrected_lower": exact_lower,
        "exact_rational_center_plus_radius_equals_corrected_upper": exact_upper,
        "frozen_interval_precision_bits": 53,
        "inference": "With exact binary64 center/radius and frozen MPFR-53 outward Interval addition, the corrected residual endpoints are contained; the compound first refusal therefore points to the separate full-control versus RPC-box affine-reference comparison.",
        "trace_limit": "The frozen Real ostream changes stream precision to 15 digits before the interval columns; the saved CSV does not identify which whole-control comparison failed or its exact ULP gap.",
        "semantic_limit": "An RPC input box hull can be broader than the correlated Taylor-model input image. Failure to contain that independent box-arithmetic reference does not by itself prove that the constructed control TM misses a real input or NN output.",
        "status": "UNDECIDED_AFTER_FIRST_REFUSAL",
        "no_digest_verification": True,
    }
    (HERE / "ANALYSIS.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({"first_refusal": analysis["first_refusal"],
                      "status": analysis["status"]}))


if __name__ == "__main__":
    main()
