#!/usr/bin/env python3
"""Read the saved first QUAD RPC and quantify an interval-only NN gate."""

import json
import math
from pathlib import Path
import statistics
import struct


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def summary(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def main():
    rpc = json.loads(SOURCE.read_text())
    inp, coeff = rpc["params"], rpc["coefficients"]
    lb, ub = inp["input_lb"], inp["input_ub"]
    assert len(lb) == len(ub) == 1024 * 12
    assert all(len(coeff[key]) == 1024 for key in ("T", "u_min", "u_max"))
    linear_widths = [[], [], []]
    residual_widths = [[], [], []]
    first = []
    for lane in range(1024):
        assert all(math.isfinite(a) and math.isfinite(b) and a <= b
                   for a, b in zip(lb[12*lane:12*lane+12], ub[12*lane:12*lane+12]))
        assert len(coeff["T"][lane]) == 3
        for output in range(3):
            row = coeff["T"][lane][output]
            assert len(row) == 12 and all(math.isfinite(value) for value in row)
            t = [f32(value) for value in row]
            low = f32(coeff["u_min"][lane][output])
            high = f32(coeff["u_max"][lane][output])
            assert math.isfinite(low) and math.isfinite(high) and low <= high
            linear = sum(abs(t[i]) * (ub[12*lane+i] - lb[12*lane+i])
                         for i in range(12))
            residual = high - low
            linear_widths[output].append(linear)
            residual_widths[output].append(residual)
            if lane == 0:
                first.append({"output": output + 1, "linear_width": linear,
                              "transported_residual_lower": low,
                              "transported_residual_upper": high,
                              "transported_residual_width": residual,
                              "linear_to_residual_width_ratio": linear / residual})
    result = {
        "scope": "saved 1024-box first CROWN RPC only; no NN evaluation or certificate",
        "source_rpc": str(SOURCE.relative_to(HERE.parent)),
        "boxes": 1024,
        "input_dimension": 12,
        "output_dimension": 3,
        "transport": "each T, u_min and u_max component converted to binary32 as in saved native C++",
        "first_box": first,
        "linear_width_by_output": [summary(values) for values in linear_widths],
        "residual_width_by_output": [summary(values) for values in residual_widths],
        "first_output_all_boxes_direct_interval_width_exceeds_saved_residual":
            all(linear > residual for linear, residual in
                zip(linear_widths[0], residual_widths[0])),
    }
    (HERE / "AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"first_box": first,
                      "first_output_linear": result["linear_width_by_output"][0],
                      "first_output_residual": result["residual_width_by_output"][0]},
                     allow_nan=False))


if __name__ == "__main__":
    main()
