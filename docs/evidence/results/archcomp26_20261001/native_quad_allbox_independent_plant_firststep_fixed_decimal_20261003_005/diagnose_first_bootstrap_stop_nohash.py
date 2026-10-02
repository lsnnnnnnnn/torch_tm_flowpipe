#!/usr/bin/env python3
"""Explain only the saved lane-7 bootstrap stop; no expanded box sweep."""

import json
from pathlib import Path

import check_saved_boxes_nohash as checker


HERE = Path(__file__).resolve().parent
ALG = checker.algebra
LANE = 7


def pair(value, limit):
    return {"bound_exact": str(value), "limit_exact": str(limit),
            "bound_approx": float(value), "limit_approx": float(limit),
            "strict_pass": value < limit}


def main():
    saved = checker.load_saved()
    box, matrix, controls, _, _, control_bounds = checker.contract(
        LANE, saved[0], saved[1], saved[2])
    H, G, MASS, INERTIA = ALG.H, ALG.G, ALG.MASS, ALG.INERTIA
    V, ANGLE, RATE10, RATE11 = ALG.V, ALG.ANGLE, ALG.RATE10, ALG.RATE11
    a4, a5, a6 = (ALG.ceil_abs(box[i]) for i in (3, 4, 5))
    u1_max = ALG.ceil_abs(control_bounds[0])
    b4 = RATE11 * V + G * ANGLE
    b5 = RATE10 * V + G * ANGLE
    other = RATE11 * V + RATE10 * V + G * ANGLE**2
    b6 = other + u1_max / MASS
    b7 = RATE10 + ANGLE**2 * RATE11 / (1 - ANGLE**2 / 2)
    b8 = RATE11
    inequalities = {
        "x10_rate": pair(H * ALG.ceil_abs(control_bounds[1]) / INERTIA, RATE10),
        "x11_rate": pair(H * ALG.ceil_abs(control_bounds[2]) / INERTIA, RATE11),
        "x4": pair(a4 + H * b4, V),
        "x5": pair(a5 + H * b5, V),
        "x6": pair(a6 + H * b6, V),
        "x7": pair(H * b7, ANGLE),
        "x8": pair(H * b8, ANGLE),
    }
    result = {
        "schema": "native-quad-fixed-decimal-lane7-bootstrap-stop-nohash-v1",
        "lane": LANE,
        "source_box_x1_to_x6_exact": [[str(value) for value in pair]
                                       for pair in box[:6]],
        "saved_rpc_control_hulls_exact": [[str(value) for value in pair]
                                          for pair in control_bounds],
        "bootstrap_inequalities": inequalities,
        "failed_inequalities": [name for name, data in inequalities.items()
                                if not data["strict_pass"]],
        "meaning": "The fixed first-box bootstrap constants cannot close this lane. No Picard substep or saved Flow* comparison ran for lane 7; no actual trajectory or Flow* counterexample is inferred.",
    }
    (HERE / "FIRST_UNDECIDED.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"lane": LANE,
                      "failed_inequalities": result["failed_inequalities"],
                      "values": {name: inequalities[name] for name in result["failed_inequalities"]}}))
    return 0 if result["failed_inequalities"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
