"""Reproduce the frozen first QUAD split/recenter endpoints; no solver or digest."""

from fractions import Fraction
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
RPC = RUNS / "native_quad_var_tail_repair_gate_20261002_007/quad/on/rpc.json"
ORIGINAL_RPC = RUNS / "native_quad_sr_octagon_gate_20261002_005/on/rpc.json"


def round53(value, direction):
    """Round an exact rational to binary64 like 53-bit MPFR, for this finite case."""
    nearest = float(value)
    exact = Fraction.from_float(nearest)
    if direction == "down" and exact > value:
        nearest = math.nextafter(nearest, -math.inf)
    elif direction == "up" and exact < value:
        nearest = math.nextafter(nearest, math.inf)
    return Fraction.from_float(nearest)


def describe(value):
    return {"hex": float(value).hex(), "exact_fraction": str(value)}


def main():
    rpc = json.loads(RPC.read_text())
    lower = rpc["params"]["input_lb"]
    upper = rpc["params"]["input_ub"]
    assert len(lower) == len(upper) == 12

    original_lower = Fraction.from_float(-0.4)
    original_upper = Fraction.from_float(0.4)
    width = round53(original_upper - original_lower, "up")
    increment = round53(width / 8, "up")
    split_upper = round53(original_lower + increment, "up")
    midpoint_sum = round53(original_lower + split_upper, "nearest")
    center = round53(midpoint_sum / 2, "nearest")
    radius = round53(split_upper - center, "up")
    represented_lower = round53(center - radius, "down")
    represented_upper = round53(center + radius, "up")
    rpc_lowers = [Fraction.from_float(lower[i]) for i in range(3)]
    rpc_uppers = [Fraction.from_float(upper[i]) for i in range(3)]
    original_rpc = json.loads(ORIGINAL_RPC.read_text())
    original_rpc_lowers = [Fraction.from_float(x)
                           for x in original_rpc["params"]["input_lb"][:3]]
    assert all(value == represented_lower for value in rpc_lowers)
    assert original_rpc_lowers == rpc_lowers
    assert all(value == represented_upper for value in rpc_uppers)
    assert original_lower < represented_lower
    assert represented_lower - original_lower == Fraction.from_float(math.ulp(-0.4))

    print(json.dumps({
        "scope": "first split of x1, x2, x3 only; frozen source and saved repair RPC",
        "arithmetic": "53-bit binary MPFR operations reproduced with exact Fraction and binary64 rounding",
        "original_interval_lower": describe(original_lower),
        "original_interval_upper": describe(original_upper),
        "first_split_upper": describe(split_upper),
        "recenter_center": describe(center),
        "recenter_radius": describe(radius),
        "recentered_tm_lower": describe(represented_lower),
        "recentered_tm_upper": describe(represented_upper),
        "saved_rpc_first_three_lower": [describe(value) for value in rpc_lowers],
        "saved_original_gate_rpc_first_three_lower": [describe(value) for value in original_rpc_lowers],
        "saved_rpc_first_three_upper": [describe(value) for value in rpc_uppers],
        "missing_lower_strip_width": describe(represented_lower - original_lower),
        "whole_declared_first_box_enclosed_by_recentered_tm": False,
        "other_boxes_checked": False,
    }, indent=2))


if __name__ == "__main__":
    main()
