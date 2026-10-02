#!/usr/bin/env python3
"""Per-box outward binary32 bias corrections from the saved QUAD CROWN batch.

This is conditional on ideal real-arithmetic CROWN affine inequalities.
It does not call CROWN, a network, or the plant.
"""

import csv
import json
from decimal import Decimal, localcontext
from fractions import Fraction
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD_RPC = ROOT / "native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json"
SLOPES = ROOT / "native_quad_crown_same_slope_batch_20261003_001/RAW_COEFFICIENTS.json"


def q(value):
    return Fraction.from_float(value)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def adjacent_f32(value, direction):
    assert math.isfinite(value) and direction in (-1, 1)
    if value == 0:
        bits = 0x00000001 if direction > 0 else 0x80000001
    else:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        bits += direction if value > 0 else -direction
    answer = struct.unpack("<f", struct.pack("<I", bits))[0]
    assert math.isfinite(answer) and (answer > value if direction > 0 else answer < value)
    return answer


def outward_f32(target, direction):
    """Greatest binary32 <= target, or least binary32 >= target."""
    candidate = f32(float(target))
    if direction < 0:
        while q(candidate) > target:
            candidate = adjacent_f32(candidate, -1)
        while q(adjacent_f32(candidate, 1)) <= target:
            candidate = adjacent_f32(candidate, 1)
        assert q(candidate) <= target < q(adjacent_f32(candidate, 1))
    else:
        while q(candidate) < target:
            candidate = adjacent_f32(candidate, 1)
        while q(adjacent_f32(candidate, -1)) >= target:
            candidate = adjacent_f32(candidate, -1)
        assert q(adjacent_f32(candidate, -1)) < target <= q(candidate)
    return candidate


def show(value):
    with localcontext() as context:
        context.prec = 26
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def quantiles(values):
    values = sorted(values)
    n = len(values)
    return {key: show(values[(n - 1) * part // 100]) for key, part in
            (("min", 0), ("p25", 25), ("median", 50), ("p75", 75),
             ("p90", 90), ("p95", 95), ("p99", 99), ("max", 100))}


def main():
    rpc = json.loads(OLD_RPC.read_text())
    raw = json.loads(SLOPES.read_text())
    old = rpc["coefficients"]
    assert raw["lA"] == raw["uA"] == old["T"]
    assert raw["lbias"] == old["u_min"] and raw["ubias"] == old["u_max"]
    lo = rpc["params"]["input_lb"]
    hi = rpc["params"]["input_ub"]
    assert len(lo) == len(hi) == 1024 * 12

    rows = []
    corrected_lower = []
    corrected_upper = []
    changed_by_lane = []
    per_output = [{"real_lower": [], "real_upper": [], "f32_lower": [],
                   "f32_upper": [], "f32_width": [], "relative_width": []}
                  for _ in range(3)]
    for lane in range(1024):
        box = [(q(lo[lane * 12 + i]), q(hi[lane * 12 + i])) for i in range(12)]
        assert all(a <= b for a, b in box)
        new_lo_row = []
        new_hi_row = []
        changed_outputs = 0
        for output in range(3):
            delta_min = Fraction(0)
            delta_max = Fraction(0)
            for slope, (a, b) in zip(old["T"][lane][output], box):
                d = q(slope) - q(f32(slope))
                delta_min += min(d * a, d * b)
                delta_max += max(d * a, d * b)
            raw_lb = q(old["u_min"][lane][output])
            raw_ub = q(old["u_max"][lane][output])
            native_lb = f32(old["u_min"][lane][output])
            native_ub = f32(old["u_max"][lane][output])
            required_lb = raw_lb + delta_min
            required_ub = raw_ub + delta_max
            chosen_lb = min(native_lb, outward_f32(required_lb, -1))
            chosen_ub = max(native_ub, outward_f32(required_ub, 1))
            assert q(chosen_lb) <= required_lb and q(chosen_ub) >= required_ub
            assert q(chosen_lb) <= q(native_lb) <= q(native_ub) <= q(chosen_ub)
            real_lower = max(Fraction(0), q(native_lb) - required_lb)
            real_upper = max(Fraction(0), required_ub - q(native_ub))
            f32_lower = q(native_lb) - q(chosen_lb)
            f32_upper = q(chosen_ub) - q(native_ub)
            original_width = q(native_ub) - q(native_lb)
            assert original_width > 0
            f32_width = f32_lower + f32_upper
            changed_outputs += f32_width > 0
            slot = per_output[output]
            for key, value in (("real_lower", real_lower), ("real_upper", real_upper),
                               ("f32_lower", f32_lower), ("f32_upper", f32_upper),
                               ("f32_width", f32_width),
                               ("relative_width", f32_width / original_width)):
                slot[key].append(value)
            rows.append({
                "lane": lane, "output": output + 1,
                "native_lower_f32": native_lb, "native_upper_f32": native_ub,
                "required_lower_real": show(required_lb),
                "required_upper_real": show(required_ub),
                "corrected_lower_f32": chosen_lb, "corrected_upper_f32": chosen_ub,
                "minimum_real_lower_expansion": show(real_lower),
                "minimum_real_upper_expansion": show(real_upper),
                "minimum_f32_lower_expansion": show(f32_lower),
                "minimum_f32_upper_expansion": show(f32_upper),
                "f32_width_increase": show(f32_width),
                "f32_relative_width_increase": show(f32_width / original_width),
            })
            new_lo_row.append(chosen_lb)
            new_hi_row.append(chosen_ub)
        corrected_lower.append(new_lo_row)
        corrected_upper.append(new_hi_row)
        changed_by_lane.append(changed_outputs)
    assert len(rows) == 1024 * 3

    with (HERE / "CORRECTIONS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "CORRECTED_BIASES.json").write_text(json.dumps({
        "u_min": corrected_lower, "u_max": corrected_upper,
        "scope": "conditional outward binary32 endpoints for saved first CROWN batch",
    }, allow_nan=False) + "\n")
    summary = []
    for output, slot in enumerate(per_output):
        summary.append({
            "output": output + 1,
            "boxes_requiring_lower_change": sum(v > 0 for v in slot["f32_lower"]),
            "boxes_requiring_upper_change": sum(v > 0 for v in slot["f32_upper"]),
            "boxes_requiring_either_change": sum(
                a > 0 or b > 0 for a, b in zip(slot["f32_lower"], slot["f32_upper"])),
            "minimum_real_lower_expansion_distribution": quantiles(slot["real_lower"]),
            "minimum_real_upper_expansion_distribution": quantiles(slot["real_upper"]),
            "minimum_f32_lower_expansion_distribution": quantiles(slot["f32_lower"]),
            "minimum_f32_upper_expansion_distribution": quantiles(slot["f32_upper"]),
            "minimum_f32_width_increase_distribution": quantiles(slot["f32_width"]),
            "f32_relative_width_increase_distribution": quantiles(slot["relative_width"]),
            "max_f32_lower_lane": max(range(1024), key=lambda lane: slot["f32_lower"][lane]),
            "max_f32_upper_lane": max(range(1024), key=lambda lane: slot["f32_upper"][lane]),
        })
    result = {
        "scope": "saved QUAD first 1024-box CROWN call, 3072 output rows; no new solver/model/plant execution",
        "basis": "recorded uA=lA and old lA/lbias/ubias numerically identical in one same-batch controller replay",
        "condition": "assume the original real-arithmetic CROWN affine inequalities are sound on the recorded RPC boxes",
        "formula": "required bias interval = [lbias + min_X((lA-f32(lA))·x), ubias + max_X((uA-f32(lA))·x)]",
        "selection": "keep old transported endpoint if already outward; otherwise choose nearest outward binary32 endpoint",
        "arithmetic": "all signs and extrema use exact Fractions lifted from parsed binary64 and chosen binary32 values",
        "source_boxes_with_any_correction": sum(n > 0 for n in changed_by_lane),
        "source_boxes_with_all_three_outputs_corrected": sum(n == 3 for n in changed_by_lane),
        "output_rows_with_correction": sum(changed_by_lane),
        "per_output": summary,
        "lane0": rows[:3],
        "limits": "A conditional correction plan, not independent CROWN/NN soundness or native Taylor-model enclosure. Plant first-step receipts must be repeated if controls are widened.",
    }
    (HERE / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"boxes": 1024, "rows": len(rows), "per_output": [{
        "output": v["output"],
        "boxes_requiring_either_change": v["boxes_requiring_either_change"],
        "max_f32_width_increase": v["minimum_f32_width_increase_distribution"]["max"],
    } for v in summary]}, indent=2))


if __name__ == "__main__":
    main()
