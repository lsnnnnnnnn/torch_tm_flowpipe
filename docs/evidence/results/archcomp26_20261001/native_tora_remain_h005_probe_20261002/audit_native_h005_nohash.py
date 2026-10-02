#!/usr/bin/env python3
"""Independently scan one saved native h=0.05 TORA run; no solver or digest."""

import argparse
import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
OLD_BOXES = HERE.parent / "native_tora_remain_full20_001/initial_boxes.json"
ROW = struct.Struct("<QQd16d")
BOXES = 12
SUBSTEPS_PER_PERIOD = 20
H = 0.05


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(values):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in values)


def scan(run, periods):
    steps = SUBSTEPS_PER_PERIOD * periods
    result = read_json(run / "RESULT.json")
    if result.get("status") != "completed" or result.get("exit_code") != 0:
        raise ValueError("supervised run did not complete with exit 0")
    log = (run / "native.log").read_text(encoding="utf-8")
    observed_periods = [int(line.split()[1]) for line in log.splitlines() if line.startswith("Step ")]
    if observed_periods != list(range(periods)) or log.splitlines().count("VERIFIED") != 1:
        raise ValueError("native period sequence/checker label differs")
    if read_json(HERE / "initial_boxes.json") != read_json(OLD_BOXES):
        raise ValueError("actual 12 initial boxes differ from frozen h=0.1 partition")

    rpc_path = run / "controller_rpc.jsonl"
    rpc = [json.loads(line) for line in rpc_path.read_text(encoding="utf-8").splitlines()]
    if len(rpc) != periods:
        raise ValueError("RPC count differs from completed control periods")
    for index, entry in enumerate(rpc):
        lower, upper, response = entry["input_lower"], entry["input_upper"], entry["response"]
        if (len(lower) != 48 or len(upper) != 48 or not finite(lower + upper)
                or any(a > b for a, b in zip(lower, upper))
                or len(response["T"]) != BOXES or len(response["u_min"]) != BOXES
                or len(response["u_max"]) != BOXES):
            raise ValueError(f"invalid RPC input/output shape at period {index}")
        for lane in range(BOXES):
            slope, low, high = response["T"][lane], response["u_min"][lane], response["u_max"][lane]
            if (len(slope) != 1 or len(slope[0]) != 4 or len(low) != 1 or len(high) != 1
                    or not finite(slope[0] + low + high) or low[0] > high[0]):
                raise ValueError(f"invalid RPC output at period {index}, lane {lane}")

    range_path = run / "ranges.bin"
    raw = range_path.read_bytes()
    if len(raw) != BOXES * steps * ROW.size:
        raise ValueError("incomplete 12-box saved range grid")
    seen = set()
    tube_union = [[math.inf, -math.inf] for _ in range(4)]
    final_endpoint = [[math.inf, -math.inf] for _ in range(4)]
    final_width_sum = [0.0] * 4
    final_width_max = [0.0] * 4
    for lane, step, h, *flat in ROW.iter_unpack(raw):
        if (lane >= BOXES or not 1 <= step <= steps or (lane, step) in seen
                or not math.isclose(h, H, rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"invalid lane/step/h: {lane}/{step}/{h}")
        seen.add((lane, step))
        for i in range(4):
            lo, hi, end_lo, end_hi = flat[4*i:4*i+4]
            if (not finite((lo, hi, end_lo, end_hi)) or not lo <= end_lo <= end_hi <= hi
                    or lo < -2 or hi > 2):
                raise ValueError(f"invalid or unsafe saved tube at {lane}/{step}/x{i+1}")
            tube_union[i][0] = min(tube_union[i][0], lo)
            tube_union[i][1] = max(tube_union[i][1], hi)
            if step == steps:
                final_endpoint[i][0] = min(final_endpoint[i][0], end_lo)
                final_endpoint[i][1] = max(final_endpoint[i][1], end_hi)
                width = end_hi - end_lo
                final_width_sum[i] += width
                final_width_max[i] = max(final_width_max[i], width)
    if len(seen) != BOXES * steps:
        raise ValueError("missing lane/step pair")
    return {"schema": "archcomp26-tora-remain-native-h005-saved-audit-nohash-v1",
            "run": str(run.relative_to(HERE)), "periods": periods, "step_s": H,
            "substeps": steps, "range_records": len(seen), "lanes": BOXES,
            "complete_grid": True, "all_saved_components_finite_ordered_endpoint_contained_and_safe": True,
            "initial_boxes_equal_frozen_h01_partition": True,
            "rpc_count": len(rpc), "native_checker_label": "VERIFIED",
            "saved_tube_union": tube_union, "final_endpoint_union": final_endpoint,
            "final_endpoint_partition_mean_width": [x / BOXES for x in final_width_sum],
            "final_endpoint_partition_max_width": final_width_max,
            "range_source_bytes": range_path.stat().st_size,
            "interpretation": "saved numeric range audit plus adjacent native checker label; no independent end-to-end floating-point NNCS proof"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--periods", type=int, choices=(1, 20), required=True)
    args = parser.parse_args()
    print(json.dumps(scan(args.run.resolve(), args.periods), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
