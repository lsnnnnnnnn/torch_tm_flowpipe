"""Stream the original native QUAD observer file; no content digest is used."""

import json
import math
import os
import struct
import sys


def scan(path):
    record = struct.Struct("<QQd48d")
    expected_boxes, expected_steps, dims = 1024, 1000, 12
    pools = [[[math.inf, -math.inf, math.inf, -math.inf] for _ in range(dims)]
             for _ in range(expected_steps)]
    all_time = [[math.inf, -math.inf, math.inf, -math.inf] for _ in range(dims)]
    terminal_widths = [[[], []] for _ in range(dims)]
    errors = {key: 0 for key in ("sequence", "step_or_box", "h", "nonfinite",
                                 "reversed_tube", "reversed_endpoint", "endpoint_outside_tube")}
    samples = []
    count = 0

    with open(path, "rb", buffering=1024 * 1024) as source:
        while chunk := source.read(record.size):
            if len(chunk) != record.size:
                errors["partial_record"] = len(chunk)
                break
            box, step, h, *values = record.unpack(chunk)
            period, in_period = divmod(count, expected_boxes * 20)
            expected = (in_period // 20, period * 20 + in_period % 20 + 1)
            if (box, step) != expected:
                errors["sequence"] += 1
                if len(samples) < 10:
                    samples.append({"record": count, "actual": [box, step], "expected": expected})
            if box >= expected_boxes or not 1 <= step <= expected_steps:
                errors["step_or_box"] += 1
                count += 1
                continue
            if h != 0.005:
                errors["h"] += 1
            pooled = pools[step - 1]
            for state in range(dims):
                tube_lo, tube_hi, end_lo, end_hi = values[4 * state:4 * state + 4]
                if not all(map(math.isfinite, (tube_lo, tube_hi, end_lo, end_hi))):
                    errors["nonfinite"] += 1
                    continue
                if tube_lo > tube_hi:
                    errors["reversed_tube"] += 1
                if end_lo > end_hi:
                    errors["reversed_endpoint"] += 1
                if not (tube_lo <= end_lo <= end_hi <= tube_hi):
                    errors["endpoint_outside_tube"] += 1
                for target in (pooled[state], all_time[state]):
                    target[0] = min(target[0], tube_lo)
                    target[1] = max(target[1], tube_hi)
                    target[2] = min(target[2], end_lo)
                    target[3] = max(target[3], end_hi)
                if step == expected_steps:
                    terminal_widths[state][0].append(tube_hi - tube_lo)
                    terminal_widths[state][1].append(end_hi - end_lo)
            count += 1

    def bounds(value):
        return {"tube_lo": value[0], "tube_hi": value[1],
                "endpoint_lo": value[2], "endpoint_hi": value[3],
                "tube_union_width": value[1] - value[0],
                "endpoint_union_width": value[3] - value[2]}

    terminal = []
    for state, value in enumerate(pools[-1]):
        row = {"state": state + 1, **bounds(value)}
        for kind, widths in zip(("tube", "endpoint"), terminal_widths[state]):
            row[f"{kind}_per_box_width_mean"] = sum(widths) / len(widths) if widths else None
            row[f"{kind}_per_box_width_max"] = max(widths) if widths else None
        terminal.append(row)
    return {
        "schema": "native-quad-paper-range-scan-v1",
        "source_path": path,
        "source_bytes": os.path.getsize(path),
        "record_format": "little-endian uint64 box, uint64 step, float64 h, 12*(tube_lo,tube_hi,endpoint_lo,endpoint_hi) float64",
        "record_size_bytes": record.size,
        "record_count": count,
        "expected_record_count": expected_boxes * expected_steps,
        "complete_unique_ordered_grid": count == expected_boxes * expected_steps and errors["sequence"] == 0 and errors["step_or_box"] == 0,
        "errors": errors,
        "error_samples": samples,
        "terminal_endpoint_x3_within_0_94_1_06": terminal[2]["endpoint_lo"] >= 0.94 and terminal[2]["endpoint_hi"] <= 1.06,
        "all_time": [{"state": state + 1, **bounds(value)} for state, value in enumerate(all_time)],
        "terminal": terminal,
        "per_step": [{"step": step + 1, "t_start": step * 0.005, "t_end": (step + 1) * 0.005,
                      "states": [{"state": state + 1, **bounds(value)} for state, value in enumerate(row)]}
                     for step, row in enumerate(pools)],
    }


if __name__ == "__main__":
    assert struct.calcsize("<QQd48d") == 408
    json.dump(scan(sys.argv[1]), sys.stdout, allow_nan=False, separators=(",", ":"))
    sys.stdout.write("\n")
