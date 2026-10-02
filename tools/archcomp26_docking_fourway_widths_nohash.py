#!/usr/bin/env python3
"""Compare four saved full-horizon Docking coordinate intervals, without solving."""

import csv
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "docs/evidence/results/archcomp26_20261001"
OUTPUT = RUNS / "docking_fourway_saved_widths_20261002.csv"
STATES = ("sx", "sy", "vx", "vy")
GPU = {
    "Huan": RUNS / "docking_huan_full40_001/detail/ranges.jsonl",
    "Xiangru": RUNS / "docking_xiangru_full40_001/detail/ranges.jsonl",
    "ours/P3": RUNS / "docking_p3_full40_001/detail/ranges.jsonl",
}
NATIVE = RUNS / "native_docking_full40_001/ranges.bin"
RECORD = struct.Struct("<QQd16d")


def check_pairs(tube, endpoint, step):
    if len(tube) != 4 or len(endpoint) != 4:
        raise ValueError(f"step {step}: expected four physical states")
    for t, e in zip(tube, endpoint):
        if not (len(t) == len(e) == 2 and all(map(math.isfinite, t + e))
                and t[0] <= t[1] and e[0] <= e[1]):
            raise ValueError(f"step {step}: invalid interval")


def read_gpu(path):
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        step = row["substep"]
        if step != len(rows) + 1 or row["accepted"] is not True:
            raise ValueError(f"{path}: missing/rejected step {step}")
        tube, endpoint = row["tube"], row["endpoint"]
        check_pairs(tube, endpoint, step)
        rows.append((tube, endpoint))
    return rows


def read_native():
    payload = NATIVE.read_bytes()
    if len(payload) != 400 * RECORD.size:
        raise ValueError("native range count is not 400")
    rows = []
    for values in RECORD.iter_unpack(payload):
        lane, step, h, *bounds = values
        if lane != 0 or step != len(rows) + 1 or abs(h - 0.1) > 1e-15:
            raise ValueError(f"native: invalid identity at step {step}")
        tube = [[bounds[4 * j], bounds[4 * j + 1]] for j in range(4)]
        endpoint = [[bounds[4 * j + 2], bounds[4 * j + 3]] for j in range(4)]
        check_pairs(tube, endpoint, step)
        rows.append((tube, endpoint))
    return rows


def summarize(method, rows, source):
    if len(rows) != 400:
        raise ValueError(f"{method}: expected 400 full-horizon steps")
    for index, state in enumerate(STATES):
        final_lo, final_hi = rows[-1][1][index]
        tube_lo = min(row[0][index][0] for row in rows)
        tube_hi = max(row[0][index][1] for row in rows)
        yield {
            "method": method, "state": state,
            "endpoint_t40_lo": final_lo, "endpoint_t40_hi": final_hi,
            "endpoint_t40_width": final_hi - final_lo,
            "whole_tube_lo": tube_lo, "whole_tube_hi": tube_hi,
            "whole_tube_width": tube_hi - tube_lo,
            "saved_substeps": len(rows), "property": "UNKNOWN",
            "source": str(source.relative_to(ROOT)),
        }


def main():
    table = []
    for method, path in GPU.items():
        table.extend(summarize(method, read_gpu(path), path))
    table.extend(summarize("Flow* native", read_native(), NATIVE))
    if len(table) != 16:
        raise ValueError("expected four methods x four states")
    with OUTPUT.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=table[0])
        writer.writeheader()
        writer.writerows(table)
    print(OUTPUT)


if __name__ == "__main__":
    main()
