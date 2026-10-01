#!/usr/bin/env python3
"""Summarize saved ACC participant-order range records without content digests."""

import csv
import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
STATES = ("x_lead", "v_lead", "a_lead", "x_ego", "v_ego", "a_ego")
METHODS = (
    ("native", "acc_native_var_tail_full50_001", "native_local_tm_range"),
    ("huan", "acc_huan_full50_001", "gpu_shared_driver_local_tm_range"),
    ("xiangru", "acc_xiangru_full50_001", "gpu_shared_driver_local_tm_range"),
    ("ours_p3", "acc_p3_full50_001", "gpu_shared_driver_local_tm_range_p3_engine"),
)
FORMAT = struct.Struct("<QQd" + "d" * 24)


def check_pair(pair):
    lo, hi = pair
    if not all(math.isfinite(x) for x in pair) or lo > hi:
        raise ValueError(f"invalid interval: {pair}")


def read_native(path):
    raw = path.read_bytes()
    if len(raw) != 50 * FORMAT.size:
        raise ValueError(f"native record bytes: {len(raw)}")
    rows = []
    for index, fields in enumerate(FORMAT.iter_unpack(raw), 1):
        lane, step, h, *values = fields
        if (lane, step, h) != (0, index, 0.1):
            raise ValueError(f"native index/step/h mismatch at record {index}")
        tube = [values[4 * k : 4 * k + 2] for k in range(6)]
        endpoint = [values[4 * k + 2 : 4 * k + 4] for k in range(6)]
        rows.append((tube, endpoint))
    return rows


def read_gpu(path):
    rows = []
    with path.open() as source:
        for index, line in enumerate(source, 1):
            item = json.loads(line)
            if (item["substep"], item["accepted"], item["local_h"]) != (
                index, True, 0.1
            ):
                raise ValueError(f"GPU index/status/h mismatch at record {index}")
            rows.append((item["tube"], item["endpoint"]))
    if len(rows) != 50:
        raise ValueError(f"GPU record count: {len(rows)}")
    return rows


def summarize(rows):
    if len(rows) != 50:
        raise ValueError("expected exactly 50 periods")
    for tube, endpoint in rows:
        if len(tube) != 6 or len(endpoint) != 6:
            raise ValueError("expected six physical states")
        for pair in tube + endpoint:
            if len(pair) != 2:
                raise ValueError("expected lo/hi interval")
            check_pair(pair)
    return [
        (
            rows[-1][1][k][0],
            rows[-1][1][k][1],
            rows[-1][1][k][1] - rows[-1][1][k][0],
            min(row[0][k][0] for row in rows),
            max(row[0][k][1] for row in rows),
        )
        for k in range(6)
    ]


def main():
    by_method = {}
    for method, directory, observer in METHODS:
        source = RUNS / directory / ("ranges.bin" if method == "native" else "ranges.jsonl")
        records = read_native(source) if method == "native" else read_gpu(source)
        by_method[method] = (source, observer, summarize(records))

    long_path = HERE / "acc_t5_endpoint_and_full_tube_long.csv"
    with long_path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("method", "observer", "state_index", "state", "source", "periods",
                         "endpoint_t5_lo", "endpoint_t5_hi", "endpoint_t5_width",
                         "tube_t0_t5_lo", "tube_t0_t5_hi"))
        for method, _, _ in METHODS:
            source, observer, values = by_method[method]
            for index, (state, value) in enumerate(zip(STATES, values)):
                writer.writerow((method, observer, index, state, source.relative_to(RUNS), 50,
                                 *(repr(v) for v in value)))

    wide_path = HERE / "acc_t5_endpoint_and_full_tube_wide.csv"
    fields = ("endpoint_t5_lo", "endpoint_t5_hi", "endpoint_t5_width",
              "tube_t0_t5_lo", "tube_t0_t5_hi")
    with wide_path.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("state_index", "state", *(f"{method}_{field}"
                        for method, _, _ in METHODS for field in fields)))
        for index, state in enumerate(STATES):
            writer.writerow((index, state, *(repr(v) for method, _, _ in METHODS
                            for v in by_method[method][2][index])))
    print(f"{long_path}: 24 data rows")
    print(f"{wide_path}: 6 data rows")


if __name__ == "__main__":
    main()
