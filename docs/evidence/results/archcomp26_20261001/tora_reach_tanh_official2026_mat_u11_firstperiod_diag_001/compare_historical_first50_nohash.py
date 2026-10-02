#!/usr/bin/env python3
"""Compare saved TORA tanh intervals with the historical full Huan run."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import struct


ROW = struct.Struct("<QQd16d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("historical_root", type=Path)
    args = parser.parse_args()
    historical = args.historical_root.resolve()
    root = Path(__file__).resolve().parent
    old_csv = historical / "evidence_v2/width_trajectory_tora_relu_tanh_matched.csv.gz"
    old = {}
    with gzip.open(old_csv, "rt", newline="") as source:
        for row in csv.DictReader(source):
            if row["arm"] != "huan" or int(row["step"]) > 50:
                continue
            key = (int(row["step"]), int(row["coordinate"]), row["view"])
            if key in old:
                raise ValueError("duplicate historical interval")
            old[key] = (float(row["union_lo"]), float(row["union_hi"]))
    if len(old) != 50 * 4 * 2:
        raise ValueError("missing historical first-period intervals")

    raw = (root / "ranges.bin").read_bytes()
    if len(raw) != 50 * ROW.size:
        raise ValueError("new first-period interval count differs")
    equal, count, maximum_difference = 0, 0, 0.0
    for index in range(50):
        lane, step, h, *values = ROW.unpack_from(raw, index * ROW.size)
        if (lane, step) != (0, index + 1) or abs(h - 0.01) > 1e-12:
            raise ValueError("new first-period lane, step, or h differs")
        for coordinate in range(1, 5):
            for view, offset in (("tube", 0), ("endpoint", 2)):
                previous = old[(step, coordinate, view)]
                current = values[(coordinate - 1) * 4 + offset:
                                 (coordinate - 1) * 4 + offset + 2]
                for a, b in zip(current, previous):
                    count += 1
                    equal += a == b
                    maximum_difference = max(maximum_difference, abs(a - b))

    full_runs = {}
    for name, folder in (("huan", "tora_relu_tanh_huan"),
                         ("xiangru", "tora_relu_tanh_xiangru"),
                         ("p3", "tora_relu_tanh_ours")):
        path = historical / "evidence_v1/suite_v1" / folder / "result.json"
        data = json.loads(path.read_text())
        steps = data["steps"]
        full_runs[name] = {
            "result_path": str(path), "status": data["status"],
            "complete": data["complete"], "attempted_steps": data["attempted_steps"],
            "accepted_lane_steps": data["accepted_lane_steps"],
            "expected_steps": data["expected_steps"],
            "step_records": len(steps),
            "all_step_records_accepted": all(row["accepted"] == [True] for row in steps),
        }
    native_path = historical / "evidence_v2/native_matched/tora_relu_tanh/result.json"
    native_log = native_path.with_name("native.log")
    native = json.loads(native_path.read_text())
    full_runs["native"] = {
        "result_path": str(native_path), "status": native["status"],
        "complete": native["complete"],
        "control_periods_started": native["control_periods_started"],
        "range_records": native["range_records"],
        "native_log_path": str(native_log),
        "author_log_verdict_line": next(
            (line for line in native_log.read_text().splitlines()
             if line in {"VERIFIED", "UNKNOWN", "FALSIFIED"}), None),
    }
    receipt = {
        "kind": "read_only_historical_tanh_reuse_audit",
        "new_firstperiod_ranges": str(root / "ranges.bin"),
        "historical_huan_interval_csv": str(old_csv),
        "compared_interval_endpoints": count,
        "exact_equal_endpoints": equal,
        "maximum_abs_difference": maximum_difference,
        "historical_full_numerical_runs": full_runs,
        "new_full_run_started": False,
        "content_digest_computed": False,
    }
    with (root / "HISTORICAL_REUSE_AUDIT.json").open("x") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"compared": count, "equal": equal,
                      "maximum_difference": maximum_difference,
                      "historical_full_methods": list(full_runs)}, indent=2))


if __name__ == "__main__":
    main()
