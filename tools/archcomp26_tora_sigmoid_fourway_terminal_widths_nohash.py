#!/usr/bin/env python3
"""Project audited official-2026 TORA sigmoid terminal boxes to one CSV."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "docs/evidence/results/archcomp26_20261001"
SOURCES = (
    ("Huan", "tora_reach_sigmoid_official2026_mat_u11_full500_huan_002/INDEPENDENT_INTERVAL_SCAN.json",
     "last_saved_endpoint_by_state"),
    ("Xiangru", "tora_reach_sigmoid_official2026_mat_u11_xiangru_full500_001/INDEPENDENT_INTERVAL_SCAN.json",
     "last_saved_endpoint_by_state"),
    ("P3", "tora_reach_sigmoid_official2026_mat_u11_p3_full500_001/INDEPENDENT_INTERVAL_SCAN.json",
     "last_saved_endpoint_by_state"),
    ("Flow* native", "native_tora_reach_sigmoid_u11_full10_001/RANGE_SCAN.json",
     "final_endpoint_union"),
)


def main():
    output = RUNS / "tora_reach_sigmoid_official2026_mat_u11_terminal_fourway.csv"
    with output.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("method", "state", "lower", "upper", "absolute_width", "source_scan"))
        for method, relative, key in SOURCES:
            path = RUNS / relative
            scan = json.loads(path.read_text())
            if not (scan.get("full_numerical_horizon_completed", True) and
                    scan.get("records", 500) == 500):
                raise ValueError(f"incomplete saved ranges: {path}")
            value = scan[key]
            intervals = [value[f"x{i}"] for i in range(1, 5)] if isinstance(value, dict) else value
            if len(intervals) != 4:
                raise ValueError(f"expected four physical states: {path}")
            for index, (lower, upper) in enumerate(intervals, 1):
                if lower > upper:
                    raise ValueError(f"reversed range: {path}")
                writer.writerow((method, f"x{index}", repr(lower), repr(upper),
                                 repr(upper - lower), path.relative_to(ROOT)))
    print(output)


if __name__ == "__main__":
    main()
