#!/usr/bin/env python3
"""Project the four audited paper-Unicycle terminal physical boxes to CSV."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "docs/evidence/results/archcomp26_20261001"
AUTHOR = ROOT / "unicycle_paper_speed_w_constant_v1"
SOURCES = (
    ("Huan", AUTHOR / "huan_full50_001/INDEPENDENT_INTERVAL_SCAN.json", "last_saved_endpoint"),
    ("Xiangru", AUTHOR / "xiangru_full50_001/INDEPENDENT_INTERVAL_SCAN.json", "last_saved_endpoint"),
    ("P3", AUTHOR / "p3_full50_001/INDEPENDENT_AUDIT.json", "terminal_physical_endpoint"),
    ("Flow* native", ROOT / "native_unicycle_paper_speed_full50_001/FULL_AUDIT.json", "terminal_physical_endpoint"),
)


def main():
    output = AUTHOR / "terminal_fourway.csv"
    with output.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("method", "state", "lower", "upper", "absolute_width", "source_scan"))
        for method, path, key in SOURCES:
            scan = json.loads(path.read_text())
            if not (scan.get("complete_numerical_horizon", True) and
                    scan.get("full_50_periods_500_substeps", True) and
                    scan.get("rows", 500) == 500):
                raise ValueError(f"incomplete source scan: {path}")
            endpoints = scan[key][:4]
            if len(endpoints) != 4:
                raise ValueError(f"expected four physical states: {path}")
            for index, (lower, upper) in enumerate(endpoints, 1):
                if lower > upper:
                    raise ValueError(f"reversed terminal range: {path}")
                writer.writerow((method, f"x{index}", repr(lower), repr(upper),
                                 repr(upper - lower), path.relative_to(ROOT.parents[3])))
    print(output)


if __name__ == "__main__":
    main()
