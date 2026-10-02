#!/usr/bin/env python3
"""Collect already saved paper-QUAD terminal bounds for twelve physical states."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "docs/evidence/results/archcomp26_20261001"
OUTPUT = RUNS / "quad_paper_fourway_saved_20261002/terminal_12states_fourway.csv"
SOURCES = {
    "Huan": RUNS / "quad_paper_huan_full50_001/parity/metrics.json",
    "Xiangru": RUNS / "quad_paper_xiangru_v1/full50_001/data/metrics.json",
    "ours/P3 driver": RUNS / "quad_paper_p3_nohash_v1/full50_001/data/RESULT.json",
}
NATIVE = RUNS / "native_quad_paper_full50_001/SCAN.json"
P3_OBSERVED = RUNS / "quad_paper_p3_nohash_v1/full50_001/data/observations.jsonl"


def append_rows(table, method, source_object, intervals, source):
    if len(intervals) != 12:
        raise ValueError(f"{method}: expected twelve physical states")
    for index, pair in enumerate(intervals, 1):
        lo, hi = pair
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError(f"{method}: invalid x{index}")
        table.append({
            "method": method, "state": f"x{index}",
            "endpoint_lo": lo, "endpoint_hi": hi, "endpoint_width": hi - lo,
            "source_object": source_object, "source": str(source.relative_to(ROOT)),
        })


def main():
    table = []
    for method, source in SOURCES.items():
        document = json.loads(source.read_text())
        if method == "ours/P3 driver":
            if document["status"] != "completed" or document["completed_substeps"] != 1000:
                raise ValueError("P3 driver did not complete")
        elif (document["B"] != 1024 or document["steps"] != 50
              or document["substeps"] != 20 or document["broken"] != 0):
            raise ValueError(f"{method}: incomplete metrics")
        hull = document["final_hull"]
        append_rows(table, method, "driver_final_hull",
                    [hull[f"x{i}"] for i in range(1, 13)], source)

    native = json.loads(NATIVE.read_text())
    if native["record_count"] != 1024000 or not native["complete_unique_ordered_grid"]:
        raise ValueError("native raw range scan incomplete")
    terminal = native["terminal"]
    if [row["state"] for row in terminal] != list(range(1, 13)):
        raise ValueError("native terminal state order changed")
    append_rows(table, "Flow* native", "scanned_saved_endpoint_union",
                [[row["endpoint_lo"], row["endpoint_hi"]] for row in terminal], NATIVE)

    observations = [json.loads(line) for line in P3_OBSERVED.read_text().splitlines()]
    if len(observations) != 1000 or observations[-1]["substep"] != 1000:
        raise ValueError("P3 saved observer incomplete")
    append_rows(table, "ours/P3 observer", "last_saved_observer_endpoint_union",
                [row[2:4] for row in observations[-1]["tube_endpoint_union_12x4"]],
                P3_OBSERVED)

    if len(table) != 60:
        raise ValueError("expected four methods plus P3 observer x twelve states")
    with OUTPUT.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=table[0])
        writer.writeheader()
        writer.writerows(table)
    print(OUTPUT)


if __name__ == "__main__":
    main()
