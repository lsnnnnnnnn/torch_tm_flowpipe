#!/usr/bin/env python3
"""Reduce saved NAV x/y interval records to compact, source-labelled curves."""

import argparse
import csv
import json
import math
from pathlib import Path
import struct


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923"
NEW = N / "runs/archcomp26_20261001"
ROW = struct.Struct("<QQd16d")
HEAD = ["instance", "method", "source_generation", "step", "t_start", "t_end",
        "state", "tube_lo", "tube_hi", "tube_union_width", "endpoint_lo",
        "endpoint_hi", "endpoint_union_width", "partition_mean_tube_width",
        "partition_max_tube_width", "partition_mean_endpoint_width",
        "partition_max_endpoint_width"]


def methods(instance):
    stem = "nav_standard" if instance == "nav-standard" else "nav_robust"
    saved = [(name, "2026-09-23 historical GPU", OLD / "suite_v1" / f"{stem}_{name}" / "ranges.bin")
             for name in ("ours", "huan", "xiangru")]
    native = (NEW / "nav_author_standard_native_full30_001" / "ranges.bin"
              if instance == "nav-standard" else OLD / "native_matched/nav_robust/ranges.bin")
    saved.append(("flowstar_native", "2026-10-02 new native" if instance == "nav-standard"
                  else "2026-09-23 historical native", native))
    return saved


def reduce_file(instance, method, generation, path, boxes, output):
    if path.stat().st_size != boxes * 600 * ROW.size:
        raise ValueError(f"saved range size differs: {path}")
    next_step = [1] * boxes
    cells = [[[math.inf, -math.inf, math.inf, -math.inf, 0.0, 0.0, 0.0, 0.0]
              for _ in range(2)] for _ in range(600)]
    overlaps = misses = count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(ROW.size):
            lane, step, h, *v = ROW.unpack(chunk)
            count += 1
            if (lane >= boxes or step != next_step[lane] or step > 600 or
                    not math.isfinite(h) or abs(h - 0.01) > 1e-12 or
                    not all(map(math.isfinite, v)) or
                    any(v[i] > v[i + 1] or v[i + 2] > v[i + 3] or
                        v[i + 2] < v[i] or v[i + 3] > v[i + 1]
                        for i in range(0, 16, 4))):
                raise ValueError(f"invalid saved interval: {path}, lane={lane}, step={step}")
            next_step[lane] += 1
            if not (v[0] > 2 or v[1] < 1 or v[4] > 2 or v[5] < 1):
                overlaps += 1
            if step == 600 and not (-0.5 <= v[2] <= v[3] <= 0.5 and
                                    -0.5 <= v[6] <= v[7] <= 0.5):
                misses += 1
            for state in range(2):
                tlo, thi, elo, ehi = v[4 * state:4 * state + 4]
                row = cells[step - 1][state]
                row[0] = min(row[0], tlo)
                row[1] = max(row[1], thi)
                row[2] = min(row[2], elo)
                row[3] = max(row[3], ehi)
                row[4] += thi - tlo
                row[5] = max(row[5], thi - tlo)
                row[6] += ehi - elo
                row[7] = max(row[7], ehi - elo)
    if count != boxes * 600 or any(step != 601 for step in next_step):
        raise ValueError(f"saved lane-step grid incomplete: {path}")
    for step in range(1, 601):
        for state, name in enumerate(("x", "y")):
            tlo, thi, elo, ehi, mean_tube, max_tube, mean_ep, max_ep = cells[step - 1][state]
            output.writerow([instance, method, generation, step, (step - 1) * 0.01,
                             step * 0.01, name, tlo, thi, thi - tlo, elo, ehi,
                             ehi - elo, mean_tube / boxes, max_tube,
                             mean_ep / boxes, max_ep])
    return {"method": method, "generation": generation, "source": str(path),
            "source_bytes": path.stat().st_size, "boxes": boxes, "steps": 600,
            "records": count, "record_bytes": ROW.size,
            "saved_tube_obstacle_overlaps": overlaps,
            "terminal_target_misses": misses,
            "terminal_xy_endpoint_union": {name: cells[-1][i][2:4]
                                           for i, name in enumerate(("x", "y"))},
            "qualification": "saved intervals only; historical/current engines and resources differ"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=("nav-standard", "nav-robust"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    boxes = 640 if args.instance == "nav-standard" else 25
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit = {"schema": "archcomp26-nav-fourway-saved-nohash-v1",
             "instance": args.instance, "initial_boxes_per_method": boxes,
             "record_layout": "little endian uint64 lane, uint64 step, float64 h, then for each of four states (tube_lo,tube_hi,endpoint_lo,endpoint_hi)",
             "source_policy": "paths, byte sizes, and parsed contents only; no content digest",
             "interpretation": "Old GPU ours/Huan/Xiangru versus new standard native or old robust native; no four-way runtime ranking or independent NNCS proof",
             "methods": []}
    with (args.output_dir / "xy_saved_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HEAD)
        for method, generation, path in methods(args.instance):
            audit["methods"].append(reduce_file(args.instance, method, generation,
                                                path, boxes, writer))
    (args.output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"instance": args.instance,
                      "methods": [(r["method"], r["records"], r["saved_tube_obstacle_overlaps"],
                                   r["terminal_target_misses"]) for r in audit["methods"]]}))


if __name__ == "__main__":
    main()
