#!/usr/bin/env python3
"""Compare saved TORA remain interval boxes at a common valid prefix."""

import csv
import json
import math
from pathlib import Path
import statistics
import struct


BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RECORD = struct.Struct("<QQd16d")
SOURCES = {
    "native": BASE / "native_tora_remain_full20_001",
    "huan": BASE / "author_tora_remain_v1/huan_full20_001",
    "xiangru": BASE / "author_tora_remain_v1/xiangru_full20_001",
    "ours_p3": BASE / "p3_tora_remain_v1/full20_001",
}
FIELDS = ("method", "window", "window_end_s", "physical_state", "status",
          "tube_lo", "tube_hi", "tube_union_width", "endpoint_lo", "endpoint_hi",
          "endpoint_union_width", "endpoint_lane_mean_width",
          "endpoint_lane_max_width", "min_saved_safe_margin", "source_run_dir")


def read_json(path):
    return json.loads(path.read_text())


def load(method, run_dir):
    is_native = method == "native"
    raw = (run_dir / ("ranges.bin" if is_native else "payload/ranges.bin")).read_bytes()
    assert len(raw) == 2400 * RECORD.size
    records = {}
    for rec in RECORD.iter_unpack(raw):
        lane, step, h, *coordinates = rec
        assert 0 <= lane < 12 and 1 <= step <= 200 and h == .1
        assert (lane, step) not in records
        records[(lane, step)] = coordinates
    assert len(records) == 2400
    observations = None if is_native else [json.loads(line) for line in
        (run_dir / "payload/observations.jsonl").read_text().splitlines()]
    if observations is not None:
        assert len(observations) == 200
        for step in range(1, 185):
            row = observations[step - 1]
            assert row["substep"] == step and row["accepted_count"] == 12
            assert row["rejected_lanes"] == []
            assert row.get("tube_inside_safe_for_accepted",
                           row.get("whole_step_all_lanes_inside_safe")) is True
    return records


def scan(records, end_step):
    union = [[math.inf, -math.inf] for _ in range(4)]
    final = [[math.inf, -math.inf] for _ in range(4)]
    widths = [[] for _ in range(4)]
    minimum_margin = math.inf
    for step in range(1, end_step + 1):
        for lane in range(12):
            values = records[(lane, step)]
            for state in range(4):
                tlo, thi, elo, ehi = values[4*state:4*state+4]
                assert all(map(math.isfinite, (tlo, thi, elo, ehi)))
                assert -2 <= tlo <= elo <= ehi <= thi <= 2
                union[state][0] = min(union[state][0], tlo)
                union[state][1] = max(union[state][1], thi)
                minimum_margin = min(minimum_margin, tlo + 2, 2 - thi)
                if step == end_step:
                    final[state][0] = min(final[state][0], elo)
                    final[state][1] = max(final[state][1], ehi)
                    widths[state].append(ehi - elo)
    return union, final, widths, minimum_margin


def main():
    rows = []
    groups = {}
    for method, run_dir in SOURCES.items():
        records = load(method, run_dir)
        source_result = read_json(run_dir / ("RESULT.json" if method == "native"
                                             else "payload/RESULT.json"))
        if method in ("huan", "xiangru"):
            prior = read_json(run_dir / "INDEPENDENT_INTERVAL_SCAN.json")
            assert prior["first_saved_tube_outside_safe_substep"] == 185
            assert prior["first_rejected_substep"] == 190
            assert source_result["author_checker_interpretation"] == "UNKNOWN"
        else:
            assert source_result["status"] == "completed"
        groups[method] = {}
        for window, end_step in (("common_prefix", 184), ("full_horizon", 200)):
            qualified = window == "common_prefix" or method in ("native", "ours_p3")
            if qualified:
                union, final, widths, margin = scan(records, end_step)
                groups[method][window] = {
                    "end_step": end_step, "tube_union": union,
                    "endpoint_union": final, "min_saved_safe_margin": margin,
                }
            else:
                union = final = widths = None
                margin = None
                groups[method][window] = None
            for i in range(4):
                rows.append({
                    "method": method, "window": window,
                    "window_end_s": end_step / 10, "physical_state": f"x{i+1}",
                    "status": ("all_12_accepted_and_saved_tubes_safe" if qualified
                               else "UNKNOWN_no_full_initial_set_result"),
                    "tube_lo": union[i][0] if qualified else None,
                    "tube_hi": union[i][1] if qualified else None,
                    "tube_union_width": union[i][1] - union[i][0] if qualified else None,
                    "endpoint_lo": final[i][0] if qualified else None,
                    "endpoint_hi": final[i][1] if qualified else None,
                    "endpoint_union_width": final[i][1] - final[i][0] if qualified else None,
                    "endpoint_lane_mean_width": statistics.mean(widths[i]) if qualified else None,
                    "endpoint_lane_max_width": max(widths[i]) if qualified else None,
                    "min_saved_safe_margin": margin,
                    "source_run_dir": str(run_dir),
                })
    native = read_json(SOURCES["native"] / "SCAN.json")
    p3 = read_json(SOURCES["ours_p3"] / "INDEPENDENT_INTERVAL_SCAN.json")
    for method, saved in (("native", native["tube_union"]),
                          ("ours_p3", p3["accepted_observed_tube_union"])):
        assert groups[method]["full_horizon"]["tube_union"] == saved
    assert ((SOURCES["huan"] / "payload/ranges.bin").read_bytes() ==
            (SOURCES["xiangru"] / "payload/ranges.bin").read_bytes())
    with (OUT / "RANGES.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "SCAN.json").write_text(json.dumps({
        "schema": "archcomp26-tora-remain-fourway-absolute-range-comparison-v1",
        "common_prefix_substeps": 184, "common_prefix_end_s": 18.4,
        "full_horizon_substeps": 200, "full_horizon_end_s": 20,
        "states": ["x1", "x2", "x3", "x4"],
        "source_runs": {k: str(v) for k, v in SOURCES.items()},
        "groups": groups,
        "huan_xiangru_saved_range_files_directly_equal": True,
        "qualification": "Saved axis-aligned boxes only; no independent floating-point NNCS proof.",
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {len(rows)} rows, four common-prefix profiles, two qualified full-horizon profiles")


if __name__ == "__main__":
    main()
