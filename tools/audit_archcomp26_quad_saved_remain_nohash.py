#!/usr/bin/env python3
"""Audit the saved QUAD x3 tube/endpoint suffix without running a solver."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/evidence/results/archcomp26_20261001"
P3 = RESULTS / "quad_paper_p3_nohash_v1/full50_001/data/observations.jsonl"
NATIVE = RESULTS / "native_quad_paper_full50_001/pooled_x3_1000steps.csv"
OUTPUT = RESULTS / "quad_paper_fourway_saved_20261002/SAVED_REMAIN_AUDIT.json"
LOW, HIGH = 0.94, 1.06


def read_p3():
    rows = []
    outside_count = 0
    max_gap = 0.0
    for line in P3.read_text().splitlines():
        row = json.loads(line)
        step = row["substep"]
        tube_lo, tube_hi, end_lo, end_hi = row["tube_endpoint_union_12x4"][2]
        if step != len(rows) + 1 or row["accepted_count"] != 1024 or row["status_counts"] != {"0": 1024}:
            raise ValueError(f"P3 incomplete at step {step}")
        if not (all(math.isfinite(x) for x in (tube_lo, tube_hi, end_lo, end_hi))
                and tube_lo <= tube_hi and end_lo <= end_hi):
            raise ValueError(f"P3 invalid x3 interval at step {step}")
        gap = max(0.0, tube_lo - end_lo, end_hi - tube_hi)
        outside_count += gap > 0
        max_gap = max(max_gap, gap)
        rows.append((step, min(tube_lo, end_lo), max(tube_hi, end_hi)))
    return rows, outside_count, max_gap


def read_native():
    rows = []
    with NATIVE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            tube_lo, tube_hi = float(row["tube_x3_lo"]), float(row["tube_x3_hi"])
            end_lo, end_hi = float(row["endpoint_x3_lo"]), float(row["endpoint_x3_hi"])
            if step != len(rows) + 1:
                raise ValueError(f"native missing/duplicate step {step}")
            if not (all(math.isfinite(x) for x in (tube_lo, tube_hi, end_lo, end_hi))
                    and tube_lo <= end_lo <= end_hi <= tube_hi):
                raise ValueError(f"native invalid x3 interval at step {step}")
            rows.append((step, tube_lo, tube_hi))
    return rows


def suffix(rows):
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 steps, got {len(rows)}")
    first = next(i for i in range(len(rows)) if all(
        LOW <= lo and hi <= HIGH for _, lo, hi in rows[i:]))
    preceding = rows[first - 1] if first else None
    return {
        "first_step": rows[first][0],
        "nominal_start_time_s": (rows[first][0] - 1) * 0.005,
        "suffix_step_count": len(rows) - first,
        "preceding_step": preceding[0] if preceding else None,
        "preceding_interval": list(preceding[1:]) if preceding else None,
        "final_interval": list(rows[-1][1:]),
    }


def main():
    p3, outside_count, max_gap = read_p3()
    native = read_native()
    result = {
        "schema": "archcomp26-quad-saved-x3-suffix-v1",
        "contract": "2026 paper equations, full 1024-box initial set, 1000 ODE substeps",
        "target_interval": [LOW, HIGH],
        "scope": "saved coordinate intervals only; not the participant temporal checker or an end-to-end floating-point NNCS proof",
        "sources": {"p3": str(P3.relative_to(ROOT)), "native": str(NATIVE.relative_to(ROOT))},
        "p3": {
            **suffix(p3),
            "interval_rule": "hull of separately saved tube and endpoint x3 unions",
            "endpoint_outside_saved_tube_step_count": outside_count,
            "maximum_endpoint_tube_gap": max_gap,
        },
        "native": {**suffix(native), "interval_rule": "saved pooled tube x3"},
        "huan_xiangru": "only final endpoint x3 saved; no all-step remain conclusion",
        "content_digest_performed": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
