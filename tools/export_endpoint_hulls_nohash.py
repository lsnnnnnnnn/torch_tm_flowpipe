#!/usr/bin/env python3
"""Export saved terminal hulls to a comparison CSV without content digests.

Example: python3 tools/export_endpoint_hulls_nohash.py \
  --input huan=/path/to/metrics.json --input p3=/path/to/metrics.json \
  --states 12 --output /path/to/endpoint_hulls.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="METHOD=METRICS_JSON")
    parser.add_argument("--states", type=int, required=True, help="number of physical states")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.states <= 0:
        parser.error("--states must be positive")

    rows = []
    seen = set()
    for item in args.input:
        method, separator, source = item.partition("=")
        if not separator or not method or not source or method in seen:
            parser.error("each --input must be a unique METHOD=METRICS_JSON")
        seen.add(method)
        path = Path(source).resolve()
        metrics = json.loads(path.read_text(encoding="utf-8"))
        hull = metrics["final_hull"]
        for index in range(1, args.states + 1):
            state = f"x{index}"
            bounds = hull[state]
            if len(bounds) != 2:
                raise ValueError(f"{path}: malformed {state}")
            lo, hi = (float(value) for value in bounds)
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError(f"{path}: invalid {state}")
            width = hi - lo
            if not math.isfinite(width):
                raise ValueError(f"{path}: nonfinite width for {state}")
            rows.append({"method": method, "state": state, "lower": lo,
                         "upper": hi, "width": width,
                         "metric_source": str(path)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} endpoint rows to {args.output}")


if __name__ == "__main__":
    main()
