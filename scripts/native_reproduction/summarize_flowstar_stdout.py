#!/usr/bin/env python3
"""Parse the stock Flowstar benchmark's native stdout fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


STEP = re.compile(
    r"^time = ([0-9.eE+-]+),\s*step = ([0-9.eE+-]+),\s*order = ([0-9]+)$"
)
TIME_COST = re.compile(r"^time cost: ([0-9.eE+-]+)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--requested-horizon", required=True, type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    text = args.stdout.read_text(encoding="utf-8", errors="strict")
    steps = []
    solver_seconds = None
    for line in text.splitlines():
        match = STEP.fullmatch(line)
        if match:
            steps.append(
                {
                    "time": float(match.group(1)),
                    "step": float(match.group(2)),
                    "order": int(match.group(3)),
                }
            )
        timing = TIME_COST.fullmatch(line)
        if timing:
            solver_seconds = float(timing.group(1))
    if not steps:
        raise ValueError("no native Flowstar step lines found")
    if solver_seconds is None:
        raise ValueError("native Flowstar time cost line missing")
    reached = steps[-1]["time"]
    payload = {
        "schema_version": 1,
        "stdout": str(args.stdout.resolve()),
        "stdout_sha256": sha256(args.stdout),
        "missing_field_fallback": False,
        "requested_horizon": args.requested_horizon,
        "reached_horizon": reached,
        "completion": bool(math.isclose(reached, args.requested_horizon, rel_tol=0, abs_tol=1e-12)),
        "accepted_segments": len(steps),
        "orders": sorted({row["order"] for row in steps}),
        "last_segment": steps[-1],
        "solver_seconds": solver_seconds,
        "flowpipe_terminated": "Flowpipe computation is terminated" in text,
        "property_status": (
            "safe"
            if "All flowpipes are safe." in text
            else "unsafe"
            if "The last flowpipe is unsafe." in text
            else "unknown"
            if "The safety is unknown." in text
            else "missing"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
