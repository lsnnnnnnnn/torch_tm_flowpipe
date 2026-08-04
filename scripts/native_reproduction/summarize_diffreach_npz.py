#!/usr/bin/env python3
"""Summarize a native DiffReach flowpipe archive without field fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


REQUIRED_KEYS = ("ts", "lowers", "uppers", "shrinked")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--requested-horizon", required=True, type=float)
    parser.add_argument(
        "--shrinked-semantics",
        choices=("initial_contraction", "placeholder"),
        default="initial_contraction",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    with np.load(args.input) as archive:
        present = list(archive.files)
        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise KeyError(f"missing required keys: {missing}; present={present}")
        ts = np.asarray(archive["ts"])
        lowers = np.asarray(archive["lowers"])
        uppers = np.asarray(archive["uppers"])
        shrinked = np.asarray(archive["shrinked"])

    if ts.ndim != 1 or not ts.size:
        raise ValueError(f"ts must be a nonempty vector, got {ts.shape}")
    if lowers.shape != uppers.shape or lowers.ndim != 3:
        raise ValueError(
            f"lower/upper shape mismatch: {lowers.shape} vs {uppers.shape}"
        )
    if lowers.shape[1] != ts.size:
        raise ValueError("time axis does not match lower/upper arrays")
    if args.shrinked_semantics == "initial_contraction" and shrinked.dtype != np.bool_:
        raise TypeError(f"shrinked must be boolean, got {shrinked.dtype}")
    if args.shrinked_semantics == "placeholder" and not np.all(shrinked == 1):
        raise ValueError("placeholder shrinked array must contain only ones")
    shrinked_bool = shrinked.astype(bool)
    false_indices = np.argwhere(~shrinked_bool)
    reached = float(ts[-1])
    payload = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "required_keys": list(REQUIRED_KEYS),
        "present_keys": present,
        "missing_field_fallback": False,
        "requested_horizon": args.requested_horizon,
        "reached_horizon": reached,
        "completion": bool(math.isclose(reached, args.requested_horizon, rel_tol=0, abs_tol=1e-12)),
        "partitions": int(lowers.shape[0]),
        "time_points": int(lowers.shape[1]),
        "state_dimensions": int(lowers.shape[2]),
        "lower_dtype": str(lowers.dtype),
        "upper_dtype": str(uppers.dtype),
        "finite": bool(np.isfinite(lowers).all() and np.isfinite(uppers).all()),
        "ordered": bool((lowers <= uppers).all()),
        "final_aggregate_lower": lowers[:, -1, :].min(axis=0).tolist(),
        "final_aggregate_upper": uppers[:, -1, :].max(axis=0).tolist(),
        "initial_shrink_flags": {
            "shape": list(shrinked.shape),
            "true": int(shrinked_bool.sum()),
            "total": int(shrinked.size),
            "rate": float(shrinked_bool.mean()),
            "all_true": bool(shrinked_bool.all()),
            "first_false_index": (
                false_indices[0].tolist() if false_indices.size else None
            ),
            "semantics": args.shrinked_semantics,
            "scope": (
                "only the initial contraction predicate returned by "
                "src.picard.remainder_picard; not all refinement rounds"
                if args.shrinked_semantics == "initial_contraction"
                else "source-declared placeholder; not a contraction result"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
