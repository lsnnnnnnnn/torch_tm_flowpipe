#!/usr/bin/env python3
"""Deterministically replay sanitized R1/R2 aggregates from private snapshots.

The full per-leaf snapshot remains private.  This verifier checks its declared
hash and internal controller-update hash before deriving only aggregate widths,
predicate counts, and lifecycle attribution for the public tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_POINTS = {
    "R1": {"segment_index": 10, "controller_period": 2},
    "R2": {"segment_index": 40, "controller_period": 5},
}
PREDICATES = (
    "finite_ok_by_leaf",
    "initial_subset_ok_by_leaf",
    "all_remainder_rounds_ok_by_leaf",
    "local_property_ok_by_leaf",
    "composed_property_ok_by_leaf",
    "numerical_ok_by_leaf",
    "overall_accepted_by_leaf",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def width_statistics(lower: Any, upper: Any) -> dict[str, float]:
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if lower_array.shape != upper_array.shape or lower_array.size == 0:
        raise ValueError("lower/upper replay arrays must have the same non-empty shape")
    if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
        raise ValueError("replay arrays must be finite")
    if not np.all(lower_array <= upper_array):
        raise ValueError("replay lower bound exceeds upper bound")
    widths = (upper_array - lower_array).reshape(-1)
    return {
        "maximum": float(np.max(widths)),
        "median": float(np.median(widths)),
        "p95": float(np.percentile(widths, 95)),
        "sum": float(np.sum(widths)),
    }


def _verify_controller_content_hash(controller_refresh: dict[str, Any]) -> str:
    claimed = controller_refresh.get("content_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("controller refresh is missing its content SHA-256")
    payload = dict(controller_refresh)
    del payload["content_sha256"]
    actual = canonical_sha256(payload)
    if actual != claimed:
        raise ValueError("controller refresh content SHA-256 mismatch")
    return actual


def summarize_point(name: str, point: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_POINTS[name]
    if int(point.get("segment_index", -1)) != expected["segment_index"]:
        raise ValueError(f"{name} has the wrong segment")
    if point.get("lane") != "baseline_native":
        raise ValueError(f"{name} must come from the baseline_native lane")
    leaf_ids = point.get("leaf_id")
    if leaf_ids != list(range(48)):
        raise ValueError(f"{name} must contain the ordered B48 leaf set")
    predicates = point.get("predicates", {})
    predicate_counts: dict[str, int] = {}
    for predicate in PREDICATES:
        values = np.asarray(predicates.get(predicate), dtype=bool)
        if values.shape != (48,):
            raise ValueError(f"{name} predicate {predicate} is not B48")
        predicate_counts[predicate] = int(np.count_nonzero(values))

    controller = point.get("controller_refresh")
    if not isinstance(controller, dict):
        raise ValueError(f"{name} is missing the following controller refresh")
    if int(controller.get("controller_period", -1)) != expected["controller_period"]:
        raise ValueError(f"{name} controller period does not match the replay contract")
    controller_content_sha256 = _verify_controller_content_hash(controller)

    property_margin = np.asarray(point.get("property_margin"), dtype=np.float64)
    if property_margin.shape != (48, 4) or not np.all(np.isfinite(property_margin)):
        raise ValueError(f"{name} property margin must have shape B48x4")
    width_attribution = point.get("width_attribution")
    if not isinstance(width_attribution, dict):
        raise ValueError(f"{name} is missing width attribution")

    return {
        "controller_period": expected["controller_period"],
        "controller_refresh": {
            "comparison_to_xiangru_observation": controller.get(
                "comparison_to_xiangru_observation"
            ),
            "content_sha256": controller_content_sha256,
            "input_width": controller["controller_input_width"],
            "output_after_width": controller["controller_output_after_width"],
            "output_before_width": controller["controller_output_before_width"],
        },
        "endpoint_width": width_statistics(
            point["endpoint"]["lower"], point["endpoint"]["upper"]
        ),
        "interval_remainder_width": width_statistics(
            point["interval_remainder"]["lower"],
            point["interval_remainder"]["upper"],
        ),
        "minimum_property_margin": float(np.min(property_margin)),
        "point_content_sha256": canonical_sha256(point),
        "predicate_true_counts": predicate_counts,
        "segment_index": expected["segment_index"],
        "tube_width": width_statistics(
            point["tube"]["lower"], point["tube"]["upper"]
        ),
        "width_attribution": width_attribution,
    }


def replay(input_path: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = file_sha256(input_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("private R1/R2 replay snapshot SHA-256 mismatch")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if set(payload) != set(EXPECTED_POINTS):
        raise ValueError("private replay snapshot must contain exactly R1 and R2")
    return {
        "input_private_snapshot_sha256": actual_sha256,
        "points": {
            name: summarize_point(name, payload[name])
            for name in EXPECTED_POINTS
        },
        "replay_contract": (
            "deterministic aggregate regeneration from hash-verified private "
            "R1/R2 snapshots; no full-loop rerun required"
        ),
        "schema": "tora_q3_r1_r2_deterministic_replay_v1",
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.input, args.expected_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input_private_snapshot_sha256": result[
                    "input_private_snapshot_sha256"
                ],
                "points": list(result["points"]),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
