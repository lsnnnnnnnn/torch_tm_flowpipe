#!/usr/bin/env python3
"""Read the saved one-box, one-period physical-state intervals without replaying."""

import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parent
ROW = struct.Struct("<QQd16d")
raw = (ROOT / "ranges.bin").read_bytes()
if len(raw) != 50 * ROW.size:
    raise ValueError(f"expected 50 interval rows, found {len(raw)} bytes")
observations = [json.loads(line) for line in (ROOT / "observations.jsonl").read_text().splitlines()]
if len(observations) != 50:
    raise ValueError("observation count differs")

union = [[math.inf, -math.inf] for _ in range(4)]
last_endpoint = None
endpoint_outside_tube = []
for index in range(50):
    lane, step, h, *flat = ROW.unpack_from(raw, index * ROW.size)
    if lane != 0 or step != index + 1 or abs(h - 0.01) > 1e-12:
        raise ValueError(f"invalid lane, step, or substep size in row {index + 1}")
    if observations[index] != {"substep": index + 1, "accepted": True, "interval_valid": True}:
        raise ValueError(f"observation differs at substep {index + 1}")
    endpoint = []
    for coordinate in range(4):
        lo, hi, end_lo, end_hi = flat[4 * coordinate:4 * coordinate + 4]
        if not (all(math.isfinite(v) for v in (lo, hi, end_lo, end_hi)) and
                lo <= hi and end_lo <= end_hi):
            raise ValueError(f"invalid physical interval at step {step}, state {coordinate + 1}")
        union[coordinate][0] = min(union[coordinate][0], lo)
        union[coordinate][1] = max(union[coordinate][1], hi)
        endpoint.append([end_lo, end_hi])
        if end_lo < lo - 1e-10 or end_hi > hi + 1e-10:
            endpoint_outside_tube.append([step, coordinate + 1])
    last_endpoint = endpoint

receipt = {
    "profile": "tora_reach_sigmoid_official2026_mat_u11_firstperiod_diag",
    "kind": "independent_scan_of_saved_numeric_intervals_only",
    "rows": 50,
    "unique_lane_steps": 50,
    "all_rows_finite_ordered": True,
    "all_observations_accepted": True,
    "tube_union_by_state": dict(zip(("x1", "x2", "x3", "x4"), union)),
    "endpoint_at_t_0_5_by_state": dict(zip(("x1", "x2", "x3", "x4"), last_endpoint)),
    "endpoint_outside_same_substep_tube": endpoint_outside_tube,
    "property_evaluated": False,
    "full_horizon_completed": False,
    "qualification": "Saved interval scan does not certify end-to-end NNCS floating-point soundness.",
}
(ROOT / "INDEPENDENT_INTERVAL_SCAN.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
