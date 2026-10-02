#!/usr/bin/env python3
"""Compare a saved-RPC lane-0 replay with the frozen first-step receipt."""

import csv
import json
import math
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "native_quad_allbox_firststep_recenter_gate_20261003_003/run"
RECORD = struct.Struct("<QQd" + "d" * 48)


def rows(root, name):
    with (root / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def binary_records(root):
    raw = (root / "ranges.bin").read_bytes()
    if len(raw) % RECORD.size:
        raise ValueError("partial binary range record")
    result = [RECORD.unpack_from(raw, i) for i in range(0, len(raw), RECORD.size)]
    for step, record in enumerate(result, 1):
        if record[:3] != (0, step, 0.005):
            raise ValueError("unexpected lane, step, or duration")
        bounds = record[3:]
        if not all(map(math.isfinite, bounds)) or any(bounds[i] > bounds[i + 1]
                                                      for i in range(0, len(bounds), 2)):
            raise ValueError("invalid native range")
    return result


def require_run(root, expected):
    state = rows(root, "state.csv")
    if len(state) != 1 or state[0]["lane"] != "0" or state[0]["status"] != "2" or \
            int(state[0]["accepted_steps"]) != expected or \
            (root / "native_exit_code.txt").read_text().strip() != "0":
        raise ValueError("replay was not accepted for requested steps")
    if "RPC_REPLAY_INPUT_MATCH" not in (root / "stdout.log").read_text():
        raise ValueError("saved RPC input match marker absent")
    if json.loads((root / "rpc.json").read_text()) != json.loads((ORIGINAL / "rpc.json").read_text()):
        raise ValueError("replayed RPC response or input differs")
    records = binary_records(root)
    octagon = rows(root, "octagon.csv")
    axes = rows(root, "terminal_axes.csv")
    if len(records) != expected or len(octagon) != 8 * expected or len(axes) != 12:
        raise ValueError("incomplete replay output")
    if not all(math.isfinite(float(row[key])) and float(row["lo"]) <= float(row["hi"])
               for row in octagon for key in ("lo", "hi")):
        raise ValueError("invalid octagon output")
    for row in axes:
        if any(not math.isfinite(float(row[key])) for key in
               ("pre_lo", "pre_hi", "composed_lo", "composed_hi")) or \
                float(row["pre_lo"]) > float(row["pre_hi"]) or \
                float(row["composed_lo"]) > float(row["composed_hi"]):
            raise ValueError("invalid terminal axes")
    return records, octagon, axes


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("first", "two"):
        raise SystemExit("usage: audit_replay_nohash.py first|two")
    mode = sys.argv[1]
    current = HERE / mode
    records, octagon, axes = require_run(current, 1 if mode == "first" else 2)
    original_records = [RECORD.unpack_from((ORIGINAL / "ranges.bin").read_bytes())]
    if original_records[0][:3] != (0, 1, 0.005):
        raise ValueError("original first range record is not lane 0, step 1")
    original_octagon = [row for row in rows(ORIGINAL, "octagon.csv") if row["lane"] == "0"]
    original_axes = [row for row in rows(ORIGINAL, "terminal_axes.csv") if row["lane"] == "0"]
    comparisons = {
        "first_step_range_equal_original": records[0] == original_records[0],
        "first_step_octagon_equal_original": octagon[:8] == original_octagon,
    }
    if mode == "first":
        comparisons["first_step_terminal_axes_equal_original"] = axes == original_axes
    else:
        first_records, first_octagon, _ = require_run(HERE / "first", 1)
        comparisons["first_step_range_equal_first_replay"] = records[0] == first_records[0]
        comparisons["first_step_octagon_equal_first_replay"] = octagon[:8] == first_octagon
    result = {
        "mode": mode,
        "saved_rpc_numeric_equal": True,
        "status": 2,
        "accepted_steps": len(records),
        "range_records": len(records),
        "octagon_rows": len(octagon),
        "terminal_axes_rows": len(axes),
        "comparisons": comparisons,
        "first_issue": next((name for name, okay in comparisons.items() if not okay), None),
    }
    (HERE / ("FIRST_REPLAY_AUDIT.json" if mode == "first" else "TWO_REPLAY_AUDIT.json")).write_text(
        json.dumps(result, indent=2) + "\n")
    if result["first_issue"]:
        raise SystemExit(result["first_issue"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
