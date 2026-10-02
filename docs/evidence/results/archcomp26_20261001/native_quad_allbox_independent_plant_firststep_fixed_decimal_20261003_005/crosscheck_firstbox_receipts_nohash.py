#!/usr/bin/env python3
"""Compare saved single-box and all-box lane-0 inputs and observers directly."""

import csv
import json
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "native_quad_initial_recenter_repair_gate_20261003_002/quad/on"
ALL = HERE.parent / "native_quad_allbox_firststep_recenter_gate_20261003_003/run"


def rows(path, lane=None):
    with path.open(newline="") as stream:
        found = list(csv.DictReader(stream))
    return [{k: value for k, value in row.items() if k != "lane"}
            for row in found if lane is None or int(row["lane"]) == lane]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def main():
    old = json.loads((OLD / "rpc.json").read_text())
    current = json.loads((ALL / "rpc.json").read_text())
    checks = {
        "rpc_input_lb_equal": old["params"]["input_lb"] == current["params"]["input_lb"][:12],
        "rpc_input_ub_equal": old["params"]["input_ub"] == current["params"]["input_ub"][:12],
    }
    for key in ("T", "u_min", "u_max"):
        before = old["coefficients"][key][0]
        after = current["coefficients"][key][0]
        if key == "T":
            before = [f32(value) for row in before for value in row]
            after = [f32(value) for row in after for value in row]
        else:
            before, after = list(map(f32, before)), list(map(f32, after))
        checks[key + "_equal_after_float32_transport"] = before == after
    for name in ("octagon.csv", "terminal_axes.csv"):
        checks[name + "_lane0_equal"] = rows(OLD / name) == rows(ALL / name, lane=0)
    result = {
        "schema": "native-quad-firstbox-saved-receipt-crosscheck-nohash-v1",
        "checks": checks,
        "all_passed": all(checks.values()),
        "meaning": "All-box lane 0 has the same saved binary64 RPC input, float32-transported coefficients, and observer rows as the earlier independently checked first physical box. This compares saved data only, not CROWN validity.",
    }
    (HERE / "FIRSTBOX_RECEIPT_CROSSCHECK.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
