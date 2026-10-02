#!/usr/bin/env python3
"""Audit the saved Airplane binary-cover sample without rerunning the solver."""

from itertools import product
import json
import math
from pathlib import Path
import re
import struct

HERE = Path(__file__).parent
OLD = HERE.parent / "native_airplane_first_reject_trace_20261002_005/airplane_first_reject_trace.cpp"
NEW = HERE / "airplane_binary6_firststep.cpp"
SPLIT = ("u", "v", "w", "phi", "theta", "psi")
INDEX = (3, 4, 5, 6, 7, 8)


def main():
    plan = json.loads((HERE / "SPLIT_PLAN.json").read_text())
    expected = []
    for bits in product((0, 1), repeat=6):
        box = [[0, 0] for _ in range(19)]
        for index, bit in zip(INDEX, bits):
            box[index] = [bit / 2, (bit + 1) / 2]
        expected.append({"bits": "".join(map(str, bits)), "box": box})
    assert plan["coordinates"] == list(SPLIT) and plan["boxes"] == expected
    assert len(expected) == len({item["bits"] for item in expected}) == 64

    source = OLD.read_text()
    for name in SPLIT:
        old = f"init_{name}(0, 1)"
        assert source.count(old) == 1
        source = source.replace(old, f"init_{name}(0, 0.5)")
    assert source == NEW.read_text()
    initial = json.loads((HERE / "run/initial_boxes.json").read_text())
    assert initial == [expected[0]["box"]]
    rpc = [json.loads(s) for s in (HERE / "run/controller_rpc.jsonl").read_text().splitlines()]
    assert len(rpc) == 1
    assert rpc[0]["input_upper"] == [b[1] for b in initial[0][:12]]
    assert rpc[0]["input_lower"] == [b[0] for b in initial[0][:12]]

    result = json.loads((HERE / "run/RESULT.json").read_text())
    native = (HERE / "run/native.log").read_text()
    rows = [line for line in native.splitlines() if line.startswith("AIRPLANE_PICARD\t")]
    assert len(rows) == 19 and all("\tsubset=1" in line for line in rows)
    assert re.search(r"PERIOD 0 FLOWPIPES 1 STATUS 2 BOX_SAFE 1", native)
    assert "ONE_STEP_NUMERIC_ACCEPTED_ONLY" in native
    assert result["status"] == "completed" and result["exit_code"] == 0

    raw = (HERE / "run/ranges.bin").read_bytes()
    assert len(raw) == 24 + 12 * 32
    lane, step, h, *values = struct.unpack("<QQd48d", raw)
    assert (lane, step, h) == (0, 1, 0.01)
    bounds = [values[4 * i:4 * i + 4] for i in range(12)]
    assert all(all(math.isfinite(v) for v in row) and
               row[0] <= row[1] and row[2] <= row[3] for row in bounds)
    safety = (HERE / "run/safety.tsv").read_text().splitlines()
    assert len(safety) == 2
    fields = safety[1].split("\t")
    assert fields[:3] == ["0", "1", "1"]
    for i, offset in zip((1, 6, 7, 8), (3, 5, 7, 9)):
        lo, hi = bounds[i][:2]
        assert (lo, hi) == (float(fields[offset]), float(fields[offset + 1]))
        assert -1 <= lo <= hi <= 1
    assert float(fields[11]) > 0

    audit = {"partition_boxes": 64, "partition_exact_by_axis_union": True,
             "sampled_boxes": 1, "sample_bits": "000000",
             "source_change": "only six initial [0,1] intervals changed to [0,0.5]",
             "controller_calls": 1, "first_step_flowpipes": 1,
             "native_status": 2, "first_picard_inclusion": "19/19",
             "range_record": {"lane": lane, "step": step, "h": h,
                              "physical_dimensions": len(bounds)},
             "sample_tube_box_safe": True, "sample_cos_theta_lower": float(fields[11]),
             "full_box_or_horizon_result": False}
    (HERE / "INDEPENDENT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
