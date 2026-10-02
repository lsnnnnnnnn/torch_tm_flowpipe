#!/usr/bin/env python3
"""Read only the two saved isolated Airplane first-step receipts."""

import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "native_airplane_first_reject_trace_20261002_004"
NAMES = ["x", "y", "z", "u", "v", "w", "phi", "theta", "psi",
         "r", "p", "q", "t", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def read(root):
    run = root / "run"
    result = json.loads((run / "RESULT.json").read_text())
    build = json.loads((root / "BUILD.json").read_text())
    log = (run / "native.log").read_text()
    box = json.loads((run / "initial_boxes.json").read_text())
    rpc_lines = (run / "controller_rpc.jsonl").read_text().splitlines()
    assert result["exit_code"] == 2 and not result["timed_out"]
    assert build["returncodes"] == [0, 0, 0]
    assert len(box) == 1 and len(box[0]) == 19
    assert box[0][3:9] == [[0, 1]] * 6
    assert all(pair == [0, 0] for pair in box[0][:3] + box[0][9:])
    assert len(rpc_lines) == 1 and (run / "ranges.bin").stat().st_size == 0
    assert re.findall(r"PERIOD 0 FLOWPIPES (\d+) STATUS (\d+)", log) == [("0", "4")]
    assert len((run / "safety.tsv").read_text().splitlines()) == 1
    rows = []
    for line in log.splitlines():
        if not line.startswith("AIRPLANE_PICARD\t"):
            continue
        parts = line.split("\t")
        index = int(parts[1])
        row = {"index": index, "coordinate": NAMES[index]}
        for part in parts[2:]:
            key, value = part.split("=", 1)
            row[key] = int(value) if key == "subset" else float(value)
        for prefix in ("old", "base", "diff", "new"):
            lo, hi = row[prefix + "_lo"], row[prefix + "_hi"]
            assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
        assert row["old_lo"] == -0.01 and row["old_hi"] == 0.01
        assert row["subset"] == int(row["new_lo"] >= -0.01 and row["new_hi"] <= 0.01)
        rows.append(row)
    return {"result": result, "box": box, "rpc": json.loads(rpc_lines[0]), "rows": rows,
            "build_source": build["airplane_source"]}


if __name__ == "__main__":
    old, new = read(OLD), read(HERE)
    assert old["rows"] == []
    assert len(new["rows"]) == 19
    assert [row["index"] for row in new["rows"]] == list(range(19))
    assert old["box"] == new["box"] and old["rpc"] == new["rpc"]
    assert old["build_source"] == new["build_source"]
    report = {
        "scope": "read-only raw-log audit; one full box, one CROWN call, one h=0.01 step per isolated run",
        "order": 3, "old_run": OLD.name, "new_run": HERE.name,
        "same_initial_box_and_rpc": True, "old_trace_rows": 0,
        "new_trace_rows": len(new["rows"]),
        "first_rejected_coordinate": next(row["coordinate"] for row in new["rows"] if not row["subset"]),
        "rejected_coordinates": [row["coordinate"] for row in new["rows"] if not row["subset"]],
        "both_native_status": "UNCOMPLETED_SAFE (4)", "both_saved_ranges": 0,
        "new_rows": new["rows"],
        "limitation": "Picard enclosure refusal, not a full T=2 result or actual trajectory counterexample",
    }
    (HERE / "INDEPENDENT_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "same_initial_box_and_rpc", "new_trace_rows", "first_rejected_coordinate",
        "rejected_coordinates", "both_native_status")}))
