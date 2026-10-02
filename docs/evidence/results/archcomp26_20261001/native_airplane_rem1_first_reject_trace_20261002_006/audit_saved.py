#!/usr/bin/env python3
"""Audit saved Airplane widened-remainder trace against the original receipts."""

import json
import math
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / "native_airplane_fullbox_order3_rem1_smoke1_001"
DEFAULT = ROOT / "native_airplane_first_reject_trace_20261002_005" / "run"
NAMES = ["x", "y", "z", "u", "v", "w", "phi", "theta", "psi",
         "r", "p", "q", "t", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def read_run(path):
    result = json.loads((path / "RESULT.json").read_text())
    box = json.loads((path / "initial_boxes.json").read_text())
    rpc_lines = (path / "controller_rpc.jsonl").read_text().splitlines()
    start = json.loads((path / "START.json").read_text())
    native = (path / "native.log").read_text()
    assert result["exit_code"] == 2 and not result["timed_out"]
    assert len(box) == 1 and len(box[0]) == 19
    assert box[0][3:9] == [[0, 1]] * 6
    assert all(pair == [0, 0] for pair in box[0][:3] + box[0][9:])
    assert len(rpc_lines) == 1
    assert re.findall(r"PERIOD 0 FLOWPIPES (\d+) STATUS (\d+)", native) == [("0", "4")]
    assert (path / "ranges.bin").stat().st_size == 0
    assert len((path / "safety.tsv").read_text().splitlines()) == 1
    return {"box": box, "rpc": json.loads(rpc_lines[0]), "start": start,
            "native": native, "result": result}


def trace_rows(native):
    rows = []
    for line in native.splitlines():
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
        assert row["old_lo"] == -1 and row["old_hi"] == 1
        assert row["subset"] == int(row["new_lo"] >= -1 and row["new_hi"] <= 1)
        rows.append(row)
    assert [row["index"] for row in rows] == list(range(19))
    return rows


if __name__ == "__main__":
    old, default, new = read_run(OLD), read_run(DEFAULT), read_run(HERE / "run")
    assert old["box"] == default["box"] == new["box"]
    assert old["rpc"] == default["rpc"] == new["rpc"]
    for field in ("AIRPLANE_MODEL", "AIRPLANE_OVERLAY", "AIRPLANE_SERVER_PYTHON",
                  "AIRPLANE_CPUSET", "AIRPLANE_NATIVE_AS", "CUDA_VISIBLE_DEVICES"):
        assert old["start"]["selected_environment"][field] == new["start"]["selected_environment"][field]
    source = (ROOT / "native_airplane_fullbox_smokes_20261002" /
              "airplane_fullbox_order3_remainder1_smoke1.cpp").read_text()
    replacements = (
        ("author_matched::reach(dynamics, result, initial_set, 0.1,",
         "author_matched::reach(dynamics, result, initial_set, 0.01,"),
        ("produced != 10", "produced != 1"),
        ('cout << "VERIFIED" << endl;', 'cout << "ONE_STEP_NUMERIC_ACCEPTED_ONLY" << endl;'),
        ('cout << "COMPLETED_SAFE_PERIODS "', 'cout << "DIAGNOSTIC_COMPLETED_CALLS "'),
    )
    for before, after in replacements:
        assert source.count(before) == 1
        source = source.replace(before, after)
    assert source == (HERE / "airplane_first_reject_trace.cpp").read_text()
    assert json.loads((HERE / "BUILD.json").read_text())["returncodes"] == [0, 0, 0]
    rows = trace_rows(new["native"])
    assert not any(line.startswith("AIRPLANE_PICARD\t") for line in old["native"].splitlines())
    report = {
        "scope": "read-only saved-file audit; one full box, one CROWN call, one h=0.01 step",
        "old_wide_run": OLD.name, "default_trace_run": DEFAULT.parent.name,
        "new_wide_trace_run": HERE.name, "same_initial_box_and_rpc_across_three": True,
        "matched_old_wide_entry_except_one_step_duration_and_output_labels": True,
        "old_and_new_native_status": "UNCOMPLETED_SAFE (4)", "old_and_new_saved_ranges": 0,
        "first_rejected_coordinate": next(row["coordinate"] for row in rows if not row["subset"]),
        "rejected_coordinates": [row["coordinate"] for row in rows if not row["subset"]],
        "rows": rows,
        "limitation": "finite interval self-map refusal in widened numerical profile; no T=2 or trajectory counterexample",
    }
    (HERE / "INDEPENDENT_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "same_initial_box_and_rpc_across_three", "first_rejected_coordinate",
        "rejected_coordinates", "old_and_new_native_status")}))
