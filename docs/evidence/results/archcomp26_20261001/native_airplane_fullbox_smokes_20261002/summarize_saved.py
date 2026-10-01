#!/usr/bin/env python3
"""Audit three saved native Airplane full-box failures without running a solver."""

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
PROFILES = (
    ("native_airplane_fullbox_order6_smoke1_001", 6, "[-0.01,0.01]"),
    ("native_airplane_fullbox_order3_smoke1_001", 3, "[-0.01,0.01]"),
    ("native_airplane_fullbox_order3_rem1_smoke1_001", 3, "[-1,1]"),
)
INITIAL = [[0, 0]] * 3 + [[0, 1]] * 6 + [[0, 0]] * 10


def main():
    rows = []
    for name, order, remainder in PROFILES:
        path = PARENT / name
        start = json.loads((path / "START.json").read_text())
        result = json.loads((path / "RESULT.json").read_text())
        ledger = json.loads((path / "initial_boxes.json").read_text())
        rpc = [json.loads(line) for line in (path / "controller_rpc.jsonl").read_text().splitlines()]
        native = (path / "native.log").read_text().splitlines()
        checks = (path / "safety.tsv").read_text().splitlines()
        server_pid = (path / "server.pid").read_text().strip()
        listeners = (path / "listeners_at_start.txt").read_text()
        assert ledger == [INITIAL] and len(rpc) == 1
        assert (path / "ranges.bin").stat().st_size == 0 and len(checks) == 1
        assert result["status"] == "failed" and result["exit_code"] == 2
        assert not result["timed_out"] and math.isfinite(result["wall_s"])
        assert start["selected_environment"]["CUDA_VISIBLE_DEVICES"] == "2"
        assert start["selected_environment"]["AIRPLANE_CPUSET"] == "10-13"
        assert start["selected_environment"]["AIRPLANE_NATIVE_AS"] == "34359738368"
        assert start["timeout_s"] == 600
        assert f'pid={server_pid},' in listeners and "127.0.0.1:5105" in listeners
        assert native[:5] == ["Step 0", "PERIOD 0 FLOWPIPES 0 STATUS 4 BOX_SAFE 1",
                              "COMPLETED_SAFE_PERIODS 0/1", "FLOWPIPE_SEGMENTS 0", "UNKNOWN"]
        request, response = rpc[0], rpc[0]["response"]
        assert len(request["input_lower"]) == len(request["input_upper"]) == 12
        assert request["input_lower"] == [float(pair[0]) for pair in INITIAL[:12]]
        assert request["input_upper"] == [float(pair[1]) for pair in INITIAL[:12]]
        assert len(response["T"]) == len(response["u_min"]) == len(response["u_max"]) == 6
        assert all(len(row) == 12 and all(math.isfinite(value) for value in row)
                   for row in response["T"])
        assert all(math.isfinite(value) for value in response["u_min"] + response["u_max"])
        rows.append({
            "run_dir": name, "order": order, "remainder_estimation": remainder,
            "ode_step_s": 0.01, "control_period_s": 0.1,
            "outer_status": result["status"], "outer_exit_code": result["exit_code"],
            "outer_wall_s": result["wall_s"], "outer_timeout_s": start["timeout_s"],
            "native_address_limit_bytes": 34359738368, "controller_rpc_calls": len(rpc),
            "accepted_flowpipe_segments": 0, "flowstar_status_code": 4,
            "flowstar_status_name": "UNCOMPLETED_SAFE", "property_samples": 0,
            "property_verdict": "UNKNOWN_NO_ACCEPTED_TUBE",
        })
    with (HERE / "RUNS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "AUDIT.json").write_text(json.dumps({
        "schema": "archcomp26-native-airplane-fullbox-smoke-audit-v1",
        "full_initial_box": INITIAL,
        "runs": rows,
        "conclusion": "All three independent profiles complete one controller RPC but accept zero Flow* substeps; no numerical tube or property verdict for the first period or T=2 exists.",
        "source_level_failure_condition": "With an empty invariant, saved Flowpipe::advance(Real,Symbolic_Remainder) can return 0 when a computed Picard remainder is not contained in the preset remainder estimate (Continuous.cpp lines 2363-2372). No per-variable computed remainder was logged.",
    }, indent=2) + "\n")
    print(json.dumps({"runs": len(rows), "walls": [row["outer_wall_s"] for row in rows],
                      "all_accepted_segments": sum(row["accepted_flowpipe_segments"] for row in rows)}))


if __name__ == "__main__":
    main()
