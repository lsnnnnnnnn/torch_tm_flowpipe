#!/usr/bin/env python3
"""Audit saved Docking native receipts; no solver or controller is run."""

import csv
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parent
RECORD = struct.Struct("<QQd16d")
STATES = ("sx", "sy", "vx", "vy")


def finite_box(values):
    return len(values) == 2 and all(math.isfinite(v) for v in values) and values[0] <= values[1]


def main():
    start = json.loads((ROOT / "START.json").read_text())
    outer = json.loads((ROOT / "RESULT.json").read_text())
    slope = json.loads((ROOT / "SLOPE_CHECK_ALL.json").read_text())
    initial = json.loads((ROOT / "initial_boxes.json").read_text())
    assert len(initial) == 1 and initial[0][:4] == [[70, 106], [70, 106], [-0.28, 0.28], [-0.28, 0.28]]
    assert len(initial[0]) == 7 and all(pair == [0, 0] for pair in initial[0][4:])
    assert start["contract_label"] == "docking-paper-fullbox-radial-v1"
    assert start["selected_environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert outer["status"] == "failed" and outer["exit_code"] == 2 and not outer["timed_out"]

    payload = (ROOT / "ranges.bin").read_bytes()
    assert len(payload) == 400 * RECORD.size
    ranges = []
    for index, values in enumerate(RECORD.iter_unpack(payload), 1):
        lane, step, h, *coordinates = values
        assert lane == 0 and step == index and abs(h - 0.1) < 1e-15
        assert all(math.isfinite(value) for value in coordinates)
        state_ranges = [coordinates[4 * j:4 * j + 4] for j in range(4)]
        assert all(lo <= hi and end_lo <= end_hi
                   for lo, hi, end_lo, end_hi in state_ranges)
        ranges.append(state_ranges)

    with (ROOT / "safety.tsv").open() as stream:
        safety = list(csv.DictReader(stream, delimiter="\t"))
    assert len(safety) == 400
    for step, row in enumerate(safety, 1):
        assert int(row["global_substep"]) == step
        assert int(row["period"]) == (step - 1) // 10
        assert int(row["local_substep"]) == (step - 1) % 10 + 1
        qlo, qhi = float(row["q_lower"]), float(row["q_upper"])
        assert math.isfinite(qlo) and math.isfinite(qhi) and qlo <= qhi

    rpc = [json.loads(line) for line in (ROOT / "controller_rpc.jsonl").read_text().splitlines()]
    assert len(rpc) == 40
    for row in rpc:
        lower, upper, response = row["input_lower"], row["input_upper"], row["response"]
        assert len(lower) == len(upper) == 4
        assert all(math.isfinite(lo) and math.isfinite(hi) and lo <= hi
                   for lo, hi in zip(lower, upper))
        assert len(response["T"]) == len(response["u_min"]) == len(response["u_max"]) == 1
        assert len(response["T"][0]) == len(response["u_min"][0]) == len(response["u_max"][0]) == 2
        assert all(len(coefficients) == 4 and all(math.isfinite(value) for value in coefficients)
                   for coefficients in response["T"][0])
        assert all(math.isfinite(value) for value in response["u_min"][0] + response["u_max"][0])
    assert rpc[0]["input_lower"] == [70, 70, -0.28, -0.28]
    assert rpc[0]["input_upper"] == [106, 106, 0.28, 0.28]
    assert slope["checked_count"] == 40 and slope["all_exact_equal"]

    lines = (ROOT / "native.log").read_text().splitlines()
    assert lines[0].startswith("INITIAL_MARGIN ")
    _, initial_lower_text, initial_upper_text = lines[0].split()
    initial_lower, initial_upper = float(initial_lower_text), float(initial_upper_text)
    assert math.isfinite(initial_lower) and math.isfinite(initial_upper)
    assert initial_lower <= initial_upper <= 0
    assert "COMPLETED_PERIODS 40/40" in lines
    assert "FLOWPIPE_SEGMENTS 400" in lines
    assert lines[-1] == "UNKNOWN"
    assert sum(line.startswith("PERIOD ") for line in lines) == 40

    first_unknown = next((int(row["global_substep"]) for row in safety
                          if float(row["q_upper"]) > 0), None)
    margin_upper = max(float(row["q_upper"]) for row in safety)
    endpoint = {name: {"lo": ranges[-1][i][2], "hi": ranges[-1][i][3],
                       "width": ranges[-1][i][3] - ranges[-1][i][2]}
                for i, name in enumerate(STATES)}
    tube = {name: {"lo": min(row[i][0] for row in ranges),
                   "hi": max(row[i][1] for row in ranges)}
            for i, name in enumerate(STATES)}
    for item in tube.values():
        item["width"] = item["hi"] - item["lo"]
    prior_endpoint_gap = max(
        abs(observed - recorded)
        for period in range(1, 40)
        for observed, recorded in zip(
            rpc[period]["input_lower"] + rpc[period]["input_upper"],
            [ranges[10 * period - 1][i][2] for i in range(4)] +
            [ranges[10 * period - 1][i][3] for i in range(4)]
        )
    )
    audit = {
        "schema": "archcomp26-native-docking-saved-audit-v1",
        "remote_run_dir": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_docking_full40_001",
        "counts": {"physical_boxes": 1, "completed_periods": 40,
                   "period_substeps": 10, "saved_tube_segments": len(ranges),
                   "saved_safety_segments": len(safety), "controller_rpc_calls": len(rpc)},
        "outer_result": outer,
        "native_property_status": "UNKNOWN",
        "initial_margin": [initial_lower, initial_upper],
        "first_unknown_substep": first_unknown,
        "first_unknown_time_interval_s": [0.1 * (first_unknown - 1), 0.1 * first_unknown],
        "first_tube_margin": [float(safety[0]["q_lower"]), float(safety[0]["q_upper"])],
        "last_tube_margin": [float(safety[-1]["q_lower"]), float(safety[-1]["q_upper"])],
        "maximum_tube_margin_upper": margin_upper,
        "box_certified_safe_segments": sum(float(row["q_upper"]) <= 0 for row in safety),
        "box_unknown_segments": sum(float(row["q_lower"]) <= 0 < float(row["q_upper"]) for row in safety),
        "box_lower_positive_segments": sum(float(row["q_lower"]) > 0 for row in safety),
        "endpoint_t40": endpoint, "tube_union_t0_t40": tube,
        "max_abs_next_rpc_vs_saved_prior_endpoint": prior_endpoint_gap,
        "slope_check_all_40": slope,
        "qualification": "Flow* interval plant tube and radial box checker are saved; CROWN converted graph bounds are not an independent end-to-end float32 ONNX certificate; positive q upper is uncertainty, not an unsafe witness",
    }
    (ROOT / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"counts": audit["counts"], "outer": outer,
                      "first_unknown": first_unknown, "margin_upper": margin_upper,
                      "slope_equal_all": slope["all_exact_equal"]}))


if __name__ == "__main__":
    main()
