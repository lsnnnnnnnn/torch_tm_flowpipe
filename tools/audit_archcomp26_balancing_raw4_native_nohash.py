#!/usr/bin/env python3
"""Audit saved native CartPole raw4 time coverage and controller calls."""

import argparse
import json
import math
from pathlib import Path
import re
import struct


ROW = struct.Struct("<QQd" + "d" * 20)  # 5 states: tube/endpoint lower/upper
INITIAL = [[-0.1, 0.1], [-0.05, 0.05], [-0.1, 0.1], [-0.05, 0.05]]


def audit(path, periods):
    result = json.loads((path / "RESULT.json").read_text())
    start = json.loads((path / "START.json").read_text())
    native_log = (path / "native.log").read_text()
    if (start["instance"] != "balancing-reach" or start["method"] != "native" or
            "balancing-fixed-repo-raw4" not in start["contract_label"]):
        raise ValueError("run receipt does not name native fixed-repository raw4 CartPole")
    ranges = path / "ranges.bin"
    if not ranges.exists():
        if (result["exit_code"] != 139 or "Error line 1: syntax error" not in native_log or
                (path / "initial_boxes.json").exists() or
                (path / "controller_rpc.jsonl").exists()):
            raise ValueError("unexpected no-range native failure")
        return {"schema": "archcomp26-balancing-native-coverage-audit-nohash-v1",
                "source": str(path), "planned_periods": periods, "saved_substeps": 0,
                "controller_calls": 0, "status": "prestep_parser_failure",
                "outer_exit_code": result["exit_code"], "property_verdict": "NO_NUMERICAL_RESULT"}

    boxes = json.loads((path / "initial_boxes.json").read_text())
    if len(boxes) != 1 or len(boxes[0]) != 6:
        raise ValueError("initial ledger is not one six-state box")
    for j, expected in enumerate(INITIAL):
        actual = boxes[0][j]
        if actual[0] > expected[0] or actual[1] < expected[1]:
            raise ValueError(f"initial physical state {j} not covered")
    if boxes[0][4:] != [[0, 0], [0, 0]]:
        raise ValueError("initial time or held force differs")

    data = ranges.read_bytes()
    if len(data) % ROW.size:
        raise ValueError("partial binary range record")
    saved = len(data) // ROW.size
    if saved > periods * 4:
        raise ValueError("more saved steps than planned")
    for index in range(saved):
        lane, step, h, *bounds = ROW.unpack_from(data, index * ROW.size)
        if lane != 0 or step != index + 1 or not math.isfinite(h) or abs(h - 0.005) > 1e-15:
            raise ValueError(f"noncontiguous lane/step or integration size at {index + 1}")
        tlo, thi, elo, ehi = bounds[16:20]
        expected_t = (index + 1) * 0.005
        if not (math.isfinite(tlo) and math.isfinite(thi) and
                math.isfinite(elo) and math.isfinite(ehi) and
                elo - 1e-12 <= expected_t <= ehi + 1e-12):
            raise ValueError(f"saved time endpoint differs at step {index + 1}")

    calls = [json.loads(line) for line in (path / "controller_rpc.jsonl").read_text().splitlines()]
    if len(calls) != (saved + 3) // 4:
        raise ValueError("saved controller calls do not match attempted periods")
    for call in calls:
        lower, upper = call["input_lower"], call["input_upper"]
        if (len(lower) != 4 or len(upper) != 4 or
                any(not math.isfinite(v) for v in lower + upper) or
                any(a > b for a, b in zip(lower, upper))):
            raise ValueError("malformed four-state controller request")
    if calls[0]["input_lower"] != [row[0] for row in INITIAL] or calls[0]["input_upper"] != [row[1] for row in INITIAL]:
        raise ValueError("first controller request does not cover the full raw initial box")

    lines = native_log.splitlines()
    complete = re.search(r"COMPLETED_PERIODS (\d+)/(\d+)", native_log)
    segments = re.search(r"FLOWPIPE_SEGMENTS (\d+)", native_log)
    target = re.search(r"TARGET_CHECKS (\d+)/400", native_log)
    last_period = re.findall(r"PERIOD (\d+) FLOWPIPES (\d+) STATUS (\d+)", native_log)
    if not complete or not segments or not target or not last_period:
        raise ValueError("native completion ledger missing")
    full_periods, planned = map(int, complete.groups())
    target_checks = int(target.group(1))
    if (planned != periods or full_periods != saved // 4 or int(segments.group(1)) != saved or
            target_checks != len((path / "target.tsv").read_text().splitlines()) - 1):
        raise ValueError("native log, saved ranges, and target ledger disagree")
    raw_range_scan = json.loads((path / "INDEPENDENT_RANGE_SCAN.json").read_text())
    if (raw_range_scan["records"] != saved or raw_range_scan["nonfinite_records"] or
            raw_range_scan["reversed_component_intervals"] or
            raw_range_scan["endpoint_outside_same_step_tube_components"]):
        raise ValueError("independent interval scan differs or rejects saved ranges")
    status = "completed_short_prefix" if periods == 1 and saved == 4 and result["exit_code"] == 0 else "early_stopped"
    if periods == 500 and (saved == 2000 or result["exit_code"] != 2):
        raise ValueError("full attempt status does not match the saved early stop")
    return {"schema": "archcomp26-balancing-native-coverage-audit-nohash-v1",
            "source": str(path), "planned_periods": periods, "planned_substeps": periods * 4,
            "saved_substeps": saved, "last_saved_t_s": saved * 0.005,
            "complete_periods": full_periods, "controller_calls": len(calls),
            "first_incomplete_period": None if saved == periods * 4 else int(last_period[-1][0]),
            "first_incomplete_period_flowpipes": None if saved == periods * 4 else int(last_period[-1][1]),
            "first_incomplete_period_native_status": None if saved == periods * 4 else int(last_period[-1][2]),
            "target_window_checks": target_checks, "status": status,
            "property_verdict": "NOT_APPLICABLE_SHORT_PREFIX" if periods == 1 else "UNKNOWN_INCOMPLETE",
            "outer_exit_code": result["exit_code"], "outer_wall_s": result["wall_s"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--periods", type=int, choices=(1, 500), required=True)
    args = parser.parse_args()
    (args.directory / "COVERAGE_AUDIT.json").write_text(
        json.dumps(audit(args.directory, args.periods), indent=2, allow_nan=False) + "\n")
