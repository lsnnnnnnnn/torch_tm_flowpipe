#!/usr/bin/env python3
"""Independently read saved Attitude campaign evidence; never launch a solver."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import struct


RECORD = struct.Struct("<QQd24d")
METHODS = ("native", "huan", "xiangru", "ours_p3")
LABELS = {"native": "flowstar_native", "huan": "huan",
          "xiangru": "xiangru", "ours_p3": "pytorch_gpu"}
UNSAFE = ((-.2, 0), (-.5, -.4), (0, .2), (-.7, -.6), (.7, .8), (-.4, -.2))
INITIAL = ((-.45, -.44), (-.55, -.54), (.65, .66),
           (-.75, -.74), (.85, .86), (-.65, -.64))


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def scan_ranges(path):
    raw = path.read_bytes()
    require(len(raw) == 60 * RECORD.size, f"{path}: expected 60 six-state records")
    union = [[math.inf, -math.inf] for _ in range(6)]
    min_gap = math.inf
    terminal = None
    for expected, values in enumerate(RECORD.iter_unpack(raw), 1):
        lane, step, h, *states = values
        require(lane == 0 and step == expected and math.isfinite(h)
                and abs(h - .05) < 1e-12, f"{path}: grid {expected}")
        separating_gaps = []
        endpoint = []
        for i, (unsafe_lo, unsafe_hi) in enumerate(UNSAFE):
            lo, hi, end_lo, end_hi = states[4*i:4*i+4]
            require(all(map(math.isfinite, (lo, hi, end_lo, end_hi)))
                    and lo <= end_lo <= end_hi <= hi,
                    f"{path}: invalid tube or endpoint at {expected}/{i}")
            if expected == 1:
                require(lo <= INITIAL[i][0] and INITIAL[i][1] <= hi,
                        f"{path}: initial box not covered at {i}")
            union[i][0] = min(union[i][0], lo)
            union[i][1] = max(union[i][1], hi)
            separating_gaps.append(max(unsafe_lo - hi, lo - unsafe_hi))
            endpoint.append([end_lo, end_hi])
        gap = max(separating_gaps)
        require(gap > 0, f"{path}: tube intersects closed official unsafe box at {expected}")
        min_gap = min(min_gap, gap)
        terminal = endpoint
    return {"range_records": 60, "minimum_box_separation_gap": min_gap,
            "full_tube_union": union, "terminal_endpoint": terminal,
            "terminal_endpoint_width": [hi-lo for lo, hi in terminal]}


def native_data(run_dir, plan):
    lines = (run_dir / "native.log").read_text().splitlines()
    require([line for line in lines if line.startswith("Step ")] ==
            [f"Step {k}" for k in range(30)]
            and "COMPLETED_PERIODS 30/30" in lines
            and "FLOWPIPE_SEGMENTS 60" in lines and "VERIFIED" in lines,
            f"{run_dir}: native completion/verdict")
    rpc = read_jsonl(run_dir / "controller_rpc.jsonl")
    require(len(rpc) == 30 and
            all(len(row["input_lower"]) == len(row["input_upper"]) == 6
                for row in rpc), f"{run_dir}: RPC request count/width")
    responses = sum('"POST / HTTP/1.1" 200' in line for line in
                    (run_dir / "server.log").read_text().splitlines())
    require(responses == 30, f"{run_dir}: HTTP response count")
    return {**scan_ranges(run_dir / "ranges.bin"), "controller_rpc": 30,
            "http_200": 30, "controller_calls": None,
            "author_verdict": "VERIFIED", "model": plan["fixed_model"]}


def gpu_data(run_dir, method, plan):
    payload = run_dir / "payload"
    start, result, metrics = (read_json(payload / name)
                              for name in ("START.json", "RESULT.json", "metrics.json"))
    model = start.get("controller", start.get("model"))
    require(model == plan["fixed_model"]
            and start["mode"] == "full"
            and start["cpu_affinity"] == [10, 11, 12, 13]
            and start.get("cuda_visible_devices", start.get("physical_gpu")) == "2",
            f"{run_dir}: GPU model/mode/resources")
    require(result["status"] == "completed" and result["driver_return"] == 0
            and result["expected_substeps"] == 60
            and result["saved_tubes_box_disjoint_official_unsafe"] is True
            and result["author_checker_lines"] == []
            and metrics["steps"] == 30 and metrics["substeps"] == 2
            and metrics["broken"] == 0
            and [row["k"] for row in metrics["ctrl_steps"]] == list(range(30)),
            f"{run_dir}: GPU completion/checker/controller count")
    require([line for line in (run_dir / "stdout.log").read_text().splitlines()
             if line.startswith("Step ")] == [f"Step {k}" for k in range(30)],
            f"{run_dir}: missing period output")
    observations = read_jsonl(payload / "observations.jsonl")
    require(len(observations) == 60 and
            all(row["substep"] == k and row["tube_box_disjoint_official_unsafe"] is True
                for k, row in enumerate(observations, 1)),
            f"{run_dir}: observations")
    if method == "ours_p3":
        require(result["completed_substeps"] == result["accepted_substeps"] == 60
                and result["all_substeps_accepted"] is True
                and all(row["accepted"] is True for row in observations),
                f"{run_dir}: P3 accepted substeps")
    else:
        require(start["backend"] == method and result["backend"] == method
                and result["observed_substeps"] == result["accepted_lane_substeps"] == 60
                and result["all_lanes_accepted"] is True
                and all(row["accepted_count"] == 1 for row in observations),
                f"{run_dir}: author accepted substeps")
    config = (payload / "config.yaml").read_text()
    require("- -x4 - 0.7" in config and "- -x4 - 0.4" not in config,
            f"{run_dir}: uncorrected unsafe box")
    return {**scan_ranges(payload / "ranges.bin"), "controller_rpc": None,
            "http_200": None, "controller_calls": 30,
            "author_verdict": "checker silence plus saved box disjointness",
            "model": model}


def main(root):
    plan, summary = read_json(root / "PLAN.json"), read_json(root / "SUMMARY.json")
    events = read_jsonl(root / "events.jsonl")
    expected = [(round_plan["round"], round_plan["phase"], method)
                for round_plan in plan["rounds"] for method in round_plan["order"]]
    require(len(expected) == len(events) == 24
            and [(event["round"], event["phase"], event["method"])
                 for event in events] == expected
            and summary["attempts_started"] == 24
            and summary["all_samples_valid"] is True
            and summary["first_failure_or_resource_stop"] is None,
            "campaign is not 24 valid runs in planned order")
    require([tuple(v) for v in plan["unsafe_box"]] == list(UNSAFE)
            and [tuple(v) for v in plan["initial_physical_box"]] == list(INITIAL),
            "plan box mismatch")
    original = (root / "original_p3_source.py").read_text()
    staged = (root / "p3_source/archcomp26_attitude_p3_gpu2_campaign_nohash.py").read_text()
    require(original.count('if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":') == 1
            and original.count("reserved for physical GPU3") == 1
            and staged == original.replace(
                'if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":',
                'if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":').replace(
                "reserved for physical GPU3", "reserved for physical GPU2"),
            "P3 staged launcher changes exceed the documented GPU2 guard adaptation")
    rows, seen = [], set()
    previous_end = None
    for event in events:
        started = datetime.fromisoformat(event["started_utc"])
        ended = datetime.fromisoformat(event["ended_utc"])
        require(started < ended and (previous_end is None or started >= previous_end),
                "run event chronology overlaps or reverses")
        previous_end = ended
        method = event["method"]
        run_dir = root / Path(event["directory"]).name
        require(method in METHODS and run_dir.is_dir() and run_dir not in seen,
                f"missing/duplicate raw run {run_dir}")
        seen.add(run_dir)
        start, result = read_json(run_dir / "START.json"), read_json(run_dir / "RESULT.json")
        require(start["instance"] == "attitude-control-avoid"
                and start["method"] == LABELS[method]
                and start["contract_label"] ==
                    "attitude-avoid-official-unsafe-corrected-full30-campaign"
                and start["argv"][:3] == ["/usr/bin/taskset", "-c", "10-13"]
                and start["selected_environment"]["CUDA_VISIBLE_DEVICES"] == "2"
                and result["status"] == event["run_status"] == "completed"
                and result["exit_code"] == event["exit_code"] == 0
                and result["timed_out"] is False
                and result["wall_s"] == event["process_wall_s"]
                and event["valid_full_property_sample"] is True,
                f"{run_dir}: outer identity/status/resources")
        if method == "native":
            require(start["selected_environment"]["ATTITUDE_CPUSET"] == "10-13"
                    and start["selected_environment"]["ATTITUDE_MODEL"] == plan["fixed_model"],
                    f"{run_dir}: native runtime")
            details = native_data(run_dir, plan)
        else:
            details = gpu_data(run_dir, method, plan)
        require(details["full_tube_union"] == event["detail"]["full_tube_union"]
                and details["terminal_endpoint"] == event["detail"]["terminal_endpoint"],
                f"{run_dir}: raw range/event mismatch")
        rows.append({"round": event["round"], "phase": event["phase"],
                     "method": method, "run_directory_remote": event["directory"],
                     "outer_wall_s": result["wall_s"], "outer_status": result["status"],
                     "periods": 30, "segments": 60, **details})
    groups = {}
    for method in METHODS:
        first = [row["outer_wall_s"] for row in rows if
                 row["method"] == method and row["phase"] == "first_process"]
        later = [row["outer_wall_s"] for row in rows if
                 row["method"] == method and row["phase"] == "later_process"]
        require(len(first) == 1 and len(later) == 5, f"{method}: sample count")
        groups[method] = {"first_process_wall_s": first[0],
                          "later_process_walls_s": later,
                          "later_median_s": statistics.median(later),
                          "later_min_s": min(later), "later_max_s": max(later)}
        require(groups[method]["later_median_s"] ==
                summary["later_process_wall_medians_s"][method]
                and [min(later), max(later)] ==
                summary["later_process_wall_minmax_s"][method],
                f"{method}: aggregate mismatch")
    with (root / "RUNS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "INDEPENDENT_AUDIT.json").open("w") as stream:
        json.dump({"schema": "archcomp26-attitude-fourway-saved-evidence-audit-v1",
                   "record_count": 24, "saved_range_records": 1440,
                   "native_rpc_requests": 180,
                   "all_saved_tubes_disjoint_from_closed_official_unsafe": True,
                   "groups": groups, "runs": rows,
                   "limits": ["Saved six-state boxes do not independently certify floating-point NN bounds.",
                              "First means a fresh process in this campaign, not a rebooted host or GPU; shared-host timing is descriptive."]},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    for method, group in groups.items():
        print(method, group["first_process_wall_s"], group["later_median_s"],
              group["later_min_s"], group["later_max_s"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
