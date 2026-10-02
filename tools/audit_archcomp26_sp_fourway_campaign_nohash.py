#!/usr/bin/env python3
"""Independently scan saved SP campaign files; never start a solver."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import struct


RECORD = struct.Struct("<QQd8d")
METHODS = ("native", "huan", "xiangru", "ours_p3")
LABELS = {"native": "flowstar_native", "huan": "huan-p2",
          "xiangru": "xiangru-original-p2", "ours_p3": "ours_p3"}


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scan_intervals(rows, name):
    require(len(rows) == 100, f"{name}: expected 100 saved substeps")
    window_tubes = []
    window_endpoints = []
    for expected, (step, tube, endpoint) in enumerate(rows, 1):
        require(step == expected and len(tube) == len(endpoint) == 2,
                f"{name}: substep or state width {expected}")
        for state in range(2):
            t, e = tube[state], endpoint[state]
            require(len(t) == len(e) == 2 and all(map(math.isfinite, (*t, *e)))
                    and t[0] <= e[0] <= e[1] <= t[1],
                    f"{name}: invalid tube/endpoint {expected}/{state}")
        if expected == 1:
            require(tube[0][0] <= 1 and tube[0][1] >= 1.175
                    and tube[1][0] <= 0 and tube[1][1] >= .2,
                    f"{name}: first tube misses initial box")
        if expected == 50:
            require(0 <= endpoint[0][0] <= endpoint[0][1] <= 1,
                    f"{name}: t=.5 endpoint violates closed property")
        if expected >= 51:
            require(0 <= tube[0][0] <= tube[0][1] <= 1,
                    f"{name}: closed-window tube outside property {expected}")
            window_tubes.append(tube[0])
            window_endpoints.append(endpoint[0])
    return {
        "records": 100,
        "closed_window_tube_records": 50,
        "closed_window_x1_tube_union": [min(v[0] for v in window_tubes),
                                         max(v[1] for v in window_tubes)],
        "closed_window_x1_tube_width_mean_max": [
            statistics.mean(v[1] - v[0] for v in window_tubes),
            max(v[1] - v[0] for v in window_tubes)],
        "closed_window_x1_endpoint_width_mean_max": [
            statistics.mean(v[1] - v[0] for v in window_endpoints),
            max(v[1] - v[0] for v in window_endpoints)],
    }


def native_data(run_dir, model):
    lines = (run_dir / "native.log").read_text().splitlines()
    steps = [line for line in lines if line.startswith("Step ")]
    require(steps == [f"Step {k}" for k in range(20)]
            and "COMPLETED_PERIODS 20/20" in lines and "VERIFIED" in lines,
            f"{run_dir}: author log incomplete")
    rpc = read_jsonl(run_dir / "controller_rpc.jsonl")
    require(len(rpc) == 20 and all(len(r["input_lower"]) == len(r["input_upper"]) == 2
                                   for r in rpc), f"{run_dir}: RPC transcript")
    posts = sum('"POST / HTTP/1.1" 200' in line
                for line in (run_dir / "server.log").read_text().splitlines())
    require(posts == 20, f"{run_dir}: HTTP response count {posts}")
    raw = (run_dir / "ranges.bin").read_bytes()
    require(len(raw) == 100 * RECORD.size, f"{run_dir}: native range file length")
    rows = []
    for expected, record in enumerate(RECORD.iter_unpack(raw), 1):
        lane, step, h, *v = record
        require(lane == 0 and step == expected and math.isfinite(h)
                and abs(h - .01) < 1e-12, f"{run_dir}: native range index/domain {expected}")
        rows.append((step, (v[0:2], v[4:6]), (v[2:4], v[6:8])))
    return {**scan_intervals(rows, run_dir), "controller_rpc": 20,
            "http_200_responses": posts, "controller_calls": None,
            "author_verdict": "VERIFIED", "model": model}


def gpu_data(run_dir, method, model, config):
    data = run_dir / "data"
    start, result, metrics = (read_json(data / name)
                              for name in ("START.json", "RESULT.json", "metrics.json"))
    require(start["model"] == model and start["source_config"] == config
            and start["physical_states"] == ["x1", "x2"]
            and start["cpu_affinity"] == [10, 11, 12, 13]
            and start.get("cuda_visible_devices", start.get("physical_gpu")) == "2"
            and start["mode"] == "full", f"{run_dir}: GPU contract/resources")
    if method != "ours_p3":
        require(start["backend"] == method and result["backend"] == method,
                f"{run_dir}: GPU method")
    require(result["status"] == "completed" and result["driver_return"] == 0
            and result["completed_substeps"] == result["expected_substeps"] == 100
            and result["all_substeps_accepted"] is True and result["metrics_broken"] == 0
            and metrics["broken"] == 0 and metrics["steps"] == 20 and metrics["substeps"] == 5
            and [s["k"] for s in metrics["ctrl_steps"]] == list(range(20)),
            f"{run_dir}: GPU completion/NN calls")
    lines = (run_dir / "stdout.log").read_text().splitlines()
    require([line for line in lines if line.startswith("Step ")] ==
            [f"Step {k}" for k in range(20)] and
            any("checked on 50 of 100 substeps, 0 partial" in line for line in lines),
            f"{run_dir}: author closed-window check")
    ranges = read_jsonl(data / "ranges.jsonl")
    rows = []
    for expected, row in enumerate(ranges, 1):
        require(row["substep"] == expected and row["accepted"] is True,
                f"{run_dir}: accepted range index {expected}")
        if "local_h" in row:
            require(abs(row["local_h"] - .01) < 1e-12,
                    f"{run_dir}: local h {expected}")
        rows.append((expected, row["tube"], row["endpoint"]))
    if method == "ours_p3":
        safety = read_jsonl(data / "safety.jsonl")
        require(len(safety) == result["safety_events"] == 50
                and result["author_safe_bounds_all_nonpositive"] is True
                and [v["event_index"] for v in safety] == list(range(1, 51))
                and all(v["max_violation_upper"] <= 0 for v in safety),
                f"{run_dir}: author safety events")
    else:
        safety = None
    return {**scan_intervals(rows, run_dir), "controller_rpc": None,
            "http_200_responses": None, "controller_calls": 20,
            "author_safety_events": len(safety) if safety is not None else None,
            "author_verdict": "window safe", "model": model}


def main(root):
    plan, summary = read_json(root / "PLAN.json"), read_json(root / "SUMMARY.json")
    events = read_jsonl(root / "events.jsonl")
    expected = [(r["round"], r["phase"], method) for r in plan["rounds"]
                for method in r["order"]]
    require(len(expected) == len(events) == 24 and
            [(e["round"], e["phase"], e["method"]) for e in events] == expected
            and summary["attempts_started"] == 24
            and summary["all_samples_valid"] is True
            and summary["first_failure_or_resource_stop"] is None,
            "campaign is not 24 valid runs in planned order")
    rows = []
    seen_dirs = set()
    for event in events:
        method = event["method"]
        run_dir = root / Path(event["directory"]).name
        require(method in METHODS and run_dir.is_dir() and run_dir not in seen_dirs,
                f"missing/duplicate raw run {run_dir}")
        seen_dirs.add(run_dir)
        start, result = read_json(run_dir / "START.json"), read_json(run_dir / "RESULT.json")
        require(start["method"] == LABELS[method]
                and start["instance"] == "single-pendulum-reach"
                and start["contract_label"] == "sp-paper-two-physical-state-full20-campaign"
                and start["argv"][:3] == ["/usr/bin/taskset", "-c", "10-13"]
                and start["selected_environment"]["CUDA_VISIBLE_DEVICES"] == "2"
                and result["status"] == event["run_status"] == "completed"
                and result["exit_code"] == event["exit_code"] == 0
                and result["timed_out"] is False
                and result["wall_s"] == event["process_wall_s"]
                and event["valid_full_property_sample"] is True,
                f"{run_dir}: outer identity/result mismatch")
        if method == "native":
            require(start["selected_environment"]["SP_CPUSET"] == "10-13"
                    and start["selected_environment"]["SP_MODEL"] == plan["fixed_model"],
                    f"{run_dir}: native runtime resource/model")
            detail = native_data(run_dir, plan["fixed_model"])
        else:
            detail = gpu_data(run_dir, method, plan["fixed_model"], plan["fixed_config"])
        require(detail["closed_window_x1_tube_union"] ==
                event["detail"]["closed_window_x1_tube_union"],
                f"{run_dir}: saved tube/event mismatch")
        rows.append({"round": event["round"], "phase": event["phase"],
                     "method": method, "run_directory_remote": event["directory"],
                     "outer_wall_s": result["wall_s"], "outer_status": result["status"],
                     "periods": 20, "substeps": 100, **detail})
    groups = {}
    for method in METHODS:
        first = [r["outer_wall_s"] for r in rows
                 if r["method"] == method and r["phase"] == "first_process"]
        later = [r["outer_wall_s"] for r in rows
                 if r["method"] == method and r["phase"] == "later_process"]
        require(len(first) == 1 and len(later) == 5, f"{method}: timing sample count")
        groups[method] = {"first_process_wall_s": first[0],
                          "later_process_walls_s": later,
                          "later_median_s": statistics.median(later),
                          "later_min_s": min(later), "later_max_s": max(later)}
        require(groups[method]["later_median_s"] ==
                summary["later_process_wall_medians_s"][method]
                and [min(later), max(later)] ==
                summary["later_process_wall_minmax_s"][method],
                f"{method}: aggregate mismatch")
    with (root / "RUNS.csv").open("x", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "INDEPENDENT_AUDIT.json").open("x") as target:
        json.dump({"schema": "archcomp26-sp-two-state-campaign-saved-evidence-audit-v1",
                   "record_count": len(rows), "saved_range_records": 2400,
                   "native_rpc_requests": 120, "all_saved_closed_window_tubes_safe": True,
                   "groups": groups, "runs": rows,
                   "limits": ["Two physical states plus an auxiliary checker clock; the official MATLAB third-state initial value is unresolved.",
                              "Saved tube scan does not independently certify floating-point NN bounds.",
                              "First means fresh process in this campaign, not a rebooted host/GPU; shared-host timing is descriptive."]},
                  target, indent=2, allow_nan=False)
        target.write("\n")
    for method, group in groups.items():
        print(method, group["first_process_wall_s"], group["later_median_s"],
              group["later_min_s"], group["later_max_s"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
