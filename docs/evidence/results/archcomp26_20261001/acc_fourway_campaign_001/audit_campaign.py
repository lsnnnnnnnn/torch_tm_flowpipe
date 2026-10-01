#!/usr/bin/env python3
"""Recheck saved ACC campaign records; write only derived CSV/JSON."""

import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics
import struct


ROOT = Path(__file__).resolve().parent
RECORD = struct.Struct("<QQd" + "d" * 24)
METHODS = ("native", "huan", "xiangru", "ours_p3")


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def margin(tube):
    # Exact rational evaluation of the saved binary64 endpoints and 7/5.
    return (Fraction.from_float(tube[0][0])
            - Fraction.from_float(tube[3][1])
            - Fraction(7, 5) * Fraction.from_float(tube[4][1]) - 10)


def native_data(run_dir):
    lines = (run_dir / "native.log").read_text().splitlines()
    steps = [line for line in lines if line.startswith("Step ")]
    require(steps == [f"Step {i}" for i in range(50)], f"{run_dir}: native steps")
    require(lines[-2] == "VERIFIED", f"{run_dir}: native verdict")
    rpc = read_lines(run_dir / "controller_rpc.jsonl")
    require(len(rpc) == 50, f"{run_dir}: RPC count")
    require(all(x["input_lower"][:2] == [30.0, 1.4] and
                x["input_upper"][:2] == [30.0, 1.4] for x in rpc),
            f"{run_dir}: fixed controller features")
    posts = sum('"POST / HTTP/1.1" 200' in line
                for line in (run_dir / "server.log").read_text().splitlines())
    require(posts == 50, f"{run_dir}: server POST count {posts}")
    raw = (run_dir / "ranges.bin").read_bytes()
    require(len(raw) == 50 * RECORD.size, f"{run_dir}: range byte count")
    margins = []
    for i, values in enumerate(RECORD.iter_unpack(raw), 1):
        lane, step, h, *state = values
        require((lane, step) == (0, i) and abs(h - .1) < 1e-12,
                f"{run_dir}: lane/step/h {i}")
        tube = []
        for j in range(6):
            lo, hi, end_lo, end_hi = state[4*j:4*j+4]
            require(all(map(math.isfinite, (lo, hi, end_lo, end_hi))) and
                    lo <= hi and end_lo <= end_hi, f"{run_dir}: invalid range {i}/{j}")
            tube.append((lo, hi))
        margins.append(margin(tube))
    minimum = min(margins)
    require(minimum > 0, f"{run_dir}: saved tube not safe")
    return 50, 50, 50, None, "VERIFIED", float(minimum)


def gpu_data(run_dir, method, fixed_model):
    data = run_dir / "data"
    start = read_json(data / "START.json")
    result = read_json(data / "RESULT.json")
    ranges = read_lines(data / "ranges.jsonl")
    safety = read_lines(data / "safety.jsonl")
    metrics = read_json(data / "metrics.json")
    require(start["model"] == fixed_model and
            start["controller_input"] == ["30", "1.4", "v_ego",
                                          "x_lead-x_ego", "v_lead-v_ego"],
            f"{run_dir}: controller contract")
    require(start["cpu_affinity"] == [10, 11, 12, 13] and
            start.get("cuda_visible_devices", start.get("physical_gpu")) == "2",
            f"{run_dir}: resources")
    require(start.get("backend", "ours_p3") == method if method != "ours_p3"
            else start.get("backend") is None, f"{run_dir}: method")
    require(len(ranges) == len(safety) == 50 and metrics["steps"] == 50,
            f"{run_dir}: range/safety/metric count")
    require(result["status"] == "completed" and result["driver_return"] == 0 and
            result["completed_substeps"] == result["expected_substeps"] == 50 and
            result["safety_events"] == 50 and
            result["adapter_feature_calls"] == result["adapter_injection_calls"] == 50,
            f"{run_dir}: GPU completion")
    margins = []
    for i, (row, check) in enumerate(zip(ranges, safety), 1):
        require(row["substep"] == check["substep"] == i and row["accepted"] is True
                and abs(row["local_h"] - .1) < 1e-12, f"{run_dir}: step {i}")
        tube = row["tube"]
        require(len(tube) == 6 and all(len(x) == 2 and all(map(math.isfinite, x))
                    and x[0] <= x[1] for x in tube), f"{run_dir}: tube {i}")
        require(check["max_violation_upper"] <= 0, f"{run_dir}: author safety {i}")
        margins.append(margin(tube))
    minimum = min(margins)
    require(minimum > 0 and result["author_safe_bounds_all_nonpositive"] is True
            and result["independent_tube_boxes_all_safe"] is True
            and abs(float(minimum) - result["independent_tube_halfspace_min_lower"]) < 1e-8,
            f"{run_dir}: saved property mismatch")
    return 50, None, 50, 50, "author_safe_and_saved_tube_safe", float(minimum)


def main():
    plan = read_json(ROOT / "PLAN.json")
    source_summary = read_json(ROOT / "SUMMARY.json")
    events = read_lines(ROOT / "events.jsonl")
    expected = [(r["round"], r["phase"], m) for r in plan["rounds"] for m in r["order"]]
    require(len(events) == len(expected) == 24, "campaign count")
    require([(e["round"], e["phase"], e["method"]) for e in events] == expected,
            "campaign order")
    require(source_summary["attempts_started"] == 24, "source summary count")
    rows = []
    for event in events:
        method = event["method"]
        run_dir = ROOT / Path(event["directory"]).name
        require(method in METHODS and run_dir.is_dir(), f"missing {run_dir}")
        outer_start = read_json(run_dir / "START.json")
        outer_result = read_json(run_dir / "RESULT.json")
        require(outer_start["method"] == method and
                outer_start["argv"][:3] == ["/usr/bin/taskset", "-c", "10-13"] and
                outer_start["selected_environment"]["CUDA_VISIBLE_DEVICES"] == "2",
                f"{run_dir}: outer identity/resources")
        require(outer_result["status"] == event["run_status"] == "completed" and
                outer_result["exit_code"] == event["exit_code"] == 0 and
                outer_result["timed_out"] is False and
                outer_result["wall_s"] == event["process_wall_s"],
                f"{run_dir}: outer result mismatch")
        if method == "native":
            require(outer_start["selected_environment"]["ACC_PORT"] == "5102",
                    f"{run_dir}: native port")
            steps, rpc, calls, safety, outcome, minimum = native_data(run_dir)
        else:
            steps, rpc, calls, safety, outcome, minimum = gpu_data(
                run_dir, method, plan["fixed_model"])
        require(event["valid_full_property_sample"] is True,
                f"{run_dir}: event invalid")
        rows.append({"round": event["round"], "phase": event["phase"],
                     "method": method, "run_directory_remote": event["directory"],
                     "run_directory_local": str(run_dir),
                     "started_utc": event["started_utc"],
                     "ended_utc": event["ended_utc"],
                     "outer_wall_s": outer_result["wall_s"],
                     "outer_status": outer_result["status"],
                     "exit_code": outer_result["exit_code"],
                     "completed_steps": steps, "rpc_count": rpc,
                     "controller_calls": calls, "safety_events": safety,
                     "range_records": 50, "property_outcome": outcome,
                     "min_saved_tube_margin": minimum,
                     "independent_saved_tube_safe": minimum > 0})
    groups = {}
    for method in METHODS:
        steady = [r["outer_wall_s"] for r in rows
                  if r["method"] == method and r["phase"] == "steady_process"]
        cold = [r["outer_wall_s"] for r in rows
                if r["method"] == method and r["phase"] == "cold_process"]
        require(len(cold) == 1 and len(steady) == 5, f"{method}: group count")
        groups[method] = {"cold_outer_wall_s": cold[0],
                          "steady_outer_wall_s": steady,
                          "steady_median_s": statistics.median(steady),
                          "steady_min_s": min(steady), "steady_max_s": max(steady)}
        require(groups[method]["steady_median_s"] ==
                source_summary["steady_process_wall_medians_s"][method],
                f"{method}: median mismatch")
    with (ROOT / "RUNS.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "INDEPENDENT_AUDIT.json").write_text(json.dumps({
        "schema": "archcomp26-acc-fourway-saved-evidence-audit-v1",
        "source_plan": "PLAN.json", "source_event_journal": "events.jsonl",
        "record_count": len(rows), "all_outer_runs_completed": True,
        "all_saved_tubes_safe": all(r["independent_saved_tube_safe"] for r in rows),
        "groups": groups, "runs": rows,
        "limits": ["Saved axis-aligned tube scan does not independently certify NN floating-point bounds.",
                   "Cold means first fresh process in this campaign, not a rebooted host or GPU.",
                   "Concurrent jobs on the host may affect process wall time."],
    }, indent=2, allow_nan=False) + "\n")
    for method, group in groups.items():
        print(method, group["cold_outer_wall_s"], group["steady_median_s"],
              group["steady_min_s"], group["steady_max_s"])


if __name__ == "__main__":
    main()
