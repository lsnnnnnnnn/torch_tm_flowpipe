"""Read original TORA sigmoid campaign records; write a no-digest audit and CSV."""

import csv
import json
import math
import statistics
import struct
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METHODS = ("native", "huan", "xiangru", "ours_p3")
ROW = struct.Struct("<QQd16d")
INITIAL = ((-0.77, -0.75), (-0.45, -0.43), (0.51, 0.54),
           (-0.3, -0.28), (0.0, 0.0), (0.0, 0.0))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def scan_ranges(path):
    raw = path.read_bytes()
    require(len(raw) == 500 * ROW.size, f"range size: {path}")
    last, h_dev = None, 0.0
    for step, row in enumerate(ROW.iter_unpack(raw), 1):
        lane, actual_step, h = row[:3]
        require(lane == 0 and actual_step == step, f"range grid: {path}:{step}")
        require(math.isfinite(h) and h > 0 and abs(h - 0.01) <= 1e-9,
                f"saved time width: {path}:{step}")
        h_dev = max(h_dev, abs(h - 0.01))
        for coordinate in range(4):
            lo, hi, endpoint_lo, endpoint_hi = row[3 + 4 * coordinate:7 + 4 * coordinate]
            require(all(map(math.isfinite, (lo, hi, endpoint_lo, endpoint_hi)))
                    and lo <= endpoint_lo <= endpoint_hi <= hi,
                    f"tube/endpoint: {path}:{step}:{coordinate}")
        last = row
    x1, x2 = (last[5], last[6]), (last[9], last[10])
    require(-0.1 <= x1[0] <= x1[1] <= 0.2
            and -0.9 <= x2[0] <= x2[1] <= -0.6,
            f"terminal target: {path}")
    return {"records": 500, "max_h_deviation": h_dev, "terminal_x1": x1,
            "terminal_x2": x2, "target_contained": True}


def inspect(event):
    method = event["method"]
    sample = ROOT / Path(event["path"]).name
    require(method in METHODS and sample.is_dir(), f"sample identity: {event}")
    outer = sample if method == "native" else sample / "outer"
    data = sample if method == "native" else sample / "payload"
    outer_start, outer_result = read_json(outer / "START.json"), read_json(outer / "RESULT.json")
    require(outer_start["instance"] == "tora-reach-sigmoid"
            and outer_start["method"] == method
            and outer_start["contract_label"] == "tora-reach-official2026-mat-u11f-campaign"
            and outer_start["selected_environment"]["CUDA_VISIBLE_DEVICES"] == "2"
            and outer_start["argv"][:3] == ["taskset", "-c", "10-13"],
            f"outer START contract/resource: {sample.name}")
    wall = outer_result["wall_s"]
    require(outer_result["status"] == "completed"
            and outer_result["exit_code"] == event["exit_code"] == 0
            and event["valid"] is True and event["wall_s"] == wall
            and math.isfinite(wall) and wall > 0,
            f"outer/event result: {sample.name}")
    if method == "native":
        boxes = read_json(sample / "initial_boxes.json")
        require(len(boxes) == 1 and len(boxes[0]) == 6
                and all(abs(boxes[0][i][j] - INITIAL[i][j]) <= 1e-15
                        for i in range(6) for j in range(2)),
                f"native initial box: {sample.name}")
        steps = [line for line in read_lines(sample / "native.log") if line.startswith("Step ")]
        rpc = read_lines(sample / "controller_rpc.jsonl")
        http = [line for line in read_lines(sample / "server.log") if " 200 " in line]
        require(steps == [f"Step {k}" for k in range(10)]
                and len(rpc) == len(http) == 10
                and "VERIFIED" in read_lines(sample / "native.log"),
                f"native step/RPC/verdict: {sample.name}")
        verdict = "VERIFIED at T=5 endpoint"
        rpc_count = 10
    else:
        inner_start, inner_result = read_json(data / "START.json"), read_json(data / "RESULT.json")
        metrics = read_json(data / "metrics.json")
        require(inner_start["gpu_physical"] == 2
                and inner_start["cpu_affinity"] == [10, 11, 12, 13]
                and inner_start["contract"]["initial_box"] == [list(pair) for pair in INITIAL]
                and inner_start["contract"]["internal_control"] == "u = 11 * f(x) + 0"
                and inner_start["contract"]["control_periods"] == 10
                and inner_start["contract"]["ode_substeps"] == 500
                and inner_result["status"] == "completed_full_numerical_horizon"
                and inner_result["observed_substeps"] == inner_result["accepted_substeps"] == 500
                and inner_result["property_evaluated"] is False
                and metrics["steps"] == 10 and metrics["broken"] == 0,
                f"inner START/RESULT/metrics: {sample.name}")
        verdict = "NOT_EVALUATED"
        rpc_count = 0
    ranges = scan_ranges(data / "ranges.bin")
    return {"round": event["round"], "phase": event["phase"], "method": method,
            "sample": sample.name, "wall_s": wall, "control_periods": 10,
            "accepted_substeps": 500, "range_records": ranges["records"],
            "rpc_requests": rpc_count, "author_property_verdict": verdict,
            "max_h_deviation": ranges["max_h_deviation"],
            "terminal_x1": ranges["terminal_x1"],
            "terminal_x2": ranges["terminal_x2"]}


def main():
    plan = read_json(ROOT / "PLAN.json")
    summary = read_json(ROOT / "SUMMARY.json")
    events = [json.loads(line) for line in read_lines(ROOT / "events.jsonl")]
    require(summary["expected"] == summary["started"] == len(events) == 24
            and summary["all_valid"] is True and summary["first_stop"] is None,
            "campaign completeness")
    expected = [(round_no, "first" if round_no == 0 else "later", method)
                for round_no in range(6)
                for method in METHODS[round_no % 4:] + METHODS[:round_no % 4]]
    actual = [(e["round"], e["phase"], e["method"]) for e in events]
    require(actual == expected and len({e["path"] for e in events}) == 24,
            "event order/uniqueness")
    require(len(plan["rounds"]) == 6 and "u=11f" in plan["contract"],
            "PLAN contract")
    for a, b in zip(events, events[1:]):
        require(datetime.fromisoformat(a["ended_utc"]) <= datetime.fromisoformat(b["started_utc"]),
                "overlapping events")
    rows = [inspect(event) for event in events]
    timing = {}
    for method in METHODS:
        first = [row["wall_s"] for row in rows if row["method"] == method and row["phase"] == "first"]
        later = [row["wall_s"] for row in rows if row["method"] == method and row["phase"] == "later"]
        require(len(first) == 1 and len(later) == summary["later_valid_counts"][method] == 5,
                f"sample counts: {method}")
        median = statistics.median(later)
        require(median == summary["later_median_wall_s"][method],
                f"campaign median: {method}")
        timing[method] = {"first_wall_s": first[0], "later_count": 5,
                          "later_median_wall_s": median,
                          "later_min_wall_s": min(later), "later_max_wall_s": max(later)}
    result = {"status": "24_original_slots_checked", "source": "local mirror of original server receipts",
              "runs_checked": 24, "saved_range_records_checked": 12000,
              "native_rpc_requests_checked": 60,
              "all_terminal_target_coordinates_contained": True,
              "timing": timing, "rows": rows,
              "qualification": "Descriptive fresh-process timing only; no independent floating-point NNCS certificate or stable speed ranking; first aborted campaign excluded; no content digest."}
    (ROOT / "INDEPENDENT_AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    columns = ["round", "phase", "method", "sample", "wall_s", "control_periods",
               "accepted_substeps", "range_records", "rpc_requests", "author_property_verdict",
               "max_h_deviation", "terminal_x1", "terminal_x2"]
    with (ROOT / "RUNS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    print(json.dumps({"status": result["status"], "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
