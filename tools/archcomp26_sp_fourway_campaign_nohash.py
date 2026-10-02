#!/usr/bin/env python3
"""Fresh-process four-method timing for the named SP two-state contract."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import socket
import statistics
import struct
import subprocess
import time


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
BASE = N / "runs/archcomp26_20261001"
PREP = BASE / "single_pendulum_prep_001"
BUILD = BASE / "native_sp_two_state_build_001/archcomp/single_pendulum"
PYTHON = N / "nncs_env/bin/python"
SUPERVISOR = BASE / "run_archcomp26_nohash.py"
CONFIG = PREP / "single_pendulum_paper_two_state.yaml"
GPU_LAUNCHER = PREP / "archcomp26_sp_two_state_gpu_nohash_v2.py"
P3_LAUNCHER = PREP / "archcomp26_sp_two_state_p3_nohash.py"
PAIR = PREP / "run_archcomp26_sp_two_state_native_pair.sh"
MODEL = PREP / "controller_single_pendulum.onnx"
OVERLAY = BASE / "native_dp_less_rpc_overlay_preflight_001/rpc_overlay"
METHODS = ("native", "huan", "xiangru", "ours_p3")
COMMON_ENV = ("CUDA_VISIBLE_DEVICES=2", "OMP_NUM_THREADS=1",
              "OPENBLAS_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1")
NATIVE_RECORD = struct.Struct("<QQd8d")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with path.open("x") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")


def gpu2_idle():
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True
    ).stdout
    rows = {int(parts[0]): (int(parts[1]), int(parts[2]))
            for line in output.splitlines() if (parts := [x.strip() for x in line.split(",")])}
    if 2 not in rows or rows[2][0] > 100 or rows[2][1] > 10:
        raise RuntimeError(f"GPU 2 is not idle: {rows.get(2)}")
    return rows[2]


def port_free():
    with socket.socket() as connection:
        return connection.connect_ex(("127.0.0.1", 5101)) != 0


def check_inputs():
    required = (PYTHON, SUPERVISOR, CONFIG, GPU_LAUNCHER, P3_LAUNCHER, PAIR,
                MODEL, BUILD / "sp_two_state_full20", BUILD / "observed_server_paper.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not OVERLAY.is_dir():
        raise FileNotFoundError({"missing_files": missing, "overlay": str(OVERLAY)})
    subprocess.run(["/usr/bin/taskset", "-c", "10-13", "/bin/true"], check=True)
    if not port_free():
        raise RuntimeError("SP RPC port 5101 is occupied")
    return gpu2_idle()


def command(method, run_dir):
    method_label = {"native": "flowstar_native", "huan": "huan-p2",
                    "xiangru": "xiangru-original-p2", "ours_p3": "ours_p3"}[method]
    args = [str(PYTHON), "-B", str(SUPERVISOR), "--run-dir", str(run_dir),
            "--instance", "single-pendulum-reach", "--method", method_label,
            "--contract-label", "sp-paper-two-physical-state-full20-campaign",
            "--cwd", str(BUILD if method == "native" else PREP),
            "--timeout-s", "120"]
    for item in COMMON_ENV:
        args.extend(("--env", item))
    if method == "native":
        for item in (f"SP_RUN_DIR={run_dir}", f"SP_WORKDIR={BUILD}",
                     f"SP_BINARY={BUILD / 'sp_two_state_full20'}",
                     f"SP_SERVER_PYTHON={PYTHON}", f"SP_OVERLAY={OVERLAY}",
                     f"SP_MODEL={MODEL}", "SP_CPUSET=10-13"):
            args.extend(("--env", item))
        child = ["/usr/bin/taskset", "-c", "10-13", "/bin/bash", str(PAIR)]
    elif method == "ours_p3":
        child = ["/usr/bin/taskset", "-c", "10-13", str(PYTHON), "-B", str(P3_LAUNCHER),
                 "--mode", "full", "--source-config", str(CONFIG), "--output", str(run_dir / "data")]
    else:
        child = ["/usr/bin/taskset", "-c", "10-13", str(PYTHON), "-B", str(GPU_LAUNCHER),
                 "--backend", method, "--mode", "full", "--source-config", str(CONFIG),
                 "--output", str(run_dir / "data")]
    return args + ["--"] + child


def scan_boxes(records):
    if len(records) != 100:
        raise ValueError(f"expected 100 range records, got {len(records)}")
    lo, hi = math.inf, -math.inf
    for step, tube, endpoint in records:
        if step < 1 or step > 100:
            raise ValueError(f"invalid substep {step}")
        for t, e in zip(tube, endpoint):
            if not all(map(math.isfinite, (*t, *e))) or t[0] > t[1] or e[0] > e[1]:
                raise ValueError(f"invalid interval at substep {step}")
            if e[0] < t[0] or e[1] > t[1]:
                raise ValueError(f"endpoint outside tube at substep {step}")
        if step == 1 and not (tube[0][0] <= 1 <= 1.175 <= tube[0][1]
                              and tube[1][0] <= 0 <= .2 <= tube[1][1]):
            raise ValueError("initial box outside first tube")
        if step == 50 and (endpoint[0][0] < 0 or endpoint[0][1] > 1):
            raise ValueError("closed-window start endpoint outside safety band")
        if step >= 51:
            lo, hi = min(lo, tube[0][0]), max(hi, tube[0][1])
    if lo < 0 or hi > 1:
        raise ValueError("saved closed-window tube outside safety band")
    return [lo, hi]


def native_ranges(path):
    data = path.read_bytes()
    if len(data) != NATIVE_RECORD.size * 100:
        raise ValueError("native range file is not 100 two-state records")
    records = []
    for expected, row in enumerate(NATIVE_RECORD.iter_unpack(data), 1):
        lane, step, h, *bounds = row
        if lane != 0 or step != expected or not math.isfinite(h) or abs(h - .01) > 1e-12:
            raise ValueError("native range record order/domain mismatch")
        records.append((step, (bounds[0:2], bounds[4:6]),
                        (bounds[2:4], bounds[6:8])))
    return scan_boxes(records)


def gpu_ranges(path):
    records = []
    for expected, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("substep") != expected or row.get("accepted") is not True:
            raise ValueError("GPU range order/acceptance mismatch")
        records.append((expected, row["tube"], row["endpoint"]))
    return scan_boxes(records)


def inspect_run(method, run_dir, exit_code):
    try:
        outer = json.loads((run_dir / "RESULT.json").read_text())
        wall = outer["wall_s"]
        if exit_code != 0 or outer.get("status") != "completed" or not math.isfinite(wall) or wall <= 0:
            raise ValueError("supervisor did not complete successfully")
        if method == "native":
            lines = (run_dir / "native.log").read_text().splitlines()
            steps = [line for line in lines if line.startswith("Step ")]
            if steps != [f"Step {k}" for k in range(20)] or "COMPLETED_PERIODS 20/20" not in lines or "VERIFIED" not in lines:
                raise ValueError("native author step/verdict mismatch")
            rpc = (run_dir / "controller_rpc.jsonl").read_text().splitlines()
            if len(rpc) != 20:
                raise ValueError("native RPC count is not 20")
            for line in rpc:
                request = json.loads(line)
                if len(request["input_lower"]) != 2 or len(request["input_upper"]) != 2:
                    raise ValueError("native RPC input width mismatch")
            tube_union = native_ranges(run_dir / "ranges.bin")
            detail = {"periods": 20, "substeps": 100, "controller_rpc": 20,
                      "author_verdict": "VERIFIED", "closed_window_x1_tube_union": tube_union}
        else:
            inner = json.loads((run_dir / "data/RESULT.json").read_text())
            metrics = json.loads((run_dir / "data/metrics.json").read_text())
            lines = (run_dir / "stdout.log").read_text().splitlines()
            steps = [line for line in lines if line.startswith("Step ")]
            if (inner.get("status") != "completed" or inner.get("driver_return") != 0
                    or inner.get("completed_substeps") != 100 or inner.get("expected_substeps") != 100
                    or inner.get("all_substeps_accepted") is not True or inner.get("metrics_broken") != 0
                    or metrics.get("broken") != 0 or len(metrics.get("ctrl_steps", [])) != 20
                    or [s.get("k") for s in metrics["ctrl_steps"]] != list(range(20))
                    or steps != [f"Step {k}" for k in range(20)]):
                raise ValueError("GPU author step/acceptance/controller mismatch")
            if not any("checked on 50 of 100 substeps, 0 partial" in line for line in lines):
                raise ValueError("GPU author closed-window checker mismatch")
            if method == "ours_p3":
                safety = [json.loads(line) for line in (run_dir / "data/safety.jsonl").read_text().splitlines()]
                if (inner.get("safety_events") != 50 or inner.get("author_safe_bounds_all_nonpositive") is not True
                        or len(safety) != 50 or any(s["max_violation_upper"] > 0 for s in safety)):
                    raise ValueError("P3 author safety event mismatch")
            tube_union = gpu_ranges(run_dir / "data/ranges.jsonl")
            detail = {"periods": 20, "substeps": 100, "controller_calls": 20,
                      "author_verdict": "window safe", "closed_window_x1_tube_union": tube_union}
        return {"run_status": outer["status"], "exit_code": exit_code,
                "process_wall_s": wall, "valid_full_property_sample": True, "detail": detail}
    except Exception as exc:
        return {"run_status": "invalid_or_failed", "exit_code": exit_code,
                "process_wall_s": None, "valid_full_property_sample": False,
                "failure": repr(exc)}


def run(root):
    root = root.resolve()
    if root.parent != BASE or root.exists():
        raise ValueError("campaign root must be a new direct child of the run base")
    initial_gpu = check_inputs()
    root.mkdir()
    shutil.copyfile(__file__, root / "runner.py")
    rounds = []
    for round_no in range(6):
        rotation = 0 if round_no == 0 else round_no % 4
        rounds.append({"round": round_no,
                       "phase": "first_process" if round_no == 0 else "later_process",
                       "order": METHODS[rotation:] + METHODS[:rotation]})
    write_json(root / "PLAN.json", {
        "schema": "archcomp26-sp-two-state-fourway-fresh-process-campaign-v1",
        "started_utc": now(), "contract": "named 2026 paper two-physical-state SP, auxiliary checker clock",
        "same_hardware": "physical GPU 2, CPU affinity 10-13, sequential fresh processes",
        "initial_gpu2_memory_mib_and_utilization_percent": initial_gpu,
        "wall_boundary": "supervisor Popen(child) to reap; native includes RPC startup and solver, GPU includes Python/NN/CUDA startup and driver",
        "first_definition": "first fresh process per method in this campaign; host/GPU not rebooted",
        "later_definition": "five later independent fresh processes per method",
        "separation_s": 2, "timeout_s_per_run": 120, "native_rpc_port": 5101,
        "fixed_model": str(MODEL), "fixed_config": str(CONFIG), "rounds": rounds,
        "content_digest_performed": False,
    })
    events, stop_reason = [], None
    try:
        with (root / "events.jsonl").open("x") as journal:
            for plan in rounds:
                for method in plan["order"]:
                    time.sleep(2)
                    try:
                        gpu2_idle()
                        if method == "native" and not port_free():
                            raise RuntimeError("SP RPC port 5101 became occupied")
                    except Exception as exc:
                        stop_reason = f"resource gate before round {plan['round']} {method}: {exc!r}"
                        break
                    name = f"{'first' if plan['round'] == 0 else 'later'}{plan['round']:02d}_{method}"
                    run_dir = root / name
                    started = now()
                    exit_code = subprocess.run(command(method, run_dir), cwd=BASE).returncode
                    event = {"phase": plan["phase"], "round": plan["round"],
                             "method": method, "directory": str(run_dir),
                             "started_utc": started, "ended_utc": now(),
                             **inspect_run(method, run_dir, exit_code)}
                    journal.write(json.dumps(event, allow_nan=False) + "\n")
                    journal.flush()
                    events.append(event)
                    print(json.dumps({"round": plan["round"], "method": method,
                                      "valid": event["valid_full_property_sample"],
                                      "wall_s": event["process_wall_s"]}), flush=True)
                    if not event["valid_full_property_sample"]:
                        stop_reason = f"first invalid run: {name}: {event.get('failure', 'see raw result')}"
                        break
                if stop_reason:
                    break
    finally:
        later = {method: [e["process_wall_s"] for e in events if e["method"] == method
                          and e["phase"] == "later_process" and e["valid_full_property_sample"]]
                 for method in METHODS}
        write_json(root / "SUMMARY.json", {
            "ended_utc": now(), "expected_attempts": 24, "attempts_started": len(events),
            "all_expected_attempts_recorded": len(events) == 24,
            "all_samples_valid": len(events) == 24 and all(e["valid_full_property_sample"] for e in events),
            "first_failure_or_resource_stop": stop_reason,
            "later_valid_counts": {method: len(values) for method, values in later.items()},
            "later_process_wall_medians_s": {method: statistics.median(values) if len(values) == 5 else None
                                             for method, values in later.items()},
            "later_process_wall_minmax_s": {method: [min(values), max(values)] if len(values) == 5 else None
                                            for method, values in later.items()},
            "scope": "descriptive fresh-process walls for the named two-state contract; no independent end-to-end NN certificate or hardware-isolated ranking",
            "content_digest_performed": False,
        })
    return 0 if len(events) == 24 and all(e["valid_full_property_sample"] for e in events) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({"ready": True, "gpu2_memory_mib_and_utilization_percent": check_inputs()}))
        return
    if args.root is None:
        parser.error("--root is required unless --check-only is used")
    raise SystemExit(run(args.root))


if __name__ == "__main__":
    main()
