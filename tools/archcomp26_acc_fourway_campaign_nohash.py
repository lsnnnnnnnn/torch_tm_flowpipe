#!/usr/bin/env python3
"""Run a new ACC participant-order four-method fresh-process timing campaign."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
BASE = N / "runs/archcomp26_20261001"
PREP = BASE / "acc_prep_001"
BUILD = BASE / "acc_native_var_tail_build_001"
PYTHON = N / "nncs_env/bin/python"
SUPERVISOR = BASE / "run_archcomp26_nohash.py"
CONFIG = PREP / "acc_participant_vrel_lead_minus_ego.yaml"
GPU_LAUNCHER = PREP / "archcomp26_acc_participant_gpu_nohash.py"
P3_LAUNCHER = PREP / "archcomp26_acc_p3_nohash.py"
PAIR = PREP / "run_native_pair.sh"
MODEL = PREP / "official_acc_controller_5_20.onnx"
OVERLAY = BASE / "native_dp_less_rpc_overlay_preflight_001/rpc_overlay"
METHODS = ("native", "huan", "xiangru", "ours_p3")
COMMON_ENV = ("CUDA_VISIBLE_DEVICES=2", "OMP_NUM_THREADS=1",
              "OPENBLAS_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1")


def now():
    return datetime.now(timezone.utc).isoformat()


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
        return connection.connect_ex(("127.0.0.1", 5102)) != 0


def check_inputs():
    required = (PYTHON, SUPERVISOR, CONFIG, GPU_LAUNCHER, P3_LAUNCHER, PAIR,
                MODEL, BUILD / "acc_native_var_tail", BUILD / "observed_server.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not OVERLAY.is_dir():
        raise FileNotFoundError({"missing_files": missing, "overlay": str(OVERLAY)})
    if not port_free():
        raise RuntimeError("ACC RPC port 5102 is occupied")
    gpu2_idle()


def command(method, run_dir):
    args = [str(PYTHON), "-B", str(SUPERVISOR), "--run-dir", str(run_dir),
            "--instance", "acc-safe-distance", "--method", method,
            "--contract-label", "acc-participant-vrel-lead-minus-ego-full50-campaign",
            "--cwd", str(BUILD if method == "native" else PREP),
            "--timeout-s", "120"]
    for item in COMMON_ENV:
        args.extend(("--env", item))
    if method == "native":
        for item in (
            f"ACC_RUN_DIR={run_dir}", f"ACC_BINARY={BUILD / 'acc_native_var_tail'}",
            f"ACC_SERVER={BUILD / 'observed_server.py'}",
            f"ACC_SERVER_PYTHON={PYTHON}", f"ACC_OVERLAY={OVERLAY}",
            "ACC_CPUSET=10-13", "ACC_PORT=5102",
        ):
            args.extend(("--env", item))
        child = ["/usr/bin/taskset", "-c", "10-13", "/bin/bash", str(PAIR)]
    else:
        if method == "ours_p3":
            child = ["/usr/bin/taskset", "-c", "10-13", str(PYTHON), "-B", str(P3_LAUNCHER),
                     "--mode", "full", "--source-config", str(CONFIG),
                     "--output", str(run_dir / "data")]
        else:
            child = ["/usr/bin/taskset", "-c", "10-13", str(PYTHON), "-B", str(GPU_LAUNCHER),
                     "--backend", method, "--mode", "full", "--source-config", str(CONFIG),
                     "--output", str(run_dir / "data")]
    return args + ["--"] + child


def inspect_run(method, run_dir, exit_code):
    outer_file = run_dir / "RESULT.json"
    outer = json.loads(outer_file.read_text()) if outer_file.is_file() else {}
    valid = exit_code == 0 and outer.get("status") == "completed"
    if method == "native":
        log_file = run_dir / "native.log"
        lines = log_file.read_text().splitlines() if log_file.is_file() else []
        steps = [line.strip() for line in lines if line.startswith("Step ")]
        verdicts = [line.strip() for line in lines if line.strip() in ("VERIFIED", "UNKNOWN", "FALSIFIED")]
        ranges = run_dir / "ranges.bin"
        rpc = run_dir / "controller_rpc.jsonl"
        detail = {"steps": len(steps), "verdict": verdicts[-1] if verdicts else None,
                  "range_records": ranges.stat().st_size // 216 if ranges.is_file() and ranges.stat().st_size % 216 == 0 else None,
                  "rpc_records": len(rpc.read_text().splitlines()) if rpc.is_file() else None}
        valid = valid and steps == [f"Step {k}" for k in range(50)] and detail["verdict"] == "VERIFIED"
        valid = valid and detail["range_records"] == detail["rpc_records"] == 50
    else:
        inner_file = run_dir / "data/RESULT.json"
        detail = json.loads(inner_file.read_text()) if inner_file.is_file() else {}
        valid = valid and detail.get("status") == "completed"
        valid = valid and detail.get("completed_substeps") == detail.get("safety_events") == 50
        valid = valid and detail.get("author_safe_bounds_all_nonpositive") is True
        valid = valid and detail.get("independent_tube_boxes_all_safe") is True
        valid = valid and detail.get("adapter_feature_calls") == detail.get("adapter_injection_calls") == 50
    return {"run_status": outer.get("status"), "exit_code": exit_code,
            "process_wall_s": outer.get("wall_s"), "valid_full_property_sample": bool(valid),
            "detail": detail}


def write_json(path, value):
    with path.open("x") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")


def run(root):
    root = root.resolve()
    if root.parent != BASE:
        raise ValueError(f"campaign root must be a new direct child of {BASE}")
    if root.exists():
        raise FileExistsError(root)
    check_inputs()
    root.mkdir()
    rounds = []
    for round_no in range(6):
        phase = "cold_process" if round_no == 0 else "steady_process"
        rotation = 0 if round_no == 0 else round_no % 4
        order = METHODS[rotation:] + METHODS[:rotation]
        rounds.append({"round": round_no, "phase": phase, "order": order})
    write_json(root / "PLAN.json", {
        "schema": "archcomp26-acc-fourway-fresh-process-campaign-v1",
        "started_utc": now(), "contract": "named ACC participant v_rel=v_lead-v_ego",
        "same_hardware": "physical GPU 2, CPU affinity 10-13, sequential processes",
        "wall_boundary": "supervisor Popen(child) to child reap; native child includes RPC startup and solver; GPU child includes Python/NN/CUDA startup and driver",
        "cold_definition": "first fresh process per method in this campaign; host and GPU were used earlier and were not rebooted",
        "steady_definition": "five later independent fresh processes per method, not in-process repetitions",
        "separation_s": 2, "timeout_s_per_run": 120,
        "fixed_model": str(MODEL), "fixed_config": str(CONFIG),
        "rounds": rounds, "content_digest_performed": False,
    })
    events = []
    try:
        with (root / "events.jsonl").open("x") as journal:
            for plan in rounds:
                for method in plan["order"]:
                    time.sleep(2)
                    gpu2_idle()
                    if method == "native" and not port_free():
                        raise RuntimeError("ACC RPC port 5102 became occupied before native run")
                    name = f"{'cold' if plan['round'] == 0 else 'steady'}{plan['round']:02d}_{method}"
                    run_dir = root / name
                    started = now()
                    command_line = command(method, run_dir)
                    exit_code = subprocess.run(command_line, cwd=BASE).returncode
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
    finally:
        steady = {method: [e["process_wall_s"] for e in events if e["method"] == method
                           and e["phase"] == "steady_process" and e["valid_full_property_sample"]]
                  for method in METHODS}
        write_json(root / "SUMMARY.json", {
            "ended_utc": now(), "expected_attempts": 24, "attempts_started": len(events),
            "all_expected_attempts_recorded": len(events) == 24,
            "all_samples_valid": len(events) == 24 and all(e["valid_full_property_sample"] for e in events),
            "steady_valid_counts": {method: len(values) for method, values in steady.items()},
            "steady_process_wall_medians_s": {method: statistics.median(values) if len(values) == 5 else None
                                              for method, values in steady.items()},
            "scope": "descriptive new-process walls under one named participant contract; no independent floating-point NN proof or stable hardware-isolated speed ranking",
            "content_digest_performed": False,
        })
    return 0 if len(events) == 24 and all(e["valid_full_property_sample"] for e in events) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    raise SystemExit(run(parser.parse_args().root))


if __name__ == "__main__":
    main()
