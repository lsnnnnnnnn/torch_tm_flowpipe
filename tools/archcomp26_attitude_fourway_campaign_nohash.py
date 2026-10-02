#!/usr/bin/env python3
"""Four-method fresh-process timing for corrected Attitude avoidance."""

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
AUTHOR = BASE / "author_attitude_avoid_v1"
P3 = BASE / "p3_attitude_avoid_v1"
BUILD = BASE / "native_attitude_avoid_build_001/archcomp/attitude_control"
PYTHON = N / "nncs_env/bin/python"
SUPERVISOR = BASE / "run_archcomp26_nohash.py"
PAIR = BASE / "run_archcomp26_attitude_avoid_native_pair.sh"
MODEL = BASE / "attitude_prep_001/official_attitude_control_3_64_torch.onnx"
OVERLAY = BASE / "native_dp_less_rpc_overlay_preflight_001/rpc_overlay"
METHODS = ("native", "huan", "xiangru", "ours_p3")
LABELS = {"native": "flowstar_native", "huan": "huan", "xiangru": "xiangru",
          "ours_p3": "pytorch_gpu"}
COMMON_ENV = ("CUDA_VISIBLE_DEVICES=2", "OMP_NUM_THREADS=1",
              "OPENBLAS_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1")
UNSAFE = ((-.2, 0), (-.5, -.4), (0, .2), (-.7, -.6), (.7, .8), (-.4, -.2))
INITIAL = ((-.45, -.44), (-.55, -.54), (.65, .66),
           (-.75, -.74), (.85, .86), (-.65, -.64))
RANGE_RECORD = struct.Struct("<QQd24d")
OLD_GPU_GUARD = 'if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":'
NEW_GPU_GUARD = 'if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


def gpu_pool_idle():
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True
    ).stdout
    rows = {int(parts[0]): (int(parts[1]), int(parts[2]))
            for line in output.splitlines() if (parts := [x.strip() for x in line.split(",")])}
    busy = {i: rows.get(i) for i in (1, 2, 3)
            if i not in rows or rows[i][0] > 100 or rows[i][1] > 10}
    if busy:
        raise RuntimeError(f"experimental GPUs 1-3 are not idle: {busy}")
    return {i: rows[i] for i in (1, 2, 3)}


def port_free():
    with socket.socket() as connection:
        return connection.connect_ex(("127.0.0.1", 5103)) != 0


def patched_p3_source():
    source = (P3 / "archcomp26_attitude_p3_nohash.py").read_text()
    if source.count(OLD_GPU_GUARD) != 1 or source.count("reserved for physical GPU3") != 1:
        raise ValueError("P3 source GPU guard differs from the reviewed version")
    return source.replace(OLD_GPU_GUARD, NEW_GPU_GUARD).replace(
        "reserved for physical GPU3", "reserved for physical GPU2")


def check_inputs():
    required = (PYTHON, SUPERVISOR, PAIR, MODEL, BUILD / "attitude_avoid_full30",
                BUILD / "attitude_avoid_full30.cpp", BUILD / "crown_paper.py",
                AUTHOR / "archcomp26_attitude_avoid_author_nohash.py",
                P3 / "archcomp26_attitude_p3_nohash.py",
                P3 / "archcomp26_attitude_avoid_author_nohash.py",
                P3 / "archcomp26_tora_remain_author_nohash.py",
                P3 / "archcomp26_dp_p3_nohash.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not OVERLAY.is_dir():
        raise FileNotFoundError({"missing_files": missing, "overlay": str(OVERLAY)})
    source = (BUILD / "attitude_avoid_full30.cpp").read_text()
    if '"-x4 - 0.7", "x4 + 0.6"' not in source or '"-x4 - 0.4"' in source:
        raise ValueError("native binary source does not express the corrected unsafe box")
    patched_p3_source()
    subprocess.run(["/usr/bin/taskset", "-c", "10-13", "/bin/true"], check=True)
    if not port_free():
        raise RuntimeError("Attitude RPC port 5103 is occupied")
    return gpu_pool_idle()


def stage_p3(root):
    stage = root / "p3_source"
    stage.mkdir()
    for name in ("archcomp26_attitude_avoid_author_nohash.py",
                 "archcomp26_tora_remain_author_nohash.py", "archcomp26_dp_p3_nohash.py"):
        shutil.copyfile(P3 / name, stage / name)
    launcher = stage / "archcomp26_attitude_p3_gpu2_campaign_nohash.py"
    with launcher.open("x") as stream:
        stream.write(patched_p3_source())
    return launcher


def command(method, run_dir, p3_launcher):
    args = [str(PYTHON), "-B", str(SUPERVISOR), "--run-dir", str(run_dir),
            "--instance", "attitude-control-avoid", "--method", LABELS[method],
            "--contract-label", "attitude-avoid-official-unsafe-corrected-full30-campaign",
            "--cwd", str(BUILD if method == "native" else AUTHOR if method != "ours_p3"
                         else p3_launcher.parent), "--timeout-s", "120"]
    for env in COMMON_ENV:
        args.extend(("--env", env))
    if method == "native":
        for env in (f"ATTITUDE_RUN_DIR={run_dir}", f"ATTITUDE_WORKDIR={BUILD}",
                    f"ATTITUDE_BINARY={BUILD / 'attitude_avoid_full30'}",
                    f"ATTITUDE_SERVER_PYTHON={PYTHON}", f"ATTITUDE_OVERLAY={OVERLAY}",
                    f"ATTITUDE_MODEL={MODEL}", "ATTITUDE_CPUSET=10-13"):
            args.extend(("--env", env))
        child = ["/usr/bin/taskset", "-c", "10-13", "/bin/bash", str(PAIR)]
    else:
        launcher = p3_launcher if method == "ours_p3" else AUTHOR / "archcomp26_attitude_avoid_author_nohash.py"
        child = ["/usr/bin/taskset", "-c", "10-13", str(PYTHON), "-B", str(launcher)]
        if method != "ours_p3":
            child.extend(("--backend", method))
        child.extend(("--mode", "full", "--output", str(run_dir / "payload")))
    return args + ["--"] + child


def scan_ranges(path):
    raw = path.read_bytes()
    if len(raw) != 60 * RANGE_RECORD.size:
        raise ValueError("expected 60 six-state range records")
    tube_union = [[math.inf, -math.inf] for _ in range(6)]
    min_gap = math.inf
    last_endpoint = None
    for expected, record in enumerate(RANGE_RECORD.iter_unpack(raw), 1):
        lane, step, h, *values = record
        if lane != 0 or step != expected or not math.isfinite(h) or abs(h - .05) > 1e-12:
            raise ValueError(f"range grid mismatch at {expected}")
        gaps, endpoint = [], []
        for i, (unsafe_lo, unsafe_hi) in enumerate(UNSAFE):
            tube_lo, tube_hi, end_lo, end_hi = values[4*i:4*i+4]
            if (not all(map(math.isfinite, (tube_lo, tube_hi, end_lo, end_hi)))
                    or not tube_lo <= end_lo <= end_hi <= tube_hi):
                raise ValueError(f"invalid tube/endpoint at {expected}/{i}")
            if expected == 1 and (tube_lo > INITIAL[i][0] or tube_hi < INITIAL[i][1]):
                raise ValueError(f"first tube misses physical initial box at state {i}")
            tube_union[i][0] = min(tube_union[i][0], tube_lo)
            tube_union[i][1] = max(tube_union[i][1], tube_hi)
            gaps.append(max(unsafe_lo - tube_hi, tube_lo - unsafe_hi))
            endpoint.append([end_lo, end_hi])
        best_gap = max(gaps)
        if best_gap <= 0:
            raise ValueError(f"saved tube intersects the closed official unsafe box at {expected}")
        min_gap = min(min_gap, best_gap)
        last_endpoint = endpoint
    return {"range_records": 60, "all_saved_tubes_disjoint": True,
            "minimum_box_separation_gap": min_gap,
            "full_tube_union": tube_union,
            "terminal_endpoint": last_endpoint,
            "terminal_endpoint_width": [hi-lo for lo, hi in last_endpoint]}


def inspect_run(method, run_dir, exit_code):
    try:
        outer = json.loads((run_dir / "RESULT.json").read_text())
        wall = outer["wall_s"]
        if (exit_code != 0 or outer.get("status") != "completed"
                or not math.isfinite(wall) or wall <= 0):
            raise ValueError("supervisor did not complete successfully")
        if method == "native":
            lines = (run_dir / "native.log").read_text().splitlines()
            steps = [line for line in lines if line.startswith("Step ")]
            if (steps != [f"Step {k}" for k in range(30)]
                    or "COMPLETED_PERIODS 30/30" not in lines
                    or "FLOWPIPE_SEGMENTS 60" not in lines or "VERIFIED" not in lines):
                raise ValueError("native author step/verdict mismatch")
            rpc = (run_dir / "controller_rpc.jsonl").read_text().splitlines()
            responses = sum('"POST / HTTP/1.1" 200' in line
                            for line in (run_dir / "server.log").read_text().splitlines())
            if len(rpc) != 30 or responses != 30:
                raise ValueError("native RPC/HTTP count mismatch")
            for line in rpc:
                row = json.loads(line)
                if len(row["input_lower"]) != 6 or len(row["input_upper"]) != 6:
                    raise ValueError("native controller input width mismatch")
            detail = {**scan_ranges(run_dir / "ranges.bin"),
                      "periods": 30, "controller_rpc": len(rpc), "http_200": responses,
                      "author_verdict": "VERIFIED"}
        else:
            payload = run_dir / "payload"
            inner = json.loads((payload / "RESULT.json").read_text())
            metrics = json.loads((payload / "metrics.json").read_text())
            start = json.loads((payload / "START.json").read_text())
            observations = [json.loads(line) for line in (payload / "observations.jsonl").read_text().splitlines()]
            lines = (run_dir / "stdout.log").read_text().splitlines()
            if (inner.get("status") != "completed" or inner.get("driver_return") != 0
                    or inner.get("expected_substeps") != 60
                    or inner.get("saved_tubes_box_disjoint_official_unsafe") is not True
                    or inner.get("author_checker_lines") != []
                    or inner.get("author_checker_interpretation") !=
                       "VERIFIED_BY_AUTHOR_CHECKER_SILENCE_AND_SAVED_BOX_DISJOINTNESS"
                    or len(observations) != 60
                    or metrics.get("steps") != 30 or metrics.get("substeps") != 2
                    or metrics.get("broken") != 0
                    or [step.get("k") for step in metrics.get("ctrl_steps", [])] != list(range(30))
                    or [line for line in lines if line.startswith("Step ")] !=
                       [f"Step {k}" for k in range(30)]):
                raise ValueError("GPU author completion/checker/controller mismatch")
            if method == "ours_p3":
                if (inner.get("completed_substeps") != 60 or inner.get("accepted_substeps") != 60
                        or inner.get("all_substeps_accepted") is not True
                        or start.get("physical_gpu") != "2"
                        or any(row.get("accepted") is not True for row in observations)):
                    raise ValueError("P3 accepted/GPU2 mismatch")
            else:
                if (inner.get("observed_substeps") != 60 or inner.get("accepted_lane_substeps") != 60
                        or inner.get("all_lanes_accepted") is not True
                        or start.get("cuda_visible_devices") != "2"
                        or start.get("backend") != method
                        or any(row.get("accepted_count") != 1 for row in observations)):
                    raise ValueError("author accepted/GPU2 mismatch")
            if (start.get("cpu_affinity") != [10, 11, 12, 13]
                    or any(row.get("substep") != k or row.get("tube_box_disjoint_official_unsafe") is not True
                           for k, row in enumerate(observations, 1))):
                raise ValueError("GPU CPU affinity/observation mismatch")
            config = (payload / "config.yaml").read_text()
            if "- -x4 - 0.7" not in config or "- -x4 - 0.4" in config:
                raise ValueError("GPU generated config did not fix official unsafe x4")
            detail = {**scan_ranges(payload / "ranges.bin"),
                      "periods": 30, "controller_calls": 30,
                      "author_verdict": "silence plus saved box disjointness"}
        return {"run_status": "completed", "exit_code": exit_code,
                "process_wall_s": wall, "valid_full_property_sample": True, "detail": detail}
    except Exception as exc:
        return {"run_status": "invalid_or_failed", "exit_code": exit_code,
                "process_wall_s": None, "valid_full_property_sample": False,
                "failure": repr(exc)}


def run(root):
    root = root.resolve()
    if root.parent != BASE or root.exists():
        raise ValueError("campaign root must be a new direct child of the run base")
    initial_gpus = check_inputs()
    root.mkdir()
    shutil.copyfile(__file__, root / "runner.py")
    p3_launcher = stage_p3(root)
    rounds = []
    for round_no in range(6):
        rotation = 0 if round_no == 0 else round_no % 4
        rounds.append({"round": round_no,
                       "phase": "first_process" if round_no == 0 else "later_process",
                       "order": METHODS[rotation:] + METHODS[:rotation]})
    write_json(root / "PLAN.json", {
        "schema": "archcomp26-attitude-corrected-unsafe-fourway-campaign-v1",
        "started_utc": now(), "contract": "corrected official six-dimensional closed unsafe box, avoid for all t in [0,3]",
        "unsafe_box": UNSAFE, "initial_physical_box": INITIAL,
        "same_hardware": "physical GPU 2, CPU affinity 10-13, sequential fresh processes",
        "initial_gpu_1_2_3_memory_mib_and_utilization_percent": initial_gpus,
        "wall_boundary": "supervisor Popen(child) to reap; native includes RPC startup and solver, GPU includes Python/NN/CUDA startup and driver",
        "first_definition": "first fresh process per method in this campaign, not a rebooted host/GPU",
        "later_definition": "five later independent fresh processes per method",
        "separation_s": 2, "timeout_s_per_run": 120, "native_rpc_port": 5103,
        "fixed_model": str(MODEL), "fixed_native_binary": str(BUILD / "attitude_avoid_full30"),
        "p3_gpu2_adaptation": {"original": str(P3 / "archcomp26_attitude_p3_nohash.py"),
                               "saved_source": str(p3_launcher),
                               "only_change": "CUDA_VISIBLE_DEVICES guard 3 -> 2 and its error message; other source copied"},
        "rounds": rounds, "content_digest_performed": False,
    })
    events, stop_reason = [], None
    try:
        with (root / "events.jsonl").open("x") as journal:
            for plan in rounds:
                for method in plan["order"]:
                    time.sleep(2)
                    try:
                        gpu_pool_idle()
                        if method == "native" and not port_free():
                            raise RuntimeError("Attitude RPC port 5103 became occupied")
                    except Exception as exc:
                        stop_reason = f"resource gate before round {plan['round']} {method}: {exc!r}"
                        break
                    name = f"{'first' if plan['round'] == 0 else 'later'}{plan['round']:02d}_{method}"
                    run_dir = root / name
                    started = now()
                    try:
                        exit_code = subprocess.run(command(method, run_dir, p3_launcher), cwd=BASE).returncode
                        receipt = inspect_run(method, run_dir, exit_code)
                    except Exception as exc:
                        receipt = {"run_status": "launch_failed", "exit_code": None,
                                   "process_wall_s": None, "valid_full_property_sample": False,
                                   "failure": repr(exc)}
                    event = {"phase": plan["phase"], "round": plan["round"],
                             "method": method, "directory": str(run_dir),
                             "started_utc": started, "ended_utc": now(),
                             **receipt}
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
            "later_valid_counts": {method: len(v) for method, v in later.items()},
            "later_process_wall_medians_s": {method: statistics.median(v) if len(v) == 5 else None
                                             for method, v in later.items()},
            "later_process_wall_minmax_s": {method: [min(v), max(v)] if len(v) == 5 else None
                                            for method, v in later.items()},
            "scope": "descriptive fresh-process walls for corrected official unsafe box; no independent end-to-end NN certificate or hardware-isolated ranking",
            "content_digest_performed": False,
        })
    return 0 if len(events) == 24 and all(e["valid_full_property_sample"] for e in events) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({"ready": True, "gpu_1_2_3_memory_mib_and_utilization_percent": check_inputs()}))
        return
    if args.root is None:
        parser.error("--root is required unless --check-only is used")
    raise SystemExit(run(args.root))


if __name__ == "__main__":
    main()
