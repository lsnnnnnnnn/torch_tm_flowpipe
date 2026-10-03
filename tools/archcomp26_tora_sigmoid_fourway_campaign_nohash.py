#!/usr/bin/env python3
"""Official TORA sigmoid u=11f: one first and five later four-way rounds.

Without --execute this is a read-only preflight; no old run ID is reused.
"""

import argparse
from datetime import datetime, timezone
import filecmp
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
OLD = {
    "huan": BASE / "tora_reach_sigmoid_official2026_mat_u11_full500_huan_002",
    "xiangru": BASE / "tora_reach_sigmoid_official2026_mat_u11_xiangru_full500_001",
    "ours_p3": BASE / "tora_reach_sigmoid_official2026_mat_u11_p3_full500_001",
}
BUILD = BASE / "native_tora_reach_sigmoid_u11_build_001/archcomp/TORA"
PAIR = N / "tools/run_archcomp26_tora_reach_sigmoid_native_pair.sh"
OVERLAY = BASE / "native_dp_less_rpc_overlay_preflight_001/rpc_overlay"
PY = N / "nncs_env/bin/python"
SUPERVISOR = BASE / "run_archcomp26_nohash.py"
METHODS = ("native", "huan", "xiangru", "ours_p3")
ROW = struct.Struct("<QQd16d")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def plan():
    return [{"round": k, "phase": "first" if k == 0 else "later",
             "order": METHODS[k % 4:] + METHODS[:k % 4]} for k in range(6)]


def gpu_idle():
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    rows = {int(p[0]): (int(p[1]), int(p[2]))
            for line in output.splitlines() if (p := [s.strip() for s in line.split(",")])}
    if 2 not in rows or rows[2][0] > 100 or rows[2][1] > 10:
        raise RuntimeError(f"GPU 2 not idle: {rows.get(2)}")
    return rows[2]


def port_free():
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", 5111)) != 0


def scan_ranges(path):
    raw = path.read_bytes()
    if len(raw) != 500 * ROW.size:
        raise ValueError(f"range count differs from 500: {path}")
    final, h_min, h_max, h_deviation = None, math.inf, -math.inf, 0.0
    for k, (lane, step, h, *v) in enumerate(ROW.iter_unpack(raw), 1):
        if (lane != 0 or step != k or not math.isfinite(h) or h <= 0
                or abs(h - .01) > 1e-9):
            raise ValueError(f"saved grid mismatch at {k}: {path}")
        h_min, h_max = min(h_min, h), max(h_max, h)
        h_deviation = max(h_deviation, abs(h - .01))
        for i in range(4):
            lo, hi, elo, ehi = v[4*i:4*i+4]
            if not all(map(math.isfinite, (lo, hi, elo, ehi))) or not lo <= elo <= ehi <= hi:
                raise ValueError(f"saved tube/endpoint invalid at {k}/{i}: {path}")
        final = ((v[2], v[3]), (v[6], v[7]))
    if not (-.1 <= final[0][0] <= final[0][1] <= .2
            and -.9 <= final[1][0] <= final[1][1] <= -.6):
        raise ValueError(f"saved T=5 endpoint is not inside target: {path}")
    return {"records": 500, "h_min": h_min, "h_max": h_max,
            "h_max_abs_deviation_from_0p01": h_deviation,
            "terminal_x1_x2": final, "target_contained": True}


def preflight(root):
    if root.parent != BASE or root.exists():
        raise ValueError("campaign root must be a new direct child of the run base")
    needed = [PY, SUPERVISOR, PAIR, BUILD / "tora_sigmoid_u11_full10",
              BUILD / "crown_paper.py", BUILD / "controller_plant_u.onnx"]
    for method, old in OLD.items():
        needed.extend(old / name for name in ("run_full.py", "author_support.py",
                      "config.yaml", "controller_plant_u.onnx", "controller_plant_u.onnx.json",
                      "nn_tora_sigmoid.mat", "START.json", "RESULT.json"))
        if method == "ours_p3":
            needed.extend(old / name for name in ("archcomp26_tora_remain_p3_nohash.py",
                          "archcomp26_tora_remain_author_nohash.py", "archcomp26_dp_p3_nohash.py"))
        result = json.loads((old / "RESULT.json").read_text())
        contract = json.loads((old / "START.json").read_text())["contract"]
        if (result.get("status") != "completed_full_numerical_horizon"
                or result.get("accepted_substeps") != 500
                or contract.get("control_periods") != 10
                or contract.get("internal_control") != "u = 11 * f(x) + 0"):
            raise ValueError(f"old {method} full receipt/contract differs")
    if any(not path.is_file() for path in needed) or not OVERLAY.is_dir():
        raise FileNotFoundError("a saved executable/model/receipt is missing")
    native_dir = BASE / "native_tora_reach_sigmoid_u11_full10_001"
    native_old = json.loads((native_dir / "RESULT.json").read_text())
    if native_old.get("status") != "completed" or native_old.get("exit_code") != 0:
        raise ValueError("old native full receipt differs")
    old_scans = {method: scan_ranges(directory / "ranges.bin")
                 for method, directory in {"native": native_dir, **OLD}.items()}
    source_model = OLD["huan"] / "controller_plant_u.onnx"
    if any(not filecmp.cmp(source_model, model, shallow=False) for model in
           [*(old / "controller_plant_u.onnx" for old in OLD.values()),
            BUILD / "controller_plant_u.onnx"]):
        raise ValueError("saved four-way ONNX files differ by direct byte comparison")
    if not port_free():
        raise RuntimeError("RPC port 5111 occupied")
    subprocess.run(["taskset", "-c", "10-13", "/bin/true"], check=True)
    return gpu_idle(), old_scans


def edit(path, changes):
    source = path.read_text()
    for old, new, count in changes:
        if source.count(old) != count:
            raise ValueError(f"unreviewed source at {path}: {old!r}")
        source = source.replace(old, new)
    path.write_text(source)


def stage(method, sample):
    old = OLD[method]
    payload = sample / "payload"
    payload.mkdir(parents=True)
    names = ["run_full.py", "author_support.py", "config.yaml",
             "controller_plant_u.onnx", "controller_plant_u.onnx.json", "nn_tora_sigmoid.mat"]
    if method == "ours_p3":
        names += ["archcomp26_tora_remain_p3_nohash.py",
                  "archcomp26_tora_remain_author_nohash.py", "archcomp26_dp_p3_nohash.py"]
    for name in names:
        shutil.copyfile(old / name, payload / name)
    edit(payload / "config.yaml", [(str(old), str(payload), 1)])
    edit(payload / "controller_plant_u.onnx.json", [(str(old), str(payload), 2)])
    edit(payload / "run_full.py", [
        (f'PROFILE = "{old.name}"', f'PROFILE = "{sample.name}"', 1),
        ('os.environ.get("CUDA_VISIBLE_DEVICES") != "3"',
         'os.environ.get("CUDA_VISIBLE_DEVICES") != "2"', 1),
        ('physical GPU 3', 'physical GPU 2', 1),
        ('"gpu_physical": 3', '"gpu_physical": 2', 1),
    ])
    if method == "ours_p3":
        for name, old_label, new_label in (
            ("archcomp26_tora_remain_p3_nohash.py", "GPU3", "GPU2"),
            ("archcomp26_dp_p3_nohash.py", "GPU 3", "GPU 2"),
        ):
            edit(payload / name, [
                ('os.environ.get("CUDA_VISIBLE_DEVICES") != "3"',
                 'os.environ.get("CUDA_VISIBLE_DEVICES") != "2"', 1),
                (old_label, new_label, 1),
            ])
    return payload


def command(method, sample, payload=None):
    output = sample if method == "native" else sample / "outer"
    args = [str(PY), "-B", str(SUPERVISOR), "--run-dir", str(output),
            "--instance", "tora-reach-sigmoid", "--method", method,
            "--contract-label", "tora-reach-official2026-mat-u11f-campaign",
            "--cwd", str(BUILD if method == "native" else payload), "--timeout-s", "120"]
    for env in ("CUDA_VISIBLE_DEVICES=2", "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1"):
        args += ["--env", env]
    if method == "native":
        for env in (f"TORA_REACH_RUN_DIR={output}", f"TORA_REACH_WORKDIR={BUILD}",
                    f"TORA_REACH_BINARY={BUILD / 'tora_sigmoid_u11_full10'}",
                    f"TORA_REACH_SERVER_PYTHON={PY}", f"TORA_REACH_OVERLAY={OVERLAY}",
                    f"TORA_REACH_MODEL={BUILD / 'controller_plant_u.onnx'}",
                    "TORA_REACH_CPUSET=10-13"):
            args += ["--env", env]
        child = ["taskset", "-c", "10-13", "/bin/bash", str(PAIR)]
    else:
        child = ["taskset", "-c", "10-13", str(PY), "-B", str(payload / "run_full.py")]
    return args + ["--"] + child


def inspect(method, sample, returncode):
    try:
        outer = sample if method == "native" else sample / "outer"
        result = json.loads((outer / "RESULT.json").read_text())
        wall = result["wall_s"]
        if returncode or result.get("status") != "completed" or not math.isfinite(wall) or wall <= 0:
            raise ValueError("outer process failed")
        if method == "native":
            lines = (sample / "native.log").read_text().splitlines()
            if ([line for line in lines if line.startswith("Step ")]
                    != [f"Step {k}" for k in range(10)] or "VERIFIED" not in lines
                    or len((sample / "controller_rpc.jsonl").read_text().splitlines()) != 10):
                raise ValueError("native period/RPC/verdict mismatch")
            data = sample
        else:
            data = sample / "payload"
            inner = json.loads((data / "RESULT.json").read_text())
            start = json.loads((data / "START.json").read_text())
            metrics = json.loads((data / "metrics.json").read_text())
            if (inner.get("status") != "completed_full_numerical_horizon"
                    or inner.get("accepted_substeps") != 500
                    or start.get("gpu_physical") != 2
                    or start.get("cpu_affinity") != [10, 11, 12, 13]
                    or metrics.get("steps") != 10 or metrics.get("broken") != 0):
                raise ValueError("GPU completion/affinity mismatch")
        ranges = scan_ranges(data / "ranges.bin")
        return {"valid": True, "wall_s": wall,
                "terminal_x1_x2": ranges["terminal_x1_x2"],
                "h_max_abs_deviation_from_0p01": ranges["h_max_abs_deviation_from_0p01"]}
    except Exception as error:
        return {"valid": False, "wall_s": None, "failure": repr(error)}


def run(root):
    initial_gpu, old_scans = preflight(root)
    root.mkdir()
    shutil.copyfile(__file__, root / "runner.py")
    save(root / "PLAN.json", {"started_utc": now(), "rounds": plan(),
         "contract": "2026 official four-sigmoid, u=11f; 1 box, 10 controls, 500 steps",
         "resource": "GPU 2; CPU 10-13; sequential fresh processes; RPC 5111",
         "initial_gpu2_memory_mib_util_percent": initial_gpu,
         "read_only_old_full_range_scans": old_scans,
         "timing": "outer supervisor Popen to reap; first process separate from five later",
         "timeout_s_per_run": 120, "content_digest_performed": False})
    events, stop = [], None
    try:
        with (root / "events.jsonl").open("x") as log:
            for round_plan in plan():
                for method in round_plan["order"]:
                    time.sleep(2)
                    try:
                        gpu_idle()
                        if method == "native" and not port_free():
                            raise RuntimeError("RPC 5111 became occupied")
                    except Exception as error:
                        stop = f"resource gate: {error!r}"
                        break
                    sample = root / f"{round_plan['phase']}{round_plan['round']:02d}_{method}"
                    started = now()
                    try:
                        payload = None if method == "native" else stage(method, sample)
                        code = subprocess.run(command(method, sample, payload), cwd=BASE).returncode
                        outcome = inspect(method, sample, code)
                    except Exception as error:
                        code, outcome = None, {"valid": False, "wall_s": None,
                                               "failure": repr(error)}
                    event = {"round": round_plan["round"], "phase": round_plan["phase"],
                             "method": method, "path": str(sample), "started_utc": started,
                             "ended_utc": now(), "exit_code": code, **outcome}
                    log.write(json.dumps(event, allow_nan=False) + "\n")
                    log.flush()
                    events.append(event)
                    print(json.dumps({"round": round_plan["round"], "method": method,
                                      "valid": outcome["valid"], "wall_s": outcome["wall_s"]}), flush=True)
                    if not outcome["valid"]:
                        stop = f"first invalid run: {sample.name}: {outcome['failure']}"
                        break
                if stop:
                    break
    finally:
        later = {m: [e["wall_s"] for e in events if e["method"] == m
                     and e["phase"] == "later" and e["valid"]] for m in METHODS}
        save(root / "SUMMARY.json", {"ended_utc": now(), "expected": 24,
             "started": len(events), "first_stop": stop,
             "all_valid": len(events) == 24 and all(e["valid"] for e in events),
             "later_valid_counts": {m: len(v) for m, v in later.items()},
             "later_median_wall_s": {m: statistics.median(v) if len(v) == 5 else None
                                     for m, v in later.items()},
             "scope": "descriptive full-numeric process time; no independent NNCS proof or stable speed ranking",
             "content_digest_performed": False})
    return 0 if len(events) == 24 and stop is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.execute:
        raise SystemExit(run(root))
    gpu, old_scans = preflight(root)
    print(json.dumps({"mode": "read_only_preflight", "root": str(root),
                      "gpu2_memory_mib_util_percent": gpu, "rounds": plan(),
                      "old_full_range_scans": old_scans,
                      "samples": 24, "content_digest_performed": False}, indent=2))


if __name__ == "__main__":
    main()
