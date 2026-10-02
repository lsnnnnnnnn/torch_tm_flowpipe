#!/usr/bin/env python3
"""One isolated Airplane first-step sample from a 64-box binary cover."""

from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
BASE = N / "runs/archcomp26_20261001/native_airplane_first_reject_trace_20261002_005"
NEW = N / "runs/archcomp26_20261001/native_airplane_binary6_firststep_20261002_007"
WORK = NEW / "build/archcomp/airplane"
RUN = NEW / "run"
SPLIT = ("u", "v", "w", "phi", "theta", "psi")
INDEX = (3, 4, 5, 6, 7, 8)


def utc():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    if NEW.exists():
        raise RuntimeError(f"run identity already exists: {NEW}")
    source = BASE / "build/archcomp/airplane/airplane_first_reject_trace.cpp"
    library = BASE / "build/libflowstar.a"
    for path in (source, library, BASE / "build/flowstar",
                 BASE / "build/archcomp/airplane/crown_paper.py"):
        if not path.exists():
            raise FileNotFoundError(path)

    NEW.mkdir()
    WORK.mkdir(parents=True)
    RUN.mkdir()
    shutil.copy2(__file__, NEW / "driver.py")
    (NEW / "build/flowstar").symlink_to((BASE / "build/flowstar").resolve(),
                                         target_is_directory=True)
    text = source.read_text()
    for name in SPLIT:
        old, new = f"init_{name}(0, 1)", f"init_{name}(0, 0.5)"
        if text.count(old) != 1:
            raise RuntimeError(f"expected one full-box initializer: {old}")
        text = text.replace(old, new)
    cpp = WORK / "airplane_binary6_firststep.cpp"
    binary = WORK / "airplane_binary6_firststep"
    cpp.write_text(text)
    for name in ("matched_reach.h", "arch_ranges.h", "crown_paper.py"):
        shutil.copy2(source.parent / name, WORK / name)

    # Every bit tuple is a closed subbox. Their union is exactly [0,1]^6.
    grid = []
    for bits in product((0, 1), repeat=6):
        box = [[0, 0] for _ in range(19)]
        for index, bit in zip(INDEX, bits):
            box[index] = [bit / 2, (bit + 1) / 2]
        grid.append({"bits": "".join(map(str, bits)), "box": box})
    assert len(grid) == 64 and grid[0]["bits"] == "000000"
    assert all(item["box"][i] == [0, 0] for item in grid for i in
               set(range(19)) - set(INDEX))
    (NEW / "SPLIT_PLAN.json").write_text(json.dumps({
        "coordinates": SPLIT, "indices": INDEX,
        "rule": "each of six [0,1] axes -> closed [0,0.5] and [0.5,1]",
        "boxes": grid, "tested_bits": "000000",
        "scope": "candidate partition; only one first step is run"
    }, indent=2) + "\n")

    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    command = ["taskset", "-c", "18", "/usr/bin/g++-15", "-O3", "-std=c++11",
               "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
               "-I", "/usr/include/jsoncpp", str(cpp), str(library),
               "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm", "-lglpk",
               "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp", "-lcurl",
               "-ljsonrpccpp-common", "-ljsonrpccpp-client",
               "-l:libboost_thread.so.1.90.0", "-o", str(binary)]
    with (NEW / "build.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                timeout=300, check=False)
    (NEW / "BUILD.json").write_text(json.dumps({
        "created_utc": utc(), "source": str(source), "library": str(library),
        "changes": "six initial intervals [0,1] -> [0,0.5]; same ODE, controller, numerical profile and one-step trace",
        "command": command, "returncode": result.returncode,
        "binary_bytes": binary.stat().st_size if binary.exists() else None
    }, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError("isolated link failed; see build.log")
    return binary


def run(binary):
    env = os.environ.copy()
    env.update({"AIRPLANE_RUN_DIR": str(RUN), "AIRPLANE_WORKDIR": str(WORK),
                "AIRPLANE_BINARY": str(binary),
                "AIRPLANE_SERVER_PYTHON": str(N / "nncs_env/bin/python"),
                "AIRPLANE_OVERLAY": str(N / "runs/archcomp26_20261001/native_dp_less_rpc_overlay_preflight_001/rpc_overlay"),
                "AIRPLANE_MODEL": str(N / "runs/archcomp26_20261001/airplane_prep_001/controller_airplane.onnx"),
                "AIRPLANE_CPUSET": "10-13", "AIRPLANE_NATIVE_AS": "34359738368",
                "CUDA_VISIBLE_DEVICES": "2", "OMP_NUM_THREADS": "1"})
    launcher = N / "tools/run_archcomp26_airplane_native_fullbox_pair.sh"
    (RUN / "START.json").write_text(json.dumps({
        "started_utc": utc(), "command": ["/bin/bash", str(launcher)],
        "selected_environment": {key: env[key] for key in (
            "AIRPLANE_RUN_DIR", "AIRPLANE_WORKDIR", "AIRPLANE_BINARY",
            "AIRPLANE_SERVER_PYTHON", "AIRPLANE_OVERLAY", "AIRPLANE_MODEL",
            "AIRPLANE_CPUSET", "AIRPLANE_NATIVE_AS", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS")}, "timeout_s": 180,
        "scope": "binary six-dimensional partition, box 000000 only, one h=0.01 step"
    }, indent=2) + "\n")
    start = time.monotonic()
    timed_out = False
    with (RUN / "launcher.stdout.log").open("x") as out, (RUN / "launcher.stderr.log").open("x") as err:
        process = subprocess.Popen(["/bin/bash", str(launcher)], cwd=WORK, env=env,
                                   stdout=out, stderr=err, start_new_session=True)
        try:
            rc = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            rc = process.wait(timeout=10)
    result = {"ended_utc": utc(), "status": "timeout" if timed_out else
              "completed" if rc == 0 else "failed", "exit_code": rc,
              "timed_out": timed_out, "wall_s": time.monotonic() - start}
    (RUN / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit(result):
    ledger = json.loads((RUN / "initial_boxes.json").read_text())
    plan = json.loads((NEW / "SPLIT_PLAN.json").read_text())
    if ledger != [plan["boxes"][0]["box"]]:
        raise RuntimeError("actual box differs from partition cell 000000")
    native = (RUN / "native.log").read_text()
    rows = [line for line in native.splitlines() if line.startswith("AIRPLANE_PICARD\t")]
    match = re.findall(r"PERIOD 0 FLOWPIPES (\d+) STATUS (\d+)", native)
    rpc = (RUN / "controller_rpc.jsonl").read_text().splitlines()
    record = {"result": result, "actual_initial_box": ledger,
              "picard_row_count": len(rows), "nonincluded_indices": [
                  int(line.split("\t")[1]) for line in rows if "\tsubset=0" in line],
              "period_status": match, "rpc_count": len(rpc),
              "range_bytes": (RUN / "ranges.bin").stat().st_size,
              "scope": "one candidate subbox, one first numerical step; no full-box or T=2 result"}
    (NEW / "AUDIT.json").write_text(json.dumps(record, indent=2) + "\n")
    if len(rpc) != 1 or not match or len(rows) != 19:
        raise RuntimeError("unexpected first-step trace structure")
    return record


if __name__ == "__main__":
    if sys.argv[1:] != ["once"]:
        raise SystemExit("usage: driver.py once")
    binary = prepare()
    result = run(binary)
    record = audit(result)
    print(json.dumps({"result": result, "audit": record}, indent=2))
