#!/usr/bin/env python3
"""Bounded one-step gate for the saved Airplane 64-cell binary cover."""

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
PREVIOUS = N / "runs/archcomp26_20261001/native_airplane_binary6_firststep_20261002_007"
NEW = N / "runs/archcomp26_20261001/native_airplane_binary6_coverage_gate_20261002_008"
WORK = NEW / "build/archcomp/airplane"
COORDS = ("u", "v", "w", "phi", "theta", "psi")
INDICES = (3, 4, 5, 6, 7, 8)
MAX_WALL_S = 600


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    if NEW.exists():
        raise RuntimeError(f"run identity already exists: {NEW}")
    source = BASE / "build/archcomp/airplane/airplane_first_reject_trace.cpp"
    library = BASE / "build/libflowstar.a"
    plan = json.loads((PREVIOUS / "SPLIT_PLAN.json").read_text())
    if len(plan["boxes"]) != 64:
        raise RuntimeError("binary cover plan has changed")
    for path in (source, library, BASE / "build/flowstar",
                 source.parent / "crown_paper.py"):
        if not path.exists():
            raise FileNotFoundError(path)
    NEW.mkdir()
    WORK.mkdir(parents=True)
    shutil.copy2(__file__, NEW / "driver.py")
    save(NEW / "SPLIT_PLAN.json", plan)
    (NEW / "build/flowstar").symlink_to((BASE / "build/flowstar").resolve(),
                                         target_is_directory=True)
    text = source.read_text()
    old = """             init_u(0, 1), init_v(0, 1), init_w(0, 1),
             init_phi(0, 1), init_theta(0, 1), init_psi(0, 1),"""
    new = """             init_u(lo(0), hi(0)), init_v(lo(1), hi(1)), init_w(lo(2), hi(2)),
             init_phi(lo(3), hi(3)), init_theta(lo(4), hi(4)), init_psi(lo(5), hi(5)),"""
    if text.count(old) != 1:
        raise RuntimeError("frozen initializers changed")
    text = text.replace(old, new)
    text = text.replace("#include <stdexcept>\n", "#include <stdexcept>\n#include <string>\n", 1)
    marker = "    // Initial set\n"
    setup = """    const char *bits_env = std::getenv("AIRPLANE_CELL_BITS");
    if (!bits_env) throw std::runtime_error("AIRPLANE_CELL_BITS required");
    const std::string bits(bits_env);
    if (bits.size() != 6 || bits.find_first_not_of("01") != std::string::npos)
        throw std::runtime_error("invalid AIRPLANE_CELL_BITS");
    auto lo = [&](size_t i) { return bits[i] == '1' ? 0.5 : 0.0; };
    auto hi = [&](size_t i) { return bits[i] == '1' ? 1.0 : 0.5; };

"""
    if text.count(marker) != 1:
        raise RuntimeError("initial-set marker changed")
    text = text.replace(marker, setup + marker)
    cpp = WORK / "airplane_binary6_coverage_gate.cpp"
    binary = WORK / "airplane_binary6_coverage_gate"
    cpp.write_text(text)
    for name in ("matched_reach.h", "arch_ranges.h", "crown_paper.py"):
        shutil.copy2(source.parent / name, WORK / name)
    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    command = ["taskset", "-c", "18", "/usr/bin/g++-15", "-O3", "-std=c++11",
               "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
               "-I", "/usr/include/jsoncpp", str(cpp), str(library),
               "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm", "-lglpk",
               "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp", "-lcurl",
               "-ljsonrpccpp-common", "-ljsonrpccpp-client",
               "-l:libboost_thread.so.1.90.0", "-o", str(binary)]
    with (NEW / "build.log").open("x") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                              timeout=300, check=False)
    save(NEW / "BUILD.json", {"created_utc": utc(), "source": str(source),
                              "library": str(library), "command": command,
                              "returncode": proc.returncode,
                              "binary_bytes": binary.stat().st_size if binary.exists() else None,
                              "change": "six saved initial intervals selected by six-bit cell ID"})
    if proc.returncode:
        raise RuntimeError("isolated build failed")
    return binary, plan


def run_cell(binary, bits, plan, deadline):
    entry = next(box for box in plan["boxes"] if box["bits"] == bits)
    run = NEW / "cells" / bits
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"AIRPLANE_RUN_DIR": str(run), "AIRPLANE_WORKDIR": str(WORK),
                "AIRPLANE_BINARY": str(binary), "AIRPLANE_CELL_BITS": bits,
                "AIRPLANE_SERVER_PYTHON": str(N / "nncs_env/bin/python"),
                "AIRPLANE_OVERLAY": str(N / "runs/archcomp26_20261001/native_dp_less_rpc_overlay_preflight_001/rpc_overlay"),
                "AIRPLANE_MODEL": str(N / "runs/archcomp26_20261001/airplane_prep_001/controller_airplane.onnx"),
                "AIRPLANE_CPUSET": "10-13", "AIRPLANE_NATIVE_AS": "34359738368",
                "CUDA_VISIBLE_DEVICES": "2", "OMP_NUM_THREADS": "1"})
    launcher = N / "tools/run_archcomp26_airplane_native_fullbox_pair.sh"
    save(run / "START.json", {"started_utc": utc(),
                              "command": ["/bin/bash", str(launcher)],
                              "selected_environment": {key: env[key] for key in (
                                  "AIRPLANE_RUN_DIR", "AIRPLANE_WORKDIR", "AIRPLANE_BINARY",
                                  "AIRPLANE_CELL_BITS", "AIRPLANE_SERVER_PYTHON",
                                  "AIRPLANE_OVERLAY", "AIRPLANE_MODEL", "AIRPLANE_CPUSET",
                                  "AIRPLANE_NATIVE_AS", "CUDA_VISIBLE_DEVICES",
                                  "OMP_NUM_THREADS")},
                              "scope": "one binary-cover cell, one h=0.01 step only"})
    started = time.monotonic()
    with (run / "launcher.stdout.log").open("x") as out, (run / "launcher.stderr.log").open("x") as err:
        process = subprocess.Popen(["/bin/bash", str(launcher)], cwd=WORK, env=env,
                                   stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            rc = process.wait(timeout=min(180, max(1, deadline - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            rc = process.wait(timeout=10)
    result = {"ended_utc": utc(), "status": "timeout" if timed_out else
              "completed" if rc == 0 else "failed", "exit_code": rc,
              "timed_out": timed_out, "wall_s": time.monotonic() - started,
              "bits": bits}
    save(run / "RESULT.json", result)
    ledger_path = run / "initial_boxes.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    rpc_path = run / "controller_rpc.jsonl"
    rpc = rpc_path.read_text().splitlines() if rpc_path.exists() else []
    native_path = run / "native.log"
    native = native_path.read_text() if native_path.exists() else ""
    range_path = run / "ranges.bin"
    range_bytes = range_path.stat().st_size if range_path.exists() else None
    status = re.findall(r"PERIOD 0 FLOWPIPES (\d+) STATUS (\d+) BOX_SAFE (\d+)", native)
    accepted = (ledger == [entry["box"]] and result["status"] == "completed" and len(rpc) == 1 and
                status == [("1", "2", "1")] and
                range_bytes == 408)
    audit = {"bits": bits, "actual_initial_box": ledger,
             "contract_ledger_matches_plan": ledger == [entry["box"]],
             "controller_calls": len(rpc), "period_status": status,
             "first_picard_rows": native.count("AIRPLANE_PICARD\t"),
             "range_bytes": range_bytes,
             "accepted_one_step": accepted, "scope": "one cell; no full period or T=2"}
    save(run / "AUDIT.json", audit)
    return audit


def main():
    started = time.monotonic()
    deadline = started + MAX_WALL_S
    binary, plan = prepare()
    all_bits = ["".join(map(str, bits)) for bits in product((0, 1), repeat=6)]
    queue = ["111111"] + sorted((bits for bits in all_bits if bits not in
                                     ("000000", "111111")),
                                    key=lambda bits: (-bits.count("1"), bits))
    record = {"started_utc": utc(), "deadline_s": MAX_WALL_S,
              "prior_accepted_cell": "000000", "queue": queue,
              "attempted": [], "status": "running",
              "scope": "only first h=0.01 step per cell; stop on first refusal"}
    for bits in queue:
        if time.monotonic() + 12 >= deadline:
            record["status"] = "time_budget_reached"
            break
        audit = run_cell(binary, bits, plan, deadline)
        record["attempted"].append({"bits": bits,
                                    "accepted_one_step": audit["accepted_one_step"],
                                    "period_status": audit["period_status"]})
        print(json.dumps(record["attempted"][-1]), flush=True)
        save(NEW / "COVERAGE_GATE.json", record)
        if not audit["accepted_one_step"]:
            record["status"] = "first_refusal"
            break
    else:
        record["status"] = "all_64_first_steps_accepted"
    record["ended_utc"] = utc()
    record["wall_s"] = time.monotonic() - started
    save(NEW / "COVERAGE_GATE.json", record)
    print(json.dumps({"status": record["status"], "attempted": len(record["attempted"]),
                      "wall_s": record["wall_s"]}), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["once"]:
        raise SystemExit("usage: driver.py once")
    main()
