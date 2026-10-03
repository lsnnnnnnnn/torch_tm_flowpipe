#!/usr/bin/env python3
"""Numerical first-step gate for the 62 unrun Airplane binary-cover cells."""

from datetime import datetime, timezone
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time

N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
BASE = N / "runs/archcomp26_20261001/native_airplane_first_reject_trace_20261002_005"
PREVIOUS = N / "runs/archcomp26_20261001/native_airplane_binary6_firststep_20261002_007"
UPPER = N / "runs/archcomp26_20261001/native_airplane_binary6_coverage_gate_20261002_008"
NEW = N / "runs/archcomp26_20261001/native_airplane_binary6_numeric_cover_20261003_001"
WORK = NEW / "build/archcomp/airplane"
MAX_WALL_S = 600


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def saved_record(path):
    if not path.is_file():
        return False, "missing ranges.bin", None
    raw = path.read_bytes()
    if len(raw) != 408:
        return False, f"ranges.bin has {len(raw)} bytes, expected 408", None
    lane, step, h, *values = struct.unpack("<QQd48d", raw)
    rows = [values[i:i + 4] for i in range(0, 48, 4)]
    good = (lane == 0 and step == 1 and math.isfinite(h) and
            abs(h - 0.01) <= 1e-9 and all(
                all(math.isfinite(v) for v in row) and
                row[0] <= row[2] <= row[3] <= row[1]
                for row in rows))
    return good, "valid" if good else "invalid range header or bounds", rows


def audit_files(run, bits, plan):
    try:
        entry = next(box for box in plan["boxes"] if box["bits"] == bits)
        ledger = json.loads((run / "initial_boxes.json").read_text())
        rpc = [json.loads(line) for line in
               (run / "controller_rpc.jsonl").read_text().splitlines()]
        result = json.loads((run / "RESULT.json").read_text())
        native = (run / "native.log").read_text()
        safety = (run / "safety.tsv").read_text().splitlines()
        record_ok, record_reason, rows = saved_record(run / "ranges.bin")
        status = re.findall(r"PERIOD 0 FLOWPIPES (\d+) STATUS (\d+) BOX_SAFE (\d+)", native)
        picard = [line for line in native.splitlines()
                  if line.startswith("AIRPLANE_PICARD\t")]
        rpc_ok = (len(rpc) == 1 and ledger == [entry["box"]] and
                  rpc[0]["input_lower"] == [v[0] for v in ledger[0][:12]] and
                  rpc[0]["input_upper"] == [v[1] for v in ledger[0][:12]])
        safety_ok = False
        if record_ok and len(safety) == 2:
            fields = safety[1].split("\t")
            safety_ok = (len(fields) == 13 and fields[:3] == ["0", "1", "1"] and
                         all(tuple(rows[index][:2]) ==
                             (float(fields[offset]), float(fields[offset + 1]))
                             for index, offset in ((1, 3), (6, 5), (7, 7), (8, 9))) and
                         math.isfinite(float(fields[11])) and
                         math.isfinite(float(fields[12])) and
                         float(fields[11]) <= float(fields[12]))
        one_step = len(status) == 1 and status[0][0] == "1" and status[0][1] in ("2", "3")
        author_safe = one_step and status[0][1] == "2" and status[0][2] == "1"
        process_ok = (one_step and not result["timed_out"] and
                      result["exit_code"] == (0 if author_safe else 2))
        numerical = (process_ok and rpc_ok and record_ok and safety_ok and
                     len(picard) == 19 and
                     all("\tsubset=1" in line for line in picard) and
                     "FLOWPIPE_SEGMENTS 1" in native)
        return {"bits": bits, "numerical_first_step_accepted": numerical,
                "native_status": status[0][1] if len(status) == 1 else None,
                "independent_box_safe": status[0][2] if len(status) == 1 else None,
                "property_status": "one_step_safe" if author_safe else
                                   "unknown" if one_step else "unavailable",
                "controller_calls": len(rpc), "first_picard_rows": len(picard),
                "range_record": record_reason, "safety_record_valid": safety_ok,
                "run_result": result, "scope": "one cell, first h=0.01 step only"}
    except (OSError, ValueError, KeyError, IndexError, TypeError, StopIteration) as exc:
        return {"bits": bits, "numerical_first_step_accepted": False,
                "property_status": "unavailable", "record_error": str(exc),
                "scope": "one cell, first h=0.01 step only"}


def prepare():
    if NEW.exists():
        raise RuntimeError(f"run identity already exists: {NEW}")
    source = BASE / "build/archcomp/airplane/airplane_first_reject_trace.cpp"
    library = BASE / "build/libflowstar.a"
    plan = json.loads((PREVIOUS / "SPLIT_PLAN.json").read_text())
    bits = [box["bits"] for box in plan["boxes"]]
    if len(bits) != 64 or set(bits) != {
            "".join(map(str, item)) for item in product((0, 1), repeat=6)}:
        raise RuntimeError("binary cover plan has changed")
    for old_bits, old_run in (("000000", PREVIOUS / "run"),
                              ("111111", UPPER / "cells/111111")):
        old_audit = audit_files(old_run, old_bits, plan)
        if not old_audit["numerical_first_step_accepted"]:
            raise RuntimeError(f"saved prior numerical acceptance missing: {old_bits}: {old_audit}")
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
    cpp = WORK / "airplane_binary6_numeric_cover.cpp"
    binary = WORK / "airplane_binary6_numeric_cover"
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
            try:
                rc = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                rc = process.wait(timeout=10)
    result = {"ended_utc": utc(), "status": "timeout" if timed_out else
              "completed" if rc == 0 else "failed", "exit_code": rc,
              "timed_out": timed_out, "wall_s": time.monotonic() - started,
              "bits": bits}
    save(run / "RESULT.json", result)
    audit = audit_files(run, bits, plan)
    save(run / "AUDIT.json", audit)
    return audit


def main():
    started = time.monotonic()
    deadline = started + MAX_WALL_S
    binary, plan = prepare()
    all_bits = ["".join(map(str, bits)) for bits in product((0, 1), repeat=6)]
    queue = sorted((bits for bits in all_bits if bits not in
                    ("000000", "111111")), key=lambda bits: (-bits.count("1"), bits))
    record = {"started_utc": utc(), "deadline_s": MAX_WALL_S,
              "prior_numerically_accepted_cells": ["000000", "111111"],
              "prior_property_status": {"000000": "one_step_safe", "111111": "unknown"},
              "queue": queue,
              "attempted": [], "status": "running",
              "scope": "only first h=0.01 step per unrun cell; stop on first numerical refusal or missing record; property tracked separately"}
    for bits in queue:
        if time.monotonic() + 12 >= deadline:
            record["status"] = "time_budget_reached"
            break
        audit = run_cell(binary, bits, plan, deadline)
        record["attempted"].append({"bits": bits,
                                    "numerical_first_step_accepted": audit["numerical_first_step_accepted"],
                                    "property_status": audit["property_status"],
                                    "native_status": audit.get("native_status")})
        print(json.dumps(record["attempted"][-1]), flush=True)
        save(NEW / "COVERAGE_GATE.json", record)
        if not audit["numerical_first_step_accepted"]:
            record["status"] = "first_numeric_gate_stop"
            break
    else:
        record["status"] = "all_64_first_steps_numerically_accepted"
    record["ended_utc"] = utc()
    record["wall_s"] = time.monotonic() - started
    save(NEW / "COVERAGE_GATE.json", record)
    print(json.dumps({"status": record["status"], "attempted": len(record["attempted"]),
                      "wall_s": record["wall_s"]}), flush=True)


def offline_check(root):
    plan = json.loads((root / "native_airplane_binary6_firststep_20261002_007/SPLIT_PLAN.json").read_text())
    bits = [box["bits"] for box in plan["boxes"]]
    expected = {"".join(map(str, item)) for item in product((0, 1), repeat=6)}
    assert len(bits) == 64 and set(bits) == expected
    prior = {}
    for key, name in (("000000", "native_airplane_binary6_firststep_20261002_007/run"),
                      ("111111", "native_airplane_binary6_coverage_gate_20261002_008/cells/111111")):
        prior[key] = audit_files(root / name, key, plan)
        assert prior[key]["numerical_first_step_accepted"], prior[key]
    assert prior["000000"]["property_status"] == "one_step_safe"
    assert prior["111111"]["property_status"] == "unknown"
    queue = sorted(expected - set(prior), key=lambda key: (-key.count("1"), key))
    assert len(queue) == 62 and queue[0] == "011111"
    print(json.dumps({"prior": {key: value["property_status"] for key, value in prior.items()},
                      "unrun_cells": len(queue), "first_unrun": queue[0],
                      "numerical_job_started": False}))


if __name__ == "__main__":
    if sys.argv[1:] == ["once"]:
        main()
    elif len(sys.argv) == 3 and sys.argv[1] == "offline":
        offline_check(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: driver.py once | driver.py offline LOCAL_EVIDENCE_ROOT")
