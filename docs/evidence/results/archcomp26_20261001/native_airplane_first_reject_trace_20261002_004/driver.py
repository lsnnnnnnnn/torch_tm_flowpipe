#!/usr/bin/env python3
"""Copy the frozen Airplane native path and record its first Picard refusal."""

from datetime import datetime, timezone
import difflib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
PREVIOUS = N / "runs/archcomp26_20261001/native_airplane_fullbox_build_001/archcomp/airplane"
NEW = N / "runs/archcomp26_20261001/native_airplane_first_reject_trace_20261002_004"
BUILD = NEW / "build"
WORK = BUILD / "archcomp/airplane"
RUN = NEW / "run"
VARIABLES = ["x", "y", "z", "u", "v", "w", "phi", "theta", "psi",
             "r", "p", "q", "t", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"expected one marker: {old[:90]}")
    return source.replace(old, new)


def build():
    if NEW.exists():
        raise RuntimeError(f"isolated run already exists: {NEW}")
    for path in (OLD / "flowstar/flowstar-toolbox/Continuous.cpp",
                 OLD / "flowstar/flowstar-toolbox/libflowstar.a",
                 PREVIOUS / "airplane_fullbox_order3_smoke1.cpp",
                 PREVIOUS / "crown_paper.py",
                 N / "tools/run_archcomp26_airplane_native_fullbox_pair.sh"):
        if not path.is_file():
            raise FileNotFoundError(path)
    NEW.mkdir()
    BUILD.mkdir()
    WORK.mkdir(parents=True)
    RUN.mkdir()
    shutil.copy2(__file__, NEW / "driver.py")
    (BUILD / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)

    original = (OLD / "flowstar/flowstar-toolbox/Continuous.cpp").read_text()
    start_marker = "int Flowpipe::advance(Flowpipe & result, const std::vector<Expression<Interval> > & ode,"
    if original.count(start_marker) != 2:
        raise RuntimeError("expected the ordinary and symbolic-remainder interval advances")
    start = original.rindex(start_marker)
    end = original.index("\nint Flowpipe::", start + 1)
    section = original[start:end]
    marker = "\n\t// add the uncertainties and the cutoff intervals onto the result\n"
    if section.count(marker) != 1:
        raise RuntimeError("symbolic-remainder first Picard branch changed")
    trace = r'''\tstd::cerr << std::setprecision(17);
\tfor(unsigned int i=0; i<rangeDim; ++i)
\t{
\t\tInterval proposed = tmvTmp.tms[i].remainder;
\t\tproposed += intDifferences[i];
\t\tstd::cerr << "AIRPLANE_PICARD\t" << i
\t\t\t<< "\told_lo=" << x.tms[i].remainder.inf()
\t\t\t<< "\told_hi=" << x.tms[i].remainder.sup()
\t\t\t<< "\tbase_lo=" << tmvTmp.tms[i].remainder.inf()
\t\t\t<< "\tbase_hi=" << tmvTmp.tms[i].remainder.sup()
\t\t\t<< "\tdiff_lo=" << intDifferences[i].inf()
\t\t\t<< "\tdiff_hi=" << intDifferences[i].sup()
\t\t\t<< "\tnew_lo=" << proposed.inf()
\t\t\t<< "\tnew_hi=" << proposed.sup()
\t\t\t<< "\tsubset=" << proposed.subseteq(x.tms[i].remainder)
\t\t\t<< std::endl;
\t}
'''.replace("\\t", "\t")
    section = section.replace(marker, trace + "\n" + marker)
    patched = once(original, original[start:end], section)
    patched = once(patched, '#include "Continuous.h"',
                   '#include "Continuous.h"\n#include <iostream>\n#include <iomanip>')
    (BUILD / "Continuous.original.cpp").write_text(original)
    (BUILD / "Continuous.cpp").write_text(patched)
    diff = difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                fromfile="frozen/Continuous.cpp", tofile="isolated/Continuous.cpp")
    (NEW / "Continuous.patch").write_text("".join(diff))
    library = BUILD / "libflowstar.a"
    shutil.copy2(OLD / "flowstar/flowstar-toolbox/libflowstar.a", library)
    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    compile_cpp = ["taskset", "-c", "18", "/usr/bin/g++-15", "-O3", "-std=c++11",
                   "-fpermissive", "-Wno-template-body", "-fopenmp",
                   "-I", str(OLD / "flowstar/flowstar-toolbox"), "-I", str(include),
                   "-I", "/usr/include/jsoncpp", "-c", str(BUILD / "Continuous.cpp"),
                   "-o", str(BUILD / "Continuous.o")]
    commands = [compile_cpp,
                ["ar", "rcs", str(library), str(BUILD / "Continuous.o")]]

    source = (PREVIOUS / "airplane_fullbox_order3_smoke1.cpp").read_text()
    source = once(source, "author_matched::reach(dynamics, result, initial_set, 0.1,",
                  "author_matched::reach(dynamics, result, initial_set, 0.01,")
    source = once(source, "produced != 10", "produced != 1")
    source = once(source, 'cout << "VERIFIED" << endl;',
                  'cout << "ONE_STEP_NUMERIC_ACCEPTED_ONLY" << endl;')
    source = once(source, 'cout << "COMPLETED_SAFE_PERIODS "',
                  'cout << "DIAGNOSTIC_COMPLETED_CALLS "')
    cpp = WORK / "airplane_first_reject_trace.cpp"
    binary = WORK / "airplane_first_reject_trace"
    cpp.write_text(source)
    for name in ("matched_reach.h", "arch_ranges.h", "crown_paper.py"):
        shutil.copy2(PREVIOUS / name, WORK / name)
    link = ["taskset", "-c", "18", "/usr/bin/g++-15", "-O3", "-std=c++11",
            "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
            "-I", "/usr/include/jsoncpp", str(cpp), str(library), "-lmpfr", "-lgmp",
            "-lgsl", "-lgslcblas", "-lm", "-lglpk", "-lcolamd", "-lamd",
            "-lz", "-lltdl", "-ljsoncpp", "-lcurl", "-ljsonrpccpp-common",
            "-ljsonrpccpp-client", "-l:libboost_thread.so.1.90.0", "-o", str(binary)]
    commands.append(link)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(OLD / "flowstar/flowstar-toolbox/Continuous.cpp"),
              "airplane_source": str(PREVIOUS / "airplane_fullbox_order3_smoke1.cpp"),
              "changes": ["copy archive and instrument first Picard only",
                          "one original full initial box, one h=0.01 step"],
              "commands": commands, "returncodes": []}
    for index, command in enumerate(commands):
        with (NEW / f"build_{index}.log").open("x") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                    timeout=300, check=False)
        record["returncodes"].append(result.returncode)
        (NEW / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"isolated build command {index} failed")
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
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["/bin/bash", str(launcher)],
        "selected_environment": {key: env[key] for key in (
            "AIRPLANE_RUN_DIR", "AIRPLANE_WORKDIR", "AIRPLANE_BINARY",
            "AIRPLANE_SERVER_PYTHON", "AIRPLANE_OVERLAY", "AIRPLANE_MODEL",
            "AIRPLANE_CPUSET", "AIRPLANE_NATIVE_AS", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS")}, "timeout_s": 180,
        "scope": "one unsplit full box, one controller call, one h=0.01 step; first refusal stops"
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
    result = {"ended_utc": datetime.now(timezone.utc).isoformat(),
              "status": "timeout" if timed_out else "completed" if rc == 0 else "failed",
              "exit_code": rc, "timed_out": timed_out, "wall_s": time.monotonic() - start}
    (RUN / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit():
    native = (RUN / "native.log").read_text()
    rows = []
    for line in native.splitlines():
        if not line.startswith("AIRPLANE_PICARD\t"):
            continue
        parts = line.split("\t")
        row = {"index": int(parts[1]), "variable": VARIABLES[int(parts[1])]}
        for field in parts[2:]:
            key, value = field.split("=", 1)
            row[key] = int(value) if key == "subset" else float(value)
        for prefix in ("old", "base", "diff", "new"):
            if not (math.isfinite(row[prefix + "_lo"]) and
                    math.isfinite(row[prefix + "_hi"]) and
                    row[prefix + "_lo"] <= row[prefix + "_hi"]):
                raise RuntimeError(f"invalid {prefix} interval in {row['variable']}")
        rows.append(row)
    status = re.findall(r"PERIOD 0 FLOWPIPES (\d+) STATUS (\d+)", native)
    rpc = (RUN / "controller_rpc.jsonl").read_text().splitlines()
    receipt = {"picard_rows": len(rows), "variables": VARIABLES,
               "failed_coordinates": [row["variable"] for row in rows if not row["subset"]],
               "rows": rows, "period_status": status, "controller_rpc_lines": len(rpc),
               "range_bytes": (RUN / "ranges.bin").stat().st_size,
               "result": json.loads((RUN / "RESULT.json").read_text()),
               "scope": "first-step numerical refusal only; no full T=2 or property result"}
    (NEW / "AUDIT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if len(rows) != len(VARIABLES) or [r["index"] for r in rows] != list(range(len(VARIABLES))):
        raise RuntimeError("missing or reordered first-Picard coordinates")
    if not status or len(rpc) != 1:
        raise RuntimeError("missing status or unexpected controller RPC count")
    return receipt


if __name__ == "__main__":
    binary = build()
    result = run(binary)
    receipt = audit()
    print(json.dumps({"run": str(NEW), "result": result,
                      "failed_coordinates": receipt["failed_coordinates"],
                      "period_status": receipt["period_status"]}))
