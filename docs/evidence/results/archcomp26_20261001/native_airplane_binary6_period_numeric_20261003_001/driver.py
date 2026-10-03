#!/usr/bin/env python3
"""One isolated Airplane held-control period for binary6 cell 111111."""

from datetime import datetime, timezone
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
ROOT = N / "runs/archcomp26_20261001"
OLD = ROOT / "native_airplane_binary6_numeric_cover_20261003_001"
CORNER = ROOT / "native_airplane_binary6_coverage_gate_20261002_008/cells/111111"
NEW = ROOT / "native_airplane_binary6_period_numeric_20261003_001"
WORK = NEW / "build/archcomp/airplane"
RUN = NEW / "cells/111111"
MAX_WALL_S = 180
RECORD = struct.Struct("<QQd48d")


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def ranges(path):
    raw = path.read_bytes()
    if len(raw) % RECORD.size:
        raise ValueError(f"invalid range byte count: {len(raw)}")
    result = []
    for offset in range(0, len(raw), RECORD.size):
        lane, step, h, *values = RECORD.unpack_from(raw, offset)
        rows = [values[i:i + 4] for i in range(0, 48, 4)]
        if (lane != 0 or step != len(result) + 1 or not math.isfinite(h) or
                abs(h - 0.01) > 1e-9 or any(
                    not all(math.isfinite(v) for v in row) or
                    not (row[0] <= row[2] <= row[3] <= row[1]) for row in rows)):
            raise ValueError(f"invalid range record {len(result) + 1}")
        result.append((lane, step, h, rows))
    return result


def rpc(path):
    lines = path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError(f"expected one controller RPC, got {len(lines)}")
    return json.loads(lines[0])


def changed_source(text):
    edits = (
        ("// Flow*: keep the full local tube and stop on any non-safe or incomplete step.",
         "// Numerically advance one held-control period; audit property from saved tubes."),
        ("author_matched::reach(dynamics, result, initial_set, 0.01, setting, safeSet, symbolic_remainder);",
         "author_matched::reach(dynamics, result, initial_set, 0.1, setting, std::vector<Constraint>{}, symbolic_remainder);"),
        ("if (result.status != COMPLETED_SAFE || produced != 1 || !independently_box_safe) {",
         "if (result.status != COMPLETED_SAFE || produced != 10) {"),
        ("ONE_STEP_NUMERIC_ACCEPTED_ONLY", "FIRST_HELD_PERIOD_NUMERIC_ACCEPTED_ONLY"),
    )
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f"source marker changed: {old}")
        text = text.replace(old, new)
    return text


def audit(run, prior):
    try:
        old_rpc, new_rpc = rpc(prior / "controller_rpc.jsonl"), rpc(run / "controller_rpc.jsonl")
        old_ranges, new_ranges = ranges(prior / "ranges.bin"), ranges(run / "ranges.bin")
        if len(old_ranges) != 1:
            raise ValueError("old high-corner receipt is not one step")
        same_box = json.loads((run / "initial_boxes.json").read_text()) == json.loads(
            (prior / "initial_boxes.json").read_text())
        native = (run / "native.log").read_text()
        outer = json.loads((run / "RESULT.json").read_text())
        status = re.findall(r"PERIOD 0 FLOWPIPES (\d+) STATUS (\d+) BOX_SAFE (\d+)", native)
        if len(status) != 1:
            raise ValueError("expected one native period status")
        produced, native_status, reported_box_safe = map(int, status[0])
        if produced != len(new_ranges) or f"FLOWPIPE_SEGMENTS {produced}" not in native:
            raise ValueError("native and saved segment counts differ")
        lines = (run / "safety.tsv").read_text().splitlines()
        if len(lines) != produced + 1:
            raise ValueError("tube-check row count differs")
        property_inside = True if produced else None
        cos_positive = True if produced else None
        for index, line in enumerate(lines[1:]):
            fields = line.split("\t")
            if len(fields) != 13 or fields[:3] != ["0", str(index + 1), str(index + 1)]:
                raise ValueError(f"invalid tube-check row {index + 1}")
            pairs = [(float(fields[j]), float(fields[j + 1])) for j in (3, 5, 7, 9, 11)]
            if any(not all(map(math.isfinite, pair)) or pair[0] > pair[1] for pair in pairs):
                raise ValueError(f"invalid tube-check bounds {index + 1}")
            for physical, pair in zip((1, 6, 7, 8), pairs[:4]):
                if list(pair) != new_ranges[index][3][physical][:2]:
                    raise ValueError(f"tube-check/range mismatch {index + 1}")
                if pair[0] < -1 or pair[1] > 1:
                    property_inside = False
            if pairs[4][0] <= 0:
                cos_positive = False
        if produced and reported_box_safe != int(property_inside and cos_positive):
            raise ValueError("native BOX_SAFE disagrees with saved tube check")
        picard = [line for line in native.splitlines() if line.startswith("AIRPLANE_PICARD\t")]
        # First-Picard subset=0 may later refine to an accepted step.
        complete = (produced == 10 and native_status == 2 and outer["exit_code"] == 0 and
                    not outer["timed_out"])
        refused = (produced < 10 and native_status == 4 and outer["exit_code"] == 2 and
                   not outer["timed_out"])
        same_first = new_ranges[0] == old_ranges[0] if new_ranges else None
        compare = {"initial_box_equal": same_box, "controller_rpc_equal": new_rpc == old_rpc,
                   "first_saved_segment_equal": same_first}
        drift = not same_box or not compare["controller_rpc_equal"] or same_first is False
        label = ("first_step_contract_drift" if drift else
                 "solver_period_domain_unresolved" if complete and not cos_positive else
                 "numeric_period_completed" if complete else
                 "first_numeric_refusal_uncompared" if refused and same_first is None else
                 "first_numeric_refusal" if refused else "unclassified_stop")
        return {"status": label, "saved_segments": produced,
                "first_numeric_refusal_step": produced + 1 if refused else None,
                "native_status": native_status, "native_box_safe": reported_box_safe,
                "outer_result": outer,
                "direct_prior_comparison": compare, "saved_picard_rows": len(picard),
                "first_picard_noninclusions": sum("\tsubset=0" in row for row in picard),
                "saved_prefix_property": None if property_inside is None else
                                         "inside" if property_inside else "unknown",
                "all_saved_cos_theta_positive": cos_positive,
                "denominator_domain_unresolved": None if cos_positive is None else not cos_positive,
                "solver_period_generated": complete,
                "numerical_period_completed": complete and not drift and bool(cos_positive),
                "scope": "one 0.1 s held-control period, high-corner cell only"}
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return {"status": "receipt_or_infrastructure_error", "error": str(exc),
                "numerical_period_completed": False,
                "scope": "one 0.1 s held-control period, high-corner cell only"}


def prepare():
    if NEW.exists():
        raise RuntimeError(f"run identity already exists: {NEW}")
    old_work = OLD / "build/archcomp/airplane"
    old_cpp = old_work / "airplane_binary6_numeric_cover.cpp"
    old_build = json.loads((OLD / "BUILD.json").read_text())
    old_ranges = ranges(CORNER / "ranges.bin")
    if len(old_ranges) != 1 or rpc(CORNER / "controller_rpc.jsonl")["input_lower"][3:9] != [0.5] * 6:
        raise RuntimeError("old high-corner receipt unavailable")
    text = changed_source(old_cpp.read_text())
    for path in (OLD / "build/flowstar", old_work / "matched_reach.h",
                 old_work / "arch_ranges.h", old_work / "crown_paper.py",
                 Path(old_build["library"])):
        if not path.exists():
            raise FileNotFoundError(path)
    WORK.mkdir(parents=True)
    shutil.copy2(__file__, NEW / "driver.py")
    (NEW / "build/flowstar").symlink_to((OLD / "build/flowstar").resolve(),
                                         target_is_directory=True)
    cpp = WORK / "airplane_binary6_period_numeric.cpp"
    binary = WORK / "airplane_binary6_period_numeric"
    cpp.write_text(text)
    for name in ("matched_reach.h", "arch_ranges.h", "crown_paper.py"):
        shutil.copy2(old_work / name, WORK / name)
    old_binary = old_work / "airplane_binary6_numeric_cover"
    command = [str(cpp) if arg == str(old_cpp) else
               str(binary) if arg == str(old_binary) else arg
               for arg in old_build["command"]]
    if str(cpp) not in command or str(binary) not in command:
        raise RuntimeError("old compiler command does not name expected source/output")
    with (NEW / "build.log").open("x") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                              timeout=300, check=False)
    save(NEW / "BUILD.json", {"created_utc": utc(), "source": str(old_cpp),
                              "library": old_build["library"], "command": command,
                              "returncode": proc.returncode,
                              "binary_bytes": binary.stat().st_size if binary.exists() else None,
                              "change": "one 0.1 s held-control period; property checked from saved tubes"})
    if proc.returncode:
        raise RuntimeError("isolated build failed")
    return binary


def main():
    binary = prepare()
    RUN.mkdir(parents=True)
    env = os.environ.copy()
    env.update({"AIRPLANE_RUN_DIR": str(RUN), "AIRPLANE_WORKDIR": str(WORK),
                "AIRPLANE_BINARY": str(binary), "AIRPLANE_CELL_BITS": "111111",
                "AIRPLANE_SERVER_PYTHON": str(N / "nncs_env/bin/python"),
                "AIRPLANE_OVERLAY": str(ROOT / "native_dp_less_rpc_overlay_preflight_001/rpc_overlay"),
                "AIRPLANE_MODEL": str(ROOT / "airplane_prep_001/controller_airplane.onnx"),
                "AIRPLANE_CPUSET": "10-13", "AIRPLANE_NATIVE_AS": "34359738368",
                "CUDA_VISIBLE_DEVICES": "2", "OMP_NUM_THREADS": "1"})
    launcher = N / "tools/run_archcomp26_airplane_native_fullbox_pair.sh"
    save(RUN / "START.json", {"started_utc": utc(), "command": ["/bin/bash", str(launcher)],
                              "selected_environment": {key: env[key] for key in (
                                  "AIRPLANE_RUN_DIR", "AIRPLANE_WORKDIR", "AIRPLANE_BINARY",
                                  "AIRPLANE_CELL_BITS", "AIRPLANE_SERVER_PYTHON",
                                  "AIRPLANE_OVERLAY", "AIRPLANE_MODEL", "AIRPLANE_CPUSET",
                                  "AIRPLANE_NATIVE_AS", "CUDA_VISIBLE_DEVICES",
                                  "OMP_NUM_THREADS")},
                              "scope": "cell 111111, first held-control period only"})
    started = time.monotonic()
    with (RUN / "launcher.stdout.log").open("x") as out, (RUN / "launcher.stderr.log").open("x") as err:
        proc = subprocess.Popen(["/bin/bash", str(launcher)], cwd=WORK, env=env,
                                stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            rc = proc.wait(timeout=MAX_WALL_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait(timeout=10)
    save(RUN / "RESULT.json", {"ended_utc": utc(),
                               "status": "timeout" if timed_out else "completed" if rc == 0 else "failed",
                               "exit_code": rc, "timed_out": timed_out,
                               "wall_s": time.monotonic() - started, "bits": "111111"})
    receipt = audit(RUN, CORNER)
    save(RUN / "AUDIT.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["numerical_period_completed"] else 2


def offline(root):
    old_cpp = root / OLD.name / "build/archcomp/airplane/airplane_binary6_numeric_cover.cpp"
    prior = root / "native_airplane_binary6_coverage_gate_20261002_008/cells/111111"
    assert len(ranges(prior / "ranges.bin")) == 1
    assert rpc(prior / "controller_rpc.jsonl")["input_lower"][3:9] == [0.5] * 6
    source = changed_source(old_cpp.read_text())
    assert "0.1, setting, std::vector<Constraint>{}, symbolic_remainder" in source
    assert "produced != 10" in source
    print(json.dumps({"prior_corner_valid": True, "isolated_source_change_valid": True,
                      "numerical_job_started": False}))


if __name__ == "__main__":
    if sys.argv[1:] == ["once"]:
        raise SystemExit(main())
    if len(sys.argv) == 3 and sys.argv[1] == "offline":
        offline(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: driver.py once | driver.py offline LOCAL_EVIDENCE_ROOT")
