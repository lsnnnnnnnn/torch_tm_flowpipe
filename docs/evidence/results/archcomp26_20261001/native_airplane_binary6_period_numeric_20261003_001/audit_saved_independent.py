#!/usr/bin/env python3
"""Audit copied Airplane 111111 period receipts offline; never start a solver."""

import json
import math
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL = HERE / "cells/111111"
PRIOR = HERE.parent / "native_airplane_binary6_coverage_gate_20261002_008/cells/111111"
OLD_CPP = HERE.parent / "native_airplane_binary6_numeric_cover_20261003_001/build/archcomp/airplane/airplane_binary6_numeric_cover.cpp"
NEW_CPP = HERE / "build/archcomp/airplane/airplane_binary6_period_numeric.cpp"
RECORD = struct.Struct("<QQd48d")
SAFETY_HEADER = "period\tlocal_substep\tglobal_substep\ty_lo\ty_hi\tphi_lo\tphi_hi\ttheta_lo\ttheta_hi\tpsi_lo\tpsi_hi\tcos_theta_lo\tcos_theta_hi"
SAFE_IDS = (1, 6, 7, 8)


def need(ok, why):
    if not ok:
        raise ValueError(why)


def read_json(path):
    return json.loads(path.read_text())


def read_ranges(path):
    raw = path.read_bytes()
    need(len(raw) % RECORD.size == 0, f"{path}: partial 408-byte record")
    out = []
    for n, offset in enumerate(range(0, len(raw), RECORD.size), 1):
        lane, step, h, *values = RECORD.unpack_from(raw, offset)
        rows = [values[i:i + 4] for i in range(0, 48, 4)]
        need((lane, step) == (0, n) and math.isfinite(h) and
             math.isclose(h, 0.01, rel_tol=0, abs_tol=1e-12),
             f"{path}: invalid range record header {n}")
        need(all(all(map(math.isfinite, row)) and row[0] <= row[2] <= row[3] <= row[1]
                 for row in rows), f"{path}: invalid tube/endpoint enclosure {n}")
        out.append(rows)
    return raw, out


def read_rpc(path, initial):
    lines = path.read_text().splitlines()
    need(len(lines) == 1, f"{path}: expected one controller RPC")
    row = json.loads(lines[0])
    need(row["input_lower"] == [v[0] for v in initial[:12]] and
         row["input_upper"] == [v[1] for v in initial[:12]],
         f"{path}: controller input differs from actual initial box")
    response = row["response"]
    need(len(response["T"]) == 6 and
         all(len(a) == 12 and all(math.isfinite(x) for x in a)
             for a in response["T"]) and
         len(response["u_min"]) == len(response["u_max"]) == 6 and
         all(math.isfinite(lo) and math.isfinite(hi) and lo <= hi
             for lo, hi in zip(response["u_min"], response["u_max"])),
         f"{path}: malformed saved affine controller response")
    return row


def check_source():
    old, new = OLD_CPP.read_text(), NEW_CPP.read_text()
    edits = (
        ("// Flow*: keep the full local tube and stop on any non-safe or incomplete step.",
         "// Numerically advance one held-control period; audit property from saved tubes."),
        ("author_matched::reach(dynamics, result, initial_set, 0.01, setting, safeSet, symbolic_remainder);",
         "author_matched::reach(dynamics, result, initial_set, 0.1, setting, std::vector<Constraint>{}, symbolic_remainder);"),
        ("if (result.status != COMPLETED_SAFE || produced != 1 || !independently_box_safe) {",
         "if (result.status != COMPLETED_SAFE || produced != 10) {"),
        ("ONE_STEP_NUMERIC_ACCEPTED_ONLY", "FIRST_HELD_PERIOD_NUMERIC_ACCEPTED_ONLY"),
    )
    for before, after in edits:
        need(old.count(before) == 1, f"old source marker missing: {before}")
        old = old.replace(before, after)
    need(new == old, "period source differs beyond four declared edits")
    need(read_json(HERE / "BUILD.json")["returncode"] == 0,
         "isolated native build did not complete")


def audit():
    check_source()
    old_start, start = read_json(PRIOR / "START.json"), read_json(CELL / "START.json")
    old_env, env = old_start["selected_environment"], start["selected_environment"]
    need(start["command"] == old_start["command"] and
         start["command"][-1].endswith("/run_archcomp26_airplane_native_fullbox_pair.sh"),
         "launcher differs from prior corner job")
    for key in ("AIRPLANE_CELL_BITS", "AIRPLANE_SERVER_PYTHON", "AIRPLANE_OVERLAY",
                "AIRPLANE_MODEL", "AIRPLANE_CPUSET", "AIRPLANE_NATIVE_AS",
                "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"):
        need(env[key] == old_env[key], f"frozen environment differs: {key}")
    need(env["AIRPLANE_CELL_BITS"] == "111111" and
         env["AIRPLANE_RUN_DIR"].endswith("/native_airplane_binary6_period_numeric_20261003_001/cells/111111") and
         env["AIRPLANE_BINARY"].endswith("/airplane_binary6_period_numeric"),
         "wrong period run identity")
    result = read_json(CELL / "RESULT.json")
    need(result["bits"] == "111111" and result["timed_out"] is False and
         math.isfinite(result["wall_s"]) and result["wall_s"] > 0,
         "timeout or malformed outer result")
    expected = [[0, 0] for _ in range(19)]
    for axis in (3, 4, 5, 6, 7, 8):
        expected[axis] = [0.5, 1]
    initial, old_initial = read_json(CELL / "initial_boxes.json"), read_json(PRIOR / "initial_boxes.json")
    need(initial == old_initial == [expected], "high-corner initial box changed")
    old_rpc = read_rpc(PRIOR / "controller_rpc.jsonl", old_initial[0])
    rpc = read_rpc(CELL / "controller_rpc.jsonl", initial[0])
    need(rpc == old_rpc, "first held controller response changed")
    need((CELL / "server.log").read_text().count('"POST / HTTP/1.1" 200 ') == 1,
         "one successful CROWN RPC not confirmed by server log")
    need((CELL / "ledger_checks.tsv").read_text().splitlines() == [SAFETY_HEADER] and
         not (CELL / "ledger.log").read_text().strip(),
         "unexpected later control/ledger activity")

    old_raw, old_ranges = read_ranges(PRIOR / "ranges.bin")
    raw, ranges = read_ranges(CELL / "ranges.bin")
    need(len(old_ranges) == 1 and len(ranges) <= 10, "unexpected saved segment count")
    if ranges:
        need(raw[:RECORD.size] == old_raw and ranges[0] == old_ranges[0],
             "new first raw segment differs from old high-corner source")
    native = (CELL / "native.log").read_text()
    periods = re.findall(r"^PERIOD (\d+) FLOWPIPES (\d+) STATUS (\d+) BOX_SAFE (\d+)$",
                         native, re.MULTILINE)
    need(len(periods) == 1 and periods[0][0] == "0" and
         int(periods[0][1]) == len(ranges), "native period/range count differs")
    status, reported_box_safe = map(int, periods[0][2:])
    need(re.findall(r"^FLOWPIPE_SEGMENTS (\d+)$", native, re.MULTILINE) ==
         [str(len(ranges))], "native saved-segment summary differs")

    safety = (CELL / "safety.tsv").read_text().splitlines()
    need(len(safety) == len(ranges) + 1 and safety[0] == SAFETY_HEADER,
         "saved tube-check rows differ from binary range count")
    safe_steps, domain_steps, cosine_lowers = [], [], []
    for n, (line, segment) in enumerate(zip(safety[1:], ranges), 1):
        fields = line.split("\t")
        need(len(fields) == 13 and fields[:3] == ["0", str(n), str(n)],
             f"invalid tube-check identity at step {n}")
        pairs = [tuple(map(float, fields[i:i + 2])) for i in (3, 5, 7, 9, 11)]
        need(all(all(map(math.isfinite, pair)) and pair[0] <= pair[1] for pair in pairs),
             f"invalid tube-check interval at step {n}")
        for pair, axis in zip(pairs[:4], SAFE_IDS):
            need(pair == tuple(segment[axis][:2]),
                 f"tube-check/range mismatch at step {n}, axis {axis}")
        theta_lo, theta_hi = pairs[2]
        # All saved theta intervals lie in the monotone positive-cosine branch.
        if -math.pi / 2 <= theta_lo <= theta_hi <= math.pi / 2:
            need(pairs[4][0] <= math.cos(theta_hi) + 1e-14 and
                 pairs[4][1] >= math.cos(theta_lo) - 1e-14,
                 f"saved cosine interval misses cos(theta) at step {n}")
        safe_steps.append(all(-1 <= lo <= hi <= 1 for lo, hi in pairs[:4]))
        domain_steps.append(pairs[4][0] > 0)
        cosine_lowers.append(pairs[4][0])
    need(reported_box_safe == int(all(safe_steps) and all(domain_steps)),
         "native BOX_SAFE differs from saved property/domain rows")

    picard = []
    for line in native.splitlines():
        if not line.startswith("AIRPLANE_PICARD\t"):
            continue
        fields = line.split("\t")
        need(len(fields) == 11 and fields[1].isdigit(), "malformed Picard trace")
        row = dict(field.split("=", 1) for field in fields[2:])
        need(set(row) == {"old_lo", "old_hi", "base_lo", "base_hi", "diff_lo",
                          "diff_hi", "new_lo", "new_hi", "subset"},
             "Picard trace fields changed")
        bounds = {key: float(value) for key, value in row.items() if key != "subset"}
        need(all(map(math.isfinite, bounds.values())) and
             bounds["old_lo"] <= bounds["old_hi"] and
             bounds["new_lo"] <= bounds["new_hi"] and row["subset"] in ("0", "1"),
             "nonfinite or invalid Picard interval")
        picard.append({"axis": int(fields[1]), "subset": int(row["subset"]), **bounds})
    need(len(picard) % 19 == 0 and
         all([row["axis"] for row in picard[i:i + 19]] == list(range(19))
             for i in range(0, len(picard), 19)),
         "Picard rows are not ordered 19-coordinate steps")
    groups = [picard[i:i + 19] for i in range(0, len(picard), 19)]
    refused_axes = [[row["axis"] for row in group if row["subset"] == 0]
                    for group in groups]
    completed = (len(ranges) == 10 and len(groups) == 10 and
                 not any(refused_axes) and status == 2 and
                 result["exit_code"] == 0 and result["status"] == "completed" and
                 "FIRST_HELD_PERIOD_NUMERIC_ACCEPTED_ONLY" in native)
    first_refusal = (len(ranges) < 10 and len(groups) == len(ranges) + 1 and
                     all(not axes for axes in refused_axes[:-1]) and
                     bool(refused_axes[-1]) and status == 4 and
                     result["exit_code"] == 2 and result["status"] == "failed" and
                     re.findall(r"^UNKNOWN$", native, re.MULTILINE) == ["UNKNOWN"])
    need(completed or first_refusal,
         "native/outer result is not a trace-supported completed period or first numerical refusal")
    need(re.findall(r"^DIAGNOSTIC_COMPLETED_CALLS (\d+)/1$", native, re.MULTILINE) ==
         ["1" if completed else "0"], "completed-call count differs")
    need(read_json(CELL / "AUDIT.json")["saved_segments"] == len(ranges),
         "runner receipt differs from original range count")
    return {
        "status": "numeric_period_completed" if completed else
                  "first_numeric_refusal" if ranges else "first_numeric_refusal_uncompared",
        "scope": "one high-corner binary cell, first held-control period only",
        "initial_box_equals_prior": True,
        "one_rpc_equals_prior": True,
        "first_408_byte_range_record_equals_prior": True if ranges else None,
        "saved_h_0p01_segments": len(ranges),
        "saved_numeric_time_s": len(ranges) * 0.01,
        "target_control_period_s": 0.1,
        "native_status": status,
        "outer_exit_code": result["exit_code"],
        "first_refused_step": len(ranges) + 1 if first_refusal else None,
        "picard_group_count": len(groups),
        "picard_rows": len(picard),
        "picard_noninclusion_axes_by_step": refused_axes,
        "first_refusal_intervals": [
            {"axis": row["axis"], "old": [row["old_lo"], row["old_hi"]],
             "candidate": [row["new_lo"], row["new_hi"]]}
            for row in groups[-1] if row["subset"] == 0
        ] if first_refusal else [],
        "all_saved_tubes_inside_property_box": all(safe_steps) if ranges else None,
        "saved_prefix_property": None if not ranges else
                                 "inside" if all(safe_steps) else "unknown",
        "all_saved_cos_theta_positive": all(domain_steps) if ranges else None,
        "minimum_saved_cos_theta_lower": min(cosine_lowers) if ranges else None,
        "ten_step_period_generated": completed,
        "limits": ["No T=2 completion or full-initial-box safety proof",
                   "No independent NN/CROWN soundness or Flowstar floating-point proof"],
    }


def main():
    try:
        outcome = audit()
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        outcome = {"status": "receipt_or_infrastructure_unknown", "error": str(exc),
                   "ten_step_period_generated": False}
    (HERE / "INDEPENDENT_AUDIT.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["status"] != "receipt_or_infrastructure_unknown" else 2


if __name__ == "__main__":
    raise SystemExit(main())
