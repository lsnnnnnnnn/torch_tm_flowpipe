#!/usr/bin/env python3
"""Audit saved Airplane first-step receipts; never start a solver or RPC."""

import json
import math
import re
import struct
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_LOW = HERE.parent / "native_airplane_binary6_firststep_20261002_007/run"
OLD_HIGH = HERE.parent / "native_airplane_binary6_coverage_gate_20261002_008/cells/111111"
AXES = ("u", "v", "w", "phi", "theta", "psi")
PHYSICAL = (3, 4, 5, 6, 7, 8)
RAW_FILES = {
    "AUDIT.json", "RESULT.json", "START.json", "controller_rpc.jsonl",
    "initial_boxes.json", "launcher.stderr.log", "launcher.stdout.log",
    "ledger.log", "ledger_checks.tsv", "listeners_at_start.txt", "native.log",
    "native.pid", "ranges.bin", "safety.tsv", "server.log", "server.pid",
}
SAFETY = ((1, 3), (6, 5), (7, 7), (8, 9))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def expected_box(bits):
    box = [[0, 0] for _ in range(19)]
    for index, digit in zip(PHYSICAL, bits):
        half = int(digit) / 2
        box[index] = [half, half + 0.5]
    return box


def audit_cell(path, bits, new):
    label = f"{path.name}:{bits}"
    if new:
        files = {item.name for item in path.iterdir() if item.is_file()}
        need(files == RAW_FILES, f"{label}: raw file set differs: {sorted(files ^ RAW_FILES)}")

    start, result = read_json(path / "START.json"), read_json(path / "RESULT.json")
    env = start["selected_environment"]
    need(start["command"][-1].endswith("/run_archcomp26_airplane_native_fullbox_pair.sh"),
         f"{label}: launcher changed")
    need(env["AIRPLANE_RUN_DIR"].endswith(str(path.relative_to(HERE.parent)).replace("\\", "/")),
         f"{label}: run identity changed")
    need((env["CUDA_VISIBLE_DEVICES"], env["AIRPLANE_CPUSET"], env["OMP_NUM_THREADS"]) ==
         ("2", "10-13", "1"), f"{label}: resource contract changed")
    if new:
        need(env["AIRPLANE_CELL_BITS"] == bits and result["bits"] == bits,
             f"{label}: selected bits mismatch")
    need(result["timed_out"] is False and math.isfinite(result["wall_s"]) and
         result["wall_s"] > 0, f"{label}: timeout or invalid wall time")

    ledger = read_json(path / "initial_boxes.json")
    need(ledger == [expected_box(bits)], f"{label}: actual initial box mismatch")
    checks = (path / "ledger_checks.tsv").read_text().splitlines()
    need(checks == [(path / "safety.tsv").read_text().splitlines()[0]] and
         not (path / "ledger.log").read_text().strip(),
         f"{label}: unexpected later ledger check or ledger error")
    rpc_lines = (path / "controller_rpc.jsonl").read_text().splitlines()
    need(len(rpc_lines) == 1, f"{label}: expected one controller RPC")
    rpc = json.loads(rpc_lines[0])
    need(rpc["input_lower"] == [v[0] for v in ledger[0][:12]] and
         rpc["input_upper"] == [v[1] for v in ledger[0][:12]],
         f"{label}: controller input differs from actual initial box")
    output = rpc["response"]
    need(len(output["T"]) == 6 and all(len(row) == 12 for row in output["T"]) and
         all(math.isfinite(v) for row in output["T"] for v in row) and
         len(output["u_min"]) == len(output["u_max"]) == 6 and
         all(math.isfinite(lo) and math.isfinite(hi) and lo <= hi
             for lo, hi in zip(output["u_min"], output["u_max"])),
         f"{label}: malformed saved controller response")
    server = (path / "server.log").read_text()
    need(server.count('"POST / HTTP/1.1" 200 ') == 1, f"{label}: HTTP receipt mismatch")

    native = (path / "native.log").read_text()
    picard = [line.split("\t") for line in native.splitlines()
              if line.startswith("AIRPLANE_PICARD\t")]
    need(len(picard) == 19 and [int(row[1]) for row in picard] == list(range(19)),
         f"{label}: first Picard coordinate coverage differs")
    for row in picard:
        fields = dict(item.split("=", 1) for item in row[2:])
        need(fields["subset"] == "1", f"{label}: Picard subset refusal")
        old_lo, old_hi = float(fields["old_lo"]), float(fields["old_hi"])
        new_lo, new_hi = float(fields["new_lo"]), float(fields["new_hi"])
        need((old_lo, old_hi) == (-0.01, 0.01) and
             all(math.isfinite(v) for v in (old_lo, old_hi, new_lo, new_hi)) and
             old_lo <= new_lo <= new_hi <= old_hi,
             f"{label}: saved Picard interval inconsistent")
    period = re.findall(r"^PERIOD (\d+) FLOWPIPES (\d+) STATUS (\d+) BOX_SAFE (\d+)$",
                        native, re.MULTILINE)
    need(len(period) == 1 and period[0][:2] == ("0", "1") and
         period[0][2] in ("2", "3"), f"{label}: no accepted first segment")
    status, box_safe = map(int, period[0][2:])
    need(re.findall(r"^FLOWPIPE_SEGMENTS (\d+)$", native, re.MULTILINE) == ["1"],
         f"{label}: saved segment count differs")

    raw = (path / "ranges.bin").read_bytes()
    need(len(raw) == 408, f"{label}: range record is not 408 bytes")
    lane, step, h, *values = struct.unpack("<QQd48d", raw)
    need((lane, step) == (0, 1) and math.isfinite(h) and
         math.isclose(h, 0.01, rel_tol=0, abs_tol=1e-12),
         f"{label}: range header differs")
    ranges = [values[4 * i:4 * i + 4] for i in range(12)]
    need(all(all(math.isfinite(v) for v in row) and
             row[0] <= row[2] <= row[3] <= row[1] for row in ranges),
         f"{label}: range or endpoint interval invalid")

    safety = (path / "safety.tsv").read_text().splitlines()
    need(len(safety) == 2, f"{label}: expected one saved safety row")
    fields = safety[1].split("\t")
    need(len(fields) == 13 and fields[:3] == ["0", "1", "1"],
         f"{label}: safety step identity differs")
    for index, offset in SAFETY:
        need((float(fields[offset]), float(fields[offset + 1])) == tuple(ranges[index][:2]),
             f"{label}: safety row differs from raw range coordinate {index}")
    cosine = tuple(map(float, fields[11:13]))
    need(all(math.isfinite(v) for v in cosine) and cosine[0] <= cosine[1],
         f"{label}: cosine interval invalid")
    independently_safe = (all(-1 <= ranges[index][0] and ranges[index][1] <= 1
                              for index, _ in SAFETY) and cosine[0] > 0)
    need(box_safe == int(independently_safe), f"{label}: BOX_SAFE differs from saved tube")
    if status == 2:
        need(independently_safe, f"{label}: author safe but saved tube is not safe")
    property_status = "one_step_safe" if status == 2 else "unknown"
    exit_code = 0 if status == 2 and independently_safe else 2
    need((result["exit_code"], result["status"]) ==
         (exit_code, "completed" if exit_code == 0 else "failed"),
         f"{label}: outer result does not match saved native status")
    terminal = "ONE_STEP_NUMERIC_ACCEPTED_ONLY" if exit_code == 0 else "UNKNOWN"
    need(terminal in native and
         re.findall(r"^DIAGNOSTIC_COMPLETED_CALLS (\d+)/1$", native, re.MULTILINE) ==
         ["1" if exit_code == 0 else "0"],
         f"{label}: native terminal receipt differs")
    if new:
        audit = read_json(path / "AUDIT.json")
        need(audit["bits"] == bits and audit["numerical_first_step_accepted"] is True and
             audit["property_status"] == property_status and
             audit["native_status"] == str(status) and
             audit["independent_box_safe"] == str(box_safe),
             f"{label}: runner audit differs from independent raw classification")
    return {"bits": bits, "native_status": status, "property_status": property_status,
            "first_picard_inclusions": len(picard), "saved_numeric_segments": 1,
            "saved_range_bytes": len(raw), "cos_theta_lower": cosine[0]}


def main():
    expected = {"".join(bits) for bits in product("01", repeat=6)}
    old = {"000000", "111111"}
    new = expected - old
    cells = HERE / "cells"
    actual = {item.name for item in cells.iterdir() if item.is_dir()}
    need(actual == new, f"62-cell complement differs: missing={sorted(new - actual)}, extra={sorted(actual - new)}")
    need(sum(1 for folder in cells.iterdir() for item in folder.iterdir() if item.is_file()) == 992,
         "62 x 16 raw files have not all been mirrored")
    plan = read_json(HERE / "SPLIT_PLAN.json")
    planned = plan["boxes"]
    need(plan["coordinates"] == list(AXES) and plan["indices"] == list(PHYSICAL) and
         len(planned) == 64 and {row["bits"] for row in planned} == expected and
         all(row["box"] == expected_box(row["bits"]) for row in planned),
         "saved partition differs from exact six-axis closed binary cover")
    need(read_json(HERE / "BUILD.json")["returncode"] == 0,
         "isolated native build did not succeed")
    new_source = HERE / "build/archcomp/airplane/airplane_binary6_numeric_cover.cpp"
    prior_source = HERE.parent / "native_airplane_binary6_coverage_gate_20261002_008/airplane_binary6_coverage_gate.cpp"
    need(new_source.read_text() == prior_source.read_text(),
         "native source differs from prior generic binary-cover entry")
    gate = read_json(HERE / "COVERAGE_GATE.json")
    queue = sorted(new, key=lambda bits: (-bits.count("1"), bits))
    need(gate["queue"] == queue and gate["status"] ==
         "all_64_first_steps_numerically_accepted" and
         [row["bits"] for row in gate["attempted"]] == queue,
         "run queue or completion receipt differs")

    prior = [audit_cell(OLD_LOW, "000000", False),
             audit_cell(OLD_HIGH, "111111", False)]
    rows = [audit_cell(cells / bits, bits, True) for bits in queue]
    for recorded, row in zip(gate["attempted"], rows):
        need(recorded["numerical_first_step_accepted"] is True and
             recorded["property_status"] == row["property_status"] and
             recorded["native_status"] == str(row["native_status"]),
             f"{row['bits']}: coverage receipt differs from raw evidence")
    all_rows = prior + rows
    need({row["bits"] for row in all_rows} == expected,
         "64-cell numerical first-step cover incomplete")
    summary = {
        "status": "saved_first_step_numeric_cover_audited",
        "new_raw_cells": len(rows), "new_raw_files": 992,
        "prior_raw_cells": len(prior), "full_binary_cover_cells": len(all_rows),
        "same_generic_cpp_as_prior_upper_cell": True,
        "all_cells_one_saved_h_0p01_numeric_step": True,
        "all_cells_first_picard_inclusions": sum(row["first_picard_inclusions"] for row in all_rows),
        "one_step_safe_cells": sum(row["property_status"] == "one_step_safe" for row in all_rows),
        "property_unknown_cells": sum(row["property_status"] == "unknown" for row in all_rows),
        "prior": prior, "new_cells": rows,
        "scope": "saved-log consistency of one numerical h=0.01 step in each binary-cover cell",
        "limits": ["No complete 0.1 s control period or T=2 horizon",
                   "No full-box safety proof or actual unsafe trajectory",
                   "No independent CROWN/NN enclosure or Flowstar floating-point proof"],
    }
    (HERE / "INDEPENDENT_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "status", "new_raw_cells", "full_binary_cover_cells", "one_step_safe_cells",
        "property_unknown_cells")}, indent=2))


if __name__ == "__main__":
    main()
