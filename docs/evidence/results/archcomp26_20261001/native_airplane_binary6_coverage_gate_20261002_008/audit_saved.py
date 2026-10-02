#!/usr/bin/env python3
"""Interpret the saved upper Airplane cell without starting a solver."""

import json
import math
from pathlib import Path
import re
import struct

HERE = Path(__file__).parent
OLD = HERE.parent / "native_airplane_first_reject_trace_20261002_005/airplane_first_reject_trace.cpp"
NEW = HERE / "airplane_binary6_coverage_gate.cpp"
RUN = HERE / "cells/111111"


def main():
    source = OLD.read_text()
    old = """             init_u(0, 1), init_v(0, 1), init_w(0, 1),
             init_phi(0, 1), init_theta(0, 1), init_psi(0, 1),"""
    new = """             init_u(lo(0), hi(0)), init_v(lo(1), hi(1)), init_w(lo(2), hi(2)),
             init_phi(lo(3), hi(3)), init_theta(lo(4), hi(4)), init_psi(lo(5), hi(5)),"""
    setup = """    const char *bits_env = std::getenv("AIRPLANE_CELL_BITS");
    if (!bits_env) throw std::runtime_error("AIRPLANE_CELL_BITS required");
    const std::string bits(bits_env);
    if (bits.size() != 6 || bits.find_first_not_of("01") != std::string::npos)
        throw std::runtime_error("invalid AIRPLANE_CELL_BITS");
    auto lo = [&](size_t i) { return bits[i] == '1' ? 0.5 : 0.0; };
    auto hi = [&](size_t i) { return bits[i] == '1' ? 1.0 : 0.5; };

"""
    assert source.count(old) == 1 and source.count("    // Initial set\n") == 1
    source = source.replace(old, new).replace("#include <stdexcept>\n",
                                             "#include <stdexcept>\n#include <string>\n", 1)
    source = source.replace("    // Initial set\n", setup + "    // Initial set\n")
    assert source == NEW.read_text()

    plan = json.loads((HERE / "SPLIT_PLAN.json").read_text())
    cell = next(item for item in plan["boxes"] if item["bits"] == "111111")
    ledger = json.loads((RUN / "initial_boxes.json").read_text())
    assert ledger == [cell["box"]]
    rpc = [json.loads(line) for line in (RUN / "controller_rpc.jsonl").read_text().splitlines()]
    assert len(rpc) == 1
    assert rpc[0]["input_lower"] == [v[0] for v in ledger[0][:12]]
    assert rpc[0]["input_upper"] == [v[1] for v in ledger[0][:12]]
    native = (RUN / "native.log").read_text()
    rows = [line for line in native.splitlines() if line.startswith("AIRPLANE_PICARD\t")]
    assert len(rows) == 19 and all("\tsubset=1" in line for line in rows)
    assert re.search(r"PERIOD 0 FLOWPIPES 1 STATUS 3 BOX_SAFE 0", native)
    assert "FLOWPIPE_SEGMENTS 1\nUNKNOWN" in native
    result = json.loads((RUN / "RESULT.json").read_text())
    assert result["status"] == "failed" and result["exit_code"] == 2

    raw = (RUN / "ranges.bin").read_bytes()
    assert len(raw) == 24 + 12 * 32
    lane, step, h, *values = struct.unpack("<QQd48d", raw)
    assert (lane, step, h) == (0, 1, 0.01)
    bounds = [values[i * 4:i * 4 + 4] for i in range(12)]
    assert all(all(math.isfinite(v) for v in row) and
               row[0] <= row[1] and row[2] <= row[3] for row in bounds)
    safety = (RUN / "safety.tsv").read_text().splitlines()
    assert len(safety) == 2
    fields = safety[1].split("\t")
    assert fields[:3] == ["0", "1", "1"]
    for index, offset in zip((1, 6, 7, 8), (3, 5, 7, 9)):
        assert tuple(bounds[index][:2]) == (float(fields[offset]), float(fields[offset + 1]))
    exceeded = {name: bounds[index][1] for name, index in
                (("phi", 6), ("theta", 7), ("psi", 8)) if bounds[index][1] > 1}
    assert set(exceeded) == {"phi", "theta", "psi"}
    assert float(fields[11]) > 0
    gate = json.loads((HERE / "COVERAGE_GATE.json").read_text())
    assert gate["status"] == "first_refusal" and [x["bits"] for x in gate["attempted"]] == ["111111"]
    assert [p.name for p in (HERE / "cells").iterdir()] == ["111111"]

    audit = {"tested_bits": "111111", "numerical_first_step_accepted": True,
             "native_status": 3, "native_status_name": "COMPLETED_UNKNOWN",
             "first_picard_inclusion": "19/19", "saved_segments": 1,
             "tube_upper_exceeds_safe_limit": exceeded,
             "cos_theta_lower": float(fields[11]), "property_result": "UNKNOWN",
             "remaining_partition_cells_not_run": 62,
             "actual_unsafe_trajectory_shown": False,
             "complete_full_box_numerical_or_property_result": False}
    (HERE / "INDEPENDENT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
