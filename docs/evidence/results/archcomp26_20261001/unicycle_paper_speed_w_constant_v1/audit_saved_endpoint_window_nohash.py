#!/usr/bin/env python3
"""Read four existing Unicycle runs for a common full-box target endpoint; no solver run."""

import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "native_unicycle_paper_speed_full50_001"
TARGET = ((-0.6, 0.6), (-0.2, 0.2), (-0.06, 0.06), (-0.3, 0.3))
ROW = struct.Struct("<QQd16d")
STEPS = 500
H = 0.02


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def valid(tube, endpoint):
    if len(tube) != len(endpoint):
        raise ValueError("tube/endpoint dimensions disagree")
    for (lo, hi), (elo, ehi) in zip(tube, endpoint):
        if not all(math.isfinite(x) for x in (lo, hi, elo, ehi)) or not lo <= elo <= ehi <= hi:
            raise ValueError("invalid saved interval or endpoint outside same-step tube")


def author(method):
    run = HERE / f"{method}_full50_001"
    result = read(run / "payload/RESULT.json")
    if (result.get("status") != "completed" or result.get("accepted_substeps") != STEPS
            or result.get("profile") != "unicycle-paper-speed-w-constant-v1"):
        raise ValueError(f"{method}: full-run outcome changed")
    path = run / "payload/ranges.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != STEPS:
        raise ValueError(f"{method}: expected 500 saved substeps")
    data = []
    for i, row in enumerate(rows, 1):
        tube, endpoint = row["tube"], row["endpoint"]
        if (row.get("substep") != i or row.get("accepted") is not True
                or row.get("solver_status_code") != 0 or len(tube) != 8):
            raise ValueError(f"{method}: invalid accepted step {i}")
        valid(tube, endpoint)
        data.append((endpoint[:4], tube[:4]))
    if result.get("terminal_physical_endpoint") != data[-1][0]:
        raise ValueError(f"{method}: final saved endpoint differs from RESULT")
    return data, {"ranges": str(path.relative_to(HERE.parent)), "bytes": path.stat().st_size,
                  "result": str((run / "payload/RESULT.json").relative_to(HERE.parent)),
                  "acceptance": "500/500 accepted observations"}


def native():
    path = NATIVE / "ranges.bin"
    raw = path.read_bytes()
    result, scan = read(NATIVE / "RESULT.json"), read(NATIVE / "RANGE_SCAN.json")
    if (len(raw) != STEPS * ROW.size or result.get("status") != "completed"
            or result.get("exit_code") != 0 or scan.get("complete_grid") is not True
            or scan.get("records") != STEPS or scan.get("boxes") != 1):
        raise ValueError("native full-run outcome/grid changed")
    data = []
    for i, (lane, step, h, *flat) in enumerate(ROW.iter_unpack(raw), 1):
        if lane != 0 or step != i or not math.isclose(h, H, abs_tol=1e-12):
            raise ValueError(f"native: invalid lane/step/h at {i}")
        tube = [[flat[4*j], flat[4*j+1]] for j in range(4)]
        endpoint = [[flat[4*j+2], flat[4*j+3]] for j in range(4)]
        valid(tube, endpoint)
        data.append((endpoint, tube))
    if scan.get("final_endpoint_union") != data[-1][0]:
        raise ValueError("native: final saved endpoint differs from prior scan")
    return data, {"ranges": str(path.relative_to(HERE.parent)), "bytes": path.stat().st_size,
                  "result": str((NATIVE / "RESULT.json").relative_to(HERE.parent)),
                  "acceptance": "500-record one-box grid; native binary has no accepted field"}


def spans(steps):
    answer = []
    for step in steps:
        if not answer or step != answer[-1][1] + 1:
            answer.append([step, step])
        else:
            answer[-1][1] = step
    return answer


def audit(method, data, source):
    endpoint_steps, tube_steps = [], []
    coordinate_steps = [[] for _ in range(4)]
    for i, (endpoint, tube) in enumerate(data, 1):
        for j in range(4):
            if TARGET[j][0] <= endpoint[j][0] <= endpoint[j][1] <= TARGET[j][1]:
                coordinate_steps[j].append(i)
        if all(TARGET[j][0] <= endpoint[j][0] <= endpoint[j][1] <= TARGET[j][1]
               for j in range(4)):
            endpoint_steps.append(i)
        if all(TARGET[j][0] <= tube[j][0] <= tube[j][1] <= TARGET[j][1]
               for j in range(4)):
            tube_steps.append(i)
    return {"method": method, "source": source,
            "saved_endpoint_target_step_spans": spans(endpoint_steps),
            "first_saved_endpoint_target_t_s": endpoint_steps[0] * H if endpoint_steps else None,
            "saved_whole_step_tube_target_step_spans": spans(tube_steps),
            "first_saved_whole_step_tube_target_interval_s":
                [(tube_steps[0]-1)*H, tube_steps[0]*H] if tube_steps else None,
            "per_coordinate_endpoint_target_step_spans":
                {f"x{j+1}": spans(v) for j, v in enumerate(coordinate_steps)},
            "last_saved_endpoint": data[-1][0]}


def main():
    output = {"schema": "unicycle-paper-speed-constant-w-saved-endpoint-window-audit-nohash-v1",
              "contract": "2026 paper RHS; w only in speed derivative and constant along each trajectory",
              "target_physical_box": TARGET, "steps": STEPS, "step_s": H,
              "test": "all four physical coordinates of one full-initial-box saved endpoint lie in target at the same accepted step; whole-step tube checked separately",
              "methods": []}
    for method in ("huan", "xiangru", "p3"):
        data, source = author(method)
        output["methods"].append(audit(method, data, source))
    data, source = native()
    output["methods"].append(audit("flowstar_native", data, source))
    output["qualification"] = (
        "No observed common endpoint inclusion is not proof of unreachability. "
        "Saved axis boxes and adjacent run outcomes are not an independent end-to-end floating-point NNCS certificate.")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
