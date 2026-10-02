#!/usr/bin/env python3
"""Independent read-only audit of completed saved QUAD first-step ledgers."""

import csv
from fractions import Fraction as F
import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = BASE / "native_quad_allbox_independent_plant_firststep_fixed_decimal_20261003_005"
ADAPTIVE = BASE / "native_quad_allbox_independent_plant_adaptive_bootstrap_20261003_006"
SAVED = BASE / "native_quad_allbox_firststep_recenter_gate_20261003_003/run"
H, G, MASS, INERTIA = F(5, 1000), F(981, 100), F(14, 10), F(54, 1000)
V, ANGLE = F(1), F(1, 100)
MARGIN = F(1, 10**9)
N = 1024


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def f32(value):
    return F.from_float(struct.unpack("<f", struct.pack("<f", float(value)))[0])


def pair(value):
    result = tuple(F(item) for item in value)
    need(len(result) == 2 and result[0] <= result[1], "invalid rational interval pair")
    return result


def ceil_abs(bounds):
    return max(abs(bounds[0]), abs(bounds[1]))


def subset(inner, outer):
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def linear_range(coefficients, box, center, radius):
    lower, upper = center - radius, center + radius
    for coefficient, (lo, hi) in zip(coefficients, box):
        lower += min(coefficient * lo, coefficient * hi)
        upper += max(coefficient * lo, coefficient * hi)
    return lower, upper


def csv_index(path, fields):
    index = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(int(row[field]) for field in fields)
            need(key not in index, f"duplicate saved CSV key: {path.name} {key}")
            index[key] = row
    return index


def saved_bound(text_lo, text_hi):
    lo, hi = float(text_lo), float(text_hi)
    need(math.isfinite(lo) and math.isfinite(hi) and lo <= hi,
         "nonfinite or reversed saved binary64 bound")
    outward = (F.from_float(math.nextafter(lo, -math.inf)),
               F.from_float(math.nextafter(hi, math.inf)))
    return outward, (F.from_float(lo), F.from_float(hi))


def load_saved():
    rpc = json.loads((SAVED / "rpc.json").read_text())
    params, coeff = rpc["params"], rpc["coefficients"]
    need(len(params["input_lb"]) == len(params["input_ub"]) == N * 12,
         "saved RPC state grid incomplete")
    need(all(len(coeff[name]) == N for name in ("T", "u_min", "u_max")),
         "saved RPC control grid incomplete")
    source = csv_index(SAVED / "source_boxes.csv", ("lane", "coord"))
    octagon = csv_index(SAVED / "octagon.csv", ("lane", "step", "view", "form"))
    axes = csv_index(SAVED / "terminal_axes.csv", ("lane", "coord"))
    need(len(source) == N * 12 and len(octagon) == N * 8 and len(axes) == N * 12,
         "saved source or observer CSV grid incomplete")
    return params, coeff, source, octagon, axes


def contract(lane, saved):
    params, coeff, source, _, _ = saved
    box = [(F.from_float(float(params["input_lb"][lane*12 + i])),
            F.from_float(float(params["input_ub"][lane*12 + i])))
           for i in range(12)]
    need(all(lo <= hi for lo, hi in box), f"reversed RPC box lane {lane}")
    for i, value in enumerate(box, 1):
        row = source[lane, i]
        physical = (F.from_float(float(row["lo"])), F.from_float(float(row["hi"])))
        need(subset(physical, value), f"RPC omits source lane {lane} coord {i}")
    need(all(box[i] == (0, 0) for i in range(6, 12)),
         f"nonzero initial angles/rates lane {lane}")
    matrix = [[f32(value) for value in row] for row in coeff["T"][lane]]
    need(len(matrix) == 3 and all(len(row) == 12 for row in matrix),
         f"controller matrix shape lane {lane}")
    controls = []
    for j in range(3):
        lo, hi = f32(coeff["u_min"][lane][j]), f32(coeff["u_max"][lane][j])
        need(lo <= hi, f"reversed controller residual lane {lane} output {j}")
        center, radius = (lo + hi) / 2, (hi - lo) / 2
        need(F.from_float(float(center)) == center and F.from_float(float(radius)) == radius,
             f"binary64 center/radius inexact lane {lane} output {j}")
        controls.append(linear_range(matrix[j], box, center, radius))
    return box, controls


def bootstrap(box, controls, adaptive):
    raw10 = H * ceil_abs(controls[1]) / INERTIA
    raw11 = H * ceil_abs(controls[2]) / INERTIA
    rate10, rate11 = (raw10 + MARGIN, raw11 + MARGIN) if adaptive else (F(3, 2000), F(1, 1000))
    a4, a5, a6 = (ceil_abs(box[i]) for i in (3, 4, 5))
    u1_max = ceil_abs(controls[0])
    b4 = rate11 * V + G * ANGLE
    b5 = rate10 * V + G * ANGLE
    other = rate11 * V + rate10 * V + G * ANGLE**2
    b6 = other + u1_max / MASS
    b7 = rate10 + ANGLE**2 * rate11 / (1 - ANGLE**2 / 2)
    checks = {
        "x10_rate": (raw10, rate10),
        "x11_rate": (raw11, rate11),
        "x4": (a4 + H * b4, V),
        "x5": (a5 + H * b5, V),
        "x6": (a6 + H * b6, V),
        "x7": (H * b7, ANGLE),
        "x8": (H * rate11, ANGLE),
    }
    need(all(left < right for left, right in checks.values()),
         "one of seven first-exit bootstrap inequalities is not strict")
    return checks, (raw10, rate10, raw11, rate11, MARGIN)


def expected_observer(lane, kind, index, saved):
    _, _, _, octagon, axes = saved
    if kind in ("tube", "endpoint"):
        row = octagon[lane, 1, 0 if kind == "tube" else 1, index]
        return saved_bound(row["lo"], row["hi"])
    row = axes[lane, index]
    return saved_bound(row["composed_lo"], row["composed_hi"])


def audit_record(record, lane, adaptive, saved):
    need(record["lane"] == lane, f"nonconsecutive ledger lane {lane}")
    need(record["strict_picard_substeps"] == 1000,
         f"Picard step count changed lane {lane}")
    box, controls = contract(lane, saved)
    need([pair(value) for value in record["control_hull_exact"]] == controls,
         f"control hull differs from saved RPC lane {lane}")
    checks, rates = bootstrap(box, controls, adaptive)
    recorded = record["strict_bootstrap"]
    expected_keys = set(checks) if adaptive else set(checks) - {"x10_rate", "x11_rate"}
    need(set(recorded) == expected_keys, f"bootstrap columns differ lane {lane}")
    for name in expected_keys:
        need(pair(recorded[name]) == checks[name],
             f"bootstrap rational bound differs lane {lane}: {name}")
    if adaptive:
        rate_record = record["adaptive_rates_exact"]
        need(tuple(F(rate_record[name]) for name in (
            "raw_x10", "chosen_x10", "raw_x11", "chosen_x11", "strict_margin")) == rates,
            f"adaptive rate record differs lane {lane}")
    rows = record["comparisons"]
    need(len(rows) == 20, f"not 20 physical comparisons lane {lane}")
    keys = set()
    unexpanded_inside = 0
    for row in rows:
        kind, index = row["kind"], row["coord_or_form"]
        key = kind, index
        need(key not in keys, f"duplicate physical comparison lane {lane} {key}")
        keys.add(key)
        need(row["method"] == ("algebraic" if kind == "terminal_composed" and index in (6, 10, 11)
                                else "directed_picard"),
             f"unexpected physical comparison method lane {lane} {key}")
        bound = pair(row["bound_exact"])
        outward, raw = expected_observer(lane, kind, index, saved)
        need(pair(row["saved_outward_exact"]) == outward,
             f"saved observer mismatch lane {lane} {key}")
        inside, raw_inside = subset(bound, outward), subset(bound, raw)
        need(row["inside_outward"] == inside and row["inside_unexpanded_binary64"] == raw_inside,
             f"comparison Boolean mismatch lane {lane} {key}")
        need(inside, f"independent physical bound outside saved observer lane {lane} {key}")
        unexpanded_inside += raw_inside
    expected = ({("tube", i) for i in range(4)} |
                {("endpoint", i) for i in range(4)} |
                {("terminal_composed", i) for i in range(1, 13)})
    need(keys == expected, f"physical comparison identities incomplete lane {lane}")
    need(record["physical_outward_comparisons_inside"] == 20 and
         record["physical_unexpanded_binary64_comparisons_inside"] == unexpanded_inside,
         f"comparison totals inconsistent lane {lane}")
    return unexpanded_inside


def audit_segment(directory, start, count, adaptive, saved, early_stop=False):
    result = json.loads((directory / "RESULT.json").read_text())
    need(Path(result["saved_run"]).resolve() == SAVED.resolve() and
         result["physical_composed_only"] is True and
         "first h=0.005 step only" in result["scope"],
         f"segment source or first-step scope changed: {directory}")
    need(result["start_lane"] == start and result["checked_boxes"] == count and
         result["next_unchecked_lane"] == start + count,
         f"segment result cardinality mismatch: {directory}")
    need(result["requested_boxes"] == (N - start if early_stop else count),
         f"segment request count mismatch: {directory}")
    if early_stop:
        need(result["first_issue"]["lane"] == start + count and
             result["first_issue"]["kind"] == "METHOD_UNDECIDED",
             "old fixed-bootstrap first stop changed")
    else:
        need(result["first_issue"] is None,
             f"segment has a first issue: {directory}")
    checkpoint = directory / "CHECKPOINT.json"
    if checkpoint.exists():
        cp = json.loads(checkpoint.read_text())
        need(cp["start_lane"] == start and cp["checked_boxes"] == count and
             cp["last_completed_lane"] == start + count - 1 and
             cp["next_unchecked_lane"] == start + count,
             f"checkpoint differs from result: {directory}")
    else:
        need(start == 0 and count == 1,
             f"missing checkpoint: {directory}")
    checked = 0
    raw_inside = 0
    with (directory / "BOX_AUDIT.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            need(checked < count, f"extra ledger row: {directory}")
            raw_inside += audit_record(record, start + checked, adaptive, saved)
            checked += 1
    need(checked == count, f"missing ledger rows: {directory}")
    return {"directory": str(directory), "start_lane": start,
            "checked_boxes": checked, "physical_outward_inside": checked * 20,
            "physical_unexpanded_binary64_inside": raw_inside,
            "bootstrap_mode": "adaptive" if adaptive else "fixed_firstbox"}


def main():
    # Do not scan a ledger while its writer is still running.
    need((ADAPTIVE / "remainder_8_1023/RESULT.json").exists(),
         "adaptive full sweep has no terminal result yet")
    for gate in (OLD, ADAPTIVE):
        self_check = json.loads((gate / "METHOD_SELF_CHECK.json").read_text())
        need(self_check["passed"] is True,
             f"Decimal primitive self-check rejected: {gate}")
    saved = load_saved()
    segments = [
        audit_segment(OLD / "probe_lane0", 0, 1, False, saved),
        audit_segment(OLD / "remainder_1_1023", 1, 6, False, saved, early_stop=True),
        audit_segment(ADAPTIVE / "probe_lane7", 7, 1, True, saved),
        audit_segment(ADAPTIVE / "remainder_8_1023", 8, 1016, True, saved),
    ]
    need(sum(item["checked_boxes"] for item in segments) == N,
         "not all 1024 source boxes are present")
    result = {
        "schema": "native-quad-allbox-firststep-independent-ledger-audit-v1",
        "status": "saved_ledgers_consistent",
        "segments": segments,
        "source_boxes_checked": N,
        "independent_control_hulls_reconstructed": N * 3,
        "strict_bootstrap_inequalities_recomputed": N * 7,
        "recorded_bootstrap_inequalities_checked": 7 * 5 + (N - 7) * 7,
        "picard_substeps_per_box": 1000,
        "saved_physical_comparisons_inside": N * 20,
        "scope": "saved source-defined 1024 QUAD first-call boxes, first paper-ODE h=0.005 plant step under each saved affine-plus-residual control hull",
        "limits": "No independent CROWN/NN proof, Flow* parser/runtime certificate, other small steps, later controls, T=5 reach-and-remain, or native production promotion",
        "native_production_gate": "CLOSED",
    }
    (HERE / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_boxes": N, "physical_inside": N * 20,
                      "bootstrap": N * 7, "status": result["status"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        (HERE / "BLOCKED.json").write_text(json.dumps({
            "status": "stopped_on_first_audit_error",
            "reason": f"{type(exc).__name__}: {exc}",
        }, indent=2) + "\n")
        raise
