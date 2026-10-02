#!/usr/bin/env python3
"""Directed plant inclusion gate on saved QUAD first-step RPC and ranges."""

import argparse
import csv
from decimal import Decimal as D
from fractions import Fraction as F
import json
import math
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "tools"))
import interval_fixed_nohash as interval  # noqa: E402
import check_native_quad_algebraic_gate_nohash as algebra  # noqa: E402

SAVED = HERE.parent / "native_quad_allbox_firststep_recenter_gate_20261003_003/run"
N = 1024


def read_rows(name, keys):
    with (SAVED / name).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = tuple(int(row[field]) for field in keys)
        if key in result:
            raise ValueError(f"duplicate {name} key {key}")
        result[key] = row
    return result


def load_saved():
    rpc = json.loads((SAVED / "rpc.json").read_text())
    params, coeff = rpc["params"], rpc["coefficients"]
    if (len(params["input_lb"]) != N * 12 or len(params["input_ub"]) != N * 12 or
            any(len(coeff[key]) != N for key in ("T", "u_min", "u_max"))):
        raise ValueError("saved RPC batch cardinality mismatch")
    source = read_rows("source_boxes.csv", ("lane", "coord"))
    octagon = read_rows("octagon.csv", ("lane", "step", "view", "form"))
    terminal = read_rows("terminal_axes.csv", ("lane", "coord"))
    if len(source) != N * 12 or len(octagon) != N * 8 or len(terminal) != N * 12:
        raise ValueError("incomplete saved source or observer grid")
    return params, coeff, source, octagon, terminal


def contract(lane, params, coeff, source):
    pairs = list(zip(params["input_lb"][lane * 12:(lane + 1) * 12],
                     params["input_ub"][lane * 12:(lane + 1) * 12]))
    box = [(F.from_float(float(lo)), F.from_float(float(hi))) for lo, hi in pairs]
    if len(box) != 12 or any(lo > hi for lo, hi in box):
        raise ValueError("invalid RPC input box")
    for coord, pair in enumerate(box, 1):
        row = source[lane, coord]
        physical = (F.from_float(float(row["lo"])), F.from_float(float(row["hi"])))
        if not algebra.subset(physical, pair):
            raise ValueError(f"RPC input omits source at lane {lane}, coord {coord}")
    if any(box[i] != (0, 0) for i in range(6, 12)):
        raise ValueError("expected x7 through x12 initially zero")

    matrix = [[algebra.binary32(value) for value in row] for row in coeff["T"][lane]]
    if len(matrix) != 3 or any(len(row) != 12 for row in matrix):
        raise ValueError("invalid CROWN matrix shape")
    controls = []
    dec_controls = []
    dec_box = [interval.I(D.from_float(float(lo)), D.from_float(float(hi)))
               for lo, hi in pairs]
    for output in range(3):
        low = algebra.binary32(coeff["u_min"][lane][output])
        high = algebra.binary32(coeff["u_max"][lane][output])
        if low > high:
            raise ValueError("reversed CROWN intercept interval")
        center, radius = (low + high) / 2, (high - low) / 2
        if F.from_float(float(center)) != center or F.from_float(float(radius)) != radius:
            raise ValueError("center or radius not exactly binary64")
        controls.append((center, radius))
        radius_decimal = D.from_float(float(radius))
        current = interval.I(D.from_float(float(center))) + interval.I(
            radius_decimal.copy_negate(), radius_decimal)
        for coord in range(12):
            current += interval.I(D.from_float(float(matrix[output][coord]))) * dec_box[coord]
        dec_controls.append(current)
    control_bounds = [algebra.linear_range(row, box, center, radius)
                      for row, (center, radius) in zip(matrix, controls)]
    for dec, exact in zip(dec_controls, control_bounds):
        if F(dec.lo) > exact[0] or F(dec.hi) < exact[1]:
            raise ValueError("directed Decimal control hull omitted exact affine hull")
    return box, matrix, controls, dec_box, dec_controls, control_bounds


def adaptive_bootstrap(box, control_bounds):
    """Choose x10/x11 limits from this box's exact control hull, then close all invariants."""
    H, G, MASS, INERTIA = algebra.H, algebra.G, algebra.MASS, algebra.INERTIA
    V, ANGLE = algebra.V, algebra.ANGLE
    margin = F(1, 10**9)
    raw10 = H * algebra.ceil_abs(control_bounds[1]) / INERTIA
    raw11 = H * algebra.ceil_abs(control_bounds[2]) / INERTIA
    rate10, rate11 = raw10 + margin, raw11 + margin
    a4, a5, a6 = (algebra.ceil_abs(box[i]) for i in (3, 4, 5))
    u1_max = algebra.ceil_abs(control_bounds[0])
    b4 = rate11 * V + G * ANGLE
    b5 = rate10 * V + G * ANGLE
    other = rate11 * V + rate10 * V + G * ANGLE**2
    b6 = other + u1_max / MASS
    b7 = rate10 + ANGLE**2 * rate11 / (1 - ANGLE**2 / 2)
    b8 = rate11
    inequalities = {
        "x10_rate": (raw10, rate10),
        "x11_rate": (raw11, rate11),
        "x4": (a4 + H * b4, V),
        "x5": (a5 + H * b5, V),
        "x6": (a6 + H * b6, V),
        "x7": (H * b7, ANGLE),
        "x8": (H * b8, ANGLE),
    }
    failed = {name: [str(left), str(right)] for name, (left, right) in inequalities.items()
              if not left < right}
    if failed:
        raise RuntimeError("adaptive first-exit bootstrap undecided: " + json.dumps(failed))
    return other, inequalities, (raw10, rate10, raw11, rate11, margin)


def directed_picard(x, u):
    h = interval.H / interval.STEPS
    tube = [None] * 4
    for substep in range(interval.STEPS):
        fx = interval.ode(x, u)
        proposal = [x[i] + interval.I(0, h) * fx[i] +
                    interval.I(interval.EPS.copy_negate(), interval.EPS)
                    for i in range(12)]
        fy = interval.ode(proposal, u)
        picard = [x[i] + interval.I(0, h) * fy[i] for i in range(12)]
        if not all(picard[i].inside(proposal[i], strict=True) for i in range(12)):
            raise RuntimeError(f"strict Picard self-inclusion undecided at substep {substep}")
        for form, value in enumerate(interval.forms(proposal)):
            tube[form] = value if tube[form] is None else interval.I(
                min(tube[form].lo, value.lo), max(tube[form].hi, value.hi))
        x = [x[i] + interval.I(h) * fy[i] for i in range(12)]
    return tube, x


def one_box(lane, saved):
    params, coeff, source, octagon, terminal = saved
    box, matrix, controls, x0, u, control_bounds = contract(lane, params, coeff, source)
    other, bootstrap, rates = adaptive_bootstrap(box, control_bounds)
    tube, endpoint = directed_picard(x0, u)
    physical = {}
    for view, values in ((0, tube), (1, interval.forms(endpoint))):
        for form, value in enumerate(values):
            physical[("tube" if view == 0 else "endpoint", form)] = (F(value.lo), F(value.hi))
    for coord, value in enumerate(endpoint, 1):
        physical[("terminal_composed", coord)] = (F(value.lo), F(value.hi))

    # x12 remains zero and the constant-control x10/x11 dynamics are linear.
    physical[("terminal_composed", 10)] = tuple(
        algebra.H * value / algebra.INERTIA for value in control_bounds[1])
    physical[("terminal_composed", 11)] = tuple(
        algebra.H * value / algebra.INERTIA for value in control_bounds[2])
    # Retain shared initial-state/controller correlation for x6.
    coeff6 = [-algebra.H * value / algebra.MASS for value in matrix[0]]
    coeff6[5] += 1
    center, radius = controls[0]
    base6 = algebra.linear_range(coeff6, box, -algebra.H * center / algebra.MASS,
                                 algebra.H * radius / algebra.MASS)
    error6 = algebra.H * other
    physical[("terminal_composed", 6)] = (base6[0] - error6, base6[1] + error6)

    comparisons = []
    for (kind, index), bound in physical.items():
        if kind in ("tube", "endpoint"):
            row = octagon[lane, 1, 0 if kind == "tube" else 1, index]
            lo, hi = row["lo"], row["hi"]
        else:
            row = terminal[lane, index]
            lo, hi = row["composed_lo"], row["composed_hi"]
        saved_outward = algebra.outward_csv_bound(lo, hi)
        saved_raw = (F.from_float(float(lo)), F.from_float(float(hi)))
        comparisons.append({
            "kind": kind, "coord_or_form": index,
            "method": "algebraic" if kind == "terminal_composed" and index in (6, 10, 11)
                      else "directed_picard",
            "bound_exact": [str(value) for value in bound],
            "saved_outward_exact": [str(value) for value in saved_outward],
            "inside_outward": algebra.subset(bound, saved_outward),
            "inside_unexpanded_binary64": algebra.subset(bound, saved_raw),
        })
    if len(comparisons) != 20:
        raise ValueError("expected 8 direction and 12 composed-axis comparisons")
    return {
        "lane": lane,
        "strict_picard_substeps": interval.STEPS,
        "strict_bootstrap": {name: [str(value) for value in pair]
                             for name, pair in bootstrap.items()},
        "adaptive_rates_exact": {
            "raw_x10": str(rates[0]), "chosen_x10": str(rates[1]),
            "raw_x11": str(rates[2]), "chosen_x11": str(rates[3]),
            "strict_margin": str(rates[4]),
        },
        "control_hull_exact": [[str(value) for value in pair] for pair in control_bounds],
        "physical_outward_comparisons_inside": sum(row["inside_outward"] for row in comparisons),
        "physical_unexpanded_binary64_comparisons_inside":
            sum(row["inside_unexpanded_binary64"] for row in comparisons),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seconds-cap", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (0 <= args.start < N and 0 < args.count <= N - args.start and args.seconds_cap > 0):
        parser.error("invalid box span or time cap")
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    saved = load_saved()
    checked = 0
    issue = None
    checkpoint = args.output / "CHECKPOINT.json"
    checkpoint.write_text(json.dumps({"start_lane": args.start, "checked_boxes": 0,
                                      "next_unchecked_lane": args.start,
                                      "seconds_cap": args.seconds_cap}) + "\n")
    with (args.output / "BOX_AUDIT.jsonl").open("w") as ledger:
        for lane in range(args.start, args.start + args.count):
            if time.perf_counter() - begin > args.seconds_cap:
                issue = {"lane": lane, "kind": "TIME_CAP_BEFORE_BOX"}
                break
            started = time.perf_counter()
            try:
                record = one_box(lane, saved)
            except (AssertionError, KeyError, RuntimeError, ValueError, ZeroDivisionError) as exc:
                issue = {"lane": lane, "kind": "METHOD_UNDECIDED", "detail": str(exc)}
                break
            record["elapsed_seconds"] = time.perf_counter() - started
            ledger.write(json.dumps(record) + "\n")
            ledger.flush()
            checked += 1
            temporary = args.output / "CHECKPOINT.tmp"
            temporary.write_text(json.dumps({"start_lane": args.start,
                                             "checked_boxes": checked,
                                             "last_completed_lane": lane,
                                             "next_unchecked_lane": lane + 1,
                                             "elapsed_seconds": time.perf_counter() - begin,
                                             "seconds_cap": args.seconds_cap}) + "\n")
            temporary.replace(checkpoint)
            failed = [row for row in record["comparisons"] if not row["inside_outward"]]
            if failed:
                issue = {"lane": lane, "kind": "INDEPENDENT_BOUND_NOT_CONTAINED",
                         "comparisons": failed}
                break
    result = {
        "schema": "native-quad-allbox-independent-plant-firststep-nohash-v1",
        "saved_run": str(SAVED),
        "start_lane": args.start, "requested_boxes": args.count,
        "checked_boxes": checked,
        "next_unchecked_lane": args.start + checked,
        "seconds_cap": args.seconds_cap,
        "elapsed_seconds": time.perf_counter() - begin,
        "first_issue": issue,
        "method": "per-box exact Fraction control hull plus 1/10^9 strict x10/x11 rate margins and all seven first-exit inequalities; 1000 strict outward-Decimal Picard substeps for first h=0.005 paper ODE step; exact-rational correlated x6 and closed-form x10/x11 endpoints; saved binary64 observer bounds expanded one ULP for CSV serialization",
        "physical_composed_only": True,
        "scope": "saved source-defined first-call RPC inputs and saved affine-plus-residual controls, all t in first h=0.005 step only",
        "limits": "No independent CROWN/NN proof, Flow* parser/runtime floating-point certificate, later ODE steps, later control calls, or full-time reach-and-remain verdict; production gate CLOSED",
    }
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "start_lane", "requested_boxes", "checked_boxes", "next_unchecked_lane",
        "elapsed_seconds", "first_issue")}))
    return 0 if issue is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
