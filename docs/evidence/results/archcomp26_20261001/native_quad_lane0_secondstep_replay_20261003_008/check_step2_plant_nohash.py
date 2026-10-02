#!/usr/bin/env python3
"""Conditional directed-interval plant inclusion for lane 0, ODE step 2."""

import json
from decimal import Decimal as D
from fractions import Fraction as F
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "native_quad_allbox_independent_plant_adaptive_bootstrap_20261003_006"
sys.path.insert(0, str(PRIOR))
import check_saved_boxes_nohash as prior  # noqa: E402

H2 = F(1, 100)


def bootstrap_two_steps(box, bounds):
    """Re-establish the seven strict first-exit inequalities at T=0.010."""
    a = prior.algebra
    margin = F(1, 10**9)
    raw10 = H2 * a.ceil_abs(bounds[1]) / a.INERTIA
    raw11 = H2 * a.ceil_abs(bounds[2]) / a.INERTIA
    rate10, rate11 = raw10 + margin, raw11 + margin
    a4, a5, a6 = (a.ceil_abs(box[i]) for i in (3, 4, 5))
    other = rate11 * a.V + rate10 * a.V + a.G * a.ANGLE**2
    inequalities = {
        "x10_rate": (raw10, rate10),
        "x11_rate": (raw11, rate11),
        "x4": (a4 + H2 * (rate11 * a.V + a.G * a.ANGLE), a.V),
        "x5": (a5 + H2 * (rate10 * a.V + a.G * a.ANGLE), a.V),
        "x6": (a6 + H2 * (other + a.ceil_abs(bounds[0]) / a.MASS), a.V),
        "x7": (H2 * (rate10 + a.ANGLE**2 * rate11 / (1 - a.ANGLE**2 / 2)),
               a.ANGLE),
        "x8": (H2 * rate11, a.ANGLE),
    }
    if any(left >= right for left, right in inequalities.values()):
        raise RuntimeError("two-step first-exit bootstrap undecided: " + json.dumps(
            {key: [str(left), str(right)] for key, (left, right) in inequalities.items()
             if left >= right}))
    return other, {key: [str(left), str(right)] for key, (left, right) in inequalities.items()}


def run():
    replay = json.loads((HERE / "TWO_REPLAY_AUDIT.json").read_text())
    if replay["first_issue"] is not None or replay["accepted_steps"] != 2:
        raise ValueError("two-step native replay has not passed first-step identity gate")
    root = HERE / "two"
    prior.SAVED = root
    rpc = json.loads((root / "rpc.json").read_text())
    source = prior.read_rows("source_boxes.csv", ("lane", "coord"))
    octagon = prior.read_rows("octagon.csv", ("lane", "step", "view", "form"))
    terminal = prior.read_rows("terminal_axes.csv", ("lane", "coord"))
    if len(octagon) != 16 or len(terminal) != 12:
        raise ValueError("incomplete two-step native observer rows")
    box, matrix, controls, initial, u, bounds = prior.contract(
        0, rpc["params"], rpc["coefficients"], source)
    other, inequalities = bootstrap_two_steps(box, bounds)

    # A single saved control hull is held for the entire first 0.1 s period.
    # The first call encloses [0,.005]; the second accumulates only [.005,.010] tube.
    _, at_first = prior.directed_picard(initial, u)
    tube_second, endpoint = prior.directed_picard(at_first, u)
    physical = {}
    for view, values in ((0, tube_second), (1, prior.interval.forms(endpoint))):
        for form, value in enumerate(values):
            physical[("tube" if view == 0 else "endpoint", form)] = (F(value.lo), F(value.hi))
    for coord, value in enumerate(endpoint, 1):
        physical[("terminal_composed", coord)] = (F(value.lo), F(value.hi))
    a = prior.algebra
    physical[("terminal_composed", 10)] = tuple(H2 * value / a.INERTIA for value in bounds[1])
    physical[("terminal_composed", 11)] = tuple(H2 * value / a.INERTIA for value in bounds[2])
    coeff6 = [-H2 * value / a.MASS for value in matrix[0]]
    coeff6[5] += 1
    center, radius = controls[0]
    base6 = a.linear_range(coeff6, box, -H2 * center / a.MASS, H2 * radius / a.MASS)
    physical[("terminal_composed", 6)] = (base6[0] - H2 * other, base6[1] + H2 * other)

    comparisons = []
    for (kind, index), actual in physical.items():
        if kind in ("tube", "endpoint"):
            saved = octagon[0, 2, 0 if kind == "tube" else 1, index]
            lo, hi = saved["lo"], saved["hi"]
        else:
            saved = terminal[0, index]
            lo, hi = saved["composed_lo"], saved["composed_hi"]
        raw = (F.from_float(float(lo)), F.from_float(float(hi)))
        outward = a.outward_csv_bound(lo, hi)
        comparisons.append({
            "kind": kind, "coord_or_form": index,
            "method": "algebraic" if kind == "terminal_composed" and index in (6, 10, 11)
                      else "directed_picard",
            "physical_exact": [str(value) for value in actual],
            "saved_raw_binary64_exact": [str(value) for value in raw],
            "saved_outward_exact": [str(value) for value in outward],
            "inside_raw_binary64": a.subset(actual, raw),
            "inside_outward": a.subset(actual, outward),
        })
    if len(comparisons) != 20:
        raise ValueError("expected eight directions and twelve terminal axes")
    return {
        "schema": "native-quad-lane0-step2-independent-plant-nohash-v1",
        "status": "conditional_contained" if all(row["inside_outward"] for row in comparisons)
                  else "independent_bound_not_contained",
        "source_lane": 0,
        "time_window": ["0.005", "0.010"],
        "saved_controller_call": 1,
        "control_contract": "saved first-call float32 CROWN affine-plus-residual hull; assumed controller enclosure",
        "strict_picard_substeps": 2000,
        "second_step_tube_substeps": [1001, 2000],
        "strict_first_exit_inequalities": inequalities,
        "inside_outward": sum(row["inside_outward"] for row in comparisons),
        "inside_raw_binary64": sum(row["inside_raw_binary64"] for row in comparisons),
        "comparisons_total": 20,
        "first_issue": next((row for row in comparisons if not row["inside_outward"]), None),
        "comparisons": comparisons,
        "production_gate": "CLOSED",
    }


def main():
    started = time.perf_counter()
    def time_limit(*_):
        raise TimeoutError("30 s cap")
    signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(30)
    try:
        result = run()
    except (AssertionError, KeyError, RuntimeError, TimeoutError, ValueError, ZeroDivisionError) as exc:
        result = {"schema": "native-quad-lane0-step2-independent-plant-nohash-v1",
                  "status": "method_undecided", "first_issue": str(exc), "production_gate": "CLOSED"}
    finally:
        signal.alarm(0)
    result["elapsed_seconds"] = time.perf_counter() - started
    (HERE / "PLANT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "comparisons"}))
    if result["status"] != "conditional_contained":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
