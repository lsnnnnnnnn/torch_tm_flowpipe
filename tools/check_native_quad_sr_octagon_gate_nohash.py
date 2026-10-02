"""Independent finite-sample audit of the isolated one-step native QUAD gate.

The SciPy trajectories are numerical witnesses inside the CROWN affine-plus-
residual relaxation used by the frozen C++ source.  This is deliberately not
an all-trajectory proof, an NN soundness check, or a full QUAD result.
"""

import csv
import itertools
import json
import math
from pathlib import Path
import struct
import sys

from scipy.integrate import solve_ivp


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def ode(_t, x, u):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = x
    s7, c7 = math.sin(x7), math.cos(x7)
    s8, c8 = math.sin(x8), math.cos(x8)
    s9, c9 = math.sin(x9), math.cos(x9)
    return [
        c8*c9*x4 + (s7*s8*c9-c7*s9)*x5 + (c7*s8*c9+s7*s9)*x6,
        c8*s9*x4 + (s7*s8*s9+c7*c9)*x5 + (c7*s8*s9-s7*c9)*x6,
        s8*x4 - s7*c8*x5 - c7*c8*x6,
        x12*x5 - x11*x6 - 9.81*s8,
        x10*x6 - x12*x4 + 9.81*c8*s7,
        x11*x4 - x10*x5 + 9.81*c8*c7 - 9.81 - u[0]/1.4,
        x10 + s7*s8/c8*x11 + c7*s8/c8*x12,
        c7*x11 - s7*x12,
        s7*x11/c8 - c7*x12/c8,
        x11*x12*(0.054-0.104)/0.054 + u[1]/0.054,
        (0.104-0.054)*x10*x12/0.054 + u[2]/0.054,
        0.0,
    ]


def solve(x0, u, rtol=1e-13, atol=1e-15):
    result = solve_ivp(lambda t, x: ode(t, x, u), (0.0, 0.005), x0,
                       method="DOP853", t_eval=(0.0, 0.0025, 0.005),
                       rtol=rtol, atol=atol, max_step=0.00125)
    if not result.success or result.y.shape != (12, 3):
        raise RuntimeError(f"numerical reference failed: {result.message}")
    return result.y


def forms(x, y):
    return (x, y, x + y, x - y)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: checker gate-evidence-directory")
    root = Path(sys.argv[1])
    off, on = root / "off", root / "on"
    state_off, state_on = read_csv(off / "state.csv"), read_csv(on / "state.csv")
    status_equal = state_off == state_on == [{"status": "2", "accepted_steps": "1"}]
    terminal_exact_equal = (off / "terminal_axes.csv").read_bytes() == (on / "terminal_axes.csv").read_bytes()
    ranges_exact_equal = (off / "ranges.bin").read_bytes() == (on / "ranges.bin").read_bytes()
    rpc_off = json.loads((off / "rpc.json").read_text())
    rpc_on = json.loads((on / "rpc.json").read_text())
    rpc_exact_equal = rpc_off == rpc_on
    rpc_lines = [json.loads(line) for line in (root / "controller_rpc.jsonl").read_text().splitlines()]
    rpc_count_valid = len(rpc_lines) == 2 and all(
        row == {"input_boxes": 1, "output_boxes": 1} for row in rpc_lines)

    octagon = read_csv(on / "octagon.csv")
    if len(octagon) != 8 or [(int(r["step"]), int(r["view"]), int(r["form"])) for r in octagon] != [
        (1, view, form) for view in range(2) for form in range(4)
    ]:
        raise RuntimeError("missing or reordered octagon directions")
    supports = {(int(r["view"]), int(r["form"])): (float(r["lo"]), float(r["hi"]))
                for r in octagon}
    terminal = read_csv(on / "terminal_axes.csv")
    if len(terminal) != 12 or [int(r["coord"]) for r in terminal] != list(range(1, 13)):
        raise RuntimeError("missing terminal axes")

    coefficients = rpc_on["coefficients"]
    T = [[f32(value) for value in row] for row in coefficients["T"][0]]
    lo = [f32(value) for value in coefficients["u_min"][0]]
    hi = [f32(value) for value in coefficients["u_max"][0]]
    center = [(a+b)/2 for a, b in zip(lo, hi)]
    radius = [(b-a)/2 for a, b in zip(lo, hi)]
    initial_lo = rpc_on["params"]["input_lb"]
    initial_hi = rpc_on["params"]["input_ub"]
    if len(initial_lo) != 12 or len(initial_hi) != 12 or len(T) != 3 or any(len(row) != 12 for row in T):
        raise RuntimeError("unexpected QUAD controller shape")
    if any(initial_lo[i] != initial_hi[i] for i in range(6, 12)):
        raise RuntimeError("this checker expects singleton initial angles and rates")
    if any(r < 0 for r in radius):
        raise RuntimeError("reversed CROWN residual bounds")

    witnesses = []
    for corners in itertools.product((0, 1), repeat=6):
        x0 = [initial_hi[i] if bit else initial_lo[i] for i, bit in enumerate(corners)]
        x0 += initial_lo[6:12]
        for residual_signs in itertools.product((-1, 1), repeat=3):
            u = [sum(T[j][i]*x0[i] for i in range(12)) + center[j]
                 + residual_signs[j]*radius[j] for j in range(3)]
            witnesses.append((tuple(x0), tuple(u), "corner"))
    midpoint = [(a+b)/2 for a, b in zip(initial_lo, initial_hi)]
    midpoint_u = [sum(T[j][i]*midpoint[i] for i in range(12)) + center[j] for j in range(3)]
    witnesses.append((tuple(midpoint), tuple(midpoint_u), "midpoint"))

    count = 0
    violations = []
    worst_overshoot = 0.0
    minimum_slack = math.inf

    def check(value, bounds, kind, witness, at_time, form):
        nonlocal count, worst_overshoot, minimum_slack
        count += 1
        lower, upper = bounds
        if not (math.isfinite(value) and lower <= upper):
            raise RuntimeError("nonfinite or reversed numerical inclusion")
        slack = min(value-lower, upper-value)
        minimum_slack = min(minimum_slack, slack)
        overshoot = max(lower-value, value-upper, 0.0)
        worst_overshoot = max(worst_overshoot, overshoot)
        if overshoot > 1e-11:
            violations.append({"kind": kind, "witness": witness, "time": at_time,
                               "coord_or_form": form, "sample": value,
                               "bounds": [lower, upper], "overshoot": overshoot})

    max_refinement_difference = 0.0
    first_trajectory = None
    first_radau_difference = None
    for index, (x0, u, label) in enumerate(witnesses):
        trajectory = solve(x0, u)
        if index == 0:
            radau = solve_ivp(lambda t, x: ode(t, x, u), (0.0, 0.005), x0,
                              method="Radau", t_eval=(0.0, 0.0025, 0.005),
                              rtol=1e-12, atol=1e-14, max_step=0.00125)
            if not radau.success:
                raise RuntimeError(f"Radau cross-check failed: {radau.message}")
            first_radau_difference = float(abs(trajectory-radau.y).max())
            first_trajectory = [
                {"time": time, "x7": float(trajectory[6, t_index]),
                 "x8": float(trajectory[7, t_index]),
                 "radau_x7": float(radau.y[6, t_index]),
                 "radau_x8": float(radau.y[7, t_index])}
                for t_index, time in enumerate((0.0, 0.0025, 0.005))
            ]
        if index < 8:
            looser = solve(x0, u, rtol=1e-11, atol=1e-13)
            max_refinement_difference = max(
                max_refinement_difference, float(abs(trajectory-looser).max()))
        for t_index, time in enumerate((0.0, 0.0025, 0.005)):
            values = forms(float(trajectory[0, t_index]), float(trajectory[1, t_index]))
            for form, value in enumerate(values):
                check(value, supports[(0, form)], "tube", index, time, form)
        values = forms(float(trajectory[0, 2]), float(trajectory[1, 2]))
        for form, value in enumerate(values):
            check(value, supports[(1, form)], "endpoint", index, 0.005, form)
        for axis in range(12):
            value = float(trajectory[axis, 2])
            row = terminal[axis]
            check(value, (float(row["pre_lo"]), float(row["pre_hi"])),
                  "terminal_pre", index, 0.005, axis+1)
            check(value, (float(row["composed_lo"]), float(row["composed_hi"])),
                  "terminal_composed", index, 0.005, axis+1)

    violation_counts = {}
    max_overshoot_by_kind = {}
    for violation in violations:
        kind = violation["kind"]
        violation_counts[kind] = violation_counts.get(kind, 0) + 1
        max_overshoot_by_kind[kind] = max(max_overshoot_by_kind.get(kind, 0.0), violation["overshoot"])
    first_x0, first_u, _ = witnesses[0]
    audit = {
        "schema": "native-quad-sr-one-step-octagon-gate-nohash-v1",
        "contract": "2026 paper QUAD equations, first of 1024 native boxes, one CROWN call, one h=0.005/order-2 symbolic-remainder step",
        "mode_status": state_on[0],
        "on_off_status_equal": status_equal,
        "on_off_terminal_axes_byte_equal": terminal_exact_equal,
        "on_off_ranges_bin_byte_equal": ranges_exact_equal,
        "on_off_rpc_json_equal": rpc_exact_equal,
        "controller_rpc_count_and_shapes_valid": rpc_count_valid,
        "reference": "SciPy DOP853 finite point trajectories using the paper ODE and float32-transported CROWN affine-plus-residual controls",
        "reference_samples": len(witnesses),
        "reference_corner_samples": 512,
        "reference_midpoint_samples": 1,
        "sample_direction_and_axis_checks": count,
        "numeric_tolerance": 1e-11,
        "max_reference_loose_strict_difference_first_eight": max_refinement_difference,
        "first_witness_dop853_radau_max_difference": first_radau_difference,
        "first_witness": {
            "initial_x1_to_x12": first_x0,
            "constant_u1_to_u3": first_u,
            "residual_choice": [-1, -1, -1],
            "affine_center_at_x": [sum(T[j][i]*first_x0[i] for i in range(12)) + center[j]
                                   for j in range(3)],
            "residual_radius": radius,
            "trajectory_x7_x8": first_trajectory,
        },
        "minimum_sample_slack": minimum_slack,
        "maximum_sample_overshoot": worst_overshoot,
        "violation_count_above_tolerance": len(violations),
        "violation_counts_by_kind": violation_counts,
        "max_overshoot_by_kind": max_overshoot_by_kind,
        "first_violations": violations[:12],
        "qualification": "finite numerical diagnostic only; no all-trajectory enclosure, NN float soundness or T=5 property proof",
    }
    (root / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in (
        "on_off_status_equal", "on_off_terminal_axes_byte_equal",
        "on_off_ranges_bin_byte_equal", "on_off_rpc_json_equal",
        "controller_rpc_count_and_shapes_valid", "reference_samples",
        "sample_direction_and_axis_checks", "minimum_sample_slack",
        "maximum_sample_overshoot", "violation_count_above_tolerance",
        "violation_counts_by_kind", "first_witness_dop853_radau_max_difference")}, indent=2))
    if not all((status_equal, terminal_exact_equal, ranges_exact_equal,
                rpc_exact_equal, rpc_count_valid)) or violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
