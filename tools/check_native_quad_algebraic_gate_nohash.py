"""Read-only algebraic check of the saved first-call RPC QUAD repair gate.

This does not run Flow*, CROWN, a numerical ODE solver, or a digest check.
The mathematical ODE constants are the exact paper decimals.  The saved RPC
coefficients are rounded to binary32 exactly as the frozen C++ uses asFloat().
The RPC/recenter domain is narrower than the C++ first physical box.
"""

import csv
import json
import math
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
import struct


REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/evidence/results/archcomp26_20261001"
GATE = EVIDENCE / "native_quad_var_tail_repair_gate_20261002_007/quad/on"
PRIOR = EVIDENCE / "native_quad_independent_interval_gate_20261002_001/repaired_AUDIT.json"
SOURCE = EVIDENCE / "native_quad_paper_full50_001/build/archcomp/Quadrotor/quad_paper_full50.cpp"

H, G, MASS, INERTIA = F(5, 1000), F(981, 100), F(14, 10), F(54, 1000)
V, ANGLE, RATE10, RATE11 = F(1), F(1, 100), F(3, 2000), F(1, 1000)


def rational_decimal(value):
    return F(Decimal(str(value)))


def binary32(value):
    return F.from_float(struct.unpack("<f", struct.pack("<f", float(value)))[0])


def pair_to_fractions(pair):
    return tuple(rational_decimal(v) for v in pair)


def subset(inner, outer):
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def linear_range(coefficients, input_box, constant=F(0), radius=F(0)):
    low, high = constant - radius, constant + radius
    for c, (a, b) in zip(coefficients, input_box):
        low += min(c * a, c * b)
        high += max(c * a, c * b)
    return low, high


def outward_csv_bound(low, high):
    # The observer saved 17 significant binary64 digits after MPFR inf/sup.
    # Expand the parsed binary64 by one ULP as in the previous interval gate.
    return (F.from_float(math.nextafter(float(low), -math.inf)),
            F.from_float(math.nextafter(float(high), math.inf)))


def native_intervals():
    output, raw = {}, {}
    with (GATE / "octagon.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            assert int(row["step"]) == 1
            kind = "tube" if int(row["view"]) == 0 else "endpoint"
            key = (kind, int(row["form"]))
            assert key not in output
            output[key] = outward_csv_bound(row["lo"], row["hi"])
            raw[key] = (F.from_float(float(row["lo"])), F.from_float(float(row["hi"])))
    with (GATE / "terminal_axes.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            coord = int(row["coord"])
            for mode in ("pre", "composed"):
                key = ("terminal_" + mode, coord)
                assert key not in output
                output[key] = outward_csv_bound(row[mode + "_lo"], row[mode + "_hi"])
                raw[key] = (F.from_float(float(row[mode + "_lo"])),
                            F.from_float(float(row[mode + "_hi"])))
    assert len(output) == 32
    return output, raw


def rpc_contract():
    record = json.loads((GATE / "rpc.json").read_text())
    params, coeff = record["params"], record["coefficients"]
    box = [(F.from_float(float(a)), F.from_float(float(b)))
           for a, b in zip(params["input_lb"], params["input_ub"])]
    assert len(box) == 12 and all(a <= b for a, b in box)
    assert all(box[i] == (0, 0) for i in range(6, 12))
    matrix = [[binary32(v) for v in row] for row in coeff["T"][0]]
    assert len(matrix) == 3 and all(len(row) == 12 for row in matrix)
    controls = []
    for lo, hi in zip(coeff["u_min"][0], coeff["u_max"][0]):
        lo, hi = binary32(lo), binary32(hi)
        assert lo <= hi
        center, radius = (lo + hi) / 2, (hi - lo) / 2
        # The frozen C++ computes center/radius in binary64 after asFloat().
        assert F.from_float(float(center)) == center
        assert F.from_float(float(radius)) == radius
        controls.append((center, radius))
    assert len(controls) == 3
    return box, matrix, controls


def ceil_abs(bounds):
    return max(abs(bounds[0]), abs(bounds[1]))


def check_bootstrap(box, control_bounds):
    # A strict first-exit argument proves these invariants for all t in [0,H].
    # x12'=0 and x12(0)=0 give x12=0; x10/x11 are linear in time.
    assert H * ceil_abs(control_bounds[1]) / INERTIA < RATE10
    assert H * ceil_abs(control_bounds[2]) / INERTIA < RATE11
    a4, a5, a6 = (ceil_abs(box[i]) for i in (3, 4, 5))
    assert box[6] == box[7] == (0, 0)
    u1_max = ceil_abs(control_bounds[0])
    # |sin z|<=|z|, cos z>=1-z²/2 for |z|<=ANGLE.
    # |1-cos(x7)cos(x8)|<=ANGLE² in the bootstrap box.
    b4 = RATE11 * V + G * ANGLE
    b5 = RATE10 * V + G * ANGLE
    other = RATE11 * V + RATE10 * V + G * ANGLE**2
    b6 = other + u1_max / MASS
    b7 = RATE10 + ANGLE**2 * RATE11 / (1 - ANGLE**2 / 2)
    b8 = RATE11
    inequalities = {
        "x4": (a4 + H * b4, V),
        "x5": (a5 + H * b5, V),
        "x6": (a6 + H * b6, V),
        "x7": (H * b7, ANGLE),
        "x8": (H * b8, ANGLE),
    }
    assert all(left < right for left, right in inequalities.values())
    return other, inequalities


def as_text(value):
    return str(Decimal(value.numerator) / Decimal(value.denominator))


def main():
    source = SOURCE.read_text()
    for expression in (
        '"x11*x4 - x10*x5 + 9.81 *cos(x8)*cos(x7) - 9.81 - u1 / 1.4"',
        '"x11*x12*(0.054 - 0.104) / 0.054 + u2 / 0.054"',
        '"(0.104 - 0.054)*x10*x12 / 0.054 + u3 / 0.054"',
        '"0",\n                        "1", "0", "0", "0"',
        "tmv_output.tms[j] += initial_sets[sub_iter].tmvPre.tms[i] * T[j][i];",
        "initial_sets[sub_iter].tmvPre.tms[u_ids[j]] = tmv_output.tms[j];",
        "Interval init_x1(-0.4, 0.4), init_x2(-0.4, 0.4), init_x3(-0.4, 0.4)",
    ):
        assert expression in source
    box, matrix, controls = rpc_contract()
    physical_low = F.from_float(-0.4)
    rpc_low = F.from_float(math.nextafter(-0.4, math.inf))
    assert all(box[i][0] == rpc_low and physical_low < rpc_low for i in range(3))
    control_bounds = [linear_range(row, box, center, radius)
                      for row, (center, radius) in zip(matrix, controls)]
    other, bootstrap = check_bootstrap(box, control_bounds)

    # Because x12=0 and controls are constant on this first step, these are
    # exact physical solution extrema for x10/x11 over the complete box.
    proven = {
        ("terminal_pre", 10): tuple(H * b / INERTIA for b in control_bounds[1]),
        ("terminal_composed", 10): tuple(H * b / INERTIA for b in control_bounds[1]),
        ("terminal_pre", 11): tuple(H * b / INERTIA for b in control_bounds[2]),
        ("terminal_composed", 11): tuple(H * b / INERTIA for b in control_bounds[2]),
    }
    coeff6 = [-H * v / MASS for v in matrix[0]]
    coeff6[5] += 1
    center1, radius1 = controls[0]
    base6 = linear_range(coeff6, box, -H * center1 / MASS, H * radius1 / MASS)
    error6 = H * other
    proven[("terminal_composed", 6)] = (base6[0] - error6, base6[1] + error6)

    prior = json.loads(PRIOR.read_text())
    assert prior["comparison_count"] == 32 and prior["picard_inclusion_passed"]
    native, raw_native = native_intervals()
    previous = {}
    for row in prior["comparisons"]:
        key = (row["kind"], row.get("coord", row.get("form")))
        assert key not in previous and pair_to_fractions(row["native"]) == native[key]
        old = pair_to_fractions(row["independent"])
        assert subset(old, native[key]) == row["inside"]
        previous[key] = row
    assert set(previous) == set(native)
    assert set(proven) == {key for key, row in previous.items() if not row["inside"]}

    rows = []
    for key, row in previous.items():
        interval = proven.get(key, pair_to_fractions(row["independent"]))
        passed = subset(interval, native[key])
        unexpanded_passed = subset(interval, raw_native[key]) if key in proven else None
        assert unexpanded_passed is not False
        raw_margins = ([interval[0] - raw_native[key][0], raw_native[key][1] - interval[1]]
                       if key in proven else None)
        rows.append({"kind": key[0], "coord_or_form": key[1],
                     "role": "physical_bound_vs_pre_numeric_column" if key[0] == "terminal_pre"
                             else "physical_composed",
                     "method": "algebraic" if key in proven else "prior_interval_picard",
                     "independent": [as_text(v) for v in interval],
                     "independent_exact": [str(v) for v in interval],
                     "native": [as_text(v) for v in native[key]],
                     "native_exact": [str(v) for v in native[key]], "inside": passed,
                     "algebraic_inside_unexpanded_binary64": unexpanded_passed,
                     "algebraic_raw_binary64_margins_exact":
                         [str(v) for v in raw_margins] if raw_margins else None,
                     "algebraic_raw_binary64_margins":
                         [as_text(v) for v in raw_margins] if raw_margins else None})
    physical = [r for r in rows if r["role"] == "physical_composed"]
    pre = [r for r in rows if r["role"] == "physical_bound_vs_pre_numeric_column"]
    # Sensitivity only: substituting nearest binary64 literals for h and
    # inertia in the closed forms is not a check of Flow*'s ODE parser.
    binary64_h, binary64_inertia = F.from_float(0.005), F.from_float(0.054)
    alternative = [tuple(binary64_h * b / binary64_inertia for b in control_bounds[j])
                   for j in (1, 2)]
    alternative_inside_raw = all(
        subset(alternative[coord - 10], raw_native[("terminal_" + mode, coord)])
        for coord in (10, 11) for mode in ("pre", "composed"))
    result = {
        "schema": "native-quad-algebraic-whole-set-first-step-gate-nohash-v1",
        "scope": "saved first-call RPC/recenter-TM input domain, first 0.005 s, saved affine-plus-residual control only",
        "covers_cpp_first_physical_box": False,
        "rpc_x1_x3_lower_vs_cpp_first_box": {
            "rpc_lower_binary64": str(float(rpc_low)),
            "cpp_first_box_lower_binary64": str(float(physical_low)),
            "gap_exact": str(rpc_low - physical_low),
            "meaning": "RPC x1-x3 lower bounds are one binary64 ULP above the C++ first physical box lower bounds",
        },
        "source": "saved repaired RPC/octagon/terminal axes plus prior independent interval audit",
        "exact_constants": {"h": "0.005", "gravity": "9.81", "mass": "1.4", "inertia": "0.054"},
        "constant_scope": "paper-equation exact-decimal ODE; no independent validation of Flow* parser or internal Real arithmetic",
        "nearest_binary64_h_inertia_sensitivity": {
            "x10_x11_closed_forms_inside_unexpanded_binary64_columns": alternative_inside_raw,
            "meaning": "hypothetical nearest-binary64 h and 0.054 only; not a Flow* parser certificate",
        },
        "rpc_rounding": "JSON binary64 input bounds; T/u_min/u_max rounded to binary32 as C++ asFloat; centers and radii checked exactly representable as binary64",
        "comparison_arithmetic": "Fraction exact; decimal fields are display approximations, *_exact fields are rational values used in all comparisons",
        "bootstrap": {k: {"strict_bound": as_text(v[0]), "limit": as_text(v[1])}
                      for k, v in bootstrap.items()},
        "x6_correlated_base": [as_text(v) for v in base6],
        "x6_correlated_base_exact": [str(v) for v in base6],
        "x6_integral_error_limit": as_text(error6),
        "x6_integral_error_limit_exact": str(error6),
        "physical_composed_passed": sum(r["inside"] for r in physical),
        "physical_composed_total": len(physical),
        "pre_numeric_column_physical_bound_inside": sum(r["inside"] for r in pre),
        "pre_numeric_column_total": len(pre),
        "pre_set_inclusion_proven": False,
        "all_numeric_comparisons_inside": all(r["inside"] for r in rows),
        "all_five_algebraic_comparisons_inside_unexpanded_binary64": all(
            r["algebraic_inside_unexpanded_binary64"] for r in rows if r["method"] == "algebraic"),
        "comparisons": rows,
        "limitation": "The RPC/recenter domain omits a one-ULP lower strip of x1-x3 from the C++ first physical box. Physical trajectory bounds lying inside pre numeric columns do not prove enclosure of the pre symbolic set. This exact-decimal one-step plant test does not validate Flow* parsing, CROWN, NN output, other boxes, later controls, or T=5.",
    }
    print(json.dumps(result, indent=2))
    if not result["all_numeric_comparisons_inside"] or len(physical) != 20 or len(pre) != 12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
