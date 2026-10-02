"""Recheck the repaired first physical QUAD box against its new saved RPC.

No solver, CROWN call, or content digest is performed here.  The earlier
directed-Decimal Picard implementation supplies 27 broad-box comparisons;
five dependency-sensitive bounds are redone with exact rational algebra.
"""

import json
import sys
from fractions import Fraction as F
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "tools"))
import check_native_quad_algebraic_gate_nohash as algebra  # noqa: E402


def load(path):
    return json.loads(path.read_text())


def pair_text(pair):
    return [str(value) for value in pair]


def main():
    algebra.GATE = HERE / "quad/on"
    algebra.SOURCE = HERE / "quad/build/quad_gate.cpp"
    previous = HERE.parent / "native_quad_var_tail_repair_gate_20261002_007"
    assert algebra.SOURCE.read_bytes() == (previous / "quad/build/quad_gate.cpp").read_bytes()
    source = algebra.SOURCE.read_text()
    for expression in (
        '"x11*x4 - x10*x5 + 9.81 *cos(x8)*cos(x7) - 9.81 - u1 / 1.4"',
        '"x11*x12*(0.054 - 0.104) / 0.054 + u2 / 0.054"',
        '"(0.104 - 0.054)*x10*x12 / 0.054 + u3 / 0.054"',
        "Interval init_x1(-0.4, 0.4), init_x2(-0.4, 0.4), init_x3(-0.4, 0.4)",
        "init_x1.split(list_x1, 8)",
        "initial_sets.resize(1)",
        "int steps = 1",
    ):
        assert expression in source

    box, matrix, controls = algebra.rpc_contract()
    source_low = F.from_float(-0.4)
    first_hi = F.from_float(float.fromhex("-0x1.3333333333333p-2"))
    source_box = [(source_low, first_hi)] * 3 + [
        (source_low, F(0)), (source_low, F.from_float(0.4)),
        (source_low, F.from_float(0.4))] + [(F(0), F(0))] * 6
    assert len(box) == len(source_box) == 12
    assert all(algebra.subset(source, rpc) for source, rpc in zip(source_box, box))
    old_rpc = load(previous / "quad/on/rpc.json")
    new_rpc = load(HERE / "quad/on/rpc.json")
    old_low = F.from_float(float(old_rpc["params"]["input_lb"][0]))
    assert old_low > source_low and old_low - source_low == F(1, 2**54)
    transported_coefficients_equal = all(
        [algebra.binary32(v) for v in new_rpc["coefficients"]["T"][0][j]] ==
        [algebra.binary32(v) for v in old_rpc["coefficients"]["T"][0][j]]
        for j in range(3)) and all(
            [algebra.binary32(v) for v in new_rpc["coefficients"][key][0]] ==
            [algebra.binary32(v) for v in old_rpc["coefficients"][key][0]]
            for key in ("u_min", "u_max"))
    assert transported_coefficients_equal

    control_bounds = [algebra.linear_range(row, box, center, radius)
                      for row, (center, radius) in zip(matrix, controls)]
    other, bootstrap = algebra.check_bootstrap(box, control_bounds)
    h, inertia, mass = algebra.H, algebra.INERTIA, algebra.MASS
    exact = {
        ("terminal_pre", 10): tuple(h * v / inertia for v in control_bounds[1]),
        ("terminal_composed", 10): tuple(h * v / inertia for v in control_bounds[1]),
        ("terminal_pre", 11): tuple(h * v / inertia for v in control_bounds[2]),
        ("terminal_composed", 11): tuple(h * v / inertia for v in control_bounds[2]),
    }
    x6_coefficients = [-h * value / mass for value in matrix[0]]
    x6_coefficients[5] += 1
    center, radius = controls[0]
    x6_base = algebra.linear_range(x6_coefficients, box,
                                   -h * center / mass, h * radius / mass)
    x6_error = h * other
    exact[("terminal_composed", 6)] = (x6_base[0] - x6_error,
                                       x6_base[1] + x6_error)

    prior = load(HERE / "INDEPENDENT_AUDIT.json")
    assert prior["picard_inclusion_passed"] and prior["comparison_count"] == 32
    native, raw_native = algebra.native_intervals()
    assert len(native) == 32
    rows = []
    broad_failures = set()
    for row in prior["comparisons"]:
        key = (row["kind"], row.get("coord", row.get("form")))
        broad = algebra.pair_to_fractions(row["independent"])
        assert key in native
        assert algebra.pair_to_fractions(row["native"]) == native[key]
        assert algebra.subset(broad, native[key]) == row["inside"]
        if not row["inside"]:
            broad_failures.add(key)
        bound = exact.get(key, broad)
        inside = algebra.subset(bound, native[key])
        raw_margins = ((bound[0] - raw_native[key][0],
                        raw_native[key][1] - bound[1]) if key in exact else None)
        role = ("pre_numeric_column_only" if key[0] == "terminal_pre"
                else "physical_composed")
        rows.append({
            "kind": key[0], "coord_or_form": key[1], "role": role,
            "method": "exact_algebraic" if key in exact else "directed_picard",
            "bound_exact": pair_text(bound), "saved_outward_exact": pair_text(native[key]),
            "inside": inside,
            "inside_saved_binary64_unexpanded":
                algebra.subset(bound, raw_native[key]) if key in exact else None,
            "algebraic_raw_binary64_margins_exact":
                pair_text(raw_margins) if raw_margins else None,
            "algebraic_raw_binary64_margins_approx":
                [float(value) for value in raw_margins] if raw_margins else None,
        })
    assert broad_failures == set(exact)
    assert all(row["inside"] for row in rows)
    assert all(row["inside_saved_binary64_unexpanded"]
               for row in rows if row["method"] == "exact_algebraic")
    physical = [row for row in rows if row["role"] == "physical_composed"]
    pre = [row for row in rows if row["role"] == "pre_numeric_column_only"]
    assert len(physical) == 20 and len(pre) == 12

    finite = load(HERE / "quad/AUDIT.json")
    scans = {mode: load(HERE / f"quad/{mode}/SCAN.json")
             for mode in ("off", "on")}
    assert finite["on_off_status_equal"] and finite["on_off_ranges_bin_byte_equal"]
    assert finite["on_off_terminal_axes_byte_equal"] and finite["on_off_rpc_json_equal"]
    assert finite["controller_rpc_count_and_shapes_valid"]
    assert finite["violation_count_above_tolerance"] == 0
    assert all(scan["complete_grid"] and scan["records"] == 1
               and scan["reversed_component_intervals"] == 0
               for scan in scans.values())

    result = {
        "schema": "native-quad-initial-recenter-repair-one-box-gate-nohash-v1",
        "scope": "source-defined first of 1024 QUAD boxes, first h=0.005 step, saved CROWN affine-plus-residual control hull",
        "source_box_covered_by_new_rpc": True,
        "source_box_covered_by_old_rpc": False,
        "source_x1_x3_lower_hex": float(source_low).hex(),
        "old_rpc_x1_x3_lower_hex": float(old_low).hex(),
        "new_rpc_x1_x3_lower_hex": [float(box[i][0]).hex() for i in range(3)],
        "source_x1_x3_upper_hex": float(first_hi).hex(),
        "old_rpc_x1_x3_upper_hex": [float(value).hex() for value in old_rpc["params"]["input_ub"][:3]],
        "new_rpc_x1_x3_upper_hex": [float(box[i][1]).hex() for i in range(3)],
        "old_omission_exact": str(old_low - source_low),
        "new_source_lower_slack_exact": [str(source_low - box[i][0]) for i in range(3)],
        "new_source_box_exact": [pair_text(pair) for pair in source_box],
        "new_rpc_box_exact": [pair_text(pair) for pair in box],
        "crown_coefficients_equal_previous_json": new_rpc["coefficients"] == old_rpc["coefficients"],
        "crown_coefficients_equal_previous_after_float32_transport": transported_coefficients_equal,
        "finite_check": {
            "observer_on_off_equal": True,
            "complete_saved_range_records_per_arm": 1,
            "numerical_samples": finite["reference_samples"],
            "sample_checks": finite["sample_direction_and_axis_checks"],
            "sample_violations_above_tolerance": finite["violation_count_above_tolerance"],
        },
        "independent_check": {
            "directed_decimal_picard_substeps": prior["substeps"],
            "strict_picard_inclusion": True,
            "broad_box_comparisons_inside": 27,
            "broad_box_comparisons_total": 32,
            "exact_algebraic_comparisons_redone": 5,
            "physical_composed_inside": sum(row["inside"] for row in physical),
            "physical_composed_total": len(physical),
            "physical_bounds_inside_pre_numeric_columns": sum(row["inside"] for row in pre),
            "pre_numeric_columns_total": len(pre),
            "pre_symbolic_set_inclusion_proven": False,
            "bootstrap_strict": {key: [str(v) for v in values]
                                 for key, values in bootstrap.items()},
            "x6_correlated_base_exact": pair_text(x6_base),
            "x6_integral_error_limit_exact": str(x6_error),
            "comparisons": rows,
        },
        "limitation": "One source-defined box and one plant step under saved relaxed control only. Numerical pre columns do not certify the pre symbolic set. This does not validate CROWN or actual network soundness, Flow* parser arithmetic, any of the other 1023 boxes, later controls, T=5, or the reach-and-remain property. Production gate remains closed.",
    }
    (HERE / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_box_covered_by_new_rpc": True,
                      "physical_composed_inside": len(physical),
                      "physical_composed_total": len(physical),
                      "pre_numeric_columns_inside": len(pre),
                      "sample_violations": finite["violation_count_above_tolerance"]}))


if __name__ == "__main__":
    main()
