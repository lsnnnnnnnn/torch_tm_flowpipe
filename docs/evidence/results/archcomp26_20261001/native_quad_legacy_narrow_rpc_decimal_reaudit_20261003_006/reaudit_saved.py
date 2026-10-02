#!/usr/bin/env python3
"""Recheck saved narrow-RPC QUAD gates without running either solver."""

import csv
from decimal import Decimal as D
from fractions import Fraction as F
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BASE = HERE.parent
sys.path.insert(0, str(REPO / "tools"))
import check_native_quad_fullset_interval_gate_nohash as interval_gate  # noqa: E402
import check_native_quad_algebraic_gate_nohash as algebra  # noqa: E402

ORIGINAL = BASE / "native_quad_sr_octagon_gate_20261002_005"
REPAIRED = BASE / "native_quad_var_tail_repair_gate_20261002_007"
PRIOR = BASE / "native_quad_independent_interval_gate_20261002_001"


def exact_pair(pair):
    return tuple(F(D(value)) for value in pair)


def key(row):
    return row["kind"], row.get("form", row.get("coord"))


def csv_bounds(root):
    root = root / "quad" if (root / "quad/on/rpc.json").exists() else root
    octagon = interval_gate.read_csv(root / "on/octagon.csv")
    terminal = interval_gate.read_csv(root / "on/terminal_axes.csv")
    if len(octagon) != 8 or len(terminal) != 12:
        raise ValueError("saved observer row count is incomplete")
    seen = 0
    for rows, pairs in ((octagon, (("lo", "hi"),)),
                        (terminal, (("pre_lo", "pre_hi"),
                                    ("composed_lo", "composed_hi")))):
        for row in rows:
            for lower, upper in pairs:
                a, b = row[lower], row[upper]
                af, bf = float(a), float(b)
                if not (math.isfinite(af) and math.isfinite(bf) and af <= bf):
                    raise ValueError("nonfinite or reversed saved observer bound")
                outward = interval_gate.saved_interval(a, b)
                if not (F(outward.lo) <= F(D(a)) <= F(D(b)) <= F(outward.hi)):
                    raise ValueError("one-ULP saved interval misses CSV decimal text")
                seen += 2
    return seen


def method_check():
    interval_gate.demo()
    original_float = D.from_float(0.1)
    ambient_negation = -original_float
    exact_negation = original_float.copy_negate()
    ambient_abs = abs(exact_negation)
    remainder = interval_gate.trig_remainder(D.from_float(0.004), 7, 5040)
    checks = {
        "old_context_negation_inexact": ambient_negation != exact_negation,
        "old_context_abs_underestimates": ambient_abs < original_float,
        "old_context_negative_remainder_inward": -remainder > remainder.copy_negate(),
        "fixed_interval_negation_contains_exact":
            (-interval_gate.I(original_float)).lo <= exact_negation <=
            (-interval_gate.I(original_float)).hi,
        "fixed_float_constructor_is_binary64_exact":
            interval_gate.I(0.1).lo == original_float,
        "fixed_trig_remainder_outward":
            F(remainder) >= F(D.from_float(0.004)) ** 7 / 5040,
        "fixed_trig_negative_radius_exact":
            F(remainder.copy_negate()) == -F(remainder),
        "substep_exact":
            interval_gate.H / interval_gate.STEPS == D("0.000005") and
            (interval_gate.H / interval_gate.STEPS) * interval_gate.STEPS == interval_gate.H,
    }
    if not all(checks.values()):
        raise ValueError(f"Decimal method self-check rejected: {checks}")
    return checks


def recheck_arm(root, prior_path, expected_count):
    fresh = interval_gate.run(root)
    previous = json.loads(prior_path.read_text())
    if not fresh["picard_inclusion_passed"] or fresh["comparison_count"] != 32:
        raise ValueError("fixed Picard or 32-column gate rejected")
    fresh_rows = {key(row): row for row in fresh["comparisons"]}
    prior_rows = {key(row): row for row in previous["comparisons"]}
    if len(fresh_rows) != 32 or len(prior_rows) != 32 or set(fresh_rows) != set(prior_rows):
        raise ValueError("saved direction or axis identities differ")
    if any(exact_pair(fresh_rows[k]["native"]) != exact_pair(prior_rows[k]["native"])
           for k in fresh_rows):
        raise ValueError("old and fixed audits refer to different saved bounds")
    count = sum(row["inside"] for row in fresh_rows.values())
    if count != expected_count:
        raise ValueError(f"fixed interval comparison count changed to {count}")
    changed = [list(k) for k in fresh_rows
               if fresh_rows[k]["inside"] != prior_rows[k]["inside"]]
    if changed:
        raise ValueError(f"old and fixed Boolean verdicts differ: {changed}")
    return fresh, {
        "old_historical_inside": sum(row["inside"] for row in prior_rows.values()),
        "fixed_outward_inside": count,
        "total": 32,
        "changed_boolean_keys": changed,
        "failed_keys": [list(key(row)) for row in fresh["comparisons"] if not row["inside"]],
        "saved_csv_bounds_checked": csv_bounds(root),
    }


def algebraic_recheck(repaired):
    box, matrix, controls = algebra.rpc_contract()
    control_bounds = [algebra.linear_range(row, box, center, radius)
                      for row, (center, radius) in zip(matrix, controls)]
    other, bootstrap = algebra.check_bootstrap(box, control_bounds)
    h, inertia = algebra.H, algebra.INERTIA
    exact = {
        ("terminal_pre", 10): tuple(h * value / inertia for value in control_bounds[1]),
        ("terminal_composed", 10): tuple(h * value / inertia for value in control_bounds[1]),
        ("terminal_pre", 11): tuple(h * value / inertia for value in control_bounds[2]),
        ("terminal_composed", 11): tuple(h * value / inertia for value in control_bounds[2]),
    }
    coeff6 = [-h * value / algebra.MASS for value in matrix[0]]
    coeff6[5] += 1
    center1, radius1 = controls[0]
    base6 = algebra.linear_range(coeff6, box, -h * center1 / algebra.MASS,
                                 h * radius1 / algebra.MASS)
    error6 = h * other
    exact[("terminal_composed", 6)] = (base6[0] - error6, base6[1] + error6)
    native, raw_native = algebra.native_intervals()
    fixed = {key(row): row for row in repaired["comparisons"]}
    failing = {k for k, row in fixed.items() if not row["inside"]}
    if failing != set(exact):
        raise ValueError("five exact algebraic keys do not match fixed gaps")
    rows = []
    for k, bounds in exact.items():
        if exact_pair(fixed[k]["native"]) != native[k]:
            raise ValueError("algebraic and fixed saved bounds differ")
        inside = algebra.subset(bounds, native[k])
        if not inside:
            raise ValueError(f"exact algebraic comparison rejected at {k}")
        rows.append({
            "kind": k[0], "coord": k[1],
            "physical_bound_exact": [str(value) for value in bounds],
            "saved_outward_exact": [str(value) for value in native[k]],
            "inside_outward": inside,
            "inside_unexpanded_binary64": algebra.subset(bounds, raw_native[k]),
            "pre_column_is_not_pre_set_proof": k[0] == "terminal_pre",
        })
    physical = [k for k in fixed if k[0] != "terminal_pre"]
    if len(physical) != 20 or not all(fixed[k]["inside"] or k in exact for k in physical):
        raise ValueError("20 composed physical comparisons not established")
    return {
        "exact_algebraic_comparisons": rows,
        "all_five_inside_outward": True,
        "fixed_picard_plus_exact_algebraic_numeric_columns_inside": 32,
        "composed_physical_state_comparisons_inside": 20,
        "composed_physical_state_comparisons_total": 20,
        "pre_symbolic_set_inclusion_proven": False,
        "strict_bootstrap": {name: [str(value) for value in pair]
                             for name, pair in bootstrap.items()},
    }


def run():
    checks = method_check()
    original, original_result = recheck_arm(
        ORIGINAL, PRIOR / "original_AUDIT.json", 23)
    repaired, repaired_result = recheck_arm(
        REPAIRED, PRIOR / "repaired_AUDIT.json", 27)
    if json.loads((ORIGINAL / "on/rpc.json").read_text()) != json.loads(
            (REPAIRED / "quad/on/rpc.json").read_text()):
        raise ValueError("old original and repair RPC contracts differ")
    source_code = algebra.SOURCE.read_text()
    if "Interval init_x1(-0.4, 0.4), init_x2(-0.4, 0.4), init_x3(-0.4, 0.4)" not in source_code:
        raise ValueError("frozen QUAD source initial box identity changed")
    rpc = json.loads((ORIGINAL / "on/rpc.json").read_text())
    rpc_lowers = [F.from_float(float(value)) for value in rpc["params"]["input_lb"][:3]]
    source_lower = F.from_float(-0.4)
    gap = F(1, 18014398509481984)
    if any(rpc_lower - source_lower != gap for rpc_lower in rpc_lowers):
        raise ValueError("source-to-RPC lower one-ULP gap changed")
    algebraic = algebraic_recheck(repaired)
    summary = {
        "schema": "native-quad-legacy-narrow-rpc-fixed-decimal-recheck-v1",
        "status": "completed_saved_evidence_recheck",
        "original_library": original_result,
        "var_tail_copy": repaired_result,
        "repaired_exact_algebraic": algebraic,
        "source_first_box_x1_x3_lower_binary64": str(float(source_lower)),
        "saved_rpc_x1_x3_lower_binary64": str(float(rpc_lowers[0])),
        "omitted_source_lower_strip_exact": str(gap),
        "float32_residual_channels_exact_center_radius_per_arm": 3,
        "source_first_box_fully_covered_by_rpc": False,
        "scope": "saved narrow first-call RPC domain, one h=0.005 paper-ODE step, saved float32 affine-plus-residual controls",
        "limits": "No source-full-first-box certificate, pre symbolic-set proof, CROWN/NN certificate, Flow* parser/runtime certificate, other boxes or steps, or T=5 property; native production gate CLOSED",
    }
    for name, value in (("METHOD_SELF_CHECK.json", checks),
                        ("ORIGINAL_FIXED_AUDIT.json", original),
                        ("REPAIRED_FIXED_AUDIT.json", repaired),
                        ("RESULT.json", summary)):
        (HERE / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"original": original_result["fixed_outward_inside"],
                      "repaired": repaired_result["fixed_outward_inside"],
                      "physical": algebraic["composed_physical_state_comparisons_inside"],
                      "source_box_covered": False}))


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        (HERE / "BLOCKED.json").write_text(json.dumps({
            "status": "stopped_on_first_uncertain_condition",
            "reason": f"{type(exc).__name__}: {exc}",
        }, indent=2) + "\n")
        raise
