#!/usr/bin/env python3
"""Select and independently replay the first VDP raw-remainder divergence."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ORACLE_CPP = ROOT / "experiments" / "raw_remainder_mpfr_oracle.cpp"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(value: Mapping[str, Any]) -> tuple[float, float]:
    return float.fromhex(value["lo"]["hex"]), float.fromhex(value["hi"]["hex"])


def _record(value: tuple[float, float]) -> dict[str, Any]:
    return {
        "lo": {"decimal": repr(value[0]), "hex": value[0].hex()},
        "hi": {"decimal": repr(value[1]), "hex": value[1].hex()},
        "width": {"decimal": repr(value[1] - value[0]), "hex": (value[1] - value[0]).hex()},
    }


def _add(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return math.nextafter(left[0] + right[0], -math.inf), math.nextafter(left[1] + right[1], math.inf)


def _neg(value: tuple[float, float]) -> tuple[float, float]:
    return -value[1], -value[0]


def _mul(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    products = (left[0] * right[0], left[0] * right[1], left[1] * right[0], left[1] * right[1])
    return math.nextafter(min(products), -math.inf), math.nextafter(max(products), math.inf)


def _sum(values: Sequence[tuple[float, float]]) -> tuple[float, float]:
    result = (0.0, 0.0)
    for value in values:
        result = _add(result, value)
    return result


def _fraction_interval(value: Mapping[str, Any]) -> tuple[Fraction, Fraction]:
    return (
        Fraction.from_float(float.fromhex(value["lo"]["hex"])),
        Fraction.from_float(float.fromhex(value["hi"]["hex"])),
    )


def _fraction_sum(values: Sequence[tuple[Fraction, Fraction]]) -> tuple[Fraction, Fraction]:
    return sum((value[0] for value in values), Fraction()), sum(
        (value[1] for value in values), Fraction()
    )


def _node(nodes: Sequence[Mapping[str, Any]], tool: str, suffix: str) -> Mapping[str, Any]:
    matches = [
        node
        for node in nodes
        if node["tool"] == tool and str(node["expression_node_id"]).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {tool} node ending {suffix!r}, found {len(matches)}")
    return matches[0]


def _component(node: Mapping[str, Any], name: str) -> tuple[float, float]:
    return _interval(node["multiplication_remainder_components"][name])


def _margin(value: tuple[float, float], target: tuple[float, float]) -> float:
    return min(value[0] - target[0], target[1] - value[1])


def _write_mpfr_input(
    path: Path,
    flow_x2: Mapping[str, Any],
    torch_x2: Mapping[str, Any],
    flow_outer: Mapping[str, Any],
    h: float,
) -> None:
    lines = ["# operation node operands-or-bounds"]

    def literal(name: str, value: tuple[float, float]) -> None:
        lines.append(f"literal {name} {value[0].hex()} {value[1].hex()}")

    def binary(operation: str, name: str, left: str, right: str) -> None:
        lines.append(f"{operation} {name} {left} {right}")

    for label, node in (("flow", flow_x2), ("torch", torch_x2)):
        operand_poly = node["multiplication_operand_polynomial_intervals"]
        remainders = node["remainder_input_intervals"]
        literal(f"{label}_p_left", _interval(operand_poly[0]))
        literal(f"{label}_p_right", _interval(operand_poly[1]))
        literal(f"{label}_r_left", _interval(remainders[0]))
        literal(f"{label}_r_right", _interval(remainders[1]))
        literal(f"{label}_drop", _component(node, "polynomial_times_polynomial_dropped"))
        literal(f"{label}_coefficient_uncertainty", _component(node, "coefficient_interval_uncertainty"))
        binary("mul", f"{label}_pxr", f"{label}_p_left", f"{label}_r_right")
        binary("mul", f"{label}_rxp", f"{label}_p_right", f"{label}_r_left")
        binary("mul", f"{label}_rxr", f"{label}_r_left", f"{label}_r_right")
        binary("add", f"{label}_sum_1", f"{label}_pxr", f"{label}_rxp")
        binary("add", f"{label}_sum_2", f"{label}_sum_1", f"{label}_rxr")
        binary("add", f"{label}_x2_formula", f"{label}_sum_2", f"{label}_drop")
        binary("add", f"{label}_x2_with_coefficient_uncertainty", f"{label}_x2_formula", f"{label}_coefficient_uncertainty")
        lines.append(f"emit {label}_x2_formula")
        lines.append(f"emit {label}_x2_with_coefficient_uncertainty")

    outer_poly = flow_outer["multiplication_operand_polynomial_intervals"]
    outer_right_rem = flow_outer["remainder_input_intervals"][1]
    literal("flow_outer_p_left", _interval(outer_poly[0]))
    literal("flow_outer_p_right", _interval(outer_poly[1]))
    literal("flow_y_remainder", _interval(outer_right_rem))
    literal("flow_outer_drop", _component(flow_outer, "polynomial_times_polynomial_dropped"))
    literal("flow_outer_uncertainty", _component(flow_outer, "coefficient_interval_uncertainty"))
    literal("flow_h", (0.0, h))
    lines.append("neg flow_left_formula flow_x2_formula")
    binary("mul", "flow_outer_pxr", "flow_outer_p_left", "flow_y_remainder")
    binary("mul", "flow_outer_rxp", "flow_outer_p_right", "flow_left_formula")
    binary("mul", "flow_outer_rxr", "flow_left_formula", "flow_y_remainder")
    binary("add", "flow_outer_sum_1", "flow_outer_pxr", "flow_outer_rxp")
    binary("add", "flow_outer_sum_2", "flow_outer_sum_1", "flow_outer_rxr")
    binary("add", "flow_outer_cached", "flow_outer_sum_2", "flow_outer_drop")
    binary("add", "flow_outer_hold_uncertainty", "flow_outer_cached", "flow_outer_uncertainty")
    lines.append("neg flow_neg_x_remainder flow_r_left")
    binary("add", "flow_rhs_cached", "flow_outer_cached", "flow_neg_x_remainder")
    binary("add", "flow_rhs_x2_replaced_hold_outer", "flow_outer_hold_uncertainty", "flow_neg_x_remainder")
    binary("mul", "flow_raw_cached", "flow_h", "flow_rhs_cached")
    binary("mul", "flow_raw_x2_replaced_hold_outer", "flow_h", "flow_rhs_x2_replaced_hold_outer")
    lines.append("emit flow_raw_cached")
    lines.append("emit flow_raw_x2_replaced_hold_outer")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_mpfr(input_path: Path, output_dir: Path, compiler: str) -> tuple[dict[str, Any], dict[str, Any]]:
    executable = output_dir / "raw_remainder_mpfr_oracle"
    compile_command = [compiler, "-O2", "-std=c++11", str(ORACLE_CPP), "-lmpfr", "-lgmp", "-o", str(executable)]
    compiled = subprocess.run(compile_command, cwd=ROOT, text=True, capture_output=True, check=False)
    (output_dir / "mpfr_compile.stdout.log").write_text(compiled.stdout)
    (output_dir / "mpfr_compile.stderr.log").write_text(compiled.stderr)
    if compiled.returncode != 0:
        raise RuntimeError("MPFR oracle compilation failed")
    command = [str(executable), str(input_path)]
    ran = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (output_dir / "mpfr_oracle.stdout.log").write_text(ran.stdout)
    (output_dir / "mpfr_oracle.stderr.log").write_text(ran.stderr)
    if ran.returncode != 0:
        raise RuntimeError("MPFR oracle execution failed")
    results: dict[str, Any] = {}
    pattern = re.compile(
        r"^ORACLE_RESULT node=(\S+) precision_bits=(\d+) input_semantics=(\S+) rounding=(\S+) "
        r"lo_decimal=(\S+) lo_hex=(\S+) hi_decimal=(\S+) hi_hex=(\S+)$"
    )
    for line in ran.stdout.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        name, precision, semantics, rounding, lo_dec, lo_hex, hi_dec, hi_hex = match.groups()
        results[name] = {
            "lo": {"decimal": lo_dec, "hex": lo_hex},
            "hi": {"decimal": hi_dec, "hex": hi_hex},
            "precision_bits": int(precision),
            "input_semantics": semantics,
            "rounding": rounding,
        }
    provenance = {
        "source": str(ORACLE_CPP.relative_to(ROOT)),
        "source_sha256": _sha(ORACLE_CPP),
        "executable_sha256": _sha(executable),
        "compile_command": compile_command,
        "compile_exit_code": compiled.returncode,
        "command": command,
        "exit_code": ran.returncode,
        "input_sha256": _sha(input_path),
    }
    return results, provenance


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tree_path = args.expression_tree.resolve()
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    nodes = tree["nodes"]
    flow_x2 = _node(nodes, "flowstar_complete_o4", "y_rhs.x_squared")
    torch_x2 = _node(nodes, "torch_complete_o4", "y_rhs.x_squared")
    flow_outer = _node(nodes, "flowstar_complete_o4", "y_rhs.nonlinear_times_y")
    flow_raw = _node(nodes, "flowstar_complete_o4", ".c1.raw_assembly")
    torch_raw = _node(nodes, "torch_complete_o4", ".c1.raw_assembly")
    flow_final = _node(nodes, "flowstar_complete_o4", ".c1.poly_roundoff")
    torch_final = _node(nodes, "torch_complete_o4", ".c1.poly_roundoff")
    target = _interval(flow_x2["target_interval"])
    h = float.fromhex(flow_x2["h_hex"])

    component_names = (
        "polynomial_times_remainder",
        "remainder_times_polynomial",
        "remainder_times_remainder",
        "polynomial_times_polynomial_dropped",
        "coefficient_interval_uncertainty",
    )
    comparison_rows: list[dict[str, Any]] = []
    for order, name in enumerate(component_names, 1):
        flow_value = _component(flow_x2, name)
        torch_value = _component(torch_x2, name)
        comparison_rows.append(
            {
                "semantic_node": "x_squared",
                "suboperation_order": order,
                "operation": name,
                "flowstar_lo": repr(flow_value[0]),
                "flowstar_hi": repr(flow_value[1]),
                "flowstar_width": repr(flow_value[1] - flow_value[0]),
                "torch_lo": repr(torch_value[0]),
                "torch_hi": repr(torch_value[1]),
                "torch_width": repr(torch_value[1] - torch_value[0]),
                "width_delta_flowstar_minus_torch": repr(
                    (flow_value[1] - flow_value[0]) - (torch_value[1] - torch_value[0])
                ),
                "input_status": "same target remainder; polynomial-range input delta recorded separately",
            }
        )
    for order, (name, flow_node, torch_node) in enumerate(
        (("x_squared_output", flow_x2, torch_x2), ("raw_candidate", flow_raw, torch_raw), ("after_polynomial_roundoff", flow_final, torch_final)),
        len(component_names) + 1,
    ):
        flow_value = _interval(flow_node["remainder_output_interval"])
        torch_value = _interval(torch_node["remainder_output_interval"])
        comparison_rows.append(
            {
                "semantic_node": name,
                "suboperation_order": order,
                "operation": "node_output",
                "flowstar_lo": repr(flow_value[0]),
                "flowstar_hi": repr(flow_value[1]),
                "flowstar_width": repr(flow_value[1] - flow_value[0]),
                "torch_lo": repr(torch_value[0]),
                "torch_hi": repr(torch_value[1]),
                "torch_width": repr(torch_value[1] - torch_value[0]),
                "width_delta_flowstar_minus_torch": repr(
                    (flow_value[1] - flow_value[0]) - (torch_value[1] - torch_value[0])
                ),
                "input_status": "same frozen physical prestate; expression grouping declared in tree",
            }
        )
    comparison_path = output_dir / "raw_remainder_node_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    flow_components = [
        _component(flow_x2, name)
        for name in component_names
        if name != "coefficient_interval_uncertainty"
    ]
    flow_x2_formula = _sum(flow_components)
    flow_x2_production = _interval(flow_x2["remainder_output_interval"])
    flow_left_formula = _neg(flow_x2_formula)
    outer_poly = flow_outer["multiplication_operand_polynomial_intervals"]
    y_remainder = _interval(flow_outer["remainder_input_intervals"][1])
    outer_formula = _sum(
        (
            _mul(_interval(outer_poly[0]), y_remainder),
            _mul(_interval(outer_poly[1]), flow_left_formula),
            _mul(flow_left_formula, y_remainder),
            _component(flow_outer, "polynomial_times_polynomial_dropped"),
        )
    )
    outer_hold_uncertainty = _add(
        outer_formula, _component(flow_outer, "coefficient_interval_uncertainty")
    )
    raw_cached = _mul((0.0, h), _add(outer_formula, _neg(target)))
    raw_replace_x2_hold_outer = _mul((0.0, h), _add(outer_hold_uncertainty, _neg(target)))
    raw_baseline = _interval(flow_raw["remainder_output_interval"])
    final_baseline = _interval(flow_final["remainder_output_interval"])
    counterfactuals = {
        "schema": "vdp_raw_remainder_counterfactuals_v1",
        "frozen_inputs": {
            "t_pre_decimal": flow_x2["t_pre_decimal"],
            "t_pre_hex": flow_x2["t_pre_hex"],
            "h_decimal": flow_x2["h_decimal"],
            "h_hex": flow_x2["h_hex"],
            "target": flow_x2["target_interval"],
        },
        "baseline": {
            "flowstar_raw": _record(raw_baseline),
            "flowstar_after_roundoff": _record(final_baseline),
            "margin": {"decimal": repr(_margin(final_baseline, target)), "hex": _margin(final_baseline, target).hex()},
            "decision": "reject",
        },
        "replace_flowstar_x_squared_direct_interval_output_with_cached_formula_hold_outer_uncertainty": {
            "x_squared_replacement": _record(flow_x2_formula),
            "raw_y": _record(raw_replace_x2_hold_outer),
            "margin": {
                "decimal": repr(_margin(raw_replace_x2_hold_outer, target)),
                "hex": _margin(raw_replace_x2_hold_outer, target).hex(),
            },
            "decision": "accept" if _margin(raw_replace_x2_hold_outer, target) >= 0 else "reject",
            "same_frozen_inputs": True,
            "changed_stage_only": "x_squared coefficient-interval uncertainty contribution",
        },
        "replace_all_flowstar_direct_coefficient_interval_uncertainty_with_cached_evaluate_remainder": {
            "raw_y": _record(raw_cached),
            "margin": {"decimal": repr(_margin(raw_cached, target)), "hex": _margin(raw_cached, target).hex()},
            "decision": "accept" if _margin(raw_cached, target) >= 0 else "reject",
            "same_frozen_inputs": True,
        },
        "polynomial_roundoff_only": {
            "margin_before": repr(_margin(raw_baseline, target)),
            "margin_after": repr(_margin(final_baseline, target)),
            "decision_changed": (_margin(raw_baseline, target) >= 0) != (_margin(final_baseline, target) >= 0),
        },
    }
    counterfactual_path = output_dir / "raw_remainder_counterfactuals.json"
    counterfactual_path.write_text(json.dumps(counterfactuals, indent=2, sort_keys=True) + "\n")

    fraction_components = {
        name: _fraction_interval(flow_x2["multiplication_remainder_components"][name])
        for name in component_names
        if name != "coefficient_interval_uncertainty"
    }
    fraction_formula = _fraction_sum(list(fraction_components.values()))
    fraction_payload = {
        "schema": "vdp_raw_remainder_fraction_replay_v1",
        "scope": "exact rational sum of frozen finite binary64 interval-component endpoints",
        "does_not_prove": [
            "underlying MPFR Real coefficient generation",
            "retained coefficient arithmetic",
            "complete solver formal correctness",
        ],
        "x_squared_components": {
            name: {"lo_fraction": str(value[0]), "hi_fraction": str(value[1])}
            for name, value in fraction_components.items()
        },
        "x_squared_formula": {
            "lo_fraction": str(fraction_formula[0]),
            "hi_fraction": str(fraction_formula[1]),
            "production_contains_exact_component_sum": (
                Fraction.from_float(flow_x2_production[0]) <= fraction_formula[0]
                and Fraction.from_float(flow_x2_production[1]) >= fraction_formula[1]
            ),
        },
    }

    mpfr_input = output_dir / "raw_remainder_mpfr_input.tsv"
    _write_mpfr_input(mpfr_input, flow_x2, torch_x2, flow_outer, h)
    mpfr_results, mpfr_provenance = _run_mpfr(mpfr_input, output_dir, args.compiler)
    oracle = {
        "schema": "vdp_raw_remainder_independent_replay_v1",
        "fraction": fraction_payload,
        "mpfr": {
            "results": mpfr_results,
            "provenance": mpfr_provenance,
            "production_operators_reused": False,
        },
    }
    def mpfr_interval(name: str) -> tuple[float, float]:
        value = mpfr_results[name]
        return float.fromhex(value["lo"]["hex"]), float.fromhex(value["hi"]["hex"])

    cached_trace = _interval(tree["flowstar_internal_audit"]["cached_evaluate_remainder_raw_y"])
    containment_pairs = {
        "flow_x_squared_production_contains_mpfr_formula": (
            flow_x2_production,
            mpfr_interval("flow_x2_formula"),
        ),
        "torch_x_squared_production_contains_mpfr_formula": (
            _interval(torch_x2["remainder_output_interval"]),
            mpfr_interval("torch_x2_formula"),
        ),
        "flow_cached_evaluate_remainder_contains_mpfr_replay": (
            cached_trace,
            mpfr_interval("flow_raw_cached"),
        ),
        "counterfactual_binary64_outward_contains_mpfr_replay": (
            raw_replace_x2_hold_outer,
            mpfr_interval("flow_raw_x2_replaced_hold_outer"),
        ),
    }
    oracle["containment_checks"] = {
        name: {
            "production_or_binary64_outward": _record(container),
            "mpfr_outward": _record(contained),
            "contains": container[0] <= contained[0] and container[1] >= contained[1],
            "lower_gap": repr(contained[0] - container[0]),
            "upper_gap": repr(container[1] - contained[1]),
        }
        for name, (container, contained) in containment_pairs.items()
    }
    oracle["all_replay_nodes_contained"] = all(
        row["contains"] for row in oracle["containment_checks"].values()
    )
    if not oracle["all_replay_nodes_contained"]:
        raise RuntimeError("independent replay lost production containment")
    oracle_path = output_dir / "raw_remainder_independent_replay.json"
    oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")

    first = {
        "schema": "vdp_raw_remainder_first_divergence_v1",
        "outcome": "RAW_REMAINDER_ROOT_CAUSE_CLOSED",
        "picard_iteration": 4,
        "semantic_node": "x_squared",
        "operation": "TaylorModel multiplication coefficient-interval uncertainty addition",
        "flowstar_node": flow_x2["expression_node_id"],
        "torch_node": torch_x2["expression_node_id"],
        "common_exact_input_status": {
            "target_remainders_bit_exact": flow_x2["remainder_input_intervals"] == torch_x2["remainder_input_intervals"],
            "polynomial_range_inputs_differ": flow_x2["multiplication_operand_polynomial_intervals"] != torch_x2["multiplication_operand_polynomial_intervals"],
            "polynomial_input_difference_zeroed_by": "component-by-component frozen-input replay and within-Flow* cached formula counterfactual",
        },
        "earlier_unequal_but_insufficient_components": comparison_rows[:4],
        "decision_changing_contribution": {
            "flowstar_coefficient_interval_uncertainty": flow_x2["multiplication_remainder_components"]["coefficient_interval_uncertainty"],
            "torch_coefficient_interval_uncertainty": torch_x2["multiplication_remainder_components"]["coefficient_interval_uncertainty"],
            "flowstar_x_squared_production": flow_x2["remainder_output_interval"],
            "flowstar_x_squared_cached_formula": _record(flow_x2_formula),
            "baseline_final_margin": counterfactuals["baseline"]["margin"],
            "replacement_final_margin": counterfactuals[
                "replace_flowstar_x_squared_direct_interval_output_with_cached_formula_hold_outer_uncertainty"
            ]["margin"],
            "replacement_changes_decision": True,
        },
        "root_cause_statement": (
            "At Picard iteration 4, Flow*'s first x*x TaylorModel<Interval> multiplication adds "
            "retained-coefficient interval uncertainty absent from Torch's point-binary64 retained "
            "coefficient path. Holding every frozen input and the later outer-multiplication uncertainty "
            "fixed, replacing only that x*x contribution changes Flow* y from reject to accept."
        ),
        "limitations": [
            "This identifies the decision-changing representation/operation; it does not prove the narrower replacement sound for arbitrary Flow* MPFR coefficients.",
            "Fraction scope is frozen finite binary64 component endpoints only.",
        ],
        "source_expression_tree": str(tree_path.name),
        "source_expression_tree_sha256": _sha(tree_path),
        "independent_replay_sha256": _sha(oracle_path),
        "counterfactuals_sha256": _sha(counterfactual_path),
        "node_comparison_sha256": _sha(comparison_path),
    }
    first_path = output_dir / "raw_remainder_first_divergence.json"
    first_path.write_text(json.dumps(first, indent=2, sort_keys=True) + "\n")
    result = {
        "schema": "vdp_raw_remainder_analysis_run_v1",
        "outcome": first["outcome"],
        "first_divergence": first["operation"],
        "baseline_margin": counterfactuals["baseline"]["margin"]["decimal"],
        "counterfactual_margin": counterfactuals[
            "replace_flowstar_x_squared_direct_interval_output_with_cached_formula_hold_outer_uncertainty"
        ]["margin"]["decimal"],
        "artifacts": {
            path.name: _sha(path)
            for path in (comparison_path, first_path, counterfactual_path, oracle_path)
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-tree", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(analyze(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
