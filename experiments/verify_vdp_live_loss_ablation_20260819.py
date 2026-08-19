#!/usr/bin/env python3
"""Fail closed on the automatically derived VDP live-loss and four-cell ledgers."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class VerificationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _interval(record: Mapping[str, Any]) -> tuple[Fraction, Fraction]:
    return (
        Fraction.from_float(float(record["lo"]["decimal"])),
        Fraction.from_float(float(record["hi"]["decimal"])),
    )


def _necessary(record: Mapping[str, Any]) -> tuple[Fraction, Fraction]:
    return Fraction(record["lo"]), Fraction(record["hi"])


def _path(
    start: str,
    targets: set[str],
    children: Mapping[str, Sequence[str]],
) -> list[str]:
    queue: list[tuple[str, list[str]]] = [(start, [start])]
    seen = {start}
    while queue:
        current, path = queue.pop(0)
        if current in targets:
            return path
        for child in children.get(current, ()):
            if child not in seen:
                seen.add(child)
                queue.append((child, [*path, child]))
    return []


def _same_row(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    live = _load(root / "live_loss_ledger.json")
    matrix = _load(root / "gate_b_same_input_matrix.json")
    gate_c = _load(root / "gate_c1_joint_closure.json")
    production = _load(root / "production_operator_ledger.json")
    summary = _load(root / "summary.json")

    _require(live["schema"] == "vdp_live_loss_production_event_ledger_v1", "live schema")
    _require(matrix["schema"] == "vdp_h2_same_input_four_cell_gate_b_v2", "four-cell schema")
    _require(summary["schema"] == "vdp_live_loss_ablation_gate_summary_v2", "summary schema")
    _require("first_extra_enclosure" not in summary, "obsolete hard-coded first loss is present")

    cells = production["gate_b_cells"]
    _require(set(cells) == {"L0", "L1", "L2", "L3"}, "four-cell coverage")
    _require(
        len({cell["same_input_sha256"] for cell in cells.values()}) == 1,
        "four-cell input hashes differ",
    )
    expected_contracts = {
        "L0": ("distributed", "generic"),
        "L1": ("factorized", "generic"),
        "L2": ("distributed", "joint"),
        "L3": ("factorized", "joint"),
    }
    for name, (expression, square) in expected_contracts.items():
        cell = cells[name]
        _require(cell["expression_graph"] == expression, f"expression graph: {name}")
        _require(cell["square_operator"] == square, f"square operator: {name}")
        _require(
            all(row["oracle"]["all_components_contained"] for row in cell["operator_stages"]),
            f"exact operator containment: {name}",
        )
        _require(
            all(
                row["production_contains_exact_bernstein"]
                for row in cell["poly_diff_range_bound"]["components"]
            ),
            f"poly_diff exact containment: {name}",
        )

    _require(gate_c["schema"] == "vdp_gate_c1_joint_factor_times_y_closure_v1", "Gate C schema")
    _require(gate_c["candidate_id"] == "C1", "Gate C candidate identity")
    _require(
        gate_c["same_input_sha256"] == cells["L0"]["same_input_sha256"],
        "Gate C input differs from Gate B",
    )
    exact_lo, exact_hi = _necessary(gate_c["exact_binary64_rational_bernstein"])
    production_lo, production_hi = _necessary(gate_c["production_interval"])
    _require(
        production_lo <= exact_lo <= exact_hi <= production_hi,
        "Gate C production interval does not contain exact oracle",
    )
    _require(gate_c["production_contains_exact_oracle"] is True, "Gate C containment claim")
    _require(
        Fraction(gate_c["fraction_of_h2_vs_exact_excess_removed"]) >= Fraction(1, 10),
        "Gate C promotion is below 10%",
    )
    _require(gate_c["promotion_at_least_10_percent"] is True, "Gate C promotion claim")
    _require(gate_c["retained_picard_polynomial_bitwise_unchanged"] is True, "Gate C changed polynomial")
    _require(gate_c["segment_no_regression_vs_h2"] is True, "Gate C segment regression")
    _require(gate_c["endpoint_no_regression_vs_h2"] is True, "Gate C endpoint regression")
    _require(gate_c["gate_c_pass"] is True, "Gate C micro gate")

    widths = matrix["cell_stage_widths"]
    for name in expected_contracts:
        _require(widths[name] == _cell_widths(cells[name]), f"stored cell widths: {name}")
    for metric, rows in matrix["factorial_effects"].items():
        for row in rows:
            component = int(row["component"])
            l0, l1, l2, l3 = (
                float(widths[name][metric][component]) for name in ("L0", "L1", "L2", "L3")
            )
            _require(
                row["factorization_main_reduction"] == ((l0 - l1) + (l2 - l3)) / 2.0,
                f"factorization effect: {metric}/{component}",
            )
            _require(
                row["joint_square_main_reduction"] == ((l0 - l2) + (l1 - l3)) / 2.0,
                f"joint-square effect: {metric}/{component}",
            )
            _require(
                row["interaction_reduction"] == l1 - l3 - l0 + l2,
                f"interaction effect: {metric}/{component}",
            )

    rows = live["rows"]
    _require(rows, "empty live-loss ledger")
    _require([row["sequence"] for row in rows] == list(range(1, len(rows) + 1)), "event order")
    stage_ids = [row["stage_id"] for row in rows]
    _require(len(stage_ids) == len(set(stage_ids)), "duplicate stage id")
    _require(
        stage_ids
        == [row["stage_id"] for row in cells["L0"]["production_raw_trace"]["execution_events"]],
        "live ledger is not the complete production trace",
    )
    seen: set[str] = set()
    children: dict[str, list[str]] = {stage_id: [] for stage_id in stage_ids}
    for row in rows:
        _require(row["input_stages"] == row["parent_stage_ids"], f"input aliases: {row['stage_id']}")
        _require(all(parent in seen for parent in row["parent_stage_ids"]), f"non-prior parent: {row['stage_id']}")
        for parent in row["parent_stage_ids"]:
            children[parent].append(row["stage_id"])
        seen.add(row["stage_id"])
    subset_targets = {row["stage_id"] for row in rows if row["operation"] == "subset_test"}
    _require(len(subset_targets) == 2, "subset target coverage")

    for row in rows:
        prod_lo, prod_hi = _interval(row["production_interval"])
        exact_lo, exact_hi = _necessary(row["necessary_enclosure"])
        _require(prod_lo <= exact_lo <= exact_hi <= prod_hi, f"necessary containment: {row['stage_id']}")
        surplus = (prod_hi - prod_lo) - (exact_hi - exact_lo)
        _require(Fraction(row["exact_surplus"]) == surplus, f"exact surplus: {row['stage_id']}")
        consumer_path = _path(row["stage_id"], subset_targets, children)
        _require(row["consumer_chain_to_final_subset"] == consumer_path, f"consumer chain: {row['stage_id']}")
        _require(row["final_subset_width_live"] is bool(consumer_path), f"live flag: {row['stage_id']}")
        material = bool(consumer_path) and Fraction(
            row["same_input_marginal"]["final_subset_width_reduction"]
        ) > 0
        _require(row["live_material"] is material, f"material flag: {row['stage_id']}")

    strict = [
        row
        for row in rows
        if row["loss_classification_eligible"] and Fraction(row["exact_surplus"]) > 0
    ]
    live_strict = [row for row in strict if row["final_subset_width_live"]]
    material = [row for row in rows if row["live_material"]]
    _require(strict and live_strict and material, "required automatic classifications")
    ranked = sorted(
        (
            row
            for row in rows
            if Fraction(row["same_input_marginal"]["final_subset_width_reduction"]) > 0
        ),
        key=lambda row: Fraction(row["same_input_marginal"]["final_subset_width_reduction"]),
        reverse=True,
    )
    classifications = {
        "first_syntactic_strict_surplus": strict[0],
        "first_live_strict_surplus": live_strict[0],
        "first_live_material_surplus": material[0],
        "largest_same_input_marginal_contributor": ranked[0],
    }
    for key, expected in classifications.items():
        _require(_same_row(live[key], expected), f"live classification: {key}")
        _require(_same_row(summary[key], expected), f"summary classification: {key}")

    baseline_final = Fraction.from_float(float(widths["L0"]["final_subset_image"][1]))
    expected_marginals = {
        "x_squared_generic": baseline_final
        - Fraction.from_float(float(widths["L2"]["final_subset_image"][1])),
        "distributed_final": baseline_final
        - Fraction.from_float(float(widths["L1"]["final_subset_image"][1])),
    }
    for marker, expected in expected_marginals.items():
        row = next(row for row in rows if marker in row["stage_id"])
        _require(
            Fraction(row["same_input_marginal"]["final_subset_width_reduction"]) == expected,
            f"same-input marginal: {marker}",
        )

    eps_rows = [row for row in rows if row["validation_eps_payment"] is not None]
    payments = live["rounding_proof"]["payments"]
    payment_ids = [row["payment_id"] for row in payments]
    _require(len(payment_ids) == len(set(payment_ids)), "validation_eps payment reused")
    _require(payment_ids == [row["stage_id"] for row in eps_rows], "validation_eps payment coverage")
    _require(live["rounding_proof"]["payment_ids_unique"] is True, "rounding uniqueness claim")
    _require(
        live["rounding_proof"]["final_subset_events_contain_complete_exact_chain"] is True,
        "end-to-end rounding containment",
    )
    _require(
        summary["gate_a_pass"] is True
        and summary["gate_b_pass"] is True
        and summary["gate_c_pass"] is True,
        "Gate A/B/C",
    )
    _require(summary["production_candidate"]["candidate_id"] == "C1", "summary candidate")
    result = {
        "status": "verified",
        "events": len(rows),
        "cells": len(cells),
        "first_syntactic": strict[0]["stage_id"],
        "first_live": live_strict[0]["stage_id"],
        "first_material": material[0]["stage_id"],
        "largest_marginal": ranked[0]["stage_id"],
        "validation_eps_payments": len(payment_ids),
        "gate_c_promotion": gate_c["fraction_of_h2_vs_exact_excess_removed"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _width(record: Mapping[str, Any]) -> float:
    return float(record["hi"]["decimal"]) - float(record["lo"]["decimal"])


def _cell_widths(cell: Mapping[str, Any]) -> dict[str, list[float]]:
    stages = cell["operator_stages"]
    square = next(
        row for row in stages if ".x_squared_" in row["stage_id"] and "times_y" not in row["stage_id"]
    )
    product = next(
        row for row in stages if row["stage_id"].endswith((".x_squared_times_y", ".factor_times_y"))
    )
    return {
        "x_squared": [_width(square["model"]["components"][0]["ordinary_remainder"])],
        "cubic_or_factor_times_y": [_width(product["model"]["components"][0]["ordinary_remainder"])],
        "raw_rhs": [_width(row) for row in cell["raw_rhs_remainder"]],
        "tau_scaled_raw_rhs": [_width(row) for row in cell["tau_times_raw_rhs"]["after_validation_eps"]],
        "poly_diff": [_width(row) for row in cell["poly_diff_validation_eps"]["after"]],
        "final_subset_image": [_width(row) for row in cell["final_validation_eps"]["after"]],
        "segment": [_width(row) for row in cell["segment"]],
        "endpoint": [_width(row) for row in cell["endpoint"]],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    verify(parse_args().root)
