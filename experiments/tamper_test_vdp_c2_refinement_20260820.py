#!/usr/bin/env python3
"""Fail-closed structural verifier and tamper suite for C2 refinement ledgers."""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class RefinementLedgerError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RefinementLedgerError(message)


def _exact_binary64(value: Any) -> Fraction:
    return Fraction.from_float(float(value))


def _oracle_interval_matches(
    production_interval: Mapping[str, Any],
    lower: Any,
    upper: Any,
) -> bool:
    return (
        Fraction(str(production_interval["lower"])) == _exact_binary64(lower)
        and Fraction(str(production_interval["upper"])) == _exact_binary64(upper)
    )


def _flowstar_binary64_ratio(old_lo: float, old_hi: float, new_lo: float, new_hi: float) -> float:
    if old_lo == old_hi:
        return math.nan
    if new_lo == new_hi:
        return 0.0
    old_width = math.nextafter(old_hi - old_lo, math.inf)
    new_width = math.nextafter(new_hi - new_lo, math.inf)
    return math.nextafter(new_width / old_width, math.inf)


def verify_refinement_ledger(
    rows: Sequence[Mapping[str, Any]],
    oracle: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    _require(bool(rows), "empty refinement ledger")
    iterations = [int(row["refinement_iteration"]) for row in rows]
    _require(iterations == list(range(1, len(rows) + 1)), "iteration sequence")
    _require(len(set(iterations)) == len(iterations), "iteration reuse")
    polynomial_hashes = {str(row["retained_polynomial_sha256"]) for row in rows}
    _require(len(polynomial_hashes) == 1, "retained polynomial hash changed")
    _require(
        polynomial_hashes == {gate["candidate_polynomial_sha256"]},
        "gate/ledger polynomial hash",
    )
    oracle_rows = oracle["iterations"]
    _require(oracle["all_contained"] is True, "oracle global containment")
    _require(
        [int(row["iteration"]) for row in oracle_rows] == iterations,
        "oracle iteration coverage",
    )
    _require(all(row["all_contained"] is True for row in oracle_rows), "oracle containment")
    for index, row in enumerate(rows):
        _require(row["phase"] == "post_accept_refinement", "phase")
        _require(row["committed"] is True, "uncommitted packaged Gate-A row")
        _require(row["flowstar_max_refinement_steps_macro"] == 490, "refinement cap")
        _require(row["flowstar_refinement_replay_limit"] == 491, "replay limit")
        _require(row["flowstar_stop_ratio"] == 0.99, "stop ratio")
        components = row["components"]
        _require(len(components) == 2, "component count")
        _require(
            [int(component["component"]) for component in components] == [0, 1],
            "component order",
        )
        _require(all(component["subset"] is True for component in components), "forged subset")
        _require(all(float(component["subset_margin"]) >= 0 for component in components), "subset margin")
        _require(row["proposed_remainder_lo"] == row["retained_remainder_lo"], "retained lower")
        _require(row["proposed_remainder_hi"] == row["retained_remainder_hi"], "retained upper")
        oracle_components = oracle_rows[index]["components"]
        _require(
            [int(component["component"]) for component in oracle_components] == [0, 1],
            "oracle component order",
        )
        for component_index, (component, oracle_component) in enumerate(
            zip(components, oracle_components)
        ):
            input_lo = row["input_remainder_lo"][0][component_index]
            input_hi = row["input_remainder_hi"][0][component_index]
            proposed_lo = row["proposed_remainder_lo"][0][component_index]
            proposed_hi = row["proposed_remainder_hi"][0][component_index]
            _require(component["input_interval"] == [input_lo, input_hi], "component input")
            _require(component["output_interval"] == [proposed_lo, proposed_hi], "component output")
            _require(input_lo <= proposed_lo <= proposed_hi <= input_hi, "recomputed subset")
            _require(
                float(component["subset_margin"])
                == min(proposed_lo - input_lo, input_hi - proposed_hi),
                "recomputed subset margin",
            )
            expected_ratio = _flowstar_binary64_ratio(
                input_lo, input_hi, proposed_lo, proposed_hi
            )
            recorded_ratio = float(component["width_ratio_new_over_old"])
            _require(
                (math.isnan(expected_ratio) and math.isnan(recorded_ratio))
                or expected_ratio == recorded_ratio,
                "recomputed width ratio",
            )
            _require(
                _oracle_interval_matches(
                    oracle_component["production_final_image"], proposed_lo, proposed_hi
                ),
                "oracle final image identity",
            )
            _require(
                _oracle_interval_matches(
                    oracle_component["production_raw_rhs_remainder"],
                    row["raw_rhs_remainder_lo"][0][component_index],
                    row["raw_rhs_remainder_hi"][0][component_index],
                ),
                "oracle raw RHS identity",
            )
            _require(
                _oracle_interval_matches(
                    oracle_component["production_poly_diff"],
                    row["poly_diff_range_lo"][0][component_index],
                    row["poly_diff_range_hi"][0][component_index],
                ),
                "oracle poly_diff identity",
            )
        reason = str(row["stop_reason"])
        _require(
            reason in {"continue", "stop_ratio", "fixed_point", "max_refinement_replays_reached"},
            "stop reason",
        )
        ratios = [float(component["width_ratio_new_over_old"]) for component in components]
        equal_vector = row["proposed_remainder_lo"] == row["input_remainder_lo"] and row[
            "proposed_remainder_hi"
        ] == row["input_remainder_hi"]
        expected_continue = any(math.isfinite(ratio) and ratio <= 0.99 for ratio in ratios)
        if equal_vector:
            _require(reason == "fixed_point" and row["continue_refining"] is False, "fixed point stop")
        elif reason == "max_refinement_replays_reached":
            _require(index + 1 == 491 and row["continue_refining"] is False, "cap stop")
        else:
            _require(row["continue_refining"] is expected_continue, "stop-ratio continuation")
            _require(
                reason == ("continue" if expected_continue else "stop_ratio"),
                "stop-ratio reason",
            )
        if index + 1 < len(rows):
            next_row = rows[index + 1]
            _require(reason == "continue", "early stop reason")
            _require(next_row["input_remainder_lo"] == row["retained_remainder_lo"], "lower chain")
            _require(next_row["input_remainder_hi"] == row["retained_remainder_hi"], "upper chain")
        else:
            _require(reason != "continue", "unterminated continuation")
    last = rows[-1]
    _require(
        int(gate["final_remainder_ledger_iteration"]) == int(last["refinement_iteration"]),
        "final iteration",
    )
    _require(gate["final_remainder_ledger_matches_last_commit"] is True, "final ledger gate")
    return {"status": "verified", "iterations": len(rows), "final_stop_reason": last["stop_reason"]}


def run(gate_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (gate_dir / "refinement_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    oracle = json.loads((gate_dir / "exact_fraction_bernstein_oracle.json").read_text(encoding="utf-8"))
    gate = json.loads((gate_dir / "gate_a.json").read_text(encoding="utf-8"))
    verify_refinement_ledger(rows, oracle, gate)
    cases = []

    def rejected(name: str, changed_rows, changed_oracle=oracle, changed_gate=gate) -> None:
        try:
            verify_refinement_ledger(changed_rows, changed_oracle, changed_gate)
        except RefinementLedgerError as exc:
            cases.append({"case": name, "rejected": True, "message": str(exc)})
            return
        raise AssertionError(f"tamper case was accepted: {name}")

    deleted = deepcopy(rows)
    del deleted[1]
    rejected("delete_iteration", deleted)

    reordered = deepcopy(rows)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    rejected("reorder_iterations", reordered)

    forged = deepcopy(rows)
    forged[0]["components"][0]["subset"] = False
    rejected("forge_subset", forged)

    reused = deepcopy(rows)
    reused[1]["refinement_iteration"] = reused[0]["refinement_iteration"]
    rejected("reuse_iteration", reused)

    stop = deepcopy(rows)
    stop[0]["stop_reason"] = "stop_ratio"
    rejected("modify_stop_reason", stop)

    final_gate = deepcopy(gate)
    final_gate["final_remainder_ledger_iteration"] = int(final_gate["final_remainder_ledger_iteration"]) - 1
    rejected("replace_final_ledger", deepcopy(rows), changed_gate=final_gate)

    swapped = deepcopy(rows)
    swapped[0]["components"][0], swapped[0]["components"][1] = (
        swapped[0]["components"][1],
        swapped[0]["components"][0],
    )
    rejected("swap_components", swapped)

    partial = deepcopy(rows)
    partial[0]["retained_remainder_lo"][0][1] = partial[0]["input_remainder_lo"][0][1]
    partial[0]["retained_remainder_hi"][0][1] = partial[0]["input_remainder_hi"][0][1]
    rejected("partial_commit", partial)

    stale = deepcopy(rows)
    stale[1]["raw_rhs_remainder_lo"] = deepcopy(stale[0]["raw_rhs_remainder_lo"])
    stale[1]["raw_rhs_remainder_hi"] = deepcopy(stale[0]["raw_rhs_remainder_hi"])
    rejected("stale_cache", stale)

    wrong_ratio = deepcopy(rows)
    wrong_ratio[0]["flowstar_stop_ratio"] = 0.9
    rejected("wrong_stop_ratio", wrong_ratio)

    result = {
        "schema": "vdp_c2_refinement_tamper_tests_v1",
        "passed": True,
        "cases": cases,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_dir", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args().gate_dir)
