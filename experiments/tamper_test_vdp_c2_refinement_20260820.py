#!/usr/bin/env python3
"""Fail-closed structural verifier and tamper suite for C2 refinement ledgers."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class RefinementLedgerError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RefinementLedgerError(message)


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
    for index, row in enumerate(rows):
        _require(row["phase"] == "post_accept_refinement", "phase")
        _require(row["committed"] is True, "uncommitted packaged Gate-A row")
        components = row["components"]
        _require(len(components) == 2, "component count")
        _require(all(component["subset"] is True for component in components), "forged subset")
        _require(all(float(component["subset_margin"]) >= 0 for component in components), "subset margin")
        _require(row["proposed_remainder_lo"] == row["retained_remainder_lo"], "retained lower")
        _require(row["proposed_remainder_hi"] == row["retained_remainder_hi"], "retained upper")
        reason = str(row["stop_reason"])
        _require(
            reason in {"continue", "stop_ratio", "fixed_point", "max_refinement_replays_reached"},
            "stop reason",
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
    oracle_rows = oracle["iterations"]
    _require(oracle["all_contained"] is True, "oracle global containment")
    _require(
        [int(row["iteration"]) for row in oracle_rows] == iterations,
        "oracle iteration coverage",
    )
    _require(all(row["all_contained"] is True for row in oracle_rows), "oracle containment")
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
