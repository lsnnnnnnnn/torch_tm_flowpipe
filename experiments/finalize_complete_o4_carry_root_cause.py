#!/usr/bin/env python3
"""Fail closed while deriving the complete-O4 carry root-cause class."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


SCHEMA = "complete_o4_carry_root_cause_v1"
EXPECTED_REPRODUCTION = {
    ("A3", 1): (1000, None),
    ("A3", 64): (1000, None),
    ("A4", 1): (319, 320),
    ("A4", 64): (333, 334),
}


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite token {token} in {path}")
        ),
    )
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _width(checkpoint: Mapping[str, Any], name: str) -> float:
    return float(checkpoint["source_intervals"][name]["max_width"])


def derive(
    *,
    reproductions: list[Mapping[str, Any]],
    divergence: Mapping[str, Any],
    substitutions: Mapping[str, Any],
    substitution_rows: list[Mapping[str, str]],
    dense: Mapping[str, Any],
    first_accounting: Mapping[str, Any],
    failure_accounting: Mapping[str, Any],
) -> dict[str, Any]:
    seen: dict[tuple[str, int], Mapping[str, Any]] = {}
    for summary in reproductions:
        key = (str(summary["cell"]), int(summary["batch"]))
        _require(key not in seen, f"duplicate reproduction cell {key}")
        seen[key] = summary
    _require(set(seen) == set(EXPECTED_REPRODUCTION), "reproduction matrix is incomplete")
    for key, (steps, failure_step) in EXPECTED_REPRODUCTION.items():
        summary = seen[key]
        _require(summary.get("reproduction_status") == "reproduced", f"{key} did not reproduce")
        _require(int(summary["completed_steps"]) == steps, f"{key} horizon mismatch")
        failure = summary.get("first_failure")
        actual_failure = None if failure is None else int(failure["step"])
        _require(actual_failure == failure_step, f"{key} failure mismatch")
        _require(summary.get("no_hidden_fallback") is True, f"{key} used fallback")

    _require(
        divergence.get("reproduction_status") == "A3_A4_FROZEN_RESULTS_REPRODUCED",
        "divergence ledger is not closed",
    )
    for cell in divergence["divergence"]:
        _require(cell["first_coefficient_bit_divergence"]["step"] == 1, "first coefficient divergence changed")
        _require(cell["first_remainder_divergence"]["step"] == 2, "first remainder divergence changed")
        _require(cell["first_physical_endpoint_divergence"]["step"] == 1, "endpoint divergence changed")
        _require(cell["first_tube_divergence"]["step"] == 1, "tube divergence changed")

    _require(substitutions.get("epsilon_decision_relevant_anywhere") is False, "epsilon is decision relevant")
    for checkpoint in substitutions["checkpoints"]:
        _require(checkpoint["all_substitutions_used_identical_prestate"] is True, "prestate mismatch")
        _require(all(checkpoint["canonical_duplicate_checks"].values()), "canonical substitution mismatch")
    pre_failure = [row for row in substitution_rows if row["checkpoint"] == "before_step_0320.npz"]
    canonical = {row["family"]: row for row in pre_failure if row["label"].endswith("complete_carry")}
    _require(set(canonical) == {"CDR", "CNI"}, "pre-failure canonical rows are missing")
    _require(canonical["CDR"]["accepted"] == "True", "CDR pre-failure decision changed")
    _require(canonical["CNI"]["accepted"] == "False", "CNI pre-failure decision changed")

    _require(dense.get("dense_cni_parity_outcome") == "DENSE_CNI_PARITY_NOT_EXPRESSIBLE", "dense contract unexpectedly expressible")
    _require(dense.get("basis_roundtrip_status") == "closed", "dense/R35 basis roundtrip is open")
    _require(dense.get("dense_api_has_native_complete_composition") is False, "dense API claim changed")
    for fixture in dense["fixtures"]:
        _require(fixture["exponent_sets_equal"] is True, "basis exponent mismatch")
        _require(fixture["coefficient_roundtrip_bit_exact"] is True, "basis coefficient mismatch")
        _require(fixture["remainder_lo_roundtrip_bit_exact"] is True, "remainder lo mismatch")
        _require(fixture["remainder_hi_roundtrip_bit_exact"] is True, "remainder hi mismatch")

    accounting = [first_accounting, failure_accounting]
    for report in accounting:
        _require(report["all_native_observer_parity_bit_exact"] is True, "observer altered native composition")
        _require(report["all_coverage_contains_native_remainder"] is True, "composition coverage failed")
        _require(report["any_double_count_detected"] is False, "double count detected")
    first = first_accounting["checkpoints"][0]
    failure = failure_accounting["checkpoints"][0]
    for checkpoint in (first, failure):
        _require(checkpoint["roundtrip_contains_before"] is True, "coordinate roundtrip failed")
        _require(
            _width(checkpoint, "endpoint_remainder_times_parameterization_polynomial") == 0.0,
            "structurally absent endpoint remainder product is nonzero",
        )
    dominant_width = _width(failure, "polynomial_times_parameterization_remainder")
    degree_overflow_width = _width(failure, "degree_gt4_dropped_polynomial")
    outer_width = _width(failure, "outer_endpoint_remainder")
    _require(failure["dominant_source_by_max_width"] == "polynomial_times_parameterization_remainder", "dominant source changed")
    _require(dominant_width > degree_overflow_width and dominant_width > outer_width, "dominance is unresolved")

    cdr_margin = float(canonical["CDR"]["minimum_target_margin"])
    cni_margin = float(canonical["CNI"]["minimum_target_margin"])
    return {
        "schema": SCHEMA,
        "source_sha": _head(),
        "outcome": "CARRY_MISSING_SYMBOLIC_SEMANTICS",
        "root_cause_class": "C4",
        "single_fix_authorization": "NO_FIX_AUTHORIZED",
        "reproduction": {
            f"{cell}_B{batch}": {
                "completed_steps": int(seen[(cell, batch)]["completed_steps"]),
                "validated_horizon": float(seen[(cell, batch)]["validated_horizon"]),
                "failure_step": None if seen[(cell, batch)]["first_failure"] is None else int(seen[(cell, batch)]["first_failure"]["step"]),
            }
            for cell, batch in sorted(seen)
        },
        "same_prestate": {
            "all_inputs_byte_identical": True,
            "epsilon_decision_relevant": False,
            "carry_family_decision_relevant": True,
            "pre_failure_checkpoint": "before_step_0320.npz",
            "cdr_margin": cdr_margin,
            "cni_margin": cni_margin,
        },
        "composition": {
            "native_observer_bit_exact": True,
            "coordinate_roundtrip_contains": True,
            "omission_detected": False,
            "double_count_detected": False,
            "first_material_pre_renormalization_remainder_width": float(first["pre_renormalization_remainder"]["max_width"]),
            "pre_failure_pre_renormalization_remainder_width": float(failure["pre_renormalization_remainder"]["max_width"]),
            "pre_failure_post_renormalization_remainder_width": float(failure["post_renormalization_remainder"]["max_width"]),
            "pre_failure_dominant_source": failure["dominant_source_by_max_width"],
            "pre_failure_dominant_source_width": dominant_width,
            "pre_failure_degree_gt4_width": degree_overflow_width,
            "pre_failure_outer_endpoint_remainder_width": outer_width,
            "dependency_loss_mechanism": failure["dependency_loss"],
        },
        "dense_parity": {
            "basis_roundtrip": "bit_exact",
            "cross_step_parity": "DENSE_CNI_PARITY_NOT_EXPRESSIBLE",
            "reason": dense["reason"],
        },
        "classification_ledger": {
            "C1": "excluded: physical hull survives center/scale normalization roundtrip",
            "C2": "excluded: the outer ordinary remainder is added once per output",
            "C3": "not selected: degree>4 overflow is orders of magnitude below the dominant materialized-parameterization-remainder term",
            "C4": "selected: R35 CNI materializes cross-step dependency into ordinary intervals and has no authoritative dense symbolic carry state",
            "C5": "excluded: widening reaches order-one scale and changes the horizon decision",
            "C6": "excluded: one mechanism dominates the failure checkpoint",
        },
        "qualification": {
            "A3": "ordinary-float64 empirical comparator only; T10 is not a soundness proof",
            "A4": "sound interval accounting within the implemented R35 contract, but not an authoritative complete-O4 dense/Flow* mirror",
        },
        "next_authorized_action": (
            "Specify and independently validate an authoritative complete-O4 cross-step symbolic-remainder contract before proposing any new carry implementation."
        ),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reproduction-summary", action="append", type=Path, required=True)
    parser.add_argument("--divergence-ledger", type=Path, required=True)
    parser.add_argument("--substitution-summary", type=Path, required=True)
    parser.add_argument("--substitution-csv", type=Path, required=True)
    parser.add_argument("--dense-parity", type=Path, required=True)
    parser.add_argument("--first-accounting", type=Path, required=True)
    parser.add_argument("--failure-accounting", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with args.substitution_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    report = derive(
        reproductions=[_load(path) for path in args.reproduction_summary],
        divergence=_load(args.divergence_ledger),
        substitutions=_load(args.substitution_summary),
        substitution_rows=rows,
        dense=_load(args.dense_parity),
        first_accounting=_load(args.first_accounting),
        failure_accounting=_load(args.failure_accounting),
    )
    inputs = [
        *args.reproduction_summary,
        args.divergence_ledger,
        args.substitution_summary,
        args.substitution_csv,
        args.dense_parity,
        args.first_accounting,
        args.failure_accounting,
    ]
    report["input_artifacts"] = [
        {"name": path.name, "sha256": _sha(path)} for path in inputs
    ]
    path = args.output_dir / "root_cause.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
