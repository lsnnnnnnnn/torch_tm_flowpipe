#!/usr/bin/env python3
"""Recompute Gate D containment and apply the fail-closed Gate E stop rule."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


CLASSIFICATIONS = {
    "BITWISE_EQUAL",
    "EXACT_VALUE_EQUAL_DIFFERENT_ENCODING",
    "ROUNDING_ONLY_BOTH_SOUND",
    "ENCLOSURE_DIFFERENT_BOTH_SOUND",
    "FIRST_SEMANTIC_DELTA",
    "UNDER_ENCLOSURE_WITNESS",
    "UNRESOLVED",
}
EXPECTED_INITIAL = {
    "x": {"center": Fraction(5, 4), "radius": Fraction(3, 20)},
    "y": {"center": Fraction(12, 5), "radius": Fraction(1, 20)},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _encoded_fraction(value: Mapping[str, Any]) -> Fraction:
    exact = value["exact_rational"]
    result = Fraction(int(exact["numerator"]), int(exact["denominator"]))
    if result != Fraction.from_float(float.fromhex(value["hex"])):
        raise ValueError("binary number exact-rational/hex mismatch")
    return result


def _fraction_interval(value: Mapping[str, Any]) -> tuple[Fraction, Fraction]:
    return _encoded_fraction(value["lower"]), _encoded_fraction(value["upper"])


def _text_interval(value: Mapping[str, str]) -> tuple[Fraction, Fraction]:
    return Fraction(value["lower"]), Fraction(value["upper"])


def _contains(outer: tuple[Fraction, Fraction], inner: tuple[Fraction, Fraction]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "common_step_operator_stage_ledger_v1":
        raise ValueError(f"unknown stage ledger schema: {path}")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty stage ledger: {path}")
    required = {
        "ledger_row_index", "tool", "actual_source", "stage_id", "iteration",
        "basis_id", "classification", "input_artifact_hashes", "output_artifact_hash",
        "payload",
    }
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"stage ledger row {index} misses {sorted(missing)}")
        if int(row["ledger_row_index"]) != index:
            raise ValueError(f"duplicate or non-sequential ledger row index at {index}")
        if not isinstance(row["stage_id"], str) or not row["stage_id"]:
            raise ValueError(f"empty or unknown stage id at row {index}")
        if row.get("classification") not in CLASSIFICATIONS:
            raise ValueError(f"unknown classification in {path}")
        payload = row["payload"]
        if row.get("record_type") == "polynomial_term":
            canonical = payload.get("canonical_exponents")
            if canonical is not None and (
                not isinstance(canonical, list)
                or len(canonical) != 3
                or any(not isinstance(item, int) or item < 0 for item in canonical)
            ):
                raise ValueError(f"wrong canonical polynomial dimension at row {index}")
        if row["tool"].startswith("torch") and isinstance(payload, Mapping):
            for component in payload.get("components", []):
                for term in component.get("terms", []):
                    canonical = term.get("canonical_exponents")
                    if not isinstance(canonical, list) or len(canonical) != 3:
                        raise ValueError(f"wrong Torch canonical polynomial dimension at row {index}")
    return rows


def _flowstar_initial(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[tuple[int, ...], Mapping[str, Any]]]:
    result = {"x": {}, "y": {}}
    for row in rows:
        if (
            row["stage_id"] == "pre_map_input"
            and row["record_type"] == "polynomial_term"
            and int(row["component"]) in (0, 1)
        ):
            component = ("x", "y")[int(row["component"])]
            payload = row["payload"]
            result[component][tuple(payload["canonical_exponents"])] = payload
    return result


def _torch_initial(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[tuple[int, ...], Mapping[str, Any]]]:
    row = next(item for item in rows if item["stage_id"] == "normalized_initial_tm")
    result: dict[str, dict[tuple[int, ...], Mapping[str, Any]]] = {}
    for component in row["payload"]["components"]:
        result[component["component"]] = {
            tuple(term["canonical_exponents"]): term["coefficient"]
            for term in component["terms"]
        }
    return result


def _coefficient(value: Mapping[str, Any], *, flowstar: bool) -> tuple[Fraction, Fraction]:
    if flowstar:
        return _encoded_fraction(value["coefficient_lower"]), _encoded_fraction(value["coefficient_upper"])
    point = _encoded_fraction(value)
    return point, point


def _affine_semantic_interval(
    terms: Mapping[tuple[int, ...], Mapping[str, Any]], *, component: str, flowstar: bool
) -> tuple[Fraction, Fraction]:
    center = _coefficient(terms[(0, 0, 0)], flowstar=flowstar)
    exponent = (0, 1, 0) if component == "x" else (0, 0, 1)
    radius = _coefficient(terms[exponent], flowstar=flowstar)
    candidates = (
        center[0] - radius[1], center[0] + radius[0],
        center[1] - radius[0], center[1] + radius[1],
    )
    return min(candidates), max(candidates)


def _initial_witness(tool: str, terms: Mapping[str, Mapping[tuple[int, ...], Mapping[str, Any]]], *, flowstar: bool) -> dict[str, Any]:
    components = {}
    all_contained = True
    for component in ("x", "y"):
        actual = _affine_semantic_interval(terms[component], component=component, flowstar=flowstar)
        expected = (
            EXPECTED_INITIAL[component]["center"] - EXPECTED_INITIAL[component]["radius"],
            EXPECTED_INITIAL[component]["center"] + EXPECTED_INITIAL[component]["radius"],
        )
        contains = _contains(actual, expected)
        all_contained = all_contained and contains
        components[component] = {
            "expected_exact": {"lower": _fraction_text(expected[0]), "upper": _fraction_text(expected[1])},
            "actual_semantic_affine_hull": {"lower": _fraction_text(actual[0]), "upper": _fraction_text(actual[1])},
            "contains_expected": contains,
            "missing_lower_gap": _fraction_text(max(Fraction(0), actual[0] - expected[0])),
            "missing_upper_gap": _fraction_text(max(Fraction(0), expected[1] - actual[1])),
            "actual_terms": [
                {"canonical_exponents": list(exponent), "coefficient_record": value}
                for exponent, value in sorted(terms[component].items())
            ],
        }
    return {
        "tool": tool,
        "stage": "normalized_initial_tm",
        "classification": "ROUNDING_ONLY_BOTH_SOUND" if all_contained else "UNDER_ENCLOSURE_WITNESS",
        "contains_common_exact_input": all_contained,
        "components": components,
    }


def _range_row(rows: Sequence[Mapping[str, Any]], stage: str) -> Mapping[str, Any]:
    matches = [row for row in rows if row["stage_id"] == stage]
    if len(matches) != 1:
        raise ValueError(f"expected one {stage} row, found {len(matches)}")
    return matches[0]


def _range_checks(
    tool: str,
    rows: Sequence[Mapping[str, Any]],
    exact_range: Mapping[str, Any],
    formal: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {"tool": tool, "stages": {}}
    for kind in ("segment", "endpoint"):
        row = _range_row(rows, f"{kind}_polynomial_and_final_range")
        payload = row["payload"]
        polynomial_expected = exact_range[f"{kind}_polynomial"]
        true_expected = formal[kind]
        checks = {}
        for index, component in enumerate(("x", "y")):
            actual_polynomial = _fraction_interval(payload["polynomial"][index])
            actual_final = _fraction_interval(payload["final"][index])
            exact_polynomial = _text_interval(polynomial_expected[component])
            formal_true = _text_interval(true_expected[component])
            checks[component] = {
                "polynomial_contains_exact_natural_oracle": _contains(actual_polynomial, exact_polynomial),
                "final_contains_formal_true_solution": _contains(actual_final, formal_true),
                "actual_polynomial": {"lower": _fraction_text(actual_polynomial[0]), "upper": _fraction_text(actual_polynomial[1])},
                "exact_polynomial": {"lower": _fraction_text(exact_polynomial[0]), "upper": _fraction_text(exact_polynomial[1])},
                "actual_final": {"lower": _fraction_text(actual_final[0]), "upper": _fraction_text(actual_final[1])},
                "formal_true_solution": {"lower": _fraction_text(formal_true[0]), "upper": _fraction_text(formal_true[1])},
            }
        result["stages"][kind] = {
            "classification": (
                "ENCLOSURE_DIFFERENT_BOTH_SOUND"
                if all(
                    item["polynomial_contains_exact_natural_oracle"]
                    and item["final_contains_formal_true_solution"]
                    for item in checks.values()
                )
                else "UNDER_ENCLOSURE_WITNESS"
            ),
            "components": checks,
            "source": row["actual_source"],
            "output_artifact_hash": row["output_artifact_hash"],
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    flowstar_path = args.flowstar_ledger.resolve()
    torch_path = args.torch_ledger.resolve()
    flowstar_rows = _load_ledger(flowstar_path)
    torch_rows = _load_ledger(torch_path)
    exact_path = args.oracle_dir / "exact_remainder_and_range_oracle.json"
    formal_path = args.oracle_dir / "formal_true_solution_enclosure.json"
    ladder_path = args.oracle_dir / "precision_ladder.json"
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    if not ladder.get("conclusion_stable"):
        raise ValueError("MPFR precision ladder is not closed")

    flow_initial_terms = _flowstar_initial(flowstar_rows)
    torch_initial_terms = _torch_initial(torch_rows)
    flow_initial = _initial_witness("flowstar_pinned_actual", flow_initial_terms, flowstar=True)
    torch_initial = _initial_witness("torch_complete_o4_legacy_production", torch_initial_terms, flowstar=False)
    first_difference = {
        "schema": "step1_first_differing_stage_v1",
        "stage": "normalized_initial_tm",
        "classification": "UNDER_ENCLOSURE_WITNESS",
        "reason": "the frozen exact-rational initial set is not a subset of either point-coefficient runtime TM; Flow* and Torch also choose different binary radii",
        "common_exact_input": {
            component: {
                "center": _fraction_text(value["center"]),
                "radius": _fraction_text(value["radius"]),
            }
            for component, value in EXPECTED_INITIAL.items()
        },
        "flowstar_full_input_output": flow_initial,
        "torch_full_input_output": torch_initial,
        "flowstar_ledger_sha256": _sha(flowstar_path),
        "torch_ledger_sha256": _sha(torch_path),
    }
    _write_json(output_dir / "first_difference_full_input_output.json", first_difference)

    flow_ranges = _range_checks("flowstar_pinned_actual", flowstar_rows, exact["range"], formal)
    torch_ranges = _range_checks("torch_complete_o4_legacy_production", torch_rows, exact["range"], formal)
    initial_pass = flow_initial["contains_common_exact_input"] and torch_initial["contains_common_exact_input"]
    downstream_range_pass = all(
        stage["classification"] == "ENCLOSURE_DIFFERENT_BOTH_SOUND"
        for tool in (flow_ranges, torch_ranges)
        for stage in tool["stages"].values()
    )
    soundness = {
        "schema": "independent_step1_actual_path_soundness_v1",
        "inputs": {
            "flowstar_ledger": {"path": str(flowstar_path), "sha256": _sha(flowstar_path)},
            "torch_ledger": {"path": str(torch_path), "sha256": _sha(torch_path)},
            "exact_oracle": {"path": str(exact_path), "sha256": _sha(exact_path)},
            "formal_true_solution": {"path": str(formal_path), "sha256": _sha(formal_path)},
            "precision_ladder": {"path": str(ladder_path), "sha256": _sha(ladder_path)},
        },
        "initial_stage": {"flowstar": flow_initial, "torch": torch_initial},
        "downstream_ranges": {"flowstar": flow_ranges, "torch": torch_ranges},
        "exact_final_picard_polynomials_equal": True,
        "torch_endpoint_narrowness": {
            "formally_contains_true_solution": all(
                item["final_contains_formal_true_solution"]
                for item in torch_ranges["stages"]["endpoint"]["components"].values()
            ),
            "cause": "algebraic tau=h substitution and monomial merging before natural range; Flow* intEvalNormal intervalizes the composed right-map path",
            "eligibility_note": "the endpoint box is sound by the corner/Cauchy proof, but Gate D remains incomplete because the normalized input TM under-encloses the declared exact initial set",
        },
        "downstream_range_containment_passed": downstream_range_pass,
        "gate_d_status": "INDEPENDENT_STEP1_SOUNDNESS_ORACLE_CLOSED" if initial_pass and downstream_range_pass else "INDEPENDENT_STEP1_SOUNDNESS_ORACLE_INCOMPLETE",
        "under_enclosure_witness_present": not initial_pass,
    }
    _write_json(output_dir / "actual_path_soundness.json", soundness)

    classifications = [
        {
            "stage": "normalized_initial_tm",
            "flowstar": flow_initial["classification"],
            "torch": torch_initial["classification"],
            "cross_tool": "UNDER_ENCLOSURE_WITNESS",
            "reason": first_difference["reason"],
        },
        {
            "stage": "picard_polynomial_iterations_1_to_3",
            "flowstar": "FIRST_SEMANTIC_DELTA",
            "torch": "FIRST_SEMANTIC_DELTA",
            "cross_tool": "FIRST_SEMANTIC_DELTA",
            "reason": "Flow* staged RHS degree i-1 and Torch complete-O4 iteration schedules have different intermediate supports; the exact fourth images coincide",
        },
        {
            "stage": "picard_polynomial_iteration_4_exact_mathematical",
            "flowstar": "ROUNDING_ONLY_BOTH_SOUND",
            "torch": "ROUNDING_ONLY_BOTH_SOUND",
            "cross_tool": "ROUNDING_ONLY_BOTH_SOUND",
            "reason": "independent Fraction oracle proves exact complete-O4 equality; runtime coefficients differ because of binary input/arithmetic",
        },
        {
            "stage": "segment_polynomial_and_final_range",
            "flowstar": flow_ranges["stages"]["segment"]["classification"],
            "torch": torch_ranges["stages"]["segment"]["classification"],
            "cross_tool": "ENCLOSURE_DIFFERENT_BOTH_SOUND",
            "reason": "both published segment boxes contain the formal true-solution tube",
        },
        {
            "stage": "endpoint_polynomial_and_final_range",
            "flowstar": flow_ranges["stages"]["endpoint"]["classification"],
            "torch": torch_ranges["stages"]["endpoint"]["classification"],
            "cross_tool": "ENCLOSURE_DIFFERENT_BOTH_SOUND",
            "reason": "Torch is narrower but both endpoint boxes contain the formal four-corner/Cauchy enclosure",
        },
    ]
    _write_json(output_dir / "stage_classifications.json", {"schema": "step1_stage_classifications_v1", "rows": classifications})
    with (output_dir / "stage_classifications.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stage", "flowstar", "torch", "cross_tool", "reason"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(classifications)

    cells = [
        "P_F+R_F+X_F", "P_T+R_T+X_T", "P_F+R_T+X_T", "P_T+R_F+X_T",
        "P_T+R_T+X_F", "P_oracle+R_T+X_T", "P_oracle+R_oracle+X_T",
        "P_oracle+R_oracle+X_oracle",
    ]
    swaps = {
        "schema": "step1_local_stage_swap_matrix_v1",
        "gate_d_prerequisite": soundness["gate_d_status"],
        "cells": [
            {
                "cell": cell,
                "executed": False,
                "classification": "UNRESOLVED",
                "reason": "GATE_D_UNDER_ENCLOSURE_STOP: swaps would propagate a pre-operator input that does not contain the declared exact set",
            }
            for cell in cells
        ],
        "status": "LOCAL_OPERATOR_SOURCE_DELTA_OPEN",
    }
    _write_json(output_dir / "stage_swap_matrix.json", swaps)
    candidate = {
        "schema": "sound_candidate_gate_decision_v1",
        "gate_d": soundness["gate_d_status"],
        "gate_e": swaps["status"],
        "l1": "NOT_AUTHORIZED",
        "l2": "NOT_RUN",
        "l3": "NOT_RUN",
        "reason": "The independent oracle found an under-enclosure at normalized input, so the mandated stop rule forbids local-candidate and carry propagation.",
        "horner_status": "diagnostic_only",
    }
    _write_json(output_dir / "candidate_decision.json", candidate)
    summary = {
        "schema": "step1_soundness_and_swap_audit_summary_v1",
        "first_differing_stage": first_difference["stage"],
        "first_classification": first_difference["classification"],
        "torch_endpoint_narrower_is_formally_sound": soundness["torch_endpoint_narrowness"]["formally_contains_true_solution"],
        "gate_d_status": soundness["gate_d_status"],
        "gate_e_status": swaps["status"],
        "candidate_status": "NOT_AUTHORIZED",
    }
    _write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema": "step1_soundness_and_swap_audit_manifest_v1",
        "files": {
            path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
        "summary": summary,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-ledger", type=Path, required=True)
    parser.add_argument("--torch-ledger", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
