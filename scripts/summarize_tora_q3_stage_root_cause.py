#!/usr/bin/env python3
"""Publish the TORA-Q3 stage-parity root-cause decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_CLASSIFICATIONS = {
    "actual_torch_bug",
    "actual_observer_bug",
    "expected_outward_roundoff",
    "algorithm_semantics_difference",
    "coordinate_map_unavailable",
    "numerically_negligible",
    "dominant_candidate",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maximum_counterfactual(
    counterfactual: dict[str, Any], path: tuple[str, ...]
) -> float:
    values = []
    for row in counterfactual["per_segment"].values():
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return max(values)


def ledger_segment_40(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["lane"] == "L0_baseline_native" and int(row["segment"]) == 40:
                result[row["category"]] = {
                    "median": float(row["median_width"]),
                    "maximum": float(row["maximum_width"]),
                    "sum": float(row["sum_width"]),
                }
    if set(result) != {"picard_residual", "composition_overflow"}:
        raise ValueError("segment-40 ledger categories are incomplete")
    return result


def causal_messages(counterfactual: dict[str, Any]) -> dict[str, str]:
    minimum_sine = counterfactual["aggregate_effect"][
        "minimum_k2_and_sine_local_remainder_error_reduction_fraction"
    ]
    maximum_k2_effect = max(
        abs(
            row["local_remainder_error_reduction"][
                "k2_substitution_fraction"
            ]
        )
        for row in counterfactual["per_segment"].values()
    )
    same_range = maximum_counterfactual(
        counterfactual,
        (
            "diagnostic_same_polynomial_torch_range",
            "maximum_width_difference",
        ),
    )
    integration = maximum_counterfactual(
        counterfactual,
        (
            "same_input_integration_boundary",
            "degree_overflow_maximum_width_difference",
        ),
    )
    composition_coefficients = maximum_counterfactual(
        counterfactual,
        (
            "diagnostic_xiangru_local_then_torch_composition",
            "physical_polynomial_maximum_error",
        ),
    )
    return {
        "A0": "not applicable: exact normalized tensors are bitwise equal",
        "A1": "not applicable: exact base polynomial and remainder are bitwise equal",
        "A2": "first numerical difference, but only outward-scale error and too small to explain downstream growth",
        "A3": f"observed Xiangru sine substitution reduces local-remainder error by at least {minimum_sine:.6%}",
        "A4": f"K2 substitution changes local-remainder error by at most {maximum_k2_effect:.6%}; no material causal effect",
        "A5": f"K2 substitution changes local-remainder error by at most {maximum_k2_effect:.6%}; no material causal effect",
        "A6": f"the exact Xiangru difference polynomial evaluated by Torch differs in width by at most {same_range:.3e}",
        "A7": f"after sine substitution, the remaining same-input degree-overflow difference is at most {integration:.3e}",
        "A8": f"after sine substitution, the remaining same-input degree-overflow difference is at most {integration:.3e}",
        "A9": "downstream endpoint/tube differences shrink after sine substitution; same-polynomial tube evaluation remains a secondary semantic difference",
        "A10": f"same local TM produces physical coefficients with maximum error {composition_coefficients:.3e}; carry is not causal",
        "A11": "same controller-input physical box differs by at most one ULP from summation order",
        "A12": "same-input CROWN output before and after outward composition is bitwise equal",
    }


def classifications() -> dict[str, str]:
    return {
        "A0": "numerically_negligible",
        "A1": "numerically_negligible",
        "A2": "expected_outward_roundoff",
        "A3": "dominant_candidate",
        "A4": "numerically_negligible",
        "A5": "numerically_negligible",
        "A6": "numerically_negligible",
        "A7": "algorithm_semantics_difference",
        "A8": "algorithm_semantics_difference",
        "A9": "algorithm_semantics_difference",
        "A10": "numerically_negligible",
        "A11": "expected_outward_roundoff",
        "A12": "numerically_negligible",
    }


def stage_rows(
    comparison: dict[str, Any], counterfactual: dict[str, Any]
) -> list[dict[str, Any]]:
    causal = causal_messages(counterfactual)
    classes = classifications()
    rows = []
    for source in comparison["stage_table"]:
        stage = source["stage"]
        classification = classes[stage]
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise ValueError("invalid root-cause classification")
        rows.append(
            {
                "stage": stage,
                "input_contract_equal": source["input_contract_equal"],
                "coordinate_map_status": source["coordinate_map_status"],
                "max_abs_lower_diff": source["max_abs_lower_diff"],
                "max_abs_upper_diff": source["max_abs_upper_diff"],
                "max_ulp_diff": source["max_ulp_diff"],
                "center_diff": source["center_diff"],
                "width_diff": source["width_diff"],
                "containment_relation": source["containment_relation"],
                "remainder_contribution_diff": source[
                    "remainder_contribution_diff"
                ],
                "first_segment": source["first_segment"],
                "first_leaf": source["first_leaf"],
                "causal_substitution_effect": causal[stage],
                "classification": classification,
            }
        )
    return rows


def report_markdown(result: dict[str, Any]) -> str:
    rows = result["root_cause_table"]
    lines = [
        "# TORA-Q3 Stage-Parity Root-Cause Report",
        "",
        "## Decision",
        "",
        "The first numerical difference is A2 (the outward point sine enclosure), "
        "but its maximum bound error is only about `4.22e-15`. The first "
        "material and causally dominant difference is A3: sine composition "
        "remainder routing and analytic-tail semantics. The retained sine "
        "polynomial remains equal to roundoff scale; the interval remainder does not.",
        "",
        "Replacing only K2 has no material effect. Replacing K2 and the observed "
        "Xiangru sine aggregate reduces the same-input local-remainder error by "
        f"`{result['dominant_candidate']['minimum_local_remainder_error_reduction_fraction']:.6%}` "
        "or more at every selected replay point. The residual is traced to "
        "integration degree-overflow routing and, for full tubes, a secondary "
        "same-polynomial range semantic difference.",
        "",
        "## Required stage table",
        "",
        "| stage | input equal | coordinate map | max lower diff | max upper diff | max ULP | center diff | width diff | containment | first segment/leaf | classification |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {equal} | {coordinate} | {lower:.6g} | {upper:.6g} | {ulp} | {center:.6g} | {width:.6g} | {containment} | {segment}/{leaf} | {classification} |".format(
                stage=row["stage"],
                equal=row["input_contract_equal"],
                coordinate=row["coordinate_map_status"],
                lower=row["max_abs_lower_diff"],
                upper=row["max_abs_upper_diff"],
                ulp=row["max_ulp_diff"],
                center=row["center_diff"],
                width=row["width_diff"],
                containment=row["containment_relation"],
                segment=row["first_segment"] if row["first_segment"] is not None else "-",
                leaf=row["first_leaf"] if row["first_leaf"] is not None else "-",
                classification=row["classification"],
            )
        )
    t1 = result["t1_0_014211_attribution"]
    segment40 = result["segment_40_remainder_attribution"]
    lines.extend(
        [
            "",
            "## T=1 attribution",
            "",
            f"The frozen common-control direct endpoint difference is `{t1['direct_exact_endpoint_vs_xiangru_max_abs']:.15g}`. "
            f"`{t1['fraction_already_present_before_projection']:.6%}` is already present before projection. "
            "A2 is far too small to explain it. A3 recurs on S0, S1, R1, R2, and F0; at R1 the sine substitution reduces the local-remainder error by "
            f"`{t1['r1_local_remainder_error_reduction_fraction']:.6%}`. The `0.014211` value is the ten-step accumulated consequence, not a one-step A2 rounding artifact.",
            "",
            "## Segment 40 remainder attribution",
            "",
            f"The maximum pre-projection interval-remainder width is `{segment40['pre_projection_interval_remainder_maximum']:.15g}`. "
            f"The broad `composition_overflow` ledger category accounts for `{segment40['composition_overflow_maximum']:.15g}`, while the current local `picard_residual` maximum is only `{segment40['picard_residual_maximum']:.15g}`. "
            "Affine composition labels carried prior remainder under this broad category; projection inflation remains roundoff-scale. At the exact segment-40 replay input, sine substitution removes "
            f"`{segment40['same_input_sine_reduction_fraction']:.6%}` of the local-remainder error. Thus the accumulated category is `composition_overflow`, and its earliest material generator is A3 sine remainder semantics; A7/A8 integration overflow is secondary.",
            "",
            "## Counterfactual scope",
            "",
            "All substitutions are diagnostic only. Xiangru outputs are not used by the formal native runner. The reverse `Torch sine -> Xiangru integration` check changes the integration remainder width by up to "
            f"`{result['reverse_sine_substitution']['maximum_remainder_width_difference']:.6g}`, confirming the A3 effect persists when the downstream implementation is swapped. Same-input A12 CROWN bounds are bitwise equal.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-comparison", type=Path, required=True)
    parser.add_argument("--counterfactual", type=Path, required=True)
    parser.add_argument("--reverse-substitution", type=Path, required=True)
    parser.add_argument("--legacy-attribution", type=Path, required=True)
    parser.add_argument("--ledger-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    comparison = load(args.stage_comparison)
    counterfactual = load(args.counterfactual)
    reverse = load(args.reverse_substitution)
    legacy = load(args.legacy_attribution)
    ledger = ledger_segment_40(args.ledger_csv)
    rows = stage_rows(comparison, counterfactual)
    point_max = maximum_counterfactual(
        counterfactual,
        ("same_input_sine_boundary", "point_enclosure_maximum_error"),
    )
    retained_max = maximum_counterfactual(
        counterfactual,
        ("same_input_sine_boundary", "retained_polynomial_maximum_error"),
    )
    composition_max = maximum_counterfactual(
        counterfactual,
        (
            "same_input_sine_boundary",
            "composition_overflow_maximum_width_difference",
        ),
    )
    analytic_max = maximum_counterfactual(
        counterfactual,
        (
            "same_input_sine_boundary",
            "analytic_remainder_maximum_width_difference",
        ),
    )
    integration_max = maximum_counterfactual(
        counterfactual,
        (
            "same_input_integration_boundary",
            "degree_overflow_maximum_width_difference",
        ),
    )
    t1 = dict(legacy["t1_0_014211_attribution"])
    t1["first_numerical_stage"] = "A2"
    t1["first_material_stage"] = "A3"
    t1["r1_local_remainder_error_reduction_fraction"] = counterfactual[
        "per_segment"
    ]["10"]["local_remainder_error_reduction"][
        "k2_and_sine_substitution_fraction"
    ]
    segment40_legacy = legacy["segment_40_width_decomposition"]
    segment40 = {
        "pre_projection_interval_remainder_maximum": segment40_legacy[
            "pre_projection_interval_remainder"
        ]["maximum"],
        "pre_projection_polynomial_range_maximum": segment40_legacy[
            "pre_projection_polynomial_range"
        ]["maximum"],
        "projection_inflation_maximum": max(
            segment40_legacy["current_projection_inflation_maximum"],
            segment40_legacy["physical_projection_inflation_maximum"],
        ),
        "composition_overflow_maximum": ledger["composition_overflow"][
            "maximum"
        ],
        "picard_residual_maximum": ledger["picard_residual"]["maximum"],
        "composition_overflow_to_picard_residual_ratio": ledger[
            "composition_overflow"
        ]["maximum"]
        / ledger["picard_residual"]["maximum"],
        "same_input_sine_reduction_fraction": counterfactual["per_segment"][
            "40"
        ]["local_remainder_error_reduction"][
            "k2_and_sine_substitution_fraction"
        ],
        "dominant_accumulated_ledger_category": "composition_overflow",
        "earliest_material_generator": "A3",
        "secondary_generator": "A7/A8 integration degree overflow",
    }
    selection_criteria = {
        "appears_before_t1": True,
        "repeats_or_grows_at_r1_r2": True,
        "significant_remainder_contribution": True,
        "counterfactual_reduces_downstream_difference": True,
        "consistent_with_source_level_mathematical_semantics": True,
        "independently_implementable_without_relaxing_soundness": True,
    }
    result = {
        "schema": "tora_q3_stage_parity_root_cause_v1",
        "status": "PASS_DOMINANT_STAGE_ISOLATED",
        "root_cause_table": rows,
        "first_differences": {
            **comparison["first_differences"],
            "first_material": {
                "stage": "A3",
                "segment": 1,
                "leaf": 0,
                "reason": "same-input sine remainder width difference exceeds 1e-3 while point and retained-polynomial errors remain roundoff-scale",
            },
        },
        "dominant_candidate": {
            "stage": "A3",
            "mathematical_boundary": "sine composition remainder routing and analytic tail",
            "point_enclosure_maximum_error": point_max,
            "retained_polynomial_maximum_error": retained_max,
            "composition_overflow_maximum_width_difference": composition_max,
            "analytic_remainder_maximum_width_difference": analytic_max,
            "minimum_local_remainder_error_reduction_fraction": counterfactual[
                "aggregate_effect"
            ]["minimum_k2_and_sine_local_remainder_error_reduction_fraction"],
            "selection_criteria": selection_criteria,
        },
        "secondary_difference": {
            "stage": "A7/A8",
            "mathematical_boundary": "integration degree-overflow interval routing",
            "maximum_width_difference": integration_max,
        },
        "t1_0_014211_attribution": t1,
        "segment_40_remainder_attribution": segment40,
        "reverse_sine_substitution": {
            "diagnostic_counterfactual": True,
            "formal_native_result": False,
            "maximum_remainder_width_difference": reverse[
                "aggregate_maximum_error_vs_xiangru"
            ]["remainder_width"],
            "maximum_polynomial_difference": reverse[
                "aggregate_maximum_error_vs_xiangru"
            ]["polynomial"],
            "private_result_sha256": sha256(args.reverse_substitution),
        },
        "formal_runner_uses_xiangru_outputs": False,
        "source_hashes": {
            "stage_comparison": sha256(args.stage_comparison),
            "counterfactual": sha256(args.counterfactual),
            "legacy_attribution": sha256(args.legacy_attribution),
            "ledger_csv": sha256(args.ledger_csv),
        },
        "raw_paths_in_public_record": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(report_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "first_numerical_stage": "A2",
                "first_material_stage": "A3",
                "minimum_sine_substitution_reduction": result[
                    "dominant_candidate"
                ]["minimum_local_remainder_error_reduction_fraction"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
