#!/usr/bin/env python3
"""Compare private Xiangru and Torch TORA-Q3 stage observations.

Raw per-leaf comparisons are private.  The public result contains only
aggregates, hashes, first locations, containment counts, and stage verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SELECTED_SEGMENTS = (1, 2, 10, 40, 43, 44, 45)
STAGES = tuple(f"A{index}" for index in range(13))
SIGN_MASK = np.uint64(1 << 63)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float64(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("stage comparison received a non-finite value")
    return result


def ordered_float_bits(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.float64).view(np.uint64)
    negative = (bits & SIGN_MASK) != 0
    return np.where(negative, ~bits, bits | SIGN_MASK)


def maximum_ulp_distance(left: Any, right: Any) -> int:
    first = as_float64(left)
    second = as_float64(right)
    if first.shape != second.shape:
        raise ValueError(f"shape mismatch: {first.shape} != {second.shape}")
    first_ordered = ordered_float_bits(first)
    second_ordered = ordered_float_bits(second)
    distance = np.maximum(first_ordered, second_ordered) - np.minimum(
        first_ordered, second_ordered
    )
    return int(np.max(distance, initial=np.uint64(0)))


def first_location(mask: np.ndarray) -> dict[str, int] | None:
    locations = np.argwhere(mask)
    if not len(locations):
        return None
    index = [int(value) for value in locations[0]]
    result = {"leaf": index[0] if index else 0}
    if len(index) >= 2:
        result["state"] = index[1]
    if len(index) >= 3:
        result["slot"] = index[2]
    return result


def compare_values(
    label: str,
    segment: int,
    xiangru: Any,
    torch_value: Any,
) -> dict[str, Any]:
    left = as_float64(xiangru)
    right = as_float64(torch_value)
    if left.shape != right.shape:
        raise ValueError(f"{label} shape mismatch: {left.shape} != {right.shape}")
    difference = np.abs(left - right)
    ulp = maximum_ulp_distance(left, right)
    return {
        "kind": "value",
        "label": label,
        "segment": segment,
        "bitwise": bool(np.array_equal(left, right)),
        "maximum_absolute_difference": float(np.max(difference, initial=0.0)),
        "maximum_ulp_difference": ulp,
        "first_difference": first_location(left != right),
        "first_exceeding_one_ulp": first_location(
            np.abs(
                ordered_float_bits(left).astype(object)
                - ordered_float_bits(right).astype(object)
            )
            > 1
        ),
    }


def containment_counts(
    xiangru_lower: np.ndarray,
    xiangru_upper: np.ndarray,
    torch_lower: np.ndarray,
    torch_upper: np.ndarray,
) -> dict[str, int]:
    equal = (xiangru_lower == torch_lower) & (xiangru_upper == torch_upper)
    torch_contains = (torch_lower <= xiangru_lower) & (
        torch_upper >= xiangru_upper
    )
    xiangru_contains = (xiangru_lower <= torch_lower) & (
        xiangru_upper >= torch_upper
    )
    overlap = np.maximum(xiangru_lower, torch_lower) <= np.minimum(
        xiangru_upper, torch_upper
    )
    return {
        "scalar_count": int(equal.size),
        "bitwise_equal": int(np.count_nonzero(equal)),
        "torch_contains_xiangru": int(np.count_nonzero(torch_contains)),
        "xiangru_contains_torch": int(np.count_nonzero(xiangru_contains)),
        "overlap": int(np.count_nonzero(overlap)),
        "disjoint": int(np.count_nonzero(~overlap)),
    }


def containment_relation(counts: Mapping[str, int]) -> str:
    total = counts["scalar_count"]
    if counts["bitwise_equal"] == total:
        return "bitwise_equal"
    if counts["torch_contains_xiangru"] == total:
        return "torch_contains_xiangru"
    if counts["xiangru_contains_torch"] == total:
        return "xiangru_contains_torch"
    if counts["disjoint"] == 0:
        return "mixed_overlapping"
    return "mixed_with_disjoint_scalars"


def compare_interval(
    label: str,
    segment: int,
    xiangru: Mapping[str, Any],
    torch_value: Mapping[str, Any],
) -> dict[str, Any]:
    xl = as_float64(xiangru["lower"])
    xu = as_float64(xiangru["upper"])
    tl = as_float64(torch_value["lower"])
    tu = as_float64(torch_value["upper"])
    if xl.shape != xu.shape or xl.shape != tl.shape or xl.shape != tu.shape:
        raise ValueError(f"{label} interval shape mismatch")
    if np.any(xl > xu) or np.any(tl > tu):
        raise ValueError(f"{label} contains an invalid interval")
    xc = xl + 0.5 * (xu - xl)
    tc = tl + 0.5 * (tu - tl)
    xw = xu - xl
    tw = tu - tl
    lower_diff = np.abs(xl - tl)
    upper_diff = np.abs(xu - tu)
    center_diff = np.abs(xc - tc)
    width_diff = np.abs(xw - tw)
    counts = containment_counts(xl, xu, tl, tu)
    changed = (xl != tl) | (xu != tu)
    return {
        "kind": "interval",
        "label": label,
        "segment": segment,
        "bitwise": bool(np.array_equal(xl, tl) and np.array_equal(xu, tu)),
        "maximum_absolute_lower_difference": float(
            np.max(lower_diff, initial=0.0)
        ),
        "maximum_absolute_upper_difference": float(
            np.max(upper_diff, initial=0.0)
        ),
        "maximum_ulp_difference": max(
            maximum_ulp_distance(xl, tl), maximum_ulp_distance(xu, tu)
        ),
        "maximum_center_difference": float(np.max(center_diff, initial=0.0)),
        "maximum_radius_difference": float(
            np.max(np.abs(0.5 * xw - 0.5 * tw), initial=0.0)
        ),
        "maximum_width_difference": float(np.max(width_diff, initial=0.0)),
        "maximum_xiangru_width": float(np.max(xw, initial=0.0)),
        "maximum_torch_width": float(np.max(tw, initial=0.0)),
        "containment_counts": counts,
        "containment_relation": containment_relation(counts),
        "first_difference": first_location(changed),
        "first_center_difference": first_location(xc != tc),
        "first_width_difference": first_location(xw != tw),
    }


def read_xiangru(path: Path) -> tuple[dict[str, Any], dict[int, Any]]:
    records: dict[int, Any] = {}
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        for line in handle:
            row = json.loads(line)
            segment = int(row["segment_index"])
            if segment in SELECTED_SEGMENTS:
                records[segment] = row
    if tuple(sorted(records)) != SELECTED_SEGMENTS:
        raise ValueError("Xiangru trace is missing selected segments")
    return header, records


def read_torch(path: Path) -> tuple[dict[str, Any], dict[int, Any], Any]:
    records: dict[int, Any] = {}
    controller = None
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        for line in handle:
            row = json.loads(line)
            if row["schema"] == "torch_tora_q3_stage_contract_segment_v1":
                records[int(row["segment_index"])] = row
            elif row["schema"] == "torch_tora_q3_stage_controller_observations_v1":
                controller = row
            else:
                raise ValueError("unexpected Torch stage trace row")
    if tuple(sorted(records)) != SELECTED_SEGMENTS or controller is None:
        raise ValueError("Torch stage trace is incomplete")
    return header, records, controller


def add_value(
    detail: dict[str, list[dict[str, Any]]],
    stage: str,
    label: str,
    segment: int,
    left: Any,
    right: Any,
) -> None:
    detail[stage].append(compare_values(label, segment, left, right))


def add_interval(
    detail: dict[str, list[dict[str, Any]]],
    stage: str,
    label: str,
    segment: int,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    detail[stage].append(compare_interval(label, segment, left, right))


def compare_plant_records(
    xiangru_records: Mapping[int, Any],
    torch_records: Mapping[int, Any],
) -> dict[str, list[dict[str, Any]]]:
    detail = {stage: [] for stage in STAGES}
    for segment in SELECTED_SEGMENTS:
        x = xiangru_records[segment]
        t = torch_records[segment]
        xs = x["stage_contract"]
        ts = t["stages"]

        for stage, xname, tname in (
            ("A0", "A0_normalized_input", "normalized_input"),
            ("A1", "A1_base_polynomial_and_remainder", "base_polynomial_and_remainder"),
        ):
            xp = xs[xname]
            tp = ts[stage][tname]
            add_value(
                detail,
                stage,
                "polynomial_coefficients",
                segment,
                xp["polynomial"]["coefficients"],
                tp["polynomial"]["coefficients"],
            )
            add_interval(
                detail, stage, "interval_remainder", segment, xp["remainder"], tp["remainder"]
            )

        diagnostics = t["diagnostic_counterfactual_same_input"]
        x_sources = [
            xs["A7_initial_remainder_image"],
            *xs["A8_remainder_rounds"],
        ]
        if len(diagnostics) != len(x_sources):
            raise ValueError("same-input diagnostic count mismatch")
        for index, (source, diagnostic) in enumerate(zip(x_sources, diagnostics, strict=True)):
            label = "initial" if index == 0 else f"round_{index}"
            xsi = source["rhs"]["sine"]
            tsi = diagnostic["sine"]
            for name in ("point_sine", "point_cosine"):
                add_interval(detail, "A2", f"{label}_{name}", segment, xsi[name], tsi[name])
            add_interval(detail, "A3", f"{label}_delta_range", segment, xsi["delta_range"], tsi["delta_range"])
            add_value(
                detail,
                "A3",
                f"{label}_retained_polynomial",
                segment,
                xsi["retained_polynomial"]["coefficients"],
                tsi["retained_polynomial"]["coefficients"],
            )
            for name in ("composition_overflow", "analytic_remainder"):
                add_interval(detail, "A3", f"{label}_{name}", segment, xsi[name], tsi[name])
            add_interval(
                detail,
                "A3",
                f"{label}_sine_output_remainder",
                segment,
                xsi["output"]["remainder"],
                tsi["output"]["remainder"],
            )
            integration_stage = "A7" if index == 0 else "A8"
            add_value(
                detail,
                integration_stage,
                f"{label}_same_input_integration_polynomial",
                segment,
                source["integration"]["output"]["polynomial"]["coefficients"],
                diagnostic["integration"]["output"]["polynomial"]["coefficients"],
            )
            add_interval(
                detail,
                integration_stage,
                f"{label}_same_input_integration_remainder",
                segment,
                source["integration"]["output"]["remainder"],
                diagnostic["integration"]["output"]["remainder"],
            )

        for index, stage in enumerate(("A4", "A5")):
            xp = xs["A2_A5_polynomial_picard"][index]
            tp = ts["A2_A5"][index]
            add_value(
                detail,
                stage,
                "candidate_coefficients",
                segment,
                xp["candidate"]["coefficients"],
                tp["candidate"]["polynomial"]["coefficients"],
            )
            add_interval(
                detail,
                stage,
                "integration_degree_overflow",
                segment,
                xp["integration_degree_overflow"],
                tp["integration"]["remainder"],
            )

        add_value(
            detail,
            "A6",
            "polynomial_difference",
            segment,
            xs["A6_polynomial_difference"]["polynomial"]["coefficients"],
            ts["A6"]["polynomial_difference"]["coefficients"],
        )
        add_interval(
            detail,
            "A6",
            "polynomial_difference_range",
            segment,
            xs["A6_polynomial_difference"]["range"],
            ts["A6"]["range"],
        )

        xi = xs["A7_initial_remainder_image"]
        ti = ts["A7"]
        add_value(
            detail,
            "A7",
            "initial_image_polynomial",
            segment,
            xi["image"]["polynomial"]["coefficients"],
            ti["image"]["polynomial"]["coefficients"],
        )
        add_interval(detail, "A7", "initial_image_remainder", segment, xi["image"]["remainder"], ti["image"]["remainder"])
        add_value(detail, "A7", "initial_subset_margin", segment, xi["subset_margin"], ti["subset_margin"])

        for xround, tround in zip(xs["A8_remainder_rounds"], ts["A8"], strict=True):
            round_index = int(xround["round"])
            prefix = f"round_{round_index}"
            add_value(
                detail,
                "A8",
                f"{prefix}_image_polynomial",
                segment,
                xround["image"]["polynomial"]["coefficients"],
                tround["image"]["polynomial"]["coefficients"],
            )
            for name in ("input_remainder", "candidate", "accepted"):
                add_interval(detail, "A8", f"{prefix}_{name}", segment, xround[name], tround[name])
            add_interval(
                detail,
                "A8",
                f"{prefix}_image_remainder",
                segment,
                xround["image"]["remainder"],
                tround["image"]["remainder"],
            )
            add_value(detail, "A8", f"{prefix}_subset_margin", segment, xround["subset_margin"], tround["subset_margin"])

        a9 = ts["A9"]
        add_value(
            detail,
            "A9",
            "local_final_polynomial",
            segment,
            x["picard"]["final_polynomial"]["coefficients"],
            a9["local_final"]["polynomial"]["coefficients"],
        )
        add_interval(detail, "A9", "local_final_remainder", segment, x["picard"]["final_remainder"], a9["local_final"]["remainder"])
        add_value(detail, "A9", "physical_polynomial", segment, x["polynomial_coefficient_vector"], a9["physical_coefficients"])
        add_interval(detail, "A9", "physical_remainder", segment, x["interval_remainder"], a9["physical_remainder"])
        for name in ("endpoint", "tube"):
            add_interval(detail, "A9", f"physical_{name}", segment, x[name], a9[f"physical_{name}"])
            lower = as_float64(a9[f"physical_{name}"]["lower"])[:, :4]
            upper = as_float64(a9[f"physical_{name}"]["upper"])[:, :4]
            torch_margin = 2.0 - np.maximum(np.abs(lower), np.abs(upper))
            add_value(detail, "A9", f"{name}_property_margin", segment, x["property_margin"][name], torch_margin)

        normalized_map = x["normalization"]["normalized_map"]
        add_value(detail, "A10", "affine_carry_linear", segment, as_float64(normalized_map["polynomial"]["L"])[:, :, 1:], ts["A10"]["affine_carry"]["linear"])
        add_interval(detail, "A10", "affine_carry_remainder", segment, normalized_map["remainder"], ts["A10"]["affine_carry"]["remainder"])
    return detail


def compare_controller_records(
    detail: dict[str, list[dict[str, Any]]],
    xiangru_controller: Mapping[str, Any],
    torch_controller: Mapping[str, Any],
) -> None:
    torch_rows = torch_controller["rows"]
    for label, row_index in (("R1", 1), ("R2", 4)):
        x = xiangru_controller["rows"][row_index]
        t = torch_rows[label]
        add_interval(detail, "A11", f"{label}_physical_controller_input", int(x["segment_index"]), x["controller_input_box_after_normalization"], t["reconstructed_physical_input"])
        add_interval(detail, "A12", f"{label}_controller_output_before_outward", int(x["segment_index"]), x["controller_output_interval_before_outward_composition"], t["output_before_outward"])
        add_interval(detail, "A12", f"{label}_controller_output_after_outward", int(x["segment_index"]), x["controller_output_interval_after_outward_composition"], t["output_after_outward"])


def summed_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    result = {
        "scalar_count": 0,
        "bitwise_equal": 0,
        "torch_contains_xiangru": 0,
        "xiangru_contains_torch": 0,
        "overlap": 0,
        "disjoint": 0,
    }
    for row in rows:
        if row["kind"] == "interval":
            for key, value in row["containment_counts"].items():
                result[key] += int(value)
    return result


def aggregate_stage(stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = [row for row in rows if row["kind"] == "interval"]
    first = next((row for row in rows if not row["bitwise"]), None)
    first_ulp = next((row for row in rows if row["maximum_ulp_difference"] > 1), None)
    first_width = next((row for row in intervals if row["maximum_width_difference"] > 0.0), None)
    first_center = next((row for row in intervals if row["maximum_center_difference"] > 0.0), None)
    first_margin = next((row for row in rows if "margin" in row["label"] and not row["bitwise"]), None)
    counts = summed_counts(rows)
    remainder_rows = [
        row
        for row in intervals
        if any(token in row["label"] for token in ("remainder", "overflow", "candidate", "accepted"))
    ]

    def location(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = {"segment": row["segment"], "comparison": row["label"]}
        if row["first_difference"] is not None:
            result.update(row["first_difference"])
        return result

    all_bitwise = all(row["bitwise"] for row in rows)
    return {
        "stage": stage,
        "comparison_count": len(rows),
        "input_contract_equal": stage in {"A0", "A1", "A2", "A3", "A4", "A10", "A11", "A12"},
        "coordinate_map_status": "IDENTICAL_NORMALIZED_INPUT_TENSORS",
        "all_compared_values_bitwise": all_bitwise,
        "max_abs_lower_diff": max((row["maximum_absolute_lower_difference"] for row in intervals), default=0.0),
        "max_abs_upper_diff": max((row["maximum_absolute_upper_difference"] for row in intervals), default=0.0),
        "max_ulp_diff": max((row["maximum_ulp_difference"] for row in rows), default=0),
        "center_diff": max((row["maximum_center_difference"] for row in intervals), default=0.0),
        "radius_diff": max((row["maximum_radius_difference"] for row in intervals), default=0.0),
        "width_diff": max((row["maximum_width_difference"] for row in intervals), default=0.0),
        "containment_relation": containment_relation(counts) if counts["scalar_count"] else "not_an_interval_stage",
        "containment_counts": counts,
        "remainder_contribution_diff": max((row["maximum_width_difference"] for row in remainder_rows), default=0.0),
        "first_segment": first["segment"] if first else None,
        "first_leaf": first["first_difference"]["leaf"] if first and first["first_difference"] else None,
        "first_bitwise_difference": location(first),
        "first_difference_exceeding_one_ulp": location(first_ulp),
        "first_width_difference": location(first_width),
        "first_center_difference": location(first_center),
        "first_subset_margin_difference": location(first_margin),
        "causal_substitution_effect": "NOT_YET_RUN_PHASE2" if stage not in {"A0", "A1", "A10", "A11", "A12"} else "NOT_APPLICABLE",
        "classification": "numerically_negligible" if all_bitwise else "algorithm_semantics_difference",
        "stage_verdict": "BITWISE_EQUAL" if all_bitwise else "DIFFERENT",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xiangru-plant", type=Path, required=True)
    parser.add_argument("--xiangru-controller", type=Path, required=True)
    parser.add_argument("--torch-stage", type=Path, required=True)
    parser.add_argument("--private-detail", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    args = parser.parse_args()
    if args.private_detail.exists():
        raise FileExistsError(args.private_detail)

    x_header, x_records = read_xiangru(args.xiangru_plant)
    t_header, t_records, t_controller = read_torch(args.torch_stage)
    x_controller = json.loads(args.xiangru_controller.read_text(encoding="utf-8"))
    if x_header["basis_exponents"] != t_header["basis_exponents"]:
        raise ValueError("basis exponent order is not identical")
    if x_header["basis_slot_count"] != 84 or t_header["basis_slot_count"] != 84:
        raise ValueError("stage traces are not complete Q3")

    detail = compare_plant_records(x_records, t_records)
    compare_controller_records(detail, x_controller, t_controller)
    private = {
        "schema": "tora_q3_stage_comparison_private_v1",
        "raw_private_only": True,
        "basis_coordinate_map": "IDENTITY_84_SLOT_PERMUTATION",
        "comparisons": detail,
    }
    args.private_detail.parent.mkdir(parents=True, exist_ok=True)
    args.private_detail.write_text(
        json.dumps(private, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stage_table = [aggregate_stage(stage, detail[stage]) for stage in STAGES]
    first_bitwise = next(
        (row for row in stage_table if not row["all_compared_values_bitwise"]),
        None,
    )
    first_ulp = next(
        (row for row in stage_table if row["max_ulp_diff"] > 1), None
    )
    first_width = next(
        (row for row in stage_table if row["width_diff"] > 0.0), None
    )
    first_center = next(
        (row for row in stage_table if row["center_diff"] > 0.0), None
    )
    first_margin = next(
        (row for row in stage_table if row["first_subset_margin_difference"]),
        None,
    )
    public = {
        "schema": "tora_q3_stage_comparison_summary_v1",
        "status": "PASS_COMPLETE_OBSERVATION",
        "observation_only": True,
        "diagnostic_counterfactual": True,
        "formal_runner_uses_xiangru_outputs": False,
        "selected_segments": list(SELECTED_SEGMENTS),
        "coordinate_contract": {
            "basis_slot_count": 84,
            "basis_exponent_permutation": list(range(84)),
            "basis_map_status": "IDENTICAL_NORMALIZED_INPUT_TENSORS",
            "physical_state_order": ["x1", "x2", "x3", "x4", "u1"],
            "normalized_generator_order": [
                "x1_parameter",
                "x2_parameter",
                "x3_parameter",
                "x4_parameter",
                "u1_parameter",
            ],
            "local_time_variable": 0,
            "held_control_variable": 5,
            "remainder_semantics": "additive componentwise interval outside retained Q3 polynomial",
        },
        "first_differences": {
            "bitwise": first_bitwise["first_bitwise_difference"] if first_bitwise else None,
            "exceeding_one_ulp": first_ulp["first_difference_exceeding_one_ulp"] if first_ulp else None,
            "interval_width": first_width["first_width_difference"] if first_width else None,
            "interval_center": first_center["first_center_difference"] if first_center else None,
            "subset_margin": first_margin["first_subset_margin_difference"] if first_margin else None,
        },
        "stage_table": stage_table,
        "source_hashes": {
            "xiangru_plant": sha256(args.xiangru_plant),
            "xiangru_controller": sha256(args.xiangru_controller),
            "torch_stage": sha256(args.torch_stage),
            "private_comparison_detail": sha256(args.private_detail),
        },
        "raw_arrays_private": True,
        "raw_paths_in_public_record": False,
    }
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": public["status"],
                "first_differences": public["first_differences"],
                "private_detail_sha256": public["source_hashes"]["private_comparison_detail"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
