#!/usr/bin/env python3
"""Run observation-only TORA-Q3 stage substitutions on frozen tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from torch_tm_flowpipe.tora_q3 import _endpoint_bounds, compose_tora_q3_tm
from torch_tm_flowpipe.tora_stage_contract import (
    SELECTED_SEGMENTS,
    model_and_carry_from_xiangru_record,
    model_from_xiangru_tm_payload,
    tensor_tree_to_lists,
)
from torch_tm_flowpipe.tora_stage_counterfactual import (
    candidate_from_xiangru_coefficients,
    compose_xiangru_local_with_torch,
    run_torch_remainder_counterfactual,
    sine_overrides_from_xiangru_stage,
    torch_range_of_xiangru_difference,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_jsonl(path: Path) -> tuple[dict[str, Any], dict[int, Any]]:
    rows: dict[int, Any] = {}
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
        for line in handle:
            row = json.loads(line)
            if row.get("schema") in {
                "xiangru_tora_q3_plant_segment_observation_v1",
                "torch_tora_q3_stage_contract_segment_v1",
            }:
                segment = int(row["segment_index"])
                if segment in SELECTED_SEGMENTS:
                    rows[segment] = row
    if tuple(sorted(rows)) != SELECTED_SEGMENTS:
        raise ValueError("counterfactual input is missing selected segments")
    return header, rows


def interval_metrics(
    reference_lower: Any,
    reference_upper: Any,
    candidate_lower: Any,
    candidate_upper: Any,
) -> dict[str, Any]:
    xl = torch.as_tensor(reference_lower, dtype=torch.float64).cpu()
    xu = torch.as_tensor(reference_upper, dtype=torch.float64).cpu()
    tl = torch.as_tensor(candidate_lower, dtype=torch.float64).cpu()
    tu = torch.as_tensor(candidate_upper, dtype=torch.float64).cpu()
    if xl.shape != xu.shape or xl.shape != tl.shape or xl.shape != tu.shape:
        raise ValueError("counterfactual interval shape mismatch")
    xc = xl + 0.5 * (xu - xl)
    tc = tl + 0.5 * (tu - tl)
    xw = xu - xl
    tw = tu - tl
    torch_contains = (tl <= xl) & (tu >= xu)
    xiangru_contains = (xl <= tl) & (xu >= tu)
    overlap = torch.maximum(xl, tl) <= torch.minimum(xu, tu)
    return {
        "maximum_absolute_lower_difference": float(torch.max(torch.abs(xl - tl))),
        "maximum_absolute_upper_difference": float(torch.max(torch.abs(xu - tu))),
        "maximum_center_difference": float(torch.max(torch.abs(xc - tc))),
        "maximum_width_difference": float(torch.max(torch.abs(xw - tw))),
        "maximum_reference_width": float(torch.max(xw)),
        "maximum_candidate_width": float(torch.max(tw)),
        "scalar_count": int(xl.numel()),
        "torch_contains_xiangru_count": int(torch.count_nonzero(torch_contains)),
        "xiangru_contains_torch_count": int(torch.count_nonzero(xiangru_contains)),
        "overlap_count": int(torch.count_nonzero(overlap)),
        "disjoint_count": int(torch.count_nonzero(~overlap)),
    }


def value_error(reference: Any, candidate: Any) -> float:
    left = torch.as_tensor(reference, dtype=torch.float64).cpu()
    right = torch.as_tensor(candidate, dtype=torch.float64).cpu()
    if left.shape != right.shape:
        raise ValueError("counterfactual value shape mismatch")
    return float(torch.max(torch.abs(left - right)))


def reduction_fraction(baseline: float, candidate: float) -> float | None:
    if baseline == 0.0:
        return None
    return 1.0 - candidate / baseline


def physical_bounds(model: Any, carry: Any) -> dict[str, torch.Tensor]:
    physical = compose_tora_q3_tm(model, carry)
    tube_lower, tube_upper = physical.range_bound(
        context="tora_counterfactual_physical_tube"
    )
    endpoint_lower, endpoint_upper = _endpoint_bounds(physical, h=0.1)
    return {
        "coefficients": physical.poly.coeffs,
        "remainder_lower": physical.rem_lo,
        "remainder_upper": physical.rem_hi,
        "endpoint_lower": endpoint_lower,
        "endpoint_upper": endpoint_upper,
        "tube_lower": tube_lower,
        "tube_upper": tube_upper,
    }


def public_variant(
    x: Mapping[str, Any],
    local_lower: Any,
    local_upper: Any,
    physical: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "local_remainder": interval_metrics(
            x["picard"]["final_remainder"]["lower"],
            x["picard"]["final_remainder"]["upper"],
            local_lower,
            local_upper,
        ),
        "physical_remainder": interval_metrics(
            x["interval_remainder"]["lower"],
            x["interval_remainder"]["upper"],
            physical["remainder_lower"],
            physical["remainder_upper"],
        ),
        "physical_endpoint": interval_metrics(
            x["endpoint"]["lower"],
            x["endpoint"]["upper"],
            physical["endpoint_lower"],
            physical["endpoint_upper"],
        ),
        "physical_tube": interval_metrics(
            x["tube"]["lower"],
            x["tube"]["upper"],
            physical["tube_lower"],
            physical["tube_upper"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xiangru-plant", type=Path, required=True)
    parser.add_argument("--torch-stage", type=Path, required=True)
    parser.add_argument("--expected-xiangru-sha256", required=True)
    parser.add_argument("--expected-torch-sha256", required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.private_output.exists():
        raise FileExistsError(args.private_output)
    x_hash = sha256(args.xiangru_plant)
    t_hash = sha256(args.torch_stage)
    if x_hash != args.expected_xiangru_sha256:
        raise ValueError("Xiangru counterfactual input hash mismatch")
    if t_hash != args.expected_torch_sha256:
        raise ValueError("Torch counterfactual input hash mismatch")
    x_header, x_rows = selected_jsonl(args.xiangru_plant)
    t_header, t_rows = selected_jsonl(args.torch_stage)
    if x_header["basis_exponents"] != t_header["basis_exponents"]:
        raise ValueError("counterfactual basis order mismatch")

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    private_rows: list[dict[str, Any]] = []
    public_rows: dict[str, Any] = {}
    for segment in SELECTED_SEGMENTS:
        x = x_rows[segment]
        t = t_rows[segment]
        stage = x["stage_contract"]
        base, carry = model_and_carry_from_xiangru_record(x, device=device)
        xiangru_k2 = candidate_from_xiangru_coefficients(
            base,
            stage["A2_A5_polynomial_picard"][1]["candidate"][
                "coefficients"
            ],
        )
        k2_only = run_torch_remainder_counterfactual(base, xiangru_k2)
        k2_and_sine = run_torch_remainder_counterfactual(
            base,
            xiangru_k2,
            sine_overrides=sine_overrides_from_xiangru_stage(
                stage, device=device
            ),
        )
        k2_physical = physical_bounds(k2_only.final, carry)
        sine_physical = physical_bounds(k2_and_sine.final, carry)
        baseline = t["stages"]["A9"]
        baseline_physical = {
            "remainder_lower": baseline["physical_remainder"]["lower"],
            "remainder_upper": baseline["physical_remainder"]["upper"],
            "endpoint_lower": baseline["physical_endpoint"]["lower"],
            "endpoint_upper": baseline["physical_endpoint"]["upper"],
            "tube_lower": baseline["physical_tube"]["lower"],
            "tube_upper": baseline["physical_tube"]["upper"],
        }
        baseline_public = public_variant(
            x,
            baseline["local_final"]["remainder"]["lower"],
            baseline["local_final"]["remainder"]["upper"],
            baseline_physical,
        )
        k2_public = public_variant(
            x, k2_only.final.rem_lo, k2_only.final.rem_hi, k2_physical
        )
        sine_public = public_variant(
            x,
            k2_and_sine.final.rem_lo,
            k2_and_sine.final.rem_hi,
            sine_physical,
        )

        same_range = torch_range_of_xiangru_difference(
            base,
            stage["A6_polynomial_difference"]["polynomial"][
                "coefficients"
            ],
        )
        range_public = interval_metrics(
            stage["A6_polynomial_difference"]["range"]["lower"],
            stage["A6_polynomial_difference"]["range"]["upper"],
            same_range[0],
            same_range[1],
        )
        xiangru_local_payload = {
            "polynomial": x["picard"]["final_polynomial"],
            "remainder": x["picard"]["final_remainder"],
        }
        same_composition = compose_xiangru_local_with_torch(
            xiangru_local_payload, carry, device=device
        )
        composition_public = public_variant(
            x,
            x["picard"]["final_remainder"]["lower"],
            x["picard"]["final_remainder"]["upper"],
            {
                "remainder_lower": same_composition.physical.rem_lo,
                "remainder_upper": same_composition.physical.rem_hi,
                "endpoint_lower": same_composition.endpoint_lower,
                "endpoint_upper": same_composition.endpoint_upper,
                "tube_lower": same_composition.tube_lower,
                "tube_upper": same_composition.tube_upper,
            },
        )
        composition_public["physical_polynomial_maximum_error"] = value_error(
            x["polynomial_coefficient_vector"],
            same_composition.physical.poly.coeffs,
        )

        baseline_remainder_error = max(
            baseline_public["local_remainder"][
                "maximum_absolute_lower_difference"
            ],
            baseline_public["local_remainder"][
                "maximum_absolute_upper_difference"
            ],
        )
        k2_remainder_error = max(
            k2_public["local_remainder"][
                "maximum_absolute_lower_difference"
            ],
            k2_public["local_remainder"][
                "maximum_absolute_upper_difference"
            ],
        )
        sine_remainder_error = max(
            sine_public["local_remainder"][
                "maximum_absolute_lower_difference"
            ],
            sine_public["local_remainder"][
                "maximum_absolute_upper_difference"
            ],
        )
        xiangru_margin_rows = [
            stage["A7_initial_remainder_image"]["subset_margin"],
            *(row["subset_margin"] for row in stage["A8_remainder_rounds"]),
        ]
        k2_margin_rows = [
            k2_only.initial_margin,
            *(row["subset_margin"] for row in k2_only.rounds),
        ]
        sine_margin_rows = [
            k2_and_sine.initial_margin,
            *(row["subset_margin"] for row in k2_and_sine.rounds),
        ]
        margin_errors = {
            "xiangru_k2_then_torch": max(
                value_error(xvalue, tvalue)
                for xvalue, tvalue in zip(
                    xiangru_margin_rows, k2_margin_rows, strict=True
                )
            ),
            "xiangru_k2_and_sine_then_torch": max(
                value_error(xvalue, tvalue)
                for xvalue, tvalue in zip(
                    xiangru_margin_rows, sine_margin_rows, strict=True
                )
            ),
        }
        diagnostic_rows = t["diagnostic_counterfactual_same_input"]
        xiangru_sources = [
            stage["A7_initial_remainder_image"],
            *stage["A8_remainder_rounds"],
        ]
        sine_boundary = {
            "point_enclosure_maximum_error": 0.0,
            "retained_polynomial_maximum_error": 0.0,
            "composition_overflow_maximum_width_difference": 0.0,
            "analytic_remainder_maximum_width_difference": 0.0,
            "sine_output_remainder_maximum_width_difference": 0.0,
            "same_input_integration_maximum_error": 0.0,
        }
        integration_boundary = {
            "propagated_remainder_maximum_width_difference": 0.0,
            "degree_overflow_maximum_width_difference": 0.0,
            "output_remainder_maximum_width_difference": 0.0,
        }
        for source, diagnostic in zip(
            xiangru_sources, diagnostic_rows, strict=True
        ):
            xsine = source["rhs"]["sine"]
            tsine = diagnostic["sine"]
            for name in ("point_sine", "point_cosine"):
                point = interval_metrics(
                    xsine[name]["lower"],
                    xsine[name]["upper"],
                    tsine[name]["lower"],
                    tsine[name]["upper"],
                )
                sine_boundary["point_enclosure_maximum_error"] = max(
                    sine_boundary["point_enclosure_maximum_error"],
                    point["maximum_absolute_lower_difference"],
                    point["maximum_absolute_upper_difference"],
                )
            sine_boundary["retained_polynomial_maximum_error"] = max(
                sine_boundary["retained_polynomial_maximum_error"],
                value_error(
                    xsine["retained_polynomial"]["coefficients"],
                    tsine["retained_polynomial"]["coefficients"],
                ),
            )
            for name, target in (
                (
                    "composition_overflow",
                    "composition_overflow_maximum_width_difference",
                ),
                (
                    "analytic_remainder",
                    "analytic_remainder_maximum_width_difference",
                ),
            ):
                comparison = interval_metrics(
                    xsine[name]["lower"],
                    xsine[name]["upper"],
                    tsine[name]["lower"],
                    tsine[name]["upper"],
                )
                sine_boundary[target] = max(
                    sine_boundary[target],
                    comparison["maximum_width_difference"],
                )
            output_comparison = interval_metrics(
                xsine["output"]["remainder"]["lower"],
                xsine["output"]["remainder"]["upper"],
                tsine["output"]["remainder"]["lower"],
                tsine["output"]["remainder"]["upper"],
            )
            sine_boundary[
                "sine_output_remainder_maximum_width_difference"
            ] = max(
                sine_boundary[
                    "sine_output_remainder_maximum_width_difference"
                ],
                output_comparison["maximum_width_difference"],
            )
            xintegration = source["integration"]["output"]
            tintegration = diagnostic["integration"]["output"]
            integration_remainder = interval_metrics(
                xintegration["remainder"]["lower"],
                xintegration["remainder"]["upper"],
                tintegration["remainder"]["lower"],
                tintegration["remainder"]["upper"],
            )
            sine_boundary["same_input_integration_maximum_error"] = max(
                sine_boundary["same_input_integration_maximum_error"],
                value_error(
                    xintegration["polynomial"]["coefficients"],
                    tintegration["polynomial"]["coefficients"],
                ),
                integration_remainder["maximum_absolute_lower_difference"],
                integration_remainder["maximum_absolute_upper_difference"],
            )
            torch_integrated = model_from_xiangru_tm_payload(
                source["integration"]["input"], device=device
            ).integrate(0)
            torch_propagated = torch_integrated.ledger.entries[
                "initial_remainder"
            ]
            torch_overflow = torch_integrated.ledger.entries[
                "integration_overflow"
            ]
            for xname, torch_value, target in (
                (
                    "propagated_remainder",
                    torch_propagated,
                    "propagated_remainder_maximum_width_difference",
                ),
                (
                    "degree_overflow",
                    torch_overflow,
                    "degree_overflow_maximum_width_difference",
                ),
            ):
                category_comparison = interval_metrics(
                    source["integration"][xname]["lower"],
                    source["integration"][xname]["upper"],
                    torch_value[0],
                    torch_value[1],
                )
                integration_boundary[target] = max(
                    integration_boundary[target],
                    category_comparison["maximum_width_difference"],
                )
            integration_boundary[
                "output_remainder_maximum_width_difference"
            ] = max(
                integration_boundary[
                    "output_remainder_maximum_width_difference"
                ],
                integration_remainder["maximum_width_difference"],
            )
        public_rows[str(segment)] = {
            "segment": segment,
            "baseline_torch": baseline_public,
            "diagnostic_xiangru_k2_then_torch_remainder": k2_public,
            "diagnostic_xiangru_k2_and_sine_then_torch_remainder": sine_public,
            "diagnostic_same_polynomial_torch_range": range_public,
            "diagnostic_xiangru_local_then_torch_composition": composition_public,
            "local_remainder_error_reduction": {
                "k2_substitution_fraction": reduction_fraction(
                    baseline_remainder_error, k2_remainder_error
                ),
                "k2_and_sine_substitution_fraction": reduction_fraction(
                    baseline_remainder_error, sine_remainder_error
                ),
                "baseline_maximum_error": baseline_remainder_error,
                "k2_maximum_error": k2_remainder_error,
                "k2_and_sine_maximum_error": sine_remainder_error,
            },
            "subset_margin_maximum_error": margin_errors,
            "same_input_sine_boundary": sine_boundary,
            "same_input_integration_boundary": integration_boundary,
        }
        private_rows.append(
            {
                "segment": segment,
                "k2_only": tensor_tree_to_lists(
                    {
                        "initial_image": {
                            "coefficients": k2_only.initial_image.poly.coeffs,
                            "remainder_lower": k2_only.initial_image.rem_lo,
                            "remainder_upper": k2_only.initial_image.rem_hi,
                        },
                        "roundoff_lower": k2_only.roundoff_lower,
                        "roundoff_upper": k2_only.roundoff_upper,
                        "rounds": k2_only.rounds,
                        "final_coefficients": k2_only.final.poly.coeffs,
                        "final_remainder_lower": k2_only.final.rem_lo,
                        "final_remainder_upper": k2_only.final.rem_hi,
                        "physical": k2_physical,
                    }
                ),
                "k2_and_observed_sine": tensor_tree_to_lists(
                    {
                        "initial_image": {
                            "coefficients": k2_and_sine.initial_image.poly.coeffs,
                            "remainder_lower": k2_and_sine.initial_image.rem_lo,
                            "remainder_upper": k2_and_sine.initial_image.rem_hi,
                        },
                        "roundoff_lower": k2_and_sine.roundoff_lower,
                        "roundoff_upper": k2_and_sine.roundoff_upper,
                        "rounds": k2_and_sine.rounds,
                        "final_coefficients": k2_and_sine.final.poly.coeffs,
                        "final_remainder_lower": k2_and_sine.final.rem_lo,
                        "final_remainder_upper": k2_and_sine.final.rem_hi,
                        "physical": sine_physical,
                    }
                ),
            }
        )

    private = {
        "schema": "tora_q3_stage_counterfactual_private_v1",
        "diagnostic_counterfactual": True,
        "formal_native_result": False,
        "rows": private_rows,
    }
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(
        json.dumps(private, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    k2_reductions = [
        row["local_remainder_error_reduction"]["k2_substitution_fraction"]
        for row in public_rows.values()
    ]
    sine_reductions = [
        row["local_remainder_error_reduction"][
            "k2_and_sine_substitution_fraction"
        ]
        for row in public_rows.values()
    ]
    public = {
        "schema": "tora_q3_stage_counterfactual_summary_v1",
        "status": "PASS_DIAGNOSTIC_COMPLETE",
        "diagnostic_counterfactual": True,
        "formal_native_result": False,
        "formal_runner_uses_xiangru_outputs": False,
        "selected_segments": list(SELECTED_SEGMENTS),
        "substitutions": {
            "xiangru_k2_to_torch_remainder": True,
            "xiangru_sine_aggregate_to_torch_downstream": True,
            "torch_sine_to_xiangru_downstream_proxy": "Torch integration of the exact Xiangru RHS is separately compared, so the integration boundary is observed rather than assumed",
            "same_polynomial_torch_range_vs_xiangru_range": True,
            "same_local_tm_torch_carry_vs_xiangru_physical_output": True,
            "same_controller_input": "recorded by the stage observation contract at R1 and R2",
        },
        "per_segment": public_rows,
        "aggregate_effect": {
            "minimum_k2_only_local_remainder_error_reduction_fraction": min(
                value for value in k2_reductions if value is not None
            ),
            "minimum_k2_and_sine_local_remainder_error_reduction_fraction": min(
                value for value in sine_reductions if value is not None
            ),
            "maximum_k2_and_sine_local_remainder_error_reduction_fraction": max(
                value for value in sine_reductions if value is not None
            ),
        },
        "source_hashes": {
            "xiangru_stage_trace": x_hash,
            "torch_stage_trace": t_hash,
            "private_counterfactual": sha256(args.private_output),
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
                "aggregate_effect": public["aggregate_effect"],
                "private_counterfactual_sha256": public["source_hashes"][
                    "private_counterfactual"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
