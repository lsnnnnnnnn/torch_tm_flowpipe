from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torch_tm_flowpipe.batched_dense_tm import dense_polynomial_picard
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    dense_tora_q3_dr_step,
    tora_q3_rhs,
)
from torch_tm_flowpipe.tora_stage_counterfactual import (
    candidate_from_xiangru_coefficients,
    run_torch_remainder_counterfactual,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE_PARITY = (
    ROOT / "outputs/tora_q3_stage_parity_fused_20260809/stage_parity"
)
ALLOWED_CLASSIFICATIONS = {
    "actual_torch_bug",
    "actual_observer_bug",
    "expected_outward_roundoff",
    "algorithm_semantics_difference",
    "coordinate_map_unavailable",
    "numerically_negligible",
    "dominant_candidate",
}


def one_leaf_model():
    lower = torch.tensor(
        [[0.60, -0.70, -0.40, 0.50]], dtype=torch.float64
    )
    upper = torch.tensor(
        [[0.61, -0.69, -0.39, 0.51]], dtype=torch.float64
    )
    return build_tora_q3_box_model(
        lower,
        upper,
        torch.tensor([9.8], dtype=torch.float64),
        torch.tensor([10.2], dtype=torch.float64),
    )


@pytest.mark.unit
@pytest.mark.regression
def test_counterfactual_remainder_runner_is_exact_when_no_stage_is_replaced() -> None:
    base = one_leaf_model()
    candidate, _trace = dense_polynomial_picard(
        tora_q3_rhs,
        base.without_remainder(),
        tau_index=0,
        order=3,
        iterations=2,
        cutoff_threshold=None,
        capture_trace=False,
    )
    diagnostic = run_torch_remainder_counterfactual(base, candidate)
    reference = dense_tora_q3_dr_step(base)
    assert torch.equal(
        diagnostic.final.poly.coeffs, reference.segment_tm.poly.coeffs
    )
    assert torch.equal(diagnostic.final.rem_lo, reference.segment_tm.rem_lo)
    assert torch.equal(diagnostic.final.rem_hi, reference.segment_tm.rem_hi)
    assert torch.equal(diagnostic.endpoint_lower, reference.endpoint_lower)
    assert torch.equal(diagnostic.endpoint_upper, reference.endpoint_upper)
    assert torch.equal(diagnostic.tube_lower, reference.tube_lower)
    assert torch.equal(diagnostic.tube_upper, reference.tube_upper)


@pytest.mark.unit
def test_counterfactual_contract_fails_closed_on_shape_or_round_drift() -> None:
    base = one_leaf_model()
    with pytest.raises(ValueError, match="coefficient shape"):
        candidate_from_xiangru_coefficients(
            base, torch.zeros((1, 5, 83), dtype=torch.float64)
        )
    candidate = candidate_from_xiangru_coefficients(
        base, base.poly.coeffs
    )
    sine = base.component(2)
    with pytest.raises(ValueError, match="initial plus ten"):
        run_torch_remainder_counterfactual(
            base, candidate, sine_overrides=[sine] * 10
        )
    with pytest.raises(ValueError, match="ten rounds"):
        run_torch_remainder_counterfactual(
            base, candidate, remainder_rounds=9
        )


@pytest.mark.regression
@pytest.mark.protocol
def test_stage_root_cause_is_complete_and_selects_a3_by_causal_evidence() -> None:
    root_cause = json.loads(
        (STAGE_PARITY / "root_cause.json").read_text(encoding="utf-8")
    )
    counterfactual = json.loads(
        (STAGE_PARITY / "counterfactual_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert root_cause["status"] == "PASS_DOMINANT_STAGE_ISOLATED"
    assert root_cause["first_differences"]["first_material"]["stage"] == "A3"
    assert root_cause["t1_0_014211_attribution"][
        "first_numerical_stage"
    ] == "A2"
    assert root_cause["t1_0_014211_attribution"][
        "first_material_stage"
    ] == "A3"
    dominant = root_cause["dominant_candidate"]
    assert dominant["stage"] == "A3"
    assert dominant["point_enclosure_maximum_error"] <= 5e-15
    assert dominant["retained_polynomial_maximum_error"] <= 3e-16
    assert dominant["composition_overflow_maximum_width_difference"] > 1e-2
    assert dominant["minimum_local_remainder_error_reduction_fraction"] > 0.98
    assert all(dominant["selection_criteria"].values())

    rows = root_cause["root_cause_table"]
    assert [row["stage"] for row in rows] == [f"A{index}" for index in range(13)]
    assert all(row["classification"] in ALLOWED_CLASSIFICATIONS for row in rows)
    assert rows[2]["classification"] == "expected_outward_roundoff"
    assert rows[3]["classification"] == "dominant_candidate"
    required = {
        "stage",
        "input_contract_equal",
        "coordinate_map_status",
        "max_abs_lower_diff",
        "max_abs_upper_diff",
        "max_ulp_diff",
        "center_diff",
        "width_diff",
        "containment_relation",
        "remainder_contribution_diff",
        "first_segment",
        "first_leaf",
        "causal_substitution_effect",
        "classification",
    }
    assert all(set(row) == required for row in rows)

    assert counterfactual["diagnostic_counterfactual"]
    assert not counterfactual["formal_native_result"]
    assert not counterfactual["formal_runner_uses_xiangru_outputs"]
    assert max(
        abs(
            row["local_remainder_error_reduction"][
                "k2_substitution_fraction"
            ]
        )
        for row in counterfactual["per_segment"].values()
    ) < 3e-6
    assert counterfactual["aggregate_effect"][
        "minimum_k2_and_sine_local_remainder_error_reduction_fraction"
    ] > 0.98


@pytest.mark.regression
@pytest.mark.protocol
def test_segment_40_category_and_reverse_substitution_are_explicit() -> None:
    root_cause = json.loads(
        (STAGE_PARITY / "root_cause.json").read_text(encoding="utf-8")
    )
    segment = root_cause["segment_40_remainder_attribution"]
    assert segment["dominant_accumulated_ledger_category"] == (
        "composition_overflow"
    )
    assert segment["earliest_material_generator"] == "A3"
    assert segment["pre_projection_interval_remainder_maximum"] == pytest.approx(
        1.2186185882008733
    )
    assert segment["composition_overflow_to_picard_residual_ratio"] > 900.0
    assert segment["same_input_sine_reduction_fraction"] > 0.98
    reverse = root_cause["reverse_sine_substitution"]
    assert reverse["diagnostic_counterfactual"]
    assert not reverse["formal_native_result"]
    assert reverse["maximum_polynomial_difference"] <= 2e-16
    assert reverse["maximum_remainder_width_difference"] > 4e-4
    assert not root_cause["formal_runner_uses_xiangru_outputs"]
    assert not root_cause["raw_paths_in_public_record"]
    report = (ROOT / "TORA_Q3_STAGE_PARITY_ROOT_CAUSE_REPORT.md").read_text(
        encoding="utf-8"
    )
    assert "first numerical difference is A2" in report
    assert "first material and causally dominant difference is A3" in report
