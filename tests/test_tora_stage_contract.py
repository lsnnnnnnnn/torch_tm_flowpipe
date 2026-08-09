from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from torch_tm_flowpipe.batched_dense_tm import BatchedMonomialBasis
from torch_tm_flowpipe.tora_stage_contract import (
    REPLAY_POINTS,
    SELECTED_SEGMENTS,
    STAGE_IDS,
    model_from_xiangru_tm_payload,
    observe_torch_integration_from_xiangru_payload,
    observe_torch_sine_from_xiangru_payload,
    validate_xiangru_stage_record,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_STAGE = (
    ROOT
    / "outputs/tora_q3_stage_parity_fused_20260809/stage_contract"
)


def comparator_module():
    path = ROOT / "scripts/compare_tora_q3_stage_contract.py"
    spec = importlib.util.spec_from_file_location("tora_stage_comparator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scalar_payload(
    *,
    center: float,
    affine: float = 0.0,
    remainder_lower: float = 0.0,
    remainder_upper: float = 0.0,
) -> dict[str, object]:
    basis = BatchedMonomialBasis.build(6, 3, "cpu")
    coefficients = torch.zeros((1, 1, 84), dtype=torch.float64)
    coefficients[0, 0, basis.constant_index] = center
    coefficients[0, 0, basis.term_index((0, 1, 0, 0, 0, 0))] = affine
    return {
        "polynomial": {"coefficients": coefficients.tolist()},
        "remainder": {
            "lower": [[remainder_lower]],
            "upper": [[remainder_upper]],
        },
    }


@pytest.mark.unit
@pytest.mark.protocol
def test_stage_contract_freezes_complete_replay_schedule() -> None:
    assert STAGE_IDS == tuple(f"A{index}" for index in range(13))
    assert SELECTED_SEGMENTS == (1, 2, 10, 40, 43, 44, 45)
    assert REPLAY_POINTS == {
        "S0": (1,),
        "S1": (2,),
        "R1": (10,),
        "R2": (40,),
        "F0": (43, 44, 45),
    }


@pytest.mark.unit
def test_observed_tm_reconstruction_is_exact_and_fails_closed() -> None:
    payload = scalar_payload(
        center=0.25,
        affine=0.5,
        remainder_lower=-0.01,
        remainder_upper=0.02,
    )
    model = model_from_xiangru_tm_payload(payload, device="cpu")
    assert model.poly.coeffs.shape == (1, 1, 84)
    assert model.poly.coeffs[0, 0, model.poly.basis.constant_index] == 0.25
    assert torch.equal(model.rem_lo, torch.tensor([[-0.01]], dtype=torch.float64))
    assert torch.equal(model.rem_hi, torch.tensor([[0.02]], dtype=torch.float64))
    assert model.domain_lo.tolist() == [[0.0, -1.0, -1.0, -1.0, -1.0, -1.0]]
    assert model.domain_hi.tolist() == [[0.1, 1.0, 1.0, 1.0, 1.0, 1.0]]

    malformed = scalar_payload(center=0.0)
    malformed["polynomial"]["coefficients"] = [[[0.0] * 83]]
    with pytest.raises(ValueError, match="shape"):
        model_from_xiangru_tm_payload(malformed, device="cpu")


@pytest.mark.unit
def test_same_input_sine_observation_contains_nonzero_remainder_samples() -> None:
    payload = scalar_payload(
        center=math.pi / 2,
        affine=0.2,
        remainder_lower=-0.01,
        remainder_upper=0.015,
    )
    observation = observe_torch_sine_from_xiangru_payload(
        payload, device="cpu", order=2
    )
    polynomial_lower = float(
        observation["output"]["polynomial_range"]["lower"][0, 0]
    )
    polynomial_upper = float(
        observation["output"]["polynomial_range"]["upper"][0, 0]
    )
    enclosure_lower = polynomial_lower + float(
        observation["output"]["remainder"]["lower"][0, 0]
    )
    enclosure_upper = polynomial_upper + float(
        observation["output"]["remainder"]["upper"][0, 0]
    )
    for generator in np.linspace(-1.0, 1.0, 17):
        for remainder in (-0.01, 0.0, 0.015):
            exact = math.sin(math.pi / 2 + 0.2 * generator + remainder)
            assert enclosure_lower <= exact <= enclosure_upper
    assert observation["replay_equivalence"]["maximum_absolute_error"] <= 1e-15


@pytest.mark.unit
def test_same_input_integration_retains_exact_time_linear_term() -> None:
    payload = scalar_payload(center=2.0)
    observation = observe_torch_integration_from_xiangru_payload(
        payload, device="cpu"
    )
    basis = BatchedMonomialBasis.build(6, 3, "cpu")
    tau = basis.term_index((1, 0, 0, 0, 0, 0))
    coefficients = observation["output"]["polynomial"]["coefficients"]
    assert coefficients[0, 0, tau] == 2.0
    assert torch.count_nonzero(coefficients).item() == 1
    remainder_lower = observation["output"]["remainder"]["lower"]
    remainder_upper = observation["output"]["remainder"]["upper"]
    assert remainder_lower.item() < 0.0 < remainder_upper.item()
    assert abs(remainder_lower.item()) <= 2.0 * torch.finfo(torch.float64).tiny
    assert abs(remainder_upper.item()) <= 2.0 * torch.finfo(torch.float64).tiny


@pytest.mark.unit
def test_stage_comparator_ulp_and_containment_metrics() -> None:
    module = comparator_module()
    value = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    neighbor = np.nextafter(value, np.array([np.inf, -np.inf, np.inf]))
    assert module.maximum_ulp_distance(value, neighbor) == 1

    comparison = module.compare_interval(
        "fixture",
        1,
        {"lower": [[-1.0]], "upper": [[1.0]]},
        {
            "lower": [[np.nextafter(-1.0, -np.inf)]],
            "upper": [[np.nextafter(1.0, np.inf)]],
        },
    )
    assert comparison["maximum_ulp_difference"] == 1
    assert comparison["containment_relation"] == "torch_contains_xiangru"
    assert comparison["maximum_width_difference"] > 0.0


@pytest.mark.regression
@pytest.mark.protocol
def test_committed_stage_observation_is_complete_and_non_invasive() -> None:
    observation = json.loads(
        (PUBLIC_STAGE / "observation_summary.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (PUBLIC_STAGE / "stage_comparison_summary.json").read_text(
            encoding="utf-8"
        )
    )
    equivalence = json.loads(
        (PUBLIC_STAGE / "instrumentation_equivalence_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert observation["status"] == "PASS"
    assert observation["observation_only"]
    assert not observation["formal_runner_uses_xiangru_outputs"]
    assert observation["instrumentation_replay"][
        "all_observed_xiangru_sine_and_integration_replays_bitwise"
    ]
    assert observation["instrumentation_replay"][
        "maximum_same_input_controller_error"
    ] <= 2e-16
    assert comparison["status"] == "PASS_COMPLETE_OBSERVATION"
    assert [row["stage"] for row in comparison["stage_table"]] == list(
        STAGE_IDS
    )
    assert comparison["stage_table"][0]["stage_verdict"] == "BITWISE_EQUAL"
    assert comparison["stage_table"][1]["stage_verdict"] == "BITWISE_EQUAL"
    assert comparison["first_differences"]["bitwise"]["comparison"].endswith(
        "point_sine"
    )
    assert comparison["raw_arrays_private"]
    assert not comparison["raw_paths_in_public_record"]
    behavior = equivalence["uninstrumented_behavior_equivalence"]
    assert equivalence["status"] == "PASS_WITH_DECLARED_TOLERANCE"
    assert behavior["accepted_leaf_counts_equal"]
    assert behavior["segment_count"] == 200
    assert behavior["certified_horizon"] == 20.0
    exporter = equivalence["instrumented_exporter_regression"]
    assert exporter["status"] == "PASS_WITH_DECLARED_TOLERANCE"
    assert exporter["categories"]["accepted_leaves"][
        "maximum_absolute_difference"
    ] == 0.0
    for category in ("endpoint", "tube", "remainder", "controller_output"):
        assert exporter["categories"][category][
            "maximum_absolute_difference"
        ] <= 1e-12


@pytest.mark.external_integration
@pytest.mark.protocol
def test_external_stage_trace_schemas_when_explicitly_supplied() -> None:
    xiangru_value = os.environ.get("TORA_XIANGRU_STAGE_TRACE_PATH")
    torch_value = os.environ.get("TORA_TORCH_STAGE_TRACE_PATH")
    if not xiangru_value or not torch_value:
        pytest.skip("explicit private stage trace paths are required")
    xiangru_path = Path(xiangru_value)
    torch_path = Path(torch_value)
    with xiangru_path.open(encoding="utf-8") as handle:
        xiangru_header = json.loads(next(handle))
    with torch_path.open(encoding="utf-8") as handle:
        torch_header = json.loads(next(handle))
    assert xiangru_header["basis_slot_count"] == 84
    assert torch_header["basis_slot_count"] == 84
    assert xiangru_header["basis_exponents"] == torch_header["basis_exponents"]


@pytest.mark.unit
def test_xiangru_stage_record_validation_rejects_unselected_or_incomplete() -> None:
    with pytest.raises(ValueError, match="selected replay"):
        validate_xiangru_stage_record(
            {
                "schema": "xiangru_tora_q3_plant_segment_observation_v1",
                "segment_index": 3,
            }
        )
