from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.run_tora_q3_h005_fallback import (
    compose_half_step_pair,
    half_step_adapter,
)
from experiments.run_tora_q3_native_hierarchical import derive_gates
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_box_model,
    identity_tora_q3_carry,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tora_q3_stage_parity_fused_20260809/native_full_loop"


def load_json(name: str) -> dict[str, object]:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_row(segment: int, *, failed_leaf: int | None = None) -> dict[str, object]:
    predicates = {}
    for name in (
        "finite_ok_by_leaf",
        "initial_subset_ok_by_leaf",
        "all_remainder_rounds_ok_by_leaf",
        "local_property_ok_by_leaf",
        "composed_property_ok_by_leaf",
        "overall_accepted_by_leaf",
    ):
        values = [True] * 48
        if failed_leaf is not None and name in {
            "composed_property_ok_by_leaf",
            "overall_accepted_by_leaf",
        }:
            values[failed_leaf] = False
        predicates[name] = values
    return {"segment_index": segment, "predicates": predicates}


@pytest.mark.regression
@pytest.mark.protocol
def test_gate_derivation_is_strict_previous_pass_only() -> None:
    rows = [accepted_row(index) for index in range(1, 44)]
    rows.append(accepted_row(44, failed_leaf=0))
    summary = {
        "completed_segments": 43,
        "first_failure": {
            "segment": 44,
            "reason": "property",
            "failed_leaf_ids": [0],
        },
    }
    gates = derive_gates(rows, summary)
    assert [row["gate"] for row in gates] == [
        "one_leaf_one_step",
        "b48_one_step",
        "b48_t1",
        "b48_t5",
        "b48_t10",
        "b48_t20",
    ]
    assert [row["status"] for row in gates] == [
        "PASS",
        "PASS",
        "PASS",
        "FAIL",
        "NOT_RUN",
        "NOT_RUN",
    ]
    assert gates[3]["certified_horizon"] == pytest.approx(4.3)
    assert gates[3]["predicate_counts"]["finite_ok"]["all"]
    assert not gates[3]["predicate_counts"]["composed_property_ok"]["all"]
    assert gates[4]["predicate_counts"] is None
    assert gates[5]["predicate_counts"] is None


@pytest.mark.regression
@pytest.mark.protocol
def test_public_hierarchical_result_is_source_config_and_trace_bound() -> None:
    public = load_json("hierarchical_gates.json")
    assert public["status"] == "CASE_C_PERFORMANCE_PASS_NATIVE_T5_GATE_FAIL"
    assert public["strict_previous_pass_only"]
    assert public["diagnostic_continuation_is_not_formal"]
    assert public["common_control_substitution_forbidden"]
    assert public["publisher_source_sha256"] == file_sha256(
        ROOT / "scripts/summarize_tora_q3_native_closure.py"
    )
    expected = {
        "baseline_native_k2": (4.3, "baseline_native", 2, 0.1),
        "k3_picard": (4.4, "k3_picard", 3, 0.1),
        "algorithm_aligned_q3": (4.3, "algorithm_aligned_q3", 2, 0.1),
        "algorithm_aligned_h005_refresh1": (
            4.4,
            "algorithm_aligned_h005_refresh1",
            2,
            0.05,
        ),
    }
    for lane, (horizon, method, rounds, step_size) in expected.items():
        implementation = public["implementations"][lane]
        assert implementation["certified_horizon"] == horizon
        assert [row["status"] for row in implementation["gates"]] == [
            "PASS",
            "PASS",
            "PASS",
            "FAIL",
            "NOT_RUN",
            "NOT_RUN",
        ]
        config = implementation["config"]
        assert config["lane"] == method
        assert config["polynomial_picard_rounds"] == rounds
        assert config["remainder_picard_rounds"] == 10
        assert config["step_size"] == step_size
        assert implementation["numerical_certificate_passed_at_failure"] is True
        for relative, digest in implementation["source_sha256"].items():
            assert file_sha256(ROOT / relative) == digest
        assert all(
            len(digest) == 64
            for digest in implementation["private_trace_sha256"].values()
        )
    assert public["xiangru_native_reference"]["status"] == "VERIFIED"
    assert public["xiangru_native_reference"]["certified_horizon"] == 20.0


@pytest.mark.regression
@pytest.mark.protocol
def test_h005_is_the_only_fallback_and_declares_its_changed_contract() -> None:
    public = load_json("hierarchical_gates.json")
    decision = public["fallback_decision"]
    assert decision["candidate_count"] == 1
    assert decision["selected"] == "algorithm_aligned_h005_refresh1"
    assert "h=0.05" in decision["contract_change"]
    assert "1.0 second" in decision["contract_change"]
    assert (
        decision["evidence"][
            "algorithm_aligned_segment44_interval_remainder_share_of_sum"
        ]
        > 0.98
    )
    config = public["implementations"][decision["selected"]]["config"]
    assert config["plant_substeps_per_reporting_step"] == 2
    assert config["plant_substeps_per_controller_refresh"] == 20
    assert config["reporting_step_size"] == 0.1
    assert config["controller_refresh_period"] == 1.0


@pytest.mark.regression
@pytest.mark.protocol
def test_t5_failure_never_publishes_t10_or_t20_torch_widths() -> None:
    public = load_json("hierarchical_gates.json")
    availability = public["torch_target_width_availability"]
    assert all(
        values == {"T5": None, "T10": None, "T20": None}
        for values in availability.values()
    )
    first_failures = {
        lane: float(item["first_failure"]["segment"]) * 0.1
        for lane, item in public["implementations"].items()
    }
    for name in (
        "endpoint_width_over_time.csv",
        "tube_width_over_time.csv",
        "remainder_width_over_time.csv",
        "property_margin_over_time.csv",
    ):
        with (OUTPUT / name).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for lane, failure_time in first_failures.items():
            selected = [row for row in rows if row["formal_lane"] == lane]
            assert selected
            assert max(float(row["physical_time"]) for row in selected) <= failure_time
            assert all(float(row["physical_time"]) < 5.0 for row in selected)
        xiangru = [row for row in rows if row["formal_lane"] == "xiangru_native_q3"]
        assert max(float(row["physical_time"]) for row in xiangru) == 20.0


@pytest.mark.regression
@pytest.mark.protocol
def test_failure_evidence_splits_property_from_numerical_certificates() -> None:
    details = load_json("failure_details.json")
    expected = {
        "baseline_native_k2": (44, [0]),
        "k3_picard": (45, [0, 1, 6]),
        "algorithm_aligned_q3": (44, [0]),
        "algorithm_aligned_h005_refresh1": (45, [0, 1, 6]),
    }
    for lane, (segment, leaves) in expected.items():
        failure = details["lanes"][lane]
        assert failure["failure_type"] == "property"
        assert failure["segment_index"] == segment
        assert failure["failed_leaf_ids"] == leaves
        assert failure["numerical_certificate_passed"] is True
        assert [row["state"] for row in failure["states"]] == list(
            ("x1", "x2", "x3", "x4", "u1")
        )
        assert len(failure["controller_input"]) == 4
        assert failure["controller_output"] is not None
        assert failure["polynomial_remainder_decomposition"]
        assert failure["ledger_widths"]
        assert min(
            row["property_margin_minimum"]
            for row in failure["states"][:4]
        ) < 0.0


@pytest.mark.regression
@pytest.mark.protocol
def test_native_public_artifacts_contain_no_private_paths_or_raw_leaf_traces() -> None:
    forbidden = (
        "/srv/",
        "private_verification_evidence",
        "controllerTora.onnx",
        "TORA_CONTROLLER_PATH=",
    )
    for path in OUTPUT.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path.name
    serialized = (OUTPUT / "hierarchical_gates.json").read_text(encoding="utf-8")
    assert '"endpoint":' not in serialized
    assert '"tube":' not in serialized
    assert '"polynomial_coefficient_vector":' not in serialized


@pytest.mark.regression
def test_h005_two_substep_macro_is_finite_and_preserves_reporting_time() -> None:
    lower = torch.tensor([[0.6, -0.7, -0.4, 0.5]], dtype=torch.float64)
    upper = torch.tensor(
        [[0.6125, -0.6833333333333333, -0.3, 0.6]], dtype=torch.float64
    )
    control = torch.tensor([10.0], dtype=torch.float64)
    base = build_tora_q3_box_model(lower, upper, control, control, h=0.1)
    pair = half_step_adapter(
        base,
        capture_trace=False,
        polynomial_picard_rounds=2,
        point_enclosure_backend="eager",
    )
    step = compose_half_step_pair(
        pair, identity_tora_q3_carry(1, device="cpu")
    )
    assert step.accepted
    assert step.segment_tm.domain_hi[0, 0].item() == 0.1
    assert torch.all(step.tube_lower <= pair.first.tube_lower)
    assert torch.all(step.tube_lower <= pair.second.tube_lower)
    assert torch.all(step.tube_upper >= pair.first.tube_upper)
    assert torch.all(step.tube_upper >= pair.second.tube_upper)
    assert torch.all(step.endpoint_lower <= pair.second.endpoint_lower)
    assert torch.all(step.endpoint_upper >= pair.second.endpoint_upper)
