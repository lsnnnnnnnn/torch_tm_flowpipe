from __future__ import annotations

import math
from pathlib import Path

import pytest

from torch_tm_flowpipe.source_carry_audit import (
    EXPECTED_RATIOS,
    FLOWSTAR_CHANNELS,
    TORCH_CHANNELS,
    accepted_flowstar_rows,
    accepted_torch_rows,
    checkpoint_reproduction,
    derive_package_verification,
    derive_same_prestate_gate,
    derive_scientific_outcome,
    derive_width_minima,
    exact_semantics_micro_oracles,
    finite_float,
    growth_and_ratio_analysis,
    interval_record,
    parse_json_cell,
    runtime_feature_summary,
    source_semantics_map_is_closed,
)
from experiments.verify_flowstar_torch_source_carry_package import verify_checksums


def flow_row(index: int, *, width: str = "1") -> dict[str, str]:
    row = {
        "status": "accepted",
        "accepted": "true",
        "accepted_step_index": str(index),
        "attempt_index_within_step": "1",
        "t_before": str(index * 0.01),
        "t_after": str((index + 1) * 0.01),
        "symbolic_J_size": str(index + 1),
        "flowstar_internal_intermediate_ranges_source_path": "Expression::Picard_ctrunc_normal",
    }
    for lo, hi in FLOWSTAR_CHANNELS.values():
        row[lo] = "0"
        row[hi] = width
    return row


def torch_row(index: int, *, width: str = "1") -> dict[str, str]:
    row = {
        "status": "accepted",
        "carry_step_index": str(index + 1),
        "segment_index": str(index),
    }
    for lo, hi in TORCH_CHANNELS.values():
        row[lo] = "0"
        row[hi] = width
    return row


@pytest.mark.unit
def test_interval_record_preserves_decimal_text_and_binary64_views() -> None:
    record = interval_record("1.0000000000000001", "1.0000000000000002")
    assert record["width_decimal"] == "1E-16"
    assert record["width"] == math.ulp(1.0)
    assert record["width_hex"] == math.ulp(1.0).hex()
    assert record["exact_zero"] is False
    assert record["below_1e_16"] is False


@pytest.mark.unit
def test_interval_record_distinguishes_exact_zero_and_subnormal() -> None:
    zero = interval_record("1.25", "1.25")
    subnormal = interval_record("0", "5e-324")
    assert zero["exact_zero"] and zero["binary64_zero"]
    assert subnormal["exact_zero"] is False
    assert subnormal["subnormal"] is True
    assert subnormal["width_hex"] == "0x0.0000000000001p-1022"


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["", "nan", "inf", "-inf"])
def test_finite_float_rejects_missing_and_nonfinite(raw: str) -> None:
    with pytest.raises(ValueError):
        finite_float(raw, field="fixture")


@pytest.mark.protocol
def test_flowstar_parser_rejects_missing_steps_and_post_failure_acceptance() -> None:
    accepted_flowstar_rows([flow_row(0), flow_row(1)])
    with pytest.raises(ValueError, match="non-contiguous"):
        accepted_flowstar_rows([flow_row(0), flow_row(2)])
    rejected = {"status": "rejected", "accepted": "false"}
    with pytest.raises(ValueError, match="after its first failure"):
        accepted_flowstar_rows([flow_row(0), rejected, flow_row(1)])


@pytest.mark.protocol
def test_torch_parser_never_zero_fills_rejected_candidate() -> None:
    accepted = torch_row(0)
    rejected = {"status": "rejected", "segment_index": "1"}
    assert accepted_torch_rows([accepted, rejected]) == [accepted]
    with pytest.raises(ValueError, match="after its first failure"):
        accepted_torch_rows([accepted, rejected, torch_row(1)])


@pytest.mark.unit
def test_minima_are_recomputed_from_raw_bounds() -> None:
    rows = [flow_row(0, width="0.25"), flow_row(1, width="0.125")]
    minima, contexts = derive_width_minima(rows, context_radius=1)
    assert {item["step"] for item in minima} == {2}
    assert {item["width_decimal"] for item in minima} == {"0.125"}
    assert all(item["below_1e_9_count"] == 0 for item in minima)
    assert len(contexts) == 8


@pytest.mark.regression
def test_published_checkpoint_ratios_are_derived_not_hardcoded_as_verdict() -> None:
    flow = [flow_row(index) for index in range(632)]
    torch = [torch_row(index) for index in range(632)]
    for step, channels in EXPECTED_RATIOS.items():
        for channel, ratio in channels.items():
            torch[step - 1][TORCH_CHANNELS[channel][1]] = repr(ratio)
    rows, verdict = checkpoint_reproduction(flow, torch)
    assert len(rows) == 16
    assert verdict["status"] == "BASELINE_CONCLUSIONS_REPRODUCED"
    assert verdict["maximum_absolute_ratio_deviation"] == 0.0
    torch[631][TORCH_CHANNELS["endpoint_x"][1]] = "99"
    _, changed = checkpoint_reproduction(flow, torch)
    assert changed["status"] == "BASELINE_NOT_REPRODUCIBLE_STOP"


@pytest.mark.unit
def test_exact_dependency_micro_oracles_are_containment_checks() -> None:
    fixtures = exact_semantics_micro_oracles()
    assert [row["fixture"] for row in fixtures] == [
        "affine_exact_carry",
        "quadratic_shared_error_cancellation",
        "cubic_x2y_shared_source_interaction",
    ]
    assert all(row["arithmetic"] == "exact_rational" for row in fixtures)
    assert all(row["intervalized_contains_shared"] for row in fixtures)
    assert fixtures[0]["width_excess"] == "0"
    assert fixtures[1]["width_excess"] != "0"
    assert fixtures[2]["width_excess"] != "0"


@pytest.mark.unit
def test_runtime_features_require_observed_queue_and_path() -> None:
    features = runtime_feature_summary(
        [flow_row(0), flow_row(1)],
        {"symbolic_remainder_enabled": "true"},
        {"reset_mode": "normalized_insertion"},
    )
    assert features["flowstar_symbolic_queue_observed_active_after_first_step"] is True
    assert features["flowstar_expression_picard_observed"] is True
    assert features["torch_direct_monomial_insertion_source_enabled"] is True
    assert features["flowstar_horner_normal_insertion_source_enabled"] is None
    assert features["flowstar_horner_evidence_class"] == (
        "SOURCE_DECLARATION_NOT_RUNTIME_OBSERVED"
    )


@pytest.mark.unit
def test_guarded_ratio_is_suppressed_below_declared_threshold() -> None:
    flow = [flow_row(index, width="0.5") for index in range(80)]
    torch = [torch_row(index, width="1") for index in range(80)]
    minima = []
    for channel, (lo, hi) in FLOWSTAR_CHANNELS.items():
        flow[39][hi] = "1e-10"
        minima.append({"channel": channel, "step": 40, "width": 1e-10})
    analysis = growth_and_ratio_analysis(
        flow, torch, minima, guarded_ratio_threshold=1e-9
    )
    assert all(
        row["guarded_ratio_at_minimum"] is None
        and row["ratio_was_guarded_at_minimum"] is False
        for row in analysis["channels"].values()
    )


@pytest.mark.unit
def test_same_prestate_and_source_map_gates_are_structural() -> None:
    denied = derive_same_prestate_gate(
        coefficient_export="15-digit decimal",
        symbolic_queue_exported=False,
        import_path_available=False,
    )
    assert denied["status"] == "SAME_PRESTATE_LOSSLESS_BRIDGE_NOT_AVAILABLE"
    allowed = derive_same_prestate_gate(
        coefficient_export="binary_exact",
        symbolic_queue_exported=True,
        import_path_available=True,
    )
    assert allowed["lossless_full_prestate_bridge"] is True
    required_stages = [
        "benchmark/model entry",
        "fixed-step reach loop",
        "cross-step carry decomposition",
        "normal polynomial composition",
        "TM multiplication remainder",
        "Picard/validator",
        "endpoint/tube range extraction",
        "serialization/parser/join",
    ]
    rows = [
        {
            "mathematical_stage": stage,
            "flowstar_source": "f.cpp:1 function",
            "torch_source": "f.py:1 function",
            "dependency_consequence": "fixture",
            "first_unequal": index == 2,
        }
        for index, stage in enumerate(required_stages)
    ]
    assert source_semantics_map_is_closed(rows) is True
    assert source_semantics_map_is_closed(rows[:-1]) is False


@pytest.mark.protocol
def test_packager_scientific_verdict_does_not_use_process_exit_code() -> None:
    audit = {
        "outcome": {"statuses": ["BASELINE_CONCLUSIONS_REPRODUCED", "NO_FIX_AUTHORIZED"]},
        "width_classification": "Z0_POSITIVE_WIDTH_ONLY_LOOKS_ZERO",
        "same_prestate_gate": "SAME_PRESTATE_LOSSLESS_BRIDGE_NOT_AVAILABLE",
        "candidate": "NO_FIX_AUTHORIZED",
        "exit_code": 99,
    }
    high_precision = {
        "falsification_result": "NO_NUMERICAL_CONTAINMENT_WITNESS_IN_TESTED_POINTS",
        "proof_status": "NOT_AN_ENCLOSURE_PROOF",
        "exit_code": 17,
    }
    first = derive_package_verification(audit, high_precision)
    audit["exit_code"] = 0
    high_precision["exit_code"] = 0
    assert derive_package_verification(audit, high_precision) == first
    assert first["scientific_outcome_uses_process_exit_code"] is False


@pytest.mark.unit
def test_scientific_outcome_requires_all_candidate_gates() -> None:
    minima = [
        {"width": 0.01, "exact_zero_count": 0, "below_1e_9_count": 0}
        for _ in FLOWSTAR_CHANNELS
    ]
    runtime = {
        "flowstar_symbolic_queue_observed_active_after_first_step": True,
        "flowstar_horner_normal_insertion_source_enabled": True,
        "torch_direct_monomial_insertion_source_enabled": True,
    }
    baseline = {"status": "BASELINE_CONCLUSIONS_REPRODUCED"}
    denied = derive_scientific_outcome(
        baseline_verdict=baseline,
        minima=minima,
        runtime_features=runtime,
        source_map_closed=True,
        lossless_full_prestate_bridge=False,
        independent_candidate_oracle_closed=False,
        flowstar_soundness_gate_closed=False,
    )
    assert denied["candidate_authorized"] is False
    assert denied["statuses"][-1] == "NO_FIX_AUTHORIZED"
    assert denied["statuses"][1] == (
        "FLOWSTAR_WIDTH_MINIMUM_POSITIVE_NOT_NUMERICALLY_NEAR_ZERO"
    )
    assert denied["statuses"][2] == (
        "SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN"
    )
    assert denied["zero_width_classification"] == "Z0_POSITIVE_WIDTH_ONLY_LOOKS_ZERO"
    allowed = derive_scientific_outcome(
        baseline_verdict=baseline,
        minima=minima,
        runtime_features=runtime,
        source_map_closed=True,
        lossless_full_prestate_bridge=True,
        independent_candidate_oracle_closed=True,
        flowstar_soundness_gate_closed=True,
        stock_copied_probe_equivalence_closed=True,
        causal_factor_split_closed=True,
        same_prestate_operator_attribution_closed=True,
    )
    assert allowed["candidate_authorized"] is True
    assert allowed["statuses"][2] == "CAUSAL_SOURCE_DELTA_CLOSED"
    assert allowed["statuses"][-1] == "SOUND_CARRY_CANDIDATE_L1"


@pytest.mark.unit
def test_json_cell_is_strict() -> None:
    assert parse_json_cell("[[1, 2]]", field="fixture") == [[1, 2]]
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_json_cell("[", field="fixture")


@pytest.mark.protocol
def test_package_checksum_verifier_is_complete_and_fail_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "nested" / "second.txt"
    second.parent.mkdir()
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    import hashlib

    rows = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
        for path in (first, second)
    )
    (tmp_path / "SHA256SUMS").write_text(rows, encoding="utf-8")
    assert verify_checksums(tmp_path) == 2
    second.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(tmp_path)
