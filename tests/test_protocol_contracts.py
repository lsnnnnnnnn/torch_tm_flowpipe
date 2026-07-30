from __future__ import annotations

from pathlib import Path

import pytest

from torch_tm_flowpipe.protocol.eligibility import (
    evaluate_primary_eligibility,
    partition_and_recompute_pareto,
    strict_required_true,
)
from torch_tm_flowpipe.protocol.provenance import (
    canonical_config_identity,
    prepare_output_directory,
)
from torch_tm_flowpipe.protocol.runtime import measure_configuration_step
from torch_tm_flowpipe.protocol.schema import (
    Applicability,
    BoundSemantics,
    FailureCategory,
    RUNTIME_BOUNDARY_VERSION,
    normalize_observation,
)


def eligible_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "tool": "torch_tm_flowpipe",
        "variant": "order2_affine_reset_selected",
        "system": "riccati",
        "h": 0.01,
        "requested_horizon": 1.0,
        "successful_horizon": 1.0,
        "completed_requested_horizon": True,
        "last_valid_step": 100,
        "failure_step": "",
        "failure_category": FailureCategory.COMPLETED.value,
        "failure_message": "",
        "requested_order": 2,
        "effective_order": 2,
        "basis_id": "complete_total_degree_2",
        "remainder_policy": "validated_interval",
        "step_policy": "fixed",
        "bound_semantics": BoundSemantics.RAW_ENDPOINT.value,
        "primary_comparable": True,
        "native_validation_passed": True,
        "analytic_containment_passed": True,
        "trajectory_sanity_passed": True,
        "all_required_repetitions_present": True,
        "runtime_repetitions": 10,
        "cold_total_configuration_time_s": 0.2,
        "steady_total_configuration_time_s": 0.1,
        "engine_internal_time_s": 0.08,
        "compile_or_jit_time_s": 0.0,
        "posthoc_validation_time_s": 0.01,
        "plot_report_time_s": 0.0,
        "runtime_boundary_version": RUNTIME_BOUNDARY_VERSION,
        "evaluation_time": 1.0,
        "width_at_evaluation_time": 2.0,
    }
    row.update(overrides)
    return row


@pytest.mark.unit
@pytest.mark.protocol
def test_required_gate_is_fail_closed() -> None:
    assert strict_required_true(True) is True
    for value in (False, None, "", "unknown", "unavailable", "passed", 1, 0):
        assert strict_required_true(value) is False


@pytest.mark.unit
@pytest.mark.protocol
def test_primary_requires_formal_repeated_complete_raw_row() -> None:
    assert evaluate_primary_eligibility(eligible_row()).eligible
    for field, bad_value in (
        ("runtime_repetitions", 1),
        ("all_required_repetitions_present", False),
        ("completed_requested_horizon", False),
        ("native_validation_passed", None),
        ("analytic_containment_passed", "unknown"),
        ("trajectory_sanity_passed", ""),
        ("bound_semantics", BoundSemantics.TIGHTENED_ENDPOINT.value),
        ("primary_comparable", False),
        ("runtime_boundary_version", "engine_only_v1"),
        ("steady_total_configuration_time_s", 0.0),
    ):
        decision = evaluate_primary_eligibility(eligible_row(**{field: bad_value}))
        assert not decision.eligible, field
        assert decision.reasons
    assert not evaluate_primary_eligibility(
        eligible_row(failure_category="")
    ).eligible


@pytest.mark.unit
@pytest.mark.protocol
def test_explicit_not_applicable_is_separate_from_boolean_gate() -> None:
    row = eligible_row(
        analytic_containment_passed="",
        analytic_containment_applicability=(
            Applicability.NOT_APPLICABLE.value
        ),
    )
    assert evaluate_primary_eligibility(row).eligible
    row["analytic_containment_applicability"] = "unknown"
    assert not evaluate_primary_eligibility(row).eligible


@pytest.mark.unit
@pytest.mark.protocol
def test_pareto_is_recomputed_after_eligibility_filtering() -> None:
    eligible = eligible_row(width_at_evaluation_time=2.0, steady_total_configuration_time_s=2.0)
    excluded = eligible_row(
        variant="single_sweep",
        width_at_evaluation_time=1.0,
        steady_total_configuration_time_s=1.0,
        runtime_repetitions=1,
        all_required_repetitions_present=False,
    )
    primary, rejected = partition_and_recompute_pareto([eligible, excluded])
    assert len(primary) == 1
    assert primary[0]["width_runtime_pareto"] is True
    assert len(rejected) == 1
    assert rejected[0]["width_runtime_pareto"] is False


@pytest.mark.unit
@pytest.mark.protocol
def test_requested_and_successful_horizon_never_alias_prefix_time() -> None:
    row = normalize_observation(
        eligible_row(
            requested_horizon=1.0,
            successful_horizon=0.4,
            completed_requested_horizon=False,
            last_valid_step=40,
            failure_step=41,
            failure_category=FailureCategory.VALIDATION_REJECTED.value,
            failure_message="candidate rejected",
            evaluation_time=0.4,
        )
    )
    assert row["requested_horizon"] == 1.0
    assert row["successful_horizon"] == 0.4
    assert row["completed_requested_horizon"] is False


@pytest.mark.unit
@pytest.mark.protocol
def test_flowstar_order_and_basis_are_in_config_identity() -> None:
    order_two = eligible_row(
        tool="flowstar",
        requested_order=2,
        effective_order=2,
        basis_id="complete_total_degree_2",
    )
    order_four = eligible_row(
        tool="flowstar",
        requested_order=4,
        effective_order=4,
        basis_id="complete_total_degree_4",
    )
    assert canonical_config_identity(order_two) != canonical_config_identity(order_four)


@pytest.mark.unit
@pytest.mark.protocol
def test_nonempty_output_directory_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "stale.csv").write_text("old\n")
    with pytest.raises(FileExistsError):
        prepare_output_directory(output)


@pytest.mark.unit
@pytest.mark.protocol
def test_resume_rejects_stale_manifest(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "RUN_MANIFEST.json").write_text(
        '{"code_sha": "old"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        prepare_output_directory(
            output,
            resume=True,
            expected_resume_manifest={"code_sha": "new"},
        )


@pytest.mark.unit
@pytest.mark.protocol
def test_total_timer_includes_completion_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_tm_flowpipe.protocol.runtime as runtime

    clock = iter((0.0, 1.0, 3.0, 7.0))
    monkeypatch.setattr(runtime.time, "perf_counter", lambda: next(clock))
    events: list[str] = []
    sample = measure_configuration_step(
        lambda: events.append("engine") or "segment",
        lambda value: events.append("range_projection_reset") or value.upper(),
    )
    assert events == ["engine", "range_projection_reset"]
    assert sample.engine_seconds == 2.0
    assert sample.total_seconds == 7.0
    assert sample.completion_result == "SEGMENT"
