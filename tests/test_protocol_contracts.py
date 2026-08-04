from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from torch_tm_flowpipe.protocol.config import expected_configuration_rows
from torch_tm_flowpipe.protocol.carry import projected_affine_box_reset
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
    BoundKind,
    BoundSemantics,
    ComparisonLane,
    FailureCategory,
    RefinementSemantics,
    RUNTIME_BOUNDARY_VERSION,
    SoundnessLevel,
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
        "effective_degree": 2,
        "basis_id": "complete_total_degree_2",
        "remainder_policy": "validated_interval",
        "step_policy": "fixed",
        "bound_semantics": BoundSemantics.RAW_ENDPOINT.value,
        "bound_kind": BoundKind.ENDPOINT.value,
        "refinement_semantics": RefinementSemantics.RAW.value,
        "endpoint_exporter_semantics": "raw_endpoint_at_requested_horizon",
        "backend_class": "torch-native",
        "backend_sha": "0123456789abcdef",
        "backend_dirty": False,
        "backend_primary_eligible": True,
        "execution_route": "torch-native",
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
        "lane": ComparisonLane.MATCHED_PLANT_BACKEND.value,
        "soundness_level": SoundnessLevel.SAFEGUARDED_FLOAT64_NOT_FULLY_PROVED.value,
        "effective_support_sha256": "a" * 64,
        "validation_status": FailureCategory.COMPLETED.value,
        "run_authority": "authoritative",
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
        ("backend_primary_eligible", False),
        ("backend_class", "patched-audit"),
        ("execution_route", "patched-audit"),
        ("runtime_boundary_version", "engine_only_v1"),
        ("steady_total_configuration_time_s", 0.0),
        ("run_authority", "smoke"),
        ("effective_support_sha256", ""),
        ("soundness_level", "unknown_level"),
    ):
        decision = evaluate_primary_eligibility(eligible_row(**{field: bad_value}))
        assert not decision.eligible, field
        assert decision.reasons
    assert not evaluate_primary_eligibility(
        eligible_row(failure_category="")
    ).eligible


@pytest.mark.unit
@pytest.mark.protocol
@pytest.mark.parametrize(
    ("bound_semantics", "bound_kind", "refinement"),
    [
        (
            BoundSemantics.RAW_ENDPOINT.value,
            BoundKind.ACCEPTED_SEGMENT.value,
            RefinementSemantics.RAW.value,
        ),
        (
            BoundSemantics.RAW_ENDPOINT.value,
            BoundKind.ENDPOINT.value,
            RefinementSemantics.TIGHTENED.value,
        ),
        (
            BoundSemantics.TUBE_BOX.value,
            BoundKind.ENDPOINT.value,
            RefinementSemantics.RAW.value,
        ),
    ],
)
def test_bound_semantics_collisions_fail_closed(
    bound_semantics: str,
    bound_kind: str,
    refinement: str,
) -> None:
    with pytest.raises(ValueError, match="collides"):
        normalize_observation(
            eligible_row(
                bound_semantics=bound_semantics,
                bound_kind=bound_kind,
                refinement_semantics=refinement,
            )
        )


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
@pytest.mark.parametrize(
    ("semantics", "refinement"),
    [
        (BoundSemantics.COLLAPSED_ENDPOINT.value, "collapsed"),
        (BoundSemantics.REPAIRED_HULL.value, "repaired"),
    ],
)
def test_diagnostic_endpoint_semantics_are_distinct_and_primary_ineligible(
    semantics: str, refinement: str
) -> None:
    row = normalize_observation(
        eligible_row(bound_semantics=semantics, refinement_semantics=refinement)
    )
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
def test_primary_pareto_compares_tool_families_cross_tool() -> None:
    left = eligible_row(
        tool="torch_tm_flowpipe",
        width_at_evaluation_time=2.0,
        steady_total_configuration_time_s=2.0,
    )
    right = eligible_row(
        tool="flowstar",
        width_at_evaluation_time=1.0,
        steady_total_configuration_time_s=1.0,
    )
    primary, rejected = partition_and_recompute_pareto([left, right])
    assert not rejected
    flags = {row["tool"]: row["width_runtime_pareto"] for row in primary}
    assert flags == {"torch_tm_flowpipe": False, "flowstar": True}


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


@pytest.mark.unit
@pytest.mark.protocol
def test_nonlinear_endpoint_is_projected_before_affine_reset() -> None:
    nonlinear_endpoint = object()
    affine_endpoint = object()
    reset_endpoint = object()
    calls: list[tuple[object, ...]] = []

    def project(value: object, basis: str, **kwargs: object):
        calls.append(("project", value, basis, kwargs))
        return affine_endpoint, ["quadratic", "cubic"]

    def reset(value: object, *, method: str):
        calls.append(("reset", value, method))
        assert value is affine_endpoint
        return reset_endpoint, {}

    result, discarded = projected_affine_box_reset(
        nonlinear_endpoint,
        project_to_basis=project,
        affine_reset=reset,
        stage="pareto_affine_reset_projection",
        iteration=7,
    )
    assert result is reset_endpoint
    assert discarded == 2
    assert calls[0][0] == "project"
    assert calls[1] == ("reset", affine_endpoint, "box")


@pytest.mark.unit
@pytest.mark.protocol
def test_versioned_profiles_enumerate_exact_expected_configs() -> None:
    root = Path(__file__).parents[1]
    benchmark = yaml.safe_load(
        (root / "benchmarks" / "canonical.yaml").read_text(
            encoding="utf-8"
        )
    )
    smoke = yaml.safe_load(
        (root / "benchmarks" / "smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    formal = yaml.safe_load(
        (root / "benchmarks" / "formal.yaml").read_text(
            encoding="utf-8"
        )
    )
    smoke_rows = expected_configuration_rows(benchmark, smoke)
    formal_rows = expected_configuration_rows(benchmark, formal)
    assert len(smoke_rows) == 12
    assert len(formal_rows) == 24
    assert len({row["config_id"] for row in formal_rows}) == 24


@pytest.mark.unit
@pytest.mark.protocol
def test_formal_profile_is_blocked_until_all_cross_tool_gates_are_verified() -> None:
    root = Path(__file__).parents[1]
    script = root / "experiments" / "consolidated_study" / "cli.py"
    module_spec = importlib.util.spec_from_file_location("consolidated_cli", script)
    assert module_spec is not None and module_spec.loader is not None
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    formal = yaml.safe_load(
        (root / "benchmarks" / "formal.yaml").read_text(encoding="utf-8")
    )
    gates = cli._load_cross_tool_gates(formal)

    assert len(gates["gates"]) == 8
    assert not any(record["verified"] for record in gates["gates"].values())
    with pytest.raises(RuntimeError, match="blocked by unverified gates"):
        cli._require_cross_tool_gates(gates)
    RefinementSemantics,
