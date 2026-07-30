from __future__ import annotations

import sys
from pathlib import Path

import pytest

from generate_report import (
    _at_requested_horizon,
    _carry_loss,
)
from collect_results import (
    _annotate_required_metrics,
    _pareto_identity,
    _supplemental_tightened_endpoint,
)
from run_ablation import _finite_max_or_unavailable
from run_pareto import (
    _maximum_measured_memory,
    _pareto_flags,
    _projected_affine_box_reset,
)


def _row(
    *,
    tool: str,
    protocol: str,
    time: float,
    width: float,
) -> dict[str, object]:
    return {
        "tool": tool,
        "variant": "candidate",
        "protocol": protocol,
        "system": "van_der_pol",
        "h": "0.01",
        "horizon": "1.0",
        "time": str(time),
        "width": str(width),
    }


def test_requested_horizon_filter_rejects_failed_prefix() -> None:
    assert _at_requested_horizon(
        _row(
            tool="torch",
            protocol="common_affine_carry",
            time=1.0,
            width=1.0,
        )
    )
    assert not _at_requested_horizon(
        _row(
            tool="flowstar",
            protocol="common_affine_carry",
            time=0.13,
            width=0.2,
        )
    )


def test_affine_box_ratio_requires_identical_absolute_time() -> None:
    affine = [
        _row(
            tool="torch",
            protocol="common_affine_carry",
            time=1.0,
            width=1.0,
        )
    ]
    mismatched_box = [
        _row(
            tool="torch",
            protocol="common_box_carry",
            time=0.5,
            width=2.0,
        )
    ]
    assert _carry_loss(affine, mismatched_box) == []

    matched_box = [
        _row(
            tool="torch",
            protocol="common_box_carry",
            time=1.0,
            width=2.0,
        )
    ]
    loss = _carry_loss(affine, matched_box)
    assert len(loss) == 1
    assert loss[0][-1] == 2.0


def test_pareto_flags_never_compare_different_tools() -> None:
    rows = [
        {
            "tool": "torch_tm_flowpipe",
            "system": "riccati",
            "evaluation_time": 0.1,
            "width_at_evaluation_time": 2.0,
            "steady_full_configuration_time_s": 2.0,
        },
        {
            "tool": "flowstar",
            "system": "riccati",
            "evaluation_time": 0.1,
            "width_at_evaluation_time": 1.0,
            "steady_full_configuration_time_s": 1.0,
        },
    ]
    _pareto_flags(rows)
    assert all(row["width_runtime_pareto"] for row in rows)


def test_required_metrics_use_matched_affine_reference_and_explicit_missing() -> None:
    spec = {
        "systems": {
            "riccati": {
                "initial_box": [[0.0, 0.1]],
            }
        }
    }
    rows = [
        {
            **_row(
                tool="torch_tm_flowpipe",
                protocol="common_affine_carry",
                time=0.01,
                width=0.2,
            ),
            "system": "riccati",
            "h": "0.01",
            "state_index": "0",
            "step_index": "1",
            "interval_kind": "endpoint_raw",
            "lower": "0.0",
            "upper": "0.2",
            "native_validation_passed": "true",
        },
        {
            **_row(
                tool="torch_tm_flowpipe",
                protocol="common_box_carry",
                time=0.01,
                width=0.4,
            ),
            "system": "riccati",
            "h": "0.01",
            "state_index": "0",
            "step_index": "1",
            "interval_kind": "endpoint_raw",
            "lower": "-0.1",
            "upper": "0.3",
            "native_validation_passed": "true",
        },
    ]

    _annotate_required_metrics(spec, rows)

    assert rows[0]["interval_center"] == 0.1
    assert rows[0]["center_shift_from_initial"] == 0.05
    assert rows[0]["dependency_loss_metric"] == "reference"
    assert rows[1]["dependency_loss_metric"] == 2.0
    assert rows[1]["accepted_step"] == 0.01
    assert rows[1]["successful_horizon"] == "0.01"
    assert rows[1]["runtime_setup_s"] == "unavailable"
    assert rows[1]["memory_measurement"] == "unavailable"


def test_literature_map_keeps_course_attachments_distinct() -> None:
    study = Path(__file__).parents[1]
    literature = (study / "LITERATURE_MAP.md").read_text(encoding="utf-8")
    missing = (study / "MATERIALS_MISSING.md").read_text(encoding="utf-8")
    assert "Lecture-12.pdf" in literature
    assert "Modeling Physics" in literature
    assert "dynamical systems, stability, and Lyapunov" in literature
    assert "584_homework2.pdf" in literature
    assert "Homework 2" in literature
    assert "Week-4-1-2.pdf" in literature
    assert "Week-4-2-3.pdf" in literature
    assert "public-schedule lecture numbers" in literature
    assert (
        sum(
            line[:1].isdigit() and ".pdf`" in line
            for line in missing.splitlines()
        )
        == 16
    )


def test_torch_pareto_projects_before_affine_only_reset() -> None:
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

    result, discarded = _projected_affine_box_reset(
        nonlinear_endpoint,
        project_to_basis=project,
        affine_reset=reset,
        stage="pareto_cuda_affine_reset_projection",
        iteration=7,
    )

    assert result is reset_endpoint
    assert discarded == 2
    assert calls[0][0] == "project"
    assert calls[1] == ("reset", affine_endpoint, "box")


def test_torch_cuda_pareto_reset_accepts_projected_nonlinear_endpoint() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    followup = Path(__file__).parents[2] / "first_order_followup"
    if str(followup) not in sys.path:
        sys.path.insert(0, str(followup))
    from export_torch_segment import rhs_from_spec
    from torch_basis import (
        affine_reset,
        normalized_initial_tm,
        project_to_basis,
    )
    from torch_tm_flowpipe import flowpipe_step_from_tm

    spec = __import__("common").load_spec()
    system = spec["systems"]["coupled_quadratic"]
    device = torch.device("cuda:0")
    current = normalized_initial_tm(
        system["initial_box"],
        order=4,
        dtype=torch.float64,
        device=device,
    )
    segment = flowpipe_step_from_tm(
        rhs_from_spec(system), current, 0.01, 4
    )
    assert segment.status == "validated"
    assert segment.endpoint_raw_tm is not None
    reset, discarded = _projected_affine_box_reset(
        segment.endpoint_raw_tm,
        project_to_basis=project_to_basis,
        affine_reset=affine_reset,
        stage="pareto_cuda_affine_reset_projection",
        iteration=1,
    )
    assert len(reset) == len(system["initial_box"])
    assert discarded > 0


def test_failed_ablation_widths_are_explicitly_unavailable() -> None:
    assert _finite_max_or_unavailable([]) == "unavailable"
    assert _finite_max_or_unavailable([float("nan"), float("inf")]) == (
        "unavailable"
    )
    assert _finite_max_or_unavailable(["unavailable", 0.2, 0.1]) == 0.2


def test_missing_memory_is_unavailable_not_zero() -> None:
    assert _maximum_measured_memory(["", None, 0.0]) == "unavailable"
    assert _maximum_measured_memory(
        ["unavailable", float("nan"), 1024, 2048]
    ) == 2048.0


def test_tightened_torch_endpoint_is_supplemental_only() -> None:
    row = {
        "tool": "torch_tm_flowpipe",
        "variant": "order1_legacy_tightened",
        "system": "riccati",
        "h": "0.01",
    }
    assert _supplemental_tightened_endpoint(row)
    assert _pareto_identity(row) == (
        "torch_tm_flowpipe",
        "order1_legacy_tightened",
        "riccati",
        "0.01",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    )
