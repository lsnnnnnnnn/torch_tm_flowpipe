from __future__ import annotations

from generate_report import (
    _at_requested_horizon,
    _carry_loss,
)
from collect_results import _annotate_required_metrics
from run_pareto import _pareto_flags


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
