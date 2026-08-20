from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/diagnose_vdp_c2_native_terminal_20260820.py"
SPEC = importlib.util.spec_from_file_location("vdp_c2_terminal_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def test_scheduler_minimum_exhaustion_uses_recorded_next_retry() -> None:
    assert DIAGNOSTIC._scheduler_minimum_exhausted(
        {"failure_type": "minimum_step_reached", "h_min": 0.002},
        {
            "h_attempted": 0.003950348390361663,
            "next_retry_h": 0.0019751741951808317,
            "terminal_internal_step_rejections": 1,
        },
        attempted_h=0.003950348390361663,
    )


@pytest.mark.parametrize(
    ("failure_type", "next_retry_h", "rejections"),
    [
        ("validation_failure", 0.001, 1),
        ("minimum_step_reached", 0.002, 1),
        ("minimum_step_reached", 0.001, 0),
    ],
)
def test_scheduler_minimum_exhaustion_fails_closed(
    failure_type: str, next_retry_h: float, rejections: int
) -> None:
    assert not DIAGNOSTIC._scheduler_minimum_exhausted(
        {"failure_type": failure_type, "h_min": 0.002},
        {
            "h_attempted": 0.004,
            "next_retry_h": next_retry_h,
            "terminal_internal_step_rejections": rejections,
        },
        attempted_h=0.004,
    )


def test_scheduler_checkpoint_reference_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="attempted h mismatch"):
        DIAGNOSTIC._scheduler_minimum_exhausted(
            {"failure_type": "minimum_step_reached", "h_min": 0.002},
            {
                "h_attempted": 0.004,
                "next_retry_h": 0.001,
                "terminal_internal_step_rejections": 1,
            },
            attempted_h=0.003,
        )
