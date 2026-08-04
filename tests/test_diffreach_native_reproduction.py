from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "three_tool_reaudit"
    / "diffreach_native_reproduction.py"
)
SPEC = importlib.util.spec_from_file_location("diffreach_native_reproduction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _boxes(steps: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros((2, steps + 1, 2))
    upper = np.ones((2, steps + 1, 2))
    return lower, upper


def test_completion_requires_every_picard_contraction() -> None:
    contraction = np.ones((4, 2, 2), dtype=bool)
    contraction[2, 1, 0] = False
    lower, upper = _boxes(4)
    result = MODULE.classify_completion(
        contraction=contraction,
        lowers=lower,
        uppers=upper,
        step_size=0.1,
    )
    assert result["validation_status"] == "validation_rejected"
    assert result["failure_category"] == "picard_contraction_rejected"
    assert result["first_failed_step_number"] == 3
    assert result["completed_horizon"] == 0.2
    assert result["upstream_scan_returned_horizon"] == 0.4
    assert result["requested_horizon_reached"] is False


def test_completion_rejects_nonfinite_returned_intervals() -> None:
    contraction = np.ones((2, 1, 2), dtype=bool)
    lower = np.zeros((1, 3, 2))
    upper = np.ones((1, 3, 2))
    upper[0, 1, 0] = np.inf
    result = MODULE.classify_completion(
        contraction=contraction,
        lowers=lower,
        uppers=upper,
        step_size=0.25,
    )
    assert result["validation_status"] == "validation_rejected"
    assert result["failure_category"] == "nonfinite_or_invalid_interval"
    assert result["completed_horizon"] == 0.0


def test_completion_accepts_finite_fully_contracted_trace() -> None:
    contraction = np.ones((3, 1, 1), dtype=bool)
    lower = np.zeros((1, 4, 1))
    upper = np.ones((1, 4, 1))
    result = MODULE.classify_completion(
        contraction=contraction,
        lowers=lower,
        uppers=upper,
        step_size=0.2,
    )
    assert result["validation_status"] == "completed"
    assert result["completed_horizon"] == pytest.approx(0.6)
    assert result["requested_horizon_reached"] is True


def test_effective_support_is_explicit_and_stable() -> None:
    assert MODULE.effective_support(1) == [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]]
    assert len(MODULE.support_sha256(2)) == 64
