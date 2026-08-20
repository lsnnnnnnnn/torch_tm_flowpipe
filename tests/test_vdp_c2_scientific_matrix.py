from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/run_vdp_c2_scientific_matrix_20260820.py"
SPEC = importlib.util.spec_from_file_location("vdp_c2_scientific_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


def test_native_noncompletion_is_a_valid_scientific_outcome() -> None:
    MATRIX._validate_runner_outcome(
        scenario="native_T10",
        lane="production_c2_candidate",
        returncode=1,
        summary={"status": "failed", "completed_requested_horizon": False},
    )


@pytest.mark.parametrize(
    ("scenario", "returncode", "status", "completed"),
    [
        ("fixed_T6p32", 1, "failed", False),
        ("native_T10", 0, "failed", False),
        ("native_T10", 1, "completed", False),
        ("native_T10", 1, "failed", True),
    ],
)
def test_runner_outcome_mismatches_fail_closed(
    scenario: str, returncode: int, status: str, completed: bool
) -> None:
    with pytest.raises(ValueError):
        MATRIX._validate_runner_outcome(
            scenario=scenario,
            lane="production_c2_candidate",
            returncode=returncode,
            summary={
                "status": status,
                "completed_requested_horizon": completed,
            },
        )


def test_unexpected_runner_exit_fails_closed() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        MATRIX._validate_runner_outcome(
            scenario="native_T10",
            lane="production_c2_candidate",
            returncode=2,
            summary={"status": "failed", "completed_requested_horizon": False},
        )
