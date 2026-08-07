from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.summarize_tora_q3_baseline_runtime import validate_summary


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "outputs/tora_q3_perf_closure_20260806/runtime"


def _valid_summary() -> dict[str, object]:
    repeat = {
        "certified_horizon": 20.0,
        "checksum": 7.0,
        "completed_segments": 200,
        "repeat": 1,
        "status": "VERIFIED",
    }
    repeats = []
    for index in range(1, 6):
        row = deepcopy(repeat)
        row["repeat"] = index
        repeats.append(row)
    return {
        "status": "PASS",
        "controller_trace_sha256": (
            "89a225add6e2c02ecb3e84b2182b2f7ea872b064dd9e5e534444552485a091d9"
        ),
        "batch": 48,
        "dtype": "float64",
        "segments_per_repeat": 200,
        "measured_repeat_count": 5,
        "repeats": repeats,
        "warmup_excluded": {
            "status": "VERIFIED",
            "completed_segments": 200,
            "certified_horizon": 20.0,
        },
    }


@pytest.mark.regression
@pytest.mark.protocol
def test_runtime_summary_validator_fails_closed_on_unstable_output() -> None:
    summary = _valid_summary()
    summary["repeats"][3]["checksum"] = 8.0  # type: ignore[index]
    with pytest.raises(ValueError, match="checksums are unstable"):
        validate_summary(summary, label="synthetic")


@pytest.mark.regression
@pytest.mark.protocol
def test_public_baseline_runtime_has_three_validated_lanes() -> None:
    summary = json.loads((RUNTIME / "baseline_runtime_summary.json").read_text())
    assert summary["status"] == "PASS"
    assert set(summary["lanes"]) == {
        "torch_py11",
        "torch_matched_crown",
        "xiangru_matched_crown",
    }
    assert all(lane["stable_status_and_output"] for lane in summary["lanes"].values())
    assert summary["workload"]["not_independent_closed_loop"]
    ratio = summary["comparisons"]["same_software_stack_descriptive_ratio"]
    assert ratio["torch_over_xiangru"] > 400.0
    assert "not_optimized_speedup" in ratio["classification"]


@pytest.mark.protocol
def test_public_baseline_runtime_contains_no_private_absolute_path() -> None:
    for path in RUNTIME.glob("baseline_runtime_*"):
        text = path.read_text(encoding="utf-8")
        assert "/srv/" not in text
        assert "/home/" not in text
        assert "private_verification_evidence" not in text
