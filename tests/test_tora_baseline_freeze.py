from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "outputs/tora_q3_perf_closure_20260806/baseline"


@pytest.mark.regression
@pytest.mark.protocol
def test_baseline_freeze_records_every_required_non_regression_hash() -> None:
    frozen = json.loads((BASELINE / "frozen_hashes.json").read_text())
    assert frozen["common_control"]["accepted_status_sha256"] == (
        "c43283b0c11e1815ad575cbd998a5f6c90ed03f5e363d1a4f3d0dba03cfba54c"
    )
    assert frozen["endpoint_tube_aggregate"]["combined_sha256"] == (
        "c1d8bc660c0e0f6a9f140a31a5a729aa7db9f22cf04be1e51c10ad4f46a4f5cd"
    )
    assert frozen["full_loop_t4_4_failure"]["semantic_summary_sha256"] == (
        "5cc519cd90cda5accc26989d88791ce25351bce34e24dea0497f9236ab224869"
    )
    assert frozen["one_step"]["coefficient_remainder_payload_sha256"] == (
        "a750b2f2174b31be1a29f51a3ab93887ef58d8a68f306035471594deaeb3f838"
    )
    assert frozen["runtime_input"]["controller_trace_sha256"] == (
        "89a225add6e2c02ecb3e84b2182b2f7ea872b064dd9e5e534444552485a091d9"
    )
    gates = frozen["common_control"]["semantic_status"]["gates"]
    assert gates[-1] == {
        "certified_horizon": 20.0,
        "completed_segments": 200,
        "expected_leaf_count": 48,
        "gate": "b48_t20",
        "status": "PASS",
    }
    assert frozen["common_control"]["semantic_status"]["not_independent_closed_loop"]


@pytest.mark.regression
def test_one_step_is_hash_identical_across_py11_and_matched_crown_stack() -> None:
    py11 = (BASELINE / "one_step_py11_summary.json").read_bytes()
    matched = (BASELINE / "one_step_matched_crown_summary.json").read_bytes()
    assert py11 == matched
