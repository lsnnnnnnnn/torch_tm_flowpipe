from __future__ import annotations

import argparse

import pytest
import torch

from experiments.run_fixed_support_descriptor_bridge import (
    CELLS,
    _run_cell,
    run,
)


@pytest.mark.unit
def test_bridge_ladder_changes_exactly_one_declared_factor() -> None:
    factors = ("support", "picard", "validator", "carry")
    assert [cell["cell"] for cell in CELLS] == ["A0", "A1", "A2", "A3", "A4"]
    for left, right in zip(CELLS, CELLS[1:]):
        changed = [factor for factor in factors if left[factor] != right[factor]]
        assert len(changed) == 1


@pytest.mark.unit
def test_bridge_stage_ledger_covers_remainder_and_constants_do_not_drift() -> None:
    row = _run_cell(
        dict(CELLS[0]),
        batch=1,
        max_steps=1,
        device=torch.device("cpu"),
    )
    assert row["completed_requested_gate"] is True
    assert row["step_size"] == 0.01
    assert row["target_remainder_radius"] == 0.01
    assert row["cutoff"] is None
    assert row["h_min"] == 0.01
    snapshot = row["snapshots"][0]
    assert snapshot["decision"] == "accept"
    assert snapshot["step"] <= row["completed_steps"]
    ledger = snapshot["stage_ledger"]
    assert ledger["coverage_sum_contains_model_remainder"] is True
    assert set(ledger["source_observations"]) == {
        "retained_polynomial_range",
        "truncation_interval",
        "cutoff_interval",
        "polynomial_times_remainder",
        "remainder_times_remainder",
        "integration_overflow",
        "raw_candidate_remainder",
    }


@pytest.mark.unit
def test_bridge_gate_requires_closed_immediate_predecessor(tmp_path) -> None:
    args = argparse.Namespace(
        output_dir=tmp_path / "missing-prior",
        max_gate="G1",
        prior_gate_summary=None,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="PRIOR_BRIDGE_GATE_EVIDENCE_REQUIRED"):
        run(args)

    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"max_gate":"G0","all_cells_completed_gate":false,'
        '"outcome":"FIXED_SUPPORT_BRIDGE_BLOCKED"}\n'
    )
    args = argparse.Namespace(
        output_dir=tmp_path / "bad-prior",
        max_gate="G1",
        prior_gate_summary=bad,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="PRIOR_BRIDGE_GATE_NOT_CLOSED"):
        run(args)


@pytest.mark.unit
def test_g0_rejects_irrelevant_prior_gate_evidence(tmp_path) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text("{}\n")
    args = argparse.Namespace(
        output_dir=tmp_path / "g0",
        max_gate="G0",
        prior_gate_summary=prior,
        device="cpu",
    )
    with pytest.raises(ValueError, match="G0 must not declare"):
        run(args)
