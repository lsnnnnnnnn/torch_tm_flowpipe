from __future__ import annotations

import json
from pathlib import Path

from experiments import replay_brusselator_sr100_terminal as replay
from experiments import run_brusselator_sr1000_parity as sr1000


ROOT = Path(__file__).resolve().parents[1]


def test_terminal_replay_is_frozen_to_one_attempt_after_published_prefix() -> None:
    assert replay.EXPECTED_ACCEPTED_STEPS == 355
    assert replay.TERMINAL_STEP == 356
    assert replay.QUEUE_CAPACITY == 100
    assert replay.CHECKPOINT_CONTRACT["accepted_prefix_steps"] == 355
    assert replay.CHECKPOINT_CONTRACT["terminal_attempt_step"] == 356


def test_sr1000_followup_contract_freezes_scope_and_numeric_fields() -> None:
    contract = json.loads(
        (ROOT / "benchmarks/brusselator_terminal_sr1000_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["scope"]["baseline_execution_order"] == [
        "sr100_terminal_replay_once",
        "torch_sr1000_once",
    ]
    assert contract["scope"]["additional_benchmarks_allowed"] is False
    assert contract["scope"]["capacity_sweep_allowed"] is False
    assert contract["scope"]["c4_numeric_fix_budget"] == 1
    sr1000 = contract["sr1000"]
    assert sr1000["queue_capacity"] == 1000
    assert sr1000["requested_steps"] == 1000
    assert sr1000["requested_horizon_decimal"] == "20"
    assert sr1000["order"] == 6
    assert sr1000["fixed_step_decimal"] == "0.02"
    assert sr1000["fixed_step_hex"] == float("0.02").hex()
    assert sr1000["target_remainder_decimal"] == ["-1e-4", "1e-4"]
    assert sr1000["cutoff_decimal"] == "1e-10"
    assert sr1000["endpoint_tightening"] is False
    assert sr1000["adaptive_retry"] is False
    decision = contract["decision"]
    assert decision["capacity_sufficient_requires_accepted_steps"] == 1000
    assert decision["capacity_insufficient_status"] == "NOT_SOLELY_QUEUE_RESET_CAPACITY"
    assert decision["material_absolute_threshold_decimal"] == "1e-12"
    assert decision["material_persistence_consecutive_boundaries"] == 3


def test_sr1000_runner_matches_machine_contract_without_capacity_knob() -> None:
    assert sr1000.QUEUE_CAPACITY == 1000
    assert sr1000.REQUESTED_STEPS == 1000
    assert sr1000.HORIZON == 20.0
    assert sr1000.ORDER == 6
    assert sr1000.STEP.hex() == "0x1.47ae147ae147bp-6"
    assert sr1000.PERSISTENCE == 3
