import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from torch_tm_flowpipe import (
    DenseRangePolicy,
    PolynomialODE,
    flowpipe_step_from_tm,
    load_terminal_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "evidence"
    / "vdp_terminal_range_closure"
    / "20260805T055556Z"
    / "02_terminal_state_replay"
    / "original_terminal_checkpoint"
)


def _load_experiment(name: str, filename: str):
    experiments = ROOT / "experiments"
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    spec = importlib.util.spec_from_file_location(name, experiments / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_hashes(trace):
    row = [item for item in trace if item.get("phase") == "polynomial_picard"][-1]
    return {
        "coefficient_sha256": row["coefficient_sha256"],
        "exponent_support_sha256": row["exponent_support_sha256"],
        "basis_hash": row["basis_hash"],
        "effective_degree": row["effective_degree"],
        "picard_iterations": row["iteration"],
    }


def _direct_replay(policy: DenseRangePolicy):
    runner = _load_experiment("terminal_closure_runner", "run_vdp_dense_backend.py")
    contract = runner.load_contract()
    checkpoint = load_terminal_checkpoint(
        CHECKPOINT,
        expected_contract=contract,
        expected_order=contract["requested_order"],
        expected_dtype=contract["dtype"],
    )
    current = checkpoint.normal_state.normalized_initial_tm(contract["requested_order"])
    return flowpipe_step_from_tm(
        PolynomialODE.from_system_spec(contract["canonical_system_spec"]),
        current,
        float(checkpoint.scheduler["h_attempted"]),
        contract["requested_order"],
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode=contract["validation_mode"],
        target_remainder_radius=contract["target_remainder_radius"],
        cutoff_threshold=contract["cutoff"],
        tm_backend="dense",
        dense_device="cpu",
        dense_range_policy=policy,
    )


def test_original_terminal_natural_rejects_and_tighter_replay_closes_unchanged_candidate():
    natural = _direct_replay(DenseRangePolicy())
    tighter = _direct_replay(
        DenseRangePolicy(
            method="adaptive_subdivision",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
            trigger="on_validation_failure",
            named_contexts=("polynomial_truncation",),
        )
    )
    reference = json.loads((CHECKPOINT / "terminal_reference.json").read_text())
    assert natural.status == "failed"
    assert natural.endpoint_raw_tm is None
    assert natural.subset_margin == reference["subset_margin"]
    assert tighter.status == "validated"
    assert tighter.endpoint_raw_tm is not None
    assert all(value >= 0.0 for row in tighter.subset_margin for value in row)
    assert _candidate_hashes(natural.backend_trace) == _candidate_hashes(tighter.backend_trace)
    assert _candidate_hashes(natural.backend_trace) == reference["candidate_hashes"]
    assert natural.backend_counters["sparse_fallback_count"] == 0
    assert tighter.backend_counters["sparse_fallback_count"] == 0


def test_recorded_tighter_replay_and_output_contract(tmp_path):
    replay = _load_experiment("terminal_closure_replay", "replay_vdp_terminal_range.py")
    output = tmp_path / "replay"
    summary = replay.run(
        replay.parse_args(
            [
                "--checkpoint",
                str(CHECKPOINT),
                "--output-dir",
                str(output),
                "--range-method",
                "adaptive_subdivision",
                "--subdivision-depth",
                "1",
                "--max-leaves",
                "4",
                "--split-vars",
                "0,1",
                "--named-contexts",
                "polynomial_truncation",
                "--trigger",
                "on_validation_failure",
                "--device",
                "cpu",
            ]
        )
    )
    assert summary["accepted"] is True
    assert summary["candidate_hashes"] == json.loads((CHECKPOINT / "terminal_reference.json").read_text())[
        "candidate_hashes"
    ]
    assert json.loads((output / "decision.json").read_text())["range_method"] == "adaptive_subdivision"
    assert json.loads((output / "summary.json").read_text())["range_method"] == "adaptive_subdivision"
    assert yaml.safe_load((output / "config_snapshot.yaml").read_text())["range_method"] == "adaptive_subdivision"

    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing non-empty"):
        replay.run(
            replay.parse_args(
                [
                    "--checkpoint",
                    str(CHECKPOINT),
                    "--output-dir",
                    str(output),
                ]
            )
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"
