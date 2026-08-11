from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from experiments.finalize_three_tool_evidence_package import (
    finalize,
    validate_checksum_coverage,
)
from experiments.run_evidence_command import run


def _make_runner(
    tmp_path: Path,
    summary: dict[str, Any],
    contract: dict[str, Any],
    *,
    extra_files: dict[str, str] | None = None,
    exit_code: int = 0,
) -> Path:
    root = tmp_path / "package"
    runner = root / "scientific" / "gate"
    script = (
        "import json,sys;from pathlib import Path;"
        "root=Path(sys.argv[1]);"
        "(root/'summary.json').write_text(sys.argv[2]);"
        "files=json.loads(sys.argv[3]);"
        "[(lambda p,v:(p.parent.mkdir(parents=True,exist_ok=True),"
        "p.write_text(v)))(root/k,v) for k,v in files.items()];"
        "sys.exit(int(sys.argv[4]))"
    )
    assert (
        run(
            argparse.Namespace(
                output_dir=runner,
                name="scientific_gate",
                source_commit="d" * 40,
                config_json=json.dumps({"scientific_summary": contract}),
                cwd=tmp_path,
                eligibility_status="scientific_gate",
                timing_eligibility="not_a_benchmark",
                expected_exit_codes=(exit_code,),
                command=[
                    sys.executable,
                    "-c",
                    script,
                    "{ARTIFACT_DIR}",
                    json.dumps(summary, allow_nan=False),
                    json.dumps(extra_files or {}),
                    str(exit_code),
                ],
            )
        )
        == 0
    )
    return root


def _operator_summary(outcome: str) -> dict[str, Any]:
    return {
        "schema": "operator_gate_v1",
        "outcome": outcome,
        "scope": "one_step",
        "batch_size": 1,
        "operator_equality": True,
        "initial_mask_equality": True,
        "later_mask_equality": True,
        "endpoint_tube_equality": True,
        "comparison": {
            "kind": "cross_tool",
            "left_tool": "diffreach",
            "right_tool": "torch",
        },
    }


def _contract(*, allowed: tuple[str, ...], **updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": "artifacts/summary.json",
        "schema": "operator_gate_v1",
        "outcome_field": "outcome",
        "allowed_outcomes": list(allowed),
        "required_fields": {},
    }
    value.update(updates)
    return value


def test_exit_zero_with_wrong_scientific_outcome_is_rejected(tmp_path: Path) -> None:
    root = _make_runner(
        tmp_path,
        _operator_summary("WRONG_OUTCOME"),
        _contract(allowed=("DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED",)),
    )
    with pytest.raises(RuntimeError, match="not allowed"):
        finalize(argparse.Namespace(run_root=root))


def test_summary_source_sha_mismatch_is_rejected(tmp_path: Path) -> None:
    summary = {
        "schema": "source_gate_v1",
        "outcome": "SOURCE_REPLAY_PASS",
        "raw_sha256": "0" * 64,
    }
    root = _make_runner(
        tmp_path,
        summary,
        _contract(
            allowed=("SOURCE_REPLAY_PASS",),
            schema="source_gate_v1",
            source_sha256_fields={"artifacts/raw.txt": "raw_sha256"},
        ),
        extra_files={"raw.txt": "raw evidence\n"},
    )
    with pytest.raises(RuntimeError, match="source SHA mismatch"):
        finalize(argparse.Namespace(run_root=root))


def test_g3_expected_exit_one_with_wrong_reason_is_rejected(tmp_path: Path) -> None:
    summary = {
        "schema": "torch_fixed_support_descriptor_bridge_run_v1",
        "outcome": "FIXED_SUPPORT_BRIDGE_BLOCKED",
        "max_gate": "G3",
        "cells": [
            {"cell": "A4", "first_failure": {"reason": "wall_clock_timeout"}}
        ],
    }
    root = _make_runner(
        tmp_path,
        summary,
        _contract(
            allowed=("FIXED_SUPPORT_BRIDGE_BLOCKED",),
            schema="torch_fixed_support_descriptor_bridge_run_v1",
            semantic_profile="bridge_g3",
        ),
        exit_code=1,
    )
    with pytest.raises(RuntimeError, match="not preregistered"):
        finalize(argparse.Namespace(run_root=root))


def test_report_number_must_match_scientific_json(tmp_path: Path) -> None:
    summary = {
        "schema": "report_gate_v1",
        "outcome": "REPORT_READY",
        "validated_horizon": 3.19,
    }
    root = _make_runner(
        tmp_path,
        summary,
        _contract(
            allowed=("REPORT_READY",),
            schema="report_gate_v1",
            report_assertions=[
                {
                    "path": "artifacts/report.md",
                    "pattern": r"validated horizon: (?P<value>[0-9.]+)",
                    "summary_field": "validated_horizon",
                    "type": "float",
                }
            ],
        ),
        extra_files={"report.md": "validated horizon: 3.33\n"},
    )
    with pytest.raises(RuntimeError, match="report/JSON mismatch"):
        finalize(argparse.Namespace(run_root=root))


def test_closed_outcome_with_missing_prerequisite_field_is_rejected(
    tmp_path: Path,
) -> None:
    summary = {
        "schema": "full_gate_v1",
        "outcome": "DIFFREACH_TORCH_DR7_FULL_HORIZON_BIT_EXACT",
    }
    root = _make_runner(
        tmp_path,
        summary,
        _contract(
            allowed=("DIFFREACH_TORCH_DR7_FULL_HORIZON_BIT_EXACT",),
            schema="full_gate_v1",
            outcome_dimension="diffreach_torch_full_horizon_status",
        ),
    )
    with pytest.raises(RuntimeError, match="required field is missing: scope"):
        finalize(argparse.Namespace(run_root=root))


def test_missing_exact_evidence_path_is_rejected(tmp_path: Path) -> None:
    root = _make_runner(
        tmp_path,
        {"schema": "path_gate_v1", "outcome": "PATHS_PRESENT"},
        _contract(
            allowed=("PATHS_PRESENT",),
            schema="path_gate_v1",
            required_paths=["artifacts/required_raw.json"],
        ),
    )
    with pytest.raises(RuntimeError, match="required evidence path is missing"):
        finalize(argparse.Namespace(run_root=root))


def test_checksum_coverage_gap_is_rejected(tmp_path: Path) -> None:
    root = _make_runner(
        tmp_path,
        {"schema": "checksum_gate_v1", "outcome": "CHECKSUM_READY"},
        _contract(allowed=("CHECKSUM_READY",), schema="checksum_gate_v1"),
    )
    finalize(argparse.Namespace(run_root=root))
    (root / "tracked_raw_after_finalization.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="coverage or digest mismatch"):
        validate_checksum_coverage(root)


def test_one_step_evidence_cannot_close_full_horizon(tmp_path: Path) -> None:
    summary = _operator_summary(
        "DIFFREACH_TORCH_DR7_FULL_HORIZON_BIT_EXACT"
    )
    root = _make_runner(
        tmp_path,
        summary,
        _contract(
            allowed=("DIFFREACH_TORCH_DR7_FULL_HORIZON_BIT_EXACT",),
            outcome_dimension="diffreach_torch_full_horizon_status",
        ),
    )
    with pytest.raises(RuntimeError, match="full-horizon closure prerequisite"):
        finalize(argparse.Namespace(run_root=root))


def test_torch_self_parity_cannot_satisfy_cross_tool_operator_gate(
    tmp_path: Path,
) -> None:
    summary = _operator_summary(
        "DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED"
    )
    summary["comparison"] = {
        "kind": "self_parity",
        "left_tool": "torch",
        "right_tool": "torch",
    }
    root = _make_runner(
        tmp_path,
        summary,
        _contract(
            allowed=("DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED",),
            outcome_dimension="diffreach_torch_operator_status",
        ),
    )
    with pytest.raises(RuntimeError, match="operator closure prerequisite"):
        finalize(argparse.Namespace(run_root=root))


def test_true_clone_marker_cannot_be_produced_from_source_root(tmp_path: Path) -> None:
    summary = {
        "schema": "torch_tm_flowpipe_true_clone_gate_v1",
        "outcome": "TRUE_FRESH_CLONE_PASS",
        "source_worktree": str(tmp_path),
        "clone_root": str(tmp_path),
        "origin_clone": True,
        "expected_sha": "e" * 40,
        "checked_out_sha": "e" * 40,
    }
    root = _make_runner(
        tmp_path,
        summary,
        _contract(
            allowed=("TRUE_FRESH_CLONE_PASS",),
            schema="torch_tm_flowpipe_true_clone_gate_v1",
            semantic_profile="true_clone",
        ),
    )
    with pytest.raises(RuntimeError, match="produced from source worktree"):
        finalize(argparse.Namespace(run_root=root))


def test_source_derived_operator_outcome_is_written_to_manifest(tmp_path: Path) -> None:
    outcome = "DIFFREACH_TORCH_DR7_OPERATOR_EQUIVALENCE_CLOSED"
    root = _make_runner(
        tmp_path,
        _operator_summary(outcome),
        _contract(
            allowed=(outcome,),
            outcome_dimension="diffreach_torch_operator_status",
        ),
    )
    manifest = finalize(argparse.Namespace(run_root=root))
    assert manifest["outcome_registry"] == {
        "diffreach_torch_operator_status": outcome
    }


def test_fixture_raw_sha_is_well_formed() -> None:
    # Keep the test module's real source fixture hash computation explicit; a
    # literal all-zero digest must never accidentally become acceptable.
    assert hashlib.sha256(b"raw evidence\n").hexdigest() != "0" * 64
