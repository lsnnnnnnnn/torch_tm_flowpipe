from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from torch_tm_flowpipe.protocol.gates import (
    REQUIRED_CROSS_TOOL_GATES,
    validate_cross_tool_gate_manifest,
)
from torch_tm_flowpipe.protocol.reaudit import (
    validate_manifest,
    validate_primary_row,
)
from torch_tm_flowpipe.protocol.schema import RUNTIME_BOUNDARY_VERSION


def _primary_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "backend": "torch-sparse",
        "lane": "matched_plant_backend",
        "completed_horizon": 10.0,
        "requested_horizon": 10.0,
        "validation_status": "completed",
        "soundness_level": "safeguarded_float64_not_fully_proved",
        "primary_eligible": True,
        "endpoint_semantics": "raw_endpoint",
        "effective_support_sha256": "f" * 64,
        "runtime_boundary": RUNTIME_BOUNDARY_VERSION,
        "backend_sha": "1" * 40,
        "run_authority": "authoritative",
    }
    row.update(changes)
    return row


@pytest.mark.unit
@pytest.mark.protocol
def test_primary_row_requires_all_fail_closed_contract_fields() -> None:
    assert validate_primary_row(_primary_row()) == []
    for field in (
        "completed_horizon",
        "validation_status",
        "endpoint_semantics",
        "effective_support_sha256",
        "runtime_boundary",
        "backend_sha",
    ):
        row = _primary_row()
        row.pop(field)
        assert validate_primary_row(row), field


@pytest.mark.unit
@pytest.mark.protocol
@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "patched-audit"},
        {"backend": "torch-dense-prototype"},
        {"completed_horizon": 8.0},
        {"validation_status": "validation_rejected"},
        {"run_authority": "smoke"},
        {"run_authority": "exploratory"},
        {"endpoint_semantics": "collapsed_endpoint"},
        {"primary_eligible": "not_applicable"},
    ],
)
def test_invalid_primary_rows_are_never_eligible(changes: dict[str, object]) -> None:
    assert validate_primary_row(_primary_row(**changes))


@pytest.mark.unit
@pytest.mark.protocol
def test_current_gate_manifest_has_exact_eight_pending_gates() -> None:
    root = Path(__file__).parents[1]
    manifest = yaml.safe_load(
        (root / "benchmarks" / "cross_tool_gates.yaml").read_text(encoding="utf-8")
    )
    decision = validate_cross_tool_gate_manifest(manifest, repo_root=root)
    assert decision.errors == ()
    assert decision.pending == REQUIRED_CROSS_TOOL_GATES
    assert not decision.passed


@pytest.mark.unit
@pytest.mark.protocol
def test_verified_gate_requires_real_checksum_report_test_and_scope(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    report = tmp_path / "report.md"
    evidence.write_text("{}\n", encoding="utf-8")
    report.write_text("evidence\n", encoding="utf-8")
    checksum = hashlib.sha256(evidence.read_bytes()).hexdigest()
    gates = {
        name: {"verified": False, "blocker": "pending", "applies_to": ["fixture"]}
        for name in REQUIRED_CROSS_TOOL_GATES
    }
    gates[REQUIRED_CROSS_TOOL_GATES[0]] = {
        "verified": True,
        "evidence": evidence.name,
        "evidence_sha256": checksum,
        "report": report.name,
        "test": "tests/test_reaudit_protocol.py::test_fixture",
        "applies_to": ["fixture"],
    }
    decision = validate_cross_tool_gate_manifest({"gates": gates}, repo_root=tmp_path)
    assert not decision.errors
    assert REQUIRED_CROSS_TOOL_GATES[0] not in decision.pending
    gates[REQUIRED_CROSS_TOOL_GATES[0]]["evidence_sha256"] = "0" * 64
    decision = validate_cross_tool_gate_manifest({"gates": gates}, repo_root=tmp_path)
    assert any("checksum mismatch" in error for error in decision.errors)


@pytest.mark.unit
@pytest.mark.protocol
def test_manifest_schema_requires_all_repository_identities() -> None:
    manifest = {
        "schema_version": "three-tool-reaudit-1.0.0",
        "run_id": "fixture",
        "started_utc": "2026-08-04T00:00:00Z",
        "started_local": "2026-08-04T00:00:00+00:00",
        "host": {},
        "software": {},
        "repositories": {
            name: {"source_kind": "missing"}
            for name in ("torch_tm_flowpipe", "flowstar", "diffreach", "xiangru")
        },
        "flowstar_backend_identity": {},
        "flowstar_binary": {},
        "benchmark_files": [],
        "execution_contract": {},
        "commands": [],
    }
    assert validate_manifest(manifest) == []
    del manifest["repositories"]["xiangru"]
    assert validate_manifest(manifest) == ["missing repository record: xiangru"]
