from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from torch_tm_flowpipe.protocol.gates import REQUIRED_CROSS_TOOL_GATES


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "three_tool_reaudit" / "20260804T060058Z"
MANIFEST = yaml.safe_load(
    (ROOT / "benchmarks" / "cross_tool_gates.yaml").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("name", REQUIRED_CROSS_TOOL_GATES)
def test_every_gate_has_current_machine_evidence(name: str) -> None:
    record = MANIFEST["gates"][name]
    path = ROOT / record["evidence"]
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["run_id"] == RUN.name
    assert evidence["gate"] == name
    assert evidence["passed"] is record["verified"]
    assert evidence["applies_to"]
    assert evidence["automated_test"]
    assert evidence["report"]
    assert evidence["inputs"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["evidence_sha256"]
    if not record["verified"]:
        assert record["blocker"]
        assert evidence["blocker"]


def test_current_support_contract_records_match_and_mismatch() -> None:
    evidence = json.loads(
        (RUN / "gate_evidence" / "order_basis_contract.json").read_text(
            encoding="utf-8"
        )
    )
    supports = evidence["facts"]["flowstar_torch_observed_support"]
    assert supports
    assert all(record["all_equal"] for record in supports.values())
    assert evidence["facts"]["diffreach_support_intentionally_not_grouped_as_complete_order"] is True
    assert evidence["passed"] is True


def test_current_summary_has_no_primary_or_headline_rows() -> None:
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "fail_closed_gate_blocked"
    assert summary["headline_comparison_generated"] is False
    assert summary["headline_pareto_generated"] is False
    assert summary["headline_speedup_generated"] is False
    assert summary["gate_counts"] == {"blocked": 3, "passed": 5}
    assert summary["rows"]
    required = {
        "backend",
        "lane",
        "completed_horizon",
        "validation_status",
        "soundness_level",
        "primary_eligible",
        "endpoint_semantics",
        "effective_support_sha256",
        "runtime_boundary",
        "backend_sha",
        "run_authority",
        "blocker",
    }
    assert all(required <= set(row) for row in summary["rows"])
    assert all(row["primary_eligible"] is False for row in summary["rows"])


def test_current_manifest_command_source_records_are_auditable() -> None:
    records = json.loads(
        (RUN / "logs" / "command_records.json").read_text(encoding="utf-8")
    )
    assert len(records) >= 10
    for record in records:
        assert record["command"]
        assert record["cwd"]
        assert type(record["exit_code"]) is int
        assert "stdout_path" in record
        assert "stderr_path" in record
        assert record["capture_kind"] in {
            "direct_stream_capture",
            "direct_reproduction_runner_capture",
            "direct_diagnostic_runner_capture",
            "structured_runner_output",
            "recovered_from_exact_documented_command_and_raw_artifacts",
        }
