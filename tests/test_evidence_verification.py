from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from torch_tm_flowpipe.evidence_verification import (
    VERIFICATION_SCHEMA,
    classify_private_path_matches,
    derive_command_claim,
    validate_verification_document,
)


def _command_evidence(directory: Path) -> dict[str, str]:
    directory.mkdir()
    values = {
        "command.txt": "python -m pytest -q\n",
        "stdout.log": "2 passed in 0.01s\n",
        "stderr.log": "",
        "exit_code.txt": "0\n",
        "started_at.txt": "2026-08-11T00:00:00Z\n",
        "finished_at.txt": "2026-08-11T00:00:01Z\n",
    }
    for name, value in values.items():
        (directory / name).write_text(value, encoding="utf-8")
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in values
    }


def test_hardcoded_pass_without_source_is_rejected():
    document = {
        "schema": VERIFICATION_SCHEMA,
        "claims": [
            {
                "claim_id": "tests",
                "status": "pass",
                "source_paths": [],
                "source_sha256": [],
                "command": "pytest -q",
                "exit_code": 0,
                "started_at": None,
                "finished_at": None,
                "derived_by": "hardcoded",
                "derivation_version": 1,
                "scope": "full tests",
                "limitations": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="without source evidence"):
        validate_verification_document(document)


def test_missing_command_source_is_not_run(tmp_path):
    claim = derive_command_claim(
        "tests",
        tmp_path / "absent",
        scope="full test suite",
        repository_root=tmp_path,
    )
    assert claim.status == "not_run"
    assert not claim.source_paths
    assert "absent" in claim.limitations[0]


def test_source_sha_mismatch_fails_closed(tmp_path):
    directory = tmp_path / "command"
    expected = _command_evidence(directory)
    expected["stdout.log"] = "0" * 64
    claim = derive_command_claim(
        "tests",
        directory,
        scope="full test suite",
        repository_root=tmp_path,
        expected_source_sha256=expected,
    )
    assert claim.status == "fail"
    assert "stdout.log" in claim.limitations[0]


def test_private_path_matches_are_classified_not_hidden(tmp_path):
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"cwd":"/srv/local/shengenli/repo"}\n', encoding="utf-8")
    result = classify_private_path_matches(
        [provenance],
        scan_root=tmp_path,
        private_prefix="/srv/local/shengenli",
        provenance_only=["provenance.json"],
    )
    assert result["private_path_present"]
    assert not result["runtime_hidden_dependency"]
    assert result["status"] == "qualified"
    assert result["matches"][0]["category"] == "provenance_only"


def test_unclassified_private_path_is_a_failure(tmp_path):
    replay = tmp_path / "replay.json"
    replay.write_text('{"input":"/srv/local/shengenli/private.json"}\n', encoding="utf-8")
    result = classify_private_path_matches(
        [replay],
        scan_root=tmp_path,
        private_prefix="/srv/local/shengenli",
    )
    assert result["status"] == "fail"
    assert result["runtime_hidden_dependency"]
    assert result["public_replay_dependency"]
