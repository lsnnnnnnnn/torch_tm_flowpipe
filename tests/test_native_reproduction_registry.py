from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts/native_reproduction/validate_registry.py"
SPEC = importlib.util.spec_from_file_location("validate_registry", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    stdout = evidence / "stdout.log"
    stderr = evidence / "stderr.log"
    stdout.write_text("native stdout\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    command = evidence / "command.json"
    command.write_text(
        json.dumps(
            {
                "argv": ["native", "--config", "config.json"],
                "cwd": "/source",
                "exit_code": 0,
                "stdout": stdout.name,
                "stderr": stderr.name,
                "stdout_sha256": _sha(stdout),
                "stderr_sha256": _sha(stderr),
            }
        ),
        encoding="utf-8",
    )
    fresh = tmp_path / "fresh.json"
    reference = tmp_path / "reference.json"
    comparison = tmp_path / "comparison.json"
    for path in (fresh, reference, comparison):
        path.write_text("{}\n", encoding="utf-8")
    row = {
        "id": "native",
        "repo_path": "/source",
        "source_sha": "a" * 40,
        "source_changed": False,
        "execution_kind": "author_native",
        "native_entrypoint": "run.py",
        "input_hashes": {"config_sha256": "b" * 64},
        "command_evidence": str(command),
        "fresh_artifacts": [{"path": str(fresh), "sha256": _sha(fresh)}],
        "reference_artifacts": [
            {"path": str(reference), "sha256": _sha(reference)}
        ],
        "comparison_result": {
            "path": str(comparison),
            "sha256": _sha(comparison),
        },
        "requested_horizon": 20.0,
        "reached_horizon": 20.0,
        "reproduction_status": "reproduced_exact",
        "completion_status": "completed",
        "certificate_status": "completed",
        "property_status": "verified",
        "soundness_level": "formal",
        "certificate_claim": "formal",
        "primary_comparison_eligible": True,
        "tolerance": None,
    }
    registry = {
        "schema_version": 1,
        "run_id": "fixture",
        "native_reproductions": [row],
        "diagnostics": [],
    }
    return registry, command


def _errors(registry: dict, root: Path) -> list[str]:
    return VALIDATOR.validate_registry(registry, root=root)


def test_valid_reproduced_row_has_closed_evidence_chain(tmp_path: Path) -> None:
    registry, _ = _fixture(tmp_path)
    assert _errors(registry, tmp_path) == []


@pytest.mark.parametrize(
    "field",
    ["source_sha", "native_entrypoint", "input_hashes", "fresh_artifacts", "reference_artifacts", "comparison_result"],
)
def test_reproduced_row_requires_identity_and_comparison(
    tmp_path: Path, field: str
) -> None:
    registry, _ = _fixture(tmp_path)
    registry["native_reproductions"][0][field] = None
    assert _errors(registry, tmp_path)


def test_source_changed_cannot_be_exact(tmp_path: Path) -> None:
    registry, _ = _fixture(tmp_path)
    registry["native_reproductions"][0]["source_changed"] = True
    assert any("source_changed=true" in error for error in _errors(registry, tmp_path))


def test_partial_horizon_cannot_complete_certificate_or_enter_primary(
    tmp_path: Path,
) -> None:
    registry, _ = _fixture(tmp_path)
    row = registry["native_reproductions"][0]
    row["reached_horizon"] = 15.0
    row["completion_status"] = "partial"
    errors = _errors(registry, tmp_path)
    assert any("completed certificate" in error for error in errors)
    assert any("primary eligible" in error for error in errors)


def test_patched_diagnostic_is_excluded_from_native_table(tmp_path: Path) -> None:
    registry, _ = _fixture(tmp_path)
    row = registry["native_reproductions"][0]
    row["execution_kind"] = "patched_diagnostic"
    row["reproduction_status"] = "patched_diagnostic_only"
    assert _errors(registry, tmp_path)


def test_nonformal_soundness_cannot_claim_formal(tmp_path: Path) -> None:
    registry, _ = _fixture(tmp_path)
    registry["native_reproductions"][0]["soundness_level"] = "unknown"
    assert any("cannot claim formal" in error for error in _errors(registry, tmp_path))


def test_declared_tolerance_requires_source(tmp_path: Path) -> None:
    registry, _ = _fixture(tmp_path)
    row = registry["native_reproductions"][0]
    row["reproduction_status"] = "reproduced_with_declared_tolerance"
    row["tolerance"] = {"value": 1e-6, "source": ""}
    assert any("tolerance.source" in error for error in _errors(registry, tmp_path))


@pytest.mark.parametrize(
    "field",
    ["argv", "cwd", "exit_code", "stdout", "stderr", "stdout_sha256", "stderr_sha256"],
)
def test_command_evidence_is_fail_closed(
    tmp_path: Path, field: str
) -> None:
    registry, command = _fixture(tmp_path)
    payload = json.loads(command.read_text(encoding="utf-8"))
    payload.pop(field)
    command.write_text(json.dumps(payload), encoding="utf-8")
    assert _errors(registry, tmp_path)
