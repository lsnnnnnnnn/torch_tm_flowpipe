import copy
import json
from pathlib import Path

from torch_tm_flowpipe.archcomp26_preflight import (
    preflight_reasons,
    validate_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def inputs():
    manifest = load("benchmarks/archcomp26/manifest.json")
    matrix = load(manifest["execution_matrix"]["path"])
    return manifest, matrix


def test_current_matrix_is_structurally_valid_and_paused():
    manifest, matrix = inputs()
    assert validate_matrix(manifest, matrix) == []
    reasons = preflight_reasons(manifest, matrix, "quad-reach", "pytorch_gpu")
    assert {
        "experiments_paused",
        "contract_unresolved",
        "support_not_supported",
        "command_argv_missing",
        "command_cwd_missing",
        "source_identity_missing",
        "source_revision_or_sha_missing",
        "binary_identity_missing",
        "runtime_timeout_missing",
        "runtime_budget_missing",
    } <= set(reasons)


def test_partial_nested_override_is_rejected():
    manifest, matrix = inputs()
    broken = copy.deepcopy(matrix)
    broken["cells"]["quad-reach"]["pytorch_gpu"] = {
        "run": {"status": "completed"}
    }
    errors = validate_matrix(manifest, broken)
    assert any("incomplete override" in error for error in errors)


def test_preflight_does_not_treat_unknown_targets_as_launchable():
    manifest, matrix = inputs()
    assert "unknown_instance" in preflight_reasons(
        manifest, matrix, "missing", "pytorch_gpu"
    )
    assert "unknown_method" in preflight_reasons(
        manifest, matrix, "quad-reach", "missing"
    )
