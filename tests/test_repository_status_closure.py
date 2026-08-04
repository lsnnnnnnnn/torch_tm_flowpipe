from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _registry_rows() -> dict[str, dict]:
    registry = json.loads(
        (ROOT / "benchmarks" / "native_reproduction_registry.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        row["id"]: row
        for row in registry["native_reproductions"] + registry["diagnostics"]
    }


def test_observed_environment_and_timeout_statuses_are_not_algorithm_failures() -> None:
    rows = _registry_rows()
    gpu = rows["xiangru_diffreach_tora_u0_gpu_v100"]
    torch = rows["torch_sparse_native_vanderpol_order4_t10"]
    assert gpu["reproduction_status"] == "environment_failed"
    assert torch["reproduction_status"] == "runtime_timeout"
    assert "not a native algorithm rejection" in gpu["notes"]
    assert "not a mathematical solver rejection" in torch["notes"]


def test_private_reference_rows_are_never_labeled_portable() -> None:
    for row in _registry_rows().values():
        paths = [item["path"] for item in row["reference_artifacts"]]
        location = row["reference_evidence_location"]
        if not paths:
            assert location == "not_applicable"
        elif all(Path(path).is_absolute() for path in paths):
            assert location == "server_local_private_reference"
        else:
            assert all(not Path(path).is_absolute() for path in paths)
            assert location == "portable_committed"


def test_human_matrix_uses_machine_statuses() -> None:
    rows = _registry_rows()
    matrix = (ROOT / "docs" / "NATIVE_REPRODUCTION_MATRIX.md").read_text(
        encoding="utf-8"
    )
    for row_id, label in (
        ("xiangru_diffreach_tora_u0_gpu_v100", "Xiangru DiffReach TORA U0, GPU"),
        ("torch_sparse_native_vanderpol_order4_t10", "our Torch sparse VDP order 4"),
    ):
        line = next(line for line in matrix.splitlines() if label in line)
        assert f"`{rows[row_id]['reproduction_status']}`" in line


def test_stale_status_reports_are_explicitly_superseded() -> None:
    for relative in (
        "docs/THREE_TOOL_FINAL_CORRECTNESS_REPORT.md",
        "audits/repository_cleanup/repository_cleanup_20260804T022536Z/FINAL_ACCEPTANCE.md",
    ):
        prefix = (ROOT / relative).read_text(encoding="utf-8")[:1200]
        assert "SUPERSEDED STATUS REPORT" in prefix
        assert "superseded_by" in prefix
