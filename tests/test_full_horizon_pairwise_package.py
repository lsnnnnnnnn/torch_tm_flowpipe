from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.verify_full_horizon_pairwise_package import verify


SOURCE = "a" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _finalize(root: Path, required: list[str]) -> None:
    payload = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "verification.json", "SHA256SUMS"}
    ]
    _write(
        root / "manifest.json",
        {
            "schema": "three_tool_full_horizon_pairwise_carry_package_v3",
            "tested_source_sha": SOURCE,
            "required_paths": required,
            "artifacts": [
                {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in payload
            ],
        },
    )
    _write(root / "verification.json", {"status": "pass"})
    checksum = root / "SHA256SUMS"
    checksum.write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(root).as_posix()}\n"
            for path in sorted(root.rglob("*")) if path.is_file() and path != checksum
        ),
        encoding="utf-8",
    )


def _package(tmp_path: Path) -> tuple[Path, list[str]]:
    root = tmp_path / "package"
    required = [
        "04_flowstar_torch_fixed_schedule/common_prefix/summary.json",
        "04_flowstar_torch_fixed_schedule/common_prefix/common_prefix.csv",
        "05_diffreach_torch_full_horizon/cross_tool_comparison/comparison.json",
        "06_carry_reproduction/a4_b1/prestates/before_step_0320.npz",
        "09_dense_cni_parity/parity.json",
        "10_root_cause/root_cause.json",
        "11_single_fix_if_authorized/no_fix_authorized.json",
        "12_pairwise_tables/summary.json",
        "13_figures/summary.json",
        "16_claim_registry_after/registry.json",
    ]
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            _write(path, {"fixture": True})
        elif path.suffix == ".npz":
            import numpy as np

            np.savez(path, state=np.arange(3, dtype=np.float64))
        else:
            path.write_text("fixture\n", encoding="utf-8")
    _write(
        root / "16_claim_registry_after/registry.json",
        {
            "flowstar_torch_fixed_schedule_status": "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY",
            "diffreach_torch_full_horizon_status": "DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED",
            "carry_semantics_status": "CARRY_MISSING_SYMBOLIC_SEMANTICS",
            "dense_cni_parity_status": "DENSE_CNI_PARITY_NOT_EXPRESSIBLE",
            "single_fix_status": "NO_FIX_AUTHORIZED",
        },
    )
    _finalize(root, required)
    return root, required


def test_package_verifier_checks_sha_paths_outcomes_and_checkpoint_load(tmp_path: Path) -> None:
    root, required = _package(tmp_path)
    result = verify(root, expected_source_sha=SOURCE, require_tracked=False, repo_root=tmp_path)
    assert result["status"] == "pass"
    assert result["npz_member_count"] == 1

    with pytest.raises(RuntimeError, match="tested source SHA"):
        verify(root, expected_source_sha="b" * 40, require_tracked=False, repo_root=tmp_path)

    missing = root / required[0]
    missing.unlink()
    with pytest.raises(RuntimeError, match="missing package path"):
        verify(root, expected_source_sha=SOURCE, require_tracked=False, repo_root=tmp_path)


def test_package_verifier_rejects_wrong_scientific_outcome(tmp_path: Path) -> None:
    root, required = _package(tmp_path)
    registry_path = root / "16_claim_registry_after/registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["single_fix_status"] = "SINGLE_IMPLEMENTATION_FIX_PROMOTED"
    _write(registry_path, registry)
    _finalize(root, required)
    with pytest.raises(RuntimeError, match="outcome mismatch"):
        verify(root, expected_source_sha=SOURCE, require_tracked=False, repo_root=tmp_path)
