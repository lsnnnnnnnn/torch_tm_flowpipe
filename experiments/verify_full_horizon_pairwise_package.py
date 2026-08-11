#!/usr/bin/env python3
"""Verify path safety, semantic outcomes, NPZ loadability, and exact SHA coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _safe(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe package path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise RuntimeError(f"missing package path: {relative}")
    return resolved


def verify(
    root: Path,
    *,
    expected_source_sha: str,
    require_tracked: bool,
    repo_root: Path,
) -> dict[str, Any]:
    root = root.resolve()
    manifest = _json(root / "manifest.json")
    if manifest.get("schema") != "three_tool_full_horizon_pairwise_carry_package_v3":
        raise RuntimeError("package schema mismatch")
    if manifest.get("tested_source_sha") != expected_source_sha:
        raise RuntimeError("tested source SHA mismatch")
    for relative in manifest["required_paths"]:
        _safe(root, str(relative))
    expected_artifacts = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "verification.json", "SHA256SUMS"}
    ]
    if manifest["artifacts"] != expected_artifacts:
        raise RuntimeError("manifest artifact coverage mismatch")
    checksums: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest) or relative in checksums:
            raise RuntimeError("invalid SHA256SUMS")
        checksums[relative] = digest
    expected_checksums = {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"
    }
    if checksums != expected_checksums:
        raise RuntimeError("SHA256SUMS coverage mismatch")
    for path in root.rglob("*.json"):
        _json(path)
    npz_members = 0
    for path in root.rglob("*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                np.asarray(archive[name])
                npz_members += 1
    registry = _json(root / "16_claim_registry_after/registry.json")
    exact = {
        "flowstar_torch_fixed_schedule_status": "FLOWSTAR_TORCH_FIXED_SCHEDULE_COMMON_PREFIX_ONLY",
        "diffreach_torch_full_horizon_status": "DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED",
        "carry_semantics_status": "CARRY_MISSING_SYMBOLIC_SEMANTICS",
        "dense_cni_parity_status": "DENSE_CNI_PARITY_NOT_EXPRESSIBLE",
        "single_fix_status": "NO_FIX_AUTHORIZED",
    }
    for field, expected in exact.items():
        if registry.get(field) != expected:
            raise RuntimeError(f"outcome mismatch: {field}")
    if require_tracked:
        untracked = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", str(path)],
                cwd=repo_root,
                capture_output=True,
            )
            if result.returncode != 0:
                untracked.append(path.relative_to(root).as_posix())
        if untracked:
            raise RuntimeError(f"package files are not tracked: {untracked}")
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", expected_source_sha, "HEAD"],
            cwd=repo_root,
        )
        if ancestry.returncode != 0:
            raise RuntimeError("package commit does not descend from tested source")
    result = {
        "schema": "three_tool_full_horizon_pairwise_carry_verification_result_v1",
        "status": "pass",
        "file_count": len(expected_checksums) + 1,
        "npz_member_count": npz_members,
        "tested_source_sha": expected_source_sha,
        "tracked_required": require_tracked,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--require-tracked", action="store_true")
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _args()
    verify(args.package_root, expected_source_sha=args.expected_source_sha, require_tracked=args.require_tracked, repo_root=args.repo_root)
