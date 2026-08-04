#!/usr/bin/env python3
"""Inventory Xiangru/private experiment inputs without aliasing the public release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(workspace: Path, output: Path) -> dict[str, Any]:
    expected = workspace / "CROWN-Reach_Development"
    public = workspace / "CROWN-Reach"
    cpp = public / "src" / "CrownReach.cpp"
    as_float: list[dict[str, Any]] = []
    if cpp.is_file():
        for number, line in enumerate(cpp.read_text(encoding="utf-8").splitlines(), 1):
            if "asFloat()" in line:
                as_float.append({"line": number, "text": line.strip()})
    archives = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in sorted(workspace.glob("*.zip"))
        if any(token in path.name.lower() for token in ("xiangru", "crown", "2026", "experiment"))
    ]
    result = {
        "schema_version": "xiangru-source-inventory-1.0.0",
        "private_expected_root": str(expected),
        "private_source_status": "available" if expected.exists() else "blocked_missing_source",
        "private_git_identity": None,
        "private_raw_timing_artifacts": [],
        "matching_archives": archives,
        "historical_timing_recomputed": False,
        "historical_timing_blocker": (
            None
            if expected.exists()
            else "private 2026 experiment source and raw JSON/log artifacts are absent"
        ),
        "public_release": {
            "path": str(public),
            "exists": public.exists(),
            "remote": _git(public, "remote", "get-url", "origin") if public.exists() else None,
            "branch": _git(public, "rev-parse", "--abbrev-ref", "HEAD") if public.exists() else None,
            "sha": _git(public, "rev-parse", "HEAD") if public.exists() else None,
            "dirty": bool(_git(public, "status", "--porcelain")) if public.exists() else None,
            "crown_cpp_path": str(cpp) if cpp.is_file() else None,
            "crown_cpp_sha256": _sha256(cpp) if cpp.is_file() else None,
            "controller_json_as_float_calls": as_float,
            "is_substitute_for_private_2026_experiment": False,
        },
        "headline_b48_ratio_allowed": False,
        "reason": "source identity, B48 workload, completion, property, and runtime boundaries cannot be verified",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.workspace.resolve(), args.output.resolve())
    print(json.dumps({"private_source_status": result["private_source_status"], "headline_b48_ratio_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
