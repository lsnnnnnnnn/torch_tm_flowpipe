#!/usr/bin/env python3
"""Collect final repository/provenance state with private raw command output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def run(root: Path, arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        arguments, cwd=root, text=True, capture_output=True
    )
    return {
        "command": arguments,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def stdout(root: Path, arguments: list[str]) -> str:
    result = run(root, arguments)
    if result["exit_code"] != 0:
        raise RuntimeError(f"command failed: {arguments}: {result['stderr']}")
    return str(result["stdout"]).strip()


def repository_summary(root: Path) -> dict[str, Any]:
    status = stdout(root, ["git", "status", "--porcelain=v1"])
    branch = stdout(root, ["git", "branch", "--show-current"])
    upstream_result = run(
        root, ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
    )
    rows = status.splitlines() if status else []
    return {
        "head": stdout(root, ["git", "rev-parse", "HEAD"]),
        "branch": branch or "DETACHED",
        "upstream": (
            str(upstream_result["stdout"]).strip()
            if upstream_result["exit_code"] == 0
            else "NONE"
        ),
        "dirty": bool(rows),
        "status_entry_count": len(rows),
        "modified_entry_count": sum(not row.startswith("??") for row in rows),
        "untracked_entry_count": sum(row.startswith("??") for row in rows),
        "diff_check_exit_code": run(root, ["git", "diff", "--check"])["exit_code"],
        "remote_names": stdout(root, ["git", "remote"]).splitlines(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-root", type=Path, required=True)
    parser.add_argument("--xiangru-root", type=Path, required=True)
    parser.add_argument("--diffreach-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--private-raw", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "torch": args.torch_root.resolve(),
        "xiangru": args.xiangru_root.resolve(),
        "diffreach": args.diffreach_root.resolve(),
    }
    raw = {
        name: {
            "status_short_branch": run(root, ["git", "status", "--short", "--branch"]),
            "status_porcelain_v2": run(root, ["git", "status", "--porcelain=v2", "--branch"]),
            "head": run(root, ["git", "rev-parse", "HEAD"]),
            "log": run(root, ["git", "log", "-10", "--oneline", "--decorate"]),
            "diff_check": run(root, ["git", "diff", "--check"]),
            "diff_stat": run(root, ["git", "diff", "--stat"]),
            "remotes": run(root, ["git", "remote", "-v"]),
        }
        for name, root in roots.items()
    }
    raw_bytes = (json.dumps(raw, indent=2, sort_keys=True) + "\n").encode()
    args.private_raw.parent.mkdir(parents=True, exist_ok=True)
    args.private_raw.write_bytes(raw_bytes)
    artifacts = {}
    for path in args.artifact:
        resolved = path.resolve()
        try:
            label = resolved.relative_to(roots["torch"]).as_posix()
        except ValueError:
            label = resolved.name
        if label in artifacts:
            raise ValueError(f"duplicate artifact label: {label}")
        artifacts[label] = sha256(resolved)
    public = {
        "schema": "tora_q3_final_provenance_v1",
        "repositories": {
            name: repository_summary(root) for name, root in roots.items()
        },
        "expected_frozen_commits": {
            "torch": "c49d74bbf48d1004f7f3818174e7f40b6200b142",
            "xiangru": "27d29050a5f214b56f211ca9cb411e734ed80230",
            "diffreach": "dd628eb443b517d6415de93e7035b4baef73963e",
        },
        "new_public_commit_created": False,
        "push_performed": False,
        "git_blocker": "BLOCKED_UNKNOWN_HISTORICAL_ASSET_AUTHORIZATION",
        "artifact_hashes": artifacts,
        "private_raw_command_log_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    public["frozen_commit_match"] = all(
        public["repositories"][name]["head"] == expected
        for name, expected in public["expected_frozen_commits"].items()
    )
    args.public_summary.parent.mkdir(parents=True, exist_ok=True)
    args.public_summary.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "frozen_commit_match": public["frozen_commit_match"],
        "git_blocker": public["git_blocker"],
    }))
    return 0 if public["frozen_commit_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
