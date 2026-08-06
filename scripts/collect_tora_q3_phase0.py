#!/usr/bin/env python3
"""Capture Phase 0 repository evidence without copying it into public artifacts.

The output directory is intentionally supplied by the caller.  For the TORA-Q3
audit it must live under the private verification-evidence root because raw Git
output can contain remote URLs, user paths, and unpublished branch names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Sequence


def run(argv: Sequence[str], cwd: Path) -> tuple[int, bytes, bytes]:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=os.environ.copy(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        return 127, b"", f"{error}\n".encode()
    return completed.returncode, completed.stdout, completed.stderr


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_record(
    output: Path,
    *,
    label: str,
    cwd: Path,
    argv: Sequence[str],
) -> dict[str, object]:
    code, stdout, stderr = run(argv, cwd)
    record_dir = output / label
    record_dir.mkdir(parents=True, exist_ok=False)
    (record_dir / "stdout.log").write_bytes(stdout)
    (record_dir / "stderr.log").write_bytes(stderr)
    metadata = {
        "schema": "tora_q3_raw_command_v1",
        "cwd": str(cwd.resolve()),
        "argv": list(argv),
        "command": shlex.join(argv),
        "exit_code": code,
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
    }
    (record_dir / "command.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def git_commands(*, fetch: bool) -> list[tuple[str, list[str]]]:
    commands = [
        ("status_short_branch", ["git", "status", "--short", "--branch"]),
        ("status_porcelain_v2", ["git", "status", "--porcelain=v2"]),
        ("head", ["git", "rev-parse", "HEAD"]),
        ("tree", ["git", "rev-parse", "HEAD^{tree}"]),
        ("log_15", ["git", "log", "-15", "--oneline", "--decorate"]),
        ("branches", ["git", "branch", "-a", "-vv"]),
        ("remotes", ["git", "remote", "-v"]),
    ]
    if fetch:
        commands.append(("fetch_all_tags", ["git", "fetch", "--all", "--tags"]))
    commands.extend(
        [
            ("ls_remote_origin", ["git", "ls-remote", "origin"]),
            ("diff_check", ["git", "diff", "--check"]),
            ("tracked_diff", ["git", "diff", "HEAD", "--binary"]),
        ]
    )
    return commands


def git_text(repo: Path, *args: str) -> str | None:
    code, stdout, _stderr = run(["git", *args], repo)
    return stdout.decode(errors="replace").strip() if code == 0 else None


def sanitized_state(repositories: dict[str, Path]) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "tora_q3_provenance_state_v1",
        "repositories": {},
    }
    states = result["repositories"]
    assert isinstance(states, dict)
    for label, repo in repositories.items():
        _code, status, _stderr = run(["git", "status", "--porcelain=v2"], repo)
        diff_code, diff, diff_stderr = run(["git", "diff", "HEAD", "--binary"], repo)
        branch = git_text(repo, "branch", "--show-current") or "detached"
        upstream = git_text(repo, "rev-parse", "--abbrev-ref", "@{upstream}")
        states[label] = {
            "commit": git_text(repo, "rev-parse", "HEAD"),
            "tree": git_text(repo, "rev-parse", "HEAD^{tree}"),
            "branch": branch,
            "upstream": upstream,
            "dirty": bool(status),
            "status_porcelain_v2_sha256": sha256_bytes(status),
            "tracked_diff_sha256": sha256_bytes(diff),
            "tracked_diff_exit_code": diff_code,
            "tracked_diff_stderr_sha256": sha256_bytes(diff_stderr),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-state", type=Path, required=True)
    parser.add_argument("--torch-root", type=Path, required=True)
    parser.add_argument("--torch-audit-root", type=Path, required=True)
    parser.add_argument("--xiangru-root", type=Path, required=True)
    parser.add_argument("--diffreach-root", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    required_repositories = {
        "torch_user_worktree": args.torch_root.resolve(),
        "xiangru": args.xiangru_root.resolve(),
        "diffreach": args.diffreach_root.resolve(),
    }
    audit_repositories = {
        **required_repositories,
        "torch_audit_worktree": args.torch_audit_root.resolve(),
    }
    manifest: list[dict[str, object]] = []
    for repo_label, repo in required_repositories.items():
        if not repo.is_dir():
            raise FileNotFoundError(repo)
        for command_label, argv in git_commands(fetch=repo_label != "diffreach"):
            manifest.append(
                write_record(
                    output,
                    label=f"{repo_label}/{command_label}",
                    cwd=repo,
                    argv=argv,
                )
            )
    # The clean detached worktree is additional evidence for the implementation
    # state; no network command is needed because it shares the Torch object DB.
    for command_label, argv in git_commands(fetch=False):
        manifest.append(
            write_record(
                output,
                label=f"torch_audit_worktree/{command_label}",
                cwd=audit_repositories["torch_audit_worktree"],
                argv=argv,
            )
        )

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    public_state = args.public_state.resolve()
    public_state.parent.mkdir(parents=True, exist_ok=True)
    public_state.write_text(
        json.dumps(sanitized_state(audit_repositories), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
