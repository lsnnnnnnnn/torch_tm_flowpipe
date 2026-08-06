#!/usr/bin/env python3
"""Print one repository/environment final-check record as deterministic JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    git = {
        name: run(command, repo)
        for name, command in {
            "commit": ["git", "rev-parse", "HEAD"],
            "branch": ["git", "branch", "--show-current"],
            "status": ["git", "status", "--porcelain=v2", "--branch"],
            "diff_check": ["git", "diff", "--check"],
        }.items()
    }
    probe = run(
        [
            str(args.python),
            "-c",
            (
                "import json,platform,sys; "
                "print(json.dumps({'python':sys.version,'executable':sys.executable,"
                "'platform':platform.platform()}))"
            ),
        ],
        repo,
    )
    environment = (
        json.loads(probe.stdout) if probe.returncode == 0 else
        {"probe_error": probe.stderr, "exit_code": probe.returncode}
    )
    payload = {
        "schema": "q3_repository_final_check_v1",
        "repo": str(repo),
        "commit": git["commit"].stdout.strip(),
        "branch": git["branch"].stdout.strip() or "detached",
        "status_porcelain_v2": git["status"].stdout.splitlines(),
        "diff_check_exit_code": git["diff_check"].returncode,
        "diff_check_stdout": git["diff_check"].stdout,
        "diff_check_stderr": git["diff_check"].stderr,
        "environment": environment,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if git["diff_check"].returncode == 0 and probe.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
