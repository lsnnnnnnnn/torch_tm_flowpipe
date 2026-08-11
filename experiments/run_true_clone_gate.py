#!/usr/bin/env python3
"""Run an H1/H2/H3 validation gate in a real temporary origin clone."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    output: Path,
    name: str,
    records: list[dict[str, Any]],
    environment: Mapping[str, str] | None = None,
) -> None:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=None if environment is None else dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path = output / f"{name}.stdout.log"
    stderr_path = output / f"{name}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    records.append(
        {
            "name": name,
            "command": list(command),
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(f"true-clone command failed: {name}")


def _git_output(command: Sequence[str], cwd: Path) -> str:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_worktree.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_sha):
        raise ValueError("expected SHA must be a full lowercase commit ID")
    origin = _git_output(["git", "remote", "get-url", "origin"], source)
    records: list[dict[str, Any]] = []
    temporary_root = Path(tempfile.mkdtemp(prefix="torch_tm_flowpipe_true_clone_"))
    clone = temporary_root / "repository"
    install = temporary_root / "install"
    install.mkdir()
    try:
        _run(
            ["git", "clone", "--no-local", "--origin", "origin", origin, str(clone)],
            cwd=temporary_root,
            output=output,
            name="clone_origin",
            records=records,
        )
        _run(
            ["git", "checkout", "--detach", args.expected_sha],
            cwd=clone,
            output=output,
            name="checkout_exact_sha",
            records=records,
        )
        checked_out = _git_output(["git", "rev-parse", "HEAD"], clone)
        cloned_origin = _git_output(["git", "remote", "get-url", "origin"], clone)
        if checked_out != args.expected_sha or cloned_origin != origin:
            raise RuntimeError("clone identity does not match the requested origin/SHA")
        _run(
            [
                str(args.python.resolve()),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(install),
                ".",
            ],
            cwd=clone,
            output=output,
            name="install",
            records=records,
        )
        environment = dict(os.environ)
        previous_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(install)
            if not previous_pythonpath
            else str(install) + os.pathsep + previous_pythonpath
        )
        if args.gate == "tested_source":
            commands = [
                (
                    "full_pytest",
                    [str(args.python.resolve()), "-m", "pytest", "-q", "-rsxX"],
                ),
                (
                    "compileall",
                    [
                        str(args.python.resolve()),
                        "-m",
                        "compileall",
                        "-q",
                        "src",
                        "experiments",
                        "tests",
                    ],
                ),
                ("diff_check", ["git", "diff", "--check"]),
                (
                    "checkpoint_load",
                    [
                        str(args.python.resolve()),
                        "experiments/validate_tracked_checkpoints.py",
                        "--output-dir",
                        str(output / "checkpoint_validation"),
                    ],
                ),
            ]
        else:
            if args.package_path is None:
                raise ValueError("package_commit/delivery gate requires --package-path")
            if args.package_path.is_absolute():
                raise ValueError("package path must be clone-relative")
            package = (clone / args.package_path).resolve()
            if not package.is_relative_to(clone.resolve()):
                raise ValueError("package path escapes the true clone")
            commands = [
                (
                    "package_checksums",
                    [
                        str(args.python.resolve()),
                        "-c",
                        (
                            "from pathlib import Path;"
                            "from experiments.finalize_three_tool_evidence_package "
                            "import validate_checksum_coverage;"
                            "validate_checksum_coverage(Path(__import__('sys').argv[1]))"
                        ),
                        str(package),
                    ],
                ),
                (
                    "focused_integrity",
                    [
                        str(args.python.resolve()),
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_evidence_verification.py",
                        "tests/test_three_tool_package_finalizer.py",
                        "tests/test_canonical_status_consistency.py",
                    ],
                ),
                ("diff_check", ["git", "diff", "--check"]),
            ]
        for name, command in commands:
            _run(
                command,
                cwd=clone,
                output=output,
                name=name,
                records=records,
                environment=environment,
            )
        status = _git_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], clone
        )
        if status:
            raise RuntimeError("true clone is dirty after validation")
        summary = {
            "schema": "torch_tm_flowpipe_true_clone_gate_v1",
            "outcome": "TRUE_FRESH_CLONE_PASS",
            "gate": args.gate,
            "temporary_root_method": "tempfile.mkdtemp",
            "source_worktree": str(source),
            "clone_root": str(clone),
            "origin_clone": True,
            "origin": origin,
            "cloned_origin": cloned_origin,
            "expected_sha": args.expected_sha,
            "checked_out_sha": checked_out,
            "package_path": (
                None if args.package_path is None else args.package_path.as_posix()
            ),
            "commands": records,
            "clean_tree": True,
        }
        _write_json(output / "summary.json", summary)
        return summary
    finally:
        # Preserve no checkout or installation dependency after the evidence
        # logs and marker have been emitted.
        import shutil

        shutil.rmtree(temporary_root)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--gate",
        choices=("tested_source", "package_commit", "delivery"),
        required=True,
    )
    parser.add_argument("--package-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
