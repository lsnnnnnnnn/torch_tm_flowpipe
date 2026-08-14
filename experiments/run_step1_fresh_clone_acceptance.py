#!/usr/bin/env python3
"""Run detached scientific-SHA acceptance in a newly cloned repository."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


FOCUSED_TESTS = (
    "tests/test_step1_oracle.py",
    "tests/test_step1_stage_oracle_audit.py",
    "tests/test_batched_dense_picard.py",
    "tests/test_batched_dense_remainder_validation.py",
)


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _command(
    records: list[dict[str, Any]],
    output: Path,
    name: str,
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (output / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    record = {
        "name": name,
        "command": shlex.join(command),
        "cwd": str(cwd) if cwd is not None else None,
        "exit_code": completed.returncode,
    }
    records.append(record)
    if completed.returncode != 0:
        _write(output / "commands.json", records)
        raise RuntimeError(
            f"{name} failed with exit {completed.returncode}: {completed.stderr[-2000:]}"
        )
    return completed


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="step1-scientific-fresh-clone-") as temporary:
        clone = Path(temporary) / "repo"
        _command(records, output, "clone", ["git", "clone", args.origin, str(clone)])
        _command(
            records,
            output,
            "checkout",
            ["git", "checkout", "--detach", args.scientific_sha],
            cwd=clone,
        )
        head = _command(
            records, output, "head", ["git", "rev-parse", "HEAD"], cwd=clone
        ).stdout.strip()
        if head != args.scientific_sha:
            raise ValueError(f"detached HEAD mismatch: {head}")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = "src"
        _command(
            records,
            output,
            "compileall",
            [sys.executable, "-m", "compileall", "-q", "src", "experiments", "tests"],
            cwd=clone,
            env=environment,
        )
        _command(
            records,
            output,
            "focused_tests",
            [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS],
            cwd=clone,
            env=environment,
        )
        _command(
            records,
            output,
            "full_pytest",
            [sys.executable, "-m", "pytest", "-q"],
            cwd=clone,
            env=environment,
        )
        package = (
            clone
            / "outputs/flowstar_torch_step1_stage_oracle_sound_carry_20260813"
            / args.package_timestamp
        )
        _command(
            records,
            output,
            "package_verifier",
            [
                sys.executable,
                "experiments/verify_step1_stage_oracle_package.py",
                "--package",
                str(package),
                "--repo",
                str(clone),
            ],
            cwd=clone,
            env=environment,
        )
        porcelain = _command(
            records,
            output,
            "porcelain",
            ["git", "status", "--porcelain=v1"],
            cwd=clone,
        ).stdout
        if porcelain:
            raise ValueError(f"fresh clone is dirty after acceptance: {porcelain!r}")
        trees = {
            name: _command(
                records,
                output,
                f"tree_{name}",
                ["git", "rev-parse", f"{args.scientific_sha}:{name}"],
                cwd=clone,
            ).stdout.strip()
            for name in ("src", "experiments", "tests")
        }
    result = {
        "schema": "step1_scientific_fresh_clone_acceptance_v1",
        "status": "pass",
        "origin": args.origin,
        "scientific_sha": args.scientific_sha,
        "detached_head": head,
        "compileall": "pass",
        "focused_tests": "pass",
        "full_pytest": "pass",
        "package_verifier": "pass",
        "git_status_porcelain_empty": True,
        "tree_hashes": trees,
        "commands": records,
    }
    _write(output / "result.json", result)
    _write(
        output / "summary.json",
        {
            "schema": "step1_scientific_fresh_clone_summary_v1",
            "status": "pass",
            "scientific_sha": args.scientific_sha,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--scientific-sha", required=True)
    parser.add_argument("--package-timestamp", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
