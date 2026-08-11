#!/usr/bin/env python3
"""Validate and finalize the three-tool evidence package without hardcoded passes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.evidence_verification import (
    classify_private_path_matches,
    derive_command_claim,
    validate_verification_document,
    verification_document,
)


REQUIRED_RUNNER_FILES = {
    "config.json",
    "summary.json",
    "stdout.log",
    "stderr.log",
    "command.txt",
    "exit_code.txt",
    "timing.json",
    "artifact_index.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _reject_nonfinite_json(path: Path) -> None:
    json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {value} in {path}")
        ),
    )


def _runner_directories(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.rglob("*")
        if path.is_dir()
        and (path / "config.json").is_file()
        and (path / "command.txt").is_file()
    )


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    runners = _runner_directories(run_root)
    if not runners:
        raise RuntimeError("evidence package contains no runner protocol directories")
    claims = []
    for runner in runners:
        missing = REQUIRED_RUNNER_FILES - {path.name for path in runner.iterdir()}
        if missing:
            raise RuntimeError(
                f"incomplete runner protocol {runner}: {sorted(missing)}"
            )
        relative = runner.relative_to(run_root).as_posix()
        runner_config = json.loads((runner / "config.json").read_text())
        expected_exit_codes = tuple(
            int(value) for value in runner_config.get("expected_exit_codes", [0])
        )

        def evaluate_exit(
            _stdout: str,
            _stderr: str,
            exit_code: int,
            *,
            expected: tuple[int, ...] = expected_exit_codes,
        ) -> tuple[str, Sequence[str]]:
            if exit_code not in expected:
                return "fail", (f"unexpected exit code {exit_code}",)
            if exit_code == 0:
                return "pass", ()
            return "qualified", (
                f"expected fail-closed/noncompletion exit code {exit_code}",
            )

        claims.append(
            derive_command_claim(
                relative.replace("/", "."),
                runner,
                scope=f"runner protocol {relative}",
                repository_root=run_root,
                evaluator=evaluate_exit,
            )
        )
    verification = verification_document(claims)
    _write_json(run_root / "verification.json", verification)
    validate_verification_document(verification, source_root=run_root)

    for path in run_root.rglob("*.json"):
        _reject_nonfinite_json(path)
    path_scan = classify_private_path_matches(
        [
            path
            for path in run_root.rglob("*")
            if path.is_file() and path.suffix in {".json", ".txt", ".log", ".csv", ".tsv"}
        ],
        scan_root=run_root,
        private_prefix="/srv/local/shengenli",
        provenance_only=[
            path.relative_to(run_root).as_posix()
            for path in run_root.rglob("*")
            if path.is_file()
            and (
                path.name
                in {
                    "command.txt",
                    "command.json",
                    "stdout.log",
                    "stderr.log",
                    "config.json",
                    "timing.json",
                    "verification.json",
                    "terminal_state.json",
                }
                or "03_native_flowstar/scalar_affine_gate/"
                in path.relative_to(run_root).as_posix()
                or path.relative_to(run_root).as_posix()
                == "00_environment/probe/artifacts/run/summary.json"
            )
        ],
    )
    if path_scan["status"] == "fail":
        raise RuntimeError("unclassified private path in evidence package")

    outcomes = json.loads(args.outcomes_json)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema": "three_tool_matched_divergence_fixed_support_package_v1",
        "run_id": run_root.name,
        "source_commit": source_commit,
        "package_root": ".",
        "runner_count": len(runners),
        "runners": [path.relative_to(run_root).as_posix() for path in runners],
        "outcomes": outcomes,
        "private_path_audit": path_scan,
        "verification_sha256": _sha(run_root / "verification.json"),
    }
    _write_json(run_root / "manifest.json", manifest)

    checksum_path = run_root / "SHA256SUMS"
    files = sorted(
        path
        for path in run_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(run_root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--outcomes-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(finalize(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
