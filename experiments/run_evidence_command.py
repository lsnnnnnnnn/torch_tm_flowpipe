#!/usr/bin/env python3
"""Run one evidence command with the mandatory portable protocol envelope."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_index.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def run(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    command = [value.replace("{ARTIFACT_DIR}", str(artifacts)) for value in args.command]
    if not command:
        raise ValueError("an evidence command is required after --")
    config = json.loads(args.config_json)
    config.update(
        {
            "schema": "torch_tm_flowpipe_evidence_runner_config_v1",
            "runner_name": args.name,
            "source_commit": args.source_commit,
            "eligibility_status": args.eligibility_status,
            "expected_exit_codes": list(args.expected_exit_codes),
        }
    )
    _json(output / "config.json", config)
    (output / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    started_wall = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=args.cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    finished_wall = time.time()
    finished_iso = datetime.now(timezone.utc).isoformat()
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    (output / "exit_code.txt").write_text(
        f"{completed.returncode}\n", encoding="utf-8"
    )
    (output / "started_at.txt").write_text(started_iso + "\n", encoding="utf-8")
    (output / "finished_at.txt").write_text(finished_iso + "\n", encoding="utf-8")
    _json(
        output / "timing.json",
        {
            "schema": "torch_tm_flowpipe_evidence_timing_v1",
            "started_unix": started_wall,
            "finished_unix": finished_wall,
            "process_wall_seconds": elapsed,
            "timing_eligibility": args.timing_eligibility,
        },
    )
    artifact_summary = artifacts / "summary.json"
    summary = {
        "schema": "torch_tm_flowpipe_evidence_runner_summary_v1",
        "name": args.name,
        "status": (
            "pass"
            if completed.returncode == 0
            else "qualified_expected_nonzero"
            if completed.returncode in args.expected_exit_codes
            else "fail"
        ),
        "exit_code": completed.returncode,
        "source_commit": args.source_commit,
        "config_sha256": _sha256(output / "config.json"),
        "eligibility_status": args.eligibility_status,
        "artifact_summary": None,
    }
    if artifact_summary.is_file():
        json.loads(artifact_summary.read_text(encoding="utf-8"))
        summary["artifact_summary"] = {
            "path": "artifacts/summary.json",
            "sha256": _sha256(artifact_summary),
        }
    _json(output / "summary.json", summary)
    _json(
        output / "artifact_index.json",
        {
            "schema": "torch_tm_flowpipe_evidence_artifact_index_v1",
            "root": ".",
            "files": _artifact_rows(output),
        },
    )
    return 0 if completed.returncode in args.expected_exit_codes else int(completed.returncode or 1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--eligibility-status", required=True)
    parser.add_argument("--timing-eligibility", default="diagnostic_only")
    parser.add_argument(
        "--expected-exit-codes",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        default=(0,),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
