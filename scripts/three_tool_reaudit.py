#!/usr/bin/env python3
"""Create and finalize the correctness-first three-tool evidence directory."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe.protocol.reaudit import (
    _jax_record,
    collect_manifest,
    repository_record,
    validate_manifest,
    validate_primary_row,
    write_json,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _paths(arguments: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "torch_tm_flowpipe": REPO_ROOT,
        "flowstar": arguments.flowstar_root,
        "diffreach": arguments.diffreach_root,
        "xiangru": arguments.xiangru_root,
    }


def initialize(arguments: argparse.Namespace) -> int:
    output = REPO_ROOT / "outputs" / "three_tool_reaudit" / arguments.run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty run directory: {output}")
    for relative in ("raw", "one_step_trace", "vdp_t10", "gate_evidence", "logs"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    now = _utc_now()
    flowstar_binary = (
        arguments.flowstar_root
        / "benchmarks"
        / "continuous"
        / "vanderpol"
        / "vanderpol"
    )
    benchmarks = [
        REPO_ROOT / "benchmarks" / "cross_tool_gates.yaml",
        REPO_ROOT / "benchmarks" / "canonical.yaml",
        REPO_ROOT / "benchmarks" / "three_tool_matched_contract.yaml",
        arguments.flowstar_root
        / "benchmarks"
        / "continuous"
        / "vanderpol"
        / "vanderpol.cpp",
    ]
    manifest = collect_manifest(
        run_id=arguments.run_id,
        repo_roots=_paths(arguments),
        benchmark_files=benchmarks,
        flowstar_binary=flowstar_binary,
        environment=os.environ,
        started_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        started_local=now.astimezone().isoformat(),
    )
    manifest["runner"] = str(Path(__file__).relative_to(REPO_ROOT))
    manifest["runner_command"] = sys.argv
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("; ".join(errors))
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "summary.json",
        {
            "run_id": arguments.run_id,
            "status": "collecting_evidence",
            "headline_comparison_generated": False,
            "reason": "cross-tool gates are fail-closed",
            "rows": [],
        },
    )
    fields = [
        "backend",
        "lane",
        "completed_horizon",
        "requested_horizon",
        "validation_status",
        "soundness_level",
        "primary_eligible",
        "endpoint_semantics",
        "effective_support_sha256",
        "runtime_boundary",
        "backend_sha",
        "run_authority",
        "blocker",
    ]
    _write_csv(output / "summary.csv", [], fields)
    _write_csv(output / "eligibility.csv", [], fields)
    print(output)
    return 0


def finalize(arguments: argparse.Namespace) -> int:
    output = REPO_ROOT / "outputs" / "three_tool_reaudit" / arguments.run_id
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    now = _utc_now()
    manifest["finalized_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["software"]["jax"] = _jax_record()
    manifest["repositories_at_finalize"] = {
        name: repository_record(path) for name, path in _paths(arguments).items()
    }
    manifest["benchmark_files_at_finalize"] = [
        {
            "path": record["path"],
            "sha256": hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest(),
        }
        for record in manifest.get("benchmark_files", [])
        if Path(record["path"]).is_file()
    ]
    command_records = output / "logs" / "command_records.json"
    if command_records.is_file():
        manifest["commands"] = json.loads(
            command_records.read_text(encoding="utf-8")
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("; ".join(errors))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    rows = list(summary.get("rows", []))
    eligibility_rows: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row_errors = validate_primary_row(row)
        row["eligibility_errors"] = ";".join(row_errors)
        row["primary_eligible"] = not row_errors
        eligibility_rows.append(row)
    summary["rows"] = eligibility_rows
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = sorted({field for row in eligibility_rows for field in row})
    required = [
        "backend",
        "lane",
        "completed_horizon",
        "validation_status",
        "soundness_level",
        "primary_eligible",
    ]
    _write_csv(output / "summary.csv", eligibility_rows, sorted(set(fields + required)))
    _write_csv(output / "eligibility.csv", eligibility_rows, sorted(set(fields + required)))
    checksum_path = output / "checksums.sha256"
    lines: list[str] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "finalize"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("run_id")
        subparser.add_argument(
            "--flowstar-root",
            type=Path,
            default=REPO_ROOT.parent / "flowstar",
        )
        subparser.add_argument(
            "--diffreach-root",
            type=Path,
            default=REPO_ROOT.parent / "DiffReach",
        )
        subparser.add_argument(
            "--xiangru-root",
            type=Path,
            default=REPO_ROOT.parent / "CROWN-Reach_Development",
        )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    return initialize(arguments) if arguments.command == "init" else finalize(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
