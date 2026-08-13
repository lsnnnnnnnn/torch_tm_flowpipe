#!/usr/bin/env python3
"""Build the compact, hash-verified source/carry audit evidence package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.source_carry_audit import derive_package_verification


DIRECTORIES = (
    "00_provenance",
    "01_baseline_reproduction",
    "02_flowstar_width_minima",
    "03_width_data_lineage",
    "04_high_precision_falsification",
    "05_flowstar_runtime_callgraph",
    "06_native_stage_traces",
    "07_same_prestate_replays",
    "08_source_semantics_map",
    "09_candidate_or_no_fix",
    "10_checkpoint_comparison",
    "11_tests",
    "12_final_clone",
)


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_gzip(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, fileobj=raw_output, mtime=0
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def qualification(relative: str) -> str:
    if relative.startswith("04_high_precision_falsification/"):
        return "high-precision replay or numerical falsification only"
    if relative.endswith("exact_semantics_micro_oracles.json"):
        return "formal primitive: exact rational fixture"
    if relative.startswith("06_native_stage_traces/"):
        return "empirical raw/native trace"
    if relative.startswith(("05_flowstar_runtime_callgraph/", "08_source_semantics_map/")):
        return "source inspection plus empirical feature trace"
    if relative.startswith(("11_tests/", "12_final_clone/")):
        return "software verification"
    return "derived empirical audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--high-precision-dir", type=Path, required=True)
    parser.add_argument("--flowstar-trace", type=Path, required=True)
    parser.add_argument("--flowstar-metadata", type=Path, required=True)
    parser.add_argument("--torch-run-dir", type=Path, required=True)
    parser.add_argument("--flowstar-runner-dir", type=Path, required=True)
    parser.add_argument("--torch-runner-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focused-test-log", type=Path)
    parser.add_argument("--full-test-log", type=Path)
    parser.add_argument("--compile-log", type=Path)
    parser.add_argument("--final-clone-dir", type=Path)
    return parser.parse_args()


def runner_record(path: Path) -> dict[str, Any]:
    summary = read_json(path / "summary.json")
    timing = read_json(path / "timing.json")
    return {
        "command": (path / "command.txt").read_text(encoding="utf-8").strip(),
        "cwd": str(ROOT),
        "environment": "00_provenance/provenance.json#environment",
        "stdout": (path / "stdout.log").read_text(encoding="utf-8"),
        "stderr": (path / "stderr.log").read_text(encoding="utf-8"),
        "exit_code": int(summary["exit_code"]),
        "runtime_seconds": float(timing["process_wall_seconds"]),
        "runner_status": summary["status"],
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    for directory in DIRECTORIES:
        (output / directory).mkdir(parents=True, exist_ok=True)

    audit = args.audit_dir.resolve()
    high_precision = args.high_precision_dir.resolve()
    torch_run = args.torch_run_dir.resolve()
    flowstar_runner = args.flowstar_runner_dir.resolve()
    torch_runner = args.torch_runner_dir.resolve()

    mappings = {
        "00_provenance/provenance.json": audit / "provenance.json",
        "00_provenance/flowstar_trace_metadata.csv": args.flowstar_metadata.resolve(),
        "00_provenance/torch_command.json": torch_run / "command.json",
        "00_provenance/torch_summary.json": torch_run / "summary.json",
        "01_baseline_reproduction/baseline_contract.json": audit / "baseline_contract.json",
        "01_baseline_reproduction/baseline_step_trace.csv": audit / "baseline_step_trace.csv",
        "01_baseline_reproduction/baseline_checkpoint_reproduction.csv": audit / "baseline_checkpoint_reproduction.csv",
        "01_baseline_reproduction/baseline_reproduction_verdict.json": audit / "baseline_reproduction_verdict.json",
        "02_flowstar_width_minima/flowstar_width_minima.csv": audit / "flowstar_width_minima.csv",
        "02_flowstar_width_minima/flowstar_width_minima_context.csv": audit / "flowstar_width_minima_context.csv",
        "03_width_data_lineage/flowstar_width_data_lineage.json": audit / "flowstar_width_data_lineage.json",
        "04_high_precision_falsification/sample_replay.csv": high_precision / "sample_replay.csv",
        "04_high_precision_falsification/high_precision_replay.csv": high_precision / "high_precision_replay.csv",
        "04_high_precision_falsification/variational_replay.csv": high_precision / "variational_replay.csv",
        "04_high_precision_falsification/summary.json": high_precision / "summary.json",
        "05_flowstar_runtime_callgraph/flowstar_runtime_features.json": audit / "flowstar_runtime_features.json",
        "05_flowstar_runtime_callgraph/flowstar_runtime_callgraph.json": audit / "source_semantics_map.json",
        "07_same_prestate_replays/same_prestate_lossless_gate.json": audit / "same_prestate_lossless_gate.json",
        "08_source_semantics_map/source_semantics_map.json": audit / "source_semantics_map.json",
        "08_source_semantics_map/exact_semantics_micro_oracles.json": audit / "exact_semantics_micro_oracles.json",
        "09_candidate_or_no_fix/candidate_or_no_fix.json": audit / "candidate_or_no_fix.json",
        "10_checkpoint_comparison/baseline_checkpoint_reproduction.csv": audit / "baseline_checkpoint_reproduction.csv",
        "10_checkpoint_comparison/width_growth_and_ratio_analysis.json": audit / "width_growth_and_ratio_analysis.json",
    }
    for relative, source in mappings.items():
        copy(source, output / relative)
    for label, runner in (("flowstar_runner", flowstar_runner), ("torch_runner", torch_runner)):
        for name in (
            "artifact_index.json",
            "command.txt",
            "config.json",
            "exit_code.txt",
            "started_at.txt",
            "finished_at.txt",
            "stdout.log",
            "stderr.log",
            "summary.json",
            "timing.json",
        ):
            copy(runner / name, output / "00_provenance" / label / name)

    copy_gzip(
        args.flowstar_trace.resolve(),
        output / "06_native_stage_traces/flowstar_trace.csv.gz",
    )
    for name in ("segments.csv", "attempts.csv", "remainder_ledger.jsonl"):
        copy_gzip(torch_run / name, output / f"06_native_stage_traces/torch_{name}.gz")
    for name in ("summary.json", "command.json", "checkpoints.csv"):
        copy(torch_run / name, output / f"06_native_stage_traces/torch_{name}")

    optional_logs = {
        "focused_tests.xml": args.focused_test_log,
        "full_pytest.xml": args.full_test_log,
        "compileall.log": args.compile_log,
    }
    for name, source in optional_logs.items():
        if source is not None:
            copy(source.resolve(), output / "11_tests" / name)

    final_clone_status: dict[str, Any]
    if args.final_clone_dir is None:
        final_clone_status = {
            "status": "FINAL_HEAD_FRESH_CLONE_NOT_VERIFIED",
            "note": "This package was created before the final remote-SHA fresh-clone gate.",
        }
    else:
        final_source = args.final_clone_dir.resolve()
        for source in sorted(final_source.iterdir()):
            if source.is_file():
                copy(source, output / "12_final_clone" / source.name)
        status_path = final_source / "verification.json"
        final_clone_status = dict(read_json(status_path))
    write_json(output / "12_final_clone/status.json", final_clone_status)

    audit_verification = read_json(audit / "verification.json")
    high_precision_summary = read_json(high_precision / "summary.json")
    verification = derive_package_verification(audit_verification, high_precision_summary)
    verification["required_directories_present"] = all(
        (output / directory).is_dir() for directory in DIRECTORIES
    )
    verification["final_clone"] = final_clone_status
    write_json(output / "verification.json", verification)

    content_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    provenance = read_json(audit / "provenance.json")
    contract = read_json(audit / "baseline_contract.json")
    manifest = {
        "schema": "flowstar_torch_source_carry_manifest_v1",
        "torch_source_sha": provenance["torch"]["source_sha"],
        "flowstar_source_sha": provenance["flowstar"]["source_sha"],
        "flowstar_dirty_diff_hash": provenance["flowstar"]["status"]["stdout"],
        "flowstar_binary_sha256": provenance["flowstar"]["probe"]["sha256"],
        "contract_sha256": contract["sha256"],
        "commands": {
            "flowstar_native": runner_record(flowstar_runner),
            "torch_native": runner_record(torch_runner),
            "packager": {
                "argv": sys.argv,
                "cwd": str(ROOT),
                "environment": "00_provenance/provenance.json#environment",
                "exit_code": 0,
                "stdout": "package verification is verification.json",
                "stderr": "",
                "runtime_seconds": "not timed by packager",
            },
        },
        "inputs": provenance["inputs"],
        "outputs": [
            {
                "path": str(path.relative_to(output)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "qualification": qualification(str(path.relative_to(output))),
            }
            for path in content_files
        ],
    }
    write_json(output / "manifest.json", manifest)

    checksum_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    checksums = "".join(
        f"{sha256(path)}  {path.relative_to(output)}\n" for path in checksum_files
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(json.dumps(verification, sort_keys=True))


if __name__ == "__main__":
    main()
