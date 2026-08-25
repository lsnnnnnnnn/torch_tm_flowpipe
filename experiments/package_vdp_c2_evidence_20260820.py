#!/usr/bin/env python3
"""Package C2 scientific evidence with a complete SHA256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Sequence


RAW_RUN_EXCLUDED_FILES = (
    "horner_stage_trace.jsonl",
    "owner_ledger.jsonl",
    "range_trace.jsonl",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_raw_runs(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*RAW_RUN_EXCLUDED_FILES),
    )


def refresh_manifest(output: Path) -> dict[str, Any]:
    """Rebuild hashes after a fail-closed package audit updates derived artifacts."""

    output = output.resolve()
    matrix = json.loads(
        (output / "02_scientific_matrix/matrix.json").read_text(encoding="utf-8")
    )
    scientific_sha = str(matrix["scientific_sha"])
    excluded = {"manifest.json", "SHA256SUMS"}
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        rows.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    manifest = {
        "schema": "vdp_c2_evidence_manifest_v1",
        "scientific_sha": scientific_sha,
        "packaging_commit_separate_from_scientific_commit": True,
        "raw_run_excluded_files": list(RAW_RUN_EXCLUDED_FILES),
        "raw_run_exclusion_rationale": (
            "derived range/Horner/owner traces are mechanically redundant; "
            "attempts, segments, remainder/refinement ledgers, summaries, commands, "
            "configs, profiles, decisions, checkpoints, and run index are retained"
        ),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def package(
    output: Path,
    *,
    gate_dir: Path,
    matrix_dir: Path,
    matrix_summary: Path,
    baseline_verification: Path,
    tests_dir: Path,
    report: Path,
    provenance_dir: Path,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("package output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    sources = (
        (provenance_dir, output / "00_provenance"),
        (baseline_verification, output / "00_provenance/baseline_verification.json"),
        (gate_dir, output / "01_step1_causal_gate"),
        (matrix_summary, output / "02_scientific_matrix/matrix.json"),
        (tests_dir, output / "04_tests"),
        (report, output / "05_report/VDP_C2_POST_ACCEPT_REFINEMENT_20260820.md"),
    )
    for source, destination in sources:
        _copy(source.resolve(), destination)
    _copy_raw_runs(matrix_dir.resolve(), output / "03_raw_runs")
    return refresh_manifest(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--baseline-verification", type=Path, required=True)
    parser.add_argument("--tests-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = package(
        args.output,
        gate_dir=args.gate_dir,
        matrix_dir=args.matrix_dir,
        matrix_summary=args.matrix_summary,
        baseline_verification=args.baseline_verification,
        tests_dir=args.tests_dir,
        report=args.report,
        provenance_dir=args.provenance_dir,
    )
    print(json.dumps(result, sort_keys=True))
