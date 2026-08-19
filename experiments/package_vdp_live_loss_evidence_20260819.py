#!/usr/bin/env python3
"""Package the VDP live-loss/C1 evidence with checksums and provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_SHA = "dbe03dcdfbf2f36b1d58013373d1d235ace1a48e"
REPORT = ROOT / "docs/VDP_LIVE_LOSS_ABLATION_B3_B4_CLOSURE_20260819.md"
PRODUCTION_PATHS = (
    "experiments/audit_vdp_h2_dense_picard_first_loss_20260817.py",
    "experiments/run_vdp_dense_backend.py",
    "experiments/tamper_test_vdp_live_loss_ablation_20260819.py",
    "experiments/verify_vdp_live_loss_ablation_20260819.py",
    "src/torch_tm_flowpipe/batched_dense_tm.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/polynomial_ode.py",
    "src/torch_tm_flowpipe/raw_remainder_trace.py",
    "src/torch_tm_flowpipe/step1_oracle.py",
    "src/torch_tm_flowpipe/terminal_checkpoint.py",
    "tests/test_batched_dense_runner_contract.py",
    "tests/test_vdp_h2_dense_picard.py",
)
RUN_FILES = (
    "command.json",
    "config_snapshot.yaml",
    "decision.json",
    "summary.json",
    "checkpoints.csv",
    "profile.csv",
    "segments.csv",
    "attempts.csv",
    "remainder_ledger.jsonl",
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_run(source: Path, destination: Path, *, checkpoint: bool = False) -> None:
    for name in RUN_FILES:
        _copy(source / name, destination / name)
    if checkpoint:
        checkpoint_source = source / "terminal_checkpoint"
        if checkpoint_source.is_dir():
            shutil.copytree(checkpoint_source, destination / "terminal_checkpoint")


def package(matrix_root: Path, output: Path) -> dict[str, Any]:
    matrix_root = matrix_root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    gates = output / "01_gates"
    for source in sorted((matrix_root / "gates").glob("*.json")):
        _copy(source, gates / source.name)
    scientific = output / "02_scientific_matrix"
    _copy(matrix_root / "matrix.json", scientific / "matrix.json")
    _copy(matrix_root / "checkpoint_widths.csv", scientific / "checkpoint_widths.csv")

    raw = output / "03_raw_runs"
    for lane in ("legacy", "h1", "h1_h2", "candidate"):
        _copy_run(matrix_root / "step1" / lane, raw / "step1" / lane, checkpoint=True)
    for lane in ("legacy", "candidate"):
        _copy_run(matrix_root / "fixed_T6p32" / lane, raw / "fixed_T6p32" / lane)
        _copy_run(
            matrix_root / "native_T10" / lane,
            raw / "native_T10" / lane,
            checkpoint=True,
        )
    for device in ("cpu", "cuda"):
        _copy_run(
            matrix_root / "consistency_T0p1" / device,
            raw / "consistency_T0p1" / device,
        )

    tests = output / "04_tests"
    for source in sorted((matrix_root / "test_logs").glob("*.xml")):
        _copy(source, tests / source.name)
    _copy(REPORT, output / "05_report" / REPORT.name)

    try:
        import torch

        environment = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - provenance must survive import failures.
        environment = {"python": platform.python_version(), "torch_import_error": repr(exc)}
    nvidia = subprocess.run(
        ["nvidia-smi"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    provenance_dir = output / "00_provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / "nvidia-smi.txt").write_text(
        nvidia.stdout + nvidia.stderr,
        encoding="utf-8",
    )
    source_hashes = {}
    for relative in PRODUCTION_PATHS:
        content = subprocess.run(
            ["git", "show", f"{SCIENTIFIC_SHA}:{relative}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        source_hashes[relative] = {"bytes": len(content), "sha256": _sha_bytes(content)}
    provenance = {
        "schema": "vdp_live_loss_c1_provenance_v1",
        "scientific_sha": SCIENTIFIC_SHA,
        "scientific_commit_time": subprocess.run(
            ["git", "show", "-s", "--format=%cI", SCIENTIFIC_SHA],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "scientific_runs_all_report_clean": True,
        "scientific_tracked_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "base_sha": "2cdb7a9509d5908baef79a02cfde18ea0682430c",
        "branch": "codex/vdp-live-loss-ablation-b3-b4-closure-20260819",
        "environment": environment,
        "source_hashes_at_scientific_sha": source_hashes,
        "packaging_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "packaging_worktree_status": subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout,
        "raw_range_trace_policy": (
            "range_trace.jsonl is omitted because it is mechanically redundant and 150+ MiB; "
            "commands/configs, every attempt, every segment, remainder ledgers, checkpoints, "
            "profiles, decisions and summaries are retained"
        ),
    }
    _write_json(provenance_dir / "provenance.json", provenance)

    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "schema": "vdp_live_loss_c1_evidence_manifest_v1",
        "scientific_sha": SCIENTIFIC_SHA,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            }
            for path in files
        ],
    }
    _write_json(output / "manifest.json", manifest)
    (output / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in manifest["files"]),
        encoding="utf-8",
    )
    result = {
        "output": str(output),
        "files": manifest["file_count"],
        "bytes": manifest["total_bytes"],
        "scientific_sha": SCIENTIFIC_SHA,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    package(args.matrix_root, args.output)
