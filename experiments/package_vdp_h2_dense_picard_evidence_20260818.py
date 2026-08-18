#!/usr/bin/env python3
"""Build the tracked raw-evidence package for the clean-SHA VDP H2 audit."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_SHA = "666c51ecc5575f203518d21f34b5c9948741fb17"
BASE_SHA = "43be6d34461e809c291a2d57e120012755d29d51"
COMPRESS_NAMES = {
    "attempts.csv",
    "segments.csv",
    "range_trace.jsonl",
    "remainder_ledger.jsonl",
}
RUN_FILE_NAMES = {
    "attempts.csv",
    "checkpoints.csv",
    "command.json",
    "config_snapshot.yaml",
    "decision.json",
    "profile.csv",
    "range_trace.jsonl",
    "remainder_ledger.jsonl",
    "segments.csv",
    "summary.json",
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _command(*args: str) -> str:
    result = subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip()


def _copy(
    source: Path,
    destination: Path,
    package_root: Path,
    source_root: Path,
    compressed_sources: list[dict[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.name in COMPRESS_NAMES and source.stat().st_size >= 64_000:
        stored = destination.with_name(destination.name + ".gz")
        stored.write_bytes(gzip.compress(source.read_bytes(), compresslevel=9, mtime=0))
        compressed_sources.append(
            {
                "source": source.relative_to(source_root).as_posix(),
                "stored": stored.relative_to(package_root).as_posix(),
                "source_bytes": source.stat().st_size,
                "source_sha256": _sha(source),
                "stored_bytes": stored.stat().st_size,
                "stored_sha256": _sha(stored),
                "compression": "gzip-9-mtime-0",
            }
        )
    else:
        shutil.copy2(source, destination)


def package(
    source_root: Path,
    run_id: str,
    *,
    full_xml: Path,
    targeted_xml: Path,
    h1_full_xml: Path,
    h1_targeted_xml: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not (source_root / "matrix/matrix.json").is_file():
        raise FileNotFoundError(source_root / "matrix/matrix.json")
    package_root = ROOT / "evidence/vdp_h2_dense_picard_first_loss" / run_id
    if package_root.exists() and any(package_root.iterdir()):
        raise FileExistsError(f"refusing non-empty package: {package_root}")
    package_root.mkdir(parents=True, exist_ok=True)
    compressed_sources: list[dict[str, Any]] = []

    for path in sorted((source_root / "gates").glob("*.json")):
        _copy(
            path,
            package_root / "01_gates" / path.name,
            package_root,
            source_root,
            compressed_sources,
        )
    for path in sorted((source_root / "matrix").glob("*")):
        if path.is_file():
            _copy(
                path,
                package_root / "02_scientific_matrix" / path.name,
                package_root,
                source_root,
                compressed_sources,
            )

    run_roots = ("step1", "fixed_T6p32", "native_T10", "cpu_T0p1", "v100_T0p1")
    for run_root in run_roots:
        base = source_root / run_root
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.name in RUN_FILE_NAMES:
                _copy(
                    path,
                    package_root / "03_raw_runs" / run_root / path.relative_to(base),
                    package_root,
                    source_root,
                    compressed_sources,
                )

    tests = {
        "h2_clean_scientific_full.xml": full_xml,
        "h2_clean_scientific_targeted.xml": targeted_xml,
        "h1_clean_base_full.xml": h1_full_xml,
        "h1_clean_base_targeted.xml": h1_targeted_xml,
    }
    (package_root / "04_tests").mkdir(parents=True, exist_ok=True)
    for name, source in tests.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, package_root / "04_tests" / name)

    source_paths = [
        ROOT / "src/torch_tm_flowpipe/batched_dense_tm.py",
        ROOT / "src/torch_tm_flowpipe/flowpipe.py",
        ROOT / "src/torch_tm_flowpipe/polynomial_ode.py",
        ROOT / "src/torch_tm_flowpipe/raw_remainder_trace.py",
        ROOT / "src/torch_tm_flowpipe/step1_oracle.py",
        ROOT / "src/torch_tm_flowpipe/terminal_checkpoint.py",
        ROOT / "experiments/audit_vdp_h2_dense_picard_first_loss_20260817.py",
        ROOT / "experiments/run_vdp_dense_backend.py",
        ROOT / "experiments/summarize_vdp_h2_dense_picard_matrix_20260818.py",
        ROOT / "experiments/package_vdp_h2_dense_picard_evidence_20260818.py",
        ROOT / "experiments/verify_vdp_h2_dense_picard_evidence_20260818.py",
        ROOT / "tests/test_vdp_h2_dense_picard.py",
        ROOT / "docs/VDP_H2_DENSE_PICARD_FIRST_LOSS_20260818.md",
    ]
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing package source files: {missing_sources}")
    matrix = json.loads((source_root / "matrix/matrix.json").read_text(encoding="utf-8"))
    provenance = {
        "run_id": run_id,
        "base_sha": BASE_SHA,
        "scientific_sha": SCIENTIFIC_SHA,
        "scientific_matrix_sha": matrix["scientific_sha"],
        "packaging_head": _command("git", "rev-parse", "HEAD"),
        "packaging_worktree_status": _command("git", "status", "--short"),
        "contract": {
            "ode": ["y", "y-x-x^2*y"],
            "initial_box_exact_decimal": [["1.1", "1.4"], ["2.35", "2.45"]],
            "order": 4,
            "fixed_h": "0.01",
            "cutoff": "1e-10",
            "target_remainder_radius": "1e-4",
            "validation_eps": "1e-12",
            "h2_validation_mode": "flowstar_raw_remainder_compat_factorized_joint",
            "native_h1_floor": 6.441433080631058,
        },
        "phase0_h1_clean_replay": {
            "detached_sha": BASE_SHA,
            "targeted": "11 passed",
            "full": "768 passed, 2 skipped",
            "package_verifier": "verified: 175 files, 25209699 bytes, six Gate-A checkpoints",
            "porcelain_before_after": "clean",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count(),
        },
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in source_paths
        },
    }
    _write_json(package_root / "00_provenance/provenance.json", provenance)
    nvidia_smi = "\n".join(line.rstrip() for line in _command("nvidia-smi").splitlines())
    _write_text(package_root / "00_provenance/nvidia-smi.txt", nvidia_smi + "\n")

    files = {
        path.relative_to(package_root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(package_root.rglob("*"))
        if path.is_file()
    }
    gate = json.loads((package_root / "01_gates/summary.json").read_text(encoding="utf-8"))
    manifest = {
        "schema": "vdp_h2_dense_picard_first_loss_evidence_v1",
        "run_id": run_id,
        "base_sha": BASE_SHA,
        "scientific_sha": SCIENTIFIC_SHA,
        "decision": matrix["decision"],
        "gate_a_pass": gate["gate_a_pass"],
        "gate_b_pass": gate["gate_b_pass"],
        "production_gates": matrix["gates"],
        "compressed_sources": compressed_sources,
        "files": files,
    }
    _write_json(package_root / "manifest.json", manifest)
    checksum_paths = [
        path
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    _write_text(
        package_root / "SHA256SUMS",
        "".join(
            f"{_sha(path)}  {path.relative_to(package_root).as_posix()}\n"
            for path in checksum_paths
        ),
    )
    result = {
        "package": package_root.relative_to(ROOT).as_posix(),
        "files": sum(path.is_file() for path in package_root.rglob("*")),
        "bytes": sum(path.stat().st_size for path in package_root.rglob("*") if path.is_file()),
    }
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full-xml", type=Path, required=True)
    parser.add_argument("--targeted-xml", type=Path, required=True)
    parser.add_argument("--h1-full-xml", type=Path, required=True)
    parser.add_argument("--h1-targeted-xml", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    package(
        args.source_root,
        args.run_id,
        full_xml=args.full_xml,
        targeted_xml=args.targeted_xml,
        h1_full_xml=args.h1_full_xml,
        h1_targeted_xml=args.h1_targeted_xml,
    )
