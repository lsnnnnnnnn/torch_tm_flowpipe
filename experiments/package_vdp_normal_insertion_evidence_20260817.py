#!/usr/bin/env python3
"""Build a deterministic tracked package for the 2026-08-17 VDP audit."""
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
SOURCE = ROOT / "outputs/vdp_normal_insertion_root_cause_20260817"
BASE_SHA = "e47ce68c61e73fc38f17fab3037d6cfe1877f3fd"
COMPRESS_NAMES = {
    "attempts.csv",
    "segments.csv",
    "remainder_ledger.jsonl",
    "range_trace.jsonl",
}
MATRIX_NAMES = {
    "attempts.csv",
    "checkpoints.csv",
    "command.json",
    "config_snapshot.yaml",
    "decision.json",
    "matrix.json",
    "matrix_stderr.txt",
    "matrix_stdout.txt",
    "profile.csv",
    "range_trace.jsonl",
    "remainder_ledger.jsonl",
    "requests.csv",
    "segments.csv",
    "summary.json",
    "terminal_reference.json",
    "terminal_state.json",
    "terminal_state_manifest.json",
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
    compressed_sources: list[dict[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.name in COMPRESS_NAMES and source.stat().st_size >= 64_000:
        stored = destination.with_name(destination.name + ".gz")
        stored.write_bytes(gzip.compress(source.read_bytes(), compresslevel=9, mtime=0))
        compressed_sources.append(
            {
                "source": source.relative_to(ROOT).as_posix(),
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


def package(run_id: str) -> dict[str, Any]:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    package_root = ROOT / "evidence/vdp_normal_insertion_root_cause" / run_id
    if package_root.exists() and any(package_root.iterdir()):
        raise FileExistsError(f"refusing non-empty package: {package_root}")
    package_root.mkdir(parents=True, exist_ok=True)
    compressed_sources: list[dict[str, Any]] = []

    gate_source = SOURCE / "gate_a_v2"
    for path in sorted(gate_source.rglob("*")):
        if path.is_file():
            _copy(
                path,
                package_root / "01_gate_a" / path.relative_to(gate_source),
                package_root,
                compressed_sources,
            )

    matrix_source = SOURCE / "scientific_matrix"
    for path in sorted(matrix_source.rglob("*")):
        if path.is_file() and path.name in MATRIX_NAMES:
            _copy(
                path,
                package_root / "02_scientific_matrix" / path.relative_to(matrix_source),
                package_root,
                compressed_sources,
            )

    tests_source = SOURCE / "tests"
    for path in sorted(tests_source.glob("*.xml")):
        _copy(path, package_root / "03_tests" / path.name, package_root, compressed_sources)

    source_paths = [
        ROOT / "src/torch_tm_flowpipe/flowpipe.py",
        ROOT / "src/torch_tm_flowpipe/__init__.py",
        ROOT / "experiments/run_vdp_dense_backend.py",
        ROOT / "experiments/audit_vdp_normal_insertion_root_cause_20260817.py",
        ROOT / "experiments/run_vdp_normal_insertion_matrix_20260817.py",
        ROOT / "experiments/package_vdp_normal_insertion_evidence_20260817.py",
        ROOT / "experiments/verify_vdp_normal_insertion_evidence_20260817.py",
        ROOT / "tests/test_dependency_preserving_insertion.py",
        ROOT / "docs/VDP_NORMAL_INSERTION_ROOT_CAUSE_FIX_20260817.md",
    ]
    provenance = {
        "run_id": run_id,
        "base_sha": BASE_SHA,
        "branch": _command("git", "branch", "--show-current"),
        "code_sha": _command("git", "rev-parse", "HEAD"),
        "worktree_status": _command("git", "status", "--short"),
        "tracked_diff_sha256": _sha_bytes(
            subprocess.run(
                ["git", "diff", "HEAD", "--binary"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
        ),
        "contract": {
            "ode": ["y", "y-x-x^2*y"],
            "initial_box_exact_decimal": [["1.1", "1.4"], ["2.35", "2.45"]],
            "order": 4,
            "cutoff": "1e-10",
            "fixed_h": "0.01",
            "target_remainder_radius": "1e-4",
            "native_legacy_floor": 6.397083942944808,
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
    gate = json.loads((package_root / "01_gate_a/summary.json").read_text(encoding="utf-8"))
    matrix = json.loads(
        (package_root / "02_scientific_matrix/matrix.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema": "vdp_normal_insertion_root_cause_evidence_v1",
        "run_id": run_id,
        "base_sha": BASE_SHA,
        "code_sha": provenance["code_sha"],
        "scientific_decision": matrix["decision"],
        "H1_gate_a_mechanism_pass": gate["H1_gate_a_mechanism_pass"],
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
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    package(args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
