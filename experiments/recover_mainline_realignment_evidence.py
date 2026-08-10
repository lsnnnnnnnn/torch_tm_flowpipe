#!/usr/bin/env python3
"""Recover the canonical 2026-08-10 evidence tree without inventing results.

The source worktree is treated as immutable.  Every source artifact is hashed
before copying, existing destination files must already match byte-for-byte,
and large text artifacts are stored as deterministic gzip streams.  The
resulting recovery inventory retains the original byte count and digest so a
fresh clone can distinguish recovered evidence from regenerated derivatives.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_RELATIVE = Path("outputs/mainline_realignment_20260810/20260810T025910Z")
REQUIRED_RECOVERED_DIRECTORIES = tuple(f"{index:02d}_" for index in range(3, 9))
LARGE_TEXT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".log", ".txt"})
DEFAULT_COMPRESSION_THRESHOLD = 5 * 1024 * 1024
REGENERATED_ROOT_FILES = frozenset(
    {
        "SHA256SUMS",
        "batch_scaling.csv",
        "claim_registry.csv",
        "failure_attribution.json",
        "full_horizon.csv",
        "manifest.json",
        "matched_contract.json",
        "native_baselines.json",
        "operator_equivalence.json",
        "short_horizon.csv",
        "soundness_matrix.csv",
        "timing.csv",
    }
)
REGENERATED_NESTED_FILES = frozenset(
    {
        "02_torch_diffreach_equivalence/evidence_index.json",
        "03_flowstar_causal_divergence/observer_equivalence.json",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(worktree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_source_sums(source_repo: Path, source_run: Path) -> None:
    sums = source_run / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError(f"source checksum manifest missing: {sums}")
    errors: list[str] = []
    for line_number, line in enumerate(sums.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            expected, repo_relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"line {line_number}: malformed")
            continue
        target = source_repo / repo_relative
        if not target.is_file():
            errors.append(f"{repo_relative}: missing")
        elif _sha256(target) != expected:
            errors.append(f"{repo_relative}: digest mismatch")
    if errors:
        raise ValueError(f"source evidence checksum failure: {errors[:10]}")


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != expected_sha256:
            raise ValueError(f"existing destination differs from source: {destination}")
        return
    shutil.copy2(source, destination)
    if _sha256(destination) != expected_sha256:
        raise ValueError(f"copy verification failed: {destination}")


def _gzip_deterministic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with source.open("rb") as source_handle, temporary.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as output:
            shutil.copyfileobj(source_handle, output, length=1024 * 1024)
    temporary.replace(destination)


def recover(
    source_repo: Path,
    destination_repo: Path,
    *,
    compression_threshold: int = DEFAULT_COMPRESSION_THRESHOLD,
) -> dict[str, Any]:
    source_repo = source_repo.resolve()
    destination_repo = destination_repo.resolve()
    source_run = source_repo / RUN_RELATIVE
    destination_run = destination_repo / RUN_RELATIVE
    if not source_run.is_dir():
        raise ValueError(f"source run is absent: {source_run}")

    top_level = {path.name for path in source_run.iterdir() if path.is_dir()}
    missing_groups = [
        prefix for prefix in REQUIRED_RECOVERED_DIRECTORIES
        if not any(name.startswith(prefix) for name in top_level)
    ]
    if missing_groups:
        raise ValueError(f"source run lacks required evidence groups: {missing_groups}")
    _verify_source_sums(source_repo, source_run)

    source_files = sorted(path for path in source_run.rglob("*") if path.is_file())
    inventory: list[dict[str, Any]] = []
    for source in source_files:
        relative = source.relative_to(source_run)
        source_size = source.stat().st_size
        source_sha256 = _sha256(source)
        destination = destination_run / relative
        regenerated = (
            relative.as_posix() in REGENERATED_ROOT_FILES
            or relative.as_posix() in REGENERATED_NESTED_FILES
            or relative.parts[0] == "figures"
        )
        if not regenerated or not destination.exists():
            _copy_verified(source, destination, source_sha256)

        stored = destination
        compressed = (
            not regenerated
            and source_size >= compression_threshold
            and source.suffix.lower() in LARGE_TEXT_SUFFIXES
        )
        if compressed:
            stored = destination.with_name(destination.name + ".gz")
            if stored.exists():
                stored.unlink()
            _gzip_deterministic(destination, stored)
            destination.unlink()
        inventory.append(
            {
                "source_relative_path": relative.as_posix(),
                "source_size": source_size,
                "source_sha256": source_sha256,
                "stored_relative_path": stored.relative_to(destination_run).as_posix(),
                "stored_size": stored.stat().st_size,
                "stored_sha256": _sha256(stored),
                "storage": (
                    "derived-regenerated"
                    if regenerated
                    else "gzip-mtime-zero" if compressed else "identity"
                ),
            }
        )

    source_label = f"{source_repo.name}/{RUN_RELATIVE.as_posix()}"
    result: dict[str, Any] = {
        "schema": "mainline_realignment_evidence_recovery_v1",
        "source_label": source_label,
        "source_commit": _git(source_repo, "rev-parse", "HEAD"),
        "source_dirty_status": _git(source_repo, "status", "--short"),
        "source_checksum_manifest_sha256": _sha256(source_run / "SHA256SUMS"),
        "destination_run": RUN_RELATIVE.as_posix(),
        "compression_threshold_bytes": compression_threshold,
        "source_file_count": len(inventory),
        "source_total_bytes": sum(row["source_size"] for row in inventory),
        "stored_total_bytes": sum(row["stored_size"] for row in inventory),
        "files": inventory,
    }
    provenance = destination_run / "00_provenance/evidence_recovery.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("--destination-repo", type=Path, default=ROOT)
    parser.add_argument(
        "--compression-threshold",
        type=int,
        default=DEFAULT_COMPRESSION_THRESHOLD,
    )
    args = parser.parse_args()
    result = recover(
        args.source_repo,
        args.destination_repo,
        compression_threshold=args.compression_threshold,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "source_commit",
                    "source_file_count",
                    "source_total_bytes",
                    "stored_total_bytes",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
