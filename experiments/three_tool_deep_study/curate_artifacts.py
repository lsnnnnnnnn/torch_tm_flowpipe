#!/usr/bin/env python3
"""Copy a passed full result into a small, commit-safe artifact bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected(source: Path) -> Iterable[Path]:
    for path in sorted(source.iterdir()):
        if path.is_file() and (
            path.suffix in {".csv", ".json", ".md"}
            or path.name == "RUN_COMPLETE"
        ):
            yield path
    for path in sorted((source / "plots").glob("*.png")):
        yield path
    for path in sorted((source / "common_segments").glob("*.json")):
        yield path
    for path in sorted((source / "flowstar_root_cause").iterdir()):
        if path.is_file() and path.suffix in {".csv", ".json"}:
            yield path


def curate(source: Path, destination: Path) -> dict[str, Any]:
    final = json.loads(
        (source / "final_acceptance.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (source / "artifact_quality_audit.json").read_text(encoding="utf-8")
    )
    if not final.get("passed", False):
        raise SystemExit("source final acceptance did not pass")
    if not final.get("require_ten_repetitions", False):
        raise SystemExit("source is not the authoritative ten-repetition run")
    if not quality.get("passed", False):
        raise SystemExit("source artifact quality audit did not pass")
    if not (source / "RUN_COMPLETE").is_file():
        raise SystemExit("source has no RUN_COMPLETE marker")
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"destination already exists and is nonempty: {destination}")

    destination.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for path in _selected(source):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(
            {
                "path": relative.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    curation = {
        "source": str(source),
        "source_run_id": source.name,
        "destination": str(destination),
        "curated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "selection": (
            "top-level CSV/JSON/Markdown, RUN_COMPLETE, PNG plots, all CIR "
            "JSON segments, and root-cause CSV/JSON; build products and logs "
            "remain in ignored scratch results"
        ),
        "file_count": len(copied),
        "total_bytes": sum(int(row["bytes"]) for row in copied),
        "files": copied,
    }
    (destination / "CURATION.json").write_text(
        json.dumps(curation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (destination / "SHA256SUMS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "bytes", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(copied)
    return curation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    destination = (
        Path(args.destination).resolve()
        if args.destination
        else HERE / "artifacts" / "authoritative" / source.name
    )
    result = curate(source, destination)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
