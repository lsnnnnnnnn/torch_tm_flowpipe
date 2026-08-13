#!/usr/bin/env python3
"""Fail-closed verification of the committed source/carry evidence package."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.source_carry_audit import (
    accepted_flowstar_rows,
    accepted_torch_rows,
    checkpoint_reproduction,
    derive_width_minima,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def verify_checksums(root: Path) -> int:
    checksum_path = root / "SHA256SUMS"
    seen: set[str] = set()
    for line_number, raw in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        expected, separator, relative = raw.partition("  ")
        if not separator or len(expected) != 64 or relative in seen:
            raise ValueError(f"invalid checksum row {line_number}")
        seen.add(relative)
        target = root / relative
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f"checksum mismatch: {relative}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if seen != actual:
        raise ValueError(
            f"checksum inventory mismatch: missing={sorted(actual-seen)}, extra={sorted(seen-actual)}"
        )
    return len(seen)


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def junit_counts(path: Path) -> dict[str, int | float]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    output: dict[str, int | float] = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "time_seconds": 0.0,
    }
    for suite in suites:
        for name in ("tests", "failures", "errors", "skipped"):
            output[name] = int(output[name]) + int(suite.attrib.get(name, 0))
        output["time_seconds"] = float(output["time_seconds"]) + float(
            suite.attrib.get("time", 0.0)
        )
    return output


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    checksum_count = verify_checksums(root)
    json_files = sorted(root.rglob("*.json"))
    loaded = {str(path.relative_to(root)): strict_json(path) for path in json_files}
    verification = loaded["verification.json"]
    manifest = loaded["manifest.json"]

    flow_all = read_gzip_csv(root / "06_native_stage_traces/flowstar_trace.csv.gz")
    torch_all = read_gzip_csv(root / "06_native_stage_traces/torch_segments.csv.gz")
    flow = accepted_flowstar_rows(flow_all)
    torch = accepted_torch_rows(torch_all)
    if len(flow) != 1000 or len(torch) != 632:
        raise ValueError("native accepted-row contract mismatch")
    rejected = sum(row.get("status") == "rejected" for row in torch_all)
    if rejected != 1:
        raise ValueError("Torch rejected-row contract mismatch")

    minima, _ = derive_width_minima(flow)
    published_minima = read_csv(
        root / "02_flowstar_width_minima/flowstar_width_minima.csv"
    )
    for derived, published in zip(minima, published_minima, strict=True):
        if (
            str(derived["channel"]) != published["channel"]
            or str(derived["step"]) != published["step"]
            or str(derived["width"]) != published["width"]
        ):
            raise ValueError(f"minimum derivation mismatch: {derived['channel']}")

    checkpoints, verdict = checkpoint_reproduction(flow, torch)
    if verdict["status"] != "BASELINE_CONCLUSIONS_REPRODUCED":
        raise ValueError("checkpoint conclusion did not reproduce")
    published_checkpoints = read_csv(
        root / "01_baseline_reproduction/baseline_checkpoint_reproduction.csv"
    )
    if len(checkpoints) != len(published_checkpoints):
        raise ValueError("checkpoint row-count mismatch")
    for derived, published in zip(checkpoints, published_checkpoints, strict=True):
        if (
            str(derived["step"]) != published["step"]
            or str(derived["channel"]) != published["channel"]
            or str(derived["ratio"]) != published["ratio"]
        ):
            raise ValueError("checkpoint ratio derivation mismatch")

    expected_statuses = [
        "BASELINE_CONCLUSIONS_REPRODUCED",
        "FLOWSTAR_WIDTH_IS_POSITIVE_NEAR_ZERO",
        "SOURCE_LEVEL_DEPENDENCY_LOSS_LOCALIZED",
        "NO_FIX_AUTHORIZED",
    ]
    if verification.get("scientific_statuses") != expected_statuses:
        raise ValueError("package scientific status mismatch")
    if verification.get("scientific_outcome_uses_process_exit_code") is not False:
        raise ValueError("package did not exclude exit code from scientific outcome")
    if len(manifest.get("outputs", [])) + 1 != checksum_count:
        # The manifest itself is written after its output inventory.
        raise ValueError("manifest/checksum inventory count mismatch")

    focused = junit_counts(root / "11_tests/focused_tests.xml")
    full = junit_counts(root / "11_tests/full_pytest.xml")
    if focused["tests"] != 18 or focused["failures"] or focused["errors"]:
        raise ValueError("focused JUnit result mismatch")
    if full["tests"] != 689 or full["skipped"] != 2 or full["failures"] or full["errors"]:
        raise ValueError("full JUnit result mismatch")

    return {
        "schema": "flowstar_torch_source_carry_fresh_clone_verification_v1",
        "status": "PASS",
        "checksum_files": checksum_count,
        "json_files_loaded": len(json_files),
        "flowstar_accepted_steps": len(flow),
        "torch_accepted_steps": len(torch),
        "torch_rejected_candidates": rejected,
        "width_minima": [
            {"channel": row["channel"], "step": row["step"], "width": row["width"]}
            for row in minima
        ],
        "checkpoint_rows_rederived": len(checkpoints),
        "focused_junit": focused,
        "full_junit": full,
        "scientific_statuses": expected_statuses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.package)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
