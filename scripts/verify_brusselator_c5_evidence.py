#!/usr/bin/env python3
"""Fail-closed verifier for the Brusselator C5 live-range closure package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.brusselator_canonical_exchange import (  # noqa: E402
    SCHEMA,
    read_records,
    take_tmv,
)


DEFAULT_ARTIFACT = ROOT / "artifacts/runs/brusselator_live_range_c5_20260828"
REQUIRED = (
    "PROVENANCE.json",
    "C4_AUDIT.json",
    "C4_FULL_PREFIX_BASELINE.json",
    "CANONICAL_OBJECT_SCHEMA.json",
    "same_object_range_matrix.csv",
    "first_live_range_divergence.json",
    "terminal_shadow_replay.json",
    "c5_authorization.json",
    "production_matrix.csv",
    "native_horizon_matrix.csv",
    "runtime_matrix.csv",
    "RESULT.json",
    "SHA256SUMS",
    "raw/canonical_exchange/index.json",
    "raw/range_replay/range_replay.json",
)
FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
C4_SHA = "26323929d6f4fee0893478f6927ae76c5129bf47"
EVIDENCE_SHA = "89d0c17c3f6b3e99ac1d068f1573bda7e4f82cbe"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _verify_sums(artifact: Path, errors: list[str]) -> None:
    sums = artifact / "SHA256SUMS"
    lines = sums.read_text(encoding="utf-8").splitlines()
    if not lines:
        errors.append("SHA256SUMS is empty")
        return
    declared: set[str] = set()
    for line in lines:
        if "  " not in line:
            errors.append(f"malformed SHA256SUMS line: {line!r}")
            continue
        expected, relative = line.split("  ", 1)
        path = artifact / relative
        if relative == "SHA256SUMS" or relative in declared or not path.is_file():
            errors.append(f"invalid SHA256SUMS target: {relative}")
            continue
        declared.add(relative)
        if _sha256(path) != expected:
            errors.append(f"SHA256 mismatch: {relative}")
    actual = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if declared != actual:
        errors.append(
            f"SHA256SUMS coverage mismatch missing={sorted(actual-declared)} extra={sorted(declared-actual)}"
        )


def verify(artifact: Path) -> dict[str, Any]:
    artifact = artifact.resolve()
    errors: list[str] = []
    for relative in REQUIRED:
        if not (artifact / relative).is_file():
            errors.append(f"missing required artifact: {relative}")
    if errors:
        return {"passed": False, "errors": errors}
    _verify_sums(artifact, errors)
    provenance = _json(artifact / "PROVENANCE.json")
    if provenance.get("c4_scientific_commit") != C4_SHA:
        errors.append("C4 scientific provenance mismatch")
    if provenance.get("c4_evidence_commit") != EVIDENCE_SHA:
        errors.append("C4 evidence provenance mismatch")
    if provenance.get("flowstar_commit") != FLOWSTAR_SHA:
        errors.append("Flow* provenance mismatch")
    if provenance.get("remote_provenance_closed") is not True:
        errors.append("remote provenance is not closed")
    if provenance.get("c5_scientific_commit") is not None:
        errors.append("no-C5 result must not claim a C5 scientific commit")

    audit = _json(artifact / "C4_AUDIT.json")
    if audit.get("status") != "C4_REFINEMENT_CONTRACT_PASSED":
        errors.append("C4 audit status failed")
    items = audit.get("items", [])
    if [row.get("item") for row in items] != list(range(1, 13)) or not all(
        row.get("passed") is True for row in items
    ):
        errors.append("C4 audit does not close all twelve items")

    baseline = _json(artifact / "C4_FULL_PREFIX_BASELINE.json")
    if baseline.get("c4_reaches_T20") is not True:
        errors.append("fresh C4 baseline did not reach T20")
    if int(baseline.get("c4", {}).get("accepted_steps", -1)) != 1000:
        errors.append("C4 accepted-step count mismatch")
    if int(baseline.get("legacy", {}).get("accepted_steps", -1)) != 357:
        errors.append("legacy accepted-step count mismatch")
    if int(baseline.get("stock_flowstar", {}).get("accepted_steps", -1)) != 1000:
        errors.append("Flow* accepted-step count mismatch")
    for lane in ("c4", "legacy"):
        if baseline.get(lane, {}).get("certificate_checks_passed") is not True:
            errors.append(f"{lane} certificate checks failed")

    schema = _json(artifact / "CANONICAL_OBJECT_SCHEMA.json")
    if schema.get("schema_id") != SCHEMA:
        errors.append("canonical schema id mismatch")
    index_path = artifact / "raw/canonical_exchange/index.json"
    index = _json(index_path)
    if index.get("exchange_schema") != SCHEMA or not index.get("objects"):
        errors.append("canonical exchange index is malformed")
    for item in index.get("objects", []):
        path = artifact / "raw/canonical_exchange" / str(item.get("filename"))
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            errors.append(f"canonical object hash mismatch: {item.get('filename')}")
            continue
        try:
            records = read_records(path)
            if int(records["accepted_step"]) != int(item["accepted_step"]):
                raise ValueError("step mismatch")
            for prefix in (
                "tm.segment_tube",
                "tm.segment_endpoint_pre_cutoff",
                "tm.segment_endpoint_raw",
                "tm.boundary_outer_full",
                "tm.boundary_outer_nonlinear",
                "tm.right_map_input",
                "tm.boundary_torch_inserted",
            ):
                take_tmv(dict(records), prefix)
        except Exception as exc:
            errors.append(f"canonical object import rejected {path.name}: {exc}")

    matrix = _csv(artifact / "same_object_range_matrix.csv")
    expected_operators = {"A", "B", "C", "D", "E", "F", "G", "H", "X1", "X2"}
    if not matrix or {row["operator"] for row in matrix} != expected_operators:
        errors.append("same-object range matrix lacks required operators")
    if any(row["exact_local_outward_contained"] != "True" for row in matrix):
        errors.append("exact/local outward containment failed")
    replay = _json(artifact / "raw/range_replay/range_replay.json")
    if replay.get("all_exact_local_outward_checks") is not True:
        errors.append("range replay exact oracle aggregate failed")
    if replay.get("matrix_sha256") != _sha256(artifact / "same_object_range_matrix.csv"):
        errors.append("range replay matrix hash mismatch")

    divergence = _json(artifact / "first_live_range_divergence.json")
    ordered = divergence.get("ordered_operator_audit", [])
    if [row.get("search_index") for row in ordered] != [1, 2, 2, 3, 4]:
        errors.append("live cause search order is not cutoff, range, insertion, normalization")
    terminal = _json(artifact / "terminal_shadow_replay.json")
    if terminal.get("c4_completed_requested_horizon") is not True:
        errors.append("terminal shadow does not record C4 T20 completion")
    if terminal.get("terminal_rejection_exists") is not False:
        errors.append("T20 C4 lane must not claim a rejected terminal attempt")

    authorization = _json(artifact / "c5_authorization.json")
    if authorization.get("authorized") is not False:
        errors.append("C5 must remain unauthorized")
    if authorization.get("status") != "LIVE_RANGE_DOMINANT_CAUSE_NOT_IDENTIFIED__NO_C5":
        errors.append("no-C5 status mismatch")
    if authorization.get("gates", {}).get("terminal_shadow_margin_materially_improved") is not False:
        errors.append("unavailable terminal shadow gate must fail closed")

    production = _csv(artifact / "production_matrix.csv")
    horizon = _csv(artifact / "native_horizon_matrix.csv")
    runtime = _csv(artifact / "runtime_matrix.csv")
    if len(production) != 12 or len(horizon) != 3 or len(runtime) != 3:
        errors.append("production/native/runtime matrix dimensions mismatch")
    result = _json(artifact / "RESULT.json")
    if result.get("status") != authorization.get("status"):
        errors.append("RESULT/authorization status mismatch")
    if result.get("c4_reaches_T20") is not True or result.get("c5_implemented") is not False:
        errors.append("RESULT scientific outcome mismatch")
    return {
        "schema": "torch_tm_flowpipe.brusselator_c5_evidence_verification/1",
        "artifact": str(artifact),
        "passed": not errors,
        "errors": errors,
        "canonical_object_count": len(index.get("objects", [])),
        "matrix_row_count": len(matrix),
        "status": result.get("status"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = verify(parse_args(argv).artifact_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
