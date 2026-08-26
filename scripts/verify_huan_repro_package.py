#!/usr/bin/env python3
"""Verify the committed Huan plant-only audit package without running science."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ENGINE_HEAD = "d5f0b68fcd36ba5f582733624f074728fe9720d8"
PRIMARY = "HUAN_SOURCE_BUILDS__PROOF_MAPPING_INCOMPLETE"
ALLOWED_MAP_STATUSES = {
    "MAPPED_AND_TESTED",
    "MAPPED_NOT_TESTED",
    "PARTIALLY_MAPPED",
    "SOURCE_MISSING",
    "ASSUMPTION_ONLY",
    "CONTRADICTED",
}
SCIENTIFIC_TABLES = (
    "step1_common_input.csv",
    "fixed_horizon_matrix.csv",
    "native_terminal.json",
    "batch_throughput.csv",
)
REQUIRED_OUTPUTS = (
    "artifact_inventory.json",
    "source_manifest.json",
    "environment.txt",
    "build.log",
    "upstream_tests.log",
    "proof_to_code_map.csv",
    "phase_d_gate.json",
    "raw_logs/proof_kernel_cpu.json",
    "raw_logs/proof_kernel_cuda.json",
    "raw_logs/phase_d3_dense_sparse.log",
    "raw_logs/phase_d4_chunk_lane.log",
    "raw_logs/phase_d5_refinement.log",
    "raw_logs/phase_d6_strict_parity.log",
    "raw_logs/refinement_boundary_cpu.json",
    "raw_logs/refinement_boundary_cuda.json",
    "raw_logs/chunk_boundary_cpu.json",
    "raw_logs/chunk_boundary_cuda.json",
    "raw_logs/upstream_path_mapped_replay.log",
    "raw_logs/torch_full_tests.log",
    "SHA256SUMS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(output_root: Path) -> list[str]:
    errors: list[str] = []
    checksum_path = output_root / "SHA256SUMS"
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            errors.append(f"malformed checksum line: {line}")
            continue
        expected[relative] = digest
    actual_paths = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    for relative in sorted(actual_paths | set(expected)):
        path = output_root / relative
        if relative not in actual_paths:
            errors.append(f"checksum target missing: {relative}")
        elif relative not in expected:
            errors.append(f"uncovered file: {relative}")
        elif _sha256(path) != expected[relative]:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def capture_header(path: Path) -> tuple[dict[str, Any], str]:
    header, separator, body = path.read_text(encoding="utf-8").partition(
        "\n--- combined stdout/stderr ---\n"
    )
    if not separator:
        raise ValueError(f"capture delimiter missing: {path}")
    return json.loads(header), body


def verify(repo_root: Path, output_root: Path) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_OUTPUTS:
        if not (output_root / relative).is_file():
            errors.append(f"required output missing: {relative}")
    if errors:
        return errors
    errors.extend(verify_checksums(output_root))

    inventory = json.loads((output_root / "artifact_inventory.json").read_text())
    manifest = json.loads((output_root / "source_manifest.json").read_text())
    gate = json.loads((output_root / "phase_d_gate.json").read_text())
    if not inventory["source_closure"]["current_clean_engine_source_available"]:
        errors.append("current clean engine source gate is not open")
    if inventory["source_closure"]["historical_dirty_experiment_state_available"]:
        errors.append("historical dirty state is incorrectly marked available")
    if manifest["git"]["head"] != ENGINE_HEAD or not manifest["git"]["clean"]:
        errors.append("engine source manifest is not the expected clean head")
    if manifest["historical_dirty_state_exact"]:
        errors.append("historical dirty result state is incorrectly exact")
    if gate["primary_decision"] != PRIMARY or gate["overall_gate_passed"]:
        errors.append("Phase D decision is not the required fail-closed primary status")
    if set(gate["scientific_deliverables"]) != set(SCIENTIFIC_TABLES):
        errors.append("scientific deliverable ledger is incomplete")
    if set(gate["scientific_deliverables"].values()) != {"NOT_RUN_D_GATE_FAILED"}:
        errors.append("scientific deliverables are not uniformly gated off")
    for name in SCIENTIFIC_TABLES:
        if (output_root / name).exists():
            errors.append(f"gated scientific table was fabricated: {name}")

    with (output_root / "proof_to_code_map.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 14 or len({row["claim_id"] for row in rows}) != 14:
        errors.append("proof map does not contain 14 unique claims")
    if {row["status"] for row in rows} - ALLOWED_MAP_STATUSES:
        errors.append("proof map contains an invalid status")
    by_id = {row["claim_id"]: row for row in rows}
    for claim in ("FP_NO_FTZ_STARTUP", "STRICT_VERSUS_PARITY", "POLYNOMIAL_ONLY_UNCONDITIONAL"):
        if by_id.get(claim, {}).get("status") != "CONTRADICTED":
            errors.append(f"required contradiction missing: {claim}")

    for device in ("cpu", "cuda"):
        kernel = json.loads((output_root / "raw_logs" / f"proof_kernel_{device}.json").read_text())
        if not kernel["gate_passed"] or kernel["d1"]["passed"] != 7:
            errors.append(f"D1 evidence failed for {device}")
        if kernel["d2"]["passed"] != kernel["d2"]["checked"] or kernel["d2"]["checked"] != 987:
            errors.append(f"D2 evidence failed for {device}")
        if device == "cuda" and not kernel["cuda_kernel_available"]:
            errors.append("CUDA evidence used no shipped fused kernel")

    upstream_header, upstream_body = capture_header(output_root / "upstream_tests.log")
    mapped_header, mapped_body = capture_header(output_root / "raw_logs" / "upstream_path_mapped_replay.log")
    if upstream_header["returncode"] != 1 or "5 failed, 997 passed, 7 skipped, 5 xfailed" not in upstream_body:
        errors.append("raw upstream test classification changed")
    if mapped_header["returncode"] != 0 or "5 passed" not in mapped_body:
        errors.append("path-mapped upstream replay did not close all five portability failures")
    torch_header, torch_body = capture_header(output_root / "raw_logs" / "torch_full_tests.log")
    if torch_header["returncode"] != 0 or "845 passed, 2 skipped" not in torch_body:
        errors.append("final complete Torch suite did not pass with the recorded count")

    report = repo_root / "docs" / "HUAN_ENGINE_REPRODUCTION_AUDIT_20260826.md"
    proof_report = repo_root / "docs" / "HUAN_ENGINE_PROOF_CONTRACT_20260826.md"
    if not report.is_file() or not proof_report.is_file():
        errors.append("required audit report missing")
    else:
        report_text = report.read_text(encoding="utf-8")
        if f"Primary status: `{PRIMARY}`" not in report_text:
            errors.append("final report primary status does not match the gate")
        if "NOT_RUN_D_GATE_FAILED" not in report_text:
            errors.append("final report omits the gated scientific deliverables")
    if (repo_root / "docs" / "HUAN_REPRO_BLOCKED_MISSING_ENGINE_SOURCE_20260826.md").exists():
        errors.append("obsolete missing-source report still exists")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    errors = verify(args.repo_root.resolve(), args.output_root.resolve())
    print(json.dumps({"schema": "torch_tm_flowpipe.huan_package_verifier/1", "ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
