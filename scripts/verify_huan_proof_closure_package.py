#!/usr/bin/env python3
"""Verify Huan closure artifacts without rerunning any scientific operation.

Scope is deliberately limited to file hashes, schemas, provenance, recorded
route invocation, and cross-file consistency.  Scientific execution belongs
exclusively to ``run_huan_scientific_gate.py`` and is never inferred from this
verifier returning success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


HUAN_HEAD = "743f6205e6408072193ad76e940e7f15030e8d3c"
FLOWSTAR_HEAD = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
TORCH_C2_SCIENTIFIC = "29c9ee8f1fe96b860052b86a2b37d79a37bbb2ca"
PRIMARY = "HUAN_PROOF_CONTRACT_CLOSED__VDP_CONTRACT_NOT_PORTABLE"
STOP = "NOT_RUN_AFTER_CONTRACT_PORTABILITY_STOP"
REQUIRED = (
    "source_manifest.json",
    "environment.txt",
    "proof_to_code_map_v2.csv",
    "strict_roundoff_sources.csv",
    "phase_d_gate_v2.json",
    "d2_schedule_inventory.csv",
    "d2_host_order_oracle.json",
    "d2_actual_cpu.json",
    "d2_actual_cuda.json",
    "raw_logs/proof_kernel_cpu.json",
    "raw_logs/proof_kernel_cuda.json",
    "dirty_patch_recovery.json",
    "dirty_patch_candidates.csv",
    "chunk_boundary_cpu_v2.json",
    "chunk_boundary_cuda_v2.json",
    "refinement_boundary_cpu_v2.json",
    "refinement_boundary_cuda_v2.json",
    "witnesses/d6_minimal_witness.json",
    "witnesses/d6_repaired_minimal_witness.json",
    "commands/phase_d_scientific_gate.log",
    "commands/vdp_huan_phase_e.log",
    "commands/huan_plant_full.log",
    "commands/torch_full.log",
    "commands/torch_full_py11.log",
    "vdp/contract_matrix.csv",
    "vdp/step1_common_input.csv",
    "vdp/refinement_ledgers.jsonl.gz",
    "vdp/fixed_horizon_matrix.csv",
    "vdp/native_terminal.json",
    "vdp/first_divergence.json",
    "vdp/phase_e_decision.json",
    "vdp/superseded_runs.json",
    "vdp/huan_final/run_index.json",
    "SHA256SUMS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def capture_header(path: Path) -> tuple[dict[str, Any], str]:
    header, separator, body = path.read_text(encoding="utf-8").partition(
        "\n--- combined stdout/stderr ---\n"
    )
    if not separator:
        raise ValueError(f"capture delimiter missing: {path}")
    return json.loads(header), body


def verify_checksums(output: Path) -> list[str]:
    errors: list[str] = []
    checksum = output / "SHA256SUMS"
    expected: dict[str, str] = {}
    if not checksum.is_file():
        return ["SHA256SUMS missing"]
    for line in checksum.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative or relative in expected:
            errors.append(f"malformed or duplicate checksum line: {line}")
        else:
            expected[relative] = digest
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != checksum
    }
    for relative in sorted(actual | set(expected)):
        path = output / relative
        if relative not in actual:
            errors.append(f"checksum target missing: {relative}")
        elif relative not in expected:
            errors.append(f"uncovered file: {relative}")
        elif _sha256(path) != expected[relative]:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def verify_d2_route_evidence(payload: Mapping[str, Any], device: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != "torch_tm_flowpipe.huan_proof_kernel_audit/2":
        return [f"D2 {device} is not route-tagged schema v2"]
    d2 = payload.get("d2")
    if not isinstance(d2, Mapping):
        return [f"D2 {device} table missing"]
    rows = d2.get("rows")
    if not isinstance(rows, list) or not rows:
        return [f"D2 {device} rows missing"]
    required = {
        "schedule_name", "execution_backend", "actual_device", "kernel_path",
        "kernel_invocation_observed", "m", "finite_hypotheses_satisfied",
        "m_u_gate", "exact_error", "computed_inflation", "containment",
    }
    for index, row in enumerate(rows, 1):
        missing = required - set(row)
        if missing:
            errors.append(f"D2 {device} row {index} missing {sorted(missing)}")
        if row.get("status") != "PASS" or not row.get("containment"):
            errors.append(f"D2 {device} row {index} failed containment")
        if not row.get("finite_hypotheses_satisfied") or not row.get("m_u_gate"):
            errors.append(f"D2 {device} row {index} violated theorem hypotheses")
        if row.get("execution_backend") != "host_python" and not row.get("kernel_invocation_observed"):
            errors.append(f"D2 {device} row {index} lacks actual invocation")
    checked = len(rows)
    if d2.get("checked") != checked or d2.get("passed") != checked:
        errors.append(f"D2 {device} counts are not row-derived")
    expected = "torch_cuda" if device == "cuda" else "torch_cpu"
    if not any(row.get("execution_backend") == expected for row in rows):
        errors.append(f"D2 {device} lacks {expected} execution")
    if device == "cuda":
        custom = [row for row in rows if row.get("execution_backend") == "custom_cuda"]
        if not custom:
            errors.append("D2 CUDA lacks custom_cuda execution")
        elif any(
            not row.get("kernel_invocation_observed")
            or not isinstance(row.get("custom_cuda_invocation_count"), int)
            or row["custom_cuda_invocation_count"] <= 0
            for row in custom
        ):
            errors.append("D2 CUDA custom route lacks nonzero invocation evidence")
    return errors


def verify_phase_d(gate: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if gate.get("schema") != "torch_tm_flowpipe.huan_scientific_gate/2":
        errors.append("Phase-D scientific gate schema mismatch")
    if gate.get("engine_head") != HUAN_HEAD:
        errors.append("Phase-D gate engine SHA mismatch")
    gates = gate.get("gates")
    required = {
        "D1_ELEMENTWISE_NO_FTZ_NONFINITE", "D2_ANY_ORDER_ACTUAL_ROUTES",
        "D3_DENSE_SPARSE_SUPPORT", "D4_CHUNK_LANE", "D5_REFINEMENT_LEDGER_CACHE",
        "D6_STRICT_ROUNDOFF",
    }
    if not isinstance(gates, Mapping) or set(gates) != required:
        errors.append("Phase-D gate set mismatch")
    elif any(item.get("passed") is not True for item in gates.values()):
        errors.append("Phase-D contains an altered or failed D gate")
    if gate.get("overall_gate_passed") is not True or gate.get("phase_e_authorized") is not True:
        errors.append("Phase-D does not authorize Phase E")
    return errors


def _blank_scientific(row: Mapping[str, str]) -> bool:
    scientific = (
        "completed_horizon", "accepted_steps", "rejected_attempts",
        "refinement_iterations", "status_code", "endpoint_x", "endpoint_y",
        "endpoint_x_width", "endpoint_y_width", "segment_tube_x", "segment_tube_y",
        "segment_tube_x_width", "segment_tube_y_width", "runtime_s",
        "peak_gpu_memory_bytes",
    )
    return all(row.get(key, "") in {"", "null"} for key in scientific)


def verify_phase_e(output: Path) -> list[str]:
    errors: list[str] = []
    decision = _json(output / "vdp/phase_e_decision.json")
    if decision.get("primary_status") != PRIMARY:
        errors.append("Phase-E primary status mismatch")
    if decision.get("huan_fixed_runs_executed") != 8 or decision.get("huan_fixed_runs_completed") != 8:
        errors.append("Phase-E Huan fixed-run count mismatch")
    if decision.get("contract_was_changed") is not False:
        errors.append("Phase-E claims the frozen contract was changed")
    if decision.get("native_parity") != "NOT_RUN_CONTRACT_NOT_PORTABLE" or decision.get("native_strict") != "NOT_RUN_CONTRACT_NOT_PORTABLE":
        errors.append("Phase-E native portability stop missing")
    authoritative = _json(output / "vdp/huan_final/run_index.json")
    if authoritative.get("engine_head") != HUAN_HEAD or authoritative.get("primary_status") != PRIMARY:
        errors.append("authoritative Huan Phase-E run index source/status mismatch")
    superseded = _json(output / "vdp/superseded_runs.json")
    if superseded.get("authoritative_huan_sha") != HUAN_HEAD or any(
        row.get("eligible_for_final_claims") is not False
        for row in superseded.get("superseded", [])
    ):
        errors.append("superseded Huan run ledger mismatch")
    rows = _csv(output / "vdp/fixed_horizon_matrix.csv")
    huan = [row for row in rows if row["lane"] in {"huan_parity", "huan_strict"}]
    stopped = [row for row in rows if row["lane"] in {"stock_flowstar", "torch_c2"}]
    if len(rows) != 16 or len(huan) != 8 or len(stopped) != 8:
        errors.append("fixed horizon matrix has the wrong lane/scenario cardinality")
    if any(row["execution_status"] != "EXECUTED" or row["completed_requested_horizon"] != "True" for row in huan):
        errors.append("a Huan fixed run is incomplete or mislabeled")
    if any(row["execution_status"] != STOP or not _blank_scientific(row) for row in stopped):
        errors.append("stopped Flow*/Torch row contains fabricated scientific data")
    native = _json(output / "vdp/native_terminal.json")
    for lane in ("huan_parity", "huan_strict"):
        if native.get(lane, {}).get("status") != "NOT_RUN_CONTRACT_NOT_PORTABLE":
            errors.append(f"{lane} native stop missing")
    divergence = _json(output / "vdp/first_divergence.json")
    if divergence.get("status") != "NOT_ADJUDICATED_CONTRACT_PORTABILITY_STOP":
        errors.append("cross-tool divergence was fabricated or mislabeled")
    if divergence.get("huan_vs_flowstar") is not None or divergence.get("huan_vs_torch_c2") is not None:
        errors.append("cross-tool divergence contains fabricated values")
    return errors


def verify_refinement_evidence(payload: Mapping[str, Any], device: str) -> list[str]:
    """Check the recorded D5 behavioral and cache-freshness contract."""
    errors: list[str] = []
    cache = payload.get("cache_freshness", {})
    if not payload.get("behavioral_passed") or not payload.get("contract_gate_passed"):
        errors.append(f"D5 {device} behavioral/contract gate failed")
    if not cache.get("stale_generation_rejected") or not cache.get("stale_owner_rejected"):
        errors.append(f"D5 {device} stale cache evidence missing")
    return errors


def verify(repo: Path, output: Path) -> list[str]:
    errors = [f"required output missing: {relative}" for relative in REQUIRED if not (output / relative).is_file()]
    required_reports = (
        repo / "docs/HUAN_STRICT_PROOF_CONTRACT_CLOSURE_20260826.md",
        repo / "docs/HUAN_FROZEN_VDP_PHASE_E_20260826.md",
        repo / "docs/strict_roundoff_accounting_graph.md",
    )
    errors.extend(f"required report missing: {path.name}" for path in required_reports if not path.is_file())
    if errors:
        return errors
    errors.extend(verify_checksums(output))
    manifest = _json(output / "source_manifest.json")
    repos = manifest.get("repositories", {})
    expected = {
        "huan": HUAN_HEAD,
        "torch_c2_scientific": TORCH_C2_SCIENTIFIC,
        "flowstar": FLOWSTAR_HEAD,
    }
    for key, head in expected.items():
        row = repos.get(key, {})
        if row.get("head") != head or row.get("tracked_clean") is not True:
            errors.append(f"source manifest mismatch: {key}")
    if manifest.get("historical_dirty_patch_recovery") != "HISTORICAL_DIRTY_PATCHES_NOT_FOUND_AFTER_BOUNDED_SEARCH":
        errors.append("historical dirty-patch conclusion mismatch")
    gate = _json(output / "phase_d_gate_v2.json")
    errors.extend(verify_phase_d(gate))
    for device in ("cpu", "cuda"):
        errors.extend(verify_d2_route_evidence(_json(output / f"raw_logs/proof_kernel_{device}.json"), device))
        refinement = _json(output / f"refinement_boundary_{device}_v2.json")
        errors.extend(verify_refinement_evidence(refinement, device))
        if not _json(output / f"chunk_boundary_{device}_v2.json").get("gate_passed"):
            errors.append(f"D4 {device} chunk gate failed")
    original = _json(output / "witnesses/d6_minimal_witness.json")
    repaired = _json(output / "witnesses/d6_repaired_minimal_witness.json")
    if original.get("classification") != "D6_CONCRETE_UNDERENCLOSURE_WITNESS_FOUND":
        errors.append("D6 original concrete witness classification missing")
    if original.get("parity_or_legacy_point_phi", {}).get("all_contained") is not False:
        errors.append("D6 original witness no longer demonstrates under-enclosure")
    if repaired.get("classification") != "D6_REPAIRED_STRICT_CONTAINS_MINIMAL_WITNESS":
        errors.append("D6 repaired witness classification missing")
    if repaired.get("strict_interval_phi", {}).get("all_contained") is not True:
        errors.append("D6 repaired strict witness is not contained")
    proof_map = _csv(output / "proof_to_code_map_v2.csv")
    if len(proof_map) != 14 or any(row["status"] not in {"MAPPED_AND_TESTED", "ASSUMPTION_ONLY"} for row in proof_map):
        errors.append("proof-to-code map status/cardinality mismatch")
    strict_sources = _csv(output / "strict_roundoff_sources.csv")
    if not strict_sources or any(row["status"] != "MAPPED_AND_TESTED" for row in strict_sources):
        errors.append("strict roundoff source ledger is incomplete")
    errors.extend(verify_phase_e(output))
    for relative in ("commands/phase_d_scientific_gate.log", "commands/vdp_huan_phase_e.log", "commands/huan_plant_full.log", "commands/torch_full_py11.log"):
        header, body = capture_header(output / relative)
        if header.get("schema") != "torch_tm_flowpipe.huan_command_capture/2" or header.get("returncode") != 0:
            errors.append(f"command capture failed or has wrong schema: {relative}")
        if relative.endswith(("huan_plant_full.log", "torch_full_py11.log")) and "passed" not in body:
            errors.append(f"full regression pass count absent: {relative}")
    failed_header, failed_body = capture_header(output / "commands/torch_full.log")
    if failed_header.get("returncode") != 2 or "No module named 'pandas'" not in failed_body:
        errors.append("Torch wrong-environment diagnostic capture changed or is missing")
    for report in required_reports[:2]:
        text = report.read_text(encoding="utf-8")
        if PRIMARY not in text or "package verifier" not in text.lower():
            errors.append(f"report omits primary status/verifier scope: {report.name}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    errors = verify(args.repo_root.resolve(), args.output_root.resolve())
    print(
        json.dumps(
            {
                "schema": "torch_tm_flowpipe.huan_proof_closure_package_verifier/1",
                "scope": "artifact_hash_schema_provenance_consistency_only__does_not_rerun_science",
                "ok": not errors,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
