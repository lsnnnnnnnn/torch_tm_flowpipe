#!/usr/bin/env python3
"""Fail-closed, recomputing verifier for the compact VDP G2 package."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET


CANDIDATE = "normalized_insertion_bounded_shared_source_o4_g2"
PARTIAL = "G2_MECHANISM_IMPROVED__PRODUCTION_GATE_NOT_MET"
VALIDATED = "G2_VDP_T10_VALIDATED"
REJECTED = "G2_SHARED_COLUMN_CARRY_REJECTED"
TOTAL = "LOSSLESS_CROSS_OPERATOR_CELL_UNAVAILABLE__TOTAL_CAUSE_OPEN"
LEGACY_NATIVE = 6.397083942944808
CHANNELS = ("endpoint_x", "endpoint_y", "segment_tube_x", "segment_tube_y")


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def classify_g2(*, production_success: bool, reached_t10: bool, mechanism_improved: bool) -> str:
    """Derive the scientific label from gates, without assuming this package's outcome."""
    if production_success and reached_t10:
        return VALIDATED
    if mechanism_improved:
        return PARTIAL
    return REJECTED


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_integrity(root: Path) -> dict[str, Any]:
    manifest = load(root / "manifest.json")
    require(manifest.get("schema") == "vdp_g2_shared_column_evidence_package_v1", "manifest schema")
    records = manifest.get("files")
    require(isinstance(records, dict), "manifest file records")
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    }
    require(set(records) == expected, "manifest coverage mismatch")
    for relative, record in records.items():
        path = root / relative
        require(path.stat().st_size == int(record["bytes"]), f"size mismatch: {relative}")
        require(sha(path) == record["sha256"], f"hash mismatch: {relative}")
    sums: dict[str, str] = {}
    for number, line in enumerate((root / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise VerificationError(f"malformed checksum line {number}") from exc
        require(len(digest) == 64 and relative not in sums, f"invalid checksum line {number}")
        sums[relative] = digest
    require(sums == {name: row["sha256"] for name, row in records.items()}, "SHA256SUMS mismatch")
    total = sum((root / relative).stat().st_size for relative in expected)
    require(total < 25 * 1024 * 1024, "package exceeds 25 MiB")
    require(manifest.get("under_25_mib") is True, "manifest size decision")
    return {"file_count": len(expected), "bytes": total, "manifest": manifest}


def verify_contract(root: Path) -> None:
    contract = load(root / "00_provenance/vdp_g2_shared_column_contract_20260815.json")
    require(contract["ode"]["rhs"] == ["y", "(1-x^2)*y-x"], "ODE drift")
    require(contract["initial_set_exact_decimal"] == [["1.1", "1.4"], ["2.35", "2.45"]], "initial set")
    require(contract["order"] == 4 and contract["fixed_schedule"]["h_decimal"] == "0.01", "O4/h")
    require(contract["target_remainder_radius"] == "0.0001" and contract["cutoff"] == "1e-10", "target/cutoff")
    require(contract["range"] == {
        "method": "adaptive_subdivision",
        "trigger": "proactive_depth1_on_named_contexts",
        "max_depth": 1,
        "max_leaves": 4,
        "split_variables": [0, 1],
        "named_contexts": ["polynomial_truncation"],
    }, "range contract")
    require(contract["g2_shape"]["vdp_total_variables"] == 6 and contract["g2_shape"]["generations"] == 2, "G2 shape")


def verify_gate_a(root: Path) -> dict[str, Any]:
    matrix = load(root / "01_gate_a/operator_matrix.json")
    rows = matrix["matrix"]
    require(len(rows) == 20 and len(matrix["positions"]) == 5, "Gate A matrix coverage")
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["operator"], row["prestate"], row["status"])
        counts[key] = counts.get(key, 0) + 1
    require(counts.get(("Flowstar", "Flowstar", "EXECUTED_NATIVE_LOSSLESS")) == 5, "Flow* native cells")
    require(counts.get(("Torch", "Torch", "EXECUTED_NATIVE_LOSSLESS")) == 5, "Torch native cells")
    require(counts.get(("Torch", "Flowstar", "UNAVAILABLE_LOSSLESS_CROSS_OPERATOR_CELL")) == 5, "Torch cross cells")
    require(counts.get(("Flowstar", "Torch", "EXECUTED_FAIL_CLOSED_SCHEMA_REFUSAL")) == 5, "Flow* cross refusal")
    bridge = matrix["flowstar_native_roundtrip_summary"]
    require(bridge["canonical_byte_roundtrips_exact"] == bridge["fixture_count"] == 28, "Flow* byte round trips")
    require(bridge["next_step_roundtrips_exact"] == 28, "Flow* continuation parity")
    require(matrix["common_component_box_used"] is False and matrix["queue_dropped"] is False, "lossy Gate A adapter")
    computed = TOTAL if counts.get(("Torch", "Flowstar", "UNAVAILABLE_LOSSLESS_CROSS_OPERATOR_CELL")) else "INVALID"
    require(computed == matrix["conclusion"] and matrix["total_cause_closed"] is False, "Gate A conclusion")
    return {"positions": 5, "cells": 20, "conclusion": computed}


def verify_oracle(root: Path) -> dict[str, Any]:
    oracle = load(root / "03_oracle/independent_oracle.json")
    require(oracle["status"] == "PASS" and oracle["checks_passed"] >= 15, "stored independent oracle")
    require(oracle["imports_project_core"] is False and oracle["sampling_used"] is False, "oracle independence labels")
    source = root / "00_provenance/source_snapshot/experiments/independent_g2_exact_oracle.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("import torch\n", "from torch_tm_flowpipe", "import torch_tm_flowpipe"):
        require(forbidden not in text, f"independent oracle forbidden import: {forbidden!r}")
    with tempfile.TemporaryDirectory(prefix="verify-g2-oracle-") as temporary:
        out = Path(temporary) / "oracle.json"
        completed = subprocess.run(
            [sys.executable, str(source), "--input", str(root / "03_oracle/blackbox.json"), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        require(completed.returncode == 0, f"independent oracle execution: {completed.stderr}")
        recomputed = load(out)
    require(recomputed["status"] == "PASS" and recomputed["checks_passed"] == oracle["checks_passed"], "recomputed oracle")
    require(recomputed["selected_exact_tables"] == oracle["selected_exact_tables"], "oracle exact table drift")
    return {"checks": oracle["checks_passed"]}


def verify_gate_b(root: Path) -> dict[str, Any]:
    fixed = load(root / "02_gate_b/fixed_owner_interventions.json")
    require(len(fixed["rows"]) == 5, "Gate B fixed positions")
    passed_controls = 0
    not_applicable = 0
    for row in fixed["rows"]:
        consumers = row["consumers"]
        require(
            consumers["g1_actual"]["consumer_output_sha256"]
            == consumers["g1_actual__metadata_tamper"]["consumer_output_sha256"],
            "G1 actual metadata control",
        )
        for name, status in row["control_status"].items():
            if status["payload_control"].startswith("NOT_APPLICABLE"):
                require(float(status.get("maximum_source_coefficient", 0.0)) <= 1e-10, "Gate B N/A cutoff")
                not_applicable += 1
                continue
            payload = consumers[f"{name}__payload_tamper_x2"]
            metadata = consumers[f"{name}__metadata_tamper"]
            require(consumers[name]["consumer_output_sha256"] != payload["consumer_output_sha256"], "Gate B payload control")
            require(consumers[name]["consumer_output_sha256"] == metadata["consumer_output_sha256"], "Gate B metadata control")
            passed_controls += 1
    terminal = load(root / "02_gate_b/terminal_owner_interventions.json")
    require(terminal["terminal_time"] > 6.0 and terminal["failure_message"], "G1 terminal prestate")
    for name, status in terminal["controls"].items():
        if status["payload_control"].startswith("NOT_APPLICABLE"):
            continue
        consumers = terminal["consumers"]
        require(consumers[name]["consumer_output_sha256"] != consumers[f"{name}__payload_tamper_x2"]["consumer_output_sha256"], "terminal payload control")
        require(consumers[name]["consumer_output_sha256"] == consumers[f"{name}__metadata_tamper"]["consumer_output_sha256"], "terminal metadata control")
    owners = fixed["rows"][-1]["owners"]
    ordinary = sum(float(row["width"]) for row in owners["carried_ordinary_owners"])
    retired = sum(float(row["width"]) for row in owners["retired_source_owners"])
    require(ordinary > 1.0 and retired < 0.01 and ordinary > retired * 100, "ordinary-owner dominance")
    return {"fixed_positions": 5, "passed_controls": passed_controls, "not_applicable": not_applicable, "terminal_time": terminal["terminal_time"]}


def verify_resume(root: Path) -> dict[str, Any]:
    audit = load(root / "03_oracle/checkpoint_resume/audit.json")
    require(audit["status"] == "PASS" and audit["checkpoint_schema"].endswith("_v4"), "G2 checkpoint audit")
    for key in (
        "checkpoint_bytes_equal",
        "fresh_resume_continuation_hashes_equal",
        "fresh_resume_end_checkpoint_bytes_equal",
        "rejected_retry_fingerprint_and_payload_immutable",
    ):
        require(audit[key] is True, key)
    require(audit["fixed_variable_count"] == 6, "resume variable shape")
    return {"accepted_steps": audit["accepted_steps"]}


def verify_matrix(root: Path) -> dict[str, Any]:
    matrix = load(root / "04_matrix/matrix.json")
    summary = load(root / "04_matrix/scientific_summary.json")
    require(matrix["request_count"] == len(matrix["rows"]) == 36, "request matrix count")
    keys = {(row["schedule"], row["mode"], float(row["requested_horizon"])) for row in matrix["rows"]}
    require(len(keys) == 36, "request matrix uniqueness")
    require(matrix["initialization"] == "exact_decimal_contract", "matrix initialization")
    selected: dict[tuple[int, str], dict[str, float]] = {}
    row_count = 0
    with gzip.open(root / "04_matrix/fixed_curve.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            step = int(row["step"])
            channel = row["channel"]
            for mode in ("flowstar", "legacy", "g1", "g2"):
                lo = float(row[f"{mode}_lo"])
                hi = float(row[f"{mode}_hi"])
                require(hi - lo == float(row[f"{mode}_width"]), f"raw width mismatch {step}/{channel}/{mode}")
            if step in (100, 300, 632):
                selected[(step, channel)] = {
                    mode: float(row[f"{mode}_width"])
                    for mode in ("flowstar", "legacy", "g1", "g2")
                }
    require(row_count == 632 * 4 and len(selected) == 12, "fixed curve coverage")
    t1 = all(selected[(100, channel)]["g2"] <= selected[(100, channel)]["g1"] for channel in CHANNELS)
    excess_gate = True
    mechanism = True
    fractions = []
    for step in (300, 632):
        for channel in CHANNELS:
            row = selected[(step, channel)]
            fraction = (row["legacy"] - row["g2"]) / (row["legacy"] - row["flowstar"])
            fractions.append(fraction)
            excess_gate &= fraction >= 0.10
            mechanism &= row["g2"] <= row["g1"]
    mechanism &= t1
    native = {
        row["mode"]: row
        for row in matrix["rows"]
        if row["schedule"] == "native" and float(row["requested_horizon"]) == 10.0
    }
    require(set(native) == {"legacy", "g1", "g2"}, "native T10 rows")
    native_gate = float(native["g2"]["completed_horizon"]) >= LEGACY_NATIVE
    fixed_g2_rows = [
        row for row in matrix["rows"]
        if row["schedule"] == "fixed" and row["mode"] == "g2"
    ]
    fixed_no_failure = bool(
        len(fixed_g2_rows) == 6
        and all(row["completed_requested_horizon"] and int(row["rejected_attempts"]) == 0 for row in fixed_g2_rows)
    )
    terminal = summary["native_terminal_details"]
    terminal_margin_gate = bool(
        native["g2"]["completed_requested_horizon"]
        or (
            terminal["g2"] is not None
            and terminal["legacy"] is not None
            and float(terminal["g2"]["subset_margin"][0][1])
            > float(terminal["legacy"]["subset_margin"][0][1])
        )
    )
    production = bool(t1 and excess_gate and native_gate and fixed_no_failure and terminal_margin_gate)
    computed = classify_g2(
        production_success=production,
        reached_t10=bool(native["g2"]["completed_requested_horizon"]),
        mechanism_improved=mechanism,
    )
    require(summary["conclusion"] == computed, "stored G2 decision")
    require(summary["gates"] == {
        "independent_oracle": True,
        "fixed_T1_all_four_no_wider_than_G1": t1,
        "fixed_T3_T6p32_all_four_remove_at_least_10pct_legacy_excess": excess_gate,
        "all_fixed_requests_complete_without_G2_rejection": fixed_no_failure,
        "native_G2_at_least_legacy_6p397083942944808": native_gate,
        "terminal_y_subset_margin_better_than_legacy_if_failure_remains": terminal_margin_gate,
        "production_success": production,
    }, "stored production gates")
    require(summary["total_cause_conclusion"] == TOTAL, "stored total-cause decision")
    return {
        "requests": 36,
        "curve_rows": row_count,
        "max_excess_fraction_removed": max(fractions),
        "g2_native_horizon": native["g2"]["completed_horizon"],
        "conclusion": computed,
    }


def verify_performance(root: Path) -> dict[str, Any]:
    perf = load(root / "05_performance/performance.json")
    require(perf["cuda_semantics"] == "implementation_consistency_only_not_formal_directed_rounding", "CUDA label")
    require(perf["kernel_only_speedup_extrapolated"] is False, "kernel extrapolation")
    if perf["cuda_available"]:
        rows = {row["device"]: row for row in perf["rows"]}
        require(set(rows) == {"cpu", "cuda"}, "performance lanes")
        require(rows["cuda"]["transfer_count"] > 0, "CUDA transfer count")
        for key in ("host_to_device_s", "dense_picard_range_validator_kernel_s", "device_to_host_s", "full_solver_runtime_s"):
            require(float(rows["cuda"][key]) > 0, f"CUDA phase timing {key}")
        speedup = rows["cpu"]["full_solver_runtime_s"] / rows["cuda"]["full_solver_runtime_s"]
        require(speedup == perf["cuda_over_cpu_full_solver_speedup"], "full solver speedup arithmetic")
        require(perf["full_solver_speedup_claimed"] == (speedup > 1.0), "speedup claim gate")
        require(perf["implementation_consistency"]["endpoint_fields_close"] is True, "CUDA endpoint consistency")
    return {"cuda_available": perf["cuda_available"], "speedup": perf["cuda_over_cpu_full_solver_speedup"]}


def test_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def verify_acceptance(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    tests = {}
    for name in ("focused", "full"):
        counts = test_counts(root / f"06_tests/{name}_pytest.xml")
        require(counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0, f"{name} tests")
        tests[name] = counts
    tamper = root / "06_tests/tamper_tests.json"
    if manifest.get("tamper_required"):
        value = load(tamper)
        require(value["passed"] is True and len(value["cases"]) == 3, "tamper tests")
        require(all(row["rejected"] for row in value["cases"]), "tamper rejection")
    fresh = load(root / "07_fresh_clone/acceptance.json")
    if manifest.get("stage") == "attestation":
        require(fresh["status"] == "PASS", "fresh clone acceptance")
        require(fresh["git_status_porcelain_empty"] is True, "fresh clone dirty")
    else:
        require(fresh["status"] in {"PENDING_SCIENTIFIC_SHA", "PASS"}, "fresh clone precommit state")
    return {"tests": tests, "fresh_clone": fresh["status"]}


def verify(package: Path) -> dict[str, Any]:
    root = package.resolve()
    require(root.is_dir(), "package missing")
    integrity = verify_integrity(root)
    manifest = integrity.pop("manifest")
    verify_contract(root)
    gate_a = verify_gate_a(root)
    gate_b = verify_gate_b(root)
    oracle = verify_oracle(root)
    resume = verify_resume(root)
    matrix = verify_matrix(root)
    performance = verify_performance(root)
    acceptance = verify_acceptance(root, manifest)
    require(manifest["conclusion"] == matrix["conclusion"], "manifest conclusion")
    require(manifest["total_cause_conclusion"] == gate_a["conclusion"] == TOTAL, "manifest total-cause")
    return {
        "schema": "vdp_g2_shared_column_verification_v1",
        "status": "PASS",
        "conclusion": matrix["conclusion"],
        "total_cause_conclusion": gate_a["conclusion"],
        "integrity": integrity,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "oracle": oracle,
        "resume": resume,
        "matrix": matrix,
        "performance": performance,
        "acceptance": acceptance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.package)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
