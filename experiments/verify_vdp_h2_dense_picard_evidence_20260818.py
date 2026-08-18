#!/usr/bin/env python3
"""Fail closed on VDP H2 evidence integrity, soundness, and failed targets."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
import xml.etree.ElementTree as ET


SCIENTIFIC_SHA = "666c51ecc5575f203518d21f34b5c9948741fb17"
BASE_SHA = "43be6d34461e809c291a2d57e120012755d29d51"
EMPTY_DIFF_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CHANNELS = ("endpoint_x", "endpoint_y", "segment_x", "segment_y")


class VerificationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_integrity(package: Path, manifest: dict[str, Any]) -> tuple[int, int]:
    expected = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    }
    records = manifest["files"]
    _require(set(records) == expected, "manifest file coverage")
    for relative, record in records.items():
        path = package / relative
        _require(path.stat().st_size == int(record["bytes"]), f"size: {relative}")
        _require(_sha(path) == record["sha256"], f"sha256: {relative}")

    checksums: dict[str, str] = {}
    for line in (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        _require(relative not in checksums and len(digest) == 64, "checksum syntax")
        checksums[relative] = digest
    checksum_expected = {
        path.relative_to(package).as_posix(): _sha(path)
        for path in package.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    _require(checksums == checksum_expected, "SHA256SUMS coverage or digest")

    for row in manifest["compressed_sources"]:
        stored = package / row["stored"]
        decompressed = gzip.decompress(stored.read_bytes())
        _require(len(decompressed) == int(row["source_bytes"]), f"decompressed bytes: {stored}")
        _require(
            hashlib.sha256(decompressed).hexdigest() == row["source_sha256"],
            f"decompressed sha256: {stored}",
        )
    return len(expected), sum((package / relative).stat().st_size for relative in expected)


def _verify_tests(package: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for path in sorted((package / "04_tests").glob("*.xml")):
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        counts = {
            key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        _require(counts["tests"] > 0, f"empty test report: {path.name}")
        _require(counts["failures"] == 0 and counts["errors"] == 0, f"failed tests: {path.name}")
        result[path.name] = counts
    _require(len(result) == 4, "test report count")
    _require(result["h1_clean_base_targeted.xml"]["tests"] == 11, "H1 targeted count")
    _require(result["h1_clean_base_full.xml"]["tests"] == 770, "H1 full count")
    _require(result["h2_clean_scientific_full.xml"]["tests"] >= 782, "H2 full count")
    return result


def verify(package: Path) -> dict[str, Any]:
    package = package.resolve()
    manifest = _load(package / "manifest.json")
    _require(manifest["schema"] == "vdp_h2_dense_picard_first_loss_evidence_v1", "schema")
    _require(manifest["base_sha"] == BASE_SHA, "base SHA")
    _require(manifest["scientific_sha"] == SCIENTIFIC_SHA, "scientific SHA")
    files, bytes_total = _verify_integrity(package, manifest)

    provenance = _load(package / "00_provenance/provenance.json")
    _require(provenance["scientific_sha"] == SCIENTIFIC_SHA, "provenance scientific SHA")
    _require(provenance["phase0_h1_clean_replay"]["porcelain_before_after"] == "clean", "Phase0 clean")
    _require(provenance["environment"]["cuda_device_0"] == "Tesla V100-SXM2-16GB", "V100 device")

    ledger = _load(package / "01_gates/production_operator_ledger.json")
    gate = _load(package / "01_gates/summary.json")
    same_input = _load(package / "01_gates/gate_b_same_input_matrix.json")
    _require(ledger["production_source_commit"] == SCIENTIFIC_SHA, "ledger SHA")
    _require(ledger["working_diff_sha256"] == EMPTY_DIFF_SHA256, "ledger dirty diff")
    _require(
        ledger["scientific_source_identity"]["all_production_paths_byte_identical"] is True,
        "scientific production source identity",
    )
    _require(len(ledger["picard_iterations"]) == 28, "Picard operator stage count")
    _require(gate["picard_iteration_count"] == 4, "Picard iteration count")
    _require(gate["operator_stage_count"] == 36, "complete operator stage count")
    _require(gate["gate_a_pass"] is True and gate["gate_b_pass"] is True, "Gate A/B")
    _require(gate["all_operator_stages_exact_bernstein_contained"] is True, "operator oracle")
    _require(gate["all_poly_diff_exact_bernstein_contained"] is True, "poly_diff oracle")
    _require(gate["same_input_byte_identity"] is True, "same input")
    first = gate["first_extra_enclosure"]
    _require(first["stage"] == "raw.B1.x_squared", "first overwide stage")
    _require(first["specific_interval_term"].startswith("R_left * R_right"), "first interval term")
    _require(
        first["strict_extra_lower_width"]
        == "54445178707350159372354900760041/5444517870735015415413993718908291383296",
        "exact first increment",
    )
    _require(gate["raw_residual_excess"]["y"]["fraction_of_legacy_excess_removed"] >= 0.10, "Gate B threshold")
    _require(all(row["no_regression"] for row in gate["raw_residual_excess"].values()), "raw no regression")
    _require(all(row["no_regression"] for row in gate["segment_excess"].values()), "segment no regression")
    _require(same_input["preregistration"]["B3"]["status"] == "not_executed_stop_after_first_pass", "B3 stop")
    _require(same_input["preregistration"]["B4"]["status"] == "not_executed_stop_after_first_pass", "B4 stop")
    for cell_name in ("B1", "B2"):
        eps_ledger = ledger["gate_b_cells"][cell_name]["validation_eps_ledger"]
        _require(eps_ledger["expected_execution_count"] == 5, f"eps expected count: {cell_name}")
        _require(eps_ledger["actual_execution_count"] == 5, f"eps actual count: {cell_name}")
        _require(eps_ledger["production_order_complete"] is True, f"eps order: {cell_name}")
        _require(eps_ledger["ordinary_residual_trace_matches"] is True, f"ordinary trace: {cell_name}")
        _require(
            [row["sequence"] for row in eps_ledger["records"]] == [1, 2, 3, 4, 5],
            f"eps sequence: {cell_name}",
        )
        _require(
            [row["stage"] for row in eps_ledger["records"]]
            == [
                "candidate_seed",
                "ordinary_residual_diagnostic",
                "tau_times_raw_rhs",
                "poly_diff",
                "final_raw_compat_image",
            ],
            f"eps stages: {cell_name}",
        )

    matrix = _load(package / "02_scientific_matrix/matrix.json")
    _require(matrix["scientific_sha"] == SCIENTIFIC_SHA, "matrix SHA")
    _require(matrix["decision"] == "H2_OPERATOR_ACCEPTED__OVERALL_SUCCESS_TARGET_FAILED", "decision")
    gates = matrix["gates"]
    for key in (
        "gate_a_exact_operator_ledger",
        "gate_b_same_input_operator",
        "T6p32_no_channel_regression_vs_H1",
        "native_at_least_6p441433080631058",
        "runtime_at_most_2x_legacy",
        "v100_all_lanes_measured",
        "cpu_v100_consistent_at_1e_12",
    ):
        _require(gates[key] is True, f"matrix gate: {key}")
    _require(gates["T1_T3_all_four_channels_remove_10pct_legacy_excess"] is False, "early failure accounting")
    _require(gates["reaches_T10_stretch"] is False, "T10 failure accounting")
    for horizon in ("T1", "T3", "T6p32"):
        _require(set(matrix["fixed"][horizon]) == set(CHANNELS), f"channels: {horizon}")
    _require(
        all(not matrix["fixed"]["T1"][channel]["h1_h2_meets_10pct"] for channel in CHANNELS),
        "T1 per-channel failures",
    )
    _require(
        sum(matrix["fixed"]["T3"][channel]["h1_h2_meets_10pct"] for channel in CHANNELS) == 2,
        "T3 two-channel pass",
    )
    _require(
        all(matrix["fixed"]["T6p32"][channel]["h1_h2_no_wider_than_h1"] for channel in CHANNELS),
        "T6.32 H1 comparison",
    )
    native = matrix["native_T10_requests"]
    _require(native["legacy"]["completed_horizon"] == 6.397083942944808, "native legacy endpoint")
    _require(native["h1"]["completed_horizon"] == 6.441433080631058, "native H1 endpoint")
    _require(native["h1_h2"]["completed_horizon"] == 6.482041958201616, "native H2 endpoint")
    _require(native["h1_h2"]["accepted_steps"] == 278, "native H2 accepted")
    _require(native["h1_h2"]["rejected_attempts"] == 44, "native H2 rejected")
    rejection = matrix["native_rejection_diagnostics"]["h1_h2"]
    _require(rejection["limiting_component"] == "y", "native H2 limiting component")
    _require(rejection["limiting_side"] == "upper", "native H2 limiting side")
    _require(rejection["subset_margin"] == -6.854200524201504e-06, "native H2 limiting margin")
    _require(
        rejection["largest_additive_validated_ledger_category"] == "polynomial_truncation",
        "native H2 largest rejection ledger category",
    )
    _require(all(value <= 2.0 for value in matrix["runtime_ratios"].values()), "runtime ratios")
    for lane in ("legacy", "h1", "h1_h2"):
        _require(matrix["v100_T0p1"][lane]["device"] == "cuda", f"V100 measured: {lane}")
        _require(matrix["cpu_v100_consistency_T0p1"][lane]["consistent_at_1e_12"], f"CPU/V100: {lane}")
        for run_group in ("fixed_T6p32_runs", "native_T10_requests", "cpu_T0p1", "v100_T0p1"):
            row = matrix[run_group][lane]
            _require(row["commit"] == SCIENTIFIC_SHA, f"run SHA: {run_group}/{lane}")
            _require(row["tracked_diff_sha256"] == EMPTY_DIFF_SHA256, f"run diff: {run_group}/{lane}")
            _require(row["worktree_dirty"] is False, f"run clean: {run_group}/{lane}")

    tests = _verify_tests(package)
    result = {
        "status": "verified",
        "files": files,
        "bytes": bytes_total,
        "operator_stages": gate["operator_stage_count"],
        "decision": matrix["decision"],
        "tests": tests,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    verify(parse_args().package)
