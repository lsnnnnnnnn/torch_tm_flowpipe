#!/usr/bin/env python3
"""Independently verify checksums and scientific claims in the G1 package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any


CONCLUSION = "T1_T3_WIDTH_CAUSE_CLOSED__EARLY_GAP_IMPROVED__TERMINAL_STILL_OPEN"
CANDIDATE = "normalized_insertion_bounded_source_ledger_o4_g1"
EXPECTED_RANGES = {
    "t1_excess_range": [0.002715258977108115, 0.008898245576982322],
    "t3_excess_range": [0.047012584088458986, 0.04881416425335772],
    "t6p32_excess_range": [0.7634365472439139, 1.4682484934615618],
}


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VerificationError(f"invalid JSON {path}: {exc}") from exc


def verify_checksums(package: Path) -> dict[str, Any]:
    manifest = load(package / "manifest.json")
    require(manifest.get("schema") == "vdp_t1_t3_bounded_source_evidence_package_v1", "manifest schema")
    records = manifest.get("files")
    require(isinstance(records, dict), "manifest file records missing")
    expected = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path not in {package / "manifest.json", package / "SHA256SUMS"}
    }
    require(set(records) == expected, "manifest coverage mismatch")
    for relative, record in records.items():
        path = package / relative
        require(path.stat().st_size == int(record["bytes"]), f"size mismatch: {relative}")
        require(sha(path) == record["sha256"], f"manifest hash mismatch: {relative}")
    sums: dict[str, str] = {}
    for number, line in enumerate((package / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise VerificationError(f"malformed SHA256SUMS line {number}") from exc
        require(relative not in sums and len(digest) == 64, f"invalid SHA256SUMS line {number}")
        sums[relative] = digest
    require(sums == {relative: record["sha256"] for relative, record in records.items()}, "SHA256SUMS mismatch")
    total = sum((package / relative).stat().st_size for relative in expected)
    require(total < 25 * 1024 * 1024, "package exceeds 25 MiB")
    require(manifest.get("under_25_mib") is True, "manifest size decision mismatch")
    require(manifest.get("conclusion") == CONCLUSION, "manifest conclusion")
    require(manifest.get("candidate") == CANDIDATE, "manifest candidate")
    return {"file_count": len(expected), "total_bytes": total}


def verify_tests(package: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for label in ("focused", "full"):
        path = package / f"06_tests/{label}_pytest.xml"
        require(path.is_file(), f"missing {label} pytest XML")
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        values = {
            key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        require(values["failures"] == 0 and values["errors"] == 0, f"{label} tests failed")
        require(values["tests"] > 0, f"{label} test suite empty")
        results[label] = values
    tamper = load(package / "06_tests/tamper_tests.json")
    require(tamper.get("passed") is True, "package tamper tests failed")
    require(
        {row.get("case") for row in tamper.get("cases", [])}
        == {
            "raw_scientific_mutation",
            "refinalized_semantic_conclusion_mutation",
            "required_oracle_file_deleted",
        },
        "package tamper coverage",
    )
    require(all(row.get("rejected") is True for row in tamper["cases"]), "tampered package accepted")
    results["tamper"] = {"cases": len(tamper["cases"]), "passed": True}
    return results


def verify(package: Path) -> dict[str, Any]:
    package = package.resolve()
    require(package.is_dir(), "package directory missing")
    checksum = verify_checksums(package)

    width = load(package / "01_width_ledger/summary.json")
    require(width["fixed_common_prefix_rows"] == 632, "fixed prefix length")
    require(width["long_width_rows"] == 2528, "width row count")
    require(width["flowstar_minima_all_gt_0p0086"] is True, "Flow* minimum decision")
    require(all(row["width"] > 0.0086 and row["hi"] - row["lo"] == row["width"] for row in width["flowstar_minima"]), "raw Flow* minima")
    require(width["native_and_fixed_schedule_ratios_mixed"] is False, "native/fixed schedule mix")
    for key, expected in EXPECTED_RANGES.items():
        require(width[key] == expected, f"legacy excess changed: {key}")
    with (package / "01_width_ledger/width_ledger.csv").open(newline="", encoding="utf-8") as handle:
        ledger_rows = list(csv.DictReader(handle))
    require(len(ledger_rows) == 2528, "machine-readable width ledger length")
    require({row["channel"] for row in ledger_rows} == {"endpoint_x", "endpoint_y", "segment_tube_x", "segment_tube_y"}, "ledger channels")

    oracle = load(package / "02_contract_oracles/independent_oracle.json")
    require(oracle["candidate"] == CANDIDATE and oracle["passed"] is True, "independent oracle decision")
    require(oracle["oracle_count"] == 13 and len(oracle["rows"]) == 13, "oracle coverage")
    require(all(row["status"] == "PASS" for row in oracle["rows"]), "oracle failure")

    consumer = load(package / "03_consumer_audit/consumer_audit.json")
    require(consumer["first_causally_active_field"] == "affine_source_coefficient_in_next_dense_picard_input", "first consumer field")
    require(consumer["all_complete_ledgers_contain_image"] is True, "ledger containment")
    require(consumer["all_payload_tampers_changed_consumer"] is True, "payload consumer tamper")
    require(consumer["all_metadata_tampers_preserved_consumer"] is True, "metadata consumer tamper")
    require(consumer["flowstar_operator_status"] == "UNAVAILABLE_LOSSLESS_FLOWSTAR_STATE_NOT_SERIALIZED", "lossy Flow* adapter used")
    require(len(consumer["rows"]) == 4, "consumer window coverage")

    terminal = load(package / "03_consumer_audit/terminal_frozen_audit.json")
    require(terminal["payload_tamper_changed_actual_consumer"] is True, "terminal payload tamper")
    require(terminal["metadata_tamper_preserved_actual_consumer"] is True, "terminal metadata tamper")
    require(terminal["rows"]["candidate_source_payload"]["status"] == "failed", "terminal candidate unexpectedly accepted")
    require(terminal["rows"]["ordinary_only_same_affine_source_set"]["subset_margin"][0][1] < terminal["rows"]["candidate_source_payload"]["subset_margin"][0][1] < 0, "terminal intervention ordering")

    summary = load(package / "04_causal_runs/scientific_summary.json")
    require(summary["conclusion"] == CONCLUSION, "scientific conclusion")
    require(summary["fixed_candidate_completed_T6p32"] is True, "fixed T6.32 incomplete")
    require(summary["ratio_crossing_time_shifted_at_0p01_resolution"] is False, "ratio crossing claim")
    reductions = [
        row["candidate_width_reduction_vs_legacy"]
        for checkpoint in summary["checkpoints"].values()
        for row in checkpoint.values()
    ]
    require(len(reductions) == 12 and all(value > 0 for value in reductions), "checkpoint reduction claim")
    require(summary["native_candidate_terminal_time"] == 6.382737816137232, "candidate terminal")
    require(summary["native_legacy_terminal_time"] == 6.397083942944808, "legacy terminal")
    require(summary["native_candidate_terminal_time"] < summary["native_legacy_terminal_time"], "terminal remains open")
    require(summary["source_policy"] == {"collapse_count_at_fixed_T6p32": 631, "fallback_count": 0, "fixed_boundary_variables": 4, "generations": 1, "live_source_count": 2}, "frozen source policy")

    performance = load(package / "05_performance/kernel_and_b1/performance.json")
    require([row["batch"] for row in performance["comparisons"]] == [1, 8, 64, 256, 512], "performance batches")
    require(all(row["cuda_over_cpu_kernel_speedup"] < 1 and row["full_solver_speedup_claimed"] is False for row in performance["comparisons"]), "GPU speedup claim")
    require(summary["performance"]["cuda_over_cpu_full_solver_speedup"] < 1, "full CUDA runtime claim")
    require(summary["performance"]["cuda_full_solver_speedup_claimed"] is False, "full solver speedup flag")

    report = (package / "00_provenance/source_snapshot/docs/VDP_T1_T3_WIDTH_CAUSAL_REPORT_20260814.md").read_text(encoding="utf-8")
    require(CONCLUSION in report and "UNAVAILABLE" in report and "Sampling-only" in report, "report claim scope")
    tests = verify_tests(package)
    result = {
        "schema": "vdp_t1_t3_bounded_source_verification_v1",
        "status": "PASS",
        "conclusion": CONCLUSION,
        "checksums": checksum,
        "tests": tests,
        "oracle_count": 13,
        "checkpoint_reductions_positive": len(reductions),
        "flowstar_minimum": min(row["width"] for row in width["flowstar_minima"]),
    }
    return result


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
