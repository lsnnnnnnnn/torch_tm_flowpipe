"""Recompute the finite claims in a resident-TM-block evidence package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from experiments.live_range_solver.verify import load_run, verify_run
from experiments.resident_tm_block.analyze import compare_online


def load(path: Path):
    with path.open() as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(root: Path) -> int:
    manifest = root / "SHA256SUMS"
    checked = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise AssertionError(f"manifest path escapes package: {relative}")
        if sha256(path) != expected:
            raise AssertionError(f"hash mismatch: {relative}")
        checked += 1
    if checked == 0:
        raise AssertionError("empty SHA256SUMS")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()

    hash_count = verify_hashes(root)
    local = load(root / "local_exact_checks.json")
    online = load(root / "online_state_comparison.json")
    performance = load(root / "PERFORMANCE_RESULT.json")
    flowstar = load(root / "MATCHED_FLOWSTAR_RESULT.json")
    horizon = load(root / "full_horizon_result.json")
    result = load(root / "RESULT.json")
    source_map = load(root / "SOURCE_MAP.json")
    independent_root = root / "raw_minimal/independent_clone/fresh_b2x2"
    independent = load(independent_root / "RESULT.json")
    independent_jobs = load(independent_root / "jobs.json")
    clone_receipt = load(root / "raw_minimal/independent_clone/CLONE_RECEIPT.json")

    assert local["passed"] is True
    assert local["totals"] == {
        "coefficient_error_interval_checks": 2752,
        "point_coefficient_checks": 2752,
        "remainder_interval_checks": 128,
    }
    assert local["retained_roundoff_witness"]["legacy_contains_exact_error"] is False
    assert local["retained_roundoff_witness"]["resident_contains_exact_error"] is True
    assert local["failure_policy"] == {
        "hidden_cpu_recompute": False,
        "nonfinite_input": "nonfinite",
        "wrong_fingerprint": "unsupported_structure",
    }

    assert online["passed"] is True and online["width_over_1p10"] == 0
    assert online["same_device_width_rows"] == 8416
    assert online["reused_parent_cpu_width_rows"] == 8368
    assert online["next_step_consumes_current_candidate_output"] is True
    assert online["saved_answers_used_to_advance"] is False
    for comparison in online["comparisons"]:
        assert not comparison["decision_mismatches"]
        assert not comparison["ordered_support_mismatches"]
        assert not comparison["ledger_category_mismatches"]

    with (root / "paired_speedups.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 10
    for plant in ("van_der_pol", "brusselator"):
        rows = [row for row in pairs if row["plant"] == plant]
        assert [int(row["pair"]) for row in rows] == list(range(5))
        ratios = [float(row["Gp_wall_s"]) / float(row["Gr_wall_s"]) for row in rows]
        assert all(row["Gr_faster_than_Gp"] == "True" for row in rows)
        decision = performance["decisions"][plant]
        assert ratios == decision["Gp_over_Gr"]
        assert statistics.median(ratios) == decision["median_Gp_over_Gr"]
        assert decision["Gr_faster_than_Gp_wins"] == 5
        assert decision["end_to_end_target_met"] is False
        assert decision["stable_strict_cpu_regression"] is False
    assert performance["end_to_end_effect"] == "small_gain"
    assert performance["engineering_target_met"] is False

    with (root / "device_residency_and_calls.csv").open(newline="") as stream:
        device_rows = list(csv.DictReader(stream))
    resident_rows = [row for row in device_rows if row["route"] == "Gr"]
    assert len(resident_rows) == 10
    for row in resident_rows:
        assert int(row["successful_lane_steps"]) == 640
        assert int(row["resident_requests"]) == 640
        assert int(row["resident_lane_outputs"]) == 1280
        assert int(row["resident_kernel_invocations"]) > 0
        assert int(row["resident_h2d_copy_operations"]) > 0
        assert int(row["resident_d2h_copy_operations"]) > 0
        assert int(row["resident_structure_fallbacks"]) == 0
        assert int(row["hardware_fallback_requests"]) == 0

    assert flowstar["passed_complete_workload"] is True
    assert flowstar["matched_flowstar_gap"] == "measured"
    assert flowstar["exact_history_contract_supported"] is False
    for row in flowstar["rows"]:
        assert row["tasks"] == 32 and row["steps_per_task"] == 20
        assert row["successful_lane_steps"] == 640
        assert row["flowstar_native_status"] == 2
        assert row["candidate_wall_over_flowstar_wall"] > 1

    assert horizon["status"] == "NOT_RUN_PERFORMANCE_NOT_USEFUL"
    assert horizon["full_horizon_new_candidate"] == "not_run"
    assert horizon["historical_parent_full_horizon_relabelled_as_new"] is False

    expected_truth = {
        "block_numerical_contract": "pass",
        "actual_device_residency": "measured",
        "online_integration": "pass",
        "local_speed_effect": "measured",
        "end_to_end_effect": "small_gain",
        "full_horizon_new_candidate": "not_run",
        "matched_flowstar_gap": "measured",
        "entire_solver_formal_proof": False,
        "full_gpu_engine": False,
    }
    assert result["truth_status"] == expected_truth
    assert result["default_enabled"] is False
    assert result["engineering_target_met"] is False
    assert result["independent_clone_verification"] == "pass"
    assert source_map["actual_scientific_runtime_commit"] == performance["source_sha"]

    assert independent["passed"] is True
    assert independent["fresh_runs"] == 4
    assert independent["successful_lane_steps_total"] == 16
    assert independent["resident_requests_total"] == 8
    assert independent["complete_performance_matrix_reexecuted"] is False
    assert independent["full_1000_step_runs_reexecuted"] == 0
    assert independent["saved_answers_used_to_advance"] is False
    assert clone_receipt["passed"] is True
    assert clone_receipt["independent_execution_commit"] == independent["clone_head"]
    assert source_map["independent_clone"]["execution_commit"] == independent["clone_head"]
    assert clone_receipt["package_verification_before_smoke"]["passed"] is True
    assert clone_receipt["package_verification_before_smoke"]["manifest_files_checked"] == 105
    assert len(independent_jobs) == 4
    assert [job["name"] for job in independent_jobs] == [
        "van_der_pol-Gp",
        "van_der_pol-Gr",
        "brusselator-Gp",
        "brusselator-Gr",
    ]
    assert all(job["returncode"] == 0 for job in independent_jobs)

    independent_runs = {}
    for plant in ("van_der_pol", "brusselator"):
        for route in ("Gp", "Gr"):
            name = f"{plant}-{route}"
            run, events = load_run(independent_root / name)
            verify_run(run, events, recompute=True)
            assert run["source_sha"] == independent["clone_head"]
            assert run["ids"] == [0, 31]
            assert run["steps"] == 2
            assert run["successful_lane_steps"] == 4
            if route == "Gr":
                assert run["counts"]["resident_requests"] == 4
                assert run["counts"].get("resident_structure_fallbacks", 0) == 0
            independent_runs[name] = run

    independent_comparisons = []
    for plant in ("van_der_pol", "brusselator"):
        comparison, _ = compare_online(
            independent_runs[f"{plant}-Gp"],
            independent_runs[f"{plant}-Gr"],
            "INDEPENDENT_CLONE_B2X2",
        )
        assert not comparison["decision_mismatches"]
        assert not comparison["ordered_support_mismatches"]
        assert not comparison["ledger_category_mismatches"]
        assert comparison["width_over_1p10"] == 0
        independent_comparisons.append(comparison)
    assert independent_comparisons == independent["comparisons"]

    suites = ET.parse(root / "tests/affected.xml").getroot()
    suite = suites.find("testsuite")
    assert suite is not None
    assert int(suite.attrib["tests"]) == 269
    assert int(suite.attrib["errors"]) == 0
    assert int(suite.attrib["failures"]) == 0
    assert int(suite.attrib["skipped"]) == 0

    receipt = {
        "schema": "resident-tm-block-package-verification-v1",
        "passed": True,
        "manifest_files_checked": hash_count,
        "affected_tests": 269,
        "formal_pairs_recomputed": len(pairs),
        "formal_resident_runs_checked": len(resident_rows),
        "online_comparisons_checked": len(online["comparisons"]),
        "flowstar_workloads_checked": len(flowstar["rows"]),
        "independent_clone_runs_recomputed": len(independent_runs),
        "independent_clone_comparisons_recomputed": len(independent_comparisons),
        "full_1000_step_runs_reexecuted": 0,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
