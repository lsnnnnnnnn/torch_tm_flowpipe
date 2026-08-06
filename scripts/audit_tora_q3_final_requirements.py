#!/usr/bin/env python3
"""Fail-closed final checklist against goal_vdp_terminal.md deliverables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "TORA_Q3_NATIVE_TORCH_IMPLEMENTATION_REPORT.md",
    "TORA_Q3_COMMON_CONTROL_COMPARISON_REPORT.md",
    "TORA_Q3_FULL_CLOSED_LOOP_COMPARISON_REPORT.md",
    "TORA_Q3_RUNTIME_REPORT.md",
    "SINE_TM_SOUNDNESS_REPORT.md",
    "PUBLIC_ARTIFACT_GOVERNANCE_AUDIT.md",
    "handoff.md",
    "outputs/tora_q3_native_matched_20260806/provenance/start_state.json",
    "outputs/tora_q3_native_matched_20260806/provenance/final_state.json",
    "outputs/tora_q3_native_matched_20260806/provenance/secret_scan_summary.json",
    "outputs/tora_q3_native_matched_20260806/contract/tora_workload_contract.json",
    "outputs/tora_q3_native_matched_20260806/contract/q3_basis_contract.json",
    "outputs/tora_q3_native_matched_20260806/contract/controller_contract.json",
    "outputs/tora_q3_native_matched_20260806/sine_tm/unit_cases.json",
    "outputs/tora_q3_native_matched_20260806/sine_tm/tora_domain_cases.json",
    "outputs/tora_q3_native_matched_20260806/q3_backend/summary.json",
    "outputs/tora_q3_native_matched_20260806/plant_one_step/summary.json",
    "outputs/tora_q3_native_matched_20260806/common_control_replay/gates.json",
    "outputs/tora_q3_native_matched_20260806/full_closed_loop/summary.json",
    "outputs/tora_q3_native_matched_20260806/comparison/summary.json",
    "outputs/tora_q3_native_matched_20260806/comparison/property_margin_over_time.csv",
    "outputs/tora_q3_native_matched_20260806/comparison/selected_leaf_overlays.csv",
    "outputs/tora_q3_native_matched_20260806/comparison/target_horizon_ratios.csv",
    "outputs/tora_q3_native_matched_20260806/comparison/root_cause_classification.json",
    "outputs/tora_q3_native_matched_20260806/runtime/summary.json",
    "outputs/tora_q3_native_matched_20260806/tests/summary.json",
    "outputs/tora_q3_native_matched_20260806/manifest.sha256",
)


def load(root: Path, relative: str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    checks = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({
            "requirement": name,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        })

    check("required_public_files", not missing, {"missing": missing})
    if not missing:
        basis = load(root, "outputs/tora_q3_native_matched_20260806/contract/q3_basis_contract.json")
        sine_unit = load(root, "outputs/tora_q3_native_matched_20260806/sine_tm/unit_cases.json")
        backend = load(root, "outputs/tora_q3_native_matched_20260806/q3_backend/summary.json")
        plant = load(root, "outputs/tora_q3_native_matched_20260806/common_control_replay/gates.json")
        comparison = load(root, "outputs/tora_q3_native_matched_20260806/comparison/summary.json")
        full = load(root, "outputs/tora_q3_native_matched_20260806/full_closed_loop/summary.json")
        runtime = load(root, "outputs/tora_q3_native_matched_20260806/runtime/summary.json")
        tests = load(root, "outputs/tora_q3_native_matched_20260806/tests/summary.json")
        scan = load(root, "outputs/tora_q3_native_matched_20260806/provenance/secret_scan_summary.json")
        check("q3_basis_84_identity", basis.get("slot_count") == 84 and basis.get("xiangru_to_torch_slot_permutation") == list(range(84)), basis.get("torch_basis_fingerprint"))
        check("sine_soundness_cases", len(sine_unit.get("cases", [])) >= 20 and all(row.get("contains_grid") for row in sine_unit.get("cases", [])) and sine_unit.get("wide_domain_fail_closed") is True, len(sine_unit.get("cases", [])))
        check("six_variable_b48_cuda_backend", backend.get("term_count") == 84 and backend.get("batch") == 48 and backend.get("variable_dimension") == 6, backend.get("device"))
        check("common_control_all_hierarchical_gates", all(row.get("status") == "PASS" for row in plant.get("gates", [])), plant.get("gates"))
        check("common_control_formal_t20_alignment", comparison.get("status") == "FORMALLY_ALIGNED" and comparison.get("compared_segments") == 200, comparison.get("horizons"))
        check("native_full_loop_fail_closed_classified", full.get("status") == "FAILED_AT_T4_4" and full.get("certified_horizon") == 4.3, full.get("first_failure"))
        check("five_repeat_runtime", runtime.get("torch_common_control", {}).get("measured_repeat_count") == 5 and runtime.get("xiangru_common_control", {}).get("measured_repeat_count") == 5, runtime.get("comparison"))
        check("quality_gates", tests.get("status") == "PASS", tests.get("commands"))
        check("no_new_sensitive_binary", scan.get("new_untracked_sensitive_binary_count") == 0, scan)
        check("no_new_public_path_or_secret_pattern", scan.get("new_deliverable_path_or_secret_pattern_match_count") == 0, scan)
        check("public_authorization_blocker_respected", scan.get("governance_status") == "BLOCKED_UNKNOWN_AUTHORIZATION", scan.get("governance_status"))
    status = "PASS" if checks and all(row["status"] == "PASS" for row in checks) else "FAIL"
    result = {
        "schema": "tora_q3_final_requirement_audit_v1",
        "status": status,
        "checks": checks,
        "allowed_final_cases": [
            "Case B: common-control plant passes; native full closed loop does not reach T20",
            "Case D: historical authorization remains unknown; only the separately initialized clean review lineage may be pushed",
        ],
        "vdp_t_6_397_issue": "UNRESOLVED_AND_OUT_OF_SCOPE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "check_count": len(checks)}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
