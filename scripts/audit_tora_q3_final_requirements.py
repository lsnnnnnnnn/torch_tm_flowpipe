#!/usr/bin/env python3
"""Fail closed on the final TORA-Q3 stage-parity delivery contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


OUTPUT = Path("outputs/tora_q3_stage_parity_fused_20260809")
REQUIRED_FILES = (
    "TORA_Q3_STAGE_PARITY_ROOT_CAUSE_REPORT.md",
    "TORA_Q3_ALGORITHM_ALIGNED_IMPLEMENTATION_REPORT.md",
    "TORA_Q3_FUSED_KERNEL_RUNTIME_REPORT.md",
    "TORA_Q3_NATIVE_T20_CLOSURE_REPORT.md",
    "handoff.md",
    "experiments/observe_tora_q3_stage_contract.py",
    "experiments/run_tora_q3_algorithm_aligned.py",
    "experiments/benchmark_tora_q3_fused_kernel.py",
    "experiments/run_tora_q3_native_hierarchical.py",
    "scripts/compare_tora_q3_stage_contract.py",
    "scripts/summarize_tora_q3_stage_parity.py",
    "scripts/summarize_tora_q3_fused_runtime.py",
    "scripts/summarize_tora_q3_native_closure.py",
    "tests/test_tora_stage_contract.py",
    "tests/test_tora_stage_comparator.py",
    "tests/test_tora_algorithm_aligned_q3.py",
    "tests/test_tora_fused_kernel.py",
    "tests/test_tora_full_loop_hierarchical.py",
    "tests/test_tora_runtime_protocol.py",
    "tests/test_tora_public_artifact_scan.py",
    f"{OUTPUT}/stage_parity/stage_first_divergence.csv",
    f"{OUTPUT}/stage_parity/r1_r2_center_radius_remainder.csv",
    f"{OUTPUT}/algorithm_aligned/one_step_gates.json",
    f"{OUTPUT}/fused_kernel/operator_telemetry.csv",
    f"{OUTPUT}/fused_kernel/t20_runtime_repeats.csv",
    f"{OUTPUT}/native_full_loop/hierarchical_gates.json",
    f"{OUTPUT}/native_full_loop/failure_horizons.csv",
    f"{OUTPUT}/native_full_loop/endpoint_width_over_time.csv",
    f"{OUTPUT}/native_full_loop/tube_width_over_time.csv",
    f"{OUTPUT}/native_full_loop/property_margin_over_time.csv",
    f"{OUTPUT}/figures/native_width_remainder_growth.svg",
    f"{OUTPUT}/figures/native_runtime_stage_breakdown.svg",
    f"{OUTPUT}/manifest.sha256",
)
REQUIRED_DIRECTORIES = (
    "provenance",
    "baseline",
    "stage_contract",
    "stage_parity",
    "algorithm_aligned",
    "fused_kernel",
    "common_control",
    "native_full_loop",
    "comparison",
    "tests",
    "figures",
)


def load(root: Path, relative: str | Path) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def git_lines(root: Path, *arguments: str) -> list[str]:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    ).stdout.splitlines()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    checks: list[dict[str, Any]] = []

    def check(requirement: str, passed: bool, evidence: object) -> None:
        checks.append(
            {
                "requirement": requirement,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
        )

    missing_files = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    missing_directories = [
        path for path in REQUIRED_DIRECTORIES if not (root / OUTPUT / path).is_dir()
    ]
    check("required_delivery_files", not missing_files, {"missing": missing_files})
    check(
        "required_output_directories",
        not missing_directories,
        {"missing": missing_directories},
    )

    phase0 = load(root, OUTPUT / "provenance/phase0.json")
    check(
        "clean_lineage_from_63efe66",
        phase0["lineage"]["start_commit"]
        == "63efe66cfe7bdda907f8255ba23cebaa9b878233"
        and phase0["lineage"]["blocked_history_merge_base"] is None,
        phase0["lineage"],
    )
    check(
        "frozen_baseline_unchanged",
        phase0["baseline_freeze"]["old_k2_semantics_unchanged"] is True
        and all(
            row["status"] == "PASS"
            for row in phase0["function_contracts"].values()
        ),
        phase0["baseline_freeze"],
    )

    observation = load(root, OUTPUT / "stage_contract/observation_summary.json")
    equivalence = load(
        root, OUTPUT / "stage_contract/instrumentation_equivalence_summary.json"
    )
    stage = load(root, OUTPUT / "stage_contract/stage_comparison_summary.json")
    check(
        "non_invasive_observation",
        observation["observation_only"] is True
        and observation["formal_runner_uses_xiangru_outputs"] is False
        and equivalence["uninstrumented_behavior_equivalence"]["status"]
        == "VERIFIED",
        {
            "status": equivalence["status"],
            "maximum_absolute_difference": equivalence["maximum_absolute_difference"],
        },
    )
    check(
        "complete_stage_contract_s0_s1_r1_r2_f0",
        [row["stage"] for row in stage["stage_table"]]
        == [f"A{index}" for index in range(13)]
        and set(observation["selected_replay_points"]) >= {
            "1",
            "2",
            "10",
            "40",
            "43",
            "44",
            "45",
        },
        {"stage_count": len(stage["stage_table"])},
    )

    root_cause = load(root, OUTPUT / "stage_parity/root_cause.json")
    check(
        "stage_level_t1_and_segment40_root_cause",
        root_cause["first_differences"]["first_material"]["stage"] == "A3"
        and root_cause["segment_40_remainder_attribution"][
            "dominant_accumulated_ledger_category"
        ]
        == "composition_overflow",
        {
            "t1": root_cause["t1_0_014211_attribution"],
            "segment40": root_cause["segment_40_remainder_attribution"],
        },
    )

    aligned = load(root, OUTPUT / "algorithm_aligned/summary.json")
    check(
        "new_sound_algorithm_aligned_lane",
        aligned["status"] == "PASS"
        and aligned["one_step"]["status"] == "PASS"
        and aligned["common_control"]["status"] == "PASS",
        {"lane": aligned["lane"], "device": aligned["device"]},
    )

    fused = load(root, OUTPUT / "fused_kernel/summary.json")
    check(
        "larger_fixed_shape_tensor_boundary",
        fused["soundness"]["status"] == "PASS"
        and fused["gates"]["P1_graph_breaks"]["deployed_fullgraph_stage_count"]
        == 4,
        fused["compilation"],
    )
    check(
        "formal_fused_runtime_protocol",
        fused["one_step"]["runtime"]["repeat_count"] == 10
        and fused["common_control_t20"]["runtime"]["repeat_count"] == 5
        and fused["common_control_t20"]["checksum_stable"] is True,
        {
            "one_step": fused["one_step"]["runtime"],
            "t20": fused["common_control_t20"]["runtime"],
        },
    )
    check(
        "performance_gates_p0_p5",
        fused["gates"]["P0_correctness_soundness"] == "PASS"
        and all(
            fused["gates"][name]["status"].startswith("PASS")
            for name in (
                "P1_graph_breaks",
                "P2_program_sync",
                "P3_aten_to",
                "P4_b48_one_step",
                "P5_common_control_t20",
            )
        ),
        fused["gates"],
    )

    hierarchy = load(root, OUTPUT / "native_full_loop/hierarchical_gates.json")
    expected = ["PASS", "PASS", "PASS", "FAIL", "NOT_RUN", "NOT_RUN"]
    check(
        "strict_native_hierarchical_gates",
        hierarchy["strict_previous_pass_only"] is True
        and all(
            [gate["status"] for gate in lane["gates"]] == expected
            for lane in hierarchy["implementations"].values()
        ),
        {
            name: lane["certified_horizon"]
            for name, lane in hierarchy["implementations"].items()
        },
    )
    check(
        "property_failure_not_numerical_failure",
        all(
            lane["first_failure"]["reason"] == "property"
            and lane["numerical_certificate_passed_at_failure"] is True
            for lane in hierarchy["implementations"].values()
        ),
        {
            name: lane["first_failure"]
            for name, lane in hierarchy["implementations"].items()
        },
    )
    check(
        "no_fabricated_t5_t10_t20_widths",
        all(
            values == {"T5": None, "T10": None, "T20": None}
            for values in hierarchy["torch_target_width_availability"].values()
        ),
        hierarchy["torch_target_width_availability"],
    )

    comparison = load(root, OUTPUT / "comparison/summary.json")
    check(
        "runtime_and_tightness_scopes_separated",
        comparison["common_control"]["workload"][
            "independent_native_closed_loop"
        ]
        is False
        and comparison["native_closed_loop"][
            "formal_cross_implementation_t20_runtime_ratio"
        ]
        is None,
        {"status": comparison["status"]},
    )

    quality_path = OUTPUT / "tests/final_validation.json"
    quality = load(root, quality_path) if (root / quality_path).is_file() else {}
    check(
        "portable_external_gpu_quality_gates",
        quality.get("status") == "PASS"
        and quality.get("compileall") == "PASS"
        and quality.get("portable", {}).get("status") == "PASS"
        and quality.get("external_integration", {}).get("status") == "PASS"
        and quality.get("gpu_focused", {}).get("status") == "PASS",
        quality,
    )

    scan_path = OUTPUT / "provenance/checkpoint7_publication_scan.json"
    scan = load(root, scan_path) if (root / scan_path).is_file() else {}
    check(
        "public_private_scan",
        scan.get("governance_status") == "PASS_CLEAN_LINEAGE"
        and scan.get("unallowlisted_path_or_credential_match_count") == 0
        and scan.get("current_tree_sensitive_suffix_candidate_count") == 0,
        {
            "status": scan.get("governance_status"),
            "unallowlisted": scan.get(
                "unallowlisted_path_or_credential_match_count"
            ),
        },
    )

    manifests = (
        root / "outputs/tora_q3_native_matched_20260806/manifest.sha256",
        root / "outputs/tora_q3_perf_closure_20260806/manifest.sha256",
        root / OUTPUT / "manifest.sha256",
    )
    manifest_equal = all(path.is_file() for path in manifests) and len(
        {path.read_bytes() for path in manifests}
    ) == 1
    manifest_paths = set()
    if manifest_equal:
        manifest_paths = {
            line.split("  ", 1)[1]
            for line in manifests[0].read_text(encoding="utf-8").splitlines()
        }
    expected_paths = {
        path
        for path in git_lines(root, "ls-files")
        if Path(path).name != "manifest.sha256"
    }
    check(
        "complete_identical_manifests",
        manifest_equal and manifest_paths == expected_paths,
        {
            "entry_count": len(manifest_paths),
            "expected_count": len(expected_paths),
        },
    )
    check(
        "readme_surface",
        subprocess.run(
            [sys.executable, "scripts/check_readme_surface.py"],
            cwd=root,
            capture_output=True,
        ).returncode
        == 0,
        "scripts/check_readme_surface.py",
    )
    check(
        "vdp_issue_explicitly_unresolved",
        "6.397083942944808" in (root / "handoff.md").read_text(encoding="utf-8")
        and "unresolved" in (root / "handoff.md").read_text(encoding="utf-8").lower(),
        "handoff.md",
    )
    check(
        "handoff_answers_22_questions",
        all(
            f"{index}. **" in (root / "handoff.md").read_text(encoding="utf-8")
            for index in range(1, 23)
        ),
        {"required_answer_count": 22},
    )
    check(
        "case_c_classification",
        hierarchy["status"] == "CASE_C_PERFORMANCE_PASS_NATIVE_T5_GATE_FAIL",
        hierarchy["status"],
    )

    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    result = {
        "schema": "tora_q3_stage_parity_final_requirement_audit_v2",
        "status": status,
        "acceptance_requirement_count": len(checks),
        "requirements": checks,
        "classification": "CASE_C_PERFORMANCE_PASS_NATIVE_T5_GATE_FAIL",
        "vdp_t_6_397_issue": "UNRESOLVED_AND_INDEPENDENT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "check_count": len(checks)}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
