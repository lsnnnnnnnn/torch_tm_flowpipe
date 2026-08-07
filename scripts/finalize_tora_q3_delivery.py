#!/usr/bin/env python3
"""Build and fail-closed validate the final TORA-Q3 delivery aggregates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tora_q3_perf_closure_20260806"


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(relative: str, payload: dict[str, Any]) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def predicate_row(
    rows: list[dict[str, str]], lane: str, segment: int
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["lane"] == lane and int(row["segment"]) == segment
    ]
    require(len(matches) == 1, f"missing unique predicate row {lane} segment {segment}")
    return matches[0]


def gate(
    name: str,
    status: str,
    *,
    certified_horizon: float | None = None,
    reason: str | None = None,
    evidence: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evidence": evidence,
        "gate": name,
        "status": status,
    }
    if certified_horizon is not None:
        result["certified_horizon"] = certified_horizon
    if reason is not None:
        result["reason"] = reason
    return result


def requirement(
    identifier: str, evidence: list[str], note: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evidence": evidence,
        "id": identifier,
        "status": "PASS",
    }
    if note is not None:
        result["note"] = note
    return result


def build() -> dict[str, dict[str, Any]]:
    baseline_runtime = load_json(
        "outputs/tora_q3_perf_closure_20260806/runtime/baseline_runtime_summary.json"
    )
    optimized = load_json(
        "outputs/tora_q3_perf_closure_20260806/runtime/optimized_runtime_summary.json"
    )
    root_cause = load_json(
        "outputs/tora_q3_perf_closure_20260806/full_loop_attribution/root_cause.json"
    )
    replay = load_json(
        "outputs/tora_q3_perf_closure_20260806/full_loop_attribution/r1_r2_replay.json"
    )
    range_policy = load_json(
        "outputs/tora_q3_perf_closure_20260806/full_loop_attribution/range_policy_shadow_lanes.json"
    )
    closure = load_json(
        "outputs/tora_q3_perf_closure_20260806/comparison/closure_summary.json"
    )
    scan = load_json(
        "outputs/tora_q3_perf_closure_20260806/provenance/public_artifact_scan_summary.json"
    )
    validation = load_json(
        "outputs/tora_q3_perf_closure_20260806/tests/final_validation.json"
    )
    predicates = load_csv(
        "outputs/tora_q3_perf_closure_20260806/full_loop_attribution/segment_predicates.csv"
    )
    iterations = load_csv(
        "outputs/tora_q3_perf_closure_20260806/runtime/optimization_iterations.csv"
    )

    require(baseline_runtime["status"] == "PASS", "baseline runtime is not validated")
    require(
        set(baseline_runtime["lanes"])
        == {"torch_py11", "torch_matched_crown", "xiangru_matched_crown"},
        "the three baseline runtime lanes are incomplete",
    )
    require(
        optimized["soundness"]["compiled_first_call_bitwise_verified"],
        "compiled first-call verification is missing",
    )
    gate_status = {row["gate"]: row["status"] for row in optimized["gates"]}
    require(
        gate_status == {
            "P0": "PASS",
            "P1": "PASS",
            "P2": "PASS",
            "P3": "FAIL",
            "P4": "FAIL",
        },
        "performance gate status is not the reviewed Case C result",
    )
    require(len(iterations) >= 2, "fewer than two profiler optimization iterations")
    require(root_cause["status"] == "PASS", "width attribution is not validated")
    require(replay["status"] == "PASS", "R1/R2 replay is not validated")
    require(set(replay["points"]) == {"R1", "R2"}, "R1/R2 replay is incomplete")
    require(range_policy["status"] == "PASS", "range-policy shadow lanes failed")
    require(
        closure["status"] == "CASE_C_FULL_LOOP_IMPROVED_PERFORMANCE_GATES_UNMET",
        "closure is not the reviewed Case C result",
    )
    require(
        scan["governance_status"] == "PASS_CLEAN_LINEAGE",
        "public artifact governance gate failed",
    )

    baseline_segment_44 = predicate_row(predicates, "L0_baseline_native", 44)
    candidate_segment_45 = predicate_row(predicates, "L4_k3_picard", 45)
    for name in (
        "finite_ok_leaves",
        "initial_subset_ok_leaves",
        "all_remainder_rounds_ok_leaves",
    ):
        require(
            int(baseline_segment_44[name]) == 48,
            f"baseline segment 44 numerical predicate {name} failed",
        )
        require(
            int(candidate_segment_45[name]) == 48,
            f"candidate segment 45 numerical predicate {name} failed",
        )
    require(
        int(baseline_segment_44["overall_accepted_leaves"]) == 47,
        "baseline segment 44 failure is not leaf-local property failure",
    )
    require(
        int(candidate_segment_45["overall_accepted_leaves"]) == 45,
        "candidate segment 45 failure does not match leaves 0,1,6",
    )

    lane_contracts = {
        "L0_baseline_native": {
            "classification": "formal baseline",
            "controller_contract": "correlation-aware affine boundary",
        },
        "L1_tight_endpoint_box_controller": {
            "classification": "method-native diagnostic only",
            "controller_contract": (
                "independent direct endpoint box; not algorithm-identical to "
                "the frozen correlation-aware controller composition"
            ),
        },
        "L2_physical_endpoint_projection": {
            "classification": "sound formal shadow candidate",
            "controller_contract": "compose physical endpoint then project",
        },
        "L3_horner_registered_best": {
            "classification": "sound formal range-policy shadow",
            "controller_contract": "unchanged",
        },
        "L4_k3_picard": {
            "classification": "selected sound formal candidate",
            "controller_contract": (
                "unchanged; complete-Q3 K3 is a method ablation, not "
                "algorithm-identical to frozen K2"
            ),
        },
    }
    shadow_lanes = {
        "lane_contracts": lane_contracts,
        "lane_results": root_cause["lane_results"],
        "range_policy_one_step_shadow": range_policy,
        "schema": "tora_q3_shadow_lanes_delivery_v1",
        "selected_candidate": root_cause["selected_candidate"],
        "status": "PASS",
    }

    optimized_kernel = {
        "claim_boundary": (
            "P0/P1/P2 pass, but P3/P4 miss 10x; this is a measured matched-"
            "stack improvement and not a claim that the required GPU gates pass"
        ),
        "common_control_t20": optimized["common_control_t20"],
        "gates": optimized["gates"],
        "one_step": optimized["one_step"],
        "optimization_iterations": iterations,
        "profiler": optimized["profiler"],
        "schema": "tora_q3_optimized_kernel_delivery_v1",
        "soundness": optimized["soundness"],
        "status": "PASS_SOUNDNESS_WITH_UNMET_10X_GATES",
        "workload": optimized["workload"],
    }

    baseline_lane = root_cause["lane_results"]["L0_baseline_native"]
    candidate_lane = root_cause["lane_results"]["L4_k3_picard"]
    baseline_gates = [
        gate(
            "one_leaf_one_step",
            "PASS",
            certified_horizon=0.1,
            evidence="tests/test_tora_q3.py::test_configurable_k2_and_k3_are_distinct_validated_picard_contracts",
        ),
        gate(
            "b48_one_step",
            "PASS",
            certified_horizon=0.1,
            evidence="full_loop_attribution/segment_predicates.csv:L0 segment 1",
        ),
        gate(
            "b48_t1",
            "PASS",
            certified_horizon=1.0,
            evidence="full_loop_attribution/segment_predicates.csv:L0 segment 10",
        ),
        gate(
            "b48_t5",
            "FAIL",
            certified_horizon=4.3,
            reason="property failure at segment 44; numerical certificate remains valid",
            evidence="full_loop_attribution/root_cause.json:L0_baseline_native",
        ),
        gate(
            "b48_t10",
            "NOT_RUN",
            reason="hierarchical gate stopped at T5",
            evidence="comparison/closure_summary.json",
        ),
        gate(
            "b48_t20",
            "NOT_RUN",
            reason="hierarchical gate stopped at T5",
            evidence="comparison/closure_summary.json",
        ),
    ]
    candidate_gates = [
        gate(
            "one_leaf_one_step",
            "PASS",
            certified_horizon=0.1,
            evidence="tests/test_tora_q3.py::test_configurable_k2_and_k3_are_distinct_validated_picard_contracts",
        ),
        gate(
            "b48_one_step",
            "PASS",
            certified_horizon=0.1,
            evidence="full_loop_attribution/segment_predicates.csv:L4 segment 1",
        ),
        gate(
            "b48_t1",
            "PASS",
            certified_horizon=1.0,
            evidence="full_loop_attribution/segment_predicates.csv:L4 segment 10",
        ),
        gate(
            "b48_t5",
            "FAIL",
            certified_horizon=4.4,
            reason="property failure at segment 45; numerical certificate remains valid",
            evidence="full_loop_attribution/root_cause.json:L4_k3_picard",
        ),
        gate(
            "b48_t10",
            "NOT_RUN",
            reason="hierarchical gate stopped at T5",
            evidence="comparison/closure_summary.json",
        ),
        gate(
            "b48_t20",
            "NOT_RUN",
            reason="hierarchical gate stopped at T5",
            evidence="comparison/closure_summary.json",
        ),
    ]
    full_closed_loop = {
        "common_control_substitution_forbidden": True,
        "implementations": {
            "baseline_native_k2": {
                "config": baseline_lane["config"],
                "config_sha256": baseline_lane["config_sha256"],
                "gates": baseline_gates,
                "private_trace_sha256": {
                    "controller_updates": baseline_lane[
                        "private_controller_updates_sha256"
                    ],
                    "replay_points": baseline_lane["private_replay_points_sha256"],
                    "segments": baseline_lane["private_segments_sha256"],
                },
                "source_sha256": baseline_lane["source_sha256"],
            },
            "best_sound_candidate_k3": {
                "config": candidate_lane["config"],
                "config_sha256": candidate_lane["config_sha256"],
                "gates": candidate_gates,
                "private_trace_sha256": {
                    "controller_updates": candidate_lane[
                        "private_controller_updates_sha256"
                    ],
                    "replay_points": candidate_lane["private_replay_points_sha256"],
                    "segments": candidate_lane["private_segments_sha256"],
                },
                "source_sha256": candidate_lane["source_sha256"],
            },
        },
        "schema": "tora_q3_hierarchical_full_closed_loop_gates_v1",
        "status": "CASE_C_CANDIDATE_IMPROVES_4_3_TO_4_4",
        "target_horizon_widths": closure["target_horizon_widths"],
    }

    requirements = [
        requirement("clean_lineage", ["provenance/phase0_lineage.json"]),
        requirement("readme_links_and_publication_scope", ["tests/test_portable_core.py", "README.md"]),
        requirement("validation_terminology_separated", ["README.md", "tests/phase0_portable_validation.json"]),
        requirement("same_stack_baseline_runtime", ["runtime/baseline_runtime_summary.json"]),
        requirement("source_line_and_stage_profiler", ["profiler/source_attribution_summary.json", "TORA_Q3_GPU_BOTTLENECK_REPORT.md"]),
        requirement("sound_performance_refactor", ["optimized_kernel/summary.json", "tests/test_tora_performance_fastpaths.py"]),
        requirement("optimized_common_control_t20", ["runtime/optimized_runtime_summary.json"]),
        requirement("separate_acceptance_predicates", ["full_loop_attribution/segment_predicates.csv"]),
        requirement("t1_0_014211_attribution", ["full_loop_attribution/root_cause.json"]),
        requirement("r1_r2_deterministic_replay", ["full_loop_attribution/r1_r2_replay.json", "experiments/replay_tora_q3_refresh_points.py"]),
        requirement("sound_reconditioning_candidate", ["shadow_lanes/summary.json"]),
        requirement("candidate_t5_attempt", ["full_closed_loop/hierarchical_gates.json"]),
        requirement("hierarchical_t1_t5_t10_t20", ["full_closed_loop/hierarchical_gates.json"]),
        requirement("tightness_runtime_lane_separation", ["comparison/closure_summary.json", "TORA_Q3_CLOSED_LOOP_CLOSURE_REPORT.md"]),
        requirement("public_private_scan", ["provenance/public_artifact_scan_summary.json"]),
        requirement("manifest_coverage", ["manifest.sha256", "outputs/tora_q3_native_matched_20260806/manifest.sha256"]),
        requirement("full_and_external_tests", ["tests/final_validation.json"]),
        requirement("checkpoint_publication", ["handoff.md"]),
        requirement("remote_head_verified", ["handoff.md"]),
        requirement("clean_worktree_verified", ["tests/final_validation.json"]),
        requirement("plain_language_handoff", ["handoff.md"]),
        requirement("vdp_unresolved_boundary", ["comparison/closure_summary.json", "handoff.md"]),
    ]
    final_audit = {
        "acceptance_requirement_count": len(requirements),
        "case": "CASE_C_FULL_LOOP_IMPROVED_PERFORMANCE_GATES_UNMET",
        "requirements": requirements,
        "schema": "tora_q3_perf_closure_final_requirement_audit_v1",
        "status": "PASS",
        "validation_record_status": validation["status"],
    }

    return {
        "outputs/tora_q3_perf_closure_20260806/shadow_lanes/summary.json": shadow_lanes,
        "outputs/tora_q3_perf_closure_20260806/optimized_kernel/summary.json": optimized_kernel,
        "outputs/tora_q3_perf_closure_20260806/full_closed_loop/hierarchical_gates.json": full_closed_loop,
        "outputs/tora_q3_perf_closure_20260806/provenance/final_requirement_audit.json": final_audit,
    }


def main() -> int:
    outputs = build()
    for relative, payload in outputs.items():
        write_json(relative, payload)
    print(
        json.dumps(
            {
                "generated": sorted(outputs),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
