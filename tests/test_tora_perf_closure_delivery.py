from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from experiments.replay_tora_q3_refresh_points import replay
from scripts.finalize_tora_q3_delivery import build


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tora_q3_perf_closure_20260806"


def load_json(relative: str) -> dict[str, object]:
    return json.loads((OUTPUT / relative).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.regression
@pytest.mark.protocol
def test_final_delivery_structure_and_requirement_audit_are_complete() -> None:
    required = (
        "baseline",
        "profiler",
        "full_loop_attribution",
        "shadow_lanes",
        "optimized_kernel",
        "full_closed_loop",
        "runtime",
        "comparison",
        "tests",
        "provenance",
    )
    assert all((OUTPUT / name).is_dir() for name in required)
    assert (OUTPUT / "manifest.sha256").is_file()
    audit = load_json("provenance/final_requirement_audit.json")
    assert audit["status"] == "PASS"
    assert audit["acceptance_requirement_count"] == 22
    assert all(row["status"] == "PASS" for row in audit["requirements"])
    generated = build()
    for relative, payload in generated.items():
        assert json.loads((ROOT / relative).read_text()) == payload


@pytest.mark.regression
@pytest.mark.protocol
def test_r1_r2_public_replay_is_hash_bound_deterministic_and_sanitized(
    tmp_path: Path,
) -> None:
    public = load_json("full_loop_attribution/r1_r2_replay.json")
    gates = load_json("full_closed_loop/hierarchical_gates.json")
    assert public["status"] == "PASS"
    assert set(public["points"]) == {"R1", "R2"}
    assert public["input_private_snapshot_sha256"] == gates["implementations"][
        "baseline_native_k2"
    ]["private_trace_sha256"]["replay_points"]
    assert public["points"]["R1"]["segment_index"] == 10
    assert public["points"]["R1"]["controller_period"] == 2
    assert public["points"]["R2"]["segment_index"] == 40
    assert public["points"]["R2"]["controller_period"] == 5
    for point in public["points"].values():
        counts = point["predicate_true_counts"]
        assert counts["finite_ok_by_leaf"] == 48
        assert counts["initial_subset_ok_by_leaf"] == 48
        assert counts["all_remainder_rounds_ok_by_leaf"] == 48
        assert len(point["point_content_sha256"]) == 64
        assert len(point["controller_refresh"]["content_sha256"]) == 64
    serialized = json.dumps(public)
    assert '"endpoint":' not in serialized
    assert '"tube":' not in serialized
    assert '"leaf_id":' not in serialized

    private_stub = tmp_path / "replay.json"
    private_stub.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot SHA-256 mismatch"):
        replay(private_stub, "0" * 64)


@pytest.mark.regression
@pytest.mark.protocol
def test_hierarchical_full_loop_reproduction_is_source_and_config_bound() -> None:
    summary = load_json("full_closed_loop/hierarchical_gates.json")
    assert summary["status"] == "CASE_C_CANDIDATE_IMPROVES_4_3_TO_4_4"
    expected = {
        "baseline_native_k2": (
            ["PASS", "PASS", "PASS", "FAIL", "NOT_RUN", "NOT_RUN"],
            4.3,
            "baseline_native",
            2,
        ),
        "best_sound_candidate_k3": (
            ["PASS", "PASS", "PASS", "FAIL", "NOT_RUN", "NOT_RUN"],
            4.4,
            "k3_picard",
            3,
        ),
    }
    for name, (statuses, horizon, lane, picard_rounds) in expected.items():
        implementation = summary["implementations"][name]
        gates = implementation["gates"]
        assert [row["gate"] for row in gates] == [
            "one_leaf_one_step",
            "b48_one_step",
            "b48_t1",
            "b48_t5",
            "b48_t10",
            "b48_t20",
        ]
        assert [row["status"] for row in gates] == statuses
        assert gates[3]["certified_horizon"] == horizon
        config = implementation["config"]
        assert config["lane"] == lane
        assert config["polynomial_picard_rounds"] == picard_rounds
        assert config["remainder_picard_rounds"] == 10
        assert config["property"] == "abs(x1..x4) <= 2"
        assert canonical_sha256(config) == implementation["config_sha256"]
        for relative, digest in implementation["source_sha256"].items():
            assert file_sha256(ROOT / relative) == digest
        assert all(
            len(digest) == 64
            for digest in implementation["private_trace_sha256"].values()
        )
    assert summary["target_horizon_widths"] == {
        "T5": "N/A (candidate first fails at segment 45)",
        "T10": "N/A",
        "T20": "N/A",
    }
    assert summary["common_control_substitution_forbidden"]


@pytest.mark.regression
@pytest.mark.protocol
def test_t4_4_and_candidate_failures_are_property_not_certificate_failures() -> None:
    with (OUTPUT / "full_loop_attribution/segment_predicates.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    checks = {
        ("L0_baseline_native", "44"): (47, 47),
        ("L4_k3_picard", "45"): (45, 45),
    }
    for key, (property_count, accepted_count) in checks.items():
        row = next(
            candidate
            for candidate in rows
            if (candidate["lane"], candidate["segment"]) == key
        )
        assert int(row["finite_ok_leaves"]) == 48
        assert int(row["initial_subset_ok_leaves"]) == 48
        assert int(row["all_remainder_rounds_ok_leaves"]) == 48
        assert int(row["composed_property_ok_leaves"]) == property_count
        assert int(row["overall_accepted_leaves"]) == accepted_count


@pytest.mark.regression
@pytest.mark.protocol
def test_optimized_kernel_public_gates_cover_sync_conversion_runtime_and_memory() -> None:
    optimized = load_json("optimized_kernel/summary.json")
    assert optimized["status"] == "PASS_SOUNDNESS_WITH_UNMET_10X_GATES"
    assert len(optimized["optimization_iterations"]) >= 2
    assert optimized["soundness"]["compiled_first_call_bitwise_verified"]
    profiler = optimized["profiler"]
    assert profiler["program_dispatch_host_scalar_sync"] == 3
    assert profiler["aten_to_reduction_percent"] >= 90.0
    assert profiler["compiled_point_kernel_graph_break_count"] == 0
    assert optimized["one_step"]["optimized_eager"]["repeats"] == 10
    assert optimized["one_step"]["optimized_compiled"]["repeats"] == 10
    t20 = optimized["common_control_t20"]
    assert len(t20["repeat_statuses"]) == 5
    assert t20["excluded_warmup_seconds"] > 0.0
    assert t20["stable_status_and_checksum"]
    assert t20["peak_cpu_resident_memory_bytes"] > 0
    assert t20["peak_cuda_memory_bytes"] > 0
    assert {row["gate"]: row["status"] for row in optimized["gates"]} == {
        "P0": "PASS",
        "P1": "PASS",
        "P2": "PASS",
        "P3": "FAIL",
        "P4": "FAIL",
    }


@pytest.mark.regression
@pytest.mark.protocol
def test_both_public_manifests_are_the_same_complete_tree_view() -> None:
    performance_manifest = OUTPUT / "manifest.sha256"
    legacy_location = ROOT / "outputs/tora_q3_native_matched_20260806/manifest.sha256"
    assert performance_manifest.read_bytes() == legacy_location.read_bytes()
