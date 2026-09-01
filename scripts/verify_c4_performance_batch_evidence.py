#!/usr/bin/env python3
"""Fail-closed verifier for the C4 reference/performance/CPU-batch package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import formal_reference_configuration  # noqa: E402


ARTIFACT_RELATIVE = Path("artifacts/runs/c4_reference_performance_batch_20260829")
REQUIRED_ARTIFACTS = (
    "PROVENANCE.json",
    "REFERENCE_CONFIG.json",
    "VDP_REGRESSION.json",
    "BRUSSELATOR_REGRESSION.json",
    "production_vs_audit_overhead.csv",
    "hotspot_profile.csv",
    "call_count_matrix.csv",
    "allocation_profile.csv",
    "flamegraph.txt",
    "profile_summary.json",
    "optimization_authorization.json",
    "optimization_result.json",
    "prefix_runtime_matrix.csv",
    "full_runtime_matrix.csv",
    "cpu_batch_equivalence.csv",
    "cpu_batch_runtime.csv",
    "cpu_batch_result.json",
    "RESULT.json",
    "SHA256SUMS",
)
REQUIRED_PROFILE_WINDOWS = {
    "brusselator_steps_1_20",
    "brusselator_steps_1_100",
    "brusselator_steps_901_1000",
    "vdp_representative_prefix",
}
REQUIRED_PROFILE_BUCKETS = {
    "polynomial Picard construction",
    "initial raw-remainder image",
    "post-accept remainder replays",
    "range bounding/subdivision",
    "polynomial multiplication/truncation/cutoff",
    "SR history propagation",
    "normalization/right-map/reset",
    "outward interval/roundoff accounting",
    "Python orchestration/allocation",
    "audit/serialization",
    "other",
}
REQUIRED_CALL_METRICS = {
    "range-bound calls",
    "RHS term-evaluation calls",
    "Taylor multiplication/truncation calls",
    "post-accept replay calls",
    "SR prepare calls",
    "SR propagate calls",
    "SR commit calls",
    "checkpoint/trace calls",
}
REQUIRED_DOCS = (
    "docs/C4_REFERENCE_PERFORMANCE_BATCH_FOUNDATION_20260829.md",
    "docs/C4_REFERENCE_CONFIGURATION.md",
    "docs/C4_PERFORMANCE_PROFILE.md",
    "docs/C4_CPU_BATCH_CONTRACT.md",
)
REQUIRED_SCRIPTS = (
    "experiments/profile_c4_reference_solver.py",
    "experiments/run_c4_performance_gate.py",
    "experiments/run_c4_cpu_batch_equivalence.py",
    "scripts/package_c4_performance_batch_evidence.py",
    "scripts/verify_c4_performance_batch_evidence.py",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(rows: Sequence[Mapping[str, str]], variant: str, steps: int) -> float:
    values = [
        float(row["wall_s"])
        for row in rows
        if row["variant"] == variant
        and int(row["steps"]) == steps
        and row["workload"] == "brusselator"
    ]
    if not values:
        raise AssertionError(f"missing Brusselator {variant} prefix {steps}")
    return statistics.median(values)


def _assert_hash_manifest(artifact_dir: Path) -> None:
    lines = [
        line.strip()
        for line in (artifact_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    declared: dict[str, str] = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        assert separator and len(digest) == 64, "malformed SHA256SUMS line"
        assert name not in declared, f"duplicate SHA256SUMS entry: {name}"
        declared[name] = digest
    expected = {
        path.name
        for path in artifact_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(declared) == expected, "SHA256SUMS file set mismatch"
    for name, digest in declared.items():
        assert _sha256(artifact_dir / name) == digest, f"SHA mismatch: {name}"


def verify(repo_root: Path, artifact_dir: Path) -> dict[str, Any]:
    for name in REQUIRED_ARTIFACTS:
        assert (artifact_dir / name).is_file(), f"missing required artifact: {name}"
    for relative in (*REQUIRED_DOCS, *REQUIRED_SCRIPTS):
        assert (repo_root / relative).is_file(), f"missing required source deliverable: {relative}"
    _assert_hash_manifest(artifact_dir)

    provenance = _read_json(artifact_dir / "PROVENANCE.json")
    assert provenance["schema"] == "torch_tm_flowpipe.c4_reference_performance_batch_provenance/1"
    assert provenance["source_package_sha"] == "ed9c305dc39c25eab23a96f4fb3775cc2d13d396"
    assert provenance["source_branch"] == "codex/torch-flowstar-brusselator-live-range-c5-20260828"
    assert provenance["branch"] == "codex/c4-reference-performance-batch-foundation-20260829"
    assert provenance["reference_scientific_sha"] == "f34b5fa4155f5475a681411b627d68345ed401ea"
    assert provenance["optimized_scientific_sha"] == "4939fb288c941a67f55cc191f4d75f8594692f47"
    assert provenance["batch_scientific_sha"] == "7608dd52e48af3ce8ae2e0a8343aae125c63b7f4"
    assert provenance["instrumentation_sha"] == "d6b543446402ef6b12717b727b236fc7c9c75af5"
    assert len(provenance["evidence_assembly_code_sha"]) == 40
    assert provenance["formal_runs_clean"] is True
    assert provenance["cpu_affinity"] == [0]
    assert provenance["cpu_contention_observed"] is False

    reference_config = _read_json(artifact_dir / "REFERENCE_CONFIG.json")
    assert reference_config == formal_reference_configuration(), "reference config drift"

    observer_rows = _rows(artifact_dir / "production_vs_audit_overhead.csv")
    assert {row["lane"] for row in observer_rows} == {
        "production_no_observer",
        "lightweight_counters",
        "full_evidence",
    }
    equality_fields = (
        "accepted_steps",
        "rejected_steps",
        "endpoint_sha256",
        "tube_sha256",
        "final_remainder",
        "queue_sha256",
        "checkpoint_sha256",
        "refinement_replay_calls",
        "refinement_stop_ratio_count",
    )
    baseline = observer_rows[0]
    for row in observer_rows[1:]:
        assert all(row[field] == baseline[field] for field in equality_fields), "observer drift"
    production = next(row for row in observer_rows if row["lane"] == "production_no_observer")
    assert int(production["trace_rows"]) == 0
    for row in observer_rows:
        for field in (
            "solver_wall_median_s",
            "artifact_serialization_s",
            "checkpoint_export_s",
            "trace_construction_s",
            "peak_rss_bytes",
            "python_positive_allocation_count",
        ):
            assert _finite(row[field]) and float(row[field]) >= 0.0

    hotspot = _rows(artifact_dir / "hotspot_profile.csv")
    assert REQUIRED_PROFILE_WINDOWS <= {row["window"] for row in hotspot}
    assert REQUIRED_PROFILE_BUCKETS <= {row["bucket"] for row in hotspot}
    for row in hotspot:
        assert _finite(row["exclusive_wall_s"])
        assert _finite(row["inclusive_wall_s"])

    calls = _rows(artifact_dir / "call_count_matrix.csv")
    assert REQUIRED_PROFILE_WINDOWS <= {row["window"] for row in calls}
    assert REQUIRED_CALL_METRICS <= {row["metric"] for row in calls}
    assert all(int(row["call_count"]) >= 0 for row in calls)
    allocations = _rows(artifact_dir / "allocation_profile.csv")
    assert REQUIRED_PROFILE_WINDOWS <= {row["window"] for row in allocations}
    assert all(int(row["peak_rss_bytes"]) > 0 for row in allocations)
    assert all(int(row["temporary_tensor_result_count"]) > 0 for row in allocations)
    assert all(int(row["temporary_tensor_logical_bytes"]) > 0 for row in allocations)
    profile_summary = _read_json(artifact_dir / "profile_summary.json")
    assert profile_summary["status"] == "PROFILE_COMPLETE"
    assert profile_summary["numerical_reference_sha"] == provenance["reference_scientific_sha"]
    assert profile_summary["instrumentation_sha"] == provenance["instrumentation_sha"]
    assert profile_summary["cpu_affinity"] == [0]

    authorization = _read_json(artifact_dir / "optimization_authorization.json")
    assert authorization["schema"] == "torch_tm_flowpipe.c4_optimization_authorization/1"
    assert authorization["authorized"] is True
    assert authorization["single_candidate_only"] is True
    assert authorization["no_cache"] is True
    assert authorization["common_to_vdp_and_brusselator"] is True
    assert authorization["b1_bitwise_oracle_passed"] is True
    assert (
        float(authorization["profile_total_fraction"]) >= 0.30
        or float(authorization["expected_end_to_end_speedup"]) >= 1.5
    )

    vdp = _read_json(artifact_dir / "VDP_REGRESSION.json")
    brusselator = _read_json(artifact_dir / "BRUSSELATOR_REGRESSION.json")
    assert vdp["passed"] is True and vdp["status"] == "VDP_NATIVE_T10_ZERO_REGRESSION_PASSED"
    assert vdp["native"]["current"]["accepted_steps"] == 246
    assert vdp["native"]["current"]["rejected_attempts"] == 35
    assert vdp["native"]["current"]["completed_horizon"] == 10.0
    assert vdp["native"]["segments_exact"] is True
    assert all(value["exact"] for value in vdp["fixed_snapshots"].values())
    assert brusselator["passed"] is True
    assert brusselator["accepted_steps"] == 1000
    assert brusselator["rejected_steps"] == 0
    assert brusselator["completed_horizon"] == 20.0
    assert brusselator["reference_vs_optimized_exact"] is True
    assert brusselator["historical_final_snapshot_exact"] is True

    prefix = _rows(artifact_dir / "prefix_runtime_matrix.csv")
    for variant in ("reference", "optimized"):
        assert sum(
            row["workload"] == "brusselator"
            and int(row["steps"]) == 100
            and row["variant"] == variant
            for row in prefix
        ) >= 5
        assert sum(
            row["workload"] == "brusselator"
            and int(row["steps"]) == 300
            and row["variant"] == variant
            for row in prefix
        ) >= 3
        assert sum(
            row["workload"] == "vdp_prefix" and row["variant"] == variant
            for row in prefix
        ) >= 3
    assert all(
        row["observer_mode"] == "production_no_observer"
        and row["timer_scope"] == "solver_only_excludes_snapshot_serialization_checkpoint"
        and _finite(row["wall_s"])
        and float(row["wall_s"]) > 0.0
        for row in prefix
    )
    for row in prefix:
        expected_sha = (
            provenance["reference_scientific_sha"]
            if row["variant"] == "reference"
            else provenance["optimized_scientific_sha"]
        )
        assert row["scientific_sha"] == expected_sha

    full = _rows(artifact_dir / "full_runtime_matrix.csv")
    assert {row["variant"] for row in full} == {"reference", "optimized"}
    assert all(int(row["steps"]) == 1000 for row in full)
    assert all(int(row["accepted_steps"]) == 1000 for row in full)
    assert all(int(row["rejected_steps"]) == 0 for row in full)
    assert all(_finite(row["wall_s"]) and float(row["wall_s"]) > 0.0 for row in full)
    for row in full:
        expected_sha = (
            provenance["reference_scientific_sha"]
            if row["variant"] == "reference"
            else provenance["optimized_scientific_sha"]
        )
        assert row["scientific_sha"] == expected_sha
        assert row["observer_mode"] == "production_no_observer"
        assert row["timer_scope"] == "solver_only_excludes_snapshot_serialization_checkpoint"

    optimization = _read_json(artifact_dir / "optimization_result.json")
    measured100 = _median(prefix, "reference", 100) / _median(prefix, "optimized", 100)
    measured300 = _median(prefix, "reference", 300) / _median(prefix, "optimized", 300)
    assert math.isclose(measured100, optimization["brusselator_prefix_100_speedup"], rel_tol=1e-12)
    assert math.isclose(measured300, optimization["brusselator_prefix_300_speedup"], rel_tol=1e-12)
    reference_full = next(row for row in full if row["variant"] == "reference")
    optimized_full = next(row for row in full if row["variant"] == "optimized")
    full_speedup = float(reference_full["wall_s"]) / float(optimized_full["wall_s"])
    assert math.isclose(full_speedup, optimization["brusselator_full_speedup"], rel_tol=1e-12)
    assert optimization["correctness_gate_passed"] is True
    assert optimization["memory_gate_passed"] == (
        int(optimized_full["solver_peak_rss_bytes"])
        <= 1.5 * int(reference_full["solver_peak_rss_bytes"])
    )
    speed_passed = measured100 >= 2.0 and measured300 >= 2.0 and full_speedup >= 2.0
    assert optimization["prefix_speed_gate_passed"] == (measured100 >= 2.0 and measured300 >= 2.0)
    assert optimization["full_speed_gate_passed"] == (full_speedup >= 2.0)
    expected_optimization_status = (
        "SEMANTICS_PRESERVING_OPTIMIZATION_CORRECT__PRODUCTION_SPEED_GATE_PASSED"
        if speed_passed
        else "SEMANTICS_PRESERVING_OPTIMIZATION_CORRECT__PRODUCTION_SPEED_GATE_FAILED"
    )
    assert optimization["status"] == expected_optimization_status

    batch_rows = _rows(artifact_dir / "cpu_batch_equivalence.csv")
    assert {row["scenario"] for row in batch_rows} == {
        "duplicate_embedding",
        "heterogeneous_lane_isolation",
        "chunk_invariance",
        "checkpoint_resume",
    }
    assert all(_truth(row["passed"]) for row in batch_rows)
    layouts = {row["layout"] for row in batch_rows}
    assert {"B1", "B2", "B8", "2xB4", "4xB2", "8xB1"} <= layouts
    batch_runtime = _rows(artifact_dir / "cpu_batch_runtime.csv")
    serial = [float(row["wall_s"]) for row in batch_runtime if row["case"] == "8x_serial_B1"]
    b8 = [float(row["wall_s"]) for row in batch_runtime if row["case"] == "B8_independent_lane_batch"]
    assert serial and b8
    assert statistics.median(b8) <= 2.0 * statistics.median(serial)
    batch_result = _read_json(artifact_dir / "cpu_batch_result.json")
    assert batch_result["scientific_sha"] == provenance["batch_scientific_sha"]
    assert batch_result["cpu_affinity"] == [0]
    assert batch_result["equivalence_passed"] is True
    assert batch_result["b8_runtime_diagnostic_passed"] is True

    result = _read_json(artifact_dir / "RESULT.json")
    expected_final = (
        "C4_REFERENCE_FROZEN__SEMANTICS_PRESERVED__CPU_SPEED_GATE_PASSED__CPU_BATCH_FOUNDATION_PASSED"
        if speed_passed
        else "C4_REFERENCE_FROZEN__CPU_BATCH_FOUNDATION_PASSED__CPU_SPEED_GATE_FAILED"
    )
    assert result["status"] == expected_final
    assert result["reference_frozen"] is True
    assert result["semantics_preserved"] is True
    assert result["cpu_batch_foundation_passed"] is True
    assert result["cpu_speed_gate_passed"] == speed_passed
    assert result["cuda_batch_next_round_authorized"] is True

    return {
        "schema": "torch_tm_flowpipe.c4_performance_batch_verification/1",
        "status": "C4_PERFORMANCE_BATCH_EVIDENCE_VERIFIED",
        "artifact_dir": str(artifact_dir),
        "package_status": result["status"],
        "files_verified": len([path for path in artifact_dir.iterdir() if path.is_file()]),
        "speed_gate_passed": speed_passed,
        "vdp_zero_regression": True,
        "brusselator_zero_regression": True,
        "cpu_batch_foundation": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir is not None
        else repo_root / ARTIFACT_RELATIVE
    )
    try:
        result = verify(repo_root, artifact_dir)
    except Exception as exc:
        print(f"C4_PERFORMANCE_BATCH_EVIDENCE_INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
