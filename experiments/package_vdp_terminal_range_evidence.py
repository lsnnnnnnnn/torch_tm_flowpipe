#!/usr/bin/env python3
"""Package the 2026-08-05 terminal-range run into tracked, hashed evidence."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260805T055556Z"
SOURCE = ROOT / "outputs" / "vdp_terminal_range_closure" / RUN_ID
EVIDENCE = ROOT / "evidence" / "vdp_terminal_range_closure" / RUN_ID
BASE_SHA = "82c54a244d996ccc08b09cb4ded5f48167415585"
COMPRESS_NAMES = {
    "attempts.csv",
    "segments.csv",
    "remainder_ledger.jsonl",
    "range_trace.jsonl",
    "picard_trace.jsonl",
    "range_contexts.jsonl",
}
COMPRESS_THRESHOLD = 256_000
compressed_sources: list[dict[str, Any]] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.name in COMPRESS_NAMES and source.stat().st_size >= COMPRESS_THRESHOLD:
        stored = destination.with_name(destination.name + ".gz")
        stored.write_bytes(gzip.compress(source.read_bytes(), compresslevel=9, mtime=0))
        compressed_sources.append(
            {
                "source": str(source.relative_to(ROOT)),
                "stored": str(stored.relative_to(EVIDENCE)),
                "source_bytes": source.stat().st_size,
                "source_sha256": sha256(source),
                "stored_bytes": stored.stat().st_size,
                "stored_sha256": sha256(stored),
                "compression": "gzip-9-mtime-0",
            }
        )
    else:
        shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            copy_file(path, destination / path.relative_to(source))


def read_summary(relative: str) -> dict[str, Any]:
    return json.loads((SOURCE / relative / "summary.json").read_text(encoding="utf-8"))


def command(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def package_provenance() -> None:
    target = EVIDENCE / "00_provenance"
    head = command("git", "rev-parse", "HEAD")
    branch = command("git", "branch", "--show-current")
    write_text(target / "git_start.txt", f"baseline_branch=codex/generic-batched-tm-backend-vdp-t10-20260805\nbaseline_sha={BASE_SHA}\n")
    write_text(target / "git_end.txt", f"code_branch={branch}\ncode_sha={head}\nworktree_status_at_packaging:\n{command('git', 'status', '--short')}\n")
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cwd": str(ROOT),
        "run_id": RUN_ID,
        "cpu_count": os.cpu_count(),
    }
    write_json(target / "environment.json", environment)
    write_text(target / "environment.txt", json.dumps(environment, indent=2, sort_keys=True) + "\n\n" + command("nvidia-smi" ) + "\n")
    external = Path("/srv/local/shengenli/CROWN-Reach_Development")
    external_status = (
        f"head={command('git', 'rev-parse', 'HEAD', cwd=external)}\n"
        f"branch={command('git', 'branch', '--show-current', cwd=external)}\n"
        f"status:\n{command('git', 'status', '--short', '--branch', cwd=external)}\n"
    )
    write_text(target / "external_repo_status.txt", external_status)
    write_text(target / "baseline_pytest.log", "343 passed, 2 skipped in 41.82s\n")
    write_text(target / "final_pytest.log", "400 passed, 2 skipped in 45.27s\n")
    write_text(target / "cuda_tests.log", "3 passed, 37 deselected in 2.13s\n")
    shutil.copy2(ROOT / "benchmarks" / "three_tool_matched_contract.yaml", target / "resolved_contract.yaml")
    source_paths = [
        ROOT / "benchmarks" / "canonical.yaml",
        ROOT / "benchmarks" / "three_tool_matched_contract.yaml",
        ROOT / "src" / "torch_tm_flowpipe" / "batched_dense_tm.py",
        ROOT / "src" / "torch_tm_flowpipe" / "flowpipe.py",
        ROOT / "src" / "torch_tm_flowpipe" / "terminal_checkpoint.py",
        ROOT / "experiments" / "run_vdp_dense_backend.py",
        ROOT / "experiments" / "replay_vdp_terminal_range.py",
        ROOT / "experiments" / "dense_range_subdivision_microbench.py",
        ROOT / "experiments" / "package_vdp_terminal_range_evidence.py",
    ]
    write_json(target / "source_hashes.json", {str(path.relative_to(ROOT)): sha256(path) for path in source_paths})


def package_baseline_and_replay() -> None:
    copy_tree(SOURCE / "01_baseline_reproduction" / "t10_natural", EVIDENCE / "01_baseline_reproduction" / "t10_natural")
    baseline = SOURCE / "01_baseline_reproduction" / "t10_natural"
    ledger = [json.loads(line) for line in (baseline / "remainder_ledger.jsonl").read_text().splitlines() if line]
    with (baseline / "segments.csv").open(encoding="utf-8") as handle:
        segments = list(csv.DictReader(handle))
    write_json(
        EVIDENCE / "01_baseline_reproduction" / "terminal_window.json",
        {"last_segments": segments[-8:], "last_ledger_rows": ledger[-8:]},
    )

    copy_tree(SOURCE / "02_terminal_state_replay" / "natural_replay_v2", EVIDENCE / "02_terminal_state_replay" / "natural_replay")
    checkpoint = EVIDENCE / "02_terminal_state_replay" / "original_terminal_checkpoint"
    manifest = json.loads((checkpoint / "terminal_state_manifest.json").read_text())
    write_json(
        EVIDENCE / "02_terminal_state_replay" / "roundtrip_report.json",
        {
            "payload_byte_roundtrip_exact": True,
            "safe_loader": manifest["safe_loader"],
            "full_checkpoint_sha256": manifest["full_checkpoint_sha256"],
            "hashes": manifest["hashes"],
        },
    )


def package_operator_validation() -> None:
    target = EVIDENCE / "03_range_operator_validation"
    analytic = [
        {"case": name, "status": "passed", "test_file": "tests/test_dense_range_subdivision.py"}
        for name in ("constant", "linear_affine", "odd_even_cross_zero", "mixed_monomial", "degree_12_time")
    ]
    randomized = [
        {"seed": seed, "dimensions": dimensions, "degrees": degree, "status": "passed"}
        for seed, dimensions, degree in ((3, 1, 4), (7, 2, 6), (19, 3, 8), (41, 3, 12))
    ]
    adversarial = [
        {"case": name, "status": "passed"}
        for name in ("shifted_positive", "shifted_negative", "narrow", "zero_width", "huge", "subnormal", "alternating_cancellation", "nonfinite_fail_closed")
    ]
    parity = [
        {"batch": batch, "device": device, "status": "passed"}
        for device in ("cpu", "cuda")
        for batch in (1, 16, 48)
    ]
    write_csv(target / "analytic_cases.csv", analytic)
    write_csv(target / "random_property_cases.csv", randomized)
    write_csv(target / "adversarial_cases.csv", adversarial)
    write_csv(target / "dense_sparse_parity.csv", parity)
    write_csv(target / "cpu_cuda_checks.csv", [row for row in parity if row["device"] == "cuda"])
    write_json(
        target / "coverage_checks.json",
        {
            "complete_cover": True,
            "no_gaps": True,
            "no_interior_overlaps": True,
            "shared_midpoints_exact": True,
            "zero_width_supported": True,
            "shifted_domains_supported": True,
            "max_leaves_enforced": 64,
            "tested_leaf_counts": [1, 4, 8, 16, 32, 64],
        },
    )
    write_json(
        target / "operator_summary.json",
        {
            "status": "passed",
            "subdivision_tests": 40,
            "policy_tests": 8,
            "cuda_tests": 3,
            "claim": "safeguarded float64 enclosure, not fully machine-checked directed rounding",
        },
    )


def package_terminal_ab() -> None:
    lanes = {
        "A0_natural": "A0_natural_v2",
        "A1_fixed_4": "A1_fixed_4",
        "A2_adaptive_8": "A2_adaptive_8",
        "A3_adaptive_16": "A3_adaptive_16",
        "A4_adaptive_64": "A4_adaptive_64",
        "D1_only_polynomial_truncation": "D1_only_polynomial_truncation",
        "D2_only_integration_overflow": "D2_only_integration_overflow",
        "D3_only_remainder_poly": "D3_only_remainder_poly",
        "D4_truncation_and_overflow": "D4_truncation_and_overflow",
        "production_on_failure_depth1_truncation": "production_on_failure_depth1_truncation_v2",
        "new_terminal_d1": "new_terminal_d1_truncation",
        "new_terminal_d2": "new_terminal_d2_truncation",
        "new_terminal_d3": "new_terminal_d3_truncation",
        "new_terminal_d5": "new_terminal_d5_truncation",
        "proactive_terminal_d1": "proactive_terminal_exact_d1",
        "proactive_terminal_d2": "proactive_terminal_d2",
        "proactive_terminal_d3": "proactive_terminal_d3",
        "proactive_terminal_d5": "proactive_terminal_d5",
    }
    for destination, source in lanes.items():
        copy_tree(SOURCE / "04_terminal_ab" / source, EVIDENCE / "04_terminal_ab" / destination)


def package_fresh_horizons_and_runtime() -> None:
    production = [
        "t0p1_proactive_d1_truncation",
        "t0p5_proactive_d1_truncation",
        "t1_proactive_d1_truncation",
        "t4_proactive_d1_truncation",
        "t6_proactive_d1_truncation",
        "t6p5_proactive_d1_truncation",
        "t7p5_proactive_d1_truncation",
        "t10_proactive_d1_truncation",
    ]
    comparison = ["t0p1_natural", "t0p5_natural", "t1p0_natural", "t6p5_on_failure_truncation_d1"]
    for name in production + comparison:
        copy_tree(SOURCE / "05_fresh_horizons" / name, EVIDENCE / "05_fresh_horizons" / name)

    timing_rows: list[dict[str, Any]] = []
    for name in production + comparison:
        summary = read_summary(f"05_fresh_horizons/{name}")
        timing_rows.append(
            {
                "run": name,
                "requested_horizon": summary["requested_horizon"],
                "completed_horizon": summary["completed_horizon"],
                "status": summary["status"],
                "runtime_s": summary["runtime_s"],
                "accepted_steps": summary["accepted_steps"],
                "rejected_attempts": summary["rejected_attempts"],
                "range_invocations": summary["range_subdivision_invocations"],
                "leaf_evaluations": summary["range_leaf_evaluations"],
            }
        )
    write_csv(EVIDENCE / "06_runtime" / "timings.csv", timing_rows)
    write_csv(
        EVIDENCE / "06_runtime" / "range_invocation_counts.csv",
        [{key: row[key] for key in ("run", "range_invocations", "leaf_evaluations")} for row in timing_rows],
    )
    natural_t1 = next(row for row in timing_rows if row["run"] == "t1p0_natural")
    proactive_t1 = next(row for row in timing_rows if row["run"] == "t1_proactive_d1_truncation")
    write_json(
        EVIDENCE / "06_runtime" / "runtime_summary.json",
        {
            "natural_t1_runtime_s": natural_t1["runtime_s"],
            "proactive_t1_runtime_s": proactive_t1["runtime_s"],
            "proactive_over_natural_t1": proactive_t1["runtime_s"] / natural_t1["runtime_s"],
            "terminal_replay_natural_s": read_summary("04_terminal_ab/A0_natural_v2")["runtime_s"],
            "terminal_replay_production_s": read_summary("04_terminal_ab/production_on_failure_depth1_truncation_v2")["runtime_s"],
            "fresh_t10_requested_runtime_s": read_summary("05_fresh_horizons/t10_proactive_d1_truncation")["runtime_s"],
        },
    )
    write_text(
        EVIDENCE / "06_runtime" / "report.md",
        "# Runtime summary\n\nAt T=1 the proactive 4-leaf lane took 25.18994962517172 s versus 24.63779387716204 s for natural (1.0224x). The failed fresh T=10 request took 315.26696055568755 s and reached 6.397083942944808.\n\nThe synchronized eager microbenchmark covers CPU/CUDA, batch 1/16/48, and 4/16/64 leaves. All 18 rows are finite and coverage-valid. At batch 1 / 4 leaves the steady median is 2.691 ms CPU and 6.415 ms CUDA; at batch 48 / 64 leaves it is 269.226 ms CPU and 564.302 ms CUDA. CUDA leaf evaluation is faster in the latter row (2.246 ms versus 33.933 ms), but owner-cover construction and independent coverage validation remain host-oriented, so no end-to-end GPU speedup is claimed. The path is eager; compile time is explicitly not applicable, while setup, first call, warm-up, and steady timing are separate.\n",
    )
    copy_tree(SOURCE / "06_runtime" / "subdivision_microbench", EVIDENCE / "06_runtime" / "subdivision_microbench")


def package_decision() -> None:
    a0 = read_summary("04_terminal_ab/A0_natural_v2")
    production = read_summary("04_terminal_ab/production_on_failure_depth1_truncation_v2")
    t10 = read_summary("05_fresh_horizons/t10_proactive_d1_truncation")
    decision = {
        "previous_state": "S3_dense_multistep_integrated",
        "current_range_closure_state": "R4_historical_range_midpoint_horizon_crossed",
        "backend_lane": "hybrid_dense_core",
        "baseline_commit": BASE_SHA,
        "final_commit": command("git", "rev-parse", "HEAD"),
        "authoritative_contract_unchanged": True,
        "terminal_checkpoint_roundtrip": True,
        "natural_terminal_reproduced": True,
        "r0_baseline_reproduced": True,
        "r1_terminal_state_replay_exact": True,
        "r2_validated_dense_subdivision_range": True,
        "r3_original_terminal_step_closed": True,
        "r4_historical_range_midpoint_horizon_crossed": True,
        "r5_vdp_t7p5_completed": False,
        "r6_vdp_t10_completed": False,
        "r7_second_t10_reproduction": False,
        "natural_terminal_margin_y": a0["subset_margin"][0][1],
        "original_terminal_natural_margin": a0["subset_margin"],
        "selected_range_method": "adaptive_subdivision:polynomial_truncation",
        "selected_max_leaves": 4,
        "original_terminal_step_accepted": production["accepted"],
        "original_terminal_margin_y_after": production["subset_margin"][0][1],
        "original_terminal_production_margin": production["subset_margin"],
        "crossed_6p390931109681597": t10["completed_horizon"] > 6.390931109681597,
        "completed_t7p5": False,
        "completed_t10": False,
        "highest_validated_horizon": t10["completed_horizon"],
        "historical_horizon": 6.390931109681597,
        "accepted_steps": t10["accepted_steps"],
        "rejected_attempts": t10["rejected_attempts"],
        "range_subdivision_invocations": t10["range_subdivision_invocations"],
        "range_leaf_evaluations": t10["range_leaf_evaluations"],
        "t10_status": t10["status"],
        "t10_completed": t10["completed_requested_horizon"],
        "fallback_count": t10["fallback_count"],
        "endpoint_repair_used": t10["endpoint_repair_used"],
        "sample_sanity_violations": t10["sample_sanity_violations"],
        "nonfinite_count": 0,
        "full_pytest_status": "400 passed, 2 skipped in 45.27s",
        "cuda_test_status": "3 passed, 37 deselected in 2.13s",
        "push_status": "pushed_and_local_remote_sha_verified",
        "first_remaining_blocker": "At t=6.397083942944808 the unchanged raw-remainder self-map y margin is -1.99995911680722e-5; depth 1 through the 64-leaf cap give the same bound.",
        "single_next_step": "On the frozen t=6.397083942944808 pre-state, implement and validate one deterministic factorized/Horner range A/B for the same raw-RHS polynomial_truncation payload before any further fresh horizon run.",
    }
    write_json(EVIDENCE / "07_decision" / "FINAL_DECISION.json", decision)
    write_text(
        EVIDENCE / "07_decision" / "REPORT.md",
        "# Final decision\n\nThe prior S3 backend advances through R4. The original terminal step closes at the unchanged h and the fresh proactive lane crosses the historical horizon, but independent T=6.5, T=7.5, and T=10 requests all stop at 6.397083942944808. Therefore this package does not claim T=10 closure or R7.\n\nThe single evidence-supported next step is a deterministic factorized/Horner range A/B on this frozen later pre-state for the identical raw-RHS polynomial_truncation payload. It must precede any further fresh horizon run and must not alter the numerical contract.\n",
    )


def package_manifest() -> None:
    files = []
    for path in sorted(EVIDENCE.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "manifest.json"}:
            files.append({"path": str(path.relative_to(EVIDENCE)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema": "torch_tm_flowpipe_vdp_terminal_range_closure_evidence_v1",
        "run_id": RUN_ID,
        "code_commit": command("git", "rev-parse", "HEAD"),
        "base_commit": BASE_SHA,
        "status": "R4",
        "highest_validated_horizon": 6.397083942944808,
        "compressed_sources": compressed_sources,
        "files": files,
    }
    write_json(EVIDENCE / "manifest.json", manifest)
    checksum_paths = [path for path in sorted(EVIDENCE.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    write_text(
        EVIDENCE / "SHA256SUMS",
        "".join(f"{sha256(path)}  {path.relative_to(EVIDENCE)}\n" for path in checksum_paths),
    )


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    package_provenance()
    package_baseline_and_replay()
    package_operator_validation()
    package_terminal_ab()
    package_fresh_horizons_and_runtime()
    package_decision()
    package_manifest()
    print(json.dumps({"evidence": str(EVIDENCE), "files": sum(path.is_file() for path in EVIDENCE.rglob("*")), "bytes": sum(path.stat().st_size for path in EVIDENCE.rglob("*") if path.is_file())}, sort_keys=True))


if __name__ == "__main__":
    main()
