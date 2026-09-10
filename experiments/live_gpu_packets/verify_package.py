"""Semantically verify the immutable packet evidence and run a bounded live replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from experiments.live_range_solver.analyze import compare_state
from experiments.live_range_solver.runner import ROOT, run_case
from experiments.live_range_solver.verify import load_run, verify_run
from experiments.range_batch_device.common import read, sha

from .analyze import check_case, diagnostic_gate, formal_summary
from .campaign import diagnostic_cases, formal_cases
from .compare_horizons import compare_horizons
from .long_campaign import CASES as LONG_CASES
from .packet_fixture import verify_fixture
from .package import (
    ALIASES, DYNAMIC_PREFIXES, combined_result, evidence_files, render_goal_audit,
    render_report,
)
from .verify_long_horizon import verify_long_horizon


DIAGNOSTIC_DERIVED = (
    "same_backend_state_equivalence.json", "diagnostic_width_comparison.csv",
    "diagnostic_width_over_1p10.csv", "packet_cost_breakdown.csv",
    "actual_launch_transfer_counts.csv",
)
FORMAL_DERIVED = ("timings_raw.csv", "paired_speedups.csv", "PERFORMANCE_RESULT.json")
HORIZON_DERIVED = (
    "full_width_comparison.csv", "full_width_summary.csv", "full_horizon_matrix.csv",
    "full_width_over_1p10.csv", "full_width_near_zero.csv",
    "FULL_HORIZON_COMPARISON_RESULT.json",
)


def check_manifest(root):
    scope = read(root / "MANIFEST_SCOPE.json")
    assert tuple(scope["excluded_dynamic_prefixes"]) == DYNAMIC_PREFIXES
    listed = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        path = Path(relative)
        assert not path.is_absolute() and ".." not in path.parts and relative not in listed
        assert not any(relative.startswith(prefix) for prefix in DYNAMIC_PREFIXES)
        assert sha(root / path) == checksum, f"checksum mismatch: {relative}"
        listed[relative] = checksum
    actual = {str(path.relative_to(root)) for path in evidence_files(root)}
    assert actual == set(listed), "unlisted or missing immutable evidence file"
    return len(listed)


def git_bytes(commit, relative):
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def check_sources(root):
    plan = read(root / "PLAN_FROZEN.json")
    source = read(root / "SOURCE_MAP.json")
    assert source["scientific_sha"] == plan["source_sha"]
    assert source["parent_delivery"] == plan["parent_delivery"]
    subprocess.run(["git", "merge-base", "--is-ancestor", source["packaging_sha"], "HEAD"],
                   cwd=ROOT, check=True)
    for relative, checksum in plan["scientific_sources"].items():
        assert sha(ROOT / relative) == checksum, f"current scientific file changed: {relative}"
        assert git_bytes(plan["source_sha"], relative) == (ROOT / relative).read_bytes()
    assert sha(root / "GOAL_SNAPSHOT.md") == plan["goal_sha256"]
    assert sha(ROOT / plan["partition_path"]) == plan["partition_sha256"]
    for relative, checksum in plan["parent_files"].items():
        assert sha(ROOT / plan["parent_artifact"] / relative) == checksum
    for relative, checksum in plan["reused_complete_object_sources"].items():
        assert sha(ROOT / relative) == checksum
    assert plan["packet_mode_default"] is False
    contract = read(root / "EXECUTION_CONTRACT.json")
    assert contract["cpu_affinity"] == [2]
    assert contract["torch_intra_op_threads"] == contract["torch_inter_op_threads"] == 1
    assert contract["max_group"] == 32 and contract["max_wait_s"] == .020
    assert contract["packet_default_enabled"] is False
    assert contract["formal_first_scratch_allocation_inside_wall"] is True
    return source


def check_tests_and_fixture(root, *, fresh_execution):
    tests = read(root / "tests/PACKET_TEST_RESULT.json")
    assert tests["passed"] and tests["source_sha"] == read(root / "PLAN_FROZEN.json")["source_sha"]
    for name, checksum in tests["files"].items():
        assert sha(root / "tests" / name) == checksum
    lifetime = read(root / "packet_lifetime_fault_checks.json")
    assert lifetime["passed"] and all(lifetime["checks"].values())
    fixture = read(root / "tests/device_packet_fixture.json")
    assert fixture["source_sha"] == tests["source_sha"]
    kernel = fixture["module_build"]
    assert kernel["source_sha256"] == sha(ROOT / "src/torch_tm_flowpipe/range_packet_cuda_kernel.cu")
    assert kernel["capability"] == [7, 0] and kernel["nvrtc_version"] == [12, 1]
    assert kernel["device_name"] == "Tesla V100-SXM2-16GB"
    assert all(flag in kernel["flags"] for flag in
               ("--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"))
    return verify_fixture(root / "tests/device_packet_fixture.json", execute=fresh_execution)


def check_campaign_receipts(root, phase):
    folder = root / phase
    plan = read(folder / "CAMPAIGN_PLAN.json")
    cases = diagnostic_cases() if phase == "diagnostic" else formal_cases()
    assert plan["cases"] == cases and plan["phase"] == phase
    assert plan["source_sha"] == read(root / "PLAN_FROZEN.json")["source_sha"]
    jobs = read(folder / "jobs.json")
    complete = read(folder / "COMPLETED.json")
    assert len(jobs) == len(cases) == complete["cases"]
    previous_end = 0.0
    receipts = []
    for case, job in zip(cases, jobs):
        assert job["case"] == case["name"] and job["status"] == "COMPLETED"
        assert job["exit_code"] == 0 and job["source_sha"] == plan["source_sha"]
        assert job["start_epoch"] >= previous_end and job["end_epoch"] > job["start_epoch"]
        previous_end = job["end_epoch"]
        assert sha(folder / f"{case['name']}.log") == job["log_sha256"]
        command = job["command"]
        assert command[:3] == ["taskset", "-c", "2"]
        assert command[4:6] == ["-m", "experiments.live_range_solver.runner"]
        assert "--warm" in command
        assert ("--diagnostic" in command) is (phase == "diagnostic")
        run, events = load_run(folder / case["name"])
        check_case(run, case, plan, diagnostic=phase == "diagnostic")
        cold = run.get("cold_startup_outside_timing")
        if run["route"] in {"G0", "Gp", "S_gpu"}:
            assert cold and cold["wall_s"] > 0
            kernel_name = ("range_packet_cuda_kernel.cu" if run["route"] == "Gp"
                           else "range_cuda_kernel.cu")
            assert cold["build"]["source_sha256"] == sha(
                ROOT / "src/torch_tm_flowpipe" / kernel_name)
            assert cold["build"]["capability"] == [7, 0]
        if run["route"] == "Gp":
            assert run["packet_limits"] == read(root / "PLAN_FROZEN.json")["packet_limits"]
            packet_timings = [packet for group in run["groups"]
                for packet in group["timing"].get("packet_timings", [])]
            assert packet_timings and packet_timings[0]["device_allocation_operations"] == 3
            assert packet_timings[0]["pinned_allocation_operations"] == 3
        # The full package pass recomputes diagnostic arithmetic once in the
        # deterministic derivative mirror below. This pass checks timelines.
        receipts.append(verify_run(run, events, recompute=False))
    assert read(folder / "CURRENT_JOB.json")["case"] == cases[-1]["name"]
    return receipts


def _symlink_children(source, target, excluded=()):
    target.mkdir(parents=True)
    for path in source.iterdir():
        if path.name in excluded:
            continue
        (target / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())


def recompute_campaign_derivatives(root):
    with tempfile.TemporaryDirectory(prefix="live-gpu-packet-derived-") as temporary:
        mirror = Path(temporary)
        (mirror / "tests").symlink_to((root / "tests").resolve(), target_is_directory=True)
        _symlink_children(root / "diagnostic", mirror / "diagnostic",
            excluded=("CORRECTNESS_GATE.json", "verification_progress.json"))
        diagnostic_gate(mirror / "diagnostic", recompute=True)
        for name in DIAGNOSTIC_DERIVED:
            assert (mirror / name).read_bytes() == (root / name).read_bytes(), name
        for name in ("CORRECTNESS_GATE.json", "verification_progress.json"):
            assert (mirror / "diagnostic" / name).read_bytes() == \
                   (root / "diagnostic" / name).read_bytes(), name
        (mirror / "formal").symlink_to((root / "formal").resolve(), target_is_directory=True)
        formal_summary(mirror / "formal")
        for name in FORMAL_DERIVED:
            assert (mirror / name).read_bytes() == (root / name).read_bytes(), name


def check_opportunity(root, *, recompute):
    opportunity = read(root / "PARENT_OPPORTUNITY.json")
    assert opportunity["projected_not_measured_online"]
    assert opportunity["snapshots_overlap_and_are_not_summable"]
    assert (root / "ready_packet_opportunity.csv").read_bytes() == \
           (root / "opportunity/ready_packet_opportunity.csv").read_bytes()
    assert (root / "PARENT_OPPORTUNITY.json").read_bytes() == \
           (root / "opportunity/PARENT_OPPORTUNITY.json").read_bytes()
    if recompute:
        with tempfile.TemporaryDirectory(prefix="live-gpu-packet-opportunity-") as temporary:
            target = Path(temporary) / "opportunity"
            subprocess.run([sys.executable, "-m",
                "experiments.live_gpu_packets.parent_opportunity", "--output", str(target)],
                cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            for name in ("ready_packet_opportunity.csv", "PARENT_OPPORTUNITY.json"):
                assert (target / name).read_bytes() == (root / "opportunity" / name).read_bytes()
    return opportunity


def check_long_horizons(root, *, recompute_comparison):
    campaign = root / "full_horizon"
    plan = read(campaign / "CAMPAIGN_PLAN.json")
    assert plan["cases"] == list(LONG_CASES) and plan["original_unpartitioned_b1"]
    jobs = read(campaign / "jobs.json")
    assert len(jobs) == 2 and read(campaign / "COMPLETED.json")["cases"] == 2
    receipts = []
    for case, job in zip(LONG_CASES, jobs):
        assert job["case"] == case["name"] and job["status"] == "COMPLETED" and job["exit_code"] == 0
        assert sha(campaign / f"{case['name']}.log") == job["log_sha256"]
        command = job["command"]
        assert command[:3] == ["taskset", "-c", "2"]
        assert command[4:6] == ["-m", "experiments.live_gpu_packets.long_horizon"]
        receipt = verify_long_horizon(campaign / case["name"], require_achieved=True)
        assert read(campaign / f"{case['name']}-verification.json") == receipt
        receipts.append(receipt)
    if recompute_comparison:
        with tempfile.TemporaryDirectory(prefix="live-gpu-packet-horizon-") as temporary:
            target = Path(temporary) / "comparison"
            compare_horizons(campaign, target)
            for name in HORIZON_DERIVED:
                assert (target / name).read_bytes() == \
                       (root / "horizon_comparison" / name).read_bytes(), name
    for alias, source in ALIASES.items():
        assert (root / alias).read_bytes() == (root / source).read_bytes(), alias
    return receipts


def check_final_summaries(root):
    assert read(root / "RESULT.json") == combined_result(root)
    result = read(root / "RESULT.json")
    assert result["packet_correctness"] == "pass"
    assert result["default_enabled"] is False
    assert result["saved_answers_used_to_advance"] is False
    assert result["cpu_periodic_correction"] is False
    assert result["full_gpu_engine"] is False and result["whole_solver_formal_proof"] is False
    assert (ROOT / "docs/live_gpu_packets/REPORT_PLAIN_CHINESE.md").read_text() == render_report(root)
    assert (ROOT / "docs/live_gpu_packets/GOAL_AUDIT.md").read_text() == render_goal_audit(root)
    return result


def bounded_live_replay():
    assert sorted(os.sched_getaffinity(0)) == [2]
    receipts, lane_steps = [], 0
    for plant in ("van_der_pol", "brusselator"):
        parent, _, _ = run_case(plant, [0, 31], 2, "G0", diagnostic=True,
                                 run_id=f"independent-{plant}-G0")
        packet, _, _ = run_case(plant, [0, 31], 2, "Gp", diagnostic=True,
                                 run_id=f"independent-{plant}-Gp")
        comparison = compare_state(parent, packet)
        assert comparison["complete_segments_compared"] == 4
        assert parent["successful_lane_steps"] == packet["successful_lane_steps"] == 4
        packet_count = sum(group["timing"].get("packet_count", 0)
                           for group in packet["groups"])
        kernels = sum(group["timing"].get("actual_kernel_invocations", 0)
                      for group in packet["groups"])
        assert packet_count > 0 and kernels == 4 * packet_count
        assert all(group["packet_mode"] for group in packet["groups"])
        lane_steps += parent["successful_lane_steps"] + packet["successful_lane_steps"]
        receipts.append(dict(plant=plant, G0_wall_s=parent["wall_s"],
            Gp_wall_s=packet["wall_s"], packet_count=packet_count,
            actual_kernel_invocations=kernels, complete_segments_compared=4))
    assert lane_steps == 16
    return dict(actual_live_runs=4, successful_lane_steps=lane_steps, cases=receipts,
                full_1000_step_runs_repeated=0)


def verify_package(root, *, fresh_execution=True, focus=None):
    root = Path(root).resolve()
    manifest_files = check_manifest(root)
    source = check_sources(root)
    if focus in {None, "packet"}:
        fixture = check_tests_and_fixture(root, fresh_execution=fresh_execution)
        if focus:
            return dict(verified=True, focus=focus, manifest_files=manifest_files, fixture=fixture)
    if focus in {None, "diagnostic"}:
        diagnostic = check_campaign_receipts(root, "diagnostic")
        if focus:
            return dict(verified=True, focus=focus, manifest_files=manifest_files,
                        diagnostic_runs=len(diagnostic))
    if focus in {None, "formal"}:
        formal = check_campaign_receipts(root, "formal")
        if focus:
            return dict(verified=True, focus=focus, manifest_files=manifest_files,
                        formal_runs=len(formal))
    if focus in {None, "horizon"}:
        horizons = check_long_horizons(root, recompute_comparison=True)
        if focus:
            return dict(verified=True, focus=focus, manifest_files=manifest_files,
                        horizons=horizons)
    check_opportunity(root, recompute=True)
    recompute_campaign_derivatives(root)
    final = check_final_summaries(root)
    replay = bounded_live_replay() if fresh_execution else None
    return dict(verified=True, manifest_files=manifest_files,
        scientific_sha=source["scientific_sha"], diagnostic_runs=len(diagnostic),
        formal_runs=len(formal), long_horizons=len(horizons), result=final,
        bounded_live_replay=replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--no-fresh-execution", action="store_true")
    parser.add_argument("--focus", choices=("packet", "diagnostic", "formal", "horizon"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        assert torch.get_num_interop_threads() == 1
    result = verify_package(args.root, fresh_execution=not args.no_fresh_execution,
                            focus=args.focus)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
