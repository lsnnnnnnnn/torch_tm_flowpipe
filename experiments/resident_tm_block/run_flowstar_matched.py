"""Build and run the same-workload Flow* B32 x 20 comparison."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any


PLANTS = ("van_der_pol", "brusselator")
STATUS_COMPLETED_SAFE = 2


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git(flowstar: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(flowstar), *args], text=True
    ).strip()


def expected_boxes(plan: dict, plant: str) -> list[list[list[str]]]:
    axes = plan["plants"][plant]["axes"]
    return [
        [axes[0][x]["outward_hex"], axes[1][y]["outward_hex"]]
        for y in range(4)
        for x in range(8)
    ]


def build(
    compiler: Path, flowstar: Path, driver: Path, binary: Path, log: Path
) -> tuple[list[str], float]:
    toolbox = flowstar / "flowstar-toolbox"
    command = [
        str(compiler), "-fpermissive", "-O3", "-g", "-std=c++11",
        "-fopenmp", "-I", str(toolbox), str(driver),
        str(toolbox / "libflowstar.a"), "-lmpfr", "-lgmp", "-lgsl",
        "-lgslcblas", "-lm", "-lglpk", "-lcolamd", "-lamd", "-lz",
        "-lltdl", "-o", str(binary),
    ]
    start = time.perf_counter_ns()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Flow* driver build failed with {result.returncode}")
    return command, elapsed


def run_plant(
    binary: Path,
    artifact: Path,
    plant: str,
    steps: int,
    core: int,
    expected: list[list[list[str]]],
) -> tuple[dict, dict]:
    raw = artifact / "raw_minimal"
    tasks_path = raw / f"{plant}.tasks.jsonl"
    summary_path = raw / f"{plant}.summary.json"
    command = [
        "taskset", "-c", str(core), str(binary), plant, str(steps),
        str(tasks_path), str(summary_path),
    ]
    env = {
        **__import__("os").environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    start = time.perf_counter_ns()
    result = subprocess.run(
        command, text=True, capture_output=True, check=False, env=env
    )
    wall_s = (time.perf_counter_ns() - start) / 1e9
    (raw / f"{plant}.stdout.log").write_text(result.stdout)
    (raw / f"{plant}.stderr.log").write_text(result.stderr)
    receipt = {
        "command": command,
        "environment": {
            name: env[name]
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "fresh_process_startup_included": True,
        "filesystem_cache_flushed": False,
        "returncode": result.returncode,
        "wall_seconds": wall_s,
    }
    save_json(raw / f"{plant}.process.json", receipt)
    if result.returncode:
        raise RuntimeError(f"Flow* {plant} failed with {result.returncode}")

    summary = load_json(summary_path)
    records = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    if len(records) != 32:
        raise AssertionError(f"{plant}: expected 32 tasks, observed {len(records)}")
    if [row["task_id"] for row in records] != list(range(32)):
        raise AssertionError(f"{plant}: task order mismatch")
    if [row["box_hex"] for row in records] != expected:
        raise AssertionError(f"{plant}: boxes differ from PARTITION_PLAN.json")
    if any(row["accepted_steps"] != steps for row in records):
        raise AssertionError(f"{plant}: incomplete task")
    if any(row["native_status"] != STATUS_COMPLETED_SAFE for row in records):
        raise AssertionError(f"{plant}: native status is not COMPLETED_SAFE")
    if summary["accepted_lane_steps"] != 32 * steps:
        raise AssertionError(f"{plant}: incomplete workload")
    if summary["exact_history_contract_supported"] is not False:
        raise AssertionError(f"{plant}: unsupported contract was not disclosed")
    return summary, receipt


def paired_medians(path: Path, plant: str) -> tuple[float, float]:
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["plant"] == plant]
    if len(rows) != 5:
        raise AssertionError(f"{plant}: expected five fixed pairs")
    return (
        statistics.median(float(row["Gp_wall_s"]) for row in rows),
        statistics.median(float(row["Gr_wall_s"]) for row in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-root", type=Path, required=True)
    parser.add_argument("--readonly-patch", type=Path, required=True)
    parser.add_argument("--partition-plan", type=Path, required=True)
    parser.add_argument("--paired-speedups", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, default=Path("/usr/bin/g++-15"))
    parser.add_argument("--core", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    flowstar = args.flowstar_root.resolve()
    artifact = args.artifact.resolve()
    raw = artifact / "raw_minimal"
    raw.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).with_name("flowstar_matched_b32.cpp").resolve()
    binary = raw / "flowstar_matched_b32"
    archive = flowstar / "flowstar-toolbox/libflowstar.a"
    license_path = flowstar / "LICENSE"
    plan = load_json(args.partition_plan)

    head = git(flowstar, "rev-parse", "HEAD")
    stock_base = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
    ancestor = subprocess.run(
        ["git", "-C", str(flowstar), "merge-base", "--is-ancestor", stock_base, head]
    ).returncode == 0
    if not ancestor:
        raise AssertionError("pinned stock Flow* base is not an ancestor")
    if git(flowstar, "diff", "--name-only"):
        raise AssertionError("Flow* has tracked working-tree modifications")

    build_command, build_s = build(
        args.compiler.resolve(), flowstar, driver, binary, raw / "build.log"
    )
    summaries: dict[str, dict] = {}
    receipts: dict[str, dict] = {}
    for plant in PLANTS:
        summaries[plant], receipts[plant] = run_plant(
            binary, artifact, plant, args.steps, args.core,
            expected_boxes(plan, plant),
        )

    compiler_version = subprocess.check_output(
        [str(args.compiler.resolve()), "--version"], text=True
    ).splitlines()[0]
    configured_remote = git(flowstar, "remote", "get-url", "origin")
    configured_remote_path = Path(configured_remote)
    canonical_remote = (
        git(configured_remote_path, "remote", "get-url", "origin")
        if configured_remote_path.is_dir()
        else configured_remote
    )
    identity = {
        "schema": "resident-tm-block-flowstar-identity-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "configured_remote": configured_remote,
        "canonical_upstream_remote": canonical_remote,
        "stock_base_sha": stock_base,
        "stock_base_is_ancestor": ancestor,
        "readonly_accessor_commit_sha": head,
        "tracked_worktree_clean": True,
        "readonly_patch_sha256": digest(args.readonly_patch.resolve()),
        "flowstar_static_archive_sha256": digest(archive),
        "flowstar_license_sha256": digest(license_path),
        "driver_sha256": digest(driver),
        "binary_sha256": digest(binary),
        "compiler": compiler_version,
        "build_command": build_command,
        "build_seconds_outside_comparison": build_s,
        "algorithm_modifications_for_this_comparison": False,
        "driver_only_addition": True,
        "readonly_export_accessors_invoked_in_timed_region": False,
    }
    save_json(artifact / "flowstar_identity.json", identity)

    contract_gap = (
        "stock Flow* does not expose the Torch solver's accepted-boundary "
        "raw-remainder refinement/replay-491/stop-ratio-0.99/atomic-commit controls"
    )
    rows = []
    for plant in PLANTS:
        baseline_median, candidate_median = paired_medians(
            args.paired_speedups.resolve(), plant
        )
        summary = summaries[plant]
        wall_s = receipts[plant]["wall_seconds"]
        records = [
            json.loads(line)
            for line in (raw / f"{plant}.tasks.jsonl").read_text().splitlines()
        ]
        rows.append({
            "plant": plant,
            "tasks": 32,
            "steps_per_task": args.steps,
            "successful_lane_steps": summary["accepted_lane_steps"],
            "h": summary["h"],
            "order": summary["order"],
            "cutoff": summary["cutoff"],
            "initial_remainder_radius": summary["initial_remainder_radius"],
            "symbolic_remainder_capacity": summary["symbolic_remainder_capacity"],
            "flowstar_process_wall_s": wall_s,
            "flowstar_pre_summary_main_s": summary["pre_summary_main_seconds"],
            "flowstar_reach_s": summary["reach_seconds"],
            "flowstar_setup_s": summary["setup_seconds"],
            "flowstar_postprocess_s": summary["postprocess_seconds"],
            "flowstar_completed_tasks": summary["completed_tasks"],
            "flowstar_native_status": STATUS_COMPLETED_SAFE,
            "flowstar_final_queue_size_min": min(row["final_queue_size"] for row in records),
            "flowstar_final_queue_size_max": max(row["final_queue_size"] for row in records),
            "single_process": True,
            "sequential_tasks": True,
            "cpu_core": args.core,
            "candidate_route": "Gr",
            "candidate_fixed_runs": 5,
            "candidate_median_wall_s": candidate_median,
            "baseline_route": "Gp",
            "baseline_fixed_runs": 5,
            "baseline_median_wall_s": baseline_median,
            "candidate_wall_over_flowstar_wall": candidate_median / wall_s,
            "baseline_wall_over_flowstar_wall": baseline_median / wall_s,
            "exact_history_contract_supported": False,
            "contract_gap": contract_gap,
        })
    with (artifact / "matched_flowstar.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison = {
        "schema": "resident-tm-block-matched-flowstar-v1",
        "passed_complete_workload": True,
        "same_equations_boxes_steps_h_order_cutoff_initial_remainder_sr_capacity": True,
        "exact_history_contract_supported": False,
        "contract_gap": contract_gap,
        "matched_flowstar_gap": "measured",
        "flowstar_is_floating_point_truth": False,
        "rows": rows,
        "identity_file": "flowstar_identity.json",
    }
    save_json(artifact / "MATCHED_FLOWSTAR_RESULT.json", comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
