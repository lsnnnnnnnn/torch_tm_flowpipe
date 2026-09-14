"""Run the required fresh B2 x 2 Gp/Gr acceptance matrix in a clone."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.live_range_solver.verify import load_run, verify_run
from experiments.resident_tm_block.analyze import compare_online


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        PYTHONPATH="src:.:tests",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="0",
    )
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    jobs = []
    runs = {}
    for plant in ("van_der_pol", "brusselator"):
        for route in ("Gp", "Gr"):
            name = f"{plant}-{route}"
            target = output / name
            command = [
                "taskset", "-c", str(args.core), sys.executable, "-m",
                "experiments.live_range_solver.runner", "--plant", plant,
                "--route", route, "--batch", "2", "--steps", "2",
                "--max-wait-s", ".020", "--max-group", "32", "--warm",
                "--diagnostic", "--output", str(target),
            ]
            started = time.time()
            process = subprocess.run(
                command, cwd=root, env=env, text=True, capture_output=True,
                check=False,
            )
            (output / f"{name}.log").write_text(process.stdout + process.stderr)
            jobs.append({
                "name": name,
                "command": command,
                "start_epoch": started,
                "end_epoch": time.time(),
                "returncode": process.returncode,
            })
            save(output / "jobs.json", jobs)
            if process.returncode:
                raise RuntimeError(f"{name} failed with {process.returncode}")
            run, events = load_run(target)
            receipt = verify_run(run, events, recompute=True)
            if run["source_sha"] != source_sha:
                raise AssertionError(f"{name}: source SHA mismatch")
            if run["ids"] != [0, 31] or run["steps"] != 2:
                raise AssertionError(f"{name}: workload mismatch")
            if run["successful_lane_steps"] != 4:
                raise AssertionError(f"{name}: incomplete workload")
            if route == "Gr":
                if run["counts"].get("resident_requests") != 4:
                    raise AssertionError(f"{name}: resident block not consumed")
                if run["counts"].get("resident_structure_fallbacks", 0):
                    raise AssertionError(f"{name}: resident structure fallback")
            runs[name] = (run, receipt)

    comparisons = []
    for plant in ("van_der_pol", "brusselator"):
        comparison, widths = compare_online(
            runs[f"{plant}-Gp"][0], runs[f"{plant}-Gr"][0],
            "INDEPENDENT_CLONE_B2X2",
        )
        if comparison["decision_mismatches"]:
            raise AssertionError(f"{plant}: decision mismatch")
        if comparison["ordered_support_mismatches"]:
            raise AssertionError(f"{plant}: support mismatch")
        if comparison["ledger_category_mismatches"]:
            raise AssertionError(f"{plant}: ledger mismatch")
        if comparison["width_over_1p10"]:
            raise AssertionError(f"{plant}: width warning")
        comparisons.append(comparison)

    result = {
        "schema": "resident-tm-block-independent-smoke-v1",
        "passed": True,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "clone_head": source_sha,
        "clone_root": str(root),
        "cpu_affinity": [args.core],
        "plants": ["van_der_pol", "brusselator"],
        "routes": ["Gp", "Gr"],
        "task_ids": [0, 31],
        "steps_per_task": 2,
        "fresh_runs": 4,
        "successful_lane_steps_per_run": 4,
        "successful_lane_steps_total": 16,
        "resident_requests_total": sum(
            runs[f"{plant}-Gr"][0]["counts"]["resident_requests"]
            for plant in ("van_der_pol", "brusselator")
        ),
        "comparisons": comparisons,
        "complete_performance_matrix_reexecuted": False,
        "full_1000_step_runs_reexecuted": 0,
        "saved_answers_used_to_advance": False,
        "jobs_file": "jobs.json",
    }
    save(output / "RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
