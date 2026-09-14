"""Run the frozen online-correctness and paired resident-block campaigns."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.live_range_solver.runner import PARTITION, ROOT
from experiments.live_range_solver.verify import load_run, verify_run
from experiments.range_batch_device.common import read, save, sha


PARENT_CPU_DIAGNOSTIC = ROOT / "artifacts/runs/live_range_solver_20260909T053007Z/diagnostic"
HISTORY_CHECKPOINTS = {
    "van_der_pol": ROOT / (
        "artifacts/runs/live_gpu_packets_20260910T023603Z/full_horizon/"
        "van_der_pol-Gp/checkpoints/step_0099"
    ),
    "brusselator": ROOT / (
        "artifacts/runs/live_gpu_packets_20260910T023603Z/full_horizon/"
        "brusselator-Gp/checkpoints/step_0999"
    ),
}
# Five fixed pairs, with both route and plant order alternated.  There is no
# adaptive stopping and no additional sample may be appended to cross a gate.
PAIR_ORDERS = (("Gp", "Gr"), ("Gr", "Gp"), ("Gp", "Gr"), ("Gr", "Gp"), ("Gp", "Gr"))


def diagnostic_cases() -> list[dict]:
    cases: list[dict] = []

    def add(plant, route, batch, steps, label, **options):
        cases.append(
            dict(
                name=f"{label}-{plant}-{route}", plant=plant, route=route,
                batch=batch, steps=steps, options=options,
            )
        )

    for plant in ("van_der_pol", "brusselator"):
        for batch in (1, 2):
            for route in ("Gp", "Gr"):
                add(plant, route, batch, 3, f"small-b{batch}")
        for batch in (8, 32):
            for route in ("Gp", "Gr"):
                add(plant, route, batch, 20, f"prefix-b{batch}")
        for route in ("Gp", "Gr"):
            add(plant, route, 2, 120, "continuous-b2")
            add(
                plant, route, 1, 3, "history-reset",
                checkpoint=str(HISTORY_CHECKPOINTS[plant].relative_to(ROOT)),
            )
    return cases


def formal_cases() -> list[dict]:
    cases: list[dict] = []
    for block, routes in enumerate(PAIR_ORDERS):
        plants = (
            ("van_der_pol", "brusselator")
            if block % 2 == 0
            else ("brusselator", "van_der_pol")
        )
        for plant in plants:
            for position, route in enumerate(routes):
                cases.append(
                    dict(
                        name=f"pair{block}-{plant}-{route}", plant=plant,
                        route=route, batch=32, steps=20,
                        pair=block, order_position=position, options={},
                    )
                )
    # Distributional CPU scale: three Q samples and one S check per plant.
    for repetition in range(3):
        plants = (
            ("van_der_pol", "brusselator")
            if repetition % 2 == 0
            else ("brusselator", "van_der_pol")
        )
        for plant in plants:
            cases.append(
                dict(
                    name=f"cpu-q{repetition}-{plant}", plant=plant, route="Q",
                    batch=32, steps=20, cpu_repetition=repetition, options={},
                )
            )
    for plant in ("brusselator", "van_der_pol"):
        cases.append(
            dict(
                name=f"cpu-s0-{plant}", plant=plant, route="S", batch=32,
                steps=20, cpu_repetition=0, options={},
            )
        )
    return cases


def cases_for(phase: str) -> list[dict]:
    return diagnostic_cases() if phase == "diagnostic" else formal_cases()


def expected_ids(case: dict) -> list[int]:
    if "ids" in case["options"]:
        return case["options"]["ids"]
    if case["batch"] == 2:
        return [0, 31]
    return read(PARTITION)["subsets"][str(case["batch"])]


def scientific_identity() -> dict[str, str]:
    paths: list[Path] = []
    for folder in (
        ROOT / "src/torch_tm_flowpipe",
        ROOT / "experiments/live_range_solver",
        ROOT / "experiments/resident_tm_block",
    ):
        paths.extend(path for path in folder.rglob("*") if path.suffix in {".py", ".cu"})
    paths.append(ROOT / "experiments/endpoint_roundoff_repair/frozen.py")
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}


def make_plan(phase: str, cases: list[dict]) -> dict:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise SystemExit("campaign requires a committed clean scientific tree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    gpu = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    return dict(
        schema="resident-tm-block-campaign-plan-v1", phase=phase,
        frozen_before_execution=True, source_sha=head, cases=cases,
        scientific_sources=scientific_identity(),
        partition_reference=str(PARTITION.relative_to(ROOT)),
        partition_sha256=sha(PARTITION), parent_commit="372cede015105a3bcdf56b514b65f7dd0551fc43",
        parent_cpu_diagnostic=str(PARENT_CPU_DIAGNOSTIC.relative_to(ROOT)),
        parent_cpu_reused_for_numerical_width_scale=phase == "diagnostic",
        history_checkpoint_origins={
            plant: str(path.relative_to(ROOT)) for plant, path in HISTORY_CHECKPOINTS.items()
        },
        history_windows_are_resumed_not_fresh_horizons=True,
        cpu_affinity=[2], available_logical_cpu_count=1,
        intra_op_threads=1, inter_op_threads=1, cuda_visible_devices="0", gpu=gpu,
        max_wait_s=0.020, max_group=32,
        cold_compile_and_startup_outside_primary_wall=True,
        current_input_preparation_inside_primary_wall=True,
        pair_orders=[list(row) for row in PAIR_ORDERS] if phase == "formal" else None,
        fixed_pair_count=5 if phase == "formal" else None,
        timing_profiler_enabled=False,
    )


def check_completed(target: Path, case: dict, plan: dict) -> None:
    run, events = load_run(target)
    assert run["source_sha"] == plan["source_sha"]
    assert run["plant"] == case["plant"] and run["route"] == case["route"]
    assert run["ids"] == expected_ids(case) and run["steps"] == case["steps"]
    assert run["successful_lane_steps"] == len(run["ids"]) * case["steps"]
    assert run["task_statuses"] == {str(index): "FINISHED" for index in run["ids"]}
    assert run["packet_mode"] is (case["route"] in {"Gp", "Gr"})
    assert run["resident_tm_block"] is (case["route"] == "Gr")
    verify_run(run, events, recompute=bool(run["diagnostic"]))


def run_campaign(phase: str, output: Path, *, resume: bool = False) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=resume)
    cases = cases_for(phase)
    manifest = output / "CAMPAIGN_PLAN.json"
    if manifest.exists():
        if not resume:
            raise SystemExit(f"existing campaign requires --resume: {output}")
        plan = read(manifest)
        assert plan["phase"] == phase and plan["cases"] == cases
        assert plan["scientific_sources"] == scientific_identity()
        assert plan["source_sha"] == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    else:
        plan = make_plan(phase, cases)
        save(manifest, plan)
    if phase == "formal":
        gate = read(output.parent / "diagnostic/CORRECTNESS_GATE.json")
        assert gate["passed"] is True and gate["source_sha"] == plan["source_sha"]

    env = dict(
        os.environ, PYTHONPATH="src:.:tests", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0",
    )
    receipts = read(output / "jobs.json") if (output / "jobs.json").exists() else []
    for case in cases:
        target = output / case["name"]
        if (target / "summary.json").exists() and resume:
            check_completed(target, case, plan)
            continue
        if target.exists():
            raise SystemExit(f"inspect incomplete output before resuming: {target}")
        command = [
            "taskset", "-c", "2", sys.executable, "-m", "experiments.live_range_solver.runner",
            "--plant", case["plant"], "--route", case["route"],
            "--batch", str(case["batch"]), "--steps", str(case["steps"]),
            "--output", str(target), "--max-wait-s", ".020", "--max-group", "32",
        ]
        if case["route"] in {"Gp", "Gr"}:
            command.append("--warm")
        if phase == "diagnostic":
            command.append("--diagnostic")
        for key, value in case["options"].items():
            command.append("--" + key.replace("_", "-"))
            if not isinstance(value, bool):
                command.append(",".join(map(str, value)) if key == "ids" else str(value))
        log = output / f"{case['name']}.log"
        started = time.time()
        with log.open("w") as stream:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            live = dict(
                case=case["name"], pid=process.pid, command=command,
                start_epoch=started, source_sha=plan["source_sha"], status="RUNNING",
            )
            save(output / "CURRENT_JOB.json", live)
            print(json.dumps(live), flush=True)
            code = process.wait()
        receipt = dict(live)
        receipt.update(
            exit_code=code, end_epoch=time.time(), log_sha256=sha(log),
            status="COMPLETED" if code == 0 else "FAILED",
        )
        receipts.append(receipt)
        save(output / "jobs.json", receipts)
        save(output / "CURRENT_JOB.json", receipt)
        if code:
            raise SystemExit(f"case failed ({code}): {log}")
        check_completed(target, case, plan)
        summary = read(target / "summary.json")
        print(
            json.dumps(
                dict(
                    completed=case["name"], wall_s=summary["wall_s"],
                    lane_steps=summary["successful_lane_steps"],
                )
            ),
            flush=True,
        )
    completed = dict(
        schema="resident-tm-block-campaign-completion-v1", phase=phase,
        cases=len(cases), source_sha=plan["source_sha"],
        utc=datetime.now(timezone.utc).isoformat(),
    )
    save(output / "COMPLETED.json", completed)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("diagnostic", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_campaign(args.phase, args.output, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
