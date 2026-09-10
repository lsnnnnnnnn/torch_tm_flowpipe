"""Run the preregistered packet diagnostic and five-block timing matrices."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.range_batch_device.common import read, save, sha
from experiments.live_range_solver.runner import PARTITION, ROOT


PARENT_ROOT = ROOT / "artifacts/runs/live_range_solver_20260909T053007Z"
HISTORY_CHECKPOINTS = {
    "van_der_pol": "artifacts/runs/boundary_execution_20260908T172756Z/raw_minimal/state_inputs/van_der_pol_before0099",
    "brusselator": "artifacts/runs/boundary_execution_20260908T172756Z/raw_minimal/state_inputs/brusselator_before0999",
}
FORMAL_ORDERS = (
    ("S", "Q", "G0", "Gp"),
    ("Gp", "G0", "Q", "S"),
    ("Q", "Gp", "S", "G0"),
    ("G0", "S", "Gp", "Q"),
    ("S", "G0", "Q", "Gp"),
)


def diagnostic_cases():
    cases = []

    def add(plant, route, batch, steps, label, **options):
        cases.append(dict(name=f"{label}-{plant}-{route}", plant=plant, route=route,
                          batch=batch, steps=steps, options=options))

    for plant in ("van_der_pol", "brusselator"):
        for batch in (1, 2):
            for route in ("S_gpu", "G0", "Gp"):
                add(plant, route, batch, 3, f"small-b{batch}")
        for batch in (8, 32):
            for route in ("G0", "Gp"):
                add(plant, route, batch, 20, f"prefix-b{batch}")
        add(plant, "Gp", 2, 120, "continuous-b2")
        for route in ("G0", "Gp"):
            add(plant, route, 1, 3, "history-reset",
                checkpoint=HISTORY_CHECKPOINTS[plant])
    return cases


def formal_cases():
    cases = []
    for block, order in enumerate(FORMAL_ORDERS):
        plants = (("van_der_pol", "brusselator") if block % 2 == 0
                  else ("brusselator", "van_der_pol"))
        for plant in plants:
            for position, route in enumerate(order):
                cases.append(dict(name=f"block{block}-{plant}-{route}", plant=plant,
                    route=route, batch=32, steps=20, block=block,
                    order_position=position, options={}))
    return cases


def matrix(phase):
    if phase == "diagnostic":
        return diagnostic_cases()
    if phase == "formal":
        return formal_cases()
    raise ValueError(phase)


def scientific_identity():
    files = []
    for folder in (ROOT / "src/torch_tm_flowpipe", ROOT / "experiments/live_range_solver",
                   ROOT / "experiments/live_gpu_packets"):
        files.extend(path for path in folder.rglob("*") if path.suffix in {".py", ".cu"})
    files.extend((ROOT / "experiments/endpoint_roundoff_repair/frozen.py",
                  ROOT / "experiments/run_vdp_dense_backend.py",
                  ROOT / "experiments/run_brusselator_sr1000_parity.py"))
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(files))}


def expected_ids(case):
    if "ids" in case["options"]:
        return case["options"]["ids"]
    if case["batch"] == 2:
        return [0, 31]
    return read(PARTITION)["subsets"][str(case["batch"])]


def make_plan(phase, cases):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise SystemExit("campaign requires a committed clean scientific tree")
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader"], text=True).strip()
    return dict(schema="live-gpu-packet-campaign-plan-v1", phase=phase,
        frozen_before_execution=True, cases=cases, source_sha=head,
        scientific_sources=scientific_identity(),
        partition_reference=str(PARTITION.relative_to(ROOT)), partition_sha256=sha(PARTITION),
        parent_delivery="e4d920aa710bd407666c60285632a4c294edb9da",
        parent_artifact=str(PARENT_ROOT.relative_to(ROOT)),
        parent_performance_sha256=sha(PARENT_ROOT / "PERFORMANCE_RESULT.json"),
        cpu_affinity=[2], intra_op_threads=1, inter_op_threads=1,
        cuda_visible_devices="0", gpu=gpu, max_wait_s=.020, max_group=32,
        cold_compile_and_startup_outside_primary_wall=True,
        first_packet_scratch_allocation_inside_primary_wall=True,
        formal_orders=[list(row) for row in FORMAL_ORDERS] if phase == "formal" else None,
        formal_samples_fixed=5 if phase == "formal" else None)


def check_completed(target, case, plan):
    from experiments.live_range_solver.verify import load_run, verify_run
    run, events = load_run(target)
    assert run["source_sha"] == plan["source_sha"]
    assert run["plant"] == case["plant"] and run["route"] == case["route"]
    assert run["ids"] == expected_ids(case) and run["steps"] == case["steps"]
    assert run["successful_tasks"] == len(run["ids"])
    verify_run(run, events, recompute=False)


def run_campaign(phase, output, *, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    cases = matrix(phase)
    manifest = output / "CAMPAIGN_PLAN.json"
    if manifest.exists():
        if not resume:
            raise SystemExit(f"existing campaign requires --resume: {output}")
        plan = read(manifest)
        assert plan["phase"] == phase and plan["cases"] == cases
        assert plan["scientific_sources"] == scientific_identity()
        assert plan["source_sha"] == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    else:
        plan = make_plan(phase, cases)
        save(manifest, plan)
    if phase == "formal":
        gate = output.parent / "diagnostic/CORRECTNESS_GATE.json"
        assert gate.exists() and read(gate)["passed"] is True
        assert read(gate)["source_sha"] == plan["source_sha"]

    env = dict(os.environ, PYTHONPATH="src:.:tests", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0")
    receipts = read(output / "jobs.json") if (output / "jobs.json").exists() else []
    for case in cases:
        target = output / case["name"]
        if (target / "summary.json").exists() and resume:
            check_completed(target, case, plan)
            continue
        if target.exists():
            raise SystemExit(f"inspect incomplete output before resuming: {target}")
        command = ["taskset", "-c", "2", sys.executable, "-m",
            "experiments.live_range_solver.runner", "--plant", case["plant"],
            "--route", case["route"], "--batch", str(case["batch"]),
            "--steps", str(case["steps"]), "--output", str(target),
            "--max-wait-s", ".020", "--max-group", "32", "--warm"]
        if phase == "diagnostic":
            command.append("--diagnostic")
        for key, value in case["options"].items():
            command.append("--" + key.replace("_", "-"))
            if not isinstance(value, bool):
                command.append(",".join(map(str, value)) if key == "ids" else str(value))
        log = output / f"{case['name']}.log"
        started = time.time()
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT)
            live = dict(case=case["name"], pid=process.pid, command=command,
                start_epoch=started, source_sha=plan["source_sha"], status="RUNNING")
            save(output / "CURRENT_JOB.json", live)
            print(json.dumps(live), flush=True)
            code = process.wait()
        receipt = dict(**live, exit_code=code, end_epoch=time.time(),
                       log_sha256=sha(log), status="COMPLETED" if code == 0 else "FAILED")
        receipts.append(receipt)
        save(output / "jobs.json", receipts)
        save(output / "CURRENT_JOB.json", receipt)
        if code:
            raise SystemExit(f"case failed ({code}): {log}")
        check_completed(target, case, plan)
        summary = read(target / "summary.json")
        print(json.dumps(dict(completed=case["name"], wall_s=summary["wall_s"],
                              lane_steps=summary["successful_lane_steps"])), flush=True)
    completed = dict(schema="live-gpu-packet-campaign-completion-v1", phase=phase,
        cases=len(cases), source_sha=plan["source_sha"],
        utc=datetime.now(timezone.utc).isoformat())
    save(output / "COMPLETED.json", completed)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("diagnostic", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_campaign(args.phase, args.output, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
