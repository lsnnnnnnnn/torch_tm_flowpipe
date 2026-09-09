"""Finite preregistered experiment matrix, launched as isolated sequential jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.range_batch_device.common import read, save, sha
from .runner import ROOT, PARTITION


def matrix(phase):
    cases = []
    def add(plant, route, batch, steps, label, **options):
        cases.append(dict(name=f"{label}-{plant}-{route}", plant=plant, route=route,
                          batch=batch, steps=steps, options=options))
    plants = ("van_der_pol", "brusselator")
    if phase == "diagnostic":
        for batch in (1, 2):
            for plant in plants:
                for route in ("S", "Q", "S_gpu", "G"):
                    add(plant, route, batch, 2, f"small-b{batch}")
        for batch in (8, 32):
            for plant in plants:
                for route in ("S", "Q", "S_gpu", "G"):
                    add(plant, route, batch, 2 if batch==32 and route=="S_gpu" else 20, f"prefix-b{batch}")
        for plant in plants:
            for route in ("S", "Q", "G"):
                add(plant, route, 2, 120, "long-b2")
        for plant in plants:
            before = "van_der_pol_before0099" if plant=="van_der_pol" else "brusselator_before0999"
            checkpoint = f"artifacts/runs/boundary_execution_20260908T172756Z/raw_minimal/state_inputs/{before}"
            for route in ("S", "Q", "S_gpu", "G"):
                add(plant, route, 1, 3, "history", checkpoint=checkpoint)
            for route in ("S", "G"):
                add(plant, route, 1, 20, "original", original=True)
        ids = list(range(0, 32, 4))
        for plant in plants:
            for route in ("Q", "G"):
                for size in (4, 2, 1):
                    for offset in range(0, 8, size):
                        add(plant, route, 1, 2, f"split-{size}-part{offset//size}", ids=ids[offset:offset+size])
                add(plant, route, 8, 2, "permuted-delayed-newest", ids=ids[::-1], delay_task="0", newest_first=True)
    elif phase == "formal":
        for batch in (1, 8, 32):
            for repetition, order in enumerate((("S", "Q", "G"), ("G", "Q", "S"), ("Q", "S", "G"))):
                for plant in plants:
                    for route in order:
                        add(plant, route, batch, 2 if batch==1 else 20, f"formal-b{batch}-rep{repetition}")
            for plant in plants:
                add(plant, "L", batch, 2 if batch==1 else 20, f"legacy-b{batch}")
    else:
        raise ValueError(phase)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["diagnostic", "formal"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    cases = matrix(args.phase)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise SystemExit("campaign requires a committed clean source tree")
    identity = {str(p.relative_to(ROOT)):sha(p) for folder in (ROOT/"src/torch_tm_flowpipe", ROOT/"experiments/live_range_solver")
                for p in sorted(folder.glob("*")) if p.suffix in {".py", ".cu"}}
    plan = dict(phase=args.phase, cases=cases, source_sha=head, scientific_sources=identity,
                partition_reference=str(PARTITION.relative_to(ROOT)), partition_sha256=sha(PARTITION),
                cpu_affinity=[2], intra_op=1, inter_op=1, cuda_visible_devices="0", max_wait_s=.020, max_group=32)
    manifest = output/"CAMPAIGN_PLAN.json"
    if manifest.exists():
        old = read(manifest)
        assert old["scientific_sources"] == identity and old["cases"] == cases, "resume changed frozen experiment"
        plan = old
    else:
        save(manifest, plan)
    if args.phase == "formal":
        gate = output.parent/"diagnostic"/"CORRECTNESS_GATE.json"
        assert gate.exists() and read(gate)["passed"] is True, "correctness must precede formal timing"
    env = dict(os.environ, PYTHONPATH="src:.:tests", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0")
    receipts = read(output/"jobs.json") if (output/"jobs.json").exists() else []
    for case in cases:
        target = output/case["name"]
        if (target/"summary.json").exists() and args.resume:
            from .verify import load_run
            run, _ = load_run(target)
            assert run["successful_tasks"] == len(run["ids"])
            continue
        if target.exists():
            raise SystemExit(f"incomplete existing output requires inspection: {target}")
        command = ["taskset", "-c", "2", sys.executable, "-m", "experiments.live_range_solver.runner",
            "--plant", case["plant"], "--route", case["route"], "--batch", str(case["batch"]),
            "--steps", str(case["steps"]), "--output", str(target), "--max-wait-s", ".020", "--warm"]
        if args.phase == "diagnostic":
            command.append("--diagnostic")
        for key, value in case["options"].items():
            option = "--"+key.replace("_", "-")
            command.append(option)
            if not isinstance(value, bool):
                command.append(",".join(map(str,value)) if key=="ids" else str(value))
        log = output/(case["name"]+".log")
        started = time.time()
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            live = dict(case=case["name"], pid=process.pid, command=command, start_epoch=started,
                        source_sha=plan["source_sha"], status="RUNNING")
            save(output/"CURRENT_JOB.json", live)
            print(json.dumps(live), flush=True)
            code = process.wait()
        receipt = dict(**live, exit_code=code, end_epoch=time.time(), log_sha256=sha(log))
        receipt["status"] = "COMPLETED" if code==0 else "FAILED"
        receipts.append(receipt)
        save(output/"jobs.json", receipts)
        save(output/"CURRENT_JOB.json", receipt)
        if code:
            raise SystemExit(f"case failed ({code}): {log}")
        summary = read(target/"summary.json")
        print(json.dumps(dict(completed=case["name"], wall_s=summary["wall_s"],
                              accepted_lane_steps=summary["accepted_lane_steps"])), flush=True)
    save(output/"COMPLETED.json", dict(phase=args.phase, cases=len(cases), source_sha=plan["source_sha"],
                                       utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
