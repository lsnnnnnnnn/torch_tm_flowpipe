"""Run the two frozen fresh original-box 1000-step GPU-range horizons."""
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
from experiments.live_range_solver.runner import ROOT

from .verify_long_horizon import verify_long_horizon


CASES = (
    dict(name="van_der_pol-Gp", plant="van_der_pol", route="Gp", steps=1000, target="T10"),
    dict(name="brusselator-Gp", plant="brusselator", route="Gp", steps=1000, target="T20"),
)


def run_campaign(artifact, *, resume=False):
    artifact = Path(artifact).resolve()
    global_plan = read(artifact / "PLAN_FROZEN.json")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT, text=True).strip()
    assert source_sha == global_plan["source_sha"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    gate = read(artifact / "diagnostic/CORRECTNESS_GATE.json")
    assert gate["passed"] and gate["source_sha"] == source_sha
    output = artifact / "full_horizon"
    output.mkdir(exist_ok=resume)
    plan_path = output / "CAMPAIGN_PLAN.json"
    plan = dict(schema="live-gpu-packet-long-campaign-plan-v1",
        frozen_before_execution=True, source_sha=source_sha, cases=list(CASES),
        original_unpartitioned_b1=True, previous_answers_loaded=False,
        cpu_periodic_correction=False, cpu_affinity=[2], cuda_visible_devices="0")
    if plan_path.exists():
        assert resume and read(plan_path) == plan
    else:
        save(plan_path, plan)
    env = dict(os.environ, PYTHONPATH="src:.:tests", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0")
    receipts = read(output / "jobs.json") if (output / "jobs.json").exists() else []
    for case in CASES:
        target = output / case["name"]
        verify_path = output / f"{case['name']}-verification.json"
        if (target / "LONG_HORIZON_RESULT.json").exists() and resume:
            verification = verify_long_horizon(target, require_achieved=True)
            if verify_path.exists():
                assert read(verify_path) == verification
            else:
                save(verify_path, verification)
            continue
        if target.exists():
            raise SystemExit(f"inspect incomplete horizon before resume: {target}")
        command = ["taskset", "-c", "2", sys.executable, "-m",
            "experiments.live_gpu_packets.long_horizon", "--plant", case["plant"],
            "--route", case["route"], "--steps", str(case["steps"]),
            "--output", str(target)]
        log = output / f"{case['name']}.log"
        started = time.time()
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT)
            current = dict(case=case["name"], pid=process.pid, command=command,
                source_sha=source_sha, start_epoch=started, status="RUNNING")
            save(output / "CURRENT_JOB.json", current)
            print(json.dumps(current), flush=True)
            code = process.wait()
        receipt = dict(**current, exit_code=code, end_epoch=time.time(),
                       log_sha256=sha(log), status="COMPLETED" if code == 0 else "FAILED")
        receipts.append(receipt)
        save(output / "jobs.json", receipts)
        save(output / "CURRENT_JOB.json", receipt)
        if code:
            raise SystemExit(f"long horizon failed ({code}): {log}")
        verification = verify_long_horizon(target, require_achieved=True)
        save(verify_path, verification)
        print(json.dumps(verification), flush=True)
    completed = dict(schema="live-gpu-packet-long-campaign-completion-v1",
        cases=2, source_sha=source_sha, utc=datetime.now(timezone.utc).isoformat())
    save(output / "COMPLETED.json", completed)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_campaign(args.artifact, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
