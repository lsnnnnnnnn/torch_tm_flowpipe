"""Run the two frozen fresh original-box 1000-step GPU-range horizons."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from experiments.range_batch_device.common import read, save, sha
from experiments.live_range_solver.runner import ROOT

from .verify_long_horizon import QUEUE_OWNER_SCHEMAS, verify_long_horizon


CASES = (
    dict(name="van_der_pol-Gp", plant="van_der_pol", route="Gp", steps=1000, target="T10"),
    dict(name="brusselator-Gp", plant="brusselator", route="Gp", steps=1000, target="T20"),
)

VERIFIER_AMENDMENT_CHANGED_FILES = (
    "experiments/live_gpu_packets/long_campaign.py",
    "experiments/live_gpu_packets/package.py",
    "experiments/live_gpu_packets/verify_long_horizon.py",
    "experiments/live_gpu_packets/verify_package.py",
    "tests/test_long_horizon_packet.py",
)
INTERMEDIATE_VERIFIER_SHA = "391c33b7a3d40c092b074aa96e3c3fe09599316e"


def _git_blob(commit, relative):
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    assert suites
    return {name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
            for name in ("tests", "failures", "errors", "skipped")}


def _bridge_identity(global_plan):
    runtime_sha = global_plan["source_sha"]
    verification_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert verification_sha != runtime_sha
    subprocess.run(["git", "merge-base", "--is-ancestor", INTERMEDIATE_VERIFIER_SHA,
                    verification_sha], cwd=ROOT, check=True)
    assert not subprocess.check_output(["git", "status", "--porcelain"],
                                       cwd=ROOT, text=True)
    changed = tuple(subprocess.check_output(
        ["git", "diff", "--name-only", runtime_sha, verification_sha],
        cwd=ROOT, text=True).splitlines())
    assert set(changed) == set(VERIFIER_AMENDMENT_CHANGED_FILES)
    records = {}
    for relative in VERIFIER_AMENDMENT_CHANGED_FILES:
        before = _git_blob(runtime_sha, relative)
        after = (ROOT / relative).read_bytes()
        assert before != after
        records[relative] = dict(old_sha256=_digest(before), new_sha256=_digest(after))
    unchanged = 0
    for relative, checksum in global_plan["scientific_sources"].items():
        assert _digest(_git_blob(runtime_sha, relative)) == checksum
        if relative not in records:
            assert sha(ROOT / relative) == checksum
            unchanged += 1
    return verification_sha, records, unchanged


def verify_existing_campaign(artifact):
    """Bridge completed runtime outputs after a verifier-only schema correction."""
    artifact = Path(artifact).resolve()
    global_plan = read(artifact / "PLAN_FROZEN.json")
    verification_sha, changed_files, unchanged = _bridge_identity(global_plan)
    runtime_sha = global_plan["source_sha"]
    gate = read(artifact / "diagnostic/CORRECTNESS_GATE.json")
    assert gate["passed"] and gate["source_sha"] == runtime_sha
    output = artifact / "full_horizon"
    plan = read(output / "CAMPAIGN_PLAN.json")
    assert plan["source_sha"] == runtime_sha and plan["cases"] == list(CASES)
    jobs = read(output / "jobs.json")
    assert len(jobs) == len(CASES)

    xml_path = artifact / "tests/verifier_amendment.xml"
    test_counts = _junit_counts(xml_path)
    assert test_counts["tests"] == 2
    assert test_counts["failures"] == test_counts["errors"] == test_counts["skipped"] == 0

    verifications = []
    for case, job in zip(CASES, jobs):
        assert job["case"] == case["name"] and job["status"] == "COMPLETED"
        assert job["exit_code"] == 0 and job["source_sha"] == runtime_sha
        assert sha(output / f"{case['name']}.log") == job["log_sha256"]
        target = output / case["name"]
        result = read(target / "LONG_HORIZON_RESULT.json")
        assert result["source_sha"] == runtime_sha and result["achieved"] is True
        verification = verify_long_horizon(target, require_achieved=True)
        assert verification["queue_owner_schema"] == QUEUE_OWNER_SCHEMAS[case["plant"]]
        save(output / f"{case['name']}-verification.json", verification)
        verifications.append(verification)

    amendment = dict(schema="live-gpu-packet-verifier-amendment-v1",
        runtime_source_sha=runtime_sha, verification_source_sha=verification_sha,
        frozen_plan_unchanged=True, runtime_outputs_reused=True,
        runtime_jobs_reexecuted=0, long_runs_verified=len(verifications),
        reason=("the frozen verifier applied the VDP c3 owner schema to both plants; "
                "Brusselator's frozen runtime contract uses accepted-boundary ownership"),
        original_failure=dict(stage="post-run offline long-horizon verification",
            wrapper_exit_code=1, computation_exit_code=0,
            frozen_unconditional_expected_schema="c3_cross_step_sr_v1",
            observed_brusselator_schema="accepted_boundary_sr_v1",
            observed_brusselator_rows=1000),
        intermediate_bridge_attempt=dict(verification_source_sha=INTERMEDIATE_VERIFIER_SHA,
            stage="amendment JUnit receipt parsing", wrapper_exit_code=1,
            reason="pytest emitted a testsuites root with counts on its child testsuite",
            amendment_or_completion_written=False),
        corrected_queue_owner_schema_by_plant=QUEUE_OWNER_SCHEMAS,
        changed_files=changed_files,
        unchanged_frozen_scientific_files=unchanged,
        amendment_test=dict(path=str(xml_path.relative_to(artifact)),
            sha256=sha(xml_path), counts=test_counts,
            command=[sys.executable, "-m", "pytest", "-q",
                "tests/test_long_horizon_packet.py",
                f"--junitxml={xml_path.relative_to(ROOT)}"]),
        verified_run_sources=[row["source_sha"] for row in
            [read(output / case["name"] / "LONG_HORIZON_RESULT.json") for case in CASES]],
        utc=datetime.now(timezone.utc).isoformat())
    save(output / "VERIFIER_AMENDMENT.json", amendment)
    completed = dict(schema="live-gpu-packet-long-campaign-completion-v1",
        cases=2, source_sha=runtime_sha, verification_source_sha=verification_sha,
        verifier_amendment="VERIFIER_AMENDMENT.json", bridged_existing_runs=True,
        runtime_jobs_reexecuted=0, utc=datetime.now(timezone.utc).isoformat())
    save(output / "COMPLETED.json", completed)
    return dict(completed=completed, amendment=amendment,
                verifications=verifications)


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
        receipt = dict(current)
        receipt.update(exit_code=code, end_epoch=time.time(), log_sha256=sha(log),
                       status="COMPLETED" if code == 0 else "FAILED")
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
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    assert not (args.resume and args.verify_existing)
    result = (verify_existing_campaign(args.artifact) if args.verify_existing
              else run_campaign(args.artifact, resume=args.resume))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
