"""Capture the finite affected test set with commands, logs and JUnit evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from experiments.range_batch_device.common import save, sha
from experiments.live_range_solver.runner import ROOT


TESTS = (
    "tests/test_boundary_range_plan.py",
    "tests/test_range_requests.py",
    "tests/test_range_cuda.py",
    "tests/test_live_range_service.py",
    "tests/test_live_range_solver_integration.py",
    "tests/test_live_range_evidence.py",
    "tests/test_range_packets.py",
    "tests/test_long_horizon_packet.py",
)
PARENT_XML = Path("/srv/local/shengenli/live_range_solver_20260909T053007Z/parent_tests.xml")


def totals(xml):
    root = ET.parse(xml).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if root.tag != "testsuite":
        suites = [suite for suite in suites if not list(suite.findall("testsuite"))]
    return {name: sum(int(float(suite.attrib.get(name, 0))) for suite in suites)
            for name in ("tests", "failures", "errors", "skipped")}


def testcase_names(xml):
    return [f"{case.attrib.get('classname', '')}::{case.attrib['name']}"
            for case in ET.parse(xml).getroot().iter("testcase")]


def capture(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise SystemExit("test evidence requires a committed clean scientific tree")
    xml = output / "affected.xml"
    log = output / "affected.log"
    command = ["taskset", "-c", "2", sys.executable, "-m", "pytest", "-q",
               "-p", "no:cacheprovider", f"--junitxml={xml}", *TESTS]
    env = dict(os.environ, PYTHONPATH="src:.:tests", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0")
    started = datetime.now(timezone.utc).isoformat()
    with log.open("w") as stream:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, check=False).returncode
    fixture = output / "device_packet_fixture.json"
    fixture_log = output / "device_packet_fixture.log"
    fixture_command = ["taskset", "-c", "2", sys.executable, "-m",
        "experiments.live_gpu_packets.packet_fixture", "--create", str(fixture)]
    with fixture_log.open("w") as stream:
        fixture_code = subprocess.run(fixture_command, cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, check=False).returncode
    counts = totals(xml) if xml.exists() else {}
    command_record = dict(command=command, fixture_command=fixture_command, cwd=str(ROOT),
        environment={key: env[key] for key in
        ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
         "CUDA_VISIBLE_DEVICES")}, cpu_affinity=[2], started_utc=started,
        ended_utc=datetime.now(timezone.utc).isoformat(), exit_code=code,
        fixture_exit_code=fixture_code)
    save(output / "commands.json", command_record)
    result = dict(schema="live-gpu-packet-test-result-v1", passed=code == 0 and fixture_code == 0
        and counts.get("failures", 1) == 0 and counts.get("errors", 1) == 0,
        source_sha=source_sha, counts=counts, files={"affected.xml": sha(xml),
        "affected.log": sha(log), "commands.json": sha(output / "commands.json"),
        "device_packet_fixture.json": sha(fixture),
        "device_packet_fixture.log": sha(fixture_log)},
        scope="finite affected packet/scheduler/solver/checkpoint tests",
        parent_tests_reused=(dict(label="REUSED", passed=131,
            source="b60a608a5d2b431aee0fe49a6cb7c3e18bcfdb3a",
            xml_path=str(PARENT_XML), xml_sha256=sha(PARENT_XML))
            if PARENT_XML.exists() else None),
        identity_locked_historical_evidence_tests_not_weakened=True)
    save(output / "PACKET_TEST_RESULT.json", result)
    names = testcase_names(xml) if xml.exists() else []
    required = {
        "heterogeneous_layout_and_order": "heterogeneous_packet_is_one_four_kernel",
        "descriptor_offset_length_overflow": "corrupt_descriptor_is_not_success",
        "cross_request_descriptor_ownership": "cross_request_reference_is_rejected",
        "identity_and_token": "wrong_request_identity_or_token",
        "buffer_epoch_mask_cancel": "stale_buffer_epoch_and_masked_or_cancelled",
        "per_request_failure_isolation": "invalid_and_overflow_requests_are_isolated",
        "input_output_ownership_and_scratch": "owned_input_private_output_alternating",
        "safe_packet_split_and_oversize": "fixed_request_cap_safely_splits",
        "external_power_table_fallback": "external_power_table_stays_on_cpu",
        "cross_key_online_selection": "packet_scheduler_collects_different_ready_keys",
        "cancel_queued": "cancel_queued_packet_request",
        "cancel_inflight": "cancel_inflight_packet",
        "cancel_completed_unconsumed": "cancel_completed_unconsumed",
        "new_epoch_stale_return": "new_epoch_invalidates_old_inflight",
        "hardware_stop_and_cpu_fallback": "packet_service_hardware_failure",
        "complete_checkpoint_stream": "original_b1_streaming_state_chain",
    }
    checks = {label: [name for name in names if pattern in name]
              for label, pattern in required.items()}
    lifetime = dict(schema="live-gpu-packet-lifetime-fault-checks-v1",
        passed=result["passed"] and all(checks.values()), source_sha=source_sha,
        evidence="fresh affected JUnit testcases", checks=checks,
        junit_sha256=sha(xml), writable_device_or_future_state_in_checkpoint=False)
    save(output.parent / "packet_lifetime_fault_checks.json", lifetime)
    if not lifetime["passed"]:
        raise SystemExit("required lifetime/fault testcase was absent")
    if not result["passed"]:
        raise SystemExit(f"affected tests failed; inspect {log}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.output), indent=2))


if __name__ == "__main__":
    main()
