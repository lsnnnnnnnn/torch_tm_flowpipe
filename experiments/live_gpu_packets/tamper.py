"""Rehash nine targeted corruptions and require semantic package rejection."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from experiments.range_batch_device.common import read, sha
from torch_tm_flowpipe.range_packets import H_REQUEST_OFFSET, R_COEFF_LO

from .package import write_manifest


SCENARIOS = (
    ("packet_offset", "packet"),
    ("request_identity", "packet"),
    ("generation", "diagnostic"),
    ("buffer_epoch", "diagnostic"),
    ("result_endpoint", "horizon"),
    ("gpu_receipt", "packet"),
    ("timing_denominator", "formal"),
    ("successful_lane_steps", "formal"),
    ("complete_horizon_flag", "horizon"),
)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tampered")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_gzip_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tampered")
    with gzip.open(temporary, "wt", compresslevel=6) as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def atomic_gzip_rows(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tampered")
    with gzip.open(temporary, "wt", compresslevel=6) as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def update_test_fixture(root, mutation):
    fixture_path = root / "tests/device_packet_fixture.json"
    fixture = read(fixture_path)
    mutation(fixture)
    atomic_json(fixture_path, fixture)
    result_path = root / "tests/PACKET_TEST_RESULT.json"
    result = read(result_path)
    result["files"][fixture_path.name] = sha(fixture_path)
    atomic_json(result_path, result)


def update_lifecycle(root, mutation):
    folder = root / "diagnostic/small-b2-van_der_pol-Gp"
    path = folder / "lifecycle.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    mutation(rows)
    atomic_gzip_rows(path, rows)
    summary_path = folder / "summary.json"
    summary = read(summary_path)
    summary["files"][path.name] = sha(path)
    atomic_json(summary_path, summary)


def update_formal_run(root, mutation):
    folder = root / "formal/block0-van_der_pol-Gp"
    path = folder / "run.json.gz"
    with gzip.open(path, "rt") as stream:
        run = json.load(stream)
    mutation(run)
    atomic_gzip_json(path, run)
    summary_path = folder / "summary.json"
    summary = read(summary_path)
    for key, value in run.items():
        if key not in {"groups", "wait_ns", "records"}:
            summary[key] = value
    summary["files"][path.name] = sha(path)
    atomic_json(summary_path, summary)


def mutate(root, scenario):
    if scenario == "packet_offset":
        def damage(record):
            request_offset = record["metadata"][H_REQUEST_OFFSET]
            record["metadata"][request_offset + R_COEFF_LO] += 1
        update_test_fixture(root, damage)
    elif scenario == "request_identity":
        update_test_fixture(root, lambda record:
            record["requests"][0].__setitem__("request_id", "rehash-wrong-request"))
    elif scenario == "gpu_receipt":
        update_test_fixture(root, lambda record:
            record["execution"].__setitem__("kernel_receipt", [1, 1, 1, 0]))
    elif scenario == "generation":
        def damage(rows):
            event = next(row for row in rows if row["event"] == "return")
            event["generation"] -= 1
        update_lifecycle(root, damage)
    elif scenario == "buffer_epoch":
        def damage(rows):
            event = next(row for row in rows if row["event"] == "dispatch")
            event["epoch"] += 1
        update_lifecycle(root, damage)
    elif scenario == "timing_denominator":
        update_formal_run(root, lambda run: run.__setitem__("wall_s", run["wall_s"] + .25))
    elif scenario == "successful_lane_steps":
        update_formal_run(root, lambda run:
            run.__setitem__("successful_lane_steps", run["successful_lane_steps"] - 1))
    elif scenario == "result_endpoint":
        run = root / "full_horizon/van_der_pol-Gp"
        steps = run / "full_horizon_steps.jsonl.gz"
        temporary = steps.with_name(steps.name + ".tampered")
        with gzip.open(steps, "rt") as source, gzip.open(
                temporary, "wt", compresslevel=6) as target:
            first = json.loads(next(source))
            old = float.fromhex(first["bounds"]["endpoint"][0][1])
            first["bounds"]["endpoint"][0][1] = math.nextafter(old, math.inf).hex()
            target.write(json.dumps(first, separators=(",", ":"), allow_nan=False) + "\n")
            shutil.copyfileobj(source, target)
        temporary.replace(steps)
        result_path = run / "LONG_HORIZON_RESULT.json"
        result = read(result_path)
        result["files"]["steps"] = sha(steps)
        atomic_json(result_path, result)
    elif scenario == "complete_horizon_flag":
        result_path = root / "full_horizon/van_der_pol-Gp/LONG_HORIZON_RESULT.json"
        result = read(result_path)
        result["achieved"] = False
        atomic_json(result_path, result)
    else:
        raise ValueError(scenario)


def run_tamper(root):
    root = Path(root).resolve()
    if not (root / "SHA256SUMS").exists():
        raise FileNotFoundError("package SHA256SUMS must exist before tamper checks")
    records, logs = [], {}
    with tempfile.TemporaryDirectory(prefix="live-gpu-packet-tamper-") as temporary:
        temporary = Path(temporary)
        for scenario, focus in SCENARIOS:
            clone = temporary / scenario
            shutil.copytree(root, clone, copy_function=os.link, symlinks=True)
            mutate(clone, scenario)
            write_manifest(clone)
            command = ["taskset", "-c", "2", sys.executable, "-m",
                "experiments.live_gpu_packets.verify_package", str(clone),
                "--no-fresh-execution", "--focus", focus]
            process = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            if process.returncode == 0:
                raise AssertionError(f"semantic verifier accepted {scenario}")
            log_name = f"{scenario}.log"
            logs[log_name] = process.stdout
            records.append(dict(scenario=scenario, focus=focus, command=command,
                exit_code=process.returncode, rejected=True,
                immutable_manifest_regenerated_after_mutation=True,
                rejection_log_sha256=None))
    output = root / "tamper"
    output.mkdir(parents=True, exist_ok=False)
    for record in records:
        name = f"{record['scenario']}.log"
        (output / name).write_text(logs[name])
        record["rejection_log_sha256"] = sha(output / name)
    result = dict(schema="live-gpu-packet-tamper-result-v1", passed=True,
        checks=len(records), outer_manifest_regenerated=True, records=records)
    atomic_json(output / "TAMPER_RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_tamper(args.root), indent=2), flush=True)


if __name__ == "__main__":
    main()
