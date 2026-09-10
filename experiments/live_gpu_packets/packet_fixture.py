"""Create and verify a small heterogeneous packet with real CUDA receipts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import torch

from experiments.live_range_solver.runner import ROOT
from experiments.range_batch_device.common import (
    read, request_from_record, request_record, result_record, save,
)
from experiments.range_batch_device.oracle import check
from torch_tm_flowpipe.range_packets import (
    RangePacketExecutor, build_range_work_packet,
)
from torch_tm_flowpipe.range_requests import RangeRequest


def fixture_requests():
    first_lo = torch.tensor([[1., -2., .5], [0., -0., 3.]], dtype=torch.float64)
    first = RangeRequest("packet-fixture-standard", ((0,), (1,), (3,)),
        first_lo, first_lo.clone(), torch.tensor([-.25], dtype=torch.float64),
        torch.tensor([.75], dtype=torch.float64))
    second_lo = torch.tensor([[1., -1.5, .25, 2.]], dtype=torch.float64)
    second = RangeRequest("packet-fixture-normal", ((0, 0), (2, 1), (1, 2), (4, 0)),
        second_lo, second_lo.clone(), torch.tensor([0., -1.], dtype=torch.float64),
        torch.tensor([.02, 1.], dtype=torch.float64), "normal", (1,), 0)
    return first, second


def create_fixture(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = fixture_requests()
    packet = build_range_work_packet(rows)
    executor = RangePacketExecutor()
    timing = {}
    results = executor.execute_packet(packet, diagnostics=True, timings=timing)
    for row in rows:
        check(row, results[row.request_id])
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    record = dict(schema="live-gpu-packet-device-fixture-v1", source_sha=source_sha,
        requests=[request_record(row) for row in rows],
        requirements=asdict(packet.requirements),
        metadata=[int(value) for value in packet.metadata.tolist()],
        numeric_hex=[float(value).hex() for value in packet.numeric.tolist()],
        results={name: result_record(value) for name, value in results.items()},
        execution=dict(buffer_epoch=timing["buffer_epoch"],
            kernel_receipt=timing["kernel_receipt"],
            actual_kernel_invocations=timing["actual_kernel_invocations"],
            h2d_copy_operations=timing["h2d_copy_operations"],
            d2h_copy_operations=timing["d2h_copy_operations"],
            packet_requirements=timing["packet_requirements"]),
        module_build=executor.module.build,
        purpose="independently rebuild offsets/ownership and confirm real four-kernel CUDA output")
    save(output, record)
    return record


def verify_fixture(path, *, execute=True):
    record = read(path)
    assert record["schema"] == "live-gpu-packet-device-fixture-v1"
    rows = tuple(request_from_record(row) for row in record["requests"])
    assert tuple(row.request_id for row in rows) == (
        "packet-fixture-standard", "packet-fixture-normal")
    packet = build_range_work_packet(rows)
    assert asdict(packet.requirements) == record["requirements"]
    assert packet.metadata.tolist() == record["metadata"], "packet descriptor/offset changed"
    assert [float(value).hex() for value in packet.numeric.tolist()] == record["numeric_hex"]
    execution = record["execution"]
    assert execution["kernel_receipt"] == [1, 1, 1, 1]
    assert execution["actual_kernel_invocations"] == 4
    assert execution["h2d_copy_operations"] == execution["d2h_copy_operations"] == 2
    assert execution["packet_requirements"] == record["requirements"]
    assert execution["buffer_epoch"] > 0
    checked = 0
    if execute:
        timing = {}
        actual = RangePacketExecutor().execute_packet(packet, diagnostics=True, timings=timing)
        assert timing["kernel_receipt"] == [1, 1, 1, 1]
        assert timing["actual_kernel_invocations"] == 4
        assert {name: result_record(value) for name, value in actual.items()} == record["results"]
        for row in rows:
            check(row, actual[row.request_id])
            checked += 1
    return dict(verified=True, actual_cuda_execution=execute,
                requests=len(rows), fraction_checked_requests=checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    if args.create:
        record = create_fixture(args.create)
        result = dict(created=True, source_sha=record["source_sha"],
            requests=len(record["requests"]), requirements=record["requirements"],
            execution=record["execution"])
    else:
        result = verify_fixture(args.verify, execute=not args.no_execute)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
