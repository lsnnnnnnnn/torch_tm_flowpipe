from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
import math
import time

import pytest
import torch

from experiments.range_batch_device.oracle import check
from test_range_requests import bits, requests
from torch_tm_flowpipe import Interval
from torch_tm_flowpipe.live_range_service import LiveRangeService, RangeCancelled
from torch_tm_flowpipe.range_packets import (
    DEFAULT_PACKET_LIMITS, H_BUFFER_EPOCH, H_CELL_OFFSET, H_OPERATION_OFFSET,
    H_OUTPUT_OFFSET, H_POWER_OFFSET, H_POWERS,
    H_REQUEST_OFFSET, R_CELL_COUNT, R_CELL_START, R_COEFF_LO, R_OPERATION_START,
    R_OUTPUT_START, R_POWER_START, R_TERMS, R_TOKEN,
    CELL_WIDTH, OUTPUT_WIDTH, POWER_WIDTH, REQUEST_WIDTH,
    RangePacketExecutor, PacketStructureError, build_range_work_packet,
)
from torch_tm_flowpipe.range_requests import RangeRequest, evaluate_range_requests, structure_key


pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module", autouse=True)
def fixed_threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def heterogeneous_rows():
    rows = []
    cases = ((1, 4, "standard", 1), (2, 6, "interval-coefficient", 3),
             (3, 4, "normal", 2), (3, 6, "standard", 4))
    for index, (variables, order, kind, outputs) in enumerate(cases):
        row = requests(1, variables, order, kind, outputs)[0]
        rows.append(replace(row, request_id=f"heterogeneous-{index}"))
    row = rows[0]
    rows.append(replace(row, request_id="same-support-different-order",
        exponents=row.exponents[::-1], coefficients_lo=row.coefficients_lo.flip(-1),
        coefficients_hi=row.coefficients_hi.flip(-1)))
    return rows


def test_heterogeneous_packet_is_one_four_kernel_submission_and_bitwise_parent_equal():
    rows = heterogeneous_rows()
    executor = RangePacketExecutor()
    timing = {}
    actual = executor.evaluate(rows, diagnostics=True, timings=timing)
    parent = evaluate_range_requests(rows, backend="cuda", diagnostics=True)
    assert len({structure_key(row) for row in rows}) == 5
    for row in rows:
        check(row, actual[row.request_id])
        assert bits(actual[row.request_id]) == bits(parent[row.request_id])
    assert timing["packet_count"] == 1
    assert timing["packet_semantic_key_counts"] == [5]
    assert timing["actual_kernel_invocations"] == 4
    assert timing["kernel_receipts"] == [[1, 1, 1, 1]]
    assert timing["h2d_copy_operations"] == timing["d2h_copy_operations"] == 2
    assert timing["device_kernel_s"] > 0
    assert timing["kernel_and_sync_host_s"] > 0


def test_empty_support_signed_zero_max_power_and_extreme_mix_match_parent():
    empty_coefficients = torch.empty((2, 0), dtype=torch.float64)
    empty = RangeRequest("empty", (), empty_coefficients, empty_coefficients.clone(),
                         torch.tensor([-.0], dtype=torch.float64),
                         torch.tensor([.0], dtype=torch.float64))
    coefficients = torch.tensor([[0., -0., 1., -1.]], dtype=torch.float64)
    tiny = RangeRequest("tiny-and-p64", ((0,), (1,), (2,), (64,)), coefficients,
        coefficients.clone(), torch.tensor([-5e-324], dtype=torch.float64),
        torch.tensor([5e-324], dtype=torch.float64))
    large_coefficients = torch.tensor([[1e-200, -1e-200]], dtype=torch.float64)
    large = RangeRequest("large-finite", ((64,), (3,)), large_coefficients,
        large_coefficients.clone(), torch.tensor([1e4], dtype=torch.float64),
        torch.tensor([1e4], dtype=torch.float64))
    rows = [empty, tiny, large]
    actual = RangePacketExecutor().evaluate(rows, diagnostics=True)
    parent = evaluate_range_requests(rows, backend="cuda", diagnostics=True)
    for row in rows:
        check(row, actual[row.request_id])
        assert bits(actual[row.request_id]) == bits(parent[row.request_id])
    assert bits(actual["empty"]) == (("0x0.0p+0", "0x0.0p+0"),
                                      ("0x0.0p+0", "0x0.0p+0"))


@pytest.mark.parametrize("damage", ["offset", "length-overflow", "power-index"])
def test_corrupt_descriptor_is_not_success_and_does_not_pollute_healthy_request(damage):
    rows = heterogeneous_rows()[:2]
    executor = RangePacketExecutor()
    packet = build_range_work_packet(rows)
    metadata = packet.metadata.clone()
    request_offset = int(metadata[H_REQUEST_OFFSET])
    if damage == "offset":
        metadata[request_offset + R_COEFF_LO] = packet.numeric.numel() + 1
    elif damage == "length-overflow":
        metadata[request_offset + R_TERMS] = 2**62
    else:
        cell_offset = int(metadata[H_CELL_OFFSET])
        operation_start = int(metadata[cell_offset + 3])
        metadata[int(metadata[H_OPERATION_OFFSET]) + operation_start] = int(metadata[H_POWERS]) + 1
    damaged = replace(packet, metadata=metadata)
    actual = executor.execute_packet(damaged, diagnostics=True)
    assert actual[rows[0].request_id].status == "packet_error"
    healthy = actual[rows[1].request_id]
    baseline = evaluate_range_requests([rows[1]], backend="cuda", diagnostics=True)[rows[1].request_id]
    check(rows[1], healthy)
    assert bits(healthy) == bits(baseline)


@pytest.mark.parametrize("damage", ["cross-numeric", "cross-operation", "cross-power",
                                    "power-owner", "cell-owner", "output-owner"])
def test_cross_request_reference_is_rejected_without_polluting_owner(damage):
    rows = heterogeneous_rows()[:2]
    executor = RangePacketExecutor()
    packet = build_range_work_packet(rows)
    metadata = packet.metadata.clone()
    request_offset = int(metadata[H_REQUEST_OFFSET])
    first = request_offset
    second = request_offset + REQUEST_WIDTH
    if damage == "cross-numeric":
        metadata[first + R_COEFF_LO] = metadata[second + R_COEFF_LO]
    elif damage == "cross-operation":
        cell_offset = int(metadata[H_CELL_OFFSET])
        first_cell = int(metadata[first + R_CELL_START])
        cell_count = int(metadata[first + R_CELL_COUNT])
        cell = next(index for index in range(first_cell, first_cell + cell_count)
                    if int(metadata[cell_offset + index*CELL_WIDTH + 4]) > 0)
        metadata[cell_offset + cell*CELL_WIDTH + 3] = metadata[second + R_OPERATION_START]
    elif damage == "cross-power":
        cell_offset = int(metadata[H_CELL_OFFSET])
        operation_offset = int(metadata[H_OPERATION_OFFSET])
        first_cell = int(metadata[first + R_CELL_START])
        cell_count = int(metadata[first + R_CELL_COUNT])
        positive = next(
            operation
            for cell in range(first_cell, first_cell + cell_count)
            for operation in range(
                int(metadata[cell_offset + cell*CELL_WIDTH + 3]),
                int(metadata[cell_offset + cell*CELL_WIDTH + 3]) +
                int(metadata[cell_offset + cell*CELL_WIDTH + 4]))
            if int(metadata[operation_offset + operation]) >= 0)
        metadata[operation_offset + positive] = metadata[second + R_POWER_START]
    elif damage == "power-owner":
        power = int(metadata[first + R_POWER_START])
        metadata[int(metadata[H_POWER_OFFSET]) + power*POWER_WIDTH] = 1
    elif damage == "cell-owner":
        cell = int(metadata[first + R_CELL_START])
        metadata[int(metadata[H_CELL_OFFSET]) + cell*CELL_WIDTH] = 1
    else:
        output = int(metadata[first + R_OUTPUT_START])
        metadata[int(metadata[H_OUTPUT_OFFSET]) + output*OUTPUT_WIDTH] = 1
    actual = executor.execute_packet(replace(packet, metadata=metadata), diagnostics=True)
    assert actual[rows[0].request_id].status == "packet_error"
    healthy = actual[rows[1].request_id]
    baseline = evaluate_range_requests(
        [rows[1]], backend="cuda", diagnostics=True)[rows[1].request_id]
    check(rows[1], healthy)
    assert bits(healthy) == bits(baseline)


def test_wrong_request_identity_or_token_is_rejected_before_scatter():
    packet = build_range_work_packet(heterogeneous_rows()[:2])
    executor = RangePacketExecutor()
    with pytest.raises(PacketStructureError, match="identity envelope"):
        executor.execute_packet(replace(packet, request_ids=("wrong", packet.request_ids[1])))
    metadata = packet.metadata.clone()
    metadata[int(metadata[H_REQUEST_OFFSET]) + R_TOKEN] += 1
    with pytest.raises(PacketStructureError, match="token"):
        executor.execute_packet(replace(packet, metadata=metadata))


def test_stale_buffer_epoch_and_masked_or_cancelled_rows_cannot_report_success():
    rows = heterogeneous_rows()[:3]
    packet = build_range_work_packet(rows[:1])
    metadata = packet.metadata.clone()
    metadata[H_BUFFER_EPOCH] = 19
    with pytest.raises(PacketStructureError, match="freshly initialized"):
        RangePacketExecutor().execute_packet(replace(packet, metadata=metadata))
    masked = replace(rows[0], request_id="masked", enabled=False)
    cancelled = replace(rows[1], request_id="cancelled", cancelled=True)
    timing = {}
    result = RangePacketExecutor().evaluate([masked, cancelled, rows[2]], timings=timing)
    assert result["masked"].status == "masked"
    assert result["cancelled"].status == "cancelled"
    assert result[rows[2].request_id].ok
    assert timing["packet_request_counts"] == [1]


def test_invalid_and_overflow_requests_are_isolated_inside_one_packet():
    rows = requests(3, 1, 4, outputs=1)
    good = replace(rows[0], request_id="good")
    invalid = replace(rows[1], request_id="invalid",
                      coefficients_lo=torch.full_like(rows[1].coefficients_lo, math.nan))
    overflow = replace(rows[2], request_id="overflow",
        domain_lo=torch.tensor([1e308], dtype=torch.float64),
        domain_hi=torch.tensor([1e308], dtype=torch.float64))
    actual = RangePacketExecutor().evaluate([invalid, overflow, good], diagnostics=True)
    assert actual["invalid"].status == "invalid"
    assert actual["overflow"].status == "overflow"
    baseline = evaluate_range_requests([good], backend="cuda", diagnostics=True)["good"]
    check(good, actual["good"])
    assert bits(actual["good"]) == bits(baseline)


def test_owned_input_private_output_alternating_packets_and_scratch_epochs():
    rows = heterogeneous_rows()
    packet = build_range_work_packet(rows[:2])
    owned_reference = packet.requests[0]
    expected = evaluate_range_requests([owned_reference], backend="cuda")[owned_reference.request_id]
    rows[0].coefficients_lo.add_(99)
    rows[0].coefficients_hi.add_(99)
    executor = RangePacketExecutor()
    first_timing = {}
    first = executor.execute_packet(packet, timings=first_timing)
    assert bits(first[owned_reference.request_id]) == bits(expected)
    saved_other = bits(first[packet.request_ids[1]])
    first[owned_reference.request_id].lo.add_(123)
    assert bits(first[packet.request_ids[1]]) == saved_other

    epochs, allocation_counts = [first_timing["buffer_epoch"]], []
    for count in (5, 1, 4, 2, 5):
        timing = {}
        result = executor.evaluate(rows[:count], timings=timing)
        epochs.extend(row["buffer_epoch"] for row in timing["packet_timings"])
        allocation_counts.append(timing["device_allocation_operations"])
        assert all(value.ok for value in result.values())
    assert epochs == sorted(set(epochs))
    assert allocation_counts[-1] == 0
    before = bits(executor.evaluate(rows[1:2])[rows[1].request_id])
    first[packet.request_ids[1]].hi.sub_(321)
    assert bits(executor.evaluate(rows[1:2])[rows[1].request_id]) == before


def test_fixed_request_cap_safely_splits_and_oversize_uses_explicit_parent_path():
    limits = replace(DEFAULT_PACKET_LIMITS, max_requests=2)
    executor = RangePacketExecutor(limits=limits)
    rows = heterogeneous_rows()
    timing = {}
    actual = executor.evaluate(rows, diagnostics=True, timings=timing)
    assert timing["packet_request_counts"] == [2, 2, 1]
    assert timing["actual_kernel_invocations"] == 12
    for row in rows:
        check(row, actual[row.request_id])

    one_term_limits = replace(DEFAULT_PACKET_LIMITS, max_terms=1)
    small_executor = RangePacketExecutor(limits=one_term_limits)
    row = replace(requests(1, 1, 2, outputs=1)[0], request_id="oversize")
    timing = {}
    actual = small_executor.evaluate([row], timings=timing)[row.request_id]
    parent = evaluate_range_requests([row], backend="cuda")[row.request_id]
    assert timing["packet_count"] == 0 and timing["oversize_parent_requests"] == 1
    assert bits(actual) == bits(parent)


def test_external_power_table_stays_on_cpu_while_healthy_packet_uses_gpu():
    coefficient = torch.tensor([[2.]], dtype=torch.float64)
    table = RangeRequest("external-table", ((2, 1),), coefficient, coefficient.clone(),
        torch.tensor([0., -1.], dtype=torch.float64),
        torch.tensor([.25, 1.], dtype=torch.float64), "normal", (1,), 0,
        {2: Interval(0., .125)})
    healthy = heterogeneous_rows()[0]
    timing = {}
    actual = RangePacketExecutor().evaluate([table, healthy], diagnostics=True, timings=timing)
    assert actual[table.request_id].status == "fallback"
    check(table, actual[table.request_id], require_terms=False, require_powers=False)
    check(healthy, actual[healthy.request_id])
    assert timing["external_fallback_requests"] == 1
    assert timing["packet_count"] == 1 and timing["actual_kernel_invocations"] == 4


def _lane(task, row):
    task.begin_attempt(diagnostic=True)
    result = task.evaluate(row)
    task.commit((result.lo, result.hi))
    task.finish()
    return result


def test_packet_scheduler_collects_different_ready_keys_without_extra_wait():
    rows = heterogeneous_rows()[:3]
    with LiveRangeService("cpu", packet_mode=True,
                          evaluator=evaluate_range_requests, max_wait_s=.5) as service:
        tasks = [service.register(str(index), None) for index in range(3)]
        with ThreadPoolExecutor(3) as pool:
            results = [future.result(timeout=5) for future in
                       [pool.submit(_lane, task, row) for task, row in zip(tasks, rows)]]
    assert all(result.ok for result in results)
    assert len(service.groups) == 1
    group = service.groups[0]
    assert group["size"] == 3 and group["reason"] == "all_waiting"
    assert group["selection"]["ready_requests"] == 3
    assert group["selection"]["ready_keys"] == group["selection"]["selected_keys"] == 3
    scheduler = group["scheduler_costs"]
    assert scheduler["ownership_copy_sum_ns"] > 0
    assert scheduler["future_wait_sum_ns"] > 0
    assert all(begin > 0 and end >= begin
               for begin, end in scheduler["future_wait_intervals_ns"])


def test_cancel_queued_packet_request_never_enters_evaluator():
    entered = Event()
    seen = []

    def recording(rows, **kwargs):
        seen.extend(row.request_id for row in rows)
        entered.set()
        return evaluate_range_requests(rows, **kwargs)

    with LiveRangeService("cpu", packet_mode=True, evaluator=recording,
                          max_wait_s=.5) as service:
        cancelled = service.register("cancelled", None)
        blocker = service.register("blocker", None)
        cancelled.begin_attempt()
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(cancelled.evaluate, heterogeneous_rows()[0])
            deadline = time.monotonic() + 2
            while cancelled.pending is None and time.monotonic() < deadline:
                time.sleep(.001)
            assert cancelled.pending is not None
            cancelled.cancel()
            blocker.finish()
            with pytest.raises(RangeCancelled):
                future.result(timeout=5)
    assert not entered.is_set() and seen == []


def test_cancel_inflight_packet_does_not_stop_healthy_request():
    entered, release = Event(), Event()

    def delayed(rows, **kwargs):
        entered.set()
        assert release.wait(5)
        return evaluate_range_requests(rows, **kwargs)

    rows = heterogeneous_rows()[:2]
    with LiveRangeService("cpu", packet_mode=True, evaluator=delayed,
                          max_wait_s=.5) as service:
        cancelled, healthy = [service.register(name, None) for name in ("cancelled", "healthy")]
        with ThreadPoolExecutor(2) as pool:
            cancelled_future = pool.submit(_lane, cancelled, rows[0])
            healthy_future = pool.submit(_lane, healthy, rows[1])
            assert entered.wait(5)
            cancelled.cancel()
            release.set()
            with pytest.raises(RangeCancelled):
                cancelled_future.result(timeout=5)
            assert healthy_future.result(timeout=5).ok
    assert service.counts["stale_discarded"] == 1


def test_submit_owns_values_before_packet_dispatch_and_caller_mutation():
    row = heterogeneous_rows()[0]
    reference = evaluate_range_requests([row], backend="cpu")[row.request_id]
    with LiveRangeService("cpu", packet_mode=True, evaluator=evaluate_range_requests,
                          max_wait_s=.5) as service:
        task = service.register("owned", None)
        blocker = service.register("blocker", None)
        task.begin_attempt()
        pending = service.submit(task, row)
        row.coefficients_lo.add_(77)
        row.coefficients_hi.add_(77)
        blocker.finish()
        result = pending.future.result(timeout=5)
        assert bits(result) == bits(reference)
        task.cancel()


def test_new_epoch_invalidates_old_inflight_packet_return():
    entered, release = Event(), Event()

    def delayed(rows, **kwargs):
        entered.set()
        assert release.wait(5)
        return evaluate_range_requests(rows, **kwargs)

    row = heterogeneous_rows()[0]
    with LiveRangeService("cpu", packet_mode=True, evaluator=delayed,
                          max_wait_s=.01) as service:
        old = service.register("same", {"accepted": 7})
        old.begin_attempt()
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(old.evaluate, row)
            assert entered.wait(5)
            accepted = old.checkpoint_state()
            fresh = service.register("same", accepted, generation=old.generation)
            release.set()
            with pytest.raises(RangeCancelled):
                future.result(timeout=5)
        assert fresh.epoch == old.epoch + 1
        assert _lane(fresh, row).ok
    assert service.counts["stale_discarded"] == 1


def test_cancel_completed_unconsumed_result_cannot_alias_next_packet():
    rows = heterogeneous_rows()[:2]
    with LiveRangeService("cuda", packet_mode=True, max_wait_s=.01) as service:
        abandoned = service.register("abandoned", None)
        abandoned.begin_attempt()
        pending = service.submit(abandoned, rows[0])
        assert pending.future.result(timeout=10).ok
        assert abandoned.pending is pending
        abandoned.cancel()
        healthy = service.register("healthy", None)
        result = _lane(healthy, rows[1])
        saved = bits(result)
        pending.future.result().lo.add_(999)
        assert bits(result) == saved
    epochs = [packet["buffer_epoch"] for group in service.groups
              for packet in group["timing"].get("packet_timings", [])]
    assert epochs == sorted(set(epochs)) and len(epochs) >= 2


@pytest.mark.parametrize("fallback", [False, True])
def test_packet_service_hardware_failure_stops_or_explicitly_recomputes(fallback):
    def broken(rows, **kwargs):
        raise RuntimeError("injected packet hardware failure")

    rows = heterogeneous_rows()[:2]
    with LiveRangeService("cuda", packet_mode=True, evaluator=broken,
                          hardware_fallback=fallback, max_wait_s=.1) as service:
        tasks = [service.register(str(index), None) for index in range(2)]
        for task in tasks:
            task.begin_attempt(diagnostic=True)
        with ThreadPoolExecutor(2) as pool:
            pending = [pool.submit(task.evaluate, row) for task, row in zip(tasks, rows)]
            results = [future.result(timeout=10) for future in pending]
        for task, result in zip(tasks, results):
            if fallback:
                assert result.ok
                task.commit(result)
                task.finish()
            else:
                assert result.status == "backend_error"
                task.reject(failed=True)
    if fallback:
        assert service.counts["hardware_fallback_requests"] == 2
        assert service.groups[0]["timing"]["failed_cuda_s"] > 0
        assert service.groups[0]["timing"]["cpu_recompute"]["total_s"] > 0
    else:
        assert service.counts["gpu_completed_requests"] == 0
