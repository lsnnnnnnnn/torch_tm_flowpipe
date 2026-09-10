from collections import Counter

import pytest
import torch

from experiments.live_range_solver.runner import run_case
from experiments.range_batch_device.common import request_from_record, result_from_record
from experiments.range_batch_device.oracle import check


@pytest.fixture(scope="module", autouse=True)
def fixed_threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


@pytest.mark.parametrize("plant", ["van_der_pol", "brusselator"])
@pytest.mark.parametrize("serial,online", [("S", "Q"), ("S_gpu", "G")])
def test_complete_two_step_same_backend_state_and_live_dependencies(plant, serial, online):
    expected, _, _ = run_case(plant, [0, 31], 2, serial, diagnostic=True, run_id="serial")
    actual, events, _ = run_case(plant, [31, 0], 2, online, diagnostic=True, run_id="online",
                                   delays={"31": .0001})
    assert actual["successful_tasks"] == expected["successful_tasks"] == 2
    assert actual["accepted_lane_steps"] == expected["accepted_lane_steps"] == 4
    for task, rows in actual["records"].items():
        assert [r["segment"] for r in rows] == [r["segment"] for r in expected["records"][task]]
        assert rows[1]["before"] == rows[0]["after"]
        assert all(r["request_counter"] > 0 for r in rows)
    inputs, consumed, waiting = {}, {}, {}
    counts = Counter()
    for event in events:
        kind = event["event"]
        if kind == "submit":
            task = event["task"]
            assert task not in waiting
            assert event["previous"] == consumed.get(task)
            waiting[task] = event["request_id"]
            inputs[event["request_id"]] = request_from_record(event["request"])
        elif kind == "return":
            result = result_from_record(event["result"])
            counts.update(check(inputs[event["request_id"]], result))
        elif kind == "consume":
            assert waiting.pop(event["task"]) == event["request_id"]
            consumed[event["task"]] = event["request_id"]
    assert not waiting and counts["terms"] > 0


@pytest.mark.parametrize("plant", ["van_der_pol", "brusselator"])
def test_packet_gpu_matches_parent_gpu_for_complete_state_under_two_legal_packet_orders(plant):
    parent, _, _ = run_case(plant, [0, 31], 2, "G0", diagnostic=True,
                            run_id="parent-gpu")
    packet, events, _ = run_case(plant, [31, 0], 2, "Gp", diagnostic=True,
        run_id="packet-gpu", delays={"31": .0001})
    split, split_events, _ = run_case(plant, [0, 31], 2, "Gp", diagnostic=True,
                                      run_id="packet-gpu-split", max_group=1)
    for actual in (packet, split):
        assert actual["successful_tasks"] == parent["successful_tasks"] == 2
        assert actual["accepted_lane_steps"] == parent["accepted_lane_steps"] == 4
        for task, rows in actual["records"].items():
            expected = parent["records"][task]
            assert [row["segment"] for row in rows] == [row["segment"] for row in expected]
            assert rows[1]["before"] == rows[0]["after"]

    def sequence(source):
        values = {}
        for event in source:
            if event["event"] != "submit":
                continue
            request = dict(event["request"])
            request.pop("request_id")
            values.setdefault(event["task"], []).append((event["generation"], event["attempt"],
                event["counter"], event["source_state"], request))
        return values

    assert sequence(events) == sequence(split_events)
    assert any(group["selection"]["selected_keys"] > 1 for group in packet["groups"])
    assert all(group["selection"]["selected_requests"] == 1 for group in split["groups"])
