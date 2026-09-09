from copy import deepcopy
import math

import pytest
import torch

from torch_tm_flowpipe.range_requests import evaluate_range_requests
from experiments.live_range_solver.runner import run_case, write_run
from experiments.live_range_solver.verify import load_run, verify_run
from experiments.range_batch_device.common import digest


@pytest.fixture(scope="module")
def evidence():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    failed = False
    def once(rows, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("evidence test: explicit hardware failure")
        return evaluate_range_requests(rows, **kwargs)
    run, events, _ = run_case("van_der_pol", [0, 31], 1, "G", run_id="tamper-fixture",
        diagnostic=True, evaluator=once, hardware_fallback=True)
    assert run["successful_tasks"] == 2
    verify_run(run, events)
    return run, events


def test_unmodified_live_artifact_recomputes(tmp_path, evidence):
    run, events = evidence
    write_run(tmp_path/"positive", run, events)
    assert verify_run(*load_run(tmp_path/"positive"))["successful_tasks"] == 2


@pytest.mark.parametrize("mutation", [
    "swap_returns", "old_generation", "cancelled_commit", "range_endpoint",
    "missing_cpu_fallback", "missing_wait", "missing_transfer", "lane_steps", "offline_as_online",
])
def test_rehashed_semantic_corruption_is_rejected(tmp_path, evidence, mutation):
    run, events = deepcopy(evidence)
    returned = [e for e in events if e["event"] == "return"]
    if mutation == "swap_returns":
        first = returned[0]
        second = next(e for e in returned if e["task"] != first["task"])
        first["result"], second["result"] = second["result"], first["result"]
        for event in (first, second):
            event["output_digest"] = digest(event["result"])
    elif mutation == "old_generation":
        returned[0]["generation"] -= 1
    elif mutation == "cancelled_commit":
        committed = deepcopy(next(e for e in events if e["event"] == "commit"))
        committed.update(event="cancel", ns=committed["ns"]-1)
        events.append(committed)
    elif mutation == "range_endpoint":
        result = returned[0]["result"]
        result["hi"]["values"][0] = (float.fromhex(result["hi"]["values"][0])+1.).hex()
        returned[0]["output_digest"] = digest(result)
    elif mutation == "missing_cpu_fallback":
        run["counts"]["hardware_fallback_requests"] = 0
    elif mutation == "missing_wait":
        run["wait_ns"][0] = 0
    elif mutation == "missing_transfer":
        group = next(g for g in run["groups"] if g["timing"].get("h2d_and_structure_s"))
        group["timing"]["h2d_and_structure_s"] = 0
    elif mutation == "lane_steps":
        run["successful_lane_steps"] += 1
        run["throughput"] = run["successful_lane_steps"]/run["wall_s"]
    elif mutation == "offline_as_online":
        run["request_source"] = "RECORDED_REQUEST_CORPUS"
        run["execution"] = "ONLINE_LIVE_SOLVE"
    # All outer manifests and summaries are genuinely regenerated, so a checksum
    # failure cannot masquerade as rejection of the semantic mutation.
    write_run(tmp_path/mutation, run, events)
    loaded = load_run(tmp_path/mutation)
    with pytest.raises(AssertionError):
        verify_run(*loaded)
