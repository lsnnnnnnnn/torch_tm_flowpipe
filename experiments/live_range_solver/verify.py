"""Recompute live provenance, state continuity, arithmetic and timing decisions.

Checks below operate on raw values and independent state transitions, after
outer file hashes have been checked. A newly hashed corrupt artifact still fails.
"""
from collections import Counter
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch
import torch_tm_flowpipe as core

from torch_tm_flowpipe.live_range_service import RequestIdentity
from torch_tm_flowpipe.range_requests import evaluate_range_requests, structure_key
from experiments.range_batch_device.common import (
    digest, read, sha, request_from_record, result_from_record, result_record,
)
from experiments.range_batch_device.oracle import check


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def decode_observer_model(value):
    """Decode only the four mathematical types needed for the common observer."""
    if isinstance(value, list):
        return [decode_observer_model(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "tensor" in value:
        assert value["tensor"] == "torch.float64"
        return torch.tensor([float.fromhex(v) for v in value["values"]], dtype=torch.float64).reshape(value["shape"])
    if "float_hex" in value:
        return float.fromhex(value["float_hex"])
    if "items" in value:
        return {tuple(k) if isinstance(k, list) else k:decode_observer_model(v) for k,v in value["items"]}
    name = value["type"]
    assert name in {"Interval", "Polynomial", "TaylorModel", "TMVector"}
    fields = {k:decode_observer_model(v) for k,v in value["fields"].items()}
    if name == "Polynomial":
        result = core.Polynomial({}, fields["n_vars"])
        result.terms.update(fields["terms"])
        return result
    return getattr(core, name)(**fields)


def check_timing(run):
    assert run["end_ns"] > run["start_ns"]
    assert close(run["wall_s"], (run["end_ns"] - run["start_ns"])/1e9), "wall time omitted work"
    assert close(run["cpu_s"], (run["cpu_end_ns"] - run["cpu_start_ns"])/1e9)
    assert run["affinity"] == [2], "main comparison changed CPU budget"
    assert run["torch_threads"] == run["torch_interop_threads"] == 1
    last = run["start_ns"]
    group_ids = []
    fallback = gpu = 0
    service_seconds = 0.
    for group in run["groups"]:
        assert last <= group["start_ns"] <= group["end_ns"] <= run["end_ns"]
        last = group["end_ns"]
        elapsed = (group["end_ns"] - group["start_ns"])/1e9
        service_seconds += elapsed
        assert group["size"] == len(group["request_ids"])
        assert 1 <= group["size"] <= run["max_group"]
        assert group["reason"] in {"max_group", "timeout", "all_waiting", "serial"}
        group_ids.extend(group["request_ids"])
        scheduler = group.get("scheduler_costs")
        if run["route"] in {"Q", "G", "G0", "Gp"}:
            assert scheduler is not None
            for name in ("ownership_copy_sum_ns", "submit_lock_wait_sum_ns",
                         "deliver_lock_wait_sum_ns", "future_set_sum_ns",
                         "consume_lock_wait_sum_ns", "future_wait_sum_ns"):
                assert scheduler[name] >= 0
            for name in ("ownership_copy_intervals_ns", "submit_lock_intervals_ns",
                         "future_wait_intervals_ns"):
                assert len(scheduler[name]) == group["size"]
                assert all(run["start_ns"] <= begin <= end <= run["end_ns"]
                           for begin, end in scheduler[name])
        timing = group["timing"]
        if not timing:
            continue  # Explicitly failed backend has no successful kernel timing.
        if group["hardware_fallback"]:
            fallback += group["size"]
            assert timing["failed_cuda_s"] > 0 and "cpu_recompute" in timing
            inner = timing["cpu_recompute"]
            assert timing["failed_cuda_s"] + inner["total_s"] <= elapsed + 1e-6
            continue
        required = ("grouping_and_packing_s", "compute_and_transfers_s", "scatter_and_wrap_s", "fallback_s")
        assert all(timing[key] >= 0 for key in required)
        assert sum(timing[key] for key in required) <= timing["total_s"] + 1e-6
        assert timing["total_s"] <= elapsed + 1e-6
        if group["backend"] == "cuda":
            groups = timing["group_sizes"]
            assert timing.get("actual_kernel_invocations", 0) == 4*len(groups)
            assert timing.get("kernel_receipts", []) == [[1, 1, 1, 1]]*len(groups)
            if groups:
                fields = ("h2d_and_structure_s", "kernel_and_sync_s", "d2h_and_checks_s")
                assert all(timing[key] > 0 for key in fields), "missing transfer or completion time"
                assert sum(timing[key] for key in fields) <= timing["compute_and_transfers_s"] + 1e-6
            if run["route"] in {"G", "G0", "Gp"}:
                assert group["completion_event"] and group["completion_confirmed"]
            gpu += sum(groups)
    assert len(group_ids) == len(set(group_ids)), "request dispatched twice"
    assert fallback == run["counts"].get("hardware_fallback_requests", 0), "hardware fallback missing"
    if run["successful_tasks"] == len(run["ids"]):
        counts = run["counts"]
        assert counts.get("returned",0) == counts.get("submitted",0), "successful task lost a range response"
        assert sum(counts.get("status_"+status,0) for status in ("ok","corrected","fallback","fallback_corrected")) == counts.get("returned",0)
        assert counts.get("gpu_completed_requests",0) == gpu, "GPU completion count differs from kernel work"
    if run["route"] in {"Q", "G", "G0", "Gp"}:
        assert run["counts"]["groups"] == len(run["groups"])
        assert len(run["wait_ns"]) == len(group_ids), "omitted queued waiting samples"
        assert all(v >= 0 for v in run["wait_ns"])
    else:
        assert not run["wait_ns"]
    return dict(service_s=service_seconds, gpu_dispatched=gpu, dispatched=len(group_ids))


def check_states(run):
    assert set(run["records"]) == {str(i) for i in run["ids"]}
    successes = accepted = completed_steps = 0
    attempt_inputs = {}
    for task, rows in run["records"].items():
        current = None
        accepted_for_task = 0
        for index, row in enumerate(rows):
            assert row["offset"] == index + 1
            if not row["accepted"]:
                assert index == len(rows)-1
                continue
            accepted += 1
            accepted_for_task += 1
            assert row["status"] == "validated"
            assert row["state_step"] == row["generation"]
            if run["diagnostic"]:
                if current is None:
                    current = row["input"]
                    assert current is not None
                before = canonical_digest(current)
                assert row["before"] == before, "next step did not consume its own complete state"
                segment = row["segment"]
                assert canonical_digest(segment) == row["segment_digest"]
                fields = segment["fields"]
                assert fields["status"] == "validated"
                assert fields["h"]["float_hex"] == row["h_hex"]
                assert fields["validation_attempts"] == row["validation_attempts"]
                assert fields["step_rejections"] == row["step_rejections"]
                current = [fields["reset_tm"], fields["flowstar_normal_state"]]
                assert current[1]["fields"]["step_index"] == row["state_step"]
                assert canonical_digest(current) == row["after"], "incomplete next-state payload"
                from .runner import core_range_cpu
                from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution
                with packed_boundary_execution(True), core_range_cpu():
                    observed = {name:[[float(iv.lo).hex(), float(iv.hi).hex()] for iv in
                                      decode_observer_model(fields[field]).range_box()]
                                for name,field in (("endpoint", "endpoint_raw_tm"), ("tube", "tm"))}
                assert observed == row["bounds"], "range table differs from common observer of saved state"
                attempt_inputs[task, row["generation"]-1] = before
        if run["task_statuses"][task] == "FINISHED":
            successes += 1
            assert accepted_for_task == (run["task_steps"] or {}).get(task, run["steps"])
            completed_steps += accepted_for_task
    assert successes == run["successful_tasks"]
    assert accepted == run["accepted_lane_steps"], "inflated accepted lane-steps"
    assert completed_steps == run["successful_lane_steps"], "failed/cancelled work in successful numerator"
    assert close(run["throughput"], completed_steps/run["wall_s"])
    return attempt_inputs


def check_lifecycle(run, events, attempt_inputs, *, recompute=True):
    registered, active, pending, previous, counters = {}, {}, {}, {}, {}
    submitted, returned, groups, raw_requests, dispatches = {}, {}, [], {}, {}
    consumed = set()
    totals = Counter()
    for event in sorted(events, key=lambda e: e["ns"]):
        kind = event["event"]
        if kind == "group":
            groups.append(event["group"])
            continue
        if kind == "service_failure":
            continue
        task, epoch = event["task"], event["epoch"]
        owner = task, epoch
        assert event["run"] == run["run_id"]
        if kind == "register":
            assert epoch == registered.get(task, -1)+1
            registered[task] = epoch
            active[owner] = dict(generation=event["generation"], attempt=0, state="RUNNABLE", source=None)
            counters[owner] = 0
            continue
        assert owner in active
        status = active[owner]
        if kind == "begin":
            assert registered[task] == epoch and status["state"] not in {"CANCELLED", "FINISHED", "FAILED"}
            assert event["generation"] == status["generation"]
            assert event["attempt"] == status["attempt"]+1
            assert owner not in pending
            status.update(attempt=event["attempt"], state="RUNNABLE", source=event["source_state"])
            expected_input = attempt_inputs.get((task, event["generation"]))
            if expected_input is not None:
                assert event["source_state"] == expected_input, "request attempt starts from wrong state"
        elif kind == "submit":
            assert registered[task] == epoch and owner not in pending
            assert status["state"] == "RUNNABLE"
            identity = RequestIdentity(**{key:event[key] for key in ("run", "task", "epoch", "generation", "attempt", "counter")})
            rid = identity.request_id
            assert rid == event["request_id"] and rid not in submitted
            assert event["generation"] == status["generation"] and event["attempt"] == status["attempt"]
            assert event["source_state"] == status["source"]
            assert event["counter"] == counters[owner]+1
            counters[owner] += 1
            assert event["previous"] == previous.get(owner), "request generated before predecessor was consumed"
            pending[owner] = rid
            submitted[rid] = event
            if event.get("request"):
                request = request_from_record(event["request"])
                assert request.request_id == rid
                assert digest(event["request"]) == event["input_digest"]
                assert digest(structure_key(request)) == digest(event["structure"])
                raw_requests[rid] = request
            status["state"] = "WAITING_RANGE"
        elif kind == "dispatch":
            rid = event["request_id"]
            assert pending[owner] == rid and rid in submitted
            assert "dispatch_ns" not in submitted[rid]
            submitted[rid] = dict(submitted[rid], dispatch_ns=event["ns"], group_index=event["group"])
            dispatches[rid] = event
        elif kind == "return":
            rid = event["request_id"]
            assert pending[owner] == rid and rid not in returned, "swapped or duplicate task return"
            assert registered[task] == epoch and status["state"] == "WAITING_RANGE"
            assert event["generation"] == status["generation"] and event["attempt"] == status["attempt"]
            assert event["source_state"] == status["source"]
            assert submitted[rid]["dispatch_ns"] <= event["ns"]
            returned[rid] = event
            totals["returned"] += 1
            totals[f"status_{event['status']}"] += 1
            if event.get("result"):
                assert digest(event["result"]) == event["output_digest"]
                output = result_from_record(event["result"])
                assert output.request_id == rid
                if output.ok:
                    request = raw_requests[rid]
                    fallback = request.step_powers is not None
                    totals.update(check(request, output, require_terms=not fallback, require_powers=not fallback))
        elif kind == "consume":
            rid = event["request_id"]
            assert pending.pop(owner) == rid and rid in returned and rid not in consumed
            assert event["generation"] == status["generation"] and event["attempt"] == status["attempt"]
            assert returned[rid]["ns"] <= event["ns"]
            previous[owner] = rid
            consumed.add(rid)
            status["state"] = "RUNNABLE"
        elif kind == "commit":
            assert registered[task] == epoch and owner not in pending
            assert status["state"] == "RUNNABLE", "cancelled/failed attempt committed"
            assert event["generation"] == status["generation"]+1
            assert event["attempt"] == status["attempt"]
            status.update(generation=event["generation"], state="ACCEPTED")
        elif kind in {"cancel", "failed", "finish", "reject"}:
            if kind != "cancel":
                assert owner not in pending
            else:
                pending.pop(owner, None)
            status["state"] = {"cancel":"CANCELLED", "failed":"FAILED", "finish":"FINISHED", "reject":"REJECTED"}[kind]
        elif kind == "discard":
            assert registered[task] != epoch or owner not in pending or status["state"] in {"CANCELLED", "FINISHED", "FAILED"}
        elif kind == "checkpoint":
            assert owner not in pending
        elif kind == "identity_error":
            assert event["returned_id"] != event["request_id"]
            pending.pop(owner, None)
        else:
            raise AssertionError(f"unknown lifecycle event {kind}")
    assert not pending, "unfinished requests left behind"
    assert len(submitted) == run["counts"].get("submitted", 0)
    for name, count in totals.items():
        if name.startswith("status_") or name == "returned":
            assert count == run["counts"].get(name, 0)
    assert groups == run["groups"], "service timeline differs from lifecycle"
    fallbacks = sum(e["structure"][5][0] == "external" for e in submitted.values())
    assert fallbacks == run["counts"].get("external_table_fallback_requests", 0), "CPU fallback omitted"
    dispatched, waits = [], []
    serial_indices = Counter()
    for index, group in enumerate(groups):
        ids = group["request_ids"]
        assert len({(submitted[r]["task"], submitted[r]["epoch"]) for r in ids}) == len(ids)
        semantic_keys = {digest(submitted[r]["structure"]) for r in ids}
        if run["route"] == "Gp":
            assert group.get("packet_mode") is True
            assert group["selection"]["selected_keys"] == len(semantic_keys)
        else:
            assert len(semantic_keys) == 1, "different semantics merged"
        expected_index = index
        if run["route"] not in {"Q", "G", "G0", "Gp"}:
            task = submitted[ids[0]]["task"]
            expected_index = serial_indices[task]
            serial_indices[task] += 1
        assert all(submitted[r]["group_index"] == expected_index for r in ids)
        assert all(submitted[r]["dispatch_ns"] <= group["start_ns"] for r in ids)
        if run["route"] in {"Q", "G", "G0", "Gp"}:
            for rid in ids:
                event = dispatches[rid]
                waits.append(event["dispatched_ns"]-event["submitted_ns"])
        dispatched.extend(ids)
        audited = [raw_requests[rid] for rid in ids if rid in raw_requests and rid in returned and returned[rid].get("result")]
        if recompute and audited:
            backend = "cpu" if group["hardware_fallback"] else group["backend"]
            expected = evaluate_range_requests(audited, backend=backend, diagnostics=True)
            for request in audited:
                assert result_record(expected[request.request_id]) == returned[request.request_id]["result"], "range endpoint or arithmetic record changed"
    assert len(dispatched) == len(set(dispatched))
    if run["route"] in {"Q", "G", "G0", "Gp"}:
        assert waits == run["wait_ns"], "wait durations disagree with request timeline"
    return dict(totals, raw_requests=len(raw_requests), consumed=len(consumed))


def verify_run(run, events, *, recompute=True):
    expected = "ONLINE_LIVE_SOLVE" if run["route"] in {"Q", "G", "G0", "Gp"} else "SERIAL_LIVE_SOLVE"
    assert run["execution"] == expected, "offline replay mislabeled as online"
    assert run["request_source"] == "CURRENT_WORKER_CALL" and not run["previous_answers_loaded"]
    timing = check_timing(run)
    inputs = check_states(run)
    arithmetic = check_lifecycle(run, events, inputs, recompute=recompute) if run["diagnostic"] else {}
    return dict(run_id=run["run_id"], timing=timing, arithmetic=arithmetic,
                accepted_lane_steps=run["accepted_lane_steps"], successful_tasks=run["successful_tasks"])


def load_run(path):
    path = Path(path)
    summary = read(path/"summary.json")
    assert all(sha(path/name) == checksum for name, checksum in summary["files"].items())
    with gzip.open(path/"run.json.gz", "rt") as inp:
        run = json.load(inp)
    events = []
    if (path/"lifecycle.jsonl.gz").exists():
        with gzip.open(path/"lifecycle.jsonl.gz", "rt") as inp:
            events = [json.loads(line) for line in inp]
    assert all(summary[key] == value for key, value in run.items() if key not in {"groups", "wait_ns", "records"})
    return run, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--no-recompute", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for path in args.paths:
        run, events = load_run(path)
        print(json.dumps(verify_run(run, events, recompute=not args.no_recompute)), flush=True)


if __name__ == "__main__":
    main()
