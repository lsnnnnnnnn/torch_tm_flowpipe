"""Real accepted-state failure, retry, asynchronous cancellation and safe resume."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import replace
from fractions import Fraction
import argparse
import gzip
import json
from pathlib import Path
from threading import Event

import torch
import torch_tm_flowpipe as core
from torch_tm_flowpipe.live_range_service import LiveRangeService, RangeCancelled
from torch_tm_flowpipe.live_range_checkpoint import save_live_range_checkpoint, load_live_range_checkpoint
from torch_tm_flowpipe.range_requests import evaluate_range_requests
from experiments.boundary_execution.state_equivalence import canonical, digest as state_digest
from experiments.endpoint_roundoff_repair.frozen import step
from experiments.range_batch_device.common import (
    digest, save, sha, request_record, request_from_record, result_from_record,
)
from experiments.range_batch_device.oracle import check
from .runner import Trace, initial, run_case


def exercise(plant, backend, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    baseline_route = "S_gpu" if backend == "cuda" else "S"
    baseline, _, _ = run_case(plant, [0, 4, 8, 31], 3, baseline_route, diagnostic=True,
                              run_id=f"fault-reference-{plant}-{backend}")
    entered, release = Event(), Event()
    trace, injections, attempts = Trace(), [], []
    failed_once = False

    def evaluate(rows, **kwargs):
        nonlocal failed_once
        for request in rows:
            if "/6:cancel/" in request.request_id and "/g1/a2/" in request.request_id:
                entered.set()
                if not release.wait(30):
                    raise RuntimeError("cancellation test coordinator failed to release backend")
        actual = list(rows)
        for index, request in enumerate(actual):
            if not failed_once and "/5:retry/" in request.request_id and "/g1/a2/" in request.request_id:
                failed_once = True
                actual[index] = replace(request, coefficients_lo=torch.full_like(request.coefficients_lo, float("nan")))
                injections.append(dict(kind="explicit_nonfinite_range_test", request=request_record(actual[index])))
        return evaluate_range_requests(actual, **kwargs)

    def advance(task, count):
        completed = 0
        while completed < count:
            current, state = task.accepted_state
            before = canonical((current, state))
            source = state_digest((current, state))
            task.begin_attempt(state_digest=source, diagnostic=True)
            record = dict(task=task.task_id, epoch=task.epoch, generation=task.generation,
                          attempt=task.attempt, input=before)
            try:
                with task.execution():
                    segment = step(plant, current, state, state.step_index+1)
                if task.status == "CANCELLED":
                    raise RangeCancelled("cancelled while the solver was unwinding")
                assert canonical((current, state)) == before
                record["segment"] = canonical(segment)
                if segment.status != "validated":
                    task.reject()
                    record.update(outcome="rejected", accepted_state=canonical(task.accepted_state))
                    attempts.append(record)
                    assert task.task_id == "retry" and task.attempt == 2
                    continue
                task.commit((segment.reset_tm, segment.flowstar_normal_state))
                completed += 1
                record.update(outcome="accepted", accepted_state=canonical(task.accepted_state))
            except RangeCancelled:
                record.update(outcome="cancelled", accepted_state=canonical(task.accepted_state))
                assert record["accepted_state"] == before
                attempts.append(record)
                return
            attempts.append(record)
        task.finish()

    service = LiveRangeService(backend, trace=trace, evaluator=evaluate,
                               run_id=f"fault-{plant}-{backend}")
    with service:
        mapping = {"cancel":0, "retry":4, "early":8, "healthy":31}
        tasks = {name:service.register(name, initial(plant, index)) for name,index in mapping.items()}
        with ThreadPoolExecutor(5) as pool:
            jobs = [pool.submit(advance, task, 1 if name == "early" else 3) for name,task in tasks.items()]
            assert entered.wait(30), "cancel lane never reached a real second attempt"
            cancelled = tasks["cancel"]
            snapshot = cancelled.checkpoint_state()
            checkpoint = output/"accepted_checkpoint"
            save_live_range_checkpoint(checkpoint, cancelled,
                scheduler=dict(accepted_steps=snapshot[1].step_index,
                               time_exact=str(snapshot[1].step_index*Fraction(.01 if plant=="van_der_pol" else .02))),
                contract=dict(plant=plant), provenance=dict(scope="LIVE_ASYNC_CANCEL_LAST_ACCEPTED"))
            restored = load_live_range_checkpoint(checkpoint, service)
            assert canonical(restored.accepted_state) == canonical(snapshot)
            assert restored.epoch == cancelled.epoch+1
            jobs.append(pool.submit(advance, restored, 2))
            release.set()
            for job in jobs:
                job.result(timeout=90)
    accepted = [r for r in attempts if r["outcome"] == "accepted"]
    for record in accepted:
        index = mapping[record["task"]]
        state_index = record["accepted_state"][1]["fields"]["step_index"]
        assert record["segment"] == baseline["records"][str(index)][state_index-1]["segment"]
    for record in attempts:
        if record["outcome"] != "accepted":
            assert record["accepted_state"] == record["input"], "failed attempt contaminated carry"
    assert failed_once and service.counts["stale_discarded"] >= 1
    assert len(accepted) == 10
    artifact = dict(plant=plant, backend=backend, test_inputs_only=True,
        baseline=baseline, attempts=attempts, events=trace.events, injections=injections,
        service_counts=dict(service.counts), checkpoints={p.name:sha(p) for p in checkpoint.iterdir()},
        expected_complete_steps={"cancel":3, "retry":3, "early":1, "healthy":3})
    verification = verify_faults(artifact)
    with gzip.open(output/"faults.json.gz", "wt") as out:
        json.dump(artifact, out, separators=(",", ":"), allow_nan=False)
    save(output/"summary.json", dict(**verification, files={"faults.json.gz":sha(output/"faults.json.gz")}))
    return verification


def verify_faults(artifact):
    from .verify import canonical_digest
    mapping = {"cancel":0, "retry":4, "early":8, "healthy":31}
    history, rejected, cancelled = {}, 0, 0
    for record in sorted(artifact["attempts"], key=lambda r:(r["task"], r["epoch"], r["attempt"])):
        task = record["task"]
        if task in history:
            assert record["input"] == history[task]
        history[task] = record["accepted_state"]
        if record["outcome"] == "accepted":
            fields = record["segment"]["fields"]
            assert record["accepted_state"] == [fields["reset_tm"], fields["flowstar_normal_state"]]
            index = fields["flowstar_normal_state"]["fields"]["step_index"]
            assert record["segment"] == artifact["baseline"]["records"][str(mapping[task])][index-1]["segment"]
        else:
            assert record["input"] == record["accepted_state"]
            rejected += record["outcome"] == "rejected"
            cancelled += record["outcome"] == "cancelled"
    assert rejected == cancelled == 1
    for task, count in artifact["expected_complete_steps"].items():
        assert history[task][1]["fields"]["step_index"] == count
    sources = {(r["task"],r["epoch"],r["attempt"]):canonical_digest(r["input"]) for r in artifact["attempts"]}
    requests, totals, cancel_times = {}, Counter(), {}
    for event in sorted(artifact["events"], key=lambda e:e["ns"]):
        if event["event"] == "submit":
            assert event["source_state"] == sources[event["task"],event["epoch"],event["attempt"]]
            requests[event["request_id"]] = request_from_record(event["request"])
        elif event["event"] == "return":
            result = result_from_record(event["result"])
            if result.ok:
                totals.update(check(requests[event["request_id"]], result))
                totals["audited_successful_returns"] += 1
        elif event["event"] == "cancel":
            cancel_times[event["task"],event["epoch"]] = event["ns"]
        elif event["event"] == "commit":
            assert (event["task"], event["epoch"]) not in cancel_times
    assert len(cancel_times) == 1
    return dict(plant=artifact["plant"], backend=artifact["backend"], successful_accepted_steps=10,
                rejected_attempts=rejected, cancelled_attempts=cancelled, arithmetic=dict(totals),
                stale_discarded=artifact["service_counts"]["stale_discarded"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summaries = []
    for plant in ("van_der_pol", "brusselator"):
        for backend in ("cpu", "cuda"):
            summary = exercise(plant, backend, args.output/f"{plant}-{backend}")
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    save(args.output/"failure_cancel_resume.json", summaries)


if __name__ == "__main__":
    main()
