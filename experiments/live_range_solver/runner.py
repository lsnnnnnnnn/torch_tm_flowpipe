"""Run actual consecutive solver steps; no request or answer replay input.

Only the unchanged initial partition or explicit complete-state checkpoint is
loaded. Diagnostics record current requests while they are being executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from fractions import Fraction
import gzip
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import torch
import torch_tm_flowpipe as core
from torch_tm_flowpipe.live_range_service import (
    LiveRangeService, RangeCancelled, RequestIdentity, cuda_startup_check,
)
from torch_tm_flowpipe.packed_boundary_range import (
    _REQUEST_DISPATCH, _TABLE_REQUEST_DISPATCH, packed_boundary_execution,
)
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from torch_tm_flowpipe.range_requests import RangeRequest, evaluate_range_requests, structure_key
from experiments.endpoint_roundoff_repair.frozen import setup, step
from experiments.boundary_execution.state_equivalence import canonical, digest as state_digest
from experiments.range_batch_device.common import (
    request_record, result_record, digest, read, save, sha,
)


ROOT = Path(__file__).resolve().parents[2]
PARTITION = ROOT / "artifacts/runs/range_batch_device_20260909T030609Z/PARTITION_PLAN.json"
AUDIT_STEPS = (1, 2, 60, 100, 119, 120)


def initial(plant, task_id, *, original=False, checkpoint=None):
    if checkpoint is not None:
        loaded = core.load_terminal_checkpoint(checkpoint)
        if loaded.contract["plant"] != plant:
            raise ValueError("checkpoint plant mismatch")
        return loaded.current, loaded.normal_state
    config, current, state = setup(plant)
    if original:
        return current, state
    box = read(PARTITION)["plants"][plant]["tasks"][task_id]
    state = core.FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [tuple(map(Fraction, b)) for b in box["exact"]], config.order)
    return state.normalized_initial_tm(config.order), state


class Trace:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def __call__(self, record):
        record = dict(record)
        if record.get("request") is not None:
            record["request"] = request_record(record["request"])
            record["input_digest"] = digest(record["request"])
        if record.get("result") is not None:
            record["result"] = result_record(record["result"])
            record["output_digest"] = digest(record["result"])
        with self.lock:
            self.events.append(record)


class SerialTask:
    """Sequential strict request route with normal checks and no worker queue."""
    def __init__(self, run, task, accepted_state, backend, trace):
        self.run_id, self.task_id = run, str(task)
        self.accepted_state, self.backend, self.trace = accepted_state, backend, trace
        self.generation = accepted_state[1].step_index
        self.epoch = self.attempt = self.counter = 0
        self.in_attempt, self.pending = False, None
        self.status, self.previous = "RUNNABLE", None
        self.counts, self.groups = Counter(), []
        self._event("register")

    def identity(self):
        return RequestIdentity(self.run_id, self.task_id, self.epoch,
                               self.generation, self.attempt, self.counter)

    def _event(self, event, **extra):
        if self.trace:
            self.trace(dict(event=event, ns=time.perf_counter_ns(), **asdict(self.identity()), **extra))

    def begin_attempt(self, *, state_digest=None, diagnostic=False):
        if self.in_attempt or self.status in {"CANCELLED", "FAILED", "FINISHED"}:
            raise RuntimeError("invalid serial attempt")
        self.attempt += 1
        self.in_attempt, self.diagnostic, self.state_digest = True, diagnostic, state_digest
        self._event("begin", source_state=state_digest)
        return self.accepted_state

    def commit(self, accepted_state):
        if not self.in_attempt or self.status == "CANCELLED":
            raise RangeCancelled("serial attempt no longer active")
        self.accepted_state = accepted_state
        self.generation += 1
        self.in_attempt, self.status = False, "ACCEPTED"
        self._event("commit")

    def reject(self, *, failed=False, message=""):
        self.in_attempt, self.status = False, "FAILED" if failed else "REJECTED"
        self._event("failed" if failed else "reject", message=message)

    def finish(self):
        self.status = "FINISHED"
        self._event("finish")

    @contextmanager
    def execution(self):
        def dispatch(exponents, cl, ch, dl, dh, states, time_variable, kind, step_powers=None):
            self.counter += 1
            identity = self.identity()
            request = RangeRequest(identity.request_id, exponents, cl[0], ch[0], dl[0], dh[0],
                                   kind, states or (), time_variable, step_powers)
            self._event("submit", request_id=request.request_id, previous=self.previous,
                        source_state=self.state_digest, structure=structure_key(request),
                        request=request if self.diagnostic else None)
            self._event("dispatch", request_id=request.request_id, previous=self.previous,
                        source_state=self.state_digest, group=len(self.groups), reason="serial")
            start = time.perf_counter_ns()
            timing = {}
            result = evaluate_range_requests([request], backend=self.backend,
                diagnostics=self.diagnostic, timings=timing)[request.request_id]
            end = time.perf_counter_ns()
            if result.request_id != request.request_id:
                raise RuntimeError("serial backend identity mismatch")
            group = dict(start_ns=start, end_ns=end, size=1, reason="serial", backend=self.backend,
                         request_ids=[request.request_id], timing=timing, hardware_fallback=None,
                         completion_event=False, completion_confirmed=None)
            self.groups.append(group)
            self._event("group", group=group)
            self.counts["submitted"] += 1
            self.counts["returned"] += 1
            self.counts[f"status_{result.status}"] += 1
            self.counts["external_table_fallback_requests"] += step_powers is not None
            self.counts["gpu_completed_requests"] += self.backend == "cuda" and result.ok and step_powers is None
            self._event("return", request_id=request.request_id, previous=self.previous,
                        source_state=self.state_digest, status=result.status,
                        result=result if self.diagnostic else None)
            self._event("consume", request_id=request.request_id, previous=self.previous,
                        source_state=self.state_digest, status=result.status)
            self.previous = request.request_id
            if not result.ok:
                raise FloatingPointError(result.message)
            return core.Interval(result.lo[0], result.hi[0])
        with prepared_remainder_replay(True), packed_boundary_execution(True):
            token = _REQUEST_DISPATCH.set(dispatch)
            table = _TABLE_REQUEST_DISPATCH.set(dispatch)
            try:
                yield
            finally:
                _TABLE_REQUEST_DISPATCH.reset(table)
                _REQUEST_DISPATCH.reset(token)


def observe(segment):
    # The observer is identical for both backends, so different states are not
    # compared through two different measurement operators.
    with prepared_remainder_replay(True), packed_boundary_execution(True), core_range_cpu():
        return {name: [[float(iv.lo).hex(), float(iv.hi).hex()] for iv in model.range_box()]
                for name, model in (("endpoint", segment.endpoint_raw_tm), ("tube", segment.tm))}


@contextmanager
def core_range_cpu():
    from torch_tm_flowpipe.range_requests import range_request_execution
    with range_request_execution("cpu"):
        yield


def run_case(plant, ids, steps, route, *, run_id="development", max_wait_s=.020,
             max_group=32, diagnostic=False, audit_steps=AUDIT_STEPS, original=False,
             checkpoint=None, delays=None, task_steps=None, evaluator=None,
             hardware_fallback=False, select_newest=False, on_step=None):
    if route not in {"L", "S", "Q", "G", "S_gpu"}:
        raise ValueError("unknown route")
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("distinct nonempty task set required")
    trace = Trace() if diagnostic else None
    backend = "cuda" if route in {"G", "S_gpu"} else "cpu"
    start_ns, cpu_start_ns = time.perf_counter_ns(), time.process_time_ns()
    initial_states = {str(i): initial(plant, i, original=original, checkpoint=checkpoint) for i in ids}
    states, records = {}, {}
    service = (LiveRangeService(backend, max_wait_s=max_wait_s, max_group=max_group,
                run_id=run_id, trace=trace, evaluator=evaluator, hardware_fallback=hardware_fallback,
                select_newest=select_newest) if route in {"Q", "G"} else None)
    tasks = {}
    startup = {}
    if route == "S_gpu":
        cold = time.perf_counter_ns()
        startup = cuda_startup_check()
        startup["wall_s"] = (time.perf_counter_ns() - cold) / 1e9

    def work(task_id):
        task = tasks[task_id]
        history = []
        count = (task_steps or {}).get(task_id, steps)
        for offset in range(count):
            if task.status == "CANCELLED":
                break
            current, state = task.accepted_state
            before = state_digest((current, state)) if diagnostic else None
            task.begin_attempt(state_digest=before, diagnostic=diagnostic and (offset + 1 in audit_steps))
            if delays and task_id in delays:
                time.sleep(delays[task_id])
            context = task.execution() if route != "L" else legacy_execution()
            step_start = time.perf_counter_ns()
            try:
                with context:
                    segment = step(plant, current, state, state.step_index + 1)
                if diagnostic and state_digest((current, state)) != before:
                    raise AssertionError("solver modified an accepted input")
                if segment.status != "validated" or segment.reset_tm is None or segment.flowstar_normal_state is None:
                    task.reject(failed=True, message=segment.message)
                    history.append(dict(offset=offset+1, accepted=False, status=segment.status,
                                        message=segment.message, before=before,
                                        segment=canonical(segment) if diagnostic else None))
                    break
                # A cancellation can invalidate the whole attempt even after its
                # last range has returned. commit performs the authoritative check.
                task.commit((segment.reset_tm, segment.flowstar_normal_state))
            except RangeCancelled as error:
                history.append(dict(offset=offset+1, accepted=False, status="cancelled", message=str(error)))
                break
            except Exception as error:
                task.reject(failed=True, message=repr(error))
                history.append(dict(offset=offset+1, accepted=False, status="exception", message=repr(error)))
                break
            row = dict(offset=offset+1, accepted=True, status=segment.status,
                       generation=task.generation, state_step=segment.flowstar_normal_state.step_index,
                       h_hex=float(segment.h).hex(), step_rejections=segment.step_rejections,
                       validation_attempts=segment.validation_attempts,
                       step_start_ns=step_start, step_end_ns=time.perf_counter_ns(),
                       request_counter=task.counter)
            if diagnostic:
                row.update(before=before, after=state_digest(task.accepted_state),
                           input=canonical((current, state)) if offset == 0 else None,
                           segment=canonical(segment), segment_digest=state_digest(segment),
                           bounds=observe(segment))
                if state_digest(segment) != row["segment_digest"]:
                    raise AssertionError("observer changed segment or history")
            history.append(row)
            if on_step is not None:
                on_step(task, offset+1)
        else:
            if task.status != "CANCELLED":
                task.finish()
        states[task_id], records[task_id] = task.accepted_state, history

    with service if service is not None else nullcontext():
        for i in ids:
            key = str(i)
            if service:
                tasks[key] = service.register(key, initial_states[key], generation=initial_states[key][1].step_index)
            else:
                tasks[key] = SerialTask(run_id, key, initial_states[key], backend, trace)
        if service:
            with ThreadPoolExecutor(max_workers=len(ids), thread_name_prefix="solver") as pool:
                futures = [pool.submit(work, str(i)) for i in ids]
                for future in futures:
                    future.result()
        else:
            for i in ids:
                work(str(i))
    end_ns, cpu_end_ns = time.perf_counter_ns(), time.process_time_ns()
    counts = service.counts if service else sum((t.counts for t in tasks.values()), Counter())
    groups = service.groups if service else [g for task in tasks.values() for g in task.groups]
    wait = service.wait_ns if service else []
    accepted = sum(sum(row["accepted"] for row in rows) for rows in records.values())
    successful = sum(t.status == "FINISHED" for t in tasks.values())
    result = dict(schema="live-range-solver-run-v1", run_id=run_id, plant=plant, route=route,
        scope="RESUMED_LOCAL_WINDOW" if checkpoint else "ORIGINAL_B1" if original else "FIXED_PARTITION",
        execution="ONLINE_LIVE_SOLVE" if service else "SERIAL_LIVE_SOLVE",
        request_source="CURRENT_WORKER_CALL", previous_answers_loaded=False,
        ids=ids, steps=steps, original=original, diagnostic=diagnostic,
        audit_steps=list(audit_steps) if diagnostic else [], task_steps=task_steps,
        max_group=max_group, max_wait_s=max_wait_s,
        start_ns=start_ns, end_ns=end_ns, wall_s=(end_ns-start_ns)/1e9,
        cpu_start_ns=cpu_start_ns, cpu_end_ns=cpu_end_ns, cpu_s=(cpu_end_ns-cpu_start_ns)/1e9,
        successful_tasks=successful, accepted_lane_steps=accepted,
        successful_lane_steps=sum(sum(r["accepted"] for r in records[k]) for k,t in tasks.items() if t.status=="FINISHED"),
        task_statuses={k:t.status for k,t in tasks.items()}, counts=dict(counts),
        groups=groups, wait_ns=wait, records=records, startup=service.startup if service else startup,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_scope="process lifetime high-water mark", affinity=sorted(os.sched_getaffinity(0)),
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0) if torch.cuda.is_initialized() else 0,
        gpu_memory_scope="process lifetime Torch allocator high-water mark",
        torch_version=torch.__version__, python=sys.version, imported_core=str(Path(core.__file__).resolve()),
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        checkpoint=str(checkpoint) if checkpoint else None, partition_sha256=sha(PARTITION))
    result["throughput"] = result["successful_lane_steps"] / result["wall_s"]
    return result, trace.events if trace else [], states


@contextmanager
def legacy_execution():
    with prepared_remainder_replay(True), packed_boundary_execution(True):
        yield


def write_run(output, result, events):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with gzip.open(output/"run.json.gz", "wt", compresslevel=6) as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    if events:
        with gzip.open(output/"lifecycle.jsonl.gz", "wt", compresslevel=6) as out:
            for event in events:
                out.write(json.dumps(event, separators=(",", ":"), allow_nan=False)+"\n")
    summary = {k:v for k,v in result.items() if k not in {"groups", "wait_ns", "records"}}
    summary.update(serialization_s=time.perf_counter()-start, files={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    save(output/"summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", choices=["van_der_pol", "brusselator"], required=True)
    parser.add_argument("--route", choices=["L", "S", "Q", "G", "S_gpu"], required=True)
    parser.add_argument("--batch", type=int, choices=[1, 2, 8, 32], default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--ids", help="comma-separated fixed partition task indices")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-wait-s", type=float, default=.020)
    parser.add_argument("--max-group", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ids = ([int(x) for x in args.ids.split(",")] if args.ids else [0, 31] if args.batch == 2
           else read(PARTITION)["subsets"][str(args.batch)])
    result, events, _ = run_case(args.plant, ids, args.steps, args.route, run_id=args.output.name,
        diagnostic=args.diagnostic, original=args.original, checkpoint=args.checkpoint,
        max_wait_s=args.max_wait_s, max_group=args.max_group)
    summary = write_run(args.output, result, events)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["successful_tasks"] != len(ids):
        raise SystemExit("not all requested tasks finished")


if __name__ == "__main__":
    main()
