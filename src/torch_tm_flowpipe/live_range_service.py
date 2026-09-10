"""Opt-in online range batching; one outstanding request per independent task.

Workers suspend at the existing sparse range adapter. A single service thread
owns CUDA submissions. Only immutable structure is reused; all numerical inputs
are copied on submission. No recorded request schedule or answer is accepted.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
import math
import threading
import time
import uuid

import torch

from .interval import Interval
from .packed_boundary_range import _REQUEST_DISPATCH, packed_boundary_execution
from .prepared_remainder_replay import prepared_remainder_replay
from .range_requests import RangeRequest, RangeResult, evaluate_range_requests, structure_key


class RangeCancelled(RuntimeError):
    """The attempt was invalidated; its accepted boundary is still available."""


class StaleRangeResponse(RuntimeError):
    """A response cannot be consumed by this task/epoch/attempt."""


@dataclass(frozen=True)
class RequestIdentity:
    run: str
    task: str
    epoch: int
    generation: int
    attempt: int
    counter: int

    @property
    def request_id(self):
        # Length prefixes avoid ambiguous task/run names containing separators.
        return (f"{len(self.run)}:{self.run}/{len(self.task)}:{self.task}/"
                f"e{self.epoch}/g{self.generation}/a{self.attempt}/r{self.counter}")


@dataclass
class _Pending:
    identity: RequestIdentity
    request: RangeRequest
    future: Future
    submitted_ns: int
    previous: str | None
    state_digest: str | None
    key: tuple
    diagnostic: bool
    ownership_copy_ns: int = 0
    submit_lock_wait_ns: int = 0
    deliver_lock_wait_ns: int = 0
    future_set_ns: int = 0
    consume_lock_wait_ns: int = 0
    future_wait_begin_ns: int = 0
    future_wait_end_ns: int = 0
    ownership_copy_begin_ns: int = 0
    ownership_copy_end_ns: int = 0
    submit_lock_begin_ns: int = 0
    submit_lock_end_ns: int = 0
    dispatched_ns: int = 0
    completed_ns: int = 0
    consumed_ns: int = 0
    group: dict | None = field(default=None, repr=False)
    group_position: int = -1


@dataclass
class RangeTask:
    service: "LiveRangeService" = field(repr=False)
    task_id: str
    epoch: int
    generation: int
    accepted_state: object = field(repr=False)
    status: str = "RUNNABLE"
    attempt: int = 0
    counter: int = 0
    pending: _Pending | None = field(default=None, repr=False)
    previous: str | None = None
    state_digest: str | None = None
    in_attempt: bool = False
    diagnostic: bool = False

    def begin_attempt(self, *, state_digest=None, diagnostic=False):
        with self.service._condition:
            self.service._live(self)
            if self.in_attempt or self.pending is not None:
                raise RuntimeError("an attempt or request is still active")
            self.attempt += 1
            self.in_attempt = True
            self.status = "RUNNABLE"
            self.state_digest = state_digest
            self.diagnostic = diagnostic
            self.service._event("begin", self, source_state=state_digest)
            return self.accepted_state

    def commit(self, accepted_state):
        """Publish the complete next state under the same lock as cancellation."""
        with self.service._condition:
            self.service._live(self)
            if not self.in_attempt or self.pending is not None:
                raise RuntimeError("commit requires a completed active attempt")
            self.accepted_state = accepted_state
            self.generation += 1
            self.in_attempt = False
            self.status = "ACCEPTED"
            self.service._event("commit", self)
            self.service._condition.notify_all()

    def reject(self, *, failed=False, message=""):
        with self.service._condition:
            if self.status in {"CANCELLED", "FAILED"}:
                return
            self.service._live(self)
            if self.pending is not None:
                raise RuntimeError("reject requires a drained request")
            self.in_attempt = False
            self.status = "FAILED" if failed else "REJECTED"
            self.service._event("failed" if failed else "reject", self, message=message)
            self.service._condition.notify_all()

    def finish(self):
        with self.service._condition:
            self.service._live(self)
            if self.pending is not None or self.in_attempt:
                raise RuntimeError("finish requires a completed boundary")
            self.status = "FINISHED"
            self.service._event("finish", self)
            self.service._condition.notify_all()

    def cancel(self):
        self.service.cancel(self)

    def checkpoint_state(self):
        """Cancel any uncommitted work and copy only the last accepted state.

        This returns ordinary mathematical objects for the existing safe JSON
        checkpoint writer, never a worker, Future, device buffer or pointer.
        """
        with self.service._condition:
            if self.in_attempt or self.pending is not None:
                self.service._cancel_locked(self)
            self.service._event("checkpoint", self)
            return deepcopy(self.accepted_state)

    def evaluate(self, request):
        pending = self.service.submit(self, request)
        pending.future_wait_begin_ns = time.perf_counter_ns()
        try:
            result = pending.future.result()
        except BaseException:
            consume_lock_begin = time.perf_counter_ns()
            with self.service._condition:
                now = time.perf_counter_ns()
                pending.consume_lock_wait_ns = now - consume_lock_begin
                pending.future_wait_end_ns = now
                future_wait_ns = now - pending.future_wait_begin_ns
                if pending.group is not None:
                    scheduler = pending.group["scheduler_costs"]
                    scheduler["consume_lock_wait_sum_ns"] += pending.consume_lock_wait_ns
                    scheduler["future_wait_sum_ns"] += future_wait_ns
                    scheduler["future_wait_intervals_ns"][pending.group_position] = [
                        pending.future_wait_begin_ns, pending.future_wait_end_ns]
                if self.pending is pending:
                    self.pending = None
                    self.status = "RUNNABLE"
                    self.service._condition.notify_all()
            raise
        consume_lock_begin = time.perf_counter_ns()
        with self.service._condition:
            now = time.perf_counter_ns()
            pending.consume_lock_wait_ns = now - consume_lock_begin
            pending.future_wait_end_ns = now
            future_wait_ns = now - pending.future_wait_begin_ns
            if pending.group is not None:
                scheduler = pending.group["scheduler_costs"]
                scheduler["consume_lock_wait_sum_ns"] += pending.consume_lock_wait_ns
                scheduler["future_wait_sum_ns"] += future_wait_ns
                scheduler["future_wait_intervals_ns"][pending.group_position] = [
                    pending.future_wait_begin_ns, pending.future_wait_end_ns]
            self.service._live(self)
            if (self.pending is not pending or pending.identity != self._identity()
                    or result.request_id != pending.identity.request_id):
                raise StaleRangeResponse("returned identity differs from the active request")
            pending.consumed_ns = time.perf_counter_ns()
            self.pending = None
            self.previous = pending.identity.request_id
            self.status = "RUNNABLE"
            self.service._request_event("consume", pending, status=result.status,
                                        future_wait_ns=future_wait_ns)
            self.service._condition.notify_all()
        # Never expose the service's copy as mutable caller storage.
        return replace(result, lo=result.lo.clone() if result.lo is not None else None,
                       hi=result.hi.clone() if result.hi is not None else None,
                       terms_lo=result.terms_lo.clone() if result.terms_lo is not None else None,
                       terms_hi=result.terms_hi.clone() if result.terms_hi is not None else None)

    def _identity(self):
        return RequestIdentity(self.service.run_id, self.task_id, self.epoch,
                               self.generation, self.attempt, self.counter)

    @contextmanager
    def execution(self):
        """Explicitly install every context in the calling worker thread."""
        def dispatch(exponents, cl, ch, dl, dh, states, time_variable, kind, step_powers=None):
            result = self.evaluate(RangeRequest("assigned-at-submit", exponents, cl[0], ch[0],
                dl[0], dh[0], kind, states or (), time_variable, step_powers))
            if not result.ok:
                raise FloatingPointError(f"range {result.status}: {result.message}")
            return Interval(result.lo[0], result.hi[0])

        from .packed_boundary_range import _TABLE_REQUEST_DISPATCH
        with prepared_remainder_replay(True), packed_boundary_execution(True):
            token = _REQUEST_DISPATCH.set(dispatch)
            table_token = _TABLE_REQUEST_DISPATCH.set(dispatch)
            try:
                yield
            finally:
                _TABLE_REQUEST_DISPATCH.reset(table_token)
                _REQUEST_DISPATCH.reset(token)


class LiveRangeService:
    """Online service with bounded ready waiting and explicit failure policy.

    Construct before registering/starting workers; enter before computations.
    ``trace`` is an optional diagnostic callback and is never needed for normal
    identity checks. ``hardware_fallback`` either stops affected tasks or reruns
    the same owned requests on CPU, with its cost and count reported explicitly.
    """
    def __init__(self, backend="cpu", *, max_group=32, max_wait_s=.020,
                 run_id=None, trace=None, hardware_fallback=False,
                 evaluator=None, select_newest=False, packet_mode=False,
                 packet_limits=None, group_callback=None, retain_groups=True,
                 retain_waits=True):
        if backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be cpu or cuda")
        if type(max_group) is not int or max_group < 1:
            raise ValueError("max_group must be positive")
        if not math.isfinite(max_wait_s) or max_wait_s <= 0:
            raise ValueError("max_wait_s must be finite and positive")
        if type(packet_mode) is not bool:
            raise ValueError("packet_mode must be bool")
        if type(retain_groups) is not bool or type(retain_waits) is not bool:
            raise ValueError("retention controls must be bool")
        if group_callback is not None and not callable(group_callback):
            raise ValueError("group_callback must be callable")
        if packet_mode and backend != "cuda" and evaluator is None:
            raise ValueError("the production packet evaluator requires the CUDA backend")
        self.backend, self.max_group, self.max_wait_s = backend, max_group, max_wait_s
        self.run_id = run_id or uuid.uuid4().hex
        self.trace, self.hardware_fallback = trace, hardware_fallback
        self.evaluator = evaluator
        self.select_newest = select_newest
        self.packet_mode, self.packet_limits = packet_mode, packet_limits
        self.group_callback = group_callback
        self.retain_groups, self.retain_waits = retain_groups, retain_waits
        self._condition = threading.Condition(threading.RLock())
        self._tasks, self._epochs, self._queues = {}, {}, {}
        self._closing, self._started = False, False
        self._ready = Future()
        self._thread = threading.Thread(target=self._serve, name=f"range-{self.run_id}", daemon=False)
        self.counts = Counter()
        self.groups = []
        self.wait_ns = []
        self.startup = {}
        self.owner_thread = None
        self.packet_executor = None

    def register(self, task_id, accepted_state, *, generation=0, minimum_epoch=0):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("nonempty task id required")
        if any(type(v) is not int or v < 0 for v in (generation, minimum_epoch)):
            raise ValueError("nonnegative integer generation and epoch required")
        with self._condition:
            if self._closing:
                raise RuntimeError("service is closed")
            if task_id in self._tasks:
                self._cancel_locked(self._tasks[task_id])
            epoch = max(self._epochs.get(task_id, -1) + 1, minimum_epoch)
            self._epochs[task_id] = epoch
            task = RangeTask(self, task_id, epoch, generation, deepcopy(accepted_state))
            self._tasks[task_id] = task
            self._event("register", task)
            self._condition.notify_all()
            return task

    def _live(self, task):
        if self._tasks.get(task.task_id) is not task:
            raise StaleRangeResponse("task was replaced by a new epoch")
        if task.status in {"CANCELLED", "FAILED", "FINISHED"} or self._closing:
            raise RangeCancelled(f"task cannot continue: {task.status}")

    def _event(self, event, task, **extra):
        if self.trace is not None:
            self.trace(dict(event=event, ns=time.perf_counter_ns(), run=self.run_id,
                task=task.task_id, epoch=task.epoch, generation=task.generation,
                attempt=task.attempt, counter=task.counter, **extra))

    def _request_event(self, event, pending, **extra):
        if self.trace is not None:
            self.trace(dict(event=event, ns=time.perf_counter_ns(), **asdict(pending.identity),
                request_id=pending.identity.request_id, previous=pending.previous,
                source_state=pending.state_digest, submitted_ns=pending.submitted_ns,
                dispatched_ns=pending.dispatched_ns, completed_ns=pending.completed_ns,
                consumed_ns=pending.consumed_ns,
                ownership_copy_ns=pending.ownership_copy_ns,
                submit_lock_wait_ns=pending.submit_lock_wait_ns,
                deliver_lock_wait_ns=pending.deliver_lock_wait_ns,
                future_set_ns=pending.future_set_ns,
                consume_lock_wait_ns=pending.consume_lock_wait_ns,
                future_wait_begin_ns=pending.future_wait_begin_ns,
                future_wait_end_ns=pending.future_wait_end_ns,
                ownership_copy_begin_ns=pending.ownership_copy_begin_ns,
                ownership_copy_end_ns=pending.ownership_copy_end_ns,
                submit_lock_begin_ns=pending.submit_lock_begin_ns,
                submit_lock_end_ns=pending.submit_lock_end_ns, **extra))

    def submit(self, task, request):
        # Ownership is established before enqueue; even views of caller tensors
        # and an external power table become independent storage.
        copy_begin = time.perf_counter_ns()
        owned = replace(request, **{name: getattr(request, name).detach().clone() for name in
            ("coefficients_lo", "coefficients_hi", "domain_lo", "domain_hi")},
            step_powers=deepcopy(request.step_powers))
        copy_end = time.perf_counter_ns()
        ownership_copy_ns = copy_end - copy_begin
        key = structure_key(owned)
        lock_begin = time.perf_counter_ns()
        with self._condition:
            lock_end = time.perf_counter_ns()
            submit_lock_wait_ns = lock_end - lock_begin
            self._live(task)
            if not self._started or not task.in_attempt or task.pending is not None:
                raise RuntimeError("submission requires one active attempt and no outstanding request")
            task.counter += 1
            identity = task._identity()
            owned = replace(owned, request_id=identity.request_id)
            pending = _Pending(identity, owned, Future(), time.perf_counter_ns(),
                               task.previous, task.state_digest, key, task.diagnostic,
                               ownership_copy_ns=ownership_copy_ns,
                               submit_lock_wait_ns=submit_lock_wait_ns,
                               ownership_copy_begin_ns=copy_begin,
                               ownership_copy_end_ns=copy_end,
                               submit_lock_begin_ns=lock_begin,
                               submit_lock_end_ns=lock_end)
            task.pending, task.status = pending, "WAITING_RANGE"
            self._queues.setdefault(key, deque()).append(pending)
            self.counts["submitted"] += 1
            self._request_event("submit", pending, request=owned if task.diagnostic else None,
                                structure=key)
            self._condition.notify_all()
            return pending

    def _cancel_locked(self, task):
        if task.status in {"CANCELLED", "FINISHED", "FAILED"}:
            return
        task.status, task.in_attempt = "CANCELLED", False
        if task.pending is not None:
            pending, task.pending = task.pending, None
            if not pending.future.done():
                future_begin = time.perf_counter_ns()
                pending.future.set_exception(RangeCancelled("cancelled uncommitted attempt"))
                pending.future_set_ns += time.perf_counter_ns() - future_begin
                if pending.group is not None:
                    pending.group["scheduler_costs"]["future_set_sum_ns"] += pending.future_set_ns
            self.counts["cancelled_requests"] += 1
        self._event("cancel", task)
        self._condition.notify_all()

    def cancel(self, task):
        with self._condition:
            self._cancel_locked(task)

    def deliver(self, pending, result):
        """Check envelope and returned ID at the only result delivery boundary."""
        lock_begin = time.perf_counter_ns()
        with self._condition:
            pending.deliver_lock_wait_ns = time.perf_counter_ns() - lock_begin
            if pending.group is not None:
                pending.group["scheduler_costs"]["deliver_lock_wait_sum_ns"] += \
                    pending.deliver_lock_wait_ns
            task = self._tasks.get(pending.identity.task)
            if (task is None or task.pending is not pending or task._identity() != pending.identity
                    or task.status != "WAITING_RANGE" or pending.future.done()):
                self.counts["stale_discarded"] += 1
                self._request_event("discard", pending, reason="inactive identity")
                return False
            if result.request_id != pending.identity.request_id:
                future_begin = time.perf_counter_ns()
                pending.future.set_exception(StaleRangeResponse("backend returned a different request id"))
                pending.future_set_ns += time.perf_counter_ns() - future_begin
                if pending.group is not None:
                    pending.group["scheduler_costs"]["future_set_sum_ns"] += pending.future_set_ns
                self.counts["identity_errors"] += 1
                self._request_event("identity_error", pending, returned_id=result.request_id)
                return False
            pending.completed_ns = time.perf_counter_ns()
            future_begin = time.perf_counter_ns()
            pending.future.set_result(result)
            pending.future_set_ns = time.perf_counter_ns() - future_begin
            if pending.group is not None:
                pending.group["scheduler_costs"]["future_set_sum_ns"] += pending.future_set_ns
            self._request_event("return", pending, result=result if pending.diagnostic else None,
                                status=result.status)
            self.counts["returned"] += 1
            self.counts[f"status_{result.status}"] += 1
            return True

    def _select(self):
        # Called under the condition lock. Cancelled queued requests are removed
        # without preventing progress of a different key or a surviving task.
        for key in list(self._queues):
            queue = self._queues[key]
            self._queues[key] = deque(p for p in queue if not p.future.done())
            if not self._queues[key]:
                del self._queues[key]
        if not self._queues:
            return None, None, None, None
        now = time.perf_counter_ns()
        ready_requests = sum(map(len, self._queues.values()))
        ready_keys = len(self._queues)
        oldest = min(self._queues, key=lambda k: self._queues[k][0].submitted_ns)
        expired = now - self._queues[oldest][0].submitted_ns >= self.max_wait_s * 1e9
        full = [k for k, q in self._queues.items() if len(q) >= self.max_group]
        parked = all(t.status in {"WAITING_RANGE", "FINISHED", "FAILED", "CANCELLED"}
                     for t in self._tasks.values())
        # Deadline takes priority to prevent starvation by a stream of full keys.
        if expired:
            key, reason = oldest, "timeout"
        elif full:
            key, reason = full[0], "max_group"
        elif parked:
            key, reason = oldest, "all_waiting"
            if self.select_newest:
                key = max(self._queues, key=lambda k: self._queues[k][0].submitted_ns)
        else:
            remaining = self.max_wait_s - (now - self._queues[oldest][0].submitted_ns) / 1e9
            return None, None, max(remaining, 1e-6), None
        if self.packet_mode:
            all_ready = sorted((pending for queue in self._queues.values() for pending in queue),
                               key=lambda pending: pending.submitted_ns)
            selected = all_ready[:self.max_group]
            take = Counter(pending.key for pending in selected)
            for selected_key, count in take.items():
                queue = self._queues[selected_key]
                for _ in range(count):
                    pending = queue.popleft()
                    assert any(pending is choice for choice in selected)
                if not queue:
                    del self._queues[selected_key]
        else:
            queue = self._queues[key]
            selected = [queue.popleft() for _ in range(min(self.max_group, len(queue)))]
            if not queue:
                del self._queues[key]
        selected_key_sizes = Counter(pending.key for pending in selected)
        selection = dict(packet_mode=self.packet_mode, ready_requests=ready_requests,
            ready_keys=ready_keys, selected_requests=len(selected),
            selected_keys=len(selected_key_sizes), remaining_ready=ready_requests-len(selected),
            selected_key_sizes=sorted(selected_key_sizes.values(), reverse=True),
            unavailable_tasks=len(self._tasks)-ready_requests)
        for pending in selected:
            pending.dispatched_ns = now
            if self.retain_waits:
                self.wait_ns.append(now - pending.submitted_ns)
            self._request_event("dispatch", pending, group=self.counts["groups"], reason=reason)
        return selected, reason, None, selection

    def _serve(self):
        self.owner_thread = threading.get_ident()
        try:
            start = time.perf_counter_ns()
            stream = None
            evaluator = self.evaluator
            if self.backend == "cuda":
                torch.cuda.set_device(0)
                stream = torch.cuda.Stream(device=0)
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                if self.backend == "cuda":
                    if self.packet_mode and evaluator is None:
                        from .range_packets import (
                            DEFAULT_PACKET_LIMITS, RangePacketExecutor,
                            packet_cuda_startup_check,
                        )
                        limits = self.packet_limits or DEFAULT_PACKET_LIMITS
                        self.packet_executor = RangePacketExecutor(device="cuda:0", limits=limits)
                        self.startup = packet_cuda_startup_check(self.packet_executor)
                        evaluator = self.packet_executor.evaluate
                    else:
                        self.startup = cuda_startup_check()
                    self.startup["stream"] = stream.cuda_stream
                if evaluator is None:
                    evaluator = evaluate_range_requests
                self.startup["wall_s"] = (time.perf_counter_ns() - start) / 1e9
                self._ready.set_result(True)
                while True:
                    with self._condition:
                        selection_begin = time.perf_counter_ns()
                        selection_cpu_begin = time.thread_time_ns()
                        selected, reason, wait, selection = self._select()
                        selection_wall_ns = time.perf_counter_ns() - selection_begin
                        selection_cpu_ns = time.thread_time_ns() - selection_cpu_begin
                        if selected is None:
                            if self._closing:
                                break
                            self._condition.wait(wait)
                            continue
                    requests = [p.request for p in selected]
                    begin = time.perf_counter_ns()
                    thread_begin = time.thread_time_ns()
                    timing, fallback_error = {}, None
                    try:
                        results = evaluator(requests, backend=self.backend,
                            diagnostics=any(p.diagnostic for p in selected), timings=timing)
                    except Exception as error:
                        if self.backend == "cuda" and self.hardware_fallback:
                            fallback_error = repr(error)
                            timing["failed_cuda_s"] = (time.perf_counter_ns() - begin) / 1e9
                            self.counts["hardware_fallback_requests"] += len(selected)
                            cpu_timing = {}
                            results = evaluate_range_requests(requests, backend="cpu",
                                diagnostics=any(p.diagnostic for p in selected), timings=cpu_timing)
                            timing["cpu_recompute"] = cpu_timing
                        else:
                            results = {r.request_id: RangeResult(r.request_id, "backend_error", message=repr(error))
                                       for r in requests}
                    completion = None
                    if stream is not None and fallback_error is None:
                        completion = torch.cuda.Event()
                        completion.record(stream)
                        completion.synchronize()
                    ended = time.perf_counter_ns()
                    scheduler_costs = dict(
                        ownership_copy_sum_ns=sum(p.ownership_copy_ns for p in selected),
                        submit_lock_wait_sum_ns=sum(p.submit_lock_wait_ns for p in selected),
                        deliver_lock_wait_sum_ns=sum(p.deliver_lock_wait_ns for p in selected),
                        future_set_sum_ns=sum(p.future_set_ns for p in selected),
                        consume_lock_wait_sum_ns=sum(p.consume_lock_wait_ns for p in selected),
                        future_wait_sum_ns=sum(
                            p.future_wait_end_ns-p.future_wait_begin_ns for p in selected
                            if p.future_wait_begin_ns and p.future_wait_end_ns),
                        ownership_copy_intervals_ns=[[p.ownership_copy_begin_ns,
                            p.ownership_copy_end_ns] for p in selected],
                        submit_lock_intervals_ns=[[p.submit_lock_begin_ns,
                            p.submit_lock_end_ns] for p in selected],
                        future_wait_intervals_ns=[
                            [p.future_wait_begin_ns, p.future_wait_end_ns]
                            if p.future_wait_begin_ns and p.future_wait_end_ns else [0, 0]
                            for p in selected])
                    group = dict(start_ns=begin, end_ns=ended, size=len(selected), reason=reason,
                        thread_cpu_ns=time.thread_time_ns()-thread_begin,
                        request_ids=[p.identity.request_id for p in selected], timing=timing,
                        request_generations=[p.identity.generation for p in selected],
                        request_attempts=[p.identity.attempt for p in selected],
                        request_wait_ns=[p.dispatched_ns-p.submitted_ns for p in selected],
                        backend=self.backend, hardware_fallback=fallback_error,
                        packet_mode=self.packet_mode, selection=selection,
                        selection_wall_ns=selection_wall_ns,
                        selection_thread_cpu_ns=selection_cpu_ns,
                        scheduler_costs=scheduler_costs,
                        completion_event=completion is not None,
                        completion_confirmed=bool(completion.query()) if completion is not None else None)
                    for position, pending in enumerate(selected):
                        pending.group = group
                        pending.group_position = position
                    if self.retain_groups:
                        self.groups.append(group)
                    self.counts["groups"] += 1
                    self.counts[f"flush_{reason}"] += 1
                    self.counts["external_table_fallback_requests"] += sum(r.step_powers is not None for r in requests)
                    if self.backend == "cuda" and fallback_error is None:
                        self.counts["gpu_completed_requests"] += sum(
                            result.ok and not result.status.startswith("fallback") for result in results.values())
                    if self.trace is not None:
                        self.trace(dict(event="group", group=group, ns=ended))
                    # Publish the immutable group envelope before waking any
                    # consumer. The callback retains the same dictionary, so
                    # per-consumer wait/lock fields filled below remain visible.
                    if self.group_callback is not None:
                        self.group_callback(group)
                    for pending in selected:
                        result = results.get(pending.identity.request_id)
                        if result is None:
                            result = RangeResult(pending.identity.request_id, "backend_error", message="backend omitted request")
                        self.deliver(pending, result)
        except BaseException as error:
            if not self._ready.done():
                self._ready.set_exception(error)
            with self._condition:
                self._closing = True
                for task in self._tasks.values():
                    self._cancel_locked(task)
                self.counts["service_failures"] += 1
                if self.trace is not None:
                    self.trace(dict(event="service_failure", message=repr(error), ns=time.perf_counter_ns()))

    def __enter__(self):
        if self._started:
            raise RuntimeError("service can only be started once")
        self._started = True
        self._thread.start()
        try:
            self._ready.result()
        except BaseException:
            self._thread.join()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        with self._condition:
            self._closing = True
            for task in self._tasks.values():
                self._cancel_locked(task)
            self._condition.notify_all()
        self._thread.join()


_SELF_TESTED = set()
_SELF_TEST_LOCK = threading.Lock()


def cuda_startup_check():
    """Once per process/device/module: check directed primitives and known power.

    A successful compile or available GPU alone does not open the service gate.
    This is deliberately small; full independent term audits live in diagnostics.
    """
    from .range_cuda import get_module, primitive_probe
    module = get_module()
    key = (module.device, module.build["ptx_sha256"])
    with _SELF_TEST_LOCK:
        if key in _SELF_TESTED:
            return dict(reused=True, build=module.build)
        a = [5e-324, -5e-324, 1e-160, -1e-160, 1., 1e30]
        b = [.5, .5, 1e-160, 1e-160, 5e-324, -1e30]
        output = primitive_probe(a, b)
        for row, x, y in zip(output, a, b):
            for index, exact in ((0, Fraction(x) + Fraction(y)), (2, Fraction(x) * Fraction(y))):
                if not Fraction(float(row[index])) <= exact <= Fraction(float(row[index + 1])):
                    raise FloatingPointError("CUDA primitive startup check failed")
        x = float.fromhex("0x1.7d3ecfa658d9bp+9")
        c = torch.ones((1, 1), dtype=torch.float64)
        domain = torch.tensor([x], dtype=torch.float64)
        request = RangeRequest("startup-pow3", ((3,),), c, c.clone(), domain, domain.clone())
        result = evaluate_range_requests([request], backend="cuda")[request.request_id]
        if not result.ok or not Fraction(float(result.lo[0])) <= Fraction(x)**3 <= Fraction(float(result.hi[0])):
            raise FloatingPointError("CUDA power startup check failed")
        _SELF_TESTED.add(key)
        return dict(reused=False, build=module.build,
                    primitive_bounds=[[float(x).hex() for x in row] for row in output],
                    pow3_bounds=[float(result.lo[0]).hex(), float(result.hi[0]).hex()])
