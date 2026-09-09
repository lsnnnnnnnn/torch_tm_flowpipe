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
    dispatched_ns: int = 0
    completed_ns: int = 0
    consumed_ns: int = 0


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
        try:
            result = pending.future.result()
        except BaseException:
            with self.service._condition:
                if self.pending is pending:
                    self.pending = None
                    self.status = "RUNNABLE"
                    self.service._condition.notify_all()
            raise
        with self.service._condition:
            self.service._live(self)
            if (self.pending is not pending or pending.identity != self._identity()
                    or result.request_id != pending.identity.request_id):
                raise StaleRangeResponse("returned identity differs from the active request")
            pending.consumed_ns = time.perf_counter_ns()
            self.pending = None
            self.previous = pending.identity.request_id
            self.status = "RUNNABLE"
            self.service._request_event("consume", pending, status=result.status)
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
                 evaluator=None, select_newest=False):
        if backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be cpu or cuda")
        if type(max_group) is not int or max_group < 1:
            raise ValueError("max_group must be positive")
        if not math.isfinite(max_wait_s) or max_wait_s <= 0:
            raise ValueError("max_wait_s must be finite and positive")
        self.backend, self.max_group, self.max_wait_s = backend, max_group, max_wait_s
        self.run_id = run_id or uuid.uuid4().hex
        self.trace, self.hardware_fallback = trace, hardware_fallback
        self.evaluator = evaluator or evaluate_range_requests
        self.select_newest = select_newest
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

    def register(self, task_id, accepted_state, *, generation=0):
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("nonempty task id required")
        with self._condition:
            if self._closing:
                raise RuntimeError("service is closed")
            if task_id in self._tasks:
                self._cancel_locked(self._tasks[task_id])
            epoch = self._epochs.get(task_id, -1) + 1
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
                source_state=pending.state_digest, **extra))

    def submit(self, task, request):
        # Ownership is established before enqueue; even views of caller tensors
        # and an external power table become independent storage.
        owned = replace(request, **{name: getattr(request, name).detach().clone() for name in
            ("coefficients_lo", "coefficients_hi", "domain_lo", "domain_hi")},
            step_powers=deepcopy(request.step_powers))
        key = structure_key(owned)
        with self._condition:
            self._live(task)
            if not self._started or not task.in_attempt or task.pending is not None:
                raise RuntimeError("submission requires one active attempt and no outstanding request")
            task.counter += 1
            identity = task._identity()
            owned = replace(owned, request_id=identity.request_id)
            pending = _Pending(identity, owned, Future(), time.perf_counter_ns(),
                               task.previous, task.state_digest, key, task.diagnostic)
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
                pending.future.set_exception(RangeCancelled("cancelled uncommitted attempt"))
            self.counts["cancelled_requests"] += 1
        self._event("cancel", task)
        self._condition.notify_all()

    def cancel(self, task):
        with self._condition:
            self._cancel_locked(task)

    def deliver(self, pending, result):
        """Check envelope and returned ID at the only result delivery boundary."""
        with self._condition:
            task = self._tasks.get(pending.identity.task)
            if (task is None or task.pending is not pending or task._identity() != pending.identity
                    or task.status != "WAITING_RANGE" or pending.future.done()):
                self.counts["stale_discarded"] += 1
                self._request_event("discard", pending, reason="inactive identity")
                return False
            if result.request_id != pending.identity.request_id:
                pending.future.set_exception(StaleRangeResponse("backend returned a different request id"))
                self.counts["identity_errors"] += 1
                self._request_event("identity_error", pending, returned_id=result.request_id)
                return False
            pending.completed_ns = time.perf_counter_ns()
            self._request_event("return", pending, result=result if pending.diagnostic else None,
                                status=result.status)
            pending.future.set_result(result)
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
            return None, None, None
        now = time.perf_counter_ns()
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
            return None, None, max(remaining, 1e-6)
        queue = self._queues[key]
        selected = [queue.popleft() for _ in range(min(self.max_group, len(queue)))]
        if not queue:
            del self._queues[key]
        for pending in selected:
            pending.dispatched_ns = now
            self.wait_ns.append(now - pending.submitted_ns)
            self._request_event("dispatch", pending, group=len(self.groups), reason=reason)
        return selected, reason, None

    def _serve(self):
        self.owner_thread = threading.get_ident()
        try:
            start = time.perf_counter_ns()
            stream = None
            if self.backend == "cuda":
                torch.cuda.set_device(0)
                stream = torch.cuda.Stream(device=0)
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                if self.backend == "cuda":
                    self.startup = cuda_startup_check()
                    self.startup["stream"] = stream.cuda_stream
                self.startup["wall_s"] = (time.perf_counter_ns() - start) / 1e9
                self._ready.set_result(True)
                while True:
                    with self._condition:
                        selected, reason, wait = self._select()
                        if selected is None:
                            if self._closing:
                                break
                            self._condition.wait(wait)
                            continue
                    requests = [p.request for p in selected]
                    begin = time.perf_counter_ns()
                    timing, fallback_error = {}, None
                    try:
                        results = self.evaluator(requests, backend=self.backend,
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
                    group = dict(start_ns=begin, end_ns=ended, size=len(selected), reason=reason,
                        request_ids=[p.identity.request_id for p in selected], timing=timing,
                        backend=self.backend, hardware_fallback=fallback_error,
                        completion_event=completion is not None,
                        completion_confirmed=bool(completion.query()) if completion is not None else None)
                    self.groups.append(group)
                    self.counts["groups"] += 1
                    self.counts[f"flush_{reason}"] += 1
                    self.counts["external_table_fallback_requests"] += sum(r.step_powers is not None for r in requests)
                    if self.backend == "cuda" and fallback_error is None:
                        self.counts["gpu_completed_requests"] += sum(
                            result.ok and not result.status.startswith("fallback") for result in results.values())
                    if self.trace is not None:
                        self.trace(dict(event="group", group=group, ns=ended))
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
