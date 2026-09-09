from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
import time

import pytest
import torch

from torch_tm_flowpipe import Interval, Polynomial
from torch_tm_flowpipe.polynomial import evaluate_interval_normal
from torch_tm_flowpipe.live_range_service import (
    LiveRangeService, RangeCancelled, StaleRangeResponse,
)
from torch_tm_flowpipe.range_requests import RangeRequest, evaluate_range_requests, structure_key
from experiments.range_batch_device.oracle import check


def request(value=2., *, exponents=((3,),)):
    c = torch.tensor([[value] * len(exponents)], dtype=torch.float64)
    d = torch.tensor([float.fromhex("0x1.7d3ecfa658d9bp+9")], dtype=torch.float64)
    return RangeRequest("caller", exponents, c, c.clone(), d, d.clone())


def lane(task, r):
    task.begin_attempt(diagnostic=True)
    result = task.evaluate(r)
    task.commit((result.lo, result.hi))
    task.finish()
    return result


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_ready_group_exact_identity_and_private_storage(backend):
    rows = [request(1.), request(2.), request(3.), request(4.)]
    with LiveRangeService(backend, max_wait_s=.1) as service:
        tasks = [service.register(str(i), {"x": torch.tensor([0.])}) for i in range(4)]
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(lane, task, row) for task, row in zip(tasks, rows)]
            results = [future.result(timeout=20) for future in futures]
        assert sum(g["size"] for g in service.groups) == 4
        for r, result in zip(rows, results):
            check(r, result)
            assert result.lo.data_ptr() != r.coefficients_lo.data_ptr()
        assert len({r.request_id for r in results}) == 4
        original = results[1].lo.clone()
        results[0].lo.add_(99)
        assert torch.equal(results[1].lo, original)
        assert tasks[0].accepted_state[0].data_ptr() != tasks[1].accepted_state[0].data_ptr()
        if backend == "cuda":
            assert all(g["completion_confirmed"] for g in service.groups)
            assert service.counts["gpu_completed_requests"] == 4


def test_support_order_and_heterogeneous_early_finish_do_not_wait_for_alignment():
    rows = [request(1., exponents=((1,), (2,))), request(2., exponents=((2,), (1,)))]
    assert structure_key(rows[0]) != structure_key(rows[1])
    with LiveRangeService(max_wait_s=.005) as service:
        tasks = [service.register(str(i), None) for i in range(3)]
        tasks[2].finish()
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(lane, task, row) for task, row in zip(tasks, rows)]
            assert all(f.result(timeout=5).ok for f in futures)
        assert len(service.groups) == 2


def test_timeout_flush_while_another_task_is_still_runnable():
    with LiveRangeService(max_wait_s=.002) as service:
        ready, slow = [service.register(name, None) for name in ("ready", "slow")]
        result = lane(ready, request())
        slow.finish()
        assert result.ok and service.groups[0]["reason"] == "timeout"
        assert service.wait_ns[0] < 1_000_000_000


def test_nan_is_task_local_and_failed_attempt_retains_boundary():
    original = {"x": torch.tensor([7.], dtype=torch.float64)}
    with LiveRangeService() as service:
        bad, good = [service.register(name, original) for name in ("bad", "good")]
        bad.begin_attempt()
        r = request()
        r = replace(r, coefficients_lo=torch.full_like(r.coefficients_lo, float("nan")))
        with ThreadPoolExecutor(2) as pool:
            failure = pool.submit(bad.evaluate, r)
            success = pool.submit(lane, good, request())
            assert failure.result(timeout=5).status == "invalid"
            bad.reject(failed=True)
            assert success.result(timeout=5).ok
        assert torch.equal(bad.accepted_state["x"], original["x"])


def test_cancel_inflight_checkpoint_restore_epoch_and_late_response():
    entered, release = Event(), Event()
    pending = []
    def delayed(rows, **kwargs):
        entered.set()
        assert release.wait(5)
        return evaluate_range_requests(rows, **kwargs)
    with LiveRangeService(evaluator=delayed) as service:
        old = service.register("same/id", {"x": torch.tensor([7.])})
        old.begin_attempt()
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(old.evaluate, request())
            assert entered.wait(5)
            pending.append(old.pending)
            saved = old.checkpoint_state()
            assert old.status == "CANCELLED" and old.pending is None
            restored = service.register("same/id", saved, generation=old.generation)
            assert restored.epoch == old.epoch + 1
            with pytest.raises(RangeCancelled):
                future.result(timeout=5)
            with pytest.raises(StaleRangeResponse):
                old.commit({"x": torch.tensor([99.])})
            release.set()
        result = lane(restored, request())
        assert result.ok
        assert service.counts["stale_discarded"] == 1
        assert float(saved["x"][0]) == 7.
        saved["x"][0] = 100.
        assert float(old.accepted_state["x"][0]) == 7.
        late = evaluate_range_requests([pending[0].request])[pending[0].identity.request_id]
        assert not service.deliver(pending[0], late)


def test_swapped_response_id_fails_without_commit_and_other_task_survives():
    def wrong(rows, **kwargs):
        outputs = evaluate_range_requests(rows, **kwargs)
        for r in rows:
            if "/3:bad/" in r.request_id:
                outputs[r.request_id] = replace(outputs[r.request_id], request_id="another-task")
        return outputs
    with LiveRangeService(evaluator=wrong) as service:
        bad, good = [service.register(name, 5) for name in ("bad", "good")]
        with ThreadPoolExecutor(2) as pool:
            failure = pool.submit(lane, bad, request())
            success = pool.submit(lane, good, request())
            with pytest.raises(StaleRangeResponse):
                failure.result(timeout=5)
            bad.reject(failed=True)
            assert success.result(timeout=5).ok
        assert bad.accepted_state == 5 and service.counts["identity_errors"] == 1


def test_retry_uses_new_attempt_and_only_consumed_predecessor():
    events = []
    with LiveRangeService(trace=events.append) as service:
        task = service.register("retry", 10)
        task.begin_attempt()
        first = task.evaluate(request())
        task.reject()
        assert task.generation == 0 and task.accepted_state == 10
        task.begin_attempt()
        second = task.evaluate(request())
        task.commit(20)
        task.finish()
    submitted = [e for e in events if e["event"] == "submit"]
    assert [e["attempt"] for e in submitted] == [1, 2]
    assert submitted[1]["previous"] == first.request_id
    assert first.request_id != second.request_id
    assert task.generation == 1


@pytest.mark.parametrize("fallback", [False, True])
def test_explicit_hardware_failure_policy(fallback):
    def broken(rows, **kwargs):
        raise RuntimeError("injected device failure")
    with LiveRangeService("cuda", evaluator=broken, hardware_fallback=fallback) as service:
        task = service.register("hardware", None)
        task.begin_attempt(diagnostic=True)
        result = task.evaluate(request())
        if fallback:
            check(request(), result)
            task.commit(1)
            task.finish()
            assert service.counts["hardware_fallback_requests"] == 1
            assert service.groups[0]["timing"]["failed_cuda_s"] > 0
        else:
            assert not result.ok
            task.reject(failed=True)
        assert service.counts["gpu_completed_requests"] == 0


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_worker_context_and_external_power_table_explicit_fallback(backend):
    events = []
    with LiveRangeService(backend, trace=events.append) as service:
        task = service.register("table", None)
        task.begin_attempt(diagnostic=True)
        with task.execution():
            value = evaluate_interval_normal(Polynomial({(2, 1): 2.}, 2),
                [Interval(0., .25), Interval(-1., 1.)], step_exp_table={2: Interval(0., .125)},
                state_var_indices=(1,), time_var_index=0)
        task.commit(value)
        task.finish()
        assert float(value.hi) >= .25
        assert service.counts["external_table_fallback_requests"] == 1
        assert service.counts["gpu_completed_requests"] == 0
    submit = next(e for e in events if e["event"] == "submit")
    returned = next(e for e in events if e["event"] == "return")
    check(submit["request"], returned["result"], require_terms=False, require_powers=False)


def test_second_outstanding_request_is_rejected():
    with LiveRangeService(max_wait_s=.1) as service:
        task = service.register("one", None)
        service.register("keep-runnable", None)
        task.begin_attempt()
        pending = service.submit(task, request())
        with pytest.raises(RuntimeError, match="outstanding"):
            service.submit(task, request())
        task.cancel()
        with pytest.raises(RangeCancelled):
            pending.future.result(timeout=5)


def test_live_checkpoint_preserves_private_diagnostics_and_mapping_order(tmp_path):
    from torch_tm_flowpipe.live_range_checkpoint import save_live_range_checkpoint, load_live_range_checkpoint
    from experiments.live_range_solver.runner import initial
    from experiments.boundary_execution.state_equivalence import canonical
    current, state = initial("van_der_pol", 0)
    state = replace(state, diagnostics={"z":.125, "_private":[("preserved", torch.tensor([.25], dtype=torch.float64))], "a":7})
    with LiveRangeService() as service:
        task = service.register("checkpoint", (current, state))
        save_live_range_checkpoint(tmp_path/"state", task, scheduler={}, contract={"plant":"van_der_pol"}, provenance={})
        resumed = load_live_range_checkpoint(tmp_path/"state", service)
        assert canonical(resumed.accepted_state) == canonical((current, state))
        assert resumed.epoch == task.epoch+1
        assert resumed.accepted_state[1].diagnostics["_private"][0][1].data_ptr() != state.diagnostics["_private"][0][1].data_ptr()
    with LiveRangeService(run_id=service.run_id) as fresh:
        resumed = load_live_range_checkpoint(tmp_path/"state", fresh)
        assert resumed.epoch > task.epoch
        assert canonical(resumed.accepted_state) == canonical((current, state))
