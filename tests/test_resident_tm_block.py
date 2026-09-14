from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import threading

import pytest
import torch

from experiments.resident_tm_block.exact_oracle import verify_result
from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.flowpipe import insert_ctrunc_normal_dependency_preserving
from torch_tm_flowpipe.live_range_service import LiveRangeService, RangeCancelled
from torch_tm_flowpipe.resident_tm_block import (
    ResidentNormalCompositionResult,
    ResidentTMBlockExecutor,
    active_dispatch,
    request_from_taylor_models,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _tm(terms, domain, order, remainder=(0.0, 0.0), split=None):
    return TaylorModel(
        Polynomial(terms, 2),
        Interval(*remainder),
        domain,
        order=order,
        truncation_range_split=split,
    )


def _request(index: int, *, order: int = 4, interval_coefficients: bool = False):
    domain = [Interval(-1.0, 1.0), Interval(-0.75, 1.0)]
    scale = 1.0 + index * 2.0**-12
    outer = TMVector(
        [
            _tm(
                {
                    (1, 0): 1.1 * scale,
                    (0, 2): -0.3,
                    (3, 0): 2.0e-9,
                    **({(0, 1): -2.0e-11} if index % 2 else {}),
                },
                domain,
                order,
                (1.0e-12, 2.0e-12),
                2 if index % 3 == 0 else None,
            ),
            _tm(
                {
                    (0, 1): -0.75,
                    (2, 1): 0.125 * scale,
                    (0, 0): 3.0e-11,
                    **({(1, 1): 4.0e-12} if index % 4 else {}),
                },
                domain,
                order,
                (-2.0e-12, 4.0e-12),
            ),
        ]
    )
    inner = TMVector(
        [
            _tm(
                {(1, 0): 1.1, (0, 1): 0.1, (2, 0): 2.0e-5},
                domain,
                order,
                (-1.0e-6, 2.0e-6),
            ),
            _tm(
                {(0, 1): 0.9, (1, 0): -0.05, (0, 2): -3.0e-5},
                domain,
                order,
                (-2.0e-6, 1.0e-6),
                3 if index % 5 == 0 else None,
            ),
        ]
    )
    request = request_from_taylor_models(
        f"lane-{index}", outer, inner, order, 1.0e-10, domain
    )
    assert request is not None
    if interval_coefficients:
        outer_lo = request.outer_lo.clone()
        outer_hi = request.outer_hi.clone()
        inner_lo = request.inner_lo.clone()
        inner_hi = request.inner_hi.clone()
        neg_inf = torch.full((), -torch.inf, dtype=torch.float64)
        pos_inf = torch.full((), torch.inf, dtype=torch.float64)
        outer_lo[0, 1] = torch.nextafter(outer_lo[0, 1], neg_inf)
        outer_hi[0, 1] = torch.nextafter(outer_hi[0, 1], pos_inf)
        inner_lo[1, 2] = torch.nextafter(inner_lo[1, 2], neg_inf)
        inner_hi[1, 2] = torch.nextafter(inner_hi[1, 2], pos_inf)
        request = replace(
            request,
            outer_lo=outer_lo,
            outer_hi=outer_hi,
            inner_lo=inner_lo,
            inner_hi=inner_hi,
        )
    return request


@pytest.fixture(scope="module")
def executor():
    return ResidentTMBlockExecutor()


def _result_signature(result):
    assert result.ok
    models = result.output.models
    return (
        tuple(
            (
                tuple(
                    (exp, float(value).hex())
                    for exp, value in model.polynomial.terms.items()
                ),
                float(model.remainder.lo).hex(),
                float(model.remainder.hi).hex(),
                model.truncation_range_split,
            )
            for model in models
        ),
        result.coefficient_error_lo.numpy().tobytes(),
        result.coefficient_error_hi.numpy().tobytes(),
    )


def test_retained_coefficient_roundoff_is_paid_and_legacy_misses_it(executor):
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    outer = TMVector([_tm({(1, 0): 1.1}, domain, 4)])
    inner = TMVector(
        [_tm({(1, 0): 1.1}, domain, 4), _tm({(0, 1): 1.0}, domain, 4)]
    )
    legacy = insert_ctrunc_normal_dependency_preserving(
        outer, inner, 4, None, domain
    )[0]
    request = request_from_taylor_models("roundoff", outer, inner, 4, None, domain)
    assert request is not None
    result = executor.evaluate([request])["roundoff"]
    checks = verify_result(request, result)
    exact_error = Fraction(1.1) * Fraction(1.1) - Fraction(float(1.1 * 1.1))
    assert exact_error != 0
    assert not (
        Fraction(float(legacy.remainder.lo))
        <= exact_error
        <= Fraction(float(legacy.remainder.hi))
    )
    assert checks["coefficient_error_interval_checks"] == len(request.outer_point[0])
    assert result.diagnostics["retained_coefficient_roundoff_width"] > 0.0


@pytest.mark.parametrize("order", [4, 6])
def test_fraction_oracle_point_interval_cutoff_truncation_and_splits(executor, order):
    request = _request(order, order=order, interval_coefficients=True)
    result = executor.evaluate([request])[request.request_id]
    checks = verify_result(request, result)
    assert checks == {
        "point_coefficient_checks": 2 * len(request.outer_point[0]),
        "coefficient_error_interval_checks": 2 * len(request.outer_point[0]),
        "remainder_interval_checks": 2,
        "independent_numeric_type": "fractions.Fraction",
        "cuda_interval_helpers_called": False,
        "legacy_cpu_result_used_as_truth": False,
    }
    assert result.diagnostics["truncation_width"] >= 0.0
    assert result.diagnostics["cutoff_width"] > 0.0


@pytest.mark.parametrize("batch", [1, 2, 8, 32])
def test_batch_reorder_and_chunking_are_bitwise_task_local(executor, batch):
    requests = [
        _request(index, interval_coefficients=index % 3 == 1)
        for index in range(batch)
    ]
    together = executor.evaluate(requests)
    reversed_results = executor.evaluate(list(reversed(requests)))
    chunks = {}
    for begin in range(0, batch, 3):
        chunks.update(executor.evaluate(requests[begin : begin + 3]))
    singles = {}
    for request in requests:
        singles.update(executor.evaluate([request]))
    for request in requests:
        expected = _result_signature(together[request.request_id])
        assert _result_signature(reversed_results[request.request_id]) == expected
        assert _result_signature(chunks[request.request_id]) == expected
        assert _result_signature(singles[request.request_id]) == expected
        verify_result(request, together[request.request_id])


def test_nonfinite_and_wrong_program_fingerprint_fail_closed(executor):
    request = _request(0)
    wrong = replace(request, request_id="wrong-program", basis_fingerprint="0" * 64)
    assert executor.evaluate([wrong])[wrong.request_id].status == "unsupported_structure"
    point = request.inner_point.clone()
    lo = request.inner_lo.clone()
    hi = request.inner_hi.clone()
    point[0, 0] = torch.inf
    lo[0, 0] = torch.inf
    hi[0, 0] = torch.inf
    nonfinite = replace(
        request,
        request_id="nonfinite",
        inner_point=point,
        inner_lo=lo,
        inner_hi=hi,
    )
    result = executor.evaluate([nonfinite])[nonfinite.request_id]
    assert result.status == "nonfinite"
    assert result.output is None


def test_mixed_structures_are_not_silently_coexecuted(executor):
    order4 = _request(4, order=4)
    order6 = _request(6, order=6)
    results = executor.evaluate([order4, order6])
    assert {result.status for result in results.values()} == {"unsupported_structure"}


def test_default_dispatch_is_off():
    assert active_dispatch() is None


def test_unsupported_variable_count_does_not_enter_device_block():
    domain = [Interval(-1.0, 1.0)]
    outer = TaylorModel(
        Polynomial({(1,): torch.tensor(1.0, dtype=torch.float64)}, 1),
        Interval(0.0, 0.0), domain, order=4,
    )
    inner = TMVector([outer])
    assert request_from_taylor_models("dim1", outer, inner, 4, None, domain) is None


def test_cancelled_resident_result_is_discarded(monkeypatch):
    request = _request(0)
    entered = threading.Event()
    release = threading.Event()
    observed = []
    with LiveRangeService(
        "cuda",
        run_id="resident-cancel-test",
        resident_tm_block=True,
        max_wait_s=0.001,
    ) as service:
        task = service.register("lane", accepted_state=("accepted", 0))
        task.begin_attempt()

        def delayed(requests, **_kwargs):
            entered.set()
            assert release.wait(10.0)
            return {
                item.request_id: ResidentNormalCompositionResult(
                    item.request_id, "backend_error", message="late test result"
                )
                for item in requests
            }

        monkeypatch.setattr(service.resident_executor, "evaluate", delayed)

        def worker():
            try:
                task.evaluate_resident(request)
            except BaseException as error:  # assertion below checks exact class
                observed.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        assert entered.wait(10.0)
        task.cancel()
        release.set()
        thread.join(10.0)
        assert not thread.is_alive()
        assert len(observed) == 1 and isinstance(observed[0], RangeCancelled)
        assert task.accepted_state == ("accepted", 0)
    # Service shutdown joins the owner thread after its late delivery attempt.
    assert service.counts["stale_discarded"] == 1


def test_execution_receipt_and_timing_denominator(executor):
    requests = [_request(index) for index in range(8)]
    timing = {}
    results = executor.evaluate(requests, timings=timing)
    assert all(result.ok for result in results.values())
    assert timing["receipt"][1] == 16
    assert timing["kernel_invocations"] == 1
    assert timing["host_synchronizations"] == 1
    assert timing["numeric_h2d_bytes"] > 0
    assert timing["numeric_d2h_bytes"] > 0
    assert timing["block_host_span_s"] >= timing["kernel_cuda_event_s"]
    assert timing["transfer_and_sync_host_span_s"] >= timing["kernel_cuda_event_s"]
    assert timing["post_d2h_checks_and_rebuild_s"] >= 0.0
    assert timing["packing_s"] > 0.0
