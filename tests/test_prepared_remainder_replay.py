"""Exact replay, lifetime and failure checks on actual repaired solver inputs."""
import copy
from dataclasses import fields, is_dataclass

import pytest
import torch

import torch_tm_flowpipe.batched_dense_tm as d
from torch_tm_flowpipe.prepared_remainder_replay import (
    PlanInvalidated, PreparedRemainderReplay, is_enabled, prepared_remainder_replay,
)
from torch_tm_flowpipe import PolynomialODE, PolynomialODETerm
from torch_tm_flowpipe.terminal_checkpoint import tmvector_hashes
from torch_tm_flowpipe import accepted_boundary_sr_queue_sha256, load_terminal_checkpoint
from experiments.endpoint_roundoff_repair.frozen import ROOT, setup, step


def exact(value):
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), exact(value.detach().cpu().tolist()))
    if isinstance(value, float):
        return value.hex()
    if is_dataclass(value):
        return {f.name: exact(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: exact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact(v) for v in value]
    return value


@pytest.fixture(scope="module")
def captured():
    torch.set_num_threads(1)
    captures = []
    original = d._post_accept_refine_raw_remainder
    def capture(*args, **kwargs):
        captures.append((args, kwargs))
        return original(*args, **kwargs)
    d._post_accept_refine_raw_remainder = capture
    try:
        _, current, state = setup("brusselator")
        segment = step("brusselator", current, state, 1)
        assert segment.status == "validated"
    finally:
        d._post_accept_refine_raw_remainder = original
    return captures[0]


def binding(captured):
    args, kwargs = captured
    rhs, base, candidate = args
    parameters = {k: kwargs[k] for k in ("tau_index", "order", "cutoff_threshold", "validation_eps")}
    return rhs, base, candidate, parameters


def reference_image(rhs, base, candidate, params, lo, hi, evidence=True):
    return d._dense_flowstar_raw_compat_image(
        rhs, base, candidate.with_remainder(lo, hi), candidate,
        **params, raw_rhs_evaluation="ordered_terms", record_evidence=evidence,
    )


def test_same_actual_remainders_all_proposals_coefficients_and_ledgers(captured):
    rhs, base, candidate, params = binding(captured)
    plan = PreparedRemainderReplay(rhs, base, candidate, **params)
    lo, hi = captured[1]["retained_lo"], captured[1]["retained_hi"]
    proposals = []
    for _ in range(8):
        old = reference_image(rhs, base, candidate, params, lo, hi)
        new = plan.image(lo, hi)
        assert exact(old) == exact(new)
        assert exact(d._atomic_refinement_decision(lo, hi, *old[:2])) == exact(d._atomic_refinement_decision(lo, hi, *new[:2]))
        proposals.append(exact(old[:2]))
        lo, hi = old[:2]
    assert proposals[0] != proposals[1]
    assert plan._raw.hits and plan._regular.hits


@pytest.mark.parametrize("limit", [0, 1, 491])
def test_independent_replay_loops_and_observer_switch(captured, limit):
    args, kwargs = captured
    results = []
    for enabled in (False, True):
        for observer in (d.DENSE_OBSERVER_NONE, d.DENSE_OBSERVER_LIGHTWEIGHT, d.DENSE_OBSERVER_FULL):
            counters = d.DenseExecutionCounters()
            with prepared_remainder_replay(enabled):
                output = d._post_accept_refine_raw_remainder(
                    *args, **{**kwargs, "observer_mode": observer, "counters": counters, "replay_limit": limit})
            numerical_counters = {k: v for k, v in counters.as_dict().items() if not k.startswith("prepared_")}
            results.append((enabled, observer, exact(output), exact(numerical_counters)))
            assert counters.prepared_plan_count == int(enabled and limit > 0)
    for observer in (d.DENSE_OBSERVER_NONE, d.DENSE_OBSERVER_LIGHTWEIGHT, d.DENSE_OBSERVER_FULL):
        old, new = [r for r in results if r[1] == observer]
        assert old[2:] == new[2:]
    assert all(row[2][:3] == results[0][2][:3] for row in results)


@pytest.mark.parametrize("change", ["candidate", "base", "domain", "h", "order", "cutoff", "ODE"])
def test_binding_changes_reject_old_plan(captured, change):
    rhs, base, candidate, params = binding(captured)
    base, candidate = copy.deepcopy((base, candidate))
    plan = PreparedRemainderReplay(rhs, base, candidate, **params)
    if change == "candidate": candidate = candidate.clone()
    elif change == "base": base = base.clone()
    elif change == "domain": candidate.domain_lo[0, 0] -= .01
    elif change == "h": candidate.domain_hi[0, params["tau_index"]] += .01
    elif change == "order": params = {**params, "order": params["order"] - 1}
    elif change == "cutoff": params = {**params, "cutoff_threshold": params["cutoff_threshold"] * 2}
    elif change == "ODE": rhs = lambda x: x
    with pytest.raises(PlanInvalidated):
        plan.check(rhs, base, candidate, **params)


def test_inplace_input_detection_and_returned_output_isolation(captured):
    rhs, base, candidate, params = binding(captured)
    base, candidate = copy.deepcopy((base, candidate))
    plan = PreparedRemainderReplay(rhs, base, candidate, **params)
    lo, hi = captured[1]["retained_lo"], captured[1]["retained_hi"]
    result = plan.image(lo, hi)
    saved = exact(result)
    result[0].add_(1.)
    result[3].ledger.entries["polynomial_truncation"][0].add_(1.)
    assert exact(plan.image(lo, hi)) == saved
    candidate.poly.coeffs.add_(0.)
    with pytest.raises(PlanInvalidated, match="in place"):
        plan.image(lo, hi)


@pytest.mark.parametrize("case", ["zero", "subnormal", "nan", "inf", "overflow"])
def test_dynamic_extremes_follow_reference(captured, case):
    rhs, base, candidate, params = binding(captured)
    plan = PreparedRemainderReplay(rhs, base, candidate, **params)
    value = {"zero": 0., "subnormal": float.fromhex("0x0.0000000000001p-1022"),
             "nan": float("nan"), "inf": float("inf"), "overflow": 1e308}[case]
    hi = torch.full_like(candidate.rem_hi, value); lo = -hi
    outcomes = []
    for evaluator in (lambda: reference_image(rhs, base, candidate, params, lo, hi), lambda: plan.image(lo, hi)):
        try:
            outcomes.append(("result", exact(evaluator())))
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            outcomes.append(("error", type(exc)))
    assert outcomes[0] == outcomes[1]


def test_atomic_decision_boundaries_unmodified():
    lo, hi = torch.tensor([[-1., -2.]], dtype=torch.float64), torch.tensor([[1., 2.]], dtype=torch.float64)
    assert d._atomic_refinement_decision(lo, hi, lo, hi)[:3] == (True, False, "fixed_point")
    # One component stagnates while the other meaningfully contracts.
    decision = d._atomic_refinement_decision(lo, hi, torch.tensor([[-1., -1.]], dtype=torch.float64), torch.tensor([[1., 1.]], dtype=torch.float64))
    assert decision[0] and decision[1]
    assert not d._atomic_refinement_decision(lo, hi, lo-1., hi)[0]


@pytest.mark.parametrize("case", ["fixed_point", "different_components", "subset_failure", "nonfinite_failure"])
def test_loop_dispatch_keeps_atomic_commit_and_failure_rollback(captured, monkeypatch, case):
    args, kwargs = captured
    decomposition = kwargs["retained_decomposition"]
    sequences = []
    for enabled in (False, True):
        calls = []
        def image(rhs, base, target, candidate, **options):
            lo, hi = target.rem_lo, target.rem_hi
            calls.append(exact((lo, hi)))
            if case == "nonfinite_failure":
                raise FloatingPointError("controlled dynamic failure")
            if case == "subset_failure":
                return lo-1., hi, {}, decomposition
            if case == "different_components" and len(calls) == 1:
                lo, hi = lo.clone(), hi.clone()
                lo[:, 1] *= .5; hi[:, 1] *= .5
            return lo.clone(), hi.clone(), {}, decomposition
        monkeypatch.setattr(d, "_dense_flowstar_raw_compat_image", image)
        with prepared_remainder_replay(enabled):
            output = d._post_accept_refine_raw_remainder(*args, **{**kwargs, "observer_mode": d.DENSE_OBSERVER_LIGHTWEIGHT})
        sequences.append((calls, exact(output)))
    assert sequences[0] == sequences[1]
    if case in {"subset_failure", "nonfinite_failure"}:
        assert sequences[0][1][:2] == exact((kwargs["retained_lo"], kwargs["retained_hi"]))


def test_failed_preparation_cannot_leak_partial_tape_into_retry(captured):
    rhs, base, candidate, params = binding(captured)
    plan = PreparedRemainderReplay(rhs, base, candidate, **params)
    huge = torch.full_like(candidate.rem_hi, 1e308)
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        plan.image(-huge, huge)
    lo, hi = captured[1]["retained_lo"], captured[1]["retained_hi"]
    with pytest.raises(PlanInvalidated, match="new plan"):
        plan.image(lo, hi)
    fresh = PreparedRemainderReplay(rhs, base, candidate, **params)
    assert exact(fresh.image(lo, hi)) == exact(reference_image(rhs, base, candidate, params, lo, hi))


def test_vdp_adaptive_rejections_and_next_h_match_repaired_archive():
    _, current, state = setup("van_der_pol")
    before = tmvector_hashes(current)
    archive = ROOT/"artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/vdp_adaptive"
    import json
    original = json.loads((archive/"endpoint_audit.jsonl").read_text().splitlines()[0])
    results = []
    for enabled in (False, True):
        with prepared_remainder_replay(enabled):
            result = step("van_der_pol", current, state, 1, h=.1, adaptive=True)
        assert result.status == "validated"
        assert result.step_rejections == original["step_rejections"] > 0
        assert result.h.hex() == original["h_hex"]
        assert tmvector_hashes(current) == before
        results.append((result.h.hex(), result.next_h.hex(), result.step_rejections,
                        tmvector_hashes(result.reset_tm), accepted_boundary_sr_queue_sha256(result.flowstar_normal_state.symbolic_queue)))
    assert results[0] == results[1]


def test_b2_different_dynamic_lanes_and_interleaved_plans(captured):
    _, base, candidate, params = binding(captured)
    def two(model):
        return d.BatchedTaylorModel(
            d.BatchedPolynomial(model.poly.coeffs.repeat(2, 1, 1), model.poly.basis),
            model.rem_lo.repeat(2, 1), model.rem_hi.repeat(2, 1),
            model.domain_lo.repeat(2, 1), model.domain_hi.repeat(2, 1),
            d.DenseRemainderLedger({k: (lo.repeat(2, 1), hi.repeat(2, 1)) for k, (lo, hi) in model.ledger.entries.items()}),
            model.range_policy,
        )
    base, candidate = two(base), two(candidate)
    # Existing two-state polynomial algebra only; this is not a batched solve.
    rhs = PolynomialODE(((PolynomialODETerm(1., (1, 1)),), (PolynomialODETerm(-1., (2, 0)),)), 2)
    plans = [PreparedRemainderReplay(rhs, base, candidate, **params) for _ in range(2)]
    for i in range(4):
        hi = torch.tensor([[1e-6*(i+1), 2e-7], [4e-5, 3e-7*(i+1)]], dtype=torch.float64)
        old = reference_image(rhs, base, candidate, params, -hi, hi)
        assert exact(plans[i % 2].image(-hi, hi)) == exact(old)


@pytest.mark.parametrize("plant,folder", [("brusselator", "brusselator_full"), ("van_der_pol", "vdp_full")])
def test_restored_checkpoint_next_step_and_queue(captured, plant, folder):
    restored = load_terminal_checkpoint(ROOT/"artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal"/folder/"checkpoint_0120", expected_dtype="float64")
    before = accepted_boundary_sr_queue_sha256(restored.normal_state.symbolic_queue)
    results = []
    for enabled in (False, True):
        with prepared_remainder_replay(enabled):
            result = step(plant, restored.current, restored.normal_state, 121)
        assert result.status == "validated"
        results.append({
            "models": [tmvector_hashes(getattr(result, name)) for name in ("tm", "endpoint_raw_tm", "reset_tm")],
            "state_models": [tmvector_hashes(result.flowstar_normal_state.tmv_pre), tmvector_hashes(result.flowstar_normal_state.tmv_right)],
            "queue": accepted_boundary_sr_queue_sha256(result.flowstar_normal_state.symbolic_queue),
            "errors": exact(result.endpoint_substitution_roundoff),
            "ledger": exact(result.dense_endpoint_ledger),
            "replays": {k: v for k, v in result.backend_counters.items() if k.startswith("post_accept_")},
        })
        assert before == accepted_boundary_sr_queue_sha256(restored.normal_state.symbolic_queue)
    assert results[0] == results[1]


def test_context_is_opt_in_nested_and_exception_safe():
    assert not is_enabled()
    with pytest.raises(RuntimeError):
        with prepared_remainder_replay():
            assert is_enabled()
            with prepared_remainder_replay(False):
                assert not is_enabled()
            assert is_enabled()
            raise RuntimeError("test")
    assert not is_enabled()
