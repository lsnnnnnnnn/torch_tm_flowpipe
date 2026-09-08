"""Bit-level execution parity and lifetime boundaries for prepared replay."""
from dataclasses import replace
from pathlib import Path
import math

import pytest
import torch

import torch_tm_flowpipe.batched_dense_tm as d
from torch_tm_flowpipe import PolynomialODE, PolynomialODETerm, load_terminal_checkpoint
from torch_tm_flowpipe.terminal_checkpoint import tmvector_hashes
from torch_tm_flowpipe.prepared_remainder_replay import (
    PreparedRemainderReplay, prepared_remainder_replay, is_enabled, supports,
)
from torch_tm_flowpipe.ode_examples import brusselator_ode
from experiments.endpoint_roundoff_repair.frozen import setup, step
from test_brusselator_c4_generic_refinement import _base, COMMON, C4


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def same_image(a, b):
    bits(a[0], b[0]); bits(a[1], b[1])
    assert a[2] == b[2]
    assert a[3].ledger.entries.keys() == b[3].ledger.entries.keys()
    for key in a[3].ledger.entries:
        for x,y in zip(a[3].ledger.entries[key], b[3].ledger.entries[key]): bits(x,y)
    bits(a[3].decomposition_lo, b[3].decomposition_lo)
    bits(a[3].decomposition_hi, b[3].decomposition_hi)
    assert torch.equal(a[3].contains_image, b[3].contains_image)


@pytest.fixture(scope='module')
def fixed():
    base=replace(_base(), range_trace=None)
    candidate,_=d.dense_polynomial_picard(brusselator_ode,base.without_remainder(),
                                         tau_index=2,order=6,cutoff_threshold=1e-10,
                                         observer_mode=d.DENSE_OBSERVER_NONE)
    return base,candidate


def image_args():
    return dict(tau_index=2,order=6,cutoff_threshold=1e-10,validation_eps=1e-12)


def image(ode,base,candidate,lo,hi,plan=None):
    return d._dense_flowstar_raw_compat_image(ode,base,candidate.with_remainder(lo,hi),candidate,
                                             prepared_replay=plan,**image_args())


def test_same_candidate_different_remainders_all_ledgers_and_coefficients(fixed):
    base,candidate=fixed
    plan=PreparedRemainderReplay(brusselator_ode,base,candidate,**image_args())
    proposals=[]
    for lo,hi in [(-1e-4,1e-4),(-1e-6,3e-6),(0.,0.),(-math.ulp(0.),math.ulp(0.)),(-2e-5,1e-5)]:
        lo=torch.full_like(candidate.rem_lo,lo); hi=torch.full_like(candidate.rem_hi,hi)
        reference=image(brusselator_ode,base,candidate,lo,hi)
        optimized=image(brusselator_ode,base,candidate,lo,hi,plan)
        same_image(reference,optimized)
        proposals.append(optimized[0])
    assert not torch.equal(proposals[0],proposals[1])
    assert plan.hits > plan.misses > 0


@pytest.mark.parametrize('change',['candidate','domain','h','order','cutoff','ode','base','scale','inplace'])
def test_binding_rejects_other_attempt_or_mutation(fixed,change):
    base,candidate=(m.clone() for m in fixed)
    ode=brusselator_ode
    plan=PreparedRemainderReplay(ode,base,candidate,**image_args())
    args=image_args()
    if change=='candidate': candidate=candidate.clone()
    elif change=='domain': candidate=replace(candidate,domain_hi=candidate.domain_hi+0.001)
    elif change=='h': args['tau_index']=1
    elif change=='order': args['order']=5
    elif change=='cutoff': args['cutoff_threshold']=1e-11
    elif change=='ode': ode=PolynomialODE(((PolynomialODETerm(1.,(1,0)),),)*2,2)
    elif change=='base': base=base.clone()
    elif change=='scale': base.poly.coeffs.mul_(1.1)
    else: candidate.poly.coeffs.add_(1e-10)
    with pytest.raises(ValueError,match='attempt'):
        plan.validate_binding(ode,base,candidate,**args)


def test_private_snapshots_and_exposed_storage_mutation_detection(fixed):
    base,candidate=(m.clone() for m in fixed)
    plan=PreparedRemainderReplay(brusselator_ode,base,candidate,**image_args())
    lo,hi=torch.full_like(candidate.rem_lo,-1e-6),torch.full_like(candidate.rem_hi,1e-6)
    first=plan.parts(lo,hi)
    before=first[0].poly.coeffs.clone()
    candidate.poly.coeffs.add_(1.)
    after=plan.parts(lo,hi)
    bits(before,after[0].poly.coeffs)  # original storage is safely isolated
    after[0].poly.coeffs.add_(1.)
    with pytest.raises(ValueError,match='storage'):
        plan.parts(lo,hi)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),1e308])
def test_nonfinite_and_overflow_same_failure(fixed,value):
    base,candidate=fixed
    plan=PreparedRemainderReplay(brusselator_ode,base,candidate,**image_args())
    outcomes=[]
    for mode in [None,plan]:
        try:
            result=image(brusselator_ode,base,candidate,torch.full_like(candidate.rem_lo,-value),
                         torch.full_like(candidate.rem_hi,value),mode)
            outcomes.append(result)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            outcomes.append((type(exc),str(exc)))
    if isinstance(outcomes[0][0],type): assert outcomes[0] == outcomes[1]
    else: same_image(*outcomes)


def test_independent_complete_loops_and_observer_modes():
    results=[]
    for mode,observer in [(False,d.DENSE_OBSERVER_FULL),(True,d.DENSE_OBSERVER_FULL),
                          (True,d.DENSE_OBSERVER_NONE),(True,d.DENSE_OBSERVER_LIGHTWEIGHT)]:
        with prepared_remainder_replay(mode):
            results.append(d.dense_picard_validate_step(brusselator_ode,_base(),validation_mode=C4,
                                                         observer_mode=observer,**COMMON))
    reference=results[0]
    assert [r for r in reference.trace if r.get('phase')=='post_accept_refinement'] == [
        r for r in results[1].trace if r.get('phase')=='post_accept_refinement']
    for optimized in results[1:]:
        assert reference.status == optimized.status == 'validated'
        for name in ['segment_tm','raw_endpoint']:
            a,b=getattr(reference,name),getattr(optimized,name)
            bits(a.poly.coeffs,b.poly.coeffs); bits(a.rem_lo,b.rem_lo); bits(a.rem_hi,b.rem_hi)
        for k,v in reference.raw_endpoint.ledger.entries.items():
            for a,b in zip(v,optimized.raw_endpoint.ledger.entries[k]): bits(a,b)


def test_zero_replays_and_failed_first_acceptance_do_not_prepare(monkeypatch):
    def forbidden(*a,**k): raise AssertionError('unexpected preparation')
    monkeypatch.setattr(PreparedRemainderReplay,'__init__',forbidden)
    with prepared_remainder_replay():
        result=d.dense_picard_validate_step(brusselator_ode,_base(),validation_mode=C4,
                    **{**COMMON,'target_remainder_radius':1e-8})
        assert result.status=='failed'
        lo,hi=torch.zeros((1,2),dtype=torch.float64),torch.zeros((1,2),dtype=torch.float64)
        out=d._post_accept_refine_raw_remainder(None,None,None,retained_lo=lo,retained_hi=hi,
              retained_decomposition=None,tau_index=2,order=6,cutoff_threshold=1e-10,
              validation_eps=1e-12,structural_fingerprint=None,replay_limit=0,
              observer_mode=d.DENSE_OBSERVER_NONE)
        assert out[0] is lo and out[1] is hi


@pytest.mark.parametrize('scenario',['fixed_point','mixed','subset_failure','nan','exception'])
def test_decision_and_failed_proposal_preserve_accepted_state(monkeypatch,scenario):
    original=d._dense_flowstar_raw_compat_image
    output=[]
    for mode in [False,True]:
        calls=0
        def inject(*a,**kw):
            nonlocal calls
            result=original(*a,**kw); calls+=1
            if calls<2:return result
            if scenario=='exception':raise FloatingPointError('injected after real evaluator')
            old=a[2]
            if scenario=='fixed_point':lo,hi=old.rem_lo,old.rem_hi
            elif scenario=='mixed':
                lo=old.rem_lo*torch.tensor([[1.,0.5]]); hi=old.rem_hi*torch.tensor([[1.,0.5]])
            elif scenario=='subset_failure':lo,hi=old.rem_lo-1.,old.rem_hi+1.
            else:lo,hi=old.rem_lo*float('nan'),old.rem_hi
            return lo,hi,*result[2:]
        monkeypatch.setattr(d,'_dense_flowstar_raw_compat_image',inject)
        with prepared_remainder_replay(mode):
            output.append(d.dense_picard_validate_step(brusselator_ode,_base(),validation_mode=C4,**COMMON))
    for attr in ['rem_lo','rem_hi']:
        bits(getattr(output[0].segment_tm,attr),getattr(output[1].segment_tm,attr))
    assert [(x['committed'],x['stop_reason']) for x in output[0].trace if x.get('phase')=='post_accept_refinement']==[
        (x['committed'],x['stop_reason']) for x in output[1].trace if x.get('phase')=='post_accept_refinement']


def test_local_b2_polynomial_graph_has_distinct_dynamic_lanes(fixed):
    base,candidate=fixed
    def batch(m):
        return d.BatchedTaylorModel(d.BatchedPolynomial(m.poly.coeffs.repeat(2,1,1),m.poly.basis),
            m.rem_lo.repeat(2,1),m.rem_hi.repeat(2,1),m.domain_lo.repeat(2,1),m.domain_hi.repeat(2,1),
            d.DenseRemainderLedger({k:(a.repeat(2,1),b.repeat(2,1)) for k,(a,b) in m.ledger.entries.items()}),m.range_policy)
    base,candidate=batch(base),batch(candidate)
    ode=PolynomialODE(((PolynomialODETerm(1.,(1,1)),),(PolynomialODETerm(-2.,(2,0)),)),2)
    plan=PreparedRemainderReplay(ode,base,candidate,**image_args())
    for radii in [[[1e-6,2e-6],[1e-4,3e-5]],[[3e-5,1e-4],[2e-6,1e-6]]]:
        hi=torch.tensor(radii,dtype=torch.float64);lo=-hi
        same_image(image(ode,base,candidate,lo,hi),image(ode,base,candidate,lo,hi,plan))


@pytest.mark.parametrize('plant,name',[('brusselator','brusselator_full'),('van_der_pol','vdp_full')])
def test_repaired_checkpoint_resume_and_input_queue_rollback(plant,name):
    from torch_tm_flowpipe import accepted_boundary_sr_queue_sha256
    path=Path(__file__).resolve().parents[1]/'artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal'/name/'checkpoint_0120'
    restored=load_terminal_checkpoint(path)
    before=accepted_boundary_sr_queue_sha256(restored.normal_state.symbolic_queue)
    results=[]
    for mode in [False,True]:
        with prepared_remainder_replay(mode):
            results.append(step(plant,restored.current,restored.normal_state,121))
        assert before==accepted_boundary_sr_queue_sha256(restored.normal_state.symbolic_queue)
    for attr in ['tm','endpoint_raw_tm','reset_tm']:
        assert tmvector_hashes(getattr(results[0],attr))==tmvector_hashes(getattr(results[1],attr))
    assert accepted_boundary_sr_queue_sha256(results[0].flowstar_normal_state.symbolic_queue)==accepted_boundary_sr_queue_sha256(results[1].flowstar_normal_state.symbolic_queue)


def test_opt_in_context_restores_and_opaque_rhs_is_not_prepared():
    assert not is_enabled()
    with prepared_remainder_replay():
        assert is_enabled()
        with prepared_remainder_replay(False): assert not is_enabled()
        assert is_enabled()
    assert not is_enabled()
    assert not supports(lambda x: x)
