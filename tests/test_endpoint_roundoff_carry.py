"""Actual next-state consumers, ownership, checkpoint and failure boundaries."""
from fractions import Fraction as F

import pytest
import torch

from torch_tm_flowpipe import (
    Interval, Polynomial, TaylorModel, TMVector, FlowstarNormalFlowpipeState,
    accepted_boundary_sr_queue_sha256, load_terminal_checkpoint, save_terminal_checkpoint, tmvector_hashes,
)
from torch_tm_flowpipe.batched_dense_tm import BatchedTaylorModel, sparse_tmvector_to_dense, dense_to_sparse_tmvector
from torch_tm_flowpipe.flowpipe import FlowpipeSegment, _flowstar_normalized_insertion_transition
from experiments.endpoint_roundoff_repair.frozen import setup, step


@pytest.mark.parametrize('entry', ['published', 'dense_internal'])
def test_two_substitutions_carry_complete_remainder(entry):
    source = TMVector([TaylorModel(Polynomial({(0,0):-1.,(0,1):100.},2),Interval.zero(),
                                  [Interval(-2.,.5),Interval(0.,.01)],order=4)])
    if entry == 'published':
        first=source.substitute_const(1,.01).drop_variable(1)
    else:
        first=dense_to_sparse_tmvector(sparse_tmvector_to_dense(source,order=4).endpoint(1,.01))
    old=tmvector_hashes(first)
    # The next model consumes the entire returned first-step remainder.
    following=first.extend_domain(Interval(0.,.02))
    if entry == 'published':
        second=following.substitute_const(1,.02).drop_variable(1)
    else:
        second=dense_to_sparse_tmvector(sparse_tmvector_to_dense(following,order=4).endpoint(1,.02))
    exact=-1+100*F(.01)
    for model in [first[0],second[0]]:
        assert F(float(model.remainder.lo)) <= exact <= F(float(model.remainder.hi))
    assert tmvector_hashes(first)==old


def test_normal_maps_current_owner_history_and_capacity_reanchor():
    h=.01
    domain=[Interval(-1.,1.),Interval(-1.,1.)]
    initial=FlowstarNormalFlowpipeState.from_initial_box([(-1.,1.),(-1.,1.)],4)
    source=TMVector(TaylorModel(Polynomial({(0,0,0):-1.,(0,0,1):100.,
                         tuple(int(j==i) for j in range(2))+(0,):2.**-20},3),
                         Interval.zero(),domain+[Interval(0.,h)],order=4) for i in range(2))
    endpoint,errors=source.substitute_const_with_roundoff(2,h)
    endpoint=endpoint.drop_variable(2)
    first=FlowpipeSegment(source,endpoint,'validated',h,4,1,endpoint_substitution_roundoff=errors)
    reset,state,diagnostic=_flowstar_normalized_insertion_transition(
        first,initial,4,cutoff_threshold=None,generic_accepted_boundary_sr=True,symbolic_queue_max_size=2)
    exact=-1+100*F(h)
    assert len(state.symbolic_queue.J)==1
    for owner,error in zip(state.symbolic_queue.J[0],errors):
        assert F(float(error.lo)) <= exact <= F(float(error.hi))
        assert owner.contains_interval(error)
    queue_hash=accepted_boundary_sr_queue_sha256(state.symbolic_queue)
    # Reporting the same source endpoint twice has no state/queue side effects.
    source.substitute_const(2,h).drop_variable(2).apply_cutoff(1e-10)
    state.endpoint_tm().range_box()
    assert accepted_boundary_sr_queue_sha256(state.symbolic_queue)==queue_hash
    # A second identity segment in the actual normalized input coordinates
    # consumes the first error via right_map + queue, with no fresh time error.
    second_source=reset.extend_domain(Interval(0.,h))
    second_endpoint,second_errors=second_source.substitute_const_with_roundoff(2,h)
    second=FlowpipeSegment(second_source,second_endpoint.drop_variable(2),'validated',h,4,1,
                           endpoint_substitution_roundoff=second_errors)
    second_reset,second_state,stats=_flowstar_normalized_insertion_transition(
        second,state,4,cutoff_threshold=None,generic_accepted_boundary_sr=True,symbolic_queue_max_size=2)
    assert stats['accepted_boundary_sr_composition_branch']=='nonlinear_plus_linear_queue'
    assert all(float(e.lo)==float(e.hi)==0. for e in second_errors)
    # Existing histories must not be put into the new owner a second time.
    assert stats['accepted_boundary_sr_current_owner_width_sum'] < stats['accepted_boundary_sr_total_interval_image_width_sum']/100
    assert second_state.symbolic_queue.reset_count==1 and not second_state.symbolic_queue.J
    for model in second_state.endpoint_tm():
        at_zero=model.polynomial.evaluate_point([0.,0.])
        assert F(float(at_zero))+F(float(model.remainder.lo)) <= exact <= F(float(at_zero))+F(float(model.remainder.hi))
    third_source=second_reset.extend_domain(Interval(0.,h))
    third_endpoint=third_source.substitute_const(2,h).drop_variable(2)
    third=FlowpipeSegment(third_source,third_endpoint,'validated',h,4,1)
    _,third_state,third_stats=_flowstar_normalized_insertion_transition(
        third,second_state,4,cutoff_threshold=None,generic_accepted_boundary_sr=True,symbolic_queue_max_size=2)
    assert third_stats['accepted_boundary_sr_composition_branch']=='full_reanchor'
    assert len(third_state.symbolic_queue.J)==1
    assert accepted_boundary_sr_queue_sha256(state.symbolic_queue)==queue_hash


@pytest.mark.parametrize('plant', ['van_der_pol','brusselator'])
def test_frozen_two_steps_and_same_version_checkpoint_resume(plant,tmp_path):
    config,current,state=setup(plant)
    first=step(plant,current,state,1)
    assert first.status=='validated',first.message
    assert first.endpoint_substitution_roundoff and all(e.is_finite() for e in first.endpoint_substitution_roundoff)
    contract={'plant':plant,'config':config.as_dict(),'endpoint_roundoff_repair':True}
    save_terminal_checkpoint(tmp_path/'before',current=first.reset_tm,normal_state=first.flowstar_normal_state,
                             scheduler={'accepted_steps':1},contract=contract,provenance={'test':'same_version'})
    restored=load_terminal_checkpoint(tmp_path/'before',expected_contract=contract,
                                     expected_order=config.order,expected_dtype='float64')
    second=step(plant,first.reset_tm,first.flowstar_normal_state,2)
    resumed=step(plant,restored.current,restored.normal_state,2)
    assert second.status==resumed.status=='validated'
    for field in ['tm','endpoint_raw_tm','reset_tm']:
        assert tmvector_hashes(getattr(second,field))==tmvector_hashes(getattr(resumed,field))
    assert accepted_boundary_sr_queue_sha256(second.flowstar_normal_state.symbolic_queue)==accepted_boundary_sr_queue_sha256(resumed.flowstar_normal_state.symbolic_queue)
    assert first.flowstar_normal_state.symbolic_queue.owner_boundary_indices==(1,)
    assert second.flowstar_normal_state.symbolic_queue.owner_boundary_indices==(1,2)


@pytest.mark.parametrize('entry', ['published','dense_internal'])
def test_substitution_failure_preserves_accepted_state(entry,monkeypatch,tmp_path):
    config,current,state=setup('van_der_pol')
    first=step('van_der_pol',current,state,1)
    assert first.status=='validated'
    state=first.flowstar_normal_state
    before=accepted_boundary_sr_queue_sha256(state.symbolic_queue)
    before_current=tmvector_hashes(first.reset_tm)
    def fail(*args,**kwargs):
        raise FloatingPointError('injected endpoint enclosure failure')
    target,name=(TaylorModel,'substitute_const_with_roundoff') if entry=='published' else (BatchedTaylorModel,'endpoint')
    monkeypatch.setattr(target,name,fail)
    rejected=step('van_der_pol',first.reset_tm,state,2)
    assert rejected.status=='failed' and rejected.endpoint_raw_tm is None
    assert rejected.reset_tm is None and 'endpoint' in rejected.message
    assert accepted_boundary_sr_queue_sha256(state.symbolic_queue)==before
    assert tmvector_hashes(first.reset_tm)==before_current


@pytest.mark.parametrize('mode,h', [
    ('normalized_insertion_bounded_source_ledger_o4_g1',.005),
    ('normalized_insertion_shared_source_columns_o4_g2',.005),
    ('normalized_insertion_structured_remainder_k16',.005),
    ('normalized_insertion_structured_total_delta_k16',.005),
])
def test_replacement_ledgers_consume_endpoint_additions_in_two_steps(mode,h,monkeypatch):
    import torch_tm_flowpipe.flowpipe as fp
    from torch_tm_flowpipe.g2_shared_column import G2_SHARED_COLUMN_CANDIDATE
    from torch_tm_flowpipe.ode_examples import van_der_pol_ode
    if 'g2' in mode:
        mode=G2_SHARED_COLUMN_CANDIDATE
    actual=fp._endpoint_remainder_decomposition
    consumed=[]
    def observe(seg):
        result=actual(seg)
        prior=seg.validated_remainder_decomposition.ledger.entries['roundoff_safeguard']
        new=result.ledger.entries['roundoff_safeguard']
        for i,error in enumerate(seg.endpoint_substitution_roundoff):
            assert F(float(new[0][0,i])) <= F(float(prior[0][0,i]))+F(float(error.lo))
            assert F(float(new[1][0,i])) >= F(float(prior[1][0,i]))+F(float(error.hi))
        required_lo,required_hi=fp._tmvector_remainder_tensor(seg.endpoint_raw_tm)
        assert torch.all(result.decomposition_lo<=required_lo) and torch.all(result.decomposition_hi>=required_hi)
        consumed.append((seg.endpoint_substitution_roundoff,result))
        return result
    monkeypatch.setattr(fp,'_endpoint_remainder_decomposition',observe)
    current=[Interval(1.1,1.4),Interval(2.35,2.45)]
    state=None
    for index in range(2):
        segment=fp.flowpipe_step_flowstar_style_adaptive(
            van_der_pol_ode,current,h=h,h_min=h,h_max=h,order=4,
            target_remainder_radius=1e-4,cutoff_threshold=1e-10,max_validation_attempts=2,
            validation_mode='flowstar_raw_remainder_compat',reset_mode=mode,
            flowstar_normal_state=state,tm_backend='dense',structured_allow_outward_renormalization=True)
        assert segment.status=='validated',segment.message
        assert len(consumed)==index+1
        current,state=segment.reset_tm,segment.flowstar_normal_state
        if state.bounded_source_ledger_state is not None:
            assert state.bounded_source_ledger_state.accepted_boundary_index==index+1
        if state.g2_shared_column_state is not None:
            assert state.g2_shared_column_state.accepted_boundary_index==index+1
