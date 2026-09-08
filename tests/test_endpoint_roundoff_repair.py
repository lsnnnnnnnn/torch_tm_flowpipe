"""Finite, rationally checked cases for both real substitution paths."""
from fractions import Fraction as F
import math

import pytest
import torch

from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
from torch_tm_flowpipe.batched_dense_tm import BatchedTaylorModel, BatchedPolynomial, sparse_tmvector_to_dense, dense_to_sparse_tmvector
from experiments.endpoint_roundoff_repair.local_oracles import CASES, check_case, assert_model_contains, exact_coefficients


def test_mixed_precision_preserves_actual_h_and_never_narrows_enclosure():
    source = TaylorModel(Polynomial({(1, 0):torch.tensor(-1., dtype=torch.float32),
                                    (1, 1):torch.tensor(100., dtype=torch.float32)}, 2),
                         Interval.zero(), [Interval(-2., .5), Interval(0., .01)], order=4)
    result = source.substitute_const(1, .01).drop_variable(1)
    assert_model_contains(source, result, .01)
    # A binary64 value outside the original domain cannot be made legal by
    # narrowing it to binary32 before checking the domain.
    narrower = [source.domain[0], Interval(0., float(torch.tensor(.01, dtype=torch.float32)))]
    with pytest.raises(ValueError, match='outside'):
        source.polynomial.substitute_const_with_roundoff(1, .01, narrower)
    from torch_tm_flowpipe.endpoint_substitution import enclose_constant_substitution
    coeff = torch.tensor([[[-1., 100.]]], dtype=torch.float32)
    qlo, qhi, elo, ehi = enclose_constant_substitution(
        coeff, [(1, 0), (1, 1)], torch.zeros((1, 1, 1), dtype=torch.float32), [(1,)], 1,
        torch.tensor([.01], dtype=torch.float64), torch.tensor([[-2., 0.]], dtype=torch.float64),
        torch.tensor([[.5, .01]], dtype=torch.float64))
    exact = -1 + 100 * F(.01)
    assert F(float(qlo.item())) <= exact <= F(float(qhi.item()))
    assert F(float(elo.item())) <= -2 * exact and F(float(ehi.item())) >= F(.5) * exact


@pytest.mark.parametrize('case', CASES)
def test_general_substitution_fraction_oracle(case):
    check_case(case)


@pytest.mark.parametrize('cutoff', [None, 1e-10])
@pytest.mark.parametrize('remainder', [(0., 0.), (-1e-6, 2e-6)])
def test_original_witness_and_space_factor(cutoff, remainder):
    for beta in [(0,), (1,)]:
        model = TaylorModel(Polynomial({beta+(0,):-1., beta+(1,):100.},2),Interval(*remainder),
                            [Interval(-3., .25),Interval(0.,.01)],order=4)
        published = model.substitute_const(1,.01).drop_variable(1).apply_cutoff(cutoff)
        dense = sparse_tmvector_to_dense(TMVector([model]),order=4)
        internal = dense_to_sparse_tmvector(dense.endpoint(1,.01).apply_cutoff(cutoff))[0]
        assert_model_contains(model,published,.01)
        assert_model_contains(model,internal,.01)


def test_batch_two_distinct_h_domains_outputs_and_zero_task():
    domain = [Interval(-2.,.5),Interval(-.1,.1)]
    tm = TMVector([TaylorModel(Polynomial({(1,0):-1.,(1,1):100.},2),Interval.zero(),domain,order=4),
                   TaylorModel(Polynomial({(0,0):2.,(0,4):-1e8},2),Interval(-1e-8,2e-8),domain,order=4)])
    one=sparse_tmvector_to_dense(tm,order=4)
    coeff=one.poly.coeffs.repeat(2,1,1)
    coeff[1,1]=0.
    lo=torch.tensor([[-2.,-.1],[.25,-.02]],dtype=torch.float64)
    hi=torch.tensor([[.5,.1],[4.,.02]],dtype=torch.float64)
    batched=BatchedTaylorModel(BatchedPolynomial(coeff,one.poly.basis),one.rem_lo.repeat(2,1),
                               one.rem_hi.repeat(2,1),lo,hi)
    h=torch.tensor([.01,-.02],dtype=torch.float64)
    endpoint=batched.endpoint(1,h)
    correction=endpoint.ledger.entries['endpoint_substitution_roundoff']
    assert float(correction[0][1,1]) == float(correction[1][1,1]) == 0.
    for b in range(2):
        single=BatchedTaylorModel(BatchedPolynomial(coeff[b:b+1],one.poly.basis),batched.rem_lo[b:b+1],
                                   batched.rem_hi[b:b+1],lo[b:b+1],hi[b:b+1])
        single_endpoint=single.endpoint(1,h[b])
        assert torch.equal(endpoint.poly.coeffs[b:b+1],single_endpoint.poly.coeffs)
        assert torch.equal(endpoint.rem_lo[b:b+1],single_endpoint.rem_lo)
        assert torch.equal(endpoint.rem_hi[b:b+1],single_endpoint.rem_hi)
        for source, result in zip(dense_to_sparse_tmvector(single),dense_to_sparse_tmvector(single_endpoint)):
            assert_model_contains(source,result,float(h[b]))


def test_dense_component_reassembly_retains_endpoint_error_ledger():
    source=TMVector([TaylorModel(Polynomial({(0,0):-1.,(0,1):100.},2),Interval.zero(),
                                  [Interval(-2.,.5),Interval(0.,.01)],order=4)]*2)
    endpoint=sparse_tmvector_to_dense(source,order=4).endpoint(1,.01)
    reassembled=BatchedTaylorModel.concat([endpoint.component(0),endpoint.component(1)])
    for category,pair in endpoint.ledger.entries.items():
        assert category in reassembled.ledger.entries
        assert all(torch.equal(a,b) for a,b in zip(pair,reassembled.ledger.entries[category]))
    lo,hi=reassembled.ledger.total(reassembled.rem_lo)
    assert torch.equal(lo,reassembled.rem_lo) and torch.equal(hi,reassembled.rem_hi)
    for original,result in zip(source,dense_to_sparse_tmvector(reassembled)):
        assert_model_contains(original,result,.01)


@pytest.mark.parametrize('h', [0., .5, -.5])
def test_subnormal_and_true_zero(h):
    tiny=math.ulp(0.)
    source=TaylorModel(Polynomial({(0,1):tiny,(1,2):-tiny},2),Interval.zero(),
                       [Interval(-.5,2.),Interval(-1.,1.)],order=4)
    dense=sparse_tmvector_to_dense(TMVector([source]),order=4)
    for result in [source.substitute_const(1,h).drop_variable(1), dense_to_sparse_tmvector(dense.endpoint(1,h))[0]]:
        assert_model_contains(source,result,h)
    if h == 0:
        _,error,_=source.polynomial.substitute_const_with_roundoff(1,h,source.domain)
        assert float(error.lo) == float(error.hi) == 0.


@pytest.mark.parametrize('coefficient,h,domain', [
    (1.,.2,(0.,.1)), (1.,float('nan'),(0.,.1)), (1.,float('inf'),(0.,.1)),
    (float('nan'),.01,(0.,.1)), (float('inf'),.01,(0.,.1)),
    (1e308,2.,(0.,2.)), (1.,1e200,(0.,1e200)),
])
def test_invalid_or_nonfinite_inputs_fail_closed(coefficient,h,domain):
    source=TaylorModel(Polynomial({(0,2):coefficient},2),Interval.zero(),
                       [Interval(-1.,1.),Interval(*domain)],order=4)
    dense=sparse_tmvector_to_dense(TMVector([source]),order=4)
    with pytest.raises((ValueError,FloatingPointError)):
        source.substitute_const(1,h)
    with pytest.raises((ValueError,FloatingPointError)):
        dense.endpoint(1,h)


def test_nonfinite_remainder_rejected_and_inactive_variable_drop_is_exact():
    source=TaylorModel(Polynomial({(1,0):1.},2),Interval(-float('inf'),float('inf')),
                       [Interval(-1.,1.),Interval(0.,1.)],order=4)
    with pytest.raises(FloatingPointError):
        source.substitute_const(1,.5)
    safe=source.with_remainder(Interval.zero()).substitute_const(1,.5)
    assert float(safe.remainder.lo) == float(safe.remainder.hi) == 0.
    assert safe.drop_variable(1).polynomial.terms[(1,)].item() == 1.
    with pytest.raises(ValueError):
        safe.drop_variable(0)


@pytest.mark.parametrize('index',[0,1,2])
def test_time_variable_position_and_severe_same_term_cancellation(index):
    remaining=[Interval(-2.,.5),Interval(.25,1.5)]
    domain=remaining[:index]+[Interval(-1.,1.)]+remaining[index:]
    terms={}
    for power,value in [(0,1e16),(1,1.),(2,-1e16)]:
        beta=(1,0)
        terms[beta[:index]+(power,)+beta[index:]]=value
    source=TaylorModel(Polynomial(terms,3),Interval.zero(),domain,order=4)
    exact=exact_coefficients(source.polynomial,index,1.)
    assert exact[(1,0)]==1
    dense=sparse_tmvector_to_dense(TMVector([source]),order=4)
    for result in [source.substitute_const(index,1.).drop_variable(index),
                   dense_to_sparse_tmvector(dense.endpoint(index,1.))[0]]:
        from experiments.endpoint_roundoff_repair.local_oracles import exact_error_range
        lo,hi=exact_error_range(exact,result.polynomial,result.domain)
        assert F(float(result.remainder.lo))<=lo<=hi<=F(float(result.remainder.hi))
