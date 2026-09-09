from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from itertools import product
import math

import pytest
import torch

from torch_tm_flowpipe import Interval, Polynomial
from torch_tm_flowpipe.packed_boundary_range import make_plan, packed_boundary_execution, _REQUEST_DISPATCH
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from torch_tm_flowpipe.range_requests import (
    RangeRequest, certified_scalar_power, compute_cpu_group, evaluate_range_requests,
    prepare_range_requests, range_request_execution, structure_key,
)
from experiments.range_batch_device.oracle import check


def requests(batch=32, n=3, order=6, kind="standard", outputs=2):
    support = tuple(e for e in product(range(order+1), repeat=n) if sum(e) <= order)
    rows = []
    for b in range(batch):
        coefficients = torch.tensor([[(-1.)**(i+o+b) * (i+1)*(b+1)/(256*(o+1))
            for i in range(len(support))] for o in range(outputs)], dtype=torch.float64)
        coefficients[:,0] = -0. if b%2 else 0.
        dl = torch.tensor([-.5 + (b%4)/32 + v/64 for v in range(n)], dtype=torch.float64)
        dh = torch.tensor([.625 + (b%8)/64 + v/64 for v in range(n)], dtype=torch.float64)
        states, t = (), None
        if kind == "normal":
            t = b % n
            states = tuple(v for v in range(n) if v != t)
            dl[list(states)] = -1; dh[list(states)] = 1
        hi = coefficients + 1/1024 if kind == "interval-coefficient" else coefficients.clone()
        rows.append(RangeRequest(str(b), support, coefficients, hi, dl, dh, kind, states, t))
    return rows


def bits(result):
    return tuple(tuple(float(x).hex() for x in value.flatten()) for value in (result.lo, result.hi))


@pytest.mark.parametrize("batch", [1,2,8,32])
@pytest.mark.parametrize("n,order", [(1,4),(2,6),(3,4),(3,6)])
@pytest.mark.parametrize("kind", ["standard","normal","interval-coefficient"])
def test_grouped_bitwise_and_exact(batch,n,order,kind):
    rows = requests(batch,n,order,kind)
    results = evaluate_range_requests(rows, diagnostics=True)
    for r in rows:
        got = results[r.request_id]
        check(r,got)
        assert got.status == "ok"
        ref = make_plan(r.exponents,n,r.state_variables if kind=="normal" else None,r.time_variable).evaluate(
            r.coefficients_lo[None],r.coefficients_hi[None],r.domain_lo[None],r.domain_hi[None])
        assert all(torch.equal(x.view(torch.int64), y[0].view(torch.int64)) for x,y in zip((got.lo,got.hi),ref))


@pytest.mark.parametrize("kind", ["standard","normal","interval-coefficient"])
def test_chunking_and_permutation(kind):
    rows=requests(kind=kind)
    whole=evaluate_range_requests(rows)
    for size in (1,2,8):
        parts={}
        for i in range(0,32,size):
            parts.update(evaluate_range_requests(rows[i:i+size]))
        assert {i:bits(v) for i,v in parts.items()}=={i:bits(v) for i,v in whole.items()}
    assert {i:bits(v) for i,v in evaluate_range_requests(rows[::-1]).items()}=={i:bits(v) for i,v in whole.items()}


def test_order_zeros_empty_and_cancellation():
    r=requests(1,1,4,outputs=1)[0]
    c=torch.tensor([[1e30,-1e30,1.,0.,-0.]],dtype=torch.float64)
    r=replace(r,coefficients_lo=c,coefficients_hi=c.clone(),domain_lo=torch.ones(1,dtype=torch.float64),domain_hi=torch.ones(1,dtype=torch.float64))
    reverse=replace(r,request_id="reversed",exponents=r.exponents[::-1],coefficients_lo=c.flip(-1),coefficients_hi=c.flip(-1))
    empty=replace(r,request_id="empty",exponents=(),coefficients_lo=c[:,:0],coefficients_hi=c[:,:0])
    groups,_,_=prepare_range_requests([r,reverse,empty])
    assert len(groups)==3
    results=evaluate_range_requests([r,reverse,empty],diagnostics=True)
    for row in (r,reverse,empty): check(row,results[row.request_id])
    assert bits(results[r.request_id]) != bits(results[reverse.request_id])
    assert bits(results["empty"]) == (("0x0.0p+0",),("0x0.0p+0",))


def test_power_counterexample_repaired_without_changing_legacy():
    x=float.fromhex("0x1.7d3ecfa658d9bp+9")
    old=Interval(x).pow_int(3)
    exact=Fraction(x)**3
    assert not Fraction(float(old.lo))<=exact<=Fraction(float(old.hi))
    a,b,changed=certified_scalar_power(x,x,3)
    assert changed and Fraction(a)<=exact<=Fraction(b)
    r=RangeRequest("pow3",((3,),),torch.ones((1,1),dtype=torch.float64),torch.ones((1,1),dtype=torch.float64),
        torch.tensor([x],dtype=torch.float64),torch.tensor([x],dtype=torch.float64))
    got=evaluate_range_requests([r],diagnostics=True)[r.request_id]
    assert got.status=="corrected"
    check(r,got)
    assert float(Interval(x).pow_int(3).lo).hex()==float(old.lo).hex()


@pytest.mark.parametrize("x", [0.,-0.,5e-324,-5e-324,1e-160,-1e-160])
@pytest.mark.parametrize("p", [0,1,2,3,4,6])
def test_subnormal_and_power_edges(x,p):
    c=torch.tensor([[1.]],dtype=torch.float64)
    r=RangeRequest("tiny",((p,),),c,c.clone(),torch.tensor([x],dtype=torch.float64),torch.tensor([x],dtype=torch.float64))
    result=evaluate_range_requests([r],diagnostics=True)["tiny"]
    check(r,result)


def test_failure_mask_cancel_and_overflow_do_not_poison_group():
    rows=requests(2,1,4,outputs=1)
    r=rows[0]
    bad=replace(r,request_id="nan",coefficients_lo=torch.full_like(r.coefficients_lo,math.nan))
    reverse=replace(r,request_id="bad-domain",domain_lo=torch.tensor([2.],dtype=torch.float64))
    overflow=replace(r,request_id="overflow",domain_lo=torch.tensor([1e308],dtype=torch.float64),domain_hi=torch.tensor([1e308],dtype=torch.float64))
    mixed=rows+[bad,reverse,overflow,replace(bad,request_id="masked",enabled=False),replace(bad,request_id="cancelled",cancelled=True)]
    got=evaluate_range_requests(mixed)
    assert {k:v.status for k,v in got.items() if k not in {"0","1"}}=={
        "nan":"invalid","bad-domain":"invalid","overflow":"overflow","masked":"masked","cancelled":"cancelled"}
    clean=evaluate_range_requests(rows)
    assert all(bits(got[r.request_id])==bits(clean[r.request_id]) for r in rows)


def test_output_ownership_next_call_and_threads():
    rows=requests(2,2,4)
    baseline=evaluate_range_requests(rows)
    before=bits(baseline["0"])
    def work(backend):
        with prepared_remainder_replay(True),packed_boundary_execution(True),range_request_execution(backend):
            p=Polynomial({(1,):2.},1)
            assert p.evaluate_interval([Interval(-.25,.5)]).is_finite()
            return {i:bits(v) for i,v in evaluate_range_requests(rows).items()}
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(work,"cpu") for _ in range(2)]
        assert futures[0].result()==futures[1].result()
    assert _REQUEST_DISPATCH.get() is None
    rows[0].coefficients_lo.add_(2); rows[0].coefficients_hi.add_(2)
    assert bits(baseline["0"])==before
    assert bits(evaluate_range_requests(rows)["0"])!=before
    other=bits(baseline["1"])
    baseline["0"].lo.add_(99)
    assert bits(baseline["1"])==other


def test_external_table_uses_original_fallback_and_is_not_ignored():
    c=torch.tensor([[2.]],dtype=torch.float64)
    r=RangeRequest("table",((2,1),),c,c.clone(),torch.tensor([0.,-1.],dtype=torch.float64),
        torch.tensor([.25,1.],dtype=torch.float64),"normal",(1,),0,{2:Interval(0.,.125)})
    result=evaluate_range_requests([r],diagnostics=True)["table"]
    assert result.status=="fallback"
    check(r,result,require_terms=False,require_powers=False)
    original=evaluate_range_requests([replace(r,step_powers=None)])["table"]
    assert result.hi>original.hi
    assert evaluate_range_requests([replace(r,step_powers={2:Interval(0.,.01)})])["table"].status=="invalid"


def test_nonnormal_domain_rejected_and_role_keys_separate():
    r=requests(1,3,4,"normal")[0]
    dl=r.domain_lo.clone();dl[1]=-2
    assert evaluate_range_requests([replace(r,domain_lo=dl)])[r.request_id].status=="invalid"
    other=replace(r,request_id="other",state_variables=(0,2),time_variable=1)
    assert structure_key(other)!=structure_key(r)
    with pytest.raises(ValueError,match="unique"):
        evaluate_range_requests([r,r])


def test_group_computation_is_shared(monkeypatch):
    import torch_tm_flowpipe.range_requests as module
    seen=[]
    original=module.compute_cpu_group
    def compute(g,**kw):
        seen.append(g.coefficients_lo.shape[0])
        return original(g,**kw)
    monkeypatch.setattr(module,"compute_cpu_group",compute)
    got=evaluate_range_requests(requests(32,2,4))
    assert len(got)==32 and seen==[32]
