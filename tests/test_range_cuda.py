from dataclasses import replace
from fractions import Fraction as Q
import math

import pytest
import torch

from torch_tm_flowpipe.range_requests import RangeRequest, evaluate_range_requests, prepare_range_requests
from torch_tm_flowpipe.range_cuda import get_module, prepare_cuda_group, primitive_probe, run_resident
from experiments.range_batch_device.oracle import check
from test_range_requests import requests,bits


pytestmark=pytest.mark.cuda


@pytest.mark.parametrize("batch",[1,2,8,32])
@pytest.mark.parametrize("n,order,kind",[(1,4,"standard"),(2,6,"interval-coefficient"),(3,4,"normal"),(3,6,"standard")])
def test_device_exact_powers_terms_sums_and_execution(batch,n,order,kind):
    rows=requests(batch,n,order,kind)
    timing={}
    output=evaluate_range_requests(rows,backend="cuda",diagnostics=True,timings=timing)
    for r in rows: check(r,output[r.request_id])
    assert timing["actual_kernel_invocations"] >= 4
    assert all(s>0 for s in timing["group_sizes"])
    assert timing["h2d_copy_operations"] == 8 * len(timing["group_sizes"])
    assert timing["d2h_copy_operations"] == 8 * len(timing["group_sizes"])
    assert timing["device_allocation_operations"] == 16 * len(timing["group_sizes"])
    assert timing["metadata_h2d_bytes"] > 0 and timing["numeric_h2d_bytes"] > 0
    assert get_module().build["nvrtc_version"]==[12,1]


@pytest.mark.parametrize("kind",["standard","normal","interval-coefficient"])
def test_device_reordering_and_chunking(kind):
    rows=requests(kind=kind)
    full={i:bits(v) for i,v in evaluate_range_requests(rows,backend="cuda").items()}
    for size in (1,2,8,32):
        result={}
        shuffled=rows[::2]+rows[1::2]
        for i in range(0,32,size): result.update(evaluate_range_requests(shuffled[i:i+size],backend="cuda"))
        assert {i:bits(v) for i,v in result.items()}==full


def test_directed_primitives_and_no_ftz():
    a=[5e-324,-5e-324,1e-160,-1e-160,1.,1e30,1e308]
    b=[.5,.5,1e-160,1e-160,5e-324,-1e30,2.]
    out=primitive_probe(a,b)
    for i,(x,y) in enumerate(zip(a,b)):
        for index,exact in [(0,Q(x)+Q(y)),(2,Q(x)*Q(y))]:
            lo,hi=map(float,out[i,index:index+2])
            assert (lo==-math.inf or Q(lo)<=exact) and (hi==math.inf or exact<=Q(hi))
    assert float(out[0,3])==5e-324 and float(out[0,2])==0.
    assert torch.isinf(out[-1,3])


@pytest.mark.parametrize("x",[0.,-0.,5e-324,-5e-324,1e-160,-1e-160,float.fromhex("0x1.7d3ecfa658d9bp+9")])
def test_device_tiny_powers_signed_zero_and_cpu_counterexample(x):
    rows=[]
    for p in (0,1,2,3,4,6):
        c=torch.tensor([[1.]],dtype=torch.float64)
        rows.append(RangeRequest(str(p),((p,),),c,c.clone(),torch.tensor([x],dtype=torch.float64),torch.tensor([x],dtype=torch.float64)))
    result=evaluate_range_requests(rows,backend="cuda",diagnostics=True)
    for r in rows: check(r,result[r.request_id])


def test_resident_mask_mutation_invalid_and_overflow_isolation():
    rows=requests(8,2,4)
    groups,_,_=prepare_range_requests(rows)
    resident=prepare_cuda_group(groups[0])
    baseline=run_resident(resident)
    base=[x.cpu() for x in baseline]
    resident.mask[1]=0
    resident.coefficients_lo[1]=math.nan
    resident.coefficients_lo[2]=math.nan
    resident.domain_lo[3]=1e308;resident.domain_hi[3]=1e308
    resident.domain_lo[4]=2.;resident.domain_hi[4]=1.
    output=[x.cpu() for x in run_resident(resident)]
    assert output[4].tolist()==[0,2,1,3,1,0,0,0]
    assert output[-1].tolist()==[1,1,1,1]
    for b in (0,5,6,7):
        assert all(torch.equal(output[i][b].view(torch.int64),base[i][b].view(torch.int64)) for i in (0,1))
    assert torch.equal(rows[1].coefficients_lo,groups[0].coefficients_lo[1])
    saved=base[0].clone()
    resident.coefficients_lo[0].add_(2);resident.coefficients_hi[0].add_(2)
    again=run_resident(resident)[0].cpu()
    assert torch.equal(base[0],saved) and not torch.equal(again[0],saved[0])


def test_device_empty_support_and_severe_cancellation():
    r=requests(1,1,4,outputs=1)[0]
    c=torch.tensor([[1e30,-1e30,1.,0.,-0.]],dtype=torch.float64)
    r=replace(r,coefficients_lo=c,coefficients_hi=c.clone(),domain_lo=torch.ones(1,dtype=torch.float64),domain_hi=torch.ones(1,dtype=torch.float64))
    empty=replace(r,request_id="empty",exponents=(),coefficients_lo=c[:,:0],coefficients_hi=c[:,:0])
    out=evaluate_range_requests([r,empty],backend="cuda",diagnostics=True)
    for row in (r,empty): check(row,out[row.request_id])
