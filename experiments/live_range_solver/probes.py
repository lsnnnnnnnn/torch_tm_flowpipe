"""Explicit state-machine inputs: changed supports, power repair and CPU fallbacks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import json
from pathlib import Path

import torch

from torch_tm_flowpipe import Interval
from torch_tm_flowpipe.live_range_service import LiveRangeService
from torch_tm_flowpipe.range_requests import RangeRequest, evaluate_range_requests, structure_key
from experiments.range_batch_device.common import (
    request_from_record,result_from_record,result_record,digest,save,sha,
)
from experiments.range_batch_device.oracle import check
from .runner import Trace


def inputs():
    c=torch.ones((1,1),dtype=torch.float64)
    x=torch.tensor([float.fromhex("0x1.7d3ecfa658d9bp+9")],dtype=torch.float64)
    cube=RangeRequest("cube",((3,),),c,c.clone(),x,x.clone())
    tiny=replace(cube,request_id="tiny",domain_lo=torch.tensor([5e-324],dtype=torch.float64),
                 domain_hi=torch.tensor([1e-160],dtype=torch.float64))
    coeff=torch.tensor([[1e30,-1e30,1.,-0.]],dtype=torch.float64)
    unit=torch.ones(1,dtype=torch.float64)
    ordered=RangeRequest("ordered",((0,),(1,),(2,),(3,)),coeff,coeff.clone(),unit,unit.clone())
    reverse=replace(ordered,request_id="reverse",exponents=ordered.exponents[::-1],
                    coefficients_lo=coeff.flip(-1),coefficients_hi=coeff.flip(-1))
    table=RangeRequest("table",((2,1),),2*c,2*c,torch.tensor([0.,-1.],dtype=torch.float64),
        torch.tensor([.25,1.],dtype=torch.float64),"normal",(1,),0,{2:Interval(0.,.125)})
    overflow=replace(cube,request_id="overflow",domain_lo=torch.tensor([1e308],dtype=torch.float64),
                     domain_hi=torch.tensor([1e308],dtype=torch.float64))
    return [cube,tiny,ordered,reverse,table,overflow]


def exercise(backend, *, hardware_fallback=False):
    trace=Trace()
    failed=False
    def evaluator(rows,**kwargs):
        nonlocal failed
        if hardware_fallback and not failed:
            failed=True
            raise RuntimeError("explicit live probe hardware fault")
        return evaluate_range_requests(rows,**kwargs)
    with LiveRangeService(backend,trace=trace,evaluator=evaluator,hardware_fallback=hardware_fallback,
                          run_id=f"probes-{backend}-{hardware_fallback}") as service:
        requests=inputs()
        tasks={r.request_id:service.register(r.request_id,r.coefficients_lo) for r in requests}
        def work(request):
            task=tasks[request.request_id]
            for iteration in range(2 if request.request_id=="ordered" else 1):
                task.begin_attempt(diagnostic=True)
                if iteration:
                    # This second request is generated only from the actual
                    # consumed first result, with a different support and count.
                    value=task.accepted_state[0].reshape(1,1)
                    request=replace(request,exponents=((1,),),coefficients_lo=value,coefficients_hi=value.clone())
                result=task.evaluate(request)
                if not result.ok:
                    task.reject(failed=True)
                    return
                task.commit((result.lo,result.hi))
            task.finish()
        with ThreadPoolExecutor(len(tasks)) as pool:
            jobs=[pool.submit(work,r) for r in requests]
            for job in jobs:
                job.result(timeout=20)
    return dict(schema="live-range-explicit-probes-v1",state_machine_test_inputs_only=True,backend=backend,
                hardware_fault=hardware_fallback,events=trace.events,groups=service.groups,counts=dict(service.counts),
                statuses={k:t.status for k,t in tasks.items()})


def verify_probe(artifact):
    submitted=[e for e in artifact["events"] if e["event"]=="submit"]
    requests={e["request_id"]:request_from_record(e["request"]) for e in submitted}
    outputs={e["request_id"]:e["result"] for e in artifact["events"] if e["event"]=="return"}
    assert len(requests)==7 and set(outputs)==set(requests)
    ordered=[e for e in submitted if e["task"]=="ordered"]
    assert len(ordered)==2 and ordered[1]["previous"]==ordered[0]["request_id"]
    assert ordered[1]["generation"]==1 and ordered[1]["attempt"]==2
    assert ordered[1]["request"]["coefficients_lo"]["values"]==outputs[ordered[0]["request_id"]]["lo"]["values"]
    checked=0
    for group in artifact["groups"]:
        rows=[requests[rid] for rid in group["request_ids"]]
        assert len({digest(structure_key(r)) for r in rows})==1
        backend="cpu" if group["hardware_fallback"] else artifact["backend"]
        expected=evaluate_range_requests(rows,backend=backend,diagnostics=True)
        for request in rows:
            result=result_from_record(outputs[request.request_id])
            assert result_record(expected[request.request_id])==outputs[request.request_id]
            if result.ok:
                external=request.step_powers is not None
                check(request,result,require_terms=not external,require_powers=not external)
                checked+=1
    assert checked==6 and artifact["statuses"]["overflow"]=="FAILED"
    assert sum(status=="FINISHED" for status in artifact["statuses"].values())==5
    assert artifact["counts"]["external_table_fallback_requests"]==1
    if artifact["hardware_fault"]:
        assert artifact["counts"]["hardware_fallback_requests"]>0
    cube=next(output for rid,output in outputs.items() if "/4:cube/" in rid)
    if artifact["backend"]=="cpu":
        assert cube["status"]=="corrected"
    return dict(backend=artifact["backend"],hardware_fault=artifact["hardware_fault"],requests=7,exact_successes=checked)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summaries=[]
    for backend,fault in (("cpu",False),("cuda",False),("cuda",True)):
        artifact=exercise(backend,hardware_fallback=fault)
        receipt=verify_probe(artifact)
        file=args.output/f"{backend}-{fault}.json.gz"
        with gzip.open(file,"wt") as stream:
            json.dump(artifact,stream,separators=(",",":"),allow_nan=False)
        summaries.append(dict(**receipt,file=file.name,sha256=sha(file)))
    save(args.output/"probe_checks.json",summaries)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=="__main__":
    main()
