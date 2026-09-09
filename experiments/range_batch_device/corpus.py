"""Capture distinct tasks at dependency barriers, and saved-state offline requests."""
import argparse
from contextlib import contextmanager
from fractions import Fraction as Q
import functools
import gzip
import inspect
import json
import math
import sys
import time

import torch
import torch_tm_flowpipe as core
import torch_tm_flowpipe.polynomial as polynomial
import torch_tm_flowpipe.accepted_boundary_sr as sr
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from torch_tm_flowpipe.range_requests import RangeRequest
from experiments.endpoint_roundoff_repair.frozen import setup,step
from experiments.boundary_execution.state_equivalence import digest as state_digest
from experiments.boundary_execution.profile import state_identity
from .common import ROOT,RUN,PARENT,save,read,sha,head,numerical_sources,request_record,digest


def partition_plan():
    plants={}
    for plant in ("van_der_pol","brusselator"):
        config,_,_=setup(plant)
        original=[tuple(map(Q,box)) for box in config.initial_decimal_box]
        axes=[]
        for (a,b),n in zip(original,(8,4)):
            edges=[a+(b-a)*i/n for i in range(n+1)]
            axis=[]
            for l,h in zip(edges,edges[1:]):
                fl,fh=float(l),float(h)
                if Q(fl)>l:fl=math.nextafter(fl,-math.inf)
                if Q(fh)<h:fh=math.nextafter(fh,math.inf)
                assert Q(fl)<=l<=h<=Q(fh)
                axis.append(dict(exact=[str(l),str(h)],outward_hex=[fl.hex(),fh.hex()]))
            assert Q(float.fromhex(axis[0]["outward_hex"][0]))<=a
            assert Q(float.fromhex(axis[-1]["outward_hex"][1]))>=b
            assert all(float.fromhex(axis[i]["outward_hex"][1])>=float.fromhex(axis[i+1]["outward_hex"][0]) for i in range(n-1))
            axes.append(axis)
        boxes=[dict(task=i*4+j,axis_indices=[i,j],exact=[axes[0][i]["exact"],axes[1][j]["exact"]],
                    outward_hex=[axes[0][i]["outward_hex"],axes[1][j]["outward_hex"]]) for i in range(8) for j in range(4)]
        plants[plant]=dict(config=config.as_dict(),original_exact=[[str(a),str(b)] for a,b in original],axes=axes,tasks=boxes)
    return dict(schema="range-distinct-tasks-v1",recorded_before_capture_utc=time.time(),
        purpose="CONCURRENT_TASK_REQUEST_CAPTURE_AND_REPLAY",grid=[8,4],steps_per_task=2,
        subsets={"1":[0],"8":list(range(0,32,4)),"32":list(range(32))},plants=plants,
        scheduler="Barrier n consists of at most the nth range request from each independent task in the same nominal step. Per-task predecessors are never advanced across a barrier. No integrated scheduler implemented.",
        benchmark_repetitions=5,benchmark_order="alternating forward/reverse",timing_partition_tuning=False)


@contextmanager
def capture_ranges(records,metadata):
    originals=[]
    def bind(owner,name,kind):
        original=getattr(owner,name)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            obj,domain=args[:2]
            domain=list(domain)
            args=(obj,domain,*args[2:])
            state_variables=();time_variable=None;table=None
            if kind=="normal":
                table=args[2] if len(args)>2 else kwargs.get("step_exp_table")
                time_variable=args[4] if len(args)>4 else kwargs.get("time_var_index",0)
                state_vars=args[3] if len(args)>3 else kwargs.get("state_var_indices")
                state_variables=tuple(sorted((set(range(obj.n_vars)) if state_vars is None else set(state_vars))-{time_variable}))
                if table is not None:
                    table={int(p):iv if isinstance(iv,core.Interval) else core.Interval(iv) for p,iv in
                        (table.items() if hasattr(table,"items") else enumerate(table))}
            exponents=tuple(sorted(obj)) if kind=="interval-coefficient" else tuple(obj.terms)
            if kind=="interval-coefficient":
                cl=torch.stack([obj[e].lo for e in exponents]) if exponents else torch.empty(0,dtype=torch.float64)
                ch=torch.stack([obj[e].hi for e in exponents]) if exponents else torch.empty(0,dtype=torch.float64)
            else:
                cl=torch.stack(list(obj.terms.values())) if exponents else torch.empty(0,dtype=torch.float64)
                ch=cl.clone()
            dl=torch.stack([d.lo for d in domain]) if domain else torch.empty(0,dtype=torch.float64)
            dh=torch.stack([d.hi for d in domain]) if domain else torch.empty(0,dtype=torch.float64)
            ordinal=len(records)
            request=RangeRequest(f"{metadata['case']}/s{metadata['step']}/r{ordinal:04d}",exponents,cl[None],ch[None],dl,dh,
                kind,state_variables,time_variable,table)
            frame=inspect.currentframe().f_back
            stack=[]
            while frame:
                filename=frame.f_code.co_filename
                if "/torch_tm_flowpipe/" in filename:
                    stack.append([filename.split("/src/")[-1],frame.f_code.co_name,frame.f_lineno])
                frame=frame.f_back
            started=time.perf_counter()
            result=original(*args,**kwargs)
            elapsed=time.perf_counter()-started
            records.append(dict(**metadata,ordinal=ordinal,call_position=stack,request=request_record(request),
                scalar_cpu_bounds=[float(result.lo).hex(),float(result.hi).hex()],
                observed_original_range_s=elapsed,request_numeric_sha256=digest(request_record(request))))
            return result
        targets=[(owner,name)]
        if not isinstance(owner,type):
            for modname,module in list(sys.modules.items()):
                if module is None or not modname.startswith("torch_tm_flowpipe"):continue
                for alias,value in list(vars(module).items()):
                    if value is original and (module,alias) not in targets: targets.append((module,alias))
        for target,alias in targets:
            originals.append((target,alias,getattr(target,alias)));setattr(target,alias,wrapped)
    bind(polynomial.Polynomial,"evaluate_interval","standard")
    bind(polynomial,"evaluate_interval_normal","normal")
    bind(sr,"_interval_polynomial_range","interval-coefficient")
    try:yield
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)


def capture_step(plant,current,state,metadata):
    before=state_digest((current,state))
    metadata=dict(metadata,plant=plant,step=state.step_index+1,source_sha=head(),source_state_sha256=before,
                  source_state=state_identity(current,state),prepared_remainder_replay=True,packed_boundary_execution=True)
    records=[]
    with prepared_remainder_replay(True),packed_boundary_execution(True),capture_ranges(records,metadata):
        observed=step(plant,current,state,state.step_index+1)
    assert observed.status=="validated",observed.message
    assert before==state_digest((current,state))
    # Separate unobserved pass supplies the real same-work step denominator.
    with prepared_remainder_replay(True),packed_boundary_execution(True):
        start=time.perf_counter();result=step(plant,current,state,state.step_index+1);elapsed=time.perf_counter()-start
    assert state_digest(result)==state_digest(observed)
    event=dict(**metadata,requests=len(records),step_wall_s=elapsed,after_state_sha256=state_digest((result.reset_tm,result.flowstar_normal_state)),
        full_segment_sha256=state_digest(result),h_hex=float(result.h).hex(),
        range_observed_s=sum(r["observed_original_range_s"] for r in records),
        timing_scope="unobserved whole step; capture and count traversal separate")
    return result,records,event


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plan-only",action="store_true");args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    path=RUN/"PARTITION_PLAN.json"
    if not path.exists():save(path,partition_plan())
    if args.plan_only:return
    plan=read(path)
    raw=RUN/"raw/corpus";raw.mkdir(parents=True,exist_ok=False)
    events=[];sources={};files=[]
    with gzip.open(raw/"independent.jsonl.gz","wt") as out:
        for plant,p in plan["plants"].items():
            config,_,_=setup(plant)
            for box in p["tasks"]:
                exact=[tuple(map(Q,b)) for b in box["exact"]]
                state=core.FlowstarNormalFlowpipeState.from_exact_decimal_box(exact,config.order)
                current=state.normalized_initial_tm(config.order)
                for _ in range(2):
                    result,records,event=capture_step(plant,current,state,dict(case=f"{plant}/task{box['task']:02d}",task=box["task"],
                        scope="CONCURRENT_TASK_REQUEST_CAPTURE_AND_REPLAY",partition_plan_sha256=sha(path)))
                    for record in records:out.write(json.dumps(record,separators=(",",":"))+"\n")
                    events.append(event);current,state=result.reset_tm,result.flowstar_normal_state
                print(json.dumps(dict(plant=plant,task=box["task"],steps=2,requests=sum(e["requests"] for e in events[-2:]))),flush=True)
    checkpoints=[PARENT/"raw_minimal/state_inputs"/case for case in (
        "brusselator_before0002","brusselator_before0101","brusselator_before0995","brusselator_before0999",
        "brusselator_before1000","brusselator_before1001","van_der_pol_before0002","van_der_pol_before0099",
        "van_der_pol_before0100","van_der_pol_before0101")]
    checkpoints += [PARENT/"raw_minimal/formal/vdp_full_candidate"/name for name in ("checkpoint_0120","checkpoint_0994")]
    with gzip.open(raw/"offline.jsonl.gz","wt") as out:
        for index,checkpoint in enumerate(checkpoints):
            loaded=core.load_terminal_checkpoint(checkpoint)
            relative=str(checkpoint.relative_to(ROOT))
            sources[relative]={p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}
            result,records,event=capture_step(loaded.contract["plant"],loaded.current,loaded.normal_state,
                dict(case=f"offline{index:02d}",task=None,scope="OFFLINE_REQUEST_REPLAY",checkpoint=relative))
            for record in records:out.write(json.dumps(record,separators=(",",":"))+"\n")
            events.append(event)
            print(json.dumps(dict(checkpoint=relative,requests=len(records))),flush=True)
    save(raw/"events.json",events)
    save(RUN/"REQUEST_CORPUS_MANIFEST.json",dict(schema="range-request-corpus-v1",source_sha=head(),
        numerical_sources=numerical_sources(),partition_plan_sha256=sha(path),checkpoint_sources=sources,
        files={str(p.relative_to(RUN)):sha(p) for p in sorted(raw.iterdir()) if p.is_file()},
        independent_tasks=64,steps_per_independent_task=2,offline_states=len(checkpoints),
        requests=sum(e["requests"] for e in events),
        scope=["OFFLINE_REQUEST_REPLAY","CONCURRENT_TASK_REQUEST_CAPTURE_AND_REPLAY"],
        integrated_flowpipe_throughput=False,legacy_full_horizon="REUSED"))


if __name__=="__main__":main()
