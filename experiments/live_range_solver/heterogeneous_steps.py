"""Explicit test-only heterogeneous h; main performance parameters never change."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import subprocess

import torch
import torch_tm_flowpipe as core
from torch_tm_flowpipe.live_range_service import LiveRangeService
from experiments.endpoint_roundoff_repair.frozen import setup
from experiments.run_vdp_dense_backend import load_contract
from experiments.boundary_execution.state_equivalence import canonical, digest as state_digest
from experiments.live_range_solver.runner import initial, SerialTask, Trace
from experiments.range_batch_device.common import save,sha,request_from_record,result_from_record
from experiments.range_batch_device.oracle import check


def step_at_h(current,state,h):
    config=setup("van_der_pol")[0]
    ode=core.PolynomialODE.from_system_spec(load_contract()["canonical_system_spec"])
    return core.flowpipe_step_flowstar_style_adaptive(
        ode,current,h=h,h_min=h,h_max=h,order=config.order,
        target_remainder_radius=config.target_remainder_radius,cutoff_threshold=config.cutoff,
        max_validation_attempts=2,validation_eps=config.validation_epsilon,
        validation_mode=config.post_accept_refinement_mode,reset_mode=config.accepted_boundary_sr_mode,
        step_policy_mode="flowstar_compat",flowstar_normal_state=state,
        flowstar_symbolic_queue_max_size=config.accepted_boundary_sr_capacity,
        right_map_center_mode=config.right_map_center_mode,right_map_range_mode=config.right_map_range_mode,
        tm_backend="dense",dense_device="cpu",dense_dtype=torch.float64,
        dense_range_policy=core.DenseRangePolicy(**config.range_policy_mapping),
        dense_observer_mode=core.DENSE_OBSERVER_NONE)


def exercise(backend):
    startup=None
    if backend=="cuda":
        with LiveRangeService("cuda",run_id="heterogeneous-startup-check") as warmup:
            startup=warmup.startup
    settings={"regular":(0,.01),"short":(31,1e-6)}
    reference={}
    for task_id,(index,h) in settings.items():
        task=SerialTask("heterogeneous-reference",task_id,initial("van_der_pol",index),backend,None)
        reference[task_id]=[]
        for _ in range(2):
            current,state=task.begin_attempt()
            with task.execution():
                segment=step_at_h(current,state,h)
            assert segment.status=="validated",segment.message
            reference[task_id].append(canonical(segment))
            task.commit((segment.reset_tm,segment.flowstar_normal_state))
        task.finish()
    trace=Trace()
    records={}
    with LiveRangeService(backend,run_id=f"heterogeneous-h-{backend}",trace=trace) as service:
        tasks={name:service.register(name,initial("van_der_pol",index)) for name,(index,h) in settings.items()}
        def work(name):
            task=tasks[name]
            rows=[]
            for _ in range(2):
                before=state_digest(task.accepted_state)
                current,state=task.begin_attempt(state_digest=before,diagnostic=True)
                with task.execution():
                    segment=step_at_h(current,state,settings[name][1])
                assert segment.status=="validated",segment.message
                assert state_digest((current,state))==before
                task.commit((segment.reset_tm,segment.flowstar_normal_state))
                rows.append(dict(input=canonical((current,state)),segment=canonical(segment),
                                 accepted_state=canonical(task.accepted_state),counter=task.counter))
            task.finish()
            records[name]=rows
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(work,name) for name in settings]
            for future in futures:
                future.result()
    artifact=dict(schema="live-heterogeneous-h-v1",test_inputs_only=True,not_a_performance_sample=True,
        plant="van_der_pol",backend=backend,settings=settings,reference=reference,records=records,
        events=trace.events,groups=service.groups,counts=dict(service.counts),startup=startup,
        affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),script_sha256=sha(__file__),
        core_scientific_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    artifact["verification"]=verify(artifact)
    return artifact


def verify(artifact):
    replay_counts=Counter()
    request_counts={}
    for task,rows in artifact["records"].items():
        for index,row in enumerate(rows):
            assert row["segment"]==artifact["reference"][task][index]
            fields=row["segment"]["fields"]
            assert fields["h"]["float_hex"]==float(artifact["settings"][task][1]).hex()
            assert row["accepted_state"]==[fields["reset_tm"],fields["flowstar_normal_state"]]
            if index:
                assert row["input"]==rows[index-1]["accepted_state"]
            counters=dict(fields["backend_counters"]["items"])
            replay_counts[counters["post_accept_replay_calls"]]+=1
        request_counts[task]=rows[-1]["counter"]
    assert len(replay_counts)>1, "test inputs did not exercise different refinement counts"
    assert len(set(request_counts.values()))>1, "test inputs did not exercise different request counts"
    requests={event["request_id"]:request_from_record(event["request"]) for event in artifact["events"] if event["event"]=="submit"}
    returns=[event for event in artifact["events"] if event["event"]=="return"]
    assert len(requests)==len(returns)
    for event in returns:
        check(requests[event["request_id"]],result_from_record(event["result"]))
    return dict(accepted_steps=4,post_accept_replay_counts={str(k):v for k,v in replay_counts.items()},
                requests_by_task=request_counts,exact_successful_returns=len(returns))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--backend",choices=["cpu","cuda"],required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    artifact=exercise(args.backend)
    path=args.output/"heterogeneous.json.gz"
    with gzip.open(path,"wt") as stream:
        json.dump(artifact,stream,separators=(",",":"),allow_nan=False)
    save(args.output/"summary.json",dict(**artifact["verification"],file=path.name,sha256=sha(path),
         test_inputs_only=True,not_a_performance_sample=True,backend=args.backend))
    print(json.dumps(artifact["verification"],indent=2),flush=True)


if __name__=="__main__":
    main()
