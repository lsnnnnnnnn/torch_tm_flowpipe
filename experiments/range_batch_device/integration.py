"""Real complete-state consumers, reset windows, rollback and bounded GPU offload."""
from contextlib import contextmanager,nullcontext
from fractions import Fraction as Q
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch_tm_flowpipe as core
import torch_tm_flowpipe.range_requests as batch
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution,_REQUEST_DISPATCH
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from experiments.endpoint_roundoff_repair.frozen import setup,step
from experiments.boundary_execution.state_equivalence import digest,canonical
from experiments.boundary_execution.profile import state_identity
from .common import ROOT,RUN,PARENT,save,read


@contextmanager
def execution(backend):
    calls=[]
    with prepared_remainder_replay(True),packed_boundary_execution(True),(
            batch.range_request_execution(backend) if backend else nullcontext()):
        fn=_REQUEST_DISPATCH.get()
        def record(*args):
            result=fn(*args)
            calls.append([float(result.lo).hex(),float(result.hi).hex()])
            return result
        token=_REQUEST_DISPATCH.set(record if fn else None)
        try:yield calls
        finally:_REQUEST_DISPATCH.reset(token)


def summarize(segment,backend,before,calls):
    assert segment.status=="validated",segment.message
    current,state=segment.reset_tm,segment.flowstar_normal_state
    assert current is not None and state is not None
    # This includes endpoint substitution E and all queue/owner payloads.
    full=digest(segment)
    with execution(backend):
        endpoint=[list(iv.to_tuple()) for iv in segment.endpoint_raw_tm.range_box()]
        tube=[list(iv.to_tuple()) for iv in segment.tm.range_box()]
    assert digest(segment)==full,"observer changed endpoint error or carry"
    return dict(step=state.step_index,before=before,after=state_identity(current,state),
        full_segment_sha256=full,complete_next_input_sha256=digest((current,state)),
        endpoint_bounds_hex=[[float(a).hex(),float(b).hex()] for a,b in endpoint],
        tube_bounds_hex=[[float(a).hex(),float(b).hex()] for a,b in tube],
        actual_range_dispatch_calls=len(calls),range_values_sha256=digest(calls),status=segment.status,
        repeated_observer_preserved=True,h_hex=float(segment.h).hex())


def run_cpu_sequence(plant,current,state,steps,source):
    original=digest((current,state));lanes=[(current,state),(current,state)]
    records=[];first=[]
    for offset in range(steps):
        pair=[]
        for lane,backend in enumerate((None,"cpu")):
            c,s=lanes[lane];before=digest((c,s));identity=state_identity(c,s)
            with execution(backend) as calls:segment=step(plant,c,s,s.step_index+1)
            assert digest((c,s))==before
            pair.append(summarize(segment,backend,identity,calls))
            lanes[lane]=segment.reset_tm,segment.flowstar_normal_state
            if offset==0:first.append(segment)
        assert pair[0]["full_segment_sha256"]==pair[1]["full_segment_sha256"],"CPU adapter changed a real full state"
        assert pair[1]["actual_range_dispatch_calls"]>0
        records.append(pair)
    assert digest((current,state))==original
    rollback=[];resume=[]
    for lane,backend in enumerate((None,"cpu")):
        segment=first[lane];c,s=segment.reset_tm,segment.flowstar_normal_state
        before=digest((c,s));assert s.symbolic_queue.J
        def fail(*args,**kwargs):raise FloatingPointError("injected local range/endpoint failure")
        if backend:
            def failed_requests(requests,**kwargs):
                return {r.request_id:batch.RangeResult(r.request_id,"overflow",message="injected local range failure") for r in requests}
            context=patch.object(batch,"evaluate_range_requests",failed_requests)
        else:context=patch.object(core.TaylorModel,"substitute_const_with_roundoff",fail)
        with execution(backend),context:failed=step(plant,c,s,s.step_index+1)
        assert failed.status=="failed" and failed.reset_tm is None and digest((c,s))==before
        rollback.append(dict(backend=backend or "legacy",before=before,after=digest((c,s)),status=failed.status))
        with tempfile.TemporaryDirectory(prefix="range-batch-resume-") as temp:
            checkpoint=Path(temp)/"state"
            core.save_terminal_checkpoint(checkpoint,current=c,normal_state=s,
                scheduler=dict(accepted_steps=s.step_index,time_exact=str(s.step_index*Q(segment.h))),
                contract=dict(plant=plant),provenance=dict(purpose="local batch resume integration"))
            loaded=core.load_terminal_checkpoint(checkpoint)
            with execution(backend):again=step(plant,loaded.current,loaded.normal_state,s.step_index+1)
            assert digest(again)==records[1][lane]["full_segment_sha256"]
            resume.append(dict(backend=backend or "legacy",step=s.step_index+1,full_segment_sha256=digest(again)))
    return dict(plant=plant,source=source,records=records,rollback=rollback,resume=resume,
        both_switches_enabled=True,reconstructed_from_published_box=False)


def initial_task(plant,task):
    plan=read(RUN/"PARTITION_PLAN.json")
    box=plan["plants"][plant]["tasks"][task]
    config,_,_=setup(plant)
    state=core.FlowstarNormalFlowpipeState.from_exact_decimal_box([tuple(map(Q,b)) for b in box["exact"]],config.order)
    return state.normalized_initial_tm(config.order),state


def verify_cpu():
    rows=[]
    for plant,name in (("van_der_pol","van_der_pol_before0099"),("brusselator","brusselator_before0999")):
        checkpoint=PARENT/"raw_minimal/state_inputs"/name
        loaded=core.load_terminal_checkpoint(checkpoint)
        rows.append(run_cpu_sequence(plant,loaded.current,loaded.normal_state,3,str(checkpoint.relative_to(ROOT))))
    for plant in ("van_der_pol","brusselator"):
        for task in (0,31):
            current,state=initial_task(plant,task)
            rows.append(run_cpu_sequence(plant,current,state,2,dict(task=task,partition="PARTITION_PLAN.json")))
    return rows


def verify_gpu():
    rows=[]
    for plant in ("van_der_pol","brusselator"):
        _,current,state=setup(plant)
        sequence=[]
        for _ in range(20):
            before=digest((current,state));identity=state_identity(current,state)
            with execution("cuda") as calls:segment=step(plant,current,state,state.step_index+1)
            assert digest((current,state))==before
            record=summarize(segment,"cuda",identity,calls)
            assert record["actual_range_dispatch_calls"]>0
            if sequence:assert before==sequence[-1]["complete_next_input_sha256"]
            sequence.append(record)
            current,state=segment.reset_tm,segment.flowstar_normal_state
        rows.append(dict(plant=plant,steps=20,first_two_steps_checked=True,records=sequence,
            purpose="optional B1 range offload integration diagnostic",full_gpu_solver=False,new_full_horizon=False))
    return rows


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cpu=verify_cpu();save(RUN/"raw/integration_cpu.json",cpu)
    print("CPU complete-state integration passed",flush=True)
    gpu=verify_gpu();save(RUN/"raw/integration_gpu.json",gpu)
    save(RUN/"integration_checks.json",dict(cpu=cpu,gpu=gpu,cpu_full_state_bitwise_equal=True,
        endpoint_error_history_failure_paths_connected=True,integrated_multi_task_scheduler=False,
        published_box_used_as_next_state=False,old_full_runs="REUSED",new_full_1000_step_runs=0,
        measured_solver_speedup_claimed=False))
    print("CUDA B1 offload: two systems, 20 steps each passed",flush=True)


if __name__=="__main__":main()
