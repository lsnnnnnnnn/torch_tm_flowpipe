"""Five alternating, unprofiled blocks including the actual sparse caller costs."""
from collections import Counter
import json
import os
import time

import torch
from torch_tm_flowpipe.range_requests import evaluate_range_requests,prepare_range_requests,compute_cpu_group
from torch_tm_flowpipe.range_cuda import get_module,prepare_cuda_group,run_resident
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from .common import RUN,save,read,read_records,digest,head,numerical_sources
from .workloads import stages,request_identity


MODES=("S_scalar_full","B_cpu_full","CPU_tensor_compute","GPU_resident","GPU_full_roundtrip")


def packed_outputs(groups,outputs,backend):
    values={};receipts=[]
    for group,output in zip(groups,outputs):
        if backend=="cuda":
            lo,hi,_,_,status,_,_,receipt=output
            lo,hi,status,receipt=(x.cpu() for x in (lo,hi,status,receipt))
            assert not bool(status.any());assert receipt.tolist()==[1,1,1,1]
            receipts.append(receipt.tolist())
        else:
            lo,hi,_,_,active,_,_=output
            assert bool(active.all())
        for b,r in enumerate(group.requests):values[r.request_id]=[float(lo[b,0]).hex(),float(hi[b,0]).hex()]
    return values,receipts


def measure_workload(workload,plant,batch):
    all_inputs=[x for _,rows in workload for x in rows]
    ids=[x.record["request"]["request_id"] for x in all_inputs]
    id_set=set(ids)
    assert len(ids)==len(set(ids))
    prepared=[prepare_range_requests([x.request() for x in rows]) for _,rows in workload]
    assert all(not fallback and not early for _,fallback,early in prepared)
    groups=[g for entries,_,_ in prepared for g in entries]
    start=time.perf_counter()
    resident=[prepare_cuda_group(g) for g in groups]
    torch.cuda.synchronize()
    resident_setup=time.perf_counter()-start
    expected={x.record["request"]["request_id"]:x.record["scalar_cpu_bounds"] for x in all_inputs}
    expected_outputs={}
    for backend in ("cpu","cuda"):
        raw={r["request_id"]:[r["result"]["lo"]["values"][0],r["result"]["hi"]["values"][0]]
             for r in read_records(RUN/f"raw/operator_outputs/{backend}.jsonl.gz") if r["request_id"] in id_set}
        assert set(raw)==set(ids);expected_outputs[backend]=raw
    samples=[];reuse=0
    def run(mode):
        nonlocal reuse
        if mode=="S_scalar_full":return [x.scalar() for x in all_inputs]
        if mode in {"B_cpu_full","GPU_full_roundtrip"}:
            result={}
            for _,rows in workload:
                result.update(evaluate_range_requests([x.request() for x in rows],backend="cpu" if mode=="B_cpu_full" else "cuda"))
            return result
        if mode=="CPU_tensor_compute":return [compute_cpu_group(g) for g in groups]
        reuse+=1
        return [run_resident(g) for g in resident]
    def validate(mode,output):
        receipts=[]
        if mode=="S_scalar_full":
            values={i:[float(v.lo).hex(),float(v.hi).hex()] for i,v in zip(ids,output)}
            assert values==expected
        elif mode in {"B_cpu_full","GPU_full_roundtrip"}:
            assert all(r.ok for r in output.values())
            values={i:[float(v.lo[0]).hex(),float(v.hi[0]).hex()] for i,v in output.items()}
            assert values==expected_outputs["cpu" if mode=="B_cpu_full" else "cuda"]
        else:
            values,receipts=packed_outputs(groups,output,"cuda" if mode=="GPU_resident" else "cpu")
            assert values==expected_outputs["cuda" if mode=="GPU_resident" else "cpu"]
        return values,receipts
    for mode in MODES:
        output=run(mode);torch.cuda.synchronize();validate(mode,output);del output
    for block in range(1,6):
        for position,mode in enumerate(MODES if block%2 else MODES[::-1]):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter_ns()
            output=run(mode)
            if mode=="GPU_resident":torch.cuda.synchronize()
            finished=time.perf_counter_ns()
            peak=torch.cuda.max_memory_allocated()
            values,receipts=validate(mode,output)
            del output
            row=dict(plant=plant,batch=batch,block=block,position=position,mode=mode,started_ns=started,finished_ns=finished,
                seconds=(finished-started)/1e9,requests=len(ids),valid_requests=len(values),terms=sum(len(x.record["request"]["exponents"]) for x in all_inputs),
                workload_sha256=request_identity(workload),output_sha256=digest(values),group_sizes=[len(g.requests) for g in groups],
                actual_kernel_invocations=sum(map(sum,receipts)) if mode=="GPU_resident" else 4*len(groups) if mode=="GPU_full_roundtrip" else 0,
                device_receipts=receipts,peak_allocated_bytes=peak,warmup=False,profiler=False,
                scope="CONCURRENT_TASK_REQUEST_CAPTURE_AND_REPLAY",includes_sparse_packing=mode in {"S_scalar_full","B_cpu_full","GPU_full_roundtrip"},
                includes_roundtrip=mode=="GPU_full_roundtrip",synchronized=True)
            samples.append(row);print(json.dumps({k:row[k] for k in ("plant","batch","block","mode","seconds","requests")}),flush=True)
    costs=[]
    # A separate instrumented traversal; never a performance denominator.
    for backend in ("cpu","cuda"):
        for key,rows in workload:
            start=time.perf_counter();requests=[x.request() for x in rows];packing=time.perf_counter()-start
            trace={};result=evaluate_range_requests(requests,backend=backend,timings=trace)
            assert all(v.ok for v in result.values())
            costs.append(dict(plant=plant,batch=batch,step=key[0],ordinal=key[1],backend=backend,
                sparse_input_packing_s=packing,**trace,scope="SEPARATE_INSTRUMENTED_TRAVERSAL"))
    residency=dict(plant=plant,batch=batch,preparation_h2d_and_structure_s=resident_setup,
        explicit_resident_reuses=reuse,requests=len(ids),groups=len(groups),
        input_bytes=sum(sum(x.numel()*x.element_size() for x in (r.coefficients_lo,r.coefficients_hi,r.domain_lo,r.domain_hi,r.keys,r.ops,r.states,r.mask)) for r in resident))
    return samples,costs,residency


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert sorted(os.sched_getaffinity(0))==[2]
    from .test_accounting import account
    account(include_evidence=False)  # all identities pass; any initial environment error is retained
    assert read(RUN/"cpu_batch_equivalence.json")["all_requests_finite"]
    assert read(RUN/"cuda_actual_invocations.json")["audit_device_executed_kernel_invocations"]>0
    raw=RUN/"raw/timing";raw.mkdir(parents=True,exist_ok=False)
    module=get_module()
    save(raw/"environment.json",dict(source_sha=head(),numerical_sources=numerical_sources(),python=os.sys.version,
        executable=os.sys.executable,torch=torch.__version__,torch_cuda=torch.version.cuda,affinity=sorted(os.sched_getaffinity(0)),
        threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),build=module.build,
        baseline_switches=dict(prepared_remainder_replay=True,packed_boundary_execution=True),
        official_scope="local requests only; no integrated solver throughput",started_utc=time.time()))
    records=list(read_records(RUN/"raw/corpus/independent.jsonl.gz"))
    samples=[];costs=[];residency=[]
    with prepared_remainder_replay(True),packed_boundary_execution(True):
        for plant in ("van_der_pol","brusselator"):
            for batch in (1,8,32):
                s,c,r=measure_workload(stages(records,plant,batch),plant,batch)
                samples+=s;costs+=c;residency.append(r)
                save(raw/"samples.json",samples);save(raw/"costs.json",costs);save(raw/"residency.json",residency)
    save(raw/"COMPLETE.json",dict(samples=len(samples),source_sha=head(),finished_utc=time.time()))


if __name__=="__main__":main()
