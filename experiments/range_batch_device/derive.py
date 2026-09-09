"""Derived measurements and decisions, recomputed from the raw timing events."""
from collections import Counter,defaultdict
from statistics import median
import csv
import math
from .common import RUN,read,read_records,digest


MODES=("S_scalar_full","B_cpu_full","CPU_tensor_compute","GPU_resident","GPU_full_roundtrip")


def check_time_row(row):
    mode=row["mode"]
    assert mode in MODES and row["batch"] in (1,8,32) and row["block"] in range(1,6)
    order=MODES if row["block"]%2 else MODES[::-1]
    assert order[row["position"]]==mode
    assert row["finished_ns"]>row["started_ns"]>0
    assert row["seconds"]==(row["finished_ns"]-row["started_ns"])/1e9
    assert row["seconds"]>0 and math.isfinite(row["seconds"])
    assert row["valid_requests"]==row["requests"]==sum(row["group_sizes"])
    assert row["warmup"] is False and row["profiler"] is False and row["synchronized"] is True
    assert row["scope"]=="CONCURRENT_TASK_REQUEST_CAPTURE_AND_REPLAY"
    assert row["includes_sparse_packing"]==(mode in {"S_scalar_full","B_cpu_full","GPU_full_roundtrip"})
    assert row["includes_roundtrip"]==(mode=="GPU_full_roundtrip")
    assert row["actual_kernel_invocations"]==(4*len(row["group_sizes"]) if mode in {"GPU_resident","GPU_full_roundtrip"} else 0)
    if mode=="GPU_resident":
        assert row["device_receipts"]==[[1,1,1,1]]*len(row["group_sizes"])
    else:assert row["device_receipts"]==[]


def check_invocations(record):
    events=record["raw_device_events"]
    assert events and record["mock"] is False and record["cpu_invocations"]==0
    assert all(e["receipt"]==[1,1,1,1] and e["rows"]>0 for e in events)
    assert record["audit_device_executed_kernel_invocations"]==sum(sum(e["receipt"]) for e in events)>0


def timing_summary(root=RUN):
    samples=read(root/"raw/timing/samples.json")
    expected={(p,b,i,m) for p in ("van_der_pol","brusselator") for b in (1,8,32) for i in range(1,6) for m in MODES}
    actual={(r["plant"],r["batch"],r["block"],r["mode"]) for r in samples}
    assert actual==expected and len(samples)==len(expected)
    groups=defaultdict(list)
    for row in samples:
        check_time_row(row);groups[row["plant"],row["batch"],row["mode"]].append(row)
    summary=[]
    for (plant,batch,mode),rows in sorted(groups.items()):
        assert len({r["workload_sha256"] for r in rows})==1 and len({r["output_sha256"] for r in rows})==1
        assert len({r["requests"] for r in rows})==1
        seconds=[r["seconds"] for r in rows];sizes=rows[0]["group_sizes"]
        summary.append(dict(plant=plant,batch=batch,mode=mode,requests=rows[0]["requests"],groups=len(sizes),
            median_s=median(seconds),min_s=min(seconds),max_s=max(seconds),requests_per_second=rows[0]["requests"]/median(seconds),
            effective_mean_group_size=sum(sizes)/len(sizes),singleton_request_fraction=sum(s==1 for s in sizes)/sum(sizes),
            median_group_size=median(sizes),max_group_size=max(sizes),peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rows)))
    return samples,summary


def decision(root=RUN):
    samples,summary=timing_summary(root)
    table={(r["plant"],r["batch"],r["mode"]):r for r in summary}
    gates={}
    for plant in ("van_der_pol","brusselator"):
        def ratio(a,b):
            return [next(r["seconds"] for r in samples if (r["plant"],r["batch"],r["block"],r["mode"])==(plant,32,i,a))/
                    next(r["seconds"] for r in samples if (r["plant"],r["batch"],r["block"],r["mode"])==(plant,32,i,b)) for i in range(1,6)]
        resident=ratio("CPU_tensor_compute","GPU_resident")
        full=ratio("B_cpu_full","GPU_full_roundtrip")
        gates[plant]=dict(resident_ratios=resident,full_roundtrip_ratios=full,
            resident_useful=min(resident)>=2.,offload_useful=min(full)>=1.2,
            singleton_request_fraction=table[plant,32,"B_cpu_full"]["singleton_request_fraction"])
    if all(g["offload_useful"] for g in gates.values()):status="RANGE_BATCH_CPU_CLOSED__CUDA_LOCAL_PILOT_USEFUL"
    elif all(g["resident_useful"] for g in gates.values()):status="RANGE_BATCH_CPU_CLOSED__CUDA_RESIDENCY_REQUIRED"
    elif any(g["singleton_request_fraction"]>.5 for g in gates.values()):status="RANGE_BATCH_CORRECT__REAL_GROUPS_TOO_FRAGMENTED"
    else:status="RANGE_BATCH_CORRECT__NO_MATERIAL_DEVICE_GAIN"
    return dict(status=status,correctness=dict(cpu_real_requests_bitwise_preserved=True,
        cpu_legacy_pow_counterexample_fixed_only_in_new_api=True,cuda_exact_operator_containment=True,whole_solver_formal_proof=False),
        local_throughput=dict(scope="CONCURRENT_TASK_REQUEST_CAPTURE_AND_REPLAY",gates=gates,
            policy="Both plants must meet each threshold in all five paired blocks; resident compares prepacked grouped CPU."),
        solver_integration=dict(cpu_complete_state_and_reset_windows=True,cuda_B1_range_only_steps_per_plant=20,
            integrated_multi_task_solver=False,measured_solver_speedup=False,new_full_1000_step_runs=0,old_full_runs="REUSED"))


def projected(root=RUN):
    _,summary=timing_summary(root)
    table={(r["plant"],r["batch"],r["mode"]):r for r in summary}
    events=read(root/"raw/corpus/events.json");plan=read(root/"PARTITION_PLAN.json")
    rows=[]
    for plant in ("van_der_pol","brusselator"):
        for batch in (1,8,32):
            subset=plan["subsets"][str(batch)]
            selected=[e for e in events if e["plant"]==plant and e.get("task") in subset and e["scope"]=="CONCURRENT_TASK_REQUEST_CAPTURE_AND_REPLAY"]
            total=sum(e["step_wall_s"] for e in selected)
            observed=sum(e["range_observed_s"] for e in selected)
            assert len(selected)==batch*2 and 0<observed<total
            cpu=table[plant,batch,"B_cpu_full"]["median_s"];gpu=table[plant,batch,"GPU_full_roundtrip"]["median_s"]
            cc=table[plant,batch,"CPU_tensor_compute"]["median_s"];gr=table[plant,batch,"GPU_resident"]["median_s"]
            rows.append(dict(plant=plant,batch=batch,label="PROJECTED",requests=sum(e["requests"] for e in selected),
                baseline_aggregate_step_s=total,observed_range_s=observed,observed_range_fraction=observed/total,
                remaining_work_s=total-observed,cpu_compute_s=cc,cpu_group_pack_scatter_overhead_s=cpu-cc,
                resident_compute_s=gr,gpu_group_pack_transfer_scatter_overhead_s=gpu-gr,
                projected_cpu_aggregate_work_speedup=total/(total-observed+cpu),
                projected_gpu_aggregate_work_speedup=total/(total-observed+gpu),
                ideal_range_elimination_upper_bound=total/(total-observed),
                assumption="serial non-range work plus collected independent range barriers; observed range share is approximate; no integrated scheduler measurement"))
    return rows


def csv_write(path,rows):
    with path.open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
