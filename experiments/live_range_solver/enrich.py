"""Additional post-measurement tables, derived only from saved actual runs.

This file is packaging code. It neither advances a solver nor changes the frozen
scientific runner. Time categories are unions of actual host intervals. A worker
step span includes its waits and is not a measurement of CPU execution alone.
"""
from collections import Counter, defaultdict
import argparse
import csv
import gzip
import json
from pathlib import Path

from experiments.range_batch_device.common import read, save, sha
from experiments.live_range_solver.analyze import csv_rows, percentile
from experiments.live_range_solver.verify import load_run


FILES=("time_partition.csv","flush_reasons.csv","grouping_summary.csv",
       "behavior_comparison.csv","BEHAVIOR_SUMMARY.json","ARITHMETIC_AUDIT_SCOPE.json",
       "startup_and_serialization.csv","original_b1_diagnostic.csv",
       "width_summary_by_coordinate.csv","near_zero_widths.csv","relative_wall_time.csv")


def raw_run(folder):
    summary=read(folder/"summary.json")
    assert sha(folder/"run.json.gz")==summary["files"]["run.json.gz"]
    with gzip.open(folder/"run.json.gz","rt") as stream:
        run=json.load(stream)
    return run,summary


def merge(intervals):
    result=[]
    for lo,hi in sorted(intervals):
        assert lo<=hi
        if result and lo<=result[-1][1]:
            result[-1]=(result[-1][0],max(result[-1][1],hi))
        else:
            result.append((lo,hi))
    return result


def duration(intervals):
    return sum(hi-lo for lo,hi in intervals)


def intersect_duration(left,right):
    total=i=j=0
    while i<len(left) and j<len(right):
        alo,ahi=left[i]
        blo,bhi=right[j]
        total+=max(0,min(ahi,bhi)-max(alo,blo))
        if ahi<=bhi:
            i+=1
        else:
            j+=1
    return total


def timeline(run):
    start,end=run["start_ns"],run["end_ns"]
    worker=merge((r["step_start_ns"],r["step_end_ns"]) for rows in run["records"].values() for r in rows if r["accepted"])
    service=merge((g["start_ns"],g["end_ns"]) for g in run["groups"])
    assert all(start<=lo<=hi<=end for lo,hi in worker+service)
    overlap=intersect_duration(worker,service)
    covered=duration(merge(worker+service))
    pieces=[overlap,duration(worker)-overlap,duration(service)-overlap,end-start-covered]
    assert min(pieces)>=0 and sum(pieces)==end-start
    return dict(run_id=run["run_id"],plant=run["plant"],route=run["route"],batch=len(run["ids"]),
        service_inside_worker_span_s=pieces[0]/1e9,worker_span_outside_service_s=pieces[1]/1e9,
        service_outside_worker_span_s=pieces[2]/1e9,outside_both_s=pieces[3]/1e9,
        wall_s=(end-start)/1e9,
        worker_span_scope="union of real step spans, including waits; not CPU execution time",
        service_scope="host evaluator spans including sync/transfers/checks; not pure GPU device time")


def behavior(row,previous_counter):
    counters=dict(row["segment"]["fields"]["backend_counters"]["items"])
    selected={k:v for k,v in counters.items() if k.startswith("post_accept_")}
    return dict(accepted=row["accepted"],status=row["status"],h_hex=row["h_hex"],
                state_step=row["state_step"],step_rejections=row["step_rejections"],
                validation_attempts=row["validation_attempts"],
                requests=row["request_counter"]-previous_counter,post_accept=selected)


def enrich(root,target=None):
    root=Path(root).resolve()
    target=Path(target).resolve() if target else root
    target.mkdir(exist_ok=True,parents=True)
    gate=read(root/"diagnostic/CORRECTNESS_GATE.json")
    assert gate["passed"] is True
    time_rows,flush,startup=[],[],[]
    wall_times={}
    distributions=defaultdict(lambda:dict(sizes=[],waiting=[],requests=0))
    for case in read(root/"formal/CAMPAIGN_PLAN.json")["cases"]:
        run,summary=raw_run(root/"formal"/case["name"])
        wall_times[run["run_id"]]=run["wall_s"]
        time_rows.append(timeline(run))
        reasons=Counter(g["reason"] for g in run["groups"])
        for reason,count in sorted(reasons.items()):
            flush.append(dict(run_id=run["run_id"],plant=run["plant"],route=run["route"],batch=len(run["ids"]),reason=reason,groups=count))
        key=(run["plant"],len(run["ids"]),run["route"])
        values=distributions[key]
        values["sizes"].extend(g["size"] for g in run["groups"])
        values["waiting"].extend(ns/1e9 for ns in run["wait_ns"])
        values["requests"]+=run["counts"].get("submitted",0)
        cold=run.get("cold_startup_outside_timing") or {}
        startup.append(dict(run_id=run["run_id"],plant=run["plant"],route=run["route"],batch=len(run["ids"]),
            cold_cuda_service_s=cold.get("wall_s",0.),compile_load_s=cold.get("build",{}).get("compile_and_load_s",0.),
            timed_service_startup_s=run["startup"].get("wall_s",0.),serialization_outside_timing_s=summary["serialization_s"],
            wall_s=run["wall_s"],rss_scope=run["rss_scope"],gpu_memory_scope=run["gpu_memory_scope"]))
    grouping=[]
    for (plant,batch,route),values in sorted(distributions.items()):
        sizes,waiting=values["sizes"],values["waiting"]
        singleton=sizes.count(1)
        grouping.append(dict(plant=plant,batch=batch,route=route,groups=len(sizes),requests=values["requests"],
            mean_size=sum(sizes)/len(sizes) if sizes else None,p50_size=percentile(sizes,.5),p95_size=percentile(sizes,.95),
            max_size=max(sizes) if sizes else None,singleton_group_fraction=singleton/len(sizes) if sizes else None,
            singleton_request_fraction=singleton/values["requests"] if values["requests"] else None,
            wait_p50_s=percentile(waiting,.5),wait_p95_s=percentile(waiting,.95),wait_max_s=max(waiting) if waiting else None))
    csv_rows(target/"time_partition.csv",time_rows)
    csv_rows(target/"flush_reasons.csv",flush)
    csv_rows(target/"grouping_summary.csv",grouping)
    csv_rows(target/"startup_and_serialization.csv",startup)
    relative=[]
    for plant in ("van_der_pol","brusselator"):
        for batch in (1,8,32):
            legacy=wall_times[f"legacy-b{batch}-{plant}-L"]
            for repetition in range(3):
                s,q,g=[wall_times[f"formal-b{batch}-rep{repetition}-{plant}-{route}"] for route in ("S","Q","G")]
                relative.append(dict(plant=plant,batch=batch,repetition=repetition,
                    G_wall_over_S_wall=g/s,G_wall_over_Q_wall=g/q,G_wall_over_L_wall=g/legacy,
                    G_throughput_over_S_throughput=s/g,G_throughput_over_Q_throughput=q/g,
                    G_throughput_over_L_throughput=legacy/g))
    csv_rows(target/"relative_wall_time.csv",relative)

    comparisons=[]
    original=[]
    for plant in ("van_der_pol","brusselator"):
        for label in ("small-b1","small-b2","prefix-b8","prefix-b32","long-b2","history","original"):
            s,_=raw_run(root/"diagnostic"/f"{label}-{plant}-S")
            g,_=raw_run(root/"diagnostic"/f"{label}-{plant}-G")
            for task,gpu_rows in g["records"].items():
                cpu_rows=s["records"][task]
                assert len(cpu_rows)==len(gpu_rows)
                cpu_counter=gpu_counter=0
                for cpu,gpu in zip(cpu_rows,gpu_rows):
                    a,b=behavior(cpu,cpu_counter),behavior(gpu,gpu_counter)
                    cpu_counter,gpu_counter=cpu["request_counter"],gpu["request_counter"]
                    comparisons.append(dict(plant=plant,case=label,task=task,step=cpu["state_step"],
                        accepted_match=a["accepted"]==b["accepted"],status_match=a["status"]==b["status"],h_match=a["h_hex"]==b["h_hex"],
                        validation_attempts_cpu=a["validation_attempts"],validation_attempts_gpu=b["validation_attempts"],
                        step_rejections_cpu=a["step_rejections"],step_rejections_gpu=b["step_rejections"],
                        requests_cpu=a["requests"],requests_gpu=b["requests"],
                        post_accept_cpu=json.dumps(a["post_accept"],sort_keys=True,separators=(",",":")),
                        post_accept_gpu=json.dumps(b["post_accept"],sort_keys=True,separators=(",",":")),
                        behavior_equal=a==b))
            if label=="original":
                for run in (s,g):
                    original.append(dict(plant=plant,route=run["route"],steps=run["steps"],wall_s=run["wall_s"],
                        requests=run["counts"].get("submitted",0),scope=run["scope"],performance_sample=False,
                        timing_scope="diagnostic includes raw-state capture and observer; not formal latency"))
    csv_rows(target/"behavior_comparison.csv",comparisons)
    different=[r for r in comparisons if not r["behavior_equal"]]
    first=[]
    for plant in ("van_der_pol","brusselator"):
        subset=[r for r in different if r["plant"]==plant]
        first.append(dict(plant=plant,first_observed_difference=min(subset,key=lambda r:(r["step"],r["case"],r["task"])) if subset else None))
    save(target/"BEHAVIOR_SUMMARY.json",dict(compared_steps=len(comparisons),different_steps=len(different),
        first_observed_differences=first,all_predefined_work_completed=True,
        difference_scope="acceptance, actual h, validation/rejection counts, request count, post-accept replay and stopping counters"))
    csv_rows(target/"original_b1_diagnostic.csv",original)
    by_route=defaultdict(Counter)
    for receipt in gate["raw_verifications"]:
        name=receipt["run_id"]
        route=name.rsplit("-",1)[1]
        by_route[route].update(receipt["arithmetic"])
    save(target/"ARITHMETIC_AUDIT_SCOPE.json",dict(
        by_route={k:dict(v) for k,v in sorted(by_route.items())},
        complete_scope="all captured requests: full B32 first two steps, all small and split/order tests, first two of each three-step history window, original B1 first two, B8 first two, long120 steps 1/2/60/100/119/120",
        additional_complete_scopes=["all successful fault/retry/cancel-boundary returns","synthetic correction/table/overflow/subnormal/order probes","heterogeneous actual refinement cases"],
        every_later_request_reaudited=False,formal_independent_fraction_audits=0,
        necessary_cpu_exact_power_correction_inside_formal_timer=True,
        unchanged_local_operator_contract="docs/range_batch_device/OPERATOR_CONTRACT.md"))
    with (root/"cross_backend_widths.csv").open(newline="") as stream:
        bounds=list(csv.DictReader(stream))
    classified=defaultdict(list)
    for row in bounds:
        classified[row["case"],row["plant"],row["view"],row["variable"]].append(row)
    coordinate=[]
    for (label,plant,view,variable),rows in sorted(classified.items()):
        summary=dict(case=label,plant=plant,view=view,variable=variable,entries=len(rows),
            near_zero=sum(r["near_zero"]=="True" for r in rows),warnings=sum(r["warning"]=="True" for r in rows))
        for metric in ("max_abs_diff","center_abs_diff","gpu_cpu_width_ratio"):
            chosen=[r for r in rows if r[metric]]
            numbers=[float(r[metric]) for r in chosen]
            for name,p in (("p50",.5),("p95",.95),("max",1.)):
                summary[metric+"_"+name]=percentile(numbers,p)
            worst=max(chosen,key=lambda r:float(r[metric])) if chosen else None
            for key in ("task","step","time","exact_time"):
                summary[metric+"_worst_"+key]=worst[key] if worst else None
        coordinate.append(summary)
    csv_rows(target/"width_summary_by_coordinate.csv",coordinate)
    csv_rows(target/"near_zero_widths.csv",[r for r in bounds if r["near_zero"]=="True"])
    return dict(files=list(FILES),behavior_differences=len(different))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    print(json.dumps(enrich(args.root,args.output),indent=2))


if __name__=="__main__":
    main()
