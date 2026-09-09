"""Derive correctness gates, width comparisons and performance from raw runs."""
from collections import Counter, defaultdict
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import statistics

from experiments.range_batch_device.common import read, save, sha
from .campaign import matrix
from .verify import load_run, verify_run


def csv_rows(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*p
    lo = int(index)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def expected_run(case, run):
    assert run["plant"] == case["plant"] and run["route"] == case["route"] and run["steps"] == case["steps"]
    from .runner import PARTITION
    assert run["partition_sha256"] == sha(PARTITION)
    expected_ids = case["options"].get("ids", [0,31] if case["batch"]==2 else read(PARTITION)["subsets"][str(case["batch"])] )
    assert run["ids"] == expected_ids
    assert run["original"] == case["options"].get("original", False)
    assert run["checkpoint"] == case["options"].get("checkpoint")
    expected_scope = "RESUMED_LOCAL_WINDOW" if run["checkpoint"] else "ORIGINAL_B1" if run["original"] else "FIXED_PARTITION"
    assert run["scope"] == expected_scope
    assert run["successful_tasks"] == len(expected_ids)
    assert run["successful_lane_steps"] == len(expected_ids)*case["steps"]
    assert run["max_wait_s"] == .020 and run["max_group"] == 32


def compare_state(reference, candidate, steps=None):
    checked = 0
    for task, rows in candidate["records"].items():
        other = reference["records"][task]
        count = min(len(rows), len(other)) if steps is None else steps
        assert len(rows) >= count and len(other) >= count
        for index in range(count):
            assert rows[index]["segment"] == other[index]["segment"], (candidate["run_id"], task, index+1)
            assert rows[index]["before"] == other[index]["before"]
            assert rows[index]["after"] == other[index]["after"]
            checked += 1
    return dict(reference=reference["run_id"], candidate=candidate["run_id"],
                task_ids=list(candidate["records"]), complete_segments_compared=checked)


def widths(reference, candidate, label):
    rows = []
    for task, gpu_steps in candidate["records"].items():
        cpu_steps = reference["records"][task]
        assert len(cpu_steps) == len(gpu_steps)
        exact_time = (cpu_steps[0]["state_step"]-1)*Fraction(float.fromhex(cpu_steps[0]["h_hex"]))
        for cpu, gpu in zip(cpu_steps, gpu_steps):
            assert cpu["h_hex"] == gpu["h_hex"] and cpu["state_step"] == gpu["state_step"]
            exact_time += Fraction(float.fromhex(cpu["h_hex"]))
            for view in ("endpoint", "tube"):
                for variable, a, b in zip(("x", "y"), cpu["bounds"][view], gpu["bounds"][view]):
                    cl,ch = map(float.fromhex,a)
                    gl,gh = map(float.fromhex,b)
                    cw,gw = ch-cl,gh-gl
                    assert cw >= 0 and gw >= 0
                    near_zero = cw <= 1e-10
                    ratio = None if near_zero else gw/cw
                    rows.append(dict(case=label, plant=reference["plant"], task=task, step=cpu["state_step"],
                        time=float(exact_time), exact_time=str(exact_time), view=view, variable=variable,
                        cpu_lo_hex=a[0], cpu_hi_hex=a[1], gpu_lo_hex=b[0], gpu_hi_hex=b[1],
                        cpu_width=cw, gpu_width=gw, lo_abs_diff=abs(gl-cl), hi_abs_diff=abs(gh-ch),
                        max_abs_diff=max(abs(gl-cl),abs(gh-ch)), center_abs_diff=abs((gl+gh)/2-(cl+ch)/2),
                        near_zero=near_zero, gpu_cpu_width_ratio=ratio, warning=ratio is not None and ratio>1.10))
    return rows


def diagnostic_gate(root, *, recompute=True):
    root = Path(root)
    plan = read(root/"CAMPAIGN_PLAN.json")
    assert plan["phase"] == "diagnostic" and plan["cases"] == matrix("diagnostic")
    receipts, lifecycle = [], []
    for case in plan["cases"]:
        run, events = load_run(root/case["name"])
        expected_run(case, run)
        assert run["source_sha"] == plan["source_sha"]
        receipt = verify_run(run, events, recompute=recompute)
        receipts.append(receipt)
        lifecycle.append(dict(run_id=run["run_id"], plant=run["plant"], route=run["route"],
            submitted=run["counts"].get("submitted",0), returned=run["counts"].get("returned",0),
            gpu_completed=run["counts"].get("gpu_completed_requests",0),
            actual_fraction_requests=receipt["arithmetic"].get("raw_requests",0),
            successful_lane_steps=run["successful_lane_steps"], task_epochs=len(run["ids"])))
        save(root/"raw_verification_progress.json", receipts)
        print(json.dumps(receipt), flush=True)
    csv_rows(root.parent/"request_lifecycle_summary.csv", lifecycle)
    equivalence, bounds, resets = [], [], []
    plants = ("van_der_pol", "brusselator")
    def load(label, plant, route):
        return load_run(root/f"{label}-{plant}-{route}")[0]
    for plant in plants:
        for label in ("small-b1", "small-b2", "prefix-b8", "prefix-b32", "long-b2", "history"):
            s,q = load(label, plant, "S"),load(label, plant, "Q")
            equivalence.append(compare_state(s,q))
            g = load(label, plant, "G")
            if label != "long-b2":
                gpu = load(label, plant, "S_gpu")
                equivalence.append(compare_state(g,gpu))
            bounds.extend(widths(s,g,label))
        bounds.extend(widths(load("original",plant,"S"),load("original",plant,"G"),"original"))
        for label in ("long-b2", "history"):
            if label=="long-b2" and plant=="brusselator":
                continue
            reset_index = 100 if plant=="van_der_pol" else 1000
            for route in ("S","Q","G"):
                run = load(label,plant,route)
                for task, rows in run["records"].items():
                    relevant = [r for r in rows if r["state_step"] in {reset_index-1,reset_index,reset_index+1}]
                    assert len(relevant)==3
                    sizes=[]
                    for row in relevant:
                        queue=row["segment"]["fields"]["flowstar_normal_state"]["fields"]["symbolic_queue"]["fields"]
                        sizes.append(len(queue["J"]))
                    assert sizes == [reset_index-1,0,1]
                    resets.append(dict(plant=plant,route=route,case=label,task=task,steps=[r["state_step"] for r in relevant],sizes=sizes))
        for route in ("Q", "G"):
            full = load("prefix-b8",plant,route)
            for size in (4,2,1):
                for offset in range(0,8,size):
                    part = load(f"split-{size}-part{offset//size}",plant,route)
                    equivalence.append(compare_state(full,part,steps=2))
            equivalence.append(compare_state(full,load("permuted-delayed-newest",plant,route),steps=2))
    csv_rows(root.parent/"cross_backend_widths.csv", bounds)
    warnings = [r for r in bounds if r["warning"]]
    csv_rows(root.parent/"width_warnings.csv", warnings)
    width_summary = []
    for plant in plants:
        for view in ("endpoint", "tube"):
            chosen = [r for r in bounds if r["plant"]==plant and r["view"]==view]
            row = dict(plant=plant, view=view, entries=len(chosen), near_zero=sum(r["near_zero"] for r in chosen),
                       warnings=sum(r["warning"] for r in chosen))
            for metric in ("max_abs_diff", "center_abs_diff", "gpu_cpu_width_ratio"):
                values = [r[metric] for r in chosen if r[metric] is not None]
                for name,p in (("p50",.5),("p95",.95),("max",1.)):
                    row[f"{metric}_{name}"] = percentile(values,p)
            worst = max(chosen,key=lambda r:r["max_abs_diff"])
            row.update(worst_case=worst["case"], worst_task=worst["task"], worst_step=worst["step"], worst_time=worst["time"])
            width_summary.append(row)
    csv_rows(root.parent/"width_summary.csv",width_summary)
    save(root.parent/"same_backend_state_equivalence.json", equivalence)
    save(root.parent/"history_reset_checks.json", resets)
    # A gate is issued only after actual checks; incomplete/no-recompute passes
    # produce a report but cannot unlock formal timing.
    gate = dict(passed=bool(recompute), source_sha=plan["source_sha"],
        full_runs=len(receipts), raw_verifications=receipts, same_backend_comparisons=equivalence,
        width_rows=len(bounds), width_warnings=len(warnings), full_fraction_scope="B32 first two, long120 preregistered steps, all small/split/history captured requests",
        all_later_requests_fraction_reaudited=False, full_1000_gpu_runs=0, entire_solver_formally_proved=False)
    save(root/"CORRECTNESS_GATE.json",gate)
    return gate


def metrics(run):
    groups = run["groups"]
    sizes = [g["size"] for g in groups]
    requests = run["counts"].get("submitted",0)
    waiting = [ns/1e9 for ns in run["wait_ns"]]
    counts = run["counts"]
    return dict(run_id=run["run_id"], plant=run["plant"], route=run["route"], batch=len(run["ids"]),
        steps=run["steps"], successful_tasks=run["successful_tasks"], accepted_lane_steps=run["accepted_lane_steps"],
        successful_lane_steps=run["successful_lane_steps"], requests=requests,
        gpu_completed_requests=counts.get("gpu_completed_requests",0),
        cpu_table_fallback=counts.get("external_table_fallback_requests",0),
        cpu_hardware_fallback=counts.get("hardware_fallback_requests",0),
        groups=len(groups), mean_group_size=statistics.mean(sizes) if sizes else None,
        median_group_size=statistics.median(sizes) if sizes else None, max_group_size=max(sizes) if sizes else None,
        singleton_request_fraction=sum(v==1 for v in sizes)/requests if requests else None,
        wait_p50_s=percentile(waiting,.5), wait_p95_s=percentile(waiting,.95), wait_max_s=max(waiting) if waiting else None,
        wall_s=run["wall_s"], cpu_s=run["cpu_s"], throughput=run["throughput"],
        peak_rss_kib=run["peak_rss_kib"], peak_gpu_allocated_bytes=run["peak_gpu_allocated_bytes"],
        evaluated_range_service_s=sum((g["end_ns"]-g["start_ns"])/1e9 for g in groups),
        evaluated_range_thread_cpu_s=sum(g.get("thread_cpu_ns",0)/1e9 for g in groups),
        gpu_h2d_s=sum(g["timing"].get("h2d_and_structure_s",0.) for g in groups),
        gpu_kernel_sync_s=sum(g["timing"].get("kernel_and_sync_s",0.) for g in groups),
        gpu_d2h_checks_s=sum(g["timing"].get("d2h_and_checks_s",0.) for g in groups),
        cold_cuda_startup_s=(run.get("cold_startup_outside_timing") or {}).get("wall_s",0.),
        measured_online=run["execution"]=="ONLINE_LIVE_SOLVE")


def formal_summary(root):
    root = Path(root)
    plan = read(root/"CAMPAIGN_PLAN.json")
    assert plan["phase"] == "formal" and plan["cases"] == matrix("formal")
    rows, grouping = [], []
    for case in plan["cases"]:
        run, events = load_run(root/case["name"])
        expected_run(case, run)
        assert not run["diagnostic"] and run["source_sha"] == plan["source_sha"]
        verify_run(run,events)
        row = metrics(run)
        rows.append(row)
        histogram = Counter(g["size"] for g in run["groups"])
        for size,count in sorted(histogram.items()):
            grouping.append(dict(run_id=run["run_id"],plant=run["plant"],route=run["route"],batch=len(run["ids"]),
                                 group_size=size,groups=count,requests=size*count))
    csv_rows(root.parent/"timings_raw.csv", rows)
    csv_rows(root.parent/"actual_grouping.csv", grouping)
    csv_rows(root.parent/"resource_usage.csv", [{k:r[k] for k in ("run_id","cpu_s","wall_s","peak_rss_kib","peak_gpu_allocated_bytes")} for r in rows])
    summaries, paired, amdahl = [], [], []
    by_name = {r["run_id"]:r for r in rows}
    for plant in ("van_der_pol", "brusselator"):
        for batch in (1,8,32):
            for route in ("S","Q","G","L"):
                chosen = [r for r in rows if r["plant"]==plant and r["batch"]==batch and r["route"]==route]
                summaries.append(dict(plant=plant,batch=batch,route=route,samples=len(chosen),
                    median_wall_s=statistics.median(r["wall_s"] for r in chosen),
                    min_wall_s=min(r["wall_s"] for r in chosen),max_wall_s=max(r["wall_s"] for r in chosen),
                    median_throughput=statistics.median(r["throughput"] for r in chosen)))
            legacy = by_name[f"legacy-b{batch}-{plant}-L"]
            for repetition in range(3):
                s,q,g = [by_name[f"formal-b{batch}-rep{repetition}-{plant}-{route}"] for route in ("S","Q","G")]
                paired.append(dict(plant=plant,batch=batch,repetition=repetition,
                    S_over_G=s["wall_s"]/g["wall_s"],Q_over_G=q["wall_s"]/g["wall_s"],
                    min_SQ_over_G=min(s["wall_s"],q["wall_s"])/g["wall_s"],L_over_G=legacy["wall_s"]/g["wall_s"],
                    Q_over_S=q["wall_s"]/s["wall_s"]))
                fraction = s["evaluated_range_service_s"]/s["wall_s"]
                amdahl.append(dict(plant=plant,batch=batch,repetition=repetition,
                    measured_S_evaluator_share=fraction,zero_evaluator_upper_reference=1/(1-fraction),
                    measured_G_service_s=g["evaluated_range_service_s"],
                    replace_only_evaluator_reference=s["wall_s"]/(s["wall_s"]-s["evaluated_range_service_s"]+g["evaluated_range_service_s"]),
                    scope="measured opted-in request evaluator; excludes caller prepacking and remaining dense CPU ranges; service wall overlaps workers"))
    csv_rows(root.parent/"end_to_end_summary.csv", summaries)
    csv_rows(root.parent/"paired_speedups.csv", paired)
    csv_rows(root.parent/"amdahl_reference.csv", amdahl)
    decisions = {}
    for plant in ("van_der_pol","brusselator"):
        chosen = [r for r in paired if r["plant"]==plant and r["batch"]==32]
        speeds = [r["min_SQ_over_G"] for r in chosen]
        decisions[plant] = dict(paired_speedups=speeds,median=statistics.median(speeds),wins=sum(v>1 for v in speeds),
            practical=statistics.median(speeds)>=1.10 and sum(v>1 for v in speeds)>=2,
            marginal=statistics.median(speeds)>=1 and sum(v>1 for v in speeds)>=2)
    if all(d["practical"] for d in decisions.values()):
        state = "LIVE_RANGE_SCHEDULER_VALIDATED__END_TO_END_PREFIX_SPEEDUP"
    elif all(d["marginal"] for d in decisions.values()):
        state = "LIVE_RANGE_SCHEDULER_VALIDATED__MARGINAL_PREFIX_GAIN"
    else:
        state = "LIVE_RANGE_SCHEDULER_VALIDATED__NO_END_TO_END_GAIN"
    result = dict(status=state, decisions=decisions, formal_samples=len(rows), measured_throughput_online=True,
        new_full_long_horizon_gpu=False, entire_solver_formally_proved=False, scalar_legacy_unconditionally_safe=False,
        default_enabled=False, numerical_scope="existing ordered range operator only", source_sha=plan["source_sha"])
    save(root.parent/"PERFORMANCE_RESULT.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["diagnostic","formal"])
    parser.add_argument("path",type=Path)
    parser.add_argument("--no-recompute",action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = diagnostic_gate(args.path,recompute=not args.no_recompute) if args.phase=="diagnostic" else formal_summary(args.path)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
