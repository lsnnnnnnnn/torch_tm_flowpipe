"""Recompute source, requests, exact containment, timing and decision evidence."""
import argparse
import ast
from collections import Counter,defaultdict
import csv
from fractions import Fraction as Q
import gzip
import json
from pathlib import Path
import subprocess

import torch
import torch_tm_flowpipe as core
from torch_tm_flowpipe.range_requests import prepare_range_requests
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution,is_enabled
from experiments.endpoint_roundoff_repair.frozen import step
from experiments.boundary_execution.state_equivalence import digest as state_digest
from .common import ROOT,RUN,PARENT,read,sha,digest,read_records,request_from_record,result_from_record,numerical_sources
from .oracle import check
from .derive import timing_summary,decision,projected,check_invocations
from .test_accounting import account


NUMERICAL_SHA="440714afc3acc250903c5ec98fb0011905671944"
BASE="ec6abec2bed30f0b7b52acb6b5b56723cea6918e"


def write_manifest(root):
    (root/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(root)}\n" for p in sorted(root.rglob("*"))
        if p.is_file() and p.name!="SHA256SUMS"))


def manifest(root):
    declared={}
    for line in (root/"SHA256SUMS").read_text().splitlines():
        checksum,name=line.split("  ",1);relative=Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts and name not in declared
        path=root/relative
        assert path.is_file() and not path.is_symlink() and sha(path)==checksum,name
        declared[name]=checksum
    assert set(declared)=={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name!="SHA256SUMS"}
    return len(declared)


def check_source(root):
    source=read(root/"SOURCE_MAP.json")
    assert source["numerical_sha"]==NUMERICAL_SHA and source["base_sha"]==BASE
    assert source["numerical_sources"]==numerical_sources()
    for path,checksum in source["numerical_sources"].items():
        assert digest_bytes(subprocess.check_output(["git","-C",str(ROOT),"show",f"{NUMERICAL_SHA}:{path}"]))==checksum
    for path,checksum in source["evidence_code_sources"].items():
        assert sha(ROOT/path)==checksum
        assert digest_bytes(subprocess.check_output(["git","-C",str(ROOT),"show",f"{source['evidence_code_sha']}:{path}"]))==checksum
    for path,checksum in source["new_test_sources"].items():
        assert sha(ROOT/path)==checksum
        assert digest_bytes(subprocess.check_output(["git","-C",str(ROOT),"show",f"{source['test_source_sha']}:{path}"]))==checksum
    assert not is_enabled()
    assert sha(root/"GOAL_FROZEN.md")==source["goal_sha256"]
    # The original scalar arithmetic and the entire solver stay byte-identical.
    for path,checksum in source["numerical_sources"].items():
        if Path(path).name in {"range_requests.py","range_cuda.py","range_cuda_kernel.cu","packed_boundary_range.py"}:continue
        assert digest_bytes(subprocess.check_output(["git","-C",str(ROOT),"show",f"{BASE}:{path}"]))==checksum
    old=ast.parse(subprocess.check_output(["git","-C",str(ROOT),"show",f"{BASE}:src/torch_tm_flowpipe/packed_boundary_range.py"]))
    new=ast.parse((ROOT/"src/torch_tm_flowpipe/packed_boundary_range.py").read_text())
    for name in ("RangePlan","make_plan","_multiply","_check","packed_boundary_execution"):
        assert ast.dump(next(n for n in old.body if getattr(n,"name",None)==name))==ast.dump(next(n for n in new.body if getattr(n,"name",None)==name))
    assert source["all_baselines"]==dict(prepared_remainder_replay=True,packed_boundary_execution=True)
    assert source["old_full_runs"]=="REUSED" and source["new_full_horizon_claimed"] is False
    assert source["report_sha256"]==sha(ROOT/"docs/range_batch_device/REPORT_PLAIN_CHINESE.md")
    parent=read(root/"parent_checks/verify.log")
    assert parent["passed"] and parent["package_checkout_sha"]==BASE and parent["scientific_sha"]=="5f37cbe0427c480ef0ebbbcaba292143bc8f4ede"
    assert int((root/"parent_checks/verify.exit").read_text())==0 and int((root/"parent_checks/local.exit").read_text())==0
    return source


def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def check_record(record,output):
    request=record["request"]
    assert request["request_id"]==f"{record['case']}/s{record['step']}/r{record['ordinal']:04d}","request ID/position mismatch"
    assert request["enabled"] is True and request["cancelled"] is False,"captured active request mask changed"
    assert record["request_numeric_sha256"]==digest(request)
    assert output["request_id"]==request["request_id"] and output["input_sha256"]==digest(request)
    r=request_from_record(request);got=result_from_record(output["result"])
    assert got.request_id==r.request_id
    groups,fallback,early=prepare_range_requests([r])
    assert len(groups)==1 and not fallback and not early
    counts=check(r,got)
    assert counts==output["exact_counts"]
    same=[float(got.lo[0]).hex(),float(got.hi[0]).hex()]==record["scalar_cpu_bounds"]
    assert same==output["legacy_bitwise_equal"]
    return counts,got,same


def check_corpus(root):
    corpus=read(root/"REQUEST_CORPUS_MANIFEST.json")
    assert corpus["numerical_sources"]==numerical_sources()
    assert corpus["partition_plan_sha256"]==sha(root/"PARTITION_PLAN.json")
    for name,checksum in corpus["files"].items():assert sha(root/name)==checksum
    for relative,files in corpus["checkpoint_sources"].items():
        assert all(sha(ROOT/relative/name)==checksum for name,checksum in files.items())
        core.load_terminal_checkpoint(ROOT/relative)
    from .corpus import partition_plan
    expected=partition_plan();saved=read(root/"PARTITION_PLAN.json")
    expected.pop("recorded_before_capture_utc");actual=dict(saved);actual.pop("recorded_before_capture_utc")
    assert digest(expected)==digest(actual)
    records=list(read_records(root/"raw/corpus/independent.jsonl.gz"))+list(read_records(root/"raw/corpus/offline.jsonl.gz"))
    ids=[r["request"]["request_id"] for r in records]
    assert len(ids)==len(set(ids))==corpus["requests"]
    tasksteps=defaultdict(list);stages=defaultdict(list)
    for record in records:
        assert record["prepared_remainder_replay"] is True and record["packed_boundary_execution"] is True
        assert record["source_sha"]==corpus["source_sha"] and record["call_position"]
        if record["scope"]=="CONCURRENT_TASK_REQUEST_CAPTURE_AND_REPLAY":
            assert record["case"]==f"{record['plant']}/task{record['task']:02d}" and 0<=record["task"]<32 and record["step"] in (1,2)
        else:assert record["scope"]=="OFFLINE_REQUEST_REPLAY" and record["task"] is None
        tasksteps[record["case"],record["step"]].append(record)
        stages[record["scope"],record["plant"],record["step"],record["ordinal"]].append(record)
    assert len(tasksteps)==64*2+12
    for rows in tasksteps.values():assert [r["ordinal"] for r in rows]==list(range(len(rows)))
    expected_groups=[]
    for key,rows in stages.items():
        groups,fallback,early=prepare_range_requests([request_from_record(r["request"]) for r in rows])
        assert not fallback and not early
        for g in groups:
            expected_groups.append(dict(scope=key[0],plant=key[1],step=key[2],ordinal=key[3],structure_sha256=digest(g.key),
                group_size=len(g.requests),terms=len(g.plan.exponents),request_ids="|".join(r.request_id for r in g.requests)))
    same_csv(root/"grouping_statistics.csv",expected_groups)
    summaries={};outputs={}
    for backend in ("cpu","cuda"):
        raw=list(read_records(root/f"raw/operator_outputs/{backend}.jsonl.gz"))
        by_id={r["request_id"]:r for r in raw}
        assert len(raw)==len(by_id) and set(by_id)==set(ids)
        counts=Counter()
        for record in records:
            count,got,same=check_record(record,by_id[record["request"]["request_id"]])
            counts.update(count);counts["requests"]+=1;counts[got.status]+=1
            counts["legacy_bitwise_equal" if same else "legacy_bitwise_different"]+=1
        summary=read(root/("cpu_batch_equivalence.json" if backend=="cpu" else "cuda_arithmetic_checks.json"))
        assert all(summary[k]==v for k,v in counts.items()) and summary["all_requests_finite"] is True
        if backend=="cpu":assert counts["legacy_bitwise_equal"]==len(records)
        summaries[backend]=dict(counts);outputs[backend]=by_id
    check_invocations(read(root/"cuda_actual_invocations.json"))
    events=read(root/"cuda_actual_invocations.json")["raw_device_events"]
    assert [(tuple(e["stage"]),e["rows"]) for e in events]==[(tuple(row[k] for k in ("scope","plant","step","ordinal")),row["group_size"]) for row in expected_groups]
    return records,outputs,summaries,tasksteps


def same_csv(path,rows):
    with path.open() as f:actual=list(csv.DictReader(f))
    assert actual==[{k:str(v) for k,v in row.items()} for row in rows],f"derived CSV mismatch: {path.name}"


def check_timing(root,records,outputs):
    samples,summary=timing_summary(root)
    plan=read(root/"PARTITION_PLAN.json")
    for plant in ("van_der_pol","brusselator"):
        for batch in (1,8,32):
            selected=[r for r in records if r["scope"]=="CONCURRENT_TASK_REQUEST_CAPTURE_AND_REPLAY" and r["plant"]==plant and r["task"] in plan["subsets"][str(batch)]]
            stages=defaultdict(list)
            for r in selected:stages[r["step"],r["ordinal"]].append(r)
            ordered=[];sizes=[]
            for _,rows in sorted(stages.items()):
                rows.sort(key=lambda r:r["task"]);ordered.extend(rows)
                groups,fallback,early=prepare_range_requests([request_from_record(r["request"]) for r in rows]);assert not fallback and not early
                sizes.extend(len(g.requests) for g in groups)
            identity=digest([r["request_numeric_sha256"] for r in ordered])
            for row in (r for r in samples if r["plant"]==plant and r["batch"]==batch):
                assert row["group_sizes"]==sizes and row["workload_sha256"]==identity
                assert row["terms"]==sum(len(r["request"]["exponents"]) for r in selected)
                if row["mode"]=="S_scalar_full":values={r["request"]["request_id"]:r["scalar_cpu_bounds"] for r in selected}
                else:
                    backend="cuda" if row["mode"].startswith("GPU") else "cpu"
                    values={r["request"]["request_id"]:[outputs[backend][r["request"]["request_id"]]["result"][k]["values"][0] for k in ("lo","hi")] for r in selected}
                assert row["output_sha256"]==digest(values)
    same_csv(root/"timing_summary.csv",summary)
    columns=["plant","batch","block","position","mode","seconds","requests","valid_requests","terms","actual_kernel_invocations","peak_allocated_bytes"]
    same_csv(root/"timing_samples.csv",[{k:r[k] for k in columns} for r in samples])
    assert read(root/"RESULT.json")==decision(root)
    assert read(root/"PROJECTED_AMDAHL.json")==projected(root)
    env=read(root/"raw/timing/environment.json")
    assert env["numerical_sources"]==numerical_sources() and env["affinity"]==[2] and env["threads"]==env["interop_threads"]==1
    assert env["baseline_switches"]==dict(prepared_remainder_replay=True,packed_boundary_execution=True)
    assert len(read(root/"raw/timing/residency.json"))==6
    for r in read(root/"raw/timing/residency.json"):assert r["explicit_resident_reuses"]==6 and r["input_bytes"]>0
    costs=read(root/"raw/timing/costs.json")
    assert costs and all(r["scope"]=="SEPARATE_INSTRUMENTED_TRAVERSAL" and all(v>=0 for k,v in r.items() if k.endswith("_s")) for r in costs)
    from .build import cost_rows
    same_csv(root/"transfer_and_packing_costs.csv",cost_rows(root))
    from .derive import width_differences
    assert read(root/"width_difference_summary.json")==width_differences(root)
    return summary


def replay_requests(root,tasksteps):
    from .corpus import capture_ranges
    from .integration import initial_task
    ordered=sorted(tasksteps.items())
    lane_states={}
    for (case,index),expected in ordered:
        first=expected[0]
        if first["scope"]=="OFFLINE_REQUEST_REPLAY":
            loaded=core.load_terminal_checkpoint(ROOT/first["checkpoint"]);current,state=loaded.current,loaded.normal_state
        elif index==1:current,state=initial_task(first["plant"],first["task"])
        else:current,state=lane_states[case]
        before=state_digest((current,state))
        assert before==first["source_state_sha256"]
        fields={"ordinal","call_position","request","scalar_cpu_bounds","observed_original_range_s","request_numeric_sha256"}
        metadata={k:v for k,v in first.items() if k not in fields}
        actual=[]
        with prepared_remainder_replay(True),packed_boundary_execution(True),capture_ranges(actual,metadata):
            result=step(first["plant"],current,state,index)
        assert result.status=="validated" and state_digest((current,state))==before
        clean=lambda rows:[{k:v for k,v in r.items() if k!="observed_original_range_s"} for r in rows]
        assert digest(clean(actual))==digest(clean(expected)),f"real request replay differs: {case} {index}"
        lane_states[case]=result.reset_tm,result.flowstar_normal_state
    return len(ordered)


def verify(root=RUN, *, replay=True):
    files=manifest(root);source=check_source(root)
    records,outputs,math,tasksteps=check_corpus(root)
    summary=check_timing(root,records,outputs)
    tests,_=account(root)
    assert tests==read(root/"tests/TEST_ACCOUNTING.json")
    integration=read(root/"integration_checks.json")
    assert integration["cpu"]==read(root/"raw/integration_cpu.json") and integration["gpu"]==read(root/"raw/integration_gpu.json")
    assert integration["cpu_full_state_bitwise_equal"] and not integration["published_box_used_as_next_state"]
    assert integration["old_full_runs"]=="REUSED" and integration["new_full_1000_step_runs"]==0
    replayed=0
    if replay:
        replayed=replay_requests(root,tasksteps)
        from .integration import verify_cpu,verify_gpu
        assert verify_cpu()==integration["cpu"]
        assert verify_gpu()==integration["gpu"]
    return dict(passed=True,files=files,numerical_sha=NUMERICAL_SHA,evidence_code_sha=source["evidence_code_sha"],
        checkout_sha=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        math=math,tests=tests["unique_totals"],request_stages_replayed=replayed,integration_replayed=replay,
        status=decision(root)["status"],full_long_runs_repeated=False,measured_solver_speedup=False)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",nargs="?",type=Path,default=RUN);p.add_argument("--no-replay",action="store_true");a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    print(json.dumps(verify(a.root,replay=not a.no_replay),indent=2),flush=True)
