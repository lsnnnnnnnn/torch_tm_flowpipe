"""Verify a complete live-range package without modifying its evidence.

Default verification includes all saved request recomputation and a bounded fresh
two-step real solve per backend/plant. It does not repeat the performance campaign
or historical 1000-step runs. Derived tables are regenerated in a temporary mirror
and compared, so rehashing a changed table or decision cannot make it pass.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import torch

from experiments.live_range_solver.analyze import diagnostic_gate, formal_summary, compare_state
from experiments.live_range_solver.faults import verify_faults
from experiments.live_range_solver.probes import verify_probe
from experiments.live_range_solver.runner import ROOT, run_case
from experiments.live_range_solver.verify import load_run, verify_run, check_lifecycle, canonical_digest
from experiments.range_batch_device.common import read, sha, digest, request_from_record
from torch_tm_flowpipe.range_requests import evaluate_range_requests


DERIVED_FILES = (
    "cross_backend_widths.csv", "width_warnings.csv", "width_summary.csv",
    "same_backend_state_equivalence.json", "history_reset_checks.json",
    "request_lifecycle_summary.csv", "timings_raw.csv", "actual_grouping.csv",
    "resource_usage.csv", "end_to_end_summary.csv", "paired_speedups.csv",
    "amdahl_reference.csv", "PERFORMANCE_RESULT.json",
)


def check_manifest(root):
    listed = {}
    for line in (root/"SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        path = Path(relative)
        assert not path.is_absolute() and ".." not in path.parts
        assert relative not in listed
        assert sha(root/path) == checksum, f"checksum mismatch: {relative}"
        listed[relative] = checksum
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and str(p.relative_to(root))!="SHA256SUMS"}
    assert actual == set(listed), "unlisted or missing evidence files"
    return len(listed)


def check_sources(root):
    source = read(root/"SOURCE_MAP.json")
    assert source["scientific_sha"]=="ef4e2f0c17518a4c4989071a596cd8631235c5ea"
    current = {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"src/torch_tm_flowpipe").iterdir() if p.suffix in {".py",".cu"}}
    assert current == source["numerical_sources"], "numerical source differs from frozen experiment"
    for path, checksum in source["experiment_sources"].items():
        assert sha(ROOT/path) == checksum, f"scientific runner changed: {path}"
    for path in dict(source["numerical_sources"],**source["experiment_sources"]):
        frozen=subprocess.check_output(["git","show",source["scientific_sha"]+":"+path],cwd=ROOT)
        assert (ROOT/path).read_bytes()==frozen, f"file does not belong to the scientific commit: {path}"
    for path, checksum in source["references"].items():
        assert sha(ROOT/path) == checksum
        frozen=subprocess.check_output(["git","show",source["parent"]+":"+path],cwd=ROOT)
        assert (ROOT/path).read_bytes()==frozen, f"fixed inherited reference changed: {path}"
    for path, checksum in source.get("packaging_sources",{}).items():
        assert sha(ROOT/path) == checksum, f"post-measurement evidence code changed: {path}"
    checkpoints=read(root/"CHECKPOINT_REFERENCES.json")
    assert len(checkpoints)==2
    for checkpoint in checkpoints:
        assert checkpoint["scope"]=="RESUMED_LOCAL_WINDOW"
        for name,checksum in checkpoint["files"].items():
            relative=checkpoint["path"]+"/"+name
            assert sha(ROOT/relative)==checksum
            frozen=subprocess.check_output(["git","show",source["parent"]+":"+relative],cwd=ROOT)
            assert (ROOT/relative).read_bytes()==frozen
        scheduler=read(ROOT/checkpoint["path"]/"terminal_state.json")["scheduler"]
        assert Fraction(scheduler["time_exact"])==scheduler["accepted_steps"]*Fraction(float.fromhex(scheduler["next_h_hex"]))
    # Loaders and fixed configuration helpers are part of the actual call chain,
    # even though they predate this integration's experiment directory.
    for path in (
        "experiments/endpoint_roundoff_repair/frozen.py",
        "experiments/run_vdp_dense_backend.py",
        "experiments/run_brusselator_sr1000_parity.py",
        "experiments/boundary_execution/state_equivalence.py",
        "experiments/range_batch_device/common.py",
        "experiments/range_batch_device/oracle.py",
    ):
        frozen = subprocess.check_output(["git","show",source["scientific_sha"]+":"+path],cwd=ROOT)
        assert (ROOT/path).read_bytes() == frozen, f"runtime dependency changed: {path}"
    kernel = subprocess.check_output(["git","show","440714afc3acc250903c5ec98fb0011905671944:src/torch_tm_flowpipe/range_cuda_kernel.cu"],cwd=ROOT)
    assert (ROOT/"src/torch_tm_flowpipe/range_cuda_kernel.cu").read_bytes() == kernel
    contract = read(root/"EXECUTION_CONTRACT.json")
    assert contract["cpu_affinity"] == [2]
    assert contract["intra_op_threads"] == contract["inter_op_threads"] == 1
    assert contract["max_group"] == 32 and contract["max_wait_s"] == .020
    assert not contract["default_enabled"] and not contract["new_gpu_full_1000"]
    assert not contract["entire_solver_formal_proof"]
    return source


def check_initial_source(run):
    """Rebuild the fixed first state; an internal digest alone is insufficient."""
    if not run["diagnostic"]:
        return 0
    from experiments.live_range_solver.runner import initial
    from experiments.boundary_execution.state_equivalence import canonical
    checkpoint=ROOT/run["checkpoint"] if run["checkpoint"] else None
    for task,rows in run["records"].items():
        expected=initial(run["plant"],int(task),original=run["original"],checkpoint=checkpoint)
        assert rows[0]["input"]==canonical(expected), "recorded first state differs from the fixed initial source"
    return len(run["records"])


def check_execution_metadata(root):
    """Keep raw process order and all accepted step spans inside the wall timer."""
    source=read(root/"SOURCE_MAP.json")
    receipts=[]
    for phase in ("diagnostic","formal"):
        folder=root/phase
        plan=read(folder/"CAMPAIGN_PLAN.json")
        complete=read(folder/"COMPLETED.json")
        jobs=read(folder/"jobs.json")
        assert complete["phase"]==phase and complete["source_sha"]==source["scientific_sha"]
        assert complete["cases"]==len(plan["cases"])==len(jobs)
        assert plan["cpu_affinity"]==[2] and plan["intra_op"]==plan["inter_op"]==1
        assert plan["cuda_visible_devices"]=="0"
        assert plan["scientific_sources"]==dict(source["numerical_sources"],**source["experiment_sources"])
        previous_end=0.
        for case,job in zip(plan["cases"],jobs):
            assert job["case"]==case["name"] and job["source_sha"]==source["scientific_sha"]
            assert job["status"]=="COMPLETED" and job["exit_code"]==0
            assert job["start_epoch"]>=previous_end and job["end_epoch"]>job["start_epoch"]
            previous_end=job["end_epoch"]
            assert sha(folder/(case["name"]+".log"))==job["log_sha256"]
            command=job["command"]
            assert command[:3]==["taskset","-c","2"]
            assert command[4:6]==["-m","experiments.live_range_solver.runner"]
            assert ("--diagnostic" in command)==(phase=="diagnostic") and "--warm" in command
            run,_=load_run(folder/case["name"])
            check_initial_source(run)
            assert run["wall_s"]<=job["end_epoch"]-job["start_epoch"]
            assert run["torch_version"]=="2.5.1+cu121" and "3.11.15" in run["python"]
            assert run["imported_core"].endswith("/repo/src/torch_tm_flowpipe/__init__.py")
            counters=0
            for task,rows in run["records"].items():
                last=run["start_ns"]
                boundary=rows[0]["input"][1]["fields"]["step_index"] if run["diagnostic"] else 0
                expected_h=float(.01 if run["plant"]=="van_der_pol" else .02).hex()
                for row in rows:
                    assert row["accepted"]
                    boundary+=1
                    assert row["generation"]==row["state_step"]==boundary
                    assert row["h_hex"]==expected_h, "main workload step changed"
                    assert last<=row["step_start_ns"]<row["step_end_ns"]<=run["end_ns"]
                    last=row["step_end_ns"]
                counters+=rows[-1]["request_counter"]
            assert counters==run["counts"].get("submitted",0)
            service_cpu=sum(g["thread_cpu_ns"] for g in run["groups"])
            assert service_cpu<=run["cpu_end_ns"]-run["cpu_start_ns"]+1000000
            for group in run["groups"]:
                assert 0<=group["thread_cpu_ns"]<=group["end_ns"]-group["start_ns"]+1000000
            if run["route"] in {"G","S_gpu"}:
                cold=run["cold_startup_outside_timing"]
                assert cold["wall_s"]>0
                build=cold["build"]
                assert build["source_sha256"]==source["numerical_sources"]["src/torch_tm_flowpipe/range_cuda_kernel.cu"]
                assert build["capability"]==[7,0] and build["nvrtc_version"]==[12,1]
                assert build["device_name"]=="Tesla V100-SXM2-16GB"
                assert all(flag in build["flags"] for flag in ("--fmad=false","--ftz=false","--prec-div=true","--prec-sqrt=true"))
            receipts.append(dict(phase=phase,run_id=run["run_id"],accepted_step_spans=sum(map(len,run["records"].values()))))
    return receipts


def check_policy(root):
    policy = read(root/"policy_pilot/POLICY_SELECTION.json")
    assert policy["source_sha"]=="b90bccd9e051e882c8e40d965891e6d6a30c9e24"
    totals = {"0.002":0., "0.02":0.}
    assert policy["candidates"] == [.002,.020] and not policy["formal_results_used"]
    assert len(policy["sample_names"]) == 8
    seen = set()
    for name in policy["sample_names"]:
        run,events = load_run(root/"policy_pilot"/name)
        verify_run(run,events,recompute=False)
        assert run["source_sha"] == policy["source_sha"]
        assert run["steps"] == 2 and run["ids"] == list(range(0,32,4))
        assert run["successful_lane_steps"] == 16 and run["successful_tasks"] == 8
        assert run["execution"] == "ONLINE_LIVE_SOLVE" and not run["previous_answers_loaded"]
        assert run["route"] in {"Q","G"}
        key = (run["plant"],run["route"],run["max_wait_s"])
        assert key not in seen
        seen.add(key)
        totals[str(run["max_wait_s"])] += run["wall_s"]
    assert totals == policy["scores"]
    assert float(min(totals,key=lambda k:(totals[k],float(k)))) == policy["chosen_wait_s"] == .020
    return policy


def check_fault_records(root):
    receipts = []
    for plant in ("van_der_pol","brusselator"):
        for backend in ("cpu","cuda"):
            folder = root/"faults"/f"{plant}-{backend}"
            summary = read(folder/"summary.json")
            assert all(sha(folder/name)==checksum for name,checksum in summary["files"].items())
            with gzip.open(folder/"faults.json.gz","rt") as stream:
                artifact = json.load(stream)
            assert artifact["baseline"]["source_sha"]==read(root/"SOURCE_MAP.json")["scientific_sha"]
            receipt = verify_faults(artifact)
            assert all(summary[key] == value for key,value in receipt.items())
            for record in artifact["attempts"]:
                if record["outcome"] == "rejected":
                    fields = record["segment"]["fields"]
                    assert fields["status"] != "validated" and fields["reset_tm"] is None
            # Check every identity/dependency/group, including the old worker's
            # discarded response. Synthetic NaNs are explicit injection records;
            # all successful real returns still receive exact rational checks.
            events = artifact["events"]
            groups = [e["group"] for e in events if e["event"]=="group"]
            dispatch = {e["request_id"]:e for e in events if e["event"]=="dispatch"}
            waits = [dispatch[rid]["dispatched_ns"]-dispatch[rid]["submitted_ns"] for g in groups for rid in g["request_ids"]]
            source = {(r["task"],r["generation"]):canonical_digest(r["input"]) for r in artifact["attempts"]}
            pseudo = dict(run_id=f"fault-{plant}-{backend}",route="Q" if backend=="cpu" else "G",
                          counts=artifact["service_counts"],groups=groups,wait_ns=waits)
            check_lifecycle(pseudo,deepcopy(events),source,recompute=False)
            for injection in artifact["injections"]:
                request = request_from_record(injection["request"])
                result = evaluate_range_requests([request],backend=backend)[request.request_id]
                assert not result.ok and result.status=="invalid"
            for name,checksum in artifact["checkpoints"].items():
                assert sha(folder/"accepted_checkpoint"/name)==checksum
            receipts.append(receipt)
    assert receipts == read(root/"faults/failure_cancel_resume.json")
    return receipts


def check_probes_and_coverage(root):
    checks = []
    for summary in read(root/"probes/probe_checks.json"):
        file = root/"probes"/summary["file"]
        assert sha(file)==summary["sha256"]
        with gzip.open(file,"rt") as stream:
            artifact=json.load(stream)
        receipt = verify_probe(artifact)
        fallback_groups=[g for g in artifact["groups"] if g["hardware_fallback"]]
        assert sum(g["size"] for g in fallback_groups)==artifact["counts"].get("hardware_fallback_requests",0)
        for group in fallback_groups:
            timing=group["timing"]
            assert timing["failed_cuda_s"]>0 and timing["cpu_recompute"]["total_s"]>0
            assert timing["failed_cuda_s"]+timing["cpu_recompute"]["total_s"]<=(group["end_ns"]-group["start_ns"])/1e9+1e-6
        assert all(summary[k]==v for k,v in receipt.items())
        checks.append(receipt)
    for summary in read(root/"coverage/coverage_summary.json"):
        file = root/"coverage"/summary["raw_file"]
        assert sha(file)==summary["sha256"]
        with gzip.open(file,"rt") as stream:
            raw = json.load(stream)
        assert all(summary[k]==v for k,v in raw["summary"].items())
        assert len(raw["submitted"]) == summary["strict_adapter_requests"]
        assert len(raw["observed"]) == summary["observed_sparse_calls"]
        assert len(raw["unmatched"]) == summary["uncovered_or_observer_calls"]
        assert sum(not e["request"]["exponents"] for e in raw["observed"]) == summary["observed_empty_support"]
        def values(request):
            request = dict(request)
            request.pop("request_id")
            return digest(request)
        observed = Counter(values(e["request"]) for e in raw["observed"])
        submitted = Counter(values(e["request"]) for e in raw["submitted"])
        unmatched = Counter(values(e["request"]) for e in raw["unmatched"])
        assert not (submitted-observed) and observed-submitted == unmatched
    return checks


def check_heterogeneous(root):
    file=Path(__file__).with_name("heterogeneous_steps.py")
    spec=importlib.util.spec_from_file_location("live_heterogeneous_verification",file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipts=[]
    for backend in ("cpu","cuda"):
        folder=root/"heterogeneous"/backend
        summary=read(folder/"summary.json")
        assert sha(folder/summary["file"])==summary["sha256"]
        with gzip.open(folder/summary["file"],"rt") as stream:
            artifact=json.load(stream)
        assert artifact["script_sha256"]==sha(file)
        assert artifact["core_scientific_sha"]==read(root/"SOURCE_MAP.json")["scientific_sha"]
        assert artifact["test_inputs_only"] and artifact["not_a_performance_sample"]
        assert artifact["torch_threads"]==artifact["interop_threads"]==1
        receipt=module.verify(artifact)
        assert all(summary[k]==v for k,v in receipt.items())
        events=artifact["events"]
        dispatch={e["request_id"]:e for e in events if e["event"]=="dispatch"}
        waits=[dispatch[rid]["dispatched_ns"]-dispatch[rid]["submitted_ns"] for g in artifact["groups"] for rid in g["request_ids"]]
        sources={(task,index):canonical_digest(row["input"]) for task,rows in artifact["records"].items() for index,row in enumerate(rows)}
        pseudo=dict(run_id=f"heterogeneous-h-{backend}",route="Q" if backend=="cpu" else "G",groups=artifact["groups"],
                    counts=artifact["counts"],wait_ns=waits)
        check_lifecycle(pseudo,deepcopy(events),sources,recompute=True)
        receipts.append(receipt)
    return receipts


def test_identities(file):
    results = {}
    for case in ET.parse(file).getroot().iter("testcase"):
        key = case.attrib.get("classname", "")+"::"+case.attrib["name"]
        assert key not in results
        results[key] = "failed" if case.find("failure") is not None or case.find("error") is not None else "skipped" if case.find("skipped") is not None else "passed"
    return results


def check_tests(root):
    provenance = read(root/"tests/TEST_PROVENANCE.json")
    original = test_identities(root/"tests/science_tests.xml")
    updated = test_identities(root/"tests/science_targeted.xml")
    assert Counter(original.values()) == {"passed":1221,"skipped":2}
    assert Counter(updated.values()) == {"passed":28}
    combined = dict(original)
    combined.update(updated)
    assert Counter(combined.values()) == {"passed":1221,"skipped":2}
    parents = test_identities(root/"tests/parent_tests.xml")
    assert Counter(parents.values()) == {"passed":131}
    assert provenance["parent"]["label"]=="REUSED"
    return dict(current_unique=dict(Counter(combined.values())),parent_reused=131)


def mirror_phase(source, target):
    target.mkdir()
    for path in source.iterdir():
        if path.is_dir() or path.name in {"CAMPAIGN_PLAN.json","COMPLETED.json","jobs.json"}:
            (target/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())


def bounded_live_replay(root):
    receipts=[]
    for plant in ("van_der_pol","brusselator"):
        for route in ("S","Q","S_gpu","G"):
            reference, reference_events = load_run(root/"diagnostic"/f"small-b2-{plant}-{route}")
            actual, events, _ = run_case(plant,[0,31],2,route,diagnostic=True,run_id=f"fresh-{plant}-{route}")
            assert actual["successful_tasks"] == 2
            assert actual["accepted_lane_steps"] == actual["successful_lane_steps"] == 4
            assert set(actual["records"]) == set(reference["records"]) == {"0","31"}
            assert all(len(rows) == 2 for rows in actual["records"].values())
            assert actual["plant"] == plant and actual["route"] == route
            assert actual["diagnostic"] and not actual["previous_answers_loaded"]
            assert actual["imported_core"] == str(ROOT/"src/torch_tm_flowpipe/__init__.py")
            equality = compare_state(reference,actual)
            assert equality["complete_segments_compared"] == 4
            def sequence(rows):
                values=defaultdict(list)
                for event in rows:
                    if event["event"]=="submit":
                        request=dict(event["request"])
                        request.pop("request_id")
                        values[event["task"]].append((event["generation"],event["attempt"],event["counter"],event["source_state"],request))
                return dict(values)
            assert sequence(events)==sequence(reference_events), "fresh live request did not match the claimed source attempt"
            equality.update(accepted_lane_steps=actual["accepted_lane_steps"],
                            actual_imported_core=actual["imported_core"],
                            actual_checkout_sha=actual["source_sha"],execution=actual["execution"])
            receipts.append(equality)
    return receipts


def verify_package(root, *, live_replay=True, recompute=True):
    root=Path(root).resolve()
    files=check_manifest(root)
    sources=check_sources(root)
    execution=check_execution_metadata(root)
    check_policy(root)
    faults=check_fault_records(root)
    probes=check_probes_and_coverage(root)
    heterogeneous=check_heterogeneous(root)
    tests=check_tests(root)
    with tempfile.TemporaryDirectory(prefix="live-range-package-verify-") as directory:
        temp=Path(directory)
        mirror_phase(root/"diagnostic",temp/"diagnostic")
        mirror_phase(root/"formal",temp/"formal")
        gate=diagnostic_gate(temp/"diagnostic",recompute=recompute)
        performance=formal_summary(temp/"formal")
        if recompute:
            by_run={r["run_id"]:r for r in gate["raw_verifications"]}
            for file in sorted((root/"incremental_checks").glob("*.json")):
                incremental=read(file)
                assert incremental["source_sha"]==sources["scientific_sha"]
                assert incremental["scope"]=="INCREMENTAL_DIAGNOSTIC_RECHECK_NOT_PERFORMANCE"
                receipt=incremental["raw_verification"]
                assert receipt==by_run[receipt["run_id"]]
                assert incremental["same_backend"] in gate["same_backend_comparisons"]
                # Incremental summaries are supplementary observations, with
                # the same raw case rechecked by the complete final gate.
                candidate,_=load_run(root/"diagnostic"/receipt["run_id"])
                reference,_=load_run(root/"diagnostic"/(receipt["run_id"].removesuffix("-G")+"-S"))
                from experiments.live_range_solver.analyze import widths
                values=widths(reference,candidate,"prefix-b32")
                assert incremental["widths"]==dict(width_rows=len(values),warnings=[r for r in values if r["warning"]],
                    max_width_ratio=max(r["gpu_cpu_width_ratio"] for r in values if not r["near_zero"]),
                    max_abs_difference=max(r["max_abs_diff"] for r in values))
        for name in DERIVED_FILES:
            assert (root/name).read_bytes()==(temp/name).read_bytes(), f"raw-derived table or decision changed: {name}"
        module_file=Path(__file__).with_name("enrich.py")
        spec=importlib.util.spec_from_file_location("live_extra_tables",module_file)
        extra=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extra)
        extra.enrich(root,temp)
        for name in extra.FILES:
            assert (root/name).read_bytes()==(temp/name).read_bytes(), f"supplemental raw-derived table changed: {name}"
        if recompute:
            assert read(root/"diagnostic/CORRECTNESS_GATE.json")==gate
        result=read(root/"RESULT.json")
        assert result["status"]==performance["status"]
        assert result["performance"]==performance
        assert result["correctness_gate_passed"] is True
        assert result["measured_throughput_online"] is True
        assert result["new_full_long_horizon_gpu"] is False
        assert result["entire_solver_formally_proved"] is False
    replay=bounded_live_replay(root) if live_replay else []
    return dict(verified=bool(recompute and live_replay),checked_files=files,scientific_sha=sources["scientific_sha"],
                checkout_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                tests=tests,execution_cases_checked=len(execution),fault_cases=len(faults),probe_cases=len(probes),heterogeneous_cases=heterogeneous,bounded_actual_replay=replay,
                fresh_accepted_lane_steps=sum(r["accepted_lane_steps"] for r in replay),
                historical_1000_repeated=False,full_performance_matrix_repeated=False,status=performance["status"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",type=Path)
    parser.add_argument("--no-live-replay",action="store_true")
    parser.add_argument("--no-recompute",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps(verify_package(args.path,live_replay=not args.no_live_replay,recompute=not args.no_recompute),indent=2),flush=True)


if __name__=="__main__":
    main()
