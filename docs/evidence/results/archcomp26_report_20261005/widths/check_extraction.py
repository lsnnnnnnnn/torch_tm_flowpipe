#!/usr/bin/env python3
"""New table correspondence QA only. No historical numerical checker is imported."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import struct

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
BASE = ROOT / "docs/evidence/results/archcomp26_20261001"


def forbidden(*args, **kwargs):
    raise RuntimeError("content digest operations prohibited")


for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
    if hasattr(hashlib, name):
        setattr(hashlib, name, forbidden)


def read(path):
    return json.loads(Path(path).read_text())


def local(path):
    p = Path(path)
    for candidate in (p, ROOT / p, BASE / p):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def saved_pair(source, instance, method, state, step, states):
    """Independently re-read one named step/state, separate from the exporter."""
    index = states.index(state)
    if instance.startswith("nav-"):
        path = BASE / "nav_fourstate_saved_reduction_20261004.json"
        doc = read(path)
        method_name = "working_p3" if method == "pytorch_gpu" else method
        run = next(r for r in doc["methods"] if r["instance"] == instance and r["method"] == method_name)
        values = run["cells"][step-1][doc["states"].index(state)]
        return {"tube": values[:2], "endpoint": values[2:4]}, str(path), "saved independent reduction; raw binary remains remote"
    if instance == "quad-reach" and method == "flowstar_native":
        path = BASE / "native_quad_paper_full50_001/SCAN.json"
        record = read(path)["per_step"][step-1]["states"][index]
        assert record["state"] == index+1
        return {k: [record[k+"_lo"],record[k+"_hi"]] for k in ("tube","endpoint")}, str(path), "saved SCAN per_step, not a rerun"
    path = local(source.get("raw_source") or source["path"])
    if path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["substep"] != step:
                    continue
                if "tube_endpoint_union_12x4" in record:
                    values = record["tube_endpoint_union_12x4"][index]
                    pair = {"tube": values[:2], "endpoint": values[2:]}
                elif isinstance(record.get("accepted"), list):
                    accepted = [i for i,v in enumerate(record["accepted"]) if v]
                    pair = {k: [min(record[k][i][index][0] for i in accepted),
                                max(record[k][i][index][1] for i in accepted)] for k in ("tube","endpoint")}
                else:
                    assert record.get("accepted", True) is True
                    pair = {k:record[k][index] for k in ("tube","endpoint")}
                return pair, str(path), "raw saved JSONL"
        raise ValueError((path,step))
    if path.suffix == ".bin":
        fmt = struct.Struct(f"<QQd{4*len(states)}d")
        selected = [r[3+4*index:3+4*index+4] for r in fmt.iter_unpack(path.read_bytes()) if r[1] == step]
        assert selected
        return {"tube": [min(v[0] for v in selected),max(v[1] for v in selected)],
                "endpoint": [min(v[2] for v in selected),max(v[3] for v in selected)]}, str(path), "raw saved binary, pooled by named step"
    if path.name == "metrics.json":
        return {"endpoint":read(path)["final_hull"][state]}, str(path), "saved driver final_hull"
    raise ValueError(path)


def main():
    sources = {s["source_id"]:s for s in read(OUT/"sources.json")}
    summaries = read(OUT/"summary.json")
    specs = read(OUT/"instances.json")
    groups = defaultdict(list)
    with (OUT/"widths_long.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            for k in ("lo","hi","absolute_width","t_start","t_end"):
                r[k] = float(r[k])
            r["step"] = int(r["step"])
            r["complete_initial_set"] = r["complete_initial_set"] == "True"
            assert r["absolute_width"] == r["hi"]-r["lo"]
            groups[r["instance_id"],r["method"],r["state"]].append(r)
    samples, missing = [], []
    for spec in specs:
        instance, states = spec["instance_id"], spec["physical_states"]
        for method, choose_last in (("pytorch_gpu",False),("huan",True)):
            candidates = [r for r in summaries if r["instance_id"] == instance and r["method"] == method and r["own_last_complete_t"] is not None]
            if not candidates:
                missing.append(dict(instance_id=instance,method=method,reason="No accepted main-contract source row exists; no diagnostic or supplementary row substituted."))
                continue
            summary = candidates[-1] if choose_last else candidates[0]
            state = summary["state"]
            rows = groups[instance,method,state]
            step = round(summary["own_last_complete_t"] / spec["step_s"]) if choose_last else min(r["step"] for r in rows)
            row = next(r for r in rows if r["step"]==step and r["geometry"]=="endpoint" and
                       (instance!="quad-reach" or method!="pytorch_gpu" or r["saved_object"]=="pooled_observer"))
            original,path,level = saved_pair(sources[row["source_id"]],instance,method,state,step,states)
            for kind,pair in original.items():
                actual = next(r for r in rows if r["step"]==step and r["geometry"]==kind and r["saved_object"]==row["saved_object"])
                assert pair == [actual["lo"],actual["hi"]], (instance,method,state,step,kind,pair,actual)
            samples.append(dict(instance_id=instance,method=method,state=state,state_index=states.index(state),
                physical_state_order=states,step=step,source=path,source_level=level,bounds=original,exact_numeric_match=True))
    # Independently check two newly available native QUAD states at opposite ends.
    for state,step in (("x1",1),("x12",1000)):
        rows=groups["quad-reach","flowstar_native",state]
        original,path,level=saved_pair({},"quad-reach","flowstar_native",state,step,[f"x{i}" for i in range(1,13)])
        for kind,pair in original.items():
            r=next(r for r in rows if r["step"]==step and r["geometry"]==kind and r["saved_object"]=="pooled_observer")
            assert pair == [r["lo"],r["hi"]]
        samples.append(dict(instance_id="quad-reach",method="flowstar_native",state=state,step=step,
                            source=path,source_level=level,bounds=original,exact_numeric_match=True))
    checked=0
    for s in summaries:
        rows=groups[s["instance_id"],s["method"],s["state"]]
        for prefix in ("common", "common_safe"):
            t=s[prefix+"_time"]
            width=s[prefix+"_max_tube_width"]
            if width is not None:
                widths=[r["absolute_width"] for r in rows if r["geometry"]=="tube" and r["complete_initial_set"] and r["t_end"]<=t+1e-12]
                assert width==max(widths)
                checked+=1
    assert len(summaries)==364 and len({(s["instance_id"],s["method"]) for s in summaries})==64
    audit=dict(status="PASS",sample_count=len(samples),samples=samples,unavailable_main_contract_samples=missing,
        summary_maximum_correspondence_checks=checked,long_width_arithmetic_rows=sum(map(len,groups.values())),
        state_order_corrections=["NAV metadata changed to executable x,y,speed,heading; state-labelled numbers were unchanged.",
          "Airplane uses authoritative sx,sy,sz,vx,vy,vz labels; raw x,y,z,u,v,w aliases occupy the same positions. No accepted main-contract rows exist."],
        limitations=["Correspondence/format QA only; no historical property checker or solver was executed.",
          "NAV raw binaries remain remote; checks reach the saved independent reduction JSON.",
          "Native QUAD checks reach the original saved SCAN JSON, not a newly executed scan.",
          "QUAD Huan/Xiangru terminal objects provide no per-step tubes.",
          "Airplane auxiliary split diagnostics and rejected proposals are not main-contract accepted flowpipes."],
        new_numerical_runs=0,old_checkers_executed=0,digest_operations=0)
    (OUT/"EXTRACTION_QA.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:audit[k] for k in ("status","sample_count","summary_maximum_correspondence_checks","long_width_arithmetic_rows")}))


if __name__ == "__main__":
    main()
