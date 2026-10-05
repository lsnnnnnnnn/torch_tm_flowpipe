#!/usr/bin/env python3
"""Export saved physical-state bounds, without solver/checker/digest execution."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/evidence/results/archcomp26_20261001"
DEFAULT_OUT = ROOT / "docs/evidence/results/archcomp26_report_20261005/widths"
METHODS = ("pytorch_gpu", "huan", "xiangru", "flowstar_native")
ALIASES = {"P3": "pytorch_gpu", "ours/P3": "pytorch_gpu", "working_p3": "pytorch_gpu",
           "current_p3": "pytorch_gpu", "Huan": "huan", "Xiangru": "xiangru",
           "Native": "flowstar_native", "native": "flowstar_native", "Flow* native": "flowstar_native"}
X4 = tuple(f"x{i}" for i in range(1, 5))
AIR = ("sx", "sy", "sz", "vx", "vy", "vz", "phi", "theta", "psi", "r", "p", "q")
DP = ("theta1", "theta2", "theta1_dot", "theta2_dot")
SPECS = {
    "acc-safe-distance": (50, .1, ("x_lead", "v_lead", "a_lead", "x_ego", "v_ego", "a_ego"), 1),
    "airplane-continuous": (200, .01, AIR, 1),
    "airplane-discrete": (20, .1, AIR, 1),
    "attitude-control-avoid": (60, .05, tuple(f"x{i}" for i in range(1, 7)), 1),
    "balancing-reach": (2000, .005, X4, 1),
    "docking-constraint": (400, .1, ("sx", "sy", "vx", "vy"), 1),
    "double-pendulum-less-robust": (100, .01, DP, 225),
    "double-pendulum-more-robust": (80, .005, DP, 225),
    "nav-standard": (600, .01, ("x", "y", "speed", "heading"), 640),
    "nav-robust": (600, .01, ("x", "y", "speed", "heading"), 25),
    "quad-reach": (1000, .005, tuple(f"x{i}" for i in range(1, 13)), 1024),
    "single-pendulum-reach": (100, .01, ("x1", "x2", "x3"), 1),
    "tora-remain": (200, .1, X4, 12),
    "tora-reach-sigmoid": (500, .01, X4, 1),
    "tora-reach-tanh": (500, .01, X4, 1),
    "unicycle-reach": (500, .02, X4, 1),
}
FIELDS = ("instance_id", "method", "contract_id", "saved_object", "state", "step",
          "t_start", "t_end", "geometry", "lo", "hi", "absolute_width",
          "available_lanes", "expected_lanes", "complete_initial_set", "source_id", "source_locator")
SAVED_SAFE_PREFIXES = {
    "double-pendulum-more-robust": (60, "docs/evidence/results/archcomp26_20261001/dp_more_fourway_saved_20261004_001/geometry.json#common_saved_safe_steps"),
    "tora-remain": (184, "docs/ARCHCOMP26_TORA_REMAIN_NUMERIC_STOP_AUDIT_20261002.md"),
}


def no_digest(*a, **kw):
    raise RuntimeError("content digest operations prohibited")


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        yield from csv.DictReader(stream)


def relative(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


class Export:
    def __init__(self):
        self.rows, self.sources, self.source_keys = [], [], {}
        self.exclusions = []
        self.cells = {(c["instance_id"], c["method"]): c for c in read_json(
            ROOT / "docs/evidence/archcomp26_delivery_index_20261004.json")["cells"]}

    def source(self, path, *, raw_source=None, note=""):
        key = (relative(path), str(raw_source or ""), note)
        if key not in self.source_keys:
            sid = f"S{len(self.sources)+1:03d}"
            self.source_keys[key] = sid
            self.sources.append(dict(source_id=sid, path=key[0], bytes=Path(path).stat().st_size,
                                     raw_source=raw_source, note=note))
        return self.source_keys[key]

    def add(self, instance, method, state, step, kind, lo, hi, lanes, sid, locator,
            saved_object="pooled_observer", contract=None):
        method = ALIASES.get(method, method)
        steps, h, states, expected = SPECS[instance]
        if method not in METHODS or state not in states or not 1 <= step <= steps:
            raise ValueError((instance, method, state, step))
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError(f"invalid saved bound: {sid}:{locator}")
        contract = contract or ("fixed-repository-raw4-supplement" if instance == "balancing-reach"
                                else "fixed-official-two-state-submission" if instance == "single-pendulum-reach"
                                else "selected-current-report-contract")
        self.rows.append(dict(zip(FIELDS, (instance, method, contract, saved_object, state, step,
            (step-1)*h, step*h, kind, lo, hi, hi-lo, lanes, expected,
            lanes == expected, sid, locator))))

    def wide_csv(self, path, instance=None, source_map=None):
        for line, row in enumerate(csv_rows(path), 2):
            method = ALIASES.get(row["method"], row["method"])
            if method not in METHODS or row.get("recorded") == "False":
                continue
            inst = instance or {"attitude": "attitude-control-avoid", "docking": "docking-constraint"}.get(
                row.get("instance"), row.get("instance"))
            raw = row.get("source") or (source_map or {}).get((inst, method))
            sid = self.source(path, raw_source=raw, note="read-only existing pooled CSV; no old checker executed")
            for kind in ("tube", "endpoint"):
                if row.get(kind+"_lo", "") != "":
                    self.add(inst, method, row["state"], int(row["step"]), kind,
                             row[kind+"_lo"], row[kind+"_hi"], SPECS[inst][3], sid, f"csv_line:{line}")

    def jsonl(self, path, instance, method, *, pooled=False):
        sid = self.source(path)
        states, expected = SPECS[instance][2:]
        with Path(path).open() as stream:
            for line, text in enumerate(stream, 1):
                row = json.loads(text)
                step = row["substep"]
                if pooled:
                    values = row["tube_endpoint_union_12x4"]
                    lanes = row["accepted_count"]
                    pairs = (([v[:2] for v in values], [v[2:] for v in values]),)
                elif row.get("accepted") is False:
                    self.exclusions.append(dict(source_id=sid, line=line, reason="recorded rejected step; no accepted bounds used"))
                    continue
                elif isinstance(row["accepted"], list):
                    keep = [i for i, value in enumerate(row["accepted"]) if value]
                    lanes = len(keep)
                    pairs = [(row["tube"][i], row["endpoint"][i]) for i in keep]
                else:
                    lanes = expected
                    pairs = [(row["tube"], row["endpoint"])]
                if not lanes:
                    continue
                for j, state in enumerate(states):
                    for which, kind in enumerate(("tube", "endpoint")):
                        self.add(instance, method, state, step, kind,
                            min(item[which][j][0] for item in pairs), max(item[which][j][1] for item in pairs),
                            lanes, sid, f"jsonl_line:{line}")

    def binary(self, path, instance, method, *, stored_states=None, observations=None):
        states, expected = SPECS[instance][2:]
        stored_states = stored_states or len(states)
        record = struct.Struct(f"<QQd{4*stored_states}d")
        sid = self.source(path)
        rejected = {}
        if observations:
            with Path(observations).open() as stream:
                rejected = {r["substep"]: set(r["rejected_lanes"]) for line in stream if (r := json.loads(line))}
        groups = {}
        with Path(path).open("rb") as stream:
            for ordinal in range(1, Path(path).stat().st_size // record.size + 1):
                raw = stream.read(record.size)
                lane, step, h, *values = record.unpack(raw)
                if not 0 <= lane < expected or not math.isclose(h, SPECS[instance][1], abs_tol=1e-9):
                    raise ValueError(f"saved lane/grid mismatch at {path}:{ordinal}")
                if lane in rejected.get(step, ()):
                    continue
                group = groups.setdefault(step, {})
                if lane in group:
                    raise ValueError(f"duplicate lane/step {path}:{lane}/{step}")
                group[lane] = values
            if stream.read(1):
                raise ValueError(f"partial binary record: {path}")
        for step, group in sorted(groups.items()):
            for j, state in enumerate(states):
                for k, kind in ((0, "tube"), (2, "endpoint")):
                    self.add(instance, method, state, step, kind,
                        min(v[4*j+k] for v in group.values()), max(v[4*j+k+1] for v in group.values()),
                        len(group), sid, f"step:{step};accepted_lanes:{len(group)}")
        if rejected:
            self.exclusions.append(dict(source_id=sid, rejected_or_frozen_records=sum(map(len,rejected.values())),
                reason="raw stored rejected/frozen lane values excluded; surviving accepted lanes remain explicitly partial"))


def collect(ex):
    geometry = BASE / "report_saved_projections_20261004_001/saved_geometry.csv"
    ex.wide_csv(geometry)
    for folder, instance in (("balancing_raw4_fourway_saved_20261004_001/output", "balancing-reach"),
                             ("dp_more_fourway_saved_20261004_001", "double-pendulum-more-robust")):
        metadata = read_json(BASE / folder / "geometry.json")
        mapping = {(instance, ALIASES[k]): v["source_path"] for k,v in metadata["series"].items()}
        ex.wide_csv(BASE / folder / "saved_bounds.csv", instance, mapping)
    nav = BASE / "nav_fourstate_saved_20261004_001"
    mapping = {(r["instance"], ALIASES.get(r["method"],r["method"])): r["source"]
               for r in read_json(nav / "AUDIT.json")["methods"]}
    ex.wide_csv(nav / "all_states_saved_curves.csv", source_map=mapping)
    ex.wide_csv(BASE / "tora_tanh_current_p3_saved_20261004_001/saved_bounds.csv", "tora-reach-tanh")
    sp_sources = {("single-pendulum-reach", method): str(BASE / (
        "native_sp_two_state_full20_001/ranges.bin" if method == "flowstar_native" else
        f"single_pendulum_two_state_{'p3' if method == 'pytorch_gpu' else method}_full20_001/data/ranges.jsonl"))
        for method in METHODS}
    ex.wide_csv(BASE / "sp_two_state_fourway_saved_20261002/saved_tubes.csv", "single-pendulum-reach", sp_sources)
    for method, label in zip(METHODS, ("ours_p3", "huan", "xiangru", "native")):
        acc = BASE / f"acc_fourway_campaign_001/steady05_{label}"
        if method == "flowstar_native":
            ex.binary(acc / "ranges.bin", "acc-safe-distance", method)
        else:
            ex.jsonl(acc / "data/ranges.jsonl", "acc-safe-distance", method)
        sigmoid = BASE / f"tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_{label}"
        ex.binary(sigmoid / ("ranges.bin" if method == "flowstar_native" else "payload/ranges.bin"),
                  "tora-reach-sigmoid", method)
    for method in METHODS:
        if method == "pytorch_gpu":
            ex.jsonl(BASE / "dp_p3_affine_split4_v1/full225_smoke_001/attempt/data/observations.jsonl",
                     "double-pendulum-less-robust", method)
        else:
            path = "native_dp_less_full20_001/ranges.bin" if method == "flowstar_native" else f"author_dp_less_v1/{method}_full20_001/payload/ranges.bin"
            ex.binary(BASE / path, "double-pendulum-less-robust", method)
        label = "p3" if method == "pytorch_gpu" else method
        if method == "flowstar_native":
            ex.binary(BASE / "native_unicycle_paper_speed_full50_001/ranges.bin", "unicycle-reach", method)
            ex.binary(BASE / "native_tora_remain_full20_001/ranges.bin", "tora-remain", method)
        else:
            ex.jsonl(BASE / f"unicycle_paper_speed_w_constant_v1/{label}_full50_001/payload/ranges.jsonl",
                     "unicycle-reach", method)
            tora = BASE / ("p3_tora_remain_v1/full20_001/payload" if method == "pytorch_gpu"
                           else f"author_tora_remain_v1/{method}_full20_001/payload")
            ex.binary(tora / "ranges.bin", "tora-remain", method, observations=tora / "observations.jsonl")
    ex.jsonl(BASE / "quad_paper_p3_nohash_v1/full50_001/data/observations.jsonl", "quad-reach", "pytorch_gpu", pooled=True)
    terminal = BASE / "quad_paper_fourway_saved_20261002/terminal_12states_fourway.csv"
    for line, row in enumerate(csv_rows(terminal), 2):
        label = row["method"]
        if label == "ours/P3 observer":
            continue
        method = {"ours/P3 driver": "pytorch_gpu"}.get(label, ALIASES.get(label,label))
        if method not in METHODS:
            continue
        sid = ex.source(terminal, raw_source=row["source"], note=row["source_object"])
        ex.add("quad-reach", method, row["state"], 1000, "endpoint", row["endpoint_lo"], row["endpoint_hi"],
               1024, sid, f"csv_line:{line}", saved_object=row["source_object"])
    # The full native source is remote. Use an explicitly supplied new reducer
    # export when present; the historical local mirror contains only x3 series.
    native_all = DEFAULT_OUT / "native_quad_allstates.csv"
    if native_all.exists():
        ex.wide_csv(native_all, "quad-reach")
    else:
        p = BASE / "native_quad_paper_full50_001/pooled_x3_1000steps.csv"
        for line, row in enumerate(csv_rows(p), 2):
            sid = ex.source(p, note="historical native local mirror only retains per-step x3")
            for kind in ("tube", "endpoint"):
                ex.add("quad-reach", "flowstar_native", "x3", int(row["step"]), kind,
                       row[kind+"_x3_lo"], row[kind+"_x3_hi"], 1024, sid, f"csv_line:{line}")


def csv_write(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def equivalent_runs():
    """Read actual new receipts; never perform a historical rerun or comparison."""
    references = {
        "acc_fourway_campaign_001/steady05_ours_p3/data": "acc-safe-distance",
        "nav_author_robust_working_p3_full30_001": "nav-robust",
        "quad_paper_p3_nohash_v1/full50_001/data": "quad-reach",
        "tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_ours_p3/payload": "tora-reach-sigmoid",
        "unicycle_paper_speed_w_constant_v1/p3_full50_001/payload": "unicycle-reach",
    }
    output = []
    for path in sorted((ROOT / "research/p3_speed_tightness_20261005/results").glob("*/run_001/candidate/SAVED_COMPARISON.json")):
        comparison = read_json(path)
        result_path = path.with_name("RESULT.json")
        result = read_json(result_path)
        if result.get("status") != "COMPLETED_SAVED_OUTPUT_EQUIVALENT":
            continue
        instance = next((i for ref,i in references.items() if comparison.get("reference", "").endswith("/"+ref)), None)
        if instance is None or result.get("completed_substeps") != SPECS[instance][0]:
            continue
        output.append(dict(instance_id=instance, run=path.parts[-4], result=relative(result_path),
            comparison=relative(path), reference=comparison["reference"],
            completed_substeps=result["completed_substeps"], status=result["status"],
            scope=comparison["scope"], width_interpretation="Existing saved widths are inherited by the recorded equivalence scope; this is acceleration, not new tightening.",
            receipt_rechecked=False, independent_end_to_end_floating_nncs_certificate=False))
    return output


def summarize(ex):
    primary, alternate = defaultdict(list), []
    observer_keys = {(r["instance_id"],r["method"],r["state"]) for r in ex.rows
                     if r["saved_object"] == "pooled_observer"}
    for row in ex.rows:
        key = (row["instance_id"], row["method"], row["state"])
        if row["instance_id"] == "quad-reach" and key in observer_keys and row["saved_object"] != "pooled_observer":
            alternate.append(row)
        else:
            primary[key].append(row)
    output, missing = [], []
    for instance, (planned_steps, h, states, expected) in SPECS.items():
        for state in states:
            endpoints = {}
            for method in METHODS:
                rows = primary[instance,method,state]
                endpoints[method] = {r["step"]: r for r in rows if r["geometry"] == "endpoint" and r["complete_initial_set"]}
            common_steps = set.intersection(*(set(v) for v in endpoints.values()))
            common = max(common_steps) if common_steps else None
            for method in METHODS:
                rows = primary[instance,method,state]
                ends = [r for r in rows if r["geometry"] == "endpoint"]
                tubes = [r for r in rows if r["geometry"] == "tube"]
                last = max(ends,key=lambda r:r["step"]) if ends else None
                last_complete = endpoints[method].get(max(endpoints[method])) if endpoints[method] else None
                final = endpoints[method].get(planned_steps)
                common_end = endpoints[method].get(common)
                complete_tubes = [r for r in tubes if r["complete_initial_set"]]
                common_tubes = [r for r in complete_tubes if common is not None and r["step"] <= common]
                common_tube_complete = common is not None and {r["step"] for r in common_tubes} == set(range(1,common+1))
                safe_step, safe_source = SAVED_SAFE_PREFIXES.get(instance, (None, None))
                safe_end = endpoints[method].get(safe_step)
                safe_tubes = [r for r in complete_tubes if safe_step is not None and r["step"] <= safe_step]
                safe_tube_complete = safe_step is not None and {r["step"] for r in safe_tubes} == set(range(1,safe_step+1))
                reasons = []
                if not rows:
                    reasons.extend(ex.cells[instance,method]["missing_or_noncomparable"])
                if instance == "single-pendulum-reach" and state == "x3":
                    reasons.append("Official third physical state has no authoritative equation/model mapping or saved ranges; two-state submission cannot supply it.")
                if instance == "balancing-reach":
                    reasons.append("Bounds are fixed-repository raw4 supplemental contract only; paper five-input controller mapping remains unresolved; no T=10 output.")
                if not final:
                    reasons.append("No complete-initial-set saved endpoint at the requested full horizon.")
                if not tubes:
                    reasons.append("No saved per-step tube for this physical state/object; endpoint or all-time union cannot replace it.")
                if common is None:
                    reasons.append("No common four-method complete-initial-set saved endpoint time for this state.")
                if common is not None and not common_tube_complete:
                    reasons.append("Per-step tube coverage of the common window is incomplete; common maximum is left null.")
                if last and not last["complete_initial_set"]:
                    reasons.append("Own last saved endpoint pools surviving accepted lanes only; it does not cover the full initial set.")
                record = dict(instance_id=instance, method=method, state=state,
                    contract_id=rows[0]["contract_id"] if rows else "unavailable",
                    expected_lanes=expected, planned_steps=planned_steps, horizon_s=planned_steps*h,
                    coverage_status="full" if final and len(endpoints[method])==planned_steps else "terminal_only" if final else "partial" if rows else "missing",
                    endpoint_steps=len(endpoints[method]), tube_steps=len(complete_tubes),
                    common_time=common*h if common is not None else None,
                    common_endpoint_lo=common_end["lo"] if common_end else None,
                    common_endpoint_hi=common_end["hi"] if common_end else None,
                    common_endpoint_width=common_end["absolute_width"] if common_end else None,
                    common_endpoint_object=common_end["saved_object"] if common_end else None,
                    common_max_tube_width=max((r["absolute_width"] for r in common_tubes),default=None) if common_tube_complete else None,
                    common_safe_time=safe_step*h if safe_step is not None else None,
                    common_safe_endpoint_lo=safe_end["lo"] if safe_end else None,
                    common_safe_endpoint_hi=safe_end["hi"] if safe_end else None,
                    common_safe_endpoint_width=safe_end["absolute_width"] if safe_end else None,
                    common_safe_max_tube_width=max((r["absolute_width"] for r in safe_tubes),default=None) if safe_tube_complete else None,
                    common_safe_source=safe_source,
                    own_last_t=last["t_end"] if last else None,
                    own_last_endpoint_lo=last["lo"] if last else None,
                    own_last_endpoint_hi=last["hi"] if last else None,
                    own_last_endpoint_width=last["absolute_width"] if last else None,
                    own_last_available_lanes=last["available_lanes"] if last else None,
                    own_last_complete_t=last_complete["t_end"] if last_complete else None,
                    own_last_complete_endpoint_width=last_complete["absolute_width"] if last_complete else None,
                    own_max_tube_width=max((r["absolute_width"] for r in tubes),default=None),
                    own_complete_prefix_max_tube_width=max((r["absolute_width"] for r in complete_tubes),default=None),
                    full_endpoint_width=final["absolute_width"] if final else None,
                    full_max_tube_width=max((r["absolute_width"] for r in complete_tubes),default=None) if len(complete_tubes)==planned_steps else None,
                    source_ids=";".join(sorted({r["source_id"] for r in rows})),
                    missing_reason=" | ".join(dict.fromkeys(reasons)))
                output.append(record)
                if reasons:
                    missing.append({k:record[k] for k in ("instance_id","method","state","coverage_status","missing_reason")})
    return output, missing, alternate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    for name in (*hashlib.algorithms_guaranteed,"new","file_digest"):
        if hasattr(hashlib,name):setattr(hashlib,name,no_digest)
    args.output.mkdir(parents=True,exist_ok=True)
    ex=Export();collect(ex)
    keys=[tuple(r[k] for k in ("instance_id","method","saved_object","state","step","geometry")) for r in ex.rows]
    if len(keys)!=len(set(keys)):raise ValueError("duplicate saved object/step/state/geometry rows")
    summary,missing,alternate=summarize(ex)
    equivalences=equivalent_runs()
    for row in summary:
        row["equivalent_p3_receipts"]=";".join(r["comparison"] for r in equivalences
            if row["method"]=="pytorch_gpu" and r["instance_id"]==row["instance_id"])
    csv_write(args.output/"widths_long.csv",ex.rows,FIELDS)
    (args.output/"widths_long.json").write_text(json.dumps({"fields":FIELDS,"rows":[[r[k] for k in FIELDS] for r in ex.rows]},separators=(",",":"),allow_nan=False)+"\n")
    for name,rows in (("summary",summary),("missing_fields",missing),("alternate_endpoint_objects",alternate)):
        csv_write(args.output/(name+".csv"),rows)
        (args.output/(name+".json")).write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
    (args.output/"sources.json").write_text(json.dumps(ex.sources,indent=2)+"\n")
    (args.output/"equivalent_p3_runs.json").write_text(json.dumps(equivalences,indent=2)+"\n")
    instances = [dict(instance_id=k, planned_steps=v[0], step_s=v[1], physical_states=v[2],
                      expected_lanes=v[3], contract_sources=ex.cells[k,"pytorch_gpu"]["contract_paths"])
                 for k,v in SPECS.items()]
    (args.output/"instances.json").write_text(json.dumps(instances,indent=2)+"\n")
    coverage = []
    for spec in instances:
        item = dict(spec)
        item["methods"] = {}
        for method in METHODS:
            rows = [r for r in summary if r["instance_id"] == spec["instance_id"] and r["method"] == method]
            item["methods"][method] = dict(
                states={r["state"]:r["coverage_status"] for r in rows},
                common_times=sorted({r["common_time"] for r in rows if r["common_time"] is not None}),
                missing_field_reasons=list(dict.fromkeys(r["missing_reason"] for r in rows if r["missing_reason"])))
        coverage.append(item)
    (args.output/"coverage_by_instance.json").write_text(json.dumps(coverage,indent=2)+"\n")
    meta=dict(schema="archcomp26-saved-widths-v1",instances=16,method_cells=64,long_rows=len(ex.rows),summary_rows=len(summary),
              definition="absolute_width=hi-lo; pooled across saved accepted lanes at one step; max_tube_width=max of these per-step widths, never all-time union width",
              common_time="latest endpoint time saved for all four methods and the full initial set, state by state",
              common_safe_fields="Only the previously recorded DP-more step60 and TORA-remain step184 safe prefixes are transcribed; null elsewhere does not imply unsafe. No property checker was executed.",
              endpoint_choice="P3 QUAD observer endpoint is primary; driver final_hull is retained separately; Huan/Xiangru QUAD use saved driver terminal only",
              supplemental_contracts=["Balancing raw4 is supplemental to the unresolved paper five-input contract","SP supplies only the fixed two-state submission; official third state is missing"],
              excluded_records=ex.exclusions,new_numerical_runs=0,old_checkers_executed=0,digest_operations=0,
              independent_end_to_end_floating_nncs_certificate=False)
    (args.output/"AUDIT.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({k:meta[k] for k in ("instances","method_cells","long_rows","summary_rows")}))


if __name__=="__main__":
    main()
