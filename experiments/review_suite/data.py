"""Recompute the review tables from selected immutable scientific evidence."""
from __future__ import annotations

import csv
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments/review_suite/registry.yaml"
PROFILE = ROOT / "benchmarks/review/fixed_profiles.json"
PLANTS = ("van_der_pol", "brusselator")
VIEWS = ("endpoint", "tube")
COORDINATES = ("x", "y")
LANES = ("cpu", "gpu-range", "flowstar")


def load_registry():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry["schema"] != "research_review_registry/1":
        raise ValueError("unknown review registry schema")
    ids = [item["id"] for item in registry["experiments"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate review experiment ID")
    return registry


def experiment(registry, experiment_id):
    return next((item for item in registry["experiments"] if item["id"] == experiment_id), None)


def fixed_experiment(registry, plant):
    item = next((item for item in registry["experiments"] if item.get("plant") == plant), None)
    if item is None or set(item["variants"]) != set(LANES):
        raise ValueError(f"missing three-lane full-horizon registry for {plant}")
    return item


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(relative):
    if not isinstance(relative, str):
        raise ValueError("source path must be a registry string")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise FileNotFoundError(f"registered source is missing: {relative}")
    return path


def profile_check():
    """Check the review copy against real live config and frozen source bytes."""
    import torch_tm_flowpipe as core
    from experiments.endpoint_roundoff_repair.frozen import MATCHED_SHA256, setup
    if not Path(core.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError(f"imported numerical package is outside this checkout: {core.__file__}")
    profile = json.loads(PROFILE.read_text())
    if profile["schema"] != "research_review_fixed_profiles/1":
        raise ValueError("unknown review profile")
    matched = source_path(profile["shared_matched_contracts_path"])
    if sha256(matched) != profile["shared_matched_contracts_sha256"] != MATCHED_SHA256:
        raise ValueError("frozen matched-contract bytes changed")
    for plant in PLANTS:
        item = profile["plants"][plant]
        contract = source_path(item["source_contract_path"])
        if sha256(contract) != item["source_contract_sha256"]:
            raise ValueError(f"frozen execution contract changed for {plant}")
        config, current, _ = setup(plant)
        if json.loads(json.dumps(config.as_dict())) != item["config"]:
            raise ValueError(f"live numerical config differs from review profile for {plant}")
        if (.01 if plant == "van_der_pol" else .02).hex() != item["fixed_h_hex"]:
            raise ValueError(f"fixed step differs for {plant}")
        actual_box = [[float(v.lo).hex(), float(v.hi).hex()] for v in current.range_box()]
        if actual_box != item["original_outward_binary64_box_hex"]:
            raise ValueError(f"initial outward box differs for {plant}")
    return profile


def read_gzip_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    if not fields:
        raise ValueError(f"empty derived table: {path}")
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def observe_full_plant(item):
    """Use the existing strict observer on complete saved CPU/Flow* models."""
    from experiments.live_gpu_packets.compare_horizons import observe_objects
    from experiments.live_gpu_packets.verify_long_horizon import verify_long_horizon

    plant = item["plant"]
    order = 4 if plant == "van_der_pol" else 6
    step_h = .01 if plant == "van_der_pol" else .02
    variants = item["variants"]
    cpu = source_path(variants["cpu"]["raw_data_path"])
    gpu = source_path(variants["gpu-range"]["raw_data_path"])
    flowstar = source_path(variants["flowstar"]["raw_data_path"])
    gpu_run = source_path(variants["gpu-range"]["result_path"]).parent
    verified = verify_long_horizon(gpu_run, require_achieved=True)
    print(f"{plant}: verified {verified['steps']} GPU accepted steps; re-observing complete CPU/Flow* objects", flush=True)
    curves = []
    previous_after = None
    elapsed = Fraction(0)
    streams = (read_gzip_jsonl(cpu), read_gzip_jsonl(gpu), read_gzip_jsonl(flowstar))
    index = 0
    for index, triple in enumerate(zip(*streams, strict=True), 1):
        cpu_row, gpu_row, flowstar_row = triple
        if not (cpu_row["step"] == gpu_row["step"] == flowstar_row["step"]):
            raise ValueError(f"scientific step index mismatch at {plant} {index}")
        if gpu_row["step"] != index or not gpu_row["accepted"] or gpu_row["status"] != "validated":
            raise ValueError(f"unaccepted GPU record at {plant} {index}")
        if gpu_row["h_hex"] != step_h.hex() or float(flowstar_row["h"]) != step_h:
            raise ValueError(f"fixed binary64 step changed at {plant} {index}")
        elapsed += Fraction(step_h)
        if Fraction(gpu_row["exact_time"]) != elapsed or Fraction(cpu_row["t_end_exact"]) != elapsed:
            raise ValueError(f"exact scientific time mismatch at {plant} {index}")
        if previous_after is not None and gpu_row["state_before"] != previous_after:
            raise ValueError(f"GPU accepted-state continuity mismatch at {plant} {index}")
        previous_after = gpu_row["state_after"]
        observed = {
            "cpu": observe_objects(cpu_row["models"], order),
            "flowstar": observe_objects(flowstar_row["models"], order),
            "gpu-range": {view: [list(map(float.fromhex, pair))
                                 for pair in gpu_row["bounds"][view]] for view in VIEWS},
        }
        for lane in LANES:
            for view in VIEWS:
                for coordinate, (lo, hi) in zip(COORDINATES, observed[lane][view]):
                    lo, hi = float(lo), float(hi)
                    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                        raise ValueError(f"nonfinite or inverted {plant} {index} {lane} range")
                    curves.append(dict(plant=plant, step=index, exact_time=str(elapsed),
                                       time=float(elapsed), lane=lane, view=view,
                                       coordinate=coordinate, lo_hex=lo.hex(), hi_hex=hi.hex(),
                                       width=hi-lo))
        if index % 100 == 0:
            print(f"{plant}: re-observed {index} / 1000 steps", flush=True)
    if index != 1000 or verified["steps"] != index:
        raise ValueError(f"full {plant} horizon is incomplete")
    return curves, dict(plant=plant, accepted_steps=index, final_exact_time=str(elapsed),
                        gpu_verification=verified, raw_sha256={
                            lane: sha256(source_path(variants[lane]["raw_data_path"])) for lane in LANES},
                        code_refs={lane: variants[lane]["code_ref"] for lane in LANES},
                        source_paths={lane: variants[lane]["raw_data_path"] for lane in LANES})


def ratio_rows(curves):
    by_key = {(r["plant"], r["step"], r["view"], r["coordinate"], r["lane"]): r for r in curves}
    output = []
    for plant in PLANTS:
        for step in range(1, 1001):
            for view in VIEWS:
                for coordinate in COORDINATES:
                    rows = {lane: by_key[(plant, step, view, coordinate, lane)] for lane in LANES}
                    for numerator, denominator in (("gpu-range", "cpu"), ("gpu-range", "flowstar"),
                                                   ("cpu", "flowstar")):
                        n, d = rows[numerator], rows[denominator]
                        zero = d["width"] <= 1e-10
                        output.append(dict(plant=plant, step=step, exact_time=n["exact_time"],
                                           time=n["time"], view=view, coordinate=coordinate,
                                           numerator=numerator, denominator=denominator,
                                           numerator_width=n["width"], denominator_width=d["width"],
                                           width_ratio="" if zero else n["width"]/d["width"],
                                           near_zero_denominator=zero,
                                           lower_abs_diff=abs(float.fromhex(n["lo_hex"])-float.fromhex(d["lo_hex"])),
                                           upper_abs_diff=abs(float.fromhex(n["hi_hex"])-float.fromhex(d["hi_hex"]))))
    return output


def width_summary(curves, registry):
    keys = sorted({(r["plant"], r["lane"], r["view"], r["coordinate"]) for r in curves})
    summary = []
    for plant, lane, view, coordinate in keys:
        chosen = [r for r in curves if (r["plant"], r["lane"], r["view"], r["coordinate"])
                  == (plant, lane, view, coordinate)]
        widths = [r["width"] for r in chosen]
        item = fixed_experiment(registry, plant)
        summary.append(dict(plant=plant, lane=lane, view=view, coordinate=coordinate,
                            samples=len(chosen), accepted_steps=len(chosen),
                            final_exact_time=chosen[-1]["exact_time"],
                            final_width=chosen[-1]["width"],
                            width_median=statistics.median(widths), width_max=max(widths),
                            source_status=item["variants"][lane]["fresh_or_reused"]))
    return summary


def auxiliary_tables(out):
    base = ROOT / "artifacts/runs"
    c3 = json.loads((base / "vdp_c3_cross_step_causal_closure_20260827/RESULT.json").read_text())
    c4 = json.loads((base / "brusselator_sr1000_c4_closure_20260828/CLOSURE_RESULT.json").read_text())
    c4_full = json.loads((base / "brusselator_live_range_c5_20260828/RESULT.json").read_text())
    rows = []
    for horizon, values in c3["fixed"].items():
        for lane in ("c2", "c3", "flowstar"):
            value = values[lane]
            rows.append(dict(plant="van_der_pol", mechanism="cross-step history", horizon=horizon,
                             lane=lane, accepted=value["accepted"], completed=value["completed"],
                             endpoint_x_width=value["widths"]["endpoint_x"],
                             endpoint_y_width=value["widths"]["endpoint_y"],
                             source="artifacts/runs/vdp_c3_cross_step_causal_closure_20260827/RESULT.json"))
    for lane, value in c3["native"].items():
        rows.append(dict(plant="van_der_pol", mechanism="cross-step history",
                         horizon="native adaptive T10", lane=lane,
                         accepted=value["accepted"],
                         completed=value.get("completed_requested_horizon", value["status"] == "completed"),
                         endpoint_x_width="", endpoint_y_width="",
                         source="artifacts/runs/vdp_c3_cross_step_causal_closure_20260827/RESULT.json"))
    rows.extend([
        dict(plant="brusselator", mechanism="post-accept refinement", horizon="historical SR1000",
             lane="legacy", accepted=c4["torch_accepted_steps"], completed=False,
             endpoint_x_width="", endpoint_y_width="",
             source="artifacts/runs/brusselator_sr1000_c4_closure_20260828/CLOSURE_RESULT.json"),
        dict(plant="brusselator", mechanism="post-accept refinement", horizon="historical SR1000",
             lane="stock Flow*", accepted=c4["stock_accepted_steps"], completed=True,
             endpoint_x_width="", endpoint_y_width="",
             source="artifacts/runs/brusselator_sr1000_c4_closure_20260828/CLOSURE_RESULT.json"),
        dict(plant="brusselator", mechanism="post-accept refinement", horizon="historical SR1000",
             lane="C4 refined", accepted=c4_full["c4_accepted_steps"], completed=c4_full["c4_reaches_T20"],
             endpoint_x_width="", endpoint_y_width="",
             source="artifacts/runs/brusselator_live_range_c5_20260828/RESULT.json"),
    ])
    write_csv(out / "mechanisms.csv", rows)
    prepared = json.loads((base / "repaired_solver_performance_20260908T034636Z/RESULT.json").read_text())
    ordered = json.loads((base / "boundary_execution_20260908T172756Z/RESULT.json").read_text())
    pairings = []
    for plant in ("vdp", "brusselator"):
        p = prepared["full_pairs"][plant]
        o = ordered["full_pairs"]["van_der_pol" if plant == "vdp" else plant][0]
        pairings.extend([
            dict(plant=plant, optimization="prepared remainder replay", baseline_s=p["reference_solve_seconds"],
                 candidate_s=p["optimized_solve_seconds"], speedup=p["speedup"],
                 accepted_steps=p["accepted_steps"], bitwise_equal=p["all_steps_bitwise_equal"],
                 samples=1, source="artifacts/runs/repaired_solver_performance_20260908T034636Z/RESULT.json"),
            dict(plant=plant, optimization="ordered tensor range after prepared replay",
                 baseline_s=o["baseline_solve_seconds"], candidate_s=o["candidate_solve_seconds"],
                 speedup=o["speedup"], accepted_steps=o["accepted_steps"],
                 bitwise_equal=o["all_steps_bitwise_equal"], samples=1,
                 source="artifacts/runs/boundary_execution_20260908T172756Z/RESULT.json"),
        ])
    write_csv(out / "cpu_pairings.csv", pairings)
    gpu = json.loads((base / "live_gpu_packets_20260910T023603Z/RESULT.json").read_text())
    batch = json.loads((base / "range_batch_device_20260909T030609Z/RESULT.json").read_text())
    live = json.loads((base / "live_range_solver_20260909T053007Z/RESULT.json").read_text())
    resident = json.loads((base / "resident_tm_block_20260914T032650Z/RESULT.json").read_text())
    route = []
    for plant in ("van_der_pol", "brusselator"):
        g = gpu["performance_decisions"][plant]
        r = resident["end_to_end"][plant]
        route.extend([
            dict(plant=plant, layer="offline range batch including pack and transfer",
                 local_speedup=statistics.median(batch["local_throughput"]["gates"][plant]["full_roundtrip_ratios"]),
                 online_speedup="", online_wall_s="", samples=5, target_met=True,
                 source="artifacts/runs/range_batch_device_20260909T030609Z/RESULT.json"),
            dict(plant=plant, layer="live range service B32x20 prefix",
                 local_speedup="", online_speedup=live["performance"]["decisions"][plant]["median"],
                 online_wall_s="", samples=3, target_met=True,
                 source="artifacts/runs/live_range_solver_20260909T053007Z/RESULT.json"),
            dict(plant=plant, layer="packet online B32x20", local_speedup="", online_speedup=g["G0_over_Gp_median"],
                 online_wall_s="", samples=5, target_met=g["packet_target_met"],
                 source="artifacts/runs/live_gpu_packets_20260910T023603Z/RESULT.json"),
            dict(plant=plant, layer="resident composition B32x20",
                 local_speedup=resident["local_block_speedups"][plant+"_cpu_over_resident_median"],
                 online_speedup=r["paired_Gp_over_Gr_median"], online_wall_s=r["Gr_wall_median_s"],
                 samples=5, target_met=r["target_met"],
                 source="artifacts/runs/resident_tm_block_20260914T032650Z/RESULT.json"),
        ])
    write_csv(out / "gpu_route.csv", route)
    # Keep the one Flow* invocation per plant distinct from five paired GPU observations.
    with (base / "resident_tm_block_20260914T032650Z/matched_flowstar.csv").open(newline="") as handle:
        matched = list(csv.DictReader(handle))
    if not matched:
        raise ValueError("empty matched Flow* source")
    write_csv(out / "matched_flowstar.csv", matched)
    return {"mechanisms.csv":len(rows), "cpu_pairings.csv":len(pairings),
            "gpu_route.csv":len(route), "matched_flowstar.csv":len(matched)}


def generate_index(registry, path):
    lines = ["# Experiment reading map", "",
             "Generated from [the single registry](../experiments/review_suite/registry.yaml) by review-suite summarize.",
             "Each entry follows question → frozen configuration/code → original evidence → derived table/figure → limited claim.",
             ""]
    for item in registry["experiments"]:
        lines.extend([f"## {item['plain_title']}", "",
                      f"**Question.** {item['research_question']}", "",
                      f"**Scope and role.** {item['observed_scope']} {item['implementation_role']}.",
                      "", f"**Configuration.** [{item['config_path']}](../{item['config_path']})",
                      "", f"**Original evidence.** {item['original_path']}",
                      "", f"**Raw data.** {item['raw_data_path']}",
                      "", f"**Derived table.** [{item['table_path']}](../{item['table_path']})",
                      "", f"**Figures.** {', '.join(item['figure_paths']) if item['figure_paths'] else 'None needed.'}",
                      "", f"**Reproduction level.** {item['reproduction_level']}.",
                      "", f"**Observed conclusion.** {item['main_conclusion']}",
                      "", f"**Run.** {item['run_command']}",
                      "", f"**Recompute.** {item['summarize_command']}",
                      "", f"**Numerical limit.** {item['numerical_limitations']}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(out):
    registry = load_registry()
    profile = profile_check()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    curves, sources = [], []
    for plant in PLANTS:
        item = fixed_experiment(registry, plant)
        rows, source = observe_full_plant(item)
        curves.extend(rows)
        sources.append(source)
    if len(curves) != 24000:
        raise ValueError(f"expected 24,000 complete four-channel curve rows, got {len(curves)}")
    ratios = ratio_rows(curves)
    summary = width_summary(curves, registry)
    write_csv(out / "bounds/curves.csv.gz", curves)
    write_csv(out / "bounds/width_ratios.csv.gz", ratios)
    write_csv(out / "summary.csv", summary)
    auxiliary = auxiliary_tables(out)
    provenance = dict(schema="research_review_provenance/1", registry_sha256=sha256(REGISTRY),
                      profile_sha256=sha256(PROFILE), generated_from="selected complete immutable registered run records",
                      observer="current strict corrected CPU range re-executed on saved CPU and Flow* complete models",
                      gpu_bounds="from GPU run's each accepted step, without CPU correction",
                      sources=sources, curve_rows=len(curves), ratio_rows=len(ratios),
                      auxiliary_rows=auxiliary,
                      numerical_source_parent=registry["parent_commit"],
                      whole_solver_formal_proof=False, resident_full_1000_steps=False)
    write_json(out / "provenance.json", provenance)
    generate_index(registry, ROOT / "docs/EXPERIMENTS.md")
    print(json.dumps({"curve_rows":len(curves),"ratio_rows":len(ratios),
                      "summary_rows":len(summary),"auxiliary":auxiliary},indent=2), flush=True)
    return provenance
