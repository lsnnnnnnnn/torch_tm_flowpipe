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
    if not (sha256(matched) == profile["shared_matched_contracts_sha256"] == MATCHED_SHA256):
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
                if len(observed[lane][view]) != len(COORDINATES):
                    raise ValueError(f"wrong {plant} {index} {lane} {view} channel count")
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
    adaptive_source = "artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/vdp_adaptive/summary.json"
    adaptive = json.loads(source_path(adaptive_source).read_text())
    if not adaptive["completed"] or adaptive["failure"] is not None:
        raise ValueError("registered saved VDP adaptive T10 is incomplete")
    adaptive_rows = [dict(plant="van_der_pol", schedule="native adaptive T10",
                          accepted_steps=adaptive["accepted_steps"],
                          rejected_attempts=adaptive["rejected_attempts"],
                          accepted_horizon_exact=adaptive["accepted_horizon_exact"],
                          solve_seconds=adaptive["solve_seconds"],
                          export_seconds=adaptive["export_seconds"],
                          source_status="reused_recomputed", source=adaptive_source)]
    write_csv(out / "adaptive.csv", adaptive_rows)
    before_source = "artifacts/runs/endpoint_roundoff_repair_20260908/before_endpoint_witness.json"
    after_source = "artifacts/runs/endpoint_roundoff_repair_20260908/after_endpoint_oracles.json"
    before = json.loads(source_path(before_source).read_text())
    after = json.loads(source_path(after_source).read_text())
    if before["exact_value"] != "3/144115188075855872":
        raise ValueError("saved exact endpoint witness changed")
    after_by_entry = {r["entry"]:r for r in after["original_witnesses"]}
    witnesses = []
    for old in before["rows"]:
        new = after_by_entry[old["entry"]]
        if new["exact"] != before["exact_value"]:
            raise ValueError("endpoint witness exact value changed")
        if all(old["contains"]) or not all(new["contains"]):
            raise ValueError("versioned endpoint before/after containment changed")
        witnesses.append(dict(witness="binary64 endpoint time substitution", entry=old["entry"],
                              exact_value=new["exact"],
                              old_local_result="excluded" if not all(old["contains"]) else "contained",
                              corrected_local_result="contained" if all(new["contains"]) else "excluded",
                              scope="two local endpoint entries, not whole-solver proof",
                              source=f"{before_source}; {after_source}"))
    from torch_tm_flowpipe import Interval
    from torch_tm_flowpipe.range_requests import certified_scalar_power
    x = float.fromhex("0x1.7d3ecfa658d9bp+9")
    exact = Fraction(x)**3
    old_power = Interval(x).pow_int(3)
    lower, upper, changed = certified_scalar_power(x, x, 3)
    old_contains = Fraction(float(old_power.lo)) <= exact <= Fraction(float(old_power.hi))
    new_contains = changed and Fraction(lower) <= exact <= Fraction(upper)
    if old_contains or not new_contains:
        raise ValueError("versioned strict power witness changed")
    witnesses.append(dict(witness="binary64 integer power", entry="strict range request power boundary",
                          exact_value=str(exact),
                          old_local_result="contained" if old_contains else "excluded",
                          corrected_local_result="contained" if new_contains else "excluded",
                          scope="certified_scalar_power entry; legacy Interval.pow_int remains",
                          source="tests/test_range_requests.py; src/torch_tm_flowpipe/range_requests.py"))
    retained_error = Fraction(1.1)**2 - Fraction(float(1.1*1.1))
    if retained_error == 0:
        raise ValueError("retained coefficient binary64 witness vanished")
    witnesses.append(dict(witness="retained point coefficient product", entry="normal composition block",
                          exact_value=str(retained_error), old_local_result="legacy sparse block excludes",
                          corrected_local_result="resident local exact-oracle test encloses",
                          scope="optional default-off resident block only",
                          source="tests/test_resident_tm_block.py; docs/resident_tm_block/NUMERICAL_CONTRACT.md"))
    external_source = "artifacts/runs/backend_reevaluation_20260907T061907Z/LOCAL_CONTAINMENT_RESULT.json"
    external = json.loads(source_path(external_source).read_text())
    witnesses.append(dict(witness="historical matrix and nonunit-scale history", entry="versioned strict local calls",
                          exact_value="exact rational oracles in fixed re-evaluation",
                          old_local_result=external["A_original_strict"],
                          corrected_local_result=external["B_old_strict"],
                          scope=external["proof_scope"], source=external_source))
    carry_source = "artifacts/runs/endpoint_roundoff_repair_20260908/two_step_carry_checks.json"
    carry = json.loads(source_path(carry_source).read_text())
    if not all(p["two_steps_passed"] and p["first_error_consumed_by_second"]
               for p in carry["paths"][:2]):
        raise ValueError("saved endpoint carry witness failed")
    witnesses.append(dict(witness="accepted next-step endpoint error ownership", entry="published and dense internal",
                          exact_value=carry["paths"][0]["exact_value"], old_local_result="not a new old/new exclusion",
                          corrected_local_result="two accepted steps consume first error",
                          scope="next-step local paths; rejected attempt does not commit",
                          source=carry_source))
    external_map = json.loads(source_path(
        "artifacts/runs/backend_reevaluation_20260907T061907Z/SOURCE_MAP.json").read_text())
    versions = {
        "binary64 endpoint time substitution": (before["source_sha"], after["source_sha"]),
        "binary64 integer power": ("legacy Interval.pow_int at current parent",
                                   "strict range-request entry at current parent"),
        "retained point coefficient product": ("legacy CPU sparse block at current parent",
                                                "resident block 8953b5ea24e69d11ac5c2890a4cdb663305e9e79"),
        "historical matrix and nonunit-scale history": (
            external_map["implementations"]["A"]["sha"],
            external_map["implementations"]["B"]["sha"]),
        "accepted next-step endpoint error ownership": ("not an old/new exclusion",
                                                        after["source_sha"]),
    }
    for row in witnesses:
        row["old_version"], row["corrected_version"] = versions[row["witness"]]
    write_csv(out / "numerical_witnesses.csv", witnesses)
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
    return {"adaptive.csv":len(adaptive_rows), "numerical_witnesses.csv":len(witnesses),
            "mechanisms.csv":len(rows), "cpu_pairings.csv":len(pairings),
            "gpu_route.csv":len(route), "matched_flowstar.csv":len(matched)}


def fresh_confirmation_table(registry, out):
    """Verify new fixed-source witnesses without replacing the saved main curves."""
    from experiments.live_gpu_packets.verify_long_horizon import verify_long_horizon
    from experiments.live_range_solver.verify import load_run, verify_run

    rows = []
    for item in registry["experiments"]:
        for confirmation in item.get("confirmations", []):
            backend = confirmation["backend"]
            raw = source_path(confirmation["raw_data_path"])
            summary_file = source_path(confirmation["summary_path"])
            summary = json.loads(summary_file.read_text())
            plant = item["plant"] if "plant" in item else "van_der_pol"
            saved = (item["variants"][backend] if "variants" in item else None)
            if backend == "cpu":
                if (not summary["completed"] or summary["failure"] is not None
                        or summary["scientific_sha"] != confirmation["code_ref"]
                        or not summary["source_clean_at_end"]):
                    raise ValueError(f"fresh CPU confirmation failed: {item['id']}")
                with source_path(confirmation["bounds_path"]).open(newline="") as stream:
                    bounds = list(csv.DictReader(stream))
                count = summary["accepted_steps"]
                if count != len(bounds) or count != sum(1 for _ in read_gzip_jsonl(raw)):
                    raise ValueError(f"fresh CPU coverage mismatch: {item['id']}")
                for index, record in enumerate(bounds, 1):
                    if int(record["step"]) != index:
                        raise ValueError(f"fresh CPU step mismatch: {item['id']} {index}")
                    for metric in ("common", "published"):
                        for view in VIEWS:
                            for coordinate in COORDINATES:
                                lo = float(record[f"{metric}_{view}_{coordinate}_lo"])
                                hi = float(record[f"{metric}_{view}_{coordinate}_hi"])
                                if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                                    raise ValueError(f"fresh CPU range invalid: {item['id']} {index} {metric} {view} {coordinate}")
                old_dir = (source_path(saved["bounds_path"]).parent if saved is not None
                           else source_path(item["raw_data_path"]).parent)
                old_bounds = old_dir / "bounds.csv"
                old_models = old_dir / "models.jsonl.gz"
                bounds_equal = source_path(confirmation["bounds_path"]).read_bytes() == old_bounds.read_bytes()
                with old_bounds.open(newline="") as stream:
                    saved_bounds = list(csv.DictReader(stream))
                if len(saved_bounds) != count:
                    raise ValueError(f"saved CPU comparison coverage changed: {item['id']}")
                equal_channels = sum(
                    all(new[f"common_{view}_{coordinate}_{side}"] == old[f"common_{view}_{coordinate}_{side}"]
                        for side in ("lo", "hi"))
                    for new, old in zip(bounds, saved_bounds, strict=True)
                    for view in VIEWS for coordinate in COORDINATES)
                with gzip.open(raw,"rb") as new_stream, gzip.open(old_models,"rb") as old_stream:
                    models_equal = new_stream.read() == old_stream.read()
                rows.append(dict(experiment=item["id"],plant=plant,backend=backend,
                                 source_sha=confirmation["code_ref"],scope=confirmation["observed_scope"],
                                 fresh_or_reused="fresh_run",complete=True,accepted_steps=count,
                                 rejected_attempts=summary["rejected_attempts"],
                                 exact_final_time=summary["accepted_horizon_exact"],
                                 four_channel_rows=4*count,
                                 saved_four_channel_equal_count=equal_channels,
                                 saved_bounds_csv_byte_equal=bounds_equal,
                                 saved_complete_models_equal=models_equal,
                                 saved_final_state_equal="not compared: complete CPU models checked",
                                 timing_boundary="solve",elapsed_s=summary["solve_seconds"],
                                 verification="source-clean, all rows and all complete model objects",
                                 raw_data_path=confirmation["raw_data_path"],
                                 summary_path=confirmation["summary_path"],
                                 raw_sha256=sha256(raw)))
            elif backend == "gpu-range":
                if (not summary["achieved"] or summary["source_sha"] != confirmation["code_ref"]
                        or summary["route"] != "Gp" or summary["plant"] != plant):
                    raise ValueError(f"fresh GPU confirmation failed: {item['id']}")
                verified = verify_long_horizon(summary_file.parent, require_achieved=True)
                old_raw = source_path(saved["raw_data_path"])
                equal = count = 0
                for index, (new, old) in enumerate(zip(read_gzip_jsonl(raw),
                                                       read_gzip_jsonl(old_raw), strict=True), 1):
                    if new["step"] != old["step"] or new["step"] != index:
                        raise ValueError(f"fresh GPU step mismatch: {item['id']} {index}")
                    for view in VIEWS:
                        if len(new["bounds"][view]) != len(COORDINATES):
                            raise ValueError(f"fresh GPU channel count invalid: {item['id']} {index} {view}")
                        for coordinate_index in range(2):
                            lo, hi = map(float.fromhex, new["bounds"][view][coordinate_index])
                            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                                raise ValueError(f"fresh GPU range invalid: {item['id']} {index} {view} {coordinate_index}")
                            count += 1
                            equal += new["bounds"][view][coordinate_index] == old["bounds"][view][coordinate_index]
                if verified["steps"] != 1000 or count != 4000:
                    raise ValueError(f"fresh GPU four-channel coverage failed: {item['id']}")
                metadata = json.loads(source_path(confirmation["metadata_path"]).read_text())
                saved_result = json.loads(source_path(saved["result_path"]).read_text())
                rows.append(dict(experiment=item["id"],plant=plant,backend=backend,
                                 source_sha=confirmation["code_ref"],scope=confirmation["observed_scope"],
                                 fresh_or_reused="fresh_run",complete=True,accepted_steps=verified["steps"],
                                 rejected_attempts="",exact_final_time=verified["exact_final_time"],
                                 four_channel_rows=count,saved_four_channel_equal_count=equal,
                                 saved_bounds_csv_byte_equal="not applicable: GPU JSONL bounds",
                                 saved_complete_models_equal="not compared: GPU accepted-state records",
                                 saved_final_state_equal=summary["final_state_sha256"] == saved_result["final_state_sha256"],
                                 timing_boundary="invocation",elapsed_s=metadata["invocation_s"],
                                 verification="accepted-state continuity and all four bound channels",
                                 raw_data_path=confirmation["raw_data_path"],
                                 summary_path=confirmation["summary_path"],raw_sha256=sha256(raw)))
            else:
                raise ValueError(f"unsupported fresh confirmation backend: {backend}")
        if item["id"] in ("vdp-resident-prefix", "brusselator-resident-prefix",
                          "vdp-resident-b2", "brusselator-resident-b2") and item["fresh_or_reused"] == "fresh_run":
            directory = (ROOT / item["original_path"]).resolve()
            if not directory.is_relative_to(ROOT) or not directory.is_dir():
                raise FileNotFoundError(f"fresh resident directory missing: {item['id']}")
            summary_file = source_path(f"{item['original_path']}/summary.json")
            summary = json.loads(summary_file.read_text())
            verified = verify_run(*load_run(directory))
            original = item["id"].endswith("prefix")
            expected_steps, expected_tasks = (20,1) if original else (4,2)
            if (summary["source_sha"] != item["code_ref"] or summary["route"] != "Gr"
                    or summary["resident_tm_block"] is not True
                    or summary["original"] is not original
                    or verified["accepted_lane_steps"] != expected_steps
                    or verified["successful_tasks"] != expected_tasks
                    or summary["counts"]["resident_requests"] < expected_steps):
                raise ValueError(f"fresh resident scope/coverage failed: {item['id']}")
            plant = "van_der_pol" if item["id"].startswith("vdp") else "brusselator"
            rows.append(dict(experiment=item["id"],plant=plant,backend="resident",
                             source_sha=item["code_ref"],scope=item["observed_scope"],
                             fresh_or_reused="fresh_run",complete=True,accepted_steps=expected_steps,
                             rejected_attempts="",exact_final_time="prefix or partition steps only",
                             four_channel_rows="not saved: optional small interface witness",
                             saved_four_channel_equal_count="",saved_bounds_csv_byte_equal="not saved",
                             saved_complete_models_equal="not compared",
                             saved_final_state_equal="not compared: different prototype/version",
                             timing_boundary="online run wall",elapsed_s=summary["wall_s"],
                             verification="resident CUDA requests and full lifecycle semantic verifier",
                             raw_data_path=f"{item['original_path']}/run.json.gz",
                             summary_path=f"{item['original_path']}/summary.json",
                             raw_sha256=sha256(directory / "run.json.gz")))
    write_csv(out / "fresh_confirmations.csv", rows)
    return len(rows)


def generate_index(registry, path):
    def links(value):
        values = value if isinstance(value, list) else [value]
        return ", ".join(f"[{relative}](../{relative})" for relative in values)

    lines = ["# Experiment reading map", "",
             "Generated from [the single registry](../experiments/review_suite/registry.yaml) by review-suite summarize.",
             "Each entry follows question → frozen configuration/code → original evidence → derived table/figure → limited claim.",
             ""]
    for item in registry["experiments"]:
        lines.extend([f"## {item['plain_title']}", "",
                      f"**Question.** {item['research_question']}", "",
                      f"**Scope and role.** {item['observed_scope']} {item['implementation_role']}.",
                      "", f"**Configuration.** {links(item['config_path'])}",
                      "", f"**Source revision.** `{item['code_ref']}`",
                      "", f"**Original evidence.** {links(item['original_path'])}",
                      "", f"**Raw data.** {links(item['raw_data_path'])}",
                      "", f"**Derived table.** {links(item['table_path'])}",
                      "", f"**Figures.** {links(item['figure_paths']) if item['figure_paths'] else 'None needed.'}",
                      "", f"**Reproduction level.** {item['reproduction_level']}.",
                      "", f"**Observed conclusion.** {item['main_conclusion']}",
                      "", f"**Run.** `{item['run_command']}`",
                      "", f"**Recompute.** `{item['summarize_command']}`",
                      "", f"**Numerical limit.** {item['numerical_limitations']}", ""])
        if item.get("variants"):
            lines.extend(["**Compared source revisions.** "+ "; ".join(
                f"{lane}: `{entry['code_ref']}` ({entry['implementation_role']})"
                for lane, entry in item["variants"].items()), ""])
        if item.get("historical_source"):
            lines.extend([f"**Historical source.** {links(item['historical_source'])}", ""])
        if item.get("confirmations"):
            lines.extend(["**Fresh confirmations.** "+ "; ".join(
                f"{c['backend']} at `{c['code_ref']}`: {links(c['summary_path'])}"
                for c in item["confirmations"]), ""])
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
    auxiliary["fresh_confirmations.csv"] = fresh_confirmation_table(registry, out)
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
