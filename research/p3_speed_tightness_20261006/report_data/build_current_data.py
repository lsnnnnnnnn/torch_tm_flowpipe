#!/usr/bin/env python3
"""Derive a new report data overlay from saved Oct 6 results; no numerical run.

Selections are explicit. Old 64 timing cells and all width rows are retained,
then only selected P3 cells are replaced. A numerical width candidate always
uses its new saved geometry, never an old equivalent-output width receipt.
"""

import argparse
import copy
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/evidence/results/archcomp26_report_20261005"
REPORT = ROOT / "docs/evidence/results/archcomp26_report_20261006"
ALIASES = {
    "attitude": "attitude-control-avoid", "attitude-corrected-unsafe": "attitude-control-avoid",
    "tora-tanh": "tora-reach-tanh", "tora-sigmoid": "tora-reach-sigmoid",
    "single-pendulum": "single-pendulum-reach", "docking": "docking-constraint",
    "quad": "quad-reach", "quad-paper": "quad-reach", "nav": "nav-robust",
    "quad-paper-1024": "quad-reach", "nav-author-robust": "nav-robust",
}
LAYERS = ("process_wall_s", "wrapper_wall_s", "payload_wall_s", "driver_elapsed_s",
          "driver_call_wall_s", "native_process_wall_s", "server_startup_s", "watchdog_process_wall_s")


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(v, (dict, list)) else v for k, v in row.items()})


def local_saved_contract(folder, instance):
    """Fill older NAV/QUAD comparison configuration omissions from saved files."""
    if instance not in ("nav-robust", "quad-reach"):
        return None
    if instance == "nav-robust":
        old = ROOT / "docs/evidence/results/archcomp26_20261001/nav_author_robust_working_p3_full30_001"
    else:
        start = folder / "candidate/START.json"
        mode = read(start).get("mode") if start.is_file() else None
        if mode not in ("batch2", "full50"):
            return None
        old = ROOT / "docs/evidence/results/archcomp26_20261001/quad_paper_p3_nohash_v1" / (mode + "_001/data")
    new = folder / "candidate/data"
    paths = [old / "config.yaml", new / "config.yaml", old / "START.json", new / "START.json"]
    if not all(p.is_file() for p in paths):
        return None
    starts = [read(p) for p in paths[2:]]
    if any("driver_argv" not in start for start in starts):
        return None
    previous, current = (start["driver_argv"] for start in starts)
    old_dir, new_dir = (str(Path(a[1]).parent) for a in (previous, current))
    result = {"config_bytes_equal": paths[0].read_bytes() == paths[1].read_bytes(),
              "driver_argv_equal_after_output_path_relocation": previous == [v.replace(new_dir, old_dir) for v in current],
              "paths": [relative(p) for p in paths],
              "initial_set_location": "config" if instance == "quad-reach" else "initial_box_ledger",
              "scope": "saved configuration and argv only; original comparison separately covers boxes and ranges"}
    if instance == "quad-reach":
        fields = ("source_config", "engine", "driver", "model", "method", "mode")
        result["source_and_method_fields_equal"] = {key: starts[0].get(key) == starts[1].get(key) for key in fields}
        if not all(result["source_and_method_fields_equal"].values()):
            raise ValueError("QUAD benchmark source/model/method route differs")
    if not result["config_bytes_equal"] or not result["driver_argv_equal_after_output_path_relocation"]:
        raise ValueError("new saved configuration/driver arguments differ")
    return result


def new_runs(instances, definitions):
    rows, pending = [], []
    for folder in sorted((STUDY / "results").iterdir()):
        if not folder.is_dir() or folder.name == "posthoc":
            continue
        files = {key: folder / name for key, name in {
            "outer": "RESULT.json", "outer_start": "START.json",
            "wrapper": "candidate/RESULT.json", "start": "candidate/START.json",
            "payload": "candidate/data/RESULT.json", "metrics": "candidate/data/metrics.json",
        }.items()}
        if not files["outer"].is_file() or not files["wrapper"].is_file():
            pending.append(relative(folder))
            continue
        data = {k: read(p) if p.is_file() else {} for k, p in files.items()}
        raw_instance = data["start"].get("instance", data["outer_start"].get("instance"))
        instance = ALIASES.get(raw_instance, raw_instance)
        wrapped, payload, metrics = data["wrapper"], data["payload"], data["metrics"]
        steps = wrapped.get("completed_substeps", payload.get("completed_substeps", payload.get("observed_substeps")))
        boxes = wrapped.get("B", metrics.get("B", data["start"].get("initial_boxes")))
        equivalent = folder / "candidate/SAVED_COMPARISON.json"
        width_receipt = folder / "candidate/WIDTH_COMPARISON.json"
        comparison = read(equivalent) if equivalent.is_file() else {}
        width_comparison = read(width_receipt) if width_receipt.is_file() else {}
        # The inherited QUAD width runner keeps its historical SAVED_COMPARISON
        # filename. Classify the receipt by its actual fields, never its name.
        if "width_rows" in comparison and "counts_by_state_geometry" in comparison:
            width_receipt, width_comparison = equivalent, comparison
        saved_equivalence_receipt = equivalent.is_file() and not ("width_rows" in comparison)
        local_contract = local_saved_contract(folder, instance)
        config_equal = (comparison.get("config_and_driver_argv_equal_after_path_relocation") is True
                        or (comparison.get("config_bytes_equal") is True
                            and comparison.get("driver_argv_equal_after_output_path_relocation") is True)
                        or (comparison.get("initial_box_ledger_and_config_byte_equal") is True
                            and comparison.get("driver_argv_equal_after_output_path_relocation") is True)
                        or (local_contract is not None and (comparison.get("initial_box_ledger_byte_equal") is True
                                                           or local_contract["initial_set_location"] == "config")))
        declared_numerical_change = (width_comparison.get("contract_equal_except_cutoff_and_artifact_path") is True
                                    or width_comparison.get("contract_equal_except_declared_parameters_and_artifact_path") is True
                                    or width_comparison.get("contract_equal_except_method_parameters_and_artifact_paths") is True)
        status = wrapped.get("status")
        qualified = (data["outer"].get("exit_code") == 0 and wrapped.get("exit_code") == 0
                     and isinstance(status, str) and status.startswith("COMPLETED"))
        full = (qualified and instance in instances and steps == instances[instance]["planned_steps"]
                and boxes == instances[instance]["expected_lanes"])
        property_record = wrapped.get("property_record", width_comparison.get("property_record"))
        property_source = ({"path": relative(files["wrapper"] if "property_record" in wrapped else width_receipt),
                            "field": "property_record"} if property_record is not None else None)
        if instance == "quad-reach" and "endpoint_target_inside" in payload:
            property_record = {"payload_endpoint_target_inside": payload["endpoint_target_inside"],
                "basis": "original payload endpoint field only; no full-time reach-remain checker or independent NNCS certificate inferred"}
            property_source = {"path": relative(files["payload"]), "field": "endpoint_target_inside"}
        timings = dict.fromkeys(LAYERS)
        timings.update(process_wall_s=data["outer"].get("wall_s"), wrapper_wall_s=wrapped.get("wall_s"),
                       payload_wall_s=payload.get("wall_s"),
                       driver_elapsed_s=wrapped.get("driver_elapsed_s", metrics.get("elapsed_s")))
        sources = dict.fromkeys(LAYERS)
        for field, key, raw_field in (("process_wall_s", "outer", "wall_s"),
                                     ("wrapper_wall_s", "wrapper", "wall_s"),
                                     ("payload_wall_s", "payload", "wall_s"),
                                     ("driver_elapsed_s", "wrapper" if "driver_elapsed_s" in wrapped else "metrics",
                                      "driver_elapsed_s" if "driver_elapsed_s" in wrapped else "elapsed_s")):
            if timings[field] is not None:
                sources[field] = {"path": relative(files[key]), "field": raw_field,
                                  "boundary_note": definitions[field]}
        resources = {
            "cpu_affinity": data["start"].get("cpu_affinity"),
            "physical_gpu": data["start"].get("gpu_physical", data["start"].get("physical_gpu",
                data["outer_start"].get("selected_environment", {}).get("CUDA_VISIBLE_DEVICES"))),
            "host": data["outer_start"].get("host"),
        }
        row = {
            "run_id": folder.name, "run_key": "20261006/" + folder.name,
            "stage": "full_horizon_candidate" if full else "prefix_candidate" if steps else "setup_or_pre_step_failure",
            "benchmark": instance, "method": "pytorch_gpu", "generation": "20261006 new candidate",
            "sample_count": 1, "status": status, "outer_status": data["outer"].get("status"),
            "error_type": wrapped.get("error_type"), "error": wrapped.get("error"),
            "payload_numerical_status": payload.get("status"),
            "payload_requested_substeps": payload.get("expected_substeps"),
            "candidate_outcome_classification": ("completed_full_named_horizon" if full else
                "completed_requested_prefix_not_full_benchmark" if qualified else
                "wrapper_qualification_failure_after_completed_requested_numeric_prefix" if payload.get("status") == "completed" and status == "exception" else
                "execution_exception_before_first_step" if steps == 0 and status == "exception" else
                "completed_or_failed_prefix_see_original_receipts"),
            "outer_exit_code": data["outer"].get("exit_code"), "completed_substeps": steps, "boxes": boxes,
            "outer_wall_s": timings["process_wall_s"], "wrapper_wall_s": timings["wrapper_wall_s"],
            "payload_wall_s": timings["payload_wall_s"], "driver_elapsed_s": timings["driver_elapsed_s"],
            "timings": timings, "timing_sources": sources, "resources": resources,
            "horizon": {"observed_substeps": steps, "boxes": boxes, "complete_named_horizon": full,
                        "benchmark_planned_steps": instances.get(instance, {}).get("planned_steps"),
                        "requested_candidate_steps": data["start"].get("fused_batch_configuration", {}).get("steps",
                            data["start"].get("complete_substeps", payload.get("expected_substeps")))},
            "started_utc": data["outer_start"].get("started_utc"), "ended_utc": data["outer"].get("ended_utc"),
            "raw_result": relative(files["outer"]), "raw_receipts": [relative(p) for p in files.values() if p.is_file()],
            "saved_comparison": relative(equivalent) if saved_equivalence_receipt else None,
            "width_comparison": relative(width_receipt) if width_receipt.is_file() else None,
            "saved_outputs_equal_claim": status == "COMPLETED_SAVED_OUTPUT_EQUIVALENT" and saved_equivalence_receipt,
            "same_physical_contract_as_current": True if (config_equal or declared_numerical_change) else None,
            "contract_comparison_evidence": {
                "path": relative(equivalent) if config_equal and equivalent.is_file() else relative(width_receipt) if declared_numerical_change else None,
                "configuration_and_arguments_equal": config_equal,
                "local_saved_configuration_comparison": local_contract,
                "configuration_equal_except_declared_numerical_parameters": declared_numerical_change,
                "note": "Completion status alone is not contract-equivalence evidence."},
            "numerical_change": data["start"].get("numerical_change", data["start"].get("numerical_changes")),
            "numerical_settings": {key: metrics.get(key) for key in ("order", "steps", "substeps")},
            "control_candidate": wrapped.get("control_candidate", data["start"].get("control_candidate")),
            "joint_candidate": wrapped.get("joint_candidate", data["start"].get("joint_candidate")),
            "implementation_revision": wrapped.get("implementation_revision", data["start"].get("implementation_revision")),
            "control_counters": wrapped.get("control_counters"),
            "nn_call_evidence": {"driver_recorded_base_calls": payload.get("nn_calls"),
                "actual_calls_including_extra_bounds": wrapped.get("control_counters", {}).get("total_nn_calls"),
                "base_source": {"path": relative(files["payload"]), "field": "nn_calls"} if "nn_calls" in payload else None,
                "actual_source": {"path": relative(files["wrapper"]), "field": "control_counters.total_nn_calls"}
                    if "total_nn_calls" in wrapped.get("control_counters", {}) else None},
            "property_record": property_record, "property_record_source": property_source,
            "stable_speed_ranking_eligible": False,
            "ranking_limitations": ["One new process; no repeated-process ranking.",
                                    "Shared-host timing; possible concurrency is not evidence of isolation.",
                                    "Payload timer excludes earlier wrapper imports; do not substitute it for process wall."],
            "qualification": "saved numerical evidence only; independent NNCS certificate unchanged",
        }
        if equivalent.is_file():
            row["comparison_scope"] = read(equivalent).get("scope")
        local_audit = Path(__file__).with_name(folder.name + "_local_read_audit.json")
        if local_audit.is_file():
            row["local_saved_data_read_audit"] = relative(local_audit)
        rows.append(row)
    return rows, pending


def tora_widths(run, old_summary):
    folder = ROOT / run["raw_result"]
    path = folder.parent / "candidate/data/ranges.bin"
    raw = path.read_bytes()
    if len(raw) != 500 * 152:
        raise ValueError("new TORA ranges require 500 x 152 bytes")
    observations = [json.loads(line) for line in path.with_name("observations.jsonl").read_text().splitlines()]
    if len(observations) != 500 or any(row.get("substep") != i or row.get("accepted") is not True
                                     or row.get("interval_valid") is not True
                                     for i, row in enumerate(observations, 1)):
        raise ValueError("new TORA saved observations do not establish 500 accepted valid records")
    changes = run.get("numerical_change") or {}
    cutoff = changes.get("cut_off_threshold", [None, None])[-1]
    order = run["numerical_settings"]["order"]
    if cutoff is None or order is None:
        raise ValueError("new TORA numerical settings must be recorded, never inferred")
    contract_id = f"official-u11-cutoff{cutoff:g}-order{order}"
    source_id = "OCT6_" + run["run_id"]
    all_rows = []
    state_rows = {"x" + str(i): {"tube": [], "endpoint": []} for i in range(1, 5)}
    for expected, values in enumerate(struct.iter_unpack("<QQd16d", raw), 1):
        lane, step, h = values[:3]
        if lane != 0 or step != expected or h != 0.01:
            raise ValueError("new TORA lane/step/time record mismatch")
        for i, state in enumerate(state_rows):
            bounds = values[3 + 4 * i:7 + 4 * i]
            for geometry, offset in (("tube", 0), ("endpoint", 2)):
                lo, hi = bounds[offset:offset + 2]
                if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                    raise ValueError("new TORA saved interval is nonfinite or reversed")
                row = dict(instance_id=run["benchmark"], method="pytorch_gpu",
                           contract_id=contract_id, saved_object="single_box_observer",
                           state=state, step=step, t_start=(step - 1) * h, t_end=step * h,
                           geometry=geometry, lo=lo, hi=hi, absolute_width=hi - lo,
                           available_lanes=1, expected_lanes=1, complete_initial_set=True,
                           source_id=source_id, source_locator=f"binary_record:{step}")
                state_rows[state][geometry].append(row)
                all_rows.append(row)
    updates, stats = [], []
    for state, geometries in state_rows.items():
        base = next(row for row in old_summary if row["instance_id"] == run["benchmark"]
                    and row["method"] == "pytorch_gpu" and row["state"] == state)
        row = copy.deepcopy(base)
        endpoint = geometries["endpoint"][-1]
        maximum_tube = max(x["absolute_width"] for x in geometries["tube"])
        row.update(contract_id=contract_id, common_endpoint_lo=endpoint["lo"],
                   common_endpoint_hi=endpoint["hi"], common_endpoint_width=endpoint["absolute_width"],
                   common_endpoint_object="single_box_observer", common_max_tube_width=maximum_tube,
                   own_last_endpoint_lo=endpoint["lo"], own_last_endpoint_hi=endpoint["hi"],
                   own_last_endpoint_width=endpoint["absolute_width"],
                   own_last_complete_endpoint_width=endpoint["absolute_width"],
                   own_max_tube_width=maximum_tube, own_complete_prefix_max_tube_width=maximum_tube,
                   full_endpoint_width=endpoint["absolute_width"], full_max_tube_width=maximum_tube,
                   source_ids=source_id, equivalent_p3_receipts="")
        updates.append(row)
        for geometry, rows in geometries.items():
            stats.append(dict(state=state, geometry=geometry, steps=500,
                              final_lo=rows[-1]["lo"], final_hi=rows[-1]["hi"],
                              final_width=rows[-1]["absolute_width"],
                              maximum_width=max(r["absolute_width"] for r in rows),
                              mean_width=sum(r["absolute_width"] for r in rows) / 500,
                              source_id=source_id))
    return all_rows, updates, stats, {"source_id": source_id, "path": relative(path), "bytes": len(raw),
                                    "raw_source": relative(path), "numerical_change": changes,
                                    "numerical_settings": run["numerical_settings"],
                                    "note": "new numerical candidate direct saved binary ranges; no old widths inherited"}


def quad_pooled_geometry(run, expected_steps):
    """Read the actual pooled 12x4 observer schema, plus the distinct driver hull."""
    data = (ROOT / run["raw_result"]).parent / "candidate/data"
    path = data / "observations.jsonl"
    source_id = "OCT6_" + run["run_id"]
    rows = []

    def add(state, step, geometry, bounds, saved_object, sid, locator):
        if len(bounds) != 2:
            raise ValueError("QUAD saved interval requires two bounds")
        lo, hi = bounds
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in bounds) or lo > hi:
            raise ValueError("QUAD saved interval is nonfinite or reversed")
        rows.append(dict(instance_id="quad-reach", method="pytorch_gpu",
            contract_id="paper-equations-1024-control-enclosure-candidate", saved_object=saved_object,
            state=state, step=step, t_start=(step-1)*.005, t_end=step*.005,
            geometry=geometry, lo=lo, hi=hi, absolute_width=hi-lo,
            available_lanes=1024, expected_lanes=1024, complete_initial_set=True,
            source_id=sid, source_locator=locator))

    with path.open() as stream:
        count = 0
        for step, text in enumerate(stream, 1):
            row = json.loads(text)
            if (row.get("substep") != step or row.get("accepted_count") != 1024
                    or row.get("status_counts") != {"0": 1024} or row.get("rejected_lanes") != []):
                raise ValueError("QUAD saved step/all-lane acceptance record differs")
            values = row["tube_endpoint_union_12x4"]
            if len(values) != 12 or any(len(bounds) != 4 for bounds in values):
                raise ValueError("QUAD observer must contain exactly 12 physical states x 4 bounds")
            for i, bounds in enumerate(values, 1):
                for geometry, offset in (("tube", 0), ("endpoint", 2)):
                    add(f"x{i}", step, geometry, bounds[offset:offset+2], "pooled_observer",
                        source_id, f"jsonl_line:{step}")
            count += 1
    if count != expected_steps:
        raise ValueError("QUAD saved sequence does not cover the required candidate horizon")
    result_path = data / "RESULT.json"
    result = read(result_path)
    if (result.get("completed_substeps") != expected_steps or result.get("expected_substeps") != expected_steps
            or result.get("accepted_lane_substeps") != expected_steps*1024
            or result.get("all_lanes_accepted") is not True or result.get("metrics_broken") != 0):
        raise ValueError("QUAD payload does not establish the complete required numerical horizon")
    for i in range(1, 13):
        add(f"x{i}", expected_steps, "endpoint", result["final_hull"][f"x{i}"], "driver_final_hull",
            source_id + "_RESULT", f"final_hull.x{i}")
    source = {"source_id": source_id, "path": relative(path), "bytes": path.stat().st_size,
              "raw_source": relative(path), "control_candidate": run.get("control_candidate"),
              "joint_candidate": run.get("joint_candidate"),
              "implementation_revision": run.get("implementation_revision"),
              "note": "new saved pooled tube/endpoint across 1024 accepted lanes; per-lane ranges, hidden TM/SR and independent NNCS proof unavailable",
              "additional_sources": [{"source_id": source_id + "_RESULT", "path": relative(result_path),
                  "bytes": result_path.stat().st_size, "raw_source": relative(result_path),
                  "note": "distinct new driver final_hull endpoint object; does not replace the pooled observer"}]}
    return rows, source


def quad_widths(run, old_summary):
    if run["benchmark"] != "quad-reach" or not run["horizon"]["complete_named_horizon"]:
        raise ValueError("QUAD main numerical geometry requires reviewed full1024x1000 evidence")
    rows, source = quad_pooled_geometry(run, 1000)
    updates, stats = [], []
    for i in range(1, 13):
        state = f"x{i}"
        geometries = {geometry: [r for r in rows if r["state"] == state and r["geometry"] == geometry
                                and r["saved_object"] == "pooled_observer"]
                      for geometry in ("tube", "endpoint")}
        endpoint = geometries["endpoint"][-1]
        maximum_tube = max(r["absolute_width"] for r in geometries["tube"])
        row = copy.deepcopy(next(r for r in old_summary if r["instance_id"] == "quad-reach"
                                and r["method"] == "pytorch_gpu" and r["state"] == state))
        row.update(contract_id=endpoint["contract_id"], common_endpoint_lo=endpoint["lo"],
                   common_endpoint_hi=endpoint["hi"], common_endpoint_width=endpoint["absolute_width"],
                   common_endpoint_object="pooled_observer", common_max_tube_width=maximum_tube,
                   own_last_endpoint_lo=endpoint["lo"], own_last_endpoint_hi=endpoint["hi"],
                   own_last_endpoint_width=endpoint["absolute_width"],
                   own_last_complete_endpoint_width=endpoint["absolute_width"],
                   own_max_tube_width=maximum_tube, own_complete_prefix_max_tube_width=maximum_tube,
                   full_endpoint_width=endpoint["absolute_width"], full_max_tube_width=maximum_tube,
                   source_ids=source["source_id"], equivalent_p3_receipts="")
        updates.append(row)
        for geometry, values in geometries.items():
            stats.append(dict(state=state, geometry=geometry, steps=1000,
                              final_lo=values[-1]["lo"], final_hi=values[-1]["hi"],
                              final_width=values[-1]["absolute_width"],
                              maximum_width=max(r["absolute_width"] for r in values),
                              mean_width=sum(r["absolute_width"] for r in values)/1000,
                              source_id=source["source_id"], saved_object="pooled_observer"))
    return rows, updates, stats, source


def quad_width_review(run, output):
    """Summarize the saved comparison CSV, including every local increase."""
    path = (ROOT / run["raw_result"]).parent / "candidate/ABSOLUTE_WIDTHS.csv"
    receipt = read(ROOT / run["width_comparison"])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != receipt["width_rows"]:
        raise ValueError("QUAD width comparison CSV is incomplete")
    counts = {key: sum(r["classification"] == key for r in rows) for key in ("narrower", "equal", "wider")}
    subset_count = sum(r["candidate_subset_reference"] == "True" for r in rows)
    if counts != receipt["counts"] or subset_count != receipt["candidate_subset_reference_rows"]:
        raise ValueError("QUAD CSV classes differ from its original receipt")
    wider = []
    for line, row in enumerate(rows, 2):
        if row["classification"] == "wider":
            old = Fraction(row["reference_width_rational"])
            delta = Fraction(row["width_difference_rational"])
            wider.append({"state": row["state"], "geometry": row["geometry"], "step": int(row["substep"]),
                "nominal_t_right": float(row["nominal_t_right"]), "reference_width": float(old),
                "candidate_width": float(Fraction(row["candidate_width_rational"])),
                "absolute_increase": float(delta), "relative_increase": float(delta/old) if old else None,
                "percent_increase": float(100*delta/old) if old else None,
                "exact_increase": row["width_difference_rational"],
                "candidate_subset_reference": row["candidate_subset_reference"] == "True",
                "source": relative(path), "source_locator": f"csv_line:{line}"})
    last_step = max(int(r["substep"]) for r in rows)
    endpoints, tubes = [], []
    for i in range(1, 13):
        state = f"x{i}"
        row = next(r for r in rows if r["state"] == state and r["geometry"] == "endpoint" and int(r["substep"]) == last_step)
        old, new = (Fraction(row[key]) for key in ("reference_width_rational", "candidate_width_rational"))
        endpoints.append({"state": state, "step": last_step, "reference_width": float(old), "candidate_width": float(new),
            "absolute_difference": float(new-old), "narrower_percent": float(100*(old-new)/old) if old else None,
            "candidate_subset_reference": row["candidate_subset_reference"] == "True"})
        values = [r for r in rows if r["state"] == state and r["geometry"] == "tube"]
        old, new = (max(Fraction(r[key]) for r in values) for key in ("reference_width_rational", "candidate_width_rational"))
        tubes.append({"state": state, "reference_max_width": float(old), "candidate_max_width": float(new),
                      "narrower_percent": float(100*(old-new)/old) if old else None})
    review = {"run_id": run["run_id"], "source_csv": relative(path), "source_receipt": run["width_comparison"],
              "last_substep": last_step, "width_rows": len(rows), "counts": counts, "subset_rows": subset_count,
              "wider_rows": wider, "endpoint_gains": endpoints, "maximum_single_step_tube": tubes,
              "scope": "arithmetic summary of saved comparison fields; no experimental rerun, no hidden-state/per-lane proof, no global containment claim"}
    write(output / (run["run_id"] + "_width_review.json"), review)
    write_csv(output / (run["run_id"] + "_endpoint_gains.csv"), endpoints)
    write_csv(output / (run["run_id"] + "_wider_rows.csv"), wider)
    return relative(output / (run["run_id"] + "_width_review.json"))


def quad_geometry_observations(run, output):
    """Derive current object differences and a saved numerical height-band suffix."""
    data = (ROOT / run["raw_result"]).parent / "candidate/data"
    path = data / "observations.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != 1000 or any(r["substep"] != i for i, r in enumerate(records, 1)):
        raise ValueError("QUAD current geometry observations require all1000 saved steps")
    if any(r.get("accepted_count") != 1024 or r.get("status_counts") != {"0": 1024}
           or r.get("rejected_lanes") != [] or len(r["tube_endpoint_union_12x4"]) != 12 for r in records):
        raise ValueError("QUAD current geometry observations need all1024 accepted physical projections")
    payload_path = data / "RESULT.json"
    payload = read(payload_path)
    last = records[-1]["tube_endpoint_union_12x4"]
    differences = []
    for i in range(12):
        pooled = last[i][2:4]
        driver = payload["final_hull"][f"x{i+1}"]
        pl, ph, dl, dh = map(Fraction, (*pooled, *driver))
        differences.append({"state": f"x{i+1}", "pooled_endpoint": pooled, "driver_final_hull": driver,
            "pooled_width": float(ph-pl), "driver_width": float(dh-dl),
            "driver_minus_pooled_lower": float(dl-pl), "driver_minus_pooled_upper": float(dh-ph),
            "driver_minus_pooled_width": float((dh-dl)-(ph-pl)),
            "max_absolute_bound_difference": float(max(abs(dl-pl), abs(dh-ph)))})
    band = (Fraction("0.94"), Fraction("1.06"))
    geometries = {key: [] for key in ("tube", "endpoint", "tube_endpoint_union")}
    extensions = []
    for record in records:
        lo, hi, elo, ehi = record["tube_endpoint_union_12x4"][2]
        if not all(math.isfinite(v) for v in (lo, hi, elo, ehi)) or lo > hi or elo > ehi:
            raise ValueError("QUAD x3 saved bounds are invalid")
        geometries["tube"].append((lo, hi))
        geometries["endpoint"].append((elo, ehi))
        geometries["tube_endpoint_union"].append((min(lo, elo), max(hi, ehi)))
        excess = max(Fraction(lo)-Fraction(elo), Fraction(ehi)-Fraction(hi), Fraction(0))
        if excess > 0:
            extensions.append({"step": record["substep"], "endpoint_excess_beyond_tube": float(excess)})
    suffixes = {}
    for geometry, pairs in geometries.items():
        failures = [step for step, (lo, hi) in enumerate(pairs, 1)
                    if Fraction(lo) < band[0] or Fraction(hi) > band[1]]
        first = max(failures, default=0)+1
        suffixes[geometry] = {"first_saved_step": first if first <= 1000 else None,
            "saved_suffix_steps": 1001-first,
            "nominal_segment_start_s": float(Fraction(first-1, 200)) if first <= 1000 else None,
            "first_saved_endpoint_time_s": float(Fraction(first, 200)) if first <= 1000 else None,
            "preceding_step": first-1 if first > 1 else None,
            "preceding_bounds": pairs[first-2] if first > 1 else None,
            "meaning": "whole-step intervals" if geometry != "endpoint" else "endpoint points only; no between-points claim"}
    combined = suffixes["tube_endpoint_union"]
    first = combined["first_saved_step"]
    notes = {"run_id": run["run_id"], "observer_source": relative(path), "payload_source": relative(payload_path),
        "steps": 1000, "scope": "pooled across1024 accepted boxes,12 physical states; no per-lane hidden-state or independent NNCS proof",
        "endpoint_object_differences": differences,
        "height_band": {"state": "x3", "limits": [0.94, 1.06],
            "source": "docs/ARCHCOMP26_QUAD_PAPER_CONTRACT_DECISION_20261001.md",
            "endpoint_target_not_all_time_safety_band": True, "bound_comparison": "saved binary64 endpoints against exact decimal target bounds",
            "suffixes": suffixes,
            "endpoint_extends_beyond_tube_steps": len(extensions),
            "maximum_endpoint_excess_beyond_tube": max((r["endpoint_excess_beyond_tube"] for r in extensions), default=0),
            "endpoint_extends_beyond_tube_within_combined_suffix_steps": sum(r["step"] >= first for r in extensions) if first is not None else None,
            "qualification": "numerical observation from this run's saved tube AND endpoint; not a participant temporal checker or full reach-and-remain proof"},
        "native_historical_geometry": {"source": "docs/evidence/results/archcomp26_20261001/quad_paper_fourway_saved_20261002/SAVED_REMAIN_AUDIT.json",
            "note": "unchanged independent historical native record; never used as the new P3 suffix"}}
    destination = output / (run["run_id"] + "_geometry_observations.json")
    write(destination, notes)
    return relative(destination), notes


def merged_widths(output, replacements):
    """Stream the complete old table; replace only selected numerical P3 objects."""
    old_path = OLD / "widths/widths_long.csv"
    counts = dict(old_rows=0, retained_rows=0, new_rows=0, output_rows=0)
    removed = {key: 0 for key in replacements}
    inserted = set()
    integer_fields = {"step", "available_lanes", "expected_lanes"}
    float_fields = {"t_start", "t_end", "lo", "hi", "absolute_width"}
    with old_path.open(newline="") as original, (output / "widths_long.csv").open("w", newline="") as table, \
            (output / "widths_long.json").open("w") as packed:
        reader = csv.DictReader(original)
        fields = reader.fieldnames
        writer = csv.DictWriter(table, fields, lineterminator="\n")
        writer.writeheader()
        packed.write('{"fields":' + json.dumps(fields, separators=(",", ":")) + ',"rows":[')

        def emit(row):
            writer.writerow(row)
            values = []
            for field in fields:
                value = row[field]
                if isinstance(value, str):
                    if field in integer_fields:
                        value = int(value) if value else None
                    elif field in float_fields:
                        value = float(value) if value else None
                    elif field == "complete_initial_set":
                        if value not in ("True", "False"):
                            raise ValueError("invalid saved completeness value")
                        value = value == "True"
                values.append(value)
            if counts["output_rows"]:
                packed.write(",")
            packed.write(json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            counts["output_rows"] += 1

        for row in reader:
            counts["old_rows"] += 1
            key = row["instance_id"] if row["method"] == "pytorch_gpu" else None
            if key in replacements:
                removed[key] += 1
                if key not in inserted:
                    for new in replacements[key]:
                        emit(new)
                        counts["new_rows"] += 1
                    inserted.add(key)
            else:
                emit(row)
                counts["retained_rows"] += 1
        packed.write("]}\n")
    if inserted != replacements.keys() or any(value != len(replacements[key]) for key, value in removed.items()):
        raise ValueError("selected new geometry does not replace the exact complete prior object")
    counts.update(replaced_old_rows=removed, source=relative(old_path),
                  policy="all unselected rows retain original field values/order; selected numerical trajectories use only their new saved ranges")
    write(output / "WIDTHS_MERGE_AUDIT.json", counts)
    return counts


def current_status(output, selections, by_id):
    status = read(OLD / "blockers/status_64cells.json")
    status.update(as_of="2026-10-06", new_candidate_index=relative(output / "RUN_INDEX.json"),
                  previous_status_index=relative(OLD / "blockers/status_64cells.json"))
    brief_fields = ("status", "exit_code", "failure", "expected_substeps", "observed_substeps",
                    "completed_substeps", "accepted_substeps", "accepted_lane_substeps",
                    "all_substeps_accepted", "full_horizon_completed", "full_horizon_covered", "property_evaluated")
    property_fields = ("property_evaluated", "author_checker_lines", "author_checker_interpretation",
                       "author_verdict_lines", "checker_verdict", "author_safe_bounds_all_nonpositive",
                       "independent_window_tube_boxes_safe", "saved_tubes_box_disjoint_official_unsafe",
                       "end_to_end_strict_certificate", "end_to_end_floating_point_nn_certificate", "property_record", "endpoint_target_inside")
    changed = []
    for cell in status["cells"]:
        instance = cell["instance_id"]
        if cell["method"] != "pytorch_gpu" or instance not in selections:
            continue
        run = by_id[selections[instance]["run_id"]]
        root = (ROOT / run["raw_result"]).parent
        receipt_paths = [root / "RESULT.json", root / "candidate/RESULT.json", root / "candidate/data/RESULT.json"]
        cell["previous_current_selected_receipts"] = cell["current_selected_receipts"]
        cell["current_selected_receipts"] = []
        current_properties = []
        for path in receipt_paths:
            raw = read(path)
            cell["current_selected_receipts"].append({"path": relative(path), "role": "2026-10-06_current_candidate",
                "original_top_level_fields": {key: raw[key] for key in brief_fields if key in raw}})
            current_properties += [{"path": relative(path), "field": key, "value": raw[key]}
                                   for key in property_fields if key in raw]
        cell["previous_start_paths"] = cell["start_paths"]
        cell["start_paths"] = [relative(root / name) for name in ("START.json", "candidate/START.json", "candidate/data/START.json")
                               if (root / name).is_file()]
        if "selected_oct5_candidate" in cell:
            cell["previous_selected_oct5_candidate"] = cell.pop("selected_oct5_candidate")
        comparison = run["saved_comparison"] or run["width_comparison"]
        cell["selected_oct6_candidate"] = {"run_id": run["run_id"], "run_index": relative(output / "RUN_INDEX.json"),
            "comparison_path": comparison, "width_policy": selections[instance]["width_policy"],
            "full_substeps": run["completed_substeps"], "boxes": run["boxes"], "sample_count": 1,
            "timings": run["timings"], "timing_sources": run["timing_sources"],
            "numerical_change": run["numerical_change"], "numerical_settings": run["numerical_settings"],
            "joint_candidate": run.get("joint_candidate"), "control_candidate": run.get("control_candidate"),
            "nn_call_evidence": run["nn_call_evidence"],
            "width_tradeoff_review": run.get("width_tradeoff_review"),
            "scope": run.get("comparison_scope") or "new full saved physical ranges; hidden TM/SR not compared",
            "qualification": "仅使用新实验保存收据及范围；无旧实验/checker重跑，不新增独立NNCS证书。"}
        cell["verdict"]["previous_result_property_fields"] = cell["verdict"]["original_result_property_fields"]
        cell["verdict"]["original_result_property_fields"] = current_properties
        cell["verdict"]["current_property_evidence_paths"] = [relative(p) for p in receipt_paths]
        if run["property_record"] is not None:
            cell["verdict"]["current_saved_geometry_observation"] = run["property_record"]
            cell["verdict"]["current_saved_geometry_observation_source"] = run["property_record_source"]
        if instance == "quad-reach" and run.get("geometry_observation_notes"):
            details = run["geometry_observation_notes"]
            suffix = details["height_band"]["suffixes"]["tube_endpoint_union"]
            cell["verdict"]["current_quad_geometry_details"] = run["geometry_observation_path"]
            suffix_text = (f"保存入[0.94,1.06]后缀从第{suffix['first_saved_step']}步、名义段起点{suffix['nominal_segment_start_s']}秒开始"
                           if suffix["first_saved_step"] is not None else "未观察到截至T=5的连续入高度带保存后缀")
            cell["verdict"]["saved_geometry_observation_zh"] = (
                "当前run自己的 pooled x3 tube∪endpoint " + suffix_text + "；"
                "driver终点与pooled观察器为不同对象，逐态差值另存。此为保存数值观察，不是完整reach-remain checker或独立证书。")
        if selections[instance]["width_policy"] == "new_saved_ranges":
            record = run["property_record"]
            if instance == "quad-reach":
                if not record or record.get("payload_endpoint_target_inside") is not True:
                    raise ValueError("new QUAD main selection lacks its own recorded terminal target observation")
                cell["verdict"]["saved_geometry_observation_zh"] += " 当前payload的T=5终点目标字段为true；没有逐盒全程几何。"
                cell["verdict"]["label_source_and_scope"] = "仅引用当前payload RESULT的endpoint_target_inside；完整数值时域不等于完整reach-remain性质证明。"
            else:
                if not record or record.get("terminal_box_in_target") is not True:
                    raise ValueError("new TORA main selection needs its own target observation; cannot inherit previous verdict")
                cell["verdict"]["saved_geometry_observation_zh"] = "新候选自己的 T=5 保存 x1/x2 终点全盒入目标；property_evaluated=false，未执行性质checker，也不是独立NNCS证书。"
                cell["verdict"]["label_source_and_scope"] = "当前P3 payload RESULT明确property_evaluated=false；目标观察来自本候选WIDTH_COMPARISON，未继承旧verdict。"
            cell["absolute_width_paths"] = [relative(output / (run["run_id"] + "_widths_long.csv")),
                                            relative(output / (run["run_id"] + "_summary.json"))]
            cell["historical_figure_paths"] = cell["figure_paths"]
            cell["figure_paths"] = []
            cell["geometry_source_scope_note"] = "新数值候选只采用自己的保存范围；历史图另列。当前新图由报告组单独生成，不借用旧轨迹。"
        cell["evidence_paths"] = list(dict.fromkeys(cell["evidence_paths"] + [relative(p) for p in receipt_paths]
            + cell["start_paths"] + [comparison, relative(output / "RUN_INDEX.json")] + cell["absolute_width_paths"]
            + ([run["geometry_observation_path"]] if run.get("geometry_observation_path") else [])))
        changed.append(instance)
    keys = {(c["instance_id"], c["method"]) for c in status["cells"]}
    if len(status["cells"]) != 64 or len(keys) != 64:
        raise ValueError("status overlay must preserve exactly 64 cells")
    status["current_selection_20261006"] = selections
    status["document_assembly_checks"]["updated_p3_cells"] = changed
    status["document_assembly_checks"]["boundary_and_reason_text_preserved"] = True
    status["document_assembly_checks"]["numeric_checks_rerun"] = False
    write(output / "status_64cells.json", status)
    return changed


def publish_data(output, timing):
    """Maintain only the assigned data subdirectories of the new report package."""
    timing_names = {"RUN_INDEX.json", "timing_index.json", "current_selected_four_way.csv", "selection_audit.json",
                    "selected_alternatives.json", "selected_alternatives.csv"}
    root_names = {"BUILD_AUDIT.json"}
    mapping = {}
    for path in output.iterdir():
        if not path.is_file():
            continue
        subdir = "timing" if path.name in timing_names else "blockers" if path.name == "status_64cells.json" else "" if path.name in root_names else "widths"
        mapping[relative(path)] = relative(REPORT / subdir / path.name)
    for old_path, new_path in mapping.items():
        source, target = ROOT / old_path, ROOT / new_path
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text()
        for original, replacement in sorted(mapping.items(), key=lambda item: -len(item[0])):
            text = text.replace(original, replacement)
        target.write_text(text)
    for name in ("campaign_statistics.csv", "historical_fast_branch.csv"):
        (REPORT / "timing" / name).write_bytes((OLD / "timing" / name).read_bytes())
    for name in ("instances.json", "coverage_by_instance.json", "missing_fields.csv", "missing_fields.json",
                 "native_quad_allstates.csv", "native_quad_allstates.json"):
        (REPORT / "widths" / name).write_bytes((OLD / "widths" / name).read_bytes())
    run_rows = []
    for key, run in timing["runs"].items():
        run_rows.append({"run_key": key, "benchmark": run.get("benchmark"), "method": run.get("method"),
                        "generation": run.get("generation"), "status": run.get("status"), **run["timings"],
                        "timing_sources": run["timing_sources"], "horizon": run["horizon"],
                        "resources": run["resources"], "raw_receipts": run["raw_receipts"],
                        "ranking_limitations": run["ranking_limitations"]})
    write_csv(REPORT / "timing/runs.csv", run_rows)
    data_readme = Path(__file__).with_name("README.zh.md").read_text()
    (REPORT / "DATA_README.zh.md").write_text(data_readme + "\n\n正式包布局：`timing/` 保存全部计时层和当前选择，`widths/` 保存完整几何与汇总，`blockers/` 保存当前 64 格及性质来源。独立 rebuild 入口和选择文件仍在上列研究目录；本文件不代表实验重复验证。\n")
    (REPORT / "timing/README.md").write_text("# 10 月 6 日计时数据\n\n`current_selected_four_way.csv` 为 64 格当前选择；`timing_index.json` 的 `current_selected_run` 是选择依据。`runs.csv` 包含旧及新增过程，`RUN_INDEX.json` 仅列 10 月 6 日研究。各层时间和源字段分列；没有完整时域的耗时不可进入完整速度排名。原 campaign 分布与历史快分支表原样保留，不包含新单次候选。\n")
    (REPORT / "widths/README.md").write_text("# 10 月 6 日保存宽度\n\n`widths_long.csv/json` 为整合全表；`summary.*` 为 16 实例四方法全部物理状态汇总。新数值主项用自己的完整保存范围替换旧轨迹；纯实现等值主项仅追加比较收据。`WIDTHS_MERGE_AUDIT.json` 记录逐行合并；`sources.json` 指向实际范围。`width_alternatives.json` 和各候选独立表保留未采用数值方案。`instances`、缺项及 native QUAD 派生表复用旧保存证据；新数值 QUAD 的备用 driver 终点对象由同一次新结果替换，其余备用对象保留。证据边界不扩展。逐态比较不能合成跨量纲整体紧度。\n")
    status = read(REPORT / "blockers/status_64cells.json")
    lines = ["# 10 月 6 日 64 格状态与证据边界", "", "数值完整、作者标签、保存几何观察和独立证明分别保留。原合同缺件、数值早停、缺保存数据及性质 UNKNOWN 的原因不因提速改变；本表仅更新当前 P3 收据。", ""]
    for cell in status["cells"]:
        lines += [f"## {cell['instance_name']} / {cell['method_label']}", "", f"状态：{cell['status']}。{cell['reason_zh']}", "",
                  f"仍需：{cell['needed_evidence_or_change_zh']}", "",
                  "当前收据：" + ("；".join("`" + r["path"] + "`" for r in cell["current_selected_receipts"]) or "合同未执行，无结果收据。"), ""]
    (REPORT / "blockers/README.zh.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path(__file__).with_name("selection.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("current"))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(STUDY / "report_data"):
        raise ValueError("derived output must remain under the new report_data directory")
    output.mkdir(parents=True, exist_ok=True)
    selections = read(args.selection)
    baseline = read(OLD / "timing/timing_index.json")
    instance_specs = read(OLD / "widths/instances.json")
    instances = {r["instance_id"]: r for r in instance_specs}
    runs, pending = new_runs(instances, baseline["timing_layer_definitions"])
    by_id = {r["run_id"]: r for r in runs}
    posthoc = []
    posthoc_path = STUDY / "results/posthoc/quad_fused_v2_saved_comparison.json"
    if posthoc_path.is_file():
        receipt = read(posthoc_path)
        posthoc.append({"path": relative(posthoc_path), "associated_run_id": "quad_paper_fused256v2_40_001",
                        "receipt": receipt,
                        "scope": "saved-data comparison after wrapper dispatch qualification failure; original failed wrapper RESULT unchanged; 40-step prefix only"})
        if "quad_paper_fused256v2_40_001" in by_id:
            by_id["quad_paper_fused256v2_40_001"]["posthoc_saved_comparison"] = posthoc[-1]
    joint_posthoc = STUDY / "results/posthoc/jointfull_vs_controlfull_saved_comparison.json"
    if joint_posthoc.is_file():
        posthoc.append({"path": relative(joint_posthoc), "associated_run_id": "quad_paper_joint_full1000_001",
            "receipt": read(joint_posthoc),
            "scope": "direct saved-data comparison between two distinct completed new full runs; joint geometry still derives from its own files, not the control-run files"})
        if "quad_paper_joint_full1000_001" in by_id:
            by_id["quad_paper_joint_full1000_001"]["posthoc_saved_comparison"] = posthoc[-1]
    index = {"scope": "20261006 new stages; old 289 attempts and Oct 5 results unchanged",
             "timing": "all source timing layers retained; one sample per new run",
             "qualification": "numerical completion, property observation and independent certificate remain separate",
             "digest_operations": 0, "runs": runs, "posthoc_saved_data_comparisons": posthoc,
             "pending_local_result_directories": pending}
    index["outcome_counts"] = {key: sum(r["candidate_outcome_classification"] == key for r in runs)
                               for key in sorted({r["candidate_outcome_classification"] for r in runs})}
    selected_csv = list(csv.DictReader((OLD / "timing/current_selected_four_way.csv").open()))
    old_keys = {(r["benchmark"], r["method"]): r["run_key"] for r in selected_csv}
    timing = copy.deepcopy(baseline)
    for row in runs:
        timing["runs"][row["run_key"]] = row
    summary = read(OLD / "widths/summary.json")
    sources = read(OLD / "widths/sources.json")
    alternatives, numerical_updates, numerical_longs = [], {}, {}
    for row in runs:
        if row["benchmark"] == "quad-reach" and row["width_comparison"]:
            row["width_tradeoff_review"] = quad_width_review(row, output)
        if row["benchmark"] == "quad-reach" and row["horizon"]["complete_named_horizon"]:
            row["geometry_observation_path"], row["geometry_observation_notes"] = quad_geometry_observations(row, output)
        if (row["benchmark"] in ("tora-reach-sigmoid", "tora-reach-tanh", "quad-reach")
                and row["width_comparison"] and row["horizon"]["complete_named_horizon"]):
            longs, updates, stats, source = (quad_widths if row["benchmark"] == "quad-reach" else tora_widths)(row, summary)
            prefix = row["run_id"]
            write(output / (prefix + "_widths_long.json"), longs)
            write_csv(output / (prefix + "_widths_long.csv"), longs)
            write(output / (prefix + "_summary.json"), updates)
            write_csv(output / (prefix + "_width_statistics.csv"), stats)
            numerical_updates[prefix] = updates
            numerical_longs[prefix] = longs
            sources.append(source)
            sources.extend(source.get("additional_sources", []))
            alternatives.append({"run_id": prefix, "benchmark": row["benchmark"],
                                 "new_width_source": source, "width_rows": len(longs),
                                 "original_comparison_receipt": row["width_comparison"],
                                 "property_record": row["property_record"],
                                 "selected": prefix == selections.get(row["benchmark"], {}).get("run_id")})
    write(output / "RUN_INDEX.json", index)
    selected_notes = []
    for instance, selection in selections.items():
        run = by_id[selection["run_id"]]
        if (run["benchmark"] != instance or not run["horizon"]["complete_named_horizon"]
                or run["same_physical_contract_as_current"] is not True):
            raise ValueError("selected run must have its complete matching physical contract")
        if selection["width_policy"] == "equivalent_saved_reference":
            if not run["saved_outputs_equal_claim"]:
                raise ValueError("selected equivalent-width policy lacks successful saved comparison")
            for row in summary:
                if row["instance_id"] == instance and row["method"] == "pytorch_gpu":
                    row["equivalent_p3_receipts"] = ";".join(filter(None, [row["equivalent_p3_receipts"], run["saved_comparison"]]))
        elif selection["width_policy"] == "new_saved_ranges":
            replacement = {r["state"]: r for r in numerical_updates[run["run_id"]]}
            summary = [replacement[r["state"]] if r["instance_id"] == instance and r["method"] == "pytorch_gpu"
                       else r for r in summary]
        else:
            raise ValueError("unknown selected width policy")
        previous_key = old_keys[instance, "pytorch_gpu"]
        previous = baseline["runs"][previous_key]
        selected_notes.append({"benchmark": instance, "previous_selected_run": previous_key,
                               "new_selected_run": run["run_key"], **selection,
                               "old_timings": previous["timings"], "new_timings": run["timings"],
                               "note": "single measurements; timing layers stay separate"})
        timing["benchmarks"][instance]["methods"]["pytorch_gpu"]["current_selected_run"] = run["run_key"]
        for row in selected_csv:
            if row["benchmark"] == instance and row["method"] == "pytorch_gpu":
                row.update(run_key=run["run_key"], generation=run["generation"], sample_count=1,
                           status=run["status"], outer_status=run["outer_status"], outer_exit_code=run["outer_exit_code"],
                           previous_selected_run=previous_key, current_selected_run=run["run_key"],
                           selection_policy=selection["reason"], complete_named_horizon=True,
                           observed_substeps=run["completed_substeps"], boxes=run["boxes"],
                           started_utc=run["started_utc"], ended_utc=run["ended_utc"],
                           cpu_affinity=run["resources"]["cpu_affinity"], physical_gpu=run["resources"]["physical_gpu"],
                           host=run["resources"]["host"], raw_receipts=run["raw_receipts"],
                           horizon_sources={"new_horizon": relative(output / "RUN_INDEX.json"), "run_id": run["run_id"]},
                           resource_sources={"new_resources": relative(output / "RUN_INDEX.json"), "run_id": run["run_id"]},
                           concurrency={"qualification": "new-run overlap not inferred from prior cohort; resource isolation unproven"},
                           ranking_limitations=run["ranking_limitations"], **run["timings"])
                for layer in LAYERS:
                    row[layer + "_source"] = run["timing_sources"][layer]
    if len(selected_csv) != 64 or len({(r["benchmark"], r["method"]) for r in selected_csv}) != 64:
        raise ValueError("new table must preserve all 64 method cells")
    for row in selected_csv:
        timing["benchmarks"][row["benchmark"]]["methods"][row["method"]]["current_selected_run"] = row["run_key"] or None
    pairwise = []
    mapped = {(r["instance_id"], r["method"], r["state"]): r for r in summary}
    for (instance, method, state), p3 in mapped.items():
        if method != "pytorch_gpu":
            continue
        for other in ("huan", "xiangru", "flowstar_native"):
            ref = mapped[instance, other, state]
            for field in ("common_endpoint_width", "common_max_tube_width"):
                a, b = p3[field], ref[field]
                if a is None or b is None:
                    continue
                pairwise.append(dict(instance_id=instance, state=state, geometry=field,
                                     reference_method=other, common_time=p3["common_time"], p3_width=a,
                                     reference_width=b, difference=a-b, relative_difference_percent=100*(a/b-1) if b else None,
                                     p3_endpoint_subset=(p3["common_endpoint_lo"] >= ref["common_endpoint_lo"]
                                         and p3["common_endpoint_hi"] <= ref["common_endpoint_hi"]) if field == "common_endpoint_width" else None))
    timing["current_selection_20261006"] = selections
    timing["as_of"] = "2026-10-06"
    timing["validation"]["new_stage_count_20261005"] = timing["validation"].pop("new_stage_count")
    timing["validation"]["new_stage_count_20261006"] = len(runs)
    timing["validation"]["total_new_optimization_stages"] = timing["validation"]["new_stage_count_20261005"] + len(runs)
    timing["validation"]["current_method_cells"] = len(selected_csv)
    timing["validation"]["current_oct6_selected_cells"] = len(selections)
    write(output / "timing_index.json", timing)
    write_csv(output / "current_selected_four_way.csv", selected_csv)
    write(output / "selection_audit.json", selected_notes)
    alternatives_by_instance = []
    for instance in sorted({r["benchmark"] for r in runs if r["benchmark"] in instances}):
        previous_key = old_keys[instance, "pytorch_gpu"]
        previous = baseline["runs"][previous_key]
        candidates = [(previous_key, previous, "previous_selected_reference")]
        candidates += [(r["run_key"], r, "selected_main" if selections.get(instance, {}).get("run_id") == r["run_id"]
                        else "unselected_new_candidate") for r in runs if r["benchmark"] == instance]
        for key, candidate, role in candidates:
            alternatives_by_instance.append({"benchmark": instance, "run_key": key, "role": role,
                "status": candidate.get("status"), "complete_named_horizon": candidate["horizon"]["complete_named_horizon"],
                **candidate["timings"], "timing_sources": candidate["timing_sources"],
                "saved_comparison": candidate.get("saved_comparison"), "width_comparison": candidate.get("width_comparison"),
                "numerical_change": candidate.get("numerical_change"),
                "qualification": "time/width tradeoff retained; no automatic fastest-only promotion or global containment claim"})
    write(output / "selected_alternatives.json", alternatives_by_instance)
    write_csv(output / "selected_alternatives.csv", alternatives_by_instance)
    write(output / "summary.json", summary)
    write_csv(output / "summary.csv", summary)
    alternate_objects = read(OLD / "widths/alternate_endpoint_objects.json")
    if selections.get("quad-reach", {}).get("width_policy") == "new_saved_ranges":
        alternate_objects = [r for r in alternate_objects if not (r["instance_id"] == "quad-reach" and r["method"] == "pytorch_gpu")]
        alternate_objects += [r for r in numerical_longs[selections["quad-reach"]["run_id"]]
                              if r["saved_object"] == "driver_final_hull"]
    write(output / "alternate_endpoint_objects.json", alternate_objects)
    write_csv(output / "alternate_endpoint_objects.csv", alternate_objects)
    write(output / "sources.json", sources)
    write(output / "pairwise_comparisons.json", pairwise)
    write_csv(output / "pairwise_comparisons.csv", pairwise)
    write(output / "width_alternatives.json", alternatives)
    merge = merged_widths(output, {instance: numerical_longs[selection["run_id"]]
                          for instance, selection in selections.items() if selection["width_policy"] == "new_saved_ranges"})
    changed_status = current_status(output, selections, by_id)
    write(output / "BUILD_AUDIT.json", {
        "baseline": relative(OLD), "source_selection": relative(args.selection.resolve()),
        "new_runs": len(runs), "selected_new_p3": len(selections), "method_cells": len(selected_csv),
        "width_summary_rows": len(summary), "pairwise_rows": len(pairwise),
        "merged_geometry": merge, "updated_status_cells": changed_status,
        "old_solver_or_checker_executed": False, "digest_operations": 0,
        "old_sources_or_results_modified": False,
        "width_policy": "new numerical candidate ranges are derived independently; equivalent trajectories retain old geometry with explicit comparison receipt",
        "selected_local_read_audits": {instance: by_id[selection["run_id"]]["local_saved_data_read_audit"]
            for instance, selection in selections.items()
            if by_id[selection["run_id"]].get("local_saved_data_read_audit")},
        "pending": pending})
    publish_data(output, timing)
    print(json.dumps({"new_runs": len(runs), "selected": len(selections), "cells": len(selected_csv),
                      "summary_rows": len(summary), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
