#!/usr/bin/env python3
"""New ACC or official u=11f TORA sigmoid run using three implementation adapters.

Frozen full runners retain their controller, strict arithmetic, observers and
property checks. This wrapper never runs the frozen campaign or a baseline.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Loading a frozen runner must not create bytecode beside its source.
sys.dont_write_bytecode = True

from horner_edge import install as install_horner
from private_outputs import install as install_private
from weighted_small_batch import install as install_weighted
from run_quad_candidate import guard_digests, reject_digest, write_new


ROOT = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
RUNS = ROOT / "runs/archcomp26_20261001"
ACC = RUNS / "acc_fourway_campaign_001/steady05_ours_p3/data"
TORA = RUNS / "tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_ours_p3/payload"
TORA_FILES = (
    "run_full.py", "author_support.py", "archcomp26_dp_p3_nohash.py",
    "archcomp26_tora_remain_author_nohash.py", "archcomp26_tora_remain_p3_nohash.py",
    "controller_plant_u.onnx", "nn_tora_sigmoid.mat",
)
RUNTIME_FIELDS = {
    "config", "wall_s", "elapsed_s", "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
    "cuda_graph_captures", "cuda_graph_hits",
}


def load_module(name, path):
    if name in sys.modules:
        raise RuntimeError(f"module already loaded: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stage_tora(snapshot, data):
    """Copy the saved inputs; rebase only the three explicit artifact paths."""
    data.mkdir(exist_ok=False)
    sizes = {}
    for name in TORA_FILES:
        raw = (snapshot / name).read_bytes()
        with (data / name).open("xb") as stream:
            stream.write(raw)
        if (data / name).read_bytes() != raw:
            raise RuntimeError(f"staged bytes differ: {name}")
        sizes[name] = len(raw)
    receipt = json.loads((snapshot / "controller_plant_u.onnx.json").read_text())
    old_root = str(Path(receipt["model_path"]).parent)
    if receipt["mat_path"] != str(Path(old_root) / "nn_tora_sigmoid.mat"):
        raise RuntimeError("saved controller receipt paths disagree")
    for name, count in (("config.yaml", 1), ("controller_plant_u.onnx.json", 2)):
        source = (snapshot / name).read_text()
        if source.count(old_root) != count:
            raise RuntimeError(f"unexpected artifact path occurrences in {name}")
        with (data / name).open("x") as stream:
            stream.write(source.replace(old_root, str(data)))
    return {"source": str(snapshot), "byte_equal_copies": sizes,
            "path_only_relocation": ["config.yaml", "controller_plant_u.onnx.json"],
            "old_artifact_root": old_root, "new_artifact_root": str(data)}


def compare_saved(instance, reference, data):
    steps = 50 if instance == "acc" else 500
    files = ("ranges.jsonl", "safety.jsonl") if instance == "acc" else (
        "ranges.bin", "observations.jsonl")
    sizes = {}
    for name in files:
        old, new = (reference / name).read_bytes(), (data / name).read_bytes()
        if old != new:
            raise RuntimeError(f"{instance} saved bytes differ: {name}")
        if name.endswith(".jsonl") and len(new.splitlines()) != steps:
            raise RuntimeError(f"{instance} incomplete saved sequence: {name}")
        if name == "ranges.bin" and len(new) != 500 * 152:
            raise RuntimeError("TORA saved range count differs from 500")
        sizes[name] = len(new)
    old_config = (reference / "config.yaml").read_text()
    new_config = (data / "config.yaml").read_text()
    if instance == "tora-sigmoid":
        receipt = json.loads((reference / "controller_plant_u.onnx.json").read_text())
        old_root = str(Path(receipt["model_path"]).parent)
        new_config = new_config.replace(str(data), old_root)
        for name in TORA_FILES:
            if (reference / name).read_bytes() != (data / name).read_bytes():
                raise RuntimeError(f"TORA frozen source/model bytes differ: {name}")
        old_receipt = (reference / "controller_plant_u.onnx.json").read_text()
        new_receipt = (data / "controller_plant_u.onnx.json").read_text()
        if old_receipt != new_receipt.replace(str(data), old_root):
            raise RuntimeError("TORA export receipt changed beyond artifact paths")
    if old_config != new_config:
        raise RuntimeError(f"{instance} generated contract changed")
    old_start = json.loads((reference / "START.json").read_text())
    new_start = json.loads((data / "START.json").read_text())
    old_argv, new_argv = old_start["driver_argv"], new_start["driver_argv"]
    old_data = str(Path(old_argv[1]).parent)
    if (new_argv[1] != str(data / "config.yaml")
            or old_argv != [value.replace(str(data), old_data) for value in new_argv]):
        raise RuntimeError(f"{instance} shared driver/arguments changed")
    compared = {}
    for name in ("RESULT.json", "metrics.json"):
        old = json.loads((reference / name).read_text())
        new = json.loads((data / name).read_text())
        fields = set(old) - RUNTIME_FIELDS
        if fields != set(new) - RUNTIME_FIELDS:
            raise RuntimeError(f"{instance} {name} scientific field set differs")
        for field in sorted(fields):
            # Preserve signed zero while comparing every non-runtime field.
            if json.dumps(old[field], sort_keys=True) != json.dumps(new[field], sort_keys=True):
                raise RuntimeError(f"{instance} {name} differs: {field}")
        compared[name] = sorted(fields)
    metrics = json.loads((data / "metrics.json").read_text())
    result = json.loads((data / "RESULT.json").read_text())
    if (metrics["B"] != 1 or metrics["order"] != 3 or metrics["broken"] != 0
            or metrics["steps"] * metrics["substeps"] != steps):
        raise RuntimeError("candidate B/order/full horizon changed")
    if instance == "acc":
        if (result["status"] != "completed" or result["completed_substeps"] != 50
                or result["all_substeps_accepted"] is not True
                or result["author_safe_bounds_all_nonpositive"] is not True
                or result["independent_tube_boxes_all_safe"] is not True
                or result["end_to_end_floating_point_nn_certificate"] is not False):
            raise RuntimeError("ACC complete numerical/property record is invalid")
        property_record = {key: result[key] for key in (
            "safety_events", "author_safe_bounds_all_nonpositive",
            "independent_tube_halfspace_min_lower", "independent_tube_boxes_all_safe")}
    else:
        if (result["status"] != "completed_full_numerical_horizon"
                or result["accepted_substeps"] != 500
                or result["property_evaluated"] is not False):
            raise RuntimeError("TORA complete numerical record is invalid")
        # Read the previous checker receipt. Never execute the old checker.
        events_path = reference.parent.parent / "events.jsonl"
        matches = [row for line in events_path.read_text().splitlines()
                   if (row := json.loads(line)).get("method") == "ours_p3"
                   and Path(row["path"]).name == reference.parent.name]
        if len(matches) != 1 or matches[0]["valid"] is not True or matches[0]["exit_code"] != 0:
            raise RuntimeError("TORA saved campaign property receipt is missing/invalid")
        event = matches[0]
        property_record = {
            "saved_receipt": str(events_path), "saved_sample": event["path"],
            "saved_campaign_valid": True, "terminal_x1_x2": event["terminal_x1_x2"],
            "basis": "checker input ranges/config equal the saved checked record; property label inherited from saved evidence",
            "payload_property_evaluated": False, "new_property_checker_executed": False,
        }
    return {"reference": str(reference), "saved_bytes_equal": sizes,
            "completed_substeps": steps, "B": 1,
            "config_equal_after_artifact_path_relocation": True,
            "driver_argv_equal_after_artifact_path_relocation": True,
            "shared_driver": old_argv[0],
            "equal_fields": compared, "excluded_runtime_fields": sorted(RUNTIME_FIELDS),
            "property_record": property_record, "digest_operations": 0,
            "scope": "all saved physical ranges, observations/property rows, all non-runtime result/metrics fields; hidden TM/SR not compared"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=("acc", "tora-sigmoid"), required=True)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weighted-mode", choices=("small", "fused"), default="small")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or set(os.sched_getaffinity(0)) != {10, 11, 12, 13}:
        raise RuntimeError("both frozen profiles require physical GPU2 and CPUs10-13")
    if json.loads((args.gate / "RESULT.json").read_text()).get("status") != "PASSED_NEW_PRIVATE_OUTPUT_GPU_GATE":
        raise RuntimeError("new private-output GPU gate has not passed")
    is_acc = args.instance == "acc"
    reference = (args.reference or (ACC if is_acc else TORA)).resolve()
    snapshot = (args.source_snapshot or (RUNS / "acc_prep_001" if is_acc else TORA)).resolve()
    output = args.output.resolve()
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (reference, snapshot)):
        raise RuntimeError("candidate output must be separate from frozen source/reference")
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    started = time.perf_counter()
    outer_path = reference.parent / ("RESULT.json" if is_acc else "outer/RESULT.json")
    old_outer = json.loads(outer_path.read_text())
    old_metrics = json.loads((reference / "metrics.json").read_text())
    write_new(output / "START.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "instance": args.instance, "source_snapshot": str(snapshot), "reference": str(reference),
        "gate": str(args.gate), "physical_gpu": 2, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "B": 1, "working_order": 3, "complete_substeps": 50 if is_acc else 500,
        "contract": ("ACC exact feature TM; 50 periods x .1s, all-time safety"
                     if is_acc else "official TORA sigmoid u=11f; 10 periods x .5s / 500 substeps, T=5 endpoint target"),
        "changes": ["private output allocation", "preloaded Horner remainder binding",
                    ("accepted weighted graph scratch: fixed one row" if args.weighted_mode == "small"
                     else "single-row two-round graph, success-only commit and original fallback")],
        "weighted_mode": args.weighted_mode,
        "fused_reference_check": ("first successful input: direct output bytes and full statistics; included in timers"
                                  if args.weighted_mode == "fused" else None),
        "reference_outer_result": str(outer_path), "reference_outer_wall_s": old_outer["wall_s"],
        "reference_driver_elapsed_s": old_metrics["elapsed_s"],
        "timing_policy": "wrapper wall includes staging/imports/preloads/checks; payload wall excludes early Torch import; driver elapsed excludes setup/preload/engine creation and includes lazy graph capture; supervisor outer wall is recorded separately",
        "timing_qualification": "old/new payload wall alone is not matched timing; no speed claim from a single candidate",
        "payload_profile_note": "TORA keeps its frozen inherited profile label; this outer START identifies the new candidate",
        "runtime_guards": "all digests and compilation prohibited; script decorators remain eager Python",
        "no_jit": True, "no_digest_operations": True,
        "dont_write_bytecode": sys.dont_write_bytecode})
    bindings, weighted, restorations = [], None, []
    result = {"status": "exception", "end_to_end_strict_certificate": False,
              "end_to_end_floating_point_nn_certificate": False}
    code = 1
    try:
        guard_digests()
        import torch
        import torch.utils.cpp_extension as extension

        def eager_script(function=None, **kwargs):
            return function if function is not None else lambda value: value

        torch.jit.script = eager_script
        torch.jit.trace = torch.compile = reject_digest
        extension.load = extension.load_inline = reject_digest

        def wrap_prepare(original_prepare):
            def prepare(*a, **kw):
                nonlocal weighted
                if bindings:
                    raise RuntimeError("candidate prepare must run exactly once")
                prepared = original_prepare(*a, **kw)
                runtime, driver = prepared[:2]
                from flowstar_gpu import cuda_kernels as ck, private_output_kernels
                from flowstar_gpu import sparse_exec, horner_edge_kernels, weighted_validation
                private = install_private(runtime, ck, private_output_kernels)
                bindings.append(private)
                write_new(output / "PRIVATE_OUTPUT_BINDING.json", private.receipt)
                if args.weighted_mode == "fused":
                    from weighted_fused_single import install as install_fused
                    weighted = install_fused(runtime, weighted_validation, sparse_exec, check_first=1)
                else:
                    weighted = install_weighted(runtime, weighted_validation, sparse_exec, rows=1)
                bindings.append(weighted)
                write_new(output / "WEIGHTED_BINDING.json", weighted.receipt)
                old_engine = driver.SparseEngine

                def create_engine(*a, **kw):
                    engine = old_engine(*a, **kw)
                    binding = install_horner(runtime, engine, sparse_exec, horner_edge_kernels)
                    bindings.append(binding)
                    write_new(output / "HORNER_EDGE_BINDING.json", binding.receipt)
                    return engine

                driver.SparseEngine = create_engine
                restorations.append((driver, "SparseEngine", old_engine))
                if is_acc:
                    old_main = driver.main

                    def main_with_first_refusal():
                        # ACC installs its saved range observer after prepare().
                        observed_advance = driver.advance_sparse

                        def advance(*a, **kw):
                            state, accepted = observed_advance(*a, **kw)
                            if not bool(accepted.all()):
                                raise ArithmeticError("first rejected lane after preserved ACC observation")
                            return state, accepted

                        driver.advance_sparse = advance
                        try:
                            return old_main()
                        finally:
                            driver.advance_sparse = observed_advance

                    driver.main = main_with_first_refusal
                    restorations.append((driver, "main", old_main))
                return prepared
            return prepare

        if is_acc:
            baseline = load_module("small1_acc_frozen", snapshot / "archcomp26_acc_p3_nohash.py")
            baseline.prepare = wrap_prepare(baseline.prepare)
            code = baseline.execute(argparse.Namespace(
                mode="full", source_config=snapshot / "acc_participant_vrel_lead_minus_ego.yaml", output=data))
        else:
            write_new(output / "STAGED_INPUTS.json", stage_tora(snapshot, data))
            for name in ("archcomp26_dp_p3_nohash", "archcomp26_tora_remain_author_nohash"):
                load_module(name, data / (name + ".py"))
            baseline = load_module("small1_tora_frozen", data / "run_full.py")
            original_load = baseline.load_module

            def load_with_prepare(name, path):
                module = original_load(name, path)
                if name == "archcomp26_tora_reach_p3_support":
                    module.prepare = wrap_prepare(module.prepare)
                return module

            baseline.load_module = load_with_prepare
            code = baseline.main()
        if code:
            raise RuntimeError(f"frozen numerical process returned {code}")
        if len(bindings) != 3 or not weighted.counters["graph_map_calls"]:
            raise RuntimeError("candidate did not consume every selected binding")
        if args.weighted_mode == "fused" and (
                not weighted.counters["fast_calls"]
                or weighted.counters["reference_checks"] != 1
                or weighted.counters["candidate_exceptions"]):
            raise RuntimeError("fused graph requires a checked fast result and zero candidate exceptions")
        comparison = compare_saved(args.instance, reference, data)
        write_new(output / "SAVED_COMPARISON.json", comparison)
        metrics = json.loads((data / "metrics.json").read_text())
        payload = json.loads((data / "RESULT.json").read_text())
        result.update(status="COMPLETED_SAVED_OUTPUT_EQUIVALENT", B=1,
                      completed_substeps=50 if is_acc else 500,
                      driver_elapsed_s=metrics["elapsed_s"], payload_wall_s=payload["wall_s"],
                      property_record=comparison["property_record"])
        code = 0
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr)
        code = 1
    finally:
        if weighted is not None:
            result["weighted_counters"] = dict(weighted.counters)
        for owner, name, original in reversed(restorations):
            setattr(owner, name, original)
        restored = 0
        for binding in reversed(bindings):
            try:
                binding.restore()
                restored += 1
            except Exception as error:
                result.update(status="RESTORE_FAILURE", restore_error=str(error))
                code = 1
        result.update(bindings_restored=restored, exit_code=code,
                      wall_s=time.perf_counter() - started,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write_new(output / "RESULT.json", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
