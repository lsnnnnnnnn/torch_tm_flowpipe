#!/usr/bin/env python3
"""New full Attitude or TORA tanh run with the qualified Oct 5 B=1 adapters.

Frozen runners are staged byte-for-byte. Tanh artifact paths alone are
relocated. The complete saved ranges, observations, contract and scientific
result/metrics fields must match the existing full reference. No old checker
is invoked; equivalent saved outputs are not an independent NNCS proof.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
RUNS = ROOT / "runs/archcomp26_20261001"
ATTITUDE = RUNS / "attitude_corrected_fourway_campaign_20261002_001"
TANH = RUNS / "tora_reach_tanh_official2026_mat_u11_workingp3_full500_20261002_001"
SOURCE_FILES = {
    "attitude": (
        "archcomp26_dp_p3_nohash.py", "archcomp26_tora_remain_author_nohash.py",
        "archcomp26_attitude_avoid_author_nohash.py",
        "archcomp26_attitude_p3_gpu2_campaign_nohash.py",
    ),
    "tora-tanh": (
        "run_full.py", "author_support.py", "archcomp26_dp_p3_nohash.py",
        "archcomp26_tora_remain_author_nohash.py", "archcomp26_tora_remain_p3_nohash.py",
        "controller_plant_u.onnx", "nn_tora_relu_tanh.mat",
    ),
}
RUNTIME_FIELDS = {
    "config", "wall_s", "elapsed_s", "driver_elapsed_s",
    "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
    "peak_allocated_bytes", "peak_reserved_bytes", "cuda_graph_captures", "cuda_graph_hits",
}


def stage(instance, source, target):
    target.mkdir(parents=True, exist_ok=False)
    copies = {}
    for name in SOURCE_FILES[instance]:
        raw = (source / name).read_bytes()
        with (target / name).open("xb") as stream:
            stream.write(raw)
        if (target / name).read_bytes() != raw:
            raise RuntimeError(f"staged source differs: {name}")
        copies[name] = len(raw)
    receipt = {"source": str(source), "target": str(target), "byte_equal_copies": copies}
    if instance == "tora-tanh":
        model = json.loads((source / "controller_plant_u.onnx.json").read_text())
        previous = str(Path(model["model_path"]).parent)
        if model["mat_path"] != str(Path(previous) / "nn_tora_relu_tanh.mat"):
            raise RuntimeError("TORA saved model receipt paths disagree")
        for name, count in (("config.yaml", 1), ("controller_plant_u.onnx.json", 2)):
            text = (source / name).read_text()
            if text.count(previous) != count:
                raise RuntimeError(f"unexpected TORA artifact path occurrences: {name}")
            with (target / name).open("x") as stream:
                stream.write(text.replace(previous, str(target)))
        receipt.update(path_only_relocation=["config.yaml", "controller_plant_u.onnx.json"],
                       old_artifact_root=previous, new_artifact_root=str(target))
    return receipt


def compare_saved(instance, reference, data):
    steps, record_bytes = (60, 216) if instance == "attitude" else (500, 152)
    sizes = {}
    for name in ("ranges.bin", "observations.jsonl"):
        old, new = (reference / name).read_bytes(), (data / name).read_bytes()
        if old != new:
            raise RuntimeError(f"saved bytes differ: {name}")
        if name == "ranges.bin" and len(new) != steps * record_bytes:
            raise RuntimeError("saved range sequence is incomplete")
        if name.endswith(".jsonl") and len(new.splitlines()) != steps:
            raise RuntimeError("saved observation sequence is incomplete")
        sizes[name] = len(new)
    previous = json.loads((reference / "START.json").read_text())
    current = json.loads((data / "START.json").read_text())
    old_argv, new_argv = previous["driver_argv"], current["driver_argv"]
    old_data = str(Path(old_argv[1]).parent)
    if old_argv != [value.replace(str(data), old_data) for value in new_argv]:
        raise RuntimeError("shared driver or arguments changed beyond output paths")
    original_config = (reference / "config.yaml").read_text()
    new_config = (data / "config.yaml").read_text()
    if instance == "tora-tanh":
        old_receipt = (reference / "controller_plant_u.onnx.json").read_text()
        old_root = str(Path(json.loads(old_receipt)["model_path"]).parent)
        new_config = new_config.replace(str(data), old_root)
        if old_receipt != (data / "controller_plant_u.onnx.json").read_text().replace(str(data), old_root):
            raise RuntimeError("TORA controller receipt changed beyond artifact paths")
        for name in SOURCE_FILES[instance]:
            if (reference / name).read_bytes() != (data / name).read_bytes():
                raise RuntimeError(f"TORA saved source/model changed: {name}")
    if original_config != new_config:
        raise RuntimeError("generated contract changed")
    fields = {}
    for name in ("RESULT.json", "metrics.json"):
        old = json.loads((reference / name).read_text())
        new = json.loads((data / name).read_text())
        selected = set(old) - RUNTIME_FIELDS
        if selected != set(new) - RUNTIME_FIELDS:
            raise RuntimeError(f"scientific field set differs: {name}")
        for key in sorted(selected):
            if json.dumps(old[key], sort_keys=True) != json.dumps(new[key], sort_keys=True):
                raise RuntimeError(f"scientific field differs: {name}/{key}")
        fields[name] = sorted(selected)
    result = json.loads((data / "RESULT.json").read_text())
    metrics = json.loads((data / "metrics.json").read_text())
    if (metrics["B"] != 1 or metrics["order"] != 3 or metrics["broken"] != 0
            or metrics["steps"] * metrics["substeps"] != steps
            or result["accepted_substeps"] != steps):
        raise RuntimeError("full numerical contract not completed")
    if instance == "attitude":
        if (result["status"] != "completed" or result["all_substeps_accepted"] is not True
                or result["saved_tubes_box_disjoint_official_unsafe"] is not True
                or result["end_to_end_strict_certificate"] is not False):
            raise RuntimeError("Attitude numerical/property record incomplete")
    elif (result["status"] != "completed_full_numerical_horizon"
          or result["property_evaluated"] is not False):
        raise RuntimeError("TORA numerical record or property boundary changed")
    return {"reference": str(reference), "saved_bytes_equal": sizes,
            "B": 1, "completed_substeps": steps, "equal_fields": fields,
            "excluded_runtime_fields": sorted(RUNTIME_FIELDS),
            "config_and_driver_argv_equal_after_path_relocation": True,
            "property_basis": "same original labels and complete saved observations; no new checker",
            "scope": "saved physical ranges and scientific observations/fields; hidden TM/SR not compared",
            "digest_operations": 0, "old_checker_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=tuple(SOURCE_FILES), required=True)
    parser.add_argument("--adapters", type=Path, required=True,
                        help="existing Oct 5 adapter directory; no files are modified")
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weighted-mode", choices=("small", "fused"), default="fused")
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        raise RuntimeError("launch this new candidate with python -B")
    is_attitude = args.instance == "attitude"
    gpu = "2" if is_attitude else "3"
    affinity = {10, 11, 12, 13} if is_attitude else {18, 19}
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu or set(os.sched_getaffinity(0)) != affinity:
        raise RuntimeError(f"preserve the saved GPU{gpu}/CPU{sorted(affinity)} profile")
    if json.loads((args.gate / "RESULT.json").read_text()).get("status") != "PASSED_NEW_PRIVATE_OUTPUT_GPU_GATE":
        raise RuntimeError("the Oct 5 private-output GPU gate must have passed")
    source = (args.source_snapshot or (ATTITUDE / "p3_source" if is_attitude else TANH)).resolve()
    reference = (args.reference or (ATTITUDE / "later05_ours_p3/payload" if is_attitude else TANH)).resolve()
    output = args.output.resolve()
    adapters = args.adapters.resolve()
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, reference, adapters)):
        raise RuntimeError("candidate output must be separate from saved sources/evidence/adapters")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sys.path.insert(0, str(adapters))
    from run_quad_candidate import guard_digests, reject_digest, write_new
    from run_small1_candidate import load_module
    from private_outputs import install as install_private
    from horner_edge import install as install_horner
    from weighted_small_batch import install as install_small
    from weighted_fused_single import install as install_fused
    data = output / "data"
    write_new(output / "START.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "instance": args.instance, "source_snapshot": str(source), "reference": str(reference),
        "adapter_sources": str(adapters), "gpu_physical": int(gpu), "cpu_affinity": sorted(affinity),
        "B": 1, "complete_substeps": 60 if is_attitude else 500,
        "changes": ["private output allocation", "preloaded Horner binding", args.weighted_mode + " single-row weighted refinement"],
        "weighted_mode": args.weighted_mode, "old_checker_executed": False,
        "dont_write_bytecode": True, "no_jit": True, "no_digest_operations": True,
        "timing_policy": "wrapper includes staging/imports/gates; original payload excludes early Torch import; use unchanged internal driver loop separately from full supervisor process wall",
        "qualification": "new full numerical candidate; no independent floating-point NNCS certificate"})
    bindings, restorations, weighted = [], [], None
    result = {"status": "exception", "end_to_end_strict_certificate": False}
    code = 1
    try:
        guard_digests()
        import torch
        import torch.utils.cpp_extension as extension
        torch.jit.script = lambda function=None, **kwargs: function if function is not None else lambda value: value
        torch.jit.trace = torch.compile = reject_digest
        extension.load = extension.load_inline = reject_digest
        staged = output / "source_snapshot" if is_attitude else data
        write_new(output / "STAGED_INPUTS.json", stage(args.instance, source, staged))
        for name in ("archcomp26_dp_p3_nohash", "archcomp26_tora_remain_author_nohash"):
            load_module(name, staged / (name + ".py"))

        def wrap_prepare(original):
            def prepare(*a, **kw):
                nonlocal weighted
                if bindings:
                    raise RuntimeError("prepare must execute exactly once")
                prepared = original(*a, **kw)
                runtime, driver = prepared[:2]
                from flowstar_gpu import cuda_kernels, private_output_kernels, sparse_exec
                from flowstar_gpu import horner_edge_kernels, weighted_validation
                private = install_private(runtime, cuda_kernels, private_output_kernels)
                bindings.append(private)
                write_new(output / "PRIVATE_OUTPUT_BINDING.json", private.receipt)
                weighted = (install_fused(runtime, weighted_validation, sparse_exec, check_first=1)
                            if args.weighted_mode == "fused" else
                            install_small(runtime, weighted_validation, sparse_exec, rows=1))
                bindings.append(weighted)
                write_new(output / "WEIGHTED_BINDING.json", weighted.receipt)
                old_engine = driver.SparseEngine

                def engine(*a, **kw):
                    value = old_engine(*a, **kw)
                    binding = install_horner(runtime, value, sparse_exec, horner_edge_kernels)
                    bindings.append(binding)
                    write_new(output / "HORNER_EDGE_BINDING.json", binding.receipt)
                    return value

                driver.SparseEngine = engine
                restorations.append((driver, "SparseEngine", old_engine))
                old_main = driver.main

                def main_with_first_rejection():
                    observed = driver.advance_sparse

                    def advance(*a, **kw):
                        state, accepted = observed(*a, **kw)
                        if not bool(accepted.all()):
                            raise ArithmeticError("first rejected lane after original saved observation")
                        return state, accepted

                    driver.advance_sparse = advance
                    try:
                        return old_main()
                    finally:
                        driver.advance_sparse = observed

                driver.main = main_with_first_rejection
                restorations.append((driver, "main", old_main))
                return prepared
            return prepare

        if is_attitude:
            load_module("archcomp26_attitude_avoid_author_nohash", staged / "archcomp26_attitude_avoid_author_nohash.py")
            baseline = load_module("expansion_attitude_frozen", staged / "archcomp26_attitude_p3_gpu2_campaign_nohash.py")
            baseline.prepare = wrap_prepare(baseline.prepare)
            code = baseline.run(argparse.Namespace(mode="full", output=data))
        else:
            baseline = load_module("expansion_tanh_frozen", staged / "run_full.py")
            original_load = baseline.load_module

            def load_with_prepare(name, path):
                module = original_load(name, path)
                if name == "archcomp26_tora_reach_p3_support":
                    module.prepare = wrap_prepare(module.prepare)
                return module

            baseline.load_module = load_with_prepare
            code = baseline.main()
        if code or len(bindings) != 3 or not weighted.counters["graph_map_calls"]:
            raise RuntimeError("full candidate failed or did not consume all bindings")
        if args.weighted_mode == "fused" and (not weighted.counters["fast_calls"]
                or weighted.counters["reference_checks"] != 1 or weighted.counters["candidate_exceptions"]):
            raise RuntimeError("fused candidate needs a checked fast result and zero exceptions")
        comparison = compare_saved(args.instance, reference, data)
        write_new(output / "SAVED_COMPARISON.json", comparison)
        metrics = json.loads((data / "metrics.json").read_text())
        payload = json.loads((data / "RESULT.json").read_text())
        result.update(status="COMPLETED_SAVED_OUTPUT_EQUIVALENT", B=1,
                      completed_substeps=60 if is_attitude else 500,
                      driver_elapsed_s=metrics["elapsed_s"], payload_wall_s=payload["wall_s"])
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
