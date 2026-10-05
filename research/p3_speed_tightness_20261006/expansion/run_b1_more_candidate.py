#!/usr/bin/env python3
"""New full Docking or named two-state SP run using the Oct 5 B=1 adapters.

SP retains its actual order-two/native-f64/reference injection path. Docking
retains order three, strict injection and its original UNKNOWN radial label.
All saved ranges/property observations must match; this is no new NNCS proof.
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
SOURCE_PATHS = {
    "docking": {
        "archcomp26_dp_p3_nohash": RUNS / "p3_attitude_avoid_v1/archcomp26_dp_p3_nohash.py",
        "archcomp26_tora_remain_author_nohash": RUNS / "author_tora_remain_v1/archcomp26_tora_remain_author_nohash.py",
        "archcomp26_docking_author_nohash": RUNS / "archcomp26_docking_author_nohash.py",
        "expansion_docking_frozen": RUNS / "archcomp26_docking_p3_nohash.py",
    },
    "single-pendulum": {
        "expansion_sp_frozen": RUNS / "single_pendulum_prep_001/archcomp26_sp_two_state_p3_nohash.py",
    },
}
REFERENCES = {
    "docking": RUNS / "docking_p3_full40_001/detail",
    "single-pendulum": RUNS / "sp_two_state_fourway_campaign_20261002_001/later05_ours_p3/data",
}
RUNTIME_FIELDS = {
    "config", "wall_s", "elapsed_s", "driver_elapsed_s", "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
    "cuda_graph_captures", "cuda_graph_hits",
}


def compare_saved(instance, reference, data):
    is_sp = instance == "single-pendulum"
    steps, safety_events, order = (100, 50, 2) if is_sp else (400, 400, 3)
    sizes = {}
    for name, count in (("ranges.jsonl", steps), ("safety.jsonl", safety_events)):
        old, new = (reference / name).read_bytes(), (data / name).read_bytes()
        if old != new or len(new.splitlines()) != count:
            raise RuntimeError(f"saved output differs or is incomplete: {name}")
        sizes[name] = len(new)
    if (reference / "config.yaml").read_bytes() != (data / "config.yaml").read_bytes():
        raise RuntimeError("generated contract bytes changed")
    old_argv = json.loads((reference / "START.json").read_text())["driver_argv"]
    new_argv = json.loads((data / "START.json").read_text())["driver_argv"]
    old_data = str(Path(old_argv[1]).parent)
    if old_argv != [value.replace(str(data), old_data) for value in new_argv]:
        raise RuntimeError("driver or arguments changed beyond output paths")
    compared = {}
    for name in ("RESULT.json", "metrics.json"):
        old, new = (json.loads((folder / name).read_text()) for folder in (reference, data))
        fields = set(old) - RUNTIME_FIELDS
        if fields != set(new) - RUNTIME_FIELDS:
            raise RuntimeError(f"scientific field set differs: {name}")
        for key in sorted(fields):
            if json.dumps(old[key], sort_keys=True) != json.dumps(new[key], sort_keys=True):
                raise RuntimeError(f"scientific field differs: {name}/{key}")
        compared[name] = sorted(fields)
    metrics = json.loads((data / "metrics.json").read_text())
    result = json.loads((data / "RESULT.json").read_text())
    if (metrics["B"] != 1 or metrics["order"] != order or metrics["broken"] != 0
            or metrics["steps"] * metrics["substeps"] != steps
            or result["status"] != "completed" or result["safety_events"] != safety_events
            or result["end_to_end_floating_point_nn_certificate"] is not False):
        raise RuntimeError("full original numerical contract not completed")
    if is_sp:
        if (result["completed_substeps"] != 100 or result["all_substeps_accepted"] is not True
                or result["author_safe_bounds_all_nonpositive"] is not True
                or result["independent_window_tube_boxes_safe"] is not True):
            raise RuntimeError("SP two-state window record incomplete")
    elif (result["observed_substeps"] != 400 or result["accepted_substeps"] != 400
          or result["checker_verdict"] != "UNKNOWN_REPORTED_BY_AUTHOR_CHECKER"):
        raise RuntimeError("Docking completion or UNKNOWN boundary changed")
    return {"reference": str(reference), "saved_bytes_equal": sizes, "equal_fields": compared,
            "completed_substeps": steps, "B": 1, "order": order,
            "excluded_runtime_fields": sorted(RUNTIME_FIELDS), "config_bytes_equal": True,
            "driver_argv_equal_after_output_path_relocation": True,
            "property_basis": "original property rows and labels retained; no old checker rerun",
            "scope": "all saved ranges/property rows and scientific fields; hidden TM/SR not compared",
            "old_checker_executed": False, "digest_operations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=tuple(SOURCE_PATHS), required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weighted-mode", choices=("small", "fused"), default="fused")
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        raise RuntimeError("launch this new candidate with python -B")
    is_sp = args.instance == "single-pendulum"
    gpu, cpus = ("2", {10, 11, 12, 13}) if is_sp else ("3", {14, 15, 16, 17})
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu or set(os.sched_getaffinity(0)) != cpus:
        raise RuntimeError(f"preserve saved GPU{gpu}/CPU{sorted(cpus)} profile")
    if json.loads((args.gate / "RESULT.json").read_text()).get("status") != "PASSED_NEW_PRIVATE_OUTPUT_GPU_GATE":
        raise RuntimeError("the Oct 5 private-output GPU gate must have passed")
    output, adapters = args.output.resolve(), args.adapters.resolve()
    reference = (args.reference or REFERENCES[args.instance]).resolve()
    protected = [reference, adapters, *[p.resolve() for p in SOURCE_PATHS[args.instance].values()]]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
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
    data, snapshot = output / "data", output / "source_snapshot"
    write_new(output / "START.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "instance": args.instance, "reference": str(reference), "adapters": str(adapters),
        "gpu_physical": int(gpu), "cpu_affinity": sorted(cpus), "B": 1,
        "complete_substeps": 100 if is_sp else 400, "working_order": 2 if is_sp else 3,
        "contract": ("named two-physical-state SP, native-f64, original injection; closed [.5,1] window"
                     if is_sp else "Docking full initial box, strict injection, rpc-float32; radial UNKNOWN retained"),
        "changes": ["private output allocation", "preloaded Horner binding", args.weighted_mode + " single-row weighted refinement"],
        "weighted_mode": args.weighted_mode, "no_jit": True, "no_digest_operations": True,
        "dont_write_bytecode": True, "old_checker_executed": False,
        "timing_policy": "wrapper includes staging/imports/gates; payload excludes early Torch import; compare unchanged internal driver and separate full supervisor wall",
        "qualification": "new full implementation candidate; no independent floating-point NNCS certificate"})
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
        snapshot.mkdir(exist_ok=False)
        copied = {}
        for module_name, path in SOURCE_PATHS[args.instance].items():
            raw, destination = path.read_bytes(), snapshot / path.name
            with destination.open("xb") as stream:
                stream.write(raw)
            if destination.read_bytes() != raw:
                raise RuntimeError(f"staged source bytes differ: {path}")
            copied[module_name] = {"source": str(path), "staged": str(destination), "bytes": len(raw), "byte_equal": True}
        write_new(output / "STAGED_INPUTS.json", copied)
        for module_name, path in SOURCE_PATHS[args.instance].items():
            baseline = load_module(module_name, snapshot / path.name)
        original_prepare = baseline.prepare if is_sp else baseline.prepare_p3

        def prepare(*a, **kw):
            nonlocal weighted
            if bindings:
                raise RuntimeError("prepare must execute exactly once")
            prepared = original_prepare(*a, **kw)
            runtime, driver = prepared[1:3] if is_sp else prepared[:2]
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
                        raise ArithmeticError("first rejected lane after preserved observation")
                    return state, accepted

                driver.advance_sparse = advance
                try:
                    return old_main()
                finally:
                    driver.advance_sparse = observed

            driver.main = main_with_first_rejection
            restorations.append((driver, "main", old_main))
            return prepared

        if is_sp:
            baseline.prepare = prepare
            code = baseline.run(argparse.Namespace(mode="full", output=data,
                source_config=RUNS / "single_pendulum_prep_001/single_pendulum_paper_two_state.yaml"))
        else:
            baseline.prepare_p3 = prepare
            previous_argv = sys.argv
            try:
                sys.argv = [str(snapshot / "archcomp26_docking_p3_nohash.py"), "--mode", "full", "--output", str(data)]
                code = baseline.main()
            finally:
                sys.argv = previous_argv
        if code or len(bindings) != 3 or not weighted.counters["graph_map_calls"]:
            raise RuntimeError("full candidate failed or did not consume all bindings")
        if args.weighted_mode == "fused" and (not weighted.counters["fast_calls"]
                or weighted.counters["reference_checks"] != 1 or weighted.counters["candidate_exceptions"]):
            raise RuntimeError("fused candidate needs one checked fast result and zero exceptions")
        write_new(output / "SAVED_COMPARISON.json", compare_saved(args.instance, reference, data))
        metrics = json.loads((data / "metrics.json").read_text())
        payload = json.loads((data / "RESULT.json").read_text())
        result.update(status="COMPLETED_SAVED_OUTPUT_EQUIVALENT", B=1,
                      completed_substeps=100 if is_sp else 400,
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
        result.update(bindings_restored=restored, exit_code=code, wall_s=time.perf_counter() - started,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write_new(output / "RESULT.json", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
