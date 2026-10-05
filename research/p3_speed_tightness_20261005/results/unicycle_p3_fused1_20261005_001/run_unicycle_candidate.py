#!/usr/bin/env python3
"""New paper Unicycle run with private outputs, preloaded Horner and weighted1.

The frozen runner owns the full contract, strict injection/endpoint, all 500
saved eight-state tube/endpoint records and the T=10 property check.
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

sys.dont_write_bytecode = True

from horner_edge import install as install_horner
from private_outputs import install as install_private
from weighted_small_batch import install as install_weighted
from run_quad_candidate import guard_digests, reject_digest, write_new


ROOT = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = ROOT / "runs/archcomp26_20261001/unicycle_paper_speed_w_constant_v1"
SNAPSHOT = OLD / "p3_full_source"
REFERENCE = OLD / "p3_full50_001/payload"
SOURCES = ("archcomp26_dp_p3_nohash", "archcomp26_tora_remain_author_nohash",
           "archcomp26_unicycle_paper_author_nohash", "archcomp26_unicycle_paper_p3_nohash")
RESULT_FIELDS = (
    "status", "method", "mode", "profile", "driver_return", "observed_substeps",
    "expected_substeps", "accepted_substeps", "first_rejection_internal_status",
    "last_accepted_t_s", "terminal_physical_endpoint", "terminal_endpoint_in_target",
    "property_verdict", "end_to_end_floating_point_nn_certificate",
)
METRIC_FIELDS = (
    "engine", "coupling", "crown_transport", "crown_input_layout", "alpha_iters",
    "B", "order", "steps", "substeps", "ctrl_steps", "broken", "final_hull",
    "final_hull_width_sum_mean",
)


def load_frozen(snapshot):
    for name in SOURCES:
        if name in sys.modules:
            raise RuntimeError(f"frozen module already loaded: {name}")
        path = snapshot / (name + ".py")
        if not path.is_file():
            raise FileNotFoundError(path)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def compare_saved(reference, candidate):
    before = (reference / "ranges.jsonl").read_bytes().splitlines(keepends=True)
    after = (candidate / "ranges.jsonl").read_bytes().splitlines(keepends=True)
    if len(before) != 500 or len(after) != 500:
        raise RuntimeError("Unicycle range sequence does not contain all 500 steps")
    for step, (old, new) in enumerate(zip(before, after), 1):
        if old != new:
            raise RuntimeError(f"Unicycle saved range bytes differ at substep {step}")
        row = json.loads(new)
        if (row["substep"] != step or row["accepted"] is not True
                or row["solver_status_code"] != 0 or row["solver_status"] != "ACTIVE"
                or len(row["tube"]) != 8 or len(row["endpoint"]) != 8):
            raise RuntimeError(f"Unicycle saved acceptance/shape mismatch at substep {step}")
    if (reference / "config.yaml").read_bytes() != (candidate / "config.yaml").read_bytes():
        raise RuntimeError("Unicycle generated contract changed")
    for filename, fields in (("RESULT.json", RESULT_FIELDS), ("metrics.json", METRIC_FIELDS)):
        old = json.loads((reference / filename).read_text())
        new = json.loads((candidate / filename).read_text())
        for field in fields:
            # Serialized numeric fields also distinguish signed zero.
            if json.dumps(old[field], sort_keys=True) != json.dumps(new[field], sort_keys=True):
                raise RuntimeError(f"Unicycle {filename} field differs: {field}")
    result = json.loads((candidate / "RESULT.json").read_text())
    if (result["status"] != "completed" or result["accepted_substeps"] != 500
            or result["terminal_endpoint_in_target"] is not True
            or result["property_verdict"] != "ENDPOINT_SUFFICIENT_FOR_REACH"
            or result["end_to_end_floating_point_nn_certificate"] is not False):
        raise RuntimeError("Unicycle complete numerical/property record is invalid")
    return {"reference": str(reference), "range_lines_byte_equal": 500,
            "range_bytes_equal": sum(map(len, after)), "state_count": 8,
            "config_yaml_byte_equal": True, "final_fields_equal": list(RESULT_FIELDS),
            "metric_fields_equal": list(METRIC_FIELDS), "digest_operations": 0,
            "scope": "all saved eight-state tube/endpoint rows, control metrics and final numerical/property fields; hidden TM/SR not compared"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weighted-mode", choices=("small", "fused"), default="small")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3" or set(os.sched_getaffinity(0)) != {32, 33, 34, 35}:
        raise RuntimeError("frozen Unicycle profile requires physical GPU3 and CPUs32-35")
    gate = json.loads((args.gate / "RESULT.json").read_text())
    if gate.get("status") != "PASSED_NEW_PRIVATE_OUTPUT_GPU_GATE":
        raise RuntimeError("new private-output GPU gate has not passed")
    output, snapshot, reference = (p.resolve() for p in
                                   (args.output, args.source_snapshot, args.reference))
    if output.is_relative_to(snapshot) or output.is_relative_to(reference):
        raise RuntimeError("candidate output must be outside frozen source/reference")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_new(output / "START.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "source_snapshot": str(snapshot), "reference": str(reference), "gate": str(args.gate),
        "gpu": "3", "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "scope": "new full paper Unicycle implementation candidate; archived source/result read-only",
        "changes": ["private output allocation", "preloaded Horner remainder binding",
                    ("accepted weighted graph scratch: fixed one row" if args.weighted_mode == "small"
                     else "single-row two-round graph, success-only commit and original fallback")],
        "weighted_mode": args.weighted_mode,
        "fused_reference_check": ("first successful input: direct output bytes and full statistics; included in timers"
                                  if args.weighted_mode == "fused" else None),
        "contract": "one original eight-state box; w'=0 and w only in speed; 50 periods / 500 steps to T=10",
        "frozen_runner_note": "inherited payload/P3_METHOD.json has stale first-period scope text; full mode, START/RESULT and all 500 saved rows determine this run's scope",
        "timing_policy": "wrapper wall includes imports/preloads; baseline payload wall includes its preflight/prepare; driver elapsed excludes setup/preload/engine creation, includes lazy graph capture",
        "timing_qualification": "Torch is imported before the payload timer to install compilation guards; compare driver elapsed separately and do not treat old/new payload wall alone as matched timing",
        "runtime_guards": "all digests and compilation prohibited; script decorators remain eager Python",
        "no_jit": True, "no_digest_operations": True})
    result = {"status": "exception", "end_to_end_strict_certificate": False,
              "end_to_end_floating_point_nn_certificate": False}
    bindings, weighted, code = [], None, 1
    try:
        guard_digests()
        import torch
        import torch.utils.cpp_extension as extension

        def eager_script(function=None, **kwargs):
            return function if function is not None else lambda value: value

        torch.jit.script = eager_script
        torch.jit.trace = torch.compile = reject_digest
        extension.load = extension.load_inline = reject_digest
        baseline = load_frozen(snapshot)
        original_prepare = baseline.prepare_p3

        def prepare(backend):
            nonlocal weighted
            if bindings:
                raise RuntimeError("candidate prepare must be called exactly once")
            prepared = original_prepare(backend)
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
            original_engine = driver.SparseEngine

            def create_engine(*a, **kw):
                engine = original_engine(*a, **kw)
                horner = install_horner(runtime, engine, sparse_exec, horner_edge_kernels)
                bindings.append(horner)
                write_new(output / "HORNER_EDGE_BINDING.json", horner.receipt)
                return engine

            driver.SparseEngine = create_engine
            original_main = driver.main

            def main_with_first_refusal():
                # The frozen author runner has installed/flushed its range observer.
                observed_advance = driver.advance_sparse

                def advance(*a, **kw):
                    state, accepted = observed_advance(*a, **kw)
                    if not bool(accepted.all()):
                        raise RuntimeError("first rejected lane after preserved Unicycle observation")
                    return state, accepted

                driver.advance_sparse = advance
                return original_main()

            driver.main = main_with_first_refusal
            return prepared

        baseline.prepare_p3 = prepare
        previous = sys.argv
        sys.argv = [str(snapshot / (SOURCES[-1] + ".py")), "--mode", "full", "--output", str(output / "data")]
        try:
            code = baseline.main()
        finally:
            sys.argv = previous
        if code:
            raise RuntimeError(f"Unicycle numerical process returned {code}")
        if len(bindings) != 3 or not weighted.counters["graph_map_calls"]:
            raise RuntimeError("Unicycle candidate did not consume every selected binding")
        if args.weighted_mode == "fused" and (
                not weighted.counters["fast_calls"]
                or weighted.counters["reference_checks"] != 1
                or weighted.counters["candidate_exceptions"]):
            raise RuntimeError("fused graph requires a checked fast result and zero candidate exceptions")
        comparison = compare_saved(reference, output / "data")
        write_new(output / "SAVED_COMPARISON.json", comparison)
        metrics = json.loads((output / "data/metrics.json").read_text())
        payload = json.loads((output / "data/RESULT.json").read_text())
        result.update(status="COMPLETED_SAVED_OUTPUT_EQUIVALENT", completed_substeps=500,
                      accepted_substeps=500, driver_elapsed_s=metrics["elapsed_s"],
                      payload_wall_s=payload["wall_s"], terminal_endpoint_in_target=True)
        code = 0
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr)
        code = 1
    finally:
        if weighted is not None:
            result["weighted_counters"] = dict(weighted.counters)
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
