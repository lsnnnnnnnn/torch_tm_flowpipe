#!/usr/bin/env python3
"""New NAV robust P3 run: private outputs plus the preloaded Horner binding.

The frozen full30 driver retains strict injection, all 600 observations, the
25-box initial set and both property checks. Only saved ranges/observations
and numerical final fields are compared; timing metadata is never compared.
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

from horner_edge import install as install_horner
from private_outputs import install as install_private
from run_quad_candidate import guard_digests, reject_digest, write_new


ROOT = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
RUNS = ROOT / "runs/archcomp26_20261001"
REFERENCE = RUNS / "nav_author_robust_working_p3_full30_001"
SNAPSHOT = RUNS / "nav_author_robust_working_p3_full30_source_001"
SOURCES = (
    "archcomp26_dp_p3_nohash", "archcomp26_tora_remain_author_nohash",
    "archcomp26_nav_author_gpu_smoke_nohash",
    "archcomp26_nav_author_p3_smoke_nohash",
    "archcomp26_nav_author_p3_full30_nohash",
)
RESULT_FIELDS = (
    "status", "instance", "method", "full_horizon_covered",
    "end_to_end_floating_point_nn_certificate", "driver_return",
    "observed_substeps", "accepted_lane_substeps", "range_records",
    "expected_substeps", "initial_boxes", "expected_lane_substeps",
    "author_verdict_lines", "last_endpoint_union", "full_initial_set_covered",
)


def load_frozen(snapshot):
    modules = {}
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
        modules[name] = module
    return modules[SOURCES[-1]], modules[SOURCES[-2]]


def compare_saved(reference, candidate):
    expected_bytes = 25 * 600 * 136
    original = (reference / "ranges.bin").read_bytes()
    actual = (candidate / "ranges.bin").read_bytes()
    if len(original) != expected_bytes or len(actual) != expected_bytes:
        raise RuntimeError("NAV ranges do not contain all 25 x 600 records")
    if original != actual:
        first = next(i for i, (a, b) in enumerate(zip(original, actual)) if a != b)
        raise RuntimeError(f"NAV range bytes differ at record {first // 136 + 1}")
    before = (reference / "observations.jsonl").read_bytes().splitlines(keepends=True)
    after = (candidate / "observations.jsonl").read_bytes().splitlines(keepends=True)
    if len(before) != 600 or len(after) != 600:
        raise RuntimeError("NAV observation sequence is incomplete")
    for step, (old, new) in enumerate(zip(before, after), 1):
        if old != new:
            raise RuntimeError(f"NAV observation differs at step {step}")
        row = json.loads(new)
        if (row["substep"] != step or row["accepted_boxes"] != 25
                or row["rejected_lanes"] or row["saved_range_records"] != 25
                or row["solver_status_counts"] != {"0": 25}
                or row["obstacle_intersecting_saved_tubes"] != 0):
            raise RuntimeError(f"NAV saved completion/property mismatch at step {step}")
    previous = json.loads((reference / "RESULT.json").read_text())
    result = json.loads((candidate / "RESULT.json").read_text())
    for field in RESULT_FIELDS:
        if previous[field] != result[field]:
            raise RuntimeError(f"NAV final field differs: {field}")
    if (result["status"] != "completed" or result["observed_substeps"] != 600
            or result["accepted_lane_substeps"] != 15000
            or result["author_verdict_lines"] != ["VERIFIED"]):
        raise RuntimeError("NAV full numerical/property record is incomplete")
    if (reference / "initial_boxes.json").read_bytes() != (candidate / "initial_boxes.json").read_bytes():
        raise RuntimeError("NAV initial-box ledger changed")
    return {"reference": str(reference), "range_bytes_equal": expected_bytes,
            "range_records_equal": 15000, "observation_lines_byte_equal": 600,
            "initial_box_ledger_byte_equal": True, "final_fields_equal": list(RESULT_FIELDS),
            "digest_operations": 0,
            "scope": "all saved four-state tube/endpoint ranges and acceptance/property records; hidden TM/SR not compared"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fast",), required=True)
    parser.add_argument("--source-snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or set(os.sched_getaffinity(0)) != {6, 7, 8, 9}:
        raise RuntimeError("frozen NAV profile requires physical GPU1 and CPUs6-9")
    gate = json.loads((args.gate / "RESULT.json").read_text())
    if gate.get("status") != "PASSED_NEW_PRIVATE_OUTPUT_GPU_GATE":
        raise RuntimeError("new private-output GPU gate has not passed")
    output, snapshot, reference = (p.resolve() for p in
                                   (args.output, args.source_snapshot, args.reference))
    if output.is_relative_to(snapshot) or output.is_relative_to(reference):
        raise RuntimeError("candidate output must be outside frozen evidence")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_new(output / "START.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "mode": args.mode, "source_snapshot": str(snapshot), "reference": str(reference),
        "gate": str(args.gate), "gpu": "1", "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "scope": "new NAV robust implementation candidate; archived reference is read-only",
        "changes": ["private output allocation", "preloaded Horner remainder binding"],
        "runtime_guards": "all digests and compilation prohibited; import-time script decorators remain eager Python",
        "no_jit": True, "no_digest_operations": True})
    result = {"status": "exception", "end_to_end_strict_certificate": False}
    bindings, code = [], 1
    try:
        guard_digests()
        import torch
        import torch.utils.cpp_extension as extension
        # Keep import-time scripted helpers as eager Python; never compile them.
        def eager_script(function=None, **kwargs):
            return function if function is not None else lambda value: value
        torch.jit.script = eager_script
        torch.jit.trace = torch.compile = reject_digest
        extension.load = extension.load_inline = reject_digest
        baseline, p3 = load_frozen(snapshot)
        original_prepare = p3.prepare_p3

        def prepare(backend):
            prepared = original_prepare(backend)
            runtime, driver = prepared[:2]
            from flowstar_gpu import cuda_kernels as ck, private_output_kernels
            from flowstar_gpu import sparse_exec, horner_edge_kernels
            private = install_private(runtime, ck, private_output_kernels)
            bindings.append(private)
            write_new(output / "PRIVATE_OUTPUT_BINDING.json", private.receipt)
            original_engine = driver.SparseEngine

            def create_engine(*a, **kw):
                engine = original_engine(*a, **kw)
                horner = install_horner(runtime, engine, sparse_exec, horner_edge_kernels)
                bindings.append(horner)
                write_new(output / "HORNER_EDGE_BINDING.json", horner.receipt)
                return engine

            driver.SparseEngine = create_engine
            return prepared

        p3.prepare_p3 = prepare
        code = baseline.run(output / "data", "nav-robust")
        if code:
            raise RuntimeError(f"NAV numerical process returned {code}")
        if len(bindings) != 2:
            raise RuntimeError("candidate did not consume both bindings exactly once")
        comparison = compare_saved(reference, output / "data")
        write_new(output / "SAVED_COMPARISON.json", comparison)
        result.update(status="COMPLETED_SAVED_OUTPUT_EQUIVALENT",
                      completed_substeps=600, accepted_lane_substeps=15000)
        code = 0
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr)
        code = 1
    finally:
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
