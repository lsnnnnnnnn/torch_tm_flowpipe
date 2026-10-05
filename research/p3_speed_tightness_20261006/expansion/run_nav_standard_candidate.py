#!/usr/bin/env python3
"""NAV standard full640 x 600 candidate: private outputs + Horner + weighted256.

The Oct 5 single-row fused adapter is inapplicable to 640 rows. This candidate
keeps the original refinement loop, both rounds, rejection paths and owned
outputs; only its existing graph chunk/padding adapter is selected. Smaller
padding increases chunk dispatches here, so speed is an empirical question.
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
REFERENCE = RUNS / "nav_author_standard_working_p3_full30_001"
SNAPSHOT = RUNS / "nav_author_standard_working_p3_full30_source_001"
RUNTIME_FIELDS = {
    "config", "run_id", "wall_s", "elapsed_s", "ended_utc", "max_rss_kib",
    "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "peak_allocated_bytes",
    "peak_reserved_bytes", "cuda_graph_captures", "cuda_graph_hits",
}


def compare_saved(reference, candidate):
    expected_bytes = 640 * 600 * 136
    original = (reference / "ranges.bin").read_bytes()
    actual = (candidate / "ranges.bin").read_bytes()
    if len(original) != expected_bytes or len(actual) != expected_bytes:
        raise RuntimeError("NAV range sequence must contain all 640 x 600 records")
    if original != actual:
        first = next(i for i, (a, b) in enumerate(zip(original, actual)) if a != b)
        raise RuntimeError(f"NAV range bytes differ at record {first // 136 + 1}")
    before = (reference / "observations.jsonl").read_bytes().splitlines(keepends=True)
    after = (candidate / "observations.jsonl").read_bytes().splitlines(keepends=True)
    if len(before) != 600 or len(after) != 600:
        raise RuntimeError("NAV observation sequence must contain all 600 steps")
    for step, (old, new) in enumerate(zip(before, after), 1):
        if old != new:
            raise RuntimeError(f"NAV observation differs at step {step}")
        row = json.loads(new)
        if (row["substep"] != step or row["accepted_boxes"] != 640
                or row["rejected_lanes"] or row["saved_range_records"] != 640
                or row["solver_status_counts"] != {"0": 640}
                or row["obstacle_intersecting_saved_tubes"] != 0):
            raise RuntimeError(f"NAV completion/property record differs at step {step}")
    for name in ("initial_boxes.json", "config.yaml"):
        if (reference / name).read_bytes() != (candidate / name).read_bytes():
            raise RuntimeError(f"NAV original contract/ledger changed: {name}")
    previous_argv = json.loads((reference / "START.json").read_text())["driver_argv"]
    current_argv = json.loads((candidate / "START.json").read_text())["driver_argv"]
    old_data = str(Path(previous_argv[1]).parent)
    if previous_argv != [value.replace(str(candidate), old_data) for value in current_argv]:
        raise RuntimeError("NAV shared driver/arguments changed beyond output paths")
    compared = {}
    for name in ("RESULT.json", "metrics.json"):
        previous, current = (json.loads((p / name).read_text()) for p in (reference, candidate))
        fields = set(previous) - RUNTIME_FIELDS
        if fields != set(current) - RUNTIME_FIELDS:
            raise RuntimeError(f"NAV scientific field set differs: {name}")
        for field in sorted(fields):
            if json.dumps(previous[field], sort_keys=True) != json.dumps(current[field], sort_keys=True):
                raise RuntimeError(f"NAV scientific field differs: {name}/{field}")
        compared[name] = sorted(fields)
    result = json.loads((candidate / "RESULT.json").read_text())
    metrics = json.loads((candidate / "metrics.json").read_text())
    if (result["status"] != "completed" or result["observed_substeps"] != 600
            or result["accepted_lane_substeps"] != 384000 or result["range_records"] != 384000
            or result["full_horizon_covered"] is not True or result["full_initial_set_covered"] is not True
            or result["author_verdict_lines"] != ["VERIFIED"]
            or result["end_to_end_floating_point_nn_certificate"] is not False
            or metrics["B"] != 640 or metrics["order"] != 3 or metrics["broken"] != 0
            or metrics["steps"] != 30 or metrics["substeps"] != 20):
        raise RuntimeError("NAV full numerical/property record incomplete")
    return {"reference": str(reference), "range_bytes_equal": expected_bytes,
            "range_records_equal": 384000, "observation_lines_byte_equal": 600,
            "initial_box_ledger_and_config_byte_equal": True, "equal_fields": compared,
            "excluded_runtime_fields": sorted(RUNTIME_FIELDS),
            "driver_argv_equal_after_output_path_relocation": True,
            "old_checker_executed": False, "digest_operations": 0,
            "scope": "all saved per-box four-state tube/endpoint ranges and acceptance/property records; hidden TM/SR not compared"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        raise RuntimeError("launch the new candidate with python -B")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or set(os.sched_getaffinity(0)) != {6, 7, 8, 9}:
        raise RuntimeError("frozen NAV profile requires physical GPU1 and CPUs6-9")
    if json.loads((args.gate / "RESULT.json").read_text()).get("status") != "PASSED_NEW_PRIVATE_OUTPUT_GPU_GATE":
        raise RuntimeError("the Oct 5 private-output GPU gate must have passed")
    output, source, reference, adapters = (p.resolve() for p in
        (args.output, args.source_snapshot, args.reference, args.adapters))
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, reference, adapters)):
        raise RuntimeError("candidate output must be separate from frozen sources/evidence/adapters")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sys.path.insert(0, str(adapters))
    from run_quad_candidate import guard_digests, reject_digest, write_new
    from run_nav_candidate import load_frozen, SOURCES
    from private_outputs import install as install_private
    from horner_edge import install as install_horner
    from weighted_chunk256 import install as install_weighted
    write_new(output / "START.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "instance": "nav-standard", "reference": str(reference), "source_snapshot": str(source),
        "adapters": str(adapters), "gpu_physical": 1, "cpu_affinity": [6, 7, 8, 9],
        "initial_boxes": 640, "ode_substeps": 600, "expected_range_bytes": 52224000,
        "changes": ["private output allocation", "preloaded Horner binding", "original two-round weighted refinement with 256-row chunks"],
        "fused": False,
        "non_fused_reason": "qualified single-row fused adapter requires B=1; keep original per-round screening, diagnostics and statistics for B=640",
        "dispatch_tradeoff": "when all 640 eligible, 256+256+128 padded to 768 total rows per round, versus two 512-row buffers; three graph calls instead of two, so speed is not assumed",
        "no_jit": True, "no_digest_operations": True, "dont_write_bytecode": True,
        "old_checker_executed": False,
        "timing_policy": "wrapper includes staging/imports/gates and direct saved comparison; payload excludes early Torch import; original internal driver loop and new supervisor full-process wall remain separate",
        "qualification": "new full implementation candidate; no independent floating-point NNCS certificate"})
    bindings, restorations, weighted = [], [], None
    result, code = {"status": "exception", "end_to_end_strict_certificate": False}, 1
    try:
        guard_digests()
        import torch
        import torch.utils.cpp_extension as extension
        torch.jit.script = lambda function=None, **kwargs: function if function is not None else lambda value: value
        torch.jit.trace = torch.compile = reject_digest
        extension.load = extension.load_inline = reject_digest
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=False)
        staged = {}
        for name in SOURCES:
            path = source / (name + ".py")
            raw = path.read_bytes()
            destination = snapshot / path.name
            with destination.open("xb") as stream:
                stream.write(raw)
            if destination.read_bytes() != raw:
                raise RuntimeError(f"staged NAV source differs: {name}")
            staged[name] = {"source": str(path), "bytes": len(raw), "byte_equal": True}
        write_new(output / "STAGED_INPUTS.json", staged)
        baseline, p3 = load_frozen(snapshot)
        if (baseline.BOXES, baseline.PERIODS, baseline.SUBSTEPS) != (640, 30, 600):
            raise RuntimeError("wrong NAV frozen source profile")
        original_prepare = p3.prepare_p3

        def prepare(backend):
            nonlocal weighted
            if bindings:
                raise RuntimeError("candidate prepare must execute exactly once")
            prepared = original_prepare(backend)
            runtime, driver = prepared[:2]
            from flowstar_gpu import cuda_kernels, private_output_kernels, sparse_exec
            from flowstar_gpu import horner_edge_kernels, weighted_validation
            private = install_private(runtime, cuda_kernels, private_output_kernels)
            bindings.append(private)
            write_new(output / "PRIVATE_OUTPUT_BINDING.json", private.receipt)
            weighted = install_weighted(runtime, weighted_validation, sparse_exec)
            bindings.append(weighted)
            write_new(output / "WEIGHTED_BINDING.json", {
                "policy": weighted.policy, "source_path": weighted.source_path,
                "weighted_path": weighted.weighted_path, "sparse_exec_path": weighted.sparse_exec_path,
                "chunk_size": 256, "rounds": 2, "instance_batch": 640,
                "scope": "new NAV batch qualification; no old checker executed"})
            old_engine = driver.SparseEngine

            def engine(*a, **kw):
                value = old_engine(*a, **kw)
                binding = install_horner(runtime, value, sparse_exec, horner_edge_kernels)
                bindings.append(binding)
                write_new(output / "HORNER_EDGE_BINDING.json", binding.receipt)
                return value

            driver.SparseEngine = engine
            restorations.append((driver, "SparseEngine", old_engine))
            return prepared

        p3.prepare_p3 = prepare
        restorations.append((p3, "prepare_p3", original_prepare))
        code = baseline.run(output / "data")
        if code or len(bindings) != 3 or not weighted.counters["graph_map_calls"]:
            raise RuntimeError("NAV full candidate failed or did not consume all bindings")
        write_new(output / "SAVED_COMPARISON.json", compare_saved(reference, output / "data"))
        metrics = json.loads((output / "data/metrics.json").read_text())
        payload = json.loads((output / "data/RESULT.json").read_text())
        result.update(status="COMPLETED_SAVED_OUTPUT_EQUIVALENT", completed_substeps=600,
                      accepted_lane_substeps=384000, driver_elapsed_s=metrics["elapsed_s"],
                      payload_wall_s=payload["wall_s"])
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
