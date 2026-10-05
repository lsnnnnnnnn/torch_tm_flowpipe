#!/usr/bin/env python3
"""One isolated paper-QUAD attempt using the preloaded private-output binding."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from private_outputs import install


ROOT = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = ROOT / "runs/archcomp26_20261001/quad_paper_p3_nohash_v1"


def reject_digest(*args, **kwargs):
    raise RuntimeError("all content digest operations are prohibited")


def guard_digests():
    for name in ("new", "file_digest", "md5", "sha1", "sha224", "sha256", "sha384",
                 "sha512", "sha3_224", "sha3_256", "sha3_384", "sha3_512",
                 "shake_128", "shake_256", "blake2b", "blake2s"):
        if hasattr(hashlib, name):
            setattr(hashlib, name, reject_digest)


def load_baseline():
    guard_digests()
    path = OLD / "archcomp26_quad_paper_p3_nohash.py"
    spec = importlib.util.spec_from_file_location("saved_paper_quad_nohash", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compare_saved(output, mode):
    reference = OLD / {"batch2": "batch2_001", "full50": "full50_001"}[mode] / "data"
    expected = 40 if mode == "batch2" else 1000
    original = (reference / "observations.jsonl").read_bytes().splitlines(keepends=True)
    candidate = (output / "observations.jsonl").read_bytes().splitlines(keepends=True)
    if len(original) != expected or len(candidate) != expected:
        raise RuntimeError("incomplete saved observation sequence")
    for number, (before, after) in enumerate(zip(original, candidate), 1):
        if before != after:
            raise RuntimeError(f"first saved observation byte difference at substep {number}")
        row = json.loads(after)
        if (row["substep"] != number or row["accepted_count"] != 1024
                or row["rejected_lanes"] or row["status_counts"] != {"0": 1024}):
            raise RuntimeError(f"saved acceptance/status difference at substep {number}")
    previous = json.loads((reference / "RESULT.json").read_text())
    actual = json.loads((output / "RESULT.json").read_text())
    fields = ["completed_substeps", "expected_substeps", "expected_lane_substeps",
              "accepted_lane_substeps", "all_lanes_accepted", "nn_calls",
              "control_refresh_steps", "sr_length", "sr_epoch", "metrics_broken",
              "final_hull", "end_to_end_strict_certificate"]
    for field in fields:
        if previous[field] != actual[field]:
            raise RuntimeError(f"saved final field differs: {field}")
    return {"reference": str(reference), "observation_lines_byte_equal": expected,
            "final_fields_equal": fields, "digest_operations": 0,
            "scope": "all saved pooled tube/endpoint rows and final fields; no per-lane TM identity claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch2", "full50"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3" or set(os.sched_getaffinity(0)) != {14, 15, 16, 17}:
        raise RuntimeError("requires physical GPU3 and CPUs14-17")
    gate = json.loads((args.gate / "RESULT.json").read_text())
    if gate.get("status") != "PASSED_NEW_PRIVATE_OUTPUT_GPU_GATE":
        raise RuntimeError("new GPU gate has not passed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_new(output / "START.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "scope": "new implementation candidate; old experiment is read-only",
        "mode": args.mode, "gate": str(args.gate), "baseline_runner": str(OLD),
        "gpu": "3", "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "no_jit": True, "no_digest_operations": True})
    result = {"status": "exception", "end_to_end_strict_certificate": False}
    binding = None
    code = 1
    try:
        baseline = load_baseline()
        saved_prepare = baseline.prepare

        def prepare(batch):
            nonlocal binding
            prepared = saved_prepare(batch)
            torch, driver = prepared[:2]
            from flowstar_gpu import cuda_kernels as ck, private_output_kernels
            binding = install(torch, ck, private_output_kernels)
            write_new(output / "OPTIMIZATION.json", binding.receipt)
            original_main = driver.main

            def main_with_first_refusal():
                # execute() has now installed its observer; retain its refusal row.
                observed_advance = driver.advance_sparse

                def advance(*a, **kw):
                    state, accepted = observed_advance(*a, **kw)
                    if not bool(accepted.all()):
                        raise RuntimeError("first rejected lane after preserved observation")
                    return state, accepted

                driver.advance_sparse = advance
                return original_main()

            driver.main = main_with_first_refusal
            return prepared

        baseline.prepare = prepare
        code = baseline.execute(argparse.Namespace(mode=args.mode,
            source_config=OLD / "quad_paper.yaml", output=output / "data"))
        if code:
            raise RuntimeError(f"candidate numerical process returned {code}")
        comparison = compare_saved(output / "data", args.mode)
        write_new(output / "SAVED_COMPARISON.json", comparison)
        result.update(status="COMPLETED_SAVED_OUTPUT_EQUIVALENT", exit_code=0,
                      completed_substeps=comparison["observation_lines_byte_equal"],
                      accepted_lane_substeps=1024 * comparison["observation_lines_byte_equal"])
        code = 0
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr)
        code = 1
    finally:
        if binding is not None:
            try:
                binding.restore()
                result["binding_restored"] = True
            except Exception as error:
                result.update(status="RESTORE_FAILURE", restore_error=str(error))
                code = 1
        result.update(exit_code=code, wall_s=time.perf_counter() - started,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write_new(output / "RESULT.json", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
