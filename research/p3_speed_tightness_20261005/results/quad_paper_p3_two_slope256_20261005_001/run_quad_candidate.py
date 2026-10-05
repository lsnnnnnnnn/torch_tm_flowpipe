#!/usr/bin/env python3
"""One isolated paper-QUAD attempt using the preloaded private-output binding."""

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
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


def load_baseline(source_runner=None):
    guard_digests()
    path = source_runner or OLD / "archcomp26_quad_paper_p3_nohash.py"
    spec = importlib.util.spec_from_file_location("saved_paper_quad_nohash", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # A new runner snapshot still reads the archived, unchanged no-digest adapters.
    module.NOHASH = OLD / "archcomp26_quad_p3_nohash"
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


def compare_widths(output, mode, destination):
    """Compare exact widths of saved binary64 bounds, without claiming containment."""
    reference = OLD / {"batch2": "batch2_001", "full50": "full50_001"}[mode] / "data"
    expected = 40 if mode == "batch2" else 1000
    before = [json.loads(line) for line in (reference / "observations.jsonl").read_text().splitlines()]
    after = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    if len(before) != expected or len(after) != expected:
        raise RuntimeError("width comparison requires complete saved sequences")
    fields = ["substep", "nominal_t_left", "nominal_t_right", "state", "geometry",
              "reference_lower", "reference_upper", "reference_width",
              "candidate_lower", "candidate_upper", "candidate_width",
              "width_difference", "reference_width_rational", "candidate_width_rational",
              "width_difference_rational", "classification", "candidate_subset_reference"]
    counts = dict(narrower=0, equal=0, wider=0)
    by_state_geometry = {}
    contained = 0
    with (destination / "ABSOLUTE_WIDTHS.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step, (old, new) in enumerate(zip(before, after), 1):
            for row in (old, new):
                if (row["substep"] != step or row["accepted_count"] != 1024
                        or row["rejected_lanes"] or row["status_counts"] != {"0": 1024}
                        or len(row["tube_endpoint_union_12x4"]) != 12):
                    raise RuntimeError(f"incomplete width source at substep {step}")
            for state, (original, candidate) in enumerate(zip(
                    old["tube_endpoint_union_12x4"], new["tube_endpoint_union_12x4"]), 1):
                if len(original) != 4 or len(candidate) != 4:
                    raise RuntimeError("expected twelve rows of tube/endpoint bounds")
                for geometry, column in (("tube", 0), ("endpoint", 2)):
                    lo, hi = original[column:column + 2]
                    cl, ch = candidate[column:column + 2]
                    if not all(math.isfinite(v) for v in (lo, hi, cl, ch)) or lo > hi or cl > ch:
                        raise RuntimeError("nonfinite or reversed saved interval")
                    rw, cw = Fraction(hi) - Fraction(lo), Fraction(ch) - Fraction(cl)
                    difference = cw - rw
                    relation = "narrower" if difference < 0 else "wider" if difference > 0 else "equal"
                    subset = cl >= lo and ch <= hi
                    counts[relation] += 1
                    key = f"x{state}:{geometry}"
                    by_state_geometry.setdefault(key, dict(narrower=0, equal=0, wider=0))[relation] += 1
                    contained += int(subset)
                    writer.writerow(dict(zip(fields, [step, (step - 1) * 0.005, step * 0.005,
                        f"x{state}", geometry, lo, hi, float(rw), cl, ch, float(cw), float(difference),
                        str(rw), str(cw), str(difference), relation, subset])))
    original_result = json.loads((reference / "RESULT.json").read_text())
    actual_result = json.loads((output / "RESULT.json").read_text())
    for field in ("completed_substeps", "accepted_lane_substeps", "nn_calls", "control_refresh_steps"):
        if original_result[field] != actual_result[field]:
            raise RuntimeError(f"width comparison changed coverage: {field}")
    return {"reference": str(reference), "candidate": str(output),
            "completed_substeps": expected, "width_rows": expected * 12 * 2,
            "counts": counts, "counts_by_state_geometry": by_state_geometry,
            "candidate_subset_reference_rows": contained,
            "width_definition": "exact upper minus lower of saved binary64 bounds pooled across all 1024 boxes",
            "classification": "exact rational width comparison; no tolerance or ratio ranking",
            "scope": "absolute widths of all 12 saved physical-state tube and endpoint projections; no per-lane or NNCS certificate",
            "control_width_note": "driver u_box_width semantics differ across CROWN modes and are not compared here",
            "digest_operations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch2", "full50"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--source-runner", type=Path, default=OLD / "archcomp26_quad_paper_p3_nohash.py")
    parser.add_argument("--crown-relax", choices=("same-slope", "two-slope"), default="same-slope")
    parser.add_argument("--weighted-chunk", choices=(128, 256), type=int, default=128)
    parser.add_argument("--compare", choices=("equivalent", "width"), default="equivalent")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3" or set(os.sched_getaffinity(0)) != {14, 15, 16, 17}:
        raise RuntimeError("requires physical GPU3 and CPUs14-17")
    gate = json.loads((args.gate / "RESULT.json").read_text())
    if gate.get("status") != "PASSED_NEW_PRIVATE_OUTPUT_GPU_GATE":
        raise RuntimeError("new GPU gate has not passed")
    if args.crown_relax == "two-slope":
        if args.compare != "width" or 'getattr(args, "crown_relax"' not in args.source_runner.read_text():
            raise RuntimeError("two-slope requires the explicit new source runner and width comparison")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_new(output / "START.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "scope": "new implementation candidate; old experiment is read-only",
        "mode": args.mode, "gate": str(args.gate), "source_runner": str(args.source_runner),
        "archived_adapters": str(OLD / "archcomp26_quad_p3_nohash"),
        "crown_relax": args.crown_relax, "weighted_chunk": args.weighted_chunk,
        "comparison": args.compare,
        "gpu": "3", "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "no_jit": True, "no_digest_operations": True})
    result = {"status": "exception", "end_to_end_strict_certificate": False}
    binding = None
    weighted_override = None
    code = 1
    try:
        baseline = load_baseline(args.source_runner)
        saved_prepare = baseline.prepare

        def prepare(batch):
            nonlocal binding, weighted_override
            prepared = saved_prepare(batch)
            torch, driver = prepared[:2]
            from flowstar_gpu import cuda_kernels as ck, private_output_kernels
            binding = install(torch, ck, private_output_kernels)
            write_new(output / "OPTIMIZATION.json", binding.receipt)
            if args.weighted_chunk == 256:
                if (batch != 1024 or len(prepared) != 15
                        or prepared[8].policy != "original_early_weighted_rounds_and_attempts_fixed128_graph_scratch"
                        or any(value for key, value in prepared[8].counters.items() if key != "chunk_size")):
                    raise RuntimeError("weighted128 must be unused before replacement")
                prepared[8].restore()
                from flowstar_gpu import weighted_validation
                from weighted_chunk256 import install as install_weighted
                weighted_override = install_weighted(torch, weighted_validation, prepared[2])
                items = list(prepared)
                items[8] = weighted_override
                prepared = tuple(items)
                write_new(output / "WEIGHTED_BINDING.json", {
                    "replaced_policy": "original_early_weighted_rounds_and_attempts_fixed128_graph_scratch",
                    "selected_policy": weighted_override.policy,
                    "source": weighted_override.source_path,
                    "installed_before_first_engine_or_graph": True,
                    "initial_counters": dict(weighted_override.counters)})
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
            source_config=OLD / "quad_paper.yaml", output=output / "data",
            crown_relax=args.crown_relax))
        if code:
            raise RuntimeError(f"candidate numerical process returned {code}")
        comparison = (compare_saved(output / "data", args.mode) if args.compare == "equivalent"
                      else compare_widths(output / "data", args.mode, output))
        write_new(output / "SAVED_COMPARISON.json", comparison)
        completed = 40 if args.mode == "batch2" else 1000
        result.update(status="COMPLETED_SAVED_OUTPUT_EQUIVALENT" if args.compare == "equivalent"
                      else "COMPLETED_SAVED_WIDTH_COMPARISON", exit_code=0,
                      completed_substeps=completed, accepted_lane_substeps=1024 * completed,
                      crown_relax=args.crown_relax, weighted_chunk=args.weighted_chunk)
        code = 0
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr)
        code = 1
    finally:
        if weighted_override is not None:
            write_new(output / "WEIGHTED_RESULT.json", {
                "policy": weighted_override.policy, "counters": weighted_override.counters})
            try:
                weighted_override.restore()
                result["weighted_binding_restored"] = True
            except Exception as error:
                result.update(status="RESTORE_FAILURE", weighted_restore_error=str(error))
                code = 1
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
