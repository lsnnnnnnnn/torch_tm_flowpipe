#!/usr/bin/env python3
"""NAV standard: original full640/600 wrapper with fixed512 two-round fusion."""
import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from weighted_fused_batch import install


def qualification(counters):
    fast, fallback = counters.get("fast_calls", 0), counters.get("fallback_calls", 0)
    expected = dict(refine_calls=600, candidate_exceptions=0, reference_checks=1,
                    reference_map_evaluations=1280, chunk_size=512,
                    original_graph_pool_releases=1 + fallback)
    differences = {key: dict(expected=value, actual=counters.get(key))
                   for key, value in expected.items() if counters.get(key) != value}
    if fast <= 0 or fast + fallback != 600:
        differences["dispatch_coverage"] = dict(expected="fast>0 and fast+fallback=600",
                                                actual=dict(fast=fast, fallback=fallback))
    all_fast_counts = dict(fast_calls=600, fallback_calls=0, graph_replays=1200,
                           replay_map_calls=2400, replay_target_rows=768000,
                           replay_padded_rows=460800)
    return dict(passed=not differences, expected=expected, differences=differences,
                all_steps_fast=all(counters.get(k) == v for k, v in all_fast_counts.items()),
                legal_original_fallbacks=fallback,
                timing_qualification="saved-output-equivalent mixed dispatch is admissible; measured speed decides promotion",
                reference="first live full640 input through original refine_accepted with original512 scratch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-wrapper", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, inherited = parser.parse_known_args()
    base_path, adapters, output = (p.resolve() for p in (args.base_wrapper, args.adapters, args.output))
    if base_path.name != "run_nav_standard_candidate.py" or not base_path.is_file():
        parser.error("requires the frozen expansion/run_nav_standard_candidate.py")
    if output == base_path.parent or output.is_relative_to(base_path.parent):
        parser.error("output must be outside the frozen wrapper folder")
    spec = importlib.util.spec_from_file_location("nav640_original_wrapper_for_fused512", base_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    sys.path.insert(0, str(adapters))
    import run_quad_candidate as helpers
    import weighted_chunk256 as weighted_module
    if any(Path(module.__file__).resolve().parent != adapters for module in (helpers, weighted_module)):
        raise RuntimeError("original helper modules did not resolve from explicit adapters")
    original_writer, original_install = helpers.write_new, weighted_module.install
    original_compare = baseline.compare_saved
    binding, installed_count = None, 0
    outer_argv = list(sys.argv)

    def write_new(path, value):
        value = dict(value)
        if path.name in ("START.json", "RESULT.json"):
            value.update(fused=True, weighted_mode="fused_batch", weighted_chunk=512,
                         instance_batch=640, fused_outer_argv=outer_argv,
                         fused_wrapper=str(Path(__file__).resolve()), base_wrapper=str(base_path))
        if path.name == "START.json":
            value.pop("non_fused_reason", None)
            value["changes"] = ["private output allocation", "preloaded Horner binding",
                                "fixed512 success-only two-round fusion over all640 rows"]
            value["dispatch_tradeoff"] = "two 512-row fused replays per refine, including128 real+384 padded rows in last chunk; original two rounds and512 scratch retained"
        if path.name == "WEIGHTED_BINDING.json":
            value.update(binding.receipt, chunk_size=512, fixed_graph_rows=512,
                         rounds=2, instance_batch=640, fused=True)
        if path.name == "RESULT.json":
            counts = dict(binding.counters) if binding is not None else {}
            value.update(fused_qualification=qualification(counts), fused_install_count=installed_count)
        original_writer(path, value)

    def install_candidate(torch, wv, se):
        nonlocal binding, installed_count
        if binding is not None or installed_count:
            raise RuntimeError("NAV640 fusion must install exactly once")
        binding = install(torch, wv, se, batch=640, rows=512, check_first=1)
        installed_count += 1
        return binding

    def compare_saved(*a, **kw):
        if binding is None or installed_count != 1:
            raise RuntimeError("NAV640 fusion was not installed exactly once")
        checked = qualification(binding.counters)
        original_writer(output / "FUSED_BATCH_QUALIFICATION.json", checked)
        if not checked["passed"]:
            raise RuntimeError(f"NAV640 fused qualification failed: {checked['differences']}")
        return original_compare(*a, **kw)

    helpers.write_new, weighted_module.install = write_new, install_candidate
    baseline.compare_saved = compare_saved
    sys.argv = [str(base_path), "--adapters", str(adapters), "--gate", str(args.gate.resolve()),
                "--output", str(output), *inherited]
    try:
        return baseline.main()
    finally:
        sys.argv = outer_argv
        helpers.write_new, weighted_module.install = original_writer, original_install
        baseline.compare_saved = original_compare


if __name__ == "__main__":
    raise SystemExit(main())
