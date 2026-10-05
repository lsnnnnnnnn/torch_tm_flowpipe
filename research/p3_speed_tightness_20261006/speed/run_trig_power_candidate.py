#!/usr/bin/env python3
"""QUAD private256 plus same-input sin/cos power reuse; no fused weighted path."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from trig_power_reuse import install
from trig_power_gate import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch2", "full50"), default="batch2")
    parser.add_argument("--base-wrapper", type=Path, required=True)
    parser.add_argument("--source-runner", type=Path)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path, output = args.base_wrapper.resolve(), args.output.resolve()
    if not base_path.is_file() or base_path.name != "run_quad_candidate.py":
        parser.error("requires the frozen October5 run_quad_candidate.py")
    if output == base_path.parent or output.is_relative_to(base_path.parent):
        parser.error("output must be outside the frozen wrapper folder")
    sys.path.insert(0, str(base_path.parent))
    spec = importlib.util.spec_from_file_location("trig_power_saved_quad_wrapper", base_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    baseline.guard_digests()
    original_load, original_write, original_compare = baseline.load_baseline, baseline.write_new, baseline.compare_saved
    outer_argv, binding, restored, gate_passed = list(sys.argv), None, False, False
    numerical_counts, gate_wall = {}, None

    def load(*a, **kw):
        runner = original_load(*a, **kw)
        original_prepare = runner.prepare

        def prepare(batch):
            prepared = original_prepare(batch)
            if batch != 1024 or len(prepared) != 15:
                raise RuntimeError("trig power scope requires the existing paper full1024 prepare")
            torch, driver, se = prepared[:3]
            direct, metadata = prepared[6], prepared[7].backend
            original_main = driver.main

            def main_with_trig_power():
                nonlocal binding, restored, gate_passed, numerical_counts, gate_wall
                if binding is not None:
                    raise RuntimeError("trig power must install exactly once")
                # Inherited private256 and first-refusal hooks are installed;
                # the production engine and its graphs do not yet exist.
                binding = install(torch, se, direct)
                original_write(output / "TRIG_POWER_BINDING.json", binding.receipt)
                try:
                    started = time.perf_counter()
                    gate_result = check(torch, se, binding, device="cuda:0", metadata_backend=metadata)
                    gate_wall = time.perf_counter() - started
                    gate_result["wall_s"] = gate_wall
                    original_write(output / "TRIG_POWER_GPU_GATE.json", gate_result)
                    if (gate_result["status"] != "PASSED_NEW_TRIG_POWER_GATE"
                            or gate_result["cuda_graph_captures"] != 4):
                        raise RuntimeError("new trig-power eager/graph gate did not finish")
                    gate_passed = True
                    before = dict(binding.counters)
                    try:
                        return original_main()
                    finally:
                        numerical_counts = {name: binding.counters[name] - before[name] for name in before}
                finally:
                    try:
                        binding.restore()
                        restored = True
                    finally:
                        original_write(output / "TRIG_POWER_RESULT.json", dict(
                            receipt=binding.receipt, counters=dict(binding.counters),
                            numerical_counter_delta=numerical_counts, last_plan=dict(binding.last_plan),
                            binding_restored=restored, new_gpu_gate_passed=gate_passed,
                            gate_wall_s=gate_wall, independent_end_to_end_floating_nncs_certificate=False))

            driver.main = main_with_trig_power
            return prepared

        runner.prepare = prepare
        return runner

    def compare(*a, **kw):
        # Preserve scientific equivalence evidence even if the optimization
        # eligibility/counter condition subsequently refuses promotion.
        comparison = original_compare(*a, **kw)
        original_write(output / "SAVED_COMPARISON_BEFORE_SPEED_GATE.json", comparison)
        if (binding is None or not restored or not gate_passed
                or numerical_counts.get("failed_execs") != 0
                or numerical_counts.get("exec_calls", 0) <= 0
                or numerical_counts.get("completed_execs") != numerical_counts.get("exec_calls")
                or numerical_counts.get("reused_powers", 0) <= 0
                or numerical_counts.get("reused_power_chains", 0) <= 0
                or binding.last_plan.get("opposite_op_pairs", 0) <= 0):
            raise RuntimeError("trig power runtime gate/counters/restore incomplete")
        return comparison

    def write(path, value):
        if path.name in ("START.json", "RESULT.json"):
            value = dict(value, trig_power_candidate=True, fused_weighted=False,
                         weighted_chunk=256, crown_relax="same-slope", comparison="equivalent",
                         trig_power_outer_argv=outer_argv, trig_power_wrapper=str(Path(__file__).resolve()),
                         base_wrapper=str(base_path), end_to_end_strict_certificate=False)
        if path.name == "START.json":
            value["scope"] = "new same-input opposite-op power reuse over qualified private256; no QUAD fused weighted path"
            value["timing_policy"] = "new tiny GPU gate included in wrapper/payload/process time and separately reported; original driver timer starts afterwards"
        if path.name == "RESULT.json":
            value.update(trig_power_numerical_counters=numerical_counts,
                         trig_power_binding_restored=restored, new_trig_gpu_gate_passed=gate_passed,
                         trig_gpu_gate_wall_s=gate_wall)
        original_write(path, value)

    baseline.load_baseline, baseline.write_new, baseline.compare_saved = load, write, compare
    sys.argv = [str(base_path), "--mode", args.mode, "--weighted-chunk", "256",
                "--compare", "equivalent", "--crown-relax", "same-slope",
                "--gate", str(args.gate.resolve()), "--output", str(output)]
    if args.source_runner is not None:
        sys.argv += ["--source-runner", str(args.source_runner.resolve())]
    try:
        return baseline.main()
    finally:
        sys.argv = outer_argv
        baseline.load_baseline, baseline.write_new, baseline.compare_saved = original_load, original_write, original_compare


if __name__ == "__main__":
    raise SystemExit(main())
