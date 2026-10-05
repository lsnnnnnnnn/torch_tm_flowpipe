#!/usr/bin/env python3
"""New QUAD anchored-control candidate through the unchanged October 5 wrapper."""
import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from quad_controller_hook import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch2", "full50"), default="batch2")
    parser.add_argument("--base-wrapper", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--source-runner", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path, output = args.base_wrapper.resolve(), args.output.resolve()
    if not base_path.is_file() or base_path.name != "run_quad_candidate.py":
        parser.error("--base-wrapper must name the saved October 5 run_quad_candidate.py")
    if output == base_path.parent or output.is_relative_to(base_path.parent):
        parser.error("new output must be outside the frozen wrapper source folder")
    sys.path.insert(0, str(base_path.parent))
    spec = importlib.util.spec_from_file_location("anchored_saved_quad_wrapper", base_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    baseline.guard_digests()
    original_load, original_write, original_compare = baseline.load_baseline, baseline.write_new, baseline.compare_widths
    outer_argv, binding, restored = list(sys.argv), None, False
    expected_refreshes = 2 if args.mode == "batch2" else 50

    def load(*a, **kw):
        runner = original_load(*a, **kw)
        original_prepare = runner.prepare

        def prepare(batch):
            prepared = original_prepare(batch)
            torch, driver = prepared[:2]
            original_main = driver.main

            def main_with_control():
                nonlocal binding, restored
                if binding is not None:
                    raise RuntimeError("control candidate must install exactly once")
                from flowstar_gpu import interval as iv, support as sp
                # The October 5 first-refusal wrapper and canonical strict
                # injection, observer and holder checks are installed by now.
                binding = install(torch, driver, iv, sp)
                original_write(output / "CONTROL_BINDING.json", binding.receipt)
                try:
                    return original_main()
                finally:
                    try:
                        binding.restore()
                        restored = True
                    finally:
                        original_write(output / "CONTROL_RESULT.json", dict(
                            receipt=binding.receipt, counters=dict(binding.counters),
                            refreshes=binding.records, binding_restored=restored,
                            independent_end_to_end_floating_nncs_certificate=False))

            driver.main = main_with_control
            return prepared

        runner.prepare = prepare
        return runner

    def compare(*a, **kw):
        expected = dict(base_nn_calls=expected_refreshes, extra_nn_calls=expected_refreshes,
                        total_nn_calls=2*expected_refreshes, injected_refreshes=expected_refreshes,
                        total_control_rows=expected_refreshes*1024*3)
        if binding is None or not restored or len(binding.records) != expected_refreshes:
            raise RuntimeError("anchored controller did not cover every refresh and restore")
        if any(binding.counters.get(k) != v for k,v in expected.items()):
            raise RuntimeError("anchored controller actual NN/refresh coverage mismatch")
        return original_compare(*a, **kw)

    def write(path, value):
        if path.name in ("START.json", "RESULT.json"):
            value = dict(value, control_candidate="same-slope polynomial + extra two-slope affine remainder",
                         control_wrapper=str(Path(__file__).resolve()), base_wrapper=str(base_path),
                         control_outer_argv=outer_argv, end_to_end_strict_certificate=False)
        if path.name == "START.json":
            value.update(scope="new numerical width candidate; original saved ranges are read only",
                         expected_actual_nn_calls=2*expected_refreshes,
                         unchanged="P3/order3, initial boxes, dynamics, strict injection/endpoint, full SR history, observers, first refusal, private output allocation, weighted256 and memory cap")
        elif path.name == "RESULT.json":
            value.update(control_counters=dict(binding.counters) if binding else {},
                         control_binding_restored=restored,
                         width_is_not_containment_or_nncs_certificate=True)
        original_write(path, value)

    baseline.load_baseline, baseline.write_new, baseline.compare_widths = load, write, compare
    sys.argv = [str(base_path), "--mode", args.mode, "--weighted-chunk", "256",
                "--compare", "width", "--crown-relax", "same-slope",
                "--gate", str(args.gate.resolve()), "--output", str(output)]
    if args.source_runner is not None:
        sys.argv += ["--source-runner", str(args.source_runner.resolve())]
    try:
        return baseline.main()
    finally:
        sys.argv = outer_argv
        baseline.load_baseline, baseline.write_new, baseline.compare_widths = original_load, original_write, original_compare


if __name__ == "__main__":
    raise SystemExit(main())
