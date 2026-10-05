#!/usr/bin/env python3
"""Run a new fused-batch candidate through the unchanged October 5 wrapper.

Only the weighted install function is substituted. The inherited resource
preflight, first refusal behavior, observers, original saved-output comparison
and reverse-order restoration remain in charge of execution.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from weighted_fused_batch import install


def qualification(counters, *, batch, rows, steps):
    expected = dict(refine_calls=steps, fast_calls=steps, fallback_calls=0,
                    candidate_exceptions=0, reference_checks=1,
                    reference_map_evaluations=2 * batch,
                    graph_replays=steps * ((batch + rows - 1) // rows),
                    replay_map_calls=steps * ((batch + rows - 1) // rows) * 2,
                    replay_target_rows=steps * batch * 2,
                    replay_padded_rows=steps * (((batch + rows - 1) // rows) * rows - batch) * 2)
    differences = {name: dict(expected=value, actual=counters.get(name))
                   for name, value in expected.items() if counters.get(name) != value}
    return dict(passed=not differences, expected=expected, differences=differences,
                scope="live fast-path use and first input equivalence; saved-output comparison separately required")


def load_wrapper(path, instance):
    # All sibling imports resolve to the user's explicit, frozen wrapper folder.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("fused_batch_saved_" + instance + "_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=("nav", "quad"), required=True)
    parser.add_argument("--mode", choices=("full600", "batch2", "full50"), required=True)
    parser.add_argument("--base-wrapper", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, inherited = parser.parse_known_args()
    if (args.instance == "nav") != (args.mode == "full600"):
        parser.error("NAV requires full600; QUAD requires batch2 or full50")
    controlled = {"--weighted-chunk", "--compare", "--crown-relax", "--mode"}
    if any(token.partition("=")[0] in controlled for token in inherited):
        parser.error("weighted scratch, comparison, relaxation and mode are fixed by this candidate")
    base_path = args.base_wrapper.resolve()
    expected_name = "run_nav_candidate.py" if args.instance == "nav" else "run_quad_candidate.py"
    if not base_path.is_file() or base_path.name != expected_name:
        parser.error(f"--base-wrapper must name the frozen {expected_name}")
    output = args.output.resolve()
    if output == base_path.parent or output.is_relative_to(base_path.parent):
        parser.error("new output must be outside the frozen wrapper source folder")
    batch, rows = (25, 32) if args.instance == "nav" else (1024, 256)
    steps = {"full600": 600, "batch2": 40, "full50": 1000}[args.mode]
    outer_argv, binding, installed_count = list(sys.argv), None, 0
    baseline = load_wrapper(base_path, args.instance)
    original_writer, original_compare = baseline.write_new, baseline.compare_saved

    def write_new(path, value):
        value = dict(value)
        if path.name in ("START.json", "RESULT.json"):
            value.update(weighted_mode="fused_batch", fused_batch_outer_argv=outer_argv,
                         fused_batch_wrapper=str(Path(__file__).resolve()),
                         base_wrapper=str(base_path), dont_write_bytecode=sys.dont_write_bytecode,
                         fused_batch_configuration=dict(batch=batch, rows=rows, steps=steps))
        if path.name == "START.json":
            value["scope"] = "new fixed-batch two-round fusion; frozen October 5 wrapper and original comparison retained"
            value["changes"] = ["private output allocation", "existing Horner/strict bindings retained",
                                f"success-only weighted fusion: B={batch}, rows={rows}, two unchanged maps"]
        if path.name == "RESULT.json":
            counts = dict(binding.counters) if binding is not None else {}
            value["fused_batch_counters"] = counts
            value["fused_batch_qualification"] = qualification(counts, batch=batch, rows=rows, steps=steps)
            value["fused_batch_install_count"] = installed_count
        original_writer(path, value)

    def install_candidate(torch, wv, se):
        nonlocal binding, installed_count
        if binding is not None or installed_count:
            raise RuntimeError("fused batch must install exactly once")
        binding = install(torch, wv, se, batch=batch, rows=rows, check_first=1)
        installed_count += 1
        original_writer(output / "FUSED_BATCH_BINDING.json", binding.receipt)
        return binding

    def compare_saved(*a, **kw):
        if binding is None or installed_count != 1:
            raise RuntimeError("fused batch installation was not consumed exactly once")
        checked = qualification(binding.counters, batch=batch, rows=rows, steps=steps)
        original_writer(output / "FUSED_BATCH_QUALIFICATION.json", checked)
        if not checked["passed"]:
            raise RuntimeError(f"fused batch qualification failed: {checked['differences']}")
        return original_compare(*a, **kw)

    baseline.write_new, baseline.compare_saved = write_new, compare_saved
    if args.instance == "nav":
        installer_owner, installer_name = baseline, "install_weighted32"
    else:
        # The inherited prepare() imports this function after retiring its
        # unused fixed128 binding. Replace only the explicit install slot.
        import weighted_chunk256
        if Path(weighted_chunk256.__file__).resolve().parent != base_path.parent:
            raise RuntimeError("weighted_chunk256 did not come from the explicit frozen wrapper folder")
        installer_owner, installer_name = weighted_chunk256, "install"
    original_install = getattr(installer_owner, installer_name)
    setattr(installer_owner, installer_name, install_candidate)
    sys.argv = [str(base_path), "--mode", "fast" if args.instance == "nav" else args.mode,
                "--weighted-chunk", str(rows), "--gate", str(args.gate.resolve()),
                "--output", str(output), *inherited]
    if args.instance == "quad":
        sys.argv += ["--compare", "equivalent", "--crown-relax", "same-slope"]
    try:
        return baseline.main()
    finally:
        sys.argv = outer_argv
        setattr(installer_owner, installer_name, original_install)
        baseline.write_new, baseline.compare_saved = original_writer, original_compare


if __name__ == "__main__":
    raise SystemExit(main())
