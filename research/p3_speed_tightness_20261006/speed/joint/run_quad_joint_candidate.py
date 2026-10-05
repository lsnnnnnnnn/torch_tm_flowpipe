#!/usr/bin/env python3
"""Optional new QUAD trig-power plus anchored-control v2 width candidate.

Dependencies are loaded from explicit frozen folders. This never claims that
the new trajectory equals either parent candidate. The October 5 wrapper owns
private256, strict execution, complete observations and first-refusal stop.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def load_named(name, path):
    """Load an explicit file once; refuse inherited modules with the same name."""
    if name in sys.modules:
        raise RuntimeError(f"module slot already occupied: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def qualify(mode, trig, control, numerical, *, gate_passed, trig_restored, control_restored):
    refreshes = 2 if mode == "batch2" else 50
    expected = dict(base_nn_calls=refreshes, extra_nn_calls=refreshes,
                    total_nn_calls=2*refreshes, injected_refreshes=refreshes,
                    total_control_rows=refreshes*1024*3)
    if (trig is None or control is None or not gate_passed
            or not trig_restored or not control_restored):
        raise RuntimeError("joint bindings/gate/restoration incomplete")
    if (numerical.get("failed_execs") != 0 or numerical.get("exec_calls", 0) <= 0
            or numerical.get("completed_execs") != numerical.get("exec_calls")
            or numerical.get("reused_powers", 0) <= 0
            or numerical.get("reused_power_chains", 0) <= 0
            or trig.last_plan.get("opposite_op_pairs", 0) <= 0):
        raise RuntimeError("joint trig-power numerical counters incomplete")
    if len(control.records) != refreshes or any(control.counters.get(k) != v for k, v in expected.items()):
        raise RuntimeError("joint controller actual NN/refresh coverage mismatch")
    return dict(status="PASSED_JOINT_BINDING_QUALIFICATION", expected_control=expected,
                trig_numerical_counters=dict(numerical), new_trig_gpu_gate_passed=True,
                trig_binding_restored=True, control_binding_restored=True,
                saved_outputs_equal_claim=False, independent_end_to_end_floating_nncs_certificate=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("batch2", "full50"), default="batch2")
    parser.add_argument("--base-wrapper", type=Path, required=True)
    parser.add_argument("--trig-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--source-runner", type=Path)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path, trig_dir, control_dir, output = (p.resolve() for p in
        (args.base_wrapper, args.trig_dir, args.control_dir, args.output))
    if not base_path.is_file() or base_path.name != "run_quad_candidate.py":
        parser.error("requires the frozen October 5 run_quad_candidate.py")
    dependencies = dict(trig=trig_dir/"trig_power_reuse.py", gate=trig_dir/"trig_power_gate.py",
                        anchor=control_dir/"anchored_control.py", control=control_dir/"quad_controller_hook.py")
    if not all(path.is_file() for path in dependencies.values()):
        parser.error("explicit frozen trig and control v2 dependency files are required")
    protected = (base_path.parent, trig_dir, control_dir, Path(__file__).resolve().parent)
    if args.source_runner is not None:
        if not args.source_runner.is_file():
            parser.error("explicit source runner must exist")
        protected += (args.source_runner.resolve().parent,)
    if output.exists() or any(output == p or output.is_relative_to(p) for p in protected):
        parser.error("output must be new and outside every frozen source folder")
    sys.path.insert(0, str(base_path.parent))
    baseline = load_named("quad_joint_saved_wrapper", base_path)
    baseline.guard_digests()
    trig_module = load_named("quad_joint_trig_power", dependencies["trig"])
    gate_module = load_named("quad_joint_trig_gate", dependencies["gate"])
    # The frozen hook imports this exact name. Register our explicit path;
    # never let Python search an unrelated anchored_control module.
    anchor = load_named("anchored_control", dependencies["anchor"])
    control_module = load_named("quad_joint_control_v2", dependencies["control"])
    if control_module.contract is not anchor.contract:
        raise RuntimeError("control hook did not bind the explicit anchor dependency")
    original_load, original_write, original_compare = baseline.load_baseline, baseline.write_new, baseline.compare_widths
    outer_argv = list(sys.argv)
    trig = control = None
    trig_restored = control_restored = gate_passed = False
    numerical, gate_wall = {}, None
    expected_refreshes = 2 if args.mode == "batch2" else 50

    def load(*a, **kw):
        runner = original_load(*a, **kw)
        original_prepare = runner.prepare

        def prepare(batch):
            prepared = original_prepare(batch)
            if batch != 1024 or len(prepared) != 15:
                raise RuntimeError("joint candidate requires the existing paper1024 prepare")
            torch, driver, se = prepared[:3]
            direct, metadata = prepared[6], prepared[7].backend
            original_main = driver.main

            def main_with_joint():
                nonlocal trig, control, trig_restored, control_restored, gate_passed, numerical, gate_wall
                if trig is not None or control is not None:
                    raise RuntimeError("joint bindings must install exactly once")
                # The inherited wrapper has installed private256 and the
                # observer/strict hooks; production engines do not exist yet.
                trig = trig_module.install(torch, se, direct)
                try:
                    original_write(output/"TRIG_POWER_BINDING.json", trig.receipt)
                    started = time.perf_counter()
                    checked = gate_module.check(torch, se, trig, device="cuda:0", metadata_backend=metadata)
                    gate_wall = time.perf_counter()-started
                    checked["wall_s"] = gate_wall
                    original_write(output/"TRIG_POWER_GPU_GATE.json", checked)
                    if checked["status"] != "PASSED_NEW_TRIG_POWER_GATE" or checked["cuda_graph_captures"] != 4:
                        raise RuntimeError("joint new trig-power GPU gate did not finish")
                    gate_passed = True
                    from flowstar_gpu import interval as iv, support as sp
                    control = control_module.install(torch, driver, iv, sp)
                    try:
                        if control.receipt.get("implementation_revision") != "v2_real_support_exponent_lookup":
                            raise RuntimeError("joint candidate requires anchored control v2")
                        original_write(output/"CONTROL_BINDING.json", control.receipt)
                        before = dict(trig.counters)
                        try:
                            return original_main()
                        finally:
                            numerical = {k: trig.counters[k]-before[k] for k in before}
                    finally:
                        try:
                            control.restore()
                            control_restored = True
                        finally:
                            original_write(output/"CONTROL_RESULT.json", dict(
                                receipt=control.receipt, counters=dict(control.counters),
                                refreshes=control.records, binding_restored=control_restored,
                                independent_end_to_end_floating_nncs_certificate=False))
                finally:
                    try:
                        trig.restore()
                        trig_restored = True
                    finally:
                        original_write(output/"TRIG_POWER_RESULT.json", dict(
                            receipt=trig.receipt, counters=dict(trig.counters),
                            numerical_counter_delta=numerical, last_plan=dict(trig.last_plan),
                            binding_restored=trig_restored, new_gpu_gate_passed=gate_passed,
                            gate_wall_s=gate_wall, independent_end_to_end_floating_nncs_certificate=False))

            driver.main = main_with_joint
            return prepared

        runner.prepare = prepare
        return runner

    def compare(*a, **kw):
        # Preserve new geometry evidence before any optimization counter gate.
        comparison = original_compare(*a, **kw)
        original_write(output/"WIDTH_COMPARISON_BEFORE_JOINT_GATE.json", comparison)
        qualification = qualify(args.mode, trig, control, numerical, gate_passed=gate_passed,
                                trig_restored=trig_restored, control_restored=control_restored)
        original_write(output/"JOINT_QUALIFICATION.json", qualification)
        return comparison

    def write(path, value):
        if path.name in ("START.json", "RESULT.json"):
            value = dict(value, joint_candidate="trig_power_plus_anchored_control_v2", fused_weighted=False,
                         weighted_chunk=256, crown_relax="same-slope", comparison="width",
                         saved_outputs_equal_claim=False, joint_outer_argv=outer_argv,
                         joint_wrapper=str(Path(__file__).resolve()), base_wrapper=str(base_path),
                         explicit_dependencies={k: str(v) for k, v in dependencies.items()},
                         end_to_end_strict_certificate=False)
        if path.name == "START.json":
            value.update(scope="one new joint numerical trajectory; complete own saved state widths",
                         expected_actual_nn_calls=2*expected_refreshes,
                         unchanged="P3/order3, initial boxes, dynamics, strict injection/endpoint, full SR history, observers, first refusal, private outputs, weighted256 and memory cap",
                         timing_policy="new tiny trig gate included in wrapper/payload/process and separately recorded, outside original driver timer; extra control model before driver timer, its bound/contract work inside")
        elif path.name == "RESULT.json":
            value.update(trig_power_numerical_counters=numerical, trig_power_binding_restored=trig_restored,
                         new_trig_gpu_gate_passed=gate_passed, trig_gpu_gate_wall_s=gate_wall,
                         control_counters=dict(control.counters) if control else {},
                         control_binding_restored=control_restored,
                         width_is_not_containment_or_nncs_certificate=True)
        elif path.name == "SAVED_COMPARISON.json":
            path = path.with_name("WIDTH_COMPARISON.json")
        return original_write(path, value)

    baseline.load_baseline, baseline.write_new, baseline.compare_widths = load, write, compare
    sys.argv = [str(base_path), "--mode", args.mode, "--weighted-chunk", "256", "--compare", "width",
                "--crown-relax", "same-slope", "--gate", str(args.gate.resolve()), "--output", str(output)]
    if args.source_runner is not None:
        sys.argv += ["--source-runner", str(args.source_runner.resolve())]
    try:
        return baseline.main()
    finally:
        sys.argv = outer_argv
        baseline.load_baseline, baseline.write_new, baseline.compare_widths = original_load, original_write, original_compare


if __name__ == "__main__":
    raise SystemExit(main())
