#!/usr/bin/env python3
"""Isolated 2026 paper-QUAD P2 parity run on Xiangru's original GPU engine."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
X = Path("/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream")
DRIVER = X / "src/flowstar_gpu/integrations/crown_reach.py"
CACHE = N / "cache_four_way_v1/xiangru"
MODEL = Path("/srv/local/shengenli/CROWN-Reach-GPU/ARCH-COMP2024/benchmarks/QUAD/quad_controller_3_64_torch.onnx")


def prohibited(*_args, **_kwargs):
    raise RuntimeError("content digest or CUDA extension build disabled for paper-QUAD run")


def guard_digests_and_builds():
    hashlib.sha256 = prohibited
    original_new = hashlib.new

    def guarded_new(name, *args, **kwargs):
        if name.lower().replace("-", "") == "sha256":
            return prohibited()
        return original_new(name, *args, **kwargs)

    hashlib.new = guarded_new
    import torch.utils.cpp_extension as extension
    extension.load = prohibited
    extension.load_inline = prohibited


def load_module(name, path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import saved CUDA library: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("Xiangru paper-QUAD arm is reserved for physical GPU 3")
    for key in list(os.environ):
        if key.startswith("FLOWSTAR_"):
            del os.environ[key]
    os.environ["TORCH_EXTENSIONS_DIR"] = str(CACHE)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(X / "src"))
    guard_digests_and_builds()
    import torch
    from flowstar_gpu import cuda_kernels as ck, tape_kernels as tk, determinism

    if torch.__version__ != "2.5.1+cu121" or not torch.cuda.is_available():
        raise RuntimeError("saved PyTorch/CUDA environment is unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not inspect.signature(determinism.enable_determinism).parameters:
        original = determinism.enable_determinism
        determinism.enable_determinism = lambda device="cuda": original()
    ck.load_cuda_extension = prohibited
    tk.load_cuda_extension = prohibited

    binaries = {}
    for owner, attr, tried, name, exports in (
        (ck, "_ext", "_tried", "flowstar_seg_kernels", ("seg_mul_iv", "seg_mul_pt", "seg_dot_pt_iv")),
        (tk, "_ext", "_tried", "flowstar_tape_kernels", ("refine_tape", "transcendental_probe")),
        (tk, "_vext", "_vtried", "flowstar_valid_kernels", ("valid_tape", "point_tape")),
    ):
        path = CACHE / name / (name + ".so")
        module = load_module(name, path)
        if not all(callable(getattr(module, export, None)) for export in exports):
            raise RuntimeError(f"saved CUDA library lacks expected exports: {path}")
        setattr(owner, attr, module)
        setattr(owner, tried, True)
        stat = path.stat()
        binaries[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not (ck.available() and tk.available() and tk.valid_available()):
        raise RuntimeError("preloaded Xiangru CUDA libraries unavailable")

    # The shared driver's metrics writer reads this optional reporting field.
    driver = load_module("archcomp26_quad_paper_xiangru_driver", DRIVER)
    if not hasattr(driver.sparse_exec_module, "COMPOSE_PARENT_ASSEMBLY"):
        driver.sparse_exec_module.COMPOSE_PARENT_ASSEMBLY = None
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_cap = min(14 * 2**30, int(total_memory * 0.9))
    torch.cuda.set_per_process_memory_fraction(memory_cap / total_memory)
    return torch, driver, binaries, memory_cap


def checked_config(source, mode, output):
    import yaml

    config = yaml.safe_load(source.read_text())
    if not (config["num_vars"] == 16 and config["num_nn_input"] == 12
            and config["num_nn_output"] == 3 and config["ode_order"] == 2
            and config["steps"] == 50 and config["step_size"] == 0.1
            and config["ode_step_size"] == 0.005
            and [entry["splits"] for entry in config["initial_set"][:6]] == [8, 8, 8, 2, 1, 1]
            and config["model_dir"] == str(MODEL) and MODEL.is_file()
            and config["constraints_target"] == ["-x3 + 0.94", "x3 - 1.06"]):
        raise ValueError("source is not the fixed 2026 paper-QUAD P2 contract")
    expr = ["".join(value.split()) for value in config["dynamics_expressions"]]
    if (expr[1] != "cos(x8)*sin(x9)*x4+(sin(x7)*sin(x8)*sin(x9)+cos(x7)*cos(x9))*x5+(cos(x7)*sin(x8)*sin(x9)-sin(x7)*cos(x9))*x6"
            or expr[3] != "x12*x5-x11*x6-9.81*sin(x8)"
            or expr[4] != "x10*x6-x12*x4+9.81*cos(x8)*sin(x7)"):
        raise ValueError("source lacks the selected 2026 paper x2/x4/x5 equations")
    config = copy.deepcopy(config)
    if mode == "smoke1":
        for entry in config["initial_set"][:6]:
            lo, hi = entry["interval"]
            count = entry["splits"]
            entry["interval"] = [lo, lo + (hi - lo) / count]
            entry["splits"] = 0
        config["split_vars"] = []
    config["steps"] = {"smoke1": 1, "batch1": 1, "batch2": 2, "full": 50}[mode]
    generated = output / "config.yaml"
    generated.write_text(yaml.safe_dump(config, sort_keys=False))
    return config, generated


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "exception", "mode": args.mode, "completed_substeps": 0}
    try:
        torch, driver, binaries, memory_cap = prepare()
        config, path = checked_config(args.source_config, args.mode, output)
        boxes = driver.make_cells(config)
        expected_boxes = 1 if args.mode == "smoke1" else 1024
        if boxes.shape != (expected_boxes, 16, 2):
            raise ValueError(f"QUAD partition mismatch: {tuple(boxes.shape)}")
        record = {
            "schema": "archcomp26-quad-paper-xiangru-parity-nohash-v1",
            "start_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "source_config": str(args.source_config),
            "generated_config": str(path),
            "engine": str(X),
            "driver": str(DRIVER),
            "model": str(MODEL),
            "controller_note": "saved server Torch ONNX directly byte-compared with fixed official 2026 Torch ONNX in prior contract audit; no digest here",
            "method": "Xiangru original sparse GPU engine, P2 order-2 parity, box same-slope CROWN, native-f64 transport",
            "qualification": "new 2026 paper-equation diagnostic; no independent end-to-end floating-point NN proof",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cuda_memory_cap_bytes": memory_cap,
            "preloaded_binaries": binaries,
        }
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
        steps = []
        original_advance = driver.advance_sparse
        with (output / "observations.jsonl").open("x") as observations:
            def observed(*call_args, **call_kwargs):
                state, accepted = original_advance(*call_args, **call_kwargs)
                accepted_cpu = accepted.detach().cpu().tolist()
                status_cpu = state.status.detach().cpu().tolist()
                step_number = len(steps) + 1
                row = {
                    "substep": step_number,
                    "accepted_count": sum(accepted_cpu),
                    "status_counts": {str(value): status_cpu.count(value) for value in sorted(set(status_cpu))},
                    "rejected_lanes": [i for i, value in enumerate(accepted_cpu) if not value],
                }
                observations.write(json.dumps(row) + "\n")
                if step_number % 20 == 0 or row["rejected_lanes"]:
                    observations.flush()
                    (output / "progress.json").write_text(json.dumps(row) + "\n")
                steps.append(row)
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(DRIVER), str(path), "--device", "cuda:0", "--engine", "sparse",
                    "--crown-domain", "box", "--crown-relax", "same-slope",
                    "--crown-transport", "native-f64", "--crown-input-layout", "native",
                    "--nn-mode", "crown", "--print-final-hull", "--metrics-json",
                    str(output / "metrics.json")]
            record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            try:
                driver_return = driver.main()
            finally:
                sys.argv = previous
        expected_substeps = config["steps"] * 20
        all_accepted = len(steps) == expected_substeps and all(
            row["accepted_count"] == expected_boxes for row in steps
        )
        metrics_path = output / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
        final_hull = metrics.get("final_hull") if metrics else None
        target = (final_hull is not None and "x3" in final_hull
                  and final_hull["x3"][0] >= 0.94 and final_hull["x3"][1] <= 1.06)
        result.update(
            status="completed" if driver_return == 0 and all_accepted else "incomplete",
            driver_return=driver_return,
            expected_substeps=expected_substeps,
            completed_substeps=len(steps),
            expected_lane_substeps=expected_boxes * expected_substeps,
            accepted_lane_substeps=sum(row["accepted_count"] for row in steps),
            all_lanes_accepted=all_accepted,
            final_hull=final_hull,
            endpoint_target_inside=bool(target) if args.mode == "full" else None,
            metrics_broken=metrics.get("broken") if metrics else None,
            driver_elapsed_s=metrics.get("elapsed_s") if metrics else None,
            wall_s=time.perf_counter() - started,
            end_to_end_floating_point_nn_certificate=False,
        )
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      completed_substeps=len(steps) if "steps" in locals() else 0,
                      accepted_lane_substeps=sum(row["accepted_count"] for row in steps) if "steps" in locals() else 0,
                      wall_s=time.perf_counter() - started)
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke1", "batch1", "batch2", "full"), required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
