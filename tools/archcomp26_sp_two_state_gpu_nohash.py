#!/usr/bin/env python3
"""Run one isolated Single Pendulum paper two-state GPU attempt."""

import argparse
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
ENGINES = {
    "huan": N / "engine_huan_sr_chunk",
    "xiangru": Path("/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream"),
}
MODEL = N / "runs/archcomp26_20261001/single_pendulum_prep_001/controller_single_pendulum.onnx"


def prohibited(*_args, **_kwargs):
    raise RuntimeError("content digest or CUDA extension build disabled")


def load_module(name, path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import saved module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(backend):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("this attempt is reserved for physical GPU 2")
    engine = ENGINES[backend]
    cache = N / "cache_four_way_v1" / backend
    for key in list(os.environ):
        if key.startswith("FLOWSTAR_"):
            del os.environ[key]
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(engine / "src"))
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
        path = cache / name / (name + ".so")
        module = load_module(name, path)
        for export in exports:
            if not callable(getattr(module, export, None)):
                raise RuntimeError(f"saved CUDA library lacks {export}: {path}")
        setattr(owner, attr, module)
        setattr(owner, tried, True)
        stat = path.stat()
        binaries[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not (ck.available() and tk.available() and tk.valid_available()):
        raise RuntimeError("preloaded CUDA libraries unavailable")
    driver_path = (engine / "integrations/crown_reach/gpu_driver.py" if backend == "huan"
                   else engine / "src/flowstar_gpu/integrations/crown_reach.py")
    driver = load_module("archcomp26_sp_two_state_" + backend + "_driver", driver_path)
    if hasattr(driver, "sparse_exec_module") and not hasattr(driver.sparse_exec_module, "COMPOSE_PARENT_ASSEMBLY"):
        driver.sparse_exec_module.COMPOSE_PARENT_ASSEMBLY = None
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_cap = min(12 * 2**30, int(total_memory * 0.9))
    torch.cuda.set_per_process_memory_fraction(memory_cap / total_memory)
    return torch, driver, engine, driver_path, binaries, memory_cap


def checked_config(source, mode, output):
    import yaml
    config = yaml.safe_load(source.read_text())
    assert config["num_vars"] == 4 and config["num_nn_input"] == 2 and config["num_nn_output"] == 1
    assert config["steps"] == 20 and config["step_size"] == 0.05
    assert config["ode_step_size"] == 0.01 and config["ode_order"] == 2
    assert [(v["name"], v["interval"]) for v in config["initial_set"]] == [
        ("x1", [1, 1.175]), ("x2", [0, 0.2]), ("t", [0, 0]), ("u1", [0, 0])]
    assert config["dynamics_expressions"] == ["x2", "2 * sin(x1) + 8 * u1", "1", "0"]
    assert config["constraints_safe"] == ["-x1", "x1 - 1"]
    assert config["constraints_safe_from"] == 0.5 and config["constraints_safe_until"] == 1.0
    assert not config.get("constraints_unsafe") and not config.get("constraints_target")
    assert config["model_dir"] == str(MODEL) and MODEL.is_file()
    assert config["input_shape"] == [-1, 2]
    if mode == "smoke1":
        config["steps"] = 1
        config["constraints_safe"] = []  # t=.05 does not reach the official window.
        config.pop("constraints_safe_from")
        config.pop("constraints_safe_until")
    generated = output / "config.yaml"
    generated.write_text(yaml.safe_dump(config, sort_keys=False))
    return config, generated


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "exception", "backend": args.backend, "mode": args.mode}
    steps = []
    try:
        torch, driver, engine, driver_path, binaries, memory_cap = prepare(args.backend)
        config, generated = checked_config(args.source_config, args.mode, output)
        cells = driver.make_cells(config)
        if tuple(cells.shape) != (1, 4, 2):
            raise ValueError(f"initial partition mismatch: {tuple(cells.shape)}")
        record = {
            "schema": "archcomp26-sp-two-state-gpu-nohash-v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend, "mode": args.mode,
            "source_config": str(args.source_config), "generated_config": str(generated),
            "engine": str(engine), "driver": str(driver_path), "model": str(MODEL),
            "physical_box_count": 1, "physical_states": ["x1", "x2"],
            "auxiliary_states": ["t", "u1"],
            "property_window": [0.5, 1.0] if args.mode == "full" else None,
            "property_note": "smoke1 checks plumbing only" if args.mode == "smoke1" else "x1 in [0,1] at all times in the closed window",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cuda_memory_cap_bytes": memory_cap, "preloaded_binaries": binaries,
            "qualification": "author checker result; no independent floating-point NN certificate",
        }
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
        original_advance = driver.advance_sparse
        with (output / "ranges.jsonl").open("x") as ranges:
            def observed(*call_args, **call_kwargs):
                state, accepted = original_advance(*call_args, **call_kwargs)
                engine_obj, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, engine_obj, 2).detach().cpu().tolist()
                endpoint_time = torch.full((1, 2), float(settings.step), dtype=torch.float64,
                                           device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, engine_obj,
                                                               endpoint_time, 2).detach().cpu().tolist()
                row = {"substep": len(steps) + 1, "accepted": bool(accepted[0].item()),
                       "tube": tube[0], "endpoint": endpoint[0]}
                ranges.write(json.dumps(row) + "\n")
                ranges.flush()
                steps.append(row)
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(driver_path), str(generated), "--device", "cuda:0", "--engine", "sparse",
                    "--crown-domain", "box", "--crown-relax", "same-slope", "--print-final-hull",
                    "--metrics-json", str(output / "metrics.json")]
            if args.backend == "xiangru":
                argv.extend(["--crown-transport", "native-f64", "--crown-input-layout", "native",
                             "--nn-mode", "crown"])
            record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            try:
                driver_return = driver.main()
            finally:
                sys.argv = previous
        expected_substeps = config["steps"] * 5
        complete = len(steps) == expected_substeps and all(row["accepted"] for row in steps)
        metrics_path = output / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        result.update(status="completed" if driver_return == 0 and complete else "incomplete",
                      driver_return=driver_return, completed_substeps=len(steps),
                      expected_substeps=expected_substeps, all_substeps_accepted=complete,
                      final_hull=metrics.get("final_hull"), metrics_broken=metrics.get("broken"),
                      driver_elapsed_s=metrics.get("elapsed_s"),
                      property_window=[0.5, 1.0] if args.mode == "full" else None,
                      wall_s=time.perf_counter() - started,
                      end_to_end_floating_point_nn_certificate=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      completed_substeps=len(steps), wall_s=time.perf_counter() - started)
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=ENGINES, required=True)
    parser.add_argument("--mode", choices=("smoke1", "full"), required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
