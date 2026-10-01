#!/usr/bin/env python3
"""Isolated ACC participant-order Huan/Xiangru run using prebuilt CUDA libraries."""

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
X = Path("/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream")
ENGINES = {"huan": N / "engine_huan_sr_chunk", "xiangru": X}
DRIVER = X / "src/flowstar_gpu/integrations/crown_reach.py"
ADAPTER = N / "repo_sr_prepare_entry/experiments/flowstar_acceleration/nncs_acc_adapter.py"
MODEL = N / "runs/archcomp26_20261001/acc_prep_001/official_acc_controller_5_20.onnx"


def prohibited(*_args, **_kwargs):
    raise RuntimeError("content digest or CUDA extension build disabled")


def load_module(name, path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import saved source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(backend):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("ACC attempt is reserved for physical GPU 2")
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
        raise RuntimeError("saved PyTorch/CUDA environment unavailable")
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
    driver = load_module("archcomp26_acc_" + backend + "_driver", DRIVER)
    if not hasattr(driver.sparse_exec_module, "COMPOSE_PARENT_ASSEMBLY"):
        driver.sparse_exec_module.COMPOSE_PARENT_ASSEMBLY = None
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_cap = min(11 * 2**30, int(total_memory * 0.9))
    torch.cuda.set_per_process_memory_fraction(memory_cap / total_memory)
    return torch, driver, engine, binaries, memory_cap


def checked_config(source, mode, output):
    import yaml
    config = yaml.safe_load(source.read_text())
    assert config["adapter"] == "acc_exact_feature_tm"
    assert config["controller_input"] == ["30", "1.4", "v_ego", "x_lead-x_ego", "v_lead-v_ego"]
    assert config["num_vars"] == 8 and config["num_nn_input"] == 5 and config["num_nn_output"] == 1
    assert config["steps"] == 50 and config["step_size"] == config["ode_step_size"] == 0.1
    assert config["ode_order"] == 3 and config["sr_queue"] == 50
    assert [(v["name"], v["interval"]) for v in config["initial_set"]] == [
        ("x_lead", [90.0, 110.0]), ("v_lead", [32.0, 32.2]), ("a_lead", [0.0, 0.0]),
        ("x_ego", [10.0, 11.0]), ("v_ego", [30.0, 30.2]), ("a_ego", [0.0, 0.0]),
        ("t", [0.0, 0.0]), ("u1", [0.0, 0.0])]
    assert config["dynamics_expressions"] == [
        "v_lead", "a_lead", "-2 * a_lead - 4 - 0.0001 * v_lead^2",
        "v_ego", "a_ego", "-2 * a_ego + 2 * u1 - 0.0001 * v_ego^2", "1", "0"]
    assert config["constraints_safe"] == ["-x_lead+x_ego+1.4*v_ego+10"]
    assert config["split_vars"] == [] and config["model_dir"] == str(MODEL) and MODEL.is_file()
    assert config["input_shape"] == [-1, 1, 1, 5]
    if mode == "smoke1":
        config["steps"] = 1
    generated = output / "config.yaml"
    generated.write_text(yaml.safe_dump(config, sort_keys=False))
    return config, generated


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "exception", "backend": args.backend, "mode": args.mode}
    records = []
    safety = []
    try:
        torch, driver, engine, binaries, memory_cap = prepare(args.backend)
        config, generated = checked_config(args.source_config, args.mode, output)
        adapter = load_module("archcomp26_acc_feature_adapter", ADAPTER)
        adapter_state = adapter.install(driver, config)

        def official_build(cfg, device, relax="same-slope", input_layout="native"):
            if input_layout != "native" or relax != "same-slope":
                raise RuntimeError("unexpected ACC controller layout or relaxation")
            from auto_LiRPA import BoundedModule
            raw = driver.build_raw_net(cfg, experimental=False)
            return BoundedModule(raw, torch.zeros(1, *cfg["input_shape"][1:], dtype=torch.float64),
                                 device=device, bound_opts=dict(cfg["bound_opts"]))

        driver.build_crown = official_build
        cells = driver.make_cells(config)
        if tuple(cells.shape) != (1, 8, 2):
            raise RuntimeError(f"ACC full initial box mismatch: {tuple(cells.shape)}")
        start_record = {
            "schema": "archcomp26-acc-participant-gpu-nohash-v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend, "mode": args.mode,
            "source_config": str(args.source_config), "generated_config": str(generated),
            "engine": str(engine), "driver": str(DRIVER), "adapter": str(ADAPTER), "model": str(MODEL),
            "physical_box_count": 1, "physical_states": ["x_lead", "v_lead", "a_lead", "x_ego", "v_ego", "a_ego"],
            "controller_input": config["controller_input"], "property": "x_lead-x_ego-1.4*v_ego-10 >= 0 for all t in [0,T]",
            "horizon_s": config["steps"] * 0.1, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "cuda_memory_cap_bytes": memory_cap,
            "preloaded_binaries": binaries,
            "qualification": "author GPU checker plus recorded interval scan; no independent floating-point NN proof",
        }
        (output / "START.json").write_text(json.dumps(start_record, indent=2) + "\n")

        original_ranges = driver.SpecGroup.ranges
        with (output / "ranges.jsonl").open("x") as ranges, (output / "safety.jsonl").open("x") as safefile:
            def capture_safe(group, boxes, active, **kwargs):
                q = original_ranges(group, boxes, active, **kwargs)
                if q.numel():
                    event = {"substep": len(records), "bounds": q.detach().cpu().tolist(),
                             "max_violation_upper": float(q[..., 1].max().item())}
                    safety.append(event)
                    safefile.write(json.dumps(event, allow_nan=False) + "\n")
                    safefile.flush()
                return q

            driver.SpecGroup.ranges = capture_safe
            original_advance = driver.advance_sparse

            def observed(*call_args, **call_kwargs):
                state, accepted = original_advance(*call_args, **call_kwargs)
                eng, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, eng, 6).detach().cpu().tolist()
                h = float(settings.step)
                endpoint_time = torch.full((1, 2), h, dtype=torch.float64, device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, eng, endpoint_time, 6).detach().cpu().tolist()
                row = {"substep": len(records) + 1, "accepted": bool(accepted[0].item()),
                       "local_h": h, "tube": tube[0], "endpoint": endpoint[0]}
                records.append(row)
                ranges.write(json.dumps(row, allow_nan=False) + "\n")
                ranges.flush()
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(DRIVER), str(generated), "--device", "cuda:0", "--engine", "sparse",
                    "--crown-domain", "box", "--crown-relax", "same-slope",
                    "--crown-transport", "rpc-float32", "--crown-input-layout", "native",
                    "--print-final-hull", "--metrics-json", str(output / "metrics.json"), "--strict"]
            start_record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(start_record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            try:
                driver_return = driver.main()
            finally:
                sys.argv = previous

        expected = config["steps"]
        complete = driver_return == 0 and len(records) == expected and all(r["accepted"] for r in records)
        safe = bool(safety) and all(s["max_violation_upper"] <= 0 for s in safety)
        margin_lower = min(r["tube"][0][0] - r["tube"][3][1] - 1.4 * r["tube"][4][1] - 10 for r in records)
        metrics_path = output / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        result.update(status="completed" if complete else "incomplete", driver_return=driver_return,
                      completed_substeps=len(records), expected_substeps=expected,
                      safety_events=len(safety), author_safe_bounds_all_nonpositive=safe,
                      independent_tube_halfspace_min_lower=margin_lower,
                      independent_tube_boxes_all_safe=margin_lower >= 0,
                      author_checker_metrics_broken=metrics.get("broken"), final_hull=metrics.get("final_hull"),
                      adapter_feature_calls=adapter_state["feature_calls"],
                      adapter_injection_calls=adapter_state["injection_calls"],
                      wall_s=time.perf_counter() - started, end_to_end_floating_point_nn_certificate=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      completed_substeps=len(records), wall_s=time.perf_counter() - started)
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" and result["author_safe_bounds_all_nonpositive"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=ENGINES, required=True)
    parser.add_argument("--mode", choices=("smoke1", "full"), required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
