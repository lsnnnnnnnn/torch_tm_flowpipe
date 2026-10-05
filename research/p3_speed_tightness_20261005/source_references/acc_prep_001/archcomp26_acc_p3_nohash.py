#!/usr/bin/env python3
"""ACC participant-order diagnostic on the saved working-P3 numerical engine."""

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


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
ENGINE = N / "engine_quad_normalization_center"
DRIVER = Path("/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream/src/flowstar_gpu/integrations/crown_reach.py")
ADAPTER = N / "repo_sr_prepare_entry/experiments/flowstar_acceleration/nncs_acc_adapter.py"
MODEL = N / "runs/archcomp26_20261001/acc_prep_001/official_acc_controller_5_20.onnx"
CACHE = N / "cache_private_entry/py311_torch251_cu126_gcc13"
RECIP_CACHE = N / "cache_reciprocal_geometric_20260928"
RECIP_SOURCE = N / "runs/quad_control_transfer_20260927/reciprocal_five_cuda_v2/sources/tape_candidate.py"
RECIP_CORE = N / "runs/reciprocal_geometric_20260928/core.py"
ENDPOINT = N / "runs/quad_targeted_recovery_20260927/strict_endpoint.py"
EXTENSIONS = {
    "flowstar_seg_kernels": CACHE / "flowstar_seg_kernels/flowstar_seg_kernels.so",
    "flowstar_sr_interval_matmul": CACHE / "flowstar_sr_interval_matmul/flowstar_sr_interval_matmul.so",
    "flowstar_sr_history_sum": CACHE / "flowstar_sr_history_sum/flowstar_sr_history_sum.so",
    "flowstar_injective_index_v2": CACHE / "flowstar_injective_index_v2/flowstar_injective_index_v2.so",
    "flowstar_seg_private_output_v1": CACHE / "flowstar_seg_private_output_v1/flowstar_seg_private_output_v1.so",
    "horner_edge_1312fa8b2aed": N / "cache_horner_entry/horner_edge_1312fa8b2aed/horner_edge_1312fa8b2aed.so",
    "flowstar_recip_geom_replay_1f9efda6325c": RECIP_CACHE / "flowstar_recip_geom_replay_1f9efda6325c/flowstar_recip_geom_replay_1f9efda6325c.so",
    "flowstar_recip_geom_valid_1f9efda6325c": RECIP_CACHE / "flowstar_recip_geom_valid_1f9efda6325c/flowstar_recip_geom_valid_1f9efda6325c.so",
}
ENV = {
    "FLOWSTAR_WEIGHTED_VALIDATION": "0", "FLOWSTAR_RECENTER_VALIDATION": "1",
    "FLOWSTAR_SELF_MAP_RETRIES": "8", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "FLOWSTAR_GLUE": "graph", "FLOWSTAR_INJECTIVE_MAPS": "1",
    "FLOWSTAR_CENTER_NORMALIZATION": "1", "FLOWSTAR_VALIDATION_POLICY": "solution_plus_one",
    "FLOWSTAR_COMPOSITION": "horner", "FLOWSTAR_INJECTIVE_GLUE": "1",
    "FLOWSTAR_SUPPORT_POLICY": "structural", "FLOWSTAR_EARLY_WEIGHTED": "1",
    "FLOWSTAR_HORNER_EDGE_CACHE": str(N / "cache_horner_entry"),
    "TORCH_EXTENSIONS_DIR": str(CACHE), "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
}


def prohibited(*_args, **_kwargs):
    raise RuntimeError("content digest or CUDA extension build disabled")


def load(name, path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def guard():
    hashlib.sha256 = prohibited
    original_new = hashlib.new

    def guarded_new(name, *args, **kwargs):
        if name.lower().replace("-", "") == "sha256":
            return prohibited()
        return original_new(name, *args, **kwargs)

    hashlib.new = guarded_new
    import torch.utils.cpp_extension as extension
    extension.load = extension.load_inline = prohibited


def check_config(source, output, mode):
    import yaml
    cfg = yaml.safe_load(source.read_text())
    if not (cfg["num_vars"] == 8 and cfg["num_nn_input"] == 5 and cfg["num_nn_output"] == 1
            and cfg["steps"] == 50 and cfg["step_size"] == cfg["ode_step_size"] == 0.1
            and cfg["ode_order"] == 3 and cfg["sr_queue"] == 50
            and cfg["adapter"] == "acc_exact_feature_tm"
            and cfg["controller_input"] == ["30", "1.4", "v_ego", "x_lead-x_ego", "v_lead-v_ego"]
            and cfg["constraints_safe"] == ["-x_lead+x_ego+1.4*v_ego+10"]
            and cfg["split_vars"] == [] and cfg["model_dir"] == str(MODEL) and MODEL.is_file()
            and cfg["input_shape"] == [-1, 1, 1, 5]):
        raise ValueError("ACC participant-order contract differs")
    if mode == "smoke1":
        cfg["steps"] = 1
    generated = output / "config.yaml"
    generated.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg, generated


def prepare(cfg):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("ACC P3 attempt is reserved for physical GPU 2")
    for key, value in ENV.items():
        os.environ[key] = value
    sys.path.insert(0, str(ENGINE / "src"))
    guard()
    import torch
    if torch.__version__ != "2.5.1+cu121" or not torch.cuda.is_available():
        raise RuntimeError("saved PyTorch/CUDA runtime unavailable")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    cap = min(11 * 2**30, int(total * 0.9))
    torch.cuda.set_per_process_memory_fraction(cap / total)

    import flowstar_gpu
    from flowstar_gpu import cuda_kernels as ck, elementary as elem
    ck.load_cuda_extension = prohibited
    ck._ext = load("flowstar_seg_kernels", EXTENSIONS["flowstar_seg_kernels"])
    ck._tried = True
    core = load("acc_p3_reciprocal_core", RECIP_CORE)
    for name in ("rec_series_valid_g", "rec_series_valid", "rec_series_replay"):
        setattr(elem, name, getattr(core, name))
    tape = load("flowstar_gpu.tape_kernels", RECIP_SOURCE)
    tape.load_cuda_extension = prohibited
    tape._ext = load("flowstar_recip_geom_replay_1f9efda6325c", EXTENSIONS["flowstar_recip_geom_replay_1f9efda6325c"])
    tape._vext = load("flowstar_recip_geom_valid_1f9efda6325c", EXTENSIONS["flowstar_recip_geom_valid_1f9efda6325c"])
    tape._tried = tape._vtried = True
    flowstar_gpu.tape_kernels = tape
    for module_name, extension_name in (
        ("sr_kernels", "flowstar_sr_interval_matmul"),
        ("sr_sum_kernels", "flowstar_sr_history_sum"),
        ("injective_index", "flowstar_injective_index_v2"),
        ("private_output_kernels", "flowstar_seg_private_output_v1"),
        ("horner_edge_kernels", "horner_edge_1312fa8b2aed"),
    ):
        module = __import__("flowstar_gpu." + module_name, fromlist=[module_name])
        module._ext = load(extension_name, EXTENSIONS[extension_name])
        if hasattr(module, "_tried"):
            module._tried = True
    if not (ck.available() and tape.available() and tape.valid_available()):
        raise RuntimeError("saved P3 CUDA libraries unavailable")

    driver = load("archcomp26_acc_p3_driver", DRIVER)
    from flowstar_gpu import sparse_exec as se
    if not hasattr(se, "COMPOSE_PARENT_ASSEMBLY"):
        se.COMPOSE_PARENT_ASSEMBLY = None  # Shared driver metrics only.
    endpoint = load("archcomp26_acc_p3_strict_endpoint", ENDPOINT)
    driver.end_of_time_s = endpoint.end_of_time_s
    adapter = load("archcomp26_acc_p3_feature_adapter", ADAPTER)
    adapter_state = adapter.install(driver, cfg)

    def official_build(config, device, relax="same-slope", input_layout="native"):
        if relax != "same-slope" or input_layout != "native":
            raise RuntimeError("unexpected ACC controller layout or relaxation")
        from auto_LiRPA import BoundedModule
        raw = driver.build_raw_net(config, experimental=False)
        return BoundedModule(raw, torch.zeros(1, *config["input_shape"][1:], dtype=torch.float64),
                             device=device, bound_opts=dict(config["bound_opts"]))

    driver.build_crown = official_build
    return torch, driver, adapter_state, cap


def execute(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "exception", "completed_substeps": 0}
    record = {
        "schema": "archcomp26-acc-participant-p3-nohash-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
        "source_config": str(args.source_config), "engine": str(ENGINE), "shared_driver": str(DRIVER),
        "feature_adapter": str(ADAPTER), "model": str(MODEL),
        "controller_input": ["30", "1.4", "v_ego", "x_lead-x_ego", "v_lead-v_ego"],
        "method": "working-P3 core, validation-P4 flags, strict endpoint, ACC exact feature TM adapter",
        "property": "x_lead-x_ego-1.4*v_ego-10 >= 0 for all t in [0,T]",
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": ENV,
        "prebuilt_libraries": {name: {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                               for name, path in EXTENSIONS.items()},
        "qualification": "new ACC P3 diagnostic; no independent floating-point neural-bound proof",
    }
    (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
    rows = []
    safety = []
    try:
        cfg, generated = check_config(args.source_config, output, args.mode)
        torch, driver, adapter_state, cap = prepare(cfg)
        cells = driver.make_cells(cfg)
        if tuple(cells.shape) != (1, 8, 2):
            raise ValueError(f"ACC initial box mismatch: {tuple(cells.shape)}")
        record.update(generated_config=str(generated), expected_substeps=cfg["steps"],
                      cuda_memory_cap_bytes=cap)
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
        original_safe = driver.SpecGroup.ranges
        original_advance = driver.advance_sparse
        with (output / "ranges.jsonl").open("x") as ranges, (output / "safety.jsonl").open("x") as safefile:
            def capture_safe(group, boxes, active, **kwargs):
                q = original_safe(group, boxes, active, **kwargs)
                if q.numel():
                    event = {"substep": len(rows), "bounds": q.detach().cpu().tolist(),
                             "max_violation_upper": float(q[..., 1].max().item())}
                    safety.append(event)
                    safefile.write(json.dumps(event, allow_nan=False) + "\n")
                    safefile.flush()
                return q

            driver.SpecGroup.ranges = capture_safe

            def observed(*call_args, **call_kwargs):
                state, accepted = original_advance(*call_args, **call_kwargs)
                eng, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, eng, 6).detach().cpu().tolist()
                endpoint_time = torch.full((1, 2), float(settings.step), dtype=torch.float64,
                                           device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, eng, endpoint_time, 6).detach().cpu().tolist()
                row = {"substep": len(rows) + 1, "accepted": bool(accepted[0].item()),
                       "local_h": float(settings.step), "tube": tube[0], "endpoint": endpoint[0]}
                rows.append(row)
                ranges.write(json.dumps(row, allow_nan=False) + "\n")
                ranges.flush()
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(DRIVER), str(generated), "--device", "cuda:0", "--engine", "sparse",
                    "--strict", "--order", "3", "--crown-domain", "box",
                    "--crown-relax", "same-slope", "--crown-transport", "rpc-float32",
                    "--crown-input-layout", "native", "--nn-mode", "crown",
                    "--print-final-hull", "--metrics-json", str(output / "metrics.json")]
            record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            try:
                code = driver.main()
            finally:
                sys.argv = previous

        complete = code == 0 and len(rows) == cfg["steps"] and all(r["accepted"] for r in rows)
        author_safe = bool(safety) and all(s["max_violation_upper"] <= 0 for s in safety)
        min_margin = min(r["tube"][0][0] - r["tube"][3][1] - 1.4 * r["tube"][4][1] - 10 for r in rows)
        result.update(status="completed" if complete else "incomplete", driver_return=code,
                      expected_substeps=cfg["steps"], completed_substeps=len(rows),
                      all_substeps_accepted=complete, safety_events=len(safety),
                      author_safe_bounds_all_nonpositive=author_safe,
                      independent_tube_halfspace_min_lower=min_margin,
                      independent_tube_boxes_all_safe=min_margin >= 0,
                      adapter_feature_calls=adapter_state["feature_calls"],
                      adapter_injection_calls=adapter_state["injection_calls"],
                      t_endpoint=rows[-1]["endpoint"] if rows else None,
                      wall_s=time.perf_counter() - started,
                      end_to_end_floating_point_nn_certificate=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      completed_substeps=len(rows), wall_s=time.perf_counter() - started)
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" and result["author_safe_bounds_all_nonpositive"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke1", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
