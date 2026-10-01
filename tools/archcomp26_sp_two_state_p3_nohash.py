#!/usr/bin/env python3
"""Run the paper two-state Single Pendulum on the saved working-P3 core."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
HELPER = N / "runs/archcomp26_20261001/acc_prep_001/archcomp26_acc_p3_nohash.py"
ENGINE = N / "engine_quad_normalization_center"
DRIVER = Path("/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream/src/flowstar_gpu/integrations/crown_reach.py")
MODEL = N / "runs/archcomp26_20261001/single_pendulum_prep_001/controller_single_pendulum.onnx"
REQUIRED_STATES = [("x1", [1, 1.175]), ("x2", [0, 0.2]), ("t", [0, 0]), ("u1", [0, 0])]


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


def checked_config(source, mode, output):
    import yaml
    cfg = yaml.safe_load(source.read_text())
    if not (
        cfg["num_vars"] == 4 and cfg["num_nn_input"] == 2 and cfg["num_nn_output"] == 1
        and cfg["steps"] == 20 and cfg["step_size"] == 0.05
        and cfg["ode_step_size"] == 0.01 and cfg["ode_order"] == 2
        and [(v["name"], v["interval"]) for v in cfg["initial_set"]] == REQUIRED_STATES
        and cfg["dynamics_expressions"] == ["x2", "2 * sin(x1) + 8 * u1", "1", "0"]
        and cfg["constraints_safe"] == ["-x1", "x1 - 1"]
        and cfg["constraints_safe_from"] == 0.5 and cfg["constraints_safe_until"] == 1.0
        and not cfg.get("constraints_unsafe") and not cfg.get("constraints_target")
        and cfg["model_dir"] == str(MODEL) and MODEL.is_file()
        and cfg["input_shape"] == [-1, 2]
        and cfg["sr_queue"] == 1000
    ):
        raise ValueError("paper two-state Single Pendulum contract differs")
    if mode == "smoke1":
        cfg["steps"] = 1
        cfg["constraints_safe"] = []  # T=.05 does not reach the property window.
        cfg.pop("constraints_safe_from")
        cfg.pop("constraints_safe_until")
    generated = output / "config.yaml"
    generated.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg, generated


def prepare():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("this P3 attempt is reserved for physical GPU 2")
    helper = load("archcomp26_acc_p3_preload_helper", HELPER)
    for key, value in helper.ENV.items():
        os.environ[key] = value
    sys.path.insert(0, str(ENGINE / "src"))
    helper.guard()
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
    ck.load_cuda_extension = helper.prohibited
    ck._ext = load("flowstar_seg_kernels", helper.EXTENSIONS["flowstar_seg_kernels"])
    ck._tried = True
    core = load("sp_p3_reciprocal_core", helper.RECIP_CORE)
    for name in ("rec_series_valid_g", "rec_series_valid", "rec_series_replay"):
        setattr(elem, name, getattr(core, name))
    tape = load("flowstar_gpu.tape_kernels", helper.RECIP_SOURCE)
    tape.load_cuda_extension = helper.prohibited
    tape._ext = load("flowstar_recip_geom_replay_1f9efda6325c", helper.EXTENSIONS["flowstar_recip_geom_replay_1f9efda6325c"])
    tape._vext = load("flowstar_recip_geom_valid_1f9efda6325c", helper.EXTENSIONS["flowstar_recip_geom_valid_1f9efda6325c"])
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
        module._ext = load(extension_name, helper.EXTENSIONS[extension_name])
        if hasattr(module, "_tried"):
            module._tried = True
    if not (ck.available() and tape.available() and tape.valid_available()):
        raise RuntimeError("saved P3 CUDA libraries unavailable")

    driver = load("archcomp26_sp_p3_driver", DRIVER)
    from flowstar_gpu import sparse_exec as se
    if not hasattr(se, "COMPOSE_PARENT_ASSEMBLY"):
        se.COMPOSE_PARENT_ASSEMBLY = None
    endpoint = load("archcomp26_sp_p3_strict_endpoint", helper.ENDPOINT)
    driver.end_of_time_s = endpoint.end_of_time_s
    return helper, torch, driver, cap


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, safety = [], []
    result = {"status": "exception", "mode": args.mode, "completed_substeps": 0}
    record = {
        "schema": "archcomp26-sp-two-state-p3-nohash-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "source_config": str(args.source_config),
        "engine": str(ENGINE), "shared_driver": str(DRIVER), "preload_helper": str(HELPER),
        "model": str(MODEL), "physical_states": ["x1", "x2"],
        "auxiliary_states": {"t": "t(0)=0,t'=1,checker only", "u1": "held control,u1'=0"},
        "property": "x1 in [0,1] for every t in the closed [0.5,1] window" if args.mode == "full" else "plumbing only; window not reached",
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "qualification": "two physical states; not an undocumented three-state MATLAB execution; no independent floating-point NN proof",
    }
    (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        cfg, generated = checked_config(args.source_config, args.mode, output)
        helper, torch, driver, cap = prepare()
        cells = driver.make_cells(cfg)
        if tuple(cells.shape) != (1, 4, 2):
            raise ValueError(f"initial box mismatch: {tuple(cells.shape)}")
        record.update(generated_config=str(generated), expected_substeps=cfg["steps"] * 5,
                      cuda_memory_cap_bytes=cap, environment=helper.ENV,
                      prebuilt_libraries={name: {"path": str(path), "bytes": path.stat().st_size,
                      "mtime_ns": path.stat().st_mtime_ns} for name, path in helper.EXTENSIONS.items()})
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
        original_advance = driver.advance_sparse
        original_safe = driver.SpecGroup.ranges
        with (output / "ranges.jsonl").open("x") as ranges, (output / "safety.jsonl").open("x") as safefile:
            def capture_safe(group, boxes, active, **kwargs):
                q = original_safe(group, boxes, active, **kwargs)
                if q.numel():
                    event = {"event_index": len(safety) + 1, "bounds": q.detach().cpu().tolist(),
                             "max_violation_upper": float(q[..., 1].max().item())}
                    safety.append(event)
                    safefile.write(json.dumps(event, allow_nan=False) + "\n")
                    safefile.flush()
                return q

            def observed(*call_args, **call_kwargs):
                state, accepted = original_advance(*call_args, **call_kwargs)
                eng, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, eng, 2).detach().cpu().tolist()
                endpoint_time = torch.full((1, 2), float(settings.step), dtype=torch.float64,
                                           device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, eng, endpoint_time, 2).detach().cpu().tolist()
                row = {"substep": len(rows) + 1, "accepted": bool(accepted[0].item()),
                       "local_h": float(settings.step), "tube": tube[0], "endpoint": endpoint[0]}
                rows.append(row)
                ranges.write(json.dumps(row, allow_nan=False) + "\n")
                ranges.flush()
                return state, accepted

            driver.SpecGroup.ranges = capture_safe
            driver.advance_sparse = observed
            argv = [str(DRIVER), str(generated), "--device", "cuda:0", "--engine", "sparse",
                    "--strict", "--order", "2", "--crown-domain", "box",
                    "--crown-relax", "same-slope", "--crown-transport", "native-f64",
                    "--crown-input-layout", "native", "--nn-mode", "crown",
                    "--print-final-hull", "--metrics-json", str(output / "metrics.json")]
            record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            try:
                driver_return = driver.main()
            finally:
                sys.argv = previous
        expected = cfg["steps"] * 5
        complete = driver_return == 0 and len(rows) == expected and all(r["accepted"] for r in rows)
        window = rows[50:] if args.mode == "full" and len(rows) == 100 else []
        min_x1 = min((r["tube"][0][0] for r in window), default=None)
        max_x1 = max((r["tube"][0][1] for r in window), default=None)
        metrics = json.loads((output / "metrics.json").read_text()) if (output / "metrics.json").is_file() else {}
        result.update(status="completed" if complete else "incomplete", driver_return=driver_return,
                      expected_substeps=expected, completed_substeps=len(rows),
                      all_substeps_accepted=complete, safety_events=len(safety),
                      author_safe_bounds_all_nonpositive=(len(safety) == 50 and
                         all(s["max_violation_upper"] <= 0 for s in safety)) if args.mode == "full" else None,
                      window_tube_x1_union=[min_x1, max_x1] if window else None,
                      independent_window_tube_boxes_safe=(min_x1 >= 0 and max_x1 <= 1) if window else None,
                      t_endpoint=rows[-1]["endpoint"] if rows else None,
                      metrics_broken=metrics.get("broken"), driver_elapsed_s=metrics.get("elapsed_s"),
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
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke1", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
