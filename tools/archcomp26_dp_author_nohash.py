#!/usr/bin/env python3
"""Isolated 2026 DP less run on an author's original GPU plant engine.

This is a diagnostic of the author arithmetic and common 225-cell controller
driver. It neither computes a content digest nor asserts an end-to-end proof.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
X = Path("/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream")
DRIVER = X / "src/flowstar_gpu/integrations/crown_reach.py"
BASE_CONFIG = N / "runs/archcomp_review_20260923/contracts/double_pendulum_less_robust.yaml"
BOXES = N / "runs/archcomp_review_20260923/double_pendulum_less_robust_boxes.json"
MODEL = Path("/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/Double_Pendulum/controller_double_pendulum_less_robust.onnx")
MORE_ROOT = N / "runs/archcomp26_20261001"
MORE_MODEL = MORE_ROOT / "native_dp_more_prep_001/controller_double_pendulum_more_robust.onnx"
MORE_BOXES = MORE_ROOT / "native_dp_more_build_001/initial_boxes.json"
ENGINES = {
    "huan": Path("/srv/local/shengenli/flowstar-gpu"),
    "xiangru": X,
}
def prohibited(*_args, **_kwargs):
    raise RuntimeError("content digest or CUDA extension build prohibited for this attempt")


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
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(backend):
    engine = ENGINES[backend]
    cache = N / "cache_four_way_v1" / backend
    for key in list(os.environ):
        if key.startswith("FLOWSTAR_"):
            del os.environ[key]
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path[:0] = [str(engine / "src"), str(N / "repo_sr_prepare_entry"), str(N / "repo_sr_prepare_entry/src")]
    guard_digests_and_builds()
    import torch
    from flowstar_gpu import cuda_kernels, tape_kernels, determinism

    if not torch.cuda.is_available() or torch.__version__ != "2.5.1+cu121":
        raise RuntimeError("expected saved PyTorch/CUDA environment is unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not inspect.signature(determinism.enable_determinism).parameters:
        original = determinism.enable_determinism
        determinism.enable_determinism = lambda device="cuda": original()
    cuda_kernels._ext = load_module("flowstar_seg_kernels", cache / "flowstar_seg_kernels/flowstar_seg_kernels.so")
    cuda_kernels._tried = True
    tape_kernels._ext = load_module("flowstar_tape_kernels", cache / "flowstar_tape_kernels/flowstar_tape_kernels.so")
    tape_kernels._vext = load_module("flowstar_valid_kernels", cache / "flowstar_valid_kernels/flowstar_valid_kernels.so")
    tape_kernels._tried = tape_kernels._vtried = True
    if not (cuda_kernels.available() and tape_kernels.available() and tape_kernels.valid_available()):
        raise RuntimeError("pre-existing CUDA libraries are unavailable")
    driver = load_module("archcomp26_dp_author_driver", DRIVER)
    if not hasattr(driver.sparse_exec_module, "COMPOSE_PARENT_ASSEMBLY"):
        driver.sparse_exec_module.COMPOSE_PARENT_ASSEMBLY = None  # final metrics only
    return torch, driver, engine, cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=ENGINES, required=True)
    parser.add_argument("--variant", choices=("less", "more"), required=True)
    parser.add_argument("--periods", type=int, choices=(1, 20), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "exception", "backend": args.backend, "periods": args.periods}
    try:
        import yaml
        import numpy as np
        torch, driver, engine, cache = prepare(args.backend)
        config = yaml.safe_load(BASE_CONFIG.read_text())
        assert config["num_vars"] == 7 and config["num_nn_input"] == 4 and config["num_nn_output"] == 2
        assert config["steps"] == 20 and config["step_size"] == 0.05 and config["ode_step_size"] == 0.01
        assert config["model_dir"] == str(MODEL) and MODEL.is_file()
        assert config["constraints_safe"] == ["-th1 - 1.7", "th1 - 2", "-th2 - 1.7", "th2 - 2", "-u1 - 1.7", "u1 - 2", "-u2 - 1.7", "u2 - 2"]
        if args.variant == "more":
            config["model_dir"] = str(MORE_MODEL)
            config["step_size"] = 0.02
            config["ode_step_size"] = 0.005
            config["constraints_safe"] = [item for v in ("th1", "th2", "u1", "u2")
                                           for item in (f"-{v} - 1.5", f"{v} - 1.5")]
            model, box_path, safe_lo, safe_hi, substeps = MORE_MODEL, MORE_BOXES, -1.5, 1.5, 4
        else:
            model, box_path, safe_lo, safe_hi, substeps = MODEL, BOXES, -1.7, 2.0, 5
        assert model.is_file() and box_path.is_file()
        cells = torch.tensor(json.loads(box_path.read_text()), dtype=torch.float64)
        assert cells.shape == (225, 7, 2)
        # The saved decimal partition deliberately rounds its last 1.3 edge
        # one binary64 step outward; do not reject that covering edge.
        assert bool((cells[:, :4, 0] >= 1.0).all() and
                    (cells[:, :4, 1] <= math.nextafter(1.3, math.inf)).all())
        assert bool((cells[:, 4:, :] == 0).all())
        config["steps"] = args.periods
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        driver.make_cells = lambda _config: cells.clone()
        driver.SR_QUEUE = int(config["sr_queue"])

        def official_build(config, device, relax="same-slope", input_layout="native"):
            assert input_layout == "native" and relax == "same-slope"
            from auto_LiRPA import BoundedModule
            raw = driver.build_raw_net(config, experimental=False)
            return BoundedModule(raw, torch.zeros(1, *config["input_shape"][1:], dtype=torch.float64),
                                 device=device, bound_opts=dict(config["bound_opts"]))

        driver.build_crown = official_build
        (output / "START.json").write_text(json.dumps({
            "started_utc": datetime.now(timezone.utc).isoformat(), "backend": args.backend,
            "method": "author order-4 strict plant, shared Xiangru driver, official controller layout, box/same-slope/rpc-float32",
            "variant": args.variant,
            "engine": str(engine), "driver": str(DRIVER), "config": str(config_path),
            "boxes": str(box_path), "controller": str(model), "cuda_cache": str(cache),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "digests": "disabled",
            "qualification": "author-method diagnostic; original RN injection and CROWN remain unqualified",
        }, indent=2) + "\n")
        steps = []
        dtype = np.dtype([("lane", "<u8"), ("step", "<u8"), ("h", "<f8"), ("bounds", "<f8", (4, 4))])
        original_advance = driver.advance_sparse
        with (output / "ranges.bin").open("xb") as ranges:
            def observed(*argv, **kwargs):
                state, accepted = original_advance(*argv, **kwargs)
                eng, settings = argv[2], argv[4]
                tube = driver.hull_ranges_s(state, eng, 4)
                endpoint_time = torch.full((225, 2), float(settings.step), dtype=torch.float64, device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, eng, endpoint_time, 4)
                bounds = torch.cat((tube, endpoint), dim=-1).detach().cpu().numpy()
                accepted_cpu = accepted.detach().cpu().tolist()
                valid = np.asarray(accepted_cpu, dtype=bool)
                if not np.isfinite(bounds[valid]).all() or not (bounds[valid, :, 0] <= bounds[valid, :, 1]).all() or not (bounds[valid, :, 2] <= bounds[valid, :, 3]).all():
                    raise FloatingPointError("accepted interval record is invalid")
                safe = bool((bounds[valid, :, 0] >= safe_lo).all() and (bounds[valid, :, 1] <= safe_hi).all())
                rows = np.empty(225, dtype=dtype)
                rows["lane"] = np.arange(225)
                rows["step"] = len(steps) + 1
                rows["h"] = float(settings.step)
                rows["bounds"] = bounds
                ranges.write(rows.tobytes())
                ranges.flush()
                steps.append({"step": len(steps) + 1, "accepted": sum(accepted_cpu),
                              "safe_tube_for_accepted": safe})
                if len(steps) % 5 == 0 or not all(accepted_cpu):
                    (output / "progress.json").write_text(json.dumps(steps[-1]) + "\n")
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(DRIVER), str(config_path), "--device", "cuda:0", "--engine", "sparse", "--strict",
                    "--crown-domain", "box", "--crown-relax", "same-slope", "--crown-transport", "rpc-float32",
                    "--crown-input-layout", "native", "--print-final-hull", "--metrics-json", str(output / "metrics.json")]
            before_argv = sys.argv
            sys.argv = argv
            try:
                driver_code = driver.main()
            finally:
                sys.argv = before_argv
        completed = len(steps) == args.periods * substeps and all(x["accepted"] == 225 for x in steps)
        result.update(status="completed" if driver_code == 0 and completed else "incomplete",
                      driver_return=driver_code, observed_substeps=len(steps), expected_substeps=args.periods * substeps,
                      accepted_lane_substeps=sum(x["accepted"] for x in steps),
                      full_time_tube_inside_safe_box=completed and all(x["safe_tube_for_accepted"] for x in steps),
                      steps=steps, driver_wall_s=time.perf_counter() - started,
                      end_to_end_floating_point_nn_certificate=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      observed_substeps=len(steps) if "steps" in locals() else 0)
        raise
    finally:
        result["wall_s"] = time.perf_counter() - started
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
