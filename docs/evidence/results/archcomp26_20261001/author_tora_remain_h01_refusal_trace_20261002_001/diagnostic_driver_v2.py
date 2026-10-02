#!/usr/bin/env python3
"""Isolated 2026 TORA-remain Huan/Xiangru run and CPU-only contract preflight."""

import argparse
from contextlib import redirect_stdout
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
SOURCE_CONFIG = N / "runs/archcomp_review_20260923/contracts/tora_homogeneous.yaml"
BOXES = N / "runs/archcomp_review_20260923/contracts/tora_homogeneous_boxes.json"
MODEL = N / "runs/archcomp26_20261001/tora_remain_prep_001/official_controllerTora_2026.onnx"
OLD_MODEL = "/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/Benchmark9-Tora/controllerTora.onnx"
ENGINES = {"huan": Path("/srv/local/shengenli/flowstar-gpu"), "xiangru": X}


def prohibited(*_args, **_kwargs):
    raise RuntimeError("content digest or CUDA extension build prohibited")


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


def checked_contract(source_config, box_path, controller, *, check_model=True):
    import yaml

    config = yaml.safe_load(source_config.read_text())
    expected = {
        "num_vars": 6, "num_nn_input": 4, "num_nn_output": 1,
        "steps": 20, "step_size": 1.0, "ode_step_size": 0.1, "ode_order": 3,
        "output_scale": 1, "output_offset": 0, "sr_queue": 1000,
        "input_shape": [-1, 1, 1, 4],
        "output_T_shape": [-1, 1, 4], "output_c_shape": [-1, 1],
        "split_vars": ["x1", "x2"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"TORA contract differs at {key}: {config.get(key)!r}")
    initial = [(item["name"], item["interval"], item.get("splits", 0))
               for item in config["initial_set"]]
    if initial != [
        ("x1", [0.6, 0.7], 4), ("x2", [-0.7, -0.6], 3),
        ("x3", [-0.4, -0.3], 0), ("x4", [0.5, 0.6], 0),
        ("t", [0.0, 0.0], 0), ("u1", [0.0, 0.0], 0),
    ]:
        raise ValueError("TORA initial set, state order, or 4x3 split differs")
    expr = ["".join(value.split()) for value in config["dynamics_expressions"]]
    if expr != ["x2", "-x1+0.1*sin(x3)", "x4", "u1-10", "1", "0"]:
        raise ValueError("TORA RHS differs; the raw NN output must be offset once in dx4")
    safe = [item.replace(" ", "") for item in config["constraints_safe"]]
    if safe != [f"{sign}{name}-2" for name in ("x1", "x2", "x3", "x4")
                for sign in ("-", "")]:
        raise ValueError("TORA all-time [-2,2]^4 safety constraints differ")
    if (config.get("constraints_safe_from") is not None or
            config.get("constraints_safe_until") is not None or
            config.get("constraints_unsafe") or config.get("constraints_target")):
        raise ValueError("TORA remain must check only the full-time safe set")
    if config["model_dir"] != OLD_MODEL or not controller.is_file():
        raise ValueError("saved source controller path or selected 2026 ONNX is missing")
    if config.get("bound_opts") != {"activation_bound_option": "same-slope", "conv_mode": "matrix"}:
        raise ValueError("TORA CROWN relaxation/layout differs")

    boxes = json.loads(box_path.read_text())
    if len(boxes) != 12:
        raise ValueError("TORA ledger must contain 12 boxes")
    pairs = set()
    for box in boxes:
        if len(box) != 6 or any(len(bounds) != 2 or bounds[0] > bounds[1] for bounds in box):
            raise ValueError("malformed TORA box")
        if box[2:] != [[-0.4, -0.3], [0.5, 0.6], [0, 0], [0, 0]]:
            raise ValueError("TORA nonsplit dimensions differ")
        pairs.add((tuple(box[0]), tuple(box[1])))
    if len(pairs) != 12:
        raise ValueError("duplicate TORA boxes")
    for coordinate, lo, hi, pieces in ((0, 0.6, 0.7, 4), (1, -0.7, -0.6, 3)):
        intervals = sorted({tuple(box[coordinate]) for box in boxes})
        if (len(intervals) != pieces or abs(intervals[0][0] - lo) > 1e-12 or
                abs(intervals[-1][1] - hi) > 1e-12 or
                any(right[0] > left[1] for left, right in zip(intervals, intervals[1:]))):
            raise ValueError(f"TORA split coordinate {coordinate} does not cover its initial range")
    if len({tuple(box[0]) for box in boxes}) * len({tuple(box[1]) for box in boxes}) != 12:
        raise ValueError("TORA ledger is not the full 4x3 Cartesian grid")

    model_info = None
    if check_model:
        import onnx
        import onnx2pytorch
        import torch

        model = onnx.load(str(controller))
        initializers = {item.name for item in model.graph.initializer}
        dynamic_inputs = [item for item in model.graph.input if item.name not in initializers]
        if len(dynamic_inputs) != 1:
            raise ValueError("expected one four-state controller input")
        input_shape = [dim.dim_value for dim in dynamic_inputs[0].type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
        if input_shape != [1, 1, 1, 4] or output_shape != [1, 1]:
            raise ValueError(f"controller I/O shape differs: {input_shape} -> {output_shape}")
        tail = [node.op_type for node in model.graph.node[-3:]]
        if tail != ["Conv", "Relu", "Flatten"]:
            raise ValueError(f"controller output graph differs: {tail}")
        net = onnx2pytorch.ConvertModel(model, experimental=False).to(torch.float64).eval()
        center = torch.tensor([[[[0.65, -0.65, -0.35, 0.55]]]], dtype=torch.float64)
        with torch.no_grad():
            raw = net(center).reshape(-1)
        if raw.numel() != 1 or not math.isfinite(float(raw[0])) or torch.cuda.is_initialized():
            raise ValueError("CPU controller center preflight failed")
        model_info = {"dynamic_input": dynamic_inputs[0].name, "input_shape": input_shape,
                      "output_shape": output_shape, "output_tail_ops": tail,
                      "center_raw_f_value": float(raw[0]),
                      "center_plant_u_value": float(raw[0]) - 10.0}
    return config, boxes, model_info


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
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu not in {"0", "1", "2", "3"}:
        raise RuntimeError("select one physical GPU through CUDA_VISIBLE_DEVICES=0..3")
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
    for owner, attr, tried, name, exports in (
        (ck, "_ext", "_tried", "flowstar_seg_kernels", ("seg_mul_iv", "seg_mul_pt", "seg_dot_pt_iv")),
        (tk, "_ext", "_tried", "flowstar_tape_kernels", ("refine_tape", "transcendental_probe")),
        (tk, "_vext", "_vtried", "flowstar_valid_kernels", ("valid_tape", "point_tape")),
    ):
        path = cache / name / (name + ".so")
        module = load_module(name, path)
        if not all(callable(getattr(module, export, None)) for export in exports):
            raise RuntimeError(f"saved CUDA library lacks expected exports: {path}")
        setattr(owner, attr, module)
        setattr(owner, tried, True)
    if not (ck.available() and tk.available() and tk.valid_available()):
        raise RuntimeError("preloaded author CUDA libraries unavailable")
    driver = load_module("archcomp26_tora_author_driver", DRIVER)
    if not hasattr(driver.sparse_exec_module, "COMPOSE_PARENT_ASSEMBLY"):
        driver.sparse_exec_module.COMPOSE_PARENT_ASSEMBLY = None
    return torch, driver, engine, cache


class Tee:
    def __init__(self, stream):
        self.stream = stream
        self.lines = []
        self.pending = ""

    def write(self, value):
        self.stream.write(value)
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip() in {"Unsafe.", "Unknown.", "Flow* terminated.", "VERIFIED", "FALSIFIED", "UNKNOWN"}:
                self.lines.append(line.strip())
        return len(value)

    def flush(self):
        self.stream.flush()


class FirstNumericalRejection(Exception):
    """Stop an isolated diagnostic after preserving its first rejected step."""


class ProbeHorizonReached(Exception):
    """Never continue this trace beyond the prior first-refusal step."""


def run(args):
    if args.backend != "huan" or args.mode != "full" or args.ode_step_size != 0.1 or not args.stop_at_first_rejection:
        raise ValueError("this isolated trace requires Huan, T=20 contract, h=0.1, first-rejection stop")
    if args.ode_step_size not in (0.1, 0.05):
        raise ValueError("supported TORA ODE step sizes are 0.1 and 0.05 s")
    if args.ode_step_size != 0.1 and (args.mode != "full" or not args.stop_at_first_rejection):
        raise ValueError("h=0.05 is an isolated full-horizon, first-rejection diagnostic")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "exception", "backend": args.backend, "mode": args.mode}
    observations = []
    try:
        import numpy as np
        import yaml
        torch, driver, engine, cache = prepare(args.backend)
        config, box_rows, _ = checked_contract(SOURCE_CONFIG, BOXES, MODEL, check_model=False)
        config["model_dir"] = str(MODEL)
        config["steps"] = 1 if args.mode == "smoke1" else 20
        config["ode_step_size"] = args.ode_step_size
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        cells = torch.tensor(box_rows, dtype=torch.float64)
        driver.make_cells = lambda _config: cells.clone()
        driver.SR_QUEUE = int(config["sr_queue"])

        def official_build(config, device, relax="same-slope", input_layout="native"):
            if relax != "same-slope" or input_layout != "native":
                raise ValueError("TORA uses native-layout same-slope CROWN")
            from auto_LiRPA import BoundedModule
            raw = driver.build_raw_net(config, experimental=False)
            return BoundedModule(raw, torch.zeros(1, 1, 1, 4, dtype=torch.float64),
                                 device=device, bound_opts=dict(config["bound_opts"]))

        driver.build_crown = official_build
        record = {
            "schema": "archcomp26-tora-remain-author-nohash-v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend, "mode": args.mode,
            "source_config": str(SOURCE_CONFIG), "generated_config": str(config_path),
            "boxes": str(BOXES), "controller": str(MODEL),
            "engine": str(engine), "shared_driver": str(DRIVER), "cuda_cache": str(cache),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "controller_boundary": "raw ONNX f(x) injected into u1; plant dx4=u1-10 applies offset once",
            "method": "author sparse engine, Taylor order 3, strict plant, box same-slope CROWN, native input layout, rpc-float32",
            "numerical_profile": {"ode_step_size": args.ode_step_size,
                                  "stop_at_first_rejection": args.stop_at_first_rejection},
            "property": "all four physical states in [-2,2] over every substep of T=20; smoke covers only T=1",
            "qualification": "shared controller driver and unqualified floating NN injection; not an independent end-to-end proof",
        }
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
        dtype = np.dtype([("lane", "<u8"), ("step", "<u8"), ("h", "<f8"),
                          ("bounds", "<f8", (4, 4))])
        tube_union = np.array([[np.inf, -np.inf]] * 4, dtype=np.float64)
        final_endpoint = None
        stopped_on_first_rejection = False
        stopped_at_probe_horizon = False
        original_advance = driver.advance_sparse
        with (output / "ranges.bin").open("xb") as ranges, (output / "observations.jsonl").open("x") as log, (output / "refinement_step190.jsonl").open("x") as trace:
            from flowstar_gpu import glue as glue_module

            def observed(*call_args, **call_kwargs):
                nonlocal final_endpoint
                step_number = len(observations) + 1
                if step_number > 190:
                    raise ProbeHorizonReached()
                original_glue_run = None
                if step_number == 190:
                    original_glue_run = glue_module.GlueCache.run

                    def observed_glue(self, key, fn, *glue_args):
                        value = original_glue_run(self, key, fn, *glue_args)
                        if key[0] == "validpost":
                            total, ok_dims, int_diff = value
                            guessed = glue_args[4]
                            proposal = total.detach().cpu().tolist()
                            accepted_dims = ok_dims.detach().cpu().tolist()
                            guess = guessed.detach().cpu().tolist()
                            difference = int_diff.detach().cpu().tolist()
                            for lane in range(len(proposal)):
                                row = {"substep": step_number, "lane": lane,
                                       "input_remainder": guess[lane],
                                       "proposal_interval": proposal[lane],
                                       "difference_bound": difference[lane],
                                       "subset_by_component": accepted_dims[lane],
                                       "lower_margin": [proposal[lane][j][0] - guess[lane][j][0] for j in range(len(guess[lane]))],
                                       "upper_margin": [guess[lane][j][1] - proposal[lane][j][1] for j in range(len(guess[lane]))]}
                                trace.write(json.dumps(row) + "\n")
                            trace.flush()
                        return value

                    glue_module.GlueCache.run = observed_glue
                try:
                    state, accepted = original_advance(*call_args, **call_kwargs)
                finally:
                    if original_glue_run is not None:
                        glue_module.GlueCache.run = original_glue_run
                engine_obj, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, engine_obj, 4)
                endpoint_time = torch.full((12, 2), float(settings.step), dtype=torch.float64,
                                           device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, engine_obj, endpoint_time, 4)
                bounds = torch.cat((tube, endpoint), dim=-1).detach().cpu().numpy()
                valid = np.asarray(accepted.detach().cpu().tolist(), dtype=bool)
                if (not np.isfinite(bounds[valid]).all() or
                        not (bounds[valid, :, 0] <= bounds[valid, :, 1]).all() or
                        not (bounds[valid, :, 2] <= bounds[valid, :, 3]).all()):
                    raise FloatingPointError("accepted TORA interval record invalid")
                if valid.any():
                    tube_union[:, 0] = np.minimum(tube_union[:, 0], bounds[valid, :, 0].min(axis=0))
                    tube_union[:, 1] = np.maximum(tube_union[:, 1], bounds[valid, :, 1].max(axis=0))
                    final_endpoint = [
                        [float(bounds[valid, i, 2].min()), float(bounds[valid, i, 3].max())]
                        for i in range(4)
                    ]
                safe = bool(valid.any() and (bounds[valid, :, 0] >= -2).all()
                            and (bounds[valid, :, 1] <= 2).all())
                rows = np.empty(12, dtype=dtype)
                rows["lane"] = np.arange(12)
                rows["step"] = step_number
                rows["h"] = float(settings.step)
                rows["bounds"] = bounds
                ranges.write(rows.tobytes())
                ranges.flush()
                row = {"substep": step_number, "accepted_count": int(valid.sum()),
                       "rejected_lanes": np.flatnonzero(~valid).tolist(),
                       "tube_inside_safe_for_accepted": safe}
                if args.stop_at_first_rejection:
                    row["status_codes"] = state.status.detach().cpu().tolist()
                log.write(json.dumps(row) + "\n")
                log.flush()
                observations.append(row)
                if args.stop_at_first_rejection and not valid.all():
                    raise FirstNumericalRejection()
                if step_number == 190:
                    raise ProbeHorizonReached()
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(DRIVER), str(config_path), "--device", "cuda:0", "--engine", "sparse",
                    "--strict", "--crown-domain", "box", "--crown-relax", "same-slope",
                    "--crown-transport", "rpc-float32", "--crown-input-layout", "native",
                    "--nn-mode", "crown", "--print-final-hull", "--metrics-json", str(output / "metrics.json")]
            record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            tee = Tee(sys.stdout)
            try:
                with redirect_stdout(tee):
                    try:
                        driver_code = driver.main()
                    except FirstNumericalRejection:
                        driver_code = None
                        stopped_on_first_rejection = True
                    except ProbeHorizonReached:
                        driver_code = None
                        stopped_at_probe_horizon = True
            finally:
                sys.argv = previous

        expected = config["steps"] * round(config["step_size"] / config["ode_step_size"])
        complete = (driver_code == 0 and len(observations) == expected
                    and all(row["accepted_count"] == 12 for row in observations))
        if stopped_on_first_rejection:
            checker = "NOT_RETURNED_AFTER_PROBE_STOP"
        elif "Unsafe." in tee.lines or "FALSIFIED" in tee.lines:
            checker = "UNSAFE_OR_FALSIFIED"
        elif any(line in tee.lines for line in ("Unknown.", "UNKNOWN", "Flow* terminated.")):
            checker = "UNKNOWN"
        elif complete and all(row["tube_inside_safe_for_accepted"] for row in observations):
            checker = "VERIFIED_BY_AUTHOR_CHECKER_SILENCE"
        else:
            checker = "UNRESOLVED"
        result.update(
            status=("stopped_first_numerical_rejection" if stopped_on_first_rejection else
                    "stopped_probe_horizon" if stopped_at_probe_horizon else
                    (("completed_short_prefix" if args.mode == "smoke1" else "completed")
                     if complete else "incomplete")),
            probe_max_observed_substeps=190,
            driver_return=driver_code, expected_substeps=expected,
            first_rejection=(observations[-1] if stopped_on_first_rejection else None),
            observed_substeps=len(observations),
            accepted_lane_substeps=sum(row["accepted_count"] for row in observations),
            all_lanes_accepted=complete,
            saved_tube_inside_safe_box=complete and all(
                row["tube_inside_safe_for_accepted"] for row in observations),
            author_checker_lines=tee.lines, author_checker_interpretation=checker,
            observed_prefix_tube_union=(tube_union.tolist() if final_endpoint else None),
            full_time_tube_union=(tube_union.tolist() if complete and args.mode == "full" else None),
            last_observed_endpoint_union=final_endpoint,
            terminal_endpoint_union=(final_endpoint if complete and args.mode == "full" else None),
            terminal_endpoint_union_width=([hi - lo for lo, hi in final_endpoint]
                                           if complete and args.mode == "full" else None),
            driver_elapsed_s=(json.loads((output / "metrics.json").read_text()).get("elapsed_s")
                              if (output / "metrics.json").is_file() else None),
            wall_s=time.perf_counter() - started,
            end_to_end_floating_point_nn_certificate=False,
        )
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                      observed_substeps=len(observations), wall_s=time.perf_counter() - started)
        raise
    finally:
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"].startswith("completed") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="CPU-only source/ledger/ONNX check")
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--boxes", type=Path, default=BOXES)
    parser.add_argument("--controller", type=Path, default=MODEL)
    parser.add_argument("--backend", choices=ENGINES)
    parser.add_argument("--mode", choices=("smoke1", "full"))
    parser.add_argument("--ode-step-size", type=float, default=0.1)
    parser.add_argument("--stop-at-first-rejection", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    guard_digests_and_builds()
    if args.preflight:
        config, boxes, model_info = checked_contract(args.source_config, args.boxes, args.controller)
        report = {"schema": "archcomp26-tora-remain-cpu-preflight-nohash-v1",
                  "checked_utc": datetime.now(timezone.utc).isoformat(),
                  "source_config": str(args.source_config.resolve()),
                  "boxes": str(args.boxes.resolve()),
                  "controller": str(args.controller.resolve()),
                  "state_order": [item["name"] for item in config["initial_set"]],
                  "box_count": len(boxes), "model": model_info,
                  "controller_boundary": "raw ONNX f(x) -> u1; plant dx4=u1-10 exactly once",
                  "horizon_s": 20, "control_period_s": 1, "ode_step_s": 0.1,
                  "safe_region": "all four physical states in [-2,2] for all t in [0,20]",
                  "gpu_initialized": False, "content_digest_computed": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as out:
            json.dump(report, out, indent=2)
            out.write("\n")
        print(f"CPU-only TORA contract preflight passed: {args.output}")
        return 0
    if args.backend is None or args.mode is None:
        parser.error("GPU run requires --backend and --mode")
    if (args.source_config != SOURCE_CONFIG or args.boxes != BOXES or
            args.controller != MODEL):
        parser.error("GPU run uses fixed remote source, ledger, and 2026 ONNX paths")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
