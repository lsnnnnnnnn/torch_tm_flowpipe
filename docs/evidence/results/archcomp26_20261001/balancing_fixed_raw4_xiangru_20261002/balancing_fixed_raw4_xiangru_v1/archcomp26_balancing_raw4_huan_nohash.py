#!/usr/bin/env python3
"""Isolated Huan run for the fixed-repository, raw-four-state CartPole profile."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import archcomp26_tora_remain_author_nohash as common


N = common.N
SOURCE = N / "runs/archcomp_review_20260923/contracts/cartpole_official_f32.yaml"
OFFICIAL = N / "engine_accelerated/benchmarks/CartPole/official"
MODEL = Path("/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/CartPole/model.onnx")
INITIAL = [[-0.1, 0.1], [-0.05, 0.05], [-0.1, 0.1], [-0.05, 0.05]]
STATE_NAMES = ["x1", "x2", "x3", "x4", "t", "u1"]
RHS = ["x2", "2*u1", "x4",
       "(0.08*0.41*(9.8*sin(x3)-2*u1*cos(x3))-0.0021*x4)/0.0105", "1", "0"]
PROPERTY = ["-x1-0.001", "x1-0.001", "-x3-0.001", "x3-0.001",
            "-x4-0.001", "x4-0.001"]


def checked_config():
    import yaml

    cfg = yaml.safe_load(SOURCE.read_text())
    expected = {"num_vars": 6, "num_nn_input": 4, "num_nn_output": 1,
                "steps": 50, "step_size": 0.02, "ode_step_size": 0.005,
                "ode_order": 6, "cut_off_threshold": 1e-6,
                "remainder_estimation": [-0.1, 0.1], "sr_queue": 1000,
                "input_shape": [-1, 4], "output_T_shape": [-1, 1, 4],
                "output_c_shape": [-1, 1], "output_scale": 1,
                "output_offset": 0, "bound_opts": {"activation_bound_option": "same-slope",
                                                  "conv_mode": "matrix"}}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"saved CartPole setting differs at {key}: {cfg.get(key)!r}")
    if ["".join(s.split()) for s in cfg["dynamics_expressions"]] != RHS:
        raise ValueError("saved CartPole equations differ from fixed MATLAB equations")
    if ([(v["name"], v["interval"]) for v in cfg["initial_set"]] !=
            list(zip(STATE_NAMES, [[-0.0375, -0.03125], [-0.015625, -0.0125],
                                   [-0.00625, 0], [-0.007375, -0.00625], [0, 0], [0, 0]]))):
        raise ValueError("saved historical CartPole source state order differs")
    if ["".join(s.split()) for s in cfg["constraints_target"]] != PROPERTY:
        raise ValueError("saved historical CartPole target expressions differ")
    if cfg.get("constraints_safe") or cfg.get("constraints_unsafe") or cfg.get("split_vars"):
        raise ValueError("saved historical CartPole source carries extra constraints or splits")
    if cfg["model_dir"] != str(MODEL) or not MODEL.is_file():
        raise ValueError("previously compared fixed four-input controller is missing")
    specs = (OFFICIAL / "specifications.txt").read_text()
    matlab = (OFFICIAL / "dynamics.m").read_text()
    for fragment in ("x1 = [-0.1, 0.1]", "x2 = [-0.05, 0.05]",
                     "x3 = [-0.1, 0.1]", "x4 = [-0.05, 0.05]",
                     "t = 10 seconds", "control period 0.02 s", "For t > 8.0 s"):
        if fragment not in specs:
            raise ValueError(f"fixed official specification differs at {fragment}")
    for fragment in ("x(1), x(2), x(3), x(4)", "dx(2,1) = 2*f;",
                     "9.8*sin(x(3))-2*f*cos(x(3))", "0.0105"):
        if fragment not in matlab:
            raise ValueError(f"fixed official MATLAB differs at {fragment}")
    cfg["steps"] = 500
    for entry, bounds in zip(cfg["initial_set"][:4], INITIAL):
        entry["interval"] = bounds
    cfg.pop("constraints_target")
    cfg["constraints_safe"] = [s.replace(" ", "") for s in PROPERTY]
    cfg["constraints_safe_from"] = 8.0
    cfg["constraints_safe_until"] = 10.0
    cfg["split_vars"] = []
    return cfg


def cpu_preflight():
    import onnx
    import onnx2pytorch
    import torch

    cfg = checked_config()
    graph = onnx.load(str(MODEL)).graph
    shape = lambda item: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
    inputs = [item for item in graph.input if item.name not in {w.name for w in graph.initializer}]
    if len(inputs) != 1 or shape(inputs[0]) != [1, 4] or shape(graph.output[0]) != [1, 1]:
        raise ValueError("fixed controller is not one raw-four-state input to one force output")
    operators = [node.op_type for node in graph.node]
    if operators != ["Gemm", "Tanh", "Gemm", "Tanh", "Gemm", "Tanh"]:
        raise ValueError(f"fixed controller operators differ: {operators}")
    net = onnx2pytorch.ConvertModel(onnx.load(str(MODEL))).to(torch.float64).eval()
    with torch.no_grad():
        output = net(torch.zeros(1, 4, dtype=torch.float64)).reshape(-1)
    if output.numel() != 1 or not math.isfinite(float(output[0])) or torch.cuda.is_initialized():
        raise ValueError("CPU zero-state controller preflight failed")
    return cfg, {"input_shape": [1, 4], "output_shape": [1, 1],
                 "operators": operators, "zero_state_force": float(output[0]),
                 "cuda_initialized": False}


def run(args):
    backend = getattr(args, "backend", "huan")
    if backend not in {"huan", "xiangru", "p3"}:
        raise ValueError(f"unsupported Balancing backend: {backend}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"status": "failed", "profile": "balancing-fixed-repo-raw4",
              "method": backend, "mode": args.mode}
    observations, checks = [], []
    try:
        cfg, model_info = cpu_preflight()
        if backend == "p3":
            cfg["ode_order"] = 3
        record = {"schema": f"archcomp26-balancing-fixed-raw4-{backend}-nohash-v1",
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "source_config": str(SOURCE), "official_text": str(OFFICIAL),
                  "controller": str(MODEL), "cpu_model_preflight": model_info,
                  "profile": "balancing-fixed-repo-raw4", "mode": args.mode,
                  "method": backend,
                  "initial_physical_box": INITIAL, "physical_box_count": 1,
                  "controller_input_order": STATE_NAMES[:4],
                  "controller_output": "raw force f held for 0.02 s",
                  "property": "x1,x3,x4 in [-0.001,0.001] for 8<t<=10; checker includes t=8 by continuity of the closed plant and target box",
                  "paper_feature5_contract": False,
                  "period_s": 0.02, "ode_step_s": 0.005,
                  "ode_order": cfg["ode_order"], "full_period_count": 500,
                  "qualification": ("working P3 plant, strict endpoint/injection, CROWN diagnostic; no independent end-to-end floating-point NN certificate"
                                    if backend == "p3" else
                                    "author GPU plant and CROWN diagnostic; no independent end-to-end floating-point NN certificate")}
        (output / "PREFLIGHT.json").write_text(json.dumps(record, indent=2) + "\n")
        if args.mode == "preflight":
            result.update(status="cpu_preflight_complete", model_info=model_info,
                          wall_s=time.perf_counter() - started)
            return 0
        expected_gpu = {"huan": "2", "xiangru": "1", "p3": "3"}[backend]
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
            raise RuntimeError(f"Balancing {backend} attempt reserved for physical GPU {expected_gpu}")
        import torch
        import yaml

        torch, driver, engine, cache = common.prepare(backend)
        total = torch.cuda.get_device_properties(0).total_memory
        cap = min(11 * 2**30, int(total * 0.9))
        torch.cuda.set_per_process_memory_fraction(cap / total)
        cfg["steps"] = 1 if args.mode == "smoke1" else 500
        if args.mode == "smoke1":
            cfg.pop("constraints_safe")
            cfg.pop("constraints_safe_from")
            cfg.pop("constraints_safe_until")
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cells = driver.make_cells(cfg)
        if tuple(cells.shape) != (1, 6, 2) or cells[0, :4].tolist() != INITIAL:
            raise ValueError("generated full CartPole initial box differs")
        driver.make_cells = lambda _cfg: cells.clone()
        driver.SR_QUEUE = 1000

        def build_crown(config, device, relax="same-slope", input_layout="native"):
            if relax != "same-slope" or input_layout != "native":
                raise ValueError("unexpected CartPole CROWN setting")
            from auto_LiRPA import BoundedModule
            raw = driver.build_raw_net(config, experimental=False)
            return BoundedModule(raw, torch.zeros(1, 4, dtype=torch.float64),
                                 device=device, bound_opts=dict(config["bound_opts"]))

        driver.build_crown = build_crown
        record.update(generated_config=str(config_path), controller_periods=cfg["steps"],
                      smoke_property_check=("disabled: 8<t<=10 is outside the 0.02 s prefix"
                                            if args.mode == "smoke1" else None),
                      gpu_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      cpu_affinity=sorted(os.sched_getaffinity(0)),
                      gpu_memory_cap_bytes=cap, engine=str(engine),
                      driver=str(common.DRIVER), cuda_cache=str(cache))
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
        original_advance = driver.advance_sparse
        original_ranges = driver.SpecGroup.ranges
        trace_context = ((output / "refinement_trace.jsonl").open("x")
                         if backend == "p3" else nullcontext(None))
        with (output / "ranges.jsonl").open("x") as rangefile, (output / "property_checks.jsonl").open("x") as checkfile, trace_context as tracefile:
            def observed(*call_args, **call_kwargs):
                if tracefile is not None:
                    step_number = len(observations) + 1

                    def trace(event):
                        tracefile.write(json.dumps({"substep": step_number, **event}, default=repr) + "\n")
                        tracefile.flush()

                    call_args = list(call_args)
                    call_args[4] = replace(call_args[4], refinement_callback=trace)
                state, accepted = original_advance(*call_args, **call_kwargs)
                step = len(observations) + 1
                okay = bool(accepted[0].item())
                status_code = int(state.status[0].item())
                row = {"substep": step, "accepted": okay,
                       "solver_status_code": status_code,
                       "solver_status": {0: "ACTIVE", 1: "FAILED_CONTRACTION",
                                         2: "DONE", 3: "FAILED_DIV"}.get(status_code, "UNRECOGNIZED"),
                       "t_interval": [(step - 1) * 0.005, step * 0.005]}
                if okay:
                    eng, settings = call_args[2], call_args[4]
                    tube = driver.hull_ranges_s(state, eng, 4)[0].detach().cpu().tolist()
                    t_end = torch.full((1, 2), float(settings.step), dtype=torch.float64,
                                       device=state.pre.device)
                    endpoint = driver.rows_range_over_time_sparse(state, eng, t_end, 4)[0].detach().cpu().tolist()
                    row.update(tube=tube, endpoint=endpoint)
                rangefile.write(json.dumps(row, allow_nan=False) + "\n")
                rangefile.flush()
                observations.append(row)
                return state, accepted

            def checked_ranges(group, boxes, active, **kwargs):
                q = original_ranges(group, boxes, active, **kwargs)
                if group.texts and group.texts == cfg.get("constraints_safe"):
                    row = {"check_index": len(checks) + 1,
                           "physical_substep": len(observations),
                           "q_intervals": q[0].detach().cpu().tolist(),
                           "all_six_upper_nonpositive": bool((q[0, :, 1] <= 0).all().item()),
                           "any_lower_positive": bool((q[0, :, 0] > 0).any().item())}
                    checkfile.write(json.dumps(row, allow_nan=False) + "\n")
                    checkfile.flush()
                    checks.append(row)
                return q

            driver.advance_sparse = observed
            driver.SpecGroup.ranges = checked_ranges
            argv = [str(common.DRIVER), str(config_path), "--device", "cuda:0",
                    "--engine", "sparse", "--strict", "--crown-domain", "box",
                    "--crown-relax", "same-slope", "--crown-transport", "rpc-float32",
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
        expected = cfg["steps"] * 4
        complete = driver_return == 0 and len(observations) == expected and all(
            row["accepted"] for row in observations)
        expected_checks = 400 if args.mode == "full" else 0
        property_complete = args.mode == "full" and len(checks) == expected_checks
        property_all_inside = property_complete and all(row["all_six_upper_nonpositive"] for row in checks)
        result.update(status=("completed_short_prefix" if args.mode == "smoke1" else "completed")
                      if complete else "early_stopped", driver_return=driver_return,
                      observed_substeps=len(observations), expected_substeps=expected,
                      accepted_substeps=sum(row["accepted"] for row in observations),
                      first_rejection_internal_status=next(
                          (row["solver_status"] for row in observations if not row["accepted"]), None),
                      property_checks=len(checks), expected_property_checks=expected_checks,
                      all_saved_property_boxes_inside=property_all_inside,
                      property_verdict=("NOT_APPLICABLE_SHORT_PREFIX" if args.mode == "smoke1"
                                        else "VERIFIED_BY_SAVED_BOX_CHECKS" if property_all_inside
                                        else "UNKNOWN_OR_INCOMPLETE"),
                      final_physical_endpoint=(observations[-1].get("endpoint") if observations else None),
                      wall_s=time.perf_counter() - started,
                      end_to_end_floating_point_nn_certificate=False)
        if args.mode == "full" and complete and not property_complete:
            result["status"] = "incomplete_property_checks"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), observed_substeps=len(observations),
                      property_checks=len(checks), wall_s=time.perf_counter() - started)
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"].startswith("completed") else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke1", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("huan", "xiangru", "p3"), default="huan")
    raise SystemExit(run(parser.parse_args()))
