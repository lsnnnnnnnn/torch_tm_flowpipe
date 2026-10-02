#!/usr/bin/env python3
"""Run the paper Unicycle with a constant disturbance only in speed."""

import argparse
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
SOURCE = N / "runs/archcomp_review_20260923/contracts/unicycle.yaml"
MODEL = N / "runs/archcomp26_20261001/unicycle_prep_001/controllerB_2026.onnx"
INITIAL = [[9.5, 9.55], [-4.5, -4.45], [2.1, 2.11], [1.5, 1.51],
           [-0.0001, 0.0001], [0, 0], [0, 0], [0, 0]]
NAMES = ["x1", "x2", "x3", "x4", "w", "t", "u1", "u2"]
SOURCE_RHS = ["x4*cos(x3)", "x4*sin(x3)", "u2+w-20", "u1+w-20", "0", "1", "0", "0"]
PAPER_RHS = ["x4 * cos(x3)", "x4 * sin(x3)", "u2 - 20", "u1 + w - 20", "0", "1", "0", "0"]
TARGET = [[-0.6, 0.6], [-0.2, 0.2], [-0.06, 0.06], [-0.3, 0.3]]
TARGET_CONSTRAINTS = ["x1-0.6", "-x1-0.6", "x2-0.2", "-x2-0.2",
                      "x3-0.06", "-x3-0.06", "x4-0.3", "-x4-0.3"]


def checked_config():
    import yaml

    cfg = yaml.safe_load(SOURCE.read_text())
    expected = {"num_vars": 8, "num_nn_input": 4, "num_nn_output": 2,
                "steps": 50, "step_size": 0.2, "ode_step_size": 0.02,
                "ode_order": 2, "cut_off_threshold": 1e-6,
                "remainder_estimation": [-0.01, 0.01], "sr_queue": 1000,
                "input_shape": [-1, 1, 1, 4], "output_T_shape": [-1, 2, 4],
                "output_c_shape": [-1, 2], "output_scale": 1,
                "output_offset": 0, "bound_opts": {"activation_bound_option": "same-slope"}}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"saved Unicycle setting differs at {key}")
    if ([(item["name"], [float(value) for value in item["interval"]])
         for item in cfg["initial_set"]] !=
            list(zip(NAMES, INITIAL)) or
            ["".join(item.split()) for item in cfg["dynamics_expressions"]] != SOURCE_RHS or
            ["".join(item.split()) for item in cfg["constraints_target"]] != TARGET_CONSTRAINTS or
            cfg.get("constraints_safe") or cfg.get("constraints_unsafe") or cfg.get("split_vars") or
            cfg["model_dir"] != "/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/Benchmark10-Unicycle/controllerB.onnx" or
            not MODEL.is_file()):
        raise ValueError("saved Unicycle source box, dynamics, target or model differs")
    cfg["dynamics_expressions"] = PAPER_RHS[:]
    cfg["model_dir"] = str(MODEL)
    return cfg


def cpu_preflight():
    import onnx
    import onnx2pytorch
    import torch

    cfg = checked_config()
    graph = onnx.load(str(MODEL)).graph
    initializers = {item.name for item in graph.initializer}
    actual_inputs = [item for item in graph.input if item.name not in initializers]
    shape = lambda item: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
    operators = [node.op_type for node in graph.node]
    if (len(actual_inputs) != 1 or shape(actual_inputs[0]) != [1, 1, 1, 4] or
            len(graph.output) != 1 or shape(graph.output[0]) != [1, 2] or
            operators != ["Sub", "Conv", "Relu", "Conv", "Relu", "Flatten"]):
        raise ValueError("fixed 2026 Unicycle controller interface/graph differs")
    net = onnx2pytorch.ConvertModel(onnx.load(str(MODEL))).to(torch.float64).eval()
    center = torch.tensor([[[[9.525, -4.475, 2.105, 1.505]]]], dtype=torch.float64)
    with torch.no_grad():
        output = net(center).reshape(-1)
    if output.numel() != 2 or not all(math.isfinite(float(x)) for x in output) or torch.cuda.is_initialized():
        raise ValueError("CPU controller center preflight failed")
    return cfg, {"input_shape": [1, 1, 1, 4], "output_shape": [1, 2],
                 "operators": operators, "center_raw_outputs": [float(x) for x in output],
                 "cuda_initialized": False}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []
    result = {"status": "failed", "method": args.backend, "mode": args.mode,
              "profile": "unicycle-paper-speed-w-constant-v1"}
    try:
        cfg, model_info = cpu_preflight()
        record = {"schema": "archcomp26-unicycle-paper-speed-w-constant-author-nohash-v1",
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "method": args.backend, "mode": args.mode,
                  "profile": "unicycle-paper-speed-w-constant-v1",
                  "source_config": str(SOURCE), "controller": str(MODEL),
                  "model_preflight": model_info, "state_order": NAMES,
                  "initial_set": INITIAL, "dynamics_expressions": PAPER_RHS,
                  "controller_input_order": NAMES[:4],
                  "controller_output": "raw (f1,f2), each held 0.2 s; plant subtracts 20 once",
                  "w_semantics": "one initial w per trajectory, w'=0, only x4'=u1-20+w",
                  "target": TARGET, "property_rule": "complete T=10 endpoint inclusion is sufficient for reach within 10 s; otherwise UNKNOWN",
                  "period_s": 0.2, "ode_step_s": 0.02, "ode_order": 2,
                  "full_periods": 50, "full_substeps": 500,
                  "qualification": "author GPU plant and CROWN diagnostic; no independent end-to-end floating-point NNCS certificate"}
        (output / "PREFLIGHT.json").write_text(json.dumps(record, indent=2) + "\n")
        if args.mode == "preflight":
            result.update(status="cpu_preflight_complete", model_info=model_info,
                          wall_s=time.perf_counter() - started)
            return 0
        if os.environ.get("CUDA_VISIBLE_DEVICES") != ("3" if args.backend == "p3" else "1"):
            raise RuntimeError("Unicycle method selected wrong physical GPU")
        import torch
        import yaml

        torch, driver, engine, cache = common.prepare(args.backend)
        torch.set_default_dtype(torch.float64)
        total = torch.cuda.get_device_properties(0).total_memory
        cap = min(11 * 2**30, int(total * 0.9))
        torch.cuda.set_per_process_memory_fraction(cap / total)
        cfg["steps"] = 1 if args.mode == "smoke1" else 50
        if args.mode == "smoke1":
            cfg.pop("constraints_target")
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cells = driver.make_cells(cfg)
        if tuple(cells.shape) != (1, 8, 2) or cells[0].tolist() != INITIAL:
            raise ValueError("generated initial box or state order differs")
        driver.make_cells = lambda _cfg: cells.clone()
        driver.SR_QUEUE = 1000

        def build_crown(config, device, relax="same-slope", input_layout="native"):
            if relax != "same-slope" or input_layout != "native":
                raise ValueError("expected original Unicycle CROWN relaxation/layout")
            from auto_LiRPA import BoundedModule
            raw = driver.build_raw_net(config, experimental=False)
            return BoundedModule(raw, torch.zeros(1, 1, 1, 4, dtype=torch.float64),
                                 device=device, bound_opts=dict(config["bound_opts"]))

        driver.build_crown = build_crown
        record.update(generated_config=str(config_path), controller_periods=cfg["steps"],
                      engine=str(engine), shared_driver=str(common.DRIVER), cuda_cache=str(cache),
                      physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      cpu_affinity=sorted(os.sched_getaffinity(0)), gpu_memory_cap_bytes=cap)
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
        original_advance = driver.advance_sparse
        with (output / "ranges.jsonl").open("x") as saved:
            def observed(*call_args, **call_kwargs):
                state, accepted = original_advance(*call_args, **call_kwargs)
                index = len(rows) + 1
                okay = bool(accepted[0].item())
                code = int(state.status[0].item())
                row = {"substep": index, "accepted": okay, "solver_status_code": code,
                       "solver_status": {0: "ACTIVE", 1: "FAILED_CONTRACTION",
                                         2: "DONE", 3: "FAILED_DIV"}.get(code, "UNRECOGNIZED"),
                       "t_interval": [(index - 1) * 0.02, index * 0.02]}
                if okay:
                    eng, settings = call_args[2], call_args[4]
                    tube = driver.hull_ranges_s(state, eng, 8)[0].detach().cpu().tolist()
                    endpoint_time = torch.full((1, 2), float(settings.step), dtype=torch.float64,
                                               device=state.pre.device)
                    endpoint = driver.rows_range_over_time_sparse(state, eng, endpoint_time, 8)[0].detach().cpu().tolist()
                    if (len(tube) != 8 or len(endpoint) != 8 or
                            any(len(v) != 2 or not all(math.isfinite(x) for x in v) or v[0] > v[1]
                                for v in tube + endpoint)):
                        raise FloatingPointError("nonfinite or inverted accepted Unicycle range")
                    row.update(tube=tube, endpoint=endpoint)
                saved.write(json.dumps(row, allow_nan=False) + "\n")
                saved.flush()
                rows.append(row)
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(common.DRIVER), str(config_path), "--device", "cuda:0",
                    "--engine", "sparse", "--strict"]
            if args.backend == "p3":
                argv.extend(("--order", "3"))
            argv += ["--crown-domain", "box",
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

        planned = cfg["steps"] * 10
        complete = len(rows) == planned and all(row["accepted"] for row in rows)
        terminal = rows[-1]["endpoint"][:4] if complete else None
        endpoint_in_target = bool(args.mode == "full" and complete and
                                  all(TARGET[j][0] <= v[0] <= v[1] <= TARGET[j][1]
                                      for j, v in enumerate(terminal)))
        result.update(status=("completed_short_prefix" if args.mode == "smoke1" else "completed")
                      if complete else "early_stopped", driver_return=driver_return,
                      observed_substeps=len(rows), expected_substeps=planned,
                      accepted_substeps=sum(row["accepted"] for row in rows),
                      first_rejection_internal_status=next(
                          (row["solver_status"] for row in rows if not row["accepted"]), None),
                      last_accepted_t_s=sum(row["accepted"] for row in rows) * 0.02,
                      terminal_physical_endpoint=terminal,
                      terminal_endpoint_in_target=endpoint_in_target,
                      property_verdict=("ENDPOINT_SUFFICIENT_FOR_REACH" if endpoint_in_target else
                                        "NOT_APPLICABLE_SHORT_PREFIX" if args.mode == "smoke1" else
                                        "UNKNOWN"),
                      wall_s=time.perf_counter() - started,
                      end_to_end_floating_point_nn_certificate=False)
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), observed_substeps=len(rows),
                      wall_s=time.perf_counter() - started)
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"].startswith("completed") else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("huan", "xiangru"), required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke1", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
