#!/usr/bin/env python3
"""Fresh one-period NAV diagnostic using the author's state order."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import archcomp26_tora_remain_author_nohash as common


N = common.N
CONTRACTS = N / "runs/archcomp_review_20260923/contracts"
PREP = N / "runs/archcomp26_20261001/nav_prep_001"
OPTIONS = {
    "nav-standard": ("nav_standard", "nn-nav-point", 640, 40, 16),
    "nav-robust": ("nav_robust", "nn-nav-set", 25, 5, 5),
}
EXPECTED_DYNAMICS = ["x3*cos(x4)", "x3*sin(x4)", "u1", "u2", "1", "0", "0"]
EXPECTED_UNSAFE = ["-x1+1", "x1-2", "-x2+1", "x2-2"]
EXPECTED_TARGET = ["-x1-0.5", "x1-0.5", "-x2-0.5", "x2-0.5"]


class FirstRejected(RuntimeError):
    pass


prepare_runtime = common.prepare


def preflight(instance):
    import yaml

    stem, model_name, box_count, x_splits, y_splits = OPTIONS[instance]
    source = CONTRACTS / (stem + ".yaml")
    ledger = CONTRACTS / (stem + "_boxes.json")
    model = PREP / ("official_" + model_name + ".onnx")
    cfg = yaml.safe_load(source.read_text())
    boxes = json.loads(ledger.read_text())
    required = {"num_vars": 7, "num_nn_input": 4, "num_nn_output": 2,
                "steps": 30, "step_size": 0.2, "ode_step_size": 0.01,
                "ode_order": 4, "input_shape": [-1, 4],
                "output_T_shape": [-1, 2, 4], "output_c_shape": [-1, 2],
                "output_scale": 1, "output_offset": 0,
                "bound_opts": {"activation_bound_option": "same-slope"},
                "split_vars": ["x1", "x2"]}
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f"NAV source contract differs at {key}: {cfg.get(key)!r}")
    if ["".join(expr.split()) for expr in cfg["dynamics_expressions"]] != EXPECTED_DYNAMICS:
        raise ValueError("NAV source dynamics differs from fixed official/author execution")
    if (["".join(expr.split()) for expr in cfg["constraints_unsafe"]] != EXPECTED_UNSAFE or
            ["".join(expr.split()) for expr in cfg["constraints_target"]] != EXPECTED_TARGET):
        raise ValueError("NAV obstacle or terminal goal differs")
    initial = cfg["initial_set"]
    if ([(item["name"], item["interval"], item.get("splits", 0)) for item in initial] !=
            [("x1", [2.9, 3.1], x_splits), ("x2", [2.9, 3.1], y_splits)] +
            [(name, [0, 0], 0) for name in ("x3", "x4", "t", "u1", "u2")]):
        raise ValueError("NAV source initial set or partition differs")
    if (len(boxes) != box_count or len(boxes[0]) != 7 or
            boxes[0][2:] != [[0, 0]] * 5 or not model.is_file() or
            not str(cfg["model_dir"]).endswith("/" + model_name + ".onnx")):
        raise ValueError("NAV first ledger box or selected official ONNX is missing")
    first = boxes[0]
    if (abs(first[0][0] - 2.9) > 1e-12 or
            abs(first[0][1] - (2.905 if x_splits == 40 else 2.94)) > 1e-12 or
            abs(first[1][0] - 2.9) > 1e-12 or
            abs(first[1][1] - (2.9125 if y_splits == 16 else 2.94)) > 1e-12):
        raise ValueError("NAV first ledger box differs from declared partition")
    return cfg, first, source, ledger, model, box_count


def run(args):
    backend = getattr(args, "backend", "huan")
    full_grid = bool(getattr(args, "all_boxes_first_period", False))
    if full_grid and (backend != "p3" or args.instance != "nav-standard"):
        raise ValueError("the full-grid first-period gate is NAV standard P3 only")
    expected_gpu = {"huan": "2", "p3": "1"}[backend]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
        raise RuntimeError(f"NAV {backend} smoke reserves physical GPU {expected_gpu}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    observations = []
    result = {"status": "exception", "instance": args.instance, "method": backend}
    try:
        import yaml

        full_cfg, first, source, ledger, model, box_count = preflight(args.instance)
        boxes = json.loads(ledger.read_text()) if full_grid else [first]
        if full_grid:
            if len(boxes) != 640:
                raise ValueError("NAV standard full-grid gate requires all 640 source boxes")
            for index, box in enumerate(boxes):
                x_index, y_index = divmod(index, 16)
                expected_x = [2.9 + 0.005 * x_index, 2.9 + 0.005 * (x_index + 1)]
                expected_y = [2.9 + 0.0125 * y_index, 2.9 + 0.0125 * (y_index + 1)]
                if (len(box) != 7 or
                        any(abs(box[axis][side] - expected[side]) > 1e-10
                            for axis, expected in enumerate((expected_x, expected_y))
                            for side in (0, 1)) or
                        box[2:] != [[0, 0]] * 5):
                    raise ValueError(f"NAV standard ledger differs at box {index}")
            shutil.copyfile(ledger, output / "initial_boxes.json")
        smoke_cfg = dict(full_cfg)
        smoke_cfg["steps"] = 1
        if backend == "p3":
            smoke_cfg["ode_order"] = 3
        smoke_cfg["model_dir"] = str(model)
        if not full_grid:
            smoke_cfg["initial_set"] = [dict(item, interval=bounds, splits=0)
                                        for item, bounds in zip(full_cfg["initial_set"], first)]
            smoke_cfg["split_vars"] = []
        smoke_cfg["constraints_target"] = []  # t=6 goal is not a t=0.2 smoke property
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(smoke_cfg, sort_keys=False))
        start = {
            "schema": "archcomp26-nav-author-smoke-nohash-v1",
            "run_id": output.name, "started_utc": datetime.now(timezone.utc).isoformat(),
            "instance": args.instance, "method": backend,
            "official_model_copy": str(model), "source_config": str(source),
            "source_box_ledger": str(ledger), "source_box_count": box_count,
            "source_box_index": 0, "first_box_exact": first,
            "smoke_config": str(config_path),
            "state_and_nn_input_order": ["x", "y", "speed", "heading"],
            "nn_output_order": ["speed_rate", "heading_rate"],
            "horizon_scope": "first of 30 control periods, t in [0,0.2] only",
            "initial_scope": "all 640 source boxes" if full_grid else "first source box only",
            "selected_initial_boxes": len(boxes),
            "source_ledger_copy": str(output / "initial_boxes.json") if full_grid else None,
            "full_contract_terminal_target": full_cfg["constraints_target"],
            "smoke_terminal_target_check": "disabled because the 6s target is not a 0.2s property",
            "full_contract_obstacle": full_cfg["constraints_unsafe"],
            "expected_ode_substeps": 20,
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "rpc_port": None, "controller_transport": "in-process CROWN, rpc-float32 quantization mode",
            "full_initial_set_covered": False,
            "full_horizon_covered": False,
            "formal_certificate": False,
        }
        (output / "START.json").write_text(json.dumps(start, indent=2) + "\n")

        torch, driver, engine, cache = prepare_runtime(backend)
        cells = torch.tensor(boxes, dtype=torch.float64)
        if tuple(cells.shape) != (len(boxes), 7, 2):
            raise ValueError("NAV smoke cell shape differs")
        driver.make_cells = lambda _cfg: cells.clone()
        driver.SR_QUEUE = int(smoke_cfg["sr_queue"])

        def build_crown(cfg, device, relax="same-slope", input_layout="native"):
            if relax != "same-slope" or input_layout != "native":
                raise ValueError("NAV CROWN relaxation or input layout differs")
            from auto_LiRPA import BoundedModule
            raw = driver.build_raw_net(cfg, experimental=False)
            return BoundedModule(raw, torch.zeros(1, 4, dtype=torch.float64),
                                 device=device, bound_opts=dict(cfg["bound_opts"]))

        driver.build_crown = build_crown
        start.update(engine=str(engine), driver=str(common.DRIVER), cuda_cache=str(cache))
        original_advance = driver.advance_sparse
        with (output / "observations.jsonl").open("x") as events:
            def observed(*call_args, **call_kwargs):
                state, accepted = original_advance(*call_args, **call_kwargs)
                step = len(observations) + 1
                valid = [bool(item) for item in accepted.detach().cpu().tolist()]
                selected = [index for index, good in enumerate(valid) if good]
                okay = all(valid)
                row = {"substep": step,
                       "accepted": valid if full_grid else valid[0],
                       "solver_status": state.status.detach().cpu().tolist()}
                if selected:
                    engine_obj, settings = call_args[2], call_args[4]
                    tube = driver.hull_ranges_s(state, engine_obj, 4)[selected].detach().cpu().tolist()
                    end_time = torch.full((len(boxes), 2), float(settings.step), dtype=torch.float64,
                                          device=state.pre.device)
                    endpoint = driver.rows_range_over_time_sparse(
                        state, engine_obj, end_time, 4)[selected].detach().cpu().tolist()
                    safe = [(t[0][0] > 2 or t[0][1] < 1 or
                             t[1][0] > 2 or t[1][1] < 1) for t in tube]
                    row.update(tube=tube if full_grid else tube[0],
                               endpoint=endpoint if full_grid else endpoint[0],
                               obstacle_excluded=safe if full_grid else safe[0])
                events.write(json.dumps(row, allow_nan=False) + "\n")
                events.flush()
                observations.append(row)
                if not okay:
                    rejected = [index for index, good in enumerate(valid) if not good]
                    raise FirstRejected(f"NAV first rejected ODE substep {step}, lanes {rejected[:20]}")
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(common.DRIVER), str(config_path), "--device", "cuda:0",
                    "--engine", "sparse", "--strict"]
            if backend == "p3":
                argv += ["--order", "3"]
            argv += ["--crown-domain", "box",
                    "--crown-relax", "same-slope", "--crown-transport", "rpc-float32",
                    "--crown-input-layout", "native", "--nn-mode", "crown",
                    "--print-final-hull", "--metrics-json", str(output / "metrics.json")]
            start["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(start, indent=2) + "\n")
            prior = sys.argv
            sys.argv = argv
            try:
                driver_code = driver.main()
            finally:
                sys.argv = prior

        accepted_total = sum(sum(row["accepted"]) if full_grid else int(row["accepted"])
                             for row in observations)
        complete = (driver_code == 0 and len(observations) == 20 and
                    accepted_total == len(boxes) * 20)
        last_endpoint = observations[-1].get("endpoint") if observations else None
        last_union = ([[min(row[axis][0] for row in last_endpoint),
                        max(row[axis][1] for row in last_endpoint)] for axis in range(4)]
                      if full_grid and complete else None)
        result.update(status="completed_short_prefix" if complete else "incomplete",
                      driver_return=driver_code, expected_substeps=20,
                      observed_substeps=len(observations),
                      initial_boxes=len(boxes), expected_lane_substeps=len(boxes) * 20,
                      accepted_lane_substeps=accepted_total,
                      all_saved_tubes_obstacle_excluded=(complete and all(
                          all(row["obstacle_excluded"]) if full_grid else row["obstacle_excluded"]
                          for row in observations)),
                      last_endpoint=(last_endpoint if not full_grid else None),
                      last_endpoint_union=last_union,
                      wall_s=time.perf_counter() - started,
                      full_initial_set_covered=full_grid, full_horizon_covered=False,
                      formal_certificate=False)
    except FirstRejected as error:
        result.update(status="early_stopped_rejected", error=str(error),
                      observed_substeps=len(observations),
                      initial_boxes=len(boxes),
                      accepted_lane_substeps=sum(
                          sum(row["accepted"]) if full_grid else int(row["accepted"])
                          for row in observations),
                      wall_s=time.perf_counter() - started)
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(),
                      observed_substeps=len(observations), wall_s=time.perf_counter() - started)
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed_short_prefix" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=OPTIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
