#!/usr/bin/env python3
"""Fresh one-box, one-period NAV Huan smoke using the author's state order."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
        smoke_cfg = dict(full_cfg)
        smoke_cfg["steps"] = 1
        if backend == "p3":
            smoke_cfg["ode_order"] = 3
        smoke_cfg["model_dir"] = str(model)
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
        cells = torch.tensor([first], dtype=torch.float64)
        if tuple(cells.shape) != (1, 7, 2):
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
                okay = bool(accepted[0].item())
                row = {"substep": step, "accepted": okay}
                if okay:
                    engine_obj, settings = call_args[2], call_args[4]
                    tube = driver.hull_ranges_s(state, engine_obj, 4)[0].detach().cpu().tolist()
                    end_time = torch.full((1, 2), float(settings.step), dtype=torch.float64,
                                          device=state.pre.device)
                    endpoint = driver.rows_range_over_time_sparse(state, engine_obj, end_time, 4)[0].detach().cpu().tolist()
                    row.update(tube=tube, endpoint=endpoint,
                               obstacle_excluded=(tube[0][0] > 2 or tube[0][1] < 1 or
                                                  tube[1][0] > 2 or tube[1][1] < 1))
                events.write(json.dumps(row, allow_nan=False) + "\n")
                events.flush()
                observations.append(row)
                if not okay:
                    raise FirstRejected(f"NAV first rejected ODE substep {step}")
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

        complete = (driver_code == 0 and len(observations) == 20 and
                    all(row["accepted"] for row in observations))
        result.update(status="completed_short_prefix" if complete else "incomplete",
                      driver_return=driver_code, expected_substeps=20,
                      observed_substeps=len(observations),
                      accepted_substeps=sum(row["accepted"] for row in observations),
                      all_saved_tubes_obstacle_excluded=(complete and all(
                          row["obstacle_excluded"] for row in observations)),
                      last_endpoint=(observations[-1].get("endpoint") if observations else None),
                      wall_s=time.perf_counter() - started,
                      full_initial_set_covered=False, full_horizon_covered=False,
                      formal_certificate=False)
    except FirstRejected as error:
        result.update(status="early_stopped_rejected", error=str(error),
                      observed_substeps=len(observations),
                      accepted_substeps=sum(row["accepted"] for row in observations),
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
