#!/usr/bin/env python3
"""Isolated TORA-remain P3 attempt on the fixed 12-box 2026 contract."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

import archcomp26_dp_p3_nohash as helper
import archcomp26_tora_remain_author_nohash as author


N = author.N
ENGINE = helper.ENGINE
DRIVER = author.DRIVER
MODEL = author.MODEL


def prepare():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("TORA P3 attempt is reserved for physical GPU3")
    for key, value in helper.ENV.items():
        os.environ[key] = value
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(ENGINE / "src"))
    helper.apply_guards()
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
    ck.load_cuda_extension = helper.prohibit
    ck._ext = helper.load_binary("flowstar_seg_kernels",
                                 helper.EXTENSIONS["flowstar_seg_kernels"])
    ck._tried = True
    core = helper.load_python("tora_p3_reciprocal_core", helper.RECIP_CORE)
    for name in ("rec_series_valid_g", "rec_series_valid", "rec_series_replay"):
        setattr(elem, name, getattr(core, name))
    tape = helper.load_python("flowstar_gpu.tape_kernels", helper.RECIP_SOURCE)
    tape.load_cuda_extension = helper.prohibit
    tape._ext = helper.load_binary("flowstar_recip_geom_replay_1f9efda6325c",
                                   helper.EXTENSIONS["flowstar_recip_geom_replay_1f9efda6325c"])
    tape._vext = helper.load_binary("flowstar_recip_geom_valid_1f9efda6325c",
                                    helper.EXTENSIONS["flowstar_recip_geom_valid_1f9efda6325c"])
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
        module._ext = helper.load_binary(extension_name, helper.EXTENSIONS[extension_name])
        if hasattr(module, "_tried"):
            module._tried = True
    if not (ck.available() and tape.available() and tape.valid_available()):
        raise RuntimeError("saved P3 CUDA libraries unavailable")

    driver = helper.load_python("archcomp26_tora_p3_driver", DRIVER)
    from flowstar_gpu import sparse_exec as se
    if not hasattr(se, "COMPOSE_PARENT_ASSEMBLY"):
        se.COMPOSE_PARENT_ASSEMBLY = None
    endpoint = helper.load_python("archcomp26_tora_p3_strict_endpoint", helper.ENDPOINT)
    driver.end_of_time_s = endpoint.end_of_time_s
    holder = {}
    old_engine = driver.SparseEngine

    def capture_engine(*args, **kwargs):
        if holder:
            raise RuntimeError("expected one TORA P3 engine")
        engine = old_engine(*args, **kwargs)
        if (engine.tables.n, engine.tables.k) != (6, 3):
            raise ValueError("TORA P3 needs 6 variables and working order 3")
        holder["engine"] = engine
        return engine

    def strict_inject(state, matrix, lower, upper, control_ids, nn_input):
        if "engine" not in holder:
            raise RuntimeError("strict injection before engine creation")
        with torch.no_grad():
            return helper.strict_injection(state, matrix, lower, upper,
                                           control_ids, nn_input, holder["engine"])

    def official_build(config, device, relax="same-slope", input_layout="native"):
        if relax != "same-slope" or input_layout != "native":
            raise ValueError("TORA uses native-layout same-slope CROWN")
        from auto_LiRPA import BoundedModule
        raw = driver.build_raw_net(config, experimental=False)
        return BoundedModule(raw, torch.zeros(1, 1, 1, 4, dtype=torch.float64),
                             device=device, bound_opts=dict(config["bound_opts"]))

    driver.SparseEngine = capture_engine
    driver.inject_controls_s = strict_inject
    driver.build_crown = official_build
    return torch, driver, cap


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    observations = []
    result = {"status": "exception", "mode": args.mode, "observed_substeps": 0}
    record = {
        "schema": "archcomp26-tora-remain-p3-nohash-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "source_config": str(author.SOURCE_CONFIG),
        "boxes": str(author.BOXES), "engine": str(ENGINE),
        "shared_driver": str(DRIVER), "model": str(MODEL),
        "controller_boundary": "raw ONNX f(x) injected into u1; plant dx4=u1-10 applies offset once",
        "property": "all four physical states in [-2,2] on every T=20 substep",
        "method": "working P3; strict plant/endpoint/injection; box same-slope CROWN; rpc-float32",
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "qualification": "new 12-box P3 diagnostic; no independent end-to-end floating NNCS proof",
        "prebuilt_libraries": {
            name: {"path": str(path), "bytes": path.stat().st_size,
                   "mtime_ns": path.stat().st_mtime_ns}
            for name, path in helper.EXTENSIONS.items()
        },
    }
    (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        import numpy as np
        import yaml
        config, boxes, _ = author.checked_contract(
            author.SOURCE_CONFIG, author.BOXES, MODEL, check_model=False)
        config["model_dir"] = str(MODEL)
        config["steps"] = 1 if args.mode == "smoke1" else 20
        config["ode_step_size"] = args.ode_step_size
        expected = config["steps"] * round(config["step_size"] / config["ode_step_size"])
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        torch, driver, cap = prepare()
        cells = torch.tensor(boxes, dtype=torch.float64)
        if tuple(cells.shape) != (12, 6, 2):
            raise ValueError("TORA 12-box initial grid shape differs")
        driver.make_cells = lambda _config: cells.clone()
        driver.SR_QUEUE = 1000
        record.update(generated_config=str(config_path),
                      expected_substeps=expected,
                      numerical_profile={"ode_step_size": args.ode_step_size},
                      cuda_memory_cap_bytes=cap)
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")

        dtype = np.dtype([("lane", "<u8"), ("step", "<u8"), ("h", "<f8"),
                          ("bounds", "<f8", (4, 4))])
        tube_union = np.array([[np.inf, -np.inf]] * 4, dtype=np.float64)
        final_endpoint = None
        prefix_safe_substeps = 0
        original_advance = driver.advance_sparse
        with (output / "ranges.bin").open("xb") as ranges, (output / "observations.jsonl").open("x") as log:
            def observed(*call_args, **call_kwargs):
                nonlocal final_endpoint, prefix_safe_substeps
                state, accepted = original_advance(*call_args, **call_kwargs)
                engine_obj, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, engine_obj, 4)
                end_time = torch.full((12, 2), float(settings.step), dtype=torch.float64,
                                      device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, engine_obj, end_time, 4)
                bounds = torch.cat((tube, endpoint), dim=-1).detach().cpu().numpy()
                valid = np.asarray(accepted.detach().cpu().tolist(), dtype=bool)
                if valid.shape != (12,) or bounds.shape != (12, 4, 4):
                    raise ValueError("TORA P3 lane/interval shape differs")
                selected = bounds[valid]
                if (not np.isfinite(selected).all() or
                        not (selected[..., 0] <= selected[..., 2]).all() or
                        not (selected[..., 2] <= selected[..., 3]).all() or
                        not (selected[..., 3] <= selected[..., 1]).all()):
                    raise FloatingPointError("accepted P3 TORA interval record invalid")
                if valid.any():
                    tube_union[:, 0] = np.minimum(tube_union[:, 0], selected[..., 0].min(axis=0))
                    tube_union[:, 1] = np.maximum(tube_union[:, 1], selected[..., 1].max(axis=0))
                    final_endpoint = [[float(selected[:, i, 2].min()),
                                       float(selected[:, i, 3].max())] for i in range(4)]
                inside = bool(valid.all() and (selected[..., 0] >= -2).all()
                              and (selected[..., 1] <= 2).all())
                step_number = len(observations) + 1
                if inside and prefix_safe_substeps == step_number - 1:
                    prefix_safe_substeps = step_number
                rows = np.empty(12, dtype=dtype)
                rows["lane"] = np.arange(12)
                rows["step"] = step_number
                rows["h"] = float(settings.step)
                rows["bounds"] = bounds
                ranges.write(rows.tobytes())
                ranges.flush()
                row = {"substep": step_number, "accepted_count": int(valid.sum()),
                       "rejected_lanes": np.flatnonzero(~valid).tolist(),
                       "whole_step_all_lanes_inside_safe": inside}
                observations.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(DRIVER), str(config_path), "--device", "cuda:0",
                    "--engine", "sparse", "--strict", "--order", "3",
                    "--crown-domain", "box", "--crown-relax", "same-slope",
                    "--crown-transport", "rpc-float32", "--crown-input-layout", "native",
                    "--nn-mode", "crown", "--print-final-hull",
                    "--metrics-json", str(output / "metrics.json")]
            record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            tee = author.Tee(sys.stdout)
            try:
                with redirect_stdout(tee):
                    driver_code = driver.main()
            finally:
                sys.argv = previous

        complete = (driver_code == 0 and len(observations) == expected and
                    all(row["accepted_count"] == 12 for row in observations))
        safe = complete and prefix_safe_substeps == expected
        if "Unsafe." in tee.lines or "FALSIFIED" in tee.lines:
            checker = "UNSAFE_OR_FALSIFIED"
        elif any(line in tee.lines for line in ("Unknown.", "UNKNOWN", "Flow* terminated.")):
            checker = "UNKNOWN"
        elif safe:
            checker = "VERIFIED_BY_AUTHOR_CHECKER_SILENCE_AND_SAVED_BOX_CONTAINMENT"
        else:
            checker = "UNRESOLVED"
        metrics = json.loads((output / "metrics.json").read_text()) if (output / "metrics.json").is_file() else {}
        result.update(
            status=("completed_short_prefix" if args.mode == "smoke1" else "completed")
            if complete else "incomplete",
            driver_return=driver_code, expected_substeps=expected,
            observed_substeps=len(observations),
            accepted_lane_substeps=sum(row["accepted_count"] for row in observations),
            all_lanes_accepted=complete,
            all_saved_tubes_inside_safe=safe,
            all_lanes_accepted_and_saved_tubes_safe_prefix_substeps=prefix_safe_substeps,
            first_rejected_substep=next((row["substep"] for row in observations
                                         if row["accepted_count"] != 12), None),
            first_saved_tube_outside_safe_substep=next(
                (row["substep"] for row in observations
                 if row["accepted_count"] == 12 and not row["whole_step_all_lanes_inside_safe"]),
                None),
            author_checker_lines=tee.lines,
            author_checker_interpretation=checker,
            observed_prefix_tube_union=tube_union.tolist() if final_endpoint else None,
            full_time_tube_union=tube_union.tolist() if complete and args.mode == "full" else None,
            last_observed_accepted_endpoint_union=final_endpoint,
            terminal_endpoint_union=final_endpoint if complete and args.mode == "full" else None,
            terminal_endpoint_union_width=([hi - lo for lo, hi in final_endpoint]
                                           if complete and args.mode == "full" else None),
            driver_elapsed_s=metrics.get("elapsed_s"),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            wall_s=time.perf_counter() - started,
            end_to_end_floating_point_nn_certificate=False,
        )
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), observed_substeps=len(observations),
                      wall_s=time.perf_counter() - started)
        raise
    finally:
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"].startswith("completed") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke1", "full"), required=True)
    parser.add_argument("--ode-step-size", type=float, choices=(0.1, 0.05), default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
