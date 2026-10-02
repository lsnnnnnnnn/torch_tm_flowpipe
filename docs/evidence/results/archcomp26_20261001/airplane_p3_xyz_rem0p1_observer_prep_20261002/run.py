#!/usr/bin/env python3
"""Isolated Airplane continuous full-box Huan/Xiangru attempt."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
HELPER = N / "runs/archcomp26_20261001/author_tora_remain_v1/archcomp26_tora_remain_author_nohash.py"
SAFE_COORDINATES = (1, 6, 7, 8)  # y, phi, theta, psi


class StopUnqualified(Exception):
    pass


def import_helper():
    spec = importlib.util.spec_from_file_location("airplane_saved_gpu_setup", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing saved GPU helper: {HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"schema": "archcomp26-airplane-continuous-fullbox-result-v1",
              "status": "exception", "backend": args.backend, "mode": args.mode}
    observations = []
    try:
        setup = import_helper()
        torch, driver, engine, cache = setup.prepare(args.backend)
        import numpy as np
        import yaml

        source = yaml.safe_load((HERE / "config.yaml").read_text())
        assert source["num_vars"] == 19 and source["num_nn_input"] == 12
        assert source["num_nn_output"] == 6 and source["steps"] == 20
        assert source["step_size"] == .1 and source["ode_step_size"] == .01 and source["ode_order"] == 3
        assert [x["interval"] for x in source["initial_set"]] == (
            [[0, 0]] * 3 + [[0.0, 1.0]] * 6 + [[0, 0]] * 10)
        assert source["constraints_safe"] == [
            "-y - 1", "y - 1", "-phi - 1", "phi - 1",
            "-theta - 1", "theta - 1", "-psi - 1", "psi - 1"]
        assert source["model_dir"] == str(
            N / "runs/archcomp26_20261001/airplane_prep_001/controller_airplane.onnx")
        config = dict(source)
        config["steps"] = 1 if args.mode == "smoke1" else 20
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        cells = torch.tensor([[item["interval"] for item in config["initial_set"]]],
                             dtype=torch.float64)
        assert cells.shape == (1, 19, 2)
        driver.make_cells = lambda _config: cells.clone()
        driver.SR_QUEUE = int(config["sr_queue"])

        receipt = {
            "schema": "archcomp26-airplane-continuous-fullbox-start-v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend, "mode": args.mode,
            "numeric_profile": "Taylor order 3 diagnostic", "ode_order": 3,
            "fixed_preflight": str(HERE / "PREFLIGHT.json"),
            "source_config": str(HERE / "config.yaml"),
            "generated_config": str(config_path),
            "model": source["model_dir"], "engine": str(engine),
            "driver": str(setup.DRIVER), "gpu_setup_helper": str(HELPER),
            "preloaded_cache": str(cache), "physical_box_count": 1,
            "physical_state_order": [item["name"] for item in source["initial_set"][:12]],
            "controller_output_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
            "property": "y,phi,theta,psi in [-1,1] for every t in [0,2]",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "method": "author sparse engine, strict plant, box same-slope CROWN, native input layout, rpc-float32",
            "stop_policy": "stop after first rejected or saved whole-step tube outside safe box",
            "qualification": "saved box/author checker only; no end-to-end floating-point NN certificate",
        }
        (output / "START.json").write_text(json.dumps(receipt, indent=2) + "\n")
        dtype = np.dtype([("lane", "<u8"), ("step", "<u8"), ("h", "<f8"),
                          ("bounds", "<f8", (12, 4))])
        union = np.array([[np.inf, -np.inf]] * 12, dtype=np.float64)
        final_endpoint = None
        original_advance = driver.advance_sparse
        stopped = None
        driver_code = None
        tee = setup.Tee(sys.stdout)
        with (output / "ranges.bin").open("xb") as ranges, (
                output / "observations.jsonl").open("x") as log:
            def observed(*call_args, **call_kwargs):
                nonlocal final_endpoint
                state, accepted = original_advance(*call_args, **call_kwargs)
                step_number = len(observations) + 1
                if not bool(accepted.all()):
                    row = {"substep": step_number, "accepted": False,
                           "saved_tube_inside_safe": None,
                           "stop_reason": "rejected_lane"}
                    log.write(json.dumps(row) + "\n"); log.flush()
                    observations.append(row)
                    raise StopUnqualified("rejected_lane")
                engine_obj, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, engine_obj, 12)
                endpoint_time = torch.full((1, 2), float(settings.step),
                                           dtype=torch.float64, device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(
                    state, engine_obj, endpoint_time, 12)
                bounds = torch.cat((tube, endpoint), dim=-1).detach().cpu().numpy()
                with (output / f"precheck_bounds_step_{step_number:04d}.npy").open("xb") as raw:
                    np.save(raw, bounds, allow_pickle=False)
                predicates = {
                    "substep": step_number,
                    "state_order": [item["name"] for item in source["initial_set"][:12]],
                    "bounds_columns": ["tube_lo", "tube_hi", "endpoint_lo", "endpoint_hi"],
                    "finite_by_bound": np.isfinite(bounds[0]).tolist(),
                    "tube_lo_le_endpoint_lo": (bounds[:, :, 0] <= bounds[:, :, 2])[0].tolist(),
                    "endpoint_lo_le_endpoint_hi": (bounds[:, :, 2] <= bounds[:, :, 3])[0].tolist(),
                    "endpoint_hi_le_tube_hi": (bounds[:, :, 3] <= bounds[:, :, 1])[0].tolist(),
                }
                (output / f"precheck_predicates_step_{step_number:04d}.json").write_text(
                    json.dumps(predicates, indent=2) + "\n")
                if (not np.isfinite(bounds).all() or
                        not (bounds[:, :, 0] <= bounds[:, :, 2]).all() or
                        not (bounds[:, :, 2] <= bounds[:, :, 3]).all() or
                        not (bounds[:, :, 3] <= bounds[:, :, 1]).all()):
                    raise FloatingPointError(f"invalid accepted interval at {step_number}")
                rows = np.empty(1, dtype=dtype)
                rows["lane"] = 0
                rows["step"] = step_number
                rows["h"] = float(settings.step)
                rows["bounds"] = bounds
                ranges.write(rows.tobytes()); ranges.flush()
                union[:, 0] = np.minimum(union[:, 0], bounds[0, :, 0])
                union[:, 1] = np.maximum(union[:, 1], bounds[0, :, 1])
                final_endpoint = bounds[0, :, 2:4].tolist()
                safe = bool((bounds[0, SAFE_COORDINATES, 0] >= -1).all() and
                            (bounds[0, SAFE_COORDINATES, 1] <= 1).all())
                row = {"substep": step_number, "accepted": True,
                       "saved_tube_inside_safe": safe,
                       "stop_reason": None if safe else "saved_tube_not_proven_safe"}
                log.write(json.dumps(row) + "\n"); log.flush()
                observations.append(row)
                if not safe:
                    raise StopUnqualified("saved_tube_not_proven_safe")
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(setup.DRIVER), str(config_path), "--device", "cuda:0",
                    "--engine", "sparse", "--strict", "--crown-domain", "box",
                    "--crown-relax", "same-slope", "--crown-transport", "rpc-float32",
                    "--crown-input-layout", "native", "--nn-mode", "crown",
                    "--metrics-json", str(output / "metrics.json")]
            receipt["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(receipt, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            try:
                with redirect_stdout(tee):
                    try:
                        driver_code = driver.main()
                    except StopUnqualified as error:
                        stopped = str(error)
            finally:
                sys.argv = previous

        expected = config["steps"] * 10
        checker_unqualified = any(line in tee.lines for line in (
            "Unsafe.", "Unknown.", "Flow* terminated.", "FALSIFIED", "UNKNOWN"))
        complete = (driver_code == 0 and not checker_unqualified and
                    len(observations) == expected and
                    all(x["accepted"] and x["saved_tube_inside_safe"]
                        for x in observations))
        result.update(
            status=("completed_short_prefix" if args.mode == "smoke1" else "completed")
            if complete else "incomplete",
            stop_reason=stopped, driver_return=driver_code,
            expected_substeps=expected, observed_substeps=len(observations),
            accepted_substeps=sum(x["accepted"] for x in observations),
            safe_saved_substeps=sum(x["saved_tube_inside_safe"] is True
                                    for x in observations),
            author_checker_lines=tee.lines,
            author_checker_unqualified=checker_unqualified,
            full_time_tube_union=union.tolist() if complete and args.mode == "full" else None,
            observed_tube_union=union.tolist() if final_endpoint else None,
            full_initial_box_terminal_endpoint=(final_endpoint if complete and
                                                args.mode == "full" else None),
            last_saved_endpoint=final_endpoint,
            wall_s=time.perf_counter() - started,
            end_to_end_floating_point_nn_certificate=False,
        )
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(),
                      observed_substeps=len(observations),
                      wall_s=time.perf_counter() - started)
        raise
    finally:
        (output / "RESULT.json").write_text(json.dumps(result, indent=2,
                                                        allow_nan=False) + "\n")
    return 0 if result["status"].startswith("completed") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("huan", "xiangru"), required=True)
    parser.add_argument("--mode", choices=("smoke1", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
