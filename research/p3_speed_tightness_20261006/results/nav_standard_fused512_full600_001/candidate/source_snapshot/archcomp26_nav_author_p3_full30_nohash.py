#!/usr/bin/env python3
"""New NAV standard working-P3 full-grid/full-horizon attempt."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

import archcomp26_nav_author_gpu_smoke_nohash as nav
import archcomp26_nav_author_p3_smoke_nohash as p3


BOXES = 640
PERIODS = 30
SUBSTEPS = 600


class FirstRejected(RuntimeError):
    pass


def checked_boxes(path):
    boxes = json.loads(path.read_text())
    if len(boxes) != BOXES:
        raise ValueError("NAV standard requires the original 640-box ledger")
    for index, box in enumerate(boxes):
        x_index, y_index = divmod(index, 16)
        expected = ((2.9 + x_index * .005, 2.9 + (x_index + 1) * .005),
                    (2.9 + y_index * .0125, 2.9 + (y_index + 1) * .0125))
        if (len(box) != 7 or box[2:] != [[0, 0]] * 5 or
                any(abs(box[axis][side] - expected[axis][side]) > 1e-10
                    for axis in (0, 1) for side in (0, 1))):
            raise ValueError(f"NAV source grid differs at box {index}")
    return boxes


def run(output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    observations = []
    accepted_total = 0
    range_records = 0
    last_endpoint_union = None
    verdict_lines = []
    result = {"schema": "archcomp26-nav-standard-working-p3-full30-nohash-v1",
              "status": "exception", "run_id": output.name, "instance": "nav-standard",
              "method": "working_p3", "full_horizon_covered": False,
              "end_to_end_floating_point_nn_certificate": False}
    try:
        import numpy as np
        import yaml

        source_config, first, source, ledger, model, count = nav.preflight("nav-standard")
        boxes = checked_boxes(ledger)
        if count != BOXES or boxes[0] != first:
            raise ValueError("NAV source first box or count differs")
        config = dict(source_config)
        config["model_dir"] = str(model)
        config["ode_order"] = 3
        if (config["steps"] != PERIODS or config["step_size"] != .2 or
                config["ode_step_size"] != .01 or
                len(config["constraints_unsafe"]) != 4 or
                len(config["constraints_target"]) != 4):
            raise ValueError("NAV full-horizon obstacle/target contract differs")
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copyfile(ledger, output / "initial_boxes.json")
        record = {
            "schema": "archcomp26-nav-standard-working-p3-full30-start-nohash-v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": output.name, "instance": "nav-standard",
            "profile": "fixed-2026-official-point-ONNX plus author executable NAV state order",
            "method": "working P3 order 3 / validation order 4, strict endpoint and injection, box same-slope CROWN, rpc-float32",
            "source_config": str(source), "source_box_ledger": str(ledger),
            "official_model_copy": str(model), "generated_config": str(config_path),
            "saved_initial_boxes": str(output / "initial_boxes.json"),
            "initial_box_count": BOXES, "control_periods": PERIODS,
            "control_period_s": .2, "ode_substeps": SUBSTEPS,
            "ode_step_s": .01, "working_order": 3, "validation_order": 4,
            "physical_state_and_nn_input_order": ["x", "y", "speed", "heading"],
            "nn_output_order": ["speed_rate", "heading_rate"],
            "all_time_closed_obstacle": "x,y in [1,2]",
            "terminal_closed_target": "x,y in [-0.5,0.5] at t=6",
            "unsafe_checker_expressions": config["constraints_unsafe"],
            "terminal_checker_expressions": config["constraints_target"],
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "timeout_s": int(os.environ.get("NAV_P3_TIMEOUT_S", "3600")),
            "gpu_allocation_cap_bytes": 11 * 2**30,
            "rpc_port": None, "controller_evaluation": "in-process CROWN",
            "stop_at_first_numerical_rejection": True,
            "range_record_format": "little-endian uint32 lane, uint32 step, float64[4,4] [tube_lo,tube_hi,end_lo,end_hi]",
            "range_record_bytes": 136,
            "qualification": "full numerical horizon only if all 640x600 accepted; property verdict separately; no independent end-to-end floating-point NNCS proof",
        }
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")

        torch, driver, engine, cache = p3.prepare_p3("p3")
        cells = torch.tensor(boxes, dtype=torch.float64)
        if tuple(cells.shape) != (BOXES, 7, 2):
            raise ValueError("NAV source cell tensor shape differs")
        driver.make_cells = lambda _cfg: cells.clone()
        driver.SR_QUEUE = int(config["sr_queue"])

        def build_crown(cfg, device, relax="same-slope", input_layout="native"):
            if relax != "same-slope" or input_layout != "native":
                raise ValueError("NAV CROWN relaxation or input layout differs")
            from auto_LiRPA import BoundedModule
            raw = driver.build_raw_net(cfg, experimental=False)
            return BoundedModule(raw, torch.zeros(1, 4, dtype=torch.float64),
                                 device=device, bound_opts=dict(cfg["bound_opts"]))

        driver.build_crown = build_crown
        record.update(engine=str(engine), shared_driver=str(nav.common.DRIVER),
                      cuda_cache=str(cache))
        dtype = np.dtype([("lane", "<u4"), ("step", "<u4"),
                          ("bounds", "<f8", (4, 4))])
        if dtype.itemsize != 136:
            raise ValueError("NAV saved range record size differs")
        with (output / "ranges.bin").open("xb") as ranges, \
                (output / "observations.jsonl").open("x") as events:
            original_advance = driver.advance_sparse

            def observed(*call_args, **call_kwargs):
                nonlocal accepted_total, range_records, last_endpoint_union
                state, accepted = original_advance(*call_args, **call_kwargs)
                step = len(observations) + 1
                valid = accepted.detach().cpu().numpy().astype(bool)
                if valid.shape != (BOXES,):
                    raise ValueError("NAV accepted mask differs from the 640-box grid")
                accepted_count = int(valid.sum())
                accepted_total += accepted_count
                row = {"substep": step, "accepted_boxes": accepted_count,
                       "rejected_lanes": np.flatnonzero(~valid).tolist(),
                       "solver_status_counts": {
                           str(code): int(count) for code, count in zip(
                               *np.unique(state.status.detach().cpu().numpy(),
                                          return_counts=True))}}
                if accepted_count != BOXES:
                    events.write(json.dumps(row) + "\n")
                    events.flush()
                    observations.append(row)
                    raise FirstRejected(f"first rejected NAV P3 substep {step}: {row['rejected_lanes'][:20]}")
                eng, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, eng, 4).detach().cpu().numpy()
                end_time = torch.full((BOXES, 2), float(settings.step),
                                      dtype=torch.float64, device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(
                    state, eng, end_time, 4).detach().cpu().numpy()
                if tube.shape != (BOXES, 4, 2) or endpoint.shape != (BOXES, 4, 2):
                    raise ValueError("NAV saved tube or endpoint shape differs")
                data = np.empty(BOXES, dtype=dtype)
                data["lane"] = np.arange(BOXES)
                data["step"] = step
                data["bounds"] = np.concatenate((tube, endpoint), axis=-1)
                ranges.write(data.tobytes())
                ranges.flush()
                range_records += BOXES
                row["saved_range_records"] = BOXES
                row["obstacle_intersecting_saved_tubes"] = int((
                    (tube[:, 0, 0] <= 2) & (tube[:, 0, 1] >= 1) &
                    (tube[:, 1, 0] <= 2) & (tube[:, 1, 1] >= 1)).sum())
                observations.append(row)
                events.write(json.dumps(row) + "\n")
                events.flush()
                bounds = data["bounds"]
                if (not np.isfinite(bounds).all() or
                        not (bounds[..., 0] <= bounds[..., 2]).all() or
                        not (bounds[..., 2] <= bounds[..., 3]).all() or
                        not (bounds[..., 3] <= bounds[..., 1]).all()):
                    raise FloatingPointError(f"invalid saved NAV P3 interval at substep {step}")
                last_endpoint_union = [[float(endpoint[:, axis, 0].min()),
                                        float(endpoint[:, axis, 1].max())]
                                       for axis in range(4)]
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(nav.common.DRIVER), str(config_path), "--device", "cuda:0",
                    "--engine", "sparse", "--strict", "--order", "3",
                    "--crown-domain", "box", "--crown-relax", "same-slope",
                    "--crown-transport", "rpc-float32", "--crown-input-layout", "native",
                    "--nn-mode", "crown", "--print-final-hull", "--metrics-json",
                    str(output / "metrics.json")]
            record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
            old_argv = sys.argv
            sys.argv = argv
            tee = nav.common.Tee(sys.stdout)
            try:
                with redirect_stdout(tee):
                    driver_code = driver.main()
            finally:
                sys.argv = old_argv
                verdict_lines = tee.lines

        complete = (driver_code == 0 and len(observations) == SUBSTEPS and
                    accepted_total == BOXES * SUBSTEPS and
                    range_records == BOXES * SUBSTEPS)
        result.update(status="completed" if complete else "incomplete",
                      driver_return=driver_code, observed_substeps=len(observations),
                      accepted_lane_substeps=accepted_total, range_records=range_records,
                      expected_substeps=SUBSTEPS,
                      expected_lane_substeps=BOXES * SUBSTEPS,
                      author_verdict_lines=verdict_lines,
                      last_endpoint_union=last_endpoint_union if complete else None,
                      full_initial_set_covered=True, full_horizon_covered=complete,
                      wall_s=time.perf_counter() - started)
    except FirstRejected as error:
        result.update(status="early_stopped_rejected", error=str(error),
                      observed_substeps=len(observations),
                      accepted_lane_substeps=accepted_total,
                      range_records=range_records,
                      author_verdict_lines=verdict_lines,
                      wall_s=time.perf_counter() - started)
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(),
                      observed_substeps=len(observations),
                      accepted_lane_substeps=accepted_total,
                      range_records=range_records,
                      author_verdict_lines=verdict_lines,
                      wall_s=time.perf_counter() - started)
        raise
    finally:
        result["ended_utc"] = datetime.now(timezone.utc).isoformat()
        result["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
