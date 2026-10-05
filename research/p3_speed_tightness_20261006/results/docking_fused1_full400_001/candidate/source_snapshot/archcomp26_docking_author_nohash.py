#!/usr/bin/env python3
"""Isolated 2026 Docking full-box Huan/Xiangru attempt with a radial checker."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

import archcomp26_tora_remain_author_nohash as common


N = common.N
MODEL = Path("/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/Docking/model.onnx")
PROPERTY = "sqrt(vx^2+vy^2)-0.2-0.002054*sqrt(sx^2+sy^2)"
INITIAL = [[70.0, 106.0], [70.0, 106.0], [-0.28, 0.28], [-0.28, 0.28]]


def config(periods):
    return {
        "run_attack": False, "num_vars": 7, "num_nn_input": 4, "num_nn_output": 2,
        "steps": periods, "step_size": 1.0, "ode_step_size": 0.1,
        "ode_order": 3, "cut_off_threshold": 1e-6,
        "remainder_estimation": [-0.01, 0.01], "sr_queue": 1000,
        "initial_set": [
            {"name": name, "interval": bounds}
            for name, bounds in zip(("sx", "sy", "vx", "vy", "t", "Fx", "Fy"),
                                    INITIAL + [[0.0, 0.0]] * 3)
        ],
        "dynamics_expressions": [
            "vx", "vy", "2*0.001027*vy+3*0.001027^2*sx+Fx/12",
            "-2*0.001027*vx+Fy/12", "1", "0", "0",
        ],
        "constraints_safe": [PROPERTY], "refine_specs": False,
        "model_dir": str(MODEL), "input_shape": [-1, 4],
        "output_T_shape": [-1, 2, 4], "output_c_shape": [-1, 2],
        "output_scale": 1, "output_offset": 0,
        "bound_opts": {"activation_bound_option": "same-slope"},
        "split_vars": [],
    }


def radial_margin(boxes, torch, iv, tr):
    """Outward interval enclosure of exact speed minus radial speed limit.

    Squared norms are nonnegative mathematically. Clamping their interval
    lower endpoints to zero removes the negative subnormal introduced by an
    outward interval addition of two zero lower endpoints.
    """
    def norm(i, j):
        squares = iv.add(iv.pow_int(boxes[:, i], 2), iv.pow_int(boxes[:, j], 2))
        squares = torch.stack((squares[:, 0].clamp_min(0.0), squares[:, 1]), dim=-1)
        value, bad = tr.sqrt_iv(squares)
        if bool(bad.any()):
            raise FloatingPointError("norm interval left nonnegative domain")
        return value

    velocity = norm(2, 3)
    position = norm(0, 1)
    c = torch.full((boxes.shape[0],), 0.2, dtype=boxes.dtype, device=boxes.device)
    k = torch.full_like(c, 0.002054)
    c = torch.stack((torch.nextafter(c, torch.full_like(c, -torch.inf)),
                     torch.nextafter(c, torch.full_like(c, torch.inf))), dim=-1)
    k = torch.stack((torch.nextafter(k, torch.full_like(k, -torch.inf)),
                     torch.nextafter(k, torch.full_like(k, torch.inf))), dim=-1)
    return iv.sub(velocity, iv.add(c, iv.mul(k, position)))


class Tee:
    def __init__(self, stream):
        self.stream = stream
        self.pending = ""
        self.property_lines = []

    def write(self, value):
        self.stream.write(value)
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip() in {"Unsafe", "Unknown", "Unsafe.", "Unknown.",
                                "Flow* terminated.", "VERIFIED", "FALSIFIED", "UNKNOWN"}:
                self.property_lines.append(line.strip())
        return len(value)

    def flush(self):
        self.stream.flush()


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start_clock = time.perf_counter()
    periods = 1 if args.mode == "smoke1" else 40
    result = {"status": "exception", "backend": args.backend, "mode": args.mode}
    observations, safety = [], []
    start = {
        "schema": "archcomp26-docking-author-radial-nohash-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "2026-paper-Docking-complete-initial-40s-all-time-radial-speed",
        "method": args.backend, "mode": args.mode,
        "model": str(MODEL), "model_identity": "2026 official bytes previously directly compared with this saved participant model",
        "driver": str(common.DRIVER), "engine": str(common.ENGINES[args.backend]),
        "initial_physical_box": INITIAL, "physical_box_count": 1,
        "physical_state_order": ["sx", "sy", "vx", "vy"],
        "controller_input": "raw four-state vector", "controller_output": "physical (Fx,Fy), held for 1s",
        "property": PROPERTY + " <= 0 for all t in [0,40]",
        "periods": periods, "period_s": 1.0, "ode_step_s": 0.1,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "qualification": "author GPU plant/CROWN diagnostic; radial checker uses outward box intervals; no independent end-to-end NN floating-point certificate",
    }
    (output / "START.json").write_text(json.dumps(start, indent=2) + "\n")
    try:
        import torch
        import yaml

        torch, driver, _engine, _cache = common.prepare(args.backend)
        from flowstar_gpu import interval as iv, transcendental as tr

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
            raise RuntimeError("Docking attempt reserved for physical GPU 3")
        if not MODEL.is_file():
            raise FileNotFoundError(MODEL)
        cfg = config(periods)
        cfg_path = output / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cells = driver.make_cells(cfg)
        if tuple(cells.shape) != (1, 7, 2) or cells[0, :4].tolist() != INITIAL:
            raise RuntimeError("Docking full initial box changed")
        initial_q = radial_margin(cells, torch, iv, tr)[0].tolist()
        if not initial_q[1] < 0:
            raise RuntimeError("Docking initial box safety precheck is not proven")
        start.update(config=str(cfg_path), initial_margin_interval=initial_q,
                     checker="box interval norms with mathematically nonnegative squared sums")
        (output / "START.json").write_text(json.dumps(start, indent=2) + "\n")

        original_ranges = driver.SpecGroup.ranges
        original_advance = driver.advance_sparse
        with (output / "ranges.jsonl").open("x") as ranges, (output / "safety.jsonl").open("x") as safefile:
            def checked_ranges(group, boxes, active, **kwargs):
                if group.texts != [PROPERTY]:
                    return original_ranges(group, boxes, active, **kwargs)
                q = radial_margin(boxes, torch, iv, tr).unsqueeze(1)
                event = {"substep": len(observations), "q_interval": q[0, 0].detach().cpu().tolist()}
                safety.append(event)
                safefile.write(json.dumps(event, allow_nan=False) + "\n")
                safefile.flush()
                return q

            def observed(*call_args, **call_kwargs):
                state, accepted = original_advance(*call_args, **call_kwargs)
                engine_obj, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, engine_obj, 7)
                endpoint_time = torch.full((1, 2), float(settings.step), dtype=torch.float64,
                                           device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, engine_obj, endpoint_time, 7)
                q = radial_margin(tube, torch, iv, tr)
                row = {"substep": len(observations) + 1,
                       "accepted": bool(accepted[0].item()),
                       "tube": tube[0, :4].detach().cpu().tolist(),
                       "endpoint": endpoint[0, :4].detach().cpu().tolist(),
                       "tube_radial_margin": q[0].detach().cpu().tolist()}
                observations.append(row)
                ranges.write(json.dumps(row, allow_nan=False) + "\n")
                ranges.flush()
                return state, accepted

            driver.SpecGroup.ranges = checked_ranges
            driver.advance_sparse = observed
            argv = [str(common.DRIVER), str(cfg_path), "--device", "cuda:0",
                    "--engine", "sparse", "--strict", "--crown-domain", "box",
                    "--crown-relax", "same-slope", "--crown-transport", "rpc-float32",
                    "--crown-input-layout", "native", "--nn-mode", "crown",
                    "--print-final-hull", "--metrics-json", str(output / "metrics.json")]
            start["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(start, indent=2) + "\n")
            prior_argv = sys.argv
            sys.argv = argv
            tee = Tee(sys.stdout)
            try:
                with redirect_stdout(tee):
                    driver_code = driver.main()
            finally:
                sys.argv = prior_argv

        expected = periods * 10
        completed = (driver_code == 0 and len(observations) == expected and
                     len(safety) == expected and all(row["accepted"] for row in observations))
        q_upper = max((row["q_interval"][1] for row in safety), default=None)
        if any(line in tee.property_lines for line in ("Unsafe", "Unsafe.", "FALSIFIED")):
            verdict = "UNSAFE_REPORTED_BY_AUTHOR_CHECKER"
        elif any(line in tee.property_lines for line in ("Unknown", "Unknown.", "UNKNOWN", "Flow* terminated.")):
            verdict = "UNKNOWN_REPORTED_BY_AUTHOR_CHECKER"
        elif completed and q_upper is not None and q_upper <= 0:
            verdict = "ALL_SAVED_BOX_CHECKS_SAFE"
        else:
            verdict = "UNRESOLVED"
        result.update(status="completed" if completed else "early_stopped",
                      driver_return=driver_code, expected_substeps=expected,
                      observed_substeps=len(observations), safety_events=len(safety),
                      accepted_substeps=sum(row["accepted"] for row in observations),
                      maximum_saved_margin_upper=q_upper,
                      checker_lines=tee.property_lines, checker_verdict=verdict,
                      final_physical_endpoint=(observations[-1]["endpoint"] if observations else None),
                      wall_s=time.perf_counter() - start_clock,
                      end_to_end_floating_point_nn_certificate=False)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(),
                      observed_substeps=len(observations), safety_events=len(safety),
                      wall_s=time.perf_counter() - start_clock)
        raise
    finally:
        if "torch" in locals() and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=common.ENGINES, required=True)
    parser.add_argument("--mode", choices=("smoke1", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
