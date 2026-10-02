#!/usr/bin/env python3
"""Run Huan/Xiangru Attitude Control with the corrected 2026 unsafe box."""

import argparse
from contextlib import redirect_stdout
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
DRIVER = common.DRIVER
SOURCE_CONFIG = N / "runs/archcomp_review_20260923/contracts/attitude_control.yaml"
BOXES = N / "runs/archcomp_review_20260923/contracts/attitude_control_boxes.json"
MODEL = N / "runs/archcomp26_20261001/attitude_prep_001/official_attitude_control_3_64_torch.onnx"
UNSAFE = [[-.2, 0], [-.5, -.4], [0, .2], [-.7, -.6], [.7, .8], [-.4, -.2]]
OLD_CONSTRAINTS = [
    "-x1 - 0.2", "x1", "-x2 - 0.5", "x2 + 0.4",
    "-x3", "x3 - 0.2", "-x4 - 0.4", "x4 + 0.6",
    "-x5 + 0.7", "x5 - 0.8", "-x6 - 0.4", "x6 + 0.2",
]


def checked_contract(*, check_model=True):
    import yaml

    config = yaml.safe_load(SOURCE_CONFIG.read_text())
    expected = {
        "num_vars": 10, "num_nn_input": 6, "num_nn_output": 3,
        "steps": 30, "step_size": 0.1, "ode_step_size": 0.05,
        "ode_order": 3, "cut_off_threshold": 1e-6,
        "remainder_estimation": [-0.01, 0.01], "sr_queue": 1000,
        "input_shape": [-1, 6], "output_T_shape": [-1, 3, 6],
        "output_c_shape": [-1, 3], "output_scale": 1, "output_offset": 0,
        "bound_opts": {"activation_bound_option": "same-slope"},
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Attitude source differs at {key}: {config.get(key)!r}")
    initials = [(row["name"], row["interval"]) for row in config["initial_set"]]
    if initials != [
        ("x1", [-.45, -.44]), ("x2", [-.55, -.54]), ("x3", [.65, .66]),
        ("x4", [-.75, -.74]), ("x5", [.85, .86]), ("x6", [-.65, -.64]),
        ("t", [0, 0]), ("u1", [0, 0]), ("u2", [0, 0]), ("u3", [0, 0]),
    ]:
        raise ValueError("Attitude initial set or state order differs")
    expr = ["".join(s.split()) for s in config["dynamics_expressions"]]
    if expr != [
        "0.25*(u1+x2*x3)", "0.5*(u2-3*x1*x3)", "u3+2*x1*x2",
        "0.5*(x2*(x4^2+x5^2+x6^2-x6)+x3*(x4^2+x5^2+x5+x6^2)+x1*(x4^2+x5^2+x6^2+1))",
        "0.5*(x1*(x4^2+x5^2+x6^2+x6)+x3*(x4^2-x4+x5^2+x6^2)+x2*(x4^2+x5^2+x6^2+1))",
        "0.5*(x1*(x4^2+x5^2-x5+x6^2)+x2*(x4^2+x4+x5^2+x6^2)+x3*(x4^2+x5^2+x6^2+1))",
        "1", "0", "0", "0",
    ]:
        raise ValueError("Attitude saved RHS differs from the fixed 2026 equations")
    if (config.get("constraints_unsafe") != OLD_CONSTRAINTS or
            config.get("constraints_safe") or config.get("constraints_target") or
            config.get("constraints_safe_from") is not None or
            config.get("constraints_safe_until") is not None):
        raise ValueError("saved Attitude property differs from the identified old checker")
    if not config["model_dir"].endswith("/Attitude-Control/attitude_control_3_64_torch.onnx"):
        raise ValueError("saved participant selected a different controller")
    boxes = json.loads(BOXES.read_text())
    if boxes != [[row[1] for row in initials]]:
        raise ValueError("Attitude one-box ledger differs from the full official initial set")
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)

    model_info = None
    if check_model:
        import onnx
        import onnx2pytorch
        import torch

        model = onnx.load(str(MODEL))
        input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
        operators = [node.op_type for node in model.graph.node]
        if input_shape != [1, 6] or output_shape != [1, 3] or operators != [
            "Gemm", "Sigmoid", "Gemm", "Sigmoid", "Gemm", "Sigmoid", "Gemm"
        ]:
            raise ValueError("selected 2026 ONNX is not the participant's 3x64 sigmoid graph")
        net = onnx2pytorch.ConvertModel(model).to(torch.float64).eval()
        center = torch.tensor([[-.445, -.545, .655, -.745, .855, -.645]], dtype=torch.float64)
        with torch.no_grad():
            output = net(center).reshape(-1)
        if output.numel() != 3 or not all(math.isfinite(float(x)) for x in output):
            raise ValueError("Attitude controller center preflight failed")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU preflight initialized CUDA")
        model_info = {"input_shape": input_shape, "output_shape": output_shape,
                      "operators": operators, "center_output": output.tolist()}
    return config, boxes, model_info


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
    started = time.perf_counter()
    result = {"status": "exception", "backend": args.backend, "mode": args.mode}
    observations = []
    try:
        import numpy as np
        import torch
        import yaml

        torch, driver, engine, cache = common.prepare(args.backend)
        config, boxes, _ = checked_contract(check_model=False)
        config["model_dir"] = str(MODEL)
        config["constraints_unsafe"][6] = "-x4 - 0.7"
        config["steps"] = 1 if args.mode == "smoke1" else 30
        config_path = output / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        cells = torch.tensor(boxes, dtype=torch.float64)
        driver.make_cells = lambda _config: cells.clone()
        driver.SR_QUEUE = 1000

        record = {
            "schema": "archcomp26-attitude-avoid-author-nohash-v1",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "backend": args.backend, "mode": args.mode,
            "source_config": str(SOURCE_CONFIG), "generated_config": str(config_path),
            "boxes": str(BOXES), "controller": str(MODEL),
            "engine": str(engine), "shared_driver": str(DRIVER), "cuda_cache": str(cache),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "property_fix": "old impossible -x4-0.4 replaced by official -x4-0.7",
            "property": "avoid official 6D unsafe box throughout T=3; smoke covers T=0.1 only",
            "method": "author sparse engine, strict plant, box same-slope CROWN, native input, rpc-float32",
            "qualification": "shared controller driver and unqualified floating NN injection; not an independent end-to-end proof",
        }
        (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
        dtype = np.dtype([("lane", "<u8"), ("step", "<u8"), ("h", "<f8"),
                          ("bounds", "<f8", (6, 4))])
        tube_union = np.array([[np.inf, -np.inf]] * 6, dtype=np.float64)
        final_endpoint = None
        original_advance = driver.advance_sparse
        with (output / "ranges.bin").open("xb") as ranges, (output / "observations.jsonl").open("x") as log:
            def observed(*call_args, **call_kwargs):
                nonlocal final_endpoint
                state, accepted = original_advance(*call_args, **call_kwargs)
                engine_obj, settings = call_args[2], call_args[4]
                tube = driver.hull_ranges_s(state, engine_obj, 6)
                endpoint_time = torch.full((1, 2), float(settings.step), dtype=torch.float64,
                                           device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(state, engine_obj, endpoint_time, 6)
                bounds = torch.cat((tube, endpoint), dim=-1).detach().cpu().numpy()
                valid = np.asarray(accepted.detach().cpu().tolist(), dtype=bool)
                if valid.any() and (not np.isfinite(bounds[valid]).all() or
                        not (bounds[valid, :, 0] <= bounds[valid, :, 1]).all() or
                        not (bounds[valid, :, 2] <= bounds[valid, :, 3]).all()):
                    raise FloatingPointError("accepted Attitude interval record invalid")
                if valid.any():
                    tube_union[:, 0] = np.minimum(tube_union[:, 0], bounds[valid, :, 0].min(axis=0))
                    tube_union[:, 1] = np.maximum(tube_union[:, 1], bounds[valid, :, 1].max(axis=0))
                    final_endpoint = [
                        [float(bounds[valid, i, 2].min()), float(bounds[valid, i, 3].max())]
                        for i in range(6)
                    ]
                step_number = len(observations) + 1
                rows = np.zeros(1, dtype=dtype)
                rows["lane"], rows["step"], rows["h"], rows["bounds"] = (
                    0, step_number, float(settings.step), bounds)
                ranges.write(rows.tobytes())
                ranges.flush()
                disjoint = bool(valid.any() and any(
                    bounds[0, i, 1] < lo or bounds[0, i, 0] > hi
                    for i, (lo, hi) in enumerate(UNSAFE)))
                row = {"substep": step_number, "accepted_count": int(valid.sum()),
                       "tube_box_disjoint_official_unsafe": disjoint}
                log.write(json.dumps(row) + "\n")
                log.flush()
                observations.append(row)
                return state, accepted

            driver.advance_sparse = observed
            argv = [str(DRIVER), str(config_path), "--device", "cuda:0", "--engine", "sparse",
                    "--strict", "--crown-domain", "box", "--crown-relax", "same-slope",
                    "--crown-transport", "rpc-float32", "--crown-input-layout", "native",
                    "--nn-mode", "crown", "--print-final-hull",
                    "--metrics-json", str(output / "metrics.json")]
            record["driver_argv"] = argv
            (output / "START.json").write_text(json.dumps(record, indent=2) + "\n")
            previous = sys.argv
            sys.argv = argv
            tee = Tee(sys.stdout)
            try:
                with redirect_stdout(tee):
                    driver_code = driver.main()
            finally:
                sys.argv = previous

        expected = config["steps"] * 2
        complete = (driver_code == 0 and len(observations) == expected
                    and all(row["accepted_count"] == 1 for row in observations))
        if any(line in tee.property_lines for line in ("Unsafe", "Unsafe.", "FALSIFIED")):
            checker = "UNSAFE_OR_FALSIFIED"
        elif any(line in tee.property_lines for line in ("Unknown", "Unknown.", "UNKNOWN", "Flow* terminated.")):
            checker = "UNKNOWN"
        elif complete and all(row["tube_box_disjoint_official_unsafe"] for row in observations):
            checker = "VERIFIED_BY_AUTHOR_CHECKER_SILENCE_AND_SAVED_BOX_DISJOINTNESS"
        else:
            checker = "UNRESOLVED"
        result.update(
            status=("completed_short_prefix" if args.mode == "smoke1" else "completed")
            if complete else "incomplete",
            driver_return=driver_code, expected_substeps=expected,
            observed_substeps=len(observations),
            accepted_lane_substeps=sum(row["accepted_count"] for row in observations),
            all_lanes_accepted=complete,
            saved_tubes_box_disjoint_official_unsafe=complete and all(
                row["tube_box_disjoint_official_unsafe"] for row in observations),
            author_checker_lines=tee.property_lines, author_checker_interpretation=checker,
            observed_prefix_tube_union=tube_union.tolist() if final_endpoint else None,
            full_time_tube_union=tube_union.tolist() if complete and args.mode == "full" else None,
            last_observed_endpoint_union=final_endpoint,
            terminal_endpoint_union=final_endpoint if complete and args.mode == "full" else None,
            terminal_endpoint_union_width=([hi - lo for lo, hi in final_endpoint]
                                           if complete and args.mode == "full" else None),
            driver_elapsed_s=(json.loads((output / "metrics.json").read_text()).get("elapsed_s")
                              if (output / "metrics.json").is_file() else None),
            wall_s=time.perf_counter() - started,
            end_to_end_floating_point_nn_certificate=False,
        )
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(),
                      observed_substeps=len(observations), wall_s=time.perf_counter() - started)
        raise
    finally:
        (output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["status"].startswith("completed") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--backend", choices=common.ENGINES)
    parser.add_argument("--mode", choices=("smoke1", "full"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common.guard_digests_and_builds()
    if args.preflight:
        config, boxes, model = checked_contract()
        report = {"schema": "archcomp26-attitude-avoid-cpu-preflight-nohash-v1",
                  "checked_utc": datetime.now(timezone.utc).isoformat(),
                  "source_config": str(SOURCE_CONFIG), "boxes": str(BOXES),
                  "selected_controller": str(MODEL),
                  "state_order": [row["name"] for row in config["initial_set"]],
                  "box_count": len(boxes), "controller": model,
                  "property_fix": "x4 lower bound -0.7, all-time avoid official box",
                  "gpu_initialized": False, "content_digest_computed": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as out:
            json.dump(report, out, indent=2)
            out.write("\n")
        print(f"CPU-only Attitude preflight passed: {args.output}")
        return 0
    if args.backend is None or args.mode is None:
        parser.error("GPU run requires --backend and --mode")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
