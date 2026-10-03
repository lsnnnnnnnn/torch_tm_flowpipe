#!/usr/bin/env python3
"""Xiangru full-box TORA reach-sigmoid official-u11 numerical horizon.

This records 10 periods to 5 s. It does not evaluate the reach target.
"""

from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
PROFILE = "later02_xiangru"
EXPECTED_BOX = [
    [-0.77, -0.75], [-0.45, -0.43], [0.51, 0.54],
    [-0.3, -0.28], [0.0, 0.0], [0.0, 0.0],
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight():
    import numpy as np
    import onnx
    import onnx2pytorch
    import torch
    import yaml
    from onnx import numpy_helper

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    expected = {
        "num_vars": 6, "num_nn_input": 4, "num_nn_output": 1,
        "steps": 10, "step_size": 0.5, "ode_step_size": 0.01,
        "ode_order": 6, "output_scale": 1, "output_offset": 0,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"diagnostic config {key} differs: {cfg.get(key)!r}")
    if [entry["name"] for entry in cfg["initial_set"]] != [
        "x1", "x2", "x3", "x4", "t", "u1"
    ] or [entry["interval"] for entry in cfg["initial_set"]] != EXPECTED_BOX:
        raise ValueError("full initial box or state order differs")
    if any(entry.get("splits") for entry in cfg["initial_set"]):
        raise ValueError("initial set unexpectedly split")
    if ["".join(expr.split()) for expr in cfg["dynamics_expressions"]] != [
        "x2", "-x1+0.1*sin(x3)", "x4", "u1", "1", "0"
    ]:
        raise ValueError("plant dynamics differs")
    if cfg.get("constraints_target") or cfg.get("constraints_safe") or cfg.get("constraints_unsafe"):
        raise ValueError("numerical horizon run must not evaluate a property")
    if Path(cfg["model_dir"]) != ROOT / "controller_plant_u.onnx":
        raise ValueError("config model path differs")

    receipt = json.loads((ROOT / "controller_plant_u.onnx.json").read_text())
    if (receipt["activations"] != ["sigmoid"] * 4 or
            receipt["scale"] != 11 or receipt["offset"] != 0 or
            receipt["model_path"] != str(ROOT / "controller_plant_u.onnx") or
            Path(receipt["mat_path"]) != ROOT / "nn_tora_sigmoid.mat"):
        raise ValueError("explicit controller export receipt differs")
    model = onnx.load(str(ROOT / "controller_plant_u.onnx"))
    ops = [node.op_type for node in model.graph.node]
    if ops != ["Gemm", "Sigmoid"] * 4 + ["Mul", "Add"]:
        raise ValueError(f"controller activation/affine graph differs: {ops}")
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if (not np.array_equal(initializers["control_scale"], np.array([11.0])) or
            not np.array_equal(initializers["control_offset"], np.array([0.0]))):
        raise ValueError("internal plant scaling differs")
    net = onnx2pytorch.ConvertModel(model, experimental=False).to(torch.float64).eval()
    center = torch.tensor([[-0.76, -0.44, 0.525, -0.29]], dtype=torch.float64)
    with torch.no_grad():
        center_u = float(net(center).reshape(-1)[0])
    if (not math.isfinite(center_u) or
            abs(center_u - receipt["forward"]["center_onnx"]) > 1e-12 or
            torch.cuda.is_initialized()):
        raise ValueError("CPU controller point preflight differs")
    return {
        "config": str(ROOT / "config.yaml"),
        "controller": str(ROOT / "controller_plant_u.onnx"),
        "mat": str(ROOT / "nn_tora_sigmoid.mat"),
        "state_order": ["x1", "x2", "x3", "x4", "t", "u1"],
        "initial_box": EXPECTED_BOX,
        "control_period_s": 0.5,
        "control_periods": 10,
        "ode_substeps": 500,
        "controller_ops": ops,
        "internal_control": "u = 11 * f(x) + 0",
        "external_control": "output_scale = 1; output_offset = 0; dx4 = u1",
        "center_plant_u": center_u,
        "property_evaluated": False,
        "gpu_initialized": False,
    }


def main():
    import numpy as np
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("this isolated run reserves physical GPU 2")
    support = load_module("tora_author_support", ROOT / "author_support.py")
    support.guard_digests_and_builds()
    started = time.perf_counter()
    observed = []
    result = {"profile": PROFILE, "method": "xiangru", "status": "exception"}
    try:
        checked = preflight()
        (ROOT / "PREFLIGHT.json").write_text(json.dumps(checked, indent=2) + "\n")
        start = {
            "profile": PROFILE,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "method": "xiangru plant engine + shared author CROWN driver",
            "gpu_physical": 2,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "contract": checked,
            "scope": "one full initial box; ten 0.5 s control periods; no target verdict",
            "qualification": "numerical flowpipe only; no end-to-end floating NNCS certificate",
        }
        (ROOT / "START.json").write_text(json.dumps(start, indent=2) + "\n")

        torch, driver, _, _ = support.prepare("xiangru")
        box = torch.tensor([EXPECTED_BOX], dtype=torch.float64)
        driver.make_cells = lambda _cfg: box.clone()
        driver.SR_QUEUE = 1000
        original_advance = driver.advance_sparse
        dtype = np.dtype([("lane", "<u8"), ("step", "<u8"), ("h", "<f8"),
                          ("bounds", "<f8", (4, 4))])

        with (ROOT / "ranges.bin").open("xb") as ranges, \
                (ROOT / "observations.jsonl").open("x") as log:
            def observed_advance(*args, **kwargs):
                state, accepted = original_advance(*args, **kwargs)
                number = len(observed) + 1
                valid = bool(accepted.reshape(-1)[0].item())
                row = {"substep": number, "accepted": valid}
                if not valid:
                    observed.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    raise ArithmeticError(f"first numerical rejection at substep {number}")
                engine, settings = args[2], args[4]
                tube = driver.hull_ranges_s(state, engine, 4)
                endpoint_time = torch.full((1, 2), float(settings.step),
                                           dtype=torch.float64, device=state.pre.device)
                endpoint = driver.rows_range_over_time_sparse(
                    state, engine, endpoint_time, 4
                )
                bounds = torch.cat((tube, endpoint), dim=-1).detach().cpu().numpy()
                if (not np.isfinite(bounds).all() or
                        not (bounds[:, :, 0] <= bounds[:, :, 1]).all() or
                        not (bounds[:, :, 2] <= bounds[:, :, 3]).all()):
                    row["interval_valid"] = False
                    observed.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    raise ArithmeticError(f"first invalid interval at substep {number}")
                item = np.empty(1, dtype=dtype)
                item["lane"] = 0
                item["step"] = number
                item["h"] = float(settings.step)
                item["bounds"] = bounds
                ranges.write(item.tobytes())
                ranges.flush()
                row["interval_valid"] = True
                observed.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                return state, accepted

            driver.advance_sparse = observed_advance
            argv = [str(support.DRIVER), str(ROOT / "config.yaml"),
                    "--device", "cuda:0", "--engine", "sparse", "--strict",
                    "--crown-domain", "box", "--crown-relax", "same-slope",
                    "--crown-transport", "rpc-float32", "--crown-input-layout", "native",
                    "--nn-mode", "crown", "--print-final-hull",
                    "--metrics-json", str(ROOT / "metrics.json")]
            start["driver_argv"] = argv
            (ROOT / "START.json").write_text(json.dumps(start, indent=2) + "\n")
            saved_argv = sys.argv
            try:
                sys.argv = argv
                code = driver.main()
            finally:
                sys.argv = saved_argv

        if code != 0 or len(observed) != 500 or not all(x["accepted"] for x in observed):
            raise RuntimeError("driver did not complete all 500 accepted substeps")
        result.update(status="completed_full_numerical_horizon", driver_return=code,
                      observed_substeps=500, accepted_substeps=500,
                      property_evaluated=False, full_horizon_completed=True)
    except BaseException as error:
        result.update(status="early_stop" if isinstance(error, ArithmeticError) else "exception",
                      error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), observed_substeps=len(observed),
                      accepted_substeps=sum(bool(x.get("accepted")) for x in observed))
        raise
    finally:
        result["wall_s"] = time.perf_counter() - started
        (ROOT / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
