#!/usr/bin/env python3
"""Numerical high-corner Airplane replay; candidate only, no set proof."""

from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import scipy
from scipy.integrate import solve_ivp

N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
MODEL = N / "runs/archcomp26_20261001/airplane_prep_001/controller_airplane.onnx"
RPC = N / "runs/archcomp26_20261001/native_airplane_binary6_coverage_gate_20261002_008/cells/111111/controller_rpc.jsonl"
OUT = N / "runs/archcomp26_20261001/native_airplane_highcorner_point_replay_20261002_009"
INITIAL = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.float64)
NAMES = ("x", "y", "z", "u", "v", "w", "phi", "theta", "psi", "r", "p", "q")


def utc():
    return datetime.now(timezone.utc).isoformat()


def controller(model, state):
    graph = model.graph
    ops = [node.op_type for node in graph.node]
    assert ops == ["MatMul", "Add", "Relu", "MatMul", "Add", "Relu",
                   "MatMul", "Add", "Relu", "MatMul", "Add"]
    assert len(graph.input) == len(graph.output) == 1
    tensors = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    tensors[graph.input[0].name] = state.astype(np.float32).reshape(1, 12)
    for node in graph.node:
        args = [tensors[name] for name in node.input]
        if node.op_type == "MatMul":
            value = args[0] @ args[1]
        elif node.op_type == "Add":
            value = args[0] + args[1]
        else:
            value = np.maximum(args[0], np.float32(0))
        assert value.dtype == np.float32 and np.isfinite(value).all()
        tensors[node.output[0]] = value
    output = tensors[graph.output[0].name]
    assert output.shape == (1, 6)
    return output[0].astype(np.float64), ops


def plant(_, s, c):
    x, y, z, u, v, w, phi, theta, psi, r, p, q = s
    Fx, Fy, Fz, Mx, My, Mz = c
    sp, cp = math.sin(phi), math.cos(phi)
    st, ct = math.sin(theta), math.cos(theta)
    ss, cs = math.sin(psi), math.cos(psi)
    dx = cs * ct * u + (-ss * cp + cs * st * sp) * v + (ss * sp + cs * st * cp) * w
    dy = ss * ct * u + (cs * cp + ss * st * sp) * v + (-cs * sp + ss * st * cp) * w
    dz = -st * u + ct * sp * v + ct * cp * w
    du = -st + Fx - q * w + r * v
    dv = ct * sp + Fy - r * u + p * w
    dw = ct * cp + Fz - p * v + q * u
    dphi = (ct * p + st * sp * q + st * cp * r) / ct
    dtheta = cp * q - sp * r
    dpsi = (sp * q + cp * r) / ct
    return [dx, dy, dz, du, dv, dw, dphi, dtheta, dpsi, Mz, Mx, My]


def main():
    if OUT.exists():
        raise RuntimeError(f"independent run identity already exists: {OUT}")
    if not MODEL.is_file() or not RPC.is_file():
        raise FileNotFoundError((MODEL, RPC))
    OUT.mkdir()
    (OUT / "START.json").write_text(json.dumps({
        "started_utc": utc(), "model": str(MODEL), "saved_relaxation_rpc": str(RPC),
        "initial_state": INITIAL.tolist(), "horizon_s": 0.01,
        "purpose": "one true-network numerical point replay; candidate only"
    }, indent=2) + "\n")
    model = onnx.load(str(MODEL))
    onnx.checker.check_model(model)
    control, ops = controller(model, INITIAL)
    rpc = json.loads(RPC.read_text().splitlines()[0])
    response = rpc["response"]
    center = np.asarray(response["T"], dtype=np.float64) @ INITIAL
    affine_lo = center + np.asarray(response["u_min"], dtype=np.float64)
    affine_hi = center + np.asarray(response["u_max"], dtype=np.float64)
    inside_relaxation = bool(np.all(affine_lo <= control) and np.all(control <= affine_hi))
    if not inside_relaxation:
        raise RuntimeError("point ONNX output is outside saved CROWN affine relaxation")
    times = np.linspace(0, 0.01, 11)
    solutions = {}
    for method in ("DOP853", "Radau"):
        sol = solve_ivp(lambda t, s: plant(t, s, control), (0, 0.01), INITIAL,
                        method=method, rtol=1e-12, atol=1e-14,
                        max_step=0.0001, t_eval=times)
        if not sol.success or sol.y.shape != (12, len(times)) or not np.isfinite(sol.y).all():
            raise RuntimeError(f"{method} integration failed: {sol.message}")
        solutions[method] = sol.y
        with (OUT / f"trajectory_{method}.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("t", *NAMES))
            writer.writerows((float(t), *[float(v) for v in sol.y[:, i]])
                             for i, t in enumerate(times))
    gap = float(np.max(np.abs(solutions["DOP853"] - solutions["Radau"])))
    sampled = solutions["DOP853"]
    above = {name: float(sampled[i].max()) for name, i in
             (("phi", 6), ("theta", 7), ("psi", 8)) if sampled[i].max() > 1}
    result = {"ended_utc": utc(), "status": "numerical_point_replay_completed",
              "scope": "one initial corner, one held true ONNX output, t in [0,0.01]",
              "model_path": str(MODEL), "onnx_input": model.graph.input[0].name,
              "onnx_output": model.graph.output[0].name, "onnx_ops": ops,
              "controller_output_Fx_Fy_Fz_Mx_My_Mz": control.tolist(),
              "inside_saved_CROWN_affine_interval": inside_relaxation,
              "CROWN_affine_lower_at_point": affine_lo.tolist(),
              "CROWN_affine_upper_at_point": affine_hi.tolist(),
              "integrators": ["DOP853", "Radau"], "rtol": 1e-12,
              "atol": 1e-14, "max_step_s": 0.0001,
              "max_state_gap_between_integrators": gap,
              "DOP853_end_state": sampled[:, -1].tolist(),
              "DOP853_sampled_angle_maxima_above_1": above,
              "candidate_only": True, "certified_unsafe_trajectory": False,
              "numpy_version": np.__version__, "onnx_version": onnx.__version__,
              "scipy_version": scipy.__version__}
    (OUT / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"control": control.tolist(), "angle_sample_maxima_above_1": above,
                      "max_method_gap": gap, "run": str(OUT)}))


if __name__ == "__main__":
    if sys.argv[1:] != ["once"]:
        raise SystemExit("usage: replay.py once")
    main()
