#!/usr/bin/env python3
"""One full-box Airplane paper-Euler-controller-first interval diagnostic.

This is an interval-only entry check, not a four-method benchmark row or a
certificate for the floating-point ONNX implementation.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
ENGINE = ROOT / "engine/src"
ASSETS = ROOT / "runs/archcomp26_20261001/airplane_prep_001"
EXPECTED_DIMS = (12, 100, 100, 20, 6)
EXPECTED_OPS = ("MatMul", "Add", "Relu") * 3 + ("MatMul", "Add")
INITIAL = [[0.0, 1.0] if i in (3, 4, 5, 6, 7, 8) else [0.0, 0.0]
           for i in range(12)]
STATE_NAMES = ("sx", "sy", "sz", "vx", "vy", "vz", "phi", "theta", "psi", "r", "p", "q")


def load_layers(model_path, torch):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path))
    graph = model.graph
    nodes = list(graph.node)
    if tuple(node.op_type for node in nodes) != EXPECTED_OPS:
        raise ValueError("fixed Airplane ONNX operator order changed")
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("fixed Airplane ONNX interface count changed")
    if graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("Airplane NN input is not FLOAT")
    if graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("Airplane NN output is not FLOAT")
    if graph.input[0].type.tensor_type.shape.dim[-1].dim_value != 12:
        raise ValueError("Airplane NN input width changed")
    if graph.output[0].type.tensor_type.shape.dim[-1].dim_value != 6:
        raise ValueError("Airplane NN output width changed")
    params = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    previous = graph.input[0].name
    layers = []
    for layer in range(4):
        matmul, add = nodes[3 * layer:3 * layer + 2]
        if (matmul.input[0] != previous or matmul.input[1] not in params
                or add.input[0] != matmul.output[0] or add.input[1] not in params):
            raise ValueError("Airplane ONNX graph wiring changed")
        weight, bias = params[matmul.input[1]], params[add.input[1]]
        if (weight.dtype != np.float32 or bias.dtype != np.float32
                or weight.shape != (EXPECTED_DIMS[layer], EXPECTED_DIMS[layer + 1])
                or bias.shape != (EXPECTED_DIMS[layer + 1],)):
            raise ValueError("Airplane ONNX weight type or shape changed")
        layers.append((torch.from_numpy(weight.astype(np.float64)),
                       torch.from_numpy(bias.astype(np.float64))))
        if layer < 3:
            relu = nodes[3 * layer + 2]
            if list(relu.input) != [add.output[0]]:
                raise ValueError("Airplane ONNX ReLU wiring changed")
            previous = relu.output[0]
        else:
            previous = add.output[0]
    if previous != graph.output[0].name:
        raise ValueError("Airplane ONNX output wiring changed")
    return layers


def network_interval(state, layers, iv, torch):
    values = state.unsqueeze(0)
    for layer, (weight, bias) in enumerate(layers):
        values = iv.add(iv.dot_point_iv(weight.T.unsqueeze(0), values.unsqueeze(1), dim=-1),
                        iv.from_point(bias).unsqueeze(0))
        if layer < 3:
            values = torch.maximum(values, torch.zeros_like(values))
        iv.assert_valid(values, f"Airplane NN layer {layer + 1}")
    return values[0]


def euler_derivative(x, control, iv, tr):
    add, sub, mul = iv.add, iv.sub, iv.mul
    _, _, _, vx, vy, vz, phi, theta, psi, r, p, q = x.unbind(0)
    fx, fy, fz, mx, my, mz = control.unbind(0)
    sphi, cphi = tr.sin_iv(phi), tr.cos_iv(phi)
    sth, cth = tr.sin_iv(theta), tr.cos_iv(theta)
    spsi, cpsi = tr.sin_iv(psi), tr.cos_iv(psi)
    if float(cth[0]) <= 0:
        raise ArithmeticError("cos(theta) interval does not exclude zero")
    reciprocal, bad = iv.rec(cth)
    if bool(bad):
        raise ArithmeticError("cos(theta) reciprocal is undefined")

    # T_psi * T_theta * T_phi * [vx,vy,vz], evaluated on the old state.
    phi_y = sub(mul(cphi, vy), mul(sphi, vz))
    phi_z = add(mul(sphi, vy), mul(cphi, vz))
    theta_x = add(mul(cth, vx), mul(sth, phi_z))
    theta_z = sub(mul(cth, phi_z), mul(sth, vx))
    dx = sub(mul(cpsi, theta_x), mul(spsi, phi_y))
    dy = add(mul(spsi, theta_x), mul(cpsi, phi_y))
    dz = theta_z

    angular_mix = add(mul(sphi, q), mul(cphi, r))
    dphi = add(p, mul(mul(sth, reciprocal), angular_mix))
    dtheta = sub(mul(cphi, q), mul(sphi, r))
    dpsi = mul(reciprocal, angular_mix)
    dvx = add(sub(sub(fx, sth), mul(q, vz)), mul(r, vy))
    dvy = add(add(add(mul(cth, sphi), fy), iv.neg(mul(r, vx))), mul(p, vz))
    dvz = add(add(add(mul(cth, cphi), fz), iv.neg(mul(p, vy))), mul(q, vx))
    import torch
    return torch.stack((dx, dy, dz, dvx, dvy, dvz,
                        dphi, dtheta, dpsi, mz, mx, my))


def run():
    import torch
    sys.path.insert(0, str(ENGINE))
    from flowstar_gpu import interval as iv, transcendental as tr

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    started = perf_counter()
    layers = load_layers(ASSETS / "controller_airplane.onnx", torch)
    state = torch.tensor(INITIAL, dtype=torch.float64)
    iv.assert_valid(state, "Airplane full initial box")
    control = network_interval(state, layers, iv, torch)
    derivative = euler_derivative(state, control, iv, tr)
    iv.assert_valid(derivative, "Airplane Euler derivative")
    # The exact paper step 1/10 lies inside this two-neighbor float64 interval.
    dt = torch.tensor([math.nextafter(0.1, -math.inf),
                       math.nextafter(0.1, math.inf)], dtype=torch.float64)
    endpoint = iv.add(state, iv.mul(dt, derivative))
    iv.assert_valid(endpoint, "Airplane first Euler endpoint")
    safety = {STATE_NAMES[i]: bool(endpoint[i, 0] >= -1 and endpoint[i, 1] <= 1)
              for i in (1, 6, 7, 8)}
    # A programming sanity check on the bounds at three points of the initial box.
    point_checks = 0
    for alpha in (0.0, 0.5, 1.0):
        point = state[:, 0] + alpha * (state[:, 1] - state[:, 0])
        value = point.unsqueeze(0)
        for layer, (weight, bias) in enumerate(layers):
            value = value @ weight + bias
            if layer < 3:
                value = torch.relu(value)
        if not bool(((control[:, 0] <= value[0]) & (value[0] <= control[:, 1])).all()):
            raise AssertionError("point controller output escaped interval bound")
        point_checks += 6
    return {
        "schema": "archcomp26-airplane-paper-euler-controller-first-interval-smoke-nohash-v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "airplane-discrete/paper-Euler-controller-first",
        "method": "CPU directed interval arithmetic on fixed ONNX and MATLAB equations; no P3/Huan/Xiangru/native solver",
        "scope": "one full 12D initial box, one synchronous forward-Euler transfer x0->x1",
        "status": "first_transfer_accepted" if all(safety.values()) else "first_transfer_property_unknown",
        "qualification": "diagnostic only; no four-method benchmark row, full 20-transfer claim, or independently established floating-point certificate",
        "source": {"model": str(ASSETS / "controller_airplane.onnx"),
                   "dynamics": str(ASSETS / "dynamics.m"),
                   "specification": str(ASSETS / "specifications.txt"),
                   "interval_engine": str(ENGINE / "flowstar_gpu/interval.py"),
                   "transcendental_engine": str(ENGINE / "flowstar_gpu/transcendental.py")},
        "state_order": STATE_NAMES, "control_order": ("Fx", "Fy", "Fz", "Mx", "My", "Mz"),
        "step_exact_rational": "1/10", "step_float64_enclosure": dt.tolist(),
        "initial": state.tolist(), "control_interval": control.tolist(),
        "derivative_interval": derivative.tolist(), "endpoint_k1": endpoint.tolist(),
        "checked_indices": [0, 1], "property_k0_inside": True,
        "property_k1_coordinate_inside": safety,
        "cos_theta_k0": tr.cos_iv(state[7]).tolist(),
        "controller_point_values_checked": point_checks,
        "wall_s": perf_counter() - started,
        "torch_version": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run()
    except Exception as exc:
        result = {"schema": "archcomp26-airplane-paper-euler-controller-first-interval-smoke-nohash-v1",
                  "recorded_utc": datetime.now(timezone.utc).isoformat(),
                  "profile": "airplane-discrete/paper-Euler-controller-first",
                  "status": "failed_before_first_transfer", "error": repr(exc)}
        (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"], result["wall_s"])


if __name__ == "__main__":
    main()
