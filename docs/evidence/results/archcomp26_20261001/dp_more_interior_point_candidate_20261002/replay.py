#!/usr/bin/env python3
"""Numerical strict-interior point replay for the fixed 2026 DP more ONNX.

This checks a candidate initial point, not an interval proof or NN certificate.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from scipy.integrate import solve_ivp


def load_controller(path):
    graph = onnx.load(str(path)).graph
    nodes = list(graph.node)
    assert [node.op_type for node in nodes] == [
        "MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add"
    ]
    assert len(graph.input) == len(graph.output) == 1
    weights = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    assert all(value.dtype == np.float32 for value in weights.values())

    def controller(state):
        values = dict(weights)
        values[graph.input[0].name] = np.asarray(state, dtype=np.float32).reshape(1, 4)
        for node in nodes:
            inputs = [values[name] for name in node.input]
            if node.op_type == "MatMul":
                output = inputs[0] @ inputs[1]
            elif node.op_type == "Add":
                output = inputs[0] + inputs[1]
            else:
                output = np.maximum(inputs[0], np.float32(0))
            values[node.output[0]] = output
        result = values[graph.output[0].name]
        assert result.shape == (1, 2) and result.dtype == np.float32
        return result[0].astype(np.float64)

    return controller


def rhs(state, control):
    th1, th2, v1, v2 = state
    t1, t2 = control
    delta = th1 - th2
    s, c = math.sin(delta), math.cos(delta)
    a = 4 * t1 + 2 * math.sin(th1) - v2 * v2 * s / 2
    b = v1 * v1 * s + 8 * t2 + 2 * math.sin(th2) - c * (-v2 * v2 * s / 2 + 4 * t1 + 2 * math.sin(th1))
    den = c * c / 2 - 1
    return np.array([v1, v2, a + c * b / (2 * den), -b / den], dtype=np.float64)


def rk4_step(state, control, h):
    k1 = rhs(state, control)
    k2 = rhs(state + h * k1 / 2, control)
    k3 = rhs(state + h * k2 / 2, control)
    k4 = rhs(state + h * k3, control)
    return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def run_rk4(controller, initial, substeps_per_period):
    state = initial.copy()
    endpoints = [state.copy()]
    controls = []
    h = 0.02 / substeps_per_period
    for _ in range(20):
        control = controller(state)
        controls.append(control.copy())
        for _ in range(substeps_per_period):
            state = rk4_step(state, control, h)
        endpoints.append(state.copy())
    return np.array(endpoints), np.array(controls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    controller = load_controller(args.model)
    initial = np.array([1.299, 1.299, 1.299, 1.299], dtype=np.float64)
    assert np.all((initial > 1.0) & (initial < 1.3))
    state = initial.copy()
    samples = [[0.0, *state]]
    period_records = []
    crossing_events = []
    for period in range(20):
        start, end = period * 0.02, (period + 1) * 0.02
        control = controller(state)

        def downward_v1_crossing(_time, y):
            return y[2] + 1.5

        downward_v1_crossing.direction = -1
        downward_v1_crossing.terminal = False
        target_times = start + np.arange(1, 5, dtype=np.float64) * 0.005
        target_times[-1] = end
        solution = solve_ivp(
            lambda _time, y: rhs(y, control), (start, end), state,
            method="DOP853", rtol=1e-12, atol=1e-14, max_step=0.001,
            t_eval=target_times, events=downward_v1_crossing,
        )
        assert solution.success and solution.y.shape == (4, 4)
        for time, value in zip(solution.t, solution.y.T):
            samples.append([float(time), *value.tolist()])
        crossing_events.extend(float(time) for time in solution.t_events[0])
        state = solution.y[:, -1]
        period_records.append({
            "period": period, "start_t": start, "end_t": end,
            "control_float32": control.tolist(), "end_state": state.tolist(),
            "nfev": solution.nfev,
        })

    points = np.asarray(samples)
    rk4_100, controls_100 = run_rk4(controller, initial, 200)
    rk4_50, controls_50 = run_rk4(controller, initial, 400)
    dop_endpoints = points[::4, 1:]
    assert dop_endpoints.shape == (21, 4)
    violations = np.argwhere(np.abs(points[:, 1:]) > 1.5)
    first_grid_violation = None if len(violations) == 0 else {
        "time": float(points[violations[0, 0], 0]),
        "coordinate": int(violations[0, 1]),
        "value": float(points[violations[0, 0], violations[0, 1] + 1]),
    }
    result = {
        "schema": "archcomp26-dp-more-numeric-interior-point-candidate-v1",
        "qualification": "numerical point-trajectory candidate only; no interval proof or floating-point NN certificate",
        "model": str(args.model), "model_ops": ["MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add"],
        "model_arithmetic": "NumPy float32 per ONNX operator; output held as float64 over each 0.02 s plant period",
        "state_order": ["theta1", "theta2", "theta1_dot", "theta2_dot"],
        "initial_state": initial.tolist(), "strictly_inside_initial_box": True,
        "periods": 20, "period_s": 0.02,
        "ode": "official 2026 continuous double-pendulum dynamics, m=L=0.5,c=0,g=1",
        "dop853": {"rtol": 1e-12, "atol": 1e-14, "max_step": 0.001},
        "rk4_step_s": [0.0001, 0.00005],
        "max_abs_endpoint_gap_dop853_vs_rk4_0p0001": float(np.max(np.abs(dop_endpoints - rk4_100))),
        "max_abs_endpoint_gap_rk4_0p0001_vs_0p00005": float(np.max(np.abs(rk4_100 - rk4_50))),
        "max_abs_control_gap_dop853_vs_rk4_0p0001": float(np.max(np.abs(np.array([row["control_float32"] for row in period_records]) - controls_100))),
        "max_abs_control_gap_rk4_0p0001_vs_0p00005": float(np.max(np.abs(controls_100 - controls_50))),
        "first_0p005_grid_violation": first_grid_violation,
        "theta1_dot_minus1p5_downward_event_times": crossing_events,
        "state_at_t_0p36": points[72, 1:].tolist(),
        "state_at_t_0p4": points[80, 1:].tolist(),
        "period_records": period_records,
    }
    (args.out / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.out / "trajectory_0p005.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["t", "theta1", "theta2", "theta1_dot", "theta2_dot"])
        writer.writerows(samples)
    print(json.dumps({key: result[key] for key in (
        "first_0p005_grid_violation", "theta1_dot_minus1p5_downward_event_times",
        "state_at_t_0p36", "max_abs_endpoint_gap_dop853_vs_rk4_0p0001",
        "max_abs_endpoint_gap_rk4_0p0001_vs_0p00005")}, indent=2))


if __name__ == "__main__":
    main()
