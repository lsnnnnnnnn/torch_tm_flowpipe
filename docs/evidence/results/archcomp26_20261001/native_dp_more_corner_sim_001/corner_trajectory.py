"""Numerical corner trajectory for the official DP more ONNX; candidate only."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import scipy
from scipy.integrate import solve_ivp


PERIOD = 0.02
PERIODS = 20
INITIAL = [1.3, 1.3, 1.3, 1.3]
RTOL = 1e-12
ATOL = 1e-14
MAX_STEP = 0.0002


def controller(model, state):
    graph = model.graph
    assert len(graph.input) == len(graph.output) == 1
    tensors = {a.name: numpy_helper.to_array(a) for a in graph.initializer}
    tensors[graph.input[0].name] = np.asarray(state, dtype=np.float32).reshape(1, 4)
    for node in graph.node:
        if node.op_type == "MatMul":
            value = tensors[node.input[0]] @ tensors[node.input[1]]
        elif node.op_type == "Add":
            value = tensors[node.input[0]] + tensors[node.input[1]]
        elif node.op_type == "Relu":
            value = np.maximum(tensors[node.input[0]], np.float32(0))
        else:
            raise ValueError(f"unsupported ONNX operation {node.op_type}")
        assert value.dtype == np.float32
        tensors[node.output[0]] = value
    output = tensors[graph.output[0].name]
    assert output.shape == (1, 2) and np.isfinite(output).all()
    return output[0].astype(np.float64)


def plant(_, state, control):
    th1, th2, v1, v2 = state
    t1, t2 = control
    delta = th1 - th2
    sine, cosine = math.sin(delta), math.cos(delta)
    a = 4*t1 + 2*math.sin(th1) - v2*v2*sine/2
    b = v1*v1*sine + 8*t2 + 2*math.sin(th2) - cosine*a
    denominator = cosine*cosine/2 - 1
    return (v1, v2, a + cosine*b/(2*denominator), -b/denominator)


def event_for(index, side):
    def event(_, state):
        return state[index] + 1.5 if side == "lower" else 1.5 - state[index]
    event.direction = -1
    event.terminal = False
    return event


EVENTS = [event_for(i, side) for i in range(4) for side in ("lower", "upper")]
EVENT_NAMES = [f"{name}_{side}" for name in
               ("theta1", "theta2", "theta1_dot", "theta2_dot")
               for side in ("lower", "upper")]


def simulate(model, method):
    state = np.array(INITIAL, dtype=np.float64)
    controls, boundaries, samples, crossings = [], [], [], []
    extrema_min = np.array(INITIAL, dtype=np.float64)
    extrema_max = extrema_min.copy()
    for k in range(PERIODS):
        t0, t1 = k*PERIOD, (k+1)*PERIOD
        control = controller(model, state)
        controls.append({"period": k, "t": t0, "state": state.tolist(),
                         "control": control.tolist()})
        sol = solve_ivp(lambda t, y: plant(t, y, control), (t0, t1), state,
                        method=method, rtol=RTOL, atol=ATOL, max_step=MAX_STEP,
                        dense_output=True, events=EVENTS)
        if not sol.success:
            raise RuntimeError(f"period {k}: {sol.message}")
        for i, times in enumerate(sol.t_events):
            for t in times:
                crossings.append({"time": float(t), "state": sol.sol(t).tolist(),
                                  "coordinate_side": EVENT_NAMES[i],
                                  "period": k, "control": control.tolist()})
        grid = np.linspace(t0, t1, 101)
        values = sol.sol(grid)
        extrema_min = np.minimum(extrema_min, values.min(axis=1))
        extrema_max = np.maximum(extrema_max, values.max(axis=1))
        for j, t in enumerate(grid):
            if k and j == 0:
                continue
            samples.append([float(t), k, *values[:, j].tolist()])
        state = sol.y[:, -1]
        boundaries.append({"period_end": k+1, "t": t1, "state": state.tolist()})
    crossings.sort(key=lambda item: item["time"])
    first = crossings[0] if crossings else None
    if first:
        k = first["period"]
        later = next((row for row in samples if row[0] > first["time"] and row[0] <= (k+1)*PERIOD), None)
        first["first_later_sample"] = (None if later is None else
                                       {"time": later[0], "state": later[2:]})
    return {"method": method, "controls": controls, "boundaries": boundaries,
            "samples": samples, "crossings": crossings,
            "first_crossing": first, "sampled_min": extrema_min.tolist(),
            "sampled_max": extrema_max.tolist(), "final_state": state.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    model = onnx.load(str(args.model))
    graph = model.graph
    ops = [node.op_type for node in graph.node]
    assert ops == ["MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add"]
    dop = simulate(model, "DOP853")
    rk = simulate(model, "RK45")
    boundary_gap = max(float(np.max(np.abs(np.array(a["state"]) - np.array(b["state"]))))
                       for a, b in zip(dop["boundaries"], rk["boundaries"]))
    first_gap = (None if not dop["first_crossing"] or not rk["first_crossing"] else
                 abs(dop["first_crossing"]["time"] - rk["first_crossing"]["time"]))
    summary = {
        "status": "numerical_candidate_only_not_a_certified_counterexample",
        "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model), "onnx_input": graph.input[0].name,
        "onnx_output": graph.output[0].name, "onnx_ops": ops,
        "controller_evaluation": "ONNX graph MatMul/Add/Relu in float32 at each period start; sample and hold",
        "plant": "official dynamics_dp.m, m=L=0.5, c=0, g=1; four-state continuous ODE",
        "initial_state": INITIAL, "period_s": PERIOD, "periods": PERIODS,
        "integration": {"methods": ["DOP853", "RK45"], "rtol": RTOL, "atol": ATOL,
                        "max_step_s": MAX_STEP, "dense_samples_per_period": 101},
        "first_crossing_DOP853": dop["first_crossing"],
        "first_crossing_RK45": rk["first_crossing"],
        "crossing_time_method_gap_s": first_gap,
        "max_boundary_state_method_gap": boundary_gap,
        "DOP853_sampled_min": dop["sampled_min"],
        "DOP853_sampled_max": dop["sampled_max"],
        "DOP853_final_state": dop["final_state"],
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "onnx_version": onnx.__version__, "content_digest": "not calculated",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "controls.json").write_text(json.dumps(dop["controls"], indent=2) + "\n")
    (args.output / "period_endpoints.json").write_text(json.dumps(dop["boundaries"], indent=2) + "\n")
    with (args.output / "trajectory.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("t", "period", "theta1", "theta2", "theta1_dot", "theta2_dot"))
        writer.writerows(dop["samples"])
    print(json.dumps({"first_crossing_DOP853": dop["first_crossing"],
                      "first_crossing_RK45": rk["first_crossing"],
                      "method_boundary_gap": boundary_gap,
                      "method_crossing_time_gap_s": first_gap}, indent=2))


if __name__ == "__main__":
    main()
