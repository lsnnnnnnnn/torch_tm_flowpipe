#!/usr/bin/env python3
"""One-period, center-point NAV interface diagnostic; no set verification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def evaluate_model(path: Path, state: np.ndarray) -> tuple[np.ndarray, list[str]]:
    model = onnx.load(str(path))
    ops = [node.op_type for node in model.graph.node]
    expected = ["MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add", "Tanh"]
    if ops != expected:
        raise ValueError(f"unexpected ONNX operator sequence: {ops}")
    values = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    values[model.graph.input[0].name] = state.reshape(1, 4).astype(np.float32)
    for node in model.graph.node:
        args = [values[name] for name in node.input]
        if node.op_type == "MatMul":
            result = args[0] @ args[1]
        elif node.op_type == "Add":
            result = args[0] + args[1]
        elif node.op_type == "Relu":
            result = np.maximum(args[0], 0)
        elif node.op_type == "Tanh":
            result = np.tanh(args[0])
        else:
            raise ValueError(node.op_type)
        if len(node.output) != 1:
            raise ValueError("expected one output per ONNX node")
        values[node.output[0]] = result
    control = values[model.graph.output[0].name]
    if control.shape != (1, 2) or not np.isfinite(control).all():
        raise ValueError(f"bad controller output: {control}")
    return control[0].astype(np.float64), ops


def flow(state: np.ndarray, control: np.ndarray, step: float = 0.01) -> np.ndarray:
    period = 0.2
    count = round(period / step)
    if not math.isclose(count * step, period, rel_tol=0, abs_tol=1e-14):
        raise ValueError("integration step must divide the control period")

    def derivative(x: np.ndarray) -> np.ndarray:
        return np.array(
            [x[2] * math.cos(x[3]), x[2] * math.sin(x[3]), control[0], control[1]],
            dtype=np.float64,
        )

    x = state.astype(np.float64).copy()
    for _ in range(count):
        k1 = derivative(x)
        k2 = derivative(x + 0.5 * step * k1)
        k3 = derivative(x + 0.5 * step * k2)
        k4 = derivative(x + step * k3)
        x += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--instance", choices=("nav-standard", "nav-robust"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = np.array([3.0, 3.0, 0.0, 0.0], dtype=np.float64)
    control, ops = evaluate_model(args.model, state)
    end = flow(state, control)
    half_step_end = flow(state, control, 0.005)
    result = {
        "run_id": args.run_id,
        "instance": args.instance,
        "status": "completed_diagnostic",
        "scope": "center point, first sampled control period only",
        "contract": {
            "state_order": ["x", "y", "speed", "heading"],
            "network_input": ["x", "y", "speed", "heading"],
            "network_output": ["speed_rate", "heading_rate"],
            "dynamics": ["speed*cos(heading)", "speed*sin(heading)", "u1", "u2"],
            "control_period_s": 0.2,
            "held_control": True,
        },
        "model_file": str(args.model),
        "onnx_ops": ops,
        "initial_state": state.tolist(),
        "control": control.tolist(),
        "state_at_0p2s": end.tolist(),
        "rk4_max_step_refinement_abs_diff": float(np.max(np.abs(end - half_step_end))),
        "complete_horizon": False,
        "initial_set_covered": False,
        "property_evaluated": False,
        "formal_certificate": False,
    }
    (args.output_dir / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
