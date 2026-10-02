#!/usr/bin/env python3
"""Audit 2026 TORA reach MAT controllers and export an explicitly chosen ONNX.

Run this with an existing Python environment containing NumPy, SciPy, ONNX,
and ONNX Runtime.  The audit and export only evaluate controllers at points;
they do not launch a reachability experiment.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from scipy.io import loadmat
from scipy.special import expit


STEMS = ("nn_tora_sigmoid", "nn_tora_relu_tanh")
ACT_OP = {"sigmoid": "Sigmoid", "relu": "Relu", "tanh": "Tanh"}


def controller(mat_path):
    data = loadmat(mat_path, squeeze_me=True)
    weights = [np.atleast_2d(np.asarray(w, dtype=np.float64)) for w in data["W"]]
    biases = [np.atleast_1d(np.asarray(b, dtype=np.float64)) for b in data["b"]]
    acts = [str(a).lower() for a in data["act_fcns"]]
    if len(weights) != 4 or len(biases) != 4 or len(acts) != 4:
        raise ValueError("expected four fully connected layers")
    width = 4
    for w, b, a in zip(weights, biases, acts):
        if w.shape[1] != width or b.shape != (w.shape[0],) or a not in ACT_OP:
            raise ValueError("invalid MAT layer shape or activation")
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError("non-finite MAT parameters")
        width = w.shape[0]
    if width != 1:
        raise ValueError("expected scalar controller output")
    return weights, biases, acts


def sample_points():
    lo = np.array([-.77, -.45, .51, -.3], dtype=np.float64)
    hi = np.array([-.75, -.43, .54, -.28], dtype=np.float64)
    return np.array(
        [(lo + hi) / 2]
        + [np.array(c) for c in itertools.product(*zip(lo, hi))]
        + [lo + (hi - lo) * t for t in (.25, .75)]
        + [np.zeros(4), np.ones(4), -np.ones(4)],
        dtype=np.float64,
    )


def forward(point, weights, biases, acts, scale=1.0, offset=0.0):
    x = point[None, :]
    for w, b, a in zip(weights, biases, acts):
        x = x @ w.T + b
        x = expit(x) if a == "sigmoid" else np.maximum(x, 0) if a == "relu" else np.tanh(x)
    return float(scale * x[0, 0] + offset)


def compare_forward(model_path, points, weights, biases, acts, scale=1.0, offset=0.0):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    actual = [float(session.run(None, {input_name: p[None, :]})[0].reshape(-1)[0]) for p in points]
    expected = [forward(p, weights, biases, acts, scale, offset) for p in points]
    error = max(abs(a - e) for a, e in zip(actual, expected))
    if not math.isfinite(error) or error > 1e-12:
        raise AssertionError(f"controller point comparison failed: {error}")
    return {"points": len(points), "max_abs_error": error, "center_mat": expected[0], "center_onnx": actual[0]}


def audit(mat_dir, old_onnx_dir, out):
    points = sample_points()
    receipt = {"kind": "point_controller_preflight_only", "sample_points": points.tolist(), "models": {}}
    for stem in STEMS:
        weights, biases, acts = controller(mat_dir / f"{stem}.mat")
        model_path = old_onnx_dir / f"{stem}.onnx"
        graph = onnx.load(str(model_path)).graph
        initializers = {v.name: numpy_helper.to_array(v) for v in graph.initializer}
        op_sequence = [n.op_type for n in graph.node]
        expected_ops = [op for a in acts for op in ("Gemm", ACT_OP[a])]
        if op_sequence != expected_ops:
            raise AssertionError(f"{stem}: old ONNX activation graph differs from MAT")
        layer_receipts = []
        for i, (w, b) in enumerate(zip(weights, biases), 1):
            old_w = initializers[f"layers.lin{i}.weight"]
            old_b = initializers[f"layers.lin{i}.bias"]
            if w.shape != old_w.shape or b.shape != old_b.shape:
                raise AssertionError(f"{stem}: layer {i} shape differs")
            w_equal = bool(np.array_equal(w, old_w))
            b_equal = bool(np.array_equal(b, old_b))
            if not w_equal or not b_equal:
                raise AssertionError(f"{stem}: layer {i} parameters differ")
            layer_receipts.append({"layer": i, "weight_shape": list(w.shape), "bias_shape": list(b.shape),
                                   "weight_elementwise_equal": w_equal, "bias_elementwise_equal": b_equal})
        receipt["models"][stem] = {
            "mat_path": str(mat_dir / f"{stem}.mat"), "old_onnx_path": str(model_path),
            "mat_activations": acts, "old_onnx_ops": op_sequence,
            "layers": layer_receipts,
            "forward": compare_forward(model_path, points, weights, biases, acts),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(out)


def build(mat_path, activations, scale, offset, out):
    weights, biases, mat_acts = controller(mat_path)
    acts = mat_acts if activations == "mat" else activations.lower().split(",")
    if len(acts) != 4 or any(a not in ACT_OP for a in acts):
        raise ValueError("--activations must be mat or four comma-separated activations")
    if not math.isfinite(scale) or not math.isfinite(offset):
        raise ValueError("control scale and offset must be finite")
    nodes, initializers = [], []
    previous = "state"
    for i, (w, b, a) in enumerate(zip(weights, biases, acts), 1):
        w_name, b_name, affine, active = f"W{i}", f"b{i}", f"z{i}", f"a{i}"
        initializers += [numpy_helper.from_array(w, w_name), numpy_helper.from_array(b, b_name)]
        nodes.append(helper.make_node("Gemm", [previous, w_name, b_name], [affine], transB=1))
        nodes.append(helper.make_node(ACT_OP[a], [affine], [active]))
        previous = active
    initializers += [numpy_helper.from_array(np.array([scale], dtype=np.float64), "control_scale"),
                     numpy_helper.from_array(np.array([offset], dtype=np.float64), "control_offset")]
    nodes += [helper.make_node("Mul", [previous, "control_scale"], ["scaled"]),
              helper.make_node("Add", ["scaled", "control_offset"], ["control_u"])]
    graph = helper.make_graph(nodes, "tora_reach_explicit_control", [helper.make_tensor_value_info("state", TensorProto.DOUBLE, [1, 4])],
                              [helper.make_tensor_value_info("control_u", TensorProto.DOUBLE, [1, 1])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)])
    model.ir_version = 6
    onnx.checker.check_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    evidence = compare_forward(out, sample_points(), weights, biases, acts, scale, offset)
    receipt = {"kind": "point_controller_export_only", "mat_path": str(mat_path), "model_path": str(out),
               "activations": acts, "control_formula": "u = scale * network(x) + offset",
               "scale": scale, "offset": offset, "forward": evidence,
               "execution_requirement": "Use ONNX output directly as plant u; disable all external control scaling."}
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--mat-dir", type=Path, required=True)
    a.add_argument("--old-onnx-dir", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--mat", type=Path, required=True)
    b.add_argument("--activations", required=True, help="mat or four comma-separated names")
    b.add_argument("--scale", type=float, required=True)
    b.add_argument("--offset", type=float, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.mat_dir, args.old_onnx_dir, args.out)
    else:
        build(args.mat, args.activations, args.scale, args.offset, args.out)


if __name__ == "__main__":
    main()
