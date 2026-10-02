"""Export the fixed official ONNX's float32 initializers for a local interval check."""

import json
import onnx
from onnx import numpy_helper

MODEL = "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_dp_more_prep_001/controller_double_pendulum_more_robust.onnx"
graph = onnx.load(MODEL).graph
assert [node.op_type for node in graph.node] == [
    "MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add"
]
weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
assert all(value.dtype.name == "float32" for value in weights.values())
assert [weights[f"dense_{i}/kernel:0"].shape for i in (1, 2, 3)] == [
    (4, 25), (25, 25), (25, 2)
]
print(json.dumps({name: value.astype("float64").tolist() for name, value in weights.items()}))
