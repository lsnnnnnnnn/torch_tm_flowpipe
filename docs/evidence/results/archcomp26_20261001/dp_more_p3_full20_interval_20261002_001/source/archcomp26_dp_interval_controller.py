#!/usr/bin/env python3
"""Directed interval enclosure of the fixed DP 4-25-25-2 ONNX controller."""

from decimal import Decimal, localcontext


EXPECTED_OPS = ("MatMul", "Add", "Relu", "MatMul", "Add", "Relu", "MatMul", "Add")
EXPECTED_SHAPES = (((4, 25), (25,)), ((25, 25), (25,)), ((25, 2), (2,)))


def load_controller(path, torch, device):
    """Treat each stored float32 weight/bias as its exact binary rational value."""
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    nodes = list(graph.node)
    if tuple(node.op_type for node in nodes) != EXPECTED_OPS:
        raise ValueError("unexpected DP controller operator sequence")
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("unexpected DP controller input/output count")
    initializers = {value.name: numpy_helper.to_array(value) for value in graph.initializer}
    layers = []
    previous = graph.input[0].name
    for layer, offset in enumerate((0, 3, 6)):
        matmul, add = nodes[offset:offset + 2]
        if (list(matmul.input[:1]) != [previous]
                or list(add.input[:1]) != [matmul.output[0]]
                or matmul.input[1] not in initializers
                or add.input[1] not in initializers):
            raise ValueError("unexpected DP controller graph wiring")
        weight = initializers[matmul.input[1]]
        bias = initializers[add.input[1]]
        if (weight.dtype != np.float32 or bias.dtype != np.float32
                or (weight.shape, bias.shape) != EXPECTED_SHAPES[layer]):
            raise ValueError("unexpected DP controller weight type/shape")
        if layer < 2:
            relu = nodes[offset + 2]
            if list(relu.input) != [add.output[0]]:
                raise ValueError("unexpected DP controller ReLU wiring")
            previous = relu.output[0]
        else:
            previous = add.output[0]
        # Every float32 number is exactly representable as float64.
        w64 = np.asarray(weight, dtype=np.float64)
        b64 = np.asarray(bias, dtype=np.float64)
        layers.append((torch.from_numpy(w64).to(device), torch.from_numpy(b64).to(device)))
    if previous != graph.output[0].name:
        raise ValueError("unexpected DP controller output wiring")
    return tuple(layers)


def network_interval(hull, layers):
    """Evaluate MatMul/Add/ReLU with outward-rounded interval primitives."""
    import torch
    from flowstar_gpu import interval as iv

    if hull.ndim != 3 or hull.shape[1:] != (4, 2):
        raise ValueError("DP controller expects [B,4,2] input intervals")
    iv.assert_valid(hull, "DP controller input")
    values = hull
    for layer_index, (weight, bias) in enumerate(layers):
        # ONNX MatMul uses row-vector x times [input,output] W.
        values = iv.dot_point_iv(weight.T.unsqueeze(0), values.unsqueeze(1), dim=-1)
        values = iv.add(values, iv.from_point(bias).unsqueeze(0))
        if layer_index < 2:
            values = torch.maximum(values, torch.zeros_like(values))
        iv.assert_valid(values, f"DP controller layer {layer_index + 1}")
    return values


def residual_interval(hull, T, layers):
    """Enclose the exact real network minus the exact stored float64 T*x."""
    from flowstar_gpu import interval as iv

    if T.shape != (hull.shape[0], 2, 4):
        raise ValueError("DP controller CROWN T has the wrong shape")
    network = network_interval(hull, layers)
    linear = iv.dot_point_iv(T, hull.unsqueeze(1), dim=-1)
    residual = iv.sub(network, linear)
    iv.assert_valid(linear, "DP controller T*x")
    iv.assert_valid(residual, "DP controller residual")
    return network, linear, residual


def decimal_residual_at_point(x, T, layers):
    """High-precision independent point check; not used to create bounds."""
    with localcontext() as context:
        context.prec = 100
        values = [Decimal.from_float(float(v)) for v in x]
        for layer_index, (weight, bias) in enumerate(layers):
            w = weight.detach().cpu().tolist()
            b = bias.detach().cpu().tolist()
            out = []
            for j in range(len(b)):
                value = sum((values[i] * Decimal.from_float(w[i][j]) for i in range(len(values))),
                            Decimal.from_float(b[j]))
                out.append(max(value, Decimal(0)) if layer_index < 2 else value)
            values = out
        matrix = T.detach().cpu().tolist()
        return [
            values[j] - sum((Decimal.from_float(matrix[j][i]) * Decimal.from_float(float(x[i]))
                             for i in range(4)), Decimal(0))
            for j in range(2)
        ]


def sample_containment(hull, T, residual, layers, lane_indices):
    """Check nine deterministic corner/interior points per selected lane."""
    checked = 0
    min_slack = None
    patterns = ((0, 0, 0, 0), (0.5, 0.5, 0.5, 0.5), (1, 1, 1, 1),
                (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
                (0, 1, 0, 1), (1, 0, 1, 0))
    for lane in lane_indices:
        bounds = hull[lane].detach().cpu().tolist()
        enclosure = residual[lane].detach().cpu().tolist()
        for pattern in patterns:
            x = [lo if alpha == 0 else hi if alpha == 1 else lo + 0.5 * (hi - lo)
                 for alpha, (lo, hi) in zip(pattern, bounds)]
            if any(not (lo <= v <= hi) for v, (lo, hi) in zip(x, bounds)):
                raise AssertionError("sample point escaped its input interval")
            values = decimal_residual_at_point(x, T[lane], layers)
            for j, value in enumerate(values):
                lower = Decimal.from_float(enclosure[j][0])
                upper = Decimal.from_float(enclosure[j][1])
                if not lower <= value <= upper:
                    raise AssertionError(f"sample residual escaped lane={lane}, output={j}, pattern={pattern}")
                slack = min(value - lower, upper - value)
                min_slack = slack if min_slack is None else min(min_slack, slack)
                checked += 1
    return {"checked_output_values": checked, "minimum_decimal_slack": str(min_slack)}
