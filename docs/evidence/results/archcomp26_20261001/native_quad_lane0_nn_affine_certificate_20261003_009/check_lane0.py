#!/usr/bin/env python3
"""CPU-only directed affine enclosure of the saved QUAD lane-0 NN residual."""

import json
from decimal import Context, Decimal as D, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
MODEL = ROOT / "research/gpu_verified_20260930/source/benchmark/quad_controller_3_64_torch.onnx"
RPC = HERE.parent / "native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json"
LO = Context(prec=80, rounding=ROUND_FLOOR)
HI = Context(prec=80, rounding=ROUND_CEILING)
NEAR = Context(prec=80, rounding=ROUND_HALF_EVEN)
ZERO, ONE, QUARTER = D(0), D(1), D("0.25")


def iv(a, b=None):
    return (a, a if b is None else b)


def add(a, b):
    return LO.add(a[0], b[0]), HI.add(a[1], b[1])


def neg(a):
    return a[1].copy_negate(), a[0].copy_negate()


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    pairs = [(x, y) for x in a for y in b]
    return min(LO.multiply(x, y) for x, y in pairs), max(HI.multiply(x, y) for x, y in pairs)


def divpos(a, b):
    assert b[0] > 0
    pairs = [(x, y) for x in a for y in b]
    return min(LO.divide(x, y) for x, y in pairs), max(HI.divide(x, y) for x, y in pairs)


def exp_point(x):
    # Decimal exp is correctly rounded to nearest; one neighboring representable
    # decimal on each side encloses the exact transcendental result.
    value = NEAR.exp(x)
    return NEAR.next_minus(value), NEAR.next_plus(value)


def sigmoid(a):
    e = exp_point(a[1].copy_negate())[0], exp_point(a[0].copy_negate())[1]
    return divpos(iv(ONE), add(iv(ONE), e))


def hvalue(x, alpha):
    return sub(sigmoid(iv(x)), mul(iv(alpha), iv(x)))


def derivative_bounds(a):
    ss = sigmoid(a)
    ends = [sigmoid(iv(x)) for x in a]
    vals = [mul(s, sub(iv(ONE), s)) for s in ends]
    upper = QUARTER if ss[0] <= D("0.5") <= ss[1] else max(v[1] for v in vals)
    return min(v[0] for v in vals), upper


def sigmoid_error(z, alpha, max_depth=32):
    """Enclose sigmoid(z)-alpha*z by monotonic pieces and tiny unresolved cells."""
    pending = [(z[0], z[1], 0)]
    pieces = []
    deepest = 0
    while pending:
        a, b, depth = pending.pop()
        deepest = max(deepest, depth)
        derivative = derivative_bounds((a, b))
        if derivative[0] >= alpha or derivative[1] <= alpha:
            ends = hvalue(a, alpha), hvalue(b, alpha)
            pieces.append((min(x[0] for x in ends), max(x[1] for x in ends)))
        elif depth >= max_depth:
            pieces.append(sub(sigmoid((a, b)), mul(iv(alpha), (a, b))))
        else:
            m = NEAR.divide(NEAR.add(a, b), D(2))
            if not a < m < b:
                pieces.append(sub(sigmoid((a, b)), mul(iv(alpha), (a, b))))
            else:
                pending.extend(((a, m, depth+1), (m, b, depth+1)))
    return (min(p[0] for p in pieces), max(p[1] for p in pieces)), deepest, len(pieces)


def wire_uint(blob, offset):
    value = shift = 0
    while True:
        byte = blob[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7


def fields(blob):
    offset = 0
    while offset < len(blob):
        tag, offset = wire_uint(blob, offset)
        key, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = wire_uint(blob, offset)
        elif wire == 2:
            size, offset = wire_uint(blob, offset)
            value = blob[offset:offset+size]
            offset += size
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = blob[offset:offset+size]
            offset += size
        else:
            raise ValueError(f"unknown ONNX wire type {wire}")
        yield key, value
    assert offset == len(blob)


def tensor(blob):
    fs = list(fields(blob))
    dims = [v for k, v in fs if k == 1]
    assert [v for k, v in fs if k == 2] == [1], "expected float32 tensor"
    name = next(v.decode() for k, v in fs if k == 8)
    raw = next(v for k, v in fs if k == 9)
    assert len(raw) == 4 * (dims[0] * (dims[1] if len(dims) == 2 else 1))
    values = [D.from_float(v) for v in struct.unpack('<'+'f'*(len(raw)//4), raw)]
    return name, (dims, values)


def load_graph():
    graph = next(v for k, v in fields(MODEL.read_bytes()) if k == 7)
    fs = list(fields(graph))
    initializers = dict(tensor(v) for k, v in fs if k == 5)
    nodes = []
    for k, v in fs:
        if k != 1:
            continue
        nf = list(fields(v))
        op = next(x.decode() for q, x in nf if q == 4)
        inputs = [x.decode() for q, x in nf if q == 1]
        outputs = [x.decode() for q, x in nf if q == 2]
        attrs = {}
        for q, blob in nf:
            if q != 5:
                continue
            af = dict(fields(blob))
            name = af[1].decode()
            assert name in ('alpha', 'beta', 'transB')
            attrs[name] = struct.unpack('<f', af[2])[0] if name in ('alpha', 'beta') else af[3]
        nodes.append((op, inputs, outputs, attrs))
    assert len(nodes) == 7 and [x[0] for x in nodes] == ['Gemm', 'Sigmoid'] * 3 + ['Gemm']
    assert len(initializers) == 8
    assert [x[1][0] for x in nodes if x[0] == 'Gemm'] == ['x'] + [nodes[i][2][0] for i in (1, 3, 5)]
    assert [x[1][0] for x in nodes if x[0] == 'Sigmoid'] == [nodes[i][2][0] for i in (0, 2, 4)]
    assert nodes[-1][2] == ['u']
    layers = []
    for node, ins, _, attrs in nodes:
        if node != 'Gemm':
            continue
        assert attrs == {'alpha': 1.0, 'beta': 1.0, 'transB': 1}
        wd, w = initializers[ins[1]]
        bd, b = initializers[ins[2]]
        assert wd == [len(b), 12 if not layers else len(layers[-1][1])]
        assert bd == [len(b)]
        layers.append(([w[j*wd[1]:(j+1)*wd[1]] for j in range(wd[0])], b))
    assert [len(b) for _, b in layers] == [64, 64, 64, 3]
    return layers, nodes


def layer_linear(state, layer):
    A, b, R = state
    weights, bias = layer
    outA, outb, outR = [], [], []
    for row, offset in zip(weights, bias):
        coeffs = []
        for axis in range(12):
            value = iv(ZERO)
            for j, weight in enumerate(row):
                value = add(value, mul(iv(weight), A[j][axis]))
            coeffs.append(value)
        base, err = iv(offset), iv(ZERO)
        for j, weight in enumerate(row):
            base = add(base, mul(iv(weight), b[j]))
            err = add(err, mul(iv(weight), R[j]))
        outA.append(coeffs)
        outb.append(base)
        outR.append(err)
    return outA, outb, outR


def range_affine(coeffs, base, err, centered):
    value = add(base, err)
    for a, x in zip(coeffs, centered):
        value = add(value, mul(a, x))
    return value


def check_box(lb, ub, layers, rpc_coeff):
    assert all(a <= b for a, b in zip(lb, ub))
    centers = [NEAR.divide(NEAR.add(a, b), D(2)) for a, b in zip(lb, ub)]
    centered = [(LO.subtract(a, c), HI.subtract(b, c)) for a, b, c in zip(lb, ub, centers)]
    A = [[iv(ONE if i == j else ZERO) for i in range(12)] for j in range(12)]
    state = (A, [iv(c) for c in centers], [iv(ZERO) for _ in range(12)])
    diagnostics = []
    for index, layer in enumerate(layers):
        state = layer_linear(state, layer)
        if index == 3:
            break
        A, b, R = state
        nextA, nextb, nextR = [], [], []
        layer_diag = []
        for j in range(len(b)):
            z = range_affine(A[j], b[j], R[j], centered)
            if z[0] == z[1]:
                alpha = ZERO
            else:
                sl, sh = sigmoid(iv(z[0])), sigmoid(iv(z[1]))
                secant = divpos(sub(sh, sl), sub(iv(z[1]), iv(z[0])))
                alpha = NEAR.divide(NEAR.add(secant[0], secant[1]), D(2))
                alpha = min(QUARTER, max(ZERO, alpha))
            error, depth, pieces = sigmoid_error(z, alpha)
            nextA.append([mul(iv(alpha), v) for v in A[j]])
            nextb.append(mul(iv(alpha), b[j]))
            nextR.append(add(mul(iv(alpha), R[j]), error))
            layer_diag.append({'neuron': j, 'z': [str(x) for x in z], 'alpha': str(alpha),
                               'error': [str(x) for x in error], 'depth': depth, 'pieces': pieces})
        diagnostics.append(layer_diag)
        state = nextA, nextb, nextR
    A, b, R = state
    outputs = []
    for j in range(3):
        T = [D.from_float(struct.unpack('<f', struct.pack('<f', v))[0]) for v in rpc_coeff['T'][0][j]]
        low = D.from_float(struct.unpack('<f', struct.pack('<f', rpc_coeff['u_min'][0][j]))[0])
        high = D.from_float(struct.unpack('<f', struct.pack('<f', rpc_coeff['u_max'][0][j]))[0])
        delta = [sub(A[j][i], iv(T[i])) for i in range(12)]
        baseline = b[j]
        for t, c in zip(T, centers):
            baseline = sub(baseline, mul(iv(t), iv(c)))
        residual = range_affine(delta, baseline, R[j], centered)
        outputs.append({'output': j+1, 'bound': [str(x) for x in residual],
                        'saved': [str(low), str(high)], 'width': str(HI.subtract(residual[1], residual[0])),
                        'left_margin': str(LO.subtract(residual[0], low)),
                        'right_margin': str(LO.subtract(high, residual[1])),
                        'included': low <= residual[0] and residual[1] <= high})
    return outputs, diagnostics


def main():
    long = D('1.12345678901234567890123456789012345678901234567890')
    assert neg(iv(long))[0].copy_negate() == long
    assert sigmoid(iv(ZERO))[0] <= D('0.5') <= sigmoid(iv(ZERO))[1]
    # The tiny parser deliberately rejects unexpected graph wiring and attributes.
    layers, nodes = load_graph()
    data = json.loads(RPC.read_text())
    lb = [D.from_float(x) for x in data['params']['input_lb'][:12]]
    ub = [D.from_float(x) for x in data['params']['input_ub'][:12]]
    outputs, diagnostics = check_box(lb, ub, layers, data['coefficients'])
    result = {'status': 'CERTIFIED_LANE0' if all(x['included'] for x in outputs) else 'UNDECIDED',
              'scope': 'selected ONNX exact binary32 parameters as real arithmetic; saved first RPC lane 0 only',
              'model': str(MODEL.relative_to(ROOT)), 'rpc': str(RPC.relative_to(ROOT)),
              'graph_ops': [n[0] for n in nodes], 'input_bounds': [[str(a), str(b)] for a, b in zip(lb, ub)],
              'outputs': outputs, 'hidden_layer_diagnostics': diagnostics,
              'sigmoid_method': 'Decimal correctly rounded exp with adjacent decimal neighbors; monotonic derivative partition'}
    (HERE / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'outputs': outputs}, indent=2))


if __name__ == '__main__':
    main()
