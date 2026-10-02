#!/usr/bin/env python3
"""Outward fixed-point intervals for one DP-more closed-loop trajectory.

This is independent of the four flowpipe engines. Integer interval endpoints
represent multiples of 10^-35; every elementary operation rounds outward.
"""

import argparse
import json
import math
from pathlib import Path

Q = 10**35
ZERO = None
ONE = None


def ceildiv(a, b):
    return -((-a) // b)


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = lo
        self.hi = lo if hi is None else hi
        assert self.lo <= self.hi

    @staticmethod
    def integer(value):
        return I(value * Q)

    @staticmethod
    def rat(numerator, denominator):
        assert denominator > 0
        return I((numerator * Q) // denominator, ceildiv(numerator * Q, denominator))

    @staticmethod
    def floating(value):
        numerator, denominator = value.as_integer_ratio()
        return I.rat(numerator, denominator)

    def __add__(self, other):
        return I(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        products = (self.lo * other.lo, self.lo * other.hi,
                    self.hi * other.lo, self.hi * other.hi)
        return I(min(products) // Q, ceildiv(max(products), Q))

    def reciprocal(self):
        assert self.lo > 0 or self.hi < 0, "division interval contains zero"
        return I((Q * Q) // self.hi, ceildiv(Q * Q, self.lo))

    def __truediv__(self, other):
        return self * other.reciprocal()

    def absmax(self):
        return max(abs(self.lo), abs(self.hi))

    def enclosed_by(self, other):
        return other.lo <= self.lo and self.hi <= other.hi

    def widen(self, scaled_radius):
        return I(self.lo - scaled_radius, self.hi + scaled_radius)

    def as_strings(self):
        def fmt(value):
            sign = "-" if value < 0 else ""
            value = abs(value)
            return f"{sign}{value // Q}.{value % Q:035d}"
        return [fmt(self.lo), fmt(self.hi)]


ZERO, ONE = I.integer(0), I.integer(1)
HALF = I.rat(1, 2)
EPS32 = I.rat(1, 2**24)
MIN32 = I.rat(1, 2**126)
TRIG_LIMIT = I.rat(3, 2).hi
assert 3**59 * Q < 2**59 * math.factorial(59)
assert 3**60 * Q < 2**60 * math.factorial(60)


def sin_point(x):
    assert x.lo == x.hi and abs(x.lo) <= TRIG_LIMIT
    term = x
    total = term
    xx = x * x
    for k in range(1, 30):
        term = -(term * xx / I.integer((2 * k) * (2 * k + 1)))
        total = total + term
    return total.widen(1)  # Lagrange remainder < 10^-35 on |x| <= 3/2.


def cos_point(x):
    assert x.lo == x.hi and abs(x.lo) <= TRIG_LIMIT
    term = ONE
    total = term
    xx = x * x
    for k in range(1, 30):
        term = -(term * xx / I.integer((2 * k - 1) * (2 * k)))
        total = total + term
    return total.widen(1)


def sin_range(x):
    assert -TRIG_LIMIT <= x.lo <= x.hi <= TRIG_LIMIT
    return I(sin_point(I(x.lo)).lo, sin_point(I(x.hi)).hi)


def cos_range(x):
    assert -TRIG_LIMIT <= x.lo <= x.hi <= TRIG_LIMIT
    smallest = 0 if x.lo <= 0 <= x.hi else min(abs(x.lo), abs(x.hi))
    largest = x.absmax()
    return I(cos_point(I(largest)).lo, cos_point(I(smallest)).hi)


def round32(x):
    radius = (EPS32 * I(x.absmax()) + MIN32).hi
    return x.widen(radius)


def matmul(values, kernel):
    result = []
    for column in zip(*kernel):
        products = [value * weight for value, weight in zip(values, column)]
        exact_sum = ZERO
        abs_sum = 0
        for product in products:
            exact_sum = exact_sum + product
            abs_sum += product.absmax()
        n = len(products)
        # gamma_(2n) <= 4n*2^-24 for n<=25; 8n leaves margin for
        # arbitrary reduction order, fused operations, and tiny terms.
        radius = (I.integer(8 * n) * (EPS32 * I(abs_sum) + MIN32)).hi
        result.append(exact_sum.widen(radius))
    return result


def controller(state, weights):
    values = [round32(x) for x in state]
    for layer in (1, 2, 3):
        values = matmul(values, weights[f"dense_{layer}/kernel:0"])
        values = [round32(x + bias) for x, bias in zip(
            values, weights[f"dense_{layer}/bias:0"])]
        if layer < 3:
            values = [I(max(0, x.lo), max(0, x.hi)) for x in values]
    return values


class Jet:
    __slots__ = ("x", "d")

    def __init__(self, x, d=None):
        self.x = x
        self.d = (ZERO,) * 4 if d is None else d

    @staticmethod
    def variable(x, coordinate):
        return Jet(x, tuple(ONE if i == coordinate else ZERO for i in range(4)))

    def __add__(self, other):
        if not isinstance(other, Jet):
            other = Jet(other)
        return Jet(self.x + other.x, tuple(a + b for a, b in zip(self.d, other.d)))

    def __neg__(self):
        return Jet(-self.x, tuple(-a for a in self.d))

    def __sub__(self, other):
        return self + -other if isinstance(other, Jet) else self + -other

    def __mul__(self, other):
        if not isinstance(other, Jet):
            other = Jet(other)
        return Jet(self.x * other.x, tuple(a * other.x + self.x * b
                                           for a, b in zip(self.d, other.d)))

    def reciprocal(self):
        inv = self.x.reciprocal()
        squared = inv * inv
        return Jet(inv, tuple(-(entry * squared) for entry in self.d))

    def __truediv__(self, other):
        if not isinstance(other, Jet):
            other = Jet(other)
        return self * other.reciprocal()


def sine(x):
    if isinstance(x, Jet):
        return Jet(sin_range(x.x), tuple(cos_range(x.x) * d for d in x.d))
    return sin_range(x)


def cosine(x):
    if isinstance(x, Jet):
        return Jet(cos_range(x.x), tuple(-(sin_range(x.x) * d) for d in x.d))
    return cos_range(x)


def plant(state, control):
    th1, th2, v1, v2 = state
    t1, t2 = control
    delta = th1 - th2
    s, c = sine(delta), cosine(delta)
    a = t1 * I.integer(4) + sine(th1) * I.integer(2) - v2 * v2 * s * HALF
    b = v1 * v1 * s + t2 * I.integer(8) + sine(th2) * I.integer(2) - c * (
        -(v2 * v2 * s * HALF) + t1 * I.integer(4) + sine(th1) * I.integer(2))
    denominator = c * c * HALF - ONE
    return (v1, v2, a + c * b / (denominator * I.integer(2)), -b / denominator)


def tube_and_step(state, control, h):
    f0 = plant(state, control)
    time = I(0, h.hi)
    tube = tuple(x + time * f for x, f in zip(state, f0))
    for _ in range(10):
        image = tuple(x + time * f for x, f in zip(state, plant(tube, control)))
        if all(z.enclosed_by(y) for z, y in zip(image, tube)):
            break
        tube = tuple(I(min(y.lo, z.lo) - 10, max(y.hi, z.hi) + 10)
                     for y, z in zip(tube, image))
    else:
        raise ArithmeticError("Picard tube did not self-enclose")
    jets = plant([Jet.variable(x, i) for i, x in enumerate(tube)],
                 [Jet(x) for x in control])
    fy = tuple(j.x for j in jets)
    second = tuple(sum_intervals(jac * flow for jac, flow in zip(j.d, fy))
                   for j in jets)
    hh = h * h * HALF
    endpoint = tuple(x + h * f + hh * acc for x, f, acc in zip(state, f0, second))
    return endpoint, tube


def sum_intervals(items):
    result = ZERO
    for item in items:
        result = result + item
    return result


def read_weights(path):
    raw = json.loads(path.read_text())
    converted = {}
    for name, array in raw.items():
        if name.endswith("kernel:0"):
            converted[name] = [[I.floating(float(item)) for item in row] for row in array]
        else:
            converted[name] = [I.floating(float(item)) for item in array]
    assert [len(converted[f"dense_{i}/kernel:0"]) for i in (1, 2, 3)] == [4, 25, 25]
    return converted


def self_check():
    assert ONE.enclosed_by(I.rat(1, 3) * I.integer(3))
    assert I.integer(-2).enclosed_by(I.rat(-1, 2).reciprocal())
    assert ZERO.enclosed_by(sin_range(ZERO))
    assert ONE.enclosed_by(cos_range(ZERO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--substeps", type=int, default=200)
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check_only:
        print("self_check_ok")
        return
    assert args.substeps > 0
    weights = read_weights(args.weights)
    initial = I.floating(float("1.299"))
    assert I.integer(1).hi < initial.lo and initial.hi < I.rat(13, 10).lo
    state = (initial,) * 4
    h = I.rat(1, 50 * args.substeps)
    result = {"schema": "dp-more-outward-point-v1", "initial": initial.as_strings(),
              "fixed_point_scale": "10^-35", "substeps_per_0p02_period": args.substeps,
              "periods_requested": 18, "ode_step": h.as_strings(),
              "nn_arithmetic": "IEEE float32 round-to-nearest input, MatMul product/reduction, Add; no reduced-mantissa backend",
              "period_records": [], "validated_steps": 0, "status": "incomplete"}
    try:
        for period in range(18):
            control = controller(state, weights)
            for _ in range(args.substeps):
                state, _tube = tube_and_step(state, control, h)
                result["validated_steps"] += 1
            result["period_records"].append({
                "period": period, "time": I.rat(period + 1, 50).as_strings(),
                "control": [x.as_strings() for x in control],
                "endpoint": [x.as_strings() for x in state]})
            print(f"period {period + 1}: theta1_dot {state[2].as_strings()}", flush=True)
        result["violation_at_t_0p36"] = state[2].hi < -I.rat(3, 2).lo
        result["status"] = "validated_violation" if result["violation_at_t_0p36"] else "not_resolved"
    except (AssertionError, ArithmeticError, ZeroDivisionError) as error:
        result["failure"] = type(error).__name__ + ": " + str(error)
        result["status"] = "first_refusal"
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
