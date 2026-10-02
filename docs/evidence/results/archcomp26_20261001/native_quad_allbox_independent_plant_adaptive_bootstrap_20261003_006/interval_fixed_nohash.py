"""Independent one-box QUAD interval gate; no Flow* execution or sampling.

Each small-step Picard enclosure contains the whole initial box and every
control in the saved affine-plus-residual hull for the entire 0.005 s step.
Decimal operations round outward.  The result concerns this short ODE gate,
not the neural network, the other 1023 boxes, or the full QUAD horizon.
"""

import csv
from decimal import Decimal as D, ROUND_CEILING, ROUND_FLOOR, localcontext
import json
import math
from pathlib import Path
import struct
import sys


PRECISION = 60
H = D("0.005")
STEPS = 1000
EPS = D("1e-9")


def down(f):
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = PRECISION, ROUND_FLOOR
        return +f()


def up(f):
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = PRECISION, ROUND_CEILING
        return +f()


class I:
    def __init__(self, lo, hi=None):
        self.lo = lo if isinstance(lo, D) else D(str(lo))
        self.hi = self.lo if hi is None else (hi if isinstance(hi, D) else D(str(hi)))
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    def __add__(self, other):
        other = interval(other)
        return I(down(lambda: self.lo + other.lo), up(lambda: self.hi + other.hi))

    __radd__ = __add__

    def __neg__(self):
        return I(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        return self + -interval(other)

    def __rsub__(self, other):
        return interval(other) + -self

    def __mul__(self, other):
        other = interval(other)
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return I(min(down(lambda a=a, b=b: a*b) for a, b in pairs),
                 max(up(lambda a=a, b=b: a*b) for a, b in pairs))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = interval(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("interval denominator contains zero")
        reciprocal = I(down(lambda: D(1)/other.hi), up(lambda: D(1)/other.lo))
        return self * reciprocal

    def inside(self, other, strict=False):
        return (other.lo < self.lo and self.hi < other.hi) if strict else (
            other.lo <= self.lo and self.hi <= other.hi)

    def pair(self):
        return [str(self.lo), str(self.hi)]


def interval(x):
    return x if isinstance(x, I) else I(x)


def trig_remainder(magnitude, power, factorial):
    return up(lambda: magnitude ** power / D(factorial))


def sine(x):
    x = interval(x)
    m = max(x.lo.copy_abs(), x.hi.copy_abs())
    if m >= D("0.1"):
        raise ValueError("small-angle sine enclosure exceeded")
    p = x - (x*x*x)/6 + (x*x*x*x*x)/120
    r = trig_remainder(m, 7, 5040)
    return p + I(r.copy_negate(), r)


def cosine(x):
    x = interval(x)
    m = max(x.lo.copy_abs(), x.hi.copy_abs())
    if m >= D("0.1"):
        raise ValueError("small-angle cosine enclosure exceeded")
    p = 1 - (x*x)/2 + (x*x*x*x)/24
    r = trig_remainder(m, 6, 720)
    return p + I(r.copy_negate(), r)


def ode(x, u):
    _, _, _, x4, x5, x6, x7, x8, x9, x10, x11, x12 = x
    s7, c7 = sine(x7), cosine(x7)
    s8, c8 = sine(x8), cosine(x8)
    s9, c9 = sine(x9), cosine(x9)
    return [
        c8*c9*x4 + (s7*s8*c9-c7*s9)*x5 + (c7*s8*c9+s7*s9)*x6,
        c8*s9*x4 + (s7*s8*s9+c7*c9)*x5 + (c7*s8*s9-s7*c9)*x6,
        s8*x4 - s7*c8*x5 - c7*c8*x6,
        x12*x5 - x11*x6 - D("9.81")*s8,
        x10*x6 - x12*x4 + D("9.81")*c8*s7,
        x11*x4 - x10*x5 + D("9.81")*c8*c7 - D("9.81") - u[0]/D("1.4"),
        x10 + s7*s8/c8*x11 + c7*s8/c8*x12,
        c7*x11 - s7*x12,
        s7*x11/c8 - c7*x12/c8,
        x11*x12*(D("0.054")-D("0.104"))/D("0.054") + u[1]/D("0.054"),
        (D("0.104")-D("0.054"))*x10*x12/D("0.054") + u[2]/D("0.054"),
        I(0),
    ]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def exact_float(x):
    return D.from_float(x)


def saved_interval(lo, hi):
    # The CSV emits binary64 bounds with 17 digits; move one binary64 ULP out.
    return I(exact_float(math.nextafter(float(lo), -math.inf)),
             exact_float(math.nextafter(float(hi), math.inf)))


def load_contract(root):
    rpc = json.loads((root / "on/rpc.json").read_text())
    params, coeff = rpc["params"], rpc["coefficients"]
    if len(params["input_lb"]) != 12 or len(params["input_ub"]) != 12:
        raise ValueError("unexpected state dimension")
    # Flow*'s Interval::inf/sup use MPFR_RNDD/RNDU; RPC binary64 bounds
    # already contain the initial set.  Keep its exact singleton zeros.
    x0 = [I(exact_float(float(a)), exact_float(float(b)))
          for a, b in zip(params["input_lb"], params["input_ub"])]
    t = [[I(exact_float(f32(v))) for v in row] for row in coeff["T"][0]]
    if len(t) != 3 or any(len(row) != 12 for row in t):
        raise ValueError("unexpected affine controller shape")
    u = []
    for j in range(3):
        low = f32(coeff["u_min"][0][j])
        high = f32(coeff["u_max"][0][j])
        if low > high:
            raise ValueError("reversed residual control interval")
        center, radius = (high+low)/2, (high-low)/2
        radius_decimal = exact_float(radius)
        control = I(exact_float(center)) + I(radius_decimal.copy_negate(), radius_decimal)
        for i in range(12):
            control += t[j][i]*x0[i]
        u.append(control)
    return x0, u


def forms(x):
    return [x[0], x[1], x[0]+x[1], x[0]-x[1]]


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def run(root):
    root = root / "quad" if (root / "quad/on/rpc.json").exists() else root
    x, u = load_contract(root)
    h = H / STEPS
    tube = [None]*4
    for step in range(STEPS):
        fx = ode(x, u)
        y = [x[i] + I(0, h)*fx[i] + I(EPS.copy_negate(), EPS) for i in range(12)]
        fy = ode(y, u)
        picard = [x[i] + I(0, h)*fy[i] for i in range(12)]
        if not all(picard[i].inside(y[i], strict=True) for i in range(12)):
            raise RuntimeError(f"Picard inclusion failed at substep {step}")
        for i, value in enumerate(forms(y)):
            tube[i] = value if tube[i] is None else I(
                min(tube[i].lo, value.lo), max(tube[i].hi, value.hi))
        x = [x[i] + I(h)*fy[i] for i in range(12)]

    support = {(int(r["view"]), int(r["form"])): saved_interval(r["lo"], r["hi"])
               for r in read_csv(root / "on/octagon.csv")}
    terminal = read_csv(root / "on/terminal_axes.csv")
    if len(support) != 8 or len(terminal) != 12:
        raise ValueError("missing native directions or terminal axes")
    comparisons = []
    for view, values in ((0, tube), (1, forms(x))):
        for form, value in enumerate(values):
            saved = support[(view, form)]
            comparisons.append({"kind": "tube" if view == 0 else "endpoint",
                                "form": form, "independent": value.pair(),
                                "native": saved.pair(), "inside": value.inside(saved)})
    for i, row in enumerate(terminal):
        if int(row["coord"]) != i+1:
            raise ValueError("reordered native terminal axes")
        for mode in ("pre", "composed"):
            saved = saved_interval(row[mode+"_lo"], row[mode+"_hi"])
            comparisons.append({"kind": "terminal_"+mode, "coord": i+1,
                                "independent": x[i].pair(), "native": saved.pair(),
                                "inside": x[i].inside(saved)})
    return {
        "schema": "native-quad-independent-whole-set-one-step-interval-gate-nohash-v1",
        "scope": "first saved QUAD initial box; every affine-plus-residual control admitted by its RPC; all t in [0,0.005]",
        "method": "1000 outward-rounded Decimal interval Picard substeps; fifth-order sine and fourth-order cosine Taylor enclosures with Lagrange remainder; frozen constant-control hull",
        "decimal_precision": PRECISION, "substeps": STEPS, "substep_s": str(h),
        "picard_inclusion_passed": True,
        "initial_state_hull": [v.pair() for v in load_contract(root)[0]],
        "control_hull": [v.pair() for v in u],
        "all_saved_comparisons_inside": all(r["inside"] for r in comparisons),
        "comparison_count": len(comparisons),
        "comparison_failures": [r for r in comparisons if not r["inside"]],
        "comparisons": comparisons,
        "limitation": "This checks a mathematical one-box/one-step constant-control relaxation against saved native intervals; it does not validate CROWN, model parsing, other boxes, later control updates, or T=5.",
    }


def demo():
    a, b = I("-0.1", "0.2"), I("2", "3")
    assert (a*b).pair() == ["-0.3", "0.6"]
    assert sine(I(0)).lo <= 0 <= sine(I(0)).hi
    assert cosine(I(0)).lo <= 1 <= cosine(I(0)).hi


if __name__ == "__main__":
    demo()
    if len(sys.argv) != 2:
        raise SystemExit("usage: checker native-quad-var-tail-repair-gate-directory")
    evidence = run(Path(sys.argv[1]))
    print(json.dumps(evidence, indent=2))
    if not evidence["all_saved_comparisons_inside"]:
        raise SystemExit(1)
