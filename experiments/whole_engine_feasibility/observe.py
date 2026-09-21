"""Lossless canonical exports and the existing exact-rational observer.

This module intentionally does not import either numerical engine or torch.
Tensor-to-host conversion and observation belong outside solver timing.
"""
from fractions import Fraction
import math

from experiments.xiangru_adoption.common import measure


def _list(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else value


def _number(value):
    result = float.fromhex(value) if isinstance(value, str) else float(value)
    if not math.isfinite(result):
        raise ValueError("canonical models require finite binary64 values")
    return result


def _interval(pair):
    if len(pair) != 2:
        raise ValueError("expected a lower/upper pair")
    lo, hi = map(_number, pair)
    if lo > hi:
        raise ValueError("inverted interval")
    return [lo.hex(), hi.hex()]


def canonical_model(coefficients, remainder, exponents, domain, *,
                    coefficient_bounds=None, variables=None):
    """Export one lane: coefficients [state, term], remainder [state, 2].

    coefficient_bounds, when provided, is (lower, upper), each [state, term].
    These intervals replace the point coefficients for observation; their radii
    are neither discarded nor added to the remainder a second time. The caller
    must supply a complete physical TM, including any needed state composition.
    All arrays may be nested lists or tensors; no batch axis is inferred.
    """
    coefficients, remainder, exponents, domain = map(
        _list, (coefficients, remainder, exponents, domain))
    domains = [_interval(pair) for pair in domain]
    if not domains or len(coefficients) != 2 or len(remainder) != 2:
        raise ValueError("expected a nonempty domain and two state components")
    for degrees in exponents:
        if len(degrees) != len(domains) or any(
                type(e) is not int or e < 0 for e in degrees):
            raise ValueError("invalid monomial degrees or variable order")
    if variables is not None and len(variables) != len(domains):
        raise ValueError("variable labels do not match the domain")
    if coefficient_bounds is None:
        lower = upper = coefficients
    else:
        if len(coefficient_bounds) != 2:
            raise ValueError("coefficient_bounds must be (lower, upper)")
        lower, upper = map(_list, coefficient_bounds)
        if len(lower) != 2 or len(upper) != 2:
            raise ValueError("coefficient bounds require two state components")
    components = []
    for state in range(2):
        if any(len(values[state]) != len(exponents)
               for values in (coefficients, lower, upper)):
            raise ValueError("coefficient/support length mismatch")
        terms = []
        for index, degrees in enumerate(exponents):
            _number(coefficients[state][index])
            pair = _interval((lower[state][index], upper[state][index]))
            if any(float.fromhex(value) != 0.0 for value in pair):
                terms.append({"degrees": list(degrees), "coefficient": pair})
        components.append({"remainder": _interval(remainder[state]), "terms": terms})
    result = {"domain": domains, "components": components}
    if variables is not None:
        result["variables"] = list(variables)
    return result


def observe_step(models, *, plant, step, h, lane=0):
    """Return four endpoint/tube x/y rows on the fixed binary64-step clock."""
    h = _number(h)
    if type(step) is not int or step < 1 or h <= 0:
        raise ValueError("positive fixed step and one-based accepted step required")
    start, end = (step - 1) * Fraction(h), step * Fraction(h)
    rows = []
    for view in ("endpoint", "tube"):
        bounds = measure(models[view])
        if len(bounds) != 2:
            raise ValueError("observer requires both x and y")
        for coordinate, (lo, hi) in zip(("x", "y"), bounds):
            _interval((lo, hi))
            width = hi - lo
            if not math.isfinite(width):
                raise ValueError("nonfinite observed width")
            rows.append(dict(plant=plant, lane=lane, step=step, view=view,
                             coordinate=coordinate, h_hex=h.hex(),
                             t_start_exact=str(start), t_end_exact=str(end),
                             time=float(end), lo=lo, hi=hi,
                             lo_hex=lo.hex(), hi_hex=hi.hex(), width=width,
                             observer="EXACT_FRACTION_COMPLETE_TM"))
    return rows


def _self_check():
    # A non-point coefficient and a crossing-zero square must retain their range.
    model = canonical_model([[1.1], [1.0]], [[0., 0.], [0., 0.]], [[2]], [[-1., 2.]],
                            coefficient_bounds=([[1.0], [-2.0]], [[1.2], [-1.0]]))
    result = measure(model)
    assert Fraction(result[0][0]) <= 0 and Fraction(result[0][1]) >= 4 * Fraction(1.2)
    assert result[1] == [-8.0, 0.0]
    rows = observe_step({"endpoint": model, "tube": model},
                        plant="check", step=3, h=.01, lane=31)
    assert len(rows) == 4 and {row["coordinate"] for row in rows} == {"x", "y"}
    assert all(Fraction(row["t_end_exact"]) == 3 * Fraction(.01) for row in rows)
    point = canonical_model([[1.1], [0.]], [[0., 0.], [0., 0.]], [[1]], [[1.1, 1.1]])
    lo, hi = measure(point)[0]
    assert Fraction(lo) <= Fraction(1.1) ** 2 <= Fraction(hi)
    print("exact observer self-check passed")


if __name__ == "__main__":
    _self_check()
