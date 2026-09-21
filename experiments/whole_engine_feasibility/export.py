"""Exact, offline observation of a sparse factored Taylor model.

This does not alter the solver. Fraction interval-coefficient substitution
encloses pre(t, tmv(r) + tmv_rem) + pre_rem, retaining all degrees.
No safety._composed_tm or rounded tensor polynomial products are used.
"""
from fractions import Fraction
import math

_ZERO = (Fraction(0), Fraction(0))
_ONE = (Fraction(1), Fraction(1))


def _interval(lo, hi=None):
    lo = Fraction(lo)
    hi = lo if hi is None else Fraction(hi)
    if lo > hi:
        raise ValueError("inverted coefficient interval")
    return lo, hi


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _multiply(a, b):
    if a[0] == a[1] and b[0] == b[1]:
        product = a[0] * b[0]
        return product, product
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(values), max(values)


def _accumulate(out, degrees, coefficient):
    value = _add(out.get(degrees, _ZERO), coefficient)
    if value == _ZERO:
        out.pop(degrees, None)
    else:
        out[degrees] = value


def _product(left, right):
    out = {}
    for a, av in left.items():
        for b, bv in right.items():
            degrees = tuple(x + y for x, y in zip(a, b))
            _accumulate(out, degrees, _multiply(av, bv))
    return out


def exact_composition(pre, pre_rem, pre_exponents, tmv, tmv_rem, tmv_exponents):
    """Compose Python coefficient lists using exact rational interval arithmetic.

    Exponents include local time first, then shared normalized symbols.
    Returns one dict[exponent_tuple, (Fraction lower, Fraction upper)] per row.
    """
    n = len(tmv)
    zero = (0,) * (n + 1)
    pre_exponents = [tuple(map(int, e)) for e in pre_exponents]
    tmv_exponents = [tuple(map(int, e)) for e in tmv_exponents]
    if any(e[0] != 0 for e in tmv_exponents):
        raise ValueError("history map must be time free")
    histories = []
    for coefficients, remainder in zip(tmv, tmv_rem):
        polynomial = {}
        for degrees, value in zip(tmv_exponents, coefficients):
            _accumulate(polynomial, degrees, _interval(value))
        _accumulate(polynomial, zero, _interval(*remainder))
        histories.append(polynomial)
    powers = []
    for i, history in enumerate(histories):
        maximum = max((e[i + 1] for e in pre_exponents), default=0)
        values = [{zero: _ONE}]
        for _ in range(maximum):
            values.append(_product(values[-1], history))
        powers.append(values)
    images = {}
    components = []
    for coefficients, remainder in zip(pre, pre_rem):
        result = {}
        _accumulate(result, zero, _interval(*remainder))
        for degrees, value in zip(pre_exponents, coefficients):
            if value == 0:
                continue
            spatial = degrees[1:]
            if spatial not in images:
                image = {zero: _ONE}
                for i, power in enumerate(spatial):
                    image = _product(image, powers[i][power])
                images[spatial] = image
            for image_degrees, coefficient in images[spatial].items():
                shifted = (image_degrees[0] + degrees[0], *image_degrees[1:])
                _accumulate(result, shifted, _multiply(_interval(value), coefficient))
        components.append(result)
    return components


def _outward(value, lower):
    rounded = float(value)
    if not math.isfinite(rounded):
        raise OverflowError("canonical coefficient exceeds finite binary64")
    represented = Fraction(rounded)
    if (lower and represented > value) or (not lower and represented < value):
        rounded = math.nextafter(rounded, -math.inf if lower else math.inf)
    if not math.isfinite(rounded):
        raise OverflowError("outward coefficient exceeds finite binary64")
    return rounded


def canonical_models(components, step):
    """Use one complete composition for both endpoint and whole-step tube."""
    step = float(step)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be positive and finite")
    rows = [
        {
            "remainder": [0.0, 0.0],
            "terms": [
                {"degrees": list(degrees),
                 "coefficient": [_outward(value[0], True), _outward(value[1], False)]}
                for degrees, value in sorted(component.items())
            ],
        }
        for component in components
    ]
    n = len(rows)
    variables = ["tau", "ux", "uy"] if n == 2 else ["tau"] + [f"u{i}" for i in range(n)]
    return {
        view: {
            "domain": [time_domain] + [[-1.0, 1.0] for _ in range(n)],
            "variables": variables,
            "components": rows,
        }
        for view, time_domain in (
            ("endpoint", [step, step]), ("tube", [0.0, step])
        )
    }


def export_sparse_state(state, tables, step, *, lane=0):
    """CPU-only exact export; call outside solve/kernel timing.

    The solver stores its pre model on full-basis IDs and its history map on
    spatial-basis IDs. Reusing one ID convention for both silently changes the
    map, so resolve them separately before substitution.
    """
    all_exponents = tables.exponents.detach().cpu().tolist()
    spatial_ids = tables.spatial_index.detach().cpu().tolist()
    pre_exponents = [all_exponents[i] for i in state.pre_sup.ids]
    tmv_exponents = [all_exponents[spatial_ids[i]] for i in state.tmv_sup.ids]
    components = exact_composition(
        state.pre[lane].detach().cpu().tolist(),
        state.pre_rem[lane].detach().cpu().tolist(),
        pre_exponents,
        state.tmv[lane].detach().cpu().tolist(),
        state.tmv_rem[lane].detach().cpu().tolist(),
        tmv_exponents,
    )
    return canonical_models(components, step)
