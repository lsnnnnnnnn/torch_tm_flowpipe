"""Experimental full-degree composition with outward binary64 coefficients.

Inputs denote exact stored finite binary64 numbers. Every nontrivial addition
or multiplication receives one nextafter step on each outward endpoint. This
encloses correctly rounded IEEE-754 operations, including cancellation and
gradual underflow. Exact 0 and +/-1 identities do not introduce roundoff.
No coefficient cutoff, truncation, point-only approximation or GPU is used.
If an intermediate or outward endpoint overflows, export fails explicitly.

This is an independent exporter; it does not replace the Fraction reference.
"""
import math
from functools import lru_cache

_ZERO = (0.0, 0.0)
_ONE = (1.0, 1.0)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('expected an exact binary64 number')
    represented = float(value)
    if not math.isfinite(represented) or represented != value:
        raise ValueError('coefficient is not a finite exactly representable binary64')
    return represented


def _interval(lo, hi=None):
    lo = _number(lo)
    hi = lo if hi is None else _number(hi)
    if lo > hi:
        raise ValueError('inverted interval')
    return lo, hi


def _outward_pair(lo, hi):
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise OverflowError('nonfinite intermediate during outward composition')
    lower = math.nextafter(lo, -math.inf)
    upper = math.nextafter(hi, math.inf)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise OverflowError('outward composition endpoint exceeds finite binary64')
    return lower, upper


def _add(a, b):
    if a == _ZERO:
        return b
    if b == _ZERO:
        return a
    if a[0] == a[1] and b[0] == b[1] and a[0] == -b[0]:
        return _ZERO
    return _outward_pair(a[0] + b[0], a[1] + b[1])


def _multiply(a, b):
    if a == _ZERO or b == _ZERO:
        return _ZERO
    if a == _ONE:
        return b
    if b == _ONE:
        return a
    if a == (-1.0, -1.0):
        return -b[1], -b[0]
    if b == (-1.0, -1.0):
        return -a[1], -a[0]
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return _outward_pair(min(values), max(values))


def _accumulate(out, degrees, coefficient):
    value = _add(out.get(degrees, _ZERO), coefficient)
    if value == _ZERO:
        out.pop(degrees, None)
    else:
        out[degrees] = value


@lru_cache(maxsize=16384)
def _sum_degrees(a, b):
    """Cache only immutable integer support metadata, never coefficients."""
    return tuple(x + y for x, y in zip(a, b))


def _product(left, right):
    out = {}
    for a, av in left.items():
        for b, bv in right.items():
            degrees = _sum_degrees(a, b)
            _accumulate(out, degrees, _multiply(av, bv))
    return out


def outward_composition(pre, pre_rem, pre_exponents, tmv, tmv_rem, tmv_exponents):
    """Same full polynomial substitution as Fraction exact_composition."""
    n = len(tmv)
    zero = (0,) * (n + 1)
    pre_exponents = [tuple(e) for e in pre_exponents]
    tmv_exponents = [tuple(e) for e in tmv_exponents]
    for exponents in [pre_exponents, tmv_exponents]:
        if not exponents or any(len(e) != n + 1 or any(type(v) is not int or v < 0 for v in e)
                                for e in exponents):
            raise ValueError('invalid exponent support')
        if len(set(exponents)) != len(exponents):
            raise ValueError('duplicate support')
    if any(e[0] for e in tmv_exponents):
        raise ValueError('history map must be time free')
    if not (len(pre) == len(pre_rem) == len(tmv_rem) == n):
        raise ValueError('component shape mismatch')
    histories = []
    for coefficients, remainder in zip(tmv, tmv_rem):
        if len(coefficients) != len(tmv_exponents) or len(remainder) != 2:
            raise ValueError('history coefficient/support mismatch')
        polynomial = {}
        for degrees, value in zip(tmv_exponents, coefficients):
            _accumulate(polynomial, degrees, _interval(value))
        _accumulate(polynomial, zero, _interval(*remainder))
        histories.append(polynomial)
    powers = []
    for i, history in enumerate(histories):
        maximum = max(e[i + 1] for e in pre_exponents)
        values = [{zero: _ONE}]
        for _ in range(maximum):
            values.append(_product(values[-1], history))
        powers.append(values)
    images, components = {}, []
    for coefficients, remainder in zip(pre, pre_rem):
        if len(coefficients) != len(pre_exponents) or len(remainder) != 2:
            raise ValueError('pre coefficient/support mismatch')
        result = {}
        _accumulate(result, zero, _interval(*remainder))
        for degrees, raw in zip(pre_exponents, coefficients):
            value = _number(raw)
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
                _accumulate(result, shifted, _multiply((value, value), coefficient))
        components.append(result)
    return components


def canonical_models(components, step):
    step = _number(step)
    if step <= 0:
        raise ValueError('step must be positive')
    n = len(components)
    rows = [{'remainder': [0.0, 0.0], 'terms': [
        {'degrees': list(degrees), 'coefficient': list(_interval(*value))}
        for degrees, value in sorted(component.items())]} for component in components]
    variables = ['tau', 'ux', 'uy'] if n == 2 else ['tau'] + [f'u{i}' for i in range(n)]
    return {view: {'domain': [time_domain] + [[-1.0, 1.0] for _ in range(n)],
                   'variables': variables, 'components': rows}
            for view, time_domain in [('endpoint', [step, step]), ('tube', [0.0, step])]}
