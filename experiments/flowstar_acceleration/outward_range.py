"""Same monomial interval range policy as common.measure, with binary64 bounds.

Stored finite binary64 coefficients and domains denote exact real endpoints.
Every nontrivial add/multiply is enclosed using nextafter in each direction.
Integer powers use multiplication, never platform libm pow. Exact 0 and +/-1
identities and known power signs avoid gratuitous widening. Gradual underflow
is enclosed; nonfinite inputs, intermediate overflow and outward overflow are
rejected. Only nonnegative integer exponents are polynomial support.

No terms, dependencies, coordinates, domains or bounding policy are changed.
This independently selectable backend does not replace the Fraction oracle.
"""
from .outward_export import _add, _interval, _multiply, _ONE

OBSERVER = 'OUTWARD_BINARY64_COMPLETE_TM'


def _pair(value):
    if len(value) != 2:
        raise ValueError('expected an interval pair')
    return _interval(*(float.fromhex(x) if isinstance(x, str) else x for x in value))


def _point_power(value, exponent):
    """Enclose one exact binary64 scalar raised to an integer exponent."""
    if exponent == 0:
        return _ONE
    if value == 0.0:
        return 0.0, 0.0
    if abs(value) == 1.0:
        sign = -1.0 if value < 0 and exponent % 2 else 1.0
        return sign, sign
    result, base, n = _ONE, (abs(value), abs(value)), exponent
    while n:
        if n & 1:
            lo, hi = _multiply(result, base)
            result = max(0.0, lo), hi
        n >>= 1
        if n:
            lo, hi = _multiply(base, base)
            base = max(0.0, lo), hi
    return (-result[1], -result[0]) if value < 0 and exponent % 2 else result


def power(domain, exponent):
    """Enclose exactly the same odd/even endpoint hull as common.power."""
    if type(exponent) is not int or exponent < 0:
        raise ValueError('invalid polynomial exponent')
    domain = _interval(*domain)
    if exponent == 0:
        return _ONE
    if exponent == 1:
        return domain
    left, right = (_point_power(x, exponent) for x in domain)
    if exponent % 2:
        return left[0], right[1]
    lower = 0.0 if domain[0] <= 0 <= domain[1] else min(left[0], right[0])
    return lower, max(left[1], right[1])


def measure(model):
    domains = [_pair(pair) for pair in model['domain']]
    powers, result = {}, []
    for component in model['components']:
        total = _pair(component['remainder'])
        for term in component['terms']:
            bounds = _pair(term['coefficient'])
            degrees = term['degrees']
            if len(degrees) != len(domains):
                raise ValueError('monomial dimension mismatch')
            for axis, (domain, exponent) in enumerate(zip(domains, degrees)):
                # Validate before lookup: bool and int keys otherwise collide.
                if type(exponent) is not int or exponent < 0:
                    raise ValueError('invalid polynomial exponent')
                key = axis, exponent
                if key not in powers:
                    powers[key] = power(domain, exponent)
                bounds = _multiply(bounds, powers[key])
            total = _add(total, bounds)
        result.append(list(total))
    return result
