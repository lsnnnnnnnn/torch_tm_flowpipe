"""Small read-only evaluator for exported complete polynomial Taylor models.

All arithmetic below is exact rational arithmetic on stored binary64 bounds.
There is no coefficient cutoff or truncation here. Final conversion is outward.
Variables retain each exporter's own legal coordinates; no coefficients from
different tools are subtracted or assumed to denote the same internal variable.
"""
from fractions import Fraction
import math

def f(value):
    return Fraction(float.fromhex(value) if isinstance(value, str) else value)

def product(a, b):
    vals = [x*y for x in a for y in b]
    return min(vals), max(vals)

def power(a, exponent):
    if exponent == 0: return Fraction(1), Fraction(1)
    lo, hi = a
    if exponent % 2: return lo**exponent, hi**exponent
    upper = max(lo**exponent, hi**exponent)
    return (Fraction() if lo <= 0 <= hi else min(lo**exponent, hi**exponent)), upper

def outward(v, upper):
    result = float(v)
    if upper and Fraction(result) < v: result = math.nextafter(result, math.inf)
    if not upper and Fraction(result) > v: result = math.nextafter(result, -math.inf)
    return result

def measure(model):
    domains = [tuple(map(f, pair)) for pair in model['domain']]
    result = []
    for component in model['components']:
        lo, hi = map(f, component['remainder'])
        for term in component['terms']:
            bounds = tuple(map(f, term['coefficient']))
            assert len(term['degrees']) == len(domains)
            for d, exponent in zip(domains, term['degrees']):
                bounds = product(bounds, power(d, exponent))
            lo += bounds[0]
            hi += bounds[1]
        result.append([outward(lo, False), outward(hi, True)])
    return result

def from_existing_canonical(records, prefix):
    """Reuse brusselator_canonical_exchange._append_tmv's lossless exporter."""
    n = int(records[f'{prefix}.domain_count'])
    domains = [[records[f'{prefix}.domain.{i}.{side}'] for side in ['lo', 'hi']] for i in range(n)]
    result = {'domain': domains, 'variables': records[f'{prefix}.variable_order'].split(','), 'components': []}
    for i in range(int(records[f'{prefix}.component_count'])):
        base = f'{prefix}.component.{i}'
        component = {'remainder': [records[f'{base}.ordinary_remainder.{side}'] for side in ['lo','hi']], 'terms': []}
        for j in range(int(records[f'{base}.term_count'])):
            key = f'{base}.term.{j}'
            c = records[f'{key}.coefficient_hex']
            component['terms'].append({'degrees': list(map(int, records[f'{key}.exponents'].split(','))), 'coefficient': [c,c]})
        result['components'].append(component)
    return result
