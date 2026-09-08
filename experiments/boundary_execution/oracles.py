"""Exact rational term oracle, independent of the packed torch arithmetic."""
from fractions import Fraction
import gzip
import json

import torch
from torch_tm_flowpipe import Interval, Polynomial, evaluate_interval_normal
from torch_tm_flowpipe.accepted_boundary_sr import _interval_polynomial_range
from torch_tm_flowpipe.packed_boundary_range import make_plan, packed_boundary_execution
from experiments.xiangru_adoption.common import f, product, power


def interval_bits(value):
    return [float(value.lo).hex(), float(value.hi).hex()]


def verify_range_record(record):
    exponents = tuple(tuple(e) for e in record['exponents'])
    dimension = len(record['domain'])
    assert 0 <= dimension <= 3 and len(exponents) <= 512
    assert all(len(e) == dimension and all(type(p) is int and 0 <= p <= 12 for p in e) for e in exponents)
    assert len(record['coefficients']) == len(exponents) and len(set(exponents)) == len(exponents)
    lows = torch.tensor([float.fromhex(c[0]) for c in record['coefficients']], dtype=torch.float64).reshape(1, 1, -1)
    highs = torch.tensor([float.fromhex(c[1]) for c in record['coefficients']], dtype=torch.float64).reshape(1, 1, -1)
    domain_lo = torch.tensor([[float.fromhex(c[0]) for c in record['domain']]], dtype=torch.float64)
    domain_hi = torch.tensor([[float.fromhex(c[1]) for c in record['domain']]], dtype=torch.float64)
    normal = record['kind'] == 'normal'
    states = tuple(record['state_variables']) if normal else None
    actual = make_plan(exponents, dimension, states, record['time_variable']).evaluate(
        lows, highs, domain_lo, domain_hi, return_terms=True)
    lo, hi, term_lo, term_hi = actual
    assert [float(lo.item()).hex(), float(hi.item()).hex()] == record['bounds']
    domains = [tuple(map(f, bounds)) for bounds in record['domain']]
    lower, upper = Fraction(0), Fraction(0)
    for index, exponent in enumerate(exponents):
        interval = tuple(map(f, record['coefficients'][index]))
        for variable, degree in enumerate(exponent):
            if degree and (states is None or variable not in states):
                interval = product(interval, power(domains[variable], degree))
        if normal:
            factor = (-Fraction(1) if any(exponent[v] % 2 for v in states)
                      else Fraction(0) if any(exponent[v] for v in states) else Fraction(1), Fraction(1))
            interval = product(interval, factor)
        assert Fraction(float(term_lo[0, 0, index])) <= interval[0] <= interval[1] <= Fraction(float(term_hi[0, 0, index]))
        lower += interval[0]; upper += interval[1]
    assert Fraction(float(lo.item())) <= lower <= upper <= Fraction(float(hi.item()))
    domain = [Interval(float.fromhex(a), float.fromhex(b)) for a, b in record['domain']]
    # Compare both real dispatches to the captured original range as well.
    for enabled in (False, True):
        with packed_boundary_execution(enabled):
            if record['kind'] == 'interval':
                coefficients = {e: Interval(float.fromhex(a), float.fromhex(b))
                                for e, (a, b) in zip(exponents, record['coefficients'])}
                value = _interval_polynomial_range(coefficients, domain, reference=Interval.zero())
            else:
                assert all(a == b for a, b in record['coefficients'])
                poly = Polynomial({e: float.fromhex(c[0]) for e, c in zip(exponents, record['coefficients'])}, dimension)
                value = (evaluate_interval_normal(poly, domain, state_var_indices=states, time_var_index=record['time_variable'])
                         if normal else poly.evaluate_interval(domain))
            assert interval_bits(value) == record['bounds']
    return len(exponents)


def verify_range_file(path):
    count = terms = 0
    with gzip.open(path, 'rt') as source:
        for line in source:
            assert len(line) < 1_000_000 and count < 2000
            terms += verify_range_record(json.loads(line))
            count += 1
    return {'range_calls': count, 'rationally_checked_terms': terms}
