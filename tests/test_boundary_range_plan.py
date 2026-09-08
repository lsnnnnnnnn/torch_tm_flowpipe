"""Independent rational interval oracle and execution-order checks."""
from fractions import Fraction as F
from itertools import product
import math

import pytest
import torch

from torch_tm_flowpipe import Interval, Polynomial, evaluate_interval_normal
from torch_tm_flowpipe.packed_boundary_range import make_plan, evaluate_polynomial


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def same_bits(a, b):
    assert a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.int64), b.contiguous().view(torch.int64))


def reference(coeff_lo, coeff_hi, domain_lo, domain_hi, exponents, *, states=None, time_variable=None):
    batch, outputs, terms = coeff_lo.shape
    lows, highs = torch.empty_like(coeff_lo), torch.empty_like(coeff_hi)
    total_lo, total_hi = torch.zeros((batch, outputs), dtype=torch.float64), torch.zeros((batch, outputs), dtype=torch.float64)
    for b in range(batch):
        domain = [Interval(lo, hi) for lo, hi in zip(domain_lo[b], domain_hi[b])]
        for o in range(outputs):
            total = Interval.zero()
            for i, e in enumerate(exponents):
                term = Interval(coeff_lo[b, o, i], coeff_hi[b, o, i])
                if states is None:
                    for p, d in zip(e, domain):
                        if p:
                            term = term * d.pow_int(p)
                else:
                    if time_variable is not None and e[time_variable]:
                        term = term * domain[time_variable].pow_int(e[time_variable])
                    factor_lo = (-1. if any(e[v] % 2 for v in states)
                                 else 0. if any(e[v] for v in states) else 1.)
                    term = term * Interval(factor_lo, 1.)
                    for v, p in enumerate(e):
                        if p and v not in states and v != time_variable:
                            term = term * domain[v].pow_int(p)
                lows[b, o, i], highs[b, o, i] = term.lo, term.hi
                total = total + term
            total_lo[b, o], total_hi[b, o] = total.lo, total.hi
    return total_lo, total_hi, lows, highs


def exact_multiply(a, b):
    products = [x * y for x in a for y in b]
    return min(products), max(products)


def exact_power(bounds, power):
    lo, hi = bounds
    if power % 2:
        return lo ** power, hi ** power
    return (F(0) if lo <= 0 <= hi else min(abs(lo), abs(hi)) ** power,
            max(abs(lo), abs(hi)) ** power)


def oracle(coeff_lo, coeff_hi, domain_lo, domain_hi, exponents, actual, *, states=None, time_variable=None):
    """Compute natural term intervals over exact stored binary64 rationals."""
    total_lo, total_hi, terms_lo, terms_hi = actual
    for b in range(coeff_lo.shape[0]):
        domain = [(F(float(lo)), F(float(hi))) for lo, hi in zip(domain_lo[b], domain_hi[b])]
        for o in range(coeff_lo.shape[1]):
            lower, upper = F(0), F(0)
            for i, e in enumerate(exponents):
                term = F(float(coeff_lo[b, o, i])), F(float(coeff_hi[b, o, i]))
                for v, p in enumerate(e):
                    if p and (states is None or v not in states):
                        term = exact_multiply(term, exact_power(domain[v], p))
                if states is not None:
                    factor = (-F(1) if any(e[v] % 2 for v in states)
                              else F(0) if any(e[v] for v in states) else F(1), F(1))
                    term = exact_multiply(term, factor)
                assert F(float(terms_lo[b, o, i])) <= term[0] <= term[1] <= F(float(terms_hi[b, o, i]))
                lower, upper = lower + term[0], upper + term[1]
            assert F(float(total_lo[b, o])) <= lower <= upper <= F(float(total_hi[b, o]))


@pytest.mark.parametrize('order', [4, 6])
@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('states,time_variable', [(None, None), ((0, 1), 2), ((1,), 0)])
def test_real_orders_distinct_batches_outputs_and_exact_term_oracle(order, batch, states, time_variable):
    exponents = tuple(e for e in product(range(order + 1), repeat=3) if sum(e) <= order)
    coeff_lo = tensor([[[((-1.) ** (i + o + b)) * (i + 1) / (17. + 3 * b + o)
                        for i in range(len(exponents))] for o in range(2)] for b in range(batch)])
    coeff_hi = coeff_lo + tensor([[[0. if i % 2 else .00001 * (b + o + 1)
                                  for i in range(len(exponents))] for o in range(2)] for b in range(batch)])
    domain_lo = tensor([[-.75, .1, -.01], [.2, -1.4, .002]][:batch])
    domain_hi = tensor([[1.25, .9, .02], [.75, .3, .1]][:batch])
    plan = make_plan(exponents, 3, states, time_variable)
    expected = reference(coeff_lo, coeff_hi, domain_lo, domain_hi, exponents, states=states, time_variable=time_variable)
    actual = plan.evaluate(coeff_lo, coeff_hi, domain_lo, domain_hi, return_terms=True)
    for a, b in zip(actual, expected):
        same_bits(a, b)
    oracle(coeff_lo, coeff_hi, domain_lo, domain_hi, exponents, actual, states=states, time_variable=time_variable)
    for b in range(batch):
        one = plan.evaluate(coeff_lo[b:b+1], coeff_hi[b:b+1], domain_lo[b:b+1], domain_hi[b:b+1], return_terms=True)
        for x, y in zip(actual, one):
            same_bits(x[b:b+1], y)


@pytest.mark.parametrize('values,domain', [
    ([1e16, 1., -1e16], (1., 1.)),
    ([math.ulp(0.), -math.ulp(0.), 0.], (-.5, .5)),
    ([-0., 0., -0.], (-0., 0.)),
    ([0., 0., 0.], (-1., 1.)),
    ([1., -2., 3.], (-.02, -.01)),
    ([1., -2., 3.], (0., .01)),
    ([1., -2., 3.], (.02, .1)),
])
def test_cancellation_zeros_subnormal_and_time_values(values, domain):
    exponents = ((0,), (1,), (6,))
    coefficients = tensor([[values]])
    lo, hi = tensor([[domain[0]]]), tensor([[domain[1]]])
    actual = make_plan(exponents, 1).evaluate(coefficients, coefficients, lo, hi, return_terms=True)
    for a, b in zip(actual, reference(coefficients, coefficients, lo, hi, exponents)):
        same_bits(a, b)
    oracle(coefficients, coefficients, lo, hi, exponents, actual)


@pytest.mark.parametrize('batch,outputs', [(1, 1), (2, 3)])
def test_empty_support_returns_exact_zero(batch, outputs):
    coefficients = torch.zeros((batch, outputs, 0), dtype=torch.float64)
    domain_lo, domain_hi = -torch.ones((batch, 2), dtype=torch.float64), torch.ones((batch, 2), dtype=torch.float64)
    result = make_plan((), 2).evaluate(coefficients, coefficients, domain_lo, domain_hi)
    for actual in result:
        same_bits(actual, torch.zeros((batch, outputs), dtype=torch.float64))


@pytest.mark.parametrize('coefficient,domain,exponent', [
    (1e308, (1., 2.), 6), (1., (-1e100, 1e100), 6),
    (float('inf'), (1., 2.), 1), (-float('inf'), (-2., -1.), 1),
    (float('inf'), (0., 1.), 1), (float('nan'), (1., 2.), 1),
    (1., (float('nan'), 1.), 2), (1., (2., 1.), 1),
    (1., (-float('inf'), float('inf')), 2),
])
def test_overflow_nan_inf_matches_reference_success_or_failure(coefficient, domain, exponent):
    coeff = tensor([[[coefficient]]])
    lo, hi = tensor([[domain[0]]]), tensor([[domain[1]]])
    exponents = ((exponent,),)
    try:
        expected = reference(coeff, coeff, lo, hi, exponents)
    except ValueError:
        with pytest.raises(ValueError):
            make_plan(exponents, 1).evaluate(coeff, coeff, lo, hi, return_terms=True)
    else:
        actual = make_plan(exponents, 1).evaluate(coeff, coeff, lo, hi, return_terms=True)
        for a, b in zip(actual, expected):
            same_bits(a, b)


def test_plan_has_no_numeric_cache_and_outputs_own_storage():
    plan = make_plan(((1,), (4,)), 1)
    coeff = tensor([[[1., -2.]], [[3., 4.]]])
    lo, hi = tensor([[-.7], [.2]]), tensor([[1.1], [.6]])
    saved = [x.clone() for x in (coeff, lo, hi)]
    first = plan.evaluate(coeff, coeff, lo, hi, return_terms=True)
    for a, b in zip((coeff, lo, hi), saved):
        same_bits(a, b)
    old = [x.clone() for x in first]
    coeff[0, 0, 0] = 9.
    lo[1, 0] = .1
    second = plan.evaluate(coeff, coeff, lo, hi, return_terms=True)
    assert not torch.equal(first[1], second[1])
    for a, b in zip(first, old):
        same_bits(a, b)
    first[2].zero_()
    for a, b in zip(second, plan.evaluate(coeff, coeff, lo, hi, return_terms=True)):
        same_bits(a, b)
    assert set(vars(plan)) == {'exponents', 'n_vars', 'stages'}
    assert all(not isinstance(value, torch.Tensor) for value in vars(plan).values())


@pytest.mark.parametrize('normal', [False, True])
def test_native_adapter_preserves_support_order_and_scalar_power_route(normal):
    poly = Polynomial({(0, 0): 2., (1, 2): -3., (4, 1): .25, (0, 6): 1e-40}, 2)
    domain = [Interval(-.7, 1.2), Interval(-.01, .03)]
    expected = (evaluate_interval_normal(poly, domain, state_var_indices=[0], time_var_index=1)
                if normal else poly.evaluate_interval(domain))
    before = {e: c.clone() for e, c in poly.terms.items()}
    actual = evaluate_polynomial(poly, domain, normal=normal, state_variables=(0,), time_variable=1)
    same_bits(expected.lo, actual.lo)
    same_bits(expected.hi, actual.hi)
    for exponent, coefficient in poly.terms.items():
        same_bits(coefficient, before[exponent])


def test_constant_without_variables_keeps_one_ordered_outward_add():
    poly = Polynomial.constant(2., 0)
    original = poly.evaluate_interval([])
    packed = evaluate_polynomial(poly, [])
    same_bits(original.lo, packed.lo)
    same_bits(original.hi, packed.hi)


def test_adapter_declines_mixed_dtype_and_tensor_coefficients_for_original_fallback():
    poly = Polynomial({(1,): torch.tensor(1., dtype=torch.float32)}, 1)
    assert evaluate_polynomial(poly, [Interval(-1., 2.)]) is NotImplemented
    poly = Polynomial({(1,): tensor([1., 2.])}, 1)
    assert evaluate_polynomial(poly, [Interval(-1., 2.)]) is NotImplemented


def test_opt_in_default_nesting_exceptions_and_independent_prepared_switch():
    from torch_tm_flowpipe import packed_boundary_execution
    from torch_tm_flowpipe.packed_boundary_range import is_enabled
    from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay, is_enabled as prepared_enabled
    assert not is_enabled() and not prepared_enabled()
    with prepared_remainder_replay(True):
        assert not is_enabled()
        with packed_boundary_execution(True):
            assert is_enabled() and prepared_enabled()
            with packed_boundary_execution(False):
                assert not is_enabled() and prepared_enabled()
            assert is_enabled()
        assert not is_enabled() and prepared_enabled()
    with pytest.raises(RuntimeError):
        with packed_boundary_execution(True):
            raise RuntimeError('restore on failure')
    assert not is_enabled() and not prepared_enabled()
    for invalid in (1, None, 'yes'):
        with pytest.raises(TypeError):
            with packed_boundary_execution(invalid):
                pass


@pytest.mark.parametrize('kind', ['standard', 'normal', 'power_table', 'float32', 'vector_coefficients'])
def test_real_dispatch_and_required_fallbacks_preserve_bits(kind):
    from torch_tm_flowpipe import packed_boundary_execution
    from torch_tm_flowpipe.packed_boundary_range import make_plan
    domain = [Interval(-.8, 1.3), Interval(-.01, .03)]
    coefficient = (torch.tensor(2., dtype=torch.float32) if kind == 'float32'
                   else tensor([2., -3.]) if kind == 'vector_coefficients' else tensor(2.))
    poly = Polynomial({(1, 2): coefficient, (0, 0): 1.}, 2)

    def evaluate():
        if kind in ('normal', 'power_table'):
            return evaluate_interval_normal(poly, domain, time_var_index=1,
                step_exp_table={2: Interval(.0001, .001)} if kind == 'power_table' else None)
        return poly.evaluate_interval(domain)

    with packed_boundary_execution(False):
        expected = evaluate()
    make_plan.cache_clear()
    with packed_boundary_execution(True):
        actual = evaluate()
    same_bits(expected.lo, actual.lo)
    same_bits(expected.hi, actual.hi)
    assert bool(make_plan.cache_info().currsize) == (kind in ('standard', 'normal'))
