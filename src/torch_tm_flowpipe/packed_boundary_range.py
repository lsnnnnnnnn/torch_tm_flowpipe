"""Ordered tensor execution of the existing sparse interval range operations.

Only immutable support metadata is cached. Variable powers retain the scalar
Interval route; terms are accumulated sequentially with outward rounding.
The production dispatcher is opt-in and defaults to the original evaluator.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache

import torch
from .interval import Interval


_ENABLED = ContextVar('packed_boundary_execution', default=False)
_REQUEST_DISPATCH = ContextVar('range_request_dispatch', default=None)
_TABLE_REQUEST_DISPATCH = ContextVar('range_table_request_dispatch', default=None)


@contextmanager
def packed_boundary_execution(enabled: bool = True):
    """Enable ordered packed sparse ranges for this context; no numeric cache."""
    if type(enabled) is not bool:
        raise TypeError('packed_boundary_execution requires a bool')
    token = _ENABLED.set(enabled)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def is_enabled() -> bool:
    return _ENABLED.get()


def _check(lo, hi):
    if bool(torch.any(torch.isnan(lo))) or bool(torch.any(torch.isnan(hi))):
        raise ValueError('interval bounds must not be NaN')
    if bool(torch.any(lo > hi)):
        raise ValueError('invalid interval with lo > hi')


def _multiply(a_lo, a_hi, b_lo, b_hi):
    candidates = torch.stack(torch.broadcast_tensors(
        a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi), dim=0)
    lo = torch.nextafter(torch.min(candidates, dim=0).values, torch.full_like(a_lo, -torch.inf))
    hi = torch.nextafter(torch.max(candidates, dim=0).values, torch.full_like(a_hi, torch.inf))
    _check(lo, hi)
    return lo, hi


@dataclass(frozen=True)
class RangePlan:
    exponents: tuple
    n_vars: int
    stages: tuple

    def evaluate(self, coefficients_lo, coefficients_hi, domain_lo, domain_hi, *, return_terms=False):
        """Evaluate [B, output, terms] intervals in their supplied term order.

        Independent terms and lanes share tensor operations. Scalar domain
        powers and the sequential outward accumulation retain reference order.
        Optional term results support per-operation containment verification.
        """
        inputs = (coefficients_lo, coefficients_hi, domain_lo, domain_hi)
        if any(x.device.type != 'cpu' or x.dtype != torch.float64 for x in inputs):
            raise TypeError('ordered range requires CPU binary64')
        if coefficients_lo.ndim != 3 or coefficients_lo.shape[-1] != len(self.exponents):
            raise ValueError('coefficients must be [batch, output, terms]')
        batch, outputs, terms = coefficients_lo.shape
        if batch < 1 or outputs < 1 or coefficients_hi.shape != coefficients_lo.shape:
            raise ValueError('coefficient interval shape mismatch')
        if domain_lo.shape != (batch, self.n_vars) or domain_hi.shape != domain_lo.shape:
            raise ValueError('domain must be [batch, variables]')
        _check(coefficients_lo, coefficients_hi)
        _check(domain_lo, domain_hi)
        # Only local storage is written. No numerical input survives this call.
        term_lo, term_hi = coefficients_lo.clone(), coefficients_hi.clone()
        powers = {}
        for kind, variable, indices, values in self.stages:
            selected = list(indices)
            if kind == 'power':
                for power in set(values):
                    if (variable, power) not in powers:
                        # Retain the original scalar pow kernel, including
                        # its odd/even rules and outward rounding positions.
                        per_batch = [Interval(domain_lo[b, variable], domain_hi[b, variable]).pow_int(power)
                                     for b in range(batch)]
                        powers[variable, power] = (torch.stack([x.lo for x in per_batch]),
                                                  torch.stack([x.hi for x in per_batch]))
                lo = torch.stack([powers[variable, power][0] for power in values], dim=-1)[:, None, :]
                hi = torch.stack([powers[variable, power][1] for power in values], dim=-1)[:, None, :]
            else:
                lo = torch.tensor(values, dtype=torch.float64).view(1, 1, -1)
                hi = torch.ones_like(lo)
            product_lo, product_hi = _multiply(term_lo[..., selected], term_hi[..., selected], lo, hi)
            term_lo[..., selected], term_hi[..., selected] = product_lo, product_hi
        total_lo = torch.zeros((batch, outputs), dtype=torch.float64)
        total_hi = torch.zeros_like(total_lo)
        negative_inf, positive_inf = torch.full_like(total_lo, -torch.inf), torch.full_like(total_hi, torch.inf)
        # No parallel sum, cumsum, reassociation, or zero-term elision.
        for index in range(terms):
            total_lo = torch.nextafter(total_lo + term_lo[..., index], negative_inf)
            total_hi = torch.nextafter(total_hi + term_hi[..., index], positive_inf)
            _check(total_lo, total_hi)
        return (total_lo, total_hi, term_lo, term_hi) if return_terms else (total_lo, total_hi)


@lru_cache(maxsize=128)
def make_plan(exponents, n_vars, state_variables=None, time_variable=None):
    """Cache only immutable support/order metadata, never tensors or values."""
    if any(len(e) != n_vars or any(type(p) is not int or p < 0 for p in e) for e in exponents):
        raise ValueError('nonnegative integer exponents required')
    stages = []

    def power_stage(variable):
        indices = tuple(i for i, e in enumerate(exponents) if e[variable])
        if indices:
            stages.append(('power', variable, indices, tuple(exponents[i][variable] for i in indices)))

    if state_variables is None:
        for variable in range(n_vars):
            power_stage(variable)
    else:
        if time_variable is not None:
            power_stage(time_variable)
        values = tuple(-1. if any(e[v] % 2 for v in state_variables)
                       else 0. if any(e[v] for v in state_variables) else 1. for e in exponents)
        if exponents:
            stages.append(('normal_factor', -1, tuple(range(len(exponents))), values))
        for variable in range(n_vars):
            if variable not in state_variables and variable != time_variable:
                power_stage(variable)
    return RangePlan(exponents, n_vars, tuple(stages))


def evaluate_polynomial(poly, domain, *, normal=False, state_variables=None, time_variable=None,
                        step_powers=None):
    # The sparse adapter admits only the established scalar CPU64 lane. Other
    # dtypes/devices/shapes keep their original evaluator in production.
    if any(x.dtype != torch.float64 or x.device.type != 'cpu' or x.numel() != 1
           for x in (*poly.terms.values(), *(x for d in domain for x in (d.lo, d.hi)))):
        return NotImplemented
    exponents = tuple(poly.terms)
    if not exponents:
        return Interval.zero()
    coeffs = torch.stack(list(poly.terms.values())).reshape(1, 1, -1)
    lo = torch.stack([x.lo for x in domain]).reshape(1, -1) if domain else torch.empty((1, 0), dtype=torch.float64)
    hi = torch.stack([x.hi for x in domain]).reshape(1, -1) if domain else torch.empty((1, 0), dtype=torch.float64)
    dispatch = _TABLE_REQUEST_DISPATCH.get() if step_powers is not None else _REQUEST_DISPATCH.get()
    if dispatch is not None:
        if step_powers is not None:
            return dispatch(exponents, coeffs, coeffs, lo, hi, tuple(state_variables),
                            time_variable, 'normal', step_powers)
        return dispatch(exponents, coeffs, coeffs, lo, hi,
                        tuple(state_variables) if normal else None, time_variable,
                        'normal' if normal else 'standard')
    plan = make_plan(exponents, poly.n_vars, tuple(state_variables) if normal else None, time_variable)
    result = plan.evaluate(coeffs, coeffs, lo, hi)
    return Interval(result[0][0, 0], result[1][0, 0])


def evaluate_interval_coefficients(coefficients, domain, *, reference):
    if any(x.dtype != torch.float64 or x.device.type != 'cpu' or x.numel() != 1
           for iv in (*coefficients.values(), *domain, reference) for x in (iv.lo, iv.hi)):
        return NotImplemented
    exponents = tuple(sorted(coefficients))
    if not exponents:
        return Interval.zero(dtype=reference.dtype, device=reference.device)
    lo = torch.stack([coefficients[e].lo for e in exponents]).reshape(1, 1, -1)
    hi = torch.stack([coefficients[e].hi for e in exponents]).reshape(1, 1, -1)
    domain_lo = torch.stack([x.lo for x in domain]).reshape(1, -1) if domain else torch.empty((1, 0), dtype=torch.float64)
    domain_hi = torch.stack([x.hi for x in domain]).reshape(1, -1) if domain else torch.empty((1, 0), dtype=torch.float64)
    dispatch = _REQUEST_DISPATCH.get()
    if dispatch is not None:
        return dispatch(exponents, lo, hi, domain_lo, domain_hi, None, None, 'interval-coefficient')
    a, b = make_plan(exponents, len(domain)).evaluate(lo, hi, domain_lo, domain_hi)
    return Interval(a[0, 0], b[0, 0])
