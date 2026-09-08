"""Outward enclosure of constant substitution, independent of point kernels.

All tensor entries denote their stored floating-point values as exact reals.
Only basic IEEE addition/multiplication and nextafter are used for enclosure;
in particular, neither pow nor a colliding scatter reduction is an oracle.
See docs/endpoint_roundoff_repair/ENDPOINT_CONTRACT.md.
"""
from __future__ import annotations

from typing import Sequence

import torch


def _finite(*values: torch.Tensor) -> None:
    if any(not bool(torch.all(torch.isfinite(value))) for value in values):
        raise FloatingPointError("endpoint substitution requires a finite enclosure")


def _down(value: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(value, torch.full_like(value, -torch.inf))


def _up(value: torch.Tensor) -> torch.Tensor:
    return torch.nextafter(value, torch.full_like(value, torch.inf))


def add(a_lo, a_hi, b_lo, b_hi):
    """Outward interval sum, preserving the exact additive identity."""
    a_zero = (a_lo == 0) & (a_hi == 0)
    b_zero = (b_lo == 0) & (b_hi == 0)
    lo = torch.where(a_zero, b_lo, torch.where(b_zero, a_lo, _down(a_lo + b_lo)))
    hi = torch.where(a_zero, b_hi, torch.where(b_zero, a_hi, _up(a_hi + b_hi)))
    return lo, hi


def _mul(a_lo, a_hi, b_lo, b_hi):
    candidates = torch.stack(torch.broadcast_tensors(
        a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi
    ))
    lo, hi = _down(candidates.amin(0)), _up(candidates.amax(0))
    a_one, b_one = (a_lo == 1) & (a_hi == 1), (b_lo == 1) & (b_hi == 1)
    lo = torch.where(a_one, b_lo, torch.where(b_one, a_lo, lo))
    hi = torch.where(a_one, b_hi, torch.where(b_one, a_hi, hi))
    zero = ((a_lo == 0) & (a_hi == 0)) | ((b_lo == 0) & (b_hi == 0))
    return torch.where(zero, torch.zeros_like(lo), lo), torch.where(zero, torch.zeros_like(hi), hi)


def _powers(lo, hi, degree):
    result = [(torch.ones_like(lo), torch.ones_like(hi))]
    if degree:
        result.append((lo, hi))
    for _ in range(2, degree + 1):
        result.append(_mul(*result[-1], lo, hi))
        _finite(*result[-1])
    return result


def enclose_constant_substitution(
    coefficients: torch.Tensor,
    exponents: Sequence[tuple[int, ...]],
    point_coefficients: torch.Tensor,
    reduced_exponents: Sequence[tuple[int, ...]],
    var_index: int,
    value: torch.Tensor,
    domain_lo: torch.Tensor,
    domain_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return q lower/upper coefficients and E lower/upper (batch, output).

    The caller supplies its retained point coefficients qhat.  For each time
    degree, reduced targets are distinct: assignments below never add floats.
    Each cross-degree sum is separately outward rounded.  E encloses the
    resulting coefficient differences on the actual remaining domain.
    """
    batch, outputs, terms = coefficients.shape
    dim = domain_lo.shape[1]
    if not 0 <= var_index < dim:
        raise IndexError(var_index)
    if coefficients.dtype not in (torch.float32, torch.float64):
        raise TypeError("endpoint substitution supports binary32/binary64")
    if domain_lo.shape != (batch, dim) or domain_hi.shape != domain_lo.shape:
        raise ValueError("endpoint substitution domain shape mismatch")
    if value.shape != (batch,):
        raise ValueError("substitution value must be scalar or [batch]")
    if point_coefficients.shape != (batch, outputs, len(reduced_exponents)) or len(exponents) != terms:
        raise ValueError("endpoint substitution coefficient shape mismatch")
    _finite(coefficients, point_coefficients, value, domain_lo, domain_hi)
    if bool(torch.any(domain_lo > domain_hi)):
        raise ValueError("endpoint substitution has an invalid domain")
    if bool(torch.any((value < domain_lo[:, var_index]) | (value > domain_hi[:, var_index]))):
        raise ValueError("endpoint substitution value is outside the model domain")
    if any(len(e) != dim or any(type(k) is not int or k < 0 for k in e) for e in exponents):
        raise ValueError("endpoint substitution requires nonnegative integer exponents")
    lookup = {e: i for i, e in enumerate(reduced_exponents)}
    if len(lookup) != len(reduced_exponents) or len(set(exponents)) != len(exponents):
        raise ValueError("endpoint substitution requires unique monomials")
    degree = max((e[var_index] for e in exponents), default=0)
    time_powers = _powers(value[:, None, None], value[:, None, None], degree)
    q_lo, q_hi = torch.zeros_like(point_coefficients), torch.zeros_like(point_coefficients)
    for power in range(degree + 1):
        sources = [i for i, e in enumerate(exponents) if e[var_index] == power]
        if not sources:
            continue
        targets = [lookup[e[:var_index] + e[var_index + 1:]] for i in sources for e in [exponents[i]]]
        term_lo, term_hi = _mul(coefficients[..., sources], coefficients[..., sources], *time_powers[power])
        lo, hi = add(q_lo[..., targets], q_hi[..., targets], term_lo, term_hi)
        _finite(lo, hi)
        q_lo[..., targets], q_hi[..., targets] = lo, hi
    err_lo, err_hi = add(q_lo, q_hi, -point_coefficients, -point_coefficients)
    exact_equal = (q_lo == q_hi) & (q_lo == point_coefficients)
    err_lo = torch.where(exact_equal, torch.zeros_like(err_lo), err_lo)
    err_hi = torch.where(exact_equal, torch.zeros_like(err_hi), err_hi)
    remaining = [i for i in range(dim) if i != var_index]
    for reduced_index, variable in enumerate(remaining):
        max_power = max((e[reduced_index] for e in reduced_exponents), default=0)
        powers = _powers(domain_lo[:, variable:variable + 1], domain_hi[:, variable:variable + 1], max_power)
        monomial_lo = torch.cat([powers[e[reduced_index]][0] for e in reduced_exponents], dim=1)
        monomial_hi = torch.cat([powers[e[reduced_index]][1] for e in reduced_exponents], dim=1)
        err_lo, err_hi = _mul(err_lo, err_hi, monomial_lo[:, None, :], monomial_hi[:, None, :])
        _finite(err_lo, err_hi)
    total_lo = torch.zeros((batch, outputs), dtype=coefficients.dtype, device=coefficients.device)
    total_hi = torch.zeros_like(total_lo)
    for i in range(len(reduced_exponents)):
        total_lo, total_hi = add(total_lo, total_hi, err_lo[..., i], err_hi[..., i])
    _finite(q_lo, q_hi, total_lo, total_hi)
    return q_lo, q_hi, total_lo, total_hi
