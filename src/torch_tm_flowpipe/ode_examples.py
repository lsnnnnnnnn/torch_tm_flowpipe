"""Polynomial ODE examples for tests, examples, and experiments."""
from __future__ import annotations

from .taylor_model import TaylorModel
from .tm_vector import TMVector


def scalar_quadratic_ode(x: TMVector, u: TMVector | None = None) -> TMVector:
    """x' = 1 + x^2.

    Exact solution for scalar samples is ``tan(t + atan(x0))`` before blow-up.
    """
    return TMVector([1.0 + x[0] * x[0]])


def harmonic_oscillator_ode(x: TMVector, u: TMVector | None = None) -> TMVector:
    """x0' = x1, x1' = -x0."""
    return TMVector([x[1], -x[0]])


def van_der_pol_ode(x: TMVector, u: TMVector | None = None, mu: float = 1.0) -> TMVector:
    """Van der Pol oscillator with polynomial right-hand side."""
    return TMVector([x[1], mu * (1.0 - x[0] * x[0]) * x[1] - x[0]])


def affine_controlled_ode(x: TMVector, u: TMVector | None = None) -> TMVector:
    """Simple controlled polynomial plant: x0' = x1 + u0, x1' = -x0 + u0."""
    if u is None or len(u) == 0:
        u0 = TaylorModel.constant(0.0, x.domain)
    else:
        u0 = u[0]
    return TMVector([x[1] + u0, -x[0] + u0])
