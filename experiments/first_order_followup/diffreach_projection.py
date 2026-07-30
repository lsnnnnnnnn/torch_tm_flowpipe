"""Sound experiment-local projection of DiffReach's Lt basis to affine form.

DiffReach represents

    P(t, z) = c + L [t, z] + t * Lt [t, z].

This adapter ranges each individual ``Lt`` term on the supplied local box,
moves the midpoint of the combined range into ``c``, and adds the residual
interval to the existing Taylor-model remainder.  It never reuses an existing
``L`` generator to encode independent overflow.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DIFFREACH_ROOT = Path(
    os.environ.get("DIFFREACH_ROOT", REPO_ROOT.parent / "DiffReach")
).resolve()
if str(DIFFREACH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFREACH_ROOT))

import jax.numpy as jnp

from src.interval import Interval
from src.polynomial import QuadPoly
from src.taylor_model import QuadTM


@dataclass(frozen=True)
class StrictAffineProjectionAudit:
    term_lower: Any
    term_upper: Any
    combined_lower: Any
    combined_upper: Any
    midpoint: Any
    residual_lower: Any
    residual_upper: Any


def _down(value: Any) -> Any:
    return jnp.nextafter(value, jnp.full_like(value, -jnp.inf))


def _up(value: Any) -> Any:
    return jnp.nextafter(value, jnp.full_like(value, jnp.inf))


def _mul_interval(coefficient: Any, lower: Any, upper: Any) -> tuple[Any, Any]:
    left = coefficient * lower
    right = coefficient * upper
    return _down(jnp.minimum(left, right)), _up(jnp.maximum(left, right))


def _square_interval(lower: Any, upper: Any) -> tuple[Any, Any]:
    zero = jnp.zeros_like(lower)
    crosses_zero = (lower <= 0) & (upper >= 0)
    lower_sq = jnp.where(crosses_zero, zero, jnp.minimum(lower * lower, upper * upper))
    upper_sq = jnp.maximum(lower * lower, upper * upper)
    return _down(lower_sq), _up(upper_sq)


def _product_interval(
    left_lower: Any,
    left_upper: Any,
    right_lower: Any,
    right_upper: Any,
) -> tuple[Any, Any]:
    products = jnp.stack(
        [
            left_lower * right_lower,
            left_lower * right_upper,
            left_upper * right_lower,
            left_upper * right_upper,
        ],
        axis=0,
    )
    return _down(jnp.min(products, axis=0)), _up(jnp.max(products, axis=0))


def strict_affine_projection(
    model: QuadTM,
    box_lo: Any,
    box_hi: Any,
    *,
    return_audit: bool = False,
) -> QuadTM | tuple[QuadTM, StrictAffineProjectionAudit]:
    """Project all ``Lt`` terms into a fresh independent interval remainder.

    ``box_lo`` and ``box_hi`` have shape ``[batch, variables]``.  Variable zero
    is local time.  The pre-existing remainder is retained exactly, apart from
    outward-rounded addition of the new independent residual.
    """
    if box_lo.shape != box_hi.shape:
        raise ValueError("box_lo and box_hi must have identical shapes")
    if box_lo.shape[-1] != model.P.V:
        raise ValueError("projection box dimension does not match QuadTM variables")

    batch, outputs, variables = model.P.Lt.shape
    del batch
    term_lowers = []
    term_uppers = []
    time_lo = box_lo[:, 0][:, None]
    time_hi = box_hi[:, 0][:, None]
    for variable in range(variables):
        coefficient = model.P.Lt[:, :, variable]
        if variable == 0:
            monomial_lo, monomial_hi = _square_interval(time_lo, time_hi)
        else:
            generator_lo = box_lo[:, variable][:, None]
            generator_hi = box_hi[:, variable][:, None]
            monomial_lo, monomial_hi = _product_interval(
                time_lo, time_hi, generator_lo, generator_hi
            )
        term_lo, term_hi = _mul_interval(coefficient, monomial_lo, monomial_hi)
        term_lowers.append(term_lo)
        term_uppers.append(term_hi)

    if term_lowers:
        term_lower = jnp.stack(term_lowers, axis=-1)
        term_upper = jnp.stack(term_uppers, axis=-1)
        combined_lower = jnp.zeros_like(model.P.c)
        combined_upper = jnp.zeros_like(model.P.c)
        for variable in range(variables):
            combined_lower = _down(combined_lower + term_lower[:, :, variable])
            combined_upper = _up(combined_upper + term_upper[:, :, variable])
    else:  # pragma: no cover - QuadPoly always has at least the time variable
        term_lower = jnp.zeros((model.P.B, outputs, 0), dtype=model.P.c.dtype)
        term_upper = term_lower
        combined_lower = jnp.zeros_like(model.P.c)
        combined_upper = jnp.zeros_like(model.P.c)

    midpoint = (combined_lower + combined_upper) * 0.5
    residual_lower = _down(combined_lower - midpoint)
    residual_upper = _up(combined_upper - midpoint)
    has_lt = jnp.any(model.P.Lt != 0, axis=-1)
    projected_c = jnp.where(has_lt, model.P.c + midpoint, model.P.c)
    projected_r_lo = jnp.where(
        has_lt, _down(model.R.lo + residual_lower), model.R.lo
    )
    projected_r_hi = jnp.where(
        has_lt, _up(model.R.hi + residual_upper), model.R.hi
    )
    projected_poly = QuadPoly(
        c=projected_c,
        L=model.P.L,
        Lt=jnp.zeros_like(model.P.Lt),
        out_shape=model.P.out_shape,
    )
    projected = QuadTM(
        projected_poly,
        Interval(projected_r_lo, projected_r_hi),
    )
    if not return_audit:
        return projected
    return projected, StrictAffineProjectionAudit(
        term_lower=term_lower,
        term_upper=term_upper,
        combined_lower=combined_lower,
        combined_upper=combined_upper,
        midpoint=midpoint,
        residual_lower=residual_lower,
        residual_upper=residual_upper,
    )
