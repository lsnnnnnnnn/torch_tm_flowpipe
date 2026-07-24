from __future__ import annotations

import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
DIFFREACH_ROOT = Path("/srv/local/shengenli/DiffReach")
for path in (EXPERIMENT, DIFFREACH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffreach_projection import strict_affine_projection
from src.interval import Interval
from src.polynomial import QuadPoly
from src.taylor_model import QuadTM


def _model(lt, *, linear=None, constant=0.0, remainder=(-0.0, 0.0)):
    lt = jnp.asarray(lt, dtype=jnp.float64).reshape((1, 1, -1))
    variables = lt.shape[-1]
    linear = (
        jnp.zeros((1, 1, variables), dtype=jnp.float64)
        if linear is None
        else jnp.asarray(linear, dtype=jnp.float64).reshape((1, 1, variables))
    )
    polynomial = QuadPoly(
        jnp.asarray([[constant]], dtype=jnp.float64),
        linear,
        lt,
    )
    return QuadTM(
        polynomial,
        Interval(
            jnp.asarray([[remainder[0]]], dtype=jnp.float64),
            jnp.asarray([[remainder[1]]], dtype=jnp.float64),
        ),
    )


def _contains(projected, original, lo, hi, points=41):
    grid = [
        np.linspace(float(lo[0, index]), float(hi[0, index]), points)
        for index in range(lo.shape[1])
    ]
    projected_iv = projected.eval_interval(lo, hi)
    p_lo = float(projected_iv.lo[0, 0])
    p_hi = float(projected_iv.hi[0, 0])
    for coordinates in np.array(np.meshgrid(*grid)).T.reshape(-1, lo.shape[1]):
        t = coordinates[0]
        value = float(original.P.c[0, 0])
        value += float(np.dot(np.asarray(original.P.L[0, 0]), coordinates))
        value += t * float(np.dot(np.asarray(original.P.Lt[0, 0]), coordinates))
        assert p_lo <= value <= p_hi


@pytest.mark.parametrize("coefficient", [3.0, -3.0])
def test_positive_and_negative_t_squared(coefficient):
    model = _model([coefficient, 0.0])
    lo = jnp.asarray([[0.0, -1.0]], dtype=jnp.float64)
    hi = jnp.asarray([[0.2, 1.0]], dtype=jnp.float64)
    projected, audit = strict_affine_projection(model, lo, hi, return_audit=True)
    assert np.count_nonzero(np.asarray(projected.P.Lt)) == 0
    _contains(projected, model, lo, hi)
    expected = sorted([0.0, coefficient * 0.2**2])
    assert float(audit.term_lower[0, 0, 0]) <= expected[0]
    assert float(audit.term_upper[0, 0, 0]) >= expected[1]


@pytest.mark.parametrize("coefficient", [2.5, -2.5])
def test_positive_and_negative_time_generator(coefficient):
    model = _model([0.0, coefficient])
    lo = jnp.asarray([[0.0, -2.0]], dtype=jnp.float64)
    hi = jnp.asarray([[0.3, 1.0]], dtype=jnp.float64)
    projected = strict_affine_projection(model, lo, hi)
    assert np.count_nonzero(np.asarray(projected.P.Lt)) == 0
    _contains(projected, model, lo, hi)


def test_asymmetric_time_multiple_generators_and_existing_remainder():
    model = _model(
        [1.0, -2.0, 0.5],
        linear=[0.1, 0.2, -0.3],
        constant=1.2,
        remainder=(-0.25, 0.75),
    )
    lo = jnp.asarray([[0.1, -2.0, 0.5]], dtype=jnp.float64)
    hi = jnp.asarray([[0.4, 3.0, 2.0]], dtype=jnp.float64)
    projected = strict_affine_projection(model, lo, hi)
    assert np.count_nonzero(np.asarray(projected.P.Lt)) == 0
    assert float(projected.R.lo[0, 0]) <= -0.25
    assert float(projected.R.hi[0, 0]) >= 0.75
    _contains(projected, model, lo, hi, points=11)


def test_projection_is_identity_when_lt_is_zero():
    model = _model(
        [0.0, 0.0],
        linear=[0.2, -0.3],
        constant=1.5,
        remainder=(-0.25, 0.75),
    )
    lo = jnp.asarray([[0.0, -1.0]], dtype=jnp.float64)
    hi = jnp.asarray([[0.1, 1.0]], dtype=jnp.float64)
    projected = strict_affine_projection(model, lo, hi)
    np.testing.assert_array_equal(projected.P.c, model.P.c)
    np.testing.assert_array_equal(projected.P.L, model.P.L)
    np.testing.assert_array_equal(projected.P.Lt, model.P.Lt)
    np.testing.assert_array_equal(projected.R.lo, model.R.lo)
    np.testing.assert_array_equal(projected.R.hi, model.R.hi)
