from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
BASELINE_EXPERIMENT = EXPERIMENT.parent / "first_order_three_way"
DIFFREACH_ROOT = Path(
    os.environ.get("DIFFREACH_ROOT", EXPERIMENT.parents[1] / "DiffReach")
).resolve()
for path in (EXPERIMENT, BASELINE_EXPERIMENT, DIFFREACH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import load_spec
from run_diffreach import (
    DiffReachPlantCore,
    _rhs,
    build_linear_tm,
    identity_parameterization,
    step_boxes,
)
from src.interval import Interval
from src.picard import remainder_picard
from src.polynomial import QuadPoly
from src.symbolic_remainder import init_symbolic_state, symbolic_step_linear
from src.taylor_model import QuadTM
import src.settings as settings


def _stock_internal_one_step(core, lower, upper):
    """The plant-only statements in DiffReach CT_Dyn_Reach.step_once."""
    batch, dimension = lower.shape
    boxes = step_boxes(batch, dimension, core.h, lower.dtype)
    parameterization = identity_parameterization(batch, dimension, lower.dtype)
    x_tm = build_linear_tm(0.5 * (lower + upper), 0.5 * (upper - lower))
    symbolic = init_symbolic_state(batch, dimension, M=1, dtype=lower.dtype)
    endpoint_tm = x_tm.evaluate_time(core.h)
    center = endpoint_tm.P.c
    scale, normalized, _ = symbolic_step_linear(
        parameterization, endpoint_tm, symbolic, boxes[2], boxes[3]
    )
    new_x0 = build_linear_tm(center, scale)
    base = new_x0.P
    poly1 = base.add(core.rhs_poly(base, boxes[0], boxes[1]).integrate_time_trunc())
    poly2 = base.add(core.rhs_poly(poly1, boxes[0], boxes[1]).integrate_time_trunc())
    polynomial_tm = QuadTM.from_poly(poly2)
    epsilon = jnp.broadcast_to(
        jnp.asarray(core.init_remainder, dtype=center.dtype), center.shape
    )
    seeded = QuadTM(
        polynomial_tm.P,
        Interval(polynomial_tm.R.lo - epsilon, polynomial_tm.R.hi + epsilon),
    )
    x_next, contraction = remainder_picard(
        core.rhs_tm,
        new_x0,
        seeded,
        core.h,
        boxes[0],
        boxes[1],
        rounds=core.frr_rounds,
        stop_ratio=core.frr_stop_ratio,
    )
    if settings.CONFIG["TRUNCATE_TO_AFFINE"]:
        x_next = x_next.truncate_to_affine(boxes[0], boxes[1])
    composed = x_next.compose_affine(normalized, core.h)
    endpoint_lo = jnp.concatenate([boxes[1][:, :1], boxes[0][:, 1:]], axis=1)
    endpoint = composed.eval_interval(endpoint_lo, boxes[1])
    return endpoint, x_next, contraction


@pytest.mark.parametrize("truncate", [False, True])
def test_custom_plant_core_one_step_matches_stock_internal_kernel(truncate):
    spec = load_spec(EXPERIMENT / "benchmark_spec.yaml")
    system = spec["systems"]["riccati"]
    old = copy.deepcopy(settings.CONFIG)
    try:
        settings.update_config(
            {
                "TRUNCATE_TO_AFFINE": truncate,
                "FP64_IN_CROWN": True,
                "BOUND_TIME_STEP": True,
                "DEBUG_LOG": False,
            }
        )
        core = DiffReachPlantCore(
            _rhs(system),
            dimension=1,
            h=0.01,
            init_remainder=0.1,
            frr_rounds=5,
            frr_stop_ratio=0.95,
            symbolic_window=100,
        )
        lower = jnp.asarray([[0.0]], dtype=jnp.float64)
        upper = jnp.asarray([[0.1]], dtype=jnp.float64)
        custom = core.verify(lower, upper, 1)
        stock_endpoint, stock_tm, stock_contraction = _stock_internal_one_step(
            core, lower, upper
        )
    finally:
        settings.CONFIG.clear()
        settings.CONFIG.update(old)
    np.testing.assert_allclose(custom[1][0, 1], stock_endpoint.lo[0], atol=0, rtol=0)
    np.testing.assert_allclose(custom[2][0, 1], stock_endpoint.hi[0], atol=0, rtol=0)
    np.testing.assert_allclose(custom[5].P.c, stock_tm.P.c, atol=0, rtol=0)
    np.testing.assert_allclose(custom[5].P.L, stock_tm.P.L, atol=0, rtol=0)
    np.testing.assert_allclose(custom[5].P.Lt, stock_tm.P.Lt, atol=0, rtol=0)
    np.testing.assert_array_equal(custom[6][0], stock_contraction)
