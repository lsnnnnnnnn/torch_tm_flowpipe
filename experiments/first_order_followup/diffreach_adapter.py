"""Plant-only DiffReach adapters used by the follow-up experiment."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASELINE_EXPERIMENT = HERE.parent / "first_order_three_way"
DIFFREACH_ROOT = Path("/srv/local/shengenli/DiffReach")
for path in (HERE, BASELINE_EXPERIMENT, DIFFREACH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax.numpy as jnp

from diffreach_projection import strict_affine_projection
from run_diffreach import DiffReachPlantCore, build_linear_tm
from src.interval import Interval
from src.picard import remainder_picard
from src.symbolic_remainder import symbolic_step_linear
from src.taylor_model import QuadTM


class StrictAffineDiffReachPlantCore(DiffReachPlantCore):
    """DiffReach plant kernel with the experiment-local strict affine projection."""

    def step_once(
        self, carry: tuple[Any, ...], unused: Any
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        del unused
        x_tm, parameterization, symbolic_state, step_lo, step_hi, eval_lo, eval_hi = carry
        endpoint_tm = x_tm.evaluate_time(self.h)
        center = endpoint_tm.P.c
        scale, normalized, symbolic_next = symbolic_step_linear(
            parameterization, endpoint_tm, symbolic_state, eval_lo, eval_hi
        )
        new_x0 = build_linear_tm(center, scale)
        base = new_x0.P
        poly1 = base.add(self.rhs_poly(base, step_lo, step_hi).integrate_time_trunc())
        poly2 = base.add(self.rhs_poly(poly1, step_lo, step_hi).integrate_time_trunc())
        polynomial_tm = QuadTM.from_poly(poly2)
        epsilon = jnp.broadcast_to(
            jnp.asarray(self.init_remainder, dtype=center.dtype),
            center.shape,
        )
        seeded = QuadTM(
            polynomial_tm.P,
            Interval(polynomial_tm.R.lo - epsilon, polynomial_tm.R.hi + epsilon),
        )
        validated, contraction = remainder_picard(
            self.rhs_tm,
            new_x0,
            seeded,
            self.h,
            step_lo,
            step_hi,
            rounds=self.frr_rounds,
            stop_ratio=self.frr_stop_ratio,
        )
        projected = strict_affine_projection(validated, step_lo, step_hi)
        composed = projected.compose_affine(normalized, self.h)
        endpoint_lo = jnp.concatenate([step_hi[:, :1], step_lo[:, 1:]], axis=1)
        endpoint = composed.eval_interval(endpoint_lo, step_hi)
        tube = composed.eval_interval(step_lo, step_hi)
        next_carry = (
            projected,
            normalized,
            symbolic_next,
            step_lo,
            step_hi,
            eval_lo,
            eval_hi,
        )
        return next_carry, (endpoint.lo, endpoint.hi, tube.lo, tube.hi, contraction)
