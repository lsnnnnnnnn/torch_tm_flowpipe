"""Bounded CPU lifecycle checks for the whole-engine adapter.

These exercise the real sparse solver, not mocked acceptance flags. They are
local lifecycle evidence; they do not certify either numerical engine.
Run with the isolated Huan engine's src directory on PYTHONPATH.
"""
import copy
from dataclasses import fields
from fractions import Fraction
import math
from types import SimpleNamespace
import unittest

import torch

from flowstar_gpu.config import Settings
from flowstar_gpu.composition import build_schedule
from flowstar_gpu.determinism import enable_determinism
from flowstar_gpu.flowpipe import ACTIVE, FAILED_CONTRACTION, build_rem_est
from flowstar_gpu.monomials import build_tables
from flowstar_gpu.ode_compiler import compile_ode
from flowstar_gpu.polynomial import build_step_tables
from flowstar_gpu.sparse_exec import initial_sparse_state
from flowstar_gpu.support import SparseEngine
from flowstar_gpu.symbolic_remainder import make_symbolic_remainder

from .baseline import frozen_case
from .candidate import advance_transaction
from .export import export_sparse_state
from .observe import observe_step


def make_case(*, lane=None, rhs=None, capacity=None):
    case = frozen_case("van_der_pol", 2)
    boxes = case["boxes"] if lane is None else [case["boxes"][lane]]
    batch = len(boxes)
    settings = Settings(step=case["h"], order=case["order"], cutoff=case["cutoff"],
                        remainder_estimation=case["remainder_estimation"],
                        sr_queue=case["sr_capacity"] if capacity is None else capacity,
                        mode="strict", device="cpu",
                        max_refinement_steps=case["refinement_limit"],
                        stop_ratio=case["stop_ratio"])
    tab = build_tables(2, settings.order).to("cpu")
    eng = SparseEngine(tab, build_step_tables(tab, settings.step), "cpu")
    sched = build_schedule(2, settings.order, "cpu")
    code = compile_ode(case["rhs_expression_strings"] if rhs is None else rhs,
                       ["x", "y"], order=settings.order - 1)
    state = initial_sparse_state(torch.tensor(boxes, dtype=torch.float64), eng, sched)
    return SimpleNamespace(case=case, boxes=boxes, settings=settings, tab=tab,
                           eng=eng, sched=sched, code=code, state=state,
                           sr=make_symbolic_remainder(batch, 2, settings.sr_queue, "cpu"),
                           rem=build_rem_est(settings, 2, batch))


def advance(ctx, rem=None):
    ctx.state, ctx.sr, accepted, statuses, reset = advance_transaction(
        ctx.state, ctx.sr, ctx.code, ctx.eng, ctx.sched, ctx.settings,
        ctx.rem if rem is None else rem)
    return accepted, statuses, reset


def observed(ctx, step, lane):
    return observe_step(export_sparse_state(ctx.state, ctx.tab, ctx.settings.step, lane=lane),
                        plant="van_der_pol", step=step, h=ctx.settings.step, lane=lane)


class CandidateLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        enable_determinism("cpu")

    def assert_record_unchanged(self, actual, saved):
        for field in fields(actual):
            left, right = getattr(actual, field.name), getattr(saved, field.name)
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right), field.name)
            else:
                self.assertEqual(left, right, field.name)

    def test_mixed_lane_rejection_rolls_back_state_and_nonempty_sr(self):
        ctx = make_case()
        self.assertEqual(advance(ctx)[0], [True, True])
        self.assertGreater(ctx.sr.queue_len, 0)
        old_state, old_sr = ctx.state, ctx.sr
        saved_state, saved_sr = copy.deepcopy(old_state), copy.deepcopy(old_sr)
        # A zero radius cannot validate the nonlinear lane's real Picard image;
        # the other lane keeps the frozen 1e-4 radius and really accepts.
        mixed_rem = ctx.rem.clone()
        mixed_rem[1].zero_()
        accepted, statuses, reset = advance(ctx, mixed_rem)
        self.assertEqual(accepted, [True, False])
        self.assertEqual(statuses, [ACTIVE, FAILED_CONTRACTION])
        self.assertFalse(reset)
        self.assertIs(ctx.state, old_state)
        self.assertIs(ctx.sr, old_sr)
        self.assert_record_unchanged(ctx.state, saved_state)
        self.assert_record_unchanged(ctx.sr, saved_sr)

    def test_vdp_lanes_match_independent_single_lane_advances(self):
        batch = make_case()
        singles = [make_case(lane=lane) for lane in range(2)]
        for step in range(1, 3):
            self.assertEqual(advance(batch)[0], [True, True])
            for lane, single in enumerate(singles):
                self.assertEqual(advance(single)[0], [True])
                together, alone = observed(batch, step, lane), observed(single, step, 0)
                for joint, independent in zip(together, alone):
                    self.assertEqual((joint["view"], joint["coordinate"]),
                                     (independent["view"], independent["coordinate"]))
                    # CPU grouping may change final rounding; this is a locality
                    # comparison, not a proof or a CPU/GPU bitwise-equality gate.
                    for side in ("lo", "hi"):
                        self.assertTrue(math.isclose(joint[side], independent[side],
                                                    rel_tol=1e-11, abs_tol=1e-13),
                                        (step, lane, joint, independent))
        for ctx in (batch, *singles):
            self.assertEqual(ctx.sr.queue_len, 2)
            self.assertTrue((ctx.state.status == ACTIVE).all().item())

    def test_small_sr_capacity_preserves_linear_enclosure_across_resets(self):
        # Analytic x'=y, y'=0, using both frozen VDP subboxes. An explicit
        # independent initial remainder exercises carrying uncertainty through
        # reset; it is a lifecycle fixture, not a change to benchmark settings.
        ctx = make_case(rhs=["y", "0"], capacity=2)
        radius = 1e-5
        ctx.state.pre_rem[..., 0] = -radius
        ctx.state.pre_rem[..., 1] = radius
        resets = []
        for step in range(1, 6):
            old_state, old_sr = ctx.state, ctx.sr
            saved_state, saved_sr = copy.deepcopy(old_state), copy.deepcopy(old_sr)
            accepted, statuses, reset = advance(ctx)
            self.assertEqual(accepted, [True, True])
            self.assertEqual(statuses, [ACTIVE, ACTIVE])
            self.assert_record_unchanged(old_state, saved_state)
            self.assert_record_unchanged(old_sr, saved_sr)
            self.assertEqual(ctx.sr.queue_len, step % 2)
            if reset:
                resets.append(step)
                self.assertTrue(torch.equal(ctx.sr.scalars, torch.ones_like(ctx.sr.scalars)))
            h = Fraction(ctx.settings.step)
            for lane, box in enumerate(ctx.boxes):
                xlo, xhi = Fraction(box[0][0]) - Fraction(radius), Fraction(box[0][1]) + Fraction(radius)
                ylo, yhi = Fraction(box[1][0]) - Fraction(radius), Fraction(box[1][1]) + Fraction(radius)
                for row in observed(ctx, step, lane):
                    tlo = h * (step if row["view"] == "endpoint" else step - 1)
                    thi = h * step
                    lo, hi = (xlo + tlo * ylo, xhi + thi * yhi) if row["coordinate"] == "x" else (ylo, yhi)
                    self.assertLessEqual(Fraction(row["lo"]), lo, (step, lane, row))
                    self.assertGreaterEqual(Fraction(row["hi"]), hi, (step, lane, row))
        self.assertEqual(resets, [2, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
