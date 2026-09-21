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


class SymbolicRemainderCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, phi_present=True, scalars_present=True):
        sr = make_symbolic_remainder(2, 3, 1000, "cpu")
        sr.reserve(40)
        sr.qlen, sr.jlen = 3, 2  # Deliberately between Phi push and J append.
        if phi_present:
            sr.phi_iv_buf = torch.empty(*sr.phi_buf.shape, 2, dtype=torch.float64)
        if scalars_present:
            sr.scalars_iv = torch.empty(*sr.scalars.shape, 2, dtype=torch.float64)
        for field in fields(sr):
            value = getattr(sr, field.name)
            if isinstance(value, torch.Tensor):
                # Nonzero sentinels throughout unused capacity catch a live-only
                # copy that would pass an ordinary short solver comparison.
                value.copy_(torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape) + 0.25)
        return sr

    def assert_same(self, left, right):
        self.assertIs(type(left), type(right))
        for field in fields(left):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, torch.Tensor):
                self.assertEqual((a.shape, a.dtype, a.device), (b.shape, b.dtype, b.device))
                self.assertTrue(torch.equal(a, b), field.name)
            else:
                self.assertEqual(a, b, field.name)

    def test_copy_preserves_all_capacity_and_optional_fields_without_aliasing(self):
        from .candidate import _copy_symbolic_remainder
        for phi_present, scalars_present in [(False, False), (True, False), (False, True), (True, True)]:
            with self.subTest(phi=phi_present, scalars=scalars_present):
                sr = self.fixture(phi_present, scalars_present)
                # Make future schema additions fail loudly until the copy is
                # extended; silently omitting a new strict field is unsafe.
                self.assertEqual({f.name for f in fields(sr)},
                                 {"scalars", "max_size", "phi_buf", "j_buf", "phi_iv_buf", "scalars_iv", "qlen", "jlen"})
                original = copy.deepcopy(sr)
                pending = _copy_symbolic_remainder(sr)
                self.assert_same(pending, original)
                for field in fields(sr):
                    value = getattr(pending, field.name)
                    if isinstance(value, torch.Tensor):
                        self.assertNotEqual(value.data_ptr(), getattr(sr, field.name).data_ptr(), field.name)
                        value.fill_(-13.0)
                pending.qlen, pending.jlen, pending.max_size = 7, 6, 71
                self.assert_same(sr, original)

    def test_failed_batch_discards_mutation_of_every_pending_field(self):
        from unittest.mock import patch
        sr = self.fixture()
        saved = copy.deepcopy(sr)
        state = SimpleNamespace(status=torch.tensor([ACTIVE, ACTIVE]))

        def destructive_failure(state, code, eng, sched, settings, rem, pending):
            self.assertIsNot(pending, sr)
            pending.reserve(80)
            for field in fields(pending):
                value = getattr(pending, field.name)
                if isinstance(value, torch.Tensor):
                    value.fill_(-19.0)
            pending.qlen, pending.jlen, pending.max_size = 8, 7, 81
            return SimpleNamespace(status=torch.tensor([ACTIVE, FAILED_CONTRACTION])), torch.tensor([True, False])

        with patch("flowstar_gpu.sparse_exec.advance_sparse", destructive_failure), \
             patch("flowstar_gpu.sparse_exec.prune_state", side_effect=AssertionError("failed batch must not commit")):
            out, out_sr, accepted, statuses, reset = advance_transaction(state, sr, None, None, None, None, None)
        self.assertIs(out, state)
        self.assertIs(out_sr, sr)
        self.assertEqual(accepted, [True, False])
        self.assertEqual(statuses, [ACTIVE, FAILED_CONTRACTION])
        self.assertFalse(reset)
        self.assert_same(sr, saved)

    def test_clone_and_deepcopy_match_through_growth_and_reset(self):
        from unittest.mock import patch
        original, cloned = make_case(capacity=18), make_case(capacity=18)
        for step in range(1, 20):
            with patch("experiments.whole_engine_feasibility.candidate._copy_symbolic_remainder", copy.deepcopy):
                old_result = advance(original)
            new_result = advance(cloned)
            self.assertEqual(old_result, new_result, step)
            self.assertEqual(new_result[0], [True, True])
            self.assert_same(original.state, cloned.state)
            self.assert_same(original.sr, cloned.sr)
        self.assertEqual(cloned.sr.queue_len, 1)
        self.assertEqual(cloned.sr.phi_buf.shape[0], 18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
