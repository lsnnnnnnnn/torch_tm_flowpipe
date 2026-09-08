"""Attempt-local reuse of polynomial-only work in ordered remainder replay.

The RHS is still called on every replay. Its ordered operation keys are checked
as they execute, so a changed expression cannot silently reuse a different
polynomial product. Only coefficient operations and polynomial-only ranges
are saved; every current-R operation and ledger check is executed again.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import time
from dataclasses import dataclass
from typing import Any

import torch

from . import batched_dense_tm as d


_ENABLED = ContextVar("prepared_remainder_replay", default=False)


@contextmanager
def prepared_remainder_replay(enabled: bool = True):
    """Opt into preparation in this context; numerical storage is never global."""
    token = _ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _ENABLED.reset(token)


def is_enabled() -> bool:
    return _ENABLED.get()


class PlanInvalidated(ValueError):
    """A plan was presented with a different binding or expression."""


@dataclass(frozen=True)
class _Operation:
    key: tuple
    poly: d.BatchedPolynomial
    fixed: tuple = ()


class _Tape:
    def __init__(self, domain_lo, domain_hi, policy, order, cutoff, raw):
        self.domain_lo, self.domain_hi = domain_lo, domain_hi
        self.policy, self.order, self.cutoff, self.raw = policy, order, cutoff, raw
        self._operations = []
        self._sealed = False
        self.position = 0
        self.preparations = 0
        self.hits = 0
        self.preparation_s = 0.

    def start(self):
        self.position = 0

    def finish(self):
        if self.position != len(self._operations):
            raise PlanInvalidated("prepared RHS operation count changed")
        if not self._sealed:
            self._operations = tuple(self._operations)
        self._sealed = True

    def operation(self, key, prepare):
        index = self.position
        if self._sealed:
            if index >= len(self._operations) or self._operations[index].key != key:
                raise PlanInvalidated("prepared RHS ordered structure or constant changed")
            result = self._operations[index]
            self.hits += 1
        else:
            started = time.perf_counter()
            poly, fixed = prepare()
            result = _Operation(key, poly, fixed)
            self._operations.append(result)
            self.preparations += 1
            self.preparation_s += time.perf_counter() - started
        self.position += 1
        return result, index

    def model(self, poly, lo, hi, ledger):
        return d.BatchedTaylorModel(poly, lo, hi, self.domain_lo, self.domain_hi,
                                   ledger, self.policy, None)

    def input(self, model, index):
        op, node = self.operation(("input", index), lambda: (model.poly, ()))
        return _Scalar(self, self.model(op.poly, model.rem_lo, model.rem_hi, model.ledger), node)

    def concat(self, scalars):
        # Preserve the ordinary concat's dynamic category union and checks.
        result = d.BatchedTaylorModel.concat([v.model for v in scalars])
        op, node = self.operation(("concat", tuple(v.node for v in scalars)), lambda: (result.poly, ()))
        return _Scalar(self, self.model(op.poly, result.rem_lo, result.rem_hi, result.ledger), node)


class _Vector:
    def __init__(self, models):
        self.models = list(models)

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index):
        return self.models[index]


class _Scalar:
    def __init__(self, tape, model, node):
        self.tape, self.model, self.node = tape, model, node

    @property
    def domain(self):
        return self.model.domain

    @staticmethod
    def concat(models):
        return _Vector(models)

    def coerce(self, value):
        if isinstance(value, _Scalar):
            if value.tape is not self.tape:
                raise PlanInvalidated("cannot combine different replay attempts")
            return value
        if not isinstance(value, (float, int)):
            raise PlanInvalidated("prepared arithmetic constants must be scalar numbers")
        key = ("constant", float(value).hex(), self.model.poly.out_dim)
        def prepare():
            model = d.BatchedTaylorModel.constants_like(value, self.model)
            return model.poly, (model.rem_lo, model.rem_hi)
        op, node = self.tape.operation(key, prepare)
        return _Scalar(self.tape, self.tape.model(op.poly, *op.fixed, d.DenseRemainderLedger.empty()), node)

    def addsub(self, other, subtract=False):
        other = self.coerce(other)
        left, right = self.model, other.model
        left._check_domain(right)
        op, node = self.tape.operation(
            ("subtract" if subtract else "add", self.node, other.node),
            lambda: (left.poly.sub(right.poly) if subtract else left.poly.add(right.poly), ()),
        )
        if subtract:
            lo, hi = d._interval_sub(left.rem_lo, left.rem_hi, right.rem_lo, right.rem_hi)
            ledger = left.ledger.merge(right.ledger.negate())
        else:
            lo, hi = d._interval_add(left.rem_lo, left.rem_hi, right.rem_lo, right.rem_hi)
            ledger = left.ledger.merge(right.ledger)
        return _Scalar(self.tape, self.tape.model(op.poly, lo, hi, ledger), node)

    def __add__(self, other):
        return self.addsub(other)

    __radd__ = __add__

    def __sub__(self, other):
        return self.addsub(other, True)

    def __rsub__(self, other):
        return self.coerce(other).__sub__(self)

    def __neg__(self):
        op, node = self.tape.operation(("negate", self.node), lambda: (self.model.poly.scale(-1.), ()))
        lo, hi = d._interval_scale(self.model.rem_lo, self.model.rem_hi, -1.)
        ledger = self.model.ledger.scale(-1.)
        return _Scalar(self.tape, self.tape.model(op.poly, lo, hi, ledger), node)

    def __mul__(self, other):
        other = self.coerce(other)
        left, right, tape = self.model, other.model, self.tape
        left._check_domain(right)
        def prepare():
            poly, trunc_lo, trunc_hi = left.poly.mul_trunc(
                right.poly, return_truncation_bound=True,
                domain_lo=tape.domain_lo, domain_hi=tape.domain_hi,
                dropped_merge_mode="merged", max_degree=tape.order if tape.raw else None,
                range_policy=tape.policy, range_trace=None, range_context="polynomial_truncation")
            p = left.poly.range_bound(tape.domain_lo, tape.domain_hi, policy=tape.policy,
                                      context="poly_times_remainder", trace=None)
            q = right.poly.range_bound(tape.domain_lo, tape.domain_hi, policy=tape.policy,
                                       context="remainder_times_poly", trace=None)
            return poly, (trunc_lo, trunc_hi, *p, *q)
        op, node = tape.operation(("multiply", self.node, other.node), prepare)
        trunc_lo, trunc_hi, p_lo, p_hi, q_lo, q_hi = op.fixed
        p_j_lo, p_j_hi = d._interval_mul(p_lo, p_hi, right.rem_lo, right.rem_hi)
        q_i_lo, q_i_hi = d._interval_mul(q_lo, q_hi, left.rem_lo, left.rem_hi)
        i_j_lo, i_j_hi = d._interval_mul(left.rem_lo, left.rem_hi, right.rem_lo, right.rem_hi)
        lo, hi = d._interval_add(trunc_lo, trunc_hi, p_j_lo, p_j_hi)
        lo, hi = d._interval_add(lo, hi, q_i_lo, q_i_hi)
        lo, hi = d._interval_add(lo, hi, i_j_lo, i_j_hi)
        ledger = d.DenseRemainderLedger.empty()
        ledger = ledger.add("polynomial_truncation", trunc_lo, trunc_hi)
        ledger = ledger.add("poly_times_remainder", p_j_lo, p_j_hi)
        ledger = ledger.add("remainder_times_poly", q_i_lo, q_i_hi)
        ledger = ledger.add("remainder_times_remainder", i_j_lo, i_j_hi)
        result = _Scalar(tape, tape.model(op.poly, lo, hi, ledger), node)
        return result.apply_cutoff() if tape.raw else result

    __rmul__ = __mul__

    def pow_int(self, exponent):
        if exponent < 0:
            raise ValueError("raw trace only supports nonnegative integer powers")
        result = self.coerce(1.)
        for _ in range(int(exponent)):
            result = result * self
        return result

    def apply_cutoff(self):
        model, tape = self.model, self.tape
        def prepare():
            poly, lo, hi = model.poly.apply_cutoff(tape.cutoff, tape.domain_lo, tape.domain_hi,
                range_policy=tape.policy, range_trace=None, range_context="cutoff")
            return poly, (lo, hi)
        op, node = tape.operation(("cutoff", self.node), prepare)
        lo, hi = d._interval_add(model.rem_lo, model.rem_hi, *op.fixed)
        ledger = model.ledger.add("cutoff", *op.fixed)
        if "endpoint_substitution_roundoff" in ledger.entries:
            ledger, lo, hi = ledger.covering_total(lo, hi)
        return _Scalar(tape, tape.model(op.poly, lo, hi, ledger), node)

    def integrate(self, tau_index):
        model, tape = self.model, self.tape
        def prepare():
            poly, lo, hi = model.poly.integrate(tau_index, domain_lo=tape.domain_lo, domain_hi=tape.domain_hi,
                return_overflow_bound=True, range_policy=tape.policy, range_trace=None,
                range_context="integration_overflow")
            return poly, (lo, hi)
        op, node = tape.operation(("integrate", self.node, tau_index), prepare)
        tau_lo = tape.domain_lo[:, tau_index].view(-1, 1)
        tau_hi = tape.domain_hi[:, tau_index].view(-1, 1)
        lo, hi = d._interval_mul(tau_lo, tau_hi, model.rem_lo, model.rem_hi)
        lo, hi = d._interval_add(lo, hi, *op.fixed)
        ledger = d.DenseRemainderLedger.empty()
        for category, (entry_lo, entry_hi) in model.ledger.entries.items():
            scaled = d._interval_mul(tau_lo, tau_hi, entry_lo, entry_hi)
            ledger = ledger.add(category, *scaled)
        ledger = ledger.add("integration_overflow", *op.fixed)
        return _Scalar(tape, tape.model(op.poly, lo, hi, ledger), node)


def _model_tensors(model):
    return (model.poly.coeffs, model.domain_lo, model.domain_hi, model.rem_lo, model.rem_hi,
            *(tensor for pair in model.ledger.entries.values() for tensor in pair))


class PreparedRemainderReplay:
    """A private snapshot and two checked operation tapes for one attempt.

    Copies isolate all saved numerical storage, including basis route tensors.
    The public evaluator returns new proposal tensors/decompositions, never a
    saved polynomial or range. Standard in-place changes to the original
    binding are rejected by constant-cost tensor version checks. New candidate,
    domain, base/scale, ODE, order, h or cutoff requires a new plan.
    """
    def __init__(self, rhs_fn, base, candidate, *, tau_index, order, cutoff_threshold, validation_eps):
        started = time.perf_counter()
        self._rhs = rhs_fn
        self._original_base, self._original_candidate = base, candidate
        self._versions = tuple((t, t._version) for m in (base, candidate) for t in _model_tensors(m))
        self._arguments = (tau_index, order, cutoff_threshold, validation_eps)
        self._base, self._candidate = copy.deepcopy((base, candidate))
        self._raw = _Tape(self._candidate.domain_lo, self._candidate.domain_hi,
                          candidate.range_policy, max(order-1, 0), cutoff_threshold, True)
        self._regular = _Tape(self._candidate.domain_lo, self._candidate.domain_hi,
                              candidate.range_policy, order, cutoff_threshold, False)
        self._difference = None
        self._failed = False
        self._setup_s = time.perf_counter() - started

    def check(self, rhs_fn, base, candidate, *, tau_index, order, cutoff_threshold, validation_eps):
        if self._failed:
            raise PlanInvalidated("failed replay attempt requires a new plan")
        if rhs_fn is not self._rhs or base is not self._original_base or candidate is not self._original_candidate:
            raise PlanInvalidated("prepared replay owner/candidate/ODE changed")
        if self._arguments != (tau_index, order, cutoff_threshold, validation_eps):
            raise PlanInvalidated("prepared replay step/order/cutoff/epsilon changed")
        if any(t._version != version for t, version in self._versions):
            raise PlanInvalidated("prepared replay input was modified in place")

    def _rhs_on_tape(self, tape, target):
        tape.start()
        current = target
        state = _Vector([tape.input(current.component(i), i) for i in range(current.poly.out_dim)])
        try:
            output = self._rhs(state)
        except TypeError as original:
            try:
                output = self._rhs(state, None)
            except TypeError:
                raise original
        values = list(output.models) if hasattr(output, "models") else list(output)
        if len(values) != current.poly.out_dim or not all(isinstance(v, _Scalar) for v in values):
            raise PlanInvalidated("prepared RHS must return arithmetic components")
        return tape.concat(values)

    def _raw_rhs(self, target):
        result = self._rhs_on_tape(self._raw, target)
        self._raw.finish()
        return result.model

    def _regular_tmp(self, target):
        result = self._rhs_on_tape(self._regular, target)
        # Match base.add(rhs.integrate(tau)).apply_cutoff exactly, including
        # both dynamic ledgers; the regular RHS is not discarded in production.
        integrated = result.integrate(self._arguments[0])
        base = self._regular.input(self._base, -1)
        result = (base + integrated).apply_cutoff()
        self._regular.finish()
        return result.model

    def _polynomial_difference(self, tmp):
        if self._difference is None:
            started = time.perf_counter()
            poly = tmp.poly.sub(self._candidate.poly)
            lo, hi = poly.range_bound(self._candidate.domain_lo, self._candidate.domain_hi,
                policy=self._candidate.range_policy, context="raw_compat_poly_diff", trace=None)
            self._difference = (poly, lo, hi)
            self._setup_s += time.perf_counter() - started
        return self._difference

    def work_counts(self):
        return {
            "prepared_plan_count": 1,
            "prepared_operation_count": self._raw.preparations + self._regular.preparations,
            "prepared_operation_hits": self._raw.hits + self._regular.hits,
            "prepared_plan_setup_s": self._setup_s + self._raw.preparation_s + self._regular.preparation_s,
        }

    def image(self, remainder_lo, remainder_hi, *, record_evidence=True):
        tau, order, cutoff, eps = self._arguments
        self.check(self._rhs, self._original_base, self._original_candidate,
                   tau_index=tau, order=order, cutoff_threshold=cutoff, validation_eps=eps)
        target = self._candidate.with_remainder(remainder_lo, remainder_hi, category="initial_remainder")
        try:
            return d._dense_flowstar_raw_compat_image(
                self._rhs, self._base, target, self._candidate,
                tau_index=tau, order=order, cutoff_threshold=cutoff, validation_eps=eps,
                raw_rhs_evaluation="ordered_terms", record_evidence=record_evidence,
                prepared_replay=self,
            )
        except Exception:
            self._failed = True
            raise
