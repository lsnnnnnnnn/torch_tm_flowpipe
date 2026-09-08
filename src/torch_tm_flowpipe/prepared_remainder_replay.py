"""Attempt-local reuse of polynomial work in ordered remainder refinement.

Only the execution structure changes. Dynamic Taylor-model arithmetic uses
the existing methods, including their interval operations and ledger checks.
No numerical data is stored in the execution-mode ContextVar.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import ast
import inspect
import textwrap
from types import FunctionType
from types import SimpleNamespace
from typing import Any

import torch

from . import batched_dense_tm as d
from .polynomial_ode import PolynomialODE
from .tm_vector import TMVector


_ENABLED = ContextVar('prepared_remainder_replay', default=False)


@contextmanager
def prepared_remainder_replay(enabled: bool = True):
    """Opt in for this context; each accepted attempt owns a fresh plan.

    The existing specialized canonical closure and opaque callables continue
    through their original evaluators.
"""
    if type(enabled) is not bool:
        raise TypeError('prepared_remainder_replay requires a bool')
    token = _ENABLED.set(enabled)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def is_enabled() -> bool:
    return _ENABLED.get()


@lru_cache(maxsize=64)
def _arithmetic_body(code, source):
    """Admit only a literal arithmetic graph, independent of R and globals.

This gate recognizes syntax, not a system name or equation. Opaque callables,
branches, attribute reads and external numerical inputs
remain on the reference evaluator.
"""
    try:
        node = ast.parse(textwrap.dedent(source)).body[0]
        if not isinstance(node, ast.FunctionDef):
            return False
        parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        state = parameters[0].arg
        unused_parameters = {p.arg for p in parameters[1:]}
        body = []
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                continue
            if isinstance(statement, ast.Delete):
                # The benchmark's ``del u`` only unbinds an unused parameter.
                # Deleting a subscript/attribute mutates storage and must stay
                # on the opaque reference path, even if the return is pure.
                if not all(isinstance(target, ast.Name) and target.id in unused_parameters
                           for target in statement.targets):
                    return False
                continue
            body.append(statement)
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            return False
        value = body[0].value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == 'TMVector' and len(value.args) == 1 and not value.keywords
                and isinstance(value.args[0], (ast.List, ast.Tuple))):
            return False
        def arithmetic(expr):
            if isinstance(expr, ast.Constant):
                return type(expr.value) in {int, float}
            if isinstance(expr, ast.Subscript):
                return (isinstance(expr.value, ast.Name) and expr.value.id == state
                        and isinstance(expr.slice, ast.Constant) and type(expr.slice.value) is int)
            if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult)):
                return arithmetic(expr.left) and arithmetic(expr.right)
            return (isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub)
                    and arithmetic(expr.operand))
        return all(arithmetic(expr) for expr in value.args[0].elts)
    except (SyntaxError, AttributeError, IndexError):
        return False


def supports(ode):
    if type(ode) is PolynomialODE:
        return True
    if type(ode) is not FunctionType or ode.__closure__ or ode.__globals__.get('TMVector') is not TMVector:
        return False
    try:
        return _arithmetic_body(ode.__code__, inspect.getsource(ode))
    except (OSError, TypeError):
        return False


def _rhs_identity(ode):
    if type(ode) is PolynomialODE:
        return id(ode), ode.components, ode.state_dim
    return id(ode), ode.__code__, id(ode.__globals__.get('TMVector'))


class _Polynomial(d.BatchedPolynomial):
    """Private polynomial node with a cache owned by one prepared attempt.

Inputs and outputs never escape the plan. Keys retain operand objects through
the graph, so identities cannot be recycled. Tensor versions are checked at
the plan boundary; no per-hit scan of coefficient arrays is needed.
"""
    def __init__(self, poly, owner):
        super().__init__(poly.coeffs, poly.basis)
        object.__setattr__(self, '_owner', owner)
        object.__setattr__(self, '_results', {})

    def _call(self, name, *args, **kwargs):
        def key(value):
            if isinstance(value, (torch.Tensor, d.BatchedPolynomial)):
                return id(value)
            if isinstance(value, (list, tuple)):
                return tuple(key(v) for v in value)
            return value
        signature = (name, tuple(key(v) for v in args), tuple((k, key(v)) for k, v in kwargs.items()))
        if signature in self._results:
            self._owner.hits += 1
            return self._results[signature]
        if self._owner.sealed:
            raise ValueError('prepared polynomial graph changed during replay')
        result = getattr(super(), name)(*args, **kwargs)
        def wrap(value):
            if isinstance(value, d.BatchedPolynomial):
                return _Polynomial(value, self._owner)
            if isinstance(value, tuple):
                return tuple(wrap(v) for v in value)
            return value
        result = wrap(result)
        self._results[signature] = result
        self._owner.misses += 1
        return result

    def add(self, *args, **kwargs): return self._call('add', *args, **kwargs)
    def sub(self, *args, **kwargs): return self._call('sub', *args, **kwargs)
    def scale(self, *args, **kwargs): return self._call('scale', *args, **kwargs)
    def component(self, *args, **kwargs): return self._call('component', *args, **kwargs)
    def mul_trunc(self, *args, **kwargs): return self._call('mul_trunc', *args, **kwargs)
    def range_bound(self, *args, **kwargs): return self._call('range_bound', *args, **kwargs)
    def apply_cutoff(self, *args, **kwargs): return self._call('apply_cutoff', *args, **kwargs)
    def integrate(self, *args, **kwargs): return self._call('integrate', *args, **kwargs)


class _Scalar:
    """Compile the actual PolynomialODE operator order, without rewriting it."""
    def __init__(self, tape, index):
        self.tape, self.index = tape, index

    @property
    def domain(self):
        return self.tape.values[self.index].domain

    def _coerce(self, other):
        if isinstance(other, _Scalar):
            if other.tape is not self.tape:
                raise ValueError('prepared expressions cannot cross attempts')
            return other
        return self.tape.constant(other, self.index)

    def __add__(self, other):
        return self.tape.operation('add', self.index, self._coerce(other).index)
    __radd__ = __add__

    def __sub__(self, other):
        return self.tape.operation('sub', self.index, self._coerce(other).index)

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)

    def __neg__(self):
        return self.tape.operation('scale', self.index, scalar=-1.0)

    def __mul__(self, other):
        result = self.tape.operation('mul_trunc', self.index, self._coerce(other).index,
                                     max_degree=self.tape.degree)
        if self.tape.raw:
            result = self.tape.operation('apply_cutoff', result.index, threshold=self.tape.cutoff)
        return result
    __rmul__ = __mul__

    @staticmethod
    def concat(values):
        return values


class _Tape:
    def __init__(self, ode, candidate, owner, *, raw, order, cutoff):
        self.raw, self.degree, self.cutoff = raw, max(order-1, 0) if raw else None, cutoff
        self.owner = owner
        self.instructions = []
        self.values = []
        for index in range(candidate.poly.out_dim):
            self.instructions.append(('input', index))
            self.values.append(candidate.component(index))
        state = [_Scalar(self, i) for i in range(candidate.poly.out_dim)]
        # Match both original dense RHS adapters, including their fallback for
        # an unused but required control parameter.
        try:
            outputs = ode(state)
        except TypeError as one_argument_error:
            try:
                outputs = ode(state, None)
            except TypeError:
                raise one_argument_error
        if hasattr(outputs, 'models'):
            outputs = outputs.models
        if len(outputs) != candidate.poly.out_dim:
            raise ValueError('prepared RHS output dimension mismatch')
        self.outputs = tuple(x.index for x in outputs)
        # First execution is the first real replay, not a discarded R0 run.
        self.first_result = self._concat(self.values)
        self.concat_poly = _Polynomial(self.first_result.poly, owner)
        self.first_result = replace(self.first_result, poly=self.concat_poly)
        self.values = []  # Do not keep any seed-R dynamic intermediates.

    def constant(self, value, template):
        model = d.BatchedTaylorModel.constants_like(value, self.values[template])
        model = replace(model, poly=_Polynomial(model.poly, self.owner))
        index = len(self.values)
        self.instructions.append(('constant', model))
        self.values.append(model)
        return _Scalar(self, index)

    def operation(self, name, left, right=None, **kwargs):
        operands = () if right is None else (self.values[right],)
        result = getattr(self.values[left], name)(*operands, **kwargs)
        index = len(self.values)
        self.instructions.append(('operation', name, left, right, kwargs))
        self.values.append(result)
        return _Scalar(self, index)

    def _concat(self, values):
        # Retain all original dynamic category concatenation and checks.
        return d.BatchedTaylorModel.concat([values[i] for i in self.outputs])

    def evaluate(self, candidate):
        values = []
        for instruction in self.instructions:
            if instruction[0] == 'input':
                value = candidate.component(instruction[1])
            elif instruction[0] == 'constant':
                value = instruction[1]
            else:
                _, name, left, right, kwargs = instruction
                operands = () if right is None else (values[right],)
                value = getattr(values[left], name)(*operands, **kwargs)
            values.append(value)
        # The component point polynomials are private fixed graph nodes. The
        # concat above still constructs its dynamic ledger on every replay.
        return replace(self._concat(values), poly=self.concat_poly)


def _versions(model):
    return tuple((id(t), t._version) for t in
                 (model.poly.coeffs, model.domain_lo, model.domain_hi,
                  model.rem_lo, model.rem_hi, model.poly.basis.exponents,
                  *(t for pair in model.ledger.entries.values() for t in pair)))


class PreparedRemainderReplay:
    """Private snapshots plus two ordered execution tapes for one attempt.

Only R changes. Polynomial truncation, cutoff masks and polynomial ranges are
cached after their first actual evaluation. The regular RHS temporary ledger
is intentionally dynamic, just like the raw RHS ledger.
"""
    def __init__(self, ode, base, candidate, *, tau_index, order, cutoff_threshold, validation_eps):
        if not supports(ode):
            raise TypeError('prepared replay requires an immutable arithmetic RHS')
        self._binding = (_rhs_identity(ode), id(base), id(candidate), _versions(base), _versions(candidate),
                         tau_index, order, cutoff_threshold, validation_eps,
                         candidate.range_policy, base.range_policy)
        self._ode = ode
        # Nodes own only these counters, not a back-reference to the plan.
        # Thus exiting an attempt releases its storage without waiting for GC.
        self._cache = SimpleNamespace(sealed=False, hits=0, misses=0)
        # deepcopy, not detach: callers cannot mutate any numerical storage
        # used by this plan. Range traces are execution observations, not inputs.
        self._base = deepcopy(replace(base, range_trace=None))
        self._candidate = deepcopy(replace(candidate, range_trace=None))
        self._base = replace(self._base, poly=_Polynomial(self._base.poly, self._cache))
        self._candidate = replace(self._candidate, poly=_Polynomial(self._candidate.poly, self._cache))
        self._tau, self._order, self._cutoff = tau_index, order, cutoff_threshold
        self._raw = self._regular = None
        self._live_candidate = None
        self._guards = ()

    @property
    def hits(self):
        return self._cache.hits

    @property
    def misses(self):
        return self._cache.misses

    def validate_binding(self, ode, base, candidate, *, tau_index, order, cutoff_threshold, validation_eps):
        binding = (_rhs_identity(ode), id(base), id(candidate), _versions(base), _versions(candidate),
                   tau_index, order, cutoff_threshold, validation_eps,
                   candidate.range_policy, base.range_policy)
        if binding != self._binding:
            raise ValueError('prepared replay belongs to a different or mutated attempt')

    def raw_rhs(self, rem_lo, rem_hi):
        if self._live_candidate is not None:
            raise ValueError('prepared replay cannot be reentered')
        if any(t._version != version for t, version in self._guards):
            raise ValueError('prepared storage was modified in place')
        candidate = self._candidate.with_remainder(rem_lo, rem_hi, category='initial_remainder')
        self._live_candidate = candidate
        if self._raw is None:
            self._raw = _Tape(self._ode, candidate, self._cache, raw=True, order=self._order, cutoff=self._cutoff)
            raw = self._raw.first_result
            self._raw.first_result = None
        else:
            raw = self._raw.evaluate(candidate)
        return raw

    def polynomial_difference(self):
        # Called only after the reference raw-RHS time scaling and inflation.
        # Even the relative order of the two independent branches is retained.
        candidate = self._live_candidate
        if candidate is None:
            raise ValueError('prepared regular RHS requires a current replay')
        self._live_candidate = None
        if self._regular is None:
            self._regular = _Tape(self._ode, candidate, self._cache, raw=False, order=self._order, cutoff=self._cutoff)
            regular = self._regular.first_result
            self._regular.first_result = None
        else:
            regular = self._regular.evaluate(candidate)
        tmp = self._base.add(regular.integrate(self._tau)).apply_cutoff(self._cutoff)
        poly_diff = tmp.poly.sub(self._candidate.poly)
        diff_lo, diff_hi = poly_diff.range_bound(
            self._candidate.domain_lo, self._candidate.domain_hi,
            policy=self._candidate.range_policy, context='raw_compat_poly_diff', trace=None,
        )
        if not self._cache.sealed:
            self._seal()
        return tmp.ledger, poly_diff, diff_lo, diff_hi

    def parts(self, rem_lo, rem_hi):
        """Inspection helper; production calls the branches in reference order."""
        raw = self.raw_rhs(rem_lo, rem_hi)
        return raw, *self.polynomial_difference()

    def _seal(self):
        # Version checks examine metadata, not coefficient arrays. No result
        # from an R-dependent computation enters this fixed-storage guard set.
        tensors, seen = [], set()
        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                tensors.append((value, value._version))
            elif isinstance(value, _Polynomial):
                visit(value.coeffs)
                for field in value.basis.__dict__.values():
                    if isinstance(field, torch.Tensor):
                        visit(field)
                visit(value._results)
            elif isinstance(value, dict):
                for item in value.values(): visit(item)
            elif isinstance(value, (tuple, list)):
                for item in value: visit(item)
        visit(self._base.poly)
        visit(self._candidate.poly)
        for tape in (self._raw, self._regular):
            visit(tape.concat_poly)
            for instruction in tape.instructions:
                if instruction[0] == 'constant':
                    visit(instruction[1].poly)
        visit((self._base.domain_lo, self._base.domain_hi,
               self._candidate.domain_lo, self._candidate.domain_hi))
        self._guards = tuple(tensors)
        self._cache.sealed = True
