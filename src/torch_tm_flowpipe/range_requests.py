"""Local ordered range batches. No solver scheduler and no numerical cache.

The CPU implementation retains RangePlan operation order and certifies its scalar
powers using exact dyadic arithmetic. Only a proven under-enclosure is widened.
The original scalar/RangePlan APIs and their public defaults are unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import math
import time
from typing import Mapping, Sequence

import torch

from .interval import Interval
from .packed_boundary_range import make_plan


@dataclass(frozen=True)
class RangeRequest:
    request_id: str
    exponents: tuple[tuple[int, ...], ...]
    coefficients_lo: torch.Tensor  # [output, term], including explicit zeros
    coefficients_hi: torch.Tensor
    domain_lo: torch.Tensor  # [variable]
    domain_hi: torch.Tensor
    kind: str = "standard"
    state_variables: tuple[int, ...] = ()
    time_variable: int | None = None
    step_powers: Mapping[int, Interval] | None = None
    enabled: bool = True
    cancelled: bool = False


@dataclass(frozen=True)
class RangeResult:
    request_id: str
    status: str
    lo: torch.Tensor | None = None
    hi: torch.Tensor | None = None
    message: str = ""
    terms_lo: torch.Tensor | None = None
    terms_hi: torch.Tensor | None = None
    powers: tuple = ()

    @property
    def ok(self):
        return self.status in {"ok", "corrected", "fallback", "fallback_corrected"}


@dataclass(frozen=True)
class PackedGroup:
    key: tuple
    requests: tuple[RangeRequest, ...]
    plan: object
    coefficients_lo: torch.Tensor
    coefficients_hi: torch.Tensor
    domain_lo: torch.Tensor
    domain_hi: torch.Tensor


def structure_key(request: RangeRequest):
    """All structural semantics; numeric power-table values are never cached."""
    return (request.exponents, request.domain_lo.numel(), request.kind,
            request.state_variables, request.time_variable,
            ("external", tuple(sorted(request.step_powers))) if request.step_powers is not None else ("domain",),
            tuple(request.coefficients_lo.shape), str(request.coefficients_lo.dtype),
            str(request.coefficients_lo.device))


def _validate_structure(r):
    if r.kind not in {"standard", "normal", "interval-coefficient"}:
        raise ValueError("unknown range kind")
    tensors = (r.coefficients_lo, r.coefficients_hi, r.domain_lo, r.domain_hi)
    if any(not isinstance(t, torch.Tensor) or t.dtype != torch.float64 or t.device.type != "cpu" for t in tensors):
        raise ValueError("request interface requires CPU binary64 tensors")
    if r.coefficients_lo.ndim != 2 or r.coefficients_lo.shape[0] < 1:
        raise ValueError("coefficients must be [positive outputs, terms]")
    if r.coefficients_hi.shape != r.coefficients_lo.shape or r.coefficients_lo.shape[1] != len(r.exponents):
        raise ValueError("coefficient/support shape mismatch")
    if r.domain_lo.ndim != 1 or r.domain_hi.shape != r.domain_lo.shape:
        raise ValueError("domain must be [variables]")
    n = r.domain_lo.numel()
    if not isinstance(r.exponents, tuple) or any(not isinstance(e, tuple) or len(e) != n or
            any(type(p) is not int or p < 0 or p > 64 for p in e) for e in r.exponents):
        raise ValueError("immutable exponent tuples with integer powers 0..64 required")
    if len(set(r.exponents)) != len(r.exponents):
        raise ValueError("duplicate support entries")
    if not isinstance(r.state_variables, tuple) or tuple(sorted(set(r.state_variables))) != r.state_variables or any(
            type(v) is not int or not 0 <= v < n for v in r.state_variables):
        raise ValueError("state variables must be sorted distinct valid indices")
    if r.time_variable is not None and (type(r.time_variable) is not int or not 0 <= r.time_variable < n):
        raise ValueError("invalid time variable")
    if r.time_variable in r.state_variables:
        raise ValueError("time and state roles must be disjoint")
    if r.kind != "normal" and (r.state_variables or r.time_variable is not None or r.step_powers is not None):
        raise ValueError("time/state/table semantics require normal kind")
    if type(r.enabled) is not bool or type(r.cancelled) is not bool:
        raise ValueError("mask and cancellation must be bool")
    if r.step_powers is not None:
        if r.time_variable is None or not isinstance(r.step_powers, Mapping):
            raise ValueError("external power table requires a time variable and mapping")
        for p, iv in r.step_powers.items():
            if type(p) is not int or not 0 <= p <= 64 or not isinstance(iv, Interval) or any(
                    t.numel() != 1 or t.dtype != torch.float64 or t.device.type != "cpu" for t in (iv.lo, iv.hi)):
                raise ValueError("invalid external power table")


def exact_power(a, b, exponent):
    """Exact range of x**exponent, for finite dyadic endpoints."""
    a, b = Fraction(a), Fraction(b)
    if exponent == 0:
        return Fraction(1), Fraction(1)
    ends = a ** exponent, b ** exponent
    return (Fraction(0) if exponent % 2 == 0 and a <= 0 <= b else min(ends), max(ends))


def _enlarge_to_exact(lo, hi, exact_lo, exact_hi):
    """Only repair an endpoint which fails its exact rational inequality."""
    changed = False
    if not math.isfinite(lo) or Fraction(lo) > exact_lo:
        lo = float(exact_lo)
        if not math.isfinite(lo):
            raise OverflowError("no finite binary64 lower enclosure")
        if Fraction(lo) > exact_lo:
            lo = math.nextafter(lo, -math.inf)
        changed = True
    if not math.isfinite(hi) or Fraction(hi) < exact_hi:
        hi = float(exact_hi)
        if not math.isfinite(hi):
            raise OverflowError("no finite binary64 upper enclosure")
        if Fraction(hi) < exact_hi:
            hi = math.nextafter(hi, math.inf)
        changed = True
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise OverflowError("no finite binary64 enclosure")
    return lo, hi, changed


def certified_scalar_power(a, b, exponent):
    """Legacy scalar power, with a proved correction when one ULP was insufficient."""
    legacy = Interval(a, b).pow_int(exponent)
    lo, hi, changed = _enlarge_to_exact(float(legacy.lo), float(legacy.hi),
                                       *exact_power(float(a), float(b), exponent))
    return lo, hi, changed


def _finite_rows(lo, hi):
    return (torch.isfinite(lo) & torch.isfinite(hi) & (lo <= hi)).reshape(lo.shape[0], -1).all(dim=1)


def prepare_range_requests(requests: Sequence[RangeRequest]):
    """Validate, group and pack privately. Masked/cancelled rows never enter math."""
    requests = tuple(requests)
    ids = [r.request_id for r in requests]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("nonempty unique request IDs required")
    grouped, results, fallbacks = {}, {}, []
    for r in requests:
        if r.cancelled is True or r.enabled is False:
            results[r.request_id] = RangeResult(r.request_id, "cancelled" if r.cancelled else "masked")
            continue
        try:
            _validate_structure(r)
            grouped.setdefault(structure_key(r), []).append(r)
        except (TypeError, ValueError, IndexError) as error:
            results[r.request_id] = RangeResult(r.request_id, "invalid", message=str(error))
    groups = []
    for key, rows in grouped.items():
        cl, ch, dl, dh = (torch.stack([getattr(r, field).detach() for r in rows]) for field in
                           ("coefficients_lo", "coefficients_hi", "domain_lo", "domain_hi"))
        valid = _finite_rows(cl, ch) & _finite_rows(dl, dh)
        if rows[0].kind != "interval-coefficient":
            valid &= (cl == ch).reshape(len(rows), -1).all(dim=1)
        if rows[0].kind == "normal" and rows[0].state_variables:
            s = list(rows[0].state_variables)
            valid &= ((dl[:, s] >= -1) & (dh[:, s] <= 1)).all(dim=1)
        indices = []
        for b, ok in enumerate(valid.tolist()):
            if not ok:
                results[rows[b].request_id] = RangeResult(rows[b].request_id, "invalid",
                    message="nonfinite/reversed interval, nonpoint coefficient, or nonnormalized state domain")
            elif rows[b].step_powers is not None:
                fallbacks.append(rows[b])
            else:
                indices.append(b)
        if indices:
            r = rows[0]
            plan = make_plan(r.exponents, dl.shape[1], r.state_variables if r.kind == "normal" else None, r.time_variable)
            groups.append(PackedGroup(key, tuple(rows[i] for i in indices), plan,
                                     cl[indices], ch[indices], dl[indices], dh[indices]))
    return groups, fallbacks, results


def compute_cpu_group(group: PackedGroup, *, diagnostics=False):
    """Shared tensor term multiplications and ordered sums; failures are per lane."""
    cl, ch, dl, dh = group.coefficients_lo, group.coefficients_hi, group.domain_lo, group.domain_hi
    batch, outputs, terms = cl.shape
    tl, th = cl.clone(), ch.clone()
    active = torch.ones(batch, dtype=torch.bool)
    corrected = [False] * batch
    power_records = [[] for _ in range(batch)]
    powers = {}
    for kind, variable, indices, values in group.plan.stages:
        if kind == "power":
            for p in sorted(set(values)):
                pl, ph = torch.zeros(batch, dtype=torch.float64), torch.zeros(batch, dtype=torch.float64)
                for b in active.nonzero().flatten().tolist():
                    try:
                        a, z, fix = certified_scalar_power(dl[b, variable], dh[b, variable], p)
                        pl[b], ph[b] = a, z
                        corrected[b] |= fix
                        if diagnostics:
                            power_records[b].append((variable, p, a, z, fix))
                    except (ValueError, OverflowError):
                        active[b] = False
                powers[variable, p] = pl, ph
            vl = torch.stack([powers[variable, p][0] for p in values], dim=-1)[:, None, :]
            vh = torch.stack([powers[variable, p][1] for p in values], dim=-1)[:, None, :]
        else:
            vl = torch.tensor(values, dtype=torch.float64).view(1, 1, -1).expand(batch, 1, -1)
            vh = torch.ones_like(vl)
        live = active.nonzero().flatten()
        if not len(live):
            break
        selected = list(indices)
        a, z = tl[live][:, :, selected], th[live][:, :, selected]
        l, h = vl[live], vh[live]
        products = torch.stack(torch.broadcast_tensors(a*l, a*h, z*l, z*h))
        low = torch.nextafter(products.min(dim=0).values, torch.full_like(a, -torch.inf))
        high = torch.nextafter(products.max(dim=0).values, torch.full_like(z, torch.inf))
        valid = _finite_rows(low, high)
        # Assign only valid lanes; failed lanes never join a subsequent stage.
        for pos, b in enumerate(live.tolist()):
            if valid[pos]:
                tl[b, :, selected], th[b, :, selected] = low[pos], high[pos]
            else:
                active[b] = False
    lo = torch.zeros((batch, outputs), dtype=torch.float64)
    hi = torch.zeros_like(lo)
    for index in range(terms):
        live = active.nonzero().flatten()
        if not len(live):
            break
        low = torch.nextafter(lo[live] + tl[live, :, index], torch.full_like(lo[live], -torch.inf))
        high = torch.nextafter(hi[live] + th[live, :, index], torch.full_like(hi[live], torch.inf))
        valid = _finite_rows(low, high)
        lo[live], hi[live] = low, high
        active[live] &= valid
    return lo, hi, tl, th, active, corrected, power_records


def _original_table_fallback(r):
    from .polynomial import Polynomial, evaluate_interval_normal
    from .packed_boundary_range import packed_boundary_execution
    domain = [Interval(a, b) for a, b in zip(r.domain_lo, r.domain_hi)]
    for p, iv in r.step_powers.items():
        if not iv.is_finite():
            raise ValueError("external power table must be finite")
        exact = exact_power(float(r.domain_lo[r.time_variable]), float(r.domain_hi[r.time_variable]), p)
        if Fraction(float(iv.lo)) > exact[0] or Fraction(float(iv.hi)) < exact[1]:
            raise ValueError("external table does not enclose the specified time domain power")
    lows, highs, changed = [], [], False
    for output in range(r.coefficients_lo.shape[0]):
        poly = Polynomial({}, r.domain_lo.numel())
        # A fresh object preserves the request's explicit zero terms and order.
        poly.terms.update({e: r.coefficients_lo[output, i].clone() for i, e in enumerate(r.exponents)})
        with packed_boundary_execution(False):
            original = evaluate_interval_normal(poly, domain, step_exp_table=r.step_powers,
                state_var_indices=r.state_variables, time_var_index=r.time_variable)
        # Independent exact containment check of this fallback's mathematical
        # operation, including the supplied (possibly wider) intervals.
        total = [Fraction(0), Fraction(0)]
        for i, e in enumerate(r.exponents):
            iv = Fraction(float(r.coefficients_lo[output, i])), Fraction(float(r.coefficients_hi[output, i]))
            factors = []
            for v, p in enumerate(e):
                if p and v not in r.state_variables:
                    factors.append(tuple(Fraction(float(x)) for x in (r.step_powers[p].lo, r.step_powers[p].hi))
                        if v == r.time_variable and p in r.step_powers else
                        exact_power(float(r.domain_lo[v]), float(r.domain_hi[v]), p))
            factors.append((Fraction(-1) if any(e[v] % 2 for v in r.state_variables)
                else Fraction(0) if any(e[v] for v in r.state_variables) else Fraction(1), Fraction(1)))
            for a, b in factors:
                products = (iv[0]*a, iv[0]*b, iv[1]*a, iv[1]*b)
                iv = min(products), max(products)
            total[0] += iv[0]; total[1] += iv[1]
        a, b, fix = _enlarge_to_exact(float(original.lo), float(original.hi), *total)
        lows.append(a); highs.append(b); changed |= fix
    return RangeResult(r.request_id, "fallback_corrected" if changed else "fallback",
        torch.tensor(lows, dtype=torch.float64), torch.tensor(highs, dtype=torch.float64),
        "external step-power table evaluated by original scalar normal evaluator")


def evaluate_range_requests(requests: Sequence[RangeRequest], *, backend="cpu", diagnostics=False, timings=None):
    """Return private per-ID intervals/status. Full preparation is included.

    ``backend='cuda'`` uses the same grouping with an independently conservative
    directed-rounding kernel. ``diagnostics`` records powers and individual terms.
    Cancellation is checked at submission; this synchronous API has no scheduler.
    """
    if backend not in {"cpu", "cuda"}:
        raise ValueError("backend must be cpu or cuda")
    start = time.perf_counter()
    groups, fallbacks, results = prepare_range_requests(requests)
    packed = time.perf_counter()
    compute_seconds = scatter_seconds = 0.
    for group in groups:
        t = time.perf_counter()
        if backend == "cpu":
            lo, hi, tl, th, active, corrected, powers = compute_cpu_group(group, diagnostics=diagnostics)
        else:
            from .range_cuda import compute_cuda_group
            lo, hi, tl, th, active, corrected, powers = compute_cuda_group(group, diagnostics=diagnostics, timings=timings)
        compute_seconds += time.perf_counter() - t
        t = time.perf_counter()
        for b, r in enumerate(group.requests):
            if not bool(active[b]):
                results[r.request_id] = RangeResult(r.request_id, "overflow", message="no finite enclosure from this arithmetic chain")
            else:
                results[r.request_id] = RangeResult(r.request_id, "corrected" if corrected[b] else "ok",
                    lo[b].clone(), hi[b].clone(), terms_lo=tl[b].clone() if diagnostics else None,
                    terms_hi=th[b].clone() if diagnostics else None, powers=tuple(powers[b]))
        scatter_seconds += time.perf_counter() - t
    fallback_start = time.perf_counter()
    for r in fallbacks:
        try:
            results[r.request_id] = _original_table_fallback(r)
        except (ValueError, OverflowError) as error:
            results[r.request_id] = RangeResult(r.request_id, "invalid", message=str(error))
    if timings is not None:
        timings.update(grouping_and_packing_s=packed-start, compute_and_transfers_s=compute_seconds,
            scatter_and_wrap_s=scatter_seconds, fallback_s=time.perf_counter()-fallback_start,
            group_sizes=[len(g.requests) for g in groups], total_s=time.perf_counter()-start)
    return results


@contextmanager
def range_request_execution(backend="cpu"):
    """Opt-in real solver B1 adapter. Other solver work stays on the CPU."""
    if backend not in {"cpu", "cuda"}:
        raise ValueError("unknown range backend")
    from .packed_boundary_range import _REQUEST_DISPATCH

    def dispatch(exponents, cl, ch, dl, dh, states, time_variable, kind):
        request = RangeRequest("solver-call", exponents, cl[0], ch[0], dl[0], dh[0], kind,
                               states or (), time_variable)
        result = evaluate_range_requests([request], backend=backend)[request.request_id]
        if not result.ok:
            raise FloatingPointError(result.message)
        return Interval(result.lo[0], result.hi[0])

    token = _REQUEST_DISPATCH.set(dispatch)
    try:
        yield
    finally:
        _REQUEST_DISPATCH.reset(token)
