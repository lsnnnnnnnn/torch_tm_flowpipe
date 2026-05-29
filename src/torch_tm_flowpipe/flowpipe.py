"""Fixed-step Taylor-model flowpipe construction for polynomial ODE prototypes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence

import torch

from .interval import Interval, ensure_interval
from .polynomial import Polynomial
from .safety import intervals_are_finite
from .taylor_model import TaylorModel
from .tm_vector import TMVector

ODEFunction = Callable[..., Sequence[TaylorModel] | TMVector]


@dataclass
class FlowpipeSegment:
    """One validated flowpipe segment.

    ``tm`` is the segment over the original dependency variables plus a local
    time variable ``tau``.  ``final_tm`` is obtained by substituting ``tau=h`` and
    dropping the local time variable, so it can be used as the initial condition
    for the next dependency-preserving step.
    """

    tm: TMVector
    final_tm: TMVector
    status: str
    h: float
    order: int
    validation_attempts: int
    message: str = ""
    tau_index: int | None = None


@dataclass
class FlowpipeResult:
    segments: List[FlowpipeSegment]
    status: str
    final_tm: TMVector
    mode: str

    @property
    def validation_attempts(self) -> int:
        return sum(seg.validation_attempts for seg in self.segments)


def _as_interval_list(x0_box: Sequence[Interval | tuple[float, float] | list[float] | float]) -> list[Interval]:
    out: list[Interval] = []
    for x in x0_box:
        if isinstance(x, Interval):
            out.append(x)
        elif isinstance(x, (tuple, list)) and len(x) == 2:
            out.append(Interval(x[0], x[1]))
        else:
            out.append(Interval.point(x))
    return out


def _zero_interval_like_domain(domain: Sequence[Interval]) -> Interval:
    if domain:
        return Interval.zero(dtype=domain[0].lo.dtype, device=domain[0].lo.device)
    return Interval.zero()


def _zero_remainder_tm(poly: Polynomial, domain: Sequence[Interval], order: int) -> TaylorModel:
    return TaylorModel(poly, _zero_interval_like_domain(domain), list(domain), order=order)


def _call_ode(ode_fn: ODEFunction, x: TMVector, u: TMVector | None) -> TMVector:
    try:
        out = ode_fn(x, u)
    except TypeError:
        out = ode_fn(x)
    if isinstance(out, TMVector):
        return out
    return TMVector(out)


def _constant_control_tms(u_box: Sequence[Any] | None, domain: Sequence[Interval], order: int) -> TMVector | None:
    if u_box is None:
        return None
    controls: list[TaylorModel] = []
    for u in u_box:
        iv = ensure_interval(u) if not isinstance(u, (tuple, list)) else Interval(u[0], u[1])
        controls.append(TaylorModel.constant(iv.mid(), domain, order=order, remainder=Interval(-iv.radius(), iv.radius())))
    return TMVector(controls)


def _affine_control_tms(affine_u: dict[str, Any] | None, domain: Sequence[Interval], order: int) -> TMVector | None:
    if affine_u is None:
        return None
    A = torch.as_tensor(affine_u.get("A"), dtype=domain[0].lo.dtype if domain else torch.float64)
    b = torch.as_tensor(affine_u.get("b", torch.zeros(A.shape[0])), dtype=A.dtype, device=A.device)
    error = affine_u.get("error", None)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if b.ndim == 0:
        b = b.reshape(1)
    n_u, n_x = A.shape
    if n_x > len(domain):
        raise ValueError("affine control has more input columns than active variables")
    variables = [TaylorModel.variable(i, domain, order=order) for i in range(n_x)]
    controls: list[TaylorModel] = []
    for j in range(n_u):
        tm = TaylorModel.constant(b[j], domain, order=order)
        for i in range(n_x):
            if float(A[j, i]) != 0.0:
                tm = tm + variables[i] * A[j, i]
        if error is not None:
            e = error[j] if isinstance(error, (list, tuple)) else error
            if isinstance(e, Interval):
                err_iv = e
            else:
                rad = torch.as_tensor(e, dtype=A.dtype, device=A.device).abs()
                err_iv = Interval(-rad, rad)
            tm = tm + err_iv
        controls.append(tm)
    return TMVector(controls)


def _make_controls(
    u_box: Sequence[Any] | None,
    affine_u: dict[str, Any] | None,
    domain: Sequence[Interval],
    order: int,
) -> TMVector | None:
    u_const = _constant_control_tms(u_box, domain, order)
    u_affine = _affine_control_tms(affine_u, domain, order)
    if u_const is not None and u_affine is not None:
        if len(u_const) != len(u_affine):
            raise ValueError("u_box and affine_u dimensions do not match")
        return TMVector(a + b for a, b in zip(u_const, u_affine))
    return u_const if u_const is not None else u_affine


def _picard_polynomial(
    ode_fn: ODEFunction,
    base_poly_ext: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    iterations: int | None = None,
) -> TMVector:
    """Construct the polynomial part of a Picard iterate.

    Remainders created by truncation are intentionally not fed back into the
    polynomial iterate.  They are accounted for by the later validation loop.
    """
    iterations = order if iterations is None else iterations
    domain = base_poly_ext.domain
    g = base_poly_ext
    for _ in range(max(1, iterations)):
        rhs = _call_ode(ode_fn, g, u_tms)
        next_models: list[TaylorModel] = []
        for x0_i, f_i in zip(base_poly_ext, rhs):
            integ = f_i.integrate(tau_index)
            tm_i = x0_i + integ
            poly, _dropped = tm_i.polynomial.truncate(order)
            next_models.append(_zero_remainder_tm(poly, domain, order))
        g = TMVector(next_models)
    return g


def _validate_picard(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate_poly: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    max_attempts: int,
    validation_eps: float,
    growth_factor: float,
) -> tuple[TMVector, str, int, str]:
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    remainders = [m.remainder.inflate(validation_eps) for m in base_ext]
    if not intervals_are_finite(remainders):
        return candidate_poly, "failed", 0, "non-finite initial remainder"

    for attempt in range(1, max_attempts + 1):
        candidate = TMVector(TaylorModel(m.polynomial, r, domain, order=order) for m, r in zip(candidate_poly, remainders))
        try:
            rhs = _call_ode(ode_fn, candidate, u_tms)
            residual_boxes: list[Interval] = []
            for base_i, cand_i, f_i in zip(base_ext, candidate, rhs):
                picard_i = base_i + f_i.integrate(tau_index)
                residual_i = picard_i - TaylorModel(cand_i.polynomial, Interval.zero(), domain, order=order)
                residual_boxes.append(residual_i.range_box().inflate(validation_eps))
        except Exception as exc:  # fail closed; caller gets a non-validated segment
            return candidate, "failed", attempt, f"validation exception: {exc}"

        if not intervals_are_finite(residual_boxes):
            return candidate, "failed", attempt, "non-finite residual interval"

        if all(r.contains_interval(rb) for r, rb in zip(remainders, residual_boxes)):
            return candidate, "validated", attempt, ""

        new_remainders: list[Interval] = []
        for r, rb in zip(remainders, residual_boxes):
            hull = Interval.hull(r, rb)
            new_remainders.append(hull.scale_about_mid(growth_factor, min_radius=validation_eps))
        remainders = new_remainders

    candidate = TMVector(TaylorModel(m.polynomial, r, domain, order=order) for m, r in zip(candidate_poly, remainders))
    return candidate, "failed", max_attempts, "Picard remainder validation did not converge"


def flowpipe_step_from_tm(
    ode_fn: ODEFunction,
    x0_tm: TMVector,
    h: float,
    order: int,
    *,
    u_box: Sequence[Any] | None = None,
    affine_u: dict[str, Any] | None = None,
    max_validation_attempts: int = 20,
    validation_eps: float = 1e-12,
    growth_factor: float = 1.25,
) -> FlowpipeSegment:
    """Build one flowpipe segment from a TM initial condition.

    The returned segment preserves dependency on the variables already present in
    ``x0_tm`` and adds one local time variable.  The segment's final TM has the
    local time variable substituted with ``h`` and dropped.
    """
    if h <= 0:
        raise ValueError("h must be positive")
    tau_interval = Interval(0.0, float(h))
    base_ext = x0_tm.extend_domain(tau_interval)
    tau_index = x0_tm.n_vars
    domain = base_ext.domain
    base_poly_ext = TMVector(
        TaylorModel(m.polynomial, Interval.zero(), domain, order=order) for m in base_ext
    )
    u_tms = _make_controls(u_box, affine_u, domain, order)
    candidate_poly = _picard_polynomial(ode_fn, base_poly_ext, tau_index, order, u_tms)
    validated, status, attempts, message = _validate_picard(
        ode_fn,
        base_ext,
        candidate_poly,
        tau_index,
        order,
        u_tms,
        max_attempts=max_validation_attempts,
        validation_eps=validation_eps,
        growth_factor=growth_factor,
    )
    final_tm = validated.substitute_const(tau_index, float(h)).drop_variable(tau_index)
    if status == "validated":
        # The segment remainder is valid for every tau in [0,h].  For multi-step
        # propagation we only need the endpoint at tau=h, so tighten the endpoint
        # remainder by re-evaluating the Picard residual at that fixed local time.
        try:
            rhs = _call_ode(ode_fn, validated, u_tms)
            final_models = []
            for base_i, cand_i, f_i in zip(base_ext, validated, rhs):
                picard_i = base_i + f_i.integrate(tau_index)
                residual_i = picard_i - TaylorModel(cand_i.polynomial, Interval.zero(), domain, order=order)
                endpoint_residual = (
                    residual_i.substitute_const(tau_index, float(h))
                    .drop_variable(tau_index)
                    .range_box()
                    .inflate(validation_eps)
                )
                endpoint_poly = cand_i.polynomial.substitute_const(tau_index, float(h)).drop_variable(tau_index)
                endpoint_domain = [d for i, d in enumerate(domain) if i != tau_index]
                final_models.append(TaylorModel(endpoint_poly, endpoint_residual, endpoint_domain, order=order))
            final_tm = TMVector(final_models)
        except Exception as exc:
            message = message or f"endpoint tightening skipped: {exc}"
    return FlowpipeSegment(
        tm=validated,
        final_tm=final_tm,
        status=status,
        h=float(h),
        order=int(order),
        validation_attempts=attempts,
        message=message,
        tau_index=tau_index,
    )


def flowpipe_step(
    ode_fn: ODEFunction,
    x0_box: Sequence[Interval | tuple[float, float] | list[float] | float],
    h: float,
    order: int,
    *,
    u_box: Sequence[Any] | None = None,
    affine_u: dict[str, Any] | None = None,
    **kwargs: Any,
) -> FlowpipeSegment:
    """Build one validated segment from an interval-box initial set."""
    domain = _as_interval_list(x0_box)
    x0_tm = TMVector.identity(domain, order=order)
    return flowpipe_step_from_tm(ode_fn, x0_tm, h, order, u_box=u_box, affine_u=affine_u, **kwargs)


def flowpipe_multi_step(
    ode_fn: ODEFunction,
    x0_box: Sequence[Interval | tuple[float, float] | list[float] | float],
    h: float,
    steps: int,
    order: int,
    *,
    mode: str = "range_only",
    u_box: Sequence[Any] | None = None,
    affine_u: dict[str, Any] | None = None,
    range_only_inflate: float = 1e-9,
    **kwargs: Any,
) -> FlowpipeResult:
    """Repeatedly propagate a polynomial ODE.

    ``mode='range_only'`` keeps the old baseline behavior: each step compresses
    the previous final Taylor model to a box and restarts with fresh identity
    variables.  ``mode='dependency_preserving'`` propagates the final Taylor
    model directly and therefore keeps symbolic dependency on the original
    initial-state variables.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if mode not in {"range_only", "dependency_preserving"}:
        raise ValueError("mode must be 'range_only' or 'dependency_preserving'")

    segments: list[FlowpipeSegment] = []
    if mode == "range_only":
        current_box = _as_interval_list(x0_box)
        current_final: TMVector | None = None
        for _ in range(steps):
            seg = flowpipe_step(ode_fn, current_box, h, order, u_box=u_box, affine_u=affine_u, **kwargs)
            segments.append(seg)
            current_box = [iv.inflate(range_only_inflate) for iv in seg.final_tm.range_box()]
            # The range-only baseline intentionally forgets symbolic dependency at
            # step boundaries.  Represent the compressed box as fresh identity TMs
            # so returned widths match the actual state passed to the next step.
            current_final = TMVector.identity(current_box, order=order)
        assert current_final is not None
        status = "validated" if all(s.status == "validated" for s in segments) else "failed"
        return FlowpipeResult(segments, status, current_final, mode)

    current_tm = TMVector.identity(_as_interval_list(x0_box), order=order)
    for _ in range(steps):
        seg = flowpipe_step_from_tm(ode_fn, current_tm, h, order, u_box=u_box, affine_u=affine_u, **kwargs)
        segments.append(seg)
        current_tm = seg.final_tm
    status = "validated" if all(s.status == "validated" for s in segments) else "failed"
    return FlowpipeResult(segments, status, current_tm, mode)
