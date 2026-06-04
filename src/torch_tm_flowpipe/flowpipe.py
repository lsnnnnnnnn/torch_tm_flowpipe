"""Fixed-step Taylor-model flowpipe construction for polynomial ODE prototypes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Sequence

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


def _float_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if torch.isfinite(torch.as_tensor(f)).item() else None


def _interval_width_value(iv: Interval) -> float | None:
    return _float_or_none(iv.width().detach().cpu())


def _interval_bound_value(value: torch.Tensor) -> float | None:
    return _float_or_none(value.detach().cpu())


def _add_width_metrics(row: dict[str, Any], prefix: str, boxes: Sequence[Interval] | None) -> None:
    if boxes is None:
        return
    names = ("x", "y")
    widths: list[float] = []
    for i, iv in enumerate(boxes[:2]):
        width = _interval_width_value(iv)
        if width is not None:
            row[f"{prefix}_width_{names[i]}"] = width
            widths.append(width)
    if widths:
        row[f"{prefix}_width_sum"] = sum(widths)


def _add_interval_bounds(row: dict[str, Any], prefix: str, boxes: Sequence[Interval] | None) -> None:
    if boxes is None:
        return
    names = ("x", "y")
    for i, iv in enumerate(boxes[:2]):
        lo = _interval_bound_value(iv.lo)
        hi = _interval_bound_value(iv.hi)
        if lo is not None:
            row[f"{prefix}_lo_{names[i]}"] = lo
        if hi is not None:
            row[f"{prefix}_hi_{names[i]}"] = hi


def _polynomial_range_boxes(tm: TMVector) -> list[Interval]:
    return [m.polynomial.evaluate_interval(m.domain) for m in tm]


def _final_range_boxes(tm: TMVector, tau_index: int, h: float) -> list[Interval] | None:
    try:
        return tm.substitute_const(tau_index, float(h)).drop_variable(tau_index).range_box()
    except Exception:
        return None


def _append_validation_diagnostic(
    diagnostics: list[dict[str, Any]] | None,
    *,
    mode: str | None,
    segment_index: int | None,
    attempt_index: int,
    h: float,
    order: int,
    candidate: TMVector | None,
    tau_index: int,
    residual_boxes: Sequence[Interval] | None,
    remainders: Sequence[Interval] | None,
    finite_residual: bool | None,
    validation_status: str,
    validation_message: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    if diagnostics is None:
        return
    row: dict[str, Any] = {
        "mode": mode or "",
        "segment_index": segment_index if segment_index is not None else "",
        "attempt_index": attempt_index,
        "h": float(h),
        "order": int(order),
        "finite_residual": finite_residual if finite_residual is not None else "",
        "validation_status": validation_status,
        "validation_message": validation_message,
    }
    if extra:
        row.update(extra)
    if candidate is not None:
        try:
            candidate_box = candidate.range_box()
            _add_width_metrics(row, "candidate_segment", candidate_box)
            _add_width_metrics(row, "total_range", candidate_box)
        except Exception:
            pass
        try:
            _add_width_metrics(row, "candidate_final", _final_range_boxes(candidate, tau_index, h))
        except Exception:
            pass
        try:
            _add_width_metrics(row, "polynomial_range", _polynomial_range_boxes(candidate))
        except Exception:
            pass
    _add_width_metrics(row, "residual", residual_boxes)
    _add_interval_bounds(row, "residual", residual_boxes)
    _add_width_metrics(row, "remainder", remainders)
    diagnostics.append(row)


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
    h: float,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    diagnostic_mode: str | None = None,
    diagnostic_segment_index: int | None = None,
    diagnostic_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
) -> tuple[TMVector, str, int, str]:
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    remainders = [m.remainder.inflate(validation_eps) for m in base_ext]
    if diagnostic_context is not None:
        diagnostics_context = diagnostic_context
    if diagnostic_mode is not None:
        diagnostics_mode = diagnostic_mode
    if diagnostic_segment_index is not None:
        diagnostics_segment_index = diagnostic_segment_index
    diag_extra = dict(diagnostics_context or {})
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)
    if not intervals_are_finite(remainders):
        message = "non-finite initial remainder"
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate_poly,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=remainders,
            finite_residual=False,
            validation_status="failed",
            validation_message=message,
            extra=diag_extra,
        )
        return candidate_poly, "failed", 0, message

    for attempt in range(1, max_attempts + 1):
        candidate = TMVector(TaylorModel(m.polynomial, r, domain, order=order) for m, r in zip(candidate_poly, remainders))
        if rhs_breakdown_callback is not None:
            callback_context = dict(diag_extra)
            if diag_mode is not None:
                callback_context["mode"] = diag_mode
            if diag_segment_index is not None:
                callback_context["segment_index"] = diag_segment_index
            callback_context["attempt_index"] = attempt
            callback_context["h"] = float(h)
            callback_context["order"] = int(order)
            try:
                rhs_breakdown_callback(candidate, order, attempt, callback_context)
            except Exception:
                pass
        try:
            rhs = _call_ode(ode_fn, candidate, u_tms)
            residual_boxes: list[Interval] = []
            for base_i, cand_i, f_i in zip(base_ext, candidate, rhs):
                picard_i = base_i + f_i.integrate(tau_index)
                residual_i = picard_i - TaylorModel(cand_i.polynomial, Interval.zero(), domain, order=order)
                residual_boxes.append(residual_i.range_box().inflate(validation_eps))
        except Exception as exc:  # fail closed; caller gets a non-validated segment
            message = f"validation exception: {exc}"
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=None,
                remainders=remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=diag_extra,
            )
            return candidate, "failed", attempt, message

        finite_residual = intervals_are_finite(residual_boxes)
        if not finite_residual:
            message = "non-finite residual interval"
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_boxes,
                remainders=remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=diag_extra,
            )
            return candidate, "failed", attempt, message

        if all(r.contains_interval(rb) for r, rb in zip(remainders, residual_boxes)):
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_boxes,
                remainders=remainders,
                finite_residual=True,
                validation_status="validated",
                validation_message="",
                extra=diag_extra,
            )
            return candidate, "validated", attempt, ""

        message = "Picard remainder validation did not converge" if attempt == max_attempts else "residual not contained by current remainder"
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=attempt,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=residual_boxes,
            remainders=remainders,
            finite_residual=True,
            validation_status="failed" if attempt == max_attempts else "needs_growth",
            validation_message=message,
            extra=diag_extra,
        )

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
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    diagnostic_mode: str | None = None,
    diagnostic_segment_index: int | None = None,
    diagnostic_context: Mapping[str, Any] | None = None,
) -> FlowpipeSegment:
    """Build one flowpipe segment from a TM initial condition.

    The returned segment preserves dependency on the variables already present in
    ``x0_tm`` and adds one local time variable.  The segment's final TM has the
    local time variable substituted with ``h`` and dropped.
    """
    if diagnostic_context is not None:
        diagnostics_context = diagnostic_context
    if diagnostic_mode is not None:
        diagnostics_mode = diagnostic_mode
    if diagnostic_segment_index is not None:
        diagnostics_segment_index = diagnostic_segment_index
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
        h=float(h),
        max_attempts=max_validation_attempts,
        validation_eps=validation_eps,
        growth_factor=growth_factor,
        diagnostics=diagnostics,
        diagnostics_mode=diagnostics_mode,
        diagnostics_segment_index=diagnostics_segment_index,
        diagnostics_context=diagnostics_context,
        rhs_breakdown_callback=rhs_breakdown_callback,
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


def _step_diagnostics_kwargs(kwargs: Mapping[str, Any], mode: str, segment_index: int) -> dict[str, Any]:
    step_kwargs = dict(kwargs)
    if step_kwargs.get("diagnostics") is not None:
        context = dict(step_kwargs.get("diagnostic_context") or step_kwargs.get("diagnostics_context") or {})
        context.setdefault("mode", mode)
        context.setdefault("segment_index", segment_index)
        step_kwargs["diagnostics_context"] = context
        step_kwargs.pop("diagnostic_context", None)
    return step_kwargs


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
        for segment_index in range(steps):
            step_kwargs = _step_diagnostics_kwargs(kwargs, mode, segment_index)
            seg = flowpipe_step(ode_fn, current_box, h, order, u_box=u_box, affine_u=affine_u, **step_kwargs)
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
    for segment_index in range(steps):
        step_kwargs = _step_diagnostics_kwargs(kwargs, mode, segment_index)
        seg = flowpipe_step_from_tm(ode_fn, current_tm, h, order, u_box=u_box, affine_u=affine_u, **step_kwargs)
        segments.append(seg)
        current_tm = seg.final_tm
    status = "validated" if all(s.status == "validated" for s in segments) else "failed"
    return FlowpipeResult(segments, status, current_tm, mode)
