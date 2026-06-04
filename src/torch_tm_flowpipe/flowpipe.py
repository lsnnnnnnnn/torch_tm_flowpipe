"""Fixed-step Taylor-model flowpipe construction for polynomial ODE prototypes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Sequence

import torch

from .interval import Interval, ensure_interval
from .polynomial import Polynomial
from .safety import intervals_are_finite
from .symbolic_remainder import SymbolicRemainderState, introduce_symbolic_remainders
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
    symbolic_remainder: bool = False
    symbolic_remainder_state: SymbolicRemainderState | None = None
    symbolic_remainder_stats: Mapping[str, Any] | None = None
    reset_tm: TMVector | None = None
    next_h: float | None = None
    step_rejections: int = 0


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


def _interval_is_zero(iv: Interval) -> bool:
    return bool(torch.all(iv.lo == 0) and torch.all(iv.hi == 0))


def _combine_remainders(base: Interval, extra: Interval) -> Interval:
    return base if _interval_is_zero(extra) else base + extra


def _symmetric_interval(radius: float, domain: Sequence[Interval]) -> Interval:
    r = abs(float(radius))
    if domain:
        dtype = domain[0].lo.dtype
        device = domain[0].lo.device
        return Interval(
            torch.as_tensor(-r, dtype=dtype, device=device),
            torch.as_tensor(r, dtype=dtype, device=device),
        )
    return Interval(-r, r)


def _unit_interval_like(iv: Interval) -> Interval:
    return Interval(
        torch.as_tensor(-1.0, dtype=iv.lo.dtype, device=iv.lo.device),
        torch.as_tensor(1.0, dtype=iv.lo.dtype, device=iv.lo.device),
    )


def _normalized_tm_from_box(x_box: Sequence[Interval | tuple[float, float] | list[float] | float], order: int) -> TMVector:
    boxes = _as_interval_list(x_box)
    var_for_dim: list[int | None] = []
    domain: list[Interval] = []
    for iv in boxes:
        if bool(torch.all(iv.radius() == 0)):
            var_for_dim.append(None)
        else:
            var_for_dim.append(len(domain))
            domain.append(_unit_interval_like(iv))

    models: list[TaylorModel] = []
    n_vars = len(domain)
    for iv, var_index in zip(boxes, var_for_dim):
        center = iv.mid()
        if var_index is None:
            models.append(TaylorModel.constant(center, domain, order=order))
            continue
        radius = iv.radius()
        poly = Polynomial.constant(center, n_vars) + Polynomial.variable(
            var_index, n_vars, dtype=center.dtype, device=center.device
        ) * radius
        models.append(TaylorModel(poly, Interval.zero(dtype=center.dtype, device=center.device), domain, order=order))
    return TMVector(models)


def _sum_interval_widths(boxes: Sequence[Interval]) -> float | str:
    widths = [_interval_width_value(iv) for iv in boxes]
    finite = [w for w in widths if w is not None]
    return sum(finite) if len(finite) == len(boxes) else ""


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
    cutoff_threshold: float | None = None,
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
            next_tm = _zero_remainder_tm(poly, domain, order).apply_cutoff(cutoff_threshold)
            next_models.append(next_tm)
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
    symbolic_remainder: bool = False,
    max_symbolic_remainders: int = 0,
) -> tuple[TMVector, str, int, str]:
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    remainders: list[Interval] = []
    for base_i, candidate_i in zip(base_ext, candidate_poly):
        remainders.append(_combine_remainders(base_i.remainder, candidate_i.remainder).inflate(validation_eps))
    if diagnostic_context is not None:
        diagnostics_context = diagnostic_context
    if diagnostic_mode is not None:
        diagnostics_mode = diagnostic_mode
    if diagnostic_segment_index is not None:
        diagnostics_segment_index = diagnostic_segment_index
    diag_extra = dict(diagnostics_context or {})
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
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


def _validate_picard_target_remainder(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate_poly: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    max_attempts: int,
    validation_eps: float,
    h: float,
    target_remainder_radius: float,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    symbolic_remainder: bool = False,
    max_symbolic_remainders: int = 0,
) -> tuple[TMVector, str, int, str]:
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    target_remainders = [_symmetric_interval(target_remainder_radius, domain) for _ in candidate_poly]
    diag_extra = dict(diagnostics_context or {})
    diag_extra.setdefault("validation_mode", "target_remainder")
    diag_extra.setdefault("target_remainder_radius", abs(float(target_remainder_radius)))
    diag_extra.setdefault("target_remainder_width", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_remainder_width_sum", _sum_interval_widths(target_remainders))
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)

    seed_remainders = [_combine_remainders(base_i.remainder, candidate_i.remainder) for base_i, candidate_i in zip(base_ext, candidate_poly)]
    candidate = TMVector(TaylorModel(m.polynomial, r, domain, order=order) for m, r in zip(candidate_poly, target_remainders))
    if not intervals_are_finite(seed_remainders):
        message = "non-finite initial remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=target_remainders,
            finite_residual=False,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message
    if not all(target.contains_interval(seed) for target, seed in zip(target_remainders, seed_remainders)):
        message = "initial or cutoff remainder exceeds target remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=target_remainders,
            finite_residual=True,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message

    for attempt in range(1, max_attempts + 1):
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
        except Exception as exc:
            message = f"validation exception: {exc}"
            extra = dict(diag_extra, subset_result=False, rejection_reason=message)
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
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        finite_residual = intervals_are_finite(residual_boxes)
        if not finite_residual:
            message = "non-finite residual interval"
            extra = dict(diag_extra, subset_result=False, rejection_reason=message)
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
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        subset_result = all(target.contains_interval(rb) for target, rb in zip(target_remainders, residual_boxes))
        message = "" if subset_result else "Picard residual not subset of target remainder"
        extra = dict(diag_extra, subset_result=subset_result, rejection_reason="" if subset_result else message)
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
            remainders=target_remainders,
            finite_residual=True,
            validation_status="validated" if subset_result else "failed",
            validation_message=message,
            extra=extra,
        )
        if subset_result:
            return candidate, "validated", attempt, ""

    return candidate, "failed", max_attempts, "Picard residual not subset of target remainder"


def flowpipe_step_from_tm(
    ode_fn: ODEFunction,
    x0_tm: TMVector,
    h: float,
    order: int,
    *,
    u_box: Sequence[Any] | None = None,
    affine_u: dict[str, Any] | None = None,
    max_validation_attempts: int | None = None,
    validation_eps: float = 1e-12,
    growth_factor: float = 1.25,
    validation_mode: str = "growth",
    target_remainder_radius: float = 1e-4,
    cutoff_threshold: float | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    symbolic_remainder: bool = False,
    max_symbolic_remainders: int = 0,
    symbolic_remainder_state: SymbolicRemainderState | None = None,
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
    if validation_mode not in {"growth", "current", "target_remainder"}:
        raise ValueError("validation_mode must be 'growth', 'current', or 'target_remainder'")
    target_mode = validation_mode == "target_remainder"
    attempt_limit = (2 if target_mode else 20) if max_validation_attempts is None else int(max_validation_attempts)
    if attempt_limit <= 0:
        raise ValueError("max_validation_attempts must be positive")

    u_tms = _make_controls(u_box, affine_u, domain, order)
    candidate_poly = _picard_polynomial(
        ode_fn,
        base_poly_ext,
        tau_index,
        order,
        u_tms,
        cutoff_threshold=cutoff_threshold,
    )
    if target_mode:
        validated, status, attempts, message = _validate_picard_target_remainder(
            ode_fn,
            base_ext,
            candidate_poly,
            tau_index,
            order,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            target_remainder_radius=target_remainder_radius,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diagnostics_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    else:
        validated, status, attempts, message = _validate_picard(
            ode_fn,
            base_ext,
            candidate_poly,
            tau_index,
            order,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            growth_factor=growth_factor,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diagnostics_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    final_tm = validated.substitute_const(tau_index, float(h)).drop_variable(tau_index)
    final_tm = final_tm.apply_cutoff(cutoff_threshold)
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
            final_tm = TMVector(final_models).apply_cutoff(cutoff_threshold)
        except Exception as exc:
            message = message or f"endpoint tightening skipped: {exc}"
    next_symbolic_state = symbolic_remainder_state
    symbolic_stats: Mapping[str, Any] | None = None
    if symbolic_remainder:
        if status == "validated":
            final_tm, next_symbolic_state, symbolic_stats = introduce_symbolic_remainders(
                final_tm,
                symbolic_remainder_state,
                max_symbolic_remainders=max_symbolic_remainders,
            )
        else:
            next_symbolic_state = symbolic_remainder_state or SymbolicRemainderState.empty(max_symbolic_remainders)
            symbolic_stats = {
                "introduced_symbols": 0,
                "active_noise_symbols": len(next_symbolic_state.symbols),
                "symbolic_remainder_width_sum": "",
                "ordinary_remainder_width_sum": "",
                "materialized_remainder_width_sum": "",
            }
    return FlowpipeSegment(
        tm=validated,
        final_tm=final_tm,
        status=status,
        h=float(h),
        order=int(order),
        validation_attempts=attempts,
        message=message,
        tau_index=tau_index,
        symbolic_remainder=bool(symbolic_remainder),
        symbolic_remainder_state=next_symbolic_state,
        symbolic_remainder_stats=symbolic_stats,
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


def flowpipe_step_flowstar_style_adaptive(
    ode_fn: ODEFunction,
    x0: TMVector | Sequence[Interval | tuple[float, float] | list[float] | float],
    h: float | None = None,
    order: int = 4,
    *,
    u_box: Sequence[Any] | None = None,
    affine_u: dict[str, Any] | None = None,
    h_min: float = 0.002,
    h_max: float = 0.1,
    target_remainder_radius: float = 1e-4,
    cutoff_threshold: float | None = 1e-10,
    max_validation_attempts: int = 2,
    validation_eps: float = 1e-12,
    grow_factor: float = 1.5,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
) -> FlowpipeSegment:
    if h_min <= 0 or h_max <= 0:
        raise ValueError("h_min and h_max must be positive")
    if h_min > h_max:
        raise ValueError("h_min must be <= h_max")
    current_tm = x0 if isinstance(x0, TMVector) else _normalized_tm_from_box(x0, order)
    h_try = min(float(h) if h is not None else float(h_max), float(h_max))
    if h_try < h_min:
        raise ValueError("initial h is below h_min")

    last_seg: FlowpipeSegment | None = None
    rejections = 0
    adaptive_attempt = 0
    while h_try + 1e-15 >= h_min:
        adaptive_attempt += 1
        context = dict(diagnostics_context or {})
        context.setdefault("mode", "flowstar_style")
        context["adaptive_attempt_index"] = adaptive_attempt
        context["h_try"] = h_try
        context["h_min"] = float(h_min)
        context["h_max"] = float(h_max)
        seg = flowpipe_step_from_tm(
            ode_fn,
            current_tm,
            h_try,
            order,
            u_box=u_box,
            affine_u=affine_u,
            max_validation_attempts=max_validation_attempts,
            validation_eps=validation_eps,
            validation_mode="target_remainder",
            target_remainder_radius=target_remainder_radius,
            cutoff_threshold=cutoff_threshold,
            diagnostics=diagnostics,
            diagnostics_context=context,
            rhs_breakdown_callback=rhs_breakdown_callback,
        )
        seg.step_rejections = rejections
        if seg.status == "validated" and intervals_are_finite(seg.final_tm.range_box()):
            seg.reset_tm = _normalized_tm_from_box(seg.final_tm.range_box(), order)
            seg.next_h = min(h_try * grow_factor, h_max)
            return seg
        last_seg = seg
        rejections += 1
        h_try *= 0.5

    assert last_seg is not None
    last_seg.step_rejections = rejections
    last_seg.next_h = h_try
    last_seg.message = last_seg.message or "target remainder validation failed before h_min"
    return last_seg


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
    initial-state variables.  ``mode='flowstar_style'`` recenters each endpoint
    box and restarts from fresh normalized variables in ``[-1, 1]``.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if mode not in {"range_only", "dependency_preserving", "flowstar_style"}:
        raise ValueError("mode must be 'range_only', 'dependency_preserving', or 'flowstar_style'")

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

    if mode == "flowstar_style":
        current_tm = _normalized_tm_from_box(_as_interval_list(x0_box), order)
        for segment_index in range(steps):
            step_kwargs = _step_diagnostics_kwargs(kwargs, mode, segment_index)
            seg = flowpipe_step_from_tm(ode_fn, current_tm, h, order, u_box=u_box, affine_u=affine_u, **step_kwargs)
            segments.append(seg)
            if seg.status != "validated" or not intervals_are_finite(seg.final_tm.range_box()):
                break
            seg.reset_tm = _normalized_tm_from_box(seg.final_tm.range_box(), order)
            current_tm = seg.reset_tm
        status = "validated" if len(segments) == steps and all(s.status == "validated" for s in segments) else "failed"
        return FlowpipeResult(segments, status, current_tm, mode)

    current_tm = TMVector.identity(_as_interval_list(x0_box), order=order)
    for segment_index in range(steps):
        step_kwargs = _step_diagnostics_kwargs(kwargs, mode, segment_index)
        seg = flowpipe_step_from_tm(ode_fn, current_tm, h, order, u_box=u_box, affine_u=affine_u, **step_kwargs)
        segments.append(seg)
        current_tm = seg.final_tm
    status = "validated" if all(s.status == "validated" for s in segments) else "failed"
    return FlowpipeResult(segments, status, current_tm, mode)
