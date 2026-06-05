"""Fixed-step Taylor-model flowpipe construction for polynomial ODE prototypes."""
from __future__ import annotations

import hashlib
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
    selective_term_stats: Mapping[str, Any] | None = None
    selective_term_details: Sequence[Mapping[str, Any]] | None = None


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


def _zero_remainder_tm(
    poly: Polynomial,
    domain: Sequence[Interval],
    order: int,
    *,
    truncation_range_split: int | None = None,
) -> TaylorModel:
    return TaylorModel(
        poly,
        _zero_interval_like_domain(domain),
        list(domain),
        order=order,
        truncation_range_split=truncation_range_split,
    )


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


def _truncation_split_value(value: int | None) -> int | None:
    if value is None:
        return None
    pieces = int(value)
    return pieces if pieces > 1 else None


def _poly_interval_with_split(poly: Polynomial, domain: Sequence[Interval], split: int | None) -> Interval:
    pieces = _truncation_split_value(split)
    if pieces is None:
        return poly.evaluate_interval(domain)
    return poly.evaluate_interval_split(domain, pieces)


def _normal_domain(domain: Sequence[Interval], tau_index: int | None = None) -> list[Interval]:
    normal: list[Interval] = []
    for i, iv in enumerate(domain):
        if tau_index is not None and i == tau_index:
            normal.append(iv)
        else:
            normal.append(_unit_interval_like(iv))
    return normal


def _poly_interval_normal(poly: Polynomial, domain: Sequence[Interval], tau_index: int | None = None) -> Interval:
    """Flow*-style normal interval evaluation for normalized local domains."""
    return poly.evaluate_interval(_normal_domain(domain, tau_index))


def _cutoff_polynomial_normal(
    poly: Polynomial,
    domain: Sequence[Interval],
    tau_index: int | None,
    threshold: float | None,
) -> tuple[Polynomial, Interval]:
    if threshold is None:
        return poly, Interval.zero(dtype=poly.dtype, device=poly.device)
    kept: dict[tuple[int, ...], Any] = {}
    removed: dict[tuple[int, ...], Any] = {}
    threshold_t = torch.as_tensor(abs(float(threshold)), dtype=poly.dtype, device=poly.device)
    for exp, coef in poly.terms.items():
        target = removed if bool(torch.all(torch.abs(coef) <= threshold_t)) else kept
        target[exp] = coef
    removed_poly = Polynomial(removed, poly.n_vars)
    removed_range = _poly_interval_normal(removed_poly, domain, tau_index) if removed else Interval.zero(dtype=poly.dtype, device=poly.device)
    return Polynomial(kept, poly.n_vars), removed_range


def _term_interval(exp: tuple[int, ...], coef: Any, domain: Sequence[Interval]) -> Interval:
    term_iv = Interval.point(coef)
    for power, dom in zip(exp, domain):
        if power:
            term_iv = term_iv * dom.pow_int(power)
    return term_iv


def _interval_abs_extent(iv: Interval) -> float:
    return max(abs(float(iv.lo.detach().cpu())), abs(float(iv.hi.detach().cpu())))


def _monomial_label(exp: tuple[int, ...]) -> str:
    names = ["x", "y", "tau"]
    parts: list[str] = []
    for i, power in enumerate(exp):
        if power == 0:
            continue
        name = names[i] if i < len(names) else f"z{i}"
        parts.append(name if power == 1 else f"{name}^{power}")
    return "1" if not parts else "*".join(parts)


def _float_value(value: Any) -> float | str:
    out = _float_or_none(value)
    return out if out is not None else ""


def _tm_terms_signature(tm: TMVector) -> str:
    parts: list[str] = []
    for state_index, model in enumerate(tm):
        for exp, coef in sorted(model.polynomial.terms.items()):
            coef_f = _float_value(coef.detach().cpu() if hasattr(coef, "detach") else coef)
            parts.append(f"{state_index}:{','.join(str(e) for e in exp)}:{coef_f}")
    return "|".join(parts)


def _tm_terms_hash(tm: TMVector) -> str:
    return hashlib.sha256(_tm_terms_signature(tm).encode("utf-8")).hexdigest()[:16]


def _tm_high_degree_term_count(tm: TMVector, output_order: int) -> int:
    return sum(1 for model in tm for exp in model.polynomial.terms if sum(exp) > int(output_order))


def _tm_max_degree(tm: TMVector) -> int:
    return max((sum(exp) for model in tm for exp in model.polynomial.terms), default=0)


def _add_term_hash_metrics(row: dict[str, Any], prefix: str, tm: TMVector, output_order: int | None) -> None:
    row[f"{prefix}_terms_hash"] = _tm_terms_hash(tm)
    row[f"{prefix}_term_count"] = sum(len(model.polynomial.terms) for model in tm)
    row[f"{prefix}_max_degree"] = _tm_max_degree(tm)
    if output_order is not None:
        row[f"{prefix}_high_degree_term_count"] = _tm_high_degree_term_count(tm, int(output_order))


def _truncate_tm_to_order(tm: TMVector, output_order: int) -> TMVector:
    truncated, _stats, _details = _truncate_tm_to_order_selective(tm, output_order, selective_top_k=None)
    return truncated


def _truncate_tm_to_order_selective(
    tm: TMVector,
    output_order: int,
    *,
    selective_top_k: int | None = None,
    result_order: int | None = None,
) -> tuple[TMVector, list[dict[str, Any]], list[dict[str, Any]]]:
    models: list[TaylorModel] = []
    stats: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    top_k = int(selective_top_k or 0)
    state_names = ("x", "y")
    for state_index, model in enumerate(tm):
        kept, dropped = model.polynomial.truncate(int(output_order))
        dropped_terms = list(dropped.terms.items())
        retained_terms: dict[tuple[int, ...], Any] = {}
        nonkept_terms: dict[tuple[int, ...], Any] = dict(dropped.terms)
        ranked: list[tuple[float, tuple[int, ...], Any, Interval]] = []
        if top_k > 0 and dropped_terms:
            for exp, coef in dropped_terms:
                term_iv = _term_interval(tuple(exp), coef, model.domain)
                ranked.append((_interval_abs_extent(term_iv), tuple(exp), coef, term_iv))
            ranked.sort(key=lambda item: item[0], reverse=True)
            for rank, (abs_extent, exp, coef, term_iv) in enumerate(ranked, start=1):
                retained = rank <= top_k
                if retained:
                    retained_terms[exp] = coef
                    nonkept_terms.pop(exp, None)
                details.append(
                    {
                        "state_index": state_index,
                        "state_dimension": state_names[state_index] if state_index < len(state_names) else f"state_{state_index}",
                        "term_rank": rank,
                        "retained": retained,
                        "monomial": _monomial_label(exp),
                        "coefficient": _float_value(coef.detach().cpu() if hasattr(coef, "detach") else coef),
                        "total_degree": sum(exp),
                        "abs_interval_contribution": abs_extent,
                        "term_interval_lo": _float_value(term_iv.lo.detach().cpu()),
                        "term_interval_hi": _float_value(term_iv.hi.detach().cpu()),
                        "term_interval_width": _float_value(term_iv.width().detach().cpu()),
                    }
                )
        sparse_poly = kept + Polynomial(retained_terms, kept.n_vars) if retained_terms else kept
        nonkept = Polynomial(nonkept_terms, kept.n_vars)
        dropped_range = _poly_interval_with_split(nonkept, model.domain, model.truncation_range_split)
        total_dropped_range = _poly_interval_with_split(dropped, model.domain, model.truncation_range_split)
        models.append(
            TaylorModel(
                sparse_poly,
                model.remainder + dropped_range,
                list(model.domain),
                order=int(result_order if result_order is not None else output_order),
                truncation_range_split=model.truncation_range_split,
            )
        )
        stats.append(
            {
                "state_index": state_index,
                "state_dimension": state_names[state_index] if state_index < len(state_names) else f"state_{state_index}",
                "selective_high_degree_terms_top_k": top_k if top_k > 0 else "",
                "selective_retained_terms_count": len(retained_terms),
                "selective_dropped_terms_count": len(dropped_terms),
                "selective_nonretained_terms_count": len(nonkept_terms),
                "selective_dropped_remainder_lo": _float_value(dropped_range.lo.detach().cpu()),
                "selective_dropped_remainder_hi": _float_value(dropped_range.hi.detach().cpu()),
                "selective_dropped_remainder_width": _float_value(dropped_range.width().detach().cpu()),
                "selective_total_dropped_width": _float_value(total_dropped_range.width().detach().cpu()),
            }
        )
    return TMVector(models), stats, details


def _aggregate_selective_stats(
    stats: Sequence[Mapping[str, Any]],
    *,
    top_k: int | None,
) -> dict[str, Any]:
    if not top_k:
        return {}
    retained = sum(int(row.get("selective_retained_terms_count") or 0) for row in stats)
    dropped = sum(int(row.get("selective_dropped_terms_count") or 0) for row in stats)
    nonretained = sum(int(row.get("selective_nonretained_terms_count") or 0) for row in stats)
    rem_width = 0.0
    total_width = 0.0
    for row in stats:
        rem_width += _float_or_none(row.get("selective_dropped_remainder_width")) or 0.0
        total_width += _float_or_none(row.get("selective_total_dropped_width")) or 0.0
    return {
        "selective_high_degree_terms_top_k": int(top_k),
        "selective_retained_terms_count": retained,
        "selective_dropped_terms_count": dropped,
        "selective_nonretained_terms_count": nonretained,
        "selective_dropped_remainder_width_sum": rem_width,
        "selective_total_dropped_width_sum": total_width,
    }


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
        for key, value in extra.items():
            if key not in row:
                row[key] = value
    if candidate is not None:
        output_order_value = _float_or_none(row.get("output_order"))
        _add_term_hash_metrics(
            row,
            "validation_candidate_inside",
            candidate,
            int(output_order_value) if output_order_value is not None else None,
        )
        _add_term_hash_metrics(
            row,
            "validation_candidate_after_internal",
            candidate,
            int(output_order_value) if output_order_value is not None else None,
        )
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


def _constant_control_tms(
    u_box: Sequence[Any] | None,
    domain: Sequence[Interval],
    order: int,
    *,
    truncation_range_split: int | None = None,
) -> TMVector | None:
    if u_box is None:
        return None
    controls: list[TaylorModel] = []
    for u in u_box:
        iv = ensure_interval(u) if not isinstance(u, (tuple, list)) else Interval(u[0], u[1])
        controls.append(
            TaylorModel.constant(
                iv.mid(),
                domain,
                order=order,
                remainder=Interval(-iv.radius(), iv.radius()),
                truncation_range_split=truncation_range_split,
            )
        )
    return TMVector(controls)


def _affine_control_tms(
    affine_u: dict[str, Any] | None,
    domain: Sequence[Interval],
    order: int,
    *,
    truncation_range_split: int | None = None,
) -> TMVector | None:
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
    variables = [
        TaylorModel.variable(i, domain, order=order, truncation_range_split=truncation_range_split)
        for i in range(n_x)
    ]
    controls: list[TaylorModel] = []
    for j in range(n_u):
        tm = TaylorModel.constant(b[j], domain, order=order, truncation_range_split=truncation_range_split)
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
    *,
    truncation_range_split: int | None = None,
) -> TMVector | None:
    u_const = _constant_control_tms(u_box, domain, order, truncation_range_split=truncation_range_split)
    u_affine = _affine_control_tms(affine_u, domain, order, truncation_range_split=truncation_range_split)
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
    truncation_range_split: int | None = None,
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
            next_tm = _zero_remainder_tm(
                poly,
                domain,
                order,
                truncation_range_split=truncation_range_split,
            ).apply_cutoff(cutoff_threshold)
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
        candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(candidate_poly, remainders)
        )
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
                residual_i = picard_i - TaylorModel(
                    cand_i.polynomial,
                    Interval.zero(),
                    domain,
                    order=order,
                    truncation_range_split=cand_i.truncation_range_split,
                )
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

    candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(candidate_poly, remainders)
        )
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
    candidate = TMVector(
        TaylorModel(
            m.polynomial,
            r,
            domain,
            order=order,
            truncation_range_split=m.truncation_range_split,
        )
        for m, r in zip(candidate_poly, target_remainders)
    )
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
                residual_i = picard_i - TaylorModel(
                    cand_i.polynomial,
                    Interval.zero(),
                    domain,
                    order=order,
                    truncation_range_split=cand_i.truncation_range_split,
                )
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



def _residual_interval_stats(prefix: str, boxes: Sequence[Interval] | None) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if boxes is None:
        return row
    names = ("x", "y")
    for i, iv in enumerate(boxes[:2]):
        name = names[i]
        lo = _interval_bound_value(iv.lo)
        hi = _interval_bound_value(iv.hi)
        width = _interval_width_value(iv)
        center = _float_or_none(iv.mid().detach().cpu())
        radius = _float_or_none(iv.radius().detach().cpu())
        row[f"{prefix}_lo_{name}"] = lo if lo is not None else ""
        row[f"{prefix}_hi_{name}"] = hi if hi is not None else ""
        row[f"{prefix}_width_{name}"] = width if width is not None else ""
        row[f"{prefix}_center_{name}"] = center if center is not None else ""
        row[f"{prefix}_radius_{name}"] = radius if radius is not None else ""
    return row


def _picard_residual_boxes(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    validation_eps: float,
) -> list[Interval]:
    domain = candidate.domain
    rhs = _call_ode(ode_fn, candidate, u_tms)
    residual_boxes: list[Interval] = []
    for base_i, cand_i, f_i in zip(base_ext, candidate, rhs):
        picard_i = base_i + f_i.integrate(tau_index)
        residual_i = picard_i - TaylorModel(
            cand_i.polynomial,
            Interval.zero(),
            domain,
            order=order,
            truncation_range_split=cand_i.truncation_range_split,
        )
        residual_boxes.append(residual_i.range_box().inflate(validation_eps))
    return residual_boxes


def _interval_list_stats(prefix: str, boxes: Sequence[Interval] | None) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if boxes is None:
        return row
    names = ("x", "y")
    widths: list[float] = []
    for i, iv in enumerate(boxes[:2]):
        name = names[i] if i < len(names) else f"state_{i}"
        lo = _interval_bound_value(iv.lo)
        hi = _interval_bound_value(iv.hi)
        width = _interval_width_value(iv)
        center = _float_or_none(iv.mid().detach().cpu())
        radius = _float_or_none(iv.radius().detach().cpu())
        row[f"{prefix}_lo_{name}"] = lo if lo is not None else ""
        row[f"{prefix}_hi_{name}"] = hi if hi is not None else ""
        row[f"{prefix}_width_{name}"] = width if width is not None else ""
        row[f"{prefix}_center_{name}"] = center if center is not None else ""
        row[f"{prefix}_radius_{name}"] = radius if radius is not None else ""
        if width is not None:
            widths.append(width)
    if widths:
        row[f"{prefix}_width_sum"] = sum(widths)
    return row


def _picard_ctrunc_normal_image(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    cutoff_threshold: float | None,
) -> TMVector:
    domain = candidate.domain
    rhs = _call_ode(ode_fn, candidate, u_tms)
    models: list[TaylorModel] = []
    for base_i, f_i in zip(base_ext, rhs):
        picard_i = base_i + f_i.integrate(tau_index)
        kept, dropped = picard_i.polynomial.truncate(order)
        trunc_range = _poly_interval_normal(dropped, domain, tau_index)
        kept, cutoff_range = _cutoff_polynomial_normal(kept, domain, tau_index, cutoff_threshold)
        models.append(
            TaylorModel(
                kept,
                picard_i.remainder + trunc_range + cutoff_range,
                domain,
                order=order,
                truncation_range_split=picard_i.truncation_range_split,
            )
        )
    return TMVector(models)


def _validate_picard_target_remainder_flowstar_ctrunc(
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
    cutoff_threshold: float | None,
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
    diag_extra.setdefault("validation_mode", "target_remainder_flowstar_ctrunc")
    diag_extra.setdefault("target_remainder_radius", abs(float(target_remainder_radius)))
    diag_extra.setdefault("target_remainder_width", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_remainder_width_sum", _sum_interval_widths(target_remainders))
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)

    seed_remainders = [_combine_remainders(base_i.remainder, candidate_i.remainder) for base_i, candidate_i in zip(base_ext, candidate_poly)]
    candidate = TMVector(
        TaylorModel(
            m.polynomial,
            r,
            domain,
            order=order,
            truncation_range_split=m.truncation_range_split,
        )
        for m, r in zip(candidate_poly, target_remainders)
    )
    if not intervals_are_finite(seed_remainders):
        message = "non-finite initial remainder"
        extra = dict(diag_extra, subset_result=False, subset_tmp_remainder=False, subset_ordinary_residual=False, rejection_reason=message)
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
        extra = dict(diag_extra, subset_result=False, subset_tmp_remainder=False, subset_ordinary_residual=False, rejection_reason=message)
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
            ordinary_residual = _picard_residual_boxes(
                ode_fn,
                base_ext,
                candidate,
                tau_index,
                order,
                u_tms,
                validation_eps=validation_eps,
            )
            tmp = _picard_ctrunc_normal_image(
                ode_fn,
                base_ext,
                candidate,
                tau_index,
                order,
                u_tms,
                cutoff_threshold=cutoff_threshold,
            )
            poly_diff_ranges: list[Interval] = []
            tmp_remainders: list[Interval] = []
            for tmp_i, cand_i in zip(tmp, candidate):
                diff_poly = tmp_i.polynomial - cand_i.polynomial
                diff_range = _poly_interval_normal(diff_poly, domain, tau_index).inflate(validation_eps)
                poly_diff_ranges.append(diff_range)
                tmp_remainders.append((tmp_i.remainder + diff_range).inflate(validation_eps))
        except Exception as exc:
            message = f"validation exception: {exc}"
            extra = dict(diag_extra, subset_result=False, subset_tmp_remainder=False, subset_ordinary_residual=False, rejection_reason=message)
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

        finite_residual = intervals_are_finite(ordinary_residual) and intervals_are_finite(tmp_remainders) and intervals_are_finite(poly_diff_ranges)
        subset_ordinary = bool(finite_residual and all(target.contains_interval(rb) for target, rb in zip(target_remainders, ordinary_residual)))
        subset_tmp = bool(finite_residual and all(target.contains_interval(rem) for target, rem in zip(target_remainders, tmp_remainders)))
        validation_decision_difference = bool(subset_tmp != subset_ordinary)
        message = "" if subset_tmp else "Flowstar ctrunc tmp remainder not subset of target remainder"
        validated_candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(candidate_poly, tmp_remainders)
        )
        extra = {
            **diag_extra,
            **_interval_list_stats("tmp_remainder", tmp_remainders),
            **_interval_list_stats("poly_diff_range", poly_diff_ranges),
            **_interval_list_stats("ordinary_residual_range", ordinary_residual),
            **_interval_list_stats("normal_eval_range", poly_diff_ranges),
            "subset_result": subset_tmp,
            "subset_tmp_remainder": subset_tmp,
            "subset_ordinary_residual": subset_ordinary,
            "validation_decision_difference": validation_decision_difference,
            "rejection_reason": "" if subset_tmp else message,
        }
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=attempt,
            h=h,
            order=order,
            candidate=validated_candidate,
            tau_index=tau_index,
            residual_boxes=ordinary_residual,
            remainders=tmp_remainders,
            finite_residual=finite_residual,
            validation_status="validated" if subset_tmp else "failed",
            validation_message=message,
            extra=extra,
        )
        if subset_tmp:
            return validated_candidate, "validated", attempt, ""
        return validated_candidate, "failed", attempt, message

    return candidate, "failed", max_attempts, "Flowstar ctrunc tmp remainder not subset of target remainder"


def _shift_candidate_constants(candidate_poly: TMVector, shifts: Sequence[torch.Tensor]) -> TMVector:
    shifted: list[TaylorModel] = []
    for model, shift in zip(candidate_poly, shifts):
        terms = {exp: coef.clone() for exp, coef in model.polynomial.terms.items()}
        zero_exp = (0,) * model.polynomial.n_vars
        shift_t = shift.to(dtype=model.polynomial.dtype, device=model.polynomial.device)
        terms[zero_exp] = terms.get(zero_exp, torch.zeros_like(shift_t)) + shift_t
        shifted.append(
            TaylorModel(
                Polynomial(terms, model.polynomial.n_vars),
                model.remainder,
                list(model.domain),
                order=model.order,
                truncation_range_split=model.truncation_range_split,
            )
        )
    return TMVector(shifted)


def _zero_shift_like(model: TaylorModel) -> torch.Tensor:
    if model.polynomial.terms:
        first = next(iter(model.polynomial.terms.values()))
        return torch.zeros((), dtype=first.dtype, device=first.device)
    if model.domain:
        return torch.zeros((), dtype=model.domain[0].dtype, device=model.domain[0].device)
    return torch.zeros((), dtype=torch.float64)


def _validate_picard_target_remainder_centered(
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
    center_correction_width_factor: float = 1.05,
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
    diag_extra.setdefault("validation_mode", "target_remainder_centered")
    diag_extra.setdefault("target_remainder_radius", abs(float(target_remainder_radius)))
    diag_extra.setdefault("target_remainder_width", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_remainder_width_sum", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("center_correction_width_factor", float(center_correction_width_factor))
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)

    seed_remainders = [_combine_remainders(base_i.remainder, candidate_i.remainder) for base_i, candidate_i in zip(base_ext, candidate_poly)]
    candidate = TMVector(
        TaylorModel(
            m.polynomial,
            r,
            domain,
            order=order,
            truncation_range_split=m.truncation_range_split,
        )
        for m, r in zip(candidate_poly, target_remainders)
    )
    if not intervals_are_finite(seed_remainders):
        message = "non-finite initial remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message, center_correction_applied=False, subset_after_correction=False)
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
        extra = dict(diag_extra, subset_result=False, rejection_reason=message, center_correction_applied=False, subset_after_correction=False)
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
            residual_before = _picard_residual_boxes(
                ode_fn,
                base_ext,
                candidate,
                tau_index,
                order,
                u_tms,
                validation_eps=validation_eps,
            )
        except Exception as exc:
            message = f"validation exception: {exc}"
            extra = dict(diag_extra, subset_result=False, rejection_reason=message, center_correction_applied=False, subset_after_correction=False)
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

        finite_residual = intervals_are_finite(residual_before)
        before_stats = _residual_interval_stats("residual_before", residual_before)
        if not finite_residual:
            message = "non-finite residual interval"
            extra = {**diag_extra, **before_stats, "subset_result": False, "rejection_reason": message, "center_correction_applied": False, "subset_after_correction": False}
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_before,
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        subset_before = all(target.contains_interval(rb) for target, rb in zip(target_remainders, residual_before))
        if subset_before:
            extra = {
                **diag_extra,
                **before_stats,
                **_residual_interval_stats("residual_after", residual_before),
                "subset_result": True,
                "rejection_reason": "",
                "center_correction_applied": False,
                "correction_value_x": 0.0,
                "correction_value_y": 0.0,
                "subset_after_correction": True,
            }
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_before,
                remainders=target_remainders,
                finite_residual=True,
                validation_status="validated",
                validation_message="",
                extra=extra,
            )
            return candidate, "validated", attempt, ""

        target_widths = [_interval_width_value(target) for target in target_remainders]
        residual_widths = [_interval_width_value(rb) for rb in residual_before]
        misses = [not target.contains_interval(rb) for target, rb in zip(target_remainders, residual_before)]
        shift_eligible = []
        for miss, rb_width, target_width in zip(misses, residual_widths, target_widths):
            shift_eligible.append(bool(miss and rb_width is not None and target_width is not None and rb_width <= target_width * float(center_correction_width_factor)))
        if any(miss and not eligible for miss, eligible in zip(misses, shift_eligible)):
            message = "Picard residual not subset of target remainder"
            extra = {
                **diag_extra,
                **before_stats,
                "subset_result": False,
                "rejection_reason": message,
                "center_correction_applied": False,
                "correction_value_x": 0.0,
                "correction_value_y": 0.0,
                "subset_after_correction": False,
            }
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_before,
                remainders=target_remainders,
                finite_residual=True,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        shifts = []
        for model, rb, eligible in zip(candidate_poly, residual_before, shift_eligible):
            shifts.append(rb.mid() if eligible else _zero_shift_like(model))
        corrected_poly = _shift_candidate_constants(candidate_poly, shifts)
        corrected_candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(corrected_poly, target_remainders)
        )
        try:
            residual_after = _picard_residual_boxes(
                ode_fn,
                base_ext,
                corrected_candidate,
                tau_index,
                order,
                u_tms,
                validation_eps=validation_eps,
            )
        except Exception as exc:
            message = f"validation exception after center correction: {exc}"
            extra = dict(
                diag_extra,
                before_stats,
                subset_result=False,
                rejection_reason=message,
                center_correction_applied=True,
                correction_value_x=_float_value(shifts[0].detach().cpu()) if len(shifts) > 0 else "",
                correction_value_y=_float_value(shifts[1].detach().cpu()) if len(shifts) > 1 else "",
                subset_after_correction=False,
            )
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=corrected_candidate,
                tau_index=tau_index,
                residual_boxes=None,
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return corrected_candidate, "failed", attempt, message

        finite_after = intervals_are_finite(residual_after)
        subset_after = bool(finite_after and all(target.contains_interval(rb) for target, rb in zip(target_remainders, residual_after)))
        message = "" if subset_after else "Picard residual not subset of target remainder after center correction"
        extra = {
            **diag_extra,
            **before_stats,
            **_residual_interval_stats("residual_after", residual_after),
            "subset_result": subset_after,
            "rejection_reason": "" if subset_after else message,
            "center_correction_applied": True,
            "correction_value_x": _float_value(shifts[0].detach().cpu()) if len(shifts) > 0 else "",
            "correction_value_y": _float_value(shifts[1].detach().cpu()) if len(shifts) > 1 else "",
            "subset_after_correction": subset_after,
        }
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=attempt,
            h=h,
            order=order,
            candidate=corrected_candidate,
            tau_index=tau_index,
            residual_boxes=residual_after,
            remainders=target_remainders,
            finite_residual=finite_after,
            validation_status="validated" if subset_after else "failed",
            validation_message=message,
            extra=extra,
        )
        if subset_after:
            return corrected_candidate, "validated", attempt, ""
        return corrected_candidate, "failed", attempt, message

    return candidate, "failed", max_attempts, "Picard residual not subset of target remainder"


def _validate_picard_target_remainder_refined(
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
    """Conservative target validation with remainder-only Picard refinement."""
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    target_remainders = [_symmetric_interval(target_remainder_radius, domain) for _ in candidate_poly]
    diag_extra = dict(diagnostics_context or {})
    diag_extra.setdefault("validation_mode", "target_remainder_refined")
    diag_extra.setdefault("target_remainder_radius", abs(float(target_remainder_radius)))
    diag_extra.setdefault("target_remainder_width", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_remainder_width_sum", _sum_interval_widths(target_remainders))
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)

    seed_remainders = [
        _combine_remainders(base_i.remainder, candidate_i.remainder).inflate(validation_eps)
        for base_i, candidate_i in zip(base_ext, candidate_poly)
    ]
    current_remainders = seed_remainders
    candidate = TMVector(
        TaylorModel(
            m.polynomial,
            r,
            domain,
            order=order,
            truncation_range_split=m.truncation_range_split,
        )
        for m, r in zip(candidate_poly, current_remainders)
    )
    if not intervals_are_finite(current_remainders):
        message = "non-finite initial remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message, refinement_pass=0)
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
            remainders=current_remainders,
            finite_residual=False,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message
    if not all(target.contains_interval(seed) for target, seed in zip(target_remainders, current_remainders)):
        message = "initial or cutoff remainder exceeds target remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message, refinement_pass=0)
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
            remainders=current_remainders,
            finite_residual=True,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message

    for attempt in range(1, max_attempts + 1):
        candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(candidate_poly, current_remainders)
        )
        if rhs_breakdown_callback is not None:
            callback_context = dict(diag_extra)
            if diag_mode is not None:
                callback_context["mode"] = diag_mode
            if diag_segment_index is not None:
                callback_context["segment_index"] = diag_segment_index
            callback_context["attempt_index"] = attempt
            callback_context["h"] = float(h)
            callback_context["order"] = int(order)
            callback_context["refinement_pass"] = attempt
            try:
                rhs_breakdown_callback(candidate, order, attempt, callback_context)
            except Exception:
                pass
        try:
            rhs = _call_ode(ode_fn, candidate, u_tms)
            residual_boxes: list[Interval] = []
            for base_i, cand_i, f_i in zip(base_ext, candidate, rhs):
                picard_i = base_i + f_i.integrate(tau_index)
                residual_i = picard_i - TaylorModel(
                    cand_i.polynomial,
                    Interval.zero(),
                    domain,
                    order=order,
                    truncation_range_split=cand_i.truncation_range_split,
                )
                residual_boxes.append(residual_i.range_box().inflate(validation_eps))
        except Exception as exc:
            message = f"validation exception: {exc}"
            extra = dict(diag_extra, subset_result=False, rejection_reason=message, refinement_pass=attempt)
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
                remainders=current_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        finite_residual = intervals_are_finite(residual_boxes)
        residual_inside_target = all(target.contains_interval(rb) for target, rb in zip(target_remainders, residual_boxes))
        residual_inside_current = all(current.contains_interval(rb) for current, rb in zip(current_remainders, residual_boxes))
        if not finite_residual:
            message = "non-finite residual interval"
        elif residual_inside_current:
            message = ""
        elif not residual_inside_target:
            message = "Picard residual not subset of target remainder"
        else:
            message = "remainder-only refinement continuing"
        extra = dict(
            diag_extra,
            subset_result=bool(finite_residual and residual_inside_target),
            rejection_reason="" if finite_residual and residual_inside_current else message,
            refinement_pass=attempt,
            residual_subset_current=bool(finite_residual and residual_inside_current),
        )
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
            remainders=current_remainders,
            finite_residual=finite_residual,
            validation_status="validated" if finite_residual and residual_inside_current else "failed",
            validation_message=message,
            extra=extra,
        )
        if not finite_residual:
            return candidate, "failed", attempt, message
        if residual_inside_current:
            return candidate, "validated", attempt, ""
        if not residual_inside_target:
            return candidate, "failed", attempt, message

        next_remainders = [Interval.hull(seed, residual).inflate(validation_eps) for seed, residual in zip(seed_remainders, residual_boxes)]
        if not all(target.contains_interval(next_r) for target, next_r in zip(target_remainders, next_remainders)):
            message = "refined remainder exceeds target remainder"
            return candidate, "failed", attempt, message
        current_remainders = next_remainders

    return candidate, "failed", max_attempts, "remainder-only target refinement did not converge"


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
    center_correction_width_factor: float = 1.05,
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
    candidate_order: int | None = None,
    truncation_range_split: int | None = None,
    selective_high_degree_terms_top_k: int | None = None,
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
    output_order = int(order)
    candidate_order_i = int(candidate_order) if candidate_order is not None else output_order
    if candidate_order_i < output_order:
        raise ValueError("candidate_order must be >= output order")
    split = _truncation_split_value(truncation_range_split)
    diag_context = dict(diagnostics_context or {})
    diag_context.setdefault("output_order", output_order)
    diag_context.setdefault("candidate_order", candidate_order_i)
    diag_context.setdefault("truncation_range_split", split or "")
    selective_top_k = int(selective_high_degree_terms_top_k or 0)
    if selective_top_k > 0:
        diag_context.setdefault("selective_high_degree_terms_top_k", selective_top_k)
    tau_interval = Interval(0.0, float(h))
    base_ext = x0_tm.extend_domain(tau_interval)
    tau_index = x0_tm.n_vars
    domain = base_ext.domain
    base_poly_ext = TMVector(
        TaylorModel(
            m.polynomial,
            Interval.zero(),
            domain,
            order=candidate_order_i,
            truncation_range_split=split,
        )
        for m in base_ext
    )
    if validation_mode not in {
        "growth",
        "current",
        "target_remainder",
        "target_remainder_refined",
        "target_remainder_centered",
        "target_remainder_flowstar_ctrunc",
    }:
        raise ValueError(
            "validation_mode must be 'growth', 'current', 'target_remainder', 'target_remainder_refined', "
            "'target_remainder_centered', or 'target_remainder_flowstar_ctrunc'"
        )
    target_mode = validation_mode in {
        "target_remainder",
        "target_remainder_refined",
        "target_remainder_centered",
        "target_remainder_flowstar_ctrunc",
    }
    attempt_limit = (2 if target_mode else 20) if max_validation_attempts is None else int(max_validation_attempts)
    if attempt_limit <= 0:
        raise ValueError("max_validation_attempts must be positive")

    u_tms = _make_controls(
        u_box,
        affine_u,
        domain,
        candidate_order_i,
        truncation_range_split=split,
    )
    candidate_poly = _picard_polynomial(
        ode_fn,
        base_poly_ext,
        tau_index,
        candidate_order_i,
        u_tms,
        cutoff_threshold=cutoff_threshold,
        truncation_range_split=split,
    )
    _add_term_hash_metrics(diag_context, "candidate_terms_before_validation", candidate_poly, output_order)
    validation_candidate_poly = candidate_poly
    validation_selective_stats: list[dict[str, Any]] = []
    validation_selective_details: list[dict[str, Any]] = []
    if selective_top_k > 0:
        validation_candidate_poly, validation_selective_stats, validation_selective_details = _truncate_tm_to_order_selective(
            candidate_poly,
            output_order,
            selective_top_k=selective_top_k,
            result_order=candidate_order_i,
        )
    _add_term_hash_metrics(diag_context, "candidate_terms_after_selective", validation_candidate_poly, output_order)
    if validation_mode == "target_remainder":
        validated, status, attempts, message = _validate_picard_target_remainder(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            target_remainder_radius=target_remainder_radius,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    elif validation_mode == "target_remainder_centered":
        validated, status, attempts, message = _validate_picard_target_remainder_centered(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            target_remainder_radius=target_remainder_radius,
            center_correction_width_factor=center_correction_width_factor,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    elif validation_mode == "target_remainder_refined":
        validated, status, attempts, message = _validate_picard_target_remainder_refined(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            target_remainder_radius=target_remainder_radius,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    elif validation_mode == "target_remainder_flowstar_ctrunc":
        validated, status, attempts, message = _validate_picard_target_remainder_flowstar_ctrunc(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            target_remainder_radius=target_remainder_radius,
            cutoff_threshold=cutoff_threshold,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    else:
        validated, status, attempts, message = _validate_picard(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            growth_factor=growth_factor,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    final_tm = validated.substitute_const(tau_index, float(h)).drop_variable(tau_index)
    final_tm = final_tm.apply_cutoff(cutoff_threshold)
    if status == "validated" and validation_mode != "target_remainder_flowstar_ctrunc":
        # The segment remainder is valid for every tau in [0,h].  For multi-step
        # propagation we only need the endpoint at tau=h, so tighten the endpoint
        # remainder by re-evaluating the Picard residual at that fixed local time.
        try:
            rhs = _call_ode(ode_fn, validated, u_tms)
            final_models = []
            for base_i, cand_i, f_i in zip(base_ext, validated, rhs):
                picard_i = base_i + f_i.integrate(tau_index)
                residual_i = picard_i - TaylorModel(
                    cand_i.polynomial,
                    Interval.zero(),
                    domain,
                    order=candidate_order_i,
                    truncation_range_split=cand_i.truncation_range_split,
                )
                endpoint_residual = (
                    residual_i.substitute_const(tau_index, float(h))
                    .drop_variable(tau_index)
                    .range_box()
                    .inflate(validation_eps)
                )
                endpoint_poly = cand_i.polynomial.substitute_const(tau_index, float(h)).drop_variable(tau_index)
                endpoint_domain = [d for i, d in enumerate(domain) if i != tau_index]
                final_models.append(TaylorModel(
                    endpoint_poly,
                    endpoint_residual,
                    endpoint_domain,
                    order=candidate_order_i,
                    truncation_range_split=cand_i.truncation_range_split,
                ))
            final_tm = TMVector(final_models).apply_cutoff(cutoff_threshold)
        except Exception as exc:
            message = message or f"endpoint tightening skipped: {exc}"
    selective_stats: dict[str, Any] = {}
    selective_details: list[dict[str, Any]] = []
    if selective_top_k > 0:
        final_tm, final_selective_stats, final_selective_details = _truncate_tm_to_order_selective(
            final_tm,
            output_order,
            selective_top_k=selective_top_k,
        )
        output_tm, output_selective_stats, output_selective_details = _truncate_tm_to_order_selective(
            validated,
            output_order,
            selective_top_k=selective_top_k,
        )
        selective_stats = _aggregate_selective_stats(validation_selective_stats or output_selective_stats, top_k=selective_top_k)
        selective_details = validation_selective_details or output_selective_details
        final_tm = final_tm.apply_cutoff(cutoff_threshold)
    else:
        final_tm = _truncate_tm_to_order(final_tm, output_order).apply_cutoff(cutoff_threshold)
        output_tm = _truncate_tm_to_order(validated, output_order)

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
        tm=output_tm,
        final_tm=final_tm,
        status=status,
        h=float(h),
        order=output_order,
        validation_attempts=attempts,
        message=message,
        tau_index=tau_index,
        symbolic_remainder=bool(symbolic_remainder),
        symbolic_remainder_state=next_symbolic_state,
        symbolic_remainder_stats=symbolic_stats,
        selective_term_stats=selective_stats or None,
        selective_term_details=selective_details or None,
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


def _is_target_containment_failure(message: str) -> bool:
    return "target remainder" in message or "not subset" in message or "target containment" in message


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
    center_correction_width_factor: float = 1.05,
    cutoff_threshold: float | None = 1e-10,
    max_validation_attempts: int = 2,
    validation_eps: float = 1e-12,
    validation_mode: str = "target_remainder",
    adaptive_order_fallback: int | None = None,
    adaptive_order_threshold_factor: float = 1.25,
    grow_factor: float = 1.5,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    candidate_order: int | None = None,
    truncation_range_split: int | None = None,
    selective_high_degree_terms_top_k: int | None = None,
) -> FlowpipeSegment:
    if h_min <= 0 or h_max <= 0:
        raise ValueError("h_min and h_max must be positive")
    if h_min > h_max:
        raise ValueError("h_min must be <= h_max")
    if validation_mode not in {
        "target_remainder",
        "target_remainder_refined",
        "target_remainder_centered",
        "target_remainder_flowstar_ctrunc",
    }:
        raise ValueError("flowstar_style adaptive validation must use a target remainder mode")
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
            validation_mode=validation_mode,
            target_remainder_radius=target_remainder_radius,
            center_correction_width_factor=center_correction_width_factor,
            cutoff_threshold=cutoff_threshold,
            diagnostics=diagnostics,
            diagnostics_context=context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            candidate_order=candidate_order,
            truncation_range_split=truncation_range_split,
            selective_high_degree_terms_top_k=selective_high_degree_terms_top_k,
        )
        seg.step_rejections = rejections
        if seg.status == "validated" and intervals_are_finite(seg.final_tm.range_box()):
            seg.reset_tm = _normalized_tm_from_box(seg.final_tm.range_box(), order)
            seg.next_h = min(h_try * grow_factor, h_max)
            return seg

        last_seg = seg
        fallback_order = int(adaptive_order_fallback or 0)
        near_min_failure = (
            h_try <= float(adaptive_order_threshold_factor) * float(h_min) + 1e-15
            or h_try * 0.5 < float(h_min) - 1e-15
        )
        should_retry_order = (
            fallback_order > int(order)
            and int(order) == 6
            and near_min_failure
            and _is_target_containment_failure(seg.message)
        )
        if should_retry_order:
            fallback_context = dict(context)
            fallback_context["adaptive_order_fallback"] = True
            fallback_context["fallback_from_order"] = int(order)
            fallback_context["h_try"] = h_try
            fallback_seg = flowpipe_step_from_tm(
                ode_fn,
                current_tm,
                h_try,
                fallback_order,
                u_box=u_box,
                affine_u=affine_u,
                max_validation_attempts=max_validation_attempts,
                validation_eps=validation_eps,
                validation_mode=validation_mode,
                target_remainder_radius=target_remainder_radius,
                center_correction_width_factor=center_correction_width_factor,
                cutoff_threshold=cutoff_threshold,
                diagnostics=diagnostics,
                diagnostics_context=fallback_context,
                rhs_breakdown_callback=rhs_breakdown_callback,
                candidate_order=None if candidate_order is None else max(int(candidate_order), fallback_order),
                truncation_range_split=truncation_range_split,
                selective_high_degree_terms_top_k=selective_high_degree_terms_top_k,
            )
            fallback_seg.step_rejections = rejections
            last_seg = fallback_seg
            if fallback_seg.status == "validated" and intervals_are_finite(fallback_seg.final_tm.range_box()):
                fallback_seg.reset_tm = _normalized_tm_from_box(fallback_seg.final_tm.range_box(), order)
                fallback_seg.next_h = min(h_try * grow_factor, h_max)
                return fallback_seg

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
