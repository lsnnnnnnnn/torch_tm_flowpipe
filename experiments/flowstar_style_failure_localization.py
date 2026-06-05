#!/usr/bin/env python3
"""Localize the Flowstar-style Van der Pol target-remainder failure."""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe import (  # noqa: E402
    Interval,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    taylor_model_mul_breakdown,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode  # noqa: E402
from torch_tm_flowpipe.safety import intervals_are_finite  # noqa: E402

import flowstar_style_rescue_vanderpol as rescue  # noqa: E402

RUN_ID = "flowstar_style_o6_target"
ORDER = 6
TARGET_REMAINDER_RADIUS = 1e-4
H_MIN = 0.002
H_MAX = 0.1

ATTEMPT_FIELDS = [
    "attempt_global_index",
    "t_start",
    "h_try",
    "t_end",
    "order",
    "accepted",
    "rejection_reason",
    "residual_width_x",
    "residual_width_y",
    "residual_width_sum",
    "target_width_x",
    "target_width_y",
    "target_width_sum",
    "residual_over_target_ratio_x",
    "residual_over_target_ratio_y",
    "residual_over_target_ratio_sum",
    "final_width_x",
    "final_width_y",
    "final_width_sum",
    "polynomial_range_width_x",
    "polynomial_range_width_y",
    "remainder_width_x",
    "remainder_width_y",
    "cutoff_uncertainty_width_x",
    "cutoff_uncertainty_width_y",
    "truncation_uncertainty_width_x",
    "truncation_uncertainty_width_y",
    "notes",
]

BREAKDOWN_FIELDS = [
    "t_start",
    "h_try",
    "expression",
    "kept_poly_range_width",
    "dropped_trunc_width",
    "polynomial_range_times_remainder_width",
    "remainder_times_remainder_width",
    "cutoff_uncertainty_width",
    "total_remainder_width",
    "finite",
]

SUMMARY_FIELDS = [
    "run_id",
    "status",
    "last_validated_t",
    "last_attempted_t",
    "num_accepted_steps",
    "num_rejected_attempts",
    "failure_reason",
    "failed_dimension_first",
    "failure_h_try",
    "residual_over_target_ratio_x",
    "residual_over_target_ratio_y",
    "residual_over_target_ratio_sum",
    "dominant_breakdown_component",
    "would_h_below_0.002_likely_help",
    "next_recommendation",
]


TRUNCATION_DETAIL_FIELDS = [
    "run_id",
    "segment_index",
    "attempt_index",
    "t_start",
    "h_try",
    "failed_state_dimension",
    "source_expression",
    "dropped_total_degree",
    "term_rank",
    "monomial",
    "coefficient",
    "abs_interval_contribution",
    "dropped_range_interval_lo",
    "dropped_range_interval_hi",
    "dropped_range_width",
    "contribution_to_residual_lo",
    "contribution_to_residual_hi",
    "contribution_shift",
    "residual_center",
    "residual_radius",
    "target_interval_lo",
    "target_interval_hi",
    "minimal_symmetric_radius_needed",
    "minimal_asymmetric_interval_lo",
    "minimal_asymmetric_interval_hi",
]

TRUNCATION_SUMMARY_FIELDS = [
    "run_id",
    "status",
    "last_validated_t",
    "failure_t_start",
    "failure_h_try",
    "failed_state_dimension",
    "dominant_source_expression",
    "dominance_shape",
    "width_or_shift",
    "minimal_symmetric_radius_needed",
    "minimal_asymmetric_interval_lo",
    "minimal_asymmetric_interval_hi",
    "tighter_range_bound_likely_fix",
    "notes",
]

RESIDUAL_SHIFT_FIELDS = [
    "run_id",
    "t_start",
    "h_try",
    "failed_dimension",
    "residual_lo_y",
    "residual_hi_y",
    "target_radius",
    "target_lo",
    "target_hi",
    "minimal_symmetric_radius_needed",
    "minimal_asymmetric_interval_lo",
    "minimal_asymmetric_interval_hi",
    "asymmetric_width",
    "symmetric_width",
    "asymmetric_much_tighter_than_symmetric",
    "accepted_semantics_changed",
]


class StepTimeout(RuntimeError):
    pass


def _call_with_timeout(fn: Any, timeout_s: float) -> Any:
    if timeout_s <= 0:
        raise StepTimeout("wall-time cap reached before validation call")
    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(_signum: int, _frame: Any) -> None:
        raise StepTimeout("wall-time cap reached during validation call")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(float(timeout_s), 1e-6))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _finite(value: Any) -> float | None:
    return rescue._finite_float(value)


def _safe_ratio(num: Any, den: Any) -> float | str:
    n = _finite(num)
    d = _finite(den)
    if n is None or d is None or d <= 0:
        return ""
    return n / d


def _width(value: Any) -> float | str:
    try:
        w = float(value.width().detach().cpu())
    except Exception:
        return ""
    return w if math.isfinite(w) else ""



def _interval_lo_hi(iv: Interval) -> tuple[float | str, float | str]:
    try:
        return float(iv.lo.detach().cpu()), float(iv.hi.detach().cpu())
    except Exception:
        return "", ""


def _interval_width_value(iv: Interval) -> float | str:
    try:
        value = float(iv.width().detach().cpu())
    except Exception:
        return ""
    return value if math.isfinite(value) else ""


def _term_interval(exp: tuple[int, ...], coef: Any, domain: Sequence[Interval]) -> Interval:
    term_iv = Interval.point(coef)
    for power, dom in zip(exp, domain):
        if power:
            term_iv = term_iv * dom.pow_int(power)
    return term_iv


def _monomial_label(exp: tuple[int, ...]) -> str:
    names = ["x", "y", "tau"]
    parts: list[str] = []
    for i, power in enumerate(exp):
        if power == 0:
            continue
        name = names[i] if i < len(names) else f"z{i}"
        parts.append(name if power == 1 else f"{name}^{power}")
    return "1" if not parts else "*".join(parts)


def _signed_interval(iv: Interval, sign: float) -> Interval:
    return iv if sign >= 0 else -iv


def _shift_label(lo: Any, hi: Any) -> str:
    lo_f = _finite(lo)
    hi_f = _finite(hi)
    if lo_f is None or hi_f is None:
        return "unknown"
    if lo_f > 0:
        return "positive"
    if hi_f < 0:
        return "negative"
    return "straddles_zero"


def _zero_detail_row(context: Mapping[str, Any], expression: str) -> dict[str, Any]:
    return {
        "run_id": context.get("run_id", RUN_ID),
        "segment_index": context.get("segment_index", ""),
        "attempt_index": context.get("attempt_index", ""),
        "t_start": context.get("t_start", ""),
        "h_try": context.get("h_try", context.get("h", "")),
        "source_expression": expression,
        "dropped_total_degree": "",
        "term_rank": "",
        "monomial": "",
        "coefficient": "",
        "abs_interval_contribution": 0.0,
        "dropped_range_interval_lo": 0.0,
        "dropped_range_interval_hi": 0.0,
        "dropped_range_width": 0.0,
        "contribution_to_residual_lo": 0.0,
        "contribution_to_residual_hi": 0.0,
        "contribution_shift": "none",
    }


def _dropped_detail_rows(
    context: Mapping[str, Any],
    expression: str,
    dropped: Any,
    domain: Sequence[Interval],
    *,
    residual_sign: float = 1.0,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    if not getattr(dropped, "terms", None):
        return [_zero_detail_row(context, expression)]
    dropped_range = dropped.evaluate_interval(domain)
    dropped_lo, dropped_hi = _interval_lo_hi(dropped_range)
    dropped_width = _interval_width_value(dropped_range)
    term_rows: list[dict[str, Any]] = []
    for exp, coef in dropped.terms.items():
        term_iv = _term_interval(tuple(exp), coef, domain)
        contrib = _signed_interval(term_iv, residual_sign)
        lo, hi = _interval_lo_hi(contrib)
        term_lo, term_hi = _interval_lo_hi(term_iv)
        abs_contrib = ""
        if term_lo != "" and term_hi != "":
            abs_contrib = max(abs(float(term_lo)), abs(float(term_hi)))
        coef_value = _finite(float(coef.detach().cpu())) if hasattr(coef, "detach") else _finite(coef)
        term_rows.append(
            {
                "run_id": context.get("run_id", RUN_ID),
                "segment_index": context.get("segment_index", ""),
                "attempt_index": context.get("attempt_index", ""),
                "t_start": context.get("t_start", ""),
                "h_try": context.get("h_try", context.get("h", "")),
                "source_expression": expression,
                "dropped_total_degree": sum(int(v) for v in exp),
                "monomial": _monomial_label(tuple(exp)),
                "coefficient": coef_value if coef_value is not None else "",
                "abs_interval_contribution": abs_contrib,
                "dropped_range_interval_lo": dropped_lo,
                "dropped_range_interval_hi": dropped_hi,
                "dropped_range_width": dropped_width,
                "contribution_to_residual_lo": lo,
                "contribution_to_residual_hi": hi,
                "contribution_shift": _shift_label(lo, hi),
            }
        )
    term_rows.sort(key=lambda row: _finite(row.get("abs_interval_contribution")) or 0.0, reverse=True)
    for rank, row in enumerate(term_rows[:top_n], start=1):
        row["term_rank"] = rank
    return term_rows[:top_n]


def _truncation_detail_rows(candidate: TMVector, order: int, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        x = candidate[0]
        y = candidate[1]
        tau_index = len(candidate.domain) - 1
        rows.extend([_zero_detail_row(context, "x"), _zero_detail_row(context, "y")])
        x_sq_poly, x_sq_dropped = x.polynomial.mul_truncate(x.polynomial, order)
        rows.extend(_dropped_detail_rows(context, "x*x", x_sq_dropped, x.domain, residual_sign=1.0))
        x_sq = x * x
        _x_sq_y_poly, x_sq_y_dropped = x_sq.polynomial.mul_truncate(y.polynomial, order)
        rows.extend(_dropped_detail_rows(context, "(x*x)*y", x_sq_y_dropped, x.domain, residual_sign=-1.0))
        rhs_y = y - x - (x_sq * y)
        _rhs_kept, rhs_dropped = rhs_y.polynomial.truncate(order)
        rows.extend(_dropped_detail_rows(context, "y - x - x*x*y", rhs_dropped, rhs_y.domain, residual_sign=1.0))
        picard_x = y.integrate(tau_index)
        _px_kept, px_dropped = picard_x.polynomial.truncate(order)
        rows.extend(_dropped_detail_rows(context, "Picard integrated x", px_dropped, picard_x.domain, residual_sign=1.0))
        picard_y = rhs_y.integrate(tau_index)
        _py_kept, py_dropped = picard_y.polynomial.truncate(order)
        rows.extend(_dropped_detail_rows(context, "Picard integrated y", py_dropped, picard_y.domain, residual_sign=1.0))
    except Exception as exc:
        row = _zero_detail_row(context, "truncation_detail_exception")
        row["monomial"] = str(exc)
        rows.append(row)
    return rows


def _component_sum(row: Mapping[str, Any]) -> float | str:
    a = _finite(row.get("p_self_times_other_remainder_width"))
    b = _finite(row.get("p_other_times_self_remainder_width"))
    if a is None and b is None:
        return ""
    return (a or 0.0) + (b or 0.0)


def _breakdown_row(context: Mapping[str, Any], expression: str, breakdown: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "t_start": context.get("t_start", ""),
        "h_try": context.get("h_try", context.get("h", "")),
        "segment_index": context.get("segment_index", ""),
        "attempt_index": context.get("attempt_index", ""),
        "order": context.get("order", ""),
        "expression": expression,
        "kept_poly_range_width": breakdown.get("kept_poly_range_width", ""),
        "dropped_trunc_width": breakdown.get("dropped_trunc_width", ""),
        "polynomial_range_times_remainder_width": _component_sum(breakdown),
        "remainder_times_remainder_width": breakdown.get("remainder_times_remainder_width", ""),
        "cutoff_uncertainty_width": 0.0,
        "total_remainder_width": breakdown.get("total_remainder_width", ""),
        "finite": breakdown.get("finite", ""),
    }


def _vdp_breakdown(candidate: TMVector, order: int, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        x = candidate[0]
        y = candidate[1]
        x_sq_breakdown = taylor_model_mul_breakdown(x, x, order)
        x_sq = x * x
        rows.append(_breakdown_row(context, "x*x", x_sq_breakdown))
        x_sq_y_breakdown = taylor_model_mul_breakdown(x_sq, y, order)
        x_sq_y = x_sq * y
        rows.append(_breakdown_row(context, "(x*x)*y", x_sq_y_breakdown))
        rhs_y = y - x - x_sq_y
        poly_range = rhs_y.polynomial.evaluate_interval(rhs_y.domain)
        total_range = rhs_y.range_box()
        finite = bool(poly_range.is_finite() and rhs_y.remainder.is_finite() and total_range.is_finite())
        rows.append(
            {
                "t_start": context.get("t_start", ""),
                "h_try": context.get("h_try", context.get("h", "")),
                "segment_index": context.get("segment_index", ""),
                "attempt_index": context.get("attempt_index", ""),
                "order": context.get("order", ""),
                "expression": "y - x - x*x*y",
                "kept_poly_range_width": _width(poly_range),
                "dropped_trunc_width": "",
                "polynomial_range_times_remainder_width": "",
                "remainder_times_remainder_width": "",
                "cutoff_uncertainty_width": 0.0,
                "total_remainder_width": _width(rhs_y.remainder),
                "finite": finite,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "t_start": context.get("t_start", ""),
                "h_try": context.get("h_try", context.get("h", "")),
                "segment_index": context.get("segment_index", ""),
                "attempt_index": context.get("attempt_index", ""),
                "order": context.get("order", ""),
                "expression": "breakdown_exception",
                "finite": False,
                "total_remainder_width": str(exc),
            }
        )
    return rows


def _finish_attempt_rows(rows: list[dict[str, Any]], start: int, *, t_start: float, global_start: int) -> None:
    for offset, row in enumerate(rows[start:]):
        row["attempt_global_index"] = global_start + offset
        row["t_start"] = t_start
        h = _finite(row.get("h_try")) or _finite(row.get("h")) or 0.0
        row["h_try"] = h
        row["t_end"] = t_start + h


def _is_rejected(row: Mapping[str, Any]) -> bool:
    return str(row.get("validation_status", "")).lower() != "validated" or str(row.get("subset_result", "")).lower() == "false"


def _target_widths() -> tuple[float, float, float]:
    width = 2.0 * TARGET_REMAINDER_RADIUS
    return width, width, 2.0 * width


def _truncation_width_for_attempt(breakdown_rows: Sequence[Mapping[str, Any]], attempt: Mapping[str, Any]) -> float | str:
    seg = str(attempt.get("segment_index", ""))
    idx = str(attempt.get("attempt_index", ""))
    vals = [
        _finite(row.get("dropped_trunc_width"))
        for row in breakdown_rows
        if str(row.get("segment_index", "")) == seg and str(row.get("attempt_index", "")) == idx
    ]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else ""


def _attempt_output_row(row: Mapping[str, Any], breakdown_rows: Sequence[Mapping[str, Any]], notes: str) -> dict[str, Any]:
    target_x, target_y, target_sum = _target_widths()
    trunc_y = _truncation_width_for_attempt(breakdown_rows, row)
    return {
        "attempt_global_index": row.get("attempt_global_index", ""),
        "t_start": row.get("t_start", ""),
        "h_try": row.get("h_try", row.get("h", "")),
        "t_end": row.get("t_end", ""),
        "order": row.get("order", ORDER),
        "accepted": not _is_rejected(row),
        "rejection_reason": row.get("rejection_reason") or row.get("validation_message", ""),
        "residual_width_x": row.get("residual_width_x", ""),
        "residual_width_y": row.get("residual_width_y", ""),
        "residual_width_sum": row.get("residual_width_sum", ""),
        "target_width_x": target_x,
        "target_width_y": target_y,
        "target_width_sum": target_sum,
        "residual_over_target_ratio_x": _safe_ratio(row.get("residual_width_x"), target_x),
        "residual_over_target_ratio_y": _safe_ratio(row.get("residual_width_y"), target_y),
        "residual_over_target_ratio_sum": _safe_ratio(row.get("residual_width_sum"), target_sum),
        "final_width_x": row.get("candidate_final_width_x", ""),
        "final_width_y": row.get("candidate_final_width_y", ""),
        "final_width_sum": row.get("candidate_final_width_sum", ""),
        "polynomial_range_width_x": row.get("polynomial_range_width_x", ""),
        "polynomial_range_width_y": row.get("polynomial_range_width_y", ""),
        "remainder_width_x": row.get("remainder_width_x", ""),
        "remainder_width_y": row.get("remainder_width_y", ""),
        "cutoff_uncertainty_width_x": 0.0,
        "cutoff_uncertainty_width_y": 0.0,
        "truncation_uncertainty_width_x": 0.0,
        "truncation_uncertainty_width_y": trunc_y,
        "notes": notes,
    }


def _dominant_component(breakdown_rows: Sequence[Mapping[str, Any]], attempt: Mapping[str, Any]) -> str:
    seg = str(attempt.get("segment_index", ""))
    idx = str(attempt.get("attempt_index", ""))
    totals = {
        "polynomial_range_times_remainder": 0.0,
        "truncation": 0.0,
        "remainder_times_remainder": 0.0,
        "cutoff_uncertainty": 0.0,
    }
    for row in breakdown_rows:
        if str(row.get("segment_index", "")) != seg or str(row.get("attempt_index", "")) != idx:
            continue
        totals["polynomial_range_times_remainder"] = max(
            totals["polynomial_range_times_remainder"], _finite(row.get("polynomial_range_times_remainder_width")) or 0.0
        )
        totals["truncation"] = max(totals["truncation"], _finite(row.get("dropped_trunc_width")) or 0.0)
        totals["remainder_times_remainder"] = max(
            totals["remainder_times_remainder"], _finite(row.get("remainder_times_remainder_width")) or 0.0
        )
        totals["cutoff_uncertainty"] = max(totals["cutoff_uncertainty"], _finite(row.get("cutoff_uncertainty_width")) or 0.0)
    return max(totals.items(), key=lambda kv: kv[1])[0]


def _failed_dimension(row: Mapping[str, Any]) -> str:
    rx = _finite(row.get("residual_over_target_ratio_x")) or 0.0
    ry = _finite(row.get("residual_over_target_ratio_y")) or 0.0
    if rx > 1.0 and ry > 1.0:
        return "x_and_y" if abs(rx - ry) < 1e-12 else ("x" if rx > ry else "y")
    if rx > 1.0:
        return "x"
    if ry > 1.0:
        return "y"
    return "none"


def _failed_dimension_from_bounds(row: Mapping[str, Any]) -> str:
    escaped = []
    for dim in ("x", "y"):
        lo = _finite(row.get(f"residual_lo_{dim}"))
        hi = _finite(row.get(f"residual_hi_{dim}"))
        if lo is not None and hi is not None and (lo < -TARGET_REMAINDER_RADIUS or hi > TARGET_REMAINDER_RADIUS):
            escaped.append(dim)
    if escaped:
        return "_and_".join(escaped)
    return _failed_dimension(row)



def _attempt_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("segment_index", "")), str(row.get("attempt_index", ""))


def _residual_metrics(row: Mapping[str, Any], dim: str) -> dict[str, Any]:
    lo = _finite(row.get(f"residual_lo_{dim}"))
    hi = _finite(row.get(f"residual_hi_{dim}"))
    if lo is None or hi is None:
        return {
            "residual_center": "",
            "residual_radius": "",
            "target_interval_lo": -TARGET_REMAINDER_RADIUS,
            "target_interval_hi": TARGET_REMAINDER_RADIUS,
            "minimal_symmetric_radius_needed": "",
            "minimal_asymmetric_interval_lo": "",
            "minimal_asymmetric_interval_hi": "",
        }
    center = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    return {
        "residual_center": center,
        "residual_radius": radius,
        "target_interval_lo": -TARGET_REMAINDER_RADIUS,
        "target_interval_hi": TARGET_REMAINDER_RADIUS,
        "minimal_symmetric_radius_needed": max(abs(lo), abs(hi)),
        "minimal_asymmetric_interval_lo": lo,
        "minimal_asymmetric_interval_hi": hi,
    }


def _enriched_truncation_rows(
    detail_rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    attempts_by_key = {_attempt_key(row): row for row in attempts}
    rows: list[dict[str, Any]] = []
    for detail in detail_rows:
        attempt = attempts_by_key.get(_attempt_key(detail))
        if not attempt:
            continue
        t_start = _finite(attempt.get("t_start")) or 0.0
        if t_start <= 2.0 or not _is_rejected(attempt):
            continue
        failed_dim = _failed_dimension_from_bounds(attempt)
        dim = "y" if "y" in failed_dim else "x"
        row = dict(detail)
        row["t_start"] = attempt.get("t_start", row.get("t_start", ""))
        row["h_try"] = attempt.get("h_try", attempt.get("h", row.get("h_try", "")))
        row["failed_state_dimension"] = failed_dim
        row.update(_residual_metrics(attempt, dim))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            _finite(row.get("t_start")) or 0.0,
            _finite(row.get("h_try")) or 0.0,
            -(_finite(row.get("abs_interval_contribution")) or 0.0),
        )
    )
    return rows


def _final_failure_attempt(attempts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    failed = [row for row in attempts if (_finite(row.get("t_start")) or 0.0) > 2.0 and _is_rejected(row)]
    if not failed:
        failed = [row for row in attempts if _is_rejected(row)]
    return failed[-1] if failed else None


def _truncation_summary_row(
    summary: Mapping[str, Any],
    final_failure: Mapping[str, Any] | None,
    enriched_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failure_key = _attempt_key(final_failure or {})
    failure_terms = [
        row for row in enriched_rows
        if _attempt_key(row) == failure_key and (_finite(row.get("abs_interval_contribution")) or 0.0) > 0.0
    ]
    by_source: dict[str, float] = {}
    for row in failure_terms:
        source = str(row.get("source_expression", ""))
        by_source[source] = by_source.get(source, 0.0) + (_finite(row.get("abs_interval_contribution")) or 0.0)
    dominant_source = max(by_source.items(), key=lambda kv: kv[1])[0] if by_source else "none"
    sorted_terms = sorted([_finite(row.get("abs_interval_contribution")) or 0.0 for row in failure_terms], reverse=True)
    total = sum(sorted_terms)
    top3 = sum(sorted_terms[:3])
    dominance_shape = "few dropped monomials" if total > 0 and top3 / total >= 0.75 else ("many small dropped monomials" if total > 0 else "no dropped terms recorded")
    failed_dim = _failed_dimension_from_bounds(final_failure or {}) if final_failure else "none"
    dim = "y" if "y" in failed_dim else "x"
    metrics = _residual_metrics(final_failure or {}, dim)
    radius = _finite(metrics.get("residual_radius"))
    min_sym = _finite(metrics.get("minimal_symmetric_radius_needed"))
    center = abs(_finite(metrics.get("residual_center")) or 0.0)
    if radius is not None and radius <= TARGET_REMAINDER_RADIUS and min_sym is not None and min_sym > TARGET_REMAINDER_RADIUS:
        width_or_shift = "residual shift"
    elif radius is not None and radius > TARGET_REMAINDER_RADIUS:
        width_or_shift = "residual width"
    else:
        width_or_shift = "contained or unknown"
    likely_fix = "possibly" if dominant_source in {"(x*x)*y", "Picard integrated y"} and center > 0 else "unknown"
    return {
        "run_id": RUN_ID,
        "status": summary.get("status", ""),
        "last_validated_t": summary.get("last_validated_t", ""),
        "failure_t_start": (final_failure or {}).get("t_start", ""),
        "failure_h_try": (final_failure or {}).get("h_try", (final_failure or {}).get("h", "")),
        "failed_state_dimension": failed_dim,
        "dominant_source_expression": dominant_source,
        "dominance_shape": dominance_shape,
        "width_or_shift": width_or_shift,
        "minimal_symmetric_radius_needed": metrics.get("minimal_symmetric_radius_needed", ""),
        "minimal_asymmetric_interval_lo": metrics.get("minimal_asymmetric_interval_lo", ""),
        "minimal_asymmetric_interval_hi": metrics.get("minimal_asymmetric_interval_hi", ""),
        "tighter_range_bound_likely_fix": likely_fix,
        "notes": "diagnostic only; accepted validation semantics unchanged",
    }


def _write_truncation_detail_report(
    out_dir: Path,
    summary_row: Mapping[str, Any],
    enriched_rows: Sequence[Mapping[str, Any]],
) -> None:
    final_terms = [row for row in enriched_rows if (_finite(row.get("abs_interval_contribution")) or 0.0) > 0.0]
    source_totals: dict[str, float] = {}
    for row in final_terms:
        source = str(row.get("source_expression", ""))
        source_totals[source] = source_totals.get(source, 0.0) + (_finite(row.get("abs_interval_contribution")) or 0.0)
    source_bits = ", ".join(f"{k}={v:.6g}" for k, v in sorted(source_totals.items(), key=lambda kv: kv[1], reverse=True)[:4]) or "none"
    lines = [
        "# Truncation Localization Report",
        "",
        f"Failure dimension near the final rejected attempts: `{summary_row.get('failed_state_dimension', '')}`.",
        f"Is truncation dominated by a few dropped monomials or many small ones? `{summary_row.get('dominance_shape', '')}`.",
        f"Are dropped terms mostly from x*x*y or Picard integration? Dominant source=`{summary_row.get('dominant_source_expression', '')}`; source totals: {source_bits}.",
        f"Is containment failure caused by width or residual shift? `{summary_row.get('width_or_shift', '')}`.",
        f"What symmetric target radius would be needed for the failed step? `{summary_row.get('minimal_symmetric_radius_needed', '')}`.",
        f"Minimal asymmetric interval needed: `[{summary_row.get('minimal_asymmetric_interval_lo', '')}, {summary_row.get('minimal_asymmetric_interval_hi', '')}]`.",
        f"Would a tighter range bound on dropped terms likely fix it? `{summary_row.get('tighter_range_bound_likely_fix', '')}`.",
        "",
        "## Output Files",
        "",
        "- `truncation_localization_summary.csv` summarizes the final failure mechanism.",
        "- `truncation_top_terms.csv` lists top dropped monomial interval contributions near rejected attempts after t>2.",
        "- `dropped_terms_near_failure.png` and `residual_shift_near_failure.png` visualize term size and residual shift.",
    ]
    (out_dir / "truncation_localization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_truncation_detail_plots(out_dir: Path, enriched_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    nonzero = [row for row in enriched_rows if (_finite(row.get("abs_interval_contribution")) or 0.0) > 0.0]
    top = sorted(nonzero, key=lambda row: _finite(row.get("abs_interval_contribution")) or 0.0, reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    if top:
        labels = [f"{row.get('source_expression', '')}:{row.get('monomial', '')}" for row in top]
        vals = [_finite(row.get("abs_interval_contribution")) or 0.0 for row in top]
        ax.barh(list(range(len(vals))), vals, color="#4c78a8")
        ax.set_yticks(list(range(len(vals))))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
    ax.set_xlabel("absolute interval contribution")
    fig.tight_layout()
    fig.savefig(out_dir / "dropped_terms_near_failure.png", dpi=160)
    plt.close(fig)

    by_attempt: dict[tuple[float, float], Mapping[str, Any]] = {}
    for row in enriched_rows:
        t = _finite(row.get("t_start"))
        h = _finite(row.get("h_try"))
        if t is not None and h is not None:
            by_attempt[(t, h)] = row
    pts = sorted(by_attempt.items())
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    if pts:
        t_vals = [t for (t, _h), _row in pts]
        lo_vals = [_finite(row.get("minimal_asymmetric_interval_lo")) or 0.0 for (_key, row) in pts]
        hi_vals = [_finite(row.get("minimal_asymmetric_interval_hi")) or 0.0 for (_key, row) in pts]
        ax.plot(t_vals, lo_vals, marker="o", markersize=2.5, label="residual lo")
        ax.plot(t_vals, hi_vals, marker="o", markersize=2.5, label="residual hi")
    ax.axhline(-TARGET_REMAINDER_RADIUS, color="#111111", linestyle="--", linewidth=0.8, label="target")
    ax.axhline(TARGET_REMAINDER_RADIUS, color="#111111", linestyle="--", linewidth=0.8)
    ax.set_xlabel("t_start")
    ax.set_ylabel("failed residual interval")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_shift_near_failure.png", dpi=160)
    plt.close(fig)


def _write_residual_shift_outputs(final_failure: Mapping[str, Any] | None) -> None:
    if not final_failure:
        return
    failed_dim = _failed_dimension_from_bounds(final_failure)
    metrics = _residual_metrics(final_failure, "y")
    lo = _finite(metrics.get("minimal_asymmetric_interval_lo"))
    hi = _finite(metrics.get("minimal_asymmetric_interval_hi"))
    min_sym = _finite(metrics.get("minimal_symmetric_radius_needed"))
    if lo is None or hi is None or min_sym is None:
        return
    asym_width = hi - lo
    sym_width = 2.0 * min_sym
    row = {
        "run_id": RUN_ID,
        "t_start": final_failure.get("t_start", ""),
        "h_try": final_failure.get("h_try", final_failure.get("h", "")),
        "failed_dimension": failed_dim,
        "residual_lo_y": lo,
        "residual_hi_y": hi,
        "target_radius": TARGET_REMAINDER_RADIUS,
        "target_lo": -TARGET_REMAINDER_RADIUS,
        "target_hi": TARGET_REMAINDER_RADIUS,
        "minimal_symmetric_radius_needed": min_sym,
        "minimal_asymmetric_interval_lo": lo,
        "minimal_asymmetric_interval_hi": hi,
        "asymmetric_width": asym_width,
        "symmetric_width": sym_width,
        "asymmetric_much_tighter_than_symmetric": asym_width < 0.75 * sym_width,
        "accepted_semantics_changed": False,
    }
    out_dir = rescue.REPO_ROOT / "outputs" / "flowstar_style_residual_shift"
    out_dir.mkdir(parents=True, exist_ok=True)
    rescue._write_csv(out_dir / "residual_shift.csv", RESIDUAL_SHIFT_FIELDS, [row])
    lines = [
        "# Residual Shift Diagnostic",
        "",
        "This is diagnostic only; accepted validation still uses the symmetric target remainder interval.",
        f"Failed y residual interval: `[{lo}, {hi}]`.",
        f"Symmetric target interval: `[-{TARGET_REMAINDER_RADIUS}, {TARGET_REMAINDER_RADIUS}]`.",
        f"Minimal symmetric target radius required: `{min_sym}`.",
        f"Minimal asymmetric interval required: `[{lo}, {hi}]`.",
        f"Would asymmetric target be much tighter than symmetric loosened target? {_yes_no(bool(row['asymmetric_much_tighter_than_symmetric']))}.",
        "Do not claim parity with shifted/asymmetric target from this diagnostic.",
    ]
    (out_dir / "residual_shift_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_truncation_detail_outputs(
    out_dir: Path,
    summary: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
) -> None:
    enriched = _enriched_truncation_rows(detail_rows, attempts)
    final_failure = _final_failure_attempt(attempts)
    summary_row = _truncation_summary_row(summary, final_failure, enriched)
    rescue._write_csv(out_dir / "truncation_localization_summary.csv", TRUNCATION_SUMMARY_FIELDS, [summary_row])
    rescue._write_csv(out_dir / "truncation_top_terms.csv", TRUNCATION_DETAIL_FIELDS, enriched)
    _write_truncation_detail_report(out_dir, summary_row, enriched)
    _make_truncation_detail_plots(out_dir, enriched)
    _write_residual_shift_outputs(final_failure)


def _make_plots(out_dir: Path, attempts: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    pts = sorted(
        [
            (
                _finite(row.get("t_end")),
                _finite(row.get("residual_width_x")),
                _finite(row.get("residual_width_y")),
                _finite(row.get("residual_width_sum")),
            )
            for row in attempts
        ],
        key=lambda p: p[0] or 0.0,
    )
    pts = [p for p in pts if p[0] is not None]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label="residual x", marker="o", markersize=2.5)
        ax.plot([p[0] for p in pts], [p[2] for p in pts], label="residual y", marker="o", markersize=2.5)
        ax.plot([p[0] for p in pts], [p[3] for p in pts], label="residual sum", linewidth=1.2)
    target_x, _target_y, target_sum = _target_widths()
    ax.axhline(target_x, color="#111111", linestyle="--", linewidth=0.8, label="target per dimension")
    ax.axhline(target_sum, color="#666666", linestyle=":", linewidth=0.8, label="target sum")
    ax.set_xlabel("t")
    ax.set_ylabel("width")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_components_near_failure.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    accepted = [(row.get("t_hi"), row.get("h")) for row in segments if row.get("status") == "validated"]
    failed = [(row.get("t_start"), row.get("h_try")) for row in attempts if row.get("accepted") is False]
    if accepted:
        ax.plot([float(t) for t, _h in accepted], [float(h) for _t, h in accepted], marker="o", markersize=2.5, label="accepted h")
    if failed:
        ax.scatter([float(t) for t, _h in failed], [float(h) for _t, h in failed], color="#d62728", s=20, label="rejected h_try")
    ax.axhline(H_MIN, color="#111111", linestyle="--", linewidth=0.8, label="Flow* min step 0.002")
    ax.set_xlabel("t")
    ax.set_ylabel("h")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "step_size_near_failure.png", dpi=160)
    plt.close(fig)


def _write_report(
    out_dir: Path,
    summary: Mapping[str, Any],
    final_failure: Mapping[str, Any] | None,
    dominant: str,
) -> None:
    ratio_sum = final_failure.get("residual_over_target_ratio_sum", "") if final_failure else ""
    h_try = final_failure.get("h_try", "") if final_failure else ""
    slight = (_finite(ratio_sum) or math.inf) < 10.0
    lines = [
        "# Flowstar-Style Failure Localization Report",
        "",
        f"Last validated t: `{summary.get('last_validated_t', '')}`.",
        f"Failure reason: `{summary.get('failure_reason', '')}`.",
        f"Which state dimension fails target containment first? `{summary.get('failed_dimension_first', '')}`.",
        f"Is residual only slightly above target or orders of magnitude above? {'slight containment miss' if slight else 'orders of magnitude or unknown'}; final width-sum ratio=`{ratio_sum}`.",
        "Width ratios are diagnostic only: containment can fail when a residual interval is shifted outside the symmetric target even if its width is below the target width.",
        f"Is the dominant term still polynomial_range * remainder? {_yes_no(dominant == 'polynomial_range_times_remainder')}.",
        f"Is failure triggered by truncation, cutoff uncertainty, interval polynomial range, or RHS aggregation? Dominant recorded component: `{dominant}`.",
        f"What h_try fails at the end? `{h_try}`.",
        f"Would h below 0.002 likely help? `{summary.get('would_h_below_0.002_likely_help', '')}`.",
        f"Should the next fix be adaptive order, remainder-only Picard refinement, tighter range bounding, or real symbolic remainder queue? `{summary.get('next_recommendation', '')}`.",
        "",
        "## Output Files",
        "",
        "- `failure_step_attempts.csv` records the focused accepted/rejected attempts near failure.",
        "- `failure_residual_breakdown.csv` records the Van der Pol RHS multiplication breakdown.",
        "- `residual_components_near_failure.png` and `step_size_near_failure.png` visualize the failure neighborhood.",
    ]
    (out_dir / "failure_localization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def run_localization(
    out_dir: Path,
    *,
    max_horizon: float,
    wall_cap_s: float,
    truncation_detail: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    current: Any = rescue._initial_box()
    t = 0.0
    h_request = H_MAX
    segment_index = 0
    attempt_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    status = "max_horizon_reached"
    failure_reason = ""
    last_attempted_t = 0.0
    start_time = time.perf_counter()

    def callback(candidate: TMVector, order: int, attempt: int, context: Mapping[str, Any]) -> None:
        callback_context = dict(context)
        callback_context["attempt_index"] = attempt
        breakdown_rows.extend(_vdp_breakdown(candidate, order, callback_context))
        if truncation_detail:
            detail_rows.extend(_truncation_detail_rows(candidate, order, callback_context))

    while t < max_horizon - 1e-15:
        elapsed = time.perf_counter() - start_time
        if elapsed >= wall_cap_s:
            status = "timeout"
            failure_reason = f"wall-time cap reached before segment {segment_index}"
            break
        h = min(h_request, H_MAX, max_horizon - t)
        local_h_min = min(H_MIN, h)
        context = {
            "run_id": RUN_ID,
            "mode": "flowstar_style",
            "order": ORDER,
            "validation_mode": "target_remainder",
            "target_remainder_radius": TARGET_REMAINDER_RADIUS,
            "cutoff_threshold": "",
            "segment_index": segment_index,
            "t_start": t,
        }
        attempt_start = len(attempt_rows)
        global_start = len(attempt_rows)
        try:
            seg = _call_with_timeout(
                lambda: flowpipe_step_flowstar_style_adaptive(
                    van_der_pol_ode,
                    current,
                    h=h,
                    order=ORDER,
                    h_min=local_h_min,
                    h_max=H_MAX,
                    target_remainder_radius=TARGET_REMAINDER_RADIUS,
                    cutoff_threshold=None,
                    max_validation_attempts=2,
                    diagnostics=attempt_rows,
                    diagnostics_context=context,
                    rhs_breakdown_callback=callback,
                ),
                wall_cap_s - elapsed,
            )
        except StepTimeout as exc:
            status = "timeout"
            failure_reason = str(exc)
            break
        _finish_attempt_rows(attempt_rows, attempt_start, t_start=t, global_start=global_start)
        last_attempted_t = t + float(seg.h)
        final_box = seg.final_tm.range_box()
        finite = intervals_are_finite(final_box)
        row_status = "validated" if seg.status == "validated" and finite else "failed"
        x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = rescue._segment_bounds(final_box)
        segments.append(
            {
                "segment_index": segment_index,
                "status": row_status,
                "t_lo": t,
                "t_hi": t + float(seg.h),
                "h": float(seg.h),
                "width_x": width_x,
                "width_y": width_y,
                "width_sum": width_sum,
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "message": seg.message,
            }
        )
        if row_status != "validated":
            status = "failed"
            failure_reason = seg.message or "validation failed"
            break
        current = seg.reset_tm if seg.reset_tm is not None else seg.final_tm
        h_request = float(seg.next_h) if seg.next_h is not None else min(float(seg.h) * 1.5, H_MAX)
        t += float(seg.h)
        segment_index += 1

    accepted_segments = [row for row in segments if row.get("status") == "validated"]
    last20 = {row["segment_index"] for row in accepted_segments[-20:]}
    focused_raw = []
    for row in attempt_rows:
        t_start = _finite(row.get("t_start")) or 0.0
        in_last20 = row.get("segment_index") in last20
        rejected_after_2 = t_start > 2.0 and _is_rejected(row)
        if in_last20 or rejected_after_2:
            focused_raw.append(row)
    focused = []
    for row in focused_raw:
        note_bits = []
        if row.get("segment_index") in last20:
            note_bits.append("last_20_accepted_window")
        if (_finite(row.get("t_start")) or 0.0) > 2.0 and _is_rejected(row):
            note_bits.append("rejected_after_t_2")
        focused.append(_attempt_output_row(row, breakdown_rows, ";".join(note_bits)))

    kept_keys = {(str(row.get("segment_index", "")), str(row.get("attempt_index", ""))) for row in focused_raw}
    focused_breakdown = [
        {field: row.get(field, "") for field in BREAKDOWN_FIELDS}
        for row in breakdown_rows
        if (str(row.get("segment_index", "")), str(row.get("attempt_index", ""))) in kept_keys
    ]
    failure_pairs = [(raw, out) for raw, out in zip(focused_raw, focused) if out.get("accepted") is False]
    first_failure_raw, first_failure = failure_pairs[0] if failure_pairs else ({}, None)
    final_failure_raw, final_failure = failure_pairs[-1] if failure_pairs else ({}, None)
    dominant = _dominant_component(breakdown_rows, final_failure_raw) if final_failure else "unknown"
    failed_dim = _failed_dimension_from_bounds(first_failure_raw) if first_failure else "none"
    ratio_sum = _finite((final_failure or {}).get("residual_over_target_ratio_sum"))
    h_fail = _finite((final_failure or {}).get("h_try"))
    below_min_help = "unknown"
    if h_fail is not None and h_fail * 0.5 < H_MIN and ratio_sum is not None and ratio_sum < 4.0:
        below_min_help = "likely yes numerically, but only as a diagnostic because it goes below Flow* min step"
    recommendation = "tighter polynomial range bounding or real symbolic remainder queue"
    if dominant == "polynomial_range_times_remainder":
        recommendation = "tighter polynomial range bounding first, then real symbolic remainder queue"
    elif dominant == "truncation":
        recommendation = "adaptive order fallback first, then tighter polynomial range bounding"
    summary = {
        "run_id": RUN_ID,
        "status": status,
        "last_validated_t": accepted_segments[-1]["t_hi"] if accepted_segments else 0.0,
        "last_attempted_t": last_attempted_t,
        "num_accepted_steps": len(accepted_segments),
        "num_rejected_attempts": sum(1 for row in attempt_rows if _is_rejected(row)),
        "failure_reason": failure_reason,
        "failed_dimension_first": failed_dim,
        "failure_h_try": h_fail if h_fail is not None else "",
        "residual_over_target_ratio_x": (final_failure or {}).get("residual_over_target_ratio_x", ""),
        "residual_over_target_ratio_y": (final_failure or {}).get("residual_over_target_ratio_y", ""),
        "residual_over_target_ratio_sum": (final_failure or {}).get("residual_over_target_ratio_sum", ""),
        "dominant_breakdown_component": dominant,
        "would_h_below_0.002_likely_help": below_min_help,
        "next_recommendation": recommendation,
    }
    rescue._write_csv(out_dir / "failure_localization_summary.csv", SUMMARY_FIELDS, [summary])
    rescue._write_csv(out_dir / "failure_step_attempts.csv", ATTEMPT_FIELDS, focused)
    rescue._write_csv(out_dir / "failure_residual_breakdown.csv", BREAKDOWN_FIELDS, focused_breakdown)
    _write_report(out_dir, summary, final_failure, dominant)
    if truncation_detail:
        _write_truncation_detail_outputs(out_dir, summary, attempt_rows, detail_rows)
    _make_plots(out_dir, focused, segments[-25:])
    return focused, focused_breakdown


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_style_failure_localization"))
    parser.add_argument("--max-horizon", type=float, default=2.2)
    parser.add_argument("--wall-cap-s", type=float, default=600.0)
    parser.add_argument("--truncation-detail", action="store_true", help="Write dropped-term truncation localization diagnostics.")
    args = parser.parse_args(argv)
    run_localization(
        args.out_dir,
        max_horizon=float(args.max_horizon),
        wall_cap_s=float(args.wall_cap_s),
        truncation_detail=bool(args.truncation_detail),
    )
    print(f"wrote failure localization outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
