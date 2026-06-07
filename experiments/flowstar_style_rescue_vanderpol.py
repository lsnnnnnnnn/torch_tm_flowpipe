#!/usr/bin/env python3
"""Flow*-style rescue experiment for the Van der Pol PyTorch TM benchmark."""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe import (  # noqa: E402
    Interval,
    TMVector,
    flowpipe_step,
    flowpipe_step_flowstar_style_adaptive,
    flowpipe_step_from_tm,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode  # noqa: E402
from torch_tm_flowpipe.safety import intervals_are_finite  # noqa: E402

OLD_BEST_T = 0.7661635
FLOWSTAR_MIN_STEP = 0.002
ORIGINAL_FLOWSTAR_SEGMENTS = (
    REPO_ROOT / "outputs" / "flowstar_benchmark_parity" / "original_flowstar" / "original_flowstar_segments.csv"
)

SUMMARY_FIELDS = [
    "run_id",
    "mode",
    "order",
    "candidate_order",
    "output_order",
    "truncation_range_split",
    "normal_eval_range_split",
    "right_map_range_mode",
    "adaptive_order_fallback",
    "fallback_from_order",
    "refinement_pass",
    "residual_subset_current",
    "validation_mode",
    "reset_mode",
    "symbolic_queue_mode",
    "flowstar_symbolic_queue_max_size",
    "cutoff_threshold",
    "target_remainder_radius",
    "center_correction_width_factor",
    "center_correction_attempts",
    "center_corrections_applied",
    "center_corrected_dimensions",
    "max_center_correction_abs",
    "max_residual_radius_after_correction",
    "selective_high_degree_terms_top_k",
    "max_selective_retained_terms_count",
    "max_selective_dropped_remainder_width_sum",
    "max_flowstar_queue_size_after",
    "max_flowstar_propagated_remainder_width_sum",
    "max_symqueue_propagated_symbolic_width_sum",
    "max_symqueue_new_symbolic_width_sum",
    "max_symqueue_materialized_width_sum",
    "max_symqueue_linear_map_norm",
    "max_total_range_width_with_symbolic",
    "max_ordinary_only_range_width",
    "max_symbolic_contribution_width",
    "max_materialized_for_output_width",
    "max_target_checked_width",
    "max_insertion_symbolic_candidate_width",
    "max_insertion_truncation_width",
    "max_insertion_cutoff_width",
    "max_inserted_endpoint_width_sum",
    "max_normalized_reset_width_sum",
    "status",
    "runtime_s",
    "validated_segments",
    "last_validated_t",
    "last_attempted_t",
    "min_h_used",
    "min_regular_h_used",
    "min_final_alignment_h",
    "h_below_flowstar_min_count",
    "max_h_used",
    "num_step_rejections",
    "num_accepted_steps",
    "num_rejected_steps",
    "num_order8_steps",
    "num_order8_attempts",
    "failure_reason",
    "final_width_sum",
    "max_width_sum",
    "max_residual_width_sum",
    "max_remainder_width_sum",
    "notes",
]

SEGMENT_FIELDS = [
    "run_id",
    "mode",
    "order",
    "candidate_order",
    "output_order",
    "truncation_range_split",
    "normal_eval_range_split",
    "right_map_range_mode",
    "validation_mode",
    "reset_mode",
    "symbolic_queue_mode",
    "flowstar_symbolic_queue_max_size",
    "cutoff_threshold",
    "target_remainder_radius",
    "center_correction_width_factor",
    "selective_high_degree_terms_top_k",
    "selective_retained_terms_count",
    "selective_dropped_terms_count",
    "selective_nonretained_terms_count",
    "selective_dropped_remainder_width_sum",
    "selective_total_dropped_width_sum",
    "segment_index",
    "status",
    "validation_attempts",
    "t_lo",
    "t_hi",
    "h",
    "x_lo",
    "x_hi",
    "y_lo",
    "y_hi",
    "width_x",
    "width_y",
    "width_sum",
    "reset_box_source",
    "reset_width_x",
    "reset_width_y",
    "reset_width_sum",
    "flowstar_queue_size_before",
    "flowstar_queue_size_after",
    "flowstar_queue_reset",
    "flowstar_propagated_remainder_width_sum",
    "flowstar_total_remainder_width_sum",
    "queue_size",
    "propagated_symbolic_width_x",
    "propagated_symbolic_width_y",
    "propagated_symbolic_width_sum",
    "new_symbolic_width_x",
    "new_symbolic_width_y",
    "new_symbolic_width_sum",
    "materialized_width_x",
    "materialized_width_y",
    "materialized_width_sum",
    "materialized_for_output_width",
    "ordinary_only_range_width",
    "symbolic_contribution_width",
    "total_range_width_with_symbolic",
    "target_checked_width",
    "linear_map_norm",
    "scalars",
    "j_count",
    "phi_l_count",
    "current_linear_map_entries",
    "current_linear_map_norm",
    "current_phi_l_map_entries",
    "current_phi_l_map_norm",
    "scalar_x",
    "scalar_y",
    "ordinary_step_remainder_width_x",
    "ordinary_step_remainder_width_y",
    "ordinary_step_remainder_width_sum",
    "current_nonlinear_remainder_width_x",
    "current_nonlinear_remainder_width_y",
    "current_nonlinear_remainder_width_sum",
    "reset_box_width_x",
    "reset_box_width_y",
    "reset_box_width_sum",
    "right_map_range_width_x",
    "right_map_range_width_y",
    "right_map_range_width_sum",
    "target_check_width_x",
    "target_check_width_y",
    "target_check_width_sum",
    "output_only_symbolic_width_x",
    "output_only_symbolic_width_y",
    "output_only_symbolic_width_sum",
    "target_check_exceeds_target",
    "output_range_includes_symbolic_contributions",
    "conservative",
    "symqueue_approximation",
    "endpoint_box_width_x",
    "endpoint_box_width_y",
    "endpoint_box_width_sum",
    "endpoint_tm_width_x",
    "endpoint_tm_width_y",
    "endpoint_tm_width_sum",
    "inserted_endpoint_width_x",
    "inserted_endpoint_width_y",
    "inserted_endpoint_width_sum",
    "normalized_reset_width_x",
    "normalized_reset_width_y",
    "normalized_reset_width_sum",
    "normal_state_right_width_x",
    "normal_state_right_width_y",
    "normal_state_right_width_sum",
    "old_right_map_range_width_x",
    "old_right_map_range_width_y",
    "old_right_map_range_width_sum",
    "normal_right_map_range_width_x",
    "normal_right_map_range_width_y",
    "normal_right_map_range_width_sum",
    "insertion_truncation_width_x",
    "insertion_truncation_width_y",
    "insertion_truncation_width",
    "insertion_truncation_width_sum",
    "insertion_cutoff_width_x",
    "insertion_cutoff_width_y",
    "insertion_cutoff_width",
    "insertion_cutoff_width_sum",
    "insertion_truncation_ordinary_width",
    "insertion_cutoff_ordinary_width",
    "insertion_symbolic_candidate_width",
    "composed_poly_range_width_x",
    "composed_poly_range_width_y",
    "composed_poly_range_width",
    "composed_poly_range_width_sum",
    "output_remainder_width_x",
    "output_remainder_width_y",
    "output_remainder_width",
    "output_remainder_width_sum",
    "scale_x",
    "scale_y",
    "center_x",
    "center_y",
    "tmv_right_degree",
    "tmv_pre_degree",
    "tmv_right_term_count",
    "tmv_pre_term_count",
    "terms_before_insertion_truncation",
    "terms_after_insertion",
    "step_rejections",
    "next_h",
    "message",
]

RESET_BOX_FIELDS = [
    "run_id",
    "segment_index",
    "t_lo",
    "t_hi",
    "x_lo",
    "x_hi",
    "y_lo",
    "y_hi",
    "h",
    "order",
    "reset_box_source",
    "validation_mode",
    "reset_mode",
    "width_x",
    "width_y",
    "width_sum",
]

VALIDATION_ATTEMPT_FIELDS = [
    "run_id",
    "mode",
    "order",
    "candidate_order",
    "output_order",
    "truncation_range_split",
    "normal_eval_range_split",
    "right_map_range_mode",
    "adaptive_order_fallback",
    "fallback_from_order",
    "refinement_pass",
    "residual_subset_current",
    "validation_mode",
    "cutoff_threshold",
    "target_remainder_radius",
    "center_correction_width_factor",
    "selective_high_degree_terms_top_k",
    "segment_index",
    "adaptive_attempt_index",
    "t_lo",
    "t_hi",
    "attempt_index",
    "h",
    "h_try",
    "h_min",
    "h_max",
    "candidate_segment_width_x",
    "candidate_segment_width_y",
    "candidate_segment_width_sum",
    "candidate_final_width_x",
    "candidate_final_width_y",
    "candidate_final_width_sum",
    "residual_width_x",
    "residual_width_y",
    "residual_width_sum",
    "residual_lo_x",
    "residual_hi_x",
    "residual_lo_y",
    "residual_hi_y",
    "remainder_width_x",
    "remainder_width_y",
    "remainder_width_sum",
    "target_remainder_width",
    "target_remainder_width_sum",
    "target_checked_width",
    "center_correction_applied",
    "correction_value_x",
    "correction_value_y",
    "residual_before_lo_x",
    "residual_before_hi_x",
    "residual_before_lo_y",
    "residual_before_hi_y",
    "residual_after_lo_x",
    "residual_after_hi_x",
    "residual_after_lo_y",
    "residual_after_hi_y",
    "residual_before_center_x",
    "residual_before_center_y",
    "residual_after_center_x",
    "residual_after_center_y",
    "residual_before_radius_x",
    "residual_before_radius_y",
    "residual_after_radius_x",
    "residual_after_radius_y",
    "subset_after_correction",
    "subset_result",
    "rejection_reason",
    "polynomial_range_width_x",
    "polynomial_range_width_y",
    "polynomial_range_width_sum",
    "total_range_width_x",
    "total_range_width_y",
    "total_range_width_sum",
    "finite_residual",
    "validation_status",
    "validation_message",
    "tmp_remainder_lo_x",
    "tmp_remainder_hi_x",
    "tmp_remainder_width_x",
    "tmp_remainder_center_x",
    "tmp_remainder_radius_x",
    "tmp_remainder_lo_y",
    "tmp_remainder_hi_y",
    "tmp_remainder_width_y",
    "tmp_remainder_center_y",
    "tmp_remainder_radius_y",
    "tmp_remainder_width_sum",
    "poly_diff_range_lo_x",
    "poly_diff_range_hi_x",
    "poly_diff_range_width_x",
    "poly_diff_range_center_x",
    "poly_diff_range_radius_x",
    "poly_diff_range_lo_y",
    "poly_diff_range_hi_y",
    "poly_diff_range_width_y",
    "poly_diff_range_center_y",
    "poly_diff_range_radius_y",
    "poly_diff_range_width_sum",
    "ordinary_residual_range_lo_x",
    "ordinary_residual_range_hi_x",
    "ordinary_residual_range_width_x",
    "ordinary_residual_range_center_x",
    "ordinary_residual_range_radius_x",
    "ordinary_residual_range_lo_y",
    "ordinary_residual_range_hi_y",
    "ordinary_residual_range_width_y",
    "ordinary_residual_range_center_y",
    "ordinary_residual_range_radius_y",
    "ordinary_residual_range_width_sum",
    "normal_eval_range_lo_x",
    "normal_eval_range_hi_x",
    "normal_eval_range_width_x",
    "normal_eval_range_center_x",
    "normal_eval_range_radius_x",
    "normal_eval_range_lo_y",
    "normal_eval_range_hi_y",
    "normal_eval_range_width_y",
    "normal_eval_range_center_y",
    "normal_eval_range_radius_y",
    "normal_eval_range_width_sum",
    "subset_tmp_remainder",
    "subset_ordinary_residual",
    "validation_decision_difference",
    "candidate_terms_before_validation_terms_hash",
    "candidate_terms_before_validation_term_count",
    "candidate_terms_before_validation_max_degree",
    "candidate_terms_before_validation_high_degree_term_count",
    "candidate_terms_after_selective_terms_hash",
    "candidate_terms_after_selective_term_count",
    "candidate_terms_after_selective_max_degree",
    "candidate_terms_after_selective_high_degree_term_count",
    "validation_candidate_inside_terms_hash",
    "validation_candidate_inside_term_count",
    "validation_candidate_inside_max_degree",
    "validation_candidate_inside_high_degree_term_count",
    "validation_candidate_after_internal_terms_hash",
    "validation_candidate_after_internal_term_count",
    "validation_candidate_after_internal_max_degree",
    "validation_candidate_after_internal_high_degree_term_count",
]

COMPARISON_FIELDS = [
    "run_id",
    "py_status",
    "py_segments",
    "py_runtime_s",
    "py_last_validated_t",
    "py_last_width_sum",
    "py_tube_width_sum",
    "flowstar_segments_over_same_horizon",
    "flowstar_last_width_sum_near_T",
    "flowstar_tube_width_sum_over_same_horizon",
    "last_width_ratio",
    "tube_width_ratio",
    "max_time_overlap_width_ratio",
    "median_time_overlap_width_ratio",
]

RATIO_TRACE_FIELDS = [
    "run_id",
    "t",
    "py_width_sum",
    "flowstar_overlap_width_sum",
    "width_ratio",
]

NORMALIZED_INSERTION_RESET_FIELDS = [
    "run_id",
    "segment_index",
    "t_lo",
    "t_hi",
    "h",
    "symbolic_queue_mode",
    "right_map_range_mode",
    "endpoint_box_width_x",
    "endpoint_box_width_y",
    "endpoint_box_width_sum",
    "endpoint_tm_width_x",
    "endpoint_tm_width_y",
    "endpoint_tm_width_sum",
    "inserted_endpoint_width_x",
    "inserted_endpoint_width_y",
    "inserted_endpoint_width_sum",
    "reset_width_sum",
    "normalized_reset_width_x",
    "normalized_reset_width_y",
    "normalized_reset_width_sum",
    "normal_state_right_width_x",
    "normal_state_right_width_y",
    "normal_state_right_width_sum",
    "old_right_map_range_width_x",
    "old_right_map_range_width_y",
    "old_right_map_range_width_sum",
    "normal_right_map_range_width_x",
    "normal_right_map_range_width_y",
    "normal_right_map_range_width_sum",
    "insertion_truncation_width_x",
    "insertion_truncation_width_y",
    "insertion_truncation_width",
    "insertion_truncation_width_sum",
    "insertion_cutoff_width_x",
    "insertion_cutoff_width_y",
    "insertion_cutoff_width",
    "insertion_cutoff_width_sum",
    "insertion_truncation_ordinary_width",
    "insertion_cutoff_ordinary_width",
    "insertion_symbolic_candidate_width",
    "composed_poly_range_width_x",
    "composed_poly_range_width_y",
    "composed_poly_range_width",
    "composed_poly_range_width_sum",
    "output_remainder_width_x",
    "output_remainder_width_y",
    "output_remainder_width",
    "output_remainder_width_sum",
    "queue_size",
    "propagated_symbolic_width_x",
    "propagated_symbolic_width_y",
    "propagated_symbolic_width_sum",
    "new_symbolic_width_x",
    "new_symbolic_width_y",
    "new_symbolic_width_sum",
    "materialized_width_x",
    "materialized_width_y",
    "materialized_width_sum",
    "materialized_for_output_width",
    "ordinary_only_range_width",
    "symbolic_contribution_width",
    "total_range_width_with_symbolic",
    "target_checked_width",
    "linear_map_norm",
    "scalars",
    "j_count",
    "phi_l_count",
    "current_linear_map_entries",
    "current_linear_map_norm",
    "current_phi_l_map_entries",
    "current_phi_l_map_norm",
    "scalar_x",
    "scalar_y",
    "ordinary_step_remainder_width_x",
    "ordinary_step_remainder_width_y",
    "ordinary_step_remainder_width_sum",
    "current_nonlinear_remainder_width_x",
    "current_nonlinear_remainder_width_y",
    "current_nonlinear_remainder_width_sum",
    "reset_box_width_x",
    "reset_box_width_y",
    "reset_box_width_sum",
    "right_map_range_width_x",
    "right_map_range_width_y",
    "right_map_range_width_sum",
    "target_check_width_x",
    "target_check_width_y",
    "target_check_width_sum",
    "output_only_symbolic_width_x",
    "output_only_symbolic_width_y",
    "output_only_symbolic_width_sum",
    "target_check_exceeds_target",
    "output_range_includes_symbolic_contributions",
    "conservative",
    "symqueue_approximation",
    "scale_x",
    "scale_y",
    "center_x",
    "center_y",
    "tmv_right_degree",
    "tmv_pre_degree",
    "tmv_right_term_count",
    "tmv_pre_term_count",
    "terms_before_insertion_truncation",
    "terms_after_insertion",
]


NEXT_FIELDS = [
    "variant_group",
    "run_id",
    "validation_mode",
    "target_remainder_radius",
    "cutoff_threshold",
    "status",
    "last_validated_t",
    "runtime_s",
    "num_accepted_steps",
    "num_rejected_steps",
    "num_order8_steps",
    "candidate_order",
    "output_order",
    "truncation_range_split",
    "center_corrections_applied",
    "selective_high_degree_terms_top_k",
    "max_selective_retained_terms_count",
    "min_regular_h_used",
    "h_below_flowstar_min_count",
    "final_width_sum",
    "last_width_ratio",
    "tube_width_ratio",
    "notes",
]


RETAINED_TERM_FIELDS = [
    "run_id",
    "segment_index",
    "t_lo",
    "t_hi",
    "status",
    "selective_high_degree_terms_top_k",
    "state_index",
    "state_dimension",
    "term_rank",
    "retained",
    "monomial",
    "coefficient",
    "total_degree",
    "abs_interval_contribution",
    "term_interval_lo",
    "term_interval_hi",
    "term_interval_width",
]

VALIDATION_PATH_TERM_FIELDS = [
    "run_id",
    "segment_index",
    "attempt_index",
    "stage",
    "terms_hash",
    "term_count",
    "max_degree",
    "high_degree_term_count",
    "validation_status",
    "subset_result",
    "rejection_reason",
]

NEXT3_FIELDS = [
    *NEXT_FIELDS,
    "center_corrected_dimensions",
    "max_center_correction_abs",
    "max_selective_dropped_remainder_width_sum",
]

NEXT4_FIELDS = [
    "comparison_item",
    "run_id",
    "status",
    "last_validated_t",
    "flowstar_validated",
    "pytorch_validated",
    "runtime_s",
    "failure_reason",
    "decision_relevance",
    "notes",
]


class StepTimeout(RuntimeError):
    pass


def _initial_box() -> list[Interval]:
    return [Interval(1.1, 1.4), Interval(2.35, 2.45)]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in fields})


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.17g}"
        return str(value)
    return value


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _max_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | str:
    vals = [_finite_float(row.get(field)) for row in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else ""


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _max_abs_fields(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> float | str:
    vals: list[float] = []
    for row in rows:
        for field in fields:
            value = _finite_float(row.get(field))
            if value is not None:
                vals.append(abs(value))
    return max(vals) if vals else ""


def _interval_tuple(iv: Interval) -> tuple[float, float]:
    return iv.to_tuple()


def _segment_bounds(box: Sequence[Interval]) -> tuple[float, float, float, float, float, float, float]:
    x_lo, x_hi = _interval_tuple(box[0])
    y_lo, y_hi = _interval_tuple(box[1])
    width_x = x_hi - x_lo
    width_y = y_hi - y_lo
    return x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_x + width_y


def _call_with_timeout(fn: Callable[[], Any], timeout_s: float) -> Any:
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


def _finish_attempt_rows(rows: list[dict[str, Any]], start: int, *, run_id: str, t_lo: float) -> None:
    for row in rows[start:]:
        row.setdefault("run_id", run_id)
        row["t_lo"] = t_lo
        h = _finite_float(row.get("h")) or 0.0
        row["t_hi"] = t_lo + h


def _segment_row(
    *,
    spec: Mapping[str, Any],
    segment_index: int,
    status: str,
    seg: Any,
    t_lo: float,
    t_hi: float,
    box: Sequence[Interval],
) -> dict[str, Any]:
    x_lo, x_hi, y_lo, y_hi, width_x, width_y, width_sum = _segment_bounds(box)
    selective_stats = dict(getattr(seg, "selective_term_stats", None) or {})
    queue_stats = dict(getattr(seg, "flowstar_symbolic_queue_stats", None) or {})
    normal_stats = dict(getattr(seg, "flowstar_normal_stats", None) or {})
    reset_box = seg.reset_tm.range_box() if getattr(seg, "reset_tm", None) is not None else box
    _rx_lo, _rx_hi, _ry_lo, _ry_hi, reset_width_x, reset_width_y, reset_width_sum = _segment_bounds(reset_box)
    reset_mode = str(spec.get("reset_mode", normal_stats.get("reset_mode", queue_stats.get("reset_mode", "normalized_endpoint_box"))))
    if reset_mode == "flowstar_symbolic_remainder_queue":
        reset_box_source = "flowstar_symbolic_remainder_queue"
    elif reset_mode in {"normalized_insertion", "normalized_insertion_symqueue", "normalized_insertion_symqueue_split", "normalized_insertion_symqueue_v2", "normalized_insertion_horner"}:
        reset_box_source = reset_mode
    else:
        reset_box_source = "normalized_endpoint_reset_box"
    row = {
        "run_id": spec["run_id"],
        "mode": spec["mode"],
        "order": getattr(seg, "order", spec["order"]),
        "candidate_order": spec.get("candidate_order", spec["order"]),
        "output_order": spec.get("order", ""),
        "truncation_range_split": spec.get("truncation_range_split", ""),
        "normal_eval_range_split": spec.get("normal_eval_range_split", ""),
        "right_map_range_mode": normal_stats.get("right_map_range_mode", spec.get("right_map_range_mode", "standard")),
        "validation_mode": spec.get("validation_mode", "growth"),
        "reset_mode": reset_mode,
        "symbolic_queue_mode": queue_stats.get("symbolic_queue_mode", spec.get("symbolic_queue_mode", "")),
        "flowstar_symbolic_queue_max_size": spec.get("flowstar_symbolic_queue_max_size", ""),
        "cutoff_threshold": "" if spec.get("cutoff_threshold") is None else spec.get("cutoff_threshold"),
        "target_remainder_radius": spec.get("target_remainder_radius", ""),
        "center_correction_width_factor": spec.get("center_correction_width_factor", ""),
        "selective_high_degree_terms_top_k": spec.get("selective_high_degree_terms_top_k", ""),
        "segment_index": segment_index,
        "status": status,
        "validation_attempts": getattr(seg, "validation_attempts", ""),
        "t_lo": t_lo,
        "t_hi": t_hi,
        "h": getattr(seg, "h", t_hi - t_lo),
        "x_lo": x_lo,
        "x_hi": x_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "width_x": width_x,
        "width_y": width_y,
        "width_sum": width_sum,
        "reset_box_source": reset_box_source,
        "reset_width_x": reset_width_x,
        "reset_width_y": reset_width_y,
        "reset_width_sum": reset_width_sum,
        "flowstar_queue_size_before": queue_stats.get("queue_size_before", ""),
        "flowstar_queue_size_after": queue_stats.get("queue_size_after", ""),
        "flowstar_queue_reset": queue_stats.get("queue_reset", ""),
        "flowstar_propagated_remainder_width_sum": queue_stats.get("propagated_remainder_width_sum", ""),
        "flowstar_total_remainder_width_sum": queue_stats.get("total_remainder_width_sum", ""),
        "queue_size": queue_stats.get("queue_size", queue_stats.get("queue_size_after", "")),
        "propagated_symbolic_width_x": queue_stats.get("propagated_symbolic_width_x", ""),
        "propagated_symbolic_width_y": queue_stats.get("propagated_symbolic_width_y", ""),
        "propagated_symbolic_width_sum": queue_stats.get("propagated_symbolic_width_sum", ""),
        "new_symbolic_width_x": queue_stats.get("new_symbolic_width_x", ""),
        "new_symbolic_width_y": queue_stats.get("new_symbolic_width_y", ""),
        "new_symbolic_width_sum": queue_stats.get("new_symbolic_width_sum", ""),
        "materialized_width_x": queue_stats.get("materialized_width_x", ""),
        "materialized_width_y": queue_stats.get("materialized_width_y", ""),
        "materialized_width_sum": queue_stats.get("materialized_width_sum", ""),
        "materialized_for_output_width": queue_stats.get("materialized_for_output_width", ""),
        "ordinary_only_range_width": queue_stats.get("ordinary_only_range_width", ""),
        "symbolic_contribution_width": queue_stats.get("symbolic_contribution_width", ""),
        "total_range_width_with_symbolic": queue_stats.get("total_range_width_with_symbolic", ""),
        "target_checked_width": queue_stats.get("target_checked_width", ""),
        "linear_map_norm": queue_stats.get("linear_map_norm", queue_stats.get("linear_map_abs_sum", "")),
        "scalars": queue_stats.get("scalars", ""),
        "j_count": queue_stats.get("j_count", ""),
        "phi_l_count": queue_stats.get("phi_l_count", ""),
        "current_linear_map_entries": queue_stats.get("current_linear_map_entries", ""),
        "current_linear_map_norm": queue_stats.get("current_linear_map_norm", ""),
        "current_phi_l_map_entries": queue_stats.get("current_phi_l_map_entries", ""),
        "current_phi_l_map_norm": queue_stats.get("current_phi_l_map_norm", ""),
        "scalar_x": queue_stats.get("scalar_x", ""),
        "scalar_y": queue_stats.get("scalar_y", ""),
        "ordinary_step_remainder_width_x": queue_stats.get("ordinary_step_remainder_width_x", ""),
        "ordinary_step_remainder_width_y": queue_stats.get("ordinary_step_remainder_width_y", ""),
        "ordinary_step_remainder_width_sum": queue_stats.get("ordinary_step_remainder_width_sum", ""),
        "current_nonlinear_remainder_width_x": queue_stats.get("current_nonlinear_remainder_width_x", ""),
        "current_nonlinear_remainder_width_y": queue_stats.get("current_nonlinear_remainder_width_y", ""),
        "current_nonlinear_remainder_width_sum": queue_stats.get("current_nonlinear_remainder_width_sum", ""),
        "reset_box_width_x": queue_stats.get("reset_box_width_x", ""),
        "reset_box_width_y": queue_stats.get("reset_box_width_y", ""),
        "reset_box_width_sum": queue_stats.get("reset_box_width_sum", ""),
        "right_map_range_width_x": queue_stats.get("right_map_range_width_x", ""),
        "right_map_range_width_y": queue_stats.get("right_map_range_width_y", ""),
        "right_map_range_width_sum": queue_stats.get("right_map_range_width_sum", ""),
        "target_check_width_x": queue_stats.get("target_check_width_x", ""),
        "target_check_width_y": queue_stats.get("target_check_width_y", ""),
        "target_check_width_sum": queue_stats.get("target_check_width_sum", ""),
        "output_only_symbolic_width_x": queue_stats.get("output_only_symbolic_width_x", ""),
        "output_only_symbolic_width_y": queue_stats.get("output_only_symbolic_width_y", ""),
        "output_only_symbolic_width_sum": queue_stats.get("output_only_symbolic_width_sum", ""),
        "target_check_exceeds_target": queue_stats.get("target_check_exceeds_target", ""),
        "output_range_includes_symbolic_contributions": queue_stats.get("output_range_includes_symbolic_contributions", ""),
        "conservative": queue_stats.get("conservative", ""),
        "symqueue_approximation": queue_stats.get("approximation", ""),
        "endpoint_box_width_sum": normal_stats.get("endpoint_box_width_sum", ""),
        "inserted_endpoint_width_sum": normal_stats.get("inserted_endpoint_width_sum", ""),
        "normalized_reset_width_sum": normal_stats.get("normalized_reset_width_sum", ""),
        "old_right_map_range_width_sum": normal_stats.get("old_right_map_range_width_sum", ""),
        "normal_right_map_range_width_sum": normal_stats.get("normal_right_map_range_width_sum", ""),
        "insertion_truncation_width": normal_stats.get("insertion_truncation_width", ""),
        "insertion_cutoff_width": normal_stats.get("insertion_cutoff_width", ""),
        "insertion_truncation_ordinary_width": normal_stats.get("insertion_truncation_ordinary_width", ""),
        "insertion_cutoff_ordinary_width": normal_stats.get("insertion_cutoff_ordinary_width", ""),
        "insertion_symbolic_candidate_width": normal_stats.get("insertion_symbolic_candidate_width", ""),
        "composed_poly_range_width": normal_stats.get("composed_poly_range_width", ""),
        "output_remainder_width": normal_stats.get("output_remainder_width", ""),
        "scale_x": normal_stats.get("scale_x", ""),
        "scale_y": normal_stats.get("scale_y", ""),
        "center_x": normal_stats.get("center_x", ""),
        "center_y": normal_stats.get("center_y", ""),
        "tmv_right_degree": normal_stats.get("tmv_right_degree", ""),
        "tmv_pre_degree": normal_stats.get("tmv_pre_degree", ""),
        "tmv_right_term_count": normal_stats.get("tmv_right_term_count", ""),
        "tmv_pre_term_count": normal_stats.get("tmv_pre_term_count", ""),
        "terms_before_insertion_truncation": normal_stats.get("terms_before_insertion_truncation", ""),
        "terms_after_insertion": normal_stats.get("terms_after_insertion", ""),
        "horner_direct_range_width_sum": normal_stats.get("direct_range_width_sum", ""),
        "horner_range_width_sum": normal_stats.get("horner_range_width_sum", ""),
        "horner_direct_normal_range_width_sum": normal_stats.get("direct_normal_range_width_sum", ""),
        "horner_normal_range_width_sum": normal_stats.get("horner_normal_range_width_sum", ""),
        "horner_minus_direct_range_width_sum": normal_stats.get("horner_minus_direct_range_width_sum", ""),
        "horner_minus_direct_normal_range_width_sum": normal_stats.get("horner_minus_direct_normal_range_width_sum", ""),
        "horner_reduced_range": normal_stats.get("horner_reduced_range", ""),
        "horner_reduced_normal_range": normal_stats.get("horner_reduced_normal_range", ""),
        "horner_changed_range": normal_stats.get("horner_changed_range", ""),
        "horner_stage_count": normal_stats.get("horner_stage_count", ""),
        "horner_time_branch_stage_count": normal_stats.get("horner_time_branch_stage_count", ""),
        "horner_state_branch_stage_count": normal_stats.get("horner_state_branch_stage_count", ""),
        "horner_y_branch_stage_count": normal_stats.get("horner_y_branch_stage_count", ""),
        "horner_truncation_width_sum": normal_stats.get("horner_truncation_width_sum", ""),
        "horner_cutoff_width_sum": normal_stats.get("horner_cutoff_width_sum", ""),
        "horner_outer_remainder_width_sum": normal_stats.get("horner_outer_remainder_width_sum", ""),
        "step_rejections": getattr(seg, "step_rejections", 0),
        "next_h": "" if getattr(seg, "next_h", None) is None else getattr(seg, "next_h"),
        "message": getattr(seg, "message", ""),
    }
    for key in (
        "endpoint_box_width_x",
        "endpoint_box_width_y",
        "endpoint_tm_width_x",
        "endpoint_tm_width_y",
        "endpoint_tm_width_sum",
        "inserted_endpoint_width_x",
        "inserted_endpoint_width_y",
        "normalized_reset_width_x",
        "normalized_reset_width_y",
        "normal_state_right_width_x",
        "normal_state_right_width_y",
        "normal_state_right_width_sum",
        "old_right_map_range_width_x",
        "old_right_map_range_width_y",
        "old_right_map_range_width_sum",
        "normal_right_map_range_width_x",
        "normal_right_map_range_width_y",
        "normal_right_map_range_width_sum",
        "insertion_truncation_width_x",
        "insertion_truncation_width_y",
        "insertion_truncation_width_sum",
        "insertion_cutoff_width_x",
        "insertion_cutoff_width_y",
        "insertion_cutoff_width_sum",
        "composed_poly_range_width_x",
        "composed_poly_range_width_y",
        "composed_poly_range_width_sum",
        "output_remainder_width_x",
        "output_remainder_width_y",
        "output_remainder_width_sum",
    ):
        row[key] = normal_stats.get(key, "")
    row.update(selective_stats)
    details = getattr(seg, "selective_term_details", None)
    if details:
        row["_selective_term_details"] = [dict(item) for item in details]
    if normal_stats.get("_horner_stage_ranges"):
        row["_horner_stage_ranges"] = [dict(item) for item in normal_stats["_horner_stage_ranges"]]
    if normal_stats.get("_horner_top_components"):
        row["_horner_top_components"] = [dict(item) for item in normal_stats["_horner_top_components"]]
    return row


def _reset_box_rows(segment_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in segment_rows:
        if row.get("status") != "validated":
            continue
        rows.append(
            {
                "run_id": row.get("run_id", ""),
                "segment_index": row.get("segment_index", ""),
                "t_lo": row.get("t_lo", ""),
                "t_hi": row.get("t_hi", ""),
                "x_lo": row.get("x_lo", ""),
                "x_hi": row.get("x_hi", ""),
                "y_lo": row.get("y_lo", ""),
                "y_hi": row.get("y_hi", ""),
                "h": row.get("h", ""),
                "order": row.get("order", ""),
                "reset_box_source": row.get("reset_box_source", "normalized_endpoint_reset_box"),
                "validation_mode": row.get("validation_mode", ""),
                "reset_mode": row.get("reset_mode", ""),
                "width_x": row.get("reset_width_x", row.get("width_x", "")),
                "width_y": row.get("reset_width_y", row.get("width_y", "")),
                "width_sum": row.get("reset_width_sum", row.get("width_sum", "")),
            }
        )
    return rows


def _summarize_run(
    spec: Mapping[str, Any],
    *,
    max_horizon: float,
    status: str,
    runtime_s: float,
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    last_attempted_t: float,
    failure_reason: str,
    notes: str,
) -> dict[str, Any]:
    validated = [row for row in segment_rows if row.get("status") == "validated"]
    h_vals = [_finite_float(row.get("h")) for row in validated]
    h_vals = [h for h in h_vals if h is not None]
    flowstar_min_step = float(spec.get("h_min", FLOWSTAR_MIN_STEP))
    adaptive = spec.get("kind") == "adaptive" or spec.get("mode") == "flowstar_style"
    regular_h_vals: list[float] = []
    final_alignment_h_vals: list[float] = []
    h_below_flowstar_min_count = 0
    for row in validated:
        h = _finite_float(row.get("h"))
        t_hi = _finite_float(row.get("t_hi"))
        if h is None:
            continue
        is_final_alignment = (
            adaptive
            and t_hi is not None
            and abs(t_hi - float(max_horizon)) <= 1e-9
            and h < flowstar_min_step - 1e-12
        )
        if is_final_alignment:
            final_alignment_h_vals.append(h)
        else:
            regular_h_vals.append(h)
            if adaptive and h < flowstar_min_step - 1e-12:
                h_below_flowstar_min_count += 1
    num_rejected_steps = sum(int(row.get("step_rejections") or 0) for row in segment_rows)
    center_rows = [row for row in attempt_rows if _truthy(row.get("center_correction_applied"))]
    center_correction_attempts = len(center_rows)
    center_corrected_dimensions = sum(
        1
        for row in center_rows
        for field in ("correction_value_x", "correction_value_y")
        if abs(_finite_float(row.get(field)) or 0.0) > 0.0
    )
    max_center_correction_abs = _max_abs_fields(center_rows, ["correction_value_x", "correction_value_y"])
    max_after_radius = _max_field(center_rows, "residual_after_radius_x")
    max_after_radius_y = _max_field(center_rows, "residual_after_radius_y")
    if max_after_radius == "":
        max_after_radius = max_after_radius_y
    elif max_after_radius_y != "":
        max_after_radius = max(float(max_after_radius), float(max_after_radius_y))
    selective_retained = _max_field(segment_rows, "selective_retained_terms_count")
    selective_drop_width = _max_field(segment_rows, "selective_dropped_remainder_width_sum")
    return {
        "run_id": spec["run_id"],
        "mode": spec["mode"],
        "order": spec["order"],
        "candidate_order": spec.get("candidate_order", spec["order"]),
        "output_order": spec.get("order", ""),
        "truncation_range_split": spec.get("truncation_range_split", ""),
        "normal_eval_range_split": spec.get("normal_eval_range_split", ""),
        "right_map_range_mode": spec.get("right_map_range_mode", "standard"),
        "validation_mode": spec.get("validation_mode", "growth"),
        "reset_mode": spec.get("reset_mode", ""),
        "symbolic_queue_mode": spec.get("symbolic_queue_mode", ""),
        "flowstar_symbolic_queue_max_size": spec.get("flowstar_symbolic_queue_max_size", ""),
        "cutoff_threshold": "" if spec.get("cutoff_threshold") is None else spec.get("cutoff_threshold"),
        "target_remainder_radius": spec.get("target_remainder_radius", ""),
        "center_correction_width_factor": spec.get("center_correction_width_factor", ""),
        "center_correction_attempts": center_correction_attempts,
        "center_corrections_applied": center_correction_attempts,
        "center_corrected_dimensions": center_corrected_dimensions,
        "max_center_correction_abs": max_center_correction_abs,
        "max_residual_radius_after_correction": max_after_radius,
        "selective_high_degree_terms_top_k": spec.get("selective_high_degree_terms_top_k", ""),
        "max_selective_retained_terms_count": selective_retained,
        "max_selective_dropped_remainder_width_sum": selective_drop_width,
        "max_flowstar_queue_size_after": _max_field(segment_rows, "flowstar_queue_size_after"),
        "max_flowstar_propagated_remainder_width_sum": _max_field(segment_rows, "flowstar_propagated_remainder_width_sum"),
        "max_symqueue_propagated_symbolic_width_sum": _max_field(segment_rows, "propagated_symbolic_width_sum"),
        "max_symqueue_new_symbolic_width_sum": _max_field(segment_rows, "new_symbolic_width_sum"),
        "max_symqueue_materialized_width_sum": _max_field(segment_rows, "materialized_width_sum"),
        "max_symqueue_linear_map_norm": _max_field(segment_rows, "linear_map_norm"),
        "max_total_range_width_with_symbolic": _max_field(segment_rows, "total_range_width_with_symbolic"),
        "max_ordinary_only_range_width": _max_field(segment_rows, "ordinary_only_range_width"),
        "max_symbolic_contribution_width": _max_field(segment_rows, "symbolic_contribution_width"),
        "max_materialized_for_output_width": _max_field(segment_rows, "materialized_for_output_width"),
        "max_target_checked_width": _max_field(segment_rows, "target_checked_width"),
        "max_insertion_symbolic_candidate_width": _max_field(segment_rows, "insertion_symbolic_candidate_width"),
        "max_insertion_truncation_width": _max_field(segment_rows, "insertion_truncation_width"),
        "max_insertion_cutoff_width": _max_field(segment_rows, "insertion_cutoff_width"),
        "max_inserted_endpoint_width_sum": _max_field(segment_rows, "inserted_endpoint_width_sum"),
        "max_normalized_reset_width_sum": _max_field(segment_rows, "normalized_reset_width_sum"),
        "status": status,
        "runtime_s": runtime_s,
        "validated_segments": len(validated),
        "last_validated_t": float(validated[-1]["t_hi"]) if validated else 0.0,
        "last_attempted_t": last_attempted_t,
        "min_h_used": min(h_vals) if h_vals else "",
        "min_regular_h_used": min(regular_h_vals) if regular_h_vals else "",
        "min_final_alignment_h": min(final_alignment_h_vals) if final_alignment_h_vals else "",
        "h_below_flowstar_min_count": h_below_flowstar_min_count,
        "max_h_used": max(h_vals) if h_vals else "",
        "num_step_rejections": num_rejected_steps,
        "num_accepted_steps": len(validated),
        "num_rejected_steps": num_rejected_steps,
        "num_order8_steps": sum(1 for row in validated if int(row.get("order") or 0) == 8),
        "num_order8_attempts": sum(1 for row in attempt_rows if int(row.get("order") or 0) == 8),
        "failure_reason": failure_reason,
        "final_width_sum": validated[-1]["width_sum"] if validated else "",
        "max_width_sum": _max_field(validated, "width_sum"),
        "max_residual_width_sum": _max_field(attempt_rows, "residual_width_sum"),
        "max_remainder_width_sum": _max_field(attempt_rows, "remainder_width_sum"),
        "notes": notes,
    }


def _run_fixed(spec: Mapping[str, Any], *, max_horizon: float, wall_cap_s: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    current_box = _initial_box()
    current_tm = TMVector.identity(current_box, order=int(spec["order"]))
    t = 0.0
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    status = "max_horizon_reached"
    failure_reason = ""
    last_attempted_t = 0.0
    start = time.perf_counter()
    segment_index = 0

    while t < max_horizon - 1e-15:
        elapsed = time.perf_counter() - start
        if elapsed >= wall_cap_s:
            status = "timeout"
            failure_reason = f"wall-time cap reached before segment {segment_index}"
            break
        h = min(float(spec["h"]), max_horizon - t)
        context = {
            "run_id": spec["run_id"],
            "mode": spec["mode"],
            "order": spec["order"],
            "candidate_order": spec.get("candidate_order", spec["order"]),
            "output_order": spec.get("order", ""),
            "truncation_range_split": spec.get("truncation_range_split", ""),
            "validation_mode": "growth",
            "cutoff_threshold": "",
            "target_remainder_radius": "",
            "segment_index": segment_index,
        }
        attempt_start = len(attempt_rows)
        try:
            if spec["mode"] == "range_only":
                seg = _call_with_timeout(
                    lambda: flowpipe_step(
                        van_der_pol_ode,
                        current_box,
                        h,
                        int(spec["order"]),
                        diagnostics=attempt_rows,
                        diagnostics_context=context,
                    ),
                    wall_cap_s - elapsed,
                )
            else:
                seg = _call_with_timeout(
                    lambda: flowpipe_step_from_tm(
                        van_der_pol_ode,
                        current_tm,
                        h,
                        int(spec["order"]),
                        diagnostics=attempt_rows,
                        diagnostics_context=context,
                    ),
                    wall_cap_s - elapsed,
                )
        except StepTimeout as exc:
            status = "timeout"
            failure_reason = str(exc)
            break
        _finish_attempt_rows(attempt_rows, attempt_start, run_id=spec["run_id"], t_lo=t)
        last_attempted_t = t + h
        final_box = seg.final_tm.range_box()
        finite = intervals_are_finite(final_box)
        row_status = "validated" if seg.status == "validated" and finite else "failed"
        segment_rows.append(_segment_row(spec=spec, segment_index=segment_index, status=row_status, seg=seg, t_lo=t, t_hi=t + h, box=final_box))
        if row_status != "validated":
            status = "failed"
            failure_reason = seg.message or "validation failed"
            break
        if spec["mode"] == "range_only":
            current_box = [iv.inflate(1e-9) for iv in final_box]
        else:
            current_tm = seg.final_tm
        t += h
        segment_index += 1

    runtime_s = time.perf_counter() - start
    notes = "validated to requested horizon" if status == "max_horizon_reached" else failure_reason
    return (
        _summarize_run(
            spec,
            max_horizon=max_horizon,
            status=status,
            runtime_s=runtime_s,
            segment_rows=segment_rows,
            attempt_rows=attempt_rows,
            last_attempted_t=last_attempted_t,
            failure_reason=failure_reason,
            notes=notes,
        ),
        segment_rows,
        attempt_rows,
    )


def _run_adaptive(spec: Mapping[str, Any], *, max_horizon: float, wall_cap_s: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    current: Any = _initial_box()
    t = 0.0
    h_request = float(spec.get("h_max", 0.1))
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    status = "max_horizon_reached"
    failure_reason = ""
    last_attempted_t = 0.0
    start = time.perf_counter()
    segment_index = 0
    flowstar_queue_state = None
    flowstar_normal_state = None

    while t < max_horizon - 1e-15:
        elapsed = time.perf_counter() - start
        if elapsed >= wall_cap_s:
            status = "timeout"
            failure_reason = f"wall-time cap reached before segment {segment_index}"
            break
        h = min(h_request, float(spec.get("h_max", 0.1)), max_horizon - t)
        local_h_min = min(float(spec.get("h_min", 0.002)), h)
        context = {
            "run_id": spec["run_id"],
            "mode": spec["mode"],
            "order": spec["order"],
            "candidate_order": spec.get("candidate_order", spec["order"]),
            "output_order": spec.get("order", ""),
            "truncation_range_split": spec.get("truncation_range_split", ""),
            "normal_eval_range_split": spec.get("normal_eval_range_split", ""),
            "right_map_range_mode": spec.get("right_map_range_mode", "standard"),
            "validation_mode": spec["validation_mode"],
            "reset_mode": spec.get("reset_mode", "normalized_endpoint_box"),
            "symbolic_queue_mode": spec.get("symbolic_queue_mode", ""),
            "flowstar_symbolic_queue_max_size": spec.get("flowstar_symbolic_queue_max_size", ""),
            "cutoff_threshold": "" if spec.get("cutoff_threshold") is None else spec.get("cutoff_threshold"),
            "target_remainder_radius": spec.get("target_remainder_radius", ""),
            "center_correction_width_factor": spec.get("center_correction_width_factor", ""),
            "selective_high_degree_terms_top_k": spec.get("selective_high_degree_terms_top_k", ""),
            "segment_index": segment_index,
        }
        attempt_start = len(attempt_rows)
        try:
            seg = _call_with_timeout(
                lambda: flowpipe_step_flowstar_style_adaptive(
                    van_der_pol_ode,
                    current,
                    h=h,
                    order=int(spec["order"]),
                    h_min=local_h_min,
                    h_max=float(spec.get("h_max", 0.1)),
                    target_remainder_radius=float(spec.get("target_remainder_radius", 1e-4)),
                    center_correction_width_factor=float(spec.get("center_correction_width_factor") or 1.05),
                    cutoff_threshold=spec.get("cutoff_threshold"),
                    max_validation_attempts=int(spec.get("max_validation_attempts", 2)),
                    validation_mode=str(spec.get("validation_mode", "target_remainder")),
                    adaptive_order_fallback=spec.get("adaptive_order_fallback"),
                    adaptive_order_threshold_factor=float(spec.get("adaptive_order_threshold_factor", 1.25)),
                    candidate_order=spec.get("candidate_order"),
                    truncation_range_split=spec.get("truncation_range_split"),
                    selective_high_degree_terms_top_k=spec.get("selective_high_degree_terms_top_k"),
                    normal_eval_range_split=spec.get("normal_eval_range_split"),
                    reset_mode=str(spec.get("reset_mode", "normalized_endpoint_box")),
                    right_map_range_mode=str(spec.get("right_map_range_mode", "standard")),
                    symbolic_queue_mode=str(spec.get("symbolic_queue_mode", "")),
                    scalar_recenter_remainder_midpoint=bool(spec.get("scalar_recenter_remainder_midpoint", False)),
                    horner_diagnostic=bool(spec.get("horner_diagnostic", False)) and segment_index >= int(spec.get("horner_diagnostic_segment_min", 0) or 0),
                    flowstar_symbolic_queue_state=flowstar_queue_state,
                    flowstar_symbolic_queue_max_size=int(spec.get("flowstar_symbolic_queue_max_size") or 100),
                    flowstar_normal_state=flowstar_normal_state,
                    diagnostics=attempt_rows,
                    diagnostics_context=context,
                ),
                wall_cap_s - elapsed,
            )
        except StepTimeout as exc:
            status = "timeout"
            failure_reason = str(exc)
            break
        _finish_attempt_rows(attempt_rows, attempt_start, run_id=spec["run_id"], t_lo=t)
        last_attempted_t = t + float(seg.h)
        final_box = seg.final_tm.range_box()
        finite = intervals_are_finite(final_box)
        row_status = "validated" if seg.status == "validated" and finite else "failed"
        segment_rows.append(_segment_row(spec=spec, segment_index=segment_index, status=row_status, seg=seg, t_lo=t, t_hi=t + float(seg.h), box=final_box))
        if row_status != "validated":
            status = "failed"
            failure_reason = seg.message or "validation failed"
            break
        flowstar_queue_state = getattr(seg, "flowstar_symbolic_queue_state", flowstar_queue_state)
        flowstar_normal_state = getattr(seg, "flowstar_normal_state", flowstar_normal_state)
        current = seg.reset_tm if seg.reset_tm is not None else seg.final_tm
        h_request = float(seg.next_h) if seg.next_h is not None else min(float(seg.h) * 1.5, float(spec.get("h_max", 0.1)))
        t += float(seg.h)
        segment_index += 1

    runtime_s = time.perf_counter() - start
    notes = "validated to requested horizon" if status == "max_horizon_reached" else failure_reason
    return (
        _summarize_run(
            spec,
            max_horizon=max_horizon,
            status=status,
            runtime_s=runtime_s,
            segment_rows=segment_rows,
            attempt_rows=attempt_rows,
            last_attempted_t=last_attempted_t,
            failure_reason=failure_reason,
            notes=notes,
        ),
        segment_rows,
        attempt_rows,
    )


def _configs() -> list[dict[str, Any]]:
    def flowstar_spec(
        run_id: str,
        *,
        order: int = 6,
        target_remainder_radius: float = 1e-4,
        cutoff_threshold: float | None = None,
        validation_mode: str = "target_remainder",
        max_validation_attempts: int = 2,
        adaptive_order_fallback: int | None = None,
        candidate_order: int | None = None,
        truncation_range_split: int | None = None,
        center_correction_width_factor: float = 1.05,
        selective_high_degree_terms_top_k: int | None = None,
        normal_eval_range_split: int | None = None,
        right_map_range_mode: str = "standard",
        scalar_recenter_remainder_midpoint: bool = False,
        reset_mode: str = "normalized_endpoint_box",
        symbolic_queue_mode: str = "",
        flowstar_symbolic_queue_max_size: int | None = None,
        horner_diagnostic: bool = False,
    ) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "run_id": run_id,
            "mode": "flowstar_style",
            "order": order,
            "validation_mode": validation_mode,
            "reset_mode": reset_mode,
            "symbolic_queue_mode": symbolic_queue_mode,
            "right_map_range_mode": right_map_range_mode,
            "flowstar_symbolic_queue_max_size": "" if flowstar_symbolic_queue_max_size is None else int(flowstar_symbolic_queue_max_size),
            "target_remainder_radius": target_remainder_radius,
            "center_correction_width_factor": center_correction_width_factor if validation_mode == "target_remainder_centered" else "",
            "cutoff_threshold": cutoff_threshold,
            "h_min": 0.002,
            "h_max": 0.1,
            "max_validation_attempts": max_validation_attempts,
            "kind": "adaptive",
        }
        if candidate_order is not None:
            spec["candidate_order"] = int(candidate_order)
        if truncation_range_split is not None:
            spec["truncation_range_split"] = int(truncation_range_split)
        if normal_eval_range_split is not None:
            spec["normal_eval_range_split"] = int(normal_eval_range_split)
        if scalar_recenter_remainder_midpoint:
            spec["scalar_recenter_remainder_midpoint"] = True
        if horner_diagnostic:
            spec["horner_diagnostic"] = True
        if selective_high_degree_terms_top_k is not None:
            spec["selective_high_degree_terms_top_k"] = int(selective_high_degree_terms_top_k)
        if adaptive_order_fallback is not None:
            spec["adaptive_order_fallback"] = adaptive_order_fallback
            spec["adaptive_order_threshold_factor"] = 1.25
        return spec

    return [
        {
            "run_id": "baseline_range_only_o6_s4",
            "mode": "range_only",
            "order": 6,
            "validation_mode": "growth",
            "h": 0.025,
            "kind": "fixed",
        },
        {
            "run_id": "baseline_dependency_preserving_o4_s1",
            "mode": "dependency_preserving",
            "order": 4,
            "validation_mode": "growth",
            "h": 0.1,
            "kind": "fixed",
        },
        flowstar_spec("flowstar_style_o4_target", order=4),
        flowstar_spec("flowstar_style_o6_target", order=6),
        flowstar_spec("flowstar_style_o4_target_cutoff", order=4, cutoff_threshold=1e-10),
        flowstar_spec("flowstar_style_o6_target_cutoff", order=6, cutoff_threshold=1e-10),
        flowstar_spec(
            "flowstar_style_o4_target_insert",
            order=4,
            reset_mode="normalized_insertion",
        ),
        flowstar_spec(
            "flowstar_style_o4_target_cutoff_insert",
            order=4,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion",
        ),
        flowstar_spec(
            "flowstar_style_o4_target_insert_horner",
            order=4,
            reset_mode="normalized_insertion_horner",
        ),
        flowstar_spec(
            "flowstar_style_o4_target_cutoff_insert_horner",
            order=4,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion_horner",
        ),
        flowstar_spec(
            "flowstar_style_o4_target_insert_scalars",
            order=4,
            reset_mode="normalized_insertion",
            scalar_recenter_remainder_midpoint=True,
        ),
        flowstar_spec(
            "flowstar_style_o4_target_insert_symqueue",
            order=4,
            reset_mode="normalized_insertion_symqueue",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o4_target_cutoff_insert_symqueue",
            order=4,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion_symqueue",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec("flowstar_style_o6_target_adaptive_order_8", order=6, adaptive_order_fallback=8),
        flowstar_spec(
            "flowstar_style_o6_target_cutoff_adaptive_order_8",
            order=6,
            cutoff_threshold=1e-10,
            adaptive_order_fallback=8,
        ),
        flowstar_spec("flowstar_style_o6_target_r2e-4", order=6, target_remainder_radius=2e-4),
        flowstar_spec("flowstar_style_o6_target_r5e-4", order=6, target_remainder_radius=5e-4),
        flowstar_spec(
            "flowstar_style_o6_target_refined",
            order=6,
            validation_mode="target_remainder_refined",
            max_validation_attempts=8,
        ),
        flowstar_spec(
            "flowstar_style_o6_target_refined_cutoff",
            order=6,
            validation_mode="target_remainder_refined",
            cutoff_threshold=1e-10,
            max_validation_attempts=8,
        ),
        flowstar_spec("flowstar_style_o6_candidate8_output6", order=6, candidate_order=8),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_insert",
            order=6,
            candidate_order=8,
            reset_mode="normalized_insertion",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_insert_scalars",
            order=6,
            candidate_order=8,
            reset_mode="normalized_insertion",
            scalar_recenter_remainder_midpoint=True,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
        ),
        flowstar_spec("flowstar_style_o6_target_truncsplit2", order=6, truncation_range_split=2),
        flowstar_spec("flowstar_style_o6_target_truncsplit4", order=6, truncation_range_split=4),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_truncsplit2",
            order=6,
            candidate_order=8,
            truncation_range_split=2,
        ),
        flowstar_spec(
            "flowstar_style_o6_target_centered",
            order=6,
            validation_mode="target_remainder_centered",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_centered",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            validation_mode="target_remainder_centered",
        ),
        flowstar_spec("flowstar_style_o6_candidate8_output6_keep1", order=6, candidate_order=8, selective_high_degree_terms_top_k=1),
        flowstar_spec("flowstar_style_o6_candidate8_output6_keep2", order=6, candidate_order=8, selective_high_degree_terms_top_k=2),
        flowstar_spec("flowstar_style_o6_candidate8_output6_keep4", order=6, candidate_order=8, selective_high_degree_terms_top_k=4),
        flowstar_spec("flowstar_style_o6_candidate8_output6_keep8", order=6, candidate_order=8, selective_high_degree_terms_top_k=8),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_keep1_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
            selective_high_degree_terms_top_k=1,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_keep2_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
            selective_high_degree_terms_top_k=2,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_keep4_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
            selective_high_degree_terms_top_k=4,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_keep8_centered",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_centered",
            selective_high_degree_terms_top_k=8,
        ),
        flowstar_spec(
            "flowstar_style_o6_target_flowstar_ctrunc",
            order=6,
            validation_mode="target_remainder_flowstar_ctrunc",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_flowstar_ctrunc",
            order=6,
            candidate_order=8,
            validation_mode="target_remainder_flowstar_ctrunc",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_flowstar_ctrunc",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            validation_mode="target_remainder_flowstar_ctrunc",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_symqueue",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            reset_mode="flowstar_symbolic_remainder_queue",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_insert",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_insert_horner",
            order=6,
            candidate_order=8,
            reset_mode="normalized_insertion_horner",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_insert_horner",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion_horner",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_insert_symqueue",
            order=6,
            candidate_order=8,
            reset_mode="normalized_insertion_symqueue",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion_symqueue",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o4_target_insert_normaleval",
            order=4,
            reset_mode="normalized_insertion",
            right_map_range_mode="normal_eval",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_insert_normaleval",
            order=6,
            candidate_order=8,
            reset_mode="normalized_insertion",
            right_map_range_mode="normal_eval",
        ),
        flowstar_spec(
            "flowstar_style_o4_target_cutoff_insert_normaleval",
            order=4,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion",
            right_map_range_mode="normal_eval",
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_insert_normaleval",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion",
            right_map_range_mode="normal_eval",
        ),
        flowstar_spec(
            "flowstar_style_o4_target_insert_normaleval_symqueue_split",
            order=4,
            reset_mode="normalized_insertion_symqueue_split",
            right_map_range_mode="normal_eval",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_insert_normaleval_symqueue_split",
            order=6,
            candidate_order=8,
            reset_mode="normalized_insertion_symqueue_split",
            right_map_range_mode="normal_eval",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o4_target_insert_symqueue_split",
            order=4,
            reset_mode="normalized_insertion_symqueue_split",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o4_target_cutoff_insert_symqueue_split",
            order=4,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion_symqueue_split",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_insert_symqueue_split",
            order=6,
            candidate_order=8,
            reset_mode="normalized_insertion_symqueue_split",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue_split",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion_symqueue_split",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o4_target_insert_symqueue_v2",
            order=4,
            reset_mode="normalized_insertion_symqueue_v2",
            symbolic_queue_mode="flowstar_linear_v2",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o4_target_cutoff_insert_symqueue_v2",
            order=4,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion_symqueue_v2",
            symbolic_queue_mode="flowstar_linear_v2",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_insert_symqueue_v2",
            order=6,
            candidate_order=8,
            reset_mode="normalized_insertion_symqueue_v2",
            symbolic_queue_mode="flowstar_linear_v2",
            flowstar_symbolic_queue_max_size=100,
        ),
        flowstar_spec(
            "flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue_v2",
            order=6,
            candidate_order=8,
            cutoff_threshold=1e-10,
            reset_mode="normalized_insertion_symqueue_v2",
            symbolic_queue_mode="flowstar_linear_v2",
            flowstar_symbolic_queue_max_size=100,
        ),
    ]


def _normalize_config_ids(config_ids: Sequence[str] | None) -> list[str]:
    if config_ids is None:
        return []
    normalized: list[str] = []
    for raw in config_ids:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                normalized.append(part)
    return normalized


def _select_configs(config_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    configs = _configs()
    selected_ids = _normalize_config_ids(config_ids)
    if not selected_ids:
        return configs
    by_id = {str(spec["run_id"]): spec for spec in configs}
    missing = [run_id for run_id in selected_ids if run_id not in by_id]
    if missing:
        raise ValueError(f"unknown config(s): {', '.join(missing)}")
    return [by_id[run_id] for run_id in selected_ids]


def _best(rows: Sequence[Mapping[str, Any]], *, mode: str | None = None, not_mode: str | None = None) -> Mapping[str, Any] | None:
    selected = []
    for row in rows:
        if mode is not None and row.get("mode") != mode:
            continue
        if not_mode is not None and row.get("mode") == not_mode:
            continue
        selected.append(row)
    if not selected:
        return None
    return max(selected, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def write_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    best_old = _best(summary_rows, not_mode="flowstar_style")
    if best_old is None:
        baseline_ids = {
            "flowstar_style_o4_target_insert",
            "flowstar_style_o6_candidate8_output6_insert",
            "flowstar_style_o4_target_cutoff_insert",
            "flowstar_style_o6_candidate8_output6_cutoff_insert",
        }
        for baseline_path in [
            out_dir / "normalized_insertion_h10_summary.csv",
            REPO_ROOT / "outputs" / "flowstar_normalized_insertion_h10" / "normalized_insertion_h10_summary.csv",
        ]:
            if not baseline_path.exists():
                continue
            baseline_rows = [row for row in _read_csv_rows(baseline_path) if row.get("run_id") in baseline_ids]
            best_old = _best(baseline_rows)
            if best_old is not None:
                break
    best_rescue = _best(summary_rows, mode="flowstar_style")
    best_rescue_t = _finite_float(best_rescue.get("last_validated_t")) if best_rescue else 0.0
    best_old_t = _finite_float(best_old.get("last_validated_t")) if best_old else 0.0
    o6_target = next((r for r in summary_rows if r.get("run_id") == "flowstar_style_o6_target"), None)
    o6_cutoff = next((r for r in summary_rows if r.get("run_id") == "flowstar_style_o6_target_cutoff"), None)
    cutoff_rows = [r for r in summary_rows if r.get("mode") == "flowstar_style" and str(r.get("cutoff_threshold", "")) not in {"", "None"}]
    no_cutoff_rows = [r for r in summary_rows if r.get("mode") == "flowstar_style" and str(r.get("cutoff_threshold", "")) in {"", "None"}]
    best_cutoff = _best(cutoff_rows)
    best_no_cutoff = _best(no_cutoff_rows)
    cutoff_msg = "inconclusive"
    if best_cutoff and best_no_cutoff:
        ct = _finite_float(best_cutoff.get("last_validated_t")) or 0.0
        nt = _finite_float(best_no_cutoff.get("last_validated_t")) or 0.0
        if ct > nt:
            cutoff_msg = "helped"
        elif ct < nt:
            cutoff_msg = "hurt"
        else:
            cutoff_msg = "tied"
    recenter_msg = "yes" if best_rescue_t > best_old_t else "no"
    target_rows = [r for r in summary_rows if str(r.get("validation_mode", "")).startswith("target_remainder")]
    max_target_rem = _max_field(target_rows, "max_remainder_width_sum")
    target_rem_float = _finite_float(max_target_rem)
    target_bounded = target_rem_float is not None and target_rem_float <= 0.0004 + 1e-15
    reached_requested = best_rescue_t >= float(max_horizon) - 1e-9
    comparison_rows = list(comparison_rows or [])
    best_comparison = max(
        comparison_rows,
        key=lambda r: (_finite_float(r.get("py_last_validated_t")) or 0.0, -(_finite_float(r.get("tube_width_ratio")) or math.inf)),
        default=None,
    )
    width_msg = "not compared"
    tightness_msg = "needs Flow* comparison"
    if best_comparison:
        last_ratio = _finite_float(best_comparison.get("last_width_ratio"))
        tube_ratio = _finite_float(best_comparison.get("tube_width_ratio"))
        width_msg = (
            f"last width ratio=`{best_comparison.get('last_width_ratio', '')}`, "
            f"tube width ratio=`{best_comparison.get('tube_width_ratio', '')}`"
        )
        tightness_msg = "yes" if last_ratio is not None and tube_ratio is not None and last_ratio <= 1.0 and tube_ratio <= 1.0 else "needs more work"
    reachability_tightness = "both" if reached_requested and tightness_msg == "yes" else ("reachability only" if reached_requested else "neither yet")

    lines = [
        "# Flowstar-Style Rescue Report",
        "",
        f"Requested max horizon: `{float(max_horizon):.17g}`.",
        f"Best old baseline in this run: `{best_old['run_id'] if best_old else ''}` at t=`{best_old_t:.17g}`.",
        f"Best flowstar_style run: `{best_rescue['run_id'] if best_rescue else ''}` at t=`{best_rescue_t:.17g}`.",
        "",
        f"Did flowstar_style beat the old best t~={OLD_BEST_T}? {'yes' if best_rescue_t > OLD_BEST_T else 'no'}.",
        f"Did flowstar_style_o6_target reach the requested horizon? {_yes_no(bool(o6_target and (_finite_float(o6_target.get('last_validated_t')) or 0.0) >= float(max_horizon) - 1e-9))}.",
        f"Did cutoff help? {cutoff_msg}.",
        f"Did target remainder stay bounded at width sum 0.0004? {_yes_no(target_bounded)}; max target-mode remainder width sum was `{max_target_rem}`.",
        f"Did recenter/rescale help compared to range_only and dependency_preserving? {recenter_msg}; best flowstar_style t=`{best_rescue_t:.17g}` vs best baseline t=`{best_old_t:.17g}`.",
        f"Best rescue candidate: `{best_rescue['run_id'] if best_rescue else ''}`.",
        f"Accepted/rejected steps for best rescue: `{best_rescue.get('num_accepted_steps', '') if best_rescue else ''}` accepted, `{best_rescue.get('num_rejected_steps', '') if best_rescue else ''}` rejected.",
        f"min_regular_h_used for best rescue: `{best_rescue.get('min_regular_h_used', '') if best_rescue else ''}`.",
        f"Did any non-final step go below Flow* min step 0.002? {_yes_no(bool(best_rescue and int(best_rescue.get('h_below_flowstar_min_count') or 0) > 0))}.",
        f"How do widths compare to original Flow* over the same horizon? {width_msg}.",
        f"Is this a reachability success, a tightness success, or both? {reachability_tightness}.",
        f"Failure mode for the best rescue candidate: `{best_rescue.get('failure_reason', '') if best_rescue else ''}`.",
        "Do not treat this as Flow* parity unless horizon 10 is reached and boxes are compared separately.",
        "",
        "## Summary Rows",
        "",
        "| run_id | status | last_validated_t | accepted | rejected | min_h_used | min_regular_h_used | non_final_h_below_0.002 | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['run_id']} | {row['status']} | {row['last_validated_t']} | {row.get('num_accepted_steps', '')} | "
            f"{row.get('num_rejected_steps', '')} | {row['min_h_used']} | {row.get('min_regular_h_used', '')} | "
            f"{row.get('h_below_flowstar_min_count', '')} | {row['failure_reason']} |"
        )
    if o6_target or o6_cutoff:
        lines.extend(
            [
                "",
                "## Selected Configs",
                "",
                "| run_id | status | last_validated_t | runtime_s | min_regular_h_used | non_final_h_below_0.002 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in [r for r in [o6_target, o6_cutoff] if r is not None]:
            lines.append(
                f"| {row['run_id']} | {row['status']} | {row['last_validated_t']} | {row['runtime_s']} | "
                f"{row.get('min_regular_h_used', '')} | {row.get('h_below_flowstar_min_count', '')} |"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rescue_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def make_plots(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]], attempt_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except Exception:
        return

    colors = {
        "baseline_range_only_o6_s4": "#1f77b4",
        "baseline_dependency_preserving_o4_s1": "#ff7f0e",
        "flowstar_style_o4_target": "#2ca02c",
        "flowstar_style_o6_target": "#d62728",
        "flowstar_style_o4_target_cutoff": "#9467bd",
        "flowstar_style_o6_target_cutoff": "#8c564b",
        "flowstar_style_o6_target_adaptive_order_8": "#e377c2",
        "flowstar_style_o6_target_cutoff_adaptive_order_8": "#7f7f7f",
        "flowstar_style_o6_target_r2e-4": "#17becf",
        "flowstar_style_o6_target_r5e-4": "#bcbd22",
        "flowstar_style_o6_target_refined": "#1f77b4",
        "flowstar_style_o6_target_refined_cutoff": "#ff7f0e",
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in segment_rows:
        if row.get("status") == "validated":
            grouped.setdefault(str(row["run_id"]), []).append(row)

    def _plot_time(var: str, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for run_id, rows in grouped.items():
            rows = sorted(rows, key=lambda r: float(r["t_hi"]))
            t = [float(r["t_hi"]) for r in rows]
            lo = [float(r[f"{var}_lo"]) for r in rows]
            hi = [float(r[f"{var}_hi"]) for r in rows]
            color = colors.get(run_id)
            ax.plot(t, [(a + b) / 2 for a, b in zip(lo, hi)], label=run_id, color=color)
            ax.fill_between(t, lo, hi, alpha=0.16, color=color)
        ax.set_xlabel("t")
        ax.set_ylabel(var)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    _plot_time("x", out_dir / "rescue_t_x.png")
    _plot_time("y", out_dir / "rescue_t_y.png")

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    for run_id, rows in grouped.items():
        color = colors.get(run_id)
        for row in rows:
            x_lo = float(row["x_lo"])
            x_hi = float(row["x_hi"])
            y_lo = float(row["y_lo"])
            y_hi = float(row["y_hi"])
            rect = patches.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, fill=False, alpha=0.22, edgecolor=color)
            ax.add_patch(rect)
        if rows:
            ax.plot(
                [(float(r["x_lo"]) + float(r["x_hi"])) / 2 for r in rows],
                [(float(r["y_lo"]) + float(r["y_hi"])) / 2 for r in rows],
                label=run_id,
                color=color,
            )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "rescue_phase_xy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id, rows in grouped.items():
        rows = [r for r in rows if str(r.get("mode")) == "flowstar_style"]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: float(r["t_hi"]))
        ax.plot(
            [float(r["t_hi"]) for r in rows],
            [float(r["h"]) for r in rows],
            marker="o",
            markersize=2.4,
            linewidth=1.0,
            label=run_id,
            color=colors.get(run_id),
        )
    ax.axhline(FLOWSTAR_MIN_STEP, color="#111111", linewidth=0.9, linestyle="--", label="Flow* min step 0.002")
    ax.set_xlabel("t")
    ax.set_ylabel("accepted h")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "step_size_trace.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id, rows in grouped.items():
        rows = [r for r in rows if str(r.get("mode")) == "flowstar_style"]
        pts = []
        for row in sorted(rows, key=lambda r: float(r["t_hi"])):
            t_hi = _finite_float(row.get("t_hi"))
            reset_width = _finite_float(row.get("reset_width_sum"))
            if t_hi is not None and reset_width is not None:
                pts.append((t_hi, reset_width))
        if pts:
            ax.plot(
                [t for t, _width in pts],
                [width for _t, width in pts],
                marker="o",
                markersize=2.4,
                linewidth=1.0,
                label=run_id,
                color=colors.get(run_id),
            )
    ax.set_xlabel("t")
    ax.set_ylabel("reset box width sum")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "reset_box_width_trace.png", dpi=160)
    plt.close(fig)

    residual_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in attempt_rows:
        if row.get("mode") == "flowstar_style":
            residual_groups.setdefault(str(row["run_id"]), []).append(row)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    target_lines: list[float] = []
    for run_id, rows in residual_groups.items():
        pts: list[tuple[float, float]] = []
        for row in rows:
            t_hi = _finite_float(row.get("t_hi"))
            residual = _finite_float(row.get("residual_width_sum"))
            if t_hi is not None and residual is not None and residual > 0:
                pts.append((t_hi, residual))
            target = _finite_float(row.get("target_remainder_width_sum"))
            if target is not None and target > 0:
                target_lines.append(target)
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=0.8, label=run_id, color=colors.get(run_id))
    if target_lines:
        ax.axhline(max(target_lines), color="#111111", linewidth=0.9, linestyle="--", label="target remainder width sum")
    ax.set_xlabel("t")
    ax.set_ylabel("residual width sum")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_vs_t.png", dpi=160)
    plt.close(fig)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rows_for_run(rows: Sequence[Mapping[str, Any]], run_id: str) -> list[Mapping[str, Any]]:
    return sorted(
        [row for row in rows if row.get("run_id") == run_id and row.get("status") == "validated"],
        key=lambda r: _finite_float(r.get("t_hi")) or 0.0,
    )


def _overlap_rows(rows: Sequence[Mapping[str, Any]], t_lo: float, t_hi: float) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for row in rows:
        row_lo = _finite_float(row.get("t_lo"))
        row_hi = _finite_float(row.get("t_hi"))
        if row_lo is None or row_hi is None:
            continue
        if row_hi > t_lo + 1e-15 and row_lo < t_hi - 1e-15:
            out.append(row)
    return out


def _nearest_time_row(rows: Sequence[Mapping[str, Any]], t: float) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    row = min(rows, key=lambda r: abs(((_finite_float(r.get("t_lo")) or 0.0) + (_finite_float(r.get("t_hi")) or 0.0)) * 0.5 - t))
    return [row]


def _tube_width_sum(rows: Sequence[Mapping[str, Any]]) -> float | str:
    if not rows:
        return ""
    xs_lo = [_finite_float(row.get("x_lo")) for row in rows]
    xs_hi = [_finite_float(row.get("x_hi")) for row in rows]
    ys_lo = [_finite_float(row.get("y_lo")) for row in rows]
    ys_hi = [_finite_float(row.get("y_hi")) for row in rows]
    vals = [v for v in [*xs_lo, *xs_hi, *ys_lo, *ys_hi] if v is not None]
    if len(vals) != len(rows) * 4:
        return ""
    return (max(v for v in xs_hi if v is not None) - min(v for v in xs_lo if v is not None)) + (
        max(v for v in ys_hi if v is not None) - min(v for v in ys_lo if v is not None)
    )


def _safe_ratio(num: Any, den: Any) -> float | str:
    n = _finite_float(num)
    d = _finite_float(den)
    if n is None or d is None or d <= 0:
        return ""
    return n / d


def _time_overlap_ratio_rows(
    run_id: str, py_rows: Sequence[Mapping[str, Any]], flow_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for py in py_rows:
        t_lo = _finite_float(py.get("t_lo"))
        t_hi = _finite_float(py.get("t_hi"))
        if t_lo is None or t_hi is None:
            continue
        flow_overlap = _overlap_rows(flow_rows, t_lo, t_hi)
        flow_width = _tube_width_sum(flow_overlap)
        ratio = _safe_ratio(py.get("width_sum"), flow_width)
        if ratio == "":
            continue
        rows.append(
            {
                "run_id": run_id,
                "t": 0.5 * (t_lo + t_hi),
                "py_width_sum": py.get("width_sum", ""),
                "flowstar_overlap_width_sum": flow_width,
                "width_ratio": ratio,
            }
        )
    return rows


def _comparison_row(
    summary: Mapping[str, Any],
    py_rows: Sequence[Mapping[str, Any]],
    flow_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_id = str(summary["run_id"])
    if not py_rows:
        return (
            {
                "run_id": run_id,
                "py_status": summary.get("status", ""),
                "py_segments": 0,
                "py_runtime_s": summary.get("runtime_s", ""),
                "py_last_validated_t": summary.get("last_validated_t", ""),
            },
            [],
        )
    t = _finite_float(summary.get("last_validated_t")) or (_finite_float(py_rows[-1].get("t_hi")) or 0.0)
    same_horizon_flow = _overlap_rows(flow_rows, 0.0, t)
    py_last = py_rows[-1]
    py_last_t_lo = _finite_float(py_last.get("t_lo")) or 0.0
    py_last_t_hi = _finite_float(py_last.get("t_hi")) or t
    flow_last = _overlap_rows(flow_rows, py_last_t_lo, py_last_t_hi) or _nearest_time_row(flow_rows, t)
    py_tube = _tube_width_sum(py_rows)
    flow_tube = _tube_width_sum(same_horizon_flow)
    flow_last_width = _tube_width_sum(flow_last)
    ratio_rows = _time_overlap_ratio_rows(run_id, py_rows, flow_rows)
    ratios = [_finite_float(row.get("width_ratio")) for row in ratio_rows]
    ratios = [r for r in ratios if r is not None]
    return (
        {
            "run_id": run_id,
            "py_status": summary.get("status", ""),
            "py_segments": len(py_rows),
            "py_runtime_s": summary.get("runtime_s", ""),
            "py_last_validated_t": summary.get("last_validated_t", ""),
            "py_last_width_sum": py_last.get("width_sum", ""),
            "py_tube_width_sum": py_tube,
            "flowstar_segments_over_same_horizon": len(same_horizon_flow),
            "flowstar_last_width_sum_near_T": flow_last_width,
            "flowstar_tube_width_sum_over_same_horizon": flow_tube,
            "last_width_ratio": _safe_ratio(py_last.get("width_sum"), flow_last_width),
            "tube_width_ratio": _safe_ratio(py_tube, flow_tube),
            "max_time_overlap_width_ratio": max(ratios) if ratios else "",
            "median_time_overlap_width_ratio": statistics.median(ratios) if ratios else "",
        },
        ratio_rows,
    )


def _add_tx_boxes(ax: Any, rows: Sequence[Mapping[str, Any]], var: str, *, color: str, label: str, alpha: float) -> None:
    import matplotlib.patches as patches

    for i, row in enumerate(rows):
        t_lo = _finite_float(row.get("t_lo"))
        t_hi = _finite_float(row.get("t_hi"))
        v_lo = _finite_float(row.get(f"{var}_lo"))
        width = _finite_float(row.get(f"width_{var}"))
        if t_lo is None or t_hi is None or v_lo is None or width is None:
            continue
        ax.add_patch(
            patches.Rectangle(
                (t_lo, v_lo),
                t_hi - t_lo,
                width,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0.7,
                label=label if i == 0 else None,
            )
        )


def _add_phase_boxes(ax: Any, rows: Sequence[Mapping[str, Any]], *, color: str, label: str, alpha: float) -> None:
    import matplotlib.patches as patches

    for i, row in enumerate(rows):
        x_lo = _finite_float(row.get("x_lo"))
        y_lo = _finite_float(row.get("y_lo"))
        width_x = _finite_float(row.get("width_x"))
        width_y = _finite_float(row.get("width_y"))
        if x_lo is None or y_lo is None or width_x is None or width_y is None:
            continue
        ax.add_patch(
            patches.Rectangle(
                (x_lo, y_lo),
                width_x,
                width_y,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0.7,
                label=label if i == 0 else None,
            )
        )


def _set_limits_from_rows(ax: Any, rows: Sequence[Mapping[str, Any]], keys: tuple[str, str], axis: str) -> None:
    vals: list[float] = []
    for row in rows:
        lo = _finite_float(row.get(keys[0]))
        hi = _finite_float(row.get(keys[1]))
        if lo is not None and hi is not None:
            vals.extend([lo, hi])
    if not vals:
        return
    pad = max((max(vals) - min(vals)) * 0.05, 1e-6)
    if axis == "x":
        ax.set_xlim(min(vals) - pad, max(vals) + pad)
    else:
        ax.set_ylim(min(vals) - pad, max(vals) + pad)


def make_flowstar_comparison_plots(
    out_dir: Path,
    py_rows: Sequence[Mapping[str, Any]],
    flow_rows: Sequence[Mapping[str, Any]],
    ratio_rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not py_rows:
        return
    t = _finite_float(py_rows[-1].get("t_hi")) or 0.0
    flow_same = _overlap_rows(flow_rows, 0.0, t)
    for var in ("x", "y"):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        _add_tx_boxes(ax, flow_same, var, color="#2ca02c", label="Original Flow*", alpha=0.16)
        _add_tx_boxes(ax, py_rows, var, color="#1f77b4", label="PyTorch rescue", alpha=0.12)
        ax.set_xlabel("t")
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7)
        ax.set_xlim(0.0, max(t, 1e-9))
        _set_limits_from_rows(ax, [*flow_same, *py_rows], (f"{var}_lo", f"{var}_hi"), "y")
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_rescue_vs_original_flowstar_t_{var}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    _add_phase_boxes(ax, flow_same, color="#2ca02c", label="Original Flow*", alpha=0.14)
    _add_phase_boxes(ax, py_rows, color="#1f77b4", label="PyTorch rescue", alpha=0.10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    _set_limits_from_rows(ax, [*flow_same, *py_rows], ("x_lo", "x_hi"), "x")
    _set_limits_from_rows(ax, [*flow_same, *py_rows], ("y_lo", "y_hi"), "y")
    fig.tight_layout()
    fig.savefig(out_dir / "overlay_rescue_vs_original_flowstar_phase_xy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id, rows in ratio_rows_by_run.items():
        pts = [
            (_finite_float(row.get("t")), _finite_float(row.get("width_ratio")))
            for row in rows
            if _finite_float(row.get("t")) is not None and _finite_float(row.get("width_ratio")) is not None
        ]
        if not pts:
            continue
        pts.sort(key=lambda p: p[0] or 0.0)
        ax.plot([p[0] for p in pts if p[0] is not None], [p[1] for p in pts if p[1] is not None], linewidth=1.0, label=run_id)
    ax.axhline(1.0, color="#111111", linewidth=0.9, linestyle="--", label="Flow* width")
    ax.set_xlabel("t")
    ax.set_ylabel("PyTorch width / Flow* overlap hull width")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "width_ratio_vs_t.png", dpi=160)
    plt.close(fig)


def write_flowstar_comparison_report(
    out_dir: Path,
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = max(comparison_rows, key=lambda r: _finite_float(r.get("py_last_validated_t")) or 0.0, default=None)
    reached = bool(best and (_finite_float(best.get("py_last_validated_t")) or 0.0) >= float(max_horizon) - 1e-9)
    width_comparable = "needs more work"
    if best:
        last_ratio = _finite_float(best.get("last_width_ratio"))
        tube_ratio = _finite_float(best.get("tube_width_ratio"))
        if last_ratio is not None and tube_ratio is not None and last_ratio <= 1.0 and tube_ratio <= 1.0:
            width_comparable = "yes"
        elif last_ratio is not None and tube_ratio is not None:
            width_comparable = "no"
    lines = [
        "# Rescue Vs Original Flow* Comparison",
        "",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        "Original Flow* boxes are parsed GNUPLOT segment boxes; this comparison uses overlap hulls, not exact segment-count matching.",
        "This is not a Flow* parity claim unless boxes are numerically identical, which is not expected here.",
        "",
        f"Best rescue config: `{best.get('run_id', '') if best else ''}`.",
        f"Reached requested horizon? {_yes_no(reached)}.",
        f"Width comparable to Flow*? {width_comparable}.",
        "",
        "## Metrics",
        "",
        "| run_id | py_status | py_last_validated_t | py_segments | last_width_ratio | tube_width_ratio | max_overlap_ratio | median_overlap_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['run_id']} | {row['py_status']} | {row['py_last_validated_t']} | {row['py_segments']} | "
            f"{row['last_width_ratio']} | {row['tube_width_ratio']} | {row['max_time_overlap_width_ratio']} | "
            f"{row['median_time_overlap_width_ratio']} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rescue_vs_flowstar_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_rescue_vs_flowstar_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> list[dict[str, Any]]:
    if not ORIGINAL_FLOWSTAR_SEGMENTS.exists():
        return []
    flow_rows = _read_csv_rows(ORIGINAL_FLOWSTAR_SEGMENTS)
    comparison_rows: list[dict[str, Any]] = []
    ratio_rows_by_run: dict[str, list[dict[str, Any]]] = {}
    ratio_trace_rows: list[dict[str, Any]] = []
    for summary in summary_rows:
        if summary.get("mode") != "flowstar_style":
            continue
        py_rows = _rows_for_run(segment_rows, str(summary["run_id"]))
        if not py_rows:
            continue
        comparison, ratio_rows = _comparison_row(summary, py_rows, flow_rows)
        comparison_rows.append(comparison)
        ratio_rows_by_run[str(summary["run_id"])] = ratio_rows
        ratio_trace_rows.extend(ratio_rows)
    if not comparison_rows:
        return []
    _write_csv(out_dir / "rescue_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
    _write_csv(out_dir / "rescue_vs_flowstar_ratio_trace.csv", RATIO_TRACE_FIELDS, ratio_trace_rows)
    write_flowstar_comparison_report(out_dir, comparison_rows, max_horizon=max_horizon)
    best = max(comparison_rows, key=lambda r: _finite_float(r.get("py_last_validated_t")) or 0.0)
    best_py_rows = _rows_for_run(segment_rows, str(best["run_id"]))
    make_flowstar_comparison_plots(out_dir, best_py_rows, flow_rows, ratio_rows_by_run)
    return comparison_rows


def _flowstar_style_reached_requested_horizon(summary_rows: Sequence[Mapping[str, Any]], max_horizon: float) -> bool:
    for row in summary_rows:
        if row.get("mode") != "flowstar_style":
            continue
        if (_finite_float(row.get("last_validated_t")) or 0.0) >= float(max_horizon) - 1e-9:
            return True
    return False


def run_experiment(
    out_dir: Path,
    *,
    max_horizon: float,
    wall_cap_s: float,
    config_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for spec in _select_configs(config_ids):
        if spec["kind"] == "fixed":
            summary, segments, attempts = _run_fixed(spec, max_horizon=max_horizon, wall_cap_s=wall_cap_s)
        else:
            summary, segments, attempts = _run_adaptive(spec, max_horizon=max_horizon, wall_cap_s=wall_cap_s)
        summary_rows.append(summary)
        segment_rows.extend(segments)
        attempt_rows.extend(attempts)
        _write_outputs(out_dir, summary_rows, segment_rows, attempt_rows, max_horizon=max_horizon)
    _write_outputs(out_dir, summary_rows, segment_rows, attempt_rows, max_horizon=max_horizon)
    make_plots(out_dir, segment_rows, attempt_rows)
    comparison_rows: list[dict[str, Any]] = []
    if max_horizon >= 5.0:
        comparison_rows = write_rescue_vs_flowstar_outputs(
            out_dir,
            summary_rows,
            segment_rows,
            max_horizon=max_horizon,
        )
        if comparison_rows:
            write_report(out_dir, summary_rows, segment_rows, max_horizon=max_horizon, comparison_rows=comparison_rows)
    write_specialized_outputs(
        out_dir,
        summary_rows,
        segment_rows,
        attempt_rows,
        max_horizon=max_horizon,
        comparison_rows=comparison_rows,
    )
    write_rescue_next_outputs(trigger_out_dir=out_dir)
    write_rescue_next2_outputs(trigger_out_dir=out_dir)
    write_rescue_next3_outputs(trigger_out_dir=out_dir)
    write_rescue_next4_outputs(trigger_out_dir=out_dir)
    return summary_rows, segment_rows, attempt_rows



def _summary_with_h5_baseline(summary_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = list(summary_rows)
    if any(row.get("run_id") == "flowstar_style_o6_target" for row in rows):
        return rows
    baseline_path = REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_summary.csv"
    if baseline_path.exists():
        for row in _read_csv_rows(baseline_path):
            if row.get("run_id") == "flowstar_style_o6_target":
                rows.insert(0, row)
                break
    return rows


def _comparison_by_run(comparison_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("run_id", "")): row for row in comparison_rows}


def _write_adaptive_order_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best(summary_rows)
    baseline_rows = _summary_with_h5_baseline([])
    baseline = next((row for row in baseline_rows if row.get("run_id") == "flowstar_style_o6_target"), None)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    baseline_t = _finite_float(baseline.get("last_validated_t")) if baseline else 2.1095541733932355
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    comp = _comparison_by_run(comparison_rows).get(str(best.get("run_id", ""))) if best else None
    order8_count = sum(int(row.get("num_order8_steps") or 0) for row in summary_rows)
    cutoff_rows = [row for row in summary_rows if str(row.get("cutoff_threshold", "")) not in {"", "None"}]
    no_cutoff_rows = [row for row in summary_rows if str(row.get("cutoff_threshold", "")) in {"", "None"}]
    cutoff_help = "inconclusive"
    if cutoff_rows and no_cutoff_rows:
        ct = max(_finite_float(r.get("last_validated_t")) or 0.0 for r in cutoff_rows)
        nt = max(_finite_float(r.get("last_validated_t")) or 0.0 for r in no_cutoff_rows)
        cutoff_help = "yes" if ct > nt else ("no" if ct < nt else "tied")
    lines = [
        "# Adaptive Order Rescue Report",
        "",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best adaptive-order variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did adaptive order fallback beat t~=2.10955? {_yes_no(bool(best_t is not None and baseline_t is not None and best_t > baseline_t))}.",
        f"Did it reach horizon 5? {_yes_no(reached)}.",
        f"Across all configs, accepted order-8 steps in this artifact: `{order8_count}`; best-run order-8 steps=`{best.get('num_order8_steps', '') if best else ''}`.",
        "If both cutoff and no-cutoff adaptive configs are present, the aggregate count is the total across those configs, not a single-run step count.",
        f"Runtime impact: best runtime_s=`{best.get('runtime_s', '') if best else ''}` vs h5 baseline runtime_s=`{baseline.get('runtime_s', '') if baseline else ''}`.",
        f"Width vs Flow* ratio: last=`{comp.get('last_width_ratio', '') if comp else ''}`, tube=`{comp.get('tube_width_ratio', '') if comp else ''}`.",
        f"Did cutoff help? {cutoff_help}.",
        "",
        "## Rows",
        "",
        "| run_id | status | last_validated_t | order8_steps | runtime_s | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('num_order8_steps', '')} | {row.get('runtime_s', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "adaptive_order_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_remainder_sensitivity_report(out_dir: Path, rows: Sequence[Mapping[str, Any]], *, max_horizon: float) -> None:
    ordered = sorted(rows, key=lambda r: _finite_float(r.get("target_remainder_radius")) or 0.0)
    base = ordered[0] if ordered else None
    reached = [row for row in ordered if (_finite_float(row.get("last_validated_t")) or 0.0) >= float(max_horizon) - 1e-9]
    base_width = _finite_float(base.get("final_width_sum")) if base else None
    lines = [
        "# Target Remainder Sensitivity Report",
        "",
        "This is diagnostic only; larger target remainders are relaxed parameters, not Flow* parity.",
        f"Does loosening target remainder reach horizon 5? {_yes_no(bool(reached))}.",
        f"Is 2e-4 enough? {_yes_no(any(row.get('run_id') == 'flowstar_style_o6_target_r2e-4' for row in reached))}.",
        f"Is 5e-4 enough? {_yes_no(any(row.get('run_id') == 'flowstar_style_o6_target_r5e-4' for row in reached))}.",
        "",
        "## Rows",
        "",
        "| radius | run_id | status | last_validated_t | final_width_sum | width_vs_1e-4 | rejected_steps |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        width = _finite_float(row.get("final_width_sum"))
        width_ratio = width / base_width if width is not None and base_width and base_width > 0 else ""
        lines.append(
            f"| {row.get('target_remainder_radius', '')} | {row.get('run_id', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('final_width_sum', '')} | {width_ratio} | {row.get('num_rejected_steps', '')} |"
        )
    lines.extend(
        [
            "",
            "Relaxed target remainders can reduce rejections only if the validated horizon improves without unacceptable width growth.",
            "Do not recommend relaxed remainders as parity unless the report explicitly labels the parameter change.",
        ]
    )
    (out_dir / "remainder_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_refined_report(out_dir: Path, summary_rows: Sequence[Mapping[str, Any]], *, max_horizon: float) -> None:
    best = _best(summary_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    lines = [
        "# Refined Target Validation Report",
        "",
        f"Best refined variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did refined validation beat t~=2.10955? {_yes_no(bool(best_t is not None and best_t > 2.1095541733932355))}.",
        f"Did it reach horizon 5? {_yes_no(reached)}.",
        f"Runtime impact: best runtime_s=`{best.get('runtime_s', '') if best else ''}`.",
        "Residual-over-target ratios are recorded in `rescue_validation_attempts.csv` via the target and residual width fields.",
        "",
        "## Rows",
        "",
        "| run_id | status | last_validated_t | runtime_s | failure_reason |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('runtime_s', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "refined_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")



def _best_comparison_for_run(
    comparison_rows: Sequence[Mapping[str, Any]],
    run_id: str,
) -> Mapping[str, Any]:
    return _comparison_by_run(comparison_rows).get(str(run_id), {})


def _adaptive_order_baselines() -> tuple[Mapping[str, str] | None, Mapping[str, str]]:
    rows = _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "adaptive_order_summary.csv")
    comps = _comparison_by_run(
        _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "rescue_vs_flowstar_comparison.csv")
    )
    best = max(rows, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0, default=None)
    return best, comps.get(str(best.get("run_id", ""))) if best else {}


def _write_candidate_order_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best(summary_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    best_comp = _best_comparison_for_run(comparison_rows, str(best.get("run_id", ""))) if best else {}
    adaptive, adaptive_comp = _adaptive_order_baselines()
    adaptive_t = _finite_float(adaptive.get("last_validated_t")) if adaptive else 2.2771582567640953
    adaptive_runtime = adaptive.get("runtime_s", "") if adaptive else ""
    best_runtime = best.get("runtime_s", "") if best else ""
    best_tube = _finite_float(best_comp.get("tube_width_ratio"))
    adaptive_tube = _finite_float(adaptive_comp.get("tube_width_ratio"))
    if best_tube is None or adaptive_tube is None:
        width_msg = "not compared"
    elif best_tube < adaptive_tube:
        width_msg = "improved"
    elif best_tube > adaptive_tube:
        width_msg = "worsened"
    else:
        width_msg = "tied"
    best_residual = _finite_float(best.get("max_residual_width_sum")) if best else None
    adaptive_residual = _finite_float(adaptive.get("max_residual_width_sum")) if adaptive else None
    if best_residual is None or adaptive_residual is None:
        residual_msg = "inconclusive"
    elif best_residual < adaptive_residual:
        residual_msg = "yes by max residual width sum"
    elif best_residual > adaptive_residual:
        residual_msg = "no; max residual width sum increased"
    else:
        residual_msg = "tied by max residual width sum"
    lines = [
        "# Candidate Order Diagnostic Report",
        "",
        "Candidate-order mode validates with a higher Picard polynomial order and truncates the accepted/output Taylor model back to output order with the dropped contribution added to interval uncertainty.",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best candidate-order variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did candidate_order=8/output_order=6 beat t~=2.277? {_yes_no(bool(best_t is not None and adaptive_t is not None and best_t > adaptive_t))}.",
        f"Did it reach horizon 5? {_yes_no(reached)}.",
        f"Width ratio vs adaptive full-order-8 fallback: {width_msg}; candidate last=`{best_comp.get('last_width_ratio', '')}`, tube=`{best_comp.get('tube_width_ratio', '')}`, adaptive tube=`{adaptive_comp.get('tube_width_ratio', '')}`.",
        f"Runtime impact: best runtime_s=`{best_runtime}` vs adaptive fallback runtime_s=`{adaptive_runtime}`.",
        f"Does it reduce truncation containment miss? {residual_msg}; candidate max_residual_width_sum=`{best.get('max_residual_width_sum', '') if best else ''}`, adaptive=`{adaptive.get('max_residual_width_sum', '') if adaptive else ''}`.",
        "",
        "## Rows",
        "",
        "| run_id | status | candidate_order | output_order | last_validated_t | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    comp_by_run = _comparison_by_run(comparison_rows)
    for row in summary_rows:
        comp = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('candidate_order', '')} | "
            f"{row.get('output_order', '')} | {row.get('last_validated_t', '')} | {row.get('runtime_s', '')} | "
            f"{comp.get('last_width_ratio', '')} | {comp.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "candidate_order_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_truncation_range_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best(summary_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    comp_by_run = _comparison_by_run(comparison_rows)
    best_comp = comp_by_run.get(str(best.get("run_id", ""))) if best else {}
    cutoff_split_rows = [row for row in summary_rows if "cutoff" in str(row.get("run_id", "")) and "truncsplit" in str(row.get("run_id", ""))]
    cutoff_msg = "not evaluated by the requested truncation-range config set" if not cutoff_split_rows else "see rows"
    lines = [
        "# Truncation Range Diagnostic Report",
        "",
        "Dropped/truncated polynomial terms are still bounded conservatively; this diagnostic only changes how their interval range is evaluated.",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best truncation-range variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Does tighter dropped-term range bounding beat t~=2.277? {_yes_no(bool(best_t is not None and best_t > 2.2771582567640953))}.",
        f"Does it reach horizon 5? {_yes_no(reached)}.",
        f"Runtime cost for best variant: runtime_s=`{best.get('runtime_s', '') if best else ''}`.",
        f"Width ratio vs Flow*: last=`{best_comp.get('last_width_ratio', '') if best_comp else ''}`, tube=`{best_comp.get('tube_width_ratio', '') if best_comp else ''}`.",
        f"Did cutoff help when combined with truncsplit? {cutoff_msg}.",
        "",
        "## Rows",
        "",
        "| run_id | status | split | candidate_order | output_order | last_validated_t | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        comp = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('truncation_range_split', '')} | "
            f"{row.get('candidate_order', '')} | {row.get('output_order', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('runtime_s', '')} | {comp.get('last_width_ratio', '')} | {comp.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "truncation_range_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")



def _retained_term_rows(segment_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segment_rows:
        details = segment.get("_selective_term_details") or []
        for detail in details:
            row = {
                "run_id": segment.get("run_id", ""),
                "segment_index": segment.get("segment_index", ""),
                "t_lo": segment.get("t_lo", ""),
                "t_hi": segment.get("t_hi", ""),
                "status": segment.get("status", ""),
                "selective_high_degree_terms_top_k": segment.get("selective_high_degree_terms_top_k", ""),
            }
            row.update(dict(detail))
            rows.append(row)
    rows.sort(
        key=lambda row: (
            _finite_float(row.get("t_lo")) or 0.0,
            _finite_float(row.get("segment_index")) or 0.0,
            0 if _truthy(row.get("retained")) else 1,
            _finite_float(row.get("term_rank")) or 0.0,
        )
    )
    if not rows:
        return []
    near_t = max(_finite_float(row.get("t_lo")) or 0.0 for row in rows)
    near_rows = [row for row in rows if (_finite_float(row.get("t_lo")) or 0.0) >= near_t - 0.25]
    return near_rows or rows


def _write_residual_centering_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best(summary_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    comp = _best_comparison_for_run(comparison_rows, str(best.get("run_id", ""))) if best else {}
    corrections = sum(int(row.get("center_corrections_applied") or 0) for row in summary_rows)
    corrected_dims = sum(int(row.get("center_corrected_dimensions") or 0) for row in summary_rows)
    max_corr = _max_field(summary_rows, "max_center_correction_abs")
    target_radii = {str(row.get("target_remainder_radius", "")) for row in summary_rows}
    target_stayed = target_radii <= {"0.0001", "0.000100000000000000", "1e-04", "1e-4", "0.00010000000000000000"}
    below_min = any(int(row.get("h_below_flowstar_min_count") or 0) > 0 for row in summary_rows)
    after_subset = sum(1 for row in attempt_rows if _truthy(row.get("center_correction_applied")) and _truthy(row.get("subset_after_correction")))
    lines = [
        "# Residual Centering Diagnostic Report",
        "",
        "This opt-in mode keeps the symmetric target remainder and accepts only after recomputing the Picard residual from the corrected candidate.",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best centered variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did centered validation beat t~=2.400737? {_yes_no(bool(best_t is not None and best_t > 2.400737667399793))}.",
        f"Did it reach horizon 5? {_yes_no(reached)}.",
        f"Center-correction attempts: `{corrections}` attempts, `{corrected_dims}` corrected dimensions; subset-after-correction rows=`{after_subset}`.",
        f"Did corrections stay small? max_abs_correction=`{max_corr}`.",
        f"Width ratio vs Flow*: last=`{comp.get('last_width_ratio', '')}`, tube=`{comp.get('tube_width_ratio', '')}`.",
        f"Did target remainder remain at 1e-4? {_yes_no(target_stayed)}.",
        f"Any non-final h below 0.002? {_yes_no(below_min)}.",
        "",
        "## Rows",
        "",
        "| run_id | status | last_validated_t | corrections | corrected_dims | max_abs_correction | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    comp_by_run = _comparison_by_run(comparison_rows)
    for row in summary_rows:
        comp_row = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('center_corrections_applied', '')} | {row.get('center_corrected_dimensions', '')} | "
            f"{row.get('max_center_correction_abs', '')} | {comp_row.get('last_width_ratio', '')} | "
            f"{comp_row.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "residual_centering_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_selective_terms_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    raw_best = _best(summary_rows)
    raw_best_t = _finite_float(raw_best.get("last_validated_t")) if raw_best else 0.0
    tied_best_rows = [
        row
        for row in summary_rows
        if raw_best_t is not None
        and (row_t := _finite_float(row.get("last_validated_t"))) is not None
        and abs(row_t - raw_best_t) <= 1e-12
    ]

    def _drop_width_key(row: Mapping[str, Any]) -> tuple[bool, float]:
        width = _finite_float(row.get("max_selective_dropped_remainder_width_sum"))
        return (width is None, width if width is not None else math.inf)

    best = min(tied_best_rows, key=_drop_width_key, default=raw_best)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    comp = _best_comparison_for_run(comparison_rows, str(best.get("run_id", ""))) if best else {}
    best_k = best.get("selective_high_degree_terms_top_k", "") if best else ""
    if len(tied_best_rows) > 1:
        best_k_summary = f"all tested K values tied on validated time; K=`{best_k}` minimized dropped remainder width"
    else:
        best_k_summary = f"`{best_k}`"
    adaptive, adaptive_comp = _adaptive_order_baselines()
    adaptive_t = _finite_float(adaptive.get("last_validated_t")) if adaptive else 2.2771582567640953
    residual_centers = _max_abs_fields(attempt_rows, ["residual_before_center_x", "residual_before_center_y", "residual_after_center_x", "residual_after_center_y"])
    lines = [
        "# Selective High-Degree Term Diagnostic Report",
        "",
        "This is diagnostic-only: sparse over-order terms are retained beyond output_order=6, so this is not fixed-order Flow* parity.",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best selective variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did selective retention beat t~=2.400737? {_yes_no(bool(best_t is not None and best_t > 2.400737667399793))}.",
        f"Did any variant reach horizon 5? {_yes_no(reached)}.",
        f"Which K worked best? {best_k_summary}.",
        f"Did keeping a few terms reduce residual shift? max recorded residual center magnitude=`{residual_centers}` (compare by row in attempts CSV).",
        f"Runtime impact: best runtime_s=`{best.get('runtime_s', '') if best else ''}`.",
        f"Width ratio vs Flow*: last=`{comp.get('last_width_ratio', '')}`, tube=`{comp.get('tube_width_ratio', '')}`.",
        f"Did this outperform full adaptive order fallback? {_yes_no(bool(best_t is not None and adaptive_t is not None and best_t > adaptive_t))}; adaptive tube=`{adaptive_comp.get('tube_width_ratio', '')}`.",
        "",
        "## Rows",
        "",
        "| run_id | K | status | last_validated_t | retained_terms | dropped_remainder_width | runtime_s | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    comp_by_run = _comparison_by_run(comparison_rows)
    for row in summary_rows:
        comp_row = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('selective_high_degree_terms_top_k', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('max_selective_retained_terms_count', '')} | "
            f"{row.get('max_selective_dropped_remainder_width_sum', '')} | {row.get('runtime_s', '')} | "
            f"{comp_row.get('last_width_ratio', '')} | {comp_row.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "selective_terms_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

def _last_failed_attempt(attempt_rows: Sequence[Mapping[str, Any]], run_id: str | None = None) -> Mapping[str, Any]:
    rows = [row for row in attempt_rows if row.get("validation_status") == "failed"]
    if run_id is not None:
        rows = [row for row in rows if row.get("run_id") == run_id]
    return rows[-1] if rows else {}


def _failure_dimension_from_attempt(row: Mapping[str, Any], prefix: str = "tmp_remainder") -> str:
    target_width = (_finite_float(row.get("target_remainder_width_sum")) or 0.0004) / 2.0
    scores: dict[str, float] = {}
    for dim in ("x", "y"):
        lo = _finite_float(row.get(f"{prefix}_lo_{dim}"))
        hi = _finite_float(row.get(f"{prefix}_hi_{dim}"))
        if lo is None or hi is None:
            lo = _finite_float(row.get(f"residual_lo_{dim}"))
            hi = _finite_float(row.get(f"residual_hi_{dim}"))
        if lo is None or hi is None:
            continue
        scores[dim] = max(abs(lo), abs(hi)) - target_width * 0.5
    return max(scores, key=scores.get, default="")


def _shift_or_width_from_attempt(row: Mapping[str, Any], prefix: str = "tmp_remainder") -> str:
    dim = _failure_dimension_from_attempt(row, prefix=prefix)
    if not dim:
        return "unknown"
    width = _finite_float(row.get(f"{prefix}_width_{dim}"))
    center = _finite_float(row.get(f"{prefix}_center_{dim}"))
    target_width = (_finite_float(row.get("target_remainder_width_sum")) or 0.0004) / 2.0
    if width is not None and center is not None and width <= target_width * 1.05 and abs(center) > max(width * 0.05, 1e-12):
        return "shift"
    if width is not None and width > target_width * 1.05:
        return "width"
    return "shift" if center is not None and abs(center) > 1e-12 else "unknown"


def _max_abs_center(rows: Sequence[Mapping[str, Any]], prefix: str) -> float | None:
    vals: list[float] = []
    for row in rows:
        for dim in ("x", "y"):
            val = _finite_float(row.get(f"{prefix}_center_{dim}"))
            if val is not None:
                vals.append(abs(val))
    return max(vals) if vals else None


def _write_ctrunc_validation_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best(summary_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    failed = _last_failed_attempt(attempt_rows, str(best.get("run_id", "")) if best else None)
    failure_dim = _failure_dimension_from_attempt(failed)
    failure_kind = _shift_or_width_from_attempt(failed)
    ordinary_shift = _max_abs_center(attempt_rows, "ordinary_residual_range")
    normal_shift = _max_abs_center(attempt_rows, "normal_eval_range")
    normal_reduced = ordinary_shift is not None and normal_shift is not None and normal_shift < ordinary_shift
    comp = _best_comparison_for_run(comparison_rows, str(best.get("run_id", ""))) if best else {}
    lines = [
        "# Flowstar Ctrunc Validation Report",
        "",
        "This opt-in mode uses a clean-room Flow*-style Picard ctrunc validation decision. It does not replace the default target-remainder validator.",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        f"Best ctrunc variant: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did flowstar_ctrunc validation beat t~=2.400737? {_yes_no(bool(best_t is not None and best_t > 2.400737667399793))}.",
        f"Did it reach horizon 5? {_yes_no(reached)}.",
        f"Which dimension still fails? `{failure_dim}`.",
        f"Is the failure still shift or width? `{failure_kind}`.",
        f"Does normal eval reduce the residual shift? {_yes_no(normal_reduced)}; ordinary max center=`{ordinary_shift if ordinary_shift is not None else ''}`, normal max center=`{normal_shift if normal_shift is not None else ''}`.",
        f"Runtime impact: best runtime_s=`{best.get('runtime_s', '') if best else ''}`.",
        f"Width ratio vs Flow*: last=`{comp.get('last_width_ratio', '')}`, tube=`{comp.get('tube_width_ratio', '')}`.",
        "",
        "## Rows",
        "",
        "| run_id | status | last_validated_t | runtime_s | tmp_subset_fail_dim | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    comp_by_run = _comparison_by_run(comparison_rows)
    for row in summary_rows:
        comp_row = comp_by_run.get(str(row.get("run_id", "")), {})
        failed_row = _last_failed_attempt(attempt_rows, str(row.get("run_id", "")))
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('runtime_s', '')} | {_failure_dimension_from_attempt(failed_row)} | "
            f"{comp_row.get('last_width_ratio', '')} | {comp_row.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "ctrunc_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _selective_validation_path_term_rows(attempt_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stages = [
        ("before_validation", "candidate_terms_before_validation"),
        ("after_selective", "candidate_terms_after_selective"),
        ("inside_validation", "validation_candidate_inside"),
        ("after_internal", "validation_candidate_after_internal"),
    ]
    for attempt in attempt_rows:
        if not attempt.get("selective_high_degree_terms_top_k"):
            continue
        for stage, prefix in stages:
            rows.append(
                {
                    "run_id": attempt.get("run_id", ""),
                    "segment_index": attempt.get("segment_index", ""),
                    "attempt_index": attempt.get("attempt_index", ""),
                    "stage": stage,
                    "terms_hash": attempt.get(f"{prefix}_terms_hash", ""),
                    "term_count": attempt.get(f"{prefix}_term_count", ""),
                    "max_degree": attempt.get(f"{prefix}_max_degree", ""),
                    "high_degree_term_count": attempt.get(f"{prefix}_high_degree_term_count", ""),
                    "validation_status": attempt.get("validation_status", ""),
                    "subset_result": attempt.get("subset_result", ""),
                    "rejection_reason": attempt.get("rejection_reason", ""),
                }
            )
    return rows


def _write_selective_validation_path_audit(out_dir: Path, attempt_rows: Sequence[Mapping[str, Any]]) -> None:
    term_rows = _selective_validation_path_term_rows(attempt_rows)
    _write_csv(out_dir / "validation_path_terms.csv", VALIDATION_PATH_TERM_FIELDS, term_rows)
    inside_rows = [row for row in term_rows if row.get("stage") == "inside_validation"]
    after_rows = [row for row in term_rows if row.get("stage") == "after_selective"]
    internal_rows = [row for row in term_rows if row.get("stage") == "after_internal"]
    inside_high = max((_finite_float(row.get("high_degree_term_count")) or 0.0 for row in inside_rows), default=0.0)
    after_high = max((_finite_float(row.get("high_degree_term_count")) or 0.0 for row in after_rows), default=0.0)
    internal_high = max((_finite_float(row.get("high_degree_term_count")) or 0.0 for row in internal_rows), default=0.0)
    hashes_match = all(
        a.get("terms_hash") == b.get("terms_hash")
        for a, b in zip(after_rows, inside_rows)
        if a.get("run_id") == b.get("run_id") and a.get("segment_index") == b.get("segment_index")
    )
    internal_hashes_match = all(
        a.get("terms_hash") == b.get("terms_hash")
        for a, b in zip(inside_rows, internal_rows)
        if a.get("run_id") == b.get("run_id") and a.get("segment_index") == b.get("segment_index")
    )
    present = inside_high > 0
    if present:
        conclusion = "retained degree >6 monomials are present during residual validation"
    elif after_high > 0:
        conclusion = "retained degree >6 monomials are created after selective construction but are missing inside validation"
    else:
        conclusion = "no retained degree >6 monomials were observed in the audited attempts"
    lines = [
        "# Selective Validation Path Audit",
        "",
        f"Conclusion: {conclusion}.",
        f"After-selective max high-degree term count: `{after_high}`.",
        f"Inside-validation max high-degree term count: `{inside_high}`.",
        f"After-internal max high-degree term count: `{internal_high}`.",
        f"Do after-selective and inside-validation term hashes match where comparable? {_yes_no(bool(hashes_match))}.",
        f"Do inside-validation and after-internal term hashes match where comparable? {_yes_no(bool(internal_hashes_match))}.",
        "",
        "The audit hashes the candidate polynomial before selective retention, after selective retention, inside the Picard residual validator, and after internal validation Taylor-model operations.",
        "",
        "## Stage Counts",
        "",
        "| stage | rows | max_high_degree_terms |",
        "| --- | ---: | ---: |",
        f"| after_selective | {len(after_rows)} | {after_high} |",
        f"| inside_validation | {len(inside_rows)} | {inside_high} |",
        f"| after_internal | {len(internal_rows)} | {internal_high} |",
    ]
    (out_dir / "validation_path_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")



def _row_by_run(rows: Sequence[Mapping[str, Any]], run_id: str) -> Mapping[str, Any]:
    return next((row for row in rows if row.get("run_id") == run_id), {})


def _write_width_control_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    previous_id = "flowstar_style_o6_candidate8_output6_cutoff"
    new_id = "flowstar_style_o6_candidate8_output6_cutoff_symqueue"
    previous = _row_by_run(summary_rows, previous_id)
    new = _row_by_run(summary_rows, new_id)
    previous_t = _finite_float(previous.get("last_validated_t")) or 0.0
    new_t = _finite_float(new.get("last_validated_t")) or 0.0
    reached = bool(new and new_t >= float(max_horizon) - 1e-9)
    comp_previous = _best_comparison_for_run(comparison_rows, previous_id)
    comp_new = _best_comparison_for_run(comparison_rows, new_id)
    prev_ratio = _finite_float(comp_previous.get("tube_width_ratio"))
    new_ratio = _finite_float(comp_new.get("tube_width_ratio"))
    ratio_improved = prev_ratio is not None and new_ratio is not None and new_ratio < prev_ratio
    previous_reset = max(
        (_finite_float(row.get("reset_width_sum")) or 0.0 for row in segment_rows if row.get("run_id") == previous_id),
        default=0.0,
    )
    new_reset = max(
        (_finite_float(row.get("reset_width_sum")) or 0.0 for row in segment_rows if row.get("run_id") == new_id),
        default=0.0,
    )
    reset_shrank = bool(previous_reset and new_reset and new_reset < previous_reset)
    queue_peak = _max_field([row for row in segment_rows if row.get("run_id") == new_id], "flowstar_queue_size_after")
    propagated_peak = _max_field([row for row in segment_rows if row.get("run_id") == new_id], "flowstar_propagated_remainder_width_sum")
    if not new:
        branch_decision = "DISCARD_BRANCH"
        recommendation = "The symbolic-queue config did not run; fix reproducibility before using this branch."
    elif reached or new_t > previous_t + 1e-12:
        branch_decision = "MERGE_CANDIDATE"
        recommendation = "Keep the opt-in queue path and run a longer follow-up after reviewing oracle evidence."
    else:
        branch_decision = "NEEDS_MORE_WORK"
        recommendation = "The queue is tighter over its short horizon but fails much earlier; implement normalized insertion/composition next."

    lines = [
        "# Flowstar Width-Control Rescue Report",
        "",
        "Chosen mechanism: Flow*-style symbolic remainder queue skeleton (`J`, `Phi_L`, `scalars`) because the original Van der Pol benchmark calls `ode.reach(..., sr)` with a symbolic queue of size 100.",
        f"Previous best `{previous_id}` reached t=`{previous_t:.17g}`.",
        f"New width-control `{new_id}` reached t=`{new_t:.17g}`.",
        f"Did the new width-control beat t~=2.400737? {_yes_no(new_t > 2.400737667399793)}.",
        f"Did it reach horizon {float(max_horizon):.17g}? {_yes_no(reached)}.",
        f"Runtime cost: previous=`{previous.get('runtime_s', '')}`, new=`{new.get('runtime_s', '')}` seconds.",
        f"Width ratio vs Flow*: previous tube=`{comp_previous.get('tube_width_ratio', '')}`, new tube=`{comp_new.get('tube_width_ratio', '')}`.",
        f"Did width ratio improve over the validated same-run horizon? {_yes_no(ratio_improved)} (not comparable as a success if the new run stops much earlier).",
        f"Did reset box width shrink vs previous best? {_yes_no(reset_shrank)}; previous max reset width sum=`{previous_reset:.17g}`, new=`{new_reset:.17g}`.",
        f"Queue peak size after accepted steps: `{queue_peak}`; propagated remainder peak width sum: `{propagated_peak}`.",
        "Did the local one-step oracle become easier? See `outputs/flowstar_one_step_oracle_after_width_control/oracle_after_width_control_report.md` when that rerun is available.",
        f"Failure mode if still failing: `{new.get('failure_reason', '')}`.",
        f"Branch decision: {branch_decision}.",
        f"Next recommendation: {recommendation}",
        "",
        "## Rows",
        "",
        "| run_id | reset_mode | status | last_validated_t | runtime_s | max_queue_after | max_propagated_width | failure_reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('reset_mode', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('runtime_s', '')} | {row.get('max_flowstar_queue_size_after', '')} | "
            f"{row.get('max_flowstar_propagated_remainder_width_sum', '')} | {row.get('failure_reason', '')} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "width_control_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    decision_lines = [
        "# Branch Decision",
        "",
        f"Decision: {branch_decision}",
        "",
        "## Evidence",
        "",
        f"- Previous best `{previous_id}` reached t=`{previous_t:.17g}`.",
        f"- New width-control `{new_id}` reached t=`{new_t:.17g}`.",
        f"- Horizon {float(max_horizon):.17g} reached: {_yes_no(reached)}.",
        f"- Width ratio improved over the validated same-run horizon: {_yes_no(ratio_improved)}.",
        f"- Reset boxes shrank against the previous best: {_yes_no(reset_shrank)}.",
        f"- Failure mode: `{new.get('failure_reason', '')}`.",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
    ]
    (out_dir / "branch_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")


def _normalized_insertion_reset_rows(segment_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in segment_rows:
        reset_mode = str(row.get("reset_mode", ""))
        run_id = str(row.get("run_id", ""))
        if reset_mode not in {"normalized_insertion", "normalized_insertion_symqueue", "normalized_insertion_symqueue_split", "normalized_insertion_symqueue_v2"} and "cutoff_insert" not in run_id:
            continue
        rows.append({field: row.get(field, "") for field in NORMALIZED_INSERTION_RESET_FIELDS})
    return rows


def _rows_for_run_any_status(rows: Sequence[Mapping[str, Any]], run_id: str) -> list[Mapping[str, Any]]:
    return sorted([row for row in rows if row.get("run_id") == run_id], key=lambda r: _finite_float(r.get("t_hi")) or 0.0)


def _max_before_time(rows: Sequence[Mapping[str, Any]], run_id: str, field: str, t_limit: float) -> float | None:
    vals: list[float] = []
    for row in rows:
        if row.get("run_id") != run_id:
            continue
        t_hi = _finite_float(row.get("t_hi"))
        val = _finite_float(row.get(field))
        if t_hi is not None and t_hi <= t_limit + 1e-12 and val is not None:
            vals.append(val)
    return max(vals) if vals else None


def _first_ratio_crossing(ratio_rows: Sequence[Mapping[str, Any]], run_id: str, threshold: float) -> float | None:
    candidates: list[float] = []
    for row in ratio_rows:
        if row.get("run_id") != run_id:
            continue
        ratio = _finite_float(row.get("width_ratio"))
        t = _finite_float(row.get("t"))
        if ratio is not None and t is not None and ratio >= threshold:
            candidates.append(t)
    return min(candidates) if candidates else None


def _ratio_crossing_lines(ratio_rows: Sequence[Mapping[str, Any]], previous_id: str, new_id: str) -> list[str]:
    lines: list[str] = []
    for threshold in (2.0, 5.0, 10.0):
        prev_t = _first_ratio_crossing(ratio_rows, previous_id, threshold)
        new_t = _first_ratio_crossing(ratio_rows, new_id, threshold)
        if prev_t is None and new_t is None:
            verdict = "tied; neither crossed"
        elif prev_t is None:
            verdict = "worse; new crossed but previous did not"
        elif new_t is None:
            verdict = "improved; new did not cross"
        elif new_t > prev_t + 1e-12:
            verdict = "improved; crossing moved later"
        elif new_t < prev_t - 1e-12:
            verdict = "worse; crossing moved earlier"
        else:
            verdict = "tied"
        lines.append(
            f"- {threshold:.0f}x crossing: previous=`{prev_t if prev_t is not None else ''}`, "
            f"normalized_insertion=`{new_t if new_t is not None else ''}`; {verdict}."
        )
    return lines


def make_normalized_insertion_plots(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    previous_id = "flowstar_style_o6_candidate8_output6_cutoff"
    new_id = "flowstar_style_o6_candidate8_output6_cutoff_insert"
    grouped = {
        previous_id: _rows_for_run_any_status(segment_rows, previous_id),
        new_id: _rows_for_run_any_status(segment_rows, new_id),
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id, rows in grouped.items():
        pts = [
            (_finite_float(row.get("t_hi")), _finite_float(row.get("reset_width_sum")))
            for row in rows
            if _finite_float(row.get("t_hi")) is not None and _finite_float(row.get("reset_width_sum")) is not None
        ]
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=2.4, linewidth=1.0, label=run_id)
    ax.set_xlabel("t")
    ax.set_ylabel("reset width sum")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "reset_width_compare.png", dpi=160)
    plt.close(fig)

    rows = _rows_for_run_any_status(segment_rows, new_id)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    pts_trunc: list[tuple[float, float]] = []
    pts_cutoff: list[tuple[float, float]] = []
    pts_rem: list[tuple[float, float]] = []
    for row in rows:
        t = _finite_float(row.get("t_hi"))
        if t is None:
            continue
        trunc = _finite_float(row.get("insertion_truncation_width"))
        cutoff = _finite_float(row.get("insertion_cutoff_width"))
        rem = _finite_float(row.get("output_remainder_width"))
        if trunc is not None:
            pts_trunc.append((t, trunc))
        if cutoff is not None:
            pts_cutoff.append((t, cutoff))
        if rem is not None:
            pts_rem.append((t, rem))
    for label, pts in (("truncation", pts_trunc), ("cutoff", pts_cutoff), ("output remainder", pts_rem)):
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=2.4, linewidth=1.0, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("uncertainty width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "insertion_uncertainty_vs_t.png", dpi=160)
    plt.close(fig)


def _write_normalized_insertion_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    previous_id = "flowstar_style_o6_candidate8_output6_cutoff"
    new_id = "flowstar_style_o6_candidate8_output6_cutoff_insert"
    previous = _row_by_run(summary_rows, previous_id)
    new = _row_by_run(summary_rows, new_id)
    previous_t = _finite_float(previous.get("last_validated_t")) or 0.0
    new_t = _finite_float(new.get("last_validated_t")) or 0.0
    reached = bool(new and new_t >= float(max_horizon) - 1e-9)
    beat_old = new_t > 2.400737667399793
    prev_reset = _max_before_time(segment_rows, previous_id, "reset_width_sum", 2.4)
    new_reset = _max_before_time(segment_rows, new_id, "reset_width_sum", 2.4)
    reset_shrank = prev_reset is not None and new_reset is not None and new_reset < prev_reset
    insertion_unc = _max_field([row for row in segment_rows if row.get("run_id") == new_id], "output_remainder_width")
    inserted_width = _max_field([row for row in segment_rows if row.get("run_id") == new_id], "inserted_endpoint_width_sum")
    unc_float = _finite_float(insertion_unc)
    inserted_float = _finite_float(inserted_width)
    uncertainty_dominated = bool(unc_float is not None and inserted_float is not None and unc_float >= inserted_float)
    comp_previous = _best_comparison_for_run(comparison_rows, previous_id)
    comp_new = _best_comparison_for_run(comparison_rows, new_id)
    ratio_rows = _read_optional_csv(out_dir / "rescue_vs_flowstar_ratio_trace.csv")
    ratio_lines = _ratio_crossing_lines(ratio_rows, previous_id, new_id) if ratio_rows else [
        "- Crossing data unavailable because `rescue_vs_flowstar_ratio_trace.csv` was not produced."
    ]
    if (float(max_horizon) >= 5.0 and reached) or new_t > previous_t + 1e-12:
        branch_decision = "MERGE_CANDIDATE"
    else:
        branch_decision = "NEEDS_MORE_WORK"

    lines = [
        "# Flowstar Normalized Insertion Rescue Report",
        "",
        "Mechanism: opt-in clean-room normal insertion/composition. The default flowpipe path is unchanged.",
        f"Previous best `{previous_id}` reached t=`{previous_t:.17g}`.",
        f"Normalized insertion `{new_id}` reached t=`{new_t:.17g}`.",
        f"Did normalized insertion beat t~=2.400737? {_yes_no(beat_old)}.",
        f"Did it reach horizon {float(max_horizon):.17g}? {_yes_no(reached)}.",
        f"Did reset widths shrink before t~=2.4? {_yes_no(reset_shrank)}; previous max reset width sum=`{prev_reset if prev_reset is not None else ''}`, new=`{new_reset if new_reset is not None else ''}`.",
        "Did width ratios vs Flow* improve at 2x/5x/10x crossing times?",
        *ratio_lines,
        f"Did insertion uncertainty dominate? {_yes_no(uncertainty_dominated)}; max output remainder width=`{insertion_unc}`, max inserted endpoint width=`{inserted_width}`.",
        f"Runtime cost: previous=`{previous.get('runtime_s', '')}`, normalized insertion=`{new.get('runtime_s', '')}` seconds.",
        f"Width ratio vs Flow*: previous last=`{comp_previous.get('last_width_ratio', '')}`, tube=`{comp_previous.get('tube_width_ratio', '')}`; new last=`{comp_new.get('last_width_ratio', '')}`, tube=`{comp_new.get('tube_width_ratio', '')}`.",
        f"Failure mode if still failing: `{new.get('failure_reason', '')}`.",
        f"One-step oracle after insertion: {'not run; normalized insertion reached the requested horizon and produced no PyTorch failure point' if reached else 'run at the new failure point if requested by the driver'}.",
        f"Branch decision: {branch_decision}.",
        "",
        "## Rows",
        "",
        "| run_id | reset_mode | status | last_validated_t | runtime_s | max_inserted_width | max_insertion_truncation | max_insertion_cutoff | failure_reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('reset_mode', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('runtime_s', '')} | {row.get('max_inserted_endpoint_width_sum', '')} | "
            f"{row.get('max_insertion_truncation_width', '')} | {row.get('max_insertion_cutoff_width', '')} | {row.get('failure_reason', '')} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "normalized_insertion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")



H10_OUTPUT_DIR_NAME = "flowstar_normalized_insertion_h10"
NORMAL_EVAL_H10_OUTPUT_DIR_NAME = "flowstar_normal_eval_h10"
SYMQUEUE_H10_OUTPUT_DIR_NAME = "flowstar_normalized_insertion_symqueue_h10"
HORNER_H10_OUTPUT_DIR_NAME = "flowstar_horner_insertion_h10"
SYMQUEUE_SPLIT_H10_OUTPUT_DIR_NAME = "flowstar_normalized_insertion_symqueue_split_h10"
SYMQUEUE_V2_H10_OUTPUT_DIR_NAME = "flowstar_normalized_insertion_symqueue_v2_h10"
LEGACY_SYMQUEUE_V2_H10_OUTPUT_DIR_NAME = "flowstar_symbolic_queue_v2_h10"
H10_CONFIG_IDS = [
    "flowstar_style_o6_candidate8_output6_cutoff_insert",
    "flowstar_style_o6_candidate8_output6_insert",
    "flowstar_style_o4_target_cutoff_insert",
    "flowstar_style_o4_target_insert",
]
SYMQUEUE_H10_CONFIG_IDS = [
    "flowstar_style_o4_target_insert_symqueue",
    "flowstar_style_o4_target_cutoff_insert_symqueue",
    "flowstar_style_o6_candidate8_output6_insert_symqueue",
    "flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue",
]
SYMQUEUE_SPLIT_H10_CONFIG_IDS = [
    "flowstar_style_o4_target_insert_symqueue_split",
    "flowstar_style_o4_target_cutoff_insert_symqueue_split",
    "flowstar_style_o6_candidate8_output6_insert_symqueue_split",
    "flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue_split",
]
SYMQUEUE_V2_H10_CONFIG_IDS = [
    "flowstar_style_o4_target_insert_symqueue_v2",
    "flowstar_style_o4_target_cutoff_insert_symqueue_v2",
    "flowstar_style_o6_candidate8_output6_insert_symqueue_v2",
    "flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue_v2",
]


def _copy_plot_if_present(out_dir: Path, src_name: str, dst_name: str) -> None:
    src = out_dir / src_name
    dst = out_dir / dst_name
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _ordered_h10_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_id = {str(row.get("run_id", "")): row for row in summary_rows}
    ordered = [by_id[run_id] for run_id in H10_CONFIG_IDS if run_id in by_id]
    ordered.extend(row for row in summary_rows if row not in ordered)
    return ordered


def _ordered_symqueue_h10_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_id = {str(row.get("run_id", "")): row for row in summary_rows}
    ordered = [by_id[run_id] for run_id in SYMQUEUE_H10_CONFIG_IDS if run_id in by_id]
    ordered.extend(row for row in summary_rows if row not in ordered)
    return ordered


def _ordered_symqueue_split_h10_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_id = {str(row.get("run_id", "")): row for row in summary_rows}
    ordered = [by_id[run_id] for run_id in SYMQUEUE_SPLIT_H10_CONFIG_IDS if run_id in by_id]
    ordered.extend(row for row in summary_rows if row not in ordered)
    return ordered


def _ordered_symqueue_v2_h10_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_id = {str(row.get("run_id", "")): row for row in summary_rows}
    ordered = [by_id[run_id] for run_id in SYMQUEUE_V2_H10_CONFIG_IDS if run_id in by_id]
    ordered.extend(row for row in summary_rows if row not in ordered)
    return ordered


def _comparison_by_run(comparison_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("run_id", "")): row for row in comparison_rows}


def _reached_requested(row: Mapping[str, Any], max_horizon: float) -> bool:
    return (_finite_float(row.get("last_validated_t")) or 0.0) >= float(max_horizon) - 1e-9


def _sample_containment_row(out_dir: Path) -> Mapping[str, str] | None:
    path = out_dir / "sample_containment_summary.csv"
    if not path.exists():
        return None
    rows = _read_csv_rows(path)
    return rows[0] if rows else None


def _sample_containment_passed(row: Mapping[str, str] | None) -> bool | None:
    if row is None:
        return None
    violations = int(float(row.get("violations_count") or 0))
    return violations == 0 and str(row.get("status", "")).lower() in {"passed", "pass", "ok"}


def _best_h10_summary(
    summary_rows: Sequence[Mapping[str, Any]], comparison_rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    if not summary_rows:
        return None
    comp_by_run = _comparison_by_run(comparison_rows)

    def key(row: Mapping[str, Any]) -> tuple[float, float, float]:
        comp = comp_by_run.get(str(row.get("run_id", "")), {})
        tube_ratio = _finite_float(comp.get("tube_width_ratio"))
        runtime = _finite_float(row.get("runtime_s"))
        return (
            _finite_float(row.get("last_validated_t")) or 0.0,
            -(tube_ratio if tube_ratio is not None else math.inf),
            -(runtime if runtime is not None else math.inf),
        )

    normalized = [row for row in summary_rows if row.get("reset_mode") == "normalized_insertion"]
    return max(normalized or list(summary_rows), key=key)


def _widths_are_flowstar_comparable(best: Mapping[str, Any] | None, comparison_rows: Sequence[Mapping[str, Any]]) -> bool | None:
    if not best:
        return None
    comp = _comparison_by_run(comparison_rows).get(str(best.get("run_id", "")))
    if not comp:
        return None
    last_ratio = _finite_float(comp.get("last_width_ratio"))
    tube_ratio = _finite_float(comp.get("tube_width_ratio"))
    if last_ratio is None or tube_ratio is None:
        return None
    return last_ratio <= 1.10 and tube_ratio <= 1.10


def make_normalized_insertion_h10_plots(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]]) -> None:
    _copy_plot_if_present(out_dir, "rescue_t_x.png", "normalized_insertion_h10_t_x.png")
    _copy_plot_if_present(out_dir, "rescue_t_y.png", "normalized_insertion_h10_t_y.png")
    _copy_plot_if_present(out_dir, "rescue_phase_xy.png", "normalized_insertion_h10_phase_xy.png")
    _copy_plot_if_present(
        out_dir,
        "overlay_rescue_vs_original_flowstar_t_x.png",
        "overlay_normalized_insertion_vs_original_flowstar_t_x.png",
    )
    _copy_plot_if_present(
        out_dir,
        "overlay_rescue_vs_original_flowstar_t_y.png",
        "overlay_normalized_insertion_vs_original_flowstar_t_y.png",
    )
    _copy_plot_if_present(
        out_dir,
        "overlay_rescue_vs_original_flowstar_phase_xy.png",
        "overlay_normalized_insertion_vs_original_flowstar_phase_xy.png",
    )
    _copy_plot_if_present(out_dir, "reset_box_width_trace.png", "reset_width_vs_t.png")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = [row for row in segment_rows if row.get("reset_mode") == "normalized_insertion"]
    if not rows:
        return
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("run_id", "")), []).append(row)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id, run_rows in grouped.items():
        pts = []
        for row in sorted(run_rows, key=lambda r: _finite_float(r.get("t_hi")) or 0.0):
            t = _finite_float(row.get("t_hi"))
            width = _finite_float(row.get("reset_width_sum"))
            if t is not None and width is not None:
                pts.append((t, width))
        if pts:
            ax.plot([t for t, _width in pts], [width for _t, width in pts], marker="o", markersize=2.4, linewidth=1.0, label=run_id)
    ax.set_xlabel("t")
    ax.set_ylabel("reset width sum")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "reset_width_vs_t.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    field_labels = [
        ("insertion_truncation_width", "truncation"),
        ("insertion_cutoff_width", "cutoff"),
        ("output_remainder_width", "output remainder"),
    ]
    for run_id, run_rows in grouped.items():
        for field, label in field_labels:
            pts = []
            for row in sorted(run_rows, key=lambda r: _finite_float(r.get("t_hi")) or 0.0):
                t = _finite_float(row.get("t_hi"))
                value = _finite_float(row.get(field))
                if t is not None and value is not None and value > 0:
                    pts.append((t, value))
            if pts:
                ax.plot([t for t, _value in pts], [value for _t, value in pts], linewidth=0.9, label=f"{run_id} {label}")
    ax.set_xlabel("t")
    ax.set_ylabel("uncertainty width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "insertion_uncertainty_vs_t.png", dpi=160)
    plt.close(fig)


def write_normalized_insertion_h10_vs_flowstar_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    comp_by_run = _comparison_by_run(comparison_rows)
    lines = [
        "# Normalized Insertion H10 Vs Original Flow* Comparison",
        "",
        f"Requested horizon: `{float(max_horizon):.17g}`.",
        "Original Flow* boxes are GNUPLOT segment boxes; adaptive PyTorch grids are not expected to match segment counts.",
        "This report is a width and overlap comparison, not an exact Flow* parity claim.",
        "",
        "## Metrics",
        "",
        "| run_id | status | runtime_s | segments | last_validated_t | min_h_used | min_regular_h_used | h_below_flowstar_min_count | max_h_used | step_rejections | final_width_sum | py_tube_width_sum | flowstar_last_width_sum | flowstar_tube_width_sum | last_width_ratio | tube_width_ratio | max_overlap_width_ratio | median_overlap_width_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _ordered_h10_rows(summary_rows):
        comp = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('runtime_s', '')} | "
            f"{row.get('validated_segments', '')} | {row.get('last_validated_t', '')} | {row.get('min_h_used', '')} | "
            f"{row.get('min_regular_h_used', '')} | {row.get('h_below_flowstar_min_count', '')} | {row.get('max_h_used', '')} | "
            f"{row.get('num_step_rejections', '')} | {row.get('final_width_sum', '')} | {comp.get('py_tube_width_sum', '')} | "
            f"{comp.get('flowstar_last_width_sum_near_T', '')} | {comp.get('flowstar_tube_width_sum_over_same_horizon', '')} | "
            f"{comp.get('last_width_ratio', '')} | {comp.get('tube_width_ratio', '')} | "
            f"{comp.get('max_time_overlap_width_ratio', '')} | {comp.get('median_time_overlap_width_ratio', '')} |"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "normalized_insertion_h10_vs_flowstar_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_normalized_insertion_h10_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best_h10_summary(summary_rows, comparison_rows)
    comp_by_run = _comparison_by_run(comparison_rows)
    best_comp = comp_by_run.get(str(best.get("run_id", ""))) if best else None
    reached_rows = [row for row in summary_rows if _reached_requested(row, max_horizon)]
    any_reached = bool(reached_rows)
    order4_reached = any(str(row.get("run_id", "")).startswith("flowstar_style_o4_") and _reached_requested(row, max_horizon) for row in summary_rows)
    order6_reached = any("o6_candidate8" in str(row.get("run_id", "")) and _reached_requested(row, max_horizon) for row in summary_rows)
    below_min_any = any(int(row.get("h_below_flowstar_min_count") or 0) > 0 for row in summary_rows)
    sample_row = _sample_containment_row(out_dir)
    sample_passed = _sample_containment_passed(sample_row)
    comparable = _widths_are_flowstar_comparable(best, comparison_rows)
    if comparable is None:
        comparable_text = "unknown; comparison data unavailable"
    elif comparable:
        comparable_text = "yes; width ratios are within 10% of Flow* over the reported comparison horizon"
    else:
        comparable_text = "no; width ratios exceed the 10% comparison threshold"
    if sample_passed is None:
        sample_text = "pending; `sample_containment_summary.csv` has not been generated yet"
    else:
        sample_text = "passed" if sample_passed else "failed"
    best_below_min = bool(best and int(best.get("h_below_flowstar_min_count") or 0) > 0)
    decision = "MERGE_CANDIDATE" if any_reached and comparable is True and sample_passed is True and not best_below_min else "NEEDS_MORE_WORK"
    if decision == "MERGE_CANDIDATE":
        recommendation = "merge experimental branch"
    elif not any_reached:
        recommendation = "investigate any h10 failure point"
    elif comparable is False:
        recommendation = "add Flow*-style symbolic remainder queue on top of normalized insertion"
    elif sample_passed is False:
        recommendation = "investigate sample containment failure point"
    else:
        recommendation = "finish sample containment and comparison verification"

    lines = [
        "# Normalized Insertion H10 Report",
        "",
        f"Did any normalized insertion config reach horizon 10? {_yes_no(any_reached)}.",
        f"Which config is best? `{best.get('run_id', '') if best else ''}`.",
        f"Did order4 reach horizon 10? {_yes_no(order4_reached)}.",
        f"Did order6/candidate8 reach horizon 10? {_yes_no(order6_reached)}.",
        f"Are widths comparable to Flow*? {comparable_text}.",
        f"Did any non-final step go below Flow* min step 0.002? {_yes_no(below_min_any)}.",
        f"Did sample containment sanity pass? {sample_text}.",
        f"Branch decision: {decision}.",
        f"Recommended next step: {recommendation}.",
        "",
        "## Best Config Metrics",
        "",
        "| run_id | status | runtime_s | segments | last_validated_t | min_regular_h_used | h_below_flowstar_min_count | final_width_sum | last_width_ratio | tube_width_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if best:
        lines.append(
            f"| {best.get('run_id', '')} | {best.get('status', '')} | {best.get('runtime_s', '')} | "
            f"{best.get('validated_segments', '')} | {best.get('last_validated_t', '')} | {best.get('min_regular_h_used', '')} | "
            f"{best.get('h_below_flowstar_min_count', '')} | {best.get('final_width_sum', '')} | "
            f"{best_comp.get('last_width_ratio', '') if best_comp else ''} | {best_comp.get('tube_width_ratio', '') if best_comp else ''} |"
        )
    lines.extend(
        [
            "",
            "## Config Status",
            "",
            "| run_id | status | runtime_s | segments | last_validated_t | min_regular_h_used | h_below_flowstar_min_count | failure_reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in _ordered_h10_rows(summary_rows):
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('runtime_s', '')} | "
            f"{row.get('validated_segments', '')} | {row.get('last_validated_t', '')} | {row.get('min_regular_h_used', '')} | "
            f"{row.get('h_below_flowstar_min_count', '')} | {row.get('failure_reason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Do not claim exact Flow* parity from this report; adaptive grids differ and segment boxes are not expected to match exactly.",
            "If only order6/candidate8 reaches h10, this is a higher-order PyTorch rescue result, not original order-4 parity.",
            "If an order4 insert config reaches h10, treat it as the closer result to original Flow* settings and highlight it separately.",
        ]
    )
    if sample_row is not None:
        lines.extend(
            [
                "",
                "## Sample Containment",
                "",
                "| run_id | samples | checked_pairs | violations | max_outside_distance | status |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
                f"| {sample_row.get('run_id', '')} | {sample_row.get('num_samples', '')} | {sample_row.get('checked_sample_time_pairs', '')} | "
                f"{sample_row.get('violations_count', '')} | {sample_row.get('max_outside_distance', '')} | {sample_row.get('status', '')} |",
            ]
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "normalized_insertion_h10_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_normalized_insertion_h10_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_csv(out_dir / "normalized_insertion_h10_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "normalized_insertion_h10_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "normalized_insertion_h10_reset_diagnostics.csv", NORMALIZED_INSERTION_RESET_FIELDS, _normalized_insertion_reset_rows(segment_rows))
    _write_csv(out_dir / "normalized_insertion_h10_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    _write_csv(out_dir / "normalized_insertion_h10_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
    write_normalized_insertion_h10_vs_flowstar_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    write_normalized_insertion_h10_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    make_normalized_insertion_h10_plots(out_dir, segment_rows)


def _normal_eval_baseline_row(run_id: str) -> Mapping[str, Any]:
    rows = _read_optional_csv(REPO_ROOT / "outputs" / H10_OUTPUT_DIR_NAME / "normalized_insertion_h10_summary.csv")
    return next((row for row in rows if row.get("run_id") == run_id), {})


def _normal_eval_comparison_by_run(comparison_rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("run_id", "")): row for row in comparison_rows}


def make_normal_eval_h10_plots(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]]) -> None:
    _copy_plot_if_present(out_dir, "width_ratio_vs_t.png", "normal_eval_width_ratio_vs_t.png")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [row for row in segment_rows if row.get("status") == "validated" and row.get("right_map_range_mode") == "normal_eval"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id in sorted({str(row.get("run_id", "")) for row in rows}):
        sub = sorted([row for row in rows if row.get("run_id") == run_id], key=lambda r: _finite_float(r.get("t_hi")) or 0.0)
        ts = [_finite_float(row.get("t_hi")) or 0.0 for row in sub]
        old_vals = [_finite_float(row.get("old_right_map_range_width_sum")) or 0.0 for row in sub]
        normal_vals = [_finite_float(row.get("normal_right_map_range_width_sum")) or 0.0 for row in sub]
        ax.plot(ts, old_vals, linewidth=0.9, label=f"{run_id} old")
        ax.plot(ts, normal_vals, linewidth=0.9, linestyle="--", label=f"{run_id} normal")
    ax.set_xlabel("t")
    ax.set_ylabel("right-map range width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "normal_eval_range_compare.png", dpi=160)
    plt.close(fig)


def write_normal_eval_h10_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best_h10_summary(summary_rows, comparison_rows)
    comp_by_run = _normal_eval_comparison_by_run(comparison_rows)
    best_comp = comp_by_run.get(str(best.get("run_id", ""))) if best else None
    o4_base = _normal_eval_baseline_row("flowstar_style_o4_target_insert")
    o6_base = _normal_eval_baseline_row("flowstar_style_o6_candidate8_output6_insert")
    o4_base_t = _finite_float(o4_base.get("last_validated_t")) or 6.4730088058091901
    o6_base_t = _finite_float(o6_base.get("last_validated_t")) or 7.4960392581387341
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    any_reached = any(_reached_requested(row, max_horizon) for row in summary_rows)
    improved_horizon = bool(best_t and best_t > max(o4_base_t, o6_base_t) + 1e-12)
    normal_rows = [row for row in summary_rows if row.get("right_map_range_mode") == "normal_eval"]
    max_old = _max_field(normal_rows, "max_inserted_endpoint_width_sum")
    reset_rows = _read_optional_csv(out_dir / "normal_eval_reset_diagnostics.csv")
    max_old_range = _max_field(reset_rows, "old_right_map_range_width_sum")
    max_normal_range = _max_field(reset_rows, "normal_right_map_range_width_sum")
    old_range_f = _finite_float(max_old_range)
    normal_range_f = _finite_float(max_normal_range)
    range_shrank = old_range_f is not None and normal_range_f is not None and normal_range_f < old_range_f
    sample_row = _sample_containment_row(out_dir)
    sample_passed = _sample_containment_passed(sample_row)
    sample_text = "pending" if sample_passed is None else ("passed" if sample_passed else "failed")
    width_ratio_text = "not compared"
    if best_comp:
        width_ratio_text = f"last=`{best_comp.get('last_width_ratio', '')}`, tube=`{best_comp.get('tube_width_ratio', '')}`"
    branch_decision = "MERGE_CANDIDATE" if (any_reached or improved_horizon or range_shrank) else "NEEDS_MORE_WORK"
    lines = [
        "# Normal Eval H10 Report",
        "",
        f"Baseline o4 no-normal-eval: t=`{o4_base_t:.17g}`, final width=`{o4_base.get('final_width_sum', '')}`.",
        f"Baseline o6 no-normal-eval: t=`{o6_base_t:.17g}`, final width=`{o6_base.get('final_width_sum', '')}`.",
        f"Best normal_eval config: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did normal_eval beat the o4 baseline t~={o4_base_t:.17g} or o6 baseline t~={o6_base_t:.17g}? {_yes_no(improved_horizon)}.",
        f"Did any config reach h10? {_yes_no(any_reached)}.",
        f"Did width ratios improve? {width_ratio_text}.",
        f"Did right_map_scaling shrink? {_yes_no(range_shrank)}; old max=`{max_old_range}`, normal max=`{max_normal_range}`, inserted max=`{max_old}`.",
        f"Did sample containment pass? {sample_text}.",
        "Did normal_eval remain conservative in tests? See pytest result for `evaluate_interval_normal` sample containment tests.",
        f"Branch decision: {branch_decision}.",
        "",
        "## Config Status",
        "",
        "| run_id | status | last_validated_t | right_map_range_mode | final_width_sum | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in _ordered_h10_rows(summary_rows):
        comp = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('right_map_range_mode', '')} | {row.get('final_width_sum', '')} | "
            f"{comp.get('last_width_ratio', '')} | {comp.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    if sample_row is not None:
        lines.extend([
            "",
            "## Sample Containment",
            "",
            "| run_id | samples | checked_pairs | violations | max_outside_distance | status |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            f"| {sample_row.get('run_id', '')} | {sample_row.get('num_samples', '')} | {sample_row.get('checked_sample_time_pairs', '')} | "
            f"{sample_row.get('violations_count', '')} | {sample_row.get('max_outside_distance', '')} | {sample_row.get('status', '')} |",
        ])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "normal_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    decision_lines = [
        "# Branch Decision",
        "",
        f"Decision: {branch_decision}",
        "",
        "## Evidence",
        "",
        f"- Horizon 10 reached: {_yes_no(any_reached)}.",
        f"- Best normal_eval t: `{best_t}`.",
        f"- Baseline o4/o6 t: `{o4_base_t:.17g}` / `{o6_base_t:.17g}`.",
        f"- Right-map range shrank: {_yes_no(range_shrank)}.",
        f"- Sample containment: {sample_text}.",
        "- No fake Flow* parity is claimed; comparison ratios are reported separately.",
        "",
    ]
    (out_dir / "branch_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")


def write_normal_eval_h10_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_csv(out_dir / "normal_eval_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "normal_eval_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "normal_eval_reset_diagnostics.csv", NORMALIZED_INSERTION_RESET_FIELDS, _normalized_insertion_reset_rows(segment_rows))
    _write_csv(out_dir / "normal_eval_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    _write_csv(out_dir / "normal_eval_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
    write_normal_eval_h10_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    make_normal_eval_h10_plots(out_dir, segment_rows)




def make_symqueue_h10_plots(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]]) -> None:
    _copy_plot_if_present(out_dir, "width_ratio_vs_t.png", "symqueue_width_ratio_vs_t.png")
    _copy_plot_if_present(
        out_dir,
        "overlay_rescue_vs_original_flowstar_t_x.png",
        "overlay_symqueue_vs_original_flowstar_t_x.png",
    )
    _copy_plot_if_present(
        out_dir,
        "overlay_rescue_vs_original_flowstar_t_y.png",
        "overlay_symqueue_vs_original_flowstar_t_y.png",
    )
    _copy_plot_if_present(
        out_dir,
        "overlay_rescue_vs_original_flowstar_phase_xy.png",
        "overlay_symqueue_vs_original_flowstar_phase_xy.png",
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in segment_rows:
        if str(row.get("reset_mode", "")) == "normalized_insertion_symqueue" and row.get("status") == "validated":
            grouped.setdefault(str(row.get("run_id", "")), []).append(row)

    def _plot_field(field: str, ylabel: str, filename: str, *, log: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        for run_id, rows in grouped.items():
            pts = []
            for row in sorted(rows, key=lambda r: _finite_float(r.get("t_hi")) or 0.0):
                t = _finite_float(row.get("t_hi"))
                value = _finite_float(row.get(field))
                if t is not None and value is not None:
                    pts.append((t, value))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=2.2, linewidth=1.0, label=run_id)
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    _plot_field("queue_size", "symbolic queue size", "symqueue_queue_size_vs_t.png")
    _plot_field("propagated_symbolic_width_sum", "propagated symbolic width sum", "symqueue_propagated_width_vs_t.png", log=True)


def _baseline_normalized_h10_rows() -> list[dict[str, str]]:
    path = REPO_ROOT / "outputs" / H10_OUTPUT_DIR_NAME / "normalized_insertion_h10_summary.csv"
    return _read_csv_rows(path) if path.exists() else []


def _baseline_t(run_id: str, fallback: float) -> float:
    for row in _baseline_normalized_h10_rows():
        if row.get("run_id") == run_id:
            return _finite_float(row.get("last_validated_t")) or fallback
    return fallback


def write_symqueue_h10_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best_h10_summary(summary_rows, comparison_rows)
    comp_by_run = _comparison_by_run(comparison_rows)
    best_comp = comp_by_run.get(str(best.get("run_id", ""))) if best else None
    any_reached = any(_reached_requested(row, max_horizon) for row in summary_rows)
    order4_reached = any(str(row.get("run_id", "")).startswith("flowstar_style_o4_") and _reached_requested(row, max_horizon) for row in summary_rows)
    order6_reached = any("o6_candidate8" in str(row.get("run_id", "")) and _reached_requested(row, max_horizon) for row in summary_rows)
    o4_base = _baseline_t("flowstar_style_o4_target_insert", 6.4730088058091901)
    o6_base = _baseline_t("flowstar_style_o6_candidate8_output6_insert", 7.4960392581387341)
    best_o4 = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in summary_rows if str(row.get("run_id", "")).startswith("flowstar_style_o4_")), default=0.0)
    best_o6 = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in summary_rows if "o6_candidate8" in str(row.get("run_id", ""))), default=0.0)
    sample_row = _sample_containment_row(out_dir)
    sample_passed = _sample_containment_passed(sample_row)
    max_materialized = _max_field(summary_rows, "max_symqueue_materialized_width_sum")
    max_propagated = _max_field(summary_rows, "max_symqueue_propagated_symbolic_width_sum")
    queue_sizes = [_finite_float(row.get("max_flowstar_queue_size_after")) for row in summary_rows]
    queue_sizes = [value for value in queue_sizes if value is not None]
    queue_stable = bool(queue_sizes and max(queue_sizes) < 100 and (_finite_float(max_materialized) or 0.0) < 1.0)
    sample_text = "pending" if sample_passed is None else ("passed" if sample_passed else "failed")
    width_reduced = "unknown"
    if best_comp:
        ratio = _finite_float(best_comp.get("tube_width_ratio"))
        width_reduced = "yes" if ratio is not None and ratio < 2.4989346486923725 else "no"
    best_text = best.get("run_id", "") if best else ""
    lines = [
        "# Normalized Insertion Plus Symbolic Queue H10 Report",
        "",
        f"Did any symqueue config reach horizon 10? {_yes_no(any_reached)}.",
        f"Did order4 reach horizon 10? {_yes_no(order4_reached)}.",
        f"Did order6/candidate8 reach horizon 10? {_yes_no(order6_reached)}.",
        f"Did symqueue improve last_validated_t over o4_insert baseline t~={o4_base:.17g}? {_yes_no(best_o4 > o4_base + 1e-12)}; best order4 t=`{best_o4:.17g}`.",
        f"Did symqueue improve last_validated_t over o6_insert baseline t~={o6_base:.17g}? {_yes_no(best_o6 > o6_base + 1e-12)}; best order6 t=`{best_o6:.17g}`.",
        f"Did symqueue reduce width ratios? {width_reduced}.",
        f"Did queue remain stable or materialize too much width? {'stable' if queue_stable else 'needs review'}; max propagated=`{max_propagated}`, max materialized=`{max_materialized}`.",
        f"Runtime cost for best config: `{best.get('runtime_s', '') if best else ''}` seconds.",
        f"Did sample containment still pass? {sample_text}.",
        f"Which config is best? `{best_text}`.",
        "Queue implementation note: this is a conservative limited queue. It propagates older interval columns through the inserted endpoint linear part and materializes propagated width on the next normalized reset; current insertion uncertainty is queued for future propagation.",
        "Branch status remains NEEDS_MORE_WORK unless h10 is reached and Flow* comparison/sample checks are acceptable.",
        "",
        "## Best Config Metrics",
        "",
        "| run_id | status | last_validated_t | runtime_s | queue_size | propagated_width | materialized_width | last_width_ratio | tube_width_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if best:
        lines.append(
            f"| {best.get('run_id', '')} | {best.get('status', '')} | {best.get('last_validated_t', '')} | "
            f"{best.get('runtime_s', '')} | {best.get('max_flowstar_queue_size_after', '')} | "
            f"{best.get('max_symqueue_propagated_symbolic_width_sum', '')} | {best.get('max_symqueue_materialized_width_sum', '')} | "
            f"{best_comp.get('last_width_ratio', '') if best_comp else ''} | {best_comp.get('tube_width_ratio', '') if best_comp else ''} |"
        )
    lines.extend([
        "",
        "## Config Status",
        "",
        "| run_id | status | last_validated_t | accepted | rejected | max_queue | max_propagated | max_new_symbolic | max_materialized | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in _ordered_symqueue_h10_rows(summary_rows):
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('num_accepted_steps', '')} | {row.get('num_rejected_steps', '')} | {row.get('max_flowstar_queue_size_after', '')} | "
            f"{row.get('max_symqueue_propagated_symbolic_width_sum', '')} | {row.get('max_symqueue_new_symbolic_width_sum', '')} | "
            f"{row.get('max_symqueue_materialized_width_sum', '')} | {row.get('failure_reason', '')} |"
        )
    if sample_row is not None:
        lines.extend([
            "",
            "## Sample Containment",
            "",
            "| run_id | samples | checked_pairs | violations | max_outside_distance | status |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            f"| {sample_row.get('run_id', '')} | {sample_row.get('num_samples', '')} | {sample_row.get('checked_sample_time_pairs', '')} | "
            f"{sample_row.get('violations_count', '')} | {sample_row.get('max_outside_distance', '')} | {sample_row.get('status', '')} |",
        ])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "symqueue_h10_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_symqueue_h10_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_csv(out_dir / "symqueue_h10_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "symqueue_h10_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "symqueue_h10_reset_diagnostics.csv", NORMALIZED_INSERTION_RESET_FIELDS, _normalized_insertion_reset_rows(segment_rows))
    _write_csv(out_dir / "symqueue_h10_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    _write_csv(out_dir / "symqueue_h10_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
    write_symqueue_h10_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    make_symqueue_h10_plots(out_dir, segment_rows)


def _baseline_symqueue_h10_rows() -> list[dict[str, str]]:
    path = REPO_ROOT / "outputs" / SYMQUEUE_H10_OUTPUT_DIR_NAME / "symqueue_h10_summary.csv"
    return _read_csv_rows(path) if path.exists() else []


def _old_symqueue_best_t(fallback: float = 3.35) -> float:
    rows = _baseline_symqueue_h10_rows()
    values = [_finite_float(row.get("last_validated_t")) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else fallback


def make_symqueue_split_h10_plots(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]]) -> None:
    _copy_plot_if_present(out_dir, "width_ratio_vs_t.png", "symqueue_split_width_ratio_vs_t.png")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in segment_rows:
        if str(row.get("reset_mode", "")) == "normalized_insertion_symqueue_split" and row.get("status") == "validated":
            grouped.setdefault(str(row.get("run_id", "")), []).append(row)

    def _plot_field(field: str, ylabel: str, filename: str, *, log: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        for run_id, rows in grouped.items():
            pts = []
            for row in sorted(rows, key=lambda r: _finite_float(r.get("t_hi")) or 0.0):
                t = _finite_float(row.get("t_hi"))
                value = _finite_float(row.get(field))
                if t is not None and value is not None:
                    pts.append((t, value))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=2.2, linewidth=1.0, label=run_id)
        ax.set_xlabel("t")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    _plot_field("queue_size", "symbolic queue size", "symqueue_split_queue_size_vs_t.png")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    fields = [
        ("ordinary_only_range_width", "ordinary-only range"),
        ("symbolic_contribution_width", "symbolic contribution"),
        ("materialized_for_output_width", "materialized for output"),
        ("total_range_width_with_symbolic", "total with symbolic"),
    ]
    for run_id, rows in grouped.items():
        for field, label in fields:
            pts = []
            for row in sorted(rows, key=lambda r: _finite_float(r.get("t_hi")) or 0.0):
                t = _finite_float(row.get("t_hi"))
                value = _finite_float(row.get(field))
                if t is not None and value is not None and value > 0:
                    pts.append((t, value))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=0.9, label=f"{run_id} {label}")
    ax.set_xlabel("t")
    ax.set_ylabel("width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "symqueue_channel_widths_vs_t.png", dpi=160)
    plt.close(fig)


def write_symqueue_split_h10_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best_h10_summary(summary_rows, comparison_rows)
    comp_by_run = _comparison_by_run(comparison_rows)
    best_comp = comp_by_run.get(str(best.get("run_id", ""))) if best else None
    any_reached = any(_reached_requested(row, max_horizon) for row in summary_rows)
    order4_reached = any(str(row.get("run_id", "")).startswith("flowstar_style_o4_") and _reached_requested(row, max_horizon) for row in summary_rows)
    order6_reached = any("o6_candidate8" in str(row.get("run_id", "")) and _reached_requested(row, max_horizon) for row in summary_rows)
    o4_base = _baseline_t("flowstar_style_o4_target_insert", 6.4730088058091901)
    o6_base = _baseline_t("flowstar_style_o6_candidate8_output6_insert", 7.4960392581387341)
    old_symqueue_t = _old_symqueue_best_t()
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    best_o4 = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in summary_rows if str(row.get("run_id", "")).startswith("flowstar_style_o4_")), default=0.0)
    best_o6 = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in summary_rows if "o6_candidate8" in str(row.get("run_id", ""))), default=0.0)
    sample_row = _sample_containment_row(out_dir)
    sample_passed = _sample_containment_passed(sample_row)
    sample_text = "pending" if sample_passed is None else ("passed" if sample_passed else "failed")
    max_symbolic = _max_field(summary_rows, "max_symbolic_contribution_width")
    max_materialized = _max_field(summary_rows, "max_materialized_for_output_width")
    symbolic_bounded = (_finite_float(max_symbolic) or 0.0) < 1.0 and (_finite_float(max_materialized) or 0.0) < 1.0
    target_stayed = all(str(row.get("target_remainder_radius", "")) in {"0.0001", "0.000100000000000000", "1e-04", "1e-4"} for row in summary_rows)
    width_ratio_text = "not compared"
    if best_comp:
        width_ratio_text = f"last=`{best_comp.get('last_width_ratio', '')}`, tube=`{best_comp.get('tube_width_ratio', '')}`"
    failure = best.get("failure_reason", "") if best else ""
    lines = [
        "# Normalized Insertion Plus Symbolic Queue Split H10 Report",
        "",
        f"Did split semantics beat old symqueue t~={old_symqueue_t:.17g}? {_yes_no(bool(best_t is not None and best_t > old_symqueue_t + 1e-12))}; best split t=`{best_t}`.",
        f"Did split beat no-queue o4 t~={o4_base:.17g}? {_yes_no(best_o4 > o4_base + 1e-12)}; best order4 t=`{best_o4:.17g}`.",
        f"Did split beat no-queue o6 t~={o6_base:.17g}? {_yes_no(best_o6 > o6_base + 1e-12)}; best order6 t=`{best_o6:.17g}`.",
        f"Did any config reach horizon 10? {_yes_no(any_reached)}.",
        f"Did order4 reach horizon 10? {_yes_no(order4_reached)}.",
        f"Did order6/candidate8 reach horizon 10? {_yes_no(order6_reached)}.",
        f"Did range boxes remain conservative and sample containment pass? {sample_text}.",
        f"Did symbolic contribution remain bounded? {_yes_no(symbolic_bounded)}; max symbolic=`{max_symbolic}`, max output materialized=`{max_materialized}`.",
        f"Did ordinary target remainder stay at 1e-4? {_yes_no(target_stayed)}.",
        f"Flow* width ratios for best config: {width_ratio_text}.",
        f"Runtime cost for best config: `{best.get('runtime_s', '') if best else ''}` seconds.",
        f"If it still fails, exact reported failure component: `{failure}`.",
        "Queue split note: propagated symbolic width is added to reported output/range boxes, while ordinary target containment checks the local Picard target remainder channel.",
        "",
        "## Best Config Metrics",
        "",
        "| run_id | status | last_validated_t | runtime_s | queue_size | ordinary_only_range | symbolic_contribution | output_materialized | total_with_symbolic | last_width_ratio | tube_width_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if best:
        lines.append(
            f"| {best.get('run_id', '')} | {best.get('status', '')} | {best.get('last_validated_t', '')} | "
            f"{best.get('runtime_s', '')} | {best.get('max_flowstar_queue_size_after', '')} | "
            f"{best.get('max_ordinary_only_range_width', '')} | {best.get('max_symbolic_contribution_width', '')} | "
            f"{best.get('max_materialized_for_output_width', '')} | {best.get('max_total_range_width_with_symbolic', '')} | "
            f"{best_comp.get('last_width_ratio', '') if best_comp else ''} | {best_comp.get('tube_width_ratio', '') if best_comp else ''} |"
        )
    lines.extend([
        "",
        "## Config Status",
        "",
        "| run_id | status | last_validated_t | accepted | rejected | max_queue | max_symbolic | max_output_materialized | target_checked_width | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in _ordered_symqueue_split_h10_rows(summary_rows):
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('num_accepted_steps', '')} | {row.get('num_rejected_steps', '')} | {row.get('max_flowstar_queue_size_after', '')} | "
            f"{row.get('max_symbolic_contribution_width', '')} | {row.get('max_materialized_for_output_width', '')} | "
            f"{row.get('max_target_checked_width', '')} | {row.get('failure_reason', '')} |"
        )
    if sample_row is not None:
        lines.extend([
            "",
            "## Sample Containment",
            "",
            "| run_id | samples | checked_pairs | violations | max_outside_distance | status |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            f"| {sample_row.get('run_id', '')} | {sample_row.get('num_samples', '')} | {sample_row.get('checked_sample_time_pairs', '')} | "
            f"{sample_row.get('violations_count', '')} | {sample_row.get('max_outside_distance', '')} | {sample_row.get('status', '')} |",
        ])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "symqueue_split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_symqueue_split_h10_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_csv(out_dir / "symqueue_split_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "symqueue_split_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "symqueue_split_reset_diagnostics.csv", NORMALIZED_INSERTION_RESET_FIELDS, _normalized_insertion_reset_rows(segment_rows))
    _write_csv(out_dir / "symqueue_split_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    _write_csv(out_dir / "symqueue_split_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
    write_symqueue_split_h10_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    make_symqueue_split_h10_plots(out_dir, segment_rows)


def _baseline_symqueue_split_h10_rows() -> list[dict[str, str]]:
    path = REPO_ROOT / "outputs" / SYMQUEUE_SPLIT_H10_OUTPUT_DIR_NAME / "symqueue_split_summary.csv"
    return _read_csv_rows(path) if path.exists() else []


def _split_symqueue_best_t(fallback: float = 3.35) -> float:
    rows = _baseline_symqueue_split_h10_rows()
    values = [_finite_float(row.get("last_validated_t")) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else fallback


def make_symqueue_v2_h10_plots(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in segment_rows:
        if str(row.get("reset_mode", "")) == "normalized_insertion_symqueue_v2" and row.get("status") == "validated":
            grouped.setdefault(str(row.get("run_id", "")), []).append(row)
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    fields = [
        ("ordinary_only_range_width", "ordinary range"),
        ("output_only_symbolic_width_sum", "output-only symbolic"),
        ("total_range_width_with_symbolic", "total with symbolic"),
    ]
    for run_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda r: _finite_float(r.get("t_hi")) or 0.0)
        for field, label in fields:
            pts = []
            for row in ordered:
                t = _finite_float(row.get("t_hi"))
                value = _finite_float(row.get(field))
                if t is not None and value is not None and value >= 0.0:
                    pts.append((t, value))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=0.9, label=f"{run_id} {label}")
    ax.set_xlabel("t")
    ax.set_ylabel("width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "queue_channels_vs_t.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id, rows in grouped.items():
        pts = []
        for row in sorted(rows, key=lambda r: _finite_float(r.get("t_hi")) or 0.0):
            t = _finite_float(row.get("t_hi"))
            value = _finite_float(row.get("reset_box_width_sum"))
            if value is None:
                value = _finite_float(row.get("reset_width_sum"))
            if t is not None and value is not None:
                pts.append((t, value))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=2.2, linewidth=1.0, label=run_id)
    ax.set_xlabel("t")
    ax.set_ylabel("reset width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "reset_width_vs_t.png", dpi=160)
    plt.close(fig)


def write_symqueue_v2_h10_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    best = _best_h10_summary(summary_rows, comparison_rows)
    comp_by_run = _comparison_by_run(comparison_rows)
    best_comp = comp_by_run.get(str(best.get("run_id", ""))) if best else None
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    split_t = _split_symqueue_best_t()
    o4_base = _baseline_t("flowstar_style_o4_target_insert", 6.4730088058091901)
    o6_base = _baseline_t("flowstar_style_o6_candidate8_output6_insert", 7.4960392581387341)
    any_reached = any(_reached_requested(row, max_horizon) for row in summary_rows)
    order4_reached = any(str(row.get("run_id", "")).startswith("flowstar_style_o4_") and _reached_requested(row, max_horizon) for row in summary_rows)
    order6_reached = any("o6_candidate8" in str(row.get("run_id", "")) and _reached_requested(row, max_horizon) for row in summary_rows)
    best_o4 = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in summary_rows if str(row.get("run_id", "")).startswith("flowstar_style_o4_")), default=0.0)
    best_o6 = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in summary_rows if "o6_candidate8" in str(row.get("run_id", ""))), default=0.0)
    v2_segments = [row for row in segment_rows if str(row.get("reset_mode", "")) == "normalized_insertion_symqueue_v2"]
    max_j = _max_field(v2_segments, "j_count")
    max_phi = _max_field(v2_segments, "phi_l_count")
    max_linear = _max_field(v2_segments, "current_linear_map_norm")
    max_scalar = _max_abs_fields(v2_segments, ["scalar_x", "scalar_y"])
    max_reset = _max_field(v2_segments, "reset_box_width_sum")
    max_output = _max_field(v2_segments, "output_only_symbolic_width_sum")
    max_target = _max_field(v2_segments, "target_check_width_sum")
    output_flags = [row for row in v2_segments if row.get("status") == "validated"]
    output_conservative = all(_truthy(row.get("output_range_includes_symbolic_contributions")) for row in output_flags) if output_flags else False
    sample_row = _sample_containment_row(out_dir)
    sample_passed = _sample_containment_passed(sample_row)
    sample_text = "pending" if sample_passed is None else ("passed" if sample_passed else "failed")
    conservative = output_conservative and sample_passed is not False
    reset_effect = "target-clean reset; propagated queue contribution is output-only in v2"
    if (_finite_float(max_output) or 0.0) == 0.0:
        reset_effect = "no propagated output-only queue contribution was observed before the reported horizon/failure"
    failure = best.get("failure_reason", "") if best else ""
    width_ratio_text = "not compared"
    if best_comp:
        width_ratio_text = f"last=`{best_comp.get('last_width_ratio', '')}`, tube=`{best_comp.get('tube_width_ratio', '')}`"

    lines = [
        "# Flowstar Linear Symbolic Queue V2 H10 Report",
        "",
        f"Did v2 beat split queue t~={split_t:.17g}? {_yes_no(bool(best_t is not None and best_t > split_t + 1e-12))}; best v2 t=`{best_t}`.",
        f"Did v2 beat no-queue o4 t~={o4_base:.17g}? {_yes_no(best_o4 > o4_base + 1e-12)}; best order4 t=`{best_o4:.17g}`.",
        f"Did v2 beat no-queue o6 t~={o6_base:.17g}? {_yes_no(best_o6 > o6_base + 1e-12)}; best order6 t=`{best_o6:.17g}`.",
        f"Did any config reach h10? {_yes_no(any_reached)}.",
        f"Did o4 reach h10? {_yes_no(order4_reached)}.",
        f"Did o6 reach h10? {_yes_no(order6_reached)}.",
        f"How large are J/Phi_L/scalars? max J=`{max_j}`, max Phi_L=`{max_phi}`, max |scalar|=`{max_scalar}`, max current L norm=`{max_linear}`.",
        f"Did v2 reduce reset width or only add output width? {reset_effect}; max reset=`{max_reset}`, max output-only symbolic=`{max_output}`.",
        f"Did sample containment pass? {sample_text}.",
        f"Is v2 conservative? {_yes_no(conservative)}; output range includes symbolic contribution for validated rows={_yes_no(output_conservative)}.",
        f"Flow* width ratios for best config: {width_ratio_text}.",
        f"Failure reason if still failed: `{failure}`.",
        "This is experimental clean-room queue propagation, not Flow* parity.",
        "",
        "## Best Config Metrics",
        "",
        "| run_id | status | last_validated_t | runtime_s | max_j | max_phi_l | max_target_check | max_output_only_symbolic | last_width_ratio | tube_width_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if best:
        best_segments = [row for row in v2_segments if row.get("run_id") == best.get("run_id")]
        lines.append(
            f"| {best.get('run_id', '')} | {best.get('status', '')} | {best.get('last_validated_t', '')} | "
            f"{best.get('runtime_s', '')} | {_max_field(best_segments, 'j_count')} | {_max_field(best_segments, 'phi_l_count')} | "
            f"{_max_field(best_segments, 'target_check_width_sum')} | {_max_field(best_segments, 'output_only_symbolic_width_sum')} | "
            f"{best_comp.get('last_width_ratio', '') if best_comp else ''} | {best_comp.get('tube_width_ratio', '') if best_comp else ''} |"
        )
    lines.extend([
        "",
        "## Config Status",
        "",
        "| run_id | status | last_validated_t | accepted | rejected | max_queue | max_output_only_symbolic | max_reset_width | conservative | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for row in _ordered_symqueue_v2_h10_rows(summary_rows):
        run_segments = [seg for seg in v2_segments if seg.get("run_id") == row.get("run_id")]
        validated_run_segments = [seg for seg in run_segments if seg.get("status") == "validated"]
        run_conservative = (
            all(_truthy(seg.get("output_range_includes_symbolic_contributions")) for seg in validated_run_segments)
            if validated_run_segments
            else False
        )
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('num_accepted_steps', '')} | {row.get('num_rejected_steps', '')} | {row.get('max_flowstar_queue_size_after', '')} | "
            f"{_max_field(run_segments, 'output_only_symbolic_width_sum')} | {_max_field(run_segments, 'reset_box_width_sum')} | "
            f"{_yes_no(run_conservative)} | {row.get('failure_reason', '')} |"
        )
    if sample_row is not None:
        lines.extend([
            "",
            "## Sample Containment",
            "",
            "| run_id | samples | checked_pairs | violations | max_outside_distance | status |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            f"| {sample_row.get('run_id', '')} | {sample_row.get('num_samples', '')} | {sample_row.get('checked_sample_time_pairs', '')} | "
            f"{sample_row.get('violations_count', '')} | {sample_row.get('max_outside_distance', '')} | {sample_row.get('status', '')} |",
        ])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "symqueue_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_symqueue_v2_h10_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_csv(out_dir / "symqueue_v2_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "symqueue_v2_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "symqueue_v2_reset_diagnostics.csv", NORMALIZED_INSERTION_RESET_FIELDS, _normalized_insertion_reset_rows(segment_rows))
    _write_csv(out_dir / "symqueue_v2_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    _write_csv(out_dir / "symqueue_v2_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
    write_symqueue_v2_h10_report(out_dir, summary_rows, segment_rows, comparison_rows, max_horizon=max_horizon)
    make_symqueue_v2_h10_plots(out_dir, segment_rows)


HORNER_EXTRA_FIELDS = [
    "horner_direct_range_width_sum",
    "horner_range_width_sum",
    "horner_direct_normal_range_width_sum",
    "horner_normal_range_width_sum",
    "horner_minus_direct_range_width_sum",
    "horner_minus_direct_normal_range_width_sum",
    "horner_reduced_range",
    "horner_reduced_normal_range",
    "horner_changed_range",
    "horner_stage_count",
    "horner_time_branch_stage_count",
    "horner_state_branch_stage_count",
    "horner_y_branch_stage_count",
    "horner_truncation_width_sum",
    "horner_cutoff_width_sum",
    "horner_outer_remainder_width_sum",
]


def _horner_reset_rows(segment_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    base_rows = _normalized_insertion_reset_rows(segment_rows)
    by_key = {(row.get("run_id"), row.get("segment_index")): row for row in segment_rows}
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        source = by_key.get((row.get("run_id"), row.get("segment_index")), {})
        rows.append({**row, **{field: source.get(field, "") for field in HORNER_EXTRA_FIELDS}})
    return rows


def _write_horner_h10_report(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    horner_rows = [row for row in summary_rows if row.get("reset_mode") == "normalized_insertion_horner"]
    best = _best(horner_rows)
    best_t = _finite_float(best.get("last_validated_t")) if best else 0.0
    reached = bool(best_t is not None and best_t >= float(max_horizon) - 1e-9)
    comp = _best_comparison_for_run(comparison_rows, str(best.get("run_id", ""))) if best else None
    o4_t = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in horner_rows if "o4" in str(row.get("run_id", ""))), default=0.0)
    o6_t = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in horner_rows if "o6" in str(row.get("run_id", ""))), default=0.0)
    validated_segments = [row for row in segment_rows if row.get("status") == "validated" and row.get("reset_mode") == "normalized_insertion_horner"]
    changed = any(str(row.get("horner_changed_range", "")).lower() == "true" for row in validated_segments)
    reduced = any(str(row.get("horner_reduced_range", "")).lower() == "true" for row in validated_segments)
    lines = [
        "# Horner Insertion H10 Report",
        "",
        f"Requested max horizon: `{float(max_horizon):.17g}`.",
        f"Best Horner insertion config: `{best.get('run_id', '') if best else ''}` at t=`{best_t}`.",
        f"Did Horner insertion beat o4 t~=6.473? {_yes_no(o4_t > 6.4730088058091901)}.",
        f"Did Horner insertion beat o6 t~=7.496? {_yes_no(o6_t > 7.4960392581387341)}.",
        f"Did any config reach h10? {_yes_no(reached)}.",
        f"Did width ratios improve? last=`{comp.get('last_width_ratio', '') if comp else ''}`, tube=`{comp.get('tube_width_ratio', '') if comp else ''}`.",
        "Did sample containment pass? see `sample_containment_summary.csv` if the sample containment command was run for this directory.",
        f"Runtime cost: best runtime_s=`{best.get('runtime_s', '') if best else ''}`.",
        f"Did Horner diagnostic materially change/reduce ranges in this run? changed={_yes_no(changed)}, reduced={_yes_no(reduced)}.",
        "If no improvement, the likely explanation is that this polynomial case is dominated by the same right-map range/scale channel or this clean-room Horner diagnostic is still incomplete relative to Flow* symbolic queue propagation.",
        "",
        "## Config Status",
        "",
        "| run_id | status | last_validated_t | accepted | rejected | final_width_sum | last_width_ratio | tube_width_ratio | failure_reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    comp_by_run = {str(row.get("run_id", "")): row for row in comparison_rows}
    for row in horner_rows:
        row_comp = comp_by_run.get(str(row.get("run_id", "")), {})
        lines.append(
            f"| {row.get('run_id', '')} | {row.get('status', '')} | {row.get('last_validated_t', '')} | "
            f"{row.get('num_accepted_steps', '')} | {row.get('num_rejected_steps', '')} | {row.get('final_width_sum', '')} | "
            f"{row_comp.get('last_width_ratio', '')} | {row_comp.get('tube_width_ratio', '')} | {row.get('failure_reason', '')} |"
        )
    (out_dir / "horner_h10_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _make_horner_stage_uncertainty_plot(out_dir: Path, segment_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in segment_rows:
        if row.get("status") == "validated" and row.get("reset_mode") == "normalized_insertion_horner":
            grouped.setdefault(str(row.get("run_id", "")), []).append(row)
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for run_id, rows in grouped.items():
        pts = []
        for row in sorted(rows, key=lambda r: _finite_float(r.get("t_hi")) or 0.0):
            t_hi = _finite_float(row.get("t_hi"))
            trunc = _finite_float(row.get("horner_truncation_width_sum")) or 0.0
            cutoff = _finite_float(row.get("horner_cutoff_width_sum")) or 0.0
            outer = _finite_float(row.get("horner_outer_remainder_width_sum")) or 0.0
            if t_hi is not None:
                pts.append((t_hi, trunc + cutoff + outer))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=1.0, label=run_id)
    ax.set_xlabel("t")
    ax.set_ylabel("Horner stage uncertainty width sum")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "horner_stage_uncertainty_vs_t.png", dpi=160)
    plt.close(fig)


def write_horner_h10_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    segment_fields = list(SEGMENT_FIELDS) + [field for field in HORNER_EXTRA_FIELDS if field not in SEGMENT_FIELDS]
    reset_fields = list(NORMALIZED_INSERTION_RESET_FIELDS) + [field for field in HORNER_EXTRA_FIELDS if field not in NORMALIZED_INSERTION_RESET_FIELDS]
    _write_csv(out_dir / "horner_h10_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "horner_h10_segments.csv", segment_fields, segment_rows)
    _write_csv(out_dir / "horner_h10_reset_diagnostics.csv", reset_fields, _horner_reset_rows(segment_rows))
    _write_csv(out_dir / "horner_h10_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    _write_csv(out_dir / "horner_h10_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
    _write_horner_h10_report(out_dir, summary_rows, segment_rows, comparison_rows, max_horizon=max_horizon)
    _copy_plot_if_present(out_dir, "width_ratio_vs_t.png", "horner_width_ratio_vs_t.png")
    _make_horner_stage_uncertainty_plot(out_dir, segment_rows)


def write_specialized_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    name = out_dir.name
    if name == "flowstar_style_rescue_adaptive_order":
        _write_csv(out_dir / "adaptive_order_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "adaptive_order_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_csv(out_dir / "adaptive_order_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
        _write_adaptive_order_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_rescue_remainder_sensitivity":
        rows = _summary_with_h5_baseline(summary_rows)
        _write_csv(out_dir / "remainder_sensitivity_summary.csv", SUMMARY_FIELDS, rows)
        _write_remainder_sensitivity_report(out_dir, rows, max_horizon=max_horizon)
    elif name == "flowstar_style_rescue_refined":
        _write_csv(out_dir / "refined_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_refined_report(out_dir, summary_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_candidate_order":
        _write_csv(out_dir / "candidate_order_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "candidate_order_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_candidate_order_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_truncation_range":
        _write_csv(out_dir / "truncation_range_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_truncation_range_report(out_dir, summary_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_residual_centering":
        _write_csv(out_dir / "residual_centering_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "residual_centering_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_csv(out_dir / "residual_centering_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
        _write_residual_centering_report(out_dir, summary_rows, attempt_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_style_selective_terms":
        _write_csv(out_dir / "selective_terms_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "selective_terms_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_selective_terms_report(out_dir, summary_rows, attempt_rows, comparison_rows, max_horizon=max_horizon)
        _write_csv(out_dir / "retained_terms_near_failure.csv", RETAINED_TERM_FIELDS, _retained_term_rows(segment_rows))
        _write_selective_validation_path_audit(out_dir, attempt_rows)
    elif name == "flowstar_style_ctrunc_validation":
        _write_csv(out_dir / "ctrunc_validation_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "ctrunc_validation_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_csv(out_dir / "ctrunc_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
        _write_ctrunc_validation_report(out_dir, summary_rows, attempt_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_width_control_rescue":
        _write_csv(out_dir / "width_control_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "width_control_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_csv(out_dir / "width_control_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
        _write_csv(out_dir / "width_control_reset_boxes.csv", RESET_BOX_FIELDS, _reset_box_rows(segment_rows))
        _write_csv(out_dir / "width_control_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
        _write_width_control_report(out_dir, summary_rows, segment_rows, comparison_rows, max_horizon=max_horizon)
    elif name == "flowstar_normalized_insertion_rescue":
        _write_csv(out_dir / "normalized_insertion_summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(out_dir / "normalized_insertion_segments.csv", SEGMENT_FIELDS, segment_rows)
        _write_csv(out_dir / "normalized_insertion_reset_diagnostics.csv", NORMALIZED_INSERTION_RESET_FIELDS, _normalized_insertion_reset_rows(segment_rows))
        _write_csv(out_dir / "normalized_insertion_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
        _write_csv(out_dir / "normalized_insertion_vs_flowstar_comparison.csv", COMPARISON_FIELDS, comparison_rows)
        _write_normalized_insertion_report(out_dir, summary_rows, segment_rows, comparison_rows, max_horizon=max_horizon)
        make_normalized_insertion_plots(out_dir, segment_rows)
    elif name == HORNER_H10_OUTPUT_DIR_NAME:
        write_horner_h10_outputs(
            out_dir,
            summary_rows,
            segment_rows,
            attempt_rows,
            max_horizon=max_horizon,
            comparison_rows=comparison_rows,
        )
    elif name == H10_OUTPUT_DIR_NAME:
        write_normalized_insertion_h10_outputs(
            out_dir,
            summary_rows,
            segment_rows,
            attempt_rows,
            max_horizon=max_horizon,
            comparison_rows=comparison_rows,
        )
    elif name == NORMAL_EVAL_H10_OUTPUT_DIR_NAME:
        write_normal_eval_h10_outputs(
            out_dir,
            summary_rows,
            segment_rows,
            attempt_rows,
            max_horizon=max_horizon,
            comparison_rows=comparison_rows,
        )
    elif name == SYMQUEUE_H10_OUTPUT_DIR_NAME:
        write_symqueue_h10_outputs(
            out_dir,
            summary_rows,
            segment_rows,
            attempt_rows,
            max_horizon=max_horizon,
            comparison_rows=comparison_rows,
        )
    elif name == SYMQUEUE_SPLIT_H10_OUTPUT_DIR_NAME:
        write_symqueue_split_h10_outputs(
            out_dir,
            summary_rows,
            segment_rows,
            attempt_rows,
            max_horizon=max_horizon,
            comparison_rows=comparison_rows,
        )
    elif name in {SYMQUEUE_V2_H10_OUTPUT_DIR_NAME, LEGACY_SYMQUEUE_V2_H10_OUTPUT_DIR_NAME}:
        write_symqueue_v2_h10_outputs(
            out_dir,
            summary_rows,
            segment_rows,
            attempt_rows,
            max_horizon=max_horizon,
            comparison_rows=comparison_rows,
        )


def _read_optional_csv(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(path) if path.exists() else []


def _variant_group(run_id: str) -> str:
    if "keep" in run_id and "centered" in run_id:
        return "selective_terms_centered"
    if "keep" in run_id:
        return "selective_high_degree_terms"
    if "centered" in run_id:
        return "residual_centering"
    if "residual_shift" in run_id:
        return "residual_shift_diagnostic"
    if "candidate8_output6" in run_id and "truncsplit" in run_id:
        return "candidate_order_truncation_split"
    if "candidate8_output6" in run_id:
        return "candidate_order_output_order"
    if "truncsplit" in run_id:
        return "truncation_range_split"
    if "adaptive_order_8" in run_id:
        return "adaptive_order_fallback"
    if "r2e-4" in run_id or "r5e-4" in run_id:
        return "relaxed_target_remainder"
    if "refined" in run_id:
        return "refined_target_validation"
    return "h5_current_best"


def write_rescue_next_outputs(*, trigger_out_dir: Path | None = None) -> None:
    if trigger_out_dir is not None:
        try:
            outputs_root = (REPO_ROOT / "outputs").resolve()
            if not trigger_out_dir.resolve().is_relative_to(outputs_root):
                return
        except Exception:
            return
    candidates: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Mapping[str, Any]] = {}
    sources = [
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "adaptive_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_remainder_sensitivity" / "remainder_sensitivity_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_remainder_sensitivity" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_refined" / "refined_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_refined" / "rescue_vs_flowstar_comparison.csv"),
    ]
    for summary_path, comparison_path in sources:
        for row in _read_optional_csv(summary_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                candidates[run_id] = dict(row)
        for row in _read_optional_csv(comparison_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                comparisons[run_id] = row
    if not candidates:
        return
    rows: list[dict[str, Any]] = []
    for run_id, row in sorted(candidates.items(), key=lambda item: (_variant_group(item[0]), item[0])):
        comp = comparisons.get(run_id, {})
        rows.append(
            {
                "variant_group": _variant_group(run_id),
                "run_id": run_id,
                "validation_mode": row.get("validation_mode", ""),
                "target_remainder_radius": row.get("target_remainder_radius", ""),
                "cutoff_threshold": row.get("cutoff_threshold", ""),
                "status": row.get("status", ""),
                "last_validated_t": row.get("last_validated_t", ""),
                "runtime_s": row.get("runtime_s", ""),
                "num_accepted_steps": row.get("num_accepted_steps", ""),
                "num_rejected_steps": row.get("num_rejected_steps", ""),
                "num_order8_steps": row.get("num_order8_steps", ""),
                "candidate_order": row.get("candidate_order", row.get("order", "")),
                "output_order": row.get("output_order", row.get("order", "")),
                "truncation_range_split": row.get("truncation_range_split", ""),
                "min_regular_h_used": row.get("min_regular_h_used", ""),
                "h_below_flowstar_min_count": row.get("h_below_flowstar_min_count", ""),
                "final_width_sum": row.get("final_width_sum", ""),
                "last_width_ratio": comp.get("last_width_ratio", ""),
                "tube_width_ratio": comp.get("tube_width_ratio", ""),
                "notes": row.get("notes", ""),
            }
        )
    out_dir = REPO_ROOT / "outputs" / "flowstar_style_rescue_next"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rescue_next_summary.csv", NEXT_FIELDS, rows)
    parity_rows = [
        row for row in rows
        if str(row.get("target_remainder_radius", "")) in {"0.0001", "1e-04", "1e-4"}
        and int(row.get("h_below_flowstar_min_count") or 0) == 0
    ]
    best = max(parity_rows or rows, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0)
    relaxed_reached = any(
        _variant_group(str(row.get("run_id", ""))) == "relaxed_target_remainder"
        and (_finite_float(row.get("last_validated_t")) or 0.0) >= 5.0 - 1e-9
        for row in rows
    )
    if (_finite_float(best.get("last_validated_t")) or 0.0) >= 5.0 - 1e-9:
        recommendation = "continue with the best parity-preserving variant and tighten polynomial range bounding."
    elif relaxed_reached:
        recommendation = "use relaxed target remainder only as a diagnostic; prioritize tighter polynomial range bounding or a symbolic remainder queue for parity."
    elif _variant_group(str(best.get("run_id", ""))) == "refined_target_validation":
        recommendation = "continue refined target validation, then add tighter polynomial range bounding."
    else:
        recommendation = "prioritize tighter polynomial range bounding, then a real Flow*-style symbolic remainder queue."
    lines = [
        "# Rescue Variant Comparison",
        "",
        f"Best variant by current decision criteria: `{best.get('run_id', '')}` at t=`{best.get('last_validated_t', '')}`.",
        f"Reached horizon 5? {_yes_no((_finite_float(best.get('last_validated_t')) or 0.0) >= 5.0 - 1e-9)}.",
        f"Width ratio vs Flow*: last=`{best.get('last_width_ratio', '')}`, tube=`{best.get('tube_width_ratio', '')}`.",
        f"Next recommendation: {recommendation}",
        "",
        "Decision criteria: highest last_validated_t, target remainder close to Flow* parameter, runtime, width ratio vs Flow*, and no non-final h below 0.002 except diagnostic runs.",
        "",
        "## Rows",
        "",
        "| group | run_id | status | last_validated_t | radius | last_width_ratio | tube_width_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant_group', '')} | {row.get('run_id', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('target_remainder_radius', '')} | "
            f"{row.get('last_width_ratio', '')} | {row.get('tube_width_ratio', '')} |"
        )
    (out_dir / "rescue_next_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")



def write_rescue_next2_outputs(*, trigger_out_dir: Path | None = None) -> None:
    if trigger_out_dir is not None:
        try:
            outputs_root = (REPO_ROOT / "outputs").resolve()
            if not trigger_out_dir.resolve().is_relative_to(outputs_root):
                return
        except Exception:
            return
    candidates: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Mapping[str, Any]] = {}
    sources = [
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "adaptive_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_candidate_order" / "candidate_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_candidate_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_truncation_range" / "truncation_range_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_truncation_range" / "rescue_vs_flowstar_comparison.csv"),
    ]
    for summary_path, comparison_path in sources:
        for row in _read_optional_csv(summary_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                candidates[run_id] = dict(row)
        for row in _read_optional_csv(comparison_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                comparisons[run_id] = row

    residual_rows = _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_style_residual_shift" / "residual_shift.csv")
    if residual_rows:
        row = residual_rows[-1]
        run_id = "residual_shift_diagnostic_y"
        candidates[run_id] = {
            "run_id": run_id,
            "validation_mode": "diagnostic_only",
            "target_remainder_radius": row.get("target_radius", "0.0001"),
            "cutoff_threshold": "",
            "status": "diagnostic_only",
            "last_validated_t": row.get("t_start", ""),
            "runtime_s": "",
            "num_accepted_steps": "",
            "num_rejected_steps": "",
            "num_order8_steps": "",
            "candidate_order": "",
            "output_order": "",
            "truncation_range_split": "",
            "min_regular_h_used": "",
            "h_below_flowstar_min_count": "",
            "final_width_sum": "",
            "notes": "diagnostic only; not an accepted run",
        }
    if not candidates:
        return

    rows: list[dict[str, Any]] = []
    for run_id, row in sorted(candidates.items(), key=lambda item: (_variant_group(item[0]), item[0])):
        comp = comparisons.get(run_id, {})
        rows.append(
            {
                "variant_group": _variant_group(run_id),
                "run_id": run_id,
                "validation_mode": row.get("validation_mode", ""),
                "target_remainder_radius": row.get("target_remainder_radius", ""),
                "cutoff_threshold": row.get("cutoff_threshold", ""),
                "status": row.get("status", ""),
                "last_validated_t": row.get("last_validated_t", ""),
                "runtime_s": row.get("runtime_s", ""),
                "num_accepted_steps": row.get("num_accepted_steps", ""),
                "num_rejected_steps": row.get("num_rejected_steps", ""),
                "num_order8_steps": row.get("num_order8_steps", ""),
                "candidate_order": row.get("candidate_order", row.get("order", "")),
                "output_order": row.get("output_order", row.get("order", "")),
                "truncation_range_split": row.get("truncation_range_split", ""),
                "min_regular_h_used": row.get("min_regular_h_used", ""),
                "h_below_flowstar_min_count": row.get("h_below_flowstar_min_count", ""),
                "final_width_sum": row.get("final_width_sum", ""),
                "last_width_ratio": comp.get("last_width_ratio", ""),
                "tube_width_ratio": comp.get("tube_width_ratio", ""),
                "notes": row.get("notes", ""),
            }
        )
    eligible = [
        row for row in rows
        if row.get("variant_group") != "residual_shift_diagnostic"
        and str(row.get("target_remainder_radius", "")) in {"0.0001", "0.000100000000000000", "1e-04", "1e-4", "0.00010000000000000000"}
        and int(row.get("h_below_flowstar_min_count") or 0) == 0
    ]
    best = max(eligible or [row for row in rows if row.get("variant_group") != "residual_shift_diagnostic"] or rows, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0)
    reached = (_finite_float(best.get("last_validated_t")) or 0.0) >= 5.0 - 1e-9
    trunc_rows = [row for row in rows if row.get("variant_group") in {"truncation_range_split", "candidate_order_truncation_split"}]
    trunc_best_t = max((_finite_float(row.get("last_validated_t")) or 0.0 for row in trunc_rows), default=0.0)
    if reached:
        recommendation = "run h10 next with the best horizon-5 variant, while keeping width and Flow* comparison checks enabled."
    elif trunc_best_t > 2.2771582567640953:
        recommendation = "continue tighter polynomial range bounding because it improved the validated horizon before moving to a symbolic remainder queue."
    else:
        recommendation = "move to a real Flow*-style symbolic remainder queue; the tested tighter range variants did not clear the bottleneck."

    out_dir = REPO_ROOT / "outputs" / "flowstar_style_rescue_next2"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rescue_next2_summary.csv", NEXT_FIELDS, rows)
    lines = [
        "# Rescue Variant Comparison Next2",
        "",
        f"Best variant by decision criteria: `{best.get('run_id', '')}` at t=`{best.get('last_validated_t', '')}`.",
        f"Reached horizon 5 with target_remainder_radius=1e-4? {_yes_no(reached)}.",
        f"Width ratio vs Flow*: last=`{best.get('last_width_ratio', '')}`, tube=`{best.get('tube_width_ratio', '')}`.",
        f"Next recommendation: {recommendation}",
        "",
        "Residual-shift rows are diagnostic only and are not treated as accepted reachability runs.",
        "Decision criteria: reaches horizon 5, no non-final h below 0.002, width ratio not worse than adaptive fallback, runtime, and no fake parity claims.",
        "",
        "## Rows",
        "",
        "| group | run_id | status | last_validated_t | candidate_order | output_order | split | last_width_ratio | tube_width_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant_group', '')} | {row.get('run_id', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('candidate_order', '')} | {row.get('output_order', '')} | "
            f"{row.get('truncation_range_split', '')} | {row.get('last_width_ratio', '')} | {row.get('tube_width_ratio', '')} |"
        )
    (out_dir / "rescue_next2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")



def write_rescue_next3_outputs(*, trigger_out_dir: Path | None = None) -> None:
    if trigger_out_dir is not None:
        try:
            outputs_root = (REPO_ROOT / "outputs").resolve()
            if not trigger_out_dir.resolve().is_relative_to(outputs_root):
                return
        except Exception:
            return
    candidates: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Mapping[str, Any]] = {}
    sources = [
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_h5" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "adaptive_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_rescue_adaptive_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_candidate_order" / "candidate_order_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_candidate_order" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_truncation_range" / "truncation_range_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_truncation_range" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_residual_centering" / "residual_centering_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_residual_centering" / "rescue_vs_flowstar_comparison.csv"),
        (REPO_ROOT / "outputs" / "flowstar_style_selective_terms" / "selective_terms_summary.csv", REPO_ROOT / "outputs" / "flowstar_style_selective_terms" / "rescue_vs_flowstar_comparison.csv"),
    ]
    for summary_path, comparison_path in sources:
        for row in _read_optional_csv(summary_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                candidates[run_id] = dict(row)
        for row in _read_optional_csv(comparison_path):
            run_id = str(row.get("run_id", ""))
            if run_id:
                comparisons[run_id] = row
    if not candidates:
        return

    rows: list[dict[str, Any]] = []
    for run_id, row in sorted(candidates.items(), key=lambda item: (_variant_group(item[0]), item[0])):
        comp = comparisons.get(run_id, {})
        rows.append(
            {
                "variant_group": _variant_group(run_id),
                "run_id": run_id,
                "validation_mode": row.get("validation_mode", ""),
                "target_remainder_radius": row.get("target_remainder_radius", ""),
                "cutoff_threshold": row.get("cutoff_threshold", ""),
                "status": row.get("status", ""),
                "last_validated_t": row.get("last_validated_t", ""),
                "runtime_s": row.get("runtime_s", ""),
                "num_accepted_steps": row.get("num_accepted_steps", ""),
                "num_rejected_steps": row.get("num_rejected_steps", ""),
                "num_order8_steps": row.get("num_order8_steps", ""),
                "candidate_order": row.get("candidate_order", row.get("order", "")),
                "output_order": row.get("output_order", row.get("order", "")),
                "truncation_range_split": row.get("truncation_range_split", ""),
                "center_corrections_applied": row.get("center_corrections_applied", ""),
                "center_corrected_dimensions": row.get("center_corrected_dimensions", ""),
                "max_center_correction_abs": row.get("max_center_correction_abs", ""),
                "selective_high_degree_terms_top_k": row.get("selective_high_degree_terms_top_k", ""),
                "max_selective_retained_terms_count": row.get("max_selective_retained_terms_count", ""),
                "max_selective_dropped_remainder_width_sum": row.get("max_selective_dropped_remainder_width_sum", ""),
                "min_regular_h_used": row.get("min_regular_h_used", ""),
                "h_below_flowstar_min_count": row.get("h_below_flowstar_min_count", ""),
                "final_width_sum": row.get("final_width_sum", ""),
                "last_width_ratio": comp.get("last_width_ratio", ""),
                "tube_width_ratio": comp.get("tube_width_ratio", ""),
                "notes": row.get("notes", ""),
            }
        )
    eligible = [
        row for row in rows
        if str(row.get("target_remainder_radius", "")) in {"0.0001", "0.000100000000000000", "1e-04", "1e-4", "0.00010000000000000000"}
        and int(row.get("h_below_flowstar_min_count") or 0) == 0
    ]
    best = max(eligible or rows, key=lambda r: (_finite_float(r.get("last_validated_t")) or 0.0, -(_finite_float(r.get("tube_width_ratio")) or math.inf)))
    reached = (_finite_float(best.get("last_validated_t")) or 0.0) >= 5.0 - 1e-9
    candidate_baseline = next((row for row in rows if row.get("run_id") == "flowstar_style_o6_candidate8_output6"), {})
    candidate_tube = _finite_float(candidate_baseline.get("tube_width_ratio"))
    best_tube = _finite_float(best.get("tube_width_ratio"))
    width_ok = best_tube is None or candidate_tube is None or best_tube <= candidate_tube or reached
    if reached:
        recommendation = "run h10 only after reviewing the horizon-5 width and Flow* comparison artifacts."
    elif _variant_group(str(best.get("run_id", ""))) in {"residual_centering", "selective_terms_centered"}:
        recommendation = "continue residual-centering refinement or selective sparse over-order terms before h10."
    elif _variant_group(str(best.get("run_id", ""))) == "selective_high_degree_terms":
        recommendation = "continue selective sparse over-order terms, then compare against a real Flow*-style symbolic remainder queue."
    else:
        recommendation = "choose between residual-centering refinement, selective sparse over-order terms, or a real Flow*-style symbolic remainder queue."

    out_dir = REPO_ROOT / "outputs" / "flowstar_style_rescue_next3"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rescue_next3_summary.csv", NEXT3_FIELDS, rows)
    lines = [
        "# Rescue Variant Comparison Next3",
        "",
        f"Best variant by decision criteria: `{best.get('run_id', '')}` at t=`{best.get('last_validated_t', '')}`.",
        f"Reached horizon 5 with target_remainder_radius=1e-4? {_yes_no(reached)}.",
        f"Width ratio vs Flow*: last=`{best.get('last_width_ratio', '')}`, tube=`{best.get('tube_width_ratio', '')}`.",
        f"Width criterion vs candidate_order baseline acceptable? {_yes_no(width_ok)}.",
        f"Target remainder stayed at 1e-4? {_yes_no(str(best.get('target_remainder_radius', '')) in {'0.0001', '0.000100000000000000', '1e-04', '1e-4', '0.00010000000000000000'})}.",
        f"Next recommendation: {recommendation}",
        "",
        "This comparison is diagnostic-only and does not claim Flow* parity.",
        "Decision criteria: reaches horizon 5, no non-final h below 0.002, target remainder 1e-4, width ratio not worse than candidate_order baseline unless horizon improves substantially, and acceptable runtime.",
        "",
        "## Rows",
        "",
        "| group | run_id | status | last_validated_t | candidate_order | output_order | K | corrections | last_width_ratio | tube_width_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant_group', '')} | {row.get('run_id', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('candidate_order', '')} | {row.get('output_order', '')} | "
            f"{row.get('selective_high_degree_terms_top_k', '')} | {row.get('center_corrections_applied', '')} | "
            f"{row.get('last_width_ratio', '')} | {row.get('tube_width_ratio', '')} |"
        )
    (out_dir / "rescue_next3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

def _bool_from_row(row: Mapping[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes", "validated", "completed"}


def write_rescue_next4_outputs(*, trigger_out_dir: Path | None = None) -> None:
    if trigger_out_dir is not None:
        try:
            outputs_root = (REPO_ROOT / "outputs").resolve()
            if not trigger_out_dir.resolve().is_relative_to(outputs_root):
                return
        except Exception:
            return
    previous_rows = _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_style_candidate_order" / "candidate_order_summary.csv")
    ctrunc_rows = _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_style_ctrunc_validation" / "ctrunc_validation_summary.csv")
    selective_rows = _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_style_selective_terms" / "selective_terms_summary.csv")
    oracle_rows = _read_optional_csv(REPO_ROOT / "outputs" / "flowstar_one_step_oracle" / "oracle_summary.csv")

    previous_best = max(previous_rows, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0, default={})
    ctrunc_best = max(ctrunc_rows, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0, default={})
    selective_best = max(selective_rows, key=lambda r: _finite_float(r.get("last_validated_t")) or 0.0, default={})

    def _oracle_order(row: Mapping[str, Any]) -> int:
        value = _finite_float(row.get("order"))
        return int(value) if value is not None else 0

    def _oracle_rollup(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        # New oracle_summary.csv is one row per Flow* order. Older committed
        # artifacts had one aggregate row, so retain a compatibility path.
        if "order" not in rows[0]:
            return dict(rows[0])
        validated_rows = [row for row in rows if _bool_from_row(row, "flowstar_validated")]
        selected = min(validated_rows, key=_oracle_order) if validated_rows else max(rows, key=_oracle_order)
        flowstar_validated = bool(validated_rows)
        pytorch_validated = any(_bool_from_row(row, "pytorch_validated") for row in rows)
        status = "completed" if flowstar_validated else str(selected.get("flowstar_status", ""))
        failure_reason = ""
        if not flowstar_validated:
            failure_reason = str(selected.get("skip_reason", ""))
            if not failure_reason and status == "not_completed":
                failure_reason = "Flow* reach did not complete; no segment boxes emitted"
            elif not failure_reason:
                failure_reason = status
        return {
            "run_id": "flowstar_one_step_oracle_candidate8_cutoff",
            "status": status,
            "flowstar_validated": flowstar_validated,
            "pytorch_validated": pytorch_validated,
            "runtime_s": selected.get("flowstar_runtime_s", ""),
            "failure_reason": failure_reason,
            "decision_relevance": "local same-box diagnostic",
            "notes": "local one-step diagnostic only; no full parity claim",
            "flowstar_best_order": selected.get("order", ""),
            "flowstar_segments": selected.get("flowstar_segments", ""),
        }

    oracle = _oracle_rollup(oracle_rows)

    rows: list[dict[str, Any]] = []
    if previous_best:
        rows.append(
            {
                "comparison_item": "previous_best_candidate_order",
                "run_id": previous_best.get("run_id", ""),
                "status": previous_best.get("status", ""),
                "last_validated_t": previous_best.get("last_validated_t", ""),
                "runtime_s": previous_best.get("runtime_s", ""),
                "failure_reason": previous_best.get("failure_reason", ""),
                "decision_relevance": "baseline t~=2.400737",
                "notes": previous_best.get("notes", ""),
            }
        )
    if oracle:
        rows.append(
            {
                "comparison_item": "one_step_oracle",
                "run_id": oracle.get("run_id", "flowstar_one_step_oracle"),
                "status": oracle.get("status", ""),
                "flowstar_validated": oracle.get("flowstar_validated", ""),
                "pytorch_validated": oracle.get("pytorch_validated", ""),
                "runtime_s": oracle.get("runtime_s", ""),
                "failure_reason": oracle.get("failure_reason", ""),
                "decision_relevance": "local same-box diagnostic",
                "notes": oracle.get("notes", ""),
            }
        )
    if ctrunc_best:
        rows.append(
            {
                "comparison_item": "flowstar_ctrunc_validation",
                "run_id": ctrunc_best.get("run_id", ""),
                "status": ctrunc_best.get("status", ""),
                "last_validated_t": ctrunc_best.get("last_validated_t", ""),
                "runtime_s": ctrunc_best.get("runtime_s", ""),
                "failure_reason": ctrunc_best.get("failure_reason", ""),
                "decision_relevance": "new validation mode",
                "notes": ctrunc_best.get("notes", ""),
            }
        )
    if selective_best:
        rows.append(
            {
                "comparison_item": "selective_validation_path",
                "run_id": selective_best.get("run_id", ""),
                "status": selective_best.get("status", ""),
                "last_validated_t": selective_best.get("last_validated_t", ""),
                "runtime_s": selective_best.get("runtime_s", ""),
                "failure_reason": selective_best.get("failure_reason", ""),
                "decision_relevance": "keepK validation-path audit/fix",
                "notes": selective_best.get("notes", ""),
            }
        )
    if not rows:
        return

    previous_t = _finite_float(previous_best.get("last_validated_t")) or 2.400737667399793
    ctrunc_t = _finite_float(ctrunc_best.get("last_validated_t")) or 0.0
    selective_t = _finite_float(selective_best.get("last_validated_t")) or 0.0
    oracle_flowstar_validated = _bool_from_row(oracle, "flowstar_validated")
    oracle_ran = bool(oracle) and str(oracle.get("status", "")) not in {"", "skipped", "compile_failed", "compile_timeout"}
    if ctrunc_t >= 5.0 - 1e-9:
        decision = "ctrunc validation reached horizon 5; run h10 next after width review."
    elif oracle and oracle_flowstar_validated and ctrunc_t < 5.0 - 1e-9:
        decision = "Flow* one-step validates but PyTorch ctrunc does not reach horizon 5; continue source archaeology for normal eval, symbolic remainder, or preconditioning details."
    elif oracle and oracle_ran and not oracle_flowstar_validated:
        decision = "Flow* one-step also fails from the PyTorch reset box; focus on width reduction before that point."
    elif oracle and not oracle_ran:
        decision = "Flow* one-step did not run; rerun the oracle before drawing a kernel conclusion."
    elif selective_t > previous_t:
        decision = "Selective validation-path fix helps; combine it with ctrunc validation."
    else:
        decision = "Nothing beats t~=2.400737 yet; next target is a real Flow*-style symbolic remainder queue."

    branch_decision = "NEEDS_MORE_WORK"
    if not oracle_rows and not ctrunc_rows and not selective_rows:
        branch_decision = "DISCARD_BRANCH"

    out_dir = REPO_ROOT / "outputs" / "flowstar_style_rescue_next4"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rescue_next4_summary.csv", NEXT4_FIELDS, rows)
    lines = [
        "# Rescue Variant Comparison Next4",
        "",
        f"Previous best candidate_order=8/output_order=6: `{previous_best.get('run_id', '')}` at t=`{previous_best.get('last_validated_t', '')}`.",
        f"Did Flow* one-step actually run? {_yes_no(oracle_ran)}.",
        f"One-step oracle Flow* validates same local box? {_yes_no(oracle_flowstar_validated)}.",
        f"Best flowstar_ctrunc validation: `{ctrunc_best.get('run_id', '')}` at t=`{ctrunc_best.get('last_validated_t', '')}`.",
        f"Best selective validation-path run: `{selective_best.get('run_id', '')}` at t=`{selective_best.get('last_validated_t', '')}`.",
        f"Branch decision: {branch_decision}.",
        f"Decision: {decision}",
        "",
        "## Rows",
        "",
        "| item | run_id | status | last_validated_t | flowstar_validated | pytorch_validated | notes |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('comparison_item', '')} | {row.get('run_id', '')} | {row.get('status', '')} | "
            f"{row.get('last_validated_t', '')} | {row.get('flowstar_validated', '')} | {row.get('pytorch_validated', '')} | {row.get('notes', '')} |"
        )
    (out_dir / "rescue_next4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    triage_lines = [
        "# Branch Triage Report",
        "",
        f"Branch decision: {branch_decision}.",
        "",
        "## Evidence",
        "",
        f"Did Flow* one-step actually run? {_yes_no(oracle_ran)}.",
        f"Did Flow* validate the local failed step? {_yes_no(oracle_flowstar_validated)}.",
        f"Previous best: `{previous_best.get('run_id', '')}` at t=`{previous_best.get('last_validated_t', '')}`.",
        f"Ctrunc best: `{ctrunc_best.get('run_id', '')}` at t=`{ctrunc_best.get('last_validated_t', '')}`.",
        f"Selective best: `{selective_best.get('run_id', '')}` at t=`{selective_best.get('last_validated_t', '')}`.",
        f"Did any variant reach horizon 5? {_yes_no(bool(ctrunc_t >= 5.0 - 1e-9 or previous_t >= 5.0 - 1e-9 or selective_t >= 5.0 - 1e-9))}.",
        "",
        "## Recommendation",
        "",
        f"Next recommendation: {decision}",
    ]
    (out_dir / "branch_triage_report.md").write_text("\n".join(triage_lines) + "\n", encoding="utf-8")

def _write_outputs(
    out_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    attempt_rows: Sequence[Mapping[str, Any]],
    *,
    max_horizon: float,
) -> None:
    _write_csv(out_dir / "rescue_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(out_dir / "rescue_segments.csv", SEGMENT_FIELDS, segment_rows)
    _write_csv(out_dir / "rescue_validation_attempts.csv", VALIDATION_ATTEMPT_FIELDS, attempt_rows)
    _write_csv(out_dir / "rescue_reset_boxes.csv", RESET_BOX_FIELDS, _reset_box_rows(segment_rows))
    write_report(out_dir, summary_rows, segment_rows, max_horizon=max_horizon)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/flowstar_style_rescue"))
    parser.add_argument("--max-horizon", type=float, default=1.0)
    parser.add_argument("--wall-cap-s", type=float, default=300.0)
    parser.add_argument("--configs", nargs="*", default=None, help="Run only selected config run_id values.")
    args = parser.parse_args(argv)
    run_experiment(
        args.out_dir,
        max_horizon=float(args.max_horizon),
        wall_cap_s=float(args.wall_cap_s),
        config_ids=args.configs,
    )
    print(f"wrote rescue outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
