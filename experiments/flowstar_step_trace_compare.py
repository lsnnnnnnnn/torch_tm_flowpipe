"""Generate Flow* and PyTorch accepted-step traces for Van der Pol diagnostics.

This is intentionally a local diagnostic runner, not a new reachability mode.
It compiles a repo-local C++ probe against the local Flow* toolbox and converts
existing PyTorch normalized-insertion diagnostics into the same step schema.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import FlowstarNormalFlowpipeState, Interval, TMVector, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode

PROBE_CPP = ROOT / "experiments" / "flowstar_probe" / "flowstar_vdp_step_trace_probe.cpp"
DEFAULT_OUT = ROOT / "outputs" / "flowstar_step_trace_compare"

SAME_SOURCE_FIELDS = [
    "flowstar_full_step_tube_source_object",
    "flowstar_full_step_tube_domain_semantics",
    "flowstar_full_step_tube_x_lo",
    "flowstar_full_step_tube_x_hi",
    "flowstar_full_step_tube_y_lo",
    "flowstar_full_step_tube_y_hi",
    "flowstar_full_step_tube_includes_cutoff_poly_diff",
    "flowstar_full_step_tube_includes_target_remainder",
    "flowstar_full_step_tube_includes_ordinary_remainder",
    "flowstar_full_step_tube_includes_symbolic_output_width",
    "flowstar_tau_h_endpoint_source_object",
    "flowstar_tau_h_endpoint_domain_semantics",
    "flowstar_tau_h_endpoint_x_lo",
    "flowstar_tau_h_endpoint_x_hi",
    "flowstar_tau_h_endpoint_y_lo",
    "flowstar_tau_h_endpoint_y_hi",
    "flowstar_tau_h_endpoint_includes_cutoff_poly_diff",
    "flowstar_tau_h_endpoint_includes_target_remainder",
    "flowstar_tau_h_endpoint_includes_ordinary_remainder",
    "flowstar_tau_h_endpoint_includes_symbolic_output_width",
    "torch_full_step_validation_candidate_source_object",
    "torch_full_step_validation_candidate_domain_semantics",
    "torch_full_step_validation_candidate_x_lo",
    "torch_full_step_validation_candidate_x_hi",
    "torch_full_step_validation_candidate_y_lo",
    "torch_full_step_validation_candidate_y_hi",
    "torch_full_step_validation_candidate_includes_cutoff_poly_diff",
    "torch_full_step_validation_candidate_includes_target_remainder",
    "torch_full_step_validation_candidate_includes_ordinary_remainder",
    "torch_full_step_validation_candidate_includes_symbolic_output_width",
    "torch_tau_h_endpoint_source_object",
    "torch_tau_h_endpoint_domain_semantics",
    "torch_tau_h_endpoint_x_lo",
    "torch_tau_h_endpoint_x_hi",
    "torch_tau_h_endpoint_y_lo",
    "torch_tau_h_endpoint_y_hi",
    "torch_tau_h_endpoint_includes_cutoff_poly_diff",
    "torch_tau_h_endpoint_includes_target_remainder",
    "torch_tau_h_endpoint_includes_ordinary_remainder",
    "torch_tau_h_endpoint_includes_symbolic_output_width",
]

LIFECYCLE_FIELDS = [
    "pre_step_box_x_lo",
    "pre_step_box_x_hi",
    "pre_step_box_y_lo",
    "pre_step_box_y_hi",
    "endpoint_box_before_center_x_lo",
    "endpoint_box_before_center_x_hi",
    "endpoint_box_before_center_y_lo",
    "endpoint_box_before_center_y_hi",
    "endpoint_before_center_source_object",
    "endpoint_before_center_domain_semantics",
    "endpoint_before_center_includes_target_remainder",
    "endpoint_before_center_includes_ordinary_remainder",
    "endpoint_before_center_includes_symbolic_output_width",
    "endpoint_before_center_includes_cutoff_poly_diff",
    "endpoint_before_center_range_eval_method",
    "endpoint_before_center_polynomial_order",
    "endpoint_before_center_dropped_terms_width_x",
    "endpoint_before_center_dropped_terms_width_y",
    "endpoint_before_center_dropped_terms_width_sum",
    "endpoint_before_center_remainder_width_x",
    "endpoint_before_center_remainder_width_y",
    "endpoint_before_center_remainder_width_sum",
    "endpoint_before_center_notes",
    "extracted_center_x",
    "extracted_center_y",
    "extracted_scale_x",
    "extracted_scale_y",
    "reset_box_after_center_scale_x_lo",
    "reset_box_after_center_scale_x_hi",
    "reset_box_after_center_scale_y_lo",
    "reset_box_after_center_scale_y_hi",
    "target_remainder_x_lo",
    "target_remainder_x_hi",
    "target_remainder_y_lo",
    "target_remainder_y_hi",
    "polynomial_range_x_lo",
    "polynomial_range_x_hi",
    "polynomial_range_y_lo",
    "polynomial_range_y_hi",
    "ordinary_remainder_x_lo",
    "ordinary_remainder_x_hi",
    "ordinary_remainder_y_lo",
    "ordinary_remainder_y_hi",
    "raw_ctrunc_residual_x_lo",
    "raw_ctrunc_residual_x_hi",
    "raw_ctrunc_residual_y_lo",
    "raw_ctrunc_residual_y_hi",
    "raw_ctrunc_polynomial_range_x_lo",
    "raw_ctrunc_polynomial_range_x_hi",
    "raw_ctrunc_polynomial_range_y_lo",
    "raw_ctrunc_polynomial_range_y_hi",
    "raw_ctrunc_remainder_x_lo",
    "raw_ctrunc_remainder_x_hi",
    "raw_ctrunc_remainder_y_lo",
    "raw_ctrunc_remainder_y_hi",
    "raw_remainder_dropped_terms_range_x_lo",
    "raw_remainder_dropped_terms_range_x_hi",
    "raw_remainder_dropped_terms_range_y_lo",
    "raw_remainder_dropped_terms_range_y_hi",
    "raw_remainder_multiplication_remainder_x_lo",
    "raw_remainder_multiplication_remainder_x_hi",
    "raw_remainder_multiplication_remainder_y_lo",
    "raw_remainder_multiplication_remainder_y_hi",
    "raw_remainder_integration_remainder_x_lo",
    "raw_remainder_integration_remainder_x_hi",
    "raw_remainder_integration_remainder_y_lo",
    "raw_remainder_integration_remainder_y_hi",
    "raw_remainder_before_accumulation_x_lo",
    "raw_remainder_before_accumulation_x_hi",
    "raw_remainder_before_accumulation_y_lo",
    "raw_remainder_before_accumulation_y_hi",
    "raw_remainder_after_integration_x_lo",
    "raw_remainder_after_integration_x_hi",
    "raw_remainder_after_integration_y_lo",
    "raw_remainder_after_integration_y_hi",
    "raw_remainder_after_dropped_terms_x_lo",
    "raw_remainder_after_dropped_terms_x_hi",
    "raw_remainder_after_dropped_terms_y_lo",
    "raw_remainder_after_dropped_terms_y_hi",
    "raw_remainder_after_cutoff_x_lo",
    "raw_remainder_after_cutoff_x_hi",
    "raw_remainder_after_cutoff_y_lo",
    "raw_remainder_after_cutoff_y_hi",
    "raw_remainder_before_poly_diff_x_lo",
    "raw_remainder_before_poly_diff_x_hi",
    "raw_remainder_before_poly_diff_y_lo",
    "raw_remainder_before_poly_diff_y_hi",
    "raw_remainder_after_poly_diff_x_lo",
    "raw_remainder_after_poly_diff_x_hi",
    "raw_remainder_after_poly_diff_y_lo",
    "raw_remainder_after_poly_diff_y_hi",
    "raw_remainder_range_enclosure_method",
    "raw_remainder_normal_domain_scaling",
    "raw_remainder_partition_missing_reason",
    "raw_ctrunc_residual_source_object",
    "raw_ctrunc_residual_domain_semantics",
    "raw_ctrunc_residual_includes_target_remainder",
    "raw_ctrunc_residual_includes_ordinary_remainder",
    "raw_ctrunc_residual_includes_cutoff_poly_diff",
    "raw_ctrunc_residual_added_component",
    "raw_ctrunc_residual_notes",
    "picard_no_remainder_range_x_lo",
    "picard_no_remainder_range_x_hi",
    "picard_no_remainder_range_y_lo",
    "picard_no_remainder_range_y_hi",
    "picard_no_remainder_polynomial_range_x_lo",
    "picard_no_remainder_polynomial_range_x_hi",
    "picard_no_remainder_polynomial_range_y_lo",
    "picard_no_remainder_polynomial_range_y_hi",
    "picard_no_remainder_remainder_x_lo",
    "picard_no_remainder_remainder_x_hi",
    "picard_no_remainder_remainder_y_lo",
    "picard_no_remainder_remainder_y_hi",
    "target_remainder_before_ctrunc_x_lo",
    "target_remainder_before_ctrunc_x_hi",
    "target_remainder_before_ctrunc_y_lo",
    "target_remainder_before_ctrunc_y_hi",
    "ordinary_remainder_missing_reason",
    "picard_no_remainder_residual_x_lo",
    "picard_no_remainder_residual_x_hi",
    "picard_no_remainder_residual_y_lo",
    "picard_no_remainder_residual_y_hi",
    "picard_ctrunc_raw_residual_x_lo",
    "picard_ctrunc_raw_residual_x_hi",
    "picard_ctrunc_raw_residual_y_lo",
    "picard_ctrunc_raw_residual_y_hi",
    "cutoff_poly_diff_x_lo",
    "cutoff_poly_diff_x_hi",
    "cutoff_poly_diff_y_lo",
    "cutoff_poly_diff_y_hi",
    "cutoff_polynomial_difference_x_lo",
    "cutoff_polynomial_difference_x_hi",
    "cutoff_polynomial_difference_y_lo",
    "cutoff_polynomial_difference_y_hi",
    "cutoff_polynomial_difference_x_width",
    "cutoff_polynomial_difference_y_width",
    "post_cutoff_residual_x_lo",
    "post_cutoff_residual_x_hi",
    "post_cutoff_residual_y_lo",
    "post_cutoff_residual_y_hi",
    *SAME_SOURCE_FIELDS,
]

TRACE_FIELDS = [
    "trace_source",
    "source",
    "mode",
    "attempt_global_index",
    "accepted_step_index",
    "step_index",
    "attempt_index_within_step",
    "adaptive_attempt_index",
    "t_before",
    "h_try",
    "h",
    "t_after",
    "h_after_if_rejected_or_next",
    "accepted",
    "rejected",
    "status",
    "rejection_reason",
    "message",
    "residual_subset_target",
    *LIFECYCLE_FIELDS,
    "target_check_width_x",
    "target_check_width_y",
    "target_check_width_sum",
    "ordinary_step_remainder_width_x",
    "ordinary_step_remainder_width_y",
    "ordinary_step_remainder_width_sum",
    "right_map_range_width_x",
    "right_map_range_width_y",
    "right_map_range_width_sum",
    "reset_width_x",
    "reset_width_y",
    "reset_width_sum",
    "output_range_width_x",
    "output_range_width_y",
    "output_range_width_sum",
    "final_segment_width_x",
    "final_segment_width_y",
    "final_segment_width_sum",
    "output_only_symbolic_width_x",
    "output_only_symbolic_width_y",
    "output_only_symbolic_width_sum",
    "queue_size",
    "j_count",
    "phi_l_count",
    "tmv_pre_range_width_x",
    "tmv_pre_range_width_y",
    "tmv_pre_range_width_sum",
    "tmv_right_range_width_x",
    "tmv_right_range_width_y",
    "tmv_right_range_width_sum",
    "tmv_right_normal_range_width_x",
    "tmv_right_normal_range_width_y",
    "tmv_right_normal_range_width_sum",
    "endpoint_pre_center_width_x",
    "endpoint_pre_center_width_y",
    "endpoint_pre_center_width_sum",
    "center_x",
    "center_y",
    "scale_x",
    "scale_y",
    "inv_scale_x",
    "inv_scale_y",
    "new_x0_width_x",
    "new_x0_width_y",
    "new_x0_width_sum",
    "target_remainder_width_x",
    "target_remainder_width_y",
    "target_remainder_width_sum",
    "picard_no_remainder_residual_width_x",
    "picard_no_remainder_residual_width_y",
    "picard_no_remainder_residual_width_sum",
    "picard_ctrunc_normal_residual_width_x",
    "picard_ctrunc_normal_residual_width_y",
    "picard_ctrunc_normal_residual_width_sum",
    "cutoff_polynomial_difference_width_x",
    "cutoff_polynomial_difference_width_y",
    "cutoff_polynomial_difference_width_sum",
    "symbolic_J_size",
    "symbolic_Phi_L_size",
    "scalar_x",
    "scalar_y",
    "symbolic_J_width_x",
    "symbolic_J_width_y",
    "symbolic_J_width_sum",
    "symbolic_propagated_width_x",
    "symbolic_propagated_width_y",
    "symbolic_propagated_width_sum",
    "final_flowpipe_width_x",
    "final_flowpipe_width_y",
    "final_flowpipe_width_sum",
    "residual_width_x",
    "residual_width_y",
    "residual_width_sum",
    "target_remainder_lo_x",
    "target_remainder_hi_x",
    "target_remainder_lo_y",
    "target_remainder_hi_y",
    "picard_ctrunc_normal_residual_lo_x",
    "picard_ctrunc_normal_residual_hi_x",
    "picard_ctrunc_normal_residual_lo_y",
    "picard_ctrunc_normal_residual_hi_y",
    "residual_lo_x",
    "residual_hi_x",
    "residual_lo_y",
    "residual_hi_y",
    "residual_over_target_x",
    "residual_over_target_y",
    "residual_over_target_sum",
]

DIFF_FIELDS = [
    "step_index",
    "t_flowstar",
    "t_noqueue",
    "t_v2",
    "flowstar_h",
    "noqueue_h",
    "v2_h",
    "flowstar_width_sum",
    "noqueue_width_sum",
    "v2_width_sum",
    "noqueue_width_ratio",
    "v2_width_ratio",
    "flowstar_residual_sum",
    "noqueue_residual_sum",
    "v2_residual_sum",
    "noqueue_residual_ratio",
    "v2_residual_ratio",
    "center_scale_delta_noqueue",
    "center_scale_delta_v2",
    "inserted_endpoint_ratio_noqueue",
    "inserted_endpoint_ratio_v2",
    "right_map_ratio_noqueue",
    "right_map_ratio_v2",
    "target_remainder_ratio_noqueue",
    "target_remainder_ratio_v2",
    "picard_residual_ratio_noqueue",
    "picard_residual_ratio_v2",
    "cutoff_poly_ratio_noqueue",
    "cutoff_poly_ratio_v2",
    "symbolic_queue_ratio_v2",
    "first_material_channel",
    "channel_attribution_valid",
    "comparison_kind",
    "alignment_warning",
    "notes",
]

STEP_ALIGNMENT_WARNING_FIELDS = [
    "step_index",
    "t_flowstar",
    "t_noqueue",
    "t_v2",
    "flowstar_h",
    "noqueue_h",
    "v2_h",
    "first_material_channel",
    "channel_attribution_valid",
    "comparison_kind",
    "alignment_warning",
]

ATTEMPT_ALIGNED_FIELDS = [
    "aligned_index",
    "t_before",
    "h_try",
    "flowstar_status",
    "torch_noqueue_status",
    "torch_v2_status",
    "first_status_divergence",
    "first_numeric_channel_divergence",
    "channel_attribution_valid",
    "flowstar_rejection_reason",
    "torch_noqueue_rejection_reason",
    "torch_v2_rejection_reason",
    "residual_ratio_noqueue_over_flowstar",
    "residual_ratio_v2_over_flowstar",
    "target_width_ratio_noqueue_over_flowstar",
    "target_width_ratio_v2_over_flowstar",
    "right_map_width_ratio_noqueue_over_flowstar",
    "right_map_width_ratio_v2_over_flowstar",
    "reset_width_ratio_noqueue_over_flowstar",
    "reset_width_ratio_v2_over_flowstar",
    "output_range_width_ratio_noqueue_over_flowstar",
    "output_range_width_ratio_v2_over_flowstar",
    "center_delta_noqueue",
    "center_delta_v2",
    "scale_delta_noqueue",
    "scale_delta_v2",
    "verdict",
]

FORCED_H_FIELDS = [
    "forced_step_index",
    "t_before",
    "h_forced",
    "flowstar_status",
    "torch_noqueue_status",
    "torch_v2_status",
    "torch_noqueue_accepts_flowstar_h",
    "torch_v2_accepts_flowstar_h",
    "first_numeric_channel_divergence",
    "channel_attribution_valid",
    "center_delta_noqueue",
    "center_delta_v2",
    "scale_delta_noqueue",
    "scale_delta_v2",
    "right_map_width_ratio_noqueue_over_flowstar",
    "right_map_width_ratio_v2_over_flowstar",
    "reset_width_ratio_noqueue_over_flowstar",
    "reset_width_ratio_v2_over_flowstar",
    "target_width_ratio_noqueue_over_flowstar",
    "target_width_ratio_v2_over_flowstar",
    "picard_residual_ratio_noqueue_over_flowstar",
    "picard_residual_ratio_v2_over_flowstar",
    "output_range_width_ratio_noqueue_over_flowstar",
    "output_range_width_ratio_v2_over_flowstar",
    "verdict",
]


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.17g}"
    return str(value)


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = _float(numerator)
    den = _float(denominator)
    if num is None or den is None or abs(den) <= 0.0:
        return None
    return num / den


def _abs_delta(a: Any, b: Any) -> float | None:
    fa = _float(a)
    fb = _float(b)
    if fa is None or fb is None:
        return None
    return abs(fa - fb)


def _max_abs_delta(reference: Mapping[str, Any], candidate: Mapping[str, Any], fields: Iterable[str]) -> float | None:
    values = [_abs_delta(reference.get(field), candidate.get(field)) for field in fields]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _sum_widths(row: Mapping[str, Any], prefix: str) -> float | None:
    explicit = _float(row.get(f"{prefix}_width_sum"))
    if explicit is not None:
        return explicit
    x = _float(row.get(f"{prefix}_width_x"))
    y = _float(row.get(f"{prefix}_width_y"))
    if x is None and y is None:
        return None
    return (x or 0.0) + (y or 0.0)


def _put_widths(out: dict[str, Any], target_prefix: str, source: Mapping[str, Any], source_prefix: str) -> None:
    for suffix in ("x", "y", "sum"):
        source_key = f"{source_prefix}_width_{suffix}"
        if suffix == "sum":
            value = source.get(source_key)
            if value in (None, ""):
                value = _sum_widths(source, source_prefix)
        else:
            value = source.get(source_key)
        out[f"{target_prefix}_width_{suffix}"] = value if value is not None else ""


def _put_bounds(out: dict[str, Any], target_prefix: str, source: Mapping[str, Any], source_prefix: str) -> None:
    for dim in ("x", "y"):
        for side in ("lo", "hi"):
            value = source.get(f"{source_prefix}_{side}_{dim}")
            out[f"{target_prefix}_{side}_{dim}"] = value if value is not None else ""


def _scalar_float(value: Any) -> float | None:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _interval_bound(interval: Interval, side: str) -> float | None:
    return _scalar_float(interval.lo if side == "lo" else interval.hi)


def _range_box(value: TMVector | Sequence[Interval] | None) -> list[Interval] | None:
    if value is None:
        return None
    if isinstance(value, TMVector):
        try:
            return list(value.range_box())
        except Exception:
            return None
    try:
        return [iv for iv in value if isinstance(iv, Interval)]
    except TypeError:
        return None


def _put_lifecycle_bounds(out: dict[str, Any], target_prefix: str, boxes: Sequence[Interval] | None) -> None:
    if boxes is None:
        return
    for dim, interval in zip(("x", "y"), boxes):
        lo = _interval_bound(interval, "lo")
        hi = _interval_bound(interval, "hi")
        out[f"{target_prefix}_{dim}_lo"] = lo if lo is not None else ""
        out[f"{target_prefix}_{dim}_hi"] = hi if hi is not None else ""


def _put_lifecycle_bounds_from_row(out: dict[str, Any], target_prefix: str, source: Mapping[str, Any], source_prefix: str) -> None:
    for dim in ("x", "y"):
        for side in ("lo", "hi"):
            value = source.get(f"{source_prefix}_{side}_{dim}")
            out[f"{target_prefix}_{dim}_{side}"] = value if value not in (None, "") else out.get(f"{target_prefix}_{dim}_{side}", "")


def _fill_lifecycle_aliases(row: dict[str, Any]) -> None:
    for dim in ("x", "y"):
        for side in ("lo", "hi"):
            aliases = [
                (f"target_remainder_{dim}_{side}", f"target_remainder_{side}_{dim}"),
                (f"post_cutoff_residual_{dim}_{side}", f"picard_ctrunc_normal_residual_{side}_{dim}"),
                (f"picard_no_remainder_residual_{dim}_{side}", f"ordinary_residual_range_{side}_{dim}"),
                (f"ordinary_remainder_{dim}_{side}", f"ordinary_residual_range_{side}_{dim}"),
                (f"raw_ctrunc_residual_{dim}_{side}", f"raw_ctrunc_residual_{side}_{dim}"),
                (f"raw_ctrunc_polynomial_range_{dim}_{side}", f"raw_ctrunc_polynomial_range_{side}_{dim}"),
                (f"raw_ctrunc_remainder_{dim}_{side}", f"raw_ctrunc_remainder_{side}_{dim}"),
                (f"raw_remainder_dropped_terms_range_{dim}_{side}", f"raw_remainder_dropped_terms_range_{side}_{dim}"),
                (f"raw_remainder_multiplication_remainder_{dim}_{side}", f"raw_remainder_multiplication_remainder_{side}_{dim}"),
                (f"raw_remainder_integration_remainder_{dim}_{side}", f"raw_remainder_integration_remainder_{side}_{dim}"),
                (f"raw_remainder_before_accumulation_{dim}_{side}", f"raw_remainder_before_accumulation_{side}_{dim}"),
                (f"raw_remainder_after_integration_{dim}_{side}", f"raw_remainder_after_integration_{side}_{dim}"),
                (f"raw_remainder_after_dropped_terms_{dim}_{side}", f"raw_remainder_after_dropped_terms_{side}_{dim}"),
                (f"raw_remainder_after_cutoff_{dim}_{side}", f"raw_remainder_after_cutoff_{side}_{dim}"),
                (f"raw_remainder_before_poly_diff_{dim}_{side}", f"raw_remainder_before_poly_diff_{side}_{dim}"),
                (f"raw_remainder_after_poly_diff_{dim}_{side}", f"raw_remainder_after_poly_diff_{side}_{dim}"),
                (f"picard_ctrunc_raw_residual_{dim}_{side}", f"raw_ctrunc_residual_{side}_{dim}"),
                (f"picard_no_remainder_range_{dim}_{side}", f"picard_no_remainder_range_{side}_{dim}"),
                (f"picard_no_remainder_polynomial_range_{dim}_{side}", f"picard_no_remainder_polynomial_range_{side}_{dim}"),
                (f"picard_no_remainder_remainder_{dim}_{side}", f"picard_no_remainder_remainder_{side}_{dim}"),
                (f"target_remainder_before_ctrunc_{dim}_{side}", f"target_remainder_before_ctrunc_{side}_{dim}"),
                (f"cutoff_poly_diff_{dim}_{side}", f"poly_diff_range_{side}_{dim}"),
                (f"cutoff_polynomial_difference_{dim}_{side}", f"poly_diff_range_{side}_{dim}"),
                (f"polynomial_range_{dim}_{side}", f"polynomial_range_{side}_{dim}"),
            ]
            for target, source in aliases:
                if row.get(target) in (None, "") and row.get(source) not in (None, ""):
                    row[target] = row.get(source, "")
        width = row.get(f"cutoff_polynomial_difference_width_{dim}")
        if row.get(f"cutoff_polynomial_difference_{dim}_width") in (None, "") and width not in (None, ""):
            row[f"cutoff_polynomial_difference_{dim}_width"] = width
    for target, source in (
        ("extracted_center_x", "center_x"),
        ("extracted_center_y", "center_y"),
        ("extracted_scale_x", "scale_x"),
        ("extracted_scale_y", "scale_y"),
    ):
        if row.get(target) in (None, "") and row.get(source) not in (None, ""):
            row[target] = row.get(source, "")


def _zero_widths(out: dict[str, Any], target_prefix: str) -> None:
    for suffix in ("x", "y", "sum"):
        out[f"{target_prefix}_width_{suffix}"] = 0.0


def _blank_widths(out: dict[str, Any], target_prefix: str) -> None:
    for suffix in ("x", "y", "sum"):
        out[f"{target_prefix}_width_{suffix}"] = ""


def _first_present(row: Mapping[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted", "validated"}


def _status(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "not_completed"
    raw = str(row.get("status", "")).strip().lower()
    if raw in {"accepted", "validated"}:
        return "accepted"
    if raw in {"rejected", "failed", "failure"}:
        return "rejected"
    if raw in {"not_completed", "missing", ""}:
        if _truthy(row.get("accepted")):
            return "accepted"
        if _truthy(row.get("rejected")):
            return "rejected"
        return "not_completed"
    return raw


def _attempt_index(row: Mapping[str, Any]) -> int | None:
    value = _first_present(row, "attempt_index_within_step", "adaptive_attempt_index", "attempt_index")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _h_try(row: Mapping[str, Any]) -> float | None:
    return _float(_first_present(row, "h_try", "h_forced", "h"))


def _t_before(row: Mapping[str, Any]) -> float | None:
    return _float(row.get("t_before"))


def _trace_source(row: Mapping[str, Any], fallback: str = "") -> str:
    return str(_first_present(row, "trace_source", "mode", "source") or fallback)


def _rejection_reason(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return ""
    return str(_first_present(row, "rejection_reason", "message", "validation_message"))


def _width_value(row: Mapping[str, Any] | None, *fields: str) -> Any:
    if row is None:
        return ""
    return _first_present(row, *fields)


def _component_delta(reference: Mapping[str, Any], candidate: Mapping[str, Any], prefix: str) -> float | None:
    return _max_abs_delta(reference, candidate, (f"{prefix}_x", f"{prefix}_y"))


def _set_width_alias(
    row: dict[str, Any],
    target_prefix: str,
    source_prefix: str,
    *,
    fallback_prefix: str | None = None,
) -> None:
    for suffix in ("x", "y", "sum"):
        target_key = f"{target_prefix}_width_{suffix}"
        if row.get(target_key) not in (None, ""):
            continue
        value = row.get(f"{source_prefix}_width_{suffix}")
        if value in (None, "") and fallback_prefix is not None:
            value = row.get(f"{fallback_prefix}_width_{suffix}")
        row[target_key] = value if value not in (None, "") else ""


def _fill_common_trace_aliases(row: dict[str, Any], *, trace_source: str, attempt_global_index: int | None = None) -> None:
    row.setdefault("trace_source", trace_source)
    row.setdefault("attempt_global_index", attempt_global_index if attempt_global_index is not None else "")
    row.setdefault("accepted_step_index", row.get("step_index", ""))
    row.setdefault("attempt_index_within_step", row.get("adaptive_attempt_index", ""))
    row.setdefault("h_try", row.get("h", ""))
    row.setdefault("rejection_reason", row.get("message", "") if _status(row) == "rejected" else "")
    row.setdefault("residual_subset_target", _status(row) == "accepted")
    _set_width_alias(row, "target_check", "target_remainder")
    _set_width_alias(row, "ordinary_step_remainder", "picard_no_remainder_residual", fallback_prefix="residual")
    _set_width_alias(row, "right_map_range", "tmv_right_normal_range", fallback_prefix="tmv_right_range")
    _set_width_alias(row, "reset", "new_x0")
    _set_width_alias(row, "output_range", "final_flowpipe")
    _set_width_alias(row, "final_segment", "final_flowpipe")
    row.setdefault("queue_size", _first_present(row, "queue_size", "symbolic_J_size"))
    row.setdefault("j_count", _first_present(row, "j_count", "symbolic_J_size"))
    row.setdefault("phi_l_count", _first_present(row, "phi_l_count", "symbolic_Phi_L_size"))
    for suffix in ("x", "y", "sum"):
        row.setdefault(f"output_only_symbolic_width_{suffix}", row.get(f"symbolic_propagated_width_{suffix}", ""))
    _fill_lifecycle_aliases(row)


def _normalized_rows(rows: Iterable[Mapping[str, Any]], trace_source: str = "") -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        out = dict(row)
        _fill_common_trace_aliases(out, trace_source=trace_source or _trace_source(out), attempt_global_index=index)
        normalized.append(out)
    return normalized


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _str(row.get(field, "")) for field in fieldnames})


def compile_probe(flowstar_root: Path, out_dir: Path, compiler: str = "g++") -> Path:
    exe = out_dir / "flowstar_vdp_step_trace_probe"
    cmd = [
        compiler,
        "-O3",
        "-w",
        "-std=c++11",
        "-I",
        str(flowstar_root / "flowstar-toolbox"),
        "-I",
        "/usr/local/include",
        str(PROBE_CPP),
        "-L",
        str(flowstar_root / "flowstar-toolbox"),
        "-L",
        "/usr/local/lib",
        "-o",
        str(exe),
        "-lflowstar",
        "-lmpfr",
        "-lgmp",
        "-lgsl",
        "-lgslcblas",
        "-lm",
        "-lglpk",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    (out_dir / "flowstar_probe_compile.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "flowstar_probe_compile.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Flow* probe compilation failed; see {out_dir / 'flowstar_probe_compile.stderr.txt'}")
    return exe


def run_flowstar_probe(exe: Path, out_dir: Path, horizon: float, max_segments: int | None) -> Path:
    trace = out_dir / "flowstar_trace.csv"
    cmd = [str(exe), str(trace), f"{horizon:.17g}"]
    if max_segments:
        cmd.append(str(max_segments))
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(out_dir), check=False)
    (out_dir / "flowstar_probe_run.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "flowstar_probe_run.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Flow* probe run failed; see {out_dir / 'flowstar_probe_run.stderr.txt'}")
    return trace


def _validation_by_adaptive_attempt(diagnostics: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in diagnostics:
        adaptive = row.get("adaptive_attempt_index")
        if adaptive in (None, ""):
            continue
        try:
            key = int(adaptive)
        except (TypeError, ValueError):
            continue
        grouped[key] = row
    return grouped


def _common_torch_row(
    *,
    mode: str,
    step_index: int,
    t_before: float,
    validation: Mapping[str, Any],
    accepted: bool,
    normal_stats: Mapping[str, Any] | None,
    target_radius: float,
    pre_step_boxes: Sequence[Interval] | None = None,
    endpoint_box_before_center_boxes: Sequence[Interval] | None = None,
    reset_box_after_center_scale_boxes: Sequence[Interval] | None = None,
    full_step_validation_candidate_boxes: Sequence[Interval] | None = None,
    tau_h_endpoint_boxes: Sequence[Interval] | None = None,
) -> dict[str, Any]:
    message = validation.get("validation_message", "") or validation.get("rejection_reason", "")
    row: dict[str, Any] = {
        "trace_source": mode,
        "source": "torch",
        "mode": mode,
        "accepted_step_index": step_index,
        "step_index": step_index,
        "attempt_index_within_step": validation.get("adaptive_attempt_index", ""),
        "adaptive_attempt_index": validation.get("adaptive_attempt_index", ""),
        "t_before": t_before,
        "h_try": validation.get("h_try", validation.get("h", "")),
        "h": validation.get("h_try", validation.get("h", "")),
        "accepted": accepted,
        "rejected": not accepted,
        "status": "accepted" if accepted else "rejected",
        "rejection_reason": "" if accepted else message,
        "message": message,
        "residual_subset_target": validation.get("subset_tmp_remainder", validation.get("subset_result", accepted)),
    }
    h = _float(row["h"])
    if h is not None:
        row["t_after"] = t_before + h

    _put_lifecycle_bounds(row, "pre_step_box", pre_step_boxes)
    _put_lifecycle_bounds(row, "endpoint_box_before_center", endpoint_box_before_center_boxes)
    _put_lifecycle_bounds(row, "reset_box_after_center_scale", reset_box_after_center_scale_boxes)
    _put_lifecycle_bounds(row, "torch_full_step_validation_candidate", full_step_validation_candidate_boxes)
    _put_lifecycle_bounds(row, "torch_tau_h_endpoint", tau_h_endpoint_boxes)
    _put_lifecycle_bounds_from_row(row, "polynomial_range", validation, "polynomial_range")

    _put_widths(row, "tmv_pre_range", validation, "candidate_segment")
    _put_widths(row, "final_flowpipe", validation, "candidate_final")
    _put_widths(row, "residual", validation, "residual")
    if "ordinary_residual_range_width_sum" in validation:
        _put_widths(row, "picard_no_remainder_residual", validation, "ordinary_residual_range")
        _put_lifecycle_bounds_from_row(row, "picard_no_remainder_residual", validation, "ordinary_residual_range")
        _put_lifecycle_bounds_from_row(row, "ordinary_remainder", validation, "ordinary_residual_range")
    else:
        _put_widths(row, "picard_no_remainder_residual", validation, "residual")
        _put_lifecycle_bounds_from_row(row, "picard_no_remainder_residual", validation, "residual")
        _put_lifecycle_bounds_from_row(row, "ordinary_remainder", validation, "residual")

    for suffix in ("x", "y"):
        row[f"target_remainder_width_{suffix}"] = 2.0 * target_radius
        row[f"target_remainder_lo_{suffix}"] = -abs(float(target_radius))
        row[f"target_remainder_hi_{suffix}"] = abs(float(target_radius))
        row[f"target_remainder_{suffix}_lo"] = -abs(float(target_radius))
        row[f"target_remainder_{suffix}_hi"] = abs(float(target_radius))
    row["target_remainder_width_sum"] = 4.0 * target_radius
    _put_lifecycle_bounds_from_row(row, "target_remainder_before_ctrunc", row, "target_remainder")

    if "tmp_remainder_width_sum" in validation:
        _put_widths(row, "picard_ctrunc_normal_residual", validation, "tmp_remainder")
        _put_bounds(row, "picard_ctrunc_normal_residual", validation, "tmp_remainder")
    else:
        _put_widths(row, "picard_ctrunc_normal_residual", validation, "residual")
        _put_bounds(row, "picard_ctrunc_normal_residual", validation, "residual")
    _put_lifecycle_bounds_from_row(row, "raw_ctrunc_residual", validation, "raw_ctrunc_residual")
    _put_lifecycle_bounds_from_row(row, "raw_ctrunc_polynomial_range", validation, "raw_ctrunc_polynomial_range")
    _put_lifecycle_bounds_from_row(row, "raw_ctrunc_remainder", validation, "raw_ctrunc_remainder")
    _put_lifecycle_bounds_from_row(row, "raw_remainder_dropped_terms_range", validation, "raw_remainder_dropped_terms_range")
    _put_lifecycle_bounds_from_row(row, "raw_remainder_multiplication_remainder", validation, "raw_remainder_multiplication_remainder")
    _put_lifecycle_bounds_from_row(row, "raw_remainder_integration_remainder", validation, "raw_remainder_integration_remainder")
    _put_lifecycle_bounds_from_row(row, "raw_remainder_before_accumulation", validation, "raw_remainder_before_accumulation")
    _put_lifecycle_bounds_from_row(row, "raw_remainder_after_integration", validation, "raw_remainder_after_integration")
    _put_lifecycle_bounds_from_row(row, "raw_remainder_after_dropped_terms", validation, "raw_remainder_after_dropped_terms")
    _put_lifecycle_bounds_from_row(row, "raw_remainder_after_cutoff", validation, "raw_remainder_after_cutoff")
    _put_lifecycle_bounds_from_row(row, "raw_remainder_before_poly_diff", validation, "raw_remainder_before_poly_diff")
    _put_lifecycle_bounds_from_row(row, "raw_remainder_after_poly_diff", validation, "raw_remainder_after_poly_diff")
    _put_lifecycle_bounds_from_row(row, "picard_ctrunc_raw_residual", validation, "raw_ctrunc_residual")
    _put_lifecycle_bounds_from_row(row, "picard_no_remainder_range", validation, "picard_no_remainder_range")
    _put_lifecycle_bounds_from_row(row, "picard_no_remainder_polynomial_range", validation, "picard_no_remainder_polynomial_range")
    _put_lifecycle_bounds_from_row(row, "picard_no_remainder_remainder", validation, "picard_no_remainder_remainder")
    for key in (
        "raw_ctrunc_residual_source_object",
        "raw_ctrunc_residual_domain_semantics",
        "raw_ctrunc_residual_includes_target_remainder",
        "raw_ctrunc_residual_includes_ordinary_remainder",
        "raw_ctrunc_residual_includes_cutoff_poly_diff",
        "raw_ctrunc_residual_added_component",
        "raw_ctrunc_residual_notes",
        "raw_remainder_range_enclosure_method",
        "raw_remainder_normal_domain_scaling",
        "raw_remainder_partition_missing_reason",
        "ordinary_remainder_missing_reason",
    ):
        if validation.get(key) not in (None, ""):
            row[key] = validation.get(key, "")
    _put_bounds(row, "residual", validation, "residual")

    if "poly_diff_range_width_sum" in validation:
        _put_widths(row, "cutoff_polynomial_difference", validation, "poly_diff_range")
        _put_lifecycle_bounds_from_row(row, "cutoff_poly_diff", validation, "poly_diff_range")
        _put_lifecycle_bounds_from_row(row, "cutoff_polynomial_difference", validation, "poly_diff_range")
    elif normal_stats:
        _put_widths(row, "cutoff_polynomial_difference", normal_stats, "insertion_cutoff")
    else:
        _zero_widths(row, "cutoff_polynomial_difference")
    for suffix in ("x", "y"):
        row[f"cutoff_polynomial_difference_{suffix}_width"] = _first_present(
            row,
            f"cutoff_polynomial_difference_width_{suffix}",
            f"poly_diff_range_width_{suffix}",
        )

    residual_sum = _float(row.get("residual_width_sum"))
    if residual_sum is not None:
        row["residual_over_target_sum"] = residual_sum / (4.0 * target_radius)
    for suffix in ("x", "y"):
        residual = _float(row.get(f"residual_width_{suffix}"))
        if residual is not None:
            row[f"residual_over_target_{suffix}"] = residual / (2.0 * target_radius)

    if normal_stats:
        _put_widths(row, "tmv_right_range", normal_stats, "old_right_map_range")
        _put_widths(row, "tmv_right_normal_range", normal_stats, "normal_right_map_range")
        _put_widths(row, "endpoint_pre_center", normal_stats, "inserted_endpoint")
        _put_widths(row, "new_x0", normal_stats, "normalized_reset")
        for key in ("center_x", "center_y", "scale_x", "scale_y"):
            row[key] = normal_stats.get(key, "")
        row["extracted_center_x"] = normal_stats.get("center_x", "")
        row["extracted_center_y"] = normal_stats.get("center_y", "")
        row["extracted_scale_x"] = normal_stats.get("scale_x", "")
        row["extracted_scale_y"] = normal_stats.get("scale_y", "")
        for suffix in ("x", "y"):
            scale = _float(normal_stats.get(f"scale_{suffix}"))
            row[f"inv_scale_{suffix}"] = "" if scale is None or scale == 0.0 else 1.0 / scale
        row["queue_size"] = normal_stats.get("queue_size_after", normal_stats.get("queue_size", 0 if mode == "torch_noqueue" else ""))
        row["j_count"] = normal_stats.get("j_count", row.get("queue_size", ""))
        row["phi_l_count"] = normal_stats.get("phi_l_count", "")
        row["symbolic_J_size"] = row.get("j_count", "")
        row["symbolic_Phi_L_size"] = row.get("phi_l_count", "")
        row["scalar_x"] = normal_stats.get("scalar_x", row.get("inv_scale_x", ""))
        row["scalar_y"] = normal_stats.get("scalar_y", row.get("inv_scale_y", ""))
        if mode == "torch_v2":
            _put_widths(row, "symbolic_J", normal_stats, "new_symbolic")
            _put_widths(row, "symbolic_propagated", normal_stats, "propagated_symbolic")
        else:
            _zero_widths(row, "symbolic_J")
            _zero_widths(row, "symbolic_propagated")
    else:
        _blank_widths(row, "tmv_right_range")
        _blank_widths(row, "tmv_right_normal_range")
        _blank_widths(row, "endpoint_pre_center")
        _blank_widths(row, "new_x0")
        _blank_widths(row, "symbolic_J")
        _blank_widths(row, "symbolic_propagated")

    if normal_stats:
        for suffix in ("x", "y", "sum"):
            row[f"output_only_symbolic_width_{suffix}"] = _first_present(
                normal_stats,
                f"output_only_symbolic_width_{suffix}",
                f"materialized_for_output_width_{suffix}",
                f"propagated_symbolic_width_{suffix}",
            )

    row["endpoint_before_center_source_object"] = "seg.final_tm.range_box"
    row["endpoint_before_center_domain_semantics"] = "physical_endpoint_after_tau_substitution_tau_dropped"
    row["endpoint_before_center_includes_target_remainder"] = "false"
    row["endpoint_before_center_includes_ordinary_remainder"] = "false"
    symbolic_output = _float(row.get("output_only_symbolic_width_sum"))
    symbolic_output_present = symbolic_output is not None and symbolic_output > 1e-300
    row["endpoint_before_center_includes_symbolic_output_width"] = "true" if symbolic_output_present else "false"
    row["endpoint_before_center_includes_cutoff_poly_diff"] = "true" if "tmp_remainder_width_sum" in validation else "unknown"
    row["endpoint_before_center_range_eval_method"] = "TMVector.range_box on seg.final_tm after substitute_const(tau=h).drop_variable(tau)"
    row["endpoint_before_center_polynomial_order"] = validation.get("order", "")
    for suffix in ("x", "y", "sum"):
        row[f"endpoint_before_center_dropped_terms_width_{suffix}"] = _first_present(
            row,
            f"cutoff_polynomial_difference_width_{suffix}",
            f"poly_diff_range_width_{suffix}",
        )
        row[f"endpoint_before_center_remainder_width_{suffix}"] = _first_present(
            row,
            f"picard_ctrunc_normal_residual_width_{suffix}",
            f"tmp_remainder_width_{suffix}",
        )
    row["endpoint_before_center_notes"] = (
        "diagnostic label: endpoint_box_before_center is the accepted final segment range, "
        "not the normalized-insertion inserted_endpoint/right_map range"
    )

    canonical_full_step_source = "Picard_ctrunc_normal_post_poly_diff_validation_candidate"
    canonical_full_step_domain = "physical_tube_over_full_step_tau_domain_before_tau_h_substitution"
    canonical_tau_h_source = "tau_h_endpoint_of_Picard_ctrunc_normal_post_poly_diff_validation_candidate"
    canonical_tau_h_domain = "physical_endpoint_tau_h_after_tau_substitution_tau_dropped"
    symbolic_output_flag = "true" if symbolic_output_present else "false"
    cutoff_flag = "true" if "tmp_remainder_width_sum" in validation else "unknown"
    row["torch_full_step_validation_candidate_source_object"] = canonical_full_step_source
    row["torch_full_step_validation_candidate_domain_semantics"] = canonical_full_step_domain
    row["torch_full_step_validation_candidate_includes_cutoff_poly_diff"] = cutoff_flag
    row["torch_full_step_validation_candidate_includes_target_remainder"] = "false"
    row["torch_full_step_validation_candidate_includes_ordinary_remainder"] = "false"
    row["torch_full_step_validation_candidate_includes_symbolic_output_width"] = symbolic_output_flag
    row["torch_tau_h_endpoint_source_object"] = canonical_tau_h_source
    row["torch_tau_h_endpoint_domain_semantics"] = canonical_tau_h_domain
    row["torch_tau_h_endpoint_includes_cutoff_poly_diff"] = cutoff_flag
    row["torch_tau_h_endpoint_includes_target_remainder"] = "false"
    row["torch_tau_h_endpoint_includes_ordinary_remainder"] = "false"
    row["torch_tau_h_endpoint_includes_symbolic_output_width"] = symbolic_output_flag

    _put_lifecycle_bounds_from_row(row, "post_cutoff_residual", row, "picard_ctrunc_normal_residual")
    _fill_common_trace_aliases(row, trace_source=mode)
    return row


def generate_torch_trace(
    *,
    mode: str,
    horizon: float,
    out_path: Path,
    target_radius: float = 1e-4,
    h_min: float = 0.002,
    h_max: float = 0.1,
    order: int = 4,
    validation_mode: str = "target_remainder_flowstar_ctrunc",
    max_segments: int | None = None,
) -> list[dict[str, Any]]:
    if mode not in {"torch_noqueue", "torch_v2"}:
        raise ValueError(f"unsupported torch trace mode: {mode}")
    current: TMVector | list[Interval] = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state: FlowstarNormalFlowpipeState | None = None
    h = h_max
    t = 0.0
    step_index = 0
    trace_rows: list[dict[str, Any]] = []
    reset_mode = "normalized_insertion" if mode == "torch_noqueue" else "normalized_insertion_symqueue_v2"
    symbolic_queue_mode = "" if mode == "torch_noqueue" else "flowstar_linear_v2"

    while t < horizon - 1e-15:
        if max_segments is not None and step_index >= max_segments:
            break
        h_try = min(h, h_max, horizon - t)
        if h_try < h_min:
            h_try = h_min
        diagnostics: list[dict[str, Any]] = []
        pre_step_boxes = _range_box(current)
        seg = flowpipe_step_flowstar_style_adaptive(
            van_der_pol_ode,
            current,
            h=h_try,
            h_min=h_min,
            h_max=h_max,
            order=order,
            target_remainder_radius=target_radius,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_mode=validation_mode,
            reset_mode=reset_mode,
            symbolic_queue_mode=symbolic_queue_mode,
            flowstar_symbolic_queue_max_size=100,
            flowstar_normal_state=normal_state,
            diagnostics=diagnostics,
            diagnostics_context={"mode": mode, "segment_index": step_index, "t_before": t},
        )

        grouped = _validation_by_adaptive_attempt(diagnostics)
        accepted_attempt = None
        if seg.status == "validated":
            accepted_h = _float(seg.h)
            for attempt, validation in grouped.items():
                if _float(validation.get("h_try", validation.get("h"))) == accepted_h:
                    accepted_attempt = attempt
            if accepted_attempt is None and grouped:
                accepted_attempt = max(grouped)

        for attempt in sorted(grouped):
            validation = grouped[attempt]
            accepted = bool(seg.status == "validated" and attempt == accepted_attempt)
            full_step_boxes = _range_box(seg.tm) if accepted else None
            endpoint_boxes = _range_box(seg.final_tm) if accepted else None
            reset_boxes = _range_box(seg.reset_tm) if accepted else None
            row = _common_torch_row(
                mode=mode,
                step_index=step_index,
                t_before=t,
                validation=validation,
                accepted=accepted,
                normal_stats=seg.flowstar_normal_stats if accepted else None,
                target_radius=target_radius,
                pre_step_boxes=pre_step_boxes,
                endpoint_box_before_center_boxes=endpoint_boxes,
                reset_box_after_center_scale_boxes=reset_boxes,
                full_step_validation_candidate_boxes=full_step_boxes,
                tau_h_endpoint_boxes=endpoint_boxes,
            )
            row["attempt_global_index"] = len(trace_rows)
            h_value = _float(row.get("h_try"))
            if accepted:
                row["h_after_if_rejected_or_next"] = seg.next_h if seg.next_h is not None else (min(h_value * 1.5, h_max) if h_value is not None else "")
            elif h_value is not None:
                row["h_after_if_rejected_or_next"] = max(h_value * 0.5, h_min)
            trace_rows.append(row)

        if seg.status != "validated" or seg.reset_tm is None:
            break

        t += float(seg.h)
        current = seg.reset_tm
        normal_state = seg.flowstar_normal_state
        h = float(seg.next_h if seg.next_h is not None else min(float(seg.h) * 1.5, h_max))
        step_index += 1

    _write_rows(out_path, TRACE_FIELDS, trace_rows)
    return trace_rows


def _accepted(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if _status(row) == "accepted"]


def _flowstar_accepted_schedule(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in _accepted(_normalized_rows(rows, "flowstar"))]


def generate_torch_forced_h_trace(
    *,
    mode: str,
    flowstar_rows: list[Mapping[str, Any]],
    out_path: Path,
    target_radius: float = 1e-4,
    order: int = 4,
    validation_mode: str = "target_remainder_flowstar_ctrunc",
) -> list[dict[str, Any]]:
    if mode not in {"torch_noqueue", "torch_v2"}:
        raise ValueError(f"unsupported torch trace mode: {mode}")
    schedule = _flowstar_accepted_schedule(flowstar_rows)
    current: TMVector | list[Interval] = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state: FlowstarNormalFlowpipeState | None = None
    trace_rows: list[dict[str, Any]] = []
    reset_mode = "normalized_insertion" if mode == "torch_noqueue" else "normalized_insertion_symqueue_v2"
    symbolic_queue_mode = "" if mode == "torch_noqueue" else "flowstar_linear_v2"

    for step_index, flow_row in enumerate(schedule):
        h_forced = _h_try(flow_row)
        if h_forced is None or h_forced <= 0.0:
            continue
        t_label = _t_before(flow_row)
        diagnostics: list[dict[str, Any]] = []
        pre_step_boxes = _range_box(current)
        seg = flowpipe_step_flowstar_style_adaptive(
            van_der_pol_ode,
            current,
            h=h_forced,
            h_min=h_forced,
            h_max=h_forced,
            order=order,
            target_remainder_radius=target_radius,
            cutoff_threshold=1e-10,
            max_validation_attempts=2,
            validation_mode=validation_mode,
            reset_mode=reset_mode,
            symbolic_queue_mode=symbolic_queue_mode,
            flowstar_symbolic_queue_max_size=100,
            flowstar_normal_state=normal_state,
            grow_factor=1.0,
            diagnostics=diagnostics,
            diagnostics_context={
                "mode": mode,
                "segment_index": step_index,
                "t_before": t_label if t_label is not None else "",
                "forced_h_replay": True,
            },
        )
        grouped = _validation_by_adaptive_attempt(diagnostics)
        if not grouped:
            row = {
                "trace_source": mode,
                "source": "torch",
                "mode": mode,
                "attempt_global_index": len(trace_rows),
                "accepted_step_index": step_index,
                "step_index": step_index,
                "attempt_index_within_step": 1,
                "adaptive_attempt_index": 1,
                "t_before": t_label if t_label is not None else "",
                "h_try": h_forced,
                "h": h_forced,
                "accepted": False,
                "rejected": False,
                "status": "not_completed",
                "message": seg.message,
            }
            _fill_common_trace_aliases(row, trace_source=mode, attempt_global_index=len(trace_rows))
            trace_rows.append(row)
            break

        accepted_attempt = None
        if seg.status == "validated":
            accepted_h = _float(seg.h)
            for attempt, validation in grouped.items():
                if _float(validation.get("h_try", validation.get("h"))) == accepted_h:
                    accepted_attempt = attempt
            if accepted_attempt is None and grouped:
                accepted_attempt = max(grouped)

        step_accepted = False
        for attempt in sorted(grouped):
            validation = grouped[attempt]
            accepted = bool(seg.status == "validated" and attempt == accepted_attempt)
            full_step_boxes = _range_box(seg.tm) if accepted else None
            endpoint_boxes = _range_box(seg.final_tm) if accepted else None
            reset_boxes = _range_box(seg.reset_tm) if accepted else None
            row = _common_torch_row(
                mode=mode,
                step_index=step_index,
                t_before=t_label if t_label is not None else 0.0,
                validation=validation,
                accepted=accepted,
                normal_stats=seg.flowstar_normal_stats if accepted else None,
                target_radius=target_radius,
                pre_step_boxes=pre_step_boxes,
                endpoint_box_before_center_boxes=endpoint_boxes,
                reset_box_after_center_scale_boxes=reset_boxes,
                full_step_validation_candidate_boxes=full_step_boxes,
                tau_h_endpoint_boxes=endpoint_boxes,
            )
            row["attempt_global_index"] = len(trace_rows)
            row["h_after_if_rejected_or_next"] = h_forced
            row["forced_h_replay"] = True
            trace_rows.append(row)
            step_accepted = step_accepted or accepted

        if not step_accepted or seg.reset_tm is None:
            break
        current = seg.reset_tm
        normal_state = seg.flowstar_normal_state

    _write_rows(out_path, TRACE_FIELDS, trace_rows)
    return trace_rows


def _channel_width(row: Mapping[str, Any] | None, channel: str) -> Any:
    if row is None:
        return ""
    if channel == "residual":
        return _width_value(row, "residual_width_sum", "picard_ctrunc_normal_residual_width_sum")
    if channel == "target":
        return _width_value(row, "target_remainder_width_sum", "target_check_width_sum")
    if channel == "right_map":
        return _width_value(row, "right_map_range_width_sum", "tmv_right_normal_range_width_sum", "tmv_right_range_width_sum")
    if channel == "reset":
        return _width_value(row, "reset_width_sum", "new_x0_width_sum", "normalized_reset_width_sum")
    if channel == "output_range":
        return _width_value(row, "output_range_width_sum", "final_segment_width_sum", "final_flowpipe_width_sum")
    if channel == "picard":
        return _width_value(row, "picard_ctrunc_normal_residual_width_sum", "residual_width_sum")
    if channel == "cutoff":
        return _width_value(row, "cutoff_polynomial_difference_width_sum", "poly_diff_range_width_sum")
    if channel == "symbolic":
        return _width_value(row, "symbolic_J_width_sum", "output_only_symbolic_width_sum", "symbolic_propagated_width_sum")
    raise ValueError(f"unknown channel: {channel}")


def _channel_ratio(flow: Mapping[str, Any], candidate: Mapping[str, Any] | None, channel: str) -> float | None:
    return _ratio(_channel_width(candidate, channel), _channel_width(flow, channel))


def _numeric_channel_divergence(
    flow: Mapping[str, Any],
    noqueue: Mapping[str, Any] | None,
    v2: Mapping[str, Any] | None,
    *,
    ratio_threshold: float = 1.25,
    delta_threshold: float = 1e-6,
) -> str:
    saw_comparable = False
    delta_channels = [
        ("center_scaling", [_component_delta(flow, row, "center") for row in (noqueue, v2) if row is not None]),
        ("center_scaling", [_component_delta(flow, row, "scale") for row in (noqueue, v2) if row is not None]),
    ]
    for label, values in delta_channels:
        for value in values:
            if value is None:
                continue
            saw_comparable = True
            if value > delta_threshold:
                return label

    ratio_channels = [
        ("right_map_range", "right_map"),
        ("reset_new_x0", "reset"),
        ("target_remainder", "target"),
        ("picard_residual", "picard"),
        ("cutoff_poly_diff", "cutoff"),
        ("output_range", "output_range"),
        ("symbolic_queue", "symbolic"),
    ]
    for label, channel in ratio_channels:
        for row in (noqueue, v2):
            value = _channel_ratio(flow, row, channel) if row is not None else None
            if value is None:
                continue
            saw_comparable = True
            if value > ratio_threshold or value < 1.0 / ratio_threshold:
                return label
    return "" if saw_comparable else "unknown"


def _int_from_fields(row: Mapping[str, Any], *fields: str, default: int = 0) -> int:
    for field in fields:
        value = _float(row.get(field))
        if value is not None:
            return int(value)
    return default


def _attempt_sort_key(row: Mapping[str, Any]) -> tuple[float, int, int, float]:
    return (
        _t_before(row) if _t_before(row) is not None else float("inf"),
        _int_from_fields(row, "accepted_step_index", "step_index"),
        _attempt_index(row) or _int_from_fields(row, "attempt_global_index"),
        _h_try(row) if _h_try(row) is not None else float("inf"),
    )


def _same_attempt(reference: Mapping[str, Any], candidate: Mapping[str, Any], *, tolerance: float) -> bool:
    ref_t = _t_before(reference)
    cand_t = _t_before(candidate)
    ref_h = _h_try(reference)
    cand_h = _h_try(candidate)
    if ref_t is None or cand_t is None or ref_h is None or cand_h is None:
        return False
    if abs(ref_t - cand_t) > tolerance or abs(ref_h - cand_h) > tolerance:
        return False
    ref_attempt = _attempt_index(reference)
    cand_attempt = _attempt_index(candidate)
    if ref_attempt is not None and cand_attempt is not None and ref_attempt != cand_attempt:
        return False
    return True


def _find_matching_attempt(
    reference: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    used: set[int],
    *,
    tolerance: float,
) -> tuple[int | None, Mapping[str, Any] | None]:
    for index, candidate in enumerate(candidates):
        if index in used:
            continue
        if _same_attempt(reference, candidate, tolerance=tolerance):
            return index, candidate
    return None, None


def compare_attempt_aligned(
    flowstar_rows: list[Mapping[str, Any]],
    noqueue_rows: list[Mapping[str, Any]],
    v2_rows: list[Mapping[str, Any]],
    *,
    tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    flow = sorted(_normalized_rows(flowstar_rows, "flowstar"), key=_attempt_sort_key)
    noq = sorted(_normalized_rows(noqueue_rows, "torch_noqueue"), key=_attempt_sort_key)
    v2 = sorted(_normalized_rows(v2_rows, "torch_v2"), key=_attempt_sort_key)
    used_noq: set[int] = set()
    used_v2: set[int] = set()
    rows: list[dict[str, Any]] = []
    for aligned_index, flow_row in enumerate(flow):
        noq_index, noq_row = _find_matching_attempt(flow_row, noq, used_noq, tolerance=tolerance)
        v2_index, v2_row = _find_matching_attempt(flow_row, v2, used_v2, tolerance=tolerance)
        if noq_index is not None:
            used_noq.add(noq_index)
        if v2_index is not None:
            used_v2.add(v2_index)

        flow_status = _status(flow_row)
        noq_status = _status(noq_row)
        v2_status = _status(v2_row)
        aligned = noq_row is not None and v2_row is not None
        status_divergence = ""
        numeric_divergence = ""
        verdict = "aligned_no_material_divergence"
        channel_valid = aligned
        if not aligned:
            verdict = "unaligned_t_or_h"
            channel_valid = False
        elif len({flow_status, noq_status, v2_status}) > 1:
            status_divergence = "adaptive_acceptance_policy"
            verdict = "adaptive_acceptance_policy"
        else:
            numeric_divergence = _numeric_channel_divergence(flow_row, noq_row, v2_row)
            if numeric_divergence:
                verdict = numeric_divergence

        rows.append(
            {
                "aligned_index": aligned_index,
                "t_before": flow_row.get("t_before", ""),
                "h_try": _first_present(flow_row, "h_try", "h"),
                "flowstar_status": flow_status,
                "torch_noqueue_status": noq_status,
                "torch_v2_status": v2_status,
                "first_status_divergence": status_divergence,
                "first_numeric_channel_divergence": numeric_divergence,
                "channel_attribution_valid": channel_valid,
                "flowstar_rejection_reason": _rejection_reason(flow_row),
                "torch_noqueue_rejection_reason": _rejection_reason(noq_row),
                "torch_v2_rejection_reason": _rejection_reason(v2_row),
                "residual_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "residual"),
                "residual_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "residual"),
                "target_width_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "target"),
                "target_width_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "target"),
                "right_map_width_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "right_map"),
                "right_map_width_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "right_map"),
                "reset_width_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "reset"),
                "reset_width_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "reset"),
                "output_range_width_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "output_range"),
                "output_range_width_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "output_range"),
                "center_delta_noqueue": _component_delta(flow_row, noq_row, "center") if noq_row is not None else None,
                "center_delta_v2": _component_delta(flow_row, v2_row, "center") if v2_row is not None else None,
                "scale_delta_noqueue": _component_delta(flow_row, noq_row, "scale") if noq_row is not None else None,
                "scale_delta_v2": _component_delta(flow_row, v2_row, "scale") if v2_row is not None else None,
                "verdict": verdict,
            }
        )
    return rows


def _find_forced_row(
    flow_row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    index: int,
    *,
    tolerance: float,
) -> Mapping[str, Any] | None:
    for candidate in candidates:
        if _same_attempt(flow_row, candidate, tolerance=tolerance):
            return candidate
    for candidate in candidates:
        cand_step = _int_from_fields(candidate, "accepted_step_index", "step_index", default=-1)
        if cand_step == index and _h_try(candidate) is not None and _h_try(flow_row) is not None:
            if abs((_h_try(candidate) or 0.0) - (_h_try(flow_row) or 0.0)) <= tolerance:
                return candidate
    return None


def compare_forced_h(
    flowstar_rows: list[Mapping[str, Any]],
    noqueue_rows: list[Mapping[str, Any]],
    v2_rows: list[Mapping[str, Any]],
    *,
    tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    schedule = _flowstar_accepted_schedule(flowstar_rows)
    noq = _normalized_rows(noqueue_rows, "torch_noqueue")
    v2 = _normalized_rows(v2_rows, "torch_v2")
    rows: list[dict[str, Any]] = []
    for index, flow_row in enumerate(schedule):
        noq_row = _find_forced_row(flow_row, noq, index, tolerance=tolerance)
        v2_row = _find_forced_row(flow_row, v2, index, tolerance=tolerance)
        noq_accepts = _status(noq_row) == "accepted"
        v2_accepts = _status(v2_row) == "accepted"
        numeric = ""
        channel_valid = False
        verdict = "forced_h_alignment_missing"
        if noq_row is not None and v2_row is not None:
            if noq_accepts and v2_accepts:
                numeric = _numeric_channel_divergence(flow_row, noq_row, v2_row)
                channel_valid = True
                verdict = numeric or "forced_h_no_material_divergence"
            else:
                verdict = "pytorch_rejects_flowstar_h"
        row = {
            "forced_step_index": index,
            "t_before": flow_row.get("t_before", ""),
            "h_forced": _first_present(flow_row, "h_try", "h"),
            "flowstar_status": _status(flow_row),
            "torch_noqueue_status": _status(noq_row),
            "torch_v2_status": _status(v2_row),
            "torch_noqueue_accepts_flowstar_h": noq_accepts,
            "torch_v2_accepts_flowstar_h": v2_accepts,
            "first_numeric_channel_divergence": numeric,
            "channel_attribution_valid": channel_valid,
            "center_delta_noqueue": _component_delta(flow_row, noq_row, "center") if noq_row is not None else None,
            "center_delta_v2": _component_delta(flow_row, v2_row, "center") if v2_row is not None else None,
            "scale_delta_noqueue": _component_delta(flow_row, noq_row, "scale") if noq_row is not None else None,
            "scale_delta_v2": _component_delta(flow_row, v2_row, "scale") if v2_row is not None else None,
            "right_map_width_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "right_map"),
            "right_map_width_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "right_map"),
            "reset_width_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "reset"),
            "reset_width_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "reset"),
            "target_width_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "target"),
            "target_width_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "target"),
            "picard_residual_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "picard"),
            "picard_residual_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "picard"),
            "output_range_width_ratio_noqueue_over_flowstar": _channel_ratio(flow_row, noq_row, "output_range"),
            "output_range_width_ratio_v2_over_flowstar": _channel_ratio(flow_row, v2_row, "output_range"),
            "verdict": verdict,
        }
        rows.append(row)
        if noq_row is None or v2_row is None or not (noq_accepts and v2_accepts):
            break
    return rows


def _channel_ratios(flow: Mapping[str, Any], torch_row: Mapping[str, Any]) -> dict[str, float | None]:
    center_scale_delta = _max_abs_delta(
        flow,
        torch_row,
        ("center_x", "center_y", "scale_x", "scale_y", "inv_scale_x", "inv_scale_y"),
    )
    return {
        "center_scale": center_scale_delta,
        "inserted_endpoint": _ratio(torch_row.get("endpoint_pre_center_width_sum"), flow.get("endpoint_pre_center_width_sum")),
        "right_map": _ratio(torch_row.get("tmv_right_normal_range_width_sum"), flow.get("tmv_right_normal_range_width_sum")),
        "target_remainder": _ratio(torch_row.get("target_remainder_width_sum"), flow.get("target_remainder_width_sum")),
        "picard_residual": _ratio(torch_row.get("picard_ctrunc_normal_residual_width_sum"), flow.get("picard_ctrunc_normal_residual_width_sum")),
        "cutoff_poly": _ratio(torch_row.get("cutoff_polynomial_difference_width_sum"), flow.get("cutoff_polynomial_difference_width_sum")),
        "symbolic_queue": _ratio(torch_row.get("symbolic_J_width_sum"), flow.get("symbolic_J_width_sum")),
    }


def _material_channel(noq: dict[str, float | None], v2: dict[str, float | None], *, ratio_threshold: float, delta_threshold: float) -> str:
    channels = [
        ("center/scaling", "center_scale"),
        ("inserted endpoint map", "inserted_endpoint"),
        ("right-map range", "right_map"),
        ("target remainder", "target_remainder"),
        ("Picard residual", "picard_residual"),
        ("cutoff / polynomial truncation", "cutoff_poly"),
        ("symbolic queue", "symbolic_queue"),
    ]
    for label, key in channels:
        values = [noq.get(key), v2.get(key)]
        for value in values:
            if value is None:
                continue
            if key == "center_scale":
                if value > delta_threshold:
                    return label
            elif value > ratio_threshold or value < 1.0 / ratio_threshold:
                return label
    return ""


def _step_alignment_warning(flow: Mapping[str, Any], noq: Mapping[str, Any], v2: Mapping[str, Any], *, tolerance: float = 1e-9) -> str:
    mismatches: list[str] = []
    pairs = [
        ("noqueue_t", flow.get("t_before"), noq.get("t_before")),
        ("v2_t", flow.get("t_before"), v2.get("t_before")),
        ("noqueue_h", flow.get("h"), noq.get("h")),
        ("v2_h", flow.get("h"), v2.get("h")),
    ]
    for label, reference, candidate in pairs:
        ref = _float(reference)
        cand = _float(candidate)
        if ref is None or cand is None:
            continue
        if abs(ref - cand) > tolerance:
            mismatches.append(f"{label} differs: Flow*={ref:.17g}, torch={cand:.17g}")
    return "; ".join(mismatches)


def align_traces(flowstar_rows: list[Mapping[str, Any]], noqueue_rows: list[Mapping[str, Any]], v2_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flow_acc = _accepted(flowstar_rows)
    noq_acc = _accepted(noqueue_rows)
    v2_acc = _accepted(v2_rows)
    total = min(len(flow_acc), len(noq_acc), len(v2_acc))
    rows: list[dict[str, Any]] = []
    for index in range(total):
        flow = flow_acc[index]
        noq = noq_acc[index]
        v2 = v2_acc[index]
        noq_channels = _channel_ratios(flow, noq)
        v2_channels = _channel_ratios(flow, v2)
        alignment_warning = _step_alignment_warning(flow, noq, v2)
        raw_channel = _material_channel(noq_channels, v2_channels, ratio_threshold=1.25, delta_threshold=1e-6)
        channel = "adaptive_step_alignment_mismatch" if alignment_warning else raw_channel
        channel_valid = not bool(alignment_warning)
        comparison_kind = "accepted_ordinal_trace_diff_noncausal" if alignment_warning else "accepted_ordinal_trace_diff"
        notes = "ratio threshold 1.25, center/scale absolute delta threshold 1e-6" if raw_channel else ""
        if alignment_warning:
            notes = f"{alignment_warning}; channel attribution marked invalid/noncausal"
        row = {
            "step_index": index,
            "t_flowstar": flow.get("t_before", ""),
            "t_noqueue": noq.get("t_before", ""),
            "t_v2": v2.get("t_before", ""),
            "flowstar_h": flow.get("h", ""),
            "noqueue_h": noq.get("h", ""),
            "v2_h": v2.get("h", ""),
            "flowstar_width_sum": flow.get("final_flowpipe_width_sum", ""),
            "noqueue_width_sum": noq.get("final_flowpipe_width_sum", ""),
            "v2_width_sum": v2.get("final_flowpipe_width_sum", ""),
            "noqueue_width_ratio": _ratio(noq.get("final_flowpipe_width_sum"), flow.get("final_flowpipe_width_sum")),
            "v2_width_ratio": _ratio(v2.get("final_flowpipe_width_sum"), flow.get("final_flowpipe_width_sum")),
            "flowstar_residual_sum": flow.get("picard_ctrunc_normal_residual_width_sum", flow.get("residual_width_sum", "")),
            "noqueue_residual_sum": noq.get("picard_ctrunc_normal_residual_width_sum", noq.get("residual_width_sum", "")),
            "v2_residual_sum": v2.get("picard_ctrunc_normal_residual_width_sum", v2.get("residual_width_sum", "")),
            "noqueue_residual_ratio": _ratio(noq.get("picard_ctrunc_normal_residual_width_sum"), flow.get("picard_ctrunc_normal_residual_width_sum")),
            "v2_residual_ratio": _ratio(v2.get("picard_ctrunc_normal_residual_width_sum"), flow.get("picard_ctrunc_normal_residual_width_sum")),
            "center_scale_delta_noqueue": noq_channels["center_scale"],
            "center_scale_delta_v2": v2_channels["center_scale"],
            "inserted_endpoint_ratio_noqueue": noq_channels["inserted_endpoint"],
            "inserted_endpoint_ratio_v2": v2_channels["inserted_endpoint"],
            "right_map_ratio_noqueue": noq_channels["right_map"],
            "right_map_ratio_v2": v2_channels["right_map"],
            "target_remainder_ratio_noqueue": noq_channels["target_remainder"],
            "target_remainder_ratio_v2": v2_channels["target_remainder"],
            "picard_residual_ratio_noqueue": noq_channels["picard_residual"],
            "picard_residual_ratio_v2": v2_channels["picard_residual"],
            "cutoff_poly_ratio_noqueue": noq_channels["cutoff_poly"],
            "cutoff_poly_ratio_v2": v2_channels["cutoff_poly"],
            "symbolic_queue_ratio_v2": v2_channels["symbolic_queue"],
            "first_material_channel": channel,
            "channel_attribution_valid": channel_valid,
            "comparison_kind": comparison_kind,
            "alignment_warning": alignment_warning,
            "notes": notes,
        }
        rows.append(row)
    return rows


def step_alignment_warnings(aligned_rows: list[Mapping[str, Any]], *, tolerance: float = 1e-9) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for row in aligned_rows:
        flow_h = _float(row.get("flowstar_h"))
        noq_h = _float(row.get("noqueue_h"))
        v2_h = _float(row.get("v2_h"))
        flow_t = _float(row.get("t_flowstar"))
        noq_t = _float(row.get("t_noqueue"))
        v2_t = _float(row.get("t_v2"))
        messages: list[str] = []
        for label, reference, candidate in (
            ("noqueue_t", flow_t, noq_t),
            ("v2_t", flow_t, v2_t),
            ("noqueue_h", flow_h, noq_h),
            ("v2_h", flow_h, v2_h),
        ):
            if reference is None or candidate is None:
                continue
            if abs(reference - candidate) > tolerance:
                messages.append(f"{label} differs: Flow*={reference:.17g}, torch={candidate:.17g}")
        if not messages:
            continue
        warnings.append(
            {
                "step_index": row.get("step_index", ""),
                "t_flowstar": row.get("t_flowstar", ""),
                "t_noqueue": row.get("t_noqueue", ""),
                "t_v2": row.get("t_v2", ""),
                "flowstar_h": row.get("flowstar_h", ""),
                "noqueue_h": row.get("noqueue_h", ""),
                "v2_h": row.get("v2_h", ""),
                "first_material_channel": "adaptive_step_alignment_mismatch",
                "channel_attribution_valid": False,
                "comparison_kind": "accepted_ordinal_trace_diff_noncausal",
                "alignment_warning": "; ".join(messages),
            }
        )
    return warnings


def _find_attempt_at(rows: Iterable[Mapping[str, Any]], *, t: float, h: float, tolerance: float = 1e-9) -> Mapping[str, Any] | None:
    normalized = _normalized_rows(rows)
    for row in normalized:
        row_t = _t_before(row)
        row_h = _h_try(row)
        if row_t is None or row_h is None:
            continue
        if abs(row_t - t) <= tolerance and abs(row_h - h) <= tolerance:
            return row
    return None


def _row_number(row: Mapping[str, Any] | None, field: str) -> str:
    if row is None:
        return ""
    return _str(_first_present(row, field))


def _attempt_fact(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "missing"
    return (
        f"status=`{_status(row)}`, residual_width_sum=`{_row_number(row, 'residual_width_sum')}`, "
        f"target_width_sum=`{_row_number(row, 'target_remainder_width_sum') or _row_number(row, 'target_check_width_sum')}`, "
        f"residual_over_target_sum=`{_row_number(row, 'residual_over_target_sum')}`"
    )


def _endpoint_subset(row: Mapping[str, Any] | None, dim: str) -> dict[str, Any]:
    if row is None:
        return {"subset": None, "residual_lo": None, "residual_hi": None, "target_lo": None, "target_hi": None}
    residual_lo = _float(row.get(f"picard_ctrunc_normal_residual_lo_{dim}"))
    residual_hi = _float(row.get(f"picard_ctrunc_normal_residual_hi_{dim}"))
    target_lo = _float(row.get(f"target_remainder_lo_{dim}"))
    target_hi = _float(row.get(f"target_remainder_hi_{dim}"))
    if target_lo is None:
        target_lo = -1e-4
    if target_hi is None:
        target_hi = 1e-4
    subset = None
    if residual_lo is not None and residual_hi is not None:
        subset = residual_lo >= target_lo and residual_hi <= target_hi
    return {
        "subset": subset,
        "residual_lo": residual_lo,
        "residual_hi": residual_hi,
        "target_lo": target_lo,
        "target_hi": target_hi,
    }


def _subset_word(value: Any) -> str:
    if value is None:
        return "unknown"
    return "yes" if bool(value) else "no"


def _predicate_endpoint_fact(label: str, row: Mapping[str, Any] | None) -> str:
    x = _endpoint_subset(row, "x")
    y = _endpoint_subset(row, "y")
    failed = [dim for dim, data in (("x", x), ("y", y)) if data["subset"] is False]
    failed_text = ";".join(failed) if failed else "none"
    return (
        f"- {label}: subset_x=`{_subset_word(x['subset'])}`, "
        f"residual_x=`[{_str(x['residual_lo'])}, {_str(x['residual_hi'])}]`, "
        f"target_x=`[{_str(x['target_lo'])}, {_str(x['target_hi'])}]`; "
        f"subset_y=`{_subset_word(y['subset'])}`, "
        f"residual_y=`[{_str(y['residual_lo'])}, {_str(y['residual_hi'])}]`, "
        f"target_y=`[{_str(y['target_lo'])}, {_str(y['target_hi'])}]`; "
        f"which_dim_failed=`{failed_text}`."
    )


def write_report(
    out_dir: Path,
    aligned: list[Mapping[str, Any]],
    attempt_aligned: list[Mapping[str, Any]],
    forced_h: list[Mapping[str, Any]],
    *,
    flowstar_rows: list[Mapping[str, Any]],
    noqueue_rows: list[Mapping[str, Any]],
    v2_rows: list[Mapping[str, Any]],
    horizon: float,
    docs: bool = True,
) -> str:
    first_accepted = next((row for row in aligned if row.get("first_material_channel")), None)
    first_attempt_status = next((row for row in attempt_aligned if row.get("first_status_divergence")), None)
    first_attempt_numeric = next((row for row in attempt_aligned if row.get("first_numeric_channel_divergence") not in (None, "", "unknown")), None)
    first_forced_numeric = next((row for row in forced_h if row.get("first_numeric_channel_divergence") not in (None, "", "unknown")), None)
    first_forced_reject = next(
        (
            row
            for row in forced_h
            if str(row.get("torch_noqueue_accepts_flowstar_h", "")).lower() == "false"
            or str(row.get("torch_v2_accepts_flowstar_h", "")).lower() == "false"
        ),
        None,
    )
    causal = first_attempt_status or first_attempt_numeric
    if first_attempt_status:
        executive = "First causal divergence: adaptive acceptance / residual validation."
        recommendation = "Expose and compare polynomial/remainder/raw-ctrunc/no-remainder decomposition."
    elif first_forced_numeric:
        executive = f"First forced-h numeric divergence: {first_forced_numeric.get('first_numeric_channel_divergence')}."
        recommendation = "Investigate Flow* right-map/preconditioning/source-order semantics."
    else:
        executive = "No causal numeric channel was isolated over this short trace."
        recommendation = "Investigate cutoff/poly-diff accounting."

    flow_025 = _find_attempt_at(flowstar_rows, t=0.0, h=0.025)
    noq_025 = _find_attempt_at(noqueue_rows, t=0.0, h=0.025)
    v2_025 = _find_attempt_at(v2_rows, t=0.0, h=0.025)
    flow_rejects_025 = _status(flow_025) == "rejected"
    noq_accepts_025 = _status(noq_025) == "accepted"
    v2_accepts_025 = _status(v2_025) == "accepted"
    forced_accepts_all = bool(forced_h) and all(
        str(row.get("torch_noqueue_accepts_flowstar_h", "")).lower() == "true"
        and str(row.get("torch_v2_accepts_flowstar_h", "")).lower() == "true"
        for row in forced_h
    )

    lines = [
        "# Flow* Step Trace Divergence Report",
        "",
        "This is a diagnostic probe, not a Flow* parity claim.",
        "",
        "## Executive conclusion",
        "",
        f"- Horizon traced: T={horizon:g}",
        f"- {executive}",
        "- Accepted ordinal comparisons are retained only as noncausal diagnostics when `t` or `h` differ.",
        "",
        "## Accepted ordinal comparison",
        "",
    ]
    if first_accepted:
        attribution_valid = str(first_accepted.get("channel_attribution_valid", "")).lower() != "false"
        lines.extend(
            [
                f"- Comparison kind: `{first_accepted.get('comparison_kind', '')}`",
                f"- First material channel: `{first_accepted.get('first_material_channel', '')}`",
                f"- Channel attribution valid: `{'true' if attribution_valid else 'false'}`",
                f"- Flow* h: `{first_accepted.get('flowstar_h', '')}`; no_queue h: `{first_accepted.get('noqueue_h', '')}`; v2 h: `{first_accepted.get('v2_h', '')}`",
                f"- Warning: {first_accepted.get('alignment_warning', '') or 'none'}",
                "",
            ]
        )
    else:
        lines.extend(["- No accepted ordinal material channel was found.", ""])

    lines.extend(
        [
            "## Attempt-aligned comparison",
            "",
            f"- Does Flow* reject h=0.025 at t=0? `{'yes' if flow_rejects_025 else 'no'}`",
            f"- Does PyTorch no_queue accept h=0.025 at t=0? `{'yes' if noq_accepts_025 else 'no'}`",
            f"- Does PyTorch v2 accept h=0.025 at t=0? `{'yes' if v2_accepts_025 else 'no'}`",
            f"- Flow* h=0.025 evidence: {_attempt_fact(flow_025)}",
            f"- no_queue h=0.025 evidence: {_attempt_fact(noq_025)}",
            f"- v2 h=0.025 evidence: {_attempt_fact(v2_025)}",
        ]
    )
    if first_attempt_status:
        lines.extend(
            [
                f"- First causal divergence: `{first_attempt_status.get('first_status_divergence')}` at t=`{first_attempt_status.get('t_before')}`, h=`{first_attempt_status.get('h_try')}`.",
                f"- Flow* rejection reason: {first_attempt_status.get('flowstar_rejection_reason', '')}",
                "",
            ]
        )
    elif first_attempt_numeric:
        lines.extend(
            [
                f"- First numeric divergence under matched attempts: `{first_attempt_numeric.get('first_numeric_channel_divergence')}`.",
                "",
            ]
        )
    else:
        lines.extend(["- No aligned attempt status divergence was found before numeric comparison became unknown or clean.", ""])

    lines.extend(
        [
            "## Acceptance predicate endpoints",
            "",
            _predicate_endpoint_fact("Flow* h=0.025", flow_025),
            _predicate_endpoint_fact("PyTorch no_queue h=0.025", noq_025),
            _predicate_endpoint_fact("PyTorch v2 h=0.025", v2_025),
            "- Width comparison is not the acceptance predicate; endpoint-wise interval inclusion is. A residual may have smaller width than the target and still fail if it is shifted outside the target interval.",
            "- Detailed component ledger: `outputs/flowstar_acceptance_predicate_audit/acceptance_predicate_ledger.csv`.",
            "",
        ]
    )

    lines.extend(["## Forced-h replay", ""])
    if first_forced_reject:
        lines.extend(
            [
                "- PyTorch does not accept every Flow* accepted h in the replayed schedule.",
                f"- First forced reject/missing step: `{first_forced_reject.get('forced_step_index')}` at h=`{first_forced_reject.get('h_forced')}`.",
                f"- no_queue accepts Flow* h: `{first_forced_reject.get('torch_noqueue_accepts_flowstar_h')}`; v2 accepts Flow* h: `{first_forced_reject.get('torch_v2_accepts_flowstar_h')}`",
            ]
        )
    else:
        lines.append(f"- Under the Flow* accepted h schedule, PyTorch accepts all replayed rows present in the ledger: `{'yes' if forced_accepts_all else 'no'}`")
    if first_forced_numeric:
        lines.extend(
            [
                f"- First numeric channel divergence: `{first_forced_numeric.get('first_numeric_channel_divergence')}` at forced step `{first_forced_numeric.get('forced_step_index')}`.",
                f"- right_map ratios no_queue/v2: `{first_forced_numeric.get('right_map_width_ratio_noqueue_over_flowstar')}` / `{first_forced_numeric.get('right_map_width_ratio_v2_over_flowstar')}`",
                f"- reset ratios no_queue/v2: `{first_forced_numeric.get('reset_width_ratio_noqueue_over_flowstar')}` / `{first_forced_numeric.get('reset_width_ratio_v2_over_flowstar')}`",
                f"- output_range ratios no_queue/v2: `{first_forced_numeric.get('output_range_width_ratio_noqueue_over_flowstar')}` / `{first_forced_numeric.get('output_range_width_ratio_v2_over_flowstar')}`",
            ]
        )
    else:
        lines.append("- First numeric channel divergence: `unknown` or not reached.")
    lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The attempt-aligned comparator is the causal guard: channel attribution is valid only when `t_before`, `h_try`, and attempt index align.",
            "- The accepted ordinal diff remains useful for regression monitoring, but its first row compares different step sizes and must not be used as first causal channel attribution.",
        ]
    )
    if first_attempt_status and first_forced_numeric:
        lines.append("- The adaptive acceptance divergence occurs before the forced-h numeric channel divergence.")
    if first_forced_numeric and first_forced_numeric.get("first_numeric_channel_divergence") in {"right_map_range", "reset_new_x0", "output_range"}:
        lines.append("- The forced-h result is consistent with the prior right_map/preconditioning/output-range width attribution thread.")
    lines.extend(
        [
            "",
            "## Next recommendation",
            "",
            f"- {recommendation}",
            "",
            "## Output files",
            "",
            "- `outputs/flowstar_step_trace_compare/flowstar_trace.csv`",
            "- `outputs/flowstar_step_trace_compare/torch_noqueue_trace.csv`",
            "- `outputs/flowstar_step_trace_compare/torch_v2_trace.csv`",
            "- `outputs/flowstar_step_trace_compare/aligned_trace_diff.csv`",
            "- `outputs/flowstar_step_trace_compare/attempt_aligned_trace_diff.csv`",
            "- `outputs/flowstar_step_trace_compare/forced_h_trace_diff.csv`",
            "- `outputs/flowstar_step_trace_compare/attempt_alignment_warnings.csv`",
            "- `outputs/flowstar_step_trace_compare/forced_h_width_channel_ledger.csv`",
            "",
            "## Limitations",
            "",
            "- The Flow* C++ probe is an oracle/instrumentation probe; this change does not add a new flowpipe mechanism or symbolic queue variant.",
            "- Fields absent in a mode are left blank in the trace and reported as unknown by the comparator.",
            "- This report does not compare PyTorch endpoint boxes to Flow* GNUPLOT segment boxes.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (out_dir / "trace_divergence_report.md").write_text(text, encoding="utf-8")
    if docs:
        (ROOT / "docs" / "flowstar_step_trace_divergence_report.md").write_text(text, encoding="utf-8")
    return text


def write_plan_doc() -> None:
    text = """# Flow* Accepted-Step Trace Plan

This diagnostic uses a repo-local C++ probe in `experiments/flowstar_probe/` and does not commit Flow* source changes.

The probe links against `/srv/local/shengenli/flowstar/flowstar-toolbox/libflowstar.a` and mirrors the local Van der Pol adaptive symbolic-remainder step path with original Flow* settings: adaptive step `0.002..0.1`, order `4`, cutoff `1e-10`, target remainder `[-1e-4, 1e-4]`, initial box `x=[1.1,1.4]`, `y=[2.35,2.45]`.

Trace rows are emitted per adaptive attempt, including rejected shrink attempts and accepted steps. The aligned comparator keeps Flow* as the diagnostic reference and converts existing PyTorch normalized insertion diagnostics for `normalized_insertion` no_queue and `normalized_insertion_symqueue_v2`/`flowstar_linear_v2` into the same columns.

Required channels:

- accepted-step timing: `t_before`, `h`, accepted/rejected status
- pre/right map range widths and normal range widths
- endpoint range before center extraction, center, scale, inverse scale
- normalized `new_x0` box/range and target remainder
- Picard no-remainder and ctrunc residual widths
- cutoff/polynomial-difference contribution when available
- symbolic queue sizes, scalar values, and symbolic width contributions
- final flowpipe segment width

This is not a Flow* parity proof. It is a short-horizon probe for localizing the first material divergence channel.
"""
    (ROOT / "docs" / "flowstar_accepted_step_trace_plan.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-root", default="/srv/local/shengenli/flowstar")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--skip-flowstar", action="store_true")
    parser.add_argument("--compare-mode", choices=["accepted_ordinal", "attempt_aligned", "forced_h", "all"], default="all")
    parser.add_argument("--flowstar-trace", type=Path)
    parser.add_argument("--torch-noqueue-trace", type=Path)
    parser.add_argument("--torch-v2-trace", type=Path)
    parser.add_argument("--torch-noqueue-forced-trace", type=Path)
    parser.add_argument("--torch-v2-forced-trace", type=Path)
    return parser.parse_args()


def _load_or_generate_flowstar(args: argparse.Namespace, out_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    if args.flowstar_trace:
        path = args.flowstar_trace.resolve()
        return path, _read_rows(path)
    flowstar_trace = out_dir / "flowstar_trace.csv"
    if not args.skip_flowstar:
        exe = compile_probe(Path(args.flowstar_root).resolve(), out_dir, compiler=args.compiler)
        flowstar_trace = run_flowstar_probe(exe, out_dir, args.horizon, args.max_segments or None)
    return flowstar_trace, _read_rows(flowstar_trace)


def _load_or_generate_torch(args: argparse.Namespace, out_dir: Path, mode: str) -> list[dict[str, Any]]:
    trace_arg = args.torch_noqueue_trace if mode == "torch_noqueue" else args.torch_v2_trace
    if trace_arg:
        return _read_rows(trace_arg.resolve())
    return generate_torch_trace(
        mode=mode,
        horizon=args.horizon,
        out_path=out_dir / f"{mode}_trace.csv",
        max_segments=args.max_segments or None,
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_plan_doc()

    _flowstar_trace, flowstar_rows = _load_or_generate_flowstar(args, out_dir)
    noqueue_rows = _load_or_generate_torch(args, out_dir, "torch_noqueue")
    v2_rows = _load_or_generate_torch(args, out_dir, "torch_v2")

    aligned: list[dict[str, Any]] = []
    attempt_aligned: list[dict[str, Any]] = []
    forced_h: list[dict[str, Any]] = []

    if args.compare_mode in {"accepted_ordinal", "all"}:
        aligned = align_traces(flowstar_rows, noqueue_rows, v2_rows)
        _write_rows(out_dir / "aligned_trace_diff.csv", DIFF_FIELDS, aligned)
        _write_rows(out_dir / "step_alignment_warnings.csv", STEP_ALIGNMENT_WARNING_FIELDS, step_alignment_warnings(aligned))

    if args.compare_mode in {"attempt_aligned", "all"}:
        attempt_aligned = compare_attempt_aligned(flowstar_rows, noqueue_rows, v2_rows)
        _write_rows(out_dir / "attempt_aligned_trace_diff.csv", ATTEMPT_ALIGNED_FIELDS, attempt_aligned)
        warnings = [row for row in attempt_aligned if str(row.get("channel_attribution_valid", "")).lower() == "false"]
        _write_rows(out_dir / "attempt_alignment_warnings.csv", ATTEMPT_ALIGNED_FIELDS, warnings)

    if args.compare_mode in {"forced_h", "all"}:
        if args.torch_noqueue_forced_trace:
            forced_noqueue_rows = _read_rows(args.torch_noqueue_forced_trace.resolve())
        elif args.torch_noqueue_trace:
            forced_noqueue_rows = noqueue_rows
        else:
            forced_noqueue_rows = generate_torch_forced_h_trace(
                mode="torch_noqueue",
                flowstar_rows=flowstar_rows,
                out_path=out_dir / "torch_noqueue_forced_h_trace.csv",
            )
        if args.torch_v2_forced_trace:
            forced_v2_rows = _read_rows(args.torch_v2_forced_trace.resolve())
        elif args.torch_v2_trace:
            forced_v2_rows = v2_rows
        else:
            forced_v2_rows = generate_torch_forced_h_trace(
                mode="torch_v2",
                flowstar_rows=flowstar_rows,
                out_path=out_dir / "torch_v2_forced_h_trace.csv",
            )
        forced_h = compare_forced_h(flowstar_rows, forced_noqueue_rows, forced_v2_rows)
        _write_rows(out_dir / "forced_h_trace_diff.csv", FORCED_H_FIELDS, forced_h)
        _write_rows(out_dir / "forced_h_width_channel_ledger.csv", FORCED_H_FIELDS, forced_h)

    write_report(
        out_dir,
        aligned,
        attempt_aligned,
        forced_h,
        flowstar_rows=flowstar_rows,
        noqueue_rows=noqueue_rows,
        v2_rows=v2_rows,
        horizon=args.horizon,
        docs=out_dir == DEFAULT_OUT.resolve(),
    )
    print(f"Wrote traces, comparators, and report to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
