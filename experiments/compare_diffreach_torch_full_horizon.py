#!/usr/bin/env python3
"""Fail-closed comparison for pinned DiffReach versus Torch DR7 traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np

from diffreach_torch_full_horizon_common import (
    DIFFREACH_SOURCE_SHA,
    PARTITION_SHA256,
    SCHEMA,
    SUPPORT_SHA256,
    write_json,
)


FIELD_ORDER = (
    "pre_model_c", "pre_model_L", "pre_model_Lt", "pre_model_R_lo", "pre_model_R_hi",
    "pre_parameterization_c", "pre_parameterization_L", "pre_parameterization_Lt",
    "pre_parameterization_R_lo", "pre_parameterization_R_hi",
    "pre_J_lo", "pre_J_hi", "pre_Phi", "pre_queue_count", "pre_inverse_scale",
    "endpoint_previous_c", "endpoint_previous_L", "endpoint_previous_Lt",
    "endpoint_previous_R_lo", "endpoint_previous_R_hi", "center", "scale", "inverse_scale",
    "normalized_c", "normalized_L", "normalized_Lt", "normalized_R_lo", "normalized_R_hi",
    "poly1_c", "poly1_L", "poly1_Lt", "poly2_c", "poly2_L", "poly2_Lt",
    "initial_inclusion_mask", "roundoff_lo", "roundoff_hi", "round_masks",
    "round_accepted_lo", "round_accepted_hi", "retained_c", "retained_L", "retained_Lt",
    "retained_R_lo", "retained_R_hi", "composed_c", "composed_L", "composed_Lt",
    "composed_R_lo", "composed_R_hi", "endpoint_lo", "endpoint_hi", "tube_lo", "tube_hi",
    "post_J_lo", "post_J_hi", "post_Phi", "post_queue_count", "queue_clear_event",
    "active_mask", "failure_mask", "prefix_tube_hull_lo", "prefix_tube_hull_hi",
)
MASK_FIELDS = (
    "initial_inclusion_mask", "round_masks", "active_mask", "failure_mask",
)
J_PHI_FIELDS = (
    "pre_J_lo", "pre_J_hi", "pre_Phi", "pre_queue_count",
    "post_J_lo", "post_J_hi", "post_Phi", "post_queue_count", "queue_clear_event",
)
ENDPOINT_TUBE_FIELDS = (
    "endpoint_lo", "endpoint_hi", "tube_lo", "tube_hi",
    "prefix_tube_hull_lo", "prefix_tube_hull_hi",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffreach-dir", type=Path, required=True)
    parser.add_argument("--torch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistered-max-ulp", type=int, default=2)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ordered_fields(fields: Iterable[str]) -> list[str]:
    available = set(fields)
    return [name for name in FIELD_ORDER if name in available] + sorted(
        available.difference(FIELD_ORDER)
    )


def _same(row_left: dict[str, Any], row_right: dict[str, Any], field: str) -> bool:
    return row_left["fields"][field] == row_right["fields"][field]


def _ordered_float_bits(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    return (~bits & ((1 << 64) - 1)) if bits & (1 << 63) else bits | (1 << 63)


def _ulp_distance(left: float, right: float) -> int:
    if math.isnan(left) or math.isnan(right):
        return (1 << 64) - 1
    return abs(_ordered_float_bits(left) - _ordered_float_bits(right))


def _metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    absolute = np.abs(left - right)
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), np.finfo(np.float64).tiny)
    relative = absolute / denominator
    flat_left = left.ravel()
    flat_right = right.ravel()
    ulps = np.fromiter(
        (_ulp_distance(a, b) for a, b in zip(flat_left, flat_right)),
        dtype=np.uint64,
        count=flat_left.size,
    )
    max_abs_flat = int(np.argmax(absolute.ravel()))
    max_ulp_flat = int(np.argmax(ulps))
    return {
        "max_abs": float(absolute.ravel()[max_abs_flat]),
        "max_abs_index": list(np.unravel_index(max_abs_flat, left.shape)),
        "max_rel": float(relative.max(initial=0.0)),
        "max_ulp": int(ulps[max_ulp_flat]) if ulps.size else 0,
        "max_ulp_index": list(np.unravel_index(max_ulp_flat, left.shape)),
    }


def _first_numeric_difference(
    diff_dir: Path,
    torch_dir: Path,
    step: int,
    fields: Iterable[str],
) -> dict[str, Any] | None:
    diff_path = diff_dir / "captures" / f"step_{step:04d}.npz"
    torch_path = torch_dir / "captures" / f"step_{step:04d}.npz"
    if not diff_path.is_file() or not torch_path.is_file():
        return {"step": step, "capture_available": False}
    with np.load(diff_path) as left, np.load(torch_path) as right:
        for field in _ordered_fields(fields):
            if field not in left or field not in right:
                continue
            left_array = left[field]
            right_array = right[field]
            if np.array_equal(left_array, right_array):
                continue
            if left_array.shape != right_array.shape:
                return {
                    "step": step,
                    "field": field,
                    "capture_available": True,
                    "shape_mismatch": [list(left_array.shape), list(right_array.shape)],
                }
            different = np.flatnonzero(left_array.ravel() != right_array.ravel())
            flat_index = int(different[0])
            index = tuple(int(value) for value in np.unravel_index(flat_index, left_array.shape))
            left_value = left_array[index].item()
            right_value = right_array[index].item()
            result: dict[str, Any] = {
                "step": step,
                "field": field,
                "index": list(index),
                "diffreach_value": left_value,
                "torch_value": right_value,
                "capture_available": True,
            }
            if np.issubdtype(left_array.dtype, np.floating):
                absolute = abs(float(left_value) - float(right_value))
                denominator = max(abs(float(left_value)), abs(float(right_value)), np.finfo(np.float64).tiny)
                result.update(
                    {
                        "absolute_delta": absolute,
                        "relative_delta": absolute / denominator,
                        "ulp_delta": _ulp_distance(float(left_value), float(right_value)),
                        "diffreach_hex": float(left_value).hex(),
                        "torch_hex": float(right_value).hex(),
                    }
                )
            return result
    return None


def main() -> int:
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    diff_summary = _load_json(args.diffreach_dir / "summary.json")
    torch_summary = _load_json(args.torch_dir / "summary.json")
    required_common = {
        "schema": SCHEMA,
        "partition_sha256": PARTITION_SHA256,
        "support_sha256": SUPPORT_SHA256,
        "dtype": "float64",
        "steps": 1000,
        "step_size_hex": float(0.01).hex(),
        "completion_status": "completed",
        "all_initial_masks_true": True,
        "undeclared_fallback_or_repair": False,
    }
    for name, expected in required_common.items():
        for tool, summary in (("DiffReach", diff_summary), ("Torch", torch_summary)):
            if summary.get(name) != expected:
                raise RuntimeError(f"{tool} contract mismatch for {name}: {summary.get(name)!r}")
    if diff_summary.get("source_sha") != DIFFREACH_SOURCE_SHA:
        raise RuntimeError("pinned DiffReach source SHA mismatch")
    if diff_summary.get("observer_inertness_bit_exact") is not True:
        raise RuntimeError("DiffReach observer inertness gate missing")

    diff_rows = _rows(args.diffreach_dir / "trace.jsonl")
    torch_rows = _rows(args.torch_dir / "trace.jsonl")
    if len(diff_rows) != 1000 or len(torch_rows) != 1000:
        raise RuntimeError("full-horizon comparison requires exactly 1000 trace rows")
    for index, (left, right) in enumerate(zip(diff_rows, torch_rows), start=1):
        if left.get("step") != index or right.get("step") != index:
            raise RuntimeError("trace step sequence is not contiguous")
        if left.get("time_hex") != right.get("time_hex"):
            raise RuntimeError(f"logical time mismatch at step {index}")

    diff_fields = set(diff_rows[0]["fields"])
    torch_fields = set(torch_rows[0]["fields"])
    common_fields = diff_fields & torch_fields
    required_fields = set(FIELD_ORDER)
    missing = sorted(required_fields - common_fields)
    if missing:
        raise RuntimeError(f"required cross-tool trace fields are missing: {missing}")
    first_by_field: dict[str, int | None] = {}
    for field in _ordered_fields(common_fields):
        first_by_field[field] = next(
            (
                index
                for index, (left, right) in enumerate(zip(diff_rows, torch_rows), start=1)
                if not _same(left, right, field)
            ),
            None,
        )
    divergent = {field: step for field, step in first_by_field.items() if step is not None}
    first_step = min(divergent.values()) if divergent else None
    first_fields = (
        [field for field in _ordered_fields(common_fields) if first_by_field[field] == first_step]
        if first_step is not None
        else []
    )
    first_detail = (
        _first_numeric_difference(args.diffreach_dir, args.torch_dir, first_step, first_fields)
        if first_step is not None
        else None
    )

    mask_equality = all(first_by_field[field] is None for field in MASK_FIELDS)
    j_phi_equality = all(first_by_field[field] is None for field in J_PHI_FIELDS)
    endpoint_tube_bit_exact = all(
        first_by_field[field] is None for field in ENDPOINT_TUBE_FIELDS
    )

    with np.load(args.diffreach_dir / "bounds.npz") as left_bounds, np.load(
        args.torch_dir / "bounds.npz"
    ) as right_bounds:
        delta_csv = args.output_dir / "endpoint_tube_delta_by_step.csv"
        endpoint_max_ulp = 0
        tube_max_ulp = 0
        endpoint_max_abs = 0.0
        tube_max_abs = 0.0
        first_endpoint_step = None
        first_tube_step = None
        with delta_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "step", "time", "endpoint_max_abs", "endpoint_max_rel", "endpoint_max_ulp",
                    "tube_max_abs", "tube_max_rel", "tube_max_ulp",
                ),
            )
            writer.writeheader()
            for zero_index in range(1000):
                endpoint_left = np.stack(
                    (left_bounds["endpoint_lo"][zero_index], left_bounds["endpoint_hi"][zero_index])
                )
                endpoint_right = np.stack(
                    (right_bounds["endpoint_lo"][zero_index], right_bounds["endpoint_hi"][zero_index])
                )
                tube_left = np.stack(
                    (left_bounds["tube_lo"][zero_index], left_bounds["tube_hi"][zero_index])
                )
                tube_right = np.stack(
                    (right_bounds["tube_lo"][zero_index], right_bounds["tube_hi"][zero_index])
                )
                endpoint_metrics = _metrics(endpoint_left, endpoint_right)
                tube_metrics = _metrics(tube_left, tube_right)
                if endpoint_metrics["max_ulp"] and first_endpoint_step is None:
                    first_endpoint_step = zero_index + 1
                if tube_metrics["max_ulp"] and first_tube_step is None:
                    first_tube_step = zero_index + 1
                endpoint_max_ulp = max(endpoint_max_ulp, endpoint_metrics["max_ulp"])
                tube_max_ulp = max(tube_max_ulp, tube_metrics["max_ulp"])
                endpoint_max_abs = max(endpoint_max_abs, endpoint_metrics["max_abs"])
                tube_max_abs = max(tube_max_abs, tube_metrics["max_abs"])
                writer.writerow(
                    {
                        "step": zero_index + 1,
                        "time": (zero_index + 1) * 0.01,
                        "endpoint_max_abs": endpoint_metrics["max_abs"],
                        "endpoint_max_rel": endpoint_metrics["max_rel"],
                        "endpoint_max_ulp": endpoint_metrics["max_ulp"],
                        "tube_max_abs": tube_metrics["max_abs"],
                        "tube_max_rel": tube_metrics["max_rel"],
                        "tube_max_ulp": tube_metrics["max_ulp"],
                    }
                )

    two_ulp_companion_containment = (
        endpoint_max_ulp <= args.preregistered_max_ulp
        and tube_max_ulp <= args.preregistered_max_ulp
    )
    endpoint_tube_equality = endpoint_tube_bit_exact or two_ulp_companion_containment
    bit_exact_all = not divergent
    closed = mask_equality and j_phi_equality and endpoint_tube_equality
    if bit_exact_all:
        outcome = "DIFFREACH_TORCH_DR7_FULL_HORIZON_BIT_EXACT"
    elif closed:
        outcome = "DIFFREACH_TORCH_DR7_FULL_HORIZON_ULP_BOUNDED"
    else:
        outcome = "DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED"

    comparison = {
        "schema": "diffreach_torch_dr7_full_horizon_comparison_v1",
        "outcome": outcome,
        "scope": "full_horizon",
        "batch_size": 64,
        "steps": 1000,
        "step_size": 0.01,
        "step_size_hex": float(0.01).hex(),
        "partition_sha256": PARTITION_SHA256,
        "support_sha256": SUPPORT_SHA256,
        "diffreach_source_sha": DIFFREACH_SOURCE_SHA,
        "comparison": {
            "kind": "cross_tool",
            "left_tool": "diffreach",
            "right_tool": "torch",
        },
        "operator_equality": bit_exact_all,
        "initial_masks_all_true": True,
        "initial_mask_equality": first_by_field["initial_inclusion_mask"] is None,
        "later_mask_equality": first_by_field["round_masks"] is None,
        "mask_equality": mask_equality,
        "j_phi_equality": j_phi_equality,
        "endpoint_tube_bit_exact": endpoint_tube_bit_exact,
        "endpoint_tube_equality": endpoint_tube_equality,
        "preregistered_max_ulp": args.preregistered_max_ulp,
        "two_ulp_companion_containment": two_ulp_companion_containment,
        "max_ulp": max(endpoint_max_ulp, tube_max_ulp),
        "endpoint_max_ulp": endpoint_max_ulp,
        "tube_max_ulp": tube_max_ulp,
        "endpoint_max_abs": endpoint_max_abs,
        "tube_max_abs": tube_max_abs,
        "first_endpoint_divergence_step": first_endpoint_step,
        "first_tube_divergence_step": first_tube_step,
        "first_divergence_step": first_step,
        "first_divergence_fields": first_fields,
        "first_divergence_detail": first_detail,
        "first_divergence_by_field": first_by_field,
        "common_field_count": len(common_fields),
        "diffreach_only_fields": sorted(diff_fields - torch_fields),
        "torch_only_fields": sorted(torch_fields - diff_fields),
        "no_hidden_fallback": True,
        "observer_inertness": True,
        "soundness_scope": "empirically sampled ordinary float64; no formal directed-rounding claim",
        "timing_eligible": closed,
        "limitations": [
            "hash equality is bit-exact after canonical dtype/shape normalization",
            "the preregistered two-ULP rule is evaluated only, never widened after observing data",
            "performance comparison is forbidden when the semantics gate is not closed",
        ],
    }
    write_json(args.output_dir / "comparison.json", comparison)
    print(json.dumps(comparison, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

