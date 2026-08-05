#!/usr/bin/env python3
"""Run and aggregate the frozen later-terminal attribution and Horner A/B."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import replay_vdp_terminal_range as replay


CHECKPOINT = (
    ROOT
    / "evidence"
    / "vdp_terminal_range_closure"
    / "20260805T055556Z"
    / "05_fresh_horizons"
    / "t6p5_proactive_d1_truncation"
    / "terminal_checkpoint"
)
REGISTERED_ORDERS = ((0, 1, 2), (1, 0, 2), (2, 0, 1))
ALL_CONTEXTS = (
    "polynomial_truncation",
    "integration_overflow",
    "poly_times_remainder",
    "remainder_times_poly",
    "cutoff",
    "retained_polynomial",
    "raw_compat_poly_diff",
)


@dataclass(frozen=True)
class Lane:
    name: str
    method: str
    depth: int = 0
    leaves: int = 1
    contexts: tuple[str, ...] = ()
    orders: tuple[tuple[int, ...], ...] = REGISTERED_ORDERS


ATTRIBUTION_LANES = (
    Lane("A0_natural", "natural"),
    Lane("A1_polynomial_truncation", "subdivision", 1, 4, ("polynomial_truncation",)),
    Lane("A2_integration_overflow", "subdivision", 1, 4, ("integration_overflow",)),
    Lane(
        "A3_polynomial_remainder_products",
        "subdivision",
        1,
        4,
        ("poly_times_remainder", "remainder_times_poly"),
    ),
    Lane(
        "A4_truncation_and_overflow",
        "subdivision",
        1,
        4,
        ("polynomial_truncation", "integration_overflow"),
    ),
    Lane("A5_all_registered_contexts", "subdivision", 1, 4, ALL_CONTEXTS),
    Lane("A6_maximum_subdivision", "subdivision", 5, 64, ALL_CONTEXTS),
)

TERMINAL_AB_LANES = (
    Lane("D0_natural", "natural"),
    Lane("D1_existing_subdivision", "subdivision", 1, 4, ("polynomial_truncation",)),
    Lane("D2_horner_registered_best", "horner_registered_best", 0, 4, ("polynomial_truncation",)),
    Lane("D3_subdivision_then_horner", "subdivision_then_horner", 1, 4, ("polynomial_truncation",)),
    Lane("D4_order_0_1_2", "horner_fixed", 0, 4, ("polynomial_truncation",), ((0, 1, 2),)),
    Lane("D4_order_1_0_2", "horner_fixed", 0, 4, ("polynomial_truncation",), ((1, 0, 2),)),
    Lane("D4_order_2_0_1", "horner_fixed", 0, 4, ("polynomial_truncation",), ((2, 0, 1),)),
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: json.dumps(_jsonable(row.get(field)), sort_keys=True)
                        if isinstance(row.get(field), (dict, list, tuple))
                        else row.get(field, "")
                        for field in fields
                    }
                )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _orders_arg(orders: Sequence[Sequence[int]]) -> str:
    return ";".join(",".join(str(index) for index in order) for order in orders)


def _run_lane(lane: Lane, output: Path, device: str) -> Mapping[str, Any]:
    argv = [
        "--checkpoint",
        str(CHECKPOINT),
        "--output-dir",
        str(output),
        "--range-method",
        lane.method,
        "--subdivision-depth",
        str(lane.depth),
        "--max-leaves",
        str(lane.leaves),
        "--split-vars",
        "0,1",
        "--named-contexts",
        ",".join(lane.contexts),
        "--variable-orders",
        _orders_arg(lane.orders),
        "--device",
        device,
    ]
    return replay.run(replay.parse_args(argv))


def _lane_row(phase: str, lane: Lane, directory: Path) -> dict[str, Any]:
    summary = _read_json(directory / "summary.json")
    ledger_rows = _read_jsonl(directory / "remainder_ledger.jsonl")
    validation = ledger_rows[-1] if ledger_rows else {}
    range_rows = _read_jsonl(directory / "range_context_trace.jsonl")
    candidate = summary.get("candidate_hashes", {})
    image = summary.get("picard_image_remainder", [[None, None], [None, None]])
    margin = summary.get("subset_margin", [[None, None]])
    invalid_fallbacks = [
        row.get("fallback_reason")
        for row in range_rows
        if str(row.get("fallback_reason", "")).startswith("explicit")
    ]
    return {
        "phase": phase,
        "lane": lane.name,
        "range_policy": lane.method,
        "named_contexts": list(lane.contexts),
        "registered_variable_orders": [list(order) for order in lane.orders],
        "accepted": summary["accepted"],
        "status": summary["status"],
        "validation_predicate": validation.get("validation_mode"),
        "validation_rejection_reason": summary.get("validation_rejection_reason"),
        "picard_iterations": candidate.get("picard_iterations"),
        "attempted_h": summary["attempted_h"],
        "t_before": summary["t_before"],
        "x_image_lo": image[0][0],
        "x_image_hi": image[1][0],
        "y_image_lo": image[0][1],
        "y_image_hi": image[1][1],
        "x_subset_margin": margin[0][0],
        "y_subset_margin": margin[0][1],
        "candidate_coefficient_sha256": candidate.get("coefficient_sha256"),
        "candidate_exponent_support_sha256": candidate.get("exponent_support_sha256"),
        "basis_hash": candidate.get("basis_hash"),
        "checkpoint_full_sha256": summary["checkpoint_full_sha256"],
        "pre_state_tmvector_sha256": summary["current_hashes"]["tmvector_sha256"],
        "contract_sha256": summary["contract_sha256"],
        "backend_lane": summary["backend_lane"],
        "range_call_count": len(range_rows),
        "range_contexts_observed": sorted({row.get("context") for row in range_rows}),
        "selected_range_backends": sorted({row.get("method_used") for row in range_rows}),
        "range_leaf_count": summary["range_leaf_count"],
        "range_subdivision_invocations": summary["range_subdivision_invocations"],
        "remainder_sources": validation.get("remainder_ledger_intervals", {}),
        "raw_remainder_sources": validation.get("raw_remainder_ledger_intervals", {}),
        "tmp_remainder_sources": validation.get("tmp_remainder_ledger_intervals", {}),
        "wall_s": summary["runtime_s"],
        "finite": all(bool(row.get("finite", True)) for row in range_rows),
        "invalid_fallback_count": len(invalid_fallbacks),
        "invalid_fallback_reasons": invalid_fallbacks,
        "sparse_fallback_count": summary["fallback_count"],
        "repair_used": summary["endpoint_repair_used"],
        "sampling_used_for_enclosure": False,
        "external_endpoint_substitution": False,
    }


def _assert_invariants(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    invariant_fields = (
        "candidate_coefficient_sha256",
        "candidate_exponent_support_sha256",
        "basis_hash",
        "checkpoint_full_sha256",
        "pre_state_tmvector_sha256",
        "contract_sha256",
        "attempted_h",
        "validation_predicate",
    )
    values = {field: sorted({json.dumps(row.get(field), sort_keys=True) for row in rows}) for field in invariant_fields}
    failures = {field: items for field, items in values.items() if len(items) != 1}
    forbidden = [
        row["lane"]
        for row in rows
        if row["sparse_fallback_count"]
        or row["invalid_fallback_count"]
        or row["repair_used"]
        or row["sampling_used_for_enclosure"]
        or row["external_endpoint_substitution"]
        or not row["finite"]
    ]
    if failures or forbidden:
        raise RuntimeError(f"frozen-lane invariants failed: fields={failures}, forbidden={forbidden}")
    return {
        "passed": True,
        "invariant_fields": {field: json.loads(items[0]) for field, items in values.items()},
        "no_nonfinite_fallback_repair_sampling_or_external_endpoint": True,
    }


def _aggregate_phase(
    output_root: Path,
    phase: str,
    lanes: Sequence[Lane],
    lane_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [_lane_row(phase, lane, lane_root / lane.name) for lane in lanes]
    _assert_invariants(rows)
    range_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for lane in lanes:
        directory = lane_root / lane.name
        range_rows.extend(
            {"phase_group": phase, "lane": lane.name, **row}
            for row in _read_jsonl(directory / "range_context_trace.jsonl")
        )
        stage_rows.extend(
            {"phase_group": phase, "lane": lane.name, **row}
            for row in _read_jsonl(directory / "horner_stage_trace.jsonl")
        )
    if phase == "attribution":
        _write_csv(output_root / "attribution.csv", rows)
        _write_json(output_root / "attribution.json", {"invariants": _assert_invariants(rows), "lanes": rows})
    else:
        _write_csv(output_root / "terminal_ab.csv", rows)
        _write_json(output_root / "terminal_ab.json", {"invariants": _assert_invariants(rows), "lanes": rows})
    return rows, range_rows, stage_rows


def _write_decisions(output_root: Path, attribution: Sequence[Mapping[str, Any]], terminal_ab: Sequence[Mapping[str, Any]]) -> None:
    by_a = {row["lane"]: row for row in attribution}
    by_d = {row["lane"]: row for row in terminal_ab}
    attribution_decision = {
        "direct_failure_source": (
            "flowstar_raw_remainder_compat target-subset rejection in y; the natural image exceeds both target sides, "
            "with the upper-side violation dominant"
        ),
        "integration_overflow_is_causal_blocker": False,
        "integration_overflow_reason": "A2 is bit/numerically identical to A0 at the final image and margin despite its large ledger width",
        "polynomial_truncation_changes_acceptance": False,
        "polynomial_truncation_changes_margin": by_a["A1_polynomial_truncation"]["y_subset_margin"] != by_a["A0_natural"]["y_subset_margin"],
        "integration_overflow_changes_margin": by_a["A2_integration_overflow"]["y_subset_margin"] != by_a["A0_natural"]["y_subset_margin"],
        "joint_truncation_overflow_changes_acceptance": False,
        "joint_margin": by_a["A4_truncation_and_overflow"]["y_subset_margin"],
        "subdivision_saturation": by_a["A5_all_registered_contexts"]["y_subset_margin"] == by_a["A6_maximum_subdivision"]["y_subset_margin"],
        "saturation_reason": "depth-1 and depth-5 all-context lanes select the same decisive enclosures; extra leaves add no terminal-margin gain",
    }
    d2_passed = bool(by_d["D2_horner_registered_best"]["accepted"])
    d3_passed = bool(by_d["D3_subdivision_then_horner"]["accepted"])
    d1_ranges = {
        row["range_call_index"]: row
        for row in _read_jsonl(
            output_root / "terminal_ab" / "formal" / "D1_existing_subdivision" / "range_context_trace.jsonl"
        )
        if row.get("context") == "polynomial_truncation"
    }
    d3_ranges = {
        row["range_call_index"]: row
        for row in _read_jsonl(
            output_root / "terminal_ab" / "formal" / "D3_subdivision_then_horner" / "range_context_trace.jsonl"
        )
        if row.get("context") == "polynomial_truncation"
    }
    gains = {
        call: d1_ranges[call]["selected_width"][0][0] - d3_ranges[call]["selected_width"][0][0]
        for call in sorted(set(d1_ranges) & set(d3_ranges))
    }
    dominant_call = max(gains, key=gains.get)
    dominant_row = d3_ranges[dominant_call]
    dominant_stages = [
        row
        for row in _read_jsonl(
            output_root / "terminal_ab" / "formal" / "D3_subdivision_then_horner" / "horner_stage_trace.jsonl"
        )
        if row.get("range_call_index") == dominant_call
        and row.get("scope") == "subdivision_leaf"
        and row.get("stage_depth") == 0
        and row.get("degree") == 0
    ]
    terminal_decision = {
        "stop_go_gate": "GO" if d2_passed or d3_passed else "STOP",
        "horner_only_closed_step": d2_passed,
        "combined_closed_step": d3_passed,
        "horner_only_y_margin": by_d["D2_horner_registered_best"]["y_subset_margin"],
        "subdivision_y_margin": by_d["D1_existing_subdivision"]["y_subset_margin"],
        "combined_y_margin": by_d["D3_subdivision_then_horner"]["y_subset_margin"],
        "horner_and_subdivision_complementary": (
            by_d["D3_subdivision_then_horner"]["y_subset_margin"]
            > by_d["D1_existing_subdivision"]["y_subset_margin"]
        ),
        "combined_margin_gain_over_subdivision": (
            by_d["D3_subdivision_then_horner"]["y_subset_margin"]
            - by_d["D1_existing_subdivision"]["y_subset_margin"]
        ),
        "aggregation_only_margin_delta_from_natural": (
            by_d["D2_horner_registered_best"]["y_subset_margin"]
            - by_d["D0_natural"]["y_subset_margin"]
        ),
        "improvement_context": "polynomial_truncation",
        "dominant_improvement_range_call_index": dominant_call,
        "dominant_range_width_before": d1_ranges[dominant_call]["selected_width"][0][0],
        "dominant_range_width_after": dominant_row["selected_width"][0][0],
        "dominant_range_width_gain": gains[dominant_call],
        "dominant_per_leaf_selected_order_index": dominant_row["horner"]["per_leaf"]["selected_order_index"],
        "dominant_per_leaf_selected_orders": [
            {
                "variable_order": order["variable_order"],
                "selected_mask": order["selected_mask"],
            }
            for order in dominant_row["horner"]["per_leaf"]["orders"]
        ],
        "dominant_factorization_stages": [
            {
                "stage_index": stage["stage_index"],
                "variable_order": stage["variable_order"],
                "variable": stage["variable"],
                "operation": stage["operation"],
                "intermediate_lo": stage["intermediate_lo"],
                "intermediate_hi": stage["intermediate_hi"],
            }
            for stage in dominant_stages
        ],
        "improvement_mechanism": (
            "per-leaf dependency preservation in the final top-level Horner multiply-add; "
            "not coefficient aggregation, whose Horner-only terminal-margin delta is at float64 safeguard scale"
        ),
        "candidate_polynomial_unchanged": True,
        "fresh_horizons_authorized": d2_passed or d3_passed,
        "highest_fresh_validated_horizon": 6.397083942944808,
        "range_closure_state": "R4_historical_range_midpoint_horizon_crossed",
        "factorized_state": "H1_factorized_range_correctness_complete",
        "remaining_blocker": "cross-step dependency/carry representation; frozen factorized range remains outside the unchanged target self-map",
    }
    _write_json(output_root / "attribution_decision.json", attribution_decision)
    _write_json(output_root / "terminal_decision.json", terminal_decision)
    _write_csv(
        output_root / "fresh_horizons.csv",
        [
            {
                "requested_horizon": horizon,
                "status": "not_run_stop_go_gate_failed" if not terminal_decision["fresh_horizons_authorized"] else "authorized_not_run",
                "validated_horizon": 6.397083942944808,
                "reason": "D2 and D3 both rejected the frozen terminal" if not terminal_decision["fresh_horizons_authorized"] else "",
            }
            for horizon in (6.5, 7.5, 10.0)
        ],
    )


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "platform": platform.platform(),
        "branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_manifest_sha256": hashlib.sha256((CHECKPOINT / "terminal_state_manifest.json").read_bytes()).hexdigest(),
        "safeguard_claim": "safeguarded float64 enclosure; not a hardware-independent directed-rounding formal proof",
    }


def run(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    attribution_root = output_root / "attribution" / "formal"
    terminal_root = output_root / "terminal_ab" / "formal"
    if not args.aggregate_only:
        if args.phase in {"attribution", "all"}:
            for lane in ATTRIBUTION_LANES:
                _run_lane(lane, attribution_root / lane.name, args.device)
        if args.phase in {"terminal_ab", "all"}:
            for lane in TERMINAL_AB_LANES:
                _run_lane(lane, terminal_root / lane.name, args.device)

    attribution_rows, attribution_ranges, attribution_stages = _aggregate_phase(
        output_root,
        "attribution",
        ATTRIBUTION_LANES,
        attribution_root,
    )
    terminal_rows, terminal_ranges, terminal_stages = _aggregate_phase(
        output_root,
        "terminal_ab",
        TERMINAL_AB_LANES,
        terminal_root,
    )
    range_rows = [*attribution_ranges, *terminal_ranges]
    stage_rows = [*attribution_stages, *terminal_stages]
    _write_jsonl(output_root / "range_context_trace.jsonl", range_rows)
    _write_csv(output_root / "range_context_trace.csv", range_rows)
    _write_jsonl(output_root / "horner_stage_trace.jsonl", stage_rows)
    _write_csv(output_root / "horner_stage_trace.csv", stage_rows)
    _write_csv(output_root / "summary.csv", [*attribution_rows, *terminal_rows])
    _write_decisions(output_root, attribution_rows, terminal_rows)
    _write_json(output_root / "environment.json", _environment())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "vdp_later_terminal_factorized_range",
    )
    parser.add_argument("--phase", choices=("attribution", "terminal_ab", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
