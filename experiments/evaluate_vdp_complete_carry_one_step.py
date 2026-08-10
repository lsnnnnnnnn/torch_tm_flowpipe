#!/usr/bin/env python3
"""Frozen-input one-step grid for baseline versus complete-polynomial carry."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_vdp_dense_backend import load_contract
from torch_tm_flowpipe import (
    DenseRangePolicy,
    Interval,
    PolynomialODE,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
)

MODES = ("normalized_insertion", "normalized_insertion_complete_polynomial")
H_GRID = (0.1, 0.05, 0.025, 0.0125, 0.005, 0.002)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return float(value)


def _box(box: Sequence[Interval]) -> list[list[float]]:
    return [[_float(interval.lo), _float(interval.hi)] for interval in box]


def _tm_record(tm: TMVector | None) -> Mapping[str, Any] | None:
    if tm is None:
        return None
    coefficients = [
        [
            {"exponent": list(exponent), "coefficient": _float(coefficient)}
            for exponent, coefficient in sorted(model.polynomial.terms.items())
        ]
        for model in tm
    ]
    canonical = json.dumps(coefficients, sort_keys=True, separators=(",", ":")).encode()
    polynomial_ranges = [model.polynomial.evaluate_interval(model.domain) for model in tm]
    return {
        "coefficient_sha256": hashlib.sha256(canonical).hexdigest(),
        "coefficients": coefficients,
        "term_counts": [len(model.polynomial.terms) for model in tm],
        "maximum_degree": max(
            (sum(exponent) for model in tm for exponent in model.polynomial.terms), default=0
        ),
        "polynomial_range": _box(polynomial_ranges),
        "remainder": _box([model.remainder for model in tm]),
        "range": _box(tm.range_box()),
    }


def _run_once(
    *,
    ode: PolynomialODE,
    initial_box: Sequence[Interval],
    contract: Mapping[str, Any],
    mode: str,
    h: float,
) -> tuple[dict[str, Any], float]:
    diagnostics: list[dict[str, Any]] = []
    range_policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )
    start = time.perf_counter()
    segment = flowpipe_step_flowstar_style_adaptive(
        ode,
        initial_box,
        h=h,
        h_min=h,
        h_max=h,
        order=int(contract["requested_order"]),
        target_remainder_radius=float(contract["target_remainder_radius"]),
        cutoff_threshold=float(contract["cutoff"]),
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode=str(contract["validation_mode"]),
        reset_mode=mode,
        step_policy_mode=str(contract["step_policy_mode"]),
        tm_backend="dense",
        dense_device="cpu",
        dense_dtype=torch.float64,
        dense_range_policy=range_policy,
        diagnostics=diagnostics,
        diagnostics_context={"experiment": "complete_carry_one_step", "h": h, "mode": mode},
    )
    elapsed = time.perf_counter() - start
    record = {
        "mode": mode,
        "attempted_h": h,
        "accepted_h": float(segment.h) if segment.status == "validated" else None,
        "status": segment.status,
        "message": segment.message,
        "validation_attempts": int(segment.validation_attempts),
        "step_rejections": int(segment.step_rejections),
        "candidate_remainder": segment.candidate_remainder,
        "picard_image_remainder": segment.picard_image_remainder,
        "subset_margin": segment.subset_margin,
        "segment_tube": _tm_record(segment.tm),
        "raw_endpoint": _tm_record(segment.endpoint_raw_tm),
        "next_step_initial_tm": _tm_record(segment.reset_tm),
        "carry": {
            key: value
            for key, value in (segment.flowstar_normal_stats or {}).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "backend_counters": dict(segment.backend_counters or {}),
        "backend_trace": [dict(row) for row in (segment.backend_trace or ())],
        "diagnostics": diagnostics,
    }
    return record, elapsed


def _maximum_coefficient_difference(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> float | None:
    if left is None or right is None:
        return None
    left_terms = left["coefficients"]
    right_terms = right["coefficients"]
    if [[row["exponent"] for row in state] for state in left_terms] != [
        [row["exponent"] for row in state] for state in right_terms
    ]:
        return None
    return max(
        (
            abs(float(a["coefficient"]) - float(b["coefficient"]))
            for left_state, right_state in zip(left_terms, right_terms)
            for a, b in zip(left_state, right_state)
        ),
        default=0.0,
    )


def run(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    initial_box = [Interval(*bounds) for bounds in contract["initial_box"]]
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for h in H_GRID:
        paired: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            record, elapsed = _run_once(
                ode=ode, initial_box=initial_box, contract=contract, mode=mode, h=h
            )
            record["single_call_wall_s"] = elapsed
            rows.append(record)
            paired[mode] = record
        baseline = paired[MODES[0]]
        candidate = paired[MODES[1]]
        comparisons.append(
            {
                "h": h,
                "status_match": baseline["status"] == candidate["status"],
                "decision_match": (
                    baseline["status"], baseline["candidate_remainder"], baseline["subset_margin"]
                )
                == (
                    candidate["status"], candidate["candidate_remainder"], candidate["subset_margin"]
                ),
                "segment_coefficient_sha256_match": (
                    baseline["segment_tube"] or {}
                ).get("coefficient_sha256")
                == (candidate["segment_tube"] or {}).get("coefficient_sha256"),
                "endpoint_coefficient_sha256_match": (
                    baseline["raw_endpoint"] or {}
                ).get("coefficient_sha256")
                == (candidate["raw_endpoint"] or {}).get("coefficient_sha256"),
                "maximum_endpoint_coefficient_abs_difference": _maximum_coefficient_difference(
                    baseline["raw_endpoint"], candidate["raw_endpoint"]
                ),
                "candidate_next_initial_equals_raw_endpoint": (
                    candidate["next_step_initial_tm"] or {}
                ).get("coefficient_sha256")
                == (candidate["raw_endpoint"] or {}).get("coefficient_sha256"),
            }
        )

    timing: list[dict[str, Any]] = []
    for mode in MODES:
        samples: list[float] = []
        for _ in range(6):
            _record, elapsed = _run_once(
                ode=ode, initial_box=initial_box, contract=contract, mode=mode, h=0.002
            )
            samples.append(elapsed)
        timing.append(
            {
                "mode": mode,
                "boundary": "resident_process_complete_validated_step_and_reset_h_0.002",
                "cold_first_call_s": samples[0],
                "warm_repetitions_s": samples[1:],
                "warm_min_s": min(samples[1:]),
                "warm_median_s": sorted(samples[1:])[len(samples[1:]) // 2],
                "warm_max_s": max(samples[1:]),
                "cuda_synchronized": None,
                "batch": 1,
            }
        )

    provenance = {
        "source_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "worktree_status": _git("status", "--short"),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.run(
                ["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True
            ).stdout
        ).hexdigest(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "contract": contract,
    }
    result = {
        "schema_version": 1,
        "experiment": "vdp_complete_polynomial_carry_one_step_grid",
        "semantics": (
            "Each row starts independently from the authoritative initial box and attempts exactly one "
            "fixed-h segment. The carry policy is applied only after validation."
        ),
        "provenance": provenance,
        "rows": rows,
        "comparisons": comparisons,
        "timing": timing,
    }
    path = output_dir / "one_step_grid.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        "artifact": str(path),
        "sha256": digest,
        "all_status_match": all(row["status_match"] for row in comparisons),
        "all_decisions_match": all(row["decision_match"] for row in comparisons),
        "all_segment_coefficients_match": all(
            row["segment_coefficient_sha256_match"] for row in comparisons
        ),
        "all_endpoint_coefficients_match": all(
            row["endpoint_coefficient_sha256_match"] for row in comparisons
        ),
        "all_candidate_carries_preserve_endpoint_coefficients": all(
            row["candidate_next_initial_equals_raw_endpoint"]
            for row in comparisons
            if next(item for item in rows if item["mode"] == MODES[1] and item["attempted_h"] == row["h"])["status"]
            == "validated"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.output_dir.resolve())
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_status_match"] and summary["all_decisions_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
