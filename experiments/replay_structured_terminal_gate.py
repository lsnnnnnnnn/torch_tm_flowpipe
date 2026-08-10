#!/usr/bin/env python3
"""Apply S1 locally to a frozen terminal ledger and enforce the prefix-history gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from torch_tm_flowpipe.structured_remainder import (
    ELIGIBLE_STRUCTURED_SOURCES,
    STRUCTURED_REMAINDER_CANDIDATE,
    STRUCTURED_REMAINDER_CAPACITY,
    initialize_structured_remainder_state,
    structured_remainder_boundary_update,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET_RADIUS = 1.0e-4
BASELINE_Y_MARGIN_MAGNITUDE = 1.99995911680722e-5
GO_Y_MARGIN = -1.599967293445776e-5


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _last_jsonl(path: Path) -> Mapping[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty ledger: {path}")
    return rows[-1]


def _interval(entry: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(entry["lo"], dtype=torch.float64),
        torch.tensor(entry["hi"], dtype=torch.float64),
    )


def _margin(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return torch.minimum(lo + TARGET_RADIUS, TARGET_RADIUS - hi)


def _local_update(
    sources: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    validated: tuple[torch.Tensor, torch.Tensor],
):
    batch, state_dim = validated[0].shape
    state = initialize_structured_remainder_state(batch, state_dim)
    identity = torch.eye(state_dim, dtype=torch.float64).expand(batch, -1, -1).clone()
    zero = torch.zeros_like(validated[0])
    return structured_remainder_boundary_update(
        state,
        typed_sources=sources,
        validated_remainder_lo=validated[0],
        validated_remainder_hi=validated[1],
        linear_map_lo=identity,
        linear_map_hi=identity,
        nonlinear_residual_lo=zero,
        nonlinear_residual_hi=zero,
        normalization_scale=torch.ones_like(zero),
        boundary_index=307,
        map_is_affine=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--early-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = json.loads((args.baseline_dir / "summary.json").read_text(encoding="utf-8"))
    ledger = _last_jsonl(args.baseline_dir / "remainder_ledger.jsonl")
    entries = ledger.get("validated_remainder_ledger_intervals")
    if not isinstance(entries, Mapping):
        raise ValueError("baseline replay lacks additive validated remainder ledger")
    typed_sources = {str(name): _interval(entry) for name, entry in entries.items()}
    validated = (
        torch.tensor(ledger["picard_image_remainder_lo"], dtype=torch.float64),
        torch.tensor(ledger["picard_image_remainder_hi"], dtype=torch.float64),
    )
    started = time.perf_counter()
    result = _local_update(typed_sources, validated)
    runtime_s = time.perf_counter() - started
    ordinary_margin = _margin(result.state.ordinary_rem_lo, result.state.ordinary_rem_hi)
    baseline_margin = torch.tensor(ledger["subset_margin"], dtype=torch.float64)
    x_regression_ok = bool(ordinary_margin[0, 0] >= 0.95 * baseline_margin[0, 0])
    y_improvement_ok = bool(ordinary_margin[0, 1] >= GO_Y_MARGIN)
    local_decision_closes = bool(torch.all(ordinary_margin >= 0))

    # A local empty-state split is useful attribution, but the frozen checkpoint
    # predates S1 and therefore cannot establish conservation across its 307-step prefix.
    prefix_structured_state_available = False
    early = json.loads(args.early_checkpoint.read_text(encoding="utf-8"))
    early_exact_typed_validated_ledger_available = bool(
        early.get("validated_remainder_ledger_intervals")
    )
    analytic_gates_passed = True
    complete_ab = prefix_structured_state_available and early_exact_typed_validated_ledger_available
    go = bool(
        complete_ab
        and analytic_gates_passed
        and bool(torch.all(result.accepted))
        and x_regression_ok
        and (local_decision_closes or y_improvement_ok)
    )

    diagnostic_rows: dict[str, Any] = {}
    for name in ELIGIBLE_STRUCTURED_SOURCES:
        if name not in typed_sources:
            diagnostic_rows[name] = {"available": False}
            continue
        isolated = _local_update({name: typed_sources[name]}, typed_sources[name])
        diagnostic_rows[name] = {
            "available": True,
            "ordinary_rem_lo": isolated.state.ordinary_rem_lo.tolist(),
            "ordinary_rem_hi": isolated.state.ordinary_rem_hi.tolist(),
            "materialized_lo": isolated.materialized_lo.tolist(),
            "materialized_hi": isolated.materialized_hi.tolist(),
            "accepted": isolated.accepted.tolist(),
        }

    checkpoint_files = sorted(path for path in args.checkpoint_dir.iterdir() if path.is_file())
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    field_map = {
        "schema": "structured_remainder_field_map_v1",
        "candidate": STRUCTURED_REMAINDER_CANDIDATE,
        "capacity": STRUCTURED_REMAINDER_CAPACITY,
        "ordinary_remainder": "ordinary_rem_lo/hi [B,S], target checked",
        "new_J": "centered eligible source radii in j_lo/hi [B,K,S]",
        "accepted_Phi": "phi_lo/hi [B,K,S,S], identity on insertion then interval-linear propagation",
        "scale_inverse_scale": "normalization_scale input and inverse_scale retained in state",
        "old_J_propagation": "Phi <- linear_map @ Phi with outward sequential reductions",
        "nonlinear_interaction": "structured_nonlinear_residual is mandatory for non-affine maps and materialized once into ordinary",
        "insertion": "eligible source center enters ordinary once; symmetric radius enters one J slot",
        "eviction": "oldest active age, then lowest slot; materialize outward into ordinary before overwrite",
        "validation": "ordinary remainder only is target checked; full materialization is conservation/output checked",
        "endpoint_tube": "ordinary plus sum(Phi@J) must be included in both outputs",
        "eligible_sources": list(ELIGIBLE_STRUCTURED_SOURCES),
        "source_id": {"polynomial_truncation": 1, "integration_overflow": 2},
        "no_double_count": "eligible radius absent from ordinary; its unique center remains ordinary",
    }
    _write_json(args.output_dir / "field_map.json", field_map)
    artifact = {
        "schema": "structured_terminal_ab_v1",
        "source_sha": source_sha,
        "candidate": STRUCTURED_REMAINDER_CANDIDATE,
        "terminal": {
            "t": baseline_summary["t_before"],
            "h": baseline_summary["attempted_h"],
            "checkpoint_hashes": {path.name: _sha256(path) for path in checkpoint_files},
            "checkpoint_full_sha256": baseline_summary["checkpoint_full_sha256"],
            "contract_sha256": baseline_summary["contract_sha256"],
            "candidate_hashes": baseline_summary["candidate_hashes"],
            "baseline_accepted": baseline_summary["accepted"],
            "baseline_status": baseline_summary["status"],
            "baseline_ordinary_lo": validated[0].tolist(),
            "baseline_ordinary_hi": validated[1].tolist(),
            "baseline_margin": baseline_margin.tolist(),
            "baseline_y_margin_magnitude_contract": BASELINE_Y_MARGIN_MAGNITUDE,
            "typed_sources": {name: {"lo": lo.tolist(), "hi": hi.tolist()} for name, (lo, hi) in typed_sources.items()},
            "validated_decomposition_contains_image": ledger["validated_remainder_decomposition_contains_image"],
            "local_empty_state_s1": {
                "accepted": result.accepted.tolist(),
                "ordinary_rem_lo": result.state.ordinary_rem_lo.tolist(),
                "ordinary_rem_hi": result.state.ordinary_rem_hi.tolist(),
                "ordinary_target_margin": ordinary_margin.tolist(),
                "active": result.state.active.tolist(),
                "age": result.state.age.tolist(),
                "source_id": result.state.source_id.tolist(),
                "j_lo": result.state.j_lo.tolist(),
                "j_hi": result.state.j_hi.tolist(),
                "phi_lo": result.state.phi_lo.tolist(),
                "phi_hi": result.state.phi_hi.tolist(),
                "propagated_symbolic_lo": result.propagated_symbolic_lo.tolist(),
                "propagated_symbolic_hi": result.propagated_symbolic_hi.tolist(),
                "new_symbolic_lo": result.new_symbolic_lo.tolist(),
                "new_symbolic_hi": result.new_symbolic_hi.tolist(),
                "nonlinear_residual_lo": result.nonlinear_residual_lo.tolist(),
                "nonlinear_residual_hi": result.nonlinear_residual_hi.tolist(),
                "evicted_materialized_lo": result.evicted_materialized_lo.tolist(),
                "evicted_materialized_hi": result.evicted_materialized_hi.tolist(),
                "decomposition_padding_lo": result.decomposition_padding_lo.tolist(),
                "decomposition_padding_hi": result.decomposition_padding_hi.tolist(),
                "materialized_lo": result.materialized_lo.tolist(),
                "materialized_hi": result.materialized_hi.tolist(),
                "conservation_mask": result.conservation_mask.tolist(),
                "source_decomposition_mask": result.source_decomposition_mask.tolist(),
                "local_decision_closes": local_decision_closes,
                "x_margin_regression_within_5_percent": x_regression_ok,
                "y_margin_at_least_go_threshold": y_improvement_ok,
                "runtime_s": runtime_s,
                "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
        },
        "early_native_split": {
            "artifact": args.early_checkpoint.as_posix(),
            "sha256": _sha256(args.early_checkpoint),
            "t_pre": early["t_pre"],
            "h": early["h_attempt"],
            "baseline_production_accepted": early["production_accepted"],
            "retained_polynomial_sha256": early["production_picard_coefficient_sha256"],
            "prefix_structured_state_available": False,
            "exact_typed_validated_ledger_available": early_exact_typed_validated_ledger_available,
            "s1_same_pre_state_executed": False,
            "reason": "the immutable pre-S1 observation artifact has no additive validated source ledger or S1 prefix state",
        },
        "analytic_gates": {
            "passed": analytic_gates_passed,
            "basis": "focused affine, harmonic, scalar quadratic, cross-correlated, zero, conservation, eviction, determinism tests",
        },
        "bounded_failure_diagnostics": {
            "materialize_all_columns_before_terminal_picard": {
                "lo": result.materialized_lo.tolist(),
                "hi": result.materialized_hi.tolist(),
            },
            "identity_phi": True,
            "source_isolation": diagnostic_rows,
            "nonlinear_residual": {"lo": result.nonlinear_residual_lo.tolist(), "hi": result.nonlinear_residual_hi.tolist()},
            "eviction": {"lo": result.evicted_materialized_lo.tolist(), "hi": result.evicted_materialized_hi.tolist()},
        },
        "prefix_structured_state_available": prefix_structured_state_available,
        "complete_same_pre_state_ab": complete_ab,
        "go": go,
        "stop_outcome": None if go else "STRUCTURED_REMAINDER_LOCAL_GATE_FAILED",
        "fresh_horizon_ladder_authorized": go,
        "interpretation": (
            "The local empty-state split is conservative attribution only. It cannot be promoted because the immutable checkpoint lacks the structured history needed to prove prefix conservation/no-double-count."
        ),
    }
    _write_json(args.output_dir / "structured_terminal_ab.json", artifact)
    print(json.dumps(artifact, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
