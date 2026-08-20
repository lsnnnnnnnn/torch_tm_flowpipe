#!/usr/bin/env python3
"""Replay the C2 map diagnostically on the real native terminal prestate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

from torch_tm_flowpipe import DenseRangePolicy, Interval, PolynomialODE, load_terminal_checkpoint
from torch_tm_flowpipe.batched_dense_tm import (
    FLOWSTAR_RAW_REMAINDER_REFINED_MODE,
    DenseExecutionCounters,
    _dense_flowstar_raw_compat_image,
    _frozen_vdp_structural_fingerprint,
    _post_accept_refine_raw_remainder,
    _subset_margin,
    dense_polynomial_picard,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.post_accept_refinement_oracle import verify_refinement_iteration


ROOT = Path(__file__).resolve().parents[1]
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
C2 = FLOWSTAR_RAW_REMAINDER_REFINED_MODE


def _vdp() -> PolynomialODE:
    return PolynomialODE.from_system_spec(
        {
            "state_names": ["x", "y"],
            "rhs": [
                {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                {
                    "terms": [
                        {"coefficient": 1.0, "powers": [0, 1]},
                        {"coefficient": -1.0, "powers": [1, 0]},
                        {"coefficient": -1.0, "powers": [2, 1]},
                    ]
                },
            ],
        }
    )


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _scheduler_minimum_exhausted(
    summary: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    *,
    attempted_h: float,
) -> bool:
    if summary.get("failure_type") != "minimum_step_reached":
        return False
    checkpoint_h = float(scheduler["h_attempted"])
    if checkpoint_h != attempted_h:
        raise ValueError("terminal checkpoint/reference attempted h mismatch")
    if int(scheduler["terminal_internal_step_rejections"]) < 1:
        return False
    h_min = float(summary["h_min"])
    next_retry_h = float(scheduler["next_retry_h"])
    return attempted_h + 1.0e-15 >= h_min and next_retry_h < h_min - 1.0e-15


def _candidate_hex(candidate) -> list[list[str]]:
    return [
        [float(value).hex() for value in candidate.poly.coeffs[0, component].tolist()]
        for component in range(candidate.poly.out_dim)
    ]


def _oracle(
    row: Mapping[str, Any],
    *,
    candidate,
    base,
    h: float,
) -> dict[str, Any]:
    return verify_refinement_iteration(
        row,
        candidate_coefficient_hex=_candidate_hex(candidate),
        candidate_exponents=candidate.poly.basis.exponents.tolist(),
        domain=[
            [float(base.domain_lo[0, index]), float(base.domain_hi[0, index])]
            for index in range(base.n_vars)
        ],
        base_remainder=[
            [float(base.rem_lo[0, component]), float(base.rem_hi[0, component])]
            for component in range(base.poly.out_dim)
        ],
        tau_interval=[0.0, h],
        validation_eps=1.0e-12,
    ).to_json()


def diagnose(run_dir: Path, output: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    summary = _load(run_dir / "summary.json")
    reference = _load(run_dir / "terminal_checkpoint/terminal_reference.json")
    checkpoint = load_terminal_checkpoint(
        run_dir / "terminal_checkpoint",
        expected_order=4,
        expected_dtype="float64",
    )
    # The serialized checkpoint contract is authoritative; the explicit load
    # above can vary as runner command schemas evolve, so validate the fields
    # used by this replay directly as well.
    if checkpoint.contract["validation_mode"] != C2:
        raise ValueError("terminal checkpoint is not the C2 production lane")
    if checkpoint.contract["reset_mode"] != summary["reset_mode"]:
        raise ValueError("terminal checkpoint/summary reset mode mismatch")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if status or hashlib.sha256(diff).hexdigest() != EMPTY_DIFF_SHA256:
        raise ValueError("terminal diagnostic requires a clean worktree")
    if summary["commit"] != head or summary["worktree_dirty"] is not False:
        raise ValueError("terminal run/scientific checkout provenance mismatch")

    h = float(reference["attempted_h"])
    scheduler_minimum_exhausted = _scheduler_minimum_exhausted(
        summary,
        checkpoint.scheduler,
        attempted_h=h,
    )
    base = sparse_tmvector_to_dense(
        checkpoint.current.extend_domain(Interval(0.0, h)),
        order=4,
        device="cpu",
        dtype=torch.float64,
        counters=DenseExecutionCounters(),
        segment_boundary=True,
        range_policy=DenseRangePolicy(
            method="adaptive_subdivision",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
            trigger="proactive_depth1_on_named_contexts",
            named_contexts=("polynomial_truncation",),
        ),
        range_trace=[],
    )
    candidate, _ = dense_polynomial_picard(
        _vdp(),
        base.without_remainder(),
        tau_index=checkpoint.current.n_vars,
        order=4,
        iterations=4,
        cutoff_threshold=1.0e-10,
    )
    target_lo = torch.full_like(candidate.rem_lo, -1.0e-4)
    target_hi = torch.full_like(candidate.rem_hi, 1.0e-4)
    target_model = candidate.with_remainder(
        target_lo, target_hi, category="initial_remainder"
    )
    image_lo, image_hi, compat, decomposition = _dense_flowstar_raw_compat_image(
        _vdp(),
        base,
        target_model,
        candidate,
        tau_index=checkpoint.current.n_vars,
        order=4,
        cutoff_threshold=1.0e-10,
        validation_eps=1.0e-12,
        raw_rhs_evaluation="canonical_factorized_joint_closure",
        raw_dependency_preserving_square=True,
    )
    first_margins = _subset_margin(target_lo, target_hi, image_lo, image_hi)
    first_subset = bool(torch.all(first_margins >= 0))
    first_row = {
        "phase": "post_accept_refinement",
        "refinement_iteration": 1,
        "input_remainder_lo": target_lo.detach().cpu().tolist(),
        "input_remainder_hi": target_hi.detach().cpu().tolist(),
        "proposed_remainder_lo": image_lo.detach().cpu().tolist(),
        "proposed_remainder_hi": image_hi.detach().cpu().tolist(),
        **compat,
    }
    first_oracle = _oracle(first_row, candidate=candidate, base=base, h=h)

    # This continuation is diagnostic only.  Production cannot commit I1
    # because I1 is not a subset of R0; starting from I1 answers whether the
    # same sound map would contract if an additional replay were hypothetically
    # permitted.
    _, _, _, theoretical_rows = _post_accept_refine_raw_remainder(
        _vdp(),
        base,
        candidate,
        retained_lo=image_lo,
        retained_hi=image_hi,
        retained_decomposition=decomposition,
        tau_index=checkpoint.current.n_vars,
        order=4,
        cutoff_threshold=1.0e-10,
        validation_eps=1.0e-12,
        structural_fingerprint=_frozen_vdp_structural_fingerprint(_vdp()),
    )
    theoretical_oracles = [
        _oracle(row, candidate=candidate, base=base, h=h)
        for row in theoretical_rows
        if row.get("committed")
    ]
    flat_margins = first_margins.detach().cpu().reshape(-1).tolist()
    limiting_component = min(range(len(flat_margins)), key=flat_margins.__getitem__)
    lower_margin = float(image_lo[0, limiting_component] - target_lo[0, limiting_component])
    upper_margin = float(target_hi[0, limiting_component] - image_hi[0, limiting_component])
    limiting_side = "lower" if lower_margin <= upper_margin else "upper"
    ledger_widths = {
        category: float((hi - lo)[0, limiting_component])
        for category, (lo, hi) in decomposition.ledger.entries.items()
    }
    largest_owner = max(ledger_widths, key=ledger_widths.__getitem__)
    first_theoretical = theoretical_rows[0] if theoretical_rows else None
    could_contract = bool(
        first_theoretical
        and first_theoretical.get("committed")
        and any(
            component["output_interval"][1] - component["output_interval"][0]
            < component["input_interval"][1] - component["input_interval"][0]
            for component in first_theoretical["components"]
        )
    )
    result = {
        "schema": "vdp_c2_native_terminal_diagnostic_v1",
        "scientific_sha": head,
        "run_completed_horizon": summary["completed_horizon"],
        "requested_horizon": summary["requested_horizon"],
        "t_before": reference["t_before"],
        "h_attempted": h,
        "h_min": summary["h_min"],
        "scheduler_next_retry_h": checkpoint.scheduler["next_retry_h"],
        "scheduler_terminal_internal_step_rejections": checkpoint.scheduler[
            "terminal_internal_step_rejections"
        ],
        "scheduler_at_h_min": scheduler_minimum_exhausted,
        "production_first_self_map_subset": first_subset,
        "production_stop_reason": "first_self_map_subset_failure" if not first_subset else "unexpected_acceptance",
        "production_refinement_committed": False,
        "first_image": {
            "input_lo": target_lo.detach().cpu().tolist(),
            "input_hi": target_hi.detach().cpu().tolist(),
            "output_lo": image_lo.detach().cpu().tolist(),
            "output_hi": image_hi.detach().cpu().tolist(),
            "subset_margins": first_margins.detach().cpu().tolist(),
            "exact_oracle": first_oracle,
        },
        "limiting_component": ("x", "y")[limiting_component],
        "limiting_side": limiting_side,
        "subset_margin": flat_margins[limiting_component],
        "theoretical_sound_continuation_from_uncommitted_I1": list(theoretical_rows),
        "theoretical_continuation_oracles": theoretical_oracles,
        "another_sound_replay_would_contract": could_contract,
        "theoretical_stop_reason": (
            theoretical_rows[-1]["stop_reason"] if theoretical_rows else "not_evaluated"
        ),
        "failure_classification": (
            "scheduler_h_min_after_first_self_map_subset_failure"
            if not first_subset and scheduler_minimum_exhausted
            else "first_self_map_subset_failure"
        ),
        "largest_additive_ledger_owner": largest_owner,
        "additive_ledger_widths": ledger_widths,
        "ownership_warning": "largest additive category is ownership only, not causal ranking",
        "production_state_mutated_by_diagnostic": False,
    }
    _write(output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(diagnose(args.run_dir, args.output), sort_keys=True))
