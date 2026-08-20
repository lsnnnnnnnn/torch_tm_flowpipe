#!/usr/bin/env python3
"""Build the clean-SHA step-1 causal gate for VDP C2 refinement."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch

from torch_tm_flowpipe import (
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    Interval,
    PolynomialODE,
    flowpipe_step_flowstar_style_adaptive,
)
from torch_tm_flowpipe.batched_dense_tm import (
    FLOWSTAR_MAX_REFINEMENT_STEPS,
    FLOWSTAR_RAW_REMAINDER_REFINED_MODE,
    FLOWSTAR_REFINEMENT_REPLAY_LIMIT,
    FLOWSTAR_STOP_RATIO,
    DenseExecutionCounters,
    _dense_polynomial_sha256,
    _frozen_vdp_structural_fingerprint,
    dense_picard_validate_step,
    dense_polynomial_picard,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.post_accept_refinement_oracle import verify_refinement_iteration


ROOT = Path(__file__).resolve().parents[1]
FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
C1 = "flowstar_raw_remainder_compat_factorized_joint_closure"
C2 = FLOWSTAR_RAW_REMAINDER_REFINED_MODE
EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED_FLOWSTAR_BLOBS = {
    "flowstar-toolbox/Continuous.cpp": "9cba9bb6fe072679c691a866bba7834c44bb6602",
    "flowstar-toolbox/TaylorModel.h": "401c759dea43c359523eec808d308a2733f8ed67",
    "flowstar-toolbox/expression.h": "f6f049f4c6ce056de2b7d6db5d13620172667a11",
    "flowstar-toolbox/Interval.cpp": "ef6dbe4a241e43e6254e8243a2e1c411ddffb9b8",
    "flowstar-toolbox/include.h": "c238d58efe1650fd7fcd53eb94bceb4381f19f97",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


def _state_and_base():
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("11/10", "7/5"), ("47/20", "49/20")], 4
    )
    policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
    )
    base = sparse_tmvector_to_dense(
        state.normalized_initial_tm(4).extend_domain(Interval(0.0, 0.01)),
        order=4,
        device="cpu",
        dtype=torch.float64,
        counters=DenseExecutionCounters(),
        segment_boundary=True,
        range_policy=policy,
        range_trace=[],
    )
    return state, base, policy


def _full_step(state: FlowstarNormalFlowpipeState, policy: DenseRangePolicy, mode: str):
    return flowpipe_step_flowstar_style_adaptive(
        _vdp(),
        state.normalized_initial_tm(4),
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1.0e-4,
        cutoff_threshold=1.0e-10,
        max_validation_attempts=2,
        validation_eps=1.0e-12,
        validation_mode=mode,
        reset_mode="normalized_insertion_dependency_preserving",
        step_policy_mode="flowstar_compat",
        flowstar_normal_state=state,
        tm_backend="dense",
        dense_device="cpu",
        dense_dtype=torch.float64,
        dense_range_policy=policy,
    )


def _box(box) -> list[dict[str, Any]]:
    return [
        {
            "lo": float(interval.lo),
            "hi": float(interval.hi),
            "width": float(interval.width()),
            "lo_hex": float(interval.lo).hex(),
            "hi_hex": float(interval.hi).hex(),
        }
        for interval in box
    ]


def _source_contract(flowstar_repo: Path) -> dict[str, Any]:
    if _git("rev-parse", FLOWSTAR_SHA, cwd=flowstar_repo) != FLOWSTAR_SHA:
        raise ValueError("pinned Flow* commit is unavailable")
    snippets = {
        "flowstar-toolbox/Continuous.cpp": (960, 1042),
        "flowstar-toolbox/TaylorModel.h": (3728, 3744),
        "flowstar-toolbox/expression.h": (1833, 1897),
        "flowstar-toolbox/Interval.cpp": (2982, 2998),
        "flowstar-toolbox/include.h": (36, 50),
    }
    files = {}
    for path, expected_blob in EXPECTED_FLOWSTAR_BLOBS.items():
        blob = _git("rev-parse", f"{FLOWSTAR_SHA}:{path}", cwd=flowstar_repo)
        if blob != expected_blob:
            raise ValueError(f"Flow* blob mismatch: {path}")
        text = _git("show", f"{FLOWSTAR_SHA}:{path}", cwd=flowstar_repo)
        start, end = snippets[path]
        lines = text.splitlines()
        files[path] = {
            "git_blob": blob,
            "line_start": start,
            "line_end": end,
            "text": "\n".join(
                f"{index}: {lines[index - 1]}" for index in range(start, end + 1)
            ),
        }
    include = files["flowstar-toolbox/include.h"]["text"]
    continuous = files["flowstar-toolbox/Continuous.cpp"]["text"]
    interval = files["flowstar-toolbox/Interval.cpp"]["text"]
    for required in (
        "#define MAX_REFINEMENT_STEPS\t490",
        "#define STOP_RATIO\t\t\t\t0.99",
    ):
        if required not in include:
            raise ValueError(f"Flow* constant not proven: {required}")
    if "rSteps <= MAX_REFINEMENT_STEPS" not in continuous:
        raise ValueError("Flow* inclusive loop bound not proven")
    if "mpfr_div(ratio, width2, width1, MPFR_RNDU)" not in interval:
        raise ValueError("Flow* widthRatio direction not proven")
    return {
        "schema": "flowstar_pinned_post_accept_refinement_contract_v1",
        "commit": FLOWSTAR_SHA,
        "files": files,
        "max_refinement_steps_macro": 490,
        "inclusive_zero_based_replay_limit": 491,
        "stop_ratio": 0.99,
        "width_ratio_direction": "new_width_divided_by_old_width",
        "zero_width_semantics": "0/0 is MPFR NaN; comparison to STOP_RATIO is false",
        "first_self_map_failure": "returns failure before refinement",
        "first_success_replacement": "complete vector replaced by first self-map image",
        "component_update": "stock source is sequential and can retain an earlier-component partial update",
        "stock_refinement_failure_retention": "earlier updated components plus failing/later old components",
        "torch_c2_update": "stronger atomic complete-vector commit; complete old vector retained on any failure",
        "intermediate_ranges": "fixed list from first Picard_ctrunc_normal; polynomial ranges are remainder-independent",
    }


def run(output_dir: Path, flowstar_repo: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if status or hashlib.sha256(diff).hexdigest() != EMPTY_DIFF_SHA256:
        raise ValueError("step-1 scientific gate requires a clean detached worktree")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if symbolic.returncode == 0 or symbolic.stdout.strip():
        raise ValueError("step-1 scientific gate requires detached HEAD")

    source_contract = _source_contract(flowstar_repo.resolve())
    _write_json(output_dir / "flowstar_pinned_contract.json", source_contract)
    state, base, policy = _state_and_base()
    common = {
        "h": 0.01,
        "order": 4,
        "tau_index": 2,
        "target_remainder_radius": 1.0e-4,
        "cutoff_threshold": 1.0e-10,
        "max_validation_attempts": 2,
        "validation_eps": 1.0e-12,
    }
    candidate, _ = dense_polynomial_picard(
        _vdp(),
        base.without_remainder(),
        tau_index=2,
        order=4,
        iterations=4,
        cutoff_threshold=1.0e-10,
    )
    c1 = dense_picard_validate_step(_vdp(), base, validation_mode=C1, **common)
    c2 = dense_picard_validate_step(_vdp(), base, validation_mode=C2, **common)
    if c1.status != "validated" or c2.status != "validated":
        raise ValueError("step-1 C1/C2 self-map did not validate")
    c1_first = next(row for row in c1.trace if row.get("phase") == "remainder_validation")
    c2_first = next(row for row in c2.trace if row.get("phase") == "remainder_validation")
    refinement = [row for row in c2.trace if row.get("phase") == "post_accept_refinement"]
    _write_jsonl(output_dir / "refinement_ledger.jsonl", refinement)

    candidate_hex = [
        [float(value).hex() for value in candidate.poly.coeffs[0, component].tolist()]
        for component in range(2)
    ]
    domain = [
        [float(base.domain_lo[0, index]), float(base.domain_hi[0, index])]
        for index in range(base.n_vars)
    ]
    base_remainder = [
        [float(base.rem_lo[0, component]), float(base.rem_hi[0, component])]
        for component in range(2)
    ]
    oracle_rows = []
    for row in refinement:
        if not row["committed"]:
            continue
        certificate = verify_refinement_iteration(
            row,
            candidate_coefficient_hex=candidate_hex,
            candidate_exponents=candidate.poly.basis.exponents.tolist(),
            domain=domain,
            base_remainder=base_remainder,
            tau_interval=[0.0, 0.01],
            validation_eps=1.0e-12,
        )
        oracle_rows.append(certificate.to_json())
    _write_json(
        output_dir / "exact_fraction_bernstein_oracle.json",
        {
            "schema": "vdp_c2_all_committed_refinements_oracle_v1",
            "production_oracle_implementation_shared": False,
            "all_contained": bool(oracle_rows) and all(row["all_contained"] for row in oracle_rows),
            "iterations": oracle_rows,
        },
    )

    full_c1 = _full_step(state, policy, C1)
    full_c2 = _full_step(state, policy, C2)
    if full_c1.status != "validated" or full_c2.status != "validated":
        raise ValueError("full step-1 publication path failed")
    c1_segment = _box(full_c1.tm.range_box())
    c2_segment = _box(full_c2.tm.range_box())
    c1_endpoint = _box(full_c1.endpoint_raw_tm.range_box())
    c2_endpoint = _box(full_c2.endpoint_raw_tm.range_box())
    channels = {
        "segment_x": (c1_segment[0]["width"], c2_segment[0]["width"]),
        "segment_y": (c1_segment[1]["width"], c2_segment[1]["width"]),
        "endpoint_x": (c1_endpoint[0]["width"], c2_endpoint[0]["width"]),
        "endpoint_y": (c1_endpoint[1]["width"], c2_endpoint[1]["width"]),
    }
    identical_fields = (
        "validation_status",
        "finite",
        "subset_result",
        "target_subset_result",
        "candidate_remainder_lo",
        "candidate_remainder_hi",
        "picard_image_remainder_lo",
        "picard_image_remainder_hi",
        "subset_margin",
        "raw_rhs_remainder_lo",
        "raw_rhs_remainder_hi",
        "poly_diff_range_lo",
        "poly_diff_range_hi",
    )
    first_acceptance_identical = all(c1_first[field] == c2_first[field] for field in identical_fields)
    c1_x_width = (
        c1_first["picard_image_remainder_hi"][0][0]
        - c1_first["picard_image_remainder_lo"][0][0]
    )
    c2_x_width = float(c2.validated_remainder_hi[0, 0] - c2.validated_remainder_lo[0, 0])
    c1_y_width = (
        c1_first["picard_image_remainder_hi"][0][1]
        - c1_first["picard_image_remainder_lo"][0][1]
    )
    c2_y_width = float(c2.validated_remainder_hi[0, 1] - c2.validated_remainder_lo[0, 1])
    flowstar_crosscheck = json.loads(
        (
            ROOT
            / "evidence/vdp_live_loss_ablation_b3_b4_closure/20260819T073038Z/01_gates/flowstar_runtime_crosscheck.json"
        ).read_text(encoding="utf-8")
    )
    flowstar_x_width = float(flowstar_crosscheck["raw_image_target_remainder_first_iteration"][0]["width"])
    gap_removed = (c1_x_width - c2_x_width) / (c1_x_width - flowstar_x_width)
    all_oracle = bool(oracle_rows) and all(row["all_contained"] for row in oracle_rows)
    all_channels_no_wider = all(c2_width <= c1_width for c1_width, c2_width in channels.values())
    gate_pass = bool(
        first_acceptance_identical
        and torch.equal(c1.segment_tm.poly.coeffs, c2.segment_tm.poly.coeffs)
        and refinement
        and all(row["committed"] for row in refinement)
        and all_oracle
        and gap_removed >= 0.5
        and c2_y_width <= c1_y_width
        and all_channels_no_wider
    )
    gate = {
        "schema": "vdp_c2_step1_causal_gate_v1",
        "scientific_sha": head,
        "worktree_clean": True,
        "detached_head": True,
        "frozen_contract": {
            "ode": "x'=y; y'=y-x-x^2*y",
            "initial_box_exact": [["11/10", "7/5"], ["47/20", "49/20"]],
            "order": 4,
            "h": 0.01,
            "target_remainder_radius": 1.0e-4,
            "cutoff": 1.0e-10,
            "validation_eps": 1.0e-12,
            "reset_mode": "normalized_insertion_dependency_preserving",
        },
        "vdp_structural_fingerprint": _frozen_vdp_structural_fingerprint(_vdp()),
        "c1_mode": C1,
        "c2_mode": C2,
        "candidate_polynomial_sha256": _dense_polynomial_sha256(candidate),
        "first_acceptance_decision_identical": first_acceptance_identical,
        "candidate_polynomial_bitwise_identical": bool(
            torch.equal(c1.segment_tm.poly.coeffs, c2.segment_tm.poly.coeffs)
        ),
        "committed_refinement_count": sum(bool(row["committed"]) for row in refinement),
        "refinement_stop_reason": refinement[-1]["stop_reason"],
        "all_committed_exact_oracle_contained": all_oracle,
        "c1_x_raw_image_width": c1_x_width,
        "c2_x_final_raw_image_width": c2_x_width,
        "flowstar_x_raw_image_width_runtime_crosscheck_only": flowstar_x_width,
        "c1_vs_flowstar_x_gap_fraction_removed": gap_removed,
        "c1_y_raw_image_width": c1_y_width,
        "c2_y_final_raw_image_width": c2_y_width,
        "y_raw_image_no_regression": c2_y_width <= c1_y_width,
        "published_channels": {
            name: {"c1_width": values[0], "c2_width": values[1], "c2_no_wider": values[1] <= values[0]}
            for name, values in channels.items()
        },
        "all_published_channels_no_wider": all_channels_no_wider,
        "endpoint_repair_used": False,
        "endpoint_tightening_used": False,
        "sampling_used_for_containment": False,
        "metadata_used_for_numeric_decision": False,
        "final_remainder_ledger_iteration": refinement[-1]["refinement_iteration"],
        "final_remainder_ledger_matches_last_commit": (
            refinement[-1]["validated_remainder_ledger_intervals"]
            == c2.validated_remainder_decomposition.ledger.intervals()
        ),
        "flowstar_constants": {
            "max_refinement_steps_macro": FLOWSTAR_MAX_REFINEMENT_STEPS,
            "replay_limit": FLOWSTAR_REFINEMENT_REPLAY_LIMIT,
            "stop_ratio": FLOWSTAR_STOP_RATIO,
        },
        "publication_phase_contract": {
            "first_validation": "C1 image R1 establishes step acceptance",
            "post_accept_refinement": "only atomic subset proposals replace R1",
            "published_segment_endpoint": "last committed remainder with unchanged candidate polynomial",
            "next_step_reset_carry": "existing dependency-preserving normal insertion after publication",
        },
        "gate_pass": gate_pass,
        "failure_code": "" if gate_pass else "POST_ACCEPT_REFINEMENT_CAUSAL_GATE_FAILED",
    }
    _write_json(output_dir / "gate_a.json", gate)
    if not gate_pass:
        raise ValueError("POST_ACCEPT_REFINEMENT_CAUSAL_GATE_FAILED")
    return gate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flowstar-repo", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.output_dir, args.flowstar_repo), sort_keys=True))
