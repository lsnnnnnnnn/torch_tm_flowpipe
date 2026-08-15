#!/usr/bin/env python3
"""Bitwise fixed-prefix fresh/resume and rejected-retry audit for G2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    PolynomialODE,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.flowpipe import flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.g2_shared_column import G2_SHARED_COLUMN_CANDIDATE

sys.path.insert(0, str(ROOT / "experiments"))
from run_vdp_dense_backend import load_contract


def policy() -> DenseRangePolicy:
    return DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
    )


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def step(ode: PolynomialODE, current: Any, normal: Any, h: float = 0.01) -> Any:
    result = flowpipe_step_flowstar_style_adaptive(
        ode,
        current,
        h=h,
        h_min=h,
        h_max=h,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode=G2_SHARED_COLUMN_CANDIDATE,
        step_policy_mode="flowstar_compat",
        flowstar_normal_state=normal,
        tm_backend="dense",
        dense_range_policy=policy(),
    )
    if result.status != "validated" or result.reset_tm is None or result.flowstar_normal_state is None:
        raise RuntimeError(f"fixed resume audit rejected: {result.message}")
    return result


def initial() -> tuple[Any, Any]:
    normal = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("1.1", "1.4"), ("2.35", "2.45")], 4
    ).with_g2_shared_columns(4)
    return normal.normalized_initial_tm(4), normal


def advance(ode: PolynomialODE, current: Any, normal: Any, count: int) -> tuple[Any, Any, list[str]]:
    hashes = []
    for _ in range(count):
        result = step(ode, current, normal)
        current = result.reset_tm
        normal = result.flowstar_normal_state
        hashes.append(digest({
            "g2": normal.g2_shared_column_state.as_dict(),
            "diagnostics": normal.diagnostics,
            "range": [
                [float(interval.lo.detach().cpu()).hex(), float(interval.hi.detach().cpu()).hex()]
                for interval in current.range_box()
            ],
        }))
    return current, normal, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    ode = PolynomialODE.from_system_spec(load_contract()["canonical_system_spec"])
    contract = {"candidate": G2_SHARED_COLUMN_CANDIDATE, "order": 4, "h": "0.01"}
    provenance = {"audit": "g2_fresh_resume_bitwise"}

    fresh_current, fresh_normal = initial()
    fresh_current, fresh_normal, fresh_hashes = advance(ode, fresh_current, fresh_normal, 20)

    prefix_current, prefix_normal = initial()
    prefix_current, prefix_normal, _ = advance(ode, prefix_current, prefix_normal, 10)
    first_manifest = save_terminal_checkpoint(
        output / "checkpoint",
        current=prefix_current,
        normal_state=prefix_normal,
        scheduler={"accepted_steps": 10, "current_time": 0.1, "h_next": 0.01},
        contract=contract,
        provenance=provenance,
    )
    loaded = load_terminal_checkpoint(
        output / "checkpoint",
        expected_contract=contract,
        expected_order=4,
        expected_dtype="float64",
    )
    second_manifest = save_terminal_checkpoint(
        output / "checkpoint_resaved",
        current=loaded.current,
        normal_state=loaded.normal_state,
        scheduler=loaded.scheduler,
        contract=loaded.contract,
        provenance=loaded.provenance,
    )
    checkpoint_bytes_equal = all(
        (output / "checkpoint" / name).read_bytes()
        == (output / "checkpoint_resaved" / name).read_bytes()
        for name in ("terminal_state.json", "terminal_state_manifest.json")
    )
    resumed_current, resumed_normal, resumed_hashes = advance(
        ode, loaded.current, loaded.normal_state, 10
    )
    end_fresh = save_terminal_checkpoint(
        output / "fresh_end",
        current=fresh_current,
        normal_state=fresh_normal,
        scheduler={"accepted_steps": 20, "current_time": 0.2, "h_next": 0.01},
        contract=contract,
        provenance=provenance,
    )
    end_resume = save_terminal_checkpoint(
        output / "resume_end",
        current=resumed_current,
        normal_state=resumed_normal,
        scheduler={"accepted_steps": 20, "current_time": 0.2, "h_next": 0.01},
        contract=contract,
        provenance=provenance,
    )
    end_bytes_equal = all(
        (output / "fresh_end" / name).read_bytes()
        == (output / "resume_end" / name).read_bytes()
        for name in ("terminal_state.json", "terminal_state_manifest.json")
    )
    continuation_equal = fresh_hashes[10:] == resumed_hashes

    pre_fingerprint = resumed_normal.g2_shared_column_state.fingerprint
    pre_payload = resumed_normal.g2_shared_column_state.retained_payload_sha256
    rejected = flowpipe_step_flowstar_style_adaptive(
        ode,
        resumed_current,
        h=0.1,
        h_min=0.1,
        h_max=0.1,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode=G2_SHARED_COLUMN_CANDIDATE,
        flowstar_normal_state=resumed_normal,
        tm_backend="dense",
        dense_range_policy=policy(),
    )
    retry_immutable = bool(
        rejected.status == "failed"
        and resumed_normal.g2_shared_column_state.fingerprint == pre_fingerprint
        and resumed_normal.g2_shared_column_state.retained_payload_sha256 == pre_payload
    )
    passed = all((checkpoint_bytes_equal, end_bytes_equal, continuation_equal, retry_immutable))
    result = {
        "schema": "g2_checkpoint_resume_atomicity_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "checkpoint_schema": first_manifest["schema"],
        "checkpoint_full_sha256": first_manifest["full_checkpoint_sha256"],
        "resaved_full_sha256": second_manifest["full_checkpoint_sha256"],
        "checkpoint_bytes_equal": checkpoint_bytes_equal,
        "fresh_resume_continuation_hashes_equal": continuation_equal,
        "fresh_resume_end_checkpoint_bytes_equal": end_bytes_equal,
        "fresh_end_full_sha256": end_fresh["full_checkpoint_sha256"],
        "resume_end_full_sha256": end_resume["full_checkpoint_sha256"],
        "rejected_retry_status": rejected.status,
        "rejected_retry_fingerprint_and_payload_immutable": retry_immutable,
        "fixed_variable_count": resumed_normal.g2_shared_column_state.variable_count,
        "accepted_steps": 20,
    }
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
