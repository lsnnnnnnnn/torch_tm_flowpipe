#!/usr/bin/env python3
"""Reconstruct the published SR100 prefix and replay its terminal rejection once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (  # noqa: E402
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
    TMVector,
    accepted_boundary_sr_queue_sha256,
    flowpipe_step_flowstar_style_adaptive,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.ode_examples import brusselator_ode  # noqa: E402
from torch_tm_flowpipe.terminal_checkpoint import MANIFEST_NAME, PAYLOAD_NAME  # noqa: E402

from experiments.run_brusselator_second_system_torch import (  # noqa: E402
    CUTOFF,
    INITIAL_DECIMAL,
    ORDER,
    QUEUE_CAPACITY,
    REMAINDER_RADIUS,
    STEP,
    _box_fields,
    _git,
    _queue_fields,
    _sha256,
    _validation_passed,
    _write_json,
)


CONTRACT_PATH = ROOT / "benchmarks/brusselator_terminal_sr1000_contract.json"
PUBLISHED_SEGMENTS = (
    ROOT
    / "artifacts/runs/brusselator_generic_core_validation_20260827"
    / "raw/torch_generic_sr100/segments.csv"
)
EXPECTED_ACCEPTED_STEPS = 355
TERMINAL_STEP = 356
BOUND_HEX_FIELDS = tuple(
    f"{prefix}_{component}_{bound}_hex"
    for prefix in ("endpoint", "tube")
    for component in ("x", "y")
    for bound in ("lo", "hi")
)
MACHINE_CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
CHECKPOINT_CONTRACT = MACHINE_CONTRACT["terminal_replay"]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_expected() -> list[dict[str, str]]:
    with PUBLISHED_SEGMENTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    accepted = [row for row in rows if row.get("status") == "accepted"]
    rejected = [row for row in rows if row.get("status") == "rejected"]
    if (
        len(accepted) != EXPECTED_ACCEPTED_STEPS
        or len(rejected) != 1
        or int(rejected[0]["step"]) != TERMINAL_STEP
    ):
        raise RuntimeError("published SR100 trace does not have the frozen terminal shape")
    return accepted


def _policy() -> DenseRangePolicy:
    return DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )


def _step(current: TMVector, state: FlowstarNormalFlowpipeState, step: int) -> tuple[Any, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    segment = flowpipe_step_flowstar_style_adaptive(
        brusselator_ode,
        current,
        h=STEP,
        h_min=STEP,
        h_max=STEP,
        order=ORDER,
        target_remainder_radius=REMAINDER_RADIUS,
        cutoff_threshold=CUTOFF,
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode=GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
        flowstar_normal_state=state,
        flowstar_symbolic_queue_max_size=QUEUE_CAPACITY,
        right_map_range_mode="standard",
        right_map_center_mode="constant",
        tm_backend="dense",
        dense_device="cpu",
        dense_dtype=torch.float64,
        dense_range_policy=_policy(),
        diagnostics=diagnostics,
        diagnostics_context={
            "system": "brusselator",
            "lane": "torch_generic_sr100_terminal_replay",
            "segment_index": step - 1,
            "t_before": (step - 1) * STEP,
        },
    )
    return segment, diagnostics


def _queue_snapshot(state: FlowstarNormalFlowpipeState) -> dict[str, Any]:
    queue = state.symbolic_queue
    if queue is None:
        raise RuntimeError("SR100 replay lost its accepted-boundary queue")
    return {
        "sha256": accepted_boundary_sr_queue_sha256(queue),
        "generation": queue.generation,
        "accepted_boundary_index": queue.accepted_boundary_index,
        "size": len(queue.J),
        "reset_count": queue.reset_count,
        "owner_generations": list(queue.owner_generations),
        "owner_boundary_indices": list(queue.owner_boundary_indices),
    }


def _checkpoint(
    directory: Path,
    *,
    current: TMVector,
    state: FlowstarNormalFlowpipeState,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    return save_terminal_checkpoint(
        directory,
        current=current,
        normal_state=state,
        scheduler={
            "current_time_hex": (EXPECTED_ACCEPTED_STEPS * STEP).hex(),
            "h_hex": STEP.hex(),
            "accepted_steps": EXPECTED_ACCEPTED_STEPS,
            "next_attempt": TERMINAL_STEP,
        },
        contract=CHECKPOINT_CONTRACT,
        provenance=provenance,
    )


def replay(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status = _git("status", "--short")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    command = {
        "schema": "brusselator_sr100_terminal_replay_command_v1",
        "argv": sys.argv,
        "cwd": str(ROOT),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "worktree_status": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "published_segments_sha256": _sha256(PUBLISHED_SEGMENTS),
        "terminal_attempt_budget": 1,
    }
    _write_json(output / "command.json", command)
    if status:
        raise RuntimeError("terminal replay requires a clean worktree")

    expected = _read_expected()
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(INITIAL_DECIMAL, ORDER)
    current = state.normalized_initial_tm(ORDER)
    prefix_signatures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step_index, expected_row in enumerate(expected, start=1):
        segment, diagnostics = _step(current, state, step_index)
        if (
            segment.status != "validated"
            or segment.endpoint_raw_tm is None
            or segment.reset_tm is None
            or segment.flowstar_normal_state is None
        ):
            raise RuntimeError(f"SR100 prefix reconstruction failed at {step_index}: {segment.message}")
        passed, records, failures = _validation_passed(diagnostics)
        if not passed or failures or records == 0:
            raise RuntimeError(f"SR100 prefix validation evidence failed at {step_index}")
        observed = {
            **_box_fields("endpoint", segment.endpoint_raw_tm.range_box()),
            **_box_fields("tube", segment.tm.range_box()),
            **_queue_fields(segment.flowstar_normal_state, step_index),
        }
        differences = [
            field
            for field in BOUND_HEX_FIELDS
            if observed[field] != expected_row[field]
        ]
        if observed["queue_hash"] != expected_row["queue_hash"]:
            differences.append("queue_hash")
        if differences:
            raise RuntimeError(
                f"SR100 prefix differs from published boundary {step_index}: {differences}"
            )
        prefix_signatures.append(
            {
                "step": step_index,
                **{field: observed[field] for field in BOUND_HEX_FIELDS},
                "queue_hash": observed["queue_hash"],
            }
        )
        current = segment.reset_tm
        state = segment.flowstar_normal_state

    prefix_sha256 = hashlib.sha256(_canonical_bytes(prefix_signatures)).hexdigest()
    provenance = {
        "producer": "torch_sr100_terminal_replay",
        "commit": command["commit"],
        "contract_sha256": command["contract_sha256"],
        "published_prefix_signature_sha256": prefix_sha256,
    }
    queue_before = _queue_snapshot(state)
    before_manifest = _checkpoint(
        output / "checkpoint_before", current=current, state=state, provenance=provenance
    )

    # The contract authorizes exactly this one terminal attempt.  There is no retry.
    terminal_started = time.perf_counter()
    terminal, diagnostics = _step(current, state, TERMINAL_STEP)
    terminal_runtime = time.perf_counter() - terminal_started

    queue_after = _queue_snapshot(state)
    after_manifest = _checkpoint(
        output / "checkpoint_after", current=current, state=state, provenance=provenance
    )
    before_dir = output / "checkpoint_before"
    after_dir = output / "checkpoint_after"
    payload_equal = (before_dir / PAYLOAD_NAME).read_bytes() == (
        after_dir / PAYLOAD_NAME
    ).read_bytes()
    manifest_equal = (before_dir / MANIFEST_NAME).read_bytes() == (
        after_dir / MANIFEST_NAME
    ).read_bytes()
    validation_passed, validation_records, validation_failures = _validation_passed(
        diagnostics
    )
    rejected_without_publication = (
        terminal.status != "validated"
        and terminal.endpoint_raw_tm is None
        and terminal.reset_tm is None
        and terminal.flowstar_normal_state is None
    )
    rollback = (
        rejected_without_publication
        and payload_equal
        and manifest_equal
        and before_manifest["full_checkpoint_sha256"]
        == after_manifest["full_checkpoint_sha256"]
        and queue_before == queue_after
    )
    result = {
        "schema": "brusselator_sr100_terminal_replay_result_v1",
        "status": "C3_TERMINAL_ROLLBACK_CLOSED" if rollback else "C3_TERMINAL_ROLLBACK_FAILED",
        "contract_sha256": command["contract_sha256"],
        "published_prefix_reconstructed_bit_exact": True,
        "published_prefix_boundaries": EXPECTED_ACCEPTED_STEPS,
        "published_prefix_signature_sha256": prefix_sha256,
        "terminal_attempt_count": 1,
        "terminal_step": TERMINAL_STEP,
        "terminal_segment_status": terminal.status,
        "terminal_message": terminal.message,
        "terminal_validation_passed": validation_passed,
        "terminal_validation_record_count": validation_records,
        "terminal_validation_failure_count": validation_failures,
        "terminal_rejected_without_publication": rejected_without_publication,
        "queue_before": queue_before,
        "queue_after": queue_after,
        "checkpoint_payload_byte_equal": payload_equal,
        "checkpoint_manifest_byte_equal": manifest_equal,
        "checkpoint_full_sha256_before": before_manifest["full_checkpoint_sha256"],
        "checkpoint_full_sha256_after": after_manifest["full_checkpoint_sha256"],
        "rollback_proved": rollback,
        "prefix_reconstruction_wall_seconds": terminal_started - started,
        "terminal_attempt_wall_seconds": terminal_runtime,
    }
    _write_json(output / "RESULT.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = replay(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["rollback_proved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
