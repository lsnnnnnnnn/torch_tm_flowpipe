#!/usr/bin/env python3
"""Exercise B1/B2/B8 independent CPU lanes against the frozen C4 plant."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_brusselator_sr1000_parity import ORDER, _policy, _step  # noqa: E402
from torch_tm_flowpipe import (  # noqa: E402
    CPUPolynomialPlantBatch,
    CPUPolynomialPlantLane,
    DENSE_OBSERVER_NONE,
    FlowstarNormalFlowpipeState,
    cpu_batch_fingerprint,
    cpu_batch_lane_fingerprint,
    load_cpu_batch_checkpoint,
    run_independent_cpu_batch,
    save_cpu_batch_checkpoint,
)


DEFAULT_BOX = (("1.48", "1.52"), ("2.98", "3.02"))
SLIGHTLY_DIFFERENT_BOX = (("1.47", "1.51"), ("2.97", "3.01"))
DESIGNED_REJECTION_BOX = (("1", "2"), ("2", "4"))
DIFFERENT_REFINEMENT_BOX = (("1.4", "1.6"), ("2.9", "3.1"))


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _lane(lane_id: str, box: Sequence[Sequence[str]] = DEFAULT_BOX) -> CPUPolynomialPlantLane:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(box, ORDER)
    return CPUPolynomialPlantLane(
        lane_id=lane_id,
        current=state.normalized_initial_tm(ORDER),
        normal_state=state,
    )


def _step_lane(lane: CPUPolynomialPlantLane):
    segment, _diagnostics = _step(
        lane.current,
        lane.normal_state,
        lane.normal_state.step_index + 1,
        _policy(),
        validation_mode="flowstar_raw_remainder_compat_refined",
        lane_label=f"cpu_batch:{lane.lane_id}",
        observer_mode=DENSE_OBSERVER_NONE,
    )
    return segment


def _scientific_payload(lane: CPUPolynomialPlantLane) -> dict[str, Any]:
    payload = cpu_batch_lane_fingerprint(lane)
    payload.pop("lane_id")
    return payload


def _payload_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _equivalence_row(
    *,
    scenario: str,
    layout: str,
    lane: CPUPolynomialPlantLane,
    oracle: CPUPolynomialPlantLane,
) -> dict[str, Any]:
    actual = _scientific_payload(lane)
    expected = _scientific_payload(oracle)
    return {
        "scenario": scenario,
        "layout": layout,
        "lane_id": lane.lane_id,
        "oracle_lane_id": oracle.lane_id,
        "actual_sha256": _payload_sha(actual),
        "oracle_sha256": _payload_sha(expected),
        "endpoint_equal": actual["last_endpoint_hashes"] == expected["last_endpoint_hashes"],
        "tube_equal": actual["last_tube_hashes"] == expected["last_tube_hashes"],
        "remainder_equal": actual["last_final_remainder"] == expected["last_final_remainder"],
        "queue_equal": actual["queue"] == expected["queue"],
        "replay_equal": (
            actual["last_replay_calls"], actual["last_committed_replays"]
        )
        == (expected["last_replay_calls"], expected["last_committed_replays"]),
        "status_equal": (
            actual["last_status"], actual["accepted_steps"], actual["rejected_steps"]
        )
        == (expected["last_status"], expected["accepted_steps"], expected["rejected_steps"]),
        "passed": actual == expected,
    }


def _timed(action):
    started = time.perf_counter()
    value = action()
    return value, time.perf_counter() - started


def _duplicate_lanes(prefix: str = "duplicate") -> tuple[CPUPolynomialPlantLane, ...]:
    return tuple(_lane(f"{prefix}_{index}") for index in range(8))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cpu is not None:
        os.sched_setaffinity(0, {int(args.cpu)})
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    affinity = sorted(os.sched_getaffinity(0))

    rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    # Duplicate embedding and diagnostic latency.  Both measurements execute
    # exactly eight independently owned B1 lanes; only the orchestration shape
    # differs.
    serial_times: list[float] = []
    serial_results: list[tuple[CPUPolynomialPlantLane, ...]] = []
    for repeat in range(args.runtime_repeats):
        serial_initial = _duplicate_lanes(f"serial_r{repeat}")
        started = time.perf_counter()
        result_lanes = []
        for lane in serial_initial:
            result = run_independent_cpu_batch(
                CPUPolynomialPlantBatch((lane,)),
                _step_lane,
                cycles=1,
            )
            result_lanes.append(result.lanes[0])
        elapsed = time.perf_counter() - started
        serial_times.append(elapsed)
        serial_results.append(tuple(result_lanes))
        runtime_rows.append(
            {
                "case": "8x_serial_B1",
                "batch_size": 1,
                "lane_steps": 8,
                "repeat": repeat,
                "wall_s": elapsed,
                "per_lane_s": elapsed / 8.0,
                "lanes_per_s": 8.0 / elapsed,
                "peak_rss_bytes": _peak_rss_bytes(),
                "rss_source": "RUSAGE_SELF_highwater",
                "cpu_affinity": ";".join(str(value) for value in affinity),
            }
        )

    b8_times: list[float] = []
    b8_results: list[CPUPolynomialPlantBatch] = []
    for repeat in range(args.runtime_repeats):
        initial = CPUPolynomialPlantBatch(_duplicate_lanes(f"batch_r{repeat}"))
        result, elapsed = _timed(
            lambda initial=initial: run_independent_cpu_batch(
                initial,
                _step_lane,
                cycles=1,
            )
        )
        b8_times.append(elapsed)
        b8_results.append(result)
        runtime_rows.append(
            {
                "case": "B8_independent_lane_batch",
                "batch_size": 8,
                "lane_steps": 8,
                "repeat": repeat,
                "wall_s": elapsed,
                "per_lane_s": elapsed / 8.0,
                "lanes_per_s": 8.0 / elapsed,
                "peak_rss_bytes": _peak_rss_bytes(),
                "rss_source": "RUSAGE_SELF_highwater",
                "cpu_affinity": ";".join(str(value) for value in affinity),
            }
        )

    oracle = serial_results[0][0]
    for batch_size in (1, 2, 8):
        initial = CPUPolynomialPlantBatch(
            tuple(_lane(f"embed_b{batch_size}_{index}") for index in range(batch_size))
        )
        embedded = run_independent_cpu_batch(initial, _step_lane, cycles=1)
        for lane in embedded.lanes:
            rows.append(
                _equivalence_row(
                    scenario="duplicate_embedding",
                    layout=f"B{batch_size}",
                    lane=lane,
                    oracle=oracle,
                )
            )

    # Heterogeneous cases prove rejection rollback and per-lane refinement stop.
    heterogeneous_initial = CPUPolynomialPlantBatch(
        (
            _lane("heterogeneous_0_frozen_accept"),
            _lane("heterogeneous_1_shifted", SLIGHTLY_DIFFERENT_BOX),
            _lane("heterogeneous_2_designed_reject", DESIGNED_REJECTION_BOX),
            _lane("heterogeneous_3_different_refinement", DIFFERENT_REFINEMENT_BOX),
            *tuple(_lane(f"heterogeneous_{index}_duplicate") for index in range(4, 8)),
        )
    )
    rejection_before = _scientific_payload(heterogeneous_initial.lanes[2])
    heterogeneous = run_independent_cpu_batch(
        heterogeneous_initial,
        _step_lane,
        cycles=1,
    )
    reference_lane = heterogeneous.lanes[0]
    for index, lane in enumerate(heterogeneous.lanes):
        payload = _scientific_payload(lane)
        if index == 2:
            passed = (
                lane.frozen
                and lane.rejected_steps == 1
                and lane.accepted_steps == 0
                and payload["current_hashes"] == rejection_before["current_hashes"]
                and payload["state_step_index"] == rejection_before["state_step_index"]
                and payload["queue"] == rejection_before["queue"]
            )
            oracle_lane = heterogeneous_initial.lanes[2]
        elif index == 3:
            passed = (
                lane.last_status == "validated"
                and lane.last_committed_replays != reference_lane.last_committed_replays
            )
            oracle_lane = lane
        elif index >= 4:
            passed = _scientific_payload(lane) == _scientific_payload(reference_lane)
            oracle_lane = reference_lane
        else:
            passed = lane.last_status == "validated" and not lane.frozen
            oracle_lane = lane
        row = _equivalence_row(
            scenario="heterogeneous_lane_isolation",
            layout="B8",
            lane=lane,
            oracle=oracle_lane,
        )
        row["passed"] = passed
        row["observed_replays"] = lane.last_committed_replays
        rows.append(row)

    # Chunk invariance uses the same lane ids and initial objects in every view.
    chunk_initial = _duplicate_lanes("chunk")
    whole = run_independent_cpu_batch(
        CPUPolynomialPlantBatch(chunk_initial),
        _step_lane,
        cycles=1,
    )
    for chunk_size in (4, 2, 1):
        chunked: list[CPUPolynomialPlantLane] = []
        for offset in range(0, 8, chunk_size):
            part = run_independent_cpu_batch(
                CPUPolynomialPlantBatch(chunk_initial[offset : offset + chunk_size]),
                _step_lane,
                cycles=1,
            )
            chunked.extend(part.lanes)
        for lane, whole_lane in zip(chunked, whole.lanes):
            rows.append(
                _equivalence_row(
                    scenario="chunk_invariance",
                    layout=f"{8 // chunk_size}xB{chunk_size}",
                    lane=lane,
                    oracle=whole_lane,
                )
            )

    # Resume a B8 after one cycle and compare with its uninterrupted successor.
    checkpoint_initial = CPUPolynomialPlantBatch(_duplicate_lanes("checkpoint"))
    checkpoint_boundary = run_independent_cpu_batch(
        checkpoint_initial,
        _step_lane,
        cycles=1,
    )
    uninterrupted = run_independent_cpu_batch(
        checkpoint_boundary,
        _step_lane,
        cycles=1,
    )
    with tempfile.TemporaryDirectory(prefix="c4_cpu_batch_checkpoint_") as temporary:
        checkpoint_path = Path(temporary) / "B8"
        contract = {
            "plant": "brusselator",
            "order": ORDER,
            "dtype": "float64",
            "lane_semantics": "independent",
        }
        save_cpu_batch_checkpoint(
            checkpoint_path,
            checkpoint_boundary,
            contract=contract,
            provenance={"script": "run_c4_cpu_batch_equivalence.py"},
        )
        loaded = load_cpu_batch_checkpoint(
            checkpoint_path,
            expected_contract=contract,
        )
        resumed = run_independent_cpu_batch(loaded, _step_lane, cycles=1)
    for lane, oracle_lane in zip(resumed.lanes, uninterrupted.lanes):
        rows.append(
            _equivalence_row(
                scenario="checkpoint_resume",
                layout="B8_resume_after_cycle_1",
                lane=lane,
                oracle=oracle_lane,
            )
        )

    serial_median = statistics.median(serial_times)
    b8_median = statistics.median(b8_times)
    slowdown = b8_median / serial_median
    equivalence_passed = all(bool(row["passed"]) for row in rows)
    runtime_passed = slowdown <= 2.0
    result = {
        "schema": "torch_tm_flowpipe.c4_cpu_batch_equivalence/1",
        "status": (
            "CPU_BATCH_FOUNDATION_PASSED"
            if equivalence_passed
            else "CPU_BATCH_LANE_ISOLATION_FAILED_STOP"
        ),
        "scientific_sha": os.environ.get("C4_SCIENTIFIC_SHA", ""),
        "batch_sizes": [1, 2, 8],
        "cpu_affinity": affinity,
        "dtype": "torch.float64",
        "device": "cpu",
        "independent_lane_orchestration": True,
        "fused_full_solver_kernel_claimed": False,
        "equivalence_passed": equivalence_passed,
        "checkpoint_resume_passed": all(
            bool(row["passed"]) for row in rows if row["scenario"] == "checkpoint_resume"
        ),
        "chunk_invariance_passed": all(
            bool(row["passed"]) for row in rows if row["scenario"] == "chunk_invariance"
        ),
        "heterogeneous_isolation_passed": all(
            bool(row["passed"])
            for row in rows
            if row["scenario"] == "heterogeneous_lane_isolation"
        ),
        "duplicate_embedding_passed": all(
            bool(row["passed"])
            for row in rows
            if row["scenario"] == "duplicate_embedding"
        ),
        "serial_b1_median_s": serial_median,
        "b8_median_s": b8_median,
        "b8_slowdown_vs_8x_serial_b1": slowdown,
        "b8_runtime_diagnostic_passed": runtime_passed,
        "architecture_note": (
            "The foundation batches independently owned B1 solver lanes with fixed CPU orchestration; "
            "full-solver tensor fusion is intentionally deferred until these semantics are the CUDA oracle."
        ),
        "whole_b8_sha256": cpu_batch_fingerprint(whole)["sha256"],
        "equivalence_rows": len(rows),
    }
    _write_csv(output_dir / "cpu_batch_equivalence.csv", rows)
    _write_csv(output_dir / "cpu_batch_runtime.csv", runtime_rows)
    _write_json(output_dir / "cpu_batch_result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-repeats", type=int, default=1)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args(argv)
    if args.runtime_repeats < 1:
        parser.error("--runtime-repeats must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(result["status"])
    return 0 if result["equivalence_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
