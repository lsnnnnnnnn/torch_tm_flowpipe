#!/usr/bin/env python3
"""Run source-bound native TORA-Q3 lanes and derive strict hierarchical gates.

The frozen native closed-loop scheduler remains the single implementation of
controller refresh, affine carry composition, and serialization.  This driver
only selects a formal plant lane, requests a diagnostic continuation after a
property failure, and derives gates which never advance past a failed gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_tora_q3_full_closed_loop as frozen_scheduler
from torch_tm_flowpipe.tora_algorithm_aligned import algorithm_aligned_q3_step


FORMAL_LANES = {
    "baseline_native_k2": "baseline_native",
    "k3_picard": "k3_picard",
    "algorithm_aligned_q3": "algorithm_aligned_q3",
}
GATE_SPECS = (
    ("one_leaf_one_step", 1, 1),
    ("b48_one_step", 1, 48),
    ("b48_t1", 10, 48),
    ("b48_t5", 50, 48),
    ("b48_t10", 100, 48),
    ("b48_t20", 200, 48),
)
PREDICATES = {
    "finite_ok": "finite_ok_by_leaf",
    "initial_subset_ok": "initial_subset_ok_by_leaf",
    "all_remainder_rounds_ok": "all_remainder_rounds_ok_by_leaf",
    "local_property_ok": "local_property_ok_by_leaf",
    "composed_property_ok": "composed_property_ok_by_leaf",
    "overall_accepted": "overall_accepted_by_leaf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def predicate_counts(
    rows: Iterable[Mapping[str, Any]], *, leaf_count: int
) -> dict[str, dict[str, int | bool]]:
    selected = list(rows)
    counts: dict[str, dict[str, int | bool]] = {}
    for public_name, raw_name in PREDICATES.items():
        values = [
            bool(value)
            for row in selected
            for value in row["predicates"][raw_name][:leaf_count]
        ]
        counts[public_name] = {
            "all": bool(values) and all(values),
            "true": sum(values),
            "total": len(values),
        }
    return counts


def derive_gates(
    rows: list[dict[str, Any]], summary: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Derive a strict previous-pass-only hierarchy from a diagnostic trace."""
    indexed = {int(row["segment_index"]): row for row in rows}
    gates: list[dict[str, Any]] = []
    previous_passed = True
    for gate_name, required_segments, leaf_count in GATE_SPECS:
        if not previous_passed:
            gates.append(
                {
                    "gate": gate_name,
                    "status": "NOT_RUN",
                    "reason": "previous hierarchical gate did not pass",
                    "required_segments": required_segments,
                    "expected_leaf_count": leaf_count,
                    "certified_horizon": None,
                    "predicate_counts": None,
                }
            )
            continue
        selected = [
            indexed[index]
            for index in range(1, required_segments + 1)
            if index in indexed
        ]
        counts = predicate_counts(selected, leaf_count=leaf_count)
        trace_complete = len(selected) == required_segments
        predicates_passed = all(
            bool(item["all"]) for item in counts.values()
        )
        passed = trace_complete and predicates_passed
        if gate_name == "one_leaf_one_step":
            completed = 1 if passed else 0
        else:
            completed = min(int(summary["completed_segments"]), required_segments)
        gate: dict[str, Any] = {
            "gate": gate_name,
            "status": "PASS" if passed else "FAIL",
            "required_segments": required_segments,
            "observed_segments": len(selected),
            "expected_leaf_count": leaf_count,
            "completed_segments": completed,
            "certified_horizon": 0.1 * completed,
            "predicate_counts": counts,
        }
        if not passed:
            gate["failure"] = summary.get("first_failure") or {
                "reason": "incomplete diagnostic trace"
            }
        gates.append(gate)
        previous_passed = passed
    return gates


def center_radius(lower: float, upper: float) -> dict[str, float]:
    center = float(lower) + 0.5 * (float(upper) - float(lower))
    return {
        "center": center,
        "radius": max(center - float(lower), float(upper) - center),
    }


def build_failure_detail(
    rows: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    first = summary.get("first_failure")
    payload: dict[str, Any] = {
        "schema": "tora_q3_native_failure_detail_v1",
        "first_failure": first,
        "available": False,
    }
    if not first or "segment" not in first:
        return payload
    segment = int(first["segment"])
    row = next((item for item in rows if int(item["segment_index"]) == segment), None)
    if row is None:
        payload["reason_detail_unavailable"] = "failure occurred before segment serialization"
        return payload
    failed_ids = [int(value) for value in first.get("failed_leaf_ids", [])]
    if not failed_ids:
        failed_ids = [
            index
            for index, accepted in enumerate(row["accepted"])
            if not accepted
        ]
    controller = next(
        (
            item
            for item in controller_rows
            if int(item["controller_period"]) == int(row["controller_period"])
        ),
        None,
    )
    leaf_details = []
    for leaf_id in failed_ids:
        states = []
        for state in range(5):
            endpoint = center_radius(
                row["endpoint"]["lower"][leaf_id][state],
                row["endpoint"]["upper"][leaf_id][state],
            )
            tube = center_radius(
                row["tube"]["lower"][leaf_id][state],
                row["tube"]["upper"][leaf_id][state],
            )
            remainder = center_radius(
                row["interval_remainder"]["lower"][leaf_id][state],
                row["interval_remainder"]["upper"][leaf_id][state],
            )
            states.append(
                {
                    "state_index": state + 1,
                    "endpoint": endpoint,
                    "tube": tube,
                    "interval_remainder": remainder,
                    "property_margin": (
                        row["property_margin"][leaf_id][state]
                        if state < 4
                        else None
                    ),
                }
            )
        controller_detail = None
        if controller is not None:
            controller_detail = {
                "input": [
                    center_radius(
                        controller["pre_controller_state_box"]["lower"][leaf_id][state],
                        controller["pre_controller_state_box"]["upper"][leaf_id][state],
                    )
                    for state in range(4)
                ],
                "output_before_outward": center_radius(
                    controller["output_before_outward"]["lower"][leaf_id][0],
                    controller["output_before_outward"]["upper"][leaf_id][0],
                ),
                "output_after_outward": center_radius(
                    controller["output_after_outward"]["lower"][leaf_id][0],
                    controller["output_after_outward"]["upper"][leaf_id][0],
                ),
            }
        leaf_details.append(
            {
                "leaf_id": leaf_id,
                "states": states,
                "controller": controller_detail,
                "numerical_certificate": {
                    name: bool(row["predicates"][raw][leaf_id])
                    for name, raw in PREDICATES.items()
                    if name not in {
                        "local_property_ok",
                        "composed_property_ok",
                        "overall_accepted",
                    }
                },
                "property_predicates": {
                    name: bool(row["predicates"][raw][leaf_id])
                    for name, raw in PREDICATES.items()
                    if name in {
                        "local_property_ok",
                        "composed_property_ok",
                        "overall_accepted",
                    }
                },
            }
        )
    payload.update(
        {
            "available": True,
            "segment_index": segment,
            "physical_time": row["physical_time"],
            "controller_period": row["controller_period"],
            "failure_type": first.get("reason"),
            "failed_leaf_ids": failed_ids,
            "failed_leaves": leaf_details,
            "polynomial_remainder_decomposition": row["width_attribution"],
            "ledger_widths": row["ledger_widths"],
        }
    )
    return payload


def aligned_step_adapter(base: Any, **kwargs: Any) -> Any:
    rounds = int(kwargs.pop("polynomial_picard_rounds", 2))
    backend = str(kwargs.pop("point_enclosure_backend", "eager"))
    capture = bool(kwargs.pop("capture_trace", False))
    if kwargs:
        raise TypeError(f"unsupported aligned scheduler arguments: {sorted(kwargs)}")
    if rounds != 2:
        raise ValueError("algorithm_aligned_q3 freezes polynomial Picard K2")
    return algorithm_aligned_q3_step(
        base,
        polynomial_picard_rounds=2,
        remainder_rounds=10,
        capture_trace=capture,
        point_enclosure_backend=backend,
    )


def invoke_frozen_scheduler(args: argparse.Namespace) -> int:
    internal_lane = FORMAL_LANES[args.formal_lane]
    argv = [
        "run_tora_q3_full_closed_loop.py",
        "--output-dir",
        str(args.output_dir),
        "--controller-trace",
        str(args.controller_trace),
        "--expected-controller-trace-sha256",
        args.expected_controller_trace_sha256,
        "--device",
        args.device,
        "--periods",
        "20",
        "--run-id",
        args.run_id,
        "--lane",
        internal_lane,
        "--point-enclosure-backend",
        "compiled",
        "--optimized-math",
        "--continue-after-property-failure",
    ]
    original_argv = sys.argv
    original_add_argument = frozen_scheduler.argparse.ArgumentParser.add_argument
    original_step = frozen_scheduler.dense_tora_q3_dr_step

    def accepting_add_argument(self: Any, *names: str, **kwargs: Any) -> Any:
        if "--lane" in names and internal_lane == "algorithm_aligned_q3":
            kwargs["choices"] = tuple(kwargs["choices"]) + (
                "algorithm_aligned_q3",
            )
        return original_add_argument(self, *names, **kwargs)

    try:
        sys.argv = argv
        frozen_scheduler.argparse.ArgumentParser.add_argument = accepting_add_argument
        if internal_lane == "algorithm_aligned_q3":
            frozen_scheduler.dense_tora_q3_dr_step = aligned_step_adapter
        return frozen_scheduler.main()
    finally:
        sys.argv = original_argv
        frozen_scheduler.argparse.ArgumentParser.add_argument = original_add_argument
        frozen_scheduler.dense_tora_q3_dr_step = original_step


def augment_run(args: argparse.Namespace, scheduler_return_code: int) -> dict[str, Any]:
    output = args.output_dir.resolve()
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_jsonl(output / "segments.jsonl")
    controller_rows = read_jsonl(output / "controller_updates.jsonl")
    summary["schema"] = "torch_native_tora_q3_hierarchical_run_summary_v1"
    summary["formal_lane"] = args.formal_lane
    summary["scheduler_return_code"] = scheduler_return_code
    summary["diagnostic_continuation_is_not_a_formal_gate"] = True
    summary["hierarchical_gate_policy"] = "strict_previous_pass_only"
    summary["source_sha256"][
        "experiments/run_tora_q3_native_hierarchical.py"
    ] = sha256(Path(__file__).resolve())
    summary["source_sha256"][
        "src/torch_tm_flowpipe/tora_algorithm_aligned.py"
    ] = sha256(ROOT / "src/torch_tm_flowpipe/tora_algorithm_aligned.py")
    summary["maximum_process_rss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    gates = derive_gates(rows, summary)
    gate_payload = {
        "schema": "tora_q3_native_hierarchical_gates_private_v1",
        "formal_lane": args.formal_lane,
        "implementation_lane": summary["lane"],
        "status": "PASS" if all(row["status"] == "PASS" for row in gates) else "FAIL",
        "strict_previous_pass_only": True,
        "diagnostic_continuation_is_not_formal": True,
        "gates": gates,
        "config": summary["config"],
        "config_sha256": summary["config_sha256"],
        "source_sha256": summary["source_sha256"],
        "private_trace_sha256": {
            "segments": summary["segments_sha256"],
            "controller_updates": summary["controller_updates_sha256"],
            "replay_points": summary["replay_points_sha256"],
        },
    }
    write_json(output / "hierarchical_gates.json", gate_payload)
    failure = build_failure_detail(rows, controller_rows, summary)
    write_json(output / "failure_detail.json", failure)
    summary["hierarchical_gates_sha256"] = sha256(
        output / "hierarchical_gates.json"
    )
    summary["failure_detail_sha256"] = sha256(output / "failure_detail.json")
    summary["hierarchical_status"] = gate_payload["status"]
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--expected-controller-trace-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--formal-lane", choices=tuple(FORMAL_LANES), required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_code = invoke_frozen_scheduler(args)
    summary = augment_run(args, scheduler_code)
    print(
        json.dumps(
            {
                "formal_lane": summary["formal_lane"],
                "hierarchical_status": summary["hierarchical_status"],
                "certified_horizon": summary["certified_horizon"],
                "first_failure": summary["first_failure"],
            },
            separators=(",", ":"),
        )
    )
    # A sound, fully captured negative formal result is a successful experiment.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
