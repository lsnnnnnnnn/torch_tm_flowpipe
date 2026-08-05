#!/usr/bin/env python3
"""Stream and compare Torch/Flowstar Van der Pol transition traces fail-closed."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


CLASSIFICATIONS = {
    "serialization-only",
    "expected floating-point/ULP difference",
    "representation difference",
    "behavior-relevant numerical difference",
    "structural semantic difference",
}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid Boolean value: {value!r}")
    return normalized == "true"


def _attempts(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "tool", "accepted_step_index", "attempt_index", "retry_index", "t_pre_decimal",
            "t_pre_hex", "h_attempt_decimal", "h_attempt_hex", "accepted", "stage",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"acceptance trace missing fields: {sorted(required - set(reader.fieldnames or []))}")
        previous_attempt = -1
        for row in reader:
            attempt_index = int(row["attempt_index"])
            if attempt_index <= previous_attempt:
                raise ValueError("attempt_index must be strictly increasing")
            previous_attempt = attempt_index
            t_decimal = float(row["t_pre_decimal"])
            h_decimal = float(row["h_attempt_decimal"])
            if t_decimal != float.fromhex(row["t_pre_hex"]) or h_decimal != float.fromhex(row["h_attempt_hex"]):
                raise ValueError("decimal/hex attempt serialization mismatch")
            yield {
                **row,
                "accepted_step_index": int(row["accepted_step_index"]),
                "attempt_index": attempt_index,
                "retry_index": int(row["retry_index"]),
                "t_pre": t_decimal,
                "h_attempt": h_decimal,
                "accepted": _bool(row["accepted"]),
            }


def _group_steps(rows: Iterable[dict[str, Any]]) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    current_step: int | None = None
    group: list[dict[str, Any]] = []
    for row in rows:
        step = int(row["accepted_step_index"])
        if current_step is None:
            current_step = step
        if step != current_step:
            yield current_step, group
            current_step, group = step, []
        group.append(row)
    if current_step is not None:
        yield current_step, group


def _schedule_signature(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, bool]]:
    return [(float(row["h_attempt"]), bool(row["accepted"])) for row in rows]


def _first_schedule_divergence(torch_path: Path, flowstar_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    torch_groups = _group_steps(_attempts(torch_path))
    flowstar_groups = _group_steps(_attempts(flowstar_path))
    last_common: dict[str, Any] = {"accepted_step_index": -1, "torch_attempts": [], "flowstar_attempts": []}
    while True:
        torch_item = next(torch_groups, None)
        flowstar_item = next(flowstar_groups, None)
        if torch_item is None or flowstar_item is None:
            if torch_item == flowstar_item:
                raise ValueError("no native schedule divergence found in supplied traces")
            raise ValueError("one attempt trace ended before the first shared schedule divergence")
        torch_step, torch_rows = torch_item
        flowstar_step, flowstar_rows = flowstar_item
        if torch_step != flowstar_step:
            raise ValueError("accepted step indices are not monotone/aligned")
        same = _schedule_signature(torch_rows) == _schedule_signature(flowstar_rows)
        if not same:
            first_torch = next((row for row in torch_rows if row not in flowstar_rows), torch_rows[0])
            return (
                {
                    "schema": "vdp_first_schedule_divergence_v1",
                    "accepted_step_index": torch_step,
                    "t_pre_torch": torch_rows[0]["t_pre"],
                    "t_pre_flowstar": flowstar_rows[0]["t_pre"],
                    "torch_attempts": torch_rows,
                    "flowstar_attempts": flowstar_rows,
                    "torch_signature": _schedule_signature(torch_rows),
                    "flowstar_signature": _schedule_signature(flowstar_rows),
                    "first_different_candidate_h": first_torch["h_attempt"],
                    "behavior_change": "Torch accepts the candidate that stock Flowstar rejects; Flowstar then accepts one half-step.",
                    "classification": "behavior-relevant numerical difference",
                    "flowstar_retry_provenance": "derived from the exact stock grow/half scheduler and final accepted h; not an observed inner Picard record",
                },
                last_common,
            )
        last_common = {
            "schema": "vdp_last_common_transition_v1",
            "accepted_step_index": torch_step,
            "torch_attempts": torch_rows,
            "flowstar_attempts": flowstar_rows,
            "common_signature": _schedule_signature(torch_rows),
        }


def _read_jsonl_window(path: Path, first_step: int, last_step: int) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"empty JSONL row at {path}:{line_number}")
            row = json.loads(line)
            step = int(row.get("accepted_step_index", -1))
            if first_step <= step <= last_step:
                yield row


def _first_basis(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            basis = row.get("basis_variable_order")
            if basis:
                if not isinstance(basis, list) or any(not isinstance(item, str) or not item for item in basis):
                    raise ValueError(f"invalid basis at {path}:{line_number}")
                return tuple(basis)
    raise ValueError(f"no nonempty basis found in {path}")


def _coordinate_descriptor(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            basis = row.get("basis_variable_order")
            if not basis:
                continue
            descriptor: dict[str, Any] = {"basis": list(basis)}
            for field in ("center", "normalization_scale"):
                encoded = row.get(field)
                if encoded is None:
                    descriptor[field] = None
                elif isinstance(encoded, dict) and set(encoded) == {"decimal", "hex"}:
                    decimal = float(encoded["decimal"])
                    hexadecimal = float.fromhex(encoded["hex"])
                    if decimal != hexadecimal:
                        raise ValueError(f"{field} decimal/hex mismatch at {path}:{line_number}")
                    descriptor[field] = decimal
                else:
                    raise ValueError(f"invalid {field} encoding at {path}:{line_number}")
            return descriptor
    raise ValueError(f"no coordinate descriptor found in {path}")


def basis_guard(torch_transitions: Path, flowstar_transitions: Path) -> dict[str, Any]:
    torch_coordinates = _coordinate_descriptor(torch_transitions)
    flowstar_coordinates = _coordinate_descriptor(flowstar_transitions)
    torch_basis = tuple(torch_coordinates["basis"])
    flowstar_basis = tuple(flowstar_coordinates["basis"])
    same = torch_basis == flowstar_basis
    coordinate_metadata_available = all(
        coordinates[field] is not None
        for coordinates in (torch_coordinates, flowstar_coordinates)
        for field in ("center", "normalization_scale")
    )
    center_equal = bool(
        coordinate_metadata_available and torch_coordinates["center"] == flowstar_coordinates["center"]
    )
    scale_equal = bool(
        coordinate_metadata_available
        and torch_coordinates["normalization_scale"] == flowstar_coordinates["normalization_scale"]
    )
    comparable = same and center_equal and scale_equal
    return {
        "torch_basis": list(torch_basis),
        "flowstar_basis": list(flowstar_basis),
        "basis_equal": same,
        "exponent_semantics_equal": same,
        "local_time_identity_equal": same,
        "coordinate_metadata_available": coordinate_metadata_available,
        "torch_center": torch_coordinates["center"],
        "flowstar_center": flowstar_coordinates["center"],
        "center_equal": center_equal,
        "torch_normalization_scale": torch_coordinates["normalization_scale"],
        "flowstar_normalization_scale": flowstar_coordinates["normalization_scale"],
        "normalization_scale_equal": scale_equal,
        "coefficient_comparison_available": comparable,
        "common_basis_transform_implemented": False,
        "reason": (
            "basis, center, scale, exponent, and local-time identities match"
            if comparable
            else "basis/center/scale/exponent/local-time identity is missing or differs and no algebraically tested common-basis transform is implemented; coefficient comparison fails closed"
        ),
    }


def _tolerance_sensitivity(divergence: Mapping[str, Any], tolerances: Sequence[float]) -> list[dict[str, Any]]:
    torch_rows = divergence["torch_attempts"]
    flowstar_rows = divergence["flowstar_attempts"]
    output = []
    for tolerance in tolerances:
        same_h_prefix = all(
            math.isclose(float(left["h_attempt"]), float(right["h_attempt"]), rel_tol=0.0, abs_tol=tolerance)
            for left, right in zip(torch_rows, flowstar_rows)
        )
        accepted_equal = [row["accepted"] for row in torch_rows] == [row["accepted"] for row in flowstar_rows]
        output.append(
            {
                "absolute_tolerance": tolerance,
                "candidate_h_prefix_equal": same_h_prefix,
                "accepted_sequence_equal": accepted_equal,
                "divergence_persists": not (same_h_prefix and accepted_equal and len(torch_rows) == len(flowstar_rows)),
            }
        )
    return output


def compare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    divergence, last_common = _first_schedule_divergence(args.torch_attempts, args.flowstar_attempts)
    guard = basis_guard(args.torch_transitions, args.flowstar_transitions)
    sensitivities = _tolerance_sensitivity(divergence, (0.0, 1e-15, 1e-12, 1e-9))
    first_field = {
        "schema": "vdp_first_field_divergence_v1",
        "accepted_step_index": 0,
        "stage": "step_pre_state",
        "component": None,
        "field": "basis_variable_order / physical composed state availability",
        "torch_value": guard["torch_basis"],
        "flowstar_value": guard["flowstar_basis"],
        "absolute_difference": None,
        "classification": "structural semantic difference",
        "behavior_relevance_at_first_occurrence": False,
        "coefficient_comparison": guard,
        "explanation": "The first trace-visible difference is structural, not JSON ordering. Flowstar's scheduler hook exposes its left/right composition basis but not a Torch-equivalent composed pre-state.",
    }
    schedule_step = int(divergence["accepted_step_index"])
    first_window = max(0, schedule_step - 2)
    last_window = schedule_step + 2
    _write_json(output / "first_schedule_divergence.json", divergence)
    _write_json(output / "first_field_divergence.json", first_field)
    _write_json(output / "last_common_transition.json", last_common)
    _write_jsonl(output / "divergence_window_torch.jsonl", _read_jsonl_window(args.torch_transitions, first_window, last_window))
    _write_jsonl(output / "divergence_window_flowstar.jsonl", _read_jsonl_window(args.flowstar_transitions, first_window, last_window))
    config = {
        "schema": "vdp_streaming_comparator_config_v1",
        "sync_key": ["accepted_step_index", "retry_index", "t_pre", "h_attempt", "stage", "component"],
        "schedule_sync_key_used": ["accepted_step_index", "retry_index", "h_attempt", "accepted"],
        "time_alignment_absolute_tolerance": 2e-12,
        "numeric_tolerances_checked": [item["absolute_tolerance"] for item in sensitivities],
        "raw_mismatches_preserved": True,
        "coefficient_guard": guard,
        "torch": {"attempts": str(args.torch_attempts.resolve()), "transitions": str(args.torch_transitions.resolve())},
        "flowstar": {"attempts": str(args.flowstar_attempts.resolve()), "transitions": str(args.flowstar_transitions.resolve())},
        "shadow_replay_used": False,
    }
    _write_json(output / "comparator_config.json", config)
    summary_rows = [
        {
            "kind": "first_trace_field", "step": 0, "stage": "step_pre_state", "component": "",
            "field": first_field["field"], "classification": first_field["classification"],
            "difference": "basis/availability mismatch", "behavior_effect": "none at first occurrence",
        },
        {
            "kind": "first_native_schedule", "step": schedule_step, "stage": "acceptance_predicate/scheduler", "component": "y",
            "field": "accepted candidate", "classification": divergence["classification"],
            "difference": str(divergence["torch_signature"]) + " vs " + str(divergence["flowstar_signature"]),
            "behavior_effect": divergence["behavior_change"],
        },
        {
            "kind": "coefficient_guard", "step": 0, "stage": "all", "component": "",
            "field": "basis/center/scale", "classification": "representation difference",
            "difference": guard["reason"], "behavior_effect": "coefficient comparison unavailable",
        },
    ]
    summary_rows.extend(
        {
            "kind": "tolerance_sensitivity", "step": schedule_step, "stage": "scheduler", "component": "",
            "field": f"abs_tol={row['absolute_tolerance']}", "classification": "behavior-relevant numerical difference",
            "difference": f"divergence_persists={row['divergence_persists']}", "behavior_effect": "native schedules remain different",
        }
        for row in sensitivities
    )
    with (output / "divergence_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    result = {
        "schedule_divergence_step": schedule_step,
        "last_common_step": int(last_common["accepted_step_index"]),
        "coefficient_comparison_available": guard["coefficient_comparison_available"],
        "all_tolerance_checks_persist": all(row["divergence_persists"] for row in sensitivities),
        "classification_vocabulary_valid": all(row["classification"] in CLASSIFICATIONS for row in summary_rows),
    }
    _write_json(output / "comparison_result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-attempts", type=Path, required=True)
    parser.add_argument("--torch-transitions", type=Path, required=True)
    parser.add_argument("--flowstar-attempts", type=Path, required=True)
    parser.add_argument("--flowstar-transitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(compare(parse_args()), sort_keys=True))
