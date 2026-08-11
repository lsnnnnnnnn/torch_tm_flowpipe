"""Canonical benchmark identities and pairwise-comparison eligibility gates."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import torch


CONTRACT_IDENTITY_SCHEMA = "torch_tm_flowpipe_vdp_identity_v1"
EXECUTION_KINDS = frozenset({"native", "matched", "diagnostic"})
OUTPUT_OBJECTS = frozenset({"endpoint", "segment_tube", "prefix_tube"})


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def binary64_record(value: float) -> dict[str, Any]:
    number = float(value)
    return {
        "decimal": repr(number),
        "hex": number.hex(),
    }


def validate_binary64_record(value: Mapping[str, Any]) -> float:
    try:
        decimal = float(str(value["decimal"]))
        hexadecimal = float.fromhex(str(value["hex"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid decimal/hex binary64 record") from exc
    if decimal.hex() != hexadecimal.hex():
        raise ValueError("decimal and hexadecimal binary64 values disagree")
    return decimal


def vdp_rhs_identity() -> dict[str, Any]:
    """Return the expression-order-independent polynomial identity for VDP."""

    outputs = (
        (((0, 1), 1.0),),
        (((0, 1), 1.0), ((1, 0), -1.0), ((2, 1), -1.0)),
    )
    return {
        "schema": CONTRACT_IDENTITY_SCHEMA,
        "object": "polynomial_rhs",
        "state_order": ["x", "y"],
        "time_variable": {
            "autonomous_rhs": True,
            "local_time_symbol": "tau",
            "complete_o4_position": 0,
            "fixed_support_position": 0,
        },
        "outputs": [
            {
                "state": ("x", "y")[output_index],
                "terms": [
                    {
                        "powers_in_state_order": list(powers),
                        "coefficient": binary64_record(coefficient),
                    }
                    for powers, coefficient in sorted(output)
                ],
            }
            for output_index, output in enumerate(outputs)
        ],
    }


def vdp_initial_set_identity() -> dict[str, Any]:
    bounds = ((1.1, 1.4), (2.35, 2.45))
    return {
        "schema": CONTRACT_IDENTITY_SCHEMA,
        "object": "initial_box",
        "state_order": ["x", "y"],
        "bounds": [
            {
                "state": name,
                "lo": binary64_record(lo),
                "hi": binary64_record(hi),
            }
            for name, (lo, hi) in zip(("x", "y"), bounds)
        ],
    }


def vdp_partition_identity(batch: int) -> dict[str, Any]:
    batch = int(batch)
    if batch <= 0:
        raise ValueError("partition batch must be positive")
    root = int(batch**0.5)
    while batch % root:
        root -= 1
    split_x, split_y = batch // root, root
    x_edges = torch.linspace(1.1, 1.4, split_x + 1, dtype=torch.float64)
    y_edges = torch.linspace(2.35, 2.45, split_y + 1, dtype=torch.float64)
    boxes: list[dict[str, Any]] = []
    for x_index in range(split_x):
        for y_index in range(split_y):
            boxes.append(
                {
                    "index": len(boxes),
                    "lo": [
                        binary64_record(float(x_edges[x_index])),
                        binary64_record(float(y_edges[y_index])),
                    ],
                    "hi": [
                        binary64_record(float(x_edges[x_index + 1])),
                        binary64_record(float(y_edges[y_index + 1])),
                    ],
                }
            )
    return {
        "schema": CONTRACT_IDENTITY_SCHEMA,
        "object": "ordered_partition_list",
        "state_order": ["x", "y"],
        "batch": batch,
        "splits": [split_x, split_y],
        "enumeration": "x_major_then_y",
        "boxes": boxes,
    }


def vdp_identity_hashes() -> dict[str, str]:
    return {
        "rhs_sha256": canonical_sha256(vdp_rhs_identity()),
        "initial_set_sha256": canonical_sha256(vdp_initial_set_identity()),
        "partition_b1_sha256": canonical_sha256(vdp_partition_identity(1)),
        "partition_b64_sha256": canonical_sha256(vdp_partition_identity(64)),
    }


def validate_comparison_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate labeling and ranking eligibility for one comparison row."""

    required = (
        "execution_kind",
        "track",
        "partition_sha256",
        "partition_count",
        "output_object",
        "requested_horizon",
        "validated_horizon",
        "completion_status",
        "formal_ranking_eligible",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError("comparison row missing labels: " + ", ".join(missing))
    execution_kind = str(row["execution_kind"])
    if execution_kind not in EXECUTION_KINDS:
        raise ValueError("comparison row execution_kind is invalid")
    output_object = str(row["output_object"])
    if output_object not in OUTPUT_OBJECTS:
        raise ValueError("endpoint/tube output object is ambiguous")
    if execution_kind == "diagnostic":
        if row.get("diagnostic_only") is not True:
            raise ValueError("diagnostic row must set diagnostic_only=true")
        if row.get("formal_ranking_eligible") is not False:
            raise ValueError("diagnostic row cannot be ranking eligible")
    requested = float(row["requested_horizon"])
    validated = float(row["validated_horizon"])
    complete = str(row["completion_status"]) == "completed" and validated >= requested
    if not complete and row.get("formal_ranking_eligible") is True:
        raise ValueError("incomplete horizon cannot be ranking eligible")
    return dict(row)


def validate_pairwise_object_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject pairwise rows that silently compare different partitions/objects."""

    if len(rows) < 2:
        raise ValueError("pairwise identity requires at least two rows")
    validated = [validate_comparison_row(row) for row in rows]
    identities = {
        (
            row["track"],
            row["execution_kind"],
            row["partition_count"],
            row["partition_sha256"],
            row["output_object"],
        )
        for row in validated
    }
    if len(identities) != 1:
        raise ValueError(
            "pairwise rows do not describe the same track/execution/partition/output object"
        )


__all__ = [
    "CONTRACT_IDENTITY_SCHEMA",
    "EXECUTION_KINDS",
    "OUTPUT_OBJECTS",
    "binary64_record",
    "canonical_bytes",
    "canonical_sha256",
    "validate_binary64_record",
    "validate_comparison_row",
    "validate_pairwise_object_identity",
    "vdp_identity_hashes",
    "vdp_initial_set_identity",
    "vdp_partition_identity",
    "vdp_rhs_identity",
]
