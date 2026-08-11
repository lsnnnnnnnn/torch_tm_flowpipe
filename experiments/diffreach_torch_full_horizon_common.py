#!/usr/bin/env python3
"""Shared, dependency-light contracts for the DiffReach/Torch DR7 audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "diffreach_torch_dr7_full_horizon_trace_v1"
PARTITION_SHA256 = "e66e54f6c4fdaba04dc8547d5d2e096d7e83219ab6139210180cea98b2663faa"
SUPPORT_SHA256 = "0ae11ee9d911d45e42294df74ef2896ecb9aeb9f3d7851c09ea90e2bb2631f5e"
DIFFREACH_SOURCE_SHA = "dd628eb443b517d6415de93e7035b4baef73963e"

X_EDGE_HEX = (
    "0x1.199999999999ap+0",
    "0x1.2333333333334p+0",
    "0x1.2cccccccccccdp+0",
    "0x1.3666666666666p+0",
    "0x1.4000000000000p+0",
    "0x1.499999999999ap+0",
    "0x1.5333333333333p+0",
    "0x1.5ccccccccccccp+0",
    "0x1.6666666666666p+0",
)
Y_EDGE_HEX = (
    "0x1.2cccccccccccdp+1",
    "0x1.2e66666666667p+1",
    "0x1.3000000000000p+1",
    "0x1.319999999999ap+1",
    "0x1.3333333333334p+1",
    "0x1.34ccccccccccdp+1",
    "0x1.3666666666667p+1",
    "0x1.3800000000000p+1",
    "0x1.399999999999ap+1",
)


def binary64_record(value: float) -> dict[str, str]:
    number = float(value)
    return {"decimal": repr(number), "hex": number.hex()}


def partition_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen Torch-linspace B64 partition without importing Torch."""

    x_edges = tuple(float.fromhex(value) for value in X_EDGE_HEX)
    y_edges = tuple(float.fromhex(value) for value in Y_EDGE_HEX)
    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for x_index in range(8):
        for y_index in range(8):
            lower.append((x_edges[x_index], y_edges[y_index]))
            upper.append((x_edges[x_index + 1], y_edges[y_index + 1]))
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def partition_identity() -> dict[str, Any]:
    lower, upper = partition_arrays()
    boxes = []
    for index in range(64):
        boxes.append(
            {
                "index": index,
                "lo": [binary64_record(value) for value in lower[index]],
                "hi": [binary64_record(value) for value in upper[index]],
            }
        )
    return {
        "schema": "torch_tm_flowpipe_vdp_identity_v1",
        "object": "ordered_partition_list",
        "state_order": ["x", "y"],
        "batch": 64,
        "splits": [8, 8],
        "enumeration": "x_major_then_y",
        "boxes": boxes,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_partition() -> None:
    actual = canonical_json_sha256(partition_identity())
    if actual != PARTITION_SHA256:
        raise RuntimeError(f"frozen B64 partition hash mismatch: {actual}")


def canonical_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "f":
        array = array.astype("<f8", copy=False)
    elif array.dtype.kind in "iu":
        array = array.astype("<i8", copy=False)
    elif array.dtype.kind == "b":
        array = array.astype(np.bool_, copy=False)
    else:
        raise TypeError(f"unsupported trace dtype {array.dtype}")
    return np.ascontiguousarray(array)


def array_record(value: Any) -> dict[str, Any]:
    array = canonical_array(value)
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    digest = hashlib.sha256(header)
    digest.update(array.tobytes(order="C"))
    return {
        "sha256": digest.hexdigest(),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def records_for_fields(fields: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: array_record(value) for name, value in sorted(fields.items())}


def parse_capture_steps(value: str) -> set[int]:
    if not value.strip():
        return set()
    result = {int(item) for item in value.split(",")}
    if any(item <= 0 for item in result):
        raise ValueError("capture steps are one-based positive integers")
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl_row(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def capture_npz(path: Path, fields: Mapping[str, Any]) -> None:
    np.savez_compressed(path, **{name: canonical_array(value) for name, value in fields.items()})


validate_partition()

