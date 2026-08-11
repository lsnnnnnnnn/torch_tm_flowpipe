"""Tensor-native records and exact comparison utilities for S1 boundary drift."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Mapping, Sequence

import torch


BOUNDARY_STAGE_NAMES = (
    "A0",
    "A1",
    "A2",
    "A3",
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "B11",
    "B12",
    "B13",
    "B14",
    "B15",
    "B16",
)

BOUNDARY_STAGE_UNITS = {
    "old normalized",
    "new normalized",
    "physical source",
    "endpoint physical",
    "tube physical",
}


def tensor_hex(value: torch.Tensor) -> list[Any]:
    """Serialize a finite float64 tensor without losing binary64 identity."""
    tensor = value.detach().cpu()
    if tensor.dtype != torch.float64:
        raise TypeError("boundary attribution float tensors must use float64")
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError("boundary attribution tensors must be finite")
    flat = [float(item).hex() for item in tensor.reshape(-1).tolist()]

    def reshape(items: list[str], shape: Sequence[int]) -> Any:
        if not shape:
            return items[0]
        if len(shape) == 1:
            return items[: shape[0]]
        stride = math.prod(shape[1:])
        return [
            reshape(items[index * stride : (index + 1) * stride], shape[1:])
            for index in range(shape[0])
        ]

    return reshape(flat, tuple(tensor.shape))


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class S1BoundaryStage:
    stage: str
    name: str
    units: str
    lo: torch.Tensor
    hi: torch.Tensor

    def __post_init__(self) -> None:
        if self.stage not in BOUNDARY_STAGE_NAMES:
            raise ValueError(f"unknown S1 boundary stage: {self.stage}")
        if self.units not in BOUNDARY_STAGE_UNITS:
            raise ValueError(f"unknown S1 boundary units: {self.units}")
        if self.lo.shape != self.hi.shape or self.lo.dtype != torch.float64 or self.hi.dtype != torch.float64:
            raise ValueError("S1 boundary stage endpoints must be matching float64 tensors")
        if not bool(
            torch.all(torch.isfinite(self.lo))
            and torch.all(torch.isfinite(self.hi))
            and torch.all(self.lo <= self.hi)
        ):
            raise ValueError("S1 boundary stage interval is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "name": self.name,
            "units": self.units,
            "shape": list(self.lo.shape),
            "lo": self.lo.detach().cpu().tolist(),
            "hi": self.hi.detach().cpu().tolist(),
            "lo_hex": tensor_hex(self.lo),
            "hi_hex": tensor_hex(self.hi),
            "width": (self.hi - self.lo).detach().cpu().tolist(),
            "lo_sha256": tensor_sha256(self.lo),
            "hi_sha256": tensor_sha256(self.hi),
        }


@dataclass(frozen=True)
class S1BoundaryAttributionRecord:
    accepted_boundary_index_before: int
    contract: str
    stages: tuple[S1BoundaryStage, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if int(self.accepted_boundary_index_before) < 0:
            raise ValueError("boundary attribution index must be nonnegative")
        names = tuple(stage.stage for stage in self.stages)
        if names != BOUNDARY_STAGE_NAMES:
            raise ValueError(
                "boundary attribution record must contain A0..B16 exactly once and in order"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "torch_tm_flowpipe_s1_boundary_attribution_v1",
            "accepted_boundary_index_before": int(self.accepted_boundary_index_before),
            "contract": self.contract,
            "stages": [stage.as_dict() for stage in self.stages],
            "diagnostics": dict(self.diagnostics),
        }


def _ordered_binary64(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def ulp_distance(left: float, right: float) -> int:
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        raise ValueError("ULP distance requires finite binary64 scalars")
    return abs(_ordered_binary64(float(left)) - _ordered_binary64(float(right)))


def compare_binary64_scalar(left: float, right: float) -> dict[str, Any]:
    left_value = float(left)
    right_value = float(right)
    return {
        "left_hex": left_value.hex(),
        "right_hex": right_value.hex(),
        "exact_binary64_equal": left_value.hex() == right_value.hex(),
        "ulp_distance": ulp_distance(left_value, right_value),
        "absolute_difference": abs(left_value - right_value),
    }


def compare_interval(
    left_lo: float,
    left_hi: float,
    right_lo: float,
    right_hi: float,
) -> dict[str, Any]:
    values = tuple(float(value) for value in (left_lo, left_hi, right_lo, right_hi))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("interval comparison requires finite endpoints")
    if values[0] > values[1] or values[2] > values[3]:
        raise ValueError("interval comparison received an inverted interval")
    left_width = values[1] - values[0]
    right_width = values[3] - values[2]
    left_contains_right = values[0] <= values[2] and values[1] >= values[3]
    right_contains_left = values[2] <= values[0] and values[3] >= values[1]
    if left_contains_right and right_contains_left:
        direction = "equal"
    elif left_contains_right:
        direction = "left_contains_right"
    elif right_contains_left:
        direction = "right_contains_left"
    else:
        direction = "incomparable"
    return {
        "left_endpoint_difference": values[2] - values[0],
        "right_endpoint_difference": values[3] - values[1],
        "left_width": left_width,
        "right_width": right_width,
        "width_ratio_right_over_left": (
            right_width / left_width
            if left_width != 0.0
            else (1.0 if right_width == 0.0 else math.inf)
        ),
        "componentwise_containment_direction": direction,
        "left_endpoint": compare_binary64_scalar(values[0], values[2]),
        "right_endpoint": compare_binary64_scalar(values[1], values[3]),
    }


__all__ = [
    "BOUNDARY_STAGE_NAMES",
    "BOUNDARY_STAGE_UNITS",
    "S1BoundaryAttributionRecord",
    "S1BoundaryStage",
    "compare_binary64_scalar",
    "compare_interval",
    "tensor_hex",
    "tensor_sha256",
    "ulp_distance",
]
