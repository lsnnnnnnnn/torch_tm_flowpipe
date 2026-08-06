"""Native Torch/auto_LiRPA controller bounds for the frozen TORA model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .tora_q3 import ToraQ3AffineBoundary


STATE_DIMENSIONS = 4
NORMALIZED_VARIABLES = 6
AUGMENTED_VARIABLES = NORMALIZED_VARIABLES + STATE_DIMENSIONS
EXPECTED_ORIGINAL_CONTROLLER_SHA256 = "52a50c6bc6b1b45b89319edc809cd4d3baca06c32f9fd2ddceb2e95007414418"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeToraController(torch.nn.Module):
    """Exact flat Linear/ReLU form of the fixed Conv ONNX graph."""

    def __init__(self, model_path: Path):
        super().__init__()
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exception:  # pragma: no cover - external suite
            raise RuntimeError("native TORA controller requires the optional onnx package") from exception
        model = onnx.load(model_path)
        initializers = {
            value.name: np.array(numpy_helper.to_array(value), copy=True)
            for value in model.graph.initializer
        }
        required = {
            "input_Mean",
            *(f"Operation_{index}_{suffix}" for index in range(1, 5) for suffix in ("W", "B")),
        }
        if not required.issubset(initializers):
            raise ValueError("controller ONNX initializer contract mismatch")
        self.register_buffer(
            "input_mean",
            torch.as_tensor(initializers["input_Mean"].reshape(4), dtype=torch.float64),
        )
        layers = []
        for index in range(1, 5):
            weight = initializers[f"Operation_{index}_W"].reshape(
                initializers[f"Operation_{index}_W"].shape[0], -1
            )
            bias = initializers[f"Operation_{index}_B"].reshape(-1)
            layer = torch.nn.Linear(
                weight.shape[1], weight.shape[0], bias=True, dtype=torch.float64
            )
            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(weight, dtype=torch.float64))
                layer.bias.copy_(torch.as_tensor(bias, dtype=torch.float64))
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = state - self.input_mean
        for layer in self.layers:
            value = torch.relu(layer(value))
        return value.reshape(-1, 1)


class NormalizedAffineController(torch.nn.Module):
    """Keep the leaf-specific normalized affine state map in the bound graph."""

    def __init__(self, controller: torch.nn.Module):
        super().__init__()
        self.controller = controller

    def forward(
        self,
        normalized_and_remainder: torch.Tensor,
        affine_weight: torch.Tensor,
        affine_coeff: torch.Tensor,
    ) -> torch.Tensor:
        normalized = normalized_and_remainder[:, :NORMALIZED_VARIABLES]
        remainder = normalized_and_remainder[:, NORMALIZED_VARIABLES:]
        state = (
            affine_weight.matmul(normalized.unsqueeze(-1)).squeeze(-1)
            + affine_coeff
            + remainder
        )
        return self.controller(state).reshape(-1, 1)


@dataclass(frozen=True)
class ToraControllerBound:
    controlled_boundary: ToraQ3AffineBoundary
    lower_slope: torch.Tensor
    upper_slope: torch.Tensor
    raw_lower_bias: torch.Tensor
    raw_upper_bias: torch.Tensor
    output_lower_before_outward: torch.Tensor
    output_upper_before_outward: torch.Tensor
    output_lower_after_outward: torch.Tensor
    output_upper_after_outward: torch.Tensor
    maximum_slope_gap: float
    timing: dict[str, float]


def _normalized_inputs(boundary: ToraQ3AffineBoundary) -> tuple[torch.Tensor, ...]:
    batch = boundary.center.shape[0]
    device = boundary.center.device
    dtype = boundary.center.dtype
    zeros = torch.zeros((batch, 1), device=device, dtype=dtype)
    ones = torch.ones((batch, NORMALIZED_VARIABLES - 1), device=device, dtype=dtype)
    input_lo = torch.cat(
        (zeros, -ones, boundary.remainder_lower[:, :STATE_DIMENSIONS]), dim=1
    )
    input_hi = torch.cat(
        (zeros, ones, boundary.remainder_upper[:, :STATE_DIMENSIONS]), dim=1
    )
    affine_weight = torch.cat(
        (
            torch.zeros((batch, STATE_DIMENSIONS, 1), device=device, dtype=dtype),
            boundary.linear[:, :STATE_DIMENSIONS, :],
        ),
        dim=2,
    )
    affine_coeff = boundary.center[:, :STATE_DIMENSIONS]
    return input_lo, input_hi, affine_weight, affine_coeff


def _next_down(value: np.ndarray) -> np.ndarray:
    return np.nextafter(value, -np.inf)


def _next_up(value: np.ndarray) -> np.ndarray:
    return np.nextafter(value, np.inf)


class ToraAutoLirpaControllerBounder:
    """Persistent same-slope CROWN bounder with host outward composition."""

    def __init__(
        self,
        model_path: Path,
        example_boundary: ToraQ3AffineBoundary,
        *,
        device: torch.device | str = "cuda",
        expected_sha256: str = EXPECTED_ORIGINAL_CONTROLLER_SHA256,
        slope_tolerance: float = 1e-12,
    ):
        try:
            from auto_LiRPA import BoundedModule
        except ImportError as exception:  # pragma: no cover - external suite
            raise RuntimeError("native TORA controller requires auto_LiRPA") from exception
        if torch.get_default_dtype() != torch.float64:
            raise RuntimeError("formal auto_LiRPA controller construction requires torch float64 default dtype")
        self.device = torch.device(device)
        self.model_path = Path(model_path).resolve()
        observed = file_sha256(self.model_path)
        if observed != expected_sha256:
            raise ValueError(
                f"controller SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
            )
        self.controller_sha256 = observed
        self.slope_tolerance = float(slope_tolerance)
        input_lo, input_hi, affine_weight, affine_coeff = _normalized_inputs(
            example_boundary
        )
        if input_lo.shape[0] != 48:
            raise ValueError("persistent formal controller graph requires B48")
        center = 0.5 * (input_lo + input_hi)
        started = time.perf_counter()
        controller = NativeToraController(self.model_path).to(
            device=self.device, dtype=torch.float64
        )
        composed = NormalizedAffineController(controller).to(
            device=self.device, dtype=torch.float64
        )
        self.bounded = BoundedModule(
            composed,
            (
                center.to(self.device),
                affine_weight.to(self.device),
                affine_coeff.to(self.device),
            ),
            device=self.device,
            bound_opts={
                "activation_bound_option": "same-slope",
                "conv_mode": "matrix",
            },
        )
        self.required: dict[str, set[str]] = defaultdict(set)
        self.required[self.bounded.output_name[0]].add(
            self.bounded.input_name[0]
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.build_seconds = time.perf_counter() - started
        self.nominal_controller = controller

    def nominal(self, state: torch.Tensor) -> torch.Tensor:
        return self.nominal_controller(
            state.to(device=self.device, dtype=torch.float64)
        )

    def bound(self, boundary: ToraQ3AffineBoundary) -> ToraControllerBound:
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm

        input_lo, input_hi, affine_weight, affine_coeff = _normalized_inputs(
            boundary
        )
        input_lo = input_lo.to(self.device)
        input_hi = input_hi.to(self.device)
        affine_weight = affine_weight.to(self.device)
        affine_coeff = affine_coeff.to(self.device)
        center = 0.5 * (input_lo + input_hi)
        bounded_input = BoundedTensor(
            center, PerturbationLpNorm(x_L=input_lo, x_U=input_hi)
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        bound_started = time.perf_counter()
        lower, upper, matrices = self.bounded.compute_bounds(
            x=(bounded_input, affine_weight, affine_coeff),
            method="CROWN",
            return_A=True,
            needed_A_dict=self.required,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        bound_seconds = time.perf_counter() - bound_started
        affine = matrices[self.bounded.output_name[0]][
            self.bounded.input_name[0]
        ]
        batch = boundary.center.shape[0]
        lower_slope = (
            affine["lA"]
            .detach()
            .cpu()
            .reshape(batch, 1, AUGMENTED_VARIABLES)
            .numpy()
        )
        upper_slope = (
            affine["uA"]
            .detach()
            .cpu()
            .reshape(batch, 1, AUGMENTED_VARIABLES)
            .numpy()
        )
        raw_lower = affine["lbias"].detach().cpu().reshape(batch, 1).numpy()
        raw_upper = affine["ubias"].detach().cpu().reshape(batch, 1).numpy()
        output_lower = lower.detach().cpu().reshape(batch, 1).numpy()
        output_upper = upper.detach().cpu().reshape(batch, 1).numpy()
        slope_gap = float(
            np.max(np.abs(lower_slope - upper_slope), initial=0.0)
        )
        if not np.isfinite(slope_gap) or slope_gap > self.slope_tolerance:
            raise RuntimeError(f"controller is not same-slope: gap={slope_gap}")

        composition_started = time.perf_counter()
        remainder_slope = lower_slope[:, :, -STATE_DIMENSIONS:]
        rem_lo = (
            boundary.remainder_lower[:, :STATE_DIMENSIONS]
            .detach()
            .cpu()
            .numpy()
        )
        rem_hi = (
            boundary.remainder_upper[:, :STATE_DIMENSIONS]
            .detach()
            .cpu()
            .numpy()
        )
        positive = np.maximum(remainder_slope, 0.0)
        negative = np.minimum(remainder_slope, 0.0)
        propagated_lo = (
            positive @ rem_lo[..., None] + negative @ rem_hi[..., None]
        ).squeeze(-1)
        propagated_hi = (
            positive @ rem_hi[..., None] + negative @ rem_lo[..., None]
        ).squeeze(-1)
        combined_lower = raw_lower + propagated_lo
        combined_upper = raw_upper + propagated_hi
        action_center = 0.5 * (combined_lower + combined_upper)

        outward_lower = np.zeros_like(raw_lower)
        outward_upper = np.zeros_like(raw_upper)
        for state in range(STATE_DIMENSIONS):
            coefficient = remainder_slope[:, :, state]
            selected_lo = np.where(
                coefficient >= 0.0,
                rem_lo[:, state, None],
                rem_hi[:, state, None],
            )
            selected_hi = np.where(
                coefficient >= 0.0,
                rem_hi[:, state, None],
                rem_lo[:, state, None],
            )
            outward_lower = _next_down(
                outward_lower + _next_down(coefficient * selected_lo)
            )
            outward_upper = _next_up(
                outward_upper + _next_up(coefficient * selected_hi)
            )
        outward_lower = _next_down(raw_lower + outward_lower)
        outward_upper = _next_up(raw_upper + outward_upper)
        controlled_center = boundary.center.detach().cpu().numpy().copy()
        controlled_linear = boundary.linear.detach().cpu().numpy().copy()
        controlled_rem_lo = boundary.remainder_lower.detach().cpu().numpy().copy()
        controlled_rem_hi = boundary.remainder_upper.detach().cpu().numpy().copy()
        controlled_center[:, 4:5] = action_center
        controlled_linear[:, 4:5, :] = lower_slope[:, :, 1:6]
        controlled_rem_lo[:, 4:5] = _next_down(outward_lower - action_center)
        controlled_rem_hi[:, 4:5] = _next_up(outward_upper - action_center)
        magnitude = np.abs(controlled_linear[:, 4:5, :])
        first_pair = _next_up(magnitude[:, :, 0] + magnitude[:, :, 1])
        second_pair = _next_up(magnitude[:, :, 2] + magnitude[:, :, 3])
        spatial = _next_up(
            _next_up(first_pair + second_pair) + magnitude[:, :, 4]
        )
        after_lower = _next_down(
            _next_down(action_center - spatial) + controlled_rem_lo[:, 4:5]
        )
        after_upper = _next_up(
            _next_up(action_center + spatial) + controlled_rem_hi[:, 4:5]
        )
        composition_seconds = time.perf_counter() - composition_started
        controlled = ToraQ3AffineBoundary(
            torch.as_tensor(
                controlled_center, dtype=torch.float64, device=self.device
            ),
            torch.as_tensor(
                controlled_linear, dtype=torch.float64, device=self.device
            ),
            torch.as_tensor(
                controlled_rem_lo, dtype=torch.float64, device=self.device
            ),
            torch.as_tensor(
                controlled_rem_hi, dtype=torch.float64, device=self.device
            ),
        )
        return ToraControllerBound(
            controlled,
            torch.as_tensor(lower_slope),
            torch.as_tensor(upper_slope),
            torch.as_tensor(raw_lower),
            torch.as_tensor(raw_upper),
            torch.as_tensor(output_lower),
            torch.as_tensor(output_upper),
            torch.as_tensor(after_lower),
            torch.as_tensor(after_upper),
            slope_gap,
            {
                "bound_seconds": bound_seconds,
                "composition_seconds": composition_seconds,
            },
        )


__all__ = [
    "EXPECTED_ORIGINAL_CONTROLLER_SHA256",
    "NativeToraController",
    "NormalizedAffineController",
    "ToraAutoLirpaControllerBounder",
    "ToraControllerBound",
    "file_sha256",
]
