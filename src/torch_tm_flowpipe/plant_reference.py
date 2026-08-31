"""Frozen C3+C4 polynomial-plant reference-lane configuration.

The objects in this module name already accepted solver behavior.  They do not
select among validation modes and they do not implement a new numerical path.
Keeping the configuration separate from the runner makes every production,
profile, and batch invocation bind the same machine-checkable contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from .batched_dense_tm import (
    FLOWSTAR_RAW_REMAINDER_GENERIC_REFINED_MODE,
    FLOWSTAR_RAW_REMAINDER_REFINED_MODE,
    FLOWSTAR_REFINEMENT_REPLAY_LIMIT,
    FLOWSTAR_STOP_RATIO,
)
from .flowpipe import (
    C3_CROSS_STEP_SYMBOLIC_QUEUE,
    GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
)


REFERENCE_LANE_NAME = "flowstar_like_polynomial_plant_reference"
REFERENCE_CONFIGURATION_SCHEMA = "torch_tm_flowpipe.polynomial_plant_reference/1"


@dataclass(frozen=True)
class FlowstarLikePolynomialPlantConfig:
    """One explicit plant contract inside the formal C3+C4 reference lane."""

    plant: Literal["van_der_pol", "brusselator"]
    initial_decimal_box: tuple[tuple[str, str], ...]
    order: int
    fixed_step: float | None
    requested_horizon: float
    target_remainder_radius: float
    cutoff: float
    validation_epsilon: float
    accepted_boundary_sr_capacity: int
    accepted_boundary_sr_mode: str
    post_accept_refinement_mode: str
    right_map_range_mode: str
    right_map_center_mode: str
    range_policy: tuple[tuple[str, Any], ...]
    step_policy: str
    endpoint_repair: bool = False
    accepted_boundary_sr_enabled: bool = True
    generic_post_accept_raw_remainder_refinement_enabled: bool = True
    refinement_replay_limit: int = FLOWSTAR_REFINEMENT_REPLAY_LIMIT
    stop_ratio: float = FLOWSTAR_STOP_RATIO
    subset_commit: str = "whole_vector_atomic"
    rhs_term_evaluation: str = "ordered_terms"
    authoritative_device: str = "cpu"
    authoritative_dtype: str = "float64"
    outward_rounding: str = "torch_nextafter_binary64"
    insertion_reset_semantics: str = "normal_constant_centered_accepted_only_commit"
    checkpoint_rollback_policy: str = "v5_full_queue_accept_commit_reject_immutable"
    lane_name: str = REFERENCE_LANE_NAME
    schema: str = REFERENCE_CONFIGURATION_SCHEMA

    def __post_init__(self) -> None:
        if self.plant not in {"van_der_pol", "brusselator"}:
            raise ValueError("formal reference supports only the two frozen polynomial plants")
        if not self.accepted_boundary_sr_enabled:
            raise ValueError("formal reference requires accepted-boundary SR")
        if not self.generic_post_accept_raw_remainder_refinement_enabled:
            raise ValueError("formal reference requires post-accept raw-remainder refinement")
        if self.accepted_boundary_sr_capacity <= 0:
            raise ValueError("accepted-boundary SR capacity must be positive")
        if self.refinement_replay_limit != FLOWSTAR_REFINEMENT_REPLAY_LIMIT:
            raise ValueError("formal reference replay limit is frozen at Flow*'s 491 evaluations")
        if self.stop_ratio != FLOWSTAR_STOP_RATIO:
            raise ValueError("formal reference STOP_RATIO is frozen at 0.99")
        if self.subset_commit != "whole_vector_atomic":
            raise ValueError("formal reference requires whole-vector atomic subset commits")
        if self.rhs_term_evaluation != "ordered_terms":
            raise ValueError("formal reference requires ordered-term polynomial RHS evaluation")
        if self.authoritative_device != "cpu" or self.authoritative_dtype != "float64":
            raise ValueError("formal reference is authoritative only on CPU float64")
        if self.endpoint_repair:
            raise ValueError("formal reference forbids endpoint repair")
        if not self.initial_decimal_box or any(len(pair) != 2 for pair in self.initial_decimal_box):
            raise ValueError("formal reference initial box must contain decimal endpoint pairs")
        if self.order <= 0 or self.requested_horizon <= 0.0:
            raise ValueError("formal reference order and horizon must be positive")
        if self.fixed_step is not None and self.fixed_step <= 0.0:
            raise ValueError("formal reference fixed step must be positive")
        if self.target_remainder_radius <= 0.0 or self.cutoff < 0.0:
            raise ValueError("formal reference remainder and cutoff are invalid")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64

    @property
    def range_policy_mapping(self) -> dict[str, Any]:
        return dict(self.range_policy)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["initial_decimal_box"] = [list(pair) for pair in self.initial_decimal_box]
        value["range_policy"] = self.range_policy_mapping
        value["fixed_step_hex"] = None if self.fixed_step is None else self.fixed_step.hex()
        value["requested_horizon_hex"] = self.requested_horizon.hex()
        value["target_remainder_radius_hex"] = self.target_remainder_radius.hex()
        value["cutoff_hex"] = self.cutoff.hex()
        value["validation_epsilon_hex"] = self.validation_epsilon.hex()
        value["stop_ratio_hex"] = self.stop_ratio.hex()
        return value

    @classmethod
    def van_der_pol(cls) -> "FlowstarLikePolynomialPlantConfig":
        """Frozen native C3 T=10 contract."""

        return cls(
            plant="van_der_pol",
            initial_decimal_box=(("1.1", "1.4"), ("2.35", "2.45")),
            order=4,
            fixed_step=None,
            requested_horizon=10.0,
            target_remainder_radius=1e-4,
            cutoff=1e-10,
            validation_epsilon=1e-12,
            accepted_boundary_sr_capacity=100,
            accepted_boundary_sr_mode=C3_CROSS_STEP_SYMBOLIC_QUEUE,
            post_accept_refinement_mode=FLOWSTAR_RAW_REMAINDER_REFINED_MODE,
            right_map_range_mode="standard",
            right_map_center_mode="constant",
            range_policy=(("method", "natural"), ("trigger", "always")),
            step_policy="flowstar_compat_native_h_min_0.002_h_max_0.1",
        )

    @classmethod
    def brusselator(cls) -> "FlowstarLikePolynomialPlantConfig":
        """Frozen fixed-step generic C4 T=20 contract."""

        return cls(
            plant="brusselator",
            initial_decimal_box=(("1.48", "1.52"), ("2.98", "3.02")),
            order=6,
            fixed_step=0.02,
            requested_horizon=20.0,
            target_remainder_radius=1e-4,
            cutoff=1e-10,
            validation_epsilon=1e-12,
            accepted_boundary_sr_capacity=1000,
            accepted_boundary_sr_mode=GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
            post_accept_refinement_mode=FLOWSTAR_RAW_REMAINDER_GENERIC_REFINED_MODE,
            right_map_range_mode="standard",
            right_map_center_mode="constant",
            range_policy=(
                ("method", "adaptive_subdivision"),
                ("max_depth", 1),
                ("max_leaves", 4),
                ("split_vars", (0, 1)),
                ("trigger", "proactive_depth1_on_named_contexts"),
                ("named_contexts", ("polynomial_truncation",)),
                ("variable_orders", ((0, 1, 2), (1, 0, 2), (2, 0, 1))),
            ),
            step_policy="fixed_no_retry",
        )


def formal_reference_configuration() -> dict[str, Any]:
    """Return the uniquely named two-plant reference suite."""

    return {
        "schema": REFERENCE_CONFIGURATION_SCHEMA,
        "name": REFERENCE_LANE_NAME,
        "selection_semantics": "explicit_frozen_configuration_not_a_portfolio",
        "plants": {
            "van_der_pol": FlowstarLikePolynomialPlantConfig.van_der_pol().as_dict(),
            "brusselator": FlowstarLikePolynomialPlantConfig.brusselator().as_dict(),
        },
    }


__all__ = [
    "FlowstarLikePolynomialPlantConfig",
    "REFERENCE_CONFIGURATION_SCHEMA",
    "REFERENCE_LANE_NAME",
    "formal_reference_configuration",
]
