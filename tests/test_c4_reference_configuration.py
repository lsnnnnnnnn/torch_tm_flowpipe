from __future__ import annotations

import pytest
import torch

from torch_tm_flowpipe import (
    C3_CROSS_STEP_SYMBOLIC_QUEUE,
    FLOWSTAR_RAW_REMAINDER_GENERIC_REFINED_MODE,
    FLOWSTAR_RAW_REMAINDER_REFINED_MODE,
    GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER,
    FlowstarLikePolynomialPlantConfig,
    REFERENCE_LANE_NAME,
    formal_reference_configuration,
)


@pytest.mark.unit
def test_formal_reference_configuration_freezes_c3_and_c4_contracts() -> None:
    vdp = FlowstarLikePolynomialPlantConfig.van_der_pol()
    brusselator = FlowstarLikePolynomialPlantConfig.brusselator()

    assert vdp.lane_name == brusselator.lane_name == REFERENCE_LANE_NAME
    assert vdp.accepted_boundary_sr_mode == C3_CROSS_STEP_SYMBOLIC_QUEUE
    assert vdp.post_accept_refinement_mode == FLOWSTAR_RAW_REMAINDER_REFINED_MODE
    assert vdp.accepted_boundary_sr_capacity == 100
    assert vdp.range_policy_mapping == {
        "method": "adaptive_subdivision",
        "max_depth": 1,
        "max_leaves": 4,
        "split_vars": (0, 1),
        "trigger": "proactive_depth1_on_named_contexts",
        "named_contexts": ("polynomial_truncation",),
        "variable_orders": ((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    }
    assert brusselator.accepted_boundary_sr_mode == GENERIC_ACCEPTED_BOUNDARY_SYMBOLIC_REMAINDER
    assert brusselator.post_accept_refinement_mode == FLOWSTAR_RAW_REMAINDER_GENERIC_REFINED_MODE
    assert brusselator.accepted_boundary_sr_capacity == 1000
    assert brusselator.fixed_step == 0.02
    assert brusselator.order == 6
    assert brusselator.torch_dtype == torch.float64
    assert brusselator.refinement_replay_limit == 491
    assert brusselator.stop_ratio == 0.99
    assert brusselator.subset_commit == "whole_vector_atomic"
    assert brusselator.rhs_term_evaluation == "ordered_terms"
    assert brusselator.endpoint_repair is False

    suite = formal_reference_configuration()
    assert suite["name"] == REFERENCE_LANE_NAME
    assert suite["selection_semantics"] == "explicit_frozen_configuration_not_a_portfolio"
    assert set(suite["plants"]) == {"van_der_pol", "brusselator"}
    assert suite["plants"]["brusselator"]["fixed_step_hex"] == (0.02).hex()


@pytest.mark.unit
def test_formal_reference_configuration_fails_closed_on_semantic_drift() -> None:
    base = FlowstarLikePolynomialPlantConfig.brusselator()
    values = dict(base.__dict__)
    values["authoritative_dtype"] = "float32"
    with pytest.raises(ValueError, match="CPU float64"):
        FlowstarLikePolynomialPlantConfig(**values)

    values = dict(base.__dict__)
    values["refinement_replay_limit"] = 490
    with pytest.raises(ValueError, match="491"):
        FlowstarLikePolynomialPlantConfig(**values)
