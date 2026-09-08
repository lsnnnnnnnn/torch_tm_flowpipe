"""The unchanged two-system numerical configurations used for revalidation."""
from pathlib import Path
import hashlib
import json

import torch
from torch_tm_flowpipe import (
    DENSE_OBSERVER_NONE, DenseRangePolicy, FlowstarLikePolynomialPlantConfig,
    FlowstarNormalFlowpipeState, PolynomialODE, flowpipe_step_flowstar_style_adaptive,
)
from experiments.run_vdp_dense_backend import load_contract
from experiments.run_brusselator_sr1000_parity import _step, _policy

ROOT = Path(__file__).resolve().parents[2]
MATCHED = ROOT/'artifacts/runs/xiangru_adoption_20260907T032448Z/MATCHED_CONTRACTS.json'
MATCHED_SHA256 = 'a23390b6f84f99faf3fc61de6711a8607acf8358dfc30663d6baac1477352025'


def setup(plant):
    assert hashlib.sha256(MATCHED.read_bytes()).hexdigest() == MATCHED_SHA256
    config = (FlowstarLikePolynomialPlantConfig.brusselator() if plant == 'brusselator'
              else FlowstarLikePolynomialPlantConfig.van_der_pol())
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(config.initial_decimal_box, config.order)
    return config, state.normalized_initial_tm(config.order), state


def step(plant, current, state, index, *, h=None, adaptive=False):
    config = (FlowstarLikePolynomialPlantConfig.brusselator() if plant == 'brusselator'
              else FlowstarLikePolynomialPlantConfig.van_der_pol())
    if plant == 'brusselator':
        assert not adaptive
        return _step(current, state, index, _policy(), validation_mode=config.post_accept_refinement_mode,
                     lane_label='endpoint_roundoff_repair', observer_mode=DENSE_OBSERVER_NONE)[0]
    ode = PolynomialODE.from_system_spec(load_contract()['canonical_system_spec'])
    return flowpipe_step_flowstar_style_adaptive(
        ode, current, h=h if h is not None else .01,
        h_min=.002 if adaptive else .01, h_max=.1 if adaptive else .01,
        order=config.order, target_remainder_radius=config.target_remainder_radius,
        cutoff_threshold=config.cutoff, max_validation_attempts=2,
        validation_eps=config.validation_epsilon, validation_mode=config.post_accept_refinement_mode,
        reset_mode=config.accepted_boundary_sr_mode, step_policy_mode='flowstar_compat',
        flowstar_normal_state=state, flowstar_symbolic_queue_max_size=config.accepted_boundary_sr_capacity,
        right_map_center_mode=config.right_map_center_mode, right_map_range_mode=config.right_map_range_mode,
        tm_backend='dense', dense_device='cpu', dense_dtype=torch.float64,
        dense_range_policy=DenseRangePolicy(**config.range_policy_mapping),
        dense_observer_mode=DENSE_OBSERVER_NONE,
    )
