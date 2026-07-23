from __future__ import annotations

import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

EXPERIMENT = Path(__file__).resolve().parents[1]
DIFFREACH_ROOT = Path("/srv/local/shengenli/DiffReach")
for path in (EXPERIMENT, DIFFREACH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import src.settings as settings
from common import load_spec
from run_diffreach import DiffReachPlantCore, _rhs, build_linear_tm, diagnose_support


@pytest.mark.parametrize("truncate", [True, False])
def test_diffreach_support_is_measured_across_integration(truncate):
    spec = load_spec(EXPERIMENT / "benchmark_spec.yaml")
    system = spec["systems"]["riccati"]
    old = dict(settings.CONFIG)
    try:
        settings.update_config(
            {
                "TRUNCATE_TO_AFFINE": truncate,
                "FP64_IN_CROWN": True,
                "BOUND_TIME_STEP": True,
                "DEBUG_LOG": False,
            }
        )
        core = DiffReachPlantCore(
            _rhs(system),
            dimension=1,
            h=0.01,
            init_remainder=0.1,
            frr_rounds=2,
            frr_stop_ratio=0.95,
            symbolic_window=10,
        )
        lower = jnp.asarray([[0.0]], dtype=jnp.float64)
        upper = jnp.asarray([[0.1]], dtype=jnp.float64)
        result = core.verify(lower, upper, 1)
        final_tm = result[5]
        support = diagnose_support(core, system, lower, upper, final_tm)
    finally:
        settings.CONFIG.clear()
        settings.CONFIG.update(old)
    assert support["after_dynamics_evaluation"]["nonzero_L_support"]
    assert support["after_time_integration"]["nonzero_Lt"] is True
    assert support["after_time_integration"]["effective_max_degree"] == 2
    if truncate:
        assert support["final_flowpipe_segment"]["nonzero_Lt"] is False
    else:
        assert support["final_flowpipe_segment"]["nonzero_Lt"] is True
