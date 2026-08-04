import pytest

from torch_tm_flowpipe import Interval, flowpipe_step_flowstar_style_adaptive
from torch_tm_flowpipe.ode_examples import van_der_pol_ode


def _fixed_vdp_step(backend, h):
    diagnostics = []
    segment = flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=h,
        h_min=h,
        h_max=h,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=2,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion",
        step_policy_mode="flowstar_compat",
        tm_backend=backend,
        diagnostics=diagnostics,
    )
    return segment, diagnostics


@pytest.mark.parametrize("h", [0.005, 0.01])
def test_vdp_order4_dense_sparse_one_step_parity(h):
    sparse, sparse_diagnostics = _fixed_vdp_step("sparse", h)
    dense, dense_diagnostics = _fixed_vdp_step("dense", h)
    assert sparse.status == dense.status == "validated"
    assert sparse.tau_index == dense.tau_index == 2
    assert dense.backend_trace[0]["basis_hash"]
    assert dense.backend_trace[0]["nonzero_coefficients"] > 0
    assert len(dense.tm.domain) == 3
    assert dense.backend_counters["sparse_fallback_count"] == 0
    assert dense.backend_counters["inner_loop_conversions"] == 0

    for dense_model, sparse_model in zip(dense.tm, sparse.tm):
        exponents = set(dense_model.polynomial.terms) | set(sparse_model.polynomial.terms)
        for exponent in exponents:
            dense_value = float(dense_model.polynomial.terms.get(exponent, 0.0))
            sparse_value = float(sparse_model.polynomial.terms.get(exponent, 0.0))
            assert dense_value == pytest.approx(sparse_value, abs=2e-14, rel=2e-14)
        assert float(dense_model.remainder.lo) == pytest.approx(float(sparse_model.remainder.lo), abs=5e-14, rel=5e-14)
        assert float(dense_model.remainder.hi) == pytest.approx(float(sparse_model.remainder.hi), abs=5e-14, rel=5e-14)
    assert dense_diagnostics[-1]["validation_status"] == sparse_diagnostics[-1]["validation_status"] == "validated"

