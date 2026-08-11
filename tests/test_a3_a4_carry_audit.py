from __future__ import annotations

import torch

from experiments.audit_cni_composition_accounting import _audited_compose
from experiments.audit_r35_dense_cni_parity import (
    dense_to_r35_coefficients,
    r35_to_dense_coefficients,
)
from experiments.run_a3_a4_same_prestate_substitutions import SUBSTITUTIONS, _one
from experiments.run_fixed_support_descriptor_bridge import _support
from torch_tm_flowpipe.fixed_support import (
    FixedSupportSymbolicRemainderState,
    fixed_support_build_linear_tm,
    fixed_support_identity_parameterization,
)


def _initial_prestate():
    support = _support("R35")
    center = torch.tensor([[1.25, 2.4]], dtype=torch.float64)
    scale = torch.tensor([[0.15, 0.05]], dtype=torch.float64)
    model = fixed_support_build_linear_tm(center, scale, support)
    parameterization = fixed_support_identity_parameterization(
        1, 2, support, dtype=torch.float64, device="cpu"
    )
    symbolic = FixedSupportSymbolicRemainderState.initialize(
        1, 2, 1000, dtype=torch.float64, device="cpu"
    )
    return model, parameterization, symbolic


def test_r35_dense_complete_o4_basis_roundtrip_is_bit_exact():
    support = _support("R35")
    generator = torch.Generator().manual_seed(20260811)
    coefficients = torch.randn((3, 2, support.num_slots), generator=generator, dtype=torch.float64)
    dense, basis = r35_to_dense_coefficients(coefficients)
    roundtrip = dense_to_r35_coefficients(dense, basis)
    assert torch.equal(roundtrip, coefficients)
    assert set(support.exponents) == {
        tuple(int(value) for value in row)
        for row in basis.exponents.detach().cpu().tolist()
    }


def test_same_prestate_substitutions_are_read_only_and_preregistered():
    assert [label for label, _, _ in SUBSTITUTIONS] == [
        "CDR_complete_carry",
        "CNI_complete_carry",
        "CDR_reciprocal_with_epsilon",
        "CDR_reciprocal_without_epsilon",
        "CNI_reciprocal_with_epsilon",
        "CNI_reciprocal_without_epsilon",
    ]
    model, parameterization, symbolic = _initial_prestate()
    originals = tuple(
        value.clone()
        for value in (
            model.polynomial.coeffs,
            model.remainder.lo,
            model.remainder.hi,
            parameterization.polynomial.coeffs,
            parameterization.remainder.lo,
            parameterization.remainder.hi,
            symbolic.phi_buffer,
            symbolic.j_buffer.lo,
            symbolic.j_buffer.hi,
            symbolic.count,
            symbolic.inverse_scale,
        )
    )
    for label, family, epsilon in SUBSTITUTIONS:
        result = _one(
            label=label,
            family=family,
            epsilon=epsilon,
            model=model,
            parameterization=parameterization,
            symbolic=symbolic,
        )
        assert result["no_state_commit"] is True
    current = (
        model.polynomial.coeffs,
        model.remainder.lo,
        model.remainder.hi,
        parameterization.polynomial.coeffs,
        parameterization.remainder.lo,
        parameterization.remainder.hi,
        symbolic.phi_buffer,
        symbolic.j_buffer.lo,
        symbolic.j_buffer.hi,
        symbolic.count,
        symbolic.inverse_scale,
    )
    assert all(torch.equal(before, after) for before, after in zip(originals, current))


def test_cni_composition_observer_is_bit_exact_and_outer_remainder_is_added_once():
    model, parameterization, _ = _initial_prestate()
    endpoint = model.evaluate_time(0.01)
    reconstructed, sources, parity = _audited_compose(endpoint, parameterization)
    native = endpoint.compose_affine(parameterization, 0.0)
    assert parity["polynomial_bit_exact"] is True
    assert parity["remainder_lo_bit_exact"] is True
    assert parity["remainder_hi_bit_exact"] is True
    assert parity["outer_remainder_add_count"] == 2
    assert torch.equal(reconstructed.polynomial.coeffs, native.polynomial.coeffs)
    assert torch.equal(reconstructed.remainder.lo, native.remainder.lo)
    assert torch.equal(reconstructed.remainder.hi, native.remainder.hi)
    assert set(sources) == {
        "degree_gt4_dropped_polynomial",
        "polynomial_times_parameterization_remainder",
        "remainder_times_remainder",
        "outer_endpoint_remainder",
    }


def test_dense_complete_o4_has_no_native_cross_step_cni_operator():
    from torch_tm_flowpipe.batched_dense_tm import BatchedTaylorModel

    assert not any(
        name in BatchedTaylorModel.__dict__
        for name in ("compose", "compose_affine", "insert", "normalized_insertion")
    )

