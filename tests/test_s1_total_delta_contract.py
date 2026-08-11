from __future__ import annotations

from fractions import Fraction
import json

import pytest
import torch

from torch_tm_flowpipe import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    Interval,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    tmvector_hashes,
)
from torch_tm_flowpipe.fixed_support_outward import OutwardIntervalTensor
from torch_tm_flowpipe.ode_examples import van_der_pol_ode
from torch_tm_flowpipe.structured_fraction_oracle import (
    fraction_complete_polynomial_difference_oracle,
)
from torch_tm_flowpipe.structured_remainder import (
    STRUCTURED_TOTAL_DELTA_CANDIDATE,
    StructuredRemainderState,
    compare_complete_polynomial_contracts,
    physical_interval_to_normal,
)
from torch_tm_flowpipe.terminal_checkpoint import PAYLOAD_NAME, _encode_normal_state


DTYPE = torch.float64
RESET_MODE = STRUCTURED_TOTAL_DELTA_CANDIDATE


def _step(current, normal_state, *, h=0.005, attempts=2):
    return flowpipe_step_flowstar_style_adaptive(
        van_der_pol_ode,
        current,
        h=h,
        h_min=h,
        h_max=h,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        max_validation_attempts=attempts,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode=RESET_MODE,
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
        flowstar_normal_state=normal_state,
    )


def _polynomial(dim, terms, *, outputs=1):
    basis = BatchedMonomialBasis.build(dim, 4)
    coefficients = torch.zeros((1, outputs, basis.num_terms), dtype=DTYPE)
    for output, exponent, coefficient in terms:
        coefficients[0, output, basis.term_index(exponent)] = coefficient
    return BatchedPolynomial(coefficients, basis)


def _comparison(polynomial, q, ordinary, structured):
    q_interval = OutwardIntervalTensor(
        torch.tensor([q[0]], dtype=DTYPE),
        torch.tensor([q[1]], dtype=DTYPE),
    )
    ordinary_interval = OutwardIntervalTensor(
        torch.tensor([ordinary[0]], dtype=DTYPE),
        torch.tensor([ordinary[1]], dtype=DTYPE),
    )
    structured_interval = OutwardIntervalTensor(
        torch.tensor([structured[0]], dtype=DTYPE),
        torch.tensor([structured[1]], dtype=DTYPE),
    )
    current_base = q_interval.add(ordinary_interval)
    coordinate = torch.eye(len(q[0]), dtype=DTYPE).unsqueeze(0)
    result = compare_complete_polynomial_contracts(
        polynomial,
        polynomial_base_domain=(q_interval.lo, q_interval.hi),
        current_base_domain=(current_base.lo, current_base.hi),
        ordinary_box=(ordinary_interval.lo, ordinary_interval.hi),
        structured_box=(structured_interval.lo, structured_interval.hi),
        coordinate_map=coordinate,
    )
    delta = ordinary_interval.add(structured_interval)
    oracle = fraction_complete_polynomial_difference_oracle(
        polynomial.coeffs,
        polynomial.basis.exponents,
        q_interval.lo,
        q_interval.hi,
        delta.lo,
        delta.hi,
        coordinate,
        coordinate,
    )
    return result, oracle


@pytest.mark.parametrize(
    "polynomial,q,ordinary,structured",
    [
        (
            _polynomial(1, [(0, (0,), 3.0), (0, (1,), -2.0)]),
            ([-0.75], [1.25]),
            ([-0.1], [0.2]),
            ([-0.2], [0.4]),
        ),
        (
            _polynomial(1, [(0, (2,), 1.5)]),
            ([-1.0], [2.0]),
            ([-0.07], [0.11]),
            ([-0.125], [0.375]),
        ),
        (
            _polynomial(2, [(0, (2, 1), 2.0)]),
            ([-0.75, -0.5], [1.25, 0.25]),
            ([-0.04, -0.03], [0.08, 0.09]),
            ([-0.1, -0.2], [0.3, 0.4]),
        ),
        (
            _polynomial(2, [(0, (3, 1), -0.75)]),
            ([-0.5, -1.0], [1.0, 0.5]),
            ([-0.08, -0.03], [0.11, 0.07]),
            ([-0.25, -0.125], [0.375, 0.25]),
        ),
    ],
)
def test_total_delta_affine_quadratic_cubic_quartic_fixtures_match_fraction_oracle(
    polynomial, q, ordinary, structured
):
    comparison, oracle = _comparison(polynomial, q, ordinary, structured)
    result = comparison.total_delta_image
    exact = oracle.total_difference[0][0]
    assert Fraction.from_float(float(result.reconstruction_lo[0, 0])) <= exact.lo
    assert Fraction.from_float(float(result.reconstruction_hi[0, 0])) >= exact.hi
    assert result.containment_mask.tolist() == [True]


def test_two_dimensional_harmonic_zero_and_single_owner_fixtures():
    harmonic = _polynomial(
        2,
        [(0, (0, 1), 1.0), (1, (1, 0), -1.0)],
        outputs=2,
    )
    comparison, _ = _comparison(
        harmonic,
        ([-1.0, -2.0], [2.0, 3.0]),
        ([-0.1, -0.2], [0.15, 0.25]),
        ([-0.05, -0.125], [0.075, 0.1]),
    )
    total = comparison.total_delta_image
    assert torch.equal(total.nonlinear_residual_lo, torch.zeros_like(total.nonlinear_residual_lo))
    assert torch.equal(total.nonlinear_residual_hi, torch.zeros_like(total.nonlinear_residual_hi))
    assert torch.equal(total.reconstruction_lo, total.total_difference_lo)
    assert torch.equal(total.reconstruction_hi, total.total_difference_hi)

    ordinary_only, _ = _comparison(
        _polynomial(1, [(0, (2,), 1.0)]),
        ([0.5], [1.0]),
        ([-0.1], [0.2]),
        ([0.0], [0.0]),
    )
    structured_only, _ = _comparison(
        _polynomial(1, [(0, (2,), 1.0)]),
        ([0.5], [1.0]),
        ([0.0], [0.0]),
        ([-0.1], [0.2]),
    )
    assert ordinary_only.total_delta_image.containment_mask.tolist() == [True]
    assert structured_only.total_delta_image.containment_mask.tolist() == [True]


def test_cancellation_duplicate_exponents_and_zero_scale_fail_closed():
    coefficients = torch.tensor([[[1.0, -1.0, 2.0**-1022]]], dtype=DTYPE)
    exponents = torch.tensor([[2], [2], [1]], dtype=torch.long)
    base_lo = torch.tensor([[-1.0]], dtype=DTYPE)
    base_hi = torch.tensor([[1.0]], dtype=DTYPE)
    ordinary_lo = torch.tensor([[-2.0**-53]], dtype=DTYPE)
    ordinary_hi = torch.tensor([[2.0**-52]], dtype=DTYPE)
    structured_lo = torch.tensor([[-2.0**-52]], dtype=DTYPE)
    structured_hi = torch.tensor([[2.0**-51]], dtype=DTYPE)
    current_base = OutwardIntervalTensor(base_lo, base_hi).add(
        OutwardIntervalTensor(ordinary_lo, ordinary_hi)
    )
    result = compare_complete_polynomial_contracts(
        (coefficients, exponents),
        polynomial_base_domain=(base_lo, base_hi),
        current_base_domain=(current_base.lo, current_base.hi),
        ordinary_box=(ordinary_lo, ordinary_hi),
        structured_box=(structured_lo, structured_hi),
        coordinate_map=torch.ones((1, 1, 1), dtype=DTYPE),
    )
    assert result.total_delta_image.containment_mask.tolist() == [True]
    with pytest.raises(ValueError, match="zero-scale"):
        physical_interval_to_normal(
            torch.tensor([[-1.0]], dtype=DTYPE),
            torch.tensor([[1.0]], dtype=DTYPE),
            forward_scale=torch.zeros((1, 1), dtype=DTYPE),
            inverse_scale=torch.ones((1, 1), dtype=DTYPE),
        )


def test_candidate_commit_publication_multiple_columns_and_eviction():
    current = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state = None
    first_eviction = None
    for boundary in range(1, 19):
        segment = _step(current, normal_state)
        assert segment.status == "validated", segment.message
        assert segment.flowstar_normal_stats["structured_candidate"] == RESET_MODE
        assert segment.flowstar_normal_stats["structured_image_contract"] == "total_delta"
        assert segment.flowstar_normal_stats["structured_raw_picard_target_changed"] is False
        assert segment.flowstar_normal_stats["structured_conservation"]
        assert segment.flowstar_normal_stats["structured_source_decomposition"]
        assert segment.flowstar_normal_stats["structured_endpoint_publication"]
        assert segment.flowstar_normal_stats["structured_tube_publication"]
        assert segment.flowstar_normal_stats["structured_published_endpoint_in_tube"]
        assert segment.flowstar_normal_stats["structured_total_self_map_containment"]
        assert segment.boundary_attribution_record.contract == "C_total_delta"
        stage = {row.stage: row for row in segment.boundary_attribution_record.stages}
        assert torch.all(stage["B13"].lo <= stage["B0"].lo)
        assert torch.all(stage["B13"].hi >= stage["B0"].hi)
        result = segment.structured_boundary_result
        assert result.accepted.tolist() == [True]
        assert result.conservation_mask.tolist() == [True]
        assert result.source_decomposition_mask.tolist() == [True]
        if any(event.reason == "capacity_eviction" for event in result.source_events):
            first_eviction = boundary
        current = segment.reset_tm
        normal_state = segment.flowstar_normal_state
    assert isinstance(normal_state.structured_remainder_state, StructuredRemainderState)
    assert normal_state.structured_remainder_state.active.sum().item() == 16
    assert first_eviction == 18


def test_rejected_candidate_attempt_is_immutable_and_has_no_partial_fallback():
    current = [Interval(1.1, 1.4), Interval(2.35, 2.45)]
    normal_state = None
    for _ in range(2):
        accepted = _step(current, normal_state)
        assert accepted.status == "validated"
        current = accepted.reset_tm
        normal_state = accepted.flowstar_normal_state
    before_current = tmvector_hashes(current)
    before_normal = _encode_normal_state(normal_state)
    rejected = _step(current, normal_state, h=0.1, attempts=1)
    assert rejected.status == "failed"
    assert rejected.reset_tm is None
    assert tmvector_hashes(current) == before_current
    assert _encode_normal_state(normal_state) == before_normal
    assert rejected.structured_state_after is None


def test_total_delta_checkpoint_roundtrip_is_bit_exact(tmp_path):
    segment = _step(
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        None,
    )
    assert segment.status == "validated"
    contract = {"candidate": RESET_MODE, "order": 4, "dtype": "float64"}
    scheduler = {"current_time": 0.005, "accepted_segment_count": 1}
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = save_terminal_checkpoint(
        first,
        current=segment.reset_tm,
        normal_state=segment.flowstar_normal_state,
        scheduler=scheduler,
        contract=contract,
        provenance={"test": "total_delta"},
    )
    payload = json.loads((first / PAYLOAD_NAME).read_text(encoding="utf-8"))
    assert payload["normal_state"]["structured_remainder"]["candidate"] == RESET_MODE
    loaded = load_terminal_checkpoint(
        first,
        expected_contract=contract,
        expected_order=4,
        expected_dtype="float64",
    )
    second_manifest = save_terminal_checkpoint(
        second,
        current=loaded.current,
        normal_state=loaded.normal_state,
        scheduler=loaded.scheduler,
        contract=loaded.contract,
        provenance=loaded.provenance,
    )
    assert first_manifest["full_checkpoint_sha256"] == second_manifest["full_checkpoint_sha256"]
    assert (first / "terminal_state.json").read_bytes() == (second / "terminal_state.json").read_bytes()
    assert (first / "terminal_state_manifest.json").read_bytes() == (second / "terminal_state_manifest.json").read_bytes()
