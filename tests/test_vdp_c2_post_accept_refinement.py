from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import torch_tm_flowpipe.batched_dense_tm as dense_core
from torch_tm_flowpipe import (
    FlowstarNormalFlowpipeState,
    PolynomialODE,
    PolynomialODETerm,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    tmvector_hashes,
)
from torch_tm_flowpipe.batched_dense_tm import (
    FLOWSTAR_RAW_REMAINDER_REFINED_MODE,
    FLOWSTAR_REFINEMENT_REPLAY_LIMIT,
    _atomic_refinement_decision,
    _frozen_vdp_structural_fingerprint,
    _post_accept_refine_raw_remainder,
    dense_picard_validate_step,
    dense_polynomial_picard,
)
from torch_tm_flowpipe.ode_examples import van_der_pol_ode
from torch_tm_flowpipe.post_accept_refinement_oracle import (
    assert_refinement_certificate,
    verify_refinement_iteration,
)

from test_vdp_h2_dense_picard import C1, H2, LEGACY, _full_h2_step, _step1_base, _vdp


C2 = FLOWSTAR_RAW_REMAINDER_REFINED_MODE
COMMON = {
    "h": 0.01,
    "order": 4,
    "tau_index": 2,
    "target_remainder_radius": 1.0e-4,
    "cutoff_threshold": 1.0e-10,
    "max_validation_attempts": 2,
    "validation_eps": 1.0e-12,
}


@pytest.fixture(scope="module")
def step1_bundle():
    state, base = _step1_base()
    candidate, _ = dense_polynomial_picard(
        _vdp(),
        base.without_remainder(),
        tau_index=2,
        order=4,
        iterations=4,
        cutoff_threshold=1.0e-10,
    )
    c1 = dense_picard_validate_step(_vdp(), base, validation_mode=C1, **COMMON)
    c2 = dense_picard_validate_step(_vdp(), base, validation_mode=C2, **COMMON)
    return state, base, candidate, c1, c2


def _first_validation(step):
    return next(row for row in step.trace if row.get("phase") == "remainder_validation")


def _refinement_rows(step):
    return [row for row in step.trace if row.get("phase") == "post_accept_refinement"]


def _candidate_hex(candidate):
    return [
        [float(value).hex() for value in candidate.poly.coeffs[0, component].tolist()]
        for component in range(candidate.poly.out_dim)
    ]


@pytest.mark.unit
@pytest.mark.parametrize("mode", [C1, C2])
@pytest.mark.parametrize(
    "rhs",
    [
        PolynomialODE(
            (
                (PolynomialODETerm(1.0, (0, 1)),),
                (PolynomialODETerm(-1.0, (1, 0)),),
            ),
            2,
        ),
        PolynomialODE(
            (
                (PolynomialODETerm(1.0, (0, 1)),),
                (
                    PolynomialODETerm(1.0, (0, 1)),
                    PolynomialODETerm(-1.0, (1, 0)),
                    PolynomialODETerm(-1.0000000000000002, (2, 1)),
                ),
            ),
            2,
        ),
        PolynomialODE(
            (
                (PolynomialODETerm(1.0, (0, 1)),),
                (
                    PolynomialODETerm(1.0, (0, 1)),
                    PolynomialODETerm(-1.0, (1, 0)),
                    PolynomialODETerm(-1.0, (2, 1)),
                    PolynomialODETerm(1.0, (0, 0)),
                ),
            ),
            2,
        ),
        PolynomialODE(
            (
                (PolynomialODETerm(1.0, (0, 1, 0)),),
                (PolynomialODETerm(-1.0, (1, 0, 0)),),
                (PolynomialODETerm(1.0, (0, 0, 1)),),
            ),
            3,
        ),
        van_der_pol_ode,
    ],
    ids=("non-vdp-2d", "changed-coefficient", "extra-term", "different-state-dim", "custom-evaluator"),
)
def test_c1_c2_structural_fingerprint_fails_closed_before_closure(step1_bundle, mode, rhs) -> None:
    _, base, _, _, _ = step1_bundle
    with pytest.raises((TypeError, ValueError), match="VDP|structural"):
        dense_picard_validate_step(rhs, base, validation_mode=mode, **COMMON)


@pytest.mark.unit
def test_frozen_vdp_structural_fingerprint_is_binary64_exact() -> None:
    fingerprint = _frozen_vdp_structural_fingerprint(_vdp())
    assert fingerprint["schema"] == "frozen_vdp_polynomial_ode_binary64_v1"
    assert len(fingerprint["sha256"]) == 64
    assert fingerprint["components"][1][2] == {
        "coefficient_hex": "-0x1.0000000000000p+0",
        "powers": [2, 1],
    }


@pytest.mark.unit
def test_default_legacy_h1_h2_and_c1_step1_binary64_snapshots_are_unchanged(step1_bundle) -> None:
    _, base, _, _, _ = step1_bundle
    expected = {
        None: (
            ("-0x1.219eedf3654e2p-20", "-0x1.de01dbc4823d8p-17"),
            ("0x1.370b564aefd29p-20", "0x1.f1464131175b0p-17"),
        ),
        LEGACY: (
            ("-0x1.0c6faed22b5ccp-20", "-0x1.18998e125201cp-16"),
            ("0x1.0c6faed22b5ccp-20", "0x1.3d0f9ef33c30dp-16"),
        ),
        H2: (
            ("-0x1.0c6faed22b5ccp-20", "-0x1.ea5bf6c560793p-17"),
            ("0x1.0c6faed22b5ccp-20", "0x1.1d3c7c6352cf1p-16"),
        ),
        C1: (
            ("-0x1.0c6faed22b5ccp-20", "-0x1.3b30093b71782p-17"),
            ("0x1.0c6faed22b5ccp-20", "0x1.16f366d380020p-16"),
        ),
    }
    for mode, (expected_lo, expected_hi) in expected.items():
        kwargs = {} if mode is None else {"validation_mode": mode}
        step = dense_picard_validate_step(_vdp(), base, **COMMON, **kwargs)
        assert tuple(float(value).hex() for value in step.segment_tm.rem_lo[0]) == expected_lo
        assert tuple(float(value).hex() for value in step.segment_tm.rem_hi[0]) == expected_hi


@pytest.mark.unit
def test_c2_first_acceptance_is_c1_identical_then_refines_atomically(step1_bundle) -> None:
    _, _, candidate, c1, c2 = step1_bundle
    assert c1.status == c2.status == "validated"
    assert torch.equal(c1.segment_tm.poly.coeffs, c2.segment_tm.poly.coeffs)
    assert torch.equal(candidate.poly.coeffs, c2.segment_tm.poly.coeffs)
    c1_first = _first_validation(c1)
    c2_first = _first_validation(c2)
    for key in (
        "validation_status",
        "finite",
        "subset_result",
        "target_subset_result",
        "candidate_remainder_lo",
        "candidate_remainder_hi",
        "picard_image_remainder_lo",
        "picard_image_remainder_hi",
        "subset_margin",
        "raw_rhs_remainder_lo",
        "raw_rhs_remainder_hi",
        "poly_diff_range_lo",
        "poly_diff_range_hi",
    ):
        assert c2_first[key] == c1_first[key]

    rows = _refinement_rows(c2)
    assert rows and all(row["committed"] for row in rows)
    assert rows[-1]["stop_reason"] == "stop_ratio"
    assert rows[0]["input_remainder_lo"] == c1_first["picard_image_remainder_lo"]
    assert rows[0]["input_remainder_hi"] == c1_first["picard_image_remainder_hi"]
    assert len({row["retained_polynomial_sha256"] for row in rows}) == 1
    assert all(all(component["subset"] for component in row["components"]) for row in rows)
    for previous, current in zip(rows, rows[1:]):
        assert current["input_remainder_lo"] == previous["retained_remainder_lo"]
        assert current["input_remainder_hi"] == previous["retained_remainder_hi"]


@pytest.mark.unit
def test_c2_step1_causal_micro_gate_and_all_published_channels(step1_bundle) -> None:
    state, _, _, c1, c2 = step1_bundle
    c1_width = c1.segment_tm.rem_hi - c1.segment_tm.rem_lo
    c2_width = c2.segment_tm.rem_hi - c2.segment_tm.rem_lo
    assert bool(torch.all(c2_width <= c1_width))
    flowstar_first_x_raw_width = 3.30228001377617e-7
    removed = float(c1_width[0, 0] - c2_width[0, 0])
    remaining_gap = float(c1_width[0, 0]) - flowstar_first_x_raw_width
    assert removed / remaining_gap >= 0.5
    assert float(c2_width[0, 1]) <= float(c1_width[0, 1])

    c1_segment = _full_h2_step(state.normalized_initial_tm(4), state, "cpu", C1)
    c2_segment = _full_h2_step(state.normalized_initial_tm(4), state, "cpu", C2)
    assert c1_segment.status == c2_segment.status == "validated"
    assert c1_segment.endpoint_tightening_applied is c2_segment.endpoint_tightening_applied is False
    c1_boxes = (c1_segment.tm.range_box(), c1_segment.endpoint_raw_tm.range_box())
    c2_boxes = (c2_segment.tm.range_box(), c2_segment.endpoint_raw_tm.range_box())
    for c1_box, c2_box in zip(c1_boxes, c2_boxes):
        for c1_interval, c2_interval in zip(c1_box, c2_box):
            assert float(c2_interval.width()) <= float(c1_interval.width())


@pytest.mark.unit
def test_c2_x_change_is_feedback_from_verified_y_remainder(step1_bundle) -> None:
    _, _, _, c1, c2 = step1_bundle
    first = _first_validation(c1)
    row = _refinement_rows(c2)[0]
    input_y = [row["input_remainder_lo"][0][1], row["input_remainder_hi"][0][1]]
    raw_x = [row["raw_rhs_remainder_lo"][0][0], row["raw_rhs_remainder_hi"][0][0]]
    assert raw_x == input_y
    first_raw_x_width = first["raw_rhs_remainder_hi"][0][0] - first["raw_rhs_remainder_lo"][0][0]
    refined_raw_x_width = raw_x[1] - raw_x[0]
    assert refined_raw_x_width < first_raw_x_width


@pytest.mark.unit
def test_every_committed_step1_refinement_passes_independent_fraction_bernstein_oracle(step1_bundle) -> None:
    _, base, candidate, _, c2 = step1_bundle
    domain = [
        [float(base.domain_lo[0, index]), float(base.domain_hi[0, index])]
        for index in range(base.n_vars)
    ]
    base_remainder = [
        [float(base.rem_lo[0, component]), float(base.rem_hi[0, component])]
        for component in range(base.poly.out_dim)
    ]
    for row in _refinement_rows(c2):
        if not row["committed"]:
            continue
        certificate = verify_refinement_iteration(
            row,
            candidate_coefficient_hex=_candidate_hex(candidate),
            candidate_exponents=candidate.poly.basis.exponents.tolist(),
            domain=domain,
            base_remainder=base_remainder,
            tau_interval=[0.0, 0.01],
            validation_eps=1.0e-12,
        )
        assert_refinement_certificate(certificate)
        assert certificate.all_contained


@pytest.mark.unit
def test_atomic_refinement_rejects_mixed_component_subset_without_partial_update(
    monkeypatch, step1_bundle
) -> None:
    _, base, candidate, c1, _ = step1_bundle
    retained_lo = torch.full_like(candidate.rem_lo, -1.0)
    retained_hi = torch.full_like(candidate.rem_hi, 1.0)

    def mixed_image(*args, **kwargs):
        del args, kwargs
        proposed_lo = torch.tensor([[-0.5, -2.0]], dtype=torch.float64)
        proposed_hi = torch.tensor([[0.5, 2.0]], dtype=torch.float64)
        return proposed_lo, proposed_hi, {}, c1.validated_remainder_decomposition

    monkeypatch.setattr(dense_core, "_dense_flowstar_raw_compat_image", mixed_image)
    final_lo, final_hi, _, rows = _post_accept_refine_raw_remainder(
        _vdp(),
        base,
        candidate,
        retained_lo=retained_lo,
        retained_hi=retained_hi,
        retained_decomposition=c1.validated_remainder_decomposition,
        tau_index=2,
        order=4,
        cutoff_threshold=1.0e-10,
        validation_eps=1.0e-12,
        structural_fingerprint=_frozen_vdp_structural_fingerprint(_vdp()),
        replay_limit=1,
    )
    assert rows[0]["stop_reason"] == "component_subset_failure"
    assert [component["subset"] for component in rows[0]["components"]] == [True, False]
    assert torch.equal(final_lo, retained_lo)
    assert torch.equal(final_hi, retained_hi)


@pytest.mark.unit
def test_refinement_fixed_point_ratio_boundaries_zero_width_subnormal_and_nonfinite() -> None:
    zero = torch.tensor([[0.0]], dtype=torch.float64)
    one = torch.tensor([[1.0]], dtype=torch.float64)
    at = torch.tensor([[0.99]], dtype=torch.float64)
    above = torch.tensor([[math.nextafter(0.99, math.inf)]], dtype=torch.float64)
    assert _atomic_refinement_decision(zero, one, zero, at)[:3] == (True, True, "continue")
    assert _atomic_refinement_decision(zero, one, zero, above)[:3] == (True, False, "stop_ratio")

    fixed = _atomic_refinement_decision(zero, zero, zero, zero)
    assert fixed[:3] == (True, False, "fixed_point")
    assert math.isnan(float(fixed[-1]))

    tiny = math.ulp(0.0)
    old_hi = torch.tensor([[4.0 * tiny]], dtype=torch.float64)
    new_lo = torch.tensor([[tiny]], dtype=torch.float64)
    new_hi = torch.tensor([[2.0 * tiny]], dtype=torch.float64)
    assert _atomic_refinement_decision(zero, old_hi, new_lo, new_hi)[0] is True

    nonfinite = _atomic_refinement_decision(
        -one,
        one,
        torch.tensor([[float("nan")]], dtype=torch.float64),
        one,
    )
    assert nonfinite[:3] == (False, False, "nonfinite_proposal")


@pytest.mark.unit
def test_refinement_replay_counts_zero_one_and_flowstar_upper_bound(monkeypatch, step1_bundle) -> None:
    _, base, candidate, c1, _ = step1_bundle

    def contracting_image(rhs_fn, base_ext, candidate_with_remainder, candidate_poly, **kwargs):
        del rhs_fn, base_ext, candidate_poly, kwargs
        return (
            candidate_with_remainder.rem_lo * 0.5,
            candidate_with_remainder.rem_hi * 0.5,
            {},
            c1.validated_remainder_decomposition,
        )

    monkeypatch.setattr(dense_core, "_dense_flowstar_raw_compat_image", contracting_image)
    for replay_limit in (0, 1, FLOWSTAR_REFINEMENT_REPLAY_LIMIT):
        _, _, _, rows = _post_accept_refine_raw_remainder(
            _vdp(),
            base,
            candidate,
            retained_lo=torch.full_like(candidate.rem_lo, -1.0),
            retained_hi=torch.full_like(candidate.rem_hi, 1.0),
            retained_decomposition=c1.validated_remainder_decomposition,
            tau_index=2,
            order=4,
            cutoff_threshold=1.0e-10,
            validation_eps=1.0e-12,
            structural_fingerprint=_frozen_vdp_structural_fingerprint(_vdp()),
            replay_limit=replay_limit,
        )
        expected_rows = 1 if replay_limit == 0 else replay_limit
        assert len(rows) == expected_rows
        assert rows[-1]["stop_reason"] == (
            "configured_zero_replays" if replay_limit == 0 else "max_refinement_replays_reached"
        )


@pytest.mark.unit
def test_one_narrowing_then_fixed_point_stops(monkeypatch, step1_bundle) -> None:
    _, base, candidate, c1, _ = step1_bundle
    calls = 0

    def narrowing_then_fixed(rhs_fn, base_ext, candidate_with_remainder, candidate_poly, **kwargs):
        nonlocal calls
        del rhs_fn, base_ext, candidate_poly, kwargs
        calls += 1
        factor = 0.5 if calls == 1 else 1.0
        return (
            candidate_with_remainder.rem_lo * factor,
            candidate_with_remainder.rem_hi * factor,
            {},
            c1.validated_remainder_decomposition,
        )

    monkeypatch.setattr(dense_core, "_dense_flowstar_raw_compat_image", narrowing_then_fixed)
    _, _, _, rows = _post_accept_refine_raw_remainder(
        _vdp(),
        base,
        candidate,
        retained_lo=torch.full_like(candidate.rem_lo, -1.0),
        retained_hi=torch.full_like(candidate.rem_hi, 1.0),
        retained_decomposition=c1.validated_remainder_decomposition,
        tau_index=2,
        order=4,
        cutoff_threshold=1.0e-10,
        validation_eps=1.0e-12,
        structural_fingerprint=_frozen_vdp_structural_fingerprint(_vdp()),
        replay_limit=10,
    )
    assert [row["stop_reason"] for row in rows] == ["continue", "fixed_point"]
    assert all(row["committed"] for row in rows)


@pytest.mark.unit
def test_failed_first_self_map_is_c1_identical_and_never_refined(step1_bundle) -> None:
    _, base, _, _, _ = step1_bundle
    common = {**COMMON, "target_remainder_radius": 5.0e-6}
    c1 = dense_picard_validate_step(_vdp(), base, validation_mode=C1, **common)
    c2 = dense_picard_validate_step(_vdp(), base, validation_mode=C2, **common)
    assert c1.status == c2.status == "failed"
    assert c1.message == c2.message
    assert c1.validation_attempts == c2.validation_attempts
    assert torch.equal(c1.segment_tm.poly.coeffs, c2.segment_tm.poly.coeffs)
    assert torch.equal(c1.segment_tm.rem_lo, c2.segment_tm.rem_lo)
    assert torch.equal(c1.segment_tm.rem_hi, c2.segment_tm.rem_hi)
    assert not _refinement_rows(c2)


@pytest.mark.unit
def test_final_remainder_decomposition_is_from_last_committed_iteration(step1_bundle) -> None:
    _, _, _, _, c2 = step1_bundle
    last = [row for row in _refinement_rows(c2) if row["committed"]][-1]
    assert last["retained_remainder_lo"] == c2.validated_remainder_lo.detach().cpu().tolist()
    assert last["retained_remainder_hi"] == c2.validated_remainder_hi.detach().cpu().tolist()
    assert (
        last["validated_remainder_ledger_intervals"]
        == c2.validated_remainder_decomposition.ledger.intervals()
    )
    assert last["validated_remainder_decomposition_lo"] == (
        c2.validated_remainder_decomposition.decomposition_lo.detach().cpu().tolist()
    )


@pytest.mark.unit
def test_c2_cpu_replay_is_bitwise_deterministic(step1_bundle) -> None:
    _, base, _, _, c2 = step1_bundle
    replay = dense_picard_validate_step(_vdp(), base, validation_mode=C2, **COMMON)
    assert replay.status == c2.status
    assert torch.equal(replay.segment_tm.poly.coeffs, c2.segment_tm.poly.coeffs)
    assert torch.equal(replay.segment_tm.rem_lo, c2.segment_tm.rem_lo)
    assert torch.equal(replay.segment_tm.rem_hi, c2.segment_tm.rem_hi)
    assert replay.trace == c2.trace


@pytest.mark.unit
def test_c2_checkpoint_resume_is_bitwise_on_cpu(tmp_path: Path) -> None:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("11/10", "7/5"), ("47/20", "49/20")], 4
    )
    first = _full_h2_step(state.normalized_initial_tm(4), state, "cpu", C2)
    assert first.status == "validated"
    assert first.reset_tm is not None and first.flowstar_normal_state is not None
    contract = {"validation_mode": C2, "reset_mode": "normalized_insertion_dependency_preserving"}
    checkpoint = tmp_path / "c2-cpu"
    save_terminal_checkpoint(
        checkpoint,
        current=first.reset_tm,
        normal_state=first.flowstar_normal_state,
        scheduler={"current_time": 0.01, "h_next": 0.01},
        contract=contract,
        provenance={"test": "c2-checkpoint-resume", "device": "cpu"},
    )
    uninterrupted = _full_h2_step(first.reset_tm, first.flowstar_normal_state, "cpu", C2)
    loaded = load_terminal_checkpoint(
        checkpoint,
        expected_contract=contract,
        expected_order=4,
        expected_dtype="float64",
    )
    resumed = _full_h2_step(loaded.current, loaded.normal_state, "cpu", C2)
    assert uninterrupted.status == resumed.status == "validated"
    assert uninterrupted.reset_tm is not None and resumed.reset_tm is not None
    assert tmvector_hashes(uninterrupted.reset_tm) == tmvector_hashes(resumed.reset_tm)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_c2_v100_implementation_consistency_only() -> None:
    _, cpu_base = _step1_base("cpu")
    _, cuda_base = _step1_base("cuda")
    cpu = dense_picard_validate_step(_vdp(), cpu_base, validation_mode=C2, **COMMON)
    cuda = dense_picard_validate_step(_vdp(), cuda_base, validation_mode=C2, **COMMON)
    assert cpu.status == cuda.status == "validated"
    assert len(_refinement_rows(cpu)) == len(_refinement_rows(cuda))
    assert torch.allclose(cpu.segment_tm.rem_lo, cuda.segment_tm.rem_lo.cpu(), rtol=1.0e-12, atol=1.0e-15)
    assert torch.allclose(cpu.segment_tm.rem_hi, cuda.segment_tm.rem_hi.cpu(), rtol=1.0e-12, atol=1.0e-15)
