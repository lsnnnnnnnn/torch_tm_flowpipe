from __future__ import annotations

from fractions import Fraction
import hashlib

import pytest
import torch

from torch_tm_flowpipe import (
    FlowstarNormalFlowpipeState,
    Interval,
    PolynomialODE,
    TaylorModel,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
)
from torch_tm_flowpipe.batched_dense_tm import (
    dense_picard_validate_step,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.polynomial import Polynomial
from torch_tm_flowpipe.source_ledger import (
    BoundedSourceLedgerState,
    accepted_successor,
    affine_lift_interval,
    collapse_source_polynomial,
    commit_or_preserve,
    metadata_tamper,
    source_payload_hash,
)


UNIT = Interval(-1.0, 1.0)


def _fraction(value: torch.Tensor) -> Fraction:
    return Fraction.from_float(float(value.detach().cpu()))


def _poly_digest(tmv: TMVector) -> str:
    rows = []
    for model in tmv:
        rows.append(
            [(exp, float(coef.detach().cpu()).hex()) for exp, coef in model.polynomial.terms.items()]
        )
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def test_affine_lift_exact_fraction_containment_asymmetric() -> None:
    lo = torch.tensor([[-0.3, 0.1]], dtype=torch.float64)
    hi = torch.tensor([[0.7, 0.10000000000000003]], dtype=torch.float64)
    witness = affine_lift_interval(lo, hi)
    for index in range(2):
        midpoint = _fraction(witness.midpoint[0, index])
        radius = _fraction(witness.radius[0, index])
        assert midpoint - radius <= _fraction(lo[0, index])
        assert midpoint + radius >= _fraction(hi[0, index])
    assert witness.contains_input.tolist() == [[True, True]]


def test_affine_source_survives_two_linear_consumers_exactly() -> None:
    domain = [UNIT]
    z = Polynomial.variable(0, 1)
    x = Polynomial.constant(1.0, 1) + z * 2.0
    after_one = x * 3.0
    after_two = after_one * -0.5
    assert float(after_two.terms[(1,)].detach().cpu()) == -3.0
    model = TaylorModel(after_two, Interval.zero(), domain, order=4)
    assert float(model.range_box().lo) <= -4.5
    assert float(model.range_box().hi) >= 1.5


def test_shared_source_cancellation_and_legacy_rebox_excess() -> None:
    z = Polynomial.variable(0, 1)
    shared_x = Polynomial.constant(1.0, 1) + z
    shared_y = Polynomial.constant(1.0, 1) + z
    assert not (shared_x - shared_y).terms
    independent_domain = [UNIT, UNIT]
    rebox_x = Polynomial.constant(1.0, 2) + Polynomial.variable(0, 2)
    rebox_y = Polynomial.constant(1.0, 2) + Polynomial.variable(1, 2)
    excess = (rebox_x - rebox_y).evaluate_interval(independent_domain)
    assert float(excess.lo) <= -2.0 and float(excess.hi) >= 2.0


def test_x_squared_y_uses_one_source_and_merges_paths() -> None:
    z = Polynomial.variable(0, 1)
    x = Polynomial.constant(1.0, 1) + z
    y = Polynomial.constant(1.0, 1) + z
    cubic = x * x * y
    expected = {(0,): 1.0, (1,): 3.0, (2,): 3.0, (3,): 1.0}
    assert {exp: float(value) for exp, value in cubic.terms.items()} == expected
    assert float(cubic.evaluate_interval([UNIT]).lo) <= 0.0
    assert float(cubic.evaluate_interval([UNIT]).hi) >= 8.0


def test_ordinary_structured_products_and_asymmetric_remainder_contain_corners() -> None:
    domain = [UNIT]
    structured = TaylorModel(
        Polynomial.constant(2.0, 1) + Polynomial.variable(0, 1) * 0.5,
        Interval(-0.2, 0.3),
        domain,
        order=4,
    )
    ordinary = TaylorModel.constant(1.0, domain, order=4, remainder=Interval(-0.4, 0.1))
    product = structured * ordinary
    bound = product.range_box()
    for z in (-1.0, 1.0):
        for r_s in (-0.2, 0.3):
            for r_o in (-0.4, 0.1):
                exact = (2.0 + 0.5 * z + r_s) * (1.0 + r_o)
                assert float(bound.lo) <= exact <= float(bound.hi)


def test_degree4_truncation_integration_and_cutoff_have_distinct_owners() -> None:
    domain = [UNIT, Interval(0.0, 0.1)]
    state = TMVector(
        [
            TaylorModel(
                Polynomial.variable(0, 2) + Polynomial.variable(1, 2),
                Interval(-0.01, 0.02),
                domain,
                order=4,
            )
        ]
    )
    dense = sparse_tmvector_to_dense(state, order=4)
    fourth = dense.mul_trunc(dense).mul_trunc(dense).mul_trunc(dense)
    fifth = fourth.mul_trunc(dense)
    integrated = fourth.integrate(1)
    cutoff = fourth.apply_cutoff(10.0)
    assert "polynomial_truncation" in fifth.ledger.entries
    assert bool(torch.any(fifth.ledger.entries["polynomial_truncation"][1] > 0))
    assert "integration_overflow" in integrated.ledger.entries
    assert bool(torch.any(integrated.ledger.entries["integration_overflow"][1] > 0))
    assert "cutoff" in cutoff.ledger.entries
    assert bool(torch.any(cutoff.ledger.entries["cutoff"][1] > 0))


def test_duplicate_exponent_merge_and_tau_substitution() -> None:
    # Polynomial construction independently supplies the same exponent twice
    # through addition; endpoint substitution must merge the resulting u term.
    u = Polynomial.variable(0, 2)
    tau = Polynomial.variable(1, 2)
    poly = u * tau + u * 2.0
    endpoint = poly.substitute_const(1, 3.0).drop_variable(1)
    assert set(endpoint.terms) == {(1,)}
    assert float(endpoint.terms[(1,)]) == 5.0


def test_source_retire_collapse_contains_exact_polynomial_range() -> None:
    # u is base, z is source.  The z-bearing polynomial has cancellations and
    # duplicate paths; collapse evaluates it only after canonical merge.
    u = Polynomial.variable(0, 2)
    z = Polynomial.variable(1, 2)
    polynomial = Polynomial.constant(2.0, 2) + u + (z * z - z * z) + u * z + z * 0.25
    witness = collapse_source_polynomial(polynomial, [UNIT, UNIT], [1])
    assert witness.retained.terms == (Polynomial.constant(2.0, 2) + u).terms
    for u_value in (-1.0, 1.0):
        for z_value in (-1.0, 1.0):
            source_value = u_value * z_value + 0.25 * z_value
            assert float(witness.collapsed.lo) <= source_value <= float(witness.collapsed.hi)
    assert witness.source_term_count == 2


def test_retry_is_atomic_and_state_hash_unchanged() -> None:
    initial = BoundedSourceLedgerState.initial(2)
    successor = accepted_successor(
        initial,
        torch.tensor([[0.1, 0.2]], dtype=torch.float64),
        ("polynomial_truncation", "integration_overflow"),
    )
    rejected = commit_or_preserve(initial, successor, accepted=False)
    assert rejected is initial
    assert rejected.fingerprint == initial.fingerprint
    accepted = commit_or_preserve(initial, successor, accepted=True)
    assert accepted.accepted_boundary_index == 1


@pytest.mark.parametrize("batch", [1, 8, 64])
def test_batch_lift_permutation_equivariance(batch: int) -> None:
    generator = torch.Generator().manual_seed(20260814 + batch)
    lo = torch.rand((batch, 3), generator=generator, dtype=torch.float64) - 1.0
    hi = lo + torch.rand((batch, 3), generator=generator, dtype=torch.float64)
    permutation = torch.arange(batch - 1, -1, -1)
    direct = affine_lift_interval(lo, hi)
    permuted = affine_lift_interval(lo[permutation], hi[permutation])
    assert torch.equal(direct.midpoint[permutation], permuted.midpoint)
    assert torch.equal(direct.radius[permutation], permuted.radius)
    assert torch.equal(direct.contains_input[permutation], permuted.contains_input)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_lift_decision_parity() -> None:
    lo = torch.tensor([[-0.3, -1e-12], [0.1, 2.0]], dtype=torch.float64)
    hi = torch.tensor([[0.7, 2e-12], [0.10000000000000003, 3.0]], dtype=torch.float64)
    cpu = affine_lift_interval(lo, hi)
    gpu = affine_lift_interval(lo.cuda(), hi.cuda())
    assert torch.equal(cpu.contains_input, gpu.contains_input.cpu())
    # Numeric payloads may differ by an ulp; each must independently contain
    # the original exact binary64 interval.
    assert bool(torch.all(gpu.represented_lo.cpu() <= lo))
    assert bool(torch.all(gpu.represented_hi.cpu() >= hi))


def test_actual_dense_picard_consumes_shared_source() -> None:
    source_domain = [UNIT]
    z = Polynomial.variable(0, 1)
    shared = TMVector(
        [
            TaylorModel(Polynomial.constant(1.0, 1) + z, Interval.zero(), source_domain, order=4),
            TaylorModel(Polynomial.constant(1.0, 1) + z, Interval.zero(), source_domain, order=4),
        ]
    ).extend_domain(Interval(0.0, 0.1))
    ordinary_domain = [UNIT]
    ordinary = TMVector(
        [
            TaylorModel.constant(1.0, ordinary_domain, order=4, remainder=UNIT),
            TaylorModel.constant(1.0, ordinary_domain, order=4, remainder=UNIT),
        ]
    ).extend_domain(Interval(0.0, 0.1))

    def rhs(value):
        delta = value.component(0).sub(value.component(1))
        return type(value).concat([delta, delta])

    source_dense = sparse_tmvector_to_dense(shared, order=4)
    ordinary_dense = sparse_tmvector_to_dense(ordinary, order=4)
    source_result = dense_picard_validate_step(
        rhs,
        source_dense,
        h=0.1,
        order=4,
        tau_index=1,
        target_remainder_radius=3.0,
        cutoff_threshold=None,
    )
    ordinary_result = dense_picard_validate_step(
        rhs,
        ordinary_dense,
        h=0.1,
        order=4,
        tau_index=1,
        target_remainder_radius=3.0,
        cutoff_threshold=None,
    )
    assert source_result.accepted and ordinary_result.accepted
    assert torch.all(source_result.picard_image_remainder_hi <= ordinary_result.picard_image_remainder_hi)
    assert torch.any(source_result.picard_image_remainder_hi < ordinary_result.picard_image_remainder_hi)


def test_payload_tamper_changes_consumer_but_metadata_tamper_does_not() -> None:
    midpoint = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    radius = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    payload = source_payload_hash(midpoint, radius)
    changed = source_payload_hash(midpoint, torch.tensor([[0.1, 0.21]], dtype=torch.float64))
    assert payload != changed
    state = accepted_successor(BoundedSourceLedgerState.initial(2), radius, ("picard_residual",))
    tampered = metadata_tamper(state, "does-not-enter-polynomial")
    assert state.fingerprint != tampered.fingerprint
    assert source_payload_hash(midpoint, radius) == payload


def test_exact_decimal_initialization_is_consumed_by_dense_picard() -> None:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("1.1", "1.4"), ("2.35", "2.45")], 4
    )
    tmv = state.normalized_initial_tm(4)
    assert tmv.range_box()[0].lo <= torch.tensor(1.1, dtype=torch.float64)
    consumed = sparse_tmvector_to_dense(tmv.extend_domain(Interval(0.0, 0.01)), order=4)
    assert consumed.poly.coeffs.shape[1] == 2
    assert state.diagnostics["initialization_contract"] == "exact_decimal_contract"


def test_production_bridge_source_payload_is_next_actual_picard_input() -> None:
    ode = PolynomialODE.from_system_spec(
        {
            "state_names": ["x", "y"],
            "rhs": [
                {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                {
                    "terms": [
                        {"coefficient": 1.0, "powers": [0, 1]},
                        {"coefficient": -1.0, "powers": [2, 1]},
                        {"coefficient": -1.0, "powers": [1, 0]},
                    ]
                },
            ],
        }
    )
    first = flowpipe_step_flowstar_style_adaptive(
        ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion_bounded_source_ledger_o4_g1",
        tm_backend="dense",
    )
    assert first.status == "validated" and first.reset_tm is not None
    assert first.flowstar_normal_state is not None
    source = first.flowstar_normal_state.bounded_source_ledger_state
    assert source is not None and source.live_source_count == 2
    assert first.reset_tm.n_vars == 4
    assert {2, 3}.issubset(first.reset_tm.active_variables())
    reset_digest = _poly_digest(first.reset_tm)

    metadata_only = metadata_tamper(source, "consumer-parity")
    metadata_state = first.flowstar_normal_state.__class__(
        **{
            **first.flowstar_normal_state.__dict__,
            "bounded_source_ledger_state": metadata_only,
        }
    )
    assert _poly_digest(metadata_state.normalized_initial_tm(4)) == reset_digest

    changed_radii = list(source.radii_hex)
    changed_radii[0] = float(float.fromhex(changed_radii[0]) * 1.01).hex()
    changed_source = source.__class__(
        **{**source.__dict__, "radii_hex": tuple(changed_radii)}
    )
    changed_state = first.flowstar_normal_state.__class__(
        **{
            **first.flowstar_normal_state.__dict__,
            "bounded_source_ledger_state": changed_source,
        }
    )
    assert _poly_digest(changed_state.normalized_initial_tm(4)) != reset_digest

    second = flowpipe_step_flowstar_style_adaptive(
        ode,
        first.reset_tm,
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion_bounded_source_ledger_o4_g1",
        flowstar_normal_state=first.flowstar_normal_state,
        tm_backend="dense",
    )
    assert second.status == "validated"
    assert second.flowstar_normal_state is not None
    after = second.flowstar_normal_state.bounded_source_ledger_state
    assert after is not None
    assert after.collapse_count == 1 and after.retired_source_count == 2


def test_bounded_source_terminal_checkpoint_v3_roundtrip(tmp_path) -> None:
    ode = PolynomialODE.from_system_spec(
        {
            "state_names": ["x", "y"],
            "rhs": [
                {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                {"terms": [
                    {"coefficient": 1.0, "powers": [0, 1]},
                    {"coefficient": -1.0, "powers": [2, 1]},
                    {"coefficient": -1.0, "powers": [1, 0]},
                ]},
            ],
        }
    )
    segment = flowpipe_step_flowstar_style_adaptive(
        ode,
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion_bounded_source_ledger_o4_g1",
        tm_backend="dense",
    )
    assert segment.reset_tm is not None and segment.flowstar_normal_state is not None
    contract = {"order": 4, "candidate": "bounded_source_g1"}
    manifest = save_terminal_checkpoint(
        tmp_path / "checkpoint",
        current=segment.reset_tm,
        normal_state=segment.flowstar_normal_state,
        scheduler={"t": 0.01, "h": 0.01},
        contract=contract,
        provenance={"test": True},
    )
    assert manifest["schema"] == "torch_tm_flowpipe_terminal_checkpoint_v3"
    restored = load_terminal_checkpoint(
        tmp_path / "checkpoint",
        expected_contract=contract,
        expected_order=4,
        expected_dtype="float64",
    )
    original_source = segment.flowstar_normal_state.bounded_source_ledger_state
    restored_source = restored.normal_state.bounded_source_ledger_state
    assert original_source is not None and restored_source is not None
    assert original_source.as_dict() == restored_source.as_dict()
    assert _poly_digest(segment.reset_tm) == _poly_digest(restored.current)
