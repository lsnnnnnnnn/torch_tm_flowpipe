from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path

import pytest
import torch

from torch_tm_flowpipe import (
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    Interval,
    PolynomialODE,
    flowpipe_step_flowstar_style_adaptive,
    load_terminal_checkpoint,
    save_terminal_checkpoint,
    tmvector_hashes,
)
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseExecutionCounters,
    _call_dense_raw_trace_rhs,
    _joint_factorized_vdp_residual_closure,
    dense_picard_validate_step,
    dense_polynomial_picard,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.raw_remainder_trace import RawRemainderTraceRecorder
from torch_tm_flowpipe.step1_oracle import RationalInterval, RationalPolynomial


LEGACY = "flowstar_raw_remainder_compat"
H2 = "flowstar_raw_remainder_compat_factorized_joint"
C1 = "flowstar_raw_remainder_compat_factorized_joint_closure"


@pytest.mark.unit
def test_exact_tensor_bernstein_micro_oracle_captures_square_dependency() -> None:
    r = RationalPolynomial(1, {(1,): Fraction(1)})
    square = r * r

    bound = square.bernstein_range((RationalInterval(0, 1),))

    assert bound.lo == 0
    assert bound.hi == 1


def _vdp() -> PolynomialODE:
    return PolynomialODE.from_system_spec(
        {
            "state_names": ["x", "y"],
            "rhs": [
                {"terms": [{"coefficient": 1.0, "powers": [0, 1]}]},
                {
                    "terms": [
                        {"coefficient": 1.0, "powers": [0, 1]},
                        {"coefficient": -1.0, "powers": [1, 0]},
                        {"coefficient": -1.0, "powers": [2, 1]},
                    ]
                },
            ],
        }
    )


def _step1_base(device: str = "cpu") -> tuple[FlowstarNormalFlowpipeState, BatchedTaylorModel]:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("11/10", "7/5"), ("47/20", "49/20")], 4
    )
    base = sparse_tmvector_to_dense(
        state.normalized_initial_tm(4).extend_domain(Interval(0.0, 0.01)),
        order=4,
        device=device,
        dtype=torch.float64,
        counters=DenseExecutionCounters(),
        segment_boundary=True,
        range_policy=DenseRangePolicy(
            method="adaptive_subdivision",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
            trigger="proactive_depth1_on_named_contexts",
            named_contexts=("polynomial_truncation",),
        ),
        range_trace=[],
    )
    return state, base


def _scalar_tm(
    *,
    constant: float,
    linear: float,
    remainder: tuple[float, float],
    order: int = 2,
) -> BatchedTaylorModel:
    basis = BatchedMonomialBasis.build(1, order, "cpu")
    coeffs = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    coeffs[0, 0, basis.term_index((0,))] = constant
    coeffs[0, 0, basis.term_index((1,))] = linear
    domain_lo = torch.tensor([[-1.0]], dtype=torch.float64)
    domain_hi = torch.tensor([[1.0]], dtype=torch.float64)
    return BatchedTaylorModel(
        BatchedPolynomial(coeffs, basis),
        torch.tensor([[remainder[0]]], dtype=torch.float64),
        torch.tensor([[remainder[1]]], dtype=torch.float64),
        domain_lo,
        domain_hi,
    )


def _q(value: torch.Tensor) -> Fraction:
    return Fraction.from_float(float(value.detach().cpu().reshape(-1)[0]))


def _rational_poly(model: BatchedTaylorModel, *, n_vars: int) -> RationalPolynomial:
    return RationalPolynomial(
        n_vars,
        {
            tuple(exponent) + (0,) * (n_vars - model.n_vars): Fraction.from_float(
                float(model.poly.coeffs[0, 0, slot])
            )
            for slot, exponent in enumerate(model.poly.basis.exponent_to_index)
            if float(model.poly.coeffs[0, 0, slot]) != 0.0
        },
    )


def _rational_variable(n_vars: int, index: int) -> RationalPolynomial:
    exponent = [0] * n_vars
    exponent[index] = 1
    return RationalPolynomial(n_vars, {tuple(exponent): Fraction(1)})


def _exact_joint_closure_oracle(
    x: BatchedTaylorModel,
    y: BatchedTaylorModel,
    retained: BatchedTaylorModel,
) -> RationalInterval:
    n_vars = x.n_vars + 2
    px = _rational_poly(x, n_vars=n_vars) + _rational_variable(n_vars, x.n_vars)
    py = _rational_poly(y, n_vars=n_vars) + _rational_variable(n_vars, x.n_vars + 1)
    retained_poly = _rational_poly(retained, n_vars=n_vars)
    residual = (RationalPolynomial.constant(n_vars, 1) - px * px) * py - px - retained_poly
    domain = tuple(
        RationalInterval(
            Fraction.from_float(float(lo)),
            Fraction.from_float(float(hi)),
        )
        for lo, hi in zip(
            torch.cat((x.domain_lo[0], x.rem_lo[0], y.rem_lo[0])),
            torch.cat((x.domain_hi[0], x.rem_hi[0], y.rem_hi[0])),
        )
    )
    return residual.bernstein_range(domain)


@pytest.mark.unit
def test_joint_square_contains_exact_asymmetric_fraction_oracle_and_is_narrower() -> None:
    # P(u)=2+u/2, u in [-1,1], R in [-1/5,1/10].  The exact extrema of
    # 2*P*R+R^2 are -24/25 and 51/100.
    model = _scalar_tm(constant=2.0, linear=0.5, remainder=(-0.2, 0.1))

    joint = model.square_trunc_dependency_preserving(max_degree=2)
    legacy = model.mul_trunc(model, max_degree=2)

    assert _q(joint.rem_lo) <= Fraction(-24, 25)
    assert _q(joint.rem_hi) >= Fraction(51, 100)
    assert float(joint.rem_hi - joint.rem_lo) < float(legacy.rem_hi - legacy.rem_lo)
    assert torch.equal(
        joint.ledger.entries["remainder_times_remainder"][0],
        torch.zeros_like(joint.rem_lo),
    )
    assert torch.equal(
        joint.ledger.entries["remainder_times_remainder"][1],
        torch.zeros_like(joint.rem_hi),
    )


@pytest.mark.unit
def test_joint_square_vertex_when_polynomial_range_crosses_zero() -> None:
    # P(u)=0.05+0.15u gives P in [-0.1,0.2].  With R in [-0.3,0.3],
    # the minimum -P^2 occurs at P=0.2,R=-0.2 and the maximum is 0.21.
    model = _scalar_tm(constant=0.05, linear=0.15, remainder=(-0.3, 0.3))

    result = model.square_trunc_dependency_preserving(max_degree=2)

    assert _q(result.rem_lo) <= Fraction(-1, 25)
    assert _q(result.rem_hi) >= Fraction(21, 100)


@pytest.mark.unit
def test_joint_square_zero_remainder_is_exact_and_order_overflow_is_enclosed() -> None:
    zero = _scalar_tm(constant=2.0, linear=0.5, remainder=(0.0, 0.0))
    exact = zero.square_trunc_dependency_preserving(max_degree=2)
    assert torch.equal(exact.rem_lo, torch.zeros_like(exact.rem_lo))
    assert torch.equal(exact.rem_hi, torch.zeros_like(exact.rem_hi))

    overflow = zero.square_trunc_dependency_preserving(max_degree=1)
    # The only discarded term is u^2/4, whose exact range is [0,1/4].
    assert _q(overflow.rem_lo) <= 0
    assert _q(overflow.rem_hi) >= Fraction(1, 4)


@pytest.mark.unit
def test_joint_square_cutoff_keeps_discarded_term_owned() -> None:
    model = _scalar_tm(constant=1.0, linear=1.0e-6, remainder=(-1.0e-8, 2.0e-8))
    result = model.square_trunc_dependency_preserving(max_degree=2).apply_cutoff(1.0e-5)

    assert "cutoff" in result.ledger.entries
    assert float(result.ledger.entries["cutoff"][1]) > 0.0
    # At u=1, r=2e-8 the exact square must remain in the published TM.
    exact = (
        Fraction(1)
        + Fraction.from_float(1.0e-6)
        + Fraction.from_float(2.0e-8)
    ) ** 2
    polynomial = Fraction.from_float(float(result.poly.coeffs[0, 0, 0]))
    assert _q(result.rem_lo) + polynomial <= exact <= _q(result.rem_hi) + polynomial


@pytest.mark.unit
def test_joint_square_cutoff_boundary_is_deterministic() -> None:
    at_boundary = _scalar_tm(
        constant=1.0,
        linear=5.0e-11,
        remainder=(0.0, 0.0),
    ).square_trunc_dependency_preserving(max_degree=2)
    linear_slot = at_boundary.poly.basis.term_index((1,))
    threshold = float(at_boundary.poly.coeffs[0, 0, linear_slot])
    removed = at_boundary.apply_cutoff(threshold)
    assert float(removed.poly.coeffs[0, 0, linear_slot]) == 0.0

    retained = at_boundary.apply_cutoff(math.nextafter(threshold, 0.0))
    assert float(retained.poly.coeffs[0, 0, linear_slot]) == threshold


@pytest.mark.unit
def test_h2_same_step1_prestate_reduces_first_y_raw_residual_without_x_regression() -> None:
    state, base = _step1_base()
    before = base.poly.coeffs.detach().cpu().numpy().tobytes()
    common = dict(
        h=0.01,
        order=4,
        tau_index=2,
        target_remainder_radius=1.0e-4,
        cutoff_threshold=1.0e-10,
        max_validation_attempts=2,
        validation_eps=1.0e-12,
    )
    legacy = dense_picard_validate_step(_vdp(), base, validation_mode=LEGACY, **common)
    recorder = RawRemainderTraceRecorder(
        run_id="h2-unit",
        tool="torch",
        source_commit="0" * 40,
        binary_sha256="0" * 64,
        checkpoint_sha256="1" * 64,
        t_pre=0.0,
        h=0.01,
        picard_iteration=4,
        normalization_scale=state.scales,
        target_intervals=((-1.0e-4, 1.0e-4), (-1.0e-4, 1.0e-4)),
    )
    h2 = dense_picard_validate_step(
        _vdp(),
        base,
        validation_mode=H2,
        raw_remainder_trace_recorder=recorder,
        **common,
    )

    assert base.poly.coeffs.detach().cpu().numpy().tobytes() == before
    assert legacy.status == h2.status == "validated"
    assert torch.equal(legacy.segment_tm.poly.coeffs, h2.segment_tm.poly.coeffs)
    legacy_row = [row for row in legacy.trace if row["phase"] == "remainder_validation"][-1]
    h2_row = [row for row in h2.trace if row["phase"] == "remainder_validation"][-1]
    legacy_lo = torch.tensor(legacy_row["raw_rhs_remainder_lo"], dtype=torch.float64)
    legacy_hi = torch.tensor(legacy_row["raw_rhs_remainder_hi"], dtype=torch.float64)
    h2_lo = torch.tensor(h2_row["raw_rhs_remainder_lo"], dtype=torch.float64)
    h2_hi = torch.tensor(h2_row["raw_rhs_remainder_hi"], dtype=torch.float64)
    assert torch.equal(legacy_lo[:, 0], h2_lo[:, 0])
    assert torch.equal(legacy_hi[:, 0], h2_hi[:, 0])
    legacy_y_width = legacy_hi[0, 1] - legacy_lo[0, 1]
    h2_y_width = h2_hi[0, 1] - h2_lo[0, 1]
    assert h2_y_width <= 0.9 * legacy_y_width
    artifact = recorder.artifact()
    assert artifact["expression_mode"] == "canonical_factorized_joint"
    ids = {node["expression_node_id"] for node in artifact["nodes"]}
    assert any("x_squared_joint" in value for value in ids)
    assert any("factor_times_y" in value for value in ids)


@pytest.mark.unit
def test_c1_step1_joint_closure_contains_fraction_oracle_and_clears_micro_gate() -> None:
    state, base = _step1_base()
    candidate, _ = dense_polynomial_picard(
        _vdp(),
        base.without_remainder(),
        tau_index=2,
        order=4,
        iterations=4,
        cutoff_threshold=1.0e-10,
    )
    target = candidate.with_remainder(
        torch.full_like(candidate.rem_lo, -1.0e-4),
        torch.full_like(candidate.rem_hi, 1.0e-4),
        category="initial_remainder",
    )
    h2_raw = _call_dense_raw_trace_rhs(
        _vdp(),
        target,
        effective_order=3,
        cutoff_threshold=1.0e-10,
        evaluation_mode="canonical_factorized_joint",
        dependency_preserving_square=True,
    )
    closure, certificate = _joint_factorized_vdp_residual_closure(
        target.component(0),
        target.component(1),
        h2_raw.component(1),
    )
    exact = _exact_joint_closure_oracle(
        target.component(0),
        target.component(1),
        h2_raw.component(1),
    )
    production = RationalInterval(_q(closure.rem_lo), _q(closure.rem_hi))
    h2_width = _q(h2_raw.rem_hi[:, 1]) - _q(h2_raw.rem_lo[:, 1])
    removed_fraction = (h2_width - production.width) / (h2_width - exact.width)

    recorder = RawRemainderTraceRecorder(
        run_id="c1-unit",
        tool="torch",
        source_commit="0" * 40,
        binary_sha256="0" * 64,
        checkpoint_sha256="1" * 64,
        t_pre=0.0,
        h=0.01,
        picard_iteration=4,
        normalization_scale=state.scales,
        target_intervals=((-1.0e-4, 1.0e-4), (-1.0e-4, 1.0e-4)),
    )
    step = dense_picard_validate_step(
        _vdp(),
        base,
        h=0.01,
        order=4,
        tau_index=2,
        target_remainder_radius=1.0e-4,
        cutoff_threshold=1.0e-10,
        max_validation_attempts=2,
        validation_eps=1.0e-12,
        validation_mode=C1,
        raw_remainder_trace_recorder=recorder,
    )

    assert exact.subseteq(production)
    assert removed_fraction >= Fraction(1, 10)
    assert certificate["operator"] == "factor_times_y_joint_tensor_bernstein"
    assert step.status == "validated"
    assert torch.equal(step.segment_tm.poly.coeffs, candidate.poly.coeffs)
    artifact = recorder.artifact()
    assert artifact["expression_mode"] == "canonical_factorized_joint_closure"
    closure_event = next(
        row for row in artifact["execution_events"]
        if row["operation"] == "factor_times_y_joint_tensor_bernstein"
    )
    old_root = next(
        row for row in artifact["execution_events"]
        if row["stage_id"].endswith("factorized_final")
    )
    assert closure_event["discard_site"] is None
    assert old_root["discard_site"] is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("x_args", "y_args", "raw_order", "cutoff"),
    [
        (
            {"constant": 0.0, "linear": 0.0, "remainder": (0.0, 0.0)},
            {"constant": 0.0, "linear": 0.0, "remainder": (0.0, 0.0)},
            0,
            None,
        ),
        (
            {"constant": 1.2, "linear": 0.4, "remainder": (-0.03, 0.01)},
            {"constant": 0.7, "linear": -0.2, "remainder": (-0.02, 0.04)},
            1,
            1.0e-2,
        ),
        (
            {
                "constant": 1.0,
                "linear": math.ldexp(1.0, -530),
                "remainder": (-math.ldexp(1.0, -540), math.ldexp(1.0, -539)),
            },
            {
                "constant": 1.0,
                "linear": -math.ldexp(1.0, -530),
                "remainder": (-math.ldexp(1.0, -541), math.ldexp(1.0, -540)),
            },
            0,
            math.ldexp(1.0, -529),
        ),
    ],
    ids=("zero", "asymmetric-cutoff-order-overflow", "underflow-cutoff-boundary"),
)
def test_c1_adversarial_joint_closure_contains_exact_oracle(
    x_args: dict[str, object],
    y_args: dict[str, object],
    raw_order: int,
    cutoff: float | None,
) -> None:
    x = _scalar_tm(**x_args)
    y = _scalar_tm(**y_args)
    state = BatchedTaylorModel.concat((x, y))
    raw = _call_dense_raw_trace_rhs(
        _vdp(),
        state,
        effective_order=raw_order,
        cutoff_threshold=cutoff,
        evaluation_mode="canonical_factorized_joint",
        dependency_preserving_square=True,
    )
    closure, _ = _joint_factorized_vdp_residual_closure(x, y, raw.component(1))
    exact = _exact_joint_closure_oracle(x, y, raw.component(1))
    production = RationalInterval(_q(closure.rem_lo), _q(closure.rem_hi))

    assert exact.subseteq(production)
    assert torch.all(torch.isfinite(closure.rem_lo))
    assert torch.all(torch.isfinite(closure.rem_hi))
    if x_args["constant"] == 0.0:
        assert production.lo <= 0 <= production.hi
        assert float(production.width) < 1.0e-300


def _full_h2_step(current, state, device: str, validation_mode: str = H2):
    return flowpipe_step_flowstar_style_adaptive(
        _vdp(),
        current,
        h=0.01,
        h_min=0.01,
        h_max=0.01,
        order=4,
        target_remainder_radius=1.0e-4,
        cutoff_threshold=1.0e-10,
        max_validation_attempts=2,
        validation_eps=1.0e-12,
        validation_mode=validation_mode,
        reset_mode="normalized_insertion_dependency_preserving",
        step_policy_mode="flowstar_compat",
        flowstar_normal_state=state,
        tm_backend="dense",
        dense_device=device,
        dense_dtype=torch.float64,
        dense_range_policy=DenseRangePolicy(
            method="adaptive_subdivision",
            max_depth=1,
            max_leaves=4,
            split_vars=(0, 1),
            trigger="proactive_depth1_on_named_contexts",
            named_contexts=("polynomial_truncation",),
        ),
    )


@pytest.mark.unit
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_h2_checkpoint_resume_is_bitwise_on_each_device(tmp_path: Path, device: str) -> None:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("11/10", "7/5"), ("47/20", "49/20")], 4
    )
    first = _full_h2_step(state.normalized_initial_tm(4), state, device)
    assert first.status == "validated"
    assert first.reset_tm is not None and first.flowstar_normal_state is not None
    contract = {"validation_mode": H2, "reset_mode": "normalized_insertion_dependency_preserving"}
    checkpoint = tmp_path / device
    save_terminal_checkpoint(
        checkpoint,
        current=first.reset_tm,
        normal_state=first.flowstar_normal_state,
        scheduler={"current_time": 0.01, "h_next": 0.01},
        contract=contract,
        provenance={"test": "h2-checkpoint-resume", "device": device},
    )
    uninterrupted = _full_h2_step(first.reset_tm, first.flowstar_normal_state, device)
    loaded = load_terminal_checkpoint(checkpoint, expected_contract=contract, expected_order=4, expected_dtype="float64")
    resumed = _full_h2_step(loaded.current, loaded.normal_state, device)

    assert uninterrupted.status == resumed.status == "validated"
    assert uninterrupted.reset_tm is not None and resumed.reset_tm is not None
    assert tmvector_hashes(uninterrupted.reset_tm) == tmvector_hashes(resumed.reset_tm)
    assert uninterrupted.flowstar_normal_state is not None
    assert resumed.flowstar_normal_state is not None
    assert uninterrupted.flowstar_normal_state.center == resumed.flowstar_normal_state.center
    assert uninterrupted.flowstar_normal_state.scales == resumed.flowstar_normal_state.scales


@pytest.mark.unit
def test_c1_checkpoint_resume_is_bitwise_on_cpu(tmp_path: Path) -> None:
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("11/10", "7/5"), ("47/20", "49/20")], 4
    )
    first = _full_h2_step(state.normalized_initial_tm(4), state, "cpu", C1)
    assert first.status == "validated"
    assert first.reset_tm is not None and first.flowstar_normal_state is not None
    contract = {"validation_mode": C1, "reset_mode": "normalized_insertion_dependency_preserving"}
    checkpoint = tmp_path / "c1-cpu"
    save_terminal_checkpoint(
        checkpoint,
        current=first.reset_tm,
        normal_state=first.flowstar_normal_state,
        scheduler={"current_time": 0.01, "h_next": 0.01},
        contract=contract,
        provenance={"test": "c1-checkpoint-resume", "device": "cpu"},
    )
    uninterrupted = _full_h2_step(first.reset_tm, first.flowstar_normal_state, "cpu", C1)
    loaded = load_terminal_checkpoint(
        checkpoint,
        expected_contract=contract,
        expected_order=4,
        expected_dtype="float64",
    )
    resumed = _full_h2_step(loaded.current, loaded.normal_state, "cpu", C1)

    assert uninterrupted.status == resumed.status == "validated"
    assert uninterrupted.reset_tm is not None and resumed.reset_tm is not None
    assert tmvector_hashes(uninterrupted.reset_tm) == tmvector_hashes(resumed.reset_tm)
    assert uninterrupted.flowstar_normal_state is not None
    assert resumed.flowstar_normal_state is not None
    assert uninterrupted.flowstar_normal_state.center == resumed.flowstar_normal_state.center
    assert uninterrupted.flowstar_normal_state.scales == resumed.flowstar_normal_state.scales


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_h2_cpu_cuda_decision_and_widths_are_consistent() -> None:
    common = dict(
        h=0.01,
        order=4,
        tau_index=2,
        target_remainder_radius=1.0e-4,
        cutoff_threshold=1.0e-10,
        max_validation_attempts=2,
        validation_eps=1.0e-12,
        validation_mode=H2,
    )
    _, cpu_base = _step1_base("cpu")
    _, cuda_base = _step1_base("cuda")
    cpu = dense_picard_validate_step(_vdp(), cpu_base, **common)
    cuda = dense_picard_validate_step(_vdp(), cuda_base, **common)

    assert cpu.status == cuda.status == "validated"
    assert torch.equal(cpu.subset_margin >= 0, cuda.subset_margin.cpu() >= 0)
    assert torch.allclose(cpu.segment_tm.rem_lo, cuda.segment_tm.rem_lo.cpu(), rtol=1.0e-12, atol=1.0e-15)
    assert torch.allclose(cpu.segment_tm.rem_hi, cuda.segment_tm.rem_hi.cpu(), rtol=1.0e-12, atol=1.0e-15)
