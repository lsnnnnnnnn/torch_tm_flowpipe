import importlib.util
import json
import sys
from pathlib import Path

import torch

from torch_tm_flowpipe import (
    BatchedMonomialBasis,
    BatchedPolynomial,
    BatchedTaylorModel,
    DenseRangePolicy,
    Interval,
    Polynomial,
    TaylorModel,
    TMVector,
    flowpipe_step_flowstar_style_adaptive,
    sparse_tmvector_to_dense,
)


def _dense_model(policy, trace, *, remainder=(-1e-3, 2e-3)):
    domain = [Interval(-1.0, 1.0), Interval(-1.0, 1.0), Interval(0.0, 0.01)]
    polynomial = Polynomial(
        {
            (0, 0, 0): 0.140625,
            (1, 0, 0): -0.75,
            (2, 0, 0): 1.0,
            (0, 1, 0): 0.25,
        },
        n_vars=3,
    )
    sparse = TMVector([TaylorModel(polynomial, Interval(*remainder), domain, order=4)])
    return sparse_tmvector_to_dense(sparse, order=4, range_policy=policy, range_trace=trace)


def test_default_policy_remains_natural_and_reproduces_legacy_interval_bound():
    basis = BatchedMonomialBasis.build(2, 4)
    generator = torch.Generator().manual_seed(7)
    coeffs = torch.randn((16, 2, basis.num_terms), generator=generator, dtype=torch.float64)
    domain_lo = torch.tensor([[-0.75, 0.125]], dtype=torch.float64).repeat(16, 1)
    domain_hi = torch.tensor([[1.25, 0.625]], dtype=torch.float64).repeat(16, 1)
    polynomial = BatchedPolynomial(coeffs, basis)
    legacy_lo, legacy_hi = polynomial.range_bound(domain_lo, domain_hi, method="interval")
    natural_lo, natural_hi = polynomial.range_bound(domain_lo, domain_hi, method="natural")
    assert torch.equal(legacy_lo, natural_lo)
    assert torch.equal(legacy_hi, natural_hi)


def test_named_context_policy_only_subdivides_pre_registered_context():
    trace = []
    policy = DenseRangePolicy(
        method="subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        named_contexts=("polynomial_truncation",),
    )
    model = _dense_model(policy, trace)
    model.mul_trunc(model)
    rows = [row for row in trace if row["phase"] == "polynomial_range"]
    by_context = {}
    for row in rows:
        by_context.setdefault(row["context"], []).append(row)
    assert {"polynomial_truncation", "poly_times_remainder", "remainder_times_poly"} <= set(by_context)
    assert all(row["leaf_count"] == 4 for row in by_context["polynomial_truncation"])
    assert all(row["leaf_count"] == 1 for row in by_context["poly_times_remainder"])
    assert all(row["leaf_count"] == 1 for row in by_context["remainder_times_poly"])


def test_dropped_terms_are_merged_then_subdivided_before_intervalization():
    basis = BatchedMonomialBasis.build(1, 4)
    left_coeffs = torch.zeros((1, 1, basis.num_terms), dtype=torch.float64)
    right_coeffs = torch.zeros_like(left_coeffs)
    # (x + x^2)(1 - x) has two x^2 routes which cancel before bounding.
    left_coeffs[0, 0, basis.term_index((1,))] = 1.0
    left_coeffs[0, 0, basis.term_index((2,))] = 1.0
    right_coeffs[0, 0, basis.term_index((0,))] = 1.0
    right_coeffs[0, 0, basis.term_index((1,))] = -1.0
    left = BatchedPolynomial(left_coeffs, basis)
    right = BatchedPolynomial(right_coeffs, basis)
    domain_lo = torch.tensor([[-1.0]], dtype=torch.float64)
    domain_hi = torch.tensor([[1.0]], dtype=torch.float64)
    trace = []
    policy = DenseRangePolicy(method="subdivision", max_depth=3, max_leaves=8, split_vars=(0,))
    _kept, lo, hi = left.mul_trunc(
        right,
        max_degree=1,
        return_truncation_bound=True,
        domain_lo=domain_lo,
        domain_hi=domain_hi,
        range_policy=policy,
        range_trace=trace,
    )
    sparse_left = Polynomial({(1,): 1.0, (2,): 1.0}, 1)
    sparse_right = Polynomial({(0,): 1.0, (1,): -1.0}, 1)
    _sparse_kept, sparse_dropped = sparse_left.mul_truncate(sparse_right, 1)
    assert sparse_dropped.terms.keys() == {(3,)}
    assert float(lo) <= -1.0 and float(hi) >= 1.0
    row = next(row for row in trace if row["context"] == "polynomial_truncation")
    assert row["leaf_count"] == 8
    assert row["coverage_valid"]


def test_cutoff_integration_and_remainder_products_use_named_range_contexts():
    trace = []
    policy = DenseRangePolicy(method="subdivision", max_depth=1, max_leaves=4, split_vars=(0, 1))
    model = _dense_model(policy, trace)
    product = model.mul_trunc(model)
    integrated = product.integrate(2)
    integrated.apply_cutoff(10.0)
    contexts = {row["context"] for row in trace if row["phase"] == "polynomial_range"}
    assert {
        "polynomial_truncation",
        "poly_times_remainder",
        "remainder_times_poly",
        "integration_overflow",
        "cutoff",
    } <= contexts
    rr_lo, rr_hi = product.ledger.entries["remainder_times_remainder"]
    expected = Interval(-1e-3, 2e-3) * Interval(-1e-3, 2e-3)
    assert float(rr_lo) <= float(expected.lo)
    assert float(rr_hi) >= float(expected.hi)


def test_ledger_categories_reconstruct_remainder_after_subdivision():
    trace = []
    model = _dense_model(DenseRangePolicy(method="subdivision", max_depth=1, max_leaves=4), trace)
    product = model.mul_trunc(model).integrate(2).apply_cutoff(1e-10)
    ledger_lo, ledger_hi = product.ledger.total(product.rem_lo)
    assert torch.allclose(ledger_lo, product.rem_lo, atol=2e-15, rtol=2e-15)
    assert torch.allclose(ledger_hi, product.rem_hi, atol=2e-15, rtol=2e-15)


def test_harmonic_analytic_one_step_regression_with_subdivision_policy():
    def harmonic(x, u=None):
        return TMVector([x[1], -x[0]])

    policy = DenseRangePolicy(
        method="subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        named_contexts=("polynomial_truncation",),
    )
    segment = flowpipe_step_flowstar_style_adaptive(
        harmonic,
        [Interval(0.9, 1.1), Interval(-0.1, 0.1)],
        h=0.01,
        h_min=0.002,
        h_max=0.1,
        order=4,
        target_remainder_radius=1e-4,
        cutoff_threshold=1e-10,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion",
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
        dense_range_policy=policy,
    )
    assert segment.status == "validated"
    assert segment.endpoint_raw_tm is not None
    endpoint = segment.endpoint_raw_tm.range_box()
    for x0, y0 in ((0.9, -0.1), (0.9, 0.1), (1.1, -0.1), (1.1, 0.1)):
        exact_x = x0 * torch.cos(torch.tensor(0.01)) + y0 * torch.sin(torch.tensor(0.01))
        exact_y = y0 * torch.cos(torch.tensor(0.01)) - x0 * torch.sin(torch.tensor(0.01))
        assert endpoint[0].contains(float(exact_x), tol=2e-9)
        assert endpoint[1].contains(float(exact_y), tol=2e-9)


def test_default_runner_t0p1_schedule_remains_seven_steps(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "run_vdp_dense_backend.py"
    spec = importlib.util.spec_from_file_location("range_policy_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = tmp_path / "natural"
    summary = module.run(module.parse_args(["--output-dir", str(output), "--tm-backend", "dense", "--horizon", "0.1"]))
    assert summary["completed_requested_horizon"] is True
    assert summary["accepted_steps"] == 7
    assert summary["dense_range_method"] == "natural"
    assert summary["range_subdivision_invocations"] == 0


def test_proactive_depth1_runner_subdivides_only_named_context_on_accepted_steps(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "run_vdp_dense_backend.py"
    spec = importlib.util.spec_from_file_location("proactive_range_policy_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = tmp_path / "proactive"
    summary = module.run(
        module.parse_args(
            [
                "--output-dir",
                str(output),
                "--tm-backend",
                "dense",
                "--horizon",
                "0.1",
                "--dense-range-method",
                "adaptive_subdivision",
                "--dense-range-trigger",
                "proactive_depth1_on_named_contexts",
                "--dense-range-max-depth",
                "1",
                "--dense-range-max-leaves",
                "4",
                "--dense-range-contexts",
                "polynomial_truncation",
            ]
        )
    )
    assert summary["completed_requested_horizon"] is True
    assert summary["range_subdivision_invocations"] > 0
    rows = [json.loads(line) for line in (output / "range_trace.jsonl").read_text().splitlines()]
    divided = [row for row in rows if row.get("phase") == "polynomial_range" and row.get("leaf_count", 1) > 1]
    assert divided
    assert {row["context"] for row in divided} == {"polynomial_truncation"}
    assert {row["leaf_count"] for row in divided} == {4}
