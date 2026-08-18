#!/usr/bin/env python3
"""Exact-rational Gate A/B audit for the first dense VDP Picard loss.

The production operators are replayed from one outward-exact step-1 prestate.
Every local semantic residual is lifted from binary64 to ``Fraction`` and
bounded by an exact tensor-product Bernstein enclosure.  Stock Flow* source is
read from its Git object database; runtime data is used only as a cross-check.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_vdp_dense_backend as authoritative
from torch_tm_flowpipe import DenseRangePolicy, FlowstarNormalFlowpipeState, Interval, PolynomialODE
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedTaylorModel,
    DenseExecutionCounters,
    _call_dense_raw_trace_rhs,
    _call_dense_rhs_evaluation,
    _inflate_tensor_interval,
    _interval_add,
    _interval_mul,
    dense_picard_validate_step,
    dense_polynomial_picard,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.step1_oracle import RationalInterval, RationalPolynomial, fraction_text


ORDER = 4
RAW_ORDER = 3
H = 0.01
CUTOFF = 1.0e-10
TARGET = 1.0e-4
EPS = 1.0e-12
LEGACY = "flowstar_raw_remainder_compat"
H2 = "flowstar_raw_remainder_compat_factorized_joint"
VARIABLES_3 = ("ux", "uy", "tau")
VARIABLES_5 = ("ux", "uy", "tau", "rx", "ry")
FLOWSTAR_LEDGER = ROOT / (
    "outputs/flowstar_torch_step1_stage_oracle_sound_carry_20260813/"
    "20260814T014356Z/03_flowstar_actual_stage_ledger/03_process/artifacts/stage_ledger.json"
)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git(*args: str, root: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _q(value: Any) -> Fraction:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1)[0]
    return Fraction.from_float(float(value))


def _number(value: Any) -> dict[str, Any]:
    number = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    exact = Fraction.from_float(number)
    return {
        "decimal": repr(number),
        "hex": number.hex(),
        "exact_rational": fraction_text(exact),
    }


def _interval(lo: Any, hi: Any) -> dict[str, Any]:
    lower = float(lo.detach().cpu()) if isinstance(lo, torch.Tensor) else float(lo)
    upper = float(hi.detach().cpu()) if isinstance(hi, torch.Tensor) else float(hi)
    if lower > upper:
        raise ValueError("reversed interval")
    return {"lo": _number(lower), "hi": _number(upper), "width": _number(upper - lower)}


def _rational_interval(value: RationalInterval) -> dict[str, Any]:
    return {
        "lo": fraction_text(value.lo),
        "hi": fraction_text(value.hi),
        "width": fraction_text(value.width),
        "outward_binary64": {
            "lo_hex": math.nextafter(float(value.lo), -math.inf).hex(),
            "hi_hex": math.nextafter(float(value.hi), math.inf).hex(),
        },
    }


def _source(function: Any) -> dict[str, Any]:
    lines, first = inspect.getsourcelines(function)
    return {
        "file": str(Path(inspect.getsourcefile(function) or "").resolve().relative_to(ROOT)),
        "function": function.__qualname__,
        "line_start": first,
        "line_end": first + len(lines) - 1,
    }


def _model_payload(model: BatchedTaylorModel) -> dict[str, Any]:
    coeffs = model.poly.coeffs.detach().cpu()
    if coeffs.shape[0] != 1:
        raise ValueError("Gate A is frozen to B1")
    components = []
    exponents = model.poly.basis.exponents.detach().cpu().tolist()
    for component in range(model.poly.out_dim):
        terms = []
        for slot, exponent in enumerate(exponents):
            coefficient = float(coeffs[0, component, slot])
            if coefficient != 0.0:
                terms.append({"exponents": exponent, "coefficient": _number(coefficient)})
        components.append(
            {
                "component": component,
                "support": [row["exponents"] for row in terms],
                "terms": terms,
                "ordinary_remainder": _interval(model.rem_lo[0, component], model.rem_hi[0, component]),
                "remainder_ledger": {
                    name: _interval(lo[0, component], hi[0, component])
                    for name, (lo, hi) in model.ledger.entries.items()
                },
            }
        )
    return {
        "basis_fingerprint": model.poly.basis.fingerprint,
        "basis_variables": list(VARIABLES_3),
        "basis_order": model.poly.basis.order,
        "components": components,
        "domain": [
            _interval(model.domain_lo[0, index], model.domain_hi[0, index])
            for index in range(model.domain_lo.shape[1])
        ],
        "coefficient_sha256": hashlib.sha256(coeffs.contiguous().numpy().tobytes()).hexdigest(),
    }


def _dense_equal(left: BatchedTaylorModel, right: BatchedTaylorModel) -> bool:
    return bool(
        torch.equal(left.poly.coeffs, right.poly.coeffs)
        and torch.equal(left.rem_lo, right.rem_lo)
        and torch.equal(left.rem_hi, right.rem_hi)
        and torch.equal(left.domain_lo, right.domain_lo)
        and torch.equal(left.domain_hi, right.domain_hi)
        and left.ledger.intervals() == right.ledger.intervals()
    )


def _poly_from_model(
    model: BatchedTaylorModel,
    component: int,
    *,
    n_vars: int,
) -> RationalPolynomial:
    terms: dict[tuple[int, ...], Fraction] = {}
    coeffs = model.poly.coeffs[0, int(component)].detach().cpu().tolist()
    for exponent, coefficient in zip(model.poly.basis.exponent_to_index, coeffs, strict=True):
        if coefficient != 0.0:
            terms[tuple(exponent) + (0,) * (n_vars - len(exponent))] = Fraction.from_float(coefficient)
    return RationalPolynomial(n_vars, terms)


def _variable(n_vars: int, index: int) -> RationalPolynomial:
    exponent = [0] * n_vars
    exponent[int(index)] = 1
    return RationalPolynomial(n_vars, {tuple(exponent): Fraction(1)})


def _domain(model: BatchedTaylorModel, *, remainder_symbols: bool) -> tuple[RationalInterval, ...]:
    result = tuple(
        RationalInterval(_q(model.domain_lo[0, index]), _q(model.domain_hi[0, index]))
        for index in range(model.domain_lo.shape[1])
    )
    if remainder_symbols:
        result += (
            RationalInterval(_q(model.rem_lo[0, 0]), _q(model.rem_hi[0, 0])),
            RationalInterval(_q(model.rem_lo[0, 1]), _q(model.rem_hi[0, 1])),
        )
    return result


def _stage_oracle(
    stage_id: str,
    model: BatchedTaylorModel,
    semantic: Sequence[RationalPolynomial],
    domain: Sequence[RationalInterval],
    *,
    source: Mapping[str, Any],
    operator_terms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(semantic) != model.poly.out_dim:
        raise ValueError("semantic/model component mismatch")
    components = []
    for component, exact in enumerate(semantic):
        represented = _poly_from_model(model, component, n_vars=exact.n_vars)
        residual = exact - represented
        global_enclosure = residual.bernstein_range(domain)
        production = RationalInterval(_q(model.rem_lo[0, component]), _q(model.rem_hi[0, component]))
        eps_q = Fraction.from_float(EPS)
        with_eps = RationalInterval(production.lo - eps_q, production.hi + eps_q)
        enclosure = global_enclosure
        subdivision: list[int] = []
        if not enclosure.subseteq(with_eps):
            subdivision = [0, 1, 2]
            enclosure = _subdivided_bernstein(
                residual,
                domain,
                split_variables=subdivision,
            )
        if not enclosure.subseteq(with_eps) and residual.n_vars == 5:
            subdivision = [0, 1, 2, 3, 4]
            enclosure = _subdivided_bernstein(
                residual,
                domain,
                split_variables=subdivision,
            )
        contained = enclosure.subseteq(production)
        contained_with_eps = enclosure.subseteq(with_eps)
        components.append(
            {
                "component": component,
                "semantic_polynomial_term_count": len(exact.terms),
                "represented_polynomial_term_count": len(represented.terms),
                "exact_residual_term_count": len(residual.terms),
                "exact_bernstein_global_residual": _rational_interval(global_enclosure),
                "exact_bernstein_residual": _rational_interval(enclosure),
                "exact_bernstein_subdivision_variables": [
                    (VARIABLES_5 if residual.n_vars == 5 else VARIABLES_3)[index]
                    for index in subdivision
                ],
                "production_ordinary_remainder": _rational_interval(production),
                "ordinary_remainder_alone_contains_exact_bernstein": contained,
                "downstream_validation_eps_reserve": _number(EPS),
                "production_plus_single_eps_reserve": _rational_interval(with_eps),
                "production_contains_exact_bernstein": contained_with_eps,
            }
        )
    return {
        "stage_id": stage_id,
        "source": dict(source),
        "model": _model_payload(model),
        "operator_terms": dict(operator_terms or {}),
        "oracle": {
            "method": "exact_binary64_rationals_tensor_product_bernstein",
            "rounding_ownership": (
                "ordinary remainder plus one explicit downstream validation_eps reserve; "
                "the reserve is a non-additive local audit witness, not injected once per stage"
            ),
            "variables": list(VARIABLES_5 if semantic[0].n_vars == 5 else VARIABLES_3),
            "components": components,
            "all_components_contained": all(row["production_contains_exact_bernstein"] for row in components),
        },
    }


def _subdivided_bernstein(
    polynomial: RationalPolynomial,
    domain: Sequence[RationalInterval],
    *,
    split_variables: Sequence[int],
) -> RationalInterval:
    boxes: list[list[RationalInterval]] = [list(domain)]
    for variable in split_variables:
        next_boxes: list[list[RationalInterval]] = []
        for box in boxes:
            interval = box[int(variable)]
            midpoint = (interval.lo + interval.hi) / 2
            for half in (RationalInterval(interval.lo, midpoint), RationalInterval(midpoint, interval.hi)):
                child = list(box)
                child[int(variable)] = half
                next_boxes.append(child)
        boxes = next_boxes
    ranges = [polynomial.bernstein_range(box) for box in boxes]
    return RationalInterval(min(item.lo for item in ranges), max(item.hi for item in ranges))


def _multiplication_terms(
    left: BatchedTaylorModel,
    right: BatchedTaylorModel,
    *,
    max_degree: int | None,
) -> dict[str, Any]:
    basis = left.poly.basis
    _, _, _, dropped_left, dropped_right, merge, unique_exp = basis.multiplication_plan_for_degree(max_degree)
    products = (
        left.poly.coeffs.index_select(-1, dropped_left)
        * right.poly.coeffs.index_select(-1, dropped_right)
    ).detach().cpu()
    before = []
    for route in range(dropped_left.numel()):
        for component in range(left.poly.out_dim):
            coefficient = float(products[0, component, route])
            if coefficient == 0.0:
                continue
            before.append(
                {
                    "component": component,
                    "left_exponents": basis.exponents[int(dropped_left[route])].detach().cpu().tolist(),
                    "right_exponents": basis.exponents[int(dropped_right[route])].detach().cpu().tolist(),
                    "dropped_exponents": unique_exp[int(merge[route])].detach().cpu().tolist(),
                    "binary64_product": _number(coefficient),
                }
            )
    after = []
    for component in range(left.poly.out_dim):
        for merged_index, exponent in enumerate(unique_exp.detach().cpu().tolist()):
            selected = products[0, component, merge.detach().cpu() == merged_index]
            coefficient = float(torch.sum(selected)) if selected.numel() else 0.0
            if coefficient != 0.0:
                after.append(
                    {
                        "component": component,
                        "dropped_exponents": exponent,
                        "binary64_merged_coefficient": _number(coefficient),
                        "incoming_nonzero_route_count": sum(
                            row["component"] == component and row["dropped_exponents"] == exponent
                            for row in before
                        ),
                    }
                )
    return {
        "max_degree": basis.order if max_degree is None else int(max_degree),
        "dropped_before_equal_exponent_merge": before,
        "dropped_after_equal_exponent_merge": after,
        "structural_dropped_route_count": int(dropped_left.numel()),
    }


def _cutoff_terms(before: BatchedTaylorModel, after: BatchedTaylorModel) -> dict[str, Any]:
    removed = []
    for component in range(before.poly.out_dim):
        for slot, exponent in enumerate(before.poly.basis.exponent_to_index):
            left = float(before.poly.coeffs[0, component, slot])
            right = float(after.poly.coeffs[0, component, slot])
            if left != right:
                removed.append(
                    {
                        "component": component,
                        "exponents": list(exponent),
                        "before": _number(left),
                        "after": _number(right),
                    }
                )
    return {"threshold": _number(CUTOFF), "removed_terms": removed}


def _build_base() -> tuple[PolynomialODE, FlowstarNormalFlowpipeState, BatchedTaylorModel]:
    contract = authoritative.load_contract()
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    state = FlowstarNormalFlowpipeState.from_exact_decimal_box(
        [("11/10", "7/5"), ("47/20", "49/20")], ORDER
    )
    policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )
    base = sparse_tmvector_to_dense(
        state.normalized_initial_tm(ORDER).extend_domain(Interval(0.0, H)),
        order=ORDER,
        device="cpu",
        dtype=torch.float64,
        counters=DenseExecutionCounters(),
        segment_boundary=True,
        range_policy=policy,
        range_trace=[],
    )
    return ode, state, base


def _picard_iteration_ledger(
    ode: PolynomialODE,
    base: BatchedTaylorModel,
) -> tuple[BatchedTaylorModel, list[dict[str, Any]]]:
    observed: list[tuple[int, BatchedTaylorModel, BatchedTaylorModel]] = []
    candidate, _ = dense_polynomial_picard(
        ode,
        base.without_remainder(),
        tau_index=2,
        order=ORDER,
        iterations=ORDER,
        cutoff_threshold=CUTOFF,
        observer=lambda iteration, pre, post: observed.append((iteration, pre, post)),
    )
    source_mul = _source(BatchedTaylorModel.mul_trunc)
    source_integrate = _source(BatchedTaylorModel.integrate)
    source_add = _source(BatchedTaylorModel.add)
    source_cutoff = _source(BatchedTaylorModel.apply_cutoff)
    base_zero = base.without_remainder()
    g = base_zero
    rows: list[dict[str, Any]] = []
    for iteration, observed_pre, observed_post in observed:
        x = g.component(0)
        y = g.component(1)
        domain = _domain(g, remainder_symbols=True)
        px = _poly_from_model(x, 0, n_vars=5) + _variable(5, 3)
        py = _poly_from_model(y, 0, n_vars=5) + _variable(5, 4)
        y_minus_x = y.sub(x)
        rows.append(
            _stage_oracle(
                f"picard.i{iteration}.y_minus_x",
                y_minus_x,
                [py - px],
                domain,
                source=_source(BatchedTaylorModel.sub),
            )
        )
        x_squared = x.mul_trunc(x)
        rows.append(
            _stage_oracle(
                f"picard.i{iteration}.x_squared",
                x_squared,
                [px * px],
                domain,
                source=source_mul,
                operator_terms=_multiplication_terms(x, x, max_degree=None),
            )
        )
        cubic = x_squared.mul_trunc(y)
        rows.append(
            _stage_oracle(
                f"picard.i{iteration}.x_squared_times_y",
                cubic,
                [px * px * py],
                domain,
                source=source_mul,
                operator_terms=_multiplication_terms(x_squared, y, max_degree=None),
            )
        )
        y_rhs = y_minus_x.sub(cubic)
        x_rhs = y
        rhs = BatchedTaylorModel.concat([x_rhs, y_rhs])
        semantic_rhs = [py, py - px - px * px * py]
        rows.append(
            _stage_oracle(
                f"picard.i{iteration}.rhs_aggregate",
                rhs,
                semantic_rhs,
                domain,
                source=_source(BatchedTaylorModel.sub),
            )
        )
        integrated = rhs.integrate(2)
        semantic_integrated = [value.integrate(2)[0] for value in semantic_rhs]
        rows.append(
            _stage_oracle(
                f"picard.i{iteration}.tau_integration",
                integrated,
                semantic_integrated,
                domain,
                source=source_integrate,
                operator_terms={
                    "integration_variable": "tau",
                    "integration_overflow": integrated.ledger.intervals().get("integration_overflow"),
                },
            )
        )
        picard = base_zero.add(integrated)
        semantic_picard = [
            _poly_from_model(base_zero, component, n_vars=5) + semantic_integrated[component]
            for component in range(2)
        ]
        rows.append(
            _stage_oracle(
                f"picard.i{iteration}.base_remainder_add",
                picard,
                semantic_picard,
                domain,
                source=source_add,
                operator_terms={"base_remainder": _model_payload(base_zero)["components"]},
            )
        )
        zeros = torch.zeros_like(picard.rem_lo)
        pre_cutoff = BatchedTaylorModel(
            picard.poly,
            zeros,
            zeros.clone(),
            picard.domain_lo,
            picard.domain_hi,
            range_policy=picard.range_policy,
            range_trace=picard.range_trace,
        )
        next_g = pre_cutoff.apply_cutoff(CUTOFF)
        if not _dense_equal(picard, observed_pre) or not _dense_equal(next_g, observed_post):
            raise RuntimeError(f"manual Picard stage replay differs at iteration {iteration}")
        rows.append(
            _stage_oracle(
                f"picard.i{iteration}.cutoff",
                next_g,
                [_poly_from_model(pre_cutoff, component, n_vars=5) for component in range(2)],
                domain,
                source=source_cutoff,
                operator_terms=_cutoff_terms(pre_cutoff, next_g),
            )
        )
        g = next_g
    if not _dense_equal(g, candidate):
        raise RuntimeError("manual Picard candidate differs from production")
    return candidate, rows


def _raw_cell(
    mode: str,
    ode: PolynomialODE,
    base: BatchedTaylorModel,
    candidate: BatchedTaylorModel,
) -> tuple[dict[str, Any], BatchedTaylorModel]:
    target_lo = torch.full_like(candidate.rem_lo, -TARGET)
    target_hi = torch.full_like(candidate.rem_hi, TARGET)
    target = candidate.with_remainder(target_lo, target_hi, category="initial_remainder")
    x = target.component(0)
    y = target.component(1)
    px = _poly_from_model(x, 0, n_vars=5) + _variable(5, 3)
    py = _poly_from_model(y, 0, n_vars=5) + _variable(5, 4)
    domain = _domain(target, remainder_symbols=True)
    rows: list[dict[str, Any]] = []
    if mode == LEGACY:
        y_minus_x = y.sub(x)
        rows.append(_stage_oracle("raw.B1.y_minus_x", y_minus_x, [py - px], domain, source=_source(BatchedTaylorModel.sub)))
        square = x.mul_trunc(x, max_degree=RAW_ORDER).apply_cutoff(CUTOFF)
        rows.append(
            _stage_oracle(
                "raw.B1.x_squared",
                square,
                [px * px],
                domain,
                source=_source(BatchedTaylorModel.mul_trunc),
                operator_terms=_multiplication_terms(x, x, max_degree=RAW_ORDER),
            )
        )
        product = square.mul_trunc(y, max_degree=RAW_ORDER).apply_cutoff(CUTOFF)
        rows.append(
            _stage_oracle(
                "raw.B1.x_squared_times_y",
                product,
                [px * px * py],
                domain,
                source=_source(BatchedTaylorModel.mul_trunc),
                operator_terms=_multiplication_terms(square, y, max_degree=RAW_ORDER),
            )
        )
        y_rhs = y_minus_x.sub(product)
        rows.append(_stage_oracle("raw.B1.distributed_final", y_rhs, [py - px - px * px * py], domain, source=_source(BatchedTaylorModel.sub)))
        evaluation = "ordered_terms"
    elif mode == H2:
        square = x.square_trunc_dependency_preserving(max_degree=RAW_ORDER).apply_cutoff(CUTOFF)
        rows.append(
            _stage_oracle(
                "raw.B2.x_squared_joint",
                square,
                [px * px],
                domain,
                source=_source(BatchedTaylorModel.square_trunc_dependency_preserving),
                operator_terms=_multiplication_terms(x, x, max_degree=RAW_ORDER),
            )
        )
        factor = 1.0 - square
        rows.append(_stage_oracle("raw.B2.one_minus_x_squared", factor, [RationalPolynomial.constant(5, 1) - px * px], domain, source=_source(BatchedTaylorModel.sub)))
        product = factor.mul_trunc(y, max_degree=RAW_ORDER).apply_cutoff(CUTOFF)
        rows.append(
            _stage_oracle(
                "raw.B2.factor_times_y",
                product,
                [(RationalPolynomial.constant(5, 1) - px * px) * py],
                domain,
                source=_source(BatchedTaylorModel.mul_trunc),
                operator_terms=_multiplication_terms(factor, y, max_degree=RAW_ORDER),
            )
        )
        y_rhs = product.sub(x)
        rows.append(_stage_oracle("raw.B2.factorized_final", y_rhs, [(RationalPolynomial.constant(5, 1) - px * px) * py - px], domain, source=_source(BatchedTaylorModel.sub)))
        evaluation = "canonical_factorized_joint"
    else:
        raise ValueError(mode)
    manual = BatchedTaylorModel.concat([y, y_rhs])
    production = _call_dense_raw_trace_rhs(
        ode,
        target,
        effective_order=RAW_ORDER,
        cutoff_threshold=CUTOFF,
        evaluation_mode=evaluation,
    )
    if not _dense_equal(manual, production):
        raise RuntimeError(f"manual raw replay differs for {mode}")

    raw_lo = production.rem_lo
    raw_hi = production.rem_hi
    tau_lo = target.domain_lo[:, 2].view(-1, 1)
    tau_hi = target.domain_hi[:, 2].view(-1, 1)
    scaled_lo, scaled_hi = _interval_mul(tau_lo, tau_hi, raw_lo, raw_hi)
    scaled_before_eps = (scaled_lo.clone(), scaled_hi.clone())
    scaled_lo, scaled_hi = _inflate_tensor_interval(scaled_lo, scaled_hi, EPS)
    base_added_lo, base_added_hi = _interval_add(base.rem_lo, base.rem_hi, scaled_lo, scaled_hi)
    regular = _call_dense_rhs_evaluation(ode, target, evaluation_mode=evaluation)
    tmp = base.add(regular.integrate(2)).apply_cutoff(CUTOFF)
    poly_diff = tmp.poly.sub(candidate.poly)
    diff_lo, diff_hi = poly_diff.range_bound(
        candidate.domain_lo,
        candidate.domain_hi,
        policy=candidate.range_policy,
        context="picard_polynomial_difference",
        trace=candidate.range_trace,
    )
    diff_before_eps = (diff_lo.clone(), diff_hi.clone())
    diff_exact_rows = []
    domain3 = _domain(candidate, remainder_symbols=False)
    for component in range(2):
        polynomial = RationalPolynomial(
            3,
            {
                tuple(exponent): Fraction.from_float(float(poly_diff.coeffs[0, component, slot]))
                for slot, exponent in enumerate(poly_diff.basis.exponent_to_index)
                if float(poly_diff.coeffs[0, component, slot]) != 0.0
            },
        )
        exact = _subdivided_bernstein(polynomial, domain3, split_variables=(0, 1))
        production_range = RationalInterval(_q(diff_lo[0, component]), _q(diff_hi[0, component]))
        diff_exact_rows.append(
            {
                "component": component,
                "exact_bernstein": _rational_interval(exact),
                "exact_bernstein_subdivision": "ux and uy bisected once; four rational boxes",
                "production_range": _rational_interval(production_range),
                "production_contains_exact_bernstein": exact.subseteq(production_range),
            }
        )
    diff_lo, diff_hi = _inflate_tensor_interval(diff_lo, diff_hi, EPS)
    assembled_lo, assembled_hi = _interval_add(base_added_lo, base_added_hi, diff_lo, diff_hi)
    final_before_eps = (assembled_lo.clone(), assembled_hi.clone())
    final_lo, final_hi = _inflate_tensor_interval(assembled_lo, assembled_hi, EPS)

    step = dense_picard_validate_step(
        ode,
        base,
        h=H,
        order=ORDER,
        tau_index=2,
        target_remainder_radius=TARGET,
        cutoff_threshold=CUTOFF,
        max_validation_attempts=2,
        validation_eps=EPS,
        validation_mode=mode,
    )
    trace = [row for row in step.trace if row.get("phase") == "remainder_validation"][-1]
    if trace["picard_image_remainder_lo"] != final_lo.detach().cpu().tolist() or trace["picard_image_remainder_hi"] != final_hi.detach().cpu().tolist():
        raise RuntimeError(f"manual final image differs for {mode}")
    segment_poly_lo, segment_poly_hi = step.segment_tm.poly.range_bound(
        step.segment_tm.domain_lo,
        step.segment_tm.domain_hi,
        policy=step.segment_tm.range_policy,
        context="retained_polynomial",
        trace=None,
    )
    segment_lo, segment_hi = _interval_add(segment_poly_lo, segment_poly_hi, step.segment_tm.rem_lo, step.segment_tm.rem_hi)
    endpoint = step.segment_tm.endpoint(2, H)
    endpoint_poly_lo, endpoint_poly_hi = endpoint.poly.range_bound(
        endpoint.domain_lo,
        endpoint.domain_hi,
        policy=endpoint.range_policy,
        context="retained_polynomial",
        trace=None,
    )
    endpoint_lo, endpoint_hi = _interval_add(endpoint_poly_lo, endpoint_poly_hi, endpoint.rem_lo, endpoint.rem_hi)
    return (
        {
            "mode": mode,
            "same_input_sha256": _sha(_model_payload(target)),
            "same_input_byte_identity": True,
            "counterfactual": mode == H2,
            "eligible_for_production": mode == H2 and all(row["oracle"]["all_components_contained"] for row in rows),
            "operator_stages": rows,
            "raw_rhs_remainder": [_interval(raw_lo[0, i], raw_hi[0, i]) for i in range(2)],
            "tau_times_raw_rhs": {
                "before_validation_eps": [_interval(scaled_before_eps[0][0, i], scaled_before_eps[1][0, i]) for i in range(2)],
                "after_validation_eps": [_interval(scaled_lo[0, i], scaled_hi[0, i]) for i in range(2)],
                "validation_eps": _number(EPS),
            },
            "base_remainder_add": {
                "input_base_remainder": [_interval(base.rem_lo[0, i], base.rem_hi[0, i]) for i in range(2)],
                "output": [_interval(base_added_lo[0, i], base_added_hi[0, i]) for i in range(2)],
                "validation_eps_at_this_stage": None,
            },
            "poly_diff_range_bound": {
                "source": _source(type(poly_diff).range_bound),
                "components": diff_exact_rows,
            },
            "poly_diff_validation_eps": {
                "before": [_interval(diff_before_eps[0][0, i], diff_before_eps[1][0, i]) for i in range(2)],
                "after": [_interval(diff_lo[0, i], diff_hi[0, i]) for i in range(2)],
                "validation_eps": _number(EPS),
            },
            "final_validation_eps": {
                "before": [_interval(final_before_eps[0][0, i], final_before_eps[1][0, i]) for i in range(2)],
                "after": [_interval(final_lo[0, i], final_hi[0, i]) for i in range(2)],
                "validation_eps": _number(EPS),
            },
            "subset_test": {
                "target": [_interval(-TARGET, TARGET) for _ in range(2)],
                "margin": step.subset_margin.detach().cpu().tolist(),
                "accepted": step.status == "validated",
            },
            "segment": [_interval(segment_lo[0, i], segment_hi[0, i]) for i in range(2)],
            "endpoint": [_interval(endpoint_lo[0, i], endpoint_hi[0, i]) for i in range(2)],
        },
        step.segment_tm,
    )


def _git_source(flowstar_root: Path, path: str) -> tuple[str, list[str]]:
    text = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=flowstar_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return hashlib.sha256(text.encode()).hexdigest(), text.splitlines()


def _snippet(lines: Sequence[str], start: int, end: int) -> dict[str, Any]:
    return {
        "line_start": start,
        "line_end": end,
        "text": "\n".join(f"{index}: {lines[index - 1]}" for index in range(start, end + 1)),
    }


def _flowstar_semantics(flowstar_root: Path) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD", root=flowstar_root)
    files: dict[str, Any] = {}
    for path, windows in {
        "benchmarks/continuous/vanderpol/vanderpol.cpp": ((23, 27),),
        "flowstar-toolbox/TaylorModel.h": ((683, 713), (3707, 3749)),
        "flowstar-toolbox/expression.h": ((1833, 1913),),
        "flowstar-toolbox/Continuous.cpp": ((960, 1025),),
    }.items():
        digest, lines = _git_source(flowstar_root, path)
        files[path] = {"git_blob_text_sha256": digest, "snippets": [_snippet(lines, a, b) for a, b in windows]}
    return {
        "source": "stock Flow* Git objects; working-tree bytes were not read",
        "head": head,
        "working_tree_clean_required": False,
        "files": files,
        "derived_semantics": {
            "vdp_expression": "(1 - x^2) * y - x",
            "evaluation_tree": ["x^2", "1-x^2", "(1-x^2)*y", "((1-x^2)*y)-x"],
            "effective_rhs_order": "k = order - 1",
            "multiplication_remainder": "P_left*R_right + P_right*R_left + R_left*R_right",
            "truncation_and_cutoff": "after each expression multiplication",
            "intermediate_ranges": "polynomial operand ranges and truncation intervals pushed during evaluate and replayed by evaluate_remainder",
            "time_integration": "replayed RHS remainder multiplied once by timeStep",
            "subset": "new remainder subseteq current candidate remainder",
        },
    }


def _flowstar_runtime_crosscheck() -> dict[str, Any]:
    data = json.loads(FLOWSTAR_LEDGER.read_text(encoding="utf-8"))
    rows = data["rows"]
    raw_rows = [
        row["payload"]
        for row in rows
        if row["stage_id"] == "refinement_raw_image"
        and row["payload"]["refinement_iteration"] == 0
        and row["payload"]["component"] in (0, 1)
    ]
    raw_rows.sort(key=lambda row: row["component"])
    ranges = {}
    for kind in ("segment", "endpoint"):
        row = next(row for row in rows if row["stage_id"] == f"{kind}_polynomial_and_final_range")
        ranges[kind] = [
            {
                "lo": float(item["lower"]["decimal"]),
                "hi": float(item["upper"]["decimal"]),
                "width": float(item["upper"]["decimal"]) - float(item["lower"]["decimal"]),
            }
            for item in row["payload"]["final"][:2]
        ]
    return {
        "role": "runtime cross-check only; never used for soundness",
        "source_ledger": str(FLOWSTAR_LEDGER.relative_to(ROOT)),
        "source_commit": raw_rows[0]["source_commit"],
        "raw_image_target_remainder_first_iteration": [
            {
                "lo": float(row["interval"]["lower"]["decimal"]),
                "hi": float(row["interval"]["upper"]["decimal"]),
                "width": float(row["interval"]["upper"]["decimal"])
                - float(row["interval"]["lower"]["decimal"]),
            }
            for row in raw_rows
        ],
        **ranges,
    }


def _width(record: Mapping[str, Any]) -> float:
    return float(record["hi"]["decimal"]) - float(record["lo"]["decimal"])


def run(output_dir: Path, flowstar_root: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ode, state, base = _build_base()
    candidate, picard_rows = _picard_iteration_ledger(ode, base)
    b1, _ = _raw_cell(LEGACY, ode, base, candidate)
    b2, _ = _raw_cell(H2, ode, base, candidate)
    if b1["same_input_sha256"] != b2["same_input_sha256"]:
        raise RuntimeError("Gate B cells did not start from byte-identical input")
    flow_source = _flowstar_semantics(flowstar_root)
    flow_runtime = _flowstar_runtime_crosscheck()

    radius = Fraction.from_float(TARGET)
    # For a shared scalar r, squaring the exact interval is stronger than the
    # unsplit Bernstein convex hull on a sign-changing box and is exact here.
    necessary_r_square = RationalInterval(Fraction(0), radius * radius)
    production_r_square = RationalInterval(-radius * radius, radius * radius)
    first_loss = {
        "stage": "raw.B1.x_squared",
        "function": "_DenseRawTraceScalar.__mul__ -> BatchedTaylorModel.mul_trunc",
        "binary_operation": "the same x Taylor model multiplied by itself",
        "specific_interval_term": "R_left * R_right with R_left and R_right actually the same R_x",
        "production_independent_interval": _rational_interval(production_r_square),
        "exact_shared_symbol_bernstein": _rational_interval(necessary_r_square),
        "strict_extra_lower_width": fraction_text(necessary_r_square.lo - production_r_square.lo),
        "classification": "FIRST_STRICTLY_WIDER_ENCLOSURE",
        "flowstar_source_equivalent_note": "stock expression retains x^2 as one AST subtree, but its generic remainder replay also uses interval multiplication; Flow* runtime is not the soundness oracle",
    }

    channels = ("x", "y")
    raw_metrics = {}
    for index, channel in enumerate(channels):
        legacy_width = _width(b1["final_validation_eps"]["after"][index])
        candidate_width = _width(b2["final_validation_eps"]["after"][index])
        flow_width = flow_runtime["raw_image_target_remainder_first_iteration"][index]["width"]
        excess = legacy_width - flow_width
        removed = legacy_width - candidate_width
        raw_metrics[channel] = {
            "flowstar_crosscheck_width": flow_width,
            "legacy_width": legacy_width,
            "candidate_width": candidate_width,
            "legacy_excess": excess,
            "removed": removed,
            "fraction_of_legacy_excess_removed": None if excess <= 0 else removed / excess,
            "no_regression": candidate_width <= legacy_width,
        }
    segment_metrics = {}
    for index, channel in enumerate(channels):
        legacy_width = _width(b1["segment"][index])
        candidate_width = _width(b2["segment"][index])
        flow_width = flow_runtime["segment"][index]["width"]
        excess = legacy_width - flow_width
        removed = legacy_width - candidate_width
        segment_metrics[channel] = {
            "flowstar_crosscheck_width": flow_width,
            "legacy_width": legacy_width,
            "candidate_width": candidate_width,
            "legacy_excess": excess,
            "removed": removed,
            "fraction_of_legacy_excess_removed": None if excess <= 0 else removed / excess,
            "no_regression": candidate_width <= legacy_width,
        }

    all_stage_sound = all(row["oracle"]["all_components_contained"] for row in picard_rows)
    all_stage_sound = all_stage_sound and all(
        row["oracle"]["all_components_contained"]
        for cell in (b1, b2)
        for row in cell["operator_stages"]
    )
    all_poly_diff_sound = all(
        row["production_contains_exact_bernstein"]
        for cell in (b1, b2)
        for row in cell["poly_diff_range_bound"]["components"]
    )
    gate_b_pass = (
        b2["eligible_for_production"]
        and any(
            row["fraction_of_legacy_excess_removed"] is not None
            and row["fraction_of_legacy_excess_removed"] >= 0.10
            for row in raw_metrics.values()
        )
        and all(row["no_regression"] for row in raw_metrics.values())
        and all(row["no_regression"] for row in segment_metrics.values())
    )
    preregistration = {
        "B1": {"status": "executed_baseline", "operator": "distributed x*x*y"},
        "B2": {"status": "executed_pass" if gate_b_pass else "executed_fail", "operator": "canonical factorized expression plus joint shared square"},
        "B3": {"status": "not_executed_stop_after_first_pass", "operator": "tau dependency-preserving integration"},
        "B4": {"status": "not_executed_stop_after_first_pass", "operator": "poly_diff dependency-preserving range"},
    }
    ledger = {
        "schema": "vdp_h2_step1_complete_operator_ledger_v1",
        "contract": {
            "initial_box_exact_decimal": [["1.1", "1.4"], ["2.35", "2.45"]],
            "h_binary64": _number(H),
            "order": ORDER,
            "cutoff_binary64": _number(CUTOFF),
            "target_remainder_binary64": _number(TARGET),
            "validation_eps_binary64": _number(EPS),
            "initialization_diagnostics": dict(state.diagnostics or {}),
        },
        "production_source_commit": _git("rev-parse", "HEAD"),
        "working_diff_sha256": hashlib.sha256(
            subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True).stdout
        ).hexdigest(),
        "outward_exact_prestate": _model_payload(base),
        "picard_iterations": picard_rows,
        "gate_b_cells": {"B1": b1, "B2": b2},
    }
    _write_json(output_dir / "production_operator_ledger.json", ledger)
    _write_json(output_dir / "flowstar_source_semantics.json", flow_source)
    _write_json(output_dir / "flowstar_runtime_crosscheck.json", flow_runtime)
    _write_json(
        output_dir / "gate_b_same_input_matrix.json",
        {
            "schema": "vdp_h2_same_input_gate_b_v1",
            "preregistration": preregistration,
            "same_input_sha256": b1["same_input_sha256"],
            "raw_residual_excess": raw_metrics,
            "segment_excess": segment_metrics,
            "gate_b_pass": gate_b_pass,
        },
    )
    summary = {
        "schema": "vdp_h2_dense_picard_first_loss_gate_summary_v1",
        "first_extra_enclosure": first_loss,
        "picard_iteration_count": 4,
        "operator_stage_count": len(picard_rows) + len(b1["operator_stages"]) + len(b2["operator_stages"]),
        "all_operator_stages_exact_bernstein_contained": all_stage_sound,
        "all_poly_diff_exact_bernstein_contained": all_poly_diff_sound,
        "same_input_byte_identity": b1["same_input_sha256"] == b2["same_input_sha256"],
        "raw_residual_excess": raw_metrics,
        "segment_excess": segment_metrics,
        "preregistration": preregistration,
        "gate_a_pass": all_stage_sound and all_poly_diff_sound,
        "gate_b_pass": gate_b_pass,
        "production_candidate": H2 if gate_b_pass else None,
        "flowstar_source_commit": flow_source["head"],
        "flowstar_runtime_is_soundness_oracle": False,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["gate_a_pass"] or not summary["gate_b_pass"]:
        raise RuntimeError("H2 Gate A/B failed")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flowstar-root", type=Path, default=Path("/srv/local/shengenli/flowstar"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.output_dir, args.flowstar_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
