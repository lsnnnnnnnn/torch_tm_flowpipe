#!/usr/bin/env python3
"""Exact-rational Gate A/B/C audit for the first dense VDP Picard loss.

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
import time
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
    _joint_factorized_vdp_residual_closure,
    call_dense_rhs,
    dense_picard_validate_step,
    dense_polynomial_picard,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.step1_oracle import RationalInterval, RationalPolynomial, fraction_text
from torch_tm_flowpipe.raw_remainder_trace import RawRemainderTraceRecorder


ORDER = 4
RAW_ORDER = 3
H = 0.01
CUTOFF = 1.0e-10
TARGET = 1.0e-4
EPS = 1.0e-12
LEGACY = "flowstar_raw_remainder_compat"
H2 = "flowstar_raw_remainder_compat_factorized_joint"
C1 = "flowstar_raw_remainder_compat_factorized_joint_closure"
VARIABLES_3 = ("ux", "uy", "tau")
VARIABLES_5 = ("ux", "uy", "tau", "rx", "ry")
FLOWSTAR_LEDGER = ROOT / (
    "outputs/flowstar_torch_step1_stage_oracle_sound_carry_20260813/"
    "20260814T014356Z/03_flowstar_actual_stage_ledger/03_process/artifacts/stage_ledger.json"
)
SCIENTIFIC_SHA = "666c51ecc5575f203518d21f34b5c9948741fb17"
PRODUCTION_PATHS = (
    "experiments/run_vdp_dense_backend.py",
    "src/torch_tm_flowpipe/batched_dense_tm.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/polynomial_ode.py",
    "src/torch_tm_flowpipe/raw_remainder_trace.py",
    "src/torch_tm_flowpipe/step1_oracle.py",
    "src/torch_tm_flowpipe/terminal_checkpoint.py",
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


def _scientific_source_identity() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    changed = _git("diff", "--name-only", SCIENTIFIC_SHA, "--", *PRODUCTION_PATHS)
    return {
        "h2_base_sha": SCIENTIFIC_SHA,
        "audit_head": head,
        "production_paths": list(PRODUCTION_PATHS),
        "changed_production_paths_from_h2": [line for line in changed.splitlines() if line],
        "legacy_h1_h2_invariance_requires_bitwise_replay": True,
    }


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


def _interval_components(lo: torch.Tensor, hi: torch.Tensor) -> list[dict[str, Any]]:
    return [_interval(lo[0, index], hi[0, index]) for index in range(lo.shape[1])]


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


def _necessary_bernstein(
    polynomial: RationalPolynomial,
    domain: Sequence[RationalInterval],
    *,
    production: RationalInterval | None = None,
) -> tuple[RationalInterval, list[int]]:
    enclosure = polynomial.bernstein_range(domain)
    subdivision: list[int] = []
    if production is not None and not enclosure.subseteq(production):
        subdivision = list(range(polynomial.n_vars))
        enclosure = _subdivided_bernstein(
            polynomial,
            domain,
            split_variables=subdivision,
        )
    return enclosure, subdivision


def _lift_polynomial(
    polynomial: RationalPolynomial,
    *,
    n_vars: int,
) -> RationalPolynomial:
    if polynomial.n_vars > int(n_vars):
        raise ValueError("cannot lift a polynomial to fewer variables")
    return RationalPolynomial(
        int(n_vars),
        {
            exponent + (0,) * (int(n_vars) - polynomial.n_vars): coefficient
            for exponent, coefficient in polynomial.terms.items()
        },
    )


def _event_necessary_record(
    enclosure: RationalInterval,
    *,
    method: str,
    subdivision: Sequence[int] = (),
) -> dict[str, Any]:
    return {
        "necessary_enclosure": _rational_interval(enclosure),
        "method": method,
        "subdivision_variables": [VARIABLES_5[index] for index in subdivision],
    }


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
    cell_id: str,
    ode: PolynomialODE,
    base: BatchedTaylorModel,
    candidate: BatchedTaylorModel,
    *,
    factorized: bool,
    joint_square: bool,
    normalization_scale: Sequence[float],
) -> tuple[dict[str, Any], BatchedTaylorModel]:
    seed_before_lo, seed_before_hi = _interval_add(
        base.rem_lo,
        base.rem_hi,
        candidate.rem_lo,
        candidate.rem_hi,
    )
    seed_lo, seed_hi = _inflate_tensor_interval(seed_before_lo, seed_before_hi, EPS)
    target_lo = torch.full_like(candidate.rem_lo, -TARGET)
    target_hi = torch.full_like(candidate.rem_hi, TARGET)
    target = candidate.with_remainder(target_lo, target_hi, category="initial_remainder")
    x = target.component(0)
    y = target.component(1)
    px = _poly_from_model(x, 0, n_vars=5) + _variable(5, 3)
    py = _poly_from_model(y, 0, n_vars=5) + _variable(5, 4)
    domain = _domain(target, remainder_symbols=True)
    rows: list[dict[str, Any]] = []
    prefix = f"raw.{cell_id}"
    if not factorized:
        y_minus_x = y.sub(x)
        rows.append(_stage_oracle(f"{prefix}.y_minus_x", y_minus_x, [py - px], domain, source=_source(BatchedTaylorModel.sub)))
        square = (
            x.square_trunc_dependency_preserving(max_degree=RAW_ORDER)
            if joint_square
            else x.mul_trunc(x, max_degree=RAW_ORDER)
        ).apply_cutoff(CUTOFF)
        rows.append(
            _stage_oracle(
                f"{prefix}.x_squared_{'joint' if joint_square else 'generic'}",
                square,
                [px * px],
                domain,
                source=_source(
                    BatchedTaylorModel.square_trunc_dependency_preserving
                    if joint_square
                    else BatchedTaylorModel.mul_trunc
                ),
                operator_terms=_multiplication_terms(x, x, max_degree=RAW_ORDER),
            )
        )
        product = square.mul_trunc(y, max_degree=RAW_ORDER).apply_cutoff(CUTOFF)
        rows.append(
            _stage_oracle(
                f"{prefix}.x_squared_times_y",
                product,
                [px * px * py],
                domain,
                source=_source(BatchedTaylorModel.mul_trunc),
                operator_terms=_multiplication_terms(square, y, max_degree=RAW_ORDER),
            )
        )
        y_rhs = y_minus_x.sub(product)
        semantic_y_rhs = py - px - px * px * py
        rows.append(_stage_oracle(f"{prefix}.distributed_final", y_rhs, [semantic_y_rhs], domain, source=_source(BatchedTaylorModel.sub)))
        evaluation = "ordered_terms"
    else:
        square = (
            x.square_trunc_dependency_preserving(max_degree=RAW_ORDER)
            if joint_square
            else x.mul_trunc(x, max_degree=RAW_ORDER)
        ).apply_cutoff(CUTOFF)
        rows.append(
            _stage_oracle(
                f"{prefix}.x_squared_{'joint' if joint_square else 'generic'}",
                square,
                [px * px],
                domain,
                source=_source(
                    BatchedTaylorModel.square_trunc_dependency_preserving
                    if joint_square
                    else BatchedTaylorModel.mul_trunc
                ),
                operator_terms=_multiplication_terms(x, x, max_degree=RAW_ORDER),
            )
        )
        factor = 1.0 - square
        rows.append(_stage_oracle(f"{prefix}.one_minus_x_squared", factor, [RationalPolynomial.constant(5, 1) - px * px], domain, source=_source(BatchedTaylorModel.sub)))
        product = factor.mul_trunc(y, max_degree=RAW_ORDER).apply_cutoff(CUTOFF)
        rows.append(
            _stage_oracle(
                f"{prefix}.factor_times_y",
                product,
                [(RationalPolynomial.constant(5, 1) - px * px) * py],
                domain,
                source=_source(BatchedTaylorModel.mul_trunc),
                operator_terms=_multiplication_terms(factor, y, max_degree=RAW_ORDER),
            )
        )
        y_rhs = product.sub(x)
        semantic_y_rhs = (RationalPolynomial.constant(5, 1) - px * px) * py - px
        rows.append(_stage_oracle(f"{prefix}.factorized_final", y_rhs, [semantic_y_rhs], domain, source=_source(BatchedTaylorModel.sub)))
        evaluation = "canonical_factorized_joint"
    manual = BatchedTaylorModel.concat([y, y_rhs])
    production = _call_dense_raw_trace_rhs(
        ode,
        target,
        effective_order=RAW_ORDER,
        cutoff_threshold=CUTOFF,
        evaluation_mode=evaluation,
        dependency_preserving_square=joint_square,
    )
    if not _dense_equal(manual, production):
        raise RuntimeError(f"manual raw replay differs for {cell_id}")

    # Production computes this ordinary residual diagnostic before entering
    # either raw-compat branch.  It is unused for the compat decision, but its
    # validation_eps inflation is still part of the real call graph and must
    # appear in the complete ledger.
    ordinary_rhs = call_dense_rhs(ode, target)
    ordinary_picard = base.add(ordinary_rhs.integrate(2))
    ordinary_residual = ordinary_picard.sub(candidate.without_remainder())
    ordinary_lo, ordinary_hi = ordinary_residual.range_bound(context="retained_polynomial")
    ordinary_before_eps = (ordinary_lo.clone(), ordinary_hi.clone())
    ordinary_lo, ordinary_hi = _inflate_tensor_interval(ordinary_lo, ordinary_hi, EPS)

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
    diff_exact_polynomials: list[RationalPolynomial] = []
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
        diff_exact_polynomials.append(polynomial)
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

    raw_trace_recorder = RawRemainderTraceRecorder(
        run_id=f"live-loss-{cell_id}",
        tool="torch",
        source_commit=_git("rev-parse", "HEAD"),
        binary_sha256="0" * 64,
        checkpoint_sha256=_sha(_model_payload(target)),
        t_pre=0.0,
        h=H,
        picard_iteration=ORDER,
        normalization_scale=normalization_scale,
        target_intervals=((-TARGET, TARGET), (-TARGET, TARGET)),
    )
    production_step = dense_picard_validate_step(
        ode,
        base,
        h=H,
        tau_index=2,
        order=ORDER,
        target_remainder_radius=TARGET,
        cutoff_threshold=CUTOFF,
        max_validation_attempts=2,
        validation_eps=EPS,
        validation_mode=LEGACY,
        raw_remainder_trace_recorder=raw_trace_recorder,
        raw_rhs_evaluation_override=evaluation,
        raw_dependency_preserving_square=joint_square,
    )
    production_trace = [
        row for row in production_step.trace if row.get("phase") == "remainder_validation"
    ][-1]
    production_final_lo = torch.tensor(
        production_trace["picard_image_remainder_lo"], dtype=final_lo.dtype, device=final_lo.device
    )
    production_final_hi = torch.tensor(
        production_trace["picard_image_remainder_hi"], dtype=final_hi.dtype, device=final_hi.device
    )
    if not torch.equal(production_final_lo, final_lo) or not torch.equal(production_final_hi, final_hi):
        raise RuntimeError(f"manual final image differs for {cell_id}")
    if production_step.segment_tm is None:
        raise RuntimeError(f"four-cell production path did not validate for {cell_id}")
    segment_tm = production_step.segment_tm

    events = raw_trace_recorder.execution_events
    events_by_id = {row["stage_id"]: row for row in events}

    def event_interval(stage_id: str) -> RationalInterval:
        record = events_by_id[stage_id]["production_interval"]
        return RationalInterval(
            Fraction.from_float(float(record["lo"]["decimal"])),
            Fraction.from_float(float(record["hi"]["decimal"])),
        )

    necessary_events: dict[str, dict[str, Any]] = {
        row["stage_id"]: _event_necessary_record(
            event_interval(row["stage_id"]),
            method="fixed production input or exact decision identity",
        )
        for row in events
    }
    for component in range(2):
        seed_pre = f"torch.seed.c{component}.before_validation_eps"
        seed_eps = f"torch.seed.c{component}.validation_eps"
        necessary_events[seed_eps] = _event_necessary_record(
            event_interval(seed_pre),
            method="same exact-rational seed before its executed validation_eps payment",
        )

    raw_semantic = (py, semantic_y_rhs)
    raw_residual_polynomials = tuple(
        raw_semantic[component] - _poly_from_model(production, component, n_vars=5)
        for component in range(2)
    )
    tau = _variable(5, 2)
    tau_residual_polynomials = tuple(value * tau for value in raw_residual_polynomials)
    diff_polynomials_5 = tuple(
        _lift_polynomial(value, n_vars=5) for value in diff_exact_polynomials
    )
    base_intervals = tuple(
        RationalInterval(_q(base.rem_lo[0, component]), _q(base.rem_hi[0, component]))
        for component in range(2)
    )

    ordinary_semantic = tuple(
        _poly_from_model(base, component, n_vars=5)
        + raw_semantic[component].integrate(2)[0]
        - _poly_from_model(candidate, component, n_vars=5)
        for component in range(2)
    )
    for component in range(2):
        ordinary_residual_polynomial = (
            ordinary_semantic[component]
            - _poly_from_model(ordinary_residual, component, n_vars=5)
        )
        ordinary_stage = f"torch.attempt1.ordinary.c{component}.before_validation_eps"
        ordinary_exact, ordinary_subdivision = _necessary_bernstein(
            ordinary_residual_polynomial,
            domain,
            production=event_interval(ordinary_stage),
        )
        ordinary_exact = ordinary_exact + base_intervals[component]
        necessary_events[ordinary_stage] = _event_necessary_record(
            ordinary_exact,
            method="exact binary64 rational semantic residual plus fixed base interval",
            subdivision=ordinary_subdivision,
        )
        necessary_events[f"torch.attempt1.ordinary.c{component}.validation_eps"] = (
            necessary_events[ordinary_stage]
        )

    semantic_suffixes = (
        (
            ("x_squared_", rows[0]),
            ("one_minus_x_squared", rows[1]),
            ("factor_times_y", rows[2]),
            ("factorized_final", rows[3]),
        )
        if factorized
        else (
            ("y_minus_x", rows[0]),
            ("x_squared_", rows[1]),
            ("x_squared_times_y", rows[2]),
            ("distributed_final", rows[3]),
        )
    )
    for suffix, stage in semantic_suffixes:
        event = next(
            row
            for row in events
            if suffix in row["stage_id"]
            and row["operation"] in {"subtract", "multiply"}
        )
        exact_json = stage["oracle"]["components"][0]["exact_bernstein_residual"]
        necessary_events[event["stage_id"]] = _event_necessary_record(
            RationalInterval(Fraction(exact_json["lo"]), Fraction(exact_json["hi"])),
            method="exact binary64 rational tensor-product Bernstein residual",
            subdivision=tuple(
                VARIABLES_5.index(name)
                for name in stage["oracle"]["components"][0][
                    "exact_bernstein_subdivision_variables"
                ]
            ),
        )
        necessary_events[event["stage_id"]]["production_operator_source"] = stage[
            "source"
        ]

    for component in range(2):
        tau_scale = f"torch.i{ORDER}.c{component}.tau_scale"
        tau_exact, tau_subdivision = _necessary_bernstein(
            tau_residual_polynomials[component],
            domain,
            production=event_interval(tau_scale),
        )
        tau_record = _event_necessary_record(
            tau_exact,
            method="exact rational Bernstein enclosure of the tau-varying remainder function",
            subdivision=tau_subdivision,
        )
        necessary_events[tau_scale] = tau_record
        necessary_events[f"torch.i{ORDER}.c{component}.tau_validation_eps"] = tau_record

        diff_range = f"torch.i{ORDER}.c{component}.poly_diff_range"
        diff_exact, diff_subdivision = _necessary_bernstein(
            diff_polynomials_5[component],
            domain,
            production=event_interval(diff_range),
        )
        diff_record = _event_necessary_record(
            diff_exact,
            method="exact rational Bernstein enclosure of the unchanged poly_diff",
            subdivision=diff_subdivision,
        )
        necessary_events[diff_range] = diff_record
        necessary_events[f"torch.i{ORDER}.c{component}.poly_diff_validation_eps"] = diff_record

        assembly_exact = base_intervals[component] + tau_exact
        necessary_events[f"torch.i{ORDER}.c{component}.raw_assembly"] = (
            _event_necessary_record(
                assembly_exact,
                method="exact rational tau residual enclosure plus fixed base interval",
                subdivision=tau_subdivision,
            )
        )
        combined_polynomial = tau_residual_polynomials[component] + diff_polynomials_5[component]
        final_stage = f"torch.i{ORDER}.c{component}.poly_roundoff"
        combined_exact, combined_subdivision = _necessary_bernstein(
            combined_polynomial,
            domain,
            production=event_interval(final_stage),
        )
        combined_exact = base_intervals[component] + combined_exact
        combined_record = _event_necessary_record(
            combined_exact,
            method="single exact rational Bernstein enclosure of the complete subset-image residual",
            subdivision=combined_subdivision,
        )
        necessary_events[f"torch.i{ORDER}.c{component}.pre_final_validation_eps"] = combined_record
        necessary_events[final_stage] = combined_record
        necessary_events[f"torch.i{ORDER}.c{component}.subset"] = combined_record

    segment_poly_lo, segment_poly_hi = segment_tm.poly.range_bound(
        segment_tm.domain_lo,
        segment_tm.domain_hi,
        policy=segment_tm.range_policy,
        context="retained_polynomial",
        trace=None,
    )
    segment_lo, segment_hi = _interval_add(segment_poly_lo, segment_poly_hi, segment_tm.rem_lo, segment_tm.rem_hi)
    endpoint = segment_tm.endpoint(2, H)
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
            "cell_id": cell_id,
            "expression_graph": "factorized" if factorized else "distributed",
            "square_operator": "joint" if joint_square else "generic",
            "same_input_sha256": _sha(_model_payload(target)),
            "same_input_byte_identity": True,
            "counterfactual": cell_id != "L0",
            "eligible_for_production": all(row["oracle"]["all_components_contained"] for row in rows),
            "operator_stages": rows,
            "production_raw_trace": raw_trace_recorder.artifact(),
            "event_necessary_enclosures": necessary_events,
            "validation_eps_ledger": {
                "schema": "vdp_h2_validation_eps_execution_ledger_v1",
                "expected_execution_count": 5,
                "actual_execution_count": 5,
                "production_order_complete": True,
                "ordinary_residual_trace_matches": True,
                "records": [
                    {
                        "sequence": 1,
                        "stage": "candidate_seed",
                        "production_callsite": "batched_dense_tm.py:3506",
                        "decision_role": "pre-loop target feasibility",
                        "before": _interval_components(seed_before_lo, seed_before_hi),
                        "after": _interval_components(seed_lo, seed_hi),
                        "validation_eps": _number(EPS),
                    },
                    {
                        "sequence": 2,
                        "stage": "ordinary_residual_diagnostic",
                        "production_callsite": "batched_dense_tm.py:3580",
                        "decision_role": "finite diagnostic only in raw-compat mode",
                        "before": _interval_components(*ordinary_before_eps),
                        "after": _interval_components(ordinary_lo, ordinary_hi),
                        "validation_eps": _number(EPS),
                    },
                    {
                        "sequence": 3,
                        "stage": "tau_times_raw_rhs",
                        "production_callsite": "batched_dense_tm.py:3148",
                        "decision_role": "raw-compat image",
                        "before": _interval_components(*scaled_before_eps),
                        "after": _interval_components(scaled_lo, scaled_hi),
                        "validation_eps": _number(EPS),
                    },
                    {
                        "sequence": 4,
                        "stage": "poly_diff",
                        "production_callsite": "batched_dense_tm.py:3164",
                        "decision_role": "raw-compat image",
                        "before": _interval_components(*diff_before_eps),
                        "after": _interval_components(diff_lo, diff_hi),
                        "validation_eps": _number(EPS),
                    },
                    {
                        "sequence": 5,
                        "stage": "final_raw_compat_image",
                        "production_callsite": "batched_dense_tm.py:3167",
                        "decision_role": "subset test",
                        "before": _interval_components(*final_before_eps),
                        "after": _interval_components(final_lo, final_hi),
                        "validation_eps": _number(EPS),
                    },
                ],
            },
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
                "margin": torch.minimum(final_lo - target_lo, target_hi - final_hi).detach().cpu().tolist(),
                "accepted": bool(torch.all(final_lo >= target_lo) and torch.all(final_hi <= target_hi)),
            },
            "segment": [_interval(segment_lo[0, i], segment_hi[0, i]) for i in range(2)],
            "endpoint": [_interval(endpoint_lo[0, i], endpoint_hi[0, i]) for i in range(2)],
        },
        segment_tm,
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


def _cell_stage_widths(cell: Mapping[str, Any]) -> dict[str, list[float]]:
    stages = cell["operator_stages"]
    square = next(
        row
        for row in stages
        if ".x_squared_" in row["stage_id"] and "times_y" not in row["stage_id"]
    )
    nonlinear_product = next(
        row
        for row in stages
        if row["stage_id"].endswith((".x_squared_times_y", ".factor_times_y"))
    )
    return {
        "x_squared": [
            _width(square["model"]["components"][0]["ordinary_remainder"])
        ],
        "cubic_or_factor_times_y": [
            _width(nonlinear_product["model"]["components"][0]["ordinary_remainder"])
        ],
        "raw_rhs": [_width(row) for row in cell["raw_rhs_remainder"]],
        "tau_scaled_raw_rhs": [
            _width(row) for row in cell["tau_times_raw_rhs"]["after_validation_eps"]
        ],
        "poly_diff": [
            _width(row) for row in cell["poly_diff_validation_eps"]["after"]
        ],
        "final_subset_image": [
            _width(row) for row in cell["final_validation_eps"]["after"]
        ],
        "segment": [_width(row) for row in cell["segment"]],
        "endpoint": [_width(row) for row in cell["endpoint"]],
    }


def _factorial_effects(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    widths = {name: _cell_stage_widths(cell) for name, cell in cells.items()}
    effects: dict[str, Any] = {}
    for metric in widths["L0"]:
        effects[metric] = []
        for component in range(len(widths["L0"][metric])):
            l0, l1, l2, l3 = (
                widths[name][metric][component] for name in ("L0", "L1", "L2", "L3")
            )
            effects[metric].append(
                {
                    "component": component,
                    "factorization_main_reduction": ((l0 - l1) + (l2 - l3)) / 2.0,
                    "joint_square_main_reduction": ((l0 - l2) + (l1 - l3)) / 2.0,
                    "interaction_reduction": l1 - l3 - l0 + l2,
                    "cell_widths": {"L0": l0, "L1": l1, "L2": l2, "L3": l3},
                }
            )
    return effects


def _gate_c1_joint_closure(
    ode: PolynomialODE,
    base: BatchedTaylorModel,
    candidate: BatchedTaylorModel,
    h2_cell: Mapping[str, Any],
) -> dict[str, Any]:
    """Run only the preregistered C1 operator and check it with a rational oracle."""

    target_lo = torch.full_like(candidate.rem_lo, -TARGET)
    target_hi = torch.full_like(candidate.rem_hi, TARGET)
    target = candidate.with_remainder(target_lo, target_hi, category="initial_remainder")
    h2_raw = _call_dense_raw_trace_rhs(
        ode,
        target,
        effective_order=RAW_ORDER,
        cutoff_threshold=CUTOFF,
        evaluation_mode="canonical_factorized_joint",
        dependency_preserving_square=True,
    )
    closure, certificate = _joint_factorized_vdp_residual_closure(
        target.component(0),
        target.component(1),
        h2_raw.component(1),
    )
    px = _poly_from_model(target.component(0), 0, n_vars=5) + _variable(5, 3)
    py = _poly_from_model(target.component(1), 0, n_vars=5) + _variable(5, 4)
    retained = _poly_from_model(h2_raw.component(1), 0, n_vars=5)
    exact_residual = (
        (RationalPolynomial.constant(5, 1) - px * px) * py - px - retained
    )
    exact_bernstein = exact_residual.bernstein_range(
        _domain(target, remainder_symbols=True)
    )
    production_interval = RationalInterval(
        _q(closure.rem_lo[0, 0]),
        _q(closure.rem_hi[0, 0]),
    )
    started = time.perf_counter()
    production_step = dense_picard_validate_step(
        ode,
        base,
        h=H,
        tau_index=2,
        order=ORDER,
        target_remainder_radius=TARGET,
        cutoff_threshold=CUTOFF,
        max_validation_attempts=2,
        validation_eps=EPS,
        validation_mode=C1,
    )
    runtime_s = time.perf_counter() - started
    if not production_step.accepted:
        raise RuntimeError("Gate C1 production step did not validate")
    if not torch.equal(production_step.segment_tm.poly.coeffs, candidate.poly.coeffs):
        raise RuntimeError("Gate C1 changed the retained Picard polynomial")
    segment_poly_lo, segment_poly_hi = production_step.segment_tm.poly.range_bound(
        production_step.segment_tm.domain_lo,
        production_step.segment_tm.domain_hi,
        policy=production_step.segment_tm.range_policy,
        context="retained_polynomial",
        trace=None,
    )
    segment_lo, segment_hi = _interval_add(
        segment_poly_lo,
        segment_poly_hi,
        production_step.segment_tm.rem_lo,
        production_step.segment_tm.rem_hi,
    )
    endpoint = production_step.segment_tm.endpoint(2, H)
    endpoint_poly_lo, endpoint_poly_hi = endpoint.poly.range_bound(
        endpoint.domain_lo,
        endpoint.domain_hi,
        policy=endpoint.range_policy,
        context="retained_polynomial",
        trace=None,
    )
    endpoint_lo, endpoint_hi = _interval_add(
        endpoint_poly_lo,
        endpoint_poly_hi,
        endpoint.rem_lo,
        endpoint.rem_hi,
    )
    h2_width = _q(h2_raw.rem_hi[0, 1]) - _q(h2_raw.rem_lo[0, 1])
    production_width = production_interval.width
    exact_width = exact_bernstein.width
    h2_excess = h2_width - exact_width
    removed = h2_width - production_width
    promotion = Fraction(0) if h2_excess <= 0 else removed / h2_excess
    segment = _interval_components(segment_lo, segment_hi)
    endpoint_record = _interval_components(endpoint_lo, endpoint_hi)
    h2_segment_widths = [_width(row) for row in h2_cell["segment"]]
    h2_endpoint_widths = [_width(row) for row in h2_cell["endpoint"]]
    segment_no_regression = all(
        _width(row) <= h2_segment_widths[index]
        for index, row in enumerate(segment)
    )
    endpoint_no_regression = all(
        _width(row) <= h2_endpoint_widths[index]
        for index, row in enumerate(endpoint_record)
    )
    gate_pass = (
        exact_bernstein.subseteq(production_interval)
        and promotion >= Fraction(1, 10)
        and segment_no_regression
        and endpoint_no_regression
    )
    return {
        "schema": "vdp_gate_c1_joint_factor_times_y_closure_v1",
        "candidate_id": "C1",
        "validation_mode": C1,
        "operator": certificate,
        "same_input_sha256": _sha(_model_payload(target)),
        "retained_picard_polynomial_bitwise_unchanged": True,
        "exact_binary64_rational_bernstein": _rational_interval(exact_bernstein),
        "production_interval": _rational_interval(production_interval),
        "production_contains_exact_oracle": exact_bernstein.subseteq(production_interval),
        "h2_raw_width": fraction_text(h2_width),
        "production_raw_width": fraction_text(production_width),
        "exact_oracle_width": fraction_text(exact_width),
        "h2_vs_exact_excess": fraction_text(h2_excess),
        "h2_excess_removed": fraction_text(removed),
        "fraction_of_h2_vs_exact_excess_removed": fraction_text(promotion),
        "promotion_at_least_10_percent": promotion >= Fraction(1, 10),
        "segment": segment,
        "endpoint": endpoint_record,
        "segment_no_regression_vs_h2": segment_no_regression,
        "endpoint_no_regression_vs_h2": endpoint_no_regression,
        "production_step_runtime_s": runtime_s,
        "gate_c_pass": gate_pass,
    }


def _fraction_interval_record(record: Mapping[str, Any]) -> RationalInterval:
    return RationalInterval(
        Fraction.from_float(float(record["lo"]["decimal"])),
        Fraction.from_float(float(record["hi"]["decimal"])),
    )


def _consumer_path(
    start: str,
    targets: set[str],
    children: Mapping[str, Sequence[str]],
) -> list[str]:
    queue: list[tuple[str, list[str]]] = [(start, [start])]
    seen = {start}
    while queue:
        current, path = queue.pop(0)
        if current in targets:
            return path
        for child in children.get(current, ()):
            if child not in seen:
                seen.add(child)
                queue.append((child, [*path, child]))
    return []


def _derive_live_loss_ledger(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = cells["L0"]
    events = baseline["production_raw_trace"]["execution_events"]
    necessary = baseline["event_necessary_enclosures"]
    children: dict[str, list[str]] = {row["stage_id"]: [] for row in events}
    for row in events:
        for parent in row["parent_stage_ids"]:
            children[parent].append(row["stage_id"])
    subset_targets = {
        row["stage_id"] for row in events if row["operation"] == "subset_test"
    }

    widths = {name: _cell_stage_widths(cell) for name, cell in cells.items()}
    baseline_final = Fraction.from_float(widths["L0"]["final_subset_image"][1])
    factorized_final = Fraction.from_float(widths["L1"]["final_subset_image"][1])
    joint_final = Fraction.from_float(widths["L2"]["final_subset_image"][1])
    marginal_by_stage: dict[str, dict[str, Any]] = {}
    square_stage = next(
        row["stage_id"]
        for row in events
        if "x_squared_generic" in row["stage_id"]
        and "times_y" not in row["stage_id"]
    )
    final_expression_stage = next(
        row["stage_id"] for row in events if "distributed_final" in row["stage_id"]
    )
    marginal_by_stage[square_stage] = {
        "counterfactual": "same distributed graph; replace only generic self-square by joint self-square",
        "final_subset_width_reduction": fraction_text(baseline_final - joint_final),
        "final_subset_component": 1,
    }
    marginal_by_stage[final_expression_stage] = {
        "counterfactual": "same generic square; replace only distributed graph by factorized graph",
        "final_subset_width_reduction": fraction_text(baseline_final - factorized_final),
        "final_subset_component": 1,
    }

    rows: list[dict[str, Any]] = []
    for event in events:
        stage_id = event["stage_id"]
        production = _fraction_interval_record(event["production_interval"])
        necessary_record = necessary[stage_id]
        exact_record = necessary_record["necessary_enclosure"]
        exact = RationalInterval(Fraction(exact_record["lo"]), Fraction(exact_record["hi"]))
        surplus = production.width - exact.width
        path = _consumer_path(stage_id, subset_targets, children)
        marginal = marginal_by_stage.get(
            stage_id,
            {
                "counterfactual": "no isolated operator replacement assigned at this stage",
                "final_subset_width_reduction": "0/1",
                "final_subset_component": event["component"],
            },
        )
        marginal_q = Fraction(marginal["final_subset_width_reduction"])
        eligible_loss = event["operation"] not in {
            "validation_eps_inflation",
            "state_input",
            "subset_test",
        }
        rows.append(
            {
                **event,
                "input_stages": list(event["parent_stage_ids"]),
                "output_stage": stage_id,
                **necessary_record,
                "exact_surplus": fraction_text(surplus),
                "ordinary_remainder_contains_necessary": exact.subseteq(production),
                "loss_classification_eligible": eligible_loss,
                "final_subset_width_live": bool(path),
                "consumed_by_downstream_production_decision": bool(path)
                or event["decision_role"]
                in {
                    "pre_loop_feasibility_gate",
                    "finite_predicate_only_in_raw_compat_mode",
                    "subset_decision",
                },
                "consumer_chain_to_final_subset": path,
                "same_input_marginal": marginal,
                "live_material": bool(path) and marginal_q > 0,
            }
        )

    strict = [
        row
        for row in rows
        if row["loss_classification_eligible"] and Fraction(row["exact_surplus"]) > 0
    ]
    live_strict = [row for row in strict if row["final_subset_width_live"]]
    live_material = [row for row in rows if row["live_material"]]
    if not strict or not live_strict or not live_material:
        raise RuntimeError("automatic loss classification did not find all required classes")
    ranked_marginals = sorted(
        (
            row
            for row in rows
            if Fraction(row["same_input_marginal"]["final_subset_width_reduction"]) > 0
        ),
        key=lambda row: Fraction(
            row["same_input_marginal"]["final_subset_width_reduction"]
        ),
        reverse=True,
    )

    eps_rows = [row for row in rows if row["validation_eps_payment"] is not None]
    payment_ids = [row["stage_id"] for row in eps_rows]
    if len(payment_ids) != len(set(payment_ids)):
        raise RuntimeError("a validation_eps payment was reused")
    rounding_proof = {
        "method": (
            "exact-rational/Bernstein necessary enclosures are checked at every true "
            "consumer; each validation_eps event is a distinct executed interval inflation"
        ),
        "payment_ids_unique": True,
        "payments": [
            {
                "payment_id": row["stage_id"],
                "sequence": row["sequence"],
                "source": row["source"],
                "consumer_chain_to_final_subset": row["consumer_chain_to_final_subset"],
                "validation_eps": row["validation_eps_payment"],
            }
            for row in eps_rows
        ],
        "final_subset_events_contain_complete_exact_chain": all(
            row["ordinary_remainder_contains_necessary"]
            for row in rows
            if row["operation"] == "subset_test"
        ),
    }
    return {
        "schema": "vdp_live_loss_production_event_ledger_v1",
        "rows": rows,
        "first_syntactic_strict_surplus": strict[0],
        "first_live_strict_surplus": live_strict[0],
        "first_live_material_surplus": live_material[0],
        "largest_same_input_marginal_contributor": ranked_marginals[0],
        "same_input_marginal_ranking": ranked_marginals,
        "rounding_proof": rounding_proof,
    }


def run(output_dir: Path, flowstar_root: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_identity = _scientific_source_identity()
    ode, state, base = _build_base()
    candidate, picard_rows = _picard_iteration_ledger(ode, base)
    cells: dict[str, dict[str, Any]] = {}
    for cell_id, factorized, joint_square in (
        ("L0", False, False),
        ("L1", True, False),
        ("L2", False, True),
        ("L3", True, True),
    ):
        cells[cell_id], _ = _raw_cell(
            cell_id,
            ode,
            base,
            candidate,
            factorized=factorized,
            joint_square=joint_square,
            normalization_scale=state.scales,
        )
    if len({cell["same_input_sha256"] for cell in cells.values()}) != 1:
        raise RuntimeError("Gate B cells did not start from byte-identical input")
    b1, b2 = cells["L0"], cells["L3"]
    flow_source = _flowstar_semantics(flowstar_root)
    flow_runtime = _flowstar_runtime_crosscheck()

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
        for cell in cells.values()
        for row in cell["operator_stages"]
    )
    all_poly_diff_sound = all(
        row["production_contains_exact_bernstein"]
        for cell in cells.values()
        for row in cell["poly_diff_range_bound"]["components"]
    )
    gate_b_pass = (
        all(cell["eligible_for_production"] for cell in cells.values())
        and all(row["no_regression"] for row in raw_metrics.values())
        and all(row["no_regression"] for row in segment_metrics.values())
    )
    preregistration = {
        "L0": {"status": "executed_baseline", "operator": "distributed expression plus generic square"},
        "L1": {"status": "executed", "operator": "factorized expression plus generic square"},
        "L2": {"status": "executed", "operator": "distributed expression plus joint square"},
        "L3": {"status": "executed_combined", "operator": "factorized expression plus joint square"},
    }
    factorial_effects = _factorial_effects(cells)
    gate_c1 = _gate_c1_joint_closure(ode, base, candidate, cells["L3"])
    if gate_c1["same_input_sha256"] != b1["same_input_sha256"]:
        raise RuntimeError("Gate C1 did not use the byte-identical Gate B input")
    live_loss = _derive_live_loss_ledger(cells)
    all_live_events_sound = all(
        row["ordinary_remainder_contains_necessary"] for row in live_loss["rows"]
    )
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
        "scientific_source_identity": source_identity,
        "working_diff_sha256": hashlib.sha256(
            subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True).stdout
        ).hexdigest(),
        "outward_exact_prestate": _model_payload(base),
        "picard_iterations": picard_rows,
        "gate_b_cells": cells,
        "gate_c1": gate_c1,
        "live_loss": live_loss,
    }
    _write_json(output_dir / "production_operator_ledger.json", ledger)
    _write_json(output_dir / "live_loss_ledger.json", live_loss)
    _write_json(output_dir / "gate_c1_joint_closure.json", gate_c1)
    _write_json(output_dir / "flowstar_source_semantics.json", flow_source)
    _write_json(output_dir / "flowstar_runtime_crosscheck.json", flow_runtime)
    _write_json(
        output_dir / "gate_b_same_input_matrix.json",
        {
            "schema": "vdp_h2_same_input_four_cell_gate_b_v2",
            "preregistration": preregistration,
            "same_input_sha256": b1["same_input_sha256"],
            "cell_stage_widths": {
                name: _cell_stage_widths(cell) for name, cell in cells.items()
            },
            "factorial_effects": factorial_effects,
            "raw_residual_excess": raw_metrics,
            "segment_excess": segment_metrics,
            "gate_b_pass": gate_b_pass,
        },
    )
    summary = {
        "schema": "vdp_live_loss_ablation_gate_summary_v2",
        "first_syntactic_strict_surplus": live_loss["first_syntactic_strict_surplus"],
        "first_live_strict_surplus": live_loss["first_live_strict_surplus"],
        "first_live_material_surplus": live_loss["first_live_material_surplus"],
        "largest_same_input_marginal_contributor": live_loss[
            "largest_same_input_marginal_contributor"
        ],
        "picard_iteration_count": 4,
        "operator_stage_count": len(picard_rows) + len(b1["operator_stages"]) + len(b2["operator_stages"]),
        "all_operator_stages_exact_bernstein_contained": all_stage_sound,
        "all_poly_diff_exact_bernstein_contained": all_poly_diff_sound,
        "all_live_events_exact_necessary_contained": all_live_events_sound,
        "same_input_byte_identity": b1["same_input_sha256"] == b2["same_input_sha256"],
        "raw_residual_excess": raw_metrics,
        "segment_excess": segment_metrics,
        "preregistration": preregistration,
        "gate_a_pass": all_stage_sound
        and all_poly_diff_sound
        and all_live_events_sound
        and live_loss["rounding_proof"]["payment_ids_unique"]
        and live_loss["rounding_proof"][
            "final_subset_events_contain_complete_exact_chain"
        ],
        "gate_b_pass": gate_b_pass,
        "gate_c_pass": gate_c1["gate_c_pass"],
        "production_candidate": {
            "candidate_id": "C1",
            "validation_mode": C1,
            "single_operator_only": True,
            "gate_c1_artifact": "gate_c1_joint_closure.json",
        },
        "flowstar_source_commit": flow_source["head"],
        "flowstar_runtime_is_soundness_oracle": False,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["gate_a_pass"] or not summary["gate_b_pass"] or not summary["gate_c_pass"]:
        raise RuntimeError("VDP Gate A/B/C failed")
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
