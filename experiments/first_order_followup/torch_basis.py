"""Experiment-local finite-basis Taylor-model construction and reset policies."""
from __future__ import annotations

import math
import sys
from itertools import combinations_with_replacement
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from torch_tm_flowpipe import Interval, TMVector
from torch_tm_flowpipe.flowpipe import FlowpipeSegment
from torch_tm_flowpipe.polynomial import Polynomial
from torch_tm_flowpipe.taylor_model import TaylorModel
import torch_tm_flowpipe.flowpipe as flowpipe_core

torch.set_default_dtype(torch.float64)

BASIS_NAMES = ("B1", "B_DR", "B2", "B3")


@dataclass(frozen=True)
class BasisProjectionRecord:
    stage: str
    iteration: int
    state_index: int
    exponent: tuple[int, ...]
    coefficient: float
    range_lower: float
    range_upper: float
    range_width: float


def exponent_is_retained(
    exponent: Sequence[int],
    basis: str,
    *,
    tau_index: int | None,
) -> bool:
    exp = tuple(int(value) for value in exponent)
    degree = sum(exp)
    if basis == "B1":
        return degree <= 1
    if basis == "B2":
        return degree <= 2
    if basis == "B3":
        return degree <= 3
    if basis != "B_DR":
        raise ValueError(f"unknown finite basis {basis!r}")
    if degree <= 1:
        return True
    if degree != 2 or tau_index is None:
        return False
    if exp[tau_index] == 2:
        return all(value == 0 for index, value in enumerate(exp) if index != tau_index)
    if exp[tau_index] != 1:
        return False
    return sum(value for index, value in enumerate(exp) if index != tau_index) == 1


def retained_dictionary(
    basis: str,
    n_vars: int,
    *,
    tau_index: int | None,
) -> tuple[tuple[int, ...], ...]:
    candidates: list[tuple[int, ...]] = [(0,) * n_vars]
    for index in range(n_vars):
        exp = [0] * n_vars
        exp[index] = 1
        candidates.append(tuple(exp))
    maximum_degree = 3 if basis == "B3" else 2
    for degree in range(2, maximum_degree + 1):
        for indices in combinations_with_replacement(range(n_vars), degree):
            exp = [0] * n_vars
            for index in indices:
                exp[index] += 1
            exp_t = tuple(exp)
            if exponent_is_retained(exp_t, basis, tau_index=tau_index):
                candidates.append(exp_t)
    return tuple(sorted(set(candidates), key=lambda exp: (sum(exp), exp)))


def _term_interval(
    exponent: tuple[int, ...],
    coefficient: Any,
    domain: Sequence[Interval],
) -> Interval:
    term = Interval.point(coefficient)
    for power, interval in zip(exponent, domain):
        if power:
            term = term * interval.pow_int(power)
    return term


def project_to_basis(
    tm: TMVector,
    basis: str,
    *,
    tau_index: int | None,
    stage: str,
    iteration: int,
    arithmetic_order: int | None = None,
) -> tuple[TMVector, list[BasisProjectionRecord]]:
    """Project polynomial support and independently range every removed term."""
    projected: list[TaylorModel] = []
    records: list[BasisProjectionRecord] = []
    for state_index, model in enumerate(tm):
        kept: dict[tuple[int, ...], Any] = {}
        remainder = model.remainder
        for exponent, coefficient in model.polynomial.terms.items():
            exponent = tuple(exponent)
            if exponent_is_retained(exponent, basis, tau_index=tau_index):
                kept[exponent] = coefficient
                continue
            contribution = _term_interval(exponent, coefficient, model.domain)
            remainder = remainder + contribution
            records.append(
                BasisProjectionRecord(
                    stage=stage,
                    iteration=int(iteration),
                    state_index=state_index,
                    exponent=exponent,
                    coefficient=float(coefficient.detach().cpu()),
                    range_lower=float(contribution.lo.detach().cpu()),
                    range_upper=float(contribution.hi.detach().cpu()),
                    range_width=float(contribution.width().detach().cpu()),
                )
            )
        projected.append(
            TaylorModel(
                Polynomial(kept, model.n_vars),
                remainder,
                list(model.domain),
                order=(
                    int(arithmetic_order)
                    if arithmetic_order is not None
                    else max(3 if basis == "B3" else 2, model.order)
                ),
                truncation_range_split=model.truncation_range_split,
            )
        )
    return TMVector(projected), records


def _finite_basis_picard(
    ode_fn: Callable[..., Any],
    base_poly_ext: TMVector,
    tau_index: int,
    basis: str,
    *,
    iterations: int,
    arithmetic_order: int,
) -> tuple[TMVector, list[BasisProjectionRecord]]:
    domain = base_poly_ext.domain
    current = base_poly_ext
    records: list[BasisProjectionRecord] = []
    for iteration in range(1, iterations + 1):
        rhs = flowpipe_core._call_ode(ode_fn, current, None)
        models = []
        for base_model, rhs_model in zip(base_poly_ext, rhs):
            models.append(base_model + rhs_model.integrate(tau_index))
        current, iteration_records = project_to_basis(
            TMVector(models),
            basis,
            tau_index=tau_index,
            stage="picard",
            iteration=iteration,
            arithmetic_order=arithmetic_order,
        )
        records.extend(iteration_records)
        # Keep a fixed arithmetic ceiling selected by the caller.  The finite
        # dictionary projection, not a changing arithmetic order, distinguishes
        # the compared bases.
        current = TMVector(
            TaylorModel(
                model.polynomial,
                model.remainder,
                domain,
                order=arithmetic_order,
                truncation_range_split=model.truncation_range_split,
            )
            for model in current
        )
    return current, records


def finite_basis_step_from_tm(
    ode_fn: Callable[..., Any],
    x0_tm: TMVector,
    h: float,
    basis: str,
    *,
    picard_iterations: int = 2,
    max_validation_attempts: int = 20,
    validation_eps: float = 1e-12,
    growth_factor: float = 1.25,
    diagnostics: list[dict[str, Any]] | None = None,
    arithmetic_order: int = 2,
) -> tuple[FlowpipeSegment, list[BasisProjectionRecord]]:
    """Construct and validate one segment using a fixed exponent dictionary."""
    if basis not in BASIS_NAMES:
        raise ValueError(f"basis must be one of {BASIS_NAMES}")
    if h <= 0:
        raise ValueError("h must be positive")
    minimum_order = 3 if basis == "B3" else 2
    if arithmetic_order < minimum_order:
        raise ValueError(
            f"{basis} requires arithmetic_order >= {minimum_order}"
        )
    tau_interval = Interval(0.0, float(h))
    base_ext = x0_tm.extend_domain(tau_interval)
    tau_index = x0_tm.n_vars
    domain = base_ext.domain
    base_poly_ext = TMVector(
        TaylorModel(
            model.polynomial,
            Interval.zero(dtype=model.remainder.dtype, device=model.remainder.device),
            domain,
            order=arithmetic_order,
        )
        for model in base_ext
    )
    candidate, records = _finite_basis_picard(
        ode_fn,
        base_poly_ext,
        tau_index,
        basis,
        iterations=picard_iterations,
        arithmetic_order=arithmetic_order,
    )
    candidate_remainders = [model.remainder.to_tuple() for model in candidate]
    validated, status, attempts, message = flowpipe_core._validate_picard(
        ode_fn,
        base_ext,
        candidate,
        tau_index,
        arithmetic_order,
        None,
        max_attempts=max_validation_attempts,
        validation_eps=validation_eps,
        growth_factor=growth_factor,
        h=float(h),
        diagnostics=diagnostics,
        diagnostics_context={
            "basis": basis,
            "picard_iterations": picard_iterations,
            "candidate_remainders_before_validation": candidate_remainders,
        },
    )
    validated, final_projection_records = project_to_basis(
        validated,
        basis,
        tau_index=tau_index,
        stage="validated_segment",
        iteration=picard_iterations,
        arithmetic_order=arithmetic_order,
    )
    records.extend(final_projection_records)
    final_tm = validated.substitute_const(tau_index, float(h)).drop_variable(tau_index)
    segment = FlowpipeSegment(
        tm=validated,
        final_tm=final_tm,
        status=status,
        h=float(h),
        order=arithmetic_order,
        validation_attempts=attempts,
        message=message,
        tau_index=tau_index,
        selective_term_stats={
            "basis": basis,
            "picard_iterations": picard_iterations,
            "candidate_remainders_before_validation": candidate_remainders,
            "validated_remainders": [
                model.remainder.to_tuple() for model in validated
            ],
        },
        selective_term_details=[
            {
                "stage": record.stage,
                "iteration": record.iteration,
                "state_index": record.state_index,
                "exponent": list(record.exponent),
                "coefficient": record.coefficient,
                "range_lower": record.range_lower,
                "range_upper": record.range_upper,
                "range_width": record.range_width,
            }
            for record in records
        ],
    )
    return segment, records


def _affine_center_generator_remainder(
    tm: TMVector,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert affine TMs on arbitrary boxes to normalized generators."""
    dimension = len(tm)
    variables = tm.n_vars
    if not dimension:
        raise ValueError("affine reset requires at least one state")
    template = tm[0].remainder.lo
    center = torch.zeros(
        dimension, dtype=template.dtype, device=template.device
    )
    generator = torch.zeros(
        (dimension, variables),
        dtype=template.dtype,
        device=template.device,
    )
    remainder_radius = torch.zeros(
        dimension, dtype=template.dtype, device=template.device
    )
    for state, model in enumerate(tm):
        for exponent, coefficient in model.polynomial.terms.items():
            degree = sum(exponent)
            if degree == 0:
                center[state] += coefficient
            elif degree == 1:
                variable = next(index for index, power in enumerate(exponent) if power)
                interval = model.domain[variable]
                center[state] += coefficient * interval.mid()
                generator[state, variable] += coefficient * interval.radius()
            else:
                raise ValueError("affine reset received a nonlinear polynomial")
        center[state] += model.remainder.mid()
        remainder_radius[state] = model.remainder.radius()
    return center, generator, remainder_radius


def affine_reset(
    tm: TMVector,
    *,
    method: str = "box",
) -> tuple[TMVector, dict[str, Any]]:
    """Return a fresh normalized affine box or QR parallelotope."""
    center, generator, remainder_radius = _affine_center_generator_remainder(tm)
    dimension = len(tm)
    if generator.shape[1] != dimension:
        # A square carried basis is required by the fixed-generator comparison.
        # Fall back to an axis-aligned affine enclosure without dropping width.
        method = "box"
    if method == "box":
        radii = torch.sum(torch.abs(generator), dim=1) + remainder_radius
        matrix = torch.diag(radii)
        condition = float("inf") if bool(torch.any(radii == 0)) else float(
            torch.linalg.cond(matrix).detach().cpu()
        )
    elif method == "qr":
        q, _ = torch.linalg.qr(generator)
        coordinates = q.T @ generator
        radii = torch.sum(torch.abs(coordinates), dim=1) + torch.abs(q.T) @ remainder_radius
        matrix = q @ torch.diag(radii)
        condition = float("inf") if bool(torch.any(radii == 0)) else float(
            torch.linalg.cond(matrix).detach().cpu()
        )
    else:
        raise ValueError("reset method must be 'box' or 'qr'")
    domain = [
        Interval(
            torch.as_tensor(-1.0, dtype=center.dtype, device=center.device),
            torch.as_tensor(1.0, dtype=center.dtype, device=center.device),
        )
        for _ in range(dimension)
    ]
    models = []
    for state in range(dimension):
        terms: dict[tuple[int, ...], Any] = {
            (0,) * dimension: center[state]
        }
        for variable in range(dimension):
            coefficient = matrix[state, variable]
            if float(coefficient) != 0.0:
                exponent = [0] * dimension
                exponent[variable] = 1
                terms[tuple(exponent)] = coefficient
        models.append(
            TaylorModel(
                Polynomial(terms, dimension),
                Interval.zero(dtype=center.dtype, device=center.device),
                domain,
                order=2,
            )
        )
    reset = TMVector(models)
    stats = {
        "method": method,
        "center": center.tolist(),
        "generator_matrix": matrix.tolist(),
        "generator_condition_number": condition,
        "input_remainder_radius": remainder_radius.tolist(),
        "output_box": [interval.to_tuple() for interval in reset.range_box()],
    }
    return reset, stats


def normalized_initial_tm(
    initial_box: Sequence[Sequence[float]],
    *,
    order: int = 2,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> TMVector:
    dimension = len(initial_box)
    device_value = torch.device(device)
    domain = [
        Interval(
            torch.as_tensor(-1.0, dtype=dtype, device=device_value),
            torch.as_tensor(1.0, dtype=dtype, device=device_value),
        )
        for _ in range(dimension)
    ]
    models = []
    for state, (lower, upper) in enumerate(initial_box):
        center = torch.as_tensor(
            0.5 * (float(lower) + float(upper)),
            dtype=dtype,
            device=device_value,
        )
        radius = torch.as_tensor(
            0.5 * (float(upper) - float(lower)),
            dtype=dtype,
            device=device_value,
        )
        constant = (0,) * dimension
        exponent = [0] * dimension
        exponent[state] = 1
        terms = {constant: center}
        if radius:
            terms[tuple(exponent)] = radius
        models.append(
            TaylorModel(
                Polynomial(terms, dimension),
                Interval.zero(dtype=dtype, device=device_value),
                domain,
                order=order,
            )
        )
    return TMVector(models)


def polynomial_group(exponent: Sequence[int], *, tau_index: int | None) -> str:
    exp = tuple(int(value) for value in exponent)
    degree = sum(exp)
    tau_power = exp[tau_index] if tau_index is not None else 0
    state_degree = degree - tau_power
    if degree == 0:
        return "constant"
    if tau_power == 1 and state_degree == 0:
        return "local_time"
    if tau_power == 2 and state_degree == 0:
        return "time_squared"
    if tau_power == 1 and state_degree == 1:
        return "time_state"
    if tau_power == 0 and state_degree == 1:
        return "affine_state"
    if tau_power == 0 and state_degree >= 2:
        return "nonlinear_state"
    return "other"


def diagnose_tm(
    tm: TMVector,
    *,
    tau_index: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state, model in enumerate(tm):
        group_l1: dict[str, float] = {}
        group_width: dict[str, float] = {}
        affine_coefficients: dict[str, float] = {}
        for exponent, coefficient in model.polynomial.terms.items():
            group = polynomial_group(exponent, tau_index=tau_index)
            group_l1[group] = group_l1.get(group, 0.0) + abs(
                float(coefficient.detach().cpu())
            )
            contribution = _term_interval(tuple(exponent), coefficient, model.domain)
            group_width[group] = group_width.get(group, 0.0) + float(
                contribution.width().detach().cpu()
            )
            if group == "affine_state":
                affine_coefficients[",".join(map(str, exponent))] = float(
                    coefficient.detach().cpu()
                )
        polynomial_range = model.polynomial.evaluate_interval(model.domain)
        total = model.range_box()
        rows.append(
            {
                "state_index": state,
                "center": float(
                    model.polynomial.terms.get(
                        (0,) * model.n_vars, torch.zeros((), dtype=torch.float64)
                    ).detach().cpu()
                ),
                "affine_generator_coefficients": affine_coefficients,
                "active_variables": sorted(model.active_variables()),
                "variable_count": model.n_vars,
                "polynomial_range": polynomial_range.to_tuple(),
                "polynomial_width": float(polynomial_range.width().detach().cpu()),
                "remainder": model.remainder.to_tuple(),
                "remainder_width": float(model.remainder.width().detach().cpu()),
                "total_range": total.to_tuple(),
                "total_width": float(total.width().detach().cpu()),
                "coefficient_l1_by_group": group_l1,
                "range_width_by_group": group_width,
                "width_sum_check": float(
                    polynomial_range.width().detach().cpu()
                    + model.remainder.width().detach().cpu()
                ),
            }
        )
    return rows


def harmonic_exact_affine(
    initial_box: Sequence[Sequence[float]],
    time_value: float,
) -> TMVector:
    """Exact rotation oracle represented on fresh normalized generators."""
    if len(initial_box) != 2:
        raise ValueError("harmonic oracle requires two states")
    initial = normalized_initial_tm(initial_box, order=2)
    c = math.cos(time_value)
    s = math.sin(time_value)
    return TMVector(
        [
            initial[0] * c + initial[1] * s,
            initial[0] * (-s) + initial[1] * c,
        ]
    )
