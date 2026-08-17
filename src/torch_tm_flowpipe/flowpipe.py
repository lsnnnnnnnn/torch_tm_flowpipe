"""Fixed-step Taylor-model flowpipe construction for polynomial ODE prototypes."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Callable, Iterable, List, Mapping, Sequence

import torch

from .interval import Interval, ensure_interval
from .polynomial import Polynomial, evaluate_interval_normal
from .safety import intervals_are_finite
from .fixed_support_outward import OutwardIntervalTensor, outward_matmul, outward_sum
from .structured_remainder import (
    ELIGIBLE_STRUCTURED_SOURCES,
    STRUCTURED_REMAINDER_CANDIDATE,
    STRUCTURED_TOTAL_DELTA_CANDIDATE,
    StructuredRemainderState,
    complete_polynomial_structured_image,
    compare_complete_polynomial_contracts,
    initialize_structured_remainder_state,
    materialize_structured_remainder,
    normal_interval_to_physical,
    physical_interval_to_normal,
    split_structured_source_center,
    structured_column_contributions,
    structured_remainder_boundary_update,
)
from .s1_boundary_attribution import (
    S1BoundaryAttributionRecord,
    S1BoundaryStage,
    tensor_hex,
    tensor_sha256,
)
from .source_ledger import (
    BOUNDED_SOURCE_LEDGER_CANDIDATE,
    BoundedSourceLedgerState,
    accepted_successor as source_ledger_accepted_successor,
    affine_lift_interval as source_ledger_affine_lift_interval,
    collapse_source_polynomial,
    commit_or_preserve as source_ledger_commit_or_preserve,
    source_payload_hash,
)
from .g2_shared_column import (
    G2_SHARED_COLUMN_CANDIDATE,
    G2SharedColumnState,
    accepted_successor as g2_accepted_successor,
    commit_or_preserve as g2_commit_or_preserve,
    owner_rows as g2_owner_rows,
    partition_source_terms as g2_partition_source_terms,
    polynomial_payload_sha256 as g2_polynomial_payload_sha256,
    polynomial_table as g2_polynomial_table,
    rotate_current_to_retained as g2_rotate_current_to_retained,
)
from .symbolic_remainder import (
    FlowstarSymbolicRemainderQueue,
    SymbolicRemainderState,
    flowstar_normalized_insertion_linear_queue_v2_reset,
    flowstar_normalized_insertion_symbolic_queue_reset,
    flowstar_symbolic_remainder_queue_reset,
    introduce_symbolic_remainders,
)
from .taylor_model import TaylorModel
from .tm_vector import TMVector

ODEFunction = Callable[..., Sequence[TaylorModel] | TMVector]

FLOWSTAR_COMPAT_STEP_SHRINK = 0.5
FLOWSTAR_COMPAT_STEP_GROW = 1.1
NORMALIZED_INSERTION_DEPENDENCY_PRESERVING = (
    "normalized_insertion_dependency_preserving"
)


@dataclass
class FlowpipeSegment:
    """One validated flowpipe segment.

    ``tm`` is the segment over the original dependency variables plus a local
    time variable ``tau``.  ``final_tm`` preserves the historical package
    behavior and is the endpoint used for dependency-preserving propagation.
    ``endpoint_raw_tm`` is obtained only by substituting ``tau=h`` in ``tm``;
    ``endpoint_tightened_tm`` is the optional fixed-time residual
    recomputation.  Keeping all three names explicit prevents experiment code
    from silently comparing different endpoint semantics.
    """

    tm: TMVector
    final_tm: TMVector
    status: str
    h: float
    order: int
    validation_attempts: int
    message: str = ""
    tau_index: int | None = None
    symbolic_remainder: bool = False
    symbolic_remainder_state: SymbolicRemainderState | None = None
    symbolic_remainder_stats: Mapping[str, Any] | None = None
    reset_tm: TMVector | None = None
    next_h: float | None = None
    step_rejections: int = 0
    selective_term_stats: Mapping[str, Any] | None = None
    selective_term_details: Sequence[Mapping[str, Any]] | None = None
    flowstar_symbolic_queue_state: FlowstarSymbolicRemainderQueue | None = None
    flowstar_symbolic_queue_stats: Mapping[str, Any] | None = None
    flowstar_normal_state: "FlowstarNormalFlowpipeState | None" = None
    flowstar_normal_stats: Mapping[str, Any] | None = None
    endpoint_raw_tm: TMVector | None = None
    endpoint_tightened_tm: TMVector | None = None
    endpoint_semantics: str = "legacy_final_tm"
    endpoint_tightening_applied: bool = False
    endpoint_tightening_validation_method: str = ""
    backend_lane: str = "sparse_reference"
    backend_counters: Mapping[str, int] | None = None
    backend_trace: Sequence[Mapping[str, Any]] | None = None
    candidate_remainder: Sequence[Sequence[float]] | None = None
    picard_image_remainder: Sequence[Sequence[float]] | None = None
    subset_margin: Sequence[Sequence[float]] | None = None
    validated_remainder_ledger: Any | None = None
    validated_remainder_decomposition: Any | None = None
    structured_boundary_result: Any | None = None
    structured_state_before: Any | None = None
    structured_state_after: Any | None = None
    endpoint_total_structured_remainder: Any | None = None
    tube_total_structured_remainder: Any | None = None
    endpoint_ordinary_remainder: Any | None = None
    tube_ordinary_remainder: Any | None = None
    endpoint_total_remainder: Any | None = None
    tube_total_remainder: Any | None = None
    endpoint_publication_mask: Any | None = None
    tube_publication_mask: Any | None = None
    boundary_attribution_record: S1BoundaryAttributionRecord | None = None
    source_ledger_boundary_result: Any | None = None
    source_ledger_state_before: Any | None = None
    source_ledger_state_after: Any | None = None

    def __post_init__(self) -> None:
        # Older experiment helpers construct FlowpipeSegment directly.  Treat
        # their final_tm as both endpoint views unless the step builder supplied
        # the explicit objects.
        if self.status == "validated" and self.endpoint_raw_tm is None:
            self.endpoint_raw_tm = self.final_tm
        if self.status == "validated" and self.endpoint_tightened_tm is None:
            self.endpoint_tightened_tm = self.final_tm


@dataclass
class FlowpipeResult:
    segments: List[FlowpipeSegment]
    status: str
    final_tm: TMVector
    mode: str

    @property
    def validation_attempts(self) -> int:
        return sum(seg.validation_attempts for seg in self.segments)


@dataclass(frozen=True)
class FlowstarNormalFlowpipeState:
    """Opt-in Flow*-style normal-composition state.

    ``tmv_pre`` is the validated left/preconditioning flow map for the current
    step. ``tmv_right`` maps the current local normalized variables back into the
    previous normal coordinate frame.
    """

    tmv_pre: TMVector
    tmv_right: TMVector
    domain: list[Interval]
    center: list[float]
    scales: list[float]
    step_index: int = 0
    diagnostics: Mapping[str, Any] | None = None
    symbolic_queue: FlowstarSymbolicRemainderQueue | None = None
    symbolic_queue_max_size: int = 100
    initial_remainders: tuple[Interval, ...] | None = None
    complete_initial_tm: TMVector | None = None
    structured_remainder_state: Any | None = None
    bounded_source_ledger_state: BoundedSourceLedgerState | None = None
    g2_shared_column_state: G2SharedColumnState | None = None
    g2_retained_source_tm: TMVector | None = None

    @staticmethod
    def from_initial_box(
        x0_box: Sequence[Interval | tuple[float, float] | list[float] | float],
        order: int,
    ) -> "FlowstarNormalFlowpipeState":
        normalized = _normalized_tm_from_box(x0_box, order)
        domain = normalized.domain
        boxes = _as_interval_list(x0_box)
        return FlowstarNormalFlowpipeState(
            tmv_pre=normalized,
            tmv_right=TMVector.identity(domain, order=order),
            domain=list(domain),
            center=[float(iv.mid().detach().cpu()) for iv in boxes],
            scales=[float(iv.radius().detach().cpu()) for iv in boxes],
            step_index=0,
            diagnostics={"reset_mode": "normalized_insertion", "initial_state": True},
        )

    @staticmethod
    def from_exact_decimal_box(
        x0_box: Sequence[tuple[str | int | Fraction, str | int | Fraction]],
        order: int,
    ) -> "FlowstarNormalFlowpipeState":
        """Build a binary64 affine TM that contains exact decimal endpoints.

        This is the opt-in ``exact_decimal_contract`` initialization lane.  It
        deliberately rejects Python floats: a float has already lost the
        decimal/rational identity that this constructor is meant to preserve.
        The midpoint is rounded to binary64 and the radius is rounded upward
        far enough to contain both exact endpoints.  All later arithmetic is
        unchanged.
        """

        if not x0_box:
            raise ValueError("exact decimal initial box must not be empty")
        centers: list[float] = []
        scales: list[float] = []
        witnesses: list[dict[str, Any]] = []
        for component, bounds in enumerate(x0_box):
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise TypeError("exact decimal bounds must be two-item tuples/lists")
            lower = _exact_fraction(bounds[0])
            upper = _exact_fraction(bounds[1])
            if lower > upper:
                raise ValueError(f"exact decimal interval {component} has lower > upper")
            exact_midpoint = (lower + upper) / 2
            center = float(exact_midpoint)
            if not math.isfinite(center):
                raise OverflowError("exact decimal midpoint is not finite in binary64")
            center_q = Fraction.from_float(center)
            required_radius = max(center_q - lower, upper - center_q)
            scale = _fraction_to_binary64_up(required_radius)
            scale_q = Fraction.from_float(scale)
            actual_lower = center_q - scale_q
            actual_upper = center_q + scale_q
            if actual_lower > lower or actual_upper < upper:
                raise AssertionError("outward affine initialization failed exact containment")
            centers.append(center)
            scales.append(scale)
            witnesses.append(
                {
                    "component": component,
                    "expected_exact_lower": _fraction_text(lower),
                    "expected_exact_upper": _fraction_text(upper),
                    "binary64_center_hex": center.hex(),
                    "binary64_scale_hex": scale.hex(),
                    "represented_exact_lower": _fraction_text(actual_lower),
                    "represented_exact_upper": _fraction_text(actual_upper),
                    "contains_expected_exact": True,
                }
            )
        normalized = _normalized_tm_from_center_scale(centers, scales, order)
        domain = normalized.domain
        return FlowstarNormalFlowpipeState(
            tmv_pre=normalized,
            tmv_right=TMVector.identity(domain, order=order),
            domain=list(domain),
            center=centers,
            scales=scales,
            step_index=0,
            diagnostics={
                "reset_mode": "normalized_insertion",
                "initial_state": True,
                "initialization_contract": "exact_decimal_contract",
                "exact_decimal_containment_witness": witnesses,
            },
        )

    def endpoint_tm(self) -> TMVector:
        models: list[TaylorModel] = []
        source_state = self.bounded_source_ledger_state
        g2_state = self.g2_shared_column_state
        for index, (model, center, scale) in enumerate(zip(self.tmv_right, self.center, self.scales)):
            physical = (model * float(scale)) + float(center)
            if source_state is not None and source_state.active[index]:
                radius = float.fromhex(source_state.radii_hex[index])
                source = TaylorModel.variable(
                    source_state.base_dim + index,
                    self.domain,
                    order=model.order,
                )
                physical = physical + source * radius
            if g2_state is not None:
                if self.g2_retained_source_tm is None:
                    raise ValueError("G2 normal state is missing retained polynomial payload")
                physical = physical + self.g2_retained_source_tm[index]
                if g2_state.fresh_active[index]:
                    radius = float.fromhex(g2_state.fresh_radii_hex[index])
                    source = TaylorModel.variable(
                        2 * g2_state.state_dim + index,
                        self.domain,
                        order=model.order,
                    )
                    physical = physical + source * radius
            models.append(physical)
        return TMVector(models)

    def range_box(self) -> list[Interval]:
        return self.endpoint_tm().range_box()

    def normalized_initial_tm(self, order: int | None = None) -> TMVector:
        if self.g2_shared_column_state is not None:
            if self.g2_retained_source_tm is None:
                raise ValueError("G2 normal state is missing retained polynomial payload")
            return _g2_shared_column_reset_tm(
                self.center,
                self.scales,
                self.g2_retained_source_tm,
                self.g2_shared_column_state,
                int(order) if order is not None else _tm_max_degree(self.tmv_pre),
                self.domain,
            )
        if self.bounded_source_ledger_state is not None:
            return _bounded_source_ledger_affine_reset_tm(
                self.center,
                self.scales,
                self.bounded_source_ledger_state,
                int(order) if order is not None else _tm_max_degree(self.tmv_pre),
                self.domain,
            )
        if self.complete_initial_tm is not None:
            selected_order = int(order) if order is not None else _tm_max_degree(self.complete_initial_tm)
            if len(self.complete_initial_tm) != len(self.center):
                raise ValueError("complete polynomial carry state dimension disagrees with center/scale")
            if self.complete_initial_tm.n_vars != len(self.domain):
                raise ValueError("complete polynomial carry variable/domain dimension disagrees")
            if _tm_max_degree(self.complete_initial_tm) > selected_order:
                raise ValueError("complete polynomial carry exceeds requested order")
            if any(
                not torch.equal(left.lo, right.lo) or not torch.equal(left.hi, right.hi)
                for left, right in zip(self.complete_initial_tm.domain, self.domain)
            ):
                raise ValueError("complete polynomial carry domain disagrees with normal state")
            return TMVector(model.clone() for model in self.complete_initial_tm)
        tmv = _normalized_tm_from_center_scale(
            self.center,
            self.scales,
            int(order) if order is not None else _tm_max_degree(self.tmv_pre),
            template_domain=self.domain,
        )
        if self.initial_remainders:
            return TMVector(
                model.with_remainder(model.remainder + rem)
                for model, rem in zip(tmv, self.initial_remainders)
            )
        return tmv

    def with_bounded_source_g1(self, order: int) -> "FlowstarNormalFlowpipeState":
        """Return the explicitly initialized fixed-2d G1 state."""

        return _initialize_bounded_source_normal_state(self, int(order))

    def with_g2_shared_columns(self, order: int) -> "FlowstarNormalFlowpipeState":
        """Return the explicitly initialized fixed-3d G2 state."""

        return _initialize_g2_shared_column_normal_state(self, int(order))

    def diagnostic_widths(self) -> dict[str, Any]:
        diagnostics = {
            "normal_state_width_sum": _sum_interval_widths(self.range_box()),
            "normal_state_right_width_sum": _sum_interval_widths(self.tmv_right.range_box()),
            "normal_state_scale_sum": sum(abs(float(s)) for s in self.scales),
        }
        if self.complete_initial_tm is not None:
            diagnostics.update(
                {
                    "complete_polynomial_carry": True,
                    "complete_carry_retained_terms": sum(
                        len(model.polynomial.terms) for model in self.complete_initial_tm
                    ),
                    "complete_carry_max_degree": _tm_max_degree(self.complete_initial_tm),
                    "complete_carry_intervalized_term_count": 0,
                    "complete_carry_remainder_width_sum": sum(
                        _interval_width_float(model.remainder) for model in self.complete_initial_tm
                    ),
                }
            )
        return diagnostics


@dataclass(frozen=True)
class HornerInsertionDiagnosticResult:
    """Diagnostic-only comparison of direct and Horner normal insertion."""

    direct_result: TaylorModel | TMVector
    horner_result: TaylorModel | TMVector
    summary: Mapping[str, Any]
    stage_ranges: Sequence[Mapping[str, Any]]
    top_components: Sequence[Mapping[str, Any]]


def _as_interval_list(x0_box: Sequence[Interval | tuple[float, float] | list[float] | float]) -> list[Interval]:
    out: list[Interval] = []
    for x in x0_box:
        if isinstance(x, Interval):
            out.append(x)
        elif isinstance(x, (tuple, list)) and len(x) == 2:
            out.append(Interval(x[0], x[1]))
        else:
            out.append(Interval.point(x))
    return out


def _exact_fraction(value: str | int | Fraction) -> Fraction:
    if isinstance(value, float):
        raise TypeError("exact decimal initialization rejects binary floating-point inputs")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"unsupported exact decimal endpoint type: {type(value).__name__}")


def _fraction_text(value: Fraction) -> str:
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def _fraction_to_binary64_up(value: Fraction) -> float:
    value = Fraction(value)
    if value < 0:
        raise ValueError("outward affine radius must be nonnegative")
    candidate = float(value)
    if not math.isfinite(candidate):
        raise OverflowError("exact affine radius is not finite in binary64")
    if Fraction.from_float(candidate) < value:
        candidate = math.nextafter(candidate, math.inf)
    return candidate


def _zero_interval_like_domain(domain: Sequence[Interval]) -> Interval:
    if domain:
        return Interval.zero(dtype=domain[0].lo.dtype, device=domain[0].lo.device)
    return Interval.zero()


def _zero_remainder_tm(
    poly: Polynomial,
    domain: Sequence[Interval],
    order: int,
    *,
    truncation_range_split: int | None = None,
) -> TaylorModel:
    return TaylorModel(
        poly,
        _zero_interval_like_domain(domain),
        list(domain),
        order=order,
        truncation_range_split=truncation_range_split,
    )


def _float_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if torch.isfinite(torch.as_tensor(f)).item() else None


def _interval_width_value(iv: Interval) -> float | None:
    return _float_or_none(iv.width().detach().cpu())


def _interval_bound_value(value: torch.Tensor) -> float | None:
    return _float_or_none(value.detach().cpu())


def _interval_is_zero(iv: Interval) -> bool:
    return bool(torch.all(iv.lo == 0) and torch.all(iv.hi == 0))


def _combine_remainders(base: Interval, extra: Interval) -> Interval:
    return base if _interval_is_zero(extra) else base + extra


def _truncation_split_value(value: int | None) -> int | None:
    if value is None:
        return None
    pieces = int(value)
    return pieces if pieces > 1 else None


def _poly_interval_with_split(poly: Polynomial, domain: Sequence[Interval], split: int | None) -> Interval:
    pieces = _truncation_split_value(split)
    if pieces is None:
        return poly.evaluate_interval(domain)
    return poly.evaluate_interval_split(domain, pieces)


def _normal_domain(domain: Sequence[Interval], tau_index: int | None = None) -> list[Interval]:
    normal: list[Interval] = []
    for i, iv in enumerate(domain):
        if tau_index is not None and i == tau_index:
            normal.append(iv)
        else:
            normal.append(_unit_interval_like(iv))
    return normal


def _poly_interval_normal(
    poly: Polynomial,
    domain: Sequence[Interval],
    tau_index: int | None = None,
    *,
    normal_eval_range_split: int | None = None,
) -> Interval:
    """Flow*-style normal interval evaluation for normalized local domains."""
    pieces = _truncation_split_value(normal_eval_range_split)
    if pieces is None:
        state_indices = [i for i in range(poly.n_vars) if i != tau_index]
        return evaluate_interval_normal(
            poly,
            domain,
            state_var_indices=state_indices,
            time_var_index=tau_index,
        )
    normal = _normal_domain(domain, tau_index)
    split_vars = [i for i in range(len(normal)) if i != tau_index]
    return poly.evaluate_interval_split(normal, pieces, split_vars=split_vars)


def _cutoff_polynomial_normal(
    poly: Polynomial,
    domain: Sequence[Interval],
    tau_index: int | None,
    threshold: float | None,
    *,
    normal_eval_range_split: int | None = None,
) -> tuple[Polynomial, Interval]:
    if threshold is None:
        return poly, Interval.zero(dtype=poly.dtype, device=poly.device)
    kept: dict[tuple[int, ...], Any] = {}
    removed: dict[tuple[int, ...], Any] = {}
    threshold_t = torch.as_tensor(abs(float(threshold)), dtype=poly.dtype, device=poly.device)
    for exp, coef in poly.terms.items():
        target = removed if bool(torch.all(torch.abs(coef) <= threshold_t)) else kept
        target[exp] = coef
    removed_poly = Polynomial(removed, poly.n_vars)
    removed_range = (
        _poly_interval_normal(
            removed_poly,
            domain,
            tau_index,
            normal_eval_range_split=normal_eval_range_split,
        )
        if removed
        else Interval.zero(dtype=poly.dtype, device=poly.device)
    )
    return Polynomial(kept, poly.n_vars), removed_range


def _term_interval(exp: tuple[int, ...], coef: Any, domain: Sequence[Interval]) -> Interval:
    term_iv = Interval.point(coef)
    for power, dom in zip(exp, domain):
        if power:
            term_iv = term_iv * dom.pow_int(power)
    return term_iv


def _interval_abs_extent(iv: Interval) -> float:
    return max(abs(float(iv.lo.detach().cpu())), abs(float(iv.hi.detach().cpu())))


def _monomial_label(exp: tuple[int, ...]) -> str:
    names = ["x", "y", "tau"]
    parts: list[str] = []
    for i, power in enumerate(exp):
        if power == 0:
            continue
        name = names[i] if i < len(names) else f"z{i}"
        parts.append(name if power == 1 else f"{name}^{power}")
    return "1" if not parts else "*".join(parts)


def _float_value(value: Any) -> float | str:
    out = _float_or_none(value)
    return out if out is not None else ""


def _tm_terms_signature(tm: TMVector) -> str:
    parts: list[str] = []
    for state_index, model in enumerate(tm):
        for exp, coef in sorted(model.polynomial.terms.items()):
            coef_f = _float_value(coef.detach().cpu() if hasattr(coef, "detach") else coef)
            parts.append(f"{state_index}:{','.join(str(e) for e in exp)}:{coef_f}")
    return "|".join(parts)


def _tm_terms_hash(tm: TMVector) -> str:
    return hashlib.sha256(_tm_terms_signature(tm).encode("utf-8")).hexdigest()[:16]


def _tm_high_degree_term_count(tm: TMVector, output_order: int) -> int:
    return sum(1 for model in tm for exp in model.polynomial.terms if sum(exp) > int(output_order))


def _tm_max_degree(tm: TMVector) -> int:
    return max((sum(exp) for model in tm for exp in model.polynomial.terms), default=0)


def preserve_complete_polynomial_carry(endpoint: Any) -> Any:
    """Clone a validated endpoint without intervalizing retained terms.

    This is the single operation changed by the experimental complete-carry
    lane.  Sparse ``TMVector`` inputs cover the current adaptive runner.  Dense
    ``BatchedTaylorModel`` inputs preserve the full ``[B, state, slot]`` tensor,
    including per-batch domains, remainders, and ledger entries, so the carry
    primitive itself is independent of batch size and device.  Endpoint-time
    substitution must already have happened; this function deliberately does
    not reinterpret coordinates or perform a range reduction.
    """
    if isinstance(endpoint, TMVector):
        if not endpoint:
            raise ValueError("complete polynomial carry requires a non-empty endpoint")
        if not intervals_are_finite(endpoint.range_box()):
            raise ValueError("complete polynomial carry rejects a non-finite endpoint")
        return TMVector(model.clone() for model in endpoint)

    # Import lazily to keep the sparse reference path free of a dense-module
    # import cycle and to preserve the package's protocol-only import behavior.
    from .batched_dense_tm import BatchedTaylorModel

    if isinstance(endpoint, BatchedTaylorModel):
        if endpoint.poly.batch <= 0 or endpoint.poly.out_dim <= 0:
            raise ValueError("complete polynomial carry requires non-empty batch and state axes")
        if not endpoint.is_finite():
            raise ValueError("complete polynomial carry rejects a non-finite endpoint")
        return endpoint.clone()
    raise TypeError("complete polynomial carry requires TMVector or BatchedTaylorModel")


def _add_term_hash_metrics(row: dict[str, Any], prefix: str, tm: TMVector, output_order: int | None) -> None:
    row[f"{prefix}_terms_hash"] = _tm_terms_hash(tm)
    row[f"{prefix}_term_count"] = sum(len(model.polynomial.terms) for model in tm)
    row[f"{prefix}_max_degree"] = _tm_max_degree(tm)
    if output_order is not None:
        row[f"{prefix}_high_degree_term_count"] = _tm_high_degree_term_count(tm, int(output_order))


def _truncate_tm_to_order(tm: TMVector, output_order: int) -> TMVector:
    truncated, _stats, _details = _truncate_tm_to_order_selective(tm, output_order, selective_top_k=None)
    return truncated


def _truncate_tm_to_order_selective(
    tm: TMVector,
    output_order: int,
    *,
    selective_top_k: int | None = None,
    result_order: int | None = None,
) -> tuple[TMVector, list[dict[str, Any]], list[dict[str, Any]]]:
    models: list[TaylorModel] = []
    stats: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    top_k = int(selective_top_k or 0)
    state_names = ("x", "y")
    for state_index, model in enumerate(tm):
        kept, dropped = model.polynomial.truncate(int(output_order))
        dropped_terms = list(dropped.terms.items())
        retained_terms: dict[tuple[int, ...], Any] = {}
        nonkept_terms: dict[tuple[int, ...], Any] = dict(dropped.terms)
        ranked: list[tuple[float, tuple[int, ...], Any, Interval]] = []
        if top_k > 0 and dropped_terms:
            for exp, coef in dropped_terms:
                term_iv = _term_interval(tuple(exp), coef, model.domain)
                ranked.append((_interval_abs_extent(term_iv), tuple(exp), coef, term_iv))
            ranked.sort(key=lambda item: item[0], reverse=True)
            for rank, (abs_extent, exp, coef, term_iv) in enumerate(ranked, start=1):
                retained = rank <= top_k
                if retained:
                    retained_terms[exp] = coef
                    nonkept_terms.pop(exp, None)
                details.append(
                    {
                        "state_index": state_index,
                        "state_dimension": state_names[state_index] if state_index < len(state_names) else f"state_{state_index}",
                        "term_rank": rank,
                        "retained": retained,
                        "monomial": _monomial_label(exp),
                        "coefficient": _float_value(coef.detach().cpu() if hasattr(coef, "detach") else coef),
                        "total_degree": sum(exp),
                        "abs_interval_contribution": abs_extent,
                        "term_interval_lo": _float_value(term_iv.lo.detach().cpu()),
                        "term_interval_hi": _float_value(term_iv.hi.detach().cpu()),
                        "term_interval_width": _float_value(term_iv.width().detach().cpu()),
                    }
                )
        sparse_poly = kept + Polynomial(retained_terms, kept.n_vars) if retained_terms else kept
        nonkept = Polynomial(nonkept_terms, kept.n_vars)
        dropped_range = _poly_interval_with_split(nonkept, model.domain, model.truncation_range_split)
        total_dropped_range = _poly_interval_with_split(dropped, model.domain, model.truncation_range_split)
        models.append(
            TaylorModel(
                sparse_poly,
                model.remainder + dropped_range,
                list(model.domain),
                order=int(result_order if result_order is not None else output_order),
                truncation_range_split=model.truncation_range_split,
            )
        )
        stats.append(
            {
                "state_index": state_index,
                "state_dimension": state_names[state_index] if state_index < len(state_names) else f"state_{state_index}",
                "selective_high_degree_terms_top_k": top_k if top_k > 0 else "",
                "selective_retained_terms_count": len(retained_terms),
                "selective_dropped_terms_count": len(dropped_terms),
                "selective_nonretained_terms_count": len(nonkept_terms),
                "selective_dropped_remainder_lo": _float_value(dropped_range.lo.detach().cpu()),
                "selective_dropped_remainder_hi": _float_value(dropped_range.hi.detach().cpu()),
                "selective_dropped_remainder_width": _float_value(dropped_range.width().detach().cpu()),
                "selective_total_dropped_width": _float_value(total_dropped_range.width().detach().cpu()),
            }
        )
    return TMVector(models), stats, details


def _aggregate_selective_stats(
    stats: Sequence[Mapping[str, Any]],
    *,
    top_k: int | None,
) -> dict[str, Any]:
    if not top_k:
        return {}
    retained = sum(int(row.get("selective_retained_terms_count") or 0) for row in stats)
    dropped = sum(int(row.get("selective_dropped_terms_count") or 0) for row in stats)
    nonretained = sum(int(row.get("selective_nonretained_terms_count") or 0) for row in stats)
    rem_width = 0.0
    total_width = 0.0
    for row in stats:
        rem_width += _float_or_none(row.get("selective_dropped_remainder_width")) or 0.0
        total_width += _float_or_none(row.get("selective_total_dropped_width")) or 0.0
    return {
        "selective_high_degree_terms_top_k": int(top_k),
        "selective_retained_terms_count": retained,
        "selective_dropped_terms_count": dropped,
        "selective_nonretained_terms_count": nonretained,
        "selective_dropped_remainder_width_sum": rem_width,
        "selective_total_dropped_width_sum": total_width,
    }


def _symmetric_interval(radius: float, domain: Sequence[Interval]) -> Interval:
    r = abs(float(radius))
    if domain:
        dtype = domain[0].lo.dtype
        device = domain[0].lo.device
        return Interval(
            torch.as_tensor(-r, dtype=dtype, device=device),
            torch.as_tensor(r, dtype=dtype, device=device),
        )
    return Interval(-r, r)


def _unit_interval_like(iv: Interval) -> Interval:
    return Interval(
        torch.as_tensor(-1.0, dtype=iv.lo.dtype, device=iv.lo.device),
        torch.as_tensor(1.0, dtype=iv.lo.dtype, device=iv.lo.device),
    )


def _normalized_tm_from_box(x_box: Sequence[Interval | tuple[float, float] | list[float] | float], order: int) -> TMVector:
    boxes = _as_interval_list(x_box)
    var_for_dim: list[int | None] = []
    domain: list[Interval] = []
    for iv in boxes:
        if bool(torch.all(iv.radius() == 0)):
            var_for_dim.append(None)
        else:
            var_for_dim.append(len(domain))
            domain.append(_unit_interval_like(iv))

    models: list[TaylorModel] = []
    n_vars = len(domain)
    for iv, var_index in zip(boxes, var_for_dim):
        center = iv.mid()
        if var_index is None:
            models.append(TaylorModel.constant(center, domain, order=order))
            continue
        radius = iv.radius()
        poly = Polynomial.constant(center, n_vars) + Polynomial.variable(
            var_index, n_vars, dtype=center.dtype, device=center.device
        ) * radius
        models.append(TaylorModel(poly, Interval.zero(dtype=center.dtype, device=center.device), domain, order=order))
    return TMVector(models)


def _normalized_tm_from_center_scale(
    centers: Sequence[Any],
    scales: Sequence[Any],
    order: int,
    *,
    template_domain: Sequence[Interval] | None = None,
) -> TMVector:
    template = list(template_domain or [])
    if template:
        dtype = template[0].lo.dtype
        device = template[0].lo.device
    else:
        dtype = torch.float64
        device = torch.device("cpu")
    domain = [Interval(torch.as_tensor(-1.0, dtype=dtype, device=device), torch.as_tensor(1.0, dtype=dtype, device=device)) for _ in centers]
    models: list[TaylorModel] = []
    n_vars = len(domain)
    for i, (center, scale) in enumerate(zip(centers, scales)):
        c = torch.as_tensor(center, dtype=dtype, device=device)
        s = torch.as_tensor(scale, dtype=dtype, device=device)
        poly = Polynomial.constant(c, n_vars)
        if bool(torch.any(s != 0)):
            poly = poly + Polynomial.variable(i, n_vars, dtype=dtype, device=device) * s
        models.append(TaylorModel(poly, Interval.zero(dtype=dtype, device=device), domain, order=order))
    return TMVector(models)


def _bounded_source_ledger_affine_reset_tm(
    centers: Sequence[Any],
    scales: Sequence[Any],
    source_state: BoundedSourceLedgerState,
    order: int,
    domain: Sequence[Interval],
) -> TMVector:
    """Materialize the fixed-shape affine reset consumed by dense Picard."""

    dim = source_state.state_dim
    domain_l = list(domain)
    if len(centers) != dim or len(scales) != dim:
        raise ValueError("bounded source reset center/scale dimension mismatch")
    if len(domain_l) != 2 * dim:
        raise ValueError("bounded source reset requires exactly 2*state_dim variables")
    dtype = domain_l[0].lo.dtype
    device = domain_l[0].lo.device
    models: list[TaylorModel] = []
    for index, (center, scale) in enumerate(zip(centers, scales)):
        poly = Polynomial.constant(
            torch.as_tensor(center, dtype=dtype, device=device), 2 * dim
        )
        scale_t = torch.as_tensor(scale, dtype=dtype, device=device)
        if bool(torch.any(scale_t != 0)):
            poly = poly + Polynomial.variable(
                index, 2 * dim, dtype=dtype, device=device
            ) * scale_t
        if source_state.active[index]:
            radius = torch.as_tensor(
                float.fromhex(source_state.radii_hex[index]),
                dtype=dtype,
                device=device,
            )
            if bool(torch.any(radius != 0)):
                poly = poly + Polynomial.variable(
                    dim + index, 2 * dim, dtype=dtype, device=device
                ) * radius
        models.append(
            TaylorModel(poly, Interval.zero(dtype=dtype, device=device), domain_l, order=order)
        )
    return TMVector(models)


def _initialize_bounded_source_normal_state(
    state: FlowstarNormalFlowpipeState,
    order: int,
) -> FlowstarNormalFlowpipeState:
    """Extend an ordinary initial normal state with inactive fixed source slots."""

    dim = len(state.center)
    if len(state.domain) != dim or state.tmv_right.n_vars != dim:
        raise ValueError("bounded source initialization requires the canonical d-variable normal state")
    tmv_pre = state.tmv_pre
    tmv_right = state.tmv_right
    domain = list(state.domain)
    for _ in range(dim):
        unit = _unit_interval_like(domain[0])
        tmv_pre = tmv_pre.extend_domain(unit)
        tmv_right = tmv_right.extend_domain(unit)
        domain.append(unit)
    source_state = BoundedSourceLedgerState.initial(dim)
    initialized = replace(
        state,
        tmv_pre=tmv_pre,
        tmv_right=tmv_right,
        domain=domain,
        bounded_source_ledger_state=source_state,
        diagnostics={
            **dict(state.diagnostics or {}),
            "reset_mode": BOUNDED_SOURCE_LEDGER_CANDIDATE,
            "bounded_source_ledger_initial_state": True,
            "source_ledger_schema": source_state.schema,
            "source_ledger_fingerprint": source_state.fingerprint,
            "source_ledger_live_source_count": 0,
        },
    )
    # Exercise the exact materialization invariant immediately rather than
    # allowing a malformed extended state to reach the Picard consumer.
    initialized.normalized_initial_tm(order)
    return initialized


def _g2_zero_retained_tm(
    state_dim: int,
    domain: Sequence[Interval],
    order: int,
) -> TMVector:
    dim = int(state_dim)
    domain_l = list(domain)
    if len(domain_l) != 3 * dim:
        raise ValueError("G2 retained payload requires exactly 3d variables")
    dtype = domain_l[0].lo.dtype
    device = domain_l[0].lo.device
    return TMVector(
        TaylorModel(
            Polynomial.zero(3 * dim, dtype=dtype, device=device),
            Interval.zero(dtype=dtype, device=device),
            domain_l,
            order=int(order),
        )
        for _ in range(dim)
    )


def _g2_shared_column_reset_tm(
    centers: Sequence[Any],
    scales: Sequence[Any],
    retained_tm: TMVector,
    source_state: G2SharedColumnState,
    order: int,
    domain: Sequence[Interval],
) -> TMVector:
    """Build the sole G2 physical reset consumed by the next dense Picard."""

    dim = source_state.state_dim
    domain_l = list(domain)
    if len(domain_l) != 3 * dim or retained_tm.n_vars != 3 * dim:
        raise ValueError("G2 reset is fixed to exactly 3d variables")
    if len(centers) != dim or len(scales) != dim or len(retained_tm) != dim:
        raise ValueError("G2 reset state dimension mismatch")
    if g2_polynomial_payload_sha256([model.polynomial for model in retained_tm]) != source_state.retained_payload_sha256:
        raise ValueError("G2 retained polynomial payload hash mismatch")
    if any(
        not torch.equal(retained_interval.lo, reset_interval.lo)
        or not torch.equal(retained_interval.hi, reset_interval.hi)
        for retained_interval, reset_interval in zip(retained_tm.domain, domain_l)
    ):
        raise ValueError("G2 retained polynomial domain mismatch")
    dtype = domain_l[0].lo.dtype
    device = domain_l[0].lo.device
    models: list[TaylorModel] = []
    for index in range(dim):
        retained = retained_tm[index]
        if bool(torch.any(retained.remainder.lo != 0) or torch.any(retained.remainder.hi != 0)):
            raise ValueError("G2 retained sources cannot carry an ordinary remainder")
        poly = retained.polynomial
        if any(
            exponent[2 * dim + offset]
            for exponent in poly.terms
            for offset in range(dim)
        ):
            raise ValueError("G2 retained payload illegally occupies the fresh bank")
        poly = poly + Polynomial.constant(
            torch.as_tensor(centers[index], dtype=dtype, device=device), 3 * dim
        )
        scale = torch.as_tensor(scales[index], dtype=dtype, device=device)
        if bool(torch.any(scale != 0)):
            poly = poly + Polynomial.variable(index, 3 * dim, dtype=dtype, device=device) * scale
        if source_state.fresh_active[index]:
            radius = torch.as_tensor(
                float.fromhex(source_state.fresh_radii_hex[index]),
                dtype=dtype,
                device=device,
            )
            if bool(torch.any(radius != 0)):
                poly = poly + Polynomial.variable(
                    2 * dim + index, 3 * dim, dtype=dtype, device=device
                ) * radius
        models.append(
            TaylorModel(
                poly,
                Interval.zero(dtype=dtype, device=device),
                domain_l,
                order=int(order),
            )
        )
    return TMVector(models)


def _initialize_g2_shared_column_normal_state(
    state: FlowstarNormalFlowpipeState,
    order: int,
) -> FlowstarNormalFlowpipeState:
    """Extend the initial normal state with two inactive d-slot banks."""

    dim = len(state.center)
    if len(state.domain) != dim or state.tmv_right.n_vars != dim:
        raise ValueError("G2 initialization requires the canonical d-variable normal state")
    tmv_pre = state.tmv_pre
    tmv_right = state.tmv_right
    domain = list(state.domain)
    for _ in range(2 * dim):
        unit = _unit_interval_like(domain[0])
        tmv_pre = tmv_pre.extend_domain(unit)
        tmv_right = tmv_right.extend_domain(unit)
        domain.append(unit)
    source_state = G2SharedColumnState.initial(dim)
    retained = _g2_zero_retained_tm(dim, domain, order)
    initialized = replace(
        state,
        tmv_pre=tmv_pre,
        tmv_right=tmv_right,
        domain=domain,
        g2_shared_column_state=source_state,
        g2_retained_source_tm=retained,
        diagnostics={
            **dict(state.diagnostics or {}),
            "reset_mode": G2_SHARED_COLUMN_CANDIDATE,
            "g2_initial_state": True,
            "g2_schema": source_state.schema,
            "g2_fingerprint": source_state.fingerprint,
            "g2_variable_count": 3 * dim,
            "g2_live_source_count": 0,
        },
    )
    initialized.normalized_initial_tm(order)
    return initialized


def _tm_with_order(model: TaylorModel, order: int) -> TaylorModel:
    return TaylorModel(
        model.polynomial,
        model.remainder,
        list(model.domain),
        order=order,
        truncation_range_split=model.truncation_range_split,
    )


def _tmvector_with_order(tmv: TMVector, order: int) -> TMVector:
    return TMVector(_tm_with_order(model, order) for model in tmv)


def _diag_add_width(diagnostics: dict[str, Any] | None, key: str, value: Interval | float | int | None) -> None:
    if diagnostics is None or value is None:
        return
    if isinstance(value, Interval):
        numeric = _interval_width_value(value)
    else:
        numeric = _float_or_none(value)
    if numeric is None:
        return
    diagnostics[key] = (_float_or_none(diagnostics.get(key)) or 0.0) + float(numeric)
    if key.endswith("_width"):
        diagnostics[f"{key}_sum"] = diagnostics[key]


def _diag_add_component_width(
    diagnostics: dict[str, Any] | None,
    key: str,
    value: Interval | float | int | None,
    component_index: int | None,
) -> None:
    if diagnostics is None or value is None or component_index is None:
        return
    if isinstance(value, Interval):
        numeric = _interval_width_value(value)
    else:
        numeric = _float_or_none(value)
    if numeric is None:
        return
    names = ("x", "y")
    name = names[component_index] if 0 <= int(component_index) < len(names) else f"state_{component_index}"
    component_key = f"{key}_{name}"
    diagnostics[component_key] = (_float_or_none(diagnostics.get(component_key)) or 0.0) + float(numeric)
    if key.endswith("_width"):
        diagnostics[f"{key}_sum"] = _float_or_none(diagnostics.get(key)) or 0.0


def _diag_add_count(diagnostics: dict[str, Any] | None, key: str, value: int) -> None:
    if diagnostics is None:
        return
    diagnostics[key] = int(diagnostics.get(key) or 0) + int(value)


def _compose_term_with_inner(
    coef: Any,
    exp: tuple[int, ...],
    inner: TMVector,
    *,
    work_order: int,
    domain: Sequence[Interval],
) -> TaylorModel:
    term = TaylorModel.constant(coef, domain, order=work_order)
    for var_index, power in enumerate(exp):
        for _ in range(int(power)):
            term = term * inner[var_index]
    return _tm_with_order(term, work_order)


def _insert_ctrunc_normal_like_scalar(
    outer: TaylorModel,
    inner: TMVector,
    order: int,
    cutoff_threshold: float | None,
    domain: Sequence[Interval] | None,
    diagnostics: dict[str, Any] | None,
    component_index: int | None = None,
) -> TaylorModel:
    if outer.polynomial.n_vars != len(inner):
        raise ValueError("outer TaylorModel variable count must match inner TMVector length")
    out_domain = list(domain or inner.domain)
    if len(out_domain) != inner.n_vars:
        raise ValueError("composition output domain must match inner TaylorModel domain")
    inner_degree = max((_tm_max_degree(TMVector([m])) for m in inner), default=0)
    outer_degree = outer.polynomial.degree()
    work_order = max(int(order), int(outer_degree) * max(1, int(inner_degree)))
    inner_work = _tmvector_with_order(inner, work_order)
    acc = TaylorModel.zero(out_domain, order=work_order, truncation_range_split=outer.truncation_range_split)
    for exp, coef in outer.polynomial.terms.items():
        term = _compose_term_with_inner(coef, tuple(exp), inner_work, work_order=work_order, domain=out_domain)
        acc = acc + term

    kept, dropped = acc.polynomial.truncate(int(order))
    trunc_range = _poly_interval_with_split(dropped, out_domain, acc.truncation_range_split)
    cutoff_kept, cutoff_range = kept.cutoff(cutoff_threshold, out_domain)
    cutoff_removed = kept - cutoff_kept
    remainder = acc.remainder + outer.remainder + trunc_range + cutoff_range
    result = TaylorModel(
        cutoff_kept,
        remainder,
        out_domain,
        order=int(order),
        truncation_range_split=acc.truncation_range_split,
    )

    composed_poly_range = cutoff_kept.evaluate_interval(out_domain)
    _diag_add_width(diagnostics, "insertion_truncation_width", trunc_range)
    _diag_add_component_width(diagnostics, "insertion_truncation_width", trunc_range, component_index)
    _diag_add_width(diagnostics, "insertion_cutoff_width", cutoff_range)
    _diag_add_component_width(diagnostics, "insertion_cutoff_width", cutoff_range, component_index)
    _diag_add_width(diagnostics, "composed_poly_range_width", composed_poly_range)
    _diag_add_component_width(diagnostics, "composed_poly_range_width", composed_poly_range, component_index)
    _diag_add_width(diagnostics, "output_remainder_width", result.remainder)
    _diag_add_component_width(diagnostics, "output_remainder_width", result.remainder, component_index)
    _diag_add_count(diagnostics, "terms_before_insertion_truncation", len(acc.polynomial.terms))
    _diag_add_count(diagnostics, "terms_after_insertion", len(result.polynomial.terms))
    if diagnostics is not None:
        owner_rows = diagnostics.setdefault("_insertion_owner_rows", [])
        if not isinstance(owner_rows, list):
            raise ValueError("insertion owner diagnostics must be a list")
        for category, owner_poly, enclosure in (
            ("insertion_truncation", dropped, trunc_range),
            ("insertion_cutoff", cutoff_removed, cutoff_range),
        ):
            payload = g2_polynomial_table(owner_poly)
            owner_rows.append(
                {
                    "category": category,
                    "component": component_index,
                    "canonical_support_sha256": hashlib.sha256(
                        json.dumps(
                            [row[0] for row in payload],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "coefficient_payload_sha256": hashlib.sha256(
                        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "term_count": len(owner_poly.terms),
                    "outward_lo_hex": float(enclosure.lo.detach().cpu()).hex(),
                    "outward_hi_hex": float(enclosure.hi.detach().cpu()).hex(),
                    "width": float(enclosure.width().detach().cpu()),
                    "containment_witness": "discarded_canonical_polynomial_outward_range",
                }
            )
    return result


def insert_ctrunc_normal_like(
    outer: TaylorModel | TMVector,
    inner: TMVector,
    order: int,
    cutoff_threshold: float | None,
    domain: Sequence[Interval] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> TaylorModel | TMVector:
    """Clean-room conservative analogue of Flow* normal insertion.

    The helper substitutes ``inner`` into ``outer`` over the normalized output
    domain, truncates to ``order``, and moves truncation/cutoff uncertainty into
    interval remainders. It is intentionally straightforward rather than a copy
    of Flow*'s Horner implementation.
    """
    if isinstance(outer, TMVector):
        models = [
            _insert_ctrunc_normal_like_scalar(
                model,
                inner,
                order,
                cutoff_threshold,
                domain,
                diagnostics,
                component_index=index,
            )
            for index, model in enumerate(outer)
        ]
        if diagnostics is not None:
            diagnostics["insertion_components"] = len(models)
        return TMVector(models)
    return _insert_ctrunc_normal_like_scalar(outer, inner, order, cutoff_threshold, domain, diagnostics)


def insert_ctrunc_normal_dependency_preserving(
    outer: TaylorModel | TMVector,
    inner: TMVector,
    order: int,
    cutoff_threshold: float | None,
    domain: Sequence[Interval] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> TaylorModel | TMVector:
    """Factorized production insertion in the frozen order ``0, ..., n-1``.

    Unlike the legacy monomial loop, the recursive Horner graph shares each
    coefficient branch before multiplying by an inserted map.  This reduces
    repeated materialization of the same ordinary remainder across sibling
    monomial paths.  Truncation, cutoff, and all remainder products remain
    outward interval operations at every multiplication stage.
    """

    canonical_order = tuple(range(len(inner)))
    stage_rows: list[dict[str, Any]] = []
    top_components: list[dict[str, Any]] = []
    if isinstance(outer, TMVector):
        result: TaylorModel | TMVector = TMVector(
            _insert_ctrunc_normal_horner_scalar(
                model,
                inner,
                int(order),
                cutoff_threshold,
                domain,
                component_index=index,
                stage_rows=stage_rows,
                top_components=top_components,
            )
            for index, model in enumerate(outer)
        )
    else:
        result = _insert_ctrunc_normal_horner_scalar(
            outer,
            inner,
            int(order),
            cutoff_threshold,
            domain,
            component_index=None,
            stage_rows=stage_rows,
            top_components=top_components,
        )
    if diagnostics is not None:
        models = result.models if isinstance(result, TMVector) else [result]
        diagnostics["insertion_dependency_preserving_used"] = True
        diagnostics["insertion_canonical_variable_order"] = list(canonical_order)
        diagnostics["insertion_components"] = len(models)
        diagnostics["insertion_factorized_multiplication_count"] = sum(
            row.get("operation") == "multiply_inserted_right_map"
            for row in stage_rows
        )
        diagnostics["_dependency_preserving_stage_rows"] = [
            dict(row) for row in stage_rows
        ]
        diagnostics["_dependency_preserving_top_components"] = sorted(
            (dict(row) for row in top_components),
            key=lambda row: _float_or_none(row.get("width")) or 0.0,
            reverse=True,
        )
        diagnostics["insertion_truncation_width"] = _stage_width_sum(
            stage_rows, "truncation_width"
        )
        diagnostics["insertion_cutoff_width"] = _stage_width_sum(
            stage_rows, "cutoff_width"
        )
        diagnostics["insertion_inner_remainder_times_poly_width"] = _stage_width_sum(
            stage_rows, "p_left_times_right_remainder_width"
        )
        diagnostics["insertion_accumulated_remainder_times_inner_poly_width"] = _stage_width_sum(
            stage_rows, "p_right_times_left_remainder_width"
        )
        diagnostics["insertion_remainder_times_poly_width"] = (
            diagnostics["insertion_inner_remainder_times_poly_width"]
            + diagnostics["insertion_accumulated_remainder_times_inner_poly_width"]
        )
        diagnostics["insertion_remainder_times_remainder_width"] = _stage_width_sum(
            stage_rows, "remainder_times_remainder_width"
        )
        diagnostics["terms_after_insertion"] = sum(
            len(model.polynomial.terms) for model in models
        )
        for component_index, model in enumerate(models):
            name = _horner_component_name(component_index)
            poly_range = model.polynomial.evaluate_interval(model.domain)
            _diag_add_width(diagnostics, "composed_poly_range_width", poly_range)
            diagnostics[f"composed_poly_range_width_{name}"] = _interval_width_float(poly_range)
            _diag_add_width(diagnostics, "output_remainder_width", model.remainder)
            diagnostics[f"output_remainder_width_{name}"] = _interval_width_float(model.remainder)
            diagnostics[f"insertion_truncation_width_{name}"] = _stage_width_sum(
                stage_rows, "truncation_width", component_index
            )
            diagnostics[f"insertion_cutoff_width_{name}"] = _stage_width_sum(
                stage_rows, "cutoff_width", component_index
            )
    return result




def _object_range_box(obj: TaylorModel | TMVector) -> list[Interval]:
    if isinstance(obj, TMVector):
        return obj.range_box()
    return [obj.range_box()]


def _object_normal_range_box(obj: TaylorModel | TMVector) -> list[Interval]:
    if isinstance(obj, TMVector):
        return _tmvector_range_box_normal(obj, None)
    return [_taylor_model_range_box_normal(obj, None)]


def _object_range_width_sum(obj: TaylorModel | TMVector) -> float:
    value = _sum_interval_widths(_object_range_box(obj))
    return _float_or_none(value) or 0.0


def _object_normal_range_width_sum(obj: TaylorModel | TMVector) -> float:
    value = _sum_interval_widths(_object_normal_range_box(obj))
    return _float_or_none(value) or 0.0


def _interval_width_float(value: Interval | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Interval):
        return _interval_width_value(value) or 0.0
    return _float_or_none(value) or 0.0


def _merged_split_value(*values: int | None) -> int | None:
    pieces = [_truncation_split_value(value) for value in values]
    pieces = [value for value in pieces if value is not None]
    return max(pieces) if pieces else None


def _polynomial_part_for_power(poly: Polynomial, var_index: int, power: int) -> Polynomial:
    terms: dict[tuple[int, ...], Any] = {}
    for exp, coef in poly.terms.items():
        if int(exp[var_index]) != int(power):
            continue
        new_exp = list(exp)
        new_exp[var_index] = 0
        new_exp_t = tuple(new_exp)
        terms[new_exp_t] = terms.get(new_exp_t, torch.zeros_like(coef)) + coef
    return Polynomial(terms, poly.n_vars)


def _tm_apply_cutoff_for_horner(
    model: TaylorModel,
    cutoff_threshold: float | None,
    domain: Sequence[Interval],
    order: int,
) -> tuple[TaylorModel, Interval]:
    if cutoff_threshold is None:
        return model, _zero_interval_like_domain(domain)
    kept, cutoff_range = model.polynomial.cutoff(cutoff_threshold, domain)
    return (
        TaylorModel(
            kept,
            model.remainder + cutoff_range,
            list(domain),
            order=int(order),
            truncation_range_split=model.truncation_range_split,
        ),
        cutoff_range,
    )


def _tm_mul_ctrunc_horner_stage(
    left: TaylorModel,
    right: TaylorModel,
    order: int,
    cutoff_threshold: float | None,
    domain: Sequence[Interval],
) -> tuple[TaylorModel, dict[str, float]]:
    split = _merged_split_value(left.truncation_range_split, right.truncation_range_split)
    kept_poly, dropped_poly = left.polynomial.mul_truncate(right.polynomial, int(order))
    left_poly_range = left.polynomial.evaluate_interval(domain)
    right_poly_range = right.polynomial.evaluate_interval(domain)
    trunc_range = _poly_interval_with_split(dropped_poly, domain, split)
    cutoff_kept, cutoff_range = kept_poly.cutoff(cutoff_threshold, domain)
    p_left_times_right_remainder = left_poly_range * right.remainder
    p_right_times_left_remainder = right_poly_range * left.remainder
    remainder_times_remainder = left.remainder * right.remainder
    remainder = (
        p_left_times_right_remainder
        + p_right_times_left_remainder
        + remainder_times_remainder
        + trunc_range
        + cutoff_range
    )
    result = TaylorModel(
        cutoff_kept,
        remainder,
        list(domain),
        order=int(order),
        truncation_range_split=split,
    )
    kept_range = cutoff_kept.evaluate_interval(domain)
    total_range = kept_range + remainder
    return result, {
        "kept_poly_range_width": _interval_width_float(kept_range),
        "truncation_width": _interval_width_float(trunc_range),
        "cutoff_width": _interval_width_float(cutoff_range),
        "p_left_times_right_remainder_width": _interval_width_float(p_left_times_right_remainder),
        "p_right_times_left_remainder_width": _interval_width_float(p_right_times_left_remainder),
        "remainder_times_remainder_width": _interval_width_float(remainder_times_remainder),
        "result_remainder_width": _interval_width_float(result.remainder),
        "result_range_width": _interval_width_float(total_range),
    }


def _horner_component_name(component_index: int | None) -> str:
    if component_index == 0:
        return "x"
    if component_index == 1:
        return "y"
    return "scalar" if component_index is None else f"state_{component_index}"


def _append_horner_stage(
    stage_rows: list[dict[str, Any]],
    top_components: list[dict[str, Any]],
    *,
    component_index: int | None,
    variable_index: int | None,
    branch: str,
    operation: str,
    result: TaylorModel,
    inserted_var: TaylorModel | None,
    power_after: int | str,
    components: Mapping[str, float] | None = None,
) -> None:
    components = dict(components or {})
    row = {
        "component_index": "" if component_index is None else int(component_index),
        "component": _horner_component_name(component_index),
        "stage_index": len(stage_rows),
        "variable_index": "" if variable_index is None else int(variable_index),
        "branch": branch,
        "operation": operation,
        "power_after": power_after,
        "inserted_var_range_width": _interval_width_float(inserted_var.range_box()) if inserted_var is not None else "",
        "result_range_width": _interval_width_float(result.range_box()),
        "result_normal_range_width": _interval_width_float(_taylor_model_range_box_normal(result, None)),
        "result_remainder_width": _interval_width_float(result.remainder),
        "result_term_count": len(result.polynomial.terms),
        "result_degree": result.polynomial.degree(),
    }
    row.update(components)
    stage_rows.append(row)
    for name, width in components.items():
        if not name.endswith("_width"):
            continue
        top_components.append({
            "component_index": row["component_index"],
            "component": row["component"],
            "stage_index": row["stage_index"],
            "variable_index": row["variable_index"],
            "branch": branch,
            "operation": operation,
            "uncertainty_component": name,
            "width": float(width),
        })


def _insert_ctrunc_normal_horner_scalar(
    outer: TaylorModel,
    inner: TMVector,
    order: int,
    cutoff_threshold: float | None,
    domain: Sequence[Interval] | None,
    *,
    component_index: int | None,
    stage_rows: list[dict[str, Any]],
    top_components: list[dict[str, Any]],
    time_var_index: int | None = None,
) -> TaylorModel:
    if outer.polynomial.n_vars != len(inner):
        raise ValueError("outer TaylorModel variable count must match inner TMVector length")
    out_domain = list(domain or inner.domain)
    if len(out_domain) != inner.n_vars:
        raise ValueError("composition output domain must match inner TaylorModel domain")
    inner_work = _tmvector_with_order(inner, int(order))

    def horner(poly: Polynomial, var_index: int) -> TaylorModel:
        if not poly.terms:
            return TaylorModel.zero(out_domain, order=int(order), truncation_range_split=outer.truncation_range_split)
        if var_index >= poly.n_vars:
            coef = poly.terms.get((0,) * poly.n_vars)
            return TaylorModel.constant(
                0.0 if coef is None else coef,
                out_domain,
                order=int(order),
                truncation_range_split=outer.truncation_range_split,
            )
        powers = sorted({int(exp[var_index]) for exp in poly.terms})
        max_power = max(powers)
        parts = {power: _polynomial_part_for_power(poly, var_index, power) for power in powers}
        acc = horner(parts[max_power], var_index + 1)
        for power in range(max_power - 1, -1, -1):
            branch = "time" if time_var_index is not None and var_index == int(time_var_index) else "state"
            acc, components = _tm_mul_ctrunc_horner_stage(
                acc,
                inner_work[var_index],
                int(order),
                cutoff_threshold,
                out_domain,
            )
            _append_horner_stage(
                stage_rows,
                top_components,
                component_index=component_index,
                variable_index=var_index,
                branch=branch,
                operation="multiply_inserted_right_map",
                result=acc,
                inserted_var=inner_work[var_index],
                power_after=power,
                components=components,
            )
            part = parts.get(power)
            if part is not None and part.terms:
                addend = horner(part, var_index + 1)
                acc = acc + addend
                acc, cutoff_range = _tm_apply_cutoff_for_horner(acc, cutoff_threshold, out_domain, int(order))
                _append_horner_stage(
                    stage_rows,
                    top_components,
                    component_index=component_index,
                    variable_index=var_index,
                    branch=branch,
                    operation="add_coefficient_branch",
                    result=acc,
                    inserted_var=None,
                    power_after=power,
                    components={"cutoff_width": _interval_width_float(cutoff_range)},
                )
        return acc

    result = horner(outer.polynomial, 0)
    if not _interval_is_zero(outer.remainder):
        result = result.with_remainder(result.remainder + outer.remainder)
        _append_horner_stage(
            stage_rows,
            top_components,
            component_index=component_index,
            variable_index=None,
            branch="outer",
            operation="add_outer_remainder",
            result=result,
            inserted_var=None,
            power_after="",
            components={"outer_remainder_width": _interval_width_float(outer.remainder)},
        )
    return result


def _stage_width_sum(stage_rows: Sequence[Mapping[str, Any]], field: str, component_index: int | None = None) -> float:
    total = 0.0
    wanted = "" if component_index is None else int(component_index)
    for row in stage_rows:
        if component_index is not None and row.get("component_index") != wanted:
            continue
        total += _float_or_none(row.get(field)) or 0.0
    return total


def _copy_horner_summary_to_diagnostics(diagnostics: dict[str, Any] | None, summary: Mapping[str, Any]) -> None:
    if diagnostics is None:
        return
    for key, value in summary.items():
        if key.startswith("_"):
            continue
        diagnostics[key] = value


def insert_ctrunc_normal_horner_diagnostic(
    outer: TaylorModel | TMVector,
    inner: TMVector,
    order: int,
    cutoff_threshold: float | None,
    domain: Sequence[Interval] | None = None,
    diagnostics: dict[str, Any] | None = None,
    *,
    time_var_index: int | None = None,
) -> HornerInsertionDiagnosticResult:
    """Compare direct sparse substitution with Horner-style insertion.

    This is a clean-room diagnostic analogue: it deliberately does not replace
    the default direct insertion path. The Horner side truncates and cutoffs
    after each multiplication by an inserted Taylor model and records the range
    and uncertainty generated at each stage.
    """
    direct_diag: dict[str, Any] = {}
    direct_result = insert_ctrunc_normal_like(
        outer,
        inner,
        int(order),
        cutoff_threshold,
        domain,
        direct_diag,
    )
    stage_rows: list[dict[str, Any]] = []
    top_components: list[dict[str, Any]] = []
    if isinstance(outer, TMVector):
        horner_models = [
            _insert_ctrunc_normal_horner_scalar(
                model,
                inner,
                int(order),
                cutoff_threshold,
                domain,
                component_index=index,
                stage_rows=stage_rows,
                top_components=top_components,
                time_var_index=time_var_index,
            )
            for index, model in enumerate(outer)
        ]
        horner_result: TaylorModel | TMVector = TMVector(horner_models)
    else:
        horner_result = _insert_ctrunc_normal_horner_scalar(
            outer,
            inner,
            int(order),
            cutoff_threshold,
            domain,
            component_index=None,
            stage_rows=stage_rows,
            top_components=top_components,
            time_var_index=time_var_index,
        )

    direct_standard = _object_range_width_sum(direct_result)
    horner_standard = _object_range_width_sum(horner_result)
    direct_normal = _object_normal_range_width_sum(direct_result)
    horner_normal = _object_normal_range_width_sum(horner_result)
    range_delta = horner_standard - direct_standard
    normal_range_delta = horner_normal - direct_normal
    change_tol = 1e-12
    summary: dict[str, Any] = {
        "direct_range_width_sum": direct_standard,
        "horner_range_width_sum": horner_standard,
        "direct_normal_range_width_sum": direct_normal,
        "horner_normal_range_width_sum": horner_normal,
        "horner_minus_direct_range_width_sum": range_delta,
        "horner_minus_direct_normal_range_width_sum": normal_range_delta,
        "horner_reduced_range": range_delta < -change_tol,
        "horner_reduced_normal_range": normal_range_delta < -change_tol,
        "horner_changed_range": abs(range_delta) > change_tol,
        "horner_stage_count": len(stage_rows),
        "horner_time_branch_stage_count": sum(1 for row in stage_rows if row.get("branch") == "time"),
        "horner_state_branch_stage_count": sum(1 for row in stage_rows if row.get("branch") == "state"),
        "horner_y_branch_stage_count": sum(1 for row in stage_rows if row.get("variable_index") == 1),
        "horner_truncation_width_sum": _stage_width_sum(stage_rows, "truncation_width"),
        "horner_cutoff_width_sum": _stage_width_sum(stage_rows, "cutoff_width"),
        "horner_p_left_times_right_remainder_width_sum": _stage_width_sum(stage_rows, "p_left_times_right_remainder_width"),
        "horner_p_right_times_left_remainder_width_sum": _stage_width_sum(stage_rows, "p_right_times_left_remainder_width"),
        "horner_remainder_times_remainder_width_sum": _stage_width_sum(stage_rows, "remainder_times_remainder_width"),
        "horner_outer_remainder_width_sum": _stage_width_sum(stage_rows, "outer_remainder_width"),
        "direct_truncation_width_sum": direct_diag.get("insertion_truncation_width", 0.0),
        "direct_cutoff_width_sum": direct_diag.get("insertion_cutoff_width", 0.0),
        "direct_output_remainder_width_sum": direct_diag.get("output_remainder_width", 0.0),
        "direct_terms_before_insertion_truncation": direct_diag.get("terms_before_insertion_truncation", ""),
        "direct_terms_after_insertion": direct_diag.get("terms_after_insertion", ""),
    }
    for component_index, name in enumerate(("x", "y")):
        summary[f"horner_truncation_width_{name}"] = _stage_width_sum(stage_rows, "truncation_width", component_index)
        summary[f"horner_cutoff_width_{name}"] = _stage_width_sum(stage_rows, "cutoff_width", component_index)
        summary[f"horner_output_remainder_width_{name}"] = _stage_width_sum(stage_rows, "result_remainder_width", component_index)
    top_components_sorted = sorted(
        top_components,
        key=lambda row: _float_or_none(row.get("width")) or 0.0,
        reverse=True,
    )
    _copy_horner_summary_to_diagnostics(diagnostics, summary)
    return HornerInsertionDiagnosticResult(
        direct_result=direct_result,
        horner_result=horner_result,
        summary=summary,
        stage_ranges=stage_rows,
        top_components=top_components_sorted,
    )

def _tmvector_constant_part(tmv: TMVector) -> list[float]:
    constants: list[float] = []
    for model in tmv:
        zero_exp = (0,) * model.polynomial.n_vars
        coef = model.polynomial.terms.get(zero_exp)
        if coef is None:
            coef = _zero_shift_like(model)
        constants.append(float(coef.detach().cpu()))
    return constants


def _tmvector_rm_constants(tmv: TMVector) -> TMVector:
    models: list[TaylorModel] = []
    for model in tmv:
        zero_exp = (0,) * model.polynomial.n_vars
        terms = {exp: coef for exp, coef in model.polynomial.terms.items() if exp != zero_exp}
        models.append(
            TaylorModel(
                Polynomial(terms, model.polynomial.n_vars),
                model.remainder,
                list(model.domain),
                order=model.order,
                truncation_range_split=model.truncation_range_split,
            )
        )
    return TMVector(models)


def _interval_magnitude(iv: Interval) -> float | None:
    lo = _interval_bound_value(iv.lo)
    hi = _interval_bound_value(iv.hi)
    if lo is None or hi is None:
        return None
    return max(abs(lo), abs(hi))


def _scale_tmvector_components(tmv: TMVector, inv_scales: Sequence[float]) -> TMVector:
    models: list[TaylorModel] = []
    for model, inv_scale in zip(tmv, inv_scales):
        models.append(model * float(inv_scale))
    return TMVector(models)


def _tmvector_add_remainders(tmv: TMVector, remainders: Sequence[Interval]) -> TMVector:
    return TMVector(model.with_remainder(model.remainder + rem) for model, rem in zip(tmv, remainders))


def _tmvector_recenter_remainders(tmv: TMVector) -> tuple[TMVector, list[float]]:
    models: list[TaylorModel] = []
    shifts: list[float] = []
    for model in tmv:
        midpoint = model.remainder.mid()
        shift = _float_or_none(midpoint.detach().cpu()) or 0.0
        shifts.append(float(shift))
        if shift == 0.0:
            models.append(model)
        else:
            models.append(model.with_remainder(model.remainder - Interval.point(midpoint)))
    return TMVector(models), shifts


def _tm_shift_polynomial_constant(model: TaylorModel, shift: float) -> TaylorModel:
    if float(shift) == 0.0:
        return model
    value = torch.as_tensor(float(shift), dtype=model.remainder.lo.dtype, device=model.remainder.lo.device)
    return TaylorModel(
        model.polynomial - Polynomial.constant(value, model.polynomial.n_vars),
        model.remainder,
        list(model.domain),
        order=model.order,
        truncation_range_split=model.truncation_range_split,
    )


def _tmvector_shift_polynomial_constants(tmv: TMVector, shifts: Sequence[float]) -> TMVector:
    return TMVector(_tm_shift_polynomial_constant(model, float(shift)) for model, shift in zip(tmv, shifts))


def _right_map_range_box_for_mode(tmv: TMVector, mode: str) -> list[Interval]:
    if mode == "normal_eval":
        return _tmvector_range_box_normal(tmv, None)
    return tmv.range_box()


def _polynomial_max_abs_diff(a: Polynomial, b: Polynomial) -> float:
    if a.terms:
        zero = torch.zeros((), dtype=a.dtype, device=a.device)
    elif b.terms:
        zero = torch.zeros((), dtype=b.dtype, device=b.device)
    else:
        return 0.0
    max_diff = 0.0
    for exp in set(a.terms) | set(b.terms):
        av = a.terms.get(exp, zero)
        bv = b.terms.get(exp, zero)
        diff = _float_or_none(torch.max(torch.abs(av - bv)).detach().cpu())
        if diff is not None:
            max_diff = max(max_diff, float(diff))
    return max_diff


def _add_right_map_centering_diagnostics(
    diagnostics: dict[str, Any],
    *,
    mode: str,
    old_center: Sequence[float],
    new_center: Sequence[float],
    inserted: TMVector,
    centered_inserted: TMVector,
    inserted_box: Sequence[Interval],
    centered_box: Sequence[Interval],
    baseline_scales: Sequence[float],
    centered_scales: Sequence[float],
    hypothetical_centered_box: Sequence[Interval],
    hypothetical_centered_scales: Sequence[float],
    applied_shifts: Sequence[float],
) -> None:
    diagnostics["right_map_center_mode"] = mode
    diagnostics["immediate_saving_source"] = "same_inserted_tm_shadow_diagnostic"
    _add_interval_bounds(diagnostics, "inserted_range", inserted_box)
    _add_width_metrics(diagnostics, "inserted_range", inserted_box)
    _add_interval_bounds(diagnostics, "centered_inserted_range", centered_box)
    _add_width_metrics(diagnostics, "centered_inserted_range", centered_box)
    _add_interval_bounds(diagnostics, "hypothetical_centered_inserted_range", hypothetical_centered_box)
    _add_width_metrics(diagnostics, "hypothetical_centered_inserted_range", hypothetical_centered_box)

    names = ("x", "y")
    shift_abs_sum = 0.0
    asymmetry_sum = 0.0
    baseline_scale_sum = 0.0
    centered_scale_sum = 0.0
    baseline_reset_sum = 0.0
    centered_reset_sum = 0.0
    hypothetical_centered_reset_sum = 0.0
    reduction_sum = 0.0
    immediate_reduction_sum = 0.0
    max_poly_diff = 0.0
    max_rem_lo_diff = 0.0
    max_rem_hi_diff = 0.0
    for i, model in enumerate(inserted):
        name = names[i] if i < len(names) else f"state_{i}"
        shift = float(applied_shifts[i]) if i < len(applied_shifts) else 0.0
        baseline_scale = float(baseline_scales[i]) if i < len(baseline_scales) else 0.0
        centered_scale = float(centered_scales[i]) if i < len(centered_scales) else baseline_scale
        hypothetical_scale = (
            float(hypothetical_centered_scales[i])
            if i < len(hypothetical_centered_scales)
            else centered_scale
        )
        baseline_reset_width = 2.0 * abs(baseline_scale)
        centered_reset_width = 2.0 * abs(centered_scale)
        hypothetical_centered_reset_width = 2.0 * abs(hypothetical_scale)
        reduction = baseline_reset_width - centered_reset_width
        reduction_relative = reduction / baseline_reset_width if baseline_reset_width > 0.0 else 0.0
        immediate_reduction = baseline_reset_width - hypothetical_centered_reset_width
        immediate_reduction_relative = immediate_reduction / baseline_reset_width if baseline_reset_width > 0.0 else 0.0

        asymmetry = 0.0
        if i < len(inserted_box):
            lo = _interval_bound_value(inserted_box[i].lo)
            hi = _interval_bound_value(inserted_box[i].hi)
            if lo is not None and hi is not None:
                asymmetry = abs(abs(float(hi)) - abs(float(lo)))

        center_value = float(old_center[i]) if i < len(old_center) else 0.0
        shifted_center_value = float(new_center[i]) if i < len(new_center) else center_value
        lhs_poly = model.polynomial + Polynomial.constant(center_value, model.polynomial.n_vars)
        centered_model = centered_inserted[i]
        rhs_poly = centered_model.polynomial + Polynomial.constant(shifted_center_value, centered_model.polynomial.n_vars)
        poly_diff = _polynomial_max_abs_diff(lhs_poly, rhs_poly)
        rem_lo_diff = abs(float((model.remainder.lo - centered_model.remainder.lo).detach().cpu()))
        rem_hi_diff = abs(float((model.remainder.hi - centered_model.remainder.hi).detach().cpu()))

        diagnostics[f"inserted_range_midpoint_shift_{name}"] = shift
        diagnostics[f"inserted_range_asymmetry_{name}"] = asymmetry
        diagnostics[f"baseline_scale_{name}"] = baseline_scale
        diagnostics[f"constant_scale_{name}"] = baseline_scale
        diagnostics[f"centered_scale_{name}"] = centered_scale
        diagnostics[f"actual_centered_scale_{name}"] = centered_scale
        diagnostics[f"hypothetical_centered_scale_{name}"] = hypothetical_scale
        diagnostics[f"baseline_reset_width_{name}"] = baseline_reset_width
        diagnostics[f"centered_reset_width_{name}"] = centered_reset_width
        diagnostics[f"hypothetical_centered_reset_width_{name}"] = hypothetical_centered_reset_width
        diagnostics[f"scale_reduction_absolute_{name}"] = reduction
        diagnostics[f"scale_reduction_relative_{name}"] = reduction_relative
        diagnostics[f"immediate_reset_reduction_absolute_{name}"] = immediate_reduction
        diagnostics[f"immediate_reset_reduction_relative_{name}"] = immediate_reduction_relative
        diagnostics[f"reconstruction_polynomial_max_abs_diff_{name}"] = poly_diff
        diagnostics[f"reconstruction_remainder_lo_diff_{name}"] = rem_lo_diff
        diagnostics[f"reconstruction_remainder_hi_diff_{name}"] = rem_hi_diff

        shift_abs_sum += abs(shift)
        asymmetry_sum += asymmetry
        baseline_scale_sum += abs(baseline_scale)
        centered_scale_sum += abs(centered_scale)
        baseline_reset_sum += baseline_reset_width
        centered_reset_sum += centered_reset_width
        hypothetical_centered_reset_sum += hypothetical_centered_reset_width
        reduction_sum += reduction
        immediate_reduction_sum += immediate_reduction
        max_poly_diff = max(max_poly_diff, poly_diff)
        max_rem_lo_diff = max(max_rem_lo_diff, rem_lo_diff)
        max_rem_hi_diff = max(max_rem_hi_diff, rem_hi_diff)

    diagnostics["inserted_range_midpoint_shift_abs_sum"] = shift_abs_sum
    diagnostics["inserted_range_asymmetry_sum"] = asymmetry_sum
    diagnostics["baseline_scale_sum"] = baseline_scale_sum
    diagnostics["constant_scale_sum"] = baseline_scale_sum
    diagnostics["centered_scale_sum"] = centered_scale_sum
    diagnostics["actual_centered_scale_sum"] = centered_scale_sum
    diagnostics["hypothetical_centered_scale_sum"] = sum(abs(float(s)) for s in hypothetical_centered_scales)
    diagnostics["baseline_reset_width_sum"] = baseline_reset_sum
    diagnostics["centered_reset_width_sum"] = centered_reset_sum
    diagnostics["hypothetical_centered_reset_width_sum"] = hypothetical_centered_reset_sum
    diagnostics["scale_reduction_absolute_sum"] = reduction_sum
    diagnostics["scale_reduction_relative_sum"] = reduction_sum / baseline_reset_sum if baseline_reset_sum > 0.0 else 0.0
    diagnostics["immediate_reset_reduction_absolute_sum"] = immediate_reduction_sum
    diagnostics["immediate_reset_reduction_relative_sum"] = (
        immediate_reduction_sum / baseline_reset_sum if baseline_reset_sum > 0.0 else 0.0
    )
    diagnostics["reconstruction_polynomial_max_abs_diff"] = max_poly_diff
    diagnostics["reconstruction_remainder_lo_diff"] = max_rem_lo_diff
    diagnostics["reconstruction_remainder_hi_diff"] = max_rem_hi_diff


def _flowstar_normalized_insertion_transition(
    seg: FlowpipeSegment,
    previous_state: FlowstarNormalFlowpipeState | None,
    order: int,
    *,
    cutoff_threshold: float | None,
    symbolic_queue: bool = False,
    symbolic_queue_split: bool = False,
    symbolic_queue_state: FlowstarSymbolicRemainderQueue | None = None,
    symbolic_queue_max_size: int = 100,
    symbolic_queue_mode: str = "",
    target_remainder_radius: float | None = None,
    scalar_recenter_remainder_midpoint: bool = False,
    right_map_range_mode: str = "standard",
    right_map_center_mode: str = "constant",
    horner_diagnostic: bool = False,
    horner_insertion: bool = False,
    dependency_preserving_insertion: bool = False,
    complete_polynomial_carry: bool = False,
) -> tuple[TMVector, FlowstarNormalFlowpipeState, dict[str, Any]]:
    prev = previous_state
    if prev is None:
        prev = FlowstarNormalFlowpipeState(
            tmv_pre=seg.final_tm,
            tmv_right=TMVector.identity(seg.final_tm.domain, order=order),
            domain=seg.final_tm.domain,
            center=_tmvector_constant_part(seg.final_tm),
            scales=[1.0 for _ in seg.final_tm],
            step_index=0,
            diagnostics={"reset_mode": "normalized_insertion", "implicit_initial_state": True},
        )
    if right_map_range_mode not in {"standard", "normal_eval"}:
        raise ValueError("right_map_range_mode must be 'standard' or 'normal_eval'")
    if right_map_center_mode not in {"constant", "range_midpoint"}:
        raise ValueError("right_map_center_mode must be 'constant' or 'range_midpoint'")
    if symbolic_queue_mode not in {"", "flowstar_linear_v2"}:
        raise ValueError("symbolic_queue_mode must be empty or 'flowstar_linear_v2'")
    symbolic_queue_v2 = symbolic_queue_mode == "flowstar_linear_v2"
    if horner_insertion and dependency_preserving_insertion:
        raise ValueError("select exactly one normal-insertion algorithm")
    if dependency_preserving_insertion:
        mode_name = NORMALIZED_INSERTION_DEPENDENCY_PRESERVING
    elif horner_insertion and symbolic_queue_v2:
        mode_name = "normalized_insertion_horner_symqueue_v2"
    elif complete_polynomial_carry:
        mode_name = "normalized_insertion_complete_polynomial"
    elif horner_insertion:
        mode_name = "normalized_insertion_horner"
    elif symbolic_queue_v2:
        mode_name = "normalized_insertion_symqueue_v2"
    elif symbolic_queue_split:
        mode_name = "normalized_insertion_symqueue_split"
    elif symbolic_queue:
        mode_name = "normalized_insertion_symqueue"
    else:
        mode_name = "normalized_insertion"
    endpoint_box = seg.final_tm.range_box()
    diagnostics: dict[str, Any] = {
        "reset_mode": mode_name,
        "right_map_range_mode": right_map_range_mode,
        "right_map_center_mode": right_map_center_mode,
        "step_index": int(prev.step_index) + 1,
        "endpoint_box_width_sum": _sum_interval_widths(endpoint_box),
        "endpoint_tm_width_sum": _sum_interval_widths(endpoint_box),
        "reset_box_width_sum": _sum_interval_widths(endpoint_box),
    }
    _add_width_metrics(diagnostics, "endpoint_box", endpoint_box)
    _add_width_metrics(diagnostics, "endpoint_tm", endpoint_box)
    center = _tmvector_constant_part(seg.final_tm)
    endpoint_without_constants = _tmvector_rm_constants(seg.final_tm)
    if dependency_preserving_insertion:
        inserted = insert_ctrunc_normal_dependency_preserving(
            endpoint_without_constants,
            prev.tmv_right,
            int(order),
            cutoff_threshold,
            prev.domain,
            diagnostics,
        )
        assert isinstance(inserted, TMVector)
    elif horner_insertion:
        horner_result = insert_ctrunc_normal_horner_diagnostic(
            endpoint_without_constants,
            prev.tmv_right,
            int(order),
            cutoff_threshold,
            prev.domain,
            diagnostics,
        )
        inserted = horner_result.horner_result
        assert isinstance(inserted, TMVector)
        diagnostics["insertion_horner_used"] = True
        diagnostics["_horner_stage_ranges"] = [dict(row) for row in horner_result.stage_ranges]
        diagnostics["_horner_top_components"] = [dict(row) for row in horner_result.top_components]
        diagnostics["insertion_truncation_width"] = horner_result.summary.get("horner_truncation_width_sum", 0.0)
        diagnostics["insertion_truncation_width_sum"] = diagnostics["insertion_truncation_width"]
        diagnostics["insertion_truncation_width_x"] = horner_result.summary.get("horner_truncation_width_x", 0.0)
        diagnostics["insertion_truncation_width_y"] = horner_result.summary.get("horner_truncation_width_y", 0.0)
        diagnostics["insertion_cutoff_width"] = horner_result.summary.get("horner_cutoff_width_sum", 0.0)
        diagnostics["insertion_cutoff_width_sum"] = diagnostics["insertion_cutoff_width"]
        diagnostics["insertion_cutoff_width_x"] = horner_result.summary.get("horner_cutoff_width_x", 0.0)
        diagnostics["insertion_cutoff_width_y"] = horner_result.summary.get("horner_cutoff_width_y", 0.0)
        diagnostics["terms_after_insertion"] = sum(len(model.polynomial.terms) for model in inserted)
        diagnostics["terms_before_insertion_truncation"] = horner_result.summary.get("direct_terms_before_insertion_truncation", "")
        for index, model in enumerate(inserted):
            component = "x" if index == 0 else ("y" if index == 1 else f"state_{index}")
            poly_range = model.polynomial.evaluate_interval(prev.domain)
            _diag_add_width(diagnostics, "composed_poly_range_width", poly_range)
            diagnostics[f"composed_poly_range_width_{component}"] = _interval_width_float(poly_range)
            _diag_add_width(diagnostics, "output_remainder_width", model.remainder)
            diagnostics[f"output_remainder_width_{component}"] = _interval_width_float(model.remainder)
    else:
        inserted = insert_ctrunc_normal_like(
            endpoint_without_constants,
            prev.tmv_right,
            int(order),
            cutoff_threshold,
            prev.domain,
            diagnostics,
        )
        assert isinstance(inserted, TMVector)
        if horner_diagnostic:
            horner_result = insert_ctrunc_normal_horner_diagnostic(
                endpoint_without_constants,
                prev.tmv_right,
                int(order),
                cutoff_threshold,
                prev.domain,
                diagnostics,
            )
            diagnostics["_horner_stage_ranges"] = [dict(row) for row in horner_result.stage_ranges]
            diagnostics["_horner_top_components"] = [dict(row) for row in horner_result.top_components]
    remainder_midpoint_shifts: list[float] = [0.0 for _ in inserted]
    if scalar_recenter_remainder_midpoint:
        inserted, remainder_midpoint_shifts = _tmvector_recenter_remainders(inserted)
        center = [float(c) + float(shift) for c, shift in zip(center, remainder_midpoint_shifts)]
        diagnostics["scalar_recenter_remainder_midpoint"] = True
        diagnostics["remainder_midpoint_shift_x"] = remainder_midpoint_shifts[0] if len(remainder_midpoint_shifts) > 0 else ""
        diagnostics["remainder_midpoint_shift_y"] = remainder_midpoint_shifts[1] if len(remainder_midpoint_shifts) > 1 else ""
        diagnostics["remainder_midpoint_shift_abs_sum"] = sum(abs(float(v)) for v in remainder_midpoint_shifts)
    old_inserted_box = inserted.range_box()
    normal_inserted_box = _tmvector_range_box_normal(inserted, None)
    _add_width_metrics(diagnostics, "old_right_map_range", old_inserted_box)
    _add_width_metrics(diagnostics, "normal_right_map_range", normal_inserted_box)
    inserted_box = normal_inserted_box if right_map_range_mode == "normal_eval" else old_inserted_box
    baseline_scales: list[float] = []
    for iv in inserted_box:
        mag = _interval_magnitude(iv)
        scale = 0.0 if mag is None or mag == 0.0 else float(mag)
        baseline_scales.append(scale)
    hypothetical_midpoint_shifts = [
        _float_or_none(iv.mid().detach().cpu()) or 0.0
        for iv in inserted_box
    ]
    hypothetical_centered_inserted = _tmvector_shift_polynomial_constants(inserted, hypothetical_midpoint_shifts)
    hypothetical_centered_box = _right_map_range_box_for_mode(hypothetical_centered_inserted, right_map_range_mode)
    hypothetical_centered_scales: list[float] = []
    for iv in hypothetical_centered_box:
        mag = _interval_magnitude(iv)
        scale = 0.0 if mag is None or mag == 0.0 else float(mag)
        hypothetical_centered_scales.append(scale)

    old_center = list(center)
    if right_map_center_mode == "range_midpoint":
        centered_inserted = hypothetical_centered_inserted
        center = [float(c) + float(shift) for c, shift in zip(center, hypothetical_midpoint_shifts)]
        centered_box = hypothetical_centered_box
        scale_box = centered_box
        inserted_for_reset = centered_inserted
        applied_shifts = hypothetical_midpoint_shifts
    else:
        centered_inserted = inserted
        centered_box = inserted_box
        scale_box = inserted_box
        inserted_for_reset = inserted
        applied_shifts = [0.0 for _ in inserted]

    scales: list[float] = []
    inv_scales: list[float] = []
    for iv in scale_box:
        mag = _interval_magnitude(iv)
        scale = 0.0 if mag is None or mag == 0.0 else float(mag)
        scales.append(scale)
        inv_scales.append(1.0 if scale == 0.0 else 1.0 / scale)
    _add_right_map_centering_diagnostics(
        diagnostics,
        mode=right_map_center_mode,
        old_center=old_center,
        new_center=center,
        inserted=inserted,
        centered_inserted=centered_inserted,
        inserted_box=inserted_box,
        centered_box=centered_box,
        baseline_scales=baseline_scales,
        centered_scales=scales,
        hypothetical_centered_box=hypothetical_centered_box,
        hypothetical_centered_scales=hypothetical_centered_scales,
        applied_shifts=applied_shifts,
    )
    tmv_right = _scale_tmvector_components(inserted_for_reset, inv_scales).apply_cutoff(cutoff_threshold)
    _add_width_metrics(diagnostics, "inserted_endpoint", scale_box)
    _add_width_metrics(diagnostics, "normal_state_right", tmv_right.range_box())
    reset_tm = _normalized_tm_from_center_scale(center, scales, int(order), template_domain=prev.domain)
    reset_box = reset_tm.range_box()
    _add_width_metrics(diagnostics, "normalized_reset", reset_box)
    next_queue = prev.symbolic_queue if prev.symbolic_queue is not None else symbolic_queue_state
    initial_remainders: tuple[Interval, ...] | None = None
    if symbolic_queue_v2:
        reset_tm, next_queue, queue_stats = flowstar_normalized_insertion_linear_queue_v2_reset(
            inserted_for_reset,
            reset_tm,
            next_queue,
            scales=scales,
            max_size=symbolic_queue_max_size,
            target_remainder_radius=target_remainder_radius,
        )
        diagnostics.update(queue_stats)
    elif symbolic_queue:
        reset_tm, next_queue, queue_stats = flowstar_normalized_insertion_symbolic_queue_reset(
            inserted_for_reset,
            reset_tm,
            next_queue,
            scales=scales,
            max_size=symbolic_queue_max_size,
            materialize_propagated_on_reset=not symbolic_queue_split,
        )
        if not symbolic_queue_split:
            initial_remainders = tuple(model.remainder for model in reset_tm)
        diagnostics.update(queue_stats)
    complete_initial_tm = preserve_complete_polynomial_carry(seg.final_tm) if complete_polynomial_carry else None
    if complete_initial_tm is not None:
        reset_tm = TMVector(model.clone() for model in complete_initial_tm)
        reset_box = reset_tm.range_box()
        _add_width_metrics(diagnostics, "normalized_reset", reset_box)
        diagnostics["complete_polynomial_carry"] = True
        diagnostics["complete_carry_retained_terms"] = sum(
            len(model.polynomial.terms) for model in complete_initial_tm
        )
        diagnostics["complete_carry_max_degree"] = _tm_max_degree(complete_initial_tm)
        diagnostics["complete_carry_intervalized_term_count"] = 0
        diagnostics["complete_carry_remainder_width_sum"] = sum(
            _interval_width_float(model.remainder) for model in complete_initial_tm
        )
        diagnostics["complete_carry_coefficient_sha256"] = hashlib.sha256(
            json.dumps(
                [
                    [
                        [list(exponent), float(coefficient.detach().cpu())]
                        for exponent, coefficient in sorted(model.polynomial.terms.items())
                    ]
                    for model in complete_initial_tm
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    state = FlowstarNormalFlowpipeState(
        tmv_pre=seg.tm,
        tmv_right=tmv_right,
        domain=list(prev.domain),
        center=center,
        scales=scales,
        step_index=int(prev.step_index) + 1,
        diagnostics=diagnostics,
        symbolic_queue=next_queue if (symbolic_queue or symbolic_queue_v2) else None,
        symbolic_queue_max_size=int(symbolic_queue_max_size),
        initial_remainders=initial_remainders,
        complete_initial_tm=complete_initial_tm,
    )
    diagnostics.update(state.diagnostic_widths())
    diagnostics["inserted_endpoint_width_sum"] = _sum_interval_widths(inserted_box)
    diagnostics["normalized_reset_width_sum"] = _sum_interval_widths(reset_box)
    if symbolic_queue_split:
        insertion_trunc = _float_or_none(diagnostics.get("insertion_truncation_width")) or 0.0
        insertion_cutoff = _float_or_none(diagnostics.get("insertion_cutoff_width")) or 0.0
        diagnostics["insertion_truncation_ordinary_width"] = 0.0
        diagnostics["insertion_cutoff_ordinary_width"] = 0.0
        diagnostics["insertion_symbolic_candidate_width"] = insertion_trunc + insertion_cutoff
    diagnostics["scale_x"] = scales[0] if len(scales) > 0 else ""
    diagnostics["scale_y"] = scales[1] if len(scales) > 1 else ""
    diagnostics["center_x"] = center[0] if len(center) > 0 else ""
    diagnostics["center_y"] = center[1] if len(center) > 1 else ""
    diagnostics["tmv_right_degree"] = _tm_max_degree(tmv_right)
    diagnostics["tmv_pre_degree"] = _tm_max_degree(seg.tm)
    diagnostics["tmv_right_term_count"] = sum(len(model.polynomial.terms) for model in tmv_right)
    diagnostics["tmv_pre_term_count"] = sum(len(model.polynomial.terms) for model in seg.tm)
    return reset_tm, state, diagnostics


def _tmvector_remainder_tensor(tmv: TMVector) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([model.remainder.lo for model in tmv]).to(torch.float64)[None, :],
        torch.stack([model.remainder.hi for model in tmv]).to(torch.float64)[None, :],
    )


def _tmvector_with_remainder_tensor(
    tmv: TMVector,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> TMVector:
    if lo.shape != (1, len(tmv)) or hi.shape != lo.shape:
        raise ValueError("flowpipe S1 bridge currently requires a batch-one remainder tensor")
    return TMVector(
        TaylorModel(
            model.polynomial,
            Interval(lo[0, index], hi[0, index]),
            model.domain,
            order=model.order,
            truncation_range_split=model.truncation_range_split,
        )
        for index, model in enumerate(tmv)
    )


def _tmvector_without_remainder(tmv: TMVector) -> TMVector:
    zero = torch.zeros((1, len(tmv)), dtype=torch.float64)
    return _tmvector_with_remainder_tensor(tmv, zero, zero)


def _box_tensor(box: Sequence[Interval]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([interval.lo for interval in box]).to(torch.float64)[None, :],
        torch.stack([interval.hi for interval in box]).to(torch.float64)[None, :],
    )


def _interval_matrix_vector(
    matrix: OutwardIntervalTensor,
    vector: OutwardIntervalTensor,
) -> OutwardIntervalTensor:
    product = outward_matmul(
        matrix,
        OutwardIntervalTensor(vector.lo[..., None], vector.hi[..., None]),
    )
    return OutwardIntervalTensor(product.lo[..., 0], product.hi[..., 0])


def _s1_padding_to_target(
    known: OutwardIntervalTensor,
    target: OutwardIntervalTensor,
) -> OutwardIntervalTensor:
    """Return the canonical exact-containment padding used by S1."""
    zero = torch.zeros_like(target.lo)
    safeguard = (
        64.0
        * torch.finfo(torch.float64).eps
        * torch.maximum(
            torch.maximum(torch.abs(known.lo), torch.abs(known.hi)),
            torch.maximum(torch.abs(target.lo), torch.abs(target.hi)),
        ).clamp_min(torch.finfo(torch.float64).tiny)
    )
    return OutwardIntervalTensor(
        torch.nextafter(
            torch.minimum(target.lo - known.lo, zero) - safeguard,
            torch.full_like(zero, -torch.inf),
        ),
        torch.nextafter(
            torch.maximum(target.hi - known.hi, zero) + safeguard,
            torch.full_like(zero, torch.inf),
        ),
    ).sanitized()


def _s1_interval_diagnostic(
    interval: OutwardIntervalTensor,
    *,
    units: str,
) -> dict[str, Any]:
    """Serialize a diagnostic interval with units and binary64 identity."""
    return {
        "units": units,
        "shape": list(interval.lo.shape),
        "lo": interval.lo.detach().cpu().tolist(),
        "hi": interval.hi.detach().cpu().tolist(),
        "lo_hex": tensor_hex(interval.lo),
        "hi_hex": tensor_hex(interval.hi),
        "lo_sha256": tensor_sha256(interval.lo),
        "hi_sha256": tensor_sha256(interval.hi),
        "width": (interval.hi - interval.lo).detach().cpu().tolist(),
    }


def verify_structured_publication(
    ordinary_lo: torch.Tensor,
    ordinary_hi: torch.Tensor,
    structured_lo: torch.Tensor,
    structured_hi: torch.Tensor,
    published_lo: torch.Tensor,
    published_hi: torch.Tensor,
) -> torch.Tensor:
    """Certify that a published interval includes ordinary plus all S1 columns."""
    total = OutwardIntervalTensor(ordinary_lo, ordinary_hi).add(
        OutwardIntervalTensor(structured_lo, structured_hi)
    )
    return ((published_lo <= total.lo) & (published_hi >= total.hi)).all(dim=1)


def _flowstar_bounded_source_ledger_transition(
    seg: FlowpipeSegment,
    previous_state: FlowstarNormalFlowpipeState,
    order: int,
    *,
    cutoff_threshold: float | None,
    right_map_range_mode: str,
    right_map_center_mode: str,
) -> tuple[TMVector, FlowstarNormalFlowpipeState, dict[str, Any]]:
    """Commit one accepted complete-O4 G1 source-ledger boundary.

    This bridge deliberately does not call either structured K16 transition.
    The fresh affine source is part of the next physical reset polynomial, so
    the dense Picard consumer sees it without a metadata adapter.
    """

    from .batched_dense_tm import REMAINDER_LEDGER_CATEGORIES

    source_before = previous_state.bounded_source_ledger_state
    if not isinstance(source_before, BoundedSourceLedgerState):
        raise ValueError("bounded source-ledger transition requires an initialized accepted state")
    if seg.status != "validated" or seg.validated_remainder_decomposition is None:
        raise ValueError("bounded source-ledger transition requires an accepted dense ledger")
    decomposition = seg.validated_remainder_decomposition
    if not bool(torch.all(decomposition.contains_image)):
        raise FloatingPointError("validated dense ledger does not contain the accepted Picard image")
    if decomposition.ledger.category_order != REMAINDER_LEDGER_CATEGORIES:
        raise ValueError("bounded source-ledger transition requires the complete ledger schema")
    dim = source_before.state_dim
    if len(seg.final_tm) != dim or len(previous_state.center) != dim:
        raise ValueError("source-ledger state dimension disagrees with the accepted endpoint")
    if len(previous_state.domain) != 2 * dim or seg.final_tm.n_vars != 2 * dim:
        raise ValueError("source-ledger production bridge requires a fixed 2d boundary domain")
    if right_map_range_mode != "standard" or right_map_center_mode != "constant":
        raise ValueError("G1 preregistration freezes standard range and constant center modes")

    diagnostics: dict[str, Any] = {
        "reset_mode": BOUNDED_SOURCE_LEDGER_CANDIDATE,
        "step_index": int(previous_state.step_index) + 1,
        "source_ledger_schema": source_before.schema,
        "source_ledger_schema_version": source_before.schema_version,
        "source_generations_retained": source_before.generations_retained,
        "source_ledger_pre_fingerprint": source_before.fingerprint,
        "source_ledger_pre_live_source_count": source_before.live_source_count,
        "source_ledger_boundary_atomicity": "accepted_only_immutable_commit",
        "source_ledger_first_consumer_field": "affine_source_coefficient_in_next_dense_picard_input",
    }

    # The outer accepted endpoint has d base and d live-source variables.  Base
    # coordinates are composed through the historical right map; source slots
    # are substituted by identity, preserving the exact same identity in all
    # nonlinear paths of the just-completed consumer step.
    inner_models = list(previous_state.tmv_right)
    inner_models.extend(
        TaylorModel.variable(dim + index, previous_state.domain, order=order)
        for index in range(dim)
    )
    augmented_inner = TMVector(inner_models)
    endpoint_without_constants = _tmvector_without_remainder(
        _tmvector_rm_constants(seg.final_tm)
    )
    inserted = insert_ctrunc_normal_like(
        endpoint_without_constants,
        augmented_inner,
        int(order),
        cutoff_threshold,
        previous_state.domain,
        diagnostics,
    )
    assert isinstance(inserted, TMVector)

    # Retire the preceding source generation after its one actual Picard
    # consumer.  Duplicate full exponents have already merged in Polynomial.
    collapsed_models: list[TaylorModel] = []
    collapse_rows: list[dict[str, Any]] = []
    retired_owner_rows: list[Mapping[str, Any]] = []
    carried_ordinary_rows: list[dict[str, Any]] = []
    for component, model in enumerate(inserted):
        ordinary_lo = float(model.remainder.lo.detach().cpu())
        ordinary_hi = float(model.remainder.hi.detach().cpu())
        carried_ordinary_rows.append(
            {
                "category": "ordinary_parameterization_composition_remainder",
                "component": component,
                "outward_lo_hex": ordinary_lo.hex(),
                "outward_hi_hex": ordinary_hi.hex(),
                "width": ordinary_hi - ordinary_lo,
                "canonical_support_sha256": hashlib.sha256(
                    f"ordinary_parameterization:{component}:{ordinary_lo.hex()}:{ordinary_hi.hex()}".encode("utf-8")
                ).hexdigest(),
                "containment_witness": "actual_inserted_TaylorModel_ordinary_remainder",
                "owner_resolution": "cumulative_preexisting_ordinary_dependency_not_recoverably_additive",
            }
        )
        retired_owner_rows.extend(
            g2_owner_rows(
                model.polynomial,
                previous_state.domain,
                component=component,
                oldest_indices=source_before.source_indices,
                current_indices=(),
                oldest_source_ids=source_before.source_ids,
            )
        )
        collapse = collapse_source_polynomial(
            model.polynomial,
            previous_state.domain,
            source_before.source_indices,
        )
        ordinary = model.remainder + collapse.collapsed
        collapsed_models.append(
            TaylorModel(
                collapse.retained,
                ordinary,
                previous_state.domain,
                order=int(order),
                truncation_range_split=model.truncation_range_split,
            )
        )
        collapse_rows.append(
            {
                "component": component,
                "source_term_count": collapse.source_term_count,
                "retained_term_count": collapse.retained_term_count,
                "source_support_sha256": collapse.source_support_sha256,
                "retained_support_sha256": collapse.retained_support_sha256,
                "collapsed_lo_hex": float(collapse.collapsed.lo.detach().cpu()).hex(),
                "collapsed_hi_hex": float(collapse.collapsed.hi.detach().cpu()).hex(),
                "collapsed_width": float(collapse.collapsed.width().detach().cpu()),
            }
        )
    collapsed = TMVector(collapsed_models)

    # The unchanged accepted remainder image is replaced by the complete
    # outward decomposition that contains it.  Its full aggregate is lifted to
    # exactly one independent source per state component.
    lift = source_ledger_affine_lift_interval(
        decomposition.decomposition_lo,
        decomposition.decomposition_hi,
    )
    if lift.midpoint.shape != (1, dim):
        raise ValueError("bounded source-ledger production bridge currently requires B1")
    old_center = _tmvector_constant_part(seg.final_tm)
    center = [
        float(value) + float(lift.midpoint[0, index].detach().cpu())
        for index, value in enumerate(old_center)
    ]

    scale_box = collapsed.range_box()
    scales: list[float] = []
    inv_scales: list[float] = []
    rebox_rows: list[dict[str, Any]] = []
    for component, interval in enumerate(scale_box):
        magnitude = _interval_magnitude(interval)
        scale = 0.0 if magnitude is None or magnitude == 0.0 else float(magnitude)
        # Normal range and later outward sums must remain inside [-1,1]
        # without a tolerance.  Enlarge the physical scale deterministically.
        if scale > 0.0:
            for _ in range(8):
                scale = math.nextafter(scale, math.inf)
        scales.append(scale)
        inv_scales.append(1.0 if scale == 0.0 else 1.0 / scale)
        rebox_rows.append(
            {
                "component": component,
                "input_lo_hex": float(interval.lo.detach().cpu()).hex(),
                "input_hi_hex": float(interval.hi.detach().cpu()).hex(),
                "input_width": float(interval.width().detach().cpu()),
                "outward_lo_hex": (-scale).hex(),
                "outward_hi_hex": scale.hex(),
                "width": 2.0 * scale,
                "symmetric_output_width": 2.0 * scale,
                "additional_width": max(0.0, 2.0 * scale - float(interval.width().detach().cpu())),
                "canonical_support_sha256": hashlib.sha256(
                    (
                        f"g1_symmetric_rebox:{component}:"
                        f"{float(interval.lo.detach().cpu()).hex()}:"
                        f"{float(interval.hi.detach().cpu()).hex()}:"
                        f"{(-scale).hex()}:{scale.hex()}"
                    ).encode("utf-8")
                ).hexdigest(),
                "containment_witness": "outward_symmetric_rebox_contains_source_free_input_interval",
                "fresh_source_excluded": True,
            }
        )
    tmv_right = _scale_tmvector_components(collapsed, inv_scales)
    right_box = tmv_right.range_box()
    if any(float(interval.lo) < -1.0 or float(interval.hi) > 1.0 for interval in right_box):
        raise FloatingPointError("bounded source-ledger normalized base right map leaves [-1,1]")

    source_after_proposed = source_ledger_accepted_successor(
        source_before,
        lift.radius,
        REMAINDER_LEDGER_CATEGORIES,
    )
    source_after = source_ledger_commit_or_preserve(
        source_before,
        source_after_proposed,
        accepted=True,
    )
    reset_tm = _bounded_source_ledger_affine_reset_tm(
        center,
        scales,
        source_after,
        int(order),
        previous_state.domain,
    )
    # Actual-consumer witness: changing a radius changes polynomial
    # coefficients, while state lineage does not enter this hash.
    reset_payload_hash = source_payload_hash(lift.midpoint, lift.radius)
    reset_coeff_hash = hashlib.sha256(
        json.dumps(
            [
                [
                    [list(exponent), float(coefficient.detach().cpu()).hex()]
                    for exponent, coefficient in sorted(model.polynomial.terms.items())
                ]
                for model in reset_tm
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    structured_width = float(torch.sum(2.0 * lift.radius).detach().cpu())
    ordinary_width = sum(float(model.remainder.width().detach().cpu()) for model in collapsed)
    fresh_owner_rows: list[dict[str, Any]] = []
    for component in range(dim):
        midpoint = float(lift.midpoint[0, component].detach().cpu())
        radius = float(lift.radius[0, component].detach().cpu())
        source_id = source_after.source_ids[component]
        fresh_owner_rows.append(
            {
                "category": "fresh_complete_validated_ledger_source",
                "component": component,
                "source_id": source_id,
                "midpoint_hex": midpoint.hex(),
                "outward_lo_hex": (-radius).hex(),
                "outward_hi_hex": radius.hex(),
                "width": 2.0 * radius,
                "canonical_support_sha256": hashlib.sha256(
                    f"{source_id}:{midpoint.hex()}:{radius.hex()}".encode("utf-8")
                ).hexdigest(),
                "containment_witness": "midpoint_plus_affine_source_contains_complete_dense_validated_ledger",
            }
        )
    dense_owner_rows: list[dict[str, Any]] = []
    for category in REMAINDER_LEDGER_CATEGORIES:
        entry_lo, entry_hi = decomposition.ledger.entries[category]
        for component in range(dim):
            lo = float(entry_lo[0, component].detach().cpu())
            hi = float(entry_hi[0, component].detach().cpu())
            dense_owner_rows.append(
                {
                    "category": category,
                    "component": component,
                    "outward_lo_hex": lo.hex(),
                    "outward_hi_hex": hi.hex(),
                    "width": hi - lo,
                    "canonical_support_sha256": hashlib.sha256(
                        f"{category}:{component}:{lo.hex()}:{hi.hex()}".encode("utf-8")
                    ).hexdigest(),
                    "containment_witness": "complete_dense_validated_ledger_contains_unchanged_image",
                }
            )
    diagnostics.update(
        {
            "source_ledger_post_fingerprint": source_after.fingerprint,
            "source_ledger_live_source_count": source_after.live_source_count,
            "source_ledger_source_ids": list(source_after.source_ids),
            "source_ledger_collapse_rows": collapse_rows,
            "source_ledger_retired_owner_rows": retired_owner_rows,
            "source_ledger_carried_ordinary_owner_rows": carried_ordinary_rows,
            "source_ledger_dense_owner_rows": dense_owner_rows,
            "source_ledger_fresh_structured_owner_rows": fresh_owner_rows,
            "source_ledger_insertion_owner_rows": list(diagnostics.pop("_insertion_owner_rows", [])),
            "source_ledger_rebox_owner_rows": rebox_rows,
            "source_ledger_owner_intervals_additive": False,
            "source_ledger_owner_interaction_policy": "owner intervals retained; no exact additive claim",
            "source_ledger_collapse_count": source_after.collapse_count,
            "source_ledger_retired_source_count": source_after.retired_source_count,
            "source_ledger_affine_lift": lift.as_dict(),
            "source_ledger_payload_sha256": reset_payload_hash,
            "source_ledger_reset_coefficients_sha256": reset_coeff_hash,
            "source_ledger_structured_width_mass": structured_width,
            "source_ledger_ordinary_width_mass": ordinary_width,
            "source_ledger_owner_categories": list(REMAINDER_LEDGER_CATEGORIES),
            "source_ledger_next_picard_input_n_vars": reset_tm.n_vars,
            "source_ledger_next_picard_input_active_variables": sorted(reset_tm.active_variables()),
            "source_ledger_no_fallback": True,
            "source_ledger_endpoint_contains_accepted_image": True,
            "scale_x": scales[0] if dim > 0 else "",
            "scale_y": scales[1] if dim > 1 else "",
            "center_x": center[0] if dim > 0 else "",
            "center_y": center[1] if dim > 1 else "",
        }
    )
    _add_width_metrics(diagnostics, "source_ledger_base_inserted", scale_box)
    _add_width_metrics(diagnostics, "source_ledger_reset", reset_tm.range_box())

    state = FlowstarNormalFlowpipeState(
        tmv_pre=seg.tm,
        tmv_right=tmv_right,
        domain=list(previous_state.domain),
        center=center,
        scales=scales,
        step_index=int(previous_state.step_index) + 1,
        diagnostics=diagnostics,
        bounded_source_ledger_state=source_after,
    )
    diagnostics.update(state.diagnostic_widths())
    return reset_tm, state, diagnostics


def _flowstar_g2_shared_column_transition(
    seg: FlowpipeSegment,
    previous_state: FlowstarNormalFlowpipeState,
    order: int,
    *,
    cutoff_threshold: float | None,
    right_map_range_mode: str,
    right_map_center_mode: str,
) -> tuple[TMVector, FlowstarNormalFlowpipeState, dict[str, Any]]:
    """Commit the fixed-3d two-generation G2 boundary, or fail closed."""

    from .batched_dense_tm import REMAINDER_LEDGER_CATEGORIES

    source_before = previous_state.g2_shared_column_state
    retained_before = previous_state.g2_retained_source_tm
    if not isinstance(source_before, G2SharedColumnState) or retained_before is None:
        raise ValueError("G2 transition requires initialized accepted source state")
    if seg.status != "validated" or seg.validated_remainder_decomposition is None:
        raise ValueError("G2 transition requires an accepted complete dense ledger")
    decomposition = seg.validated_remainder_decomposition
    if not bool(torch.all(decomposition.contains_image)):
        raise FloatingPointError("G2 complete ledger does not contain accepted Picard image")
    if decomposition.ledger.category_order != REMAINDER_LEDGER_CATEGORIES:
        raise ValueError("G2 requires the complete O4 owner-ledger schema")
    dim = source_before.state_dim
    if len(seg.final_tm) != dim or len(previous_state.center) != dim:
        raise ValueError("G2 state dimension disagrees with accepted endpoint")
    if len(previous_state.domain) != 3 * dim or seg.final_tm.n_vars != 3 * dim:
        raise ValueError("G2 boundary shape must remain exactly 3d")
    if retained_before.n_vars != 3 * dim or len(retained_before) != dim:
        raise ValueError("G2 retained source payload shape mismatch")
    if right_map_range_mode != "standard" or right_map_center_mode != "constant":
        raise ValueError("G2 preregistration freezes standard range and constant center modes")

    diagnostics: dict[str, Any] = {
        "reset_mode": G2_SHARED_COLUMN_CANDIDATE,
        "step_index": int(previous_state.step_index) + 1,
        "g2_schema": source_before.schema,
        "g2_schema_version": source_before.schema_version,
        "g2_source_generations": 2,
        "g2_pre_fingerprint": source_before.fingerprint,
        "g2_pre_live_source_count": source_before.live_source_count,
        "g2_boundary_atomicity": "accepted_only_immutable_commit",
        "g2_variable_count": 3 * dim,
        "g2_no_fallback": True,
    }

    # Compose base dependency through the right map and both source banks by
    # identity.  Thus a source ID is shared by x and y and by every nonlinear
    # path in the just-completed real dense Picard consumer.
    inner_models = list(previous_state.tmv_right)
    inner_models.extend(
        TaylorModel.variable(dim + index, previous_state.domain, order=order)
        for index in range(2 * dim)
    )
    augmented_inner = TMVector(inner_models)
    endpoint_without_constants = _tmvector_without_remainder(
        _tmvector_rm_constants(seg.final_tm)
    )
    inserted = insert_ctrunc_normal_like(
        endpoint_without_constants,
        augmented_inner,
        int(order),
        cutoff_threshold,
        previous_state.domain,
        diagnostics,
    )
    assert isinstance(inserted, TMVector)

    # Collapse every term containing oldest, including oldest*current terms.
    # Only the surviving current-bearing polynomial is renamed into retained
    # slots; the source-free part and all collapse/remainder mass are reboxed.
    base_models: list[TaylorModel] = []
    retained_models: list[TaylorModel] = []
    collapse_rows: list[dict[str, Any]] = []
    retired_owner_rows: list[Mapping[str, Any]] = []
    carried_ordinary_rows: list[dict[str, Any]] = []
    for component, model in enumerate(inserted):
        ordinary_lo = float(model.remainder.lo.detach().cpu())
        ordinary_hi = float(model.remainder.hi.detach().cpu())
        carried_ordinary_rows.append(
            {
                "category": "ordinary_parameterization_composition_remainder",
                "component": component,
                "outward_lo_hex": ordinary_lo.hex(),
                "outward_hi_hex": ordinary_hi.hex(),
                "width": ordinary_hi - ordinary_lo,
                "canonical_support_sha256": hashlib.sha256(
                    f"ordinary_parameterization:{component}:{ordinary_lo.hex()}:{ordinary_hi.hex()}".encode("utf-8")
                ).hexdigest(),
                "containment_witness": "actual_inserted_TaylorModel_ordinary_remainder",
                "owner_resolution": "cumulative_preexisting_ordinary_dependency_not_recoverably_additive",
            }
        )
        retired_owner_rows.extend(
            g2_owner_rows(
                model.polynomial,
                previous_state.domain,
                component=component,
                oldest_indices=source_before.oldest_indices,
                current_indices=source_before.current_indices,
                oldest_source_ids=source_before.retained_source_ids,
            )
        )
        collapse = collapse_source_polynomial(
            model.polynomial,
            previous_state.domain,
            source_before.oldest_indices,
        )
        surviving = g2_partition_source_terms(
            collapse.retained,
            source_before.current_indices,
        )
        ordinary = model.remainder + collapse.collapsed
        base_models.append(
            TaylorModel(
                surviving.source_free,
                ordinary,
                previous_state.domain,
                order=int(order),
                truncation_range_split=model.truncation_range_split,
            )
        )
        rotated = g2_rotate_current_to_retained(
            surviving.source_bearing,
            dim,
        )
        retained_models.append(
            TaylorModel(
                rotated,
                Interval.zero(
                    dtype=rotated.dtype,
                    device=rotated.device,
                ),
                previous_state.domain,
                order=int(order),
                truncation_range_split=model.truncation_range_split,
            )
        )
        collapse_rows.append(
            {
                "component": component,
                "oldest_source_term_count": collapse.source_term_count,
                "surviving_term_count": collapse.retained_term_count,
                "retained_current_term_count": len(rotated.terms),
                "oldest_support_sha256": collapse.source_support_sha256,
                "surviving_support_sha256": collapse.retained_support_sha256,
                "current_support_sha256": surviving.source_bearing_support_sha256,
                "collapsed_lo_hex": float(collapse.collapsed.lo.detach().cpu()).hex(),
                "collapsed_hi_hex": float(collapse.collapsed.hi.detach().cpu()).hex(),
                "collapsed_width": float(collapse.collapsed.width().detach().cpu()),
                "containment_witness": "canonical_merge_then_single_outward_evaluation",
            }
        )
    base = TMVector(base_models)
    retained_after = TMVector(retained_models)
    retained_payload_hash = g2_polynomial_payload_sha256(
        [model.polynomial for model in retained_after]
    )
    retained_active = tuple(
        any(
            exponent[dim + source]
            for model in retained_after
            for exponent in model.polynomial.terms
        )
        for source in range(dim)
    )

    lift = source_ledger_affine_lift_interval(
        decomposition.decomposition_lo,
        decomposition.decomposition_hi,
    )
    if lift.midpoint.shape != (1, dim):
        raise ValueError("G2 production bridge currently requires B1")
    old_center = _tmvector_constant_part(seg.final_tm)
    center = [
        float(value) + float(lift.midpoint[0, index].detach().cpu())
        for index, value in enumerate(old_center)
    ]

    base_box = base.range_box()
    scales: list[float] = []
    inv_scales: list[float] = []
    rebox_rows: list[dict[str, Any]] = []
    for component, interval in enumerate(base_box):
        magnitude = _interval_magnitude(interval)
        scale = 0.0 if magnitude is None or magnitude == 0.0 else float(magnitude)
        if scale > 0.0:
            for _ in range(8):
                scale = math.nextafter(scale, math.inf)
        scales.append(scale)
        inv_scales.append(1.0 if scale == 0.0 else 1.0 / scale)
        rebox_rows.append(
            {
                "component": component,
                "input_lo_hex": float(interval.lo.detach().cpu()).hex(),
                "input_hi_hex": float(interval.hi.detach().cpu()).hex(),
                "input_width": float(interval.width().detach().cpu()),
                "outward_lo_hex": (-scale).hex(),
                "outward_hi_hex": scale.hex(),
                "width": 2.0 * scale,
                "symmetric_output_width": 2.0 * scale,
                "additional_width": max(0.0, 2.0 * scale - float(interval.width().detach().cpu())),
                "canonical_support_sha256": hashlib.sha256(
                    (
                        f"g2_symmetric_rebox:{component}:"
                        f"{float(interval.lo.detach().cpu()).hex()}:"
                        f"{float(interval.hi.detach().cpu()).hex()}:"
                        f"{(-scale).hex()}:{scale.hex()}"
                    ).encode("utf-8")
                ).hexdigest(),
                "containment_witness": "outward_symmetric_rebox_contains_source_free_input_interval",
                "retained_source_excluded": True,
                "fresh_source_excluded": True,
            }
        )
    tmv_right = _scale_tmvector_components(base, inv_scales)
    right_box = tmv_right.range_box()
    if any(float(interval.lo) < -1.0 or float(interval.hi) > 1.0 for interval in right_box):
        raise FloatingPointError("G2 normalized base right map leaves [-1,1]")

    source_after_proposed = g2_accepted_successor(
        source_before,
        lift.radius,
        REMAINDER_LEDGER_CATEGORIES,
        retained_payload_sha256=retained_payload_hash,
        retained_active=retained_active,
    )
    source_after = g2_commit_or_preserve(
        source_before,
        source_after_proposed,
        accepted=True,
    )
    reset_tm = _g2_shared_column_reset_tm(
        center,
        scales,
        retained_after,
        source_after,
        int(order),
        previous_state.domain,
    )

    dense_owner_rows: list[dict[str, Any]] = []
    for category in REMAINDER_LEDGER_CATEGORIES:
        entry_lo, entry_hi = decomposition.ledger.entries[category]
        for component in range(dim):
            lo = float(entry_lo[0, component].detach().cpu())
            hi = float(entry_hi[0, component].detach().cpu())
            dense_owner_rows.append(
                {
                    "category": category,
                    "component": component,
                    "outward_lo_hex": lo.hex(),
                    "outward_hi_hex": hi.hex(),
                    "width": hi - lo,
                    "canonical_support_sha256": hashlib.sha256(
                        f"{category}:{component}:{lo.hex()}:{hi.hex()}".encode("utf-8")
                    ).hexdigest(),
                    "containment_witness": "complete_dense_validated_ledger_contains_unchanged_image",
                    "additivity": "ledger_additive_before_affine_lift",
                }
            )

    reset_coeff_hash = hashlib.sha256(
        json.dumps(
            [g2_polynomial_table(model.polynomial) for model in reset_tm],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    ordinary_width = sum(float(model.remainder.width().detach().cpu()) for model in base)
    fresh_width = float(torch.sum(2.0 * lift.radius).detach().cpu())
    fresh_owner_rows: list[dict[str, Any]] = []
    for component in range(dim):
        midpoint = float(lift.midpoint[0, component].detach().cpu())
        radius = float(lift.radius[0, component].detach().cpu())
        source_id = source_after.fresh_source_ids[component]
        fresh_owner_rows.append(
            {
                "category": "fresh_complete_validated_ledger_source",
                "component": component,
                "source_id": source_id,
                "midpoint_hex": midpoint.hex(),
                "outward_lo_hex": (-radius).hex(),
                "outward_hi_hex": radius.hex(),
                "width": 2.0 * radius,
                "canonical_support_sha256": hashlib.sha256(
                    f"{source_id}:{midpoint.hex()}:{radius.hex()}".encode("utf-8")
                ).hexdigest(),
                "containment_witness": "midpoint_plus_affine_source_contains_complete_dense_validated_ledger",
            }
        )
    retained_ranges = retained_after.range_box()
    retained_width = sum(float(interval.width().detach().cpu()) for interval in retained_ranges)
    diagnostics.update(
        {
            "g2_post_fingerprint": source_after.fingerprint,
            "g2_retained_source_ids": list(source_after.retained_source_ids),
            "g2_fresh_source_ids": list(source_after.fresh_source_ids),
            "g2_retained_active": list(source_after.retained_active),
            "g2_fresh_active": list(source_after.fresh_active),
            "g2_live_source_count": source_after.live_source_count,
            "g2_collapse_count": source_after.collapse_count,
            "g2_retired_source_count": source_after.retired_source_count,
            "g2_collapse_rows": collapse_rows,
            "g2_retired_owner_rows": retired_owner_rows,
            "g2_carried_ordinary_owner_rows": carried_ordinary_rows,
            "g2_dense_owner_rows": dense_owner_rows,
            "g2_fresh_structured_owner_rows": fresh_owner_rows,
            "g2_insertion_owner_rows": list(diagnostics.pop("_insertion_owner_rows", [])),
            "g2_rebox_owner_rows": rebox_rows,
            "g2_affine_lift": lift.as_dict(),
            "g2_retained_payload_sha256": retained_payload_hash,
            "g2_reset_coefficients_sha256": reset_coeff_hash,
            "g2_ordinary_collapsed_width_mass": ordinary_width,
            "g2_retained_shared_source_width_mass": retained_width,
            "g2_fresh_structured_width_mass": fresh_width,
            "g2_owner_intervals_additive": False,
            "g2_owner_interaction_policy": "per-owner intervals retained; no exact additive claim",
            "g2_next_picard_input_n_vars": reset_tm.n_vars,
            "g2_next_picard_input_active_variables": sorted(reset_tm.active_variables()),
            "g2_endpoint_contains_accepted_image": True,
            "scale_x": scales[0] if dim > 0 else "",
            "scale_y": scales[1] if dim > 1 else "",
            "center_x": center[0] if dim > 0 else "",
            "center_y": center[1] if dim > 1 else "",
        }
    )
    _add_width_metrics(diagnostics, "g2_base_inserted", base_box)
    _add_width_metrics(diagnostics, "g2_retained_source", retained_ranges)
    _add_width_metrics(diagnostics, "g2_reset", reset_tm.range_box())

    state = FlowstarNormalFlowpipeState(
        tmv_pre=seg.tm,
        tmv_right=tmv_right,
        domain=list(previous_state.domain),
        center=center,
        scales=scales,
        step_index=int(previous_state.step_index) + 1,
        diagnostics=diagnostics,
        g2_shared_column_state=source_after,
        g2_retained_source_tm=retained_after,
    )
    diagnostics.update(state.diagnostic_widths())
    return reset_tm, state, diagnostics


def _flowstar_structured_insertion_transition(
    seg: FlowpipeSegment,
    previous_state: FlowstarNormalFlowpipeState,
    order: int,
    *,
    cutoff_threshold: float | None,
    target_remainder_radius: float | None,
    scalar_recenter_remainder_midpoint: bool,
    right_map_range_mode: str,
    right_map_center_mode: str,
    horner_diagnostic: bool,
    allow_outward_renormalization: bool = True,
    image_contract: str = "current",
) -> tuple[TMVector, FlowstarNormalFlowpipeState, dict[str, Any]]:
    """Carry K16 S1 through one already accepted complete-O4 boundary."""
    from .batched_dense_tm import (
        REMAINDER_LEDGER_CATEGORIES,
        sparse_tmvector_to_dense,
    )

    structured = previous_state.structured_remainder_state
    if image_contract not in {"current", "total_delta"}:
        raise ValueError("unknown structured complete-polynomial image contract")
    if not isinstance(structured, StructuredRemainderState):
        raise ValueError("S1 transition requires an initialized structured prestate")
    if seg.status != "validated" or seg.validated_remainder_ledger is None:
        raise ValueError("S1 transition requires an accepted tensor-native validated ledger")
    if structured.batch != 1 or structured.state_dim != len(seg.final_tm):
        raise ValueError("S1 flowpipe bridge requires a compatible batch-one state")

    # Reconstruct the full right map only for the unchanged baseline scale and
    # center calculation. The canonical poststate below stores the polynomial
    # and structured decomposition separately.
    pre_total = materialize_structured_remainder(structured)
    pre_q_lo, pre_q_hi = _box_tensor(
        _tmvector_without_remainder(previous_state.tmv_right).range_box()
    )
    full_previous_right = _tmvector_with_remainder_tensor(
        previous_state.tmv_right,
        pre_total.lo,
        pre_total.hi,
    )
    full_previous = replace(
        previous_state,
        tmv_right=full_previous_right,
        structured_remainder_state=None,
    )
    reset_tm, baseline_state, diagnostics = _flowstar_normalized_insertion_transition(
        seg,
        full_previous,
        order,
        cutoff_threshold=cutoff_threshold,
        target_remainder_radius=target_remainder_radius,
        scalar_recenter_remainder_midpoint=scalar_recenter_remainder_midpoint,
        right_map_range_mode=right_map_range_mode,
        right_map_center_mode=right_map_center_mode,
        horner_diagnostic=horner_diagnostic,
    )

    ordinary_right = _tmvector_with_remainder_tensor(
        previous_state.tmv_right,
        structured.ordinary_rem_lo,
        structured.ordinary_rem_hi,
    )
    base_lo, base_hi = _box_tensor(ordinary_right.range_box())
    old_columns = structured_column_contributions(structured).sum(dim=1)
    endpoint_dense = sparse_tmvector_to_dense(
        _tmvector_without_remainder(seg.final_tm),
        order=int(order),
        dtype=torch.float64,
    )
    identity = torch.eye(
        structured.state_dim, dtype=torch.float64
    ).unsqueeze(0)
    contract_shadow = compare_complete_polynomial_contracts(
        endpoint_dense.poly,
        polynomial_base_domain=(pre_q_lo, pre_q_hi),
        current_base_domain=(base_lo, base_hi),
        ordinary_box=(structured.ordinary_rem_lo, structured.ordinary_rem_hi),
        structured_box=(old_columns.lo, old_columns.hi),
        coordinate_map=identity,
    )
    endpoint_image = (
        contract_shadow.total_delta_image
        if image_contract == "total_delta"
        else contract_shadow.current_image
    )
    selected_base_lo, selected_base_hi = (
        (pre_q_lo, pre_q_hi)
        if image_contract == "total_delta"
        else (base_lo, base_hi)
    )
    selected_perturbation = (
        pre_total
        if image_contract == "total_delta"
        else old_columns
    )
    endpoint_polynomial_lo, endpoint_polynomial_hi = endpoint_dense.poly.range_bound(
        selected_base_lo,
        selected_base_hi,
        context="s1_boundary_attribution_endpoint_polynomial",
    )
    tube_dense = sparse_tmvector_to_dense(
        _tmvector_without_remainder(seg.tm),
        order=int(order),
        dtype=torch.float64,
    )
    tube_base_lo = torch.cat(
        [selected_base_lo, torch.zeros((1, 1), dtype=torch.float64)], dim=1
    )
    tube_base_hi = torch.cat(
        [selected_base_hi, torch.full((1, 1), float(seg.h), dtype=torch.float64)], dim=1
    )
    tube_coordinate = torch.zeros(
        (1, structured.state_dim + 1, structured.state_dim), dtype=torch.float64
    )
    tube_coordinate[:, : structured.state_dim, :] = identity
    tube_image = complete_polynomial_structured_image(
        tube_dense.poly,
        (tube_base_lo, tube_base_hi),
        (selected_perturbation.lo, selected_perturbation.hi),
        tube_coordinate,
        (
            torch.zeros(1, dtype=torch.float64),
            torch.full((1,), float(seg.h), dtype=torch.float64),
        ),
        tau_index=structured.state_dim,
    )
    new_scale = torch.tensor([baseline_state.scales], dtype=torch.float64)
    new_inverse = torch.where(
        new_scale == 0,
        torch.ones_like(new_scale),
        1.0 / new_scale,
    )
    scale_rows = OutwardIntervalTensor.point(new_inverse[:, :, None])
    A_old_normal_to_new_normal = OutwardIntervalTensor(
        endpoint_image.affine_map_lo,
        endpoint_image.affine_map_hi,
    ).mul(scale_rows)
    nonlinear_normal = physical_interval_to_normal(
        endpoint_image.nonlinear_residual_lo,
        endpoint_image.nonlinear_residual_hi,
        forward_scale=new_scale,
        inverse_scale=new_inverse,
    )

    typed_sources = {
        category: tuple(value.clone() for value in seg.validated_remainder_ledger.entries[category])
        for category in REMAINDER_LEDGER_CATEGORIES
    }
    eligible_centers: list[OutwardIntervalTensor] = []
    eligible_symmetric: list[OutwardIntervalTensor] = []
    for category in ELIGIBLE_STRUCTURED_SOURCES:
        center_source, symmetric_source = split_structured_source_center(
            *typed_sources[category]
        )
        eligible_centers.append(center_source)
        eligible_symmetric.append(symmetric_source)
    zero_physical = OutwardIntervalTensor.zeros_like(
        structured.ordinary_rem_lo
    )
    eligible_center_total = outward_sum(eligible_centers) if eligible_centers else zero_physical
    eligible_symmetric_total = outward_sum(eligible_symmetric) if eligible_symmetric else zero_physical
    ineligible_physical = outward_sum(
        [
            OutwardIntervalTensor(*typed_sources[category])
            for category in REMAINDER_LEDGER_CATEGORIES
            if category not in ELIGIBLE_STRUCTURED_SOURCES
        ]
    )
    endpoint_carry_image = OutwardIntervalTensor(
        endpoint_image.reconstruction_lo
        if image_contract == "total_delta"
        else endpoint_image.total_difference_lo,
        endpoint_image.reconstruction_hi
        if image_contract == "total_delta"
        else endpoint_image.total_difference_hi,
    ).add(eligible_symmetric_total)
    tube_carry_image = OutwardIntervalTensor(
        tube_image.reconstruction_lo
        if image_contract == "total_delta"
        else tube_image.total_difference_lo,
        tube_image.reconstruction_hi
        if image_contract == "total_delta"
        else tube_image.total_difference_hi,
    ).add(eligible_symmetric_total)
    endpoint_in_tube = (
        (tube_carry_image.lo <= endpoint_carry_image.lo)
        & (tube_carry_image.hi >= endpoint_carry_image.hi)
    ).all(dim=1)
    if not bool(torch.all(endpoint_in_tube)):
        raise FloatingPointError("S1 endpoint structured image is not contained in tube image")
    target_normal_lo, target_normal_hi = _tmvector_remainder_tensor(
        baseline_state.tmv_right
    )
    target_physical = normal_interval_to_physical(
        target_normal_lo,
        target_normal_hi,
        forward_scale=new_scale,
    )
    target_normal_effective = physical_interval_to_normal(
        target_physical.lo,
        target_physical.hi,
        forward_scale=new_scale,
        inverse_scale=new_inverse,
    )

    propagated_known = _interval_matrix_vector(
        A_old_normal_to_new_normal,
        pre_total,
    )
    normalized_source_terms = [
        physical_interval_to_normal(
            *typed_sources[category],
            forward_scale=new_scale,
            inverse_scale=new_inverse,
        )
        for category in REMAINDER_LEDGER_CATEGORIES
    ]
    known = outward_sum(
        [propagated_known, *normalized_source_terms, nonlinear_normal]
    )
    padding_normal = _s1_padding_to_target(known, target_normal_effective)

    # Phase-2 shadow only: evaluate the complete retained polynomial over
    # Q + Delta with Delta = R_o + Z.  This path is recorded for causal and
    # oracle comparison but does not feed the committed boundary update.
    total_delta_image = contract_shadow.total_delta_image
    total_A_old_normal_to_new_normal = OutwardIntervalTensor(
        total_delta_image.affine_map_lo,
        total_delta_image.affine_map_hi,
    ).mul(scale_rows)
    total_delta_propagated = _interval_matrix_vector(
        total_A_old_normal_to_new_normal,
        pre_total,
    )
    total_delta_nonlinear_normal = physical_interval_to_normal(
        total_delta_image.nonlinear_residual_lo,
        total_delta_image.nonlinear_residual_hi,
        forward_scale=new_scale,
        inverse_scale=new_inverse,
    )
    total_delta_known = outward_sum(
        [total_delta_propagated, *normalized_source_terms, total_delta_nonlinear_normal]
    )
    total_delta_padding = _s1_padding_to_target(
        total_delta_known,
        target_normal_effective,
    )
    current_reconstruction_with_padding = known.add(padding_normal)
    total_delta_reconstruction_with_padding = total_delta_known.add(
        total_delta_padding
    )
    total_delta_contains_target = (
        (total_delta_reconstruction_with_padding.lo <= target_normal_effective.lo)
        & (total_delta_reconstruction_with_padding.hi >= target_normal_effective.hi)
    ).all(dim=1)
    current_contains_target = (
        (current_reconstruction_with_padding.lo <= target_normal_effective.lo)
        & (current_reconstruction_with_padding.hi >= target_normal_effective.hi)
    ).all(dim=1)
    padding_physical = normal_interval_to_physical(
        padding_normal.lo,
        padding_normal.hi,
        forward_scale=new_scale,
    )
    padding_physical_lo = padding_physical.lo
    padding_physical_hi = padding_physical.hi
    for _ in range(4):
        padding_physical_lo = torch.nextafter(
            padding_physical_lo, torch.full_like(padding_physical_lo, -torch.inf)
        )
        padding_physical_hi = torch.nextafter(
            padding_physical_hi, torch.full_like(padding_physical_hi, torch.inf)
        )
    padding_physical = OutwardIntervalTensor(
        padding_physical_lo, padding_physical_hi
    )
    reset_existing = OutwardIntervalTensor(*typed_sources["reset_or_reconditioning"])
    reset_with_padding = reset_existing.add(padding_physical)
    typed_sources["reset_or_reconditioning"] = (
        reset_with_padding.lo,
        reset_with_padding.hi,
    )

    boundary = structured_remainder_boundary_update(
        structured,
        typed_sources=typed_sources,
        validated_remainder_lo=target_physical.lo,
        validated_remainder_hi=target_physical.hi,
        A_old_normal_to_new_normal_lo=A_old_normal_to_new_normal.lo,
        A_old_normal_to_new_normal_hi=A_old_normal_to_new_normal.hi,
        nonlinear_residual_lo=nonlinear_normal.lo,
        nonlinear_residual_hi=nonlinear_normal.hi,
        new_forward_scale=new_scale,
        boundary_index=structured.accepted_boundary_index,
        map_is_affine=False,
    )
    reconstructed_before_renormalization = OutwardIntervalTensor(
        boundary.materialized_lo,
        boundary.materialized_hi,
    )
    if not bool(torch.all(boundary.accepted)):
        raise FloatingPointError(
            "S1 accepted-boundary conservation failed: "
            f"{boundary.failure_reason}; "
            f"source={boundary.source_decomposition_mask.detach().cpu().tolist()}; "
            f"conservation={boundary.conservation_mask.detach().cpu().tolist()}; "
            f"pre=({boundary.pre_split_lo.detach().cpu().tolist()},"
            f"{boundary.pre_split_hi.detach().cpu().tolist()}); "
            f"validated=({target_normal_effective.lo.detach().cpu().tolist()},"
            f"{target_normal_effective.hi.detach().cpu().tolist()})"
        )

    next_right = _tmvector_without_remainder(baseline_state.tmv_right)
    next_total = materialize_structured_remainder(boundary.state)
    right_poly_lo, right_poly_hi = _box_tensor(next_right.range_box())
    right_total = OutwardIntervalTensor(right_poly_lo, right_poly_hi).add(next_total)
    domain_gate = ((right_total.lo >= -1.0) & (right_total.hi <= 1.0)).all(dim=1)
    renormalization_count = 0
    # The baseline scale is formed from a binary64 interval magnitude.  A
    # subsequent outward sum can therefore exceed one by a few ulps even when
    # the represented physical set is unchanged.  Enlarge only the affected
    # forward scales and transform every normalized owner together; this is a
    # coordinate change, not a tolerance on the [-1,1] obligation.
    while (
        allow_outward_renormalization
        and not bool(torch.all(domain_gate))
        and renormalization_count < 4
    ):
        magnitude = torch.maximum(torch.abs(right_total.lo), torch.abs(right_total.hi))
        needs_scale = magnitude > 1.0
        if not bool(torch.any(needs_scale)):
            break
        safe_scale = new_scale.clone()
        requested = torch.where(needs_scale, new_scale * magnitude, new_scale)
        for _ in range(8):
            requested = torch.where(
                needs_scale,
                torch.nextafter(requested, torch.full_like(requested, torch.inf)),
                requested,
            )
        safe_scale = torch.where(needs_scale, requested, safe_scale)
        ratio = torch.where(
            safe_scale == 0,
            torch.ones_like(safe_scale),
            new_scale / safe_scale,
        )
        ordinary_rescaled = OutwardIntervalTensor(
            boundary.state.ordinary_rem_lo,
            boundary.state.ordinary_rem_hi,
        ).mul(OutwardIntervalTensor.point(ratio))
        phi_rescaled = OutwardIntervalTensor(
            boundary.state.phi_lo,
            boundary.state.phi_hi,
        ).mul(OutwardIntervalTensor.point(ratio[:, None, :, None]))
        safe_inverse = torch.where(
            safe_scale == 0,
            torch.ones_like(safe_scale),
            1.0 / safe_scale,
        )
        rescaled_structured = replace(
            boundary.state,
            ordinary_rem_lo=ordinary_rescaled.lo,
            ordinary_rem_hi=ordinary_rescaled.hi,
            phi_lo=phi_rescaled.lo,
            phi_hi=phi_rescaled.hi,
            inverse_scale=safe_inverse,
        )
        next_right = _scale_tmvector_components(
            next_right,
            [float(value) for value in ratio[0].detach().cpu()],
        )
        next_total = materialize_structured_remainder(rescaled_structured)
        boundary = replace(
            boundary,
            state=rescaled_structured,
            materialized_lo=next_total.lo,
            materialized_hi=next_total.hi,
        )
        new_scale = safe_scale
        new_inverse = safe_inverse
        safe_scales = [float(value) for value in safe_scale[0].detach().cpu()]
        reset_tm = _normalized_tm_from_center_scale(
            baseline_state.center,
            safe_scales,
            int(order),
            template_domain=baseline_state.domain,
        )
        baseline_state = replace(
            baseline_state,
            tmv_right=next_right,
            scales=safe_scales,
        )
        right_poly_lo, right_poly_hi = _box_tensor(next_right.range_box())
        right_total = OutwardIntervalTensor(right_poly_lo, right_poly_hi).add(next_total)
        domain_gate = ((right_total.lo >= -1.0) & (right_total.hi <= 1.0)).all(dim=1)
        renormalization_count += 1
    if not bool(torch.all(domain_gate)):
        raise FloatingPointError(
            "S1 normalized right-map total leaves [-1,1]: "
            f"lo={right_total.lo.detach().cpu().tolist()}; "
            f"hi={right_total.hi.detach().cpu().tolist()}; "
            f"poly_lo={right_poly_lo.detach().cpu().tolist()}; "
            f"poly_hi={right_poly_hi.detach().cpu().tolist()}; "
            f"remainder_lo={next_total.lo.detach().cpu().tolist()}; "
            f"remainder_hi={next_total.hi.detach().cpu().tolist()}"
        )

    active_endpoint = structured_column_contributions(boundary.state).sum(dim=1)
    active_endpoint_physical = normal_interval_to_physical(
        active_endpoint.lo,
        active_endpoint.hi,
        forward_scale=new_scale,
    )
    ordinary_endpoint_physical = normal_interval_to_physical(
        boundary.state.ordinary_rem_lo,
        boundary.state.ordinary_rem_hi,
        forward_scale=new_scale,
    )
    endpoint_split_total = ordinary_endpoint_physical.add(active_endpoint_physical)
    endpoint_materialized_total = normal_interval_to_physical(
        next_total.lo,
        next_total.hi,
        forward_scale=new_scale,
    )
    # The two routes differ only by outward addition grouping.  Publish their
    # hull so the first-class total certifies both the split and direct
    # materialization semantics.
    endpoint_total = OutwardIntervalTensor(
        torch.minimum(endpoint_split_total.lo, endpoint_materialized_total.lo),
        torch.maximum(endpoint_split_total.hi, endpoint_materialized_total.hi),
    )
    ineligible_tube_terms = [
        OutwardIntervalTensor(*typed_sources[category])
        for category in REMAINDER_LEDGER_CATEGORIES
        if category not in ELIGIBLE_STRUCTURED_SOURCES
    ]
    if image_contract == "total_delta":
        tube_affine = OutwardIntervalTensor(
            tube_image.affine_map_lo,
            tube_image.affine_map_hi,
        )
        tube_affine_ordinary = _interval_matrix_vector(
            tube_affine,
            OutwardIntervalTensor(
                structured.ordinary_rem_lo,
                structured.ordinary_rem_hi,
            ),
        )
        tube_affine_structured = _interval_matrix_vector(
            tube_affine,
            old_columns,
        )
        tube_nonlinear = OutwardIntervalTensor(
            tube_image.nonlinear_residual_lo,
            tube_image.nonlinear_residual_hi,
        )
        tube_ordinary = outward_sum(
            [
                tube_affine_ordinary,
                tube_nonlinear,
                *ineligible_tube_terms,
                eligible_center_total,
            ]
        )
        tube_structured_image = tube_affine_structured.add(
            eligible_symmetric_total
        )
    else:
        tube_ordinary = outward_sum(
            [*ineligible_tube_terms, eligible_center_total]
        )
        tube_structured_image = tube_carry_image
    tube_total = tube_ordinary.add(tube_structured_image)
    tube_publication_padding = OutwardIntervalTensor.zeros_like(tube_total.lo)
    if image_contract == "total_delta":
        tube_publication_padding = _s1_padding_to_target(
            tube_total,
            endpoint_total,
        )
        tube_ordinary = tube_ordinary.add(tube_publication_padding)
        tube_total = tube_ordinary.add(tube_structured_image)
    published_endpoint_in_tube = (
        (tube_total.lo <= endpoint_total.lo)
        & (tube_total.hi >= endpoint_total.hi)
    ).all(dim=1)
    endpoint_publication = verify_structured_publication(
        ordinary_endpoint_physical.lo,
        ordinary_endpoint_physical.hi,
        active_endpoint_physical.lo,
        active_endpoint_physical.hi,
        endpoint_total.lo,
        endpoint_total.hi,
    )
    tube_publication = verify_structured_publication(
        tube_ordinary.lo,
        tube_ordinary.hi,
        tube_structured_image.lo,
        tube_structured_image.hi,
        tube_total.lo,
        tube_total.hi,
    )
    required_publication = endpoint_publication & tube_publication
    if image_contract == "total_delta":
        required_publication = required_publication & published_endpoint_in_tube
    if not bool(torch.all(required_publication)):
        raise FloatingPointError(
            "S1 endpoint/tube publication omitted a represented contribution: "
            f"endpoint_publication={endpoint_publication.detach().cpu().tolist()}; "
            f"tube_publication={tube_publication.detach().cpu().tolist()}; "
            f"published_endpoint_in_tube={published_endpoint_in_tube.detach().cpu().tolist()}; "
            f"endpoint_total=({endpoint_total.lo.detach().cpu().tolist()},"
            f"{endpoint_total.hi.detach().cpu().tolist()}); "
            f"tube_total=({tube_total.lo.detach().cpu().tolist()},"
            f"{tube_total.hi.detach().cpu().tolist()})"
        )
    endpoint_affine = OutwardIntervalTensor(
        endpoint_image.affine_map_lo,
        endpoint_image.affine_map_hi,
    )
    endpoint_affine_ordinary = _interval_matrix_vector(
        endpoint_affine,
        OutwardIntervalTensor(
            structured.ordinary_rem_lo,
            structured.ordinary_rem_hi,
        ),
    )
    endpoint_affine_structured = _interval_matrix_vector(
        endpoint_affine,
        old_columns,
    )
    total_endpoint_affine = OutwardIntervalTensor(
        total_delta_image.affine_map_lo,
        total_delta_image.affine_map_hi,
    )
    total_endpoint_affine_ordinary = _interval_matrix_vector(
        total_endpoint_affine,
        OutwardIntervalTensor(
            structured.ordinary_rem_lo,
            structured.ordinary_rem_hi,
        ),
    )
    total_endpoint_affine_structured = _interval_matrix_vector(
        total_endpoint_affine,
        old_columns,
    )
    seg.boundary_attribution_record = S1BoundaryAttributionRecord(
        accepted_boundary_index_before=int(structured.accepted_boundary_index),
        contract=(
            "C_total_delta" if image_contract == "total_delta" else "C_current"
        ),
        stages=(
            S1BoundaryStage("A0", "prestate polynomial Q range", "old normalized", pre_q_lo, pre_q_hi),
            S1BoundaryStage(
                "A1",
                "prestate ordinary remainder R_o",
                "old normalized",
                structured.ordinary_rem_lo,
                structured.ordinary_rem_hi,
            ),
            S1BoundaryStage("A2", "prestate structured total Z", "old normalized", old_columns.lo, old_columns.hi),
            S1BoundaryStage("A3", "materialized total R_o + Z", "old normalized", pre_total.lo, pre_total.hi),
            S1BoundaryStage(
                "B0",
                "canonical baseline insertion target R_base",
                "new normalized",
                target_normal_effective.lo,
                target_normal_effective.hi,
            ),
            S1BoundaryStage(
                "B1",
                "endpoint complete polynomial P over current base",
                "endpoint physical",
                endpoint_polynomial_lo,
                endpoint_polynomial_hi,
            ),
            S1BoundaryStage(
                "B2",
                "selected complete-polynomial base box",
                "old normalized",
                selected_base_lo,
                selected_base_hi,
            ),
            S1BoundaryStage(
                "B3",
                "selected complete-polynomial perturbation box",
                "old normalized",
                selected_perturbation.lo,
                selected_perturbation.hi,
            ),
            S1BoundaryStage(
                "B4",
                "affine map A",
                "endpoint physical",
                endpoint_image.affine_map_lo,
                endpoint_image.affine_map_hi,
            ),
            S1BoundaryStage(
                "B5",
                "A times ordinary remainder",
                "endpoint physical",
                endpoint_affine_ordinary.lo,
                endpoint_affine_ordinary.hi,
            ),
            S1BoundaryStage(
                "B6",
                "A times structured total",
                "endpoint physical",
                endpoint_affine_structured.lo,
                endpoint_affine_structured.hi,
            ),
            S1BoundaryStage(
                "B7",
                "structured nonlinear residual N",
                "endpoint physical",
                endpoint_image.nonlinear_residual_lo,
                endpoint_image.nonlinear_residual_hi,
            ),
            S1BoundaryStage(
                "B8",
                "typed eligible centers",
                "physical source",
                eligible_center_total.lo,
                eligible_center_total.hi,
            ),
            S1BoundaryStage(
                "B9",
                "typed eligible symmetric sources",
                "physical source",
                eligible_symmetric_total.lo,
                eligible_symmetric_total.hi,
            ),
            S1BoundaryStage(
                "B10",
                "ineligible typed sources",
                "physical source",
                ineligible_physical.lo,
                ineligible_physical.hi,
            ),
            S1BoundaryStage("B11", "known before padding", "new normalized", known.lo, known.hi),
            S1BoundaryStage("B12", "padding", "new normalized", padding_normal.lo, padding_normal.hi),
            S1BoundaryStage(
                "B13",
                "reconstructed total before renormalization",
                "new normalized",
                reconstructed_before_renormalization.lo,
                reconstructed_before_renormalization.hi,
            ),
            S1BoundaryStage(
                "B14",
                "reconstructed total after renormalization",
                "new normalized",
                next_total.lo,
                next_total.hi,
            ),
            S1BoundaryStage("B15", "published endpoint total", "endpoint physical", endpoint_total.lo, endpoint_total.hi),
            S1BoundaryStage("B16", "published tube total", "tube physical", tube_total.lo, tube_total.hi),
        ),
        diagnostics={
            "complete_polynomial_contract": (
                "base=range(Q), perturbation=R_o+Z"
                if image_contract == "total_delta"
                else "base=range(Q+R_o), perturbation=Z"
            ),
            "structured_image_decomposition_padding_lo": endpoint_image.decomposition_padding_lo.detach().cpu().tolist(),
            "structured_image_decomposition_padding_hi": endpoint_image.decomposition_padding_hi.detach().cpu().tolist(),
            "structured_image_containment": bool(torch.all(endpoint_image.containment_mask)),
            "source_decomposition": bool(torch.all(boundary.source_decomposition_mask)),
            "conservation": bool(torch.all(boundary.conservation_mask)),
            "outward_renormalization_count": int(renormalization_count),
            "published_endpoint_in_tube": bool(
                torch.all(published_endpoint_in_tube)
            ),
            "tube_publication_padding": _s1_interval_diagnostic(
                tube_publication_padding,
                units="tube physical",
            ),
            "C_total_delta_shadow": {
                "status": (
                    "selected_production_contract_outcome_B"
                    if image_contract == "total_delta"
                    else "diagnostic_only_not_production"
                ),
                "contract": "base=range(Q), perturbation=R_o+Z",
                "base_Q": _s1_interval_diagnostic(
                    OutwardIntervalTensor(pre_q_lo, pre_q_hi),
                    units="old normalized",
                ),
                "delta_R_o_plus_Z": _s1_interval_diagnostic(
                    pre_total,
                    units="old normalized",
                ),
                "affine_map_A_total": _s1_interval_diagnostic(
                    total_endpoint_affine,
                    units="endpoint physical",
                ),
                "A_total_times_R_o": _s1_interval_diagnostic(
                    total_endpoint_affine_ordinary,
                    units="endpoint physical",
                ),
                "A_total_times_Z": _s1_interval_diagnostic(
                    total_endpoint_affine_structured,
                    units="endpoint physical",
                ),
                "N_total": _s1_interval_diagnostic(
                    OutwardIntervalTensor(
                        total_delta_image.nonlinear_residual_lo,
                        total_delta_image.nonlinear_residual_hi,
                    ),
                    units="endpoint physical",
                ),
                "known_before_padding": _s1_interval_diagnostic(
                    total_delta_known,
                    units="new normalized",
                ),
                "current_known_before_padding": _s1_interval_diagnostic(
                    known,
                    units="new normalized",
                ),
                "padding": _s1_interval_diagnostic(
                    total_delta_padding,
                    units="new normalized",
                ),
                "current_padding": _s1_interval_diagnostic(
                    padding_normal,
                    units="new normalized",
                ),
                "reconstructed_after_padding": _s1_interval_diagnostic(
                    total_delta_reconstruction_with_padding,
                    units="new normalized",
                ),
                "current_reconstructed_after_padding": _s1_interval_diagnostic(
                    current_reconstruction_with_padding,
                    units="new normalized",
                ),
                "canonical_target_contained_after_padding": bool(
                    torch.all(total_delta_contains_target)
                ),
                "current_target_contained_after_padding": bool(
                    torch.all(current_contains_target)
                ),
                "current_unpadded_contains_total_delta_direct": bool(
                    torch.all(contract_shadow.current_contains_total_delta_mask)
                ),
                "total_not_wider_before_padding_mask": (
                    (total_delta_known.hi - total_delta_known.lo)
                    <= (known.hi - known.lo)
                ).detach().cpu().tolist(),
                "total_not_wider_after_padding_mask": (
                    (
                        total_delta_reconstruction_with_padding.hi
                        - total_delta_reconstruction_with_padding.lo
                    )
                    <= (
                        current_reconstruction_with_padding.hi
                        - current_reconstruction_with_padding.lo
                    )
                ).detach().cpu().tolist(),
                "nonlinear_route_count": int(
                    total_delta_image.proof_diagnostics["nonlinear_route_count"]
                ),
                "ordinary_structured_mixed_routes": "included_in_materialized_Delta_degree_ge_2",
            },
        },
    )
    seg.structured_boundary_result = boundary
    seg.structured_state_before = structured
    seg.structured_state_after = boundary.state
    seg.endpoint_total_structured_remainder = active_endpoint_physical
    seg.tube_total_structured_remainder = tube_structured_image
    seg.endpoint_ordinary_remainder = ordinary_endpoint_physical
    seg.tube_ordinary_remainder = tube_ordinary
    seg.endpoint_total_remainder = endpoint_total
    seg.tube_total_remainder = tube_total
    seg.endpoint_publication_mask = endpoint_publication
    seg.tube_publication_mask = tube_publication

    diagnostics.update(
        {
            "structured_candidate": (
                STRUCTURED_TOTAL_DELTA_CANDIDATE
                if image_contract == "total_delta"
                else STRUCTURED_REMAINDER_CANDIDATE
            ),
            "structured_image_contract": image_contract,
            "structured_boundary_index_before": structured.accepted_boundary_index,
            "structured_boundary_index_after": boundary.state.accepted_boundary_index,
            "structured_active_columns": int(boundary.state.active.sum().item()),
            "structured_event_count": int(boundary.state.event_count.sum().item()),
            "structured_conservation": bool(torch.all(boundary.conservation_mask)),
            "structured_source_decomposition": bool(torch.all(boundary.source_decomposition_mask)),
            "structured_total_self_map_containment": bool(torch.all(domain_gate)),
            "structured_outward_renormalization_count": renormalization_count,
            "structured_endpoint_in_tube": bool(torch.all(endpoint_in_tube)),
            "structured_published_endpoint_in_tube": bool(
                torch.all(published_endpoint_in_tube)
            ),
            "structured_tube_publication_padding_lo": tube_publication_padding.lo.detach().cpu().tolist(),
            "structured_tube_publication_padding_hi": tube_publication_padding.hi.detach().cpu().tolist(),
            "structured_endpoint_publication": bool(torch.all(endpoint_publication)),
            "structured_tube_publication": bool(torch.all(tube_publication)),
            "structured_raw_picard_target_changed": False,
            "structured_ordinary_target_margin": seg.subset_margin,
            "structured_endpoint_total_remainder_lo": active_endpoint_physical.lo.detach().cpu().tolist(),
            "structured_endpoint_total_remainder_hi": active_endpoint_physical.hi.detach().cpu().tolist(),
            "structured_published_endpoint_total_lo": endpoint_total.lo.detach().cpu().tolist(),
            "structured_published_endpoint_total_hi": endpoint_total.hi.detach().cpu().tolist(),
            "structured_published_tube_total_lo": tube_total.lo.detach().cpu().tolist(),
            "structured_published_tube_total_hi": tube_total.hi.detach().cpu().tolist(),
            "structured_nonlinear_residual_lo": endpoint_image.nonlinear_residual_lo.detach().cpu().tolist(),
            "structured_nonlinear_residual_hi": endpoint_image.nonlinear_residual_hi.detach().cpu().tolist(),
        }
    )
    state = replace(
        baseline_state,
        tmv_right=next_right,
        diagnostics=diagnostics,
        structured_remainder_state=boundary.state,
    )
    return reset_tm, state, diagnostics


def _sum_interval_widths(boxes: Sequence[Interval]) -> float | str:
    widths = [_interval_width_value(iv) for iv in boxes]
    finite = [w for w in widths if w is not None]
    return sum(finite) if len(finite) == len(boxes) else ""


def _add_width_metrics(row: dict[str, Any], prefix: str, boxes: Sequence[Interval] | None) -> None:
    if boxes is None:
        return
    names = ("x", "y")
    widths: list[float] = []
    for i, iv in enumerate(boxes[:2]):
        width = _interval_width_value(iv)
        if width is not None:
            row[f"{prefix}_width_{names[i]}"] = width
            widths.append(width)
    if widths:
        row[f"{prefix}_width_sum"] = sum(widths)


def _add_interval_bounds(row: dict[str, Any], prefix: str, boxes: Sequence[Interval] | None) -> None:
    if boxes is None:
        return
    names = ("x", "y")
    for i, iv in enumerate(boxes[:2]):
        lo = _interval_bound_value(iv.lo)
        hi = _interval_bound_value(iv.hi)
        if lo is not None:
            row[f"{prefix}_lo_{names[i]}"] = lo
        if hi is not None:
            row[f"{prefix}_hi_{names[i]}"] = hi


def _polynomial_range_boxes(tm: TMVector) -> list[Interval]:
    return [m.polynomial.evaluate_interval(m.domain) for m in tm]


def _final_range_boxes(tm: TMVector, tau_index: int, h: float) -> list[Interval] | None:
    try:
        return tm.substitute_const(tau_index, float(h)).drop_variable(tau_index).range_box()
    except Exception:
        return None


def _append_validation_diagnostic(
    diagnostics: list[dict[str, Any]] | None,
    *,
    mode: str | None,
    segment_index: int | None,
    attempt_index: int,
    h: float,
    order: int,
    candidate: TMVector | None,
    tau_index: int,
    residual_boxes: Sequence[Interval] | None,
    remainders: Sequence[Interval] | None,
    finite_residual: bool | None,
    validation_status: str,
    validation_message: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    if diagnostics is None:
        return
    row: dict[str, Any] = {
        "mode": mode or "",
        "segment_index": segment_index if segment_index is not None else "",
        "attempt_index": attempt_index,
        "h": float(h),
        "order": int(order),
        "finite_residual": finite_residual if finite_residual is not None else "",
        "validation_status": validation_status,
        "validation_message": validation_message,
    }
    if extra:
        for key, value in extra.items():
            if key not in row:
                row[key] = value
    if candidate is not None:
        output_order_value = _float_or_none(row.get("output_order"))
        _add_term_hash_metrics(
            row,
            "validation_candidate_inside",
            candidate,
            int(output_order_value) if output_order_value is not None else None,
        )
        _add_term_hash_metrics(
            row,
            "validation_candidate_after_internal",
            candidate,
            int(output_order_value) if output_order_value is not None else None,
        )
        try:
            candidate_box = candidate.range_box()
            _add_width_metrics(row, "candidate_segment", candidate_box)
            _add_width_metrics(row, "total_range", candidate_box)
        except Exception:
            pass
        try:
            _add_width_metrics(row, "candidate_final", _final_range_boxes(candidate, tau_index, h))
        except Exception:
            pass
        try:
            polynomial_boxes = _polynomial_range_boxes(candidate)
            _add_width_metrics(row, "polynomial_range", polynomial_boxes)
            _add_interval_bounds(row, "polynomial_range", polynomial_boxes)
        except Exception:
            pass
    _add_width_metrics(row, "residual", residual_boxes)
    _add_interval_bounds(row, "residual", residual_boxes)
    _add_width_metrics(row, "remainder", remainders)
    diagnostics.append(row)


def _call_ode(ode_fn: ODEFunction, x: TMVector, u: TMVector | None) -> TMVector:
    try:
        out = ode_fn(x, u)
    except TypeError:
        out = ode_fn(x)
    if isinstance(out, TMVector):
        return out
    return TMVector(out)


def _constant_control_tms(
    u_box: Sequence[Any] | None,
    domain: Sequence[Interval],
    order: int,
    *,
    truncation_range_split: int | None = None,
) -> TMVector | None:
    if u_box is None:
        return None
    controls: list[TaylorModel] = []
    for u in u_box:
        iv = ensure_interval(u) if not isinstance(u, (tuple, list)) else Interval(u[0], u[1])
        controls.append(
            TaylorModel.constant(
                iv.mid(),
                domain,
                order=order,
                remainder=Interval(-iv.radius(), iv.radius()),
                truncation_range_split=truncation_range_split,
            )
        )
    return TMVector(controls)


def _affine_control_tms(
    affine_u: dict[str, Any] | None,
    domain: Sequence[Interval],
    order: int,
    *,
    truncation_range_split: int | None = None,
) -> TMVector | None:
    if affine_u is None:
        return None
    A = torch.as_tensor(affine_u.get("A"), dtype=domain[0].lo.dtype if domain else torch.float64)
    b = torch.as_tensor(affine_u.get("b", torch.zeros(A.shape[0])), dtype=A.dtype, device=A.device)
    error = affine_u.get("error", None)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if b.ndim == 0:
        b = b.reshape(1)
    n_u, n_x = A.shape
    if n_x > len(domain):
        raise ValueError("affine control has more input columns than active variables")
    variables = [
        TaylorModel.variable(i, domain, order=order, truncation_range_split=truncation_range_split)
        for i in range(n_x)
    ]
    controls: list[TaylorModel] = []
    for j in range(n_u):
        tm = TaylorModel.constant(b[j], domain, order=order, truncation_range_split=truncation_range_split)
        for i in range(n_x):
            if float(A[j, i]) != 0.0:
                tm = tm + variables[i] * A[j, i]
        if error is not None:
            e = error[j] if isinstance(error, (list, tuple)) else error
            if isinstance(e, Interval):
                err_iv = e
            else:
                rad = torch.as_tensor(e, dtype=A.dtype, device=A.device).abs()
                err_iv = Interval(-rad, rad)
            tm = tm + err_iv
        controls.append(tm)
    return TMVector(controls)


def _make_controls(
    u_box: Sequence[Any] | None,
    affine_u: dict[str, Any] | None,
    domain: Sequence[Interval],
    order: int,
    *,
    truncation_range_split: int | None = None,
) -> TMVector | None:
    u_const = _constant_control_tms(u_box, domain, order, truncation_range_split=truncation_range_split)
    u_affine = _affine_control_tms(affine_u, domain, order, truncation_range_split=truncation_range_split)
    if u_const is not None and u_affine is not None:
        if len(u_const) != len(u_affine):
            raise ValueError("u_box and affine_u dimensions do not match")
        return TMVector(a + b for a, b in zip(u_const, u_affine))
    return u_const if u_const is not None else u_affine


def _picard_polynomial(
    ode_fn: ODEFunction,
    base_poly_ext: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    iterations: int | None = None,
    cutoff_threshold: float | None = None,
    truncation_range_split: int | None = None,
) -> TMVector:
    """Construct the polynomial part of a Picard iterate.

    Remainders created by truncation are intentionally not fed back into the
    polynomial iterate.  They are accounted for by the later validation loop.
    """
    iterations = order if iterations is None else iterations
    domain = base_poly_ext.domain
    g = base_poly_ext
    for _ in range(max(1, iterations)):
        rhs = _call_ode(ode_fn, g, u_tms)
        next_models: list[TaylorModel] = []
        for x0_i, f_i in zip(base_poly_ext, rhs):
            integ = f_i.integrate(tau_index)
            tm_i = x0_i + integ
            poly, _dropped = tm_i.polynomial.truncate(order)
            next_tm = _zero_remainder_tm(
                poly,
                domain,
                order,
                truncation_range_split=truncation_range_split,
            ).apply_cutoff(cutoff_threshold)
            next_models.append(next_tm)
        g = TMVector(next_models)
    return g


def _validate_picard(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate_poly: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    max_attempts: int,
    validation_eps: float,
    growth_factor: float,
    h: float,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    diagnostic_mode: str | None = None,
    diagnostic_segment_index: int | None = None,
    diagnostic_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    symbolic_remainder: bool = False,
    max_symbolic_remainders: int = 0,
) -> tuple[TMVector, str, int, str]:
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    remainders: list[Interval] = []
    for base_i, candidate_i in zip(base_ext, candidate_poly):
        remainders.append(_combine_remainders(base_i.remainder, candidate_i.remainder).inflate(validation_eps))
    if diagnostic_context is not None:
        diagnostics_context = diagnostic_context
    if diagnostic_mode is not None:
        diagnostics_mode = diagnostic_mode
    if diagnostic_segment_index is not None:
        diagnostics_segment_index = diagnostic_segment_index
    diag_extra = dict(diagnostics_context or {})
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)
    if not intervals_are_finite(remainders):
        message = "non-finite initial remainder"
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate_poly,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=remainders,
            finite_residual=False,
            validation_status="failed",
            validation_message=message,
            extra=diag_extra,
        )
        return candidate_poly, "failed", 0, message

    for attempt in range(1, max_attempts + 1):
        candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(candidate_poly, remainders)
        )
        if rhs_breakdown_callback is not None:
            callback_context = dict(diag_extra)
            if diag_mode is not None:
                callback_context["mode"] = diag_mode
            if diag_segment_index is not None:
                callback_context["segment_index"] = diag_segment_index
            callback_context["attempt_index"] = attempt
            callback_context["h"] = float(h)
            callback_context["order"] = int(order)
            try:
                rhs_breakdown_callback(candidate, order, attempt, callback_context)
            except Exception:
                pass
        try:
            residual_boxes = _picard_residual_boxes(
                ode_fn,
                base_ext,
                candidate,
                tau_index,
                order,
                u_tms,
                validation_eps=validation_eps,
            )
        except Exception as exc:  # fail closed; caller gets a non-validated segment
            message = f"validation exception: {exc}"
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=None,
                remainders=remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=diag_extra,
            )
            return candidate, "failed", attempt, message

        finite_residual = intervals_are_finite(residual_boxes)
        if not finite_residual:
            message = "non-finite residual interval"
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_boxes,
                remainders=remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=diag_extra,
            )
            return candidate, "failed", attempt, message

        if all(r.contains_interval(rb) for r, rb in zip(remainders, residual_boxes)):
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_boxes,
                remainders=remainders,
                finite_residual=True,
                validation_status="validated",
                validation_message="",
                extra=diag_extra,
            )
            return candidate, "validated", attempt, ""

        message = "Picard remainder validation did not converge" if attempt == max_attempts else "residual not contained by current remainder"
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=attempt,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=residual_boxes,
            remainders=remainders,
            finite_residual=True,
            validation_status="failed" if attempt == max_attempts else "needs_growth",
            validation_message=message,
            extra=diag_extra,
        )

        new_remainders: list[Interval] = []
        for r, rb in zip(remainders, residual_boxes):
            hull = Interval.hull(r, rb)
            new_remainders.append(hull.scale_about_mid(growth_factor, min_radius=validation_eps))
        remainders = new_remainders

    candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(candidate_poly, remainders)
        )
    return candidate, "failed", max_attempts, "Picard remainder validation did not converge"


def _validate_picard_target_remainder(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate_poly: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    max_attempts: int,
    validation_eps: float,
    h: float,
    target_remainder_radius: float,
    normal_eval_range_split: int | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    symbolic_remainder: bool = False,
    max_symbolic_remainders: int = 0,
) -> tuple[TMVector, str, int, str]:
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    target_remainders = [_symmetric_interval(target_remainder_radius, domain) for _ in candidate_poly]
    diag_extra = dict(diagnostics_context or {})
    diag_extra.setdefault("validation_mode", "target_remainder_normal_eval" if normal_eval_range_split is not None else "target_remainder")
    diag_extra.setdefault("target_remainder_radius", abs(float(target_remainder_radius)))
    if normal_eval_range_split is not None:
        diag_extra.setdefault("normal_eval_range_split", int(normal_eval_range_split))
    diag_extra.setdefault("target_remainder_width", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_remainder_width_sum", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_checked_width", _sum_interval_widths(target_remainders))
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)

    seed_remainders = [_combine_remainders(base_i.remainder, candidate_i.remainder) for base_i, candidate_i in zip(base_ext, candidate_poly)]
    candidate = TMVector(
        TaylorModel(
            m.polynomial,
            r,
            domain,
            order=order,
            truncation_range_split=m.truncation_range_split,
        )
        for m, r in zip(candidate_poly, target_remainders)
    )
    if not intervals_are_finite(seed_remainders):
        message = "non-finite initial remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=target_remainders,
            finite_residual=False,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message
    if not all(target.contains_interval(seed) for target, seed in zip(target_remainders, seed_remainders)):
        message = "initial or cutoff remainder exceeds target remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=target_remainders,
            finite_residual=True,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message

    for attempt in range(1, max_attempts + 1):
        if rhs_breakdown_callback is not None:
            callback_context = dict(diag_extra)
            if diag_mode is not None:
                callback_context["mode"] = diag_mode
            if diag_segment_index is not None:
                callback_context["segment_index"] = diag_segment_index
            callback_context["attempt_index"] = attempt
            callback_context["h"] = float(h)
            callback_context["order"] = int(order)
            try:
                rhs_breakdown_callback(candidate, order, attempt, callback_context)
            except Exception:
                pass
        try:
            residual_boxes = _picard_residual_boxes(
                ode_fn,
                base_ext,
                candidate,
                tau_index,
                order,
                u_tms,
                validation_eps=validation_eps,
                normal_eval_range_split=normal_eval_range_split,
            )
        except Exception as exc:
            message = f"validation exception: {exc}"
            extra = dict(diag_extra, subset_result=False, rejection_reason=message)
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=None,
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        finite_residual = intervals_are_finite(residual_boxes)
        if not finite_residual:
            message = "non-finite residual interval"
            extra = dict(diag_extra, subset_result=False, rejection_reason=message)
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_boxes,
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        subset_result = all(target.contains_interval(rb) for target, rb in zip(target_remainders, residual_boxes))
        message = "" if subset_result else "Picard residual not subset of target remainder"
        extra = dict(
            diag_extra,
            subset_result=subset_result,
            rejection_reason="" if subset_result else message,
            **(_interval_list_stats("normal_eval_range", residual_boxes) if normal_eval_range_split is not None else {}),
        )
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=attempt,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=residual_boxes,
            remainders=target_remainders,
            finite_residual=True,
            validation_status="validated" if subset_result else "failed",
            validation_message=message,
            extra=extra,
        )
        if subset_result:
            return candidate, "validated", attempt, ""

    return candidate, "failed", max_attempts, "Picard residual not subset of target remainder"



def _residual_interval_stats(prefix: str, boxes: Sequence[Interval] | None) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if boxes is None:
        return row
    names = ("x", "y")
    for i, iv in enumerate(boxes[:2]):
        name = names[i]
        lo = _interval_bound_value(iv.lo)
        hi = _interval_bound_value(iv.hi)
        width = _interval_width_value(iv)
        center = _float_or_none(iv.mid().detach().cpu())
        radius = _float_or_none(iv.radius().detach().cpu())
        row[f"{prefix}_lo_{name}"] = lo if lo is not None else ""
        row[f"{prefix}_hi_{name}"] = hi if hi is not None else ""
        row[f"{prefix}_width_{name}"] = width if width is not None else ""
        row[f"{prefix}_center_{name}"] = center if center is not None else ""
        row[f"{prefix}_radius_{name}"] = radius if radius is not None else ""
    return row


def _taylor_model_range_box_normal(
    model: TaylorModel,
    tau_index: int | None,
    *,
    normal_eval_range_split: int | None = None,
) -> Interval:
    poly_range = _poly_interval_normal(
        model.polynomial,
        model.domain,
        tau_index,
        normal_eval_range_split=normal_eval_range_split,
    )
    return poly_range + model.remainder


def _tmvector_range_box_normal(
    tmv: TMVector,
    tau_index: int | None = None,
    *,
    normal_eval_range_split: int | None = None,
) -> list[Interval]:
    return [
        _taylor_model_range_box_normal(
            model,
            tau_index,
            normal_eval_range_split=normal_eval_range_split,
        )
        for model in tmv
    ]


def _picard_residual_boxes(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    validation_eps: float,
    normal_eval_range_split: int | None = None,
) -> list[Interval]:
    domain = candidate.domain
    rhs = _call_ode(ode_fn, candidate, u_tms)
    residual_boxes: list[Interval] = []
    for base_i, cand_i, f_i in zip(base_ext, candidate, rhs):
        picard_i = base_i + f_i.integrate(tau_index)
        residual_i = picard_i - TaylorModel(
            cand_i.polynomial,
            Interval.zero(),
            domain,
            order=order,
            truncation_range_split=cand_i.truncation_range_split,
        )
        if normal_eval_range_split is None:
            residual_box = residual_i.range_box()
        else:
            residual_box = _taylor_model_range_box_normal(
                residual_i,
                tau_index,
                normal_eval_range_split=normal_eval_range_split,
            )
        residual_boxes.append(residual_box.inflate(validation_eps))
    return residual_boxes


def _interval_list_stats(prefix: str, boxes: Sequence[Interval] | None) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if boxes is None:
        return row
    names = ("x", "y")
    widths: list[float] = []
    for i, iv in enumerate(boxes[:2]):
        name = names[i] if i < len(names) else f"state_{i}"
        lo = _interval_bound_value(iv.lo)
        hi = _interval_bound_value(iv.hi)
        width = _interval_width_value(iv)
        center = _float_or_none(iv.mid().detach().cpu())
        radius = _float_or_none(iv.radius().detach().cpu())
        row[f"{prefix}_lo_{name}"] = lo if lo is not None else ""
        row[f"{prefix}_hi_{name}"] = hi if hi is not None else ""
        row[f"{prefix}_width_{name}"] = width if width is not None else ""
        row[f"{prefix}_center_{name}"] = center if center is not None else ""
        row[f"{prefix}_radius_{name}"] = radius if radius is not None else ""
        if width is not None:
            widths.append(width)
    if widths:
        row[f"{prefix}_width_sum"] = sum(widths)
    return row


def _picard_ctrunc_normal_image(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    cutoff_threshold: float | None,
) -> TMVector:
    domain = candidate.domain
    rhs = _call_ode(ode_fn, candidate, u_tms)
    models: list[TaylorModel] = []
    for base_i, f_i in zip(base_ext, rhs):
        picard_i = base_i + f_i.integrate(tau_index)
        kept, dropped = picard_i.polynomial.truncate(order)
        trunc_range = _poly_interval_normal(dropped, domain, tau_index)
        kept, cutoff_range = _cutoff_polynomial_normal(kept, domain, tau_index, cutoff_threshold)
        models.append(
            TaylorModel(
                kept,
                picard_i.remainder + trunc_range + cutoff_range,
                domain,
                order=order,
                truncation_range_split=picard_i.truncation_range_split,
            )
        )
    return TMVector(models)


def _picard_ctrunc_normal_partition_boxes(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    cutoff_threshold: float | None,
) -> dict[str, list[Interval]]:
    domain = candidate.domain
    rhs = _call_ode(ode_fn, candidate, u_tms)
    zero = Interval.zero(dtype=domain[0].lo.dtype, device=domain[0].lo.device) if domain else Interval.zero()
    dropped_terms: list[Interval] = []
    multiplication_remainders: list[Interval] = []
    integration_remainders: list[Interval] = []
    before_accumulation: list[Interval] = []
    after_integration: list[Interval] = []
    after_dropped_terms: list[Interval] = []
    after_cutoff: list[Interval] = []
    for base_i, f_i in zip(base_ext, rhs):
        integrated = f_i.integrate(tau_index)
        picard_i = base_i + integrated
        kept, dropped = picard_i.polynomial.truncate(order)
        trunc_range = _poly_interval_normal(dropped, domain, tau_index)
        _kept_after_cutoff, cutoff_range = _cutoff_polynomial_normal(kept, domain, tau_index, cutoff_threshold)
        dropped_total = trunc_range + cutoff_range
        before = base_i.remainder
        after_int = picard_i.remainder
        after_drop = after_int + trunc_range
        after_cut = after_drop + cutoff_range
        dropped_terms.append(dropped_total)
        multiplication_remainders.append(integrated.remainder)
        integration_remainders.append(after_int)
        before_accumulation.append(before if before is not None else zero)
        after_integration.append(after_int)
        after_dropped_terms.append(after_drop)
        after_cutoff.append(after_cut)
    return {
        "raw_remainder_dropped_terms_range": dropped_terms,
        "raw_remainder_multiplication_remainder": multiplication_remainders,
        "raw_remainder_integration_remainder": integration_remainders,
        "raw_remainder_before_accumulation": before_accumulation,
        "raw_remainder_after_integration": after_integration,
        "raw_remainder_after_dropped_terms": after_dropped_terms,
        "raw_remainder_after_cutoff": after_cutoff,
        "raw_remainder_before_poly_diff": after_cutoff,
    }


class _FlowstarRawRemainderTraceTM:
    """Internal Expression::evaluate_remainder-style tracer for compat mode."""

    def __init__(
        self,
        tm: TaylorModel,
        replay_remainder: Interval | None = None,
        *,
        order: int,
        tau_index: int,
        cutoff_threshold: float | None,
    ):
        self.tm = tm
        self.replay_remainder = tm.remainder if replay_remainder is None else replay_remainder
        self._order = int(order)
        self._tau_index = int(tau_index)
        self._cutoff_threshold = cutoff_threshold

    @property
    def domain(self) -> list[Interval]:
        return list(self.tm.domain)

    @property
    def polynomial(self) -> Polynomial:
        return self.tm.polynomial

    @property
    def remainder(self) -> Interval:
        return self.replay_remainder

    @property
    def order(self) -> int:
        return self._order

    @property
    def truncation_range_split(self) -> int | None:
        return self.tm.truncation_range_split

    def _wrap(self, tm: TaylorModel, replay_remainder: Interval) -> "_FlowstarRawRemainderTraceTM":
        return _FlowstarRawRemainderTraceTM(
            tm,
            replay_remainder,
            order=self._order,
            tau_index=self._tau_index,
            cutoff_threshold=self._cutoff_threshold,
        )

    def _coerce(self, other: Any) -> "_FlowstarRawRemainderTraceTM":
        if isinstance(other, _FlowstarRawRemainderTraceTM):
            return other
        if isinstance(other, TaylorModel):
            tm = TaylorModel(
                other.polynomial,
                other.remainder,
                list(other.domain),
                order=self._order,
                truncation_range_split=other.truncation_range_split,
            )
            return self._wrap(tm, tm.remainder)
        const = TaylorModel.constant(other, self.domain, order=self._order, truncation_range_split=self.truncation_range_split)
        return self._wrap(const, const.remainder)

    def __add__(self, other: Any) -> "_FlowstarRawRemainderTraceTM":
        other = self._coerce(other)
        tm = TaylorModel(
            self.polynomial + other.polynomial,
            self.replay_remainder + other.replay_remainder,
            self.domain,
            order=self._order,
            truncation_range_split=self.truncation_range_split,
        )
        return self._wrap(tm, tm.remainder)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "_FlowstarRawRemainderTraceTM":
        other = self._coerce(other)
        tm = TaylorModel(
            self.polynomial - other.polynomial,
            self.replay_remainder - other.replay_remainder,
            self.domain,
            order=self._order,
            truncation_range_split=self.truncation_range_split,
        )
        return self._wrap(tm, tm.remainder)

    def __rsub__(self, other: Any) -> "_FlowstarRawRemainderTraceTM":
        return self._coerce(other) - self

    def __neg__(self) -> "_FlowstarRawRemainderTraceTM":
        tm = TaylorModel(
            -self.polynomial,
            -self.replay_remainder,
            self.domain,
            order=self._order,
            truncation_range_split=self.truncation_range_split,
        )
        return self._wrap(tm, tm.remainder)

    def __mul__(self, other: Any) -> "_FlowstarRawRemainderTraceTM":
        other = self._coerce(other)
        poly_product = self.polynomial * other.polynomial
        kept, dropped = poly_product.truncate(self._order)
        int_trunc = _poly_interval_normal(dropped, self.domain, self._tau_index)
        kept, cutoff_range = _cutoff_polynomial_normal(
            kept,
            self.domain,
            self._tau_index,
            self._cutoff_threshold,
        )
        int_trunc = int_trunc + cutoff_range
        left_poly_range = _poly_interval_normal(self.polynomial, self.domain, self._tau_index)
        right_poly_range = _poly_interval_normal(other.polynomial, other.domain, self._tau_index)
        replay = (
            left_poly_range * other.replay_remainder
            + right_poly_range * self.replay_remainder
            + self.replay_remainder * other.replay_remainder
            + int_trunc
        )
        tm = TaylorModel(
            kept,
            replay,
            self.domain,
            order=self._order,
            truncation_range_split=self.truncation_range_split,
        )
        return self._wrap(tm, replay)

    __rmul__ = __mul__

    def pow_int(self, exponent: int) -> "_FlowstarRawRemainderTraceTM":
        if exponent < 0:
            raise ValueError("flowstar raw remainder compat only supports nonnegative integer powers")
        out = self._coerce(1.0)
        for _ in range(int(exponent)):
            out = out * self
        return out


class _FlowstarRawRemainderTraceVector:
    def __init__(self, models: Iterable[_FlowstarRawRemainderTraceTM]):
        self.models = list(models)

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index: int) -> _FlowstarRawRemainderTraceTM:
        return self.models[index]

    @property
    def domain(self) -> list[Interval]:
        return self.models[0].domain if self.models else []

    @property
    def n_vars(self) -> int:
        return len(self.domain)


def _coerce_flowstar_trace_output(
    value: Any,
    exemplar: _FlowstarRawRemainderTraceTM,
) -> _FlowstarRawRemainderTraceTM:
    if isinstance(value, _FlowstarRawRemainderTraceTM):
        return value
    return exemplar._coerce(value)


def _flowstar_raw_remainder_compat_boxes(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    cutoff_threshold: float | None,
    validation_eps: float,
) -> dict[str, list[Interval]]:
    rhs_order = max(int(order) - 1, 0)
    trace_models = [
        _FlowstarRawRemainderTraceTM(
            TaylorModel(
                model.polynomial,
                model.remainder,
                list(model.domain),
                order=rhs_order,
                truncation_range_split=model.truncation_range_split,
            ),
            model.remainder,
            order=rhs_order,
            tau_index=tau_index,
            cutoff_threshold=cutoff_threshold,
        )
        for model in candidate
    ]
    trace_state = _FlowstarRawRemainderTraceVector(trace_models)
    trace_controls = None
    if u_tms is not None:
        trace_controls = _FlowstarRawRemainderTraceVector(
            _FlowstarRawRemainderTraceTM(
                TaylorModel(
                    model.polynomial,
                    model.remainder,
                    list(model.domain),
                    order=rhs_order,
                    truncation_range_split=model.truncation_range_split,
                ),
                model.remainder,
                order=rhs_order,
                tau_index=tau_index,
                cutoff_threshold=cutoff_threshold,
            )
            for model in u_tms
        )
    try:
        out = ode_fn(trace_state, trace_controls)
    except TypeError:
        out = ode_fn(trace_state)
    raw_outputs = list(out) if not isinstance(out, TMVector) else list(out)
    if len(raw_outputs) != len(candidate):
        raise ValueError("flowstar raw remainder compat RHS dimension mismatch")
    exemplar = trace_models[0]
    rhs = [_coerce_flowstar_trace_output(value, exemplar) for value in raw_outputs]
    time_step = candidate.domain[tau_index]
    before_x0 = [(model.replay_remainder * time_step).inflate(validation_eps) for model in rhs]
    after_x0 = [(base_i.remainder + rem).inflate(validation_eps) for base_i, rem in zip(base_ext, before_x0)]
    return {
        "flowstar_raw_remainder_compat_rhs_remainder": [model.replay_remainder for model in rhs],
        "accumulated_remainder_before_x0_add": before_x0,
        "accumulated_remainder_after_x0_add": after_x0,
        "flowstar_raw_remainder_compat_raw_ctrunc_residual": after_x0,
    }


def _flowstar_raw_remainder_compat_check(
    target_remainders: Sequence[Interval],
    accumulated_before_x0_add: Sequence[Interval],
    base_remainders: Sequence[Interval],
    poly_diff_ranges: Sequence[Interval],
    *,
    validation_eps: float = 0.0,
) -> tuple[list[Interval], list[bool]]:
    check_remainders = [
        (base + before + diff).inflate(validation_eps)
        for target, before, base, diff in zip(target_remainders, accumulated_before_x0_add, base_remainders, poly_diff_ranges)
    ]
    subset = [target.contains_interval(rem) for target, rem in zip(target_remainders, check_remainders)]
    return check_remainders, subset


def _validate_picard_target_remainder_flowstar_ctrunc(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate_poly: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    max_attempts: int,
    validation_eps: float,
    h: float,
    target_remainder_radius: float,
    cutoff_threshold: float | None,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    symbolic_remainder: bool = False,
    max_symbolic_remainders: int = 0,
    raw_remainder_mode: str = "",
) -> tuple[TMVector, str, int, str]:
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    if raw_remainder_mode not in {"", "flowstar_compat"}:
        raise ValueError("raw_remainder_mode must be empty or 'flowstar_compat'")
    compat_mode = raw_remainder_mode == "flowstar_compat"
    target_remainders = [_symmetric_interval(target_remainder_radius, domain) for _ in candidate_poly]
    diag_extra = dict(diagnostics_context or {})
    diag_extra.setdefault("validation_mode", "flowstar_raw_remainder_compat" if compat_mode else "target_remainder_flowstar_ctrunc")
    diag_extra.setdefault("raw_remainder_mode", "flowstar_compat" if compat_mode else "")
    diag_extra.setdefault("flowstar_raw_remainder_compat_enabled", compat_mode)
    diag_extra.setdefault("target_remainder_radius", abs(float(target_remainder_radius)))
    diag_extra.setdefault("target_remainder_width", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_remainder_width_sum", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_checked_width", _sum_interval_widths(target_remainders))
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)

    seed_remainders = [_combine_remainders(base_i.remainder, candidate_i.remainder) for base_i, candidate_i in zip(base_ext, candidate_poly)]
    candidate = TMVector(
        TaylorModel(
            m.polynomial,
            r,
            domain,
            order=order,
            truncation_range_split=m.truncation_range_split,
        )
        for m, r in zip(candidate_poly, target_remainders)
    )
    if not intervals_are_finite(seed_remainders):
        message = "non-finite initial remainder"
        extra = dict(diag_extra, subset_result=False, subset_tmp_remainder=False, subset_ordinary_residual=False, rejection_reason=message)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=target_remainders,
            finite_residual=False,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message
    if not all(target.contains_interval(seed) for target, seed in zip(target_remainders, seed_remainders)):
        message = "initial or cutoff remainder exceeds target remainder"
        extra = dict(diag_extra, subset_result=False, subset_tmp_remainder=False, subset_ordinary_residual=False, rejection_reason=message)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=target_remainders,
            finite_residual=True,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message

    for attempt in range(1, max_attempts + 1):
        if rhs_breakdown_callback is not None:
            callback_context = dict(diag_extra)
            if diag_mode is not None:
                callback_context["mode"] = diag_mode
            if diag_segment_index is not None:
                callback_context["segment_index"] = diag_segment_index
            callback_context["attempt_index"] = attempt
            callback_context["h"] = float(h)
            callback_context["order"] = int(order)
            try:
                rhs_breakdown_callback(candidate, order, attempt, callback_context)
            except Exception:
                pass
        try:
            ordinary_residual = _picard_residual_boxes(
                ode_fn,
                base_ext,
                candidate,
                tau_index,
                order,
                u_tms,
                validation_eps=validation_eps,
            )
            tmp = _picard_ctrunc_normal_image(
                ode_fn,
                base_ext,
                candidate,
                tau_index,
                order,
                u_tms,
                cutoff_threshold=cutoff_threshold,
            )
            raw_ctrunc_remainders: list[Interval] = []
            raw_ctrunc_polynomial_ranges = _polynomial_range_boxes(tmp)
            raw_partition_boxes = _picard_ctrunc_normal_partition_boxes(
                ode_fn,
                base_ext,
                candidate,
                tau_index,
                order,
                u_tms,
                cutoff_threshold=cutoff_threshold,
            )
            poly_diff_ranges: list[Interval] = []
            tmp_remainders: list[Interval] = []
            for tmp_i, cand_i in zip(tmp, candidate):
                raw_remainder = tmp_i.remainder.inflate(validation_eps)
                raw_ctrunc_remainders.append(raw_remainder)
                diff_poly = tmp_i.polynomial - cand_i.polynomial
                diff_range = _poly_interval_normal(diff_poly, domain, tau_index).inflate(validation_eps)
                poly_diff_ranges.append(diff_range)
                tmp_remainders.append((raw_remainder + diff_range).inflate(validation_eps))
            raw_partition_boxes["raw_remainder_after_poly_diff"] = tmp_remainders
            compat_boxes: dict[str, list[Interval]] = {}
            compat_remainders: list[Interval] | None = None
            compat_subset_by_dim: list[bool] = []
            if compat_mode:
                compat_boxes = _flowstar_raw_remainder_compat_boxes(
                    ode_fn,
                    base_ext,
                    candidate,
                    tau_index,
                    order,
                    u_tms,
                    cutoff_threshold=cutoff_threshold,
                    validation_eps=validation_eps,
                )
                compat_remainders, compat_subset_by_dim = _flowstar_raw_remainder_compat_check(
                    target_remainders,
                    compat_boxes["accumulated_remainder_before_x0_add"],
                    [base_i.remainder for base_i in base_ext],
                    poly_diff_ranges,
                    validation_eps=validation_eps,
                )
        except Exception as exc:
            message = f"validation exception: {exc}"
            extra = dict(diag_extra, subset_result=False, subset_tmp_remainder=False, subset_ordinary_residual=False, rejection_reason=message)
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=None,
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        finite_residual = intervals_are_finite(ordinary_residual) and intervals_are_finite(tmp_remainders) and intervals_are_finite(poly_diff_ranges)
        if compat_mode:
            finite_residual = finite_residual and compat_remainders is not None and intervals_are_finite(compat_remainders)
        subset_ordinary = bool(finite_residual and all(target.contains_interval(rb) for target, rb in zip(target_remainders, ordinary_residual)))
        subset_tmp = bool(finite_residual and all(target.contains_interval(rem) for target, rem in zip(target_remainders, tmp_remainders)))
        subset_check = bool(subset_tmp if not compat_mode else all(compat_subset_by_dim))
        validation_decision_difference = bool(subset_check != subset_ordinary)
        message = "" if subset_check else (
            "Flowstar raw remainder compat residual not subset of target remainder"
            if compat_mode
            else "Flowstar ctrunc tmp remainder not subset of target remainder"
        )
        checked_remainders = compat_remainders if compat_mode and compat_remainders is not None else tmp_remainders
        validated_candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(candidate_poly, checked_remainders)
        )
        compat_raw = compat_boxes.get("flowstar_raw_remainder_compat_raw_ctrunc_residual") if compat_mode else None
        raw_stats = compat_raw if compat_raw is not None else raw_ctrunc_remainders
        if compat_mode:
            raw_partition_boxes["accumulated_remainder_before_x0_add"] = compat_boxes.get("accumulated_remainder_before_x0_add", [])
            raw_partition_boxes["accumulated_remainder_after_x0_add"] = compat_boxes.get("accumulated_remainder_after_x0_add", [])
        extra = {
            **diag_extra,
            **_interval_list_stats("tmp_remainder", tmp_remainders),
            **_interval_list_stats("flowstar_raw_remainder_compat_check_remainder", compat_remainders),
            **_interval_list_stats("current_raw_ctrunc_residual", raw_ctrunc_remainders),
            **_interval_list_stats("raw_ctrunc_residual", raw_stats),
            **_interval_list_stats("raw_ctrunc_remainder", raw_stats),
            **_interval_list_stats("raw_ctrunc_polynomial_range", raw_ctrunc_polynomial_ranges),
            **_interval_list_stats("target_remainder_before_ctrunc", target_remainders),
            **_interval_list_stats("flowstar_raw_remainder_compat_rhs_remainder", compat_boxes.get("flowstar_raw_remainder_compat_rhs_remainder") if compat_mode else None),
            **_interval_list_stats("flowstar_raw_remainder_compat_raw_ctrunc_residual", compat_raw),
            **_interval_list_stats("accumulated_remainder_before_x0_add", raw_partition_boxes.get("accumulated_remainder_before_x0_add")),
            **_interval_list_stats("accumulated_remainder_after_x0_add", raw_partition_boxes.get("accumulated_remainder_after_x0_add")),
            **_interval_list_stats("raw_remainder_dropped_terms_range", raw_partition_boxes.get("raw_remainder_dropped_terms_range")),
            **_interval_list_stats("raw_remainder_multiplication_remainder", raw_partition_boxes.get("raw_remainder_multiplication_remainder")),
            **_interval_list_stats("raw_remainder_integration_remainder", raw_partition_boxes.get("raw_remainder_integration_remainder")),
            **_interval_list_stats("raw_remainder_before_accumulation", raw_partition_boxes.get("raw_remainder_before_accumulation")),
            **_interval_list_stats("raw_remainder_after_integration", raw_partition_boxes.get("raw_remainder_after_integration")),
            **_interval_list_stats("raw_remainder_after_dropped_terms", raw_partition_boxes.get("raw_remainder_after_dropped_terms")),
            **_interval_list_stats("raw_remainder_after_cutoff", raw_partition_boxes.get("raw_remainder_after_cutoff")),
            **_interval_list_stats("raw_remainder_before_poly_diff", raw_partition_boxes.get("raw_remainder_before_poly_diff")),
            **_interval_list_stats("raw_remainder_after_poly_diff", raw_partition_boxes.get("raw_remainder_after_poly_diff")),
            **_interval_list_stats("poly_diff_range", poly_diff_ranges),
            **_interval_list_stats("ordinary_residual_range", ordinary_residual),
            **_interval_list_stats("normal_eval_range", poly_diff_ranges),
            "raw_ctrunc_residual_source_object": (
                "flowstar_raw_remainder_compat Expression::evaluate_remainder replay before poly_diff_range"
                if compat_mode
                else "_picard_ctrunc_normal_image returned TaylorModel.remainder before poly_diff_range"
            ),
            "raw_ctrunc_residual_domain_semantics": "physical_remainder_interval_over_full_step_tau_domain_before_cutoff_polyDiff",
            "raw_ctrunc_residual_includes_target_remainder": False,
            "raw_ctrunc_residual_includes_ordinary_remainder": False,
            "raw_ctrunc_residual_includes_cutoff_poly_diff": False,
            "raw_ctrunc_residual_added_component": "none_before_cutoff_polyDiff",
            "raw_ctrunc_residual_notes": (
                "compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set"
                if compat_mode
                else "raw ctrunc remainder is recorded before adding poly_diff_range; target remainder is only the containment set"
            ),
            "raw_remainder_range_enclosure_method": (
                "Flowstar Expression::evaluate_remainder-style replay over polynomial ranges and truncation intervals"
                if compat_mode
                else "Polynomial.truncate plus _poly_interval_normal on dropped terms; cutoff range from _cutoff_polynomial_normal"
            ),
            "raw_remainder_normal_domain_scaling": "none_after_Picard_call; intervals are over the physical full-step tau domain",
            "raw_remainder_partition_missing_reason": (
                "compat mode reconstructs the replayed raw remainder; Flow* source-level intermediate_ranges remain the exact reference"
                if compat_mode
                else "PyTorch exposes diagnostic reconstruction of dropped/integration/multiplication partitions; Flow* source-level intermediate_ranges partition remains the reference for exact attribution"
            ),
            "ordinary_remainder_missing_reason": "PyTorch ordinary_residual_range is exposed separately; it is not included in raw_ctrunc_residual",
            "subset_result": subset_check,
            "subset_tmp_remainder": subset_tmp,
            "subset_flowstar_raw_remainder_compat": subset_check if compat_mode else "",
            "subset_flowstar_raw_remainder_compat_x": compat_subset_by_dim[0] if compat_mode and len(compat_subset_by_dim) > 0 else "",
            "subset_flowstar_raw_remainder_compat_y": compat_subset_by_dim[1] if compat_mode and len(compat_subset_by_dim) > 1 else "",
            "subset_ordinary_residual": subset_ordinary,
            "validation_decision_difference": validation_decision_difference,
            "rejection_reason": "" if subset_check else message,
        }
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=attempt,
            h=h,
            order=order,
            candidate=validated_candidate,
            tau_index=tau_index,
            residual_boxes=ordinary_residual,
            remainders=checked_remainders,
            finite_residual=finite_residual,
            validation_status="validated" if subset_check else "failed",
            validation_message=message,
            extra=extra,
        )
        if subset_check:
            return validated_candidate, "validated", attempt, ""
        return validated_candidate, "failed", attempt, message

    return candidate, "failed", max_attempts, "Flowstar ctrunc tmp remainder not subset of target remainder"


def _shift_candidate_constants(candidate_poly: TMVector, shifts: Sequence[torch.Tensor]) -> TMVector:
    shifted: list[TaylorModel] = []
    for model, shift in zip(candidate_poly, shifts):
        terms = {exp: coef.clone() for exp, coef in model.polynomial.terms.items()}
        zero_exp = (0,) * model.polynomial.n_vars
        shift_t = shift.to(dtype=model.polynomial.dtype, device=model.polynomial.device)
        terms[zero_exp] = terms.get(zero_exp, torch.zeros_like(shift_t)) + shift_t
        shifted.append(
            TaylorModel(
                Polynomial(terms, model.polynomial.n_vars),
                model.remainder,
                list(model.domain),
                order=model.order,
                truncation_range_split=model.truncation_range_split,
            )
        )
    return TMVector(shifted)


def _zero_shift_like(model: TaylorModel) -> torch.Tensor:
    if model.polynomial.terms:
        first = next(iter(model.polynomial.terms.values()))
        return torch.zeros((), dtype=first.dtype, device=first.device)
    if model.domain:
        return torch.zeros((), dtype=model.domain[0].dtype, device=model.domain[0].device)
    return torch.zeros((), dtype=torch.float64)


def _validate_picard_target_remainder_centered(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate_poly: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    max_attempts: int,
    validation_eps: float,
    h: float,
    target_remainder_radius: float,
    center_correction_width_factor: float = 1.05,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    symbolic_remainder: bool = False,
    max_symbolic_remainders: int = 0,
) -> tuple[TMVector, str, int, str]:
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    target_remainders = [_symmetric_interval(target_remainder_radius, domain) for _ in candidate_poly]
    diag_extra = dict(diagnostics_context or {})
    diag_extra.setdefault("validation_mode", "target_remainder_centered")
    diag_extra.setdefault("target_remainder_radius", abs(float(target_remainder_radius)))
    diag_extra.setdefault("target_remainder_width", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_remainder_width_sum", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_checked_width", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("center_correction_width_factor", float(center_correction_width_factor))
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)

    seed_remainders = [_combine_remainders(base_i.remainder, candidate_i.remainder) for base_i, candidate_i in zip(base_ext, candidate_poly)]
    candidate = TMVector(
        TaylorModel(
            m.polynomial,
            r,
            domain,
            order=order,
            truncation_range_split=m.truncation_range_split,
        )
        for m, r in zip(candidate_poly, target_remainders)
    )
    if not intervals_are_finite(seed_remainders):
        message = "non-finite initial remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message, center_correction_applied=False, subset_after_correction=False)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=target_remainders,
            finite_residual=False,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message
    if not all(target.contains_interval(seed) for target, seed in zip(target_remainders, seed_remainders)):
        message = "initial or cutoff remainder exceeds target remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message, center_correction_applied=False, subset_after_correction=False)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=target_remainders,
            finite_residual=True,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message

    for attempt in range(1, max_attempts + 1):
        if rhs_breakdown_callback is not None:
            callback_context = dict(diag_extra)
            if diag_mode is not None:
                callback_context["mode"] = diag_mode
            if diag_segment_index is not None:
                callback_context["segment_index"] = diag_segment_index
            callback_context["attempt_index"] = attempt
            callback_context["h"] = float(h)
            callback_context["order"] = int(order)
            try:
                rhs_breakdown_callback(candidate, order, attempt, callback_context)
            except Exception:
                pass
        try:
            residual_before = _picard_residual_boxes(
                ode_fn,
                base_ext,
                candidate,
                tau_index,
                order,
                u_tms,
                validation_eps=validation_eps,
            )
        except Exception as exc:
            message = f"validation exception: {exc}"
            extra = dict(diag_extra, subset_result=False, rejection_reason=message, center_correction_applied=False, subset_after_correction=False)
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=None,
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        finite_residual = intervals_are_finite(residual_before)
        before_stats = _residual_interval_stats("residual_before", residual_before)
        if not finite_residual:
            message = "non-finite residual interval"
            extra = {**diag_extra, **before_stats, "subset_result": False, "rejection_reason": message, "center_correction_applied": False, "subset_after_correction": False}
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_before,
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        subset_before = all(target.contains_interval(rb) for target, rb in zip(target_remainders, residual_before))
        if subset_before:
            extra = {
                **diag_extra,
                **before_stats,
                **_residual_interval_stats("residual_after", residual_before),
                "subset_result": True,
                "rejection_reason": "",
                "center_correction_applied": False,
                "correction_value_x": 0.0,
                "correction_value_y": 0.0,
                "subset_after_correction": True,
            }
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_before,
                remainders=target_remainders,
                finite_residual=True,
                validation_status="validated",
                validation_message="",
                extra=extra,
            )
            return candidate, "validated", attempt, ""

        target_widths = [_interval_width_value(target) for target in target_remainders]
        residual_widths = [_interval_width_value(rb) for rb in residual_before]
        misses = [not target.contains_interval(rb) for target, rb in zip(target_remainders, residual_before)]
        shift_eligible = []
        for miss, rb_width, target_width in zip(misses, residual_widths, target_widths):
            shift_eligible.append(bool(miss and rb_width is not None and target_width is not None and rb_width <= target_width * float(center_correction_width_factor)))
        if any(miss and not eligible for miss, eligible in zip(misses, shift_eligible)):
            message = "Picard residual not subset of target remainder"
            extra = {
                **diag_extra,
                **before_stats,
                "subset_result": False,
                "rejection_reason": message,
                "center_correction_applied": False,
                "correction_value_x": 0.0,
                "correction_value_y": 0.0,
                "subset_after_correction": False,
            }
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=residual_before,
                remainders=target_remainders,
                finite_residual=True,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        shifts = []
        for model, rb, eligible in zip(candidate_poly, residual_before, shift_eligible):
            shifts.append(rb.mid() if eligible else _zero_shift_like(model))
        corrected_poly = _shift_candidate_constants(candidate_poly, shifts)
        corrected_candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(corrected_poly, target_remainders)
        )
        try:
            residual_after = _picard_residual_boxes(
                ode_fn,
                base_ext,
                corrected_candidate,
                tau_index,
                order,
                u_tms,
                validation_eps=validation_eps,
            )
        except Exception as exc:
            message = f"validation exception after center correction: {exc}"
            extra = dict(
                diag_extra,
                before_stats,
                subset_result=False,
                rejection_reason=message,
                center_correction_applied=True,
                correction_value_x=_float_value(shifts[0].detach().cpu()) if len(shifts) > 0 else "",
                correction_value_y=_float_value(shifts[1].detach().cpu()) if len(shifts) > 1 else "",
                subset_after_correction=False,
            )
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=corrected_candidate,
                tau_index=tau_index,
                residual_boxes=None,
                remainders=target_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return corrected_candidate, "failed", attempt, message

        finite_after = intervals_are_finite(residual_after)
        subset_after = bool(finite_after and all(target.contains_interval(rb) for target, rb in zip(target_remainders, residual_after)))
        message = "" if subset_after else "Picard residual not subset of target remainder after center correction"
        extra = {
            **diag_extra,
            **before_stats,
            **_residual_interval_stats("residual_after", residual_after),
            "subset_result": subset_after,
            "rejection_reason": "" if subset_after else message,
            "center_correction_applied": True,
            "correction_value_x": _float_value(shifts[0].detach().cpu()) if len(shifts) > 0 else "",
            "correction_value_y": _float_value(shifts[1].detach().cpu()) if len(shifts) > 1 else "",
            "subset_after_correction": subset_after,
        }
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=attempt,
            h=h,
            order=order,
            candidate=corrected_candidate,
            tau_index=tau_index,
            residual_boxes=residual_after,
            remainders=target_remainders,
            finite_residual=finite_after,
            validation_status="validated" if subset_after else "failed",
            validation_message=message,
            extra=extra,
        )
        if subset_after:
            return corrected_candidate, "validated", attempt, ""
        return corrected_candidate, "failed", attempt, message

    return candidate, "failed", max_attempts, "Picard residual not subset of target remainder"


def _validate_picard_target_remainder_refined(
    ode_fn: ODEFunction,
    base_ext: TMVector,
    candidate_poly: TMVector,
    tau_index: int,
    order: int,
    u_tms: TMVector | None,
    *,
    max_attempts: int,
    validation_eps: float,
    h: float,
    target_remainder_radius: float,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    symbolic_remainder: bool = False,
    max_symbolic_remainders: int = 0,
) -> tuple[TMVector, str, int, str]:
    """Conservative target validation with remainder-only Picard refinement."""
    domain = candidate_poly.domain
    if len(base_ext) != len(candidate_poly):
        raise ValueError("base and candidate dimensions differ")
    target_remainders = [_symmetric_interval(target_remainder_radius, domain) for _ in candidate_poly]
    diag_extra = dict(diagnostics_context or {})
    diag_extra.setdefault("validation_mode", "target_remainder_refined")
    diag_extra.setdefault("target_remainder_radius", abs(float(target_remainder_radius)))
    diag_extra.setdefault("target_remainder_width", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_remainder_width_sum", _sum_interval_widths(target_remainders))
    diag_extra.setdefault("target_checked_width", _sum_interval_widths(target_remainders))
    if symbolic_remainder:
        diag_extra.setdefault("symbolic_remainder", True)
        diag_extra.setdefault("queue_size", int(max_symbolic_remainders))
    diag_mode = diag_extra.pop("mode", diagnostics_mode)
    diag_segment_index = diag_extra.pop("segment_index", diagnostics_segment_index)

    seed_remainders = [
        _combine_remainders(base_i.remainder, candidate_i.remainder).inflate(validation_eps)
        for base_i, candidate_i in zip(base_ext, candidate_poly)
    ]
    current_remainders = seed_remainders
    candidate = TMVector(
        TaylorModel(
            m.polynomial,
            r,
            domain,
            order=order,
            truncation_range_split=m.truncation_range_split,
        )
        for m, r in zip(candidate_poly, current_remainders)
    )
    if not intervals_are_finite(current_remainders):
        message = "non-finite initial remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message, refinement_pass=0)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=current_remainders,
            finite_residual=False,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message
    if not all(target.contains_interval(seed) for target, seed in zip(target_remainders, current_remainders)):
        message = "initial or cutoff remainder exceeds target remainder"
        extra = dict(diag_extra, subset_result=False, rejection_reason=message, refinement_pass=0)
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=0,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=None,
            remainders=current_remainders,
            finite_residual=True,
            validation_status="failed",
            validation_message=message,
            extra=extra,
        )
        return candidate, "failed", 0, message

    for attempt in range(1, max_attempts + 1):
        candidate = TMVector(
            TaylorModel(
                m.polynomial,
                r,
                domain,
                order=order,
                truncation_range_split=m.truncation_range_split,
            )
            for m, r in zip(candidate_poly, current_remainders)
        )
        if rhs_breakdown_callback is not None:
            callback_context = dict(diag_extra)
            if diag_mode is not None:
                callback_context["mode"] = diag_mode
            if diag_segment_index is not None:
                callback_context["segment_index"] = diag_segment_index
            callback_context["attempt_index"] = attempt
            callback_context["h"] = float(h)
            callback_context["order"] = int(order)
            callback_context["refinement_pass"] = attempt
            try:
                rhs_breakdown_callback(candidate, order, attempt, callback_context)
            except Exception:
                pass
        try:
            rhs = _call_ode(ode_fn, candidate, u_tms)
            residual_boxes: list[Interval] = []
            for base_i, cand_i, f_i in zip(base_ext, candidate, rhs):
                picard_i = base_i + f_i.integrate(tau_index)
                residual_i = picard_i - TaylorModel(
                    cand_i.polynomial,
                    Interval.zero(),
                    domain,
                    order=order,
                    truncation_range_split=cand_i.truncation_range_split,
                )
                residual_boxes.append(residual_i.range_box().inflate(validation_eps))
        except Exception as exc:
            message = f"validation exception: {exc}"
            extra = dict(diag_extra, subset_result=False, rejection_reason=message, refinement_pass=attempt)
            _append_validation_diagnostic(
                diagnostics,
                mode=diag_mode,
                segment_index=diag_segment_index,
                attempt_index=attempt,
                h=h,
                order=order,
                candidate=candidate,
                tau_index=tau_index,
                residual_boxes=None,
                remainders=current_remainders,
                finite_residual=False,
                validation_status="failed",
                validation_message=message,
                extra=extra,
            )
            return candidate, "failed", attempt, message

        finite_residual = intervals_are_finite(residual_boxes)
        residual_inside_target = all(target.contains_interval(rb) for target, rb in zip(target_remainders, residual_boxes))
        residual_inside_current = all(current.contains_interval(rb) for current, rb in zip(current_remainders, residual_boxes))
        if not finite_residual:
            message = "non-finite residual interval"
        elif residual_inside_current:
            message = ""
        elif not residual_inside_target:
            message = "Picard residual not subset of target remainder"
        else:
            message = "remainder-only refinement continuing"
        extra = dict(
            diag_extra,
            subset_result=bool(finite_residual and residual_inside_target),
            rejection_reason="" if finite_residual and residual_inside_current else message,
            refinement_pass=attempt,
            residual_subset_current=bool(finite_residual and residual_inside_current),
        )
        _append_validation_diagnostic(
            diagnostics,
            mode=diag_mode,
            segment_index=diag_segment_index,
            attempt_index=attempt,
            h=h,
            order=order,
            candidate=candidate,
            tau_index=tau_index,
            residual_boxes=residual_boxes,
            remainders=current_remainders,
            finite_residual=finite_residual,
            validation_status="validated" if finite_residual and residual_inside_current else "failed",
            validation_message=message,
            extra=extra,
        )
        if not finite_residual:
            return candidate, "failed", attempt, message
        if residual_inside_current:
            return candidate, "validated", attempt, ""
        if not residual_inside_target:
            return candidate, "failed", attempt, message

        next_remainders = [Interval.hull(seed, residual).inflate(validation_eps) for seed, residual in zip(seed_remainders, residual_boxes)]
        if not all(target.contains_interval(next_r) for target, next_r in zip(target_remainders, next_remainders)):
            message = "refined remainder exceeds target remainder"
            return candidate, "failed", attempt, message
        current_remainders = next_remainders

    return candidate, "failed", max_attempts, "remainder-only target refinement did not converge"


def _flowpipe_step_from_tm_hybrid_dense(
    ode_fn: ODEFunction,
    x0_tm: TMVector,
    h: float,
    order: int,
    *,
    max_validation_attempts: int | None,
    validation_eps: float,
    validation_mode: str,
    target_remainder_radius: float,
    cutoff_threshold: float | None,
    diagnostics: list[dict[str, Any]] | None,
    diagnostics_mode: str | None,
    diagnostics_segment_index: int | None,
    diagnostics_context: Mapping[str, Any] | None,
    candidate_order: int | None,
    dense_device: torch.device | str,
    dense_dtype: torch.dtype,
    u_box: Sequence[Any] | None,
    affine_u: dict[str, Any] | None,
    symbolic_remainder: bool,
    selective_high_degree_terms_top_k: int | None,
    normal_eval_range_split: int | None,
    dense_range_policy: Any | None,
) -> FlowpipeSegment:
    """Dense Picard/validation with one sparse bridge at each segment boundary."""
    from .batched_dense_tm import (
        BatchedTaylorModel,
        DenseRangePolicy,
        DenseExecutionCounters,
        dense_picard_validate_step,
        dense_to_sparse_tmvector,
        sparse_tmvector_to_dense,
    )

    if dense_dtype != torch.float64:
        raise ValueError("the correctness dense flowpipe lane requires torch.float64")
    if u_box is not None or affine_u is not None:
        raise NotImplementedError("hybrid_dense_core currently supports uncontrolled polynomial plants")
    if symbolic_remainder:
        raise NotImplementedError("symbolic remainder carry is not yet part of hybrid_dense_core")
    if int(selective_high_degree_terms_top_k or 0) > 0:
        raise NotImplementedError("selective high-degree output truncation is not implemented for the dense core")
    if candidate_order is not None and int(candidate_order) != int(order):
        raise ValueError("hybrid_dense_core requires candidate_order == requested order")
    if normal_eval_range_split not in {None, 0, 1}:
        raise NotImplementedError("split normal evaluation is not yet implemented in the dense core")
    if h <= 0:
        raise ValueError("h must be positive")

    tau_interval = Interval(0.0, float(h))
    base_ext_sparse = x0_tm.extend_domain(tau_interval)
    tau_index = x0_tm.n_vars
    counters = DenseExecutionCounters()
    range_policy = dense_range_policy or DenseRangePolicy()
    initial_policy = (
        DenseRangePolicy()
        if range_policy.trigger == "on_validation_failure" and range_policy.method != "natural"
        else range_policy
    )
    range_trace: list[dict[str, Any]] = []
    base_ext_dense = sparse_tmvector_to_dense(
        base_ext_sparse,
        order=int(order),
        device=dense_device,
        dtype=dense_dtype,
        counters=counters,
        segment_boundary=True,
        range_policy=initial_policy,
        range_trace=range_trace,
    )

    def _with_policy(model: BatchedTaylorModel, policy: DenseRangePolicy, trace: list[dict[str, Any]]) -> BatchedTaylorModel:
        return BatchedTaylorModel(
            model.poly,
            model.rem_lo,
            model.rem_hi,
            model.domain_lo,
            model.domain_hi,
            model.ledger,
            policy,
            trace,
        )

    def _validate(model: BatchedTaylorModel):
        device = model.poly.coeffs.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            return dense_picard_validate_step(
                ode_fn,
                model,
                h=float(h),
                order=int(order),
                tau_index=tau_index,
                target_remainder_radius=target_remainder_radius,
                cutoff_threshold=cutoff_threshold,
                max_validation_attempts=2 if max_validation_attempts is None else int(max_validation_attempts),
                validation_eps=validation_eps,
                validation_mode=validation_mode,
                counters=counters,
            )
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            counters.dense_kernel_s += time.perf_counter() - started

    lane_trace: list[Mapping[str, Any]] = []
    try:
        dense_result = _validate(base_ext_dense)
        lane_trace.extend(dense_result.trace)
        lane_trace.extend(range_trace)
        lane_trace.append(
            {
                "phase": "range_validation_lane",
                "range_method": initial_policy.method,
                "subdivision_depth": initial_policy.max_depth,
                "validation_status": dense_result.status,
                "natural_validation_failed": initial_policy.method == "natural" and not dense_result.accepted,
                "subdivision_validation_passed": False,
            }
        )
        if (
            not dense_result.accepted
            and range_policy.trigger == "on_validation_failure"
            and range_policy.method != "natural"
        ):
            for depth in range(1, int(range_policy.max_depth) + 1):
                level_trace: list[dict[str, Any]] = []
                level_policy = replace(range_policy, max_depth=depth, trigger="always")
                level_result = _validate(_with_policy(base_ext_dense, level_policy, level_trace))
                lane_trace.extend(level_result.trace)
                lane_trace.extend(level_trace)
                lane_trace.append(
                    {
                        "phase": "range_validation_lane",
                        "range_method": level_policy.method,
                        "subdivision_depth": depth,
                        "validation_status": level_result.status,
                        "natural_validation_failed": True,
                        "subdivision_validation_passed": level_result.accepted,
                    }
                )
                dense_result = level_result
                if level_result.accepted:
                    break
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        lane_trace.extend(range_trace)
        lane_trace.append(
            {
                "phase": "range_fail_closed",
                "finite": False,
                "range_method": range_policy.method,
                "rejection_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        segment_tm = dense_to_sparse_tmvector(base_ext_dense, counters=counters, segment_boundary=True)
        return FlowpipeSegment(
            tm=segment_tm,
            final_tm=x0_tm,
            status="failed",
            h=float(h),
            order=int(order),
            validation_attempts=0,
            message=f"dense polynomial range failed closed: {type(exc).__name__}: {exc}",
            tau_index=tau_index,
            endpoint_raw_tm=None,
            endpoint_tightened_tm=None,
            endpoint_semantics="unpublished_rejected_step",
            endpoint_tightening_applied=False,
            endpoint_tightening_validation_method="not_applied_range_failure",
            backend_lane="hybrid_dense_core",
            backend_counters=counters.as_dict(),
            backend_trace=tuple(lane_trace),
        )
    subdivision_rows = [
        row for row in lane_trace
        if row.get("phase") == "polynomial_range" and int(row.get("leaf_count", 1)) > 1
    ]
    counters.range_subdivision_invocations += len(subdivision_rows)
    counters.range_leaf_evaluations += sum(int(row.get("leaf_count", 0)) for row in subdivision_rows)
    dense_result = replace(dense_result, trace=tuple(lane_trace))
    segment_tm = dense_to_sparse_tmvector(
        dense_result.segment_tm,
        counters=counters,
        segment_boundary=True,
    )
    endpoint_raw_tm = (
        segment_tm.substitute_const(tau_index, float(h)).drop_variable(tau_index).apply_cutoff(cutoff_threshold)
        if dense_result.accepted
        else None
    )
    final_tm = endpoint_raw_tm if endpoint_raw_tm is not None else x0_tm

    context = dict(diagnostics_context or {})
    mode = context.pop("mode", diagnostics_mode)
    segment_index = context.pop("segment_index", diagnostics_segment_index)
    if diagnostics is not None:
        for dense_row in dense_result.trace:
            if dense_row.get("phase") != "remainder_validation":
                continue
            row = dict(context)
            row.update(dense_row)
            row.setdefault("mode", mode)
            row.setdefault("segment_index", segment_index)
            row.setdefault("h", float(h))
            row.setdefault("order", int(order))
            row.setdefault("finite_residual", bool(dense_row.get("finite", True)))
            row.setdefault("validation_message", dense_row.get("rejection_reason", ""))
            diagnostics.append(row)

    candidate_pair = [
        dense_result.candidate_remainder_lo.detach().cpu().reshape(-1).tolist(),
        dense_result.candidate_remainder_hi.detach().cpu().reshape(-1).tolist(),
    ]
    image_pair = [
        dense_result.picard_image_remainder_lo.detach().cpu().reshape(-1).tolist(),
        dense_result.picard_image_remainder_hi.detach().cpu().reshape(-1).tolist(),
    ]
    return FlowpipeSegment(
        tm=segment_tm,
        final_tm=final_tm,
        status="validated" if dense_result.accepted else "failed",
        h=float(h),
        order=int(order),
        validation_attempts=dense_result.validation_attempts,
        message=dense_result.message,
        tau_index=tau_index,
        endpoint_raw_tm=endpoint_raw_tm,
        endpoint_tightened_tm=endpoint_raw_tm,
        endpoint_semantics=("endpoint_raw_segment_substitution" if dense_result.accepted else "unpublished_rejected_step"),
        endpoint_tightening_applied=False,
        endpoint_tightening_validation_method="not_applied_dense_raw_endpoint",
        backend_lane="hybrid_dense_core",
        backend_counters=counters.as_dict(),
        backend_trace=dense_result.trace,
        candidate_remainder=candidate_pair,
        picard_image_remainder=image_pair,
        subset_margin=dense_result.subset_margin.detach().cpu().tolist(),
        validated_remainder_ledger=dense_result.validated_remainder_decomposition.ledger,
        validated_remainder_decomposition=dense_result.validated_remainder_decomposition,
    )


def flowpipe_step_from_tm(
    ode_fn: ODEFunction,
    x0_tm: TMVector,
    h: float,
    order: int,
    *,
    u_box: Sequence[Any] | None = None,
    affine_u: dict[str, Any] | None = None,
    max_validation_attempts: int | None = None,
    validation_eps: float = 1e-12,
    growth_factor: float = 1.25,
    validation_mode: str = "growth",
    target_remainder_radius: float = 1e-4,
    center_correction_width_factor: float = 1.05,
    cutoff_threshold: float | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_mode: str | None = None,
    diagnostics_segment_index: int | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    symbolic_remainder: bool = False,
    max_symbolic_remainders: int = 0,
    symbolic_remainder_state: SymbolicRemainderState | None = None,
    diagnostic_mode: str | None = None,
    diagnostic_segment_index: int | None = None,
    diagnostic_context: Mapping[str, Any] | None = None,
    candidate_order: int | None = None,
    truncation_range_split: int | None = None,
    selective_high_degree_terms_top_k: int | None = None,
    normal_eval_range_split: int | None = None,
    tm_backend: str = "sparse",
    dense_device: torch.device | str = "cpu",
    dense_dtype: torch.dtype = torch.float64,
    dense_range_policy: Any | None = None,
) -> FlowpipeSegment:
    """Build one flowpipe segment from a TM initial condition.

    The returned segment preserves dependency on the variables already present in
    ``x0_tm`` and adds one local time variable.  The segment's final TM has the
    local time variable substituted with ``h`` and dropped.
    """
    if tm_backend not in {"sparse", "dense"}:
        raise ValueError("tm_backend must be 'sparse' or 'dense'")
    if diagnostic_context is not None:
        diagnostics_context = diagnostic_context
    if diagnostic_mode is not None:
        diagnostics_mode = diagnostic_mode
    if diagnostic_segment_index is not None:
        diagnostics_segment_index = diagnostic_segment_index
    if tm_backend == "dense":
        return _flowpipe_step_from_tm_hybrid_dense(
            ode_fn,
            x0_tm,
            h,
            order,
            max_validation_attempts=max_validation_attempts,
            validation_eps=validation_eps,
            validation_mode=validation_mode,
            target_remainder_radius=target_remainder_radius,
            cutoff_threshold=cutoff_threshold,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diagnostics_context,
            candidate_order=candidate_order,
            dense_device=dense_device,
            dense_dtype=dense_dtype,
            u_box=u_box,
            affine_u=affine_u,
            symbolic_remainder=symbolic_remainder,
            selective_high_degree_terms_top_k=selective_high_degree_terms_top_k,
            normal_eval_range_split=normal_eval_range_split,
            dense_range_policy=dense_range_policy,
        )
    if h <= 0:
        raise ValueError("h must be positive")
    output_order = int(order)
    candidate_order_i = int(candidate_order) if candidate_order is not None else output_order
    if candidate_order_i < output_order:
        raise ValueError("candidate_order must be >= output order")
    split = _truncation_split_value(truncation_range_split)
    normal_split = _truncation_split_value(normal_eval_range_split)
    diag_context = dict(diagnostics_context or {})
    diag_context.setdefault("output_order", output_order)
    diag_context.setdefault("candidate_order", candidate_order_i)
    diag_context.setdefault("truncation_range_split", split or "")
    diag_context.setdefault("normal_eval_range_split", normal_split or "")
    selective_top_k = int(selective_high_degree_terms_top_k or 0)
    if selective_top_k > 0:
        diag_context.setdefault("selective_high_degree_terms_top_k", selective_top_k)
    tau_interval = Interval(0.0, float(h))
    base_ext = x0_tm.extend_domain(tau_interval)
    tau_index = x0_tm.n_vars
    domain = base_ext.domain
    base_poly_ext = TMVector(
        TaylorModel(
            m.polynomial,
            Interval.zero(),
            domain,
            order=candidate_order_i,
            truncation_range_split=split,
        )
        for m in base_ext
    )
    if validation_mode not in {
        "growth",
        "current",
        "target_remainder",
        "target_remainder_normal_eval",
        "target_remainder_refined",
        "target_remainder_centered",
        "target_remainder_flowstar_ctrunc",
        "flowstar_raw_remainder_compat",
    }:
        raise ValueError(
            "validation_mode must be 'growth', 'current', 'target_remainder', 'target_remainder_normal_eval', "
            "'target_remainder_refined', 'target_remainder_centered', 'target_remainder_flowstar_ctrunc', "
            "or 'flowstar_raw_remainder_compat'"
        )
    target_mode = validation_mode in {
        "target_remainder",
        "target_remainder_normal_eval",
        "target_remainder_refined",
        "target_remainder_centered",
        "target_remainder_flowstar_ctrunc",
        "flowstar_raw_remainder_compat",
    }
    attempt_limit = (2 if target_mode else 20) if max_validation_attempts is None else int(max_validation_attempts)
    if attempt_limit <= 0:
        raise ValueError("max_validation_attempts must be positive")

    u_tms = _make_controls(
        u_box,
        affine_u,
        domain,
        candidate_order_i,
        truncation_range_split=split,
    )
    candidate_poly = _picard_polynomial(
        ode_fn,
        base_poly_ext,
        tau_index,
        candidate_order_i,
        u_tms,
        cutoff_threshold=cutoff_threshold,
        truncation_range_split=split,
    )
    _add_term_hash_metrics(diag_context, "candidate_terms_before_validation", candidate_poly, output_order)
    validation_candidate_poly = candidate_poly
    validation_selective_stats: list[dict[str, Any]] = []
    validation_selective_details: list[dict[str, Any]] = []
    if selective_top_k > 0:
        validation_candidate_poly, validation_selective_stats, validation_selective_details = _truncate_tm_to_order_selective(
            candidate_poly,
            output_order,
            selective_top_k=selective_top_k,
            result_order=candidate_order_i,
        )
    _add_term_hash_metrics(diag_context, "candidate_terms_after_selective", validation_candidate_poly, output_order)
    if validation_mode in {"target_remainder", "target_remainder_normal_eval"}:
        validated, status, attempts, message = _validate_picard_target_remainder(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            target_remainder_radius=target_remainder_radius,
            normal_eval_range_split=normal_split if validation_mode == "target_remainder_normal_eval" else None,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    elif validation_mode == "target_remainder_centered":
        validated, status, attempts, message = _validate_picard_target_remainder_centered(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            target_remainder_radius=target_remainder_radius,
            center_correction_width_factor=center_correction_width_factor,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    elif validation_mode == "target_remainder_refined":
        validated, status, attempts, message = _validate_picard_target_remainder_refined(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            target_remainder_radius=target_remainder_radius,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    elif validation_mode in {"target_remainder_flowstar_ctrunc", "flowstar_raw_remainder_compat"}:
        validated, status, attempts, message = _validate_picard_target_remainder_flowstar_ctrunc(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            target_remainder_radius=target_remainder_radius,
            cutoff_threshold=cutoff_threshold,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
            raw_remainder_mode="flowstar_compat" if validation_mode == "flowstar_raw_remainder_compat" else "",
        )
    else:
        validated, status, attempts, message = _validate_picard(
            ode_fn,
            base_ext,
            validation_candidate_poly,
            tau_index,
            candidate_order_i,
            u_tms,
            h=float(h),
            max_attempts=attempt_limit,
            validation_eps=validation_eps,
            growth_factor=growth_factor,
            diagnostics=diagnostics,
            diagnostics_mode=diagnostics_mode,
            diagnostics_segment_index=diagnostics_segment_index,
            diagnostics_context=diag_context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            symbolic_remainder=symbolic_remainder,
            max_symbolic_remainders=max_symbolic_remainders,
        )
    endpoint_raw_tm = validated.substitute_const(tau_index, float(h)).drop_variable(tau_index)
    endpoint_raw_tm = endpoint_raw_tm.apply_cutoff(cutoff_threshold)
    endpoint_tightened_tm = endpoint_raw_tm
    endpoint_tightening_applied = False
    endpoint_tightening_validation_method = "not_applied"
    if status == "validated" and validation_mode not in {"target_remainder_flowstar_ctrunc", "flowstar_raw_remainder_compat"}:
        # The segment remainder is valid for every tau in [0,h].  For multi-step
        # propagation we only need the endpoint at tau=h, so tighten the endpoint
        # remainder by re-evaluating the Picard residual at that fixed local time.
        try:
            rhs = _call_ode(ode_fn, validated, u_tms)
            final_models = []
            for base_i, cand_i, f_i in zip(base_ext, validated, rhs):
                picard_i = base_i + f_i.integrate(tau_index)
                residual_i = picard_i - TaylorModel(
                    cand_i.polynomial,
                    Interval.zero(),
                    domain,
                    order=candidate_order_i,
                    truncation_range_split=cand_i.truncation_range_split,
                )
                endpoint_residual = (
                    residual_i.substitute_const(tau_index, float(h))
                    .drop_variable(tau_index)
                    .range_box()
                    .inflate(validation_eps)
                )
                endpoint_poly = cand_i.polynomial.substitute_const(tau_index, float(h)).drop_variable(tau_index)
                endpoint_domain = [d for i, d in enumerate(domain) if i != tau_index]
                final_models.append(TaylorModel(
                    endpoint_poly,
                    endpoint_residual,
                    endpoint_domain,
                    order=candidate_order_i,
                    truncation_range_split=cand_i.truncation_range_split,
                ))
            endpoint_tightened_tm = TMVector(final_models).apply_cutoff(cutoff_threshold)
            endpoint_tightening_applied = True
            endpoint_tightening_validation_method = (
                "fixed_time_picard_residual_interval_evaluation"
            )
        except Exception as exc:
            message = message or f"endpoint tightening skipped: {exc}"
            endpoint_tightening_validation_method = (
                f"skipped_after_exception:{type(exc).__name__}"
            )
    selective_stats: dict[str, Any] = {}
    selective_details: list[dict[str, Any]] = []
    if selective_top_k > 0:
        endpoint_tightened_tm, final_selective_stats, final_selective_details = _truncate_tm_to_order_selective(
            endpoint_tightened_tm,
            output_order,
            selective_top_k=selective_top_k,
        )
        endpoint_raw_tm, _, _ = _truncate_tm_to_order_selective(
            endpoint_raw_tm,
            output_order,
            selective_top_k=selective_top_k,
        )
        output_tm, output_selective_stats, output_selective_details = _truncate_tm_to_order_selective(
            validated,
            output_order,
            selective_top_k=selective_top_k,
        )
        selective_stats = _aggregate_selective_stats(validation_selective_stats or output_selective_stats, top_k=selective_top_k)
        selective_details = validation_selective_details or output_selective_details
        endpoint_tightened_tm = endpoint_tightened_tm.apply_cutoff(cutoff_threshold)
        endpoint_raw_tm = endpoint_raw_tm.apply_cutoff(cutoff_threshold)
    else:
        endpoint_tightened_tm = _truncate_tm_to_order(
            endpoint_tightened_tm, output_order
        ).apply_cutoff(cutoff_threshold)
        endpoint_raw_tm = _truncate_tm_to_order(
            endpoint_raw_tm, output_order
        ).apply_cutoff(cutoff_threshold)
        output_tm = _truncate_tm_to_order(validated, output_order)

    final_tm = endpoint_tightened_tm
    next_symbolic_state = symbolic_remainder_state
    symbolic_stats: Mapping[str, Any] | None = None
    if symbolic_remainder:
        if status == "validated":
            final_tm, next_symbolic_state, symbolic_stats = introduce_symbolic_remainders(
                final_tm,
                symbolic_remainder_state,
                max_symbolic_remainders=max_symbolic_remainders,
            )
            endpoint_tightened_tm = final_tm
        else:
            next_symbolic_state = symbolic_remainder_state or SymbolicRemainderState.empty(max_symbolic_remainders)
            symbolic_stats = {
                "introduced_symbols": 0,
                "active_noise_symbols": len(next_symbolic_state.symbols),
                "symbolic_remainder_width_sum": "",
                "ordinary_remainder_width_sum": "",
                "materialized_remainder_width_sum": "",
            }
    return FlowpipeSegment(
        tm=output_tm,
        final_tm=final_tm,
        status=status,
        h=float(h),
        order=output_order,
        validation_attempts=attempts,
        message=message,
        tau_index=tau_index,
        symbolic_remainder=bool(symbolic_remainder),
        symbolic_remainder_state=next_symbolic_state,
        symbolic_remainder_stats=symbolic_stats,
        selective_term_stats=selective_stats or None,
        selective_term_details=selective_details or None,
        endpoint_raw_tm=endpoint_raw_tm,
        endpoint_tightened_tm=endpoint_tightened_tm,
        endpoint_semantics=(
            "endpoint_tightened_fixed_time_residual"
            if endpoint_tightening_applied
            else "endpoint_raw_segment_substitution"
        ),
        endpoint_tightening_applied=endpoint_tightening_applied,
        endpoint_tightening_validation_method=endpoint_tightening_validation_method,
    )


def flowpipe_step(
    ode_fn: ODEFunction,
    x0_box: Sequence[Interval | tuple[float, float] | list[float] | float],
    h: float,
    order: int,
    *,
    u_box: Sequence[Any] | None = None,
    affine_u: dict[str, Any] | None = None,
    **kwargs: Any,
) -> FlowpipeSegment:
    """Build one validated segment from an interval-box initial set."""
    domain = _as_interval_list(x0_box)
    x0_tm = TMVector.identity(domain, order=order)
    return flowpipe_step_from_tm(ode_fn, x0_tm, h, order, u_box=u_box, affine_u=affine_u, **kwargs)


def _is_target_containment_failure(message: str) -> bool:
    return "target remainder" in message or "not subset" in message or "target containment" in message


def flowpipe_step_flowstar_style_adaptive(
    ode_fn: ODEFunction,
    x0: TMVector | Sequence[Interval | tuple[float, float] | list[float] | float],
    h: float | None = None,
    order: int = 4,
    *,
    u_box: Sequence[Any] | None = None,
    affine_u: dict[str, Any] | None = None,
    h_min: float = 0.002,
    h_max: float = 0.1,
    target_remainder_radius: float = 1e-4,
    center_correction_width_factor: float = 1.05,
    cutoff_threshold: float | None = 1e-10,
    max_validation_attempts: int = 2,
    validation_eps: float = 1e-12,
    validation_mode: str = "target_remainder",
    adaptive_order_fallback: int | None = None,
    adaptive_order_threshold_factor: float = 1.25,
    grow_factor: float = 1.5,
    step_policy_mode: str = "",
    diagnostics: list[dict[str, Any]] | None = None,
    diagnostics_context: Mapping[str, Any] | None = None,
    rhs_breakdown_callback: Callable[[TMVector, int, int, Mapping[str, Any]], None] | None = None,
    candidate_order: int | None = None,
    truncation_range_split: int | None = None,
    selective_high_degree_terms_top_k: int | None = None,
    normal_eval_range_split: int | None = None,
    reset_mode: str = "normalized_endpoint_box",
    flowstar_symbolic_queue_state: FlowstarSymbolicRemainderQueue | None = None,
    flowstar_symbolic_queue_max_size: int = 100,
    flowstar_normal_state: FlowstarNormalFlowpipeState | None = None,
    scalar_recenter_remainder_midpoint: bool = False,
    right_map_range_mode: str = "standard",
    right_map_center_mode: str = "constant",
    symbolic_queue_mode: str = "",
    horner_diagnostic: bool = False,
    tm_backend: str = "sparse",
    dense_device: torch.device | str = "cpu",
    dense_dtype: torch.dtype = torch.float64,
    dense_range_policy: Any | None = None,
    structured_allow_outward_renormalization: bool = True,
) -> FlowpipeSegment:
    if h_min <= 0 or h_max <= 0:
        raise ValueError("h_min and h_max must be positive")
    if h_min > h_max:
        raise ValueError("h_min must be <= h_max")
    if validation_mode not in {
        "target_remainder",
        "target_remainder_normal_eval",
        "target_remainder_refined",
        "target_remainder_centered",
        "target_remainder_flowstar_ctrunc",
        "flowstar_raw_remainder_compat",
    }:
        raise ValueError("flowstar_style adaptive validation must use a target remainder mode")
    normal_insertion_modes = {
        "normalized_insertion",
        NORMALIZED_INSERTION_DEPENDENCY_PRESERVING,
        "normalized_insertion_complete_polynomial",
        "normalized_insertion_symqueue",
        "normalized_insertion_symqueue_split",
        "normalized_insertion_symqueue_v2",
        "normalized_insertion_horner",
        "normalized_insertion_horner_symqueue_v2",
        "normalized_insertion_structured_remainder_k16",
        "normalized_insertion_structured_total_delta_k16",
        BOUNDED_SOURCE_LEDGER_CANDIDATE,
        G2_SHARED_COLUMN_CANDIDATE,
    }
    if reset_mode not in {"normalized_endpoint_box", "flowstar_symbolic_remainder_queue", *normal_insertion_modes}:
        raise ValueError(
            "reset_mode must be 'normalized_endpoint_box', 'flowstar_symbolic_remainder_queue', "
            "'normalized_insertion', 'normalized_insertion_complete_polynomial', "
            "'normalized_insertion_symqueue', "
            "'normalized_insertion_symqueue_split', 'normalized_insertion_symqueue_v2', "
            f"'{NORMALIZED_INSERTION_DEPENDENCY_PRESERVING}', "
            "'normalized_insertion_horner', or "
            "'normalized_insertion_structured_remainder_k16', or "
            "'normalized_insertion_structured_total_delta_k16', or "
            f"'{BOUNDED_SOURCE_LEDGER_CANDIDATE}', or "
            f"'{G2_SHARED_COLUMN_CANDIDATE}'"
        )
    if right_map_range_mode not in {"standard", "normal_eval"}:
        raise ValueError("right_map_range_mode must be 'standard' or 'normal_eval'")
    if right_map_center_mode not in {"constant", "range_midpoint"}:
        raise ValueError("right_map_center_mode must be 'constant' or 'range_midpoint'")
    if symbolic_queue_mode not in {"", "flowstar_linear_v2"}:
        raise ValueError("symbolic_queue_mode must be empty or 'flowstar_linear_v2'")
    if step_policy_mode not in {"", "flowstar_compat"}:
        raise ValueError("step_policy_mode must be empty or 'flowstar_compat'")
    if tm_backend not in {"sparse", "dense"}:
        raise ValueError("tm_backend must be 'sparse' or 'dense'")
    step_shrink_factor = FLOWSTAR_COMPAT_STEP_SHRINK
    effective_grow_factor = FLOWSTAR_COMPAT_STEP_GROW if step_policy_mode == "flowstar_compat" else float(grow_factor)
    normal_state = flowstar_normal_state
    if reset_mode in normal_insertion_modes and normal_state is None and not isinstance(x0, TMVector):
        normal_state = FlowstarNormalFlowpipeState.from_initial_box(x0, order)
        if reset_mode == BOUNDED_SOURCE_LEDGER_CANDIDATE:
            normal_state = _initialize_bounded_source_normal_state(normal_state, order)
        elif reset_mode == G2_SHARED_COLUMN_CANDIDATE:
            normal_state = _initialize_g2_shared_column_normal_state(normal_state, order)
        elif reset_mode in {
            "normalized_insertion_structured_remainder_k16",
            "normalized_insertion_structured_total_delta_k16",
        }:
            structured = initialize_structured_remainder_state(
                1,
                len(normal_state.center),
                dtype=torch.float64,
                device=dense_device,
            )
            initial_scale = torch.tensor(
                [normal_state.scales], dtype=torch.float64, device=dense_device
            )
            initial_inverse = torch.where(
                initial_scale == 0,
                torch.ones_like(initial_scale),
                1.0 / initial_scale,
            )
            structured = replace(structured, inverse_scale=initial_inverse)
            normal_state = replace(
                normal_state,
                structured_remainder_state=structured,
                diagnostics={
                    **dict(normal_state.diagnostics or {}),
                    "reset_mode": reset_mode,
                    "structured_initial_state": True,
                },
            )
    current_tm = (
        normal_state.normalized_initial_tm(order)
        if reset_mode in normal_insertion_modes and normal_state is not None
        else (x0 if isinstance(x0, TMVector) else _normalized_tm_from_box(x0, order))
    )
    queue_state = flowstar_symbolic_queue_state

    def _assign_reset(seg: FlowpipeSegment, accepted_h: float) -> FlowpipeSegment:
        nonlocal queue_state, normal_state
        if reset_mode == "flowstar_symbolic_remainder_queue":
            reset_tm, queue_state, queue_stats = flowstar_symbolic_remainder_queue_reset(
                seg.final_tm,
                queue_state,
                max_size=flowstar_symbolic_queue_max_size,
            )
            seg.reset_tm = reset_tm
            seg.flowstar_symbolic_queue_state = queue_state
            seg.flowstar_symbolic_queue_stats = {**queue_stats, "reset_mode": reset_mode}
        elif reset_mode in normal_insertion_modes:
            if reset_mode == G2_SHARED_COLUMN_CANDIDATE:
                assert normal_state is not None
                source_before = normal_state.g2_shared_column_state
                try:
                    reset_tm, normal_state, normal_stats = _flowstar_g2_shared_column_transition(
                        seg,
                        normal_state,
                        order,
                        cutoff_threshold=cutoff_threshold,
                        right_map_range_mode=right_map_range_mode,
                        right_map_center_mode=right_map_center_mode,
                    )
                except (FloatingPointError, RuntimeError, ValueError) as exc:
                    seg.status = "failed"
                    seg.message = (
                        "G2 shared-column accepted-boundary gate failed closed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    seg.reset_tm = None
                    seg.flowstar_normal_state = normal_state
                    seg.source_ledger_state_before = source_before
                    seg.source_ledger_state_after = source_before
                    seg.next_h = accepted_h
                    return seg
                seg.reset_tm = reset_tm
                seg.flowstar_normal_state = normal_state
                seg.flowstar_normal_stats = {**normal_stats, "reset_mode": reset_mode}
                seg.source_ledger_state_before = source_before
                seg.source_ledger_state_after = normal_state.g2_shared_column_state
                seg.source_ledger_boundary_result = {
                    "accepted": True,
                    "pre_fingerprint": source_before.fingerprint if source_before else "",
                    "post_fingerprint": (
                        normal_state.g2_shared_column_state.fingerprint
                        if normal_state.g2_shared_column_state
                        else ""
                    ),
                    "next_picard_input_coefficients_sha256": normal_stats.get(
                        "g2_reset_coefficients_sha256", ""
                    ),
                }
                seg.flowstar_symbolic_queue_state = None
                seg.flowstar_symbolic_queue_stats = {"reset_mode": reset_mode}
                seg.next_h = min(accepted_h * effective_grow_factor, h_max)
                return seg
            if reset_mode == BOUNDED_SOURCE_LEDGER_CANDIDATE:
                assert normal_state is not None
                source_before = normal_state.bounded_source_ledger_state
                try:
                    reset_tm, normal_state, normal_stats = _flowstar_bounded_source_ledger_transition(
                        seg,
                        normal_state,
                        order,
                        cutoff_threshold=cutoff_threshold,
                        right_map_range_mode=right_map_range_mode,
                        right_map_center_mode=right_map_center_mode,
                    )
                except (FloatingPointError, RuntimeError, ValueError) as exc:
                    seg.status = "failed"
                    seg.message = (
                        "bounded source-ledger accepted-boundary gate failed closed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    seg.reset_tm = None
                    seg.flowstar_normal_state = normal_state
                    seg.source_ledger_state_before = source_before
                    seg.source_ledger_state_after = source_before
                    seg.next_h = accepted_h
                    return seg
                seg.reset_tm = reset_tm
                seg.flowstar_normal_state = normal_state
                seg.flowstar_normal_stats = {**normal_stats, "reset_mode": reset_mode}
                seg.source_ledger_state_before = source_before
                seg.source_ledger_state_after = normal_state.bounded_source_ledger_state
                seg.source_ledger_boundary_result = {
                    "accepted": True,
                    "pre_fingerprint": source_before.fingerprint if source_before else "",
                    "post_fingerprint": (
                        normal_state.bounded_source_ledger_state.fingerprint
                        if normal_state.bounded_source_ledger_state
                        else ""
                    ),
                    "next_picard_input_coefficients_sha256": normal_stats.get(
                        "source_ledger_reset_coefficients_sha256", ""
                    ),
                }
                seg.flowstar_symbolic_queue_state = None
                seg.flowstar_symbolic_queue_stats = {"reset_mode": reset_mode}
                seg.next_h = min(accepted_h * effective_grow_factor, h_max)
                return seg
            if reset_mode in {
                "normalized_insertion_structured_remainder_k16",
                "normalized_insertion_structured_total_delta_k16",
            }:
                assert normal_state is not None
                try:
                    reset_tm, normal_state, normal_stats = _flowstar_structured_insertion_transition(
                        seg,
                        normal_state,
                        order,
                        cutoff_threshold=cutoff_threshold,
                        target_remainder_radius=target_remainder_radius,
                        scalar_recenter_remainder_midpoint=scalar_recenter_remainder_midpoint,
                        right_map_range_mode=right_map_range_mode,
                        right_map_center_mode=right_map_center_mode,
                        horner_diagnostic=horner_diagnostic,
                        allow_outward_renormalization=structured_allow_outward_renormalization,
                        image_contract=(
                            "total_delta"
                            if reset_mode
                            == "normalized_insertion_structured_total_delta_k16"
                            else "current"
                        ),
                    )
                except (FloatingPointError, RuntimeError, ValueError) as exc:
                    seg.status = "failed"
                    seg.message = f"S1 accepted-boundary gate failed closed: {type(exc).__name__}: {exc}"
                    seg.reset_tm = None
                    seg.flowstar_normal_state = normal_state
                    seg.structured_state_before = normal_state.structured_remainder_state
                    seg.structured_state_after = normal_state.structured_remainder_state
                    seg.next_h = accepted_h
                    return seg
                seg.reset_tm = reset_tm
                seg.flowstar_normal_state = normal_state
                seg.flowstar_normal_stats = {**normal_stats, "reset_mode": reset_mode}
                seg.flowstar_symbolic_queue_state = None
                seg.flowstar_symbolic_queue_stats = {"reset_mode": reset_mode}
                seg.next_h = min(accepted_h * effective_grow_factor, h_max)
                return seg
            use_v2 = reset_mode in {
                "normalized_insertion_symqueue_v2",
                "normalized_insertion_horner_symqueue_v2",
            } or symbolic_queue_mode == "flowstar_linear_v2"
            use_symqueue = reset_mode in {"normalized_insertion_symqueue", "normalized_insertion_symqueue_split"} or use_v2
            use_split = reset_mode == "normalized_insertion_symqueue_split"
            use_horner = reset_mode in {
                "normalized_insertion_horner",
                "normalized_insertion_horner_symqueue_v2",
            }
            use_dependency_preserving = (
                reset_mode == NORMALIZED_INSERTION_DEPENDENCY_PRESERVING
            )
            use_complete_polynomial = reset_mode == "normalized_insertion_complete_polynomial"
            reset_tm, normal_state, normal_stats = _flowstar_normalized_insertion_transition(
                seg,
                normal_state,
                order,
                cutoff_threshold=cutoff_threshold,
                symbolic_queue=use_symqueue,
                symbolic_queue_split=use_split,
                symbolic_queue_state=queue_state,
                symbolic_queue_max_size=flowstar_symbolic_queue_max_size,
                symbolic_queue_mode="flowstar_linear_v2" if use_v2 else symbolic_queue_mode,
                target_remainder_radius=target_remainder_radius,
                scalar_recenter_remainder_midpoint=scalar_recenter_remainder_midpoint,
                right_map_range_mode=right_map_range_mode,
                right_map_center_mode=right_map_center_mode,
                horner_diagnostic=horner_diagnostic,
                horner_insertion=use_horner,
                dependency_preserving_insertion=use_dependency_preserving,
                complete_polynomial_carry=use_complete_polynomial,
            )
            symbolic_output_remainders = normal_stats.get("_symbolic_output_remainders")
            if (use_split or use_v2) and symbolic_output_remainders:
                seg.final_tm = _tmvector_add_remainders(seg.final_tm, symbolic_output_remainders)
                seg.tm = _tmvector_add_remainders(seg.tm, symbolic_output_remainders)
            if use_symqueue and normal_state is not None:
                queue_state = normal_state.symbolic_queue
            seg.reset_tm = reset_tm
            seg.flowstar_normal_state = normal_state
            seg.flowstar_normal_stats = {**normal_stats, "reset_mode": reset_mode}
            seg.flowstar_symbolic_queue_state = queue_state
            seg.flowstar_symbolic_queue_stats = (
                {**normal_stats, "reset_mode": reset_mode} if use_symqueue else {"reset_mode": reset_mode}
            )
        else:
            seg.reset_tm = _normalized_tm_from_box(seg.final_tm.range_box(), order)
            seg.flowstar_symbolic_queue_state = queue_state
            seg.flowstar_symbolic_queue_stats = {"reset_mode": reset_mode}
        seg.next_h = min(accepted_h * effective_grow_factor, h_max)
        return seg
    h_try = min(float(h) if h is not None else float(h_max), float(h_max))
    if h_try < h_min:
        raise ValueError("initial h is below h_min")

    last_seg: FlowpipeSegment | None = None
    rejections = 0
    adaptive_attempt = 0
    while h_try + 1e-15 >= h_min:
        adaptive_attempt += 1
        context = dict(diagnostics_context or {})
        context.setdefault("mode", "flowstar_style")
        context["adaptive_attempt_index"] = adaptive_attempt
        context["h_try"] = h_try
        context["h_min"] = float(h_min)
        context["h_max"] = float(h_max)
        context["step_policy_mode"] = step_policy_mode
        context["step_shrink_factor"] = step_shrink_factor
        context["step_grow_factor"] = effective_grow_factor
        seg = flowpipe_step_from_tm(
            ode_fn,
            current_tm,
            h_try,
            order,
            u_box=u_box,
            affine_u=affine_u,
            max_validation_attempts=max_validation_attempts,
            validation_eps=validation_eps,
            validation_mode=validation_mode,
            target_remainder_radius=target_remainder_radius,
            center_correction_width_factor=center_correction_width_factor,
            cutoff_threshold=cutoff_threshold,
            diagnostics=diagnostics,
            diagnostics_context=context,
            rhs_breakdown_callback=rhs_breakdown_callback,
            candidate_order=candidate_order,
            truncation_range_split=truncation_range_split,
            selective_high_degree_terms_top_k=selective_high_degree_terms_top_k,
            normal_eval_range_split=normal_eval_range_split,
            tm_backend=tm_backend,
            dense_device=dense_device,
            dense_dtype=dense_dtype,
            dense_range_policy=dense_range_policy,
        )
        seg.step_rejections = rejections
        if seg.status == "validated" and intervals_are_finite(seg.final_tm.range_box()):
            return _assign_reset(seg, h_try)

        last_seg = seg
        fallback_order = int(adaptive_order_fallback or 0)
        near_min_failure = (
            h_try <= float(adaptive_order_threshold_factor) * float(h_min) + 1e-15
            or h_try * step_shrink_factor < float(h_min) - 1e-15
        )
        should_retry_order = (
            fallback_order > int(order)
            and int(order) == 6
            and near_min_failure
            and _is_target_containment_failure(seg.message)
        )
        if should_retry_order:
            fallback_context = dict(context)
            fallback_context["adaptive_order_fallback"] = True
            fallback_context["fallback_from_order"] = int(order)
            fallback_context["h_try"] = h_try
            fallback_seg = flowpipe_step_from_tm(
                ode_fn,
                current_tm,
                h_try,
                fallback_order,
                u_box=u_box,
                affine_u=affine_u,
                max_validation_attempts=max_validation_attempts,
                validation_eps=validation_eps,
                validation_mode=validation_mode,
                target_remainder_radius=target_remainder_radius,
                center_correction_width_factor=center_correction_width_factor,
                cutoff_threshold=cutoff_threshold,
                diagnostics=diagnostics,
                diagnostics_context=fallback_context,
                rhs_breakdown_callback=rhs_breakdown_callback,
                candidate_order=None if candidate_order is None else max(int(candidate_order), fallback_order),
                truncation_range_split=truncation_range_split,
                selective_high_degree_terms_top_k=selective_high_degree_terms_top_k,
                normal_eval_range_split=normal_eval_range_split,
                tm_backend=tm_backend,
                dense_device=dense_device,
                dense_dtype=dense_dtype,
                dense_range_policy=dense_range_policy,
            )
            fallback_seg.step_rejections = rejections
            last_seg = fallback_seg
            if fallback_seg.status == "validated" and intervals_are_finite(fallback_seg.final_tm.range_box()):
                return _assign_reset(fallback_seg, h_try)

        rejections += 1
        h_try *= step_shrink_factor

    assert last_seg is not None
    last_seg.step_rejections = rejections
    last_seg.next_h = h_try
    base_message = last_seg.message or "target remainder validation failed"
    last_seg.message = f"{base_message}; minimum step reached before h_min={h_min:g}"
    return last_seg

def _step_diagnostics_kwargs(kwargs: Mapping[str, Any], mode: str, segment_index: int) -> dict[str, Any]:
    step_kwargs = dict(kwargs)
    if step_kwargs.get("diagnostics") is not None:
        context = dict(step_kwargs.get("diagnostic_context") or step_kwargs.get("diagnostics_context") or {})
        context.setdefault("mode", mode)
        context.setdefault("segment_index", segment_index)
        step_kwargs["diagnostics_context"] = context
        step_kwargs.pop("diagnostic_context", None)
    return step_kwargs


def flowpipe_multi_step(
    ode_fn: ODEFunction,
    x0_box: Sequence[Interval | tuple[float, float] | list[float] | float],
    h: float,
    steps: int,
    order: int,
    *,
    mode: str = "range_only",
    u_box: Sequence[Any] | None = None,
    affine_u: dict[str, Any] | None = None,
    range_only_inflate: float = 1e-9,
    **kwargs: Any,
) -> FlowpipeResult:
    """Repeatedly propagate a polynomial ODE.

    ``mode='range_only'`` keeps the old baseline behavior: each step compresses
    the previous final Taylor model to a box and restarts with fresh identity
    variables.  ``mode='dependency_preserving'`` propagates the final Taylor
    model directly and therefore keeps symbolic dependency on the original
    initial-state variables.  ``mode='flowstar_style'`` recenters each endpoint
    box and restarts from fresh normalized variables in ``[-1, 1]``.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if mode not in {"range_only", "dependency_preserving", "flowstar_style"}:
        raise ValueError("mode must be 'range_only', 'dependency_preserving', or 'flowstar_style'")

    segments: list[FlowpipeSegment] = []
    if mode == "range_only":
        current_box = _as_interval_list(x0_box)
        current_final = TMVector.identity(current_box, order=order)
        for segment_index in range(steps):
            step_kwargs = _step_diagnostics_kwargs(kwargs, mode, segment_index)
            seg = flowpipe_step(ode_fn, current_box, h, order, u_box=u_box, affine_u=affine_u, **step_kwargs)
            segments.append(seg)
            if seg.status != "validated":
                break
            current_box = [iv.inflate(range_only_inflate) for iv in seg.final_tm.range_box()]
            # The range-only baseline intentionally forgets symbolic dependency at
            # step boundaries.  Represent the compressed box as fresh identity TMs
            # so returned widths match the actual state passed to the next step.
            current_final = TMVector.identity(current_box, order=order)
        status = (
            "validated"
            if len(segments) == steps
            and all(s.status == "validated" for s in segments)
            else "failed"
        )
        return FlowpipeResult(segments, status, current_final, mode)

    if mode == "flowstar_style":
        current_tm = _normalized_tm_from_box(_as_interval_list(x0_box), order)
        for segment_index in range(steps):
            step_kwargs = _step_diagnostics_kwargs(kwargs, mode, segment_index)
            seg = flowpipe_step_from_tm(ode_fn, current_tm, h, order, u_box=u_box, affine_u=affine_u, **step_kwargs)
            segments.append(seg)
            if seg.status != "validated" or not intervals_are_finite(seg.final_tm.range_box()):
                break
            seg.reset_tm = _normalized_tm_from_box(seg.final_tm.range_box(), order)
            current_tm = seg.reset_tm
        status = "validated" if len(segments) == steps and all(s.status == "validated" for s in segments) else "failed"
        return FlowpipeResult(segments, status, current_tm, mode)

    current_tm = TMVector.identity(_as_interval_list(x0_box), order=order)
    for segment_index in range(steps):
        step_kwargs = _step_diagnostics_kwargs(kwargs, mode, segment_index)
        seg = flowpipe_step_from_tm(ode_fn, current_tm, h, order, u_box=u_box, affine_u=affine_u, **step_kwargs)
        segments.append(seg)
        if seg.status != "validated":
            break
        current_tm = seg.final_tm
    status = (
        "validated"
        if len(segments) == steps
        and all(s.status == "validated" for s in segments)
        else "failed"
    )
    return FlowpipeResult(segments, status, current_tm, mode)
