"""Lossless, language-neutral Brusselator live-range exchange objects.

The audit schema is ordered UTF-8 ``key=value``.  Every floating-point value
is represented by its finite binary64 hexadecimal spelling; decimal strings
are display-only and never accepted by the importer.  The minimal Flow*
harness consumes the same file directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .accepted_boundary_sr import (
    AcceptedBoundarySRPrepared,
    commit_accepted_boundary_sr,
    prepare_accepted_boundary_sr,
    split_endpoint_taylor_map,
)
from .flowpipe import (
    FlowstarNormalFlowpipeState,
    insert_ctrunc_normal_dependency_preserving,
)
from .interval import Interval
from .polynomial import Polynomial
from .symbolic_remainder import (
    ACCEPTED_BOUNDARY_SR_OWNER_SCHEMA,
    FlowstarSymbolicRemainderQueue,
    accepted_boundary_sr_queue_sha256,
)
from .taylor_model import TaylorModel
from .tm_vector import TMVector


SCHEMA = "torch_tm_flowpipe.brusselator_live_range_exchange/1"
STATE_DIMENSION = 2
ORDER = 6
STEP = 0.02
CUTOFF = 1.0e-10
TAU_INDEX = 2
VARIABLE_ORDER = ("ux", "uy", "tau")
FLOWSTAR_VARIABLE_ORDER = ("tau", "ux", "uy")


@dataclass(frozen=True)
class ExchangeBuild:
    records: tuple[tuple[str, str], ...]
    prepared: AcceptedBoundarySRPrepared
    reconstructed_endpoint: TMVector
    reconstructed_post_right: TMVector
    queue_after_sha256: str


def _hex(value: Any) -> str:
    number = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
    if not math.isfinite(number):
        raise ValueError("canonical exchange rejects nonfinite binary64")
    return number.hex()


def _unhex(value: str) -> float:
    if not value.startswith(("0x", "-0x")):
        raise ValueError(f"canonical numeric value is not hexadecimal: {value!r}")
    number = float.fromhex(value)
    if not math.isfinite(number) or number.hex() != value:
        raise ValueError(f"noncanonical or nonfinite binary64: {value!r}")
    return number


def _append_interval(rows: list[tuple[str, str]], key: str, value: Interval) -> None:
    lo = _hex(value.lo)
    hi = _hex(value.hi)
    if _unhex(lo) > _unhex(hi):
        raise ValueError(f"inverted canonical interval: {key}")
    rows.extend(((f"{key}.lo", lo), (f"{key}.hi", hi)))


def _append_tmv(
    rows: list[tuple[str, str]],
    prefix: str,
    value: TMVector,
    *,
    variable_order: Sequence[str],
) -> None:
    if value.n_vars != len(variable_order):
        raise ValueError(f"{prefix} variable-order dimension mismatch")
    rows.extend(
        (
            (f"{prefix}.component_count", str(len(value))),
            (f"{prefix}.variable_count", str(value.n_vars)),
            (f"{prefix}.variable_order", ",".join(variable_order)),
            (f"{prefix}.domain_count", str(len(value.domain))),
        )
    )
    for variable, domain in enumerate(value.domain):
        _append_interval(rows, f"{prefix}.domain.{variable}", domain)
    for component, model in enumerate(value):
        base = f"{prefix}.component.{component}"
        terms = sorted(model.polynomial.terms.items())
        rows.extend(
            (
                (f"{base}.order", str(int(model.order or ORDER))),
                (f"{base}.term_count", str(len(terms))),
            )
        )
        for term_index, (exponents, coefficient) in enumerate(terms):
            term = f"{base}.term.{term_index}"
            rows.extend(
                (
                    (f"{term}.exponents", ",".join(str(int(v)) for v in exponents)),
                    (f"{term}.total_degree", str(sum(int(v) for v in exponents))),
                    (f"{term}.coefficient_hex", _hex(coefficient)),
                )
            )
        _append_interval(rows, f"{base}.ordinary_remainder", model.remainder)


def _take(records: dict[str, str], key: str) -> str:
    try:
        return records.pop(key)
    except KeyError as error:
        raise ValueError(f"canonical exchange is missing {key}") from error


def _unsigned(records: dict[str, str], key: str) -> int:
    text = _take(records, key)
    if not text.isdigit():
        raise ValueError(f"canonical exchange has invalid unsigned field {key}")
    return int(text)


def _take_interval(records: dict[str, str], key: str) -> Interval:
    lo = _unhex(_take(records, f"{key}.lo"))
    hi = _unhex(_take(records, f"{key}.hi"))
    if lo > hi:
        raise ValueError(f"canonical exchange has inverted interval {key}")
    return Interval(lo, hi)


def parse_records(text: str) -> dict[str, str]:
    if not text or not text.endswith("\n"):
        raise ValueError("canonical exchange must be nonempty and newline-terminated")
    records: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.count("=") != 1:
            raise ValueError(f"invalid canonical record at line {line_number}")
        key, value = line.split("=", 1)
        if not key or not value or key in records:
            raise ValueError(f"empty or duplicate canonical field at line {line_number}")
        records[key] = value
    if records.get("schema") != SCHEMA:
        raise ValueError("canonical exchange schema mismatch")
    return records


def read_records(path: Path) -> dict[str, str]:
    return parse_records(Path(path).read_text(encoding="utf-8"))


def take_tmv(
    records: dict[str, str],
    prefix: str,
    *,
    expected_variable_order: Sequence[str] | None = None,
) -> TMVector:
    component_count = _unsigned(records, f"{prefix}.component_count")
    variable_count = _unsigned(records, f"{prefix}.variable_count")
    variable_order = tuple(_take(records, f"{prefix}.variable_order").split(","))
    if len(variable_order) != variable_count:
        raise ValueError(f"{prefix} variable order is malformed")
    if expected_variable_order is not None and variable_order != tuple(expected_variable_order):
        raise ValueError(f"{prefix} variable order mismatch")
    domain_count = _unsigned(records, f"{prefix}.domain_count")
    if domain_count != variable_count:
        raise ValueError(f"{prefix} domain dimension mismatch")
    domain = [_take_interval(records, f"{prefix}.domain.{i}") for i in range(domain_count)]
    models: list[TaylorModel] = []
    for component in range(component_count):
        base = f"{prefix}.component.{component}"
        order = _unsigned(records, f"{base}.order")
        term_count = _unsigned(records, f"{base}.term_count")
        terms: dict[tuple[int, ...], torch.Tensor] = {}
        for term_index in range(term_count):
            term = f"{base}.term.{term_index}"
            exponent_text = _take(records, f"{term}.exponents")
            try:
                exponent = tuple(int(item) for item in exponent_text.split(","))
            except ValueError as error:
                raise ValueError(f"{term} exponent table is malformed") from error
            if len(exponent) != variable_count or any(value < 0 for value in exponent):
                raise ValueError(f"{term} exponent dimension/sign is invalid")
            if _unsigned(records, f"{term}.total_degree") != sum(exponent):
                raise ValueError(f"{term} total degree mismatch")
            if sum(exponent) > order or exponent in terms:
                raise ValueError(f"{term} is duplicate or exceeds order")
            coefficient = _unhex(_take(records, f"{term}.coefficient_hex"))
            if coefficient == 0.0:
                raise ValueError(f"{term} contains an explicit zero coefficient")
            terms[exponent] = torch.tensor(coefficient, dtype=torch.float64)
        remainder = _take_interval(records, f"{base}.ordinary_remainder")
        models.append(
            TaylorModel(
                Polynomial(terms, variable_count),
                remainder,
                list(domain),
                order=order,
            )
        )
    return TMVector(models)


def _rm_constants(value: TMVector) -> TMVector:
    zero = (0,) * value.n_vars
    models: list[TaylorModel] = []
    for model in value:
        terms = dict(model.polynomial.terms)
        terms.pop(zero, None)
        models.append(
            TaylorModel(
                Polynomial(terms, model.n_vars),
                model.remainder,
                list(model.domain),
                order=model.order,
                truncation_range_split=model.truncation_range_split,
            )
        )
    return TMVector(models)


def _append_real_vector(rows: list[tuple[str, str]], key: str, values: Sequence[Any]) -> None:
    rows.append((f"{key}.count", str(len(values))))
    rows.extend((f"{key}.{index}", _hex(value)) for index, value in enumerate(values))


def _append_interval_vector(
    rows: list[tuple[str, str]], key: str, values: Sequence[Interval]
) -> None:
    rows.append((f"{key}.count", str(len(values))))
    for index, value in enumerate(values):
        _append_interval(rows, f"{key}.{index}", value)


def _append_real_matrix(
    rows: list[tuple[str, str]], key: str, matrix: Sequence[Sequence[Any]]
) -> None:
    row_count = len(matrix)
    column_count = len(matrix[0]) if matrix else 0
    if any(len(row) != column_count for row in matrix):
        raise ValueError(f"ragged matrix {key}")
    rows.extend(((f"{key}.rows", str(row_count)), (f"{key}.cols", str(column_count))))
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            rows.append((f"{key}.{row_index}.{column_index}", _hex(value)))


def _append_interval_matrix(
    rows: list[tuple[str, str]], key: str, matrix: Sequence[Sequence[Interval]]
) -> None:
    row_count = len(matrix)
    column_count = len(matrix[0]) if matrix else 0
    if any(len(row) != column_count for row in matrix):
        raise ValueError(f"ragged interval matrix {key}")
    rows.extend(((f"{key}.rows", str(row_count)), (f"{key}.cols", str(column_count))))
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            _append_interval(rows, f"{key}.{row_index}.{column_index}", value)


def _append_queue(
    rows: list[tuple[str, str]], key: str, queue: FlowstarSymbolicRemainderQueue
) -> None:
    rows.extend(
        (
            (f"{key}.owner_schema", queue.owner_schema),
            (f"{key}.max_size", str(queue.max_size)),
            (f"{key}.generation", str(queue.generation)),
            (f"{key}.accepted_boundary_index", str(queue.accepted_boundary_index)),
            (f"{key}.reset_count", str(queue.reset_count)),
            (f"{key}.owner_count", str(len(queue.owner_generations))),
        )
    )
    for index, (generation, boundary) in enumerate(
        zip(queue.owner_generations, queue.owner_boundary_indices, strict=True)
    ):
        rows.extend(
            (
                (f"{key}.owner.{index}.generation", str(generation)),
                (f"{key}.owner.{index}.boundary", str(boundary)),
            )
        )
    _append_real_vector(rows, f"{key}.scalars", queue.scalars)
    _append_interval_vector(rows, f"{key}.scalars_iv", queue.scalars_iv)
    rows.append((f"{key}.J_count", str(len(queue.J))))
    for index, column in enumerate(queue.J):
        _append_interval_vector(rows, f"{key}.J.{index}", column)
    rows.append((f"{key}.Phi_L_count", str(len(queue.Phi_L))))
    for index, matrix in enumerate(queue.Phi_L):
        _append_real_matrix(rows, f"{key}.Phi_L.{index}", matrix)
    rows.append((f"{key}.Phi_L_iv_count", str(len(queue.Phi_L_iv))))
    for index, matrix in enumerate(queue.Phi_L_iv):
        _append_interval_matrix(rows, f"{key}.Phi_L_iv.{index}", matrix)


def _step_tables() -> tuple[list[Interval], list[float]]:
    tau = Interval(0.0, STEP)
    tube = [Interval.point(1.0)]
    for power in range(1, ORDER + 1):
        tube.append(tau.pow_int(power))
    endpoint = [1.0]
    for _ in range(ORDER):
        endpoint.append(endpoint[-1] * STEP)
    return tube, endpoint


def _tmv_binary_equal(left: TMVector, right: TMVector) -> bool:
    rows_left: list[tuple[str, str]] = []
    rows_right: list[tuple[str, str]] = []
    order = tuple(f"v{index}" for index in range(left.n_vars))
    if right.n_vars != left.n_vars:
        return False
    _append_tmv(rows_left, "tm", left, variable_order=order)
    _append_tmv(rows_right, "tm", right, variable_order=order)
    return rows_left == rows_right


def build_exchange_records(
    *,
    pre_state: FlowstarNormalFlowpipeState,
    post_state: FlowstarNormalFlowpipeState,
    accepted_step: int,
    checkpoint_sha256: str,
    torch_solver_commit: str,
    flowstar_commit: str,
    source_hashes: Mapping[str, str],
) -> ExchangeBuild:
    if accepted_step < 1 or pre_state.step_index != accepted_step - 1:
        raise ValueError("canonical exchange prestate boundary mismatch")
    if post_state.step_index != accepted_step:
        raise ValueError("canonical exchange poststate boundary mismatch")
    tube = post_state.tmv_pre
    if tube.n_vars != 3:
        raise ValueError("Brusselator segment must use ux,uy,tau")
    endpoint_pre_cutoff = tube.substitute_const(TAU_INDEX, STEP).drop_variable(TAU_INDEX)
    endpoint = endpoint_pre_cutoff.apply_cutoff(CUTOFF)
    endpoint_without_constants = _rm_constants(endpoint)
    diagnostics: dict[str, Any] = {}
    prepared = prepare_accepted_boundary_sr(
        endpoint_without_constants,
        pre_state.tmv_right,
        domain=pre_state.domain,
        order=ORDER,
        cutoff_threshold=CUTOFF,
        queue_state=pre_state.symbolic_queue,
        queue_capacity=1000,
        previous_accepted_boundary_index=pre_state.step_index,
        compose=insert_ctrunc_normal_dependency_preserving,
        diagnostics=diagnostics,
        owner_schema=ACCEPTED_BOUNDARY_SR_OWNER_SCHEMA,
    )
    committed = commit_accepted_boundary_sr(
        prepared,
        normalization_scales=post_state.scales,
        cutoff_threshold=CUTOFF,
    )
    if not _tmv_binary_equal(committed.normalized_right_map, post_state.tmv_right):
        raise AssertionError("canonical boundary replay did not reconstruct post right map")
    if post_state.symbolic_queue is None:
        raise AssertionError("canonical C4 poststate lost its SR queue")
    queue_after_sha = accepted_boundary_sr_queue_sha256(committed.queue_after)
    if queue_after_sha != accepted_boundary_sr_queue_sha256(post_state.symbolic_queue):
        raise AssertionError("canonical boundary replay did not reconstruct post SR queue")
    linear, nonlinear = split_endpoint_taylor_map(endpoint_without_constants)
    tube_table, endpoint_table = _step_tables()
    rows: list[tuple[str, str]] = [
        ("schema", SCHEMA),
        ("object_role", "same_input_live_boundary_range_and_composition"),
        ("accepted_step", str(accepted_step)),
        ("state_dimension", str(STATE_DIMENSION)),
        ("order", str(ORDER)),
        ("polynomial_variable_order", ",".join(VARIABLE_ORDER)),
        ("flowstar_harness_variable_order", ",".join(FLOWSTAR_VARIABLE_ORDER)),
        ("tau_index", str(TAU_INDEX)),
        ("fixed_step_hex", STEP.hex()),
        ("cutoff_threshold_hex", CUTOFF.hex()),
        ("checkpoint_sha256", checkpoint_sha256),
        ("source.torch_solver_commit", torch_solver_commit),
        ("source.flowstar_commit", flowstar_commit),
        ("boundary.composition_branch", prepared.composition_branch),
        ("boundary.owner_schema", prepared.queue_before.owner_schema),
        ("boundary.owner_generation", str(prepared.accepted_boundary_index)),
        ("labels.reporting_endpoint", "tm.segment_endpoint_raw"),
        ("labels.reporting_tube", "tm.segment_tube"),
        ("labels.boundary_normalization", "tm.boundary_torch_inserted"),
        ("labels.right_map_construction", "tm.right_map_input"),
        ("labels.composition_truncation", "tm.boundary_outer_nonlinear"),
        ("labels.cutoff_payment", "cutoff_threshold_hex"),
        ("labels.picard_validation", "not_recomputed_by_range_harness"),
        ("labels.next_step_initialization", "post.center,post.scale"),
    ]
    for key, value in sorted(source_hashes.items()):
        rows.append((f"source.file_sha256.{key}", value))
    _append_interval_vector(rows, "table.step_exp", tube_table)
    _append_real_vector(rows, "table.step_end_exp", endpoint_table)
    _append_real_vector(rows, "pre.center", pre_state.center)
    _append_real_vector(rows, "pre.scale", pre_state.scales)
    _append_real_vector(rows, "post.center", post_state.center)
    _append_real_vector(rows, "post.scale", post_state.scales)
    _append_tmv(rows, "tm.segment_tube", tube, variable_order=VARIABLE_ORDER)
    _append_tmv(
        rows,
        "tm.segment_endpoint_pre_cutoff",
        endpoint_pre_cutoff,
        variable_order=VARIABLE_ORDER[:2],
    )
    _append_tmv(rows, "tm.segment_endpoint_raw", endpoint, variable_order=VARIABLE_ORDER[:2])
    _append_tmv(
        rows,
        "tm.boundary_outer_full",
        endpoint_without_constants,
        variable_order=VARIABLE_ORDER[:2],
    )
    _append_tmv(
        rows,
        "tm.boundary_outer_nonlinear",
        nonlinear,
        variable_order=VARIABLE_ORDER[:2],
    )
    _append_tmv(
        rows,
        "tm.right_map_input",
        pre_state.tmv_right,
        variable_order=VARIABLE_ORDER[:2],
    )
    _append_tmv(
        rows,
        "tm.boundary_torch_inserted",
        prepared.inserted,
        variable_order=VARIABLE_ORDER[:2],
    )
    _append_tmv(
        rows,
        "tm.right_map_torch_post_cutoff",
        post_state.tmv_right,
        variable_order=VARIABLE_ORDER[:2],
    )
    _append_real_matrix(rows, "boundary.linear", linear)
    _append_interval_vector(rows, "boundary.sr_propagated_history", prepared.propagated_history)
    _append_interval_vector(rows, "boundary.sr_current_owner", prepared.current_owner)
    _append_queue(rows, "queue.before", prepared.queue_before)
    keys = [key for key, _ in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("canonical exchange builder generated duplicate keys")
    return ExchangeBuild(
        records=tuple(rows),
        prepared=prepared,
        reconstructed_endpoint=endpoint,
        reconstructed_post_right=committed.normalized_right_map,
        queue_after_sha256=queue_after_sha,
    )


def write_records(path: Path, records: Iterable[tuple[str, str]]) -> str:
    payload = "".join(f"{key}={value}\n" for key, value in records).encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def object_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "CUTOFF",
    "ExchangeBuild",
    "FLOWSTAR_VARIABLE_ORDER",
    "ORDER",
    "SCHEMA",
    "STATE_DIMENSION",
    "STEP",
    "TAU_INDEX",
    "VARIABLE_ORDER",
    "build_exchange_records",
    "object_sha256",
    "parse_records",
    "read_records",
    "take_tmv",
    "write_records",
]
