"""Read-only, common-schema traces for raw Taylor-model remainder expressions.

The recorder deliberately owns no solver state.  Callers pass already-computed
models and intervals after each production operation; serialization and hashing
therefore cannot feed back into a validation decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch


SCHEMA = "torch_tm_flowpipe_raw_remainder_expression_tree_v1"
NODE_FIELDS = (
    "run_id",
    "tool",
    "source_commit",
    "binary_sha256",
    "checkpoint_sha256",
    "t_pre_decimal",
    "t_pre_hex",
    "h_decimal",
    "h_hex",
    "picard_iteration",
    "state_component",
    "expression_node_id",
    "parent_node_ids",
    "operation",
    "polynomial_order_before",
    "polynomial_order_after",
    "retained_support_sha256",
    "dropped_support_sha256",
    "polynomial_interval",
    "remainder_input_intervals",
    "remainder_output_interval",
    "roundoff_interval",
    "cutoff_interval",
    "integration_overflow_interval",
    "normalization_scale",
    "target_interval",
    "subset_margin",
    "decision",
)


def float_record(value: float) -> dict[str, str]:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("trace numbers must be finite")
    return {"decimal": repr(number), "hex": number.hex()}


def interval_record(lo: float, hi: float) -> dict[str, Any]:
    lo_f = float(lo)
    hi_f = float(hi)
    if not math.isfinite(lo_f) or not math.isfinite(hi_f) or lo_f > hi_f:
        raise ValueError(f"invalid finite interval [{lo_f!r}, {hi_f!r}]")
    return {
        "lo": float_record(lo_f),
        "hi": float_record(hi_f),
        "width": float_record(hi_f - lo_f),
    }


def tensor_interval_record(lo: torch.Tensor, hi: torch.Tensor) -> dict[str, Any]:
    lo_t = lo.detach().cpu().reshape(-1)
    hi_t = hi.detach().cpu().reshape(-1)
    if lo_t.numel() != 1 or hi_t.numel() != 1:
        raise ValueError("a raw-expression node must record exactly one state component")
    return interval_record(float(lo_t[0]), float(hi_t[0]))


def _sha_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    descriptor = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + tensor.numpy().tobytes()).hexdigest()


def _support_sha(model: Any, *, retained: bool) -> str:
    coeffs = model.poly.coeffs.detach()
    active = torch.any(coeffs != 0, dim=(0, 1))
    exponents = model.poly.basis.exponents[active].detach().cpu().contiguous()
    payload = {
        "kind": "retained" if retained else "dropped-not-materialized",
        "exponents": exponents.tolist() if retained else [],
        "coefficient_sha256": _sha_tensor(coeffs) if retained else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _zero_interval() -> dict[str, Any]:
    return interval_record(0.0, 0.0)


def _ledger_interval(model: Any, category: str) -> dict[str, Any]:
    entry = model.ledger.entries.get(category)
    if entry is None:
        return _zero_interval()
    return tensor_interval_record(*entry)


@dataclass
class RawRemainderTraceRecorder:
    run_id: str
    tool: str
    source_commit: str
    binary_sha256: str
    checkpoint_sha256: str
    t_pre: float
    h: float
    picard_iteration: int
    normalization_scale: Sequence[float]
    target_intervals: Sequence[tuple[float, float]]
    nodes: list[dict[str, Any]] = field(default_factory=list)
    component_roots: dict[int, str] = field(default_factory=dict)
    _operation_counts: dict[str, int] = field(default_factory=dict)

    def _next_id(self, operation: str, component: int) -> str:
        count = self._operation_counts.get(operation, 0) + 1
        self._operation_counts[operation] = count
        semantic: dict[tuple[str, int], str] = {
            # PolynomialODE preserves the canonical distributed form
            # y - x - x^2*y.  These are production operations, not a
            # post-hoc rewrite to Flow*'s factored (1-x^2)*y-x tree.
            ("subtract", 1): "y_rhs.y_minus_x",
            ("multiply", 1): "y_rhs.x_squared",
            ("multiply", 2): "y_rhs.x_squared_times_y",
            ("subtract", 2): "y_rhs.distributed_final",
        }
        label = semantic.get((operation, count), f"{operation}_{count:03d}")
        return f"torch.i{self.picard_iteration}.c{component}.{label}"

    def add_model_node(
        self,
        *,
        operation: str,
        component: int,
        model: Any,
        parents: Sequence[str] = (),
        remainder_inputs: Sequence[tuple[torch.Tensor, torch.Tensor]] = (),
        order_before: int | None = None,
        order_after: int | None = None,
        node_id: str | None = None,
        polynomial_interval: tuple[torch.Tensor, torch.Tensor] | None = None,
        roundoff: tuple[float, float] = (0.0, 0.0),
        subset_margin: float | None = None,
        decision: str = "not_applicable",
    ) -> str:
        identifier = node_id or self._next_id(operation, int(component))
        if any(row["expression_node_id"] == identifier for row in self.nodes):
            raise ValueError(f"duplicate raw-remainder node id: {identifier}")
        if polynomial_interval is None:
            polynomial_interval = model.poly.range_bound(
                model.domain_lo,
                model.domain_hi,
                policy=model.range_policy,
                context="raw_expression_observer_polynomial",
                trace=[],
            )
            polynomial = tensor_interval_record(*polynomial_interval)
        else:
            polynomial = tensor_interval_record(*polynomial_interval)
        node = {
            "run_id": self.run_id,
            "tool": self.tool,
            "source_commit": self.source_commit,
            "binary_sha256": self.binary_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "t_pre_decimal": repr(float(self.t_pre)),
            "t_pre_hex": float(self.t_pre).hex(),
            "h_decimal": repr(float(self.h)),
            "h_hex": float(self.h).hex(),
            "picard_iteration": int(self.picard_iteration),
            "state_component": int(component),
            "expression_node_id": identifier,
            "parent_node_ids": list(parents),
            "operation": operation,
            "polynomial_order_before": order_before,
            "polynomial_order_after": order_after,
            "retained_support_sha256": _support_sha(model, retained=True),
            "dropped_support_sha256": _support_sha(model, retained=False),
            "polynomial_interval": polynomial,
            "remainder_input_intervals": [tensor_interval_record(lo, hi) for lo, hi in remainder_inputs],
            "remainder_output_interval": tensor_interval_record(model.rem_lo, model.rem_hi),
            "roundoff_interval": interval_record(*roundoff),
            "cutoff_interval": _ledger_interval(model, "cutoff"),
            "integration_overflow_interval": _ledger_interval(model, "integration_overflow"),
            "normalization_scale": float_record(float(self.normalization_scale[int(component)])),
            "target_interval": interval_record(*self.target_intervals[int(component)]),
            "subset_margin": None if subset_margin is None else float_record(float(subset_margin)),
            "decision": decision,
            "multiplication_remainder_components": {
                "polynomial_times_polynomial_dropped": _ledger_interval(model, "polynomial_truncation"),
                "polynomial_times_remainder": _ledger_interval(model, "poly_times_remainder"),
                "remainder_times_polynomial": _ledger_interval(model, "remainder_times_poly"),
                "remainder_times_remainder": _ledger_interval(model, "remainder_times_remainder"),
                "coefficient_interval_uncertainty": _zero_interval(),
                "interval_evaluation_dependency": _zero_interval(),
                "outward_rounding": interval_record(*roundoff),
            },
        }
        missing = set(NODE_FIELDS) - set(node)
        if missing:
            raise AssertionError(f"trace implementation omitted common fields: {sorted(missing)}")
        self.nodes.append(node)
        return identifier

    def register_component_root(self, component: int, node_id: str) -> None:
        self.component_roots[int(component)] = node_id

    def finalize_decision(self, margins: Sequence[float], accepted: bool) -> None:
        decision = "accept" if accepted else "reject"
        for node in self.nodes:
            component = int(node["state_component"])
            node["subset_margin"] = float_record(float(margins[component]))
            node["decision"] = decision

    def artifact(self) -> dict[str, Any]:
        validate_expression_dag(self.nodes)
        return {
            "schema": SCHEMA,
            "node_fields": list(NODE_FIELDS),
            "nodes": self.nodes,
        }


def validate_expression_dag(nodes: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for index, node in enumerate(nodes):
        missing = set(NODE_FIELDS) - set(node)
        if missing:
            raise ValueError(f"node {index} missing fields: {sorted(missing)}")
        node_id = str(node["expression_node_id"])
        if not node_id or node_id in seen:
            raise ValueError(f"invalid or duplicate node id: {node_id!r}")
        parents = [str(value) for value in node["parent_node_ids"]]
        unresolved = [parent for parent in parents if parent not in seen]
        if unresolved:
            raise ValueError(f"node {node_id} has non-prior parents: {unresolved}")
        seen.add(node_id)


__all__ = [
    "NODE_FIELDS",
    "RawRemainderTraceRecorder",
    "SCHEMA",
    "float_record",
    "interval_record",
    "validate_expression_dag",
]
