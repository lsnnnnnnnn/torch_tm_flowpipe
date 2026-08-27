"""Bit-exact accepted-state snapshots for adaptive/fixed replay gates."""
from __future__ import annotations

import hashlib
import json
import struct
from typing import Any, Mapping, Sequence

import torch

from .fixed_support_outward import OutwardIntervalTensor
from .interval import Interval
from .structured_remainder import StructuredRemainderState
from .tm_vector import TMVector


STATE_EQUALITY_SCHEMA = "torch_tm_flowpipe_adaptive_fixed_state_equality_v1"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous().reshape(-1)
    values = tensor.tolist()
    formats = {
        torch.float64: "=d",
        torch.float32: "=f",
        torch.int64: "=q",
        torch.int32: "=i",
        torch.int16: "=h",
        torch.int8: "=b",
        torch.uint8: "=B",
        torch.bool: "=?",
    }
    try:
        format_string = formats[tensor.dtype]
    except KeyError as exc:
        raise ValueError(
            f"unsupported accepted-state tensor dtype: {tensor.dtype}"
        ) from exc
    return b"".join(struct.pack(format_string, item) for item in values)


def _tensor_record(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    raw = _tensor_bytes(value)
    record: dict[str, Any] = {
        "kind": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "raw_bytes_hex": raw.hex(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }
    flat = tensor.reshape(-1).tolist()
    if tensor.dtype in {torch.float32, torch.float64}:
        record["values_hex"] = [float(item).hex() for item in flat]
    else:
        record["values"] = flat
    return record


def _python_scalar_record(value: bool | int | float | str | None) -> dict[str, Any]:
    if isinstance(value, bool):
        raw = struct.pack("=?", value)
        dtype = "python_bool"
        encoded: Any = value
    elif isinstance(value, int):
        raw = str(value).encode("ascii")
        dtype = "python_int"
        encoded = value
    elif isinstance(value, float):
        raw = struct.pack("=d", value)
        dtype = "python_float_binary64"
        encoded = value.hex()
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        dtype = "python_str"
        encoded = value
    elif value is None:
        raw = b"null"
        dtype = "python_none"
        encoded = None
    else:  # pragma: no cover - callers constrain scalar types
        raise TypeError(type(value).__name__)
    return {
        "kind": "scalar",
        "shape": [],
        "dtype": dtype,
        "device": "host",
        "value": encoded,
        "raw_bytes_hex": raw.hex(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _add_scalar(
    fields: dict[str, dict[str, Any]], path: str, value: bool | int | float | str | None
) -> None:
    fields[path] = _python_scalar_record(value)


def _add_interval(
    fields: dict[str, dict[str, Any]], path: str, value: Interval
) -> None:
    fields[f"{path}.lo"] = _tensor_record(torch.as_tensor(value.lo))
    fields[f"{path}.hi"] = _tensor_record(torch.as_tensor(value.hi))


def _add_tmvector(
    fields: dict[str, dict[str, Any]], path: str, value: TMVector | None
) -> None:
    _add_scalar(fields, f"{path}.present", value is not None)
    if value is None:
        return
    _add_scalar(fields, f"{path}.length", len(value))
    for model_index, model in enumerate(value):
        model_path = f"{path}.models[{model_index}]"
        _add_scalar(fields, f"{model_path}.n_vars", model.n_vars)
        _add_scalar(fields, f"{model_path}.order", model.order)
        split = model.truncation_range_split
        _add_scalar(
            fields,
            f"{model_path}.truncation_range_split",
            None if split is None else int(split),
        )
        for exponent, coefficient in sorted(model.polynomial.terms.items()):
            exponent_name = ",".join(str(item) for item in exponent)
            fields[f"{model_path}.terms[{exponent_name}]"] = _tensor_record(
                torch.as_tensor(coefficient)
            )
        _add_interval(fields, f"{model_path}.remainder", model.remainder)
        for domain_index, interval in enumerate(model.domain):
            _add_interval(fields, f"{model_path}.domain[{domain_index}]", interval)


def _add_float_tree(
    fields: dict[str, dict[str, Any]], path: str, value: Any
) -> None:
    if isinstance(value, torch.Tensor):
        fields[path] = _tensor_record(value)
    elif isinstance(value, (bool, int, float, str)) or value is None:
        _add_scalar(fields, path, value)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _add_float_tree(fields, f"{path}.{key}", value[key])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        _add_scalar(fields, f"{path}.length", len(value))
        for index, item in enumerate(value):
            _add_float_tree(fields, f"{path}[{index}]", item)
    else:
        raise TypeError(f"unsupported accepted-state leaf at {path}: {type(value).__name__}")


def _add_structured_state(
    fields: dict[str, dict[str, Any]],
    path: str,
    state: StructuredRemainderState | None,
) -> None:
    _add_scalar(fields, f"{path}.present", state is not None)
    if state is None:
        return
    for name in state.__dataclass_fields__:
        value = getattr(state, name)
        field_path = f"{path}.{name}"
        if isinstance(value, torch.Tensor):
            fields[field_path] = _tensor_record(value)
        elif isinstance(value, int):
            _add_scalar(fields, field_path, value)
        elif value is None:
            _add_scalar(fields, field_path, None)
        else:
            raise TypeError(
                f"unsupported structured state field {name}: {type(value).__name__}"
            )


def _add_normal_state(
    fields: dict[str, dict[str, Any]], path: str, state: Any | None
) -> None:
    _add_scalar(fields, f"{path}.present", state is not None)
    if state is None:
        return
    _add_tmvector(fields, f"{path}.tmv_pre", state.tmv_pre)
    _add_tmvector(fields, f"{path}.tmv_right", state.tmv_right)
    _add_tmvector(fields, f"{path}.complete_initial_tm", state.complete_initial_tm)
    for index, interval in enumerate(state.domain):
        _add_interval(fields, f"{path}.domain[{index}]", interval)
    _add_float_tree(fields, f"{path}.center", state.center)
    _add_float_tree(fields, f"{path}.scales", state.scales)
    _add_scalar(fields, f"{path}.step_index", state.step_index)
    _add_scalar(
        fields, f"{path}.symbolic_queue_max_size", state.symbolic_queue_max_size
    )
    _add_scalar(fields, f"{path}.symbolic_queue_present", state.symbolic_queue is not None)
    queue = state.symbolic_queue
    if queue is not None:
        _add_scalar(fields, f"{path}.symbolic_queue.owner_schema", queue.owner_schema)
        _add_scalar(fields, f"{path}.symbolic_queue.max_size", queue.max_size)
        _add_scalar(fields, f"{path}.symbolic_queue.generation", queue.generation)
        _add_scalar(
            fields,
            f"{path}.symbolic_queue.accepted_boundary_index",
            queue.accepted_boundary_index,
        )
        _add_scalar(fields, f"{path}.symbolic_queue.reset_count", queue.reset_count)
        _add_float_tree(fields, f"{path}.symbolic_queue.scalars", queue.scalars)
        _add_float_tree(
            fields,
            f"{path}.symbolic_queue.owner_generations",
            queue.owner_generations,
        )
        _add_float_tree(
            fields,
            f"{path}.symbolic_queue.owner_boundary_indices",
            queue.owner_boundary_indices,
        )
        for column_index, column in enumerate(queue.J):
            for component_index, interval in enumerate(column):
                _add_interval(
                    fields,
                    f"{path}.symbolic_queue.J[{column_index}][{component_index}]",
                    interval,
                )
        _add_float_tree(fields, f"{path}.symbolic_queue.Phi_L", queue.Phi_L)
        for matrix_index, matrix in enumerate(queue.Phi_L_iv):
            for row_index, row in enumerate(matrix):
                for column_index, interval in enumerate(row):
                    _add_interval(
                        fields,
                        f"{path}.symbolic_queue.Phi_L_iv[{matrix_index}][{row_index}][{column_index}]",
                        interval,
                    )
        for index, interval in enumerate(queue.scalars_iv):
            _add_interval(fields, f"{path}.symbolic_queue.scalars_iv[{index}]", interval)
    initial = state.initial_remainders
    _add_scalar(fields, f"{path}.initial_remainders.present", initial is not None)
    if initial is not None:
        for index, interval in enumerate(initial):
            _add_interval(fields, f"{path}.initial_remainders[{index}]", interval)
    structured = state.structured_remainder_state
    if structured is not None and not isinstance(structured, StructuredRemainderState):
        raise TypeError("accepted normal state contains an unsupported structured state")
    _add_structured_state(fields, f"{path}.structured_remainder", structured)


def _add_outward_interval(
    fields: dict[str, dict[str, Any]], path: str, value: Any | None
) -> None:
    _add_scalar(fields, f"{path}.present", value is not None)
    if value is None:
        return
    if not isinstance(value, OutwardIntervalTensor):
        raise TypeError(f"{path} is not an OutwardIntervalTensor")
    fields[f"{path}.lo"] = _tensor_record(value.lo)
    fields[f"{path}.hi"] = _tensor_record(value.hi)


def accepted_segment_state_snapshot(segment: Any) -> dict[str, Any]:
    """Serialize every accepted-state field used by the S1 commit path."""

    if getattr(segment, "status", None) != "validated":
        raise ValueError("accepted-state equality requires a validated segment")
    fields: dict[str, dict[str, Any]] = {}
    _add_tmvector(fields, "segment.tm", segment.tm)
    _add_tmvector(fields, "segment.final_tm", segment.final_tm)
    _add_tmvector(fields, "segment.reset_tm", segment.reset_tm)
    _add_tmvector(fields, "segment.endpoint_raw_tm", segment.endpoint_raw_tm)
    _add_tmvector(
        fields, "segment.endpoint_tightened_tm", segment.endpoint_tightened_tm
    )
    _add_float_tree(fields, "segment.candidate_remainder", segment.candidate_remainder)
    _add_float_tree(fields, "segment.picard_image_remainder", segment.picard_image_remainder)
    _add_normal_state(fields, "segment.flowstar_normal_state", segment.flowstar_normal_state)
    structured = segment.structured_state_after
    if structured is not None and not isinstance(structured, StructuredRemainderState):
        raise TypeError("accepted segment contains an unsupported structured poststate")
    _add_structured_state(fields, "segment.structured_state_after", structured)
    for name in (
        "endpoint_total_structured_remainder",
        "tube_total_structured_remainder",
        "endpoint_ordinary_remainder",
        "tube_ordinary_remainder",
        "endpoint_total_remainder",
        "tube_total_remainder",
    ):
        _add_outward_interval(fields, f"segment.{name}", getattr(segment, name))
    for name in ("endpoint_publication_mask", "tube_publication_mask"):
        value = getattr(segment, name)
        _add_scalar(fields, f"segment.{name}.present", value is not None)
        if value is not None:
            fields[f"segment.{name}"] = _tensor_record(value)
    ledger = segment.validated_remainder_ledger
    _add_scalar(fields, "segment.validated_remainder_ledger.present", ledger is not None)
    if ledger is not None:
        entries = getattr(ledger, "entries", None)
        if not isinstance(entries, Mapping):
            raise TypeError("validated remainder ledger does not expose entries")
        for category in sorted(entries):
            lo, hi = entries[category]
            fields[
                f"segment.validated_remainder_ledger.{category}.lo"
            ] = _tensor_record(lo)
            fields[
                f"segment.validated_remainder_ledger.{category}.hi"
            ] = _tensor_record(hi)
    ordered = {path: fields[path] for path in sorted(fields)}
    return {
        "schema": STATE_EQUALITY_SCHEMA,
        "field_count": len(ordered),
        "fields": ordered,
        "state_sha256": hashlib.sha256(_canonical_bytes(ordered)).hexdigest(),
    }


def compare_accepted_segment_states(natural: Any, fixed: Any) -> dict[str, Any]:
    """Compare accepted states exactly and report the first mismatch fail closed."""

    natural_snapshot = accepted_segment_state_snapshot(natural)
    fixed_snapshot = accepted_segment_state_snapshot(fixed)
    natural_fields = natural_snapshot["fields"]
    fixed_fields = fixed_snapshot["fields"]
    all_paths = sorted(set(natural_fields) | set(fixed_fields))
    comparisons: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    for path in all_paths:
        left = natural_fields.get(path)
        right = fixed_fields.get(path)
        equal = left == right
        row = {
            "path": path,
            "relation": "bit_exact",
            "equal": equal,
            "natural": left,
            "fixed": right,
        }
        comparisons.append(row)
        if not equal and first_mismatch is None:
            first_mismatch = row
    return {
        "schema": STATE_EQUALITY_SCHEMA,
        "status": "pass" if first_mismatch is None else "fail",
        "comparison_policy": "bit_exact_for_all_registered_accepted_state_fields",
        "outward_containment_relations": [],
        "field_count": len(all_paths),
        "natural_state_sha256": natural_snapshot["state_sha256"],
        "fixed_state_sha256": fixed_snapshot["state_sha256"],
        "first_mismatch": first_mismatch,
        "comparisons": comparisons,
    }


__all__ = [
    "STATE_EQUALITY_SCHEMA",
    "accepted_segment_state_snapshot",
    "compare_accepted_segment_states",
]
