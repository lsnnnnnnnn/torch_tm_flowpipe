"""Safe, inspectable checkpoint I/O for terminal flowpipe replay.

The payload is canonical JSON and stores every floating-point scalar using its
exact hexadecimal representation.  Loading never invokes pickle or user code.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .flowpipe import FlowstarNormalFlowpipeState
from .interval import Interval
from .polynomial import Polynomial
from .taylor_model import TaylorModel
from .tm_vector import TMVector


SCHEMA = "torch_tm_flowpipe_terminal_checkpoint_v1"
PAYLOAD_NAME = "terminal_state.json"
MANIFEST_NAME = "terminal_state_manifest.json"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _scalar_hex(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("checkpoint scalars must contain exactly one value")
        scalar = float(value.detach().cpu())
    else:
        # Routing a Python float through torch.as_tensor would use the
        # process-wide default dtype (commonly float32) and erase low bits.
        scalar = float(value)
    if not torch.isfinite(torch.as_tensor(scalar, dtype=torch.float64)):
        raise ValueError("checkpoint state contains a non-finite scalar")
    return scalar.hex()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    supported = {"float64": torch.float64, "float32": torch.float32}
    try:
        return supported[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported checkpoint dtype: {name}") from exc


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    raise TypeError(f"unsupported checkpoint diagnostic value: {type(value).__name__}")


def _encode_interval(interval: Interval) -> dict[str, str]:
    return {"lo_hex": _scalar_hex(interval.lo), "hi_hex": _scalar_hex(interval.hi)}


def _decode_interval(value: Mapping[str, Any], *, dtype: torch.dtype) -> Interval:
    try:
        lo = float.fromhex(str(value["lo_hex"]))
        hi = float.fromhex(str(value["hi_hex"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid checkpoint interval encoding") from exc
    return Interval(torch.tensor(lo, dtype=dtype), torch.tensor(hi, dtype=dtype))


def _encode_tmvector(tmv: TMVector) -> dict[str, Any]:
    if not isinstance(tmv, TMVector) or not tmv.models:
        raise TypeError("checkpoint requires a non-empty TMVector")
    dtype = tmv[0].polynomial.dtype
    device = tmv[0].polynomial.device
    models: list[dict[str, Any]] = []
    for model in tmv:
        if model.polynomial.dtype != dtype or model.polynomial.device != device:
            raise ValueError("checkpoint TMVector has mixed polynomial dtype/device")
        models.append(
            {
                "n_vars": model.n_vars,
                "order": model.order,
                "truncation_range_split": model.truncation_range_split,
                "terms": [
                    {"exponent": list(exponent), "coefficient_hex": _scalar_hex(coefficient)}
                    for exponent, coefficient in sorted(model.polynomial.terms.items())
                ],
                "remainder": _encode_interval(model.remainder),
                "domain": [_encode_interval(interval) for interval in model.domain],
            }
        )
    return {
        "dtype": _dtype_name(dtype),
        "source_device": str(device),
        "models": models,
    }


def _decode_tmvector(value: Mapping[str, Any]) -> TMVector:
    dtype = _dtype_from_name(str(value.get("dtype", "")))
    models_value = value.get("models")
    if not isinstance(models_value, list) or not models_value:
        raise ValueError("checkpoint TMVector models must be a non-empty list")
    models: list[TaylorModel] = []
    for encoded in models_value:
        if not isinstance(encoded, Mapping):
            raise ValueError("invalid checkpoint Taylor model")
        n_vars = int(encoded.get("n_vars", -1))
        terms_value = encoded.get("terms")
        if not isinstance(terms_value, list):
            raise ValueError("checkpoint polynomial terms must be a list")
        terms: dict[tuple[int, ...], torch.Tensor] = {}
        for term in terms_value:
            if not isinstance(term, Mapping):
                raise ValueError("invalid checkpoint polynomial term")
            exponent = tuple(int(item) for item in term.get("exponent", []))
            if len(exponent) != n_vars:
                raise ValueError("checkpoint exponent dimension mismatch")
            try:
                coefficient = float.fromhex(str(term["coefficient_hex"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid checkpoint coefficient encoding") from exc
            terms[exponent] = torch.tensor(coefficient, dtype=dtype)
        domain_value = encoded.get("domain")
        if not isinstance(domain_value, list) or len(domain_value) != n_vars:
            raise ValueError("checkpoint domain dimension mismatch")
        domain = [_decode_interval(item, dtype=dtype) for item in domain_value]
        remainder_value = encoded.get("remainder")
        if not isinstance(remainder_value, Mapping):
            raise ValueError("invalid checkpoint remainder")
        models.append(
            TaylorModel(
                Polynomial(terms, n_vars=n_vars),
                _decode_interval(remainder_value, dtype=dtype),
                domain,
                order=int(encoded["order"]),
                truncation_range_split=encoded.get("truncation_range_split"),
            )
        )
    return TMVector(models)


def _tmvector_hashes(encoded: Mapping[str, Any]) -> dict[str, str]:
    models = encoded.get("models", [])
    coefficient_payload = [
        [term.get("coefficient_hex") for term in model.get("terms", [])]
        for model in models
    ]
    exponent_payload = [
        [term.get("exponent") for term in model.get("terms", [])]
        for model in models
    ]
    domain_payload = [model.get("domain", []) for model in models]
    return {
        "coefficient_sha256": _sha256_json(coefficient_payload),
        "exponent_support_sha256": _sha256_json(exponent_payload),
        "domain_sha256": _sha256_json(domain_payload),
        "tmvector_sha256": _sha256_json(encoded),
    }


def tmvector_hashes(tmv: TMVector) -> dict[str, str]:
    """Return stable coefficient, support, domain, and complete TM hashes."""
    return _tmvector_hashes(_encode_tmvector(tmv))


def _encode_normal_state(state: FlowstarNormalFlowpipeState) -> dict[str, Any]:
    if state.symbolic_queue is not None:
        raise ValueError("non-empty symbolic queue checkpointing is not supported by this canonical lane")
    return {
        "tmv_pre": _encode_tmvector(state.tmv_pre),
        "tmv_right": _encode_tmvector(state.tmv_right),
        "domain": [_encode_interval(interval) for interval in state.domain],
        "center_hex": [_scalar_hex(value) for value in state.center],
        "scales_hex": [_scalar_hex(value) for value in state.scales],
        "step_index": int(state.step_index),
        "diagnostics": _json_safe(state.diagnostics),
        "symbolic_queue_present": False,
        "symbolic_queue_max_size": int(state.symbolic_queue_max_size),
        "initial_remainders": (
            [_encode_interval(interval) for interval in state.initial_remainders]
            if state.initial_remainders is not None
            else None
        ),
    }


def _decode_normal_state(value: Mapping[str, Any]) -> FlowstarNormalFlowpipeState:
    if bool(value.get("symbolic_queue_present")):
        raise ValueError("symbolic queue payload is unsupported and cannot be silently dropped")
    tmv_pre_value = value.get("tmv_pre")
    tmv_right_value = value.get("tmv_right")
    if not isinstance(tmv_pre_value, Mapping) or not isinstance(tmv_right_value, Mapping):
        raise ValueError("checkpoint normal state is missing TM vectors")
    tmv_pre = _decode_tmvector(tmv_pre_value)
    tmv_right = _decode_tmvector(tmv_right_value)
    dtype = tmv_pre[0].polynomial.dtype
    domain_value = value.get("domain")
    if not isinstance(domain_value, list):
        raise ValueError("checkpoint normal domain must be a list")
    initial_value = value.get("initial_remainders")
    initial = None
    if initial_value is not None:
        if not isinstance(initial_value, list):
            raise ValueError("checkpoint initial remainders must be a list or null")
        initial = tuple(_decode_interval(item, dtype=dtype) for item in initial_value)
    try:
        center = [float.fromhex(str(item)) for item in value.get("center_hex", [])]
        scales = [float.fromhex(str(item)) for item in value.get("scales_hex", [])]
    except ValueError as exc:
        raise ValueError("invalid checkpoint center/scale encoding") from exc
    return FlowstarNormalFlowpipeState(
        tmv_pre=tmv_pre,
        tmv_right=tmv_right,
        domain=[_decode_interval(item, dtype=dtype) for item in domain_value],
        center=center,
        scales=scales,
        step_index=int(value.get("step_index", -1)),
        diagnostics=value.get("diagnostics") if isinstance(value.get("diagnostics"), Mapping) else None,
        symbolic_queue=None,
        symbolic_queue_max_size=int(value.get("symbolic_queue_max_size", 100)),
        initial_remainders=initial,
    )


@dataclass(frozen=True)
class TerminalCheckpoint:
    current: TMVector
    normal_state: FlowstarNormalFlowpipeState
    scheduler: Mapping[str, Any]
    contract: Mapping[str, Any]
    provenance: Mapping[str, Any]
    manifest: Mapping[str, Any]


def save_terminal_checkpoint(
    output_dir: Path,
    *,
    current: TMVector,
    normal_state: FlowstarNormalFlowpipeState,
    scheduler: Mapping[str, Any],
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write a new checkpoint directory, refusing any pre-existing content."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty checkpoint directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    current_encoded = _encode_tmvector(current)
    normal_encoded = _encode_normal_state(normal_state)
    payload = {
        "schema": SCHEMA,
        "current": current_encoded,
        "normal_state": normal_encoded,
        "scheduler": _json_safe(scheduler),
        "contract": _json_safe(contract),
        "provenance": _json_safe(provenance),
    }
    payload_bytes = _canonical_bytes(payload)
    hashes = {
        "current": _tmvector_hashes(current_encoded),
        "normal_tmv_pre": _tmvector_hashes(normal_encoded["tmv_pre"]),
        "normal_tmv_right": _tmvector_hashes(normal_encoded["tmv_right"]),
    }
    manifest = {
        "schema": SCHEMA,
        "payload_file": PAYLOAD_NAME,
        "payload_sha256": _sha256_bytes(payload_bytes),
        "full_checkpoint_sha256": _sha256_json({"payload_sha256": _sha256_bytes(payload_bytes), "hashes": hashes}),
        "contract_sha256": _sha256_json(payload["contract"]),
        "dtype": current_encoded["dtype"],
        "source_device": current_encoded["source_device"],
        "hashes": hashes,
        "safe_loader": "canonical_json_float_hex_no_pickle",
        "pytorch_version": torch.__version__,
    }
    (output_dir / PAYLOAD_NAME).write_bytes(payload_bytes)
    (output_dir / MANIFEST_NAME).write_bytes(_canonical_bytes(manifest))
    return manifest


def load_terminal_checkpoint(
    checkpoint: Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
    expected_order: int | None = None,
    expected_dtype: str | None = None,
) -> TerminalCheckpoint:
    """Load a checkpoint after verifying its manifest and declared contract."""
    checkpoint = Path(checkpoint)
    directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
    payload_path = checkpoint if checkpoint.is_file() and checkpoint.name == PAYLOAD_NAME else directory / PAYLOAD_NAME
    manifest_path = directory / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_bytes = payload_path.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid terminal checkpoint files: {exc}") from exc
    if manifest.get("schema") != SCHEMA:
        raise ValueError("terminal checkpoint schema mismatch")
    actual_payload_sha = _sha256_bytes(payload_bytes)
    if actual_payload_sha != manifest.get("payload_sha256"):
        raise ValueError("terminal checkpoint payload SHA256 mismatch")
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("terminal checkpoint payload is not valid JSON") from exc
    if payload.get("schema") != SCHEMA:
        raise ValueError("terminal checkpoint payload schema mismatch")
    contract_value = payload.get("contract")
    if not isinstance(contract_value, Mapping):
        raise ValueError("terminal checkpoint contract mapping is missing")
    if _sha256_json(contract_value) != manifest.get("contract_sha256"):
        raise ValueError("terminal checkpoint manifest contract SHA256 mismatch")
    declared_hashes = manifest.get("hashes")
    expected_full_sha = _sha256_json({"payload_sha256": actual_payload_sha, "hashes": declared_hashes})
    if expected_full_sha != manifest.get("full_checkpoint_sha256"):
        raise ValueError("terminal checkpoint full SHA256 mismatch")
    if expected_contract is not None and _sha256_json(_json_safe(expected_contract)) != manifest.get("contract_sha256"):
        raise ValueError("terminal checkpoint contract mismatch")
    current_value = payload.get("current")
    normal_value = payload.get("normal_state")
    if not isinstance(current_value, Mapping) or not isinstance(normal_value, Mapping):
        raise ValueError("terminal checkpoint is missing state payloads")
    if expected_dtype is not None and str(current_value.get("dtype")) != str(expected_dtype):
        raise ValueError("terminal checkpoint dtype mismatch")
    current = _decode_tmvector(current_value)
    normal_state = _decode_normal_state(normal_value)
    if expected_order is not None:
        models = [*current.models, *normal_state.tmv_pre.models, *normal_state.tmv_right.models]
        if any(model.order != int(expected_order) for model in models):
            raise ValueError("terminal checkpoint order mismatch")
    actual_hashes = {
        "current": tmvector_hashes(current),
        "normal_tmv_pre": tmvector_hashes(normal_state.tmv_pre),
        "normal_tmv_right": tmvector_hashes(normal_state.tmv_right),
    }
    if actual_hashes != manifest.get("hashes"):
        raise ValueError("terminal checkpoint reconstructed state hash mismatch")
    scheduler = payload.get("scheduler")
    contract = contract_value
    provenance = payload.get("provenance")
    if not isinstance(scheduler, Mapping) or not isinstance(contract, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("terminal checkpoint metadata mappings are missing")
    return TerminalCheckpoint(current, normal_state, scheduler, contract, provenance, manifest)


__all__ = [
    "MANIFEST_NAME",
    "PAYLOAD_NAME",
    "SCHEMA",
    "TerminalCheckpoint",
    "load_terminal_checkpoint",
    "save_terminal_checkpoint",
    "tmvector_hashes",
]
