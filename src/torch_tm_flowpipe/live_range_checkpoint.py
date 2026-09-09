"""Safe accepted-boundary checkpoints with lossless live-task diagnostic state.

The established terminal writer preserves the mathematical carry but intentionally
filters private diagnostics and sorts mappings. This opt-in sidecar preserves
those ordinary values and their order, without changing the historical format.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from .terminal_checkpoint import load_terminal_checkpoint, save_terminal_checkpoint


def _encode(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("live checkpoint contains a non-CPU diagnostic tensor")
        return {"tensor":str(value.dtype).removeprefix("torch."), "shape":list(value.shape),
                "values":[float(v).hex() for v in value.flatten().tolist()] if value.is_floating_point()
                         else value.flatten().tolist()}
    if isinstance(value, dict):
        return {"mapping":[[_encode(k), _encode(v)] for k,v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"tuple" if isinstance(value, tuple) else "list":[_encode(v) for v in value]}
    raise TypeError(f"unsupported live checkpoint metadata: {type(value).__name__}")


def _decode(value):
    if not isinstance(value, dict):
        return value
    if "float_hex" in value:
        return float.fromhex(value["float_hex"])
    if "tensor" in value:
        dtypes = {name:getattr(torch,name) for name in ("float64", "float32", "int64", "int32", "bool")}
        dtype = dtypes[value["tensor"]]
        values = [float.fromhex(v) for v in value["values"]] if dtype in {torch.float32,torch.float64} else value["values"]
        return torch.tensor(values,dtype=dtype).reshape(value["shape"])
    if "mapping" in value:
        return {_decode(k):_decode(v) for k,v in value["mapping"]}
    if "tuple" in value:
        return tuple(_decode(v) for v in value["tuple"])
    if "list" in value:
        return [_decode(v) for v in value["list"]]
    raise ValueError("unknown live checkpoint metadata tag")


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_live_range_checkpoint(path, task, *, scheduler, contract, provenance):
    current, state = task.checkpoint_state()
    path = Path(path)
    save_terminal_checkpoint(path, current=current, normal_state=state, scheduler=scheduler,
                             contract=contract, provenance=provenance)
    payload = dict(schema="live-range-checkpoint-v1", task=task.task_id,
        previous_run=task.service.run_id, previous_epoch=task.epoch, generation=task.generation,
        diagnostics=_encode(state.diagnostics),
        terminal_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()})
    wrapper = dict(payload=payload, sha256=_digest(payload))
    (path/"live_range_metadata.json").write_text(json.dumps(wrapper, indent=2, allow_nan=False)+"\n")
    return path


def load_live_range_checkpoint(path, service):
    path = Path(path)
    wrapper = json.loads((path/"live_range_metadata.json").read_text())
    payload = wrapper["payload"]
    if payload["schema"] != "live-range-checkpoint-v1" or _digest(payload) != wrapper["sha256"]:
        raise ValueError("live checkpoint metadata checksum mismatch")
    for name, checksum in payload["terminal_files"].items():
        if Path(name).name != name or hashlib.sha256((path/name).read_bytes()).hexdigest() != checksum:
            raise ValueError("live checkpoint terminal payload mismatch")
    loaded = load_terminal_checkpoint(path)
    state = replace(loaded.normal_state, diagnostics=_decode(payload["diagnostics"]))
    if state.step_index != payload["generation"]:
        raise ValueError("live checkpoint accepted boundary version mismatch")
    # Advance even in a fresh process whose caller explicitly reused a run ID.
    return service.register(payload["task"], (loaded.current, state), generation=payload["generation"],
                            minimum_epoch=payload["previous_epoch"]+1)
