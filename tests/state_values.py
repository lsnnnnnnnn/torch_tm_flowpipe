from dataclasses import fields, is_dataclass

import torch

TIMING_FIELDS = frozenset({'host_to_device_s', 'dense_kernel_s', 'device_to_host_s'})


def canonical(value):
    """All numerical payloads and decisions; omit only three elapsed counters."""
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
        return {'tensor': str(value.dtype), 'shape': list(value.shape),
                'values': [v.hex() for v in values] if value.is_floating_point() else values}
    if isinstance(value, float):
        return {'float_hex': value.hex()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if is_dataclass(value):
        return {'type': type(value).__name__, 'fields': {f.name: canonical(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, dict):
        return {'items': [[canonical(k), canonical(v)] for k, v in value.items() if k not in TIMING_FIELDS]}
    if isinstance(value, (tuple, list)):
        return [canonical(x) for x in value]
    raise TypeError(f'unrecorded boundary payload type: {type(value)}')
