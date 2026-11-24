"""Factor construction from the retained transferred sparse executor."""
import torch
from flowstar_gpu import interval as iv


def _sr_linear_prepare(x0f, scalars, scalars_iv, pos_t, present, strict):
    """Pure SR inputs; callers own all persistent state and graph metadata."""
    B, n, _ = x0f.shape
    dt, dev = x0f.dtype, x0f.device
    x0p = torch.cat([x0f, torch.zeros(B, n, 1, dtype=dt, device=dev)], dim=-1)
    lin_a = x0p[..., pos_t]
    x0_other = x0f.clone()
    x0_other.index_fill_(-1, present, 0.0)
    phi_i = lin_a * scalars.unsqueeze(1)  # [B, n, n]
    phi_i_iv = None
    if strict:
        phi_i_iv = iv.mul(
            iv.from_point(lin_a),
            (
                scalars_iv.unsqueeze(1)
                if scalars_iv is not None
                else iv.from_point(scalars.unsqueeze(1))
            ),
        )
    return lin_a, x0_other, phi_i, phi_i_iv
