#!/usr/bin/env python3
"""Directed affine ReLU enclosure for the fixed DP controller and CROWN map.

For every hidden/output neuron this keeps ``A*x + b + R``, where A, b, and R
are intervals and x is the original four-state box. All linear products and
reductions use the engine's directed interval primitives. A crossing ReLU uses
an arbitrary stored binary64 slope alpha in [0,1], with the exact residual
identity ``relu(z) - alpha*z in [0,max(-alpha*l,(1-alpha)*u)]`` for z in [l,u].
"""


def _summary(values):
    return {"max": float(values.max().item()), "mean": float(values.mean().item())}


def affine_residual_interval(hull, T, layers):
    import torch
    from flowstar_gpu import interval as iv

    if hull.ndim != 3 or hull.shape[1:] != (4, 2):
        raise ValueError("DP affine controller expects [B,4,2] hull")
    if T.shape != (hull.shape[0], 2, 4):
        raise ValueError("DP affine controller expects [B,2,4] CROWN map")
    iv.assert_valid(hull, "DP affine controller input")
    B = hull.shape[0]
    ident = torch.eye(4, dtype=torch.float64, device=hull.device).expand(B, -1, -1)
    A = iv.from_point(ident)
    b = iv.from_point(torch.zeros((B, 4), dtype=torch.float64, device=hull.device))
    R = b.clone()
    diagnostics = []

    for layer_index, (weight, bias) in enumerate(layers):
        if weight.shape != (A.shape[1], bias.numel()):
            raise ValueError("DP affine controller layer dimension mismatch")
        # Current A: [B,input,original-input,2]. Sum across the current input
        # dimension while preserving the original four x-symbols.
        A = iv.dot_point_iv(
            weight.T[None, :, None, :], A.permute(0, 2, 1, 3).unsqueeze(1), dim=-1
        )
        b = iv.add(
            iv.dot_point_iv(weight.T.unsqueeze(0), b.unsqueeze(1), dim=-1),
            iv.from_point(bias).unsqueeze(0),
        )
        R = iv.dot_point_iv(weight.T.unsqueeze(0), R.unsqueeze(1), dim=-1)
        for name, value in (("A", A), ("b", b), ("R", R)):
            iv.assert_valid(value, f"DP affine layer {layer_index + 1} {name}")

        layer_record = {
            "layer": layer_index + 1,
            "A_interval_width": _summary(A[..., 1] - A[..., 0]),
            "b_interval_width": _summary(b[..., 1] - b[..., 0]),
            "R_interval_width_before_relu": _summary(R[..., 1] - R[..., 0]),
        }
        if layer_index < 2:
            # Preactivation range uses interval coefficients, the same original
            # input box, and the already accumulated abstract remainder.
            pre = iv.add(iv.add(iv.sum(iv.mul(A, hull.unsqueeze(1)), dim=-1), b), R)
            iv.assert_valid(pre, f"DP affine preactivation {layer_index + 1}")
            low, high = pre[..., 0], pre[..., 1]
            inactive = high <= 0
            active = low >= 0
            crossing = ~(inactive | active)
            denom = torch.where(crossing, high - low, torch.ones_like(high))
            alpha = torch.where(crossing, high / denom, active.to(torch.float64))
            if not bool(torch.isfinite(alpha).all()) or not bool(((alpha >= 0) & (alpha <= 1)).all()):
                raise FloatingPointError("DP affine ReLU slope left [0,1]")

            # Exact real product bounds for -alpha*l and (1-alpha)*u. The
            # stored alpha is a point constant; 1-alpha is an interval to
            # account for its floating-point subtraction.
            neg_alpha_low = iv.neg(iv.mul_point(iv.from_point(low), alpha))
            one_minus_alpha = iv.sub(iv.from_point(torch.ones_like(alpha)), iv.from_point(alpha))
            one_minus_alpha_high = iv.mul(one_minus_alpha, iv.from_point(high))
            error_upper = torch.maximum(neg_alpha_low[..., 1], one_minus_alpha_high[..., 1])
            error_upper = torch.maximum(error_upper, torch.zeros_like(error_upper))
            e = torch.stack((torch.zeros_like(error_upper), error_upper), dim=-1)

            scaled_A = iv.mul_point(A, alpha.unsqueeze(-1))
            scaled_b = iv.mul_point(b, alpha)
            scaled_R = iv.add(iv.mul_point(R, alpha), e)
            A = torch.where(active[..., None, None], A,
                            torch.where(inactive[..., None, None], torch.zeros_like(A), scaled_A))
            b = torch.where(active[..., None], b,
                            torch.where(inactive[..., None], torch.zeros_like(b), scaled_b))
            R = torch.where(active[..., None], R,
                            torch.where(inactive[..., None], torch.zeros_like(R), scaled_R))
            layer_record["relu_active"] = int(active.sum().item())
            layer_record["relu_inactive"] = int(inactive.sum().item())
            layer_record["relu_crossing"] = int(crossing.sum().item())
            layer_record["relu_error_upper"] = _summary(error_upper)
            layer_record["R_interval_width_after_relu"] = _summary(R[..., 1] - R[..., 0])
            for name, value in (("A", A), ("b", b), ("R", R)):
                iv.assert_valid(value, f"DP affine post-ReLU {layer_index + 1} {name}")
        diagnostics.append(layer_record)

    affine_network = iv.add(iv.add(iv.sum(iv.mul(A, hull.unsqueeze(1)), dim=-1), b), R)
    delta_A = iv.sub(A, iv.from_point(T))
    residual = iv.add(iv.add(iv.sum(iv.mul(delta_A, hull.unsqueeze(1)), dim=-1), b), R)
    iv.assert_valid(affine_network, "DP affine network output")
    iv.assert_valid(residual, "DP affine controller residual")
    diagnostics.append({
        "final_A_minus_T_interval_width": _summary(delta_A[..., 1] - delta_A[..., 0]),
        "final_residual_width": _summary(residual[..., 1] - residual[..., 0]),
    })
    return affine_network, residual, diagnostics


def partitioned_affine_residual_interval(hull, T, layers):
    """Tighten one global T residual over a complete 2^4 input partition."""
    import torch
    from flowstar_gpu import interval as iv

    network, whole, layers_diagnostics = affine_residual_interval(hull, T, layers)
    low, high = hull[..., 0], hull[..., 1]
    midpoint = low + (high - low) * 0.5
    if not bool((torch.isfinite(midpoint) & (low <= midpoint) & (midpoint <= high)).all()):
        raise FloatingPointError("DP controller midpoint failed to partition input box")

    joined = None
    cover_low, cover_high = None, None
    for mask in range(16):
        part = hull.clone()
        for dim in range(4):
            part[:, dim, 1 - ((mask >> dim) & 1)] = midpoint[:, dim]
        iv.assert_valid(part, "DP controller partition")
        cover_low = part[..., 0] if cover_low is None else torch.minimum(cover_low, part[..., 0])
        cover_high = part[..., 1] if cover_high is None else torch.maximum(cover_high, part[..., 1])
        _, sub, _ = affine_residual_interval(part, T, layers)
        joined = sub if joined is None else torch.stack(
            (torch.minimum(joined[..., 0], sub[..., 0]),
             torch.maximum(joined[..., 1], sub[..., 1])), dim=-1
        )
    if not (torch.equal(cover_low, low) and torch.equal(cover_high, high)):
        raise AssertionError("DP controller partitions do not cover the input box")
    iv.assert_valid(joined, "DP controller partition union")
    tight = torch.stack(
        (torch.maximum(whole[..., 0], joined[..., 0]),
         torch.minimum(whole[..., 1], joined[..., 1])), dim=-1
    )
    iv.assert_valid(tight, "DP controller residual intersection")
    whole_width = whole[..., 1] - whole[..., 0]
    tight_width = tight[..., 1] - tight[..., 0]
    diagnostics = {
        "whole_layers": layers_diagnostics,
        "subboxes_per_input_box": 16,
        "global_T_reused": True,
        "whole_width_max": float(whole_width.max().item()),
        "union_width_max": float((joined[..., 1] - joined[..., 0]).max().item()),
        "tight_width_max": float(tight_width.max().item()),
        "improved_outputs": int((tight_width < whole_width).sum().item()),
    }
    return network, tight, diagnostics
