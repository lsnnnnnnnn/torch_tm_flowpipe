"""Conditional control-remainder contraction, retaining the injected polynomial.

Both the existing control TM and the supplied affine NN envelopes must enclose
the same controller on certified_hull. This does not certify floating CROWN.
No state, support, SR history, coefficient, or input tensor is modified here.
"""


def contract(torch, iv, sp, *, eng, sup, input_rows, input_rem,
             control_rows, control_rem, lower_A, upper_A, lower_b, upper_b,
             certified_hull):
    """Return owned (contracted remainder, diagnostics), or fail before writes.

    For x=P_x(r)+R_x and the unchanged control polynomial P_u(r), range
    lA*P_x+lb-P_u+lA*R_x and uA*P_x+ub-P_u+uA*R_x. The lower endpoint
    of the former and upper endpoint of the latter enclose NN(x)-P_u.
    Intersect with the existing remainder of that exact same P_u.
    """
    if input_rows.ndim != 3 or control_rows.ndim != 3:
        raise ValueError("expected batched coefficient rows")
    batch, inputs, terms = input_rows.shape
    outputs = control_rows.shape[1]
    shapes = ((control_rows, (batch, outputs, terms)),
              (input_rem, (batch, inputs, 2)), (control_rem, (batch, outputs, 2)),
              (lower_A, (batch, outputs, inputs)), (upper_A, (batch, outputs, inputs)),
              (lower_b, (batch, outputs)), (upper_b, (batch, outputs)),
              (certified_hull, (batch, inputs, 2)))
    tensors = [input_rows] + [v for v, _ in shapes]
    if batch < 1 or inputs < 1 or outputs < 1 or any(v.shape != shape for v, shape in shapes):
        raise ValueError("incompatible control-certificate shapes")
    if any(v.dtype != torch.float64 or v.device != input_rows.device or
           not bool(torch.isfinite(v).all()) for v in tensors):
        raise ValueError("finite matching float64 tensors required")
    if any(not bool((v[..., 0] <= v[..., 1]).all()) for v in (input_rem, control_rem, certified_hull)):
        raise ValueError("unordered interval input")
    if (sup.spatial or sup.size != terms or not sup.ids or sup.ids[0] != 0
            or sup.n != eng.tables.n or sup.k != eng.tables.k):
        raise ValueError("matching FULL support with constant slot required")
    # The frozen SparseEngine has ids_t/factor, but no exponents method.
    # Reuse its support module's canonical CPU exponent lookup, as the
    # original strict injection does on this engine.
    exponents = sp._exps_for(sup)
    if exponents.shape != (terms, sup.n+1) or bool((exponents[:, 0] != 0).any()):
        raise ValueError("control sampling must be time-free")
    actual_hull = iv.add(sp.range_normal_s(input_rows, eng, sup), input_rem)
    if not bool(((certified_hull[..., 0] <= actual_hull[..., 0]) &
                 (actual_hull[..., 1] <= certified_hull[..., 1])).all()):
        raise ValueError("affine certificate domain does not cover the represented inputs")

    coefficients = iv.from_point(input_rows.transpose(-1, -2)).unsqueeze(1)
    anchor = iv.from_point(control_rows)

    def difference_range(slope, bias):
        coeff = iv.dot_point_iv(slope.unsqueeze(2), coefficients, dim=-1)
        coeff[..., 0, :] = iv.add(coeff[..., 0, :], iv.from_point(bias))
        difference = iv.sub(coeff, anchor)
        polynomial = sp.range_normal_iv_s(difference, eng, sup)
        remainder = iv.dot_point_iv(slope, input_rem.unsqueeze(1), dim=-1)
        return iv.add(polynomial, remainder)

    lower = difference_range(lower_A, lower_b)
    upper = difference_range(upper_A, upper_b)
    proposal = torch.stack((lower[..., 0], upper[..., 1]), -1)
    if not bool(torch.isfinite(proposal).all()) or not bool((proposal[..., 0] <= proposal[..., 1]).all()):
        raise FloatingPointError("nonfinite or contradictory affine envelopes")
    result = torch.stack((torch.maximum(control_rem[..., 0], proposal[..., 0]),
                          torch.minimum(control_rem[..., 1], proposal[..., 1])), -1)
    if not bool((result[..., 0] <= result[..., 1]).all()):
        raise FloatingPointError("same-polynomial enclosure intersection is disjoint")
    changed = (result != control_rem).any(-1)
    return result.clone(), dict(
        contracted_rows=int(changed.sum()), total_rows=batch*outputs,
        polynomial_unchanged=True, remainder_subset_of_input=True,
        proposal=proposal.clone(), certificate_domain_checked=True,
        independent_end_to_end_floating_nncs_certificate=False)
