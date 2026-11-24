import json

def strict_injection(st, T, L, U, u_ids, nn_in, eng):
    """Dimension-generic form of the frozen strict QUAD injection arithmetic."""
    import torch
    from flowstar_gpu import interval as iv, support as sp

    n, m, B = eng.tables.n, T.shape[1], st.pre.shape[0]
    if ((n, nn_in, m, tuple(u_ids)) not in ((6, 4, 1, (5,)),
                                          (7, 4, 2, (5, 6)), (10, 6, 3, (7, 8, 9)),
                                          (16, 12, 3, (13, 14, 15)),
                                          (19, 12, 6, (13, 14, 15, 16, 17, 18)))
            or st.pre.ndim != 3 or st.pre.shape[1] != n
            or st.pre.shape[2] != st.pre_sup.size or st.pre_sup.n != n
            or st.pre_sup.k != eng.tables.k or st.pre_sup.spatial
            or not st.pre_sup.ids or st.pre_sup.ids[0] != 0
            or max(st.pre_sup.degs) > eng.tables.k):
        raise ValueError("strict injection requires an approved time-free FULL layout")
    device = st.pre.device
    if (T.shape != (B, m, nn_in) or L.shape != (B, m) or U.shape != (B, m)
            or st.pre_rem.shape != (B, n, 2) or st.status.shape != (B,)
            or st.status.dtype != torch.int8 or st.status.device != device
            or device != torch.device(eng.device)
            or any(v.dtype != torch.float64 or v.device != device
                   for v in (st.pre, st.pre_rem, T, L, U))
            or not bool(((st.status >= 0) & (st.status <= 3)).all())):
        raise ValueError("DP injection requires matching binary64 rows and certificates")
    finite = lambda x: bool(torch.isfinite(x).all())
    interval = lambda x: finite(x) and bool((x[..., 0] <= x[..., 1]).all())
    checks = {
        "pre_finite": finite(st.pre),
        "T_finite": finite(T),
        "L_finite": finite(L),
        "U_finite": finite(U),
        "pre_rem_finite": finite(st.pre_rem),
        "pre_rem_ordered": bool((st.pre_rem[..., 0] <= st.pre_rem[..., 1]).all()),
        "L_U_ordered": bool((L <= U).all()),
    }
    if not all(checks.values()):
        checks["bad_pre_rem_rows"] = torch.nonzero(
            ~(torch.isfinite(st.pre_rem).all(-1) & (st.pre_rem[..., 0] <= st.pre_rem[..., 1]))
        ).cpu().tolist()[:12]
        checks["bad_control_rows"] = torch.nonzero(
            ~(torch.isfinite(L) & torch.isfinite(U) & (L <= U))
        ).cpu().tolist()[:12]
        checks["L"] = L.detach().cpu().tolist()
        checks["U"] = U.detach().cpu().tolist()
        checks["U_minus_L"] = (U - L).detach().cpu().tolist()
        checks["T"] = T.detach().cpu().tolist()
        raise ValueError("DP injection invalid input: " + json.dumps(checks))
    exps = eng.exponents(st.pre_sup) if hasattr(eng, "exponents") else sp._exps_for(st.pre_sup)
    if exps.shape != (st.pre_sup.size, n + 1) or bool((exps[:, 0] != 0).any()):
        raise ValueError("DP controller sampling requires a time-free FULL support")
    owners = (st.pre.untyped_storage().data_ptr(), st.pre_rem.untyped_storage().data_ptr())
    if owners[0] == owners[1] or any(
        x.untyped_storage().data_ptr() in owners for x in (st.tmv, st.tmv_rem, T, L, U)
    ):
        raise ValueError("DP control rows must own storage separate from plant and certificate")

    # Preserve the original RN polynomial and charge every interval error.
    center = (U + L) * 0.5
    point = torch.einsum("bmi,bit->bmt", T, st.pre[:, :nn_in])
    point[..., 0] += center
    if not finite(center) or not finite(point):
        raise FloatingPointError("control polynomial overflow")
    input_rem = iv.dot_point_iv(T, st.pre_rem[:, :nn_in].unsqueeze(1), dim=-1)
    coeff = iv.dot_point_iv(
        T.unsqueeze(2), iv.from_point(st.pre[:, :nn_in].transpose(-1, -2)).unsqueeze(1), dim=-1
    )
    coeff[..., 0, :] = iv.add(coeff[..., 0, :], iv.from_point(center))
    coefficient_error = iv.sub(coeff, iv.from_point(point))
    error_range = sp.range_normal_iv_s(coefficient_error, eng, st.pre_sup)
    bias_residual = iv.sub(torch.stack((L, U), -1), iv.from_point(center))
    remainder = iv.add(iv.add(input_rem, bias_residual), error_range)
    if not all(interval(x) for x in (input_rem, coeff, coefficient_error, error_range, bias_residual, remainder)):
        raise FloatingPointError("strict DP injection produced invalid bounds")
    st.pre[:, u_ids] = point.clone()
    st.pre_rem[:, u_ids] = remainder.clone()
