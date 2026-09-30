"""Strict control-boundary endpoint substitution, independent of frozen drivers.

Call once on an owned, all-active committed state before the NN refresh.
Only pre/pre_sup/pre_rem are replaced. Historical tmv and live SR are retained;
the next advance carries the added error into its fresh J through x0_rem.
This does not qualify CROWN or RN control injection, or repair earlier gaps.
"""


def end_of_time_s(st, eng):
    """Return the charged error; reject invalid input/output before state writes."""
    import torch
    from flowstar_gpu import interval as iv, support as sp

    if (st.pre.dtype != torch.float64 or st.pre_rem.dtype != torch.float64
            or st.pre.ndim != 3 or tuple(st.pre_rem.shape) != (*st.pre.shape[:2], 2)
            or st.pre.shape[1] != eng.tables.n or st.pre.shape[0] == 0
            or tuple(st.status.shape) != (st.pre.shape[0],)
            or st.pre.device != eng.tables.exponents.device or st.pre_rem.device != st.pre.device
            or st.status.device != st.pre.device
            or st.pre_sup.spatial or st.pre_sup.n != eng.tables.n
            or st.pre_sup.k != eng.tables.k or st.pre.shape[-1] != st.pre_sup.size):
        raise ValueError('strict endpoint requires a matching FULL-support float64 state')
    if (not bool((st.status == 0).all()) or not bool(torch.isfinite(st.pre).all())
            or not bool(torch.isfinite(st.pre_rem).all())
            or not bool((st.pre_rem[..., 0] <= st.pre_rem[..., 1]).all())):
        raise ValueError('strict endpoint requires all active, finite, ordered leaves')
    point, support, error = sp.evaluate_time_end_s_with_roundoff(st.pre, eng, st.pre_sup)
    remainder = iv.add(st.pre_rem, error)
    if (support.spatial or support.n != eng.tables.n or support.k != eng.tables.k
            or tuple(point.shape) != (*st.pre.shape[:2], support.size)
            or error.shape != st.pre_rem.shape
            or not bool((eng.tables.exponents[list(support.ids), 0] == 0).all())
            or not bool(torch.isfinite(point).all() and torch.isfinite(error).all()
                        and torch.isfinite(remainder).all())
            or not bool((error[..., 0] <= error[..., 1]).all()
                        and (remainder[..., 0] <= remainder[..., 1]).all())):
        raise ValueError('strict endpoint produced an invalid or nonfinite enclosure')
    # Own outputs before exposing them; failed checks above leave the state intact.
    point, remainder, error = point.clone(), remainder.clone(), error.clone()
    st.pre, st.pre_sup, st.pre_rem = point, support, remainder
    return error
