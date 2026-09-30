# Mixed-status candidate: numeric body unchanged from frozen helper below.
# All original lanes must remain finite/ordered; original boundary operations
# still update all rows, including inactive lanes, while status is never changed.
# The full NN batch/domain is not filtered. This does not revive a failed lane.
# Nonfinite input is explicitly unsupported, with no partial writes.
# Frozen original source SHA: a79b19ad1bd97ae87b40f81239352a8b9859621757bd044df6fd615630c64310
"""Conditional strict QUAD injection for a supplied valid affine certificate.

Assumption, NOT proved here: T*x+L <= controller(x) <= T*x+U on the
entire represented input domain. L/U are affine bias bounds, not output
box bounds. CROWN's floating-point certificate remains unqualified.

Keep the original RN point coefficients; charge their coefficient errors,
the input remainders, and the exact bias displacement from the chosen RN
center. Only pre/pre_rem control rows 13:16 are written after all checks.
There is deliberately no SR argument and no external SR append/reset.
"""
import torch
from flowstar_gpu import interval as iv, support as sp


def _finite(value):return bool(torch.isfinite(value).all())
def _interval(value):return _finite(value) and bool((value[...,0]<=value[...,1]).all())


@torch.no_grad()
def inject_controls_s(st,T,L,U,u_ids,nn_in,*,eng):
    """Return owned diagnostics; reject unsupported/overflow inputs before writes."""
    if (not isinstance(u_ids,(tuple,list)) or tuple(u_ids)!=(13,14,15) or nn_in!=12
            or eng.tables.n!=16 or eng.tables.k not in [2,3,4]
            or st.pre.ndim!=3 or st.pre.shape[0]<1 or st.pre.shape[1]!=16
            or st.pre.shape[-1]!=st.pre_sup.size
            or st.pre_sup.n!=16 or st.pre_sup.k!=eng.tables.k or st.pre_sup.spatial
            or not st.pre_sup.ids or st.pre_sup.ids[0]!=0 or max(st.pre_sup.degs)>eng.tables.k):
        raise ValueError('injection requires the qualified QUAD FULL-support layout')
    B=st.pre.shape[0];device=st.pre.device
    if (T.shape!=(B,3,12) or L.shape!=(B,3) or U.shape!=(B,3)
            or st.pre_rem.shape!=(B,16,2) or st.status.shape!=(B,)
            or st.status.dtype!=torch.int8 or st.status.device!=device
            or device!=eng._cutoff_zero.device
            or any(v.dtype!=torch.float64 or v.device!=device for v in [st.pre,st.pre_rem,T,L,U])
            or not bool(((st.status>=0)&(st.status<=3)).all())):
        raise ValueError('injection requires matching float64 leaves, original statuses and certificates')
    if (not all(_finite(v) for v in [st.pre,T,L,U]) or not _interval(st.pre_rem)
            or not bool((L<=U).all())):
        raise ValueError('injection requires finite coefficients and ordered intervals')
    exps=eng.exponents(st.pre_sup) if hasattr(eng,'exponents') else sp._exps_for(st.pre_sup)
    if exps.shape!=(st.pre_sup.size,17) or (exps[:,0]!=0).any():
        raise ValueError('control sampling requires a time-free FULL support')
    # The actual callers own separate state buffers. Reject aliases that could
    # make a control-row write mutate tmv, the certificate, or the other buffer.
    pre_owner=st.pre.untyped_storage().data_ptr();rem_owner=st.pre_rem.untyped_storage().data_ptr()
    protected=[st.tmv,st.tmv_rem,T,L,U]
    if pre_owner==rem_owner or any(v.untyped_storage().data_ptr() in [pre_owner,rem_owner] for v in protected):
        raise ValueError('injection needs owned state buffers disjoint from protected tensors')

    # EXACTLY the old point path. Its RN outputs are the polynomial we certify.
    center=(U+L)*0.5
    point=torch.einsum('bmi,bit->bmt',T,st.pre[:,:12])
    point[...,0]+=center
    if not _finite(center) or not _finite(point):
        raise FloatingPointError('original RN point construction overflowed')

    # Keep the original input-remainder dot. Reordering the coefficient
    # enclosure puts the 12-term reduction last; the point path above is not
    # reordered. All quantities below are interval enclosures of exact reals.
    input_rem=iv.dot_point_iv(T,st.pre_rem[:,:12].unsqueeze(1),dim=-1)
    coeff=iv.dot_point_iv(T.unsqueeze(2),iv.from_point(st.pre[:,:12].transpose(-1,-2)).unsqueeze(1),dim=-1)
    coeff[...,0,:]=iv.add(coeff[...,0,:],iv.from_point(center))
    coefficient_error=iv.sub(coeff,iv.from_point(point))
    error_range=sp.range_normal_iv_s(coefficient_error,eng,st.pre_sup)
    bias_residual=iv.sub(torch.stack((L,U),-1),iv.from_point(center))
    remainder=iv.add(iv.add(input_rem,bias_residual),error_range)
    if not all(_interval(v) for v in [input_rem,coeff,coefficient_error,error_range,bias_residual,remainder]):
        raise FloatingPointError('strict injection produced nonfinite or unordered bounds')
    point,remainder=point.clone(),remainder.clone()
    receipt=dict(center=center.clone(),point_rows=point.clone(),coefficient_error=coefficient_error.clone(),
        coefficient_error_range=error_range.clone(),weighted_input_remainder=input_rem.clone(),
        bias_residual=bias_residual.clone(),new_control_remainder=remainder.clone())
    st.pre[:,13:16]=point
    st.pre_rem[:,13:16]=remainder
    return receipt
