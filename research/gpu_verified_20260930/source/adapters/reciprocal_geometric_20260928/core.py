from __future__ import annotations
"""Pinned a3fb reciprocal routes with a shared finite-geometric tail.

Isolated candidate only: install.py selects all Python and CUDA routes together.
Every original finite-polynomial/replay recurrence precedes the new tail intact.
"""
import torch
from flowstar_gpu import elementary as elem, interval as iv

# Original local helper bindings keep the copied arithmetic prefix unchanged.
_split_const=elem._split_const
_checked_rec=elem._checked_rec
_one_iv=elem._one_iv
_pad=elem._pad
poly=elem.poly
tm_mul_valid=elem.tm_mul_valid


def normalized_tail(c, rec_c, u, order):
    """Return full rec(C)*(-U)^order/(1+U), plus per-lane bad mask.

    rec_c must be the checked enclosure of 1/C (computed or cache-bound).
    C is checked again so replay cannot lose a sanitized reciprocal's bad flag.
    No infinite-series assumption; ordinary polynomial errors stay with callers.
    """
    if type(order) is not int or order<1 or c.shape!=rec_c.shape or c.shape!=u.shape or c.ndim!=2 or c.shape[-1]!=2:
        raise ValueError('matching [B,2] intervals and positive integer order required')
    if any(v.dtype!=torch.float64 or v.device!=c.device for v in (c,rec_c,u)):
        raise ValueError('same-device binary64 intervals required')
    denominator=iv.add(_one_iv(u),u)
    numerator=iv.pow_int(iv.neg(u),order)
    quotient,bad=elem._checked_div(numerator,denominator)
    _,bad_c=elem._checked_rec(c)  # replay must recover reciprocal-overflow bad too
    tail=iv.mul(rec_c,quotient)
    bad=bad | bad_c
    for value in (c,rec_c,u,denominator,numerator,quotient,tail):
        bad=bad | ~torch.isfinite(value).all(dim=-1) | (value[...,0]>value[...,1])
    return elem._sanitize(tail,bad),bad


def rec_series_valid_g(coeffs, rem, k, cache, base, tabs, make_ctx):
    """Generic rec_taylor body. Returns (res_c, res_r, bad)."""
    c, tm_f = _split_const(coeffs)
    rec_c, bad = _checked_rec(c)
    cache[:, base] = c
    cache[:, base + 1] = rec_c
    cursor = base + 2
    tmf_c = iv.mul(tm_f, rec_c.unsqueeze(1))
    tmf_c_rem = iv.mul(rem, rec_c)
    ctx = make_ctx(tmf_c, tmf_c_rem)
    res_c = ctx.fresh_one()
    res_r = torch.zeros_like(rem)
    f_range = ctx.range_f()
    for _ in range(k, 0, -1):
        res_c = iv.neg(res_c)
        res_r = iv.neg(res_r)
        res_c, res_r = ctx.mul_valid(res_c, res_r, cache, cursor)
        cursor += 3
        res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))
    res_c = iv.mul(res_c, rec_c.unsqueeze(1))
    res_r = iv.mul(res_r, rec_c)
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, tmf_c_rem)
    tail, bad2 = normalized_tail(c, rec_c, tm_range, k + 1)
    res_r = iv.add(res_r, tail)
    return res_c, res_r, bad | bad2


def rec_series_valid(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    cache: torch.Tensor,
    base: int,
    tabs: ElemTables,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TaylorModel<Interval>::rec_taylor (TaylorModel.h:1343).

    Horner form of 1 - F/c + (F/c)^2 - ... over tmF_c = F * (1/c), then * 1/c,
    plus the shared normalized finite-geometric tail; original Horner order.
    Cache: c; 1/c; per iteration triple; rangeF_c. Bad where 0 in c.
    """
    c, tm_f = _split_const(coeffs)
    rec_c, bad = _checked_rec(c)  # [B, 2] const_part.rec_assign()
    cache[:, base] = c  # c_f
    cache[:, base + 1] = rec_c
    cursor = base + 2

    tmf_c = iv.mul(tm_f, rec_c.unsqueeze(1))  # tmF_c = tmF * (1/c)
    tmf_c_rem = iv.mul(rem, rec_c)
    t_width = coeffs.shape[1]
    res_c = torch.zeros_like(coeffs)
    res_c[:, 0] = 1.0
    res_r = torch.zeros_like(rem)
    f_range = poly.range_normal_iv(tmf_c, tables, step)  # tmF_cPolyRange (once)

    for _ in range(k, 0, -1):
        # result *= -1: interval mul by [-1,-1] is exact negation.
        res_c = iv.neg(res_c)
        res_r = iv.neg(res_r)
        kept, res_r = tm_mul_valid(
            res_c, res_r, tmf_c, tmf_c_rem, k, tables, step, cutoff_threshold,
            cache, cursor, f_range,
        )
        cursor += 3
        res_c = _pad(kept, t_width)
        res_c[:, 0] = iv.add(res_c[:, 0], _one_iv(res_c[:, 0]))

    res_c = iv.mul(res_c, rec_c.unsqueeze(1))
    res_r = iv.mul(res_r, rec_c)
    cache[:, cursor] = f_range
    tm_range = iv.add(f_range, tmf_c_rem)
    tail, bad2 = normalized_tail(c, rec_c, tm_range, k + 1)
    res_r = iv.add(res_r, tail)
    return res_c, res_r, bad | bad2


def rec_series_replay(
    rem: torch.Tensor, cache: torch.Tensor, base: int, k: int, tabs: ElemTables
) -> tuple[torch.Tensor, torch.Tensor]:
    """rec_taylor_only_remainder (expression.h:88). Returns (rem, bad)."""
    c_f = cache[:, base]
    const_part = cache[:, base + 1]  # 1/c
    cursor = base + 2
    r2 = iv.mul(rem, const_part)  # tmF_c_remainder = remainder * const_part
    result = torch.zeros_like(rem)
    for _ in range(k, 0, -1):
        result = iv.neg(result)  # result *= -1
        t = iv.mul(cache[:, cursor], r2)  # P1 x I2
        t = iv.add(t, iv.mul(cache[:, cursor + 1], result))  # P2 x I1
        t = iv.add(t, iv.mul(r2, result))  # I2 x I1
        t = iv.add(t, cache[:, cursor + 2])  # truncation
        cursor += 3
        result = t
    result = iv.mul(result, const_part)
    tm_range = iv.add(cache[:, cursor], r2)
    tail, bad = normalized_tail(c_f, const_part, tm_range, k + 1)
    return iv.add(result, tail), bad
