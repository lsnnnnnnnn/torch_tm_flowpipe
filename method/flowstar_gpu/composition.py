"""Taylor-model composition: substitute the history map `tmv` into `tmv_of_x0`.

Flow* stage (c) of advance() (Continuous.cpp:884-886): the previous flowpipe
evaluated at t = delta, `tmv_of_x0` (a TM over the CANONICAL local coords r),
is composed with `tmv` (the TM mapping the original normalized initial-set
coords s into r): new_tmv = tmv_of_x0 o tmv. Flow* does this by converting
tmv_of_x0 to Horner form and recursively substituting (insert_ctrunc_normal,
TaylorModel.h:1026/3378) — a data-dependent recursion that batches poorly.

We restructure (plan D3, ALGORITHM.md "composition bookkeeping difference"):

  1. Build the TM image under `tmv` of EVERY spatial basis monomial
     m = prod_i r_i^{e_i}, degree 2..k, via the static parent DAG
         M[m] = M[parent(m)] * tmv[lastvar(m)]
     where parent(m) = m with the lowest-index nonzero exponent decremented —
     one truncated TM multiply per basis monomial, schedule fixed per (n, k).
  2. Contract: new_tmv_i = sum_m a[i, m] * M[m], where a = tmv_of_x0's
     coefficients — a batched GEMM on the polynomial parts plus an interval
     dot on the remainders, then + tmv_of_x0's own remainder.

The polynomial result agrees with Flow* to rounding; the REMAINDER bookkeeping
differs (both sound): Flow* accumulates truncation mass along its Horner
recursion, we accumulate it along the DAG chain. Parity for this stage is
therefore judged functionally (range/width), per plan D7.

Both tmv and tmv_of_x0 are TIME-FREE (evaluate_time removed t; tmv never had
it), so all polynomials here live in the spatial sub-basis and the image
multiplies run on the dedicated spatial product tables (poly.mul_point_spatial
/ ctrunc_normal_spatial / cutoff_normal_spatial — identical semantics to the
full-basis path on t-free inputs, ~(T/Ts)^2 less pair traffic).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from . import interval as iv
from . import polynomial as poly
from .monomials import MonomialTables, basis_size
from .polynomial import StepTables
from .rounding import dot_error_bound


@dataclass(frozen=True)
class CompositionSchedule:
    """Static per-(n, k) DAG schedule for building spatial-monomial images.

    Spatial basis indexing here is 0..Ts-1 in the tables' spatial order
    (tables.spatial_index maps it into the full basis). Attributes:

        n, k: dimensions.
        Ts: spatial basis size.
        parent [Ts] i64: spatial index of the parent (-1 for degree <= 1).
        lastvar [Ts] i64: which tmv component multiplies the parent
            (-1 for degree <= 1).
        var_image [n] i64: spatial index of each degree-1 monomial r_i.
    """

    n: int
    k: int
    Ts: int
    parent: torch.Tensor
    lastvar: torch.Tensor
    var_image: torch.Tensor
    # Level-batched build schedule: images of equal degree are independent
    # (parents live one level down), so each level runs as ONE batched multiply
    # — numerically identical to the sequential build, just vectorized.
    # levels[d-2] = (members [L_d], parents [L_d], lastvars [L_d]) for degree d.
    levels: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...] = ()


# Debug-only precondition checking (each check is a host-device sync — never
# enabled in production loops; tests flip it to exercise the guard).
CHECK_PRECONDITIONS = False


@lru_cache(maxsize=32)
def build_schedule(n: int, k: int, device: str = "cpu") -> CompositionSchedule:
    """Construct the DAG schedule from the monomial tables (cached per device).

    Passing the production device here moves every index tensor ONCE — the
    per-call .to(device) copies this replaces were one H2D transfer per level
    chunk per compose() call."""
    from .monomials import build_tables  # local import to avoid cycle at module load

    tables = build_tables(n, k)
    sp_full = tables.spatial_index.numpy()  # [Ts] indices into full basis
    exps = tables.exponents.numpy()[sp_full][:, 1:]  # [Ts, n] spatial exponents
    Ts = sp_full.shape[0]
    assert Ts == basis_size(n, k)

    # Map spatial exponent tuple -> spatial index for parent lookup.
    index_of = {tuple(row): i for i, row in enumerate(exps)}

    parent = np.full(Ts, -1, dtype=np.int64)
    lastvar = np.full(Ts, -1, dtype=np.int64)
    var_image = np.full(n, -1, dtype=np.int64)
    for m in range(Ts):
        e = exps[m]
        deg = int(e.sum())
        if deg == 1:
            var_image[int(np.nonzero(e)[0][0])] = m
            continue
        if deg == 0:
            continue
        v = int(np.nonzero(e)[0][0])  # lowest-index nonzero exponent
        pe = e.copy()
        pe[v] -= 1
        parent[m] = index_of[tuple(pe)]
        lastvar[m] = v

    tt = torch.from_numpy

    # Group degree >= 2 monomials into levels for batched building.
    degs = exps.sum(axis=1)  # [Ts]
    levels = []
    for d in range(2, k + 1):
        members = np.nonzero(degs == d)[0].astype(np.int64)  # [L_d]
        if members.size == 0:  # pragma: no cover  # unreachable guard: r_1^d exists for every 2 <= d <= k (build_tables enforces k >= 2)
            continue
        levels.append((tt(members), tt(parent[members]), tt(lastvar[members])))

    return CompositionSchedule(
        n=n,
        k=k,
        Ts=Ts,
        parent=tt(parent).to(device),
        lastvar=tt(lastvar).to(device),
        var_image=tt(var_image).to(device),
        levels=tuple(
            (m.to(device), p.to(device), lv.to(device)) for (m, p, lv) in levels
        ),
    )


def _spatial_to_full(coeffs_sp: torch.Tensor, tables: MonomialTables) -> torch.Tensor:
    """Embed spatial coefficients [..., Ts] into the full working basis [..., T]."""
    out = torch.zeros(*coeffs_sp.shape[:-1], tables.T, dtype=coeffs_sp.dtype, device=coeffs_sp.device)
    out[..., tables.spatial_index] = coeffs_sp
    return out


def compose(
    a_coeffs_sp: torch.Tensor,
    a_rem: torch.Tensor,
    g_coeffs_sp: torch.Tensor,
    g_rem: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
    schedule: CompositionSchedule,
    order: int,
    cutoff_threshold: float,
    strict: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """new_tmv = a o g: substitute TM vector g into polynomial TM vector a.

    Args:
        a_coeffs_sp [B, n, Ts]: tmv_of_x0's spatial coefficients, CONSTANT
            REMOVED (Flow* rmConstant runs before composition; slot 0 must be
            zero and is asserted).
        a_rem [B, n, 2]: tmv_of_x0's remainders (added at the end, mirroring
            insert_ctrunc_normal's final `result.remainder += this->remainder`).
        g_coeffs_sp [B, n, Ts], g_rem [B, n, 2]: the history map tmv.
        order / cutoff_threshold: truncation degree and cutoff for the image
            multiplies (Flow* passes tm_setting.order and cutoff).
        strict: tier-S — Rump-bound the contraction GEMM's coefficient
            roundoff (Flow* leaves it unbounded, GOTCHAS #15).

    Returns (coeffs_sp [B, n, Ts], rem [B, n, 2]) of the composed TM vector.
    """
    B, n, Ts = a_coeffs_sp.shape
    dev, dt = a_coeffs_sp.device, a_coeffs_sp.dtype
    if CHECK_PRECONDITIONS and bool((a_coeffs_sp[..., 0] != 0).any()):
        raise ValueError("compose: constant of tmv_of_x0 must be removed first (rmConstant)")

    # --- 1. build monomial images M[m] over the spatial basis ---------------
    # Image tensors: M_coeffs [B, Ts, Ts] (image m's coefficients), M_rem [B, Ts, 2].
    m_coeffs = torch.zeros(B, Ts, Ts, dtype=dt, device=dev)
    m_rem = torch.zeros(B, Ts, 2, dtype=dt, device=dev)

    # Degree 0: constant monomial -> constant-1 TM (never referenced: slot 0 of
    # a is zero) — still set for completeness.
    m_coeffs[:, 0, 0] = 1.0

    # Degree 1: image of r_i is g_i itself.
    vi = schedule.var_image  # [n] (schedule tensors live on the production device)
    m_coeffs[:, vi] = g_coeffs_sp  # [B, n, Ts]
    m_rem[:, vi] = g_rem

    # Degree >= 2, ascending by LEVEL: all images of degree d have parents of
    # degree d-1, so each level is one batched truncated TM multiply
    # M[members] = M[parents] * g[lastvars]. Identical arithmetic to a
    # per-monomial loop (independent multiplies), vectorized (658ms -> see
    # PROGRESS.md M3 profiling; compose dominated the step at 70%).
    # Spatial fast path: every polynomial here is t-free, so the dedicated
    # spatial tables apply (identical semantics, ~5x less pair traffic).
    g_range = poly.range_normal_spatial(g_coeffs_sp, tables)  # [B, n, 2]
    # Level chunking: a level multiply materializes [B, Lc, Ps] pair products
    # (Ps = Ts^2). At quadrotor scale (Ts = 2380, Ps = 5.7M) an unchunked
    # 455-member level would need ~80 GB — split members so the pair tensor
    # stays under ~2 GiB (identical arithmetic: members are independent).
    ps_pairs = tables.sp_pair_i.shape[0]
    chunk = max(1, int(2 * 2**30 / 8 / max(1, B * ps_pairs)))
    for members_all, parents_all, lastvars_all in schedule.levels:
      for lo in range(0, members_all.shape[0], chunk):
        members = members_all[lo : lo + chunk]
        parents = parents_all[lo : lo + chunk]
        lastvars = lastvars_all[lo : lo + chunk]
        pa = m_coeffs[:, parents]  # [B, Lc, Ts]
        gv = g_coeffs_sp[:, lastvars]  # [B, Lc, Ts]
        if strict:
            prod, coefficient_roundoff = poly.mul_point_spatial_with_roundoff(
                pa, gv, tables
            )
        else:
            prod = poly.mul_point_spatial(pa, gv, tables)  # [B, Lc, Ts2]
            coefficient_roundoff = None
        pa_range = poly.range_normal_spatial(pa, tables)  # [B, L, 2]
        # Remainder algebra of mul_insert_ctrunc_normal (I1*I2 + P2*I1 + P1*I2):
        p_rem = m_rem[:, parents]  # [B, L, 2]
        gv_rem = g_rem[:, lastvars]  # [B, L, 2]
        rem = iv.mul(p_rem, gv_rem)
        rem = iv.add(rem, iv.mul(g_range[:, lastvars], p_rem))
        rem = iv.add(rem, iv.mul(pa_range, gv_rem))
        kept, tail = poly.ctrunc_normal_spatial(prod, tables, order)  # [B, L, K], [B, L, 2]
        kept, round_rng = poly.cutoff_normal_spatial(kept, tables, cutoff_threshold)
        rem = iv.add(rem, iv.add(tail, round_rng))
        if coefficient_roundoff is not None:
            # The exact coefficient product equals the retained tier-P point
            # polynomial plus this ranged error.  Charge it once here; later
            # image levels propagate it through ``p_rem`` like any other
            # ordinary remainder contribution.
            rem = iv.add(rem, coefficient_roundoff)
        out_row = torch.zeros(B, members.shape[0], Ts, dtype=dt, device=dev)
        out_row[..., : kept.shape[-1]] = kept
        m_coeffs[:, members] = out_row
        m_rem[:, members] = rem

    # --- 2. contract with a's coefficients ----------------------------------
    # Polynomial part: tier-P GEMM (Flow* composes with plain Real arithmetic).
    out_coeffs = torch.einsum("bim,bmt->bit", a_coeffs_sp, m_coeffs)  # [B, n, Ts]

    # Remainders: interval dot of point weights a[i, m] with image remainders,
    # then + a's own remainder (Flow*: result.remainder += this->remainder).
    w_rem = iv.dot_point_iv(
        a_coeffs_sp, m_rem.unsqueeze(1), dim=-1
    )  # [B, n, 2]: point weights [B, n, Ts] x image remainders [B, 1, Ts, 2], reduce m
    out_rem = iv.add(w_rem, a_rem)

    if strict:
        # Rump bound for the contraction GEMM (inner dim Ts) applied to the
        # composed polynomial's range: |dC[t]| <= err[t]; range impact is
        # sum_t err[t] * |factor(t)| — symmetric.
        abs_dot = torch.einsum("bim,bmt->bit", a_coeffs_sp.abs(), m_coeffs.abs())
        gemm_err = dot_error_bound(abs_dot, Ts)  # [B, n, Ts]
        full_err = _spatial_to_full(gemm_err, tables)  # [B, n, T]
        err_rng = iv.dot_point_iv(full_err, torch.abs(step.factor(0, tables.T, full_err.dim())), dim=-1)
        sym = err_rng[..., 1]  # [B, n]
        out_rem = iv.add(out_rem, torch.stack((-sym, sym), dim=-1))

    return out_coeffs, out_rem
