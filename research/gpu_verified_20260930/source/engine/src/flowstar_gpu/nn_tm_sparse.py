"""S2-1 — TM-through-NN on the union-support sparse representation.

The dense path (nn_tm.py) moves [rows, T, 2] tensors through T²-pair products;
at CartPole-B256 / QUAD-order-4 scale that is 719 GiB-class traffic (measured;
the Sprint-1 TIMEOUT wall). Control-time supports are 20-40× smaller than T
(instrumented: CartPole S* ≈ 400 of T = 1716; QUAD-order-4 S* ≈ 200-800 of
T = 5985), so the activation algebra runs here on Support-aligned tensors with
per-activation specializations, exactly the M10 tape pattern:

  * the series tier reuses the GENERIC sigmoidal body
    (elementary._sigmoidal_series_valid_g — bitwise-pinned against the dense
    original through DenseSeriesCtx) driven by a SparseValidCtx over a
    precomputed PairBind chain;
  * the quadratic tier's one tm_mul runs through tm_mul_valid_s on its own
    precomputed bind; the fit + centered-form error are layout-free scalars;
  * the box tier is a constant slot — slot 0 on EVERY support by the id-0
    invariant;
  * `specialize_act` builds the whole per-(sup_F, k) plan on CPU once and
    caches it on the engine; input supports are MONOTONE-ACCUMULATED per
    (net, layer) key (the measured failure mode is cutoff-boundary support
    oscillation minting fresh binds/keys every control step — supersets are
    exact by the zero-removal argument, so accumulation never loses
    soundness).

Soundness: identical algebra to the dense path on the support's slots plus
structural zeros elsewhere — pinned by the dense≡sparse equality tests
(superset supports included) and the MC containment suite parametrized over
both paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import elementary as el
from . import interval as iv
from . import support as sp
from .nn_tm import _quadratic_fit_error
from .rounding import next_down, next_up
from .sparse_exec import (
    SparseValidCtx,
    _eng_cache,
    _kept,
    _sup0,
    tm_mul_valid_s,
)

# Transient budget per activation mul (see nn_tm._CHUNK_BUDGET_BYTES; the
# sparse pair counts are what make this budget go 30-900x further).
_CHUNK_BUDGET_BYTES = 2e9


@dataclass
class ActSpec:
    """CPU-precomputed plan for one activation over one argument support."""

    sup_f: sp.Support
    k: int
    chain: list  # k entries (PairBind, keep, sup_in) — the pow chain
    acc: list  # k entries (sup_union, emb_res, emb_pow) — res accumulation
    sup_series: sp.Support  # final res support
    quad_pair: tuple  # (PairBind, keep, sup_kept) for (F-mid)^2
    sup_quad: sp.Support  # union(sup_f, quad kept)
    q_emb_f: torch.Tensor  # sup_f -> sup_quad
    q_emb_sq: torch.Tensor  # quad kept -> sup_quad
    sup_act: sp.Support  # dispatch union: series ∪ quad (slot 0 ∈ both)
    emb_series: torch.Tensor  # sup_series -> sup_act
    emb_quad: torch.Tensor  # sup_quad -> sup_act
    max_pairs: int  # chunk arithmetic


def specialize_act(sup_f: sp.Support, k: int, eng: sp.SparseEngine) -> ActSpec:
    """Build (and engine-cache) the activation plan for argument support
    sup_f at series/truncation order k. Pure CPU set algebra + table builds;
    every pair/union bind is value-cached at the support level."""
    cache = _eng_cache(eng, "_actspec")
    key = (sup_f, k)
    hit = cache.get(key)
    if hit is not None:
        return hit

    sup_zero = _sup0(eng)
    chain, acc = [], []
    cur = sup_zero
    res_s = sup_zero
    max_pairs = 0
    for _ in range(k):
        pb = eng.pair(cur, sup_f)
        kept_sup, keep = _kept(pb, k)
        chain.append((pb, keep, cur))
        max_pairs = max(max_pairs, pb.pair_ia.shape[0])
        su, er, ep = eng.union(res_s, kept_sup)
        acc.append((su, er, ep))
        res_s = su
        cur = kept_sup

    pb_q = eng.pair(sup_f, sup_f)
    kept_q_sup, keep_q = _kept(pb_q, k)
    max_pairs = max(max_pairs, pb_q.pair_ia.shape[0])
    sup_quad, ef, esq = eng.union(sup_f, kept_q_sup)
    sup_act, e_ser, _ = eng.union(res_s, sup_quad)
    _, _, e_quad = eng.union(res_s, sup_quad)

    spec = ActSpec(
        sup_f=sup_f, k=k, chain=chain, acc=acc, sup_series=res_s,
        quad_pair=(pb_q, keep_q, kept_q_sup), sup_quad=sup_quad,
        q_emb_f=ef, q_emb_sq=esq, sup_act=sup_act,
        emb_series=e_ser, emb_quad=e_quad, max_pairs=int(max_pairs),
    )
    cache[key] = spec
    return spec


def _collapse_iv_s(
    coeffs_iv: torch.Tensor, rem: torch.Tensor, eng: sp.SparseEngine,
    sup: sp.Support,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse twin of nn_tm._collapse_iv: interval coeffs on `sup` -> point
    midpoints + radius range (via the support's factor rows) into rem."""
    mid = (coeffs_iv[..., 0] + coeffs_iv[..., 1]) * 0.5
    mid_iv = torch.stack((next_down(mid), next_up(mid)), dim=-1)
    rad = iv.sub(coeffs_iv, mid_iv)
    prod = iv.mul(rad, eng.factor(sup))
    return mid, iv.add(rem, iv.sum(prod, dim=-1))


def _embed_rows(t: torch.Tensor, emb: torch.Tensor, size: int) -> torch.Tensor:
    """[R, S(,2)] -> [R, size(,2)] placing slots at emb (structural zeros
    elsewhere — exact by the superset argument)."""
    if t.dim() == 3:
        out = t.new_zeros(t.shape[0], size, 2)
        out[:, emb] = t
    else:
        out = t.new_zeros(t.shape[0], size)
        out[:, emb] = t
    return out


def _quadratic_candidate_s(
    kind: str, coeffs: torch.Tensor, rem: torch.Tensor, arg: torch.Tensor,
    spec: ActSpec, eng: sp.SparseEngine, cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse P4 tier: q(F) = a0 + a1·(F−mid) + a2·(F−mid)² + E on sup_quad.

    coeffs [R, Sf] point on spec.sup_f, rem [R, 2], arg [R, 2] (range of
    poly + rem). One tm_mul_valid_s on the precomputed bind; the fit + error
    (`nn_tm._quadratic_fit_error`) are layout-free.
    """
    a0, a1, a2, mid, E = _quadratic_fit_error(kind, arg)
    fc = iv.from_point(coeffs)  # [R, Sf, 2]
    fc[:, 0] = iv.sub(fc[:, 0], iv.from_point(mid))
    f_range = iv.sub(arg, iv.from_point(mid))
    pb_q, keep_q, _ = spec.quad_pair
    cache = coeffs.new_zeros(coeffs.shape[0], 3, 2)
    sq, sq_r = tm_mul_valid_s(fc, rem, fc, rem, pb_q, keep_q, spec.sup_f,
                              eng, cutoff, cache, 0, f_range)
    size = spec.sup_quad.size
    q_c = iv.add(
        iv.mul_point(_embed_rows(fc, spec.q_emb_f, size), a1.unsqueeze(-1)),
        iv.mul_point(_embed_rows(sq, spec.q_emb_sq, size), a2.unsqueeze(-1)),
    )
    q_c[:, 0] = iv.add(q_c[:, 0], iv.from_point(a0))
    q_r = iv.add(iv.mul_point(rem, a1), iv.mul_point(sq_r, a2))
    q_r = iv.add(q_r, E)
    return _collapse_iv_s(q_c, q_r, eng, spec.sup_quad)


def _act_series_s(
    kind: str, coeffs: torch.Tensor, rem: torch.Tensor, spec: ActSpec,
    k: int, eng: sp.SparseEngine, cutoff: float, tabs: el.ElemTables,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse per-neuron dispatch: {series, box, quadratic}, narrowest
    certified remainder wins. coeffs [R, Sf] point on spec.sup_f ->
    (point coeffs [R, S_act] on spec.sup_act, rem [R, 2], bad [R] False)."""
    from .nn_act_kinds import get_kind
    from .nn_tm import _quadratic_fit_error

    kobj = get_kind(kind) if isinstance(kind, str) else kind
    f_iv = kobj.f_iv

    R = coeffs.shape[0]
    size = spec.sup_act.size
    arg = iv.add(sp.range_normal_s(coeffs, eng, spec.sup_f), rem)  # [R, 2]

    if kobj.has_series:
        series_g = (el.tanh_series_valid_g if kobj.name == "tanh"
                    else el.sigmoid_series_valid_g)

        def mkctx(F, Fr):
            return SparseValidCtx(eng, spec.chain, spec.acc, F, Fr,
                                  spec.sup_f, cutoff)

        cache = coeffs.new_zeros(R, 4 * k + 2, 2)
        s_c, s_r = series_g(iv.from_point(coeffs), rem, k, cache, 0, tabs,
                            mkctx)
        s_pt, s_rem = _collapse_iv_s(s_c, s_r, eng, spec.sup_series)
        out_c = _embed_rows(s_pt, spec.emb_series, size)
        out_r = s_rem
    else:
        # nonsmooth: degree-1 fit tier in the series slot (no tm_mul)
        a0, a1, _, midl, El = _quadratic_fit_error(kobj, arg, degree=1)
        lin = coeffs * a1.unsqueeze(-1)
        lin[:, 0] = lin[:, 0] + (a0 - a1 * midl)
        out_c = _embed_rows(lin, spec.q_emb_f, spec.sup_quad.size)
        out_c = _embed_rows(out_c, spec.emb_quad, size)
        out_r = iv.add(iv.mul_point(rem, a1), El)
    box = f_iv(arg)
    box_mid = (box[..., 0] + box[..., 1]) * 0.5
    box_c = coeffs.new_zeros(R, size)
    box_c[:, 0] = box_mid
    box_r = iv.sub(box, torch.stack((box_mid, box_mid), dim=-1))

    quad_pt, quad_rem = _quadratic_candidate_s(kobj, coeffs, rem, arg, spec,
                                               eng, cutoff)
    quad_c = _embed_rows(quad_pt, spec.emb_quad, size)

    w_series = out_r[..., 1] - out_r[..., 0]
    w_box = box_r[..., 1] - box_r[..., 0]
    w_quad = quad_rem[..., 1] - quad_rem[..., 0]
    w_series = torch.where(torch.isfinite(w_series), w_series,
                           torch.full_like(w_series, torch.inf))
    w_quad = torch.where(torch.isfinite(w_quad), w_quad,
                         torch.full_like(w_quad, torch.inf))
    pick = torch.stack((w_series, w_box, w_quad)).argmin(dim=0)  # [R]
    out_c = torch.where((pick == 1).unsqueeze(-1), box_c, out_c)
    out_r = torch.where((pick == 1).unsqueeze(-1), box_r, out_r)
    out_c = torch.where((pick == 2).unsqueeze(-1), quad_c, out_c)
    out_r = torch.where((pick == 2).unsqueeze(-1), quad_rem, out_r)

    # S2-2 exact-regime override (sup_act ⊇ sup_f via sup_quad)
    ident, zero = kobj.exact_regimes(arg)
    if bool(ident.any()):
        f_emb = _embed_rows(_embed_rows(coeffs, spec.q_emb_f,
                                        spec.sup_quad.size),
                            spec.emb_quad, size)
        out_c = torch.where(ident.unsqueeze(-1), f_emb, out_c)
        out_r = torch.where(ident.unsqueeze(-1), rem, out_r)
    if bool(zero.any()):
        out_c = torch.where(zero.unsqueeze(-1), torch.zeros_like(out_c), out_c)
        out_r = torch.where(zero.unsqueeze(-1), torch.zeros_like(out_r), out_r)
    return out_c, out_r, torch.zeros(R, dtype=torch.bool, device=coeffs.device)


def _accum_net_sup(eng: sp.SparseEngine, key, sup: sp.Support) -> sp.Support:
    """Monotone per-(net, layer) support accumulation: the returned support
    only ever GROWS across control steps, keeping ActSpec/PairBind keys
    stable (supersets are exact — structural zeros)."""
    store = _eng_cache(eng, "_nn_sups")
    prev = store.get(key)
    if prev is None or prev == sup:
        store[key] = sup if prev is None else prev
        return store[key]
    su, _, _ = eng.union(prev, sup)
    store[key] = su
    return su


def nn_tm_bounds_s(
    layers: list,
    coeffs: torch.Tensor,
    sup_in: sp.Support,
    rem: torch.Tensor,
    k_series: int,
    eng: sp.SparseEngine,
    cutoff: float,
    net_key: str = "nn",
) -> tuple[torch.Tensor, sp.Support, torch.Tensor, torch.Tensor]:
    """Sparse twin of nn_tm.nn_tm_bounds.

    coeffs [B, nn_in, S_in] point on sup_in, rem [B, nn_in, 2] ->
    (u_c [B, m, S_out], sup_out, u_r [B, m, 2], ok [B]). Affine layers are
    layout-blind (per-slot linear combinations never change the support);
    activations run the sparse dispatch with row-chunking against the
    ActSpec's measured max pair count.
    """
    B = coeffs.shape[0]
    dev = coeffs.device
    tabs = el.elem_tables(k_series + 1, str(dev))
    def _grow(t: torch.Tensor, frm: sp.Support, to: sp.Support) -> torch.Tensor:
        if frm == to:
            return t
        emb = eng.embed_idx(frm, to)
        out = t.new_zeros(*t.shape[:-1], to.size)
        out[..., emb] = t
        return out

    sup_cur = _accum_net_sup(eng, (net_key, "in"), sup_in)
    coeffs = _grow(coeffs, sup_in, sup_cur)
    cur_c, cur_r = coeffs, rem
    ok = torch.ones(B, dtype=torch.bool, device=dev)
    li = 0
    for entry in layers:
        if entry[0] == "linear":
            _, W, b = entry
            W = W.to(dev)
            b = b.to(dev)
            cur_c = torch.einsum("oi,bis->bos", W, cur_c)
            cur_c[..., 0] += b
            cur_r = iv.dot_point_iv(W.unsqueeze(0), cur_r.unsqueeze(1), dim=-1)
        else:
            _, kind = entry
            grown_sup = _accum_net_sup(eng, (net_key, li), sup_cur)
            cur_c = _grow(cur_c, sup_cur, grown_sup)
            sup_cur = grown_sup
            spec = specialize_act(sup_cur, k_series, eng)
            w = cur_c.shape[1]
            flat_c = cur_c.reshape(B * w, -1)
            flat_r = cur_r.reshape(B * w, 2)
            chunk = max(1, int(_CHUNK_BUDGET_BYTES / (spec.max_pairs * 8 * 6)))
            if flat_c.shape[0] <= chunk:
                out_c, out_r, bad = _act_series_s(
                    kind, flat_c, flat_r, spec, k_series, eng, cutoff, tabs)
            else:
                pieces = [
                    _act_series_s(kind, flat_c[i:i + chunk],
                                  flat_r[i:i + chunk], spec, k_series, eng,
                                  cutoff, tabs)
                    for i in range(0, flat_c.shape[0], chunk)
                ]
                out_c = torch.cat([p[0] for p in pieces])
                out_r = torch.cat([p[1] for p in pieces])
                bad = torch.cat([p[2] for p in pieces])
            cur_c = out_c.reshape(B, w, -1)
            cur_r = out_r.reshape(B, w, 2)
            ok &= ~bad.reshape(B, w).any(dim=1)
            sup_cur = spec.sup_act
            li += 1
    return cur_c, sup_cur, cur_r, ok
