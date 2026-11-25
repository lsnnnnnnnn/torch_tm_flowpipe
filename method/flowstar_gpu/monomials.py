"""Monomial basis tables for dense batched Taylor-model arithmetic (plan D1).

The dense TM representation stores polynomial coefficients as one tensor slot
per monomial of the basis

    B(n, d) = { t^a0 * x1^a1 * ... * xn^an : a0 + ... + an <= d }

over the n+1 variables (t, x1..xn), ordered DEGREE-MAJOR GRADED-LEX: primary
key total degree ascending, secondary key the exponent tuple lexicographically.
Degree-major ordering gives the two structural properties everything relies on:

  * B(n, d') is a PREFIX of B(n, d) for d' <= d — so truncation to a lower
    degree is a contiguous slice, and the order-k basis is a prefix of the
    order-2k product basis;
  * per-degree blocks are contiguous — degree masks are cheap.

A `MonomialTables(n, k)` instance carries the EXTENDED basis of degree <= 2k
(size T2), because a product of two degree-<=k polynomials (size T) lands in
degree <= 2k before conservative truncation (mirroring Flow*: full product,
then ctrunc_normal moves the degree->k tail into the remainder).

All tables are built once on CPU with numpy (integer combinatorics only — no
rounding concerns) and moved to a torch device with `.to(device)`.

Determinism note (plan D9): polynomial products gather pair products
PRE-SORTED by output slot (`pair_i`/`pair_j`) and reduce them with
`torch.segment_reduce` over `seg_len` (see polynomial._segment_sum_with);
`index_add_` remains only for the integrate/evaluate-time scatters (the
integration targets are INJECTIVE; the evaluate-time gather's few collisions
ride on the deterministic index_add_ kernel). Per-slot accumulation keeps
error at the (m_o-1)·u segment level, and the tier-I Rump bound
`rounding.dot_error_bound(·, seg_len)` covers any within-slot order.
History: a cumsum-segment alternative was rejected — cumsum differences
accumulate error across the WHOLE pair vector, ~1e-7 relative in realistic
sizes, far beyond the 1e-12 step-1 parity gate — and a deterministic
`index_add_` accumulation, while bitwise-stable, measured ~12x slower than
segment_reduce on CUDA (polynomial._segment_sum_with).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import torch

# Spatial range categories for the closed-form intEvalNormal rule
# (Flow* Term::intEvalNormal, Term.h:165-201; see ARCHITECTURE.md):
# over the normalized domain x in [-1, 1]^n a monomial's spatial factor is
#   CAT_ONE  = [1, 1]  — every spatial exponent is 0
#   CAT_HALF = [0, 1]  — some nonzero exponent, all of them even
#   CAT_SYM  = [-1, 1] — at least one odd exponent
CAT_ONE, CAT_HALF, CAT_SYM = 0, 1, 2

# The three category intervals as a lookup table, indexed by category: [3, 2].
CATEGORY_INTERVALS = np.array([[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]])


def basis_size(nvars: int, maxdeg: int) -> int:
    """Number of monomials of total degree <= maxdeg in nvars variables:
    C(nvars + maxdeg, maxdeg)."""
    return math.comb(nvars + maxdeg, maxdeg)


def _enumerate_exponents(nvars: int, maxdeg: int) -> np.ndarray:
    """All exponent vectors with sum <= maxdeg, degree-major graded-lex sorted.

    Output: [T2, nvars] int64 where T2 = basis_size(nvars, maxdeg).
    """
    rows: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 0:
            rows.append(tuple(prefix))
            return
        for e in range(remaining + 1):
            prefix.append(e)
            rec(prefix, remaining - e, slots - 1)
            prefix.pop()

    rec([], maxdeg, nvars)
    exps = np.array(rows, dtype=np.int64)  # [T2, nvars], lex order within total-degree mix
    # Degree-major graded-lex: stable sort by (total degree, lex tuple). The
    # enumeration above is already lex-sorted, so a stable sort on total degree
    # alone yields the required order.
    order = np.argsort(exps.sum(axis=1), kind="stable")  # [T2]
    return exps[order]


def _radix_encode(exps: np.ndarray, radix: int) -> np.ndarray:
    """Encode exponent rows [P, nvars] to unique int64 keys via mixed radix.

    key = sum_v exps[:, v] * radix^v. Safe iff radix^nvars < 2^63 — asserted.
    Used for O(P log T2) product-index lookup with searchsorted.
    """
    nvars = exps.shape[1]
    if radix**nvars >= 2**63:
        raise ValueError(f"radix encoding overflow: {radix}^{nvars} >= 2^63")
    weights = radix ** np.arange(nvars, dtype=np.int64)  # [nvars]
    return exps @ weights  # [P]


@dataclass(frozen=True)
class MonomialTables:
    """Precomputed index tables for TM arithmetic at fixed (n, k).

    Naming: T = basis size at degree <= k (working basis), T2 at degree <= 2k
    (product basis, of which the working basis is the first T entries).
    Production tensors live on one device; `.to(device)` returns a moved copy
    (verification-only tables stay on CPU — see below).

    Attributes (shapes in brackets):
        n, k: state dimension / TM order.
        T, T2, Ts: working-basis, product-basis, spatial-sub-basis sizes.
        exponents [T2, n+1] i64: exponent vectors; column 0 is the t-exponent.
        t_deg [T2] i64: cached t-exponents (= exponents[:, 0]).
        prefix_len [2k+2] i64: prefix_len[d] = #monomials of total degree < d,
            i.e. basis of degree <= d is exponents[:prefix_len[d+1]]. (Stored
            with a leading 0 so both forms are one index away.)
        spatial_index [Ts] i64: indices (into the product basis) of the
            monomials with t_deg == 0 and total degree <= k — the spatial
            sub-basis used by composition (tmv is time-free).
        pair_i [P] i64, pair_j [P] i64, P = T*T: working-basis factor indices
            per product pair, PRE-SORTED by product-monomial slot; the
            polynomial product gathers a[..., pair_i] * b[..., pair_j] and
            reduces with torch.segment_reduce over seg_len
            (polynomial._segment_sum_with).
        seg_len [T2] i64: pairs landing on each product monomial — both the
            segment_reduce lengths and the per-slot reduction length m fed to
            rounding.dot_error_bound for interval products.
        cat_iv [T2, 2] f64: the spatial-category interval per monomial
            (CATEGORY_INTERVALS[spatial_cat]) — gathered once so range
            evaluation is a pure dot against precomputed factors.
        int_map [T2] i64: index of the t-integrated monomial (t-exp + 1) in the
            product basis, or -1 where integration would exceed degree 2k.
        int_scale [T2] f64: 1 / (t_deg + 1) in RN — tier-P integration scale
            (Flow* Polynomial::integral_time divides by degrees[0]+1 in RN too).
        int_scale_iv [T2, 2] f64: outward-rounded interval enclosure of the
            same reciprocal, for tier-I integration.
        eval_t_spatial [T2] i64: index in the SPATIAL sub-basis of this
            monomial with its t-part stripped, or -1 if the spatial part has
            degree > k (cannot happen for working-basis monomials; guard for
            product-basis use).
        device: torch device of the production tensors.

    Verification-only tables (never shipped to GPU — `to()` leaves them on
    CPU; no production reader, kept so tests can re-derive the kernel inputs
    independently):
        total_deg [T2] i64: cached total-degree sums.
        spatial_cat [T2] i64: CAT_* per monomial (t-exponent EXCLUDED) —
            production reads the derived cat_iv instead.
        pair_perm [P] i64: flattened pair indices (i*T + j) in the sorted
            order (pair_i = pair_perm // T, pair_j = pair_perm % T).
        seg_offsets [T2+1] i64: segment boundaries into the sorted pair list
            (seg_offsets[1:] - seg_offsets[:-1] == seg_len).
        pair_out [P] i64: product-monomial index per sorted pair (= the
            segment id each sorted pair belongs to).
    """

    n: int
    k: int
    T: int
    T2: int
    Ts: int
    exponents: torch.Tensor
    total_deg: torch.Tensor  # verification-only: never shipped to GPU
    t_deg: torch.Tensor
    spatial_cat: torch.Tensor  # verification-only: never shipped to GPU
    prefix_len: torch.Tensor
    spatial_index: torch.Tensor
    pair_perm: torch.Tensor  # verification-only: never shipped to GPU
    pair_i: torch.Tensor
    pair_j: torch.Tensor
    seg_offsets: torch.Tensor  # verification-only: never shipped to GPU
    pair_out: torch.Tensor  # verification-only: never shipped to GPU
    seg_len: torch.Tensor
    cat_iv: torch.Tensor
    int_map: torch.Tensor
    int_scale: torch.Tensor
    int_scale_iv: torch.Tensor
    eval_t_spatial: torch.Tensor
    # Spatial product tables (composition fast path): products of two t-free
    # degree<=k monomials live in the t-free degree<=2k sub-basis (size Ts2).
    # Same layout contract as the full tables: sp_pair_i/sp_pair_j index the
    # SPATIAL working basis (0..Ts-1), pairs pre-sorted by their output slot in
    # the spatial-2k basis (0..Ts2-1) for deterministic segment sums over
    # sp_seg_len; sp_prefix_len gives degree prefixes WITHIN the spatial
    # ordering; sp_cat_iv are the normal-domain range factors (pure category
    # intervals — no time powers, t_deg == 0 throughout).
    Ts2: int
    sp_pair_i: torch.Tensor
    sp_pair_j: torch.Tensor
    sp_seg_len: torch.Tensor
    sp_prefix_len: torch.Tensor
    sp_cat_iv: torch.Tensor
    device: str
    # Host-side mirrors of the prefix tables plus two structural validity
    # lengths, so hot-path truncation/integration never call .item()/.any()
    # on device tensors (each such call is a full host-device sync — profiled
    # at 122 syncs per advance() before these existed, the dominant B=1 cost):
    #   prefix_len_cpu / sp_prefix_len_cpu: tuple mirrors of the tensors above.
    #   eval_t_valid_len: largest m such that eval_t_spatial[:m] has no -1
    #       (a monomial's t-stripped image leaves the spatial basis only past
    #       degree k, so this is a pure build-time property).
    prefix_len_cpu: tuple = ()
    sp_prefix_len_cpu: tuple = ()
    eval_t_valid_len: int = 0

    def to(self, device: str) -> MonomialTables:
        """Copy with the production tensors moved to `device` (no-op if already
        there). Verification-only tables stay on CPU — tests read the CPU copy."""
        if device == self.device:
            return self
        moved = {
            f: getattr(self, f).to(device)
            for f in (
                "exponents",
                "t_deg",
                "prefix_len",
                "spatial_index",
                "pair_i",
                "pair_j",
                "seg_len",
                "cat_iv",
                "int_map",
                "int_scale",
                "int_scale_iv",
                "eval_t_spatial",
                "sp_pair_i",
                "sp_pair_j",
                "sp_seg_len",
                "sp_prefix_len",
                "sp_cat_iv",
            )
        }
        return replace(self, device=device, **moved)


@lru_cache(maxsize=32)
def build_tables(n: int, k: int) -> MonomialTables:
    """Build (and cache) the tables for n state variables at TM order k, on CPU.

    Pure integer combinatorics plus two carefully-rounded reciprocal tables;
    every step is deterministic, so the cache is safe process-wide.
    """
    if n < 1:
        raise ValueError(f"need n >= 1 state variables, got {n}")
    if k < 2:
        # Flow* enforces order >= 2 (settings); the tables would be legal for
        # k >= 1 but downstream Picard assumes k >= 2, so fail early.
        raise ValueError(f"need TM order k >= 2, got {k}")

    nvars = n + 1  # (t, x1..xn)
    exps = _enumerate_exponents(nvars, 2 * k)  # [T2, n+1]
    T2 = exps.shape[0]
    total = exps.sum(axis=1)  # [T2]
    tdeg = exps[:, 0].copy()  # [T2]
    T = basis_size(nvars, k)

    # prefix_len[d] = #monomials with total degree < d  (length 2k+2, leading 0)
    prefix = np.zeros(2 * k + 2, dtype=np.int64)
    counts = np.bincount(total, minlength=2 * k + 1)  # [2k+1]
    prefix[1:] = np.cumsum(counts)
    assert prefix[k + 1] == T, "degree-major prefix property violated"

    # Spatial categories (t-exponent excluded): CAT_SYM if any odd spatial
    # exponent, else CAT_HALF if any nonzero (even) spatial exponent, else CAT_ONE.
    spatial = exps[:, 1:]  # [T2, n]
    has_odd = (spatial % 2 == 1).any(axis=1)  # [T2]
    has_nonzero = (spatial > 0).any(axis=1)  # [T2]
    cat = np.where(has_odd, CAT_SYM, np.where(has_nonzero, CAT_HALF, CAT_ONE))  # [T2]

    # Spatial sub-basis: t-free monomials of the WORKING basis (degree <= k).
    spatial_index = np.nonzero((tdeg == 0) & (total <= k))[0]  # [Ts]
    Ts = spatial_index.shape[0]
    assert Ts == basis_size(n, k)

    # --- product pair tables -------------------------------------------------
    # Every ordered pair (i, j) of working-basis monomials multiplies into the
    # product basis: out_exp = exps[i] + exps[j], always of degree <= 2k.
    radix = 2 * k + 1  # any single exponent in the product basis is <= 2k
    keys2 = _radix_encode(exps, radix)  # [T2] keys of the product basis
    key_order = np.argsort(keys2)  # [T2] (keys are unique)
    keys2_sorted = keys2[key_order]

    # Pair keys WITHOUT materializing the [T, T, n+1] exponent-sum tensor
    # (1 TB at quadrotor's T=3060): mixed-radix encoding is linear, so
    # key(a + b) = key(a) + key(b) whenever no per-variable digit overflows —
    # guaranteed here because radix = 2k+1 exceeds every product exponent.
    wkeys = _radix_encode(exps[:T], radix)  # [T]
    pair_keys = (wkeys[:, None] + wkeys[None, :]).reshape(-1)  # [P], P = T*T
    pos = np.searchsorted(keys2_sorted, pair_keys)  # [P]
    pair_out = key_order[pos]  # [P] product-basis index per pair
    assert (keys2[pair_out] == pair_keys).all(), "product lookup failed"

    # Sort pairs by output monomial -> segment-sum layout (stable: keeps the
    # (i, j) lex order within a segment fixed, which pins the cumsum order and
    # hence bitwise reproducibility of tier-P products).
    perm = np.argsort(pair_out, kind="stable")  # [P]
    pair_out_sorted = pair_out[perm]  # [P]
    seg_len = np.bincount(pair_out_sorted, minlength=T2)  # [T2]
    seg_offsets = np.zeros(T2 + 1, dtype=np.int64)
    seg_offsets[1:] = np.cumsum(seg_len)
    pair_i = perm // T  # [P] left working-basis factor per sorted pair
    pair_j = perm % T  # [P] right factor

    # --- integration in t ----------------------------------------------------
    # \int_0^t: exponent[0] += 1 (degree + 1), coefficient /= (t_deg + 1).
    int_exp = exps.copy()
    int_exp[:, 0] += 1
    int_keys = _radix_encode(int_exp, radix + 1)  # radix+1: exponents reach 2k+1
    keys2b = _radix_encode(exps, radix + 1)
    order_b = np.argsort(keys2b)
    pos_b = np.searchsorted(keys2b[order_b], int_keys)
    pos_b_clipped = np.clip(pos_b, 0, T2 - 1)
    cand = order_b[pos_b_clipped]  # [T2] candidate target index
    ok = (total + 1 <= 2 * k) & (keys2b[cand] == int_keys)  # [T2]
    int_map = np.where(ok, cand, -1)  # [T2]

    # Tier-P scale: RN double division, exactly what Flow* does on Real.
    int_scale = 1.0 / (tdeg + 1.0)  # [T2] RN
    # Tier-I enclosure of 1/(d0+1): one RN division per endpoint, stepped
    # outward one ulp with numpy nextafter (CPU build time — cheap and exact).
    lo_iv = np.nextafter(int_scale, -np.inf)
    hi_iv = np.nextafter(int_scale, np.inf)
    exact = (tdeg + 1) & (tdeg)  # power-of-two denominators divide exactly:
    # (d0+1) is a power of two  <=>  (d0+1) & d0 == 0  -> RN result is exact,
    # keep the tight point value instead of the 1-ulp inflation.
    pow2 = exact == 0
    int_scale_iv = np.stack(
        (np.where(pow2, int_scale, lo_iv), np.where(pow2, int_scale, hi_iv)), axis=1
    )  # [T2, 2]

    # --- evaluate-time gather ------------------------------------------------
    # Substituting t := delta maps monomial (a0, s) to spatial monomial s (times
    # delta^a0). Target index within the SPATIAL sub-basis order.
    strip_exp = exps.copy()
    strip_exp[:, 0] = 0
    strip_keys = _radix_encode(strip_exp, radix)
    spatial_keys = keys2[spatial_index]  # [Ts] (spatial basis keys, sorted? not nec.)
    sp_order = np.argsort(spatial_keys)
    pos_s = np.searchsorted(spatial_keys[sp_order], strip_keys)
    pos_s_clipped = np.clip(pos_s, 0, Ts - 1)
    cand_s = sp_order[pos_s_clipped]
    ok_s = spatial_keys[cand_s] == strip_keys
    eval_t_spatial = np.where(ok_s, cand_s, -1)  # [T2]

    # --- spatial product tables (composition fast path) ---------------------
    # Spatial-2k basis: t-free monomials of the product basis, in inherited
    # (degree-major) order. The spatial working basis (deg <= k) is its prefix.
    sp2_index = np.nonzero(tdeg == 0)[0]  # [Ts2] full-basis indices
    Ts2 = sp2_index.shape[0]
    sp_pos = -np.ones(T2, dtype=np.int64)
    sp_pos[sp2_index] = np.arange(Ts2)  # full idx -> spatial-2k idx
    assert (sp_pos[spatial_index] == np.arange(Ts)).all(), "spatial prefix violated"

    sp_exps = exps[sp2_index][:, 1:]  # [Ts2, n] spatial exponents
    sp_total = sp_exps.sum(axis=1)
    sp_prefix = np.zeros(2 * k + 2, dtype=np.int64)
    sp_prefix[1:] = np.cumsum(np.bincount(sp_total, minlength=2 * k + 1))

    wsp = sp_exps[:Ts]  # [Ts, n] working spatial basis
    # Same linearity trick as the full-basis pair keys (no [Ts, Ts, n] blowup).
    radix_s = 2 * k + 1
    sp_keys2 = _radix_encode(sp_exps, radix_s)
    sp_order = np.argsort(sp_keys2)
    wsp_keys = _radix_encode(wsp, radix_s)  # [Ts]
    sp_pair_keys = (wsp_keys[:, None] + wsp_keys[None, :]).reshape(-1)  # [Ps]
    sp_pos2 = np.searchsorted(sp_keys2[sp_order], sp_pair_keys)
    sp_pair_out_raw = sp_order[sp_pos2]  # [Ps], Ps = Ts*Ts
    assert (sp_keys2[sp_pair_out_raw] == sp_pair_keys).all()
    sp_perm = np.argsort(sp_pair_out_raw, kind="stable")
    sp_pair_out = sp_pair_out_raw[sp_perm]
    sp_seg_len = np.bincount(sp_pair_out, minlength=Ts2)
    sp_pair_i = sp_perm // Ts
    sp_pair_j = sp_perm % Ts

    tt = torch.from_numpy

    return MonomialTables(
        n=n,
        k=k,
        T=T,
        T2=T2,
        Ts=Ts,
        exponents=tt(exps),
        total_deg=tt(total),
        t_deg=tt(tdeg),
        spatial_cat=tt(cat.astype(np.int64)),
        prefix_len=tt(prefix),
        spatial_index=tt(spatial_index),
        pair_perm=tt(perm),
        pair_i=tt(pair_i),
        pair_j=tt(pair_j),
        seg_offsets=tt(seg_offsets),
        pair_out=tt(pair_out_sorted),
        seg_len=tt(seg_len),
        cat_iv=tt(CATEGORY_INTERVALS[cat]),
        int_map=tt(int_map),
        int_scale=tt(int_scale),
        int_scale_iv=tt(int_scale_iv),
        eval_t_spatial=tt(eval_t_spatial),
        Ts2=Ts2,
        sp_pair_i=tt(sp_pair_i),
        sp_pair_j=tt(sp_pair_j),
        sp_seg_len=tt(sp_seg_len),
        sp_prefix_len=tt(sp_prefix),
        sp_cat_iv=tt(CATEGORY_INTERVALS[cat[sp2_index]]),
        device="cpu",
        prefix_len_cpu=tuple(int(v) for v in prefix),
        sp_prefix_len_cpu=tuple(int(v) for v in sp_prefix),
        eval_t_valid_len=int(
            np.argmax(eval_t_spatial < 0) if (eval_t_spatial < 0).any() else T2
        ),
    )
