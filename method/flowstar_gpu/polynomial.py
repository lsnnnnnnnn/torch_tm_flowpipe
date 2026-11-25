"""Dense batched polynomial operations over the monomial basis (plan D1/D3).

A "polynomial batch" is a coefficient tensor whose LAST dimension indexes the
monomial basis of a `MonomialTables` instance:

    point coefficients   : [..., M]      (tier P — Flow*'s Real)
    interval coefficients: [..., M, 2]   (tier I)

where M is either T (working basis, degree <= k) or T2 (product basis, degree
<= 2k); every function documents which. Leading dims are arbitrary (typically
[B, n] = batch x state-dim).

Flow* correspondence (ALGORITHM.md): `mul`* = Polynomial::operator* followed by
the caller's ctrunc/cutoff; `ctrunc_*` = Polynomial::ctrunc_normal;
`cutoff_*` = Polynomial::cutoff_normal (Real-coefficient semantics);
`range_normal` = Polynomial::intEvalNormal (closed form); `integrate_t` =
Polynomial::integral_time; `evaluate_time_end` = TaylorModelVec::evaluate_time
at the step endpoint.

Step tables: `StepTables` plays the role of Flow*'s step_exp_table /
step_end_exp_table pair (settings.cpp construct_step_exp_table): interval
powers of [0, delta] for range evaluation, point powers of delta for endpoint
evaluation — both precomputed once per (settings, tables) and reused every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import interval as iv
from .monomials import MonomialTables
from .rounding import dot_error_bound, next_down, next_up

# ---------------------------------------------------------------------------
# Step tables ([0, delta]^j and delta^j)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepTables:
    """Per-(delta, tables) precomputed time-power tables.

    Two layouts, selected by `lanes` (M6 adaptive stepping made the step size
    per-lane; fixed-step runs keep the original unbatched layout):

        lanes == 0 (fixed delta, shared across the batch):
            pow_iv / end_iv [2k+1, 2], end_pow [2k+1], factor_iv [T2, 2]
        lanes == B (per-lane deltas):
            pow_iv / end_iv [B, 2k+1, 2], end_pow [B, 2k+1], factor_iv [B, T2, 2]

    Consumers MUST go through the accessor methods (factor / end_gather / dt),
    which return broadcast-ready views for [B, n, M]-shaped coefficient
    tensors in both layouts — never index the raw fields by monomial directly.

    Field semantics: pow_iv = interval powers [0, delta]^j via successive
    interval multiplication (Flow* construct_step_exp_table structure);
    end_pow = RN point powers delta^j (step_end_exp_table is Real in Flow*);
    end_iv = outward interval powers of the binary64 point delta, used only
    by strict endpoint error evaluation (distinct from the tube [0, delta]);
    factor_iv = cat_iv[m] * [0, delta]^t_deg[m] — the closed-form intEvalNormal
    weight. `delta` keeps the scalar step for lanes == 0; batched tables carry
    the per-lane deltas tensor instead.
    """

    delta: float
    pow_iv: torch.Tensor
    end_pow: torch.Tensor
    factor_iv: torch.Tensor
    end_iv: torch.Tensor
    lanes: int = 0
    deltas: torch.Tensor | None = None  # [B] per-lane steps when lanes > 0

    def factor(self, lo: int, hi: int, ref_rank: int = 3) -> torch.Tensor:
        """Range factors for monomials [lo, hi), broadcast-ready.

        ref_rank = rank of the consuming coefficient tensor WITHOUT its
        endpoint dim (e.g. 3 for [B, n, M], 2 for a per-slot [B, M]). Fixed
        layout returns [M', 2] (right-aligned broadcast is always correct);
        batched layout returns [B, 1...1, M', 2] with ref_rank-2 singletons so
        the lane dim lines up on the left.
        """
        if self.lanes == 0:
            return self.factor_iv[lo:hi]
        f = self.factor_iv[:, lo:hi]  # [B, M', 2]
        for _ in range(ref_rank - 2):
            f = f.unsqueeze(1)
        return f

    def end_gather(self, t_deg: torch.Tensor, ref_rank: int = 3) -> torch.Tensor:
        """delta^t_deg[m] point powers: t_deg [M] -> [M] or [B, 1...1, M]."""
        if self.lanes == 0:
            return self.end_pow[t_deg]
        e = self.end_pow[:, t_deg]  # [B, M]
        for _ in range(ref_rank - 2):
            e = e.unsqueeze(1)
        return e

    def end_iv_gather(self, t_deg: torch.Tensor, ref_rank: int = 3) -> torch.Tensor:
        """Strict endpoint powers; end_gather's layout plus an interval axis."""
        if self.lanes == 0:
            return self.end_iv[t_deg]
        e = self.end_iv[:, t_deg]  # [B, M, 2]
        for _ in range(ref_rank - 2):
            e = e.unsqueeze(1)
        return e

    def dt(self, ref_rank: int = 2) -> torch.Tensor:
        """The interval [0, delta]: [2] or [B, 1...1, 2].

        ref_rank = endpoint-stripped rank of the consumer (2 for [B, n, 2]).
        Unlike factor/end_gather, dt carries NO trailing data dim, so it
        inserts ref_rank-1 singletons after the lane dim ([B, n] consumer ->
        [B, 1, 2] here).
        """
        if self.lanes == 0:
            return self.pow_iv[1]
        d = self.pow_iv[:, 1]  # [B, 2]
        for _ in range(ref_rank - 1):
            d = d.unsqueeze(1)
        return d


def build_step_tables(tables: MonomialTables, delta: float) -> StepTables:
    """Build the (unbatched) time-power tables for step size `delta`."""
    if delta <= 0:
        raise ValueError(f"step size must be positive, got {delta}")
    dev = tables.exponents.device
    base = torch.tensor([0.0, delta], dtype=torch.float64, device=dev)  # [2] = [0, delta]
    pow_iv = iv.powers(base, 2 * tables.k)  # [2k+1, 2]

    # RN point powers of delta by successive multiplication (matches Flow*'s
    # intProd *= intStep structure taken at the sup endpoint).
    end = [torch.tensor(1.0, dtype=torch.float64, device=dev)]
    for _ in range(2 * tables.k):
        end.append(end[-1] * delta)
    end_pow = torch.stack(end)  # [2k+1]
    end_iv = iv.powers(iv.from_point(base[1]), 2 * tables.k)  # [2k+1, 2]

    factor_iv = iv.mul(tables.cat_iv, pow_iv[tables.t_deg])  # [T2, 2]
    return StepTables(delta=delta, pow_iv=pow_iv, end_pow=end_pow,
                      factor_iv=factor_iv, end_iv=end_iv)


def build_step_tables_batched(tables: MonomialTables, deltas: torch.Tensor) -> StepTables:
    """Per-lane tables for adaptive stepping: deltas [B] -> lanes == B layout.

    Same successive-multiplication structure as the fixed builder, vectorized
    over lanes. Cheap (2k interval muls on [B, 2] + one gather) — rebuilt
    whenever any lane's step changes rather than cached per delta value.
    """
    if bool((deltas <= 0).any()):
        raise ValueError("all step sizes must be positive")
    B = deltas.shape[0]
    base = torch.stack((torch.zeros_like(deltas), deltas), dim=-1)  # [B, 2] = [0, delta]
    pow_iv = iv.powers(base, 2 * tables.k).movedim(0, 1)  # [B, 2k+1, 2]

    end = [torch.ones_like(deltas)]
    for _ in range(2 * tables.k):
        end.append(end[-1] * deltas)
    end_pow = torch.stack(end, dim=1)  # [B, 2k+1]
    end_iv = iv.powers(iv.from_point(deltas), 2 * tables.k).movedim(0, 1)

    # cat_iv [T2, 2] x per-lane time powers gathered by t_deg -> [B, T2, 2].
    factor_iv = iv.mul(tables.cat_iv.unsqueeze(0), pow_iv[:, tables.t_deg])
    return StepTables(
        delta=float("nan"), pow_iv=pow_iv, end_pow=end_pow, factor_iv=factor_iv,
        end_iv=end_iv,
        lanes=B, deltas=deltas,
    )


# ---------------------------------------------------------------------------
# Range evaluation (closed-form intEvalNormal)
# ---------------------------------------------------------------------------


def range_normal(coeffs: torch.Tensor, tables: MonomialTables, step: StepTables) -> torch.Tensor:
    """Range of a point-coefficient polynomial over [0, delta] x [-1, 1]^n.

    coeffs [..., M] with M <= T2 (a PREFIX of the product basis — degree-major
    ordering makes every truncation a prefix) -> interval [..., 2].

    This is Flow*'s hot-path Polynomial::intEvalNormal: per monomial the
    normal-domain factor is precomputed (step.factor_iv), so the whole range is
    one rigorous point-x-interval dot product.
    """
    m_basis = coeffs.shape[-1]
    return iv.dot_point_iv(coeffs, step.factor(0, m_basis, coeffs.dim()), dim=-1)


def range_normal_iv(
    coeffs_iv: torch.Tensor, tables: MonomialTables, step: StepTables
) -> torch.Tensor:
    """Range of an INTERVAL-coefficient polynomial over the normal domain.

    coeffs_iv [..., M, 2] -> [..., 2]: sum_m coeffs_iv[m] * factor_iv[m],
    rigorous interval multiply then interval sum (Rump-bounded reduction).
    """
    m_basis = coeffs_iv.shape[-2]
    prod = iv.mul(coeffs_iv, step.factor(0, m_basis, coeffs_iv.dim() - 1))  # [..., M, 2]
    return iv.sum(prod, dim=-1)  # dim indexes the endpoint-stripped shape


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------

# Memo of expanded segment-length tensors keyed by (leading shape, device):
# segment_reduce needs a lengths tensor matching the operand's leading dims;
# the same shapes recur every step, so materialize once. (Never grows beyond a
# handful of entries per run.)
_seglen_cache: dict[tuple, torch.Tensor] = {}


def _segment_sum_with(
    prods: torch.Tensor, seg_len: torch.Tensor, n_out: int, tag: str
) -> torch.Tensor:
    """Deterministic per-slot sum of sorted pair products: [..., P] -> [..., n_out].

    Uses torch.segment_reduce over a pre-sorted pair layout. WHY not
    index_add_: under torch.use_deterministic_algorithms the CUDA index_add_
    kernel is ~12x slower (measured 171ms vs 18ms at [1024, 21, P=15876]);
    segment_reduce is contention-free by construction. segment_reduce is a
    beta op with no documented determinism guarantee, so our determinism test
    suite asserts bitwise stability across perturbed reruns on every device
    (tests/unit/test_determinism.py) — that empirical gate is the authority.

    The `tag` MUST identify the segmentation (basis family + (n, k)): equal
    leading shapes over different bases would otherwise collide in the memo —
    a collision here once produced silently wrong products, caught by the
    sympy-exactness property test.
    """
    key = (tag, n_out, prods.shape[:-1], str(prods.device))
    lengths = _seglen_cache.get(key)
    if lengths is None:
        lengths = seg_len.expand(*prods.shape[:-1], n_out).contiguous()
        _seglen_cache[key] = lengths
    return torch.segment_reduce(prods, "sum", lengths=lengths, axis=prods.dim() - 1)


def _segment_sum(prods: torch.Tensor, tables: MonomialTables) -> torch.Tensor:
    """Full-basis segment sum: [..., P] -> [..., T2] (see _segment_sum_with)."""
    return _segment_sum_with(
        prods, tables.seg_len, tables.T2, f"full:{tables.n}:{tables.k}"
    )


def mul_point(a: torch.Tensor, b: torch.Tensor, tables: MonomialTables) -> torch.Tensor:
    """Full product of two point-coefficient polynomials (tier P).

    a [..., T], b [..., T] -> [..., T2]. Pure RN arithmetic — EXACTLY what
    Flow* does on Real coefficients (its coefficient roundoff here is unbounded
    too, GOTCHAS #15). No _err variant is needed: point-coefficient roundoff in
    the Picard tier is bounded by Flow*'s own intDifferences mechanism — the
    validated pass re-derives the polynomial with interval coefficients and
    ranges the point-vs-interval difference (stage (i) of advance()), which
    absorbs any RN accumulation error made here.

    Deterministic: pair products are gathered pre-sorted by output slot
    (tables.pair_i / pair_j) and segment-summed per slot (see _segment_sum).
    """
    prods = a[..., tables.pair_i] * b[..., tables.pair_j]  # [..., P] RN
    return _segment_sum(prods, tables)


def mul_iv(a_iv: torch.Tensor, b_iv: torch.Tensor, tables: MonomialTables) -> torch.Tensor:
    """Full product of two INTERVAL-coefficient polynomials (tier I).

    a_iv [..., T, 2], b_iv [..., T, 2] -> [..., T2, 2].

    Per sorted pair, a rigorous interval multiply; per output slot, endpoint
    sums accumulate in RN via deterministic segment sums and are pushed outward
    by the per-slot Rump bound (reduction length = tables.seg_len), which keeps
    ANY accumulation order sound.
    """
    pa = a_iv[..., tables.pair_i, :]  # [..., P, 2]
    pb = b_iv[..., tables.pair_j, :]  # [..., P, 2]
    p = iv.mul(pa, pb)  # [..., P, 2] rigorous per-pair product

    lo_hat = _segment_sum(p[..., 0], tables)  # [..., T2]
    hi_hat = _segment_sum(p[..., 1], tables)
    mag_hat = _segment_sum(iv.mag(p), tables)
    err = dot_error_bound(mag_hat, tables.seg_len)  # [..., T2]
    return torch.stack((next_down(lo_hat - err), next_up(hi_hat + err)), dim=-1)


# ---------------------------------------------------------------------------
# Truncation and cutoff (both move dropped mass into a remainder interval)
# ---------------------------------------------------------------------------


def ctrunc_normal(
    coeffs: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservative truncation to total degree <= order (Flow* ctrunc_normal).

    coeffs [..., M] (point, M <= T2) -> (kept [..., K], tail_range [..., 2])
    where K = prefix_len[order+1]: the degree-major prefix property makes the
    kept part a contiguous slice; the dropped suffix is interval-evaluated over
    the normal domain and returned for the caller to add into a remainder.
    """
    keep = tables.prefix_len_cpu[order + 1]
    m_basis = coeffs.shape[-1]
    if keep >= m_basis:
        # Nothing to drop: zero tail contribution.
        zero = torch.zeros(*coeffs.shape[:-1], 2, dtype=coeffs.dtype, device=coeffs.device)
        return coeffs, zero
    dropped = coeffs[..., keep:]  # [..., M-K]
    tail = iv.dot_point_iv(dropped, step.factor(keep, m_basis, dropped.dim()), dim=-1)  # [..., 2]
    return coeffs[..., :keep], tail


def ctrunc_normal_iv(
    coeffs_iv: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval-coefficient version of `ctrunc_normal`.

    coeffs_iv [..., M, 2] -> (kept [..., K, 2], tail_range [..., 2]).
    """
    keep = tables.prefix_len_cpu[order + 1]
    m_basis = coeffs_iv.shape[-2]
    if keep >= m_basis:
        zero = torch.zeros(*coeffs_iv.shape[:-2], 2, dtype=coeffs_iv.dtype, device=coeffs_iv.device)
        return coeffs_iv, zero
    dropped = coeffs_iv[..., keep:, :]  # [..., M-K, 2]
    prod = iv.mul(dropped, step.factor(keep, m_basis, dropped.dim() - 1))  # [..., M-K, 2]
    tail = iv.sum(prod, dim=-1)  # [..., 2] (endpoint-stripped dim convention)
    return coeffs_iv[..., :keep, :], tail


def cutoff_normal(
    coeffs: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move small-coefficient monomials into the remainder (Flow* cutoff_normal).

    coeffs [..., M] (point) -> (kept [..., M] with small slots zeroed,
    dropped_range [..., 2]).

    Semantics for Real coefficients (Term.h:398, Term<Real>::cutoff): a term is
    dropped iff its coefficient lies in [-eps, eps] (closed — subseteq), i.e.
    |c| <= eps. NOTE: Flow* applies this to EVERY term including the constant
    (no exemption in Polynomial::cutoff_normal's term loop), so we do too —
    parity over aesthetics.
    """
    m_basis = coeffs.shape[-1]
    small = coeffs.abs() <= cutoff_threshold  # [..., M] bool
    dropped = torch.where(small, coeffs, torch.zeros_like(coeffs))  # [..., M]
    rng = iv.dot_point_iv(dropped, step.factor(0, m_basis, dropped.dim()), dim=-1)  # [..., 2]
    kept = torch.where(small, torch.zeros_like(coeffs), coeffs)
    return kept, rng


def nctrunc(coeffs: torch.Tensor, tables: MonomialTables, order: int) -> torch.Tensor:
    """NON-conservative truncation: discard degree > order terms (Flow* nctrunc).

    coeffs [..., M] -> [..., K], K = prefix_len[order+1]. Used ONLY in the
    polynomial-only Picard pass (Flow* evaluate_no_remainder), where dropped
    mass is recovered later by the validated pass — no remainder bookkeeping.
    """
    keep = tables.prefix_len_cpu[order + 1]
    return coeffs[..., :keep]


def cutoff_drop(coeffs: torch.Tensor, cutoff_threshold: float) -> torch.Tensor:
    """Drop-variant cutoff: zero coefficients with |c| <= eps, record nothing.

    Mirrors Polynomial::cutoff(threshold) (single-argument, no remainder) as
    used inside mul_no_remainder. coeffs [..., M] -> [..., M].
    """
    small = coeffs.abs() <= cutoff_threshold  # [..., M]
    return torch.where(small, torch.zeros_like(coeffs), coeffs)


def cutoff_normal_interval(
    coeffs_iv: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    max_width: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cutoff for INTERVAL coefficients — Term<Interval>::cutoff (Term.h:412).

    coeffs_iv [..., M, 2] -> (kept [..., M, 2], dropped_range [..., 2]).

    Flow* order of tests (MAX_WIDTH first):
      1. width >= MAX_WIDTH (1e-12): SPLIT — keep only a thin midpoint
         enclosure as the coefficient; the radius part's normal-domain range
         goes to the remainder. (This is what stops interval coefficients from
         inflating during the validated pass.)
      2. else if the midpoint enclosure lies in [-eps, eps]: DROP the whole
         term into the remainder.
      3. else keep unchanged.

    Midpoint enclosure per Flow* Interval::midpoint(Interval&):
    [((lo+up)/2) rounded down, ((lo+up)/2) rounded up] — we emulate the
    directed rounding with one nextafter step each way (sound superset).
    """
    lo_c, hi_c = coeffs_iv[..., 0], coeffs_iv[..., 1]  # [..., M]
    m_basis = coeffs_iv.shape[-2]

    w = next_up(hi_c - lo_c)  # [..., M] width, rounded up (Flow* width())
    mid_rn = (lo_c + hi_c) * 0.5  # [..., M] RN midpoint
    mid_iv = torch.stack((next_down(mid_rn), next_up(mid_rn)), dim=-1)  # [..., M, 2]

    wide = w >= max_width  # [..., M] case 1
    small = (~wide) & (mid_iv[..., 0] >= -cutoff_threshold) & (
        mid_iv[..., 1] <= cutoff_threshold
    )  # [..., M] case 2 (midpoint enclosure inside the closed threshold box)

    # Case-1 radius part: original coeff minus the midpoint enclosure
    # (interval subtraction — same widening Flow*'s remove_midpoint performs).
    radius = iv.sub(coeffs_iv, mid_iv)  # [..., M, 2]

    # Assemble the polynomial whose range goes into the remainder:
    #   wide terms contribute their radius, small terms their whole coefficient.
    zeros2 = torch.zeros_like(coeffs_iv)
    to_rem = torch.where(wide.unsqueeze(-1), radius, zeros2)
    to_rem = torch.where(small.unsqueeze(-1), coeffs_iv, to_rem)  # [..., M, 2]
    prod = iv.mul(to_rem, step.factor(0, m_basis, to_rem.dim() - 1))  # [..., M, 2]
    dropped_range = iv.sum(prod, dim=-1)  # [..., 2]

    kept = torch.where(wide.unsqueeze(-1), mid_iv, coeffs_iv)
    kept = torch.where(small.unsqueeze(-1), zeros2, kept)
    return kept, dropped_range


# ---------------------------------------------------------------------------
# Spatial-basis fast path (composition: everything is t-free)
# ---------------------------------------------------------------------------
# Products of two degree<=k SPATIAL monomials live in the t-free degree<=2k
# sub-basis (size Ts2 << T2). All range factors are pure category intervals
# (t_deg == 0 => the [0, delta]^0 = [1,1] time power drops out), so these
# variants match the full-basis semantics EXACTLY on t-free inputs while
# moving ~5x less memory (P = Ts^2 vs T^2 pairs).


def mul_point_spatial(a: torch.Tensor, b: torch.Tensor, tables: MonomialTables) -> torch.Tensor:
    """Spatial tier-P product: a [..., Ts], b [..., Ts] -> [..., Ts2]."""
    prods = a[..., tables.sp_pair_i] * b[..., tables.sp_pair_j]  # [..., Ps] RN
    return _segment_sum_with(
        prods, tables.sp_seg_len, tables.Ts2, f"sp:{tables.n}:{tables.k}"
    )


def mul_iv_spatial(
    a_iv: torch.Tensor, b_iv: torch.Tensor, tables: MonomialTables
) -> torch.Tensor:
    """Rigorous spatial coefficient product.

    This is the t-free counterpart of :func:`mul_iv`.  It is used by strict
    composition to enclose the exact coefficient generated from the stored
    binary64 operands while preserving the ordinary tier-P point result.
    """
    pa = a_iv[..., tables.sp_pair_i, :]
    pb = b_iv[..., tables.sp_pair_j, :]
    products = iv.mul(pa, pb)
    tag = f"sp:{tables.n}:{tables.k}"
    lo_hat = _segment_sum_with(products[..., 0], tables.sp_seg_len, tables.Ts2, tag)
    hi_hat = _segment_sum_with(products[..., 1], tables.sp_seg_len, tables.Ts2, tag)
    mag_hat = _segment_sum_with(iv.mag(products), tables.sp_seg_len, tables.Ts2, tag)
    error = dot_error_bound(mag_hat, tables.sp_seg_len)
    return torch.stack((next_down(lo_hat - error), next_up(hi_hat + error)), dim=-1)


def range_normal_iv_spatial(
    coeffs_iv: torch.Tensor, tables: MonomialTables
) -> torch.Tensor:
    """Range a t-free interval-coefficient polynomial on ``[-1,1]^n``."""
    m_basis = coeffs_iv.shape[-2]
    terms = iv.mul(coeffs_iv, tables.sp_cat_iv[:m_basis])
    return iv.sum(terms, dim=-1)


def mul_point_spatial_with_roundoff(
    a: torch.Tensor, b: torch.Tensor, tables: MonomialTables
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tier-P spatial product plus its strict coefficient-error range.

    The point coefficients are bit-identical to :func:`mul_point_spatial`.
    Independently, degenerate interval inputs are multiplied by the rigorous
    tier-I path.  Subtracting the retained point coefficient yields an
    interval polynomial enclosing *only* coefficient formation roundoff; its
    range is returned for insertion exactly once into the ordinary remainder.
    """
    point = mul_point_spatial(a, b, tables)
    exact_coeff = mul_iv_spatial(iv.from_point(a), iv.from_point(b), tables)
    coefficient_error = iv.sub(exact_coeff, iv.from_point(point))
    return point, range_normal_iv_spatial(coefficient_error, tables)


def range_normal_spatial(coeffs: torch.Tensor, tables: MonomialTables) -> torch.Tensor:
    """Range of a t-free point polynomial over [-1, 1]^n: [..., M<=Ts2] -> [..., 2]."""
    m_basis = coeffs.shape[-1]
    return iv.dot_point_iv(coeffs, tables.sp_cat_iv[:m_basis], dim=-1)


def ctrunc_normal_spatial(
    coeffs: torch.Tensor, tables: MonomialTables, order: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatial conservative truncation: [..., M] -> ([..., K], tail [..., 2])."""
    keep = tables.sp_prefix_len_cpu[order + 1]
    m_basis = coeffs.shape[-1]
    if keep >= m_basis:
        zero = torch.zeros(*coeffs.shape[:-1], 2, dtype=coeffs.dtype, device=coeffs.device)
        return coeffs, zero
    tail = iv.dot_point_iv(coeffs[..., keep:], tables.sp_cat_iv[keep:m_basis], dim=-1)
    return coeffs[..., :keep], tail


def cutoff_normal_spatial(
    coeffs: torch.Tensor, tables: MonomialTables, cutoff_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatial cutoff: small coeffs -> remainder. [..., M] -> ([..., M], [..., 2])."""
    m_basis = coeffs.shape[-1]
    small = coeffs.abs() <= cutoff_threshold  # [..., M]
    dropped = torch.where(small, coeffs, torch.zeros_like(coeffs))
    rng = iv.dot_point_iv(dropped, tables.sp_cat_iv[:m_basis], dim=-1)
    kept = torch.where(small, torch.zeros_like(coeffs), coeffs)
    return kept, rng


# ---------------------------------------------------------------------------
# Integration and time evaluation
# ---------------------------------------------------------------------------


def integrate_t(
    coeffs: torch.Tensor, tables: MonomialTables, max_deg_in: int
) -> torch.Tensor:
    """\\int_0^t of a point-coefficient polynomial (Flow* integral_time, tier P).

    coeffs [..., M] with all nonzero mass at total degree <= max_deg_in
    (< 2k, caller guarantees — Picard integrates degree i-1 truncations) ->
    [..., M'] on the degree <= max_deg_in+1 prefix, where each monomial's
    coefficient is c / (t_deg+1) placed at its t-bumped monomial.

    The scatter targets int_map[m], which is INJECTIVE (t-bump is injective),
    so index_copy-style assignment applies: no duplicate accumulation at all.
    """
    m_in = tables.prefix_len_cpu[max_deg_in + 1]
    if coeffs.shape[-1] < m_in:
        m_in = coeffs.shape[-1]
    src = coeffs[..., :m_in] * tables.int_scale[:m_in]  # [..., m_in] RN scale
    tgt = tables.int_map[:m_in]  # [m_in], all valid (degree < 2k by contract)
    # int_map[m] == -1 iff total_deg[m] == 2k, and the [:m_in] prefix contains
    # a degree-2k monomial iff m_in > prefix_len[2k] — a pure host-int test
    # (no device sync; the tensor scan this replaces was a per-call sync).
    if m_in > tables.prefix_len_cpu[2 * tables.k]:
        raise ValueError("integrate_t: input polynomial exceeds degree 2k-1")
    m_out = tables.prefix_len_cpu[max_deg_in + 2]
    out = torch.zeros(*coeffs.shape[:-1], m_out, dtype=coeffs.dtype, device=coeffs.device)
    out.index_add_(-1, tgt, src)  # injective targets: plain placement
    return out


def integrate_t_iv(
    coeffs_iv: torch.Tensor, tables: MonomialTables, max_deg_in: int
) -> torch.Tensor:
    """\\int_0^t for INTERVAL coefficients (validated pass).

    coeffs_iv [..., M, 2] -> [..., M', 2]: same monomial mapping as
    `integrate_t`; the 1/(t_deg+1) scale uses its rigorous interval enclosure
    (tables.int_scale_iv), and the injective placement needs no reduction
    bound (Flow* Polynomial<Interval>::integral_time divides coefficients in
    directed rounding likewise).
    """
    m_in = tables.prefix_len_cpu[max_deg_in + 1]
    if coeffs_iv.shape[-2] < m_in:
        m_in = coeffs_iv.shape[-2]
    src = iv.mul(coeffs_iv[..., :m_in, :], tables.int_scale_iv[:m_in])  # [..., m_in, 2]
    tgt = tables.int_map[:m_in]  # [m_in]
    if m_in > tables.prefix_len_cpu[2 * tables.k]:  # same host-int test as integrate_t
        raise ValueError("integrate_t_iv: input polynomial exceeds degree 2k-1")
    m_out = tables.prefix_len_cpu[max_deg_in + 2]
    out = torch.zeros(
        *coeffs_iv.shape[:-2], m_out, 2, dtype=coeffs_iv.dtype, device=coeffs_iv.device
    )
    # index_add_ over the monomial dim (dim -2): targets are injective.
    out.index_add_(-2, tgt, src)
    return out


def evaluate_time_end(
    coeffs: torch.Tensor, tables: MonomialTables, step: StepTables
) -> torch.Tensor:
    """Substitute t := delta (the step END) into a point polynomial (tier P).

    coeffs [..., M] (M <= T2) -> spatial coefficients [..., Ts]: each monomial
    t^a0 * s maps to spatial monomial s with weight delta^a0 (RN point powers,
    matching Flow*'s evaluate_time with step_end_exp_table on Real).

    Different t-powers of the same spatial monomial COLLIDE here (t^2*x and x
    both land on x) — deterministic index_add_ accumulates them.
    """
    m_basis = coeffs.shape[-1]
    src = coeffs * step.end_gather(tables.t_deg[:m_basis], coeffs.dim())  # [..., M] RN
    tgt = tables.eval_t_spatial[:m_basis]  # [M]
    # Valid iff every t-stripped image lies in the spatial basis — a build-time
    # property (eval_t_valid_len), tested with host ints (no device sync).
    if m_basis > tables.eval_t_valid_len:
        raise ValueError("evaluate_time_end: spatial degree exceeds k")
    out = torch.zeros(*coeffs.shape[:-1], tables.Ts, dtype=coeffs.dtype, device=coeffs.device)
    out.index_add_(-1, tgt, src)
    return out


def evaluate_time_end_with_roundoff(
    coeffs: torch.Tensor, tables: MonomialTables, step: StepTables
) -> tuple[torch.Tensor, torch.Tensor]:
    """Endpoint substitution plus strict range of coefficient roundoff.

    The first result is exactly :func:`evaluate_time_end`.  The second ranges
    the difference between that retained point polynomial and a rigorous
    interval-coefficient evaluation of the same binary64 coefficients and
    exact powers of the binary64 step size, including power-table roundoff.
    """
    point = evaluate_time_end(coeffs, tables, step)
    m_basis = coeffs.shape[-1]
    factors = step.end_iv_gather(tables.t_deg[:m_basis], coeffs.dim())
    src_iv = iv.mul(iv.from_point(coeffs), factors)
    tgt = tables.eval_t_spatial[:m_basis]
    if m_basis > tables.eval_t_valid_len:
        raise ValueError("evaluate_time_end_with_roundoff: spatial degree exceeds k")
    shape = (*coeffs.shape[:-1], tables.Ts)
    lo_hat = torch.zeros(shape, dtype=coeffs.dtype, device=coeffs.device)
    hi_hat = torch.zeros_like(lo_hat)
    mag_hat = torch.zeros_like(lo_hat)
    lo_hat.index_add_(-1, tgt, src_iv[..., 0])
    hi_hat.index_add_(-1, tgt, src_iv[..., 1])
    mag_hat.index_add_(-1, tgt, iv.mag(src_iv))
    counts = torch.bincount(tgt, minlength=tables.Ts)
    error = dot_error_bound(mag_hat, counts)
    exact_coeff = torch.stack((next_down(lo_hat - error), next_up(hi_hat + error)), dim=-1)
    coefficient_error = iv.sub(exact_coeff, iv.from_point(point))
    return point, range_normal_iv_spatial(coefficient_error, tables)
