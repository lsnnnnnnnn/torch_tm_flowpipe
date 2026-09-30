"""CROWN -> Taylor-model coupling algebra (poly-CROWN campaign, P2a).

The production coupling (gpu_driver.inject_controls, mirroring
CrownReach.cpp:157-167) consumes a SAME-SLOPE CROWN result: lA == uA, so the
controller enclosure collapses to one point-coefficient degree-1 form
T·x + [c - r, c + r]. That forfeits the tighter standard relaxations and all
of alpha-CROWN (the s-shaped same-slope path has no optimizable parameters)
— loss point L3 of docs/RESEARCH_POLY_NN.md.

This module consumes the STANDARD two-slope result instead. auto_LiRPA's
contract is

    for all x in X:   lA·x + lbias  <=  NN(x)  <=  uA·x + ubias        (*)

with lA != uA in general. Pick ANY point matrix A_mid and offset c_mid (we
use the midpoints — the choice affects tightness, never soundness) and write

    NN(x) = A_mid·x + c_mid + resid(x).

From (*), for every x in X:

    resid(x) >= (lA - A_mid)·x + (lbias - c_mid)
    resid(x) <= (uA - A_mid)·x + (ubias - c_mid)

so a rigorous interval evaluation of the two affine envelopes over X encloses
resid over the whole cell, and u := A_mid·TM(x) + c_mid + resid is a sound
degree-1 Taylor model of the controller — the "interval-coefficient degree-1
TM via midpoint split" of RESEARCH_POLY_NN §3/P2. With lA == uA the algebra
degenerates to the legacy same-slope injection (resid = [lbias - c_mid,
ubias - c_mid]) up to <= 1-ulp outward rounding — the differential test.

Float-soundness posture (same trust model as the driver, documented at
gpu_driver.crown_bounds): auto_LiRPA's (*) is certified in float64
round-to-nearest — we take its outputs as given, and everything DOWNSTREAM
of them here is outward-rounded interval arithmetic (iv.sub/mul/sum with
directed rounding), so this module adds no new unsoundness beyond the
upstream contract.

Shapes (B batch, m controller outputs, n NN inputs):
    lA, uA           [B, m, n]   lower/upper slope matrices
    lbias, ubias     [B, m]      lower/upper offsets
    x_range          [B, n, 2]   NN-input enclosure (THE box CROWN was
                                 certified over — hull_ranges output)
    -> A_mid [B, m, n], c_mid [B, m], resid [B, m, 2]
"""

from __future__ import annotations

import torch

from . import interval as iv
from . import polynomial as poly


def two_slope_tm(
    lA: torch.Tensor,
    uA: torch.Tensor,
    lbias: torch.Tensor,
    ubias: torch.Tensor,
    x_range: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Midpoint-split a two-slope CROWN result into (A_mid, c_mid, resid).

    resid [B, m, 2] is generally ASYMMETRIC (kept so — symmetrizing would
    only widen). The caller injects u = A_mid·TM(x) + c_mid with remainder
    resid + A_mid-weighted input-row remainders (the latter exactly as the
    legacy inject does with T = A_mid).
    """
    # Midpoints in RN: ANY point choice is sound, the residual is bounded
    # against whatever we return (module docstring).
    A_mid = (lA + uA) * 0.5  # [B, m, n]
    c_mid = (lbias + ubias) * 0.5  # [B, m]

    # Interval slope differences: RN subtraction outward-rounded, so the true
    # real-arithmetic difference is enclosed entrywise.
    A_mid_iv = iv.from_point(A_mid)  # [B, m, n, 2]
    d_up = iv.sub(iv.from_point(uA), A_mid_iv)  # [B, m, n, 2]
    d_lo = iv.sub(iv.from_point(lA), A_mid_iv)  # [B, m, n, 2]

    # Envelope evaluation over the cell: interval dot against x_range plus the
    # outward-rounded bias difference. upper envelope's sup bounds resid
    # above; lower envelope's inf bounds it below.
    x1 = x_range.unsqueeze(1)  # [B, 1, n, 2]
    up = iv.add(
        iv.sum(iv.mul(d_up, x1), dim=-1),  # [B, m, 2]
        iv.sub(iv.from_point(ubias), iv.from_point(c_mid)),
    )
    lo = iv.add(
        iv.sum(iv.mul(d_lo, x1), dim=-1),
        iv.sub(iv.from_point(lbias), iv.from_point(c_mid)),
    )
    resid = torch.stack((lo[..., 0], up[..., 1]), dim=-1)  # [B, m, 2]
    return A_mid, c_mid, resid


# ---------------------------------------------------------------------------
# P1a — CROWN over the TM domain (kills loss point L1).
#
# Instead of interval-hulling the state TMs to a box and bounding NN(x) over
# it, split each state row into its affine part and a bounded tail,
#
#     x_i = b_i + sum_j A[i, j] · r_j + e_i,      e_i ∈ tail_i,   r ∈ [-1,1]^n
#
# (b: constant slot, A: degree-1 slots, tail: rigorous range of the degree>=2
# terms plus the row's remainder), and bound the PREPENDED network
#
#     u = NN( (r · A) + b + e )
#
# w.r.t. r and e. The matmul against the per-cell constant A is EXACT under
# CROWN (BoundMatMul with an unperturbed operand — no relaxation, no
# intermediate bounds), so no cross-variable correlation dies: thin/rotated
# cells keep their geometry. The output envelopes lA_r·r + lbias <= u <=
# uA_r·r + ubias are functions of the flowpipe's own domain symbols, so the
# midpoint split (two_slope_tm over the [-1,1] box) yields a degree-1 TM in r
# injected with ZERO further wrapping — no input-remainder dot, the
# remainders already live in e. This subsumes DiffReach's Thm IV.3
# reparameterization (credit: arXiv 2605.25346) inside auto_LiRPA, with the
# residual fed from a high-order TM tail instead of a quasi-quadratic
# intervalization.
# ---------------------------------------------------------------------------

# Per-(n, k) cache of the global one-hot monomial ids: _onehot_ids[(n, k)]
# = list over symbols j (t excluded) of the FULL-basis id whose exponent
# vector is one-hot at symbol j. Value-keyed (never by tables identity).
_onehot_ids: dict = {}


def onehot_ids(tables) -> list[int]:
    """Full-basis ids of the degree-1 monomials r_j (t excluded), in symbol
    order — the slots the affine part A lives in. [n] python ints."""
    key = (tables.n, tables.k)
    r = _onehot_ids.get(key)
    if r is None:
        exps = tables.exponents[: tables.T].cpu()  # [T, n+1]; col 0 is t
        deg1 = exps.sum(dim=1) == 1
        r = []
        for j in range(tables.n):
            hit = torch.nonzero(deg1 & (exps[:, j + 1] == 1)).view(-1)
            r.append(int(hit[0]))
        _onehot_ids[key] = r
    return r


def affine_parts(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    tables,
    step,
    nn_in: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split dense spatial TM rows into (A, b, tail).

    coeffs [B, n, T] (control-time rows: t-slots zero after end_of_time),
    rem [B, n, 2] -> A [B, nn_in, n_sym], b [B, nn_in], tail [B, nn_in, 2]
    where tail rigorously encloses (row - affine part) over the domain: the
    outward range of the degree>=2 coefficients plus the row remainder.
    """
    slots = onehot_ids(tables)  # [n_sym]
    b = coeffs[:, :nn_in, 0].clone()  # [B, nn_in]
    A = coeffs[:, :nn_in][..., slots].clone()  # [B, nn_in, n_sym]
    tail_coeffs = coeffs[:, :nn_in].clone()
    tail_coeffs[..., 0] = 0.0
    tail_coeffs[..., slots] = 0.0
    tail_rng = poly.range_normal(tail_coeffs, tables, step)  # [B, nn_in, 2]
    return A, b, iv.add(tail_rng, rem[:, :nn_in])


def affine_parts_s(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    eng,
    sup,
    nn_in: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse twin of affine_parts on a FULL support.

    coeffs [B, n, S] on sup (ids[0] == 0 by the Support invariant), rem
    [B, n, 2]. A symbol whose one-hot id is absent from the support has a
    structurally ZERO affine coefficient (the sparse representation's
    absent-means-zero contract).
    """
    from .support import range_normal_s

    slots = onehot_ids(eng.tables)
    local = {g: i for i, g in enumerate(sup.ids)}
    b = coeffs[:, :nn_in, 0].clone()  # [B, nn_in] (slot 0 == constant)
    B = coeffs.shape[0]
    n_sym = eng.tables.n
    A = coeffs.new_zeros(B, nn_in, n_sym)
    tail_coeffs = coeffs[:, :nn_in].clone()
    tail_coeffs[..., 0] = 0.0
    for j, g in enumerate(slots):
        s = local.get(g)
        if s is not None:
            A[..., j] = coeffs[:, :nn_in, s]
            tail_coeffs[..., s] = 0.0
    tail_rng = range_normal_s(tail_coeffs, eng, sup)  # [B, nn_in, 2]
    return A, b, iv.add(tail_rng, rem[:, :nn_in])


class PrelayerNet(torch.nn.Module):
    """u = NN((r·A) + b + e), the exact affine prelayer over the TM domain.

    Forward args (ALL batched; A/b enter as plain unperturbed inputs so the
    per-cell affine map costs CROWN nothing):
        r [B, n_sym] — perturbed, box [-1, 1]^n_sym;
        A [B, n_sym, nn_in]; b [B, nn_in] — constants per cell;
        e [B, nn_in] — perturbed, box tail (the TM's high-order + remainder).
    in_shape: the wrapped net's expected input shape with a leading -1
    (config["input_shape"]), e.g. (-1, 4) or NCHW (-1, 1, 1, k).
    """

    def __init__(self, net: torch.nn.Module, in_shape: tuple):
        super().__init__()
        self.net = net
        self.in_shape = tuple(in_shape)

    def forward(self, r, A, b, e):
        x = torch.matmul(r.unsqueeze(1), A).squeeze(1) + b + e  # [B, nn_in]
        # Flatten to [B, m]: conv-form nets emit [B, 1, m]-shaped raw
        # outputs, and alpha-CROWN's best-A tracking assumes the canonical
        # [batch, spec] output layout (a 3-D output crashed _update_A_dict's
        # mask expansion).
        return self.net(x.reshape(self.in_shape)).reshape(r.shape[0], -1)


def build_prelayer(net: torch.nn.Module, n_sym: int, nn_in: int,
                   in_shape: tuple, device: str):
    """BoundedModule over PrelayerNet, built ONCE per benchmark (A/b/e are
    inputs, so per-control-step changes need no rebuild)."""
    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        from auto_LiRPA import BoundedModule

    z = torch.zeros
    dummy = (
        z(1, n_sym, dtype=torch.float64),
        z(1, n_sym, nn_in, dtype=torch.float64),
        z(1, nn_in, dtype=torch.float64),
        z(1, nn_in, dtype=torch.float64),
    )
    # conv_mode='matrix': Patches-mode backward A's crash against this
    # graph's non-perturbed constant roots (A/b) on conv-form nets
    # (patches.py matmul expects the root's shape to broadcast to the
    # PERTURBED input's; unicycle NCHW hit expand([1,1,1,4] -> [1, n_sym])).
    # Patches is a memory optimization for large convs; these controller
    # convs are tiny, and matrix mode is the general, exact path.
    return BoundedModule(
        PrelayerNet(net, in_shape), dummy, device=device,
        bound_opts={"conv_mode": "matrix"},
    )


def prelayer_bounds(
    bm,
    A: torch.Tensor,
    b: torch.Tensor,
    tail: torch.Tensor,
    scale: float = 1.0,
    offset: float = 0.0,
    method: str = "CROWN",
    alpha_iters: int = 0,
    alpha_objective: str = "default",
    r_constraints: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-slope CROWN of the prelayer model w.r.t. r ∈ [-1,1]^n_sym.

    A [B, nn_in, n_sym] and b [B, nn_in] in the affine_parts convention
    (x_i = b_i + sum_j A[i, j]·r_j; transposed here for the r·Aᵀ matmul),
    tail [B, nn_in, 2] (asymmetric — fed as e's box directly, no
    symmetrization). e's columns are concretized by auto_LiRPA into the
    biases (exactly the interval the tail spans), so the returned envelopes
    lA_r·r + lbias <= u <= uA_r·r + ubias hold for every r in the unit box
    and every tail realization. The affine output map
    u = (NN_raw - offset)·scale is applied sign-safely.

    -> lA_r/uA_r [B, m, n_sym], lbias/ubias [B, m].
    """
    from collections import defaultdict

    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    A = A.transpose(1, 2).contiguous()  # [B, n_sym, nn_in] for r·A
    Bsz, n_sym, _ = A.shape
    ones = A.new_ones(Bsz, n_sym)
    r0 = A.new_zeros(Bsz, n_sym)
    # S2-5: optional safe-set halfspaces (nn_clip, memo R2 direction) on the
    # r-input — concretization then runs over box ∩ {not-yet-violating}.
    ptb_r = (PerturbationLpNorm(x_L=-ones, x_U=ones,
                                constraints=r_constraints)
             if r_constraints is not None
             else PerturbationLpNorm(x_L=-ones, x_U=ones))
    ptb_e = PerturbationLpNorm(x_L=tail[..., 0].contiguous(),
                               x_U=tail[..., 1].contiguous())
    bt_r = BoundedTensor(r0, ptb_r)
    bt_e = BoundedTensor(tail[..., 0].contiguous(), ptb_e)
    if alpha_iters > 0 and alpha_objective != "stock":
        # S2-3: the custom alpha loop over A-matrix objectives replaces the
        # single CROWN pass (nn_alpha; 'resid' minimizes the width the
        # midpoint split consumes — the A4 finding made optimizable).
        from .nn_alpha import alpha_optimize

        lA_r, uA_r, lb_r, ub_r = alpha_optimize(
            bm, (bt_r, A, b, bt_e), 0, ones, iterations=alpha_iters,
            objective=alpha_objective)
        lA = (lA_r * scale).reshape(Bsz, -1, n_sym)
        uA = (uA_r * scale).reshape(Bsz, -1, n_sym)
        lb = ((lb_r - offset) * scale).reshape(Bsz, -1)
        ub = ((ub_r - offset) * scale).reshape(Bsz, -1)
    else:
        required_A = defaultdict(set)
        required_A[bm.output_name[0]].add(bm.input_name[0])
        _, _, A_dict = bm.compute_bounds(
            x=(bt_r, A, b, bt_e), method=method, return_A=True,
            needed_A_dict=required_A,
        )
        Ar = A_dict[bm.output_name[0]][bm.input_name[0]]
        lA = (Ar["lA"] * scale).reshape(Bsz, -1, n_sym).detach()
        uA = (Ar["uA"] * scale).reshape(Bsz, -1, n_sym).detach()
        lb = ((Ar["lbias"] - offset) * scale).reshape(Bsz, -1).detach()
        ub = ((Ar["ubias"] - offset) * scale).reshape(Bsz, -1).detach()
    if scale < 0:
        lA, uA, lb, ub = uA, lA, ub, lb
    return lA, uA, lb, ub


def tm_domain_split(
    lA_r: torch.Tensor,
    uA_r: torch.Tensor,
    lbias: torch.Tensor,
    ubias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Midpoint-split r-domain envelopes over the unit box.

    two_slope_tm specialization: the certification domain of prelayer_bounds
    IS [-1,1]^n_sym, so the residual evaluation box is constant.
    -> A_mid [B, m, n_sym], c_mid [B, m], resid [B, m, 2].
    """
    Bsz, _, n_sym = lA_r.shape
    unit = lA_r.new_ones(Bsz, n_sym)
    r_range = torch.stack((-unit, unit), dim=-1)  # [B, n_sym, 2]
    return two_slope_tm(lA_r, uA_r, lbias, ubias, r_range)


def inject_affine_domain(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    tables,
    A_mid: torch.Tensor,
    c_mid: torch.Tensor,
    resid: torch.Tensor,
    u_ids: list,
) -> None:
    """Write u rows as degree-1 TMs in r DIRECTLY (dense).

    coeffs [B, n, T] / rem [B, n, 2] mutated in place on rows u_ids:
    polynomial part c_mid + sum_j A_mid[:, :, j]·r_j, remainder resid. No
    input-remainder dot — the state remainders were folded into the tail
    before bounding (prelayer_bounds), so adding them again would double
    count.
    """
    slots = onehot_ids(tables)
    coeffs[:, u_ids] = 0.0
    coeffs[:, u_ids, 0] = c_mid
    for j, g in enumerate(slots):
        coeffs[:, u_ids, g] = A_mid[..., j]
    rem[:, u_ids] = resid
