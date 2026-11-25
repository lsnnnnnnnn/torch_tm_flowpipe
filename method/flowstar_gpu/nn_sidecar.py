"""S2-4 — zonotope-generator sidecar: symbolic remainders through the net.

Beyond POLAR's product chain: per neuron-row, the intra-net error state is a
point generator matrix G [R, g] over net-internal fresh symbols ε ∈ [-1,1]^g,
carried alongside the TM polynomial:

    row value ∈ poly(r) + G·ε + rem,   ε ∈ [-1,1]^g shared across rows.

* Affine layers act EXACTLY on G (G ← W·G; the RN matmul's roundoff is
  Rump-inflated into rem — the engine's dot policy), so later layers ROTATE
  AND CANCEL earlier errors instead of accumulating boxed magnitudes (the
  PZ lesson, and the reason Verisig-without-tricks explodes).
* Activations pass G through the tier's DERIVATIVE/SUBDIFFERENTIAL interval
  D (nn_act_kinds.df_iv — one protocol for smooth and nonsmooth): by the
  (Lebourg) mean-value theorem, σ(F + G·ε) ∈ σ(F) + D·(G·ε), realized as
  G ← d_mid·G with the spread (D − d_mid)·range(G·ε) widened into rem, and
  the activation tier's own error E contributed as a FRESH generator column
  (mid into the constant, radius as the new coefficient) — POLAR's chain is
  the special case where D-passing is skipped and errors go straight to
  intervals.
* Generator budget: `reduce_gens` folds the smallest-magnitude columns into
  rem (sound order reduction) to cap g.
* At injection (the approved v1 boundary): `concretize_gens` collapses G·ε
  into the row remainder — intra-net cancellation has already happened.

Soundness: every operation is an enclosure argument on ε-boxes — pinned by
brute-force sampled containment through multi-layer nets in the tests.
"""

from __future__ import annotations

import torch

from . import interval as iv
from .nn_act_kinds import ActKind
from .rounding import dot_error_bound, next_down, next_up


def gen_range(G: torch.Tensor) -> torch.Tensor:
    """[R, g] -> [R, 2]: the exact range of G·ε over ε ∈ [-1,1]^g,
    outward-rounded (RN row sum + a-priori dot bound)."""
    s = G.abs().sum(dim=-1)
    err = dot_error_bound(G.abs().sum(dim=-1), max(G.shape[-1], 1))
    hi = next_up(s + err)
    return torch.stack((-hi, hi), dim=-1)


def affine_gens(W: torch.Tensor, G: torch.Tensor,
                rem: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """G' = W·G with the RN matmul's roundoff Rump-inflated into rem.

    W [out, in], G [B, in, g], rem [B, out, 2] (the affine layer's already-
    weighted remainder) -> (G' [B, out, g], rem')."""
    Gp = torch.einsum("oi,big->bog", W, G)
    mag = torch.einsum("oi,big->bog", W.abs(), G.abs())
    err = dot_error_bound(mag.sum(dim=-1), max(W.shape[1], 1))  # [B, out]
    slack = torch.stack((-next_up(err), next_up(err)), dim=-1)
    return Gp, iv.add(rem, slack)


def act_gens(
    kind: ActKind, arg: torch.Tensor, G: torch.Tensor, rem: torch.Tensor,
    tier_err: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pass G through an activation via the derivative interval; mint a
    fresh generator for the tier error.

    arg [R, 2]: the FULL argument range (poly + G·ε + rem — what the tier's
    error was certified over); G [R, g]; rem [R, 2]; tier_err [R, 2] (the
    tier's certified error, to be carried symbolically).
    -> (G' [R, g+1], rem', const_shift [R] — add to the poly constant).
    """
    D = kind.df_iv(arg)  # [R, 2]
    d_mid = (D[..., 0] + D[..., 1]) * 0.5
    Gp = G * d_mid.unsqueeze(-1)
    # spread: (D − d_mid) · range(G·ε), outward
    spread = iv.mul(iv.sub(D, iv.from_point(d_mid)), gen_range(G))
    # RN scaling roundoff of Gp: 1-ulp per entry, bounded by the row mass
    mass = next_up(Gp.abs().sum(dim=-1))
    ulp = torch.finfo(G.dtype).eps
    rough = next_up(mass * ulp * max(G.shape[-1], 1))
    rem2 = iv.add(rem, iv.add(spread, torch.stack((-rough, rough), dim=-1)))
    # fresh generator column for the tier error: mid -> constant, rad -> coeff
    mid = (tier_err[..., 0] + tier_err[..., 1]) * 0.5
    rad = next_up(torch.maximum(tier_err[..., 1] - mid,
                                mid - tier_err[..., 0]))
    Gp = torch.cat((Gp, rad.unsqueeze(-1)), dim=-1)
    return Gp, rem2, mid


def reduce_gens(G: torch.Tensor, rem: torch.Tensor, g_max: int,
                shared_cols: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap the generator count by folding the smallest columns into rem.

    shared_cols: ε symbols are shared across rows, so column selection is by
    TOTAL column mass (a column dropped for one row must drop for all —
    otherwise the shared-symbol correlation would be broken unsoundly).
    Folding a column c adds ±|G[:, c]| to each row's remainder (sound: the
    exact range of that column's contribution)."""
    g = G.shape[-1]
    if g <= g_max:
        return G, rem
    mass = G.abs().sum(dim=tuple(range(G.dim() - 1)))  # [g]
    keep_idx = torch.topk(mass, g_max).indices.sort().values
    drop_mask = torch.ones(g, dtype=torch.bool, device=G.device)
    drop_mask[keep_idx] = False
    dropped = G[..., drop_mask].abs().sum(dim=-1)  # [R...] per-row mass
    hi = next_up(dropped)
    rem2 = iv.add(rem, torch.stack((-hi, hi), dim=-1))
    return G[..., keep_idx].contiguous(), rem2


def concretize_gens(G: torch.Tensor,
                    rem: torch.Tensor) -> torch.Tensor:
    """The v1 injection boundary: fold G·ε entirely into the remainder."""
    return iv.add(rem, gen_range(G))
