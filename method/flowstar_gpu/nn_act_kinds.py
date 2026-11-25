"""S2-2 — activation-kind triples for the TM dispatch, nonsmooth included.

The Sprint-1 dispatch needed derivatives only for the ERROR bound's centered
mean-value form — never for the fitted polynomial. This module generalizes
the interface to a per-kind triple

    f_pt(x)          RN pointwise evaluation (fit sampling; no rigor needed)
    f_iv([lo,hi])    rigorous interval enclosure of the image
    df_iv([lo,hi])   rigorous enclosure of the derivative — for NONSMOOTH
                     kinds, of the CLARKE SUBDIFFERENTIAL over the interval;
                     Lebourg's mean-value theorem for Lipschitz functions
                     makes  E(s) ∈ E(c) + ∂E([piece])·(s − c)  sound with
                     subdifferential enclosures, which is all the centered
                     form ever used.

plus per-kind structure the dispatch exploits:

    exact_regimes([lo,hi]) -> (is_identity, is_zero) masks — ranges where
        the activation IS an affine map (ReLU above/below the kink): those
        rows bypass every tier with an EXACT result;
    kink_points -> sorted breakpoints for KINK-AWARE SUBDIVISION: the error
        enclosure splits its pieces exactly at the kinks so only the
        kink-containing piece carries the wide subdifferential ([0,1] for
        ReLU) — every other piece gets the tight one-sided derivative.

Smooth kinds (tanh/sigmoid) present the same interface (df = true
derivative through the codomain-bounded identities), so nn_tm's dispatch
and the S2-4 sidecar consume one protocol for everything.
"""

from __future__ import annotations

import torch

from . import interval as iv


class ActKind:
    """One activation's rigorous evaluation triple + structure."""

    name: str
    kink_points: tuple  # sorted breakpoints (empty for smooth kinds)
    has_series: bool  # dedicated Taylor series available (smooth only)

    def f_pt(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def f_iv(self, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def df_iv(self, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def exact_regimes(self, a: torch.Tensor):
        """(identity_mask, zero_mask) over [..., 2] ranges; default: none."""
        z = torch.zeros(a.shape[:-1], dtype=torch.bool, device=a.device)
        return z, z


class TanhKind(ActKind):
    name = "tanh"
    kink_points = ()
    has_series = True

    def f_pt(self, x):
        return torch.tanh(x)

    def f_iv(self, a):
        from .transcendental import tanh_iv

        return tanh_iv(a)

    def df_iv(self, a):
        # tanh' = 1 − y², y = tanh(a) ⊆ [-1, 1] — finite and tight always.
        y = self.f_iv(a)
        one = torch.zeros_like(y)
        one[..., 0] = 1.0
        one[..., 1] = 1.0
        return iv.sub(one, iv.pow_int(y, 2))


class SigmoidKind(ActKind):
    name = "sigmoid"
    kink_points = ()
    has_series = True

    def f_pt(self, x):
        return torch.sigmoid(x)

    def f_iv(self, a):
        from .transcendental import sigmoid_iv

        return sigmoid_iv(a)

    def df_iv(self, a):
        # sigmoid' = y(1 − y), y ⊆ [0, 1].
        y = self.f_iv(a)
        one = torch.zeros_like(y)
        one[..., 0] = 1.0
        one[..., 1] = 1.0
        return iv.mul(y, iv.sub(one, y))


class ReLUKind(ActKind):
    name = "relu"
    kink_points = (0.0,)
    has_series = False

    def f_pt(self, x):
        return torch.relu(x)

    def f_iv(self, a):
        # EXACT: max is monotone; no rounding occurs.
        return torch.clamp(a, min=0.0)

    def df_iv(self, a):
        # Clarke subdifferential enclosure: [0,0] below, [1,1] above,
        # [0,1] on kink-crossing intervals. Piecewise-exact.
        lo = (a[..., 0] > 0).to(a.dtype)  # derivative lower bound
        hi = (a[..., 1] > 0).to(a.dtype)  # derivative upper bound
        return torch.stack((lo, hi), dim=-1)

    def exact_regimes(self, a):
        return a[..., 0] >= 0, a[..., 1] <= 0


class LeakyReLUKind(ActKind):
    has_series = False
    kink_points = (0.0,)

    def __init__(self, slope: float = 0.01):
        self.slope = float(slope)
        self.name = f"leakyrelu{slope:g}"

    def f_pt(self, x):
        return torch.nn.functional.leaky_relu(x, self.slope)

    def f_iv(self, a):
        # monotone (slope > 0): endpoint map; the negative branch's RN
        # multiply gets one outward step per endpoint.
        from .rounding import next_down, next_up

        lo = torch.where(a[..., 0] >= 0, a[..., 0],
                         next_down(a[..., 0] * self.slope))
        hi = torch.where(a[..., 1] >= 0, a[..., 1],
                         next_up(a[..., 1] * self.slope))
        return torch.stack((lo, hi), dim=-1)

    def df_iv(self, a):
        lo = torch.where(a[..., 0] > 0,
                         torch.ones_like(a[..., 0]),
                         torch.full_like(a[..., 0], self.slope))
        hi = torch.where(a[..., 1] > 0, torch.ones_like(a[..., 1]),
                         torch.full_like(a[..., 1], self.slope))
        return torch.stack((torch.minimum(lo, hi), torch.maximum(lo, hi)),
                           dim=-1)

    def exact_regimes(self, a):
        z = torch.zeros(a.shape[:-1], dtype=torch.bool, device=a.device)
        return a[..., 0] >= 0, z  # identity above; never exactly zero


KINDS: dict[str, ActKind] = {
    "tanh": TanhKind(),
    "sigmoid": SigmoidKind(),
    "relu": ReLUKind(),
}


def get_kind(name: str) -> ActKind:
    k = KINDS.get(name)
    if k is None:
        raise ValueError(f"unsupported activation for TM propagation: {name}")
    return k


def kink_edges(kind: ActKind, lo: torch.Tensor, hi: torch.Tensor,
               n_pieces: int) -> torch.Tensor:
    """Per-row subdivision edges [R, N+1] in [0, 1] fractional coords,
    KINK-AWARE: each kink strictly inside (lo, hi) replaces the nearest
    uniform edge, so exactly the piece containing the kink carries the wide
    subdifferential. Uniform edges when no kink applies."""
    R = lo.shape[0]
    edges = torch.linspace(0.0, 1.0, n_pieces + 1, dtype=lo.dtype,
                           device=lo.device).expand(R, -1).contiguous()
    width = (hi - lo).clamp(min=1e-300)
    for kp in kind.kink_points:
        frac = (torch.as_tensor(kp, dtype=lo.dtype, device=lo.device)
                - lo) / width  # [R]
        inside = (frac > 0) & (frac < 1)
        if bool(inside.any()):
            # replace the nearest INTERIOR edge with the kink fraction
            idx = (frac * n_pieces).round().long().clamp(1, n_pieces - 1)
            rows = torch.nonzero(inside).view(-1)
            edges[rows, idx[rows]] = frac[rows]
    return edges
