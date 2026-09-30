"""S2-3 — custom α-CROWN loop optimizing A-matrix objectives.

Sprint-1's A4 measurement: the stock α objective (maximize concretized
endpoints) is COUNTERPRODUCTIVE for the midpoint-split coupling, which pays
the envelope-pair spread Σ_i|uA−lA|_i·rad_i + (ubias−lbias). This module
drives the alphas against that consumed width directly.

Machinery facts this build rests on (verified in Verifier_Development,
branch combined-fixes — file:line in the Sprint-2 research report):
  * one grad-enabled `compute_bounds(method='backward', bound_lower=True,
    bound_upper=True)` produces BOTH planes in a single pass; alphas are
    SHARED tensors with the lower/upper split on dim 0, so one scalar loss
    coupling lA and uA reaches both slices in one backward;
  * the returned A_dict is unconditionally DETACHED, but the
    graph-connected planes survive on `model.roots()[i].lA/.uA`
    (internal (spec, batch, ...) layout) until the next set_input;
  * graph-connected biases are recovered as bound − root-contribution:
    lbias = lb − (lA·x̂ − Σ|lA|·rad),  ubias = ub − (uA·x̂ + Σ|uA|·rad) —
    every term autograd-connected. This also folds the OTHER perturbed
    root's certified contribution (the prelayer's tail input e) into the
    bias exactly as the consumed residual pays it;
  * soundness of every iterate: ReLU clamps its alphas in-path on every
    pass (relu.py:617-640) and s-shaped clamps inside bound_relax against
    the sound tangent tables — a mid-optimization alpha can therefore
    never produce an unsound relaxation; `clip_alpha()` after each step
    keeps the parameters in-range for the optimizer too.

The loop keeps the BEST alphas by objective value and returns the planes
from a final pass at those alphas, detached, in the (batch, spec, ...)
convention prelayer_bounds' consumers expect.
"""

from __future__ import annotations

import torch


def _root_planes(bm, root_name: str):
    """Graph-connected (lA, uA) of one input root, transposed to
    [batch, spec, n]. None if the root received no plane."""
    for r in bm.roots():
        if r.name == root_name:
            lA = getattr(r, "lA", None)
            uA = getattr(r, "uA", None)
            fix = lambda t: (None if t is None
                             else t.transpose(0, 1).reshape(
                                 t.shape[1], t.shape[0], -1))
            return fix(lA), fix(uA)
    raise KeyError(f"root {root_name!r} not found")


def consumed_width(lA, uA, lbias, ubias, rad, weights=None):
    """The width the midpoint-split injection pays, per (batch, spec):
    Σ_i |uA−lA|_i·rad_i + (ubias − lbias). rad [batch, n] (certified input
    radii; ones for the prelayer's unit box). Autograd-safe (plain RN)."""
    spread = ((uA - lA).abs() * rad.unsqueeze(1)).sum(dim=-1)
    w = spread + (ubias - lbias)
    if weights is not None:
        w = w * weights
    return w


def alpha_optimize(
    bm,
    x_tuple: tuple,
    r_index: int,
    rad: torch.Tensor,
    iterations: int = 20,
    lr: float = 0.5,
    lr_decay: float = 0.98,
    objective: str = "resid",
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimize alphas of `bm` against an A-matrix objective; return the
    best iterate's (lA, uA, lbias, ubias) w.r.t. input `r_index`, detached.

    x_tuple: the compute_bounds inputs (BoundedTensors + constants) —
    exactly what prelayer_bounds builds. rad [batch, n]: certified radii of
    the r input (ones for the unit box). objective: 'resid' (the consumed
    width — O1/O2 of the plan), or 'default' (the stock −Σlb+Σub control,
    for the ablation).
    """
    from collections import defaultdict

    root_name = bm.input_name[r_index]
    out_name = bm.output_name[0]
    needed = defaultdict(set)
    needed[out_name].add(root_name)

    bm.init_alpha(x_tuple)
    acts = bm.get_enabled_opt_act()
    for node in acts:
        node.opt_start()
    alphas = []
    for node in acts:
        alphas.extend(list(node.alpha.values()))
        for v in node.alpha.values():
            v.requires_grad_()
    opt = torch.optim.Adam(alphas, lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

    best = {"val": None, "alphas": None}

    def snapshot():
        return [
            {k: v.detach().clone() for k, v in node.alpha.items()}
            for node in acts
        ]

    def restore(snap):
        with torch.no_grad():
            for node, s in zip(acts, snap):
                for k in node.alpha:
                    node.alpha[k].copy_(s[k])

    def one_pass(need_grad: bool):
        ctx = torch.enable_grad() if need_grad else torch.no_grad()
        with ctx:
            lb, ub, A_dict = bm.compute_bounds(
                x=x_tuple, method="backward", bound_lower=True,
                bound_upper=True, return_A=True, needed_A_dict=needed)
            lA, uA = _root_planes(bm, root_name)
            # bias recovery (module docstring): x̂ = the r root's nominal
            # value; its box is x̂ ± rad.
            r_nominal = x_tuple[r_index].reshape(lA.shape[0], -1)
            lb2 = lb.reshape(lA.shape[0], -1)
            ub2 = ub.reshape(lA.shape[0], -1)
            l_contrib = (lA * r_nominal.unsqueeze(1)).sum(-1) - (
                lA.abs() * rad.unsqueeze(1)).sum(-1)
            u_contrib = (uA * r_nominal.unsqueeze(1)).sum(-1) + (
                uA.abs() * rad.unsqueeze(1)).sum(-1)
            lbias = lb2 - l_contrib
            ubias = ub2 - u_contrib
        return lb2, ub2, lA, uA, lbias, ubias

    for it in range(iterations):
        lb2, ub2, lA, uA, lbias, ubias = one_pass(need_grad=True)
        if objective == "resid":
            loss = consumed_width(lA, uA, lbias, ubias, rad, weights).sum()
        elif objective == "default":
            loss = (ub2 - lb2).sum()
        else:
            raise ValueError(f"unknown objective {objective!r}")
        val = float(loss.detach())
        if best["val"] is None or val < best["val"]:
            best["val"] = val
            best["alphas"] = snapshot()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for node in acts:
            node.clip_alpha()
        sched.step()

    if best["alphas"] is not None:
        restore(best["alphas"])
    _, _, lA, uA, lbias, ubias = one_pass(need_grad=False)
    for node in acts:
        node.opt_end()
    return (lA.detach(), uA.detach(), lbias.detach(), ubias.detach())
