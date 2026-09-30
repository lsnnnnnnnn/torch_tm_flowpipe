"""S2-5 stage 1 — safe-set halfspaces in the flowpipe's domain (Clip-and-Verify
for the NNCS coupling).

Soundness contract: docs/CLIP_CONFINEMENT_MEMO.md. This module produces, per
cell and per safe constraint q(x) ≤ 0, a halfspace in r-space

    A_r · r + b_r ≤ 0     with     { r : q(x(r)) ≤ 0 } ⊆ { r : A_r·r + b_r ≤ 0 }

(the RELAXATION direction, memo R2 — every truly-safe point satisfies the
halfspace) via the interval mean-value (centered) form over the cell:

    q(x(r)) ∈ q(x̂) + G · (x(r) − x̂),   G ⊇ ∇q(hull),  x̂ = the affine constant,

with x(r) − x̂ = A_aff·r + tail from the state rows' affine split
(nn_coupling.affine_parts). Extracting the point-linear part L(r) = ĝᵀA_aff·r
(ĝ = mid G) leaves a residual interval; the halfspace offset is
mid(q(x̂)) + lower(residual), rounded DOWN — every rounding pushes the
halfspace OUTWARD (weaker conditioning, never unsound).

Gradients come from symbolic differentiation of the constraint AST
(expr_parser nodes) evaluated with the engine's rigorous interval ops
(safety.eval_expr_interval) over the certified hull. Constraints whose
gradient enclosure is unbounded (log/sqrt domain edges — including cells
whose hull CONTAINS a gradient singularity, e.g. Docking's ||v|| at v = 0)
or whose halfspace is inert over the unit box are DROPPED — dropping
constraints only weakens the conditioning, never the soundness.

Regime (measured in the R2 tests): the centered form's residual scales
QUADRATICALLY with the cell's spread while the linear part scales linearly,
so the halfspace cuts decisively on THIN near-boundary cells — the
fine-split regime where verification stalls — and is inert on fat ones
(the drop rule then removes it at zero cost).
"""

from __future__ import annotations

import torch

from . import interval as iv
from .expr_parser import Bin, Node, Num, NumIv, Un, Var
from .rounding import next_down
from .safety import eval_expr_interval

_ZERO = Num(0.0)
_ONE = Num(1.0)


def diff_ast(node: Node, j: int) -> Node:
    """∂node/∂x_j as an AST (no simplification beyond zero/one pruning)."""
    if isinstance(node, (Num, NumIv)):
        return _ZERO
    if isinstance(node, Var):
        return _ONE if node.index == j else _ZERO
    if isinstance(node, Un):
        da = diff_ast(node.a, j)
        if da is _ZERO:
            return _ZERO
        if node.op == "neg":
            return Un("neg", da)
        if node.op == "sin":
            return Bin("*", Un("cos", node.a), da)
        if node.op == "cos":
            return Un("neg", Bin("*", Un("sin", node.a), da))
        if node.op == "exp":
            return Bin("*", Un("exp", node.a), da)
        if node.op == "log":
            return Bin("/", da, node.a)
        if node.op == "sqrt":
            return Bin("/", da, Bin("*", Num(2.0), Un("sqrt", node.a)))
        raise AssertionError(f"unknown unary {node.op}")
    assert isinstance(node, Bin)
    if node.op in ("+", "-"):
        da, db = diff_ast(node.a, j), diff_ast(node.b, j)
        if db is _ZERO:
            return da
        if da is _ZERO:
            return db if node.op == "+" else Un("neg", db)
        return Bin(node.op, da, db)
    if node.op == "*":
        da, db = diff_ast(node.a, j), diff_ast(node.b, j)
        terms = []
        if da is not _ZERO:
            terms.append(Bin("*", da, node.b))
        if db is not _ZERO:
            terms.append(Bin("*", node.a, db))
        if not terms:
            return _ZERO
        return terms[0] if len(terms) == 1 else Bin("+", terms[0], terms[1])
    if node.op == "/":
        da, db = diff_ast(node.a, j), diff_ast(node.b, j)
        t1 = _ZERO if da is _ZERO else Bin("/", da, node.b)
        t2 = (_ZERO if db is _ZERO
              else Bin("/", Bin("*", node.a, db), Bin("^", node.b, Num(2.0))))
        if t2 is _ZERO:
            return t1
        if t1 is _ZERO:
            return Un("neg", t2)
        return Bin("-", t1, t2)
    if node.op == "^":
        k = int(node.b.value)
        da = diff_ast(node.a, j)
        if da is _ZERO or k == 0:
            return _ZERO
        base = node.a if k == 2 else Bin("^", node.a, Num(float(k - 1)))
        return Bin("*", Bin("*", Num(float(k)), base), da)
    raise AssertionError(f"unknown binary {node.op}")


def _max_var_index(node: Node) -> int:
    """Largest Var slot the expression reads (-1 for constant expressions)."""
    if isinstance(node, Var):
        return node.index
    if isinstance(node, (Num, NumIv)):
        return -1
    if isinstance(node, Un):
        return _max_var_index(node.a)
    return max(_max_var_index(node.a), _max_var_index(node.b))  # Bin


def safe_halfspaces(
    exprs: list,
    n_states: int,
    A_aff: torch.Tensor,
    b_aff: torch.Tensor,
    tail: torch.Tensor,
    hull: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-cell halfspaces A_r·r + b_r ≤ 0 relaxing {q ≤ 0} (memo R1/R2).

    exprs: constraint ASTs over the first n_states variables;
    A_aff [B, n_states, n_sym], b_aff [B, n_states], tail [B, n_states, 2]
    (nn_coupling.affine_parts of the state rows, certified over the FULL
    cell); hull [B, n_states, 2] (the certified state enclosure the
    gradients are evaluated over).

    -> (A_r [B, C, n_sym], b_r [B, C], keep [B, C] bool). keep=False rows
    are dropped constraints (unbounded gradient or inert halfspace) — their
    A/b are zeroed; consumers must mask them out.
    """
    B, nx, n_sym = A_aff.shape
    dev = A_aff.device
    A_rows, b_rows, keeps = [], [], []
    x_hat = iv.from_point(b_aff)  # [B, nx, 2] degenerate box at x̂
    for q in exprs:
        if _max_var_index(q) >= nx:
            # The constraint reads a state OUTSIDE the NN-input subspace —
            # the affine parts cover the first nx rows only, so no r-space
            # halfspace can represent it. DROP it (a dropped cut is always
            # sound; same contract as the gradient/inert drops below). Hit
            # by ArtificialPancreas, whose safe set constrains plasma
            # glucose while the controller reads other rows.
            A_rows.append(A_aff.new_zeros(B, n_sym))
            b_rows.append(A_aff.new_zeros(B))
            keeps.append(torch.zeros(B, dtype=torch.bool, device=dev))
            continue
        q_at_hat = eval_expr_interval(q, x_hat)  # [B, 2]
        grads = [diff_ast(q, j) for j in range(nx)]
        G = torch.stack([eval_expr_interval(g, hull) for g in grads],
                        dim=1)  # [B, nx, 2]
        g_hat = (G[..., 0] + G[..., 1]) * 0.5  # [B, nx] point choice
        # L's coefficients, exact-in-intent: interval product ĝᵀA per symbol
        # minus the RN point row gives the extraction-rounding interval.
        A_pt = torch.einsum("bi,bis->bs", g_hat, A_aff)  # [B, n_sym] RN
        gA_iv = iv.sum(iv.mul(iv.from_point(g_hat).unsqueeze(2),
                              iv.from_point(A_aff)), dim=1)  # [B, n_sym, 2]
        rho = iv.sub(gA_iv, iv.from_point(A_pt))  # [B, n_sym, 2] rounding
        # residual = (q(x̂) − mid) + (G − ĝ)·range(x − x̂) + ĝ·tail + ρ·[-1,1]
        mid_q = (q_at_hat[..., 0] + q_at_hat[..., 1]) * 0.5  # [B]
        r0 = iv.sub(q_at_hat, iv.from_point(mid_q))  # [B, 2]
        # range(x − x̂) per state: [−Σ|A_i|, +Σ|A_i|] ⊕ tail_i
        a_abs = A_aff.abs().sum(dim=-1)  # [B, nx]
        dx = iv.add(iv.make(-a_abs, a_abs), tail)  # [B, nx, 2]
        t1 = iv.sum(iv.mul(iv.sub(G, iv.from_point(g_hat)), dx), dim=-1)
        t2 = iv.dot_point_iv(g_hat, tail, dim=-1)  # [B, 2]
        rho_span = iv.sum(iv.mul(rho, iv.make(
            -torch.ones_like(A_pt), torch.ones_like(A_pt))), dim=-1)
        resid = iv.add(iv.add(r0, t1), iv.add(t2, rho_span))  # [B, 2]
        b_row = next_down(mid_q + resid[..., 0])  # DOWN: outward halfspace
        # drop rules: unbounded gradient/residual; inert over the unit box
        finite = torch.isfinite(G).all(dim=(1, 2)) & torch.isfinite(b_row) \
            & torch.isfinite(A_pt).all(dim=-1)
        inert = (A_pt.abs().sum(dim=-1) + b_row) <= 0  # max over box ≤ 0
        keep = finite & ~inert
        A_rows.append(torch.where(keep.unsqueeze(-1), A_pt,
                                  torch.zeros_like(A_pt)))
        b_rows.append(torch.where(keep, b_row, torch.zeros_like(b_row)))
        keeps.append(keep)
    return (torch.stack(A_rows, dim=1), torch.stack(b_rows, dim=1),
            torch.stack(keeps, dim=1))
