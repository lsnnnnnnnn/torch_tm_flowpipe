"""Batched symbolic remainders (Flow* Symbolic_Remainder, Continuous.cpp:2123).

The wrapping-effect killer (RTSS'16): instead of re-boxing the previous steps'
remainders through every new composition, keep a queue of the per-step LINEAR
maps Phi and remainder columns J, and propagate old remainders through the
exact composed linear maps:

    step m:  tmv_of_x0 = c0 + A*r + N(r)        (A linear, N nonlinear)
      * only N goes through TM composition (its remainder -> J_{m});
      * A is applied to tmv's POLYNOMIALS directly (tier-P matrix x poly);
      * the linear image of history remainders is  sum_q PhiProd_q * J_{q-1},
        where PhiProd_q accumulates A_m * A_{m-1} * ... (scaled by the running
        diagonal preconditioners — `scalars` folds last step's invS into A).

Flow* structural quirks mirrored exactly (see the source lines in flowpipe.py):
  * Phi_L[0] is dead storage — the update and accumulation loops start at 1;
  * scalars start at 1 and become 0 for point dimensions (S == 0);
  * the queue is reset by the REACH loop (not advance) once len(J) reaches
    max_size, after t += step (Continuous.h:891-894) — see `reset_if_full`,
    which is the ONE implementation every reach loop (dense, sparse, the NNCS
    driver) must call, and whose docstring carries the soundness argument.

Batched layout (queue length is step-synchronous across lanes, so one global
buffer serves the whole batch): geometrically growing buffers

    phi_buf [capacity, B, n, n] f64  point matrices (entries [0, qlen))
    j_buf   [capacity, B, n, 2] f64  interval columns (entries [0, qlen))
    scalars [B, n] f64, qlen int

Buffers instead of Python lists because the per-step queue update is then ONE
einsum over the live slice and the J accumulation reads views with no restack
— the list version cost O(qlen) kernel launches + two O(qlen) stacks per step,
which dominated long-queue runs (PROGRESS: laubloomis sr1000 on cpu, and every
CROWN-Reach benchmark: queue 1000 across control steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import interval as iv
from . import sr_kernels, sr_sum_kernels


@dataclass
class SymbolicRemainder:
    """Batched SR state; see module docstring for semantics and shapes."""

    scalars: torch.Tensor
    max_size: int
    phi_buf: torch.Tensor
    j_buf: torch.Tensor
    # Strict-only interval enclosure of the exact Phi matrices represented by
    # ``phi_buf``.  Allocated lazily so parity mode keeps its memory contract.
    phi_iv_buf: torch.Tensor | None = None
    # Strict enclosure of the exact reciprocal preconditioners represented by
    # ``scalars``.  None means the initial exact all-ones state.
    scalars_iv: torch.Tensor | None = None
    qlen: int = 0  # phi entries (pushed by propagate, BEFORE the branch)
    jlen: int = 0  # J entries (pushed by append_j, AFTER composition)

    @property
    def queue_len(self) -> int:
        """Number of COMPLETED SR steps == J.size() — the quantity Flow*
        branches on (Continuous.cpp:2184) and resets on (Continuous.h:891).
        phi leads J by one entry between propagate() and append_j(); using
        qlen here would misclassify the first SR step (caught by coverage:
        the first-step full-composition branch went dead)."""
        return self.jlen

    def reset(self) -> None:
        """Flow* Symbolic_Remainder::reset: clear queues, scalars back to 1.

        Buffer contents beyond qlen are never read, so zeroing is unnecessary.
        """
        self.qlen = 0
        self.jlen = 0
        self.scalars = torch.ones_like(self.scalars)
        if self.scalars_iv is not None:
            self.scalars_iv = iv.from_point(self.scalars)

    def reset_if_full(self) -> bool:
        """Flow*'s queue-capacity rule, verbatim. Returns True iff it reset.

        WHAT FLOW* DOES AT CAPACITY (established from the reference source, not
        guessed — B1). Every symbolic-remainder reach loop closes its step body
        with the same guarded reset:

            if(symbolic_remainder.J.size() >= symbolic_remainder.max_size)
                symbolic_remainder.reset(currentFlowpipe.tmvPre.tms.size());

        ODE<>::reach_symbolic_remainder                Continuous.h:891-894
        ...reach_symbolic_remainder_adaptive_stepsize   Continuous.h:966-969
        ...reach_symbolic_remainder_adaptive_order      Continuous.h:1062-1065
        reach_inv_symbolic_remainder (+ its two adaptive twins)
                                       Continuous.h:2141-2144, 2402, 2653
        DDE (discrete)                            Discrete.h:670-673, 1024-1027
        our own C++ oracle mirrors it at tools/oracle/oracle_dump.cpp:280-281.

        So Flow* NEVER errors at capacity — it drops the symbolic history and
        starts a fresh one. Specifically Symbolic_Remainder::reset
        (Continuous.cpp:35-42) does exactly three things:

            scalars.clear(); scalars.resize(dim, 1);   // back to all-ones
            J.clear();                                 // remainder columns gone
            Phi_L.clear();                             // linear maps gone

        WHERE (load-bearing, a wrong reset point is a soundness bug):
          * AFTER a step completed successfully (`res == 1`); a failed advance
            returns from the loop instead, leaving the queue untouched;
          * AFTER the new flowpipe was stored and safety-checked — i.e. at the
            end of the step body, so a reset is always observed *between* two
            advances and never between the Phi push and the J push of one
            advance. (The exact spot relative to `t += step` differs by variant
            — after it in the fixed-step and adaptive-order loops, before it in
            the adaptive-stepsize one, Continuous.h:966-971 — which is
            immaterial: the SR machinery never reads `t`. What matters is that
            no advance intervenes.)
          * the test is on `J.size()` (our `jlen`/`queue_len`), not
            `Phi_L.size()` — those differ by one inside an advance;
          * `>=`, not `>`: with max_size == 1000 the reset fires the instant the
            1000th J lands, so the queue never exceeds max_size and the 1001st
            step is a fresh first step. A run of EXACTLY 1000 SR steps
            therefore never resets (which is why the shipped CROWN-Reach suite
            — quad_official / quad_reduced sit at exactly 50 x 20 = 1000
            substeps — never exercised this path).

        WHY IT IS SOUND (nothing is discarded):
          The queue is a TIGHTNESS device, not a container of otherwise-lost
          information. At every step the SR path writes the FULL remainder into
          the flowpipe it emits — Continuous.cpp:2241
              `result.tmv.tms[i].remainder = J_ip1[i][0] + J_i[i][0]`
          (ours: `new_tmv_rem = iv.add(j_new, j_lin)`, flowpipe.py /
          sparse_exec.py), where J_ip1 is this step's own new remainder and
          J_i = sum_q PhiProd_q * J_{q-1} is the linear image of the whole
          history. The queue stores only J_ip1, because the history's
          contribution is REDERIVED each step from the Phi products. Hence the
          emitted flowpipe (tmv/tmvPre polynomials + remainders) is already a
          self-contained, valid over-approximation independent of the queue.
          Clearing the queue drops the factorization, not the enclosure.

          The step after a reset then sees `queue_len == 0` and takes the
          first-SR-step branch (Continuous.cpp:2247-2289; ours: the `else` in
          the (c-SR) block), which composes the FULL local map — linear part
          included — against the previous flowpipe's `tmv` *with its remainder*,
          and takes the resulting composed remainder as the new J[0]. The
          accumulated remainder is therefore FOLDED IN by a rigorous TM
          composition, not dropped. `scalars` back to 1 is consistent with
          that: the post-reset step's Phi_L_i becomes Phi_L[0], which is dead
          storage (all Flow* loops start at index 1), so the scaling it would
          have carried is never read.

          Independent confirmation that this is how Flow* understands a reset:
          `reach_inv_symbolic_remainder` ALSO resets — unconditionally, queue
          length irrelevant — whenever an invariant CONTRACTS the flowpipe's
          domain (Continuous.h:2036, 2094), because the symbolic history's
          anchor changed. Flow* treats reset as "re-anchor the history", which
          is only ever safe because the flowpipe carries the full remainder.
          (We have no invariant contraction in the SR path, so that site has no
          counterpart here.)

        THE COST IS TIGHTNESS, AND ONLY TIGHTNESS: across the reset boundary the
        accumulated remainder is re-boxed once through interval composition
        instead of riding an exact point-matrix product, so the wrapping effect
        gets one free step. MEASURED against an otherwise identical run with a
        queue too large to fill (B1; docs/ALGORITHM.md carries the table):

          vanderpol, step 0.02 order 5, queue 100, 500 steps (4 resets)
            first post-reset step  1.0001x wider tmvPre remainder
            step 499 (after 4)     1.072x
          tora_relu_tanh through the NNCS driver, ode step 0.01 order 6,
          queue 1000, 1250 substeps (1 reset)
            first post-reset substep  1.00001x
            250 substeps later        1.0021x
            final per-variable hull   1.0002x - 1.0011x

        In exchange the per-step queue einsum stops growing without bound, which
        is why Flow* does this at all.
        """
        if self.queue_len < self.max_size:
            return False
        self.reset()
        return True

    def reserve(self, needed: int) -> None:
        """Grow storage for live entries without changing the reset capacity.

        Only [0, qlen) and [0, jlen) are read. Unused future history does not
        need max_size matrices up front, especially for large batches.
        """
        if needed > self.max_size:
            raise RuntimeError("symbolic remainder capacity reached before reset")
        if needed <= self.phi_buf.shape[0]:
            return
        capacity = min(self.max_size, max(needed, 2 * self.phi_buf.shape[0]))
        def grow(buffer, live):
            expanded = buffer.new_zeros((capacity, *buffer.shape[1:]))
            expanded[:live].copy_(buffer[:live])
            return expanded
        # Grow the largest storage first. Releasing its old allocation lets
        # the smaller point/J buffers reuse it and lowers the live copy peak.
        if self.phi_iv_buf is not None:
            self.phi_iv_buf = grow(self.phi_iv_buf, self.qlen)
        self.phi_buf = grow(self.phi_buf, self.qlen)
        self.j_buf = grow(self.j_buf, self.jlen)

    def append_j(self, j_new: torch.Tensor) -> None:
        """Push this step's remainder column J (called after composition,
        mirroring Flow*'s symbolic_remainder.J.push_back at Continuous.cpp:2292).

        j_new [B, n, 2]. Pairs with the phi entry propagate() just pushed:
        after both, phi and J have equal length qlen.
        """
        self.j_buf[self.qlen - 1] = j_new
        self.jlen = self.qlen


def make_symbolic_remainder(
    batch: int, n: int, max_size: int, device: str
) -> SymbolicRemainder:
    """Fresh SR state (Flow* ctor: scalars = 1, empty queues)."""
    if max_size < 1:
        raise ValueError(f"max_size must be >= 1, got {max_size}")
    return SymbolicRemainder(
        scalars=torch.ones(batch, n, dtype=torch.float64, device=device),
        max_size=max_size,
        phi_buf=torch.zeros(min(16, max_size), batch, n, n, dtype=torch.float64, device=device),
        j_buf=torch.zeros(min(16, max_size), batch, n, 2, dtype=torch.float64, device=device),
    )


def preconditioning_scale(
    rng: torch.Tensor, *, strict: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose S and its point reciprocal without erasing strict tiny ranges.

    A positive subnormal radius is enlarged to DBL_MIN, whose reciprocal is
    finite. Only a genuinely zero radius uses the no-scaling point branch.
    Parity retains the historical approximate-zero/clamped reciprocal policy.
    """
    radius = iv.mag(rng)
    tiny = torch.finfo(radius.dtype).tiny
    if strict:
        point_dim = radius == 0
        radius = torch.where(point_dim, torch.zeros_like(radius), radius.clamp(min=tiny))
        denominator = torch.where(point_dim, torch.ones_like(radius), radius)
    else:
        point_dim = radius <= tiny
        radius = torch.where(point_dim, torch.zeros_like(radius), radius)
        denominator = radius.clamp(min=1e-300)
    scale = torch.where(point_dim, torch.ones_like(radius), 1.0 / denominator)
    return radius, point_dim, scale


def _matmul_iv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Interval matrix product ``a @ b`` with arbitrary leading Q axis.

    ``a`` is ``[B,i,k,2]`` and ``b`` is ``[Q,B,k,j,2]``.  The result is
    ``[Q,B,i,j,2]``; every multiply and reduction is outward enclosed.
    """
    products = iv.mul(a.unsqueeze(0).unsqueeze(-2), b.unsqueeze(2))
    return iv.sum(products, dim=3)


def propagate(
    sr: SymbolicRemainder,
    phi_i: torch.Tensor,
    *,
    strict: bool = False,
    phi_i_iv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Queue update + linear image of history remainders (Continuous.cpp:2161-2177).

    phi_i [B, n, n]: this step's linear map with last step's invS folded in
    (right-scaled columns). Mutates the queue exactly like Flow*:
        for q in 1..len-1: phi[q] = phi_i @ phi[q];  then push phi_i
    — realized as ONE batched einsum over the live slice. Returns
    J_i [B, n, 2] = sum_{q=1..len-1} phi[q] * j[q-1], the exact linear image of
    all previous remainders (phi[0] is dead storage by construction).

    The J accumulation is vectorized (Rump-bounded interval sum) rather than
    Flow*'s sequential adds — sound for any order (rounding.py), ulp-level
    parity impact only.
    """
    B, n, _ = phi_i.shape
    sr.reserve(sr.qlen + 1)

    if strict:
        if phi_i_iv is None:
            raise ValueError("strict symbolic propagation requires phi_i_iv")
        if sr.phi_iv_buf is None:
            sr.phi_iv_buf = torch.zeros(
                *sr.phi_buf.shape, 2,
                dtype=sr.phi_buf.dtype,
                device=sr.phi_buf.device,
            )

    if sr.qlen > 1:
        live = sr.phi_buf[1 : sr.qlen]  # [Q-1, B, n, n] view
        # Strict history uses interval matrices for J; retain the public point
        # history too, but bound einsum's output and possible RHS-layout copy.
        # Each Q slice is independent and keeps the same k contraction.
        point_bytes_per_q = B * n * n * phi_i.element_size()
        chunk_q = max(1, (64 * 1024 * 1024) // point_bytes_per_q)
        if strict and live.is_cuda and live.shape[0] > chunk_q:
            for begin in range(0, live.shape[0], chunk_q):
                block = live[begin : begin + chunk_q]
                block.copy_(torch.einsum("bij,qbjk->qbik", phi_i, block))
        else:
            live.copy_(torch.einsum("bij,qbjk->qbik", phi_i, live))
        if strict:
            assert sr.phi_iv_buf is not None and phi_i_iv is not None
            live_iv = sr.phi_iv_buf[1 : sr.qlen]
            # The fused interval kernel snapshots each old matrix before its
            # in-place update and avoids the Q*B*n**3 broadcast temporary.
            if sr_kernels.supported(phi_i_iv, live_iv) and sr_kernels.available():
                sr_kernels.left_multiply_(phi_i_iv, live_iv)
            else:
                live_iv.copy_(_matmul_iv(phi_i_iv, live_iv))
    if sr.qlen >= sr.max_size:  # pragma: no cover  # guarded by the reach-loop reset (Continuous.h:891); defensive only
        raise RuntimeError("symbolic-remainder queue overflow: caller missed reset()")
    sr.phi_buf[sr.qlen] = phi_i
    if strict:
        assert sr.phi_iv_buf is not None and phi_i_iv is not None
        sr.phi_iv_buf[sr.qlen] = phi_i_iv
    sr.qlen += 1

    if sr.qlen <= 1:
        return torch.zeros(B, n, 2, dtype=phi_i.dtype, device=phi_i.device)

    q = sr.qlen - 1
    js = sr.j_buf[:q].movedim(0, 1)  # [B, Q, n, 2] (j[q-1] pairing)
    if strict:
        assert sr.phi_iv_buf is not None
        history = sr.phi_iv_buf[1 : q + 1]
        columns = sr.j_buf[:q]
        if sr_sum_kernels.supported(history, columns) and sr_sum_kernels.available():
            return sr_sum_kernels.sum_history(history, columns)
        phis_iv = history.movedim(0, 1)  # [B,Q,n,n,2]
        terms_iv = iv.mul(phis_iv, js.unsqueeze(2))  # [B,Q,n,n,2]
        terms = iv.sum(terms_iv, dim=-1)  # [B,Q,n,2]
        return iv.sum(terms, dim=1)

    phis = sr.phi_buf[1 : q + 1].movedim(0, 1)  # [B, Q, n, n] view (no copy)
    # Point-matrix x interval-vector per queue entry: row i gets
    # sum_k phis[..., i, k] * js[..., k, :]; then interval-sum over the queue.
    terms = iv.dot_point_iv(phis, js.unsqueeze(2), dim=-1)  # [B, Q, n, 2]
    return iv.sum(terms, dim=1)  # [B, n, 2]
