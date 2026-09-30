"""Closed-loop neural-network-controlled system (NNCS) verification (M7).

Sampled-data control loop, the POLAR/Verisig/Flow*-Feedback shape: every
`control_period` seconds the controller reads the state x and holds u constant
while the plant x' = f(x, u) flows. Verification alternates

    1. bound u = NN(x) over the current reachable state set, batched over all
       B branch-and-bound cells at once — via auto_LiRPA (user directive: the
       INTERNAL development tree, never a re-implemented CROWN);
    2. flow the EXTENDED system (x, u) with u' = 0 for one period using the
       flowstar_gpu reach machinery (all soundness guarantees apply);
    3. extract a sound end-of-period state box per cell and repeat.

v1 abstraction (documented conservatism): the controller bounds enter as
per-cell INTERVALS (CROWN-concretized), and the state hands over between
periods as a BOX — the classic interval-NNCS loop. The tighter degree-1-TM
controller coupling (CROWN lA/uA matrices as Taylor models over the state
symbols) is the planned v2; the auto_LiRPA plumbing for it (return_A) is
already verified.

Float-soundness note: auto_LiRPA computes in float64 round-to-nearest without
directed rounding (its own soundness model). We hedge its concretized bounds
outward before use, in two layers (see `crown_control_bounds`):
`CROWN_ULP_SLACK` ulp steps for auto_LiRPA's own concretization roundoff,
plus an absolute+relative pad `CROWN_ABS_SLACK * (1 + |u|)` because the
"exact" f64 forward pass the bounds must cover is not a single value: BLAS
reassociation makes NN(x) batch-size-DEPENDENT (the adversarial suite
measured the same input's output moving ~64 ulps between a batch-of-1 and a
batch-of-10 forward on a 2-16-16-1 net), so no fixed ulp count anchored to
one evaluation can cover every evaluation order. The pad gives ~3 orders of
headroom over that spread at unit scale and stays negligible against CROWN
box widths (~1e-3..1e-1). Very deep/wide controllers may need a larger pad —
size it against a measured batch-spread before trusting new architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import interval as iv
from . import polynomial as poly
from .config import Settings
from .flowpipe import ACTIVE, DONE, FlowpipeBatch, ReachResult, reach
from .monomials import build_tables
from .rounding import dot_error_bound, next_down, next_up

# Outward hedges on CROWN-concretized controller bounds (see module docstring).
# WHY two layers (adversarial-suite findings, PROGRESS 2026-07-27): a 9-ulp
# shortfall at the original slack of 4 prompted the bump to 16, but the deeper
# finding is that ulp counting alone cannot be sufficient — the f64 forward
# pass is batch-size-dependent (measured spread ~64 ulps ~ 8.7e-19 on the
# 2-16-16-1 test net), so CROWN_ABS_SLACK pads by an evaluation-order-
# independent absolute+relative epsilon on top.
CROWN_ULP_SLACK = 16
CROWN_ABS_SLACK = 1e-15


@dataclass
class NNCSResult:
    """Per-period trace of a closed-loop verification run.

    state_boxes: list over periods of [B, n, 2] cpu tensors — the sound
        end-of-period state enclosures (index 0 = the initial boxes).
    control_boxes: list over periods of [B, m, 2] — the held u bounds used
        during each period.
    status [B]: final lane status (flowpipe codes; DONE = completed all periods).
    safety [B] or None: sticky safety verdicts when a safe set was given.
    reaches: the per-period ReachResult objects (records only if requested).
    """

    state_boxes: list[torch.Tensor]
    control_boxes: list[torch.Tensor] = field(default_factory=list)
    status: torch.Tensor | None = None
    safety: torch.Tensor | None = None
    reaches: list[ReachResult] = field(default_factory=list)


def crown_control_bounds(
    controller: torch.nn.Module,
    state_boxes: torch.Tensor,
    method: str = "CROWN",
) -> torch.Tensor:
    """Bound u = NN(x) over per-cell state boxes with auto_LiRPA.

    state_boxes [B, n, 2] -> u bounds [B, m, 2], outward-stepped by
    CROWN_ULP_SLACK ulps. The whole batch is bounded in ONE auto_LiRPA call
    (its native batching — the B&B shape this project exists for).
    """
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

    x_l = state_boxes[..., 0]  # [B, n]
    x_u = state_boxes[..., 1]  # [B, n]
    x_mid = (x_l + x_u) * 0.5

    bm = BoundedModule(controller, x_mid[:1], device=state_boxes.device)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=x_l, x_U=x_u)
    bx = BoundedTensor(x_mid, ptb)
    lb, ub = bm.compute_bounds(x=(bx,), method=method)  # [B, m] each
    # Drop auto_LiRPA's autograd graph: callers may run under an enabled grad
    # context (the adversarial PGD attacks do), and un-detached bounds would
    # leak grad-tracking into every downstream flowpipe tensor.
    lb, ub = lb.detach(), ub.detach()

    for _ in range(CROWN_ULP_SLACK):
        lb = next_down(lb)
        ub = next_up(ub)
    # Evaluation-order pad: covers the batch-size-dependent f64 forward spread
    # (module docstring); the (1 + |u|) form keeps it meaningful near u = 0.
    pad = CROWN_ABS_SLACK * (1.0 + torch.maximum(lb.abs(), ub.abs()))  # [B, m]
    return torch.stack((lb - pad, ub + pad), dim=-1)  # [B, m, 2]


def end_of_period_box(
    fp: FlowpipeBatch,
    tables,
    step: poly.StepTables,
    sched,
    settings: Settings,
) -> torch.Tensor:
    """Sound state box at the END of the last completed step: [B, n_ext, 2].

    Evaluates tmvPre at t = delta and composes with the history map tmv (via
    the spatial monomial images — the same machinery as safety refinement), so
    the box is exact-composition-tight rather than a Lipschitz hand-wave.
    """
    from .safety import _spatial_images

    pre_rem = fp.pre_rem
    if settings.mode == "strict":
        x_end, endpoint_error = poly.evaluate_time_end_with_roundoff(fp.pre_coeffs, tables, step)
        pre_rem = iv.add(pre_rem, endpoint_error)
    else:
        x_end = poly.evaluate_time_end(fp.pre_coeffs, tables, step)  # [B, n, Ts]
    m_coeffs, m_rem = _spatial_images(
        fp.tmv_coeffs, fp.tmv_rem, tables, step, sched, settings
    )  # [B, Ts, Ts], [B, Ts, 2]

    # Composed end-state polynomial over s: coeffs' = x_end @ M  (tier P), with
    # the image remainders entering through the point weights.
    comp = torch.einsum("bnm,bms->bns", x_end, m_coeffs)  # [B, n, Ts]
    if settings.mode == "strict":
        abs_dot = torch.einsum("bnm,bms->bns", x_end.abs(), m_coeffs.abs())
        error = dot_error_bound(abs_dot, x_end.shape[-1])
        composition_error = poly.range_normal_iv_spatial(torch.stack((-error, error), dim=-1), tables)
        pre_rem = iv.add(pre_rem, composition_error)
    w_rem = iv.dot_point_iv(x_end, m_rem.unsqueeze(1), dim=-1)  # [B, n, 2]
    rng = poly.range_normal_spatial(comp, tables)  # [B, n, 2] over s in [-1,1]^n
    total = iv.add(rng, iv.add(w_rem, pre_rem))  # + image rems + pre/roundoff rem
    return total


def verify_nncs(
    plant_rhs: list[str],
    state_vars: list[str],
    control_vars: list[str],
    controller: torch.nn.Module,
    init_boxes: torch.Tensor,
    control_period: float,
    n_periods: int,
    settings: Settings,
    safe_set=None,
    record_tms: bool = False,
    crown_method: str = "CROWN",
) -> NNCSResult:
    """Verify a sampled-data NN-controlled system for n_periods periods.

    plant_rhs: f(x, u) component strings over state_vars + control_vars, ONE
        PER STATE VARIABLE (the u' = 0 rows are appended internally).
    init_boxes [B, n, 2]: initial state cells (the B&B batch).
    settings: fixed-step settings for the intra-period flow; control_period
        must be an integer multiple of settings.step (asserted) so period ends
        land exactly on step boundaries.

    Returns NNCSResult; lanes failing anywhere (contraction, division domain,
    UNSAFE) freeze from that period on, mirroring reach's lane semantics.
    """
    n = len(state_vars)
    m = len(control_vars)
    if len(plant_rhs) != n:
        raise ValueError(f"{len(plant_rhs)} plant components for {n} state vars")
    steps_per_period = control_period / settings.step
    if abs(steps_per_period - round(steps_per_period)) > 1e-9:
        raise ValueError("control_period must be an integer multiple of settings.step")
    if settings.step_min > 0:
        raise NotImplementedError("NNCS uses fixed-step flows (v1)")

    ext_vars = list(state_vars) + list(control_vars)
    ext_rhs = list(plant_rhs) + ["0"] * m

    boxes = init_boxes.to(dtype=torch.float64)
    B = boxes.shape[0]
    result = NNCSResult(state_boxes=[boxes.cpu().clone()])
    alive = torch.ones(B, dtype=torch.bool)
    safety_acc = torch.zeros(B, dtype=torch.int8) if safe_set is not None else None

    for _ in range(n_periods):
        if not bool(alive.any()):
            break

        # 1. controller bounds over the current cells (batched CROWN).
        u_boxes = crown_control_bounds(controller, boxes, method=crown_method)  # [B, m, 2]
        result.control_boxes.append(u_boxes.cpu().clone())

        # 2. flow the extended system one period.
        ext_boxes = torch.cat((boxes, u_boxes), dim=1)  # [B, n+m, 2]
        r = reach(
            ext_rhs, ext_vars, ext_boxes, control_period, settings,
            record_tms=record_tms, safe_set=safe_set,
        )
        result.reaches.append(r)

        # Lane bookkeeping: a lane survives the period iff it completed every
        # step (DONE at the horizon) — failures freeze it from here on.
        period_ok = (r.status == DONE) & (
            r.steps_completed == round(steps_per_period)
        )
        if safety_acc is not None:
            from . import safety as sf

            safety_acc = torch.where(
                (safety_acc != sf.UNSAFE) & (r.safety != sf.SAFE),
                r.safety,
                safety_acc,
            )
            period_ok = period_ok & (r.safety != sf.UNSAFE)
        alive = alive & period_ok

        # 3. end-of-period state box (drop the control dims).
        end_ext = _final_box_from_reach(r, ext_rhs, ext_vars, settings)  # [B, n+m, 2]
        new_boxes = end_ext[:, :n]  # [B, n, 2]
        boxes = torch.where(alive.view(B, 1, 1), new_boxes.to(boxes.device), boxes)
        result.state_boxes.append(boxes.cpu().clone())

    result.status = torch.where(
        alive, torch.full((B,), DONE, dtype=torch.int8), torch.full((B,), ACTIVE, dtype=torch.int8)
    )
    # Replace the placeholder ACTIVE for dead lanes with their real failure
    # codes. Sound because dead lanes are bitwise-frozen: their boxes stop
    # updating, so every later period re-runs the deterministic pipeline on
    # identical inputs and fails them with the SAME status code — hence
    # reaches[-1].status faithfully reproduces each dead lane's first failure.
    if result.reaches:
        last_status = result.reaches[-1].status
        result.status = torch.where(alive, result.status, last_status)
    result.safety = safety_acc
    return result


def _final_box_from_reach(
    r: ReachResult, ext_rhs: list[str], ext_vars: list[str], settings: Settings
) -> torch.Tensor:
    """End-of-period box from a reach run's final internal state.

    reach() sets ReachResult.final_fp (the final on-device FlowpipeBatch) on
    exit of BOTH the fixed-step and adaptive paths, so the end-of-period box
    composes directly from it — no record_tms needed on long runs.
    """
    fp = r.final_fp
    n_ext = fp.n
    tables = build_tables(n_ext, settings.order).to(fp.pre_coeffs.device.type)
    step = poly.build_step_tables(tables, settings.step)
    from .composition import build_schedule

    sched = build_schedule(n_ext, settings.order, settings.device)
    return end_of_period_box(fp, tables, step, sched, settings)
