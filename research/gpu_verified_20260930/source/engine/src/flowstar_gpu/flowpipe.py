"""Batched flowpipe state, the advance() step kernel, and the reach() driver.

This is the top of the M2 pipeline: a faithful, batched replication of Flow*'s
fixed-step `Flowpipe::advance` (Continuous.cpp:857-1043) and the `ODE::reach`
loop (Continuous.h:589-663), for polynomial ODEs without symbolic remainders.
Each numbered stage below cites its Flow* source lines; docs/ALGORITHM.md has
the full mapping table.

State (see ARCHITECTURE.md): a flowpipe batch is
    pre_coeffs [B, n, T]  tmvPre — flowmap over (t, r1..rn), full working basis
    pre_rem    [B, n, 2]
    tmv_coeffs [B, n, Ts] tmv — history map over s in [-1,1]^n, spatial basis
    tmv_rem    [B, n, 2]
    status     [B] i8     ACTIVE / FAILED_CONTRACTION / DONE / FAILED_DIV
with implicit domain [0, delta] x [-1, 1]^n per step.

Status semantics: a lane that fails the contraction check freezes (its tensors
stop being updated; arithmetic keeps running on the frozen values harmlessly —
plan D5); Flow* single-instance equivalent is aborting with UNCOMPLETED.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import interval as iv
from . import polynomial as poly
from .composition import CompositionSchedule, build_schedule, compose
from .config import Settings
from .monomials import MonomialTables, build_tables
from .ode_compiler import CompiledODE, compile_ode, exec_point, exec_replay, exec_valid
from .polynomial import StepTables, build_step_tables, build_step_tables_batched
from .rounding import dot_error_bound
from .symbolic_remainder import (
    SymbolicRemainder, make_symbolic_remainder, preconditioning_scale, propagate,
)

# Lane status codes (int8).
ACTIVE = 0
FAILED_CONTRACTION = 1
DONE = 2
# M5: a div/log/sqrt domain violation (e.g. denominator range contains 0) —
# Flow* would silently take rec()'s [-1e5, 1e5] fallback or exit(1); we freeze
# the lane loudly instead (GOTCHAS #2), with the same frozen-lane semantics as
# a contraction failure.
FAILED_DIV = 3
# All fail-closed arithmetic/domain events share the historical status value 3
# so existing consumers remain compatible.  The refinement ledger's
# ``bad_nonfinite_mask`` distinguishes a generic non-finite event from a
# division/transcendental domain failure.
FAILED_ARITHMETIC = FAILED_DIV

# Flow* refinement constants (include.h:37,45) — Settings can override.
THRESHOLD_HIGH = 1e-12  # reach loop epsilon-start (Continuous.h:601)
INITIAL_SIMP = 1e-4  # SR-path-only aggressive tmv cutoff (include.h, GOTCHAS #4)


def _lane_nonfinite(*tensors: torch.Tensor) -> torch.Tensor:
    """Return a batch mask for NaN/Inf in any supplied tensor."""
    if not tensors:
        raise ValueError("_lane_nonfinite needs at least one tensor")
    batch = tensors[0].shape[0]
    bad = torch.zeros(batch, dtype=torch.bool, device=tensors[0].device)
    for tensor in tensors:
        if tensor.shape[0] != batch:
            raise ValueError("non-finite lane check received mismatched batches")
        if tensor.numel() == 0:
            continue
        dims = tuple(range(1, tensor.dim()))
        finite = torch.isfinite(tensor)
        if dims:
            finite = finite.all(dim=dims)
        bad |= ~finite
    return bad


@dataclass
class FlowpipeBatch:
    """One batch of flowpipes (see module docstring for shapes)."""

    pre_coeffs: torch.Tensor
    pre_rem: torch.Tensor
    tmv_coeffs: torch.Tensor
    tmv_rem: torch.Tensor
    status: torch.Tensor

    @property
    def batch(self) -> int:
        return self.pre_coeffs.shape[0]

    @property
    def n(self) -> int:
        return self.pre_coeffs.shape[1]


@dataclass
class StepRecord:
    """Per-step outputs for parity/analysis (CPU tensors).

    delta_used: the stored domain[0] sup for this step — equals the step size
    except on the final step, where Flow* clamps it to the remaining time
    (Continuous.h:607-613). A plain float on the fixed-step path; the adaptive
    path stores a [B] cpu tensor of per-lane (last-step-clamped) advances.
    """

    pre_coeffs: torch.Tensor
    pre_rem: torch.Tensor
    tmv_coeffs: torch.Tensor
    tmv_rem: torch.Tensor
    delta_used: float | torch.Tensor
    active_mask: torch.Tensor  # [B] bool: lanes that produced this step


@dataclass
class ReachResult:
    """Result of a reach() run.

    steps_completed [B]: number of flowpipes each lane produced.
    status [B]: final lane status (ACTIVE lanes become DONE at the horizon).
    records: per-step TM dumps (only when record_tms=True — B=1 parity runs).
    """

    steps_completed: torch.Tensor
    status: torch.Tensor
    records: list[StepRecord] = field(default_factory=list)
    # Per-lane safety verdict when reach() was given a safe set (M6): values
    # from safety.py (UNSAFE=-1 / SAFE=0 / UNKNOWN=1). UNSAFE stops the lane
    # immediately (Flow* returns COMPLETED_UNSAFE, Continuous.h:622-625);
    # UNKNOWN is sticky (Continuous.h:626-629). None without a safe set.
    safety: torch.Tensor | None = None
    # The final on-device flowpipe state (M7: NNCS reads the end-of-period box
    # from it without forcing record_tms on long runs).
    final_fp: FlowpipeBatch | None = None


@dataclass
class RefinementCacheState:
    """Generation/ownership certificate for refinement replay proposals.

    ``cache`` and ``tails`` contain polynomial ranges/truncation tails and are
    immutable across refinement; the candidate remainder is supplied afresh
    to every replay.  This audit state proves which mixed candidate vector a
    proposal generation consumed.  Each replay advances ``generation`` and
    transfers ownership to the resulting sequential partial-commit vector.
    """

    cache_id: str
    tails_id: str
    generation: int
    owner_remainder: torch.Tensor
    source_remainder: torch.Tensor

    def require(
        self,
        expected_generation: int,
        expected_owner: torch.Tensor | None = None,
    ) -> None:
        if self.generation != expected_generation:
            raise RuntimeError(
                "stale refinement cache generation: "
                f"cache={self.cache_id} has {self.generation}, "
                f"expected {expected_generation}"
            )
        if expected_owner is not None and not torch.equal(
            self.owner_remainder, expected_owner
        ):
            raise RuntimeError(
                "stale refinement cache owner: proposal input does not match "
                f"generation {expected_generation} for cache={self.cache_id}"
            )

    def commit(self, remainder: torch.Tensor) -> None:
        self.generation += 1
        self.owner_remainder = remainder.clone()


def _refinement_cache_id(tensor: torch.Tensor, prefix: str) -> str:
    return f"{prefix}:{tensor.untyped_storage().data_ptr():x}:{tuple(tensor.shape)}"


def _trace_refinement(settings: Settings, record: dict[str, object]) -> None:
    callback = settings.refinement_callback
    if callback is not None:
        callback(record)


def initial_flowpipe(boxes: torch.Tensor, tables: MonomialTables) -> FlowpipeBatch:
    """Box initial sets -> canonical flowpipes (Flow* Flowpipe(box), :260).

    boxes [B, n, 2] -> tmvPre_i = mid_i + rad_i * r_i with r in [-1, 1]^n and
    tmv = identity. Midpoint/radius in RN like Flow*'s Real path (the tiny
    midpoint rounding is absorbed because rad is taken as sup(box - mid),
    outward-rounded here for soundness).
    """
    B, n, _ = boxes.shape
    dev, dt = boxes.device, boxes.dtype
    # Half-before-add avoids spurious overflow for finite same-sign endpoints;
    # the outward radius below absorbs its RN/subnormal error.
    mid = boxes[..., 0] * 0.5 + boxes[..., 1] * 0.5  # [B, n] RN midpoint
    # Sound radius: max distance from mid to either endpoint, rounded up 1 ulp.
    from .rounding import next_up

    rad = next_up(torch.maximum(boxes[..., 1] - mid, mid - boxes[..., 0]))  # [B, n]
    rad = torch.where(boxes[..., 1] == boxes[..., 0], torch.zeros_like(rad), rad)

    pre = torch.zeros(B, n, tables.T, dtype=dt, device=dev)
    pre[..., 0] = mid
    # Degree-1 slots of r_i: spatial var i is full-basis monomial with exponent
    # e_{i+1}; via schedule var_image -> spatial idx -> full idx.
    sched = build_schedule(n, tables.k)
    var_full = tables.spatial_index[sched.var_image.to(tables.spatial_index.device)]  # [n]
    for i in range(n):
        pre[:, i, var_full[i]] = rad[:, i]

    tmv = torch.zeros(B, n, tables.Ts, dtype=dt, device=dev)
    for i in range(n):
        tmv[:, i, sched.var_image[i]] = 1.0

    zero2 = torch.zeros(B, n, 2, dtype=dt, device=dev)
    return FlowpipeBatch(
        pre_coeffs=pre,
        pre_rem=zero2.clone(),
        tmv_coeffs=tmv,
        tmv_rem=zero2.clone(),
        status=torch.zeros(B, dtype=torch.int8, device=dev),
    )


def build_rem_est(
    settings: Settings,
    n: int,
    batch: int,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Materialise `settings.remainder_estimation` as the a-priori guess tensor.

    THE single place the scalar-vs-per-variable form is resolved, so the dense
    `reach()` loop, the adaptive loop, the sparse engine and the NNCS driver
    (integrations/crown_reach/gpu_driver.py) cannot drift apart.

    Mirrors CrownSettings.cpp:86-88 in the scalar case
    (`vector<Interval>(numVars, I)`) and Flow*'s native per-variable
    `std::vector<Interval> remainder_estimation` (settings.h:126) in the tuple
    case; Continuous.cpp:963 consumes it as `remainder_estimation[i]`, i.e. the
    i-th entry belongs to the i-th DECLARED variable.

    Args:
        settings: run settings; `.remainder_estimation` is a float radius or a
            per-variable tuple of radii of length n.
        n: ODE variable count (states + t + controls — every declared var).
        batch: lane count B; the returned tensor is expanded over it.
        device: target device; defaults to `settings.device`.

    Returns:
        [B, n, 2] float64 — entry [b, i] is the interval [-v_i, +v_i]. The same
        radii for every lane (Flow* has no per-instance estimate either).

    Raises:
        ValueError: tuple form whose length != n. Loud on purpose: a silently
            truncated or broadcast vector would change which variables get
            which slack, and the resulting run would be a different
            computation than the config asked for.
    """
    re = settings.remainder_estimation
    if isinstance(re, tuple):
        if len(re) != n:
            raise ValueError(
                f"remainder_estimation has {len(re)} entries but the ODE has "
                f"{n} declared variables; give one radius per variable in "
                f"declaration order, or a single float to broadcast"
            )
        re_vals = torch.tensor(re, dtype=torch.float64)  # [n]
    else:
        re_vals = torch.full((n,), float(re), dtype=torch.float64)  # [n]
    dev = settings.device if device is None else device
    rem = torch.stack((-re_vals, re_vals), dim=-1).to(dev)  # [n, 2]
    return rem.unsqueeze(0).expand(batch, n, 2).contiguous()  # [B, n, 2]


def _validate_and_refine(
    code: CompiledODE,
    x: torch.Tensor,
    new_x0: torch.Tensor,
    x_rem: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
    settings: Settings,
    strict: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stages (h)-(k) of advance(): validated Picard, difference bound,
    contraction check, refinement — the delta-DEPENDENT half of the step,
    shared between the fixed path and each adaptive retry (Flow* re-runs
    exactly this block per candidate step, Continuous.cpp:1380-1410).

    x [B, n, T] polynomial iterate; new_x0 [B, n, T]; x_rem [B, n, 2] guess.
    Returns (refined remainders [B, n, 2], ok [B], bad [B]).
    """
    B, n, T = x.shape
    order = settings.order

    # (h) Validated Picard (Continuous.cpp:966-969): interval coefficients,
    #     cached intermediate ranges, truncation at order-1 then integration.
    #     `bad` [B] collects div/log/sqrt domain violations (M5) — those lanes
    #     are frozen as FAILED_DIV below, exactly like contraction failures.
    bad = torch.zeros(B, dtype=torch.bool, device=x.device)
    f_iv, f_rem, cache, tails = exec_valid(
        code, x, x_rem, order - 1, tables, step, settings.cutoff, bad_out=bad
    )
    bad |= _lane_nonfinite(x, new_x0, x_rem, f_iv, f_rem, cache, tails)
    int_f_iv = poly.integrate_t_iv(f_iv, tables, max_deg_in=order - 1)  # [B, n, K, 2]
    tmv_tmp_coeffs = torch.zeros(B, n, T, 2, dtype=x.dtype, device=x.device)
    tmv_tmp_coeffs[..., : int_f_iv.shape[-2], :] = int_f_iv
    tmv_tmp_coeffs = iv.add(tmv_tmp_coeffs, iv.from_point(new_x0))  # + x0
    tmv_tmp_rem = iv.mul(f_rem, step.dt(f_rem.dim() - 1).expand_as(f_rem))  # rem *= [0, delta]

    # (i) Bound the interval-vs-point polynomial difference
    #     (Continuous.cpp:971-982) — the roundoff rigor mechanism.
    diff_iv = iv.sub(tmv_tmp_coeffs, iv.from_point(x))  # [B, n, T, 2]
    int_diff = poly.range_normal_iv(diff_iv, tables, step)  # [B, n, 2]

    # (j) Contraction check (Continuous.cpp:985-1006): Picard image inside the
    #     guess, per lane over all dims. Bad lanes (M5) can never pass.
    total = iv.add(tmv_tmp_rem, int_diff)  # [B, n, 2]
    bad |= _lane_nonfinite(tmv_tmp_coeffs, tmv_tmp_rem, int_diff, total)
    subset_dims = iv.contains(x_rem, total)  # [B, n]
    ok = subset_dims.all(dim=-1) & ~bad  # [B]

    cache_state = None
    if settings.refinement_callback is not None:
        cache_state = RefinementCacheState(
            cache_id=_refinement_cache_id(cache, "range-cache"),
            tails_id=_refinement_cache_id(tails, "strict-tails"),
            generation=0,
            owner_remainder=total.clone(),
            source_remainder=x_rem.clone(),
        )
        for lane in range(B):
            attempted_step = (
                step.delta
                if step.lanes == 0
                else float(step.deltas[lane].item())
            )
            _trace_refinement(
                settings,
                {
                    "event": "initial_self_map",
                    "lane": lane,
                    "initial_self_map_ok": bool(ok[lane].item()),
                    "input_remainder_vector": x_rem[lane].detach().cpu().tolist(),
                    "proposal_interval": total[lane].detach().cpu().tolist(),
                    "attempted_step": attempted_step,
                    "subset_by_component": subset_dims[lane].detach().cpu().tolist(),
                    "subset_margin_by_component": {
                        "lower": (
                            total[lane, :, 0] - x_rem[lane, :, 0]
                        ).detach().cpu().tolist(),
                        "upper": (
                            x_rem[lane, :, 1] - total[lane, :, 1]
                        ).detach().cpu().tolist(),
                    },
                    "bad_nonfinite_mask": bool(bad[lane].item()),
                    "input_generation": 0,
                    "proposal_generation": 0,
                    "cache_id": cache_state.cache_id,
                    "tails_id": cache_state.tails_id,
                },
            )

    # (k) Refinement — shared with the sparse path (identical [B, n, 2] math).
    cur, bad = refine_loop(code, total, ok, bad, cache, tails, strict, int_diff,
                           step, settings, cache_state=cache_state)
    return cur, ok, bad


def refine_loop(
    code: CompiledODE,
    accepted: torch.Tensor,
    ok: torch.Tensor,
    bad: torch.Tensor,
    cache: torch.Tensor,
    tails: torch.Tensor,
    strict: bool,
    int_diff: torch.Tensor,
    step: StepTables,
    settings: Settings,
    *,
    cache_state: RefinementCacheState | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refinement loop (Continuous.cpp:1008-1036): remainder-only replay until
    improvement stalls (STOP_RATIO) or MAX_REFINEMENT_STEPS. Representation-
    free ([B, n, 2] interval work over the exec_valid cache), so the dense and
    sparse advance paths share it verbatim. Returns (refined rems, bad)."""
    B, n, _ = accepted.shape
    cur = accepted  # [B, n, 2] current remainders (junk for failed lanes)
    refining = ok.clone()  # [B]
    if cache_state is None and settings.refinement_callback is not None:
        cache_state = RefinementCacheState(
            cache_id=_refinement_cache_id(cache, "range-cache"),
            tails_id=_refinement_cache_id(tails, "strict-tails"),
            generation=0,
            owner_remainder=accepted.clone(),
            source_remainder=accepted.clone(),
        )
    tracked = cache_state is not None
    trace_enabled = settings.refinement_callback is not None
    expected_generation = 0
    if tracked:
        assert cache_state is not None
        cache_state.require(expected_generation, cur)
    iterations = (
        torch.zeros(B, dtype=torch.int64, device=accepted.device)
        if trace_enabled
        else None
    )
    for iteration in range(settings.max_refinement_steps):
        if not bool(refining.any()):
            break
        if tracked:
            assert cache_state is not None
            cache_state.require(expected_generation, cur)
        input_cur = cur if trace_enabled else None
        input_refining = refining.clone() if trace_enabled else None
        bad_replay = torch.zeros_like(bad)  # [B]
        new_rem = exec_replay(code, cur, cache, tails, strict, bad_out=bad_replay)
        bad_replay |= _lane_nonfinite(new_rem)
        # A replay-time domain violation on a refining lane fails it like the
        # valid-pass one (division domains only shrink during refinement, so
        # this cannot fire for a lane exec_valid accepted — pure defense).
        bad = bad | (bad_replay & refining)
        refining = refining & ~bad_replay
        # Prevent already-failed lanes' Inf/NaN payloads from contaminating
        # subsequent vectorized arithmetic; their state is frozen below.
        new_rem = torch.where(
            bad_replay.view(B, 1, 1), torch.zeros_like(new_rem), new_rem
        )
        new_rem = iv.mul(new_rem, step.dt(new_rem.dim() - 1).expand_as(new_rem))
        new_rem = iv.add(new_rem, int_diff)
        sub_ok = iv.contains(cur, new_rem)  # [B, n] refined still inside current
        # Flow* processes dims sequentially and BREAKS at the first non-subset
        # dim, leaving earlier dims updated (GOTCHAS #5). Replicate: dims
        # before the first failure update, later ones keep their value.
        fail_any = ~sub_ok  # [B, n]
        first_fail = torch.where(
            fail_any.any(dim=-1),
            fail_any.float().argmax(dim=-1),
            torch.full((B,), n, device=sub_ok.device, dtype=torch.long),
        )  # [B] index of first failing dim, n if none
        dim_idx = torch.arange(n, device=sub_ok.device).unsqueeze(0)  # [1, n]
        update = (dim_idx < first_fail.unsqueeze(-1)) & refining.unsqueeze(-1)  # [B, n]
        # Improvement test per updated dim: width(new)/width(old) <= STOP_RATIO.
        ratio = iv.width(new_rem) / iv.width(cur).clamp(min=5e-324)  # [B, n]
        improved = (ratio <= settings.stop_ratio) & update  # [B, n]
        cur = torch.where(update.unsqueeze(-1), new_rem, cur)
        # Lane keeps refining iff no dim failed AND some dim still improves.
        refining = refining & (~fail_any.any(dim=-1)) & improved.any(dim=-1)
        if trace_enabled:
            assert iterations is not None and input_refining is not None
            iterations += input_refining.long()
        if tracked:
            assert cache_state is not None
            cache_state.commit(cur)
            expected_generation += 1
        if trace_enabled:
            assert cache_state is not None and input_cur is not None
            assert input_refining is not None
            for lane in range(B):
                if not bool(input_refining[lane].item()):
                    continue
                for component in range(n):
                    committed = bool(update[lane, component].item())
                    _trace_refinement(
                        settings,
                        {
                            "event": "refinement_component",
                            "lane": lane,
                            "iteration": iteration,
                            "component_order": component,
                            "initial_self_map_ok": bool(ok[lane].item()),
                            "input_remainder_vector": input_cur[lane].detach().cpu().tolist(),
                            "input_generation": expected_generation - 1,
                            "proposal_interval": new_rem[lane, component].detach().cpu().tolist(),
                            "proposal_generation": expected_generation,
                            "subset_result": bool(sub_ok[lane, component].item()),
                            "commit_result": committed,
                            "commit_mask": update[lane].detach().cpu().tolist(),
                            "vector_after_commit": cur[lane].detach().cpu().tolist(),
                            "stop_ratio_inputs": {
                                "new_width": float(iv.width(new_rem[lane, component]).item()),
                                "old_width": float(iv.width(input_cur[lane, component]).item()),
                                "ratio": float(ratio[lane, component].item()),
                                "threshold": settings.stop_ratio,
                            },
                            "stop_decision": bool(refining[lane].item()),
                            "bad_nonfinite_mask": bool(bad_replay[lane].item()),
                            "cache_id": cache_state.cache_id,
                            "cache_generation": cache_state.generation,
                            "tails_id": cache_state.tails_id,
                        },
                    )

    if trace_enabled:
        assert cache_state is not None and iterations is not None
        for lane in range(B):
            _trace_refinement(
                settings,
                {
                    "event": "final_remainder_owner",
                    "lane": lane,
                    "initial_self_map_ok": bool(ok[lane].item()),
                    "iterations": int(iterations[lane].item()),
                    "final_accepted_remainder": cur[lane].detach().cpu().tolist(),
                    "final_generation": cache_state.generation,
                    "cache_id": cache_state.cache_id,
                    "tails_id": cache_state.tails_id,
                    "bad_nonfinite_mask": bool(bad[lane].item()),
                },
            )

    return cur, bad


def advance(
    fp: FlowpipeBatch,
    code: CompiledODE,
    tables: MonomialTables,
    step: StepTables,
    sched: CompositionSchedule,
    settings: Settings,
    rem_est: torch.Tensor,
    sr: SymbolicRemainder | None = None,
    step_prev: StepTables | None = None,
) -> tuple[FlowpipeBatch, torch.Tensor]:
    """One integration step for every ACTIVE lane (Continuous.cpp:857-1043;
    with `sr` given, the symbolic-remainder variant Continuous.cpp:2123-2418).

    rem_est [B, n, 2]: the a-priori remainder guess (Flow* remainder_estimation).
    Returns (new flowpipe batch, ok_mask [B] bool). Lanes with ok_mask False
    failed the contraction check this step; the caller freezes them. Frozen /
    already-inactive lanes get their OLD state copied through. In SR mode the
    queue is mutated for ALL lanes (queue length is step-synchronous; frozen
    lanes' entries are junk-but-harmless — their flowpipe state never updates).
    """
    B, n, T = fp.pre_coeffs.shape
    strict = settings.mode == "strict"
    order = settings.order
    cutoff_eps = settings.cutoff

    # (a) Evaluate tmvPre at t = delta (Continuous.cpp:866): polynomial
    #     substitution t := delta; the remainder rides along unchanged.
    #     ADAPTIVE NOTE: this delta belongs to the PREVIOUS accepted step (the
    #     new flowpipe starts at its end) — Flow* runs stages (a)-(f) BEFORE
    #     setStepsize(new) for exactly this reason (Continuous.cpp:1233-1338).
    if strict:
        x0_sp_full, endpoint_roundoff = poly.evaluate_time_end_with_roundoff(
            fp.pre_coeffs, tables, step_prev or step
        )
        x0_rem = iv.add(fp.pre_rem, endpoint_roundoff)
    else:
        x0_sp_full = poly.evaluate_time_end(fp.pre_coeffs, tables, step_prev or step)  # [B, n, Ts]
        x0_rem = fp.pre_rem  # [B, n, 2]

    # (b) Split off the constant part c0 (Continuous.cpp:869-882; the
    #     remainder-midpoint absorption there is commented out — GOTCHAS #3).
    c0 = x0_sp_full[..., 0].clone()  # [B, n]
    x0_sp = x0_sp_full.clone()
    x0_sp[..., 0] = 0.0

    if sr is None:
        # (c) Compose with the history map (Continuous.cpp:884-886).
        new_tmv_coeffs, new_tmv_rem = compose(
            x0_sp,
            x0_rem,
            fp.tmv_coeffs,
            fp.tmv_rem,
            tables,
            step,
            sched,
            order,
            cutoff_eps,
            strict,
        )  # [B, n, Ts], [B, n, 2]
    else:
        # (c-SR) Symbolic-remainder composition (Continuous.cpp:2151-2292).
        var_img = sched.var_image  # [n] spatial idx of r_i (on device)
        # Linear part A[b, i, j] = coeff of r_j in component i; nonlinear rest.
        lin_a = x0_sp[..., var_img]  # [B, n, n]
        x0_other = x0_sp.clone()
        x0_other[..., var_img] = 0.0  # degree >= 2 terms only
        # (decompose puts the REMAINDER on the nonlinear part: TaylorModel.h:1091)
        phi_i = lin_a * sr.scalars.unsqueeze(1)  # [B, n, n] right-scaled columns
        phi_i_iv = None
        if strict:
            # The stored point Phi stays unchanged, while this interval matrix
            # encloses the exact product of its binary64 factors.  Subsequent
            # strict queue products propagate this enclosure rather than
            # treating rounded Phi entries as exact.
            phi_i_iv = iv.mul(
                iv.from_point(lin_a),
                (
                    sr.scalars_iv.unsqueeze(1)
                    if sr.scalars_iv is not None
                    else iv.from_point(sr.scalars.unsqueeze(1))
                ),
            )
        j_lin = propagate(
            sr, phi_i, strict=strict, phi_i_iv=phi_i_iv
        )  # [B, n, 2] linear image of history rems

        if sr.queue_len > 0:
            # Nonlinear-only composition; its remainder is this step's J entry.
            comp_c, j_new = compose(
                x0_other, x0_rem, fp.tmv_coeffs, fp.tmv_rem,
                tables, step, sched, order, cutoff_eps, strict,
            )
            # Linear part applied to tmv POLYNOMIALS only (tier-P matrix x poly;
            # the linear x remainder interaction lives in the J machinery).
            lin_part = torch.einsum("bij,bjt->bit", lin_a, fp.tmv_coeffs)  # [B, n, Ts]
            new_tmv_coeffs = comp_c + lin_part
            new_tmv_rem = iv.add(j_new, j_lin)
            if strict:
                # Charge both the linear GEMM reduction and the final tier-P
                # coefficient addition.  The error polynomial is ranged over
                # the actual spatial basis and inserted exactly once.
                abs_dot = torch.einsum(
                    "bij,bjt->bit", lin_a.abs(), fp.tmv_coeffs.abs()
                )
                lin_radius = dot_error_bound(abs_dot, n)
                lin_error = torch.stack((-lin_radius, lin_radius), dim=-1)
                exact_add = iv.add(iv.from_point(comp_c), iv.from_point(lin_part))
                add_error = iv.sub(exact_add, iv.from_point(new_tmv_coeffs))
                coeff_error = iv.add(lin_error, add_error)
                # This is a fresh raw-coordinate error, not old history.
                j_new = iv.add(j_new, poly.range_normal_iv_spatial(coeff_error, tables))
                new_tmv_rem = iv.add(j_new, j_lin)
        else:
            # First SR step: full composition, exactly like non-SR
            # (Continuous.cpp:2247-2290).
            new_tmv_coeffs, new_tmv_rem = compose(
                x0_sp, x0_rem, fp.tmv_coeffs, fp.tmv_rem,
                tables, step, sched, order, cutoff_eps, strict,
            )
            j_new = new_tmv_rem
        if not strict:
            sr.append_j(j_new)

    # (e) Diagonal preconditioning (Continuous.cpp:921-951): scale the new
    #     local set to the unit box; restart Picard from c0 + diag(S) * r.
    tmv_full = torch.zeros(B, n, T, dtype=x0_sp.dtype, device=x0_sp.device)
    tmv_full[..., tables.spatial_index] = new_tmv_coeffs
    rng = iv.add(poly.range_normal(tmv_full, tables, step), new_tmv_rem)  # [B, n, 2]
    S, point_dim, scale = preconditioning_scale(rng, strict=strict)
    scale_iv = None
    pre_scale_coeffs = new_tmv_coeffs
    if strict:
        safe_s = torch.where(point_dim, torch.ones_like(S), S)
        scale_iv, _scale_bad = iv.rec(iv.from_point(safe_s))
        scale_iv = torch.where(
            point_dim.unsqueeze(-1), iv.from_point(torch.ones_like(S)), scale_iv
        )
    new_tmv_coeffs = new_tmv_coeffs * scale.unsqueeze(-1)  # [B, n, Ts] tier-P
    if strict:
        assert scale_iv is not None
        exact_scaled_coeffs = iv.mul(
            iv.from_point(pre_scale_coeffs), scale_iv.unsqueeze(-2)
        )
        scale_coeff_error = iv.sub(
            exact_scaled_coeffs, iv.from_point(new_tmv_coeffs)
        )
        new_tmv_rem = iv.mul(new_tmv_rem, scale_iv)
        scale_error = poly.range_normal_iv_spatial(scale_coeff_error, tables)
        new_tmv_rem = iv.add(new_tmv_rem, scale_error)
        if sr is not None:
            # J lives before normalization: transport only the fresh error
            # back by S. Never append the full remainder (it includes j_lin).
            j_new = iv.add(j_new, iv.mul_point(scale_error, S))
    else:
        new_tmv_rem = iv.mul_point(new_tmv_rem, scale)  # remainder scales too

    if sr is not None:
        # Flow* Continuous.cpp:2296-2322: scalars remember this step's invS
        # (0 for point dims). Parity keeps Flow*'s INITIAL_SIMP (1e-4);
        # strict uses the configured cutoff for the normalized right map.
        sr.scalars = torch.where(point_dim, torch.zeros_like(scale), scale)
        if strict:
            assert scale_iv is not None
            sr.scalars_iv = torch.where(
                point_dim.unsqueeze(-1),
                iv.from_point(torch.zeros_like(scale)),
                scale_iv,
            )
        tmv_full2 = torch.zeros(B, n, T, dtype=x0_sp.dtype, device=x0_sp.device)
        tmv_full2[..., tables.spatial_index] = new_tmv_coeffs
        kept, dropped_rng = poly.cutoff_normal(
            tmv_full2, tables, step, settings.cutoff if strict else INITIAL_SIMP
        )
        new_tmv_coeffs = kept[..., tables.spatial_index]
        new_tmv_rem = iv.add(new_tmv_rem, dropped_rng)
        if strict:
            j_new = iv.add(j_new, iv.mul_point(dropped_rng, S))
            sr.append_j(j_new)

    new_x0 = torch.zeros(B, n, T, dtype=x0_sp.dtype, device=x0_sp.device)
    new_x0[..., 0] = c0
    var_full = tables.spatial_index[sched.var_image]  # [n]
    # One advanced-indexing scatter (rows are distinct: (i, var_full[i]) pairs).
    new_x0[:, torch.arange(n, device=x0_sp.device), var_full] = S

    # (f) Polynomial Picard iterations (Continuous.cpp:952-957): exactly
    #     `order` sweeps, the i-th truncating f at degree (i-1, min 1).
    x = new_x0
    for i in range(1, order + 1):
        k_i = i - 1 if i > 1 else 1
        f = exec_point(code, x, k_i, tables, step, cutoff_eps)  # [B, n, T]
        int_f = poly.integrate_t(f, tables, max_deg_in=min(k_i, 2 * tables.k - 1))
        x_new = new_x0.clone()
        x_new[..., : int_f.shape[-1]] += int_f  # x0 + \int f dt (tier-P adds)
        x = x_new

    # (g) A-priori remainder guess (Continuous.cpp:961-964).
    x_rem = rem_est.clone()  # [B, n, 2]

    cur, ok, bad = _validate_and_refine(
        code, x, new_x0, x_rem, tables, step, settings, strict
    )

    # (l) Emit (Continuous.cpp:1038-1042) — only ok lanes advance; failed and
    #     inactive lanes carry their previous state (frozen). Failure kind:
    #     FAILED_DIV where a domain violation fired (M5), else contraction.
    was_active = fp.status == ACTIVE  # [B]
    ok = ok & ~bad & was_active
    sel = ok.view(B, 1, 1)
    fail_code = torch.where(
        bad,
        torch.full_like(fp.status, FAILED_DIV),
        torch.full_like(fp.status, FAILED_CONTRACTION),
    )  # [B]
    out = FlowpipeBatch(
        pre_coeffs=torch.where(sel, x, fp.pre_coeffs),
        pre_rem=torch.where(sel, cur, fp.pre_rem),
        tmv_coeffs=torch.where(sel, new_tmv_coeffs, fp.tmv_coeffs),
        tmv_rem=torch.where(sel, new_tmv_rem, fp.tmv_rem),
        status=torch.where(ok | ~was_active, fp.status, fail_code),
    )
    return out, ok


def advance_adaptive(
    fp: FlowpipeBatch,
    code: CompiledODE,
    tables: MonomialTables,
    sched: CompositionSchedule,
    settings: Settings,
    rem_est: torch.Tensor,
    prev_deltas: torch.Tensor,
    cand_deltas: torch.Tensor,
) -> tuple[FlowpipeBatch, torch.Tensor, torch.Tensor]:
    """Adaptive-stepsize step (Flow* advance_adaptive_stepsize, Continuous.cpp:1233).

    prev_deltas [B]: each lane's PREVIOUS accepted step (stage (a) evaluates
    the old flowpipe at ITS OWN end time). cand_deltas [B]: the entry
    candidates (grown by the caller, Continuous.h:726-730). Per-lane retry:
    a failing lane halves its candidate (LAMBDA_DOWN = 0.5, include.h) until
    the contraction passes or the step would fall below settings.step_min
    (then the lane fails permanently, like Flow* returning 0).

    Returns (out, ok [B], accepted_deltas [B]). The polynomial half of the
    step is delta-independent (Flow* reuses it across retries; so do we —
    _validate_and_refine is the only re-run work). No SR support (guarded in
    Settings — no shipped benchmark combines them).

    Retry batching note: every retry re-runs the validated pass for ALL lanes
    at their CURRENT candidate; lanes whose candidate did not change reproduce
    their previous result bitwise (deterministic pipeline), so convergence is
    per-lane monotone exactly as in Flow*'s scalar loop.
    """
    B, n, T = fp.pre_coeffs.shape
    strict = settings.mode == "strict"
    order = settings.order
    cutoff_eps = settings.cutoff

    step_prev = build_step_tables_batched(tables, prev_deltas)

    # Stages (a)-(g) — identical to advance() with per-lane previous tables.
    if strict:
        x0_sp_full, endpoint_roundoff = poly.evaluate_time_end_with_roundoff(
            fp.pre_coeffs, tables, step_prev
        )
        x0_rem = iv.add(fp.pre_rem, endpoint_roundoff)
    else:
        x0_sp_full = poly.evaluate_time_end(fp.pre_coeffs, tables, step_prev)  # [B, n, Ts]
        x0_rem = fp.pre_rem
    c0 = x0_sp_full[..., 0].clone()  # [B, n]
    x0_sp = x0_sp_full.clone()
    x0_sp[..., 0] = 0.0

    new_tmv_coeffs, new_tmv_rem = compose(
        x0_sp, x0_rem, fp.tmv_coeffs, fp.tmv_rem,
        tables, step_prev, sched, order, cutoff_eps, strict,
    )

    tmv_full = torch.zeros(B, n, T, dtype=x0_sp.dtype, device=x0_sp.device)
    tmv_full[..., tables.spatial_index] = new_tmv_coeffs
    rng = iv.add(poly.range_normal(tmv_full, tables, step_prev), new_tmv_rem)
    S = iv.mag(rng)  # [B, n]
    point_dim = S <= 2.2250738585072014e-308
    S = torch.where(point_dim, torch.zeros_like(S), S)
    scale = torch.where(point_dim, torch.ones_like(S), 1.0 / S.clamp(min=1e-300))
    scale_iv = None
    pre_scale_coeffs = new_tmv_coeffs
    if strict:
        safe_s = torch.where(point_dim, torch.ones_like(S), S)
        scale_iv, _scale_bad = iv.rec(iv.from_point(safe_s))
        scale_iv = torch.where(
            point_dim.unsqueeze(-1), iv.from_point(torch.ones_like(S)), scale_iv
        )
    new_tmv_coeffs = new_tmv_coeffs * scale.unsqueeze(-1)
    if strict:
        assert scale_iv is not None
        exact_scaled_coeffs = iv.mul(
            iv.from_point(pre_scale_coeffs), scale_iv.unsqueeze(-2)
        )
        scale_coeff_error = iv.sub(
            exact_scaled_coeffs, iv.from_point(new_tmv_coeffs)
        )
        new_tmv_rem = iv.mul(new_tmv_rem, scale_iv)
        new_tmv_rem = iv.add(
            new_tmv_rem,
            poly.range_normal_iv_spatial(scale_coeff_error, tables),
        )
    else:
        new_tmv_rem = iv.mul_point(new_tmv_rem, scale)

    new_x0 = torch.zeros(B, n, T, dtype=x0_sp.dtype, device=x0_sp.device)
    new_x0[..., 0] = c0
    var_full = tables.spatial_index[sched.var_image]
    new_x0[:, torch.arange(n, device=x0_sp.device), var_full] = S

    x = new_x0
    for i in range(1, order + 1):
        k_i = i - 1 if i > 1 else 1
        f = exec_point(code, x, k_i, tables, step_prev, cutoff_eps)
        int_f = poly.integrate_t(f, tables, max_deg_in=min(k_i, 2 * tables.k - 1))
        x_new = new_x0.clone()
        x_new[..., : int_f.shape[-1]] += int_f
        x = x_new

    x_rem = rem_est.clone()  # [B, n, 2]

    # Per-lane retry loop over candidate steps (Continuous.cpp:1379-1410).
    was_active = fp.status == ACTIVE  # [B]
    deltas = cand_deltas.clone()  # [B]
    dead = torch.zeros(B, dtype=torch.bool, device=x.device)  # below step_min
    while True:
        step_v = build_step_tables_batched(tables, deltas)
        cur, ok_v, bad = _validate_and_refine(
            code, x, new_x0, x_rem, tables, step_v, settings, strict
        )
        need_retry = was_active & ~dead & ~ok_v & ~bad  # [B]
        halved = deltas * 0.5  # LAMBDA_DOWN (include.h:61)
        can = need_retry & (halved >= settings.step_min)
        newly_dead = need_retry & ~can
        dead = dead | newly_dead
        if not bool(can.any()):
            break
        deltas = torch.where(can, halved, deltas)

    ok = ok_v & ~bad & ~dead & was_active  # [B]
    sel = ok.view(B, 1, 1)
    fail_code = torch.where(
        bad,
        torch.full_like(fp.status, FAILED_DIV),
        torch.full_like(fp.status, FAILED_CONTRACTION),
    )
    out = FlowpipeBatch(
        pre_coeffs=torch.where(sel, x, fp.pre_coeffs),
        pre_rem=torch.where(sel, cur, fp.pre_rem),
        tmv_coeffs=torch.where(sel, new_tmv_coeffs, fp.tmv_coeffs),
        tmv_rem=torch.where(sel, new_tmv_rem, fp.tmv_rem),
        status=torch.where(ok | ~was_active, fp.status, fail_code),
    )
    return out, ok, deltas


def reach(
    rhs: list[str],
    var_names: list[str],
    boxes: torch.Tensor,
    time_horizon: float,
    settings: Settings,
    record_tms: bool = False,
    safe_set=None,
) -> ReachResult:
    """Fixed-step reachability (Flow* ODE::reach, Continuous.h:589-663).

    safe_set: optional safety.SafeSet — per-step verdicts per lane, with
    Flow*'s stop-on-UNSAFE / sticky-UNKNOWN aggregation (Continuous.h:615-635).

    boxes [B, n, 2] initial boxes (same ODE across the batch). Replicates the
    loop quirks verbatim: t starts at THRESHOLD_HIGH, accumulates in plain
    double, and the final flowpipe is computed for the FULL step with only its
    recorded delta clamped to the remaining time.
    """
    from .determinism import enable_determinism

    enable_determinism(settings.device)

    # M5: elementary/div cache layouts are sized for the order the validated
    # pass runs at — Flow*'s k = order - 1 (Picard_ctrunc_normal, TaylorModel.h:3711).
    code = compile_ode(rhs, var_names, order=settings.order - 1)
    n = code.n
    if boxes.shape[1] != n:
        raise ValueError(f"boxes have {boxes.shape[1]} dims, ODE has {n}")

    tables = build_tables(n, settings.order).to(settings.device)
    step = build_step_tables(tables, settings.step)
    sched = build_schedule(n, settings.order, settings.device)

    boxes = boxes.to(device=settings.device, dtype=torch.float64)
    fp = initial_flowpipe(boxes, tables)
    B = fp.batch

    rem_est = build_rem_est(settings, n, B)  # [B, n, 2]

    result = ReachResult(
        steps_completed=torch.zeros(B, dtype=torch.int64),
        status=torch.zeros(B, dtype=torch.int8),
    )

    if safe_set is not None:
        result.safety = torch.zeros(B, dtype=torch.int8)  # SAFE

    if settings.step_min > 0:
        return _reach_adaptive(
            fp, code, tables, sched, settings, rem_est, time_horizon, record_tms,
            result, safe_set,
        )

    sr = None
    if settings.sr_queue > 0:
        sr = make_symbolic_remainder(B, n, settings.sr_queue, settings.device)

    delta = settings.step
    t = THRESHOLD_HIGH
    while t < time_horizon:
        fp, ok = advance(fp, code, tables, step, sched, settings, rem_est, sr)
        if not bool(ok.any()):
            break

        remaining = time_horizon - t
        delta_used = delta if remaining >= delta else remaining

        if safe_set is not None:
            fp = _apply_safety(fp, ok, safe_set, tables, step, settings, result)

        result.steps_completed += ok.cpu().long()
        if record_tms:
            result.records.append(
                StepRecord(
                    pre_coeffs=fp.pre_coeffs.cpu().clone(),
                    pre_rem=fp.pre_rem.cpu().clone(),
                    tmv_coeffs=fp.tmv_coeffs.cpu().clone(),
                    tmv_rem=fp.tmv_rem.cpu().clone(),
                    delta_used=delta_used,
                    active_mask=ok.cpu().clone(),
                )
            )

        t += delta  # plain double accumulation, like Flow*

        # SR queue reset once full — LAST statement of the step body, after the
        # time bump, like Flow* (Continuous.h:891-894). SymbolicRemainder.
        # reset_if_full is the single implementation of that rule (its docstring
        # carries the placement + soundness argument); every reach loop in the
        # repo, including integrations/crown_reach/gpu_driver.py, calls it.
        if sr is not None:
            sr.reset_if_full()

    result.status = fp.status.cpu().clone()
    # Lanes still ACTIVE at the horizon are DONE.
    result.status[result.status == ACTIVE] = DONE
    result.final_fp = fp
    return result


def _reach_adaptive(
    fp: FlowpipeBatch,
    code: CompiledODE,
    tables: MonomialTables,
    sched: CompositionSchedule,
    settings: Settings,
    rem_est: torch.Tensor,
    time_horizon: float,
    record_tms: bool,
    result: ReachResult,
    safe_set=None,
) -> ReachResult:
    """Adaptive reach loop (Flow* reach_adaptive_stepsize, Continuous.h:665-747).

    Per-lane time and step bookkeeping: each lane accumulates its own t in
    plain double (like Flow*'s scalar loop), grows its candidate by LAMBDA_UP
    (1.1) after success — capped by KEEPING the current step when the grown
    value would exceed step_max - THRESHOLD_HIGH (Flow* sets new_stepsize = -1,
    i.e. "leave tables unchanged", Continuous.h:726-730) — and stops at its own
    horizon crossing. In adaptive records, delta_used is a [B] cpu tensor of
    per-lane (last-step-clamped) advances.
    """
    B = fp.batch
    dev = fp.pre_coeffs.device
    lambda_up = 1.1  # LAMBDA_UP (include.h:62)

    t = torch.full((B,), THRESHOLD_HIGH, dtype=torch.float64, device=dev)
    # step == step_max in adaptive mode; initial tables start there
    # (setAdaptiveStepsize -> setStepsize(step_max), Continuous.cpp:176-180).
    prev_deltas = torch.full((B,), settings.step, dtype=torch.float64, device=dev)
    cand = prev_deltas.clone()

    while True:
        # Freeze lanes that reached their horizon (per-lane DONE).
        finished = (fp.status == ACTIVE) & (t >= time_horizon)
        fp.status = torch.where(finished, torch.full_like(fp.status, DONE), fp.status)
        if not bool((fp.status == ACTIVE).any()):
            break

        fp, ok, accepted = advance_adaptive(
            fp, code, tables, sched, settings, rem_est, prev_deltas, cand
        )
        if not bool(ok.any()):
            break

        remaining = (time_horizon - t).clamp(min=0.0)  # [B]
        clamped = torch.minimum(accepted, remaining)  # last-step clamp, per lane

        if safe_set is not None:
            step_v = build_step_tables_batched(tables, accepted)
            fp = _apply_safety(fp, ok, safe_set, tables, step_v, settings, result)

        result.steps_completed += ok.cpu().long()
        if record_tms:
            result.records.append(
                StepRecord(
                    pre_coeffs=fp.pre_coeffs.cpu().clone(),
                    pre_rem=fp.pre_rem.cpu().clone(),
                    tmv_coeffs=fp.tmv_coeffs.cpu().clone(),
                    tmv_rem=fp.tmv_rem.cpu().clone(),
                    delta_used=clamped.cpu().clone(),
                    active_mask=ok.cpu().clone(),
                )
            )

        t = torch.where(ok, t + clamped, t)
        prev_deltas = torch.where(ok, accepted, prev_deltas)
        grown = accepted * lambda_up
        keep = grown > settings.step - THRESHOLD_HIGH  # cap: keep current step
        cand = torch.where(ok, torch.where(keep, accepted, grown), cand)

    result.status = fp.status.cpu().clone()
    result.status[result.status == ACTIVE] = DONE
    result.final_fp = fp
    return result


def _apply_safety(
    fp: FlowpipeBatch,
    ok: torch.Tensor,
    safe_set,
    tables: MonomialTables,
    step: StepTables,
    settings: Settings,
    result: ReachResult,
) -> FlowpipeBatch:
    """Per-step safety verdicts + Flow*'s aggregation (Continuous.h:615-635).

    UNSAFE on a fresh flowpipe stops its lane (status DONE — mirroring Flow*'s
    immediate COMPLETED_UNSAFE return); UNKNOWN downgrades the lane verdict
    stickily; lanes already UNSAFE never upgrade.
    """
    from . import safety as sf

    v = sf.check_step(
        fp.pre_coeffs, fp.pre_rem, fp.tmv_coeffs, fp.tmv_rem,
        safe_set, tables, step, settings,
    ).cpu()  # [B]
    stepped = ok.cpu()
    cur = result.safety  # [B] int8
    hit_unsafe = stepped & (v == sf.UNSAFE)
    hit_unknown = stepped & (v == sf.UNKNOWN) & (cur != sf.UNSAFE)
    cur[hit_unsafe] = sf.UNSAFE
    cur[hit_unknown & (cur == sf.SAFE)] = sf.UNKNOWN
    # Stop unsafe lanes on-device.
    dev_unsafe = hit_unsafe.to(fp.status.device)
    fp.status = torch.where(
        dev_unsafe, torch.full_like(fp.status, DONE), fp.status
    )
    return fp
