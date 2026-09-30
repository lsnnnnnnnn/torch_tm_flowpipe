"""Batched safety checking (Flow* safetyChecking, Continuous.cpp:10930-11066).

A safe set is a conjunction of constraints q_j(x) <= b_j (Flow* `Constraint`:
an expression over the state variables plus a bound). Per flowpipe the verdict
is one of (Flow* include.h:68-70):

    SAFE (0)     every constraint's sup over the flowpipe is <= its bound
    UNSAFE (-1)  some constraint's inf EXCEEDS its bound — the flowpipe lies
                 entirely outside the safe set (no intersection at all)
    UNKNOWN (1)  neither provable

Algorithm, mirroring Flow* stage for stage:
  1. BOX PHASE: interval-evaluate each q_j on the flowpipe's box (composed
     range over the whole domain). Decides most steps for the benchmark-style
     linear constraints.
  2. REFINEMENT: compose q_j over the flowpipe's Taylor model (giving a TM of
     q_j along the pipe), then branch-and-bound over the TIME dimension only
     (Flow* never splits space — GOTCHAS/plan D7): bisect [0, delta] until
     every piece proves SAFE or pieces reach width REFINEMENT_PREC (1e-5) and
     the verdict stays UNKNOWN.

Batched deviations (documented, sound):
  * the time bisection runs breadth-synchronized across lanes with a frontier
    cap (`MAX_FRONTIER` pieces per lane); overflowing lanes settle for UNKNOWN
    — never a wrong verdict, only a weaker one;
  * q's TM is built with our tape machinery (exec_valid on the composed
    flowpipe TM) — same remainder-bookkeeping caveats as composition (plan D7:
    functional parity).

E1 (2026-07-30) — refined ranges for EXTERNAL checkers
------------------------------------------------------
`check_step` answers the Flow* three-valued safe-set question. Other checkers
(the CROWN-Reach NNCS driver, `integrations/crown_reach/gpu_driver.py`) need a
different shape: they own their own verdict algebra (safe set / unsafe set /
target set, per-lane flags, early exit) and only need, per constraint, a
*number* — an enclosure of q_j over the flowpipe. Historically they used the
interval HULL of the flowpipe (per-variable ranges, then interval AST
evaluation), which throws away every correlation between state variables and is
therefore hopeless on COUPLED constraints such as `||v|| <= c + k*||r||`
(recorded as LIMIT 6).

`refine_ranges` (+ its sparse-state twin `refine_ranges_sparse`) exposes the
machinery above as exactly that number: a sound interval enclosure of each
q_j over the composed Taylor model (tmvPre o tmv), refined by the SAME
breadth-synced time bisection and the SAME frontier cap. `hull_undecided` and
`tighten` are the two glue predicates callers need to use it as an
*escalation*: cheap hull first, refined enclosure only where the hull cannot
decide, and the two intersected so the answer can only get tighter.

Soundness argument for that escalation (also spelled out at each function):
  1. the hull enclosure and the refined enclosure both OVER-approximate the
     same set (q_j over the current flowpipe), by different routes;
  2. if the hull already decides the caller's question (every constraint's sup
     within bound => provably inside; some constraint's inf beyond bound =>
     provably outside), no refinement can overturn it — both routes enclose the
     truth, so the decided side is a proof. Skipping refinement there is exact,
     not an approximation (and it is what keeps the cost of the previously
     shipped benchmarks unchanged);
  3. where the hull is undecided, the refined enclosure is itself an
     over-approximation, so any verdict read off it is a proof as well;
  4. the intersection of two enclosures of the same set is an enclosure of that
     set, so `tighten(hull, refined)` is sound and never wider than the hull:
     verdicts can only become tighter, never flip to a wrong answer;
  5. when the frontier cap or the REFINEMENT_PREC floor stops the bisection,
     the enclosure returned is simply the coarser one accumulated so far — the
     caller's question then stays undecided (UNKNOWN). Overflow never produces
     a SAFE/VERIFIED claim.

F2 (2026-07-30) — enclosures over a SUB-INTERVAL of the step's time
-------------------------------------------------------------------
Some specifications are only asserted over a time WINDOW (the official
ARCH-COMP CartPole property is `for t in [8, 10] s: x1, x3, x4 in [-b, b]` on a
10 s horizon). A checker that walks the flowpipe step by step therefore meets
three kinds of step: entirely outside the window (must NOT be checked — a
violation there is not a violation of the property, so checking it would
manufacture a spurious falsification), entirely inside (checked as before), and
STRADDLING a window endpoint. The straddling step is the interesting one: it
may neither be skipped (a violation in its in-window part would escape) nor
checked whole (the out-of-window part is not part of the property, and a
whole-step check that decides "entirely outside the safe set" would again be
attributing an out-of-window violation to the property). It has to be checked
on the INTERSECTED sub-interval.

Both enclosure routes therefore take an explicit time domain:
  * `rows_range_over_time` is the interval-hull route restricted to a
    sub-interval: it is `poly.range_normal(...) + rem` with the monomial time
    factor `[0, delta]^t_deg` replaced by `piece^t_deg` (that is exactly how
    `StepTables.factor_iv` is built, monomials.py `cat_iv` x time powers), so
    at `piece == [0, delta]` it reproduces the whole-step hull bit for bit;
  * `refine_ranges(..., time_domain=piece)` seeds the SAME bisection frontier
    at `piece` instead of `[0, delta]`, so the composed-TM route is restricted
    to the same sub-interval (the frontier only ever covers its seed, see the
    coverage argument in `refine_ranges`).
Both stay sound for any `piece` (they enclose q_j over `piece` x [-1,1]^n); the
requirement `piece subset [0, delta]` is about MEANING, not soundness — outside
the step's own time domain the flowpipe's Taylor model is not an enclosure of
the plant. Callers own the intersection arithmetic and must round it OUTWARD
(covering a little more of the step is sound in both directions as long as the
step really does intersect the window, because the enclosure then still covers
in-window times; covering less could drop an in-window sliver).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import interval as iv
from . import polynomial as poly
from .expr_parser import Bin, Node, Num, NumIv, Un, Var, parse
from .monomials import MonomialTables
from .ode_compiler import CompiledODE, compile_ode, exec_valid
from .polynomial import StepTables

# Verdicts (Flow* include.h:68-70).
UNSAFE = -1
SAFE = 0
UNKNOWN = 1

REFINEMENT_PREC = 1e-5  # include.h:110 — time-bisection floor
MAX_FRONTIER = 64  # batched cap on simultaneous time pieces per lane

# Transient-memory budget for _spatial_images' [B, M, Ps] pair products
# (member-axis chunking, bitwise-neutral — see the comment there). Module
# level so tests can shrink it to exercise the chunked path on small bases.
_SPEC_IMG_BUDGET_BYTES = 2e9


@dataclass(frozen=True)
class SafeSet:
    """Compiled safe set: conjunction of q_j(x) <= bound_j.

    exprs: parsed constraint ASTs (over the state variables).
    bounds [C]: the b_j.
    tapes: compiled tapes of the q_j for TM composition in refinement.
    """

    exprs: tuple[Node, ...]
    bounds: torch.Tensor
    tapes: tuple[CompiledODE, ...]


def compile_safe_set(
    constraints: list[str], bounds: list[float], var_names: list[str], order: int
) -> SafeSet:
    """Parse `q_j <= bounds[j]` constraint expressions against the state vars.

    `order` sizes the refinement tapes' cache layout: Flow* evaluates safety
    TMs at tm_setting.order (Continuous.cpp:10989), unlike the Picard k-1.
    """
    if len(constraints) != len(bounds):
        raise ValueError(f"{len(constraints)} constraints vs {len(bounds)} bounds")
    exprs = tuple(parse(c, var_names) for c in constraints)
    tapes = tuple(
        compile_ode([c], var_names, order=order, require_square=False)
        for c in constraints
    )
    return SafeSet(
        exprs=exprs, bounds=torch.tensor(bounds, dtype=torch.float64), tapes=tapes
    )


def eval_expr_interval(node: Node, box: torch.Tensor) -> torch.Tensor:
    """Interval-evaluate a constraint AST on state boxes: box [B, n, 2] -> [B, 2].

    Mirrors AST_Node::evaluate(Interval&, domain) (expression.h:600-682) —
    plain interval arithmetic, no Taylor models. Division by a zero-crossing
    interval yields [-inf, inf] (sound; the verdict then falls to UNKNOWN
    rather than Flow*'s [-1e5, 1e5] fallback, GOTCHAS #2).
    """
    if isinstance(node, Num):
        c = torch.full((box.shape[0],), node.value, dtype=box.dtype, device=box.device)
        return iv.from_point(c)  # [B, 2]
    if isinstance(node, NumIv):
        lo = torch.full((box.shape[0],), node.lo, dtype=box.dtype, device=box.device)
        hi = torch.full((box.shape[0],), node.hi, dtype=box.dtype, device=box.device)
        return iv.make(lo, hi)  # [B, 2]
    if isinstance(node, Var):
        return box[:, node.index]  # [B, 2]
    if isinstance(node, Un):
        a = eval_expr_interval(node.a, box)
        if node.op == "neg":
            return iv.neg(a)
        from . import transcendental as tr

        if node.op == "sin":
            return tr.sin_iv(a)
        if node.op == "cos":
            return tr.cos_iv(a)
        if node.op == "exp":
            return tr.exp_iv(a)
        if node.op == "log":
            return tr.log_iv(a)[0]
        if node.op == "sqrt":
            return tr.sqrt_iv(a)[0]
        raise AssertionError(f"unknown unary {node.op}")
    assert isinstance(node, Bin)
    if node.op == "^":
        base = eval_expr_interval(node.a, box)
        return iv.pow_int(base, int(node.b.value)) if int(node.b.value) >= 1 else iv.from_point(
            torch.ones(box.shape[0], dtype=box.dtype, device=box.device)
        )
    a = eval_expr_interval(node.a, box)
    b = eval_expr_interval(node.b, box)
    if node.op == "+":
        return iv.add(a, b)
    if node.op == "-":
        return iv.sub(a, b)
    if node.op == "*":
        return iv.mul(a, b)
    if node.op == "/":
        return iv.div(a, b)[0]
    raise AssertionError(f"unknown binary {node.op}")


def compose_flowpipe_full(
    pre_coeffs: torch.Tensor,
    pre_rem: torch.Tensor,
    m_coeffs: torch.Tensor,
    m_rem: torch.Tensor,
    tables: MonomialTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose tmvPre with PREBUILT spatial monomial images, KEEPING time.

    Unlike the advance-time composition (which first substitutes t := delta),
    safety needs the flowpipe as a TM over (t, s): each full-basis monomial
    t^d0 * r^e maps to t^d0 * image(r^e). Given the images M over the spatial
    basis (m_coeffs [B, Ts, Ts], m_rem [B, Ts, 2] — from composition's DAG),
    the composed coefficient tensor scatters per t-degree block:

        out[i, (d0, sp2)] = sum_m pre[i, (d0, m)] * M[m, sp2]

    pre_coeffs [B, n, T] -> (composed [B, n, T2'], rem [B, n, 2]) where the
    output lives on the full PRODUCT basis (degree <= 2k: spatial images have
    degree <= k and t-power adds d0 <= k).
    """
    B, n, T = pre_coeffs.shape
    dev, dt = pre_coeffs.device, pre_coeffs.dtype
    out = torch.zeros(B, n, tables.T2, dtype=dt, device=dev)

    # Group working-basis monomials by t-degree; for block d0 the spatial part
    # indexes the spatial basis, and targets are the t^d0-carrying monomials of
    # the product basis (found via exponent arithmetic on the CPU tables once).
    t_deg = tables.t_deg[:T]  # [T]
    rem = pre_rem.clone()  # [B, n, 2] — pre's own remainder rides along
    for d0 in range(0, tables.k + 1):
        rows = (t_deg == d0).nonzero(as_tuple=True)[0]  # [Rd] working-basis idx
        if rows.numel() == 0:
            continue
        sp_of = tables.eval_t_spatial[rows]  # [Rd] spatial index of each row
        w = pre_coeffs[..., rows]  # [B, n, Rd] coefficients of this block
        # Contract with images: [B, n, Rd] x [B, Rd, Ts] -> [B, n, Ts]
        img = m_coeffs[:, sp_of]  # [B, Rd, Ts]
        blk = torch.einsum("bnr,brs->bns", w, img)  # [B, n, Ts] (tier P)
        # Scatter into the t^d0 block of the product basis: target index of
        # spatial monomial s2 with t-degree d0.
        tgt = _tpow_targets(tables, d0)  # [Ts]
        out[..., tgt] += blk
        # Remainder: sum_m |w| pattern — point weights x image remainders,
        # times the t^d0 range factor [0..1]*delta^d0 <= handled by caller's
        # range evaluation; here the IMAGE remainders enter scaled by w:
        w_rem = iv.dot_point_iv(w, m_rem[:, sp_of].unsqueeze(1), dim=-1)  # [B, n, 2]
        # t^d0 over [0, delta] lies in [0, 1]-scaled powers; bounding with the
        # WORST monomial factor ([-1,1]-symmetric never occurs for t powers,
        # [0,1] does) keeps this sound for any delta <= 1; general case uses
        # max(1, delta^d0) — callers pass delta <= 1 in every shipped config,
        # asserted upstream.
        rem = iv.add(rem, w_rem if d0 == 0 else iv.mul(w_rem, _unit01(w_rem)))
    return out, rem


def _unit01(like: torch.Tensor) -> torch.Tensor:
    """The interval [0, 1] broadcast like `like` ([..., 2])."""
    z = torch.zeros_like(like[..., 0])
    return torch.stack((z, torch.ones_like(z)), dim=-1)


_tpow_cache: dict[tuple[int, int, int, str], torch.Tensor] = {}


def _tpow_targets(tables: MonomialTables, d0: int) -> torch.Tensor:
    """Product-basis indices of t^d0 * (spatial basis monomial s), s in order.

    Built once per (tables, d0) from exponent arithmetic (CPU) and cached.
    """
    key = (tables.n, tables.k, d0, str(tables.exponents.device))
    hit = _tpow_cache.get(key)
    if hit is not None:
        return hit
    sp_exps = tables.exponents[tables.spatial_index]  # [Ts, n+1] (t column zero)
    target_exps = sp_exps.clone()
    target_exps[:, 0] = d0
    # Match rows in the product basis by exponent equality (small tables; the
    # O(Ts * T2) comparison runs once per (n, k, d0) and is cached).
    eq = (tables.exponents.unsqueeze(0) == target_exps.unsqueeze(1)).all(dim=-1)  # [Ts, T2]
    tgt = eq.float().argmax(dim=-1).to(torch.long)  # [Ts]
    if not bool(eq.any(dim=-1).all()):
        raise AssertionError("t-power target lookup failed")  # pragma: no cover  # basis closed under t-multiplication up to 2k
    _tpow_cache[key] = tgt
    return tgt


# ---------------------------------------------------------------------------
# Shared refinement machinery (used by check_step AND refine_ranges)
# ---------------------------------------------------------------------------


def _require_unit_step(step: StepTables) -> None:
    """Guard the composition precondition delta <= 1.

    compose_flowpipe_full bounds the t^d0 monomial factor by the interval
    [0, 1]; that is only valid for delta <= 1 (every shipped benchmark uses
    delta <= 0.1). Raising here keeps the precondition loud rather than
    silently unsound.
    """
    max_delta = float(step.deltas.max()) if step.lanes else step.delta
    if max_delta > 1.0:
        raise ValueError(
            f"safety refinement requires step <= 1 (got {max_delta}): the "
            "t^d0 remainder bound in compose_flowpipe_full assumes [0,1]"
        )


def _step_time_iv(step: StepTables, B: int) -> torch.Tensor:
    """The per-lane time domain [0, delta] as [B, 2] (fixed OR per-lane steps).

    step.dt() already carries the lane dim in adaptive/per-lane mode ([B, 1, 2]
    at ref_rank 2); the fixed-step table holds one shared [2] that broadcasts.
    """
    dt_iv = step.dt() if step.lanes else step.pow_iv[1].expand(B, 2)  # [B, 2]-ish
    return dt_iv.reshape(B, 2)


def _composed_tm(
    pre_coeffs: torch.Tensor,
    pre_rem: torch.Tensor,
    tmv_coeffs: torch.Tensor,
    tmv_rem: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
    settings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The flowpipe as ONE Taylor model over (t, s): (tmvPre o tmv), time kept.

    This is the step that preserves CORRELATIONS between state variables: every
    component is a polynomial in the SAME symbols (t, s1..sn), so a constraint
    evaluated on it sees the cancellations an interval hull cannot.

    pre_coeffs [B, n, T], pre_rem [B, n, 2], tmv_coeffs [B, n, Ts],
    tmv_rem [B, n, 2] -> (kept [B, n, T] point coefficients on the WORKING
    basis, rem [B, n, 2] the matching remainder).

    The composed polynomial lives on the product basis (degree <= 2k); it is
    conservatively truncated back to the working basis (ctrunc at order k, the
    dropped suffix's range folded into the remainder) because exec_valid — the
    tape evaluator used for the constraint itself — consumes working-basis
    coefficients.
    """
    n = pre_coeffs.shape[1]
    from .composition import build_schedule

    sched = build_schedule(n, tables.k, tables.device)
    # Monomial images of the spatial basis under tmv: m_coeffs [B, Ts, Ts],
    # m_rem [B, Ts, 2] (composition's level DAG, rebuilt locally).
    m_coeffs, m_rem = _spatial_images(tmv_coeffs, tmv_rem, tables, step, sched, settings)
    comp_c, comp_r = compose_flowpipe_full(
        pre_coeffs, pre_rem, m_coeffs, m_rem, tables
    )  # [B, n, T2], [B, n, 2]
    kept, tail = poly.ctrunc_normal(comp_c, tables, step, tables.k)  # [B, n, T], [B, n, 2]
    return kept, iv.add(comp_r, tail)


def _constraint_tm(
    tape: CompiledODE,
    kept: torch.Tensor,
    comp_rem: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
    settings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TM of ONE constraint q_j over the composed flowpipe TM.

    kept [B, n, T] / comp_rem [B, n, 2] from _composed_tm ->
    (q_c [B, T, 2] interval coefficients, q_r [B, 2] remainder,
     bad [B] bool: lanes where a div/log/sqrt domain was violated — their
    output is garbage and the caller MUST NOT trust it, GOTCHAS #2).
    """
    B = kept.shape[0]
    bad = torch.zeros(B, dtype=torch.bool, device=kept.device)
    q_c, q_r, _, _ = exec_valid(
        tape, kept, comp_rem, tables.k, tables, step, settings.cutoff, bad_out=bad,
    )  # q_c [B, 1, T, 2], q_r [B, 1, 2] — single-component tape
    return q_c[:, 0], q_r[:, 0], bad


def _split_frontier(frontier: torch.Tensor, split: torch.Tensor) -> torch.Tensor:
    """Bisect the marked time pieces: [B, W, 2] x [B, W] bool -> [B, 2W, 2].

    Unmarked pieces collapse to zero-width dummies at their own left endpoint
    (a SUBSET of the piece they replace) so the frontier stays a rectangular
    tensor; every caller must have already consumed those pieces' ranges.
    """
    # RN midpoint: what matters for COVERAGE of the parent piece is only
    # lo <= mid <= hi (guaranteed: rounding is monotone and *0.5 is exact), so
    # [lo, mid] u [mid, hi] = [lo, hi] whatever the rounding does.
    mid = (frontier[..., 0] + frontier[..., 1]) * 0.5  # [B, W]
    left = torch.stack((frontier[..., 0], mid), dim=-1)  # [B, W, 2]
    right = torch.stack((mid, frontier[..., 1]), dim=-1)  # [B, W, 2]
    dummy = torch.stack((frontier[..., 0], frontier[..., 0]), dim=-1)  # [B, W, 2]
    left = torch.where(split.unsqueeze(-1), left, dummy)
    right = torch.where(split.unsqueeze(-1), right, dummy)
    return torch.cat((left, right), dim=1)  # [B, 2W, 2]


def check_step(
    pre_coeffs: torch.Tensor,
    pre_rem: torch.Tensor,
    tmv_coeffs: torch.Tensor,
    tmv_rem: torch.Tensor,
    safe: SafeSet,
    tables: MonomialTables,
    step: StepTables,
    settings,
) -> torch.Tensor:
    """Safety verdict per lane for one flowpipe batch: -> [B] int8 in {-1,0,1}.

    Raises ValueError for step sizes > 1: compose_flowpipe_full's remainder
    bound treats t^d0 over [0, delta] as within [0, 1], which requires
    delta <= 1 (every shipped benchmark uses delta <= 0.1; the guard makes the
    precondition loud instead of silently unsound).

    Follows Flow* safetyChecking: box phase, then time-bisection refinement of
    the composed constraint TMs for still-ambiguous lanes.
    """
    B, n, _ = pre_coeffs.shape
    dev = pre_coeffs.device
    if len(safe.exprs) == 0:
        return torch.full((B,), SAFE, dtype=torch.int8, device=dev)

    # Soundness precondition of the refinement composition (see
    # compose_flowpipe_full's remainder comment): t^d0 subset of [0, 1] needs
    # delta <= 1. Guard here so the requirement can never rot silently.
    _require_unit_step(step)

    bounds = safe.bounds.to(dev)  # [C]

    # --- 1. box phase ------------------------------------------------------
    # Flowpipe box: range of the composed TM. Composing costs one DAG build;
    # the box equals range(pre over (t, r)) only when tmv maps into [-1,1]^n
    # (true by preconditioning) WIDENED by tmv's remainder — evaluating pre's
    # range directly over [-1,1]^n is Flow*-equivalent up to that remainder,
    # which we add via the composed construction below only in refinement.
    # For the box phase we take the sound, cheap route: range(pre) inflated by
    # a first-order tmv-remainder term is replaced by the composed range when
    # any constraint is near its bound; start with range(pre) + pre_rem.
    box = iv.add(poly.range_normal(pre_coeffs, tables, step), pre_rem)  # [B, n, 2]

    verdict = torch.full((B,), SAFE, dtype=torch.int8, device=dev)
    ambiguous = torch.zeros(B, dtype=torch.bool, device=dev)
    for j, expr in enumerate(safe.exprs):
        q = eval_expr_interval(expr, box)  # [B, 2]
        is_unsafe = bounds[j] < q[..., 0]  # bound < inf: no intersection
        not_contained = ~(q[..., 1] <= bounds[j])
        verdict = torch.where(
            is_unsafe, torch.full_like(verdict, UNSAFE), verdict
        )
        ambiguous = ambiguous | (not_contained & ~is_unsafe)
    ambiguous = ambiguous & (verdict != UNSAFE)
    if not bool(ambiguous.any()):
        return verdict

    # --- 2. time-bisection refinement -------------------------------------
    # Build the composed flowpipe TM over (t, s) once, then per constraint a
    # TM of q_j over it; evaluate ranges over bisected time pieces.
    kept, comp_rem = _composed_tm(
        pre_coeffs, pre_rem, tmv_coeffs, tmv_rem, tables, step, settings
    )  # kept [B, n, T] point coeffs, comp_rem [B, n, 2]
    # Bisection frontier seed: the time interval [0, delta] per lane, [B, 2].
    dt_iv = _step_time_iv(step, B)

    for j, tape in enumerate(safe.tapes):
        lanes = ambiguous.clone()
        if not bool(lanes.any()):
            continue  # pragma: no cover  # unreachable: ambiguous is nonempty (early return above) and loop-invariant
        # q_j over the composed TM: tape-evaluate with interval coefficients.
        q_c, q_r, badq = _constraint_tm(tape, kept, comp_rem, tables, step, settings)
        verdict = torch.where(
            badq & ambiguous, torch.full_like(verdict, UNKNOWN), verdict
        )

        frontier = dt_iv.unsqueeze(1).clone()  # [B, 1, 2]
        alive = lanes & ~badq
        while bool(alive.any()) and frontier.shape[1] <= MAX_FRONTIER:
            rng = _range_over_time(q_c, q_r, frontier, tables)  # [B, W, 2]
            piece_safe = rng[..., 1] <= bounds[j]  # [B, W]
            piece_dead = frontier[..., 1] - frontier[..., 0] <= REFINEMENT_PREC
            pending = ~piece_safe & alive.unsqueeze(1)  # [B, W]
            hit_floor = pending & piece_dead
            # Lanes with an ambiguous floor-width piece: UNKNOWN, stop.
            floor_lanes = hit_floor.any(dim=1)
            verdict = torch.where(
                floor_lanes & alive, torch.full_like(verdict, UNKNOWN), verdict
            )
            alive = alive & ~floor_lanes
            still = pending & ~piece_dead & alive.unsqueeze(1)  # [B, W]
            if not bool(still.any()):
                break
            # Split every still-pending piece in half; proven/irrelevant pieces
            # collapse to zero-width dummies (range check trivially safe).
            frontier = _split_frontier(frontier, still)  # [B, 2W, 2]
        # Overflowed lanes stay ambiguous -> UNKNOWN (cap documented above).
        verdict = torch.where(
            alive & (frontier.shape[1] > MAX_FRONTIER),
            torch.full_like(verdict, UNKNOWN),
            verdict,
        )
    return verdict


# ---------------------------------------------------------------------------
# E1: refined constraint ranges for external checkers (see module docstring)
# ---------------------------------------------------------------------------


def hull_undecided(q: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Lanes whose membership question the HULL enclosures leave open.

    q [B, C, 2]: an enclosure of each constraint q_j over the flowpipe (however
    obtained — the caller's interval-hull evaluation, typically).
    bounds [C]: the b_j of `q_j <= b_j`. -> [B] bool.

    A lane is DECIDED when either
      * every constraint's sup is within bound  (sup_j <= b_j for all j) — the
        flowpipe is entirely inside the region {q <= b}: PROVEN, because each
        enclosure over-approximates the truth; or
      * some constraint's inf exceeds its bound (inf_j > b_j) — every point of
        the flowpipe violates that constraint, so the flowpipe is entirely
        OUTSIDE the region: also proven (enclosure inf <= true inf).
    Anything else straddles a boundary and is the only case where escalating to
    the refined (composed-TM) enclosure can change the answer.

    Note both the conjunctive ("safe set: all q_j <= 0 must hold") and the
    membership ("is the flowpipe inside/outside this unsafe/target region")
    readings of a constraint conjunction have the SAME undecided set, which is
    why one predicate serves every check site: "not provably inside AND not
    provably outside".
    """
    inside = (q[..., 1] <= bounds).all(dim=1)  # [B] all sups within bound
    outside = (q[..., 0] > bounds).any(dim=1)  # [B] some inf beyond bound
    return ~inside & ~outside


def tighten(hull_q: torch.Tensor, refined_q: torch.Tensor) -> torch.Tensor:
    """Intersect two sound enclosures of the same quantity: [B, C, 2] each.

    Both arguments over-approximate q_j over the flowpipe (different routes:
    interval hull vs composed Taylor model), so their intersection does too —
    and it is never wider than either. Using it makes the refined path
    MONOTONE: a verdict can only get tighter, never flip to a wrong answer,
    even if a particular constraint happens to compose worse than it hulls.

    `refined_q` entries of [-inf, +inf] mean "no information" (lane/constraint
    not refined, or the constraint tape flagged a domain violation) and leave
    the hull untouched.

    The min/max are then clamped back into the hull's own endpoints: with two
    VALID intervals (lo <= hi each) that both contain a common point the raw
    intersection cannot be inverted, so the clamp is unreachable arithmetic
    insurance — it costs one min/max and guarantees the result can never read
    as simultaneously "provably inside" and "provably outside" should a future
    enclosure bug make the two disjoint.
    """
    lo = torch.maximum(hull_q[..., 0], refined_q[..., 0])  # [B, C]
    hi = torch.minimum(hull_q[..., 1], refined_q[..., 1])  # [B, C]
    lo = torch.minimum(lo, hull_q[..., 1])
    hi = torch.maximum(hi, hull_q[..., 0])
    return torch.stack((lo, hi), dim=-1)  # [B, C, 2]


def rows_range_over_time(
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    piece: torch.Tensor,
    tables: MonomialTables,
) -> torch.Tensor:
    """Interval hull of a flowpipe's tmvPre rows over a time SUB-INTERVAL (F2).

    coeffs [B, n, M] point coefficients on a PREFIX of the product basis (what
    `flowpipe.pre_coeffs` carries), rem [B, n, 2] the matching remainder,
    piece [B, 2] a per-lane time interval -> [B, n, 2].

    This is the same quantity `poly.range_normal(coeffs, tables, step) + rem`
    computes (the whole-step interval hull the NNCS driver has always used),
    with the time factor taken over `piece` instead of the step's own
    [0, delta]: monomial m contributes `cat_iv[m] * piece^t_deg[m]`, where
    `cat_iv` (monomials.py) is the monomial's SPATIAL factor ranged over
    [-1, 1]^n. That factorization is exactly how `StepTables.factor_iv` is
    precomputed for the whole step, so `piece == [0, delta]` reproduces the
    whole-step hull.

    Sound for any `piece` (it encloses each row over `piece` x [-1, 1]^n); the
    precondition `piece subset [0, delta]` is about MEANING, not soundness —
    see the module docstring's F2 section.
    """
    m_basis = coeffs.shape[-1]
    t_deg = tables.t_deg[:m_basis]  # [M] time exponent per monomial
    kmax = int(t_deg.max())  # working basis: <= tables.k; product basis: <= 2k
    tp = iv.powers(piece, kmax).movedim(0, 1)  # [B, kmax+1, 2] piece^j
    fac = iv.mul(
        tables.cat_iv[:m_basis].view(1, m_basis, 2),  # [1, M, 2] spatial factor
        tp[:, t_deg],  # [B, M, 2] time factor
    )  # [B, M, 2]
    rng = iv.dot_point_iv(coeffs, fac.unsqueeze(1), dim=-1)  # [B, n, 2]
    return iv.add(rng, rem)


def rows_range_over_time_sparse(
    st, eng, piece: torch.Tensor, idx_hi: int
) -> torch.Tensor:
    """rows_range_over_time for a SPARSE flowpipe state -> [B, idx_hi, 2].

    st: sparse_exec.SparseState; eng: support.SparseEngine; piece [B, 2] the
    time sub-interval; idx_hi: number of leading TM rows wanted (the driver
    asks for all declared variables). Re-embedding the support-aligned
    coefficients onto the complete degree<=k support is EXACT (the added slots
    are structural zeros), exactly as in `refine_ranges_sparse`; this path only
    runs on the at most two window-straddling steps of a run, so the dense
    re-embed is not on any hot path.
    """
    from . import support as spm

    tables = eng.tables
    sup_pre = spm.full_support(tables, tables.k, spatial=False)  # size T
    pre = spm.embed_s(st.pre[:, :idx_hi], eng, st.pre_sup, sup_pre)  # [B, idx_hi, T]
    return rows_range_over_time(pre, st.pre_rem[:, :idx_hi], piece, tables)


def refine_ranges(
    pre_coeffs: torch.Tensor,
    pre_rem: torch.Tensor,
    tmv_coeffs: torch.Tensor,
    tmv_rem: torch.Tensor,
    safe: SafeSet,
    tables: MonomialTables,
    step: StepTables,
    settings,
    lanes: torch.Tensor,
    time_domain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Composed-TM + time-bisection enclosures of every constraint: [B, C, 2].

    The tight counterpart of the caller's interval-hull evaluation: instead of
    evaluating q_j on per-variable ranges (correlations destroyed), q_j is
    evaluated as a TAYLOR MODEL over the composed flowpipe (tmvPre o tmv) —
    all components share the symbols (t, s1..sn), so cancellations survive —
    and the resulting (t, s)-TM is bounded over a bisected time frontier.

    Arguments are the flowpipe tensors check_step takes (pre_coeffs [B, n, T],
    pre_rem [B, n, 2], tmv_coeffs [B, n, Ts], tmv_rem [B, n, 2]) plus
    lanes [B] bool: which lanes to spend refinement on (the caller's hull
    pre-filter, see hull_undecided). Raises ValueError for delta > 1 exactly
    like check_step (same composition precondition).

    time_domain [B, 2] optional (F2): the time interval the enclosure must
    cover, seeding the bisection frontier in place of the step's own
    [0, delta]. Pass the INTERSECTION of the step's time interval with a
    specification's time window (in the step's LOCAL coordinates, i.e. 0 =
    start of this step) to get an enclosure of q_j over the in-window part of
    the step only. `None` = the whole step, i.e. the historical behaviour.

    Returns [B, C, 2] where, per (lane, constraint):
      * refined lanes carry a SOUND enclosure of q_j over the whole flowpipe;
      * unrefined lanes (lanes[i] False) and lanes the constraint tape flagged
        bad carry [-inf, +inf] — "no information", which `tighten` folds away.

    Bisection policy (per constraint, breadth-synchronized over lanes):
      * two goals are chased simultaneously — proving sup <= b_j (the flowpipe
        satisfies the constraint) and proving inf > b_j (it violates it
        everywhere). A piece entirely beyond the bound (its inf > b_j) PROVES
        sup > b_j and retires the first goal; a piece entirely within (sup <=
        b_j) proves inf <= b_j and retires the second. When both are retired
        the truth straddles the bound: no amount of bisection can decide it, so
        the lane stops splitting immediately (this is what keeps the cost
        bounded on genuinely-undecidable lanes);
      * a piece is split while it blocks a goal still being chased, its width
        is above REFINEMENT_PREC, and the frontier can still double within
        MAX_FRONTIER. Otherwise it RETIRES: its range is folded into the
        lane's accumulator.
    Soundness of the accumulator: the retired pieces plus the live ones always
    COVER the seed interval (a split replaces a piece by two halves whose union
    is the piece; a retired piece is never split again), and the loop only exits when
    nothing is left to split — i.e. after every live piece has been folded. So
    the accumulated [min lo, max hi] is a union of enclosures over a cover of
    the seed time interval: an enclosure of q_j over the flowpipe restricted to
    that interval (the whole flowpipe when `time_domain` is None). Collapsed
    dummies re-folded on later iterations are SUBSETS of their already-folded
    parent, so they can only add values that are already accounted for.
    """
    B, _n, _ = pre_coeffs.shape
    dev, dtyp = pre_coeffs.device, pre_coeffs.dtype
    C = len(safe.tapes)
    # "No information" default: [-inf, +inf] per (lane, constraint), [B, C, 2].
    out = torch.stack(
        (
            torch.full((B, C), -torch.inf, dtype=dtyp, device=dev),
            torch.full((B, C), torch.inf, dtype=dtyp, device=dev),
        ),
        dim=-1,
    )
    if C == 0 or not bool(lanes.any()):
        return out  # nothing to refine (empty spec, or the hull decided all)

    _require_unit_step(step)
    bounds = safe.bounds.to(dev)  # [C]
    kept, comp_rem = _composed_tm(
        pre_coeffs, pre_rem, tmv_coeffs, tmv_rem, tables, step, settings
    )  # [B, n, T], [B, n, 2]
    # Seed of the bisection: the whole step [0, delta], or the caller's
    # sub-interval (F2). Everything downstream only ever COVERS this seed, so
    # restricting it restricts the enclosure to that part of the step.
    dt_iv = (
        _step_time_iv(step, B) if time_domain is None
        else time_domain.reshape(B, 2).to(device=dev, dtype=dtyp)
    )  # [B, 2]

    for j, tape in enumerate(safe.tapes):
        q_c, q_r, bad = _constraint_tm(tape, kept, comp_rem, tables, step, settings)
        live = lanes & ~bad  # [B] lanes we both want AND may trust
        frontier = dt_iv.unsqueeze(1).clone()  # [B, 1, 2] time pieces
        # Accumulator over RETIRED pieces, seeded empty (lo = +inf, hi = -inf).
        acc_lo = torch.full((B,), torch.inf, dtype=dtyp, device=dev)  # [B]
        acc_hi = torch.full((B,), -torch.inf, dtype=dtyp, device=dev)  # [B]
        chase_sup = live.clone()  # [B] still trying to prove sup <= b_j
        chase_inf = live.clone()  # [B] still trying to prove inf  > b_j
        while True:
            rng = _range_over_time(q_c, q_r, frontier, tables)  # [B, W, 2]
            within = rng[..., 1] <= bounds[j]  # [B, W] piece entirely <= b_j
            beyond = rng[..., 0] > bounds[j]  # [B, W] piece entirely  > b_j
            # A single piece beyond the bound proves sup > b_j (its inf is a
            # lower bound of real attained values); a single piece within
            # proves inf <= b_j. Either kills the corresponding goal for good.
            chase_sup = chase_sup & ~beyond.any(dim=1)  # [B]
            chase_inf = chase_inf & ~within.any(dim=1)  # [B]
            # Pieces blocking a goal still being chased.
            pending = (chase_sup.unsqueeze(1) & ~within) | (
                chase_inf.unsqueeze(1) & ~beyond
            )  # [B, W]
            wide = (frontier[..., 1] - frontier[..., 0]) > REFINEMENT_PREC  # [B, W]
            split = pending & wide & live.unsqueeze(1)  # [B, W]
            if 2 * frontier.shape[1] > MAX_FRONTIER:
                # Frontier cap (the documented batched deviation): stop
                # refining and settle for the coarser enclosure. The caller's
                # question then stays undecided — never a SAFE claim.
                split = torch.zeros_like(split)
            # Retire (fold) every piece that will not be split. On the final
            # iteration split is all-False, so every live piece is folded.
            fold = live.unsqueeze(1) & ~split  # [B, W]
            acc_lo = torch.minimum(
                acc_lo, torch.where(fold, rng[..., 0], torch.inf).amin(dim=1)
            )
            acc_hi = torch.maximum(
                acc_hi, torch.where(fold, rng[..., 1], -torch.inf).amax(dim=1)
            )
            if not bool(split.any()):
                break
            frontier = _split_frontier(frontier, split)  # [B, 2W, 2]
        out[:, j, 0] = torch.where(live, acc_lo, out[:, j, 0])
        out[:, j, 1] = torch.where(live, acc_hi, out[:, j, 1])
    return out


def refine_ranges_sparse(
    st,
    eng,
    safe: SafeSet,
    settings,
    lanes: torch.Tensor,
    time_domain: torch.Tensor | None = None,
) -> torch.Tensor:
    """refine_ranges for a SPARSE flowpipe state (sparse_exec.SparseState).

    The sparse engine stores support-aligned coefficient tensors (pre [B, n, Sp]
    on FULL-basis ids, tmv [B, n, St] on SPATIAL ids); the refinement machinery
    is dense (its composition DAG and product-basis scatter index the full
    tables). Re-embedding onto the complete degree<=k supports is EXACT — the
    added slots hold structural zeros (support.py's zero-removal argument in
    reverse) — so this is a pure layout change, not an approximation.

    st: SparseState; eng: support.SparseEngine (carries tables + step tables);
    lanes [B] bool and time_domain [B, 2] | None as in refine_ranges.
    -> [B, C, 2].
    """
    from . import support as spm

    tables, step = eng.tables, eng.step
    sup_pre = spm.full_support(tables, tables.k, spatial=False)  # size T
    sup_tmv = spm.full_support(tables, tables.k, spatial=True)  # size Ts
    pre = spm.embed_s(st.pre, eng, st.pre_sup, sup_pre)  # [B, n, T]
    tmv = spm.embed_s(st.tmv, eng, st.tmv_sup, sup_tmv)  # [B, n, Ts]
    return refine_ranges(
        pre, st.pre_rem, tmv, st.tmv_rem, safe, tables, step, settings, lanes,
        time_domain=time_domain,
    )


def _spatial_images(
    tmv_coeffs: torch.Tensor,
    tmv_rem: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
    sched,
    settings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monomial images of the spatial basis under tmv (composition DAG reuse).

    Same build as composition.compose step 1; returns
    (m_coeffs [B, Ts, Ts], m_rem [B, Ts, 2]).
    """
    B, n, Ts = tmv_coeffs.shape
    dev, dt = tmv_coeffs.device, tmv_coeffs.dtype
    m_coeffs = torch.zeros(B, Ts, Ts, dtype=dt, device=dev)
    m_rem = torch.zeros(B, Ts, 2, dtype=dt, device=dev)
    m_coeffs[:, 0, 0] = 1.0
    vi = sched.var_image.to(dev)
    m_coeffs[:, vi] = tmv_coeffs
    m_rem[:, vi] = tmv_rem
    g_range = poly.range_normal_spatial(tmv_coeffs, tables)  # [B, n, 2]
    # Transient budget for the [B, M, Ps] pair products below. Per-member
    # work is INDEPENDENT (mul_point_spatial reduces along the pair axis
    # only), so chunking the member axis is bitwise-neutral — unlike batch
    # chunking of a reduction, no association changes. ACC-Chain n=20 at
    # B=1024 asked for 85.49 GiB unchunked (the E1-2 cost cliff hitting the
    # spec-refinement escalation, Phase-F oom cells).
    Ps = tables.sp_pair_i.shape[0]
    per_member = max(1, B * Ps * tmv_coeffs.element_size())
    m_chunk = max(1, int(_SPEC_IMG_BUDGET_BYTES // per_member))
    for members, parents, lastvars in sched.levels:
        members, parents, lastvars = members.to(dev), parents.to(dev), lastvars.to(dev)
        for s in range(0, members.shape[0], m_chunk):
            mem = members[s:s + m_chunk]
            pa = m_coeffs[:, parents[s:s + m_chunk]]
            gv = tmv_coeffs[:, lastvars[s:s + m_chunk]]
            prod = poly.mul_point_spatial(pa, gv, tables)
            pa_range = poly.range_normal_spatial(pa, tables)
            p_rem = m_rem[:, parents[s:s + m_chunk]]
            gv_rem = tmv_rem[:, lastvars[s:s + m_chunk]]
            rem = iv.mul(p_rem, gv_rem)
            rem = iv.add(rem, iv.mul(g_range[:, lastvars[s:s + m_chunk]], p_rem))
            rem = iv.add(rem, iv.mul(pa_range, gv_rem))
            kept, tail = poly.ctrunc_normal_spatial(prod, tables, tables.k)
            kept, round_rng = poly.cutoff_normal_spatial(kept, tables, settings.cutoff)
            rem = iv.add(rem, iv.add(tail, round_rng))
            out_row = torch.zeros(B, mem.shape[0], Ts, dtype=dt, device=dev)
            out_row[..., : kept.shape[-1]] = kept
            m_coeffs[:, mem] = out_row
            m_rem[:, mem] = rem
    return m_coeffs, m_rem


def _range_over_time(
    q_c: torch.Tensor, q_r: torch.Tensor, pieces: torch.Tensor, tables: MonomialTables
) -> torch.Tensor:
    """Range of a (t, s)-TM over time pieces x [-1,1]^n.

    q_c [B, T, 2] interval coefficients (working basis), q_r [B, 2] remainder,
    pieces [B, W, 2] time sub-intervals -> [B, W, 2].
    Per-monomial factor: cat_iv x piece^t_deg — piece powers via iv.powers.
    """
    T = q_c.shape[1]
    tp = iv.powers(pieces, tables.k)  # [k+1, B, W, 2]
    tp = tp.movedim(0, 2)  # [B, W, k+1, 2]
    fac = iv.mul(
        tables.cat_iv[:T].view(1, 1, T, 2),
        tp[:, :, tables.t_deg[:T]],  # [B, W, T, 2]
    )  # [B, W, T, 2]
    prod = iv.mul(q_c.unsqueeze(1), fac)  # [B, W, T, 2]
    rng = iv.sum(prod, dim=-1)  # [B, W, 2]
    return iv.add(rng, q_r.unsqueeze(1))
