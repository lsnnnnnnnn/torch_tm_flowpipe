#!/usr/bin/env python3
"""CROWN-Reach with a GPU plant engine: flowstar_gpu replaces Flow*-on-CPU.

A faithful port of CROWN-Reach's verification loop (src/CrownReach.cpp +
src/CrownSettings.cpp + src/crown.py at commit 7b90f30) onto the batched
flowstar_gpu machinery. The C++/RPC split disappears: CROWN (auto_LiRPA) and
the Taylor-model plant engine live in one Python process on one GPU, and the
branch-and-bound cells become the batch dimension instead of thread-pool tasks.

Loop semantics preserved from the original (per control step):
  1. interval hull of the NN-input Taylor models -> CROWN input boxes [B, nn_in];
  2. one batched CROWN call with 'same-slope' relaxations -> shared linear map
     T = lA (== uA) and bias interval [u_min, u_max] per cell;
  3. SYMBOLIC composition (CrownReach.cpp:157-167): u_j's Taylor model :=
     sum_i T[j,i] * TM(x_i) + center, remainder += [-radius, +radius] — written
     in place into the flowpipe state; the linear-relaxation gap is the ONLY
     conservatism introduced, exactly like the original;
  4. one control period of Taylor-model integration with per-cell symbolic
     remainders persisting ACROSS control steps (queue 1000, like
     CrownReach.cpp:65), and Flow*'s capacity rule: once the queue holds 1000
     completed substeps it is RESET at the end of that substep — see
     "Symbolic-remainder queue capacity" below;
  5. end-of-period handoff as a Taylor model (fp_end_of_time semantics:
     tmvPre := tmvPre(t=delta), remainder and history map unchanged) — never a
     box.

Differences from the original (all documented, none loosening):
  * CROWN coefficients stay float64 by default (the C++ side truncates them to
    float32 via Json::Value::asFloat, CrownReach.cpp:149-152). The optional
    `--crown-transport rpc-float32` mode reproduces that legacy transport for
    semantic parity experiments; it is not the target implementation;
  * constraint checks (safe/unsafe/target) run on interval hulls with the
    box-phase rule only (Flow* adds a time-bisection refinement; on these
    benchmarks the box phase decides — any discrepancy would surface as extra
    UNKNOWNs on our side, never as a wrong verdict). Since E1 (2026-07-30) the
    hull is optionally only a PRE-FILTER: see "Refined specification checking"
    below;
  * mode="parity" mirrors Flow* width-for-width; --strict enables the sounder
    tier-S bounds.

Refined specification checking (E1, config key `refine_specs`)
-------------------------------------------------------------
The interval hull evaluates each constraint on per-variable ranges, so it
destroys every correlation between state variables. That is fine for the
axis-aligned box specs of the shipped ARCH-COMP suite and hopeless for COUPLED
ones such as spacecraft docking's `||v|| <= 0.2 + 2n||r||` (it bounds ||v|| and
||r|| independently). This was recorded as LIMIT 6.

With `refine_specs: true` in the benchmark config (or `--refine-specs on`),
every spec site here becomes a two-stage check:

  1. CHEAP SOUND PRE-FILTER — the interval hull, exactly as before. If it
     already decides the lane (provably inside the region, or provably outside
     it), that is a proof and the lane is done. Nothing is recomputed, so
     benchmarks whose verdicts the hull already decides cost the same as before
     and print the same lines;
  2. ESCALATION — for lanes the hull leaves straddling a boundary, the
     constraints are evaluated on the COMPOSED Taylor model (tmvPre o tmv, all
     components over the same symbols, so correlations survive) with Flow*'s
     time-bisection refinement over the step interval and safety.py's frontier
     cap (MAX_FRONTIER = 64 pieces/lane; overflow keeps the coarser enclosure,
     i.e. the lane stays UNDECIDED — never a verified claim).

The two enclosures are intersected (`safety.tighten`), which makes the
escalation monotone: a verdict can only become tighter, never wrong. Default is
OFF (`refine_specs` absent == false) so previously recorded results reproduce
bit-for-bit; the flag exists to run the same benchmark both ways.

Specification TIME WINDOWS (F2, config keys `constraints_safe_from/_until`)
---------------------------------------------------------------------------
CROWN-Reach's config grammar has three constraint groups and each fixes WHEN it
is checked: `constraints_safe` at every step, `constraints_unsafe` over the
accumulated flowpipes, `constraints_target` on `fp_end_of_time` only. Several
real specifications are asserted only over a time WINDOW — the official
ARCH-COMP CartPole property is `for t in [8, 10] s: x1, x3, x4 in [-0.001,
0.001]` on a 10 s horizon — and neither "every step" nor "final step" is that
property. Phase D had to ship CartPole with a target-only spec, which tests a
strictly weaker (necessary-only) condition.

The grammar therefore gains ONE optional pair of scalars:

    constraints_safe: [...]        # unchanged
    constraints_safe_from:  8.0    # optional, default -inf (= horizon start)
    constraints_safe_until: 10.0   # optional, default +inf (= horizon end)

meaning: the conjunction must hold for all t in [from, until], and nothing is
asserted outside it. Absent keys reproduce the old semantics exactly, and the
CPU baseline's yaml-cpp reader ignores keys it does not know, so files stay
readable by both engines (the baseline gained the matching capability at
control-period granularity — see CROWN-Reach/comparison/ANALYSIS_NOTES.md F2).
A window on `constraints_target` is deliberately NOT added: "in the target
region throughout [a, b]" IS a windowed safe set, so one mechanism covers both,
and `constraints_unsafe` keeps its whole-horizon meaning.

Per ODE substep the driver classifies the step's time interval [t_lo, t_hi]
against the window and does one of three things (`window_piece`):
  * NO overlap of positive length -> the safe set is NOT checked. Checking it
    would let an out-of-window violation be reported as a violation of the
    property, which is a spurious falsification, not a conservative one;
  * FULLY inside -> checked exactly as before (same tensors, same numbers);
  * STRADDLING an endpoint -> checked on the INTERSECTED sub-interval, in the
    step's local time coordinate: the hull pre-filter becomes
    `safety.rows_range_over_time` (the whole-step hull with the monomial time
    factor restricted) and the E1 escalation seeds its time-bisection frontier
    at the same sub-interval (`refine_ranges(..., time_domain=...)`). Skipping
    it would drop the in-window part; taking it whole would drag the
    out-of-window part into the verdict.
The intersection endpoints are rounded OUTWARD (next_down/next_up) so a
floating-point subtraction can never shave an in-window sliver off a straddling
step; widening within a step that genuinely overlaps the window is sound in
both directions, because the enclosure then still covers in-window times.

PER-VARIABLE remainder estimation (F1, config key `remainder_estimation`)
-------------------------------------------------------------------------
Flow*'s Picard remainder estimate has always been a `std::vector<Interval>`,
one interval per declared variable (settings.h:126; setter settings.h:153 /
settings.cpp:228; read per-index at Continuous.cpp:963). CrownSettings.cpp's
reader (pre-F1) filled that vector with `numVars` copies of ONE interval, and this driver used
to copy that choice — so the config's scalar was a CROWN-Reach limitation, never
a Flow* one.

Why it mattered: stage (j) of the validated Picard step demands
`total_i(r) subseteq r_i` for EVERY i separately (Continuous.cpp:985-1006). With
a uniform radius eps that becomes, for every i,

    eps >= delta * (b_i + eps * L_i),   L_i = sum_j |J_ij|  (abs Jacobian ROW SUM)

which has a solution ONLY IF `delta * max_i L_i < 1`. One badly scaled row
therefore vetoes every scalar estimate at that step size — for ArtificialPancreas
rows Isc1/Qsto2/Qgut give delta*L_i = 11.7 / 507.6 / 507.6 at delta = 0.2, so no
scalar can ever contract (measured: Phase D swept five decades and every one
failed at step 0). Per-variable radii need only `rho(delta*|J|) < 1`, which for
that plant is 0.37, because the offending entries are COLUMNS of variables whose
own rows are zero (the control u) or bounded (z1, z2).

`parse_rem_est` therefore accepts both yaml forms — `[-a, a]` (uniform, legacy,
bit-identical) and a list of `num_vars` `[-a_i, a_i]` rows (per-variable). No
soundness argument changes: the estimate is only the Picard iteration's STARTING
guess, and stage (j) verifies containment per variable at run time whatever it
was seeded with. A wrong estimate makes a lane FAIL, never pass.

Symbolic-remainder queue capacity (B1, 2026-07-30)
--------------------------------------------------
CrownReach.cpp:65 builds ONE `Symbolic_Remainder(initial_set, 1000)` per cell
and hands the same object to every `dynamics.reach(...)` call, so the Phi_L/J
queue keeps growing across control periods: a run needs
`steps * (step_size / ode_step_size)` SR entries in total, not one control
period's worth. Flow*'s reach loop caps that at `max_size` by RESETTING the
queue at the end of the substep in which `J.size()` reaches it
(Continuous.h:891-894); it never errors. This driver now does the same via
`SymbolicRemainder.reset_if_full()`, whose docstring is the authoritative note
on what a reset keeps, what it drops, and why that is sound.

Before B1 the driver simply never reset, so the 1001st substep hit the
defensive guard in symbolic_remainder.propagate and the run died with
"symbolic-remainder queue overflow: caller missed reset()". That made every
instance with `horizon / ode_step_size > 1000` unrunnable — CartPole at its
official 10 s horizon (2000 substeps), ArtificialPancreas at 720 min, F16-GCAS
— i.e. exactly the long-horizon benchmarks. The shipped ARCH-COMP suite never
tripped it because its longest configs (quad_official, quad_reduced) sit at
EXACTLY 50 x 20 = 1000 substeps, one substep short of the reset.

Usage: python gpu_driver.py <config.yaml> [--device cuda:0] [--strict]
Prints the same protocol lines the C++ tool prints ("Step k", verdicts,
"time cost: <s>") so CROWN-Reach's own result parser applies unchanged.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from flowstar_gpu import interval as iv  # noqa: E402
from flowstar_gpu import polynomial as poly  # noqa: E402
from flowstar_gpu.composition import build_schedule  # noqa: E402
from flowstar_gpu.config import Settings  # noqa: E402
from flowstar_gpu.determinism import enable_determinism  # noqa: E402
from flowstar_gpu.expr_parser import parse  # noqa: E402
from flowstar_gpu.flowpipe import (  # noqa: E402
    ACTIVE,
    advance,
    build_rem_est,
    initial_flowpipe,
)
from flowstar_gpu.monomials import build_tables  # noqa: E402
from flowstar_gpu.ode_compiler import compile_ode  # noqa: E402
from flowstar_gpu.rounding import next_up  # noqa: E402
from flowstar_gpu.safety import (  # noqa: E402
    compile_safe_set,
    eval_expr_interval,
    hull_undecided,
    refine_ranges,
    refine_ranges_sparse,
    rows_range_over_time,
    rows_range_over_time_sparse,
    tighten,
)
from flowstar_gpu.sparse_exec import (  # noqa: E402
    advance_sparse,
    initial_sparse_state,
    prune_state,
)
from flowstar_gpu.nn_coupling import (  # noqa: E402
    affine_parts,
    affine_parts_s,
    build_prelayer,
    inject_affine_domain,
    onehot_ids,
    prelayer_bounds,
    tm_domain_split,
    two_slope_tm,
)
from flowstar_gpu.support import SparseEngine, range_normal_s  # noqa: E402
from flowstar_gpu.symbolic_remainder import make_symbolic_remainder  # noqa: E402
from flowstar_gpu import interval as ivm  # noqa: E402
from flowstar_gpu import support as spm  # noqa: E402
from flowstar_gpu import cuda_kernels, tape_kernels  # noqa: E402
from flowstar_gpu import sparse_exec as sparse_exec_module  # noqa: E402

SR_QUEUE = 1000  # CrownReach.cpp:65 — hardcoded for every benchmark

# Outward hedge on CROWN-concretized quantities (float64 RN outputs; same
# policy as flowstar_gpu.nncs.CROWN_ULP_SLACK).
CROWN_ULP_SLACK = 4


class FlatInputModel(torch.nn.Module):
    """Expose a flat batch input while preserving an ONNX model's layout."""

    def __init__(self, model: torch.nn.Module, input_shape):
        super().__init__()
        self.model = model
        self.sample_shape = tuple(input_shape[1:])

    def forward(self, x):
        return self.model(x.reshape(-1, *self.sample_shape))


def build_raw_net(config: dict, *, experimental: bool = False) -> torch.nn.Module:
    """ONNX -> onnx2pytorch float64 module (crown.py:69-71)."""
    import onnx
    import onnx2pytorch

    onnx_model = onnx.load(str(config["_config_dir"].parent / config["model_dir"]))
    model_ori = onnx2pytorch.ConvertModel(onnx_model, experimental=experimental)
    model_ori.to(torch.float64)
    model_ori.training = False
    return model_ori


def build_crown(
    config: dict,
    device: str,
    relax: str = "same-slope",
    input_layout: str = "native",
):
    """Mirror crown.py:69-73 — ONNX -> onnx2pytorch -> BoundedModule.

    relax: "same-slope" (the reference tool's bound_opts, crown.py:73) or
    "two-slope" (auto_LiRPA defaults: adaptive ReLU, standard tangent-line
    s-shaped — the P2a coupling, docs/RESEARCH_POLY_NN.md §3).
    """
    from auto_LiRPA import BoundedModule

    if input_layout not in ("native", "flat"):
        raise ValueError(f"unknown CROWN input layout: {input_layout}")
    model_ori = build_raw_net(config, experimental=input_layout == "flat")
    input_shape = tuple(config["input_shape"])
    if input_layout == "flat":
        model_ori = FlatInputModel(model_ori, input_shape)
        dummy_shape = (1, math.prod(input_shape[1:]))
    else:
        dummy_shape = (1, *input_shape[1:])
    bound_opts = (
        {"activation_bound_option": "same-slope"}
        if relax == "same-slope" else {}
    )
    if input_layout == "flat":
        # Patches retain the internal NCHW constant roots while the exposed
        # perturbed root is flat. Matrix mode gives every root one compatible
        # [B, spec, input] representation and is stable across reused calls.
        bound_opts["conv_mode"] = "matrix"
    lirpa_model = BoundedModule(
        model_ori,
        torch.zeros(*dummy_shape, dtype=torch.float64),
        device=device,
        bound_opts=bound_opts,
    )
    return lirpa_model


def crown_bounds(lirpa_model, config, input_lb, input_ub, input_layout="native"):
    """Mirror crown.py:65-79 in-process: boxes -> (T, u_min, u_max), float64.

    input_lb/ub [B, nn_in] -> T [B, m, nn_in], u_min/u_max [B, m].
    """
    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    # crown.py:67-68 views the flat arrays into input_shape (conv-form nets
    # like unicycle use NCHW [-1, 1, 1, k]); the perturbation must match.
    if input_layout == "native":
        ishape = tuple(config["input_shape"])
        input_lb = input_lb.reshape(ishape)
        input_ub = input_ub.reshape(ishape)
    elif input_layout != "flat":
        raise ValueError(f"unknown CROWN input layout: {input_layout}")
    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    bt = BoundedTensor(input_lb, ptb)
    required_A = defaultdict(set)
    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
    _, _, A_dict = lirpa_model.compute_bounds(
        x=(bt,), method="CROWN", return_A=True, needed_A_dict=required_A
    )
    A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]
    scale = config["output_scale"]
    offset = config["output_offset"]
    T = (A["lA"] * scale).view(tuple(config["output_T_shape"]))  # [B, m, nn_in]
    u_min = ((A["lbias"] - offset) * scale).view(tuple(config["output_c_shape"]))
    u_max = ((A["ubias"] - offset) * scale).view(tuple(config["output_c_shape"]))
    return T, u_min, u_max


def apply_crown_transport(T, u_min, u_max, mode: str):
    """Apply the declared CROWN-to-plant transport semantics.

    Native CROWN-Reach sends Python float64 values through JSON, then reads
    every T/u_min/u_max scalar with Json::Value::asFloat before assigning it to
    Flow* Real (CrownReach.cpp:149-152).  The round trip below reproduces that
    one behavior-changing transport step without moving the tensor to CPU.
    The production GPU pipeline deliberately keeps float64 end to end.
    """
    if mode == "native-f64":
        return T, u_min, u_max
    if mode == "rpc-float32":
        return tuple(
            value.to(dtype=torch.float32).to(dtype=torch.float64)
            for value in (T, u_min, u_max)
        )
    raise ValueError(f"unknown CROWN transport mode: {mode}")


def crown_bounds_two_slope(
    lirpa_model, config, input_lb, input_ub, input_layout="native"
):
    """Standard (two-slope) CROWN: boxes -> (lA, uA, lbias, ubias), scaled.

    Same call shape as crown_bounds but consuming BOTH slope matrices; the
    lirpa_model must have been built with relax="two-slope" (same-slope's
    lA == uA would make the split pointless, not unsound). The affine u
    transformation u = (NN_raw(x) - offset) * scale is applied to both
    envelopes; a negative scale swaps their roles.

    input_lb/ub [B, nn_in] -> lA/uA [B, m, nn_in], lbias/ubias [B, m].
    """
    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    if input_layout == "native":
        ishape = tuple(config["input_shape"])
        input_lb = input_lb.reshape(ishape)
        input_ub = input_ub.reshape(ishape)
    elif input_layout != "flat":
        raise ValueError(f"unknown CROWN input layout: {input_layout}")
    ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
    bt = BoundedTensor(input_lb, ptb)
    required_A = defaultdict(set)
    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
    _, _, A_dict = lirpa_model.compute_bounds(
        x=(bt,), method="CROWN", return_A=True, needed_A_dict=required_A
    )
    A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]
    scale = float(config["output_scale"])
    offset = float(config["output_offset"])
    t_shape = tuple(config["output_T_shape"])
    c_shape = tuple(config["output_c_shape"])
    lA = (A["lA"] * scale).view(t_shape)  # [B, m, nn_in]
    uA = (A["uA"] * scale).view(t_shape)
    lb = ((A["lbias"] - offset) * scale).view(c_shape)  # [B, m]
    ub = ((A["ubias"] - offset) * scale).view(c_shape)
    if scale < 0:
        lA, uA, lb, ub = uA, lA, ub, lb
    return lA, uA, lb, ub


def parse_rem_est(node, num_vars: int) -> float | tuple[float, ...]:
    """Read the yaml `remainder_estimation` key in either of its two forms.

    Both forms are legal Flow* settings, because Flow*'s own estimate has
    ALWAYS been a `std::vector<Interval>` — one interval per declared variable
    (settings.h:126, setter at settings.h:153 / settings.cpp:228, consumed
    per-index at Continuous.cpp:963). CrownSettings.cpp's reader merely chose,
    before F1, to fill that vector with `numVars` copies of a single interval; the scalar in
    the config was never a Flow* restriction.

    Accepted (a yaml reader distinguishes them by whether entry 0 is a scalar
    or a sequence, so ONE file still feeds both engines):

        remainder_estimation: [-0.01, 0.01]      # UNIFORM (legacy, unchanged)

        remainder_estimation:                    # PER-VARIABLE, num_vars rows
        - [-5.4, 5.4]                            #   in initial_set order
        - [-12.9, 12.9]
        - ...

    Radii must be symmetric in both forms: Settings stores a radius and builds
    [-v, +v], and the CPU baseline is fed the same file, so an asymmetric entry
    would make the two engines run different computations (make_configs.py
    check V14 enforces this at config-generation time; this is the runtime
    backstop).

    Args:
        node: the parsed yaml value — [lo, hi], or a list of num_vars [lo, hi].
        num_vars: declared-variable count n (states + t + controls).

    Returns:
        float radius (uniform form) or tuple of num_vars radii (per-variable
        form, in declaration order) — exactly what Settings accepts.

    Raises:
        ValueError: wrong row count, non-pair rows, or an asymmetric interval.
    """
    rows = list(node)
    per_var = isinstance(rows[0], (list, tuple))

    def radius(pair, where: str) -> float:
        pair = list(pair)
        if len(pair) != 2:
            raise ValueError(
                f"remainder_estimation{where} must be [lo, hi], got {pair}"
            )
        lo, hi = float(pair[0]), float(pair[1])
        # hi IS the radius, so it must be >= 0.  hi == 0 (the degenerate
        # interval [0, 0]) is legal and meaningful: a variable whose RHS is
        # identically 0 — every NNCS control input — has Picard remainder
        # exactly 0, and giving it slack only amplifies into the rows that read
        # it (dIsc1/du = 58.6 on ArtificialPancreas is precisely what made that
        # benchmark unrunnable).
        if hi < 0.0:
            raise ValueError(
                f"remainder_estimation{where} radius must be >= 0, "
                f"got [{lo}, {hi}]"
            )
        if lo + hi != 0.0:
            raise ValueError(
                f"remainder_estimation{where} = [{lo}, {hi}] is not symmetric; "
                f"both engines represent the estimate as a radius, so an "
                f"asymmetric box cannot be honoured identically on each"
            )
        return hi

    if not per_var:
        return radius(rows, "")
    if len(rows) != num_vars:
        raise ValueError(
            f"per-variable remainder_estimation has {len(rows)} rows but "
            f"num_vars = {num_vars}; give one [lo, hi] per declared variable "
            f"in initial_set order (states, then t, then controls)"
        )
    return tuple(radius(r, f"[{i}]") for i, r in enumerate(rows))


def make_cells(config: dict) -> torch.Tensor:
    """Cartesian split grid, replicating CrownSettings.cpp:164-206 (`split_vars`).

    Returns [B, n, 2] boxes in the same cell ordering as split_recursive
    (per-variable sub-intervals in declaration order, outermost variable
    varying slowest).

    NAME-KEYED, and deliberately so.  Each sub-interval is written into the slot
    of the variable it belongs to, because `per_var` is built by walking
    `initial_set` in declaration order.  The CPU reference tool is NOT: its
    `split_recursive` appends the split variables first and the rest afterwards
    (`CrownSettings.cpp:182-201`), then passes that vector to `Flowpipe`
    POSITIONALLY, so unless `split_vars` is a PREFIX of the declared order it
    integrates a permuted initial set.

    Consequence for anyone comparing engines cell by cell: branch indices are
    comparable across engines ONLY when `split_vars` is a declared prefix.  When
    it is not, our cell i and the CPU tool's branch i are boxes of different
    problems -- 28 CPU verdicts were withdrawn for exactly this reason.  See
    `docs/DEFECTS.md` CR-1 and `CROWN-Reach/comparison/SUITE_RESULTS.md` §3.2.
    (The earlier version of this docstring claimed comparability unconditionally;
    that claim was wrong and is corrected here, 2026-08-01.)
    """
    per_var = []
    for entry in config["initial_set"]:
        lo, hi = float(entry["interval"][0]), float(entry["interval"][1])
        k = int(entry.get("splits", 0) or 0)
        if k and entry["name"] in config.get("split_vars", []):
            edges = [lo + (hi - lo) * i / k for i in range(k + 1)]
            per_var.append([[edges[i], edges[i + 1]] for i in range(k)])
        else:
            per_var.append([[lo, hi]])
    cells = [list(combo) for combo in itertools.product(*per_var)]
    return torch.tensor(cells, dtype=torch.float64)


def hull_ranges(fp, tables, step, idx_hi: int) -> torch.Tensor:
    """Interval hulls of TM rows [0, idx_hi): mirrors CrownReach.cpp:71
    (tmvPre.tms[i].intEval over the flowpipe domain). [B, idx_hi, 2]."""
    rng = poly.range_normal(fp.pre_coeffs[:, :idx_hi], tables, step)  # [B, k, 2]
    return iv.add(rng, fp.pre_rem[:, :idx_hi])


def hull_ranges_s(st, eng, idx_hi: int) -> torch.Tensor:
    """Sparse twin of hull_ranges: [B, idx_hi, 2] on the support layout."""
    rng = range_normal_s(st.pre[:, :idx_hi], eng, st.pre_sup)  # [B, k, 2]
    return iv.add(rng, st.pre_rem[:, :idx_hi])


def _inject_core_s(st, T, c, extra_rem, u_ids, nn_in: int) -> None:
    """Sparse injection core: u := T·TM(x) + c with remainder
    (T-weighted input remainders) + extra_rem. The constant monomial is
    local slot 0 by the Support id-0 invariant. T [B, m, nn_in], c [B, m],
    extra_rem [B, m, 2]."""
    new_rows = torch.einsum("bmi,bit->bmt", T, st.pre[:, :nn_in])  # [B, m, Sp]
    new_rows[..., 0] += c
    w_rem = iv.dot_point_iv(T, st.pre_rem[:, :nn_in].unsqueeze(1), dim=-1)
    st.pre[:, u_ids] = new_rows
    st.pre_rem[:, u_ids] = iv.add(w_rem, extra_rem)


def inject_controls_s(st, T, u_min, u_max, u_ids, nn_in: int) -> None:
    """Sparse twin of inject_controls (same-slope box coupling)."""
    c = (u_max + u_min) * 0.5  # [B, m]
    r = next_up((u_max - u_min) * 0.5)
    _inject_core_s(st, T, c, torch.stack((-r, r), dim=-1), u_ids, nn_in)


def inject_controls_two_slope_s(st, A_mid, c_mid, resid, u_ids, nn_in: int) -> None:
    """Sparse P2a injection: midpoint-split two-slope CROWN (nn_coupling
    .two_slope_tm output) — resid is the certified asymmetric residual."""
    _inject_core_s(st, A_mid, c_mid, resid, u_ids, nn_in)


def _ensure_onehot_support_s(st, eng) -> dict:
    """Make st.pre's support contain the constant + one-hot monomial ids
    (remapping st.pre onto the union support if needed — supersets are
    exact, growth is one-way so graph caches re-key at most once).
    Returns the {global id -> local slot} map of the (possibly new) support.
    """
    from flowstar_gpu.support import make_support

    tab = eng.tables
    slots = onehot_ids(tab)
    have = set(st.pre_sup.ids)
    if any(g not in have for g in slots):
        need = make_support(tab.n, tab.k, False, tuple(sorted({0, *slots})))
        sup_u, emb_a, _ = eng.union(st.pre_sup, need)
        newp = st.pre.new_zeros(*st.pre.shape[:-1], sup_u.size)
        newp[..., emb_a] = st.pre
        st.pre, st.pre_sup = newp, sup_u
    return {g: i for i, g in enumerate(st.pre_sup.ids)}


def _affine_rows_s(st, eng, A_mid, c_mid, local) -> torch.Tensor:
    """[B, m, S] coefficient rows of the degree-1-in-r TMs c_mid + A_mid·r
    on st.pre_sup (local: the id->slot map from _ensure_onehot_support_s)."""
    rows = st.pre.new_zeros(A_mid.shape[0], A_mid.shape[1], st.pre_sup.size)
    rows[..., 0] = c_mid
    for j, g in enumerate(onehot_ids(eng.tables)):
        rows[..., local[g]] = A_mid[..., j]
    return rows


def inject_affine_domain_s(st, eng, A_mid, c_mid, resid, u_ids) -> None:
    """Sparse P1a injection: write u rows as degree-1 TMs in r directly.

    A_mid [B, m, n_sym] / c_mid [B, m] / resid [B, m, 2] come from
    nn_coupling.tm_domain_split — no input-remainder dot (the state
    remainders were folded into the prelayer's tail input; adding them
    again would double count).
    """
    local = _ensure_onehot_support_s(st, eng)
    st.pre[:, u_ids] = _affine_rows_s(st, eng, A_mid, c_mid, local)
    st.pre_rem[:, u_ids] = resid


def end_of_time_s(st, eng) -> None:
    """Sparse fp_end_of_time: tmvPre := tmvPre(t=delta) (support shrinks to
    the t-free ids); remainders and tmv untouched."""
    sp_c, sup = spm.evaluate_time_end_s(st.pre, eng, st.pre_sup)
    st.pre = sp_c
    st.pre_sup = sup


def _inject_core(fp, T, c, extra_rem, u_ids, nn_in: int) -> None:
    """Dense injection core: u := T·TM(x) + c with remainder
    (T-weighted input remainders) + extra_rem.

    Polynomial part is a linear combination of the input rows (tier P, like
    Flow*'s Real arithmetic in TaylorModel::operator*/+=); the remainder is
    the point-weighted interval dot of T with the input rows' remainders
    (TM*scalar scales the remainder too) plus the coupling-specific
    extra_rem. T [B, m, nn_in], c [B, m], extra_rem [B, m, 2]."""
    new_rows = torch.einsum("bmi,bit->bmt", T, fp.pre_coeffs[:, :nn_in])
    new_rows[..., 0] += c  # constant slot
    w_rem = iv.dot_point_iv(T, fp.pre_rem[:, :nn_in].unsqueeze(1), dim=-1)  # [B, m, 2]
    new_rem = iv.add(w_rem, extra_rem)

    fp.pre_coeffs[:, u_ids] = new_rows
    fp.pre_rem[:, u_ids] = new_rem


def inject_controls(fp, T, u_min, u_max, u_ids, nn_in: int) -> None:
    """The symbolic CROWN->TM composition (CrownReach.cpp:157-167), batched.

    u_j := sum_i T[j,i]*TM(x_i) + c_j with remainder sum_i |T|-scaled rems
    + [-r_j, r_j]. T [B, m, nn_in]; writes fp.pre_coeffs/pre_rem rows u_ids.
    """
    c = (u_max + u_min) * 0.5  # [B, m] RN midpoint (CrownReach.cpp:102)
    r = (u_max - u_min) * 0.5  # [B, m]
    r = next_up(r)  # sound radius under RN subtraction
    _inject_core(fp, T, c, torch.stack((-r, r), dim=-1), u_ids, nn_in)


def inject_controls_two_slope(fp, A_mid, c_mid, resid, u_ids, nn_in: int) -> None:
    """Dense P2a injection: midpoint-split two-slope CROWN.

    u := A_mid·TM(x) + c_mid with remainder (A_mid-weighted input
    remainders) + resid, where (A_mid, c_mid, resid) come from
    nn_coupling.two_slope_tm — resid certifies NN(x) - (A_mid·x + c_mid)
    over the SAME hull the CROWN call was given, so u is a sound
    degree-1 TM of the controller with an asymmetric interval offset."""
    _inject_core(fp, A_mid, c_mid, resid, u_ids, nn_in)


def end_of_time(fp, tables, step) -> None:
    """fp_end_of_time handoff (Continuous.h:1180-1203): tmvPre := tmvPre(t=d).

    The t-substituted polynomial is spatial; re-embed it into the full basis
    (t-slots zero). Remainders and the history map tmv are untouched.
    """
    sp = poly.evaluate_time_end(fp.pre_coeffs, tables, step)  # [B, n, Ts]
    fp.pre_coeffs.zero_()
    fp.pre_coeffs[..., tables.spatial_index] = sp


def hull_constraint_ranges(exprs, boxes: torch.Tensor) -> torch.Tensor:
    """Interval-hull enclosure of every constraint q_j: [C] x [B, n, 2] ->
    [B, C, 2].

    Mirrors CrownReach.cpp's box-phase spec check (AST interval evaluation on
    the flowpipe's per-variable ranges). Sound but correlation-blind: this is
    the cheap pre-filter that `SpecGroup.ranges` may escalate.
    """
    return torch.stack([eval_expr_interval(e, boxes) for e in exprs], dim=1)


def decide_constraints(q: torch.Tensor):
    """Membership verdict per lane from constraint enclosures q [B, C, 2] ->
    (all_sup_le0 [B], any_inf_gt0 [B]) on the CPU.

    all_sup_le0: every q_j's sup <= 0 — the flowpipe is entirely INSIDE the
    region {all q_j <= 0}; any_inf_gt0: some q_j's inf > 0 — every point of the
    flowpipe violates that constraint, so the flowpipe is entirely OUTSIDE.
    Both are proofs because each row of q over-approximates its constraint.
    """
    return (q[..., 1] <= 0).all(dim=1).cpu(), (q[..., 0] > 0).any(dim=1).cpu()


def window_piece(t_lo: float, t_hi: float, t_from: float, t_until: float):
    """Which part of the step [t_lo, t_hi] a spec window [t_from, t_until] asks
    about, in the step's LOCAL time coordinate (F2).

    Returns None when the two intervals do not overlap in POSITIVE length — the
    group must then not be checked on this step at all. Otherwise returns
    (a, b) with 0 <= a <= b <= t_hi - t_lo, the local sub-interval the check
    must cover; (0.0, t_hi - t_lo) means "the whole step", i.e. the caller can
    reuse its ordinary whole-step enclosures unchanged.

    Positive-length overlap (`t_lo < t_until and t_hi > t_from`) rather than
    closed-interval intersection, so a step that merely TOUCHES the window at
    one endpoint is not checked whole for the sake of a measure-zero instant:
    that instant is the neighbouring step's own endpoint, and Flow* flowpipes
    are closed on their time domain, so it is still covered. With window
    endpoints aligned to the step grid (which make_configs.py check V23
    enforces) the retained steps tile [t_from, t_until] exactly.

    Endpoints are rounded OUTWARD (one ulp down / up, then clamped into
    [0, delta]) because `t_from - t_lo` is a floating-point subtraction of two
    numbers of very different magnitude: an inward error of one ulp would leave
    an ulp-wide sliver of the window unchecked, while an outward error only
    drags in a sliver of a step that already overlaps the window — sound in
    both directions, since the enclosure then still covers in-window times.
    """
    if not (t_lo < t_until and t_hi > t_from):
        return None
    delta = t_hi - t_lo
    a = 0.0 if t_from <= t_lo else min(max(math.nextafter(t_from - t_lo, -math.inf), 0.0), delta)
    b = delta if t_until >= t_hi else min(max(math.nextafter(t_until - t_lo, math.inf), 0.0), delta)
    return (a, b)


class SpecGroup:
    """One constraint group (safe / unsafe / target) and how to enclose it.

    Holds the parsed ASTs (hull evaluation, always) and — when `refine` is on —
    the compiled constraint tapes needed for the composed-TM refinement
    (`safety.compile_safe_set` with bounds 0, since every CROWN-Reach
    constraint is `q_j <= 0`).

    `refiner` is set by main() to a closure over the LIVE flowpipe state
    (bound at call time, so it always sees the current substep's flowpipe);
    leaving it None keeps the historical hull-only behaviour.

    `window` (F2) is the (from, until) absolute-time interval the group is
    asserted over, defaulting to (-inf, +inf) = the whole horizon. The group
    itself does not walk time — main() classifies each step with
    `window_piece` and passes the resulting sub-interval down as
    `time_domain`.
    """

    def __init__(self, texts, var_names, order: int, refine: bool,
                 window=(-float("inf"), float("inf"))):
        self.texts = list(texts or [])
        self.exprs = [parse(c, var_names) for c in self.texts]
        # bounds are all zero: CROWN-Reach writes constraints as q_j <= 0.
        self.safe = (
            compile_safe_set(self.texts, [0.0] * len(self.texts), var_names, order)
            if refine and self.texts
            else None
        )
        self.refiner = None
        self.escalations = 0  # lanes escalated (reported at the end; diagnostic)
        self.window = (float(window[0]), float(window[1]))
        self.checked_steps = 0  # steps on which this group WAS checked (F2 diag)
        self.partial_steps = 0  # of those, how many straddled a window endpoint

    def __bool__(self) -> bool:
        return bool(self.exprs)

    def windowed(self) -> bool:
        """Whether a time window was configured at all (F2 diagnostics)."""
        return self.window != (-float("inf"), float("inf"))

    def ranges(self, boxes: torch.Tensor, active: torch.Tensor,
               time_domain: torch.Tensor | None = None) -> torch.Tensor:
        """Constraint enclosures for the current flowpipe: -> [B, C, 2].

        boxes [B, n, 2] the flowpipe's interval hull; active [B] bool the lanes
        whose verdict still matters (broken lanes are excluded so a dead lane
        can never trigger refinement work).

        Stage 1 is always the hull. Stage 2 runs only for lanes the hull leaves
        undecided, and only if refinement is configured; the results are
        INTERSECTED, so the returned enclosure is never wider than the hull's
        (verdicts tighten or stay put — they cannot flip).

        time_domain [B, 2] | None (F2): a sub-interval of the step's local time
        to restrict the ESCALATION to. It must describe the same sub-interval
        `boxes` was computed over, or the intersection would mix enclosures of
        two different quantities; main() derives both from one `window_piece`
        result.
        """
        q = hull_constraint_ranges(self.exprs, boxes)  # [B, C, 2]
        if self.safe is None or self.refiner is None:
            return q
        bounds = self.safe.bounds.to(q.device)  # [C] all zeros
        lanes = hull_undecided(q, bounds) & active.to(q.device)  # [B]
        if not bool(lanes.any()):
            return q  # the hull decided every live lane: refinement is moot
        self.escalations += int(lanes.sum())
        return tighten(q, self.refiner(self.safe, lanes, time_domain))  # [B, C, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--strict", action="store_true",
                    help="tier-S bounds (sounder than Flow*); default parity")
    ap.add_argument("--engine", choices=("dense", "sparse"), default="sparse",
                    help="plant representation (union-support sparse is the "
                    "M10 default; dense is the M9 baseline path)")
    ap.add_argument("--order", type=int, default=None,
                    help="override the config's TM order (tightness/speed "
                    "tradeoff studies; lower = faster, looser)")
    ap.add_argument("--splits", default=None,
                    help="override the initial-set split grid, e.g. "
                    "'x1:8,x2:8' (unlisted vars unsplit); '' clears splits")
    ap.add_argument("--print-final-hull", action="store_true",
                    help="print the aggregated final-time hull per variable "
                    "(tightness metric for cross-tool comparisons)")
    ap.add_argument("--refine-specs", choices=("auto", "on", "off"), default="auto",
                    help="composed-Taylor-model specification checking (E1, the "
                    "LIMIT-6 fix for coupled/non-axis-aligned constraints): "
                    "'auto' honours the config key `refine_specs` (absent = "
                    "off, i.e. the historical hull-only behaviour)")
    ap.add_argument("--crown-relax", choices=("same-slope", "two-slope"),
                    default="same-slope",
                    help="NN relaxation + injection coupling. 'same-slope' is "
                    "the reference tool's (crown.py:73): lA == uA, box "
                    "concretized offsets. 'two-slope' (P2a, "
                    "docs/RESEARCH_POLY_NN.md §3) consumes standard CROWN "
                    "lA != uA as an interval-coefficient degree-1 TM via "
                    "midpoint split — tighter relaxations, asymmetric "
                    "residual, and the prerequisite for alpha-CROWN.")
    ap.add_argument("--crown-domain", choices=("box", "tm-affine", "hybrid"),
                    default="hybrid",
                    help="NN input abstraction. 'box' interval-hulls the "
                    "state TMs (the reference coupling; loss point L1). "
                    "'tm-affine' (P1a, docs/RESEARCH_POLY_NN.md §3) bounds "
                    "NN((r·A)+b+e) over the flowpipe's own domain r via an "
                    "exact per-cell affine prelayer, with the high-order "
                    "tail + remainders as the bounded input e; the output "
                    "is a degree-1 TM in r injected with zero wrapping. "
                    "'hybrid' (THE DEFAULT since the 2026-08-05 Workstream-A "
                    "gate: zero verdict changes, tighter-or-equal hulls on "
                    "every gate benchmark) computes BOTH the same-slope box "
                    "coupling and the tm-affine coupling each control step "
                    "(2 CROWN calls, ~free per the A0 budget) and injects, "
                    "per (cell, output), whichever certified remainder is "
                    "narrower — sound because each candidate is an "
                    "independently certified enclosure and u rows are "
                    "written independently. Use '--crown-domain box' to "
                    "reproduce the pre-campaign records (M10 sweeps, "
                    "SUITE_RESULTS). tm-affine/hybrid imply two-slope "
                    "consumption of the prelayer output (--crown-relax "
                    "only affects the box candidate).")
    ap.add_argument(
        "--crown-transport", choices=("native-f64", "rpc-float32"),
        default="native-f64",
        help="transport from CROWN bounds into the plant. 'native-f64' is "
             "the target in-process path; 'rpc-float32' reproduces the "
             "legacy C++ Json::Value::asFloat truncation for baseline parity "
             "and is valid only with box/same-slope/crown")
    ap.add_argument(
        "--crown-input-layout", choices=("native", "flat"), default="native",
        help="auto_LiRPA input graph layout. 'native' reproduces the ONNX "
             "NCHW wrapper. 'flat' exposes [B, nn_in] and reshapes inside a "
             "thin module, avoiding repeated-call batch-state corruption in "
             "the current onnx2pytorch/auto_LiRPA stack; bounds are otherwise "
             "semantically identical")
    ap.add_argument("--nn-mode", choices=("crown", "tm", "hybrid3"),
                    default="crown",
                    help="controller bounding method. 'crown' = linear "
                    "enclosures via auto_LiRPA (see --crown-domain). 'tm' "
                    "(P3, docs/RESEARCH_POLY_NN.md §3) propagates the state "
                    "TMs THROUGH the network with the engine's validated "
                    "elementary series (tanh/sigmoid only — ReLU nets are "
                    "refused at startup): u becomes a FULL order-k TM in "
                    "the flowpipe's domain symbols, keeping the network's "
                    "curvature that every linear coupling pays as interval. "
                    "A lane whose series guard fires is marked broken "
                    "(printed like a contraction failure). 'hybrid3' "
                    "(smooth nets only) extends --crown-domain hybrid's "
                    "per-row selection with the TM-through candidate: per "
                    "(cell, output) the narrowest of {same-slope box, "
                    "tm-affine prelayer, TM-through} — a lane where the TM "
                    "candidate fails is NOT broken, its rows just fall "
                    "back to the CROWN candidates.")
    ap.add_argument("--crown-alpha-objective",
                    choices=("stock", "resid", "default"), default="stock",
                    help="S2-3: which objective --crown-alpha optimizes. "
                    "'stock' = auto_LiRPA CROWN-Optimized (the Sprint-1 A4 "
                    "path, measured counterproductive); 'resid' = the "
                    "custom loop minimizing the consumed midpoint-split "
                    "width Σ|uA−lA|·rad + (ubias−lbias) (nn_alpha); "
                    "'default' = the custom loop with the stock bounds "
                    "objective (ablation control).")
    ap.add_argument("--crown-clip", choices=("off", "on"), default="off",
                    help="S2-5 stage 1: condition the prelayer candidate's "
                    "CROWN input on the safe set (nn_clip halfspaces in "
                    "r-space; soundness: docs/CLIP_CONFINEMENT_MEMO.md). "
                    "Effective in the hybrid/hybrid3 prelayer candidate; "
                    "cells whose conditioned bounds are nonfinite fall "
                    "back to the unconditioned candidates per row.")
    ap.add_argument("--crown-select", choices=("rem", "range"), default="rem",
                    help="hybrid/hybrid3 per-row candidate selection. 'rem' "
                    "(the Sprint-1/2 behaviour) picks the narrowest CERTIFIED "
                    "REMAINDER — a greedy proxy that ignores the polynomial "
                    "part and can pick rows that wrap worse downstream "
                    "(dp_less; measured on the hard suite up to "
                    "VERIFIED->UNKNOWN, cstr_runaway_official_B2048 hybrid3). "
                    "'range' picks the narrowest TOTAL range proxy "
                    "2*sum|nonconstant coeffs| + remainder width. Either "
                    "pick is sound: every candidate is an independently "
                    "certified enclosure of the same u row.")
    ap.add_argument("--crown-alpha", type=int, default=0, metavar="ITERS",
                    help="alpha-CROWN iterations for the tm-affine/hybrid "
                    "prelayer bounds (0 = plain CROWN, the default). "
                    "Per-benchmark opt-in: 60-320x the per-call cost buys "
                    "optimized slopes; the A0 record prices it. Applies "
                    "only to the prelayer candidate (same-slope has no "
                    "optimizable parameters).")
    ap.add_argument("--metrics-json", default=None,
                    help="write per-control-step tightness metrics (NN-input "
                    "hull widths, CROWN output-box widths, injected u-row "
                    "remainder widths) plus the final hull to this JSON file. "
                    "Pure instrumentation: stdout is byte-identical with or "
                    "without it (the output wording is parser-load-bearing, "
                    "GOTCHAS E1-4). The poly-CROWN campaign's A/B metric.")
    ap.add_argument("--profile-components", action="store_true",
                    help="record CUDA-event controller/dynamics attribution in "
                    "--metrics-json; diagnostic instrumentation, not a formal "
                    "core timing arm")
    ap.add_argument("--nvtx-phases", action="store_true",
                    help="emit controller and dynamics/certificate NVTX ranges "
                    "for external profiling; pure diagnostic instrumentation")
    ap.add_argument("--nvtx-detail", action="store_true",
                    help="emit nested sparse-advance NVTX ranges; requires no "
                    "algorithm or dispatch change")
    args = ap.parse_args()
    sparse_exec_module.NVTX_DETAIL = bool(args.nvtx_detail)

    if args.crown_transport == "rpc-float32" and not (
        args.crown_domain == "box"
        and args.crown_relax == "same-slope"
        and args.nn_mode == "crown"
    ):
        ap.error("--crown-transport rpc-float32 requires "
                 "--crown-domain box --crown-relax same-slope --nn-mode crown")

    cfg_path = Path(args.config).resolve()
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    config["_config_dir"] = cfg_path.parent  # configs/; model paths are relative to src/ (its parent)

    warnings.filterwarnings("ignore")
    if args.order is not None:
        config["ode_order"] = args.order
    if args.splits is not None:
        # Rewrite the split grid: "x1:8,x2:8" (empty string = no splits).
        grid = {}
        if args.splits.strip():
            for part in args.splits.split(","):
                k, v = part.split(":")
                grid[k.strip()] = int(v)
        for e in config["initial_set"]:
            e.pop("splits", None)
            if e["name"] in grid:
                e["splits"] = grid[e["name"]]
        config["split_vars"] = list(grid)
    enable_determinism()
    torch.set_default_dtype(torch.float64)  # crown.py:58 — auto_LiRPA internals
    torch.backends.cuda.matmul.allow_tf32 = False  # crown.py:59-60
    torch.backends.cudnn.allow_tf32 = False

    num_vars = int(config["num_vars"])
    nn_in = int(config["num_nn_input"])
    nn_out = int(config["num_nn_output"])
    n = num_vars  # states + t + u's, all first-class variables (CrownSettings)
    var_names = [e["name"] for e in config["initial_set"]]
    u_ids = list(range(num_vars - nn_out, num_vars))  # CrownSettings.cpp:60-64

    settings = Settings(
        step=float(config["ode_step_size"]),
        order=int(config["ode_order"]),
        cutoff=float(config["cut_off_threshold"]),
        remainder_estimation=parse_rem_est(config["remainder_estimation"], n),
        mode="strict" if args.strict else "parity",
        device=args.device,
    )
    control_period = float(config["step_size"])
    steps = int(config["steps"])
    substeps = round(control_period / settings.step)

    # --- setup (outside the timed window, mirroring CrownReach.cpp) --------
    lirpa_model = prelayer_model = nn_layers = None
    crown_method = ("CROWN-Optimized"
                    if args.crown_alpha > 0
                    and args.crown_alpha_objective == "stock" else "CROWN")
    alpha_kwargs = (
        {"alpha_iters": args.crown_alpha,
         "alpha_objective": args.crown_alpha_objective}
        if args.crown_alpha > 0 and args.crown_alpha_objective != "stock"
        else {})
    if args.nn_mode in ("tm", "hybrid3"):
        from flowstar_gpu.nn_tm import layers_from_onnx, nn_tm_bounds
        from flowstar_gpu.nn_tm_sparse import nn_tm_bounds_s

        # Same path resolution as build_crown (absolute model_dir passes
        # through). A net the MLP parser cannot express DEGRADES hybrid3
        # to the plain 2-way hybrid instead of erroring — the
        # wholesale-ReLU decision assumed parseability, not universality.
        # (Conv-as-MLP and Sub-normalization graphs parse since 2026-08-06;
        # what remains out of scope is genuinely non-affine structure.)
        try:
            nn_layers = layers_from_onnx(
                str(config["_config_dir"].parent / config["model_dir"]))
        except ValueError as exc:
            if args.nn_mode == "hybrid3":
                print(f"# hybrid3: TM candidate unavailable ({exc}); "
                      f"running 2-way hybrid", flush=True)
                args.nn_mode = "crown"
                nn_layers = None
            else:
                raise
        out_scale = float(config["output_scale"])
        out_offset = float(config["output_offset"])
    if args.nn_mode != "tm" and args.crown_domain in ("tm-affine", "hybrid"):
        prelayer_model = build_prelayer(
            build_raw_net(config), num_vars, nn_in,
            tuple(config["input_shape"]), args.device)
        if args.crown_alpha > 0:
            prelayer_model.set_bound_opts(
                {"optimize_bound_args": {"iteration": args.crown_alpha}})
        out_scale = float(config["output_scale"])
        out_offset = float(config["output_offset"])
    if args.nn_mode != "tm" and args.crown_domain in ("box", "hybrid"):
        lirpa_model = build_crown(
            config, args.device, relax=args.crown_relax,
            input_layout=args.crown_input_layout)
    cells = make_cells(config)  # [B, n, 2]
    B = cells.shape[0]

    code = compile_ode(config["dynamics_expressions"], var_names,
                       order=settings.order - 1)
    tables = build_tables(n, settings.order).to(args.device)
    step = poly.build_step_tables(tables, settings.step)
    sched = build_schedule(n, settings.order, args.device)
    sparse = args.engine == "sparse"
    if sparse:
        eng = SparseEngine(tables, step, args.device)
        st = initial_sparse_state(cells.to(args.device), eng, sched)
    else:
        fp = initial_flowpipe(cells.to(args.device), tables)
    sr = make_symbolic_remainder(B, n, SR_QUEUE, args.device)
    rem_est = build_rem_est(settings, n, B, args.device)  # [B, n, 2]

    # --- specification groups ----------------------------------------------
    # `refine_specs` (config key; --refine-specs on/off overrides) turns on the
    # composed-TM escalation described in the module docstring. Default OFF so
    # every previously recorded benchmark result reproduces unchanged.
    refine_specs = (
        bool(config.get("refine_specs", False)) if args.refine_specs == "auto"
        else args.refine_specs == "on"
    )
    # F2: the OPTIONAL time window on `constraints_safe`. Absent keys give
    # (-inf, +inf), i.e. the historical "checked at every step" semantics.
    safe_window = (
        float(config.get("constraints_safe_from", -math.inf)),
        float(config.get("constraints_safe_until", math.inf)),
    )
    if safe_window[0] >= safe_window[1]:
        raise ValueError(
            f"constraints_safe_from/_until = {safe_window} is empty; the window "
            f"must satisfy from < until (an empty window asserts nothing, which "
            f"would make every run trivially VERIFIED)"
        )
    if config.get("constraints_safe") and not (
        safe_window[0] < steps * control_period and safe_window[1] > 0.0
    ):
        # A window disjoint from [0, horizon] would leave `constraints_safe`
        # unchecked on EVERY step, and an unchecked safe set prints nothing,
        # which both result parsers read as VERIFIED. Refuse it here rather
        # than emit a vacuous proof.
        raise ValueError(
            f"constraints_safe window {safe_window} does not overlap the horizon "
            f"[0, {steps * control_period}]: no step would ever be checked, and "
            f"a silent run parses as VERIFIED"
        )
    safe_spec = SpecGroup(config.get("constraints_safe") or [], var_names,
                          settings.order, refine_specs, window=safe_window)
    unsafe_spec = SpecGroup(config.get("constraints_unsafe") or [], var_names,
                            settings.order, refine_specs)
    target_spec = SpecGroup(config.get("constraints_target") or [], var_names,
                            settings.order, refine_specs)

    def refine(safe, lanes, time_domain=None):
        """Composed-TM enclosures of `safe`'s constraints for `lanes` [B] bool.

        Closes over the LIVE plant state (`st` sparse / `fp` dense — rebound
        every substep, read here at call time), so a SpecGroup never holds a
        stale flowpipe. `time_domain` [B, 2] | None restricts the enclosure to a
        sub-interval of the step's local time (F2). -> [B, C, 2] ([-inf, +inf]
        where not refined).
        """
        if sparse:
            return refine_ranges_sparse(st, eng, safe, settings, lanes,
                                        time_domain=time_domain)
        return refine_ranges(fp.pre_coeffs, fp.pre_rem, fp.tmv_coeffs, fp.tmv_rem,
                             safe, tables, step, settings, lanes,
                             time_domain=time_domain)

    if refine_specs:
        for grp in (safe_spec, unsafe_spec, target_spec):
            grp.refiner = refine

    # --- tightness metrics (poly-CROWN campaign A/B instrumentation) -------
    # Collected only under --metrics-json; never printed (stdout wording is
    # parser-load-bearing). Widths are exact f64 subtractions of quantities
    # the run computes anyway; the per-step cost is a handful of [B, m]
    # reductions.
    metrics = None
    if args.metrics_json:
        metrics = {
            "config": str(cfg_path),
            "engine": args.engine,
            "coupling": (args.nn_mode if args.nn_mode in ("tm", "hybrid3")
                         else args.crown_domain
                         if args.crown_domain in ("tm-affine", "hybrid")
                         else args.crown_relax),
            "crown_transport": args.crown_transport,
            "crown_input_layout": args.crown_input_layout,
            "segment_cuda_kernel_available": cuda_kernels.available(),
            "tape_cuda_kernel_available": tape_kernels.available(),
            "valid_cuda_kernel_available": tape_kernels.valid_available(),
            "alpha_iters": args.crown_alpha,
            "B": B,
            "order": settings.order,
            "steps": steps,
            "substeps": substeps,
            "ctrl_steps": [],
        }

    # Persistent flags mirroring the C++ loop's outcome bookkeeping.
    broken = torch.zeros(B, dtype=torch.bool)  # Flow*-terminated lanes
    safe_unknown = False
    safe_unsafe = False
    unsafe_hit_inside = False  # some hull entirely inside the unsafe set
    unsafe_all_outside = True  # every hull provably outside the unsafe set

    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
        if metrics is not None:
            torch.cuda.reset_peak_memory_stats(args.device)
    t0 = time.perf_counter()  # CrownReach.cpp:48 equivalent
    component_events = []

    for k in range(steps):
        print(f"Step {k}", flush=True)
        if args.nvtx_phases and torch.device(args.device).type == "cuda":
            torch.cuda.nvtx.range_push("gpu_flowfull/controller")
        if args.profile_components and torch.device(args.device).type == "cuda":
            ctrl_start = torch.cuda.Event(enable_timing=True)
            ctrl_end = torch.cuda.Event(enable_timing=True)
            dyn_start = torch.cuda.Event(enable_timing=True)
            dyn_end = torch.cuda.Event(enable_timing=True)
            ctrl_start.record()

        # (1)+(2) hull -> batched CROWN (in-process; no RPC, no float32).
        if sparse:
            hull = hull_ranges_s(st, eng, nn_in)  # [B, nn_in, 2]
        else:
            hull = hull_ranges(fp, tables, step, nn_in)  # [B, nn_in, 2]
        # (3) symbolic injection into the u rows.
        if args.nn_mode == "tm":
            # P3: u = NN(x) propagated in TM arithmetic — a FULL order-k
            # polynomial in r plus the genuine composition remainder. Sparse
            # mode runs the union-support path directly (S2-1: no dense
            # expansion; the activation algebra works on the measured
            # supports — 30-900x less pair traffic at CartPole/QUAD scale).
            u_sup = None
            if sparse:
                u_c, u_sup, u_r, ok_tm = nn_tm_bounds_s(
                    nn_layers, st.pre[:, :nn_in], st.pre_sup,
                    st.pre_rem[:, :nn_in], settings.order, eng,
                    settings.cutoff, net_key="ctrl")
            else:
                u_c, u_r, ok_tm = nn_tm_bounds(
                    nn_layers, fp.pre_coeffs[:, :nn_in],
                    fp.pre_rem[:, :nn_in], settings.order, tables, step,
                    settings.cutoff)
            # u = (raw − offset)·scale: coefficients tier-P RN (like the
            # einsum path), remainder scaled outward.
            if out_scale != 1.0 or out_offset != 0.0:
                u_c = u_c * out_scale
                u_c[..., 0] -= out_offset * out_scale
                u_r = iv.mul_point(
                    u_r, torch.full_like(u_r[..., 0], out_scale))
            newly = (~ok_tm.cpu()) & (~broken)
            if bool(newly.any()):
                broken |= newly
                for bidx in newly.nonzero(as_tuple=True)[0].tolist():
                    print("Flow* terminated.")
                    print(f"Broken branch: {bidx}")
                if bool(broken.all()):
                    break
            if sparse:
                # u_c lives on u_sup (S2-1): union it into the state's
                # support (one-way growth), embed both sides onto the
                # union, and write.
                sup_new, emb_state, emb_u = eng.union(st.pre_sup, u_sup)
                if sup_new != st.pre_sup:
                    newp = st.pre.new_zeros(*st.pre.shape[:-1], sup_new.size)
                    newp[..., emb_state] = st.pre
                    st.pre, st.pre_sup = newp, sup_new
                u_full = u_c.new_zeros(B, u_c.shape[1], sup_new.size)
                u_full[..., emb_u] = u_c
                st.pre[:, u_ids] = u_full
                st.pre_rem[:, u_ids] = u_r
            else:
                fp.pre_coeffs[:, u_ids] = u_c
                fp.pre_rem[:, u_ids] = u_r
            w_u_metric = u_r[..., 1] - u_r[..., 0]  # [B, m]
        elif args.crown_domain == "hybrid":
            # Candidate SS: the same-slope box coupling, built but not yet
            # written. Candidate TA: the tm-affine prelayer coupling. Inject
            # per (cell, output) whichever certified remainder is narrower —
            # each candidate is an independently sound enclosure of the SAME
            # u_j = NN(x)_j, and rows are written independently, so any
            # per-row mix is sound. In sparse mode the support union happens
            # FIRST so both candidates share one coefficient layout.
            T, u_min, u_max = crown_bounds(
                lirpa_model, config,
                hull[..., 0].contiguous(), hull[..., 1].contiguous(),
                input_layout=args.crown_input_layout)
            c1 = (u_max + u_min) * 0.5
            r1 = next_up((u_max - u_min) * 0.5)
            if sparse:
                local = _ensure_onehot_support_s(st, eng)
                x_rows, x_rems = st.pre[:, :nn_in], st.pre_rem[:, :nn_in]
            else:
                x_rows = fp.pre_coeffs[:, :nn_in]
                x_rems = fp.pre_rem[:, :nn_in]
            rows_ss = torch.einsum("bmi,bit->bmt", T, x_rows)
            rows_ss[..., 0] += c1
            rem_ss = iv.add(
                iv.dot_point_iv(T, x_rems.unsqueeze(1), dim=-1),
                torch.stack((-r1, r1), dim=-1))  # [B, m, 2]

            if sparse:
                Aaf, baf, tail = affine_parts_s(
                    st.pre, st.pre_rem, eng, st.pre_sup, nn_in)
            else:
                Aaf, baf, tail = affine_parts(
                    fp.pre_coeffs, fp.pre_rem, tables, step, nn_in)
            r_constraints = None
            # Windowed specs (F2): conditioning on "state still safe" is
            # licensed by the confinement induction ONLY at times the safe
            # set is asserted AND already checked — the substep enclosures
            # are closed in time, so x(t_k) is covered once t_k lies inside
            # [from, until] (window_piece checks a step that merely touches
            # the window). Before the window opens, a trajectory may
            # legitimately be outside the safe box (CartPole's official
            # initial set is), and clipping there would be UNSOUND.
            t_k = k * control_period
            clip_licensed = safe_spec.window[0] <= t_k <= safe_spec.window[1]
            if args.crown_clip == "on" and safe_spec and clip_licensed:
                from flowstar_gpu.nn_clip import safe_halfspaces

                A_hs, b_hs, _keep = safe_halfspaces(
                    safe_spec.exprs, nn_in, Aaf, baf, tail,
                    hull[:, :nn_in])
                r_constraints = (A_hs, b_hs)
            lA, uA, lbias, ubias = prelayer_bounds(
                prelayer_model, Aaf, baf, tail, out_scale, out_offset,
                method=crown_method, r_constraints=r_constraints,
                **alpha_kwargs)
            A_mid, c_mid, resid = tm_domain_split(lA, uA, lbias, ubias)
            # conditioned bounds can be nonfinite on already-deciding cells
            # (memo: infeasible = route to the safety verdict, which the
            # per-substep checks do) — exclude those rows from this
            # candidate rather than poisoning the flowpipe.
            bad_ta = ~torch.isfinite(resid).all(dim=-1)  # [B, m]
            if bool(bad_ta.any()):
                resid = torch.where(
                    bad_ta.unsqueeze(-1),
                    torch.tensor([-torch.inf, torch.inf],
                                 device=resid.device).expand_as(resid),
                    resid)
                A_mid = torch.where(bad_ta.unsqueeze(-1),
                                    torch.zeros_like(A_mid), A_mid)
                c_mid = torch.where(bad_ta, torch.zeros_like(c_mid), c_mid)
            if sparse:
                rows_ta = _affine_rows_s(st, eng, A_mid, c_mid, local)
            else:
                rows_ta = torch.zeros_like(rows_ss)
                rows_ta[..., 0] = c_mid
                for _j, _g in enumerate(onehot_ids(tables)):
                    rows_ta[..., _g] = A_mid[..., _j]

            cand_rows = [rows_ss, rows_ta]
            cand_rems = [rem_ss, resid]
            if args.nn_mode == "hybrid3":
                # third candidate: TM-through-NN (P3). A row where the TM
                # path failed is excluded by an infinite width, never a
                # broken lane — the CROWN candidates always exist.
                u_sup_h = None
                if sparse:
                    u_c, u_sup_h, u_r, ok_tm = nn_tm_bounds_s(
                        nn_layers, st.pre[:, :nn_in], st.pre_sup,
                        st.pre_rem[:, :nn_in], settings.order, eng,
                        settings.cutoff, net_key="ctrl")
                else:
                    u_c, u_r, ok_tm = nn_tm_bounds(
                        nn_layers, fp.pre_coeffs[:, :nn_in],
                        fp.pre_rem[:, :nn_in], settings.order, tables,
                        step, settings.cutoff)
                if out_scale != 1.0 or out_offset != 0.0:
                    u_c = u_c * out_scale
                    u_c[..., 0] -= out_offset * out_scale
                    u_r = iv.mul_point(
                        u_r, torch.full_like(u_r[..., 0], out_scale))
                u_r = torch.where(
                    ok_tm.view(-1, 1, 1).expand_as(u_r), u_r,
                    torch.full_like(u_r, torch.inf) *
                    torch.tensor([-1.0, 1.0], device=u_r.device))
                if sparse:
                    # the TM candidate lives on u_sup_h (S2-1): union it
                    # into the state's support (one-way growth), remap the
                    # ALREADY-BUILT candidates and the state, embed the TM
                    # rows onto the union.
                    sup_new, emb_state, emb_u = eng.union(st.pre_sup, u_sup_h)
                    if sup_new != st.pre_sup:
                        newp = st.pre.new_zeros(*st.pre.shape[:-1],
                                                sup_new.size)
                        newp[..., emb_state] = st.pre
                        st.pre, st.pre_sup = newp, sup_new
                        for ci in range(len(cand_rows)):
                            newr = cand_rows[ci].new_zeros(
                                *cand_rows[ci].shape[:-1], sup_new.size)
                            newr[..., emb_state] = cand_rows[ci]
                            cand_rows[ci] = newr
                    u_full = u_c.new_zeros(B, u_c.shape[1], sup_new.size)
                    u_full[..., emb_u] = u_c
                    cand_rows.append(u_full)
                else:
                    cand_rows.append(u_c)
                cand_rems.append(u_r)

            if args.crown_select == "range":
                # total-range proxy: polynomial span bound over the domain
                # (slot 0 is the constant — id-0 invariant — so the
                # nonconstant coefficient mass bounds the span by 2*sum|c|)
                # plus the certified remainder width. Comparator only: the
                # injection below is sound for ANY pick.
                widths = torch.stack(
                    [2.0 * rw[..., 1:].abs().sum(dim=-1)
                     + (re[..., 1] - re[..., 0])
                     for rw, re in zip(cand_rows, cand_rems)])  # [C, B, m]
            else:
                widths = torch.stack(
                    [r[..., 1] - r[..., 0] for r in cand_rems])  # [C, B, m]
            pick = widths.argmin(dim=0)  # [B, m]
            rows = cand_rows[0]
            rem_u = cand_rems[0]
            for ci in range(1, len(cand_rows)):
                is_ci = (pick == ci).unsqueeze(-1)
                rows = torch.where(is_ci, cand_rows[ci], rows)
                rem_u = torch.where(is_ci, cand_rems[ci], rem_u)
            if sparse:
                st.pre[:, u_ids] = rows
                st.pre_rem[:, u_ids] = rem_u
            else:
                fp.pre_coeffs[:, u_ids] = rows
                fp.pre_rem[:, u_ids] = rem_u
            w_u_metric = rem_u[..., 1] - rem_u[..., 0]  # [B, m]
            hybrid_ta_frac = float((pick == 1).double().mean())
            hybrid_tm_frac = (float((pick == 2).double().mean())
                              if len(cand_rows) > 2 else None)
        elif args.crown_domain == "tm-affine":
            # P1a: bound over the flowpipe's own domain r. hull is still
            # computed above for the metrics record (the L1 proxy) and to
            # keep the loop structure identical, but CROWN never sees it.
            if sparse:
                Aaf, baf, tail = affine_parts_s(
                    st.pre, st.pre_rem, eng, st.pre_sup, nn_in)
            else:
                Aaf, baf, tail = affine_parts(
                    fp.pre_coeffs, fp.pre_rem, tables, step, nn_in)
            lA, uA, lbias, ubias = prelayer_bounds(
                prelayer_model, Aaf, baf, tail, out_scale, out_offset,
                method=crown_method, **alpha_kwargs)
            A_mid, c_mid, resid = tm_domain_split(lA, uA, lbias, ubias)
            if sparse:
                inject_affine_domain_s(st, eng, A_mid, c_mid, resid, u_ids)
            else:
                inject_affine_domain(fp.pre_coeffs, fp.pre_rem, tables,
                                     A_mid, c_mid, resid, u_ids)
            w_u_metric = resid[..., 1] - resid[..., 0]  # [B, m]
        elif args.crown_relax == "two-slope":
            lA, uA, lbias, ubias = crown_bounds_two_slope(
                lirpa_model, config,
                hull[..., 0].contiguous(), hull[..., 1].contiguous(),
                input_layout=args.crown_input_layout)
            A_mid, c_mid, resid = two_slope_tm(lA, uA, lbias, ubias, hull)
            if sparse:
                inject_controls_two_slope_s(st, A_mid, c_mid, resid, u_ids, nn_in)
            else:
                inject_controls_two_slope(fp, A_mid, c_mid, resid, u_ids, nn_in)
            w_u_metric = resid[..., 1] - resid[..., 0]  # [B, m]
        else:
            T, u_min, u_max = crown_bounds(
                lirpa_model, config,
                hull[..., 0].contiguous(), hull[..., 1].contiguous(),
                input_layout=args.crown_input_layout)
            T, u_min, u_max = apply_crown_transport(
                T, u_min, u_max, args.crown_transport)
            if sparse:
                inject_controls_s(st, T, u_min, u_max, u_ids, nn_in)
            else:
                inject_controls(fp, T, u_min, u_max, u_ids, nn_in)
            w_u_metric = u_max - u_min  # [B, m]

        if metrics is not None:
            # Three widths, three loss points: NN-input hull width (the L1
            # box), the coupling's non-symbolic output gap (L2+L3: the
            # concretized box width under same-slope, the residual width
            # under two-slope — per-coupling semantics, labeled by the
            # 'coupling' field), and the injected u-row remainder width
            # (what the plant actually integrates for a control period —
            # the honest cross-arm metric, since P1/P2 arms redistribute
            # where the gap lands).
            w_x = hull[..., 1] - hull[..., 0]  # [B, nn_in]
            w_u = w_u_metric  # [B, m]
            u_rem = (st.pre_rem[:, u_ids] if sparse
                     else fp.pre_rem[:, u_ids])  # [B, m, 2]
            w_rem = u_rem[..., 1] - u_rem[..., 0]  # [B, m]
            live = (~broken).to(w_u.device)  # [B]
            nlive = int(live.sum())
            if nlive:
                lsel = live.nonzero(as_tuple=True)[0]
                entry = {
                    "k": k,
                    "active": nlive,
                    "x_hull_width_mean": w_x[lsel].mean(dim=0).tolist(),
                    "x_hull_width_max": w_x[lsel].max(dim=0).values.tolist(),
                    "u_box_width_mean": w_u[lsel].mean(dim=0).tolist(),
                    "u_box_width_max": w_u[lsel].max(dim=0).values.tolist(),
                    "u_rem_width_mean": w_rem[lsel].mean(dim=0).tolist(),
                    "u_rem_width_max": w_rem[lsel].max(dim=0).values.tolist(),
                }
                if args.crown_domain == "hybrid" and args.nn_mode != "tm":
                    # ablation telemetry: fraction of (cell, output) rows
                    # that chose the tm-affine / TM-through candidates
                    entry["ta_frac"] = hybrid_ta_frac
                    if args.nn_mode == "hybrid3":
                        entry["tm_frac"] = hybrid_tm_frac
                metrics["ctrl_steps"].append(entry)

        # (4) one control period of TM integration, SR persisting.
        if args.nvtx_phases and torch.device(args.device).type == "cuda":
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("gpu_flowfull/dynamics_certificate")
        if args.profile_components and torch.device(args.device).type == "cuda":
            ctrl_end.record()
            dyn_start.record()
        for j in range(substeps):
            if sparse:
                if args.nvtx_detail and torch.device(args.device).type == "cuda":
                    torch.cuda.nvtx.range_push("gpu_flowfull/driver/advance_sparse")
                st, ok = advance_sparse(st, code, eng, sched, settings, rem_est, sr)
                if args.nvtx_detail and torch.device(args.device).type == "cuda":
                    torch.cuda.nvtx.range_pop()
                st = prune_state(st, eng)
                fp = st  # status/verdict bookkeeping below reads .status only
            else:
                fp, ok = advance(fp, code, tables, step, sched, settings, rem_est, sr)
            if args.nvtx_detail and torch.device(args.device).type == "cuda":
                torch.cuda.nvtx.range_push("gpu_flowfull/driver/post_advance")
            newly_broken = (~ok.cpu()) & (~broken) & (fp.status.cpu() != ACTIVE)
            broken |= newly_broken
            if bool(newly_broken.any()):
                for bidx in newly_broken.nonzero(as_tuple=True)[0].tolist():
                    print("Flow* terminated.")
                    print(f"Broken branch: {bidx}")
            # F2: where this substep sits relative to the safe set's time
            # window. The flowpipe just produced covers [m*delta, (m+1)*delta]
            # with m = the number of substeps completed before it, so the
            # absolute time interval is exact integer arithmetic on the step.
            m_sub = k * substeps + j
            t_lo, t_hi = m_sub * settings.step, (m_sub + 1) * settings.step
            piece = window_piece(t_lo, t_hi, *safe_spec.window)  # (a, b) | None
            # per-substep constraint bookkeeping: hull pre-filter, escalated to
            # the composed-TM check for undecided lanes when `refine_specs` is on
            # (SpecGroup.ranges; identical verdict algebra either way).
            if (safe_spec and piece is not None) or unsafe_spec:
                hull_now = (hull_ranges_s(st, eng, n) if sparse
                            else hull_ranges(fp, tables, step, n))  # [B, n, 2]
                active = ~broken  # [B] cpu
                if safe_spec and piece is not None:
                    safe_spec.checked_steps += 1
                    # A step FULLY inside the window keeps the whole-step hull
                    # (identical numbers to the pre-F2 driver); one STRADDLING a
                    # window endpoint gets both enclosure routes restricted to
                    # the intersected sub-interval instead.
                    if piece == (0.0, t_hi - t_lo):
                        boxes, tdom = hull_now, None
                    else:
                        safe_spec.partial_steps += 1
                        tdom = torch.tensor([list(piece)] * B, dtype=torch.float64,
                                            device=args.device)  # [B, 2]
                        boxes = (rows_range_over_time_sparse(st, eng, tdom, n)
                                 if sparse else
                                 rows_range_over_time(fp.pre_coeffs[:, :n],
                                                      fp.pre_rem[:, :n], tdom,
                                                      tables))  # [B, n, 2]
                    # safe set: all q <= 0 must HOLD; sup > 0 => not provably
                    # inside (unknown); inf > 0 for some q => entirely outside.
                    q_all = safe_spec.ranges(boxes, active, time_domain=tdom)
                    for c in range(q_all.shape[1]):
                        q = q_all[:, c]  # [B, 2]
                        if bool(((q[..., 0] > 0) & ~broken.to(q.device)).any()):
                            safe_unsafe = True
                        elif bool(((q[..., 1] > 0) & ~broken.to(q.device)).any()):
                            safe_unknown = True
                if unsafe_spec:
                    all_in, any_out = decide_constraints(
                        unsafe_spec.ranges(hull_now, active)
                    )
                    if bool((all_in & active).any()):
                        unsafe_hit_inside = True
                    if not bool((any_out | ~active).all()):
                        unsafe_all_outside = False
            if bool(broken.all()):
                if args.nvtx_detail and torch.device(args.device).type == "cuda":
                    torch.cuda.nvtx.range_pop()
                break
            # CrownReach.cpp:163-166: a detected safe-set violation aborts the
            # whole run immediately (in_safeset != 0 -> break).
            if safe_unsafe:
                if args.nvtx_detail and torch.device(args.device).type == "cuda":
                    torch.cuda.nvtx.range_pop()
                break
            # SR queue at capacity -> Flow*'s reset (B1). LAST statement of the
            # substep body, exactly where reach_symbolic_remainder puts it
            # (Continuous.h:891-894): after the step is stored and safety
            # checked, and skipped when the step failed / aborted the run, both
            # of which `return` in Flow*. The queue persists ACROSS control
            # periods here just as it does in CrownReach.cpp (one
            # Symbolic_Remainder per cell, reused by every dynamics.reach call),
            # so a reset can and does land mid-benchmark; without this line the
            # 1001st substep raised "symbolic-remainder queue overflow", which
            # made every horizon/ode_step_size > 1000 instance unrunnable.
            # Soundness: see SymbolicRemainder.reset_if_full's docstring — the
            # emitted flowpipe already carries the full remainder, so the reset
            # drops the symbolic factorization, never the enclosure.
            sr.reset_if_full()
            if args.nvtx_detail and torch.device(args.device).type == "cuda":
                torch.cuda.nvtx.range_pop()
        if args.profile_components and torch.device(args.device).type == "cuda":
            dyn_end.record()
            component_events.append((ctrl_start, ctrl_end, dyn_start, dyn_end))
        if args.nvtx_phases and torch.device(args.device).type == "cuda":
            torch.cuda.nvtx.range_pop()
        if bool(broken.all()) or safe_unsafe:
            break

        # (5) TM handoff to the next control step.
        if sparse:
            end_of_time_s(st, eng)
        else:
            end_of_time(fp, tables, step)

    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - t0

    # --- verdicts, mirroring CrownReach.cpp:169-234 -------------------------
    if bool(broken.any()):
        pass  # "Flow* terminated." lines already printed -> UNKNOWN via parser
    if safe_spec:
        if safe_unsafe:
            print("Unsafe.")
        elif safe_unknown:
            print("Unknown.")
    if unsafe_spec and not bool(broken.any()):
        if unsafe_hit_inside:
            print("Unsafe")
        elif not unsafe_all_outside:
            print("Unknown")
        # provably outside everywhere: silence == verified (parser default)
    if target_spec and not bool(broken.any()):
        final_hull = (hull_ranges_s(st, eng, n) if sparse
                      else hull_ranges(fp, tables, step, n))  # post-handoff
        all_in, any_out = decide_constraints(
            target_spec.ranges(final_hull, ~broken)
        )
        if bool(all_in.all()):
            print("VERIFIED")
        elif bool(any_out.any()):
            print("FALSIFIED")
        else:
            print("UNKNOWN")

    if args.print_final_hull:
        # Aggregated (min-lo, max-hi over cells) final-time hull per variable
        # — the tightness metric used by the cross-tool comparison docs.
        fh = (hull_ranges_s(st, eng, n) if sparse
              else hull_ranges(fp, tables, step, n)).cpu()  # [B, n, 2]
        lo = fh[..., 0].min(dim=0).values
        hi = fh[..., 1].max(dim=0).values
        for i, nm in enumerate(var_names):
            print(f"HULL {nm} {lo[i]:.12g} {hi[i]:.12g}")

    if refine_specs:
        # How much escalation the run actually needed (lane-checks whose hull
        # was undecided). Printed ONLY when the new path is enabled, so the
        # default output stays byte-identical to the pre-E1 driver.
        #
        # WORDING IS LOAD-BEARING (GOTCHAS E1-4): both result parsers
        # (CROWN-Reach submit/run.py:parse_verifier_output and
        # comparison/parse_results.py) lowercase every output line and
        # SUBSTRING-match "unsafe"/"unreachable"/"falsified" -> FALSIFIED and
        # "flow* terminated"/"killed"/"unknown" -> UNKNOWN. A field named
        # `unsafe=0` would therefore turn every refined run into FALSIFIED.
        # Hence safeset/avoidset/targetset — no keyword substring anywhere.
        print(f"REFINE escalated lane-checks: safeset={safe_spec.escalations} "
              f"avoidset={unsafe_spec.escalations} "
              f"targetset={target_spec.escalations}")

    if safe_spec.windowed():
        # F2 audit line: how the window actually landed on the substep grid.
        # Same wording discipline as the REFINE line above (GOTCHAS E1-4): the
        # group is called `safeset`, never `constraints_unsafe`-like, because
        # both result parsers substring-match "unsafe" -> FALSIFIED.
        print(f"SPEC WINDOW safeset t in [{safe_spec.window[0]:.12g}, "
              f"{safe_spec.window[1]:.12g}]: checked on {safe_spec.checked_steps} "
              f"of {steps * substeps} substeps, {safe_spec.partial_steps} partial")

    if metrics is not None:
        # Final-time hull: aggregated per variable over LIVE lanes (broken
        # lanes carry stale rows), plus the mean per-lane summed width — the
        # scalar each arm is compared on.
        fh = (hull_ranges_s(st, eng, n) if sparse
              else hull_ranges(fp, tables, step, n)).cpu()  # [B, n, 2]
        live = ~broken  # [B] cpu
        metrics["broken"] = int(broken.sum())
        metrics["elapsed_s"] = elapsed
        metrics["nvtx_phases"] = bool(args.nvtx_phases)
        metrics["nvtx_detail"] = bool(args.nvtx_detail)
        metrics["compose_parent_assembly"] = (
            sparse_exec_module.COMPOSE_PARENT_ASSEMBLY
        )
        if torch.device(args.device).type == "cuda":
            metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            metrics["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        graph_owner = getattr(eng, "_graphs", None) if sparse else None
        metrics["cuda_graph_captures"] = (
            int(graph_owner.captures) if graph_owner is not None else 0
        )
        metrics["cuda_graph_hits"] = (
            int(graph_owner.hits) if graph_owner is not None else 0
        )
        if args.profile_components and component_events:
            controller_s = sum(
                start.elapsed_time(end) for start, end, _, _ in component_events
            ) / 1000.0
            dynamics_s = sum(
                start.elapsed_time(end) for _, _, start, end in component_events
            ) / 1000.0
            metrics["component_profile"] = {
                "method": "CUDA events on default stream; diagnostic only",
                "periods": len(component_events),
                "controller_coupling_cuda_seconds": controller_s,
                "dynamics_certificate_cuda_seconds": dynamics_s,
                "accounted_cuda_seconds": controller_s + dynamics_s,
                "unattributed_core_seconds": elapsed - controller_s - dynamics_s,
            }
        if bool(live.any()):
            fhl = fh[live]
            metrics["final_hull"] = {
                nm: [float(fhl[:, i, 0].min()), float(fhl[:, i, 1].max())]
                for i, nm in enumerate(var_names)
            }
            metrics["final_hull_width_sum_mean"] = float(
                (fhl[..., 1] - fhl[..., 0]).sum(dim=1).mean()
            )
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=1)

    print(f"time cost: {elapsed:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
