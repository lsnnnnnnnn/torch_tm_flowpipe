"""P3 — Taylor models THROUGH smooth neural controllers (poly-CROWN campaign).

Kills loss point L2 (linear-in-x relaxation) for tanh/sigmoid controllers:
instead of certifying ANY linear enclosure of NN over a set, propagate the
state Taylor models through the network in validated TM arithmetic —

    affine layer   W·TM + b : exact polynomial combination + remainder dot
                              (the inject_controls einsum pattern);
    activation     σ(TM)    : composition through the engine's VALIDATED
                              elementary series (tanh = 1 − 2/(exp(2F)+1),
                              sigmoid = 1/(exp(−F)+1) — exp_series_valid +
                              rec_series_valid with their Lagrange tails,
                              the exact ops the plant integrator trusts);

so u keeps the network's curvature as a full order-k polynomial in the
flowpipe's own domain symbols r, and the injected remainder is the genuine
composition error, not a Θ(diam²) relaxation gap.

v1 scope decisions (user-approved, POLYCROWN_CAMPAIGN.md):
  * point coefficients BETWEEN layers: each activation runs on
    interval-lifted coefficients and its result is collapsed back to
    midpoints with the radius range folded into the row remainder
    (`_collapse_iv`) — per-layer boxing of the σ-composition error only;
    the affine parts stay symbolic, which is the dominant cancellation
    channel for 2–3-layer controller MLPs. Symbolic remainders through the
    net (POLAR's product chain) are the v2 escalation if remainder growth
    is observed.
  * (Sprint-2 update) nonsmooth kinds INCLUDING ReLU propagate through the
    activation-kind protocol (nn_act_kinds): exact tiers on stable neurons,
    Clarke-subdifferential centered errors on unstable ones.
  * dense basis; the sparse-support specialization comes with the
    QUAD-class scaling work.

Soundness: every operation is either exact-in-structure (einsum polynomial
combination, tier-P like Flow*'s Real arithmetic, with the coefficient
roundoff NOT bounded — matching the engine's parity trust model) or a
validated interval op (series, remainder algebra, collapses — outward
rounded). MC containment tests pin the end-to-end claim.
"""

from __future__ import annotations

import torch

from . import elementary as el
from . import interval as iv
from .monomials import MonomialTables
from .polynomial import StepTables
from .rounding import next_down, next_up

# Transient-memory budget per activation tm_mul (the [rows, P] product):
# rows are chunked so one mul stays under this. CartPole at B=256 asked for
# 719 GiB unchunked. Module-level so tests can shrink it to exercise the
# chunked path on small bases.
_CHUNK_BUDGET_BYTES = 2e9


def layers_from_onnx(path: str) -> list[tuple]:
    """Parse a plain-MLP ONNX controller into [("linear", W, b) | ("act", name)].

    Supports the controller shapes this campaign meets: Gemm (with transB
    handled), MatMul+Add pairs, Tanh/Sigmoid/Relu activations (Sprint-2
    lifted the ReLU refusal — nn_act_kinds owns the nonsmooth tiers),
    Sub with a constant operand (input normalization; folded into an
    affine layer), Conv layers that are MLP layers in disguise (kernel
    spanning the full spatial extent, or 1x1 on a 1x1 map — the
    ARCH-COMP tora conv-form controllers), and shape-only nodes
    (Flatten/Reshape/Identity/Squeeze/Unsqueeze) which are ignored — the
    TM rows are already flat. Anything else raises ValueError (hybrid3
    degrades to the 2-way hybrid on such nets).

    W [out, in] and b [out] are float64 tensors.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    inits = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()).to(torch.float64)
             for i in model.graph.initializer}
    # Spatial dims of the data path, for the Conv-as-MLP lowering. Taken from
    # the one graph input that is NOT an initializer (legacy exporters list
    # weights as graph inputs too); None for 1-D/2-D inputs (pure MLPs).
    spatial: tuple[int, int] | None = None
    channels = 1
    for gi in model.graph.input:
        if gi.name in inits:
            continue
        dims = [d.dim_value for d in gi.type.tensor_type.shape.dim]
        if len(dims) == 4:
            spatial = (dims[2], dims[3])
            channels = dims[1]
    layers: list[tuple] = []
    pending_matmul: torch.Tensor | None = None

    def flush_pending() -> None:
        # A bias-free MatMul is a complete linear layer once the next node is
        # not its Add. Without this, consecutive MatMuls (Docking's
        # preprocess-scale then fc_1) silently DROPPED the first one — caught
        # by the onnx2pytorch reference-forward check, 2026-08-06.
        nonlocal pending_matmul
        if pending_matmul is not None:
            layers.append(("linear", pending_matmul,
                           torch.zeros(pending_matmul.shape[0],
                                       dtype=torch.float64)))
            pending_matmul = None

    for node in model.graph.node:
        op = node.op_type
        if op in ("Flatten", "Reshape", "Identity", "Squeeze", "Unsqueeze",
                  "Cast", "Constant"):
            continue
        if op == "Gemm":
            flush_pending()
            attrs = {a.name: a for a in node.attribute}
            w = None
            b = torch.zeros(0)
            for name in node.input[1:]:
                if name in inits:
                    t = inits[name]
                    if t.dim() == 2:
                        w = t
                    else:
                        b = t
            if w is None:
                raise ValueError(f"Gemm without initializer weight: {node.name}")
            if not (attrs.get("transB") and attrs["transB"].i):
                w = w.T.contiguous()  # to [out, in]
            alpha = attrs.get("alpha")
            beta = attrs.get("beta")
            if (alpha and alpha.f != 1.0) or (beta and beta.f != 1.0):
                raise ValueError(f"Gemm with alpha/beta != 1: {node.name}")
            if b.numel() == 0:
                b = torch.zeros(w.shape[0], dtype=torch.float64)
            layers.append(("linear", w, b.reshape(-1)))
        elif op == "MatMul":
            flush_pending()
            for name in node.input:
                if name in inits:
                    # x @ W convention: stored [in, out] -> [out, in]
                    pending_matmul = inits[name].T.contiguous()
            if pending_matmul is None:
                raise ValueError(f"MatMul without initializer: {node.name}")
        elif op == "Add":
            b = None
            for name in node.input:
                if name in inits:
                    b = inits[name].reshape(-1)
            if pending_matmul is not None:
                if b is None:
                    raise ValueError(f"Add after MatMul without bias: {node.name}")
                layers.append(("linear", pending_matmul, b))
                pending_matmul = None
            else:
                raise ValueError(f"free-standing Add unsupported: {node.name}")
        elif op == "Sub":
            # Sub with a constant operand is an affine op:
            #   x - c  ->  W=I,  b=-c      (input normalization, the common case)
            #   c - x  ->  W=-I, b=+c
            # With a MatMul pending, fold into it instead of emitting I.
            c = None
            const_first = False
            for pos, name in enumerate(node.input):
                if name in inits:
                    c = inits[name].reshape(-1)
                    const_first = pos == 0
            if c is None:
                raise ValueError(f"Sub without constant operand: {node.name}")
            if pending_matmul is not None:
                w = -pending_matmul if const_first else pending_matmul
                layers.append(("linear", w, c if const_first else -c))
                pending_matmul = None
            else:
                eye = torch.eye(c.numel(), dtype=torch.float64)
                layers.append(("linear", -eye if const_first else eye,
                               c if const_first else -c))
        elif op == "Conv":
            flush_pending()
            attrs = {a.name: a for a in node.attribute}
            w = inits.get(node.input[1]) if len(node.input) > 1 else None
            if w is None or w.dim() != 4:
                raise ValueError(f"Conv without 4-D initializer weight: "
                                 f"{node.name}")
            kh, kw = int(w.shape[2]), int(w.shape[3])
            pads = list(attrs["pads"].ints) if "pads" in attrs else [0] * 4
            strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
            dil = list(attrs["dilations"].ints) if "dilations" in attrs else [1, 1]
            group = attrs["group"].i if "group" in attrs else 1
            if (any(pads) or strides != [1, 1] or dil != [1, 1] or group != 1
                    or spatial is None or (kh, kw) != spatial):
                raise ValueError(
                    f"Conv is not an MLP layer in disguise (kernel {kh}x{kw}, "
                    f"spatial {spatial}, pads {pads}, strides {strides}, "
                    f"group {group}): {node.name}")
            if channels != 1 and spatial != (1, 1):
                # flatten-order ambiguity (channel-major vs spatial-major)
                # cannot arise when either factor is degenerate; refuse the
                # genuinely 2-D case rather than guess.
                raise ValueError(
                    f"Conv over multi-channel non-1x1 map unsupported: "
                    f"{node.name}")
            b = inits[node.input[2]].reshape(-1) if len(node.input) > 2 else \
                torch.zeros(w.shape[0], dtype=torch.float64)
            # [outC, inC, kh, kw] -> [outC, inC*kh*kw]; input flatten order
            # (NCHW) matches because channels==1 or spatial==(1,1).
            layers.append(("linear", w.reshape(w.shape[0], -1).contiguous(), b))
            spatial = (1, 1)
            channels = int(w.shape[0])
        elif op in ("Tanh", "Sigmoid", "Relu"):
            if pending_matmul is not None:
                layers.append(("linear", pending_matmul,
                               torch.zeros(pending_matmul.shape[0],
                                           dtype=torch.float64)))
                pending_matmul = None
            layers.append(("act", op.lower()))
        else:
            raise ValueError(
                f"unsupported op for TM-through-NN: {op} (node {node.name})")
    if pending_matmul is not None:
        layers.append(("linear", pending_matmul,
                       torch.zeros(pending_matmul.shape[0], dtype=torch.float64)))
    if not any(t[0] == "linear" for t in layers):
        raise ValueError("no affine layers found — not an MLP controller?")
    return layers


def _collapse_iv(
    coeffs_iv: torch.Tensor,
    rem: torch.Tensor,
    tables: MonomialTables,
    step: StepTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interval coefficients -> point midpoints + widened remainder.

    coeffs_iv [R, T, 2], rem [R, 2] -> (mid [R, T], rem' [R, 2]) with
    rem' ⊇ rem + range(coeffs_iv − mid) over the normal domain — the same
    split cutoff_normal_interval performs, taken to completion (every
    coefficient becomes a point).
    """
    mid = (coeffs_iv[..., 0] + coeffs_iv[..., 1]) * 0.5  # [R, T] RN
    mid_iv = torch.stack((next_down(mid), next_up(mid)), dim=-1)
    rad = iv.sub(coeffs_iv, mid_iv)  # [R, T, 2] encloses coeff − mid
    m_basis = coeffs_iv.shape[-2]
    prod = iv.mul(rad, step.factor(0, m_basis, rad.dim() - 1))
    return mid, iv.add(rem, iv.sum(prod, dim=-1))


def _quadratic_fit_error(
    kind, arg: torch.Tensor, degree: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """P4: least-squares quadratic fit of σ over per-row ranges + RIGOROUS
    error enclosure.

    arg [R, 2] -> (a0, a1, a2, mid [R], E [R, 2]) with
    σ(s) ∈ a0 + a1·(s−mid) + a2·(s−mid)² + E for every s ∈ arg. The fit is
    UNRIGOROUS (RN least squares over 9 samples — any q is sound once its
    error is bounded); the rigor is E: sup/inf of σ − q enclosed by 16-piece
    subdivided interval evaluation (each piece's enclosure contains the true
    difference there; the hull contains the extremes) — the CORA NFM'23
    fit-over-interval recipe with the sampling replaced by a sound bound.
    Degenerate (zero-width) ranges are handled by the same algebra: every
    piece collapses to the point and E is ~ulp-wide.
    """
    from .nn_act_kinds import get_kind, kink_edges

    kobj = get_kind(kind) if isinstance(kind, str) else kind
    f_pt, f_iv = kobj.f_pt, kobj.f_iv
    lo, hi = arg[..., 0], arg[..., 1]  # [R]
    mid = (lo + hi) * 0.5
    half = (hi - lo) * 0.5  # [R] >= 0

    # fit in normalized coords t ∈ [-1, 1] (conditioning), then rescale
    S = 9
    ts = torch.linspace(-1.0, 1.0, S, dtype=arg.dtype, device=arg.device)
    xs = mid.unsqueeze(-1) + half.unsqueeze(-1) * ts  # [R, S]
    ys = f_pt(xs)  # [R, S]
    cols = [torch.ones_like(ts), ts] + ([ts * ts] if degree >= 2 else [])
    A = torch.stack(cols, dim=-1)  # [S, degree+1]
    # least squares via the constant pseudoinverse (A is row-independent),
    # applied with an EXPLICIT fixed-length reduction: both cuSOLVER lstsq
    # and BLAS matmul are batch-size-dependent in association order, which
    # broke the chunked-rows bitwise-equality contract; an elementwise
    # product summed over the constant S-axis reduces every row identically
    # regardless of how many rows share the call.
    pinv = torch.linalg.pinv(A)  # [3, S]
    sol = (pinv.unsqueeze(0) * ys.unsqueeze(1)).sum(dim=-1)  # [R, degree+1]
    safe_half = half.clamp(min=1e-300)
    a0 = sol[:, 0]
    a1 = sol[:, 1] / safe_half
    a2 = (sol[:, 2] / (safe_half * safe_half) if degree >= 2
          else torch.zeros_like(a1))
    # zero-width rows: constant fit at the point value
    degen = half <= 0
    a0 = torch.where(degen, f_pt(mid), a0)
    a1 = torch.where(degen, torch.zeros_like(a1), a1)
    a2 = torch.where(degen, torch.zeros_like(a2), a2)

    # rigorous error over N subdivisions, CENTERED (mean-value) form:
    #   E(s) ∈ E(c_i) + E'(piece_i)·(s − c_i)   for s ∈ piece_i,
    # E' = σ' − q' with σ' expressed through σ's OWN enclosure (tanh' =
    # 1 − y², sigmoid' = y·(1−y), y = f_iv(piece) ⊆ codomain — always
    # finite/tight). The naive subtraction f_iv(piece) − q_iv(piece) wraps
    # by the width of BOTH enclosures (~0.3 claimed vs ~0.03 true over
    # [-1,1]); the centered form's wrap is O(width²·|E''|) per piece.
    N = 16
    # KINK-AWARE edges (S2-2): for nonsmooth kinds one edge lands exactly on
    # each interior kink, so only that piece carries the wide
    # subdifferential; uniform (bitwise-identical to the historical
    # linspace) for smooth kinds.
    edges = kink_edges(kobj, lo, hi, N)  # [R, N+1]
    p_lo = lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * edges[:, :-1]  # [R, N]
    p_hi = lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * edges[:, 1:]
    piece = torch.stack((p_lo, p_hi), dim=-1)  # [R, N, 2]
    ctr = (p_lo + p_hi) * 0.5  # [R, N]
    ctr_iv = iv.from_point(ctr)  # [R, N, 2]

    # E(c_i): point-argument enclosures (f_iv of a degenerate interval is a
    # few-ulp band; q at c_i in interval arithmetic against exact-float a_i)
    d_ctr = iv.sub(ctr_iv, iv.from_point(mid).unsqueeze(1))  # [R, N, 2]
    q_ctr = iv.add(
        iv.from_point(a0.unsqueeze(-1).expand(-1, N)),
        iv.add(
            iv.mul_point(d_ctr, a1.unsqueeze(-1)),
            iv.mul_point(iv.pow_int(d_ctr, 2), a2.unsqueeze(-1)),
        ),
    )
    e_ctr = iv.sub(f_iv(ctr_iv), q_ctr)  # [R, N, 2]

    # E'(piece) = σ'(piece) − q'(piece); σ' via the kind's derivative /
    # Clarke-subdifferential enclosure (Lebourg's MVT licenses the centered
    # form for Lipschitz kinds — the S2-2 generalization)
    fp = kobj.df_iv(piece)  # [R, N, 2]
    d_pc = iv.sub(piece, iv.from_point(mid).unsqueeze(1))  # [R, N, 2]
    qp = iv.add(iv.from_point(a1.unsqueeze(-1).expand(-1, N)),
                iv.mul_point(d_pc, (2.0 * a2).unsqueeze(-1)))
    ep = iv.sub(fp, qp)  # [R, N, 2]

    halfw = ((p_hi - p_lo) * 0.5).unsqueeze(-1)  # [R, N, 1]
    dev = torch.cat((-halfw, halfw), dim=-1)  # [R, N, 2] = s − c_i range
    e_pc = iv.add(e_ctr, iv.mul(ep, dev))  # [R, N, 2]
    E = torch.stack(
        (e_pc[..., 0].amin(dim=1), e_pc[..., 1].amax(dim=1)), dim=-1)
    return a0, a1, a2, mid, E


def _quadratic_candidate(
    kind: str,
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    arg: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """P4 TM image: q(F) = a0 + a1·(F−mid) + a2·(F−mid)² + E, ONE tm_mul.

    coeffs [R, T] point / rem [R, 2] / arg [R, 2] (range of poly + rem, the
    SAME enclosure the fit error was certified over) -> point coeffs +
    remainder, collapsed like the other tiers.
    """
    a0, a1, a2, mid, E = _quadratic_fit_error(kind, arg)  # kind: str | ActKind
    fc = iv.from_point(coeffs)  # [R, T, 2]
    fc[:, 0] = iv.sub(fc[:, 0], iv.from_point(mid))  # F − mid
    f_range = iv.sub(arg, iv.from_point(mid))  # range(F − mid)
    cache = coeffs.new_zeros(coeffs.shape[0], 3, 2)
    sq_kept, sq_r = el.tm_mul_valid(
        fc, rem, fc, rem, k, tables, step, cutoff, cache, 0, f_range)
    sq_c = el._pad(sq_kept, coeffs.shape[1])
    q_c = iv.add(
        iv.mul_point(fc, a1.unsqueeze(-1)),
        iv.mul_point(sq_c, a2.unsqueeze(-1)),
    )
    q_c[:, 0] = iv.add(q_c[:, 0], iv.from_point(a0))
    q_r = iv.add(iv.mul_point(rem, a1), iv.mul_point(sq_r, a2))
    q_r = iv.add(q_r, E)
    return _collapse_iv(q_c, q_r, tables, step)


def _act_series(
    kind: str,
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff: float,
    tabs: el.ElemTables,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """σ(TM) for one batch of rows via the dedicated validated series.

    coeffs [R, T] point, rem [R, 2] -> (point coeffs [R, T], rem [R, 2],
    bad [R] bool — always False: tanh/sigmoid are entire and their series
    (elementary.tanh_series_valid / sigmoid_series_valid: ODE-recurrence
    coefficients + derivative-polynomial Lagrange tails) are finite for ANY
    argument range, including the deep-saturation regimes where an
    exp-composite formulation overflows (Docking's ±100-scale
    pre-activations — the v1 chain broke there at control step 0).
    """
    from .nn_act_kinds import get_kind

    R, T = coeffs.shape
    kobj = get_kind(kind) if isinstance(kind, str) else kind
    f_iv = kobj.f_iv

    from . import polynomial as poly

    arg_pre = iv.add(poly.range_normal(coeffs, tables, step), rem)  # [R, 2]

    if kobj.has_series:
        series = (el.tanh_series_valid if kobj.name == "tanh"
                  else el.sigmoid_series_valid)
        cache = coeffs.new_zeros(R, 4 * k + 2, 2)  # documented slot budget
        r_c, r_r = series(iv.from_point(coeffs), rem, k, tables, step,
                          cutoff, cache, 0, tabs)
        out_c, out_r = _collapse_iv(r_c, r_r, tables, step)
    else:
        # no Taylor series for nonsmooth kinds — the linear fit tier
        # (degree-1, no tm_mul) takes the series slot in the dispatch
        a0, a1, _, midl, El = _quadratic_fit_error(kobj, arg_pre, degree=1)
        lin_c = coeffs * a1.unsqueeze(-1)
        lin_c[:, 0] = lin_c[:, 0] + (a0 - a1 * midl)
        lin_r = iv.add(iv.mul_point(rem, a1), El)
        out_c, out_r = lin_c, lin_r

    # Per-neuron dispatch: tanh/sigmoid Taylor series have convergence
    # radius pi/2 (poles at ±i·pi/2), so over a WIDE pre-activation range
    # the Lagrange tail explodes by construction — while the degree-0
    # enclosure σ(range) is trivially sound and TIGHT exactly there
    # (saturated neurons are near-constant). Each row keeps whichever
    # certified remainder is narrower: the series where it converges, the
    # box where it cannot. Same per-row-mix soundness argument as the
    # driver's hybrid coupling — both candidates independently enclose
    # σ(F(r) + rem) for every r.
    arg = arg_pre
    box = f_iv(arg)  # [R, 2]
    box_mid = (box[..., 0] + box[..., 1]) * 0.5
    box_c = torch.zeros_like(coeffs)
    box_c[:, 0] = box_mid
    box_r = iv.sub(box, torch.stack((box_mid, box_mid), dim=-1))  # [R, 2]

    # P4 tier: quadratic fit-over-interval (one tm_mul; the CORA recipe on
    # TM bookkeeping). Wins the middle regime — ranges too wide for the
    # order-k Taylor tail, too narrow for the box to be tight.
    quad_c, quad_r = _quadratic_candidate(
        kobj, coeffs, rem, arg, k, tables, step, cutoff)

    w_series = out_r[..., 1] - out_r[..., 0]
    w_box = box_r[..., 1] - box_r[..., 0]
    w_quad = quad_r[..., 1] - quad_r[..., 0]
    w_series = torch.where(torch.isfinite(w_series), w_series,
                           torch.full_like(w_series, torch.inf))
    w_quad = torch.where(torch.isfinite(w_quad), w_quad,
                         torch.full_like(w_quad, torch.inf))
    pick = torch.stack((w_series, w_box, w_quad)).argmin(dim=0)  # [R]
    out_c = torch.where((pick == 1).unsqueeze(-1), box_c, out_c)
    out_r = torch.where((pick == 1).unsqueeze(-1), box_r, out_r)
    out_c = torch.where((pick == 2).unsqueeze(-1), quad_c, out_c)
    out_r = torch.where((pick == 2).unsqueeze(-1), quad_r, out_r)

    # S2-2 exact-regime override: rows whose range makes the activation an
    # affine map bypass every tier with an EXACT result (ReLU stable
    # neurons: identity above the kink, zero below).
    ident, zero = kobj.exact_regimes(arg)
    if bool(ident.any()):
        out_c = torch.where(ident.unsqueeze(-1), coeffs, out_c)
        out_r = torch.where(ident.unsqueeze(-1), rem, out_r)
    if bool(zero.any()):
        out_c = torch.where(zero.unsqueeze(-1), torch.zeros_like(out_c),
                            out_c)
        out_r = torch.where(zero.unsqueeze(-1), torch.zeros_like(out_r),
                            out_r)
    return out_c, out_r, torch.zeros(R, dtype=torch.bool, device=coeffs.device)


def nn_tm_bounds(
    layers: list[tuple],
    coeffs: torch.Tensor,
    rem: torch.Tensor,
    k_series: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate state TMs through an MLP controller in TM arithmetic.

    coeffs [B, nn_in, T] point (control-time rows), rem [B, nn_in, 2] ->
    (u_coeffs [B, m, T], u_rem [B, m, 2], ok [B] bool). The output is a
    FULL order-k TM of u in the flowpipe's domain symbols — inject as-is,
    no linearization anywhere.
    """
    B = coeffs.shape[0]
    dev = coeffs.device
    tabs = el.elem_tables(k_series + 1, str(dev))
    cur_c, cur_r = coeffs, rem  # [B, w, T], [B, w, 2]
    ok = torch.ones(B, dtype=torch.bool, device=dev)
    for entry in layers:
        if entry[0] == "linear":
            _, W, b = entry
            W = W.to(dev)
            b = b.to(dev)
            cur_c = torch.einsum("oi,bit->bot", W, cur_c)
            cur_c[..., 0] += b
            cur_r = iv.dot_point_iv(W.unsqueeze(0), cur_r.unsqueeze(1), dim=-1)
        else:
            _, kind = entry
            w = cur_c.shape[1]
            flat_c = cur_c.reshape(B * w, -1)
            flat_r = cur_r.reshape(B * w, 2)
            # Row-chunking: the series' tm_mul materializes [rows, P(, 2)]
            # products (P = pair count, T² pairs at order k) — CartPole at
            # B=256 asked for 719 GiB unchunked. Budget ~2 GiB of transient
            # per mul; each chunk runs the full dispatch independently
            # (rows are independent neurons).
            P = int(tables.pair_i.shape[0])
            chunk = max(1, int(_CHUNK_BUDGET_BYTES / (P * 8 * 6)))
            if flat_c.shape[0] <= chunk:
                out_c, out_r, bad = _act_series(
                    kind, flat_c, flat_r, k_series, tables, step, cutoff, tabs)
            else:
                pieces = [
                    _act_series(kind, flat_c[i:i + chunk], flat_r[i:i + chunk],
                                k_series, tables, step, cutoff, tabs)
                    for i in range(0, flat_c.shape[0], chunk)
                ]
                out_c = torch.cat([p[0] for p in pieces])
                out_r = torch.cat([p[1] for p in pieces])
                bad = torch.cat([p[2] for p in pieces])
            cur_c = out_c.reshape(B, w, -1)
            cur_r = out_r.reshape(B, w, 2)
            ok &= ~bad.reshape(B, w).any(dim=1)
    return cur_c, cur_r, ok
