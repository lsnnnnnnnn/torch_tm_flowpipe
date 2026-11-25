"""torch.compile fusion of the eager glue between interpreter kernels (M11).

Profiled motivation (2026-07-29, bench/profile_sparse.py at high B): with the
one-launch interpreter kernels carrying the validated/point/refinement
passes, 40-60% of a high-B advance is the eager GLUE between them —
integrate/union-embed/precondition/diff/emit chains of small torch ops
(quad_official B=1024: ~16 of 37.7 ms/advance). Measured on the same
shapes, torch.inductor fuses such chains ~6.5x over eager and MATCHES our
hand-written elementwise kernels (0.129 vs 0.130 ms on the T2 experiment
chain), while additionally fusing ACROSS ops — which per-op kernels cannot.
This module applies exactly that: the interpreter kernels keep the
heavyweight passes (inductor cannot express a block-per-lane TM
interpreter), inductor fuses the glue.

Contracts:
  * Compiled regions run the PURE-TORCH interval arithmetic (RN + nextafter
    + Rump reduction bounds): iv-op kernel dispatch would graph-break per
    call, so `run()` pins interval.USE_KERNELS/support.USE_KERNELS to False
    for the region (dynamo guards on those globals — they must hold the
    same value at trace and replay). Soundness is unchanged — the torch
    path is the original sound emulation, and the Rump bounds license ANY
    reduction order, including inductor's fusions/reorderings. Results are
    ulp-level different from (equal-or-wider than) the kernel path; the
    advance-level differential gates cover this.
  * DETERMINISM RULE: no colliding index_add_ inside compiled regions
    (inductor may lower scatter-adds to atomics; colliding targets would be
    order-nondeterministic). Injective index_copy_/index_add_ are fine —
    each slot written once. evaluate_time_end_s (colliding by design) stays
    OUTSIDE compiled glue.
  * Keys: like the graph tier, every distinct support signature is its own
    compiled entry (dynamic=False; shapes are static per specialization).
    Compilation cost ~1-4 s per region per signature, amortized exactly
    like graph captures; the support-accumulation stabilizer keeps the key
    count small.

Escape hatches: GLUE_COMPILE flag / FLOWSTAR_NO_COMPILE=1 env — regions
then run the identical eager code (the compiled and eager paths are the
same Python function).
"""

from __future__ import annotations

import os

import torch

from . import interval as _iv
from . import support as _sup

# Glue execution mode — set FLOWSTAR_GLUE=eager|graph|compile.
# MEASURED OUTCOME (2026-07-29, docs/OPTIMIZATION.md §9): "eager" is the
# DEFAULT because at high B the eager glue's Python dispatch is hidden
# behind asynchronous GPU execution — the engine is bandwidth/compute-bound
# there, so:
#   "graph"   (CUDA-graph the regions) measured NEUTRAL: quad_official
#             83.0 -> 83.4 s, nav@2560 19.8 -> 19.8 s, cartpole@4096 -6%,
#             dp_less@900 +7% (capture cost on short runs) — not worth the
#             extra graph-lifetime constraints (GOTCHAS M10-1/M10-7 class).
#   "compile" (torch.inductor fusion) removes BANDWIDTH, the true remaining
#             lever, and matches our hand-fused kernels per-op — but its
#             2-5 s per-region-per-signature compile cost is a net LOSS on
#             one-shot verification (cartpole@4096 11.6 -> 30.5 s cold,
#             17.8 s inductor-cache-warm). It pays only in amortized
#             many-run regimes — the DiffReach/XLA design point. Kept as
#             the documented opt-in for exactly that workload.
GLUE_MODE = os.environ.get("FLOWSTAR_GLUE", "eager")
if os.environ.get("FLOWSTAR_NO_COMPILE", "0") == "1":  # back-compat alias
    GLUE_MODE = "eager"
assert GLUE_MODE in ("eager", "graph", "compile"), GLUE_MODE

# Compile only pays where the glue is bandwidth-bound: at small B the eager
# glue is a few hundred microseconds while each region compile costs 1-4 s —
# a pure regression for B=1 early-abort benchmarks. Below this batch size
# regions run eager (identical code, identical semantics).
GLUE_MIN_B = 64


class GlueCache:
    """Per-engine dispatcher for glue regions, keyed by the caller's
    (region, support-signature) tuples — mirroring GraphCache's discipline."""

    def __init__(self) -> None:  # noqa: D107 — trivial container
        self._fns: dict = {}
        self.compiles = 0
        self.calls = 0
        self.eng = None  # set by glue_cache (graph mode borrows GraphCache)

    def run(self, key, fn, *args):
        """Run `fn(*args)` per GLUE_MODE (graph / compile / eager).

        fn must be effect-free on Python state (tensor math only), read only
        its arguments plus engine-lifetime cached tensors (the GraphCache
        keepalive discipline), and contain no colliding scatter-adds. The
        kernel dispatch inside is pinned off for graph/compile execution so
        the recorded/traced ops are pure torch — sound: the torch path is
        the original sound emulation and Rump bounds license any reduction
        order (module docstring).
        """
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        first = tensors[0]
        if (GLUE_MODE == "eager" or not first.is_cuda
                or (GLUE_MODE == "compile" and first.shape[0] < GLUE_MIN_B)):
            return fn(*args)
        if GLUE_MODE == "compile":
            # Inductor path: kernel dispatch pinned off (extension calls
            # would graph-break per op; the torch fallback is the original
            # sound emulation and inductor fuses it — module docstring).
            ent = self._fns.get(key)
            if ent is None:
                ent = torch.compile(fn, dynamic=False)
                self._fns[key] = ent
                self.compiles += 1
            saved = (_iv.USE_KERNELS, _sup.USE_KERNELS)
            _iv.USE_KERNELS = False
            _sup.USE_KERNELS = False
            try:
                out = ent(*args)
            finally:
                _iv.USE_KERNELS, _sup.USE_KERNELS = saved
        else:
            # Graph mode: capture the CURRENT kernel stream (fused kernels
            # INCLUDED — graph capture is their native tier), so replays are
            # bitwise-identical to the eager glue at any batch size.
            from .graphing import graph_cache

            idx = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]

            def call(*ts):
                rebuilt = list(args)
                for i, t in zip(idx, ts):
                    rebuilt[i] = t
                return fn(*rebuilt)

            gc = graph_cache(self.eng, str(first.device))
            keep = tuple(tensors)
            out = gc.run(("glue",) + key, call, list(keep), keepalive=keep)
        self.calls += 1
        return out


def glue_cache(eng) -> GlueCache:
    """Per-engine singleton (lazily attached, like graph_cache)."""
    gc = getattr(eng, "_glue", None)
    if gc is None:
        gc = GlueCache()
        gc.eng = eng
        eng._glue = gc
    return gc
