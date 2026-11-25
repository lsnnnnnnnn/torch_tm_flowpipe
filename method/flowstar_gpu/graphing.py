"""CUDA-graph capture of specialized tape segments (M10 launch-overhead fix).

Profiled motivation: at B = 1..64 the sparse advance is INTERPRETER-bound —
exec_valid_s + the refinement replay walk a Python tape emitting thousands of
micro-kernels (163k launches/advance at quad_reduced; 122 host syncs before
the M10 sync purge), so wall time is ~10x the GPU busy time. A CUDA graph
records the kernel stream ONCE (paying the Python cost at capture) and
replays it with a single launch — zero Python, near-zero launch overhead.

Correctness model:
  * A graph replays EXACTLY the kernel sequence recorded at capture, on
    static buffers. We key every graph by the full shape/spec signature of
    its segment, copy inputs into the static buffers, replay, and hand out
    the static outputs (consumed/copied before the next replay of the same
    graph). Results are BITWISE identical to the eager execution of the same
    ops (same kernels, same order, same reduction shapes) — the determinism
    suite asserts this.
  * Capture-safety contract for wrapped segments: pure GPU torch ops (or our
    fused kernels), no host syncs (.item()/.cpu()/data-dependent Python), no
    shape-dependent control flow beyond what the key pins. The segments
    wrapped here (specialized tape executions, the refinement body) satisfy
    it by construction: their control flow is frozen in the specialization.
  * Warmup: each segment runs eagerly `WARMUP` times on a side stream before
    capture (cuBLAS/cuDNN workspace setup must happen outside capture).

Fallback: `GraphCache.enabled = False` (or CPU tensors) short-circuits to
eager calls — the graph layer NEVER changes semantics, only dispatch.
"""

from __future__ import annotations

import torch

# Two eager warmup runs before capture: the wrapped segments contain no
# cuBLAS/cuDNN calls (elementwise + our fused kernels + segment/index ops
# only — einsum stays outside graphs), so the usual 3-run guidance is overly
# conservative and capture cost is a first-order term on short runs (dp_more
# aborts at step 5; every capture spent there is pure overhead).
WARMUP = 2


class GraphCache:
    """Owner of captured segments, keyed by caller-provided signatures.

    One instance per run (attach to the SparseEngine). Every entry holds the
    static input tensors, the captured torch.cuda.CUDAGraph, and the static
    outputs. Every entry owns a PRIVATE memory pool — see the comment in
    _capture (M10-7): pool sharing is unsafe when captures interleave with
    replays, which support-drift-triggered captures make routine here.
    """

    def __init__(self, device: str):
        from . import cuda_kernels as ck

        self.device = device
        # Graph capture REQUIRES the fused kernels: the torch fallback path
        # reduces via torch.segment_reduce, which is not capture-safe
        # (GOTCHAS M10-4). Without this gate, FLOWSTAR_NO_CUDA_KERNEL=1 on
        # the sparse CUDA path crashed at first capture (doc-verification
        # workflow finding, confirmed by execution); now it falls back to
        # eager execution instead.
        self.enabled = torch.device(device).type == "cuda" and ck.available()
        self.pool = None
        self._capture_stream = None
        self._segs: dict = {}
        self.hits = 0
        self.captures = 0

    def run(self, key, fn, inputs: list, keepalive: tuple = ()):
        """Execute `fn(*statics)` under the graph for `key`.

        inputs: list of tensors — the tensors fn reads that CHANGE between
        calls. Every OTHER tensor fn's closure touches must either live for
        the engine's lifetime (engine/lru-cached tables) or be passed in
        `keepalive`: a replayed graph dereferences the capture-time buffers,
        so a freed closure temporary means replays read recycled memory —
        this exact bug produced 4-decade remainder blowups on the graph HIT
        path before keepalive existed (M10 GOTCHA; regression-tested).
        Returns fn's outputs as the STATIC buffers: consume or clone them
        before the same key runs again. Non-CUDA or disabled: eager call.
        """
        if not self.enabled or not inputs[0].is_cuda:
            return fn(*inputs)
        ent = self._segs.get(key)
        if ent is None:
            ent = self._capture(key, fn, inputs, keepalive)
        statics, graph, out, _ka = ent
        for s, x in zip(statics, inputs):
            s.copy_(x)
        graph.replay()
        self.hits += 1
        return out

    def _capture(self, key, fn, inputs: list, keepalive: tuple = ()):
        """Warmup eagerly on a side stream, then record fn's kernel stream."""
        statics = [x.clone() for x in inputs]
        # Reuse this engine's capture stream, retaining PRIVATE graph pools.
        # A fresh stream per signature causes cuBLAS to keep another workspace
        # for every stream once graph regions include matrix products.
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=self.device)
        side = self._capture_stream
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(WARMUP):
                fn(*statics)
        torch.cuda.current_stream(self.device).wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        # PRIVATE memory pool per graph (M10-7): sharing one pool across
        # entries corrupts data when a NEW capture happens after other
        # graphs have replayed (allocator reuse decisions no longer match
        # the replay-time liveness; observed as all-lanes contraction
        # failure when the glue regions joined the pool). Support drift
        # makes mid-run captures normal here, so pool sharing is unsafe by
        # construction in this engine. Costs some reserved memory per graph.
        with torch.cuda.device(self.device):
            # Use the public capture API on our warmed side stream. The
            # torch.cuda.graph convenience context also performs gc.collect()
            # and empty_cache() on EVERY new support signature. Those global
            # memory-reclamation passes dominate short solves; our private
            # pools and explicit keepalive ownership do not require them.
            torch.cuda.synchronize(self.device)
            with torch.cuda.stream(side):
                graph.capture_begin()
                try:
                    out = fn(*statics)
                finally:
                    graph.capture_end()
        ent = (statics, graph, out, keepalive)
        self._segs[key] = ent
        self.captures += 1
        return ent


def graph_cache(eng, device: str) -> GraphCache:
    """Per-engine singleton GraphCache (attached lazily)."""
    gc = getattr(eng, "_graphs", None)
    if gc is None:
        gc = GraphCache(device)
        eng._graphs = gc
    return gc
