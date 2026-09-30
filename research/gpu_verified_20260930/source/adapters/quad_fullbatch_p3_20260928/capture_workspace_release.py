"""Isolated unused-cache release after graph warmup, before private capture.

The copied original _capture has one inserted helper call after its existing
device synchronization. No graph, output, input or keepalive cache is cleared.
Counter changes are allocator observations, not numerical-value evidence.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,inspect,time

GRAPHING_SHA='62f1b07f581b4c996525e5fa1b17796eee6aba0d39bf67de55dfb3e1ae69f4d4'
POLICY='release_unused_cache_after_original_warmup_sync_before_private_graph_capture'

def install(torch,graphing,record=None):
    assert hashlib.sha256(Path(graphing.__file__).read_bytes()).hexdigest()==GRAPHING_SHA
    original=graphing.GraphCache._capture
    assert inspect.getfile(original)==str(Path(graphing.__file__))
    assert original.__qualname__=='GraphCache._capture' and graphing.torch is torch
    WARMUP=graphing.WARMUP;assert WARMUP==2
    releases=[]

    def release_unused(owner,key):
        # Original _capture has already synchronized this device. This runs
        # before capture_begin, never from within the captured fn or warmup.
        assert not torch.cuda.is_current_stream_capturing()
        before_allocated=torch.cuda.memory_allocated(owner.device)
        before_reserved=torch.cuda.memory_reserved(owner.device)
        start=time.perf_counter();torch.cuda.empty_cache()
        after_allocated=torch.cuda.memory_allocated(owner.device)
        after_reserved=torch.cuda.memory_reserved(owner.device)
        row=dict(sequence=len(releases),device=str(owner.device),key_type=type(key).__name__,
            existing_graph_count=len(owner._segs),completed_captures=owner.captures,
            allocated_before=before_allocated,allocated_after=after_allocated,
            reserved_before=before_reserved,reserved_after=after_reserved,
            allocated_drop_bytes=before_allocated-after_allocated,
            released_reserved_bytes=before_reserved-after_reserved,elapsed_s=time.perf_counter()-start)
        releases.append(row)
        if record is not None:record(row)
        assert after_allocated<=before_allocated,'Unused-cache release increased allocated bytes'
        assert after_reserved<=before_reserved,'Unused-cache release increased reserved bytes'

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
            release_unused(self, key)
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

    graphing.GraphCache._capture=_capture
    def restore():
        assert graphing.GraphCache._capture is _capture
        graphing.GraphCache._capture=original
    return SimpleNamespace(policy=POLICY,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_graphing_sha256=GRAPHING_SHA,releases=releases,restore=restore)
