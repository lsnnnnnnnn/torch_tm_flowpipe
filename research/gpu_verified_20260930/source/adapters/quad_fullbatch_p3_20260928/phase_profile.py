"""Read-only outer-phase diagnostic for the frozen selective40 execution.

Install last; restore first. begin_step before the numerical advance and
finish_step at observe entry, AFTER its original end-of-advance sync. This
adapter never synchronizes. CUDA spans are stream elapsed times, not kernel
sums. Capturing stages have CPU wall time only. Nested calls have counts only.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect
import time

PINS = {
    'graphing': '62f1b07f581b4c996525e5fa1b17796eee6aba0d39bf67de55dfb3e1ae69f4d4',
    'se': '56e590796dbae42685b0331c7f41b53b62a3f73ed9be74e22738493820353ddf',
    'wv': '8460ab06bba4beb42e07eb3f968a647a4ba492f267bdde806a96e1ded5466299',
    'host': '1a7d9bbf0f16a4f3726b75a27ff938ff87b918141a5f0fe1504f26198dc4035b',
}
EAGER_SHA = 'c4fb7b098a10ab98d9e9250fc88460ade130225ef710ea5734522df03ca0453b'
VALID_SHA = '6b97a9ed19b8433b79be15f97381241bc9a79c071b90fa4a81f12c4d216dcc40'
WEIGHTED_SHA = 'ae343c546e2253896142f88c39e1df5e43561498aaf7540e722a1bb6d66d409b'
POLICY = 'outer_phase_cpu_wall_same_stream_events_after_original_sync_no_nested_timing'
SE_PHASES = {'compose_s': 'compose', '_structural_picard': 'structural_picard',
    '_graphed_valid': 'ordinary_validation', '_refine_dispatch': 'ordinary_refinement',
    '_retry_self_map': 'self_map_retry'}
WV_PHASES = {'refine_accepted': 'weighted_accepted', 'validate_failed': 'weighted_failed',
    'validate_recentered': 'weighted_recentered'}
GRAPH_PHASES = ('endpoint', 'sr_linear_prepare', 'combine_linear', 'normalize', 'validpost_checked')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bind(torch, graphing, se, wv, host, record=None):
    """Dispatch core; fake CUDA check does not imply CUDA qualification."""
    active = None
    owner = None
    capture_depth = phase_depth = 0
    rows, pending, capture_rows, completed_rows = [], [], [], []
    nested, graph_counts = {}, {}
    captures = 0
    hooks = []
    counters = dict(begun_steps=0, finished_steps=0, aborted_steps=0,
                    capture_suppressed_hooks=0, restored=False)

    def suppressed():
        # Short circuit before querying CUDA during warmup/capture.
        return active is None or capture_depth or torch.cuda.is_current_stream_capturing()

    def stream_id():
        stream = torch.cuda.current_stream()
        return stream, int(stream.cuda_stream)

    def shapes(args):
        return [list(v.shape) for v in args if isinstance(v, torch.Tensor)]

    def timed(label, fn, args, kwargs, metadata=None):
        nonlocal phase_depth
        if suppressed():
            if capture_depth: counters['capture_suppressed_hooks'] += 1
            return fn(*args, **kwargs)
        if phase_depth:
            nested[label] = nested.get(label, 0) + 1
            return fn(*args, **kwargs)
        stream, sid = stream_id()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        before = captures
        started = time.perf_counter()
        phase_depth += 1
        success = False
        try:
            result = fn(*args, **kwargs)
            success = True
            return result
        finally:
            elapsed = time.perf_counter() - started
            phase_depth -= 1
            end_stream, eid = stream_id()
            reason = ('call_raised' if not success else 'capture_occurred' if captures != before
                      else 'stream_changed' if sid != eid else None)
            pair = None
            if reason is None:
                end_event.record(end_stream)
                pair = (start_event, end_event)
            row = dict(phase=label, cpu_call_wall_s=elapsed, gpu_stream_ms=None,
                       gpu_span_omitted_reason=reason, start_stream_id=sid, end_stream_id=eid,
                       captures_during_call=captures-before, input_shapes=shapes(args),
                       metadata={} if metadata is None else metadata, scope='mutually_exclusive_outer_phase')
            rows.append(row)
            pending.append(pair)

    def patch(obj, name, replacement):
        old = getattr(obj, name)
        hooks.append((obj, name, old, replacement))
        setattr(obj, name, replacement)

    raw_run = graphing.GraphCache.run
    raw_capture = graphing.GraphCache._capture

    def capture(cache, key, fn, inputs, keepalive=()):
        nonlocal capture_depth, captures
        outer = active is not None and capture_depth == 0 and not torch.cuda.is_current_stream_capturing()
        started = time.perf_counter() if outer else None
        capture_depth += 1
        try:
            return raw_capture(cache, key, fn, inputs, keepalive)
        finally:
            capture_depth -= 1
            if outer:
                captures += 1
                # CPU scalars only. No event, tensor read, callback or IO in capture.
                capture_rows.append(dict(cache_id=id(cache), key_id=id(key),
                    cpu_capture_wall_s=time.perf_counter()-started))

    def run(cache, key, fn, inputs, keepalive=()):
        if suppressed():
            if capture_depth: counters['capture_suppressed_hooks'] += 1
            return raw_run(cache, key, fn, inputs, keepalive)
        first = key[0] if isinstance(key, tuple) and key else None
        label = first if isinstance(first, str) and first in GRAPH_PHASES else None
        if isinstance(key, tuple) and key[:2] == ('glue', 'emit'): label = 'glue:emit'
        before = (cache.captures, cache.hits)
        metadata = dict(cache_id=id(cache), key_id=id(key),
            key_family=first if isinstance(first, str) else type(first).__name__,
            existing_entry=key in cache._segs, input_shapes=shapes(inputs))
        try:
            if label is not None:
                return timed(label, raw_run, (cache, key, fn, inputs, keepalive), {}, metadata)
            return raw_run(cache, key, fn, inputs, keepalive)
        finally:
            counts = graph_counts.setdefault(str(id(cache)), dict(calls=0, captures=0, hits=0))
            counts['calls'] += 1
            counts['captures'] += cache.captures-before[0]
            counts['hits'] += cache.hits-before[1]

    patch(graphing.GraphCache, '_capture', capture)
    patch(graphing.GraphCache, 'run', run)
    for obj, mapping in [(se, SE_PHASES), (wv, WV_PHASES),
                         (host.HostFactorLedger, {'propagate': 'sr_host_propagate', 'finish_step': 'sr_host_finish'})]:
        for name, label in mapping.items():
            old = getattr(obj, name)
            def wrapper(*args, _old=old, _label=label, **kwargs):
                return timed(_label, _old, args, kwargs)
            patch(obj, name, wrapper)

    def begin_step(step, eng):
        nonlocal active, owner, captures
        assert active is None and capture_depth == phase_depth == 0
        assert not torch.cuda.is_current_stream_capturing()
        assert type(step) is int and step > 0
        if owner is None: owner = eng
        assert eng is owner
        assert not rows and not pending and not capture_rows and not nested and not graph_counts
        active = step
        captures = 0
        counters['begun_steps'] += 1

    def clear():
        nonlocal active
        rows.clear(); pending.clear(); capture_rows.clear(); nested.clear(); graph_counts.clear()
        active = None

    def finish_step(step, *, original_sync_completed, advance_wall_s=None):
        assert active == step and capture_depth == phase_depth == 0
        assert original_sync_completed is True and not torch.cuda.is_current_stream_capturing()
        # query is nonblocking: never wait or synchronize from this adapter.
        for row, pair in zip(rows, pending):
            if pair is not None:
                start, end = pair
                assert start.query() and end.query(), 'Original step sync must complete all events'
                row['gpu_stream_ms'] = float(start.elapsed_time(end))
        result = dict(step=step, policy=POLICY, advance_wall_s=advance_wall_s,
            outer_phases=list(rows), capture_calls=list(capture_rows), graph_counters=dict(graph_counts),
            nested_call_counts=dict(nested), capture_count=captures,
            original_sync_completed=True, profiler_added_synchronizations=0,
            note='Outer calls are mutually exclusive; nested counts and capture wall belong inside them and must not be added. CUDA spans include stream idle gaps and are not pure kernel sums; capture-containing spans are omitted. Uncovered advance work remains.')
        clear()
        counters['finished_steps'] += 1
        completed_rows.append(result)
        if record is not None: record(result)
        return result

    def abort_step():
        assert capture_depth == phase_depth == 0
        if active is not None:
            clear()
            counters['aborted_steps'] += 1

    def restore():
        nonlocal owner
        assert active is None and capture_depth == phase_depth == 0
        for obj, name, old, replacement in hooks:
            assert getattr(obj, name) is replacement, name
        for obj, name, old, replacement in reversed(hooks):
            setattr(obj, name, old)
        owner = None
        counters['restored'] = True

    return SimpleNamespace(policy=POLICY, begin_step=begin_step, finish_step=finish_step,
        abort_step=abort_step, restore=restore, counters=counters, rows=completed_rows)


def install(torch, graphing, se, wv, host, record=None):
    for name, obj in [('graphing', graphing), ('se', se), ('wv', wv), ('host', host)]:
        assert sha(obj.__file__) == PINS[name], name
    assert graphing.torch is torch and graphing.WARMUP == 2
    expected = [(graphing.GraphCache.run, EAGER_SHA), (graphing.GraphCache._capture, EAGER_SHA),
                (se._graphed_valid, VALID_SHA), (wv.refine_accepted, WEIGHTED_SHA)]
    expected += [(getattr(se, name), PINS['se']) for name in SE_PHASES if name != '_graphed_valid']
    expected += [(getattr(wv, name), PINS['wv']) for name in WV_PHASES if name != 'refine_accepted']
    expected += [(getattr(host.HostFactorLedger, name), PINS['host']) for name in ['propagate', 'finish_step']]
    for fn, digest in expected:
        assert sha(inspect.getfile(fn)) == digest, fn.__qualname__
    binding = _bind(torch, graphing, se, wv, host, record)
    binding.source_sha256 = sha(__file__)
    binding.source_pins = dict(PINS, eager=EAGER_SHA, valid128=VALID_SHA, weighted128=WEIGHTED_SHA)
    return binding
