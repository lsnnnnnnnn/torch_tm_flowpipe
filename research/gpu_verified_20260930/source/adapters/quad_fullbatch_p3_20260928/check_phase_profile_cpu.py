"""CPU dispatch check of actual frozen GraphCache with a fake CUDA surface.

No CUDA arithmetic, concurrency, timing accuracy or trajectory qualification.
"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(a, graphing_path):
    assert sha(graphing_path) == a.PINS['graphing']
    g = load('phase_cpu_actual_frozen_graphing', graphing_path)
    state = NS(capturing=False, current=None, records=0, completed=0, events=0,
               syncs=0, nested_calls=0, queries=0)

    class Tensor:
        def __init__(self, value=17):
            self.value = value; self.shape = (1024, 16, 3); self.is_cuda = True
        def clone(self): return Tensor(self.value)
        def copy_(self, other): self.value = other.value; return self

    class Stream:
        next_id = 1
        def __init__(self, **kwargs):
            self.cuda_stream = Stream.next_id; Stream.next_id += 1
        def wait_stream(self, other): pass

    default = Stream(); state.current = default
    class Context:
        def __init__(self, stream=None): self.stream = stream
        def __enter__(self):
            self.previous = state.current
            if self.stream is not None: state.current = self.stream
        def __exit__(self, *args): state.current = self.previous

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing and state.current is default and not state.capturing
            state.events += 1; self.serial = None
        def record(self, stream):
            assert state.current is default and stream is default and not state.capturing
            state.records += 1; self.serial = state.records
        def query(self):
            state.queries += 1
            assert not state.capturing
            return self.serial is not None and self.serial <= state.completed
        def elapsed_time(self, other):
            assert self.serial <= other.serial <= state.completed
            return float(other.serial-self.serial)

    class Graph:
        def capture_begin(self): assert not state.capturing; state.capturing = True
        def capture_end(self): assert state.capturing; state.capturing = False
        def replay(self): assert not state.capturing

    def sync(*args):
        assert not state.capturing
        state.syncs += 1; state.completed = state.records
    t = NS(Tensor=Tensor, cuda=NS(Event=Event, Stream=Stream, CUDAGraph=Graph,
        current_stream=lambda *args: state.current, is_current_stream_capturing=lambda: state.capturing,
        stream=lambda s: Context(s), device=lambda *args: Context(), synchronize=sync))
    g.torch = t
    def cache():
        c = g.GraphCache.__new__(g.GraphCache)
        c.device='cuda:0'; c.enabled=True; c.pool=None; c._capture_stream=None
        c._segs={}; c.hits=c.captures=0
        return c

    token=object(); inputs=Tensor(); calls=[]
    def original(*args, **kwargs): calls.append(('original', args, kwargs)); return token
    se=NS(**{name:original for name in a.SE_PHASES})
    wv=NS(**{name:original for name in a.WV_PHASES})
    class Ledger:
        propagate=original
        finish_step=original
    host=NS(HostFactorLedger=Ledger)
    c=cache(); owner=object(); reports=[]
    raw=(g.GraphCache.run,g.GraphCache._capture,se.compose_s,Ledger.propagate)
    b=a._bind(t,g,se,wv,host,reports.append)
    assert se.compose_s(inputs, label='inactive') is token and state.events==0
    negative=[]
    def reject(name, call):
        try: call()
        except AssertionError: negative.append(name)
        else: raise AssertionError('Invalid operation accepted: '+name)
    b.begin_step(1,owner)
    reject('overlap',lambda:b.begin_step(1,owner))
    def nested(value):
        state.nested_calls += 1
        assert se.compose_s(value) is token
        assert wv.refine_accepted(value) is token
        return value
    # Real GraphCache invokes nested twice on the side stream and once in
    # capture. No profiling Event creation/record may occur in any of them.
    assert c.run(('validpost_checked','miss'),nested,[inputs]).value == inputs.value
    assert state.nested_calls==3 and state.events==2 and state.records==1
    assert state.syncs==1  # Original frozen _capture sync only.
    assert c.run(('validpost_checked','miss'),nested,[inputs]).value == inputs.value
    assert state.nested_calls==3 and state.events==4 and state.records==3
    reject('finish_without_sync_declaration',lambda:b.finish_step(1,original_sync_completed=False))
    reject('unfinished_event',lambda:b.finish_step(1,original_sync_completed=True))
    sync(); before=state.syncs
    first=b.finish_step(1,original_sync_completed=True,advance_wall_s=.5)
    assert state.syncs==before and len(first['outer_phases'])==2
    assert first['outer_phases'][0]['gpu_span_omitted_reason']=='capture_occurred'
    assert first['outer_phases'][0]['gpu_stream_ms'] is None
    assert first['outer_phases'][1]['gpu_stream_ms']==1.
    assert first['capture_count']==1 and len(first['capture_calls'])==1
    assert first['graph_counters'][str(id(c))]==dict(calls=2,captures=1,hits=2)
    assert b.counters['capture_suppressed_hooks']==6

    # Outer weighted phase owns the nested ordinary phase; count, never time twice.
    b.restore()
    def outer(value):
        assert se.compose_s(value) is token
        return token
    wv.refine_accepted=outer
    b=a._bind(t,g,se,wv,host,reports.append);b.begin_step(2,owner)
    assert wv.refine_accepted(inputs) is token
    sync(); second=b.finish_step(2,original_sync_completed=True)
    assert [r['phase'] for r in second['outer_phases']]==['weighted_accepted']
    assert second['nested_call_counts']=={'compose':1}

    # External capture, as distinct from the _capture sentinel, also bypasses.
    b.begin_step(3,owner); events=state.events; queries=state.queries
    state.capturing=True
    try: assert se.compose_s(inputs) is token
    finally: state.capturing=False
    assert (state.events,state.queries)==(events,queries)
    sync(); third=b.finish_step(3,original_sync_completed=True)
    assert third['outer_phases']==[]
    reject('different_owner',lambda:b.begin_step(4,object()))

    b.restore()
    def changed_stream(*args):
        state.current=Stream()
        return token
    se.compose_s=changed_stream
    b=a._bind(t,g,se,wv,host,reports.append);b.begin_step(4,owner)
    assert se.compose_s(inputs) is token
    state.current=default
    sync(); changed=b.finish_step(4,original_sync_completed=True)
    assert changed['outer_phases'][0]['gpu_span_omitted_reason']=='stream_changed'
    assert changed['outer_phases'][0]['gpu_stream_ms'] is None

    # Exceptions are forwarded unchanged; abort releases pending events without
    # querying them or introducing a synchronize, and all hooks restore.
    b.restore(); error=ValueError('original failure')
    def failing(*args): raise error
    se.compose_s=failing
    b=a._bind(t,g,se,wv,host,reports.append);b.begin_step(5,owner)
    try: se.compose_s(inputs)
    except ValueError as caught: assert caught is error
    else: raise AssertionError('Original exception lost')
    q,s=state.queries,state.syncs
    b.abort_step(); assert (state.queries,state.syncs)==(q,s)
    b.restore()
    assert g.GraphCache.run is raw[0] and g.GraphCache._capture is raw[1]
    assert se.compose_s is failing and Ledger.propagate is raw[3]
    # JSON serialization recursively fails if a receipt retains tensors/events/
    # streams/functions; output is only metadata, counters and CPU scalars.
    json.dumps(reports,allow_nan=False)
    return dict(real_frozen_GraphCache_used=True,warmup_calls=2,capture_calls=1,
        no_events_or_queries_in_warmup_or_capture=True,original_return_objects_preserved=True,
        original_exception_preserved=True,mutually_exclusive_phases_and_nested_counts=True,
        deferred_queries_require_completed_events=True,profiler_sync_calls=0,
        changed_stream_span_omitted=True,
        all_hooks_restored=True,receipts_json_only=True,negative_cases=negative,
        fake_cuda_reports=reports)


def main(args):
    a=load('candidate_phase_profile',HERE/'phase_profile.py')
    result=dict(status='passed',device='cpu',script_sha256=sha(__file__),adapter_sha256=sha(HERE/'phase_profile.py'),
        **check(a,args.graphing_source),CUDA_tested=False,
        scope='Actual frozen GraphCache control flow with fake CUDA, not numeric or CUDA timing qualification. Actual layered installation and complete selective40 byte pairing remain required.')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fake_cuda_reports'}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphing-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
