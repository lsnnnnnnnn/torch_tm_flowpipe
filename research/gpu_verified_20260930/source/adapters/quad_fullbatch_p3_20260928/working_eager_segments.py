"""Candidate: five pure working-engine regions use their original eager fn.

Install over qualified working-prune/postwarm hooks, before a fresh engine.
Bracket each advance/prune with begin_step/finish_step. Nested warmup/capture
and every other graph call keep the original dispatch. No graph pool sharing.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,inspect

GRAPHING_SHA='62f1b07f581b4c996525e5fa1b17796eee6aba0d39bf67de55dfb3e1ae69f4d4'
WORKING_PRUNE_SHA='3a174e02724ee3166eff9f1b330c137e0c1a4b51f64a6162ea2a6ed96814e21f'
POSTWARM_SHA='259c479a113bb1c905da7c49ae888ba50002ec84c06acfbb253bd9b45edfd5ee'
POLICY='active_fresh_working_engine_five_regions_original_fn_eager_other_graphs_unchanged'
FAMILIES=('endpoint','sr_linear_prepare','combine_linear','normalize','glue:emit')

def family(key):
    if not isinstance(key,tuple) or not key:return None
    if key[0] in FAMILIES[:4]:return key[0]
    if len(key)>1 and key[:2]==('glue','emit'):return 'glue:emit'
    return None

def _bind(torch,graphing,record):
    """Dispatch only; separate from source admission for the CPU fake check."""
    raw_run=graphing.GraphCache.run;raw_capture=graphing.GraphCache._capture
    active=owner=None;depth=0;rows=[];step_counts=None
    counters=dict(begun_steps=0,finished_steps=0,eager_calls=0,forwarded_calls=0,
        nested_forwarded_calls=0,input_version_checks=0,by_family={k:0 for k in FAMILIES})
    def capture(cache,*args,**kwargs):
        nonlocal depth
        depth+=1
        try:return raw_capture(cache,*args,**kwargs)
        finally:depth-=1
    def run(cache,key,fn,inputs,keepalive=()):
        # Do not inspect input versions or alter nested original dispatch.
        if depth or torch.cuda.is_current_stream_capturing():
            counters['nested_forwarded_calls']+=1
            return raw_run(cache,key,fn,inputs,keepalive)
        kind=family(key)
        if active is None or cache is not getattr(active,'_graphs',None) or kind is None:
            counters['forwarded_calls']+=1
            return raw_run(cache,key,fn,inputs,keepalive)
        assert cache.enabled and key not in cache._segs,'Fresh eager region must have no existing captured graph'
        assert inputs and all(isinstance(t,torch.Tensor) and t.is_cuda for t in inputs)
        assert inputs[0].shape[0]==1024 and inputs[0].dtype==torch.float64
        versions=[t._version for t in inputs]
        # Preserve fn, args, strides and result; do not call the old used-key hook.
        try:value=fn(*inputs)
        finally:assert [t._version for t in inputs]==versions,'Pure region changed a caller input'
        counters['input_version_checks']+=1;counters['eager_calls']+=1
        counters['by_family'][kind]+=1;step_counts[kind]+=1
        if record is not None:record(dict(event='working_eager_returned',step=counters['begun_steps'],
            family=kind,input_shapes=[list(t.shape) for t in inputs],input_versions_unchanged=True))
        return value
    def begin_step(eng):
        nonlocal active,owner,step_counts
        assert active is None and depth==0 and not torch.cuda.is_current_stream_capturing()
        assert (eng.tables.n,eng.tables.k)==(16,3) and str(eng.device).startswith('cuda')
        if owner is None:
            assert getattr(eng,'_graphs',None) is None,'Install/use before the first working-engine graph'
            owner=eng
        else:assert eng is owner,'A binding admits one fresh working engine only'
        active=eng;step_counts={k:0 for k in FAMILIES};counters['begun_steps']+=1
    def finish_step(eng):
        nonlocal active,step_counts
        assert active is eng and owner is eng and depth==0 and not torch.cuda.is_current_stream_capturing()
        assert step_counts['endpoint']==step_counts['sr_linear_prepare']==step_counts['normalize']==step_counts['glue:emit']==1
        assert step_counts['combine_linear'] in (0,1)
        row=dict(step=counters['begun_steps'],eager_calls=sum(step_counts.values()),by_family=dict(step_counts),
            input_versions_unchanged=True,other_graph_calls_forwarded=True,nested_capture_calls_forwarded=True)
        counters['finished_steps']+=1;rows.append(row);active=None;step_counts=None
        if record is not None:record(dict(event='working_eager_step_finished',**row))
        return row
    graphing.GraphCache.run=run;graphing.GraphCache._capture=capture
    def restore():
        nonlocal active,owner,step_counts
        assert depth==0 and graphing.GraphCache.run is run and graphing.GraphCache._capture is capture
        graphing.GraphCache.run=raw_run;graphing.GraphCache._capture=raw_capture
        active=owner=step_counts=None
    return SimpleNamespace(policy=POLICY,counters=counters,rows=rows,begin_step=begin_step,
        finish_step=finish_step,restore=restore)

def install(torch,graphing,record=None):
    assert hashlib.sha256(Path(graphing.__file__).read_bytes()).hexdigest()==GRAPHING_SHA
    assert graphing.torch is torch and graphing.WARMUP==2
    raw=graphing.GraphCache.run;cap=graphing.GraphCache._capture
    assert hashlib.sha256(Path(inspect.getfile(raw)).read_bytes()).hexdigest()==WORKING_PRUNE_SHA
    assert raw.__qualname__=='install.<locals>.run'
    assert hashlib.sha256(Path(inspect.getfile(cap)).read_bytes()).hexdigest()==POSTWARM_SHA
    assert cap.__qualname__=='install.<locals>._capture'
    binding=_bind(torch,graphing,record)
    binding.source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    binding.original_graphing_sha256=GRAPHING_SHA
    binding.wrapped_working_prune_sha256=WORKING_PRUNE_SHA
    binding.wrapped_postwarm_sha256=POSTWARM_SHA
    return binding
