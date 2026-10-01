"""Evict only unused working-engine graph entries after a complete prune.

The caller brackets the original advance with begin_step and calls finish_step
only after original prune returns. No state/SR/support/plan/tape is modified.
Actual live-state/SR byte and storage checks belong to the paired runtime gate.
"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import inspect,time

GRAPHING_SHA='62f1b07f581b4c996525e5fa1b17796eee6aba0d39bf67de55dfb3e1ae69f4d4'
POLICY='after_complete_prune_evict_working_graph_keys_unused_in_current_advance'

def install(torch,graphing,record=None):
    assert Path(graphing.__file__).is_file()
    raw=graphing.GraphCache.run
    assert inspect.getfile(raw)==str(Path(graphing.__file__)) and raw.__qualname__=='GraphCache.run'
    assert graphing.torch is torch
    active=None;target=None;used=set();initial=set();rows=[]
    counters=dict(begun_steps=0,finished_steps=0,working_run_calls=0,evicted_entries=0)

    def run(cache,key,fn,inputs,keepalive=()):
        nonlocal target
        if active is not None and cache is getattr(active,'_graphs',None):
            if target is None:target=cache
            assert cache is target and cache.enabled
            used.add(key);counters['working_run_calls']+=1
        return raw(cache,key,fn,inputs,keepalive)

    def begin_step(eng):
        nonlocal active,target,initial
        assert active is None,'Previous advance must finish prune or abort the process'
        assert eng.tables.n==16 and eng.tables.k==3 and str(eng.device).startswith('cuda')
        active=eng;target=getattr(eng,'_graphs',None)
        initial=set() if target is None else set(target._segs)
        used.clear();counters['begun_steps']+=1

    def category(key):
        if isinstance(key,tuple) and key:
            if key[0]=='glue':return 'glue:'+str(key[1])
            if isinstance(key[0],str):return key[0]
            if len(key)>1 and key[1]=='refine':return 'refine'
        return type(key).__name__

    def memory(device):
        free,total=torch.cuda.mem_get_info(device)
        return dict(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),
            device_free=free,device_total=total,device_used=total-free)

    def finish_step(eng):
        nonlocal active,target
        assert active is eng and target is getattr(eng,'_graphs',None) and target is not None
        assert target.enabled and used and not torch.cuda.is_current_stream_capturing()
        current=set(target._segs)
        assert used<=current,'Working graph entries disappeared during the advance'
        assert (current-initial)<=used,'Untracked direct working capture is unsupported'
        stale=current-used
        # Never save entry/statics/output/keepalive objects in receipts. Holding
        # those objects here would keep exactly the private pools being evicted.
        groups=dict(sorted(Counter(category(key) for key in stale).items()))
        captures,hits=target.captures,target.hits
        stream_id=id(target._capture_stream)
        start=time.perf_counter()
        if stale:torch.cuda.synchronize(target.device)
        before=memory(target.device)
        for key in stale:del target._segs[key]
        if stale:torch.cuda.empty_cache()
        after=memory(target.device)
        assert set(target._segs)==used and target is eng._graphs
        assert id(target._capture_stream)==stream_id and (target.captures,target.hits)==(captures,hits)
        row=dict(sequence=len(rows),nkeys_before=len(current),used_keys=len(used),evicted_entries=len(stale),
            nkeys_after=len(target._segs),evicted_categories=groups,working_owner_id=id(eng),
            graph_owner_id=id(target),capture_stream_id=stream_id,captures=captures,hits=hits,
            before=before,after=after,elapsed_s=time.perf_counter()-start,
            synchronized_and_empty_cache=bool(stale),validation_and_weighted_untouched=True)
        counters['finished_steps']+=1;counters['evicted_entries']+=len(stale)
        rows.append(row)
        active=None;target=None;used.clear();initial.clear()
        if record is not None:record(row)
        return row

    graphing.GraphCache.run=run
    def restore():
        nonlocal active,target
        assert graphing.GraphCache.run is run
        graphing.GraphCache.run=raw
        active=None;target=None;used.clear();initial.clear()
    return SimpleNamespace(policy=POLICY,source_path=str(Path(__file__)),
        original_graphing_path=str(Path(graphing.__file__)),begin_step=begin_step,finish_step=finish_step,
        counters=counters,rows=rows,restore=restore)
