"""Isolated original early weighted refinement with 128-row graph scratch.

Only active refine_accepted graph calls use the smaller block/padding.
Ordinary validation, eager failed-lane validation, SR and all maths are intact.
"""
from pathlib import Path
from types import SimpleNamespace
import inspect

POLICY='original_early_weighted_rounds_and_attempts_fixed128_graph_scratch'


def pad128(torch,value):
    lanes=value.shape[0];assert 0<lanes<=128
    return torch.cat((value,value[-1:].expand(128-lanes,*value.shape[1:])),dim=0) if lanes<128 else value


def install(torch,wv,se):
    assert Path(wv.__file__).is_file() and Path(se.__file__).is_file()
    raw_evaluate,raw_refine=wv._evaluate_map,wv.refine_accepted
    assert inspect.getfile(raw_evaluate)==inspect.getfile(raw_refine)==str(Path(wv.__file__))
    assert raw_evaluate.__name__=='_evaluate_map' and raw_refine.__name__=='refine_accepted'
    active=False;counters=dict(refine_calls=0,graph_map_calls=0,target_rows=0,padded_rows=0,chunk_size=128)
    def evaluate(code,coefficients,point,candidate,plan,eng,cutoff,use_graph):
        if not active or not use_graph or not candidate.is_cuda:
            return raw_evaluate(code,coefficients,point,candidate,plan,eng,cutoff,use_graph)
        lanes=candidate.shape[0];assert 0<lanes<=128
        assert coefficients.shape[0]==point.shape[0]==lanes
        coefficients,point,candidate=(pad128(torch,v) for v in (coefficients,point,candidate))
        from flowstar_gpu.graphing import GraphCache
        cache=getattr(eng,'_weighted_graphs',None)
        if cache is None:cache=eng._weighted_graphs=GraphCache(str(candidate.device))
        key=(se._code_serial(code),id(plan),tuple(coefficients.shape),tuple(point.shape),tuple(candidate.shape),candidate.dtype,cutoff)
        if key not in cache._segs and len(cache._segs)>=1:cache._segs.pop(next(iter(cache._segs)))
        def fn(a,b,c):return wv._map(code,a,b,c,plan,eng,cutoff)
        output=cache.run(key,fn,[coefficients,point,candidate],keepalive=(code,plan))
        counters['graph_map_calls']+=1;counters['target_rows']+=lanes;counters['padded_rows']+=128-lanes
        return tuple(value[:lanes].clone() for value in output)
    def refine(code,x,sup,initial,initial_sup,var_sups,original,current,eligible,eng,cutoff,
               *,rounds=2,chunk_size=512,use_graph=True,trace=None):
        nonlocal active
        if not use_graph or not x.is_cuda:
            return raw_refine(code,x,sup,initial,initial_sup,var_sups,original,current,eligible,eng,cutoff,
                rounds=rounds,chunk_size=chunk_size,use_graph=use_graph,trace=trace)
        assert not active and chunk_size==512,'Initial scope is the original early-weighted default only'
        active=True
        try:
            result=raw_refine(code,x,sup,initial,initial_sup,var_sups,original,current,eligible,eng,cutoff,
                rounds=rounds,chunk_size=128,use_graph=use_graph,trace=trace)
            counters['refine_calls']+=1;return result
        finally:active=False
    wv._evaluate_map,wv.refine_accepted=evaluate,refine
    def restore():
        assert not active and wv._evaluate_map is evaluate and wv.refine_accepted is refine
        wv._evaluate_map,wv.refine_accepted=raw_evaluate,raw_refine
    return SimpleNamespace(policy=POLICY,source_path=str(Path(__file__)),
        weighted_path=str(Path(wv.__file__)),sparse_exec_path=str(Path(se.__file__)),
        counters=counters,restore=restore)


def check_local():
    """Padding/ownership only, not an original numeric/GPU parity check."""
    import torch
    eq=lambda a,b:torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    for rows in [1,2,10,127,128]:
        x=torch.arange(rows*6,dtype=torch.float64).reshape(rows,2,3);x[0,0,0]=-0.0
        saved=x.clone();padded=pad128(torch,x)
        assert padded.shape==(128,2,3) and eq(padded[:rows],saved)
        assert rows==128 or eq(padded[rows:],x[-1:].expand(128-rows,2,3).contiguous())
        owned=padded[:rows].clone();padded.zero_();assert eq(owned,saved)
    for rows in [0,129]:
        try:pad128(torch,torch.zeros(rows,2,3))
        except AssertionError:pass
        else:raise AssertionError('Bad row count accepted')
    return dict(status='passed_CPU_padding_only',row_counts=[1,2,10,127,128],target_bytes_preserved=True,
        last_row_padding_bytes_equal=True,owned_clones_survive_replay=True,invalid_counts_rejected=[0,129],
        actual_weighted_map_run=False,actual_refinement_run=False,CUDA_run=False)


if __name__=='__main__':
    import json
    print(json.dumps(check_local()))
