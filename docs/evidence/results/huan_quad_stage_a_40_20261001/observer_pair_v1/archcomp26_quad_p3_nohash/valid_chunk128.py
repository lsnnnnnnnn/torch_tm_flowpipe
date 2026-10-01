"""Candidate: fixed full1024 validated-RHS scratch chunks; not CUDA-qualified.

Only se._graphed_valid is replaced. Original numeric dispatcher and spec remain.
Never install after a full-batch vd graph exists for the same engine/spec.
Use fresh owners, and discard those owners before restoring full-batch dispatch.
"""
from pathlib import Path
from types import SimpleNamespace
import inspect

SPARSE_EXEC_SHA='56e590796dbae42685b0331c7f41b53b62a3f73ed9be74e22738493820353ddf'
POLICY='full1024_validated_RHS_original_dispatch_fixed128_owned_full_outputs'


def collect(torch,raw,args):
    spec,code,x,xrem,bad,eng,cutoff,tabs=args
    assert x.ndim==3 and x.shape[0]==1024 and x.dtype==torch.float64
    n=x.shape[1];assert xrem.shape==(1024,n,2) and xrem.dtype==x.dtype and xrem.device==x.device
    assert bad.shape==(1024,) and bad.dtype==torch.bool and bad.device==x.device
    expected=[(1024,n,spec.sup_out_union.size,2),(1024,n,2),(1024,code.n_cache,2),(1024,max(code.n_strict,1),2),(1024,)]
    versions=[v._version for v in (x,xrem,bad)];outputs=None
    for start in range(0,1024,128):
        stop=start+128
        part=raw(spec,code,x[start:stop],xrem[start:stop],bad[start:stop],eng,cutoff,tabs)
        assert len(part)==5
        for i,value in enumerate(part):
            assert tuple(value.shape)==(128,*expected[i][1:]) and value.device==x.device
            assert value.dtype==(torch.bool if i==4 else torch.float64)
        if outputs is None:outputs=tuple(torch.empty(shape,dtype=part[i].dtype,device=x.device) for i,shape in enumerate(expected))
        # Graph outputs are static aliases. Copy now, before another replay.
        for target,value in zip(outputs,part):target[start:stop].copy_(value)
        del part,target,value
    assert [v._version for v in (x,xrem,bad)]==versions,'Original graph/tape inputs must stay immutable'
    return outputs


def install(torch,se):
    path=Path(se.__file__);assert path.is_file()
    raw=se._graphed_valid;assert inspect.getfile(raw)==str(path) and raw.__name__=='_graphed_valid'
    counters=dict(calls=0,block_calls=0,registered_specs=0,full_batch=1024,chunk_size=128)
    def wrapper(spec,code,x,xrem,bad,eng,cutoff,tabs):
        assert x.is_cuda and x.shape[0]==1024 and x.shape[1]==16 and eng.tables.k==4
        assert code.order==spec.k==4 and eng.tables.n==16 and len(spec.out_slots)==16
        # Eager exec_valid_s mutates caller bad in place. Graph/tape dispatch
        # owns its mutable bad buffer; reject eager before any numeric call.
        gc=se.graph_cache(eng,str(x.device));assert gc.enabled,'Enabled CUDA graphs required; no eager fallback'
        key=('vd',id(spec))
        owners=getattr(eng,'_codex_valid128_owners',None)
        if owners is None:owners={};eng._codex_valid128_owners=owners
        stamp=(id(code),float(cutoff),id(tabs),tuple(x.shape[1:]),x.dtype,x.device)
        if id(spec) not in owners:
            assert key not in gc._segs,'Existing vd graph has an unqualified batch shape; use a fresh owner'
            owners[id(spec)]=(spec,stamp);counters['registered_specs']+=1
        else:assert owners[id(spec)][0] is spec and owners[id(spec)][1]==stamp
        if key in gc._segs:
            statics=gc._segs[key][0]
            assert [tuple(v.shape) for v in statics]==[(128,*x.shape[1:]),(128,16,2),(128,)]
        output=collect(torch,raw,(spec,code,x,xrem,bad,eng,cutoff,tabs))
        assert eng._graphs is gc and gc.enabled
        if key in gc._segs:
            assert [tuple(v.shape) for v in gc._segs[key][0]]==[(128,*x.shape[1:]),(128,16,2),(128,)]
        counters['calls']+=1;counters['block_calls']+=8
        return output
    se._graphed_valid=wrapper
    def restore():
        assert se._graphed_valid is wrapper;se._graphed_valid=raw
    return SimpleNamespace(policy=POLICY,source_path=str(Path(__file__)),
        sparse_exec_path=str(path),counters=counters,restore=restore)


def check_local():
    """CPU assembly/alias fixture only; no actual engine or GPU qualification."""
    import torch
    torch.set_num_threads(1)
    x=torch.arange(1024*2*3,dtype=torch.float64).reshape(1024,2,3)
    x[73,1,2]=float('nan')  # Must be copied, not cleared to manufacture success.
    xr=torch.zeros(1024,2,2,dtype=torch.float64);bad=torch.arange(1024)%73==0
    originals=[v.clone() for v in (x,xr,bad)]
    spec=SimpleNamespace(sup_out_union=SimpleNamespace(size=3));code=SimpleNamespace(n_cache=4,n_strict=2)
    storage=[torch.empty(128,2,3,2,dtype=torch.float64),torch.empty(128,2,2,dtype=torch.float64),
        torch.empty(128,4,2,dtype=torch.float64),torch.empty(128,2,2,dtype=torch.float64),torch.empty(128,dtype=torch.bool)]
    def raw(_s,_c,p,r,b,*ignored):
        storage[0].copy_(p[...,None].expand(-1,-1,-1,2));storage[1].copy_(r)
        storage[2].copy_(p[:,0,0,None,None].expand(-1,4,2));storage[3].copy_(r);storage[4].copy_(b)
        return tuple(storage)
    actual=collect(torch,raw,(spec,code,x,xr,bad,None,1e-6,None))
    expected=[x[...,None].expand(-1,-1,-1,2).contiguous(),xr,x[:,0,0,None,None].expand(-1,4,2).contiguous(),xr,bad]
    eq=lambda a,b:torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    assert all(eq(a,b) for a,b in zip(actual,expected)) and all(eq(a,b) for a,b in zip((x,xr,bad),originals))
    # A second whole call reuses every mock static. Earlier returned buffers
    # must survive, just as the real step1 outputs must survive step2 replay.
    x2=x+1;bad2=~bad
    second=collect(torch,raw,(spec,code,x2,xr,bad2,None,1e-6,None))
    expected2=[x2[...,None].expand(-1,-1,-1,2).contiguous(),xr,x2[:,0,0,None,None].expand(-1,4,2).contiguous(),xr,bad2]
    assert all(eq(a,b) for a,b in zip(second,expected2)) and all(eq(a,b) for a,b in zip(actual,expected))
    for v in storage:v.zero_()
    assert all(eq(a,b) for a,b in zip(actual,expected)), 'Returned full outputs must own storage'
    rejected=False
    try:collect(torch,raw,(spec,code,x[:1000],xr[:1000],bad[:1000],None,1e-6,None))
    except AssertionError:rejected=True
    assert rejected
    return dict(status='passed_CPU_assembly_only',five_owned_outputs_bytes_equal=True,static_alias_overwrite_survived=True,
        first_full_outputs_survive_second_full_call=True,second_full_outputs_bytes_equal=True,
        original_inputs_bytes_unchanged=True,nonfinite_and_bad_rows_preserved=True,non1024_rejected=True,
        actual_numeric_dispatch_tested=False,CUDA_tested=False)


if __name__=='__main__':
    import json
    print(json.dumps(check_local()))
