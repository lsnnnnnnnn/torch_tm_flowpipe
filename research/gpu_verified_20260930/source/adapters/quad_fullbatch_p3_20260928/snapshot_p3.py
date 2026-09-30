"""P3 plant plus unchanged streaming SR/host-ledger checkpoint for a finished full-B step.

The caller owns one state/SR. No parent/trial copies or per-step snapshots are
introduced. MANIFEST.json is written last and is required for restoration.
"""
from pathlib import Path
import hashlib,json,math

SCHEMA='quad-fullbatch-P3-plant-hostsr-v1'
PLANT=('pre','pre_rem','tmv','tmv_rem','status')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def exponents(eng,sup):
    ids=list(sup.ids)
    return eng.tables.exponents[eng.tables.spatial_index[ids] if sup.spatial else ids].detach().cpu()

def validate(v,eng,torch):
    assert v['n']==eng.tables.n==16 and v['k']==eng.tables.k==3 and v['h']==eng.step.delta==.005
    assert v['pre_ids'] and v['tmv_ids'] and tuple(sorted(set(v['pre_ids'])))==tuple(v['pre_ids'])
    assert tuple(sorted(set(v['tmv_ids'])))==tuple(v['tmv_ids'])
    for prefix,variables in [('pre',17),('tmv',16)]:
        ids=v[prefix+'_ids'];exp=v[prefix+'_exponents']
        assert ids[0]==0 and all(type(i) is int and 0<=i<math.comb(variables+3,3) for i in ids)
        assert exp.shape==(len(ids),17) and exp.dtype==torch.int64 and exp.device.type=='cpu'
        assert bool((exp>=0).all() and (exp.sum(-1)<=3).all())
        if prefix=='tmv':assert bool((exp[:,0]==0).all())
    B=v['pre'].shape[0];assert B==1024
    for name,shape,dtype in [('pre',(B,16,len(v['pre_ids'])),torch.float64),('tmv',(B,16,len(v['tmv_ids'])),torch.float64),
                            ('pre_rem',(B,16,2),torch.float64),('tmv_rem',(B,16,2),torch.float64),
                            ('cap',(B,16,2),torch.float64),('status',(B,),torch.int8)]:
        x=v[name];assert x.shape==shape and x.dtype==dtype and x.device.type=='cpu',name
        if name=='status':assert bool(((x>=0)&(x<=3)).all())
        else:
            assert bool(torch.isfinite(x).all()),'Finite-state full-B boundary contract: '+name
            if name.endswith('_rem') or name=='cap':assert bool((x[...,0]<=x[...,1]).all())
    assert bool((v['cap'][...,0]==-.1).all() and (v['cap'][...,1]==.1).all())

def validate_progress(progress,v=None):
    step=progress['completed_step'];phase=progress['phase']
    assert type(step) is int and 0<=step<=1000
    assert phase in ['committed_before_endpoint_handoff','after_strict_endpoint','after_controller']
    if phase!='committed_before_endpoint_handoff':
        assert step%20==0
        if v is not None:assert bool((v['pre_exponents'][:,0]==0).all())

def save(path,st,sr,ledger,eng,cap,*,metadata,progress,host):
    import torch
    path=Path(path);assert not path.exists()
    ledger._ready(sr)
    assert ledger.batch==1024 and ledger.n==16 and ledger.max_size==1000
    assert torch.equal(st.status.detach().cpu(),ledger.current_status)
    assert st.pre_sup.n==st.tmv_sup.n==16 and st.pre_sup.k==st.tmv_sup.k==3
    assert st.pre_sup.spatial is False and st.tmv_sup.spatial is True
    assert progress['completed_step']==ledger.epoch+ledger.length
    validate_progress(progress)
    assert metadata['host_adapter_sha256']==sha(host.__file__)
    v={k:getattr(st,k).detach().to('cpu',copy=True) for k in PLANT}
    v.update(pre_ids=tuple(st.pre_sup.ids),tmv_ids=tuple(st.tmv_sup.ids),pre_exponents=exponents(eng,st.pre_sup),
        tmv_exponents=exponents(eng,st.tmv_sup),n=16,k=3,h=.005,cap=cap.detach().to('cpu',copy=True))
    validate(v,eng,torch)
    validate_progress(progress,v)
    path.mkdir(parents=True)
    torch.save(v,path/'plant.pt');del v
    # Host module saves original SR then actual factors in <=16MiB chunks.
    ledger.save_components(path/'sr',sr)
    manifest=dict(schema=SCHEMA,serializer_sha256=sha(__file__),metadata=metadata,progress=progress,
        files={'plant.pt':sha(path/'plant.pt'),'sr/MANIFEST.json':sha(path/'sr/MANIFEST.json')})
    write(path/'MANIFEST.json',manifest)
    return manifest

def restore(path,eng,*,metadata,host,raw_factory):
    import torch
    from flowstar_gpu import sparse_exec as se,support as sp
    path=Path(path);m=json.loads((path/'MANIFEST.json').read_text())
    assert m['schema']==SCHEMA and m['serializer_sha256']==sha(__file__)
    assert m['metadata']==metadata and metadata['host_adapter_sha256']==sha(host.__file__)
    validate_progress(m['progress'])
    assert set(m['files'])=={'plant.pt','sr/MANIFEST.json'}
    for name,digest in m['files'].items():assert sha(path/name)==digest,name
    v=torch.load(path/'plant.pt',map_location='cpu',weights_only=True);validate(v,eng,torch)
    validate_progress(m['progress'],v)
    pre=sp.make_support(16,3,False,tuple(v['pre_ids']));tmv=sp.make_support(16,3,True,tuple(v['tmv_ids']))
    assert torch.equal(exponents(eng,pre),v['pre_exponents']) and torch.equal(exponents(eng,tmv),v['tmv_exponents'])
    st=se.SparseState(**{k:v[k].to(eng.device) for k in PLANT},pre_sup=pre,tmv_sup=tmv)
    cap=v['cap'].to(eng.device);del v
    sr,ledger=host.restore_components(path/'sr',raw_factory,eng.device)
    assert ledger.batch==1024 and ledger.n==16 and ledger.max_size==1000
    assert m['progress']['completed_step']==ledger.epoch+ledger.length
    assert torch.equal(st.status.detach().cpu(),ledger.current_status)
    return st,sr,ledger,cap,m['progress']
