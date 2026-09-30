"""Isolated B2/n16 K20 strict SR reassociation; no reciprocal/NN changes.

The caller must admit the separately qualified reciprocal baseline. This module
owns each trial's actual interval-factor ledger and leaves point updates/J intact.
B1024 is rejected: the present deepcopy storage contract has not been qualified.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,inspect,json
import torch

SYMBOLIC_SHA='ea124b5ab72c80a25181cf4673cd835ee171e1b1266b9500ef19362469754972'
SPARSE_SHA='56e590796dbae42685b0331c7f41b53b62a3f73ed9be74e22738493820353ddf'
SCHEMA='actual-factor-owned-k20-v1'
K=20

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

@dataclass
class Ledger:
    factors: torch.Tensor
    length: int=0
    epoch: int=0
    last_rebuilt: int=0
    enabled: bool=True
    def reserve(self,needed,max_size):
        assert needed<=max_size
        if needed>len(self.factors):
            size=min(max_size,max(needed,2*len(self.factors)))
            new=self.factors.new_zeros((size,*self.factors.shape[1:]));new[:self.length].copy_(self.factors[:self.length]);self.factors=new

def ledger(sr):
    value=getattr(sr,'reassociation',None)
    assert isinstance(value,Ledger),'Checkpoint/SR lacks actual factors; cumulative Phi cannot reconstruct them'
    assert sr.phi_buf.shape[1:]==(2,16,16) and sr.max_size==1000,'Only B2/n16/SR1000 is admitted'
    assert value.length==sr.qlen and sr.jlen in [sr.qlen,sr.qlen-1]
    assert 0<=value.length<=len(value.factors)<=1000
    assert value.epoch>=0 and value.epoch%1000==0 and 0<=value.last_rebuilt<=value.length
    assert value.last_rebuilt==(value.length//K*K if value.enabled else 0)
    assert value.factors.shape[1:]==(2,16,16,2) and value.factors.dtype==torch.float64 and value.factors.device==sr.phi_buf.device
    return value

def attach_new(sr,enabled=True):
    assert sr.qlen==sr.jlen==0 and not hasattr(sr,'reassociation')
    assert sr.phi_buf.shape[1:]==(2,16,16) and sr.max_size==1000,'Only B2/n16/SR1000 is admitted'
    sr.reassociation=Ledger(sr.phi_buf.new_zeros((min(16,sr.max_size),2,16,16,2)),enabled=enabled)
    ledger(sr);return sr

def extra_fingerprint(sr):
    v=ledger(sr)
    return dict(sr_reassociation_schema=SCHEMA,sr_factor_ledger=tensor_sha(v.factors[:v.length]),
        sr_factor_length=v.length,sr_factor_capacity=len(v.factors),sr_factor_epoch=v.epoch,
        sr_reassociation_interval=K,sr_last_rebuilt=v.last_rebuilt,sr_reassociation_enabled=v.enabled)

def payload(sr):
    v=ledger(sr)
    return dict(extra_fingerprint(sr),sr_factors=v.factors[:v.length].detach().cpu().clone())

def restore_ledger(sr,saved):
    assert saved['sr_reassociation_schema']==SCHEMA and saved['sr_reassociation_interval']==K
    assert isinstance(getattr(sr,'reassociation',None),Ledger)
    f=saved['sr_factors'];length=saved['sr_factor_length'];capacity=saved['sr_factor_capacity']
    assert length==sr.qlen and 0<=length<=capacity<=sr.max_size
    assert f.shape==(length,2,16,16,2) and f.dtype==torch.float64 and bool(torch.isfinite(f).all())
    assert bool((f[...,0]<=f[...,1]).all()) and tensor_sha(f)==saved['sr_factor_ledger']
    assert isinstance(saved['sr_reassociation_enabled'],bool)
    # All data are checked before replacing the newly-created restore ledger.
    new=Ledger(f.new_zeros((capacity,2,16,16,2),device=sr.phi_buf.device),length,
        saved['sr_factor_epoch'],saved['sr_last_rebuilt'],saved['sr_reassociation_enabled'])
    new.factors[:length].copy_(f.to(sr.phi_buf.device));sr.reassociation=new;ledger(sr)
    assert extra_fingerprint(sr)=={k:saved[k] for k in extra_fingerprint(sr)}

def validate_saved_sr(saved):
    """No missing live history may silently become the factory's zeros."""
    q=saved['qlen'];assert type(q) is int and q==saved['jlen'] and 0<=q<=1000
    assert saved['sr_max_size']==1000 and q<=saved['sr_capacity']<=1000
    shapes={'scalars':(2,16),'phi_buf':(q,2,16,16),'j_buf':(q,2,16,2),
            'scalars_iv':(2,16,2),'phi_iv_buf':(q,2,16,16,2)}
    for name,shape in shapes.items():
        value=saved.get('sr_'+name)
        if value is None:
            assert q==0 and name in ['scalars_iv','phi_iv_buf'],('missing live SR field',name)
            continue
        assert isinstance(value,torch.Tensor) and value.shape==shape and value.dtype==torch.float64 and bool(torch.isfinite(value).all()),name
        if name in ['scalars_iv','phi_iv_buf','j_buf']:assert bool((value[...,0]<=value[...,1]).all()),name
    if q==0:assert bool((saved['sr_scalars']==1).all())
    cap=saved['cap'];assert cap.shape==(2,16,2) and cap.dtype==torch.float64 and bool(torch.isfinite(cap).all() and (cap[...,0]<=cap[...,1]).all())
    assert bool((cap[...,0]==-.1).all() and (cap[...,1]==.1).all())

def balanced_products(factors,sym):
    """Same doubling suffix topology as the qualified actual-factor probe."""
    work=factors[1:].clone();B,n=factors.shape[1:3];offset=1
    while offset<len(work):
        count=len(work)-offset
        left=work[offset:].clone().reshape(count*B,n,n,2).contiguous()
        prior=work[:-offset].clone().reshape(1,count*B,n,n,2).contiguous()
        if factors.is_cuda:
            assert sym.sr_kernels.available() and sym.sr_kernels.supported(left,prior)
            sym.sr_kernels.left_multiply_(left,prior)
        else:prior.copy_(sym._matmul_iv(left,prior))
        work[:-offset].copy_(prior.reshape(count,B,n,n,2));offset*=2
    assert bool(torch.isfinite(work).all() and (work[...,0]<=work[...,1]).all())
    return work

def sum_history(sr,sym):
    q=sr.qlen-1;history=sr.phi_iv_buf[1:q+1];columns=sr.j_buf[:q]
    if history.is_cuda:
        assert sym.sr_sum_kernels.available() and sym.sr_sum_kernels.supported(history,columns)
        return sym.sr_sum_kernels.sum_history(history,columns)
    terms=sym.iv.mul(history.movedim(0,1),columns.movedim(0,1).unsqueeze(2))
    return sym.iv.sum(sym.iv.sum(terms,dim=-1),dim=1)

class Installation:
    def __init__(self,c,b=None,enabled=True):
        assert isinstance(enabled,bool)
        self.c,self.b,self.enabled=c,b,enabled
        self.raw_prop=c.se.propagate;self.raw_factory=c.make_symbolic_remainder
        self.raw_fp=c.fingerprint;self.raw_save=c.save_state
        self.raw_restore=b.restore if b is not None else None
        self.sym=inspect.getmodule(self.raw_prop)
        assert sha(inspect.getfile(self.raw_prop))==SYMBOLIC_SHA and self.raw_prop.__name__=='propagate'
        assert sha(inspect.getfile(c.se))==SPARSE_SHA
        base=self.raw_factory.__globals__['SymbolicRemainder'];raw_reset=base.reset
        class OwnedSR(base):
            def reset(self):
                v=ledger(self);assert self.qlen==self.jlen==v.length==self.max_size,'Only original successful full reset is admitted'
                epoch=v.epoch+v.length
                raw_reset(self);v.length=0;v.epoch=epoch;v.last_rebuilt=0
                ledger(self)
        self.owned_class=OwnedSR
        c.make_symbolic_remainder=self.make;c.fingerprint=self.fingerprint;c.save_state=self.save
        c.se.propagate=self.propagate
        if b is not None:b.restore=self.restore
    def make(self,*args,**kwargs):
        sr=self.raw_factory(*args,**kwargs);sr.__class__=self.owned_class
        return attach_new(sr,self.enabled)
    def fingerprint(self,st,sr):return dict(self.raw_fp(st,sr),**extra_fingerprint(sr))
    def propagate(self,sr,phi_i,*,strict=False,phi_i_iv=None):
        assert strict is True and phi_i_iv is not None
        v=ledger(sr);assert sr.jlen==sr.qlen
        assert phi_i.shape==(2,16,16) and phi_i_iv.shape==(2,16,16,2)
        assert phi_i.dtype==phi_i_iv.dtype==torch.float64 and phi_i.device==phi_i_iv.device==sr.phi_buf.device
        assert bool(torch.isfinite(phi_i).all() and torch.isfinite(phi_i_iv).all() and (phi_i_iv[...,0]<=phi_i_iv[...,1]).all())
        index=v.length;v.reserve(index+1,sr.max_size);v.factors[index].copy_(phi_i_iv)
        result=self.raw_prop(sr,phi_i,strict=True,phi_i_iv=phi_i_iv)
        v.length=index+1;assert sr.qlen==v.length and sr.jlen==index
        if v.enabled and v.length%K==0:
            new=balanced_products(v.factors[:v.length],self.sym)
            sr.phi_iv_buf[1:sr.qlen].copy_(new)
            result=sum_history(sr,self.sym);v.last_rebuilt=v.length
        ledger(sr);return result
    def save(self,path,st,sr,eng,cap=None,step=None,phase='trial'):
        v=ledger(sr);assert step is not None and step==v.epoch+v.length
        self.raw_save(path,st,sr,eng,cap,step,phase)
        obj=torch.load(path,map_location='cpu',weights_only=True);obj.update(payload(sr));torch.save(obj,path)
    def restore(self,path,eng):
        """Metadata-aware frozen restore plus ledger BEFORE final full fingerprint."""
        c=self.c;saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['sr_reassociation_schema']==SCHEMA and saved['sr_reassociation_enabled']==self.enabled
        validate_saved_sr(saved)
        assert saved['n']==eng.tables.n and saved['k']==eng.tables.k and saved['h']==eng.step.delta
        pre=eng.support(saved['pre_ids']);tmv=eng.support(saved['tmv_ids'],True)
        assert torch.equal(saved['pre_exponents'],torch.from_numpy(eng.exponents(pre).copy()))
        spatial=torch.from_numpy(eng.exponents(tmv).copy())
        assert torch.equal(saved['tmv_exponents'],torch.cat((torch.zeros((len(spatial),1),dtype=torch.long),spatial),1))
        st=c.se.SparseState(**{name:saved[name].to(eng.device) for name in ['pre','pre_rem','tmv','tmv_rem','status']},pre_sup=pre,tmv_sup=tmv)
        sr=self.make(len(st.pre),saved['n'],saved['sr_max_size'],eng.device);sr.reserve(saved['sr_capacity'])
        for name in ['scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf']:
            value=saved.get('sr_'+name)
            if value is None:continue
            if name.startswith('phi') or name=='j_buf':
                target=getattr(sr,name)
                if target is None:
                    target=torch.empty((saved['sr_capacity'],*value.shape[1:]),dtype=value.dtype,device=eng.device);setattr(sr,name,target)
                target[:len(value)].copy_(value)
            else:setattr(sr,name,value.to(eng.device))
        sr.qlen,sr.jlen=saved['qlen'],saved['jlen']
        # The original restore fingerprints here; attach before that check.
        empty=sr.reassociation;empty.length=sr.qlen
        if len(empty.factors)<sr.qlen:empty.factors=empty.factors.new_zeros((saved['sr_factor_capacity'],2,16,16,2))
        restore_ledger(sr,saved)
        assert saved['step']==sr.reassociation.epoch+sr.reassociation.length
        side=json.loads(Path(path).with_suffix('.json').read_text())
        assert sha(path)==side['state_sha256'] and json.loads(json.dumps(self.fingerprint(st,sr)))==side['state_fingerprint']
        return st,sr,saved['cap'].to(eng.device)
    def uninstall(self):
        c=self.c;c.se.propagate=self.raw_prop;c.make_symbolic_remainder=self.raw_factory;c.fingerprint=self.raw_fp;c.save_state=self.raw_save
        if self.b is not None:self.b.restore=self.raw_restore

def install(c,b=None,*,enabled=True):return Installation(c,b,enabled)
