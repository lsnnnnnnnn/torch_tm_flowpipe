"""CPU-owned actual factors; only K20 SR reconstruction is lane-chunked.

An isolated storage candidate, not an installed full-driver adapter. Call
propagate with real pre-step int8 status, then finish_step after original J
append/status emission. Original point/J/scalars and full-batch sum stay intact.
"""
from pathlib import Path
import hashlib, inspect, json
import torch

SCHEMA='host-actual-factor-k20-v1'
SYMBOLIC_SHA='ea124b5ab72c80a25181cf4673cd835ee171e1b1266b9500ef19362469754972'
K=20
SR_FIELDS=('scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf')
BLOCK_BYTES=16*1024**2

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(x):
    if x.numel()==0:return hashlib.sha256(b'').hexdigest()
    owned=x.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(owned).cast('B')).hexdigest()
def signature(x):
    if isinstance(x,torch.Tensor):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=digest(x))
    if isinstance(x,dict):return {k:signature(v) for k,v in x.items()}
    return x

def cpu_copy(x):return x.detach().to(device='cpu',copy=True).contiguous()
def status_copy(x,batch):
    assert isinstance(x,torch.Tensor) and x.dtype==torch.int8 and x.shape==(batch,)
    y=cpu_copy(x);assert bool(((y>=0)&(y<=3)).all());return y

def interval_valid(x):
    return torch.isfinite(x).all(dim=(-1,-2,-3)) & (x[...,0]<=x[...,1]).all(dim=(-1,-2))

def balanced_chunk(factors,sym):
    """Same doubling suffix topology; no all-lane finite assertion."""
    work=factors[1:].clone();batch,n=factors.shape[1:3];offset=1
    while offset<len(work):
        count=len(work)-offset
        left=work[offset:].clone().reshape(count*batch,n,n,2).contiguous()
        prior=work[:-offset].clone().reshape(1,count*batch,n,n,2).contiguous()
        if work.is_cuda:
            assert sym.sr_kernels.available() and sym.sr_kernels.supported(left,prior)
            sym.sr_kernels.left_multiply_(left,prior)
        else:prior.copy_(sym._matmul_iv(left,prior))
        work[:-offset].copy_(prior.reshape(count,batch,n,n,2))
        del left,prior
        offset*=2
    return work

def full_history_sum(sr,sym):
    q=sr.qlen-1;history=sr.phi_iv_buf[1:q+1];columns=sr.j_buf[:q]
    if history.is_cuda:
        assert sym.sr_sum_kernels.available() and sym.sr_sum_kernels.supported(history,columns)
        return sym.sr_sum_kernels.sum_history(history,columns)
    terms=sym.iv.mul(history.movedim(0,1),columns.movedim(0,1).unsqueeze(2))
    return sym.iv.sum(sym.iv.sum(terms,dim=-1),dim=1)

class HostFactorLedger:
    def __init__(self,batch,n=16,max_size=1000,lane_chunk=16,status=None,enabled=True):
        assert type(batch) is int and 0<batch<=1024 and 1<=n<=32 and max_size==1000
        assert type(lane_chunk) is int and lane_chunk>0 and type(enabled) is bool
        self.batch,self.n,self.max_size,self.lane_chunk=batch,n,max_size,lane_chunk
        self.enabled=enabled;self.length=0;self.epoch=0;self.pending=None;self.poisoned=False
        self.current_status=status_copy(torch.zeros(batch,dtype=torch.int8,device='cpu') if status is None else status,batch)
        self.factors=torch.empty((max_size,batch,n,n,2),dtype=torch.float64,device='cpu')
        self.point_finite=torch.empty((max_size,batch),dtype=torch.bool,device='cpu')
        self.factor_valid=torch.empty_like(self.point_finite)
        self.status_before=torch.empty((max_size,batch),dtype=torch.int8,device='cpu')
        self.eligible=torch.zeros((max_size,batch),dtype=torch.bool,device='cpu')
        self.replaced=torch.zeros_like(self.eligible);self.fallback=torch.zeros_like(self.eligible)
        self.last_rebuilt_lane=torch.zeros(batch,dtype=torch.int64,device='cpu')
        self._raw=None
    def _ready(self,sr):
        assert not self.poisoned and self.pending is None,'Unfinished/poisoned SR cannot advance, save or reset'
        assert sr.max_size==self.max_size and sr.qlen==sr.jlen==self.length
        assert sr.phi_buf.shape[1:]==(self.batch,self.n,self.n)
    def propagate(self,sr,phi_i,phi_i_iv,*,status_before,raw_propagate,sym):
        self._ready(sr);assert self.length<self.max_size
        status=status_copy(status_before,self.batch)
        assert torch.equal(status,self.current_status),'Status must be the original current state; no revival'
        assert phi_i.shape==(self.batch,self.n,self.n) and phi_i_iv.shape==(*phi_i.shape,2)
        assert phi_i.dtype==phi_i_iv.dtype==torch.float64 and phi_i.device==phi_i_iv.device==sr.phi_buf.device
        if self._raw is None:
            assert inspect.getmodule(raw_propagate) is sym and sha(inspect.getfile(raw_propagate))==SYMBOLIC_SHA
            self._raw=raw_propagate
        assert raw_propagate is self._raw
        index=self.length;token=self.epoch+index+1
        # Synchronous independent snapshots precede any mutation by raw_prop.
        self.factors[index].copy_(phi_i_iv.detach().to('cpu'))
        self.point_finite[index].copy_(torch.isfinite(phi_i).all(dim=(-1,-2)).to('cpu'))
        self.factor_valid[index].copy_(interval_valid(self.factors[index])&self.point_finite[index])
        self.status_before[index].copy_(status)
        self.eligible[index].zero_();self.replaced[index].zero_();self.fallback[index].zero_()
        self.pending=token
        try:
            original=raw_propagate(sr,phi_i,strict=True,phi_i_iv=phi_i_iv)
            self.length=index+1;assert sr.qlen==self.length and sr.jlen==index
            if not self.enabled or self.length%K:return original
            eligible=(status==0)&self.factor_valid[:self.length].all(0)
            self.eligible[index].copy_(eligible)
            # Only scratch is chunked. Original Phi/J retain their full-B layout.
            for start in range(0,self.batch,self.lane_chunk):
                ids=torch.arange(start,min(start+self.lane_chunk,self.batch),device='cpu')[eligible[start:start+self.lane_chunk]]
                if not len(ids):continue
                stage=self.factors[:self.length,ids].contiguous().to(sr.phi_buf.device)
                work=balanced_chunk(stage,sym)
                good=interval_valid(work).all(0).to('cpu')
                accepted_ids=ids[good]
                if len(accepted_ids):
                    device_ids=accepted_ids.to(sr.phi_buf.device)
                    sr.phi_iv_buf[1:self.length,device_ids]=work[:,good.to(work.device)]
                    self.replaced[index,accepted_ids]=True
                    self.last_rebuilt_lane[accepted_ids]=self.length
                    del device_ids
                del accepted_ids,good,work,stage,ids
            self.fallback[index].copy_(eligible&~self.replaced[index])
            if not bool(self.replaced[index].any()):return original
            # Both matrices and all J are still in original full-B order. This
            # preserves the original two-stage n then Q reduction.
            rebuilt=full_history_sum(sr,sym)
            result=torch.where(self.replaced[index].to(original.device)[:,None,None],rebuilt,original)
            del rebuilt
            return result
        except BaseException:
            self.poisoned=True
            raise
    def finish_step(self,sr,status_after,*,token):
        assert not self.poisoned and self.pending==token==self.epoch+self.length
        assert sr.qlen==sr.jlen==self.length
        status=status_copy(status_after,self.batch)
        inactive=self.current_status!=0
        assert torch.equal(status[inactive],self.current_status[inactive]),'Original inactive lane must not revive/change status'
        self.current_status=status;self.pending=None
    def reset_if_full(self,sr):
        self._ready(sr)
        if self.length<self.max_size:
            assert sr.reset_if_full() is False;return False
        assert sr.reset_if_full() is True and sr.qlen==sr.jlen==0
        self.epoch+=self.length;self.length=0;self.last_rebuilt_lane.zero_()
        # Unused capacity bytes are not logical history; no per-control reset.
        self._ready(sr);return True
    def _snapshot_views(self,sr):
        self._ready(sr);q=self.length
        saved=dict(schema=SCHEMA,batch=self.batch,n=self.n,max_size=self.max_size,K=K,
            lane_chunk=self.lane_chunk,enabled=self.enabled,epoch=self.epoch,length=q,
            current_status=self.current_status,last_rebuilt_lane=self.last_rebuilt_lane,
            sr_capacity=len(sr.phi_buf),qlen=sr.qlen,jlen=sr.jlen)
        for name in ['factors','point_finite','factor_valid','status_before','eligible','replaced','fallback']:
            saved[name]=getattr(self,name)[:q]
        for name in SR_FIELDS:
            value=getattr(sr,name)
            saved['sr_'+name]=None if value is None else value[:q] if name in ['phi_buf','phi_iv_buf','j_buf'] else value
        return saved
    def payload(self,sr):
        assert self.batch<=32,'Whole-dict snapshots are small-fixture only; full batch requires save_components'
        return {k:cpu_copy(v) if isinstance(v,torch.Tensor) else v for k,v in self._snapshot_views(sr).items()}
    def save_checkpoint(self,path,sr):
        path=Path(path);assert not path.exists() and not path.with_suffix('.json').exists()
        saved=self.payload(sr);validate_payload(saved)
        torch.save(saved,path)
        path.with_suffix('.json').write_text(json.dumps(dict(schema=SCHEMA,state_sha256=sha(path),fingerprint=signature(saved)),indent=2)+'\n')
    def save_components(self,path,sr):
        """Bounded16MiB owned D2H blocks; manifest commits only after all files."""
        path=Path(path);assert not path.exists()
        views=self._snapshot_views(sr);validate_payload(views)
        path.mkdir(parents=True);meta={k:v for k,v in views.items() if not isinstance(v,torch.Tensor)}
        manifest=dict(schema=SCHEMA+'-components',adapter_sha256=sha(__file__),symbolic_sha256=SYMBOLIC_SHA,
            block_bytes=BLOCK_BYTES,metadata=meta,tensors={})
        # Process original SR first, then CPU ledger. No clone of a complete
        # SR or factor tensor is held at any point in this component path.
        keys=['sr_'+name for name in SR_FIELDS]+[k for k,v in views.items() if isinstance(v,torch.Tensor) and not k.startswith('sr_')]
        for key in keys:
            value=views[key]
            if value is None:continue
            assert value.is_contiguous();flat=value.detach().view(-1)
            entry=dict(shape=list(value.shape),dtype=str(value.dtype),blocks=[]);raw_hash=hashlib.sha256()
            count=max(1,BLOCK_BYTES//value.element_size())
            for start in range(0,flat.numel(),count):
                stop=min(start+count,flat.numel());owned=cpu_copy(flat[start:stop])
                filename=f'{key}_{start}.pt';target=path/filename;torch.save(owned,target)
                raw_hash.update(memoryview(owned.view(torch.uint8).numpy()).cast('B'))
                entry['blocks'].append(dict(file=filename,start=start,stop=stop,sha256=sha(target),size=target.stat().st_size))
                del owned,target
            entry['raw_sha256']=raw_hash.hexdigest();manifest['tensors'][key]=entry
            del flat,value
        (path/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')

def validate_payload(saved):
    required={'schema','batch','n','max_size','K','lane_chunk','enabled','epoch','length','current_status',
        'last_rebuilt_lane','sr_capacity','qlen','jlen','factors','point_finite','factor_valid','status_before',
        'eligible','replaced','fallback'}|{'sr_'+key for key in SR_FIELDS}
    assert set(saved)==required and saved['schema']==SCHEMA and saved['K']==K
    b,n,q=saved['batch'],saved['n'],saved['length']
    assert type(b) is int and b>0 and 1<=n<=32 and saved['max_size']==1000
    assert type(q) is int and 0<=q<=1000 and saved['qlen']==saved['jlen']==q
    assert q<=saved['sr_capacity']<=1000 and type(saved['enabled']) is bool
    assert type(saved['epoch']) is int and saved['epoch']>=0 and saved['epoch']%1000==0
    assert type(saved['lane_chunk']) is int and saved['lane_chunk']>0
    status=status_copy(saved['current_status'],b)
    shapes={'factors':((q,b,n,n,2),torch.float64),'point_finite':((q,b),torch.bool),
        'factor_valid':((q,b),torch.bool),'status_before':((q,b),torch.int8),
        'eligible':((q,b),torch.bool),'replaced':((q,b),torch.bool),'fallback':((q,b),torch.bool),
        'last_rebuilt_lane':((b,),torch.int64),
        'sr_scalars':((b,n),torch.float64),'sr_scalars_iv':((b,n,2),torch.float64),
        'sr_phi_buf':((q,b,n,n),torch.float64),'sr_phi_iv_buf':((q,b,n,n,2),torch.float64),'sr_j_buf':((q,b,n,2),torch.float64)}
    for name,(shape,dtype) in shapes.items():
        value=saved[name]
        if value is None:
            assert q==0 and name in ['sr_scalars_iv','sr_phi_iv_buf'];continue
        assert isinstance(value,torch.Tensor) and tuple(value.shape)==shape and value.dtype==dtype,name
        if not name.startswith('sr_'):assert value.device.type=='cpu',name
    # Inactive lanes may legitimately retain the original nonfinite failure
    # outputs. Active strict interval histories must still be valid. Point Phi
    # is diagnostic in this strict path and is not promoted into a certificate.
    active=status==0
    assert bool(torch.isfinite(saved['sr_scalars'][active.to(saved['sr_scalars'].device)]).all())
    for name in ['sr_scalars_iv','sr_phi_iv_buf','sr_j_buf']:
        value=saved[name]
        if value is None:continue
        device_active=active.to(value.device)
        if name=='sr_scalars_iv':
            selected=value[device_active]
            assert bool(torch.isfinite(selected).all() and (selected[...,0]<=selected[...,1]).all()),name
            del selected
        else:
            rows=max(1,BLOCK_BYTES//max(1,value[0:1].numel()*value.element_size()))
            for first in range(0,q,rows):
                selected=value[first:first+rows,device_active]
                assert bool(torch.isfinite(selected).all() and (selected[...,0]<=selected[...,1]).all()),name
                del selected
    if q==0:
        assert bool((saved['sr_scalars']==1).all())
        if saved['epoch']>0:assert saved['sr_scalars_iv'] is not None and saved['sr_phi_iv_buf'] is not None
        if saved['sr_scalars_iv'] is not None:assert bool((saved['sr_scalars_iv']==1).all())
    actual=torch.empty((q,b),dtype=torch.bool,device='cpu')
    rows=max(1,BLOCK_BYTES//(b*n*n*2*8))
    for first in range(0,q,rows):actual[first:first+rows]=interval_valid(saved['factors'][first:first+rows])&saved['point_finite'][first:first+rows]
    assert torch.equal(actual,saved['factor_valid'])
    last=torch.zeros(b,dtype=torch.int64,device='cpu');history=torch.ones(b,dtype=torch.bool,device='cpu')
    for index in range(q):
        current=status_copy(saved['status_before'][index],b)
        future=saved['status_before'][index+1] if index+1<q else status
        assert torch.equal(future[current!=0],current[current!=0])
        history&=actual[index]
        expected=(current==0)&history if saved['enabled'] and (index+1)%K==0 else torch.zeros(b,dtype=torch.bool,device='cpu')
        assert torch.equal(saved['eligible'][index],expected)
        replaced=saved['replaced'][index]
        assert bool((~replaced|expected).all()) and torch.equal(saved['fallback'][index],expected&~replaced)
        last[replaced]=index+1
    assert torch.equal(last,saved['last_rebuilt_lane'])

def restore_checkpoint(path,raw_factory,device):
    path=Path(path);side=json.loads(path.with_suffix('.json').read_text())
    assert side['schema']==SCHEMA and sha(path)==side['state_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['batch']<=32,'Whole-dict restore is small-fixture only; full batch requires restore_components'
    assert signature(saved)==side['fingerprint'];validate_payload(saved)
    ledger=HostFactorLedger(saved['batch'],saved['n'],saved['max_size'],saved['lane_chunk'],saved['current_status'],saved['enabled'])
    ledger.length=saved['length'];ledger.epoch=saved['epoch'];q=ledger.length
    ledger.last_rebuilt_lane.copy_(saved['last_rebuilt_lane'])
    for name in ['factors','point_finite','factor_valid','status_before','eligible','replaced','fallback']:
        getattr(ledger,name)[:q].copy_(saved[name])
    sr=raw_factory(ledger.batch,ledger.n,ledger.max_size,device);sr.reserve(saved['sr_capacity'])
    for name in SR_FIELDS:
        value=saved['sr_'+name]
        if value is None:continue
        if name in ['phi_buf','phi_iv_buf','j_buf']:
            target=getattr(sr,name)
            if target is None:
                target=torch.empty((saved['sr_capacity'],*value.shape[1:]),dtype=value.dtype,device=device);setattr(sr,name,target)
            target[:q].copy_(value.to(device))
        else:setattr(sr,name,value.to(device))
    sr.qlen,sr.jlen=q,q;ledger._ready(sr)
    assert signature(ledger.payload(sr))==side['fingerprint']
    return sr,ledger

def restore_components(path,raw_factory,device):
    """Restore block-by-block into owned destinations; no full CPU snapshot."""
    path=Path(path);manifest=json.loads((path/'MANIFEST.json').read_text())
    assert manifest['schema']==SCHEMA+'-components' and manifest['adapter_sha256']==sha(__file__)
    assert manifest['symbolic_sha256']==sha(inspect.getfile(raw_factory))==SYMBOLIC_SHA
    assert manifest['block_bytes']==BLOCK_BYTES
    meta=manifest['metadata'];b,n,q=meta['batch'],meta['n'],meta['length']
    assert type(b) is int and 0<b<=1024 and 1<=n<=32 and meta['max_size']==1000
    assert type(q) is int and 0<=q<=meta['sr_capacity']<=1000 and meta['qlen']==meta['jlen']==q
    ledger=HostFactorLedger(b,n,1000,meta['lane_chunk'],enabled=meta['enabled'])
    ledger.length=q;ledger.epoch=meta['epoch']
    sr=raw_factory(b,n,1000,device);sr.reserve(meta['sr_capacity']);sr.qlen=sr.jlen=q
    for name in ['scalars_iv','phi_iv_buf']:
        key='sr_'+name
        if key in manifest['tensors']:
            shape=(b,n,2) if name=='scalars_iv' else (meta['sr_capacity'],b,n,n,2)
            setattr(sr,name,torch.empty(shape,dtype=torch.float64,device=device))
        else:assert key in meta and meta[key] is None
    views=ledger._snapshot_views(sr)
    assert {k:v for k,v in views.items() if not isinstance(v,torch.Tensor)}==meta
    assert {k for k,v in views.items() if isinstance(v,torch.Tensor)}==set(manifest['tensors'])
    for key,entry in manifest['tensors'].items():
        target=views[key];assert list(target.shape)==entry['shape'] and str(target.dtype)==entry['dtype']
        flat=target.view(-1);end=0;raw_hash=hashlib.sha256()
        for block in entry['blocks']:
            assert Path(block['file']).name==block['file'] and block['file']==f'{key}_{end}.pt'
            assert block['start']==end and end<block['stop']<=flat.numel()
            file=path/block['file'];assert file.stat().st_size==block['size'] and sha(file)==block['sha256']
            owned=torch.load(file,map_location='cpu',weights_only=True)
            assert owned.shape==(block['stop']-end,) and owned.dtype==target.dtype
            assert owned.numel()*owned.element_size()<=BLOCK_BYTES
            raw_hash.update(memoryview(owned.view(torch.uint8).numpy()).cast('B'))
            flat[end:block['stop']].copy_(owned);end=block['stop'];del owned
        assert end==flat.numel() and raw_hash.hexdigest()==entry['raw_sha256']
        del flat,target
    validate_payload(views);ledger._ready(sr)
    return sr,ledger
