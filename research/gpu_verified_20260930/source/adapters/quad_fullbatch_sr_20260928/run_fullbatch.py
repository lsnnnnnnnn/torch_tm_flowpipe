"""Original full1024 QUAD/P2 driver, bounded40 and its own fresh cold20->21.

Only independent boundary/reciprocal/SR adapters are installed. The original
main retains NN, pruning, safety and reset control flow. No whole-SR deepcopy.
CROWN remains conditional; this runner does not assert a strict NNCS theorem.
"""
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import argparse, hashlib, importlib.util, json, sys, time, traceback

HERE=Path(__file__).parent
PINS={
 'bootstrap.py':'1a49b0384371cdb4101aa32fec64f846aa724336725a02b35447e7968b558435',
 'snapshot.py':'6adc9a430c236e37749a8b175e19925c5c68dedef7c01b3cb47b168f11010da2',
 'strict_endpoint_fullbatch.py':'864e949d425803f1f17b36d453970bbb32f6f47c2c547b382ee36cf591e7a33e',
 'strict_injection_fullbatch.py':'652a678e667553fede0fbead4ec435028ec830c461620d486617c250d0ad9276',
 'check_boundaries.py':'10b750e009daf17a183a957d6fc8ee06c535f8d00bab248c71b9bd9b13e87ddb',
 'check_snapshot_cpu.py':'4280022f30e0f0fd089702a327bcffcb2e043eee99ea0db643a80840214cf382',
 '../quad_sr_reassociate_20260928/host_ledger.py':'1a7d9bbf0f16a4f3726b75a27ff938ff87b918141a5f0fe1504f26198dc4035b',
 '../quad_sr_reassociate_20260928/check_host_ledger_cuda.py':'5b58ee21c98e2c945743dfdc5664baac4f37fcfa9eca246b1ee908152f8e2070',
}
HOST_SHA=PINS['../quad_sr_reassociate_20260928/host_ledger.py']
HOST_CHECK_SHA=PINS['../quad_sr_reassociate_20260928/check_host_ledger_cuda.py']
HOST_CPU_SHA='c704ae33b03f8216c4e7df98d3abea3c166ebdfd1e57a1193374cd7259e0abc9'
HOST_IDENTITY_SHA='94906e51208aa6d04cc605fdef0eb1e84eb698fa0734f26e443447ceab49a668'
SNAPSHOT_CPU_SHA='c5ea93df1d17ecfa7cfa8c859cedbbf3f22bbd3889aec33ebf97e17981e859be'
GIB=1024**3
POLICY='original_full1024_P2_directNN_reciprocal5_host_actual_factors_K20_chunk16'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def gate_receipts(boundary,small,capacity,snapshot,original,small_sha):
    """Pure protocol gate, shared with the CPU negative checks."""
    assert boundary['status']=='passed' and boundary['device']=='cuda'
    assert boundary['script_sha256']==PINS['check_boundaries.py']
    assert boundary['adapter_sha256']=={'endpoint':PINS['strict_endpoint_fullbatch.py'],'injection':PINS['strict_injection_fullbatch.py']}
    assert boundary['source_diff_only_status_guard_and_docs'] is True
    assert boundary['fraction_endpoint_components']==64 and boundary['fraction_coefficient_components']==36
    assert len(boundary['negative_cases_rejected'])==8
    assert [r['batch'] for r in boundary['cases']]==[4,1024]
    for row in boundary['cases']:
        assert row['status_codes']==[0,1,2,3]
        for key in ['endpoint_and_injection_all_rows_bytes_equal_to_frozen_numeric_body','status_unchanged','all_nn_rows_retained']:assert row[key] is True
    for gate,mode in [(small,'small'),(capacity,'capacity')]:
        assert gate['status']=='passed' and gate['device']=='cuda' and gate['mode']==mode
        assert gate['script_sha256']==HOST_CHECK_SHA and gate['host_sha256']==HOST_SHA
        assert gate['cpu_gate_sha256']==HOST_CPU_SHA and gate['identity_sha256']==HOST_IDENTITY_SHA
        assert gate['common_sha256']==original['common_script_sha256']
        assert gate['torch_allocation_cap_bytes']==11*GIB and gate['external_watch_limit_bytes']==int(11.5*GIB)
        assert gate['dispatch']['matrix_calls']>0 and gate['dispatch']['history_sum_calls']>0
        assert gate['dispatch']['hooks_restored'] is True and gate['no_original_extension_rebuild'] is True
    assert capacity['small_gate_sha256']==small_sha
    s=small['small'];assert s['step_comparisons']==160
    assert s['rows']==[dict(batch=2,chunk=k,enabled=e,steps=40) for e,k in [(False,16),(True,1),(True,16)]]
    for key in ['all_original_SR_and_image_bytes_equal','mixed_bad_lanes_preserve_raw','stream_cold_and_reset_passed','inactive_revival_rejected']:assert s[key] is True
    cap=capacity['capacity']
    for key,value in [('batch',1024),('n',16),('capacity',1000),('chunk',16),('initial_q',999),('full_batch_propagate_calls',1),('real_template_batch',2),('real_template_steps',1000)]:assert cap[key]==value
    assert cap['stream_restore_all_fields_equal'] is True and cap['original_reset_and_epoch1000'] is True
    assert cap['full999_field_hashes'] and cap['full1000_field_hashes']
    for gate in [boundary,small,capacity]:
        assert gate['engine_python_sha256']==original['engine']['python_sha256']
        assert gate['extensions']==original['extensions'] and len(gate['extensions'])==8
        assert gate['torch_version']==small['torch_version']==original['torch_version']
    assert snapshot['status']=='passed' and snapshot['device']=='cpu'
    assert snapshot['serializer_sha256']==PINS['snapshot.py'] and snapshot['script_sha256']==PINS['check_snapshot_cpu.py']
    assert snapshot['batch']==1024 and snapshot['n']==16 and snapshot['working_order']==2 and snapshot['negative_count']==10
    assert snapshot['full_plant_support_cap_status_and_SR_ledger_bytes_equal'] is True
    assert snapshot['next_rawprop_image_all_bytes_equal'] is True


def initialize(args):
    for rel,digest in PINS.items():assert sha(HERE/rel)==digest,rel
    assert sha(args.snapshot_check/'RESULT.json')==SNAPSHOT_CPU_SHA
    boot=load('fullbatch_bootstrap',HERE/'bootstrap.py')
    # Admission reads files only and precedes the original library bootstrap.
    original=read(boot.N/'runs/quad_normalization_center_20260924/combined/result.json')
    original=dict(original,reference_sha256=boot.REFERENCE_SHA,common_script_sha256=boot.COMMON_SHA)
    paths={k:getattr(args,k)/'RESULT.json' for k in ['boundary_check','host_small_check','host_capacity_check','snapshot_check']}
    receipts={k:read(p) for k,p in paths.items()}
    gate_receipts(receipts['boundary_check'],receipts['host_small_check'],receipts['host_capacity_check'],receipts['snapshot_check'],original,sha(paths['host_small_check']))
    ctx=boot.initialize(args)
    ctx.boot=boot;ctx.host=load('fullbatch_host',HERE/'../quad_sr_reassociate_20260928/host_ledger.py')
    ctx.snapshot=load('fullbatch_snapshot',HERE/'snapshot.py')
    ctx.endpoint=load('fullbatch_endpoint',HERE/'strict_endpoint_fullbatch.py')
    ctx.injection=load('fullbatch_injection',HERE/'strict_injection_fullbatch.py')
    total=ctx.torch.cuda.get_device_properties(0).total_memory
    assert total>=11*GIB;ctx.torch.cuda.set_per_process_memory_fraction(11*GIB/total,0)
    # Original config has no intermediate safe/unsafe abort; target is at1000.
    assert not ctx.cfg.get('constraints_safe') and not ctx.cfg.get('constraints_unsafe')
    ctx.identity=dict(ctx.identity,bootstrap_sha256=PINS['bootstrap.py'],script_sha256=sha(__file__),policy=POLICY,
        host_adapter_sha256=HOST_SHA,host_cuda_checker_sha256=HOST_CHECK_SHA,host_cuda_fixture_identity_sha256=HOST_IDENTITY_SHA,snapshot_sha256=PINS['snapshot.py'],
        mixed_endpoint_sha256=PINS['strict_endpoint_fullbatch.py'],mixed_injection_sha256=PINS['strict_injection_fullbatch.py'],
        qualification_receipts={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        original_target_steps=1000,execution_budget_steps=40,sr_allocation='preallocated_full_capacity',sr_capacity=1000,
        sr_factor_storage='CPU_owned_actual_factors',sr_lane_chunk=16,sr_reassociation_period=20,
        release_unused_cache=True,release_cache_threshold_bytes=2*GIB,torch_allocation_cap_bytes=11*GIB,external_watch_limit_bytes=int(11.5*GIB),
        limitation='Conditional CROWN/same-slope certificates. Fullbatch40 plus owncold is a bounded execution qualification, not the1000-step target or an end-to-end strict NNCS certificate.')
    return ctx


def own40_gate(source,identity):
    source=Path(source);inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==identity
    assert r['input_sha256']==sha(source/'INPUT.json')
    assert inp['mode']=='bounded40' and inp['original_target_steps']==1000 and inp['budget_steps']==40
    assert identity['script_sha256']==sha(__file__) and identity['policy']==POLICY
    assert r['mode']=='bounded40' and r['status']=='bounded_prefix_completed'
    assert r['original_target_steps']==1000 and r['budget_steps']==40
    assert r['completed_step']==40 and r['accepted_lane_steps']==40960 and r['all_steps_all_lanes_accepted'] is True
    assert r['controller_calls']==2 and r['controller_refresh_steps']==[0,20]
    assert set(r['snapshots'])=={'committed_20','after_endpoint_20','after_controller_20','committed_21','committed_40'}
    for name,digest in r['snapshots'].items():assert sha(source/name/'MANIFEST.json')==digest
    return dict(source_input_sha256=sha(source/'INPUT.json'),source_result_sha256=sha(source/'RESULT.json'))


class BudgetStop(Exception):pass

class Runtime:
    def __init__(self,ctx,args):
        self.ctx,self.args,self.d,self.t,self.h=ctx,args,ctx.d,ctx.torch,ctx.host
        self.completed=0;self.held=None;self.held_signature=None;self.phase='committed_before_endpoint_handoff'
        self.st=self.sr=self.ledger=self.eng=self.cap=None;self.pending_status=None;self.propagate_count=0
        self.broken=self.t.zeros(1024,dtype=self.t.bool,device='cpu');self.last_accepted=self.t.zeros(1024,dtype=self.t.int64,device='cpu')
        self.rows=[];self.snapshots={};self.controller_steps=[];self.controller_calls=0;self.cache_releases=[]
        self.times={k:0. for k in ['advance_s','observer_s','controller_s','controller_setup_s','boundary_s','checkpoint_s','reserve_s','cache_release_s']}
        names=['make_cells','SparseEngine','initial_sparse_state','make_symbolic_remainder','build_rem_est','build_crown','crown_bounds','inject_controls_s','end_of_time_s','advance_sparse','prune_state']
        self.raw={k:getattr(self.d,k) for k in names}
        from flowstar_gpu import symbolic_remainder as sym
        self.sym=sym;self.rawprop=sym.propagate
        assert ctx.c.se.propagate is self.rawprop
        for name in names:setattr(self.d,name,getattr(self,name))
        ctx.c.se.propagate=self.propagate
    def sync(self):self.t.cuda.synchronize()
    def owned(self,x):return x.detach().to('cpu',copy=True).contiguous()
    def signature(self,value):
        if isinstance(value,self.t.Tensor):
            flat=value.detach().contiguous().view(-1);h=hashlib.sha256();count=max(1,self.h.BLOCK_BYTES//value.element_size())
            for start in range(0,flat.numel(),count):
                block=self.owned(flat[start:start+count]);h.update(memoryview(block.view(self.t.uint8).numpy()).cast('B'));del block
            return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=h.hexdigest())
        if isinstance(value,dict):return {k:self.signature(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [self.signature(v) for v in value]
        return value
    def sr_signature(self):return self.signature(self.ledger._snapshot_views(self.sr))
    def protected(self,physical=False):
        v={k:getattr(self.st,k) for k in ['tmv','tmv_rem','status']}
        v.update(tmv_ids=list(self.st.tmv_sup.ids),sr=self.sr_signature())
        if physical:v.update(pre_physical=self.st.pre[:,:13],pre_rem_physical=self.st.pre_rem[:,:13],pre_ids=list(self.st.pre_sup.ids))
        return self.signature(v)
    def artifact(self,name,value):
        def cpu(v):
            if isinstance(v,self.t.Tensor):return self.owned(v)
            if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
            return v
        path=self.args.output/(name+'.pt');value=cpu(value);self.t.save(value,path)
        write(path.with_suffix('.json'),dict(source_identity=self.ctx.identity,step=self.completed,signature=self.signature(value),pt_sha256=sha(path)))
    def progress(self):
        return dict(completed_step=self.completed,attempted_step=self.completed,next_step=self.completed+1,phase=self.phase,
            next_control_refresh_step=self.completed if self.phase!='after_controller' and self.completed%20==0 else ((self.completed//20)+1)*20,
            sr_history_origin_step=self.ledger.epoch,held_controller_step=self.held,held_controller_signature=self.held_signature,
            broken=self.broken.tolist(),last_accepted_step=self.last_accepted.tolist())
    def save(self,name):
        t=time.perf_counter();m=self.ctx.snapshot.save(self.args.output/name,self.st,self.sr,self.ledger,self.eng,self.cap,
            metadata=self.ctx.identity,progress=self.progress(),host=self.h)
        self.snapshots[name]=sha(self.args.output/name/'MANIFEST.json');self.times['checkpoint_s']+=time.perf_counter()-t
        return m
    def make_cells(self,config):
        assert config['steps']==50 and config['ode_order']==2
        return self.ctx.boxes.clone()
    def SparseEngine(self,*a,**kw):
        assert self.eng is None;self.eng=self.raw['SparseEngine'](*a,**kw)
        self.eng.horner_edge_kernel=self.ctx.edge.horner_edge
        return self.eng
    def initial_sparse_state(self,cells,eng,sched):
        assert eng is self.eng and self.t.equal(cells.cpu(),self.ctx.boxes)
        self.st=self.raw['initial_sparse_state'](cells,eng,sched);self.sched=sched
        s=self.st;p=self.owned(s.pre);tmv=self.owned(s.tmv)
        exps=self.ctx.snapshot.exponents(eng,s.pre_sup).tolist();texps=self.ctx.snapshot.exponents(eng,s.tmv_sup).tolist()
        assert exps[0]==[0]*17 and not s.pre_sup.spatial and s.tmv_sup.spatial
        expected=self.t.zeros_like(p);expected[:,:,0]=p[:,:,0]
        target=self.t.zeros_like(tmv);checks=0
        for dim in range(16):
            e=[0]*17;e[dim+1]=1;slot=exps.index(e);tslot=texps.index(e)
            expected[:,dim,slot]=p[:,dim,slot];target[:,dim,tslot]=1
            for lane in range(1024):
                center=Fraction(float(p[lane,dim,0]));radius=Fraction(float(p[lane,dim,slot]));lo,hi=map(lambda z:Fraction(float(z)),self.ctx.boxes[lane,dim])
                assert radius>=0 and center-radius<=lo<=hi<=center+radius;checks+=1
        assert self.t.equal(p,expected) and self.t.equal(tmv,target)
        assert not bool(s.pre_rem.any() or s.tmv_rem.any() or s.status.any())
        write(self.args.output/'INITIAL_COVERAGE.json',dict(status='passed',source_identity=self.ctx.identity,fraction_components=checks,
            pre=self.signature(p),tmv=self.signature(tmv),zero_remainders_and_active=True,archived_boxes_order_unchanged=True))
        return s
    def attach_reset(self):
        rawreset=self.sr.reset
        def reset():
            ledger=self.ledger;ledger._ready(self.sr)
            assert ledger.length==ledger.max_size==1000,'No early/control reset'
            rawreset();assert self.sr.qlen==self.sr.jlen==0
            ledger.epoch+=ledger.length;ledger.length=0;ledger.last_rebuilt_lane.zero_();ledger._ready(self.sr)
        self.sr.reset=reset
    def make_symbolic_remainder(self,B,n,max_size,device):
        assert (B,n,max_size)==(1024,16,1000) and self.sr is None
        t=time.perf_counter();self.sr=self.raw['make_symbolic_remainder'](B,n,max_size,device)
        assert self.sr.qlen==self.sr.jlen==0
        self.t.cuda.empty_cache();self.sr.reserve(1000)
        self.ledger=self.h.HostFactorLedger(B,n,max_size,lane_chunk=16,status=self.st.status)
        self.attach_reset();self.sync();self.times['reserve_s']+=time.perf_counter()-t
        return self.sr
    def build_rem_est(self,*a,**kw):
        self.cap=self.raw['build_rem_est'](*a,**kw);assert self.cap.shape==(1024,16,2)
        assert bool((self.cap[...,0]==-.1).all() and (self.cap[...,1]==.1).all());return self.cap
    def build_crown(self,*a,**kw):
        assert kw==dict(relax='same-slope',input_layout='native')
        t=time.perf_counter();model=self.raw['build_crown'](*a,**kw);self.sync();self.times['controller_setup_s']+=time.perf_counter()-t;return model
    def crown_bounds(self,model,config,lb,ub,**kw):
        assert self.completed in [0,20] and self.completed not in self.controller_steps
        assert kw=={'input_layout':'native'} and lb.shape==ub.shape==(1024,12)
        assert self.phase==('committed_before_endpoint_handoff' if self.completed==0 else 'after_strict_endpoint')
        self.sync();t=time.perf_counter();T,L,U=self.raw['crown_bounds'](model,config,lb,ub,**kw);self.sync();self.times['controller_s']+=time.perf_counter()-t
        assert T.shape==(1024,3,12) and L.shape==U.shape==(1024,3)
        self.controller_calls+=1;self.controller_steps.append(self.completed)
        value=dict(input_lb=lb,input_ub=ub,T=T,L=L,U=U)
        self.nn_signature=self.signature(value);self.artifact('controller_'+str(self.completed),value)
        return T,L,U
    def inject_controls_s(self,st,T,L,U,u_ids,nn_in):
        assert st is self.st and self.controller_steps[-1]==self.completed and nn_in==12 and list(u_ids)==[13,14,15]
        assert self.signature(dict(T=T,L=L,U=U))=={k:self.nn_signature[k] for k in ['T','L','U']},'native-f64 transport changed NN certificate'
        t=time.perf_counter();before=self.protected(physical=True)
        receipt=self.ctx.injection.inject_controls_s(st,T,L,U,u_ids,nn_in,eng=self.eng)
        assert self.protected(physical=True)==before
        self.artifact('transfer_'+str(self.completed),receipt)
        self.held=self.completed;self.held_signature=self.nn_signature;self.phase='after_controller'
        self.sync();self.times['boundary_s']+=time.perf_counter()-t
        if self.completed==20:self.save('after_controller_20')
    def end_of_time_s(self,st,eng):
        assert st is self.st and eng is self.eng and self.completed in [20,40]
        assert self.phase=='committed_before_endpoint_handoff'
        self.save('committed_'+str(self.completed))
        if self.completed==40:raise BudgetStop()
        t=time.perf_counter();before=self.protected();error=self.ctx.endpoint.end_of_time_s(st,eng)
        assert self.protected()==before
        self.artifact('endpoint_20',dict(error=error));self.phase='after_strict_endpoint'
        self.sync();self.times['boundary_s']+=time.perf_counter()-t;self.save('after_endpoint_20')
    def propagate(self,sr,phi_i,strict=False,phi_i_iv=None):
        assert strict is True and phi_i_iv is not None and sr is self.sr and self.pending_status is not None
        self.propagate_count+=1;assert self.propagate_count==1
        return self.ledger.propagate(sr,phi_i,phi_i_iv,status_before=self.pending_status,raw_propagate=self.rawprop,sym=self.sym)
    def advance_sparse(self,st,code,eng,sched,settings,cap,sr):
        assert st is self.st and eng is self.eng and cap is self.cap and sr is self.sr
        assert self.completed<40 and self.phase in ['after_controller','committed_before_endpoint_handoff']
        if self.completed==21:self.save('committed_21')
        self.pending_status=st.status.clone();self.propagate_count=0;self.sync();t=time.perf_counter()
        try:
            state,ok=self.raw['advance_sparse'](st,code,eng,sched,settings,cap,sr)
            assert self.propagate_count==1
            self.ledger.finish_step(sr,state.status,token=self.completed+1)
        finally:self.pending_status=None
        self.sync();duration=time.perf_counter()-t;self.times['advance_s']+=duration
        self.completed+=1;self.st=state;self.phase='committed_before_endpoint_handoff';self.ok=ok.detach().cpu().clone()
        assert self.ok.shape==(1024,) and self.ok.dtype==self.t.bool
        self.broken|=(~self.ok)&(state.status.cpu()!=self.d.ACTIVE)
        self.last_accepted[self.ok]=self.completed
        self.observe(duration)
        allocated=self.t.cuda.memory_allocated();reserved=self.t.cuda.memory_reserved()
        if reserved-allocated>2*GIB:
            t=time.perf_counter();self.t.cuda.empty_cache();self.sync();assert self.t.cuda.memory_allocated()==allocated
            elapsed=time.perf_counter()-t;self.times['cache_release_s']+=elapsed
            self.cache_releases.append(dict(step=self.completed,allocated=allocated,reserved_before=reserved,reserved_after=self.t.cuda.memory_reserved(),elapsed_s=elapsed))
        return state,ok
    def prune_state(self,st,eng):
        assert st is self.st and eng is self.eng
        self.st=self.raw['prune_state'](st,eng);return self.st
    def observe(self,duration):
        t=time.perf_counter();st=self.st
        versions=[(getattr(st,k),getattr(st,k)._version) for k in ['pre','pre_rem','tmv','tmv_rem','status']]
        tube=self.d.hull_ranges_s(st,self.eng,12)
        piece=self.t.full((1024,2),.005,dtype=self.t.float64,device=st.pre.device)
        endpoint=self.d.rows_range_over_time_sparse(st,self.eng,piece,12)
        bounds=self.owned(self.t.cat((tube,endpoint),dim=-1))
        assert all(x._version==v for x,v in versions)
        assert bool(self.t.isfinite(bounds[self.ok]).all())
        assert bool((bounds[self.ok,:,0]<=bounds[self.ok,:,1]).all() and (bounds[self.ok,:,2]<=bounds[self.ok,:,3]).all())
        self.artifact('observer_'+str(self.completed),dict(bounds=bounds,accepted=self.ok,status=st.status))
        row=dict(step=self.completed,accepted=self.ok.tolist(),status=st.status.cpu().tolist(),advance_s=duration,
            accepted_lanes=int(self.ok.sum()),broken_lanes=int(self.broken.sum()),sr_length=self.ledger.length,host_epoch=self.ledger.epoch)
        self.rows.append(row)
        with (self.args.output/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.sync();self.times['observer_s']+=time.perf_counter()-t
        write(self.args.output/'progress.json',dict(step=self.completed,accepted_lane_steps=sum(r['accepted_lanes'] for r in self.rows),phase=self.phase))
        if self.completed%20==0:print('PROGRESS',self.completed,row['accepted_lanes'],flush=True)


def snapshot_summary(path):
    """Verify every streamed file, compare numerical raw hashes, not zip metadata."""
    path=Path(path);m=read(path/'MANIFEST.json');assert m['serializer_sha256']==PINS['snapshot.py']
    for name,digest in m['files'].items():assert sha(path/name)==digest
    s=read(path/'sr/MANIFEST.json');assert s['adapter_sha256']==HOST_SHA
    for tensor in s['tensors'].values():
        for block in tensor['blocks']:
            p=path/'sr'/block['file'];assert p.stat().st_size==block['size'] and sha(p)==block['sha256']
    return m,s

def compare_snapshot(runtime,left,right):
    a,sa=snapshot_summary(left);b,sb=snapshot_summary(right)
    assert a['metadata']==b['metadata']==runtime.ctx.identity and a['progress']==b['progress']
    ta=runtime.t.load(Path(left)/'plant.pt',map_location='cpu',weights_only=True)
    tb=runtime.t.load(Path(right)/'plant.pt',map_location='cpu',weights_only=True)
    assert runtime.signature(ta)==runtime.signature(tb)
    assert sa['metadata']==sb['metadata'] and set(sa['tensors'])==set(sb['tensors'])
    for name in sa['tensors']:
        for key in ['shape','dtype','raw_sha256']:assert sa['tensors'][name][key]==sb['tensors'][name][key],(name,key)
    return dict(plant_all_fields=True,SR_and_host_all_components=True,progress=True)

def compare_artifact(runtime,left,right):
    left,right=Path(left),Path(right)
    for p in [left,right]:
        side=read(p.with_suffix('.json'));assert side['source_identity']==runtime.ctx.identity and side['pt_sha256']==sha(p)
        data=runtime.t.load(p,map_location='cpu',weights_only=True);assert side['signature']==runtime.signature(data)
    a=runtime.t.load(left,map_location='cpu',weights_only=True);b=runtime.t.load(right,map_location='cpu',weights_only=True)
    assert runtime.signature(a)==runtime.signature(b)
    return True


def cold(runtime,source):
    x=runtime;ctx=x.ctx;d=x.d;source=Path(source)
    binding=own40_gate(source,ctx.identity)
    # Original factory sequence except initial state/SR: restore exactly one SR.
    cfg=ctx.cfg;names=[e['name'] for e in cfg['initial_set']]
    settings=d.Settings(step=.005,order=2,cutoff=1e-6,remainder_estimation=d.parse_rem_est(cfg['remainder_estimation'],16),mode='strict',device='cuda:0')
    model=d.build_crown(cfg,'cuda:0',relax='same-slope',input_layout='native')
    code=d.compile_ode(cfg['dynamics_expressions'],names,order=1)
    tables=d.build_tables(16,2).to('cuda:0');step=d.poly.build_step_tables(tables,.005);sched=d.build_schedule(16,2,'cuda:0');eng=d.SparseEngine(tables,step,'cuda:0')
    st,sr,ledger,cap,progress=ctx.snapshot.restore(source/'committed_20',eng,metadata=ctx.identity,host=x.h,raw_factory=x.raw['make_symbolic_remainder'])
    assert progress['completed_step']==20 and progress['phase']=='committed_before_endpoint_handoff' and progress['held_controller_step']==0
    assert sr.max_size==len(sr.phi_buf)==1000 and ledger.length==20 and ledger.epoch==0
    expected_cap=x.raw['build_rem_est'](settings,16,1024,'cuda:0')
    assert x.signature(cap)==x.signature(expected_cap);del expected_cap
    x.st,x.sr,x.ledger,x.cap,x.completed=st,sr,ledger,cap,20
    x.held=progress['held_controller_step'];x.held_signature=progress['held_controller_signature']
    x.broken=x.t.tensor(progress['broken'],dtype=x.t.bool,device='cpu');x.last_accepted=x.t.tensor(progress['last_accepted_step'],dtype=x.t.int64,device='cpu')
    assert not bool(x.broken.any()) and bool((x.last_accepted==20).all());x.attach_reset()
    d.end_of_time_s(st,eng)
    hull=d.hull_ranges_s(st,eng,12)
    T,L,U=d.crown_bounds(model,cfg,hull[...,0].contiguous(),hull[...,1].contiguous(),input_layout='native')
    T,L,U=d.apply_crown_transport(T,L,U,'native-f64');d.inject_controls_s(st,T,L,U,[13,14,15],12)
    st,ok=d.advance_sparse(st,code,eng,sched,settings,cap,sr);st=d.prune_state(st,eng)
    assert bool(ok.all()) and not bool(x.broken.any());assert sr.reset_if_full() is False
    x.save('committed_21')
    checks={name:compare_snapshot(x,source/name,x.args.output/name) for name in ['committed_20','after_endpoint_20','after_controller_20','committed_21']}
    for name in ['endpoint_20','controller_20','transfer_20','observer_21']:checks[name]=compare_artifact(x,source/(name+'.pt'),x.args.output/(name+'.pt'))
    assert x.controller_calls==1 and x.controller_steps==[20]
    return dict(status='passed',mode='cold20_to21',checks=checks,**binding)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception',mode=args.mode)
    try:
        ctx=initialize(args);x=Runtime(ctx,args)
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode=args.mode,original_target_steps=1000,budget_steps=40,
            driver_argv=ctx.boot.driver_argv(ctx),source40=str(args.source40) if args.source40 else None))
        if args.mode=='cold20_to21':result=cold(x,args.source40)
        else:
            argv=sys.argv;sys.argv=ctx.boot.driver_argv(ctx)
            try:
                returned=ctx.d.main();result=dict(status='original_driver_stopped',mode=args.mode,driver_return=returned)
            except BudgetStop:
                assert x.completed==40 and x.ledger.length==x.sr.qlen==x.sr.jlen==40
                result=dict(status='bounded_prefix_completed' if all(all(r['accepted']) for r in x.rows) else 'bounded_prefix_with_failed_lanes',mode=args.mode)
            finally:sys.argv=argv
        result.update(source_identity=ctx.identity,original_target_steps=1000,budget_steps=40,completed_step=x.completed,
            accepted_lane_steps=sum(r['accepted_lanes'] for r in x.rows),all_steps_all_lanes_accepted=bool(x.rows) and all(all(r['accepted']) for r in x.rows),
            controller_calls=x.controller_calls,controller_refresh_steps=x.controller_steps,snapshots=x.snapshots,timing_components=x.times,
            cache_releases=x.cache_releases,fullbatch_qualification=False,end_to_end_strict_certificate=False)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if 'x' in locals():result.update(completed_step=x.completed,phase=x.phase,source_identity=ctx.identity,snapshots=x.snapshots)
        raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-start)
        if (args.output/'INPUT.json').exists():result['input_sha256']=sha(args.output/'INPUT.json')
        if 'ctx' in locals():result.update(max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved())
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['bounded40','cold20_to21'],required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check','host-capacity-check','snapshot-check','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40',type=Path);a=p.parse_args();assert (a.source40 is not None)==(a.mode=='cold20_to21');main(a)
