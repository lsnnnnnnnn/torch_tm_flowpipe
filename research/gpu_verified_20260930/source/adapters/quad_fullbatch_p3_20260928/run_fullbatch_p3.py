"""Explicit workP3/point2/validation4 variant of the original full1024 driver.

Original full-batch NN/status/prune/reset loop; dense working3 and on-demand
validation4. Bounded40 and its own fresh cold20->21, never an old P2 gate.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
BASE_SHA='002e38983c0b5a5c54313a96a61bee0e65f4ccff9e9823f9296de5010c21f304'
PINS={
 'hybrid_metadata.py':'7fd6080c4268478696dc92aac4e397027c7fe0830ba22b3297927e985f8d0dd8',
 'snapshot_p3.py':'9b3f89a7975c84691a7724029270af37ad5e8f3a9f6a6699aac2a339b593bf97',
 'check_hybrid_cpu.py':'f8b3140d1f90e839455a676cb1f886af88281464520fcd125f53fb8c9127a560',
 'check_snapshot_p3.py':'0937f765a5d9c05c53b5744846f98785981e437c9008882a436886682f24c839',
}
SNAPSHOT_CPU_SHA='b3c4b429d6d731f06638a7f8d83948d23ddfa8fba40f8464df91b1c1411113f3'
HYBRID_CPU_SHA='4260c108b3f86e8bcece71b0a4e0e4e2a6033f46094d831697abbd4387be8522'
METADATA_CUDA_SHA='acabe4b4b8ee914cdd33f894e5bcaf3600441b9f583cf8ad6b33acbb0743a018'
HYBRID_CUDA_CHECK_SHA='a552d130ef4dacd69801af002c52080cc2ebcb7df86d08b55c0c1fc0d290b037'
HYBRID_SOURCE_RESULT='5be1ffd7d822f50d0d79adfe5e9528687bcdff691110bd639ddfe4fa9492c177'
HYBRID_SOURCE_SCRIPT='9994a46f0e3ef1b0d1e78c85760f8b50b0416b9cfb16bb7ae3d4cd5e5246f65b'
POLICY='full1024_workP3_point2_validation4_hybrid_directNN_reciprocal5_hostK20'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
BASE=HERE.parent/'quad_fullbatch_sr_20260928/run_fullbatch.py'
assert sha(BASE)==BASE_SHA
w=load('fullbatch_p3_frozen_base',BASE)
BudgetStop=w.BudgetStop

def plain(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value
def config_sha(value):return hashlib.sha256(json.dumps(plain(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def driver_argv(ctx):return ctx.boot.driver_argv(ctx)+['--order','3']

def hybrid_cuda_gate(g,identity):
    assert sha(HERE/'check_hybrid_cubic800.py')==HYBRID_CUDA_CHECK_SHA
    assert g['status']=='passed' and g['device']=='cuda:0'
    assert g['script_sha256']==HYBRID_CUDA_CHECK_SHA
    assert g['adapter_sha256']==PINS['hybrid_metadata.py']
    assert g['engine_python_sha256']==identity['engine']['python_sha256']
    assert g['extensions']==identity['extensions'] and len(g['extensions'])==8
    assert g['reciprocal_extensions']==identity['reciprocal_extensions'] and g['reciprocal_candidate']==identity['reciprocal_candidate']
    assert g['prior_hybrid_CPU_result_sha256']==HYBRID_CPU_SHA
    assert g['source_result_sha256']==HYBRID_SOURCE_RESULT and g['source_script_sha256']==HYBRID_SOURCE_SCRIPT
    assert len(g['source_files_sha256'])==11 and g['source_files_sha256']['RESULT.json']==HYBRID_SOURCE_RESULT
    assert all(len(v)==64 for v in g['source_files_sha256'].values())
    assert (g['working_order'],g['point_code_order'],g['validation_order'])==(3,2,4)
    for key in ['working3_dense','validation4_metadata','dense4_not_constructed']:assert g[key] is True
    assert g['blocked_dense4_negative']==['monomials','support','driver']
    assert g['dense_builder_calls'] and all(v==dict(n=16,k=3) for v in g['dense_builder_calls'])
    assert len(g['prefix_receipts'])==1 and g['prefix_receipts'][0]['prefix_checked'] is True
    assert g['prefix_receipts'][0]['full_degree_six_prefix']==dict(full=100947,spatial=74613)
    assert len(g['rows'])==4
    groups=[[497,609],[625,737,753,849,865,881,961,977,993,1009]]
    for row,(label,lanes) in zip(g['rows'],[(label,lanes) for lanes in groups for label in ['P2_zero_slots_r4','proposed_P3_r4']]):
        assert row['label']==label and row['global_lanes']==lanes
        assert row['source_sha256']==g['source_files_sha256'][row['source_file']]
        for key in ['same_R_image_bytes_equal','same_R_bad_bytes_equal','accepted_bytes_equal','emitted_remainder_bytes_equal',
                    'events_float_bytes_equal','statistics_equal','input_all_bytes_unchanged']:assert row[key] is True
        expected=[True]*len(lanes) if label=='proposed_P3_r4' else [lane==609 for lane in lanes]
        assert row['accepted']==expected

def initialize(args):
    for name,digest in PINS.items():assert sha(HERE/name)==digest,name
    paths={k:getattr(args,k)/'RESULT.json' for k in ['p3_snapshot_check','hybrid_cpu_check','metadata_check','hybrid_check']}
    assert sha(paths['p3_snapshot_check'])==SNAPSHOT_CPU_SHA and sha(paths['hybrid_cpu_check'])==HYBRID_CPU_SHA
    assert sha(paths['metadata_check'])==METADATA_CUDA_SHA
    snap,local,meta=(read(paths[k]) for k in ['p3_snapshot_check','hybrid_cpu_check','metadata_check'])
    assert snap['status']=='passed' and snap['working_order']==3 and snap['batch']==1024
    assert snap['serializer_sha256']==PINS['snapshot_p3.py'] and snap['script_sha256']==PINS['check_snapshot_p3.py']
    assert snap['nonzero_cubic_full_and_spatial'] and snap['full_plant_support_cap_status_and_SR_ledger_bytes_equal']
    assert snap['next_rawprop_image_all_bytes_equal'] and snap['negative_count']==11
    assert local['status']=='passed' and local['device']=='cpu' and local['adapter_sha256']==PINS['hybrid_metadata.py']
    assert local['all_state_support_SR_acceptance_bytes_equal'] and local['validation_owner_and_cache_distinct']
    ctx=w.initialize(args)  # Frozen source/8SO/reciprocal/SR/boundary admissions; no engine/NN yet.
    baseline_identity=copy.deepcopy(ctx.identity)
    assert snap['engine_python_sha256']==local['engine_python_sha256']==meta['engine_python_sha256']==ctx.identity['engine']['python_sha256']
    assert meta['extensions']==ctx.identity['extensions'] and meta['torch_version']==ctx.torch.__version__
    assert meta['status']=='passed' and meta['device']=='cuda' and meta['dense_sparse_complete_bytes_equal'] and meta['weighted_plan_fields_equal']
    assert meta['standard_no_callback_cases']==meta['diagnostic_callback_cases']==5
    assert meta['fraction_component_checks_per_backend']==4000
    cuda=read(paths['hybrid_check']);hybrid_cuda_gate(cuda,ctx.identity)
    assert cuda['torch_version']==ctx.torch.__version__
    original_cfg=plain(ctx.cfg);ctx.cfg=copy.deepcopy(ctx.cfg);ctx.cfg['ode_order']=3
    resolved=plain(ctx.cfg)
    assert {k for k in original_cfg if original_cfg[k]!=resolved[k]}=={'ode_order'}
    ctx.snapshot=load('fullbatch_p3_snapshot',HERE/'snapshot_p3.py')
    ctx.hybrid_module=load('fullbatch_p3_hybrid',HERE/'hybrid_metadata.py')
    ctx.hybrid=ctx.hybrid_module.install(ctx.identity,args.metadata_backend,extra_modules=(ctx.c,ctx.d))
    assert meta['backend_sha256']==cuda['backend_sha256']==ctx.hybrid.backend_sha256
    assert cuda['dependencies']==ctx.hybrid.dependencies
    ctx.identity=dict(ctx.identity,script_sha256=sha(__file__),policy=POLICY,
        base_runner_sha256=BASE_SHA,base_qualification_identity=baseline_identity,
        settings=dict(ctx.identity['settings'],order=3),working_total_degree=3,point_code_order=2,validation_order=4,
        metadata_policy=ctx.hybrid.policy,hybrid_adapter_sha256=PINS['hybrid_metadata.py'],
        metadata_backend_sha256=ctx.hybrid.backend_sha256,metadata_dependencies=ctx.hybrid.dependencies,
        snapshot_sha256=PINS['snapshot_p3.py'],p3_qualification_receipts={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        config_overrides={'ode_order':dict(original=2,resolved=3)},resolved_config=resolved,resolved_config_sha256=config_sha(resolved),
        fullbatch_qualification=False,end_to_end_strict_certificate=False,
        limitation='Explicit workP3/point2/validation4 variant; original P2 result retained. Own40 and owncold only; CROWN/same-slope certificates remain conditional.')
    return ctx

def own40_gate(source,identity):
    source=Path(source);inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==identity
    assert identity['script_sha256']==sha(__file__) and identity['policy']==POLICY
    assert identity['working_total_degree']==3 and identity['point_code_order']==2 and identity['validation_order']==4
    assert r['input_sha256']==sha(source/'INPUT.json')
    assert inp['mode']==r['mode']=='bounded40' and inp['original_target_steps']==r['original_target_steps']==1000
    assert inp['budget_steps']==r['budget_steps']==40 and r['status']=='bounded_prefix_completed'
    assert r['completed_step']==40 and r['accepted_lane_steps']==40960 and r['all_steps_all_lanes_accepted'] is True
    assert r['controller_calls']==2 and r['controller_refresh_steps']==[0,20]
    assert r['pruned_checkpoint20_support_cache_verified'] is True
    assert len(r['hybrid_validation_owners'])==1
    owner=r['hybrid_validation_owners'][0]
    assert owner['n']==16 and owner['working']==3 and owner['validation']==4 and owner['prefix_checked'] is True
    assert owner['full_degree_six_prefix']==dict(full=100947,spatial=74613)
    assert set(r['snapshots'])=={'committed_20','after_endpoint_20','after_controller_20','committed_21','committed_40'}
    for name,digest in r['snapshots'].items():assert sha(source/name/'MANIFEST.json')==digest
    return dict(source_input_sha256=sha(source/'INPUT.json'),source_result_sha256=sha(source/'RESULT.json'))

class Runtime(w.Runtime):
    def save(self,name):
        if name=='committed_20':
            assert self.eng._acc_sups[('state',)]==(self.st.pre_sup,self.st.tmv_sup)
            self.pruned_checkpoint20_verified=True
        return super().save(name)
    def make_cells(self,config):
        assert config['steps']==50 and config['ode_order']==3
        assert plain(config)==self.ctx.identity['resolved_config']
        return self.ctx.boxes.clone()
    def SparseEngine(self,*args,**kwargs):
        eng=super().SparseEngine(*args,**kwargs)
        assert eng.tables.n==16 and eng.tables.k==3 and not isinstance(eng,self.ctx.hybrid.backend.MetadataEngine)
        return eng
    def advance_sparse(self,st,code,eng,sched,settings,cap,sr):
        assert settings.order==3 and settings.step==.005 and code.order==2 and eng.tables.k==3
        return super().advance_sparse(st,code,eng,sched,settings,cap,sr)

def snapshot_summary(path):
    path=Path(path);m=read(path/'MANIFEST.json');assert m['serializer_sha256']==PINS['snapshot_p3.py']
    for name,digest in m['files'].items():assert sha(path/name)==digest
    sr=read(path/'sr/MANIFEST.json');assert sr['adapter_sha256']==w.HOST_SHA
    for tensor in sr['tensors'].values():
        for block in tensor['blocks']:
            p=path/'sr'/block['file'];assert p.stat().st_size==block['size'] and sha(p)==block['sha256']
    return m,sr

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

compare_artifact=w.compare_artifact


def cold(runtime,source):
    x=runtime;ctx=x.ctx;d=x.d;source=Path(source)
    binding=own40_gate(source,ctx.identity)
    # Original factory sequence except initial state/SR: restore exactly one SR.
    cfg=ctx.cfg;names=[e['name'] for e in cfg['initial_set']]
    settings=d.Settings(step=.005,order=3,cutoff=1e-6,remainder_estimation=d.parse_rem_est(cfg['remainder_estimation'],16),mode='strict',device='cuda:0')
    model=d.build_crown(cfg,'cuda:0',relax='same-slope',input_layout='native')
    code=d.compile_ode(cfg['dynamics_expressions'],names,order=2)
    tables=d.build_tables(16,3).to('cuda:0');step=d.poly.build_step_tables(tables,.005);sched=d.build_schedule(16,3,'cuda:0');eng=d.SparseEngine(tables,step,'cuda:0')
    st,sr,ledger,cap,progress=ctx.snapshot.restore(source/'committed_20',eng,metadata=ctx.identity,host=x.h,raw_factory=x.raw['make_symbolic_remainder'])
    assert progress['completed_step']==20 and progress['phase']=='committed_before_endpoint_handoff' and progress['held_controller_step']==0
    assert sr.max_size==len(sr.phi_buf)==1000 and ledger.length==20 and ledger.epoch==0
    expected_cap=x.raw['build_rem_est'](settings,16,1024,'cuda:0')
    assert x.signature(cap)==x.signature(expected_cap);del expected_cap
    x.st,x.sr,x.ledger,x.cap,x.completed=st,sr,ledger,cap,20
    x.held=progress['held_controller_step'];x.held_signature=progress['held_controller_signature']
    x.broken=x.t.tensor(progress['broken'],dtype=x.t.bool,device='cpu');x.last_accepted=x.t.tensor(progress['last_accepted_step'],dtype=x.t.int64,device='cpu')
    assert not bool(x.broken.any()) and bool((x.last_accepted==20).all());x.attach_reset()
    ctx.c.se._eng_cache(eng,'_acc_sups')[('state',)]=(st.pre_sup,st.tmv_sup)
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
            driver_argv=driver_argv(ctx),source40=str(args.source40) if args.source40 else None))
        if args.mode=='cold20_to21':result=cold(x,args.source40)
        else:
            argv=sys.argv;sys.argv=driver_argv(ctx)
            try:
                returned=ctx.d.main();result=dict(status='original_driver_stopped',mode=args.mode,driver_return=returned)
            except BudgetStop:
                assert x.completed==40 and x.ledger.length==x.sr.qlen==x.sr.jlen==40
                result=dict(status='bounded_prefix_completed' if all(all(r['accepted']) for r in x.rows) else 'bounded_prefix_with_failed_lanes',mode=args.mode)
            finally:sys.argv=argv
        result.update(source_identity=ctx.identity,original_target_steps=1000,budget_steps=40,completed_step=x.completed,
            accepted_lane_steps=sum(r['accepted_lanes'] for r in x.rows),all_steps_all_lanes_accepted=bool(x.rows) and all(all(r['accepted']) for r in x.rows),
            controller_calls=x.controller_calls,controller_refresh_steps=x.controller_steps,snapshots=x.snapshots,timing_components=x.times,
            cache_releases=x.cache_releases,hybrid_validation_owners=ctx.hybrid.receipts,guarded_dense_aliases=ctx.hybrid.guarded_dense_aliases,
            pruned_checkpoint20_support_cache_verified=getattr(x,'pruned_checkpoint20_verified',False),fullbatch_qualification=False,end_to_end_strict_certificate=False)
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
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check','host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40',type=Path);a=p.parse_args();assert (a.source40 is not None)==(a.mode=='cold20_to21');main(a)
