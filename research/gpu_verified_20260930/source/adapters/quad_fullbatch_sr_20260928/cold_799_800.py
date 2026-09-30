"""Fresh full-B799->800 replay from its actual captured parent; no NN refresh.

Restore one original SR and host ledger. Original advance/prune/reset and all
saved800 plant/history/observer fields must match. Source files stay untouched.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback
HERE=Path(__file__).parent
CAPTURE_SHA='d7a1109b808f6845afbe94c5555f7e3be16583b979f2c10f46d274c66f61664a'
LONG_SHA='7b465c72046b4c23497ace95f4b3d74cba8fd556cbb48a5454ae0af1b3772c98'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def load():
    p=HERE/'run_fullbatch_long.py';assert sha(p)==LONG_SHA
    spec=importlib.util.spec_from_file_location('cold799_frozen_long',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
l=load();w=l.w;read,write=l.read,l.write


def admission(args,baseline):
    assert sha(HERE/'capture_799_800.py')==CAPTURE_SHA
    source=args.capture;inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert sha(source/'RESULT.json')==args.capture_result_sha256 and r['input_sha256']==sha(source/'INPUT.json')
    assert r['script_sha256']==CAPTURE_SHA and inp['mode']==r['mode']=='capture799_800'
    assert r['status']=='passed_capture799_and_matched800' and r['completed_step']==800
    assert r['overall_horizon_completed'] is False and r['original_target_steps']==1000 and r['budget_steps']==800
    expected=dict(baseline,script_sha256=CAPTURE_SHA,capture_wrapper_parent_sha256=LONG_SHA,
        execution_budget_steps=800,reference_long_result_sha256=inp['source_identity']['reference_long_result_sha256'],
        checkpoint_policy='exactly799parent_and800poststep_before_endpoint;full_original_batch_and_loop')
    assert inp['source_identity']==r['source_identity']==expected
    assert r['controller_calls']==40 and r['controller_refresh_steps']==list(range(0,800,20))
    assert r['no_endpoint800_or_controller800'] is True and r['first_failure']['step']==800 and len(r['first_failure']['failed_lanes'])==12
    assert r['accepted_lane_steps']==799*1024+1012
    check=r['comparisons'];assert check['observer_numerical_signatures_equal_steps']==check['step_status_masks_equal_steps']==800
    assert check['SR_ledger_component_count']==14 and check['reference_large_component_bytes_reread'] is False
    for key in ['plant800_saved_signature_equal','progress800_equal','SR_ledger800_manifest_raw_hashes_equal','held780_controller_and_transfer_signatures_equal']:assert check[key] is True
    assert set(r['snapshots'])=={'committed_799','committed_800'}
    roots={}
    for name,k in [('committed_799',799),('committed_800',800)]:
        p=source/name/'MANIFEST.json';assert sha(p)==r['snapshots'][name];m=read(p)
        assert m['metadata']==expected and m['serializer_sha256']==w.PINS['snapshot.py']
        g=m['progress'];assert g['completed_step']==k and g['next_step']==k+1 and g['phase']=='committed_before_endpoint_handoff'
        assert g['held_controller_step']==780 and g['next_control_refresh_step']==800 and g['sr_history_origin_step']==0
        roots[k]=m
    support=read(source/'PRUNE_SUPPORT_799.json');assert sha(source/'PRUNE_SUPPORT_799.json')==r['prune_support799_sha256']
    assert support['source_identity']==expected and support['completed_step']==799 and support['accumulator_key']==['state']
    assert support['source_sparse_exec_sha256']==baseline['engine']['python_sha256']['src/flowstar_gpu/sparse_exec.py']
    assert support['snapshot_manifest_sha256']==r['snapshots']['committed_799']
    assert len(support['supports'])==2
    for value,spatial in zip(support['supports'],[False,True]):
        assert value['n']==16 and value['k']==2 and value['spatial'] is spatial and value['ids']==sorted(set(value['ids'])) and value['ids'][0]==0
    rows=[json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines() if line.strip()]
    assert [v['step'] for v in rows]==list(range(1,801))
    assert all(len(v['accepted'])==len(v['status'])==1024 and all(type(x) is bool for x in v['accepted']) for v in rows)
    assert all(all(v['accepted']) for v in rows[:-1]) and sum(rows[-1]['accepted'])==1012
    assert not any(roots[799]['progress']['broken']) and roots[799]['progress']['last_accepted_step']==[799]*1024
    for name in ['controller_780','transfer_780']:
        p=source/(name+'.json');side=read(p)
        assert side['source_identity']==expected and side['step']==780 and sha(source/(name+'.pt'))==side['pt_sha256']
    assert read(source/'controller_780.json')['signature']==roots[799]['progress']['held_controller_signature']
    files=['INPUT.json','RESULT.json','steps.jsonl','PRUNE_SUPPORT_799.json','controller_780.json','controller_780.pt','transfer_780.json','transfer_780.pt',
        'committed_799/MANIFEST.json','committed_800/MANIFEST.json','observer_800.json','observer_800.pt']
    return dict(source=source,metadata=expected,roots=roots,support=support,last=rows[-1],
        receipt=dict(source_capture=str(source),capture_result_sha256=args.capture_result_sha256,capture_input_sha256=sha(source/'INPUT.json'),
            source_files_sha256={name:sha(source/name) for name in files},capture_script_sha256=CAPTURE_SHA,long_script_sha256=LONG_SHA))


def forbidden(*a,**kw):raise AssertionError('Held780 replay must not construct/update NN or apply endpoint/injection')

class ColdRuntime(l.LongRuntime):
    build_crown=forbidden
    crown_bounds=forbidden
    inject_controls_s=forbidden
    end_of_time_s=forbidden


def compare_snapshot(x,source,actual,metadata):
    # Reuse the actual full component-file hash verifier. Unlike short cold,
    # this diagnostic has its own identity, so validate each side separately.
    a,sa=w.snapshot_summary(source);b,sb=w.snapshot_summary(actual)
    assert a['metadata']==metadata and b['metadata']==x.ctx.identity
    assert a['progress']==b['progress']
    va=x.t.load(Path(source)/'plant.pt',map_location='cpu',weights_only=True)
    vb=x.t.load(Path(actual)/'plant.pt',map_location='cpu',weights_only=True)
    assert x.signature(va)==x.signature(vb);del va,vb
    assert sa['metadata']==sb['metadata'] and set(sa['tensors'])==set(sb['tensors'])
    for name in sa['tensors']:
        for key in ['shape','dtype','raw_sha256']:assert sa['tensors'][name][key]==sb['tensors'][name][key],(name,key)
    return dict(all_plant_support_cap_status_bytes=True,all_SR_ledger_component_raw_hashes=True,
        component_count=len(sa['tensors']),all_saved_component_file_hashes_verified=True,progress_equal=True,metadata_identities_checked_separately=True)


def compare_observer(x,source,actual,metadata):
    for p,identity in [(source,metadata),(actual,x.ctx.identity)]:
        side=read(p.with_suffix('.json'));assert side['source_identity']==identity and side['step']==800 and side['pt_sha256']==sha(p)
        value=x.t.load(p,map_location='cpu',weights_only=True);assert x.signature(value)==side['signature'];del value
    a=x.t.load(source,map_location='cpu',weights_only=True);b=x.t.load(actual,map_location='cpu',weights_only=True)
    assert x.signature(a)==x.signature(b)
    return dict(bounds_all1024x12_bytes=True,accepted_status_bytes=True,source_and_actual_PT_hashes_verified=True)


def replay(ctx,args,source):
    x=ColdRuntime(ctx,args);d=x.d;cfg=ctx.cfg;started=time.perf_counter()
    # Verify the entire saved799 component-file chain before restore/allocation.
    root,manifest=w.snapshot_summary(source['source']/'committed_799')
    assert root==source['roots'][799] and manifest['metadata']['qlen']==manifest['metadata']['jlen']==799
    x.times['parent_stream_file_validation_s']=time.perf_counter()-started
    settings=d.Settings(step=.005,order=2,cutoff=1e-6,remainder_estimation=d.parse_rem_est(cfg['remainder_estimation'],16),mode='strict',device='cuda:0')
    code=d.compile_ode(cfg['dynamics_expressions'],[v['name'] for v in cfg['initial_set']],order=1)
    tables=d.build_tables(16,2).to('cuda:0');step=d.poly.build_step_tables(tables,.005);sched=d.build_schedule(16,2,'cuda:0');eng=d.SparseEngine(tables,step,'cuda:0')
    started=time.perf_counter();assert x.sr is None and x.ledger is None
    # Pass the actual frozen factory, not a counting wrapper: the host loader
    # checks inspect.getfile(raw_factory) against the symbolic source SHA.
    st,sr,ledger,cap,progress=ctx.snapshot.restore(source['source']/'committed_799',eng,
        metadata=source['metadata'],host=x.h,raw_factory=x.raw['make_symbolic_remainder'])
    x.times['parent_stream_restore_s']=time.perf_counter()-started
    assert progress==source['roots'][799]['progress'] and sr.qlen==sr.jlen==ledger.length==799 and ledger.epoch==0
    assert sr.max_size==len(sr.phi_buf)==1000 and ledger.batch==1024 and ledger.lane_chunk==16
    assert bool((st.status==0).all()) and bool((ledger.current_status==0).all())
    expected_cap=x.raw['build_rem_est'](settings,16,1024,'cuda:0');assert x.signature(cap)==x.signature(expected_cap);del expected_cap
    x.st,x.sr,x.ledger,x.cap,x.completed=st,sr,ledger,cap,799
    x.held=progress['held_controller_step'];x.held_signature=progress['held_controller_signature'];x.phase=progress['phase']
    x.broken=x.t.tensor(progress['broken'],dtype=x.t.bool,device='cpu');x.last_accepted=x.t.tensor(progress['last_accepted_step'],dtype=x.t.int64,device='cpu')
    x.attach_reset()
    values=source['support']['supports']
    for current,saved in zip((st.pre_sup,st.tmv_sup),values):
        assert dict(n=current.n,k=current.k,spatial=current.spatial,ids=list(current.ids))==saved
    assert not getattr(eng,'_acc_sups',{})
    eng._acc_sups={('state',):(st.pre_sup,st.tmv_sup)}  # exact recorded source layout
    # Verify actual held certificate bytes, but never apply them a second time.
    for name in ['controller_780','transfer_780']:
        p=source['source']/(name+'.pt');value=x.t.load(p,map_location='cpu',weights_only=True)
        assert x.signature(value)==read(p.with_suffix('.json'))['signature'];del value
    before_files=source['receipt']['source_files_sha256']
    st,ok=d.advance_sparse(st,code,eng,sched,settings,cap,sr);st=d.prune_state(st,eng)
    assert x.completed==800 and x.ok.tolist()==source['last']['accepted'] and st.status.cpu().tolist()==source['last']['status']
    assert sr.reset_if_full() is False and sr.qlen==sr.jlen==ledger.length==800 and ledger.epoch==0
    assert x.controller_calls==0 and x.controller_steps==[] and x.held==780
    x.save('committed_800')
    started=time.perf_counter();checks=compare_snapshot(x,source['source']/'committed_800',args.output/'committed_800',source['metadata'])
    observer=compare_observer(x,source['source']/'observer_800.pt',args.output/'observer_800.pt',source['metadata'])
    for name,digest in before_files.items():assert sha(source['source']/name)==digest
    x.times['full_output_comparison_s']=time.perf_counter()-started
    return dict(status='passed',mode='cold799_to800',completed_step=800,replayed_step=800,parent_step=799,
        accepted=x.ok.tolist(),status_codes=st.status.cpu().tolist(),held_controller_step=780,
        controller_calls=0,no_endpoint_or_control_update=True,one_SR_restore_no_parent_deepcopy=True,parent_snapshot_resaved=False,
        source799_all_component_file_and_raw_hashes_verified=True,checks=checks,observer=observer,
        restored_prune_layout_exactly_recorded=True,snapshots=x.snapshots,timing_components=x.times,cache_releases=x.cache_releases,
        first_failure=x.first_failure,source_receipt=source['receipt'],source_identity=ctx.identity,
        scope='Cold replay of full1024 original failed800, not accepted continuation or a new NNCS proof. Source799 streamed+rawhashed on restore; source800 component-file hashes and saved component raw hashes compared.')


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception',mode='cold799_to800')
    try:
        ctx=l.initialize(args);source=admission(args,dict(ctx.identity))
        ctx.identity=dict(ctx.identity,script_sha256=sha(__file__),cold_wrapper_parent_sha256=LONG_SHA,capture_script_sha256=CAPTURE_SHA,
            capture_input_sha256=source['receipt']['capture_input_sha256'],capture_result_sha256=args.capture_result_sha256,
            execution_budget_steps=1,parent_completed_step=799,replayed_step=800,held_controller_step=780,
            checkpoint_policy='restore_only_capture799_once;save_only_replayed800',no_endpoint_or_control_update=True)
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode='cold799_to800',source=source['receipt'],source_checkpoint_identity=source['metadata']))
        result=replay(ctx,args,source)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-start)
        if (args.output/'INPUT.json').exists():result['input_sha256']=sha(args.output/'INPUT.json')
        if 'ctx' in locals():result.update(max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved())
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check','host-capacity-check','snapshot-check','output','source40','cold','capture']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['source40-result-sha256','cold-result-sha256','capture-result-sha256']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
