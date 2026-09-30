"""Own40/cold for selective audit of the unchanged working-graph eviction.

Nonempty eviction keeps full before/after state/SR/host byte checks. A no-op
keeps storage/version/capacity checks and never claims a mathematical byte
check. All40 observers and five full snapshots must match the original e16c40.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
SHORT_SHA='e16c197e8338c7798a98509ade44804ca119278a26334aaa47d1a2ac6d38316a'
HELPER_SHA='3658f1fced8b47a601b06e261b9a86fa193339dd1df55be64db9a418c558c9e7'
CPU_CHECK_SHA='ad78926c489429d44ef74391b1cd4411ab842d85c077778064f9404f5fa22fe1'
CPU_RESULT_SHA='a1eaf12b68fc6868681db5b12ba4b0158a46878fa26e952827116749caf2ad69'
REFERENCE_INPUT_SHA='8628b1684844c639eade578bfde63cd58b2ec7a00f04c6bfce2fdd3b93494bd6'
REFERENCE_RESULT_SHA='d667153737fc42239b296ff37fce6aed8a79862c20ee93c6ca2c154bc943f3dd'
POLICY='full1024_P3_eager_selective_working_eviction_audit_own40_cold'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

e=load('selective_audit_frozen_eager',HERE/'run_fullbatch_p3_working_eager.py',SHORT_SHA)
h,b,w=e.h,e.b,e.w
BudgetStop=e.BudgetStop;driver_argv=e.driver_argv;eager_counters=e.eager_counters;eager_log_gate=e.log_gate
BASE_SHA,ADAPTER_SHA,CHECK_RESULT_SHA=e.BASE_SHA,e.ADAPTER_SHA,e.CHECK_RESULT_SHA
OBSERVER_ENTRY_SHA,ENDPOINT_POLICY=e.OBSERVER_ENTRY_SHA,e.ENDPOINT_POLICY
EAGER_ADAPTER_SHA,EAGER_POLICY,EAGER_RESULT_SHA=e.EAGER_ADAPTER_SHA,e.EAGER_POLICY,e.EAGER_RESULT_SHA
SNAPSHOTS=('committed_20','after_endpoint_20','after_controller_20','committed_21','committed_40')
P3=w.w.w.w
FULL_HASH_STEPS=[2,3,4,9,10,20,28,30,33]

def qualification(args,identity=None):
    assert len(HELPER_SHA)==len(CPU_CHECK_SHA)==len(CPU_RESULT_SHA)==64,'Selective helper CPU gate is not frozen'
    assert sha(HERE/'selective_prune_audit.py')==HELPER_SHA
    assert sha(args.selective_audit_checker)==CPU_CHECK_SHA
    path=args.selective_audit_check/'RESULT.json';assert sha(path)==CPU_RESULT_SHA
    r=read(path)
    assert r['status']=='passed' and r['device']=='cpu'
    assert r['script_sha256']==CPU_CHECK_SHA and r['helper_sha256']==HELPER_SHA
    # Frozen CPU-result content supplies the complete small dispatch contract.
    reference=Path(args.reference40)
    assert sha(reference/'INPUT.json')==REFERENCE_INPUT_SHA and sha(reference/'RESULT.json')==REFERENCE_RESULT_SHA
    base=read(reference/'INPUT.json')['source_identity']
    assert base['script_sha256']==SHORT_SHA
    if identity is not None:assert base==identity
    e.own40_gate(reference,base)
    return dict(helper_sha256=HELPER_SHA,cpu_checker_sha256=CPU_CHECK_SHA,cpu_result_sha256=CPU_RESULT_SHA,
        reference_path=str(reference),reference_input_sha256=REFERENCE_INPUT_SHA,reference_result_sha256=REFERENCE_RESULT_SHA)

def initialize(args):
    qualification(args);ctx=e.initialize(args);receipt=qualification(args,ctx.identity)
    assert not ctx.hybrid.receipts
    ctx.selective=load('fullbatch_selective_prune_plan',HERE/'selective_prune_audit.py',HELPER_SHA)
    ctx.selective_reference_identity=dict(ctx.identity);ctx.selective_reference=Path(args.reference40)
    rows=[json.loads(line) for line in (ctx.selective_reference/'working_prune.jsonl').read_text().splitlines()]
    ctx.selective_reference_math={r['step']:r['mathematical_signature'] for r in rows if r['event']=='invariance_checked'}
    assert set(ctx.selective_reference_math)==set(range(1,41))
    ctx.selective_reference_evictions={r['step']:{k:r['eviction'][k] for k in ['evicted_entries','evicted_categories']} for r in rows if r['event']=='eviction_complete'}
    assert set(ctx.selective_reference_evictions)==set(range(1,41))
    assert [step for step,v in ctx.selective_reference_evictions.items() if v['evicted_entries']]==FULL_HASH_STEPS
    ctx.identity=dict(ctx.identity,script_sha256=sha(__file__),policy=POLICY,base_eager_entry_sha256=SHORT_SHA,
        selective_prune_helper_sha256=HELPER_SHA,selective_prune_qualification=receipt,
        working_graph_instrumentation='Nonempty stale plan: original full mathematical before/after hashes plus storage/version/capacity. Empty stale plan: storage/version/capacity only; fullbytes explicitly not checked.',
        fullbatch_qualification=False,end_to_end_strict_certificate=False,
        timing_scope='Advance includes eager dispatch/logging; observer and selective postprune audit are separate. Process includes checkpoint IO and final reference pairing. No-op postprune steps do not hash payloads.',
        limitation='Audit scheduling only; frozen eviction and all solver/NN/SR/validation128/observer64 remain. Own40 requires complete original-e16c observer/five-snapshot byte pairing; owncold remains mandatory. No-op guards do not claim a full payload byte audit.')
    return ctx

class Runtime(e.Runtime):
    def prune_state(self,*args,**kwargs):
        # Skip only b37 full-audit wrapper and e16c eager finish. The original
        # prune is called once; both finishes retain their original order.
        value=b.w.Runtime.prune_state(self,*args,**kwargs);started=time.perf_counter()
        helper=self.ctx.selective;binding=self.ctx.working_prune
        plan=helper.peek(binding,self.eng);storage=b.probe.storage_signature(self)
        full=bool(plan['stale_count'])
        if self.args.mode=='bounded40':
            assert dict(evicted_entries=plan['stale_count'],evicted_categories=plan['stale_categories'])==self.ctx.selective_reference_evictions[self.completed]
        before=b.probe.mathematical_signature(self) if full else None
        helper.assert_same_plan(binding,self.eng,plan)
        self.ctx.working_prune_current_step=self.completed
        eviction=binding.finish_step(self.eng);helper.assert_finished(plan,eviction)
        after=b.probe.mathematical_signature(self) if full else None
        assert b.probe.storage_signature(self)==storage
        if full:
            assert after==before
            assert before==self.ctx.selective_reference_math[self.completed]
        row=dict(event='selective_invariance_checked',step=self.completed,
            phase='after_original_prune_before_poststep_safety_reset',plan=plan,eviction=eviction,
            mathematical_bytes_checked=full,before_after_mathematical_bytes_equal=True if full else None,
            reference_mathematical_bytes_equal=True if full else None,
            mathematical_signature=before,capacity_and_storage_unchanged=True,storage_before=storage,
            elapsed_s=time.perf_counter()-started)
        b.record(self.ctx,row)
        self.ctx.working_prune_rows.append({k:row[k] for k in ['step','elapsed_s','mathematical_bytes_checked',
            'before_after_mathematical_bytes_equal','reference_mathematical_bytes_equal','capacity_and_storage_unchanged']})
        self.ctx.eager_segments.finish_step(self.eng)
        return value

def log_gate(root,steps,result):
    path=Path(root)/'working_prune.jsonl';assert sha(path)==result['working_prune_log_sha256']
    rows=[json.loads(line) for line in path.read_text().splitlines()];assert len(rows)==2*len(steps)
    summaries=[]
    for i,step in enumerate(steps):
        event,row=rows[2*i:2*i+2]
        assert event['event']=='eviction_complete' and row['event']=='selective_invariance_checked'
        assert event['step']==row['step']==step and event['eviction']==row['eviction']
        plan=row['plan'];assert plan['schema']=='frozen_working_prune_readonly_plan_v1'
        full=bool(plan['stale_count']);assert row['mathematical_bytes_checked'] is full
        assert row['eviction']['evicted_entries']==plan['stale_count']
        assert row['eviction']['evicted_categories']==plan['stale_categories']
        assert row['eviction']['validation_and_weighted_untouched'] is True
        assert row['capacity_and_storage_unchanged'] is True
        if full:
            assert row['before_after_mathematical_bytes_equal'] is True and row['reference_mathematical_bytes_equal'] is True
            assert isinstance(row['mathematical_signature'],dict)
        else:
            assert row['before_after_mathematical_bytes_equal'] is None and row['reference_mathematical_bytes_equal'] is None
            assert row['mathematical_signature'] is None
        storage=row['storage_before']
        assert storage['sr_max_size']==storage['host_max_size']==storage['sr_capacity']==1000
        assert storage['qlen']==storage['jlen']==step
        summaries.append({k:row[k] for k in ['step','elapsed_s','mathematical_bytes_checked',
            'before_after_mathematical_bytes_equal','reference_mathematical_bytes_equal','capacity_and_storage_unchanged']})
    assert summaries==result['working_prune_checks']
    if steps==list(range(1,41)):
        assert [r['step'] for r in summaries if r['mathematical_bytes_checked']]==FULL_HASH_STEPS
        assert sum(not r['mathematical_bytes_checked'] for r in summaries)==31

def compare_reference40(x):
    """Actual output comparison with separate truthful identities; no projection."""
    reference=x.ctx.selective_reference;files={};started=time.perf_counter()
    rows=[json.loads(line) for line in (x.args.output/'working_prune.jsonl').read_text().splitlines()]
    evictions={r['step']:{k:r['eviction'][k] for k in ['evicted_entries','evicted_categories']} for r in rows if r['event']=='eviction_complete'}
    assert evictions==x.ctx.selective_reference_evictions
    for step in range(1,41):
        name='observer_'+str(step)
        values=[]
        for root,identity in [(reference,x.ctx.selective_reference_identity),(x.args.output,x.ctx.identity)]:
            side=read(root/(name+'.json'));assert side['source_identity']==identity and side['step']==step
            assert sha(root/(name+'.pt'))==side['pt_sha256']
            data=x.t.load(root/(name+'.pt'),map_location='cpu',weights_only=True)
            signature=x.signature(data);assert signature==side['signature'];values.append(signature);del data
        assert values[0]==values[1]
        for suffix in ['json','pt']:
            p=name+'.'+suffix;files[p]=dict(reference=sha(reference/p),candidate=sha(x.args.output/p))
    snapshots={}
    for name in SNAPSHOTS:
        a,sa=P3.snapshot_summary(reference/name);c,sc=P3.snapshot_summary(x.args.output/name)
        assert a['metadata']==x.ctx.selective_reference_identity and c['metadata']==x.ctx.identity
        assert a['progress']==c['progress'] and sa['metadata']==sc['metadata']
        assert set(sa['tensors'])==set(sc['tensors']) and len(sa['tensors'])==14
        for key in sa['tensors']:
            for field in ['shape','dtype','raw_sha256']:assert sa['tensors'][key][field]==sc['tensors'][key][field]
        values=[]
        for root in [reference,x.args.output]:
            data=x.t.load(root/name/'plant.pt',map_location='cpu',weights_only=True);values.append(x.signature(data));del data
        assert values[0]==values[1]
        snapshots[name]=dict(plant_all_fields=True,SR_and_host_all_components=True,progress=True,
            reference_manifest_sha256=sha(reference/name/'MANIFEST.json'),candidate_manifest_sha256=sha(x.args.output/name/'MANIFEST.json'),
            component_count=14,all_component_file_hashes_verified=True)
    value=dict(status='passed',source_identity=x.ctx.identity,reference_input_sha256=REFERENCE_INPUT_SHA,
        reference_result_sha256=REFERENCE_RESULT_SHA,observer_steps=list(range(1,41)),observer_all_tensor_bytes_equal=True,
        observer_files_sha256=files,snapshots=snapshots,eviction_sequence_equal=True,full_hash_steps=FULL_HASH_STEPS,noop_steps=31,
        reference_prune_log_sha256=sha(reference/'working_prune.jsonl'),candidate_prune_log_sha256=sha(x.args.output/'working_prune.jsonl'),elapsed_s=time.perf_counter()-started,
        scope='Actual CPU observer/plant tensor signatures and every saved component-file hash plus all14 raw payload hashes. Source/candidate identities checked separately.')
    write(x.args.output/'REFERENCE40_COMPARISON.json',value)
    return dict(sha256=sha(x.args.output/'REFERENCE40_COMPARISON.json'),elapsed_s=value['elapsed_s'],observer_steps=40,snapshots=5)

def reference_pair_gate(source,identity,result):
    receipt=read(source/'REFERENCE40_COMPARISON.json')
    assert sha(source/'REFERENCE40_COMPARISON.json')==result['reference40_comparison']['sha256']
    assert receipt['source_identity']==identity and receipt['status']=='passed'
    assert receipt['reference_input_sha256']==REFERENCE_INPUT_SHA and receipt['reference_result_sha256']==REFERENCE_RESULT_SHA
    assert receipt['observer_steps']==list(range(1,41)) and receipt['observer_all_tensor_bytes_equal'] is True
    assert receipt['eviction_sequence_equal'] is True and receipt['full_hash_steps']==FULL_HASH_STEPS and receipt['noop_steps']==31
    assert sha(source/'working_prune.jsonl')==receipt['candidate_prune_log_sha256']==result['working_prune_log_sha256']
    reference=Path(identity['selective_prune_qualification']['reference_path'])
    assert sha(reference/'INPUT.json')==REFERENCE_INPUT_SHA and sha(reference/'RESULT.json')==REFERENCE_RESULT_SHA
    assert sha(reference/'working_prune.jsonl')==receipt['reference_prune_log_sha256']
    assert set(receipt['observer_files_sha256'])=={f'observer_{step}.{ext}' for step in range(1,41) for ext in ['json','pt']}
    for name,digests in receipt['observer_files_sha256'].items():
        assert sha(source/name)==digests['candidate'] and sha(reference/name)==digests['reference']
    assert set(receipt['snapshots'])==set(SNAPSHOTS)
    for name,row in receipt['snapshots'].items():
        assert all(row[k] is True for k in ['plant_all_fields','SR_and_host_all_components','progress','all_component_file_hashes_verified'])
        assert row['component_count']==14
        assert sha(source/name/'MANIFEST.json')==row['candidate_manifest_sha256']==result['snapshots'][name]
        assert sha(reference/name/'MANIFEST.json')==row['reference_manifest_sha256']

def own40_gate(source,identity):
    source=Path(source);inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==identity
    assert identity['script_sha256']==r['script_sha256']==sha(__file__) and identity['policy']==POLICY
    assert identity['base_eager_entry_sha256']==SHORT_SHA and identity['selective_prune_helper_sha256']==HELPER_SHA
    assert identity['selective_prune_qualification']['cpu_result_sha256']==CPU_RESULT_SHA
    reference_pair_gate(source,identity,r)
    assert identity['base_observer64_entry_sha256']==OBSERVER_ENTRY_SHA
    assert identity['working_eager_adapter_sha256']==EAGER_ADAPTER_SHA and identity['working_eager_policy']==EAGER_POLICY
    assert identity['working_eager_qualification']['sha256']==EAGER_RESULT_SHA and r['working_eager_restored'] is True
    eager_counters(r['working_eager_counters'],40);eager_log_gate(source,list(range(1,41)),r)
    assert identity['base_working_prune_entry_sha256']==h.BASE_SHA and identity['base_working_prune_identity']['script_sha256']==h.BASE_SHA
    assert identity['horner_cleanup_adapter_sha256']==h.ADAPTER_SHA and identity['horner_cleanup_policy']==h.CLEANUP_POLICY
    assert identity['horner_cleanup_qualification']['sha256']==h.CHECK_RESULT_SHA
    assert identity['horner_original_source_sha256']==h.ORIGINAL_SOURCE_SHA and identity['horner_inverse_AST_equal'] is True
    assert identity['horner_original_function_sha256']==h.ORIGINAL_FUNCTION_SHA and identity['horner_candidate_function_sha256']==h.CANDIDATE_FUNCTION_SHA
    assert r['horner_cleanup_restored'] is True
    assert identity['base_horner_cleanup_entry_sha256']==BASE_SHA
    assert identity['observer_endpoint_helper_sha256']==ADAPTER_SHA and identity['observer_endpoint_policy']==ENDPOINT_POLICY
    assert identity['observer_endpoint_qualification']['sha256']==CHECK_RESULT_SHA
    assert r['observer_endpoint64_calls']==40 and r['observer_driver_function_unchanged'] is True
    assert identity['base_valid128_entry_sha256']==b.BASE_SHA and identity['base_valid128_identity']['script_sha256']==b.BASE_SHA
    assert identity['working_graph_adapter_sha256']==b.ADAPTER_SHA and identity['working_graph_policy']==b.PRUNE_POLICY
    gate=identity['working_graph_qualification']
    assert gate['checker_sha256']==b.CHECK_SHA and gate['reference_result_sha256']==b.REFERENCE_RESULT_SHA and gate['candidate_result_sha256']==b.CHECK_RESULT_SHA
    assert identity['ordinary_valid_adapter_sha256']==w.ADAPTER_SHA and identity['valid128_qualification']['sha256']==w.CHECK_RESULT_SHA
    assert identity['ordinary_valid_chunk_size']==128 and identity['ordinary_valid_full_rows']==1024
    assert identity['cache_release_adapter_sha256']==w.w.ADAPTER_SHA and identity['cache_release_qualification']['sha256']==w.w.CHECK_RESULT_SHA
    assert identity['weighted128_qualification']['sha256']==w.w.w.CHECK_RESULT_SHA
    assert (identity['batch_size'],identity['working_total_degree'],identity['point_code_order'],identity['validation_order'])==(1024,3,2,4)
    assert identity['early_weighted_chunk_size']==identity['early_weighted_graph_padding']==128
    assert r['input_sha256']==sha(source/'INPUT.json')
    assert inp['mode']==r['mode']=='bounded40' and inp['original_target_steps']==r['original_target_steps']==1000
    assert inp['budget_steps']==r['budget_steps']==40 and r['status']=='bounded_prefix_completed'
    assert r['completed_step']==40 and r['accepted_lane_steps']==40960 and r['all_steps_all_lanes_accepted'] is True
    assert r['controller_calls']==2 and r['controller_refresh_steps']==[0,20]
    assert r['pruned_checkpoint20_support_cache_verified'] is True
    assert len(r['hybrid_validation_owners'])==1
    owner=r['hybrid_validation_owners'][0]
    assert (owner['n'],owner['working'],owner['validation'])==(16,3,4) and owner['prefix_checked'] is True
    assert owner['full_degree_six_prefix']==dict(full=100947,spatial=74613)
    counters=r['weighted128_counters']
    assert counters['chunk_size']==128 and 0<counters['refine_calls']<=40 and counters['graph_map_calls']>0
    w.counters_gate(r['valid128_counters'],40);b.counters_gate(r['working_graph_counters'],40)
    for field in ['working_graph_hook_restored','valid_adapter_restored','weighted_adapter_restored','cache_release_adapter_restored']:assert r[field] is True
    log_gate(source,list(range(1,41)),r)
    assert r['postwarm_releases']
    for row in r['postwarm_releases']:
        assert row['allocated_after']<=row['allocated_before'] and row['reserved_after']<=row['reserved_before']
    assert set(r['snapshots'])=={'committed_20','after_endpoint_20','after_controller_20','committed_21','committed_40'}
    for name,digest in r['snapshots'].items():assert sha(source/name/'MANIFEST.json')==digest
    return dict(source_input_sha256=sha(source/'INPUT.json'),source_result_sha256=sha(source/'RESULT.json'))

def cold(runtime,source):
    # w is60df: 60df -> 6e74 -> 2c6e ->22f. No old identity projection.
    base=w.w.w.w;raw=base.own40_gate;base.own40_gate=own40_gate
    try:return base.cold(runtime,source)
    finally:base.own40_gate=raw

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception',mode=args.mode)
    try:
        ctx=initialize(args);x=Runtime(ctx,args)
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode=args.mode,original_target_steps=1000,budget_steps=40,
            driver_argv=driver_argv(ctx),source40=str(args.source40) if args.source40 else None))
        if args.mode=='cold20_to21':
            result=cold(x,args.source40);w.counters_gate(ctx.valid128.counters,1);b.counters_gate(ctx.working_prune.counters,1)
            assert x.observer_endpoint64_calls==1;eager_counters(ctx.eager_segments.counters,1)
        else:
            argv=sys.argv;sys.argv=driver_argv(ctx)
            try:
                returned=ctx.d.main();result=dict(status='original_driver_stopped',mode=args.mode,driver_return=returned)
            except BudgetStop:
                assert x.completed==40 and x.ledger.length==x.sr.qlen==x.sr.jlen==40
                result=dict(status='bounded_prefix_completed' if all(all(r['accepted']) for r in x.rows) else 'bounded_prefix_with_failed_lanes',mode=args.mode)
                w.counters_gate(ctx.valid128.counters,40);b.counters_gate(ctx.working_prune.counters,40)
                assert x.observer_endpoint64_calls==40;eager_counters(ctx.eager_segments.counters,40)
                if result['status']=='bounded_prefix_completed':result['reference40_comparison']=compare_reference40(x)
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
        if 'ctx' in locals():
            assert ctx.d.rows_range_over_time_sparse is ctx.observer_original_range
            result['observer_driver_function_unchanged']=True
            if 'x' in locals():result['observer_endpoint64_calls']=x.observer_endpoint64_calls
            result.update(max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved(),
                valid128_counters=dict(ctx.valid128.counters),weighted128_counters=dict(ctx.weighted128.counters),
                postwarm_releases=list(ctx.postwarm.releases),postwarm_release_s=sum(row['elapsed_s'] for row in ctx.postwarm.releases),
                working_graph_counters=dict(ctx.working_prune.counters),working_prune_checks=list(ctx.working_prune_rows),
                working_prune_instrumentation_s=sum(row['elapsed_s'] for row in ctx.working_prune_rows),
                timing_scope='Advance includes eager dispatch/logging; excludes observer and selective postprune audit. Process includes reference40 comparison and checkpoint IO. No-op steps are storage/version guards, not full payload checks.')
            result['working_eager_counters']=dict(ctx.eager_segments.counters)
            ctx.eager_segments.restore();result['working_eager_restored']=True
            ctx.working_eager_log.close();result['working_eager_log_sha256']=sha(args.output/'working_eager.jsonl')
            ctx.horner_cleanup.restore();result['horner_cleanup_restored']=True
            ctx.working_prune.restore();result['working_graph_hook_restored']=True
            ctx.working_prune_log.close();result['working_prune_log_sha256']=sha(args.output/'working_prune.jsonl')
            ctx.valid128.restore();result['valid_adapter_restored']=True
            ctx.postwarm.restore();ctx.postwarm_log.close();result['cache_release_adapter_restored']=True
            ctx.weighted128.restore();result['weighted_adapter_restored']=True
            if result['status'] in ['bounded_prefix_completed','passed']:
                log_gate(args.output,[21] if args.mode=='cold20_to21' else list(range(1,41)),result)
                eager_log_gate(args.output,[21] if args.mode=='cold20_to21' else list(range(1,41)),result)
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['bounded40','cold20_to21'],required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker',
        'working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','observer-check','observer-checker','working-eager-check','working-eager-checker','selective-audit-check','selective-audit-checker','reference40','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40',type=Path);a=p.parse_args();assert (a.source40 is not None)==(a.mode=='cold20_to21');main(a)
