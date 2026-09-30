"""Own full1024 P3 40/cold with five working graph regions dispatched eagerly.

Baseline171dd keeps128 validation, observer64, full SR and full postprune
checks. The nine-step diagnostic gate is necessary but never replaces this
entry's own40/cold. No original source or graph pool policy is modified.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
BASE_SHA='b89c5710f5faa9dc1b363eba96d27cbbc7bc2a7006df9aa2d305f39cf202b2e7'
ADAPTER_SHA='d8d87a51a7db600685a79ff19193a56986dab6e84b4825a30570caaf157c8872'
CHECK_SHA='42d1eb24c1a59f03ad208eb44a5615b87e2ccd456b8af1c9f59c0a823740de61'
CHECK_RESULT_SHA='42f2beba0ac922987cf16f995a9165940ae323c96a4d1b7c18c88df3df6c75ab'
POLICY='full1024_P3_original128_observer64_five_working_regions_eager'
OBSERVER_ENTRY_SHA='171dd167496895b15fd1373de16986d30118b3338c4140a4f5c91b98eee353c9'
EAGER_ADAPTER_SHA='c4fb7b098a10ab98d9e9250fc88460ade130225ef710ea5734522df03ca0453b'
EAGER_CHECK_SHA='01e454d3e88f2f4dfc09e27be513c153f947eb5a690f27919bb3d8eecce9da82'
EAGER_RESULT_SHA='ac8a1ba7f792ea05349e53ad93762f940042c3095b5415c50c789e25de446971'
EAGER_POLICY='active_fresh_working_engine_five_regions_original_fn_eager_other_graphs_unchanged'
REFERENCE_INPUT_SHA='52201b09ddb6060a1ba797b6e15f69ab455941d7d1179e849d68739daced8938'
REFERENCE_PRUNE_SHA='8986261abbad0fa27f067f54e2b7a381f8b1ccb4fcbbc82b110e838f5e926d04'
REFERENCE_WATCH_SHA='9bb887788c0064d4626ffd13bfd89f0cd0649b33857c429301ee2e03c303c4f3'
ENDPOINT_POLICY='observer_endpoint_only_original_dense3_range_full1024_lane64'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
o=load('working_eager_frozen_observer_entry',HERE/'run_fullbatch_p3_observer64.py',OBSERVER_ENTRY_SHA)
h=o.h;b=o.b;w=o.w
BudgetStop=o.BudgetStop
driver_argv=o.driver_argv

FAMILIES=('endpoint','sr_linear_prepare','combine_linear','normalize','glue:emit')
def eager_counters(counters,steps):
    assert counters['begun_steps']==counters['finished_steps']==steps
    assert set(counters['by_family'])==set(FAMILIES)
    assert all(counters['by_family'][k]==steps for k in FAMILIES if k!='combine_linear')
    assert 0<=counters['by_family']['combine_linear']<=steps
    assert counters['eager_calls']==sum(counters['by_family'].values())==counters['input_version_checks']
    assert counters['forwarded_calls']>0 and counters['nested_forwarded_calls']>=0

def qualification(args,identity=None):
    assert len(EAGER_RESULT_SHA)==64,'Actual nine-step CUDA byte gate is not yet frozen'
    assert sha(args.working_eager_checker)==EAGER_CHECK_SHA
    root=args.working_eager_check;path=root/'RESULT.json';assert sha(path)==EAGER_RESULT_SHA
    r=read(path);inp=read(root/'INPUT.json');sid=r['source_identity']
    assert r['status']=='diagnostic_nine_steps_bytes_equal' and r['script_sha256']==EAGER_CHECK_SHA
    assert r['source_entry_sha256']==OBSERVER_ENTRY_SHA and r['adapter_sha256']==EAGER_ADAPTER_SHA
    assert r['input_sha256']==sha(root/'INPUT.json') and inp['source_identity']==sid
    assert sid['script_sha256']==EAGER_CHECK_SHA and sid['policy']=='diagnostic_full1024_P3_five_working_segments_eager_original128_observer64'
    assert sid['working_eager_adapter_sha256']==EAGER_ADAPTER_SHA and sid['working_eager_policy']==EAGER_POLICY
    assert sid['diagnostic_only'] is True and r['diagnostic_only'] is True and r['no_timing_or_execution_qualification'] is True
    assert sid['algorithm_identity']['script_sha256']==OBSERVER_ENTRY_SHA
    if identity is not None:assert sid['algorithm_identity']==identity
    assert inp['reference_failed'] is True and inp['reference_watch_sha256']==REFERENCE_WATCH_SHA
    refs=inp['reference_files_sha256'];assert refs['INPUT.json']==REFERENCE_INPUT_SHA and refs['working_prune.jsonl']==REFERENCE_PRUNE_SHA
    assert r['reference_steps']==list(range(1,10)) and r['accepted_lanes_by_step']==[1024]*9
    assert r['completed_advance_count']==r['observer_endpoint64_calls']==9 and r['phase']=='committed_before_endpoint_handoff'
    for key in ['working_eager_restored','observer_driver_function_unchanged','horner_cleanup_restored',
        'working_graph_hook_restored','valid_adapter_restored','cache_release_adapter_restored','weighted_adapter_restored']:assert r[key] is True
    eager_counters(r['working_eager_counters'],9);w.counters_gate(r['valid128_counters'],9);b.counters_gate(r['working_graph_counters'],9)
    weighted=r['weighted128_counters'];assert weighted['chunk_size']==128 and 0<weighted['refine_calls']<=9 and weighted['graph_map_calls']>0
    assert [v['step'] for v in r['working_graph_checks']]==list(range(1,10))
    assert all(v['before_after_mathematical_bytes_equal'] and v['capacity_and_storage_unchanged'] for v in r['working_graph_checks'])
    prune_bytes=(root/'working_prune.jsonl').read_bytes().splitlines(keepends=True);assert len(prune_bytes)==18
    files={'RESULT.json':EAGER_RESULT_SHA,'INPUT.json':sha(root/'INPUT.json'),'working_prune.jsonl':sha(root/'working_prune.jsonl')}
    for step in range(1,10):
        pair=read(root/f'PAIRED_{step}.json');a=read(root/f'observer_{step}.json')
        ev=json.loads(prune_bytes[2*step-2]);row=json.loads(prune_bytes[2*step-1])
        assert ev['event']=='eviction_complete' and row['event']=='invariance_checked'
        assert ev['step']==row['step']==pair['step']==step and ev['eviction']==row['eviction']
        assert row['before_after_mathematical_bytes_equal'] is True and row['capacity_and_storage_unchanged'] is True
        assert pair['reference_input_sha256']==REFERENCE_INPUT_SHA
        assert pair['full_mathematical_state_SR_host_bytes_equal'] is True and pair['observer_file_bytes_equal'] is True
        assert pair['current_prune_record_sha256']==hashlib.sha256(b''.join(prune_bytes[:2*step])).hexdigest()
        assert pair['dispatch']['step']==step and pair['dispatch']['input_versions_unchanged'] is True
        assert a['source_identity']==sid and a['step']==step
        assert sha(root/f'observer_{step}.pt')==a['pt_sha256']==refs[f'observer_{step}.pt']
        for name in [f'PAIRED_{step}.json',f'observer_{step}.json',f'observer_{step}.pt']:files[name]=sha(root/name)
    assert sha(root/'working_eager.jsonl')==r['working_eager_log_sha256']
    files['working_eager.jsonl']=r['working_eager_log_sha256']
    return dict(path=str(path),sha256=EAGER_RESULT_SHA,checker_sha256=EAGER_CHECK_SHA,adapter_sha256=EAGER_ADAPTER_SHA,
        source_files_sha256=files,reference_watch_sha256=REFERENCE_WATCH_SHA,reference_failed=True,
        paired_steps=list(range(1,10)),trajectory_qualification=False)

def log_gate(root,steps,result):
    path=Path(root)/'working_eager.jsonl';assert sha(path)==result['working_eager_log_sha256']
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    finished=[v for v in rows if v['event']=='working_eager_step_finished']
    assert [v['actual_step'] for v in finished]==steps
    assert [v['step'] for v in finished]==list(range(1,len(steps)+1))
    assert len(rows)==result['working_eager_counters']['eager_calls']+len(steps)
    for i,step in enumerate(steps,1):
        calls=[v for v in rows if v['event']=='working_eager_returned' and v['actual_step']==step]
        end=finished[i-1];counts={k:sum(v['family']==k for v in calls) for k in FAMILIES}
        assert end['by_family']==counts and end['eager_calls']==len(calls)
        assert all(v['step']==i and v['input_versions_unchanged'] is True for v in calls)
        assert end['input_versions_unchanged'] is True

def initialize(args):
    qualification(args)
    assert sha(HERE/'working_eager_segments.py')==EAGER_ADAPTER_SHA
    ctx=o.initialize(args);receipt=qualification(args,ctx.identity);assert not ctx.hybrid.receipts
    adapter=load('fullbatch_working_eager_adapter',HERE/'working_eager_segments.py',EAGER_ADAPTER_SHA)
    from flowstar_gpu import graphing
    ctx.working_eager_log=(args.output/'working_eager.jsonl').open('w');ctx.working_eager_actual_step=None
    def record(row):
        ctx.working_eager_log.write(json.dumps(dict(row,actual_step=ctx.working_eager_actual_step),allow_nan=False)+'\n')
        ctx.working_eager_log.flush()
    ctx.eager_segments=adapter.install(ctx.torch,graphing,record=record)
    assert ctx.eager_segments.policy==EAGER_POLICY
    ctx.identity=dict(ctx.identity,script_sha256=sha(__file__),policy=POLICY,base_observer64_entry_sha256=OBSERVER_ENTRY_SHA,
        working_eager_adapter_sha256=EAGER_ADAPTER_SHA,working_eager_policy=EAGER_POLICY,working_eager_qualification=receipt,
        working_eager_source_sha256=dict(graphing=adapter.GRAPHING_SHA,working_prune=adapter.WORKING_PRUNE_SHA,postwarm=adapter.POSTWARM_SHA),
        fullbatch_qualification=False,end_to_end_strict_certificate=False,
        limitation='Only five active working-engine regions bypass GraphCache.run and execute original fn on original inputs. Validation128, observer64, private pools for other regions, original GLUE=graph branches and full postprune bytes/storage checks remain. Nine paired steps against a resource-failed reference do not replace this identity own40/cold. Conditional CROWN remains separate.')
    return ctx

class Runtime(o.Runtime):
    def advance_sparse(self,st,code,eng,sched,settings,cap,sr):
        self.ctx.working_eager_actual_step=self.completed+1
        self.ctx.eager_segments.begin_step(eng)
        return super().advance_sparse(st,code,eng,sched,settings,cap,sr)
    def prune_state(self,*args,**kwargs):
        value=super().prune_state(*args,**kwargs)
        self.ctx.eager_segments.finish_step(self.eng)
        return value

def own40_gate(source,identity):
    source=Path(source);inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==identity
    assert identity['script_sha256']==r['script_sha256']==sha(__file__) and identity['policy']==POLICY
    assert identity['base_observer64_entry_sha256']==OBSERVER_ENTRY_SHA
    assert identity['working_eager_adapter_sha256']==EAGER_ADAPTER_SHA and identity['working_eager_policy']==EAGER_POLICY
    assert identity['working_eager_qualification']['sha256']==EAGER_RESULT_SHA and r['working_eager_restored'] is True
    eager_counters(r['working_eager_counters'],40);log_gate(source,list(range(1,41)),r)
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
    b.log_gate(source,list(range(1,41)),r)
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
                timing_scope='Process includes full postprune hash/storage checks and cache-policy IO; advance timings exclude this instrumentation. Not a paired speed result.')
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
                b.log_gate(args.output,[21] if args.mode=='cold20_to21' else list(range(1,41)),result)
                log_gate(args.output,[21] if args.mode=='cold20_to21' else list(range(1,41)),result)
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['bounded40','cold20_to21'],required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker',
        'working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','observer-check','observer-checker','working-eager-check','working-eager-checker','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40',type=Path);a=p.parse_args();assert (a.source40 is not None)==(a.mode=='cold20_to21');main(a)
