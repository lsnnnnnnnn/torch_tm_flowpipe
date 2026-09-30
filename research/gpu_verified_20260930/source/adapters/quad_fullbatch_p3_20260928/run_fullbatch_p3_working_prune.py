"""Full1024 P3 own40/cold with selective working-graph retention.

Frozen60df initialization and all numerical Runtime/driver calls are reused.
Every original prune is followed by the qualified cache policy and full active
state/SR/host-history byte and storage checks. Their cost is included in process
time, recorded separately, and is not a solver-speed measurement.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
BASE_SHA='60df1ef31d6491e4d436a2724b3f9ef529c1432baeae0e4de071da2b5a4b8b31'
ADAPTER_SHA='3a174e02724ee3166eff9f1b330c137e0c1a4b51f64a6162ea2a6ed96814e21f'
CHECK_SHA='54d51b192ecc312d184cb78aaef7aa2afc2098e20688cb556bd43c869ed7610c'
REFERENCE_RESULT_SHA='a27104d0d57b7080ed536bb66fd2b8e32f0482b913450cb4827b61b37de88c33'
CHECK_RESULT_SHA='11d82a0138bd91c6a9edc3cc166f4f04f6d4cc7824ad39c1065af15eb0ce5a15'
POLICY='full1024_P3_ordinaryvalid128_selective_working_graph_prune_full_byte_checks'
PRUNE_POLICY='after_complete_prune_evict_working_graph_keys_unused_in_current_advance'

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

# Reuse the exact signatures and reference-file checks exercised by the paired
# probe. Its import only loads modules; its diagnostic Runtime/main are not run.
probe=load('fullbatch_working_prune_frozen_probe',HERE/'probe_working_graph_prune.py',CHECK_SHA)
w=probe.w
assert sha(w.__file__)==BASE_SHA
BudgetStop=w.BudgetStop
driver_argv=w.driver_argv

def counters_gate(counters,steps):
    assert counters['begun_steps']==counters['finished_steps']==steps
    assert counters['working_run_calls']>=steps and counters['evicted_entries']>=0

def qualification(args,identity=None):
    reference=Path(args.working_prune_reference);candidate=Path(args.working_prune_check)
    assert sha(reference/'RESULT.json')==REFERENCE_RESULT_SHA
    assert sha(candidate/'RESULT.json')==CHECK_RESULT_SHA
    result=read(candidate/'RESULT.json');inp=read(candidate/'INPUT.json')
    source_identity=result['source_identity']
    base=source_identity['algorithm_identity']
    if identity is not None:assert base==identity
    assert base['script_sha256']==BASE_SHA and base['policy']==w.POLICY
    reference_files=probe.reference_gate(reference,base)
    assert inp['reference_files_sha256']==reference_files
    assert result['input_sha256']==sha(candidate/'INPUT.json')
    assert inp['source_identity']==source_identity
    assert source_identity['script_sha256']==result['script_sha256']==CHECK_SHA
    assert source_identity['policy']==probe.POLICY and source_identity['probe_mode']=='candidate3'
    assert source_identity['working_graph_adapter_sha256']==ADAPTER_SHA
    assert result['source_entry_sha256']==BASE_SHA and result['mode']==inp['mode']=='candidate3'
    assert result['status']=='diagnostic_candidate_three_steps' and result['completed_advance_count']==3
    assert result['accepted_lanes_by_step']==[1024]*3 and result['reference_steps_bytes_equal']==[1,2]
    assert result['candidate_step3_has_reference'] is False and result['diagnostic_only'] is True
    assert result['no_timing_or_execution_qualification'] is True
    for field in ['working_graph_hook_restored','valid_adapter_restored','cache_release_adapter_restored','weighted_adapter_restored']:
        assert result[field] is True
    counters_gate(result['working_graph_counters'],3)
    assert [row['evicted_entries'] for row in result['evictions']]==[0,5,6]
    assert result['working_graph_counters']['evicted_entries']==11
    assert set(result['digest_files_sha256'])=={f'DIGEST_{step}.json' for step in [1,2,3]}
    files={'INPUT.json':sha(candidate/'INPUT.json'),'RESULT.json':CHECK_RESULT_SHA}
    for step in [1,2,3]:
        name=f'DIGEST_{step}.json';digest=result['digest_files_sha256'][name]
        assert sha(candidate/name)==digest;files[name]=digest
        row=read(candidate/name)
        assert row['source_identity']==source_identity and row['step']==step
        assert row['before_after_mathematical_bytes_equal'] is True and row['capacity_and_storage_unchanged'] is True
        assert row['eviction']==result['evictions'][step-1]
        storage=row['storage_before']
        assert storage['sr_max_size']==storage['host_max_size']==storage['sr_capacity']==1000
        assert storage['qlen']==storage['jlen']==step
        if step<=2:
            assert row['reference_mathematical_bytes_equal'] is True and row['reference_observer_all_tensor_bytes_equal'] is True
            assert row['mathematical_signature']==read(reference/name)['mathematical_signature']
    return dict(checker_sha256=CHECK_SHA,adapter_sha256=ADAPTER_SHA,
        reference_path=str(reference),reference_result_sha256=REFERENCE_RESULT_SHA,reference_files_sha256=reference_files,
        candidate_path=str(candidate),candidate_result_sha256=CHECK_RESULT_SHA,candidate_files_sha256=files,
        paired_steps=[1,2],candidate_unpaired_step=3,trajectory_qualification=False)

def record(ctx,row):
    ctx.working_prune_log.write(json.dumps(row,allow_nan=False)+'\n');ctx.working_prune_log.flush()

def initialize(args):
    assert sha(HERE/'working_graph_prune.py')==ADAPTER_SHA
    qualification(args)
    ctx=w.initialize(args)
    receipt=qualification(args,ctx.identity)
    assert not ctx.hybrid.receipts,'Install before any engine/graph exists'
    adapter=load('fullbatch_working_graph_prune_adapter',HERE/'working_graph_prune.py',ADAPTER_SHA)
    from flowstar_gpu import graphing
    ctx.working_prune_log=(args.output/'working_prune.jsonl').open('w')
    ctx.working_prune_rows=[];ctx.working_prune_current_step=None
    def on_eviction(eviction):
        record(ctx,dict(event='eviction_complete',step=ctx.working_prune_current_step,eviction=eviction))
    ctx.working_prune=adapter.install(ctx.torch,graphing,record=on_eviction)
    assert ctx.working_prune.policy==PRUNE_POLICY
    base=dict(ctx.identity)
    ctx.identity=dict(base,script_sha256=sha(__file__),policy=POLICY,base_valid128_entry_sha256=BASE_SHA,
        base_valid128_identity=base,working_graph_adapter_sha256=ADAPTER_SHA,working_graph_policy=PRUNE_POLICY,
        working_graph_qualification=receipt,working_graph_instrumentation='Every postprune: full active state/SR/host history hashes plus storage/version/capacity invariance. Included in process time; solver advance timing excludes these checks.',
        fullbatch_qualification=False,end_to_end_strict_certificate=False,
        limitation='Only unused working-engine graph entries are removed after each original prune. Validation/weighted caches, all numerical calls, original1024 NN rows, h/cap/full SR remain. Paired gate covers steps1/2 and unpaired candidate3; this identity requires its own40/cold. Conditional CROWN remains separate.')
    return ctx

class Runtime(w.Runtime):
    def advance_sparse(self,st,code,eng,sched,settings,cap,sr):
        self.ctx.working_prune.begin_step(eng)
        return super().advance_sparse(st,code,eng,sched,settings,cap,sr)
    def prune_state(self,*args,**kwargs):
        value=super().prune_state(*args,**kwargs);start=time.perf_counter()
        before=probe.mathematical_signature(self);storage=probe.storage_signature(self)
        self.ctx.working_prune_current_step=self.completed
        eviction=self.ctx.working_prune.finish_step(self.eng)
        after=probe.mathematical_signature(self);storage_after=probe.storage_signature(self)
        assert after==before,'Cache pruning changed active mathematical state/SR/history'
        assert storage_after==storage,'Cache pruning changed storage/version/capacity'
        row=dict(event='invariance_checked',step=self.completed,phase='after_original_prune_before_poststep_safety_reset',
            before_after_mathematical_bytes_equal=True,capacity_and_storage_unchanged=True,
            mathematical_signature=before,storage_before=storage,eviction=eviction,elapsed_s=time.perf_counter()-start)
        record(self.ctx,row)
        self.ctx.working_prune_rows.append(dict(step=self.completed,elapsed_s=row['elapsed_s'],
            before_after_mathematical_bytes_equal=True,capacity_and_storage_unchanged=True))
        return value

def log_gate(source,steps,result):
    path=Path(source)/'working_prune.jsonl';assert sha(path)==result['working_prune_log_sha256']
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==2*len(steps)
    for index,step in enumerate(steps):
        event,checked=rows[2*index:2*index+2]
        assert event['event']=='eviction_complete' and checked['event']=='invariance_checked'
        assert event['step']==checked['step']==step and event['eviction']==checked['eviction']
        assert checked['before_after_mathematical_bytes_equal'] is True and checked['capacity_and_storage_unchanged'] is True
        storage=checked['storage_before']
        assert storage['sr_max_size']==storage['host_max_size']==storage['sr_capacity']==1000
        assert storage['qlen']==storage['jlen']==step
        assert checked['eviction']['validation_and_weighted_untouched'] is True
    assert [row['step'] for row in result['working_prune_checks']]==steps
    assert all(row['before_after_mathematical_bytes_equal'] and row['capacity_and_storage_unchanged'] for row in result['working_prune_checks'])

def own40_gate(source,identity):
    source=Path(source);inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==identity
    assert identity['script_sha256']==r['script_sha256']==sha(__file__) and identity['policy']==POLICY
    assert identity['base_valid128_entry_sha256']==BASE_SHA and identity['base_valid128_identity']['script_sha256']==BASE_SHA
    assert identity['working_graph_adapter_sha256']==ADAPTER_SHA and identity['working_graph_policy']==PRUNE_POLICY
    gate=identity['working_graph_qualification']
    assert gate['checker_sha256']==CHECK_SHA and gate['reference_result_sha256']==REFERENCE_RESULT_SHA and gate['candidate_result_sha256']==CHECK_RESULT_SHA
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
    w.counters_gate(r['valid128_counters'],40);counters_gate(r['working_graph_counters'],40)
    for field in ['working_graph_hook_restored','valid_adapter_restored','weighted_adapter_restored','cache_release_adapter_restored']:assert r[field] is True
    log_gate(source,list(range(1,41)),r)
    assert r['postwarm_releases']
    for row in r['postwarm_releases']:
        assert row['allocated_after']<=row['allocated_before'] and row['reserved_after']<=row['reserved_before']
    assert set(r['snapshots'])=={'committed_20','after_endpoint_20','after_controller_20','committed_21','committed_40'}
    for name,digest in r['snapshots'].items():assert sha(source/name/'MANIFEST.json')==digest
    return dict(source_input_sha256=sha(source/'INPUT.json'),source_result_sha256=sha(source/'RESULT.json'))

def cold(runtime,source):
    # 60df -> 6e74 -> 2c6e -> 22f. Use the unchanged original cold body.
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
            result=cold(x,args.source40);w.counters_gate(ctx.valid128.counters,1);counters_gate(ctx.working_prune.counters,1)
        else:
            argv=sys.argv;sys.argv=driver_argv(ctx)
            try:
                returned=ctx.d.main();result=dict(status='original_driver_stopped',mode=args.mode,driver_return=returned)
            except BudgetStop:
                assert x.completed==40 and x.ledger.length==x.sr.qlen==x.sr.jlen==40
                result=dict(status='bounded_prefix_completed' if all(all(r['accepted']) for r in x.rows) else 'bounded_prefix_with_failed_lanes',mode=args.mode)
                w.counters_gate(ctx.valid128.counters,40);counters_gate(ctx.working_prune.counters,40)
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
            result.update(max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved(),
                valid128_counters=dict(ctx.valid128.counters),weighted128_counters=dict(ctx.weighted128.counters),
                postwarm_releases=list(ctx.postwarm.releases),postwarm_release_s=sum(row['elapsed_s'] for row in ctx.postwarm.releases),
                working_graph_counters=dict(ctx.working_prune.counters),working_prune_checks=list(ctx.working_prune_rows),
                working_prune_instrumentation_s=sum(row['elapsed_s'] for row in ctx.working_prune_rows),
                timing_scope='Process includes full postprune hash/storage checks and cache-policy IO; advance timings exclude this instrumentation. Not a paired speed result.')
            ctx.working_prune.restore();result['working_graph_hook_restored']=True
            ctx.working_prune_log.close();result['working_prune_log_sha256']=sha(args.output/'working_prune.jsonl')
            ctx.valid128.restore();result['valid_adapter_restored']=True
            ctx.postwarm.restore();ctx.postwarm_log.close();result['cache_release_adapter_restored']=True
            ctx.weighted128.restore();result['weighted_adapter_restored']=True
            if result['status'] in ['bounded_prefix_completed','passed']:
                log_gate(args.output,[21] if args.mode=='cold20_to21' else list(range(1,41)),result)
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['bounded40','cold20_to21'],required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker',
        'working-prune-reference','working-prune-check','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40',type=Path);a=p.parse_args();assert (a.source40 is not None)==(a.mode=='cold20_to21');main(a)
