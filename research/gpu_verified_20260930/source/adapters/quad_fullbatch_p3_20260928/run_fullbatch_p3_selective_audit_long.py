"""Full1024 P3 selective-audit long, admitted only by its own40 and cold.

The original50-period numerical loop is unchanged. Only nonempty working
graph eviction hashes the full live state/SR/host payload before and after;
no-op steps keep storage/version/capacity guards. CROWN remains conditional.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
SHORT_SHA='660c75cf22346e71d5044a60b5bbd62320ed38e542727cbcb6805fad909095e0'
DONOR_LONG_SHA='3efc0a39166856042ad22a4d1085ee780f11ba06c84889cf0a80ac4fd9dc7366'
P2_LONG_SHA='7b465c72046b4c23497ace95f4b3d74cba8fd556cbb48a5454ae0af1b3772c98'
POLICY='full1024_P3_original128_observer64_eager_1000_selective_postprune_audit'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def load(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
assert sha(HERE/'run_fullbatch_p3_working_eager_long.py')==DONOR_LONG_SHA
w=load('fullbatch_p3_selective_long_ownshort',HERE/'run_fullbatch_p3_selective_audit.py',SHORT_SHA)
p2=load('fullbatch_p3_selective_long_original_limits',HERE.parent/'quad_fullbatch_sr_20260928/run_fullbatch_long.py',P2_LONG_SHA)
read,write=w.read,w.write
b=w.b

def admission(args,short_identity):
    assert len(args.source40_result_sha256)==len(args.cold_result_sha256)==64
    assert sha(args.source40/'RESULT.json')==args.source40_result_sha256
    assert sha(args.cold/'RESULT.json')==args.cold_result_sha256
    links=w.own40_gate(args.source40,short_identity)
    assert read(args.source40/'RESULT.json')['script_sha256']==SHORT_SHA
    inp=read(args.cold/'INPUT.json');r=read(args.cold/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==short_identity
    assert r['script_sha256']==SHORT_SHA and r['input_sha256']==sha(args.cold/'INPUT.json')
    assert inp['mode']==r['mode']=='cold20_to21' and r['status']=='passed'
    assert inp['original_target_steps']==r['original_target_steps']==1000
    assert inp['budget_steps']==r['budget_steps']==40
    assert Path(inp['source40']).resolve()==args.source40.resolve()
    for key,digest in links.items():assert r[key]==digest
    assert r['completed_step']==21 and r['accepted_lane_steps']==1024 and r['all_steps_all_lanes_accepted'] is True
    assert r['controller_calls']==1 and r['controller_refresh_steps']==[20]
    assert r['pruned_checkpoint20_support_cache_verified'] is True
    assert len(r['hybrid_validation_owners'])==1
    v=r['hybrid_validation_owners'][0]
    assert (v['n'],v['working'],v['validation'])==(16,3,4) and v['prefix_checked'] is True
    assert v['full_degree_six_prefix']==dict(full=100947,spatial=74613)
    states=['committed_20','after_endpoint_20','after_controller_20','committed_21']
    assert set(r['snapshots'])==set(states)
    assert set(r['checks'])==set(states+['endpoint_20','controller_20','transfer_20','observer_21'])
    for name in states:
        assert r['checks'][name]==dict(plant_all_fields=True,SR_and_host_all_components=True,progress=True)
        assert sha(args.cold/name/'MANIFEST.json')==r['snapshots'][name]
    for name in ['endpoint_20','controller_20','transfer_20','observer_21']:assert r['checks'][name] is True
    assert r['observer_endpoint64_calls']==1 and r['observer_driver_function_unchanged'] is True
    for key in ['working_eager_restored','horner_cleanup_restored','working_graph_hook_restored',
        'valid_adapter_restored','cache_release_adapter_restored','weighted_adapter_restored']:assert r[key] is True
    w.eager_counters(r['working_eager_counters'],1)
    w.w.counters_gate(r['valid128_counters'],1);w.b.counters_gate(r['working_graph_counters'],1)
    weighted=r['weighted128_counters']
    assert weighted['chunk_size']==128 and 0<=weighted['refine_calls']<=1
    assert weighted['graph_map_calls']>=0
    assert r['postwarm_releases']
    for row in r['postwarm_releases']:
        assert row['allocated_after']<=row['allocated_before'] and row['reserved_after']<=row['reserved_before']
    w.log_gate(args.cold,[21],r);w.eager_log_gate(args.cold,[21],r)
    # Bind the actual cold observer bytes to the own40 observer, independently
    # of the cold checks boolean. The original eight comparisons remain above.
    for name in ['endpoint_20','controller_20','transfer_20','observer_21']:
        left=read(args.source40/(name+'.json'));right=read(args.cold/(name+'.json'))
        assert left['source_identity']==right['source_identity']==short_identity
        assert left['step']==right['step']==(21 if name=='observer_21' else 20)
        assert left['signature']==right['signature']
        assert sha(args.source40/(name+'.pt'))==left['pt_sha256']
        assert sha(args.cold/(name+'.pt'))==right['pt_sha256']
    return dict(source40_path=str(args.source40),source40_result_sha256=args.source40_result_sha256,
        source40_input_sha256=links['source_input_sha256'],cold_path=str(args.cold),cold_result_sha256=args.cold_result_sha256,
        cold_input_sha256=sha(args.cold/'INPUT.json'),qualified_short_runner_sha256=SHORT_SHA)


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
            assert row['before_after_mathematical_bytes_equal'] is True
            assert row['reference_mathematical_bytes_equal'] is (True if step<=40 else None)
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
    if len(steps)>=40:
        assert [r['step'] for r in summaries if r['step']<=40 and r['mathematical_bytes_checked']]==w.FULL_HASH_STEPS
        assert sum(not r['mathematical_bytes_checked'] for r in summaries if r['step']<=40)==31

class LongRuntime(w.Runtime):
    # Verbatim original long numerical methods. They use self.ctx for the P3
    # serializer/identity and do not call the old P2 initialization or Runtime.
    crown_bounds=p2.LongRuntime.crown_bounds
    inject_controls_s=p2.LongRuntime.inject_controls_s
    sr_signature=p2.LongRuntime.sr_signature
    plant_record=p2.LongRuntime.plant_record
    capture_failure=p2.LongRuntime.capture_failure
    advance_sparse=p2.LongRuntime.advance_sparse
    end_of_time_s=p2.LongRuntime.end_of_time_s

    def __init__(self,ctx,args):
        super().__init__(ctx,args)
        self.first_failure=None;self.failure_pending=False;self.failure_snapshot_attempted=False
        self.snapshot_errors=[];self.boundary_guard_checks=0

    def advance_core(self,st,code,eng,sched,settings,cap,sr):
        assert settings.order==3 and settings.step==.005 and code.order==2 and eng.tables.k==3
        # The reused long advance dispatches here, bypassing short advance.
        # Each begin runs exactly once; selective prune completes both bindings.
        self.ctx.working_eager_actual_step=self.completed+1
        self.ctx.eager_segments.begin_step(eng)
        self.ctx.working_prune.begin_step(eng)
        return p2.LongRuntime.advance_core(self,st,code,eng,sched,settings,cap,sr)


    def prune_state(self,*args,**kwargs):
        # Skip only b37 full-audit wrapper and e16c eager finish. The original
        # prune is called once; both finishes retain their original order.
        value=b.w.Runtime.prune_state(self,*args,**kwargs);started=time.perf_counter()
        helper=self.ctx.selective;binding=self.ctx.working_prune
        plan=helper.peek(binding,self.eng);storage=b.probe.storage_signature(self)
        full=bool(plan['stale_count'])
        if self.completed in self.ctx.selective_reference_evictions:
            assert dict(evicted_entries=plan['stale_count'],evicted_categories=plan['stale_categories'])==self.ctx.selective_reference_evictions[self.completed]
        before=b.probe.mathematical_signature(self) if full else None
        helper.assert_same_plan(binding,self.eng,plan)
        self.ctx.working_prune_current_step=self.completed
        eviction=binding.finish_step(self.eng);helper.assert_finished(plan,eviction)
        after=b.probe.mathematical_signature(self) if full else None
        assert b.probe.storage_signature(self)==storage
        if full:
            assert after==before
            if self.completed in self.ctx.selective_reference_math:
                assert before==self.ctx.selective_reference_math[self.completed]
        row=dict(event='selective_invariance_checked',step=self.completed,
            phase='after_original_prune_before_poststep_safety_reset',plan=plan,eviction=eviction,
            mathematical_bytes_checked=full,before_after_mathematical_bytes_equal=True if full else None,
            reference_mathematical_bytes_equal=True if full and self.completed in self.ctx.selective_reference_math else None,
            mathematical_signature=before,capacity_and_storage_unchanged=True,storage_before=storage,
            elapsed_s=time.perf_counter()-started)
        b.record(self.ctx,row)
        self.ctx.working_prune_rows.append({k:row[k] for k in ['step','elapsed_s','mathematical_bytes_checked',
            'before_after_mathematical_bytes_equal','reference_mathematical_bytes_equal','capacity_and_storage_unchanged']})
        self.ctx.eager_segments.finish_step(self.eng)
        return value

def initialize(args):
    ctx=w.initialize(args)  # All actual short gates/adapters, before any engine.
    short_identity=dict(ctx.identity);gate=admission(args,short_identity)
    ctx.identity=dict(short_identity,script_sha256=sha(__file__),policy=POLICY,
        qualified_short_identity=short_identity,execution_budget_steps=1000,long_admission=gate,
        reused_original_long_sha256=P2_LONG_SHA,donor_eager_long_sha256=DONOR_LONG_SHA,
        checkpoint_policy='first_failure_finished_once_then_reset1000_before_and_after_final_endpoint',
        boundary_SR_guard='metadata_storage_shape_dtype_version_sentinels; own_selective_40_and_cold_artifacts_qualified',
        postprune_SR_guard='Nonempty working eviction: full active state/SR/host payload before/after and storage/version/capacity. No-op: storage/version/capacity only. Before original reset at step1000.',
        timing_scope='Advance includes eager dispatch and per-region JSONL logging, excludes observer and selective postprune checks. Process includes conditional full-history hashing, cache-policy IO, and checkpoints; no speed claim.',
        limitation='Own selective40 and cold are mandatory. Qualification compared40 observers,5 full snapshots and9 eviction-step full mathematical signatures to e16c40;31 no-op steps had storage guards only. During long, full hashes run only on eviction; reference pairing stops after40. Original128 validation, observer64, fullSR/K20 and per-lane failure semantics remain. Conditional CROWN is not an end-to-end strict certificate.')
    return ctx

DriverOutput=p2.DriverOutput


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();result=dict(status='exception',mode='long1000')
    try:
        ctx=initialize(args);x=LongRuntime(ctx,args)
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode='long1000',original_target_steps=1000,budget_steps=1000,
            driver_argv=w.driver_argv(ctx),source40=str(args.source40),cold=str(args.cold)))
        argv,stdout=sys.argv,sys.stdout;tee=DriverOutput(stdout,args.output/'driver.stdout.txt')
        sys.argv=w.driver_argv(ctx);sys.stdout=tee
        try:returned=ctx.d.main()
        finally:sys.argv=argv;sys.stdout=stdout;tee.close()
        x.capture_failure('original_driver_return_after_original_abort_without_extra_reset')
        t=time.perf_counter();x.artifact('terminal_plant',x.plant_record());x.times['checkpoint_s']+=time.perf_counter()-t
        all_accepted=x.completed==1000 and len(x.rows)==1000 and all(all(r['accepted']) for r in x.rows)
        if all_accepted:
            assert x.controller_calls==50 and x.controller_steps==list(range(0,1000,20))
            assert x.phase=='after_strict_endpoint' and x.ledger.epoch==1000 and x.ledger.length==0
            assert {'committed_1000','after_endpoint_1000'}<=set(x.snapshots)
        if x.completed==1000:
            status='target_horizon_completed_all_lanes' if all_accepted else 'horizon_visited_with_failed_lanes'
        else:status='original_driver_stopped_before_target'
        result=dict(status=status,mode='long1000',driver_return=returned,original_driver_verdict_lines=tee.verdicts,
            overall_horizon_completed=all_accepted,original_target_steps=1000,budget_steps=1000,completed_step=x.completed,
            accepted_lane_steps=sum(r['accepted_lanes'] for r in x.rows),all_steps_all_lanes_accepted=all_accepted,
            controller_calls=x.controller_calls,controller_refresh_steps=x.controller_steps,
            first_failure=x.first_failure,snapshot_errors=x.snapshot_errors,snapshots=x.snapshots,
            final_progress=x.progress(),timing_components=x.times,cache_releases=x.cache_releases,
            boundary_SR_sentinel_checks=x.boundary_guard_checks,hybrid_validation_owners=ctx.hybrid.receipts,
            guarded_dense_aliases=ctx.hybrid.guarded_dense_aliases,
            fullbatch_qualification=False,end_to_end_strict_certificate=False,source_identity=ctx.identity)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),overall_horizon_completed=False)
        if 'x' in locals():
            # An unfinished propagate is never serialized as a committed state.
            result.update(completed_step=x.completed,phase=x.phase,first_failure=x.first_failure,
                snapshots=x.snapshots,snapshot_errors=x.snapshot_errors,pending_SR_token=x.ledger.pending if x.ledger else None,
                source_identity=ctx.identity,timing_components=x.times,cache_releases=x.cache_releases)
        raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-started)
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
                timing_scope='Process includes eviction-step full state/SR/host-history hashes, storage-only no-op guards and cache-policy IO. Advance excludes observer/postprune, but includes eager dispatch and its immediate JSONL logging. No reference math comparison beyond40; no solver-speed claim.')
            result['working_eager_counters']=dict(ctx.eager_segments.counters)
            ctx.eager_segments.restore();result['working_eager_restored']=True
            ctx.working_eager_log.close();result['working_eager_log_sha256']=sha(args.output/'working_eager.jsonl')
            ctx.horner_cleanup.restore();result['horner_cleanup_restored']=True
            ctx.working_prune.restore();result['working_graph_hook_restored']=True
            ctx.working_prune_log.close();result['working_prune_log_sha256']=sha(args.output/'working_prune.jsonl')
            ctx.valid128.restore();result['valid_adapter_restored']=True
            ctx.postwarm.restore();ctx.postwarm_log.close();result['cache_release_adapter_restored']=True
            ctx.weighted128.restore();result['weighted_adapter_restored']=True
            if result['status'] in ['target_horizon_completed_all_lanes','horizon_visited_with_failed_lanes','original_driver_stopped_before_target']:
                steps=list(range(1,x.completed+1))
                assert x.observer_endpoint64_calls==x.completed
                w.eager_counters(result['working_eager_counters'],x.completed)
                w.b.counters_gate(result['working_graph_counters'],x.completed)
                log_gate(args.output,steps,result);w.eager_log_gate(args.output,steps,result)
                if result['overall_horizon_completed']:w.w.counters_gate(result['valid128_counters'],1000)
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker',
        'working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','observer-check','observer-checker',
        'working-eager-check','working-eager-checker','selective-audit-check','selective-audit-checker','reference40','output','source40','cold']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40-result-sha256',required=True);p.add_argument('--cold-result-sha256',required=True)
    a=p.parse_args();a.mode='long1000';main(a)
