"""Diagnostic phase timing of the frozen selective40 path.

All solver calls, full1024, strict cap, SR1000 and numerical byte pairing
remain. Instrumentation does not qualify an optimized algorithm or a speedup.
CUDA execution requires the frozen CPU dispatch check and original own40.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
ENTRY_SHA='660c75cf22346e71d5044a60b5bbd62320ed38e542727cbcb6805fad909095e0'
PROFILE_SHA='0398dc1bb032fe85c7ef0a192e5667a12890934d85a71d5ab717adf360023166'
PROFILE_CHECK_SHA='86a21df970bc7bf29c3816b52f9a7edd593a6bfa662faa6dcf8c16b3f29bc025'
PROFILE_RESULT_SHA='864870168dbd722eed75963b09bdf499b30ececb3ef774c76b5add37332bb94b'
OWN40_INPUT_SHA='09b10002fbcf1f037195b77995db1bd6b957934fdf2d23116f0c68a32ecc0556'
OWN40_RESULT_SHA='bc844880ded347ad3a440fdfb25ab7fbc4792d7e3a00b6dde6222be86d032bd6'
POLICY='diagnostic_full1024_P3_selective40_outer_phase_timing'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
s=load('phase_profile_frozen_selective40',HERE/'run_fullbatch_p3_selective_audit.py',ENTRY_SHA)
read,write=s.read,s.write
w,b=s.w,s.b
BudgetStop=s.BudgetStop
driver_argv=s.driver_argv
eager_counters,eager_log_gate,log_gate=s.eager_counters,s.eager_log_gate,s.log_gate
compare_reference40=s.compare_reference40

def initialize(args):
    assert len(PROFILE_SHA)==len(PROFILE_CHECK_SHA)==len(PROFILE_RESULT_SHA)==64
    assert sha(HERE/'phase_profile.py')==PROFILE_SHA
    assert sha(args.profile_checker)==PROFILE_CHECK_SHA
    assert sha(args.profile_check/'RESULT.json')==PROFILE_RESULT_SHA
    check=read(args.profile_check/'RESULT.json')
    assert check['status']=='passed' and check['device']=='cpu'
    assert check['adapter_sha256']==PROFILE_SHA and check['script_sha256']==PROFILE_CHECK_SHA
    assert sha(args.profile_reference40/'INPUT.json')==OWN40_INPUT_SHA
    assert sha(args.profile_reference40/'RESULT.json')==OWN40_RESULT_SHA
    ctx=s.initialize(args)
    s.own40_gate(args.profile_reference40,ctx.identity)
    ctx.phase_profile_helper=load('quad_diagnostic_phase_profile',HERE/'phase_profile.py',PROFILE_SHA)
    algorithm_identity=dict(ctx.identity)
    ctx.identity=dict(ctx.identity,script_sha256=sha(__file__),policy=POLICY,
        algorithm_identity=algorithm_identity,diagnostic_only=True,phase_profile_adapter_sha256=PROFILE_SHA,
        phase_profile_cpu_result_sha256=PROFILE_RESULT_SHA,
        phase_profile_reference40_input_sha256=OWN40_INPUT_SHA,phase_profile_reference40_result_sha256=OWN40_RESULT_SHA,
        timing_scope='CPU outer-call wall and same-stream event span are diagnostic. Capture-containing phases have no GPU span. Event queries occur after the existing advance sync, before observer. No new device sync.',
        limitation='Diagnostic40 only; original numerical path and full byte reference checks remain. No solver speedup, fullT5 result or NNCS certificate claimed.')
    return ctx

class Runtime(s.Runtime):
    def __init__(self,ctx,args):
        super().__init__(ctx,args)
        from flowstar_gpu import graphing,weighted_validation
        ctx.phase_profile_log=(args.output/'phase_profile.jsonl').open('w',buffering=1)
        def record(row):ctx.phase_profile_log.write(json.dumps(row,allow_nan=False)+'\n')
        ctx.phase_profile=ctx.phase_profile_helper.install(ctx.torch,graphing,ctx.c.se,weighted_validation,ctx.host,record=record)
    def advance_sparse(self,st,code,eng,sched,settings,cap,sr):
        self.ctx.phase_profile.begin_step(self.completed+1,eng)
        try:return super().advance_sparse(st,code,eng,sched,settings,cap,sr)
        except BaseException:
            self.ctx.phase_profile.abort_step()
            raise
    def observe(self,duration):
        # Frozen002 has already synchronized advance and host finish here.
        self.ctx.phase_profile.finish_step(self.completed,original_sync_completed=True,advance_wall_s=duration)
        return super().observe(duration)

def main(args):
    assert args.mode=='bounded40' and args.source40 is None
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
        result.update(status='exception',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
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
            if hasattr(ctx,'phase_profile'):
                ctx.phase_profile.abort_step()
                ctx.phase_profile.restore();result['phase_profile_restored']=True
                ctx.phase_profile_log.close()
                result['phase_profile_log_sha256']=sha(args.output/'phase_profile.jsonl')
                result['phase_profile_rows']=list(ctx.phase_profile.rows)
                result['phase_profile_counters']=dict(ctx.phase_profile.counters)
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
        result.update(diagnostic_only=True,no_speedup_claim=True)
        if result['status']=='bounded_prefix_completed':
            assert 'reference40_comparison' in result
            assert result['phase_profile_restored'] is True
            assert [v['step'] for v in result['phase_profile_rows']]==list(range(1,41))
            pc=result['phase_profile_counters']
            assert pc['begun_steps']==pc['finished_steps']==40 and pc['aborted_steps']==0 and pc['restored'] is True
            assert all(v['original_sync_completed'] is True and v['profiler_added_synchronizations']==0 for v in result['phase_profile_rows'])
            result['status']='diagnostic40_bytes_equal'
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['bounded40'],default='bounded40')
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker',
        'working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','observer-check','observer-checker','working-eager-check','working-eager-checker','selective-audit-check','selective-audit-checker','reference40','profile-reference40','profile-check','profile-checker','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.source40=None;main(a)
