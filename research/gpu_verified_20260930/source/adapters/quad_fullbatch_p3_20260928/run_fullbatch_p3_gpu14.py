"""User-requested GPU budget experiment: same complete P3 numerical run.

Original11.5GiB resource failure remains; no model/solver/batch/history change.
All original own40/cold gates execute before raising the current allocation
cap. The original LongRuntime is reused without modification.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
BASE_LONG_SHA='49da2756eb7637e1d3feeae27927bacd78f93511aa61bc802b863d2b3e1fec46'
WATCH_SHA='4298689de9ccf8a97c0007334dcdbe6aedeb246b8acead81a4b1873941a56350'
GIB=1024**3
ALLOCATOR_BYTES=27*GIB//2
GPU_GUARD_BYTES=14*GIB
RSS_GUARD_BYTES=23*GIB//2
POLICY='full1024_P3_selective_audit_1000_user_gpu14_budget'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
base=load('gpu14_frozen_selective_long',HERE/'run_fullbatch_p3_selective_audit_long.py',BASE_LONG_SHA)
read,write=base.read,base.write
w,b=base.w,base.b
LongRuntime=base.LongRuntime
DriverOutput=base.DriverOutput
log_gate=base.log_gate

def initialize(args):
    assert sha(HERE/'nncs_watchdog_gpu14.py')==WATCH_SHA
    ctx=base.initialize(args)  # Historical11/11.5GiB own40/cold identities stay intact.
    previous=dict(ctx.identity)
    assert previous['script_sha256']==BASE_LONG_SHA
    assert previous['torch_allocation_cap_bytes']==11*GIB
    assert previous['external_watch_limit_bytes']==23*GIB//2
    total=ctx.torch.cuda.get_device_properties(0).total_memory
    assert total>=GPU_GUARD_BYTES+GIB, 'This budget needs at least1GiB device headroom'
    ctx.torch.cuda.set_per_process_memory_fraction(ALLOCATOR_BYTES/total,0)
    ctx.identity=dict(previous,script_sha256=sha(__file__),policy=POLICY,
        base_long_entry_sha256=BASE_LONG_SHA,
        base_long_identity_sha256=hashlib.sha256(json.dumps(previous,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        torch_allocation_cap_bytes=ALLOCATOR_BYTES,external_watch_limit_bytes=GPU_GUARD_BYTES,
        resource_budget=dict(allocator_bytes=ALLOCATOR_BYTES,gpu_guard_bytes=GPU_GUARD_BYTES,
            rss_guard_bytes=RSS_GUARD_BYTES,torch_device_total_bytes=total,watchdog_sha256=WATCH_SHA,
            allocation_set_after_original_bootstrap_and_admission=True,
            previous_allocator_bytes=11*GIB,previous_gpu_guard_bytes=23*GIB//2,
            reason='User requested higher GPU budget after589 accepted steps hit11.5GiB guard; same current GPU, original numerical path and full1000 target.'),
        limitation=previous['limitation']+' Resource-only variant:13.5GiB Torch allocator,14GiB measured process GPU guard; RSS11.5GiB unchanged. Prior11.5GiB failure is preserved and not relabelled; no speedup or full completion follows from changing this budget.')
    return ctx

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
        result.update(status='exception',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),overall_horizon_completed=False)
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
