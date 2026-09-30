"""GPU14 resource variant of the unchanged full1024 P3 phase-profile40 diagnostic.

Every original admission/own40 gate runs before raising the allocator cap.
The profiler, Runtime, 40-step loop and actual byte pairing remain frozen.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
BASE_PROFILE_SHA='b4d9927d89b61b2829de0733c00344b70282645c6254d6d2db61fde65927498d'
WATCH_SHA='4298689de9ccf8a97c0007334dcdbe6aedeb246b8acead81a4b1873941a56350'
PROFILE_SHA='0398dc1bb032fe85c7ef0a192e5667a12890934d85a71d5ab717adf360023166'
GIB=1024**3
ALLOCATOR_BYTES=27*GIB//2
GPU_GUARD_BYTES=14*GIB
RSS_GUARD_BYTES=23*GIB//2
POLICY='diagnostic_full1024_P3_selective40_outer_phase_timing_gpu14'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
base=load('gpu14_frozen_phase_profile40',HERE/'run_fullbatch_p3_phase_profile.py',BASE_PROFILE_SHA)
read,write=base.read,base.write
w,b=base.w,base.b
Runtime=base.Runtime
BudgetStop=base.BudgetStop
driver_argv=base.driver_argv
eager_counters,eager_log_gate,log_gate=base.eager_counters,base.eager_log_gate,base.log_gate
compare_reference40=base.compare_reference40

def resource_gate(args):
    assert sha(HERE/'nncs_watchdog_gpu14.py')==WATCH_SHA
    path=args.resource_check/'RESULT.json'
    assert len(args.resource_check_sha256)==64 and sha(path)==args.resource_check_sha256
    result=read(path)
    assert result['status']=='passed' and result['device']=='cpu' and result['CUDA_tested'] is False
    assert result['entry_sha256']==sha(__file__) and result['base_sha256']==BASE_PROFILE_SHA
    assert result['script_sha256']==sha(HERE/'check_phase_profile_gpu14_cpu.py')
    assert result['watchdog_sha256']==WATCH_SHA and result['profile_helper_sha256']==PROFILE_SHA
    for name in ['main_AST_identical','Runtime_exact_same_object','original_gates_before_allocation_increase',
                 'previous_identity_unchanged','resource_identity_separate','invalid_admissions_do_not_raise_budget']:
        assert result[name] is True, name
    return result

def initialize(args):
    resource_gate(args)
    ctx=base.initialize(args)  # All original profile/own40/numerical admission gates first.
    previous=dict(ctx.identity)
    assert previous['script_sha256']==BASE_PROFILE_SHA
    assert previous['torch_allocation_cap_bytes']==11*GIB
    assert previous['external_watch_limit_bytes']==23*GIB//2
    total=ctx.torch.cuda.get_device_properties(0).total_memory
    assert total>=GPU_GUARD_BYTES+GIB, 'This budget needs at least1GiB device headroom'
    ctx.torch.cuda.set_per_process_memory_fraction(ALLOCATOR_BYTES/total,0)
    ctx.identity=dict(previous,script_sha256=sha(__file__),policy=POLICY,
        base_profile_entry_sha256=BASE_PROFILE_SHA,
        base_profile_identity_sha256=hashlib.sha256(json.dumps(previous,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        resource_cpu_result_sha256=args.resource_check_sha256,
        torch_allocation_cap_bytes=ALLOCATOR_BYTES,external_watch_limit_bytes=GPU_GUARD_BYTES,
        resource_budget=dict(allocator_bytes=ALLOCATOR_BYTES,gpu_guard_bytes=GPU_GUARD_BYTES,
            rss_guard_bytes=RSS_GUARD_BYTES,torch_device_total_bytes=total,watchdog_sha256=WATCH_SHA,
            allocation_set_after_original_bootstrap_and_admission=True,
            previous_allocator_bytes=11*GIB,previous_gpu_guard_bytes=23*GIB//2),
        limitation=previous['limitation']+' Resource-only profile40 variant:13.5GiB allocator,14GiB measured GPU guard,RSS11.5GiB unchanged. Actual original40 byte pairing remains mandatory; no speedup/five-repeat or additional fullT5 qualification.')
    return ctx

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
        'working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','observer-check','observer-checker','working-eager-check','working-eager-checker','selective-audit-check','selective-audit-checker','reference40','profile-reference40','profile-check','profile-checker','resource-check','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--resource-check-sha256',required=True)
    a=p.parse_args();a.source40=None;main(a)
