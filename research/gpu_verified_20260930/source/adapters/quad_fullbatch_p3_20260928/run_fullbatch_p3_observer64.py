"""Own full1024 P3 40/cold with observer-only endpoint scratch blocks.

Frozen b89 supplies original128 validation, Runtime, driver, NN and SR.
Only the endpoint call inside Runtime.observe uses the pure lane64 helper.
No driver/safety/NN range function is patched, and no GC policy is added.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
BASE_SHA='b89c5710f5faa9dc1b363eba96d27cbbc7bc2a7006df9aa2d305f39cf202b2e7'
ADAPTER_SHA='d8d87a51a7db600685a79ff19193a56986dab6e84b4825a30570caaf157c8872'
CHECK_SHA='42d1eb24c1a59f03ad208eb44a5615b87e2ccd456b8af1c9f59c0a823740de61'
CHECK_RESULT_SHA='42f2beba0ac922987cf16f995a9165940ae323c96a4d1b7c18c88df3df6c75ab'
POLICY='full1024_P3_original128_horner_cleanup_observer_endpoint64'
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
h=load('observer64_frozen_horner_entry',HERE/'run_fullbatch_p3_horner_cleanup.py',BASE_SHA)
b=h.b;w=h.w
BudgetStop=h.BudgetStop
driver_argv=h.driver_argv

def qualification(args,identity=None,torch_version=None):
    assert len(CHECK_SHA)==len(CHECK_RESULT_SHA)==64,'Actual observer endpoint CUDA qualification is not yet frozen'
    assert sha(args.observer_checker)==CHECK_SHA
    path=args.observer_check/'RESULT.json';assert sha(path)==CHECK_RESULT_SHA
    gate=read(path)
    assert gate['status']=='passed' and gate['device']=='cuda:0' and gate['script_sha256']==CHECK_SHA
    assert gate['adapter_sha256']==ADAPTER_SHA and gate['policy']==ENDPOINT_POLICY
    assert gate['source_capture_passed'] is False and gate['source_run_is_trajectory_qualification'] is False
    assert gate['source_capture_status']=='gpu_guard_11.5GiB' and gate['source_capture_result_present'] is False
    assert (gate['input_rows'],gate['physical_rows'],gate['n'],gate['working_order'],gate['actual_cases'],gate['basis_slots'],gate['chunk_size'])==(1024,12,16,3,2,1140,64)
    assert gate['source_identity']['script_sha256']==BASE_SHA and gate['source_identity']['policy']==h.POLICY
    if identity is not None:assert gate['source_identity']==identity
    if torch_version is not None:assert gate['torch_version']==torch_version
    assert gate['source_capture_sha256']=='0d4062b47996a4e5ae814c92f3274f6b441109969138b3ebfbc27b7edbd2749e'
    assert gate['capture_input_sha256']=='d26ffe06640b4aa688198c72d30e429fae6d49155b56cd0a9c8a1d259429e742'
    assert gate['capture_watch_sha256']=='b58e90e8edc5c6208ed1fde8bfe70a59393491d272be0db5fd4c8197135fdf39'
    for key in ['all_endpoint_bytes_equal','all_captured_original_endpoint_bytes_equal','previous_owned_outputs_survive_next_call',
        'actual_input_values_changed','original_function_unchanged','table_bytes_unchanged','no_graph_created','runtime_hooks_restored']:assert gate[key] is True
    assert [row['step'] for row in gate['rows']]==[1,2]
    for row in gate['rows']:
        assert (row['full_batch'],row['chunk_size'],row['block_calls'])==(1024,64,16)
        for key in ['all_endpoint_bytes_equal','captured_original_endpoint_bytes_equal','inputs_bytes_and_versions_unchanged','finite_ordered']:assert row[key] is True
        out=row['output_signature'];assert out['shape']==[1024,12,2] and out['dtype']=='torch.float64' and len(out['sha256'])==64
    assert gate['source_files_sha256']['INPUT.json']==gate['capture_input_sha256']
    for step in [1,2]:
        for name in [f'OBSERVER_INPUT_{step}.pt',f'OBSERVER_INPUT_{step}.json',f'observer_{step}.pt']:
            assert len(gate['source_files_sha256'][name])==64
        assert gate['source_files_sha256'][f'observer_{step}.pt']==gate['reference_files_sha256'][f'observer_{step}.pt']
    return gate,dict(path=str(path),sha256=CHECK_RESULT_SHA,checker_sha256=CHECK_SHA,
        helper_sha256=ADAPTER_SHA,source_files_sha256=gate['source_files_sha256'],
        fixed_inputs_only=True,source_capture_passed=False,trajectory_qualification=False)


def initialize(args):
    qualification(args)
    assert sha(HERE/'observer_endpoint64.py')==ADAPTER_SHA
    ctx=h.initialize(args);gate,receipt=qualification(args,ctx.identity,ctx.torch.__version__)
    assert not ctx.hybrid.receipts
    helper=load('fullbatch_observer_endpoint64_helper',HERE/'observer_endpoint64.py',ADAPTER_SHA)
    assert helper.POLICY==ENDPOINT_POLICY
    verified=helper.verify_sources(ctx.d.rows_range_over_time_sparse)
    assert gate['original_sources_sha256']==verified
    ctx.observer_endpoint64=helper;ctx.observer_original_range=ctx.d.rows_range_over_time_sparse
    ctx.identity=dict(ctx.identity,script_sha256=sha(__file__),policy=POLICY,base_horner_cleanup_entry_sha256=BASE_SHA,
        observer_endpoint_helper_sha256=ADAPTER_SHA,observer_endpoint_policy=ENDPOINT_POLICY,
        observer_endpoint_qualification=receipt,observer_endpoint_lane_chunk=64,observer_endpoint_physical_rows=12,
        observer_scope='Only Runtime.observe endpoint values; original driver NN/safety/range functions are unchanged.',
        fullbatch_qualification=False,end_to_end_strict_certificate=False,
        limitation='Original full1024/P3/point2/validation4 and128 ordinary/weighted validation retained. Only experiment-observer endpoint temporary arrays use64 lanes, with every original1140 basis slot and original directed operation order. Input-capture resource failure remains recorded. Fixed-input qualification is not own40/cold, a full nonlinear trajectory proof or a speed result; conditional CROWN remains separate.')
    return ctx


class Runtime(h.Runtime):
    def __init__(self,ctx,args):
        super().__init__(ctx,args);self.observer_endpoint64_calls=0
    def observe(self,duration):
        t=time.perf_counter();st=self.st
        versions=[(getattr(st,k),getattr(st,k)._version) for k in ['pre','pre_rem','tmv','tmv_rem','status']]
        tube=self.d.hull_ranges_s(st,self.eng,12)
        piece=self.t.full((1024,2),.005,dtype=self.t.float64,device=st.pre.device)
        endpoint=self.ctx.observer_endpoint64.endpoint64(st,self.eng,piece,12,raw=self.d.rows_range_over_time_sparse)
        self.observer_endpoint64_calls+=1
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


def own40_gate(source,identity):
    source=Path(source);inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==identity
    assert identity['script_sha256']==r['script_sha256']==sha(__file__) and identity['policy']==POLICY
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
            assert x.observer_endpoint64_calls==1
        else:
            argv=sys.argv;sys.argv=driver_argv(ctx)
            try:
                returned=ctx.d.main();result=dict(status='original_driver_stopped',mode=args.mode,driver_return=returned)
            except BudgetStop:
                assert x.completed==40 and x.ledger.length==x.sr.qlen==x.sr.jlen==40
                result=dict(status='bounded_prefix_completed' if all(all(r['accepted']) for r in x.rows) else 'bounded_prefix_with_failed_lanes',mode=args.mode)
                w.counters_gate(ctx.valid128.counters,40);b.counters_gate(ctx.working_prune.counters,40)
                assert x.observer_endpoint64_calls==40
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
            ctx.horner_cleanup.restore();result['horner_cleanup_restored']=True
            ctx.working_prune.restore();result['working_graph_hook_restored']=True
            ctx.working_prune_log.close();result['working_prune_log_sha256']=sha(args.output/'working_prune.jsonl')
            ctx.valid128.restore();result['valid_adapter_restored']=True
            ctx.postwarm.restore();ctx.postwarm_log.close();result['cache_release_adapter_restored']=True
            ctx.weighted128.restore();result['weighted_adapter_restored']=True
            if result['status'] in ['bounded_prefix_completed','passed']:
                b.log_gate(args.output,[21] if args.mode=='cold20_to21' else list(range(1,41)),result)
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['bounded40','cold20_to21'],required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker',
        'working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','observer-check','observer-checker','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40',type=Path);a=p.parse_args();assert (a.source40 is not None)==(a.mode=='cold20_to21');main(a)
