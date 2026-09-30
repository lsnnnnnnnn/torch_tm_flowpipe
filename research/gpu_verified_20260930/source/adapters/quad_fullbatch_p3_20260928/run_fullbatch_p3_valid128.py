"""Full1024 P3 with ordinary validated-RHS scratch partitioned into128 rows.

Frozen6e74 initialization/Runtime/driver/NN/SR/cold arithmetic are reused. Only
the original ordinary validation dispatcher is additionally wrapped. A fixed-
input CUDA gate is required, followed by this identity's own40 and cold run.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
BASE_SHA='6e74c716ac3ec9da791c6e968f73df0cc400030d218857152629976109213674'
ADAPTER_SHA='6b97a9ed19b8433b79be15f97381241bc9a79c071b90fa4a81f12c4d216dcc40'
CHECK_SHA='30a49eea8ff44be2c8bf7d3f41c60f4ffbfc09a5cecd84108b0e77d4c999fd28'
CHECK_RESULT_SHA='053947f483678ed087b849f18c90f34cc827215c0dea884541ae072d978307cf'
POLICY='full1024_P3_hybrid_directNN_reciprocal5_hostK20_weighted128_postwarm_ordinaryvalid128'
VALID_POLICY='full1024_validated_RHS_original_dispatch_fixed128_owned_full_outputs'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,path,digest):
    assert sha(path)==digest
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
w=load('fullbatch_valid128_frozen_postwarm',HERE/'run_fullbatch_p3_cache_release.py',BASE_SHA)
Runtime=w.Runtime
BudgetStop=w.BudgetStop
driver_argv=w.driver_argv

def counters_gate(counters,min_calls):
    assert counters['full_batch']==1024 and counters['chunk_size']==128
    assert counters['calls']>=min_calls and counters['block_calls']==8*counters['calls']
    assert 1<=counters['registered_specs']<=counters['calls']

def qualification(args,identity=None):
    assert len(CHECK_SHA)==len(CHECK_RESULT_SHA)==64,'Actual valid128 CUDA qualification is not frozen'
    assert sha(args.valid_checker)==CHECK_SHA
    path=args.valid_check/'RESULT.json';assert sha(path)==CHECK_RESULT_SHA
    g=read(path)
    assert g['status']=='passed' and g['device']=='cuda:0'
    assert g['script_sha256']==CHECK_SHA and g['adapter_sha256']==ADAPTER_SHA
    assert g['source_capture_passed'] is False and g['source_run_is_trajectory_qualification'] is False
    assert g['input_rows']==1024 and (g['working_order'],g['point_code_order'],g['validation_order'])==(3,2,4)
    for key in ['all_five_outputs_all_three_cases_bytes_equal','consecutive_actual_inputs_changed',
                'same_owner_second_graph_hit','runtime_hooks_restored']:assert g[key] is True
    assert len(g['source_capture_sha256'])==len(g['source_audit_sha256'])==64
    assert all(name in g['source_files_sha256'] for name in ['INPUT.json','VALID_INPUT_1.pt','VALID_INPUT_2.pt'])
    assert g['max_cuda_allocated_bytes']>0 and g['max_cuda_reserved_bytes']>0
    assert len(g['arms'])==2
    for arm,chunk in zip(g['arms'],[1024,128]):
        assert arm['chunk']==chunk and len(arm['cases'])==3
        for i,(row,case) in enumerate(zip(arm['cases'],['step1','step2','step2_marked_bad']),1):
            assert row['case']==case and row['captures']==1 and row['hits']==i*(8 if chunk==128 else 1)
            assert row['graph_input_rows']==chunk
            for key in ['input_bytes_unchanged','marked_bad_preserved','owned_previous_outputs_survive_next_replay']:
                assert row[key] is True
            assert set(row['outputs'])=={f'outputs/{j}' for j in range(5)}
        if chunk==1024:assert arm['counters'] is None
        else:
            counters_gate(arm['counters'],3)
            assert arm['counters']['calls']==3 and arm['counters']['registered_specs']==1
    if identity is not None:assert g['source_identity']==identity
    assert g['source_identity']['script_sha256']==BASE_SHA and g['source_identity']['policy']==w.POLICY
    return g,dict(path=str(path),sha256=CHECK_RESULT_SHA,checker_path=str(args.valid_checker),
        checker_sha256=CHECK_SHA,adapter_sha256=ADAPTER_SHA,source_capture_sha256=g['source_capture_sha256'],
        source_audit_sha256=g['source_audit_sha256'],source_files_sha256=g['source_files_sha256'],
        source_capture_passed=False,source_run_is_trajectory_qualification=False)

def initialize(args):
    assert sha(HERE/'valid_chunk128.py')==ADAPTER_SHA
    qualification(args)  # Placeholder or failed/foreign gate stops before bootstrap.
    ctx=w.initialize(args)
    gate,receipt=qualification(args,ctx.identity)
    assert gate['torch_version']==ctx.torch.__version__
    assert not ctx.hybrid.receipts,'Install before any validation engine/spec/graph exists'
    adapter=load('fullbatch_ordinary_valid128_adapter',HERE/'valid_chunk128.py',ADAPTER_SHA)
    ctx.valid128=adapter.install(ctx.torch,ctx.c.se)
    assert ctx.valid128.policy==VALID_POLICY
    base_identity=dict(ctx.identity)
    ctx.identity=dict(base_identity,script_sha256=sha(__file__),policy=POLICY,
        base_postwarm_entry_sha256=BASE_SHA,base_postwarm_identity=base_identity,
        ordinary_valid_adapter_sha256=ADAPTER_SHA,ordinary_valid_policy=ctx.valid128.policy,
        ordinary_valid_chunk_size=128,ordinary_valid_full_rows=1024,valid128_qualification=receipt,
        fullbatch_qualification=False,end_to_end_strict_certificate=False,
        limitation='Only ordinary validated-RHS scratch additionally uses128 rows; all1024 original boxes/NN rows, P3/point2/validation4, h/cap/SR and full outputs remain. Existing weighted128/postwarm policies stay. Own new40/cold required; fixed-input qualification is not a trajectory certificate. Conditional CROWN remains separate.')
    assert (ctx.identity['batch_size'],ctx.identity['working_total_degree'],ctx.identity['point_code_order'],ctx.identity['validation_order'])==(1024,3,2,4)
    return ctx

def own40_gate(source,identity):
    source=Path(source);inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==identity
    assert identity['script_sha256']==r['script_sha256']==sha(__file__) and identity['policy']==POLICY
    assert identity['base_postwarm_entry_sha256']==BASE_SHA and identity['ordinary_valid_adapter_sha256']==ADAPTER_SHA
    assert identity['ordinary_valid_policy']==VALID_POLICY and identity['valid128_qualification']['sha256']==CHECK_RESULT_SHA
    assert identity['ordinary_valid_chunk_size']==128 and identity['ordinary_valid_full_rows']==1024
    assert identity['cache_release_adapter_sha256']==w.ADAPTER_SHA
    assert identity['cache_release_policy']==w.CACHE_POLICY and identity['cache_release_qualification']['sha256']==w.CHECK_RESULT_SHA
    assert identity['weighted128_qualification']['sha256']==w.w.CHECK_RESULT_SHA
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
    counters_gate(r['valid128_counters'],40)
    assert r['valid_adapter_restored'] is True and r['weighted_adapter_restored'] is True
    assert r['cache_release_adapter_restored'] is True and r['postwarm_releases']
    for row in r['postwarm_releases']:
        assert row['allocated_after']<=row['allocated_before'] and row['reserved_after']<=row['reserved_before']
    assert set(r['snapshots'])=={'committed_20','after_endpoint_20','after_controller_20','committed_21','committed_40'}
    for name,digest in r['snapshots'].items():assert sha(source/name/'MANIFEST.json')==digest
    return dict(source_input_sha256=sha(source/'INPUT.json'),source_result_sha256=sha(source/'RESULT.json'))

def cold(runtime,source):
    # 6e74 -> 2c6e -> 22f. Only replace the gate used by the original cold body.
    base=w.w.w;raw=base.own40_gate;base.own40_gate=own40_gate
    try:return base.cold(runtime,source)
    finally:base.own40_gate=raw

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception',mode=args.mode)
    try:
        ctx=initialize(args);x=Runtime(ctx,args)
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode=args.mode,original_target_steps=1000,budget_steps=40,
            driver_argv=driver_argv(ctx),source40=str(args.source40) if args.source40 else None))
        if args.mode=='cold20_to21':
            result=cold(x,args.source40);counters_gate(ctx.valid128.counters,1)
        else:
            argv=sys.argv;sys.argv=driver_argv(ctx)
            try:
                returned=ctx.d.main();result=dict(status='original_driver_stopped',mode=args.mode,driver_return=returned)
            except BudgetStop:
                assert x.completed==40 and x.ledger.length==x.sr.qlen==x.sr.jlen==40
                result=dict(status='bounded_prefix_completed' if all(all(r['accepted']) for r in x.rows) else 'bounded_prefix_with_failed_lanes',mode=args.mode)
                counters_gate(ctx.valid128.counters,40)
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
                postwarm_releases=list(ctx.postwarm.releases),postwarm_release_s=sum(row['elapsed_s'] for row in ctx.postwarm.releases))
            ctx.valid128.restore();result['valid_adapter_restored']=True
            ctx.postwarm.restore();ctx.postwarm_log.close();result['cache_release_adapter_restored']=True
            ctx.weighted128.restore();result['weighted_adapter_restored']=True
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['bounded40','cold20_to21'],required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40',type=Path);a=p.parse_args();assert (a.source40 is not None)==(a.mode=='cold20_to21');main(a)
