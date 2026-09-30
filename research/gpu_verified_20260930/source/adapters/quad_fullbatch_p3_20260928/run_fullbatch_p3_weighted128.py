"""Full1024 P3 with only early-weighted graph scratch reduced to128.

The frozen22f Runtime, original main and cold numerical body are unchanged.
The actual passed same-input CUDA qualification is pinned before execution.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
BASE_SHA='22f4dda16dcfd30563e6652ec8af82034b3b138a7505dbf31714a8bd55820a72'
ADAPTER_SHA='ae343c546e2253896142f88c39e1df5e43561498aaf7540e722a1bb6d66d409b'
CHECK_SHA='2e81fdd59833fb395521836ccc831e4822f6d98842dc8bfab32bc50b6bc46aa0'
CHECK_RESULT_SHA='141c3f8dc285d7ecee34afc0195401c2b3cb702997af0dd6492f491a91d1aa08'
PAYLOAD_SHA='895bad79fc31e5f6eefeb89f98e3401f0d7e2c2ec79a6ef4b4ddf678bed4fbe2'
AUDIT_SHA='5050e737a81fb4dcd023880df9cb6becf047a173536bfbb085a6f39fc48456f4'
CAPTURE_SHA='91cb4b9fcf1a7531cf5a2d89eba4ff02018140b6f8b83e7ad52d5f84ea17d4b5'
POLICY='full1024_P3_hybrid_directNN_reciprocal5_hostK20_earlyweighted128'

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
w=load('fullbatch_weighted128_frozen_P3',HERE/'run_fullbatch_p3.py',BASE_SHA)
Runtime=w.Runtime
BudgetStop=w.BudgetStop

def qualification(args,identity=None):
    # The actual passed CUDA result is pinned and reviewed. CPU/mock receipts
    # cannot admit a trajectory.
    assert len(CHECK_SHA)==len(CHECK_RESULT_SHA)==64,'actual weighted128 CUDA qualification is not frozen'
    assert sha(args.weighted_checker)==CHECK_SHA
    path=args.weighted_check/'RESULT.json';assert sha(path)==CHECK_RESULT_SHA
    g=read(path)
    assert g['status']=='passed' and g['device']=='cuda:0'
    assert g['script_sha256']==CHECK_SHA and g['adapter_sha256']==ADAPTER_SHA
    assert g['input_rows']==1024 and g['source_payload_sha256']==PAYLOAD_SHA
    assert g['source_entry_sha256']==BASE_SHA and g['source_capture_sha256']==CAPTURE_SHA
    assert g['source_audit_sha256']==AUDIT_SHA
    assert g['source_payload_complete'] is True
    assert g['source_capture_status']=='gpu_guard_11.5GiB' and g['source_capture_passed'] is False
    assert g['source_result_missing'] is True and g['source_sidecar_missing'] is True
    assert g['source_run_is_trajectory_qualification'] is False
    assert (g['working_order'],g['point_code_order'],g['validation_order'])==(3,2,4)
    assert (g['original_chunk_size'],g['candidate_chunk_size'])==(512,128)
    cases=g['parity_cases'];assert len(cases)==2
    assert {v['eligible_count'] for v in cases}=={1024,133}
    for row in cases:
        assert row['input_rows']==1024
        for key in ['output_bytes_equal','statistics_equal','input_bytes_unchanged',
                    'owned_outputs_survive_replay','all_target_map_bytes_equal',
                    'events_float_bytes_equal','eager_failed32_bytes_equal']:
            assert row[key] is True
    assert g['eager_failed32_unchanged'] is True and g['hooks_restored'] is True
    arms=g['arms'];assert len(arms)==4
    for arm,(eligible,chunk) in zip(arms,[(1024,512),(1024,128),(133,512),(133,128)]):
        assert (arm['input_rows'],arm['eligible_count'],arm['chunk_size'])==(1024,eligible,chunk)
        assert arm['graph']['enabled'] is True and arm['graph']['captures']>=1 and arm['graph']['hits']>0
        assert arm['graph']['rows']==[chunk]
        assert arm['input_bytes_unchanged'] is True and arm['owned_outputs_survive_replay'] is True
        if chunk==128:
            assert arm['counters']['chunk_size']==128 and arm['counters']['refine_calls']==1
            assert arm['counters']['graph_map_calls']==arm['graph']['hits']
            if eligible==133:assert arm['counters']['padded_rows']>0
    if identity is not None:
        assert g['engine_python_sha256']==identity['engine']['python_sha256']
        assert g['extensions']==identity['extensions'] and len(g['extensions'])==8
        assert g['reciprocal_extensions']==identity['reciprocal_extensions']
        assert g['reciprocal_candidate']==identity['reciprocal_candidate']
        assert g['hybrid_adapter_sha256']==identity['hybrid_adapter_sha256']
        assert g['metadata_backend_sha256']==identity['metadata_backend_sha256']
    return g,dict(path=str(path),sha256=CHECK_RESULT_SHA,checker_path=str(args.weighted_checker),
                  checker_sha256=CHECK_SHA,source_payload_sha256=PAYLOAD_SHA,source_audit_sha256=AUDIT_SHA,
                  source_capture_status=g['source_capture_status'],source_capture_passed=False,
                  source_result_missing=True,source_sidecar_missing=True,source_run_is_trajectory_qualification=False)

def initialize(args):
    assert sha(HERE/'weighted_chunk128.py')==ADAPTER_SHA
    qualification(args)  # Fail closed before the original bootstrap/engine work.
    ctx=w.initialize(args)
    gate,receipt=qualification(args,ctx.identity)
    assert gate['torch_version']==ctx.torch.__version__
    from flowstar_gpu import weighted_validation as weighted
    adapter=load('fullbatch_weighted128_adapter',HERE/'weighted_chunk128.py',ADAPTER_SHA)
    ctx.weighted128=adapter.install(ctx.torch,weighted,ctx.c.se)
    base_identity=dict(ctx.identity)
    ctx.identity=dict(base_identity,script_sha256=sha(__file__),policy=POLICY,
        base_P3_entry_sha256=BASE_SHA,base_P3_identity=base_identity,
        weighted_adapter_sha256=ADAPTER_SHA,weighted_adapter_policy=ctx.weighted128.policy,
        weighted128_qualification=receipt,early_weighted_chunk_size=128,early_weighted_graph_padding=128,
        ordinary_failed_validation_chunk_size=32,early_weighted_rounds=2,early_weighted_attempts_per_round=1,
        fullbatch_qualification=False,end_to_end_strict_certificate=False,
        limitation='Only early-weighted graph scratch uses128 rows. All1024 original boxes/NN rows, P3/point2/validation4, cap, rounds and SR remain unchanged. Own40/cold are required; conditional CROWN certificates remain a separate limitation.')
    assert (ctx.identity['batch_size'],ctx.identity['working_total_degree'],ctx.identity['point_code_order'],ctx.identity['validation_order'])==(1024,3,2,4)
    return ctx

def own40_gate(source,identity):
    source=Path(source);inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==identity
    assert identity['script_sha256']==r['script_sha256']==sha(__file__) and identity['policy']==POLICY
    assert identity['base_P3_entry_sha256']==BASE_SHA and identity['weighted_adapter_sha256']==ADAPTER_SHA
    assert identity['weighted128_qualification']['sha256']==CHECK_RESULT_SHA
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
    assert r['weighted_adapter_restored'] is True
    assert set(r['snapshots'])=={'committed_20','after_endpoint_20','after_controller_20','committed_21','committed_40'}
    for name,digest in r['snapshots'].items():assert sha(source/name/'MANIFEST.json')==digest
    return dict(source_input_sha256=sha(source/'INPUT.json'),source_result_sha256=sha(source/'RESULT.json'))

def cold(runtime,source):
    # Change only which independently identified short run is admitted. The
    # frozen cold body and its full-byte snapshot/artifact comparators remain.
    raw=w.own40_gate;w.own40_gate=own40_gate
    try:return w.cold(runtime,source)
    finally:w.own40_gate=raw

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception',mode=args.mode)
    try:
        ctx=initialize(args);x=Runtime(ctx,args)
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode=args.mode,original_target_steps=1000,budget_steps=40,
            driver_argv=w.driver_argv(ctx),source40=str(args.source40) if args.source40 else None))
        if args.mode=='cold20_to21':result=cold(x,args.source40)
        else:
            argv=sys.argv;sys.argv=w.driver_argv(ctx)
            try:
                returned=ctx.d.main();result=dict(status='original_driver_stopped',mode=args.mode,driver_return=returned)
            except BudgetStop:
                assert x.completed==40 and x.ledger.length==x.sr.qlen==x.sr.jlen==40
                result=dict(status='bounded_prefix_completed' if all(all(r['accepted']) for r in x.rows) else 'bounded_prefix_with_failed_lanes',mode=args.mode)
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
                          weighted128_counters=dict(ctx.weighted128.counters))
            ctx.weighted128.restore();result['weighted_adapter_restored']=True
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['bounded40','cold20_to21'],required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40',type=Path);a=p.parse_args();assert (a.source40 is not None)==(a.mode=='cold20_to21');main(a)
