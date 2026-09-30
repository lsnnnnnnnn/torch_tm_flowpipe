"""Own full1024 P3 40/cold with isolated Horner closure lifetime cleanup.

Frozen b37c supplies the complete Runtime, selective graph pruning, full state
hash checks, driver and controls. Only the Horner recursive self-cell lifetime
changes. A pinned real CUDA gate and this identity's own40/cold are mandatory.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
BASE_SHA='b37cbdc4be1d7f4c2e84b096404913a84ead85c47995687511b66242a8eb3f09'
ADAPTER_SHA='f98958be75a46d49444d98b5c196ecef997ce6f63b70bfd2e541ab4e62ff0ad0'
CHECK_SHA='dcda3d41f27007448fb43e263bbb6050cb1fa24e6567c754e0947f8b2c614777'
CHECK_RESULT_SHA='c800901b55925334274cbb8e171acc56cc308cdd5fc599089607dd91526ee0eb'
CPU_CHECK_SHA='2d9c99f0962c16c15f6ae219e9b2998d924915e48ff5d418b3d8db2958735a8a'
CPU_RESULT_SHA='7ee8bbf63b341d4e83fbdd771eded2906cafe12791fd96e7c98c116b4ffffcb0'
CAPTURE_SHA='f693ac64f93787389b40d755a3647359ba4e7e535c7e87f70d5264544efc7aea'
CAPTURE_RESULT_SHA='c853691f7c9af50ce13558495d92e3d1a1d3d95a4aa18afcd95c019a1516e481'
ORIGINAL_SOURCE_SHA='d090e314f710d1d4763ea299336e0dba4dbada642b43700f09fec7c238977f4f'
ORIGINAL_FUNCTION_SHA='b67c07931e6ce230aced11a9ca7354b776aca0bd0d3f8081f07970321673014d'
CANDIDATE_FUNCTION_SHA='5ba2e410214f8193876d3c7bba62cf6d73aff3951d0fb86e083ec3045b54c4f3'
POLICY='full1024_P3_ordinaryvalid128_selective_working_graph_prune_horner_self_cell_cleanup'
CLEANUP_POLICY='clear_recursive_horner_evaluate_cell_after_original_return_expression'

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
b=load('horner_cleanup_frozen_working_prune',HERE/'run_fullbatch_p3_working_prune.py',BASE_SHA)
w=b.w
Runtime=b.Runtime
BudgetStop=b.BudgetStop
driver_argv=b.driver_argv

def qualification(args,identity=None):
    assert len(CHECK_SHA)==len(CHECK_RESULT_SHA)==64,'Real CUDA Horner cleanup qualification is not frozen'
    assert sha(args.horner_cleanup_checker)==CHECK_SHA
    path=args.horner_cleanup_check/'RESULT.json';assert sha(path)==CHECK_RESULT_SHA
    gate=read(path)
    assert gate['status']=='passed' and gate['device']=='cuda:0' and gate['script_sha256']==CHECK_SHA
    assert gate['adapter_sha256']==ADAPTER_SHA and gate['cpu_checker_sha256']==CPU_CHECK_SHA and gate['cpu_result_sha256']==CPU_RESULT_SHA
    assert gate['source_capture_sha256']==CAPTURE_SHA and gate['capture_result_sha256']==CAPTURE_RESULT_SHA
    assert (gate['input_rows'],gate['n'],gate['working_order'],gate['actual_cases'])==(1024,16,3,2)
    for key in ['all_coeff_remainder_support_bytes_equal','source_capture_passed','horner_original_restored','runtime_hooks_restored']:assert gate[key] is True
    assert gate['source_run_is_trajectory_qualification'] is False
    assert gate['source_identity']['script_sha256']==BASE_SHA and gate['source_identity']['policy']==b.POLICY
    if identity is not None:assert gate['source_identity']==identity
    assert gate['source_files_sha256']['RESULT.json']==CAPTURE_RESULT_SHA
    for step in [1,2]:
        for kind in ['INPUT','OUTPUT']:
            for suffix in ['pt','json']:assert len(gate['source_files_sha256'][f'COMPOSE_{kind}_{step}.{suffix}'])==64
    assert len(gate['arms'])==2
    for arm,candidate in zip(gate['arms'],[False,True]):
        assert arm['candidate'] is candidate and arm['captures']==2 and arm['hits']==3
        assert arm['actual_case2_repeated_same_key'] is True and arm['changed_data_same_key_test'] is False
        for key in ['cross_new_key_original_graph_outputs_preserved','owned_clones_survive_replay_and_gc','no_synthetic_inputs']:assert arm[key] is True
        assert arm['adapter_installed_before_owner'] is candidate
        assert len(arm['cases'])==2
        for step,row in enumerate(arm['cases'],1):
            assert row['step']==row['graph_captures']==row['graph_hits']==step
            for key in ['eager_actual_bytes_equal','graph_actual_bytes_equal','inputs_bytes_and_versions_unchanged']:assert row[key] is True
        if candidate:
            assert arm['inverse_AST_equal'] is True
            assert arm['original_function_sha256']==ORIGINAL_FUNCTION_SHA and arm['candidate_function_sha256']==CANDIDATE_FUNCTION_SHA
    return gate,dict(path=str(path),sha256=CHECK_RESULT_SHA,checker_path=str(args.horner_cleanup_checker),checker_sha256=CHECK_SHA,
        adapter_sha256=ADAPTER_SHA,cpu_checker_sha256=CPU_CHECK_SHA,cpu_result_sha256=CPU_RESULT_SHA,
        capture_sha256=CAPTURE_SHA,capture_result_sha256=CAPTURE_RESULT_SHA,source_files_sha256=gate['source_files_sha256'],
        fixed_inputs_only=True,changed_data_same_key_test=False,trajectory_qualification=False)

def initialize(args):
    qualification(args)  # PENDING stops before the original bootstrap/engine.
    assert sha(HERE/'horner_closure_cleanup.py')==ADAPTER_SHA
    ctx=b.initialize(args)
    gate,receipt=qualification(args,ctx.identity)
    assert gate['torch_version']==ctx.torch.__version__ and not ctx.hybrid.receipts
    adapter=load('fullbatch_horner_cleanup_adapter',HERE/'horner_closure_cleanup.py',ADAPTER_SHA)
    from flowstar_gpu import sparse_horner
    ctx.horner_cleanup=adapter.install(sparse_horner)
    binding=ctx.horner_cleanup
    assert binding.policy==CLEANUP_POLICY and binding.original_source_sha256==ORIGINAL_SOURCE_SHA
    assert binding.original_function_sha256==ORIGINAL_FUNCTION_SHA and binding.candidate_function_sha256==CANDIDATE_FUNCTION_SHA
    assert binding.inverse_AST_equal is True
    base=dict(ctx.identity)
    ctx.identity=dict(base,script_sha256=sha(__file__),policy=POLICY,base_working_prune_entry_sha256=BASE_SHA,
        base_working_prune_identity=base,horner_cleanup_adapter_sha256=ADAPTER_SHA,horner_cleanup_policy=CLEANUP_POLICY,
        horner_original_source_sha256=binding.original_source_sha256,horner_original_function_sha256=binding.original_function_sha256,
        horner_candidate_function_sha256=binding.candidate_function_sha256,horner_inverse_AST_equal=True,
        horner_cleanup_qualification=receipt,fullbatch_qualification=False,end_to_end_strict_certificate=False,
        limitation='Only additional change is Horner recursive-cell lifetime cleanup after original return expression. Full original1024/P3/point2/validation4, controls, cap/h/SR, ordinary128/weighted128/postwarm/selective working-cache policy and all postprune byte checks remain. No explicit GC diagnostic is installed. Fixed-input CUDA gate does not replace own40/cold. Conditional CROWN remains separate.')
    return ctx

def own40_gate(source,identity):
    source=Path(source);inp=read(source/'INPUT.json');r=read(source/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==identity
    assert identity['script_sha256']==r['script_sha256']==sha(__file__) and identity['policy']==POLICY
    assert identity['base_working_prune_entry_sha256']==BASE_SHA and identity['base_working_prune_identity']['script_sha256']==BASE_SHA
    assert identity['horner_cleanup_adapter_sha256']==ADAPTER_SHA and identity['horner_cleanup_policy']==CLEANUP_POLICY
    assert identity['horner_cleanup_qualification']['sha256']==CHECK_RESULT_SHA
    assert identity['horner_original_source_sha256']==ORIGINAL_SOURCE_SHA and identity['horner_inverse_AST_equal'] is True
    assert identity['horner_original_function_sha256']==ORIGINAL_FUNCTION_SHA and identity['horner_candidate_function_sha256']==CANDIDATE_FUNCTION_SHA
    assert r['horner_cleanup_restored'] is True
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
        else:
            argv=sys.argv;sys.argv=driver_argv(ctx)
            try:
                returned=ctx.d.main();result=dict(status='original_driver_stopped',mode=args.mode,driver_return=returned)
            except BudgetStop:
                assert x.completed==40 and x.ledger.length==x.sr.qlen==x.sr.jlen==40
                result=dict(status='bounded_prefix_completed' if all(all(r['accepted']) for r in x.rows) else 'bounded_prefix_with_failed_lanes',mode=args.mode)
                w.counters_gate(ctx.valid128.counters,40);b.counters_gate(ctx.working_prune.counters,40)
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
        'working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40',type=Path);a=p.parse_args();assert (a.source40 is not None)==(a.mode=='cold20_to21');main(a)
