"""P3 root1/B2 from t0: five-route reciprocal fix, original direct NN domain.

Reuse the frozen transfer/metadata/plant/endpoint/observer loops unchanged.
Own completed40 and own cold20->NN->21 are mandatory before a longer run.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time

TRANSFER_SHA='c16302e3b1a3cbdaa949f6379a7c1e806699c564a45bf4c0012d9a8477905a08'
PINS={'core.py':'765660877a93aaf12ef2e3dc6f565c5e01358f645b3dcb54ce9d3f09c957233e',
      'install.py':'99193a654708e8aab4402d90553c01387bf47a871d982313abc094421502c8ac',
      'preflight.py':'c7e3ab9bf370fda257261750f9cde8c7d3cd65e328e2d4ec838cd2bf91211543',
      'check_paths.py':'81dd23abb3c463b929ef913b8ffb433bd363dac13e4e117742b30369ab6c30fc'}
SETTINGS=dict(step=.005,order=3,cutoff=1e-6,remainder_estimation=.1,mode='strict',device='cuda')
POLICY='finite_geometric_reciprocal_tail_all_five_routes_v1'
DOMAIN='original_direct_pre_plus_R_union_box'
LIMITATION=('Five-route reciprocal candidate with original direct NN input domain and conditional strict transfer. '
    'CROWN floating-point certificate and same-slope validity remain unqualified; not an end-to-end strict NNCS certificate.')
BINDINGS=('reciprocal_policy','reciprocal_candidate','reciprocal_extensions','reciprocal_build_path',
          'reciprocal_build_result_sha256','reciprocal_check_path','reciprocal_check_result_sha256',
          'reciprocal_check_script_sha256','controller_input_domain_policy','frozen_transfer_runner_sha256')
TRANSFER_FIELDS={'completed_step','working_order','pre_ids','center','point_rows','coefficient_error',
                 'coefficient_error_range','weighted_input_remainder','bias_residual','new_control_remainder'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def candidate_gate(args):
    """Check actual qualification/build/source/binary receipts before bootstrap."""
    directory=Path(__file__).parent.parent/'reciprocal_geometric_20260928'
    for name,digest in PINS.items():assert sha(directory/name)==digest,name
    build=read(args.candidate_build/'RESULT.json');gate=read(args.candidate_check/'RESULT.json')
    assert build['status']=='passed' and build['device']=='cuda' and build['script_sha256']==PINS['preflight.py']
    assert gate['status']=='passed' and gate['device']=='cuda' and gate['script_sha256']==PINS['check_paths.py']
    assert gate['build_result_sha256']==sha(args.candidate_build/'RESULT.json')
    assert gate['source_files']=={name:PINS[name] for name in ('core.py','install.py')}
    assert gate['candidate']==build['candidate'] and gate['extensions']==build['extensions']
    candidate=gate['candidate']
    assert candidate['core_sha256']==PINS['core.py'] and candidate['installer_sha256']==PINS['install.py']
    assert candidate['covers']==['generic','dense','python_replay','cuda_replay','cuda_valid'] and candidate['fallback_disabled_routes']==[]
    names=candidate['extension_names'];assert set(names)=={'replay','valid'} and len(set(names.values()))==2
    assert set(gate['extensions'])==set(names.values())
    for name,entry in gate['extensions'].items():
        assert Path(entry['path']).resolve().is_relative_to(Path(candidate['cache_root']).resolve())
        assert sha(entry['path'])==entry['sha256'],name
    assert gate['all_five_routes_installed'] is True and gate['original_prefix_AST_equal_all_three_routes'] is True
    assert gate['no_cuda_compile_or_gpu_execution'] is False
    assert gate['analytic_fraction_witnesses']==864 and gate['full_output_fraction_witnesses']==1512
    rows=gate['rows'];assert len(rows)==8 and {(v['order'],v['route']) for v in rows}==set((k,r) for k in range(1,5) for r in ('generic','dense'))
    for row in rows:
        assert all(row[k] is True for k in ('old_new_coeff_cache_strict_tails_bytes_equal','prefix_fields_bytes_equal',
            'instrumented_bare_bytes_equal','cached_overflow_bad_retained'))
    rows=gate['cuda_rows'];assert len(rows)==4 and {r['order'] for r in rows}==set(range(1,5))
    for row in rows:
        checks=row['valid_microtape_all_fields_equal']
        assert set(checks)=={'coeff','remainder','cache','strict_tails','bad'} and all(v is True for v in checks.values())
        assert all(row[k] is True for k in ('actual_valid_tape_dispatch','python_cuda_replay_cur_bad_bytes_equal','actual_replay_tape_dispatch'))
    return directory,build,gate


def source40(source,identity):
    meta=read(source/'INPUT.json');done=read(source/'RESULT.json')
    assert done['status']=='target_completed' and done['mode']=='experiment' and done['working_order']==3
    assert done['completed_step']==done['root_covered_to_step']==done['target_step']==40 and done['all_required_leaves']==2
    assert done['script_sha256']==sha(__file__) and done['input_sha256']==sha(source/'INPUT.json')
    assert meta['settings']==SETTINGS and meta['mode']=='split_from_zero_strict_endpoint'
    assert meta['root_id']==1 and meta['leaf_count']==2 and meta['original_root_count']==1024
    assert meta['controller_policy']=='union_box' and meta['requested_periods']==2 and meta['target_step']==40
    assert meta['working_total_degree']==meta['spatial_composition_order']==3 and meta['point_code_order']==2 and meta['validation_order']==4
    assert meta['sr_queue']==1000 and meta['end_to_end_strict_certificate'] is False
    assert all(meta[k]==v for k,v in identity.items())
    assert all(done[k]==identity[k] for k in BINDINGS)
    assert done['end_to_end_strict_certificate'] is False and 'original_all1024_common_prefix_unchanged' not in done
    return meta,done


def long_gate(args,identity):
    if args.periods<=2:return
    assert args.qualification and args.replay_check,'own reciprocal40 and own cold replay required'
    source40(args.qualification,identity);cold=read(args.replay_check/'RESULT.json')
    assert cold['status']=='passed' and cold['mode']=='cold_boundary_replay' and cold['working_order']==3
    assert cold['script_sha256']==sha(__file__)
    assert cold['source_input_sha256']==sha(args.qualification/'INPUT.json') and cold['source_result_sha256']==sha(args.qualification/'RESULT.json')
    assert all(cold[k]==identity[k] for k in BINDINGS)
    assert all(cold[k] is True for k in ('all_endpoint_bytes_match','all_controller_bytes_match','all_step21_bytes_match','all_transfer_receipt_bytes_match'))
    assert set(cold['transfer_receipt_comparisons'])==TRANSFER_FIELDS and all(v is True for v in cold['transfer_receipt_comparisons'].values())
    assert cold['step21_comparisons'] and all(v is True for v in cold['step21_comparisons'].values())
    assert cold['end_to_end_strict_certificate'] is False


def initialize(args):
    global t,candidate_handle
    directory,build,gate=candidate_gate(args)
    frozen=Path(__file__).parent.parent/'quad_control_transfer_20260927/run_quad_transfer.py'
    assert sha(frozen)==TRANSFER_SHA
    t=load('reciprocal_frozen_transfer',frozen);ctx=t.initialize(args)
    c,b,endpoint,edge,driver,model,cfg,identity,model_s=ctx
    assert not t.m.engines and t.m.settings(3)==SETTINGS,'install must precede any engine/graph/tape'
    assert build['common_sha256']==identity['common_script_sha256']
    assert Path(build['engine_root']).resolve()==c.E.resolve()
    assert build['baseline_extensions']==identity['extensions']
    assert gate['torch_version']==build['torch_version']==c.torch.__version__
    installer=load('reciprocal_candidate_installer',directory/'install.py')
    candidate_handle=installer.install(Path(gate['candidate']['cache_root']),allow_build=False,
        binary_paths={name:v['path'] for name,v in gate['extensions'].items()})
    assert candidate_handle.metadata==gate['candidate']
    assert candidate_handle.tape.available() and candidate_handle.tape.valid_available()
    assert candidate_handle.loads==gate['extensions']
    identity.update(script_sha256=sha(__file__),frozen_transfer_runner_sha256=TRANSFER_SHA,
        reciprocal_policy=POLICY,reciprocal_candidate=candidate_handle.metadata,reciprocal_extensions=dict(candidate_handle.loads),
        reciprocal_build_path=str(args.candidate_build),reciprocal_build_result_sha256=sha(args.candidate_build/'RESULT.json'),
        reciprocal_check_path=str(args.candidate_check),reciprocal_check_result_sha256=sha(args.candidate_check/'RESULT.json'),
        reciprocal_check_script_sha256=PINS['check_paths.py'],controller_input_domain_policy=DOMAIN,
        metadata_variant='on-demand graded-lex; pinned a3fb plus separately identified five-route reciprocal correction')
    # t.refresh, direct hull, endpoint, setup, observer, experiment and replay
    # remain untouched. Replace only admission and the old-SHA writer closure.
    t.m.r.long_gate=long_gate
    def write_metadata(path,value):
        if Path(path).name=='INPUT.json':
            assert value['script_sha256']==sha(__file__) and value['reciprocal_policy']==POLICY
            assert value['controller_input_domain_policy']==DOMAIN and value['control_transfer_policy']==t.POLICY
            value['limitation']=LIMITATION
            value['controller']+='; conditional strict transfer from t0; original direct NN input domain'
            value['timing_scope']+='; includes five-route candidate loads, transfer byte gates and receipts'
        write(path,value)
    t.m.r.write=write_metadata
    return c,b,endpoint,edge,driver,model,cfg,identity,model_s


def main(args):
    assert args.working_order==3 and 40<=args.max_steps<=1000 and args.max_steps%20==0
    if args.replay:assert args.max_steps==40,'this entry cold-replays its own completed40'
    args.periods=args.max_steps//20;args.check_local=False;args.p3_parity=None
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception');identity=None
    try:
        ctx=initialize(args);identity=ctx[7]
        if args.replay:
            source40(args.replay,identity)
            result=t.m.r.replay(args,out,ctx)
            result['transfer_receipt_comparisons']=t.compare_transfer(args.replay,out,ctx[0],ctx[1])
            assert set(result['transfer_receipt_comparisons'])==TRANSFER_FIELDS and all(result['transfer_receipt_comparisons'].values())
            result['all_transfer_receipt_bytes_match']=True
        else:result=t.m.r.experiment(args,out,ctx)
        result.pop('original_all1024_common_prefix_unchanged',None)
    except BaseException as e:result.update(status='exception',error_type=type(e).__name__,error=str(e));raise
    finally:
        result.update(script_sha256=sha(__file__),frozen_transfer_runner_sha256=TRANSFER_SHA,
            end_to_end_strict_certificate=False,limitation=LIMITATION,process_body_s=time.perf_counter()-start)
        if identity is not None:
            result.update({k:identity[k] for k in BINDINGS})
            result.update(injection_adapter_sha256=t.ADAPTER_SHA,injection_check_script_sha256=t.CHECK_SHA,
                control_transfer_policy=t.POLICY,frozen_metadata_runner_sha256=t.METADATA_RUNNER_SHA,
                metadata_backend_sha256=t.m.BACKEND_SHA,pair_budgets=[dict(e.pair_budget) for e in t.m.engines],
                max_cuda_allocated_bytes=t.m.c.torch.cuda.max_memory_allocated())
        write(out/'RESULT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--working-order',type=int,choices=[3],default=3)
    p.add_argument('--max-steps',type=int,default=40)
    for name in ('endpoint-check','metadata-check','injection-check','candidate-build','candidate-check','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('qualification','replay-check','replay'):p.add_argument('--'+name,type=Path)
    main(p.parse_args())
