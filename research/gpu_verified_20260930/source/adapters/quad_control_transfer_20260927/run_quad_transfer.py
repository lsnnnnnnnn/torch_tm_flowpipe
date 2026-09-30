"""P3 QUAD root1/B2 from zero with conditional strict controller transfer.

Reuse frozen metadata/plant/NN/endpoint loops. Only replace control injection.
Forty steps plus a new cold20->NN->21 gate are required before a longer run.
This does not qualify the floating-point CROWN certificate itself.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time

METADATA_RUNNER_SHA='a6e06bf73aa88e182f61c0e9666c30e2bc2cadbef8f4c1f0dc9ea5363a4043fe'
ADAPTER_SHA='a79b19ad1bd97ae87b40f81239352a8b9859621757bd044df6fd615630c64310'
CHECK_SHA='ff35186cf01de2f63fbd5f9d344d743d86021dd9f135918daa0381f5738c5e4a'
POLICY='given valid T,L,U: original RN point plus outward coefficient-error, T*R and bias-minus-actual-center charges'
LIMITATION='Conditional transfer of a supplied valid affine certificate only. CROWN floating-point certificate and same-slope validity remain unqualified; not an end-to-end strict NNCS certificate.'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);v=importlib.util.module_from_spec(spec)
    sys.modules[name]=v;spec.loader.exec_module(v);return v


def long_gate(args,identity):
    if args.periods<=2:return
    assert args.qualification and args.replay_check,'new transfer40 and its own cold replay are required'
    meta=read(args.qualification/'INPUT.json');done=read(args.qualification/'RESULT.json');cold=read(args.replay_check/'RESULT.json')
    assert done['status']=='target_completed' and done['completed_step']==40 and done['input_sha256']==sha(args.qualification/'INPUT.json')
    assert done['script_sha256']==sha(__file__) and done['injection_adapter_sha256']==ADAPTER_SHA
    assert meta['settings']==m.settings(3) and meta['mode']=='split_from_zero_strict_endpoint'
    assert meta['root_id']==1 and meta['leaf_count']==2 and meta['controller_policy']=='union_box'
    assert all(meta[k]==v for k,v in identity.items())
    assert cold['status']=='passed' and cold['mode']=='cold_boundary_replay' and cold['working_order']==3
    assert cold['script_sha256']==sha(__file__) and cold['source_input_sha256']==sha(args.qualification/'INPUT.json')
    assert cold['source_result_sha256']==sha(args.qualification/'RESULT.json')
    assert cold['injection_adapter_sha256']==ADAPTER_SHA
    assert all(cold[k] for k in ['all_endpoint_bytes_match','all_controller_bytes_match','all_step21_bytes_match','all_transfer_receipt_bytes_match'])


def refresh(c,b,state,sr,eng,driver,model,cfg,completed,out):
    """Bind the current engine for exactly one injection, preserving old refresh."""
    original_inject=driver.inject_controls_s;receipts=[]
    def inject(st,T,L,U,u_ids,nn_in):
        assert st is state and tuple(u_ids)==(13,14,15) and nn_in==12 and not receipts
        # This repeats the pinned original point construction only as a byte gate;
        # no old RN radius is used as the new uncertainty certificate.
        point=c.torch.einsum('bmi,bit->bmt',T,st.pre[:,:12]);point[...,0]+=(U+L)*.5
        value=adapter.inject_controls_s(st,T,L,U,u_ids,nn_in,eng=eng)
        assert b.byte_equal(st.pre[:,13:16],point)
        receipts.append(value)
    driver.inject_controls_s=inject
    try:held=original_refresh(c,b,state,sr,eng,driver,model,cfg,completed,out)
    finally:driver.inject_controls_s=original_inject
    assert len(receipts)==1
    path=out/f'transfer_at_{completed}.pt'
    c.torch.save(dict(completed_step=completed,working_order=eng.tables.k,pre_ids=state.pre_sup.ids,
        receipt={k:v.detach().cpu() for k,v in receipts[0].items()}),path)
    audit=read(out/f'injection_at_{completed}.json')
    assert audit['physical_clock_rows_and_live_sr_unchanged'] and audit['controller_sha256']==held['sha256']
    record=dict(completed_step=completed,adapter_sha256=ADAPTER_SHA,policy=POLICY,
        receipt_path=str(path),receipt_sha256=sha(path),controller_sha256=held['sha256'],
        original_same_device_point_bytes_equal=True,physical_clock_support_tmv_status_live_sr_bytes_unchanged=True,
        injection_audit_sha256=sha(out/f'injection_at_{completed}.json'),end_to_end_strict_certificate=False)
    write(path.with_suffix('.json'),record)
    held['conditional_transfer']=record
    return held


def compare_transfer(source,out,c,b):
    checks={};old=source/'transfer_at_20.pt';new=out/'transfer_at_20.pt'
    for label,directory,path in [('source',source,old),('replay',out,new)]:
        side=read(path.with_suffix('.json'))
        assert side['adapter_sha256']==ADAPTER_SHA and side['policy']==POLICY and side['receipt_sha256']==sha(path)
        assert side['controller_sha256']==sha(directory/'controller_at_20.pt')
        assert side['injection_audit_sha256']==sha(directory/'injection_at_20.json')
        assert side['original_same_device_point_bytes_equal'] and side['physical_clock_support_tmv_status_live_sr_bytes_unchanged']
    actual=c.torch.load(new,map_location='cpu',weights_only=True);saved=c.torch.load(old,map_location='cpu',weights_only=True)
    assert set(actual)==set(saved) and set(actual['receipt'])==set(saved['receipt'])
    for k in ['completed_step','working_order','pre_ids']:checks[k]=actual[k]==saved[k]
    for k in actual['receipt']:checks[k]=b.byte_equal(actual['receipt'][k],saved['receipt'][k])
    write(out/'TRANSFER_CHECK.json',checks);assert all(checks.values()),'cold transfer receipt differs'
    return checks


def initialize(args):
    global m,adapter,original_refresh
    here=Path(__file__).parent;source=here.parent/'quad_sparse_metadata_20260927/run_quad_metadata.py'
    assert sha(source)==METADATA_RUNNER_SHA and sha(here/'strict_injection.py')==ADAPTER_SHA and sha(here/'check_injection.py')==CHECK_SHA
    # Check the adapter gate before importing remote-only runner/NN initialization.
    gate=read(args.injection_check/'RESULT.json')
    assert gate['status']=='passed' and gate['device']=='cuda' and gate['adapter_sha256']==ADAPTER_SHA and gate['script_sha256']==CHECK_SHA
    assert gate['exact_coefficient_checks']==72 and gate['exact_point_component_checks']==162 and gate['analytic_composed_checks']==60
    assert len(gate['negative_fail_before_mutation'])==8 and gate['deliberately_omitted_control_remainder_misses']>0
    assert all(gate[k] for k in ['original_same_device_point_bytes_equal','sr_boundary_unchanged','old_J_bytes_unchanged',
        'fresh_compose_received_exact_remainder','exactly_one_normal_J_append','instrumented_bare_full_state_and_live_SR_bytes_equal'])
    m=load('transfer_frozen_metadata_runner',source);ctx=m.initialize(args)
    c,b,endpoint,edge,driver,model,cfg,identity,model_s=ctx
    assert gate['engine_python_sha256']==identity['engine']['python_sha256'] and gate['extensions']==identity['extensions']
    assert gate['backend_sha256']==m.BACKEND_SHA and gate['driver_sha256']==identity['driver_sha256'] and gate['torch_version']==c.torch.__version__
    assert all(identity['environment'].get(k)==v for k,v in gate['environment'].items())
    assert {row['order'] for row in gate['actual_boundaries']}=={3,4}
    for row in gate['actual_boundaries']:
        assert row['same_device_original_point_bytes_equal'] and row['archived_point_bytes_equal']
        assert row['source_identity']['script_sha256']==METADATA_RUNNER_SHA
        assert all(row['source_identity'][k]==identity[k] for k in ['config_sha256','model_sha256','boxes_sha256'])
    adapter=load('transfer_strict_injection',here/'strict_injection.py')
    identity.update(script_sha256=sha(__file__),frozen_metadata_runner_sha256=METADATA_RUNNER_SHA,
        injection_adapter_sha256=ADAPTER_SHA,injection_check_script_sha256=CHECK_SHA,
        injection_check_path=str(args.injection_check),injection_check_result_sha256=sha(args.injection_check/'RESULT.json'),
        control_transfer_policy=POLICY,control_transfer_start_step=0,controller_certificate_qualified=False)
    original_refresh=m.r.refresh;m.r.refresh=refresh;m.r.long_gate=long_gate
    original_write=m.r.write
    def write_metadata(path,value):
        if Path(path).name=='INPUT.json':
            assert value['script_sha256']==sha(__file__) and value['control_transfer_policy']==POLICY
            value['limitation']=LIMITATION
            value['controller']+='; conditional strict transfer from t0'
            value['timing_scope']+='; includes transfer byte gates and receipts'
        original_write(path,value)
    m.r.write=write_metadata
    return c,b,endpoint,edge,driver,model,cfg,identity,model_s


def main(args):
    assert args.working_order==3 and 40<=args.max_steps<=1000 and args.max_steps%20==0
    args.periods=args.max_steps//20;args.check_local=False;args.p3_parity=None
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();result=dict(status='exception')
    try:
        ctx=initialize(args)
        if args.replay:
            result=m.r.replay(args,out,ctx)
            result['transfer_receipt_comparisons']=compare_transfer(args.replay,out,ctx[0],ctx[1])
            result['all_transfer_receipt_bytes_match']=True
        else:result=m.r.experiment(args,out,ctx)
        # That inherited constant describes an earlier historical comparison,
        # not this newly charged trajectory or all 1024 original roots.
        result.pop('original_all1024_common_prefix_unchanged',None)
    except BaseException as exc:result.update(status='exception',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result.update(script_sha256=sha(__file__),frozen_metadata_runner_sha256=METADATA_RUNNER_SHA,
            injection_adapter_sha256=ADAPTER_SHA,injection_check_script_sha256=CHECK_SHA,control_transfer_policy=POLICY,
            end_to_end_strict_certificate=False,limitation=LIMITATION,process_body_s=time.perf_counter()-started)
        if 'm' in globals():
            result['pair_budgets']=[dict(e.pair_budget) for e in m.engines]
            result['metadata_backend_sha256']=m.BACKEND_SHA
            if hasattr(m,'c'):result['max_cuda_allocated_bytes']=m.c.torch.cuda.max_memory_allocated()
        write(out/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--working-order',type=int,choices=[3],default=3)
    p.add_argument('--max-steps',type=int,default=40);p.add_argument('--endpoint-check',type=Path,required=True)
    p.add_argument('--metadata-check',type=Path,required=True);p.add_argument('--injection-check',type=Path,required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--replay-check',type=Path)
    p.add_argument('--replay',type=Path);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
