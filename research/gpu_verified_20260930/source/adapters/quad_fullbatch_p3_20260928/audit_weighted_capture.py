"""CPU audit of a completed payload from an interrupted input-only capture.

This never turns the source capture into a successful run or trajectory gate.
It compiles the archived ODE only; no engine, table, extension or GPU is loaded.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,importlib.util,json,os,sys,time,traceback,zipfile

CAPTURE_SHA='91cb4b9fcf1a7531cf5a2d89eba4ff02018140b6f8b83e7ad52d5f84ea17d4b5'
ENTRY_SHA='22f4dda16dcfd30563e6652ec8af82034b3b138a7505dbf31714a8bd55820a72'
BASIS_SHA='5181161bf78c16273b6b5725e1e6a449d7d3617c59b05a9db7656fe3eb551cd3'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only process required'
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    source=a.source;inp=read(source/'INPUT.json');identity=inp['source_identity'];watch=read(a.watch)
    assert sha(a.capture_script)==CAPTURE_SHA and sha(a.entry_script)==ENTRY_SHA
    assert identity['script_sha256']==CAPTURE_SHA and identity['source_entry_sha256']==ENTRY_SHA
    base=identity['algorithm_identity'];assert base['script_sha256']==ENTRY_SHA
    assert identity['diagnostic_only'] is True and base['end_to_end_strict_certificate'] is False
    assert inp['stop']=='first_refine_accepted_entry_before_body'
    assert inp['torch_cap_bytes']==11*1024**3 and inp['external_guard_bytes']==int(11.5*1024**3)
    assert watch['status']=='gpu_guard_11.5GiB' and watch['returncode']!=0
    assert watch['peak_owned_gpu_bytes']>int(11.5*1024**3)
    assert not (source/'RESULT.json').exists() and not (source/'WEIGHTED_INPUT.json').exists()
    cfg=base['resolved_config'];assert hashlib.sha256(canonical(cfg).encode()).hexdigest()==base['resolved_config_sha256']
    assert cfg['ode_order']==3 and cfg['steps']==50
    assert base['config_overrides']=={'ode_order':{'original':2,'resolved':3}}
    assert (base['working_total_degree'],base['point_code_order'],base['validation_order'])==(3,2,4)
    source_hashes={}
    for key in ['config','model','driver','boxes']:
        path=Path(base[key+'_path']);assert sha(path)==base[key+'_sha256'];source_hashes[key]=sha(path)
    root=Path(base['engine']['root'])
    assert base['engine']['head']=='a3fb2e94ba976aaf498c4a9cb3f98165cddcc272'
    for name,digest in base['engine']['python_sha256'].items():assert sha(root/name)==digest,name
    # Compile only: unchanged tape/cache layout, not execution of old reciprocal.
    sys.path.insert(0,str(root/'src'))
    from flowstar_gpu.ode_compiler import compile_ode
    compiled=dataclasses.asdict(compile_ode(cfg['dynamics_expressions'],[v['name'] for v in cfg['initial_set']],order=4))
    assert sha(a.basis)==BASIS_SHA;basis=load('capture_audit_basis',a.basis).GradedLexBasis(17,8)
    path=source/'WEIGHTED_INPUT.pt';before=sha(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert any(v.endswith('/data.pkl') for v in archive.namelist())
        zip_entries=len(archive.infolist())
    p=torch.load(path,map_location='cpu',weights_only=True)
    scalar=dict(n=16,batch=1024,working_order=3,point_code_order=2,validation_order=4,support_family=4,
        h=.005,cutoff=1e-6,rounds=2,chunk_size=512,use_graph=True,trace_present=False,attempted_step=1,held_controller_step=0)
    tensor_names=['x','initial','original','current','eligible','support_exponents','initial_exponents']
    assert set(p)==set(scalar)|set(tensor_names)|{'support_ids','initial_ids','variable_support_ids','variable_exponents','code','code_sha256'}
    for key,value in scalar.items():assert type(p[key]) is type(value) and p[key]==value,key
    assert canonical(p['code'])==canonical(compiled)
    assert p['code_sha256']==hashlib.sha256(canonical(compiled).encode()).hexdigest()
    tensors={};support_report={}
    def tensor(name,t,dtype,shape):
        assert isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==dtype and tuple(t.shape)==tuple(shape),name
        assert t.is_contiguous() and bool(torch.isfinite(t).all()),name
        digest=hashlib.sha256(memoryview(t.view(torch.uint8).numpy()).cast('B')).hexdigest()
        tensors[name]=dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size(),sha256=digest)
    def support(name,ids,exps):
        assert isinstance(ids,list) and ids and all(type(i) is int for i in ids)
        assert ids==sorted(set(ids));tensor(name,exps,torch.int64,(len(ids),17))
        rows=exps.tolist();assert all(basis.unrank(i)==tuple(e) and basis.rank(e)==i for i,e in zip(ids,rows))
        assert all(sum(e)<=3 for e in rows),'Working P3 slots must be retained without higher-degree fabrication'
        support_report[name]=dict(size=len(ids),max_degree=max(map(sum,rows)))
    support('support_exponents',p['support_ids'],p['support_exponents'])
    support('initial_exponents',p['initial_ids'],p['initial_exponents'])
    assert len(p['variable_support_ids'])==len(p['variable_exponents'])==16
    union=set(p['support_ids'])
    for j,(ids,exps) in enumerate(zip(p['variable_support_ids'],p['variable_exponents'])):
        support('variable_exponents_'+str(j),ids,exps);assert set(ids)<=union
        missing=sorted(union-set(ids));positions=[p['support_ids'].index(i) for i in missing]
        assert not bool(torch.count_nonzero(p['x'][:,j,positions]))
    tensor('x',p['x'],torch.float64,(1024,16,len(p['support_ids'])))
    tensor('initial',p['initial'],torch.float64,(1024,16,len(p['initial_ids'])))
    tensor('original',p['original'],torch.float64,(1024,16,2))
    tensor('current',p['current'],torch.float64,(1024,16,2));tensor('eligible',p['eligible'],torch.bool,(1024,))
    for name in ['original','current']:assert bool((p[name][...,0]<=p[name][...,1]).all()),name
    assert bool((p['original'][...,0]==-.1).all()) and bool((p['original'][...,1]==.1).all())
    selected=p['current'][p['eligible']];assert len(selected)>0
    assert bool(((selected[...,0]>=-.1)&(selected[...,1]<=.1)&(selected[...,0]<=0)&(selected[...,1]>=0)).all())
    assert not torch.cuda.is_initialized();assert sha(path)==before
    return dict(status='passed',source_capture_status=watch['status'],source_capture_passed=False,
        classification='complete_payload_from_resource_terminated_capture',source_result_missing=True,source_sidecar_missing=True,
        suitable_for_fixed_input_comparison=True,suitable_for_trajectory_admission=False,
        source_pt_sha256=before,source_pt_size_bytes=path.stat().st_size,source_input_sha256=sha(source/'INPUT.json'),
        source_watch_sha256=sha(a.watch),source_command_sha256=sha(a.command),source_capture_sha256=CAPTURE_SHA,
        source_entry_sha256=ENTRY_SHA,source_identity=identity,actual_source_hashes=source_hashes,
        source_engine_python_sha256=base['engine']['python_sha256'],code_sha256=p['code_sha256'],code_recompiled_bytes_equal=True,
        zip_crc_passed=True,zip_entries=zip_entries,all_tensor_payloads_cpu_finite=True,supports=support_report,tensors=tensors,
        eligible_lanes=int(p['eligible'].sum()),cap_all_binary64_minus_plus_point1=True,cuda_initialized=False,
        watch=watch,limitations=['No source RESULT/sidecar or completed advance; source resource failure remains unchanged.',
        'No independent reconstruction of upstream Picard/NN/SR values; only intact archived inputs for new local comparisons.',
        'Candidate SO and mathematical qualification are inherited provenance, not re-executed by this CPU artifact audit.'])

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='failed')
    try:result=audit(a)
    except BaseException as e:result.update(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        result.update(audit_script_sha256=sha(__file__),audit_process_s=time.perf_counter()-start,device='cpu')
        (a.output/'AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['source','watch','command','capture-script','entry-script','basis','output']:parser.add_argument('--'+key,type=Path,required=True)
    main(parser.parse_args())
