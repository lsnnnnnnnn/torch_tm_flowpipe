"""Two actual observer inputs: original full1024 vs endpoint-only lane64.

Both results must also equal the captured and archived b89 endpoints.
Source capture was resource-terminated after both payloads/outputs were saved; No SR,
NN, advance, synthetic input or trajectory qualification is performed here.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,sys,time,traceback,zipfile

HERE=Path(__file__).parent
CAPTURE_SHA='0d4062b47996a4e5ae814c92f3274f6b441109969138b3ebfbc27b7edbd2749e'
CAPTURE_INPUT_SHA='d26ffe06640b4aa688198c72d30e429fae6d49155b56cd0a9c8a1d259429e742'
CAPTURE_PT_SHA={1:'3c0b65f1e9afc54f4c6ab1ad7a31e02c0a8c37dd4f34669c083f1984e29e289d',
    2:'5ac5e7f07bb9688e30fc11ff737b58af39da29a78f0f3475cc622db3d7fe46e0'}
WATCH_SHA='b58e90e8edc5c6208ed1fde8bfe70a59393491d272be0db5fd4c8197135fdf39'
REFERENCE_SHA={
    'INPUT.json':'5f9de4dd247a596a52f27045279b74fa874d68d721d50508506dcf8985fffaeb',
    'observer_1.pt':'b7423f8ffccaf206ca069d2d27791111eec822dff1a4035a888f9cfc13564cb8',
    'observer_1.json':'8f7ef1cef0b7f2b9c6405496dfd13744f79bcaa9e25553a5dcb449af85530d2a',
    'observer_2.pt':'f46170dbaa8b9495bc08ca462e3c92d3067a480f97ebb9c376cfe57948b90626',
    'observer_2.json':'06000fe764c917cac39431832b28282813d3362de260ef453bdc34e42f2f8ede',
}
ADAPTER_SHA='d8d87a51a7db600685a79ff19193a56986dab6e84b4825a30570caaf157c8872'

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
source=load('observer_endpoint_actual_capture',HERE/'capture_observer_inputs.py',CAPTURE_SHA)
w=source.w

def signature(torch,value):
    if isinstance(value,torch.Tensor):
        value=value.detach().to('cpu',copy=True).contiguous()
        return dict(shape=list(value.shape),dtype=str(value.dtype),
            sha256=hashlib.sha256(memoryview(value.view(torch.uint8).numpy()).cast('B')).hexdigest())
    if isinstance(value,dict):return {k:signature(torch,v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [signature(torch,v) for v in value]
    return value

def equal(torch,a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(
        a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))

def load_pt(torch,path):
    with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
    return torch.load(path,map_location='cpu',weights_only=True)

def payloads(args,torch,identity):
    assert len(WATCH_SHA)==64,'Actual resource-failure watch receipt is not yet frozen'
    root=args.capture;inp=read(root/'INPUT.json');watch=read(args.capture_watch)
    assert sha(root/'INPUT.json')==CAPTURE_INPUT_SHA and sha(args.capture_watch)==WATCH_SHA
    assert watch['status']=='gpu_guard_11.5GiB' and watch['returncode']==-15
    assert watch['peak_owned_gpu_bytes']>int(11.5*1024**3)
    # The source remains failed. Do not create a missing RESULT or pair2.
    assert not (root/'RESULT.json').exists() and not (root/'OBSERVER_PAIR_2.json').exists()
    assert not (root/'observer_2.json').exists()
    sid=inp['source_identity'];assert sid['algorithm_identity']==identity
    assert sid['script_sha256']==CAPTURE_SHA and sid['diagnostic_entry_sha256']==source.ENTRY_SHA
    assert sid['diagnostic_only'] is True
    reference=args.reference_observers
    for name,digest in REFERENCE_SHA.items():assert sha(reference/name)==digest
    assert read(reference/'INPUT.json')['source_identity']==identity
    files={'INPUT.json':CAPTURE_INPUT_SHA};pairs=[]
    for step in [1,2]:
        name=f'OBSERVER_INPUT_{step}.pt';side=name[:-3]+'.json'
        assert sha(root/name)==CAPTURE_PT_SHA[step]
        p=load_pt(torch,root/name);meta=read(root/side)
        assert meta['source_identity']==sid and meta['step']==step and meta['input_sha256']==CAPTURE_INPUT_SHA
        assert meta['pt_sha256']==CAPTURE_PT_SHA[step] and meta['pt_bytes']==(root/name).stat().st_size
        assert meta['original_observer_executed_before_this_file'] is False
        assert signature(torch,p)==meta['signature']
        files[name]=CAPTURE_PT_SHA[step];files[side]=sha(root/side)
        output=f'observer_{step}.pt';out=load_pt(torch,root/output)
        assert sha(root/output)==REFERENCE_SHA[output], 'Captured output differs from frozen b89 observer'
        expected=load_pt(torch,reference/output);ref_meta=read(reference/f'observer_{step}.json')
        assert ref_meta['source_identity']==identity and ref_meta['step']==step
        assert ref_meta['pt_sha256']==REFERENCE_SHA[output]
        assert signature(torch,out)==signature(torch,expected)==ref_meta['signature']
        files[output]=REFERENCE_SHA[output]
        assert p['schema']=='actual_fullbatch_observer_input_v1' and p['step']==step
        assert p['source_entry_sha256']==source.ENTRY_SHA and p['source_script_sha256']==CAPTURE_SHA
        assert p['input_sha256']==CAPTURE_INPUT_SHA
        assert (p['batch'],p['physical_rows'],p['n'],p['working_order'],p['h'])==(1024,12,16,3,.005)
        for field in ['config_sha256','model_sha256','driver_sha256']:assert p[field]==identity[field]
        assert p['engine_python_sha256']==identity['engine']['python_sha256']
        assert p['pre'].ndim==3 and p['pre'].shape[:2]==(1024,12) and p['pre_rem'].shape==(1024,12,2)
        assert p['piece'].shape==(1024,2) and torch.equal(p['piece'],torch.full((1024,2),.005,dtype=torch.float64,device='cpu'))
        for value in [p['pre'],p['pre_rem'],p['piece'],out['bounds']]:
            assert value.device.type=='cpu' and value.dtype==torch.float64 and value.is_contiguous()
            assert bool(torch.isfinite(value).all())
        assert bool((p['pre_rem'][...,0]<=p['pre_rem'][...,1]).all())
        assert out['bounds'].shape==(1024,12,4)
        assert bool((out['bounds'][...,0]<=out['bounds'][...,1]).all() and (out['bounds'][...,2]<=out['bounds'][...,3]).all())
        assert p['accepted'].dtype==torch.bool and p['accepted'].shape==(1024,) and bool(p['accepted'].all())
        assert p['status'].dtype==torch.int8 and p['status'].shape==(1024,) and bool((p['status']==0).all())
        assert equal(torch,p['accepted'],out['accepted']) and equal(torch,p['status'],out['status'])
        pairs.append((p,out))
    assert any(not equal(torch,pairs[0][0][key],pairs[1][0][key]) for key in ['pre','pre_rem']), 'Actual inputs must change between cases'
    return pairs,files

def experiment(ctx,adapter,pairs):
    torch,d=ctx.torch,ctx.d
    from flowstar_gpu import support as sp,interval as iv
    raw=d.rows_range_over_time_sparse;source_hashes=adapter.verify_sources(raw)
    assert iv.USE_KERNELS and not ctx.hybrid.receipts
    tables=d.build_tables(16,3).to('cuda:0');step=d.poly.build_step_tables(tables,.005)
    eng=sp.SparseEngine(tables,step,'cuda:0')
    assert (tables.n,tables.k,tables.T)==(16,3,1140)
    table_signature=signature(torch,dict(t_deg=tables.t_deg[:1140],cat_iv=tables.cat_iv[:1140]))
    rows=[];held=[]
    for p,out in pairs:
        assert p['table_identity']==dict(n=tables.n,k=tables.k,T=tables.T,T2=tables.T2,
            step_delta=step.delta,step_lanes=step.lanes,iv_use_kernels=True,class_name=type(tables).__name__)
        meta=p['pre_support'];assert (meta['n'],meta['k'],meta['spatial'])==(16,3,False)
        sup=sp.make_support(16,3,False,tuple(meta['ids']))
        assert list(sup.ids)==meta['ids'] and equal(torch,torch.from_numpy(sp._exps_for(sup).copy()),meta['exps'])
        assert sup.size==p['pre'].shape[-1]
        st=SimpleNamespace(pre=p['pre'].to('cuda:0',copy=True),pre_rem=p['pre_rem'].to('cuda:0',copy=True),pre_sup=sup)
        piece=p['piece'].to('cuda:0',copy=True);inputs=(st.pre,st.pre_rem,piece)
        versions=[v._version for v in inputs];before=[signature(torch,v) for v in inputs]
        assert iv._kern(st.pre,piece)
        full=raw(st,eng,piece,12);candidate=adapter.endpoint64(st,eng,piece,12,raw=raw)
        torch.cuda.synchronize()
        assert equal(torch,full,candidate) and equal(torch,full,out['bounds'][...,2:4])
        assert bool(torch.isfinite(candidate).all() and (candidate[...,0]<=candidate[...,1]).all())
        assert [v._version for v in inputs]==versions and [signature(torch,v) for v in inputs]==before
        assert full.untyped_storage().data_ptr()!=candidate.untyped_storage().data_ptr()
        for old,expected in held:assert equal(torch,old,expected),'Previous public output changed after next call'
        held.extend([(full,out['bounds'][...,2:4]),(candidate,out['bounds'][...,2:4])])
        rows.append(dict(step=p['step'],full_batch=1024,chunk_size=64,block_calls=16,
            all_endpoint_bytes_equal=True,captured_original_endpoint_bytes_equal=True,
            inputs_bytes_and_versions_unchanged=True,finite_ordered=True,
            output_signature=signature(torch,candidate)))
        del st,piece,inputs,full,candidate
    assert d.rows_range_over_time_sparse is raw
    assert signature(torch,dict(t_deg=tables.t_deg[:1140],cat_iv=tables.cat_iv[:1140]))==table_signature
    assert getattr(eng,'_graphs',None) is None,'Observer range must not build CUDA graphs'
    return rows,source_hashes,table_signature

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception')
    try:
        assert len(WATCH_SHA)==64,'Actual resource-failure receipt has not been admitted'
        adapter=load('observer_endpoint64_candidate',HERE/'observer_endpoint64.py',ADAPTER_SHA)
        ctx=w.initialize(args);pairs,files=payloads(args,ctx.torch,ctx.identity)
        rows,source_hashes,table_signature=experiment(ctx,adapter,pairs)
        for name,digest in files.items():assert sha(args.capture/name)==digest
        result=dict(status='passed',device='cuda:0',torch_version=ctx.torch.__version__,source_identity=ctx.identity,
            adapter_sha256=ADAPTER_SHA,policy=adapter.POLICY,source_capture_sha256=CAPTURE_SHA,
            capture_input_sha256=CAPTURE_INPUT_SHA,capture_watch_sha256=WATCH_SHA,source_files_sha256=files,
            reference_files_sha256=REFERENCE_SHA,original_sources_sha256=source_hashes,
            table_signature=table_signature,actual_cases=2,input_rows=1024,physical_rows=12,n=16,working_order=3,
            basis_slots=1140,chunk_size=64,rows=rows,all_endpoint_bytes_equal=True,
            all_captured_original_endpoint_bytes_equal=True,previous_owned_outputs_survive_next_call=True,
            actual_input_values_changed=True,original_function_unchanged=True,table_bytes_unchanged=True,no_graph_created=True,
            source_capture_passed=False,source_capture_status='gpu_guard_11.5GiB',source_capture_result_present=False,
            source_run_is_trajectory_qualification=False,
            max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved(),
            scope='Only two actual observer endpoint inputs, full1024 versus lane64 with original1140-slot directed CUDA accumulation and captured endpoint bytes. No NN/SR/advance, full40/cold, memory-guard relief or trajectory qualification.')
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        if 'ctx' in locals():
            ctx.horner_cleanup.restore();ctx.working_prune.restore();ctx.working_prune_log.close();ctx.valid128.restore()
            ctx.postwarm.restore();ctx.postwarm_log.close();ctx.weighted128.restore();result['runtime_hooks_restored']=True
        result.update(script_sha256=sha(__file__),elapsed_s=time.perf_counter()-start)
        write(args.output/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','elapsed_s']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','capture-watch','reference-observers','common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check',
        'host-small-check','host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend',
        'metadata-check','hybrid-check','weighted-check','weighted-checker','cache-release-check','cache-release-checker',
        'valid-check','valid-checker','working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.mode='observer_endpoint64_fixed_inputs';args.source40=None;main(args)
