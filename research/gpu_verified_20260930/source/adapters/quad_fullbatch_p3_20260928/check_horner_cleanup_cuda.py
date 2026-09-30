"""Actual two-compose Horner eager/capture/replay lifetime comparison.

No SR, NN, advance, new mathematical inputs or trajectory qualification. Each
arm owns fresh working tables and graphs. Same-key replay repeats actual case2;
different support keys are never described as changed-data same-key replay.
"""
from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,struct,sys,time,traceback

HERE=Path(__file__).parent
CAPTURE_SHA='f693ac64f93787389b40d755a3647359ba4e7e535c7e87f70d5264544efc7aea'
ADAPTER_SHA='f98958be75a46d49444d98b5c196ecef997ce6f63b70bfd2e541ab4e62ff0ad0'
CPU_CHECK_SHA='2d9c99f0962c16c15f6ae219e9b2998d924915e48ff5d418b3d8db2958735a8a'
CPU_RESULT_SHA='7ee8bbf63b341d4e83fbdd771eded2906cafe12791fd96e7c98c116b4ffffcb0'
CAPTURE_RESULT_SHA='c853691f7c9af50ce13558495d92e3d1a1d3d95a4aa18afcd95c019a1516e481'

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

source=load('horner_cleanup_actual_capture',HERE/'capture_compose_inputs.py',CAPTURE_SHA)
w=source.w

def signature(torch,value):
    if isinstance(value,torch.Tensor):
        value=value.detach().to('cpu',copy=True).contiguous()
        return dict(shape=list(value.shape),dtype=str(value.dtype),
            sha256=hashlib.sha256(memoryview(value.view(torch.uint8).numpy()).cast('B')).hexdigest())
    if isinstance(value,dict):return {k:signature(torch,v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [signature(torch,v) for v in value]
    return value

def exact(torch,a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(
            a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(torch,a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(torch,x,y) for x,y in zip(a,b))
    if isinstance(a,float):return struct.pack('<d',a)==struct.pack('<d',b)
    return a==b

def support_summary(torch,sp,sup):
    return dict(n=sup.n,k=sup.k,spatial=sup.spatial,ids=list(sup.ids),
        exps=torch.from_numpy(sp._exps_for(sup).copy()))

def cpu_gate(args):
    assert sha(HERE/'check_horner_cleanup_cpu.py')==CPU_CHECK_SHA
    path=args.cpu_check/'RESULT.json';assert sha(path)==CPU_RESULT_SHA
    r=read(path)
    assert r['status']=='passed' and r['device']=='cpu' and r['script_sha256']==CPU_CHECK_SHA
    assert r['adapter_sha256']==ADAPTER_SHA and r['original_source_sha256']==source.HORNER_SHA
    for field in ['inverse_AST_equal','original_candidate_coeff_remainder_support_all_bytes_equal','inputs_unchanged',
        'changed_input_output_changed','held_original_and_candidate_outputs_survive_second_call_and_gc',
        'exception_propagated_to_fixture','original_function_restored']:assert r[field] is True
    assert not any(r['candidate_weakrefs_alive_without_gc'].values())
    assert not any(r['exception_weakrefs_alive_after_handler'].values())
    return r

def payloads(args,torch,identity):
    assert len(CAPTURE_RESULT_SHA)==64,'Actual capture receipt is not yet pinned'
    root=args.capture;result=read(root/'RESULT.json');inp=read(root/'INPUT.json')
    assert sha(root/'RESULT.json')==CAPTURE_RESULT_SHA
    assert result['status']=='diagnostic_two_compose_pairs_captured'
    assert result['script_sha256']==CAPTURE_SHA and result['source_entry_sha256']==source.ENTRY_SHA
    assert result['input_sha256']==sha(root/'INPUT.json')
    assert result['source_identity']==inp['source_identity']
    sid=inp['source_identity'];assert sid['algorithm_identity']==identity
    assert sid['script_sha256']==CAPTURE_SHA and sid['diagnostic_entry_sha256']==source.ENTRY_SHA
    assert result['diagnostic_only'] is True and result['no_timing_or_execution_qualification'] is True
    assert result['completed_advance_count']==2 and result['accepted_lanes_by_step']==[1024,1024]
    assert [v['step'] for v in result['compose_pairs']]==[1,2]
    files={'INPUT.json':sha(root/'INPUT.json'),'RESULT.json':CAPTURE_RESULT_SHA};pairs=[]
    for step,record in enumerate(result['compose_pairs'],1):
        values=[]
        for kind in ['input','output']:
            name=f'COMPOSE_{kind.upper()}_{step}.pt';meta_name=name[:-3]+'.json'
            receipt=record[kind];assert receipt['file']==name
            assert sha(root/name)==receipt['pt_sha256'] and sha(root/meta_name)==receipt['sidecar_sha256']
            meta=read(root/meta_name);p=torch.load(root/name,map_location='cpu',weights_only=True)
            assert meta['source_identity']==sid and meta['input_sha256']==result['input_sha256']
            assert meta['pt_sha256']==receipt['pt_sha256'] and meta['pt_bytes']==(root/name).stat().st_size
            assert signature(torch,p)==meta['signature']
            files[name]=receipt['pt_sha256'];files[meta_name]=receipt['sidecar_sha256'];values.append(p)
        p,out=values
        assert p['schema']=='actual_horner_compose_v1' and out['schema']=='actual_horner_result_v1'
        assert p['step']==out['step']==step and out['input_pt_sha256']==record['input']['pt_sha256']
        assert p['source_entry_sha256']==source.ENTRY_SHA and p['source_script_sha256']==CAPTURE_SHA
        assert p['horner_sha256']==source.HORNER_SHA and p['input_sha256']==result['input_sha256']
        assert (p['n'],p['batch'],p['order'],p['h'],p['cutoff'])==(16,1024,3,.005,1e-6)
        assert p['strict'] is p['horner_edge_enabled'] is p['iv_use_kernels'] is True
        assert set(p['inputs'])=={'a_coeffs','a_rem','g_coeffs','g_rem'}
        for name,v in p['inputs'].items():
            assert v.device.type=='cpu' and v.dtype==torch.float64 and v.is_contiguous()
            assert v.ndim==3 and v.shape[:2]==(1024,16) and bool(torch.isfinite(v).all())
            if name.endswith('_rem'):assert v.shape[-1]==2 and bool((v[...,0]<=v[...,1]).all())
        assert out['coeff'].shape[:2]==out['remainder'].shape[:2]==(1024,16)
        assert out['remainder'].shape[-1]==2
        assert all(t.dtype==torch.float64 and bool(torch.isfinite(t).all()) for t in [out['coeff'],out['remainder']])
        assert bool((out['remainder'][...,0]<=out['remainder'][...,1]).all())
        pairs.append((p,out))
    return pairs,files

def arm(ctx,adapter,pairs,candidate):
    torch,d,se=ctx.torch,ctx.d,ctx.c.se
    from flowstar_gpu import support as sp,sparse_horner as sh,interval as iv
    original=sh.compose_horner;binding=adapter.install(sh) if candidate else None
    # The function is installed before this owner's engine or graph exists.
    try:
        tables=d.build_tables(16,3).to('cuda:0');eng=sp.SparseEngine(tables,d.poly.build_step_tables(tables,.005),'cuda:0')
        eng.horner_edge_kernel=ctx.edge.horner_edge;sched=d.build_schedule(16,3,'cuda:0')
        assert getattr(eng,'_graphs',None) is None and se.COMPOSITION_MODE=='horner' and iv.USE_KERNELS
        outputs=[];records=[];raw_prior=[];clones=[];keys=[]
        def cpu_output(out):
            return dict(coeff=out[0].detach().to('cpu',copy=True).contiguous(),
                remainder=out[1].detach().to('cpu',copy=True).contiguous(),support=support_summary(torch,sp,out[2]))
        def inputs(p):
            vals=[p['inputs'][name].to('cuda:0',copy=True) for name in ['a_coeffs','a_rem','g_coeffs','g_rem']]
            supports=[]
            for name in ['support_a','support_g']:
                meta=p[name];assert (meta['n'],meta['k'],meta['spatial'])==(16,3,True)
                s=sp.make_support(16,3,True,tuple(meta['ids']))
                assert exact(torch,support_summary(torch,sp,s),meta);supports.append(s)
            assert vals[0].shape[-1]==supports[0].size and vals[2].shape[-1]==supports[1].size
            return vals,supports
        for p,expected in pairs:
            vals,supports=inputs(p);versions=[v._version for v in vals]
            args=(*vals,*supports,p['order'],p['cutoff'],p['strict'],sched,eng)
            eager=sh.compose_horner(*args);torch.cuda.synchronize()
            expected={k:expected[k] for k in ['coeff','remainder','support']}
            assert exact(torch,cpu_output(eager),expected),'Eager differs from actual captured result'
            del eager
            out=se.compose_s(*args);torch.cuda.synchronize()
            actual=cpu_output(out);assert exact(torch,actual,expected),'Graph differs from actual captured result'
            cache=eng._graphs;assert cache.enabled
            key=('compose',se.COMPOSITION_MODE,id(eng.horner_edge_kernel),iv.USE_KERNELS,*supports,
                p['order'],p['cutoff'],p['strict'],1024,torch.float64)
            assert key in cache._segs and cache._segs[key][2] is out
            assert cache.captures==len(set(keys+[key])) and cache.hits==len(keys)+1
            for old_key,old_out,old_cpu in raw_prior:
                if old_key!=key:assert exact(torch,cpu_output(old_out),old_cpu),'New private graph capture changed another key output'
            for held,expected_cpu in clones:assert exact(torch,cpu_output(held),expected_cpu)
            held=(out[0].clone(),out[1].clone(),out[2])
            assert all(held[i].untyped_storage().data_ptr()!=out[i].untyped_storage().data_ptr() for i in [0,1])
            clones.append((held,actual));raw_prior.append((key,out,actual));keys.append(key)
            assert [v._version for v in vals]==versions
            assert all(exact(torch,v,p['inputs'][name]) for v,name in zip(vals,['a_coeffs','a_rem','g_coeffs','g_rem']))
            outputs.append(actual);records.append(dict(step=p['step'],eager_actual_bytes_equal=True,graph_actual_bytes_equal=True,
                inputs_bytes_and_versions_unchanged=True,graph_captures=cache.captures,graph_hits=cache.hits,
                output_signature=signature(torch,actual)))
            del vals,args,out
        assert len(set(keys))==2,'Cross-new-key lifetime qualification needs two distinct actual support keys'
        p,expected=pairs[-1];vals,supports=inputs(p);versions=[v._version for v in vals]
        args=(*vals,*supports,p['order'],p['cutoff'],p['strict'],sched,eng)
        again=se.compose_s(*args);torch.cuda.synchronize()
        assert cache.captures==2 and cache.hits==3 and exact(torch,cpu_output(again),outputs[-1])
        assert [v._version for v in vals]==versions
        assert all(exact(torch,v,p['inputs'][name]) for v,name in zip(vals,['a_coeffs','a_rem','g_coeffs','g_rem']))
        gc.collect();torch.cuda.synchronize()
        for held,expected_cpu in clones:assert exact(torch,cpu_output(held),expected_cpu)
        for key,old_out,old_cpu in raw_prior:assert exact(torch,cpu_output(old_out),old_cpu)
        return outputs,dict(candidate=candidate,cases=records,captures=cache.captures,hits=cache.hits,
            actual_case2_repeated_same_key=True,changed_data_same_key_test=False,
            cross_new_key_original_graph_outputs_preserved=True,owned_clones_survive_replay_and_gc=True,
            no_synthetic_inputs=True,adapter_installed_before_owner=bool(candidate),
            inverse_AST_equal=binding.inverse_AST_equal if binding else None,
            original_function_sha256=binding.original_function_sha256 if binding else None,
            candidate_function_sha256=binding.candidate_function_sha256 if binding else None)
    finally:
        if binding is not None:binding.restore()
        assert sh.compose_horner is original

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception')
    try:
        cpu=cpu_gate(args);assert len(CAPTURE_RESULT_SHA)==64
        ctx=w.initialize(args);torch=ctx.torch
        pairs,files=payloads(args,torch,ctx.identity)
        adapter=load('horner_cleanup_cuda_candidate',HERE/'horner_closure_cleanup.py',ADAPTER_SHA)
        left,lr=arm(ctx,adapter,pairs,False);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        right,rr=arm(ctx,adapter,pairs,True)
        assert exact(torch,left,right)
        for field in ['inverse_AST_equal','original_function_sha256','candidate_function_sha256']:assert rr[field]==cpu[field]
        for name,digest in files.items():assert sha(args.capture/name)==digest
        result=dict(status='passed',device='cuda:0',torch_version=torch.__version__,source_identity=ctx.identity,
            adapter_sha256=ADAPTER_SHA,source_capture_sha256=CAPTURE_SHA,capture_result_sha256=CAPTURE_RESULT_SHA,
            source_files_sha256=files,cpu_checker_sha256=CPU_CHECK_SHA,cpu_result_sha256=CPU_RESULT_SHA,
            arms=[lr,rr],input_rows=1024,n=16,working_order=3,actual_cases=2,
            all_coeff_remainder_support_bytes_equal=True,source_capture_passed=True,
            source_run_is_trajectory_qualification=False,horner_original_restored=True,
            max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            scope='Actual full1024 Horner cases1/2, direct eager and two private graphs plus actualcase2 repeated HIT. Original/candidate and original captured results agree. No changed-data same-key case, SR/NN/advance/40-step admission, memory-speed improvement or full trajectory proof.')
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        if 'ctx' in locals():
            ctx.working_prune.restore();ctx.working_prune_log.close();ctx.valid128.restore()
            ctx.postwarm.restore();ctx.postwarm_log.close();ctx.weighted128.restore();result['runtime_hooks_restored']=True
        result.update(script_sha256=sha(__file__),elapsed_s=time.perf_counter()-start)
        write(args.output/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','elapsed_s']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','cpu-check','common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check',
        'host-small-check','host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend',
        'metadata-check','hybrid-check','weighted-check','weighted-checker','cache-release-check','cache-release-checker',
        'valid-check','valid-checker','working-prune-reference','working-prune-check','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.mode='horner_cleanup_fixed_inputs';args.source40=None;main(args)
