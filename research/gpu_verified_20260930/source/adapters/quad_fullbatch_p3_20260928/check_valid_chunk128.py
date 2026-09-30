"""Real consecutive ordinary-validation inputs: full1024 versus fixed128.

Two fresh owners each replay the saved first and second inputs, then a marked
bad-mask case. All five outputs are compared bytewise. No SR/NN/advance.
"""
from pathlib import Path
import argparse,dataclasses,gc,hashlib,importlib.util,json,struct,sys,time,traceback

HERE=Path(__file__).parent
CAPTURE_SHA='43fbb6e07a20040a0292150af74dc3403a2b11c4fe2e0a854b74c6aadfcad80d'
AUDIT_SHA='a1c8a2dd8ed3fcb477c1be6771ae1a9c0d31a481cdffc82e815b85e509134422'
ADAPTER_SHA='6b97a9ed19b8433b79be15f97381241bc9a79c071b90fa4a81f12c4d216dcc40'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(name,p,digest):
    assert sha(p)==digest
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
source=load('valid128_frozen_capture',HERE/'capture_valid_inputs.py',CAPTURE_SHA)
w=source.w
def same(torch,a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))
def exact(torch,a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and same(torch,a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(torch,a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(torch,x,y) for x,y in zip(a,b))
    if isinstance(a,float):return struct.pack('<d',a)==struct.pack('<d',b)
    return a==b
def fingerprints(torch,v,name='payload'):
    if isinstance(v,torch.Tensor):
        assert v.device.type=='cpu' and v.is_contiguous()
        return {name:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size(),
            sha256=hashlib.sha256(memoryview(v.view(torch.uint8).numpy()).cast('B')).hexdigest())}
    result={}
    items=v.items() if isinstance(v,dict) else enumerate(v) if isinstance(v,(list,tuple)) else []
    for k,x in items:result.update(fingerprints(torch,x,name+'/'+str(k)))
    return result

def payloads(a,torch):
    assert len(AUDIT_SHA)==64 and sha(a.capture_audit/'AUDIT.json')==AUDIT_SHA
    audit=read(a.capture_audit/'AUDIT.json');inp=read(a.capture/'INPUT.json')
    assert audit['status']=='passed' and audit['classification']=='complete_payloads_from_resource_terminated_capture'
    assert audit['source_capture_passed'] is False and audit['suitable_for_fixed_input_comparison'] is True
    assert audit['suitable_for_trajectory_admission'] is False and audit['source_result_missing'] is True
    assert audit['source_second_sidecar_missing'] is True
    assert not (a.capture/'RESULT.json').exists() and not (a.capture/'VALID_INPUT_2.json').exists()
    assert audit['capture_script_sha256']==CAPTURE_SHA and audit['source_entry_sha256']==source.ENTRY_SHA
    assert audit['source_identity']==inp['source_identity']
    assert audit['source_input_sha256']==sha(a.capture/'INPUT.json')
    files=audit['files_sha256']
    for name,digest in files.items():assert sha(a.capture/name)==digest
    assert len(audit['payloads'])==2
    result=[]
    for i,meta in enumerate(audit['payloads'],1):
        pt=a.capture/f'VALID_INPUT_{i}.pt'
        assert meta['attempted_step']==i and sha(pt)==meta['pt_sha256'] and pt.stat().st_size==meta['pt_size_bytes']
        p=torch.load(pt,map_location='cpu',weights_only=True)
        assert fingerprints(torch,p)==meta['tensors']
        assert p['source_input_sha256']==audit['source_input_sha256'] and p['source_script_sha256']==CAPTURE_SHA
        assert (p['n'],p['batch'],p['working_order'],p['point_code_order'],p['validation_order'])==(16,1024,3,2,4)
        assert p['attempted_step']==i and p['h']==.005 and p['cutoff']==1e-6
        assert p['x'].shape==(1024,16,72) and p['bad'].shape==(1024,)
        assert bool(torch.isfinite(p['x']).all()) and same(torch,p['x_rem'],p['cap'])
        assert bool((p['cap'][...,0]==-.1).all() and (p['cap'][...,1]==.1).all())
        assert not bool(p['bad'].any()) and p['code_sha256']==source.canonical_sha(p['code'])
        if i==1:
            first=read(a.capture/'VALID_INPUT_1.json')
            assert first['tensors']==meta['tensors'] and first['pt_sha256']==meta['pt_sha256']
            assert first['original_dispatch_executed'] is True and first['input_versions_unchanged'] is True
        result.append(p)
    assert exact(torch,result[0]['specialization'],result[1]['specialization'])
    assert exact(torch,result[0]['actual_spec_layout'],result[1]['actual_spec_layout'])
    assert exact(torch,result[0]['elementary_tables'],result[1]['elementary_tables'])
    assert not same(torch,result[0]['x'],result[1]['x']),'Consecutive input refresh must actually change coefficients'
    return result,inp['source_identity']['algorithm_identity'],files

def arm(ctx,adapter,ps,chunk):
    torch,d,se=ctx.torch,ctx.d,ctx.c.se
    from flowstar_gpu import support as sp,elementary as elem
    tables=d.build_tables(16,3).to('cuda:0')
    work=sp.SparseEngine(tables,d.poly.build_step_tables(tables,.005),'cuda:0')
    work.horner_edge_kernel=ctx.edge.horner_edge
    eng=se.validation_engine_for_order(work,4)
    names=[v['name'] for v in ctx.cfg['initial_set']]
    code2=d.compile_ode(ctx.cfg['dynamics_expressions'],names,order=2)
    code=se.validation_code_for_order(code2,4,eng);tabs=elem.elem_tables(4,'cuda:0')
    assert source.canonical_sha(dataclasses.asdict(code))==ps[0]['code_sha256']==ps[1]['code_sha256']
    def support(v):
        s=sp.make_support(v['n'],v['k'],v['spatial'],tuple(v['ids']))
        assert exact(torch,source.support_summary(s,eng),v);return s
    p=ps[0];v=p['specialization']
    spec=se.specialize(code,tuple(support(s) for s in v['variable_supports']),v['k'],support(v['x_support']),eng,
        **{k:v[k] for k in ['validated_full_leaves','validated_direct_leaves','validated_linear_leaves']})
    assert exact(torch,source.spec_summary(spec,eng),p['actual_spec_layout'])
    for key,value in p['elementary_tables'].items():assert exact(torch,getattr(tabs,key),value)
    raw=se._graphed_valid;binding=adapter.install(torch,se) if chunk==128 else None
    outputs=[];receipts=[];retained=None;retained_cpu=None
    try:
        for case,p in zip(['step1','step2','step2_marked_bad'],[ps[0],ps[1],ps[1]]):
            ins={name:p[name].to('cuda:0',copy=True) for name in ['x','x_rem','bad']}
            if case.endswith('marked_bad'):ins['bad'][::73]=True
            before={k:v.cpu().clone() for k,v in ins.items()};versions={k:v._version for k,v in ins.items()}
            value=se._graphed_valid(spec,code,ins['x'],ins['x_rem'],ins['bad'],eng,1e-6,tabs)
            torch.cuda.synchronize()
            assert len(value)==5
            current=tuple(t.detach().to('cpu',copy=True).contiguous() for t in value)
            assert all(same(torch,v,before[k]) and v._version==versions[k] for k,v in ins.items())
            cache=eng._graphs;assert cache.enabled and ('vd',id(spec)) in cache._segs
            assert eng._vtape[id(spec)] is None,'Actual capacity fallback must still select graph dispatch'
            assert cache.captures==1 and cache.hits==(len(outputs)+1)*(8 if chunk==128 else 1)
            assert cache._segs[('vd',id(spec))][0][0].shape[0]==chunk
            assert bool(current[4][before['bad']].all())
            if retained is not None:assert all(same(torch,a,b) for a,b in zip(retained,retained_cpu))
            # Public candidate outputs already own storage. Original outputs
            # are static, so retain explicit clones for the reference arm.
            retained=tuple(value) if chunk==128 else tuple(t.clone() for t in value)
            retained_cpu=current
            outputs.append(current)
            receipts.append(dict(case=case,input_bytes_unchanged=True,marked_bad_preserved=True,
                outputs=fingerprints(torch,current,'outputs'),captures=cache.captures,hits=cache.hits,
                graph_input_rows=chunk,owned_previous_outputs_survive_next_replay=True))
        return outputs,dict(chunk=chunk,cases=receipts,counters=dict(binding.counters) if binding else None)
    finally:
        if binding is not None:binding.restore()
        assert se._graphed_valid is raw

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();r=dict(status='exception')
    try:
        ctx=w.initialize(a);torch=ctx.torch
        ps,identity,files=payloads(a,torch);assert identity==ctx.identity
        adapter=load('ordinary_valid128_candidate',HERE/'valid_chunk128.py',ADAPTER_SHA)
        left,lr=arm(ctx,adapter,ps,1024);gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        right,rr=arm(ctx,adapter,ps,128)
        for x,y in zip(left,right):assert all(same(torch,u,v) for u,v in zip(x,y))
        for name,digest in files.items():assert sha(a.capture/name)==digest
        r=dict(status='passed',device='cuda:0',torch_version=torch.__version__,source_identity=ctx.identity,source_files_sha256=files,source_capture_sha256=CAPTURE_SHA,
            source_audit_sha256=AUDIT_SHA,adapter_sha256=ADAPTER_SHA,source_capture_passed=False,source_run_is_trajectory_qualification=False,
            arms=[lr,rr],all_five_outputs_all_three_cases_bytes_equal=True,
            consecutive_actual_inputs_changed=True,same_owner_second_graph_hit=True,
            input_rows=1024,working_order=3,point_code_order=2,validation_order=4,
            max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            scope='Fixed real consecutive step1/step2 validation inputs plus explicitly marked bad-mask case. Same spec and refreshed graph inputs per owner. No SR/NN/advance/full-trajectory or speed qualification.')
    except BaseException as e:r.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        if 'ctx' in locals():
            ctx.postwarm.restore();ctx.postwarm_log.close();ctx.weighted128.restore();r['runtime_hooks_restored']=True
        r.update(script_sha256=sha(__file__),process_s=time.perf_counter()-start)
        write(a.output/'RESULT.json',r);print(json.dumps({k:r[k] for k in ['status','process_s']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['capture','capture-audit','common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.mode='valid128_fixed_input_check';a.source40=None;main(a)
