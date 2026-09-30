"""Capture the first real full1024 P3 refine_accepted input, then stop.

No weighted map runs and no SR/checkpoint is serialized. This interrupts an
unfinished advance and cannot be admitted as a trajectory or committed state.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,importlib.util,inspect,json,sys,time,traceback

HERE=Path(__file__).parent
ENTRY_SHA='22f4dda16dcfd30563e6652ec8af82034b3b138a7505dbf31714a8bd55820a72'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
p=HERE/'run_fullbatch_p3.py';assert sha(p)==ENTRY_SHA
s=importlib.util.spec_from_file_location('weighted_input_frozen_entry',p)
w=importlib.util.module_from_spec(s);sys.modules[s.name]=w;s.loader.exec_module(w)
class CapturedInputStop(Exception):pass

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();result=dict(status='exception')
    raw=None;capture=None
    try:
        ctx=w.initialize(a);torch=ctx.torch;algorithm_identity=ctx.identity
        ctx.identity=dict(algorithm_identity,script_sha256=sha(__file__),source_entry_sha256=ENTRY_SHA,
            algorithm_identity=algorithm_identity,diagnostic_only=True,
            limitation='Read-only weighted-input capture during unfinished first advance; no weighted map, SR serialization, accepted-step or trajectory claim.')
        w.write(a.output/'INPUT.json',dict(source_identity=ctx.identity,driver_argv=w.driver_argv(ctx),
            stop='first_refine_accepted_entry_before_body',torch_cap_bytes=11*1024**3,external_guard_bytes=int(11.5*1024**3)))
        from flowstar_gpu import weighted_validation as weighted
        raw=weighted.refine_accepted;signature=inspect.signature(raw)
        x=w.Runtime(ctx,a)
        def fingerprint(t):
            assert t.device.type=='cpu' and t.is_contiguous()
            h=hashlib.sha256();h.update(memoryview(t.view(torch.uint8).numpy()).cast('B'))
            return dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size(),sha256=h.hexdigest())
        def hook(*args,**kwargs):
            nonlocal capture
            assert capture is None and x.completed==0 and x.held==0
            bound=signature.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
            eng,code=v['eng'],v['code'];assert eng.tables.n==16 and eng.tables.k==code.order==4
            assert isinstance(eng,ctx.hybrid.backend.MetadataEngine) and eng.step.delta==.005
            assert v['sup'].k==v['initial_sup'].k==4 and not v['sup'].spatial and not v['initial_sup'].spatial
            assert len(v['var_sups'])==16 and all(sup.k==4 and not sup.spatial for sup in v['var_sups'])
            assert v['x'].shape==(1024,16,v['sup'].size) and v['initial'].shape==(1024,16,v['initial_sup'].size)
            assert v['eligible'].shape==(1024,) and v['eligible'].dtype==torch.bool
            for name in ['x','initial','original','current']:assert v[name].dtype==torch.float64
            assert v['original'].shape==v['current'].shape==(1024,16,2)
            assert v['cutoff']==1e-6 and v['rounds']==2 and v['chunk_size']==512 and v['use_graph'] is True
            assert v['trace'] is None
            tensors={name:v[name] for name in ['x','initial','original','current','eligible']}
            versions={name:value._version for name,value in tensors.items()}
            torch.cuda.synchronize();start=time.perf_counter()
            payload={name:value.detach().to('cpu',copy=True).contiguous() for name,value in tensors.items()}
            exps=lambda sup:torch.tensor(eng.exponents(sup).copy(),dtype=torch.int64,device='cpu')
            payload.update(support_ids=list(v['sup'].ids),support_exponents=exps(v['sup']),
                initial_ids=list(v['initial_sup'].ids),initial_exponents=exps(v['initial_sup']),
                variable_support_ids=[list(sup.ids) for sup in v['var_sups']],
                variable_exponents=[exps(sup) for sup in v['var_sups']])
            assert all(value._version==versions[name] for name,value in tensors.items())
            code_value=dataclasses.asdict(code)
            code_json=json.dumps(code_value,sort_keys=True,separators=(',',':'),allow_nan=False)
            code_sha=hashlib.sha256(code_json.encode()).hexdigest()
            payload.update(code=code_value,code_sha256=code_sha,n=16,batch=1024,working_order=3,point_code_order=2,
                validation_order=4,support_family=4,h=.005,cutoff=v['cutoff'],rounds=v['rounds'],chunk_size=v['chunk_size'],
                use_graph=v['use_graph'],trace_present=False,attempted_step=1,held_controller_step=0)
            data=a.output/'WEIGHTED_INPUT.pt';torch.save(payload,data)
            summary={name:fingerprint(value) for name,value in payload.items() if isinstance(value,torch.Tensor)}
            capture=dict(source_identity=ctx.identity,pt_sha256=sha(data),pt_size_bytes=data.stat().st_size,
                code_sha256=code_sha,code_order=code.order,n_cache=code.n_cache,n_strict=code.n_strict,
                supports=dict(full_size=v['sup'].size,initial_size=v['initial_sup'].size,variable_sizes=[sup.size for sup in v['var_sups']]),
                tensors=summary,eligible_lanes=int(payload['eligible'].sum()),rounds=v['rounds'],chunk_size=v['chunk_size'],
                use_graph=v['use_graph'],cutoff=v['cutoff'],input_versions_unchanged=True,
                completed_advance_count=x.completed,attempted_step=1,held_controller_step=x.held,
                pending_SR_token=x.ledger.pending,SR_not_serialized=True,weighted_body_not_called=True,
                capture_s=time.perf_counter()-start)
            w.write(a.output/'WEIGHTED_INPUT.json',capture)
            raise CapturedInputStop()
        weighted.refine_accepted=hook
        argv=sys.argv;sys.argv=w.driver_argv(ctx)
        try:
            returned=ctx.d.main();raise RuntimeError('Original driver returned without requested weighted input: '+str(returned))
        except CapturedInputStop:
            result=dict(status='captured_weighted_input_only',capture=capture,no_committed_step=True)
        finally:sys.argv=argv
    except BaseException as e:
        result.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        if raw is not None:weighted.refine_accepted=raw;result['weighted_hook_restored']=True
        result.update(script_sha256=sha(__file__),source_entry_sha256=ENTRY_SHA,process_s=time.perf_counter()-started,
            diagnostic_only=True,no_timing_or_execution_qualification=True)
        if 'ctx' in locals():result.update(source_identity=ctx.identity,max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved())
        w.write(a.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.mode='capture_weighted_input';a.source40=None;main(a)
