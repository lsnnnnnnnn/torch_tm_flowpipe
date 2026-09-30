"""Capture real ordinary-validation inputs for steps1 and2, then stop.

The first original dispatcher runs unchanged. The second does not run. All1024
rows and the original NN/SR trajectory are retained; SR is never serialized.
Payloads contain CPU tensors and primitive reconstruction data only.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,importlib.util,inspect,json,sys,time,traceback

HERE=Path(__file__).parent
ENTRY_SHA='6e74c716ac3ec9da791c6e968f73df0cc400030d218857152629976109213674'
SPARSE_SHA='56e590796dbae42685b0331c7f41b53b62a3f73ed9be74e22738493820353ddf'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
path=HERE/'run_fullbatch_p3_cache_release.py';assert sha(path)==ENTRY_SHA
spec=importlib.util.spec_from_file_location('ordinary_valid_capture_frozen_entry',path)
w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
class CapturedInputsStop(Exception):pass

def support_summary(s,eng):
    import torch
    return dict(n=s.n,k=s.k,spatial=s.spatial,ids=list(s.ids),
        exponents=torch.tensor(eng.exponents(s).copy(),dtype=torch.int64,device='cpu'))

def spec_summary(tape,eng):
    """CPU-only reconstruction comparison; no opaque PairBind serialization."""
    owned=lambda t:t.detach().to('cpu',copy=True).contiguous()
    instructions=[]
    for ins in tape.instrs:
        row={name:getattr(ins,name) for name in ['op','dst','a','b','var','const','crem_lo','crem_hi','power','cache_base','strict_slot','keep_len']}
        row.update(sup_out=support_summary(ins.sup_out,eng),gather=owned(ins.gather) if ins.gather is not None else None)
        instructions.append(row)
    return dict(k=tape.k,instructions=instructions,out_slots=list(tape.out_slots),
        out_supports=[support_summary(s,eng) for s in tape.out_sups],output_union=support_summary(tape.sup_out_union,eng),
        out_embeds=[owned(t) for t in tape.out_embeds])

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    result=dict(status='exception');raw=None;captures=[];first_body_completed=False;first_spec=None;first_engine=None
    try:
        ctx=w.initialize(args);torch=ctx.torch;se=ctx.c.se;algorithm_identity=ctx.identity
        assert sha(se.__file__)==SPARSE_SHA
        ctx.identity=dict(algorithm_identity,script_sha256=sha(__file__),source_entry_sha256=ENTRY_SHA,
            algorithm_identity=algorithm_identity,diagnostic_only=True,
            limitation='Two ordinary-validation input captures; original first dispatcher continues, second dispatch is not executed. No full second step, own40/cold, speed or SR-checkpoint qualification.')
        w.write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode=args.mode,driver_argv=w.driver_argv(ctx),
            stop='second_graphed_valid_entry_before_original_dispatch',torch_cap_bytes=11*1024**3,
            external_guard_bytes=int(11.5*1024**3)))
        runtime=w.Runtime(ctx,args);raw=se._graphed_valid;signature=inspect.signature(raw)
        assert inspect.getfile(raw)==str(Path(se.__file__)) and raw.__name__=='_graphed_valid'
        def owned(t):return t.detach().to('cpu',copy=True).contiguous()
        def fingerprint(t):
            assert t.device.type=='cpu' and t.is_contiguous()
            return dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size(),
                sha256=hashlib.sha256(memoryview(t.view(torch.uint8).numpy()).cast('B')).hexdigest())
        def hook(*positional,**keywords):
            nonlocal first_body_completed,first_spec,first_engine
            number=len(captures)+1;assert number in [1,2]
            # Refuse to mislabel a retry in step1 as the desired step2 input.
            assert runtime.completed==number-1 and runtime.held==0
            bound=signature.bind(*positional,**keywords);bound.apply_defaults();v=bound.arguments
            tape,code,eng=v['spec'],v['code'],v['eng']
            assert isinstance(eng,ctx.hybrid.backend.MetadataEngine)
            assert eng.tables.n==16 and eng.tables.k==code.order==tape.k==4 and eng.step.delta==.005
            assert v['cutoff_eps']==1e-6
            matches=[key for key,value in eng._spec.items() if value is tape];assert len(matches)==1
            key=matches[0];assert len(key)==7 and key[0]==se._code_serial(code) and key[1]==tape.k
            x_sup,var_sups=key[2],key[3]
            assert not x_sup.spatial and x_sup.k==4 and len(var_sups)==16
            assert v['x'].shape==(1024,16,x_sup.size) and v['x_rem'].shape==(1024,16,2)
            assert v['x'].dtype==v['x_rem'].dtype==torch.float64
            assert v['bad'].shape==(1024,) and v['bad'].dtype==torch.bool
            assert runtime.cap.shape==(1024,16,2)
            if number==1:first_spec,first_engine=tape,eng
            else:assert first_body_completed and tape is first_spec and eng is first_engine
            inputs={name:v[name] for name in ['x','x_rem','bad']};inputs['cap']=runtime.cap
            versions={name:t._version for name,t in inputs.items()}
            torch.cuda.synchronize();start=time.perf_counter()
            payload={name:owned(t) for name,t in inputs.items()}
            assert bool(torch.isfinite(payload['x']).all() and torch.isfinite(payload['x_rem']).all())
            assert bool((payload['x_rem'][...,0]<=payload['x_rem'][...,1]).all())
            assert bool((payload['cap'][...,0]==-.1).all() and (payload['cap'][...,1]==.1).all())
            reconstruction=dict(k=key[1],x_support=support_summary(x_sup,eng),variable_supports=[support_summary(s,eng) for s in var_sups],
                validated_full_leaves=key[4],validated_direct_leaves=key[5],validated_linear_leaves=key[6])
            # Pair/union tables are rebuilt by the pinned specialize function
            # from this exact cache key. Preserve actual instruction layouts as
            # an independent reconstruction check, without pickling GPU binds.
            layout=spec_summary(tape,eng)
            tabs=v['tabs'];assert dataclasses.is_dataclass(tabs) and tabs.max_order==4
            tables={f.name:owned(getattr(tabs,f.name)) if isinstance(getattr(tabs,f.name),torch.Tensor)
                else getattr(tabs,f.name) for f in dataclasses.fields(tabs)}
            code_value=dataclasses.asdict(code)
            payload.update(code=code_value,code_sha256=canonical_sha(code_value),specialization=reconstruction,
                actual_spec_layout=layout,elementary_tables=tables,n=16,batch=1024,working_order=3,point_code_order=2,
                validation_order=4,h=.005,cutoff=v['cutoff_eps'],attempted_step=number,held_controller_step=0,
                source_input_sha256=sha(args.output/'INPUT.json'),source_script_sha256=sha(__file__),source_entry_sha256=ENTRY_SHA)
            assert all(t._version==versions[name] for name,t in inputs.items())
            fingerprints={}
            def collect(value,name):
                if isinstance(value,torch.Tensor):fingerprints[name]=fingerprint(value)
                elif isinstance(value,dict):
                    for k,item in value.items():collect(item,name+'/'+str(k))
                elif isinstance(value,(list,tuple)):
                    for i,item in enumerate(value):collect(item,name+'/'+str(i))
            collect(payload,'payload')
            destination=args.output/('VALID_INPUT_'+str(number)+'.pt');torch.save(payload,destination)
            receipt=dict(source_identity=ctx.identity,pt_sha256=sha(destination),pt_size_bytes=destination.stat().st_size,
                source_input_sha256=payload['source_input_sha256'],code_sha256=payload['code_sha256'],tensors=fingerprints,
                completed_advance_count=runtime.completed,attempted_step=number,held_controller_step=runtime.held,
                pending_SR_token=runtime.ledger.pending,SR_not_serialized=True,input_versions_unchanged=True,
                spec_same_object_as_first=tape is first_spec,engine_same_object_as_first=eng is first_engine,
                reconstruction='Pinned specialize(code,var_supports,k,x_support,engine,three_saved_flags); actual layouts and ElemTables saved for comparison.',
                original_dispatch_executed=False,capture_s=time.perf_counter()-start)
            w.write(destination.with_suffix('.json'),receipt);captures.append(receipt)
            # Drop owned CPU payload before normal step1 execution; no extra
            # GPU tensors were created by capture.
            del payload
            if number==2:raise CapturedInputsStop()
            output=raw(*positional,**keywords);first_body_completed=True
            receipt['original_dispatch_executed']=True;w.write(destination.with_suffix('.json'),receipt)
            return output
        se._graphed_valid=hook
        argv=sys.argv;sys.argv=w.driver_argv(ctx)
        try:
            returned=ctx.d.main();raise RuntimeError('Original driver returned before both inputs: '+str(returned))
        except CapturedInputsStop:
            assert len(captures)==2 and runtime.completed==1 and first_body_completed
            result=dict(status='captured_two_valid_inputs_only',captures=captures,completed_first_step=True,
                first_step_accepted_lanes=runtime.rows[0]['accepted_lanes'],second_dispatch_executed=False,no_complete_second_step=True)
        finally:sys.argv=argv
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        if raw is not None:se._graphed_valid=raw;result['valid_hook_restored']=True
        result.update(script_sha256=sha(__file__),source_entry_sha256=ENTRY_SHA,process_s=time.perf_counter()-started,
            diagnostic_only=True,no_timing_or_execution_qualification=True)
        if 'ctx' in locals():
            result.update(source_identity=ctx.identity,max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),
                max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved(),weighted128_counters=dict(ctx.weighted128.counters),
                postwarm_releases=list(ctx.postwarm.releases))
            ctx.postwarm.restore();ctx.postwarm_log.close();result['cache_release_adapter_restored']=True
            ctx.weighted128.restore();result['weighted_adapter_restored']=True
        w.write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['capture_valid_inputs'],default='capture_valid_inputs')
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.source40=None;main(args)
