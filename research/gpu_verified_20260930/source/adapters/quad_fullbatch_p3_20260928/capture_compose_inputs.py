"""Capture actual full1024 Horner operands/results outside CUDA capture.

The frozen original compose_s is called once with its original arguments.
Only independent CPU snapshots are written. Stop after original prune2;
partial capture files never constitute a trajectory qualification.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
ENTRY_SHA='b37cbdc4be1d7f4c2e84b096404913a84ead85c47995687511b66242a8eb3f09'
HORNER_SHA='d090e314f710d1d4763ea299336e0dba4dbada642b43700f09fec7c238977f4f'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
path=HERE/'run_fullbatch_p3_working_prune.py';assert sha(path)==ENTRY_SHA
spec=importlib.util.spec_from_file_location('compose_capture_frozen_entry',path)
w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
class DiagnosticStop(Exception):pass

class Runtime(w.Runtime):
    def prune_state(self,*args,**kwargs):
        value=super().prune_state(*args,**kwargs)
        if self.completed==2:raise DiagnosticStop()
        return value

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception');receipts=[]
    raw=None
    try:
        ctx=w.initialize(args);original=ctx.identity
        from flowstar_gpu import sparse_horner,support as sp
        assert sha(sparse_horner.__file__)==HORNER_SHA
        ctx.identity=dict(original,script_sha256=sha(__file__),diagnostic_entry_sha256=ENTRY_SHA,
            algorithm_identity=original,diagnostic_only=True,
            limitation='Actual compose operands and immediate results, CPU snapshots only. Capture overhead changes timing/GC; not a performance or trajectory qualification.')
        w.write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode=args.mode,
            driver_argv=w.driver_argv(ctx),stop='after_prune2_before_poststep_safety_reset',
            torch_cap_bytes=11*1024**3,external_guard_bytes=int(11.5*1024**3)))
        runtime=Runtime(ctx,args);raw=ctx.c.se.compose_s
        assert ctx.c.se.COMPOSITION_MODE=='horner'
        def support(s):
            return dict(n=s.n,k=s.k,spatial=s.spatial,ids=list(s.ids),
                exps=ctx.torch.from_numpy(sp._exps_for(s).copy()))
        def save(name,payload):
            path=args.output/(name+'.pt');ctx.torch.save(payload,path)
            receipt=dict(pt_sha256=sha(path),pt_bytes=path.stat().st_size,signature=runtime.signature(payload),
                source_identity=ctx.identity,input_sha256=sha(args.output/'INPUT.json'))
            w.write(path.with_suffix('.json'),receipt)
            return dict(file=path.name,pt_sha256=receipt['pt_sha256'],sidecar_sha256=sha(path.with_suffix('.json')))
        def capture(*a,**kw):
            assert not kw and len(a)==11 and not ctx.torch.cuda.is_current_stream_capturing()
            ac,ar,gc,gr,sa,sg,order,cutoff,strict,sched,eng=a
            step=runtime.completed+1;assert step in [1,2]
            assert eng is runtime.eng and eng.tables.n==16 and eng.tables.k==3
            assert ac.shape[:2]==gc.shape[:2]==(1024,16) and order==3 and cutoff==1e-6 and strict is True
            assert all(t.dtype==ctx.torch.float64 and t.is_cuda for t in (ac,ar,gc,gr))
            assert eng.horner_edge_kernel is ctx.edge.horner_edge
            payload=dict(schema='actual_horner_compose_v1',step=step,source_entry_sha256=ENTRY_SHA,
                source_script_sha256=sha(__file__),horner_sha256=HORNER_SHA,input_sha256=sha(args.output/'INPUT.json'),
                inputs={name:runtime.owned(t) for name,t in zip(['a_coeffs','a_rem','g_coeffs','g_rem'],[ac,ar,gc,gr])},
                support_a=support(sa),support_g=support(sg),order=order,cutoff=cutoff,strict=strict,
                h=.005,batch=1024,n=16,horner_edge_enabled=True,iv_use_kernels=bool(sp.iv.USE_KERNELS))
            receipt=dict(step=step,input=save(f'COMPOSE_INPUT_{step}',payload))
            versions=[t._version for t in (ac,ar,gc,gr)]
            out=raw(*a,**kw)
            assert [t._version for t in (ac,ar,gc,gr)]==versions
            coeff,rem,sup=out
            output=dict(schema='actual_horner_result_v1',step=step,input_pt_sha256=receipt['input']['pt_sha256'],
                coeff=runtime.owned(coeff),remainder=runtime.owned(rem),support=support(sup))
            receipt['output']=save(f'COMPOSE_OUTPUT_{step}',output);receipts.append(receipt)
            return out
        ctx.c.se.compose_s=capture
        argv=sys.argv;sys.argv=w.driver_argv(ctx)
        try:
            returned=ctx.d.main();result.update(status='original_driver_returned',driver_return=returned)
        except DiagnosticStop:
            assert runtime.completed==2 and [row['accepted_lanes'] for row in runtime.rows]==[1024]*2
            assert [row['step'] for row in receipts]==[1,2]
            result.update(status='diagnostic_two_compose_pairs_captured',accepted_lanes_by_step=[1024]*2)
        finally:sys.argv=argv
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),source_entry_sha256=ENTRY_SHA,process_s=time.perf_counter()-start,
            diagnostic_only=True,no_timing_or_execution_qualification=True,compose_pairs=receipts)
        if (args.output/'INPUT.json').exists():result['input_sha256']=sha(args.output/'INPUT.json')
        if raw is not None:ctx.c.se.compose_s=raw;result['compose_hook_restored']=True
        if 'runtime' in locals():result.update(completed_advance_count=runtime.completed)
        if 'ctx' in locals():
            result.update(source_identity=ctx.identity,max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),
                max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved(),working_graph_counters=dict(ctx.working_prune.counters))
            ctx.working_prune.restore();ctx.working_prune_log.close();result['working_graph_hook_restored']=True
            ctx.valid128.restore();ctx.postwarm.restore();ctx.postwarm_log.close();ctx.weighted128.restore()
            result['valid_adapter_restored']=result['cache_release_adapter_restored']=result['weighted_adapter_restored']=True
        w.write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['diagnostic_two_compose_pairs'],default='diagnostic_two_compose_pairs')
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker',
        'working-prune-reference','working-prune-check','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.source40=None;main(args)
