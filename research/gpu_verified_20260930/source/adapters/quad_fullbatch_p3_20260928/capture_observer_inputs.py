"""Capture two actual observer inputs on frozen b89 full1024/P3/128.

Save owned CPU inputs before the original observer, call that observer once,
and link its unchanged artifacts. No SR dump or observer replacement. A clean
diagnostic stop follows original prune2, before driver poststep safety/reset.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
ENTRY_SHA='b89c5710f5faa9dc1b363eba96d27cbbc7bc2a7006df9aa2d305f39cf202b2e7'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
path=HERE/'run_fullbatch_p3_horner_cleanup.py';assert sha(path)==ENTRY_SHA
spec=importlib.util.spec_from_file_location('observer_capture_frozen128_entry',path)
w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)
class DiagnosticStop(Exception):pass

class Runtime(w.Runtime):
    def __init__(self,ctx,args):
        super().__init__(ctx,args);self.observer_pairs=[]
    def observe(self,duration):
        from flowstar_gpu import support as sp,interval as iv
        st,eng,step=self.st,self.eng,self.completed
        assert step in [1,2] and not self.t.cuda.is_current_stream_capturing()
        assert st.pre.shape[:2]==(1024,16) and st.pre_rem.shape==(1024,16,2)
        assert st.pre.dtype==st.pre_rem.dtype==self.t.float64 and st.pre.is_cuda
        assert (eng.tables.n,eng.tables.k,eng.step.delta,eng.step.lanes)==(16,3,.005,0)
        assert not st.pre_sup.spatial and (st.pre_sup.n,st.pre_sup.k)==(16,3)
        versions=[(getattr(st,k),getattr(st,k)._version) for k in ['pre','pre_rem','tmv','tmv_rem','status']]
        support=dict(n=st.pre_sup.n,k=st.pre_sup.k,spatial=st.pre_sup.spatial,ids=list(st.pre_sup.ids),
            exps=self.t.from_numpy(sp._exps_for(st.pre_sup).copy()))
        payload=dict(schema='actual_fullbatch_observer_input_v1',step=step,
            source_entry_sha256=ENTRY_SHA,source_script_sha256=sha(__file__),input_sha256=sha(self.args.output/'INPUT.json'),
            pre=self.owned(st.pre[:,:12]),pre_rem=self.owned(st.pre_rem[:,:12]),pre_support=support,
            accepted=self.owned(self.ok),status=self.owned(st.status),
            piece=self.t.full((1024,2),.005,dtype=self.t.float64,device='cpu'),
            batch=1024,physical_rows=12,n=16,working_order=3,h=.005,
            table_identity=dict(n=eng.tables.n,k=eng.tables.k,T=eng.tables.T,T2=eng.tables.T2,
                step_delta=eng.step.delta,step_lanes=eng.step.lanes,iv_use_kernels=bool(iv.USE_KERNELS),
                class_name=type(eng.tables).__name__),
            config_sha256=self.ctx.identity['config_sha256'],model_sha256=self.ctx.identity['model_sha256'],
            driver_sha256=self.ctx.identity['driver_sha256'],engine_python_sha256=self.ctx.identity['engine']['python_sha256'])
        path=self.args.output/f'OBSERVER_INPUT_{step}.pt';self.t.save(payload,path)
        meta=dict(source_identity=self.ctx.identity,input_sha256=payload['input_sha256'],step=step,
            pt_sha256=sha(path),pt_bytes=path.stat().st_size,signature=self.signature(payload),
            original_observer_executed_before_this_file=False)
        w.write(path.with_suffix('.json'),meta)
        assert all(t._version==v for t,v in versions)
        # Exactly the original observer. It writes observer_N.pt/json itself.
        value=super().observe(duration)
        assert all(t._version==v for t,v in versions)
        out_path=self.args.output/f'observer_{step}.pt';side=out_path.with_suffix('.json')
        original=w.read(side)
        assert original['source_identity']==self.ctx.identity and original['step']==step
        assert sha(out_path)==original['pt_sha256']
        pair=dict(schema='actual_fullbatch_observer_pair_v1',step=step,source_identity=self.ctx.identity,
            input=dict(file=path.name,pt_sha256=meta['pt_sha256'],sidecar_sha256=sha(path.with_suffix('.json'))),
            output=dict(file=out_path.name,pt_sha256=original['pt_sha256'],sidecar_sha256=sha(side)),
            original_observer_called_once=True,input_state_versions_unchanged=True,
            phase='original_observer_returned_before_original_prune')
        pair_path=self.args.output/f'OBSERVER_PAIR_{step}.json';w.write(pair_path,pair)
        self.observer_pairs.append(dict(step=step,file=pair_path.name,sha256=sha(pair_path)))
        return value
    def prune_state(self,*args,**kwargs):
        value=super().prune_state(*args,**kwargs)
        if self.completed==2:raise DiagnosticStop()
        return value

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception')
    try:
        ctx=w.initialize(args);original=ctx.identity
        assert ctx.valid128.counters['chunk_size']==ctx.weighted128.counters['chunk_size']==128
        ctx.identity=dict(original,script_sha256=sha(__file__),diagnostic_entry_sha256=ENTRY_SHA,
            algorithm_identity=original,diagnostic_only=True,
            limitation='Owned CPU observer inputs, then the unchanged original observer once per step. Capture changes timing/GC; no solver-speed or trajectory qualification. No64 adapter, observer chunking, new GC or SR dump is installed.')
        w.write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode=args.mode,driver_argv=w.driver_argv(ctx),
            stop='after_original_prune2_before_poststep_safety_reset',torch_cap_bytes=11*1024**3,
            external_guard_bytes=int(11.5*1024**3)))
        runtime=Runtime(ctx,args);argv=sys.argv;sys.argv=w.driver_argv(ctx)
        try:
            returned=ctx.d.main();result.update(status='original_driver_returned',driver_return=returned)
        except DiagnosticStop:
            assert runtime.completed==2 and [r['accepted_lanes'] for r in runtime.rows]==[1024,1024]
            assert [r['step'] for r in runtime.observer_pairs]==[1,2]
            result.update(status='diagnostic_two_observer_pairs_captured',accepted_lanes_by_step=[1024,1024])
        finally:sys.argv=argv
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),source_entry_sha256=ENTRY_SHA,process_s=time.perf_counter()-start,
            diagnostic_only=True,no_timing_or_execution_qualification=True)
        if (args.output/'INPUT.json').exists():result['input_sha256']=sha(args.output/'INPUT.json')
        if 'runtime' in locals():result.update(completed_advance_count=runtime.completed,observer_pairs=runtime.observer_pairs)
        if 'ctx' in locals():
            result.update(source_identity=ctx.identity,max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),
                max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved(),working_graph_counters=dict(ctx.working_prune.counters))
            ctx.horner_cleanup.restore();result['horner_cleanup_restored']=True
            ctx.working_prune.restore();ctx.working_prune_log.close();result['working_graph_hook_restored']=True
            ctx.valid128.restore();ctx.postwarm.restore();ctx.postwarm_log.close();ctx.weighted128.restore()
            result['valid_adapter_restored']=result['cache_release_adapter_restored']=result['weighted_adapter_restored']=True
        w.write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['diagnostic_two_observer_pairs'],default='diagnostic_two_observer_pairs')
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker',
        'working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.source40=None;main(args)
