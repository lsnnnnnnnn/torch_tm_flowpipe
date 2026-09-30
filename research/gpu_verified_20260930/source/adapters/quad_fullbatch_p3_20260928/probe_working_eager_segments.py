"""Nine-step byte comparison of five working segments executed without graphs.

The frozen observer64 entry retains128 validation, full1024 and full SR1000.
Stop after original prune9, before driver post-step safety/reset: diagnostic
only, not an own40/cold qualification, committed checkpoint or timing result.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
ENTRY_SHA='171dd167496895b15fd1373de16986d30118b3338c4140a4f5c91b98eee353c9'
ADAPTER_SHA='c4fb7b098a10ab98d9e9250fc88460ade130225ef710ea5734522df03ca0453b'
REFERENCE_INPUT_SHA='52201b09ddb6060a1ba797b6e15f69ab455941d7d1179e849d68739daced8938'
REFERENCE_PRUNE_SHA='8986261abbad0fa27f067f54e2b7a381f8b1ccb4fcbbc82b110e838f5e926d04'
REFERENCE_WATCH_SHA='9bb887788c0064d4626ffd13bfd89f0cd0649b33857c429301ee2e03c303c4f3'
POLICY='diagnostic_full1024_P3_five_working_segments_eager_original128_observer64'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
w=load('working_eager_probe_frozen_observer',HERE/'run_fullbatch_p3_observer64.py',ENTRY_SHA)

def reference_gate(args,identity=None):
    ref=args.reference;watch=args.reference_watch
    assert sha(ref/'INPUT.json')==REFERENCE_INPUT_SHA and sha(ref/'working_prune.jsonl')==REFERENCE_PRUNE_SHA
    assert sha(watch)==REFERENCE_WATCH_SHA and read(watch)['status']=='gpu_guard_11.5GiB'
    assert not (ref/'RESULT.json').exists(),'Reference remains a resource-failed partial run'
    inp=read(ref/'INPUT.json');source=inp['source_identity']
    assert source['script_sha256']==ENTRY_SHA and source['policy']==w.POLICY
    if identity is not None:assert source==identity
    rows=[json.loads(line) for line in (ref/'working_prune.jsonl').read_text().splitlines()]
    assert len(rows)==18
    files={'INPUT.json':REFERENCE_INPUT_SHA,'working_prune.jsonl':REFERENCE_PRUNE_SHA}
    mathematical=[]
    for i in range(9):
        eviction,row=rows[2*i:2*i+2];step=i+1
        assert eviction['event']=='eviction_complete' and row['event']=='invariance_checked'
        assert eviction['step']==row['step']==step and eviction['eviction']==row['eviction']
        assert row['before_after_mathematical_bytes_equal'] is True and row['capacity_and_storage_unchanged'] is True
        storage=row['storage_before'];assert storage['sr_max_size']==storage['host_max_size']==storage['sr_capacity']==1000
        assert storage['qlen']==storage['jlen']==step
        meta=read(ref/f'observer_{step}.json')
        assert meta['source_identity']==source and meta['step']==step
        assert sha(ref/f'observer_{step}.pt')==meta['pt_sha256']
        for suffix in ['pt','json']:
            name=f'observer_{step}.{suffix}';files[name]=sha(ref/name)
        mathematical.append(row['mathematical_signature'])
    return source,mathematical,files

class DiagnosticStop(Exception):pass

class Runtime(w.Runtime):
    def __init__(self,ctx,args,reference_math):
        self.reference_math=reference_math;self.reference_steps=[]
        super().__init__(ctx,args)
    def advance_sparse(self,st,code,eng,sched,settings,cap,sr):
        assert self.completed<9
        self.ctx.eager_segments.begin_step(eng)
        return super().advance_sparse(st,code,eng,sched,settings,cap,sr)
    def prune_state(self,*args,**kwargs):
        value=super().prune_state(*args,**kwargs);step=self.completed
        assert 1<=step<=9
        # Reuse the just-completed frozen full mathematical/storage check.
        row=json.loads((self.args.output/'working_prune.jsonl').read_text().splitlines()[-1])
        assert row['event']=='invariance_checked' and row['step']==step
        assert row['mathematical_signature']==self.reference_math[step-1],f'Mathematical mismatch at {step}'
        old=read(self.args.reference/f'observer_{step}.json')
        new=read(self.args.output/f'observer_{step}.json')
        assert new['signature']==old['signature']
        assert sha(self.args.output/f'observer_{step}.pt')==new['pt_sha256']==old['pt_sha256']
        assert self.rows[-1]['accepted_lanes']==1024 and self.rows[-1]['sr_length']==step
        dispatch=self.ctx.eager_segments.finish_step(self.eng)
        free,total=self.t.cuda.mem_get_info()
        memory=dict(allocated=self.t.cuda.memory_allocated(),reserved=self.t.cuda.memory_reserved(),
            device_free=free,device_total=total,device_used=total-free)
        self.reference_steps.append(step)
        write(self.args.output/f'PAIRED_{step}.json',dict(step=step,reference_input_sha256=REFERENCE_INPUT_SHA,
            full_mathematical_state_SR_host_bytes_equal=True,observer_file_bytes_equal=True,
            current_prune_record_sha256=sha(self.args.output/'working_prune.jsonl'),dispatch=dispatch,memory=memory))
        if step==9:raise DiagnosticStop()
        return value

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception')
    log=None
    try:
        assert len(ADAPTER_SHA)==64,'Freeze reviewed adapter before execution'
        reference_gate(args)
        ctx=w.initialize(args)
        original_identity,reference_math,reference_files=reference_gate(args,ctx.identity)
        assert not ctx.hybrid.receipts
        adapter=load('working_eager_segments_adapter',HERE/'working_eager_segments.py',ADAPTER_SHA)
        from flowstar_gpu import graphing
        log=(args.output/'working_eager.jsonl').open('w')
        def record(row):log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
        ctx.eager_segments=adapter.install(ctx.torch,graphing,record=record)
        ctx.identity=dict(ctx.identity,script_sha256=sha(__file__),policy=POLICY,
            algorithm_identity=original_identity,working_eager_adapter_sha256=ADAPTER_SHA,
            working_eager_policy=ctx.eager_segments.policy,diagnostic_only=True,
            limitation='Five working segments execute the original fn on original inputs; all other graphs and GLUE=graph arithmetic/storage branches remain. Nine paired steps only; reference itself failed its full40 by resource guard. No own40/cold, full trajectory or speed qualification.')
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode=args.mode,driver_argv=w.driver_argv(ctx),
            reference_files_sha256=reference_files,reference_watch_sha256=REFERENCE_WATCH_SHA,
            reference_failed=True,torch_cap_bytes=11*1024**3,external_guard_bytes=int(11.5*1024**3)))
        x=Runtime(ctx,args,reference_math)
        argv=sys.argv;sys.argv=w.driver_argv(ctx)
        try:
            returned=ctx.d.main();result=dict(status='original_driver_returned_before_diagnostic_stop',driver_return=returned)
        except DiagnosticStop:
            assert x.completed==9 and x.reference_steps==list(range(1,10))
            assert x.observer_endpoint64_calls==9 and x.ledger.length==x.sr.qlen==x.sr.jlen==9
            w.w.counters_gate(ctx.valid128.counters,9);w.b.counters_gate(ctx.working_prune.counters,9)
            result=dict(status='diagnostic_nine_steps_bytes_equal',reference_steps=x.reference_steps,
                accepted_lanes_by_step=[row['accepted_lanes'] for row in x.rows],
                diagnostic_only=True,no_timing_or_execution_qualification=True)
        finally:sys.argv=argv
        for name,digest in reference_files.items():assert sha(args.reference/name)==digest
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),source_entry_sha256=ENTRY_SHA,adapter_sha256=ADAPTER_SHA,process_s=time.perf_counter()-start)
        if (args.output/'INPUT.json').exists():result['input_sha256']=sha(args.output/'INPUT.json')
        if 'ctx' in locals():
            result.update(source_identity=ctx.identity,max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),
                max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved(),valid128_counters=dict(ctx.valid128.counters),
                weighted128_counters=dict(ctx.weighted128.counters),working_graph_counters=dict(ctx.working_prune.counters),
                working_graph_checks=list(ctx.working_prune_rows))
            if hasattr(ctx,'eager_segments'):
                result['working_eager_counters']=dict(ctx.eager_segments.counters)
                ctx.eager_segments.restore();result['working_eager_restored']=True
            if log is not None:log.close();result['working_eager_log_sha256']=sha(args.output/'working_eager.jsonl')
            assert ctx.d.rows_range_over_time_sparse is ctx.observer_original_range
            result['observer_driver_function_unchanged']=True
            ctx.horner_cleanup.restore();result['horner_cleanup_restored']=True
            ctx.working_prune.restore();ctx.working_prune_log.close();result['working_graph_hook_restored']=True
            ctx.valid128.restore();result['valid_adapter_restored']=True
            ctx.postwarm.restore();ctx.postwarm_log.close();result['cache_release_adapter_restored']=True
            ctx.weighted128.restore();result['weighted_adapter_restored']=True
        if 'x' in locals():result.update(completed_advance_count=x.completed,phase=x.phase,observer_endpoint64_calls=x.observer_endpoint64_calls)
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['diagnostic_nine_steps'],default='diagnostic_nine_steps')
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker',
        'working-prune-reference','working-prune-check','horner-cleanup-check','horner-cleanup-checker','observer-check','observer-checker',
        'reference','reference-watch','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.source40=None;main(a)
