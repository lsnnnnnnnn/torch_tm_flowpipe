"""Paired reference2/candidate3 probe of selective working-graph retention.

Original full1024 Runtime and driver calls remain. Only hashes of complete
active mathematical state/SR/host history are added, never a full SR dump.
The stop follows prune, before the driver's post-step safety/reset block.
Candidate step3 has no completed reference step3 and is not a parity claim.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
ENTRY_SHA='60df1ef31d6491e4d436a2724b3f9ef529c1432baeae0e4de071da2b5a4b8b31'
ADAPTER_SHA='3a174e02724ee3166eff9f1b330c137e0c1a4b51f64a6162ea2a6ed96814e21f'
POLICY='paired_original_reference2_selective_working_graph_candidate3'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,path,digest):
    assert sha(path)==digest
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
w=load('working_graph_probe_frozen_entry',HERE/'run_fullbatch_p3_valid128.py',ENTRY_SHA)
class DiagnosticStop(Exception):pass

def support(s):return dict(n=s.n,k=s.k,spatial=s.spatial,ids=list(s.ids))

def mathematical_signature(runtime):
    state={name:getattr(runtime.st,name) for name in ['pre','pre_rem','tmv','tmv_rem','status']}
    return runtime.signature(dict(state=state,pre_support=support(runtime.st.pre_sup),tmv_support=support(runtime.st.tmv_sup),
        cap=runtime.cap,accepted=runtime.ok,progress=runtime.progress(),
        active_SR_and_host_history=runtime.ledger._snapshot_views(runtime.sr)))

def storage_signature(runtime):
    """In-process pointer/capacity invariance only, never cross-process parity."""
    def tensor(value):
        if value is None:return None
        return dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),device=str(value.device),
            storage_ptr=value.untyped_storage().data_ptr(),storage_bytes=value.untyped_storage().nbytes(),
            storage_offset=value.storage_offset(),version=value._version)
    return dict(sr_max_size=runtime.sr.max_size,host_max_size=runtime.ledger.max_size,
        sr_capacity=runtime.sr.phi_buf.shape[0],qlen=runtime.sr.qlen,jlen=runtime.sr.jlen,
        state={name:tensor(getattr(runtime.st,name)) for name in ['pre','pre_rem','tmv','tmv_rem','status']},
        cap=tensor(runtime.cap),sr={name:tensor(getattr(runtime.sr,name)) for name in runtime.h.SR_FIELDS},
        host={name:tensor(getattr(runtime.ledger,name)) for name in ['factors','point_finite','factor_valid','status_before',
            'eligible','replaced','fallback','current_status','last_rebuilt_lane']})

def reference_gate(path,algorithm_identity):
    path=Path(path);result=read(path/'RESULT.json');inp=read(path/'INPUT.json')
    assert result['status']=='diagnostic_reference_two_steps' and result['mode']==inp['mode']=='reference2'
    assert result['script_sha256']==sha(__file__) and result['input_sha256']==sha(path/'INPUT.json')
    assert result['source_identity']==inp['source_identity']
    assert result['source_identity']['algorithm_identity']==algorithm_identity
    assert result['source_entry_sha256']==ENTRY_SHA and result['completed_advance_count']==2
    assert result['accepted_lanes_by_step']==[1024,1024]
    assert result['valid_adapter_restored'] and result['cache_release_adapter_restored'] and result['weighted_adapter_restored']
    assert set(result['digest_files_sha256'])=={'DIGEST_1.json','DIGEST_2.json'}
    files={'RESULT.json':sha(path/'RESULT.json'),'INPUT.json':sha(path/'INPUT.json')}
    for name,digest in result['digest_files_sha256'].items():
        assert sha(path/name)==digest;files[name]=digest
    for step in [1,2]:
        meta=read(path/f'observer_{step}.json')
        assert meta['source_identity']==inp['source_identity'] and meta['step']==step
        assert sha(path/f'observer_{step}.pt')==meta['pt_sha256']
        for suffix in ['json','pt']:
            name=f'observer_{step}.{suffix}';files[name]=sha(path/name)
    return files

class Runtime(w.Runtime):
    def __init__(self,ctx,args,cache_binding=None):
        self.cache_binding=cache_binding;self.digests={};self.reference_matches=[];self.evictions=[]
        super().__init__(ctx,args)
    def advance_sparse(self,st,code,eng,sched,settings,cap,sr):
        assert self.completed<(2 if self.args.mode=='reference2' else 3)
        if self.cache_binding is not None:self.cache_binding.begin_step(eng)
        return super().advance_sparse(st,code,eng,sched,settings,cap,sr)
    def prune_state(self,*args,**kwargs):
        value=super().prune_state(*args,**kwargs);step=self.completed
        assert 1<=step<=(2 if self.args.mode=='reference2' else 3)
        before=mathematical_signature(self);storage_before=storage_signature(self)
        row=dict(step=step,source_identity=self.ctx.identity,mathematical_signature=before,
            phase='after_original_prune_before_poststep_safety_reset',SR_not_dumped=True)
        if self.cache_binding is not None:
            eviction=self.cache_binding.finish_step(self.eng)
            after=mathematical_signature(self);storage_after=storage_signature(self)
            assert after==before,'Cache pruning changed active mathematical state'
            assert storage_after==storage_before,'Cache pruning changed state/SR/host capacity or storage'
            row.update(before_after_mathematical_bytes_equal=True,capacity_and_storage_unchanged=True,eviction=eviction)
            self.evictions.append(eviction)
        row.update(storage_before=storage_before)
        if self.args.mode=='candidate3' and step<=2:
            reference=read(self.args.reference/f'DIGEST_{step}.json')
            assert reference['step']==step and reference['mathematical_signature']==before
            old_meta=read(self.args.reference/f'observer_{step}.json')
            new_meta=read(self.args.output/f'observer_{step}.json')
            for directory,meta in [(self.args.reference,old_meta),(self.args.output,new_meta)]:
                path=directory/f'observer_{step}.pt';assert sha(path)==meta['pt_sha256']
                actual=self.t.load(path,map_location='cpu',weights_only=True)
                assert self.signature(actual)==meta['signature']
            assert old_meta['signature']==new_meta['signature']
            row.update(reference_mathematical_bytes_equal=True,reference_observer_all_tensor_bytes_equal=True)
            self.reference_matches.append(step)
        path=self.args.output/f'DIGEST_{step}.json';write(path,row);self.digests[path.name]=sha(path)
        if step==(2 if self.args.mode=='reference2' else 3):raise DiagnosticStop()
        return value

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception',mode=args.mode)
    binding=None;reference_files={}
    try:
        ctx=w.initialize(args);algorithm_identity=ctx.identity
        if args.mode=='candidate3':
            reference_files=reference_gate(args.reference,algorithm_identity)
            assert len(ADAPTER_SHA)==64,'Working-graph candidate is not frozen'
            adapter=load('paired_working_graph_candidate',HERE/'working_graph_prune.py',ADAPTER_SHA)
            from flowstar_gpu import graphing
            binding=adapter.install(ctx.torch,graphing)
        ctx.identity=dict(algorithm_identity,script_sha256=sha(__file__),policy=POLICY,probe_mode=args.mode,
            algorithm_identity=algorithm_identity,source_entry_sha256=ENTRY_SHA,
            working_graph_adapter_sha256=ADAPTER_SHA if binding is not None else None,
            diagnostic_only=True,fullbatch_qualification=False,
            limitation='Hash-only active state/SR preservation and two-step paired diagnostic. Candidate step3 is unpaired. Stops after prune before poststep safety/reset; not a committed checkpoint, own40/cold or speed result.')
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode=args.mode,driver_argv=w.driver_argv(ctx),
            reference_files_sha256=reference_files,torch_cap_bytes=11*1024**3,external_guard_bytes=int(11.5*1024**3)))
        runtime=Runtime(ctx,args,binding)
        argv=sys.argv;sys.argv=w.driver_argv(ctx)
        try:
            returned=ctx.d.main();result.update(status='original_driver_returned_before_diagnostic_stop',driver_return=returned)
        except DiagnosticStop:
            assert runtime.completed==(2 if args.mode=='reference2' else 3)
            assert [row['accepted_lanes'] for row in runtime.rows]==[1024]*runtime.completed
            if args.mode=='candidate3':assert runtime.reference_matches==[1,2]
            result.update(status='diagnostic_reference_two_steps' if args.mode=='reference2' else 'diagnostic_candidate_three_steps',
                reference_steps_bytes_equal=runtime.reference_matches,candidate_step3_has_reference=False)
        finally:sys.argv=argv
        for name,digest in reference_files.items():assert sha(args.reference/name)==digest
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),source_entry_sha256=ENTRY_SHA,process_s=time.perf_counter()-start,
            diagnostic_only=True,no_timing_or_execution_qualification=True)
        if (args.output/'INPUT.json').exists():result['input_sha256']=sha(args.output/'INPUT.json')
        if 'runtime' in locals():result.update(completed_advance_count=runtime.completed,phase=runtime.phase,
            digest_files_sha256=runtime.digests,accepted_lanes_by_step=[row['accepted_lanes'] for row in runtime.rows],
            evictions=runtime.evictions)
        if binding is not None:
            result['working_graph_counters']=dict(binding.counters);binding.restore();result['working_graph_hook_restored']=True
        if 'ctx' in locals():
            result.update(source_identity=ctx.identity,max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),
                max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved(),valid128_counters=dict(ctx.valid128.counters),
                weighted128_counters=dict(ctx.weighted128.counters),postwarm_releases=list(ctx.postwarm.releases))
            ctx.valid128.restore();result['valid_adapter_restored']=True
            ctx.postwarm.restore();ctx.postwarm_log.close();result['cache_release_adapter_restored']=True
            ctx.weighted128.restore();result['weighted_adapter_restored']=True
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['reference2','candidate3'],required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check',
        'host-capacity-check','snapshot-check','p3-snapshot-check','hybrid-cpu-check','metadata-backend','metadata-check','hybrid-check',
        'weighted-check','weighted-checker','cache-release-check','cache-release-checker','valid-check','valid-checker','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference',type=Path);args=parser.parse_args();args.source40=None
    assert (args.reference is not None)==(args.mode=='candidate3');main(args)
