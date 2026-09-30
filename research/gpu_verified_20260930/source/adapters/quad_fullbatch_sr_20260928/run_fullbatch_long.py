"""Full1024/P2 original50-period run, admitted by its own bounded40 and cold.

The frozen short runner and numerical adapters are unchanged. Copied short
methods differ only in execution limits and checkpoint scheduling.
This remains conditional on unqualified CROWN/same-slope certificates.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback

HERE=Path(__file__).parent
SHORT_SHA='002e38983c0b5a5c54313a96a61bee0e65f4ccff9e9823f9296de5010c21f304'
POLICY='original_full1024_P2_directNN_reciprocal5_hostK20_original1000'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def frozen():
    path=HERE/'run_fullbatch.py';assert sha(path)==SHORT_SHA
    spec=importlib.util.spec_from_file_location('fullbatch_long_frozen_short',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

w=frozen()
read,write=w.read,w.write


def admission(args,short_identity):
    assert len(args.source40_result_sha256)==len(args.cold_result_sha256)==64
    assert sha(args.source40/'RESULT.json')==args.source40_result_sha256
    assert sha(args.cold/'RESULT.json')==args.cold_result_sha256
    links=w.own40_gate(args.source40,short_identity)
    inp=read(args.cold/'INPUT.json');r=read(args.cold/'RESULT.json')
    assert inp['source_identity']==r['source_identity']==short_identity
    assert r['script_sha256']==SHORT_SHA and r['input_sha256']==sha(args.cold/'INPUT.json')
    assert inp['mode']==r['mode']=='cold20_to21' and r['status']=='passed'
    assert inp['original_target_steps']==r['original_target_steps']==1000
    assert inp['budget_steps']==r['budget_steps']==40
    assert Path(inp['source40']).resolve()==args.source40.resolve()
    for key,digest in links.items():assert r[key]==digest
    assert r['completed_step']==21 and r['accepted_lane_steps']==1024 and r['all_steps_all_lanes_accepted'] is True
    assert r['controller_calls']==1 and r['controller_refresh_steps']==[20]
    states=['committed_20','after_endpoint_20','after_controller_20','committed_21']
    assert set(r['snapshots'])==set(states)
    assert set(r['checks'])==set(states+['endpoint_20','controller_20','transfer_20','observer_21'])
    for name in states:
        assert r['checks'][name]==dict(plant_all_fields=True,SR_and_host_all_components=True,progress=True)
        assert sha(args.cold/name/'MANIFEST.json')==r['snapshots'][name]
    for name in ['endpoint_20','controller_20','transfer_20','observer_21']:assert r['checks'][name] is True
    return dict(source40_path=str(args.source40),source40_result_sha256=args.source40_result_sha256,
        source40_input_sha256=links['source_input_sha256'],cold_path=str(args.cold),cold_result_sha256=args.cold_result_sha256,
        cold_input_sha256=sha(args.cold/'INPUT.json'),qualified_short_runner_sha256=SHORT_SHA)



class LongRuntime(w.Runtime):
    def crown_bounds(self,model,config,lb,ub,**kw):
        assert 0<=self.completed<1000 and self.completed%20==0 and self.completed not in self.controller_steps
        assert kw=={'input_layout':'native'} and lb.shape==ub.shape==(1024,12)
        assert self.phase==('committed_before_endpoint_handoff' if self.completed==0 else 'after_strict_endpoint')
        self.sync();t=time.perf_counter();T,L,U=self.raw['crown_bounds'](model,config,lb,ub,**kw);self.sync();self.times['controller_s']+=time.perf_counter()-t
        assert T.shape==(1024,3,12) and L.shape==U.shape==(1024,3)
        self.controller_calls+=1;self.controller_steps.append(self.completed)
        value=dict(input_lb=lb,input_ub=ub,T=T,L=L,U=U)
        self.nn_signature=self.signature(value);self.artifact('controller_'+str(self.completed),value)
        return T,L,U
    def inject_controls_s(self,st,T,L,U,u_ids,nn_in):
        assert st is self.st and self.controller_steps[-1]==self.completed and nn_in==12 and list(u_ids)==[13,14,15]
        assert self.signature(dict(T=T,L=L,U=U))=={k:self.nn_signature[k] for k in ['T','L','U']},'native-f64 transport changed NN certificate'
        t=time.perf_counter();before=self.protected(physical=True)
        receipt=self.ctx.injection.inject_controls_s(st,T,L,U,u_ids,nn_in,eng=self.eng)
        assert self.protected(physical=True)==before
        self.artifact('transfer_'+str(self.completed),receipt)
        self.held=self.completed;self.held_signature=self.nn_signature;self.phase='after_controller'
        self.sync();self.times['boundary_s']+=time.perf_counter()-t
    def advance_core(self,st,code,eng,sched,settings,cap,sr):
        assert st is self.st and eng is self.eng and cap is self.cap and sr is self.sr
        assert self.completed<1000 and self.phase in ['after_controller','committed_before_endpoint_handoff']
        self.pending_status=st.status.clone();self.propagate_count=0;self.sync();t=time.perf_counter()
        try:
            state,ok=self.raw['advance_sparse'](st,code,eng,sched,settings,cap,sr)
            assert self.propagate_count==1
            self.ledger.finish_step(sr,state.status,token=self.completed+1)
        finally:self.pending_status=None
        self.sync();duration=time.perf_counter()-t;self.times['advance_s']+=duration
        self.completed+=1;self.st=state;self.phase='committed_before_endpoint_handoff';self.ok=ok.detach().cpu().clone()
        assert self.ok.shape==(1024,) and self.ok.dtype==self.t.bool
        self.broken|=(~self.ok)&(state.status.cpu()!=self.d.ACTIVE)
        self.last_accepted[self.ok]=self.completed
        self.observe(duration)
        allocated=self.t.cuda.memory_allocated();reserved=self.t.cuda.memory_reserved()
        if reserved-allocated>2*w.GIB:
            t=time.perf_counter();self.t.cuda.empty_cache();self.sync();assert self.t.cuda.memory_allocated()==allocated
            elapsed=time.perf_counter()-t;self.times['cache_release_s']+=elapsed
            self.cache_releases.append(dict(step=self.completed,allocated=allocated,reserved_before=reserved,reserved_after=self.t.cuda.memory_reserved(),elapsed_s=elapsed))
        return state,ok
    def __init__(self,ctx,args):
        super().__init__(ctx,args)
        self.first_failure=None;self.failure_pending=False;self.failure_snapshot_attempted=False
        self.snapshot_errors=[];self.boundary_guard_checks=0
    def sr_signature(self):
        # Boundary helpers have no SR/ledger arguments. The actual own40/cold
        # gate compared all bytes; long-run guards avoid repeatedly D2H-hashing
        # multi-GiB history. These are mutation sentinels, NOT all-byte audits.
        views=self.ledger._snapshot_views(self.sr);guard={}
        for key,value in views.items():
            if isinstance(value,self.t.Tensor):
                guard[key]=(value.untyped_storage().data_ptr(),value.storage_offset(),
                    tuple(value.shape),tuple(value.stride()),str(value.dtype),str(value.device),value._version)
            else:guard[key]=value
        self.boundary_guard_checks+=1
        return guard
    def plant_record(self):
        value={key:getattr(self.st,key) for key in ['pre','pre_rem','tmv','tmv_rem','status']}
        value.update(cap=self.cap,pre_ids=list(self.st.pre_sup.ids),tmv_ids=list(self.st.tmv_sup.ids),
            pre_exponents=self.ctx.snapshot.exponents(self.eng,self.st.pre_sup),tmv_exponents=self.ctx.snapshot.exponents(self.eng,self.st.tmv_sup),
            progress=self.progress(),sr_qlen=self.sr.qlen,sr_jlen=self.sr.jlen,host_epoch=self.ledger.epoch,host_length=self.ledger.length,
            scope='Plant/support/cap/progress only; history is only present in an explicit streaming snapshot.')
        return value
    def capture_failure(self,location):
        if not self.failure_pending or self.failure_snapshot_attempted:return
        self.failure_snapshot_attempted=True
        name='first_failure_finished_'+str(self.completed)
        self.first_failure['capture_location']=location
        self.first_failure['snapshot_name']=name
        self.artifact('first_failure_plant',self.plant_record())
        began=time.perf_counter()
        try:
            self.save(name)
            self.first_failure['snapshot_status']='saved_finished_SR_original_poststep'
        except (ValueError,AssertionError) as exc:
            # Finite serializer must never sanitize dead-lane NaNs/Inf. Keep
            # the original loop outcome and explicitly record unsupported dump.
            self.times['checkpoint_s']+=time.perf_counter()-began
            self.first_failure['snapshot_status']='not_saved_finite_snapshot_contract'
            self.snapshot_errors.append(dict(name=name,error_type=type(exc).__name__,error=str(exc)))
        self.failure_pending=False
    def advance_sparse(self,*args,**kwargs):
        self.capture_failure('next_original_advance_entry_after_prior_poststep_reset')
        state,ok=self.advance_core(*args,**kwargs)
        if self.first_failure is None and not bool(ok.all()):
            self.first_failure=dict(step=self.completed,held_controller_step=self.held,
                failed_lanes=(~self.ok).nonzero(as_tuple=True)[0].tolist(),status=state.status.cpu().tolist(),
                scope='Original driver per-lane refusal; remaining lanes continue without changing batch/order.')
            self.failure_pending=True
        return state,ok
    def end_of_time_s(self,st,eng):
        assert st is self.st and eng is self.eng and 0<self.completed<=1000 and self.completed%20==0
        assert self.phase=='committed_before_endpoint_handoff'
        self.capture_failure('original_endpoint_entry_after_poststep_reset')
        if self.completed==1000:
            assert self.sr.qlen==self.sr.jlen==self.ledger.length==0 and self.ledger.epoch==1000
            assert not bool(self.ledger.last_rebuilt_lane.any())
            self.save('committed_1000')
        t=time.perf_counter();before=self.protected();error=self.ctx.endpoint.end_of_time_s(st,eng)
        assert self.protected()==before
        self.artifact('endpoint_'+str(self.completed),dict(error=error));self.phase='after_strict_endpoint'
        self.sync();self.times['boundary_s']+=time.perf_counter()-t
        if self.completed==1000:self.save('after_endpoint_1000')
        # Return to original main, including final target evaluation. No
        # TargetStop/BudgetStop, extra NN refresh, or new property semantics.


def initialize(args):
    ctx=w.initialize(args)  # all original CUDA gates/SO checks, before engines
    short_identity=dict(ctx.identity);gate=admission(args,short_identity)
    ctx.identity=dict(short_identity,script_sha256=sha(__file__),policy=POLICY,
        qualified_short_identity=short_identity,execution_budget_steps=1000,long_admission=gate,
        checkpoint_policy='first_failure_finished_once_then_reset1000_before_and_after_final_endpoint',
        boundary_SR_guard='metadata_storage_shape_dtype_version_sentinels; fullbytes_qualified_in_own40_cold',
        limitation='Original full1024/P2 trajectory with conditional CROWN/same-slope certificates. Per-lane failure is retained;1000 overall completion requires every lane accepted every step. Final original target output remains conditional.')
    return ctx


class DriverOutput:
    """Tee unchanged original output; identify only exact original verdict lines."""
    def __init__(self,stream,path):self.stream=stream;self.file=Path(path).open('w');self.pending='';self.verdicts=[]
    def write(self,text):
        self.stream.write(text);self.file.write(text);self.pending+=text
        while '\n' in self.pending:
            line,self.pending=self.pending.split('\n',1)
            if line.strip() in ['VERIFIED','FALSIFIED','UNKNOWN','Unsafe','Unsafe.','Unknown.']:self.verdicts.append(line.strip())
        return len(text)
    def flush(self):self.stream.flush();self.file.flush()
    def close(self):self.flush();self.file.close()


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();result=dict(status='exception',mode='long1000')
    try:
        ctx=initialize(args);x=LongRuntime(ctx,args)
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode='long1000',original_target_steps=1000,budget_steps=1000,
            driver_argv=ctx.boot.driver_argv(ctx),source40=str(args.source40),cold=str(args.cold)))
        argv,stdout=sys.argv,sys.stdout;tee=DriverOutput(stdout,args.output/'driver.stdout.txt')
        sys.argv=ctx.boot.driver_argv(ctx);sys.stdout=tee
        try:returned=ctx.d.main()
        finally:sys.argv=argv;sys.stdout=stdout;tee.close()
        x.capture_failure('original_driver_return_after_original_abort_without_extra_reset')
        t=time.perf_counter();x.artifact('terminal_plant',x.plant_record());x.times['checkpoint_s']+=time.perf_counter()-t
        all_accepted=x.completed==1000 and len(x.rows)==1000 and all(all(r['accepted']) for r in x.rows)
        if all_accepted:
            assert x.controller_calls==50 and x.controller_steps==list(range(0,1000,20))
            assert x.phase=='after_strict_endpoint' and x.ledger.epoch==1000 and x.ledger.length==0
            assert {'committed_1000','after_endpoint_1000'}<=set(x.snapshots)
        if x.completed==1000:
            status='target_horizon_completed_all_lanes' if all_accepted else 'horizon_visited_with_failed_lanes'
        else:status='original_driver_stopped_before_target'
        result=dict(status=status,mode='long1000',driver_return=returned,original_driver_verdict_lines=tee.verdicts,
            overall_horizon_completed=all_accepted,original_target_steps=1000,budget_steps=1000,completed_step=x.completed,
            accepted_lane_steps=sum(r['accepted_lanes'] for r in x.rows),all_steps_all_lanes_accepted=all_accepted,
            controller_calls=x.controller_calls,controller_refresh_steps=x.controller_steps,
            first_failure=x.first_failure,snapshot_errors=x.snapshot_errors,snapshots=x.snapshots,
            final_progress=x.progress(),timing_components=x.times,cache_releases=x.cache_releases,
            boundary_SR_sentinel_checks=x.boundary_guard_checks,
            end_to_end_strict_certificate=False,source_identity=ctx.identity)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),overall_horizon_completed=False)
        if 'x' in locals():
            # An unfinished propagate is never serialized as a committed state.
            result.update(completed_step=x.completed,phase=x.phase,first_failure=x.first_failure,
                snapshots=x.snapshots,snapshot_errors=x.snapshot_errors,pending_SR_token=x.ledger.pending if x.ledger else None,
                source_identity=ctx.identity,timing_components=x.times,cache_releases=x.cache_releases)
        raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-started)
        if (args.output/'INPUT.json').exists():result['input_sha256']=sha(args.output/'INPUT.json')
        if 'ctx' in locals():result.update(max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved())
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check','host-capacity-check','snapshot-check','output','source40','cold']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source40-result-sha256',required=True);p.add_argument('--cold-result-sha256',required=True)
    main(p.parse_args())
