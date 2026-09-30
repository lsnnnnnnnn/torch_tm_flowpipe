"""Same full-B run to800: capture the missing real799 parent and matched800.

Original main, NN, status/prune/reset and all numerical methods are retained.
Reference observer PTs/SR blocks are not reread: compare their bound sidecars
and manifest raw hashes, explicitly distinct from independent byte reauditing.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback
HERE=Path(__file__).parent
LONG_SHA='7b465c72046b4c23497ace95f4b3d74cba8fd556cbb48a5454ae0af1b3772c98'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def load():
    p=HERE/'run_fullbatch_long.py';assert sha(p)==LONG_SHA
    spec=importlib.util.spec_from_file_location('capture_frozen_long',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
l=load();read,write=l.read,l.write


def reference_gate(args,identity):
    source=args.reference_long;inp=read(source/'INPUT.json');done=read(source/'RESULT.json')
    assert sha(source/'RESULT.json')==args.reference_result_sha256
    assert done['input_sha256']==sha(source/'INPUT.json') and done['script_sha256']==LONG_SHA
    assert inp['source_identity']==done['source_identity']==identity
    assert inp['mode']==done['mode']=='long1000'
    assert done['status']=='original_driver_stopped_before_target' and done['overall_horizon_completed'] is False
    assert done['original_target_steps']==inp['original_target_steps']==1000 and done['completed_step']==813
    assert done['first_failure']['step']==800 and len(done['first_failure']['failed_lanes'])==12
    rows=[json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines() if line.strip()]
    assert [v['step'] for v in rows]==list(range(1,814))
    assert all(len(v['accepted'])==len(v['status'])==1024 and all(type(b) is bool for b in v['accepted']) for v in rows)
    assert all(all(v['accepted']) for v in rows[:799]) and sum(rows[799]['accepted'])==1012 and not any(rows[-1]['accepted'])
    assert sum(sum(v['accepted']) for v in rows)==done['accepted_lane_steps']
    path=source/'first_failure_finished_800';m=read(path/'MANIFEST.json')
    assert done['snapshots']['first_failure_finished_800']==sha(path/'MANIFEST.json')
    assert m['metadata']==identity and m['serializer_sha256']==l.w.PINS['snapshot.py']
    assert m['progress']['completed_step']==800 and m['progress']['held_controller_step']==780
    assert m['progress']['phase']=='committed_before_endpoint_handoff' and m['progress']['sr_history_origin_step']==0
    assert sha(path/'sr/MANIFEST.json')==m['files']['sr/MANIFEST.json']
    sr=read(path/'sr/MANIFEST.json');assert sr['adapter_sha256']==l.w.HOST_SHA
    assert sr['metadata']['length']==sr['metadata']['qlen']==sr['metadata']['jlen']==800 and sr['metadata']['epoch']==0
    assert sr['metadata']['batch']==1024 and sr['metadata']['n']==16 and sr['metadata']['sr_capacity']==1000
    plant=read(source/'first_failure_plant.json')
    assert plant['source_identity']==identity and plant['step']==800 and len(plant['pt_sha256'])==64
    assert plant['signature']['progress']==m['progress']
    signatures={};bindings={}
    for k in range(1,801):
        p=source/f'observer_{k}.json';v=read(p)
        assert v['source_identity']==identity and v['step']==k and len(v['pt_sha256'])==64
        assert set(v['signature'])=={'bounds','accepted','status'}
        for name,shape,dtype in [('bounds',[1024,12,4],'torch.float64'),('accepted',[1024],'torch.bool'),('status',[1024],'torch.int8')]:
            assert v['signature'][name]['shape']==shape and v['signature'][name]['dtype']==dtype and len(v['signature'][name]['sha256'])==64
        signatures[k]=v['signature'];bindings[str(k)]=sha(p)
    certificate=read(source/'controller_780.json');transfer=read(source/'transfer_780.json')
    for v in [certificate,transfer]:assert v['source_identity']==identity and v['step']==780
    assert m['progress']['held_controller_signature']==certificate['signature']
    files={str(p.relative_to(source)):sha(p) for p in [source/'INPUT.json',source/'RESULT.json',source/'steps.jsonl',path/'MANIFEST.json',path/'sr/MANIFEST.json',source/'first_failure_plant.json',source/'controller_780.json',source/'transfer_780.json']}
    return dict(source=source,identity=identity,rows=rows,manifest=m,sr=sr,plant=plant,signatures=signatures,
        receipt=dict(reference_long=str(source),reference_files_sha256=files,observer_sidecar_sha256=bindings,
            reference_observer_PTs_read=False,reference_SR_component_blocks_read=False,reference_plant_PT_read=False,
            scope='Actual reference INPUT/RESULT,step logs,observer numerical signatures,and bound800 component manifest declarations; no independent source PT/block rehash.'))


class CaptureStop(Exception):pass

class CaptureRuntime(l.LongRuntime):
    def __init__(self,ctx,args,reference):
        super().__init__(ctx,args);self.reference=reference;self.observer_equal=[];self.capture_comparisons={}
    def observe(self,duration):
        super().observe(duration)
        assert self.completed<=800
        actual=read(self.args.output/f'observer_{self.completed}.json')
        assert actual['signature']==self.reference['signatures'][self.completed],('observer numerical signature changed',self.completed)
        row=self.rows[-1];old=self.reference['rows'][self.completed-1]
        for key in ['step','accepted','status','accepted_lanes','broken_lanes','sr_length','host_epoch']:assert row[key]==old[key],(self.completed,key)
        self.observer_equal.append(self.completed)
    def advance_sparse(self,*args,**kwargs):
        assert self.completed<800
        if self.completed==799:
            assert self.held==780 and self.sr.qlen==self.sr.jlen==self.ledger.length==799 and self.ledger.epoch==0
            assert len(self.observer_equal)==799 and not bool(self.st.status.any()) and self.first_failure is None
            # Record the actual small layout accumulator for future cold prune;
            # no cache is changed here, and it equals the saved supports.
            supports=self.eng._acc_sups[('state',)]
            assert supports==(self.st.pre_sup,self.st.tmv_sup)
            self.save('committed_799')
            write(self.args.output/'PRUNE_SUPPORT_799.json',dict(source_identity=self.ctx.identity,completed_step=799,
                source_sparse_exec_sha256=self.ctx.identity['engine']['python_sha256']['src/flowstar_gpu/sparse_exec.py'],
                accumulator_key=['state'],supports=[dict(n=s.n,k=s.k,spatial=s.spatial,ids=list(s.ids)) for s in supports],
                snapshot_manifest_sha256=self.snapshots['committed_799']))
        return super().advance_sparse(*args,**kwargs)
    def end_of_time_s(self,st,eng):
        if self.completed!=800:return super().end_of_time_s(st,eng)
        assert st is self.st and eng is self.eng and self.phase=='committed_before_endpoint_handoff'
        assert self.held==780 and self.controller_calls==40 and self.controller_steps==list(range(0,800,20))
        assert self.sr.qlen==self.sr.jlen==self.ledger.length==800 and self.ledger.epoch==0
        assert self.first_failure['step']==800 and len(self.first_failure['failed_lanes'])==12
        assert self.observer_equal==list(range(1,801))
        # At this exact original main location prune/postchecks/reset_if_full
        # have run;800<1000 means original reset_if_full returned false.
        self.save('committed_800');self.artifact('captured_plant_800',self.plant_record())
        mine=read(self.args.output/'committed_800/MANIFEST.json');sr=read(self.args.output/'committed_800/sr/MANIFEST.json')
        assert mine['metadata']==self.ctx.identity and self.reference['manifest']['metadata']==self.reference['identity']
        assert mine['progress']==self.reference['manifest']['progress']
        assert sr['metadata']==self.reference['sr']['metadata'] and set(sr['tensors'])==set(self.reference['sr']['tensors'])
        for name,field in sr['tensors'].items():
            old=self.reference['sr']['tensors'][name]
            for key in ['shape','dtype','raw_sha256']:assert field[key]==old[key],(name,key)
        assert read(self.args.output/'captured_plant_800.json')['signature']==self.reference['plant']['signature']
        for name in ['controller_780','transfer_780']:
            assert read(self.args.output/(name+'.json'))['signature']==read(self.reference['source']/(name+'.json'))['signature']
        self.capture_comparisons=dict(observer_numerical_signatures_equal_steps=800,step_status_masks_equal_steps=800,
            plant800_saved_signature_equal=True,progress800_equal=True,SR_ledger800_manifest_raw_hashes_equal=True,
            SR_ledger_component_count=len(sr['tensors']),held780_controller_and_transfer_signatures_equal=True,
            reference_large_component_bytes_reread=False)
        # Do not call long super here: that would save a third first-failure
        # history and then execute endpoint800. Neither is part of this probe.
        raise CaptureStop()


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();result=dict(status='exception',mode='capture799_800')
    try:
        ctx=l.initialize(args);reference=reference_gate(args,dict(ctx.identity))
        ctx.identity=dict(ctx.identity,script_sha256=sha(__file__),capture_wrapper_parent_sha256=LONG_SHA,
            execution_budget_steps=800,reference_long_result_sha256=args.reference_result_sha256,
            checkpoint_policy='exactly799parent_and800poststep_before_endpoint;full_original_batch_and_loop')
        x=CaptureRuntime(ctx,args,reference)
        write(args.output/'INPUT.json',dict(source_identity=ctx.identity,mode='capture799_800',original_target_steps=1000,budget_steps=800,
            driver_argv=ctx.boot.driver_argv(ctx),reference=reference['receipt']))
        argv,stdout=sys.argv,sys.stdout;tee=l.DriverOutput(stdout,args.output/'driver.stdout.txt');sys.argv=ctx.boot.driver_argv(ctx);sys.stdout=tee
        try:
            ctx.d.main();raise AssertionError('Original main stopped before requested matched800 capture')
        except CaptureStop:
            assert set(x.snapshots)=={'committed_799','committed_800'} and len(x.capture_comparisons)>0
        finally:sys.argv=argv;sys.stdout=stdout;tee.close()
        result=dict(status='passed_capture799_and_matched800',mode='capture799_800',completed_step=x.completed,
            original_target_steps=1000,budget_steps=800,overall_horizon_completed=False,
            accepted_lane_steps=sum(r['accepted_lanes'] for r in x.rows),first_failure=x.first_failure,
            controller_calls=x.controller_calls,controller_refresh_steps=x.controller_steps,no_endpoint800_or_controller800=True,
            comparisons=x.capture_comparisons,snapshots=x.snapshots,prune_support799_sha256=sha(args.output/'PRUNE_SUPPORT_799.json'),
            source_identity=ctx.identity,reference=reference['receipt'],timing_components=x.times,cache_releases=x.cache_releases,
            limitation='Actual799 is the replay parent;800 is a mixed failed poststep state. Saved reference signature/hash declarations matched; source large PT/SR blocks were not reread. CROWN remains conditional.')
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if 'x' in locals():result.update(completed_step=x.completed,snapshots=x.snapshots,source_identity=ctx.identity,observer_equal_through=max(x.observer_equal,default=0))
        raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-started)
        if (args.output/'INPUT.json').exists():result['input_sha256']=sha(args.output/'INPUT.json')
        if 'ctx' in locals():result.update(max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved())
        write(args.output/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check','host-capacity-check','snapshot-check','output','source40','cold','reference-long']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['source40-result-sha256','cold-result-sha256','reference-result-sha256']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
