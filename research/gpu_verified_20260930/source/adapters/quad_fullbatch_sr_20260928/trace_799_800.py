"""Read-only full1024 step800 trace, usable only after complete bare/output parity.

Only saved diagnostics select lanes. The original advance, SR factory,
propagation, ledger, validation chunks, prune and reset retain their batch.
The callback can change refinement dispatch; parity is a mandatory result gate.
"""
from pathlib import Path
from dataclasses import replace
import argparse, copy, hashlib, importlib.util, inspect, json, math, sys, time, traceback

HERE=Path(__file__).parent
COLD_SHA='d7284d93b7944ed21d3b1246aa41e2313d8bee132df908b19c0b0db7886b056e'
FAILED=[497,609,625,737,753,849,865,881,961,977,993,1009]
SELECTED=sorted(FAILED+[496,608])


def cold_helper():
    path=HERE/'cold_799_800.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==COLD_SHA
    spec=importlib.util.spec_from_file_location('fullbatch800_trace_cold',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def safe(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [safe(v) for v in value]
    return value


def main(a,c):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    result=dict(status='exception',trace_usable=False,mode='trace799_to800')
    hooks=[];events=[];proposals=[];context=[];holder={};settings_calls=[]
    def hook(module,name,value):
        hooks.append((module,name,getattr(module,name)));setattr(module,name,value)
    def record(event):
        if int(event.get('lane',-1)) in SELECTED:events.append(copy.deepcopy(event))
    try:
        ctx=c.l.initialize(a);baseline=dict(ctx.identity);source=c.admission(a,baseline)
        assert source['last']['accepted']==[i not in FAILED for i in range(1024)]
        bare=c.read(a.bare/'RESULT.json');bi=c.read(a.bare/'INPUT.json')
        expected_bare=dict(baseline,script_sha256=COLD_SHA,cold_wrapper_parent_sha256=c.LONG_SHA,capture_script_sha256=c.CAPTURE_SHA,
            capture_input_sha256=source['receipt']['capture_input_sha256'],capture_result_sha256=a.capture_result_sha256,
            execution_budget_steps=1,parent_completed_step=799,replayed_step=800,held_controller_step=780,
            checkpoint_policy='restore_only_capture799_once;save_only_replayed800',no_endpoint_or_control_update=True)
        assert c.sha(a.bare/'RESULT.json')==a.bare_result_sha256
        assert bare['status']=='passed' and bare['mode']==bi['mode']=='cold799_to800'
        assert bare['script_sha256']==COLD_SHA and bare['input_sha256']==c.sha(a.bare/'INPUT.json')
        assert bare['source_identity']==bi['source_identity']==expected_bare
        assert bare['source_receipt']==bi['source']==source['receipt'] and bi['source_checkpoint_identity']==source['metadata']
        assert bare['parent_step']==799 and bare['completed_step']==bare['replayed_step']==800
        assert bare['controller_calls']==0 and bare['no_endpoint_or_control_update'] is True and bare['held_controller_step']==780
        assert bare['one_SR_restore_no_parent_deepcopy'] is True and bare['source799_all_component_file_and_raw_hashes_verified'] is True
        assert bare['restored_prune_layout_exactly_recorded'] is True
        assert bare['accepted']==source['last']['accepted'] and bare['status_codes']==source['last']['status']
        assert bare['checks']==dict(all_plant_support_cap_status_bytes=True,all_SR_ledger_component_raw_hashes=True,
            component_count=14,all_saved_component_file_hashes_verified=True,progress_equal=True,metadata_identities_checked_separately=True)
        assert bare['observer']==dict(bounds_all1024x12_bytes=True,accepted_status_bytes=True,source_and_actual_PT_hashes_verified=True)
        assert set(bare['snapshots'])=={'committed_800'}
        assert c.sha(a.bare/'committed_800/MANIFEST.json')==bare['snapshots']['committed_800']
        # The frozen compare below verifies every bare component file, while
        # keeping only one live full SR. No full parent deepcopy is introduced.
        bare_files={n:c.sha(a.bare/n) for n in ['INPUT.json','RESULT.json','committed_800/MANIFEST.json','observer_800.json','observer_800.pt']}
        ctx.identity=dict(baseline,script_sha256=c.sha(__file__),trace_cold_helper_sha256=COLD_SHA,
            capture_script_sha256=c.CAPTURE_SHA,capture_input_sha256=source['receipt']['capture_input_sha256'],
            capture_result_sha256=a.capture_result_sha256,bare_result_sha256=a.bare_result_sha256,
            execution_budget_steps=1,parent_completed_step=799,replayed_step=800,held_controller_step=780,
            diagnostic_lanes=SELECTED,diagnostic_callback_enabled=True,callback_may_change_dispatch=True,
            checkpoint_policy='restore_only_capture799_once;save_only_traced800;compare_capture_and_bare_all_fields',no_endpoint_or_control_update=True)
        c.write(a.output/'INPUT.json',dict(source_identity=ctx.identity,mode='trace799_to800',source=source['receipt'],
            source_checkpoint_identity=source['metadata'],bare_path=str(a.bare),bare_source_files_sha256=bare_files))
        torch=ctx.torch;se=ctx.c.se
        from flowstar_gpu import weighted_validation as wv
        assert se.VALIDATION_POLICY=='solution_plus_one'
        assert c.sha(se.__file__)==baseline['engine']['python_sha256']['src/flowstar_gpu/sparse_exec.py']
        assert c.sha(wv.__file__)==baseline['engine']['python_sha256']['src/flowstar_gpu/weighted_validation.py']
        raw_settings=ctx.d.Settings
        def settings(*values,**kw):
            answer=raw_settings(*values,**kw);assert answer.order==2 and answer.step==.005 and answer.refinement_callback is None
            settings_calls.append(True);assert len(settings_calls)==1
            return replace(answer,refinement_callback=record)
        hook(ctx.d,'Settings',settings)
        raw_runtime=c.ColdRuntime
        def runtime(*values,**kw):
            answer=raw_runtime(*values,**kw);assert not holder;holder['runtime']=answer;return answer
        hook(c,'ColdRuntime',runtime)
        def select(value,local):return value[local].detach().to('cpu',copy=True).contiguous()
        def map_events(label,candidate,image,dimensions,bad):
            assert candidate.shape==image.shape==(1024,16,2) and dimensions.shape==(1024,16) and bad.shape==(1024,)
            take=torch.tensor(SELECTED,device=candidate.device,dtype=torch.long)
            owned=[select(v,take) for v in [candidate,image,dimensions,bad]]
            for i,lane in enumerate(SELECTED):
                record(dict(event=label,lane=lane,input_remainder_vector=owned[0][i].tolist(),proposal_interval=owned[1][i].tolist(),
                    subset_by_component=owned[2][i].tolist(),bad=bool(owned[3][i])))
        raw_retry=se._retry_self_map
        def retry(*values,**kw):
            values=list(values);assert len(values)==9;evaluate=values[7]
            map_events('unweighted_before_retries',values[0],values[1],values[2],values[4])
            def observed(candidate):
                answer=evaluate(candidate)
                map_events('unweighted_retry',candidate,answer[0],answer[1],answer[3]);return answer
            values[7]=observed;answer=raw_retry(*values,**kw)
            map_events('unweighted_after_retries',answer[0],answer[1],answer[2],answer[4]);return answer
        hook(se,'_retry_self_map',retry)

        def proposal(label,values,global_lanes,early=False):
            code,point,sup,initial,isup,vs,cap=values[:7];eng=values[9 if early else 8]
            eligible=values[8 if early else 7]
            assert eng.tables.n==16 and eng.tables.k==code.order==3 and eng.step.delta==.005
            assert point.shape[1]==initial.shape[1]==16 and point.shape[0]==len(global_lanes)
            assert len(set(global_lanes))==len(global_lanes) and all(0<=lane<1024 for lane in global_lanes)
            local=[i for i,lane in enumerate(global_lanes) if lane in SELECTED]
            if not local:return None
            pick=torch.tensor(local,device=point.device,dtype=torch.long)
            exps=eng.tables.exponents[list(sup.ids)].detach().cpu().clone()
            iexps=eng.tables.exponents[list(isup.ids)].detach().cpu().clone()
            data=dict(label=label,global_lanes=[global_lanes[i] for i in local],call_batch=point.shape[0],
                x=select(point,pick),initial=select(initial,pick),cap=select(cap,pick),eligible=select(eligible,pick),
                support_ids=list(sup.ids),initial_ids=list(isup.ids),exponents=exps,initial_exponents=iexps,
                variable_support_ids=[list(v.ids) for v in vs],n=16,k=3,code_order=3,working_order=2,h=.005,step=800,
                scope='Actual inputs/proposals only. Recenter polynomial is not accepted unless original full validation accepts; no new certificate.')
            if early:data['accepted_remainder']=select(values[7],pick)
            return data,pick
        def save_proposal(prepared,answer,kind):
            if prepared is None:return
            data,pick=prepared
            if kind=='recenter':
                data.update(recovered=select(answer[0],pick),returned_point=select(answer[1],pick),returned_remainder=select(answer[2],pick))
            elif kind=='early_weighted':data['returned_remainder']=select(answer[0],pick)
            else:data.update(recovered=select(answer[0],pick),returned_remainder=select(answer[1],pick))
            path=a.output/f'proposal_{len(proposals)}.pt';torch.save(data,path)
            proposals.append(dict(file=path.name,sha256=c.sha(path),label=data['label'],global_lanes=data['global_lanes'],
                call_batch=data['call_batch'],max_support_degree=int(data['exponents'].sum(-1).max()),eligible=data['eligible'].tolist()))
        raw_failed=wv.validate_failed
        def failed(*values,**kw):
            label=context[-1] if context else 'weighted';batch=values[1].shape[0]
            if label=='recenter':
                # Frozen validate_recentered supplies the actual chunk-to-full
                # mapping in local_trace; never infer global IDs from row data.
                callback=kw['trace'];assert callback.__name__=='local_trace'
                assert Path(callback.__code__.co_filename).resolve()==Path(wv.__file__).resolve()
                selected=inspect.getclosurevars(callback).nonlocals['selected']
                assert selected.dtype==torch.int64 and selected.shape==(batch,)
                lanes=selected.detach().cpu().tolist()
            else:
                assert batch==1024;lanes=list(range(1024))
            prepared=proposal(label+'_actual_candidate',values,lanes)
            answer=raw_failed(*values,**kw);save_proposal(prepared,answer,'failed');return answer
        hook(wv,'validate_failed',failed)
        for name,label in [('validate_recentered','recenter'),('refine_accepted','early_weighted')]:
            original=getattr(wv,name)
            def observed(*values,_raw=original,_label=label,**kw):
                assert values[1].shape[0]==1024
                prepared=proposal(_label+'_entry',values,list(range(1024)),early=_label=='early_weighted')
                context.append(_label)
                try:answer=_raw(*values,**kw)
                finally:context.pop()
                save_proposal(prepared,answer,_label);return answer
            hook(wv,name,observed)
        replay=c.replay(ctx,a,source)  # one full original advance and original full-output gates
        assert len(settings_calls)==1 and holder['runtime'].ok.tolist()==bare['accepted']
        x=holder['runtime']
        bare_check=c.compare_snapshot(x,a.bare/'committed_800',a.output/'committed_800',expected_bare)
        bare_observer=c.compare_observer(x,a.bare/'observer_800.pt',a.output/'observer_800.pt',expected_bare)
        for name,digest in bare_files.items():assert c.sha(a.bare/name)==digest
        final={}
        for event in events:
            if 'subset_by_component' not in event:continue
            assert len(event['subset_by_component'])==16
            bad_indices=[i for i,yes in enumerate(event['subset_by_component']) if not yes]
            final[(event['event'],event['lane'])]=dict(event=event['event'],lane=event['lane'],failed_indices=bad_indices,
                failed_variables=[ctx.cfg['initial_set'][i]['name'] for i in bad_indices],input_remainder=event.get('input_remainder_vector'),
                proposal=event.get('proposal_interval'),bad=event.get('bad',event.get('bad_nonfinite_mask')),attempt=event.get('attempt'))
        result=dict(replay,status='passed',trace_usable=True,mode='trace799_to800',source_identity=ctx.identity,
            bare_result_sha256=a.bare_result_sha256,bare_all_output_checks=bare_check,bare_observer=bare_observer,
            diagnostic_lanes=SELECTED,original_failed_lanes=FAILED,neighbour_accepted_lanes=[496,608],
            events=len(events),proposals=proposals,component_failures=list(final.values()),
            diagnostic_dispatch_may_change=True,actual_execution_batch=1024,working_order=2,validation_order=3,
            interpretation='Only after emitted plant/support/cap/status/observer and all14 liveSR/ledger components match both capture800 and bare800. Callback may change dispatch; no map/cap/controller/history alteration and no claimed continuation.')
    except BaseException as exc:
        result.update(status='failed',trace_usable=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        for module,name,old in reversed(hooks):setattr(module,name,old)
        restored=all(getattr(module,name) is old for module,name,old in hooks)
        result.update(hooks_restored=restored,script_sha256=c.sha(__file__),cold_helper_sha256=COLD_SHA,process_s=time.perf_counter()-started)
        if not restored:result.update(status='failed',trace_usable=False)
        if (a.output/'INPUT.json').exists():result['input_sha256']=c.sha(a.output/'INPUT.json')
        if 'ctx' in locals():result.update(max_cuda_allocated_bytes=ctx.torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=ctx.torch.cuda.max_memory_reserved())
        c.write(a.output/'TRACE.json',safe(events));c.write(a.output/'PROPOSALS.json',safe(proposals));c.write(a.output/'RESULT.json',safe(result))
        print(json.dumps(safe({k:result[k] for k in ['status','trace_usable','hooks_restored','process_s']})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check','host-capacity-check','snapshot-check','output','source40','cold','capture','bare']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['source40-result-sha256','cold-result-sha256','capture-result-sha256','bare-result-sha256']:p.add_argument('--'+name,required=True)
    main(p.parse_args(),cold_helper())
