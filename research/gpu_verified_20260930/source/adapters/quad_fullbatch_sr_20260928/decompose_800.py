"""Same-input x6 weighted-map decomposition; no SR restore or trajectory.

Missing original rows are explicit diagnostic padding at the recorded
validate_failed entry sizes. Its internal map may have filtered more rows;
that original map batch/placement is unknown. Both paddings must reproduce
all16 recorded target images bitwise before interpreting x6.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, hashlib, importlib.util, json, math, sys, time, traceback

HERE=Path(__file__).parent
LONG_SHA='7b465c72046b4c23497ace95f4b3d74cba8fd556cbb48a5454ae0af1b3772c98'
TRACE_SHA='2661587779647c1abb46ab5288bae5ec9a930e1e943b92f094cba7ba5c7cfe5a'
TRACE_RESULT='5da842a9e42652c341595ca924598fa36367ce6b7dce7287ca83f5808a6cab67'
FAILED=[497,609,625,737,753,849,865,881,961,977,993,1009]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load_long():
    path=HERE/'run_fullbatch_long.py';assert sha(path)==LONG_SHA
    spec=importlib.util.spec_from_file_location('map800_original_bootstrap',path);m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def same(torch,a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))
def pad(torch,value,batch,mode):
    assert 0<len(value)<batch and mode in ['zero','repeat']
    extra=torch.zeros((batch-len(value),*value.shape[1:]),dtype=value.dtype,device=value.device) if mode=='zero' else value[:1].expand(batch-len(value),*value.shape[1:])
    return torch.cat((value,extra),dim=0)
def outward(x,upper):
    v=float(x)
    return math.nextafter(v,math.inf if upper else -math.inf) if (F(v)<x if upper else F(v)>x) else v
def exact_group(rows):return [outward(sum((F(row[i]) for row in rows),F(0)),bool(i)) for i in [0,1]]


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();result=dict(status='exception',decomposition_usable=False)
    try:
        assert sha(a.trace/'RESULT.json')==TRACE_RESULT and sha(HERE/'trace_799_800.py')==TRACE_SHA
        prior=read(a.trace/'RESULT.json');ti=read(a.trace/'INPUT.json');events=read(a.trace/'TRACE.json')
        assert prior['status']=='passed' and prior['trace_usable'] is True and prior['hooks_restored'] is True
        assert prior['script_sha256']==TRACE_SHA and prior['input_sha256']==sha(a.trace/'INPUT.json')
        assert prior['source_identity']==ti['source_identity'] and prior['actual_execution_batch']==1024
        assert prior['working_order']==2 and prior['validation_order']==3 and prior['original_failed_lanes']==FAILED
        assert prior['no_endpoint_or_control_update'] is True and prior['held_controller_step']==780
        assert prior['bare_all_output_checks']['component_count']==14
        assert all(v is True for k,v in prior['bare_all_output_checks'].items() if k!='component_count')
        assert all(prior['bare_observer'].values()) and all(prior['observer'].values())
        assert len(events)==prior['events'] and prior['proposals']==read(a.trace/'PROPOSALS.json')
        records=[r for r in prior['proposals'] if r['label']=='recenter_actual_candidate']
        assert len(records)==2 and sorted(lane for r in records for lane in r['global_lanes'])==FAILED
        last={}
        for lane in FAILED:
            e=next(e for e in reversed(events) if e['event']=='recentered_self_map' and e['lane']==lane)
            summary=next(e for e in prior['component_failures'] if e['event']=='recentered_self_map' and e['lane']==lane)
            assert e['attempt']==summary['attempt']==3 and e['bad'] is summary['bad'] is False
            assert e['input_remainder_vector']==summary['input_remainder'] and e['proposal_interval']==summary['proposal']
            assert [i for i,v in enumerate(e['subset_by_component']) if not v]==summary['failed_indices']==[5]
            last[lane]=e
        source_files={n:sha(a.trace/n) for n in ['RESULT.json','INPUT.json','TRACE.json','PROPOSALS.json']+[r['file'] for r in records]}
        l=load_long();ctx=l.initialize(a);baseline=dict(ctx.identity);c,d,torch=ctx.c,ctx.d,ctx.torch
        # Pin every inherited algorithm/config/library field; the trace only
        # replaced its identity/checkpoint/one-step fields, not the model.
        for key,value in baseline.items():
            if key not in ['script_sha256','execution_budget_steps','checkpoint_policy']:assert prior['source_identity'][key]==value,key
        assert prior['source_identity']['trace_cold_helper_sha256']=='d7284d93b7944ed21d3b1246aa41e2313d8bee132df908b19c0b0db7886b056e'
        from flowstar_gpu import weighted_validation as wv
        assert sha(wv.__file__)==baseline['engine']['python_sha256']['src/flowstar_gpu/weighted_validation.py']
        names=[v['name'] for v in ctx.cfg['initial_set']]
        code=d.compile_ode(ctx.cfg['dynamics_expressions'],names,order=1)
        tables=d.build_tables(16,2).to('cuda:0');working=d.SparseEngine(tables,d.poly.build_step_tables(tables,.005),'cuda:0')
        working.horner_edge_kernel=ctx.edge.horner_edge
        eng=c.se.validation_engine_for_order(working,3);code=c.se.validation_code_for_order(code,3,eng)
        output=[]
        for record in records:
            path=a.trace/record['file'];assert sha(path)==record['sha256']
            seed=torch.load(path,map_location='cpu',weights_only=True);lanes=seed['global_lanes'];batch=seed['call_batch'];m=len(lanes)
            assert lanes==record['global_lanes'] and (batch,m) in [(32,2),(25,10)]
            assert seed['n']==16 and seed['k']==seed['code_order']==3 and seed['working_order']==2 and seed['h']==.005 and seed['step']==800
            assert seed['x'].dtype==seed['initial'].dtype==seed['cap'].dtype==torch.float64
            assert bool(seed['eligible'].all()) and bool(torch.isfinite(seed['x']).all() and torch.isfinite(seed['initial']).all())
            assert bool((seed['cap'][...,0]==-.1).all() and (seed['cap'][...,1]==.1).all())
            sup=c.sp.make_support(16,3,False,tuple(seed['support_ids']));initial_sup=c.sp.make_support(16,3,False,tuple(seed['initial_ids']))
            var_sups=tuple(c.sp.make_support(16,3,False,tuple(ids)) for ids in seed['variable_support_ids'])
            assert same(torch,seed['exponents'],eng.tables.exponents[list(sup.ids)].cpu())
            assert same(torch,seed['initial_exponents'],eng.tables.exponents[list(initial_sup.ids)].cpu())
            point=seed['x'].to('cuda:0');plan=wv._plan(code,sup,initial_sup,var_sups,eng,1e-6);assert plan['sup']==sup
            expected_initial=torch.zeros_like(point[:,:,plan['zero_positions']]);expected_initial[:,:,plan['initial_positions']]=seed['initial'].to(point.device)
            assert same(torch,point[:,:,plan['zero_positions']],expected_initial),'candidate initial polynomial mismatch'
            rem=torch.tensor([last[lane]['input_remainder_vector'] for lane in lanes],dtype=torch.float64,device=point.device)
            expected=torch.tensor([last[lane]['proposal_interval'] for lane in lanes],dtype=torch.float64)
            assert rem.shape==(m,16,2) and bool(torch.isfinite(rem).all())
            assert bool((rem[...,0]<=0).all() and (rem[...,1]>=0).all() and (rem[...,0]>=-.1).all() and (rem[...,1]<=.1).all())
            gates=[];bare=None
            for mode in ['zero','repeat']:
                p=pad(torch,point,batch,mode);R=pad(torch,rem,batch,mode)
                image,bad=wv._map(code,p,p,R,plan,eng,1e-6)
                assert not bool(bad[:m].any()) and same(torch,image[:m],expected),'diagnostic batch did not reproduce archived all16 map bytes; stop attribution'
                gates.append(dict(padding=mode,diagnostic_batch=batch,recorded_validate_failed_call_batch=batch,target_rows=list(range(m)),global_lanes=lanes,all16_target_image_bytes_equal=True,all_target_bad_false=True))
                if mode=='repeat':bare=(image.detach().clone(),bad.detach().clone())
            assert not same(torch,p[m:],torch.zeros_like(p[m:])),'padding variants must actually differ'
            captured={};raw_valid=c.se.exec_valid_s
            def observe(*values,**kw):
                answer=raw_valid(*values,**kw);assert not captured
                captured.update({k:v.detach().clone() for k,v in zip(['field','tail','ranges','strict_tails'],answer)});return answer
            c.se.exec_valid_s=observe
            try:image,bad=wv._map(code,p,p,R,plan,eng,1e-6)
            finally:c.se.exec_valid_s=raw_valid
            assert same(torch,image,bare[0]) and same(torch,bad,bare[1]) and c.se.exec_valid_s is raw_valid
            poly=torch.zeros((batch,16,plan['residual_size'],2),dtype=p.dtype,device=p.device)
            poly[:,:,plan['field_targets']]=c.iv.mul(captured['field'],plan['weights'])
            subtract=c.iv.mul(c.iv.from_point(p[:,:,plan['shifted_positions']]),plan['h_iv'])
            targets=plan['shifted_targets'];poly[:,:,targets]=c.iv.sub(poly[:,:,targets],subtract)
            terms=c.iv.mul(poly,plan['factors']);residual=c.iv.sum(terms,dim=-1)
            tail=c.iv.mul(captured['tail'],plan['h_iv']);reconstructed=c.iv.add(residual,tail)
            assert same(torch,image,reconstructed),'full diagnostic batch arithmetic reconstruction differs'
            # Decode residual rows from the actual plan's two source maps.
            exps=[None]*plan['residual_size']
            for target,index in zip(plan['field_targets'].cpu().tolist(),plan['spec'].sup_out_union.ids):exps[target]=eng.tables.exponents[index].cpu().tolist()
            for target,position in zip(plan['shifted_targets'].cpu().tolist(),plan['shifted_positions'].cpu().tolist()):
                e=seed['exponents'][position].tolist();assert e[0]>0;e[0]-=1
                assert exps[target] is None or exps[target]==e;exps[target]=e
            assert all(e is not None for e in exps)
            components=[]
            for local,lane in enumerate(lanes):
                intervals=terms[local,5].cpu().tolist();bound=residual[local,5].cpu().tolist()
                exact=[sum((F(row[i]) for row in intervals),F(0)) for i in [0,1]]
                assert F(bound[0])<=exact[0] and F(bound[1])>=exact[1]
                groups={}
                for label,fn in [('spatial_degree',lambda e:sum(e[1:])),('time_power',lambda e:e[0])]:
                    grouped={}
                    for e,v in zip(exps,intervals):grouped.setdefault(fn(e),[]).append(v)
                    groups[label]={str(k):exact_group(v) for k,v in grouped.items()}
                top=sorted(range(len(intervals)),key=lambda j:max(map(abs,intervals[j])),reverse=True)[:12]
                components.append(dict(lane=lane,variable='x6',last_attempt=3,candidate=rem[local,5].cpu().tolist(),
                    polynomial_residual=bound,h_times_ordinary_RHS_remainder=tail[local,5].cpu().tolist(),image=image[local,5].cpu().tolist(),
                    groups=groups,top_terms=[dict(exponents=exps[j],interval=intervals[j]) for j in top]))
            payload=dict(global_lanes=lanes,recorded_validate_failed_call_batch=batch,diagnostic_batch=batch,diagnostic_target_rows=list(range(m)),point=point.cpu(),initial=seed['initial'],candidate=rem.cpu(),
                polynomial=poly[:m].cpu(),contributions=terms[:m].cpu(),residual=residual[:m].cpu(),scaled_tail=tail[:m].cpu(),image=image[:m].cpu(),bad=bad[:m].cpu(),
                residual_exponents=torch.tensor(exps,dtype=torch.int64),factors=plan['factors'].cpu(),
                **{k:v[:m].cpu() for k,v in captured.items()})
            out=a.output/('decomposed_'+path.stem+'.pt');torch.save(payload,out)
            output.append(dict(proposal_file=path.name,proposal_sha256=sha(path),file=out.name,sha256=sha(out),padding_gates=gates,
                instrumented_full_diagnostic_batch_matches_bare=True,reconstructed_full_diagnostic_image_bytes_equal=True,components=components))
        for name,digest in source_files.items():assert sha(a.trace/name)==digest
        result=dict(status='passed',decomposition_usable=True,trace_result_sha256=TRACE_RESULT,trace_script_sha256=TRACE_SHA,
            source_files_sha256=source_files,bootstrap_long_sha256=LONG_SHA,algorithm_identity=baseline,records=output,
            working_order=2,validation_order=3,state_variable='x6',same_candidate_and_cap=True,no_SR_restore=True,no_advance_or_NN=True,
            scope='Conditional same-input read-only weighted map diagnostic. Recorded validate_failed call sizes32/25 are used for diagnostic zero/repeat padding; its internally filtered original map batch and original row positions are unknown, not reconstructed. All16 selected map images/bad reproduce trace bytes before decomposition. Groups classify weighted polynomial defect terms, not individual original RHS expression terms. No candidate/R/cap changes, discarded tail, accepted continuation, speed comparison or end-to-end CROWN certificate.')
    except BaseException as exc:result.update(status='failed',decomposition_usable=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=sha(__file__),process_s=time.perf_counter()-started)
        write(a.output/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','decomposition_usable','process_s']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check','host-capacity-check','snapshot-check','output','source40','cold','trace']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['source40-result-sha256','cold-result-sha256']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
