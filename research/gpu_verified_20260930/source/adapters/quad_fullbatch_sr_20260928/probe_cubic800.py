"""One proposed cubic enrichment of the saved step-800 polynomials.

Local candidate validation only: no SR restore, advance, NN, or trajectory.
Old r3 maps must reproduce first; new P3 approximation receives full r4 proof.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, importlib.util, json, math, sys, time, traceback

HERE=Path(__file__).parent
HELPER_SHA='e80a5a7568e3703b1c43fb5b662c4cc9dfbe738e64be5cb6ec69426dcef03cc2'
DECOMPOSITION_RESULT='43da6f735f4c5c21e1ace9e55a84666a44150f2a3d556fd41a9a41445e5eeb9f'


def helper():
    import hashlib
    p=HERE/'decompose_800.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==HELPER_SHA
    spec=importlib.util.spec_from_file_location('cubic800_frozen_decomposition',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def enrich(torch,seed,zero_poly,residual_exps):
    """Rounded point proposal only; zero_poly must come from the R=0 map."""
    old=[tuple(x) for x in seed['exponents'].tolist()];exps=[tuple(x) for x in residual_exps]
    assert len(old)==len(set(old)) and max(map(sum,old))<=2 and seed['n']==16 and seed['h']==.005
    assert zero_poly.dtype==torch.float64 and zero_poly.shape==(len(seed['x']),16,len(exps),2)
    assert bool(torch.isfinite(zero_poly).all() and (zero_poly[...,0]<=zero_poly[...,1]).all())
    proposals=[];new=set(old)
    for j,e in enumerate(exps):
        if e[0]!=0 or sum(e[1:])!=2:continue
        target=(1,)+e[1:];assert target not in old and sum(target)==3
        for row in range(12):
            for lane in range(len(seed['x'])):
                lo,hi=map(F,zero_poly[lane,row,j].tolist());exact=(lo+hi)/(2*F(seed['h']));value=float(exact)
                assert math.isfinite(value)
                if value:
                    new.add(target);proposals.append(dict(local_lane=lane,row=row,exponents=list(target),
                        zero_R_residual_interval=[float(lo),float(hi)],exact_coefficient=str(exact),rounded_coefficient=value))
    exps=sorted(new,key=lambda e:(sum(e),e));position={e:i for i,e in enumerate(exps)}
    base=torch.zeros((len(seed['x']),16,len(exps)),dtype=torch.float64)
    for j,e in enumerate(old):base[:,:,position[e]]=seed['x'][:,:,j]
    point=base.clone()
    for p in proposals:point[p['local_lane'],p['row'],position[tuple(p['exponents'])]]=p['rounded_coefficient']
    assert proposals and all(exps[j][0]==1 and sum(exps[j])==3 and row<12 for _,row,j in (point!=base).nonzero().tolist())
    same=helper().same
    assert same(torch,point[:,12:],base[:,12:])
    zero_positions=[i for i,e in enumerate(exps) if e[0]==0]
    expected=torch.zeros_like(point[:,:,zero_positions]);zero_exps=[exps[i] for i in zero_positions]
    for j,e in enumerate(seed['initial_exponents'].tolist()):expected[:,:,zero_exps.index(tuple(e))]=seed['initial'][:,:,j]
    assert same(torch,point[:,:,zero_positions],expected)
    return base,point,exps,proposals


def pack(torch,sp,coeff,exps,eng):
    # All candidate and initial terms are <=3; no extended P8 table dictionary.
    prefix=[tuple(e) for e in eng.tables.exponents[:eng.tables.T].cpu().tolist()]
    lookup={e:i for i,e in enumerate(prefix)};mapping={lookup[tuple(e)]:j for j,e in enumerate(exps)}
    sup=sp.make_support(16,4,False,tuple(sorted(mapping)))
    assert tuple(mapping)==sup.ids and len(mapping)==len(exps)
    value=coeff[:,:,[mapping[i] for i in sup.ids]].to(eng.tables.exponents.device)
    return value,sup


def map_parts(ctx,wv,code,point,R,plan,eng):
    """Read-only capture, then exact reconstruction of the unchanged map."""
    c,torch=ctx.c,ctx.torch
    bare=tuple(v.detach().clone() for v in wv._map(code,point,point,R,plan,eng,1e-6))
    captured={};raw=c.se.exec_valid_s
    def observe(*values,**kw):
        answer=raw(*values,**kw);assert not captured
        captured.update(field=answer[0].detach().clone(),tail=answer[1].detach().clone());return answer
    c.se.exec_valid_s=observe
    try:actual=wv._map(code,point,point,R,plan,eng,1e-6)
    finally:c.se.exec_valid_s=raw
    same=helper().same
    assert all(same(torch,a,b) for a,b in zip(actual,bare)) and c.se.exec_valid_s is raw
    poly=torch.zeros((len(point),16,plan['residual_size'],2),dtype=point.dtype,device=point.device)
    poly[:,:,plan['field_targets']]=c.iv.mul(captured['field'],plan['weights'])
    target=plan['shifted_targets']
    poly[:,:,target]=c.iv.sub(poly[:,:,target],c.iv.mul(c.iv.from_point(point[:,:,plan['shifted_positions']]),plan['h_iv']))
    reconstructed=c.iv.add(c.iv.sum(c.iv.mul(poly,plan['factors']),dim=-1),c.iv.mul(captured['tail'],plan['h_iv']))
    assert same(torch,reconstructed,actual[0])
    exps=[None]*plan['residual_size']
    for j,i in zip(plan['field_targets'].cpu().tolist(),plan['spec'].sup_out_union.ids):exps[j]=eng.tables.exponents[i].cpu().tolist()
    for j,pos in zip(plan['shifted_targets'].cpu().tolist(),plan['shifted_positions'].cpu().tolist()):
        e=eng.tables.exponents[plan['sup'].ids[pos]].cpu().tolist();assert e[0]>0;e[0]-=1
        assert exps[j] is None or exps[j]==e;exps[j]=e
    assert all(e is not None for e in exps)
    return poly,exps,actual


def inputs(a,h):
    assert h.sha(a.decomposition/'RESULT.json')==DECOMPOSITION_RESULT
    result=h.read(a.decomposition/'RESULT.json')
    assert result['status']=='passed' and result['decomposition_usable'] is True and result['script_sha256']==HELPER_SHA
    assert result['trace_result_sha256']==h.TRACE_RESULT and result['trace_script_sha256']==h.TRACE_SHA
    for name,digest in result['source_files_sha256'].items():assert h.sha(a.trace/name)==digest
    assert sorted(lane for rec in result['records'] for c in rec['components'] for lane in [c['lane']])==h.FAILED
    for rec in result['records']:
        assert rec['instrumented_full_diagnostic_batch_matches_bare'] and rec['reconstructed_full_diagnostic_image_bytes_equal']
        assert {x['padding'] for x in rec['padding_gates']}=={'zero','repeat'}
        assert all(x['all16_target_image_bytes_equal'] and x['all_target_bad_false'] for x in rec['padding_gates'])
        assert h.sha(a.decomposition/rec['file'])==rec['sha256'] and h.sha(a.trace/rec['proposal_file'])==rec['proposal_sha256']
    return result


def local_check(a,h):
    import torch
    prior=inputs(a,h);checks=[]
    for rec in prior['records']:
        seed=torch.load(a.trace/rec['proposal_file'],map_location='cpu',weights_only=True)
        exp=[0]*17;exp[5]=exp[10]=1
        poly=torch.zeros((len(seed['x']),16,1,2),dtype=torch.float64);poly[:,5,0,:]=.005*2
        base,point,exps,changes=enrich(torch,seed,poly,[exp])
        j=exps.index((1,)+tuple(exp[1:]));assert bool((point[:,5,j]==2.).all())
        assert len(changes)==len(seed['x']) and all(h.same(torch,base[:,:,exps.index(tuple(e))],seed['x'][:,:,i]) for i,e in enumerate(seed['exponents'].tolist()))
        assert max(map(sum,exps))==3
        checks.append(dict(global_lanes=seed['global_lanes'],synthetic_extra_terms=len(changes),all_old_coefficients_bytes_unchanged=True,p0_bytes_unchanged=True))
    return dict(status='passed_local_construction',rows=checks,actual_zero_R_map_run=False,actual_r4_validation_run=False,
        scope='CPU proposal construction on actual saved P2 coefficients with one explicitly synthetic zero-R residual coefficient; no engine, numerical map, SR or GPU.')


def experiment(a,h):
    prior=inputs(a,h);l=h.load_long();ctx=l.initialize(a);c,d,torch=ctx.c,ctx.d,ctx.torch
    assert ctx.identity==prior['algorithm_identity']
    from flowstar_gpu import weighted_validation as wv
    assert h.sha(wv.__file__)==ctx.identity['engine']['python_sha256']['src/flowstar_gpu/weighted_validation.py']
    names=[v['name'] for v in ctx.cfg['initial_set']]
    code1=d.compile_ode(ctx.cfg['dynamics_expressions'],names,order=1)
    tables=d.build_tables(16,2).to('cuda:0');working2=d.SparseEngine(tables,d.poly.build_step_tables(tables,.005),'cuda:0')
    working2.horner_edge_kernel=ctx.edge.horner_edge
    eng3=c.se.validation_engine_for_order(working2,3);code3=c.se.validation_code_for_order(code1,3,eng3)
    prepared=[]
    for rec in prior['records']:
        seed=torch.load(a.trace/rec['proposal_file'],map_location='cpu',weights_only=True)
        archived=torch.load(a.decomposition/rec['file'],map_location='cpu',weights_only=True)
        lanes=seed['global_lanes'];m=len(lanes);batch=seed['call_batch']
        assert (batch,m) in [(32,2),(25,10)] and archived['global_lanes']==lanes
        assert seed['n']==16 and seed['k']==seed['code_order']==3 and seed['working_order']==2 and seed['h']==.005 and seed['step']==800
        assert h.same(torch,seed['x'],archived['point']) and h.same(torch,seed['initial'],archived['initial'])
        assert seed['cap'].shape==(m,16,2) and bool((seed['cap'][...,0]==-.1).all() and (seed['cap'][...,1]==.1).all())
        sup=c.sp.make_support(16,3,False,tuple(seed['support_ids']));isup=c.sp.make_support(16,3,False,tuple(seed['initial_ids']))
        variables=tuple(c.sp.make_support(16,3,False,tuple(ids)) for ids in seed['variable_support_ids'])
        assert h.same(torch,seed['exponents'],eng3.tables.exponents[list(sup.ids)].cpu())
        assert h.same(torch,seed['initial_exponents'],eng3.tables.exponents[list(isup.ids)].cpu())
        point=seed['x'].cuda();R=archived['candidate'].cuda();plan=wv._plan(code3,sup,isup,variables,eng3,1e-6);assert plan['sup']==sup
        for mode in ['zero','repeat']:
            padded=h.pad(torch,point,batch,mode);rpad=h.pad(torch,R,batch,mode)
            image,bad=wv._map(code3,padded,padded,rpad,plan,eng3,1e-6)
            assert h.same(torch,image[:m],archived['image']) and h.same(torch,bad[:m],archived['bad']) and not bool(bad[:m].any())
        # Proposal uses R=0 freshly; old nonzero-R residual is never substituted.
        zero=torch.zeros_like(rpad);poly,rexps,(image,bad)=map_parts(ctx,wv,code3,padded,zero,plan,eng3)
        assert not bool(bad.any()) and not bool(zero.any())
        zpath=a.output/('zero_R_'+rec['proposal_file']);torch.save(dict(global_lanes=lanes,point=seed['x'],candidate=zero[:m].cpu(),polynomial=poly[:m].cpu(),residual_exponents=rexps,image=image[:m].cpu(),bad=bad[:m].cpu()),zpath)
        base,new,exps,corrections=enrich(torch,seed,poly[:m].cpu(),rexps)
        prepared.append((rec,seed,archived,base,new,exps,corrections,dict(file=zpath.name,sha256=h.sha(zpath))))
    # A fresh working P3 family and point-code2 define independent r4 validation.
    tables=d.build_tables(16,3).to('cuda:0');working3=d.SparseEngine(tables,d.poly.build_step_tables(tables,.005),'cuda:0')
    working3.horner_edge_kernel=ctx.edge.horner_edge
    code2=d.compile_ode(ctx.cfg['dynamics_expressions'],names,order=2)
    eng4=c.se.validation_engine_for_order(working3,4);code4=c.se.validation_code_for_order(code2,4,eng4)
    rows=[]
    for rec,seed,archived,base,new,exps,corrections,zero_receipt in prepared:
        initial,isup=pack(torch,c.sp,seed['initial'],seed['initial_exponents'].tolist(),eng4)
        baseline,sup=pack(torch,c.sp,base,exps,eng4);candidate,csup=pack(torch,c.sp,new,exps,eng4);assert csup==sup
        old_to_new={tuple(e):sid for e,sid in zip(exps,sup.ids)}
        variables=[]
        for row,old_ids in enumerate(seed['variable_support_ids']):
            old_exp=[tuple(eng3.tables.exponents[i].cpu().tolist()) for i in old_ids]
            added=[tuple(p['exponents']) for p in corrections if p['row']==row]
            variables.append(c.sp.make_support(16,4,False,tuple(sorted({old_to_new[e] for e in old_exp+added}))))
        variables=tuple(variables);plan=wv._plan(code4,sup,isup,variables,eng4,1e-6);assert plan['sup']==sup
        cap=seed['cap'].cuda();oldR=archived['candidate'].cuda();lanes=seed['global_lanes'];group=[]
        for label,point in [('P2_zero_slots_r4',baseline),('proposed_P3_r4',candidate)]:
            expected=torch.zeros_like(point[:,:,plan['zero_positions']]);expected[:,:,plan['initial_positions']]=initial
            assert h.same(torch,point[:,:,plan['zero_positions']],expected)
            before=(point.detach().clone(),initial.detach().clone(),cap.detach().clone())
            torch.cuda.synchronize();start=time.perf_counter()
            image,bad=wv._map(code4,point,point,oldR,plan,eng4,1e-6)
            same_R_image=image.detach().cpu().clone();same_R_bad=bad.detach().cpu().clone()
            events=[];ok,rem,info=wv.validate_failed(code4,point,sup,initial,isup,variables,cap,
                torch.ones(len(lanes),dtype=torch.bool,device=point.device),eng4,1e-6,max_attempts=4,chunk_size=32,trace=events.append,use_graph=False)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            assert all(h.same(torch,x,y) for x,y in zip((point,initial,cap),before))
            assert bool(c.iv.contains(cap[ok],rem[ok]).all()) and bool(torch.isfinite(rem[ok]).all())
            for event in events:event['global_lane']=lanes[event['lane']]
            path=a.output/(label+'_'+rec['proposal_file'])
            torch.save(dict(global_lanes=lanes,point=point.cpu(),initial=initial.cpu(),cap=cap.cpu(),support_ids=list(sup.ids),initial_ids=list(isup.ids),
                exponents=eng4.tables.exponents[list(sup.ids)].cpu(),initial_exponents=eng4.tables.exponents[list(isup.ids)].cpu(),variable_support_ids=[list(v.ids) for v in variables],
                same_R=oldR.cpu(),same_R_image=same_R_image,same_R_bad=same_R_bad,accepted=ok.cpu(),emitted_remainder=rem.cpu(),
                failed_remainders_are_invalid=True,working_candidate_degree=2 if label.startswith('P2') else 3,validation_order=4,h=.005,cutoff=1e-6),path)
            tracepath=path.with_suffix('.trace.json');h.write(tracepath,events)
            group.append(dict(label=label,global_lanes=lanes,accepted=ok.cpu().tolist(),all_accepted=bool(ok.all()),
                same_R_bad=same_R_bad.tolist(),same_R_contains_by_component=c.iv.contains(oldR.cpu(),same_R_image).tolist(),
                elapsed_s=elapsed,statistics=info,file=path.name,sha256=h.sha(path),trace_file=tracepath.name,trace_sha256=h.sha(tracepath),
                input_initial_cap_bytes_unchanged=True))
        rows.append(dict(proposal=rec['proposal_file'],proposal_sha256=rec['proposal_sha256'],old_r3_both_padding_all16_bytes=True,
            zero_R=zero_receipt,corrections=corrections,point_constructor='RN(exact midpoint of zero-R time0 spatial2 residual / h)',arms=group))
    inputs(a,h)
    return dict(status='completed_local_candidate_validation',rows=rows,algorithm_identity=ctx.identity,working_candidate_order=3,
        fresh_working_family=3,point_code_order=2,validation_order=4,all_12_proposed_candidates_accepted=all(all(row['arms'][1]['accepted']) for row in rows),
        original_initial_and_cap_preserved=True,no_SR_restore=True,no_advance_or_NN=True,
        scope='A one-time cubic polynomial proposal, not the standard P3 predictor or a complete P3 step. All16 full r4 weighted maps and original max4 retry validation retain ordinary tails and original cap. P2 zero-slot control shares r4/support. Failed lanes have no certificate. No SR continuation, full1024 acceptance, trajectory success, performance comparison or end-to-end CROWN proof.')


def main(a):
    h=helper();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception')
    try:result=local_check(a,h) if a.check_local else experiment(a,h)
    except BaseException as e:result.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        result.update(script_sha256=h.sha(__file__),helper_sha256=HELPER_SHA,decomposition_result_sha256=DECOMPOSITION_RESULT,process_s=time.perf_counter()-start)
        h.write(a.output/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status','process_s']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-local',action='store_true')
    for name in ['trace','decomposition','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['common','candidate-build','candidate-check','endpoint-check','injection-check','boundary-check','host-small-check','host-capacity-check','snapshot-check','source40','cold']:p.add_argument('--'+name,type=Path)
    for name in ['source40-result-sha256','cold-result-sha256']:p.add_argument('--'+name)
    main(p.parse_args())
