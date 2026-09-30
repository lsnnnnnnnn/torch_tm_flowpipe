"""Finite mixed-status boundary qualification; no NN or QUAD trajectory."""
from pathlib import Path
from fractions import Fraction as F
import argparse,copy,hashlib,importlib.util,json,os,sys,time

HERE=Path(__file__).parent
PINS={'endpoint':'f4310a1b0484fd344ae894641302e146fcd232a8c7b296546a0fcdc94b0cd326',
      'injection':'a79b19ad1bd97ae87b40f81239352a8b9859621757bd044df6fd615630c64310'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();result={'status':'exception'}
    try:
        oldpaths={'endpoint':HERE.parent/'quad_targeted_recovery_20260927/strict_endpoint.py',
                  'injection':HERE.parent/'quad_control_transfer_20260927/strict_injection.py'}
        paths={k:HERE/f'strict_{k}_fullbatch.py' for k in oldpaths}
        substitutions={'endpoint':('not bool((st.status == 0).all())','(st.status.dtype != torch.int8 or not bool(((st.status >= 0) & (st.status <= 3)).all()))'),
                       'injection':('not bool((st.status==0).all())','not bool(((st.status>=0)&(st.status<=3)).all())')}
        for k,p in oldpaths.items():
            assert sha(p)==PINS[k]
            s=p.read_text();old,new=substitutions[k];assert s.count(old)==1;s=s.replace(old,new)
            s=s.replace('all active, finite, ordered leaves','valid statuses, finite, ordered leaves').replace('matching active float64 leaves and certificates','matching float64 leaves, original statuses and certificates').replace('owned, all-active committed state','owned, finite committed state with original status codes')
            assert paths[k].read_text().endswith(s),'Only explicit status guards and documentation may differ'
        identity=json.loads(a.identity.read_text());expected=identity.get('engine_python_sha256',identity.get('engine',identity).get('python_sha256'))
        assert expected and len(expected)==36
        for name,digest in expected.items():assert sha(a.engine_root/name)==digest,name
        os.environ.update(FLOWSTAR_COMPOSITION='horner',FLOWSTAR_GLUE='graph',FLOWSTAR_SUPPORT_POLICY='structural',
            FLOWSTAR_INJECTIVE_MAPS='1',FLOWSTAR_INJECTIVE_GLUE='1',FLOWSTAR_VALIDATION_POLICY='solution_plus_one',
            FLOWSTAR_CENTER_NORMALIZATION='1',FLOWSTAR_WEIGHTED_VALIDATION='0',FLOWSTAR_RECENTER_VALIDATION='1',
            FLOWSTAR_EARLY_WEIGHTED='1',FLOWSTAR_SELF_MAP_RETRIES='8')
        sys.path.insert(0,str(a.engine_root/'src'))
        import torch
        from flowstar_gpu import sparse_exec as se,support as sp
        from flowstar_gpu.monomials import build_tables
        from flowstar_gpu.polynomial import build_step_tables
        from flowstar_gpu.composition import build_schedule
        torch.set_default_dtype(torch.float64)
        extensions={}
        if a.device=='cuda':
            assert a.common and sha(a.common)=='1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
            c=load('full_boundary_bootstrap',a.common);assert c.E.resolve()==a.engine_root.resolve();edge=c.bootstrap();extensions=c.ORIGINAL['extensions']
        else:torch.set_num_threads(1);torch.set_num_interop_threads(1)
        tab=build_tables(16,2).to(a.device);eng=sp.SparseEngine(tab,build_step_tables(tab,.005),a.device);sched=build_schedule(16,2,a.device)
        if a.device=='cuda':eng.horner_edge_kernel=edge.horner_edge
        modules={k:load('mixed_'+k,p) for k,p in paths.items()};old={k:load('original_'+k,p) for k,p in oldpaths.items()}
        def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        def state(st):return {k:digest(getattr(st,k)) for k in ['pre','pre_rem','tmv','tmv_rem','status']}|{'pre_support':st.pre_sup.ids,'tmv_support':st.tmv_sup.ids}
        def eq(x,y):return x.shape==y.shape and x.dtype==y.dtype and digest(x)==digest(y)
        exps=tab.exponents.cpu().tolist();index={tuple(e):i for i,e in enumerate(exps)}
        desired=[(0,)*17,(1,)+(0,)*16,(2,)+(0,)*16,(0,1)+(0,)*15,(1,1)+(0,)*15,(0,2)+(0,)*15]
        sup=sp.make_support(16,2,False,tuple(sorted(index[e] for e in desired)))
        checks=[];negative=[];fraction_endpoint=fraction_coeff=0
        for batch in [4,1024]:
            base=se.initial_sparse_state(torch.zeros(batch,16,2,device=a.device),eng,sched)
            base.pre_sup=sup;base.pre=torch.zeros(batch,16,sup.size,device=a.device)
            for pos,e in enumerate(desired):base.pre[:,:,sup.ids.index(index[e])]=[.1,1e6,-2e8,.3,-.7,.2][pos]
            base.pre*=((torch.arange(batch,device=a.device)%4)*.125+1)[:,None,None]
            base.pre_rem[:]=torch.tensor([-.0001,.0002],device=a.device)
            base.status.copy_((torch.arange(batch,device=a.device)%4).to(torch.int8))
            st=copy.deepcopy(base);ref=copy.deepcopy(base);ref.status.zero_() # numeric oracle fixture only; production never changes status
            protected=[digest(st.tmv),digest(st.tmv_rem),digest(st.status)]
            err=modules['endpoint'].end_of_time_s(st,eng);referr=old['endpoint'].end_of_time_s(ref,eng)
            assert eq(err,referr) and eq(st.pre,ref.pre) and eq(st.pre_rem,ref.pre_rem) and st.pre_sup==ref.pre_sup
            assert protected==[digest(st.tmv),digest(st.tmv_rem),digest(st.status)]
            if batch==4:
                before_exps=[exps[i] for i in base.pre_sup.ids];after_exps=[exps[i] for i in st.pre_sup.ids]
                for b in range(batch):
                    for row in range(16):
                        exact={}
                        for value,e in zip(base.pre[b,row].cpu().tolist(),before_exps):
                            k=(0,*e[1:]);exact[k]=exact.get(k,F(0))+F(value)*F(.005)**e[0]
                        for value,e in zip(st.pre[b,row].cpu().tolist(),after_exps):exact[tuple(e)]-=F(value)
                        lo=hi=F(0)
                        for e,v in exact.items():
                            dom=(-1,1) if any(k%2 for k in e[1:]) else ((0,1) if any(e[1:]) else (1,1))
                            lo+=min(v*dom[0],v*dom[1]);hi+=max(v*dom[0],v*dom[1])
                        assert F(float(err[b,row,0]))<=lo<=hi<=F(float(err[b,row,1]));fraction_endpoint+=1
            before=copy.deepcopy(st);T=torch.zeros(batch,3,12,device=a.device);T[:,:,0]=.1;T[:,:,1]=-.7
            L=torch.ones(batch,3,device=a.device);U=torch.nextafter(L,torch.full_like(L,float('inf')))
            rec=modules['injection'].inject_controls_s(st,T,L,U,[13,14,15],12,eng=eng)
            refrec=old['injection'].inject_controls_s(ref,T,L,U,[13,14,15],12,eng=eng)
            assert eq(st.pre,ref.pre) and eq(st.pre_rem,ref.pre_rem) and set(rec)==set(refrec)
            assert all(eq(rec[k],refrec[k]) for k in rec)
            assert eq(st.pre[:,:13],before.pre[:,:13]) and eq(st.pre_rem[:,:13],before.pre_rem[:,:13])
            assert protected==[digest(st.tmv),digest(st.tmv_rem),digest(st.status)]
            if batch==4:
                for b in range(batch):
                    for row in range(3):
                        for j in range(st.pre_sup.size):
                            value=sum((F(float(T[b,row,i]))*F(float(before.pre[b,i,j])) for i in range(12)),F(0))
                            value+=(F(float(rec['center'][b,row])) if j==0 else 0)-F(float(st.pre[b,13+row,j]))
                            assert F(float(rec['coefficient_error'][b,row,j,0]))<=value<=F(float(rec['coefficient_error'][b,row,j,1]));fraction_coeff+=1
                for kind,template in [('endpoint',base),('injection',before)]:
                    for badness in ['invalid_status','status_dtype','inactive_nan','inverted_remainder']:
                        bad=copy.deepcopy(template)
                        if badness=='invalid_status':bad.status[1]=4
                        if badness=='status_dtype':bad.status=bad.status.to(torch.int64)
                        if badness=='inactive_nan':bad.pre[1,0,0]=float('nan')
                        if badness=='inverted_remainder':bad.pre_rem[1,0]=torch.tensor([1.,-1.],device=a.device)
                        snapshot=state(bad)
                        try:
                            if kind=='endpoint':modules[kind].end_of_time_s(bad,eng)
                            else:modules[kind].inject_controls_s(bad,T,L,U,[13,14,15],12,eng=eng)
                        except (ValueError,FloatingPointError):pass
                        else:raise AssertionError((kind,badness))
                        assert state(bad)==snapshot;negative.append(kind+':'+badness)
            checks.append(dict(batch=batch,status_codes=[0,1,2,3],endpoint_and_injection_all_rows_bytes_equal_to_frozen_numeric_body=True,status_unchanged=True,all_nn_rows_retained=True))
        result=dict(status='passed',device=a.device,engine_python_sha256=expected,extensions=extensions,torch_version=torch.__version__,
            adapter_sha256={k:sha(p) for k,p in paths.items()},original_sha256=PINS,source_diff_only_status_guard_and_docs=True,
            cases=checks,fraction_endpoint_components=fraction_endpoint,fraction_coefficient_components=fraction_coeff,
            negative_cases_rejected=negative,end_to_end_strict_certificate=False,
            limits='Finite saved-state mixed statuses only. No CROWN, ODE trajectory, SR, nonfinite-domain recovery or whole-driver claim; failed statuses are never revived. Numeric all-active reference is a separate synthetic oracle only.')
    finally:
        result.update(script_sha256=sha(__file__),process_body_s=time.perf_counter()-started)
        (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['engine-root','identity','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--common',type=Path)
    main(p.parse_args())
