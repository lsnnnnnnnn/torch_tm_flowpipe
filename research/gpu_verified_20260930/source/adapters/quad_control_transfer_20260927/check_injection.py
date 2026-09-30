"""Conditional injection CPU/CUDA gate: exact arithmetic, atomicity and live SR.

No CROWN calls or certificate qualification. CUDA uses pinned archived binaries.
Each --actual directory is a saved strict-endpoint boundary20 from the old runner.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, ast, copy, hashlib, importlib.util, itertools, json, math, os, sys, time

BACKEND_SHA='3140024445ebac6845d037c6399e2ad8658ae63281eb066c1e87356c99bd990b'
COMMON_SHA='1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
DRIVER_SHA='1bc3aeea0e9216fe8432ae78f18c9592d19658c23cdde02b90a54d0b14c08a31'
ENV=dict(FLOWSTAR_COMPOSITION='horner',FLOWSTAR_GLUE='graph',FLOWSTAR_SUPPORT_POLICY='structural',
    FLOWSTAR_INJECTIVE_MAPS='1',FLOWSTAR_INJECTIVE_GLUE='1',FLOWSTAR_VALIDATION_POLICY='solution_plus_one',
    FLOWSTAR_CENTER_NORMALIZATION='1',FLOWSTAR_WEIGHTED_VALIDATION='0',FLOWSTAR_RECENTER_VALIDATION='1',
    FLOWSTAR_EARLY_WEIGHTED='1',FLOWSTAR_SELF_MAP_RETRIES='8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);v=importlib.util.module_from_spec(spec)
    sys.modules[name]=v;spec.loader.exec_module(v);return v
def frac(v):return F(float(v))


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result=dict(status='exception',device=a.device,script_sha256=sha(__file__),
        scope='Conditional affine-certificate transfer and a small nonempty-SR continuation; no CROWN or full NNCS qualification')
    try:
        identity=json.loads(a.identity.read_text())
        expected=identity.get('engine_python_sha256',identity.get('engine',identity).get('python_sha256'))
        assert expected and len(expected)==36
        for rel,digest in expected.items():assert sha(a.engine_root/rel)==digest,rel
        os.environ.update(ENV);sys.path.insert(0,str(a.engine_root/'src'))
        import torch
        from flowstar_gpu import sparse_exec as se,interval as iv,support as sp,config
        from flowstar_gpu.ode_compiler import compile_ode
        from flowstar_gpu.symbolic_remainder import make_symbolic_remainder
        torch.set_default_dtype(torch.float64)
        edge=None
        if a.device=='cuda':
            assert a.common and sha(a.common)==COMMON_SHA
            common=load('injection_bootstrap',a.common);assert common.E.resolve()==a.engine_root.resolve()
            edge=common.bootstrap();result['extensions']=common.ORIGINAL['extensions']
        else:torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert all(os.environ[k]==v for k,v in ENV.items())
        assert sha(a.backend)==BACKEND_SHA and sha(a.driver)==DRIVER_SHA
        backend=load('injection_backend',a.backend);backend.install(dict(python_sha256=expected))
        adapter=load('injection_adapter',Path(__file__).with_name('strict_injection.py'))
        tree=ast.parse(a.driver.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['_inject_core_s','inject_controls_s']]
        old=dict(torch=torch,iv=iv,next_up=iv.next_up);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(a.driver),'exec'),old)
        result.update(adapter_sha256=sha(adapter.__file__),backend_sha256=sha(a.backend),driver_sha256=sha(a.driver),
            engine_python_sha256=expected,torch_version=torch.__version__,environment=ENV)
        def engine(k):
            eng,sched=backend.engine(16,k,.005,a.device)
            if edge is not None:eng.horner_edge_kernel=edge.horner_edge
            return eng,sched
        def th(v):
            if v is None:return None
            return dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        def history(sr):
            return {key:th(getattr(sr,key)[:sr.qlen] if key.startswith('phi') and getattr(sr,key) is not None
                else getattr(sr,key)[:sr.jlen] if key=='j_buf' else getattr(sr,key))
                for key in ['scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf']}|dict(qlen=sr.qlen,jlen=sr.jlen,max_size=sr.max_size)
        def state_id(st):
            return {key:th(getattr(st,key)) for key in ['pre','pre_rem','tmv','tmv_rem','status']}|dict(pre_ids=st.pre_sup.ids,tmv_ids=st.tmv_sup.ids)
        def unchanged(st,before):
            assert th(st.pre[:,:13])==th(before.pre[:,:13]) and th(st.pre_rem[:,:13])==th(before.pre_rem[:,:13])
            assert st.pre_sup==before.pre_sup and st.tmv_sup==before.tmv_sup
            assert all(th(getattr(st,k))==th(getattr(before,k)) for k in ['tmv','tmv_rem','status'])
        def ipoly(row,exps,boxes,rem):
            lo,hi=map(frac,rem)
            for c,e in zip(row,exps):
                l=u=frac(c)
                for (left,right),degree in zip(boxes,e):
                    degree=int(degree)  # Fraction exponent must use unbounded Python int.
                    if not degree:continue
                    low=F(0) if degree%2==0 and left<=0<=right else min(left**degree,right**degree)
                    high=max(left**degree,right**degree);terms=[l*low,l*high,u*low,u*high];l,u=min(terms),max(terms)
                lo+=l;hi+=u
            return lo,hi
        def composed(st,eng,z):
            pre=eng.exponents(st.pre_sup).tolist();tmv=eng.exponents(st.tmv_sup).tolist();rows=[]
            for lane in range(2):
                mid=[ipoly(row,tmv,[(v,v) for v in z],rem) for row,rem in zip(st.tmv[lane].cpu(),st.tmv_rem[lane].cpu())]
                rows.append([ipoly(row,pre,[(F(.005),F(.005))]+mid,rem) for row,rem in zip(st.pre[lane].cpu(),st.pre_rem[lane].cpu())])
            return rows

        # Adjacent bias plus odd/even and top-degree support, both lanes distinct.
        exact_checks=0;point_checks=0;actual=[];negatives=[]
        for k in [2,3,4]:
            eng,sched=engine(k);base=se.initial_sparse_state(torch.zeros(2,16,2,device=a.device),eng,sched)
            exps=[(0,)*17,(0,1)+(0,)*15,(0,k)+(0,)*15,(0,k-1,1)+(0,)*14]
            sup=eng.support(tuple(eng.tables.full.rank(e) for e in exps));base.pre_sup=sup
            base.pre=torch.zeros(2,16,sup.size,device=a.device)
            base.pre[:,0]=torch.tensor([.1,.3,-.2,.7],device=a.device)
            base.pre[:,1]=torch.tensor([-.1,.9,.2,-.4],device=a.device)
            base.pre[1,:2]*=-1;base.pre[0,2,0]=math.nextafter(0.,1.)
            base.pre_rem[:,:12]=torch.tensor([-2.**-35,2.**-34],device=a.device)
            T=torch.zeros(2,3,12,device=a.device);T[:,:,0]=.1;T[:,:,1]=-.7;T[:,:,2]=.5
            L=torch.ones(2,3,device=a.device);U=torch.full_like(L,math.nextafter(1.,math.inf))
            L[1]=-math.nextafter(1.,math.inf);U[1]=-1
            st=copy.deepcopy(base);reference=copy.deepcopy(base)
            old['inject_controls_s'](reference,T,L,U,[13,14,15],12)
            receipt=adapter.inject_controls_s(st,T,L,U,[13,14,15],12,eng=eng)
            assert th(st.pre)==th(reference.pre);unchanged(st,base)
            for b,m,j in itertools.product(range(2),range(3),range(sup.size)):
                truth=sum((frac(T[b,m,i])*frac(base.pre[b,i,j]) for i in range(12)),F(0))
                truth+=(frac(receipt['center'][b,m]) if j==0 else 0)-frac(st.pre[b,13+m,j])
                lo,hi=map(frac,receipt['coefficient_error'][b,m,j]);assert lo<=truth<=hi;exact_checks+=1
            for z0,z1 in itertools.product([F(-1),F(0),F(1)],repeat=2):
                z=[z0,z1]+[F(0)]*14;boxes=[(F(0),F(0))]+[(v,v) for v in z]
                for b,m in itertools.product(range(2),range(3)):
                    vals=[ipoly(base.pre[b,i].cpu(),eng.exponents(sup),boxes,base.pre_rem[b,i].cpu()) for i in range(12)]
                    lows=[];highs=[]
                    for i,(lo,hi) in enumerate(vals):
                        v=frac(T[b,m,i]);lows.append(min(v*lo,v*hi));highs.append(max(v*lo,v*hi))
                    truth=(sum(lows,F(0))+frac(L[b,m]),sum(highs,F(0))+frac(U[b,m]))
                    lo,hi=ipoly(st.pre[b,13+m].cpu(),eng.exponents(sup),boxes,st.pre_rem[b,13+m].cpu())
                    assert lo<=truth[0]<=truth[1]<=hi;point_checks+=1
            if k==3:
                for name in ['nan','reversed_bias','reversed_rem','inactive','point_overflow','timeful','tmv_alias','wrong_dtype']:
                    bad=copy.deepcopy(base);tt=T.clone();ll=L.clone();uu=U.clone()
                    if name=='nan':tt[0,0,0]=float('nan')
                    elif name=='reversed_bias':ll[0,0]=2
                    elif name=='reversed_rem':bad.pre_rem[0,0]=torch.tensor([1.,-1.],device=a.device)
                    elif name=='inactive':bad.status[1]=1
                    elif name=='point_overflow':ll[:]=uu[:]=torch.finfo(torch.float64).max
                    elif name=='timeful':
                        extra=eng.tables.full.rank((1,)+(0,)*16);new=eng.support((*bad.pre_sup.ids,extra))
                        bad.pre=sp.embed_s(bad.pre,eng,bad.pre_sup,new);bad.pre_sup=new
                    elif name=='tmv_alias':bad.tmv=bad.pre
                    elif name=='wrong_dtype':tt=tt.float()
                    before=state_id(bad)
                    try:adapter.inject_controls_s(bad,tt,ll,uu,[13,14,15],12,eng=eng)
                    except (ValueError,FloatingPointError):pass
                    else:raise AssertionError('negative accepted '+name)
                    assert state_id(bad)==before;negatives.append(name)

        # Real accepted first step, then boundary injection into an already live
        # SR. Same held-control ODE throughout; no callback changes validation.
        eng,sched=engine(3);boxes=torch.zeros(2,16,2,device=a.device)
        boxes[:,0]=torch.tensor([[.125,.25],[-.5,-.25]],device=a.device)
        initial=se.initial_sparse_state(boxes,eng,sched);st=copy.deepcopy(initial)
        sr=make_symbolic_remainder(2,16,1000,a.device);sr.reserve(8)
        names=[f'x{i+1}' for i in range(16)];ode=['x14']+['0']*11+['1','0','0','0']
        code=compile_ode(ode,names,order=2);settings=config.Settings(step=.005,order=3,cutoff=0.,remainder_estimation=.1,mode='strict',device=a.device)
        cap=torch.tensor([-.1,.1],device=a.device).expand(2,16,2).clone()
        st,ok=se.advance_sparse(st,code,eng,sched,settings,cap,sr);assert bool(ok.all())
        st=se.prune_state(st,eng);assert sr.qlen==sr.jlen==1
        backend.endpoint(st,eng);before=copy.deepcopy(st);old_history=history(sr);old_j=sr.j_buf[:1].clone()
        T=torch.zeros(2,3,12,device=a.device);L=torch.zeros(2,3,device=a.device);U=L.clone();L[:,0]=.125;U[:,0]=.25
        receipt=adapter.inject_controls_s(st,T,L,U,[13,14,15],12,eng=eng)
        unchanged(st,before);assert history(sr)==old_history
        _,_,second_error=sp.evaluate_time_end_s_with_roundoff(st.pre,eng,st.pre_sup)
        expected_rem=iv.add(st.pre_rem,second_error)
        captured=[];appended=[];compose=se.compose_s
        tagged_sr=copy.deepcopy(sr);append=tagged_sr.append_j
        def tap_compose(*args,**kwargs):captured.append(args[1].clone());return compose(*args,**kwargs)
        def tap_append(v):appended.append(v.clone());return append(v)
        se.compose_s=tap_compose;tagged_sr.append_j=tap_append
        try:tagged,ok=se.advance_sparse(copy.deepcopy(st),code,eng,sched,settings,cap,tagged_sr)
        finally:se.compose_s=compose;tagged_sr.append_j=append
        assert bool(ok.all()) and tagged_sr.qlen==tagged_sr.jlen==2
        assert len(captured)==1 and th(captured[0])==th(expected_rem)
        assert len(appended)==1 and th(appended[0])==th(tagged_sr.j_buf[1]) and th(old_j)==th(tagged_sr.j_buf[:1])
        bare_sr=copy.deepcopy(sr);bare,ok=se.advance_sparse(copy.deepcopy(st),code,eng,sched,settings,cap,bare_sr)
        assert bool(ok.all()) and state_id(bare)==state_id(tagged) and history(bare_sr)==history(tagged_sr)
        omitted=copy.deepcopy(st);omitted.pre_rem[:,13:16]=0;omitted_sr=copy.deepcopy(sr)
        bad,ok=se.advance_sparse(omitted,code,eng,sched,settings,cap,omitted_sr);assert bool(ok.all())
        analytic=misses=0
        for z0 in [F(-1),F(-1,2),F(0),F(1,2),F(1)]:
            z=[z0]+[F(0)]*15;good,wrong=composed(bare,eng,z),composed(bad,eng,z)
            for lane,u in itertools.product(range(2),[F(1,8),F(1,4)]):
                x0=ipoly(initial.pre[lane,0].cpu(),eng.exponents(initial.pre_sup),[(F(0),F(0))]+[(v,v) for v in z],initial.pre_rem[lane,0].cpu())[0]
                for row,target in [(0,x0+F(.005)*u),(12,2*F(.005)),(13,u)]:
                    assert good[lane][row][0]<=target<=good[lane][row][1];analytic+=1
                    misses+=not(wrong[lane][row][0]<=target<=wrong[lane][row][1])
        assert misses>0

        for directory in a.actual:
            source_meta=json.loads((directory/'INPUT.json').read_text())
            assert source_meta['engine']['python_sha256']==expected
            assert source_meta['metadata_backend_sha256']==BACKEND_SHA and source_meta['driver_sha256']==DRIVER_SHA
            path=directory/'after_strict_endpoint_20.pt';side=json.loads(path.with_suffix('.json').read_text());assert sha(path)==side['state_sha256']
            certpath=directory/'controller_at_20.pt';done=json.loads((directory/'RESULT.json').read_text())
            assert [c['sha256'] for c in done['controllers'] if c['completed_step']==20]==[sha(certpath)]
            saved=torch.load(path,map_location=a.device,weights_only=True);cert=torch.load(certpath,map_location=a.device,weights_only=True)
            eng,_=engine(saved['k']);st=se.SparseState(**{k:saved[k].clone() for k in ['pre','pre_rem','tmv','tmv_rem','status']},pre_sup=eng.support(saved['pre_ids']),tmv_sup=eng.support(saved['tmv_ids'],True))
            assert eng.exponents(st.pre_sup).tolist()==saved['pre_exponents'].cpu().tolist()
            T,L,U=[v.expand(len(st.pre),*v.shape[1:]).clone() for v in cert['zero_target']]
            before=copy.deepcopy(st);ref=copy.deepcopy(st);old['inject_controls_s'](ref,T,L,U,[13,14,15],12)
            r=adapter.inject_controls_s(st,T,L,U,[13,14,15],12,eng=eng);assert th(st.pre)==th(ref.pre);unchanged(st,before)
            archived_path=directory/'after_controller_20.pt'
            assert sha(archived_path)==json.loads(archived_path.with_suffix('.json').read_text())['state_sha256']
            archived=torch.load(archived_path,map_location=a.device,weights_only=True)
            actual.append(dict(directory=str(directory),order=saved['k'],snapshot_sha256=sha(path),certificate_sha256=sha(certpath),
                source_input_sha256=sha(directory/'INPUT.json'),source_result_sha256=sha(directory/'RESULT.json'),
                source_identity={k:source_meta[k] for k in ['script_sha256','config_sha256','model_sha256','boxes_sha256']},
                archived_after_controller_sha256=sha(archived_path),
                same_device_original_point_bytes_equal=True,archived_point_bytes_equal=th(st.pre)==th(archived['pre']),
                coefficient_error_range=r['coefficient_error_range'].cpu().tolist(),bias_residual=r['bias_residual'].cpu().tolist(),
                old_control_remainder=ref.pre_rem[:,13:16].cpu().tolist(),new_control_remainder=st.pre_rem[:,13:16].cpu().tolist(),
                lower_delta=(st.pre_rem[:,13:16,0]-ref.pre_rem[:,13:16,0]).cpu().tolist(),upper_delta=(st.pre_rem[:,13:16,1]-ref.pre_rem[:,13:16,1]).cpu().tolist()))
            torch.save({k:v.detach().cpu() for k,v in r.items()},a.output/f'actual_order{saved["k"]}_receipt.pt')
        result.update(status='passed',exact_coefficient_checks=exact_checks,exact_point_component_checks=point_checks,
            negative_fail_before_mutation=negatives,original_same_device_point_bytes_equal=True,
            sr_boundary_unchanged=True,old_J_bytes_unchanged=True,fresh_compose_received_exact_remainder=True,
            exactly_one_normal_J_append=True,instrumented_bare_full_state_and_live_SR_bytes_equal=True,
            sr_qlen_before=1,sr_qlen_after=2,analytic_composed_checks=analytic,deliberately_omitted_control_remainder_misses=misses,
            actual_boundaries=actual)
    except BaseException as exc:result.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['process_s']=time.perf_counter()-start
        (a.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['engine_python_sha256','extensions','actual_boundaries']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--backend',type=Path,required=True);p.add_argument('--driver',type=Path,required=True)
    p.add_argument('--common',type=Path);p.add_argument('--actual',action='append',type=Path,default=[])
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
