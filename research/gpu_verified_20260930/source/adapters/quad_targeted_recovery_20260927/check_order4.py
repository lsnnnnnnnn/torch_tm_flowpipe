"""Small P4 solution/P5 validation qualification; never build n16 tables.

CPU or CUDA n2/B2 polynomial forcing, exact Fraction solution samples,
strict endpoint handoff with nonempty SR, and rejected zero-cap/domain trials.
CUDA reuses the pinned eight-extension bootstrap; this script never compiles.
This is a bounded arithmetic gate, not CROWN or full-QUAD qualification.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, copy, hashlib, importlib.util, itertools, json, math, os, sys, time

ADAPTER_SHA='f4310a1b0484fd344ae894641302e146fcd232a8c7b296546a0fcdc94b0cd326'
COMMON_SHA='1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
ENV=dict(FLOWSTAR_COMPOSITION='horner',FLOWSTAR_GLUE='graph',FLOWSTAR_SUPPORT_POLICY='structural',
    FLOWSTAR_INJECTIVE_MAPS='1',FLOWSTAR_INJECTIVE_GLUE='1',FLOWSTAR_VALIDATION_POLICY='solution_plus_one',
    FLOWSTAR_CENTER_NORMALIZATION='1',FLOWSTAR_WEIGHTED_VALIDATION='0',FLOWSTAR_RECENTER_VALIDATION='1',
    FLOWSTAR_EARLY_WEIGHTED='1',FLOWSTAR_SELF_MAP_RETRIES='8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def polynomial(row,exponents,domains,remainder):
    """Exact interval extension of binary64 coefficients at Fraction domains."""
    lo,hi=map(lambda v:F(float(v)),remainder)
    for value,powers in zip(row,exponents):
        low=high=F(float(value))
        for (left,right),power in zip(domains,powers):
            if not power:continue
            a=F(0) if power%2==0 and left<=0<=right else min(left**power,right**power)
            b=max(left**power,right**power);terms=[low*a,low*b,high*a,high*b]
            low,high=min(terms),max(terms)
        lo+=low;hi+=high
    return lo,hi

def forcing_range(kind,z):
    if kind in [3,4]:return z**kind,z**kind
    if kind=='reciprocal':return 1/z,1/z
    # Rational Taylor polynomial with |derivative| <= 1 Lagrange tail.
    offset=int(kind=='sin');degree=38+offset
    value=sum(((-1)**j*z**(2*j+offset)/math.factorial(2*j+offset) for j in range(20)),F(0))
    tail=abs(z)**(degree+1)/math.factorial(degree+1)
    return value-tail,value+tail

def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    report=dict(status='exception',device=a.device,working_order=4,validation_order=5,n=2,B=2,
        script_sha256=sha(__file__),scope='Small exact polynomial forcing and nonempty-SR/endpoint gate only; no NN or n16 metadata claim')
    try:
        identity=json.loads(a.identity.read_text());identity=identity.get('engine',identity)
        for rel,digest in identity['python_sha256'].items():assert sha(a.engine_root/rel)==digest,rel
        os.environ.update(ENV);sys.path.insert(0,str(a.engine_root/'src'))
        import torch
        from flowstar_gpu import support as sp,interval as iv,sparse_exec as se,config
        from flowstar_gpu.monomials import build_tables
        from flowstar_gpu.polynomial import build_step_tables
        from flowstar_gpu.composition import build_schedule
        from flowstar_gpu.ode_compiler import compile_ode
        from flowstar_gpu.symbolic_remainder import make_symbolic_remainder
        path=Path(__file__).with_name('strict_endpoint.py');assert sha(path)==ADAPTER_SHA
        adapter=load('order4_strict_endpoint',path);edge=None
        if a.device=='cuda':
            assert a.common and sha(a.common)==COMMON_SHA
            common=load('order4_pinned_bootstrap',a.common)
            assert common.E.resolve()==a.engine_root.resolve();edge=common.bootstrap()
            report['extensions']=common.ORIGINAL['extensions']
        else:torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert all(os.environ[k]==v for k,v in ENV.items())
        assert se.VALIDATION_POLICY=='solution_plus_one'
        torch.set_default_dtype(torch.float64)
        report.update(torch_version=torch.__version__,engine_python_sha256=identity['python_sha256'],
            adapter_sha256=ADAPTER_SHA,environment={k:os.environ[k] for k in ENV})
        tab=build_tables(2,4).to(a.device)
        def engine():
            value=sp.SparseEngine(tab,build_step_tables(tab,.1),a.device)
            if edge is not None:value.horner_edge_kernel=edge.horner_edge
            return value,build_schedule(2,4,a.device)
        eng,sched=engine()
        settings=config.Settings(step=.1,order=4,cutoff=0.,remainder_estimation=.05,mode='strict',device=a.device)
        cap=torch.tensor([-.05,.05],device=a.device).expand(2,2,2).clone()
        base_boxes=torch.tensor([[[.125,.25],[-.25,.375]],[[-.5,-.25],[.25,.5]]],device=a.device)
        boxes=base_boxes
        def tensor_sha(value):
            return None if value is None else hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        def fingerprint(st,sr):
            result={k:tensor_sha(getattr(st,k)) for k in ['pre','pre_rem','tmv','tmv_rem','status']}
            for key in ['scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf']:
                value=getattr(sr,key)
                if value is not None and key.startswith('phi'):value=value[:sr.qlen]
                if value is not None and key=='j_buf':value=value[:sr.jlen]
                result['sr_'+key]=tensor_sha(value)
            return result|dict(pre_support=st.pre_sup.ids,tmv_support=st.tmv_sup.ids,qlen=sr.qlen,jlen=sr.jlen)
        def fresh():
            st=se.initial_sparse_state(boxes,eng,sched);sr=make_symbolic_remainder(2,2,1000,a.device);sr.reserve(8)
            return st,sr
        def attempt(st,sr,code,budget):
            before=fingerprint(st,sr);trial,history=copy.deepcopy(st),copy.deepcopy(sr)
            result,ok=se.advance_sparse(trial,code,eng,sched,settings,budget,history)
            assert fingerprint(st,sr)==before,'speculative advance changed parent'
            assert history.qlen==history.jlen==sr.jlen+1
            return result,ok,history
        def composed(st,lane,z):
            full=tab.exponents.cpu().tolist();spatial=tab.spatial_index.cpu().tolist()
            inner=[polynomial(row,[full[spatial[i]][1:] for i in st.tmv_sup.ids],[(v,v) for v in z],rem)
                for row,rem in zip(st.tmv[lane].cpu(),st.tmv_rem[lane].cpu())]
            return [polynomial(row,[full[i] for i in st.pre_sup.ids],[(F(.1),F(.1))]+inner,rem)
                for row,rem in zip(st.pre[lane].cpu(),st.pre_rem[lane].cpu())]
        cases=[];grid=list(itertools.product([F(-1),F(-1,2),F(0),F(1,2),F(1)],repeat=2))
        for power,expression in [(3,'z*z*z'),(4,'z*z*z*z'),('reciprocal','1/z'),('sin','sin(z)'),('cos','cos(z)')]:
            # Stateful support/graph caches belong to this independent trajectory.
            eng,sched=engine()
            boxes=base_boxes.clone()
            if power=='reciprocal':boxes[:,1]=torch.tensor([[.5,.75],[1.,1.25]],device=a.device)
            st,sr=fresh();initial=copy.deepcopy(st)
            code=compile_ode([expression,'0'],['x','z'],order=3)
            receipts=[];checks=0;retained_degree4=False
            for step in [1,2,3,4]:
                report['active_case']=dict(forcing_power=power,step=step)
                if step==3:
                    before=fingerprint(st,sr);error=adapter.end_of_time_s(st,eng);after=fingerprint(st,sr)
                    for key in before:
                        if key not in ['pre','pre_rem','pre_support']:assert before[key]==after[key],key
                    assert not st.pre_sup.spatial and st.pre_sup.k==4
                    assert bool((tab.exponents[list(st.pre_sup.ids),0]==0).all())
                    assert bool(torch.isfinite(error).all())
                new,ok,next_sr=attempt(st,sr,code,cap);assert bool(ok.all()),(power,step,ok)
                assert next_sr.qlen==next_sr.jlen==step
                if sr.jlen:assert tensor_sha(sr.j_buf[:sr.jlen])==tensor_sha(next_sr.j_buf[:sr.jlen])
                st,sr=se.prune_state(new,eng),next_sr
                v=se.validation_engine_for_order(eng,5);assert v.tables.k==5 and v.tables.n==2
                assert torch.equal(v.tables.exponents[:tab.T2].cpu(),tab.exponents.cpu())
                deg=tab.exponents[list(st.pre_sup.ids)].sum(-1)
                retained_degree4|=bool((st.pre[...,deg==4]!=0).any())
                for z in grid:
                    for lane in range(2):
                        initial_values=[polynomial(row,tab.exponents[list(initial.pre_sup.ids)].cpu().tolist(),[(F(0),F(0))]+[(v,v) for v in z],[0.,0.])[0] for row in initial.pre[lane].cpu()]
                        x0,z0=initial_values;fl,fu=forcing_range(power,z0)
                        exact=[(x0+step*F(.1)*fl,x0+step*F(.1)*fu),(z0,z0)]
                        for bound,(low,high) in zip(composed(st,lane,z),exact):assert bound[0]<=low<=high<=bound[1];checks+=1
                receipts.append(dict(step=step,accepted=ok.cpu().tolist(),qlen=sr.qlen,jlen=sr.jlen,
                    pre_support_size=st.pre_sup.size,tmv_support_size=st.tmv_sup.size,
                    ordinary_remainder_nonzero=bool(st.pre_rem.any())))
            assert retained_degree4
            cases.append(dict(forcing=expression,component_checks=checks,steps=receipts,retained_degree4=True,
                nonempty_sr_strict_endpoint_after_step2=True))
        # Negatives use fresh parent state and discarded speculative history.
        boxes=base_boxes
        negative=[]
        for expression in ['z*z*z*z','sin(z)','cos(z)']:
            eng,sched=engine();st,sr=fresh();code=compile_ode([expression,'0'],['x','z'],order=3)
            _,ok,_=attempt(st,sr,code,torch.zeros_like(cap));assert not bool(ok.any()),'zero cap unexpectedly accepted'
            negative.append(dict(case='zero_cap',forcing=expression,accepted=ok.cpu().tolist(),parent_state_sr_unchanged=True))
        eng,sched=engine()
        bad_boxes=boxes.clone();bad_boxes[:,1]=torch.tensor([-.125,.125],device=a.device)
        st=se.initial_sparse_state(bad_boxes,eng,sched)
        code=compile_ode(['1/z','0'],['x','z'],order=3)
        _,ok,_=attempt(st,sr,code,cap);assert not bool(ok.any()),'singular-domain lane accepted'
        negative.append(dict(case='reciprocal_zero_domain',accepted=ok.cpu().tolist(),parent_state_sr_unchanged=True))
        if a.device=='cuda':torch.cuda.synchronize()
        report.update(status='passed',cases=cases,negative=negative,
            actual_validation_engine_order=5,total_fraction_component_checks=sum(r['component_checks'] for r in cases))
    except BaseException as exc:
        report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['process_body_s']=time.perf_counter()-started
        (a.output/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['engine_python_sha256','extensions']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--common',type=Path);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
