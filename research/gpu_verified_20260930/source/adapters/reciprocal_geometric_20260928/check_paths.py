"""Bounded CPU/CUDA qualification of the installed five-route candidate."""
from pathlib import Path
from fractions import Fraction as F
from types import SimpleNamespace
import argparse, ast, hashlib, importlib.util, inspect, itertools, json, sys, textwrap, time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def equal(a,b):
    import torch
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.detach().cpu().contiguous().view(torch.uint8),b.detach().cpu().contiguous().view(torch.uint8))
def polynomial(coeff,exps,z):
    lo=F(0);hi=F(0)
    for pair,e in zip(coeff.tolist(),exps):
        v=F(0)**int(e[0])*z**int(e[1]);ends=[F(t)*v for t in pair];lo+=min(ends);hi+=max(ends)
    return lo,hi
def prefix(fn):
    body=[]
    for n in ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0].body:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):continue
        body.append(ast.dump(n,include_attributes=False))
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='tm_range':return body
    raise AssertionError('missing prefix')
def capture(fn,call):
    source,line=inspect.getsourcelines(fn)
    target,=[line+n.lineno-1 for n in ast.walk(ast.parse(textwrap.dedent(''.join(source))))
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
        and any(isinstance(t,ast.Name) and t.id in ('lag','tail') for t in n.targets[0].elts)]
    snapshots=[]
    def tracer(frame,event,arg):
        if frame.f_code is not fn.__code__:return None
        if event=='line' and frame.f_lineno==target:
            snapshots.append({k:v.detach().clone() for k,v in frame.f_locals.items()
                if k in ('res_c','res_r','f_range','tm_range','rec_c','c','result','r2','const_part','c_f')})
        return tracer
    assert sys.gettrace() is None
    try:sys.settrace(tracer);result=call()
    finally:sys.settrace(None)
    assert len(snapshots)==1
    return result,snapshots[0]


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception');handle=None
    try:
        sys.path.insert(0,str(a.engine_root/'src'))
        import torch
        from flowstar_gpu import elementary as elem,interval as iv,sparse_exec as se,support as sp,ode_compiler as oc
        from flowstar_gpu.monomials import build_tables
        from flowstar_gpu.polynomial import build_step_tables
        if a.device=='cuda':
            assert a.common and sha(a.common)=='1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
            common=load('geometric_pinned_common',a.common);assert common.E.resolve()==a.engine_root.resolve();common.bootstrap()
            torch.set_default_dtype(torch.float64)
            assert torch.get_num_threads()==1 and torch.get_default_dtype()==torch.float64
        else:
            torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        installer=load('geometric_installer',Path(__file__).with_name('install.py'))
        pre=json.loads(a.build_result.read_text()) if a.build_result else None
        if pre:
            assert pre['status']=='passed' and pre['device']=='cuda'
            for entry in pre['extensions'].values():assert sha(entry['path'])==entry['sha256']
        handle=installer.install(a.cache,allow_build=a.allow_build,binary_paths={k:v['path'] for k,v in pre['extensions'].items()} if pre else None)
        core=handle.core;tk=handle.tape
        if pre:assert pre['candidate']==handle.metadata
        assert all(prefix(fn)==prefix(getattr(core,name)) for name,fn in handle.original_functions.items())
        assert tk._ext is tk._vext is None and not tk._tried and not tk._vtried
        assert tk.refine_tape_run.__globals__ is tk.__dict__ and tk.exec_valid_tape.__globals__ is tk.__dict__
        assert sys.modules['flowstar_gpu.tape_kernels'] is tk
        installer.emit(handle,a.output/'sources');build=[]
        if a.device=='cuda':
            for name,fn in [('replay',tk.available),('valid',tk.valid_available)]:
                t=time.perf_counter();assert fn();torch.cuda.synchronize();build.append(dict(extension=name,wall_s=time.perf_counter()-t))
            assert len(handle.loads)==2 and all(Path(v['path']).is_relative_to(a.cache.resolve()) for v in handle.loads.values())
        tensor=lambda x:torch.tensor(x,device=a.device,dtype=torch.float64)
        analytic=0
        for order in range(1,7):
            for ci,ui in itertools.product([[2.,2.],[-2.,-2.],[1.5,2.5],[-2.5,-1.5]], [[-.25,.125],[1.5,2.],[-3.,-2.],[0.,0.]]):
                C,U=tensor([ci]),tensor([ui]);rec,badc=elem._checked_rec(C);tail,bad=core.normalized_tail(C,rec,U,order)
                assert not bool((bad|badc).any())
                for c,u in itertools.product([F(ci[0]),sum(map(F,ci))/2,F(ci[1])],[F(ui[0]),sum(map(F,ui))/2,F(ui[1])]):
                    value=(-u)**order/(c*(1+u));assert F(float(tail[0,0]))<=value<=F(float(tail[0,1]))
                    assert sum(((-u)**i for i in range(order)),F(0))/c+value==1/(c*(1+u));analytic+=1
        negatives=[]
        for name,ci,ui in [('C_zero',[0.,0.],[0.,0.]),('C_cross',[-1.,1.],[0.,0.]),
                ('cached_reciprocal_overflow',[float.fromhex('0x0.0000000000001p-1022')]*2,[0.,0.]),
                ('pole_cross',[2.,2.],[-1.5,-.5]),('pole_touch',[2.,2.],[-1.,-.5]),
                ('nonfinite',[float('inf')]*2,[0.,0.]),('reversed',[2.,1.],[0.,0.])]:
            C,U=tensor([ci]),tensor([ui]);rec,oldbad=elem._checked_rec(C);tail,bad=core.normalized_tail(C,rec,U,2)
            assert bool(bad.all()),name;negatives.append(name)
        rows=[];oracle=0;cuda_rows=[];counterexamples=0
        cases=[('positive',2.,[0.,0.]),('negative',-2.,[0.,0.]),('small_positive',.5,[-.015625,.03125]),
               ('small_negative',-.5,[-.03125,.015625]),('nonzero_R',2.,[.03125,.0625]),
               ('U_large_positive',1.,[3.,4.]),('U_large_negative',1.,[-4.,-3.])]
        for k in range(1,5):
            basis=max(k,3);tables=build_tables(1,basis).to(a.device);step=build_step_tables(tables,.005)
            eng=sp.SparseEngine(tables,step,a.device);exps=tables.exponents.cpu().tolist();exps_t=[tuple(v) for v in exps]
            sup=sp.make_support(1,basis,False,tuple(range(tables.T)));code=oc.compile_ode(['1/x1'],['x1'],order=k)
            spec=se.specialize(code,(sup,),k,sup,eng);out_exps=[exps[i] for i in spec.sup_out_union.ids]
            x=torch.zeros(len(cases),1,tables.T,device=a.device);r=tensor([v[2] for v in cases]).unsqueeze(1)
            for b,(_,c,_) in enumerate(cases):x[b,0,0]=c
            x[:,0,exps_t.index((0,1))]=.125;x[:,0,exps_t.index((0,3))]=.015625
            initial=[x.clone(),r.clone()]
            def execute(route,fn):
                attr='rec_series_valid_g' if route=='generic' else 'rec_series_valid'
                current=getattr(elem,attr);setattr(elem,attr,fn);bad=torch.zeros(len(x),device=a.device,dtype=torch.bool)
                try:
                    values=se.exec_valid_s(spec,code,x,r,eng,0.,elem.elem_tables(k,a.device),bad) if route=='generic' else oc.exec_valid(code,x,r,k,tables,step,0.,bad)
                finally:setattr(elem,attr,current)
                return tuple(v.detach().clone() for v in (*values,bad))
            outputs={}
            for route,attr in [('generic','rec_series_valid_g'),('dense','rec_series_valid')]:
                old,p_old=capture(handle.original_functions[attr],lambda:execute(route,handle.original_functions[attr]))
                new,p_new=capture(getattr(core,attr),lambda:execute(route,getattr(core,attr)))
                bare=execute(route,getattr(core,attr));assert all(equal(v,w) for v,w in zip(new,bare))
                assert p_old.keys()==p_new.keys() and all(equal(v,p_new[key]) for key,v in p_old.items())
                assert all(equal(old[i],new[i]) for i in (0,2,3)) and not new[-1].any()
                assert bool((new[3]!=0).any()) if k<3 else True
                oe=out_exps if route=='generic' else exps[:tables.T]
                checked=0
                for b,(_,c,ri) in enumerate(cases):
                    for z,rr in itertools.product([F(-1),F(-1,2),F(0),F(1,2),F(1)],[F(ri[0]),sum(map(F,ri))/2,F(ri[1])]):
                        truth=1/(F(c)+F(.125)*z+F(.015625)*z**3+rr);lo,hi=polynomial(new[0][b,0].cpu(),oe,z)
                        assert lo+F(float(new[1][b,0,0]))<=truth<=hi+F(float(new[1][b,0,1])),(k,route,b,z,rr)
                        oracle+=1;checked+=1
                def replay(fn,rr,cache=new[2],tails=new[3]):
                    current=elem.rec_series_replay;elem.rec_series_replay=fn;bad=torch.zeros(len(rr),device=a.device,dtype=torch.bool)
                    try:value=oc.exec_replay(code,rr,cache,tails,True,bad)
                    finally:elem.rec_series_replay=current
                    return value.clone(),bad.clone()
                replay_count=0
                for scale in (1.,.5):
                    rr=r*scale
                    old_replay,old_pre=capture(handle.original_functions['rec_series_replay'],lambda:replay(handle.original_functions['rec_series_replay'],rr))
                    new_replay,new_pre=capture(core.rec_series_replay,lambda:replay(core.rec_series_replay,rr))
                    assert old_pre.keys()==new_pre.keys() and all(equal(v,new_pre[key]) for key,v in old_pre.items())
                    assert all(equal(v,w) for v,w in zip(new_replay,replay(core.rec_series_replay,rr)))
                    assert not new_replay[1].any()
                    for b,(_,c,ri) in enumerate(cases):
                        for z,e in itertools.product([F(-1),F(0),F(1)],(0,1)):
                            truth=1/(F(c)+F(.125)*z+F(.015625)*z**3+F(float(rr[b,0,e])));lo,hi=polynomial(new[0][b,0].cpu(),oe,z)
                            assert lo+F(float(new_replay[0][b,0,0]))<=truth<=hi+F(float(new_replay[0][b,0,1])),('replay',k,route,b,z)
                            oracle+=1;replay_count+=1
                # Corrupt only cached C, retaining a finite sanitized reciprocal.
                corrupt=new[2].clone();base=next(ins.cache_base for ins in code.instrs if ins.op=='div')
                corrupt[0,base]=float.fromhex('0x0.0000000000001p-1022');corrupt[0,base+1]=0.
                _,bad=replay(core.rec_series_replay,r,corrupt);assert bad[0] and not bad[1:].any()
                outputs[route]=new;rows.append(dict(order=k,route=route,valid_fraction_checks=checked,replay_fraction_checks=replay_count,
                    old_new_coeff_cache_strict_tails_bytes_equal=True,prefix_fields_bytes_equal=True,instrumented_bare_bytes_equal=True,
                    cached_overflow_bad_retained=True,nonzero_strict_tail=bool((new[3]!=0).any())))
            assert equal(x,initial[0]) and equal(r,initial[1])
            # Real generic/dense mixed pole masks, with an unchanged valid lane.
            original_x=x;original_r=r
            x=original_x[:3].clone();r=torch.zeros(3,1,2,device=a.device);x[:2]=0.;x[0,0,0]=0.;x[1,0,0]=1.;x[1,0,exps_t.index((0,1))]=2.
            for route,attr in [('generic','rec_series_valid_g'),('dense','rec_series_valid')]:
                mixed=execute(route,getattr(core,attr));assert mixed[-1].tolist()==[True,True,False]
                mx,mr=x,r;x,r=mx[2:].clone(),mr[2:].clone();alone=execute(route,getattr(core,attr));x,r=mx,mr
                assert all(equal(v[2:],w) for v,w in zip(mixed,alone))
                if a.device=='cuda' and route=='generic':
                    vm=tk.serialize_valid_tape(spec,code,eng,0.,a.device);mb=torch.zeros(3,device=a.device,dtype=torch.bool)
                    value=(*tk.exec_valid_tape(vm,x,r,mb),mb)
                    assert all(equal(v,w) for v,w in zip(value,mixed)),'CUDA mixed pole lanes differ'
            x,r=original_x,original_r
            # CPU checks lowering identity without compiling any CUDA source.
            old_vt=handle.original_tape.serialize_valid_tape(spec,code,eng,0.,a.device)
            vt=tk.serialize_valid_tape(spec,code,eng,0.,a.device)
            assert len(old_vt.opc)-len(vt.opc)==2
            assert int((vt.opc==tk._VOP['LAG_REC']).sum())==1
            if a.device=='cuda':
                bad=torch.zeros(len(x),device=a.device,dtype=torch.bool);value=(*tk.exec_valid_tape(vt,x,r,bad),bad)
                comparisons={name:equal(v,w) for name,v,w in zip(('coeff','remainder','cache','strict_tails','bad'),value,outputs['generic'])}
                assert all(comparisons.values()),comparisons
                dispatch=se._graphed_valid(spec,code,x,r,torch.zeros_like(bad),eng,0.,elem.elem_tables(k,a.device))
                assert all(equal(v,w) for v,w in zip(value,dispatch)) and se._eng_cache(eng,'_vtape')[id(spec)] is not None
                from flowstar_gpu import flowpipe,config
                accepted=tensor([[[-.03125,.03125]]]*len(x));ok=torch.ones_like(bad);zero=torch.zeros_like(accepted)
                settings=config.Settings(step=.005,order=basis,cutoff=0.,remainder_estimation=.1,mode='strict',device='cuda',max_refinement_steps=1)
                replay_tape=tk.serialize_replay_tape(code,'cuda')
                expected=flowpipe.refine_loop(code,accepted,ok,torch.zeros_like(bad),value[2],value[3],True,zero,step,settings)
                actual=tk.refine_tape_run(replay_tape,accepted,ok,torch.zeros_like(bad),value[2],value[3],True,zero,(0.,.005),settings.stop_ratio,1)
                assert all(equal(v,w) for v,w in zip(actual[:2],expected))
                corrupt=value[2].clone();base=next(ins.cache_base for ins in code.instrs if ins.op=='div')
                corrupt[0,base]=float.fromhex('0x0.0000000000001p-1022');corrupt[0,base+1]=0.
                invalid=tk.refine_tape_run(replay_tape,accepted,ok,torch.zeros_like(bad),corrupt,value[3],True,zero,(0.,.005),settings.stop_ratio,1)
                invalid_expected=flowpipe.refine_loop(code,accepted,ok,torch.zeros_like(bad),corrupt,value[3],True,zero,step,settings)
                assert invalid[1][0] and all(equal(v,w) for v,w in zip(invalid[:2],invalid_expected))
                dispatched=se._refine_dispatch(code,accepted,ok,torch.zeros_like(bad),value[2],value[3],True,zero,step,settings,eng)
                assert all(equal(v,w) for v,w in zip(actual[:2],dispatched)) and se._eng_cache(eng,'_rtape')
                cuda_rows.append(dict(order=k,valid_microtape_all_fields_equal=comparisons,actual_valid_tape_dispatch=True,
                    python_cuda_replay_cur_bad_bytes_equal=True,actual_replay_tape_dispatch=True))
        result=dict(status='passed',device=a.device,torch_version=torch.__version__,candidate=handle.metadata,build=build,extensions=handle.loads,
            analytic_fraction_witnesses=analytic,full_output_fraction_witnesses=oracle,domain_negatives=negatives,rows=rows,cuda_rows=cuda_rows,
            original_prefix_AST_equal_all_three_routes=True,all_five_routes_installed=True,
            no_cuda_compile_or_gpu_execution=a.device=='cpu',build_result_sha256=sha(a.build_result) if a.build_result else None,
            source_files={name:sha(Path(__file__).with_name(name)) for name in ('core.py','install.py')},
            scope='Bounded numerical/dispatch qualification, not an actual QUAD advance or end-to-end NNCS certificate.')
    except BaseException as e:result.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        if handle is not None:handle.restore()
        result.update(script_sha256=sha(__file__),process_body_s=time.perf_counter()-start)
        (a.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k in ('status','error','device','analytic_fraction_witnesses','full_output_fraction_witnesses','process_body_s')}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--common',type=Path);p.add_argument('--cache',type=Path,required=True);p.add_argument('--allow-build',action='store_true');p.add_argument('--build-result',type=Path)
    run(p.parse_args())
