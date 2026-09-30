"""CPU/CUDA differential gate: frozen dense metadata versus isolated sparse tables.

First collect the small dense oracle. Then forbid all imported build_tables
aliases and run sparse P4/P5, including nonempty SR/strict endpoint/refinement.
n16 resource qualification builds active metadata only; no full QUAD advance.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, copy, dataclasses, hashlib, importlib.util, itertools, json, os, random, sys, time

ENV=dict(FLOWSTAR_COMPOSITION='horner',FLOWSTAR_GLUE='graph',FLOWSTAR_SUPPORT_POLICY='structural',
    FLOWSTAR_INJECTIVE_MAPS='1',FLOWSTAR_INJECTIVE_GLUE='1',FLOWSTAR_VALIDATION_POLICY='solution_plus_one',
    FLOWSTAR_CENTER_NORMALIZATION='1',FLOWSTAR_WEIGHTED_VALIDATION='0',FLOWSTAR_RECENTER_VALIDATION='1',
    FLOWSTAR_EARLY_WEIGHTED='1',FLOWSTAR_SELF_MAP_RETRIES='8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    result=dict(status='exception',device=a.device,script_sha256=sha(__file__),scope='metadata and small P4/P5 differential only; no QUAD or NN claim')
    try:
        source=json.loads(a.identity.read_text())
        expected=source.get('engine_python_sha256',source.get('engine',source).get('python_sha256'))
        assert expected and len(expected)==36
        for rel,digest in expected.items():assert sha(a.engine_root/rel)==digest,rel
        os.environ.update(ENV);sys.path.insert(0,str(a.engine_root/'src'))
        import numpy as np
        import torch
        from flowstar_gpu import support as sp,sparse_exec as se,interval as iv,config,monomials,weighted_validation as wv
        from flowstar_gpu.monomials import build_tables
        from flowstar_gpu.polynomial import build_step_tables
        from flowstar_gpu.composition import build_schedule
        from flowstar_gpu.ode_compiler import compile_ode
        from flowstar_gpu.symbolic_remainder import make_symbolic_remainder
        torch.set_default_dtype(torch.float64)
        here=Path(__file__).parent;backend=load('quad_metadata_backend',here/'metadata_backend.py')
        helpers_path=here.parent/'quad_targeted_recovery_20260927/check_order4.py'
        assert sha(helpers_path)=='6a3579b48ce3243e79607ca520ca94554b53c181e3beada3465489c81bece41f'
        helpers=load('metadata_check_order4_helpers',helpers_path)
        endpoint_path=helpers_path.with_name('strict_endpoint.py')
        assert sha(endpoint_path)=='f4310a1b0484fd344ae894641302e146fcd232a8c7b296546a0fcdc94b0cd326'
        strict=load('metadata_dense_endpoint',endpoint_path)
        edge=None
        if a.device=='cuda':
            assert a.common and sha(a.common)=='1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
            common=load('metadata_bootstrap',a.common);assert common.E.resolve()==a.engine_root.resolve()
            edge=common.bootstrap();result['extensions']=common.ORIGINAL['extensions']
        else:torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert all(os.environ[k]==v for k,v in ENV.items())
        result.update(backend_sha256=sha(here/'metadata_backend.py'),engine_python_sha256=expected,
                      torch_version=torch.__version__,environment=ENV)

        def frozen(value):
            if isinstance(value,torch.Tensor):
                raw=value.detach().cpu().contiguous()
                return dict(shape=list(raw.shape),dtype=str(raw.dtype),sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())
            if isinstance(value,np.ndarray):return frozen(torch.from_numpy(value.copy()))
            if dataclasses.is_dataclass(value):
                return {f.name:frozen(getattr(value,f.name)) for f in dataclasses.fields(value) if f.name!='tag'}
            if isinstance(value,dict):return {str(k):frozen(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [frozen(v) for v in value]
            return value
        def fingerprint(st,sr):
            result={k:frozen(getattr(st,k)) for k in ['pre','pre_rem','tmv','tmv_rem','status','pre_sup','tmv_sup']}
            for k in ['scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf']:
                v=getattr(sr,k)
                if v is not None and k.startswith('phi'):v=v[:sr.qlen]
                if v is not None and k=='j_buf':v=v[:sr.jlen]
                result[k]=frozen(v)
            return result|dict(qlen=sr.qlen,jlen=sr.jlen,max_size=sr.max_size,phi_capacity=sr.phi_buf.shape[0],j_capacity=sr.j_buf.shape[0])
        def factory(sparse,n=2,k=4):
            if sparse:eng,sched=backend.engine(n,k,.1,a.device)
            else:
                tab=build_tables(n,k).to(a.device);eng=sp.SparseEngine(tab,build_step_tables(tab,.1),a.device)
                sched=build_schedule(n,k,a.device)
            if edge is not None:eng.horner_edge_kernel=edge.horner_edge
            return eng,sched
        def rows(eng,sup):
            if isinstance(eng,backend.MetadataEngine):return eng.exponents(sup).tolist()
            return sp._exps_for(sup).tolist()
        def plans(eng):
            v=se.validation_engine_for_order(eng,5)
            return [frozen(p) for p in getattr(v,'_weighted_plans',{}).values()]
        graph_receipts=[]
        def graphs(eng,sparse,expression,instrumented):
            owners=[eng,se.validation_engine_for_order(eng,5)];items=[]
            for owner in owners:
                for attr in ['_graphs','_weighted_graphs']:
                    value=getattr(owner,attr,None)
                    if value is not None:
                        items.append(dict(order=owner.tables.k,cache=attr,enabled=value.enabled,
                            captures=value.captures,hits=value.hits,retained=len(value._segs)))
            tapes=sum(len(getattr(owner,'_rtape',{})) for owner in owners)
            if a.device=='cuda':
                captures=sum(v['captures'] for v in items);hits=sum(v['hits'] for v in items)
                assert captures>0 and hits>captures,items
                assert all(v['enabled'] for v in items)
                if not instrumented:assert tapes>0,'standard no-callback replay tape was not exercised'
            graph_receipts.append(dict(sparse=sparse,expression=expression,instrumented=instrumented,caches=items,replay_tapes=tapes))
        settings=config.Settings(step=.1,order=4,cutoff=0.,remainder_estimation=.05,mode='strict',device=a.device)
        cap=torch.tensor([-.05,.05],device=a.device).expand(2,2,2).clone()
        base=torch.tensor([[[.125,.25],[-.25,.375]],[[-.5,-.25],[.25,.5]]],device=a.device)
        grid=list(itertools.product([F(-1),F(-1,2),F(0),F(1,2),F(1)],repeat=2))
        cases=[(3,'z*z*z'),(4,'z*z*z*z'),('reciprocal','1/z'),('sin','sin(z)'),('cos','cos(z)')]
        def trajectories(sparse,instrumented):
            receipts=[];component_checks=0
            for kind,expression in cases:
                eng,sched=factory(sparse);boxes=base.clone()
                if kind=='reciprocal':boxes[:,1]=torch.tensor([[.5,.75],[1.,1.25]],device=a.device)
                state=se.initial_sparse_state(boxes,eng,sched);initial=copy.deepcopy(state)
                sr=make_symbolic_remainder(2,2,1000,a.device);sr.reserve(8)
                code=compile_ode([expression,'0'],['x','z'],order=3)
                checkpoints=[]
                for step in range(1,5):
                    boundary=None
                    if step==3:
                        before=fingerprint(state,sr)
                        error=(backend.endpoint if sparse else strict.end_of_time_s)(state,eng)
                        after=fingerprint(state,sr)
                        for k in before:
                            if k not in ['pre','pre_rem','pre_sup']:assert before[k]==after[k],k
                        boundary=dict(error=frozen(error),state=after)
                    before=fingerprint(state,sr);trial,history=copy.deepcopy(state),copy.deepcopy(sr)
                    trace=[];options=dataclasses.replace(settings,refinement_callback=trace.append) if instrumented else settings
                    value,ok=se.advance_sparse(trial,code,eng,sched,options,cap,history)
                    assert fingerprint(state,sr)==before and bool(ok.all()),(sparse,expression,step)
                    assert history.qlen==history.jlen==step
                    if sr.jlen:assert frozen(sr.j_buf[:sr.jlen])==frozen(history.j_buf[:sr.jlen])
                    state,sr=se.prune_state(value,eng),history
                    v=se.validation_engine_for_order(eng,5);assert v.tables.k==5
                    for z in grid:
                        for lane in range(2):
                            initial_values=[helpers.polynomial(row,rows(eng,initial.pre_sup),[(F(0),F(0))]+[(v,v) for v in z],[0.,0.])[0] for row in initial.pre[lane].cpu()]
                            x0,z0=initial_values;fl,fu=helpers.forcing_range(kind,z0)
                            inner=[helpers.polynomial(row,rows(eng,state.tmv_sup),[(v,v) for v in z],rem) for row,rem in zip(state.tmv[lane].cpu(),state.tmv_rem[lane].cpu())]
                            outer=[helpers.polynomial(row,rows(eng,state.pre_sup),[(F(.1),F(.1))]+inner,rem) for row,rem in zip(state.pre[lane].cpu(),state.pre_rem[lane].cpu())]
                            for bound,(lo,hi) in zip(outer,[(x0+step*F(.1)*fl,x0+step*F(.1)*fu),(z0,z0)]):
                                assert bound[0]<=lo<=hi<=bound[1];component_checks+=1
                    # Different allocations must not have equal pointer addresses.
                    # Preserve within-attempt pointer aliasing plus label/shape;
                    # all numerical trace values remain exact comparison inputs.
                    pointers={}
                    for event in trace:
                        for key in ['cache_id','tails_id']:
                            if key in event:
                                label,address,shape=event[key].split(':',2)
                                pointer=(label,address)
                                if pointer not in pointers:pointers[pointer]=len(pointers)
                                event[key]=f'{label}:allocation{pointers[pointer]}:{shape}'
                    checkpoints.append(dict(step=step,accepted=frozen(ok),state=fingerprint(state,sr),boundary=boundary,trace=trace))
                receipts.append(dict(kind=kind,steps=checkpoints,weighted_plans=plans(eng)))
                graphs(eng,sparse,expression,instrumented)
            negatives=[]
            for expression in ['z*z*z*z','sin(z)','cos(z)','1/z']:
                eng,sched=factory(sparse);boxes=base.clone()
                if expression=='1/z':boxes[:,1]=torch.tensor([-.125,.125],device=a.device)
                state=se.initial_sparse_state(boxes,eng,sched);sr=make_symbolic_remainder(2,2,1000,a.device);sr.reserve(8)
                before=fingerprint(state,sr);trial,history=copy.deepcopy(state),copy.deepcopy(sr)
                code=compile_ode([expression,'0'],['x','z'],order=3)
                value,ok=se.advance_sparse(trial,code,eng,sched,settings,cap if expression=='1/z' else torch.zeros_like(cap),history)
                assert not bool(ok.any()) and fingerprint(state,sr)==before
                negatives.append(dict(expression=expression,accepted=frozen(ok),state=fingerprint(value,history)))
            return dict(cases=receipts,negative=negatives,component_checks=component_checks)

        # Metadata fixtures have arbitrary full/spatial supports and all possible
        # integration denominators, not just the supports arising in five ODEs.
        rng=random.Random(270927);fixtures=[]
        for n,k in [(1,4),(2,4),(2,5),(3,2),(3,3),(3,4),(3,5)]:
            tab=build_tables(n,k).to(a.device);ref=sp.SparseEngine(tab,build_step_tables(tab,.1),a.device)
            for spatial,work in [(False,tab.T),(True,tab.Ts)]:
                sa=sp.make_support(n,k,spatial,tuple(rng.sample(range(work),min(40,work))))
                sb=sp.make_support(n,k,spatial,tuple(rng.sample(range(work),min(30,work))))
                fields=dict(a=sa,b=sb,exponents=sp._exps_for(sa),pair=ref.pair(sa,sb),union=ref.union(sa,sb),
                    embed_plan=ref.embed_plan(sa,ref.union(sa,sb)[0]))
                if spatial:fields.update(cat=ref.cat(sa),full=sp.spatial_to_full_ids(n,k,sa))
                else:fields.update(factor=ref.factor(sa),integ=ref.integ(sa),integ_plan=ref.integ_plan(sa),
                    evalt=ref.evalt(sa),evalt_interval=ref.evalt_interval(sa),evalt_error=ref.evalt_error(sa))
                fixtures.append((n,k,spatial,sa.ids,sb.ids,frozen(fields)))
        dense={mode:trajectories(False,mode=='diagnostic_callback') for mode in ['standard_no_callback','diagnostic_callback']}
        (a.output/'DENSE.json').write_text(json.dumps(dense,indent=2,allow_nan=False)+'\n')
        backend.install(dict(python_sha256=expected))
        def forbidden(*args,**kwargs):raise AssertionError('dense build_tables reached after sparse installation')
        aliases=[]
        for name,module in list(sys.modules.items()):
            if name.startswith('flowstar_gpu') and module is not None and getattr(module,'build_tables',None) is build_tables:
                aliases.append(name);module.build_tables=forbidden
        assert 'flowstar_gpu.monomials' in aliases and 'flowstar_gpu.support' in aliases
        count=0
        for n,k,spatial,ia,ib,expected_fields in fixtures:
            eng,_=backend.engine(n,k,.1,a.device);sa=eng.support(ia,spatial);sb=eng.support(ib,spatial)
            fields=dict(a=sa,b=sb,exponents=eng.exponents(sa),pair=eng.pair(sa,sb),union=eng.union(sa,sb),
                embed_plan=eng.embed_plan(sa,eng.union(sa,sb)[0]))
            if spatial:fields.update(cat=eng.cat(sa),full=eng.spatial_to_full(sa))
            else:fields.update(factor=eng.factor(sa),integ=eng.integ(sa),integ_plan=eng.integ_plan(sa),
                evalt=eng.evalt(sa),evalt_interval=eng.evalt_interval(sa),evalt_error=eng.evalt_error(sa))
            assert frozen(fields)==expected_fields,(n,k,spatial,'metadata byte mismatch')
            count+=sa.size*sb.size
        sparse={mode:trajectories(True,mode=='diagnostic_callback') for mode in ['standard_no_callback','diagnostic_callback']}
        (a.output/'SPARSE.json').write_text(json.dumps(sparse,indent=2,allow_nan=False)+'\n')
        assert sparse==dense,'dense/sparse trajectory, plans, trace, endpoint or complete live SR mismatch'
        eng,sched=backend.engine(16,4,.005,a.device);v=se.validation_engine_for_order(eng,5)
        assert v.tables.k==5 and eng.pair_budget is v.pair_budget
        test=torch.zeros((2,16,2),device=a.device);test[:,:,1]=.125
        state=se.initial_sparse_state(test,eng,sched)
        one=v.support(tuple(v.tables.onehot_ids(False)))
        product=v.pair(one,one)
        for ia in range(one.size):
            for ib in range(one.size):
                expected_id=v.tables.full.rank(tuple(map(int,v.exponents(one)[ia]+v.exponents(one)[ib])))
                candidates=(product.pair_ia==ia)&(product.pair_jb==ib)
                assert int(candidates.sum())==1
                local=int(torch.where(candidates)[0][0]);slot=int(torch.searchsorted(product.seg_offsets[1:],torch.tensor(local,device=a.device),right=True))
                assert product.sup_out.ids[slot]==expected_id
        # Resource refusal is pre-allocation; no fallback to truncated validation.
        limited,_=backend.engine(16,4,.005,a.device);limited.pair_budget['limit']=0
        refused=False
        try:limited.pair(state.pre_sup,state.pre_sup)
        except MemoryError:refused=True
        assert refused and limited.pair_budget['bindings']==0
        if a.device=='cuda':
            torch.cuda.synchronize();result['max_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        result.update(status='passed',metadata_fixtures=len(fixtures),ordered_pairs_compared=count,
            dense_sparse_complete_bytes_equal=True,weighted_plan_fields_equal=True,negative_cases=4,
            standard_no_callback_cases=5,diagnostic_callback_cases=5,graph_receipts=graph_receipts,
            fraction_component_checks_per_backend=sum(v['component_checks'] for v in dense.values()),forbidden_dense_aliases=aliases,
            n16=dict(working_order=4,validation_order=5,initial_pre_support=state.pre_sup.size,
                working_basis=eng.tables.T,validation_basis=v.tables.T,pair_budget=v.pair_budget,
                global_pair_tables_built=False,resource_refusal_before_binding=True))
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['process_s']=time.perf_counter()-started
        (a.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['engine_python_sha256','extensions','graph_receipts']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--common',type=Path);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
