"""Small CPU wiring differential; no QUAD/fullbatch/CUDA qualification claim."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time, traceback

ENV=dict(FLOWSTAR_COMPOSITION='horner',FLOWSTAR_GLUE='graph',FLOWSTAR_SUPPORT_POLICY='structural',
    FLOWSTAR_INJECTIVE_MAPS='1',FLOWSTAR_INJECTIVE_GLUE='1',FLOWSTAR_VALIDATION_POLICY='solution_plus_one',
    FLOWSTAR_CENTER_NORMALIZATION='1',FLOWSTAR_WEIGHTED_VALIDATION='0',FLOWSTAR_RECENTER_VALIDATION='1',
    FLOWSTAR_EARLY_WEIGHTED='1',FLOWSTAR_SELF_MAP_RETRIES='8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception',device='cpu')
    try:
        identity=json.loads(a.identity.read_text());identity=identity.get('source_identity',identity)
        for rel,digest in identity['engine']['python_sha256'].items():assert sha(a.engine_root/rel)==digest
        os.environ.update(ENV);sys.path.insert(0,str(a.engine_root/'src'))
        import torch
        from flowstar_gpu import support as sp,sparse_exec as se,config,monomials
        from flowstar_gpu.polynomial import build_step_tables
        from flowstar_gpu.composition import build_schedule
        from flowstar_gpu.ode_compiler import compile_ode
        from flowstar_gpu.symbolic_remainder import make_symbolic_remainder
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        def fp(v):
            if isinstance(v,torch.Tensor):
                c=v.detach().cpu().contiguous();return dict(shape=list(c.shape),dtype=str(c.dtype),sha256=hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest())
            if isinstance(v,sp.Support):return dict(n=v.n,k=v.k,spatial=v.spatial,ids=v.ids,degs=v.degs,prefix=v.prefix_by_deg)
            if isinstance(v,dict):return {k:fp(x) for k,x in v.items()}
            if isinstance(v,(tuple,list)):return [fp(x) for x in v]
            return v
        settings=config.Settings(step=.005,order=3,cutoff=1e-6,remainder_estimation=.1,mode='strict',device='cpu')
        cap=torch.tensor([-.1,.1]).expand(2,2,2).clone()
        def trajectory():
            tabs=monomials.build_tables(2,3);eng=sp.SparseEngine(tabs,build_step_tables(tabs,.005),'cpu')
            sched=build_schedule(2,3,'cpu');code=compile_ode(['sin(z)+z*z*z','0'],['x','z'],order=2)
            boxes=torch.tensor([[[.125,.25],[-.25,.375]],[[-.5,-.25],[.25,.5]]])
            st=se.initial_sparse_state(boxes,eng,sched);sr=make_symbolic_remainder(2,2,1000,'cpu');sr.reserve(4)
            records=[]
            for step in range(1,4):
                st,ok=se.advance_sparse(st,code,eng,sched,settings,cap,sr);assert bool(ok.all())
                st=se.prune_state(st,eng)
                records.append(fp(dict(state={k:getattr(st,k) for k in ['pre','pre_rem','tmv','tmv_rem','pre_sup','tmv_sup','status']},
                    accepted=ok,scalars=sr.scalars,scalars_iv=sr.scalars_iv,Phi=sr.phi_buf[:sr.qlen],
                    Phi_iv=sr.phi_iv_buf[:sr.qlen],J=sr.j_buf[:sr.jlen],qlen=sr.qlen,jlen=sr.jlen)))
            return records,eng
        dense,deng=trajectory();del deng
        here=Path(__file__).parent;adapter=load('fullbatch_hybrid_check_adapter',here/'hybrid_metadata.py')
        binding=adapter.install(identity,a.backend)
        hybrid,heng=trajectory();assert dense==hybrid
        v=se.validation_engine_for_order(heng,4)
        assert isinstance(v,binding.backend.MetadataEngine) and se.validation_engine_for_order(heng,4) is v
        assert se.validation_engine_for_order(heng,3) is heng and v.tables.k==4 and heng.tables.k==3
        assert len(binding.receipts)==1
        negative=[]
        for label,fn in [('dense4 blocked',lambda:monomials.build_tables(16,4)),
                         ('dense support alias4 blocked',lambda:sp.build_tables(16,4)),
                         ('validation5 rejected',lambda:se.validation_engine_for_order(heng,5)),
                         ('support2 rejected',lambda:sp.make_support(2,2,False,(0,)))]:
            try:fn()
            except (AssertionError,ValueError,RuntimeError):negative.append(label)
            else:raise AssertionError(label)
        result=dict(status='passed',device='cpu',script_sha256=sha(__file__),adapter_sha256=sha(here/'hybrid_metadata.py'),
            backend_sha256=adapter.BACKEND_SHA,engine_python_sha256=identity['engine']['python_sha256'],torch_version=torch.__version__,
            working_order=3,point_code_order=2,validation_order=4,steps=3,all_state_support_SR_acceptance_bytes_equal=True,
            nonempty_SR=True,validation_owner_and_cache_distinct=True,working_dense_support_factories_retained=True,
            prefix_receipts=binding.receipts,guarded_dense_aliases=binding.guarded_dense_aliases,negative_cases_rejected=negative,
            scope='CPU n2/B2 three accepted polynomial+sin steps; dense3/dense4 versus dense3/on-demand4. No QUAD, fullB, new reciprocal arithmetic, CUDA dispatch or throughput claim.')
    except BaseException as e:result.update(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        result['process_s']=time.perf_counter()-start
        (a.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['engine-root','identity','backend','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
