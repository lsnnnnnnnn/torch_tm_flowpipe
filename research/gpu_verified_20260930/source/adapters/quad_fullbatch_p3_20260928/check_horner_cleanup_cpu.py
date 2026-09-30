"""Small original/candidate Horner byte and object-lifetime CPU checks.

No CUDA, engine source edits, global GC policy change outside this process, or
full trajectory qualification. Exception checks release the traceback first.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,gc,hashlib,importlib.util,json,os,sys,time,traceback,weakref

HERE=Path(__file__).parent
ADAPTER_SHA='f98958be75a46d49444d98b5c196ecef997ce6f63b70bfd2e541ab4e62ff0ad0'
EXEC_SHA='56e590796dbae42685b0331c7f41b53b62a3f73ed9be74e22738493820353ddf'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();result=dict(status='exception')
    was_enabled=gc.isenabled();old_profile=sys.getprofile();binding=None
    try:
        assert sha(HERE/'horner_closure_cleanup.py')==ADAPTER_SHA
        assert sha(args.engine_root/'src/flowstar_gpu/sparse_exec.py')==EXEC_SHA
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['FLOWSTAR_COMPOSITION']='horner'
        sys.path.insert(0,str(args.engine_root/'src'))
        import torch
        from flowstar_gpu import sparse_exec as se,sparse_horner as sh,support as sp
        from flowstar_gpu.monomials import build_tables
        from flowstar_gpu.polynomial import build_step_tables
        from flowstar_gpu.composition import build_schedule
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        spec=importlib.util.spec_from_file_location('isolated_horner_cleanup',HERE/'horner_closure_cleanup.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        assert sha(sh.__file__)==adapter.SOURCE_SHA and se.COMPOSITION_MODE=='horner'
        tabs=build_tables(2,3);eng=sp.SparseEngine(tabs,build_step_tables(tabs,.005),'cpu')
        sched=build_schedule(2,3,'cpu');sup=sp.make_support(2,3,True,range(tabs.Ts))
        exps=sp._exps_for(sup).tolist();slot={tuple(e):i for i,e in enumerate(exps)}
        def fp(tensor):return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        def output_fp(output):return dict(coeff=fp(output[0]),remainder=fp(output[1]),support_ids=list(output[2].ids),support_degrees=list(output[2].degs))
        def inputs(offset):
            a=torch.zeros(2,2,sup.size);g=torch.zeros_like(a)
            a[:,0,slot[(3,0)]]=.125+offset;a[:,1,slot[(1,1)]]=-.25
            a[:,0,0]=.1;a[:,1,slot[(0,2)]]=.3
            g[:,0,slot[(1,0)]]=.5;g[:,1,slot[(0,1)]]=.25
            g[:,0,slot[(2,0)]]=.0625;g[:,1,slot[(1,1)]]=-.03125
            ar=torch.tensor([[[-.001,.002],[-.002,.001]],[[-.003,.002],[-.001,.004]]]);gr=ar.clone()
            return a,ar,g,gr
        def run(offset=0):
            values=inputs(offset);refs={f'input/{i}':weakref.ref(value) for i,value in enumerate(values)}
            original_input=[fp(value) for value in values];target=sh.compose_horner.__code__
            def profile(frame,event,arg):
                if event=='call' and frame.f_code.co_name=='evaluate' and frame.f_back.f_code is target:
                    function=frame.f_back.f_locals['evaluate'];refs['evaluate']=weakref.ref(function)
                    for name,cell in zip(function.__code__.co_freevars,function.__closure__):
                        value=cell.cell_contents
                        if isinstance(value,torch.Tensor):refs['closure/'+name]=weakref.ref(value)
            sys.setprofile(profile)
            try:out=se.compose_s(*values,sup,sup,3,1e-6,True,sched,eng)
            finally:sys.setprofile(old_profile)
            assert original_input==[fp(value) for value in values]
            assert all(bool(torch.isfinite(value).all()) for value in out[:2])
            return out,refs
        alive=lambda refs:{name:ref() is not None for name,ref in refs.items()}
        gc.collect();gc.disable()
        original_out,original_refs=run();original_alive=alive(original_refs)
        assert original_alive['evaluate'] and original_alive['input/0'] and original_alive['input/2'] and original_alive['input/3']
        assert not original_alive['input/1']
        gc.collect();assert not any(alive(original_refs).values())
        binding=adapter.install(sh)
        candidate_out,candidate_refs=run();candidate_alive=alive(candidate_refs)
        assert not any(candidate_alive.values()),candidate_alive
        original_signature=output_fp(original_out);candidate_signature=output_fp(candidate_out)
        assert original_signature==candidate_signature
        changed_out,changed_refs=run(.125)
        assert not any(alive(changed_refs).values())
        assert output_fp(changed_out)!=candidate_signature
        gc.collect();assert output_fp(candidate_out)==candidate_signature and output_fp(original_out)==original_signature
        class InjectedFailure(Exception):pass
        exception_refs={};saved_tree=binding.candidate.__globals__['_tree']
        def fail_tree(support,engine):
            # Retain no frame or strong function reference in the fixture.
            exception_refs['evaluate']=weakref.ref(sys._getframe(1).f_locals['evaluate'])
            raise InjectedFailure('fixture tree failure after evaluate definition')
        def exception_call():
            values=inputs(0)
            for i,value in enumerate(values):exception_refs[f'input/{i}']=weakref.ref(value)
            try:se.compose_s(*values,sup,sup,3,1e-6,True,sched,eng)
            except InjectedFailure:pass
            else:raise AssertionError('Expected fixture exception')
        binding.candidate.__globals__['_tree']=fail_tree
        try:exception_call()
        finally:binding.candidate.__globals__['_tree']=saved_tree
        exception_alive=alive(exception_refs)
        assert not any(exception_alive.values()),exception_alive
        assert not gc.isenabled()
        result=dict(status='passed',device='cpu',torch_version=torch.__version__,batch=2,n=2,working_order=3,
            adapter_sha256=ADAPTER_SHA,original_source_sha256=binding.original_source_sha256,
            original_sparse_exec_sha256=EXEC_SHA,original_function_sha256=binding.original_function_sha256,
            candidate_function_sha256=binding.candidate_function_sha256,inverse_AST_equal=binding.inverse_AST_equal,
            original_candidate_coeff_remainder_support_all_bytes_equal=True,inputs_unchanged=True,
            original_weakrefs_alive_before_gc=original_alive,candidate_weakrefs_alive_without_gc=candidate_alive,
            changed_input_output_changed=True,held_original_and_candidate_outputs_survive_second_call_and_gc=True,
            exception_propagated_to_fixture=True,exception_weakrefs_alive_after_handler=exception_alive,
            exception_scope='Fixture failure in _tree after evaluator definition; traceback not retained. No claim of immediate tensor release while a caller retains an exception traceback.',
            output_signature=candidate_signature,
            scope='Small original CPU n2/B2/P3 nonlinear Horner with nonzero remainders, byte equality and closure lifetime only. No CUDA capture/replay, QUAD memory-peak or trajectory qualification.')
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        sys.setprofile(old_profile)
        if binding is not None:binding.restore();result['original_function_restored']=True
        if was_enabled:gc.enable()
        else:gc.disable()
        result.update(script_sha256=sha(__file__),elapsed_s=time.perf_counter()-start)
        (args.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
