"""Isolated real-root QUAD runner: sparse metadata, unchanged a3fb arithmetic.

P4 requires a passed same-order P3 dense/sparse comparison. First use
--max-steps 1, then 20/40 and cold20->21. Longer runs require both 40 gates.
"""
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction as F
import argparse, hashlib, importlib.util, json, sys, time

RUNNER_SHA='8775959155cebf4b7314ff828bb846f1754f9e53cd5bd41907ee7d7c8c6efc0d'
BACKEND_SHA='3140024445ebac6845d037c6399e2ad8658ae63281eb066c1e87356c99bd990b'
CHECK_SHA='7a4677083946bc96e9a56f3e442027b416eb52bbf548a449d0c2b2c288ef5408'
engines=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def settings(order):
    assert order in [2,3,4]
    return dict(step=.005,order=order,cutoff=1e-6,remainder_estimation=.1,mode='strict',device='cuda')

def save_state(path,st,sr,eng,cap=None,step=None,phase='trial'):
    torch=c.torch
    obj={k:getattr(st,k).detach().cpu() for k in ['pre','pre_rem','tmv','tmv_rem','status']}
    spatial=torch.from_numpy(eng.exponents(st.tmv_sup).copy())
    obj.update(pre_ids=st.pre_sup.ids,tmv_ids=st.tmv_sup.ids,
        pre_exponents=torch.from_numpy(eng.exponents(st.pre_sup).copy()),
        tmv_exponents=torch.cat((torch.zeros((len(spatial),1),dtype=torch.long),spatial),1),
        qlen=sr.qlen,jlen=sr.jlen,sr_max_size=sr.max_size,sr_capacity=sr.phi_buf.shape[0],
        n=eng.tables.n,k=eng.tables.k,h=eng.step.delta,step=step,phase=phase,cap=cap.cpu() if cap is not None else None)
    for k in ['scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf']:
        value=getattr(sr,k)
        if value is not None:
            if k.startswith('phi'):value=value[:sr.qlen]
            if k=='j_buf':value=value[:sr.jlen]
            obj['sr_'+k]=value.detach().cpu()
    torch.save(obj,path)

def restore(path,eng):
    torch=c.torch;saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['n']==eng.tables.n and saved['k']==eng.tables.k and saved['h']==eng.step.delta
    pre=eng.support(saved['pre_ids']);tmv=eng.support(saved['tmv_ids'],True)
    assert torch.equal(saved['pre_exponents'],torch.from_numpy(eng.exponents(pre).copy()))
    spatial=torch.from_numpy(eng.exponents(tmv).copy())
    assert torch.equal(saved['tmv_exponents'],torch.cat((torch.zeros((len(spatial),1),dtype=torch.long),spatial),1))
    st=c.se.SparseState(**{name:saved[name].to(eng.device) for name in ['pre','pre_rem','tmv','tmv_rem','status']},pre_sup=pre,tmv_sup=tmv)
    sr=c.make_symbolic_remainder(len(st.pre),saved['n'],saved['sr_max_size'],eng.device);sr.reserve(saved['sr_capacity'])
    for name in ['scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf']:
        value=saved.get('sr_'+name)
        if value is None:continue
        if name.startswith('phi') or name=='j_buf':
            target=getattr(sr,name)
            if target is None:
                target=torch.empty((saved['sr_capacity'],*value.shape[1:]),dtype=value.dtype,device=eng.device);setattr(sr,name,target)
            target[:len(value)].copy_(value)
        else:setattr(sr,name,value.to(eng.device))
    sr.qlen,sr.jlen=saved['qlen'],saved['jlen'];side=read(path.with_suffix('.json'))
    assert sha(path)==side['state_sha256'] and json.loads(json.dumps(c.fingerprint(st,sr)))==side['state_fingerprint']
    return st,sr,saved['cap'].to(eng.device)

def initial_coverage(st,boxes,eng):
    exps=[tuple(row) for row in eng.exponents(st.pre_sup).tolist()];receipts=[]
    assert not bool(st.pre_rem.any() or st.tmv_rem.any())
    for lane,box in enumerate(boxes):
        rows=[]
        for i,(row,(lo,hi)) in enumerate(zip(st.pre[lane].cpu().tolist(),box)):
            unit=tuple(int(j==i+1) for j in range(17));constant=(0,)*17
            nonzero={e:F(v) for e,v in zip(exps,row) if v};assert set(nonzero)<={constant,unit}
            center=nonzero.get(constant,F(0));radius=nonzero.get(unit,F(0))
            assert radius>=0 and center-radius<=F(lo) and center+radius>=F(hi)
            rows.append(dict(center=float(center),radius=float(radius),low_excess=str(F(lo)-(center-radius)),high_excess=str(center+radius-F(hi))))
        receipts.append(dict(leaf=lane,affine_rows=rows))
    return receipts

def range_over_time(st,eng,piece,idx_hi):
    """Original full working-prefix reduction, using a separate observer view."""
    from flowstar_gpu.safety import rows_range_over_time
    torch=c.torch;sup=eng.full_support(eng.tables.k)
    assert st.pre_sup.n==sup.n and st.pre_sup.k==sup.k and not st.pre_sup.spatial
    assert all(0<=i<eng.tables.T for i in st.pre_sup.ids) and max(st.pre_sup.degs)<=eng.tables.k
    if not hasattr(eng,'_observer_prefix'):
        exps=eng.exponents(sup)
        view=SimpleNamespace(t_deg=torch.tensor(exps[:,0].copy(),dtype=torch.long,device=eng.device),cat_iv=eng._categories(sup))
        eng._observer_prefix=(sup,view)
    original_sup,view=eng._observer_prefix;assert original_sup==sup
    coefficients=c.sp.embed_s(st.pre[:,:idx_hi],eng,st.pre_sup,sup)
    return rows_range_over_time(coefficients,st.pre_rem[:,:idx_hi],piece,view)

def setup(common,order,edge,device='cuda'):
    eng,sched=backend.engine(16,order,.005,device)
    if device=='cuda':eng.horner_edge_kernel=edge.horner_edge
    engines.append(eng);return eng,sched

def long_gate(args,identity):
    if args.periods<=2:return
    assert args.qualification and args.replay_check
    meta=read(args.qualification/'INPUT.json');done=read(args.qualification/'RESULT.json');cold=read(args.replay_check/'RESULT.json')
    assert done['status']=='target_completed' and done['completed_step']==40 and done['input_sha256']==sha(args.qualification/'INPUT.json')
    assert meta['settings']==settings(args.working_order) and meta['mode']=='split_from_zero_strict_endpoint'
    assert meta['root_id']==1 and meta['leaf_count']==2 and meta['controller_policy']=='union_box'
    assert all(meta[k]==v for k,v in identity.items())
    assert cold['status']=='passed' and cold['mode']=='cold_boundary_replay' and cold['working_order']==args.working_order
    assert cold['script_sha256']==sha(__file__) and cold['source_input_sha256']==sha(args.qualification/'INPUT.json')
    assert cold['source_result_sha256']==sha(args.qualification/'RESULT.json')
    assert cold['all_endpoint_bytes_match'] and cold['all_controller_bytes_match'] and cold['all_step21_bytes_match']

def initialize(args):
    global c,b,r,backend
    here=Path(__file__).parent;source=here.parent/'quad_targeted_recovery_20260927/run_order_from_zero.py'
    assert sha(source)==RUNNER_SHA and sha(here/'metadata_backend.py')==BACKEND_SHA and sha(here/'check_backend.py')==CHECK_SHA
    r=load('metadata_frozen_runner',source)
    # Existing initializer verifies original model/config/boxes/extensions and
    # the old strict adapter qualification. No reachability step runs in it.
    c,b,_,edge,driver,model,cfg,identity,model_s=r.initialize(args)
    backend=load('metadata_real_backend',here/'metadata_backend.py')
    gate=read(args.metadata_check/'RESULT.json')
    assert gate['status']=='passed' and gate['device']=='cuda' and gate['backend_sha256']==BACKEND_SHA and gate['script_sha256']==CHECK_SHA
    assert gate['engine_python_sha256']==c.ORIGINAL['engine']['python_sha256'] and gate['torch_version']==c.torch.__version__
    assert gate['dense_sparse_complete_bytes_equal'] and gate['weighted_plan_fields_equal']
    assert gate['standard_no_callback_cases']==gate['diagnostic_callback_cases']==5
    assert gate['fraction_component_checks_per_backend']==4000 and gate['n16']['global_pair_tables_built'] is False
    backend.install(c.ORIGINAL)
    original_build=c.build_tables
    def forbidden(*a,**k):raise RuntimeError('global dense metadata is forbidden for this run')
    aliases=[]
    for name,module in list(sys.modules.items()):
        if module is not None and getattr(module,'build_tables',None) is original_build:
            aliases.append(name);module.build_tables=forbidden
    assert 'flowstar_gpu.monomials' in aliases and 'flowstar_gpu.support' in aliases
    c.save_state=save_state;b.restore=restore;b.initial_coverage=initial_coverage
    driver.end_of_time_s=backend.endpoint;driver.rows_range_over_time_sparse=range_over_time
    r.setup=setup;r.settings=settings;r.long_gate=long_gate;r.ENDPOINT_SHA=BACKEND_SHA
    identity.update(dense_endpoint_adapter_sha256=identity['endpoint_adapter_sha256'],
        endpoint_adapter_sha256=BACKEND_SHA,script_sha256=sha(__file__),frozen_order_runner_sha256=RUNNER_SHA,
        metadata_backend_sha256=BACKEND_SHA,metadata_check_script_sha256=CHECK_SHA,
        metadata_check_path=str(args.metadata_check),metadata_check_result_sha256=sha(args.metadata_check/'RESULT.json'),
        metadata_variant='on-demand graded-lex; unchanged a3fb arithmetic/policy',
        metadata_dependency_sha256={name:sha(here/name) for name in ['graded_lex.py','legacy_sparse_metadata.py']},
        metadata_pair_max=4_000_000,metadata_pair_retained_budget_bytes=2*1024**3,
        original_dense_builders_disabled=True)
    if args.working_order==4:
        assert args.p3_parity,'P4 requires the actual same-order P3 dense/sparse parity gate'
        parity=read(args.p3_parity/'RESULT.json')
        assert parity['status']=='passed' and parity['mode']=='p3_dense_sparse_parity'
        assert parity['script_sha256']==sha(here/'compare_p3.py')
        assert parity['metadata_backend_sha256']==BACKEND_SHA and parity['sparse_runner_sha256']==sha(__file__)
        assert parity['cold20_to21_all_bytes_equal'] and sha(parity['cold_result_path'])==parity['cold_result_sha256']
        sparse_input=Path(parity['sparse_input_path']);assert sha(sparse_input)==parity['sparse_input_sha256']
        pm=read(sparse_input)
        for key in ['config_sha256','model_sha256','driver_sha256','boxes_sha256','engine','extensions']:
            assert pm[key]==identity[key],key
        identity.update(p3_parity_path=str(args.p3_parity),p3_parity_result_sha256=sha(args.p3_parity/'RESULT.json'),
                        p3_parity_script_sha256=sha(here/'compare_p3.py'))
    return c,b,SimpleNamespace(end_of_time_s=backend.endpoint),edge,driver,model,cfg,identity,model_s

def one_step(args,out,ctx):
    c,b,_,edge,driver,model,cfg,identity,model_s=ctx;torch=c.torch;order=args.working_order
    eng,sched=setup(c,order,edge);leaves,cover=b.root_leaves(read(identity['boxes_path']),'split')
    state=c.se.initial_sparse_state(torch.tensor(leaves,dtype=torch.float64,device='cuda'),eng,sched)
    sr=c.make_symbolic_remainder(2,16,1000,'cuda');sr.reserve(1000)
    cap=torch.tensor([-.1,.1],dtype=torch.float64,device='cuda').expand(2,16,2).clone()
    meta=dict(identity,mode='metadata_one_step_resource_probe',root_id=1,leaf_count=2,coverage=cover,
        initial_affine_coverage=initial_coverage(state,leaves,eng),settings=settings(order),
        working_total_degree=order,spatial_composition_order=order,point_code_order=order-1,validation_order=order+1,
        controller_policy='union_box',target_step=1,end_to_end_strict_certificate=False,
        limitation='CROWN and RN controller injection remain unqualified; this is one root B2 only')
    write(out/'INPUT.json',meta);held=r.refresh(c,b,state,sr,eng,driver,model,cfg,0,out)
    r.checkpoint(c,b,out/'committed_0.pt',state,sr,eng,cap,0,held,meta,r.AFTER_CONTROLLER)
    code=c.compile_ode(cfg['dynamics_expressions'],[row['name'] for row in cfg['initial_set']],order=order-1)
    value,ok,history,elapsed=c.attempt(state,sr,code,eng,sched,c.config.Settings(**settings(order)),cap)
    if bool(ok.all()):
        value=c.se.prune_state(value,eng);assert history.reset_if_full() is False
        r.checkpoint(c,b,out/'committed_1.pt',value,history,eng,cap,1,held,meta)
        observation=r.observe(c,driver,value,eng)
    else:
        save_state(out/'failed_trial_1.pt',value,history,eng,cap,1,'speculative_no_commit');observation={}
    return dict(status='target_completed' if bool(ok.all()) else 'numerical_rejection',mode='one_step_resource_probe',
        working_order=order,target_step=1,completed_step=int(bool(ok.all())),accepted=ok.cpu().tolist(),
        advance_s=elapsed,controller_setup_s=model_s,controller=held,observation=observation,input_sha256=sha(out/'INPUT.json'))

def main(args):
    assert 1<=args.max_steps<=1000 and (args.max_steps==1 or args.max_steps%20==0)
    args.periods=max(1,args.max_steps//20);args.check_local=False
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();result=dict(status='exception')
    try:
        ctx=initialize(args)
        result=r.replay(args,out,ctx) if args.replay else one_step(args,out,ctx) if args.max_steps==1 else r.experiment(args,out,ctx)
    except BaseException as exc:
        result.update(status='exception',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result.update(script_sha256=sha(__file__),metadata_backend_sha256=BACKEND_SHA,
            frozen_order_runner_sha256=RUNNER_SHA,metadata_check_script_sha256=CHECK_SHA,
            metadata_dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in ['graded_lex.py','legacy_sparse_metadata.py']},
            process_body_s=time.perf_counter()-started,pair_budgets=[dict(eng.pair_budget) for eng in engines])
        if 'c' in globals():result['max_cuda_allocated_bytes']=c.torch.cuda.max_memory_allocated()
        write(out/'RESULT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--working-order',type=int,choices=[2,3,4],default=3)
    p.add_argument('--max-steps',type=int,default=1);p.add_argument('--endpoint-check',type=Path,required=True)
    p.add_argument('--metadata-check',type=Path,required=True);p.add_argument('--p3-parity',type=Path)
    p.add_argument('--qualification',type=Path);p.add_argument('--replay-check',type=Path)
    p.add_argument('--replay',type=Path);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
