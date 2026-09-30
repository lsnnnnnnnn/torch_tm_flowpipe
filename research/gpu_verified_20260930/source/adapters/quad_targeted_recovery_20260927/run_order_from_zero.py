"""Matched root1 x5-split B2 runs from zero, explicit P2/P3 and strict boundaries.

Both orders use the same original plant/controller settings except working
order. Forty steps and an exact cold20->strict endpoint->NN->21 gate precede
long runs. No engine or frozen driver is modified; no failed leaf is dropped.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, copy, hashlib, importlib.util, json, os, sys, time

OLD=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_split_multistep_20260927')
COMMON_SHA='1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
BASE_SHA='faf2e9c2a893e66bb7fed68bd1e813ff40a156111dcb9fc41a5860952977dcc2'
ENDPOINT_SHA='f4310a1b0484fd344ae894641302e146fcd232a8c7b296546a0fcdc94b0cd326'
ENDPOINT_CHECK_SHA='7eb2c1a217238eadcea04d8a373eb00c4eb48b219de9fac2bf3d8cb5db69d188'
PHASE='committed_before_endpoint_handoff'
AFTER_ENDPOINT='strict_endpoint_complete_before_controller'
AFTER_CONTROLLER='committed_after_controller_injection'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def settings(order):
    if order not in [2,3]:raise ValueError('working order4 is not qualified here; no P4 run is started')
    return dict(step=.005,order=order,cutoff=1e-6,remainder_estimation=.1,mode='strict',device='cuda')


def setup(c,order,edge,device='cuda'):
    settings(order)
    tab=c.build_tables(16,order).to(device)
    eng=c.sp.SparseEngine(tab,c.build_step_tables(tab,.005),device)
    if device=='cuda':eng.horner_edge_kernel=edge.horner_edge
    return eng,c.build_schedule(16,order,device)


def schedule(step,qlen,phase):
    assert 0<=step<=1000 and qlen==step%1000
    return dict(completed_step=step,next_step=step+1,phase=phase,
        next_control_refresh_step=step if step%20==0 and phase in [PHASE,AFTER_ENDPOINT] else (step//20+1)*20,
        sr_history_origin_step=step-qlen,qlen=qlen,jlen=qlen)


def checkpoint(c,b,path,st,sr,eng,cap,step,held,meta,phase=PHASE):
    assert sr.qlen==sr.jlen
    c.save_state(path,st,sr,eng,cap,step,phase)
    write(path.with_suffix('.json'),dict(metadata=meta,held_controller=held,
        **schedule(step,sr.jlen,phase),state_sha256=sha(path),state_fingerprint=c.fingerprint(st,sr)))


def boundary(c,b,adapter,st,sr,eng,completed,out):
    assert completed>0 and completed%20==0 and sr.qlen==sr.jlen==completed%1000
    before=c.fingerprint(st,sr);error=adapter.end_of_time_s(st,eng);after=c.fingerprint(st,sr)
    unchanged=['tmv','tmv_rem','status','tmv_support','qlen','jlen']+[k for k in before if k.startswith('sr_')]
    assert all(before[k]==after[k] for k in unchanged)
    path=out/f'endpoint_at_{completed}.pt'
    c.torch.save(dict(completed_step=completed,working_order=eng.tables.k,error=error.detach().cpu()),path)
    receipt=dict(completed_step=completed,adapter_sha256=ENDPOINT_SHA,error_path=str(path),error_sha256=sha(path),
        before=before,after=after,tmv_status_all_live_sr_bytes_unchanged=True)
    write(path.with_suffix('.json'),receipt)
    return receipt


def refresh(c,b,state,sr,eng,driver,model,cfg,completed,out):
    before=c.fingerprint(state,sr);physical=state.pre[:,:13].clone();rem=state.pre_rem[:,:13].clone()
    held=b.controller_refresh(state,eng,driver,model,cfg,completed,out);after=c.fingerprint(state,sr)
    unchanged=['pre_support','tmv','tmv_rem','tmv_support','status','qlen','jlen']+[k for k in before if k.startswith('sr_')]
    assert all(before[k]==after[k] for k in unchanged)
    assert b.byte_equal(physical,state.pre[:,:13]) and b.byte_equal(rem,state.pre_rem[:,:13])
    write(out/f'injection_at_{completed}.json',dict(completed_step=completed,before=before,after=after,
        physical_clock_rows_and_live_sr_unchanged=True,controller_sha256=held['sha256']))
    return held


def observe(c,driver,state,eng):
    torch=c.torch;started=time.perf_counter();tube=driver.hull_ranges_s(state,eng,12)
    endpoint=driver.rows_range_over_time_sparse(state,eng,torch.full((2,2),.005,dtype=torch.float64,device='cuda'),12)
    assert bool(torch.isfinite(tube).all() and torch.isfinite(endpoint).all())
    return dict(tube=tube.cpu().tolist(),endpoint=endpoint.cpu().tolist(),
        union_tube=torch.stack((tube[...,0].amin(0),tube[...,1].amax(0)),-1).cpu().tolist(),
        union_endpoint=torch.stack((endpoint[...,0].amin(0),endpoint[...,1].amax(0)),-1).cpu().tolist(),
        observer_s=time.perf_counter()-started)


def initialize(args):
    here=Path(__file__).parent
    assert sha(OLD/'root_from_zero.py')==BASE_SHA and sha(OLD/'continuation.py')==COMMON_SHA
    assert sha(here/'strict_endpoint.py')==ENDPOINT_SHA and sha(here/'check_strict_endpoint.py')==ENDPOINT_CHECK_SHA
    b=load('order_root_frozen',OLD/'root_from_zero.py');c=load('order_common_frozen',OLD/'continuation.py');b.c=c
    adapter=load('order_strict_endpoint',here/'strict_endpoint.py')
    gate=read(args.endpoint_check/'RESULT.json');original=c.ORIGINAL
    assert gate['status']=='passed' and gate['device']=='cuda' and gate['adapter_sha256']==ENDPOINT_SHA
    assert gate['check_script_sha256']==ENDPOINT_CHECK_SHA and gate['engine_python_sha256']==original['engine']['python_sha256']
    assert gate['torch_version']==c.torch.__version__
    assert all(gate[k] for k in ['nonempty_sr_boundary_unchanged','fresh_compose_received_exact_charged_remainder',
        'normal_single_J_append','old_J_bytes_unchanged','ordinary_remainder_nonzero'])
    assert gate['deliberately_omitted_marker_misses']>0
    for path,value in [(original['config'],original['config_sha256']),(original['model'],original['model_sha256']),
                       (original['driver_path'],original['driver_sha256'])]:assert sha(path)==value
    boxes_path=c.N/'runs/archcomp_failure_20260923/huan_box_parity_preload/boxes.json'
    assert sha(boxes_path)==b.BOXES_SHA==original['boxes_sha256']
    cfg_path=Path(original['config']);cfg=c.yaml.safe_load(cfg_path.read_text());cfg['_config_dir']=cfg_path.parent
    assert str((cfg_path.parent.parent/cfg['model_dir']).resolve())==original['model']
    assert float(cfg['ode_step_size'])==.005 and int(cfg['ode_order'])==2 and float(cfg['cut_off_threshold'])==1e-6
    assert cfg['remainder_estimation']==[-.1,.1] and cfg['sr_queue']==1000 and cfg['step_size']==.1
    assert len(cfg['initial_set'])==cfg['num_vars']==16 and cfg['num_nn_input']==12 and cfg['num_nn_output']==3
    edge=c.bootstrap();torch=c.torch;torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert c.se.VALIDATION_POLICY=='solution_plus_one'
    for key,value in {'FLOWSTAR_EARLY_WEIGHTED':'1','FLOWSTAR_RECENTER_VALIDATION':'1',
                      'FLOWSTAR_CENTER_NORMALIZATION':'1','FLOWSTAR_WEIGHTED_VALIDATION':'0','FLOWSTAR_SELF_MAP_RETRIES':'8'}.items():
        assert os.environ[key]==value
    driver=b.load_module('order_official_driver',Path(original['driver_path']))
    driver.end_of_time_s=adapter.end_of_time_s
    from auto_LiRPA import BoundedModule
    started=time.perf_counter()
    model=BoundedModule(driver.build_raw_net(cfg,experimental=False),
        torch.zeros(1,*cfg['input_shape'][1:],dtype=torch.float64),device='cuda:0',bound_opts=dict(cfg['bound_opts']))
    torch.cuda.synchronize();model_s=time.perf_counter()-started
    identity=dict(config_path=original['config'],config_sha256=original['config_sha256'],
        model_path=original['model'],model_sha256=original['model_sha256'],
        driver_path=original['driver_path'],driver_sha256=original['driver_sha256'],
        boxes_path=str(boxes_path),boxes_sha256=sha(boxes_path),engine=original['engine'],extensions=original['extensions'],
        frozen_base_sha256=BASE_SHA,common_script_sha256=COMMON_SHA,script_sha256=sha(__file__),
        endpoint_adapter_sha256=ENDPOINT_SHA,endpoint_check_script_sha256=ENDPOINT_CHECK_SHA,
        endpoint_check_path=str(args.endpoint_check),endpoint_check_result_sha256=sha(args.endpoint_check/'RESULT.json'),
        environment={k:v for k,v in os.environ.items() if k.startswith('FLOWSTAR_') or k in
            ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','TORCH_EXTENSIONS_DIR']})
    return c,b,adapter,edge,driver,model,cfg,identity,model_s


def long_gate(args,identity):
    if args.periods<=2:return
    assert args.qualification and args.replay_check,'long run requires same-order completed40 and cold boundary gate'
    meta=read(args.qualification/'INPUT.json');done=read(args.qualification/'RESULT.json');cold=read(args.replay_check/'RESULT.json')
    assert done['status']=='target_completed' and done['completed_step']==40 and done['input_sha256']==sha(args.qualification/'INPUT.json')
    assert meta['settings']==settings(args.working_order) and meta['mode']=='split_from_zero_strict_endpoint'
    assert meta['root_id']==1 and meta['leaf_count']==2 and meta['controller_policy']=='union_box'
    assert all(meta[k]==v for k,v in identity.items())
    assert cold['status']=='passed' and cold['mode']=='cold_boundary_replay' and cold['working_order']==args.working_order
    assert cold['script_sha256']==sha(__file__) and cold['source_input_sha256']==sha(args.qualification/'INPUT.json')
    assert cold['source_result_sha256']==sha(args.qualification/'RESULT.json')
    assert cold['all_endpoint_bytes_match'] and cold['all_controller_bytes_match'] and cold['all_step21_bytes_match']


def experiment(args,out,ctx):
    c,b,adapter,edge,driver,model,cfg,identity,model_s=ctx;torch=c.torch
    long_gate(args,identity);order=args.working_order;eng,sched=setup(c,order,edge)
    leaves,coverage=b.root_leaves(read(identity['boxes_path']),'split');assert len(leaves)==2
    state=c.se.initial_sparse_state(torch.tensor(leaves,dtype=torch.float64,device='cuda'),eng,sched)
    sr=c.make_symbolic_remainder(2,16,1000,'cuda');sr.reserve(1000)
    cap=torch.tensor([-.1,.1],dtype=torch.float64,device='cuda').expand(2,16,2).clone()
    config=c.config.Settings(**settings(order));code=c.compile_ode(cfg['dynamics_expressions'],[r['name'] for r in cfg['initial_set']],order=order-1)
    meta=dict(identity,mode='split_from_zero_strict_endpoint',root_id=1,leaf_count=2,original_root_count=1024,
        requested_periods=args.periods,target_step=args.periods*20,coverage=coverage,
        initial_affine_coverage=b.initial_coverage(state,leaves,eng),settings=settings(order),
        original_config_ode_order=2,working_total_degree=order,spatial_composition_order=order,
        point_code_order=order-1,validation_order=order+1,dynamics=cfg['dynamics_expressions'],variable_names=[r['name'] for r in cfg['initial_set']],
        controller_policy='union_box',controller='Official box/same-slope/native-f64, B1024 row1; zero/repeat padding checked each refresh',
        controller_setup_s=model_s,endpoint='strict with_roundoff error charged to pre_rem before every nonzero control boundary',
        observer='Local pre+R tube and endpoint hulls of all12 physical states, not composed-tmv',
        sr_queue=1000,sr_history_origin_step=0,end_to_end_strict_certificate=False,
        limitation='CROWN floating point and general RN control injection remain unqualified. Strict endpoint starts at t0 and does not repair old runs.',
        timing_scope='Instrumented original root1 B2 experiment, two CROWN calls per refresh; not full1024 benchmark speed',
        qualification=(str(args.qualification) if args.qualification else None),replay_check=(str(args.replay_check) if args.replay_check else None))
    write(out/'INPUT.json',meta);rows=[];controllers=[];held=None;completed=0;status='running';phase=PHASE;started=time.perf_counter()
    try:
        for period in range(args.periods):
            boundary_state=copy.deepcopy(state)
            if completed:boundary(c,b,adapter,boundary_state,sr,eng,completed,out)
            if completed==20:checkpoint(c,b,out/'after_strict_endpoint_20.pt',boundary_state,sr,eng,cap,20,held,meta,AFTER_ENDPOINT)
            held=refresh(c,b,boundary_state,sr,eng,driver,model,cfg,completed,out);controllers.append(held)
            state=boundary_state;phase=AFTER_CONTROLLER
            if completed==20:checkpoint(c,b,out/'after_controller_20.pt',state,sr,eng,cap,20,held,meta,phase)
            for substep in range(20):
                step=completed+1
                trial,ok,tsr,elapsed=c.attempt(state,sr,code,eng,sched,config,cap)
                row=dict(step=step,accepted=ok.cpu().tolist(),root_covered=bool(ok.all()),advance_s=elapsed,
                    parent_state_sr_unchanged=True,trial_sr_length=tsr.jlen,held_controller_step=held['completed_step'])
                if not bool(ok.all()):
                    status='numerical_rejection';rows.append(row)
                    checkpoint(c,b,out/'last_committed.pt',state,sr,eng,cap,completed,held,meta,
                        AFTER_CONTROLLER if substep==0 else PHASE)
                    c.save_state(out/f'failed_trial_{step}.pt',trial,tsr,eng,cap,step,'speculative_no_commit')
                    with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True);break
                state,sr=c.se.prune_state(trial,eng),tsr;completed=step;phase=PHASE
                row['sr_reset_if_full']=sr.reset_if_full()
                assert sr.qlen==sr.jlen==completed%1000 and row['sr_reset_if_full']==(completed==1000)
                row.update(observe(c,driver,state,eng),committed_sr_length=sr.jlen)
                rows.append(row)
                with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                if completed in [20,21] or completed==args.periods*20:
                    checkpoint(c,b,out/f'committed_{completed}.pt',state,sr,eng,cap,completed,held,meta)
                if completed%20==0:
                    progress=dict(completed_step=completed,root_covered=True,sr_length=sr.jlen,elapsed_s=time.perf_counter()-started)
                    write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
            if status=='numerical_rejection':break
        else:status='target_completed'
    except BaseException as exc:
        status='exception'
        checkpoint(c,b,out/'exception_committed.pt',state,sr,eng,cap,completed,held,meta,phase)
        write(out/'EXCEPTION.json',dict(error_type=type(exc).__name__,error=str(exc),completed_step=completed,phase=phase))
        raise
    finally:
        write(out/'RUN_PROGRESS.json',dict(status=status,completed_step=completed,attempted_steps=len(rows),
            input_sha256=sha(out/'INPUT.json'),last_attempt=rows[-1] if rows else None))
    return dict(status=status,mode='experiment',working_order=order,completed_step=completed,target_step=args.periods*20,
        root_covered_to_step=completed,controllers=controllers,total_advance_s=sum(r['advance_s'] for r in rows),
        run_body_s=time.perf_counter()-started,controller_setup_s=model_s,input_sha256=sha(out/'INPUT.json'),
        all_required_leaves=2,last_attempt=({k:v for k,v in rows[-1].items() if k not in ['tube','endpoint','union_tube','union_endpoint']} if rows else None),
        original_all1024_common_prefix_unchanged=677,stop_phase=PHASE if status=='target_completed' else 'see_failure_checkpoint')


def replay(args,out,ctx):
    c,b,adapter,edge,driver,model,cfg,identity,_=ctx;torch=c.torch;source=args.replay
    meta=read(source/'INPUT.json');done=read(source/'RESULT.json');order=args.working_order
    assert done['status']=='target_completed' and done['completed_step']>=40 and done['input_sha256']==sha(source/'INPUT.json')
    assert meta['mode']=='split_from_zero_strict_endpoint' and meta['settings']==settings(order)
    assert meta['working_total_degree']==meta['spatial_composition_order']==order and meta['point_code_order']==order-1 and meta['validation_order']==order+1
    assert all(meta[k]==v for k,v in identity.items())
    side20=read(source/'committed_20.json');side21=read(source/'committed_21.json')
    for step,side in [(20,side20),(21,side21)]:
        assert side['metadata']==meta and side['state_sha256']==sha(source/f'committed_{step}.pt')
        assert all(side[k]==v for k,v in schedule(step,step,PHASE).items())
    assert side20['held_controller']['completed_step']==0 and side21['held_controller']['completed_step']==20
    eng,sched=setup(c,order,edge);state,sr,cap=b.restore(source/'committed_20.pt',eng)
    endpoint=boundary(c,b,adapter,state,sr,eng,20,out);old_endpoint=read(source/'endpoint_at_20.json')
    actual=torch.load(out/'endpoint_at_20.pt',map_location='cpu',weights_only=True)
    saved=torch.load(source/'endpoint_at_20.pt',map_location='cpu',weights_only=True)
    echecks=dict(charged_error=b.byte_equal(actual['error'],saved['error']),working_order=actual['working_order']==saved['working_order']==order,
        before=endpoint['before']==old_endpoint['before'],after=endpoint['after']==old_endpoint['after'],
        adapter=old_endpoint['adapter_sha256']==ENDPOINT_SHA,source_sha256=sha(source/'endpoint_at_20.pt')==old_endpoint['error_sha256'])
    # Fingerprint support tuples are JSON lists in the saved sidecar.
    for key in ['before','after']:echecks[key]=json.loads(json.dumps(endpoint[key]))==old_endpoint[key]
    write(out/'ENDPOINT_CHECK.json',echecks);assert all(echecks.values()),'strict endpoint cold bytes differ'
    after_endpoint=read(source/'after_strict_endpoint_20.json')
    assert after_endpoint['metadata']==meta and after_endpoint['phase']==AFTER_ENDPOINT and after_endpoint['next_control_refresh_step']==20
    assert after_endpoint['state_sha256']==sha(source/'after_strict_endpoint_20.pt')
    assert after_endpoint['state_fingerprint']==json.loads(json.dumps(c.fingerprint(state,sr)))
    refresh(c,b,state,sr,eng,driver,model,cfg,20,out)
    after_controller=read(source/'after_controller_20.json')
    assert after_controller['metadata']==meta and after_controller['phase']==AFTER_CONTROLLER and after_controller['next_control_refresh_step']==40
    assert after_controller['state_sha256']==sha(source/'after_controller_20.pt')
    assert after_controller['state_fingerprint']==json.loads(json.dumps(c.fingerprint(state,sr)))
    assert sha(source/'controller_at_20.pt')==side21['held_controller']['sha256']
    actual=torch.load(out/'controller_at_20.pt',map_location='cpu',weights_only=True)
    saved=torch.load(source/'controller_at_20.pt',map_location='cpu',weights_only=True)
    checks={k:actual[k]==saved[k] for k in ['completed_step','next_step','root_id','phase','input_layout','transport','all_padding_outputs_byte_equal','output_byte_equal']}
    for key in ['input_root_hull','input_leaf_hulls']:checks[key]=b.byte_equal(actual[key],saved[key])
    for padding in ['zero_target','repeat_target']:
        for name,a,e in zip(['T','lower','upper'],actual[padding],saved[padding]):checks[padding+'_'+name]=b.byte_equal(a,e)
    write(out/'CONTROLLER_CHECK.json',checks);assert all(checks.values()),'cold NN certificate differs'
    code=c.compile_ode(meta['dynamics'],meta['variable_names'],order=order-1)
    trial,ok,tsr,elapsed=c.attempt(state,sr,code,eng,sched,c.config.Settings(**settings(order)),cap)
    expected=[r for r in (json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()) if r['step']==21]
    assert len(expected)==1 and bool(ok.all()) and ok.cpu().tolist()==expected[0]['accepted']
    trial=c.se.prune_state(trial,eng);assert tsr.reset_if_full() is False
    c.save_state(out/'replayed_21.pt',trial,tsr,eng,cap,21,PHASE)
    actual=torch.load(out/'replayed_21.pt',map_location='cpu',weights_only=True)
    saved=torch.load(source/'committed_21.pt',map_location='cpu',weights_only=True)
    sch={k:(b.byte_equal(actual[k],saved[k]) if isinstance(actual[k],torch.Tensor) else actual[k]==saved[k])
         if k in actual and k in saved else False for k in sorted(set(actual)|set(saved))}
    sch['accepted_mask']=ok.cpu().tolist()==expected[0]['accepted']
    measured=observe(c,driver,trial,eng)
    for key in ['tube','endpoint','union_tube','union_endpoint']:
        sch['observer_'+key]=b.byte_equal(torch.tensor(measured[key],dtype=torch.float64),torch.tensor(expected[0][key],dtype=torch.float64))
    write(out/'STATE_CHECK.json',sch);assert all(sch.values()),'cold21 state/support/cap/live SR bytes differ'
    return dict(status='passed',mode='cold_boundary_replay',working_order=order,source_input_sha256=sha(source/'INPUT.json'),
        source_result_sha256=sha(source/'RESULT.json'),all_endpoint_bytes_match=True,all_controller_bytes_match=True,
        all_step21_bytes_match=True,step21_comparisons=sch,accepted=ok.cpu().tolist(),advance_s=elapsed)


def check_local(args):
    """Exact archived initial boxes and actual CPU tables; no NN/advance/CUDA claim."""
    assert args.engine_root and args.boxes and args.identity
    identity=read(args.identity)['engine']
    for rel,digest in identity['python_sha256'].items():assert sha(args.engine_root/rel)==digest
    sys.path.insert(0,str(args.engine_root/'src'))
    import torch
    from flowstar_gpu import support as sp,sparse_exec as se,config
    from flowstar_gpu.monomials import build_tables
    from flowstar_gpu.polynomial import build_step_tables
    from flowstar_gpu.composition import build_schedule
    from flowstar_gpu.symbolic_remainder import make_symbolic_remainder
    base=Path(__file__).parent.parent/'quad_split_multistep_20260927/root_from_zero.py';assert sha(base)==BASE_SHA
    b=load('order_local_base',base);assert sha(args.boxes)==b.BOXES_SHA
    leaves,coverage=b.root_leaves(read(args.boxes),'split');assert coverage['covered']
    c=SimpleNamespace(build_tables=build_tables,sp=sp,build_step_tables=build_step_tables,build_schedule=build_schedule)
    rows=[]
    for order in [2,3]:
        eng,sched=setup(c,order,None,'cpu');state=se.initial_sparse_state(torch.tensor(leaves,dtype=torch.float64),eng,sched)
        assert config.Settings(**settings(order)).order==order
        receipts=b.initial_coverage(state,leaves,eng);assert len(receipts)==2
        rows.append(dict(order=order,leaf_count=2,affine_coverage_passed=True,initial_max_degree=int(eng.tables.exponents[list(state.pre_sup.ids)].sum(-1).max())))
    sr=make_symbolic_remainder(2,16,1000,'cpu');sr.reserve(1000);sr.qlen=sr.jlen=1000
    assert sr.reset_if_full() and sr.qlen==sr.jlen==0
    assert schedule(1000,sr.jlen,PHASE)['sr_history_origin_step']==1000
    assert schedule(20,20,PHASE)['next_control_refresh_step']==20
    assert schedule(20,20,AFTER_ENDPOINT)['next_control_refresh_step']==20
    assert schedule(20,20,AFTER_CONTROLLER)['next_control_refresh_step']==40
    print(json.dumps(dict(status='passed',rows=rows,capacity_reset_at1000_checked=True,
        scope='CPU archived root coverage, actual P2/P3 basis and checkpoint schedule only; no numerical/NN/CUDA qualification.')))


def main(args):
    settings(args.working_order)
    if args.check_local:return check_local(args)
    assert args.output and args.endpoint_check
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();result=dict(status='exception')
    try:
        ctx=initialize(args)
        result=replay(args,out,ctx) if args.replay else experiment(args,out,ctx)
    except BaseException as exc:
        result.update(status='exception',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result.update(script_sha256=sha(__file__),common_helper_sha256=COMMON_SHA,endpoint_adapter_sha256=ENDPOINT_SHA,
            process_body_s=time.perf_counter()-started)
        write(out/'RESULT.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--working-order',type=int,choices=[2,3,4],default=3)
    p.add_argument('--periods',type=int,choices=range(1,51),default=2);p.add_argument('--endpoint-check',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--replay',type=Path)
    p.add_argument('--qualification',type=Path);p.add_argument('--replay-check',type=Path)
    p.add_argument('--check-local',action='store_true');p.add_argument('--engine-root',type=Path);p.add_argument('--boxes',type=Path);p.add_argument('--identity',type=Path)
    main(p.parse_args())
