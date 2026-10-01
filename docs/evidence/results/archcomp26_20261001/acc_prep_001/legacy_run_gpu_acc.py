"""Author NNCS diagnostics with a shared controller driver; not a strict NNCS certificate."""
import argparse
import copy
import numpy as np
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

N = Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z')
X = Path('/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['ours', 'huan', 'xiangru'], required=True)
    p.add_argument('--mode', choices=['strict', 'parity'], required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--record', action='store_true')
    p.add_argument('--leaf-policy',choices=['truncated','defer_linear_leaves'],default='defer_linear_leaves')
    p.add_argument('--boxes',type=Path)
    p.add_argument('--lane-start',type=int,default=0)
    p.add_argument('--lane-count',type=int)
    p.add_argument('--widths',action='store_true')
    p.add_argument('--stop-any-period',action='store_true')
    p.add_argument('--release-unused-cache',action='store_true')
    p.add_argument('--official-controller',action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output/'result.json').exists(), 'Never overwrite a completed attempt'
    harness = load('frozen_four_way', N/'runs/four_way_run_engine_v1.py')
    q=json.loads((N/'runs/linear_leaf_validation_v2/CHECK.json').read_text());assert q['status']=='passed'
    info=json.loads((N/'runs/linear_leaf_validation_v2/INTEGRATION.json').read_text())
    harness.ENGINES['ours']=(Path(info['root']),info['head'])
    candidate, identity = harness.prepare(a.backend, a.mode)
    if a.backend == 'ours':
        # The polynomial-only optimization is inapplicable to sin dynamics.
        os.environ['FLOWSTAR_VALIDATION_POLICY'] = a.leaf_policy
    engine = harness.ENGINES[a.backend][0]
    import torch
    import yaml
    from flowstar_gpu import cuda_kernels, tape_kernels
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    driver_path = X/'src/flowstar_gpu/integrations/crown_reach.py'
    d = load('shared_author_driver', driver_path)
    if a.backend == 'ours':
        assert d.sparse_exec_module.VALIDATION_POLICY==a.leaf_policy
    metrics_compatibility = None
    if not hasattr(d.sparse_exec_module, 'COMPOSE_PARENT_ASSEMBLY'):
        # Read only by the author's final metrics writer; not a solver option in these engines.
        d.sparse_exec_module.COMPOSE_PARENT_ASSEMBLY = 'not_declared_by_engine'
        metrics_compatibility = 'Missing final-report field labeled not_declared_by_engine'
    cfg = yaml.safe_load(a.config.read_text())
    adapter_state=None
    if cfg.get('adapter')=='acc_exact_feature_tm':
        adapter=load('acc_feature_adapter',N/'repo_sr_prepare_entry/experiments/flowstar_acceleration/nncs_acc_adapter.py')
        adapter_state=adapter.install(d,cfg)
    if a.official_controller:
        d.SR_QUEUE=int(cfg['sr_queue'])
        def official_build(config,device,relax='same-slope',input_layout='native'):
            assert input_layout=='native' and relax=='same-slope'
            from auto_LiRPA import BoundedModule
            raw=d.build_raw_net(config,experimental=False)
            return BoundedModule(raw,torch.zeros(1,*config['input_shape'][1:],dtype=torch.float64),device=device,bound_opts=dict(config['bound_opts']))
        d.build_crown=official_build
    original_cells = d.make_cells(cfg)
    all_boxes = json.loads(a.boxes.read_text()) if a.boxes else original_cells.tolist()
    stop_lane = len(all_boxes) if a.lane_count is None else a.lane_start + a.lane_count
    assert 0 <= a.lane_start < stop_lane <= len(all_boxes)
    cells = torch.tensor(all_boxes[a.lane_start:stop_lane],dtype=torch.float64)
    d.make_cells = lambda config: cells.clone()
    lane_ids = list(range(a.lane_start,stop_lane))
    physical = int(cfg.get('physical_dims',cfg['num_nn_input']))
    substeps = round(cfg['step_size']/cfg['ode_step_size'])
    assert substeps > 0
    class PeriodFailure(Exception): pass

    model = (a.config.parent.parent/cfg['model_dir']).resolve()
    result = dict(backend=a.backend, mode=a.mode, engine=identity, leaf_policy=a.leaf_policy,
        driver_path=str(driver_path), driver_sha256=harness.SHA(driver_path),
        runner_sha256=harness.SHA(__file__), config=str(a.config),
        config_sha256=harness.SHA(a.config), model=str(model), model_sha256=harness.SHA(model),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), affinity=sorted(os.sched_getaffinity(0)),
        torch_version=torch.__version__, timing_scope='instrumented diagnostic; not five-repeat formal timing',
        controller=('official server layout/options; shared box/same-slope/rpc-float32' if a.official_controller else 'shared Xiangru box/same-slope/flat/rpc-float32')+'; RN injection and controller unqualified',
        end_to_end_strict_certificate=False, metrics_compatibility=metrics_compatibility, attempted_steps=0, accepted_lane_steps=0,
        expected_steps=round(cfg['step_size']/cfg['ode_step_size'])*cfg['steps'],
        B=int(d.make_cells(cfg).shape[0]), steps=[], extensions={}, status='running')
    result.update(total_lanes=len(all_boxes),lane_ids=lane_ids,boxes_sha256=harness.SHA(a.boxes) if a.boxes else None,
        stop_any_period=a.stop_any_period,observer='local-domain tmvPre interval hull; endpoint direct time interval; not composed-tmv',
        timing_components={'advance_s':0.,'observer_s':0.,'controller_s':0.,'controller_setup_s':0.,'cache_release_s':0.},cache_releases=[],release_unused_cache=a.release_unused_cache)
    loaded=[]
    assert all([cuda_kernels.available(), tape_kernels.available(), tape_kernels.valid_available()])
    loaded.extend([cuda_kernels._ext,tape_kernels._ext,tape_kernels._vext])
    if a.backend == 'ours':
        assert a.mode == 'strict'
        from flowstar_gpu import sr_kernels, sr_sum_kernels, injective_index
        for m in [sr_kernels,sr_sum_kernels,injective_index]:
            assert m.available()
            loaded.append(m._ext)
        result['private_output']=candidate._select_private_output('on','cuda',cuda_kernels)
        loaded.append(cuda_kernels._ext)
        edge,result['horner_edge']=candidate._select_horner_edge('on','cuda','horner')
        loaded.append(edge)
        sparse_engine=d.SparseEngine
        def configured_engine(*args,**kwargs):
            eng=sparse_engine(*args,**kwargs)
            eng.horner_edge_kernel=edge.horner_edge
            return eng
        d.SparseEngine=configured_engine
        # Retain our repaired endpoint arithmetic; don't import the author's RN endpoint into strict plant.
        own_path=engine/'integrations/crown_reach/gpu_driver.py'
        own=load('repaired_endpoint_driver',own_path)
        d.end_of_time_s=own.end_of_time_s
        result['endpoint_driver']={'path':str(own_path),'sha256':harness.SHA(own_path)}
    for m in loaded:
        result['extensions'][m.__name__]={'path':m.__file__,'sha256':harness.SHA(m.__file__)}
    result['environment']={k:v for k,v in os.environ.items() if k.startswith('FLOWSTAR_') or k in ['TORCH_EXTENSIONS_DIR','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS']}
    trace=gzip.open(a.output/'factored.jsonl.gz','wt',compresslevel=1) if a.record else None
    ranges=(a.output/'ranges.bin').open('wb') if a.widths else None
    witnesses=gzip.open(a.output/'witnesses.jsonl.gz','wt',compresslevel=1) if a.widths else None
    dtype=np.dtype([('lane','<u8'),('step','<u8'),('h','<f8'),('bounds','<f8',(physical,4))])
    assert dtype.itemsize==24+32*physical
    for name,field in [('build_crown','controller_setup_s'),('crown_bounds','controller_s')]:
        fn=getattr(d,name)
        def timed(*args,_fn=fn,_field=field,**kw):
            torch.cuda.synchronize();t=time.perf_counter()
            try:return _fn(*args,**kw)
            finally:
                torch.cuda.synchronize();result['timing_components'][_field]+=time.perf_counter()-t
        setattr(d,name,timed)
    advance=d.advance_sparse
    def observed(*args,**kwargs):
        torch.cuda.synchronize();t=time.perf_counter()
        state,ok=advance(*args,**kwargs)
        torch.cuda.synchronize();result['timing_components']['advance_s']+=time.perf_counter()-t
        accepted=ok.detach().cpu().tolist()
        result['attempted_steps']+=1
        result['accepted_lane_steps']+=sum(accepted)
        result['steps'].append({'step':result['attempted_steps'],'accepted':accepted,'status':state.status.detach().cpu().tolist()})
        if ranges:
            t=time.perf_counter();eng=args[2];h=float(args[4].step)
            versions=[(tensor,tensor._version) for tensor in (state.pre,state.pre_rem,state.tmv,state.tmv_rem,state.status)]
            tube=d.hull_ranges_s(state,eng,physical)
            piece=torch.full((len(lane_ids),2),h,dtype=torch.float64,device=state.pre.device)
            endpoint=d.rows_range_over_time_sparse(state,eng,piece,physical)
            bounds=torch.cat((tube,endpoint),dim=-1).detach().cpu().numpy()
            assert all(tensor._version==v for tensor,v in versions), 'observer mutated state'
            valid=np.asarray(accepted,dtype=bool)
            assert np.isfinite(bounds[valid]).all()
            assert (bounds[valid,:,0]<=bounds[valid,:,1]).all() and (bounds[valid,:,2]<=bounds[valid,:,3]).all()
            rows=np.empty(len(lane_ids),dtype=dtype)
            rows['lane']=lane_ids;rows['step']=result['attempted_steps'];rows['h']=h;rows['bounds']=bounds
            ranges.write(rows.tobytes());ranges.flush()
            if result['attempted_steps']==1 or result['attempted_steps']%substeps==0 or not all(accepted):
                sample=copy.copy(state)
                for field in ['pre','pre_rem','tmv','tmv_rem']:setattr(sample,field,getattr(state,field)[:1])
                witness=candidate.save_factored(sample,eng.tables,step=result['attempted_steps'],h=h,lane_ids=lane_ids[:1],accepted=accepted[:1])
                witness['observed_bounds']=bounds[:1].tolist()
                witnesses.write(json.dumps(witness,allow_nan=False,separators=(',',':'))+'\n');witnesses.flush()
            result['timing_components']['observer_s']+=time.perf_counter()-t
        if trace:
            record=candidate.save_factored(state,args[2].tables,step=result['attempted_steps'],h=float(args[4].step),lane_ids=list(range(len(accepted))),accepted=accepted)
            trace.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
        allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
        if a.release_unused_cache and reserved-allocated > 2*2**30:
            t=time.perf_counter();torch.cuda.empty_cache();torch.cuda.synchronize()
            assert torch.cuda.memory_allocated()==allocated, 'cache release changed live allocations'
            result['timing_components']['cache_release_s']+=time.perf_counter()-t
            result['cache_releases'].append(dict(step=result['attempted_steps'],allocated=allocated,reserved_before=reserved,reserved_after=torch.cuda.memory_reserved()))
        if result['attempted_steps'] % 20 == 0 or not all(accepted):
            (a.output/'progress.json').write_text(json.dumps(dict(attempted_steps=result['attempted_steps'], accepted_lane_steps=result['accepted_lane_steps'], elapsed_s=time.perf_counter()-start,allocated_bytes=allocated,reserved_bytes=torch.cuda.memory_reserved()))+'\n')
            print('PROGRESS',result['attempted_steps'],result['accepted_lane_steps'],'allocated',allocated,'reserved',torch.cuda.memory_reserved(),flush=True)
        if a.stop_any_period and result['attempted_steps']%substeps==0 and not all(accepted):
            raise PeriodFailure('Native ARCH-COMP stops after any failed lane at the period boundary')
        return state,ok
    d.advance_sparse=observed
    argv=[str(driver_path),str(a.config),'--device','cuda:0','--engine','sparse',
        '--crown-domain','box','--crown-relax','same-slope','--crown-transport','rpc-float32',
        '--crown-input-layout',('native' if a.official_controller else 'flat'),'--print-final-hull','--metrics-json',str(a.output/'metrics.json')]
    if a.mode=='strict':argv.append('--strict')
    result['driver_argv']=argv
    sys.argv=argv
    start=time.perf_counter()
    try:
        result['returncode']=d.main()
        result['complete']=result['attempted_steps']==result['expected_steps'] and result['accepted_lane_steps']==result['expected_steps']*result['B']
        result['status']='completed' if result['complete'] else 'incomplete'
    except PeriodFailure as exc:
        result.update(status='incomplete',complete=False,error=str(exc),stop_reason='native_any_failure_period_boundary')
    except BaseException as exc:
        result.update(status='error',error=repr(exc))
        traceback.print_exc()
    finally:
        if trace:trace.close()
        if ranges:ranges.close();witnesses.close()
        if adapter_state is not None:
            result['acc_adapter']={k:v for k,v in adapter_state.items() if k!='cache'}
            result['acc_adapter_sha256']=harness.SHA(adapter.__file__)
        result['driver_wall_s']=time.perf_counter()-start
        result['source_unchanged']=candidate.source_identity(engine)==identity
        result['binaries_unchanged']=all(harness.SHA(v['path'])==v['sha256'] for v in result['extensions'].values())
        assert result['source_unchanged'] and result['binaries_unchanged']
        (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['backend','mode','status','attempted_steps','accepted_lane_steps','driver_wall_s']}),flush=True)
    return 0 if result['status']=='completed' else 1

if __name__=='__main__':
    raise SystemExit(main())
