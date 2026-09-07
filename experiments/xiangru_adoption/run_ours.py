"""One fresh frozen CPU-reference run; solve and export timers are separate."""
import argparse
import csv
from dataclasses import replace
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from common import from_existing_canonical, measure

def main():
    process_start = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--plant', choices=['brusselator','van_der_pol'], required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--adaptive', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = args.source.resolve()
    git = lambda *a: subprocess.check_output(['git','-C',str(source),*a],text=True).strip()
    assert git('rev-parse','HEAD') == '4939fb288c941a67f55cc191f4d75f8594692f47'
    assert not git('status','--porcelain')
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sys.path[:0] = [str(source/'src'), str(source)]
    import torch
    import torch_tm_flowpipe as core
    from experiments.profile_c4_reference_solver import _vdp_initial
    from experiments.run_brusselator_sr1000_parity import _step, _policy
    from torch_tm_flowpipe.brusselator_canonical_exchange import _append_tmv
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert Path(core.__file__).resolve().is_relative_to(source/'src')
    config = (core.FlowstarLikePolynomialPlantConfig.brusselator() if args.plant == 'brusselator'
              else core.FlowstarLikePolynomialPlantConfig.van_der_pol())
    setup_start = time.perf_counter()
    state = core.FlowstarNormalFlowpipeState.from_exact_decimal_box(config.initial_decimal_box, config.order)
    current = state.normalized_initial_tm(config.order)
    if args.plant == 'van_der_pol': ode, current, state = _vdp_initial()
    policy = core.DenseRangePolicy(**config.range_policy_mapping)
    h = 0.1 if args.adaptive else (0.02 if args.plant == 'brusselator' else 0.01)
    target = 10.0 if args.adaptive else args.steps*h
    solve_seconds = time.perf_counter()-setup_start
    export_seconds = 0.0
    total_time = Fraction()
    rows, failure = [], None
    with gzip.open(args.output/'models.jsonl.gz','wt') as objects:
      for index in range(1, (10000 if args.adaptive else args.steps)+1):
        if total_time >= Fraction(target): break
        started = time.perf_counter()
        if args.plant == 'brusselator':
            segment, _ = _step(current,state,index,_policy(),validation_mode='flowstar_raw_remainder_compat_refined',
                               lane_label='matched_adoption',observer_mode=core.DENSE_OBSERVER_NONE)
        else:
            segment = core.flowpipe_step_flowstar_style_adaptive(
                ode,current,h=min(h,float(Fraction(target)-total_time)),
                h_min=0.002 if args.adaptive else 0.01, h_max=0.1 if args.adaptive else 0.01,
                order=config.order,target_remainder_radius=config.target_remainder_radius,
                cutoff_threshold=config.cutoff,max_validation_attempts=2,
                validation_eps=config.validation_epsilon,validation_mode=config.post_accept_refinement_mode,
                reset_mode=config.accepted_boundary_sr_mode,step_policy_mode='flowstar_compat',
                flowstar_normal_state=state,flowstar_symbolic_queue_max_size=config.accepted_boundary_sr_capacity,
                right_map_center_mode=config.right_map_center_mode,right_map_range_mode=config.right_map_range_mode,
                tm_backend='dense',dense_device='cpu',dense_dtype=torch.float64,
                dense_range_policy=policy,dense_observer_mode=core.DENSE_OBSERVER_NONE)
        solve_seconds += time.perf_counter()-started
        if segment.status != 'validated' or segment.reset_tm is None:
            failure = {'attempted_step':index,'status':segment.status,'message':segment.message,'h':segment.h}
            break
        current, state = segment.reset_tm, segment.flowstar_normal_state
        export_start = time.perf_counter()
        t_before = total_time
        total_time += Fraction(segment.h)
        row = {'step':index,'t_start':float(t_before),'t_end':float(total_time),'h':segment.h,
               't_start_exact':str(t_before),'t_end_exact':str(total_time),'safety_check_passed':True,
               'queue_size':len(state.symbolic_queue.J),'queue_reset_count':state.symbolic_queue.reset_count}
        record = {'step':index,'plant':args.plant,'t_start_exact':str(t_before),'t_end_exact':str(total_time),'models':{}}
        for name, tm in [('endpoint',segment.endpoint_raw_tm),('tube',segment.tm)]:
            records = []
            var_names = ('ux','uy','tau') if tm.n_vars == 3 else ('ux','uy')
            _append_tmv(records,name,tm,variable_order=var_names)
            model = from_existing_canonical(dict(records),name)
            record['models'][name] = model
            published = [[float(v.lo),float(v.hi)] for v in tm.range_box()]
            common = measure(model)
            for metric, bounds in [('published',published),('common',common)]:
                for dim, (lo,hi) in zip(['x','y'],bounds):
                    row[f'{metric}_{name}_{dim}_lo'] = lo
                    row[f'{metric}_{name}_{dim}_hi'] = hi
        objects.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
        rows.append(row)
        export_seconds += time.perf_counter()-export_start
        if index % 25 == 0 or index == 1:
            print(json.dumps({'accepted_steps':index,'time':float(total_time),'solve_seconds':solve_seconds}),flush=True)
        if args.adaptive and segment.next_h is not None: h=segment.next_h
    with (args.output/'bounds.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ['step','t_start','t_end'])
        writer.writeheader();writer.writerows(rows)
    result={'schema':'xiangru_our_reference_run/1','plant':args.plant,'mode':'our_cpu_reference',
            'adaptive':args.adaptive,'requested_horizon':target,'accepted_steps':len(rows),
            'accepted_horizon':float(total_time),'accepted_horizon_exact':str(total_time),
            'failure':failure,'completed':float(total_time)>=target-1e-12,
            'solve_seconds':solve_seconds,'export_seconds':export_seconds,
            'inside_process_seconds':time.perf_counter()-process_start,
            'source_sha':git('rev-parse','HEAD'),'source_clean_at_start':True,
            'runner_sha':subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip(),
            'imported_package':core.__file__,'python':sys.executable,'python_version':sys.version,
            'torch_version':torch.__version__,'cuda_build':torch.version.cuda,'device':'cpu',
            'cpu_threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0)),
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'final_queue_size':len(state.symbolic_queue.J) if state.symbolic_queue else None,
            'final_queue_reset_count':state.symbolic_queue.reset_count if state.symbolic_queue else None,
            'config':config.to_dict() if hasattr(config,'to_dict') else repr(config)}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__': main()
