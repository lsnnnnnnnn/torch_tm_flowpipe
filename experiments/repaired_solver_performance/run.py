"""Matched clean CPU runs; exporters and scheduler reused from endpoint repair."""
import argparse
import csv
from fractions import Fraction as F
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import torch
import torch_tm_flowpipe as core
from torch_tm_flowpipe.brusselator_canonical_exchange import _append_tmv
from experiments.xiangru_adoption.common import from_existing_canonical, measure
from experiments.endpoint_roundoff_repair.frozen import ROOT, MATCHED, MATCHED_SHA256, setup, step
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def interval_hex(values):
    return [[float(v.lo).hex(),float(v.hi).hex()] for v in values]


def checkpoint(output, current, state, total, h, config, provenance, scheduler_time):
    return core.save_terminal_checkpoint(
        output, current=current, normal_state=state,
        scheduler={'accepted_steps':state.step_index,'time_exact':str(total),'next_h_hex':h.hex(),
                   'scheduler_time_hex':scheduler_time.hex()},
        contract=config, provenance=provenance,
    )


def main():
    process_start=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plant',choices=['van_der_pol','brusselator'],required=True)
    parser.add_argument('--scientific-sha',required=True)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--adaptive',action='store_true')
    parser.add_argument('--mode',choices=['reference','optimized'],required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.steps<1 or (args.adaptive and args.plant!='van_der_pol'):
        parser.error('positive steps and the frozen VDP adaptive lane are required')
    git=lambda *a: subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
    assert git('rev-parse','HEAD')==args.scientific_sha, 'wrong scientific SHA'
    assert not git('status','--porcelain'), 'scientific run must start clean'
    assert Path(core.__file__).resolve().is_relative_to(ROOT/'src'), 'wrong imported solver'
    assert hashlib.sha256(MATCHED.read_bytes()).hexdigest()==MATCHED_SHA256
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    setup_started=time.perf_counter()
    config,current,state=setup(args.plant)
    restored=None
    if args.checkpoint:
        restored=core.load_terminal_checkpoint(args.checkpoint,expected_dtype='float64',expected_order=config.order)
        assert restored.contract['plant']==args.plant
        current,state=restored.current,restored.normal_state
    start_index=state.step_index
    fixed_h=.02 if args.plant=='brusselator' else .01
    h=.1 if args.adaptive else fixed_h
    if restored and args.adaptive:
        h=float.fromhex(restored.scheduler['next_h_hex'])
    requested=F(10) if args.adaptive else F(args.steps,50 if args.plant=='brusselator' else 100)
    frozen_config={'plant':args.plant,'adaptive':args.adaptive,'config':config.as_dict(),
                   'fixed_h_hex':None if args.adaptive else fixed_h.hex(),
                   'requested_steps':None if args.adaptive else args.steps,
                   'start_boundary':start_index,'mode':args.mode,
                   'requested_horizon_exact_nominal':str(requested),
                   'endpoint_ad_hoc_repair':False,'endpoint_substitution_roundoff_required':True,
                   'matched_contracts_sha256':MATCHED_SHA256}
    provenance={'scientific_sha':args.scientific_sha,'source_clean_at_start':True,
                'source_root':str(ROOT),'imported_package':core.__file__,
                'data_origin':'FRESH_REPAIRED_REFERENCE' if args.mode=='reference' else 'FRESH_OPTIMIZED',
                'mode':args.mode,'python':sys.executable,
                'python_version':sys.version,'torch_version':torch.__version__,
                'device':'cpu','dtype':'float64','threads':torch.get_num_threads(),
                'interop_threads':torch.get_num_interop_threads(),'affinity':sorted(os.sched_getaffinity(0)),
                'observer_mode':core.DENSE_OBSERVER_NONE,
                'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'loadavg_start':Path('/proc/loadavg').read_text().split()[:3],
                'cpu_stat_start':next(line for line in Path('/proc/stat').read_text().splitlines()
                                      if line.startswith(f'cpu{min(os.sched_getaffinity(0))} ')),
                'source_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT/'src/torch_tm_flowpipe').glob('*.py'))}}
    write_json(output/'execution_contract.json',frozen_config)
    write_json(output/'source.json',provenance)
    setup_stopped=time.perf_counter()
    setup_seconds=setup_stopped-setup_started
    total=F(restored.scheduler['time_exact']) if restored else F(0)
    starting_total=total
    # Preserve the original native scheduler's binary64 clock and terminal
    # step rule. The independent rational clock below reports the actual sum.
    scheduler_time=float.fromhex(restored.scheduler['scheduler_time_hex']) if restored and args.adaptive else float(total)
    solve_seconds=export_seconds=0.
    rows=[]
    failure=None
    rejections=0
    refinement_totals={}
    timing_events=[]
    checkpoints={1,2,20,90,99,100,101,120,980,999,1000}
    with gzip.open(output/'models.jsonl.gz','wt') as models, (output/'bounds.csv').open('x') as bounds, (output/'endpoint_audit.jsonl').open('x') as audit:
        writer=None
        for index in range(start_index+1,(10000 if args.adaptive else start_index+args.steps)+1):
            if args.adaptive and scheduler_time>=float(requested)-1e-12:
                break
            before_current,before_state=current,state
            attempted_h=fixed_h
            if args.adaptive:
                remaining=float(requested)-scheduler_time
                if remaining < .002-1e-15:
                    failure={'attempted_step':index,'status':'failed',
                             'message':'remaining horizon is below authoritative h_min; no clipped sub-minimum endpoint was published',
                             'remaining_h_hex':remaining.hex()}
                    break
                attempted_h=min(h,.1,remaining)
                if 0. < remaining-attempted_h < .002:
                    attempted_h=remaining
            started=time.perf_counter()
            try:
                with prepared_remainder_replay(args.mode=='optimized'):
                    segment=step(args.plant,current,state,index,h=attempted_h,adaptive=args.adaptive)
            except Exception as exc:
                solve_seconds+=time.perf_counter()-started
                failure={'attempted_step':index,'status':'exception','message':f'{type(exc).__name__}: {exc}',
                         'attempted_h_hex':attempted_h.hex(),'traceback':traceback.format_exc()}
                break
            stopped=time.perf_counter()
            solve_seconds+=stopped-started
            timing_events.append({'step':index,'numerical_start':started,'numerical_stop':stopped})
            rejections+=segment.step_rejections
            for name,value in (segment.backend_counters or {}).items():
                if name.startswith('post_accept_') or name=='polynomial_picard_iterations':
                    refinement_totals[name]=refinement_totals.get(name,0)+value
            if segment.status!='validated' or segment.reset_tm is None:
                failure={'attempted_step':index,'status':segment.status,'message':segment.message,
                         'attempted_h_hex':attempted_h.hex(),'returned_h_hex':segment.h.hex(),
                         'backend_counters':segment.backend_counters,'backend_trace':segment.backend_trace}
                break
            export_start=time.perf_counter()
            if not args.adaptive:
                assert segment.h.hex()==fixed_h.hex(), 'fixed binary64 h changed'
            current,state=segment.reset_tm,segment.flowstar_normal_state
            previous=total
            total+=F(segment.h)
            scheduler_time=scheduler_time+segment.h if args.adaptive else index*fixed_h
            queue=state.symbolic_queue
            assert queue is not None
            core.validate_accepted_boundary_sr_queue(queue,expected_boundary_index=index)
            errors=segment.endpoint_substitution_roundoff
            assert errors is not None and all(e.is_finite() for e in errors)
            internal_errors=segment.dense_endpoint_ledger.entries['endpoint_substitution_roundoff']
            assert all(bool(torch.all(torch.isfinite(e))) for e in internal_errors)
            row={'step':index,'t_start':float(previous),'t_end':float(total),'h':segment.h,'h_hex':segment.h.hex(),
                 'scheduler_time_hex':scheduler_time.hex(),
                 't_start_exact':str(previous),'t_end_exact':str(total),'safety_check_passed':True,
                 'queue_size':len(queue.J),'queue_reset_count':queue.reset_count}
            record={'step':index,'plant':args.plant,'t_start_exact':str(previous),'t_end_exact':str(total),'models':{}}
            for name,tm in [('endpoint',segment.endpoint_raw_tm),('tube',segment.tm)]:
                canonical=[]
                _append_tmv(canonical,name,tm,variable_order=('ux','uy','tau') if tm.n_vars==3 else ('ux','uy'))
                model=from_existing_canonical(dict(canonical),name)
                record['models'][name]=model
                published=[[float(v.lo),float(v.hi)] for v in tm.range_box()]
                common=measure(model)
                for metric,result in [('published',published),('common',common)]:
                    for dim,(lo,hi) in zip(['x','y'],result):
                        row[f'{metric}_{name}_{dim}_lo']=lo
                        row[f'{metric}_{name}_{dim}_hi']=hi
            if writer is None:
                writer=csv.DictWriter(bounds,fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            bounds.flush()
            models.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
            audit_row={'step':index,'h_hex':segment.h.hex(),'t_end_exact':str(total),
                       'substitution_error_hex':interval_hex(errors),
                       'internal_substitution_error_hex':[[float(internal_errors[0][0,i]).hex(),float(internal_errors[1][0,i]).hex()] for i in range(2)],
                       'endpoint_remainder_hex':interval_hex([m.remainder for m in segment.endpoint_raw_tm]),
                       'queue_sha256':core.accepted_boundary_sr_queue_sha256(queue),
                       'queue_size':len(queue.J),'queue_reset_count':queue.reset_count,
                       'new_owner_hex':interval_hex(queue.J[-1]) if queue.J else None,
                       'owner_boundary':queue.owner_boundary_indices[-1] if queue.J else None,
                       'center_hex':[x.hex() for x in state.center], 'scales_hex':[x.hex() for x in state.scales],
                       'state_hashes':{name:core.tmvector_hashes(model) for name,model in
                                      [('current',current),('normal_pre',state.tmv_pre),('normal_right',state.tmv_right)]},
                       'dense_endpoint_ledger_hex':{name:[[float(x).hex() for x in lo.reshape(-1)],
                                                         [float(x).hex() for x in hi.reshape(-1)]]
                                                    for name,(lo,hi) in segment.dense_endpoint_ledger.entries.items()},
                       'composition_branch':state.diagnostics.get('c3_composition_branch',state.diagnostics.get('accepted_boundary_sr_composition_branch')),
                       'step_rejections':segment.step_rejections,'validation_attempts':segment.validation_attempts,
                       'refinement_counters':{k:v for k,v in segment.backend_counters.items() if k.startswith('post_accept_')}}
            audit.write(json.dumps(audit_row,separators=(',',':'),allow_nan=False)+'\n')
            audit.flush()
            rows.append(row)
            if args.adaptive and segment.next_h is not None:
                h=segment.next_h
            if index in checkpoints or (not args.adaptive and index==start_index+args.steps):
                checkpoint(output/f'checkpoint_{index:04d}',current,state,total,h,frozen_config,provenance,scheduler_time)
            export_stopped=time.perf_counter()
            export_seconds+=export_stopped-export_start
            timing_events[-1].update(export_start=export_start,export_stop=export_stopped)
            if index==1 or index%20==0:
                print(json.dumps({'accepted_steps':index,'time':float(total),'solve_seconds':solve_seconds,
                                  'export_seconds':export_seconds,'queue_size':len(queue.J),'status':'RUNNING'}),flush=True)
    if failure is not None:
        started=time.perf_counter()
        checkpoint(output/'checkpoint_before_failure',before_current,before_state,total,h,frozen_config,provenance,scheduler_time)
        write_json(output/'failure.json',failure)
        export_seconds+=time.perf_counter()-started
    completed=failure is None and ((scheduler_time>=float(requested)-1e-12) if args.adaptive else len(rows)==args.steps)
    summary={**provenance,'schema':'repaired_solver_performance_run/1','plant':args.plant,
             'adaptive':args.adaptive,'requested_horizon':float(requested),'requested_steps':None if args.adaptive else args.steps,
             'accepted_steps':len(rows),'rejected_attempts':rejections,'accepted_horizon':float(total),
             'start_boundary':start_index,'starting_time_exact':str(starting_total),
             'accepted_horizon_exact':str(total),'fixed_step_hex':None if args.adaptive else fixed_h.hex(),
             'scheduler_time_hex':scheduler_time.hex(),
             'completed':completed,'failure':failure,'solve_seconds':solve_seconds,'export_seconds':export_seconds,
             'setup_seconds':setup_seconds,'plan_construction_included_in_solve':True,
             'loadavg_end':Path('/proc/loadavg').read_text().split()[:3],
             'cpu_stat_end':next(line for line in Path('/proc/stat').read_text().splitlines()
                                if line.startswith(f'cpu{min(os.sched_getaffinity(0))} ')),
             'inside_process_seconds':time.perf_counter()-process_start,'refinement_totals':refinement_totals,
             'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             'final_queue_size':len(state.symbolic_queue.J) if state.symbolic_queue else None,
             'final_queue_reset_count':state.symbolic_queue.reset_count if state.symbolic_queue else None,
             'source_clean_at_end':not git('status','--porcelain'),'config':frozen_config,
             'config_digest':hashlib.sha256(json.dumps(frozen_config,sort_keys=True).encode()).hexdigest()}
    assert git('rev-parse','HEAD')==args.scientific_sha and summary['source_clean_at_end']
    write_json(output/'timing_events.json',timing_events)
    write_json(output/'initialization_event.json',{'start':setup_started,'stop':setup_stopped,
               'scope':'configuration, initial/checkpoint state, runtime provenance and initial metadata writes'})
    write_json(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['plant','completed','accepted_steps','rejected_attempts','accepted_horizon_exact','solve_seconds','export_seconds','failure']},indent=2),flush=True)


if __name__=='__main__':
    main()
