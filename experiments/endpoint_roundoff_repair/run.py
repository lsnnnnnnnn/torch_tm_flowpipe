"""Fresh CPU runs bound to the exact clean repaired scientific commit."""
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


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def interval_hex(values):
    return [[float(v.lo).hex(),float(v.hi).hex()] for v in values]


def checkpoint(output, current, state, total, h, config, provenance):
    return core.save_terminal_checkpoint(
        output, current=current, normal_state=state,
        scheduler={'accepted_steps':state.step_index,'time_exact':str(total),'next_h_hex':h.hex()},
        contract=config, provenance=provenance,
    )


def main():
    process_start=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plant',choices=['van_der_pol','brusselator'],required=True)
    parser.add_argument('--scientific-sha',required=True)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--adaptive',action='store_true')
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
    config,current,state=setup(args.plant)
    fixed_h=.02 if args.plant=='brusselator' else .01
    h=.1 if args.adaptive else fixed_h
    requested=F(10) if args.adaptive else F(args.steps,50 if args.plant=='brusselator' else 100)
    frozen_config={'plant':args.plant,'adaptive':args.adaptive,'config':config.as_dict(),
                   'fixed_h_hex':None if args.adaptive else fixed_h.hex(),
                   'requested_steps':None if args.adaptive else args.steps,
                   'requested_horizon_exact_nominal':str(requested),
                   'endpoint_ad_hoc_repair':False,'endpoint_substitution_roundoff_required':True,
                   'matched_contracts_sha256':MATCHED_SHA256}
    provenance={'scientific_sha':args.scientific_sha,'source_clean_at_start':True,
                'source_root':str(ROOT),'imported_package':core.__file__,
                'data_origin':'FRESH_ENDPOINT_REPAIRED','python':sys.executable,
                'python_version':sys.version,'torch_version':torch.__version__,
                'device':'cpu','dtype':'float64','threads':torch.get_num_threads(),
                'interop_threads':torch.get_num_interop_threads(),'affinity':sorted(os.sched_getaffinity(0)),
                'observer_mode':core.DENSE_OBSERVER_NONE,
                'source_files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT/'src/torch_tm_flowpipe').glob('*.py'))}}
    write_json(output/'execution_contract.json',frozen_config)
    write_json(output/'source.json',provenance)
    total=F(0)
    solve_seconds=export_seconds=0.
    rows=[]
    failure=None
    rejections=0
    refinement_totals={}
    checkpoints={1,2,20,99,100,101,120,999,1000}
    with gzip.open(output/'models.jsonl.gz','wt') as models, (output/'bounds.csv').open('x') as bounds, (output/'endpoint_audit.jsonl').open('x') as audit:
        writer=None
        for index in range(1,(10000 if args.adaptive else args.steps)+1):
            if args.adaptive and total>=requested:
                break
            attempted_h=min(h,float(requested-total)) if args.adaptive else fixed_h
            before_current,before_state=current,state
            started=time.perf_counter()
            try:
                segment=step(args.plant,current,state,index,h=attempted_h,adaptive=args.adaptive)
            except Exception as exc:
                solve_seconds+=time.perf_counter()-started
                failure={'attempted_step':index,'status':'exception','message':f'{type(exc).__name__}: {exc}',
                         'attempted_h_hex':attempted_h.hex(),'traceback':traceback.format_exc()}
                break
            solve_seconds+=time.perf_counter()-started
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
            queue=state.symbolic_queue
            assert queue is not None
            core.validate_accepted_boundary_sr_queue(queue,expected_boundary_index=index)
            errors=segment.endpoint_substitution_roundoff
            assert errors is not None and all(e.is_finite() for e in errors)
            internal_errors=segment.dense_endpoint_ledger.entries['endpoint_substitution_roundoff']
            assert all(bool(torch.all(torch.isfinite(e))) for e in internal_errors)
            row={'step':index,'t_start':float(previous),'t_end':float(total),'h':segment.h,'h_hex':segment.h.hex(),
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
                       'composition_branch':state.diagnostics.get('c3_composition_branch',state.diagnostics.get('accepted_boundary_sr_composition_branch')),
                       'step_rejections':segment.step_rejections,'validation_attempts':segment.validation_attempts,
                       'refinement_counters':{k:v for k,v in segment.backend_counters.items() if k.startswith('post_accept_')}}
            audit.write(json.dumps(audit_row,separators=(',',':'),allow_nan=False)+'\n')
            audit.flush()
            rows.append(row)
            if args.adaptive and segment.next_h is not None:
                h=segment.next_h
            if index in checkpoints or (not args.adaptive and index==args.steps):
                checkpoint(output/f'checkpoint_{index:04d}',current,state,total,h,frozen_config,provenance)
            export_seconds+=time.perf_counter()-export_start
            if index==1 or index%20==0:
                print(json.dumps({'accepted_steps':index,'time':float(total),'solve_seconds':solve_seconds,
                                  'export_seconds':export_seconds,'queue_size':len(queue.J),'status':'RUNNING'}),flush=True)
    if failure is not None:
        started=time.perf_counter()
        checkpoint(output/'checkpoint_before_failure',before_current,before_state,total,h,frozen_config,provenance)
        write_json(output/'failure.json',failure)
        export_seconds+=time.perf_counter()-started
    completed=failure is None and ((total>=requested) if args.adaptive else len(rows)==args.steps)
    summary={**provenance,'schema':'endpoint_roundoff_repaired_run/1','plant':args.plant,
             'adaptive':args.adaptive,'requested_horizon':float(requested),'requested_steps':None if args.adaptive else args.steps,
             'accepted_steps':len(rows),'rejected_attempts':rejections,'accepted_horizon':float(total),
             'accepted_horizon_exact':str(total),'fixed_step_hex':None if args.adaptive else fixed_h.hex(),
             'completed':completed,'failure':failure,'solve_seconds':solve_seconds,'export_seconds':export_seconds,
             'inside_process_seconds':time.perf_counter()-process_start,'refinement_totals':refinement_totals,
             'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             'final_queue_size':len(state.symbolic_queue.J) if state.symbolic_queue else None,
             'final_queue_reset_count':state.symbolic_queue.reset_count if state.symbolic_queue else None,
             'source_clean_at_end':not git('status','--porcelain'),'config':frozen_config}
    assert git('rev-parse','HEAD')==args.scientific_sha and summary['source_clean_at_end']
    write_json(output/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['plant','completed','accepted_steps','rejected_attempts','accepted_horizon_exact','solve_seconds','export_seconds','failure']},indent=2),flush=True)


if __name__=='__main__':
    main()
