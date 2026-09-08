"""Separate instrumented windows: exclusive wall attribution and cProfile self time."""
import argparse
from collections import defaultdict
import cProfile
import csv
import functools
import hashlib
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time

import torch
import torch_tm_flowpipe.batched_dense_tm as d
import torch_tm_flowpipe.accepted_boundary_sr as sr
import torch_tm_flowpipe.flowpipe as flow
import torch_tm_flowpipe.endpoint_substitution as endpoint
import torch_tm_flowpipe.prepared_remainder_replay as prepared
from torch_tm_flowpipe import load_terminal_checkpoint
from experiments.endpoint_roundoff_repair.frozen import setup,step


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plant',choices=['brusselator','van_der_pol'],required=True)
    p.add_argument('--mode',choices=['reference','optimized'],required=True)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--light',action='store_true',help='only time phase boundaries for Amdahl; no cProfile')
    args=p.parse_args()
    source_clean=not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    _,current,state=setup(args.plant)
    if args.checkpoint:
        restored=load_terminal_checkpoint(args.checkpoint)
        current,state=restored.current,restored.normal_state
    stack=[]; seconds=defaultdict(float); counts=defaultdict(int); originals=[]
    def wrap(owner,name,op,phase=None,fixed=False):
        original=getattr(owner,name)
        @functools.wraps(original)
        def timed(*a,**kw):
            parent=stack[-1] if stack else {'phase':'other','fixed':False}
            effective=phase or parent['phase']
            if op=='image' and effective!='replay':effective='initial'
            entry={'phase':effective,'fixed':fixed or parent['fixed'],'children':0.}
            key=(effective,op,entry['fixed'])
            started=time.perf_counter();stack.append(entry)
            try:return original(*a,**kw)
            finally:
                elapsed=time.perf_counter()-started;stack.pop()
                seconds[key]+=elapsed-entry['children'];counts[key]+=1
                if stack:stack[-1]['children']+=elapsed
        setattr(owner,name,timed);originals.append((owner,name,original))
    for name,phase in [('dense_polynomial_picard','candidate'),('_post_accept_refine_raw_remainder','replay')]:
        wrap(d,name,name,phase)
    wrap(d,'_dense_flowstar_raw_compat_image','image')
    wrap(flow,'_flowstar_normalized_insertion_transition','boundary','boundary')
    for name in ['prepare_accepted_boundary_sr','commit_accepted_boundary_sr']:
        wrap(sr,name,name,'history')
    wrap(endpoint,'enclose_constant_substitution','endpoint_correction','boundary')
    if not args.light:
        for name in ['mul_trunc','range_bound','apply_cutoff','integrate','add','sub','scale','component']:
            wrap(d.BatchedPolynomial,name,'polynomial_'+name,fixed=True)
        for name in ['_interval_add','_interval_sub','_interval_mul','_interval_scale','_inflate_tensor_interval']:
            wrap(d,name,'interval_arithmetic')
        for owner in [d.BatchedTaylorModel,d.DenseRemainderLedger]:
            wrap(owner,'__post_init__','object_checks_and_small_tensors')
        wrap(prepared._Polynomial,'_call','cache_dispatch')
        wrap(prepared._Tape,'__init__','prepare_graph')
        wrap(prepared.PreparedRemainderReplay,'__init__','prepare_snapshot',fixed=True)
        wrap(prepared.PreparedRemainderReplay,'validate_binding','binding_checks')
        wrap(prepared.PreparedRemainderReplay,'_seal','seal_storage',fixed=True)
    rows=[]
    try:
        for index in range(state.step_index+1,state.step_index+args.steps+1):
            seconds.clear();counts.clear()
            history_length=len(state.symbolic_queue.J) if state.symbolic_queue else 0
            started=time.perf_counter()
            with prepared.prepared_remainder_replay(args.mode=='optimized'):
                result=step(args.plant,current,state,index)
            elapsed=time.perf_counter()-started
            assert result.status=='validated',result.message
            accounted=sum(seconds.values())
            assert accounted<=elapsed+1e-8
            seconds[('other','unattributed',False)]=elapsed-accounted
            for (phase,op,fixed),value in seconds.items():
                if phase=='replay' and fixed:category='refinement_fixed_preparation'
                elif op=='object_checks_and_small_tensors':category='python_objects_and_small_tensor_checks'
                elif phase=='replay':category='refinement_dynamic_and_dispatch'
                elif op.startswith('polynomial_') or fixed:category='polynomial_multiply_truncate_range'
                elif phase=='candidate':category='candidate_construction_excluding_classified_polynomial_work'
                elif phase=='initial':category='initial_remainder_excluding_classified_polynomial_work'
                elif phase=='history':category='history'
                elif phase=='boundary':category='boundary_and_endpoint_correction'
                else:category='other_including_unclassified_python'
                rows.append(dict(plant=args.plant,mode=args.mode,step=index,history_length=history_length,
                    replays=result.backend_counters['post_accept_replay_calls'],phase=phase,operation=op,
                    dependency_fixed=fixed,category=category,exclusive_seconds=value,calls=counts.get((phase,op,fixed),0),
                    step_seconds=elapsed,instrumentation='light_phase_timer' if args.light else 'nested_function_timer'))
            current,state=result.reset_tm,result.flowstar_normal_state
            print(json.dumps({'step':index,'seconds':elapsed,'replays':result.backend_counters['post_accept_replay_calls']}),flush=True)
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)
    with (args.output/'exclusive.csv').open('w') as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if not args.light:
        profiler=cProfile.Profile()
        with prepared.prepared_remainder_replay(args.mode=='optimized'):
            profiler.enable();step(args.plant,current,state,index+1);profiler.disable()
        profiler.dump_stats(str(args.output/'self_times.pstats'))
        stats=pstats.Stats(profiler)
        with (args.output/'self_times.csv').open('w') as out:
            w=csv.writer(out);w.writerow(['file','line','function','primitive_calls','calls','self_seconds','inclusive_seconds_not_additive'])
            for key,v in sorted(stats.stats.items(),key=lambda x:x[1][2],reverse=True):w.writerow([*key,*v[:4]])
    (args.output/'source.json').write_text(json.dumps({'sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'source_clean_at_start':source_clean,'imported_solver':d.__file__,'python':sys.executable,
         'affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),'torch':torch.__version__,
         'runtime_hashes':{str(Path(m.__file__).name):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [d,prepared]},
         'mode':args.mode,'checkpoint':str(args.checkpoint),'profile_is_not_production_timing':True,
         'attribution':'Every wrapper subtracts all child wrapper wall time. Each exclusive atom appears once; residual closes to step wall time. cProfile is a separate next step, and only self times are additive.'},indent=2)+'\n')


if __name__=='__main__':main()
