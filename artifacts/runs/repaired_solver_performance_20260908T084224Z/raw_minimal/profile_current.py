"""Five matched diagnostic windows; no instrumented time is a speed denominator."""
import cProfile
import csv
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import pstats
import subprocess
import time

import torch
import torch_tm_flowpipe as core
import torch_tm_flowpipe.batched_dense_tm as dense
import torch_tm_flowpipe.flowpipe as flow
from experiments.endpoint_roundoff_repair.frozen import setup,step
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay

prepared=importlib.import_module('torch_tm_flowpipe.prepared_remainder_replay')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'matched_profiles'
SOURCE=ROOT/'candidate_source'
OUT.mkdir(exist_ok=False)
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
assert Path(core.__file__).resolve().is_relative_to(SOURCE/'src')
torch.set_num_threads(1);torch.set_num_interop_threads(1)


def write_csv(path,rows):
    with path.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class Scopes:
    def __init__(self):
        self.stack=[];self.rows={};self.active=False;self.originals=[]

    def wrap(self,obj,name,category,stage=None):
        original=getattr(obj,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            if not self.active:return original(*args,**kwargs)
            inherited=self.stack[-1][2] if self.stack else 'other_solve'
            current=stage or inherited
            group=category(current) if callable(category) else category
            frame=[time.perf_counter(),0.,current]
            self.stack.append(frame)
            try:return original(*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-frame[0]
                self.stack.pop()
                if self.stack:self.stack[-1][1]+=elapsed
                row=self.rows.setdefault((current,group),[0.,0.,0])
                row[0]+=elapsed-frame[1];row[1]+=elapsed;row[2]+=1
        setattr(obj,name,measured);self.originals.append((obj,name,original))

    def install(self):
        self.wrap(dense,'dense_polynomial_picard','candidate_polynomial','candidate')
        self.wrap(dense,'_post_accept_refine_raw_remainder','refinement_dynamic','post_accept')
        # The initial image has no surrounding refinement scope. Other images
        # inherit post_accept, including the prepared evaluator.
        original=dense._dense_flowstar_raw_compat_image
        self.wrap(dense,'_dense_flowstar_raw_compat_image',lambda stage:'refinement_dynamic' if stage=='post_accept' else 'initial_remainder')
        for name in ['mul_trunc','range_bound','integrate','apply_cutoff','add','sub','scale']:
            self.wrap(dense.BatchedPolynomial,name,lambda stage:'refinement_fixed_polynomial' if stage=='post_accept' else 'polynomial_multiply_truncate_range')
        self.wrap(dense,'_prepare_vdp_refinement_static_cache','refinement_fixed_polynomial')
        self.wrap(dense,'_joint_factorized_vdp_residual_closure','refinement_dynamic')
        self.wrap(prepared.PreparedRemainderReplay,'__init__','plan_snapshot_and_binding')
        self.wrap(prepared.PreparedRemainderReplay,'check','plan_binding_checks')
        self.wrap(prepared._Tape,'operation','fixed_plan_operation_dispatch')
        self.wrap(dense.BatchedTaylorModel,'endpoint','boundary_endpoint','boundary')
        self.wrap(flow,'_flowstar_normalized_insertion_transition','boundary_endpoint','boundary')
        self.wrap(flow,'_vdp_frozen_c3_accepted_boundary_prepare','cross_step_history','history')
        for name in ['prepare_accepted_boundary_sr','commit_accepted_boundary_sr']:
            if hasattr(flow,name):self.wrap(flow,name,'cross_step_history','history')
        for cls,name in [(core.Interval,'__init__'),(core.Polynomial,'__init__'),(core.TaylorModel,'__init__'),
                         (dense.BatchedPolynomial,'__post_init__'),(dense.BatchedTaylorModel,'__post_init__'),
                         (dense.DenseRemainderLedger,'__post_init__')]:
            self.wrap(cls,name,'python_objects_and_small_tensor_validation')

    def uninstall(self):
        for obj,name,original in reversed(self.originals):setattr(obj,name,original)


all_steps=[];all_phases=[]
for plant,start in [('brusselator',1),('brusselator',101),('brusselator',981),('van_der_pol',1),('van_der_pol',91)]:
    window=f'{plant}_{start:04d}_{start+19:04d}'
    for execution in ['reference','prepared_remainder_replay']:
        for instrumentation in ['production_diagnostic','scoped','cprofile']:
            _,current,state=setup(plant)
            if start>1:
                restored=core.load_terminal_checkpoint(ROOT/'fresh_reference'/plant/f'checkpoint_{start-1:04d}',expected_dtype='float64')
                current,state=restored.current,restored.normal_state
            profiler=cProfile.Profile();scopes=Scopes()
            if instrumentation=='scoped':scopes.install()
            elapsed_total=0.
            with prepared_remainder_replay(execution=='prepared_remainder_replay'):
                for index in range(start,start+20):
                    scopes.rows={};scopes.active=instrumentation=='scoped'
                    history=len(state.symbolic_queue.J) if state.symbolic_queue else 0
                    if instrumentation=='cprofile':profiler.enable()
                    before=time.perf_counter()
                    segment=step(plant,current,state,index)
                    elapsed=time.perf_counter()-before
                    if instrumentation=='cprofile':profiler.disable()
                    scopes.active=False
                    assert segment.status=='validated',segment.message
                    elapsed_total+=elapsed
                    current,state=segment.reset_tm,segment.flowstar_normal_state
                    counters=segment.backend_counters
                    row=dict(window=window,plant=plant,execution_mode=execution,instrumentation=instrumentation,
                             step=index,history_before=history,replay_calls=counters['post_accept_replay_calls'],solve_seconds=elapsed)
                    row.update({k:counters.get(k,0) for k in ['prepared_plan_count','prepared_operation_count','prepared_operation_hits','prepared_plan_setup_s']})
                    all_steps.append(row)
                    if instrumentation=='scoped':
                        measured=sum(v[0] for v in scopes.rows.values())
                        scopes.rows[('other_solve','unwrapped_dispatch_and_other')]=[elapsed-measured,elapsed-measured,1]
                        for (stage,category),(exclusive,inclusive,calls) in scopes.rows.items():
                            all_phases.append(dict(window=window,plant=plant,execution_mode=execution,step=index,
                                history_before=history,replay_calls=counters['post_accept_replay_calls'],stage=stage,category=category,
                                exclusive_seconds=exclusive,inclusive_seconds_diagnostic_only=inclusive,calls=calls))
            scopes.uninstall()
            if instrumentation=='cprofile':
                prefix=OUT/f'{window}_{execution}'
                profiler.dump_stats(str(prefix.with_suffix('.pstats')))
                stats=pstats.Stats(profiler)
                rows=[dict(file=f,line=l,function=n,primitive_calls=v[0],total_calls=v[1],self_seconds=v[2],inclusive_seconds=v[3]) for (f,l,n),v in stats.stats.items()]
                write_csv(prefix.with_suffix('.csv'),sorted(rows,key=lambda r:-r['self_seconds']))
            write_csv(OUT/'steps.csv',all_steps)
            if all_phases:write_csv(OUT/'phases.csv',all_phases)
            print(json.dumps(dict(window=window,execution=execution,instrumentation=instrumentation,solve_seconds=elapsed_total)),flush=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==sha
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
(OUT/'method.json').write_text(json.dumps(dict(source_sha=sha,source_root=str(SOURCE),
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),affinity=sorted(os.sched_getaffinity(0)),
    threads=1,dtype='float64',production_denominator=False,
    exclusive_method='Every nested wrapper subtracts measured child wall time; only exclusive_seconds may be summed. The residual closes each measured step total. cProfile self time is a different run.',
    classification='stage is the enclosing selected operation. Polynomial calls during post_accept use fixed coefficients/domain. Python-object and small-tensor validation constructors are disjoint; their torch calls are inside constructor wall time, so this category is not pure Python. Unwrapped residual is not a claim that all remaining time is Python.',
    preparation='The plan counter includes deep-copy binding, first fixed operation callbacks and fixed polynomial-difference setup. It is a subcomponent inside solve. Dispatch and binding checks execute during every replay and remain in solve.',
    export='No serialization occurs inside measured step. Formal production export/checkpoint and whole-process times are recorded separately by run.py and benchmark.py.',
),indent=2)+'\n')
