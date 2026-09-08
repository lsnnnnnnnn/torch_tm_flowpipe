"""Bounded baseline measurement, with timing scopes disjoint by subtraction."""
import cProfile
import csv
import functools
import gzip
import itertools
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
from experiments.endpoint_roundoff_repair.frozen import ROOT, setup, step
from torch_tm_flowpipe.brusselator_canonical_exchange import _append_tmv
from experiments.xiangru_adoption.common import from_existing_canonical

OUT = Path(__file__).resolve().parent/'baseline_profile'
OLD = ROOT/'artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal'
OUT.mkdir(exist_ok=False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert not subprocess.check_output(['git','status','--porcelain'], text=True).strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == '7e41f33f515f5315b0dec7003a5b06ed0a79afd5'


def write_csv(path, rows):
    with path.open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


class Scopes:
    def __init__(self):
        self.stack=[]; self.rows={}; self.active=False; self.refining=0
        self.originals=[]

    def wrap(self, obj, name, category):
        original=getattr(obj,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            if not self.active:
                return original(*args,**kwargs)
            group=category() if callable(category) else category
            frame=[time.perf_counter(),0.0]
            self.stack.append(frame)
            refining=name=='_post_accept_refine_raw_remainder'
            self.refining+=refining
            try:
                return original(*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-frame[0]
                self.refining-=refining
                self.stack.pop()
                if self.stack: self.stack[-1][1]+=elapsed
                row=self.rows.setdefault(group,[0.0,0.0,0])
                row[0]+=elapsed-frame[1]; row[1]+=elapsed; row[2]+=1
        setattr(obj,name,measured)
        self.originals.append((obj,name,original))

    def install(self):
        self.wrap(dense,'dense_polynomial_picard','candidate_polynomial')
        self.wrap(dense,'_post_accept_refine_raw_remainder','refinement_dynamic')
        self.wrap(dense,'_dense_flowstar_raw_compat_image',lambda:'refinement_dynamic' if self.refining else 'initial_remainder')
        for name in ['mul_trunc','range_bound','integrate','apply_cutoff','add','sub','scale']:
            self.wrap(dense.BatchedPolynomial,name,lambda:'refinement_fixed_polynomial' if self.refining else 'polynomial_multiply_truncate_range')
        self.wrap(dense,'_prepare_vdp_refinement_static_cache','refinement_fixed_polynomial')
        self.wrap(dense,'_joint_factorized_vdp_residual_closure','refinement_dynamic')
        self.wrap(dense.BatchedTaylorModel,'endpoint','boundary_endpoint')
        self.wrap(flow,'_flowstar_normalized_insertion_transition','boundary_endpoint')
        self.wrap(flow,'_vdp_frozen_c3_accepted_boundary_prepare','cross_step_history')
        # Imported SR routines are measured where the flowpipe actually calls them.
        for name in ['prepare_accepted_boundary_sr','commit_accepted_boundary_sr']:
            if hasattr(flow,name): self.wrap(flow,name,'cross_step_history')

    def uninstall(self):
        for obj,name,original in reversed(self.originals): setattr(obj,name,original)


all_rows=[]; all_phases=[]; restorations=[]
for plant,start,count in [('brusselator',1,20),('brusselator',101,20),('van_der_pol',1,20)]:
    label=f'{plant}_{start:04d}_{start+count-1:04d}'
    results=[]
    for mode in ['production','scoped','cprofile']:
        config,current,state=setup(plant)
        if start>1:
            checkpoint=OLD/f'{plant}_full'/f'checkpoint_{start-1:04d}'
            restored=core.load_terminal_checkpoint(checkpoint,expected_dtype='float64')
            current,state=restored.current,restored.normal_state
        scopes=Scopes()
        if mode=='scoped': scopes.install()
        profiler=cProfile.Profile()
        total=0.0
        for index in range(start,start+count):
            scopes.rows={}; scopes.active=mode=='scoped'
            history=len(state.symbolic_queue.J) if state.symbolic_queue else 0
            if mode=='cprofile': profiler.enable()
            before=time.perf_counter()
            segment=step(plant,current,state,index)
            elapsed=time.perf_counter()-before
            if mode=='cprofile': profiler.disable()
            scopes.active=False
            assert segment.status=='validated',segment.message
            total+=elapsed
            current,state=segment.reset_tm,segment.flowstar_normal_state
            all_rows.append(dict(window=label,plant=plant,mode=mode,step=index,history_before=history,
                replay_calls=segment.backend_counters['post_accept_replay_calls'],solve_seconds=elapsed))
            if mode=='scoped':
                scoped=sum(v[0] for v in scopes.rows.values())
                scopes.rows['python_tensor_dispatch_and_other']=[elapsed-scoped,elapsed-scoped,1]
                for category,(exclusive,inclusive,calls) in scopes.rows.items():
                    all_phases.append(dict(window=label,plant=plant,step=index,history_before=history,
                        replay_calls=segment.backend_counters['post_accept_replay_calls'],category=category,
                        exclusive_seconds=exclusive,inclusive_seconds_diagnostic_only=inclusive,calls=calls))
            if start>1 and index==start and mode=='production':
                archive=OLD/f'{plant}_full'
                with gzip.open(archive/'models.jsonl.gz','rt') as handle:
                    old=json.loads(next(itertools.islice(handle,index-1,index)))
                for name,tm in [('endpoint',segment.endpoint_raw_tm),('tube',segment.tm)]:
                    values=[]; _append_tmv(values,name,tm,variable_order=('ux','uy','tau') if tm.n_vars==3 else ('ux','uy'))
                    assert from_existing_canonical(dict(values),name)==old['models'][name]
                audit=json.loads((archive/'endpoint_audit.jsonl').read_text().splitlines()[index-1])
                assert core.accepted_boundary_sr_queue_sha256(state.symbolic_queue)==audit['queue_sha256']
                restorations.append(dict(plant=plant,checkpoint=str(checkpoint),resumed_step=index,models_and_queue_bit_identical=True))
        scopes.uninstall()
        if mode=='cprofile':
            profiler.dump_stats(str(OUT/f'{label}.pstats'))
            stats=pstats.Stats(profiler)
            rows=[dict(file=f,line=l,function=n,primitive_calls=v[0],total_calls=v[1],self_seconds=v[2],inclusive_seconds=v[3]) for (f,l,n),v in stats.stats.items()]
            write_csv(OUT/f'{label}_cprofile.csv',sorted(rows,key=lambda r:-r['self_seconds']))
        results.append(dict(mode=mode,solve_seconds=total))
        print(json.dumps(dict(window=label,mode=mode,solve_seconds=total)),flush=True)
    write_csv(OUT/'steps.csv',all_rows)
    write_csv(OUT/'phases.csv',all_phases)
    (OUT/'restore_checks.json').write_text(json.dumps(restorations,indent=2)+'\n')
(OUT/'method.json').write_text(json.dumps(dict(
    source_sha='7e41f33f515f5315b0dec7003a5b06ed0a79afd5',
    observer_mode='production_no_observer',affinity=sorted(os.sched_getaffinity(0)),
    phase_method='Nested scope wall time minus all measured child scopes. Only exclusive_seconds may be summed. cProfile and scoped are separate diagnostic runs; production contains no wrappers or profiler.',
    fixed_definition='BatchedPolynomial operations depend on retained coefficients and fixed domain/policy only; VDP existing static construction is counted once.',
    production_semantics='Original frozen.step, unchanged final repaired source.',
),indent=2)+'\n')
