"""Capture real accepted attempts and compare two independently evolving loops."""
import argparse
from dataclasses import fields
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch_tm_flowpipe.batched_dense_tm as d
from torch_tm_flowpipe import load_terminal_checkpoint, tmvector_hashes, accepted_boundary_sr_queue_sha256
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from experiments.endpoint_roundoff_repair.frozen import ROOT, setup, step


def tensor(value):
    return {'shape':list(value.shape),'dtype':str(value.dtype).removeprefix('torch.'),
            'values':[float(v).hex() for v in value.reshape(-1)]}


def untensor(value):
    return torch.tensor([float.fromhex(v) for v in value['values']],dtype=getattr(torch,value['dtype'])).reshape(value['shape'])


def ledger(value):
    return {name:[tensor(lo),tensor(hi)] for name,(lo,hi) in value.entries.items()}


def unledger(value):
    return d.DenseRemainderLedger({name:tuple(untensor(x) for x in pair) for name,pair in value.items()})


def model(value):
    return {'coefficients':tensor(value.poly.coeffs),'basis_dim':value.poly.basis.dim,'order':value.poly.basis.order,
            'remainder':[tensor(value.rem_lo),tensor(value.rem_hi)],'domain':[tensor(value.domain_lo),tensor(value.domain_hi)],
            'ledger':ledger(value.ledger),'policy':{f.name:getattr(value.range_policy,f.name) for f in fields(value.range_policy)}}


def unmodel(value):
    policy=dict(value['policy'])
    for key in ['split_vars','named_contexts','variable_orders']:
        if key in policy:
            policy[key]=tuple(tuple(x) if isinstance(x,list) else x for x in policy[key])
    return d.BatchedTaylorModel(d.BatchedPolynomial(untensor(value['coefficients']),d.BatchedMonomialBasis.build(value['basis_dim'],value['order'])),
        *(untensor(x) for x in value['remainder']),*(untensor(x) for x in value['domain']),
        unledger(value['ledger']),d.DenseRangePolicy(**policy))


def decomposition(value):
    return {'ledger':ledger(value.ledger),**{name:tensor(getattr(value,name)) for name in
            ['decomposition_lo','decomposition_hi','padding_lo','padding_hi','contains_image']}}


def undecomposition(value):
    return d.DenseValidatedRemainderDecomposition(unledger(value['ledger']),
        **{name:untensor(v) for name,v in value.items() if name!='ledger'})


def result(value):
    return {'lo':tensor(value[0]),'hi':tensor(value[1]),'decomposition':decomposition(value[2]),'rows':value[3]}


def equal(a,b):
    # JSON round-trip canonicalizes tuple/list representation; hex float
    # normalization distinguishes signed zero and never introduces tolerances.
    def canonical(x):
        if isinstance(x,float):return {'binary64':x.hex()}
        if isinstance(x,dict):return {k:canonical(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
        return x
    assert canonical(a)==canonical(b),'bitwise replay mismatch'


def verify_record(record):
    from torch_tm_flowpipe.ode_examples import brusselator_ode
    from torch_tm_flowpipe import PolynomialODE
    from experiments.run_vdp_dense_backend import load_contract
    ode=brusselator_ode if record['plant']=='brusselator' else PolynomialODE.from_system_spec(load_contract()['canonical_system_spec'])
    base,candidate=unmodel(record['base']),unmodel(record['candidate'])
    kwargs={**record['kwargs'],'retained_lo':untensor(record['retained_lo']),
            'retained_hi':untensor(record['retained_hi']),
            'retained_decomposition':undecomposition(record['retained_decomposition'])}
    for mode in [False,True]:
        with prepared_remainder_replay(mode):
            actual=result(d._post_accept_refine_raw_remainder(ode,base,candidate,**kwargs))
        equal(actual,record['optimized' if mode else 'reference'])
    equal(record['reference'],record['optimized'])
    return len(record['reference']['rows'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plant',required=True,choices=['brusselator','van_der_pol'])
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--adaptive',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source_clean=not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    _,current,state=setup(args.plant)
    h=.1 if args.adaptive else (.02 if args.plant=='brusselator' else .01)
    if args.checkpoint:
        restored=load_terminal_checkpoint(args.checkpoint)
        current,state=restored.current,restored.normal_state
        if args.adaptive:h=float.fromhex(restored.scheduler['next_h_hex'])
    args.output.mkdir(parents=True,exist_ok=False)
    original=d._post_accept_refine_raw_remainder
    records=[]
    active_index=0
    def compare(ode,base,candidate,**kwargs):
        actual_kwargs={k:v for k,v in kwargs.items() if k not in ['counters','retained_lo','retained_hi','retained_decomposition']}
        actual_kwargs['observer_mode']=d.DENSE_OBSERVER_FULL
        record={'plant':args.plant,'step':active_index,'base':model(base),'candidate':model(candidate),
                'retained_lo':tensor(kwargs['retained_lo']),'retained_hi':tensor(kwargs['retained_hi']),
                'retained_decomposition':decomposition(kwargs['retained_decomposition']),'kwargs':actual_kwargs}
        results=[]
        for mode in [False,True]:
            started=time.perf_counter()
            with prepared_remainder_replay(mode):
                value=original(ode,base,candidate,**{**kwargs,'observer_mode':d.DENSE_OBSERVER_FULL,
                               'counters':None if mode else kwargs.get('counters')})
            results.append(value)
            record['optimized' if mode else 'reference']=result(value)
            record['optimized_evidence_seconds' if mode else 'reference_evidence_seconds']=time.perf_counter()-started
        equal(record['reference'],record['optimized'])
        records.append(record)
        return (*results[0][:3],results[0][3] if kwargs.get('observer_mode')!=d.DENSE_OBSERVER_NONE else ())
    d._post_accept_refine_raw_remainder=compare
    states=[]
    try:
        for active_index in range(state.step_index+1,state.step_index+args.steps+1):
            segment=step(args.plant,current,state,active_index,h=h,adaptive=args.adaptive)
            assert segment.status=='validated',segment.message
            current,state=segment.reset_tm,segment.flowstar_normal_state
            if args.adaptive:h=segment.next_h or h
            states.append({'step':active_index,'h_hex':segment.h.hex(),'step_rejections':segment.step_rejections,
                'endpoint':tmvector_hashes(segment.endpoint_raw_tm),'tube':tmvector_hashes(segment.tm),
                'endpoint_error':[[float(x.lo).hex(),float(x.hi).hex()] for x in segment.endpoint_substitution_roundoff],
                'queue':accepted_boundary_sr_queue_sha256(state.symbolic_queue)})
            print(json.dumps({'step':active_index,'matched_rounds':len(records[-1]['reference']['rows'])}),flush=True)
    finally:
        d._post_accept_refine_raw_remainder=original
    with gzip.open(args.output/'replays.jsonl.gz','wt') as out:
        for record in records:out.write(json.dumps(record,allow_nan=False)+'\n')
    (args.output/'states.json').write_text(json.dumps(states,indent=2)+'\n')
    (args.output/'source.json').write_text(json.dumps({'sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_clean_at_start':source_clean,'imported_solver':d.__file__,'python':sys.executable,
        'affinity':sorted(os.sched_getaffinity(0)),'threads':torch.get_num_threads(),'torch':torch.__version__,
        'dense_source_sha256':hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest(),
        'root':str(ROOT),'checkpoint':str(args.checkpoint),'profile_or_evidence_not_production_timing':True,
        'records':len(records),'matched_rounds':sum(len(x['reference']['rows']) for x in records)},indent=2)+'\n')


if __name__=='__main__':main()
