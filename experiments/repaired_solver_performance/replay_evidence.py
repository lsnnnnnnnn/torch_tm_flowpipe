"""Save and independently replay actual per-round inputs; never used for timing."""
import argparse
from dataclasses import asdict, fields, is_dataclass
import gzip
import json
from pathlib import Path
import subprocess
import time

import torch
import torch_tm_flowpipe.batched_dense_tm as d
from torch_tm_flowpipe import load_terminal_checkpoint
from torch_tm_flowpipe.prepared_remainder_replay import PreparedRemainderReplay, prepared_remainder_replay
from experiments.endpoint_roundoff_repair.frozen import ROOT, setup, step


def exact(value):
    if isinstance(value, torch.Tensor):
        return {"dtype":str(value.dtype), "shape":list(value.shape), "values":exact(value.detach().cpu().tolist())}
    if isinstance(value, float): return value.hex()
    if is_dataclass(value): return {f.name:exact(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value, dict): return {k:exact(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [exact(v) for v in value]
    return value


def decode_tensor(value):
    def numbers(v):
        return [numbers(x) for x in v] if isinstance(v,list) else float.fromhex(v) if isinstance(v,str) else v
    return torch.tensor(numbers(value['values']),dtype=getattr(torch,value['dtype'].split('.')[-1])).reshape(value['shape'])


def encode_model(model):
    return dict(coefficients=exact(model.poly.coeffs),basis_dim=model.poly.basis.dim,basis_order=model.poly.basis.order,
                remainder=[exact(model.rem_lo),exact(model.rem_hi)],domain=[exact(model.domain_lo),exact(model.domain_hi)],
                ledger={k:[exact(lo),exact(hi)] for k,(lo,hi) in model.ledger.entries.items()},range_policy=asdict(model.range_policy))


def decode_model(value):
    basis=d.BatchedMonomialBasis.build(value['basis_dim'],value['basis_order'],'cpu')
    policy={**value['range_policy']}
    for key in ('split_vars','named_contexts'):
        policy[key]=tuple(policy[key])
    return d.BatchedTaylorModel(d.BatchedPolynomial(decode_tensor(value['coefficients']),basis),
        *(decode_tensor(v) for v in value['remainder']),*(decode_tensor(v) for v in value['domain']),
        d.DenseRemainderLedger({k:tuple(decode_tensor(v) for v in pair) for k,pair in value['ledger'].items()}),
        d.DenseRangePolicy(**policy),None)


def proposal_record(lo,hi,result):
    return dict(input_lo=exact(lo),input_hi=exact(hi),proposal=exact(result),
                decision=exact(d._atomic_refinement_decision(lo,hi,*result[:2])))


def capture_window(plant,count,checkpoint=None):
    _,current,state=setup(plant)
    if checkpoint is not None:
        restored=load_terminal_checkpoint(checkpoint,expected_dtype='float64')
        assert restored.contract['plant']==plant
        current,state=restored.current,restored.normal_state
    initial_step=state.step_index
    original_loop=d._post_accept_refine_raw_remainder
    original_image=d._dense_flowstar_raw_compat_image
    captured=[]
    for index in range(initial_step+1,initial_step+count+1):
        active=False
        proposals=[]
        binding=[]
        def image(*args,**kwargs):
            output=original_image(*args,**kwargs)
            if active:
                proposals.append(proposal_record(args[2].rem_lo,args[2].rem_hi,output))
            return output
        def loop(*args,**kwargs):
            nonlocal active
            kwargs={**kwargs,'observer_mode':d.DENSE_OBSERVER_FULL}
            active=True
            try: output=original_loop(*args,**kwargs)
            finally: active=False
            binding.append((args,kwargs,output))
            return output
        d._post_accept_refine_raw_remainder=loop
        d._dense_flowstar_raw_compat_image=image
        try:
            with prepared_remainder_replay(False):
                reference=step(plant,current,state,index)
        finally:
            d._post_accept_refine_raw_remainder=original_loop
            d._dense_flowstar_raw_compat_image=original_image
        assert reference.status=='validated',reference.message
        assert len(binding)==1
        args,kwargs,reference_loop=binding[0]
        rhs,base,candidate=args
        params={k:kwargs[k] for k in ('tau_index','order','cutoff_threshold','validation_eps')}
        same=[]
        # Generic preparation is deliberately not substituted for the existing
        # canonical VDP closure. VDP checks the unchanged path and whole loop.
        plan=PreparedRemainderReplay(rhs,base,candidate,**params) if plant=='brusselator' else None
        vdp_cache=None
        if plan is None:
            seed=candidate.with_remainder(kwargs['retained_lo'],kwargs['retained_hi'])
            vdp_cache=d._prepare_vdp_refinement_static_cache(rhs,base,candidate,seed,**params)
        for record in proposals:
            lo,hi=decode_tensor(record['input_lo']),decode_tensor(record['input_hi'])
            if plan is not None:
                new=plan.image(lo,hi)
            else:
                with prepared_remainder_replay(True):
                    new=original_image(rhs,base,candidate.with_remainder(lo,hi),candidate,**params,
                        raw_rhs_evaluation='canonical_factorized_joint_closure',raw_dependency_preserving_square=True,
                        refinement_static_cache=vdp_cache,record_evidence=True)
            compared=proposal_record(lo,hi,new)
            assert compared==record, (plant,index,'same input')
            same.append(compared)
        with prepared_remainder_replay(True):
            optimized_loop=original_loop(*args,**kwargs)
        assert exact(reference_loop)==exact(optimized_loop),(plant,index,'independent loop')
        captured.append(dict(
            plant=plant,step=index,parameters=params,base=encode_model(base),candidate=encode_model(candidate),
            retained_lo=exact(kwargs['retained_lo']),retained_hi=exact(kwargs['retained_hi']),
            retained_decomposition=exact(kwargs['retained_decomposition']),
            reference_proposals=proposals,prepared_same_input_proposals=same,
            reference_loop=exact(reference_loop),independent_optimized_loop=exact(optimized_loop),
            endpoint_error=exact(reference.endpoint_substitution_roundoff),
            work_counts=plan.work_counts() if plan is not None else {'existing_vdp_cache':True},
        ))
        current,state=reference.reset_tm,reference.flowstar_normal_state
    return captured


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plant',choices=['brusselator','van_der_pol'],required=True)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    started=time.perf_counter()
    rows=capture_window(args.plant,args.steps,args.checkpoint)
    with gzip.open(args.output,'xt') as handle:
        for row in rows:handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    summary=dict(source_sha=sha,plant=args.plant,steps=len(rows),start_step=rows[0]['step'],
                 checkpoint=str(args.checkpoint),proposals=sum(len(r['reference_proposals']) for r in rows),
                 same_input_and_independent_loop_bit_identical=True,seconds_including_evidence=time.perf_counter()-started,
                 production_timing=False)
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
