"""Bounded independent recomputation from serialized actual replay inputs."""
import argparse
import gzip
import json
from pathlib import Path

import torch
import torch_tm_flowpipe.batched_dense_tm as d
from torch_tm_flowpipe import PolynomialODE
from torch_tm_flowpipe.ode_examples import brusselator_ode
from torch_tm_flowpipe.prepared_remainder_replay import PreparedRemainderReplay, prepared_remainder_replay
from experiments.endpoint_roundoff_repair.frozen import load_contract
from experiments.repaired_solver_performance.compare import read, require
from experiments.repaired_solver_performance.replay_evidence import decode_model, decode_tensor, exact, proposal_record


def decomposition(value):
    ledger=d.DenseRemainderLedger({k:tuple(decode_tensor(x) for x in pair) for k,pair in value['ledger']['entries'].items()})
    return d.DenseValidatedRemainderDecomposition(
        ledger,*(decode_tensor(value[k]) for k in ('decomposition_lo','decomposition_hi','padding_lo','padding_hi','contains_image')),
        value['source_schema'],value['source_schema_version'])


def verify_replay_file(path,*,recompute=True):
    path=Path(path);summary=read(path.with_suffix('.summary.json'))
    count=proposals=0;previous=None
    with gzip.open(path,'rt') as stream:
        for line in stream:
            row=json.loads(line);plant=row['plant'];index=row['step'];params=row['parameters']
            require(plant==summary['plant'] and plant in ('brusselator','van_der_pol'),'replay plant identity')
            require(previous is None or index==previous+1,'replay window step sequence');previous=index
            require(params['order']==(6 if plant=='brusselator' else 4) and params['tau_index']==2,'frozen replay order/time index')
            require(params['cutoff_threshold']==1e-10 and params['validation_eps']==1e-12,'frozen replay budgets')
            require(row['reference_proposals']==row['prepared_same_input_proposals'],'saved same-input proposal mismatch')
            require(row['reference_loop']==row['independent_optimized_loop'],'saved independent replay loop mismatch')
            require(len(row['reference_proposals'])==len(row['reference_loop'][3]),'per-round proposal coverage')
            if not recompute:
                count+=1;proposals+=len(row['reference_proposals']);continue
            for model in (row['base'],row['candidate']):
                require(model['basis_dim']==3 and model['basis_order']==params['order'],'replay polynomial basis')
                require(model['coefficients']['shape'][:2]==[1,2],'saved replay is the actual B1 solve')
            base,candidate=decode_model(row['base']),decode_model(row['candidate'])
            h=.02 if plant=='brusselator' else .01
            require(float(candidate.domain_lo[0,2]).hex()==0.0.hex() and float(candidate.domain_hi[0,2]).hex()==h.hex(),'actual fixed h in replay domain')
            rhs=brusselator_ode if plant=='brusselator' else PolynomialODE.from_system_spec(load_contract()['canonical_system_spec'])
            lo,hi=decode_tensor(row['retained_lo']),decode_tensor(row['retained_hi'])
            mode='ordered_terms' if plant=='brusselator' else 'canonical_factorized_joint_closure'
            cache=None
            if plant=='van_der_pol':
                cache=d._prepare_vdp_refinement_static_cache(rhs,base,candidate,candidate.with_remainder(lo,hi),**params)
            prepared=PreparedRemainderReplay(rhs,base,candidate,**params) if plant=='brusselator' else None
            for saved in row['reference_proposals']:
                rlo,rhi=decode_tensor(saved['input_lo']),decode_tensor(saved['input_hi'])
                reference=d._dense_flowstar_raw_compat_image(
                    rhs,base,candidate.with_remainder(rlo,rhi),candidate,**params,
                    raw_rhs_evaluation=mode,raw_dependency_preserving_square=True if cache is not None else None,
                    refinement_static_cache=cache,record_evidence=True)
                actual=proposal_record(rlo,rhi,reference)
                require(actual==saved,f'proposal/ledger/coefficient recomputation step {index}')
                if prepared is not None:
                    require(proposal_record(rlo,rhi,prepared.image(rlo,rhi))==saved,'prepared same-input recomputation')
                proposals+=1
            kwargs=dict(**params,retained_lo=lo,retained_hi=hi,retained_decomposition=decomposition(row['retained_decomposition']),
                        structural_fingerprint=None if cache is None else d._frozen_vdp_structural_fingerprint(rhs),
                        raw_rhs_evaluation=mode,raw_dependency_preserving_square=True if cache is not None else None,
                        observer_mode=d.DENSE_OBSERVER_FULL)
            for enabled,key in ((False,'reference_loop'),(True,'independent_optimized_loop')):
                with prepared_remainder_replay(enabled):
                    output=d._post_accept_refine_raw_remainder(rhs,base,candidate,**kwargs)
                require(exact(output)==row[key],'independently recomputed R/commit/stop sequence')
            count+=1
    require(count==summary['steps'] and proposals==summary['proposals'],'window counts recomputed')
    require(summary['same_input_and_independent_loop_bit_identical'] is True and summary['production_timing'] is False,'evidence cannot be a production denominator')
    return dict(file=path.name,plant=summary['plant'],steps=count,proposals=proposals,recomputed=recompute,passed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    result=verify_replay_file(args.path)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
