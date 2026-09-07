"""Two-step diagnostics only after a confirmed defect; never a speed gate."""
from dataclasses import asdict
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from common import measure

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch
    import flowstar_gpu
    from flowstar_gpu.config import Settings
    from flowstar_gpu.determinism import enable_determinism
    from flowstar_gpu.monomials import build_tables
    from flowstar_gpu.polynomial import build_step_tables
    from flowstar_gpu.composition import build_schedule
    from flowstar_gpu.ode_compiler import compile_ode
    from flowstar_gpu import flowpipe as dense, sparse_exec as se, support as sp, safety
    from flowstar_gpu import cuda_kernels as ck, tape_kernels as tk
    from flowstar_gpu.symbolic_remainder import make_symbolic_remainder
    enable_determinism();torch.set_num_threads(1)
    contract=json.loads(args.contract.read_text());rows=[];equivalence=[]
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    objects=(args.output/'models.jsonl').open('w')
    saved={}
    def tensor_model(c,r,tab,domain):
        cs=c[0].cpu();rs=r[0].cpu();exps=tab.exponents[:cs.shape[-1]].cpu().tolist()
        return {'domain':domain,'variables':['tau','ux','uy'],'components':[
            {'remainder':rs[i].tolist(),'terms':[{'degrees':d,'coefficient':[float(v),float(v)]}
             for d,v in zip(exps,cs[i]) if float(v)!=0]} for i in range(2)]}
    for name,config in contract['plants'].items():
      for device,backend in [('cpu','dense'),('cpu','sparse'),('cuda','dense'),('cuda','sparse')]:
       for mode in ['parity','strict']:
        batches=[1,2,8] if device=='cuda' and backend=='sparse' and mode=='strict' else [1]
        for B in batches:
          settings=Settings(step=float(config['step']['decimal']),order=config['order'],cutoff=1e-10,
                            remainder_estimation=1e-4,sr_queue=config['sr_capacity'],mode=mode,device=device)
          tab=build_tables(2,settings.order).to(device);step=build_step_tables(tab,settings.step)
          sched=build_schedule(2,settings.order,device)
          code=compile_ode(config['rhs_expression_strings'],['x','y'],order=settings.order-1)
          boxes=torch.tensor([[b['enclosing_binary64'] for b in config['initial_representation']]],dtype=torch.float64,device=device).repeat(B,1,1)
          sr=make_symbolic_remainder(B,2,settings.sr_queue,device)
          rem=dense.build_rem_est(settings,2,B)
          eng=sp.SparseEngine(tab,step,device)
          state=se.initial_sparse_state(boxes,eng,sched) if backend=='sparse' else dense.initial_flowpipe(boxes,tab)
          for index in [1,2]:
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            if backend=='sparse':
                state,ok=se.advance_sparse(state,code,eng,sched,settings,rem,sr)
                state=se.prune_state(state,eng)
                pre=torch.zeros(B,2,tab.T,dtype=torch.float64,device=device)
                pre[...,list(state.pre_sup.ids)]=state.pre
                tmv=torch.zeros(B,2,tab.Ts,dtype=torch.float64,device=device)
                tmv[...,list(state.tmv_sup.ids)]=state.tmv
                pre_rem,tmv_rem=state.pre_rem,state.tmv_rem
            else:
                state,ok=dense.advance(state,code,tab,step,sched,settings,rem,sr)
                pre,pre_rem,tmv,tmv_rem=state.pre_coeffs,state.pre_rem,state.tmv_coeffs,state.tmv_rem
            if device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            row={'plant':name,'device':device,'backend':backend,'mode':mode,'batch':B,'step':index,
                 'accepted':ok.cpu().tolist(),'status':state.status.cpu().tolist(),'diagnostic_step_seconds':elapsed,
                 'elapsed_includes_cold_specialization_or_build':True,'adoption_eligible':False,
                 'reason':'Confirmed SR and normalization inclusion defects; bounded diagnostic only.',
                 'settings':asdict(settings),'queue_len':sr.queue_len}
            if not bool(ok.all()):rows.append(row);break
            tensors=[pre,pre_rem,tmv,tmv_rem]
            key=(name,device,backend,mode,index)
            if B==1:saved[key]=[v.cpu().clone() for v in tensors]
            elif key in saved:
                for lane in range(B):
                    equivalence.append({'plant':name,'device':device,'backend':backend,'mode':mode,
                        'batch':B,'step':index,'lane':lane,
                        'bitwise_equal':all(torch.equal(v[lane:lane+1].cpu(),expected) for v,expected in zip(tensors,saved[key]))})
            if B==1:
                c,r=safety._composed_tm(pre,pre_rem,tmv,tmv_rem,tab,step,settings)
                record={k:row[k] for k in ['plant','device','backend','mode','step']};record['models']={}
                row['bounds']={}
                for label,interval in [('endpoint',[settings.step,settings.step]),('tube',[0.,settings.step])]:
                    model=tensor_model(c,r,tab,[interval,[-1.,1.],[-1.,1.]])
                    record['models'][label]=model
                    # Actual integration-driver range convention, explicitly
                    # distinguished from the complete composed export.
                    published=safety.rows_range_over_time(pre,pre_rem,torch.tensor([interval],dtype=torch.float64,device=device),tab)[0].cpu().tolist()
                    row['bounds'][label]={'published_pre_range':published,'common_composed_range':measure(model)}
                objects.write(json.dumps(record,allow_nan=False)+'\n');objects.flush()
            rows.append(row)
            sr.reset_if_full()
          print(json.dumps({'plant':name,'device':device,'backend':backend,'mode':mode,'B':B,'steps':index}),flush=True)
    objects.close()
    result={'schema':'xiangru_short_diagnostics/1','imported_package':flowstar_gpu.__file__,
            'python':sys.executable,'torch_version':torch.__version__,'cuda_build':torch.version.cuda,
            'segment_extension':ck._ext.__file__ if ck._ext else None,
            'tape_extension':tk._ext.__file__ if tk._ext else None,'rows':rows,'batch_equivalence':equivalence}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
