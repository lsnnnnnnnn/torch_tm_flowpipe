"""Bounded rejection/isolation/invalid-input checks on the matched VDP input."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    from flowstar_gpu.config import Settings
    from flowstar_gpu.flowpipe import initial_flowpipe, advance, reach, build_rem_est
    from flowstar_gpu.monomials import build_tables
    from flowstar_gpu.polynomial import build_step_tables
    from flowstar_gpu.composition import build_schedule
    from flowstar_gpu.ode_compiler import compile_ode
    from flowstar_gpu.symbolic_remainder import make_symbolic_remainder
    torch.set_num_threads(1)
    rows=[];fields=['pre_coeffs','pre_rem','tmv_coeffs','tmv_rem']
    rhs=['y','y - x - x*x*y'];box=[[1.1,1.4],[2.35,2.45]]
    for device in ['cpu','cuda']:
      for mode in ['parity','strict']:
        settings=Settings(step=0.01,order=4,sr_queue=100,mode=mode,device=device)
        tab=build_tables(2,4).to(device);step=build_step_tables(tab,0.01);sched=build_schedule(2,4,device)
        code=compile_ode(rhs,['x','y'],order=3)
        boxes=torch.tensor([box,[[-8.,8.],[-8.,8.]]],dtype=torch.float64,device=device)
        fp=initial_flowpipe(boxes,tab);sr=make_symbolic_remainder(2,2,100,device)
        before={k:getattr(fp,k).clone() for k in fields}
        out,ok=advance(fp,code,tab,step,sched,settings,build_rem_est(settings,2,2),sr)
        out2,ok2=advance(out,code,tab,step,sched,settings,build_rem_est(settings,2,2),sr)
        one=reach(rhs,['x','y'],boxes[:1],0.02,settings,record_tms=True)
        rows.append({'device':device,'mode':mode,'check':'mixed_fate_rejection',
                     'first_ok':ok.tolist(),'second_ok':ok2.tolist(),
                     'failed_flowpipe_frozen':all(torch.equal(getattr(out2,k)[1],before[k][1]) for k in fields),
                     'successful_lane_equal_to_individual':all(torch.equal(getattr(out2,k)[0],getattr(one.final_fp,k)[0]) for k in fields),
                     'sr_queue_mutates_for_failed_lane':sr.qlen>0,
                     'semantics':'Upstream freezes published flowpipe but advances a global SR queue with junk for stopped lanes; no supported resume of a stopped lane.'})
        invalid=boxes[:1].clone();invalid[0,0,0]=float('nan')
        try:
            r=reach(rhs,['x','y'],invalid,0.01,settings)
            rows.append({'device':device,'mode':mode,'check':'nonfinite_input',
                         'accepted_steps':r.steps_completed.tolist(),'status':r.status.tolist(),
                         'invalid_success_published':bool((r.steps_completed>0).any()),'exception':None})
        except Exception as exc:
            rows.append({'device':device,'mode':mode,'check':'nonfinite_input','invalid_success_published':False,
                         'exception':type(exc).__name__+': '+str(exc)})
    try: Settings(step=0.1,step_min=0.002,order=4,sr_queue=100,mode='strict')
    except NotImplementedError as exc: rows.append({'check':'adaptive_plus_sr','supported':False,'exception':str(exc)})
    try: reach(rhs,['x','y'],torch.zeros(1,3,2,dtype=torch.float64),0.01,Settings(step=0.01,order=4,mode='strict',device='cpu'))
    except ValueError as exc: rows.append({'check':'mismatched_task_dimension','rejected':True,'exception':str(exc)})
    rows.append({'check':'checkpoint_resume','status':'NO_PUBLIC_CHECKPOINT_RESUME_CONTRACT',
                 'reason':'No candidate checkpoint API or documented resume promise found in the plant execution modules; no new resume feature added.'})
    from flowstar_gpu import sparse_exec as se, support as sp
    contract=json.loads((a.output.parent/'MATCHED_CONTRACTS.json').read_text())
    for plant,cfg in contract['plants'].items():
        settings=Settings(step=float(cfg['step']['decimal']),order=cfg['order'],sr_queue=cfg['sr_capacity'],mode='strict',device='cuda')
        tab=build_tables(2,settings.order).to('cuda');step=build_step_tables(tab,settings.step)
        sched=build_schedule(2,settings.order,'cuda');code=compile_ode(cfg['rhs_expression_strings'],['x','y'],order=settings.order-1)
        boxes=torch.tensor([[b['enclosing_binary64'] for b in box] for box in cfg['different_task_boxes_binary'][:2]],dtype=torch.float64,device='cuda')
        def run_two(inputs):
            eng=sp.SparseEngine(tab,step,'cuda');st=se.initial_sparse_state(inputs,eng,sched)
            queue=make_symbolic_remainder(len(inputs),2,settings.sr_queue,'cuda');states=[]
            for _ in range(2):
                st,ok=se.advance_sparse(st,code,eng,sched,settings,build_rem_est(settings,2,len(inputs)),queue)
                st=se.prune_state(st,eng)
                pre=torch.zeros(len(inputs),2,tab.T,dtype=torch.float64,device='cuda');pre[...,list(st.pre_sup.ids)]=st.pre
                tmv=torch.zeros(len(inputs),2,tab.Ts,dtype=torch.float64,device='cuda');tmv[...,list(st.tmv_sup.ids)]=st.tmv
                states.append(([x.cpu().clone() for x in [pre,st.pre_rem,tmv,st.tmv_rem]],ok.tolist()))
            return states
        batch=run_two(boxes)
        for lane in range(2):
            individual=run_two(boxes[lane:lane+1])
            for i,((actual,ok),(expected,one_ok)) in enumerate(zip(batch,individual),start=1):
                rows.append({'check':'different_preregistered_boxes_batch_equivalence','plant':plant,'device':'cuda','mode':'strict','backend':'sparse',
                             'batch':2,'lane':lane,'step':i,'accepted':ok[lane] and one_ok[0],
                             'bitwise_equal':all(torch.equal(x[lane:lane+1],y) for x,y in zip(actual,expected)),
                             'scope':'Two-step diagnostic, not long-prefix correctness admission.'})
    a.output.write_text(json.dumps({'schema':'xiangru_state_checks/1','rows':rows},indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
