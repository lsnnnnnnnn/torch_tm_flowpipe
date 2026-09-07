"""Exact inclusion checks for the production sparse diagonal preconditioner.

This is an algebra test, not a third ODE benchmark. The input Taylor model is
the finite constant a, with zero remainder. Scaling by S and its computed
inverse must still represent that same exact binary64 constant afterwards.
"""
import argparse
from fractions import Fraction
import inspect
import json
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch
    from flowstar_gpu.sparse_exec import _g_precond
    from flowstar_gpu import cuda_kernels as ck
    torch.set_num_threads(1)
    assert ck.available()
    rows=[]
    for device in ['cpu','cuda']:
        for a in [1.1, 1.3, 0.1, 1e-305]:
            tensor=lambda v:torch.tensor(v,dtype=torch.float64,device=device)
            original=tensor([[[a]]]);remainder=tensor([[[0.0,0.0]]])
            cat=tensor([[1.0,1.0]]);c0=tensor([[0.0]])
            pos=torch.tensor([1],device=device); arange=torch.tensor([0],device=device)
            scaled,rem,S,point,inv,x0=_g_precond(original,remainder,cat,c0,pos,arange,2)
            s,c=Fraction(S.item()),Fraction(scaled.item())
            l,u=[Fraction(v) for v in rem.flatten().tolist()]
            lower,upper=s*(c+l),s*(c+u)
            exact=Fraction(a)
            rows.append({'device':device,'a':a,'a_hex':a.hex(),'S':S.item(),'inverse':inv.item(),
                         'scaled_coefficient':scaled.item(),'scaled_remainder':rem.flatten().tolist(),
                         'point_dimension':point.item(),'exact_input_fraction':str(exact),
                         'exact_reconstructed_lower':str(lower),'exact_reconstructed_upper':str(upper),
                         'exact_error_of_point':str(s*c-exact),'contained':lower<=exact<=upper})
        original=tensor([[[0.1,1.1]]]); remainder=tensor([[[0.0,0.0]]])
        scaled,rem,S,point,inv,x0=_g_precond(original,remainder,tensor([[1.0,1.0],[-1.0,1.0]]),c0,pos,arange,2)
        for u in [-1,1]:
            exact=Fraction(0.1)+u*Fraction(1.1)
            c=Fraction(scaled[0,0,0].item())+u*Fraction(scaled[0,0,1].item())
            s=Fraction(S.item());l,h=map(Fraction,rem.flatten().tolist())
            lower,upper=s*(c+l),s*(c+h)
            rows.append({'device':device,'a':'0.1 + 1.1*u','u':u,'original_coefficients':[0.1,1.1],
                         'S':S.item(),'inverse':inv.item(),'scaled_coefficients':scaled.flatten().tolist(),
                         'scaled_remainder':rem.flatten().tolist(),'exact_input_fraction':str(exact),
                         'exact_reconstructed_lower':str(lower),'exact_reconstructed_upper':str(upper),
                         'exact_error_of_point':str(s*c-exact),'contained':lower<=exact<=upper})
    fn=Path(inspect.getsourcefile(_g_precond)).resolve()
    root=fn.parents[2]
    result={'schema':'xiangru_preconditioning_exact/1','source_function':str(fn),
            'source_line':inspect.getsourcelines(_g_precond)[1],
            'source_sha':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
            'mode_dispatch':'The production preconditioning function has no strict-mode branch; same call for parity and strict.',
            'claim':'Local normalization representation inclusion; not a claim these states occur in frozen trajectories.',
            'rows':rows}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ['device','a','contained','exact_error_of_point']} for r in rows],indent=2))

if __name__=='__main__':main()
