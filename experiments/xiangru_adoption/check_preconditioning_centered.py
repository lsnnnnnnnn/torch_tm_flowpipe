"""Check whole-set inclusion with the production zero-constant invariant."""
from fractions import Fraction
from pathlib import Path
import argparse
import inspect
import json
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch
    from flowstar_gpu.sparse_exec import _g_precond
    from flowstar_gpu import cuda_kernels as ck
    assert ck.available();torch.set_num_threads(1)
    rows=[]
    for device in ['cpu','cuda']:
        for a,e in [(.1,0.),(1.1,2.**-52)]:
            t=lambda v:torch.tensor(v,dtype=torch.float64,device=device)
            coefficients=t([[[0.,a,0.],[0.,0.,a]]])
            remainder=t([[[-e,e],[-e,e]]])
            scaled,rem,S,point,inv,x0=_g_precond(
                coefficients,remainder,t([[1.,1.],[-1.,1.],[-1.,1.]]),t([[0.,0.]]),
                torch.tensor([1,2],device=device),torch.arange(2,device=device),3)
            exact_lo,exact_hi=-Fraction(a)-Fraction(e),Fraction(a)+Fraction(e)
            for dim in range(2):
                c=Fraction(scaled[0,dim,dim+1].item());s=Fraction(S[0,dim].item())
                l,h=map(Fraction,rem[0,dim].tolist())
                lo,hi=s*(-abs(c)+l),s*(abs(c)+h)
                rows.append({'device':device,'state':dim,'a':a,'remainder_radius':e,
                    'original_constant_coefficient':0.,'original_remainder':[-e,e],
                    'domain':[[-1.,1.],[-1.,1.]],'state_dimension':2,
                    'S':float(s),'inverse':inv[0,dim].item(),'scaled_linear_coefficient':float(c),
                    'scaled_constant_coefficient':scaled[0,dim,0].item(),'scaled_remainder':rem[0,dim].tolist(),
                    'exact_original_lower':str(exact_lo),'exact_original_upper':str(exact_hi),
                    'exact_reconstructed_lower':str(lo),'exact_reconstructed_upper':str(hi),
                    'lower_contained':lo<=exact_lo,'upper_contained':hi>=exact_hi,
                    'whole_set_contained':lo<=exact_lo and hi>=exact_hi,
                    'upper_bound_error':str(hi-exact_hi)})
    fn=Path(inspect.getsourcefile(_g_precond)).resolve()
    result={'schema':'xiangru_centered_preconditioning_exact/1','source_function':str(fn),
        'source_sha':subprocess.check_output(['git','-C',str(fn.parents[2]),'rev-parse','HEAD'],text=True).strip(),
        'production_invariants':['Two state components','Zero polynomial constant in both components',
            'Independent normalized spatial variables in [-1,1]','Symmetric remainder containing zero',
            'Finite normal scales; no history matrix is used'],
        'claim':'Whole image set inclusion for centered affine maps; not a claim these operands occur on the frozen trajectories.',
        'rows':rows}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ['device','state','a','remainder_radius','whole_set_contained','upper_bound_error']} for r in rows],indent=2))


if __name__=='__main__':main()
