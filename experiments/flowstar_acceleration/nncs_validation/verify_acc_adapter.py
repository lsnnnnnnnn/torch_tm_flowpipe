from pathlib import Path
from fractions import Fraction as F
import sys,random,math,json,hashlib
import torch,yaml
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');sys.path[:0]=[str(N/'engine/src'),str(N/'repo/experiments/flowstar_acceleration')]
from nncs_acc_adapter import feature_tms,inject_tms,install
from nncs_screen import load_driver

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64)
rng=random.Random(418412);B,S=4,13;feature_checks=0;injection_checks=0

def contains(poly,rem,exact,monomial):
 value=sum(F(float(x))*m for x,m in zip(poly,monomial))
 assert value+F(float(rem[0]))<=exact<=value+F(float(rem[1])),(value,rem,exact)

for case in range(20):
 pre=torch.tensor([math.ldexp(rng.uniform(-1,1),rng.randint(-70,70)) for _ in range(B*8*S)]).reshape(B,8,S)
 if case==0:pre.fill_(math.ldexp(1.,-1074))
 if case==1:pre[:,3]=pre[:,0];pre[:,4]=pre[:,1] # exact cancellation
 rem=torch.tensor([[[-1e-11,1e-11] for _ in range(8)] for _ in range(B)])
 snapshot=pre.clone();f,fr,feature_error=feature_tms(pre,rem);assert torch.equal(pre,snapshot)
 T=torch.tensor([math.ldexp(rng.uniform(-2,2),rng.randint(-20,20)) for _ in range(B*5)]).reshape(B,1,5)
 lo=torch.tensor([[rng.uniform(-1,0)] for _ in range(B)]);hi=torch.tensor([[rng.uniform(0,1)] for _ in range(B)])
 rows,rems,injection_error=inject_tms(f,fr,T,lo,hi)
 for sample in range(10):
  mon=[F(1)]+[F(rng.randrange(-16,17),16) for _ in range(S-1)]
  for b in range(B):
   x=[]
   for i in range(8):
    endpoint=sample%2
    x.append(sum(F(float(co))*v for co,v in zip(pre[b,i],mon))+F(float(rem[b,i,endpoint])))
   exact=[F(30),F(1.4),x[4],x[0]-x[3],x[1]-x[4]]
   for j in range(5):contains(f[b,j],fr[b,j],exact[j],mon);feature_checks+=1
   for bias in [lo[b,0],hi[b,0]]:
    target=sum(F(float(T[b,0,j]))*exact[j] for j in range(5))+F(float(bias))
    contains(rows[b,0],rems[b,0],target,mon);injection_checks+=1
# Known initial feature box values and ONNX/LiRPA CPU input/output shape.
cfg=yaml.safe_load((N/'repo/experiments/flowstar_acceleration/configs_nncs/acc_period1.yaml').read_text());cfg['_config_dir']=N/'repo/experiments/flowstar_acceleration/configs_nncs';d=load_driver(N/'engine');state=install(d,cfg)
model=d.build_crown(cfg,'cpu','same-slope')
lb=torch.tensor([[30.,1.4,30.,79.,1.8]]);ub=torch.tensor([[30.,1.4,30.2,100.,2.2]])
T,lo,hi=d.crown_bounds(model,cfg,lb,ub)
assert T.shape==(1,1,5) and lo.shape==hi.shape==(1,1) and torch.isfinite(T).all() and torch.isfinite(lo).all() and torch.isfinite(hi).all()
report={'status':'passed','exact_fraction_feature_evaluations':feature_checks,'exact_fraction_mock_bias_injections':injection_checks,'random_cancellation_subnormal_cases':20,'batch':B,'polynomial_coefficients':S,'samples_per_case':10,'input_unchanged':True,'real_controller_cpu_shape':{'input':[1,1,1,5],'slope':list(T.shape),'bias':list(lo.shape)},'bound_opts':{'activation_bound_option':'same-slope','conv_mode':'matrix'},'source_sha256':hashlib.sha256((N/'repo/experiments/flowstar_acceleration/nncs_acc_adapter.py').read_bytes()).hexdigest(),'qualification':'Only adapter affine arithmetic was checked against exact rationals. The real CROWN call establishes layout and dependency compatibility, not CROWN strictness.'}
Path('acc_adapter_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
