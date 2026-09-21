"""Test actual strict SR dispatch and queue/reset transitions with exact rational oracles."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math,random,sys,torch
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z')
sys.path.insert(0,str(N/'engine_endpoint_graph/src'))
from flowstar_gpu import symbolic_remainder as sr, sr_kernels as sk, sr_sum_kernels as sj
from flowstar_gpu.flowpipe import _lane_nonfinite

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64);torch.cuda.set_device(0)
assert sk.available() and sj.available()
rng=random.Random(443);calls=[];products=0;jsums=0;resets=0;observations=[]
original=sk.left_multiply_
def counted(a,h):
 calls.append({'Q':h.shape[0],'B':h.shape[1],'n':h.shape[2]})
 return original(a,h)
sk.left_multiply_=counted
sum_calls=[];sum_original=sj.sum_history
def sum_counted(p,j):
 sum_calls.append({'Q':p.shape[0],'B':p.shape[1],'n':p.shape[2]});return sum_original(p,j)
sj.sum_history=sum_counted

def interval(shape):
 x=torch.tensor([[rng.uniform(-.3,.3),rng.uniform(-.3,.3)] for _ in range(math.prod(shape))]).reshape(*shape,2)
 return torch.sort(x,dim=-1).values

def exact_dot(xs,ys):
 lo=F(0);hi=F(0)
 for x,y in zip(xs,ys):
  p=[F(float(a))*F(float(b)) for a in x for b in y]
  lo+=min(p);hi+=max(p)
 return lo,hi

def encloses(got,want):
 assert math.isfinite(float(got[0])) and math.isfinite(float(got[1]))
 assert F(float(got[0]))<=want[0]<=want[1]<=F(float(got[1])),(got,want)

for B,n,maximum,steps in [(2,1,3,11),(3,3,3,11),(2,3,20,43)]:
 state=sr.make_symbolic_remainder(B,n,maximum,'cuda');assert state.phi_buf.shape[0]==min(16,maximum)
 for step in range(steps):
  a=interval((B,n,n));ag=a.cuda();oldq=state.qlen;old=state.phi_iv_buf[:oldq].cpu().clone() if state.phi_iv_buf is not None else None
  before=len(calls);before_sum=len(sum_calls);out=sr.propagate(state,ag.mean(-1),strict=True,phi_i_iv=ag).cpu()
  assert len(calls)-before==int(oldq>1)
  assert len(sum_calls)-before_sum==int(oldq>0)
  cur=state.phi_iv_buf[:state.qlen].cpu();assert torch.equal(cur[-1],a)
  if oldq:
   assert torch.equal(cur[0],old[0]) # dead Phi[0] is never modified
  for q in range(1,oldq):
   for b in range(B):
    for i in range(n):
     for j in range(n):
      encloses(cur[q,b,i,j],exact_dot(a[b,i],old[q,b,:,j]));products+=1
  savedj=state.j_buf[:oldq].cpu()
  for b in range(B):
   for i in range(n):
    xs=[];ys=[]
    for q in range(oldq):
     xs.extend(cur[q+1,b,i]);ys.extend(savedj[q,b])
    encloses(out[b,i],exact_dot(xs,ys));jsums+=1
  assert state.queue_len==oldq and state.qlen==oldq+1
  state.append_j(interval((B,n)).cuda());assert state.queue_len==oldq+1
  beforecap=state.phi_buf.shape[0];reset=state.reset_if_full()
  assert reset==((oldq+1)>=maximum)
  if reset:
   resets+=1;assert state.qlen==state.jlen==0 and torch.equal(state.scalars,torch.ones_like(state.scalars));assert state.phi_buf.shape[0]==beforecap
  observations.append({'n':n,'step':step+1,'old_Q':oldq,'capacity':state.phi_buf.shape[0],'reset':reset})
# Real dispatch falls back for unsupported non-contiguous left matrices.
state=sr.make_symbolic_remainder(2,3,4,'cuda')
for _ in range(2):
 a=interval((2,3,3)).cuda();sr.propagate(state,a.mean(-1),strict=True,phi_i_iv=a);state.append_j(interval((2,3)).cuda())
a=interval((2,3,3)).cuda().transpose(1,2);assert not a.is_contiguous();before=len(calls)
out=sr.propagate(state,a.mean(-1),strict=True,phi_i_iv=a);assert len(calls)==before and torch.isfinite(out).all()
# Invalid lane remains rejected after actual kernel dispatch; distinct lane stays finite.
state=sr.make_symbolic_remainder(2,2,4,'cuda')
for _ in range(2):
 a=torch.ones((2,2,2,2),device='cuda');sr.propagate(state,a.mean(-1),strict=True,phi_i_iv=a);state.append_j(torch.ones((2,2,2),device='cuda'))
a=torch.ones((2,2,2,2),device='cuda');a[0,0,0,0]=float('nan');out=sr.propagate(state,torch.ones((2,2,2),device='cuda'),strict=True,phi_i_iv=a)
assert _lane_nonfinite(out).tolist()==[True,False]
result={'status':'passed','actual_kernel_dispatches':len(calls),'actual_sum_dispatches':len(sum_calls),'sum_shapes':sum_calls,'dispatch_shapes':calls,'exact_fraction_matrix_dots':products,'exact_fraction_history_sums':jsums,'reset_count':resets,'capacity_16_to_20_exercised':any(x['capacity']==20 for x in observations),'observations':observations,'noncontiguous_fallback_checked':True,'invalid_lane_rejection_preserved':True,'files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(sr.__file__),Path(sk.__file__),Path(sj.__file__)]}}
Path('integrated_oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['dispatch_shapes','observations']},indent=2))
