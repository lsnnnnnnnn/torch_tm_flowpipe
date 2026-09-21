"""Independent exact rational oracle for sum_q Phi[q] @ J[q]."""
from fractions import Fraction as F
from pathlib import Path
import sys,random,math,json,hashlib,statistics,time,gc
import torch
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');sys.path[:0]=[str(N/'engine_endpoint_graph/src')]
from flowstar_gpu import sr_sum_kernels as sk
from flowstar_gpu.flowpipe import _lane_nonfinite
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
assert sk.available();rng=random.Random(993174);checked=0;terms=0;tests={}

def intervals(shape):
 values=[]
 for _ in range(math.prod(shape)):
  a=math.ldexp(rng.uniform(-1,1),rng.randint(-40,40));b=math.ldexp(rng.uniform(-1,1),rng.randint(-40,40));values.append([min(a,b),max(a,b)])
 return torch.tensor(values).reshape(*shape,2)

def oracle(phi,j,b,i):
 lo=F(0);hi=F(0)
 for q in range(phi.shape[0]):
  for k in range(phi.shape[3]):
   products=[F(float(x))*F(float(y)) for x in phi[q,b,i,k] for y in j[q,b,k]]
   lo+=min(products);hi+=max(products)
 return lo,hi

def check(phi,j,out):
 global checked,terms
 assert out.shape==(phi.shape[1],phi.shape[2],2)
 for b in range(phi.shape[1]):
  for i in range(phi.shape[2]):
   lo,hi=oracle(phi,j,b,i);lower,upper=[float(v) for v in out[b,i]]
   assert not (math.isnan(lower) or math.isnan(upper)),(b,i,lower,upper)
   assert lower<=upper and (lower==-math.inf or F(lower)<=lo) and (upper==math.inf or hi<=F(upper)),(b,i,lower,upper,str(lo),str(hi))
   checked+=1;terms+=phi.shape[0]*phi.shape[2]

def apply(phi,j):
 pg=phi.cuda();jg=j.cuda();p_before=pg.clone();j_before=jg.clone();assert sk.supported(pg,jg)
 out=sk.sum_history(pg,jg);torch.cuda.synchronize();assert torch.equal(pg,p_before) and torch.equal(jg,j_before)
 return out.cpu()

for q,b,n in [(0,3,4),(1,3,1),(1,3,3),(20,3,1),(20,3,7),(1000,3,1),(1000,2,3),(1,2,31),(1,2,32)]:
 phi=intervals((q,b,n,n));j=intervals((q,b,n));out=apply(phi,j);check(phi,j,out);tests[f'Q{q}_B{b}_n{n}']={'all_exact_fraction_outputs_enclosed':True}
# Identity over20 queue entries covers every row at the n limit and batch isolation.
phi=torch.eye(32).expand(20,3,32,32).clone().unsqueeze(-1).expand(20,3,32,32,2).contiguous();j=torch.ones(20,3,32,2)
for b in range(3):j[:,b]*=b+1
out=apply(phi,j);assert torch.equal(out,torch.tensor([20.,40.,60.]).view(3,1,1).expand(3,32,2));tests['identity_n32_batch_isolation']=True
# Signed cancellation, exact zero, half-subnormal contributions and finite overflow.
tiny=math.ldexp(1.,-1074)
phi=torch.tensor([[[[[tiny,tiny],[tiny,tiny]],[[1e150,1e150],[-1e150,-1e150]]]],[[[[tiny,tiny],[tiny,tiny]],[[1e150,1e150],[-1e150,-1e150]]]]])
j=torch.full((2,1,2,2),.5);out=apply(phi,j);check(phi,j,out);assert out[0,0,1]>0.;tests['cancellation_subnormal']={'upper':float(out[0,0,1])}
phi=torch.zeros(20,2,3,3,2);j=intervals((20,2,3));out=apply(phi,j);check(phi,j,out);tests['exact_zero']=True
phi=torch.full((1,1,1,1,2),1e308);j=torch.full((1,1,1,2),2.);out=apply(phi,j);check(phi,j,out);assert _lane_nonfinite(out).tolist()==[True];tests['finite_input_overflow_visible_rejection']=True
# Invalid source values or reversed endpoints cannot disappear through fmin/fmax.
for tensor_name in ['phi','j']:
 for value in [math.nan,math.inf,-math.inf,'reversed']:
  phi=torch.ones(20,2,3,3,2,device='cuda');j=torch.ones(20,2,3,2,device='cuda')
  target=phi if tensor_name=='phi' else j;index=(5,0,1,1) if tensor_name=='phi' else (5,0,1)
  if value=='reversed':target[index]=torch.tensor([2.,1.],device='cuda')
  else:target[index+(0,)]=value
  out=sk.sum_history(phi,j);torch.cuda.synchronize();assert _lane_nonfinite(out).tolist()==[True,False]
  assert torch.equal(out[1],torch.full((3,2),60.,device='cuda'))
tests['nonfinite_reversed_inputs_and_lane_gate']=True
# Repeated calls, storage offset views, nondefault stream and read-only alias.
phi=intervals((22,2,3,3));j=intervals((22,2,3));pg=phi.cuda();jg=j.cuda();torch.cuda.synchronize();stream=torch.cuda.Stream()
with torch.cuda.stream(stream):
 out=sk.sum_history(pg[1:21],jg[1:21]);again=sk.sum_history(pg[1:21],jg[1:21])
stream.synchronize();check(phi[1:21],j[1:21],out.cpu());assert torch.equal(out,again) and torch.equal(pg.cpu(),phi) and torch.equal(jg.cpu(),j)
p=intervals((20,2,1,1)).cuda();alias=p.view(20,2,1,2);assert sk.supported(p,alias);check(p.cpu(),alias.cpu(),sk.sum_history(p,alias).cpu())
tests['stream_offset_repeat_readonly_alias']=True
p=intervals((2,2,3,3)).cuda();j=intervals((2,2,3)).cuda()
assert not sk.supported(p.cpu(),j.cpu()) and not sk.supported(p.float(),j.float()) and not sk.supported(p.transpose(2,3),j) and not sk.supported(p,j.transpose(1,2))
assert not sk.supported(torch.empty((1,1,33,33,2),device='cuda'),torch.empty((1,1,33,2),device='cuda'))
try:sk.sum_history(p.transpose(2,3),j)
except ValueError:pass
else:raise AssertionError('unsupported layout accepted')
tests['metadata_fallback_gate']=True
# QUAD realistic memory scales, same one extra linear-Q work tensor.
bench=[]
for Q in [20,1000]:
 B,n=1024,16;gc.collect();torch.cuda.empty_cache()
 phi=torch.full((Q,B,n,n,2),.001,device='cuda');j=torch.full((Q,B,n,2),.01,device='cuda')
 out=sk.sum_history(phi,j);torch.cuda.synchronize();del out;gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 base_alloc=torch.cuda.memory_allocated();base_reserved=torch.cuda.memory_reserved();events=[];wall=time.perf_counter()
 for _ in range(10):
  begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();out=sk.sum_history(phi,j);end.record();events.append((begin,end));del out
 torch.cuda.synchronize();wall=time.perf_counter()-wall;samples=[a.elapsed_time(b) for a,b in events]
 peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
 verify=sk.sum_history(phi,j);torch.cuda.synchronize();expected=F(.001)*F(.01)*Q*n
 for b,i in [(0,0),(1023,15),(512,7)]:
  low,high=[float(x) for x in verify[b,i]];assert F(low)<=expected<=F(high)
 bench.append({'Q':Q,'B':B,'n':n,'workspace_bytes':Q*B*n*16,'output_bytes':B*n*16,'old_single_product_bytes':Q*B*n*n*16,'baseline_allocated_bytes':base_alloc,'peak_allocated_bytes':peak,'incremental_peak_allocated_bytes':peak-base_alloc,'baseline_reserved_bytes':base_reserved,'peak_reserved_bytes':reserved,'cuda_ms_median':statistics.median(samples),'cuda_ms_min':min(samples),'cuda_ms_max':max(samples),'iterations':10,'loop_wall_s_including_allocations_events':wall,'measurement_excludes_subsequent_oracle_observer':True})
 del phi,j,verify;gc.collect();torch.cuda.empty_cache()
report={'status':'passed','exact_fraction_output_count':checked,'exact_fraction_product_terms':terms,'tests':tests,'quad_benchmarks':bench,'source_sha256':hashlib.sha256(Path(sk.__file__).read_bytes()).hexdigest(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'integration_status':'root-integrated 5d1bac0; same Fraction oracle and shape/edge coverage as prototype','speed_claim':'standalone operator timings only'}
Path('validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
