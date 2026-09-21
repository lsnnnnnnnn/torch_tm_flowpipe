from pathlib import Path
from fractions import Fraction as F
import importlib.util,sys,torch,json,random,time
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');sys.path.insert(0,str(N/'engine_sr_memory/src'))
from flowstar_gpu import symbolic_remainder as new,sr_kernels,sr_sum_kernels
spec=importlib.util.spec_from_file_location('flowstar_gpu.symbolic_remainder_reference',N/'engine_endpoint_graph/src/flowstar_gpu/symbolic_remainder.py');ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
assert sr_kernels.available() and sr_sum_kernels.available();torch.manual_seed(2388)
Q,B,n=37,1024,16
phi=torch.rand(Q,B,n,n,device='cuda')*.02-.01;j=torch.rand(Q,B,n,2,device='cuda')*.001;j=torch.sort(j,dim=-1).values;phiv=torch.stack((phi-1e-8,phi+1e-8),-1);left=torch.rand(B,n,n,device='cuda')*.02-.01;leftiv=torch.stack((left-1e-8,left+1e-8),-1)
def state(mod):
 s=mod.make_symbolic_remainder(B,n,40,'cuda');s.phi_buf=torch.zeros(40,B,n,n,device='cuda');s.j_buf=torch.zeros(40,B,n,2,device='cuda');s.phi_iv_buf=torch.zeros(40,B,n,n,2,device='cuda');s.phi_buf[:Q]=phi;s.j_buf[:Q]=j;s.phi_iv_buf[:Q]=phiv;s.qlen=s.jlen=Q;return s
sref=state(ref);snew=state(new);calls=[];ein=torch.einsum
def counted(expr,*args,**kw):
 if expr=='bij,qbjk->qbik':calls.append(args[1].shape[0])
 return ein(expr,*args,**kw)
torch.einsum=counted
want=ref.propagate(sref,left,strict=True,phi_i_iv=leftiv);refcalls=calls[:];calls.clear();got=new.propagate(snew,left,strict=True,phi_i_iv=leftiv);newcalls=calls[:];torch.einsum=ein;torch.cuda.synchronize()
assert refcalls==[36] and newcalls==[32,4]
point_equal=torch.equal(sref.phi_buf,snew.phi_buf);point_diff=float((sref.phi_buf-snew.phi_buf).abs().max());assert point_equal
assert torch.equal(want,got) and torch.equal(sref.phi_iv_buf,snew.phi_iv_buf) and torch.equal(sref.j_buf,snew.j_buf) and snew.qlen==sref.qlen==38
cpu_phi=snew.phi_iv_buf[1:38].cpu();cpu_j=j.cpu();cpu_got=got.cpu();checks=0
for b,i in [(0,0),(0,15),(511,7),(1023,0),(1023,15)]:
 low=F(0);high=F(0)
 for q in range(Q):
  for k in range(n):
   terms=[F(float(x))*F(float(y)) for x in cpu_phi[q,b,i,k] for y in cpu_j[q,b,k]];low+=min(terms);high+=max(terms)
 assert F(float(cpu_got[b,i,0]))<=low<=high<=F(float(cpu_got[b,i,1]));checks+=1
result={'status':'passed','actual_reference_einsum_Qs':refcalls,'actual_candidate_einsum_Qs':newcalls,'point_history_bitwise_equal':point_equal,'point_max_abs_diff':point_diff,'strict_result_and_interval_history_bitwise_equal':True,'Q':Q,'B':B,'n':n,'exact_fraction_output_checks':checks,'exact_fraction_terms':checks*Q*n,'queue_and_J_unchanged':True}
Path('chunk_oracle.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
