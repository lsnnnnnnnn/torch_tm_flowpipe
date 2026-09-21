from pathlib import Path
import torch,json,time,sys,gc
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');sys.path.insert(0,str(N/'engine_sr_memory/src'))
from flowstar_gpu.symbolic_remainder import make_symbolic_remainder

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
rows=[]
def snap(name,t):
 torch.cuda.synchronize();s=torch.cuda.memory_stats();segments=torch.cuda.memory_snapshot()
 row={'stage':name,'seconds':time.perf_counter()-t,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'inactive_split_bytes':s.get('inactive_split_bytes.all.current'),'inactive_bytes':sum(b['size'] for seg in segments for b in seg['blocks'] if b['state']=='inactive'),'active_awaiting_free_bytes':sum(b['size'] for seg in segments for b in seg['blocks'] if b['state']=='active_awaiting_free'),'segment_count':len(segments)}
 rows.append(row);Path('probe.json').write_text(json.dumps({'stages':rows},indent=2)+'\n');print(json.dumps(row),flush=True);torch.cuda.reset_peak_memory_stats()
q,B,n=512,1024,16;t=time.perf_counter();sr=make_symbolic_remainder(B,n,1000,'cuda');sr.phi_buf=torch.ones(q,B,n,n,device='cuda');sr.j_buf=torch.ones(q,B,n,2,device='cuda');sr.phi_iv_buf=torch.ones(q,B,n,n,2,device='cuda');sr.qlen=sr.jlen=q;snap('initial_capacity512',t)
# Exact allocation/copy order of reserve(513), without changing the engine.
for name,live in [('phi_iv_buf',sr.qlen),('phi_buf',sr.qlen),('j_buf',sr.jlen)]:
 t=time.perf_counter();buffer=getattr(sr,name);expanded=buffer.new_zeros((1000,*buffer.shape[1:]));expanded[:live].copy_(buffer[:live]);setattr(sr,name,expanded);del expanded,buffer;snap('grow_'+name,t)
left=torch.eye(n,device='cuda').expand(B,n,n).contiguous();live=sr.phi_buf[1:sr.qlen];t=time.perf_counter();live.copy_(torch.einsum('bij,qbjk->qbik',left,live));snap('full_point_einsum_Q511',t);assert bool((live==1).all())
t=time.perf_counter();torch.cuda.empty_cache();snap('diagnostic_empty_cache',t)
t=time.perf_counter()
for start in range(1,sr.qlen,32):
 live=sr.phi_buf[start:min(start+32,sr.qlen)];live.copy_(torch.einsum('bij,qbjk->qbik',left,live))
snap('chunk32_point_einsum_Q511',t);assert bool((sr.phi_buf[:sr.qlen]==1).all())
Path('probe.json').write_text(json.dumps({'stages':rows,'source':'candidate reserve largest-first allocation order copied exactly; fixed B1024 n16 capacity512->1000; no engine changes','allocator_cap_GiB':9,'all_points_unchanged_for_identity':True,'warning':'standalone synthetic storage probe, no NNCS graph/CROWN context or run-speed claim'},indent=2)+'\n')
