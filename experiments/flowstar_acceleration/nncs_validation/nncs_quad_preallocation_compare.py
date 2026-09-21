from pathlib import Path
import torch,json,hashlib,time
R=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/nncs_quad_memory_20260922');torch.set_num_threads(1);rows=[];start=time.perf_counter();count=elements=0
for step in [1,2,16,17,20]:
 a=torch.load(R/f'lazy_period1_state/snapshots/step{step:04d}.pt',map_location='cpu',weights_only=True);b=torch.load(R/f'preallocated_period1_state/snapshots/step{step:04d}.pt',map_location='cpu',weights_only=True)
 assert a['pre_support']==b['pre_support'] and a['tmv_support']==b['tmv_support']
 for group in ['state','sr']:
  for key,x in a[group].items():
   y=b[group][key]
   if key=='capacity':continue
   if isinstance(x,torch.Tensor):
    assert x.shape==y.shape and x.dtype==y.dtype
    assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)),(step,group,key)
    count+=1;elements+=x.numel()
   else:assert x==y,(step,group,key)
 rows.append({'step':step,'lazy_capacity':a['sr']['capacity'],'preallocated_capacity':b['sr']['capacity'],'qlen':a['sr']['qlen'],'jlen':a['sr']['jlen'],'all_active_state_and_history_bytes_identical':True})
result={'status':'passed','checkpoints':rows,'tensor_comparisons':count,'tensor_elements_compared':elements,'method':'direct equality of contiguous uint8 views, including signed-zero bit patterns; support and all scalar metadata equal except explicitly compared capacities','wall_s':time.perf_counter()-start,'qualification':'real QUAD B1024/order2/h.005 same original control period; controller remains unqualified'}
(R/'preallocation_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
