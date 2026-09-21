"""Bound inactive allocator caching, preserving live tensors and private graphs."""
from pathlib import Path
import json,sys,runpy,time,torch
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');sys.path.insert(0,str(N/'repo/experiments/flowstar_acceleration'));import nncs_screen
original=nncs_screen.load_driver;out=Path(sys.argv[sys.argv.index('--output')+1]);step=0
report={'policy':'after successful step, if reserved minus allocated exceeds2GiB, empty_cache; do not touch live tensors or graph objects','threshold_bytes':2*2**30,'cleanup_time_included_in_outer_plant_advance_and_process_wall':True,'events':[],'total_cleanup_s':0.}

def load(engine):
 d=original(engine);advance=d.advance_sparse
 def observed(*args,**kw):
  global step
  state,ok=advance(*args,**kw);step+=1
  allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
  if bool(ok.all()) and reserved-allocated>2*2**30:
   g=args[2]._graphs;old_captures=g.captures;old_segments=len(g._segs)
   torch.cuda.synchronize();t=time.perf_counter();torch.cuda.empty_cache();torch.cuda.synchronize();elapsed=time.perf_counter()-t
   after_allocated=torch.cuda.memory_allocated();after_reserved=torch.cuda.memory_reserved()
   assert after_allocated==allocated and g.captures==old_captures and len(g._segs)==old_segments
   report['events'].append({'ode_step':step,'allocated_before':allocated,'allocated_after':after_allocated,'reserved_before':reserved,'reserved_after':after_reserved,'elapsed_s':elapsed,'graph_captures_unchanged':old_captures,'graph_segments_unchanged':old_segments})
   report['total_cleanup_s']+=elapsed
   (out/'cache_cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
  return state,ok
 d.advance_sparse=observed;return d
nncs_screen.load_driver=load
runpy.run_path(str(N/'repo/experiments/flowstar_acceleration/nncs_candidate.py'),run_name='__main__')
