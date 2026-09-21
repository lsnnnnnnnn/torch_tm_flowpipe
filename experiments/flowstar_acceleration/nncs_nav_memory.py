"""Read allocator/pool state through step200, then one explicit empty-cache diagnostic."""
from pathlib import Path
import json,sys,runpy,torch
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');sys.path.insert(0,str(N/'repo/experiments/flowstar_acceleration'));import nncs_screen
original=nncs_screen.load_driver;out=Path(sys.argv[sys.argv.index('--output')+1]);step=0;records=[]
class DiagnosticComplete(RuntimeError):pass

def snapshot(label,eng):
 torch.cuda.synchronize();snap=torch.cuda.memory_snapshot();stats=torch.cuda.memory_stats();pools={}
 for seg in snap:
  key=str(seg.get('segment_pool_id','unknown'));p=pools.setdefault(key,{'segments':0,'total_size':0,'allocated_size':0,'active_size':0,'streams':[],'block_bytes':{}})
  p['segments']+=1
  for k in ['total_size','allocated_size','active_size']:p[k]+=seg.get(k,0)
  if seg.get('stream') not in p['streams']:p['streams'].append(seg.get('stream'))
  for block in seg['blocks']:p['block_bytes'][block['state']]=p['block_bytes'].get(block['state'],0)+block['size']
 g=eng._graphs
 record={'label':label,'step':step,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'stats':{k:v for k,v in stats.items() if k.endswith('.current')},'graph':{'hits':g.hits,'captures':g.captures,'key_reprs':[repr(k) for k in g._segs]},'pools':pools}
 (out/(label+'_snapshot.json')).write_text(json.dumps(snap,indent=2)+'\n');records.append(record);(out/'memory_diagnostic.json').write_text(json.dumps(records,indent=2)+'\n')
 return record

def load(engine):
 d=original(engine);advance=d.advance_sparse
 def observed(*args,**kw):
  global step
  state,ok=advance(*args,**kw);step+=1
  if step%20==0:snapshot(f'step{step}',args[2])
  if step==200:
   before=records[-1];torch.cuda.empty_cache();after=snapshot('step200_after_empty_cache',args[2])
   (out/'empty_cache_diagnostic.json').write_text(json.dumps({'status':'diagnostic_complete','accepted_final_step':ok.cpu().tolist(),'before_reserved':before['reserved'],'after_reserved':after['reserved'],'released_bytes':before['reserved']-after['reserved'],'scope':'single diagnostic empty_cache after step200; no full run or speed comparison'},indent=2)+'\n')
   raise DiagnosticComplete('stop after memory snapshot and one empty_cache diagnostic at step200')
  return state,ok
 d.advance_sparse=observed;return d
nncs_screen.load_driver=load
runpy.run_path(str(N/'repo/experiments/flowstar_acceleration/nncs_candidate.py'),run_name='__main__')
