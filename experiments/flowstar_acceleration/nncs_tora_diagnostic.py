"""Record actual repeated CROWN inputs and model node layout without altering them."""
import json,sys,runpy,torch
from pathlib import Path
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');sys.path.insert(0,str(N/'repo/experiments/flowstar_acceleration'))
import nncs_screen
original=nncs_screen.load_driver
output=Path(sys.argv[sys.argv.index('--output')+1]);calls=0

def layout(model):
 rows=[]
 for name,node in model.named_modules():
  if not name:continue
  row={'name':name,'type':type(node).__name__}
  for key in ['output_shape','input_shape','batch_dim','perturbed','from_input','value','lower','upper','interval','lA','uA','linear','patches','alpha','_ori_beta','cached_bounds']:
   if hasattr(node,key):
    val=getattr(node,key)
    if isinstance(val,torch.Tensor):row[key]={'shape':list(val.shape),'dtype':str(val.dtype),'device':str(val.device)}
    elif isinstance(val,(tuple,list)):row[key]=[{'shape':list(x.shape)} if isinstance(x,torch.Tensor) else str(x)[:150] for x in val]
    elif isinstance(val,dict):row[key]={'keys':list(val)}
    elif isinstance(val,(str,int,float,bool,type(None))):row[key]=val
  rows.append(row)
 return rows

def load(engine):
 d=original(engine);crown=d.crown_bounds
 def wrapped(model,cfg,lower,upper):
  global calls
  calls+=1;output.mkdir(parents=True,exist_ok=True)
  torch.save({'lower':lower.detach().cpu(),'upper':upper.detach().cpu()},output/f'crown_input_{calls}.pt')
  report={'call':calls,'lower_shape':list(lower.shape),'upper_shape':list(upper.shape),'input_shape_config':cfg['input_shape'],'before':layout(model),'bound_opts':str(getattr(model,'bound_opts',None))}
  try:
   ans=crown(model,cfg,lower,upper);report['outputs']=[list(x.shape) for x in ans];return ans
  except Exception as exc:report['exception']=repr(exc);raise
  finally:
   report['after']=layout(model);(output/f'crown_layout_{calls}.json').write_text(json.dumps(report,indent=2)+'\n')
 d.crown_bounds=wrapped;return d
nncs_screen.load_driver=load
runpy.run_path(str(N/'repo/experiments/flowstar_acceleration/nncs_candidate.py'),run_name='__main__')
