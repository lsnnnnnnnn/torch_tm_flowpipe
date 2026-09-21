from pathlib import Path
import sys,json,torch,yaml,traceback,gc,hashlib
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');sys.path.insert(0,str(N/'repo/experiments/flowstar_acceleration'));from nncs_screen import load_driver
R=N/'runs/nncs_full_20260922/tora_homogeneous_diagnostic';d=load_driver(N/'engine');cfg_path=N/'repo/experiments/flowstar_acceleration/configs_nncs/tora_homogeneous_full.yaml';cfg=yaml.safe_load(cfg_path.read_text());cfg['_config_dir']=cfg_path.parent
from auto_LiRPA import BoundedModule
torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
result={'inputs':[{'path':str(R/f'crown_input_{j}.pt'),'sha256':hashlib.sha256((R/f'crown_input_{j}.pt').read_bytes()).hexdigest()} for j in [1,2,3]],'modes':{}}
answers={}
for mode in ['default','matrix']:
 model=d.build_crown(cfg,'cuda:0','same-slope') if mode=='default' else BoundedModule(d.build_raw_net(cfg),torch.zeros(1,1,1,4),device='cuda:0',bound_opts={'activation_bound_option':'same-slope','conv_mode':'matrix'})
 rows=[];answers[mode]=[]
 for j in [1,2,3]:
  inputs=torch.load(R/f'crown_input_{j}.pt',weights_only=True);lo,hi=inputs['lower'].cuda(),inputs['upper'].cuda();row={'call':j,'lower_shape':list(lo.shape),'upper_shape':list(hi.shape),'B':lo.shape[0]}
  try:
   out=d.crown_bounds(model,cfg,lo,hi);torch.cuda.synchronize();assert all(torch.isfinite(x).all() for x in out)
   row.update(status='passed',output_shapes=[list(x.shape) for x in out]);answers[mode].append([x.detach().cpu() for x in out])
  except Exception as exc:row.update(status='error',exception=repr(exc));traceback.print_exc()
  rows.append(row)
  if row['status']=='error':break
 result['modes'][mode]={'actual_conv_mode':model.bound_opts.get('conv_mode'),'calls':rows}
 del model;gc.collect();torch.cuda.empty_cache()
result['same_inputs_first_two_output_max_absolute_differences']=[[float((x-y).abs().max()) for x,y in zip(a,b)] for a,b in zip(answers['default'],answers['matrix'])]
assert result['modes']['default']['calls'][-1]['status']=='error'
assert len(result['modes']['matrix']['calls'])==3 and all(x['status']=='passed' for x in result['modes']['matrix']['calls'])
Path('tora_replay.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(answers,'controller_outputs.pt');print(json.dumps(result,indent=2))
