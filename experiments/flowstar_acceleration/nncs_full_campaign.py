"""Sequential original-horizon screens, preserving each original YAML contract."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--engine',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');D=N/'repo/experiments/flowstar_acceleration/configs_nncs';S=N/'repo/experiments/flowstar_acceleration/nncs_candidate.py';W=N/'runs/nncs_candidate_20260922/nncs_watchdog.py';V=N/'runs/nncs_candidate_20260922/verify_candidate_trace.py'
a.output.mkdir(parents=True,exist_ok=True)
sequence=[('acc',300),('single_pendulum',300),('tora_homogeneous',300),('attitude_control',300),('cartpole',600),('double_pendulum_less_robust',600),('double_pendulum_more_robust',600),('tora_relu_tanh',600),('tora_sigmoid',600),('unicycle',600),('nav_robust',600),('nav_standard',900)]
summary={'engine':str(a.engine),'engine_sha_at_start':subprocess.check_output(['git','-C',str(a.engine),'rev-parse','HEAD'],text=True).strip(),'controller_strict_qualified':False,'configuration_scope':'original full horizon; original box/B/h/order/controller preserved; explicit horner+graph candidate engine','environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','FLOWSTAR_COMPOSITION','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCH_EXTENSIONS_DIR']},'cases':[{'id':n,'timeout_s':cap,'status':'pending','config':str(D/(n+'_full.yaml'))} for n,cap in sequence]}
def save():
 (a.output/'campaign.json').write_text(json.dumps(summary,indent=2)+'\n')
save()
for row in summary['cases']:
 name=row['id'];out=a.output/name;row['status']='running';save();print('START',name,flush=True)
 command=[sys.executable,str(W),'--output',str(out),'--timeout',str(row['timeout_s']),'--',sys.executable,str(S),'--engine',str(a.engine),'--output',str(out),'--benchmark',name,'--config',row['config'],'--scope','full']
 subprocess.run(command,check=True)
 row['process']=json.loads((out/'process.json').read_text())
 result=out/(name+'_result.json')
 if result.exists():
  data=json.loads(result.read_text());row['status']=data['status'];row['result_path']=str(result)
  for key in ['expected_steps','attempted_steps','accepted_lane_steps','accepted_steps_all_lanes','configured_horizon_completed','sr_kernel_dispatches','timing','engine_sha','engine_sha_at_end','error']:
   if key in data:row[key]=data[key]
  with (out/'trace_verification.log').open('w') as log:
   checked=subprocess.run([sys.executable,str(V),str(out),name],stdout=log,stderr=subprocess.STDOUT)
  row['trace_verification_returncode']=checked.returncode
 else:
  row['status']=row['process']['status'];progress=out/'progress.json'
  if progress.exists():row['last_progress']=json.loads(progress.read_text())
 metrics=out/(name+'_metrics.json')
 if metrics.exists():
  met=json.loads(metrics.read_text());row['driver_broken']=met.get('broken');row['driver_elapsed_s']=met.get('elapsed_s');row['final_hull']=met.get('final_hull')
 stdout=(out/'stdout.log').read_text()
 row['driver_protocol_lines']=[x for x in stdout.splitlines() if x.strip() in ['VERIFIED','FALSIFIED','UNKNOWN','Unknown.','Unsafe.','Flow* terminated.'] or x.startswith('Broken branch:')]
 save();print('FINISH',name,row['status'],row.get('accepted_lane_steps'),flush=True)
summary['completed']=True;save()
