"""Observed candidate NNCS campaign. Strict plant; CROWN/coupling unqualified."""
from __future__ import annotations
import argparse,copy,gzip,hashlib,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
import yaml
import torch
from nncs_screen import CONTROLLER,load_driver,sha,write

p=argparse.ArgumentParser();p.add_argument('--reserve-sr-full',action='store_true');p.add_argument('--snapshot-dir',type=Path);p.add_argument('--expected-engine-sha',required=True);p.add_argument('--engine',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--benchmark',required=True);p.add_argument('--scope',choices=['period1','full'],required=True);p.add_argument('--crown-conv-mode',choices=['default','matrix'],default='default');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
assert os.environ.get('FLOWSTAR_COMPOSITION')=='horner'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
started=time.perf_counter();d=load_driver(a.engine);cfg=yaml.safe_load(a.config.read_text())
from flowstar_gpu import cuda_kernels,tape_kernels,sr_kernels,sr_sum_kernels,sparse_exec,glue
assert sparse_exec.COMPOSITION_MODE=='horner'
assert glue.GLUE_MODE=='graph' and sparse_exec.SUPPORT_POLICY=='measured'
t=time.perf_counter();availability={name:fn() for name,fn in [('interval',cuda_kernels.available),('tape',tape_kernels.available),('validation',tape_kernels.valid_available),('sr',sr_kernels.available),('sr_j_sum',sr_sum_kernels.available)]};assert all(availability.values());torch.cuda.synchronize();load_s=time.perf_counter()-t
engine_sha=subprocess.check_output(['git','-C',str(a.engine),'rev-parse','HEAD'],text=True).strip()
assert engine_sha==a.expected_engine_sha
assert not subprocess.check_output(['git','-C',str(a.engine),'status','--short'],text=True)
files=[a.engine/'integrations/crown_reach/gpu_driver.py',*sorted((a.engine/'src/flowstar_gpu').glob('*.py'))]
result={'benchmark':a.benchmark,'scope':a.scope,'strict_plant':True,'controller':CONTROLLER,'certificate':False,'device':'cuda:0','physical_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),'engine_sha':engine_sha,'engine_files_sha256':{str(x.relative_to(a.engine)):sha(x) for x in files},'runner_sha256':sha(__file__),'config':str(a.config),'config_sha256':sha(a.config),'composition':'horner','glue':'graph','support_policy':'measured','graph_required':True,'extensions':availability,'extension_binary_sha256':{group:{'binary':ext.__file__,'sha256':sha(ext.__file__)} for group,ext in [('interval',cuda_kernels._ext),('tape',tape_kernels._ext),('validation',tape_kernels._vext),('sr_phi',sr_kernels._ext),('sr_j_sum',sr_sum_kernels._ext)]},'expected_steps':round(cfg['step_size']/cfg['ode_step_size'])*cfg['steps'],'attempted_steps':0,'accepted_steps_all_lanes':0,'accepted_lane_steps':0,'all_steps_accepted':True,'sr_kernel_dispatches':0,'sr_dispatch_shapes':{},'sr_sum_dispatches':0,'sr_sum_shapes':{},'cache_cleanup':{'threshold_bytes':2*2**30,'events':[],'total_cleanup_s':0.0,'cost_included_in_plant_advance_driver_and_process_wall':True},'timing':{'extension_cached_load_s':load_s,'controller_setup_s':0.,'controller_bound_s':0.,'plant_advance_s':0.,'observer_s':0.,'trace_io_s':0.},'step_summaries':[],'trace_format':'gzip JSONL, endpoint and tube for every cell and coordinate; per-step widths in result'}
write(a.output/'started.json',{'pid':os.getpid(),'engine_sha':engine_sha,'config':str(a.config),'scope':a.scope,'expected_steps':result['expected_steps']})
trace=gzip.open(a.output/(a.benchmark+'_trace.jsonl.gz'),'wt',compresslevel=1)
# Python counters observe actual extension invocations (captures count; replay is recorded by GraphCache).
result['extension_dispatch_counts']={}
for group,extension in [('interval',cuda_kernels._ext),('tape',tape_kernels._ext),('validation',tape_kernels._vext),('sr_phi',sr_kernels._ext),('sr_j_sum',sr_sum_kernels._ext)]:
 counts=result['extension_dispatch_counts'][group]={}
 for name,fn in list(vars(extension).items()):
  if not name.startswith('_') and callable(fn):
   def counted(*args,_fn=fn,_name=name,_counts=counts,**kw):
    _counts[_name]=_counts.get(_name,0)+1
    return _fn(*args,**kw)
   setattr(extension,name,counted)

def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,float) and not math.isfinite(v):return None
 return v

def memory():
 return {'rss_bytes':psutil.Process().memory_info().rss,'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved(),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()}

def wrapped_time(name,field):
 original=getattr(d,name)
 def wrapped(*args,**kw):
  torch.cuda.synchronize();t=time.perf_counter()
  try:return original(*args,**kw)
  finally:torch.cuda.synchronize();result['timing'][field]+=time.perf_counter()-t
 setattr(d,name,wrapped)
if a.crown_conv_mode=='matrix':
 assert a.benchmark=='tora_homogeneous'
 def source_crown(config,device,relax='same-slope'):
  from auto_LiRPA import BoundedModule
  assert relax=='same-slope'
  return BoundedModule(d.build_raw_net(config),torch.zeros(1,*tuple(config['input_shape'])[1:],dtype=torch.float64),device=device,bound_opts={'activation_bound_option':'same-slope','conv_mode':'matrix'})
 d.build_crown=source_crown
 result['source_controller_options']={'activation_bound_option':'same-slope','conv_mode':'matrix','source':'xiangru_upstream submit/CROWN-Reach/archcomp/TORA/crown.py:44','reason':'restore explicit source mode omitted by generic driver'}
acc_state=None
if cfg.get('adapter')=='acc_exact_feature_tm':
 from nncs_acc_adapter import install,ACCUnknown
 acc_state=install(d,cfg)
 result['controller']=dict(CONTROLLER,known_gap='CROWN RN affine/bias computations have no proved outward roundoff accounting. ACC adapter feature and injection arithmetic has explicit interval residuals validated independently of CROWN. Source RPC asFloat arithmetic is not reproduced.')
 result['adapter']={'kind':'acc_exact_feature_tm','sr_queue':50,'feature_mapping':cfg['controller_input'],'adapter_sha256':sha(Path(__file__).with_name('nncs_acc_adapter.py'))}
wrapped_time('build_crown','controller_setup_s');wrapped_time('crown_bounds','controller_bound_s')
original_sr=sr_kernels.left_multiply_
def sr_dispatch(left,history):
 result['sr_kernel_dispatches']+=1
 key='x'.join(str(x) for x in history.shape)
 result['sr_dispatch_shapes'][key]=result['sr_dispatch_shapes'].get(key,0)+1
 return original_sr(left,history)
sr_kernels.left_multiply_=sr_dispatch
original_sum=sr_sum_kernels.sum_history
def sum_dispatch(phi,j):
 result['sr_sum_dispatches']+=1
 key='x'.join(str(x) for x in phi.shape)
 result['sr_sum_shapes'][key]=result['sr_sum_shapes'].get(key,0)+1
 return original_sum(phi,j)
sr_sum_kernels.sum_history=sum_dispatch
# Only actual SR capacity growth triggers this additional allocator cleanup.
# All elapsed time is inside the existing plant advance/driver/process totals.
from flowstar_gpu import symbolic_remainder
reserve= symbolic_remainder.SymbolicRemainder.reserve
result['allocator_cap_GiB']=9
result['sr_growth_events']=[]
result['point_history_policy']='strict CUDA point einsum output chunks <=64MiB; actual branch independently validated'
def pool_summary():
 groups={}
 for seg in torch.cuda.memory_snapshot():
  key=str(seg.get('segment_pool_id',(0,0)))
  group=groups.setdefault(key,{'reserved':0,'active_allocated':0,'inactive':0,'active_awaiting_free':0})
  group['reserved']+=seg['total_size']
  for block in seg['blocks']:
   group[block['state']]=group.get(block['state'],0)+block['size']
 return groups

def observed_reserve(state,needed):
 if needed<=state.phi_buf.shape[0] or not state.phi_buf.is_cuda:return reserve(state,needed)
 torch.cuda.synchronize();row={'ode_step':result['attempted_steps']+1,'qlen':state.qlen,'old_capacity':state.phi_buf.shape[0],'needed':needed,'before':memory(),'pools_before':pool_summary()};t=time.perf_counter()
 allocated=torch.cuda.memory_allocated();torch.cuda.empty_cache();torch.cuda.synchronize();assert torch.cuda.memory_allocated()==allocated
 row['cleanup_s']=time.perf_counter()-t;row['after_cleanup']=memory();row['pools_after_cleanup']=pool_summary()
 result['sr_growth_events'].append(row);write(a.output/'sr_growth_events.json',result['sr_growth_events'])
 t=time.perf_counter()
 try:
  reserve(state,needed);torch.cuda.synchronize();row['after_reserve']=memory();row['new_capacity']=state.phi_buf.shape[0];row['pools_after_reserve']=pool_summary()
 except Exception as exc:
  row['error']=repr(exc);row['after_failure']=memory();row['pools_after_failure']=pool_summary();raise
 finally:
  row['reserve_s']=time.perf_counter()-t;write(a.output/'sr_growth_events.json',result['sr_growth_events'])
symbolic_remainder.SymbolicRemainder.reserve=observed_reserve
result['sr_allocation']='preallocated_full_capacity' if a.reserve_sr_full else 'lazy_growth'
result['timing']['sr_initialization_s']=0.
result['timing']['snapshot_s']=0.
make_sr=d.make_symbolic_remainder
def initialize_sr(*args,**kw):
 t=time.perf_counter();state=make_sr(*args,**kw)
 if a.reserve_sr_full:
  assert state.qlen==state.jlen==0 and state.max_size==1000 and a.benchmark=='quad_submit_native'
  state.reserve(state.max_size)
  assert state.qlen==state.jlen==0 and state.phi_buf.shape[0]==1000
  result['sr_initialization']={'qlen':state.qlen,'jlen':state.jlen,'capacity':state.phi_buf.shape[0],'interval_history_initially_absent':state.phi_iv_buf is None,'memory':memory(),'included_in_child_and_process_wall':True}
 torch.cuda.synchronize();result['timing']['sr_initialization_s']+=time.perf_counter()-t
 return state
d.make_symbolic_remainder=initialize_sr
advance=d.advance_sparse

def observed(*args,**kw):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 try:state,ok=advance(*args,**kw);torch.cuda.synchronize()
 except Exception:
  result['timing']['plant_advance_s']+=time.perf_counter()-t
  result['interrupted_advance_after_completed_step']=result['attempted_steps'];raise
 allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
 if bool(ok.all()) and reserved-allocated>2*2**30:
  clear_t=time.perf_counter();torch.cuda.empty_cache();torch.cuda.synchronize();clear_s=time.perf_counter()-clear_t
  assert torch.cuda.memory_allocated()==allocated
  result['cache_cleanup']['events'].append({'ode_step':result['attempted_steps']+1,'allocated_before':allocated,'allocated_after':torch.cuda.memory_allocated(),'reserved_before':reserved,'reserved_after':torch.cuda.memory_reserved(),'elapsed_s':clear_s})
  result['cache_cleanup']['total_cleanup_s']+=clear_s
 solve_s=time.perf_counter()-t
 result['timing']['plant_advance_s']+=solve_s;result['attempted_steps']+=1
 accepted=ok.cpu().tolist();result['all_steps_accepted'] &= all(accepted)
 result['accepted_steps_all_lanes']+=int(all(accepted));result['accepted_lane_steps']+=sum(accepted)
 if a.snapshot_dir is not None and result['attempted_steps'] in [1,2,16,17,20]:
  ts=time.perf_counter();history=args[6];a.snapshot_dir.mkdir(parents=True,exist_ok=True)
  sample={'state':{k:getattr(state,k).detach().cpu().clone() for k in ['pre','pre_rem','tmv','tmv_rem','status']},'pre_support':state.pre_sup.ids,'tmv_support':state.tmv_sup.ids,'sr':{'qlen':history.qlen,'jlen':history.jlen,'capacity':history.phi_buf.shape[0],'max_size':history.max_size,'phi':history.phi_buf[:history.qlen].detach().cpu().clone(),'j':history.j_buf[:history.jlen].detach().cpu().clone(),'phi_iv':history.phi_iv_buf[:history.qlen].detach().cpu().clone(),'scalars':history.scalars.detach().cpu().clone(),'scalars_iv':history.scalars_iv.detach().cpu().clone()}}
  torch.save(sample,a.snapshot_dir/f'step{result["attempted_steps"]:04d}.pt');del sample
  result['timing']['snapshot_s']+=time.perf_counter()-ts
 eng=args[2];graphs=getattr(eng,'_graphs',None)
 assert graphs is not None and graphs.enabled,'required CUDA graph path unavailable'
 solver_memory=memory();t=time.perf_counter()
 tube=d.hull_ranges_s(state,eng,cfg['num_vars'])
 original_pre=state.pre;original_pre_values=state.pre.clone();original_sup=state.pre_sup
 endpoint_state=copy.copy(state);d.end_of_time_s(endpoint_state,eng)
 endpoint=d.hull_ranges_s(endpoint_state,eng,cfg['num_vars'])
 assert state.pre is original_pre and state.pre_sup is original_sup
 assert torch.allclose(state.pre,original_pre_values,rtol=0,atol=0,equal_nan=True),'observer mutated solver polynomial'
 tube=tube.cpu();endpoint=endpoint.cpu();width_tube=tube[...,1]-tube[...,0];width_endpoint=endpoint[...,1]-endpoint[...,0]
 summary={'ode_step':result['attempted_steps'],'accepted_lanes':sum(accepted),'total_lanes':len(accepted),'plant_advance_s':solve_s,'endpoint_width_max_by_coordinate':width_endpoint.max(0).values.tolist(),'endpoint_width_mean_by_coordinate':width_endpoint.mean(0).tolist(),'tube_width_max_by_coordinate':width_tube.max(0).values.tolist(),'tube_width_mean_by_coordinate':width_tube.mean(0).tolist(),'graph':{'enabled':graphs.enabled,'hits':graphs.hits,'captures':graphs.captures,'segments':len(graphs._segs)},'sr_kernel_dispatches':result['sr_kernel_dispatches'],'sr_sum_dispatches':result['sr_sum_dispatches'],'extension_call_totals':{k:sum(v.values()) for k,v in result['extension_dispatch_counts'].items()},'solver_memory':solver_memory,'observer_memory':memory()}
 record={'ode_step':result['attempted_steps'],'state_order':[x['name'] for x in cfg['initial_set']],'accepted':accepted,'status':state.status.tolist(),'endpoint_definition':'production end_of_time_s on shallow state copy, then hull_ranges_s','endpoint':endpoint.tolist(),'tube':tube.tolist()}
 observer_s=time.perf_counter()-t;result['timing']['observer_s']+=observer_s;summary['observer_s']=observer_s
 t=time.perf_counter();trace.write(json.dumps(clean(record),allow_nan=False,separators=(',',':'))+'\n');trace.flush();io_s=time.perf_counter()-t
 result['timing']['trace_io_s']+=io_s;summary['trace_io_s']=io_s;result['step_summaries'].append(clean(summary))
 write(a.output/'progress.json',{'attempted_steps':result['attempted_steps'],'expected_steps':result['expected_steps'],'accepted_lane_steps':result['accepted_lane_steps'],'last_step':clean(summary),'timing':result['timing'],'cache_cleanup':result['cache_cleanup'],'sr_growth_events':result['sr_growth_events']})
 print('NNCS_OBSERVER '+json.dumps({'step':result['attempted_steps'],'accepted':sum(accepted),'B':len(accepted),'solve_s':solve_s,'observer_s':observer_s,'io_s':io_s,'allocated_GiB':solver_memory['cuda_allocated_bytes']/2**30,'reserved_GiB':solver_memory['cuda_reserved_bytes']/2**30,'sr_kernel_dispatches':result['sr_kernel_dispatches'],'sr_sum_dispatches':result['sr_sum_dispatches']},allow_nan=False),flush=True)
 return state,ok

d.advance_sparse=observed
sys.argv=[str(a.engine/'integrations/crown_reach/gpu_driver.py'),str(a.config),'--device','cuda:0','--engine','sparse','--strict','--crown-domain','box','--crown-relax','same-slope','--print-final-hull','--metrics-json',str(a.output/(a.benchmark+'_metrics.json'))]
result['driver_argv']=sys.argv.copy()
try:
 result['driver_returncode']=d.main()
 result['configured_horizon_completed']=result['attempted_steps']==result['expected_steps']
 if not result['all_steps_accepted']:result['status']='plant_numerical_rejection_controller_unqualified'
 elif result['configured_horizon_completed']:result['status']='plant_'+a.scope+'_accepted_controller_unqualified'
 else:result['status']='driver_early_stop_controller_unqualified'
except Exception as exc:
 if acc_state is not None and isinstance(exc,ACCUnknown):
  result.update(status='driver_early_stop_property_unknown_controller_unqualified',configured_horizon_completed=False,error=repr(exc));print('UNKNOWN',flush=True)
 else:
  result.update(status='error',error=repr(exc));traceback.print_exc()
finally:
 trace.close();result['timing']['child_wall_s']=time.perf_counter()-started
 result['timing']['unattributed_setup_bookkeeping_s']=result['timing']['child_wall_s']-sum(v for k,v in result['timing'].items() if k!='child_wall_s')
 result['engine_files_unchanged']=all(sha(p)==result['engine_files_sha256'][str(p.relative_to(a.engine))] for p in files)
 result['engine_sha_at_end']=subprocess.check_output(['git','-C',str(a.engine),'rev-parse','HEAD'],text=True).strip()
 if acc_state is not None:
  result['adapter']['observations']={k:v for k,v in acc_state.items() if k!='cache'}
 result['final_memory']=memory();write(a.output/(a.benchmark+'_result.json'),clean(result))
sys.exit(1 if result['status']=='error' else 0)
