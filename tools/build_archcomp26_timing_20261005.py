#!/usr/bin/env python3
"""Derive timing tables from saved local receipts only. Never runs a solver/checker."""
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/evidence/results/archcomp26_report_20261005/timing'
R = ROOT / 'docs/evidence/results/archcomp26_20261001'
REMOTE = '/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/'
AI = ROOT / 'docs/evidence/archcomp26_nohash_attempts_20261001.json'
DI = ROOT / 'docs/evidence/archcomp26_delivery_index_20261004.json'
NI = ROOT / 'research/p3_speed_tightness_20261005/RUN_INDEX.json'
REPORT = ROOT / 'docs/ARCHCOMP26_REPORT_HISTORY_20261004.md'
METHODS = ['pytorch_gpu', 'huan', 'xiangru', 'flowstar_native']
CURRENT_P3 = {
 'acc-safe-distance':'acc_p3_fast1_20261005_001/run_001',
 'nav-robust':'nav_robust_p3_fast32_20261005_001/run_001',
 'quad-reach':'quad_paper_p3_private256_full1000_20261005_001/run_001',
 'tora-reach-sigmoid':'tora_sigmoid_official_u11_p3_fused1_20261005_001/run_001',
 'unicycle-reach':'unicycle_p3_fused1_20261005_001/run_001',
}
LAYERS = ['process_wall_s', 'wrapper_wall_s', 'payload_wall_s', 'driver_elapsed_s',
          'driver_call_wall_s', 'native_process_wall_s', 'server_startup_s', 'watchdog_process_wall_s']
BOUNDARIES = {
 'process_wall_s': 'outer supervisor process start-to-reap wall; may include server startup',
 'wrapper_wall_s': 'candidate Python wrapper wall including its stated staging/import/check work',
 'payload_wall_s': 'benchmark runner wall; new candidates may preload Torch before this timer',
 'driver_elapsed_s': 'driver internal elapsed_s; excludes runner setup; includes its own lazy capture work',
 'driver_call_wall_s': 'caller wall around driver invocation; not the driver internal elapsed_s',
 'native_process_wall_s': 'native process wall; separate from RPC server startup and supervisor wall',
 'server_startup_s': 'RPC server startup duration only',
 'watchdog_process_wall_s': 'watchdog process duration; distinct from outer supervisor wall',
}
# These are the named report horizons, not each diagnostic's requested horizon.
HORIZONS = {
 'acc-safe-distance': (50, 5, 1, 248), 'airplane-continuous': (200, 2, 1, 260),
 'airplane-discrete': (None, None, None, 262), 'attitude-control-avoid': (60, 3, 1, 16),
 'balancing-reach': (2000, 10, 1, 276), 'docking-constraint': (400, 40, 1, 16),
 'double-pendulum-less-robust': (100, 1, 225, 42),
 'double-pendulum-more-robust': (80, .4, 225, 45),
 'nav-standard': (600, 6, 640, 121), 'nav-robust': (600, 6, 25, 121),
 'quad-reach': (1000, 5, 1024, 16), 'single-pendulum-reach': (100, 1, 1, 16),
 'tora-remain': (200, 20, 12, 16), 'tora-reach-sigmoid': (500, 5, 1, 75),
 'tora-reach-tanh': (500, 5, 1, 79), 'unicycle-reach': (500, 10, 1, 14),
}
CACHE = {}
WARNINGS = []

def pathstr(p):
 p = Path(p).resolve()
 return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)

def read(p):
 p = Path(p)
 if not p.exists(): return {}
 if p not in CACHE:
  try: CACHE[p] = json.loads(p.read_text())
  except (ValueError, UnicodeError): return {}
 return CACHE[p]

def ref(p, field, note=None):
 d = {'path': pathstr(p), 'field': field}
 if note: d['boundary_note'] = note
 return d

def number(x): return isinstance(x, (int,float)) and not isinstance(x,bool)
def method(x): return {'pytorch_gpu_p3_port':'pytorch_gpu','ours_p3':'pytorch_gpu','native':'flowstar_native'}.get(x,x)
def bench(x): return x.replace('double-pendulum-less-robust-continuous','double-pendulum-less-robust').replace('double-pendulum-more-robust-continuous','double-pendulum-more-robust')
def mapped(s): return R / s[len(REMOTE):] if s.startswith(REMOTE) else Path(s)

def newrun(key, b, m, generation):
 return {'run_key':key, 'benchmark':b, 'method':m, 'generation':generation,
  'sample_count':1, 'timings':dict.fromkeys(LAYERS), 'timing_sources':dict.fromkeys(LAYERS),
  'missing_timing_reasons':{}, 'raw_receipts':[], 'resources':{'cpu_affinity':None,'physical_gpu':None,'host':None},
  'resource_sources':{}, 'horizon':{}, 'horizon_sources':{}, 'started_utc':None,'ended_utc':None,
  'timestamp_sources':{}, 'stable_speed_ranking_eligible':False,
  'ranking_limitations':['Shared-host descriptive measurements; resource isolation and repeated current-version stability are not established.'],
  'same_physical_contract_as_current':None, 'same_implementation_as_current':None}

def put(run,k,value,p,field):
 if not number(value): return
 if run['timings'][k] is None:
  run['timings'][k]=value;run['timing_sources'][k]=ref(p,field,BOUNDARIES[k])

def resource(run,d,p):
 for dst,keys in [('cpu_affinity',['cpu_affinity','affinity','cpus','cpu_set']),('physical_gpu',['physical_gpu','cuda_visible_devices','gpu_visible','gpu']),('host',['host'])]:
  if run['resources'][dst] is None:
   for k in keys:
    if k in d:
     run['resources'][dst]=d[k];run['resource_sources'][dst]=ref(p,k);break
 for envkey in ['selected_environment','environment']:
  env=d.get(envkey,{})
  if isinstance(env,dict) and run['resources']['physical_gpu'] is None and 'CUDA_VISIBLE_DEVICES' in env:
   run['resources']['physical_gpu']=env['CUDA_VISIBLE_DEVICES'];run['resource_sources']['physical_gpu']=ref(p,envkey+'.CUDA_VISIBLE_DEVICES')
  if isinstance(env,dict) and run['resources']['cpu_affinity'] is None:
   cpu_entries=[(k,v) for k,v in env.items() if k.endswith('_CPUSET')]
   if len(cpu_entries)==1:
    k,v=cpu_entries[0];run['resources']['cpu_affinity']=v;run['resource_sources']['cpu_affinity']=ref(p,envkey+'.'+k)
 argv=d.get('argv',d.get('command',[]))
 if isinstance(argv,list) and run['resources']['cpu_affinity'] is None:
  for i,x in enumerate(argv[:-2]):
   if str(x).endswith('taskset') and argv[i+1]=='-c':
    run['resources']['cpu_affinity']=argv[i+2];run['resource_sources']['cpu_affinity']=ref(p,'argv' if 'argv'in d else 'command','taskset -c argument');break

def consume(run, paths, new=False):
 for p in paths:
  d=read(p)
  if not isinstance(d,dict) or not d:continue
  run['raw_receipts'].append(pathstr(p))
  outer='wall_s' in d and 'timed_out' in d and ('exit_code' in d or 'pid' in d)
  wrapper=new and p.parent.name=='candidate'
  if 'wall_s' in d:
   put(run,'process_wall_s' if outer else ('wrapper_wall_s' if wrapper else 'payload_wall_s'),d['wall_s'],p,'wall_s')
  for field,layer in [('supervisor_wall_s','process_wall_s'),('native_process_wall_s','native_process_wall_s'),('server_startup_s','server_startup_s'),('driver_wall_s','driver_call_wall_s'),('driver_elapsed_s','driver_elapsed_s')]:
   put(run,layer,d.get(field),p,field)
  if p.name=='process.json':put(run,'watchdog_process_wall_s',d.get('process_wall_s'),p,'process_wall_s')
  if p.name=='metrics.json':put(run,'driver_elapsed_s',d.get('elapsed_s'),p,'elapsed_s')
  if isinstance(d.get('metrics'),dict):put(run,'driver_elapsed_s',d['metrics'].get('elapsed_s'),p,'metrics.elapsed_s')
  resource(run,d,p)
  sp=p.with_name('START.json');sd=read(sp)
  if sd: resource(run,sd,sp)
  for k in ['started_utc','ended_utc']:
   srcd,srcp=(sd,sp) if k=='started_utc' and sd.get(k) else (d,p)
   if srcd.get(k) and run[k] is None:
    run[k]=srcd[k];run['timestamp_sources'][k]=ref(srcp,k)
  for outkey,keys in {
   'requested_substeps':['expected_substeps','expected_steps'],
   'observed_substeps':['observed_substeps','completed_substeps','attempted_steps'],
   'accepted_substeps':['accepted_substeps'],
   'accepted_lane_substeps':['accepted_lane_substeps','accepted_lane_steps'],
   'boxes':['initial_boxes','B'], 'saved_range_records':['range_records'],
  }.items():
   for k in keys:
    if number(d.get(k)) and outkey not in run['horizon']:
     run['horizon'][outkey]=d[k];run['horizon_sources'][outkey]=ref(p,k);break
  if outer:
   run.setdefault('outer_status',d.get('status'));run.setdefault('outer_exit_code',d.get('exit_code'))
  else:run.setdefault('payload_status',d.get('status'))
 for layer in LAYERS:
  if run['timings'][layer] is None:run['missing_timing_reasons'][layer]='No separately recorded duration with this boundary in the located saved receipts; not inferred from another layer or timestamps.'

def paths_for(a,cell=None):
 paths=[]
 # Explicit pointers first preserve separately located supervisors and inner payloads.
 for k in ['local_supervisor_result','local_result','local_payload_result','local_detail_result']:
  if a.get(k) and (ROOT/a[k]).exists():paths.append(ROOT/a[k])
 if cell:
  paths.extend(ROOT/x['local_result'] for x in cell.get('selected_current_receipts',[]))
 base=mapped(a.get('run_dir',''))
 candidates=[base]
 if a.get('local_summary'):
  parent=(ROOT/a['local_summary']).parent
  candidates.append(parent/base.name)
 # Do not sweep unrelated sibling jobs. Only fixed known receipt locations.
 for b in candidates:
  for sub in ['', 'supervisor','outer','attempt','run','data','payload','detail','attempt/data']:
   paths.append(b/sub/'RESULT.json')
  paths.append(b/'SUPERVISOR_RESULT.json')
  paths.extend([b/'parity/launcher_result.json', b/'parity/metrics.json'])
 seen=set();valid=[]
 for p in paths:
  if p.exists() and p not in seen:seen.add(p);valid.append(p)
 for p in list(valid):
  for mp in [p.with_name('metrics.json'),p.parent/'data/metrics.json',p.parent/'payload/metrics.json',p.parent/'detail/metrics.json']:
   if mp.exists() and mp not in seen:seen.add(mp);valid.append(mp)
 return valid

def attempt_run(a,n,cell=None):
 b,m=bench(a['instance']),method(a['method']);run=newrun(f'attempt_{n:03}',b,m,'20261001-04 saved attempt')
 consume(run,paths_for(a,cell))
 run.update(attempt_number=n,status=a.get('status'),role=a.get('role'),profile=a.get('profile'),
  attempt_source=ref(AI,f'attempts[{n-1}]'),run_dir=a.get('run_dir'),
  matrix_eligible_as_recorded=a.get('matrix_eligible'),timing_eligible_as_recorded=a.get('timing_eligible'),
  campaign_id=a.get('campaign_id'),reason_as_recorded=a.get('reason'))
 for dst,keys in {
  'requested_substeps':['ode_substeps_requested','expected_substeps'],
  'observed_substeps':['ode_substeps_observed','observed_ode_substeps','observed_substeps','ode_substeps','completed_substeps'],
  'accepted_substeps':['accepted_ode_substeps','accepted_substeps'],
  'accepted_lane_substeps':['accepted_lane_substeps','accepted_box_substeps'],
  'boxes':['initial_boxes'], 'saved_range_records':['range_records','saved_range_records'],
  'requested_control_periods':['control_periods_requested','control_periods'],
  'completed_control_periods':['control_periods_completed'],
  'first_rejected_substep':['first_rejected_ode_substep','first_rejected_substep'],
 }.items():
  for k in keys:
   if number(a.get(k)) and dst not in run['horizon']:
    run['horizon'][dst]=a[k];run['horizon_sources'][dst]=ref(AI,f'attempts[{n-1}].{k}');break
 run['horizon']['complete_named_horizon']=a.get('complete_numerical_run',False) is True and a.get('matrix_eligible',True) is not False
 run['horizon_sources']['complete_named_horizon']=ref(AI,f'attempts[{n-1}].complete_numerical_run','also requires matrix_eligible is not false; diagnostic completion does not fill main contract')
 if not run['horizon']['complete_named_horizon']:run['ranking_limitations'].append('Short, rejected, diagnostic, changed-profile or otherwise incomplete evidence cannot supply a full-horizon speed rank.')
 if cell:
  run['same_physical_contract_as_current']=True
  run['contract_sources']=cell.get('contract_paths',[])
  run['horizon']['complete_named_horizon']=cell['coverage_status']=='current_numeric_full'
  run['horizon_sources']['complete_named_horizon']=ref(DI,f'cells[{CELLS.index(cell)}].coverage_status')
  run['current_selected']=True
  if cell['coverage_status']!='current_numeric_full':run['ranking_limitations'].append(cell.get('property_and_evidence_limits','No complete current numerical horizon.'))
 if not any(x is not None for x in run['timings'].values()):
  run['unclassified_index_wall_s']=a.get('wall_s')
  run['unclassified_index_wall_source']=ref(AI,f'attempts[{n-1}].wall_s') if 'wall_s'in a else None
  run['unclassified_index_wall_reason']='Original index duration retained separately; no located raw receipt establishes its boundary.'
 return run

def historical(p,b,m,audit,key=None):
 p=Path(p);d=read(p);run=newrun(key or f'historical_{b}_{m}',b,m,'20260923 historical same-contract reuse')
 consume(run,[p,p.with_name('metrics.json')])
 run.update(status=d.get('status'),same_physical_contract_as_current=True,same_implementation_as_current=False,contract_audit=audit)
 run['horizon']['complete_named_horizon']=d.get('complete',False) is True
 run['horizon_sources']['complete_named_horizon']=ref(p,'complete')
 run['ranking_limitations'].append('Historical engine/process boundary/resources differ from current P3; same physical contract does not imply a matched timing experiment.')
 return run

def csvwrite(name,rows,columns):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
  for row in rows:
   w.writerow({k:json.dumps(row.get(k),ensure_ascii=False,separators=(',',':')) if isinstance(row.get(k),(dict,list)) else row.get(k) for k in columns})

def flat(run):
 row={k:run.get(k) for k in ['benchmark','method','run_key','generation','sample_count','status','outer_status','outer_exit_code','same_physical_contract_as_current','stable_speed_ranking_eligible','reason_as_recorded','started_utc','ended_utc']}
 row.update(run['timings']);row.update(run['horizon']);row.update(run['resources'])
 for k in LAYERS:row[k+'_source']=run['timing_sources'][k]
 row.update(raw_receipts=run['raw_receipts'],horizon_sources=run['horizon_sources'],resource_sources=run['resource_sources'],ranking_limitations=run['ranking_limitations'],concurrency=run.get('concurrency'))
 return row

ATTEMPTS=read(AI)['attempts'];CELLS=read(DI)['cells'];NEW=read(NI)['runs']
SELECTED={c['selected_current_attempt_number']:c for c in CELLS if c.get('selected_current_attempt_number')}
RUNS={r['run_key']:r for i,a in enumerate(ATTEMPTS,1) for r in [attempt_run(a,i,SELECTED.get(i))]}
BENCHMARKS={}
for b,(steps,T,B,line) in HORIZONS.items():
 BENCHMARKS[b]={'named_horizon':{'substeps':steps,'time_s':T,'initial_boxes':B,'source':ref(REPORT,f'line {line}','named report contract; diagnostics may request fewer steps')},'methods':{}}
for ci,c in enumerate(CELLS):
 b,m=c['instance_id'],c['method'];n=c.get('selected_current_attempt_number');selected=f'attempt_{n:03}' if n else None
 hp=c.get('historical_full_result_path');hist=None
 if hp:
  h=historical(hp,b,m,c.get('historical_same_contract_audit'));RUNS[h['run_key']]=h;hist=h['run_key']
  # The delivery contract audit confirms complete named horizon, independently of inconsistent historical key names.
  h['horizon']['complete_named_horizon']=c['coverage_status']=='historical_same_contract_numeric_full'
  h['horizon_sources']['complete_named_horizon']=ref(DI,f'cells[{ci}].coverage_status')
 display=hist if c['coverage_status']=='historical_same_contract_numeric_full' else selected
 BENCHMARKS[b]['methods'][m]={
  'coverage_status':c['coverage_status'],'contract_note':c['contract_note'],'contract_paths':c['contract_paths'],
  'displayed_run':display,'selected_current_attempt':selected,'historical_same_contract_run':hist,
  'current_attempts':[f'attempt_{i:03}' for i,a in enumerate(ATTEMPTS,1) if bench(a['instance'])==b and method(a['method'])==m],
  'new_p3_variants':[], 'campaign':None, 'missing_or_noncomparable':c.get('missing_or_noncomparable',[]),
  'cannot_claim_reasons':[c.get('property_and_evidence_limits',''),c.get('time_qualification','')],
  'source':ref(DI,f'cells[{ci}]')}

# Read each new stage's raw layered receipts; index is enumeration and comparison provenance only.
for i,n in enumerate(NEW):
 rid=n['run_id'];b=('acc-safe-distance' if rid.startswith('acc_') else 'nav-robust' if rid.startswith('nav_') else 'quad-reach' if rid.startswith('quad_') else 'tora-reach-sigmoid' if rid.startswith('tora_') else 'unicycle-reach')
 rp=NI.parent/n['raw_result'];run=newrun(rid+'/'+n['stage'],b,'pytorch_gpu','20261005 implementation candidate')
 consume(run,[rp,rp.parent/'candidate/RESULT.json',rp.parent/'candidate/data/RESULT.json',rp.parent/'candidate/data/metrics.json'],new=True)
 run.update(status=n['status'],qualification_source=ref(NI,f'runs[{i}]'),same_physical_contract_as_current=n['stage']!='gate_001',same_implementation_as_current=False,saved_comparison=n.get('saved_comparison'),comparison_scope=n.get('comparison_scope'))
 run['horizon']['complete_named_horizon']=n['completed_substeps']==HORIZONS[b][0] and 'EQUIVALENT' in n['status']
 run['horizon_sources']['complete_named_horizon']=ref(NI,f'runs[{i}].completed_substeps','equals named full horizon and saved-output-equivalent completion status')
 if n['completed_substeps'] is not None:
  run['horizon']['observed_substeps']=n['completed_substeps'];run['horizon_sources']['observed_substeps']=ref(NI,f'runs[{i}].completed_substeps')
 if n.get('boxes') is not None:
  run['horizon']['boxes']=n['boxes'];run['horizon_sources']['boxes']=ref(NI,f'runs[{i}].boxes')
 sd=read(rp.parent/'candidate/START.json')
 run['timing_policy']=sd.get('timing_policy',sd.get('timing_qualification'))
 run['timing_policy_source']=ref(rp.parent/'candidate/START.json','timing_policy') if sd else None
 run['ranking_limitations'] += ['One newly qualified process only; no repeated current-candidate median.', 'Early Torch/preload placement differs from the baseline payload boundary; compare each recorded layer separately.']
 if not run['horizon']['complete_named_horizon']:run['ranking_limitations'].append('Gate or 40-step prefix is not the full benchmark horizon; two-slope width comparison does not establish equivalent full output.')
 RUNS[run['run_key']]=run;BENCHMARKS[b]['methods']['pytorch_gpu']['new_p3_variants'].append(run['run_key'])

CAMPAIGNS={
 'acc-safe-distance':'acc_fourway_campaign_001',
 'single-pendulum-reach':'sp_two_state_fourway_campaign_20261002_001',
 'attitude-control-avoid':'attitude_corrected_fourway_campaign_20261002_001',
 'tora-reach-sigmoid':'tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002'}
STATS=[]
for b,name in CAMPAIGNS.items():
 base=R/name;summary=read(base/'SUMMARY.json');events=[json.loads(x) for x in (base/'events.jsonl').read_text().splitlines() if x.strip()]
 for m in METHODS:
  evs=[(i,e) for i,e in enumerate(events) if method(e['method'])==m]
  ids=[];first=[];later=[]
  for ei,e in evs:
   remote=e.get('directory',e.get('path'))
   found=[r for r in RUNS.values() if r.get('run_dir')==remote]
   assert len(found)==1,(name,remote)
   run=found[0];ids.append(run['run_key'])
   assert run['horizon']['complete_named_horizon'],run['run_key']
   field='process_wall_s' if 'process_wall_s'in e else 'wall_s'
   assert run['timings']['process_wall_s']==e[field],(run['run_key'],field)
   run['campaign_event_source']=ref(base/'events.jsonl',f'line {ei+1}')
   (first if e['round']==0 else later).append(run)
  assert len(first)==1 and len(later)==5,(name,m)
  stats={}
  for layer in LAYERS:
   present=[r for r in later if r['timings'][layer] is not None];values=[r['timings'][layer] for r in present]
   stats[layer]={'n':len(values),'median':statistics.median(values) if values else None,'min':min(values) if values else None,'max':max(values) if values else None,'input_sources':[r['timing_sources'][layer] for r in present]}
   STATS.append({'benchmark':b,'method':m,'campaign':name,'phase':'later fresh process','timing_layer':layer,**stats[layer]})
  mk=next(k for k in ['steady_process_wall_medians_s','later_process_wall_medians_s','later_median_wall_s'] if k in summary)
  original_method=next(e['method'] for _,e in evs)
  assert stats['process_wall_s']['median']==summary[mk][original_method],(name,m)
  stats['process_wall_s']['recorded_median_source']=ref(base/'SUMMARY.json',mk+'.'+original_method)
  for minmaxkey in ['later_process_wall_minmax_s']:
   if minmaxkey in summary:
    assert [stats['process_wall_s']['min'],stats['process_wall_s']['max']]==summary[minmaxkey][original_method]
  BENCHMARKS[b]['methods'][m]['campaign']={'id':name,'first_process':first[0]['run_key'],'later_processes':[r['run_key'] for r in later], 'later_statistics':stats,'source':pathstr(base/'SUMMARY.json'),'qualification':'Five sequential fresh processes in the saved campaign; descriptive statistics on a shared host, not a hardware-isolated stable rank.'}

HISTORICAL=[]
# Locate the existing external preserved evidence explicitly.
H=Path('/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923')
for b,name in [('nav-standard','nav_standard_ours'),('nav-robust','nav_robust_ours'),('tora-reach-tanh','tora_relu_tanh_ours')]:
 p=H/'evidence_v1/suite_v1'/name/'result.json'
 h=historical(p,b,'pytorch_gpu','Current delivery index audits the physical contract, but not current implementation.', 'old_linear_leaf_'+name)
 h['generation']='historical engine_linear_leaf_v2; excluded from current ranking';h['current_ranking_eligible']=False
 HISTORICAL.append(h)
for name in ['weighted_chunk256_full1000_v1','weighted_chunk128_samewrapper_full1000_v1']:
 base=ROOT/'docs/evidence/results/huan_quad_stage_a_40_20261001'/name/'run_001'
 if (base/'RESULT.json').exists():
  h=newrun('old_author_quad_'+name,'quad-reach','pytorch_gpu','old author-equation QUAD, not paper QUAD')
  consume(h,[base/'RESULT.json',base/'data/RESULT.json',base/'data/metrics.json',base/'watch/process.json'])
  h.update(same_physical_contract_as_current=False,current_ranking_eligible=False)
  h['ranking_limitations']=['Different QUAD plant/model contract; historical regression timing only.'];HISTORICAL.append(h)
p=ROOT/'research/gpu_verified_20260930/report/evidence/huan_parity_campaign.json';d=read(p)
if d:
 for i,row in enumerate(d.get('rows',[])):
  h=newrun('old_huan_parity_'+str(i+1),'quad-reach','huan','old author-equation parity campaign; not paper QUAD')
  for f,l in [('process_wall_s','process_wall_s'),('driver_call_wall_s','driver_call_wall_s'),('author_elapsed_s','driver_elapsed_s')]:put(h,l,row.get(f),p,f'rows[{i}].{f}')
  h.update(same_physical_contract_as_current=False,current_ranking_eligible=False)
  h['ranking_limitations']=['Different QUAD contract and original parity guarantee; excluded from current ranking.'];HISTORICAL.append(h)

# Known overlap derives only from saved timestamp windows. No interval implies no isolation proof.
windows=[]
for run in RUNS.values():
 try:
  start=datetime.fromisoformat(run['started_utc']);end=datetime.fromisoformat(run['ended_utc'])
  if end>=start:windows.append((start,end,run))
 except (TypeError,ValueError):pass
for run in RUNS.values():
 overlaps=[]
 own=next(((s,e) for s,e,r in windows if r is run),None)
 if own:
  overlaps=[r['run_key'] for s,e,r in windows if r is not run and s<own[1] and own[0]<e]
 run['concurrency']={'known_overlapping_indexed_runs':overlaps,'overlap_detected_in_indexed_cohort':bool(overlaps) if own else None,'qualification':'Only located saved timestamp windows are checked. Absence of listed overlap does not establish host/GPU isolation.'}
 run['timing_sample_count']=int(any(v is not None for v in run['timings'].values()))

# Validate source values directly, with no identity or content-summary operation.
def source_value(source):
 data=read(ROOT/source['path'])
 for part in re.findall(r'([^\.\[\]]+)|\[(\d+)\]',source['field']):
  key,num=part;data=data[int(num)] if num else data[key]
 return data
for run in list(RUNS.values())+HISTORICAL:
 for layer,value in run['timings'].items():
  if value is not None:assert value==source_value(run['timing_sources'][layer]),(run['run_key'],layer)
assert len(CELLS)==64 and len(ATTEMPTS)==289 and len(NEW)==13
assert sum(len(x['methods']) for x in BENCHMARKS.values())==64
assert RUNS['attempt_114']['timings']['process_wall_s'] is None
assert RUNS['attempt_115']['timings']['process_wall_s'] is None
assert RUNS['attempt_101']['timings']['process_wall_s'] is None
assert RUNS['attempt_091']['horizon']['complete_named_horizon'] and RUNS['attempt_091']['outer_exit_code']==2
assert not RUNS['attempt_201']['horizon']['complete_named_horizon']
for b,bd in BENCHMARKS.items():
 for m,cell in bd['methods'].items():
  cell['current_selected_run']=CURRENT_P3[b] if b in CURRENT_P3 and m=='pytorch_gpu' else cell['displayed_run']
  if cell['current_selected_run']!=cell['displayed_run']:
   assert RUNS[cell['current_selected_run']]['horizon']['complete_named_horizon'] is True
  cell['current_selection_policy']='Explicit 2026-10-05 report selection; five qualified full P3 candidates replace their baseline display only in the separate current table. Original displayed_run and campaign remain unchanged.'
OUT.mkdir(parents=True,exist_ok=True)
RESULT={'schema':'archcomp26-layered-timing-v1','derivation':'Read saved local JSON and campaign events only; no network, solver, old checker, content digest, timestamp-derived duration, or benchmark rerun.', 'timing_layer_definitions':BOUNDARIES,'benchmarks':BENCHMARKS,'runs':RUNS,'historical_fast_branch':HISTORICAL,'validation':{'benchmark_count':16,'method_cell_count':64,'attempt_count':289,'new_stage_count':13,'campaign_count':4,'campaign_valid_later_samples_per_method':5,'all_present_timing_values_equal_exact_source_fields':True,'all_campaign_process_medians_equal_saved_summary':True},'warnings':WARNINGS}
(OUT/'timing_index.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2)+'\n')
rows=[flat(r) for r in RUNS.values()];columns=list(dict.fromkeys(k for row in rows for k in row))
csvwrite('runs.csv',rows,columns)
four=[]
for b,bd in BENCHMARKS.items():
 for m,cell in bd['methods'].items():
  run=RUNS.get(cell['displayed_run']);row=flat(run) if run else {'benchmark':b,'method':m}
  row.update(coverage_status=cell['coverage_status'],contract_note=cell['contract_note'],cannot_claim_reasons=cell['cannot_claim_reasons'],named_horizon=bd['named_horizon'],campaign_later_n=5 if cell['campaign'] else None,campaign_later_statistics=cell['campaign']['later_statistics'] if cell['campaign'] else None,new_p3_variants=cell['new_p3_variants'])
  four.append(row)
csvwrite('four_way.csv',four,list(dict.fromkeys(k for row in four for k in row)))
current=[]
for b,bd in BENCHMARKS.items():
 for m,cell in bd['methods'].items():
  run=RUNS.get(cell['current_selected_run']);row=flat(run) if run else {'benchmark':b,'method':m}
  row.update(baseline_displayed_run=cell['displayed_run'],current_selected_run=cell['current_selected_run'],selection_policy=cell['current_selection_policy'],coverage_status=cell['coverage_status'],contract_note=cell['contract_note'],cannot_claim_reasons=cell['cannot_claim_reasons'],named_horizon=bd['named_horizon'],baseline_campaign_later_n=5 if cell['campaign'] else None,baseline_campaign_later_statistics=cell['campaign']['later_statistics'] if cell['campaign'] else None)
  current.append(row)
assert sum(a.get('run_key')!=b.get('run_key') for a,b in zip(four,current))==5
csvwrite('current_selected_four_way.csv',current,list(dict.fromkeys(k for row in current for k in row)))
csvwrite('campaign_statistics.csv',STATS,['benchmark','method','campaign','phase','timing_layer','n','median','min','max','input_sources'])
hrows=[flat(r) for r in HISTORICAL];csvwrite('historical_fast_branch.csv',hrows,list(dict.fromkeys(k for row in hrows for k in row)))
(OUT/'README.md').write_text('''# Saved timing audit — 2026-10-05

`timing_index.json` is indexed by benchmark, then method; run keys refer to `runs`.
`four_way.csv` preserves the original 64 delivery-index selections. `current_selected_four_way.csv` explicitly replaces five P3 cells with the full ACC small1, NAV robust fast32, QUAD private256 full1000, official sigmoid fused1 and Unicycle fused1 candidates; its baseline key is retained in every row. These are separate tables, not a silent revision of the baseline. `runs.csv` retains all 289 original attempts, eight historical reused cells and 13 new stages. `campaign_statistics.csv` gives each timing layer separately. `historical_fast_branch.csv` is excluded from the current comparison.

The displayed run is the delivery-index selection (or its explicitly audited historical full run), not a fastest-run selection. Five later-process statistics are preserved separately from that selected sample. All durations retain exact source paths and fields. Missing duration layers stay null; no timestamp subtraction, startup summation, payload-to-process substitution, failure extrapolation, or benchmark rerun is used.

A complete numeric horizon is distinct from the property verdict and independent floating-point NNCS qualification. Native Docking finishes all 400 steps despite outer exit 2. TORA-remain Huan/Xiangru observe the requested steps but reject lanes, so cannot supply full accepted-flowpipe times. Airplane discrete has no authoritative transition contract and no elapsed sample. New QUAD 40-step candidates remain prefixes of the 1000-step contract. New implementation candidates have one process each; their preloading changes the payload timing boundary.

Overlap lists cover only available saved start/end windows in this audit cohort. Empty lists do not prove isolated hardware. Existing historical identity metadata is neither copied nor checked.
''')
print(json.dumps(RESULT['validation'],ensure_ascii=False))
print('outputs',pathstr(OUT))
