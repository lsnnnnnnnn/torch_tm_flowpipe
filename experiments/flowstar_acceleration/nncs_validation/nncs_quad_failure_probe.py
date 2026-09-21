"""Read production initial validation at the first QUAD rejection, no callback fallback."""
from pathlib import Path
from fractions import Fraction
import sys,os,time,json,hashlib,copy,gzip,subprocess,collections,torch
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');E=N/'engine_sr_memory';R=N/'runs/nncs_quad_memory_20260922';S=N/'repo/experiments/flowstar_acceleration';O=Path.cwd();cfg=S/'configs_nncs/quad_submit_native_full.yaml'
sys.path.insert(0,str(S));from nncs_screen import load_driver
assert subprocess.check_output(['git','-C',str(E),'rev-parse','HEAD'],text=True).strip()=='15e527e9dd281e71897c8b23bc4a917dbbf9bb3b';assert not subprocess.check_output(['git','-C',str(E),'status','--short'],text=True)
assert [os.environ[x] for x in ['FLOWSTAR_COMPOSITION','FLOWSTAR_GLUE','FLOWSTAR_SUPPORT_POLICY']]==['horner','graph','measured']
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(9*2**30/torch.cuda.get_device_properties(0).total_memory)
d=load_driver(E)
from flowstar_gpu import sparse_exec as se,cuda_kernels as ck,tape_kernels as tk,sr_kernels as sk,sr_sum_kernels as sj
assert ck.available() and tk.available() and tk.valid_available() and sk.available() and sj.available()
reference=json.loads((R/'quad_full_preallocated/quad_submit_native_result.json').read_text());expected={x['ode_step']:x['accepted_lanes'] for x in reference['step_summaries']};records={}
with gzip.open(R/'quad_full_preallocated/quad_submit_native_trace.jsonl.gz','rt') as f:
 for i,line in enumerate(f,1):
  if i in [594,595]:records[i]=json.loads(line)
  if i==595:break
step=0;guess=None;events=[];counts=[];checks=[];started=time.perf_counter()
make=d.make_symbolic_remainder
def allocated(*a,**kw):
 sr=make(*a,**kw);assert sr.qlen==sr.jlen==0 and sr.max_size==1000;torch.cuda.empty_cache();sr.reserve(sr.max_size);return sr
d.make_symbolic_remainder=allocated
refine=se._refine_dispatch
def inspect(code,total,ok,bad,*a,**kw):
 if step in [594,595]:
  event={'ode_step':step,'total':total.detach().cpu().tolist(),'ok':ok.detach().cpu().tolist(),'bad_nonfinite_or_domain':bad.detach().cpu().tolist(),'input_remainder':guess.detach().cpu().tolist(),'capture_point':'production _refine_dispatch input, before refinement; normal tape path remains selected'};events.append(event)
 out=refine(code,total,ok,bad,*a,**kw)
 if step in [594,595]:event['bad_after_refinement']=out[1].detach().cpu().tolist()
 return out
se._refine_dispatch=inspect
class ProbeComplete(Exception):pass
advance=d.advance_sparse
def observed(*a,**kw):
 global step,guess
 step+=1;guess=a[5];state,ok=advance(*a,**kw);n=int(ok.sum());assert n==expected[step],(step,n,expected[step]);counts.append(n)
 if bool(ok.all()) and torch.cuda.memory_reserved()-torch.cuda.memory_allocated()>2*2**30:torch.cuda.empty_cache()
 if step in [594,595]:
  eng=a[2];tube=d.hull_ranges_s(state,eng,16);endpoint_state=copy.copy(state);d.end_of_time_s(endpoint_state,eng);endpoint=d.hull_ranges_s(endpoint_state,eng,16)
  checks.append({'step':step,'endpoint_matches':endpoint.cpu().tolist()==records[step]['endpoint'],'tube_matches':tube.cpu().tolist()==records[step]['tube'],'status_matches':state.status.cpu().tolist()==records[step]['status'],'status_counts':dict(collections.Counter(state.status.cpu().tolist()))})
  assert all(checks[-1][k] for k in ['endpoint_matches','tube_matches','status_matches'])
 if step==595:raise ProbeComplete()
 return state,ok
d.advance_sparse=observed;sys.argv=[str(E/'integrations/crown_reach/gpu_driver.py'),str(cfg),'--device','cuda:0','--engine','sparse','--strict','--crown-domain','box','--crown-relax','same-slope']
try:d.main()
except ProbeComplete:pass
else:raise AssertionError('probe did not stop at first failure')
(O/'initial_validation_events.json').write_text(json.dumps(events,indent=2)+'\n')
rows=[];names=records[595]['state_order']
for ev in events:
 bad=[i for i,v in enumerate(ev['bad_nonfinite_or_domain']) if v];failed=[i for i,v in enumerate(ev['ok']) if not v];violations=[];worst=None
 for lane,totals in enumerate(ev['total']):
  for dim,(lo,hi) in enumerate(totals):
   lower,upper=ev['input_remainder'][lane][dim];ratio=max(Fraction(abs(lo)),Fraction(abs(hi)))/Fraction(upper)
   entry={'lane':lane,'coordinate':names[dim],'dimension_zero_based':dim,'input_remainder':[lower,upper],'proposal':[lo,hi],'max_absolute_proposal_ratio_to_positive_guess':float(ratio),'exact_ratio_numerator':ratio.numerator,'exact_ratio_denominator':ratio.denominator}
   if worst is None or ratio>worst[0]:worst=(ratio,entry)
   if lo<lower or hi>upper:violations.append(entry)
 rows.append({'step':ev['ode_step'],'initial_selfmap_failed_lanes':len(failed),'nonfinite_or_domain_lanes':bad,'post_refinement_bad_lanes':[i for i,v in enumerate(ev['bad_after_refinement']) if v],'violation_count':len(violations),'violating_coordinates':dict(collections.Counter(v['coordinate'] for v in violations)),'worst_proposal':worst[1],'violations':violations})
result={'status':'passed','engine_sha':'15e527e9dd281e71897c8b23bc4a917dbbf9bb3b','mathematical_config_unchanged':True,'sr_allocation':'preallocated_full_capacity','scope':'diagnostic replay through first rejected step595, not full horizon','production_refinement_dispatch_preserved':True,'all_595_accepted_counts_equal_formal_run':True,'endpoint_tube_status_checks':checks,'rows':rows,'wall_s_excluding_reference_trace_read':time.perf_counter()-started,'config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest()}
(O/'failure_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}));print(json.dumps([{k:v for k,v in row.items() if k!='violations'} for row in rows],indent=2))
