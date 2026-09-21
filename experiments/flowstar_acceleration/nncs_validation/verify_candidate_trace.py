from pathlib import Path
import gzip,json,math,sys,hashlib
R=Path(sys.argv[1]);name=sys.argv[2]
r=json.loads((R/(name+'_result.json')).read_text());metrics=R/(name+'_metrics.json');m=json.loads(metrics.read_text()) if metrics.exists() else None
count=accepted=0;last=None;all_finite=True
for line in gzip.open(R/(name+'_trace.jsonl.gz'),'rt'):
 x=json.loads(line);last=x;count+=1;accepted+=sum(x['accepted'])
 assert len(x['accepted'])==len(x['endpoint'])==len(x['tube'])
 for field in ['endpoint','tube']:
  for cell in x[field]:
   assert len(cell)==len(x['state_order'])
   if not all(lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and lo<=hi for lo,hi in cell):all_finite=False
assert count==r['attempted_steps'] and accepted==r['accepted_lane_steps']
trace=R/(name+'_trace.jsonl.gz');digest=hashlib.sha256()
with trace.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
verification={'verifier_version':2,'streamed_one_step_at_a_time':True,'records':count,'all_records_finite_ordered_intervals':all_finite,'accepted_lane_steps':accepted,'trace_gzip_bytes':trace.stat().st_size,'trace_sha256':digest.hexdigest(),'observer_checked_state_immutability':True}
if last and m and r.get('configured_horizon_completed') and r['all_steps_accepted']:
 hull={nm:[min(cell[j][0] for cell in last['endpoint']),max(cell[j][1] for cell in last['endpoint'])] for j,nm in enumerate(last['state_order'])}
 verification['last_endpoint_matches_driver_final_hull_exactly']=hull==m['final_hull'];assert hull==m['final_hull']
(R/'trace_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
assert all_finite or not r['all_steps_accepted']
print(json.dumps(verification,indent=2))
