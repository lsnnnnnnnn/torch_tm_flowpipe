from pathlib import Path
import zlib,json,math,hashlib,collections,time
R=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/nncs_quad5_20260922/quad_full');name='quad_submit_native';trace=R/(name+'_trace.jsonl.gz');start=time.perf_counter()
progress=json.loads((R/'progress.json').read_text());process=json.loads((R/'process.json').read_text());digest=hashlib.sha256();dec=zlib.decompressobj(31);pending=b'';count=accepted=0;finite=True;rows=[];last=None
observed={}
for line in (R/'stdout.log').read_text().splitlines():
 if line.startswith('NNCS_OBSERVER '):
  x=json.loads(line[len('NNCS_OBSERVER '):]);observed[x['step']]=x

def consume(line):
 global count,accepted,finite,last
 x=json.loads(line);count+=1;last=x;assert x['ode_step']==count and len(x['accepted'])==len(x['endpoint'])==len(x['tube'])==1024
 acc=sum(x['accepted']);accepted+=acc;row={'ode_step':count,'accepted_lanes':acc,'total_lanes':1024,'status_counts':dict(collections.Counter(x['status'])),'observed':observed.get(count)}
 for field in ['endpoint','tube']:
  maxima=[];means=[]
  for i in range(16):
   widths=[]
   for cell in x[field]:
    assert len(cell)==16;lo,hi=cell[i];valid=lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and lo<=hi;finite &= valid
    if valid:widths.append(hi-lo)
   maxima.append(max(widths) if widths else None);means.append(sum(widths)/len(widths) if widths else None)
  row[field+'_width_max_by_coordinate']=maxima;row[field+'_width_mean_by_coordinate']=means
 rows.append(row)
with trace.open('rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):
  digest.update(block);pending+=dec.decompress(block)
  while b'\n' in pending:
   line,pending=pending.split(b'\n',1)
   if line:consume(line)
assert count==progress['attempted_steps']==512 and accepted==progress['accepted_lane_steps']==512*1024 and not pending and not dec.eof and finite
verification={'verifier_version':'truncated-gzip-complete-flushed-lines','raw_trace_unchanged':True,'gzip_footer_present':dec.eof,'streamed_one_step_at_a_time':True,'records':count,'all_records_finite_ordered_intervals':finite,'accepted_lane_steps':accepted,'trace_gzip_bytes':trace.stat().st_size,'trace_sha256':digest.hexdigest(),'observer_checked_state_immutability':True,'status_counts_over_records':dict(collections.Counter(str(v) for x in rows for v,c in x['status_counts'].items() for _ in range(c)))}
verification['last_record_step']=last['ode_step'];verification['audit_wall_s']=time.perf_counter()-start
(R/'trace_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
(R/'recovered_step_summaries.json').write_text(json.dumps(rows,indent=2)+'\n')
recovery={'status':'resource_guard_prefix_accepted_controller_unqualified','scope':'full','expected_steps':1000,'attempted_steps':count,'count_definition':'completed logged advance returns; next advance was interrupted before its observer returned','interrupted_next_step':513,'accepted_steps_all_lanes':sum(x['accepted_lanes']==1024 for x in rows),'accepted_lane_steps':accepted,'all_steps_accepted':True,'configured_horizon_completed':False,'sr_kernel_dispatches':progress['last_step']['sr_kernel_dispatches'],'sr_sum_dispatches':progress['last_step']['sr_sum_dispatches'],'timing':dict(progress['timing'],recorded_prefix_only=True,process_wall_including_interrupted_advance=process['process_wall_s']),'extension_dispatch_counts':{k:{'total_at_last_completed_step':v} for k,v in progress['last_step']['extension_call_totals'].items()},'cache_cleanup':progress['cache_cleanup'],'step_summaries':rows,'last_graph':progress['last_step']['graph'],'runner_result_absent_due_to_sigterm':True,'hard_budget_overshoot_bytes':process['peak_owned_gpu_bytes']-process['hard_budget_bytes'],'engine_files_unchanged':'not finalized due to SIGTERM; separate git/source hash check required'}
(R/'prefix_recovery.json').write_text(json.dumps(recovery,indent=2)+'\n')
print(json.dumps(verification,indent=2))
