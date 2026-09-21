from pathlib import Path
import json,zlib,time
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');R=N/'runs/nncs_quad_memory_20260922';name='quad_submit_native_trace.jsonl.gz'
def lines(p):
 decoder=zlib.decompressobj(31);pending=b''
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):
   pending+=decoder.decompress(chunk)
   while b'\n' in pending:
    line,pending=pending.split(b'\n',1)
    if line:yield line
cases={
 'largest_first15e_vs_preallocated':(R/'quad_full_largest_first'/name,R/'quad_full_preallocated'/name),
 '5d_vs_preallocated':(N/'runs/nncs_quad5_20260922/quad_full'/name,R/'quad_full_preallocated'/name),
 'chunked97d_vs_preallocated':(R/'quad_full_chunked'/name,R/'quad_full_preallocated'/name)}
result={};start=time.perf_counter()
for key,(a,b) in cases.items():
 first=None;count=0;field_differences=[]
 for i,(x,y) in enumerate(zip(lines(a),lines(b)),1):
  count=i
  if x!=y and first is None:
   first=i;xx=json.loads(x);yy=json.loads(y);field_differences=[k for k in xx if xx[k]!=yy[k]]
  if i==512:break
 assert count==512
 result[key]={'records':count,'all_serialized_records_equal':first is None,'first_nonidentical_record':first,'first_numeric_field_differences':field_differences,'left':str(a),'right':str(b)}
result['wall_s']=time.perf_counter()-start
(R/'prefix_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
