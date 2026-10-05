#!/usr/bin/env python3
"""Check report tables and links against the saved timing audit, without executing benchmarks."""
import json
import re
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'tools/build_archcomp26_timing_20261005.py').exists())
OUT=Path(__file__).resolve().parent
D=json.loads((OUT/'timing_index.json').read_text())
REPORT=ROOT/'docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md'
text=REPORT.read_text()
metadata=json.loads((OUT.parent/'report_metadata.json').read_text())
methods=['pytorch_gpu','huan','xiangru','flowstar_native']
labels={'pytorch_gpu':'P3','huan':'Huan','xiangru':'Xiangru','flowstar_native':'原生 Flow*'}
layers={'process_wall_s':'外层 process','wrapper_wall_s':'候选 wrapper','payload_wall_s':'runner payload','driver_elapsed_s':'内部 driver','driver_call_wall_s':'调用 driver','native_process_wall_s':'原生子进程','server_startup_s':'控制器启动','watchdog_process_wall_s':'watchdog'}
def sec(v):return '—' if v is None else f'{v:.3f}'
def row_values(line):return [s.strip() for s in line.strip().strip('|').split('|')]
issues=[];layer_cells=0;links_checked=0;process_cells=0;campaign_cells=0
for item in metadata:
 b=item['instance_id'];head=f"## {item['number']} {item['title']}"
 block=text.split(head+'\n',1)[1].split('\n## ',1)[0]
 selected=[D['runs'].get(D['benchmarks'][b]['methods'][m]['current_selected_run']) for m in methods]
 timing=block.split('### 所选进程各层时间',1)[1].split('### ',1)[0]
 rows={row_values(line)[0]:row_values(line)[1:] for line in timing.splitlines() if line.startswith('|')}
 for field,label in layers.items():
  values=[r['timings'][field] if r else None for r in selected]
  expected=[sec(v) for v in values]
  if any(v is not None for v in values):
   if rows.get(label)!=expected:issues.append({'benchmark':b,'layer':field,'actual':rows.get(label),'expected':expected})
   layer_cells+=4
 source=next(line for line in block.splitlines() if line.startswith('原始结果：'))
 for m,r in zip(methods,selected):
  if r is None:continue
  match=re.search(r'\['+re.escape(labels[m])+r'\]\(([^)]+)\)',source)
  if not match:issues.append({'benchmark':b,'method':m,'missing_source_link':True});continue
  path=(REPORT.parent/match.group(1)).resolve()
  valid=[(ROOT/p).resolve() for p in r['raw_receipts']]
  if path not in valid or not path.exists():issues.append({'benchmark':b,'method':m,'link_not_selected_receipt':str(path)})
  links_checked+=1
 # The top table has a distinct title label rather than layer names.
 topline=next(line for line in text.split('### 四组既有重复进程计时',1)[0].splitlines() if line.startswith('| '+item['title']+' |'))
 actual=row_values(topline)[1:]
 expected=[]
 for m,r in zip(methods,selected):
  cell=D['benchmarks'][b]['methods'][m]
  if cell['coverage_status']=='contract_blocked':v='合同缺失'
  elif r and r['horizon']['complete_named_horizon']:
   v=sec(r['timings']['process_wall_s'])
   if cell['coverage_status']=='historical_same_contract_numeric_full' and r['timings']['process_wall_s'] is not None:v+=' 历史'
  else:v='未全程'
  expected.append(v)
 if actual!=expected:issues.append({'benchmark':b,'top_actual':actual,'top_expected':expected})
 process_cells+=4
 cell0=D['benchmarks'][b]['methods']['pytorch_gpu']
 if cell0['campaign']:
  campaign_text=text.split('### 四组既有重复进程计时',1)[1].split('## 五个新 P3',1)[0]
  row=next(line for line in campaign_text.splitlines() if line.startswith('| '+item['title']+' |'))
  expected=[]
  for m in methods:
   stats=D['benchmarks'][b]['methods'][m]['campaign']['later_statistics']['process_wall_s']
   expected.append(f"{sec(stats['median'])} [{sec(stats['min'])}, {sec(stats['max'])}]")
  if row_values(row)[1:]!=expected:issues.append({'benchmark':b,'campaign_mismatch':True})
  campaign_cells+=4
out={'status':'PASSED' if not issues else 'ISSUES_FOUND','report':str(REPORT.relative_to(ROOT)),
 'timing_source':'docs/evidence/results/archcomp26_report_20261005/timing/timing_index.json',
 'full_process_table_cells_checked':process_cells,'section_layer_cells_checked':layer_cells,
 'selected_original_result_links_checked':links_checked,'campaign_statistic_cells_checked':campaign_cells,
 'issues':issues,'scope':'Report formatting and exact selected source mapping only; no solver/checker run, statistical significance or certificate claim.'}
(OUT/'report_time_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False))
assert not issues
