#!/usr/bin/env python3
"""Extract existing saved scan fields; no original checker or range scanner is run."""
import csv
import json
import math
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'tools/build_archcomp26_timing_20261005.py').exists())
BASE = ROOT/'docs/evidence/results/archcomp26_20261001/native_quad_paper_full50_001'
OUT = Path(__file__).resolve().parent
SRC = BASE/'SCAN.json'
scan = json.loads(SRC.read_text())
assert scan['record_count']==scan['expected_record_count']==1024000
assert scan['source_bytes']==417792000 and scan['record_size_bytes']==408
assert scan['complete_unique_ordered_grid'] is True
assert all(v==0 for v in scan['errors'].values())
assert len(scan['per_step'])==1000 and len(scan['terminal'])==12
fields=['tube_lo','tube_hi','endpoint_lo','endpoint_hi']
means=['tube_per_box_width_mean','tube_per_box_width_max','endpoint_per_box_width_mean','endpoint_per_box_width_max']
rows=[]
source_relative=str(SRC.relative_to(ROOT))
for i,step in enumerate(scan['per_step']):
 assert step['step']==i+1 and len(step['states'])==12
 for j,state in enumerate(step['states']):
  assert state['state']==j+1
  values=[state[k] for k in fields]
  assert all(math.isfinite(v) for v in values)
  assert values[0]<=values[2]<=values[3]<=values[1]
  row={'method':'flowstar_native','state':f'x{j+1}','step':i+1,'t_start':step['t_start'],'t_end':step['t_end'],
   **{k:state[k] for k in fields},'available_lanes':1024,
   'source':scan['source_path'],'saved_field_source':f'{source_relative}#/per_step/{i}/states/{j}',
   'time_source':f'{source_relative}#/per_step/{i}',
   **{k:scan['terminal'][j][k] if i==999 else None for k in means},
   'per_box_statistics_source':f'{source_relative}#/terminal/{j}' if i==999 else None}
  rows.append(row)
# Independently cross-read the existing published x3 curve and terminal CSV.
oldx=list(csv.DictReader((BASE/'pooled_x3_1000steps.csv').open()))
oldt=list(csv.DictReader((BASE/'terminal_12states.csv').open()))
assert len(oldx)==1000 and len(oldt)==12
compared=0
for i,old in enumerate(oldx):
 new=rows[i*12+2]
 for dst,src in [('tube_lo','tube_x3_lo'),('tube_hi','tube_x3_hi'),('endpoint_lo','endpoint_x3_lo'),('endpoint_hi','endpoint_x3_hi')]:
  assert new[dst]==float(old[src]);compared+=1
for j,old in enumerate(oldt):
 for k in fields+means:
  assert rows[999*12+j][k]==float(old[k]);compared+=1
with (OUT/'native_quad_allstates.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
# Direct read-back of every exported numerical endpoint; never an identity summary.
back=list(csv.DictReader((OUT/'native_quad_allstates.csv').open()))
assert len(back)==12000
for a,b in zip(rows,back):
 for k in fields+['t_start','t_end']:
  assert a[k]==float(b[k])
receipt={
 'schema':'native-quad-existing-allstates-extract-v1','status':'COMPLETED_SAVED_FIELD_EXTRACTION',
 'source_saved_scan':source_relative,'source_raw_path_recorded_by_scan':scan['source_path'],
 'source_bytes_recorded_by_scan':scan['source_bytes'],'record_count_recorded_by_scan':scan['record_count'],
 'record_format_recorded_by_scan':scan['record_format'],'record_size_bytes_recorded_by_scan':scan['record_size_bytes'],
 'scan_complete_unique_ordered_grid':scan['complete_unique_ordered_grid'],'scan_recorded_errors':scan['errors'],
 'output_rows':len(rows),'steps':1000,'states':12,'lanes':1024,
 'geometries':['per-step tube pooled union','per-step endpoint pooled union'],
 'per_box_statistics':'Only terminal step 1000 mean/max were saved in original SCAN; earlier step columns remain null.',
 'direct_comparisons':{'existing_x3_bounds_and_terminal_bounds_statistics':compared,'exported_csv_bounds_and_times':len(rows)*6,'all_equal':True},
 'source_receipt':str((BASE/'RESULT.json').relative_to(ROOT)),
 'source_receipt_fields':{k:json.loads((BASE/'RESULT.json').read_text())[k] for k in ['status','exit_code','wall_s','ended_utc']},
 'actions':'Read existing local SCAN and two existing CSVs; export their recorded numerical fields only. No SSH, new experiment, solver, old checker, range scan, or identity calculation.',
 'qualification':'Saved scan describes numerical observer ranges, not an independent NN/CROWN/native floating-point certificate. The native production gate remains closed.',
 'source_extractor':str(Path(__file__).resolve().relative_to(ROOT)),
}
(OUT/'native_quad_allstates.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'rows':len(rows),'direct_cross_file_values':compared,'csv_numeric_values_checked':len(rows)*6}))
