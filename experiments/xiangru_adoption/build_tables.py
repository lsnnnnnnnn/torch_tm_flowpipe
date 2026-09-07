"""Recompute this comparison's compact tables directly from run records."""
import argparse
import csv
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
if __package__:
    from .common import measure
else:
    from common import measure

KINDS=['endpoint','tube'];DIMS=['x','y'];VIEWS=['published','common']

def read_csv(p):
    with p.open() as h:return list(csv.DictReader(h))

def write_csv(p,rows):
    if not rows:raise ValueError(f'No evidence rows for {p}')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)

def weighted_quantile(pairs,q):
    pairs=sorted((v,Fraction(w)) for v,w in pairs)
    target=Fraction(str(q))*sum((w for _,w in pairs),Fraction());total=Fraction()
    for v,w in pairs:
        total+=w
        if total>=target:return v
    return pairs[-1][0]

def native_rows(directory):
    result=[]
    with gzip.open(directory/'models.jsonl.gz','rt') as h:
      for line in h:
        obj=json.loads(line);step=obj['step'];delta=Fraction(obj['h'])
        row={'step':step,'t_start':float((step-1)*delta),'t_end':float(step*delta),
             't_start_exact':str((step-1)*delta),'t_end_exact':str(step*delta),'safety_check_passed':obj['safety_check_passed']}
        for kind in KINDS:
            for view,bounds in [('published',obj['published'][kind]),('common',measure(obj['models'][kind]))]:
                for dim,(lo,hi) in zip(DIMS,bounds):
                    row[f'{view}_{kind}_{dim}_lo']=lo;row[f'{view}_{kind}_{dim}_hi']=hi
        result.append(row)
    return result

def candidate_rows(summary,plant,mode):
    result=[]
    for obj in summary['rows']:
        if (obj['plant'],obj['device'],obj['backend'],obj['mode'],obj['batch'])!=(plant,'cuda','sparse',mode,1):continue
        if not all(obj['accepted']):continue
        delta=Fraction(obj['settings']['step']);step=obj['step']
        row={'step':step,'t_start':float((step-1)*delta),'t_end':float(step*delta),
             't_start_exact':str((step-1)*delta),'t_end_exact':str(step*delta),'safety_check_passed':True}
        for kind in KINDS:
            for view,key in [('published','published_pre_range'),('common','common_composed_range')]:
                for dim,(lo,hi) in zip(DIMS,obj['bounds'][kind][key]):
                    row[f'{view}_{kind}_{dim}_lo']=lo;row[f'{view}_{kind}_{dim}_hi']=hi
        result.append(row)
    return result

def tables(root):
    contract=json.loads((root/'MATCHED_CONTRACTS.json').read_text())
    short=json.loads((root/'raw_minimal/candidate_short/summary.json').read_text())
    widths=[];horizons=[];timings=[]
    for plant,cfg in contract['plants'].items():
        shortname='brusselator' if plant=='brusselator' else 'vdp'
        ndir=root/'raw_minimal'/('native_'+shortname)
        odir=root/'raw_minimal'/('our_'+shortname+'_full')
        native=native_rows(ndir);ours=read_csv(odir/'bounds.csv')
        lanes={'ours':ours,'flowstar':native,
               'strict':candidate_rows(short,plant,'strict'),'parity':candidate_rows(short,plant,'parity')}
        indexed={lane:{(Fraction(r['t_start_exact']),Fraction(r['t_end_exact'])):r for r in rows} for lane,rows in lanes.items()}
        domains=sorted(set().union(*(set(v) for v in indexed.values())))
        for a,b in domains:
          for view in VIEWS:
           for kind in KINDS:
            for dim in DIMS:
                row={'plant':plant,'view':view,'range_kind':kind,'state':dim,
                     't_start':float(a),'t_end':float(b),'t_start_exact':str(a),'t_end_exact':str(b),
                     'duration':float(b-a),'candidate_eligible':False}
                for lane,records in indexed.items():
                    raw=records.get((a,b));prefix=f'{view}_{kind}_{dim}'
                    lo=float(raw[prefix+'_lo']) if raw else None
                    hi=float(raw[prefix+'_hi']) if raw else None
                    row[lane+'_lo']=lo;row[lane+'_hi']=hi;row[lane+'_width']=hi-lo if raw else None
                    row[lane+'_missing_reason']='' if raw else ('correctness_gate_stopped_long_run' if lane in ['strict','parity'] else 'not_accepted')
                for num,den in [('strict','ours'),('strict','flowstar'),('strict','parity'),('ours','flowstar')]:
                    label=num+'_vs_'+den;x=row[num+'_width'];y=row[den+'_width']
                    valid=x is not None and y is not None
                    row[label]=x/y if valid and y>1e-12 else None
                    row[label+'_absolute_width_difference']=x-y if valid else None
                    row[label+'_lower_shift']=row[num+'_lo']-row[den+'_lo'] if valid else None
                    row[label+'_upper_shift']=row[num+'_hi']-row[den+'_hi'] if valid else None
                    row[label+'_center_shift']=(row[num+'_lo']+row[num+'_hi']-row[den+'_lo']-row[den+'_hi'])/2 if valid else None
                    row[label+'_disjoint']=(row[num+'_hi']<row[den+'_lo'] or row[den+'_hi']<row[num+'_lo']) if valid else None
                widths.append(row)
        for lane,directory in [('our_cpu_reference',odir),('native_flowstar',ndir)]:
            s=json.loads((directory/'summary.json').read_text())
            horizons.append({'plant':plant,'mode':lane,'step_policy':'fixed','requested_horizon':float(cfg['requested_horizon']['decimal']),
                             'accepted_horizon':s['accepted_horizon'],'accepted_steps':s['accepted_steps'],
                             'status':'COMPLETED' if s['accepted_steps']==cfg['requested_steps'] else 'STOPPED',
                             'source':'raw_minimal/'+directory.name+'/summary.json','reason':s.get('failure') or ''})
            timings.append({'plant':plant,'mode':lane,'B':1,'device':'cpu','measurement':'complete_first_solve_in_fresh_process',
                            'measured':True,'solve_seconds':s['solve_seconds'],'export_seconds':s['export_seconds'],
                            'repeats':1,'source':'raw_minimal/'+directory.name+'/summary.json',
                            'throughput_tasks_per_second':1/s['solve_seconds'],'steps_per_second':s['accepted_steps']/s['solve_seconds'],
                            'requested_task_completed':s['accepted_steps']==cfg['requested_steps'],
                            'failure_count':int(s['accepted_steps']!=cfg['requested_steps']),
                            'reason':'One observed run; no warm-repeat median or variance claim.'})
            if s['accepted_steps']!=cfg['requested_steps']:
                timings[-1]['measurement']='first_solve_until_rejection_in_fresh_process'
                timings[-1]['throughput_tasks_per_second']=None
            process=directory/'process_time.txt'
            if process.exists():
                metrics=dict(v.split('=',1) for v in process.read_text().split() if '=' in v)
                timings.append({'plant':plant,'mode':lane,'B':1,'device':'cpu','measurement':'fresh_process_including_export',
                                'measured':True,'process_seconds':float(metrics['elapsed_seconds']),
                                'export_seconds':s['export_seconds'],'peak_rss_bytes':int(metrics['peak_rss_kb'])*1024,
                                'repeats':1,'source':'raw_minimal/'+directory.name+'/process_time.txt','reason':''})
        for mode in ['parity','strict']:
            rr=lanes[mode]
            horizons.append({'plant':plant,'mode':'xiangru_'+mode,'step_policy':'fixed','requested_horizon':float(cfg['requested_horizon']['decimal']),
                             'accepted_horizon':None,'algorithm_accepted_diagnostic_horizon':rr[-1]['t_end'] if rr else 0,
                             'accepted_steps':None,'diagnostic_steps':len(rr),'status':'NOT_RUN_AFTER_CONFIRMED_DEFECT',
                             'source':'raw_minimal/candidate_short/summary.json','reason':'Local exact inclusion failures; two-step outputs are diagnostics.'})
            for B in [1,8,32]:
              for measurement in ['warm_complete_solve','fresh_process_complete_solve']:
                timings.append({'plant':plant,'mode':'xiangru_'+mode,'B':B,'device':'cuda','measurement':measurement,
                                'measured':False,'solve_seconds':None,'export_seconds':None,'repeats':0,
                                'reason':'Correctness prerequisite failed; no extrapolation from short diagnostics.'})
        for B in [8,32]:
          for lane in ['our_cpu_reference','native_flowstar']:
            timings.append({'plant':plant,'mode':lane,'B':B,'device':'cpu','measurement':'serial_batch_complete_solve',
                            'measured':False,'solve_seconds':None,'repeats':0,'reason':'Paired batch speed campaign not opened after correctness gate failure; no estimated speed gate.'})
    horizons.extend([
        {'plant':'van_der_pol','mode':'xiangru_strict','step_policy':'adaptive_plus_sr','requested_horizon':10,'status':'UNSUPPORTED','reason':'Settings raises NotImplementedError.'},
        {'plant':'van_der_pol','mode':'our_cpu_reference','step_policy':'adaptive_plus_sr','requested_horizon':10,'status':'SUPPORTED_HISTORICALLY_NOT_REBENCHMARKED','reason':'Candidate combination unsupported; paired adaptive run skipped by goal section 17.5.'}])
    summaries=[];checkpoints=[]
    for plant,cfg in contract['plants'].items():
     for view in VIEWS:
      for kind in KINDS:
       for dim in DIMS:
        relevant=[r for r in widths if (r['plant'],r['view'],r['range_kind'],r['state'])==(plant,view,kind,dim)]
        for comparison in ['strict_vs_ours','strict_vs_flowstar','strict_vs_parity','ours_vs_flowstar']:
            num,den=comparison.split('_vs_')
            comparable=[r for r in relevant if r[num+'_width'] is not None and r[den+'_width'] is not None]
            found=[r for r in comparable if r[comparison] is not None]
            base={'plant':plant,'view':view,'range_kind':kind,'state':dim,'comparison':comparison,
                  'scope':'DIAGNOSTIC_ONLY' if comparison.startswith('strict') else 'ACCEPTED_COMMON_PREFIX',
                  'requested_duration':float(cfg['requested_horizon']['decimal']),
                  'compared_duration':math.fsum(r['duration'] for r in comparable),
                  'ratio_duration':math.fsum(r['duration'] for r in found),
                  'near_zero_denominator_duration':math.fsum(r['duration'] for r in comparable if r[comparison] is None),
                  'missing_duration':float(cfg['requested_horizon']['decimal'])-math.fsum(r['duration'] for r in comparable),
                  'full_requested_coverage':len(comparable)==cfg['requested_steps']}
            if comparable:
                base.update(max_absolute_width_difference=max(abs(r[comparison+'_absolute_width_difference']) for r in comparable),
                            max_absolute_lower_shift=max(abs(r[comparison+'_lower_shift']) for r in comparable),
                            max_absolute_upper_shift=max(abs(r[comparison+'_upper_shift']) for r in comparable),
                            max_absolute_center_shift=max(abs(r[comparison+'_center_shift']) for r in comparable),
                            disjoint_duration=math.fsum(r['duration'] for r in comparable if r[comparison+'_disjoint']),
                            duration_above_1_10_plus_1e_12=math.fsum(r['duration'] for r in comparable if r[num+'_width']>1.10*r[den+'_width']+1e-12))
            if found:
                worst=max(found,key=lambda r:r[comparison]);pairs=[(r[comparison],r['duration']) for r in found]
                base.update(max_ratio=worst[comparison],max_at_time=worst['t_end'],
                            weighted_median=weighted_quantile(pairs,.5),weighted_p95=weighted_quantile(pairs,.95))
                base.update(worst_numerator_width=worst[num+'_width'],worst_denominator_width=worst[den+'_width'])
            summaries.append(base)
            def checkpoint_row(h,selection,requested_time):
                return {**{k:base[k] for k in ['plant','view','range_kind','state','comparison','scope']},
                        'selection':selection,'requested_time':requested_time,
                        'actual_time':h['t_end'] if h else None,
                        'ratio':h[comparison] if h else None,
                        'numerator_width':h[num+'_width'] if h else None,
                        'denominator_width':h[den+'_width'] if h else None,
                        'numerator_lo':h[num+'_lo'] if h else None,
                        'numerator_hi':h[num+'_hi'] if h else None,
                        'denominator_lo':h[den+'_lo'] if h else None,
                        'denominator_hi':h[den+'_hi'] if h else None,
                        'absolute_width_difference':h[comparison+'_absolute_width_difference'] if h else None,
                        'reason':'' if h and h[num+'_width'] is not None and h[den+'_width'] is not None else 'missing_or_stopped'}
            for t in cfg['checkpoints']:
                hits=[r for r in relevant if abs(r['t_end']-t)<=1e-12]
                h=hits[0] if hits else None
                checkpoints.append(checkpoint_row(h,'specified_checkpoint',t))
            if found:checkpoints.append(checkpoint_row(worst,'maximum_ratio_in_observed_prefix',None))
    equivalence=[dict(r,box_set='repeated_full_box',scope='TWO_STEP_DIAGNOSTIC') for r in short['batch_equivalence']]
    state_checks=json.loads((root/'candidate_state_checks_with_distinct.json').read_text())
    equivalence.extend(dict(r,box_set='first_two_preregistered_distinct_boxes',scope='TWO_STEP_DIAGNOSTIC')
                       for r in state_checks['rows'] if r['check']=='different_preregistered_boxes_batch_equivalence')
    return {'widths_full_prefix.csv':widths,'width_summary.csv':summaries,'width_at_checkpoints.csv':checkpoints,
            'horizon_matrix.csv':horizons,'timings_raw.csv':timings,'timing_summary.csv':timings,
            'batch_equivalence.csv':equivalence}

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    result=tables(a.root)
    for name,rows in result.items():write_csv(a.root/name,rows)
    print({k:len(v) for k,v in result.items()})

if __name__=='__main__':main()
