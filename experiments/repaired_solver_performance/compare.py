"""Exact saved-run comparison; timing and provenance are checked separately."""
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path

from experiments.endpoint_roundoff_repair.frozen import MATCHED, MATCHED_SHA256


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value):
    if isinstance(value, float): return value.hex()
    if isinstance(value, dict): return {k:exact(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [exact(v) for v in value]
    return value


def read_run(path):
    path=Path(path)
    return dict(path=path,source=read(path/'source.json'),summary=read(path/'summary.json'),
                contract=read(path/'execution_contract.json'),bounds=list(csv.DictReader((path/'bounds.csv').open())),
                audits=[json.loads(line) for line in (path/'endpoint_audit.jsonl').read_text().splitlines()],
                step_timings=read(path/'step_timings.json'))


def validate_run(run):
    source,summary,contract=run['source'],run['summary'],run['contract']
    require(hashlib.sha256(MATCHED.read_bytes()).hexdigest()==MATCHED_SHA256,'frozen contract source digest')
    matched=read(MATCHED)
    require(contract['config']==matched['plants'][summary['plant']]['our_frozen_configuration'],'effective mathematical configuration changed')
    mode=source['execution_mode']
    require(mode in ('reference','prepared_remainder_replay'),'unknown execution mode')
    origin='FRESH_REPAIRED_REFERENCE' if mode=='reference' else 'FRESH_OPTIMIZED'
    require(source['data_origin']==origin,'fresh/reused identity or execution mode mismatch')
    for key in ('scientific_sha','source_root','imported_package','data_origin','execution_mode','config_digest',
                'python','python_version','torch_version','device','dtype','threads','interop_threads','affinity','observer_mode','source_files'):
        require(source[key]==summary[key],f'summary/source mismatch: {key}')
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(digest==source['config_digest'],'actual configuration digest')
    require((source['device'],source['dtype'],source['threads'],source['interop_threads'])==('cpu','float64',1,1),'CPU binary64 single-thread contract')
    require(source['observer_mode']=='production_no_observer','profile/observer cannot be a production denominator')
    require(source['source_clean_at_start'] and summary['source_clean_at_end'],'scientific source must remain clean')
    require(contract['endpoint_substitution_roundoff_required'] and not contract['endpoint_ad_hoc_repair'],'endpoint correction contract changed')
    require(Path(source['imported_package']).is_relative_to(Path(source['source_root'])/'src'),'actual package import path')
    rows,audits,timings=run['bounds'],run['audits'],run['step_timings']
    require(len(rows)==len(audits)==len(timings)==summary['accepted_steps'],'incomplete per-step records')
    require(summary['completed'] and summary['failure'] is None,'requested run did not complete')
    start=int(summary.get('start_step',1));initial=Fraction(summary.get('initial_time_exact','0'));total=initial
    capacity=contract['config']['accepted_boundary_sr_capacity']
    scheduler=0.;next_h=.1;solve=0.;rejections=0
    counters={};work={}
    for index,(row,audit,timing) in enumerate(zip(rows,audits,timings,strict=True),start):
        require(int(row['step'])==audit['step']==timing['step']==index,'actual step sequence')
        require(audit['execution_mode']==mode and audit['data_origin']==origin,'per-step mode/identity changed')
        h=float.fromhex(row['h_hex'])
        require(math.isfinite(h) and h>0 and float(row['h']).hex()==h.hex(),'actual h encoding')
        require(audit['h_hex']==row['h_hex'],'audit h disagrees')
        require(row['t_start_exact']==str(total),'actual initial time')
        total+=Fraction(h)
        require(row['t_end_exact']==audit['t_end_exact']==str(total),'actual binary64 sum(h)')
        require(float(row['t_end']).hex()==float(total).hex(),'display time changed')
        if contract['adaptive']:
            remaining=summary['requested_horizon']-scheduler
            attempted=min(next_h,.1,remaining)
            if 0.<remaining-attempted<.002: attempted=remaining
            for _ in range(audit['step_rejections']): attempted*=.5
            require(.002<=attempted<=.1 and attempted.hex()==h.hex(),'original adaptive decision/h sequence')
            next_h=min(attempted*1.1,.1);scheduler+=h
            require(row['scheduler_time_hex']==scheduler.hex(),'adaptive scheduler clock')
        else:
            require(row['h_hex']==contract['fixed_h_hex'],'fixed h clipped/changed')
        require(int(row['queue_size'])==audit['queue_size']==index%capacity,'history queue capacity boundary')
        require(audit['queue_reset_count']==index//capacity,'history reset count')
        if index%capacity:require(audit['owner_boundary']==index,'current owner boundary')
        rejections+=audit['step_rejections']
        for k,v in audit['refinement_counters'].items():counters[k]=counters.get(k,0)+v
        for k,v in audit.get('execution_work',{}).items():work[k]=work.get(k,0)+v
        seconds=timing['solve_seconds']
        require(math.isfinite(seconds) and seconds>0,'invalid solve timing')
        require((timing['finished_perf']-timing['started_perf']).hex()==seconds.hex(),'changed raw timing event')
        solve+=seconds
    require(total==Fraction(summary['accepted_horizon_exact']) and float(total).hex()==summary['accepted_horizon'].hex(),'summary actual completion time')
    require(solve.hex()==summary['solve_seconds'].hex(),'solve time must be derived from all raw steps')
    require(rejections==summary['rejected_attempts'],'rejection counter recomputation')
    for k,v in counters.items():require(v==summary['refinement_totals'][k],'replay stop/commit counter recomputation')
    require(exact(work)==exact(summary.get('execution_work_totals',{})),'prepared work count/time recomputation')
    if mode=='prepared_remainder_replay' and summary['plant']=='brusselator':
        require(work['prepared_plan_count']==len(rows) and work['prepared_operation_hits']>0,'prepared mechanism was not actually used')
        require(0<work['prepared_plan_setup_s']<solve,'plan preparation must be inside solve')
    else:
        require(work.get('prepared_plan_count',0)==0,'unexpected prepared path')
    require(summary['numerical_with_setup_seconds']==summary['setup_seconds']+solve,'initialization excluded from numerical total')
    require(summary['inside_process_seconds']>=solve+summary['setup_seconds']+summary['export_seconds'],'overlapping timing categories')
    if contract['adaptive']:
        require(summary['scheduler_time_hex']==scheduler.hex() and scheduler>=10.-1e-12,'adaptive horizon incomplete')
    else:
        require(len(rows)==contract['requested_steps'],'fixed requested steps incomplete')
    return dict(plant=summary['plant'],steps=len(rows),rejections=rejections,actual_sum_h=str(total-initial),
                solve_seconds=solve,setup_seconds=summary['setup_seconds'],export_seconds=summary['export_seconds'])


AUDIT_IDENTITIES={'execution_mode','data_origin','execution_work'}


def compare_runs(reference_path,optimized_path):
    old,new=read_run(reference_path),read_run(optimized_path)
    old_summary,new_summary=validate_run(old),validate_run(new)
    require(old['source']['execution_mode']=='reference' and new['source']['execution_mode']=='prepared_remainder_replay','comparison lane roles')
    require(old['contract']==new['contract'],'unmatched mathematical workloads')
    for key in ('python','python_version','torch_version','threads','interop_threads','affinity','device','dtype','observer_mode'):
        require(old['source'][key]==new['source'][key],f'unmatched environment: {key}')
    require(len(old['bounds'])==len(new['bounds']),'unmatched coverage')
    widths=[]
    for before,after,a,b in zip(old['bounds'],new['bounds'],old['audits'],new['audits'],strict=True):
        require(set(before)==set(after),'bounds schema mismatch')
        for key in before:
            if key.endswith(('_lo','_hi')):
                require(float(before[key]).hex()==float(after[key]).hex(),f'bitwise range mismatch step {before["step"]}: {key}')
            else: require(before[key]==after[key],f'time/queue metadata mismatch: {key}')
        require(exact({k:v for k,v in a.items() if k not in AUDIT_IDENTITIES})==exact({k:v for k,v in b.items() if k not in AUDIT_IDENTITIES}),f'bitwise E/remainder/state/queue/replay mismatch at step {before["step"]}')
        for view in ('published','common'):
            for kind in ('endpoint','tube'):
                for component in ('x','y'):
                    key=f'{view}_{kind}_{component}'
                    lo,hi=float(before[key+'_lo']),float(before[key+'_hi'])
                    new_lo,new_hi=float(after[key+'_lo']),float(after[key+'_hi'])
                    widths.append(dict(plant=old_summary['plant'],step=int(before['step']),t_end_exact=before['t_end_exact'],
                        view=view,kind=kind,component=component,reference_lo_hex=lo.hex(),reference_hi_hex=hi.hex(),
                        optimized_lo_hex=new_lo.hex(),optimized_hi_hex=new_hi.hex(),bit_identical=True,
                        width_ratio=None if hi==lo else (new_hi-new_lo)/(hi-lo)))
    with gzip.open(old['path']/'models.jsonl.gz','rt') as a,gzip.open(new['path']/'models.jsonl.gz','rt') as b:
        count=0
        for left,right in zip(a,b,strict=True):
            require(exact(json.loads(left))==exact(json.loads(right)),'complete endpoint/tube polynomial or remainder changed')
            count+=1
        require(count==old_summary['steps'],'complete model coverage')
    return dict(reference=old_summary,optimized=new_summary,all_steps_bit_identical=True,
                solve_speedup=old_summary['solve_seconds']/new_summary['solve_seconds'],
                with_setup_speedup=(old_summary['solve_seconds']+old_summary['setup_seconds'])/(new_summary['solve_seconds']+new_summary['setup_seconds']),
                peak_rss_ratio=new['summary']['peak_rss_bytes']/old['summary']['peak_rss_bytes'],widths=widths)
