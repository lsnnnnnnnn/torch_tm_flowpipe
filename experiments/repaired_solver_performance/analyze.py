"""Small raw-record readers shared by the report builder and bounded verifier."""
import csv
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

from experiments.endpoint_roundoff_repair.frozen import ROOT, MATCHED_SHA256, setup
from experiments.xiangru_adoption.common import measure
from experiments.repaired_solver_performance.compare import compare_pair

SCIENTIFIC_SHA='1551ab57aef7324f91882beeba9d368f36b3cdd5'
RUNTIME_SHA='1551ab57aef7324f91882beeba9d368f36b3cdd5'
BASE_SHA='7e41f33f515f5315b0dec7003a5b06ed0a79afd5'
FULL_NAMES=('brusselator_full_reference','brusselator_full_optimized',
            'vdp_full_reference','vdp_full_optimized','vdp_adaptive_optimized')
WINDOWS={'brusselator_early':('brusselator',1),'brusselator_middle':('brusselator',101),
         'brusselator_late':('brusselator',981),'vdp_early':('van_der_pol',1),
         'vdp_boundary':('van_der_pol',91)}
REUSED_INPUT_PATHS=(
    'experiments/endpoint_roundoff_repair/frozen.py','experiments/run_vdp_dense_backend.py',
    'experiments/run_brusselator_sr1000_parity.py',
    'artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/runtime_bridge.json',
    *[f'artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/{run}/{name}'
      for run in ['brusselator_full','vdp_full','vdp_adaptive']
      for name in ['models.jsonl.gz','bounds.csv','endpoint_audit.jsonl','summary.json']],
    *[f'artifacts/runs/xiangru_adoption_20260907T032448Z/raw_minimal/{run}/{name}'
      for run in ['native_brusselator','native_vdp'] for name in ['models.jsonl.gz','summary.json']],
)


def read_json(path):return json.loads(Path(path).read_text())
def read_csv(path):return list(csv.DictReader(Path(path).open()))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path,rows):
    with Path(path).open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def hex_equal(a,b):
    assert float(a).hex()==float(b).hex(),(a,b)


def verify_run(path,command=None,*,recompute_common=True):
    """Recompute counts, exact h/clock, range observer, and raw timing sums."""
    path=Path(path)
    summary=read_json(path/'summary.json');source=read_json(path/'source.json')
    contract=read_json(path/'execution_contract.json')
    assert summary['config']==contract
    assert summary['config_digest']==hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    expected_config=setup(summary['plant'])[0].as_dict()
    assert json.loads(json.dumps(expected_config))==contract['config'],'frozen configuration changed'
    assert contract['matched_contracts_sha256']==MATCHED_SHA256
    assert contract['endpoint_substitution_roundoff_required'] is True and contract['endpoint_ad_hoc_repair'] is False
    assert summary['scientific_sha']==source['scientific_sha']==SCIENTIFIC_SHA
    assert summary['source_clean_at_start'] and summary['source_clean_at_end'] and source['source_clean_at_start']
    for key,value in source.items():assert summary[key]==value,key
    assert summary['observer_mode']=='production_no_observer'
    assert summary['device']=='cpu' and summary['dtype']=='float64'
    assert summary['threads']==summary['interop_threads']==1 and summary['affinity']==[2]
    assert summary['mode']==contract['mode']
    assert summary['plant']==contract['plant'] and summary['adaptive']==contract['adaptive']
    assert summary['start_boundary']==contract['start_boundary'] and summary['requested_steps']==contract['requested_steps']
    assert summary['data_origin']=={'reference':'FRESH_REPAIRED_REFERENCE','optimized':'FRESH_OPTIMIZED'}[summary['mode']]
    assert summary['completed'] and summary['failure'] is None and summary['plan_construction_included_in_solve']
    if command:
        assert command['exit_code']==0 and command['source_sha']==SCIENTIFIC_SHA
        assert command['mode']==summary['mode'] and command['affinity']==summary['affinity']
        assert command['python']==summary['python'] and command['cwd']==summary['source_root']
        assert command['argv'][command['argv'].index('--mode')+1]==summary['mode']
        assert command['whole_process_seconds']>=summary['inside_process_seconds']
        assert command['whole_process_seconds']>=summary['solve_seconds']+summary['export_seconds']
    rows=read_csv(path/'bounds.csv')
    audit=[json.loads(line) for line in (path/'endpoint_audit.jsonl').read_text().splitlines()]
    timing=read_json(path/'timing_events.json')
    assert len(rows)==len(audit)==len(timing)==summary['accepted_steps']
    start=summary['start_boundary'];total=Fraction(summary['starting_time_exact'])
    scheduler=0. if start==0 else float(total)
    if not summary['adaptive']:
        assert summary['accepted_steps']==summary['requested_steps']
        fixed=float.fromhex(contract['fixed_h_hex'])
        assert fixed==(.02 if summary['plant']=='brusselator' else .01)
        assert total==start*Fraction(fixed)
    counts={};rejections=0
    for offset,(row,entry,event) in enumerate(zip(rows,audit,timing),1):
        index=start+offset
        assert int(row['step'])==entry['step']==event['step']==index
        h=float.fromhex(row['h_hex']);assert math.isfinite(h) and h>0
        hex_equal(row['h'],h);assert entry['h_hex']==h.hex()
        assert row['t_start_exact']==str(total)
        if not summary['adaptive']:assert h.hex()==fixed.hex()
        else:assert .002<=h<=.1
        total+=Fraction(h)
        assert row['t_end_exact']==entry['t_end_exact']==str(total)
        hex_equal(row['t_end'],float(total))
        scheduler=scheduler+h if summary['adaptive'] else index*fixed
        assert row['scheduler_time_hex']==scheduler.hex()
        assert row['safety_check_passed']=='True'
        assert int(row['queue_size'])==entry['queue_size']
        assert int(row['queue_reset_count'])==entry['queue_reset_count']
        capacity=contract['config']['accepted_boundary_sr_capacity']
        assert entry['queue_size']==index%capacity
        assert entry['queue_reset_count']==index//capacity
        for key,value in entry['refinement_counters'].items():counts[key]=counts.get(key,0)+value
        rejections+=entry['step_rejections']
        assert event['numerical_start']<event['numerical_stop']<=event['export_start']<event['export_stop']
        for key in ['substitution_error_hex','internal_substitution_error_hex','endpoint_remainder_hex']:
            assert len(entry[key])==2
            for lo,hi in entry[key]:
                lo,hi=float.fromhex(lo),float.fromhex(hi)
                assert math.isfinite(lo) and math.isfinite(hi) and lo<=hi
        assert 'endpoint_substitution_roundoff' in entry['dense_endpoint_ledger_hex']
        assert set(entry['state_hashes'])=={'current','normal_pre','normal_right'}
    assert summary['accepted_horizon_exact']==str(total)
    hex_equal(summary['accepted_horizon'],float(total))
    hex_equal(summary['requested_horizon'],float(Fraction(contract['requested_horizon_exact_nominal'])))
    assert summary['scheduler_time_hex']==scheduler.hex()
    if not summary['adaptive']:assert summary['fixed_step_hex']==contract['fixed_h_hex']
    assert summary['rejected_attempts']==rejections
    for key,value in counts.items():assert summary['refinement_totals'][key]==value
    # The exported backend counters belong to each accepted segment. Rejected
    # attempts are timed, but their discarded segment counters are not added.
    assert summary['refinement_totals']['polynomial_picard_iterations']==len(rows)*contract['config']['order']
    assert summary['final_queue_size']==audit[-1]['queue_size']
    assert summary['final_queue_reset_count']==audit[-1]['queue_reset_count']
    if summary['adaptive']:assert scheduler>=10.-1e-12
    hex_equal(summary['solve_seconds'],sum(e['numerical_stop']-e['numerical_start'] for e in timing))
    hex_equal(summary['export_seconds'],sum(e['export_stop']-e['export_start'] for e in timing))
    initialization=read_json(path/'initialization_event.json')
    hex_equal(summary['setup_seconds'],initialization['stop']-initialization['start'])
    assert initialization['start']<initialization['stop']<=timing[0]['numerical_start']
    assert all(a['export_stop']<=b['numerical_start'] for a,b in zip(timing,timing[1:]))
    if recompute_common:
        with gzip.open(path/'models.jsonl.gz','rt') as models:
            count=0
            for count,line in enumerate(models,1):
                record=json.loads(line);row=rows[count-1]
                assert record['step']==int(row['step']) and record['plant']==summary['plant']
                assert record['t_start_exact']==row['t_start_exact'] and record['t_end_exact']==row['t_end_exact']
                for name in ['endpoint','tube']:
                    bounds=measure(record['models'][name])
                    for dim,(lo,hi) in zip(['x','y'],bounds):
                        hex_equal(lo,row[f'common_{name}_{dim}_lo']);hex_equal(hi,row[f'common_{name}_{dim}_hi'])
            assert count==len(rows)
    return summary


def timing_tables(root):
    root=Path(root);commands=read_json(root/'commands.json');rows=[]
    for command in commands:
        if command['module']!='run':continue
        s=read_json(root/command['name']/'summary.json')
        match=re.search(r'_pair(\d+)_',command['name'])
        group=command['name'].split('_pair')[0] if match else command['name'].removesuffix('_reference').removesuffix('_optimized')
        rows.append({'run':command['name'],'group':group,'repeat':int(match.group(1)) if match else 1,
            'plant':s['plant'],'mode':s['mode'],'data_origin':s['data_origin'],'source_sha':s['scientific_sha'],
            'purpose':command['purpose'],'setup_seconds':s['setup_seconds'],'solve_seconds':s['solve_seconds'],
            'export_seconds':s['export_seconds'],'whole_process_seconds':command['whole_process_seconds'],
            'peak_rss_bytes':s['peak_rss_bytes'],'accepted':s['accepted_steps'],'rejected':s['rejected_attempts'],
            'actual_h_sum':str(Fraction(s['accepted_horizon_exact'])-Fraction(s['starting_time_exact']))})
    summary=[]
    for group,mode in sorted({(r['group'],r['mode']) for r in rows}):
        selected=[r for r in rows if r['group']==group and r['mode']==mode]
        item={'group':group,'mode':mode,'count':len(selected)}
        for field in ['setup_seconds','solve_seconds','export_seconds','whole_process_seconds','peak_rss_bytes']:
            values=[r[field] for r in selected]
            for statistic,fn in [('median',median),('min',min),('max',max)]:item[field+'_'+statistic]=fn(values)
        summary.append(item)
    return rows,summary


def profile_tables(root):
    root=Path(root);profiles=[];work=[];remaining=[];amdahl=[]
    for window,(plant,start) in WINDOWS.items():
        audit={r['step']:r for r in map(json.loads,(root/f'{window}_pair1_reference'/'endpoint_audit.jsonl').read_text().splitlines())}
        light={}
        for mode in ['reference','optimized']:
            rows=read_csv(root/f'{window}_profile_{mode}'/'exclusive.csv')
            assert {int(r['step']) for r in rows}==set(range(start,start+20))
            for row in rows:profiles.append({'window':window,**row})
            for index in range(start,start+20):
                selected=[r for r in rows if int(r['step'])==index]
                assert all(r['plant']==plant and r['mode']==mode for r in selected)
                assert all(int(r['replays'])==audit[index]['refinement_counters']['post_accept_replay_calls'] for r in selected)
                capacity=1000 if plant=='brusselator' else 100
                assert all(int(r['history_length'])==(index-1)%capacity for r in selected)
                assert all(float(r['exclusive_seconds'])>=0 and int(r['calls'])>=0 for r in selected)
                total=sum(float(r['exclusive_seconds']) for r in selected)
                assert abs(total-float(selected[0]['step_seconds']))<1e-10
                def seconds(test):return sum(float(r['exclusive_seconds']) for r in selected if test(r))
                def calls(op):return sum(int(r['calls']) for r in selected if r['phase']=='replay' and r['operation']==op)
                work.append({'window':window,'plant':plant,'mode':mode,'step':index,
                    'history_length':int(selected[0]['history_length']),'refinement_rounds':int(selected[0]['replays']),
                    'polynomial_multiplications_executed_in_replay':calls('polynomial_mul_trunc'),
                    'polynomial_range_evaluations_executed_in_replay':calls('polynomial_range_bound'),
                    'prepared_plan_constructions':calls('prepare_snapshot'),'cache_dispatches':calls('cache_dispatch'),
                    'binding_checks':calls('binding_checks'),
                    'fixed_replay_seconds':seconds(lambda r:r['phase']=='replay' and r['dependency_fixed']=='True'),
                    'all_replay_seconds':seconds(lambda r:r['phase']=='replay'),'step_seconds':total})
            if mode=='optimized':
                for category in sorted({r['category'] for r in rows}):
                    value=sum(float(r['exclusive_seconds']) for r in rows if r['category']==category)
                    remaining.append({'window':window,'category':category,'seconds':value,
                        'fraction_of_profiled_step_time':value/sum(float(r['exclusive_seconds']) for r in rows),
                        'scope':'exclusive_20_step_profile_not_production_denominator'})
            light_rows=read_csv(root/f'{window}_light_{mode}'/'exclusive.csv')
            light[mode]={'total':sum(float(r['exclusive_seconds']) for r in light_rows),
                         'replay':sum(float(r['exclusive_seconds']) for r in light_rows if r['phase']=='replay')}
        f=light['reference']['replay']/light['reference']['total']
        speed=light['reference']['replay']/light['optimized']['replay']
        paired=[compare_pair(root/f'{window}_pair{i}_reference',root/f'{window}_pair{i}_optimized')['speedup'] for i in [1,2,3]]
        amdahl.append({'window':window,'selected_fraction':f,'local_speedup_including_preparation_and_checks':speed,
            'predicted_total_speedup':1/((1-f)+f/speed),'max_possible_speedup':1/(1-f),
            'same_light_workload_measured_total':light['reference']['total']/light['optimized']['total'],
            'unprofiled_three_pair_speedup_median':median(paired),'unprofiled_three_pair_speedup_min':min(paired),
            'unprofiled_three_pair_speedup_max':max(paired),'light_reference_seconds':light['reference']['total'],
            'light_optimized_seconds':light['optimized']['total'],
            'prediction_scope':'this_same_20_step_workload; not a claimed full-horizon Amdahl bound'})
    return profiles,work,remaining,amdahl


def full_width_rows(root):
    rows=[]
    for plant,prefix in [('brusselator','brusselator'),('van_der_pol','vdp')]:
        left=read_csv(Path(root)/(prefix+'_full_reference')/'bounds.csv')
        right=read_csv(Path(root)/(prefix+'_full_optimized')/'bounds.csv')
        assert len(left)==len(right)==1000
        for a,b in zip(left,right):
            assert a['step']==b['step']
            for view in ['common','published']:
                for metric in ['endpoint','tube']:
                    for dim in ['x','y']:
                        name=f'{view}_{metric}_{dim}'
                        lo,hi,olo,ohi=[float(v) for v in [a[name+'_lo'],a[name+'_hi'],b[name+'_lo'],b[name+'_hi']]]
                        hex_equal(lo,olo);hex_equal(hi,ohi)
                        width=hi-lo;owidth=ohi-olo
                        rows.append({'plant':plant,'step':int(a['step']),'t_end_exact':a['t_end_exact'],
                            'view':view,'metric':metric,'component':dim,'reference_lo_hex':lo.hex(),'reference_hi_hex':hi.hex(),
                            'optimized_lo_hex':olo.hex(),'optimized_hi_hex':ohi.hex(),'reference_width':width,
                            'optimized_width':owidth,'width_ratio':owidth/width if width else 1.,'bitwise_equal':True})
    return rows


def decision(root):
    root=Path(root)
    full={name:read_json(root/name/'summary.json') for name in FULL_NAMES}
    for name,s in full.items():
        assert s['completed'] and s['scientific_sha']==SCIENTIFIC_SHA
        if not s['adaptive']:assert s['accepted_steps']==1000 and s['rejected_attempts']==0 and s['start_boundary']==0
    pairs={prefix:compare_pair(root/(prefix+'_full_reference'),root/(prefix+'_full_optimized')) for prefix in ['brusselator','vdp']}
    initialization_inclusive={prefix:{
        mode:full[f'{prefix}_full_{mode}']['setup_seconds']+full[f'{prefix}_full_{mode}']['solve_seconds']
        for mode in ['reference','optimized']} for prefix in ['brusselator','vdp']}
    for values in initialization_inclusive.values():values['speedup']=values['reference']/values['optimized']
    prefix_ratios={}
    for plant in ['brusselator','van_der_pol']:
        prefix_ratios[plant]=[compare_pair(root/f'{plant}_prefix100_pair{i}_reference',root/f'{plant}_prefix100_pair{i}_optimized')['speedup'] for i in [1,2,3]]
    b=pairs['brusselator']['speedup'];v=pairs['vdp']['speedup']
    vdp_ok=v>=1/1.1 and median(prefix_ratios['van_der_pol'])>=1/1.1
    target=b>=1.5 and vdp_ok
    useful=b>1 and min(prefix_ratios['brusselator'])>1
    status=('REPAIRED_REFERENCE_PRESERVED__PREPARED_REPLAY_SPEED_TARGET_MET' if target else
            'REPAIRED_REFERENCE_PRESERVED__USEFUL_SPEEDUP_BELOW_TARGET' if useful else
            'REPAIRED_REFERENCE_PRESERVED__NO_USEFUL_SPEEDUP')
    return {'status':status,'scientific_sha':SCIENTIFIC_SHA,'runtime_sha':RUNTIME_SHA,'default_reference_preserved':True,
        'all_fixed_endpoint_components_compared':4000,'all_fixed_tube_components_compared':4000,
        'common_and_published_views_compared':True,'full_pairs':pairs,'prefix100_speedups':prefix_ratios,
        'full_setup_plus_solve_seconds':initialization_inclusive,
        'brusselator_target_speedup':1.5,'vdp_no_more_than_10_percent_slowdown':vdp_ok,
        'full_timing_repeats_per_mode':1,'full_timing_confidence':'borderline_single_pair' if abs(b-1.5)/1.5<.1 else 'single_pair_with_repeated_windows_and_prefixes',
        'peak_rss_tradeoff':{name:value['peak_rss_ratio']>1.5 for name,value in pairs.items()},
        'adaptive_speed_claim':False,'whole_solver_formally_proved':False,'gpu_backend_choice':'undecided'}


def flowstar_widths(raw):
    rows=[];summaries=[]
    for plant,prefix,old_name in [('brusselator','brusselator','native_brusselator'),('van_der_pol','vdp','native_vdp')]:
        old=ROOT/'artifacts/runs/xiangru_adoption_20260907T032448Z/raw_minimal'/old_name
        fresh=read_csv(raw/(prefix+'_full_optimized')/'bounds.csv')
        with gzip.open(old/'models.jsonl.gz','rt') as handle:
            models=[json.loads(line) for line in handle]
        assert len(fresh)==len(models)==1000
        for item,model in zip(fresh,models):
            assert int(item['step'])==model['step'] and float(item['h']).hex()==float(model['h']).hex()
            for metric in ['endpoint','tube']:
                bounds=measure(model['models'][metric])
                for dim,(lo,hi) in zip(['x','y'],bounds):
                    width=hi-lo
                    ours=float(item[f'common_{metric}_{dim}_hi'])-float(item[f'common_{metric}_{dim}_lo'])
                    assert width>1e-12
                    rows.append({'plant':plant,'step':int(item['step']),'t_end_exact':item['t_end_exact'],
                        'metric':metric,'component':dim,'optimized_width':ours,'flowstar_width':width,
                        'ratio':ours/width,'flowstar_data_origin':'REUSED_MATCHED_REFERENCE'})
        for metric in ['endpoint','tube']:
            for dim in ['x','y']:
                selected=[r for r in rows if r['plant']==plant and r['metric']==metric and r['component']==dim]
                values=sorted(r['ratio'] for r in selected)
                maximum=max(selected,key=lambda r:r['ratio'])
                summaries.append({'plant':plant,'metric':metric,'component':dim,'p50':median(values),
                    'p95':values[949],'maximum':maximum['ratio'],'maximum_t_end_exact':maximum['t_end_exact'],
                    'weighting':'equal actual fixed h durations','flowstar_data_origin':'REUSED_MATCHED_REFERENCE'})
    return rows,summaries
