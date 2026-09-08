"""Derive boundary execution evidence from raw numerical and timing events."""
import csv
from collections import defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import torch
from torch_tm_flowpipe import Interval, Polynomial
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution
from experiments.endpoint_roundoff_repair.frozen import ROOT, MATCHED_SHA256, setup
from experiments.xiangru_adoption.common import measure
from experiments.boundary_execution.compare import compare_pair, compare_parent_archive, check_switches

PARENT_SHA='f6af6f67565a954d0c68f88a50cc9c181a2d0b08'
SCIENTIFIC_SHA='5f37cbe0427c480ef0ebbbcaba292143bc8f4ede'
PARENT_RUNTIME_SHA='1551ab57aef7324f91882beeba9d368f36b3cdd5'
WINDOWS={'brusselator_early':('brusselator',0),'brusselator_middle':('brusselator',100),
         'brusselator_late':('brusselator',980),'vdp_early':('van_der_pol',0),
         'vdp_boundary':('van_der_pol',90)}


def read_json(path):return json.loads(Path(path).read_text())
def read_csv(path):return list(csv.DictReader(Path(path).open()))
def hex_equal(a,b):assert float(a).hex()==float(b).hex(),(a,b)


def published_bounds(model):
    domain=[Interval(*(float.fromhex(x) for x in bounds)) for bounds in model['domain']]
    result=[]
    with packed_boundary_execution(False):
        for component in model['components']:
            assert all(t['coefficient'][0]==t['coefficient'][1] for t in component['terms'])
            terms={tuple(t['degrees']):float.fromhex(t['coefficient'][0]) for t in component['terms']}
            assert len(terms)==len(component['terms'])
            interval=Polynomial(terms,len(domain)).evaluate_interval(domain)+Interval(*(float.fromhex(x) for x in component['remainder']))
            result.append(interval.to_tuple())
    return result


def verify_run(path,command=None,*,recompute_common=True):
    """Recompute counts, exact h/clock, range observer, and raw timing sums."""
    path=Path(path)
    summary=read_json(path/'summary.json');source=read_json(path/'source.json')
    contract=read_json(path/'execution_contract.json')
    for record in [summary,source,contract]:check_switches(record)
    assert summary['schema']=='boundary_execution_run/1'
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
    assert summary['data_origin']=={'baseline':'FRESH_PREPARED_BASELINE','candidate':'FRESH_PACKED_BOUNDARY_CANDIDATE'}[summary['mode']]
    assert summary['completed'] and summary['failure'] is None and summary['plan_construction_included_in_solve']
    if command:
        assert command['exit_code']==0 and command['source_sha']==SCIENTIFIC_SHA
        assert command['mode']==summary['mode'] and command['affinity']==summary['affinity']
        assert command['plant']==summary['plant']
        assert command['python']==summary['python'] and command['cwd']==summary['source_root']
        assert command['argv'][command['argv'].index('--mode')+1]==summary['mode']
        assert command['argv'][command['argv'].index('--plant')+1]==summary['plant']
        assert ('--adaptive' in command['argv'])==summary['adaptive']
        if not summary['adaptive']:
            assert int(command['argv'][command['argv'].index('--steps')+1])==summary['requested_steps']
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
                    published=published_bounds(record['models'][name])
                    for dim,(lo,hi) in zip(['x','y'],published):
                        hex_equal(lo,row[f'published_{name}_{dim}_lo']);hex_equal(hi,row[f'published_{name}_{dim}_hi'])
                    for dim,(lo,hi) in zip(['x','y'],bounds):
                        hex_equal(lo,row[f'common_{name}_{dim}_lo']);hex_equal(hi,row[f'common_{name}_{dim}_hi'])
            assert count==len(rows)
    return summary


def full_width_rows(raw):
    rows=[]
    for plant,prefix in [('brusselator','brusselator'),('van_der_pol','vdp')]:
        left=read_csv(Path(raw)/f'{prefix}_full_baseline/bounds.csv')
        right=read_csv(Path(raw)/f'{prefix}_full_candidate/bounds.csv')
        assert len(left)==len(right)==1000
        for a,b in zip(left,right):
            assert a['step']==b['step'] and a['t_end_exact']==b['t_end_exact']
            for view in ('published','common'):
                for metric in ('endpoint','tube'):
                    for component in ('x','y'):
                        key=f'{view}_{metric}_{component}'
                        lo,hi,clo,chi=[float(v) for v in (a[key+'_lo'],a[key+'_hi'],b[key+'_lo'],b[key+'_hi'])]
                        hex_equal(lo,clo);hex_equal(hi,chi)
                        rows.append(dict(plant=plant,step=int(a['step']),t_end_exact=a['t_end_exact'],
                            view=view,metric=metric,component=component,baseline_lo_hex=lo.hex(),baseline_hi_hex=hi.hex(),
                            candidate_lo_hex=clo.hex(),candidate_hi_hex=chi.hex(),baseline_width=hi-lo,candidate_width=chi-clo,
                            width_ratio=(chi-clo)/(hi-lo) if hi!=lo else 1.,bitwise_equal=True))
    return rows


def profile_cost_rows(profiles):
    profiles=Path(profiles);result=[]
    semantics=read_json(profiles.parent/'tensor_count_semantics.json')
    assert semantics['torch']==torch.__version__
    mutable=semantics['mutable_returns_per_call']
    for operation,returns in mutable.items():
        namespace,name,overload=operation.split('.')
        schema=getattr(getattr(getattr(torch.ops,namespace),name),overload)._schema
        assert schema.is_mutable and len(schema.returns)==returns==1
        assert str(schema.returns[0].type)=='Tensor' and schema.returns[0].alias_info.is_write
    for window in sorted([*WINDOWS,'brusselator_after_reset']):
        wall,counts=profiles/(window+'_wall'),profiles/(window+'_counts')
        assert read_json(wall/'states.json')==read_json(counts/'states.json'),'profiling changed numerical states'
        other={(r['step'],r['category']):r for r in read_csv(counts/'breakdown.csv')}
        self_returns=defaultdict(int)
        for event in read_csv(counts/'tensor_operations.csv'):
            self_returns[event['step'],event['category']]+=int(event['calls'])*mutable.get(event['operation'],0)
        records=read_csv(wall/'breakdown.csv')
        for row in records:
            match=other[row['step'],row['category']]
            for name in ['calls','interval_constructions','history_length','history_length_after']:
                assert row[name]==match[name],name
            row.update(tensor_outputs=match['tensor_outputs'],explicit_copy_events=match['explicit_copy_events'],window=window)
            row['inplace_tensor_self_returns']=self_returns[row['step'],row['category']]
            row['tensor_result_creations']=int(row['tensor_outputs'])-row['inplace_tensor_self_returns']
            row['actual_entry_details']=f'raw_minimal/profiles/{window}_wall/operations.csv'
            row['binding_details']=f'raw_minimal/profiles/{window}_wall/source.json'
            result.append(row)
        for index in {r['step'] for r in records}:
            step_rows=[r for r in records if r['step']==index]
            assert len(step_rows)==9 and {r['category'] for r in step_rows}==set('ABCDEFGH')|{'outside_boundary'}
            assert all(float(r['seconds'])>=0 for r in step_rows)
            assert abs(sum(float(r['seconds']) for r in step_rows)-float(step_rows[0]['step_seconds']))<1e-10
            capacity=1000 if step_rows[0]['plant']=='brusselator' else 100
            assert all(int(r['history_length'])==(int(index)-1)%capacity and int(r['history_length_after'])==int(index)%capacity for r in step_rows)
    return result


def remaining_cost_rows(raw):
    raw=Path(raw);rows=[]
    mutable=read_json(raw/'tensor_count_semantics.json')['mutable_returns_per_call']
    cases=read_json(raw/'remaining_profiles/COMPLETE.json')['cases']
    assert cases==['brusselator_before0995','brusselator_before1001','van_der_pol_before0100']
    for case in cases:
        root=raw/'remaining_profiles'/case
        source=read_json(root/'source.json')
        assert source['scientific_sha']==SCIENTIFIC_SHA
        assert source['prepared_remainder_replay'] is True and source['packed_boundary_execution'] is True
        assert source['affinity']==[2] and source['threads']==1
        assert source['harness_sha256']==hashlib.sha256((ROOT/'experiments/boundary_execution/remaining_profile.py').read_bytes()).hexdigest()
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in source['scientific_sources'].items())
        records=read_csv(root/'breakdown.csv')
        self_returns=defaultdict(int)
        for op in read_csv(root/'tensor_operations.csv'):
            self_returns[op['category']]+=int(op['calls'])*mutable.get(op['operation'],0)
        counts={r['category']:r for r in records if r['pass_kind']=='counts'}
        wall=[r for r in records if r['pass_kind']=='wall']
        assert len(wall)==len(counts)==9
        assert abs(sum(float(r['seconds']) for r in wall)-float(wall[0]['step_seconds']))<1e-10
        for row in wall:
            other=counts[row['category']]
            assert row['calls']==other['calls'] and row['interval_constructions']==other['interval_constructions']
            row.update(case=case, tensor_outputs=other['tensor_outputs'], explicit_copy_events=other['explicit_copy_events'],
                tensor_result_creations=int(other['tensor_outputs'])-self_returns[row['category']],
                fraction=float(row['seconds'])/float(row['step_seconds']))
            rows.append(row)
    return rows


def runtime_rows(raw):
    raw=Path(raw);rows=[]
    for command in read_json(raw/'commands.json'):
        assert command['module']=='run'
        s=verify_run(raw/command['name'],command,recompute_common=False)
        rows.append(dict(run=command['name'],purpose=command['purpose'],plant=s['plant'],mode=s['mode'],
            prepared_remainder_replay=s['prepared_remainder_replay'],packed_boundary_execution=s['packed_boundary_execution'],
            source_sha=s['scientific_sha'],setup_seconds=s['setup_seconds'],solve_seconds=s['solve_seconds'],
            export_seconds=s['export_seconds'],whole_process_seconds=command['whole_process_seconds'],
            peak_rss_bytes=s['peak_rss_bytes'],accepted_steps=s['accepted_steps'],rejected_attempts=s['rejected_attempts'],
            start_boundary=s['start_boundary'],actual_h_sum=str(Fraction(s['accepted_horizon_exact'])-Fraction(s['starting_time_exact'])),
            python=s['python'],torch=s['torch_version'],affinity=str(s['affinity']),threads=s['threads']))
    return rows


def decision(raw):
    raw=Path(raw);full={};short={};prefixes={};reverse={}
    for plant,prefix in [('brusselator','brusselator'),('van_der_pol','vdp')]:
        pairs=[compare_pair(raw/f'{prefix}_full_baseline',raw/f'{prefix}_full_candidate')]
        groups=[name for name,(p,_) in WINDOWS.items() if p==plant]+[f'{plant}_prefix100']
        for group in groups:
            short[group]=[compare_pair(raw/f'{group}_pair{i}_baseline',raw/f'{group}_pair{i}_candidate')['speedup'] for i in (1,2,3)]
        prefixes[plant]=short[f'{plant}_prefix100']
        directions=[v for group in groups for v in short[group]]+[pairs[0]['speedup']]
        near=plant=='brusselator' and 1.35<=pairs[0]['speedup']<=1.65
        conflict=min(directions)<1.<max(directions)
        reverse[plant]=dict(first_full_speedup=pairs[0]['speedup'],all_short_speedups=directions[:-1],
            near_target_predeclared_interval=[1.35,1.65],near_target=near,conflicting_directions=conflict,reverse_pair_required=near or conflict)
        if near or conflict:
            pairs.append(compare_pair(raw/f'{prefix}_full_reverse_baseline',raw/f'{prefix}_full_reverse_candidate'))
        for pair in pairs:
            for key in ['baseline','candidate']:
                s=read_json(raw/pair[key]/'summary.json')
                assert s['accepted_steps']==1000 and s['start_boundary']==0 and s['rejected_attempts']==0
        full[plant]=pairs
    assert reverse==read_json(raw/'reverse_pair_decisions.json')
    bruss=[p['speedup'] for p in full['brusselator']]
    vdp=[p['speedup'] for p in full['van_der_pol']]
    vdp_ok=min(vdp)>=1/1.1 and median(prefixes['van_der_pol'])>=1/1.1
    rss_ok=all(p['peak_rss_ratio']<=1.5 for pairs in full.values() for p in pairs)
    target=min(bruss)>=1.5 and vdp_ok and rss_ok
    useful=min(bruss)>1 and min(prefixes['brusselator'])>1
    status=('BOUNDARY_EXECUTION_PRESERVED__TARGET_SPEEDUP_OBSERVED' if target else
            'BOUNDARY_EXECUTION_PRESERVED__USEFUL_BELOW_TARGET' if useful else
            'BOUNDARY_EXECUTION_PRESERVED__NO_MATERIAL_SPEEDUP')
    adaptive=read_json(raw/'vdp_adaptive_candidate/summary.json')
    assert adaptive['completed']
    return dict(status=status,scientific_sha=SCIENTIFIC_SHA,parent_sha=PARENT_SHA,
        full_pairs=full,short_speedups=short,prefix100_speedups=prefixes,
        full_speedup_statistics={plant:dict(count=len(pairs),median=median([p['speedup'] for p in pairs]),
            minimum=min(p['speedup'] for p in pairs),maximum=max(p['speedup'] for p in pairs)) for plant,pairs in full.items()},
        timing_confidence={plant:'single_complete_pair_with_three_repeated_short_pairs' if len(pairs)==1 else 'two_complete_pairs_in_reverse_order_with_all_records_retained' for plant,pairs in full.items()},
        vdp_no_stable_slowdown_over_10_percent=vdp_ok,peak_rss_within_1p5=rss_ok,
        fixed_steps_compared_per_system=1000,observers=['published','common'],endpoint_components_compared=4000,tube_components_compared=4000,
        adaptive_accepted=adaptive['accepted_steps'],adaptive_rejected=adaptive['rejected_attempts'],
        adaptive_exact_h_sum=adaptive['accepted_horizon_exact'],adaptive_scheduler_time_hex=adaptive['scheduler_time_hex'],
        adaptive_speed_claim=False,prepared_in_both_modes=True,boundary_default_enabled=False,
        whole_solver_formally_proved=False,gpu_throughput_claim=False)
