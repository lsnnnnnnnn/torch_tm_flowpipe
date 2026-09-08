"""Bounded derivation from this experiment's saved production records."""
import csv
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import statistics

from experiments.repaired_solver_performance.compare import compare_runs,exact,read,read_run,require,validate_run

PLANTS=('brusselator','van_der_pol')
WINDOWS=(('brusselator',1),('brusselator',101),('brusselator',981),('van_der_pol',1),('van_der_pol',91))
BASE='7e41f33f515f5315b0dec7003a5b06ed0a79afd5'
REFERENCE='e2d00f1b9e1824e8591f6655397bfc3762d6d0ed'
CANDIDATE='f627d6489155c775d4a7f3afde27c88982494ba2'
RUNTIME='ae6e21aefcfcebb20d616b19914a2240fca47e6a'
REPAIRED='0714e475ed9e73bec31619c9c690d1fd63de3d36'
OLD_PACKAGE='artifacts/runs/endpoint_roundoff_repair_20260908'


def csv_rows(path):
    with Path(path).open() as handle:return list(csv.DictReader(handle))


def write_csv(path,rows):
    with Path(path).open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def compare_archive(archive,new_path):
    """Historical repair is a decision/width anchor, never a fresh denominator."""
    new=read_run(new_path);validate_run(new)
    old_summary=read(archive/'summary.json')
    require(old_summary['scientific_sha']=='196a50e9131336d68df07ad0af353deca0092d19','reused repair scientific identity')
    require(old_summary['data_origin']=='FRESH_ENDPOINT_REPAIRED','historical source bytes must retain original labels')
    for key in ('accepted_steps','rejected_attempts','accepted_horizon_exact','scheduler_time_hex','completed'):
        require(old_summary[key]==new['summary'][key],f'reused repaired decisions differ: {key}')
    old_bounds=csv_rows(archive/'bounds.csv')
    old_audits=[json.loads(line) for line in (archive/'endpoint_audit.jsonl').read_text().splitlines()]
    require(exact(old_bounds)==exact(new['bounds']),'reused repaired all-step bounds/h/scheduler differ')
    for a,b in zip(old_audits,new['audits'],strict=True):
        require(exact(a)==exact({k:b[k] for k in a}),'reused repaired E/remainder/normal state/queue/replay decisions differ')
    with gzip.open(archive/'models.jsonl.gz','rt') as a,gzip.open(new_path/'models.jsonl.gz','rt') as b:
        for left,right in zip(a,b,strict=True):
            require(exact(json.loads(left))==exact(json.loads(right)),'reused repaired full model mismatch')
    return dict(archive_data_origin='REUSED_ENDPOINT_REPAIRED',new_data_origin=new['source']['data_origin'],
                new_scientific_sha=new['source']['scientific_sha'],steps=len(old_bounds),rejections=new['summary']['rejected_attempts'],
                actual_sum_h=new['summary']['accepted_horizon_exact'],scheduler_time_hex=new['summary']['scheduler_time_hex'],
                all_h_bounds_models_E_normal_state_queue_replay_bit_identical=True,speed_claim=False)


def log_terminal(path):
    """The final printed summary duplicates export time independently of tables."""
    lines=path.read_text().splitlines()
    start=max(i for i,line in enumerate(lines) if line=='{')
    terminal=json.loads('\n'.join(lines[start:]))
    progress=[json.loads(line) for line in lines if line.startswith('{"accepted_steps":')]
    return terminal,progress


def production_records(root):
    raw=root/'raw_minimal';rows=[];runs={}
    reference_commands=read(raw/'fresh_reference/commands.json')
    require(len(reference_commands)==2 and {c['plant'] for c in reference_commands}==set(PLANTS),'fresh full reference command coverage')
    candidate_commands=read(raw/'production/commands.json')
    require(len(candidate_commands)==45,'finite preregistered production sequence incomplete')
    complete=read(raw/'production/COMPLETE.json')
    require(complete==dict(source_sha=CANDIDATE,production_runs=45,matched_short_pairs=21),'production completion identity')
    ordered=[]
    for command in reference_commands:
        plant=command['plant'];ordered.append((f'full_{plant}_reference','full',command,raw/'fresh_reference'/plant,raw/'fresh_reference'/f'{plant}.log'))
    for command in candidate_commands:
        name=command['name'];ordered.append((name,command['stage'],command,raw/'production'/name,raw/'production'/f'{name}.log'))
    for name,stage,command,directory,log in ordered:
        require(command['exit_code']==0,'nonzero production exit')
        run=read_run(directory);validate_run(run);summary=run['summary'];source=run['source']
        expected=REFERENCE if name.endswith('_reference') and stage=='full' else CANDIDATE
        require(source['scientific_sha']==expected,'production scientific source role')
        require(source['affinity']==[3],'formal CPU affinity changed')
        argv=command['argv']
        for option,value in [('--plant',summary['plant']),('--mode',source['execution_mode']),('--scientific-sha',expected)]:
            require(option in argv and argv[argv.index(option)+1]==value,f'command versus source: {option}')
        require(command['cwd']==source['source_root'],'actual production working/import root')
        require(command['whole_process_seconds']>=summary['inside_process_seconds'],'whole process shorter than inside process')
        require(command['finish_unix']>command['start_unix'],'invalid UTC timing events')
        require(abs((command['finish_unix']-command['start_unix'])-command['whole_process_seconds'])<.02,'whole-process clock disagreement')
        terminal,progress=log_terminal(log)
        for key,value in terminal.items():require(exact(value)==exact(summary[key]),f'log/summary timing or completion disagreement: {key}')
        if progress and progress[-1]['accepted_steps']==int(run['bounds'][-1]['step']):
            for key in ('solve_seconds','export_seconds'):
                require(exact(progress[-1][key])==exact(summary[key]),'last progress timing event differs')
        work=summary.get('execution_work_totals',{})
        rows.append(dict(run=name,stage=stage,plant=summary['plant'],execution_mode=source['execution_mode'],
            data_origin=source['data_origin'],source_sha=expected,config_digest=source['config_digest'],
            source_root=source['source_root'],imported_package=source['imported_package'],python=source['python'],
            python_version=source['python_version'],torch_version=source['torch_version'],affinity='3',threads=1,dtype='float64',
            start_step=summary.get('start_step',1),steps=summary['accepted_steps'],rejections=summary['rejected_attempts'],
            actual_sum_h=summary.get('advanced_time_exact',summary['accepted_horizon_exact']),
            setup_seconds=summary['setup_seconds'],checkpoint_load_seconds=summary.get('checkpoint_load_seconds',0.),
            plan_preparation_inside_solve_seconds=work.get('prepared_plan_setup_s',0.),
            solve_seconds=summary['solve_seconds'],numerical_with_setup_seconds=summary['numerical_with_setup_seconds'],
            export_seconds=summary['export_seconds'],inside_process_seconds=summary['inside_process_seconds'],
            whole_process_seconds=command['whole_process_seconds'],peak_rss_bytes=summary['peak_rss_bytes'],
            replay_calls=summary['refinement_totals']['post_accept_replay_calls'],
            prepared_plans=work.get('prepared_plan_count',0),prepared_operations=work.get('prepared_operation_count',0),
            prepared_operation_hits=work.get('prepared_operation_hits',0),
            start_unix=command['start_unix'],finish_unix=command['finish_unix'],
            load_before=json.dumps(command['load_before']),load_after=json.dumps(command['load_after'])))
        runs[name]=run
    # All production tasks share one CPU and must not overlap in wall clock.
    chronological=sorted(rows,key=lambda r:r['start_unix'])
    require(all(a['finish_unix']<=b['start_unix'] for a,b in zip(chronological,chronological[1:])),'overlapping formal production processes')
    return rows,runs


def timing_pairs(root):
    production=root/'raw_minimal/production';pairs=[];summaries=[]
    for stage,windows,count in [('window',WINDOWS,20),('prefix',[(p,1) for p in PLANTS],100)]:
        for plant,start in windows:
            group=[]
            for repetition in range(1,4):
                name=f'{stage}_{plant}_{start:04d}_{count:04d}_pair{repetition}'
                reference=production/f'{name}_reference';optimized=production/f'{name}_prepared_remainder_replay'
                result=compare_runs(reference,optimized);result.pop('widths')
                pairs.append(dict(name=name,stage=stage,plant=plant,start_step=start,count=count,repetition=repetition,
                    order=['reference','prepared_remainder_replay'] if repetition%2 else ['prepared_remainder_replay','reference'],**result))
                group.append(result)
            row=dict(stage=stage,plant=plant,start_step=start,steps=count,pairs=3)
            for key,values in [('reference_solve_seconds',[v['reference']['solve_seconds'] for v in group]),
                               ('optimized_solve_seconds',[v['optimized']['solve_seconds'] for v in group]),
                               ('solve_speedup',[v['solve_speedup'] for v in group]),('peak_rss_ratio',[v['peak_rss_ratio'] for v in group])]:
                row.update({key+'_min':min(values),key+'_median':statistics.median(values),key+'_max':max(values)})
            summaries.append(row)
    require(exact(pairs)==exact(read(production/'pairs.json')),'saved short pair derivation differs')
    commands=read(production/'commands.json')
    for pair in pairs:
        actual=[c['mode'] for c in commands if c['name'].startswith(pair['name']+'_')]
        require(actual==pair['order'],'alternating pair order differs')
    return pairs,summaries


def derive(root,repository):
    rows,runs=production_records(root)
    pairs,timing_summary=timing_pairs(root)
    full={};widths=[]
    for plant in PLANTS:
        reference=root/'raw_minimal/fresh_reference'/plant
        optimized=root/'raw_minimal/production'/f'full_{plant}_prepared_remainder_replay'
        result=compare_runs(reference,optimized);widths+=result.pop('widths')
        require(result['reference']['steps']==result['optimized']['steps']==1000,'full fixed coverage')
        require(result['reference']['rejections']==result['optimized']['rejections']==0,'fixed rejection behavior')
        result['confidence']='borderline_single_pair' if plant=='brusselator' and abs(result['solve_speedup']/1.5-1)<.1 else 'single_full_pair_with_repeated_windows_and_prefixes'
        require(exact(result)==exact(read(root/'raw_minimal/production'/f'full_{plant}_comparison.json')),'saved full comparison differs')
        full[plant]=result
    old=repository/OLD_PACKAGE/'raw_minimal'
    adaptive=compare_archive(old/'vdp_adaptive',root/'raw_minimal/production/adaptive_van_der_pol_prepared_remainder_replay')
    archive_bridges={plant:compare_archive(old/('brusselator_full' if plant=='brusselator' else 'vdp_full'),root/'raw_minimal/fresh_reference'/plant) for plant in PLANTS}
    bruss=full['brusselator']['solve_speedup']
    vdp_ratios=[r['solve_speedup_median'] for r in timing_summary if r['plant']=='van_der_pol']
    vdp_no_stable_slowdown=all(r>=1/1.1 for r in vdp_ratios)
    if bruss>=1.5 and vdp_no_stable_slowdown:
        status='REPAIRED_REFERENCE_PRESERVED__PREPARED_REPLAY_SPEED_TARGET_MET'
    elif bruss>1.05 and vdp_no_stable_slowdown:
        status='REPAIRED_REFERENCE_PRESERVED__USEFUL_SPEEDUP_BELOW_TARGET'
    else:status='REPAIRED_REFERENCE_PRESERVED__NO_USEFUL_SPEEDUP'
    result=dict(status=status,brusselator_full_solve_speedup=bruss,brusselator_target=1.5,
        brusselator_confidence=full['brusselator']['confidence'],van_der_pol_full_solve_speedup=full['van_der_pol']['solve_speedup'],
        van_der_pol_no_stable_slowdown_over_10_percent=vdp_no_stable_slowdown,
        all_fixed_steps_bit_identical=True,fixed_accepted_steps={p:1000 for p in PLANTS},
        fixed_rejections={p:0 for p in PLANTS},width_rows=len(widths),
        width_rows_per_view=len(widths)//2,adaptive=adaptive,
        peak_rss_ratios={p:full[p]['peak_rss_ratio'] for p in PLANTS},
        time_memory_tradeoff=any(full[p]['peak_rss_ratio']>1.5 for p in PLANTS),
        mathematical_equivalence='binary64 equality; no tolerance; all endpoint/tube models, E, state/queue hashes and replay decisions',
        preparation_in_solve=True,adaptive_speed_claim=False,flowstar_fresh_speed_claim=False,
        whole_solver_formally_proved=False,gpu_backend_decided=False)
    full_summaries=dict(fixed_matched_pairs=full,
        fixed_run_records={name:run['summary'] for name,run in runs.items() if name.startswith('full_')},
        adaptive_run_record=runs['adaptive_van_der_pol_prepared_remainder_replay']['summary'],adaptive_archive_equivalence=adaptive)
    return dict(timings_raw=rows,timing_summary=timing_summary,pairs=pairs,full=full,full_summaries=full_summaries,widths=widths,
                archive_bridges=archive_bridges,result=result)
