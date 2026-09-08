"""Disjoint diagnostic summaries and explicit reuse of existing Flow* widths."""
from collections import defaultdict
from fractions import Fraction
import math
from pathlib import Path

from experiments.repaired_solver_performance.compare import read,require
from experiments.repaired_solver_performance.results import CANDIDATE,OLD_PACKAGE,PLANTS,WINDOWS,csv_rows
from experiments.xiangru_adoption.build_tables import weighted_quantile


def profile_tables(root):
    raw=root/'raw_minimal/matched_profiles';method=read(raw/'method.json')
    require(method['source_sha']==CANDIDATE and method['production_denominator'] is False,'diagnostic scientific/timing identity')
    steps=csv_rows(raw/'steps.csv');phases=csv_rows(raw/'phases.csv')
    require(len(steps)==600,'five windows, two modes, three instrumentation lanes, twenty steps')
    indexed={}
    for row in steps:
        key=(row['window'],row['execution_mode'],row['instrumentation'],int(row['step']))
        require(key not in indexed,'duplicate profile step');indexed[key]=row
    phase_groups=defaultdict(list)
    for row in phases:phase_groups[(row['window'],row['execution_mode'],int(row['step']))].append(row)
    for (window,mode,index),group in phase_groups.items():
        actual=sum(float(r['exclusive_seconds']) for r in group)
        recorded=float(indexed[(window,mode,'scoped',index)]['solve_seconds'])
        require(abs(actual-recorded)<1e-9,'exclusive profile scopes do not close to the measured step')
    aggregate=defaultdict(lambda:[0.,0])
    for row in phases:
        key=tuple(row[k] for k in ('window','plant','execution_mode','stage','category'))
        aggregate[key][0]+=float(row['exclusive_seconds']);aggregate[key][1]+=int(row['calls'])
    windows=[]
    for (window,plant,mode,stage,category),(seconds,calls) in sorted(aggregate.items()):
        total=sum(float(r['solve_seconds']) for r in steps if r['window']==window and r['execution_mode']==mode and r['instrumentation']=='scoped')
        windows.append(dict(window=window,plant=plant,execution_mode=mode,stage=stage,category=category,
            exclusive_seconds=seconds,calls=calls,exclusive_fraction=seconds/total,scoped_solve_seconds=total,
            production_speed_denominator=False))
    work=[];amdahl=[];hotspots=[]
    for plant,start in WINDOWS:
        window=f'{plant}_{start:04d}_{start+19:04d}'
        for index in range(start,start+20):
            a=indexed[(window,'reference','production_diagnostic',index)]
            b=indexed[(window,'prepared_remainder_replay','production_diagnostic',index)]
            require(a['replay_calls']==b['replay_calls'],'diagnostic path removed replay rounds')
            row=dict(window=window,plant=plant,step=index,history_before=int(a['history_before']),replay_calls=int(a['replay_calls']),
                optimized_plan_count=int(b['prepared_plan_count']),optimized_ordered_operations_prepared=int(b['prepared_operation_count']),
                optimized_ordered_operation_hits=int(b['prepared_operation_hits']),plan_setup_inside_diagnostic_solve_seconds=float(b['prepared_plan_setup_s']))
            for mode,label in [('reference','reference'),('prepared_remainder_replay','optimized')]:
                group=phase_groups[(window,mode,index)]
                fixed=[r for r in group if r['stage']=='post_accept' and r['category']=='refinement_fixed_polynomial']
                row[label+'_fixed_polynomial_calls']=sum(int(r['calls']) for r in fixed)
                row[label+'_fixed_polynomial_exclusive_seconds']=sum(float(r['exclusive_seconds']) for r in fixed)
                row[label+'_whole_refinement_scoped_seconds']=sum(float(r['exclusive_seconds']) for r in group if r['stage']=='post_accept')
            work.append(row)
        ref=[r for r in windows if r['window']==window and r['execution_mode']=='reference']
        opt=[r for r in windows if r['window']==window and r['execution_mode']=='prepared_remainder_replay']
        old_total=sum(r['exclusive_seconds'] for r in ref)
        old_slice=sum(r['exclusive_seconds'] for r in ref if r['stage']=='post_accept')
        new_slice=sum(r['exclusive_seconds'] for r in opt if r['stage']=='post_accept')
        f=old_slice/old_total;s=old_slice/new_slice
        a=sum(float(r['solve_seconds']) for r in steps if r['window']==window and r['execution_mode']=='reference' and r['instrumentation']=='production_diagnostic')
        b=sum(float(r['solve_seconds']) for r in steps if r['window']==window and r['execution_mode']=='prepared_remainder_replay' and r['instrumentation']=='production_diagnostic')
        amdahl.append(dict(window=window,plant=plant,selected_fraction=f,local_speedup_including_prepare_and_checks=s,
            predicted_total_speedup=1/((1-f)+f/s),max_possible_speedup=1/(1-f),diagnostic_uninstrumented_total_speedup=a/b,
            scoped_overhead_reference=old_total/a-1,scoped_overhead_optimized=sum(r['exclusive_seconds'] for r in opt)/b-1,
            claim='window diagnostic estimate, not a full-run prediction or formal speed denominator'))
        profile=csv_rows(raw/f'{window}_prepared_remainder_replay.csv')
        total=sum(float(r['self_seconds']) for r in profile)
        for rank,row in enumerate(sorted(profile,key=lambda r:-float(r['self_seconds']))[:25],1):
            hotspots.append(dict(window=window,plant=plant,rank=rank,file=row['file'],line=int(row['line']),function=row['function'],
                primitive_calls=int(row['primitive_calls']),total_calls=int(row['total_calls']),self_seconds=float(row['self_seconds']),
                self_fraction=float(row['self_seconds'])/total,cprofile_total_seconds=total,production_speed_denominator=False))
    return dict(profile_windows=windows,replay_work_counts=work,remaining_hotspots=hotspots,amdahl=amdahl)


def flowstar_tables(root,repository):
    # These bounds were already measured from complete Flow* objects by the
    # shared observer; their exact committed files are anchored in SOURCE_MAP.
    old=csv_rows(repository/OLD_PACKAGE/'widths_full_prefix.csv')
    old={(r['plant'],int(r['step']),r['range_kind'],r['state']):r for r in old if r['view']=='common'}
    widths=[];summary=[]
    for plant in PLANTS:
        fresh=csv_rows(root/'raw_minimal/production'/f'full_{plant}_prepared_remainder_replay/bounds.csv')
        for row in fresh:
            for kind in ('endpoint','tube'):
                for component in ('x','y'):
                    prior=old[(plant,int(row['step']),kind,component)]
                    require(prior['flowstar_role']=='REUSED_MATCHED_REFERENCE','Flow* reuse identity')
                    require(prior['t_end_exact']==row['t_end_exact'] and prior['t_start_exact']==row['t_start_exact'],'matched Flow* actual time')
                    prefix=f'common_{kind}_{component}'
                    lo,hi=float(row[prefix+'_lo']),float(row[prefix+'_hi'])
                    flo,fhi=float(prior['flowstar_lo']),float(prior['flowstar_hi'])
                    require(lo.hex()==float(prior['repaired_lo']).hex() and hi.hex()==float(prior['repaired_hi']).hex(),'reuse width bridge to saved repaired result')
                    widths.append(dict(plant=plant,step=int(row['step']),kind=kind,component=component,t_end_exact=row['t_end_exact'],
                        optimized_lo=lo,optimized_hi=hi,flowstar_lo=flo,flowstar_hi=fhi,
                        optimized_over_flowstar_width=(hi-lo)/(fhi-flo),data_origin='REUSED_MATCHED_REFERENCE',
                        duration=float(Fraction(row['t_end_exact'])-Fraction(row['t_start_exact']))))
        for kind in ('endpoint','tube'):
            for component in ('x','y'):
                selected=[r for r in widths if (r['plant'],r['kind'],r['component'])==(plant,kind,component)]
                worst=max(selected,key=lambda r:r['optimized_over_flowstar_width'])
                pairs=[(r['optimized_over_flowstar_width'],r['duration']) for r in selected]
                summary.append(dict(plant=plant,kind=kind,component=component,P50=weighted_quantile(pairs,.5),P95=weighted_quantile(pairs,.95),
                    maximum=worst['optimized_over_flowstar_width'],maximum_at_step=worst['step'],maximum_at_time=float(Fraction(worst['t_end_exact'])),
                    data_origin='REUSED_MATCHED_REFERENCE',fresh_flowstar_timing=False))
    return dict(widths=widths,summary=summary)
