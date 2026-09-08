"""Assemble only measured results, figures and the bounded evidence package."""
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
import subprocess
import xml.etree.ElementTree as ET

from experiments.endpoint_roundoff_repair.frozen import ROOT, MATCHED_SHA256
from experiments.xiangru_adoption.common import measure
from experiments.repaired_solver_performance.analyze import (
    BASE_SHA, RUNTIME_SHA, SCIENTIFIC_SHA, FULL_NAMES, WINDOWS, REUSED_INPUT_PATHS, read_json, read_csv,
    write_json, write_csv, timing_tables, full_width_rows, profile_tables, decision, flowstar_widths,
)
from experiments.repaired_solver_performance.verify import check_replay_file, sha


def hashes(root):
    root=Path(root)
    lines=[f'{sha(p)}  {p.relative_to(root)}' for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (root/'SHA256SUMS').write_text('\n'.join(lines)+'\n')


def collect_formal(source,target):
    target.mkdir(parents=True,exist_ok=True)
    commands=read_json(source/'commands.json')
    assert (source/'SCHEDULE_COMPLETED.json').is_file(),'finite scientific schedule is not finished'
    for command in commands:
        assert command['exit_code']==0
        folder=target/command['name'];folder.mkdir(exist_ok=True)
        # The numerical records and all actual timing inputs are preserved.
        # Most repeated checkpoints are redundant; only the new required input
        # boundary980 is copied from the full reference below.
        for path in (source/command['name']).iterdir():
            if path.is_file() and path.suffix!='.pstats':shutil.copy2(path,folder/path.name)
        for suffix in ['.log','.exit']:shutil.copy2(source/(command['name']+suffix),target/(command['name']+suffix))
    for name in ['commands.json','SCHEDULE_COMPLETED.json','repaired_archive_equivalence.json','vdp90_source.json']:
        shutil.copy2(source/name,target/name)
    shutil.copytree(source/'brusselator_full_reference/checkpoint_0980',target/'brusselator_full_reference/checkpoint_0980',dirs_exist_ok=True)


def figures(output,widths,timings,work,remaining,amdahl,flow):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25,'figure.dpi':130})
    directory=output/'figures';directory.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(directory/(name+'.png'));fig.savefig(directory/(name+'.pdf'));plt.close(fig)
    for plant in ['van_der_pol','brusselator']:
        fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
        for ax,(metric,dim) in zip(axes.flat,[(m,d) for m in ['endpoint','tube'] for d in ['x','y']]):
            selected=[r for r in widths if r['plant']==plant and r['view']=='common' and r['metric']==metric and r['component']==dim]
            ax.plot([float(Fraction(r['t_end_exact'])) for r in selected],[r['width_ratio'] for r in selected])
            ax.set(title=f'{metric} {dim}',ylim=(.999,1.001),ylabel='optimized / repaired reference',xlabel='time')
        fig.suptitle(f'{plant}: all 1000 steps, bit-identical bounds')
        save(fig,plant+'_full_width_equivalence')
        fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
        for ax,(metric,dim) in zip(axes.flat,[(m,d) for m in ['endpoint','tube'] for d in ['x','y']]):
            selected=[r for r in flow if r['plant']==plant and r['metric']==metric and r['component']==dim]
            ax.plot([float(Fraction(r['t_end_exact'])) for r in selected],[r['ratio'] for r in selected])
            ax.axhline(1.,color='gray',ls='--');ax.set(title=f'{metric} {dim}',ylabel='width ratio',xlabel='time')
        fig.suptitle(f'{plant}: optimized / REUSED_MATCHED_REFERENCE Flow*')
        save(fig,plant+'_reused_flowstar_widths')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    selected=[r for r in timings if r['run'] in FULL_NAMES and not r['run'].startswith('vdp_adaptive')]
    labels=['Bruss ref','Bruss opt','VDP ref','VDP opt']
    selected=sorted(selected,key=lambda r:FULL_NAMES.index(r['run']))
    for ax,field,title in zip(axes,['setup_seconds','solve_seconds','export_seconds'],['Initialization + provenance','Numerical solve (includes plans)','Model / trace / checkpoint export']):
        values=[r[field] for r in selected];ax.bar(labels,values,color=['#345995','#e8a838']*2)
        ax.set(title=title,ylabel='seconds');ax.tick_params(axis='x',rotation=25)
        for i,value in enumerate(values):ax.text(i,value,f'{value:.3f}',ha='center',va='bottom',fontsize=8)
    save(fig,'full_solve_setup_export_times')
    names=list(WINDOWS)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for offset,mode,color in [(-.18,'reference','#345995'),(.18,'optimized','#e8a838')]:
        counts=[sum(r['polynomial_multiplications_executed_in_replay'] for r in work if r['window']==w and r['mode']==mode) for w in names]
        times=[sum(r['fixed_replay_seconds'] for r in work if r['window']==w and r['mode']==mode) for w in names]
        for ax,values in zip(axes,[counts,times]):ax.bar([i+offset for i in range(len(names))],values,.36,label=mode,color=color)
    for ax in axes:ax.set_xticks(range(len(names)),names,rotation=30,ha='right');ax.legend()
    axes[0].set(ylabel='actual polynomial multiplications',title='Fixed multiplication work in each 20-step replay window')
    axes[1].set(ylabel='exclusive instrumented seconds',title='Fixed replay work: first preparation included')
    save(fig,'fixed_preparation_counts_and_time')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    xs=list(range(len(amdahl)))
    axes[0].bar([x-.18 for x in xs],[r['predicted_total_speedup'] for r in amdahl],.36,label='Amdahl, same light window')
    axes[0].bar([x+.18 for x in xs],[r['unprofiled_three_pair_speedup_median'] for r in amdahl],.36,label='unprofiled pair median')
    axes[0].set_xticks(xs,[r['window'] for r in amdahl],rotation=30,ha='right');axes[0].set_ylabel('total speedup');axes[0].legend(fontsize=8)
    selected=[r for r in remaining if r['window']=='brusselator_late']
    selected=sorted(selected,key=lambda r:r['seconds'],reverse=True)[:6]
    axes[1].barh([r['category'].replace('_',' ') for r in selected],[r['seconds'] for r in selected])
    axes[1].invert_yaxis();axes[1].set(title='Remaining optimized late-window costs',xlabel='exclusive instrumented seconds')
    save(fig,'amdahl_and_remaining_hotspots')


def assemble(work_root,output):
    work_root,output=Path(work_root),Path(output)
    formal=work_root/'evidence/formal'
    assert read_json(formal/'SCHEDULE_COMPLETED.json')['source_sha']==SCIENTIFIC_SHA
    output.mkdir(parents=True,exist_ok=True)
    raw=output/'raw_minimal/formal';collect_formal(formal,raw)
    checkpoint=work_root/'evidence/measurements/prepare_vdp_boundary90/checkpoint_0090'
    shutil.copytree(checkpoint,output/'raw_minimal/checkpoint_0090',dirs_exist_ok=True)
    for name in ['candidate_decision.json','RUN_CONTEXT.json']:
        if name=='RUN_CONTEXT.json':shutil.copy2(work_root/'evidence'/name,output/name)
        else:shutil.copy2(work_root/'evidence'/name,output/'raw_minimal/prototype_candidate_decision.json')
    for mode in ['reference','optimized']:
        shutil.copytree(work_root/f'evidence/prototype_light_{mode}',output/f'raw_minimal/prototype_light_{mode}',dirs_exist_ok=True)
    shutil.copy2(work_root/'GOAL_FROZEN.md',output/'raw_minimal/GOAL_FROZEN.md')
    shutil.copy2(work_root/'measurement_instrumentation_adjustment.json',output/'raw_minimal/measurement_instrumentation_adjustment.json')
    write_json(output/'SOURCE_MAP.json',{'base_sha':BASE_SHA,'runtime_sha':RUNTIME_SHA,'scientific_sha':SCIENTIFIC_SHA,
        'repaired_reference_runtime_sha':'0714e475ed9e73bec31619c9c690d1fd63de3d36',
        'repaired_archive_sha':'196a50e9131336d68df07ad0af353deca0092d19',
        'old_unrepaired_sha':'4939fb288c941a67f55cc191f4d75f8594692f47','flowstar_sha':'b85a3211748cb77b736fe4ad42ee02d8d2b81148',
        'matched_contracts_sha256':MATCHED_SHA256,'goal_sha256':sha(work_root/'GOAL_FROZEN.md'),
        'runtime_sources':read_json(raw/'brusselator_full_reference/source.json')['source_files'],
        'reused_input_hashes':{path:sha(ROOT/path) for path in REUSED_INPUT_PATHS},
        'data_roles':{'reference':'FRESH_REPAIRED_REFERENCE','optimized':'FRESH_OPTIMIZED','repaired_archive':'REUSED_ENDPOINT_REPAIRED',
                      'old_unrepaired':'OLD_UNREPAIRED_ARCHIVE','flowstar':'REUSED_MATCHED_REFERENCE'},
        'package_is_separate_from_scientific_commit':True,'native_flowstar_rerun':False,
        'scientific_source_root':str(work_root/'repo'),'new_branch':read_json(work_root/'bootstrap.json')['BRANCH']})
    commands=read_json(raw/'commands.json')
    production={c['name']:read_json(raw/c['name']/'summary.json') for c in commands if c['module']=='run'}
    write_json(output/'EXECUTION_CONTRACT.json',{name:s['config'] for name,s in production.items()})
    write_json(output/'full_run_summaries.json',{name:production[name] for name in FULL_NAMES})
    widths=full_width_rows(raw);write_csv(output/'full_width_equivalence.csv',widths)
    timings,summary=timing_tables(raw);write_csv(output/'timings_raw.csv',timings);write_csv(output/'timing_summary.csv',summary)
    profiles,work,remaining,amdahl=profile_tables(raw)
    for name,rows in [('profile_windows.csv',profiles),('replay_work_counts.csv',work),('remaining_hotspots.csv',remaining)]:write_csv(output/name,rows)
    write_json(output/'candidate_decision.json',{'mechanism':'attempt_local_prepared_polynomial_replay','fallback_hotspot_used':False,
        'selection_evidence':'raw_minimal/prototype_candidate_decision.json','same_workload_predictions':amdahl,
        'full_horizon_predictions_extrapolated_from_single_window':False})
    replay=[]
    for name in [*(w+'_replay' for w in WINDOWS),'vdp_adaptive_replay_initial','vdp_adaptive_replay_after99']:
        records,rounds=check_replay_file(raw/name/'replays.jsonl.gz',recompute=False)
        replay.append({'window':name.removesuffix('_replay'),'steps':len(records),'proposal_rounds':rounds,
                       'independent_loops_and_same_inputs_bitwise_equal':True})
    write_json(output/'same_input_replay_equivalence.json',replay)
    result=decision(raw);write_json(output/'RESULT.json',result)
    flow,flow_summary=flowstar_widths(raw)
    write_csv(output/'reused_flowstar_widths.csv',flow);write_json(output/'reused_flowstar_width_summary.json',flow_summary)
    figures(output,widths,timings,work,remaining,amdahl,flow)
    hashes(output)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(assemble(args.work_root,args.output),indent=2),flush=True)
