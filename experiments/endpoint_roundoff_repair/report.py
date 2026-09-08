"""Tables/figures for the endpoint repair, using the existing object measurer."""
import argparse,json,math
from fractions import Fraction as F
from pathlib import Path
from experiments.xiangru_adoption.build_tables import read_csv,write_csv,native_rows,weighted_quantile

PLANTS=(('van_der_pol','vdp',10),('brusselator','brusselator',20))
ROLES={'old':'OLD_UNREPAIRED_ARCHIVE','repaired':'FRESH_ENDPOINT_REPAIRED','flowstar':'REUSED_MATCHED_REFERENCE'}


def derive(root,archive):
    widths=[];summaries=[];timings=[];runs={};impacts={}
    for plant,short,horizon in PLANTS:
        olddir=archive/'raw_minimal'/f'our_{short}_full'
        native=archive/'raw_minimal'/f'native_{short}'
        fresh=root/'raw_minimal'/f'{short}_full'
        new_summary=json.loads((fresh/'summary.json').read_text())
        runs[plant]=new_summary
        records={'old':read_csv(olddir/'bounds.csv'),'repaired':read_csv(fresh/'bounds.csv'),'flowstar':native_rows(native)}
        assert len(records['old'])==len(records['flowstar'])==1000
        indexed={key:{int(r['step']):r for r in values} for key,values in records.items()}
        alert_count=0
        for step in range(1,1001):
          old=indexed['old'][step]
          for view in ['published','common']:
           for kind in ['endpoint','tube']:
            for dim in ['x','y']:
                row={'plant':plant,'step':step,'view':view,'range_kind':kind,'state':dim,
                     't_start_exact':old['t_start_exact'],'t_end_exact':old['t_end_exact'],
                     't_start':float(old['t_start']),'t_end':float(old['t_end']),
                     'duration':float(F(old['t_end_exact'])-F(old['t_start_exact']))}
                for role,data in indexed.items():
                    source=data.get(step)
                    prefix=f'{view}_{kind}_{dim}'
                    lo=float(source[prefix+'_lo']) if source else None
                    hi=float(source[prefix+'_hi']) if source else None
                    row[role+'_lo']=lo;row[role+'_hi']=hi;row[role+'_width']=hi-lo if source else None
                    row[role+'_role']=ROLES[role]
                    if source and source['t_end_exact']!=old['t_end_exact']:
                        raise ValueError('actual binary64 time mismatch in width comparison')
                for denominator in ['old','flowstar']:
                    prefix='repaired_vs_'+denominator
                    valid=row['repaired_width'] is not None
                    row[prefix]=row['repaired_width']/row[denominator+'_width'] if valid and row[denominator+'_width']>1e-12 else None
                    row[prefix+'_absolute_width_difference']=row['repaired_width']-row[denominator+'_width'] if valid else None
                    row[prefix+'_lower_shift']=row['repaired_lo']-row[denominator+'_lo'] if valid else None
                    row[prefix+'_upper_shift']=row['repaired_hi']-row[denominator+'_hi'] if valid else None
                    row[prefix+'_center_shift']=(row['repaired_lo']+row['repaired_hi']-row[denominator+'_lo']-row[denominator+'_hi'])/2 if valid else None
                row['engineering_width_alert']=row['repaired_width'] is not None and abs(row['repaired_vs_old_absolute_width_difference'])>max(1e-12,.01*row['old_width'])
                alert_count+=int(row['engineering_width_alert'])
                widths.append(row)
        for view in ['published','common']:
         for kind in ['endpoint','tube']:
          for dim in ['x','y']:
           for denominator in ['old','flowstar']:
            comparison='repaired_vs_'+denominator
            selected=[r for r in widths if (r['plant'],r['view'],r['range_kind'],r['state'])==(plant,view,kind,dim)]
            compared=[r for r in selected if r['repaired_width'] is not None]
            ratios=[r for r in compared if r[comparison] is not None]
            row={'plant':plant,'view':view,'range_kind':kind,'state':dim,'comparison':comparison,
                 'accepted_steps':len(compared),'requested_steps':1000,'full_requested_coverage':len(compared)==1000,
                 'compared_duration':math.fsum(r['duration'] for r in compared),
                 'near_zero_denominator_steps':len(compared)-len(ratios)}
            if ratios:
                worst=max(ratios,key=lambda r:r[comparison])
                pairs=[(r[comparison],r['duration']) for r in ratios]
                row.update(P50=weighted_quantile(pairs,.5),P95=weighted_quantile(pairs,.95),
                           max_ratio=worst[comparison],max_at_time=worst['t_end'],max_at_step=worst['step'])
            for metric in ['absolute_width_difference','lower_shift','upper_shift','center_shift']:
                if compared:
                    worst=max(compared,key=lambda r:abs(r[comparison+'_'+metric]))
                    row['max_abs_'+metric]=abs(worst[comparison+'_'+metric])
                    row['max_abs_'+metric+'_at_time']=worst['t_end']
            row['engineering_width_alert_steps']=sum(r['engineering_width_alert'] for r in compared) if denominator=='old' else None
            summaries.append(row)
        for role,directory in [('old',olddir),('repaired',fresh),('flowstar',native)]:
            data=json.loads((directory/'summary.json').read_text())
            timings.append({'plant':plant,'mode':role,'data_role':ROLES[role],
                            'accepted_steps':data['accepted_steps'],'accepted_horizon':data['accepted_horizon'],
                            'solve_seconds':data['solve_seconds'],'export_seconds':data['export_seconds'],
                            'single_observed_run':True,
                            'source':('raw_minimal/'+directory.name+'/summary.json' if role=='repaired'
                                      else 'archive:artifacts/runs/xiangru_adoption_20260907T032448Z/raw_minimal/'+directory.name+'/summary.json')})
        impacts[plant]={'old_completed_computation_preserved':True,'old_unconditional_oracle_claim_withdrawn':True,
                        'counterexample_proves_fixed_trajectory_miss':False,'fixed_run_revalidated':new_summary['completed'],
                        'accepted_steps':new_summary['accepted_steps'],'actual_horizon_exact':new_summary['accepted_horizon_exact'],
                        'new_rejected_attempts':new_summary['rejected_attempts'],'new_refinement_totals':new_summary['refinement_totals'],
                        'old_fixed_refinement_counts':'not_recorded_in_frozen_full_run',
                        'engineering_width_alert_rows':alert_count,'whole_solver_formally_proved':False}
    adaptive=json.loads((root/'raw_minimal/vdp_adaptive/summary.json').read_text())
    runs['van_der_pol_adaptive']=adaptive
    timings.append({'plant':'van_der_pol','mode':'repaired_adaptive','data_role':ROLES['repaired'],
                    'accepted_steps':adaptive['accepted_steps'],'accepted_horizon':adaptive['accepted_horizon'],
                    'solve_seconds':adaptive['solve_seconds'],'export_seconds':adaptive['export_seconds'],
                    'single_observed_run':True,'source':'raw_minimal/vdp_adaptive/summary.json'})
    return {'widths_full_prefix.csv':widths,'width_summary.csv':summaries,'timings.csv':timings},runs,impacts


def figures(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    directory=root/'figures';directory.mkdir(exist_ok=True)
    colors={'old':'#ba7627','repaired':'#1270ad','flowstar':'#404040'}
    labels={'old':'Old CPU (unrepaired archive)','repaired':'CPU with endpoint correction','flowstar':'Flow* (reused matched reference)'}
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.18,'figure.dpi':140})
    for plant,short,horizon in PLANTS:
        for view in ['published','common']:
            fig,axes=plt.subplots(2,2,figsize=(11,6.5),sharex=True)
            for i,kind in enumerate(['endpoint','tube']):
             for j,dim in enumerate(['x','y']):
                ax=axes[i,j]
                selected=[r for r in rows if (r['plant'],r['view'],r['range_kind'],r['state'])==(plant,view,kind,dim)]
                for role in colors:
                    got=[r for r in selected if r[role+'_lo'] is not None]
                    ts=[r['t_end'] for r in got]
                    lo=[r[role+'_lo'] for r in got];hi=[r[role+'_hi'] for r in got]
                    ax.plot(ts,lo,color=colors[role],lw=.85,ls='--' if role=='old' else '-',label=labels[role])
                    ax.plot(ts,hi,color=colors[role],lw=.85,ls='--' if role=='old' else '-')
                ax.set_title(f'{kind}: {dim}');ax.set_xlim(0,horizon);ax.set_xlabel('Time')
            axes[0,0].legend(fontsize=6.8)
            fig.suptitle(f'{plant.replace("_"," ").title()} | {view} complete bounds')
            fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(directory/f'{short}_{view}_bounds.png');plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,6.5),sharex=True)
        for i,kind in enumerate(['endpoint','tube']):
         for j,dim in enumerate(['x','y']):
            ax=axes[i,j]
            selected=[r for r in rows if (r['plant'],r['view'],r['range_kind'],r['state'])==(plant,'common',kind,dim)]
            for comparison,label,color in [('repaired_vs_old','Repaired / archived CPU','#ba7627'),('repaired_vs_flowstar','Repaired / Flow*','#1270ad')]:
                got=[r for r in selected if r[comparison] is not None]
                ax.plot([r['t_end'] for r in got],[r[comparison] for r in got],label=label,color=color,lw=1)
            ax.axhline(1,color='grey',ls=':',lw=.7);ax.set_xlim(0,horizon);ax.set_xlabel('Time')
            ax.set_ylabel('Width ratio');ax.set_title(f'{kind}: {dim}')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{plant.replace("_"," ").title()} | Common complete-object width ratios')
        fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(directory/f'{short}_width_ratios.png');fig.savefig(directory/f'{short}_width_ratios.pdf');plt.close(fig)
        audit=[json.loads(line) for line in (root/f'raw_minimal/{short}_full/endpoint_audit.jsonl').read_text().splitlines()]
        fig,ax=plt.subplots(figsize=(9,3.5))
        for component,name in enumerate(['x','y']):
            magnitudes=[max(abs(float.fromhex(v)) for v in r['substitution_error_hex'][component]) for r in audit]
            ax.semilogy([float(F(r['t_end_exact'])) for r in audit],[v if v else np.nan for v in magnitudes],label=name,lw=1)
        ax.set_xlim(0,horizon);ax.set_xlabel('Time');ax.set_ylabel('Max absolute correction');ax.legend()
        ax.set_title(f'{plant.replace("_"," ").title()} | Added endpoint coefficient error')
        fig.tight_layout();fig.savefig(directory/f'{short}_endpoint_error.png');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args()
    tables,runs,impacts=derive(args.root,args.archive)
    for name,rows in tables.items():write_csv(args.root/name,rows)
    (args.root/'full_run_summaries.json').write_text(json.dumps(runs,indent=2)+'\n')
    (args.root/'historical_claim_impact.json').write_text(json.dumps(impacts,indent=2)+'\n')
    figures(args.root,tables['widths_full_prefix.csv'])
    print(json.dumps({name:len(rows) for name,rows in tables.items()}))
