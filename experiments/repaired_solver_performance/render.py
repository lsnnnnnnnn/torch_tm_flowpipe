"""Standalone figures for exact widths, inclusive solve cost and diagnostics."""
from collections import defaultdict
from fractions import Fraction


def figures(root,data,diagnostics,flowstar):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'figure.dpi':150,'axes.grid':True,'grid.alpha':.18})
    colors=['#5c7d9a','#dd923c']

    def save(fig,name):
        fig.tight_layout();fig.savefig(out/f'{name}.png');fig.savefig(out/f'{name}.pdf');plt.close(fig)

    for plant in ('brusselator','van_der_pol'):
        for view in ('published','common'):
            fig,axes=plt.subplots(2,2,figsize=(9,5.3),sharex=True,sharey=True)
            for i,kind in enumerate(('endpoint','tube')):
                for j,component in enumerate(('x','y')):
                    ax=axes[i,j]
                    rows=[r for r in data['widths'] if (r['plant'],r['view'],r['kind'],r['component'])==(plant,view,kind,component)]
                    ax.plot([float(Fraction(r['t_end_exact'])) for r in rows],[r['width_ratio'] for r in rows],color=colors[0],lw=1)
                    ax.set_title(f'{kind}: {component} | 1000 exact bound pairs')
                    ax.set_ylim(.995,1.005);ax.set_xlabel('Time');ax.set_ylabel('Prepared / repaired width')
            fig.suptitle(f'{plant.replace("_"," ").title()} | {view} complete-object ranges',y=1.01)
            save(fig,f'{plant}_{view}_width_equivalence')
        fig,axes=plt.subplots(2,2,figsize=(9,5.3),sharex=True)
        for i,kind in enumerate(('endpoint','tube')):
            for j,component in enumerate(('x','y')):
                rows=[r for r in flowstar['widths'] if (r['plant'],r['kind'],r['component'])==(plant,kind,component)]
                axes[i,j].plot([float(Fraction(r['t_end_exact'])) for r in rows],[r['optimized_over_flowstar_width'] for r in rows],lw=1)
                axes[i,j].axhline(1,color='grey',ls=':');axes[i,j].set_title(f'{kind}: {component}')
                axes[i,j].set_xlabel('Time');axes[i,j].set_ylabel('Prepared / reused Flow* width')
        fig.suptitle(f'{plant.replace("_"," ").title()} | existing common observer; Flow* reused',y=1.01)
        save(fig,f'{plant}_flowstar_reused_widths')

    full=[r for r in data['timings_raw'] if r['stage']=='full']
    full.sort(key=lambda r:(r['plant'],r['execution_mode']!='reference'))
    labels=[('Bruss' if r['plant']=='brusselator' else 'VDP')+'\n'+('reference' if r['execution_mode']=='reference' else 'prepared') for r in full]
    fig,axes=plt.subplots(1,3,figsize=(12,4.4))
    for ax,key,title in zip(axes,('solve_seconds','setup_seconds','export_seconds'),('Full numerical solve (includes plan)','Initialization','Export and checkpoints')):
        values=[r[key] for r in full]
        bars=ax.bar(range(4),values,color=[colors[i%2] for i in range(4)])
        ax.bar_label(bars,fmt='%.3f',fontsize=8);ax.set_xticks(range(4),labels);ax.set_title(title);ax.set_ylabel('Seconds');ax.margins(y=.18)
    save(fig,'full_timing_categories')

    groups=defaultdict(list)
    for row in diagnostics['replay_work_counts']:groups[row['window']].append(row)
    keys=list(groups);labels=[k.replace('brusselator','Bruss').replace('van_der_pol','VDP').replace('_',' ') for k in keys]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,suffix,title in zip(axes,('fixed_polynomial_calls','fixed_polynomial_exclusive_seconds'),('Actual polynomial-method calls in refinement','Disjoint polynomial time in refinement')):
        for i,lane in enumerate(('reference','optimized')):
            values=[sum(r[lane+'_'+suffix] for r in groups[key]) for key in keys]
            ax.bar(np.arange(len(keys))+(i-.5)*.36,values,width=.36,label=lane,color=colors[i])
        ax.set_xticks(range(len(keys)),labels,rotation=25,ha='right');ax.set_title(title);ax.legend()
    axes[0].set_ylabel('Calls (nested method invocations are counted)')
    axes[1].set_ylabel('Exclusive seconds; diagnostic wrappers only')
    fig.suptitle('Same R sequence; VDP retains its existing specialized cache',y=1.01)
    save(fig,'fixed_preparation_repetition')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    selected=['brusselator_0101_0120','brusselator_0981_1000','van_der_pol_0091_0110']
    stages=['candidate','post_accept','history','boundary','other_solve']
    bottom=np.zeros(len(selected))
    for stage in stages:
        values=[]
        for window in selected:
            rows=[r for r in diagnostics['profile_windows'] if r['window']==window and r['execution_mode']=='prepared_remainder_replay']
            values.append(sum(r['exclusive_seconds'] for r in rows if r['stage']==stage)/sum(r['exclusive_seconds'] for r in rows))
        axes[0].bar(range(len(selected)),values,bottom=bottom,label=stage);bottom+=values
    axes[0].set_xticks(range(len(selected)),['Bruss 101–120','Bruss 981–1000','VDP 91–110'],rotation=15)
    axes[0].set_ylabel('Fraction of scoped solve');axes[0].set_title('Remaining optimized time, disjoint stages');axes[0].legend(fontsize=7)
    estimates=diagnostics['amdahl'];x=np.arange(len(estimates))
    formal={(r['plant'],r['start_step']):r['solve_speedup_median'] for r in data['timing_summary'] if r['stage']=='window'}
    predicted=[r['predicted_total_speedup'] for r in estimates]
    measured=[formal[(r['plant'],int(r['window'].split('_')[-2]))] for r in estimates]
    axes[1].bar(x-.18,predicted,.36,label='Amdahl from disjoint measured slice',color=colors[0])
    axes[1].bar(x+.18,measured,.36,label='Formal paired median',color=colors[1])
    axes[1].axhline(1,color='grey',ls=':');axes[1].set_xticks(x,labels,rotation=25,ha='right')
    axes[1].set_ylabel('Window total speedup');axes[1].set_title('Window estimate versus production');axes[1].legend(fontsize=7)
    save(fig,'remaining_time_and_amdahl')
