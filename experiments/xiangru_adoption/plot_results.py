"""Export standalone figures with the entire requested time axis visible."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read(p):
    with p.open() as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    figs=a.root/'figures';figs.mkdir(exist_ok=True)
    rows=read(a.root/'widths_full_prefix.csv');horizons=read(a.root/'horizon_matrix.csv')
    historical=json.loads((a.root/'raw_minimal/historical_plot_markers.json').read_text())['markers']
    colors={'ours':'#1670b7','flowstar':'#242424','strict':'#d05e14','parity':'#58823b'}
    labels={'ours':'Our CPU reference','flowstar':'Native Flow*',
            'strict':'Xiangru strict: short diagnostic','parity':'Xiangru parity: short diagnostic'}
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.18,'figure.dpi':130})
    for plant,title,end in [('brusselator','Brusselator',20),('van_der_pol','Van der Pol',10)]:
        fig,axes=plt.subplots(2,2,figsize=(11,6.5),sharex=True)
        for i,kind in enumerate(['endpoint','tube']):
          for j,dim in enumerate(['x','y']):
            ax=axes[i,j];rr=[r for r in rows if (r['plant'],r['view'],r['range_kind'],r['state'])==(plant,'published',kind,dim)]
            for lane in labels:
                got=[r for r in rr if r[lane+'_lo']!='']
                ts=[float(r['t_end']) for r in got]
                lo=[float(r[lane+'_lo']) for r in got];hi=[float(r[lane+'_hi']) for r in got]
                ax.plot(ts,lo,color=colors[lane],lw=.8,ls='--' if lane in ['strict','parity'] else '-',label=labels[lane])
                ax.plot(ts,hi,color=colors[lane],lw=.8,ls='--' if lane in ['strict','parity'] else '-')
                if lane in ['ours','flowstar']:
                    ax.fill_between(ts,lo,hi,color=colors[lane],alpha=.11)
                    if ts and ts[-1]<end-1e-10:ax.axvline(ts[-1],color='red',ls=':',label=f'{labels[lane]} stopped at {ts[-1]:g}')
            for mark in historical:
                if mark['plant']==plant:
                    ax.axvline(mark['time'],color='#8c6b47',ls='-.',lw=.8,label=mark['plot_label'])
            ax.set_title(f'{dim}: {kind} bounds');ax.set_xlim(0,end);ax.set_xlabel('Time')
        axes[0,0].legend(fontsize=7)
        fig.suptitle(title+' | Published bounds, complete requested time axis')
        fig.text(.5,.01,'Candidate curves stop after two diagnostic steps; exact defects prevent treating them as guaranteed enclosures.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.03,1,.95]);fig.savefig(figs/(plant+'_bounds.png'));plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,6.5),sharex=True)
        for i,kind in enumerate(['endpoint','tube']):
          for j,dim in enumerate(['x','y']):
            ax=axes[i,j];rr=[r for r in rows if (r['plant'],r['view'],r['range_kind'],r['state'])==(plant,'common',kind,dim)]
            for key,color,label in [('strict_vs_flowstar','#d05e14','Strict / Flow*'),('strict_vs_ours','#7d44ac','Strict / our CPU')]:
                got=[r for r in rr if r[key]!='']
                ax.plot([float(r['t_end']) for r in got],[float(r[key]) for r in got],color=color,marker='o',markersize=3,ls='--',label=label+' (diagnostic only)')
            ax.axhline(1.10,color='grey',ls=':',lw=.8)
            ax.set_title(f'{dim}: {kind} width');ax.set_xlim(0,end);ax.set_xlabel('Time');ax.set_ylabel('Width ratio')
            ax.text(.37,.53,'No full-prefix strict comparison:\nlong run stopped by exact correctness gate.',transform=ax.transAxes,fontsize=8)
        axes[0,0].legend(fontsize=7)
        fig.suptitle(title+' | Common-measurement ratios, full requested time axis')
        fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(figs/(plant+'_strict_ratios.png'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,plant,end in zip(axes,['brusselator','van_der_pol'],[20,10]):
        rr=[r for r in horizons if r['plant']==plant and r['step_policy']=='fixed']
        names=[];vals=[]
        for r in rr:
            name={'our_cpu_reference':'Our CPU','native_flowstar':'Flow*','xiangru_parity':'Parity (diagnostic)','xiangru_strict':'Strict (diagnostic)'}[r['mode']]
            v=float(r.get('accepted_horizon') or r.get('algorithm_accepted_diagnostic_horizon') or 0)
            names.append(name);vals.append(v)
        ax.barh(names,vals,color=['#1670b7','#242424','#58823b','#d05e14']);ax.set_xlim(0,end*1.12)
        for i,v in enumerate(vals):ax.text(v+end*.015,i,f'{v:g}',va='center')
        ax.set_title(plant.replace('_',' ').title());ax.set_xlabel('Actual executed fixed-step horizon')
    fig.text(.5,.01,'Adaptive + history: unsupported by candidate; our existing adaptive reference is retained and is not counted as a new fixed-step run.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(figs/'horizons.png');plt.close(fig)
    print(figs)

if __name__=='__main__':main()
