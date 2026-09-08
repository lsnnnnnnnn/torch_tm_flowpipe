"""The finite, preregistered matched production sequence for this experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.repaired_solver_performance.compare import compare_runs, read, require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--fresh-reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve();out=args.output.resolve()
    sha=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    require(not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(),'dirty scientific source')
    # This command is launched only after the reference supervisor has ended.
    reference_commands=read(args.fresh_reference/'commands.json')
    require({r['plant'] for r in reference_commands}=={'brusselator','van_der_pol'} and all(r['exit_code']==0 for r in reference_commands),'fresh reference processes must be complete')
    for plant in ('brusselator','van_der_pol'):
        require(read(args.fresh_reference/plant/'summary.json')['completed'],'fresh reference incomplete')
    out.mkdir(parents=True,exist_ok=False)
    commands=[];pairs=[]
    env=os.environ.copy()
    env.update(PYTHONPATH=f'{source}/src:{source}',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env['PATH']=str(Path(sys.executable).parent)+':'+env['PATH']

    def run(name,plant,count,mode,checkpoint=None,adaptive=False,stage='window'):
        directory=out/name
        require(not directory.exists(),'existing output must be inspected, never overwritten/restarted')
        argv=['taskset','-c','3',sys.executable,'-m','experiments.repaired_solver_performance.run',
              '--plant',plant,'--steps',str(count),'--mode',mode,'--scientific-sha',sha,'--output',str(directory)]
        if checkpoint is not None:argv+=['--checkpoint',str(checkpoint)]
        if adaptive:argv+=['--adaptive']
        before=time.perf_counter();utc=time.time();load=list(os.getloadavg())
        print(json.dumps(dict(event='START',run=name,mode=mode,source_sha=sha)),flush=True)
        with (out/f'{name}.log').open('x') as log:
            process=subprocess.Popen(argv,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
            (out/'active_process.json').write_text(json.dumps(dict(name=name,pid=process.pid,argv=argv,start_unix=utc),indent=2)+'\n')
            code=process.wait()
        row=dict(name=name,stage=stage,plant=plant,mode=mode,source_sha=sha,cwd=str(source),argv=argv,
                 start_unix=utc,finish_unix=time.time(),whole_process_seconds=time.perf_counter()-before,
                 load_before=load,load_after=list(os.getloadavg()),exit_code=code)
        commands.append(row)
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        (out/f'{name}.exit').write_text(str(code)+'\n')
        (out/'active_process.json').write_text(json.dumps(dict(name=name,pid=process.pid,terminal=True,exit_code=code),indent=2)+'\n')
        require(code==0,f'production command failed: {name}')
        summary=read(directory/'summary.json')
        print(json.dumps(dict(event='DONE',run=name,solve_seconds=summary['solve_seconds'],completed=summary['completed'])),flush=True)
        return directory

    def pairs_for(plant,start,count,stage):
        checkpoint=None if start==1 else args.fresh_reference/plant/f'checkpoint_{start-1:04d}'
        if checkpoint is not None:require(checkpoint.is_dir(),'required complete checkpoint missing')
        for repetition in range(3):
            order=('reference','prepared_remainder_replay') if repetition%2==0 else ('prepared_remainder_replay','reference')
            directories={}
            name=f'{stage}_{plant}_{start:04d}_{count:04d}_pair{repetition+1}'
            for mode in order:
                directories[mode]=run(f'{name}_{mode}',plant,count,mode,checkpoint,stage=stage)
            compared=compare_runs(directories['reference'],directories['prepared_remainder_replay'])
            compared.pop('widths')
            pairs.append(dict(name=name,stage=stage,plant=plant,start_step=start,count=count,repetition=repetition+1,
                              order=list(order),**compared))
            (out/'pairs.json').write_text(json.dumps(pairs,indent=2)+'\n')
            print(json.dumps(dict(event='EXACT_PAIR',name=name,speedup=compared['solve_speedup'])),flush=True)

    for plant,start in [('brusselator',1),('brusselator',101),('brusselator',981),('van_der_pol',1),('van_der_pol',91)]:
        pairs_for(plant,start,20,'window')
    for plant in ('brusselator','van_der_pol'):
        pairs_for(plant,1,100,'prefix')
    for plant in ('brusselator','van_der_pol'):
        directory=run(f'full_{plant}_prepared_remainder_replay',plant,1000,'prepared_remainder_replay',stage='full')
        compared=compare_runs(args.fresh_reference/plant,directory)
        compared.pop('widths')
        compared['confidence']='borderline_single_pair' if plant=='brusselator' and abs(compared['solve_speedup']/1.5-1)<.1 else 'single_full_pair_with_repeated_windows_and_prefixes'
        (out/f'full_{plant}_comparison.json').write_text(json.dumps(compared,indent=2)+'\n')
        print(json.dumps(dict(event='FULL_EXACT',plant=plant,**compared)),flush=True)
    run('adaptive_van_der_pol_prepared_remainder_replay','van_der_pol',1000,'prepared_remainder_replay',adaptive=True,stage='adaptive_consistency_no_speed_claim')
    (out/'COMPLETE.json').write_text(json.dumps(dict(source_sha=sha,production_runs=len(commands),matched_short_pairs=len(pairs)),indent=2)+'\n')


if __name__=='__main__':main()
