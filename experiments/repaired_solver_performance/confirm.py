"""The two preregistered adjacent full pairs after the machine-load change."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.repaired_solver_performance.compare import compare_runs,read,require
from experiments.repaired_solver_performance.results import CANDIDATE,PLANTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();work=args.work_root.resolve();source=args.source.resolve()
    registration=read(work/'timing_confirmation_preregistration.json')
    require(read(work/'production/COMPLETE.json')['production_runs']==45,'original finite sequence must finish first')
    require(registration['new_full_pairs']==2 and registration['initial_full_result_not_yet_available'],'fixed pre-result confirmation plan')
    require(subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==CANDIDATE,'confirmation scientific SHA')
    require(not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(),'dirty confirmation source')
    out=work/'confirmation';out.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(PYTHONPATH=f'{source}/src:{source}',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env['PATH']=str(Path(sys.executable).parent)+':'+env['PATH']
    commands=[]
    for plant in PLANTS:
        directories={}
        for mode in registration['order'][plant]:
            name=f'confirm_full_{plant}_{mode}';directory=out/name
            argv=['taskset','-c','3',sys.executable,'-m','experiments.repaired_solver_performance.run',
                '--plant',plant,'--steps','1000','--mode',mode,'--scientific-sha',CANDIDATE,'--output',str(directory)]
            before=time.perf_counter();utc=time.time();load=list(os.getloadavg())
            print(json.dumps(dict(event='START',run=name)),flush=True)
            with (out/f'{name}.log').open('x') as log:
                process=subprocess.Popen(argv,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                (out/'active_process.json').write_text(json.dumps(dict(name=name,pid=process.pid,start_unix=utc),indent=2)+'\n')
                code=process.wait()
            row=dict(name=name,stage='confirmation_full',plant=plant,mode=mode,source_sha=CANDIDATE,cwd=str(source),argv=argv,
                start_unix=utc,finish_unix=time.time(),whole_process_seconds=time.perf_counter()-before,
                load_before=load,load_after=list(os.getloadavg()),exit_code=code)
            commands.append(row);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            (out/f'{name}.exit').write_text(str(code)+'\n')
            (out/'active_process.json').write_text(json.dumps(dict(name=name,pid=process.pid,terminal=True,exit_code=code),indent=2)+'\n')
            require(code==0,'confirmation child failed')
            summary=read(directory/'summary.json');directories[mode]=directory
            print(json.dumps(dict(event='DONE',run=name,solve_seconds=summary['solve_seconds'])),flush=True)
        compared=compare_runs(directories['reference'],directories['prepared_remainder_replay']);compared.pop('widths')
        compared['confidence']='borderline_single_pair' if plant=='brusselator' and abs(compared['solve_speedup']/1.5-1)<.1 else 'one_adjacent_confirmation_full_pair_with_repeated_windows_and_prefixes'
        (out/f'{plant}_comparison.json').write_text(json.dumps(compared,indent=2)+'\n')
        print(json.dumps(dict(event='EXACT_FULL_PAIR',plant=plant,**compared)),flush=True)
    (out/'COMPLETE.json').write_text(json.dumps(dict(source_sha=CANDIDATE,production_runs=4,full_pairs=2),indent=2)+'\n')


if __name__=='__main__':main()
