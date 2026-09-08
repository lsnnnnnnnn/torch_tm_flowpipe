"""Finite sequential matched measurement schedule; never restart a partial job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=2)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    args.output.mkdir(parents=True,exist_ok=True)
    commands_path=args.output/'commands.json'
    commands=json.loads(commands_path.read_text()) if commands_path.exists() else []
    env={**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    old=root/'artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal'
    def job(name,module,plant,*,mode=None,steps=20,checkpoint=None,adaptive=False,light=False,purpose):
        existing=next((r for r in commands if r['name']==name),None)
        if existing is not None:
            assert existing.get('exit_code')==0,f'Unfinished or failed job {name}; inspect its live handle before any restart'
            if module=='run':
                assert json.loads((args.output/name/'summary.json').read_text())['completed']
            return
        output=args.output/name
        assert not output.exists(),f'Unregistered output {output}; do not overwrite'
        argv=['taskset','-c',str(args.cpu),sys.executable,'-m',f'experiments.repaired_solver_performance.{module}',
              '--plant',plant,'--steps',str(steps),'--output',str(output)]
        if module=='run':argv+=['--scientific-sha',sha]
        if mode:argv+=['--mode',mode]
        if checkpoint:argv+=['--checkpoint',str(checkpoint)]
        if adaptive:argv+=['--adaptive']
        if light:argv+=['--light']
        log=args.output/(name+'.log')
        started=time.perf_counter()
        record={'name':name,'purpose':purpose,'module':module,'plant':plant,'mode':mode,'source_sha':sha,
                'argv':argv,'cwd':str(root),'import_path':str(root/'src'),'python':sys.executable,
                'affinity':[args.cpu],'environment':{k:env[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
                'log':log.name,'wall_started_utc':time.time(),'monotonic_start':started}
        with log.open('x') as out:
            process=subprocess.Popen(argv,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
            record['pid']=process.pid
            record['proc_start_ticks']=Path(f'/proc/{process.pid}/stat').read_text().split()[21]
            commands.append(record);commands_path.write_text(json.dumps(commands,indent=2)+'\n')
            print(json.dumps({'job':name,'pid':process.pid,'status':'RUNNING'}),flush=True)
            code=process.wait()
        record.update(exit_code=code,whole_process_seconds=time.perf_counter()-started,wall_finished_utc=time.time())
        commands_path.write_text(json.dumps(commands,indent=2)+'\n')
        (args.output/(name+'.exit')).write_text(str(code)+'\n')
        assert code==0,f'Job failed: {name}; see {log}'
        if module=='run':
            summary=json.loads((output/'summary.json').read_text())
            assert summary['completed'],f'Numerical job incomplete: {name}'
            record['solve_seconds']=summary['solve_seconds']
            commands_path.write_text(json.dumps(commands,indent=2)+'\n')
        print(json.dumps({'job':name,'status':'COMPLETED','whole_process_seconds':record['whole_process_seconds']}),flush=True)

    # The missing VDP boundary is reached once from a complete repaired state.
    job('prepare_vdp_boundary90','run','van_der_pol',mode='reference',steps=70,
        checkpoint=old/'vdp_full/checkpoint_0020',purpose='checkpoint_preparation_not_speed_denominator')
    vdp90=args.output/'prepare_vdp_boundary90/checkpoint_0090'

    def window(name,plant,checkpoint):
        for repeat in range(3):
            order=['reference','optimized'] if repeat%2==0 else ['optimized','reference']
            for mode in order:
                job(f'{name}_pair{repeat+1}_{mode}','run',plant,mode=mode,checkpoint=checkpoint,purpose='representative_20_step_pair')
        for mode in ['reference','optimized']:
            job(f'{name}_profile_{mode}','profile',plant,mode=mode,checkpoint=checkpoint,purpose='exclusive_profile_not_production_timing')
            job(f'{name}_light_{mode}','profile',plant,mode=mode,checkpoint=checkpoint,light=True,purpose='light_phase_timing_for_same_workload_Amdahl')
        job(f'{name}_replay','replay',plant,checkpoint=checkpoint,purpose='same_input_and_independent_replay_equivalence')

    window('brusselator_early','brusselator',None)
    window('brusselator_middle','brusselator',old/'brusselator_full/checkpoint_0100')
    window('vdp_early','van_der_pol',None)
    window('vdp_boundary','van_der_pol',vdp90)
    for plant in ['brusselator','van_der_pol']:
        for repeat in range(3):
            for mode in (['reference','optimized'] if repeat%2==0 else ['optimized','reference']):
                job(f'{plant}_prefix100_pair{repeat+1}_{mode}','run',plant,mode=mode,steps=100,purpose='matched_100_step_prefix')
    job('brusselator_full_reference','run','brusselator',mode='reference',steps=1000,purpose='fresh_fixed_1000')
    # No valid inherited boundary980 exists. The one full sequential reference
    # above supplies it; these windows never reinitialize from published boxes.
    window('brusselator_late','brusselator',args.output/'brusselator_full_reference/checkpoint_0980')
    job('brusselator_full_optimized','run','brusselator',mode='optimized',steps=1000,purpose='fresh_fixed_1000')
    for mode in ['reference','optimized']:
        job(f'vdp_full_{mode}','run','van_der_pol',mode=mode,steps=1000,purpose='fresh_fixed_1000')
    job('vdp_adaptive_optimized','run','van_der_pol',mode='optimized',adaptive=True,purpose='fresh_adaptive_equivalence_no_speed_claim')
    job('vdp_adaptive_replay_initial','replay','van_der_pol',steps=3,adaptive=True,purpose='adaptive_representative_decisions')
    job('vdp_adaptive_replay_after99','replay','van_der_pol',steps=3,adaptive=True,
        checkpoint=old/'vdp_adaptive/checkpoint_0099',purpose='adaptive_representative_decisions')
    (args.output/'SCHEDULE_COMPLETED.json').write_text(json.dumps({'source_sha':sha,'jobs':len(commands)},indent=2)+'\n')


if __name__=='__main__':main()
