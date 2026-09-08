"""Finite sequential matched measurement schedule; never restart a partial job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.boundary_execution.compare import compare_pair, compare_parent_archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=2)
    parser.add_argument('--scientific-root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--checks',type=Path,required=True)
    args=parser.parse_args()
    root=args.scientific_root.resolve()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    for name in ['tests','state_checks']:
        completed=json.loads((args.checks/name/'COMPLETE.json').read_text())
        assert completed['scientific_sha']==sha
    assert json.loads((args.checks/'state_checks/COMPLETE.json').read_text())['jobs']==6
    args.output.mkdir(parents=True,exist_ok=True)
    commands_path=args.output/'commands.json'
    commands=json.loads(commands_path.read_text()) if commands_path.exists() else []
    env={**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    old=root/'artifacts/runs/repaired_solver_performance_20260908T034636Z/raw_minimal/formal'
    def gate(name,reference,optimized):
        result=compare_pair(args.output/reference,args.output/optimized)
        (args.output/(name+'_equivalence.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'gate':name,'bitwise_equal':True,'speedup':result['speedup']}),flush=True)
    def job(name,module,plant,*,mode=None,steps=20,checkpoint=None,adaptive=False,light=False,purpose):
        existing=next((r for r in commands if r['name']==name),None)
        if existing is not None:
            assert existing.get('exit_code')==0,f'Unfinished or failed job {name}; inspect its live handle before any restart'
            if module=='run':
                assert json.loads((args.output/name/'summary.json').read_text())['completed']
            return
        output=args.output/name
        assert not output.exists(),f'Unregistered output {output}; do not overwrite'
        argv=['taskset','-c',str(args.cpu),sys.executable,'-m',f'experiments.boundary_execution.{module}',
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
                'launcher_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
                'prepared_remainder_replay':True,'packed_boundary_execution':mode=='candidate',
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

    windows=[('brusselator_early','brusselator',None),
             ('brusselator_middle','brusselator',args.inputs/'brusselator100'),
             ('brusselator_late','brusselator',args.inputs/'brusselator980'),
             ('vdp_early','van_der_pol',None),
             ('vdp_boundary','van_der_pol',args.inputs/'vdp90')]
    short_groups={'brusselator':[],'van_der_pol':[]}
    for name,plant,checkpoint in windows:
        short_groups[plant].append(name)
        for repeat in (1,2,3):
            for mode in (['baseline','candidate'] if repeat%2 else ['candidate','baseline']):
                job(f'{name}_pair{repeat}_{mode}','run',plant,mode=mode,steps=20,
                    checkpoint=checkpoint,purpose='representative_20_step_pair')
            gate(f'{name}_pair{repeat}',f'{name}_pair{repeat}_baseline',f'{name}_pair{repeat}_candidate')
    for plant in ['brusselator','van_der_pol']:
        name=f'{plant}_prefix100'
        short_groups[plant].append(name)
        for repeat in (1,2,3):
            for mode in (['baseline','candidate'] if repeat%2 else ['candidate','baseline']):
                job(f'{name}_pair{repeat}_{mode}','run',plant,mode=mode,steps=100,purpose='matched_100_step_prefix')
            gate(f'{name}_pair{repeat}',f'{name}_pair{repeat}_baseline',f'{name}_pair{repeat}_candidate')
    reverse_decisions={}
    for plant,prefix in [('brusselator','brusselator'),('van_der_pol','vdp')]:
        for mode in ['baseline','candidate']:
            job(f'{prefix}_full_{mode}','run',plant,mode=mode,steps=1000,purpose='fresh_fixed_1000')
        gate(f'{prefix}_full',f'{prefix}_full_baseline',f'{prefix}_full_candidate')
        full=compare_pair(args.output/f'{prefix}_full_baseline',args.output/f'{prefix}_full_candidate')
        speeds=[compare_pair(args.output/f'{name}_pair{i}_baseline',args.output/f'{name}_pair{i}_candidate')['speedup']
                for name in short_groups[plant] for i in (1,2,3)]
        directions=speeds+[full['speedup']]
        conflicting=min(directions)<1.<max(directions)
        near=plant=='brusselator' and 1.35<=full['speedup']<=1.65
        reverse_decisions[plant]={'first_full_speedup':full['speedup'],'all_short_speedups':speeds,
            'near_target_predeclared_interval':[1.35,1.65], 'near_target':near,
            'conflicting_directions':conflicting,'reverse_pair_required':near or conflicting}
        (args.output/'reverse_pair_decisions.json').write_text(json.dumps(reverse_decisions,indent=2)+'\n')
        if near or conflicting:
            for mode in ['candidate','baseline']:
                job(f'{prefix}_full_reverse_{mode}','run',plant,mode=mode,steps=1000,purpose='conditional_reverse_full_pair')
            gate(f'{prefix}_full_reverse',f'{prefix}_full_reverse_baseline',f'{prefix}_full_reverse_candidate')
    job('vdp_adaptive_candidate','run','van_der_pol',mode='candidate',adaptive=True,
        purpose='frozen_adaptive_equivalence_no_speed_claim')
    bridges=[compare_parent_archive(args.output/name,old/archive) for name,archive in
             [('brusselator_full_baseline','brusselator_full_optimized'),
              ('vdp_full_baseline','vdp_full_optimized'),('vdp_adaptive_candidate','vdp_adaptive_optimized')]]
    (args.output/'parent_archive_equivalence.json').write_text(json.dumps(bridges,indent=2)+'\n')
    (args.output/'SCHEDULE_COMPLETED.json').write_text(json.dumps({'source_sha':sha,'jobs':len(commands)},indent=2)+'\n')


if __name__=='__main__':main()
