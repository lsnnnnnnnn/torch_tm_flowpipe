from pathlib import Path
import json, os, subprocess, time
repo=Path(__file__).resolve().parent/'repo'
art=repo/'artifacts/runs/our_solver_replay_performance_20260907T074937Z'
sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
baseenv=os.environ.copy()
baseenv.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(repo/'src')+':'+str(repo))
groups=[('root','py11',['tests']),('torch_basis','py11',['experiments/first_order_followup/tests/test_torch_basis.py']),('diffreach_followup','diffreach312',['experiments/first_order_followup/tests/test_diffreach_projection.py','experiments/first_order_followup/tests/test_diffreach_parity.py']),('first_order_three_way','py11',['experiments/first_order_three_way/tests/test_benchmark.py']),('diffreach_support','diffreach312',['experiments/first_order_three_way/tests/test_diffreach_support.py']),('common_contract','py11',['experiments/three_way_common_contract/tests']),('comparison_repair','py11',['experiments/three_way_comparison_repair/tests']),('deep_study','py11',['experiments/three_tool_deep_study/tests'])]
commands={'complete_matrix':[], 'counting':'Only the eight isolated complete groups count toward totals; focused repetitions are excluded.'}
for name,envname,targets in groups:
    py=Path('/srv/local/shengenli/miniforge3/envs')/envname/'bin/python'
    rel='tests/'+name
    argv=['taskset','-c','5',str(py),'-m','pytest','-q','-p','no:cacheprovider',*targets,'--junitxml='+str(art/(rel+'.xml'))]
    commands['complete_matrix'].append(dict(group=name,source_sha=sha,source_clean_at_start=True,argv=argv,cwd=str(repo),environment={k:baseenv[k] for k in ['PYTHONPATH','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},python=str(py),log=rel+'.log',xml=rel+'.xml',exit=rel+'.exit'))
(art/'tests/commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for command in commands['complete_matrix']:
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==sha
    env=baseenv|{'PATH':str(Path(command['python']).parent)+os.pathsep+baseenv['PATH']}
    started=time.perf_counter()
    print(json.dumps({'group':command['group'],'status':'RUNNING','source_sha':sha}),flush=True)
    with (art/command['log']).open('x') as log:
        proc=subprocess.Popen(command['argv'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        (art/'tests/live_process.json').write_text(json.dumps({'group':command['group'],'pid':proc.pid,'source_sha':sha})+'\n')
        code=proc.wait()
    (art/command['exit']).write_text(str(code)+'\n')
    print(json.dumps({'group':command['group'],'exit_code':code,'wall_seconds':time.perf_counter()-started}),flush=True)
(art/'tests/live_process.json').unlink()
