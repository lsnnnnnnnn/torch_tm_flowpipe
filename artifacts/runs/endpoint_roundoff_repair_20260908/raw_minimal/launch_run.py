from pathlib import Path
import argparse,json,os,subprocess,time

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('plant')
p.add_argument('steps',type=int)
p.add_argument('--cpu',type=int,default=2)
p.add_argument('--adaptive',action='store_true')
a=p.parse_args()
repo=root/'repo'
sha='196a50e9131336d68df07ad0af353deca0092d19'
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
           PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src')+':'+str(repo))
output=root/'evidence/raw_minimal'/a.name
argv=['taskset','-c',str(a.cpu),'/srv/local/shengenli/miniforge3/envs/py11/bin/python','-m','experiments.endpoint_roundoff_repair.run',
      '--plant',a.plant,'--steps',str(a.steps),'--scientific-sha',sha,'--output',str(output)]
if a.adaptive:argv.append('--adaptive')
started=time.perf_counter()
with output.with_suffix('.log').open('x') as log:
    child=subprocess.Popen(argv,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    record=dict(name=a.name,pid=child.pid,scientific_sha=sha,argv=argv,output=str(output))
    output.with_suffix('.process.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'name':a.name,'pid':child.pid,'status':'RUNNING'}),flush=True)
    result=child.wait()
output.with_suffix('.exit').write_text(str(result)+'\n')
record.update(exit_code=result,wall_seconds=time.perf_counter()-started)
output.with_suffix('.process.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'name':a.name,'exit_code':result,'wall_seconds':record['wall_seconds']}),flush=True)
if (output/'summary.json').exists():
    summary=json.loads((output/'summary.json').read_text())
    print(json.dumps({k:summary[k] for k in ['plant','completed','accepted_steps','rejected_attempts','accepted_horizon_exact','solve_seconds','export_seconds','failure']},indent=2),flush=True)
