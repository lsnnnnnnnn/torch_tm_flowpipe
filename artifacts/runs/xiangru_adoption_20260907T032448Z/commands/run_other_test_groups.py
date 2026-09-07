from pathlib import Path
import os, json, subprocess, time
root=Path(__file__).resolve().parent
repo=root/'our_audit'
out=root/'other_test_groups';out.mkdir(exist_ok=False)
groups=[
 ('torch_basis','py11',['experiments/first_order_followup/tests/test_torch_basis.py']),
 ('diffreach_followup','diffreach312',['experiments/first_order_followup/tests/test_diffreach_projection.py','experiments/first_order_followup/tests/test_diffreach_parity.py']),
 ('first_order_three_way','py11',['experiments/first_order_three_way/tests/test_benchmark.py']),
 ('diffreach_support','diffreach312',['experiments/first_order_three_way/tests/test_diffreach_support.py']),
 ('common_contract','py11',['experiments/three_way_common_contract/tests']),
 ('comparison_repair','py11',['experiments/three_way_comparison_repair/tests']),
 ('deep_study','py11',['experiments/three_tool_deep_study/tests'])]
env=dict(os.environ)
env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(repo/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',JAX_PLATFORMS='cpu')
if Path('/srv/local/shengenli/DiffReach').is_dir():env['DIFFREACH_ROOT']='/srv/local/shengenli/DiffReach'
records=[]
for name,environment,paths in groups:
 command=['taskset','-c','4',f'/srv/local/shengenli/miniforge3/envs/{environment}/bin/python','-m','pytest','-q',*paths,'--junitxml='+str(out/(name+'.xml'))]
 start=time.time()
 with (out/(name+'.log')).open('w') as log:result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 records.append({'name':name,'argv':command,'returncode':result.returncode,'seconds':time.time()-start,'tested_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()})
 (out/'commands_and_results.json').write_text(json.dumps({'contract':'Seven isolated experiment groups from scripts/run_complete_pytest.sh; root tests run separately.','explicit_environment':{k:env[k] for k in ['PYTHONPATH','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','JAX_PLATFORMS','DIFFREACH_ROOT'] if k in env},'groups':records},indent=2)+'\n')
 print(name,result.returncode,flush=True)
