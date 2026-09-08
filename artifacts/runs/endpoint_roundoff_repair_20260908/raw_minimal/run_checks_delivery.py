from pathlib import Path
import argparse,json,os,subprocess,time

root=Path(__file__).resolve().parent
repo=root/'finish'
evidence=root/'evidence/tests/delivery'
evidence.mkdir(exist_ok=True)
old=Path('/srv/local/shengenli/our_solver_replay_performance_20260907T074937Z/repo')
parser=argparse.ArgumentParser()
parser.add_argument('matrix',choices=['local','full'])
args=parser.parse_args()
local=[('local','py11',[
    'tests/test_endpoint_roundoff_repair.py','tests/test_endpoint_roundoff_carry.py',
    'tests/test_our_reference_endpoint_containment.py','tests/test_taylor_model.py',
    'tests/test_polynomial.py','tests/test_batched_dense_tm.py',
    'tests/test_batched_dense_picard.py','tests/test_batched_dense_carry.py',
    'tests/test_batched_dense_remainder_validation.py','tests/test_accepted_boundary_sr.py',
    'tests/test_g2_shared_column.py','tests/test_bounded_source_ledger.py',
    'tests/test_s1_total_delta_contract.py','tests/test_c4_observer_lanes.py'],repo)]
groups=[('root','py11',['tests','--ignore=tests/test_our_solver_performance_evidence.py'],repo),
        ('old_stop_snapshot','py11',['tests/test_our_solver_performance_evidence.py'],old),
        ('torch_basis','py11',['experiments/first_order_followup/tests/test_torch_basis.py'],repo),
        ('diffreach_followup','diffreach312',['experiments/first_order_followup/tests/test_diffreach_projection.py','experiments/first_order_followup/tests/test_diffreach_parity.py'],repo),
        ('first_order_three_way','py11',['experiments/first_order_three_way/tests/test_benchmark.py'],repo),
        ('diffreach_support','diffreach312',['experiments/first_order_three_way/tests/test_diffreach_support.py'],repo),
        ('common_contract','py11',['experiments/three_way_common_contract/tests'],repo),
        ('comparison_repair','py11',['experiments/three_way_comparison_repair/tests'],repo),
        ('deep_study','py11',['experiments/three_tool_deep_study/tests'],repo)]
commands=[]
for name,envname,targets,cwd in (local if args.matrix=='local' else groups):
    sha=subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(cwd),'status','--porcelain'],text=True).strip()
    py=Path('/srv/local/shengenli/miniforge3/envs')/envname/'bin/python'
    env=os.environ.copy()
    env.update(DIFFREACH_ROOT='/srv/local/shengenli/DiffReach',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(cwd/'src')+':'+str(cwd),PATH=str(py.parent)+os.pathsep+env['PATH'])
    argv=['taskset','-c','5',str(py),'-m','pytest','-q','-p','no:cacheprovider',*targets,'--junitxml='+str(evidence/(name+'.xml'))]
    commands.append(dict(group=name,source_sha=sha,source_clean_at_start=True,cwd=str(cwd),argv=argv,
                         environment={k:env[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','DIFFREACH_ROOT','JAX_PLATFORMS']},
                         log='tests/delivery/'+name+'.log',xml='tests/delivery/'+name+'.xml',exit='tests/delivery/'+name+'.exit'))
    print(json.dumps({'group':name,'source_sha':sha,'status':'RUNNING'}),flush=True)
    started=time.perf_counter()
    with (evidence/(name+'.log')).open('x') as log:
        process=subprocess.Popen(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        (evidence/'live_process.json').write_text(json.dumps({'group':name,'pid':process.pid,'source_sha':sha})+'\n')
        code=process.wait()
    (evidence/(name+'.exit')).write_text(str(code)+'\n')
    commands[-1]['exit_code']=code
    (evidence/(args.matrix+'_commands.json')).write_text(json.dumps(commands,indent=2)+'\n')
    print(json.dumps({'group':name,'exit_code':code,'seconds':time.perf_counter()-started}),flush=True)
(evidence/'live_process.json').unlink()
