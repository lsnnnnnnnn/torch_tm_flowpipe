"""Two missing real windows, followed by independent serialized-input replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'candidate_source'
OUT=ROOT/'replay_windows'
env=os.environ.copy()
env.update(PYTHONPATH=f'{SOURCE}/src:{SOURCE}',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
records=[]
for plant,start in [('brusselator',981),('van_der_pol',91)]:
    target=OUT/f'{plant}_{start:04d}.jsonl.gz'
    argv=['taskset','-c','6',sys.executable,'-m','experiments.repaired_solver_performance.replay_evidence',
          '--plant',plant,'--steps','20','--checkpoint',str(ROOT/'fresh_reference'/plant/f'checkpoint_{start-1:04d}'),'--output',str(target)]
    before=time.perf_counter()
    with target.with_suffix('.log').open('x') as log:
        code=subprocess.call(argv,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(argv=argv,exit_code=code,whole_process_seconds=time.perf_counter()-before))
    (OUT/'late_commands.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records[-1]),flush=True)
    assert code==0
for target in sorted(OUT.glob('*.jsonl.gz')):
    argv=['taskset','-c','6',sys.executable,'-m','experiments.repaired_solver_performance.verify_replay',str(target),
          '--output',str(target.with_suffix('.verification.json'))]
    before=time.perf_counter()
    with target.with_suffix('.verification.log').open('x') as log:
        code=subprocess.call(argv,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(argv=argv,exit_code=code,whole_process_seconds=time.perf_counter()-before))
    (OUT/'late_commands.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records[-1]),flush=True)
    assert code==0
