"""Wait for the confirmed live B1 run, then execute VDP once on the same CPU."""
from pathlib import Path
import json
import os
import subprocess
import time

root=Path(__file__).resolve().parent
tracked=json.loads((root/'our_brusselator_process.json').read_text())
proc=Path('/proc')/str(tracked['pid'])
while proc.exists():
    try:
        if (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=tracked['start_ticks']:break
    except FileNotFoundError:break
    time.sleep(20)
summary=json.loads((root/'our_brusselator_full/summary.json').read_text())
if not summary['completed'] or summary['accepted_steps']!=1000:
    raise RuntimeError('Brusselator did not complete; inspect terminal evidence before continuing.')
out=root/'our_vdp_full'
if out.exists():raise FileExistsError('VDP output already exists; never start a duplicate run.')
command=['/usr/bin/time','-f','elapsed_seconds=%e peak_rss_kb=%M','-o',str(root/'our_vdp_process_time.txt'),
         'taskset','-c','2','/srv/local/shengenli/miniforge3/envs/py11/bin/python',
         str(root/'runner_vdp_scientific/experiments/xiangru_adoption/run_ours.py'),
         '--source',str(root/'our_optimized'),'--plant','van_der_pol','--steps','1000','--output',str(out)]
env=dict(os.environ)
env.update(PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
(root/'our_vdp_command.json').write_text(json.dumps({'argv':command,'explicit_environment':{k:env[k] for k in ['PYTHONPATH','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}},indent=2)+'\n')
with (root/'our_vdp_full.log').open('w') as log:
    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    (root/'our_vdp_supervised_process.json').write_text(json.dumps({'pid':process.pid,'started_unix':time.time()},indent=2)+'\n')
    returncode=process.wait()
(root/'our_vdp_exit.json').write_text(json.dumps({'returncode':returncode,'finished_unix':time.time()},indent=2)+'\n')
if returncode:raise RuntimeError('VDP runner failed; do not relabel or automatically restart it.')
print((out/'summary.json').read_text(),flush=True)
