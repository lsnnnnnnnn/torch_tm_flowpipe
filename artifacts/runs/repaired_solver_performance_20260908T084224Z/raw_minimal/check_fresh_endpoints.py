import json
from pathlib import Path
import subprocess
import time
import torch
from experiments.repaired_solver_performance.verify import verify_endpoints

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1);torch.set_num_interop_threads(1)
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
for plant in ('brusselator','van_der_pol'):
    started=time.perf_counter()
    result=verify_endpoints(ROOT/'fresh_reference'/plant)
    result.update(helper_sha=sha,seconds=time.perf_counter()-started,long_ODE_solve=False)
    (ROOT/f'fresh_{plant}_endpoint_recomputation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
