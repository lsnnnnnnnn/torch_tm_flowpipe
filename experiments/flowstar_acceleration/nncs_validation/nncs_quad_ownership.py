from pathlib import Path
import sys,json,hashlib,time
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');E=N/'engine_sr_memory';sys.path.insert(0,str(E/'src'))
from flowstar_gpu.determinism import enable_determinism
enable_determinism()
import torch,pytest
from flowstar_gpu import cuda_kernels,tape_kernels,sr_kernels,sr_sum_kernels
assert cuda_kernels.available() and tape_kernels.available() and tape_kernels.valid_available() and sr_kernels.available() and sr_sum_kernels.available()
p=E/'tests/unit/test_sparse_graph_ownership.py';source=p.read_text();line='    monkeypatch.setattr(se, "VALIDATION_POLICY", validation_policy)';assert source.count(line)==1
# This worktree intentionally lacks optional deferred validation. Preserve every
# lifecycle assertion; replace only the missing policy switch by an assertion.
source=source.replace(line,'    assert validation_policy == "truncated"')
rows=[]
for support in ['structural','measured']:
 code=source.replace('monkeypatch.setattr(se, "SUPPORT_POLICY", "structural")',f'monkeypatch.setattr(se, "SUPPORT_POLICY", "{support}")');namespace={};exec(compile(code,str(p),'exec'),namespace)
 for plant in ['van_der_pol','brusselator']:
  t=time.perf_counter()
  with pytest.MonkeyPatch.context() as patch:
   namespace['test_public_outputs_survive_failure_recovery_and_frozen_lane_replays'](plant,'truncated',patch,None)
  torch.cuda.synchronize();rows.append({'plant':plant,'support_policy':support,'validation_policy':'truncated (unchanged 5d)','passed':True,'seconds':time.perf_counter()-t})
result={'status':'passed','tests':rows,'upstream_test_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'adaptation':'same lifecycle assertions, enable_determinism fixture invoked directly; unavailable deferred-policy toggle replaced by assertion of frozen truncated policy; additionally both structural and measured support tested'}
Path('ownership.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
