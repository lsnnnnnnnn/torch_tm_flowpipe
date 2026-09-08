"""Replay one actual step after a saved 120-step repaired checkpoint."""
import argparse,gzip,itertools,json,subprocess
from pathlib import Path
import torch
from torch_tm_flowpipe import load_terminal_checkpoint,accepted_boundary_sr_queue_sha256
from torch_tm_flowpipe.brusselator_canonical_exchange import _append_tmv
from experiments.xiangru_adoption.common import from_existing_canonical
from experiments.endpoint_roundoff_repair.frozen import ROOT,step

parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
git=lambda *a:subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
assert not git('status','--porcelain')
assert git('rev-parse','HEAD')=='0714e475ed9e73bec31619c9c690d1fd63de3d36'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
checkpoint=args.checkpoint or args.run/'checkpoint_0120'
restored=load_terminal_checkpoint(checkpoint,expected_dtype='float64')
plant=restored.contract['plant']
index=restored.normal_state.step_index+1
with gzip.open(args.run/'models.jsonl.gz','rt') as source:
    original=json.loads(next(itertools.islice(source,index-1,index)))
assert original['step']==index
audit=next(json.loads(line) for line in (args.run/'endpoint_audit.jsonl').read_text().splitlines() if json.loads(line)['step']==index)
before=accepted_boundary_sr_queue_sha256(restored.normal_state.symbolic_queue)
result=step(plant,restored.current,restored.normal_state,index)
assert result.status=='validated',result.message
checks={}
for name,model in [('endpoint',result.endpoint_raw_tm),('tube',result.tm)]:
    values=[]
    _append_tmv(values,name,model,variable_order=('ux','uy','tau') if model.n_vars==3 else ('ux','uy'))
    canonical=from_existing_canonical(dict(values),name)
    checks[name]=canonical==original['models'][name]
    assert checks[name],name
after=accepted_boundary_sr_queue_sha256(result.flowstar_normal_state.symbolic_queue)
assert after==audit['queue_sha256']
assert before==accepted_boundary_sr_queue_sha256(restored.normal_state.symbolic_queue)
data={'scientific_sha':git('rev-parse','HEAD'),'plant':plant,'loaded_boundary':index-1,'resumed_step':index,
      'checkpoint':str(checkpoint),'continuous_run':args.run.name,'bit_identical_models':checks,
      'continuous_and_resumed_queue_identical':True,'input_state_unchanged':True,
      'queue_before':before,'queue_after':after,'passed':True}
with args.output.open('x') as output:json.dump(data,output,indent=2);output.write('\n')
print(json.dumps(data,indent=2))
