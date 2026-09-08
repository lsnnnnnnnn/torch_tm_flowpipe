"""Distinct semantic attacks remain invalid after the outer hashes are updated."""
import csv
import gzip
import json
from pathlib import Path
import shutil

import pytest
import torch

from experiments.repaired_solver_performance.verify import verify,verify_manifest,write_manifest

REPO=Path(__file__).resolve().parents[3]
BUNDLE=REPO/'artifacts/runs/repaired_solver_performance_20260908T084224Z'


@pytest.mark.parametrize('attack',['endpoint_error','proposal','h','mode','timing','reused_as_fresh'])
def test_rehashed_semantic_attack_is_rejected(tmp_path,attack):
    torch.set_num_threads(1)
    root=tmp_path/'bundle';shutil.copytree(BUNDLE,root)
    optimized=root/'raw_minimal/production/full_brusselator_prepared_remainder_replay'
    if attack=='endpoint_error':
        path=optimized/'endpoint_audit.jsonl'
        lines=path.read_text().splitlines();row=json.loads(lines[0])
        row['substitution_error_hex'][0]=['0x0.0p+0','0x0.0p+0']
        lines[0]=json.dumps(row);path.write_text('\n'.join(lines)+'\n')
        expected='E/remainder/state/queue/replay'
    elif attack=='proposal':
        path=root/'raw_minimal/replay_windows/brusselator_0001.jsonl.gz'
        with gzip.open(path,'rt') as handle:rows=[json.loads(line) for line in handle]
        rows[0]['reference_proposals'][0]['proposal'][0]['values'][0][0]='0x0.0p+0'
        with gzip.open(path,'wt') as handle:
            for row in rows:handle.write(json.dumps(row)+'\n')
        expected='same-input proposal mismatch'
    elif attack=='h':
        path=optimized/'bounds.csv'
        with path.open() as handle:rows=list(csv.DictReader(handle))
        rows[0]['h_hex']=float(.01).hex()
        with path.open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        expected='h encoding|audit h|fixed h'
    else:
        path=optimized/('source.json' if attack=='mode' else 'step_timings.json') if attack!='reused_as_fresh' else root/'SOURCE_MAP.json'
        value=json.loads(path.read_text())
        if attack=='mode':value['execution_mode']='reference';expected='identity|mode|summary/source'
        elif attack=='timing':value[0]['solve_seconds']+=1.;expected='raw timing event'
        else:value['reused']['flowstar']['data_origin']='FRESH_REPAIRED_REFERENCE';expected='reused Flow.*cannot become fresh'
        path.write_text(json.dumps(value)+'\n')
    write_manifest(root)
    verify_manifest(root)  # The rejection must come from meaning, not SHA256.
    with pytest.raises(ValueError,match=expected):verify(root,recompute=False,repository=REPO)
