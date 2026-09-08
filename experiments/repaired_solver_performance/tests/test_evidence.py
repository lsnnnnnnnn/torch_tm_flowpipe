"""Six distinct semantic alterations must fail after an outer manifest rehash."""
import gzip
import json
import os
from pathlib import Path
import shutil

import pytest

from experiments.endpoint_roundoff_repair.frozen import ROOT
from experiments.repaired_solver_performance.build import hashes
from experiments.repaired_solver_performance.verify import manifest, verify

PACKAGE=Path(os.environ.get('PREPARED_REPLAY_EVIDENCE',str(ROOT/'artifacts/runs/repaired_solver_performance_20260908T034636Z')))


def replace_bytes(path,content):
    # Copies below use hardlinks to avoid duplicating hundreds of MB. Atomic
    # replacement is essential: never write through a link to the original.
    temporary=path.with_name(path.name+'.changed')
    temporary.write_bytes(content);temporary.replace(path)


@pytest.fixture
def package(tmp_path):
    assert (PACKAGE/'SHA256SUMS').is_file(),'build the real complete artifact before these tests'
    target=tmp_path/'package'
    shutil.copytree(PACKAGE,target,copy_function=os.link)
    yield target


@pytest.mark.parametrize('kind',['endpoint_error','proposal','h','mode','timing','reused_as_fresh'])
def test_rehashed_semantic_change_is_rejected(package,kind):
    raw=package/'raw_minimal/formal'
    if kind=='endpoint_error':
        path=raw/'brusselator_full_optimized/endpoint_audit.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['substitution_error_hex'][0][0]='0x0.0p+0'
        replace_bytes(path,(''.join(json.dumps(r)+'\n' for r in rows)).encode())
    elif kind=='proposal':
        path=raw/'brusselator_early_replay/replays.jsonl.gz'
        rows=[json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
        # Change the same mathematical proposal in both stored columns: mere
        # agreement of two claimed outputs is insufficient; actual recompute
        # of the original evaluator must reject the altered result.
        for mode in ['reference','optimized']:
            rows[0][mode]['rows'][0]['proposed_remainder_lo'][0][0]-=1e-7
        replace_bytes(path,gzip.compress((''.join(json.dumps(r)+'\n' for r in rows)).encode()))
    elif kind=='h':
        path=raw/'brusselator_full_optimized/bounds.csv'
        import csv,io
        rows=list(csv.DictReader(io.StringIO(path.read_text())))
        rows[0]['h_hex']=float(.01).hex()
        text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        replace_bytes(path,text.getvalue().encode())
    elif kind=='mode':
        path=raw/'brusselator_full_optimized/source.json'
        data=json.loads(path.read_text());data['mode']='reference'
        replace_bytes(path,json.dumps(data).encode())
    elif kind=='timing':
        path=raw/'brusselator_full_optimized/summary.json'
        data=json.loads(path.read_text());data['solve_seconds']*=.5
        replace_bytes(path,json.dumps(data).encode())
    else:
        path=package/'SOURCE_MAP.json';data=json.loads(path.read_text())
        data['data_roles']['repaired_archive']='FRESH_REPAIRED_REFERENCE'
        replace_bytes(path,json.dumps(data).encode())
    (package/'SHA256SUMS').unlink()
    hashes(package)
    manifest(package)  # the rejection must not be just an outer hash mismatch
    with pytest.raises((AssertionError,ValueError)):
        verify(package)
