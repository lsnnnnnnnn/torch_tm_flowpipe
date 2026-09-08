"""Only this optimization's six substantive evidence tamper cases."""
import gzip
import json
import os
from pathlib import Path
import shutil

import pytest

from experiments.boundary_execution.analyze import ROOT, verify_run
from experiments.boundary_execution.build import write_manifest
from experiments.boundary_execution.compare import compare_pair
from experiments.boundary_execution.verify import manifest, check_source, check_result


@pytest.fixture
def evidence(tmp_path):
    source = Path(os.environ.get('BOUNDARY_EXECUTION_EVIDENCE',
        ROOT/'artifacts/runs/boundary_execution_20260908T172756Z'))
    if not (source/'RESULT.json').is_file():
        pytest.skip('This evidence test requires the completed boundary execution package.')
    target = tmp_path/'evidence'
    shutil.copytree(source, target)
    manifest(target)
    return target


def change_json(path, function):
    value = json.loads(path.read_text()); function(value)
    path.write_text(json.dumps(value, indent=2)+'\n')


@pytest.mark.parametrize('field', ['numerical_coefficient', 'endpoint_error', 'actual_switch',
                                   'prepared_denominator_seconds', 'scientific_source', 'result_identity'])
def test_boundary_evidence_rejects_semantic_tampering_after_rehash(evidence, field):
    formal = evidence/'raw_minimal/formal'
    baseline, candidate = formal/'brusselator_full_baseline', formal/'brusselator_full_candidate'
    if field in {'numerical_coefficient', 'endpoint_error'}:
        check = lambda: compare_pair(baseline, candidate)
        check()
        if field == 'numerical_coefficient':
            path = candidate/'models.jsonl.gz'
            records = [json.loads(line) for line in gzip.open(path, 'rt')]
            records[0]['models']['endpoint']['components'][0]['terms'][0]['coefficient'] = ['0x0.0p+0']*2
            with gzip.open(path, 'wt') as out:
                for record in records:
                    out.write(json.dumps(record, separators=(',', ':'))+'\n')
        else:
            path = candidate/'endpoint_audit.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert records[0]['substitution_error_hex'][0] != ['0x0.0p+0']*2
            records[0]['substitution_error_hex'][0] = ['0x0.0p+0']*2
            path.write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in records))
    elif field in {'actual_switch', 'prepared_denominator_seconds'}:
        check = lambda: verify_run(baseline, recompute_common=False)
        check()
        if field == 'actual_switch':
            change_json(baseline/'source.json', lambda s: s.update(prepared_remainder_replay=False))
        else:
            change_json(baseline/'summary.json', lambda s: s.update(solve_seconds=s['solve_seconds']+500.))
    elif field == 'scientific_source':
        check = lambda: check_source(evidence)
        check()
        change_json(evidence/'SOURCE_MAP.json', lambda s: s.update(scientific_sha=s['parent_sha']))
    else:
        check = lambda: check_result(evidence)
        check()
        target = 'BOUNDARY_EXECUTION_PRESERVED__TARGET_SPEEDUP_OBSERVED'
        alternate = 'BOUNDARY_EXECUTION_PRESERVED__NO_MATERIAL_SPEEDUP'
        change_json(evidence/'RESULT.json', lambda s: s.update(status=alternate if s['status'] == target else target))
    write_manifest(evidence)
    manifest(evidence)  # Byte hashes pass: the semantic gate must still reject.
    with pytest.raises(AssertionError):
        check()
