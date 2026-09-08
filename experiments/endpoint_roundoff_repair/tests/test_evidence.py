"""Semantic tampering must fail even after recomputing every outer hash."""
import csv,hashlib,json,shutil
from pathlib import Path
import pytest,torch
from experiments.endpoint_roundoff_repair.verify import verify

REPO=Path(__file__).resolve().parents[3]
BUNDLE=REPO/'artifacts/runs/endpoint_roundoff_repair_20260908'


def manifest(root):
    rows=[f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}' for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256SUMS']
    (root/'SHA256SUMS').write_text('\n'.join(rows)+'\n')


@pytest.mark.parametrize('attack', ['remove_error','decimal_time','changed_h','reused_as_fresh','width','completion_time','result_status'])
def test_rehashed_semantic_tampering_is_rejected(tmp_path,attack):
    torch.set_num_threads(1)
    root=tmp_path/'evidence';shutil.copytree(BUNDLE,root)
    if attack=='remove_error':
        p=root/'raw_minimal/vdp_full/endpoint_audit.jsonl'
        lines=p.read_text().splitlines();row=json.loads(lines[0]);row['substitution_error_hex']=[['0x0.0p+0','0x0.0p+0']]*2
        lines[0]=json.dumps(row);p.write_text('\n'.join(lines)+'\n')
    elif attack in ['decimal_time','changed_h','width']:
        p=root/('widths_full_prefix.csv' if attack=='width' else 'raw_minimal/vdp_full/bounds.csv')
        rows=list(csv.DictReader(p.open()))
        if attack=='decimal_time':rows[0]['t_end_exact']='1/100'
        elif attack=='changed_h':rows[0]['h_hex']=float(.02).hex()
        else:rows[0]['repaired_width']=str(float(rows[0]['repaired_width'])+1.)
        with p.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    else:
        p=root/('SOURCE_MAP.json' if attack=='reused_as_fresh' else 'raw_minimal/vdp_full/summary.json' if attack=='completion_time' else 'RESULT.json')
        data=json.loads(p.read_text())
        if attack=='reused_as_fresh':data['data_roles']['flowstar']='FRESH_ENDPOINT_REPAIRED'
        elif attack=='completion_time':data['accepted_horizon']=9.
        else:data['status']='ENDPOINT_REPAIR_CLOSED__FULL_RUN_BEHAVIOR_CHANGED'
        p.write_text(json.dumps(data)+'\n')
    manifest(root)
    with pytest.raises((ValueError,AssertionError),match='correction|time|table|relabeled|summaries|status|binary64|step|interval'):
        verify(root,REPO)
