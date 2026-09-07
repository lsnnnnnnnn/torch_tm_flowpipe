"""Small exact range checks and direct tampering of the actual evidence pack."""
from fractions import Fraction
import json
from pathlib import Path
import shutil
import pytest

ROOT=Path(__file__).resolve().parents[1]
from experiments.xiangru_adoption.common import measure
from experiments.xiangru_adoption.verify_evidence import checksum, verify

ARTIFACTS=ROOT/'artifacts/runs/xiangru_adoption_20260907T032448Z'

def test_exact_range_even_power_and_interval_coefficient():
    model={'domain':[[-2.,1.]],'components':[{'remainder':[-.125,.25],
            'terms':[{'degrees':[2],'coefficient':[.5,.75]}]}]}
    assert measure(model)==[[-.125,3.25]]
    tiny=2.0**-52
    model={'domain':[[1.,1.]],'components':[{'remainder':[tiny,tiny],
            'terms':[{'degrees':[0],'coefficient':[1.1,1.1]}]}]}
    lo,hi=measure(model)[0];exact=Fraction(1.1)+Fraction(tiny)
    assert Fraction(lo)<=exact<=Fraction(hi)

@pytest.mark.parametrize('kind',['width','horizon','mode','runtime'])
def test_actual_package_tamper_survives_rehashed_outer_manifest(tmp_path,kind):
    # This deliberately requires the completed package; the final test run
    # must not silently skip the goal's four direct tamper checks.
    assert (ARTIFACTS/'ADOPTION_RESULT.json').exists()
    copied=tmp_path/'evidence';shutil.copytree(ARTIFACTS,copied)
    if kind in ['width','runtime']:
        import csv
        name='widths_full_prefix.csv' if kind=='width' else 'timings_raw.csv'
        p=copied/name
        with p.open() as h:rows=list(csv.DictReader(h))
        key='strict_width' if kind=='width' else 'solve_seconds'
        selected=next(r for r in rows if r.get(key))
        selected[key]=str(float(selected[key])*1.2+0.01)
        with p.open('w') as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    elif kind=='horizon':
        p=copied/'raw_minimal/our_brusselator_full/summary.json';data=json.loads(p.read_text())
        data['accepted_horizon']-=0.02;p.write_text(json.dumps(data))
    else:
        p=copied/'raw_minimal/candidate_short/summary.json';data=json.loads(p.read_text())
        data['rows'][0]['mode']='strict' if data['rows'][0]['mode']=='parity' else 'parity'
        p.write_text(json.dumps(data))
    (copied/'SHA256SUMS').write_text(checksum(copied))
    with pytest.raises(ValueError):verify(copied)
