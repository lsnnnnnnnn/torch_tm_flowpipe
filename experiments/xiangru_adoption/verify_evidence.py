"""A bounded verifier for this decision, not a general research audit system."""
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from build_tables import read_csv, tables, write_csv
from common import measure

def require(value,message):
    if not value:raise ValueError(message)

def checksum(root):
    paths=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    return ''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(root))+'\n' for p in paths)

def verify(root,check_hashes=True):
    if check_hashes:require((root/'SHA256SUMS').read_text()==checksum(root),'outer hashes differ')
    contract=json.loads((root/'MATCHED_CONTRACTS.json').read_text())
    source=json.loads((root/'source_manifest.json').read_text())
    require(source['candidate']['actual_runtime_sha']=='1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3','candidate identity changed')
    require(source['ours_numerical_reference']=='4939fb288c941a67f55cc191f4d75f8594692f47','our reference identity changed')
    for name,order,step,queue,horizon in [('brusselator',6,.02,1000,20),('van_der_pol',4,.01,100,10)]:
        c=contract['plants'][name]
        require((c['state_dimension'],c['order'],float(c['step']['decimal']),c['sr_capacity'],float(c['requested_horizon']['decimal']))==(2,order,step,queue,horizon),'frozen contract changed')
    known=json.loads((root/'known_witness_replay.json').read_text())
    phis=[];js=[];exact=[Fraction(),Fraction()]
    for k,matrix in enumerate(known['matrices']):
        a=[[Fraction(v) for v in row] for row in matrix]
        for q in range(1,len(phis)):
            b=phis[q];phis[q]=[[sum((a[i][j]*b[j][l] for j in range(2)),Fraction()) for l in range(2)] for i in range(2)]
        phis.append(a)
        exact=[sum((phis[q][i][j]*js[q-1][j] for q in range(1,len(phis)) for j in range(2)),Fraction()) for i in range(2)]
        if k<len(known['historical_j_columns']):js.append(list(map(Fraction,known['historical_j_columns'][k])))
    require(str(exact[1])=='-4503599627370497/38685626227668133590597632','original exact witness changed')
    require({(r['device'],r['parsed_mode']) for r in known['runs']}=={(d,m) for d in ['cpu','cuda'] for m in ['parity','strict']},'witness mode/device coverage')
    for r in known['runs']:
        contained=[Fraction(lo)<=v<=Fraction(hi) for (lo,hi),v in zip(r['bounds'],exact)]
        require(contained==r['contains_exact'] and not all(contained),'known witness inclusion recomputation')
        require(r['parsed_mode']==r['requested_mode'],'witness mode mismatch')
        if r['device']=='cuda':require(r['cuda_kernel_names'],'CUDA execution unproven')
    scale=json.loads((root/'preconditioning_exact_singleton.json').read_text())
    singleton=[r for r in scale['rows'] if r.get('set_inclusion_witness')]
    require(len(singleton)==2,'singleton scaling witness missing')
    for r in singleton:
        exact=Fraction(r['original_coefficient'])+Fraction(r['original_remainder'][0])
        c=Fraction(r['scaled_coefficient']);s=Fraction(r['S']);lo,hi=map(Fraction,r['scaled_remainder'])
        require(not s*(c+lo)<=exact<=s*(c+hi),'scaling witness not independently reproduced')
        require(str(exact)==r['exact_input_fraction'],'scaling input identity')
    short=json.loads((root/'raw_minimal/candidate_short/summary.json').read_text())
    for r in short['rows']:
        require(r['mode']==r['settings']['mode'],'actual candidate mode differs from parsed Settings')
        require(r['device']==r['settings']['device'],'candidate device mismatch')
        require(r['step']<=2 and r['adoption_eligible'] is False,'defective path used beyond diagnostic scope')
    # Validate source tables against complete mathematical exports, independently
    # of the aggregate tables and their outer hashes.
    for shortname in ['brusselator','vdp']:
        d=root/'raw_minimal'/('our_'+shortname+'_full')
        summary=json.loads((d/'summary.json').read_text());bounds=read_csv(d/'bounds.csv')
        count=0;last=Fraction()
        with gzip.open(d/'models.jsonl.gz','rt') as h:
          for line,row in zip(h,bounds,strict=True):
            obj=json.loads(line);count+=1
            require(Fraction(obj['t_start_exact'])==last,'our physical time discontinuity')
            last=Fraction(obj['t_end_exact'])
            require(obj['step']==int(row['step']) and str(last)==row['t_end_exact'],'our raw object/time row mismatch')
            for kind in ['endpoint','tube']:
              for dim,(lo,hi) in zip(['x','y'],measure(obj['models'][kind])):
                require(float(row[f'common_{kind}_{dim}_lo'])==lo and float(row[f'common_{kind}_{dim}_hi'])==hi,'our raw common bound tampered')
        require(count==summary['accepted_steps'] and float(last)==summary['accepted_horizon'],'our actual horizon differs from raw objects')
        nd=root/'raw_minimal'/('native_'+shortname);ns=json.loads((nd/'summary.json').read_text())
        with gzip.open(nd/'models.jsonl.gz','rt') as h:native=[json.loads(line) for line in h]
        require(len(native)==ns['accepted_steps'],'native actual step count differs')
        native_time=sum((Fraction(r['h']) for r in native),Fraction())
        require(float(native_time)==ns['accepted_horizon'],'native actual horizon differs')
        require(all(r['state_dimension']==2 for r in native),'extra native clock state')
    expected=tables(root)
    with tempfile.TemporaryDirectory(prefix='xiangru_recompute_') as td:
        for name,rows in expected.items():
            p=Path(td)/name;write_csv(p,rows)
            require(p.read_bytes()==(root/name).read_bytes(),'raw cross-check rejected '+name)
    decision=json.loads((root/'ADOPTION_RESULT.json').read_text())
    require(decision['decision']=='DO_NOT_ADOPT_YET__CONFIRMED_CORRECTNESS_DEFECT','decision contradicts exact evidence')
    require(decision['optional_backend_added'] is False and decision['candidate_numerical_patch'] is False,'adoption or patch contradicts failed gate')
    return {'status':'VERIFIED','exact_witnesses_recomputed':2,'aggregate_tables_recomputed':len(expected),
            'scope':'Committed evidence cross-check; this does not rerun the long numerical experiments.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--write-hashes',action='store_true');a=p.parse_args()
    if a.write_hashes:(a.root/'SHA256SUMS').write_text(checksum(a.root))
    print(json.dumps(verify(a.root),indent=2))

if __name__=='__main__':main()
